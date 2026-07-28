// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Builder renders one or more FilterResults (or shaders derived from them) into a new FilterResult — Merge() for
// src-over stacking, Blur() over the backend's blur engine, and Eval() for the shader-combining filters (arithmetic,
// displacement, lighting, morphology, convolution, magnifier).

package filtercore

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// builderInput is one FilterResult added to a Builder, with the sampling it should use when converted to a shader.
type builderInput struct {
	sampleBounds *geom.IRect
	image        FilterResult
	sampling     shaders.SamplingOptions
	flags        ShaderFlags
}

// Builder accumulates FilterResult inputs and combines them into a single output FilterResult.
type Builder struct {
	inputs []builderInput
	ctx    Context
}

// NewBuilder returns a Builder that evaluates against ctx.
func NewBuilder(ctx *Context) *Builder {
	return &Builder{ctx: *ctx}
}

// Add adds input with default sampling (for Merge/Blur inputs).
func (b *Builder) Add(input *FilterResult) *Builder {
	b.inputs = append(b.inputs, builderInput{image: *input, sampling: DefaultSampling})
	return b
}

// AddSampled adds input with explicit sample bounds, flags, and sampling (for Eval inputs).
func (b *Builder) AddSampled(input *FilterResult, sampleBounds *geom.IRect, flags ShaderFlags, sampling shaders.SamplingOptions) *Builder {
	b.inputs = append(b.inputs, builderInput{
		image:        *input,
		sampleBounds: copyOptRect(sampleBounds),
		flags:        flags,
		sampling:     sampling,
	})
	return b
}

// outputBounds returns the context's desired output, optionally intersected with explicit bounds.
func (b *Builder) outputBounds(explicitOutput *geom.IRect) geom.IRect {
	output := b.ctx.DesiredOutput()
	if explicitOutput != nil && !output.Intersect(*explicitOutput) {
		return geom.IRect{}
	}
	return output
}

// createInputShaders converts each added input to a shader, sampled over outputBounds (or its own sample bounds when
// set). The inputs are always evaluated in layer space; upstream's parameter-space lane exists only for the runtime and
// picture filters, neither of which is ported.
func (b *Builder) createInputShaders(outputBounds geom.IRect) []shaders.Shader {
	inputShaders := make([]shaders.Shader, 0, len(b.inputs))
	for i := range b.inputs {
		input := &b.inputs[i]
		// Assume the input shader is evaluated once per output pixel unless specified otherwise.
		sampleBounds := outputBounds
		if input.sampleBounds != nil {
			sampleBounds = *input.sampleBounds
		}
		inputShaders = append(inputShaders, input.image.asShader(&b.ctx, input.sampling, input.flags, sampleBounds))
	}
	return inputShaders
}

// drawShader fills outputBounds with shader (src blend, no dst content), returning the result as a FilterResult.
func (b *Builder) drawShader(shader shaders.Shader, outputBounds geom.IRect) FilterResult {
	if shader == nil {
		return FilterResult{}
	}
	surface := newAutoSurface(&b.ctx, outputBounds, boundaryTransparent, false)
	if surface.valid {
		surface.drawPaint(&PaintParams{Shader: shader, Blend: raster.BlendSrc})
	}
	return surface.snap()
}

// Eval converts the added inputs to shaders, lets shaderFn combine them (a nil element means that input was fully
// transparent; a nil return means the output is fully transparent), and fills the output bounds with the result.
func (b *Builder) Eval(shaderFn func(inputs []shaders.Shader) shaders.Shader, explicitOutput *geom.IRect) FilterResult {
	outputBounds := b.outputBounds(explicitOutput)
	if outputBounds.IsEmpty() {
		return FilterResult{}
	}
	inputShaders := b.createInputShaders(outputBounds)
	return b.drawShader(shaderFn(inputShaders), outputBounds)
}

// Merge src-over stacks all added inputs into a single output.
func (b *Builder) Merge() FilterResult {
	if len(b.inputs) == 0 {
		// Should have been routed to imagefilter.Empty() before this.
		panic("Builder.Merge with no inputs")
	}
	if len(b.inputs) == 1 {
		return b.inputs[0].image
	}

	mergedBounds := b.inputs[0].image.LayerBounds()
	for i := 1; i < len(b.inputs); i++ {
		mergedBounds.Join(b.inputs[i].image.LayerBounds())
	}
	outputBounds := b.outputBounds(&mergedBounds)

	surface := newAutoSurface(&b.ctx, outputBounds, boundaryTransparent, false)
	if surface.valid {
		for i := range b.inputs {
			b.inputs[i].image.draw(&b.ctx, surface.device, true, nil)
		}
	}
	return surface.snap()
}

// Blur gaussian-blurs the single input via the backend's blur engine, deriving sample and output bounds from the sigma.
func (b *Builder) Blur(sigma geom.Size) FilterResult {
	if len(b.inputs) != 1 {
		panic("Builder.Blur requires exactly one input")
	}

	blurEngine := b.ctx.Backend().BlurEngine()
	algorithm := blurEngine.FindAlgorithm()
	if algorithm == nil {
		return FilterResult{}
	}

	radii := sizeCeil(geom.Size{Width: 3 * sigma.Width, Height: 3 * sigma.Height})
	maxOutput := b.inputs[0].image.LayerBounds().Inset(-radii.Width, -radii.Height)
	outputBounds := b.outputBounds(&maxOutput)
	if outputBounds.IsEmpty() {
		return FilterResult{}
	}

	// The source pixels read from the input image; computable internally because the blur's access pattern is well
	// defined.
	sampleBounds := outputBounds.Inset(-radii.Width, -radii.Height)

	sx, sy := float32(1), float32(1)
	if sigma.Width > algorithm.MaxSigma() {
		sx = algorithm.MaxSigma() / sigma.Width
	}
	if sigma.Height > algorithm.MaxSigma() {
		sy = algorithm.MaxSigma() / sigma.Height
	}
	// For identity scale factors this rescale() is a no-op when possible; otherwise it also resolves color
	// filters/transforms like resolve() but can defer the tile mode. Overscaling is always allowed since the low-res
	// sigma adjusts for any extra scale factor.
	rescaleCtx := b.ctx.WithNewDesiredOutput(sampleBounds)
	lowResImage := b.inputs[0].image.rescale(
		&rescaleCtx,
		geom.Size{Width: sx, Height: sy},
		algorithm.SupportsOnlyDecalTiling(),
		true,
	)
	if lowResImage.image == nil {
		return FilterResult{}
	}

	// Map sigma into the low-res image's pixel space; rescale() produces a scale+translate transform so the inverse
	// scale factors come straight off the matrix.
	invScaleX := ieeeFloatDivide(1, lowResImage.transform.Get(geom.MScaleX))
	invScaleY := ieeeFloatDivide(1, lowResImage.transform.Get(geom.MScaleY))
	lowResSigma := geom.Size{
		Width:  min(sigma.Width*invScaleX, algorithm.MaxSigma()),
		Height: min(sigma.Height*invScaleY, algorithm.MaxSigma()),
	}
	lowResMaxOutput := geom.IRectWH(lowResImage.image.Width(), lowResImage.image.Height())

	var srcRelativeOutput geom.IRect
	if lowResImage.tileMode == shaders.TileRepeat || lowResImage.tileMode == shaders.TileMirror {
		// Periodic tiling was deferred when down-sampling and can further defer to after the blur; the low-res output
		// is 1-to-1 with the low-res image.
		srcRelativeOutput = lowResMaxOutput
	} else {
		// For decal and clamp tiling the blurred image stops being interesting outside the radii outset, so redo the
		// max output analysis with outputBounds mapped into pixel space.
		var ok bool
		srcRelativeOutput, ok = inverseMapIRect(&lowResImage.transform, outputBounds)
		if !ok {
			panic("rescale transform must be invertible")
		}
		lowResRadii := sizeCeil(geom.Size{Width: 3 * lowResSigma.Width, Height: 3 * lowResSigma.Height})
		lowResMaxOutput = lowResMaxOutput.Inset(-lowResRadii.Width, -lowResRadii.Height)
		srcRelativeOutput = RelevantSubset(lowResMaxOutput, srcRelativeOutput, lowResImage.tileMode)
		// Guard against pathological float/int mixing changing emptiness (TODO 40042624 upstream).
		if srcRelativeOutput.IsEmpty() {
			return FilterResult{}
		}
		// Include 1px of blur output so it can be sampled during the upscale (crbug.com/1500021).
		srcRelativeOutput = srcRelativeOutput.Inset(-1, -1)
	}

	lowResBlur := lowResImage.image
	blurOutputBounds := srcRelativeOutput
	tileMode := lowResImage.tileMode
	if !algorithm.SupportsOnlyDecalTiling() &&
		lowResImage.canClampToTransparentBoundary(boundsSimple) {
		// Manually account for the known pixel padding, since the blur engine isn't aware of it.
		lowResBlur = lowResBlur.MakePixelOutset()
		blurOutputBounds = blurOutputBounds.Offset(1, 1)
		tileMode = shaders.TileClamp
	}

	lowResBlur = algorithm.Blur(lowResSigma, lowResBlur,
		geom.IRectSize(lowResBlur.Dimensions()), tileMode, blurOutputBounds)
	if lowResBlur == nil {
		return FilterResult{}
	}

	result := NewFilterResult(lowResBlur, geom.IPoint{X: srcRelativeOutput.Left, Y: srcRelativeOutput.Top})
	if lowResImage.tileMode == shaders.TileClamp || lowResImage.tileMode == shaders.TileDecal {
		// Undo the outset padding added to srcRelativeOutput before invoking the blur.
		result = result.insetByPixel()
	}

	result.transform.PostConcat(&lowResImage.transform)
	if lowResImage.tileMode == shaders.TileDecal {
		// Recalculate the output bounds from the blur output; rounding may make the final image slightly larger than
		// the original, and this avoids unnecessary layer-bounds cropping.
		mapped := mapIRect(&result.transform, geom.IRectSize(result.image.Dimensions()))
		outputBounds = b.outputBounds(&mapped)
	}
	result.layerBounds = outputBounds
	result.tileMode = lowResImage.tileMode
	return result
}
