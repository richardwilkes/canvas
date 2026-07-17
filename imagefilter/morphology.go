// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The morphology image filter nodes (dilate/erode), implemented as repeated linear/sparse morphology shader passes.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// maxLinearMorphologyRadius is the largest radius the linear accumulation pass handles before switching to
// kernel-doubling sparse passes (keep in sync with the constant in shaders' morphology kernel).
const maxLinearMorphologyRadius = 14

// maxMorphologyRadii is the layer-space radius clamp (crbug.com/1123035).
const maxMorphologyRadii = 256

type morphologyFilter struct {
	filterDefaults
	dilate bool
	radii  geom.Size // parameter space
}

func makeMorphology(dilate bool, radii geom.Size, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if radii.Width < 0 || radii.Height < 0 {
		return nil // invalid
	}
	filter := input
	if radii.Width > 0 || radii.Height > 0 {
		f := &morphologyFilter{dilate: dilate, radii: radii}
		f.base = filtercore.NewFilterBase(filter)
		filter = f
	}
	// Otherwise both radii are 0, so the kernel is always the identity function, in which case we just need to apply
	// the 'cropRect' to the 'input'.
	if cropRect != nil {
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// Dilate builds a filter that expands the input's non-transparent regions by radiusX/radiusY.
func Dilate(radiusX, radiusY float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeMorphology(true, geom.Size{Width: radiusX, Height: radiusY}, input, cropRect)
}

// Erode builds a filter that shrinks the input's non-transparent regions by radiusX/radiusY.
func Erode(radiusX, radiusY float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeMorphology(false, geom.Size{Width: radiusX, Height: radiusY}, input, cropRect)
}

// GPUEvaluable implements filtercore.GPUEvaluable: the linear/sparse morphology kernels convert to FPs
// (gpu/gl/filterkernelfp.go).
func (f *morphologyFilter) GPUEvaluable() bool { return true }

// layerRadii maps the parameter-space radii into layer space, rounding to integers and clamping to maxMorphologyRadii.
func (f *morphologyFilter) layerRadii(mapping *filtercore.Mapping) geom.ISize {
	radii := mapping.ParamToLayerSize(f.radii).ToRound()
	return geom.ISize{
		Width:  min(radii.Width, maxMorphologyRadii),
		Height: min(radii.Height, maxMorphologyRadii),
	}
}

// requiredInput outsets bounds by the layer-space radii: the required input is the kernel outset, regardless of morph
// type.
func (f *morphologyFilter) requiredInput(mapping *filtercore.Mapping, bounds geom.IRect) geom.IRect {
	radii := f.layerRadii(mapping)
	return bounds.Inset(-radii.Width, -radii.Height)
}

// kernelOutputBounds computes the output extent from bounds: dilate expands the output by the radii (transparent pixels
// get overridden by the max), erode contracts it (edge pixels get overridden by the min to transparent black).
func (f *morphologyFilter) kernelOutputBounds(mapping *filtercore.Mapping, bounds geom.IRect) geom.IRect {
	radii := f.layerRadii(mapping)
	if f.dilate {
		return bounds.Inset(-radii.Width, -radii.Height)
	}
	return bounds.Inset(radii.Width, radii.Height)
}

// morphologyPass runs repeated shader passes along one axis, starting with a linear pass of up to
// maxLinearMorphologyRadius and doubling with sparse passes after that.
func morphologyPass(ctx *filtercore.Context, input *filtercore.FilterResult, dilate, dirX bool, radius int32) filtercore.FilterResult {
	axisDelta := func(step int32) geom.ISize {
		if dirX {
			return geom.ISize{Width: step}
		}
		return geom.ISize{Height: step}
	}
	axisOffset := func(step int32) geom.Point {
		if dirX {
			return geom.Point{X: float32(step)}
		}
		return geom.Point{Y: float32(step)}
	}

	// The first iteration will sample a full kernel outset from the final output.
	sampleBounds := ctx.DesiredOutput()
	d := axisDelta(radius)
	sampleBounds = sampleBounds.Inset(-d.Width, -d.Height)

	childOutput := *input
	appliedRadius := int32(0)
	for radius > appliedRadius {
		if !childOutput.IsValid() {
			return filtercore.FilterResult{} // eroded or dilated transparent black is still transparent
		}

		// The first iteration uses up to the max linear radius with a linear accumulation pass; after that the radius
		// doubles each step until the target radius is reached.
		var stepRadius int32
		if appliedRadius == 0 {
			stepRadius = min(maxLinearMorphologyRadius, radius)
		} else {
			stepRadius = min(radius-appliedRadius, appliedRadius)
		}

		stepCtx := *ctx
		if appliedRadius+stepRadius < radius {
			// Intermediate steps need to output what will be sampled on the next iteration.
			d = axisDelta(stepRadius)
			stepCtx = ctx.WithNewDesiredOutput(sampleBounds.Inset(d.Width, d.Height))
		} // else the last iteration should output what was originally requested

		builder := filtercore.NewBuilder(&stepCtx)
		builder.AddSampled(&childOutput, &sampleBounds, filtercore.ShaderFlagSampledRepeatedly,
			filtercore.DefaultSampling)
		first := appliedRadius == 0
		childOutput = builder.Eval(func(inputs []shaders.Shader) shaders.Shader {
			if first {
				return shaders.NewLinearMorphology(inputs[0], axisOffset(1), dilate, stepRadius)
			}
			return shaders.NewSparseMorphology(inputs[0], axisOffset(stepRadius), dilate)
		}, nil)

		sampleBounds = stepCtx.DesiredOutput()
		appliedRadius += stepRadius
	}
	return childOutput
}

// OnFilterImage evaluates the child over the kernel-outset input, then applies separate X and Y morphology passes.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *morphologyFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	requiredInput := f.requiredInput(ctx.Mapping(), ctx.DesiredOutput())
	inputCtx := ctx.WithNewDesiredOutput(requiredInput)
	childOutput := f.base.ChildOutput(0, &inputCtx)

	// If childOutput completely fulfilled requiredInput, maxOutput will match the context's desired output; a smaller
	// output image restricts what the morphology can produce.
	maxOutput := f.kernelOutputBounds(ctx.Mapping(), childOutput.LayerBounds())
	if !maxOutput.Intersect(ctx.DesiredOutput()) {
		return filtercore.FilterResult{}
	}

	// The X pass has to preserve the extra rows to later be consumed by the Y pass.
	radii := f.layerRadii(ctx.Mapping())
	maxOutputX := maxOutput.Inset(0, -radii.Height)
	xCtx := ctx.WithNewDesiredOutput(maxOutputX)
	childOutput = morphologyPass(&xCtx, &childOutput, f.dilate, true, radii.Width)
	yCtx := ctx.WithNewDesiredOutput(maxOutput)
	return morphologyPass(&yCtx, &childOutput, f.dilate, false, radii.Height)
}

// OnInputLayerBounds requires the child's output outset by the layer-space radii.
func (f *morphologyFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	requiredInput := f.requiredInput(mapping, desiredOutput)
	return f.base.InputLayerBounds(0, mapping, requiredInput, contentBounds)
}

// OnOutputLayerBounds applies kernelOutputBounds to the child's output.
func (f *morphologyFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	childOutput := f.base.OutputLayerBounds(0, mapping, contentBounds)
	if childOutput == nil {
		return nil
	}
	out := f.kernelOutputBounds(mapping, *childOutput)
	return &out
}

// OnComputeFastBounds expands or shrinks the input's fast bounds by the radii (see kernelOutputBounds for the
// dilate/erode rationale).
func (f *morphologyFilter) OnComputeFastBounds(src geom.Rect) geom.Rect {
	bounds := src
	if in := f.base.Input(0); in != nil {
		bounds = in.OnComputeFastBounds(src)
	}
	if f.dilate {
		return bounds.Outset(f.radii.Width, f.radii.Height)
	}
	return bounds.Inset(f.radii.Width, f.radii.Height)
}
