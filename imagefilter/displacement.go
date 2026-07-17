// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The displacement-map image filter node: displaces a color image's pixels using a per-pixel vector read from another
// (displacement) image's channels.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// ColorChannel selects which channel of a pixel to read.
type ColorChannel int32

// ColorChannel values.
const (
	ChannelR ColorChannel = iota
	ChannelG
	ChannelB
	ChannelA
)

// displacementSampling is nearest-neighbor, to match historical behavior (skbug.com/40045448).
var displacementSampling = shaders.SamplingOptions{Filter: shaders.FilterNearest}

type displacementFilter struct {
	filterDefaults
	xChannel ColorChannel
	yChannel ColorChannel
	scale    float32
}

// Input image filter indices.
const (
	displacementInput = 0
	displacementColor = 1
)

// DisplacementMap builds a filter that displaces color's pixels using vectors read from displacement's
// xChannel/yChannel, scaled by scale, optionally cropped to cropRect.
func DisplacementMap(xChannel, yChannel ColorChannel, scale float32, displacement, color filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if xChannel < ChannelR || xChannel > ChannelA || yChannel < ChannelR || yChannel > ChannelA {
		return nil
	}
	if !geom.IsFinite(scale) {
		return nil
	}
	f := &displacementFilter{xChannel: xChannel, yChannel: yChannel, scale: scale}
	f.base = filtercore.NewFilterBase(displacement, color)
	var filter filtercore.Filter = f
	if cropRect != nil {
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// GPUEvaluable implements filtercore.GPUEvaluable: the displacement kernel converts to an FP
// (gpu/gl/filterkernelfp.go).
func (f *displacementFilter) GPUEvaluable() bool { return true }

// outsetByMaxDisplacement outsets bounds by the maximum possible pixel displacement; treating scale as a size (rather
// than a vector) accounts for its absolute magnitude when transforming from parameter to layer space.
func (f *displacementFilter) outsetByMaxDisplacement(mapping *filtercore.Mapping, bounds geom.IRect) geom.IRect {
	maxDisplacement := filtercore.SizeCeil(mapping.ParamToLayerSize(
		geom.Size{Width: 0.5 * f.scale, Height: 0.5 * f.scale},
	))
	return bounds.Inset(-maxDisplacement.Width, -maxDisplacement.Height)
}

// OnFilterImage evaluates the color and displacement children and combines them into the displaced output, falling back
// to a plain constant-offset transform when the displacement map is missing.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *displacementFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	requiredColorInput := f.outsetByMaxDisplacement(ctx.Mapping(), ctx.DesiredOutput())
	colorCtx := ctx.WithNewDesiredOutput(requiredColorInput)
	colorOutput := f.base.ChildOutput(displacementColor, &colorCtx)
	if !colorOutput.IsValid() {
		return filtercore.FilterResult{} // no non-transparent-black colors to displace
	}

	// When the color image filter is unrestricted, its output will be maxDisplacement larger than this filter's desired
	// output; if cropped, the final output can be restricted, accounting for how the clipped colorOutput might still be
	// displaced.
	outputBounds := f.outsetByMaxDisplacement(ctx.Mapping(), colorOutput.LayerBounds())
	if !outputBounds.Intersect(ctx.DesiredOutput()) {
		// None of the non-transparent-black colors can be displaced into the desired bounds.
		return filtercore.FilterResult{}
	}

	displacementCtx := ctx.WithNewDesiredOutput(outputBounds)
	displacementOutput := f.base.ChildOutput(displacementInput, &displacementCtx)

	// The scale is a vector (not a size) so negations are preserved on the final displacement.
	scale := ctx.Mapping().ParamToLayerVector(geom.Point{X: f.scale, Y: f.scale})
	if !displacementOutput.IsValid() {
		// A null displacement map means transparent black, but (0,0,0,0) becomes the vector (-scale/2, -scale/2)
		// applied to the color image: a simple transform.
		var constantDisplacement geom.Matrix
		constantDisplacement.SetTranslate(-0.5*scale.X, -0.5*scale.Y)
		return colorOutput.ApplyTransform(&ctx, &constantDisplacement, displacementSampling)
	}

	// Per-pixel displacement affecting the color image: evaluate each pixel within outputBounds.
	builder := filtercore.NewBuilder(&ctx)
	builder.AddSampled(&displacementOutput, &outputBounds, filtercore.ShaderFlagNone, filtercore.DefaultSampling)
	builder.AddSampled(&colorOutput, &requiredColorInput, filtercore.ShaderFlagNonTrivialSampling,
		displacementSampling)
	xSel, ySel := int32(f.xChannel), int32(f.yChannel)
	return builder.Eval(func(inputs []shaders.Shader) shaders.Shader {
		// A transparent color output means nothing to displace; a valid displacement image that failed to produce a
		// shader is treated as transparent black.
		if inputs[displacementColor] == nil {
			return nil
		}
		return shaders.NewDisplacement(inputs[displacementInput], inputs[displacementColor],
			scale, xSel, ySel)
	}, &outputBounds)
}

// OnInputLayerBounds requires the color input outset by the maximum displacement (since pixels that far away can move
// into the desired output), joined with the displacement input's own requirement.
func (f *displacementFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	// Pixels up to the maximum displacement away from desiredOutput can be moved into those bounds, so require that
	// outset buffer from the color map.
	requiredInput := f.outsetByMaxDisplacement(mapping, desiredOutput)
	requiredInput = f.base.InputLayerBounds(displacementColor, mapping, requiredInput, contentBounds)
	// Accumulate the required input for the displacement filter to cover the original desired out.
	requiredInput.Join(f.base.InputLayerBounds(displacementInput, mapping, desiredOutput,
		contentBounds))
	return requiredInput
}

// OnOutputLayerBounds outsets the color child's output bounds by the maximum possible displacement.
func (f *displacementFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	colorOutput := f.base.OutputLayerBounds(displacementColor, mapping, contentBounds)
	if colorOutput == nil {
		return nil
	}
	out := f.outsetByMaxDisplacement(mapping, *colorOutput)
	return &out
}

// OnComputeFastBounds outsets the color input's fast bounds by half the (absolute) scale.
func (f *displacementFilter) OnComputeFastBounds(src geom.Rect) geom.Rect {
	colorBounds := src
	if in := f.base.Input(displacementColor); in != nil {
		colorBounds = in.OnComputeFastBounds(src)
	}
	maxDisplacement := 0.5 * absf(f.scale)
	return colorBounds.Outset(maxDisplacement, maxDisplacement)
}
