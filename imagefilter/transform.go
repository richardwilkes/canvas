// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The matrix-transform image filter node, and the Offset factory that lowers to it with nearest-neighbor sampling.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

type matrixTransformFilter struct {
	filterDefaults
	transform geom.Matrix // parameter space
	sampling  shaders.SamplingOptions
}

// MatrixTransform builds a filter that applies transform to the input with the given sampling. Returns nil for
// non-invertible transforms.
func MatrixTransform(transform *geom.Matrix, sampling shaders.SamplingOptions, input filtercore.Filter) filtercore.Filter {
	if _, ok := transform.Invert(); !ok {
		return nil
	}
	f := &matrixTransformFilter{transform: *transform, sampling: sampling}
	f.base = filtercore.NewFilterBase(input)
	return f
}

// Offset builds a filter that translates the input by (dx, dy) using nearest-neighbor sampling (matching the legacy
// implementation's rounding to layer-space pixels), with the crop rect applying only to the offset's output.
func Offset(dx, dy float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	var t geom.Matrix
	t.SetTranslate(dx, dy)
	offset := MatrixTransform(&t, shaders.SamplingOptions{Filter: shaders.FilterNearest}, input)
	if cropRect != nil {
		offset = Crop(*cropRect, shaders.TileDecal, offset)
	}
	return offset
}

// OnCTMCapability: the transform handles any layer matrix.
func (f *matrixTransformFilter) OnCTMCapability() filtercore.MatrixCapability {
	return filtercore.MatrixCapabilityComplex
}

// requiredInput inverse-maps desiredOutput through the layer-space transform, outset by 1px when sampling needs a
// filter kernel wider than nearest-neighbor.
func (f *matrixTransformFilter) requiredInput(mapping *filtercore.Mapping, desiredOutput geom.IRect) geom.IRect {
	layerTransform := mapping.ParamToLayerMatrix(&f.transform)
	requiredInput, ok := filtercore.InverseMapIRect(&layerTransform, desiredOutput)
	if !ok {
		return geom.IRect{}
	}
	// Any filtering beyond nearest neighbor needs an extra buffer of pixels for the kernel.
	if f.sampling != (shaders.SamplingOptions{}) {
		requiredInput = requiredInput.Inset(-1, -1)
	}
	return requiredInput
}

// GPUEvaluable implements filtercore.GPUEvaluable: offset/matrix-transform lower to a transform draw over the
// image-shader lanes.
func (f *matrixTransformFilter) GPUEvaluable() bool { return true }

// OnFilterImage evaluates the child over the inverse-mapped required input, then applies the forward transform to the
// result.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *matrixTransformFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	requiredInput := f.requiredInput(ctx.Mapping(), ctx.DesiredOutput())
	inputCtx := ctx.WithNewDesiredOutput(requiredInput)
	childOutput := f.base.ChildOutput(0, &inputCtx)
	transform := ctx.Mapping().ParamToLayerMatrix(&f.transform)
	return childOutput.ApplyTransform(&ctx, &transform, f.sampling)
}

// OnInputLayerBounds requires the inverse-mapped desired output from the child.
func (f *matrixTransformFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	requiredInput := f.requiredInput(mapping, desiredOutput)
	return f.base.InputLayerBounds(0, mapping, requiredInput, contentBounds)
}

// OnOutputLayerBounds maps the child's output bounds through the layer-space transform.
func (f *matrixTransformFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	childOutput := f.base.OutputLayerBounds(0, mapping, contentBounds)
	if childOutput == nil {
		return nil
	}
	layerTransform := mapping.ParamToLayerMatrix(&f.transform)
	out := filtercore.MapIRect(&layerTransform, *childOutput)
	return &out
}

// OnComputeFastBounds maps the input's fast bounds through the parameter-space transform.
func (f *matrixTransformFilter) OnComputeFastBounds(src geom.Rect) geom.Rect {
	bounds := src
	if in := f.base.Input(0); in != nil {
		bounds = in.OnComputeFastBounds(src)
	}
	mapped, _ := f.transform.MapRect(bounds)
	return mapped
}
