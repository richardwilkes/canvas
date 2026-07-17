// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The color-filter image filter node: applies a shaders.ColorFilter to its input image, collapsing adjacent
// color-filter nodes into a single composed filter.

package imagefilter

import (
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

type colorFilterFilter struct {
	cf shaders.ColorFilter
	filterDefaults
}

// ColorFilter applies cf to the input, collapsing adjacent color-filter nodes into a single composed filter.
func ColorFilter(cf shaders.ColorFilter, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if cf != nil && input != nil {
		if inputCF, ok := input.OnIsColorFilterNode(); ok {
			// Collapse the hierarchy by combining the two color filters into one.
			cf = colorfilter.NewCompose(cf, inputCF)
			input = input.Base().Input(0)
		}
	}

	filter := input
	if cf != nil {
		f := &colorFilterFilter{cf: cf}
		f.base = filtercore.NewFilterBase(filter)
		filter = f
	}
	if cropRect != nil {
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// OnCTMCapability: color filtering is independent of the layer matrix.
func (f *colorFilterFilter) OnCTMCapability() filtercore.MatrixCapability {
	return filtercore.MatrixCapabilityComplex
}

// OnAffectsTransparentBlack reports whether cf maps transparent black to a non-transparent color.
func (f *colorFilterFilter) OnAffectsTransparentBlack() bool {
	return filtercore.ColorFilterAffectsTransparentBlack(f.cf)
}

// OnIsColorFilterNode reports that this filter is a pure color-filter node, returning cf.
func (f *colorFilterFilter) OnIsColorFilterNode() (shaders.ColorFilter, bool) {
	return f.cf, true
}

// GPUEvaluable implements filtercore.GPUEvaluable: FilterResult.ApplyColorFilter lowers to ColorFilterShader, whose FP
// conversion covers every publicly reachable color filter.
func (f *colorFilterFilter) GPUEvaluable() bool { return true }

// OnFilterImage applies cf to the child's output.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *colorFilterFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	childOutput := f.base.ChildOutput(0, &ctx)
	return childOutput.ApplyColorFilter(&ctx, f.cf)
}

// OnInputLayerBounds reports the child's required input bounds unchanged: a color filter does not alter which pixels
// are needed.
func (f *colorFilterFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	return f.base.InputLayerBounds(0, mapping, desiredOutput, contentBounds)
}

// OnOutputLayerBounds reports the child's output bounds, unbounded if cf affects transparent black: only this node's
// transparency effect matters here, children account for their own.
func (f *colorFilterFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	if filtercore.ColorFilterAffectsTransparentBlack(f.cf) {
		return nil
	}
	return f.base.OutputLayerBounds(0, mapping, contentBounds)
}

// OnComputeFastBounds returns an unbounded rect if cf affects transparent black, otherwise defers to the input's fast
// bounds (or bounds itself, for a leaf).
func (f *colorFilterFilter) OnComputeFastBounds(bounds geom.Rect) geom.Rect {
	if filtercore.ColorFilterAffectsTransparentBlack(f.cf) {
		return filtercore.MakeILarge().ToRect()
	}
	if in := f.base.Input(0); in != nil {
		return in.OnComputeFastBounds(bounds)
	}
	return bounds
}
