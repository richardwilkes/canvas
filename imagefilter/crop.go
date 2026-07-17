// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The crop image filter node: crop + tile-mode application, the building block every factory's crop rect lowers to,
// plus the Empty() and Tile() compositions.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

type cropFilter struct {
	filterDefaults
	cropRect geom.Rect // parameter space
	tileMode shaders.TileMode
}

// Crop crops the input to rect (applying tileMode outside it). Returns nil for invalid rects.
func Crop(rect geom.Rect, tileMode shaders.TileMode, input filtercore.Filter) filtercore.Filter {
	if !isValidRect(rect) {
		return nil
	}
	f := &cropFilter{cropRect: rect, tileMode: tileMode}
	f.base = filtercore.NewFilterBase(input)
	return f
}

// Empty returns a filter producing transparent black, via a crop to an empty rect (whose implementation handles empty
// rects gracefully).
func Empty() filtercore.Filter {
	return Crop(geom.Rect{}, shaders.TileDecal, nil)
}

// Tile builds a filter that crops the input to src with repeat tiling, wrapped in a decal crop to dst.
func Tile(src, dst geom.Rect, input filtercore.Filter) filtercore.Filter {
	filter := Crop(src, shaders.TileRepeat, input)
	return Crop(dst, shaders.TileDecal, filter)
}

// OnAffectsTransparentBlack: non-decal tiling fills unbounded output.
func (f *cropFilter) OnAffectsTransparentBlack() bool { return f.tileMode != shaders.TileDecal }

// IgnoreInputsAffectsTransparentBlack: a crop bounds whatever its input floods.
func (f *cropFilter) IgnoreInputsAffectsTransparentBlack() bool { return true }

// cropRectLayer maps the parameter-space crop rect into layer space: round out for decal (fractional coverage is
// accurate there), round in for the other modes so clamping can't magnify rasterization transparency.
func (f *cropFilter) cropRectLayer(mapping *filtercore.Mapping) geom.IRect {
	crop := mapping.ParamToLayerRect(f.cropRect)
	if f.tileMode == shaders.TileDecal {
		return filtercore.RoundOut(crop)
	}
	return filtercore.RoundIn(crop)
}

// requiredInput reports the subset of the child's output that the crop and tile mode actually need to satisfy
// outputBounds.
func (f *cropFilter) requiredInput(mapping *filtercore.Mapping, outputBounds geom.IRect) geom.IRect {
	return filtercore.RelevantSubset(f.cropRectLayer(mapping), outputBounds, f.tileMode)
}

// GPUEvaluable implements filtercore.GPUEvaluable: crop/tile lower to subset + tile-mode bookkeeping over the
// image-shader lanes.
func (f *cropFilter) GPUEvaluable() bool { return true }

// OnFilterImage evaluates the child over the minimal required input, then applies the crop rect and tile mode to the
// result.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *cropFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	cropInput := f.requiredInput(ctx.Mapping(), ctx.DesiredOutput())
	inputCtx := ctx.WithNewDesiredOutput(cropInput)
	childOutput := f.base.ChildOutput(0, &inputCtx)
	// 'cropInput' is the optimal input to satisfy the crop rect, but the actual crop rect must be passed for the tile
	// mode to apply correctly.
	return childOutput.ApplyCrop(&ctx, f.cropRectLayer(ctx.Mapping()), f.tileMode)
}

// OnInputLayerBounds requires only the subset of the child's output the crop and tile mode need.
func (f *cropFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	requiredInput := f.requiredInput(mapping, desiredOutput)
	return f.base.InputLayerBounds(0, mapping, requiredInput, contentBounds)
}

// OnOutputLayerBounds intersects the child's output with the crop rect; non-decal tiling makes the visual output
// unbounded whenever any non-transparent content survives the crop.
func (f *cropFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	childOutput := f.base.OutputLayerBounds(0, mapping, contentBounds)
	crop := f.cropRectLayer(mapping)
	if childOutput != nil && !crop.Intersect(*childOutput) {
		// The content within the crop rect is fully transparent regardless of tiling.
		empty := geom.IRect{}
		return &empty
	}
	// Non-transparent content exists within the crop; non-decal tiling makes the visual output unbounded even if the
	// underlying data is smaller.
	if f.tileMode == shaders.TileDecal {
		return &crop
	}
	return nil
}

// OnComputeFastBounds intersects the input's fast bounds with the crop rect, returning an unbounded rect if tiling is
// not decal.
func (f *cropFilter) OnComputeFastBounds(bounds geom.Rect) geom.Rect {
	inputBounds := bounds
	if in := f.base.Input(0); in != nil {
		inputBounds = in.OnComputeFastBounds(bounds)
	}
	if !inputBounds.Intersect(f.cropRect) {
		return geom.Rect{}
	}
	if f.tileMode == shaders.TileDecal {
		return inputBounds
	}
	return filtercore.MakeILarge().ToRect()
}
