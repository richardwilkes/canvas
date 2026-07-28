// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The merge image filter node: src-over stacking of all inputs.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

type mergeFilter struct {
	filterDefaults
}

// Merge builds a filter that src-over stacks the given filters, optionally cropped to cropRect. The slice is not
// retained: the caller remains free to reuse or mutate it.
func Merge(filters []filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if len(filters) == 0 {
		return Empty()
	}
	f := &mergeFilter{}
	f.base = filtercore.NewFilterBase(filters...)
	var filter filtercore.Filter = f
	if cropRect != nil {
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// OnCTMCapability reports that merging is independent of the layer matrix.
func (f *mergeFilter) OnCTMCapability() filtercore.MatrixCapability {
	return filtercore.MatrixCapabilityComplex
}

// GPUEvaluable implements filtercore.GPUEvaluable: inputs composite via image draws onto one surface.
func (f *mergeFilter) GPUEvaluable() bool { return true }

// OnFilterImage evaluates every child and src-over merges the results.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *mergeFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	builder := filtercore.NewBuilder(&ctx)
	for i := 0; i < f.base.CountInputs(); i++ {
		childOutput := f.base.ChildOutput(i, &ctx)
		builder.Add(&childOutput)
	}
	return builder.Merge()
}

// OnInputLayerBounds returns the union of all child input bounds so one source image can provide for all of them.
func (f *mergeFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	union := f.base.InputLayerBounds(0, mapping, desiredOutput, contentBounds)
	for i := 1; i < f.base.CountInputs(); i++ {
		union.Join(f.base.InputLayerBounds(i, mapping, desiredOutput, contentBounds))
	}
	return union
}

// OnOutputLayerBounds returns the union of all child outputs, unbounded if any child is.
func (f *mergeFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	var union geom.IRect
	for i := 0; i < f.base.CountInputs(); i++ {
		out := f.base.OutputLayerBounds(i, mapping, contentBounds)
		if out == nil {
			return nil
		}
		if i == 0 {
			union = *out
		} else {
			union.Join(*out)
		}
	}
	return &union
}

// OnComputeFastBounds uses the default union implementation; the factory guarantees at least one input.
func (f *mergeFilter) OnComputeFastBounds(src geom.Rect) geom.Rect {
	return filtercore.DefaultComputeFastBounds(f, src)
}
