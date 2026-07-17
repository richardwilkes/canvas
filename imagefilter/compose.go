// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The compose image filter node: outer(inner(source)), the one place where the context's source image changes mid-DAG.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
)

const (
	composeOuter = 0
	composeInner = 1
)

type composeFilter struct {
	filterDefaults
}

// Compose builds outer(inner(source)); a nil outer or inner returns the other unchanged.
func Compose(outer, inner filtercore.Filter) filtercore.Filter {
	if outer == nil {
		return inner
	}
	if inner == nil {
		return outer
	}
	f := &composeFilter{}
	// Compose only uses the source if the inner filter does; any outer reference to source rebinds to the inner's
	// result.
	f.base = filtercore.NewFilterBaseUsesSrc(inner.Base().UsesSource(), outer, inner)
	return f
}

// OnCTMCapability reports that composition itself is independent of the layer matrix (the wrapped filters carry their
// own capability).
func (f *composeFilter) OnCTMCapability() filtercore.MatrixCapability {
	return filtercore.MatrixCapabilityComplex
}

// GPUEvaluable implements filtercore.GPUEvaluable: pure plumbing — the node itself draws nothing.
func (f *composeFilter) GPUEvaluable() bool { return true }

// OnFilterImage evaluates the inner filter first, then re-evaluates the outer filter with the inner's result rebound as
// the context's source image.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *composeFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	// The expected output of the inner filter, given the source image's layer bounds as content.
	srcBounds := ctx.Source().LayerBounds()
	innerOutputBounds := f.base.OutputLayerBounds(composeInner, ctx.Mapping(), &srcBounds)
	// The required input for the outer filter to cover the desired output.
	outerRequiredInput := f.base.InputLayerBounds(composeOuter, ctx.Mapping(),
		ctx.DesiredOutput(), innerOutputBounds)

	// Evaluate the inner filter and pass that to the outer filter as its source.
	innerCtx := ctx.WithNewDesiredOutput(outerRequiredInput)
	innerResult := f.base.ChildOutput(composeInner, &innerCtx)
	outerCtx := ctx.WithNewSource(&innerResult)
	return f.base.ChildOutput(composeOuter, &outerCtx)
}

// OnInputLayerBounds threads the desired output through the outer filter (using the inner's mapped content bounds) to
// get the inner filter's desired output, then asks the inner filter for its required input.
func (f *composeFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	// The inner filter's output bounds become the outer filter's content bounds.
	var outerContentBounds *geom.IRect
	if contentBounds != nil {
		outerContentBounds = f.base.OutputLayerBounds(composeInner, mapping, contentBounds)
	} // else leave the outer's content bounds unbounded

	innerDesiredOutput := f.base.InputLayerBounds(composeOuter, mapping, desiredOutput,
		outerContentBounds)
	return f.base.InputLayerBounds(composeInner, mapping, innerDesiredOutput, contentBounds)
}

// OnOutputLayerBounds computes the inner filter's output bounds and feeds them as content bounds to the outer filter.
func (f *composeFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	innerBounds := f.base.OutputLayerBounds(composeInner, mapping, contentBounds)
	// Even if innerBounds is unbounded, the outer filter may restrict it (e.g. it contains a crop).
	return f.base.OutputLayerBounds(composeOuter, mapping, innerBounds)
}

// OnComputeFastBounds runs the inner filter's fast bounds through the outer filter's fast bounds.
func (f *composeFilter) OnComputeFastBounds(src geom.Rect) geom.Rect {
	inner := f.base.Input(composeInner).OnComputeFastBounds(src)
	return f.base.Input(composeOuter).OnComputeFastBounds(inner)
}
