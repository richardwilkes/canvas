// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The image filter contract: the Filter interface every image filter implements (the OnX methods), the FilterBase
// struct carrying the input DAG edges, the recursion helpers, and the public aggregate entry points (FilterImage with
// caching, GetInputBounds/GetOutputBounds, AffectsTransparentBlack, CTMCapability, AsAColorFilter, computeFastBounds).
// Concrete filters live in the imagefilter package; the canvas layer machinery consumes this interface.

package filtercore

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// Filter is the interface every image filter node implements. Bounds passed as *geom.IRect model optional values: nil
// contentBounds means unbounded/unknown content, and a nil return from OnOutputLayerBounds means the filter's output is
// unbounded. Implementations must not retain or mutate rects through the pointers.
type Filter interface {
	// Base returns the embedded FilterBase (inputs + usesSource + uniqueID).
	Base() *FilterBase

	// OnFilterImage evaluates the filter, producing its output FilterResult for the given context. ctx is deliberately
	// passed by value: a pointer handed through this interface call would defeat escape analysis and heap-allocate a
	// Context at every node of the filter DAG, so implementations suppress gocritic's hugeParam check instead.
	OnFilterImage(ctx Context) FilterResult

	// OnInputLayerBounds reports the layer-space bounds required from this filter's input(s) to cover desiredOutput.
	OnInputLayerBounds(mapping *Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect

	// OnOutputLayerBounds reports the layer-space bounds of this filter's output (nil = unbounded).
	OnOutputLayerBounds(mapping *Mapping, contentBounds *geom.IRect) *geom.IRect

	// OnComputeFastBounds computes a conservative bounds estimate without a matrix (the default, union-of-children
	// behavior is available via DefaultComputeFastBounds).
	OnComputeFastBounds(bounds geom.Rect) geom.Rect

	// OnAffectsTransparentBlack reports whether this filter alone (ignoring its inputs) can turn transparent black into
	// a non-transparent result.
	OnAffectsTransparentBlack() bool

	// IgnoreInputsAffectsTransparentBlack reports whether AffectsTransparentBlack should skip recursing into this
	// filter's inputs.
	IgnoreInputsAffectsTransparentBlack() bool

	// OnCTMCapability reports the layer-matrix complexity this filter can handle (most filters:
	// MatrixCapabilityScaleTranslate).
	OnCTMCapability() MatrixCapability

	// OnIsColorFilterNode reports whether this filter is a pure color-filter node, returning the color filter when it
	// is.
	OnIsColorFilterNode() (shaders.ColorFilter, bool)
}

var nextFilterUniqueID atomic.Int32

// FilterBase holds the data common to every filter: the input filters (entries may be nil, meaning the dynamic source
// image), whether the DAG references the source, and a globally unique ID for the result cache.
type FilterBase struct {
	inputs   []Filter
	usesSrc  bool
	uniqueID int32
}

// NewFilterBase builds a FilterBase with inferred usesSrc: true when any input is nil or itself uses the source.
func NewFilterBase(inputs ...Filter) FilterBase {
	usesSrc := false
	for _, in := range inputs {
		if in == nil || in.Base().usesSrc {
			usesSrc = true
			break
		}
	}
	return newFilterBase(inputs, usesSrc)
}

// NewFilterBaseUsesSrc builds a FilterBase with an explicit usesSrc override (used by compose).
func NewFilterBaseUsesSrc(usesSrc bool, inputs ...Filter) FilterBase {
	return newFilterBase(inputs, usesSrc)
}

func newFilterBase(inputs []Filter, usesSrc bool) FilterBase {
	id := nextFilterUniqueID.Add(1)
	for id == 0 {
		id = nextFilterUniqueID.Add(1)
	}
	return FilterBase{inputs: inputs, usesSrc: usesSrc, uniqueID: id}
}

// CountInputs returns the number of input filters.
func (b *FilterBase) CountInputs() int { return len(b.inputs) }

// Input returns the input filter at i (may be nil).
func (b *FilterBase) Input(i int) Filter { return b.inputs[i] }

// UsesSource reports whether the filter DAG references the dynamic source image.
func (b *FilterBase) UsesSource() bool { return b.usesSrc }

// InputLayerBounds returns the required input for the input filter at index, or contentBounds ∩ desiredOutput when that
// input is nil (the identity/source case).
func (b *FilterBase) InputLayerBounds(index int, mapping *Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	if child := b.inputs[index]; child != nil {
		return child.OnInputLayerBounds(mapping, desiredOutput, contentBounds)
	}
	// NOTE: We don't calculate the intersection between content and the root desired output because the desired output
	// can expand or contract as it propagates through the filter graph.
	visibleContent := desiredOutput
	if contentBounds != nil && !visibleContent.Intersect(*contentBounds) {
		return geom.IRect{}
	}
	return visibleContent
}

// OutputLayerBounds returns the output for the input filter at index, or contentBounds when nil (identity applied to
// the source).
func (b *FilterBase) OutputLayerBounds(index int, mapping *Mapping, contentBounds *geom.IRect) *geom.IRect {
	if child := b.inputs[index]; child != nil {
		return child.OnOutputLayerBounds(mapping, contentBounds)
	}
	return copyOptRect(contentBounds)
}

// ChildOutput evaluates the input filter at index, or returns the context's dynamic source when the input is nil.
func (b *FilterBase) ChildOutput(index int, ctx *Context) FilterResult {
	if child := b.inputs[index]; child != nil {
		return FilterImage(child, ctx)
	}
	return *ctx.Source()
}

// copyOptRect clones an optional rect so callers can't alias stored state.
func copyOptRect(r *geom.IRect) *geom.IRect {
	if r == nil {
		return nil
	}
	c := *r
	return &c
}

///////////////////////////////////////////////////////////////////////////////
// public aggregate entry points

// FilterImage is the cache-wrapped evaluation entry point for a filter.
func FilterImage(f Filter, ctx *Context) FilterResult {
	ctx.markVisitedImageFilter()

	var result FilterResult
	layerMatrix := ctx.Mapping().LayerMatrix()
	if ctx.DesiredOutput().IsEmpty() || !layerMatrix.IsFinite() {
		return result
	}

	// Some image filters that operate on the source image still affect transparent black, so if there is clipping, we
	// may have optimized away the source image as an empty input, but still need to run the filter on it: usesSrc is
	// not equivalent to the source being non-nil.
	base := f.Base()
	key := filterCacheKey{
		uniqueID: base.uniqueID,
		matrix:   layerMatrix.As9(),
		output:   ctx.DesiredOutput(),
	}
	if base.usesSrc && ctx.Source().image != nil {
		key.srcGenID = ctx.Source().image.UniqueID()
		key.srcSubset = ctx.Source().image.Subset()
	}
	cache := ctx.Backend().Cache()
	if cache != nil {
		if cached, ok := cache.get(key); ok {
			ctx.markCacheHit()
			return cached
		}
	}

	result = f.OnFilterImage(*ctx)

	if cache != nil {
		cache.set(key, &result)
	}
	return result
}

// GetInputBounds returns the smallest layer-space bounds that provide sufficient input to cover desiredOutput (in
// device space). knownContentBounds is in parameter space, or nil when unknown.
func GetInputBounds(f Filter, mapping *Mapping, desiredOutput geom.IRect, knownContentBounds *geom.Rect) geom.IRect {
	desiredBounds := mapping.DeviceToLayerIRect(desiredOutput)
	var contentBounds *geom.IRect
	if knownContentBounds != nil {
		cb := RoundOut(mapping.ParamToLayerRect(*knownContentBounds))
		contentBounds = &cb
	}
	return f.OnInputLayerBounds(mapping, desiredBounds, contentBounds)
}

// GetOutputBounds returns the device-space bounds of the DAG's output for content covering contentBounds (parameter
// space). ok=false means the output is unbounded (fills whatever clipped device it's drawn into).
func GetOutputBounds(f Filter, mapping *Mapping, contentBounds geom.Rect) (geom.IRect, bool) {
	layerContent := RoundOut(mapping.ParamToLayerRect(contentBounds))
	filterOutput := f.OnOutputLayerBounds(mapping, &layerContent)
	if filterOutput == nil {
		return geom.IRect{}, false
	}
	return mapping.LayerToDeviceIRect(*filterOutput), true
}

// AffectsTransparentBlack reports whether f, or any filter in its input DAG, can turn transparent black into a
// non-transparent result.
func AffectsTransparentBlack(f Filter) bool {
	if f.OnAffectsTransparentBlack() {
		return true
	}
	if f.IgnoreInputsAffectsTransparentBlack() {
		return false
	}
	base := f.Base()
	for i := range base.inputs {
		if in := base.inputs[i]; in != nil && AffectsTransparentBlack(in) {
			return true
		}
	}
	return false
}

// CanComputeFastBounds reports whether f's fast bounds are meaningful (false if it affects transparent black, since
// then the true bounds are unbounded).
func CanComputeFastBounds(f Filter) bool { return !AffectsTransparentBlack(f) }

// DefaultComputeFastBounds is the default OnComputeFastBounds implementation: the union of all children's fast bounds
// (nil children pass src through).
func DefaultComputeFastBounds(f Filter, src geom.Rect) geom.Rect {
	base := f.Base()
	if len(base.inputs) == 0 {
		return src
	}
	combined := src
	if in := base.inputs[0]; in != nil {
		combined = in.OnComputeFastBounds(src)
	}
	for i := 1; i < len(base.inputs); i++ {
		if in := base.inputs[i]; in != nil {
			combined.Join(in.OnComputeFastBounds(src))
		} else {
			combined.Join(src)
		}
	}
	return combined
}

// CTMCapability returns the minimum layer-matrix capability over the filter and its non-nil inputs.
func CTMCapability(f Filter) MatrixCapability {
	result := f.OnCTMCapability()
	base := f.Base()
	for i := range base.inputs {
		if in := base.inputs[i]; in != nil {
			if c := CTMCapability(in); c < result {
				result = c
			}
		}
	}
	return result
}

// AsAColorFilter reports true only for a pure color-filter node with no input whose filter leaves transparent black
// untouched.
func AsAColorFilter(f Filter) (shaders.ColorFilter, bool) {
	cf, ok := f.OnIsColorFilterNode()
	if !ok {
		return nil, false
	}
	if f.Base().Input(0) != nil || ColorFilterAffectsTransparentBlack(cf) {
		return nil, false
	}
	return cf, true
}

// ColorFilterAffectsTransparentBlack reports whether cf turns (0,0,0,0) into anything else.
func ColorFilterAffectsTransparentBlack(cf shaders.ColorFilter) bool {
	if cf == nil {
		return false
	}
	out, ok := shaders.FilterColor4f(cf, colorcore.PMColor4f{})
	if !ok {
		return true // be conservative if the filter can't run on the pipeline
	}
	return out != colorcore.PMColor4f{}
}

// MakeILarge returns a huge-but-safe integer rect standing in for "unbounded" at the public filter-bounds boundary.
func MakeILarge() geom.IRect {
	const large = int32(1) << 29 // 1<<29 keeps width/height within int32 range
	return geom.IRect{Left: -large, Top: -large, Right: large, Bottom: large}
}
