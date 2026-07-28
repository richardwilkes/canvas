// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The Backend, Context, Stats, and FilterCache types. The Context carries no color space (everything is sRGB) and the
// Backend's color type is pinned to N32; surface props / pixel-geometry plumbing is likewise dropped (filter surfaces
// have unknown geometry, so text inside them renders grayscale AA per the layer-props rule, E.4). The cache is scoped
// to the Backend instance, which one canvas filter evaluation owns — enough for the same within-DAG reuse (e.g. drop
// shadow referencing its input twice) without needing a global byte budget.

package filtercore

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// PaintParams is the paint configuration filtercore draws with: shader/color + color filter + blend + AA + dither, as
// constructed by FilterResult and autoSurface.
type PaintParams struct {
	Shader      shaders.Shader
	ColorFilter shaders.ColorFilter
	Color       colorcore.Color4f // used when Shader is nil
	Blend       raster.BlendMode
	AntiAlias   bool
	Dither      bool
}

// Device is the canvas surface filtercore renders through. The canvas package implements it over BitmapDevice; a GPU
// implementation arrives with the GL backend. Methods correspond to the draw entry points FilterResult.Draw and
// autoSurface consume.
type Device interface {
	// Size returns the device's pixel dimensions.
	Size() geom.ISize
	// LocalToDevice returns the current local-to-device transform.
	LocalToDevice() geom.Matrix
	// SetLocalToDevice installs a device-space transform (the autoSurface's canvas translate/concat collapse to this
	// since filtercore does its own save bookkeeping).
	SetLocalToDevice(m geom.Matrix)
	// DevClipBounds returns the device-space clip bounds.
	DevClipBounds() geom.IRect
	// PushClipStack/PopClipStack save and restore the clip stack.
	PushClipStack()
	PopClipStack()
	// ClipRect intersects the clip with r (in local coordinates), optionally anti-aliased.
	ClipRect(r geom.Rect, aa bool)
	// ClipIRect intersects the clip with the device-space integer rect (equivalent to a canvas clipIRect call with the
	// autoSurface's identity-translation local matrix).
	ClipIRect(r geom.IRect)
	// DrawPaint fills the clip with the paint.
	DrawPaint(pp *PaintParams)
	// DrawRect fills r (mapped by LocalToDevice) with the paint.
	DrawRect(r geom.Rect, pp *PaintParams)
	// DrawSpecial draws the subset view of img using the given device-space matrix (LocalToDevice is ignored),
	// sampling, and paint (shader ignored). On the raster backend fast/strict sampling constraints coincide (both
	// sample the subset view), so no constraint parameter exists.
	DrawSpecial(img *SpecialImage, deviceMatrix geom.Matrix, sampling shaders.SamplingOptions,
		pp *PaintParams)
	// DrawImageRect draws img with the strict source-rect constraint (MakeFromImage's fractional-src lane; geometry is
	// in local coordinates via LocalToDevice). The image is polymorphic: raster devices resolve a texture
	// backing to CPU pixels, the GPU device samples it directly.
	DrawImageRect(img imagecore.DrawableImage, src, dst geom.Rect, sampling shaders.SamplingOptions,
		pp *PaintParams)
	// SnapSpecial returns a special image viewing subset of this device's pixels (sharing the storage; devices snapped
	// by filtercore are not drawn to afterwards).
	SnapSpecial(subset geom.IRect) *SpecialImage
}

// Backend provides the backend-specific services of the filter pipeline.
type Backend interface {
	// MakeDevice creates an offscreen intermediate device (transparent-initialized N32 premul).
	MakeDevice(size geom.ISize) Device
	// MakeImage wraps the subset of an existing image as a SpecialImage: on the raster backend, lazy images decode and
	// texture backings resolve to CPU; on the GPU backend, a texture backing is wrapped without readback.
	MakeImage(subset geom.IRect, image imagecore.DrawableImage) *SpecialImage
	// BlurEngine returns the backend's blur algorithm provider.
	BlurEngine() BlurEngine
	// Cache returns the filter-result cache (never nil).
	Cache() *FilterCache
}

// Stats collects counters for a single filter evaluation.
type Stats struct {
	NumVisitedImageFilters    int
	NumCacheHits              int
	NumOffscreenSurfaces      int
	NumShaderClampedDraws     int
	NumShaderBasedTilingDraws int
}

// Context carries the mapping, desired output, and source image for a filter evaluation. It is a value type; the
// WithNewX helpers return copies.
type Context struct {
	backend       Backend
	stats         *Stats
	source        FilterResult
	mapping       Mapping
	desiredOutput geom.IRect
}

// NewContext builds a Context from its component parts (there is no color space — everything is sRGB).
func NewContext(backend Backend, mapping *Mapping, desiredOutput geom.IRect, source *FilterResult, stats *Stats) Context {
	return Context{
		backend:       backend,
		mapping:       *mapping,
		desiredOutput: desiredOutput,
		source:        *source,
		stats:         stats,
	}
}

// Backend returns the backend.
func (c *Context) Backend() Backend { return c.backend }

// Mapping returns the parameter/layer/device mapping.
func (c *Context) Mapping() *Mapping { return &c.mapping }

// DesiredOutput returns the layer-space bounds the filtered image will be clipped to.
func (c *Context) DesiredOutput() geom.IRect { return c.desiredOutput }

// Source returns the image to use whenever an expected input filter is nil.
func (c *Context) Source() *FilterResult { return &c.source }

// WithNewMapping returns a copy of c with mapping replaced.
func (c *Context) WithNewMapping(mapping *Mapping) Context {
	out := *c
	out.mapping = *mapping
	return out
}

// WithNewDesiredOutput returns a copy of c with desiredOutput replaced.
func (c *Context) WithNewDesiredOutput(desiredOutput geom.IRect) Context {
	out := *c
	out.desiredOutput = desiredOutput
	return out
}

// WithNewSource returns a copy of c with source replaced.
func (c *Context) WithNewSource(source *FilterResult) Context {
	out := *c
	out.source = *source
	return out
}

// stats tracking

func (c *Context) markVisitedImageFilter() {
	if c.stats != nil {
		c.stats.NumVisitedImageFilters++
	}
}

func (c *Context) markCacheHit() {
	if c.stats != nil {
		c.stats.NumCacheHits++
	}
}

// MarkNewSurface records that a new offscreen surface was created (exported: the canvas layer-snap lane counts its
// offscreen too).
func (c *Context) MarkNewSurface() {
	if c.stats != nil {
		c.stats.NumOffscreenSurfaces++
	}
}

func (c *Context) markShaderBasedTilingRequired(tileMode shaders.TileMode) {
	if c.stats != nil {
		if tileMode == shaders.TileClamp {
			c.stats.NumShaderClampedDraws++
		} else {
			c.stats.NumShaderBasedTilingDraws++
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// FilterCache

// filterCacheKey identifies a memoized filter evaluation: filter unique ID, layer matrix, desired output, and the
// source image's generation ID + subset (when the filter graph references the source).
type filterCacheKey struct {
	uniqueID  int32
	matrix    [9]float32
	output    geom.IRect
	srcGenID  uint32
	srcSubset geom.IRect
}

// FilterCache memoizes filter results during an evaluation.
type FilterCache struct {
	entries map[filterCacheKey]FilterResult
}

// NewFilterCache returns an empty cache.
func NewFilterCache() *FilterCache {
	return &FilterCache{entries: make(map[filterCacheKey]FilterResult)}
}

func (fc *FilterCache) get(key filterCacheKey) (FilterResult, bool) {
	r, ok := fc.entries[key]
	return r, ok
}

func (fc *FilterCache) set(key filterCacheKey, result *FilterResult) {
	fc.entries[key] = *result
}
