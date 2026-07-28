// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The paint: solid color, antialias, blend mode, style, stroke geometry (consumed by the stroker), and the attached
// shaders, color/mask/image filters, and path effects.

package canvas

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stroke"
)

// Style selects whether geometry is filled, stroked, or both.
type Style uint8

// Style values.
const (
	StyleFill Style = iota
	StyleStroke
	StyleStrokeAndFill
)

// StrokeCap selects the treatment of stroke endpoints.
type StrokeCap uint8

// StrokeCap values.
const (
	CapButt StrokeCap = iota
	CapRound
	CapSquare
)

// StrokeJoin selects the treatment of stroke corners.
type StrokeJoin uint8

// StrokeJoin values.
const (
	JoinMiter StrokeJoin = iota
	JoinRound
	JoinBevel
)

// Paint carries the drawing state that is publicly reachable: color, blend mode, style, stroke geometry, the AA/dither
// flags, the attached path effect (concrete effects live in the patheffect package), the attached shader (shaders
// package), the attached color filter (colorfilter package), the attached mask filter (maskfilter package), and the
// attached image filter (imagefilter package, by filtercore.Filter interface). When a shader is set, the paint color's
// alpha still modulates the shader's output.
type Paint struct {
	PathEffect  stroke.PathEffect
	Shader      shaders.Shader
	ColorFilter shaders.ColorFilter
	MaskFilter  maskfilter.MaskFilter
	ImageFilter filtercore.Filter
	Color       colorcore.Color
	BlendMode   raster.BlendMode
	Style       Style
	StrokeWidth float32
	MiterLimit  float32
	Cap         StrokeCap
	Join        StrokeJoin
	AntiAlias   bool
	Dither      bool
	// forceRasterPipeline (internal) makes chooseBlitter skip the legacy fixed-point image-sampler lane and always
	// build the raster-pipeline blitter. The drawAtlas CPU lowering sets it: the atlas lane's sampling is defined in
	// terms of the pipeline's float bilerp, not the legacy sampler's 4-bit weights.
	forceRasterPipeline bool
}

// NewPaint returns a paint with the defaults: opaque black, src-over, fill style, stroke width 0, miter limit 4, butt
// cap, miter join, no AA, no dither.
func NewPaint() *Paint {
	return &Paint{
		Color:      0xFF000000,
		BlendMode:  raster.BlendSrcOver,
		MiterLimit: 4,
	}
}

// affectsAlpha reports whether the color filter can change alpha.
func affectsAlpha(cf shaders.ColorFilter) bool {
	return cf != nil && !cf.IsAlphaUnchanged()
}

// nothingToDraw reports whether a draw with this paint can be skipped entirely (e.g. a fully transparent color in a
// blend mode where that leaves the destination unchanged).
func (p *Paint) nothingToDraw() bool {
	switch p.BlendMode {
	case raster.BlendSrcOver, raster.BlendSrcATop, raster.BlendDstOut, raster.BlendDstOver,
		raster.BlendPlus:
		if p.Color.A() == 0 {
			return !affectsAlpha(p.ColorFilter) && p.ImageFilter == nil
		}
		return false
	case raster.BlendDst:
		return true
	default:
		return false
	}
}

// canComputeFastBounds reports whether computeFastBounds is usable: an image filter must not affect transparent black,
// and a path effect must be able to compute fast bounds (a nil-bounds dry run).
func (p *Paint) canComputeFastBounds() bool {
	if p.ImageFilter != nil && !filtercore.CanComputeFastBounds(p.ImageFilter) {
		return false
	}
	if p.PathEffect != nil && !p.PathEffect.ComputeFastBounds(nil) {
		return false
	}
	return true
}

// StrokeSpec bundles the paint's stroke-relevant fields (style, stroke geometry, path effect) for callers outside the
// canvas package that measure text or fill with this paint — the text-measurement and glyph-intercept use cases,
// where the caller hands the spec to the font/textblob packages exactly as the internal draw path does.
func (p *Paint) StrokeSpec() stroke.PaintSpec { return p.strokeSpec() }

// strokeSpec bundles the paint's stroke-relevant fields for the stroke package.
func (p *Paint) strokeSpec() stroke.PaintSpec {
	return stroke.PaintSpec{
		PathEffect: p.PathEffect,
		Style:      stroke.PaintStyle(p.Style),
		Width:      p.StrokeWidth,
		MiterLimit: p.MiterLimit,
		Cap:        stroke.Cap(p.Cap),
		Join:       stroke.Join(p.Join),
	}
}

// strokeInflationRadius returns how far a stroke with these parameters can extend beyond the source geometry.
func strokeInflationRadius(join StrokeJoin, miterLimit float32, strokeCap StrokeCap, strokeWidth float32) float32 {
	return stroke.GetInflationRadius(stroke.Join(join), miterLimit, stroke.Cap(strokeCap), strokeWidth)
}

// computeFastBounds grows orig by the paint's effects (path effect, stroke, mask filter, image filter) to a
// conservative bounds for the draw.
func (p *Paint) computeFastBounds(orig geom.Rect) geom.Rect {
	// ultra fast-case: filling with no frills
	if p.Style == StyleFill && p.PathEffect == nil && p.MaskFilter == nil && p.ImageFilter == nil {
		return orig
	}
	src := orig
	if p.PathEffect != nil {
		// ComputeFastBounds takes the address of the working rect. Passing &src directly would force src to the heap on
		// every draw that reaches here (stroke/mask-filter/image-filter draws, the common non-fill case), even though
		// the path effect is usually nil. Confine the address-taken copy to this branch so the allocation only happens
		// when an effect is present.
		peBounds := src
		p.PathEffect.ComputeFastBounds(&peBounds)
		src = peBounds
	}
	if p.Style != StyleFill {
		radius := strokeInflationRadius(p.Join, p.MiterLimit, p.Cap, p.StrokeWidth)
		src = src.Outset(radius, radius)
	}
	if p.MaskFilter != nil {
		src = p.MaskFilter.ComputeFastBounds(src)
	}
	if p.ImageFilter != nil {
		src = p.ImageFilter.OnComputeFastBounds(src)
	}
	return src
}

// FillPath applies this paint's path effect and, for stroked styles, its stroke geometry to src, writing the resulting
// fill path into dst at the given resolution scale (a larger resScale asks for finer curve approximation) and optional
// cull rect. It returns true if dst is a real fill to be painted with the paint's fill settings; false means the paint
// draws nothing beyond a hairline and dst holds the modified path. unison calls this directly.
func (p *Paint) FillPath(src, dst *path.Path, cullRect *geom.Rect, resScale float32) bool {
	spec := p.strokeSpec()
	return stroke.FillPathWithPaintResScale(src, &spec, dst, cullRect, resScale)
}
