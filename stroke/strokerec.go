// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Rec and FillPathWithPaint: the paint→stroker bridge. PaintSpec carries the slice of paint state the stroker
// consumes, so this package stays independent of the canvas paint type.

package stroke

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// PaintStyle identifies whether a paint fills, strokes, or both. The values match sk_paint_style_t.
type PaintStyle uint8

// PaintStyle values.
const (
	PaintStyleFill PaintStyle = iota
	PaintStyleStroke
	PaintStyleStrokeAndFill
)

// PaintSpec is the stroke-relevant slice of a paint's settings.
type PaintSpec struct {
	PathEffect PathEffect
	Style      PaintStyle
	Width      float32
	MiterLimit float32
	Cap        Cap
	Join       Join
}

// InitStyle selects the initial style for a freshly constructed Rec.
type InitStyle uint8

// InitStyle values.
const (
	InitStyleHairline InitStyle = iota
	InitStyleFill
)

// Style is the resolved style of a Rec: hairline, fill, stroke, or both.
type Style uint8

// Style values.
const (
	StyleHairline Style = iota
	StyleFill
	StyleStroke
	StyleStrokeAndFill
)

// fillStyleWidth marks a Rec as fill-only: must be < 0, since ==0 means hairline, and >0 means normal stroke.
const fillStyleWidth = float32(-1)

// paintDefaultsMiterLimit is the default miter limit for a new paint.
const paintDefaultsMiterLimit = float32(4)

// Rec holds the resolved stroking parameters (width, cap, join, miter limit, and whether both stroke and fill are
// drawn) derived from a paint.
type Rec struct {
	resScale      float32
	width         float32
	miterLimit    float32
	cap           Cap
	join          Join
	strokeAndFill bool
}

// NewStrokeRec returns a Rec initialized to the given style with default miter limit.
func NewStrokeRec(s InitStyle) Rec {
	width := float32(0)
	if s == InitStyleFill {
		width = fillStyleWidth
	}
	return Rec{
		resScale:   1,
		width:      width,
		miterLimit: paintDefaultsMiterLimit,
	}
}

// NewStrokeRecFromPaint returns a Rec derived from paint's own style.
func NewStrokeRecFromPaint(paint *PaintSpec, resScale float32) Rec {
	return NewStrokeRecFromPaintStyle(paint, paint.Style, resScale)
}

// NewStrokeRecFromPaintStyle returns a Rec derived from paint, overriding its style.
func NewStrokeRecFromPaintStyle(paint *PaintSpec, styleOverride PaintStyle, resScale float32) Rec {
	rec := Rec{resScale: resScale}

	switch styleOverride {
	case PaintStyleFill:
		rec.width = fillStyleWidth
		rec.strokeAndFill = false
	case PaintStyleStroke:
		rec.width = paint.Width
		rec.strokeAndFill = false
	case PaintStyleStrokeAndFill:
		if paint.Width == 0 {
			// hairline+fill == fill
			rec.width = fillStyleWidth
			rec.strokeAndFill = false
		} else {
			rec.width = paint.Width
			rec.strokeAndFill = true
		}
	default:
		// fall back on just fill
		rec.width = fillStyleWidth
		rec.strokeAndFill = false
	}

	// copy these from the paint, regardless of our "style"
	rec.miterLimit = paint.MiterLimit
	rec.cap = paint.Cap
	rec.join = paint.Join
	return rec
}

// Style returns the resolved stroke style.
func (r *Rec) Style() Style {
	switch {
	case r.width < 0:
		return StyleFill
	case r.width == 0:
		return StyleHairline
	case r.strokeAndFill:
		return StyleStrokeAndFill
	default:
		return StyleStroke
	}
}

// Width returns the stroke width.
func (r *Rec) Width() float32 { return r.width }

// Miter returns the miter limit.
func (r *Rec) Miter() float32 { return r.miterLimit }

// Cap returns the cap style.
func (r *Rec) Cap() Cap { return r.cap }

// Join returns the join style.
func (r *Rec) Join() Join { return r.join }

// IsHairlineStyle reports whether the style is hairline.
func (r *Rec) IsHairlineStyle() bool { return r.Style() == StyleHairline }

// IsFillStyle reports whether the style is fill.
func (r *Rec) IsFillStyle() bool { return r.Style() == StyleFill }

// SetFillStyle sets the style to fill.
func (r *Rec) SetFillStyle() {
	r.width = fillStyleWidth
	r.strokeAndFill = false
}

// SetHairlineStyle sets the style to hairline.
func (r *Rec) SetHairlineStyle() {
	r.width = 0
	r.strokeAndFill = false
}

// SetStrokeStyle sets the style to stroke (or stroke-and-fill). Note, if width==0, then this request is taken to mean:
// strokeAndFill==true -> new style will be Fill; strokeAndFill==false -> new style will be Hairline.
func (r *Rec) SetStrokeStyle(width float32, strokeAndFill bool) {
	if strokeAndFill && width == 0 {
		// hairline+fill == fill
		r.SetFillStyle()
	} else {
		r.width = width
		r.strokeAndFill = strokeAndFill
	}
}

// SetStrokeParams sets the cap, join, and miter limit.
func (r *Rec) SetStrokeParams(strokeCap Cap, join Join, miterLimit float32) {
	r.cap = strokeCap
	r.join = join
	r.miterLimit = miterLimit
}

// ResScale returns the resolution scale used to tessellate curves.
func (r *Rec) ResScale() float32 { return r.resScale }

// SetResScale sets the resolution scale used to tessellate curves.
func (r *Rec) SetResScale(rs float32) { r.resScale = rs }

// NeedToApply reports whether this specifies any thick stroking, i.e. ApplyToPath will return true.
func (r *Rec) NeedToApply() bool {
	style := r.Style()
	return style == StyleStroke || style == StyleStrokeAndFill
}

// ApplyToPath applies these stroke parameters to src, returning the result in dst. If there was no change (i.e. style
// == hairline or fill) this returns false and dst is unchanged. src and dst may be the same path.
func (r *Rec) ApplyToPath(dst, src *path.Path) bool {
	if r.width <= 0 { // hairline or fill
		return false
	}

	stroker := NewStroke()
	stroker.SetCap(r.cap)
	stroker.SetJoin(r.join)
	stroker.SetMiterLimit(r.miterLimit)
	stroker.SetWidth(r.width)
	stroker.SetDoFill(r.strokeAndFill)
	stroker.SetResScale(r.resScale)
	stroker.StrokePath(src, dst)
	return true
}

// HasEqualEffect reports whether two Recs produce equal paths. Equality of produced paths does not take the
// ResScale parameter into account.
func (r *Rec) HasEqualEffect(other *Rec) bool {
	if !r.NeedToApply() {
		return r.Style() == other.Style()
	}
	return r.width == other.width &&
		(r.join != JoinMiter || r.miterLimit == other.miterLimit) &&
		r.cap == other.cap &&
		r.join == other.join &&
		r.strokeAndFill == other.strokeAndFill
}

// InflationRadius returns a conservative outset to apply to a geometry's bounds to account for this stroke.
func (r *Rec) InflationRadius() float32 {
	return GetInflationRadius(r.join, r.miterLimit, r.cap, r.width)
}

// GetInflationRadiusForPaint returns the conservative bounds outset for paint under style.
func GetInflationRadiusForPaint(paint *PaintSpec, style PaintStyle) float32 {
	width := paint.Width
	if style == PaintStyleFill {
		width = -1
	}
	return GetInflationRadius(paint.Join, paint.MiterLimit, paint.Cap, width)
}

// GetInflationRadius returns a conservative outset to apply to a geometry's bounds to account for a stroke with the
// given parameters.
func GetInflationRadius(join Join, miterLimit float32, strokeCap Cap, strokeWidth float32) float32 {
	if strokeWidth < 0 { // fill
		return 0
	} else if strokeWidth == 0 {
		// The result is in local (pre-transform) space, but a hairline's width is defined in device space, so no
		// local-space number can be exact here: under a matrix with scale s, this 1 maps to s device pixels while the
		// hairline always paints ~1 device pixel wide. The 1 is a nominal stand-in, not a computed bound.
		//
		// Callers needing device-space accuracy must inflate after the transform rather than scaling this result;
		// tessChopPathIfNecessary and strokeTessellateOp both do so by asking for the inflation of an equivalent
		// device-space stroke of width 1. Callers that inflate in local space (Style.AdjustBounds, computeFastBounds)
		// are safe without compensating, because every consumer of their bounds adds at least a device pixel of slop:
		// Canvas.quickRejectBounds outsets the clip by 1 for AA/hairline slop, and getUnclippedShapeDevBounds rounds
		// out to whole pixels. A hairline paints at most half a device pixel beyond its geometry: square and round caps
		// extend by the half-width, and the hairline lane scan-converts the polyline without building miter join
		// geometry, so the miter limit cannot push it further. That is why this branch ignores join, miterLimit and
		// cap, and why a device pixel of slop covers a hairline at any scale. Probed by sweeping a hairline near the
		// clip edge across scales from 1 down to 0.001: the clipping threshold is identical at every scale, so this
		// value is not load-bearing.
		return 1
	}

	// since we're stroked, outset the rect by the radius (and join type, caps)
	multiplier := float32(1)
	if join == JoinMiter {
		multiplier = max(multiplier, miterLimit)
	}
	if strokeCap == CapSquare {
		multiplier = max(multiplier, float32(math.Sqrt2))
	}
	return strokeWidth / 2 * multiplier
}

///////////////////////////////////////////////////////////////////////////////

// FillPathWithPaint returns the filled equivalent of the stroked path in dst (dst is reset first), and reports whether
// dst should be drawn filled (false means the paint was a hairline, and dst should be drawn as a hairline instead).
// cullRect is consumed only by path effects.
func FillPathWithPaint(src *path.Path, paint *PaintSpec, dst *path.Path, cullRect *geom.Rect, ctm *geom.Matrix) bool {
	// This builds into a pooled working path that is copied into dst at the end (Set, reusing dst's storage) instead of
	// moving its slices away, so the working storage returns to the pool for the next draw; src is read only up to that
	// final copy, so src == dst stays safe.
	builder := path.Borrow()

	if !src.IsFinite() {
		dst.Set(builder)
		path.Recycle(builder)
		return false
	}

	resScale := ctm.ComputeResScaleForStroking()
	rec := NewStrokeRecFromPaint(paint, resScale)

	srcPtr := src
	var filtered *path.Path
	if paint.PathEffect != nil {
		// FilterPath takes the address of rec (it may rewrite the stroke params, e.g. dash's line special-case, or
		// 2D-line switching to fill). Passing &rec directly would force rec to the heap on every stroked draw, even the
		// common no-path-effect one. Give the effect a frame-local copy so the escape is confined to this branch, then
		// read the (possibly rewritten) rec back — identical to mutating rec in place.
		peRec := rec
		if paint.PathEffect.FilterPath(builder, src, &peRec, cullRect, ctm) {
			// the effect's output becomes the new source, and stroking (if any) re-applies to it
			filtered = builder
			srcPtr = filtered
			builder = path.Borrow()
		}
		rec = peRec
	}
	if !rec.ApplyToPath(builder, srcPtr) {
		builder.Set(srcPtr) // no thick stroke; pass the source through
	}

	if !builder.IsFinite() {
		builder.Rewind()
	}
	dst.Set(builder)
	path.Recycle(builder)
	if filtered != nil {
		path.Recycle(filtered)
	}
	return !rec.IsHairlineStyle()
}

// FillPathWithPaintResScale is the resScale-only overload of FillPathWithPaint: the ctm is Scale(resScale, resScale).
func FillPathWithPaintResScale(src *path.Path, paint *PaintSpec, dst *path.Path, cullRect *geom.Rect, resScale float32) bool {
	var ctm geom.Matrix
	ctm.SetScale(resScale, resScale)
	return FillPathWithPaint(src, paint, dst, cullRect, &ctm)
}
