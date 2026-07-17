// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Quad represents 4 points forming an arbitrary quadrilateral, stored in CCW triangle-strip order (top-left,
// bottom-left, top-right, bottom-right), with a classification that tells the AA tessellation code how general the
// geometry is. 4-lane float vectors are represented as [4]float32, with the small shuffle/blend helpers at the bottom
// of this file standing in for SIMD lane operations.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// QuadType classifies how general a quad's geometry is.
type QuadType int32

// QuadType values, from most to least specific.
const (
	// QuadTypeAxisAligned: the 4 points remain an axis-aligned rectangle; their logical indices may not respect TL, BL,
	// TR, BR ordering if the transform was a 90 degree rotation or mirror.
	QuadTypeAxisAligned QuadType = iota
	// QuadTypeRectilinear: a rectangle subjected to a rotation, its corners are right angles.
	QuadTypeRectilinear
	// QuadTypeGeneral: arbitrary 2D quadrilateral; ws are all 1.
	QuadTypeGeneral
	// QuadTypePerspective: allows the w coordinates to be non-unity.
	QuadTypePerspective
)

// Quad is a general quadrilateral: 4 (x, y, w) vertices plus a QuadType classification. The zero value is an
// uninitialized axis-aligned quad with ws == 1.
type Quad struct {
	xs       [4]float32
	ys       [4]float32
	ws       [4]float32
	quadType QuadType
}

// MakeQuadFromRectNoTransform creates an axis-aligned quad directly from a rect, with no matrix applied.
func MakeQuadFromRectNoTransform(r geom.Rect) Quad {
	return Quad{
		xs: [4]float32{r.Left, r.Left, r.Right, r.Right},
		ys: [4]float32{r.Top, r.Bottom, r.Top, r.Bottom},
		ws: [4]float32{1, 1, 1, 1},
	}
}

// mapRectTranslateScale maps r's corners through m, taking a fast path when m is only a translation and/or scale.
func mapRectTranslateScale(r geom.Rect, m *geom.Matrix) (xs, ys [4]float32) {
	tm := m.Type()
	// r as (L, T, R, B) lanes.
	l, t, rr, b := r.Left, r.Top, r.Right, r.Bottom
	if tm > 0 { // > kIdentity_Mask
		tx, ty := m.Get(geom.MTransX), m.Get(geom.MTransY)
		if tm <= geom.TypeTranslate {
			l += tx
			t += ty
			rr += tx
			b += ty
		} else {
			sx, sy := m.Get(geom.MScaleX), m.Get(geom.MScaleY)
			l = l*sx + tx
			t = t*sy + ty
			rr = rr*sx + tx
			b = b*sy + ty
		}
	}
	return [4]float32{l, l, rr, rr}, [4]float32{t, b, t, b}
}

// mapQuadGeneral maps the quad points (qx, qy) through the general matrix m, producing device coordinates and w values
// (all 1 when m has no perspective).
func mapQuadGeneral(qx, qy [4]float32, m *geom.Matrix) (xs, ys, ws [4]float32) {
	sx, kx, tx := m.Get(geom.MScaleX), m.Get(geom.MSkewX), m.Get(geom.MTransX)
	ky, sy, ty := m.Get(geom.MSkewY), m.Get(geom.MScaleY), m.Get(geom.MTransY)
	for i := range 4 {
		xs[i] = sx*qx[i] + (kx*qy[i] + tx)
		ys[i] = ky*qx[i] + (sy*qy[i] + ty)
	}
	if m.HasPerspective() {
		p0, p1, p2 := m.Get(geom.MPersp0), m.Get(geom.MPersp1), m.Get(geom.MPersp2)
		for i := range 4 {
			ws[i] = p0*qx[i] + (p1*qy[i] + p2)
		}
	} else {
		ws = [4]float32{1, 1, 1, 1}
	}
	return xs, ys, ws
}

// quadTypeForTransformedRect classifies the quad type that results from mapping a rect through m.
func quadTypeForTransformedRect(m *geom.Matrix) QuadType {
	switch {
	case m.RectStaysRect():
		return QuadTypeAxisAligned
	case m.PreservesRightAngles():
		return QuadTypeRectilinear
	case m.HasPerspective():
		return QuadTypePerspective
	default:
		return QuadTypeGeneral
	}
}

// quadTypeForPoints classifies the quad type for a general 4-point quad ('pts' in TL, TR, BR, BL order).
func quadTypeForPoints(pts *[4]geom.Point, m *geom.Matrix) QuadType {
	if m.HasPerspective() {
		return QuadTypePerspective
	}
	// If 'pts' was formed directly from a rect's corners and not transformed further, it is safe to use the quad type
	// derived from 'm'. Otherwise don't waste any more time and assume the most general 2D quad.
	if pts[0].X == pts[3].X && pts[1].X == pts[2].X &&
		pts[0].Y == pts[1].Y && pts[2].Y == pts[3].Y {
		return quadTypeForTransformedRect(m)
	}
	return QuadTypeGeneral
}

// MakeQuadFromRect creates a quad by mapping r through m.
func MakeQuadFromRect(r geom.Rect, m *geom.Matrix) Quad {
	tm := m.Type()
	if tm <= geom.TypeScale|geom.TypeTranslate {
		xs, ys := mapRectTranslateScale(r, m)
		return Quad{xs: xs, ys: ys, ws: [4]float32{1, 1, 1, 1}, quadType: QuadTypeAxisAligned}
	}
	rx := [4]float32{r.Left, r.Left, r.Right, r.Right}
	ry := [4]float32{r.Top, r.Bottom, r.Top, r.Bottom}
	xs, ys, ws := mapQuadGeneral(rx, ry, m)
	return Quad{xs: xs, ys: ys, ws: ws, quadType: quadTypeForTransformedRect(m)}
}

// MakeQuadFromPoints creates a quad from 4 points in TL, TR, BR, BL order, rearranged into CCW tri-strip order and
// mapped through m.
func MakeQuadFromPoints(pts *[4]geom.Point, m *geom.Matrix) Quad {
	// Rearrange from TL, TR, BR, BL order into CCW tri-strip order (TL, BL, TR, BR).
	xs := [4]float32{pts[0].X, pts[3].X, pts[1].X, pts[2].X}
	ys := [4]float32{pts[0].Y, pts[3].Y, pts[1].Y, pts[2].Y}
	quadType := quadTypeForPoints(pts, m)
	if m.IsIdentity() {
		return Quad{xs: xs, ys: ys, ws: [4]float32{1, 1, 1, 1}, quadType: quadType}
	}
	mx, my, mw := mapQuadGeneral(xs, ys, m)
	return Quad{xs: mx, ys: my, ws: mw, quadType: quadType}
}

// Point3 returns the i'th vertex's raw (x, y, w) coordinates, without perspective division.
func (q *Quad) Point3(i int) (x, y, w float32) { return q.xs[i], q.ys[i], q.ws[i] }

// Point returns the i'th vertex, dividing by w for perspective quads.
func (q *Quad) Point(i int) geom.Point {
	if q.quadType == QuadTypePerspective {
		return geom.Point{X: q.xs[i] / q.ws[i], Y: q.ys[i] / q.ws[i]}
	}
	return geom.Point{X: q.xs[i], Y: q.ys[i]}
}

// Bounds returns the quad's device-space bounding rect.
func (q *Quad) Bounds() geom.Rect {
	if q.quadType == QuadTypePerspective {
		return q.projectedBounds()
	}
	minOf := func(c *[4]float32) float32 { return min(min(c[0], c[1]), min(c[2], c[3])) }
	maxOf := func(c *[4]float32) float32 { return max(max(c[0], c[1]), max(c[2], c[3])) }
	return geom.Rect{
		Left: minOf(&q.xs), Top: minOf(&q.ys), Right: maxOf(&q.xs),
		Bottom: maxOf(&q.ys),
	}
}

// IsFinite reports whether every coordinate of the quad is finite.
func (q *Quad) IsFinite() bool {
	// If any coordinate is infinity or NaN, then multiplying it with 0 will make accum NaN.
	accum := float32(0)
	for i := range 4 {
		accum *= q.xs[i]
		accum *= q.ys[i]
		accum *= q.ws[i]
	}
	return accum == 0
}

// X returns the i'th vertex's x coordinate.
func (q *Quad) X(i int) float32 { return q.xs[i] }

// Y returns the i'th vertex's y coordinate.
func (q *Quad) Y(i int) float32 { return q.ys[i] }

// W returns the i'th vertex's w coordinate.
func (q *Quad) W(i int) float32 { return q.ws[i] }

// IW returns 1/w for the i'th vertex via plain division, letting inf/NaN happen for w == 0.
func (q *Quad) IW(i int) float32 { return 1 / q.ws[i] }

// X4f returns all 4 x coordinates as a lane vector.
func (q *Quad) X4f() [4]float32 { return q.xs }

// Y4f returns all 4 y coordinates as a lane vector.
func (q *Quad) Y4f() [4]float32 { return q.ys }

// W4f returns all 4 w coordinates as a lane vector.
func (q *Quad) W4f() [4]float32 { return q.ws }

// Xs returns a pointer to the mutable x lanes.
func (q *Quad) Xs() *[4]float32 { return &q.xs }

// Ys returns a pointer to the mutable y lanes.
func (q *Quad) Ys() *[4]float32 { return &q.ys }

// Ws returns a pointer to the mutable w lanes.
func (q *Quad) Ws() *[4]float32 { return &q.ws }

// QuadType returns the quad's classification.
func (q *Quad) QuadType() QuadType { return q.quadType }

// HasPerspective reports whether the quad's w coordinates can be non-unity.
func (q *Quad) HasPerspective() bool { return q.quadType == QuadTypePerspective }

// scalarIsInt reports whether v has no fractional part.
func scalarIsInt(v float32) bool { return v == float32(math.Floor(float64(v))) }

// aaAffectsRect reports whether any AA-enabled edge is at a non-integer coordinate; an AA-enabled edge at an integer
// coordinate could be drawn non-AA and be visually identical.
func aaAffectsRect(edgeFlags gpu.QuadAAFlags, ql, qt, qr, qb float32) bool {
	return (edgeFlags&gpu.QuadAAFlagLeft != 0 && !scalarIsInt(ql)) ||
		(edgeFlags&gpu.QuadAAFlagRight != 0 && !scalarIsInt(qr)) ||
		(edgeFlags&gpu.QuadAAFlagTop != 0 && !scalarIsInt(qt)) ||
		(edgeFlags&gpu.QuadAAFlagBottom != 0 && !scalarIsInt(qb))
}

// AAHasEffectOnRect reports whether enabling AA on the given edges would visually change an axis-aligned rect. Only
// valid when the quad type is axis-aligned.
func (q *Quad) AAHasEffectOnRect(edgeFlags gpu.QuadAAFlags) bool {
	if q.quadType != QuadTypeAxisAligned {
		panic("aaHasEffectOnRect requires an axis-aligned quad")
	}
	// If rect, ws must all be 1s so no need to divide.
	return aaAffectsRect(edgeFlags, q.xs[0], q.ys[0], q.xs[3], q.ys[3])
}

// AsRect returns the quad's bounds and true if this quad is axis-aligned and still has its top-left corner at v0.
func (q *Quad) AsRect() (geom.Rect, bool) {
	if q.quadType != QuadTypeAxisAligned {
		return geom.Rect{}, false
	}
	r := q.Bounds()
	// v0 at the geometric top-left is unique amongst axis-aligned vertex orders (90, 180, 270 rotations or axis flips
	// all move v0).
	return r, q.xs[0] == r.Left && q.ys[0] == r.Top
}

// SetQuadType sets the quad's classification, automatically ensuring ws are 1 if the new type is not perspective.
func (q *Quad) SetQuadType(newType QuadType) {
	if newType != QuadTypePerspective && q.quadType == QuadTypePerspective {
		q.ws = [4]float32{1, 1, 1, 1}
	}
	q.quadType = newType
}

// DrawQuad is the common work unit of a pair of device and local coordinates plus the edge flags controlling
// anti-aliasing.
type DrawQuad struct {
	Device    Quad
	Local     Quad
	EdgeFlags gpu.QuadAAFlags
}

//////////////////////////////////////////////////////////////////////////////
// 4-lane float and mask helpers used by the quad geometry code.

// nextCW rotates points/edge values clockwise assuming tri-strip order.
func nextCW(v [4]float32) [4]float32 { return [4]float32{v[2], v[0], v[3], v[1]} }

// nextCCW rotates counterclockwise.
func nextCCW(v [4]float32) [4]float32 { return [4]float32{v[1], v[3], v[0], v[2]} }

// nextDiag is two rotations in either direction.
func nextDiag(v [4]float32) [4]float32 { return [4]float32{v[3], v[2], v[1], v[0]} }

func nextCWMask(v [4]bool) [4]bool  { return [4]bool{v[2], v[0], v[3], v[1]} }
func nextCCWMask(v [4]bool) [4]bool { return [4]bool{v[1], v[3], v[0], v[2]} }

func f4Splat(v float32) [4]float32 { return [4]float32{v, v, v, v} }

func f4IfThenElse(mask [4]bool, t, e [4]float32) [4]float32 {
	var r [4]float32
	for i := range 4 {
		if mask[i] {
			r[i] = t[i]
		} else {
			r[i] = e[i]
		}
	}
	return r
}

func maskAny(m [4]bool) bool { return m[0] || m[1] || m[2] || m[3] }
func maskAll(m [4]bool) bool { return m[0] && m[1] && m[2] && m[3] }

func maskAnd(a, b [4]bool) [4]bool {
	return [4]bool{a[0] && b[0], a[1] && b[1], a[2] && b[2], a[3] && b[3]}
}

func maskOr(a, b [4]bool) [4]bool {
	return [4]bool{a[0] || b[0], a[1] || b[1], a[2] || b[2], a[3] || b[3]}
}

func maskNot(m [4]bool) [4]bool { return [4]bool{!m[0], !m[1], !m[2], !m[3]} }

func maskXor(a, b [4]bool) [4]bool {
	return [4]bool{a[0] != b[0], a[1] != b[1], a[2] != b[2], a[3] != b[3]}
}
