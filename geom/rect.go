// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import "math"

// Rect is a float32 left/top/right/bottom rectangle. A Rect is "empty" unless left < right and top < bottom; NaN
// components make a rect empty, since the emptiness check uses negated comparisons.
type Rect struct {
	Left   float32
	Top    float32
	Right  float32
	Bottom float32
}

// RectLTRB returns a Rect from its left, top, right, bottom edges.
func RectLTRB(l, t, r, b float32) Rect {
	return Rect{Left: l, Top: t, Right: r, Bottom: b}
}

// RectXYWH returns a Rect from its origin and dimensions.
func RectXYWH(x, y, w, h float32) Rect {
	return Rect{Left: x, Top: y, Right: x + w, Bottom: y + h}
}

// RectWH returns a Rect at the origin with the given dimensions.
func RectWH(w, h float32) Rect {
	return Rect{Right: w, Bottom: h}
}

// IsEmpty reports whether the rect is empty. Written with negated comparisons so NaN components report empty.
func (r Rect) IsEmpty() bool {
	return !(r.Left < r.Right && r.Top < r.Bottom)
}

// IsSorted reports whether Left <= Right and Top <= Bottom.
func (r Rect) IsSorted() bool {
	return r.Left <= r.Right && r.Top <= r.Bottom
}

// IsFinite reports whether all four edges are finite.
func (r Rect) IsFinite() bool {
	return IsFinite(r.Left, r.Top, r.Right, r.Bottom)
}

// Width returns Right - Left.
func (r Rect) Width() float32 {
	return r.Right - r.Left
}

// Height returns Bottom - Top.
func (r Rect) Height() float32 {
	return r.Bottom - r.Top
}

// CenterX returns the horizontal center, computed as left*0.5 + right*0.5 to avoid overflow.
func (r Rect) CenterX() float32 {
	return r.Left*0.5 + r.Right*0.5
}

// CenterY returns the vertical center.
func (r Rect) CenterY() float32 {
	return r.Top*0.5 + r.Bottom*0.5
}

// Center returns the center point.
func (r Rect) Center() Point {
	return Point{X: r.CenterX(), Y: r.CenterY()}
}

// Sort swaps edges in place so the rect is sorted.
func (r *Rect) Sort() {
	*r = r.Sorted()
}

// Sorted returns the rect with edges swapped as needed so Left <= Right and Top <= Bottom.
func (r Rect) Sorted() Rect {
	return Rect{
		Left:   min32(r.Left, r.Right),
		Top:    min32(r.Top, r.Bottom),
		Right:  max32(r.Left, r.Right),
		Bottom: max32(r.Top, r.Bottom),
	}
}

// Offset returns the rect translated by (dx, dy).
func (r Rect) Offset(dx, dy float32) Rect {
	return Rect{Left: r.Left + dx, Top: r.Top + dy, Right: r.Right + dx, Bottom: r.Bottom + dy}
}

// Inset returns the rect shrunk by (dx, dy) on each side (use negative values to outset).
func (r Rect) Inset(dx, dy float32) Rect {
	return Rect{Left: r.Left + dx, Top: r.Top + dy, Right: r.Right - dx, Bottom: r.Bottom - dy}
}

// Outset returns the rect grown by (dx, dy) on each side.
func (r Rect) Outset(dx, dy float32) Rect {
	return r.Inset(-dx, -dy)
}

// Intersect sets r to the intersection of r and other and returns true, or returns false leaving r unchanged when the
// intersection is empty. NaN handling depends on operand order: the *other* rect's edges are the first max/min
// argument, so NaN in other's edges is kept (forcing a false return) while NaN in the receiver's edges is silently
// discarded. Note this is the opposite of Intersects below, which uses the receiver's edges first. Locked in by the
// oracle probe TestRectIntersectProbe.
func (r *Rect) Intersect(other Rect) bool {
	l := max32(other.Left, r.Left)
	rt := min32(other.Right, r.Right)
	t := max32(other.Top, r.Top)
	b := min32(other.Bottom, r.Bottom)
	// The negated comparison returns false if any value is NaN.
	if !(l < rt && t < b) {
		return false
	}
	*r = Rect{Left: l, Top: t, Right: rt, Bottom: b}
	return true
}

// Intersects reports whether r and other overlap, using the receiver's edges first in the max/min operand order — the
// opposite of Intersect (see there); the two differ observably when NaN is involved.
func (r Rect) Intersects(other Rect) bool {
	l := max32(r.Left, other.Left)
	rt := min32(r.Right, other.Right)
	t := max32(r.Top, other.Top)
	b := min32(r.Bottom, other.Bottom)
	return l < rt && t < b
}

// Join expands r to include other. Empty operands are ignored; joining onto an empty rect adopts the other rect.
func (r *Rect) Join(other Rect) {
	if other.IsEmpty() {
		return
	}
	if r.IsEmpty() {
		*r = other
		return
	}
	r.Left = min32(r.Left, other.Left)
	r.Top = min32(r.Top, other.Top)
	r.Right = max32(r.Right, other.Right)
	r.Bottom = max32(r.Bottom, other.Bottom)
}

// ContainsPoint reports whether (x, y) is inside r, half-open on the right/bottom edges.
func (r Rect) ContainsPoint(x, y float32) bool {
	return x >= r.Left && x < r.Right && y >= r.Top && y < r.Bottom
}

// ContainsRect reports true only when both rects are non-empty and other is entirely inside r.
func (r Rect) ContainsRect(other Rect) bool {
	return !other.IsEmpty() && !r.IsEmpty() &&
		r.Left <= other.Left && r.Top <= other.Top &&
		r.Right >= other.Right && r.Bottom >= other.Bottom
}

// Round returns an IRect with every edge rounded using saturating round.
func (r Rect) Round() IRect {
	return IRect{
		Left:   RoundToInt(r.Left),
		Top:    RoundToInt(r.Top),
		Right:  RoundToInt(r.Right),
		Bottom: RoundToInt(r.Bottom),
	}
}

// RoundOut returns an IRect with left/top floored and right/bottom ceiled.
func (r Rect) RoundOut() IRect {
	return IRect{
		Left:   FloorToInt(r.Left),
		Top:    FloorToInt(r.Top),
		Right:  CeilToInt(r.Right),
		Bottom: CeilToInt(r.Bottom),
	}
}

// RoundOutRect is like RoundOut but stays in float32, without integer conversion.
func (r Rect) RoundOutRect() Rect {
	return Rect{
		Left:   FloorToScalar(r.Left),
		Top:    FloorToScalar(r.Top),
		Right:  CeilToScalar(r.Right),
		Bottom: CeilToScalar(r.Bottom),
	}
}

// RoundIn returns an IRect with left/top ceiled and right/bottom floored.
func (r Rect) RoundIn() IRect {
	return IRect{
		Left:   CeilToInt(r.Left),
		Top:    CeilToInt(r.Top),
		Right:  FloorToInt(r.Right),
		Bottom: FloorToInt(r.Bottom),
	}
}

// ToQuad returns the four corners in left-top, right-top, right-bottom, left-bottom order.
func (r Rect) ToQuad() [4]Point {
	return [4]Point{
		{X: r.Left, Y: r.Top},
		{X: r.Right, Y: r.Top},
		{X: r.Right, Y: r.Bottom},
		{X: r.Left, Y: r.Bottom},
	}
}

// SetBounds sets r to the bounds of pts and returns true. An empty pts sets r to the empty rect and still returns true;
// only a non-finite coordinate is a failure, setting r to the empty rect and returning false.
func (r *Rect) SetBounds(pts []Point) bool {
	if len(pts) == 0 {
		*r = Rect{}
		return true
	}
	l, t, rt, b := pts[0].X, pts[0].Y, pts[0].X, pts[0].Y
	// Non-finite values are detected by multiplying an accumulator by every coordinate: any NaN or infinity poisons the
	// product.
	nx, ny := float32(0), float32(0)
	for _, p := range pts {
		if p.X < l {
			l = p.X
		}
		if p.Y < t {
			t = p.Y
		}
		if p.X > rt {
			rt = p.X
		}
		if p.Y > b {
			b = p.Y
		}
		nx *= p.X
		ny *= p.Y
	}
	if nx == 0 && ny == 0 {
		*r = Rect{Left: l, Top: t, Right: rt, Bottom: b}
		return true
	}
	*r = Rect{}
	return false
}

// BoundsOrEmpty returns the bounds of pts, or the empty rect if pts is empty or non-finite.
func BoundsOrEmpty(pts []Point) Rect {
	var r Rect
	r.SetBounds(pts)
	return r
}

// SetBoundsNoCheck is like SetBounds, except that when pts contain a non-finite value the rect is poisoned to all-NaN
// instead of set empty. Matrix.MapRect's general affine path uses this, so non-finite corners produce a NaN rect there,
// observably different from an empty one.
func (r *Rect) SetBoundsNoCheck(pts []Point) {
	if !r.SetBounds(pts) {
		nan := float32(math.NaN())
		*r = Rect{Left: nan, Top: nan, Right: nan, Bottom: nan}
	}
}

// IRect is an int32 left/top/right/bottom rectangle.
type IRect struct {
	Left   int32
	Top    int32
	Right  int32
	Bottom int32
}

// IRectLTRB returns an IRect from its left, top, right, bottom edges.
func IRectLTRB(l, t, r, b int32) IRect {
	return IRect{Left: l, Top: t, Right: r, Bottom: b}
}

// IRectXYWH returns an IRect from its origin and dimensions.
func IRectXYWH(x, y, w, h int32) IRect {
	return IRect{Left: x, Top: y, Right: x + w, Bottom: y + h}
}

// IRectWH returns an IRect at the origin with the given dimensions.
func IRectWH(w, h int32) IRect {
	return IRect{Right: w, Bottom: h}
}

// IRectSize returns an IRect at the origin with the given size.
func IRectSize(s ISize) IRect {
	return IRect{Right: s.Width, Bottom: s.Height}
}

// Width64 returns Right - Left computed in int64, avoiding int32 overflow.
func (r IRect) Width64() int64 {
	return int64(r.Right) - int64(r.Left)
}

// Height64 returns Bottom - Top computed in int64, avoiding int32 overflow.
func (r IRect) Height64() int64 {
	return int64(r.Bottom) - int64(r.Top)
}

// Width returns Right - Left (may be invalid if the true width overflows int32; see Width64).
func (r IRect) Width() int32 {
	return r.Right - r.Left
}

// Height returns Bottom - Top (may be invalid if the true height overflows int32; see Height64).
func (r IRect) Height() int32 {
	return r.Bottom - r.Top
}

// Size returns the dimensions of the rect.
func (r IRect) Size() ISize {
	return ISize{Width: r.Width(), Height: r.Height()}
}

// IsEmpty reports true when width or height is not positive, or when either overflows int32.
func (r IRect) IsEmpty() bool {
	w := r.Width64()
	h := r.Height64()
	if w <= 0 || h <= 0 {
		return true
	}
	return (w|h)&^math.MaxInt32 != 0
}

// isEmpty64 is a cheaper emptiness test used internally by Join and Intersect.
func (r IRect) isEmpty64() bool {
	return r.Right <= r.Left || r.Bottom <= r.Top
}

// Offset returns the rect translated by (dx, dy).
func (r IRect) Offset(dx, dy int32) IRect {
	return IRect{Left: r.Left + dx, Top: r.Top + dy, Right: r.Right + dx, Bottom: r.Bottom + dy}
}

// Inset returns the rect shrunk by (dx, dy) on each side.
func (r IRect) Inset(dx, dy int32) IRect {
	return IRect{Left: r.Left + dx, Top: r.Top + dy, Right: r.Right - dx, Bottom: r.Bottom - dy}
}

// Sorted returns the rect with edges swapped as needed so Left <= Right and Top <= Bottom.
func (r IRect) Sorted() IRect {
	return IRect{
		Left:   minI32(r.Left, r.Right),
		Top:    minI32(r.Top, r.Bottom),
		Right:  maxI32(r.Left, r.Right),
		Bottom: maxI32(r.Top, r.Bottom),
	}
}

// Intersect sets r to the intersection and returns true, or returns false leaving r unchanged.
func (r *IRect) Intersect(other IRect) bool {
	l := maxI32(r.Left, other.Left)
	rt := minI32(r.Right, other.Right)
	t := maxI32(r.Top, other.Top)
	b := minI32(r.Bottom, other.Bottom)
	if l >= rt || t >= b {
		return false
	}
	*r = IRect{Left: l, Top: t, Right: rt, Bottom: b}
	return true
}

// Intersects reports whether r and other overlap.
func (r IRect) Intersects(other IRect) bool {
	tmp := r
	return tmp.Intersect(other)
}

// Join expands r to include other. Empty operands are ignored; joining onto an empty rect adopts the other rect.
func (r *IRect) Join(other IRect) {
	if other.isEmpty64() {
		return
	}
	if r.isEmpty64() {
		*r = other
		return
	}
	r.Left = minI32(r.Left, other.Left)
	r.Top = minI32(r.Top, other.Top)
	r.Right = maxI32(r.Right, other.Right)
	r.Bottom = maxI32(r.Bottom, other.Bottom)
}

// ContainsPoint reports whether (x, y) is inside r, half-open on the right/bottom edges.
func (r IRect) ContainsPoint(x, y int32) bool {
	return x >= r.Left && x < r.Right && y >= r.Top && y < r.Bottom
}

// ContainsRect reports true only when both rects are non-empty and other is entirely inside r.
func (r IRect) ContainsRect(other IRect) bool {
	return !other.isEmpty64() && !r.isEmpty64() &&
		r.Left <= other.Left && r.Top <= other.Top &&
		r.Right >= other.Right && r.Bottom >= other.Bottom
}

// ToRect converts to a float Rect.
func (r IRect) ToRect() Rect {
	return Rect{
		Left:   float32(r.Left),
		Top:    float32(r.Top),
		Right:  float32(r.Right),
		Bottom: float32(r.Bottom),
	}
}

// min32/max32 use an ordering where, when an operand is NaN, the comparison is false and the first operand is returned.
// Go's builtin min/max instead propagate NaN from either operand, which would change the NaN behavior in
// Intersect/Sort, so they must not be substituted here.
func min32(a, b float32) float32 {
	if b < a {
		return b
	}
	return a
}

func max32(a, b float32) float32 {
	if b > a {
		return b
	}
	return a
}

func minI32(a, b int32) int32 {
	if b < a {
		return b
	}
	return a
}

func maxI32(a, b int32) int32 {
	if b > a {
		return b
	}
	return a
}
