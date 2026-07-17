// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

// RRectType classifies a rounded rect that uses uniform radii on all four corners (the complex nine-patch types are not
// represented here).
type RRectType uint8

const (
	// RRectEmpty is zero width or height.
	RRectEmpty RRectType = iota
	// RRectRect is non-zero width and height, zero radii.
	RRectRect
	// RRectOval has radii that are half the dimensions on both axes.
	RRectOval
	// RRectSimple is non-zero width, height and uniform radii.
	RRectSimple
)

// RRect is a rounded rect using a single (rx, ry) radius pair applied uniformly to all four corners.
type RRect struct {
	Rect    Rect
	RadiusX float32
	RadiusY float32
	Type    RRectType
}

// MakeRRect builds an RRect with uniform radii: the rect is sorted, non-finite inputs collapse to zero radii, oversized
// radii are scaled down proportionally, and the type is classified.
func MakeRRect(rect Rect, rx, ry float32) RRect {
	var rr RRect
	if !rr.initializeRect(rect) {
		return rr
	}
	if !IsFinite(rx, ry) {
		rx = 0
		ry = 0
	}
	if rr.Rect.Width() < rx+rx || rr.Rect.Height() < ry+ry {
		// Proportionally scale down all radii to fit, matching setRectXY.
		scale := min32(IEEEFloatDivide(rr.Rect.Width(), rx+rx), IEEEFloatDivide(rr.Rect.Height(), ry+ry))
		rx *= scale
		ry *= scale
	}
	if rx <= 0 || ry <= 0 {
		rr.Type = RRectRect
		return rr
	}
	rr.RadiusX = rx
	rr.RadiusY = ry
	rr.Type = RRectSimple
	if rx >= rr.Rect.Width()*0.5 && ry >= rr.Rect.Height()*0.5 {
		rr.Type = RRectOval
	}
	return rr
}

// initializeRect sorts the rect, collapsing non-finite rects to empty. Returns false when the result is empty.
func (rr *RRect) initializeRect(rect Rect) bool {
	if !rect.IsFinite() {
		*rr = RRect{}
		return false
	}
	rr.Rect = rect.Sorted()
	if rr.Rect.IsEmpty() {
		*rr = RRect{Rect: rr.Rect}
		return false
	}
	return true
}

// IsEmpty reports whether the rrect is empty.
func (rr RRect) IsEmpty() bool {
	return rr.Type == RRectEmpty
}

// Transform maps the rrect through m, returning false unless the matrix preserves axis alignment. The rect maps through
// the matrix and the radii are deduced from the mapped corner-conic control points (the per-corner spread this uniform
// storage cannot represent is discarded).
func (rr RRect) Transform(m *Matrix) (RRect, bool) {
	if m.IsIdentity() {
		return rr, true
	}
	// preservesAxisAlignment == rectStaysRect
	if !m.RectStaysRect() {
		return RRect{}, false
	}
	newRect, _ := m.MapRect(rr.Rect)
	if !newRect.IsFinite() {
		return RRect{}, false
	}
	switch rr.Type {
	case RRectEmpty:
		return RRect{}, true
	case RRectRect:
		return MakeRRect(newRect, 0, 0), true
	case RRectOval:
		return MakeRRect(newRect, newRect.Width()*0.5, newRect.Height()*0.5), true
	default:
	}
	// Map the upper-left corner conic's control points and deduce (rx, ry) from the deltas; the same deduction applies
	// to each corner.
	pts := [3]Point{
		{X: rr.Rect.Left, Y: rr.Rect.Top + rr.RadiusY},
		{X: rr.Rect.Left, Y: rr.Rect.Top},
		{X: rr.Rect.Left + rr.RadiusX, Y: rr.Rect.Top},
	}
	m.MapPointsInPlace(pts[:])
	v10 := pts[1].Sub(pts[0])
	v21 := pts[2].Sub(pts[1])
	var rx, ry float32
	switch {
	case v10.X != 0:
		rx = ScalarAbs(v10.X)
		ry = ScalarAbs(v21.Y)
	case v10.Y == 0:
		rx = ScalarAbs(v21.X)
		ry = ScalarAbs(v21.Y)
	default:
		rx = ScalarAbs(v21.X)
		ry = ScalarAbs(v10.Y)
	}
	return MakeRRect(newRect, rx, ry), true
}
