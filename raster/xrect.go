// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// XRect: a rectangle whose coordinates are 16.16 Fixed values. It gets its own type (rather than reusing IRect) so
// Fixed and integer rects cannot be confused.

package raster

import "github.com/richardwilkes/canvas/geom"

// XRect is an IRect-shaped rectangle holding Fixed (16.16) coordinates.
type XRect struct {
	Left   Fixed
	Top    Fixed
	Right  Fixed
	Bottom Fixed
}

// XRectFromIRect promotes integer coordinates to Fixed. Does not check for overflow if the coordinates exceed 32K.
func XRectFromIRect(r geom.IRect) XRect {
	return XRect{
		Left:   Fixed(r.Left) << 16,
		Top:    Fixed(r.Top) << 16,
		Right:  Fixed(r.Right) << 16,
		Bottom: Fixed(r.Bottom) << 16,
	}
}

// XRectFromRect promotes scalar coordinates to Fixed. Does not check for overflow if the coordinates exceed 32K.
func XRectFromRect(r geom.Rect) XRect {
	return XRect{
		Left:   FloatToFixed(r.Left),
		Top:    FloatToFixed(r.Top),
		Right:  FloatToFixed(r.Right),
		Bottom: FloatToFixed(r.Bottom),
	}
}

// IsEmpty reports whether the rect has zero or negative width or height.
func (x XRect) IsEmpty() bool {
	return x.Left >= x.Right || x.Top >= x.Bottom
}

// Round rounds each Fixed coordinate to an integer.
func (x XRect) Round() geom.IRect {
	return geom.IRectLTRB(FixedRoundToInt(x.Left), FixedRoundToInt(x.Top),
		FixedRoundToInt(x.Right), FixedRoundToInt(x.Bottom))
}

// RoundOut floors left/top and ceils right/bottom.
func (x XRect) RoundOut() geom.IRect {
	return geom.IRectLTRB(FixedFloorToInt(x.Left), FixedFloorToInt(x.Top),
		FixedCeilToInt(x.Right), FixedCeilToInt(x.Bottom))
}

// Intersect sets x to the intersection when the two rects overlap, else x is unchanged and the result is false.
func (x *XRect) Intersect(other XRect) bool {
	l := max(x.Left, other.Left)
	t := max(x.Top, other.Top)
	r := min(x.Right, other.Right)
	b := min(x.Bottom, other.Bottom)
	if l >= r || t >= b {
		return false
	}
	x.Left, x.Top, x.Right, x.Bottom = l, t, r, b
	return true
}
