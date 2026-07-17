// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A double-precision axis-aligned bounding box. The quad, conic, and cubic bounds lanes are present (set/add plus
// setBoundsQuad/setBoundsConic/setBoundsCubic, which use each curve's extrema).

package pathops

import "math"

// dRect is a double-precision bounding box.
type dRect struct {
	left, top, right, bottom float64
}

// set collapses r to a single point.
func (r *dRect) set(pt dPoint) {
	r.left = pt.x
	r.right = pt.x
	r.top = pt.y
	r.bottom = pt.y
}

// add grows r to include pt.
func (r *dRect) add(pt dPoint) {
	r.left = math.Min(r.left, pt.x)
	r.top = math.Min(r.top, pt.y)
	r.right = math.Max(r.right, pt.x)
	r.bottom = math.Max(r.bottom, pt.y)
}

// width returns the horizontal extent of r.
func (r dRect) width() float64 { return r.right - r.left }

// height returns the vertical extent of r.
func (r dRect) height() float64 { return r.bottom - r.top }

// valid reports whether r's edges are properly ordered (left <= right, top <= bottom).
func (r dRect) valid() bool { return r.left <= r.right && r.top <= r.bottom }

// intersects reports whether the two (sorted) rectangles overlap.
func (r dRect) intersects(o dRect) bool {
	return o.left <= r.right && r.left <= o.right && o.top <= r.bottom && r.top <= o.bottom
}

// setBoundsQuadFull sets r to the bounds of the whole curve.
func (r *dRect) setBoundsQuadFull(curve dQuad) {
	r.setBoundsQuad(curve, curve, 0, 1)
}

// setBoundsQuad sets r to the bounds of the sub-curve sub, mapping its extrema T values back onto the parent curve
// (spanning the T range [startT, endT]) to accumulate points on the parent's exact curve.
func (r *dRect) setBoundsQuad(curve, sub dQuad, startT, endT float64) {
	r.set(sub.pts[0])
	r.add(sub.pts[2])
	var tValues [2]float64
	roots := 0
	if !sub.monotonicInX() {
		roots = quadFindExtrema(sub.pts[0].x, sub.pts[1].x, sub.pts[2].x, &tValues[0])
	}
	if !sub.monotonicInY() {
		roots += quadFindExtrema(sub.pts[0].y, sub.pts[1].y, sub.pts[2].y, &tValues[roots])
	}
	for index := 0; index < roots; index++ {
		t := startT + (endT-startT)*tValues[index]
		r.add(curve.ptAtT(t))
	}
}

// setBoundsConicFull sets r to the bounds of the whole curve.
func (r *dRect) setBoundsConicFull(curve dConic) {
	r.setBoundsConic(curve, curve, 0, 1)
}

// setBoundsConic sets r to the bounds of the sub-conic sub, mapping its extrema T values back onto the parent conic
// (spanning the T range [startT, endT]) to accumulate points on the parent's exact curve.
func (r *dRect) setBoundsConic(curve, sub dConic, startT, endT float64) {
	r.set(sub.pts.pts[0])
	r.add(sub.pts.pts[2])
	var tValues [2]float64
	roots := 0
	if !sub.monotonicInX() {
		roots = conicFindExtrema(sub.pts.pts[0].x, sub.pts.pts[1].x, sub.pts.pts[2].x, sub.weight, &tValues[0])
	}
	if !sub.monotonicInY() {
		roots += conicFindExtrema(sub.pts.pts[0].y, sub.pts.pts[1].y, sub.pts.pts[2].y, sub.weight, &tValues[roots])
	}
	for index := 0; index < roots; index++ {
		t := startT + (endT-startT)*tValues[index]
		r.add(curve.ptAtT(t))
	}
}

// setBoundsCubicFull sets r to the bounds of the whole curve.
func (r *dRect) setBoundsCubicFull(curve dCubic) {
	r.setBoundsCubic(curve, curve, 0, 1)
}

// setBoundsCubic sets r to the bounds of the sub-curve sub, mapping its extrema T values back onto the parent curve
// (spanning the T range [startT, endT]) to accumulate points on the parent's exact curve.
func (r *dRect) setBoundsCubic(curve, sub dCubic, startT, endT float64) {
	r.set(sub.pts[0])
	r.add(sub.pts[3])
	var tValues [4]float64
	roots := 0
	if !sub.monotonicInX() {
		roots = cubicFindExtrema(sub.pts[0].x, sub.pts[1].x, sub.pts[2].x, sub.pts[3].x, tValues[:])
	}
	if !sub.monotonicInY() {
		roots += cubicFindExtrema(sub.pts[0].y, sub.pts[1].y, sub.pts[2].y, sub.pts[3].y, tValues[roots:])
	}
	for index := 0; index < roots; index++ {
		t := startT + (endT-startT)*tValues[index]
		r.add(curve.ptAtT(t))
	}
}
