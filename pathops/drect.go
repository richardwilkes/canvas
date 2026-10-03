// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A double-precision axis-aligned bounding box, with quad, conic, and cubic bounds built from each curve's extrema.

package pathops

import "math"

type dRect struct {
	left, top, right, bottom float64
}

func (r *dRect) set(pt dPoint) {
	r.left = pt.x
	r.right = pt.x
	r.top = pt.y
	r.bottom = pt.y
}

func (r *dRect) add(pt dPoint) {
	r.left = math.Min(r.left, pt.x)
	r.top = math.Min(r.top, pt.y)
	r.right = math.Max(r.right, pt.x)
	r.bottom = math.Max(r.bottom, pt.y)
}

func (r dRect) width() float64 { return r.right - r.left }

func (r dRect) height() float64 { return r.bottom - r.top }

func (r dRect) valid() bool { return r.left <= r.right && r.top <= r.bottom }

// intersects reports whether the two (sorted) rectangles overlap.
func (r dRect) intersects(o dRect) bool {
	return o.left <= r.right && r.left <= o.right && o.top <= r.bottom && r.top <= o.bottom
}

func (r *dRect) setBoundsQuadFull(curve dQuad) {
	r.setBoundsQuad(curve, curve, 0, 1)
}

// setBoundsQuad sets r to the bounds of sub, the part of curve spanning the T range [startT, endT]. Extrema found on
// sub are mapped back to T values on curve so the points added lie exactly on the parent.
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

func (r *dRect) setBoundsConicFull(curve dConic) {
	r.setBoundsConic(curve, curve, 0, 1)
}

// setBoundsConic is setBoundsQuad for conics.
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

func (r *dRect) setBoundsCubicFull(curve dCubic) {
	r.setBoundsCubic(curve, curve, 0, 1)
}

// setBoundsCubic is setBoundsQuad for cubics.
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
