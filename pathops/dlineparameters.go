// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A parameterized line ax + by + c = 0 whose signed evaluation (pointDistance) gives the (optionally normalized)
// distance from a point to the line. quadEndPointsSE/lineEndPoints/cubicEndPointsSE build the line through a chosen
// pair of curve or segment endpoints; normalize scales it to a unit normal; the control-point distance helpers measure
// how far a curve's off-curve points sit from that line (used by dQuad.isLinear and dCubic.isLinear to decide whether a
// curve is flat enough to treat as a line). The bool-returning quadEndPoints/cubicEndPoints/cubicPart forms
// additionally pick a non-degenerate endpoint pair and break cw/ccw ties; dx/dy/pointDistance are consumed by
// opAngle.setSpans (opangle.go).

package pathops

import "math"

// lineParameters holds the implicit line coefficients (a, b, c) of ax + by + c = 0.
type lineParameters struct {
	a, b, c float64
}

// quadEndPointsSE sets the receiver to the line through quad points s and e.
func (lp *lineParameters) quadEndPointsSE(pts dQuad, s, e int) {
	lp.a = pts.pts[s].y - pts.pts[e].y
	lp.b = pts.pts[e].x - pts.pts[s].x
	lp.c = pts.pts[s].x*pts.pts[e].y - pts.pts[e].x*pts.pts[s].y
}

// normalSquared returns a^2 + b^2, the squared length of the line's normal vector.
func (lp *lineParameters) normalSquared() float64 {
	return lp.a*lp.a + lp.b*lp.b
}

// normalize scales the coefficients so a^2 + b^2 == 1, making pointDistance return an actual Euclidean distance.
// Returns false (zeroing the coefficients) when the normal is approximately zero.
func (lp *lineParameters) normalize() bool {
	normal := math.Sqrt(lp.normalSquared())
	if approximatelyZero(normal) {
		lp.a, lp.b, lp.c = 0, 0, 0
		return false
	}
	reciprocal := 1 / normal
	lp.a *= reciprocal
	lp.b *= reciprocal
	lp.c *= reciprocal
	return true
}

// controlPtDistanceQuad returns the signed distance from the quad's control point (index 1) to the line.
func (lp *lineParameters) controlPtDistanceQuad(pts dQuad) float64 {
	return lp.a*pts.pts[1].x + lp.b*pts.pts[1].y + lp.c
}

// lineEndPoints sets the receiver to the line through the two segment points.
func (lp *lineParameters) lineEndPoints(pts dLine) {
	lp.a = pts.pts[0].y - pts.pts[1].y
	lp.b = pts.pts[1].x - pts.pts[0].x
	lp.c = pts.pts[0].x*pts.pts[1].y - pts.pts[1].x*pts.pts[0].y
}

// quadEndPoints sets the receiver to the line through quad points 0 and 1, falling back to the (0,2) line when that
// tangent is degenerate, and biasing the a coefficient on a horizontal cw/ccw tie. Returns false only when the fallback
// (0,2) line is used.
func (lp *lineParameters) quadEndPoints(pts dQuad) bool {
	lp.quadEndPointsSE(pts, 0, 1)
	if lp.dy() != 0 {
		return true
	}
	if lp.dx() == 0 {
		lp.quadEndPointsSE(pts, 0, 2)
		return false
	}
	if lp.dx() < 0 { // only worry about y bias when breaking cw/ccw tie
		return true
	}
	// The tangent is horizontal, so the cw/ccw comparison is a tie. Bias a to slightly negative (dy returns -a) when
	// the quad's far end is above its start, so the sort has a deterministic answer rather than an arbitrary one.
	if pts.pts[0].y > pts.pts[2].y {
		lp.a = dblEpsilon
	}
	return true
}

// cubicEndPoints sets the receiver to the line through cubic point 0 and the first non-degenerate later point, with a
// horizontal-tangent tie-break bias on the a coefficient. Returns false only when the cubic degenerates to the (0,3)
// line.
func (lp *lineParameters) cubicEndPoints(pts dCubic) bool {
	endIndex := 1
	lp.cubicEndPointsSE(pts, 0, endIndex)
	if lp.dy() != 0 {
		return true
	}
	if lp.dx() == 0 {
		endIndex++
		lp.cubicEndPointsSE(pts, 0, endIndex) // endIndex == 2
		if lp.dy() != 0 {
			return true
		}
		if lp.dx() == 0 {
			endIndex++
			lp.cubicEndPointsSE(pts, 0, endIndex) // line, endIndex == 3
			return false
		}
	}
	// The tangent is horizontal, so the cw/ccw comparison below is a tie; bumping the a coefficient breaks it
	// deterministically. See quadEndPoints for the same bias on quads.
	if lp.dx() < 0 { // only worry about y bias when breaking cw/ccw tie
		return true
	}
	// if cubic tangent is on x axis, look at next control point to break tie; the control point may be approximate, so
	// it must move significantly to account for error
	endIndex++
	if notAlmostEqualUlps(pts.pts[0].y, pts.pts[endIndex].y) {
		if pts.pts[0].y > pts.pts[endIndex].y {
			lp.a = dblEpsilon // push it from 0 to slightly negative (dy returns -a)
		}
		return true
	}
	if endIndex == 3 {
		return true
	}
	// endIndex == 2
	if pts.pts[0].y > pts.pts[3].y {
		lp.a = dblEpsilon // push it from 0 to slightly negative (dy returns -a)
	}
	return true
}

// cubicPart returns the signed distance from the far control point of a cubic section to its endpoint line, choosing
// point 3 when points 0 and 1 coincide (or point 2 lies on the 0-1 ray).
func (lp *lineParameters) cubicPart(part dCubic) float64 {
	lp.cubicEndPoints(part)
	if part.pts[0].equals(part.pts[1]) ||
		(dLine{pts: [2]dPoint{part.pts[0], part.pts[1]}}).nearRay(part.pts[2]) {
		return lp.pointDistance(part.pts[3])
	}
	return lp.pointDistance(part.pts[2])
}

// pointDistance returns the signed distance from pt to the line (unnormalized unless normalize() was called first).
func (lp *lineParameters) pointDistance(pt dPoint) float64 {
	return lp.a*pt.x + lp.b*pt.y + lp.c
}

// dx returns the line's x tangent component.
func (lp *lineParameters) dx() float64 { return lp.b }

// dy returns the line's y tangent component.
func (lp *lineParameters) dy() float64 { return -lp.a }

// cubicEndPointsSE sets the receiver to the line through cubic points s and e.
func (lp *lineParameters) cubicEndPointsSE(pts dCubic, s, e int) {
	lp.a = pts.pts[s].y - pts.pts[e].y
	lp.b = pts.pts[e].x - pts.pts[s].x
	lp.c = pts.pts[s].x*pts.pts[e].y - pts.pts[e].x*pts.pts[s].y
}

// controlPtDistanceCubic returns the signed distance from the cubic's control point (index 1 or 2) to the line.
func (lp *lineParameters) controlPtDistanceCubic(pts dCubic, index int) float64 {
	return lp.a*pts.pts[index].x + lp.b*pts.pts[index].y + lp.c
}
