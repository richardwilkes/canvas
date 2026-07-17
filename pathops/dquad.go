// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The double-precision quadratic Bezier primitive. Present are the quad/line intersection layer's members (ptAtT, the
// monotonic checks, isLinear, the quadratic-root helpers quadRootsReal/quadRootsValidT/ quadAddValidTs,
// quadFindExtrema) and the curve-vs-curve members the curve-pair intersection solver consumes (hullIntersects,
// otherPts, subDivide in both forms, dxdyAtT, quadSetABC, align, chopAt, collapsed, controlsInside). The generic-curve
// interface and its quad wrapper (thin adapters over these members) live in tcurve.go; the span layer that consumes
// them lives in tspan.go; the solver body itself lands in a later session.

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// dQuad is a quadratic Bezier defined by three double-precision points.
type dQuad struct {
	pts [3]dPoint
}

// set widens the float32 host points into q.
func (q *dQuad) set(pts [3]geom.Point) {
	q.pts[0].set(pts[0])
	q.pts[1].set(pts[1])
	q.pts[2].set(pts[2])
}

// ptAtT evaluates the quad at parameter t using the standard Bernstein basis.
func (q dQuad) ptAtT(t float64) dPoint {
	if t == 0 {
		return q.pts[0]
	}
	if t == 1 {
		return q.pts[2]
	}
	oneT := 1 - t
	a := oneT * oneT
	b := 2 * oneT * t
	c := t * t
	return dPoint{
		x: a*q.pts[0].x + b*q.pts[1].x + c*q.pts[2].x,
		y: a*q.pts[0].y + b*q.pts[1].y + c*q.pts[2].y,
	}
}

// monotonicInX reports whether the quad's x coordinate is monotonic over t in [0,1].
func (q dQuad) monotonicInX() bool { return between(q.pts[0].x, q.pts[1].x, q.pts[2].x) }

// monotonicInY reports whether the quad's y coordinate is monotonic over t in [0,1].
func (q dQuad) monotonicInY() bool { return between(q.pts[0].y, q.pts[1].y, q.pts[2].y) }

// isLinear reports whether the control point lies (within tolerance) on the chord through the two selected endpoints,
// i.e. whether the quad is flat enough to be treated as a line.
func (q dQuad) isLinear(startIndex, endIndex int) bool {
	var lp lineParameters
	lp.quadEndPointsSE(q, startIndex, endIndex)
	// The normalize is load-bearing, not incidental: it makes controlPtDistanceQuad a true distance, which is what
	// approximatelyZeroWhenComparedTo's tolerance is defined against below. Skipping it to elide the sqrt would compare
	// an unnormalized value against the same tolerance, silently changing which quads count as linear.
	lp.normalize()
	distance := lp.controlPtDistanceQuad(q)
	tiniest := math.Min(math.Min(math.Min(math.Min(math.Min(q.pts[0].x, q.pts[0].y),
		q.pts[1].x), q.pts[1].y), q.pts[2].x), q.pts[2].y)
	largest := math.Max(math.Max(math.Max(math.Max(math.Max(q.pts[0].x, q.pts[0].y),
		q.pts[1].x), q.pts[1].y), q.pts[2].x), q.pts[2].y)
	largest = math.Max(largest, -tiniest)
	return approximatelyZeroWhenComparedTo(distance, largest)
}

// handleZeroRoots solves the degenerate A==0 (linear, b*t+c=0) case of the quadratic root finder.
func handleZeroRoots(b, c float64, s []float64) int {
	if approximatelyZero(b) {
		s[0] = 0
		return b2i(c == 0)
	}
	s[0] = -c / b
	return 1
}

// quadRootsReal computes the real roots of A*t^2 + B*t + C, smaller first. Does not discard roots outside [0,1].
func quadRootsReal(a, b, c float64, s []float64) int {
	if a == 0 {
		return handleZeroRoots(b, c, s)
	}
	p := b / (2 * a)
	q := c / a
	if approximatelyZero(a) && (approximatelyZeroInverse(p) || approximatelyZeroInverse(q)) {
		return handleZeroRoots(b, c, s)
	}
	// normal form: x^2 + px + q = 0
	p2 := p * p
	if !almostDequalUlps(p2, q) && p2 < q {
		return 0
	}
	sqrtD := 0.0
	if p2 > q {
		sqrtD = math.Sqrt(p2 - q)
	}
	s[0] = sqrtD - p
	s[1] = -sqrtD - p
	return 1 + b2i(!almostDequalUlps(s[0], s[1]))
}

// quadAddValidTs filters s down to the roots that lie in [0,1] (snapping near-0/near-1 to exact), dropping approximate
// duplicates.
func quadAddValidTs(s []float64, realRoots int, t []float64) int {
	foundRoots := 0
	for index := 0; index < realRoots; index++ {
		tValue := s[index]
		if approximatelyZeroOrMore(tValue) && approximatelyOneOrLess(tValue) {
			if approximatelyLessThanZero(tValue) {
				tValue = 0
			} else if approximatelyGreaterThanOne(tValue) {
				tValue = 1
			}
			dup := false
			for idx2 := 0; idx2 < foundRoots; idx2++ {
				if approximatelyEqual(t[idx2], tValue) {
					dup = true
					break
				}
			}
			if !dup {
				t[foundRoots] = tValue
				foundRoots++
			}
		}
	}
	return foundRoots
}

// quadRootsValidT returns the real roots of A*t^2 + B*t + C that lie within [0,1].
func quadRootsValidT(a, b, c float64, t []float64) int {
	var s [2]float64
	realRoots := quadRootsReal(a, b, c, s[:])
	return quadAddValidTs(s[:], realRoots, t)
}

// validUnitDivide computes numer/denom, flipping the sign of both operands first if numer is negative, and reports 0
// (leaving *ratio untouched) if the result is not a proper fraction in (0,1) — including the underflow case where numer
// is so much smaller than denom that the quotient rounds to exactly 0.
func validUnitDivide(numer, denom float64, ratio *float64) int {
	if numer < 0 {
		numer = -numer
		denom = -denom
	}
	if denom == 0 || numer == 0 || numer >= denom {
		return 0
	}
	r := numer / denom
	if r == 0 { // catch underflow if numer <<<< denom
		return 0
	}
	*ratio = r
	return 1
}

// quadFindExtrema finds the one t (pinned to (0,1)) at which one coordinate of the quad is stationary, given that
// coordinate's three control values a,b,c. Quad'(t) = A*t + B with A = 2(a-2b+c), B = 2(b-a), solved as t = -B/A.
func quadFindExtrema(a, b, c float64, tValue *float64) int {
	return validUnitDivide(a-b, a-b-b+c, tValue)
}

// dQuadPair holds the five points of a quad split into two adjoining sub-quads (the split point is shared, at index 2).
type dQuadPair struct {
	pts [5]dPoint
}

// first returns the sub-quad over points [0..2].
func (p *dQuadPair) first() dQuad { return dQuad{pts: [3]dPoint{p.pts[0], p.pts[1], p.pts[2]}} }

// second returns the sub-quad over points [2..4].
func (p *dQuadPair) second() dQuad { return dQuad{pts: [3]dPoint{p.pts[2], p.pts[3], p.pts[4]}} }

// collapsed reports whether all three control points of q are approximately coincident.
func (q dQuad) collapsed() bool {
	return q.pts[0].approximatelyEqual(q.pts[1]) && q.pts[0].approximatelyEqual(q.pts[2])
}

// controlsInside reports whether the middle control point projects between the two endpoints (both chord dot products
// positive).
func (q dQuad) controlsInside() bool {
	v01 := q.pts[0].sub(q.pts[1])
	v02 := q.pts[0].sub(q.pts[2])
	v12 := q.pts[1].sub(q.pts[2])
	return v02.dot(v01) > 0 && v02.dot(v12) > 0
}

// otherPts returns the two control points other than the one at index oddMan, computed without a branch: the bit
// twiddling picks the off-curve index, and a negative intermediate is clamped to zero by `end &= ^(end >> 2)` (Go's
// signed >> is arithmetic, so this works the same way it does in C).
func (q dQuad) otherPts(oddMan int) [2]dPoint {
	var endPt [2]dPoint
	for opp := 1; opp < 3; opp++ {
		end := (oddMan ^ opp) - oddMan // choose a value not equal to oddMan
		end &= ^(end >> 2)             // if the value went negative, set it to zero
		endPt[opp-1] = q.pts[end]
	}
	return endPt
}

// pointInTriangle reports whether test lies strictly inside the triangle formed by the three points, via barycentric
// coordinates (see blackpawn.com/texts/pointinpoly).
func pointInTriangle(pts [3]dPoint, test dPoint) bool {
	v0 := pts[2].sub(pts[0])
	v1 := pts[1].sub(pts[0])
	v2 := test.sub(pts[0])
	dot00 := v0.dot(v0)
	dot01 := v0.dot(v1)
	dot02 := v0.dot(v2)
	dot11 := v1.dot(v1)
	dot12 := v1.dot(v2)
	// Compute barycentric coordinates.
	denom := dot00*dot11 - dot01*dot01
	u := dot11*dot02 - dot01*dot12
	v := dot00*dot12 - dot01*dot02
	// Check if the point is in the triangle.
	if denom >= 0 {
		return u >= 0 && v >= 0 && u+v < denom
	}
	return u <= 0 && v <= 0 && u+v > denom
}

// matchesEnd reports whether test exactly equals either endpoint of the quad.
func matchesEnd(pts [3]dPoint, test dPoint) bool {
	return pts[0].equals(test) || pts[2].equals(test)
}

// hullIntersects is a quick reject that rotates all points onto a line through each pair of this quad's endpoints; if
// q2's points all fall on the line or on the opposite side from this quad's odd man, the curves at most touch at
// endpoints. Returns whether the hulls may intersect beyond the endpoints; the returned isLinear is meaningful only
// when result is true (an early false return leaves it at its zero value) and reports whether this quad's hull
// collapsed to a line.
func (q dQuad) hullIntersects(q2 dQuad) (result, isLinear bool) {
	linear := true
	for oddMan := 0; oddMan < 3; oddMan++ {
		endPt := q.otherPts(oddMan)
		origX := endPt[0].x
		origY := endPt[0].y
		adj := endPt[1].x - origX
		opp := endPt[1].y - origY
		sign := (q.pts[oddMan].y-origY)*adj - (q.pts[oddMan].x-origX)*opp
		if approximatelyZero(sign) {
			continue
		}
		linear = false
		foundOutlier := false
		for n := 0; n < 3; n++ {
			test := (q2.pts[n].y-origY)*adj - (q2.pts[n].x-origX)*opp
			if test*sign > 0 && !preciselyZero(test) {
				foundOutlier = true
				break
			}
		}
		if !foundOutlier {
			return false, linear
		}
	}
	if linear && !matchesEnd(q.pts, q2.pts[0]) && !matchesEnd(q.pts, q2.pts[2]) {
		// If the end point of the opposite quad is inside the hull that is nearly a line, then representing the quad as
		// a line may cause the intersection to be missed. Check to see if the endpoint is in the triangle.
		if pointInTriangle(q.pts, q2.pts[0]) || pointInTriangle(q.pts, q2.pts[2]) {
			linear = false
		}
	}
	return true, linear
}

// hullIntersectsConic dispatches hull-intersection testing to the conic overload (conic.hullIntersectsQuad).
func (q dQuad) hullIntersectsConic(conic dConic) (sects, isLinear bool) {
	return conic.hullIntersectsQuad(q)
}

// hullIntersectsCubic dispatches hull-intersection testing to the cubic overload (cubic.hullIntersectsQuad).
func (q dQuad) hullIntersectsCubic(cubic dCubic) (sects, isLinear bool) {
	return cubic.hullIntersectsQuad(q)
}

// dxdyAtT returns the (unnormalized) tangent vector at t, equal to Quad'(t)/2.
func (q dQuad) dxdyAtT(t float64) dVector {
	a := t - 1
	b := 1 - 2*t
	c := t
	result := dVector{
		x: a*q.pts[0].x + b*q.pts[1].x + c*q.pts[2].x,
		y: a*q.pts[0].y + b*q.pts[1].y + c*q.pts[2].y,
	}
	if result.x == 0 && result.y == 0 {
		if zeroOrOne(t) {
			result = q.pts[2].sub(q.pts[0])
		}
		// else: an interior zero tangent (a cusp) has no defined fallback direction; result stays {0,0}.
	}
	return result
}

// interpQuadCoords evaluates one coordinate of the quad at t via De Casteljau's algorithm, given that coordinate's
// three control-point values.
func interpQuadCoords(p0, p1, p2, t float64) float64 {
	if t == 0 {
		return p0
	}
	if t == 1 {
		return p2
	}
	ab := dInterp(p0, p1, t)
	bc := dInterp(p1, p2, t)
	return dInterp(ab, bc, t)
}

// subDivide returns the sub-quad over the T range [t1, t2]. The endpoints are evaluated directly; the middle control
// point is recovered from the value at the range midpoint via B = 2*D - (A + C)/2.
func (q dQuad) subDivide(t1, t2 float64) dQuad {
	if t1 == 0 && t2 == 1 {
		return q
	}
	var dst dQuad
	ax := interpQuadCoords(q.pts[0].x, q.pts[1].x, q.pts[2].x, t1)
	ay := interpQuadCoords(q.pts[0].y, q.pts[1].y, q.pts[2].y, t1)
	dx := interpQuadCoords(q.pts[0].x, q.pts[1].x, q.pts[2].x, (t1+t2)/2)
	dy := interpQuadCoords(q.pts[0].y, q.pts[1].y, q.pts[2].y, (t1+t2)/2)
	cx := interpQuadCoords(q.pts[0].x, q.pts[1].x, q.pts[2].x, t2)
	cy := interpQuadCoords(q.pts[0].y, q.pts[1].y, q.pts[2].y, t2)
	dst.pts[0].x = ax
	dst.pts[0].y = ay
	dst.pts[2].x = cx
	dst.pts[2].y = cy
	dst.pts[1].x = 2*dx - (ax+cx)/2
	dst.pts[1].y = 2*dy - (ay+cy)/2
	return dst
}

// align snaps dstPt's coordinate to the endpoint's when the endpoint and the middle control point share that coordinate
// exactly.
func (q dQuad) align(endIndex int, dstPt *dPoint) {
	if q.pts[endIndex].x == q.pts[1].x {
		dstPt.x = q.pts[endIndex].x
	}
	if q.pts[endIndex].y == q.pts[1].y {
		dstPt.y = q.pts[endIndex].y
	}
}

// subDivideCtrl finds the middle control point of the sub-quad with the given desired endpoints a and c, by
// intersecting the tangent rays at a and c. Requires t1 != t2.
func (q dQuad) subDivideCtrl(a, c dPoint, t1, t2 float64) dPoint {
	sub := q.subDivide(t1, t2)
	b0 := dLine{pts: [2]dPoint{a, {x: sub.pts[1].x + (a.x - sub.pts[0].x), y: sub.pts[1].y + (a.y - sub.pts[0].y)}}}
	b1 := dLine{pts: [2]dPoint{c, {x: sub.pts[1].x + (c.x - sub.pts[2].x), y: sub.pts[1].y + (c.y - sub.pts[2].y)}}}
	i := newIntersections()
	i.intersectRayLine(b0, b1)
	var b dPoint
	if i.used == 1 && i.ts[0][0] >= 0 && i.ts[1][0] >= 0 {
		b = i.pt(0)
	} else {
		return dPointMid(b0.pts[1], b1.pts[1])
	}
	if t1 == 0 || t2 == 0 {
		q.align(0, &b)
	}
	if t1 == 1 || t2 == 1 {
		q.align(2, &b)
	}
	if almostBequalUlps(b.x, a.x) {
		b.x = a.x
	} else if almostBequalUlps(b.x, c.x) {
		b.x = c.x
	}
	if almostBequalUlps(b.y, a.y) {
		b.y = a.y
	} else if almostBequalUlps(b.y, c.y) {
		b.y = c.y
	}
	return b
}

// interpQuadCoordsChop is the classic one-t subdivision of one coordinate, producing the five coordinate values of the
// two resulting sub-quads (which share d4, the split point).
func interpQuadCoordsChop(p0, p1, p2, t float64) (d0, d2, d4, d6, d8 float64) {
	ab := dInterp(p0, p1, t)
	bc := dInterp(p1, p2, t)
	return p0, ab, dInterp(ab, bc, t), bc, p2
}

// chopAt splits the quad at t into two adjoining sub-quads.
func (q dQuad) chopAt(t float64) dQuadPair {
	var dst dQuadPair
	dst.pts[0].x, dst.pts[1].x, dst.pts[2].x, dst.pts[3].x, dst.pts[4].x = interpQuadCoordsChop(q.pts[0].x, q.pts[1].x, q.pts[2].x, t)
	dst.pts[0].y, dst.pts[1].y, dst.pts[2].y, dst.pts[3].y, dst.pts[4].y = interpQuadCoordsChop(q.pts[0].y, q.pts[1].y, q.pts[2].y, t)
	return dst
}

// quadSetABC converts the three control values of one coordinate (the A,B,C of A*t*t + 2*B*t*(1-t) + C*(1-t)*(1-t))
// into the plain-polynomial coefficients a,b,c so that a*t*t + b*t + c evaluates the same coordinate.
func quadSetABC(p0, p1, p2 float64) (a, b, c float64) {
	a = p0     // a = A
	b = 2 * p1 // b =     2*B
	c = p2     // c =             C
	b -= c     // b =     2*B -   C
	a -= b     // a = A - 2*B +   C
	b -= c     // b =     2*B - 2*C
	return a, b, c
}
