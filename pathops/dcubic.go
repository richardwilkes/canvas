// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The double-precision cubic Bezier primitive. Present are the members reachable from the cubic/line intersection
// layer, the order reducer, and the cubic bounds (set/debugSet, ptAtT, the monotonic checks, isLinear, and the root
// machinery cubicCoefficients/cubicRootsReal/cubicRootsValidT/cubicFindExtrema/ findInflections plus the
// searchRoots/binarySearch used when a solved root does not land on the axis) and the curve-vs-curve members the cubic
// solver consumes (hullIntersects in all three forms over convexHull, otherPts, dxdyAtT, chopAt, subDivide, collapsed,
// controlsInside), the edge-builder members (findMaxCurvature, calcPrecision, toFloatPoints, complexBreak), and the
// opSegment members (align, subDivideAD). The convex-hull machinery lives in dcubichull.go.

package pathops

import (
	"math"
	"sort"

	"github.com/richardwilkes/canvas/geom"
)

// skDoublePI is pi at double precision, used by the cubic root solver's trigonometric branch.
const skDoublePI = 3.14159265358979323846264338327950288

// searchAxis selects which coordinate binarySearch/searchRoots solves against.
type searchAxis int

const (
	xAxis searchAxis = iota
	yAxis
)

// dCubic is a cubic Bezier defined by four double-precision points.
type dCubic struct {
	pts [4]dPoint
}

// axis returns the point's x (xAxis) or y (yAxis) coordinate, selected by index rather than by field name.
func (p dPoint) axis(a searchAxis) float64 {
	if a == xAxis {
		return p.x
	}
	return p.y
}

// set widens the float32 host points to double precision.
func (c *dCubic) set(pts [4]geom.Point) {
	c.pts[0].set(pts[0])
	c.pts[1].set(pts[1])
	c.pts[2].set(pts[2])
	c.pts[3].set(pts[3])
}

// debugSet stores the double-precision points directly (no float32 rounding), as used by the test corpus.
func (c *dCubic) debugSet(pts [4]dPoint) { c.pts = pts }

// ptAtT evaluates the cubic at parameter t using the Bernstein-basis (De Casteljau) form.
func (c dCubic) ptAtT(t float64) dPoint {
	if t == 0 {
		return c.pts[0]
	}
	if t == 1 {
		return c.pts[3]
	}
	oneT := 1 - t
	oneT2 := oneT * oneT
	a := oneT2 * oneT
	b := 3 * oneT2 * t
	t2 := t * t
	cc := 3 * oneT * t2
	d := t2 * t
	return dPoint{
		x: a*c.pts[0].x + b*c.pts[1].x + cc*c.pts[2].x + d*c.pts[3].x,
		y: a*c.pts[0].y + b*c.pts[1].y + cc*c.pts[2].y + d*c.pts[3].y,
	}
}

// monotonicInX reports whether the cubic's x-coordinate is monotonic along t (both control points' x values lie between
// the endpoints').
func (c dCubic) monotonicInX() bool {
	return preciselyBetween(c.pts[0].x, c.pts[1].x, c.pts[3].x) &&
		preciselyBetween(c.pts[0].x, c.pts[2].x, c.pts[3].x)
}

// monotonicInY reports whether the cubic's y-coordinate is monotonic along t (both control points' y values lie between
// the endpoints').
func (c dCubic) monotonicInY() bool {
	return preciselyBetween(c.pts[0].y, c.pts[1].y, c.pts[3].y) &&
		preciselyBetween(c.pts[0].y, c.pts[2].y, c.pts[3].y)
}

// isLinear reports whether both control points lie (within tolerance) on the chord through the two selected endpoints.
// When the endpoints are approximately equal it degrades to the quad test over the first three points.
func (c dCubic) isLinear(startIndex, endIndex int) bool {
	if c.pts[0].approximatelyDEqual(c.pts[3]) {
		q := dQuad{pts: [3]dPoint{c.pts[0], c.pts[1], c.pts[2]}}
		return q.isLinear(0, 2)
	}
	var lineParameters lineParameters
	lineParameters.cubicEndPointsSE(c, startIndex, endIndex)
	// The normalize is load-bearing, not incidental: it makes controlPtDistanceCubic a true distance, which is what
	// approximatelyZeroWhenComparedTo's tolerance is defined against below. Skipping it to elide the sqrt would compare
	// an unnormalized value against the same tolerance, silently changing which cubics count as linear.
	lineParameters.normalize()
	tiniest := math.Min(math.Min(math.Min(math.Min(math.Min(math.Min(math.Min(c.pts[0].x, c.pts[0].y),
		c.pts[1].x), c.pts[1].y), c.pts[2].x), c.pts[2].y), c.pts[3].x), c.pts[3].y)
	largest := math.Max(math.Max(math.Max(math.Max(math.Max(math.Max(math.Max(c.pts[0].x, c.pts[0].y),
		c.pts[1].x), c.pts[1].y), c.pts[2].x), c.pts[2].y), c.pts[3].x), c.pts[3].y)
	largest = math.Max(largest, -tiniest)
	distance := lineParameters.controlPtDistanceCubic(c, 1)
	if !approximatelyZeroWhenComparedTo(distance, largest) {
		return false
	}
	distance = lineParameters.controlPtDistanceCubic(c, 2)
	return approximatelyZeroWhenComparedTo(distance, largest)
}

// cubicCoefficients converts one coordinate's four Bezier control values (p0..p3) into the cubic polynomial
// coefficients A*t^3 + B*t^2 + C*t + D.
func cubicCoefficients(p0, p1, p2, p3 float64) (a, b, c, d float64) {
	a = p3         // d
	b = p2 * 3     // 3*c
	c = p1 * 3     // 3*b
	d = p0         // a
	a -= d - c + b // A =   -a + 3*b - 3*c + d
	b += 3*d - 2*c // B =  3*a - 6*b + 3*c
	c -= 3 * d     // C = -3*a + 3*b
	return a, b, c, d
}

// cubicRootsReal finds the real roots of A*t^3 + B*t^2 + C*t + D, not filtered to [0,1]. Degenerate leading/trailing
// coefficients drop to the quadratic solver; the general case uses the standard trigonometric/Cardano recipe for
// depressed cubics.
func cubicRootsReal(a, b, c, d float64, s []float64) int {
	if approximatelyZero(a) &&
		approximatelyZeroWhenComparedTo(a, b) &&
		approximatelyZeroWhenComparedTo(a, c) &&
		approximatelyZeroWhenComparedTo(a, d) { // we're just a quadratic
		return quadRootsReal(b, c, d, s)
	}
	if approximatelyZeroWhenComparedTo(d, a) &&
		approximatelyZeroWhenComparedTo(d, b) &&
		approximatelyZeroWhenComparedTo(d, c) { // 0 is one root
		num := quadRootsReal(a, b, c, s)
		for i := 0; i < num; i++ {
			if approximatelyZero(s[i]) {
				return num
			}
		}
		s[num] = 0
		return num + 1
	}
	if approximatelyZero(a + b + c + d) { // 1 is one root
		num := quadRootsReal(a, a+b, -d, s)
		for i := 0; i < num; i++ {
			if almostDequalUlps(s[i], 1) {
				return num
			}
		}
		s[num] = 1
		return num + 1
	}
	var aa, bb, cc float64
	{
		invA := 1 / a
		aa = b * invA
		bb = c * invA
		cc = d * invA
	}
	a2 := aa * aa
	q := (a2 - bb*3) / 9
	r := (2*a2*aa - 9*aa*bb + 27*cc) / 54
	r2 := r * r
	q3 := q * q * q
	r2MinusQ3 := r2 - q3
	adiv3 := aa / 3
	var root float64
	rootsIdx := 0
	if r2MinusQ3 < 0 { // we have 3 real roots
		// the divide/root can, due to finite precision, be slightly outside of -1...1
		theta := math.Acos(math.Max(-1, math.Min(r/math.Sqrt(q3), 1)))
		neg2RootQ := -2 * math.Sqrt(q)

		root = neg2RootQ*math.Cos(theta/3) - adiv3
		s[rootsIdx] = root
		rootsIdx++

		root = neg2RootQ*math.Cos((theta+2*skDoublePI)/3) - adiv3
		if !almostDequalUlps(s[0], root) {
			s[rootsIdx] = root
			rootsIdx++
		}
		root = neg2RootQ*math.Cos((theta-2*skDoublePI)/3) - adiv3
		if !almostDequalUlps(s[0], root) && (rootsIdx == 1 || !almostDequalUlps(s[1], root)) {
			s[rootsIdx] = root
			rootsIdx++
		}
	} else { // we have 1 real root
		sqrtR2MinusQ3 := math.Sqrt(r2MinusQ3)
		bigA := math.Abs(r) + sqrtR2MinusQ3
		bigA = math.Cbrt(bigA) // cube root
		if r > 0 {
			bigA = -bigA
		}
		if bigA != 0 {
			bigA += q / bigA
		}
		root = bigA - adiv3
		s[rootsIdx] = root
		rootsIdx++
		if almostDequalUlps(r2, q3) {
			root = -bigA/2 - adiv3
			if !almostDequalUlps(s[0], root) {
				s[rootsIdx] = root
				rootsIdx++
			}
		}
	}
	return rootsIdx
}

// cubicRootsValidT returns the [0,1]-valid roots, snapping roots slightly outside the unit interval (within a small
// band) to the exact endpoints.
func cubicRootsValidT(a, b, c, d float64, t []float64) int {
	var s [3]float64
	realRoots := cubicRootsReal(a, b, c, d, s[:])
	foundRoots := quadAddValidTs(s[:], realRoots, t)
	for index := 0; index < realRoots; index++ {
		tValue := s[index]
		if !approximatelyOneOrLess(tValue) && between(1, tValue, 1.00005) {
			dup := false
			for idx2 := 0; idx2 < foundRoots; idx2++ {
				if approximatelyEqual(t[idx2], 1) {
					dup = true
					break
				}
			}
			if !dup {
				t[foundRoots] = 1
				foundRoots++
			}
		} else if !approximatelyZeroOrMore(tValue) && between(-0.00005, tValue, 0) {
			dup := false
			for idx2 := 0; idx2 < foundRoots; idx2++ {
				if approximatelyEqual(t[idx2], 0) {
					dup = true
					break
				}
			}
			if !dup {
				t[foundRoots] = 0
				foundRoots++
			}
		}
	}
	return foundRoots
}

// cubicFindExtrema locates the extrema of one coordinate over t in (0,1): p0..p3 are that coordinate's four control
// values. cubic'(t) = A*t^2 + B*t + C (with A,B,C divided by 3), solved for its valid t roots.
func cubicFindExtrema(p0, p1, p2, p3 float64, tValues []float64) int {
	a := p0
	b := p1
	c := p2
	d := p3
	bigA := d - a + 3*(b-c)
	bigB := 2 * (a - b - b + c)
	bigC := b - a
	return quadRootsValidT(bigA, bigB, bigC, tValues)
}

// collapsed reports whether all four control points are approximately coincident.
func (c dCubic) collapsed() bool {
	return c.pts[0].approximatelyEqual(c.pts[1]) && c.pts[0].approximatelyEqual(c.pts[2]) &&
		c.pts[0].approximatelyEqual(c.pts[3])
}

// controlsInside reports whether both middle control points project between the two endpoints (all four chord dot
// products positive).
func (c dCubic) controlsInside() bool {
	v01 := c.pts[0].sub(c.pts[1])
	v02 := c.pts[0].sub(c.pts[2])
	v03 := c.pts[0].sub(c.pts[3])
	v13 := c.pts[1].sub(c.pts[3])
	v23 := c.pts[2].sub(c.pts[3])
	return v03.dot(v01) > 0 && v03.dot(v02) > 0 && v03.dot(v13) > 0 && v03.dot(v23) > 0
}

// otherPts returns the three control points other than the (endpoint) oddMan. For oddMan==0 the offset starts at 1
// (points 1,2,3); otherwise at 0 (points 0,1,2).
func (c dCubic) otherPts(oddMan int) [3]dPoint {
	offset := 0
	if oddMan == 0 {
		offset = 1
	}
	return [3]dPoint{c.pts[offset], c.pts[offset+1], c.pts[offset+2]}
}

// cubicDerivativeAtT evaluates the derivative of one coordinate at t: p0..p3 are that coordinate's four control values.
// c'(t) = 3((b-a)(1-t)^2 + 2(c-b)t(1-t) + (d-c)t^2).
func cubicDerivativeAtT(p0, p1, p2, p3, t float64) float64 {
	oneT := 1 - t
	return 3 * ((p1-p0)*oneT*oneT + 2*(p2-p1)*t*oneT + (p3-p2)*t*t)
}

// dxdyAtT returns the (unnormalized) tangent vector at t.
func (c dCubic) dxdyAtT(t float64) dVector {
	result := dVector{
		x: cubicDerivativeAtT(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, t),
		y: cubicDerivativeAtT(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y, t),
	}
	if result.x == 0 && result.y == 0 {
		switch t {
		case 0:
			result = c.pts[2].sub(c.pts[0])
		case 1:
			result = c.pts[3].sub(c.pts[1])
		}
		// else: this degenerate case (a zero tangent away from an endpoint) is left unresolved; result stays {0,0}.
		if result.x == 0 && result.y == 0 && zeroOrOne(t) {
			result = c.pts[3].sub(c.pts[0])
		}
	}
	return result
}

// hullIntersectsPoints is the quick reject that rotates all of pts relative to a line formed by a pair of this cubic's
// hull points. If pts lie on the line or on the opposite side from this cubic's odd man, the curves at most touch at
// endpoints. Returns whether the hulls may intersect beyond the endpoints; the returned isLinear (meaningful only when
// the result is true — the early false return leaves it at its false value) reports whether this cubic's hull collapsed
// to a line.
func (c dCubic) hullIntersectsPoints(pts []dPoint) (result, isLinear bool) {
	linear := true
	var hullOrder [4]int
	hullCount := c.convexHull(&hullOrder)
	end1 := hullOrder[0]
	hullIndex := 0
	var endPt [2]dPoint
	endPt[0] = c.pts[end1]
	for {
		hullIndex = (hullIndex + 1) % hullCount
		end2 := hullOrder[hullIndex]
		endPt[1] = c.pts[end2]
		origX := endPt[0].x
		origY := endPt[0].y
		adj := endPt[1].x - origX
		opp := endPt[1].y - origY
		oddManMask := otherTwo(end1, end2)
		oddMan := end1 ^ oddManMask
		sign := (c.pts[oddMan].y-origY)*adj - (c.pts[oddMan].x-origX)*opp
		oddMan2 := end2 ^ oddManMask
		sign2 := (c.pts[oddMan2].y-origY)*adj - (c.pts[oddMan2].x-origX)*opp
		skip := sign*sign2 < 0
		if !skip && approximatelyZero(sign) {
			sign = sign2
			if approximatelyZero(sign) {
				skip = true
			}
		}
		if !skip {
			linear = false
			foundOutlier := false
			for n := 0; n < len(pts); n++ {
				test := (pts[n].y-origY)*adj - (pts[n].x-origX)*opp
				if test*sign > 0 && !preciselyZero(test) {
					foundOutlier = true
					break
				}
			}
			if !foundOutlier {
				return false, linear
			}
			endPt[0] = endPt[1]
			end1 = end2
		}
		if hullIndex == 0 {
			break
		}
	}
	return true, linear
}

// hullIntersectsCubic reports whether this cubic's hull intersects c2's hull.
func (c dCubic) hullIntersectsCubic(c2 dCubic) (sects, isLinear bool) {
	return c.hullIntersectsPoints(c2.pts[:])
}

// hullIntersectsQuad reports whether this cubic's hull intersects quad's hull.
func (c dCubic) hullIntersectsQuad(quad dQuad) (sects, isLinear bool) {
	return c.hullIntersectsPoints(quad.pts[:])
}

// hullIntersectsConic reports whether this cubic's hull intersects the conic's hull, treating the conic's control
// points as a three-point hull.
func (c dCubic) hullIntersectsConic(conic dConic) (sects, isLinear bool) {
	return c.hullIntersectsPoints(conic.pts.pts[:])
}

// dCubicPair holds the seven points of a cubic split into two adjoining sub-cubics (the split point is shared, at index
// 3).
type dCubicPair struct {
	pts [7]dPoint
}

// first returns the sub-cubic over points [0..3].
func (p *dCubicPair) first() dCubic {
	return dCubic{pts: [4]dPoint{p.pts[0], p.pts[1], p.pts[2], p.pts[3]}}
}

// second returns the sub-cubic over points [3..6].
func (p *dCubicPair) second() dCubic {
	return dCubic{pts: [4]dPoint{p.pts[3], p.pts[4], p.pts[5], p.pts[6]}}
}

// interpCubicCoords evaluates one coordinate at t from the four control-point values, using De Casteljau interpolation.
func interpCubicCoords(p0, p1, p2, p3, t float64) float64 {
	ab := dInterp(p0, p1, t)
	bc := dInterp(p1, p2, t)
	cd := dInterp(p2, p3, t)
	abc := dInterp(ab, bc, t)
	bcd := dInterp(bc, cd, t)
	return dInterp(abc, bcd, t)
}

// interpCubicCoordsChop performs the classic one-t subdivision of one coordinate, producing the seven coordinate values
// (the two sub-cubics share d6=abcd). Seven scalar results are the natural shape of the subdivision; a struct would
// only obscure it.
func interpCubicCoordsChop(p0, p1, p2, p3, t float64) (d0, d2, d4, d6, d8, d10, d12 float64) { //nolint:gocritic // see above
	ab := dInterp(p0, p1, t)
	bc := dInterp(p1, p2, t)
	cd := dInterp(p2, p3, t)
	abc := dInterp(ab, bc, t)
	bcd := dInterp(bc, cd, t)
	abcd := dInterp(abc, bcd, t)
	return p0, ab, abc, abcd, bcd, cd, p3
}

// chopAt splits the cubic at t into two adjoining sub-cubics. The t==0.5 case uses the exact binary-fraction form.
func (c dCubic) chopAt(t float64) dCubicPair {
	var dst dCubicPair
	if t == 0.5 {
		dst.pts[0] = c.pts[0]
		dst.pts[1].x = (c.pts[0].x + c.pts[1].x) / 2
		dst.pts[1].y = (c.pts[0].y + c.pts[1].y) / 2
		dst.pts[2].x = (c.pts[0].x + 2*c.pts[1].x + c.pts[2].x) / 4
		dst.pts[2].y = (c.pts[0].y + 2*c.pts[1].y + c.pts[2].y) / 4
		dst.pts[3].x = (c.pts[0].x + 3*(c.pts[1].x+c.pts[2].x) + c.pts[3].x) / 8
		dst.pts[3].y = (c.pts[0].y + 3*(c.pts[1].y+c.pts[2].y) + c.pts[3].y) / 8
		dst.pts[4].x = (c.pts[1].x + 2*c.pts[2].x + c.pts[3].x) / 4
		dst.pts[4].y = (c.pts[1].y + 2*c.pts[2].y + c.pts[3].y) / 4
		dst.pts[5].x = (c.pts[2].x + c.pts[3].x) / 2
		dst.pts[5].y = (c.pts[2].y + c.pts[3].y) / 2
		dst.pts[6] = c.pts[3]
		return dst
	}
	dst.pts[0].x, dst.pts[1].x, dst.pts[2].x, dst.pts[3].x, dst.pts[4].x, dst.pts[5].x, dst.pts[6].x = interpCubicCoordsChop(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, t)
	dst.pts[0].y, dst.pts[1].y, dst.pts[2].y, dst.pts[3].y, dst.pts[4].y, dst.pts[5].y, dst.pts[6].y = interpCubicCoordsChop(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y, t)
	return dst
}

// subDivide returns the sub-cubic over the parameter range [t1, t2]. The endpoints are evaluated directly; the middle
// control points are recovered from the values at the two thirds of the range (the 27/8 recovery formula for a cubic
// Bezier).
func (c dCubic) subDivide(t1, t2 float64) dCubic {
	if t1 == 0 || t2 == 1 {
		if t1 == 0 && t2 == 1 {
			return c
		}
		if t1 == 0 {
			pair := c.chopAt(t2)
			return pair.first()
		}
		pair := c.chopAt(t1)
		return pair.second()
	}
	var dst dCubic
	ax := interpCubicCoords(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, t1)
	ay := interpCubicCoords(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y, t1)
	ex := interpCubicCoords(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, (t1*2+t2)/3)
	ey := interpCubicCoords(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y, (t1*2+t2)/3)
	fx := interpCubicCoords(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, (t1+t2*2)/3)
	fy := interpCubicCoords(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y, (t1+t2*2)/3)
	dx := interpCubicCoords(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, t2)
	dy := interpCubicCoords(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y, t2)
	mx := ex*27 - ax*8 - dx
	my := ey*27 - ay*8 - dy
	nx := fx*27 - ax - dx*8
	ny := fy*27 - ay - dy*8
	dst.pts[0].x = ax
	dst.pts[0].y = ay
	dst.pts[1].x = (mx*2 - nx) / 18
	dst.pts[1].y = (my*2 - ny) / 18
	dst.pts[2].x = (nx*2 - mx) / 18
	dst.pts[2].y = (ny*2 - my) / 18
	dst.pts[3].x = dx
	dst.pts[3].y = dy
	return dst
}

// align snaps the control point at dstPt onto an axis of the segment endpoint when the end and control share that
// coordinate exactly.
func (c dCubic) align(endIndex, ctrlIndex int, dstPt *dPoint) {
	if c.pts[endIndex].x == c.pts[ctrlIndex].x {
		dstPt.x = c.pts[endIndex].x
	}
	if c.pts[endIndex].y == c.pts[ctrlIndex].y {
		dstPt.y = c.pts[endIndex].y
	}
}

// subDivideAD returns the two interior control points of the sub-cubic over [t1, t2], given its exact endpoints a and
// d. The directly computed control points are nudged so the endpoints land on a and d, with axis alignment at the
// shared t==0/1 ends.
func (c dCubic) subDivideAD(a, d dPoint, t1, t2 float64) [2]dPoint {
	sub := c.subDivide(t1, t2)
	var dst [2]dPoint
	dst[0] = sub.pts[1].plusVector(a.sub(sub.pts[0]))
	dst[1] = sub.pts[2].plusVector(d.sub(sub.pts[3]))
	if t1 == 0 || t2 == 0 {
		if t1 == 0 {
			c.align(0, 1, &dst[0])
		} else {
			c.align(0, 1, &dst[1])
		}
	}
	if t1 == 1 || t2 == 1 {
		if t1 == 1 {
			c.align(3, 2, &dst[0])
		} else {
			c.align(3, 2, &dst[1])
		}
	}
	if almostBequalUlps(dst[0].x, a.x) {
		dst[0].x = a.x
	}
	if almostBequalUlps(dst[0].y, a.y) {
		dst[0].y = a.y
	}
	if almostBequalUlps(dst[1].x, d.x) {
		dst[1].x = d.x
	}
	if almostBequalUlps(dst[1].y, d.y) {
		dst[1].y = d.y
	}
	return dst
}

// findInflections returns the t values (in [0,1]) where the cubic's curvature changes sign.
func (c dCubic) findInflections(tValues []float64) int {
	ax := c.pts[1].x - c.pts[0].x
	ay := c.pts[1].y - c.pts[0].y
	bx := c.pts[2].x - 2*c.pts[1].x + c.pts[0].x
	by := c.pts[2].y - 2*c.pts[1].y + c.pts[0].y
	cx := c.pts[3].x + 3*(c.pts[1].x-c.pts[2].x) - c.pts[0].x
	cy := c.pts[3].y + 3*(c.pts[1].y-c.pts[2].y) - c.pts[0].y
	return quadRootsValidT(bx*cy-by*cx, ax*cy-ay*cx, ax*by-ay*bx, tValues)
}

// binarySearch refines a t in [minT,maxT] where the cubic's chosen axis coordinate meets axisIntercept, giving up
// (returning -1) once changing t no longer moves the point.
func (c dCubic) binarySearch(minT, maxT, axisIntercept float64, axis searchAxis) float64 {
	t := (minT + maxT) / 2
	step := (t - minT) / 2
	cubicAtT := c.ptAtT(t)
	calcPos := cubicAtT.axis(axis)
	calcDist := calcPos - axisIntercept
	for {
		priorT := math.Max(minT, t-step)
		lessPt := c.ptAtT(priorT)
		if approximatelyEqualHalf(lessPt.x, cubicAtT.x) && approximatelyEqualHalf(lessPt.y, cubicAtT.y) {
			return -1 // binary search found no point at this axis intercept
		}
		lessDist := lessPt.axis(axis) - axisIntercept
		lastStep := step
		step /= 2
		var takePrior bool
		if calcDist > 0 {
			takePrior = calcDist > lessDist
		} else {
			takePrior = calcDist < lessDist
		}
		proceed := true
		if takePrior {
			t = priorT
		} else {
			nextT := t + lastStep
			if nextT > maxT {
				return -1
			}
			morePt := c.ptAtT(nextT)
			if approximatelyEqualHalf(morePt.x, cubicAtT.x) && approximatelyEqualHalf(morePt.y, cubicAtT.y) {
				return -1 // binary search found no point at this axis intercept
			}
			moreDist := morePt.axis(axis) - axisIntercept
			var skip bool
			if calcDist > 0 {
				skip = calcDist <= moreDist
			} else {
				skip = calcDist >= moreDist
			}
			if skip {
				proceed = false // skip the update below and re-check the loop condition at the current t
			} else {
				t = nextT
			}
		}
		if proceed {
			testAtT := c.ptAtT(t)
			cubicAtT = testAtT
			calcPos = cubicAtT.axis(axis)
			calcDist = calcPos - axisIntercept
		}
		if approximatelyEqual(calcPos, axisIntercept) {
			break
		}
	}
	return t
}

// searchRoots partitions [0,1] at the cubic's extrema, inflections, and endpoints, then binary-searches each monotone
// interval for a crossing of axisIntercept. extremeTs must have room for the appended inflections plus the 0 and 1
// endpoints (six slots).
func (c dCubic) searchRoots(extremeTs []float64, extrema int, axisIntercept float64, axis searchAxis, validRoots []float64) int {
	extrema += c.findInflections(extremeTs[extrema:])
	extremeTs[extrema] = 0
	extrema++
	extremeTs[extrema] = 1
	sort.Float64s(extremeTs[:extrema+1])
	validCount := 0
	for index := 0; index < extrema; {
		minT := extremeTs[index]
		index++
		maxT := extremeTs[index]
		if minT == maxT {
			continue
		}
		newT := c.binarySearch(minT, maxT, axisIntercept, axis)
		if newT >= 0 {
			if validCount >= 3 {
				return 0
			}
			validRoots[validCount] = newT
			validCount++
		}
	}
	return validCount
}

// gPrecisionUnit is the divisor used by calcPrecision to scale a cubic's control-polygon length into a
// curvature-comparison unit.
const gPrecisionUnit = 256

// calcPrecision returns the rough scale of the cubic (its control-polygon length over gPrecisionUnit), used to judge
// whether curvature is extreme.
func (c dCubic) calcPrecision() float64 {
	return (c.pts[1].sub(c.pts[0]).length() +
		c.pts[2].sub(c.pts[1]).length() +
		c.pts[3].sub(c.pts[2]).length()) / gPrecisionUnit
}

// formulateF1DotF2 computes, for one coordinate, the coefficients of F1 dot F2 (the first derivative dotted with the
// second) as a cubic in t: CCt³ + 3BCt² + (2BB+CA)t + AB. p0..p3 are that coordinate's four control values.
func formulateF1DotF2(p0, p1, p2, p3 float64) [4]float64 {
	a := p1 - p0
	b := p2 - 2*p1 + p0
	c := p3 + 3*(p1-p2) - p0
	return [4]float64{c * c, 3 * b * c, 2*b*b + c*a, a * b}
}

// findMaxCurvature returns the t values in (0,1) where F1 dot F2 == 0 (the first derivative perpendicular to the
// second) — candidate locations of maximum curvature.
func (c dCubic) findMaxCurvature(tValues []float64) int {
	coeffX := formulateF1DotF2(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x)
	coeffY := formulateF1DotF2(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y)
	var coeff [4]float64
	for i := range coeff {
		coeff[i] = coeffX[i] + coeffY[i]
	}
	return cubicRootsValidT(coeff[0], coeff[1], coeff[2], coeff[3], tValues)
}

// toFloatPoints converts the double control points to float32, forcing tiny magnitudes to zero, and reports whether
// every resulting coordinate is finite.
func (c dCubic) toFloatPoints(pts *[4]geom.Point) bool {
	for i := 0; i < 4; i++ {
		pts[i] = forceSmallToZero(geom.Point{X: float32(c.pts[i].x), Y: float32(c.pts[i].y)})
	}
	return pointsAreFinite(pts[:])
}

// complexBreak returns, for a self-intersecting or high-curvature cubic, the t values (in increasing input order, at
// most three) at which it should be split before intersection, writing them into t and returning their count (0 for a
// simple monotonic cubic).
func complexBreak(pointsPtr [4]geom.Point, t []float32) int {
	var cubic dCubic
	cubic.set(pointsPtr)
	if cubic.monotonicInX() && cubic.monotonicInY() {
		return 0
	}
	cubicType, tt, ss := geom.ClassifyCubic(pointsPtr)
	switch cubicType {
	case geom.CubicLoop:
		td, te, sd, se := tt[0], tt[1], ss[0], ss[1]
		if roughlyBetween(0, td, sd) && roughlyBetween(0, te, se) {
			t[0] = float32((td*se + te*sd) / (2 * sd * se))
			return b2i(t[0] > 0 && t[0] < 1)
		}
		fallthrough // fall through if no t value found
	case geom.CubicSerpentine, geom.CubicLocalCusp, geom.CubicCuspAtInfinity:
		var inflectionTs [2]float64
		infTCount := cubic.findInflections(inflectionTs[:])
		var maxCurvature [3]float64
		roots := cubic.findMaxCurvature(maxCurvature[:])
		if infTCount == 2 {
			for index := 0; index < roots; index++ {
				if between(inflectionTs[0], maxCurvature[index], inflectionTs[1]) {
					t[0] = float32(maxCurvature[index])
					return b2i(t[0] > 0 && t[0] < 1)
				}
			}
		} else {
			resultCount := 0
			// constant found through experimentation -- maybe there's a better way
			precision := cubic.calcPrecision() * 2
			for index := 0; index < roots; index++ {
				testT := maxCurvature[index]
				if testT <= 0 || testT >= 1 {
					continue
				}
				// don't call dxdyAtT since we want (0,0) results
				dPt := dVector{
					x: cubicDerivativeAtT(cubic.pts[0].x, cubic.pts[1].x, cubic.pts[2].x, cubic.pts[3].x, testT),
					y: cubicDerivativeAtT(cubic.pts[0].y, cubic.pts[1].y, cubic.pts[2].y, cubic.pts[3].y, testT),
				}
				if dPt.length() < precision {
					t[resultCount] = float32(testT)
					resultCount++
				}
			}
			if resultCount == 0 && infTCount == 1 {
				t[0] = float32(inflectionTs[0])
				resultCount = b2i(t[0] > 0 && t[0] < 1)
			}
			return resultCount
		}
	default:
	}
	return 0
}

// subDivideCubicPts builds a dCubic from the float32 points a and returns the sub-cubic over [t1, t2].
func subDivideCubicPts(a [4]geom.Point, t1, t2 float64) dCubic {
	var cubic dCubic
	cubic.set(a)
	return cubic.subDivide(t1, t2)
}
