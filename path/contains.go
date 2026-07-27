// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import (
	"github.com/richardwilkes/canvas/geom"
)

// polyEval3 evaluates a quadratic polynomial a*t^2 + b*t + c at t via Horner's method.
func polyEval3(a, b, c, t float32) float32 {
	return (a*t+b)*t + c
}

// polyEval4 evaluates a cubic polynomial a*t^3 + b*t^2 + c*t + d at t via Horner's method.
func polyEval4(a, b, c, d, t float32) float32 {
	return ((a*t+b)*t+c)*t + d
}

// between reports whether b lies between a and c (inclusive).
func between(a, b, c float32) bool {
	return (a-b)*(c-b) <= 0
}

// evalCubicPts evaluates one coordinate of a cubic Bezier at t, given the four control points' values for that
// coordinate.
func evalCubicPts(c0, c1, c2, c3, t float32) float32 {
	a := c3 + 3*(c1-c2) - c0
	b := 3 * (c2 - c1 - c1 + c0)
	c := 3 * (c1 - c0)
	return polyEval4(a, b, c, c0, t)
}

// checkOnCurve reports whether (x, y) lies on the segment from start to end. For a horizontal segment the X range it
// accepts is half-open, inclusive of start.X but exclusive of end.X, in either direction; otherwise it requires an
// exact match with start. That asymmetry is the point, not an oversight: Contains walks the segments in order, so each
// shared endpoint is one segment's end and the next one's start, and accepting both ends would credit the same vertex
// to onCurveCount twice and flip the parity Contains resolves ties with.
func checkOnCurve(x, y float32, start, end geom.Point) bool {
	if start.Y == end.Y {
		return between(start.X, x, end.X) && x != end.X
	}
	return x == start.X && y == start.Y
}

// chopMonoCubicAtY finds t such that the Y-monotonic cubic's y(t) == y, via bisection.
func chopMonoCubicAtY(pts []geom.Point, y float32) (float32, bool) {
	var ycrv [4]float32
	ycrv[0] = pts[0].Y - y
	ycrv[1] = pts[1].Y - y
	ycrv[2] = pts[2].Y - y
	ycrv[3] = pts[3].Y - y

	// Check that the endpoints straddle zero.
	var tNeg, tPos float32 // Negative and positive function parameters.
	switch {
	case ycrv[0] < 0:
		if ycrv[3] < 0 {
			return 0, false
		}
		tNeg = 0
		tPos = 1
	case ycrv[0] > 0:
		if ycrv[3] > 0 {
			return 0, false
		}
		tNeg = 1
		tPos = 0
	default:
		return 0, true
	}

	const tol = 1.0 / 65536 // 1e-5 for float.
	for {
		tMid := (tPos + tNeg) / 2
		y01 := geom.ScalarInterp(ycrv[0], ycrv[1], tMid)
		y12 := geom.ScalarInterp(ycrv[1], ycrv[2], tMid)
		y23 := geom.ScalarInterp(ycrv[2], ycrv[3], tMid)
		y012 := geom.ScalarInterp(y01, y12, tMid)
		y123 := geom.ScalarInterp(y12, y23, tMid)
		y0123 := geom.ScalarInterp(y012, y123, tMid)
		if y0123 == 0 {
			return tMid, true
		}
		if y0123 < 0 {
			tNeg = tMid
		} else {
			tPos = tMid
		}
		if !(geom.ScalarAbs(tPos-tNeg) > tol) { // NaN-safe
			break
		}
	}
	return (tNeg + tPos) / 2, true
}

// windingMonoCubic computes the winding contribution of a Y-monotonic cubic segment at (x, y), incrementing
// *onCurveCount when the point lies exactly on the curve.
func windingMonoCubic(pts []geom.Point, x, y float32, onCurveCount *int) int {
	y0 := pts[0].Y
	y3 := pts[3].Y

	dir := 1
	if y0 > y3 {
		y0, y3 = y3, y0
		dir = -1
	}
	if y < y0 || y > y3 {
		return 0
	}
	if checkOnCurve(x, y, pts[0], pts[3]) {
		*onCurveCount++
		return 0
	}
	if y == y3 {
		return 0
	}

	// quickreject or quickaccept
	minV, maxV := pts[0].X, pts[0].X
	for i := 1; i < 4; i++ {
		if pts[i].X < minV {
			minV = pts[i].X
		}
		if pts[i].X > maxV {
			maxV = pts[i].X
		}
	}
	if x < minV {
		return 0
	}
	if x > maxV {
		return dir
	}

	// compute the actual x(t) value
	t, ok := chopMonoCubicAtY(pts, y)
	if !ok {
		return 0
	}
	xt := evalCubicPts(pts[0].X, pts[1].X, pts[2].X, pts[3].X, t)
	if geom.ScalarNearlyEqual(xt, x) {
		if x != pts[3].X || y != pts[3].Y { // don't test end points; they're start points
			*onCurveCount++
			return 0
		}
	}
	if xt < x {
		return dir
	}
	return 0
}

// windingCubic chops a cubic at its Y extrema into monotonic pieces and sums their winding contributions at (x, y).
func windingCubic(pts []geom.Point, x, y float32, onCurveCount *int) int {
	var dst [10]geom.Point
	n := geom.ChopCubicAtYExtrema(pts, dst[:])
	w := 0
	for i := 0; i <= n; i++ {
		w += windingMonoCubic(dst[i*3:i*3+4], x, y, onCurveCount)
	}
	return w
}

// conicEvalNumerator evaluates the numerator of the conic's rational quadratic at t for one coordinate axis, given that
// axis's three control-point values (src0, src1, src2) and weight w.
func conicEvalNumerator(src0, src1, src2, w, t float32) float64 {
	src2w := src1 * w
	c := src0
	a := src2 - 2*src2w + c
	b := 2 * (src2w - c)
	return float64(polyEval3(a, b, c, t))
}

// conicEvalDenominator evaluates the denominator of the conic's rational quadratic at t, given weight w.
func conicEvalDenominator(w, t float32) float64 {
	b := 2 * (w - 1)
	c := float32(1)
	a := -b
	return float64(polyEval3(a, b, c, t))
}

// windingMonoConic computes the winding contribution of a Y-monotonic conic at (x, y), incrementing *onCurveCount when
// the point lies exactly on the curve.
func windingMonoConic(conic *geom.Conic, x, y float32, onCurveCount *int) int {
	pts := conic.Pts[:]
	y0 := pts[0].Y
	y2 := pts[2].Y

	dir := 1
	if y0 > y2 {
		y0, y2 = y2, y0
		dir = -1
	}
	if y < y0 || y > y2 {
		return 0
	}
	if checkOnCurve(x, y, pts[0], pts[2]) {
		*onCurveCount++
		return 0
	}
	if y == y2 {
		return 0
	}

	var roots [2]float32
	a := pts[2].Y
	b := pts[1].Y*conic.W - y*conic.W + y
	c := pts[0].Y
	a += c - 2*b // A = a + c - 2*(b*w - yCept*w + yCept)
	b -= c       // B = b*w - w * yCept + yCept - a
	c -= y
	n := geom.FindUnitQuadRoots(a, 2*b, c, &roots)
	var xt float32
	if n == 0 {
		// zero roots are returned only when y0 == y. Need [0] if dir == 1 and [2] if dir == -1.
		xt = pts[1-dir].X
	} else {
		t := roots[0]
		xt = float32(conicEvalNumerator(pts[0].X, pts[1].X, pts[2].X, conic.W, t) /
			conicEvalDenominator(conic.W, t))
	}
	if geom.ScalarNearlyEqual(xt, x) {
		if x != pts[2].X || y != pts[2].Y { // don't test end points; they're start points
			*onCurveCount++
			return 0
		}
	}
	if xt < x {
		return dir
	}
	return 0
}

// isMonoQuad reports whether the quadratic with the given Y coordinates is already Y-monotonic.
func isMonoQuad(y0, y1, y2 float32) bool {
	if y0 == y1 {
		return true
	}
	if y0 < y1 {
		return y1 <= y2
	}
	return y1 >= y2
}

// windingConic chops a conic at its Y extrema (if needed) into monotonic pieces and sums their winding contributions at
// (x, y).
func windingConic(pts []geom.Point, x, y, weight float32, onCurveCount *int) int {
	conic := geom.MakeConic(pts[0], pts[1], pts[2], weight)
	var chopped [2]geom.Conic
	// If the data points are very large, the conic may not be monotonic but may also fail to chop. Then, the chopper
	// does not split the original conic in two.
	isMono := isMonoQuad(pts[0].Y, pts[1].Y, pts[2].Y) || !conic.ChopAtYExtrema(&chopped)
	var w int
	if isMono {
		w = windingMonoConic(&conic, x, y, onCurveCount)
	} else {
		w = windingMonoConic(&chopped[0], x, y, onCurveCount)
		w += windingMonoConic(&chopped[1], x, y, onCurveCount)
	}
	return w
}

// windingMonoQuad computes the winding contribution of a Y-monotonic quadratic at (x, y), incrementing *onCurveCount
// when the point lies exactly on the curve.
func windingMonoQuad(pts []geom.Point, x, y float32, onCurveCount *int) int {
	y0 := pts[0].Y
	y2 := pts[2].Y

	dir := 1
	if y0 > y2 {
		y0, y2 = y2, y0
		dir = -1
	}
	if y < y0 || y > y2 {
		return 0
	}
	if checkOnCurve(x, y, pts[0], pts[2]) {
		*onCurveCount++
		return 0
	}
	if y == y2 {
		return 0
	}

	var roots [2]float32
	n := geom.FindUnitQuadRoots(pts[0].Y-2*pts[1].Y+pts[2].Y,
		2*(pts[1].Y-pts[0].Y),
		pts[0].Y-y,
		&roots)
	var xt float32
	if n == 0 {
		// zero roots are returned only when y0 == y. Need [0] if dir == 1 and [2] if dir == -1.
		xt = pts[1-dir].X
	} else {
		t := roots[0]
		c := pts[0].X
		a := pts[2].X - 2*pts[1].X + c
		b := 2 * (pts[1].X - c)
		xt = polyEval3(a, b, c, t)
	}
	if geom.ScalarNearlyEqual(xt, x) {
		if x != pts[2].X || y != pts[2].Y { // don't test end points; they're start points
			*onCurveCount++
			return 0
		}
	}
	if xt < x {
		return dir
	}
	return 0
}

// windingQuad chops a quadratic at its Y extrema (if needed) into monotonic pieces and sums their winding contributions
// at (x, y).
func windingQuad(pts []geom.Point, x, y float32, onCurveCount *int) int {
	var storage [5]geom.Point
	span := pts
	n := 0

	if !isMonoQuad(pts[0].Y, pts[1].Y, pts[2].Y) {
		n = geom.ChopQuadAtYExtrema(pts, storage[:])
		span = storage[:]
	}
	w := windingMonoQuad(span, x, y, onCurveCount)
	if n > 0 {
		w += windingMonoQuad(span[2:], x, y, onCurveCount)
	}
	return w
}

// windingLine computes the winding contribution of a line segment at (x, y), incrementing *onCurveCount when the point
// lies exactly on the line.
func windingLine(pts []geom.Point, x, y float32, onCurveCount *int) int {
	x0 := pts[0].X
	y0 := pts[0].Y
	x1 := pts[1].X
	y1 := pts[1].Y

	dy := y1 - y0

	dir := 1
	if y0 > y1 {
		y0, y1 = y1, y0
		dir = -1
	}
	if y < y0 || y > y1 {
		return 0
	}
	if checkOnCurve(x, y, pts[0], pts[1]) {
		*onCurveCount++
		return 0
	}
	if y == y1 {
		return 0
	}
	// Explicitly rounded products: cross is compared against exactly zero below (a point on the line must yield cross
	// == 0), and letting the compiler fuse a multiply into the subtract (as Go does on arm64) leaves a nonzero residue
	// that misclassifies on-curve points. See geom.Point.Cross for the policy.
	cross := float32((x1-x0)*(y-pts[0].Y)) - float32(dy*(x-x0))

	if cross == 0 {
		// zero cross means the point is on the line, and since the case where y of the query point is at the end point
		// is handled above, we can be sure that we're on the line (excluding the end point) here.
		if x != x1 || y != pts[1].Y {
			*onCurveCount++
		}
		dir = 0
	} else if signAsInt(cross) == dir {
		dir = 0
	}
	return dir
}

// signAsInt returns the sign of x as -1, 0, or 1.
func signAsInt(x float32) int {
	switch {
	case x < 0:
		return -1
	case x > 0:
		return 1
	default:
		return 0
	}
}

// tangentCubic appends the tangent vector at (x, y) to *tangents for each point where the cubic passes through (x, y),
// after chopping at Y extrema.
func tangentCubic(pts []geom.Point, x, y float32, tangents *[]geom.Point) {
	if !between(pts[0].Y, y, pts[1].Y) && !between(pts[1].Y, y, pts[2].Y) &&
		!between(pts[2].Y, y, pts[3].Y) {
		return
	}
	if !between(pts[0].X, x, pts[1].X) && !between(pts[1].X, x, pts[2].X) &&
		!between(pts[2].X, x, pts[3].X) {
		return
	}
	var dst [10]geom.Point
	n := geom.ChopCubicAtYExtrema(pts, dst[:])
	for i := 0; i <= n; i++ {
		c := dst[i*3 : i*3+4]
		t, ok := chopMonoCubicAtY(c, y)
		if !ok {
			continue
		}
		xt := evalCubicPts(c[0].X, c[1].X, c[2].X, c[3].X, t)
		if !geom.ScalarNearlyEqual(x, xt) {
			continue
		}
		*tangents = append(*tangents, geom.EvalCubicTangentAt(c, t))
	}
}

// tangentConic appends the tangent vector at (x, y) to *tangents for each point where the conic passes through (x, y).
func tangentConic(pts []geom.Point, x, y, w float32, tangents *[]geom.Point) {
	if !between(pts[0].Y, y, pts[1].Y) && !between(pts[1].Y, y, pts[2].Y) {
		return
	}
	if !between(pts[0].X, x, pts[1].X) && !between(pts[1].X, x, pts[2].X) {
		return
	}
	var roots [2]float32
	a := pts[2].Y
	b := pts[1].Y*w - y*w + y
	c := pts[0].Y
	a += c - 2*b // A = a + c - 2*(b*w - yCept*w + yCept)
	b -= c       // B = b*w - w * yCept + yCept - a
	c -= y
	n := geom.FindUnitQuadRoots(a, 2*b, c, &roots)
	for index := 0; index < n; index++ {
		t := roots[index]
		xt := float32(conicEvalNumerator(pts[0].X, pts[1].X, pts[2].X, w, t) / conicEvalDenominator(w, t))
		if !geom.ScalarNearlyEqual(x, xt) {
			continue
		}
		conic := geom.MakeConic(pts[0], pts[1], pts[2], w)
		*tangents = append(*tangents, conic.EvalTangentAt(t))
	}
}

// tangentQuad appends the tangent vector at (x, y) to *tangents for each point where the quadratic passes through (x,
// y).
func tangentQuad(pts []geom.Point, x, y float32, tangents *[]geom.Point) {
	if !between(pts[0].Y, y, pts[1].Y) && !between(pts[1].Y, y, pts[2].Y) {
		return
	}
	if !between(pts[0].X, x, pts[1].X) && !between(pts[1].X, x, pts[2].X) {
		return
	}
	var roots [2]float32
	n := geom.FindUnitQuadRoots(pts[0].Y-2*pts[1].Y+pts[2].Y,
		2*(pts[1].Y-pts[0].Y),
		pts[0].Y-y,
		&roots)
	for index := 0; index < n; index++ {
		t := roots[index]
		c := pts[0].X
		a := pts[2].X - 2*pts[1].X + c
		b := 2 * (pts[1].X - c)
		xt := polyEval3(a, b, c, t)
		if !geom.ScalarNearlyEqual(x, xt) {
			continue
		}
		*tangents = append(*tangents, geom.EvalQuadTangentAt(pts, t))
	}
}

// tangentLine appends the line's direction vector to *tangents if it passes through (x, y).
func tangentLine(pts []geom.Point, x, y float32, tangents *[]geom.Point) {
	y0 := pts[0].Y
	y1 := pts[1].Y
	if !between(y0, y, y1) {
		return
	}
	x0 := pts[0].X
	x1 := pts[1].X
	if !between(x0, x, x1) {
		return
	}
	dx := x1 - x0
	dy := y1 - y0
	if !geom.ScalarNearlyEqual((x-x0)*dy, dx*(y-y0)) {
		return
	}
	*tangents = append(*tangents, geom.Pt(dx, dy))
}

// containsInclusive reports whether (x, y) lies within r, inclusive of the edges.
func containsInclusive(r geom.Rect, x, y float32) bool {
	return r.Left <= x && x <= r.Right && r.Top <= y && y <= r.Bottom
}

// Contains reports whether the point (x, y) is in the filled area of the path, honoring the fill type (including
// inverse fills).
func (p *Path) Contains(x, y float32) bool {
	isInverse := p.IsInverseFillType()
	if p.IsEmpty() {
		return isInverse
	}

	if !containsInclusive(p.Bounds(), x, y) {
		return isInverse
	}

	iter := NewIter(p, true)
	w := 0
	onCurveCount := 0
	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		switch verb {
		case VerbMove, VerbClose:
		case VerbLine:
			w += windingLine(pts[:2], x, y, &onCurveCount)
		case VerbQuad:
			w += windingQuad(pts[:3], x, y, &onCurveCount)
		case VerbConic:
			w += windingConic(pts[:3], x, y, iter.ConicWeight(), &onCurveCount)
		case VerbCubic:
			w += windingCubic(pts[:4], x, y, &onCurveCount)
		}
	}
	evenOddFill := p.fillType == FillEvenOdd || p.fillType == FillInverseEvenOdd
	if evenOddFill {
		w &= 1
	}
	if w != 0 {
		return !isInverse
	}
	if onCurveCount <= 1 {
		return (onCurveCount > 0) != isInverse
	}
	if onCurveCount&1 != 0 || evenOddFill {
		return (onCurveCount&1 != 0) != isInverse
	}
	// If the point touches an even number of curves, and the fill is winding, check for coincidence. Count coincidence
	// as places where the on curve points have identical tangents.
	iter.SetPath(p, true)
	var tangents []geom.Point
	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		oldCount := len(tangents)
		switch verb {
		case VerbMove, VerbClose:
		case VerbLine:
			tangentLine(pts[:2], x, y, &tangents)
		case VerbQuad:
			tangentQuad(pts[:3], x, y, &tangents)
		case VerbConic:
			tangentConic(pts[:3], x, y, iter.ConicWeight(), &tangents)
		case VerbCubic:
			tangentCubic(pts[:4], x, y, &tangents)
		}
		if len(tangents) > oldCount {
			last := len(tangents) - 1
			tangent := tangents[last]
			if geom.ScalarNearlyZero(tangent.LengthSqd()) {
				tangents = tangents[:last]
			} else {
				for index := 0; index < last; index++ {
					test := tangents[index]
					if geom.ScalarNearlyZero(test.Cross(tangent)) &&
						signAsInt(tangent.X*test.X) <= 0 &&
						signAsInt(tangent.Y*test.Y) <= 0 {
						// Remove both entries: drop the last one, then remove index via move-from-end.
						tangents = tangents[:last]
						tangents[index] = tangents[len(tangents)-1]
						tangents = tangents[:len(tangents)-1]
						break
					}
				}
			}
		}
	}
	return (len(tangents) != 0) != isInverse
}
