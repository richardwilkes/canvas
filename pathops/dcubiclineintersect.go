// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The cubic/line intersection solver (the general intersect plus the axis-aligned horizontal/vertical lanes and the ray
// form), implemented by the lineCubicIntersections struct. The ray solve rotates the cubic into the line's frame,
// solves the resulting cubic, and — when a solved root does not land on the axis — re-solves via the extrema/inflection
// partition (searchRoots/binarySearch). Also implements the cubic case of near-endpoint handling (nearPointForCubic):
// it drops a perpendicular from a point and ray-intersects the cubic.

package pathops

import "math"

// lineCubicIntersections solves for the intersection t values between a cubic and a line.
type lineCubicIntersections struct {
	line          *dLine
	intersections *intersections
	cubic         dCubic
	allowNear     bool
}

// newLineCubicIntersections builds a solver for cubic c against line l, storing results in i. The result cap is set to
// 4 to allow a short partial coincidence plus a discrete intersection.
func newLineCubicIntersections(c dCubic, l *dLine, i *intersections) *lineCubicIntersections {
	i.setMax(4)
	return &lineCubicIntersections{cubic: c, line: l, intersections: i, allowNear: true}
}

// setAllowNear controls whether near (not just exact) endpoint matches are accepted.
func (lc *lineCubicIntersections) setAllowNear(allow bool) { lc.allowNear = allow }

// checkCoincident scans adjacent intersection results and, when the cubic's midpoint between them also lies on the
// line, marks the pair (or folds one into the other) as a coincident run rather than two discrete crossings.
func (lc *lineCubicIntersections) checkCoincident() {
	last := lc.intersections.used - 1
	for index := 0; index < last; {
		cubicMidT := (lc.intersections.ts[0][index] + lc.intersections.ts[0][index+1]) / 2
		cubicMidPt := lc.cubic.ptAtT(cubicMidT)
		t := lc.line.nearPoint(cubicMidPt, nil)
		if t < 0 {
			index++
			continue
		}
		switch {
		case lc.intersections.isCoincident(index):
			lc.intersections.removeOne(index)
			last--
		case lc.intersections.isCoincident(index + 1):
			lc.intersections.removeOne(index + 1)
			last--
		default:
			lc.intersections.setCoincident(index)
			index++
		}
		lc.intersections.setCoincident(index)
	}
}

// intersectRay rotates the cubic's control points so the line is horizontal (the signed perpendicular distance becomes
// each rotated x), solves the resulting cubic, and — if a root's rotated x is not approximately zero — re-solves
// against the extrema/inflection partition after also computing the along-line rotation (rotated y).
func (lc *lineCubicIntersections) intersectRay(roots []float64) int {
	adj := lc.line.pts[1].x - lc.line.pts[0].x
	opp := lc.line.pts[1].y - lc.line.pts[0].y
	var c dCubic
	for n := 0; n < 4; n++ {
		c.pts[n].x = (lc.cubic.pts[n].y-lc.line.pts[0].y)*adj - (lc.cubic.pts[n].x-lc.line.pts[0].x)*opp
	}
	a, b, cc, d := cubicCoefficients(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x)
	count := cubicRootsValidT(a, b, cc, d, roots)
	for index := 0; index < count; index++ {
		calcPt := c.ptAtT(roots[index])
		if !approximatelyZero(calcPt.x) {
			for n := 0; n < 4; n++ {
				c.pts[n].y = (lc.cubic.pts[n].y-lc.line.pts[0].y)*opp + (lc.cubic.pts[n].x-lc.line.pts[0].x)*adj
			}
			var extremeTs [6]float64
			extrema := cubicFindExtrema(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, extremeTs[:])
			count = c.searchRoots(extremeTs[:], extrema, 0, xAxis, roots)
			break
		}
	}
	return count
}

// intersect finds all intersections between the cubic and a general (non-axis-aligned) line, including exact and near
// endpoint matches and coincident runs.
func (lc *lineCubicIntersections) intersect() int {
	lc.addExactEndPoints()
	if lc.allowNear {
		lc.addNearEndPoints()
	}
	var rootVals [3]float64
	roots := lc.intersectRay(rootVals[:])
	for index := 0; index < roots; index++ {
		cubicT := rootVals[index]
		lineT := lc.findLineT(cubicT)
		ok, cT, lT, pt := lc.pinTs(cubicT, lineT, dPoint{}, pointUninitialized)
		if ok && lc.uniqueAnswer(cT, pt) {
			lc.intersections.insert(cT, lT, pt)
		}
	}
	lc.checkCoincident()
	return lc.intersections.used
}

// cubicHorizontalIntersect finds the cubic's t roots where its y-coordinate meets axisIntercept, re-solving via the y
// extrema/inflection partition when a root drifts off.
func cubicHorizontalIntersect(c dCubic, axisIntercept float64, roots []float64) int {
	a, b, cc, d := cubicCoefficients(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y)
	d -= axisIntercept
	count := cubicRootsValidT(a, b, cc, d, roots)
	for index := 0; index < count; index++ {
		calcPt := c.ptAtT(roots[index])
		if !approximatelyEqual(calcPt.y, axisIntercept) {
			var extremeTs [6]float64
			extrema := cubicFindExtrema(c.pts[0].y, c.pts[1].y, c.pts[2].y, c.pts[3].y, extremeTs[:])
			count = c.searchRoots(extremeTs[:], extrema, axisIntercept, yAxis, roots)
			break
		}
	}
	return count
}

// horizontalIntersect finds all intersections between the cubic and a horizontal line segment [left,right] at
// y=axisIntercept, including exact and near endpoint matches and coincident runs.
func (lc *lineCubicIntersections) horizontalIntersect(axisIntercept, left, right float64, flipped bool) int {
	lc.addExactHorizontalEndPoints(left, right, axisIntercept)
	if lc.allowNear {
		lc.addNearHorizontalEndPoints(left, right, axisIntercept)
	}
	var roots [3]float64
	count := cubicHorizontalIntersect(lc.cubic, axisIntercept, roots[:])
	for index := 0; index < count; index++ {
		cubicT := roots[index]
		pt := dPoint{x: lc.cubic.ptAtT(cubicT).x, y: axisIntercept}
		lineT := (pt.x - left) / (right - left)
		ok, cT, lT, outPt := lc.pinTs(cubicT, lineT, pt, pointInitialized)
		if ok && lc.uniqueAnswer(cT, outPt) {
			lc.intersections.insert(cT, lT, outPt)
		}
	}
	if flipped {
		lc.intersections.flip()
	}
	lc.checkCoincident()
	return lc.intersections.used
}

// uniqueAnswer reports whether (cubicT, pt) is a genuinely new intersection, rejecting it if an existing result already
// lands at the same point (directly, or via a coincident midpoint).
func (lc *lineCubicIntersections) uniqueAnswer(cubicT float64, pt dPoint) bool {
	for inner := 0; inner < lc.intersections.used; inner++ {
		if !lc.intersections.pt(inner).equals(pt) {
			continue
		}
		existingCubicT := lc.intersections.ts[0][inner]
		if cubicT == existingCubicT {
			return false
		}
		// check if midway on cubic is also same point. If so, discard this
		cubicMidT := (existingCubicT + cubicT) / 2
		cubicMidPt := lc.cubic.ptAtT(cubicMidT)
		if cubicMidPt.approximatelyEqual(pt) {
			return false
		}
	}
	return true
}

// cubicVerticalIntersect finds the cubic's t roots where its x-coordinate meets axisIntercept, re-solving via the x
// extrema/inflection partition when a root drifts off.
func cubicVerticalIntersect(c dCubic, axisIntercept float64, roots []float64) int {
	a, b, cc, d := cubicCoefficients(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x)
	d -= axisIntercept
	count := cubicRootsValidT(a, b, cc, d, roots)
	for index := 0; index < count; index++ {
		calcPt := c.ptAtT(roots[index])
		if !approximatelyEqual(calcPt.x, axisIntercept) {
			var extremeTs [6]float64
			extrema := cubicFindExtrema(c.pts[0].x, c.pts[1].x, c.pts[2].x, c.pts[3].x, extremeTs[:])
			count = c.searchRoots(extremeTs[:], extrema, axisIntercept, xAxis, roots)
			break
		}
	}
	return count
}

// verticalIntersect finds all intersections between the cubic and a vertical line segment [top,bottom] at
// x=axisIntercept, including exact and near endpoint matches and coincident runs.
func (lc *lineCubicIntersections) verticalIntersect(axisIntercept, top, bottom float64, flipped bool) int {
	lc.addExactVerticalEndPoints(top, bottom, axisIntercept)
	if lc.allowNear {
		lc.addNearVerticalEndPoints(top, bottom, axisIntercept)
	}
	var roots [3]float64
	count := cubicVerticalIntersect(lc.cubic, axisIntercept, roots[:])
	for index := 0; index < count; index++ {
		cubicT := roots[index]
		pt := dPoint{x: axisIntercept, y: lc.cubic.ptAtT(cubicT).y}
		lineT := (pt.y - top) / (bottom - top)
		ok, cT, lT, outPt := lc.pinTs(cubicT, lineT, pt, pointInitialized)
		if ok && lc.uniqueAnswer(cT, outPt) {
			lc.intersections.insert(cT, lT, outPt)
		}
	}
	if flipped {
		lc.intersections.flip()
	}
	lc.checkCoincident()
	return lc.intersections.used
}

// addExactEndPoints adds the cubic endpoints that lie exactly on the line first, so the 0/1 cubic t values are exact.
func (lc *lineCubicIntersections) addExactEndPoints() {
	for cIndex := 0; cIndex < 4; cIndex += 3 {
		lineT := lc.line.exactPoint(lc.cubic.pts[cIndex])
		if lineT < 0 {
			continue
		}
		cubicT := float64(cIndex >> 1)
		lc.intersections.insert(cubicT, lineT, lc.cubic.pts[cIndex])
	}
}

// addNearEndPoints adds the cubic endpoints that lie near (but not exactly on) the line, then adds the line's endpoints
// that lie near the cubic.
func (lc *lineCubicIntersections) addNearEndPoints() {
	for cIndex := 0; cIndex < 4; cIndex += 3 {
		cubicT := float64(cIndex >> 1)
		if lc.intersections.hasT(cubicT) {
			continue
		}
		lineT := lc.line.nearPoint(lc.cubic.pts[cIndex], nil)
		if lineT < 0 {
			continue
		}
		lc.intersections.insert(cubicT, lineT, lc.cubic.pts[cIndex])
	}
	lc.addLineNearEndPoints()
}

// addLineNearEndPoints adds the line's endpoints that lie near the cubic.
func (lc *lineCubicIntersections) addLineNearEndPoints() {
	for lIndex := 0; lIndex < 2; lIndex++ {
		lineT := float64(lIndex)
		if lc.intersections.hasOppT(lineT) {
			continue
		}
		cubicT := nearPointForCubic(lc.cubic, lc.line.pts[lIndex], lc.line.pts[1-lIndex])
		if cubicT < 0 {
			continue
		}
		lc.intersections.insert(cubicT, lineT, lc.line.pts[lIndex])
	}
}

// addExactHorizontalEndPoints adds the cubic endpoints that lie exactly on the horizontal line segment [left,right] at
// height y.
func (lc *lineCubicIntersections) addExactHorizontalEndPoints(left, right, y float64) {
	for cIndex := 0; cIndex < 4; cIndex += 3 {
		lineT := exactPointH(lc.cubic.pts[cIndex], left, right, y)
		if lineT < 0 {
			continue
		}
		cubicT := float64(cIndex >> 1)
		lc.intersections.insert(cubicT, lineT, lc.cubic.pts[cIndex])
	}
}

// addNearHorizontalEndPoints adds the cubic endpoints that lie near the horizontal line segment [left,right] at height
// y, then adds the line's endpoints that lie near the cubic.
func (lc *lineCubicIntersections) addNearHorizontalEndPoints(left, right, y float64) {
	for cIndex := 0; cIndex < 4; cIndex += 3 {
		cubicT := float64(cIndex >> 1)
		if lc.intersections.hasT(cubicT) {
			continue
		}
		lineT := nearPointH(lc.cubic.pts[cIndex], left, right, y)
		if lineT < 0 {
			continue
		}
		lc.intersections.insert(cubicT, lineT, lc.cubic.pts[cIndex])
	}
	lc.addLineNearEndPoints()
}

// addExactVerticalEndPoints adds the cubic endpoints that lie exactly on the vertical line segment [top,bottom] at x.
func (lc *lineCubicIntersections) addExactVerticalEndPoints(top, bottom, x float64) {
	for cIndex := 0; cIndex < 4; cIndex += 3 {
		lineT := exactPointV(lc.cubic.pts[cIndex], top, bottom, x)
		if lineT < 0 {
			continue
		}
		cubicT := float64(cIndex >> 1)
		lc.intersections.insert(cubicT, lineT, lc.cubic.pts[cIndex])
	}
}

// addNearVerticalEndPoints adds the cubic endpoints that lie near the vertical line segment [top,bottom] at x, then
// adds the line's endpoints that lie near the cubic.
func (lc *lineCubicIntersections) addNearVerticalEndPoints(top, bottom, x float64) {
	for cIndex := 0; cIndex < 4; cIndex += 3 {
		cubicT := float64(cIndex >> 1)
		if lc.intersections.hasT(cubicT) {
			continue
		}
		lineT := nearPointV(lc.cubic.pts[cIndex], top, bottom, x)
		if lineT < 0 {
			continue
		}
		lc.intersections.insert(cubicT, lineT, lc.cubic.pts[cIndex])
	}
	lc.addLineNearEndPoints()
}

// findLineT returns the line's parameter t at which the line passes through the cubic point at parameter t.
func (lc *lineCubicIntersections) findLineT(t float64) float64 {
	xy := lc.cubic.ptAtT(t)
	dx := lc.line.pts[1].x - lc.line.pts[0].x
	dy := lc.line.pts[1].y - lc.line.pts[0].y
	if math.Abs(dx) > math.Abs(dy) {
		return (xy.x - lc.line.pts[0].x) / dx
	}
	return (xy.y - lc.line.pts[0].y) / dy
}

// pinTs pins the t values into [0,1], requires the pinned line and cubic points to roughly agree, snaps the point to a
// curve endpoint when it lands on one, and rejects out-of-range answers. Returns the (possibly updated) cubicT, lineT,
// and point.
func (lc *lineCubicIntersections) pinTs(cubicT, lineT float64, pt dPoint, ptSet pinTPoint) (ok bool, newCubicT, newLineT float64, newPt dPoint) {
	if !approximatelyOneOrLess(lineT) {
		return false, cubicT, lineT, pt
	}
	if !approximatelyZeroOrMore(lineT) {
		return false, cubicT, lineT, pt
	}
	cT := pinT(cubicT)
	cubicT = cT
	lT := pinT(lineT)
	lineT = lT
	lPt := lc.line.ptAtT(lT)
	cPt := lc.cubic.ptAtT(cT)
	if !lPt.roughlyEqual(cPt) {
		return false, cubicT, lineT, pt
	}
	// The t values are accepted at roughlyEqual granularity: the check above only established that the two points are
	// roughly equal, so a pair that is roughly but not approximately equal keeps whichever t the solver produced. The
	// quad/quad lane refines such a pair with a binary search; this lane does not, and so reports a coarser t.
	if lT == 0 || lT == 1 || (ptSet == pointUninitialized && cT != 0 && cT != 1) {
		pt = lPt
	} else if ptSet == pointUninitialized {
		pt = cPt
	}
	gridPt := pt.asPoint()
	if gridPt == lc.line.pts[0].asPoint() {
		lineT = 0
	} else if gridPt == lc.line.pts[1].asPoint() {
		lineT = 1
	}
	if gridPt == lc.cubic.pts[0].asPoint() && approximatelyEqual(cubicT, 0) {
		cubicT = 0
	} else if gridPt == lc.cubic.pts[3].asPoint() && approximatelyEqual(cubicT, 1) {
		cubicT = 1
	}
	return true, cubicT, lineT, pt
}

// horizontalCubic intersects cubic with the horizontal line segment [left,right] at height y, recording results into
// in.
func (in *intersections) horizontalCubic(cubic dCubic, left, right, y float64, flipped bool) int {
	line := dLine{pts: [2]dPoint{{x: left, y: y}, {x: right, y: y}}}
	c := newLineCubicIntersections(cubic, &line, in)
	return c.horizontalIntersect(y, left, right, flipped)
}

// verticalCubic intersects cubic with the vertical line segment [top,bottom] at x, recording results into in.
func (in *intersections) verticalCubic(cubic dCubic, top, bottom, x float64, flipped bool) int {
	line := dLine{pts: [2]dPoint{{x: x, y: top}, {x: x, y: bottom}}}
	c := newLineCubicIntersections(cubic, &line, in)
	return c.verticalIntersect(x, top, bottom, flipped)
}

// intersectCubicLine intersects cubic with the general line, recording results into in.
func (in *intersections) intersectCubicLine(cubic dCubic, line dLine) int {
	c := newLineCubicIntersections(cubic, &line, in)
	c.setAllowNear(in.allowNear)
	return c.intersect()
}

// intersectRayCubic computes the raw roots of the cubic crossing the infinite line through line's two points, without
// endpoint or coincidence handling; the resulting points are evaluated from the cubic.
func (in *intersections) intersectRayCubic(cubic dCubic, line dLine) int {
	c := newLineCubicIntersections(cubic, &line, in)
	in.used = c.intersectRay(in.ts[0][:])
	for index := 0; index < in.used; index++ {
		in.pts[index] = cubic.ptAtT(in.ts[0][index])
	}
	return in.used
}

// nearPointForCubic drops a perpendicular from xy (using opp to orient it) and ray-intersects the cubic, returning the
// t of the nearest crossing if it is within ULP tolerance of xy, else -1.
func nearPointForCubic(curve dCubic, xy, opp dPoint) float64 {
	minX := curve.pts[0].x
	maxX := minX
	for index := 1; index <= 3; index++ {
		minX = math.Min(minX, curve.pts[index].x)
		maxX = math.Max(maxX, curve.pts[index].x)
	}
	if !almostBetweenUlps(minX, xy.x, maxX) {
		return -1
	}
	minY := curve.pts[0].y
	maxY := minY
	for index := 1; index <= 3; index++ {
		minY = math.Min(minY, curve.pts[index].y)
		maxY = math.Max(maxY, curve.pts[index].y)
	}
	if !almostBetweenUlps(minY, xy.y, maxY) {
		return -1
	}
	i := newIntersections()
	perp := dLine{pts: [2]dPoint{xy, {x: xy.x + opp.y - xy.y, y: xy.y + xy.x - opp.x}}}
	i.intersectRayCubic(curve, perp)
	minIndex := -1
	minDist := math.MaxFloat32
	for index := 0; index < i.used; index++ {
		dist := xy.distance(i.pt(index))
		if minDist > dist {
			minDist = dist
			minIndex = index
		}
	}
	if minIndex < 0 {
		return -1
	}
	largest := math.Max(math.Max(maxX, maxY), -math.Min(minX, minY))
	if !almostEqualUlpsPin(largest, largest+minDist) { // is distance within ULPS tolerance?
		return -1
	}
	return pinT(i.ts[0][minIndex])
}
