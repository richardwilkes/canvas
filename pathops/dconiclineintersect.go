// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The conic/line intersection solver (the general intersect plus the axis-aligned horizontal/vertical lanes and the ray
// form), implemented by the lineConicIntersections struct. Also implements the conic case of near-endpoint handling
// (nearPointForConic): it drops a perpendicular from a point and ray-intersects the conic.

package pathops

import "math"

// lineConicIntersections solves for the intersection t values between a conic and a line.
type lineConicIntersections struct {
	line          *dLine
	intersections *intersections
	conic         dConic
	allowNear     bool
}

// newLineConicIntersections builds a solver for conic c against line l, storing results in i. The result cap is set to
// 4 to allow a short partial coincidence plus a discrete intersection.
func newLineConicIntersections(c dConic, l *dLine, i *intersections) *lineConicIntersections {
	i.setMax(4)
	return &lineConicIntersections{conic: c, line: l, intersections: i, allowNear: true}
}

// setAllowNear controls whether near (not just exact) endpoint matches are accepted.
func (lc *lineConicIntersections) setAllowNear(allow bool) { lc.allowNear = allow }

// checkCoincident scans adjacent intersection results and, when the conic's midpoint between them also lies on the
// line, marks the pair (or folds one into the other) as a coincident run rather than two discrete crossings.
func (lc *lineConicIntersections) checkCoincident() {
	last := lc.intersections.used - 1
	for index := 0; index < last; {
		conicMidT := (lc.intersections.ts[0][index] + lc.intersections.ts[0][index+1]) / 2
		conicMidPt := lc.conic.ptAtT(conicMidT)
		t := lc.line.nearPoint(conicMidPt, nil)
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

// validT forms the quadratic A*t^2 + 2B*t + C from the three rotated coordinate values (r0/r1/r2) and the conic's
// weight, then solves for its valid t roots. The weight is a float32 that promotes to double precision in the products,
// so B is computed in double.
func (lc *lineConicIntersections) validT(r0, r1, r2, axisIntercept float64, roots []float64) int {
	a := r2
	b := r1*float64(lc.conic.weight) - axisIntercept*float64(lc.conic.weight) + axisIntercept
	c := r0
	a += c - 2*b // A = a + c - 2*(b*w - xCept*w + xCept)
	b -= c       // B = b*w - w*xCept + xCept - a
	c -= axisIntercept
	return quadRootsValidT(a, 2*b, c, roots)
}

// horizontalIntersectRoots finds the conic's t values where its y-coordinate equals axisIntercept.
func (lc *lineConicIntersections) horizontalIntersectRoots(axisIntercept float64, roots []float64) int {
	return lc.validT(lc.conic.pts.pts[0].y, lc.conic.pts.pts[1].y, lc.conic.pts.pts[2].y, axisIntercept, roots)
}

// verticalIntersectRoots finds the conic's t values where its x-coordinate equals axisIntercept.
func (lc *lineConicIntersections) verticalIntersectRoots(axisIntercept float64, roots []float64) int {
	return lc.validT(lc.conic.pts.pts[0].x, lc.conic.pts.pts[1].x, lc.conic.pts.pts[2].x, axisIntercept, roots)
}

// rayRoots rotates the conic's control points into the line's frame (a signed cross product against the line direction)
// and solves for the t roots where the conic crosses the line.
func (lc *lineConicIntersections) rayRoots(roots []float64) int {
	adj := lc.line.pts[1].x - lc.line.pts[0].x
	opp := lc.line.pts[1].y - lc.line.pts[0].y
	var r [3]float64
	for n := 0; n < 3; n++ {
		r[n] = (lc.conic.pts.pts[n].y-lc.line.pts[0].y)*adj - (lc.conic.pts.pts[n].x-lc.line.pts[0].x)*opp
	}
	return lc.validT(r[0], r[1], r[2], 0, roots)
}

// intersect finds all intersections between the conic and a general (non-axis-aligned) line, including exact and near
// endpoint matches and coincident runs.
func (lc *lineConicIntersections) intersect() int {
	lc.addExactEndPoints()
	if lc.allowNear {
		lc.addNearEndPoints()
	}
	var rootVals [2]float64
	roots := lc.rayRoots(rootVals[:])
	for index := 0; index < roots; index++ {
		conicT := rootVals[index]
		lineT := lc.findLineT(conicT)
		ok, cT, lT, pt := lc.pinTs(conicT, lineT, dPoint{}, pointUninitialized)
		if ok && lc.uniqueAnswer(cT, pt) {
			lc.intersections.insert(cT, lT, pt)
		}
	}
	lc.checkCoincident()
	return lc.intersections.used
}

// horizontalIntersect finds all intersections between the conic and a horizontal line segment [left,right] at
// y=axisIntercept, including exact and near endpoint matches and coincident runs.
func (lc *lineConicIntersections) horizontalIntersect(axisIntercept, left, right float64, flipped bool) int {
	lc.addExactHorizontalEndPoints(left, right, axisIntercept)
	if lc.allowNear {
		lc.addNearHorizontalEndPoints(left, right, axisIntercept)
	}
	var rootVals [2]float64
	count := lc.horizontalIntersectRoots(axisIntercept, rootVals[:])
	for index := 0; index < count; index++ {
		conicT := rootVals[index]
		pt := lc.conic.ptAtT(conicT)
		lineT := (pt.x - left) / (right - left)
		ok, cT, lT, outPt := lc.pinTs(conicT, lineT, pt, pointInitialized)
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

// verticalIntersect finds all intersections between the conic and a vertical line segment [top,bottom] at
// x=axisIntercept, including exact and near endpoint matches and coincident runs.
func (lc *lineConicIntersections) verticalIntersect(axisIntercept, top, bottom float64, flipped bool) int {
	lc.addExactVerticalEndPoints(top, bottom, axisIntercept)
	if lc.allowNear {
		lc.addNearVerticalEndPoints(top, bottom, axisIntercept)
	}
	var rootVals [2]float64
	count := lc.verticalIntersectRoots(axisIntercept, rootVals[:])
	for index := 0; index < count; index++ {
		conicT := rootVals[index]
		pt := lc.conic.ptAtT(conicT)
		lineT := (pt.y - top) / (bottom - top)
		ok, cT, lT, outPt := lc.pinTs(conicT, lineT, pt, pointInitialized)
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

// uniqueAnswer reports whether (conicT, pt) is a genuinely new intersection, rejecting it if an existing result already
// lands at the same point (directly, or via a coincident midpoint).
func (lc *lineConicIntersections) uniqueAnswer(conicT float64, pt dPoint) bool {
	for inner := 0; inner < lc.intersections.used; inner++ {
		if !lc.intersections.pt(inner).equals(pt) {
			continue
		}
		existingConicT := lc.intersections.ts[0][inner]
		if conicT == existingConicT {
			return false
		}
		// check if midway on conic is also same point. If so, discard this
		conicMidT := (existingConicT + conicT) / 2
		conicMidPt := lc.conic.ptAtT(conicMidT)
		if conicMidPt.approximatelyEqual(pt) {
			return false
		}
	}
	return true
}

// addExactEndPoints adds the conic endpoints that lie exactly on the line first, so the 0/1 conic t values are exact.
func (lc *lineConicIntersections) addExactEndPoints() {
	for cIndex := 0; cIndex < 3; cIndex += 2 {
		lineT := lc.line.exactPoint(lc.conic.pts.pts[cIndex])
		if lineT < 0 {
			continue
		}
		conicT := float64(cIndex >> 1)
		lc.intersections.insert(conicT, lineT, lc.conic.pts.pts[cIndex])
	}
}

// addNearEndPoints adds the conic endpoints that lie near (but not exactly on) the line, then adds the line's endpoints
// that lie near the conic.
func (lc *lineConicIntersections) addNearEndPoints() {
	for cIndex := 0; cIndex < 3; cIndex += 2 {
		conicT := float64(cIndex >> 1)
		if lc.intersections.hasT(conicT) {
			continue
		}
		lineT := lc.line.nearPoint(lc.conic.pts.pts[cIndex], nil)
		if lineT < 0 {
			continue
		}
		lc.intersections.insert(conicT, lineT, lc.conic.pts.pts[cIndex])
	}
	lc.addLineNearEndPoints()
}

// addLineNearEndPoints adds the line's endpoints that lie near the conic.
func (lc *lineConicIntersections) addLineNearEndPoints() {
	for lIndex := 0; lIndex < 2; lIndex++ {
		lineT := float64(lIndex)
		if lc.intersections.hasOppT(lineT) {
			continue
		}
		conicT := nearPointForConic(lc.conic, lc.line.pts[lIndex], lc.line.pts[1-lIndex])
		if conicT < 0 {
			continue
		}
		lc.intersections.insert(conicT, lineT, lc.line.pts[lIndex])
	}
}

// addExactHorizontalEndPoints adds the conic endpoints that lie exactly on the horizontal line segment [left,right] at
// height y.
func (lc *lineConicIntersections) addExactHorizontalEndPoints(left, right, y float64) {
	for cIndex := 0; cIndex < 3; cIndex += 2 {
		lineT := exactPointH(lc.conic.pts.pts[cIndex], left, right, y)
		if lineT < 0 {
			continue
		}
		conicT := float64(cIndex >> 1)
		lc.intersections.insert(conicT, lineT, lc.conic.pts.pts[cIndex])
	}
}

// addNearHorizontalEndPoints adds the conic endpoints that lie near the horizontal line segment [left,right] at height
// y, then adds the line's endpoints that lie near the conic.
func (lc *lineConicIntersections) addNearHorizontalEndPoints(left, right, y float64) {
	for cIndex := 0; cIndex < 3; cIndex += 2 {
		conicT := float64(cIndex >> 1)
		if lc.intersections.hasT(conicT) {
			continue
		}
		lineT := nearPointH(lc.conic.pts.pts[cIndex], left, right, y)
		if lineT < 0 {
			continue
		}
		lc.intersections.insert(conicT, lineT, lc.conic.pts.pts[cIndex])
	}
	lc.addLineNearEndPoints()
}

// addExactVerticalEndPoints adds the conic endpoints that lie exactly on the vertical line segment [top,bottom] at x.
func (lc *lineConicIntersections) addExactVerticalEndPoints(top, bottom, x float64) {
	for cIndex := 0; cIndex < 3; cIndex += 2 {
		lineT := exactPointV(lc.conic.pts.pts[cIndex], top, bottom, x)
		if lineT < 0 {
			continue
		}
		conicT := float64(cIndex >> 1)
		lc.intersections.insert(conicT, lineT, lc.conic.pts.pts[cIndex])
	}
}

// addNearVerticalEndPoints adds the conic endpoints that lie near the vertical line segment [top,bottom] at x, then
// adds the line's endpoints that lie near the conic.
func (lc *lineConicIntersections) addNearVerticalEndPoints(top, bottom, x float64) {
	for cIndex := 0; cIndex < 3; cIndex += 2 {
		conicT := float64(cIndex >> 1)
		if lc.intersections.hasT(conicT) {
			continue
		}
		lineT := nearPointV(lc.conic.pts.pts[cIndex], top, bottom, x)
		if lineT < 0 {
			continue
		}
		lc.intersections.insert(conicT, lineT, lc.conic.pts.pts[cIndex])
	}
	lc.addLineNearEndPoints()
}

// findLineT returns the line's parameter t at which the line passes through the conic point at parameter t.
func (lc *lineConicIntersections) findLineT(t float64) float64 {
	xy := lc.conic.ptAtT(t)
	dx := lc.line.pts[1].x - lc.line.pts[0].x
	dy := lc.line.pts[1].y - lc.line.pts[0].y
	if math.Abs(dx) > math.Abs(dy) {
		return (xy.x - lc.line.pts[0].x) / dx
	}
	return (xy.y - lc.line.pts[0].y) / dy
}

// pinTs pins the t values into [0,1], snaps the point to a curve endpoint when it lands on one, and rejects
// out-of-range or duplicate answers. Returns the (possibly updated) conicT, lineT, and point.
func (lc *lineConicIntersections) pinTs(conicT, lineT float64, pt dPoint, ptSet pinTPoint) (ok bool, newConicT, newLineT float64, newPt dPoint) {
	if !approximatelyOneOrLessDouble(lineT) {
		return false, conicT, lineT, pt
	}
	if !approximatelyZeroOrMoreDouble(lineT) {
		return false, conicT, lineT, pt
	}
	qT := pinT(conicT)
	conicT = qT
	lT := pinT(lineT)
	lineT = lT
	if lT == 0 || lT == 1 || (ptSet == pointUninitialized && qT != 0 && qT != 1) {
		pt = lc.line.ptAtT(lT)
	} else if ptSet == pointUninitialized {
		pt = lc.conic.ptAtT(qT)
	}
	gridPt := pt.asPoint()
	if dPointsApproximatelyEqual(gridPt, lc.line.pts[0].asPoint()) {
		pt = lc.line.pts[0]
		lineT = 0
	} else if dPointsApproximatelyEqual(gridPt, lc.line.pts[1].asPoint()) {
		pt = lc.line.pts[1]
		lineT = 1
	}
	if lc.intersections.used > 0 && approximatelyEqual(lc.intersections.ts[1][0], lineT) {
		return false, conicT, lineT, pt
	}
	if gridPt == lc.conic.pts.pts[0].asPoint() {
		pt = lc.conic.pts.pts[0]
		conicT = 0
	} else if gridPt == lc.conic.pts.pts[2].asPoint() {
		pt = lc.conic.pts.pts[2]
		conicT = 1
	}
	return true, conicT, lineT, pt
}

// horizontalConic intersects conic with the horizontal line segment [left,right] at height y, recording results into
// in.
func (in *intersections) horizontalConic(conic dConic, left, right, y float64, flipped bool) int {
	line := dLine{pts: [2]dPoint{{x: left, y: y}, {x: right, y: y}}}
	c := newLineConicIntersections(conic, &line, in)
	return c.horizontalIntersect(y, left, right, flipped)
}

// verticalConic intersects conic with the vertical line segment [top,bottom] at x, recording results into in.
func (in *intersections) verticalConic(conic dConic, top, bottom, x float64, flipped bool) int {
	line := dLine{pts: [2]dPoint{{x: x, y: top}, {x: x, y: bottom}}}
	c := newLineConicIntersections(conic, &line, in)
	return c.verticalIntersect(x, top, bottom, flipped)
}

// intersectConicLine intersects conic with the general line, recording results into in.
func (in *intersections) intersectConicLine(conic dConic, line dLine) int {
	c := newLineConicIntersections(conic, &line, in)
	c.setAllowNear(in.allowNear)
	return c.intersect()
}

// intersectRayConic computes the raw roots of the conic crossing the infinite line through line's two points, without
// endpoint or coincidence handling; the resulting points are evaluated from the conic.
func (in *intersections) intersectRayConic(conic dConic, line dLine) int {
	c := newLineConicIntersections(conic, &line, in)
	in.used = c.rayRoots(in.ts[0][:])
	for index := 0; index < in.used; index++ {
		in.pts[index] = conic.ptAtT(in.ts[0][index])
	}
	return in.used
}

// nearPointForConic drops a perpendicular from xy (using opp to orient it) and ray-intersects the conic, returning the
// t of the nearest crossing if it is within ULP tolerance of xy, else -1.
func nearPointForConic(curve dConic, xy, opp dPoint) float64 {
	minX := curve.pts.pts[0].x
	maxX := minX
	for index := 1; index <= 2; index++ {
		minX = math.Min(minX, curve.pts.pts[index].x)
		maxX = math.Max(maxX, curve.pts.pts[index].x)
	}
	if !almostBetweenUlps(minX, xy.x, maxX) {
		return -1
	}
	minY := curve.pts.pts[0].y
	maxY := minY
	for index := 1; index <= 2; index++ {
		minY = math.Min(minY, curve.pts.pts[index].y)
		maxY = math.Max(maxY, curve.pts.pts[index].y)
	}
	if !almostBetweenUlps(minY, xy.y, maxY) {
		return -1
	}
	i := newIntersections()
	perp := dLine{pts: [2]dPoint{xy, {x: xy.x + opp.y - xy.y, y: xy.y + xy.x - opp.x}}}
	i.intersectRayConic(curve, perp)
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
