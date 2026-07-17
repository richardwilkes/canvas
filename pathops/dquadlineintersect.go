// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Quadratic/line intersection: the general intersect plus the axis-aligned horizontal/vertical lanes and the ray form,
// driven through the lineQuadraticIntersections helper. Also includes the quad case of the curve near-point test,
// reached by the near-endpoint handling: it drops a perpendicular from a point and ray-intersects the quad.

package pathops

import "math"

// pinTPoint reports whether the intersection point passed to pinTs has already been computed by the caller
// (pointInitialized) or still needs to be derived from the pinned T values (pointUninitialized).
type pinTPoint int

const (
	pointUninitialized pinTPoint = iota
	pointInitialized
)

// lineQuadraticIntersections drives the quad/line intersection: it holds the quad and line being tested, the
// accumulating intersections result, and whether near (not just exact) endpoint coincidences count.
type lineQuadraticIntersections struct {
	line          *dLine
	intersections *intersections
	quad          dQuad
	allowNear     bool
}

// newLineQuadraticIntersections builds a helper over q, l, and i, sizing i to allow up to five results (to cover short
// partial coincidence plus discrete intersections) and defaulting allowNear to true.
func newLineQuadraticIntersections(q dQuad, l *dLine, i *intersections) *lineQuadraticIntersections {
	i.setMax(5)
	return &lineQuadraticIntersections{quad: q, line: l, intersections: i, allowNear: true}
}

// setAllowNear sets whether near (not just exact) endpoint coincidences are reported.
func (lq *lineQuadraticIntersections) setAllowNear(allow bool) { lq.allowNear = allow }

// checkCoincident scans the accumulated intersections for adjacent pairs whose quad-T midpoint also lies on the line,
// marking them (or the surviving one, if one is dropped as redundant) coincident.
func (lq *lineQuadraticIntersections) checkCoincident() {
	last := lq.intersections.used - 1
	for index := 0; index < last; {
		quadMidT := (lq.intersections.ts[0][index] + lq.intersections.ts[0][index+1]) / 2
		quadMidPt := lq.quad.ptAtT(quadMidT)
		t := lq.line.nearPoint(quadMidPt, nil)
		if t < 0 {
			index++
			continue
		}
		switch {
		case lq.intersections.isCoincident(index):
			lq.intersections.removeOne(index)
			last--
		case lq.intersections.isCoincident(index + 1):
			lq.intersections.removeOne(index + 1)
			last--
		default:
			lq.intersections.setCoincident(index)
			index++
		}
		lq.intersections.setCoincident(index)
	}
}

// rayRoots rotates the line and quad so the line becomes horizontal, then solves the resulting quadratic for its T
// roots on the quad.
func (lq *lineQuadraticIntersections) rayRoots(roots []float64) int {
	adj := lq.line.pts[1].x - lq.line.pts[0].x
	opp := lq.line.pts[1].y - lq.line.pts[0].y
	var r [3]float64
	for n := 0; n < 3; n++ {
		r[n] = (lq.quad.pts[n].y-lq.line.pts[0].y)*adj - (lq.quad.pts[n].x-lq.line.pts[0].x)*opp
	}
	a := r[2]
	b := r[1]
	c := r[0]
	a += c - 2*b // A = a - 2*b + c
	b -= c       // B = -(b - c)
	return quadRootsValidT(a, 2*b, c, roots)
}

// intersect computes the full quad/line intersection: exact and near endpoint coincidences first, then the ray-root
// crossings (pinned, deduplicated, and endpoint-snapped), then a final coincidence sweep.
func (lq *lineQuadraticIntersections) intersect() int {
	lq.addExactEndPoints()
	if lq.allowNear {
		lq.addNearEndPoints()
	}
	var rootVals [2]float64
	roots := lq.rayRoots(rootVals[:])
	for index := 0; index < roots; index++ {
		quadT := rootVals[index]
		lineT := lq.findLineT(quadT)
		ok, qT, lT, pt := lq.pinTs(quadT, lineT, dPoint{}, pointUninitialized)
		if ok && lq.uniqueAnswer(qT, pt) {
			lq.intersections.insert(qT, lT, pt)
		}
	}
	lq.checkCoincident()
	return lq.intersections.used
}

// horizontalIntersectRoots solves the quad's y(t) - axisIntercept = 0 for its T roots, i.e. where the quad crosses the
// horizontal line y=axisIntercept.
func (lq *lineQuadraticIntersections) horizontalIntersectRoots(axisIntercept float64, roots []float64) int {
	d := lq.quad.pts[2].y  // f
	e := lq.quad.pts[1].y  // e
	fv := lq.quad.pts[0].y // d
	d += fv - 2*e          // D = d - 2*e + f
	e -= fv                // E = -(d - e)
	fv -= axisIntercept
	return quadRootsValidT(d, 2*e, fv, roots)
}

// horizontalIntersect intersects the quad with the horizontal segment y=axisIntercept spanning x=left to x=right, with
// T on that segment flipped when requested.
func (lq *lineQuadraticIntersections) horizontalIntersect(axisIntercept, left, right float64, flipped bool) int {
	lq.addExactHorizontalEndPoints(left, right, axisIntercept)
	if lq.allowNear {
		lq.addNearHorizontalEndPoints(left, right, axisIntercept)
	}
	var rootVals [2]float64
	roots := lq.horizontalIntersectRoots(axisIntercept, rootVals[:])
	for index := 0; index < roots; index++ {
		quadT := rootVals[index]
		pt := lq.quad.ptAtT(quadT)
		lineT := (pt.x - left) / (right - left)
		ok, qT, lT, outPt := lq.pinTs(quadT, lineT, pt, pointInitialized)
		if ok && lq.uniqueAnswer(qT, outPt) {
			lq.intersections.insert(qT, lT, outPt)
		}
	}
	if flipped {
		lq.intersections.flip()
	}
	lq.checkCoincident()
	return lq.intersections.used
}

// uniqueAnswer reports whether (quadT, pt) is not already represented among the accumulated intersections — either by
// an identical quad T at the same point, or by the quad's midpoint between the two T values also landing on pt (which
// would make the new answer redundant).
func (lq *lineQuadraticIntersections) uniqueAnswer(quadT float64, pt dPoint) bool {
	for inner := 0; inner < lq.intersections.used; inner++ {
		if !lq.intersections.pt(inner).equals(pt) {
			continue
		}
		existingQuadT := lq.intersections.ts[0][inner]
		if quadT == existingQuadT {
			return false
		}
		// check if midway on quad is also same point. If so, discard this
		quadMidT := (existingQuadT + quadT) / 2
		quadMidPt := lq.quad.ptAtT(quadMidT)
		if quadMidPt.approximatelyEqual(pt) {
			return false
		}
	}
	return true
}

// verticalIntersectRoots solves the quad's x(t) - axisIntercept = 0 for its T roots, i.e. where the quad crosses the
// vertical line x=axisIntercept.
func (lq *lineQuadraticIntersections) verticalIntersectRoots(axisIntercept float64, roots []float64) int {
	d := lq.quad.pts[2].x  // f
	e := lq.quad.pts[1].x  // e
	fv := lq.quad.pts[0].x // d
	d += fv - 2*e          // D = d - 2*e + f
	e -= fv                // E = -(d - e)
	fv -= axisIntercept
	return quadRootsValidT(d, 2*e, fv, roots)
}

// verticalIntersect intersects the quad with the vertical segment x=axisIntercept spanning y=top to y=bottom, with T on
// that segment flipped when requested.
func (lq *lineQuadraticIntersections) verticalIntersect(axisIntercept, top, bottom float64, flipped bool) int {
	lq.addExactVerticalEndPoints(top, bottom, axisIntercept)
	if lq.allowNear {
		lq.addNearVerticalEndPoints(top, bottom, axisIntercept)
	}
	var rootVals [2]float64
	roots := lq.verticalIntersectRoots(axisIntercept, rootVals[:])
	for index := 0; index < roots; index++ {
		quadT := rootVals[index]
		pt := lq.quad.ptAtT(quadT)
		lineT := (pt.y - top) / (bottom - top)
		ok, qT, lT, outPt := lq.pinTs(quadT, lineT, pt, pointInitialized)
		if ok && lq.uniqueAnswer(qT, outPt) {
			lq.intersections.insert(qT, lT, outPt)
		}
	}
	if flipped {
		lq.intersections.flip()
	}
	lq.checkCoincident()
	return lq.intersections.used
}

// addExactEndPoints adds the quad endpoints that lie exactly on the line first, so the 0/1 quad T values are recorded
// exactly rather than recovered later from an approximate root.
func (lq *lineQuadraticIntersections) addExactEndPoints() {
	for qIndex := 0; qIndex < 3; qIndex += 2 {
		lineT := lq.line.exactPoint(lq.quad.pts[qIndex])
		if lineT < 0 {
			continue
		}
		quadT := float64(qIndex >> 1)
		lq.intersections.insert(quadT, lineT, lq.quad.pts[qIndex])
	}
}

// addNearEndPoints adds quad endpoints (0 and 1) that lie only near, not exactly on, the line, plus the line endpoints
// that lie near the quad.
func (lq *lineQuadraticIntersections) addNearEndPoints() {
	for qIndex := 0; qIndex < 3; qIndex += 2 {
		quadT := float64(qIndex >> 1)
		if lq.intersections.hasT(quadT) {
			continue
		}
		lineT := lq.line.nearPoint(lq.quad.pts[qIndex], nil)
		if lineT < 0 {
			continue
		}
		lq.intersections.insert(quadT, lineT, lq.quad.pts[qIndex])
	}
	lq.addLineNearEndPoints()
}

// addLineNearEndPoints adds the line's endpoints that lie near (but not necessarily on) the quad.
func (lq *lineQuadraticIntersections) addLineNearEndPoints() {
	for lIndex := 0; lIndex < 2; lIndex++ {
		lineT := float64(lIndex)
		if lq.intersections.hasOppT(lineT) {
			continue
		}
		quadT := nearPointForQuad(lq.quad, lq.line.pts[lIndex], lq.line.pts[1-lIndex])
		if quadT < 0 {
			continue
		}
		lq.intersections.insert(quadT, lineT, lq.line.pts[lIndex])
	}
}

// addExactHorizontalEndPoints adds the quad endpoints that lie exactly on the horizontal segment [left,right] at y.
func (lq *lineQuadraticIntersections) addExactHorizontalEndPoints(left, right, y float64) {
	for qIndex := 0; qIndex < 3; qIndex += 2 {
		lineT := exactPointH(lq.quad.pts[qIndex], left, right, y)
		if lineT < 0 {
			continue
		}
		quadT := float64(qIndex >> 1)
		lq.intersections.insert(quadT, lineT, lq.quad.pts[qIndex])
	}
}

// addNearHorizontalEndPoints adds quad endpoints that lie only near the horizontal segment [left,right] at y, plus the
// line's own near-endpoints.
func (lq *lineQuadraticIntersections) addNearHorizontalEndPoints(left, right, y float64) {
	for qIndex := 0; qIndex < 3; qIndex += 2 {
		quadT := float64(qIndex >> 1)
		if lq.intersections.hasT(quadT) {
			continue
		}
		lineT := nearPointH(lq.quad.pts[qIndex], left, right, y)
		if lineT < 0 {
			continue
		}
		lq.intersections.insert(quadT, lineT, lq.quad.pts[qIndex])
	}
	lq.addLineNearEndPoints()
}

// addExactVerticalEndPoints adds the quad endpoints that lie exactly on the vertical segment [top,bottom] at x.
func (lq *lineQuadraticIntersections) addExactVerticalEndPoints(top, bottom, x float64) {
	for qIndex := 0; qIndex < 3; qIndex += 2 {
		lineT := exactPointV(lq.quad.pts[qIndex], top, bottom, x)
		if lineT < 0 {
			continue
		}
		quadT := float64(qIndex >> 1)
		lq.intersections.insert(quadT, lineT, lq.quad.pts[qIndex])
	}
}

// addNearVerticalEndPoints adds quad endpoints that lie only near the vertical segment [top,bottom] at x, plus the
// line's own near-endpoints.
func (lq *lineQuadraticIntersections) addNearVerticalEndPoints(top, bottom, x float64) {
	for qIndex := 0; qIndex < 3; qIndex += 2 {
		quadT := float64(qIndex >> 1)
		if lq.intersections.hasT(quadT) {
			continue
		}
		lineT := nearPointV(lq.quad.pts[qIndex], top, bottom, x)
		if lineT < 0 {
			continue
		}
		lq.intersections.insert(quadT, lineT, lq.quad.pts[qIndex])
	}
	lq.addLineNearEndPoints()
}

// findLineT returns the line's T value at the point the quad reaches at parameter t, projecting along whichever axis
// the line varies more in (to avoid dividing by a near-zero span).
func (lq *lineQuadraticIntersections) findLineT(t float64) float64 {
	xy := lq.quad.ptAtT(t)
	dx := lq.line.pts[1].x - lq.line.pts[0].x
	dy := lq.line.pts[1].y - lq.line.pts[0].y
	if math.Abs(dx) > math.Abs(dy) {
		return (xy.x - lq.line.pts[0].x) / dx
	}
	return (xy.y - lq.line.pts[0].y) / dy
}

// pinTs pins quadT and lineT into range, snaps the point to a curve endpoint when it lands on one, and rejects
// out-of-range or duplicate answers. Returns the (possibly updated) quadT, lineT, and point.
func (lq *lineQuadraticIntersections) pinTs(quadT, lineT float64, pt dPoint, ptSet pinTPoint) (ok bool, newQuadT, newLineT float64, newPt dPoint) {
	if !approximatelyOneOrLessDouble(lineT) {
		return false, quadT, lineT, pt
	}
	if !approximatelyZeroOrMoreDouble(lineT) {
		return false, quadT, lineT, pt
	}
	qT := pinT(quadT)
	quadT = qT
	lT := pinT(lineT)
	lineT = lT
	if lT == 0 || lT == 1 || (ptSet == pointUninitialized && qT != 0 && qT != 1) {
		pt = lq.line.ptAtT(lT)
	} else if ptSet == pointUninitialized {
		pt = lq.quad.ptAtT(qT)
	}
	gridPt := pt.asPoint()
	if dPointsApproximatelyEqual(gridPt, lq.line.pts[0].asPoint()) {
		pt = lq.line.pts[0]
		lineT = 0
	} else if dPointsApproximatelyEqual(gridPt, lq.line.pts[1].asPoint()) {
		pt = lq.line.pts[1]
		lineT = 1
	}
	if lq.intersections.used > 0 && approximatelyEqual(lq.intersections.ts[1][0], lineT) {
		return false, quadT, lineT, pt
	}
	if gridPt == lq.quad.pts[0].asPoint() {
		pt = lq.quad.pts[0]
		quadT = 0
	} else if gridPt == lq.quad.pts[2].asPoint() {
		pt = lq.quad.pts[2]
		quadT = 1
	}
	return true, quadT, lineT, pt
}

// horizontalQuad intersects quad with the horizontal segment y=y spanning x=left to x=right.
func (in *intersections) horizontalQuad(quad dQuad, left, right, y float64, flipped bool) int {
	line := dLine{pts: [2]dPoint{{x: left, y: y}, {x: right, y: y}}}
	q := newLineQuadraticIntersections(quad, &line, in)
	return q.horizontalIntersect(y, left, right, flipped)
}

// verticalQuad intersects quad with the vertical segment x=x spanning y=top to y=bottom.
func (in *intersections) verticalQuad(quad dQuad, top, bottom, x float64, flipped bool) int {
	line := dLine{pts: [2]dPoint{{x: x, y: top}, {x: x, y: bottom}}}
	q := newLineQuadraticIntersections(quad, &line, in)
	return q.verticalIntersect(x, top, bottom, flipped)
}

// intersectQuadLine intersects quad with line in general (non-axis-aligned) position.
func (in *intersections) intersectQuadLine(quad dQuad, line dLine) int {
	q := newLineQuadraticIntersections(quad, &line, in)
	q.setAllowNear(in.allowNear)
	return q.intersect()
}

// intersectRayQuad computes the raw quad/line ray roots without endpoint or coincidence handling, with the resulting
// points computed from the quad.
func (in *intersections) intersectRayQuad(quad dQuad, line dLine) int {
	q := newLineQuadraticIntersections(quad, &line, in)
	in.used = q.rayRoots(in.ts[0][:])
	for index := 0; index < in.used; index++ {
		in.pts[index] = quad.ptAtT(in.ts[0][index])
	}
	return in.used
}

// nearPointForQuad drops a perpendicular from xy (using opp to orient it) and ray-intersects curve, returning the T of
// the nearest crossing if it is within ULPs tolerance of xy, else -1.
func nearPointForQuad(curve dQuad, xy, opp dPoint) float64 {
	minX := curve.pts[0].x
	maxX := minX
	for index := 1; index <= 2; index++ {
		minX = math.Min(minX, curve.pts[index].x)
		maxX = math.Max(maxX, curve.pts[index].x)
	}
	if !almostBetweenUlps(minX, xy.x, maxX) {
		return -1
	}
	minY := curve.pts[0].y
	maxY := minY
	for index := 1; index <= 2; index++ {
		minY = math.Min(minY, curve.pts[index].y)
		maxY = math.Max(maxY, curve.pts[index].y)
	}
	if !almostBetweenUlps(minY, xy.y, maxY) {
		return -1
	}
	i := newIntersections()
	perp := dLine{pts: [2]dPoint{xy, {x: xy.x + opp.y - xy.y, y: xy.y + xy.x - opp.x}}}
	i.intersectRayQuad(curve, perp)
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
