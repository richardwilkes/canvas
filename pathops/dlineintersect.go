// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Line/line intersection: the bounded intersect (intersectLineLine), the unbounded ray form (intersectRayLine — its
// first curve consumer is the quad subdivision control-point recovery in dquad.go's subDivideCtrl), the axis-aligned
// horizontal/vertical lanes, and the coincident-line cleanup. These are the base cases the curve-vs-line intersections
// reduce to.

package pathops

import "math"

// cleanUpParallelLines collapses an over-full intersection result to at most two entries and, for coincident lines,
// marks both endpoints coincident.
func (in *intersections) cleanUpParallelLines(parallel bool) {
	for in.used > 2 {
		in.removeOne(1)
	}
	if in.used == 2 && !parallel {
		startMatch := in.ts[0][0] == 0 || zeroOrOne(in.ts[1][0])
		endMatch := in.ts[0][1] == 1 || zeroOrOne(in.ts[1][1])
		if (!startMatch && !endMatch) || approximatelyEqual(in.ts[0][0], in.ts[0][1]) {
			if startMatch && endMatch && (in.ts[0][0] != 0 || !zeroOrOne(in.ts[1][0])) &&
				in.ts[0][1] == 1 && zeroOrOne(in.ts[1][1]) {
				in.removeOne(0)
			} else {
				in.removeOne(b2i(endMatch))
			}
		}
	}
	if in.used == 2 {
		in.isCoin[0] = 0x03
		in.isCoin[1] = 0x03
	}
}

// computePoints fills in.pts from the T values already stored in in.ts[0], evaluating them against line, and sets
// in.used.
func (in *intersections) computePoints(line dLine, used int) {
	in.pts[0] = line.ptAtT(in.ts[0][0])
	in.used = used
	if used == 2 {
		in.pts[1] = line.ptAtT(in.ts[0][1])
	}
}

// intersectRayLine computes the unbounded line/line crossing, returning raw (possibly out-of-[0,1]) T values on each
// line. For coincident rays it returns the two endpoints; for parallel, non-coincident rays it returns 0.
func (in *intersections) intersectRayLine(a, b dLine) int {
	in.max = 2
	aLen := a.pts[1].sub(a.pts[0])
	bLen := b.pts[1].sub(b.pts[0])
	// Slopes match when denom goes to zero.
	denom := bLen.y*aLen.x - aLen.y*bLen.x
	var used int
	if !approximatelyZero(denom) {
		ab0 := a.pts[0].sub(b.pts[0])
		numerA := ab0.y*bLen.x - bLen.y*ab0.x
		numerB := ab0.y*aLen.x - aLen.y*ab0.x
		numerA /= denom
		numerB /= denom
		in.ts[0][0] = numerA
		in.ts[1][0] = numerB
		used = 1
	} else {
		// See if the axis intercepts match; if not, the rays are parallel and distinct.
		if !almostEqualUlps(aLen.x*a.pts[0].y-aLen.y*a.pts[0].x,
			aLen.x*b.pts[0].y-aLen.y*b.pts[0].x) {
			in.used = 0
			return 0
		}
		// There's no single correct intersection point for two coincident rays, so return a fixed choice: T on a is 0,
		// T on b is 1 (note ts[1][0] is set to 0 below and then immediately overwritten to 1 — that odd assignment
		// order is intentional, not a bug).
		in.ts[0][0] = 0
		in.ts[1][0] = 0
		in.ts[1][0] = 1
		in.ts[1][1] = 1
		used = 2
	}
	in.computePoints(a, used)
	return in.used
}

// intersectLineLine computes the bounded intersection of two line segments: a proper crossing point, plus (for
// coincident or near-coincident lines) up to two shared endpoints. It assumes the two lines are not both exactly
// horizontal or both exactly vertical in the degenerate sense the coincident cleanup below handles.
func (in *intersections) intersectLineLine(a, b dLine) int {
	in.max = 3 // note that we clean up so that there is no more than two in the end
	// see if end points intersect the opposite line
	var t float64
	for iA := 0; iA < 2; iA++ {
		if t = b.exactPoint(a.pts[iA]); t >= 0 {
			in.insert(float64(iA), t, a.pts[iA])
		}
	}
	for iB := 0; iB < 2; iB++ {
		if t = a.exactPoint(b.pts[iB]); t >= 0 {
			in.insert(t, float64(iB), b.pts[iB])
		}
	}
	axLen := a.pts[1].x - a.pts[0].x
	ayLen := a.pts[1].y - a.pts[0].y
	bxLen := b.pts[1].x - b.pts[0].x
	byLen := b.pts[1].y - b.pts[0].y
	axByLen := axLen * byLen
	ayBxLen := ayLen * bxLen
	// Detect parallel lines the same way opAngle.orderable (opangle.go) does, so that "not parallel" here implies the
	// two segments are also sortable there.
	var unparallel bool
	if in.allowNear {
		unparallel = notAlmostEqualUlpsPin(axByLen, ayBxLen)
	} else {
		unparallel = notAlmostDequalUlps(axByLen, ayBxLen)
	}
	if unparallel && in.used == 0 {
		ab0y := a.pts[0].y - b.pts[0].y
		ab0x := a.pts[0].x - b.pts[0].x
		numerA := ab0y*bxLen - byLen*ab0x
		numerB := ab0y*axLen - ayLen*ab0x
		denom := axByLen - ayBxLen
		if between(0, numerA, denom) && between(0, numerB, denom) {
			in.ts[0][0] = numerA / denom
			in.ts[1][0] = numerB / denom
			in.computePoints(a, 1)
		}
	}
	// Track that both sets of end points are near each other (the lines are entirely coincident) even when the end
	// points are not exactly the same, so either end can mate to the next set of lines.
	if in.allowNear || !unparallel {
		var aNearB [2]float64
		var bNearA [2]float64
		var aNotB [2]bool
		var bNotA [2]bool
		nearCount := 0
		for index := 0; index < 2; index++ {
			t = b.nearPoint(a.pts[index], &aNotB[index])
			aNearB[index] = t
			if t >= 0 {
				nearCount++
			}
			t = a.nearPoint(b.pts[index], &bNotA[index])
			bNearA[index] = t
			if t >= 0 {
				nearCount++
			}
		}
		if nearCount > 0 {
			// Skip if each segment contributes to one end point.
			if nearCount != 2 || aNotB[0] == aNotB[1] {
				for iA := 0; iA < 2; iA++ {
					if !aNotB[iA] {
						continue
					}
					nearer := 0
					if aNearB[iA] > 0.5 {
						nearer = 1
					}
					if !bNotA[nearer] {
						continue
					}
					in.insertNear(float64(iA), float64(nearer), a.pts[iA])
					aNearB[iA] = -1
					bNearA[nearer] = -1
					nearCount -= 2
				}
			}
			if nearCount > 0 {
				for iA := 0; iA < 2; iA++ {
					if aNearB[iA] >= 0 {
						in.insert(float64(iA), aNearB[iA], a.pts[iA])
					}
				}
				for iB := 0; iB < 2; iB++ {
					if bNearA[iB] >= 0 {
						in.insert(bNearA[iB], float64(iB), b.pts[iB])
					}
				}
			}
		}
	}
	in.cleanUpParallelLines(!unparallel)
	return in.used
}

// horizontalCoincident reports how line relates to the horizontal line y=y: 0 if line's Y range doesn't reach y, 2 if
// line is itself (nearly) horizontal at y, else 1 (a single transversal crossing).
func horizontalCoincident(line dLine, y float64) int {
	minV := line.pts[0].y
	maxV := line.pts[1].y
	if minV > maxV {
		minV, maxV = maxV, minV
	}
	if minV > y || maxV < y {
		return 0
	}
	if almostEqualUlps(minV, maxV) && maxV-minV < math.Abs(line.pts[0].x-line.pts[1].x) {
		return 2
	}
	return 1
}

// horizontalIntercept returns the (pinned) T at which line crosses the horizontal line y=y.
func horizontalIntercept(line dLine, y float64) float64 {
	return pinT((y - line.pts[0].y) / (line.pts[1].y - line.pts[0].y))
}

// horizontalLine intersects line with the horizontal segment y=y spanning x=left to x=right, with T on that segment
// flipped when requested.
func (in *intersections) horizontalLine(line dLine, left, right, y float64, flipped bool) int {
	in.max = 3 // clean up parallel at the end will limit the result to 2 at the most
	var t float64
	leftPt := dPoint{x: left, y: y}
	if t = line.exactPoint(leftPt); t >= 0 {
		in.insert(t, b2f(flipped), leftPt)
	}
	if left != right {
		rightPt := dPoint{x: right, y: y}
		if t = line.exactPoint(rightPt); t >= 0 {
			in.insert(t, b2f(!flipped), rightPt)
		}
		for index := 0; index < 2; index++ {
			if t = exactPointH(line.pts[index], left, right, y); t >= 0 {
				in.insert(float64(index), flipT(flipped, t), line.pts[index])
			}
		}
	}
	result := horizontalCoincident(line, y)
	if result == 1 && in.used == 0 {
		in.ts[0][0] = horizontalIntercept(line, y)
		xIntercept := line.pts[0].x + in.ts[0][0]*(line.pts[1].x-line.pts[0].x)
		if between(left, xIntercept, right) {
			in.ts[1][0] = (xIntercept - left) / (right - left)
			if flipped {
				for index := 0; index < result; index++ {
					in.ts[1][index] = 1 - in.ts[1][index]
				}
			}
			in.pts[0].x = xIntercept
			in.pts[0].y = y
			in.used = 1
		}
	}
	if in.allowNear || result == 2 {
		if t = line.nearPoint(leftPt, nil); t >= 0 {
			in.insert(t, b2f(flipped), leftPt)
		}
		if left != right {
			rightPt := dPoint{x: right, y: y}
			if t = line.nearPoint(rightPt, nil); t >= 0 {
				in.insert(t, b2f(!flipped), rightPt)
			}
			for index := 0; index < 2; index++ {
				if t = nearPointH(line.pts[index], left, right, y); t >= 0 {
					in.insert(float64(index), flipT(flipped, t), line.pts[index])
				}
			}
		}
	}
	in.cleanUpParallelLines(result == 2)
	return in.used
}

// verticalCoincident reports how line relates to the vertical line x=x: 0 if line's X range doesn't reach x, 2 if line
// is itself (nearly) vertical at x, else 1 (a single transversal crossing).
func verticalCoincident(line dLine, x float64) int {
	minV := line.pts[0].x
	maxV := line.pts[1].x
	if minV > maxV {
		minV, maxV = maxV, minV
	}
	if !preciselyBetween(minV, x, maxV) {
		return 0
	}
	if almostEqualUlps(minV, maxV) {
		return 2
	}
	return 1
}

// verticalIntercept returns the (pinned) T at which line crosses the vertical line x=x.
func verticalIntercept(line dLine, x float64) float64 {
	return pinT((x - line.pts[0].x) / (line.pts[1].x - line.pts[0].x))
}

// verticalLine intersects line with the vertical segment x=x spanning y=top to y=bottom, with T on that segment flipped
// when requested.
func (in *intersections) verticalLine(line dLine, top, bottom, x float64, flipped bool) int {
	in.max = 3 // cleanup parallel lines will bring this back in line
	var t float64
	topPt := dPoint{x: x, y: top}
	if t = line.exactPoint(topPt); t >= 0 {
		in.insert(t, b2f(flipped), topPt)
	}
	if top != bottom {
		bottomPt := dPoint{x: x, y: bottom}
		if t = line.exactPoint(bottomPt); t >= 0 {
			in.insert(t, b2f(!flipped), bottomPt)
		}
		for index := 0; index < 2; index++ {
			if t = exactPointV(line.pts[index], top, bottom, x); t >= 0 {
				in.insert(float64(index), flipT(flipped, t), line.pts[index])
			}
		}
	}
	result := verticalCoincident(line, x)
	if result == 1 && in.used == 0 {
		in.ts[0][0] = verticalIntercept(line, x)
		yIntercept := line.pts[0].y + in.ts[0][0]*(line.pts[1].y-line.pts[0].y)
		if between(top, yIntercept, bottom) {
			in.ts[1][0] = (yIntercept - top) / (bottom - top)
			if flipped {
				for index := 0; index < result; index++ {
					in.ts[1][index] = 1 - in.ts[1][index]
				}
			}
			in.pts[0].x = x
			in.pts[0].y = yIntercept
			in.used = 1
		}
	}
	if in.allowNear || result == 2 {
		if t = line.nearPoint(topPt, nil); t >= 0 {
			in.insert(t, b2f(flipped), topPt)
		}
		if top != bottom {
			bottomPt := dPoint{x: x, y: bottom}
			if t = line.nearPoint(bottomPt, nil); t >= 0 {
				in.insert(t, b2f(!flipped), bottomPt)
			}
			for index := 0; index < 2; index++ {
				if t = nearPointV(line.pts[index], top, bottom, x); t >= 0 {
					in.insert(float64(index), flipT(flipped, t), line.pts[index])
				}
			}
		}
	}
	in.cleanUpParallelLines(result == 2)
	return in.used
}

// flipT returns 1-t when flipped is set, else t unchanged.
func flipT(flipped bool, t float64) float64 {
	if flipped {
		return 1 - t
	}
	return t
}
