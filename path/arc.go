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
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// arcIsLonePoint reports whether the arc described by oval/startAngle/sweepAngle degenerates to a single point,
// returning that point when it does.
func arcIsLonePoint(oval geom.Rect, startAngle, sweepAngle float32) (geom.Point, bool) {
	if sweepAngle == 0 && (startAngle == 0 || startAngle == 360) {
		// Chrome uses this path to move into and out of ovals. If not treated as a special case the moves can distort
		// the oval's bounding box (and break the circle special case).
		return geom.Pt(oval.Right, oval.CenterY()), true
	} else if oval.Width() == 0 && oval.Height() == 0 {
		// Chrome will sometimes create 0 radius round rects. Having degenerate quad segments in the path prevents the
		// path from being recognized as a rect.
		return geom.Pt(oval.Right, oval.Top), true
	}
	return geom.Point{}, false
}

// anglesToUnitVectors returns the unit vectors pointing at the start/stop points for the given start/sweep angles, plus
// the rotation direction.
func anglesToUnitVectors(startAngle, sweepAngle float32) (startV, stopV geom.Point, dir geom.RotationDirection) {
	startRad := geom.DegreesToRadians(startAngle)
	stopRad := geom.DegreesToRadians(startAngle + sweepAngle)

	startV.Y = geom.ScalarSinSnapToZero(startRad)
	startV.X = geom.ScalarCosSnapToZero(startRad)
	stopV.Y = geom.ScalarSinSnapToZero(stopRad)
	stopV.X = geom.ScalarCosSnapToZero(stopRad)

	// If the sweep angle is nearly (but less than) 360, then due to precision loss in radians-conversion and/or
	// sin/cos, we may end up with coincident vectors, which will fool BuildUnitArc into doing nothing (bad) instead of
	// drawing a nearly complete circle (good). e.g. canvas.drawArc(0, 359.99, ...) -vs- canvas.drawArc(0, 359.9, ...).
	// We try to detect this edge case, and tweak the stop vector.
	if startV == stopV {
		sw := geom.ScalarAbs(sweepAngle)
		if sw < 360 && sw > 359 {
			// make a guess at a tiny angle (in radians) to tweak by
			deltaRad := float32(math.Copysign(1.0/512, float64(sweepAngle)))
			// not sure how much will be enough, so we use a loop
			for {
				stopRad -= deltaRad
				stopV.Y = geom.ScalarSinSnapToZero(stopRad)
				stopV.X = geom.ScalarCosSnapToZero(stopRad)
				if startV != stopV {
					break
				}
			}
		}
	}
	if sweepAngle > 0 {
		dir = geom.RotationCW
	} else {
		dir = geom.RotationCCW
	}
	return startV, stopV, dir
}

// buildArcConics builds the conic sections approximating the arc from start to stop. If it returns 0, the caller should
// just line-to singlePt; otherwise it should ignore singlePt and append the returned number of conics.
func buildArcConics(oval geom.Rect, start, stop geom.Point, dir geom.RotationDirection, conics *[geom.MaxConicsForArc]geom.Conic) (singlePt geom.Point, count int) {
	var matrix geom.Matrix
	matrix.SetScale(oval.Width()*0.5, oval.Height()*0.5)
	matrix.PostTranslate(oval.CenterX(), oval.CenterY())

	count = geom.BuildUnitArc(start, stop, dir, &matrix, conics)
	if count == 0 {
		singlePt = matrix.MapPoint(stop)
	}
	return singlePt, count
}

// ArcToOval appends an arc that is part of the ellipse bounded by oval, from startAngle through sweepAngle (degrees,
// zero along the positive x-axis, positive sweeps clockwise). When forceMoveTo is false and the path is not empty, a
// line connects the last point to the arc's first point.
func (p *Path) ArcToOval(oval geom.Rect, startAngle, sweepAngle float32, forceMoveTo bool) *Path {
	if oval.Width() < 0 || oval.Height() < 0 {
		return p
	}

	startAngle = float32(math.Mod(float64(startAngle), 360))

	if len(p.verbs) == 0 {
		forceMoveTo = true
	}

	if lonePt, isLone := arcIsLonePoint(oval, startAngle, sweepAngle); isLone {
		if forceMoveTo {
			return p.MoveToPt(lonePt)
		}
		return p.LineToPt(lonePt)
	}

	startV, stopV, dir := anglesToUnitVectors(startAngle, sweepAngle)

	// Adds a move-to to 'pt' if forceMoveTo is true. Otherwise a lineTo unless we're sufficiently close to 'pt'
	// currently. This prevents spurious lineTos when adding a series of contiguous arcs from the same oval.
	addPt := func(pt geom.Point) {
		if forceMoveTo {
			p.MoveToPt(pt)
		} else if lastPt, has := p.LastPt(); !has ||
			!geom.ScalarNearlyEqual(lastPt.X, pt.X) ||
			!geom.ScalarNearlyEqual(lastPt.Y, pt.Y) {
			p.LineToPt(pt)
		}
	}

	// At this point, we know that the arc is not a lone point, but startV == stopV indicates that the sweepAngle is too
	// small such that anglesToUnitVectors cannot handle it.
	if startV == stopV {
		endAngle := geom.DegreesToRadians(startAngle + sweepAngle)
		radiusX := oval.Width() / 2
		radiusY := oval.Height() / 2
		// We do not use the snap-to-zero sin/cos here. When sin(startAngle) is 0 and sweepAngle is very small and
		// radius is huge, the expected behavior here is to draw a line. But snapping sin(endAngle) to 0 would draw a
		// dot instead.
		addPt(geom.Pt(oval.CenterX()+radiusX*geom.ScalarCos(endAngle),
			oval.CenterY()+radiusY*geom.ScalarSin(endAngle)))
		return p
	}

	var conics [geom.MaxConicsForArc]geom.Conic
	singlePt, count := buildArcConics(oval, startV, stopV, dir, &conics)
	if count > 0 {
		addPt(conics[0].Pts[0])
		for i := 0; i < count; i++ {
			p.ConicToPt(conics[i].Pts[1], conics[i].Pts[2], conics[i].W)
		}
	} else {
		addPt(singlePt)
	}
	return p
}

// AddArc appends an arc as a new contour; sweeps of 360 degrees or more starting on a 90-degree boundary append an oval
// instead.
func (p *Path) AddArc(oval geom.Rect, startAngle, sweepAngle float32) *Path {
	if oval.IsEmpty() || sweepAngle == 0 {
		return p
	}

	if sweepAngle >= 360 || sweepAngle <= -360 {
		// We can treat the arc as an oval if it begins at one of our legal starting positions.
		startOver90 := startAngle / 90
		startOver90I := geom.RoundToScalar(startOver90)
		if geom.ScalarNearlyEqual(startOver90-startOver90I, 0) {
			// Index 1 is at startAngle == 0.
			startIndex := float32(math.Mod(float64(startOver90I+1), 4))
			if startIndex < 0 {
				startIndex += 4
			}
			dir := geom.DirectionCW
			if sweepAngle <= 0 {
				dir = geom.DirectionCCW
			}
			return p.AddOvalWithStart(oval, dir, uint(startIndex))
		}
	}
	return p.ArcToOval(oval, startAngle, sweepAngle, true)
}

// ArcToTangent appends an arc contained by the tangent from the last point to (x1, y1) and the tangent from (x1, y1) to
// (x2, y2), on a circle of the given radius (PostScript arct / HTML Canvas arcTo). A connecting line is appended first
// when needed; degenerate tangents reduce to a line to (x1, y1).
func (p *Path) ArcToTangent(x1, y1, x2, y2, radius float32) *Path {
	p.injectMoveToIfNeeded()

	if radius == 0 {
		return p.LineTo(x1, y1)
	}

	// need to know our prev pt so we can construct tangent vectors
	start, _ := p.LastPt()

	// need double precision for these calcs.
	beforeX, beforeY, beforeOK := normalizeDouble(float64(x1)-float64(start.X), float64(y1)-float64(start.Y))
	afterX, afterY, afterOK := normalizeDouble(float64(x2)-float64(x1), float64(y2)-float64(y1))
	cosh := beforeX*afterX + beforeY*afterY
	sinh := beforeX*afterY - beforeY*afterX

	// If the previous point equals the first point, before will be denormalized. If the two points equal, after will be
	// denormalized. If the second point equals the first point, sinh will be zero. In all these cases, we cannot
	// construct an arc, so we construct a line to the first point.
	if !beforeOK || !afterOK || geom.ScalarNearlyZero(geom.DoubleToScalar(sinh)) {
		return p.LineTo(x1, y1)
	}

	// safe to convert back to floats now
	dist := geom.ScalarAbs(geom.DoubleToScalar(float64(radius) * (1 - cosh) / sinh))
	xx := x1 - dist*float32(beforeX)
	yy := y1 - dist*float32(beforeY)

	after := geom.Pt(float32(afterX), float32(afterY))
	after.SetLength(dist)
	p.LineTo(xx, yy)
	weight := geom.ScalarSqrt(geom.DoubleToScalar(0.5 + cosh*0.5))
	return p.ConicTo(x1, y1, x1+after.X, y1+after.Y, weight)
}

// normalizeDouble normalizes (x, y) in double precision, reporting false when the result is not finite (zero-length or
// non-finite input).
func normalizeDouble(x, y float64) (nx, ny float64, ok bool) {
	mag := math.Sqrt(x*x + y*y)
	nx = x / mag
	ny = y / mag
	return nx, ny, !math.IsNaN(nx) && !math.IsInf(nx, 0) && !math.IsNaN(ny) && !math.IsInf(ny, 0)
}

// ArcToRotated appends an SVG-style elliptical arc from the last point to (x, y), on an oval with radii (rx, ry)
// rotated by xAxisRotate degrees, choosing one of four possible routes via largeArc and sweep. Note that SVG's
// sweep-flag is opposite the integer value of sweep (SVG uses 1 for clockwise; geom.DirectionCW is zero).
func (p *Path) ArcToRotated(rx, ry, angle float32, arcLarge ArcSize, arcSweep geom.PathDirection, x, y float32) *Path {
	p.injectMoveToIfNeeded()
	var srcPts [2]geom.Point
	srcPts[0], _ = p.LastPt()
	// If rx = 0 or ry = 0 then this arc is treated as a straight line segment (a "lineto") joining the endpoints.
	// http://www.w3.org/TR/SVG/implnote.html#ArcOutOfRangeParameters
	if rx == 0 || ry == 0 {
		return p.LineTo(x, y)
	}
	// If the current point and target point for the arc are identical, it should be treated as a zero length path. This
	// ensures continuity in animations.
	srcPts[1] = geom.Pt(x, y)
	if srcPts[0] == srcPts[1] {
		return p.LineTo(x, y)
	}
	rx = geom.ScalarAbs(rx)
	ry = geom.ScalarAbs(ry)
	midPointDistance := srcPts[0].Sub(srcPts[1]).Scaled(0.5)

	var pointTransform geom.Matrix
	pointTransform.SetRotate(-angle)

	transformedMidPoint := pointTransform.MapPoint(midPointDistance)
	squareRx := rx * rx
	squareRy := ry * ry
	squareX := transformedMidPoint.X * transformedMidPoint.X
	squareY := transformedMidPoint.Y * transformedMidPoint.Y

	// Check if the radii are big enough to draw the arc, scale radii if not.
	// http://www.w3.org/TR/SVG/implnote.html#ArcCorrectionOutOfRangeRadii
	radiiScale := squareX/squareRx + squareY/squareRy
	if radiiScale > 1 {
		radiiScale = geom.ScalarSqrt(radiiScale)
		rx *= radiiScale
		ry *= radiiScale
	}

	pointTransform.SetScale(1/rx, 1/ry)
	pointTransform.PreRotate(-angle)

	var unitPts [2]geom.Point
	pointTransform.MapPoints(unitPts[:], srcPts[:])
	delta := unitPts[1].Sub(unitPts[0])

	d := delta.X*delta.X + delta.Y*delta.Y
	scaleFactorSquared := max32(1/d-0.25, 0)

	scaleFactor := geom.ScalarSqrt(scaleFactorSquared)
	if (arcSweep == geom.DirectionCCW) != (arcLarge == ArcSizeLarge) { // flipped from the original implementation
		scaleFactor = -scaleFactor
	}
	delta = delta.Scaled(scaleFactor)
	centerPoint := unitPts[0].Add(unitPts[1]).Scaled(0.5)
	centerPoint.X -= delta.Y
	centerPoint.Y += delta.X
	unitPts[0] = unitPts[0].Sub(centerPoint)
	unitPts[1] = unitPts[1].Sub(centerPoint)
	theta1 := float32(math.Atan2(float64(unitPts[0].Y), float64(unitPts[0].X)))
	theta2 := float32(math.Atan2(float64(unitPts[1].Y), float64(unitPts[1].X)))
	thetaArc := theta2 - theta1
	if thetaArc < 0 && arcSweep == geom.DirectionCW { // arcSweep flipped from the original implementation
		thetaArc += float32(math.Pi) * 2
	} else if thetaArc > 0 && arcSweep != geom.DirectionCW { // arcSweep flipped from the original implementation
		thetaArc -= float32(math.Pi) * 2
	}

	// Very tiny angles cause our subsequent math to go wonky (skbug.com/40040578) so we do a quick check here. The
	// precise tolerance amount is just made up. PI/million happens to fix the bug in 9272, but a larger value is
	// probably ok too.
	if geom.ScalarAbs(thetaArc) < float32(math.Pi)/(1000*1000) {
		return p.LineTo(x, y)
	}

	pointTransform.SetRotate(angle)
	pointTransform.PreScale(rx, ry)

	// the arc may be slightly bigger than 1/4 circle, so allow up to 1/3rd
	segments := geom.CeilToInt(geom.ScalarAbs(thetaArc / (2 * float32(math.Pi) / 3)))
	thetaWidth := thetaArc / float32(segments)
	t := float32(math.Tan(float64(0.5 * thetaWidth)))
	if !geom.IsFinite(t) {
		return p
	}
	startTheta := theta1
	w := geom.ScalarSqrt(0.5 + geom.ScalarCos(thetaWidth)*0.5)
	scalarIsInteger := func(scalar float32) bool {
		return scalar == geom.FloorToScalar(scalar)
	}
	expectIntegers := geom.ScalarNearlyZero(float32(math.Pi)/2-geom.ScalarAbs(thetaWidth)) &&
		scalarIsInteger(rx) && scalarIsInteger(ry) &&
		scalarIsInteger(x) && scalarIsInteger(y)

	for i := int32(0); i < segments; i++ {
		endTheta := startTheta + thetaWidth
		sinEndTheta := geom.ScalarSinSnapToZero(endTheta)
		cosEndTheta := geom.ScalarCosSnapToZero(endTheta)

		unitPts[1] = geom.Pt(cosEndTheta, sinEndTheta).Add(centerPoint)
		unitPts[0] = unitPts[1]
		unitPts[0].X += t * sinEndTheta
		unitPts[0].Y += -t * cosEndTheta
		var mapped [2]geom.Point
		pointTransform.MapPoints(mapped[:], unitPts[:])
		// Computing the arc width introduces rounding errors that cause arcs to start outside their marks. A round rect
		// may lose convexity as a result. If the input values are on integers, place the conic on integers as well.
		if expectIntegers {
			for j := range mapped {
				mapped[j].X = geom.RoundToScalar(mapped[j].X)
				mapped[j].Y = geom.RoundToScalar(mapped[j].Y)
			}
		}
		p.ConicToPt(mapped[0], mapped[1], w)
		startTheta = endTheta
	}

	// The final point should match the input point (by definition); replace it to ensure that rounding errors in the
	// above math don't cause any problems.
	p.SetLastPt(x, y)
	return p
}

// RArcToRotated is ArcToRotated with the end point relative to the last point.
func (p *Path) RArcToRotated(rx, ry, xAxisRotate float32, largeArc ArcSize, sweep geom.PathDirection, dx, dy float32) *Path {
	currentPoint, _ := p.LastPt()
	return p.ArcToRotated(rx, ry, xAxisRotate, largeArc, sweep, currentPoint.X+dx, currentPoint.Y+dy)
}

func max32(a, b float32) float32 {
	if a < b {
		return b
	}
	return a
}
