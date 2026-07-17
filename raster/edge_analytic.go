// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// AnalyticEdge: the fixed-point edges the analytic-AA scan converter walks. Unlike the non-AA Edge, positions are
// tracked as continuous (snapped) Fixed y values rather than discrete pixel rows.

package raster

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// analyticAccuracy is the default accuracy: y is snapped to multiples of 1/(1<<accuracy) of a pixel.
const analyticAccuracy = 2

// kInverseTableSize is FDot6One * 16.
const kInverseTableSize = 1024

// quickInverse returns -4194304 / x with C-style truncating division (equivalent to a 1025-entry lookup table whose
// entries are exactly -4194304 / (1024 - i), verified against the table's endpoints and spot values, so the direct
// division reproduces it bit-for-bit). x == 0 maps to 0 (updateLine reaches it when a non-zero Fixed slope truncates to
// zero FDot6).
func quickInverse(x FDot6) Fixed {
	if x == 0 {
		return 0
	}
	return Fixed((FDot6One << 16) / x)
}

// quickDiv computes a/b as Fixed via the inverse table when |b| is in [8, 1024) and |a| is small enough that the 32-bit
// product cannot overflow.
func quickDiv(a, b FDot6) Fixed {
	const kMinBits = 3
	const kMaxBits = 31
	const kMaxAbsA = 1 << (kMaxBits - (22 - kMinBits))
	absA := a
	if absA < 0 {
		absA = -absA
	}
	absB := b
	if absB < 0 {
		absB = -absB
	}
	if absB >= 1<<kMinBits && absB < kInverseTableSize && absA < kMaxAbsA {
		return Fixed(int32(a)*int32(quickInverse(b))) >> 6
	}
	return FDot6Div(a, b)
}

// snapY rounds y to the nearest analyticAccuracy grid line. It computes in unsigned so negative y wraps rather than
// shifting in sign bits.
func snapY(y Fixed) Fixed {
	const accuracy = analyticAccuracy
	return Fixed(int32((uint32(y) + uint32(FixedOne>>(accuracy+1))) >> (16 - accuracy) << (16 - accuracy)))
}

// analyticCurve is implemented by the quad/cubic analytic edges; it lets the walker advance an edge's current segment
// without knowing the concrete type.
type analyticCurve interface {
	updateCurve() bool
	keepContinuous()
}

// AnalyticEdge is one edge walked by the analytic-AA scan converter. The edge runs from y = UpperY to y = LowerY (both
// Fixed), with X the current x at Y and DX the slope.
type AnalyticEdge struct {
	Next, Prev *AnalyticEdge

	curve analyticCurve // non-nil for quad/cubic edges

	X      Fixed
	DX     Fixed
	UpperX Fixed // the x value when Y = UpperY
	Y      Fixed // the current y
	UpperY Fixed
	LowerY Fixed
	// DY is abs(1/DX); may be MaxInt32 when DX is close to 0. Only used for blitting trapezoids.
	DY Fixed

	EdgeType   EdgeType // remembers the *initial* edge type
	CurveCount int8     // only used by quads (+) and cubics (-)
	CurveShift uint8    // applied to all Dx/DDx/DDDx except for the cubic's DShift exception
	Winding    int8
}

// goY updates X, Y so that Y == y.
func (e *AnalyticEdge) goY(y Fixed) {
	if y == e.Y+FixedOne {
		e.X += e.DX
		e.Y = y
	} else if y != e.Y {
		// Drop lower digits as our alpha only has 8 bits (DX and y - UpperY may be greater than FixedOne).
		e.X = e.UpperX + FixedMul(e.DX, y-e.UpperY)
		e.Y = y
	}
}

// goYShift updates X, Y so that Y == y, given y - Y is exactly FixedOne >> yShift.
func (e *AnalyticEdge) goYShift(y Fixed, yShift int) {
	e.Y = y
	e.X += e.DX >> yShift
}

// update returns true if we're NOT done with this edge.
func (e *AnalyticEdge) update() bool {
	if e.CurveCount != 0 {
		return e.curve.updateCurve()
	}
	return false
}

// SetLine represents a straight line as a single analytic edge. X/Y are converted the same way (times 4, to FDot6, then
// to Fixed) as quads/cubics; otherwise the order of the edges might be wrong due to the precision limit.
func (e *AnalyticEdge) SetLine(p0, p1 geom.Point) bool {
	const accuracy = analyticAccuracy
	const multiplier = 1 << analyticAccuracy
	x0 := FDot6ToFixed(FloatToFDot6(p0.X*multiplier)) >> accuracy
	y0 := snapY(FDot6ToFixed(FloatToFDot6(p0.Y*multiplier)) >> accuracy)
	x1 := FDot6ToFixed(FloatToFDot6(p1.X*multiplier)) >> accuracy
	y1 := snapY(FDot6ToFixed(FloatToFDot6(p1.Y*multiplier)) >> accuracy)

	winding := WindingCW
	if y0 > y1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
		winding = WindingCCW
	}

	// are we a zero-height line?
	dy := FixedToFDot6(y1 - y0)
	if dy == 0 {
		return false
	}
	dx := FixedToFDot6(x1 - x0)
	slope := quickDiv(dx, dy)
	absSlope := slope
	if absSlope < 0 {
		absSlope = -absSlope
	}

	e.X = x0
	e.DX = slope
	e.UpperX = x0
	e.Y = y0
	e.UpperY = y0
	e.LowerY = y1
	switch {
	case dx == 0 || slope == 0:
		e.DY = math.MaxInt32
	case absSlope < kInverseTableSize:
		e.DY = quickInverse(FDot6(absSlope))
	default:
		e.DY = FixedAbs(quickDiv(dy, dx))
	}
	e.EdgeType = EdgeTypeLine
	e.CurveCount = 0
	e.Winding = winding
	e.CurveShift = 0

	return true
}

// updateLine sets up a line segment between the given fixed-point endpoints; the caller computes the slope once and
// sends it in (to avoid a second division); y snapping already happened.
func (e *AnalyticEdge) updateLine(x0, y0, x1, y1, slope Fixed) bool {
	// We don't chop at y extrema for cubics, so y is not guaranteed to be increasing for them. In that case, we have to
	// swap x/y and negate the winding.
	if y0 > y1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
		e.Winding = -e.Winding
	}

	dx := FixedToFDot6(x1 - x0)
	dy := FixedToFDot6(y1 - y0)

	// are we a zero-height line?
	if dy == 0 {
		return false
	}

	// abs of the FDot6 conversion, in that order: the arithmetic shift floors negative slopes away from zero, so e.g.
	// slope -3734 gives absSlope 4, not 3.
	absSlope := FixedToFDot6(slope)
	if absSlope < 0 {
		absSlope = -absSlope
	}
	e.X = x0
	e.DX = slope
	e.UpperX = x0
	e.Y = y0
	e.UpperY = y0
	e.LowerY = y1
	switch {
	case dx == 0 || slope == 0:
		e.DY = math.MaxInt32
	case absSlope < kInverseTableSize:
		e.DY = quickInverse(absSlope)
	default:
		e.DY = FixedAbs(quickDiv(dy, dx))
	}

	return true
}

///////////////////////////////////////////////////////////////////////////////

// AnalyticQuadraticEdge is a Y-monotonic quadratic curve edge for the analytic-AA scan converter, walked as a sequence
// of forward-differenced line segments.
type AnalyticQuadraticEdge struct {
	AnalyticEdge
	qx, qy         Fixed
	qDx, qDy       Fixed
	qDDx, qDDy     Fixed
	qLastX, qLastY Fixed

	// snap y to integer points in the middle of the curve to accelerate AAA path filling
	snappedX, snappedY Fixed
}

// keepContinuous uses X as the starting x to ensure continuity; without it, we may break the sorted edge list.
func (q *AnalyticQuadraticEdge) keepContinuous() {
	q.snappedX = q.X
	q.snappedY = q.Y
}

// setQuadraticWithoutUpdate sets up the polynomial coefficients for a Y-monotonic quad, without computing the first
// line segment.
func (q *AnalyticQuadraticEdge) setQuadraticWithoutUpdate(pts []geom.Point, shift int) bool {
	scale := float32(int32(1) << (shift + 6))
	x0 := FDot6(pts[0].X * scale)
	y0 := FDot6(pts[0].Y * scale)
	x1 := FDot6(pts[1].X * scale)
	y1 := FDot6(pts[1].Y * scale)
	x2 := FDot6(pts[2].X * scale)
	y2 := FDot6(pts[2].Y * scale)

	winding := WindingCW
	if y0 > y2 {
		x0, x2 = x2, x0
		y0, y2 = y2, y0
		winding = WindingCCW
	}

	top := FDot6Round(y0)
	bot := FDot6Round(y2)

	// are we a zero-height quad (line)?
	if top == bot {
		return false
	}

	// compute the number of steps needed (1 << shift). Before this, shift is the AA scale-up factor; after, it is the
	// curve shift.
	dx := (x1<<1 - x0 - x2) >> 2
	dy := (y1<<1 - y0 - y2) >> 2
	shift = diffToShift(dx, dy, shift)
	// need at least 1 subdivision for our bias trick
	if shift == 0 {
		shift = 1
	} else if shift > maxCoeffShift {
		shift = maxCoeffShift
	}

	q.Winding = winding
	q.EdgeType = EdgeTypeQuad
	q.CurveCount = int8(1 << shift)

	// In polynomial form At^2 + Bt + C: A = p0 - 2p1 + p2, B = 2(p1 - p0), C = p0. A and B are stored at 1/2 their
	// actual value to guard against overflow, applying a 2x scale during updateQuadratic; hence CurveShift = shift - 1.
	q.CurveShift = uint8(shift - 1)

	a := fdot6ToFixedDiv2(x0 - x1 - x1 + x2) // 1/2 the real value
	b := FDot6ToFixed(x1 - x0)               // 1/2 the real value

	q.qx = FDot6ToFixed(x0)
	q.qDx = b + (a >> shift)  // biased by shift
	q.qDDx = a >> (shift - 1) // biased by shift

	a = fdot6ToFixedDiv2(y0 - y1 - y1 + y2)
	b = FDot6ToFixed(y1 - y0)

	q.qy = FDot6ToFixed(y0)
	q.qDy = b + (a >> shift)
	q.qDDy = a >> (shift - 1)

	q.qLastX = FDot6ToFixed(x2)
	q.qLastY = FDot6ToFixed(y2)

	return true
}

// SetQuadratic sets up the line segments for a Y-monotonic quad, computing the first segment.
func (q *AnalyticQuadraticEdge) SetQuadratic(pts []geom.Point) bool {
	if !q.setQuadraticWithoutUpdate(pts, analyticAccuracy) {
		return false
	}
	q.qx >>= analyticAccuracy
	q.qy >>= analyticAccuracy
	q.qDx >>= analyticAccuracy
	q.qDy >>= analyticAccuracy
	q.qDDx >>= analyticAccuracy
	q.qDDy >>= analyticAccuracy
	q.qLastX >>= analyticAccuracy
	q.qLastY >>= analyticAccuracy
	q.qy = snapY(q.qy)
	q.qLastY = snapY(q.qLastY)

	q.EdgeType = EdgeTypeQuad

	q.snappedX = q.qx
	q.snappedY = q.qy

	q.curve = q
	return q.updateCurve()
}

// updateCurve advances the quad edge to its next forward-differenced line segment.
func (q *AnalyticQuadraticEdge) updateCurve() bool {
	success := false
	count := int(q.CurveCount)
	oldx := q.qx
	oldy := q.qy
	dx := q.qDx
	dy := q.qDy
	var newx, newy, newSnappedX, newSnappedY Fixed
	shift := int(q.CurveShift)

	for {
		var slope Fixed
		count--
		if count > 0 {
			newx = oldx + (dx >> shift)
			newy = oldy + (dy >> shift)
			// only snap when dy is large enough and dx/dy isn't too large
			if FixedAbs(dy>>shift) >= FixedOne*2 && int64(FixedAbs(dy))<<6 > int64(FixedAbs(dx)) {
				diffY := FixedToFDot6(newy - q.snappedY)
				if diffY != 0 {
					slope = quickDiv(FixedToFDot6(newx-q.snappedX), diffY)
				} else {
					slope = math.MaxInt32
				}
				newSnappedY = min(q.qLastY, FixedRoundToFixed(newy))
				newSnappedX = newx - FixedMul(slope, newy-newSnappedY)
			} else {
				newSnappedY = min(q.qLastY, snapY(newy))
				newSnappedX = newx
				diffY := FixedToFDot6(newSnappedY - q.snappedY)
				if diffY != 0 {
					slope = quickDiv(FixedToFDot6(newx-q.snappedX), diffY)
				} else {
					slope = math.MaxInt32
				}
			}
			dx += q.qDDx
			dy += q.qDDy
		} else { // last segment
			newx = q.qLastX
			newy = q.qLastY
			newSnappedY = newy
			newSnappedX = newx
			diffY := FixedToFDot6(newy - q.snappedY)
			if diffY != 0 {
				slope = quickDiv(FixedToFDot6(newx-q.snappedX), diffY)
			} else {
				slope = math.MaxInt32
			}
		}
		if slope < math.MaxInt32 {
			success = q.updateLine(q.snappedX, q.snappedY, newSnappedX, newSnappedY, slope)
		}
		oldx = newx
		oldy = newy
		if count <= 0 || success {
			break
		}
	}

	q.qx = newx
	q.qy = newy
	q.qDx = dx
	q.qDy = dy
	q.snappedX = newSnappedX
	q.snappedY = newSnappedY
	q.CurveCount = int8(count)
	return success
}

///////////////////////////////////////////////////////////////////////////////

// AnalyticCubicEdge is a Y-monotonic cubic curve edge for the analytic-AA scan converter, walked as a sequence of
// forward-differenced line segments.
type AnalyticCubicEdge struct {
	AnalyticEdge
	cx, cy         Fixed
	cDx, cDy       Fixed
	cDDx, cDDy     Fixed
	cDDDx, cDDDy   Fixed
	cLastX, cLastY Fixed

	snappedY Fixed // to make sure that y is increasing with smooth jump and snapping

	cubicDShift uint8 // applied to cDx and cDy
}

// keepContinuous uses X and Y as the starting point to ensure continuity across segments.
func (c *AnalyticCubicEdge) keepContinuous() {
	c.cx = c.X
	c.snappedY = c.Y
}

// setCubicWithoutUpdate sets up the polynomial coefficients for a Y-monotonic cubic, without computing the first line
// segment.
func (c *AnalyticCubicEdge) setCubicWithoutUpdate(pts []geom.Point, shift int) bool {
	scale := float32(int32(1) << (shift + 6))
	x0 := FDot6(pts[0].X * scale)
	y0 := FDot6(pts[0].Y * scale)
	x1 := FDot6(pts[1].X * scale)
	y1 := FDot6(pts[1].Y * scale)
	x2 := FDot6(pts[2].X * scale)
	y2 := FDot6(pts[2].Y * scale)
	x3 := FDot6(pts[3].X * scale)
	y3 := FDot6(pts[3].Y * scale)

	winding := WindingCW
	if y0 > y3 {
		x0, x3 = x3, x0
		x1, x2 = x2, x1
		y0, y3 = y3, y0
		y1, y2 = y2, y1
		winding = WindingCCW
	}

	top := FDot6Round(y0)
	bot := FDot6Round(y3)

	// are we a zero-height cubic (line)?
	if top == bot {
		return false
	}

	// compute the number of steps needed (1 << shift)
	dx := cubicDeltaFromLine(x0, x1, x2, x3)
	dy := cubicDeltaFromLine(y0, y1, y2, y3)
	// add 1 (by observation)
	shift = diffToShift(dx, dy, 2) + 1
	if shift > maxCoeffShift {
		shift = maxCoeffShift
	}

	// Since our incoming data is initially shifted down by 10 (or 8 in antialias), the most we can shift up is 8.
	// However, we compute coefficients with a 3*, so the safest upshift is really 6.
	upShift := 6 // largest safe value
	downShift := shift + upShift - 10
	if downShift < 0 {
		downShift = 0
		upShift = 10 - shift
	}

	c.Winding = winding
	c.EdgeType = EdgeTypeCubic
	c.CurveCount = int8(int32(-1) << shift)
	c.CurveShift = uint8(shift)
	c.cubicDShift = uint8(downShift)

	b := fdot6UpShift(3*(x1-x0), upShift)
	cc := fdot6UpShift(3*(x0-x1-x1+x2), upShift)
	d := fdot6UpShift(x3+3*(x1-x2)-x0, upShift)

	c.cx = FDot6ToFixed(x0)
	c.cDx = b + (cc >> shift) + (d >> (2 * shift)) // biased by shift
	c.cDDx = 2*cc + (3*d)>>(shift-1)               // biased by 2*shift
	c.cDDDx = (3 * d) >> (shift - 1)               // biased by 2*shift

	b = fdot6UpShift(3*(y1-y0), upShift)
	cc = fdot6UpShift(3*(y0-y1-y1+y2), upShift)
	d = fdot6UpShift(y3+3*(y1-y2)-y0, upShift)

	c.cy = FDot6ToFixed(y0)
	c.cDy = b + (cc >> shift) + (d >> (2 * shift))
	c.cDDy = 2*cc + (3*d)>>(shift-1)
	c.cDDDy = (3 * d) >> (shift - 1)

	c.cLastX = FDot6ToFixed(x3)
	c.cLastY = FDot6ToFixed(y3)

	return true
}

// SetCubic sets up the line segments for a Y-monotonic cubic, computing the first segment.
func (c *AnalyticCubicEdge) SetCubic(pts []geom.Point) bool {
	if !c.setCubicWithoutUpdate(pts, analyticAccuracy) {
		return false
	}

	c.cx >>= analyticAccuracy
	c.cy >>= analyticAccuracy
	c.cDx >>= analyticAccuracy
	c.cDy >>= analyticAccuracy
	c.cDDx >>= analyticAccuracy
	c.cDDy >>= analyticAccuracy
	c.cDDDx >>= analyticAccuracy
	c.cDDDy >>= analyticAccuracy
	c.cLastX >>= analyticAccuracy
	c.cLastY >>= analyticAccuracy
	c.cy = snapY(c.cy)
	c.snappedY = c.cy
	c.cLastY = snapY(c.cLastY)

	c.EdgeType = EdgeTypeCubic

	c.curve = c
	return c.updateCurve()
}

// updateCurve advances the cubic edge to its next forward-differenced line segment.
func (c *AnalyticCubicEdge) updateCurve() bool {
	var success bool
	count := int(c.CurveCount)
	oldx := c.cx
	oldy := c.cy
	var newx, newy Fixed
	ddshift := int(c.CurveShift)
	dshift := int(c.cubicDShift)

	for {
		count++
		if count < 0 {
			newx = oldx + (c.cDx >> dshift)
			c.cDx += c.cDDx >> ddshift
			c.cDDx += c.cDDDx

			newy = oldy + (c.cDy >> dshift)
			c.cDy += c.cDDy >> ddshift
			c.cDDy += c.cDDDy
		} else { // last segment
			newx = c.cLastX
			newy = c.cLastY
		}

		// we want to say assert(oldy <= newy), but our finite fixedpoint doesn't always achieve that, so we have to
		// explicitly pin it here.
		if newy < oldy {
			newy = oldy
		}

		newSnappedY := snapY(newy)
		// likewise pin newSnappedY to cLastY.
		if c.cLastY < newSnappedY {
			newSnappedY = c.cLastY
			count = 0
		}

		var slope Fixed
		if FixedToFDot6(newSnappedY-c.snappedY) == 0 {
			slope = math.MaxInt32
		} else {
			slope = FDot6Div(FixedToFDot6(newx-oldx), FixedToFDot6(newSnappedY-c.snappedY))
		}

		success = c.updateLine(oldx, c.snappedY, newx, newSnappedY, slope)

		oldx = newx
		oldy = newy
		c.snappedY = newSnappedY
		if count >= 0 || success {
			break
		}
	}

	c.cx = newx
	c.cy = newy
	c.CurveCount = int8(count)
	return success
}
