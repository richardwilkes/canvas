// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Edge approximates monotonic curves with one or more line segments in a form that makes walking horizontal scanlines
// efficient. Curve edges forward-difference through a power-of-2 number of segments chosen by a flatness heuristic.

package raster

import (
	"math/bits"

	"github.com/richardwilkes/canvas/geom"
)

// EdgeType remembers an edge's initial segment type.
type EdgeType int8

// EdgeType values.
const (
	EdgeTypeLine EdgeType = iota
	EdgeTypeQuad
	EdgeTypeCubic
)

// Edge windings.
const (
	WindingCW  int8 = 1
	WindingCCW int8 = -1
)

// computeDY computes the fractional-Y offset from top to y0, correctly favoring the lower pixel when y0 is on a 1/2
// pixel boundary.
func computeDY(top int32, y0 FDot6) FDot6 {
	return FDot6(top<<6) + 32 - y0
}

// segmenter is implemented by curve edges that can advance to their next line segment.
type segmenter interface {
	nextSegment() bool
}

// Edge is one line segment of a path edge, in a form suited to horizontal scanline traversal. The current line segment
// starts at (X, FirstY + 0.5), has slope DxDy (run over rise, geared toward horizontal scanlines) and stops once Y gets
// to LastY + 0.5. FirstY/LastY are discrete pixels, treated as halfway inside the pixel.
type Edge struct {
	// Next and Prev join edges in the scan converter's doubly-linked list.
	Next, Prev *Edge

	owner segmenter // non-nil for quad/cubic edges; dispatches nextSegment to the derived type

	X    Fixed
	DxDy Fixed

	FirstY int32
	LastY  int32

	EdgeType EdgeType
	Winding  int8

	segmentCount uint8 // only non-zero for quads and cubics
	// curveShift is how much to shift the derivatives to multiply by deltaT when forward-differencing: log_2(N) for
	// cubics and log_2(N) - 1 for quadratics.
	curveShift uint8
}

// hasNextSegment reports whether the edge has more curve segments to walk.
func (e *Edge) hasNextSegment() bool {
	return e.segmentCount != 0
}

// nextSegment dispatches to the owning curve edge.
func (e *Edge) nextSegment() bool {
	return e.owner.nextSegment()
}

// SetLine represents a straight line as a single segment (no clip). Returns false if the line has a height of 0 pixels.
func (e *Edge) SetLine(p0, p1 geom.Point) bool {
	return e.setLineClip(p0, p1, nil)
}

// SetLineClipped represents a straight line as a single segment, clipped to clip.
func (e *Edge) SetLineClipped(p0, p1 geom.Point, clip *geom.IRect) bool {
	return e.setLineClip(p0, p1, clip)
}

func (e *Edge) setLineClip(p0, p1 geom.Point, clip *geom.IRect) bool {
	x0 := FloatToFDot6(p0.X)
	y0 := FloatToFDot6(p0.Y)
	x1 := FloatToFDot6(p1.X)
	y1 := FloatToFDot6(p1.Y)

	winding := WindingCW
	if y0 > y1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
		winding = WindingCCW
	}

	top := FDot6Round(y0)
	bot := FDot6Round(y1)

	// are we a zero-height line?
	if top == bot {
		return false
	}
	// are we completely above or below the clip?
	if clip != nil && (top >= clip.Bottom || bot <= clip.Top) {
		return false
	}

	slope := FDot6Div(x1-x0, y1-y0)
	dy := computeDY(top, y0)

	// Note that FixedMul(Fixed, FDot6) produces results in FDot6
	e.X = FDot6ToFixed(x0 + FDot6(FixedMul(slope, Fixed(dy))))
	e.DxDy = slope
	e.FirstY = top
	e.LastY = bot - 1
	e.EdgeType = EdgeTypeLine
	e.segmentCount = 0
	e.Winding = winding
	e.curveShift = 0

	if clip != nil {
		e.chopLineWithClip(clip)
	}
	return true
}

// updateLine sets up a line between the provided fixed-point endpoints and updates the base fields to line up with the
// closest pixel center. Returns false if this segment would start and stop in the same row.
func (e *Edge) updateLine(xStart, yStart, xEnd, yEnd Fixed) bool {
	y0 := FixedToFDot6(yStart)
	y1 := FixedToFDot6(yEnd)

	top := FDot6Round(y0)
	bot := FDot6Round(y1)

	// are we a zero-height line?
	if top == bot {
		return false
	}

	x0 := FixedToFDot6(xStart)
	x1 := FixedToFDot6(xEnd)

	slope := FDot6Div(x1-x0, y1-y0)
	dy := computeDY(top, y0)

	// Note that FixedMul(Fixed, FDot6) produces results in FDot6
	e.X = FDot6ToFixed(x0 + FDot6(FixedMul(slope, Fixed(dy))))
	e.DxDy = slope
	e.FirstY = top
	e.LastY = bot - 1

	return true
}

func (e *Edge) chopLineWithClip(clip *geom.IRect) {
	top := e.FirstY
	// clip the line to the top
	if top < clip.Top {
		e.X += e.DxDy * Fixed(clip.Top-top)
		e.FirstY = clip.Top
	}
}

///////////////////////////////////////////////////////////////////////////////

// maxCoeffShift limits the number of lines used to approximate a curve (2^6); segment counts must fit the uint8
// segmentCount.
const maxCoeffShift = 6

// cheapDistance approximates the distance from (0,0) to (dx, dy) as max + min/2.
func cheapDistance(dx, dy FDot6) FDot6 {
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	if dx > dy {
		return dx + (dy / 2)
	}
	return dy + (dx / 2)
}

// clz counts leading zeros of a 32-bit value; 32 for zero.
func clz(x uint32) int {
	return bits.LeadingZeros32(x)
}

// diffToShift computes how many subdivision levels (each cutting error by 1/4) reduce the distance heuristic to roughly
// 1/8 pixel (adjusted by accuracy).
func diffToShift(dx, dy FDot6, accuracy int) int {
	// cheap calc of distance from center of p0-p2 to the center of the curve
	dist := cheapDistance(dx, dy)

	// shift down dist (it is currently in dot6); down by 3 should give us 1/8 pixel accuracy (assuming our dist is
	// accurate...); this is chosen by heuristic: make it as big as possible (to minimize segments), but small enough so
	// that our curves still look smooth
	dist = (dist + (1 << (2 + accuracy))) >> (3 + accuracy)

	// each subdivision (shift value) cuts this dist (error) by 1/4
	return (32 - clz(uint32(dist))) >> 1
}

// fdot6ToFixedDiv2 computes fdot6ToFixed(value >> 1) without throwing away the low bit — a modified up-shift.
func fdot6ToFixedDiv2(value FDot6) Fixed {
	return Fixed(value << (16 - 6 - 1))
}

// QuadraticEdge is a Y-monotonic quadratic curve edge, walked as a sequence of forward-differenced line segments.
type QuadraticEdge struct {
	Edge
	// qx/qy are the non-rounded points the current line segment ends at. qDxDt/qDyDt are the first derivatives
	// evaluated at the midpoint of the next segment, stored at half value to avoid overflow (the forward-difference
	// step multiplies by 2/N instead of 1/N). qD2xDt2/qD2yDt2 are the second derivatives pre-multiplied by 1/N.
	qx, qy         Fixed
	qDxDt, qDyDt   Fixed
	qD2xDt2        Fixed
	qD2yDt2        Fixed
	qLastX, qLastY Fixed // exact curve endpoints, used verbatim for the last segment
}

// SetQuadratic sets up the line segments for a Y-monotonic quad. Returns false if the whole curve is of height 0.
func (q *QuadraticEdge) SetQuadratic(pts []geom.Point) bool {
	x0 := FloatToFDot6(pts[0].X)
	y0 := FloatToFDot6(pts[0].Y)
	x1 := FloatToFDot6(pts[1].X)
	y1 := FloatToFDot6(pts[1].Y)
	x2 := FloatToFDot6(pts[2].X)
	y2 := FloatToFDot6(pts[2].Y)

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

	// compute the number of steps needed (2^shift) from the distance between the curve at t=1/2 and the midpoint of the
	// p0..p2 baseline: 1/4 (-p0 + 2 p1 - p2).
	deltaX := (2*x1 - x0 - x2) >> 2
	deltaY := (2*y1 - y0 - y2) >> 2
	shift := diffToShift(deltaX, deltaY, 0)

	// We need at least 2 line segments to be able to save the derivatives as half their values to avoid overflow.
	if shift == 0 {
		shift = 1
	} else if shift > maxCoeffShift {
		shift = maxCoeffShift
	}

	q.Winding = winding
	q.EdgeType = EdgeTypeQuad
	q.segmentCount = uint8(1 << shift)

	// In polynomial form At^2 + Bt + C: A = p0 - 2p1 + p2, B = 2(p1 - p0), C = p0. A and B are stored at 1/2 their
	// value to guard against overflow, applying a 2x scale during nextSegment; hence curveShift = shift - 1.
	q.curveShift = uint8(shift - 1)

	aHalf := fdot6ToFixedDiv2(x0 - x1 - x1 + x2)
	bHalf := FDot6ToFixed(x1 - x0)

	// slope at the midpoint of the first segment: 1/2 (2A*t + B) at t = 1/(2N) => A/2 * 1/2^shift + B/2
	q.qDxDt = bHalf + (aHalf >> shift)
	// second derivative pre-multiplied by 1/N: A/2 * 1/2^(shift-1)
	q.qD2xDt2 = aHalf >> (shift - 1)

	aHalf = fdot6ToFixedDiv2(y0 - y1 - y1 + y2)
	bHalf = FDot6ToFixed(y1 - y0)

	q.qDyDt = bHalf + (aHalf >> shift)
	q.qD2yDt2 = aHalf >> (shift - 1)

	q.qx = FDot6ToFixed(x0)
	q.qy = FDot6ToFixed(y0)
	q.qLastX = FDot6ToFixed(x2)
	q.qLastY = FDot6ToFixed(y2)

	q.owner = q
	return q.nextSegment()
}

func (q *QuadraticEdge) nextSegment() bool {
	var success bool
	count := int(q.segmentCount)
	oldx := q.qx
	oldy := q.qy
	dx := q.qDxDt
	dy := q.qDyDt
	var newx, newy Fixed
	shift := int(q.curveShift)

	for {
		count--
		if count > 0 {
			newx = oldx + (dx >> shift)
			dx += q.qD2xDt2
			newy = oldy + (dy >> shift)
			dy += q.qD2yDt2
		} else { // last segment
			newx = q.qLastX
			newy = q.qLastY
		}
		success = q.updateLine(oldx, oldy, newx, newy)
		oldx = newx
		oldy = newy
		if count <= 0 || success {
			break
		}
	}

	q.qx = newx
	q.qy = newy
	q.qDxDt = dx
	q.qDyDt = dy
	q.segmentCount = uint8(count)
	return success
}

///////////////////////////////////////////////////////////////////////////////

// fdot6UpShift converts an FDot6 value to Fixed by shifting up.
func fdot6UpShift(x FDot6, upShift int) Fixed {
	return Fixed(x << upShift)
}

// cubicDeltaFromLine computes the cubic's maximum deviation from the p0-p3 baseline, using:
//
//	f(1/3) - b = (8a - 15b + 6c + d) / 27
//	f(2/3) - c = (a + 6b - 15c + 8d) / 27
//
// using 19/512 to approximate 1/27.
func cubicDeltaFromLine(a, b, c, d FDot6) FDot6 {
	oneThird := (a*8 - b*15 + 6*c + d) * 19 >> 9
	twoThird := (a + 6*b - c*15 + d*8) * 19 >> 9
	if oneThird < 0 {
		oneThird = -oneThird
	}
	if twoThird < 0 {
		twoThird = -twoThird
	}
	if oneThird > twoThird {
		return oneThird
	}
	return twoThird
}

// CubicEdge is a Y-monotonic cubic curve edge, walked as a sequence of forward-differenced line segments.
type CubicEdge struct {
	Edge
	// Analogous to QuadraticEdge's state, one derivative order deeper. cLastX/cLastY are the exact curve endpoints used
	// verbatim for the last segment so cumulative forward-difference error can't produce a dramatically different line.
	cx, cy           Fixed
	cDxDt, cDyDt     Fixed
	cD2xDt2, cD2yDt2 Fixed
	cD3xDt3, cD3yDt3 Fixed
	cLastX, cLastY   Fixed
	cubicDShift      uint8 // applied to cDxDt and cDyDt only
}

// SetCubic sets up the line segments for a Y-monotonic cubic. Returns false if the whole curve is of height 0.
func (c *CubicEdge) SetCubic(pts []geom.Point) bool {
	x0 := FloatToFDot6(pts[0].X)
	y0 := FloatToFDot6(pts[0].Y)
	x1 := FloatToFDot6(pts[1].X)
	y1 := FloatToFDot6(pts[1].Y)
	x2 := FloatToFDot6(pts[2].X)
	y2 := FloatToFDot6(pts[2].Y)
	x3 := FloatToFDot6(pts[3].X)
	y3 := FloatToFDot6(pts[3].Y)

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

	// compute the number of steps needed (1 << shift). Can't use (center of curve - center of baseline), since the
	// center of the curve need not be the max delta from the baseline (it could even be coincident), so we try just
	// looking at the two off-curve points.
	dx := cubicDeltaFromLine(x0, x1, x2, x3)
	dy := cubicDeltaFromLine(y0, y1, y2, y3)
	// add 1 (by observation)
	shift := diffToShift(dx, dy, 2) + 1
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
	c.segmentCount = uint8(1 << shift)
	c.curveShift = uint8(shift)
	c.cubicDShift = uint8(downShift)

	// In polynomial form At^3 + Bt^2 + Ct + D:
	//   A = -p0 + 3p1 - 3p2 + p3, B = 3p0 - 6p1 + 3p2, C = -3p0 + 3p1, D = p0.
	a := fdot6UpShift(x3+3*(x1-x2)-x0, upShift)
	b := fdot6UpShift(3*(x0-2*x1+x2), upShift)
	cc := fdot6UpShift(3*(x1-x0), upShift)

	c.cDxDt = (a >> (2 * shift)) + (b >> shift) + cc
	c.cD2xDt2 = (3 * a >> (shift - 1)) + 2*b
	c.cD3xDt3 = 3 * a >> (shift - 1)

	a = fdot6UpShift(y3+3*(y1-y2)-y0, upShift)
	b = fdot6UpShift(3*(y0-2*y1+y2), upShift)
	cc = fdot6UpShift(3*(y1-y0), upShift)

	c.cDyDt = (a >> (2 * shift)) + (b >> shift) + cc
	c.cD2yDt2 = (3 * a >> (shift - 1)) + 2*b
	c.cD3yDt3 = 3 * a >> (shift - 1)

	c.cx = FDot6ToFixed(x0)
	c.cy = FDot6ToFixed(y0)
	c.cLastX = FDot6ToFixed(x3)
	c.cLastY = FDot6ToFixed(y3)

	c.owner = c
	return c.nextSegment()
}

func (c *CubicEdge) nextSegment() bool {
	var success bool
	count := int(c.segmentCount)
	oldx := c.cx
	oldy := c.cy
	var newx, newy Fixed
	ddshift := int(c.curveShift)
	dshift := int(c.cubicDShift)

	for {
		count--
		if count > 0 {
			newx = oldx + (c.cDxDt >> dshift)
			c.cDxDt += c.cD2xDt2 >> ddshift
			c.cD2xDt2 += c.cD3xDt3

			newy = oldy + (c.cDyDt >> dshift)
			c.cDyDt += c.cD2yDt2 >> ddshift
			c.cD2yDt2 += c.cD3yDt3
		} else { // last segment
			newx = c.cLastX
			newy = c.cLastY
		}

		// we want to say assert(oldy <= newy), but our finite fixedpoint doesn't always achieve that, so we have to
		// explicitly pin it here.
		if newy < oldy {
			newy = oldy
		}

		success = c.updateLine(oldx, oldy, newx, newy)
		oldx = newx
		oldy = newy
		if count <= 0 || success {
			break
		}
	}

	c.cx = newx
	c.cy = newy
	c.segmentCount = uint8(count)
	return success
}
