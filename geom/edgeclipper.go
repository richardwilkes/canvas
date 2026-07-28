// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// EdgeClipper clips a single line/quad/cubic segment against a rect, emitting the clipped pieces (with vertical closure
// lines for the portions chopped away on the left/right) via an iterator. The path-level driver that feeds whole paths
// through this lives in the path package (ClipPath).

package geom

// ClipVerb identifies a segment emitted by the EdgeClipper.
type ClipVerb uint8

// ClipVerb values.
const (
	ClipVerbLine ClipVerb = iota
	ClipVerbQuad
	ClipVerbCubic
)

// EdgeClipper limits. ClipCubic chops at the Y extrema and then at the X extrema, but each axis' derivative is a single
// quadratic, so there are at most 2 chops per axis over the whole curve (chopping in Y does not manufacture new X
// extrema); at most 4 chops means at most 5 monotonic pieces reach clipMonoCubic. A piece emits at most 3 verbs and 8
// points — the left vline (2), the clipped cubic (4), then the right vline (2) — for a worst case of 15 verbs and 40
// points. Quads bound tighter (at most 1 chop per axis, so 3 pieces of 3 verbs / 7 points) and a clipped line yields at
// most 3 verbs / 6 points, so the cubic case sets the limits below.
const (
	edgeClipperMaxVerbs  = 18
	edgeClipperMaxPoints = 54
)

// EdgeClipper clips line/quad/cubic segments against a rect. It is initialized with a segment and a clip via
// ClipLine/ClipQuad/ClipCubic, after which Next is called until it returns done.
type EdgeClipper struct {
	currPoint         int
	currVerb          int
	verbStop          int
	points            [edgeClipperMaxPoints]Point
	verbs             [edgeClipperMaxVerbs]ClipVerb
	canCullToTheRight bool
}

// NewEdgeClipper returns an EdgeClipper. canCullToTheRight enables dropping segments wholly to the right of the clip
// (valid for filling when a right edge is guaranteed by other machinery, invalid for convex fills that need both
// edges).
func NewEdgeClipper(canCullToTheRight bool) *EdgeClipper {
	return &EdgeClipper{canCullToTheRight: canCullToTheRight}
}

// Reset reinitializes the clipper to the state NewEdgeClipper produces, reusing the receiver's storage so a hot caller
// can clip many paths without allocating a fresh clipper each time.
func (e *EdgeClipper) Reset(canCullToTheRight bool) {
	*e = EdgeClipper{canCullToTheRight: canCullToTheRight}
}

// CanCullToTheRight reports the culling mode.
func (e *EdgeClipper) CanCullToTheRight() bool {
	return e.canCullToTheRight
}

func quickRejectY(bounds, clip Rect) bool {
	return bounds.Top >= clip.Bottom || bounds.Bottom <= clip.Top
}

func clampLE(value *float32, maxValue float32) {
	if *value > maxValue {
		*value = maxValue
	}
}

func clampGE(value *float32, minValue float32) {
	if *value < minValue {
		*value = minValue
	}
}

// sortIncreasingY copies src into dst sorted to be increasing in Y (src must be monotonic in Y) and returns whether the
// order was reversed.
func sortIncreasingY(dst, src []Point, count int) bool {
	// we need the data to be monotonically increasing in Y
	if src[0].Y > src[count-1].Y {
		for i := 0; i < count; i++ {
			dst[i] = src[count-i-1]
		}
		return true
	}
	copy(dst[:count], src[:count])
	return false
}

// ClipLine clips the line p0..p1 against clip, returning true if anything was emitted.
func (e *EdgeClipper) ClipLine(p0, p1 Point, clip Rect) bool {
	e.currPoint = 0
	e.currVerb = 0

	var lines [LineClipperMaxPoints]Point
	pts := [2]Point{p0, p1}
	lineCount := ClipLine(&pts, clip, &lines, e.canCullToTheRight)
	for i := 0; i < lineCount; i++ {
		e.appendLine(lines[i], lines[i+1])
	}

	e.verbStop = e.currVerb
	e.currPoint = 0
	e.currVerb = 0
	return e.verbStop != 0
}

// chopMonoQuadAt solves F(t) = target where F is the quad interpolation of the coefficients c0, c1, c2.
func chopMonoQuadAt(c0, c1, c2, target float32) (float32, bool) {
	// Solve F(t) = y where F(t) := [0](1-t)^2 + 2[1]t(1-t) + [2]t^2. We solve for t, using the quadratic equation,
	// hence we have to rearrange our coefficients to look like At^2 + Bt + C.
	a := c0 - c1 - c1 + c2
	b := 2 * (c1 - c0)
	c := c0 - target
	var roots [2]float32
	if FindUnitQuadRoots(a, b, c, &roots) > 0 {
		return roots[0], true
	}
	return 0, false
}

func chopMonoQuadAtY(pts []Point, y float32) (float32, bool) {
	return chopMonoQuadAt(pts[0].Y, pts[1].Y, pts[2].Y, y)
}

func chopMonoQuadAtX(pts []Point, x float32) (float32, bool) {
	return chopMonoQuadAt(pts[0].X, pts[1].X, pts[2].X, x)
}

// chopQuadInY modifies pts in place so that it is clipped in Y to the clip rect.
func chopQuadInY(pts []Point, clip Rect) {
	var tmp [5]Point

	// are we partially above
	if pts[0].Y < clip.Top {
		if t, ok := chopMonoQuadAtY(pts, clip.Top); ok {
			// take the 2nd chopped quad
			ChopQuadAt(pts, tmp[:], t)
			// clamp to clean up imprecise numerics in the chop
			tmp[2].Y = clip.Top
			clampGE(&tmp[3].Y, clip.Top)

			pts[0] = tmp[2]
			pts[1] = tmp[3]
		} else {
			// if chopMonoQuadAtY failed, then we may have hit inexact numerics so we just clamp against the top
			for i := 0; i < 3; i++ {
				if pts[i].Y < clip.Top {
					pts[i].Y = clip.Top
				}
			}
		}
	}

	// are we partially below
	if pts[2].Y > clip.Bottom {
		if t, ok := chopMonoQuadAtY(pts, clip.Bottom); ok {
			ChopQuadAt(pts, tmp[:], t)
			// clamp to clean up imprecise numerics in the chop
			clampLE(&tmp[1].Y, clip.Bottom)
			tmp[2].Y = clip.Bottom

			pts[1] = tmp[1]
			pts[2] = tmp[2]
		} else {
			// if chopMonoQuadAtY failed, then we may have hit inexact numerics so we just clamp against the bottom
			for i := 0; i < 3; i++ {
				if pts[i].Y > clip.Bottom {
					pts[i].Y = clip.Bottom
				}
			}
		}
	}
}

// clipMonoQuad clips a quad that is monotonic in both X and Y.
func (e *EdgeClipper) clipMonoQuad(srcPts []Point, clip Rect) {
	var pts [3]Point
	reverse := sortIncreasingY(pts[:], srcPts, 3)

	// are we completely above or below
	if pts[2].Y <= clip.Top || pts[0].Y >= clip.Bottom {
		return
	}

	// Now chop so that pts is contained within clip in Y
	chopQuadInY(pts[:], clip)

	if pts[0].X > pts[2].X {
		pts[0], pts[2] = pts[2], pts[0]
		reverse = !reverse
	}

	// Now chop in X as needed, and record the segments

	if pts[2].X <= clip.Left { // wholly to the left
		e.appendVLine(clip.Left, pts[0].Y, pts[2].Y, reverse)
		return
	}
	if pts[0].X >= clip.Right { // wholly to the right
		if !e.canCullToTheRight {
			e.appendVLine(clip.Right, pts[0].Y, pts[2].Y, reverse)
		}
		return
	}

	var tmp [5]Point

	// are we partially to the left
	if pts[0].X < clip.Left {
		if t, ok := chopMonoQuadAtX(pts[:], clip.Left); ok {
			ChopQuadAt(pts[:], tmp[:], t)
			e.appendVLine(clip.Left, tmp[0].Y, tmp[2].Y, reverse)
			// clamp to clean up imprecise numerics in the chop
			tmp[2].X = clip.Left
			clampGE(&tmp[3].X, clip.Left)

			pts[0] = tmp[2]
			pts[1] = tmp[3]
		} else {
			// if chopMonoQuadAtX failed, then we may have hit inexact numerics so we just clamp against the left
			e.appendVLine(clip.Left, pts[0].Y, pts[2].Y, reverse)
			return
		}
	}

	// are we partially to the right
	if pts[2].X > clip.Right {
		if t, ok := chopMonoQuadAtX(pts[:], clip.Right); ok {
			ChopQuadAt(pts[:], tmp[:], t)
			// clamp to clean up imprecise numerics in the chop
			clampLE(&tmp[1].X, clip.Right)
			tmp[2].X = clip.Right

			e.appendQuad(tmp[:3], reverse)
			e.appendVLine(clip.Right, tmp[2].Y, tmp[4].Y, reverse)
		} else {
			// if chopMonoQuadAtX failed, then we may have hit inexact numerics so we just clamp against the right
			pts[1].X = min32(pts[1].X, clip.Right)
			pts[2].X = min32(pts[2].X, clip.Right)
			e.appendQuad(pts[:], reverse)
		}
	} else { // wholly inside the clip
		e.appendQuad(pts[:], reverse)
	}
}

// ClipQuad clips the quad srcPts (3 points) against clip, returning true if anything was emitted.
func (e *EdgeClipper) ClipQuad(srcPts []Point, clip Rect) bool {
	e.currPoint = 0
	e.currVerb = 0

	bounds := BoundsOrEmpty(srcPts[:3])
	if !quickRejectY(bounds, clip) {
		var monoY [5]Point
		countY := ChopQuadAtYExtrema(srcPts, monoY[:])
		for y := 0; y <= countY; y++ {
			var monoX [5]Point
			countX := ChopQuadAtXExtrema(monoY[y*2:y*2+3], monoX[:])
			for x := 0; x <= countX; x++ {
				e.clipMonoQuad(monoX[x*2:x*2+3], clip)
			}
		}
	}

	e.verbStop = e.currVerb
	e.currPoint = 0
	e.currVerb = 0
	return e.verbStop != 0
}

// monoCubicClosestT performs a binary search for the t that evaluates closest to x on a monotonic cubic axis (c0..c3
// are the axis coordinates: X or Y from each of the curve's 4 control points).
func monoCubicClosestT(c0, c1, c2, c3, x float32) float32 {
	t := float32(0.5)
	var lastT float32
	var bestT float32
	step := float32(0.25)
	d := c0
	a := c3 + 3*(c1-c2) - d
	b := 3 * (c2 - c1 - c1 + d)
	c := 3 * (c1 - d)
	x -= d
	closest := ScalarMax
	for {
		loc := ((a*t+b)*t + c) * t
		dist := ScalarAbs(loc - x)
		if closest > dist {
			closest = dist
			bestT = t
		}
		lastT = t
		if loc < x {
			t += step
		} else {
			t -= step
		}
		step *= 0.5
		if closest <= 0.25 || lastT == t {
			break
		}
	}
	return bestT
}

// chopMonoCubicAtYClamped performs the accurate double-precision chop with the closest-t fallback.
func chopMonoCubicAtYClamped(src, dst []Point, y float32) {
	if ChopMonoCubicAtY(src, dst, y) {
		return
	}
	ChopCubicAt(src, dst, monoCubicClosestT(src[0].Y, src[1].Y, src[2].Y, src[3].Y, y))
}

// chopMonoCubicAtXClamped is the X-axis analog of chopMonoCubicAtYClamped.
func chopMonoCubicAtXClamped(src, dst []Point, x float32) {
	if ChopMonoCubicAtX(src, dst, x) {
		return
	}
	ChopCubicAt(src, dst, monoCubicClosestT(src[0].X, src[1].X, src[2].X, src[3].X, x))
}

// chopCubicInY modifies pts in place so that it is clipped in Y to the clip rect.
func chopCubicInY(pts []Point, clip Rect) {
	// are we partially above
	if pts[0].Y < clip.Top {
		var tmp [7]Point
		chopMonoCubicAtYClamped(pts, tmp[:], clip.Top)

		// For a large range in the points, we can do a poor job of chopping, such that the t we computed resulted in
		// the lower cubic still being partly above the clip.
		//
		// If just the first or first 2 Y values are above clip.Top, we can just smash them down. If the first 3 Ys are
		// above clip.Top, we can't smash all 3, as that can really distort the cubic. In this case, we take the first
		// output (tmp[3..6]) and treat it as a guess, and re-chop against clip.Top. Then we fall through to checking if
		// we need to smash the first 1 or 2 Y values.
		if tmp[3].Y < clip.Top && tmp[4].Y < clip.Top && tmp[5].Y < clip.Top {
			var tmp2 [4]Point
			copy(tmp2[:], tmp[3:7])
			chopMonoCubicAtYClamped(tmp2[:], tmp[:], clip.Top)
		}

		// tmp[3, 4].Y should both be below clip.Top. Since we can't trust the numerics of the chopper, we force those
		// conditions now
		tmp[3].Y = clip.Top
		clampGE(&tmp[4].Y, clip.Top)

		pts[0] = tmp[3]
		pts[1] = tmp[4]
		pts[2] = tmp[5]
	}

	// are we partially below
	if pts[3].Y > clip.Bottom {
		var tmp [7]Point
		chopMonoCubicAtYClamped(pts, tmp[:], clip.Bottom)
		tmp[3].Y = clip.Bottom
		clampLE(&tmp[2].Y, clip.Bottom)

		pts[1] = tmp[1]
		pts[2] = tmp[2]
		pts[3] = tmp[3]
	}
}

// clipMonoCubic clips a cubic that is monotonic in both X and Y.
func (e *EdgeClipper) clipMonoCubic(src []Point, clip Rect) {
	var pts [4]Point
	reverse := sortIncreasingY(pts[:], src, 4)

	// are we completely above or below
	if pts[3].Y <= clip.Top || pts[0].Y >= clip.Bottom {
		return
	}

	// Now chop so that pts is contained within clip in Y
	chopCubicInY(pts[:], clip)

	if pts[0].X > pts[3].X {
		pts[0], pts[3] = pts[3], pts[0]
		pts[1], pts[2] = pts[2], pts[1]
		reverse = !reverse
	}

	// Now chop in X as needed, and record the segments

	if pts[3].X <= clip.Left { // wholly to the left
		e.appendVLine(clip.Left, pts[0].Y, pts[3].Y, reverse)
		return
	}
	if pts[0].X >= clip.Right { // wholly to the right
		if !e.canCullToTheRight {
			e.appendVLine(clip.Right, pts[0].Y, pts[3].Y, reverse)
		}
		return
	}

	// are we partially to the left
	if pts[0].X < clip.Left {
		var tmp [7]Point
		chopMonoCubicAtXClamped(pts[:], tmp[:], clip.Left)
		e.appendVLine(clip.Left, tmp[0].Y, tmp[3].Y, reverse)

		// tmp[3, 4].X should both be to the right of clip.Left. Since we can't trust the numerics of the chopper, we
		// force those conditions now
		tmp[3].X = clip.Left
		clampGE(&tmp[4].X, clip.Left)

		pts[0] = tmp[3]
		pts[1] = tmp[4]
		pts[2] = tmp[5]
	}

	// are we partially to the right
	if pts[3].X > clip.Right {
		var tmp [7]Point
		chopMonoCubicAtXClamped(pts[:], tmp[:], clip.Right)
		tmp[3].X = clip.Right
		clampLE(&tmp[2].X, clip.Right)

		e.appendCubic(tmp[:4], reverse)
		e.appendVLine(clip.Right, tmp[3].Y, tmp[6].Y, reverse)
	} else { // wholly inside the clip
		e.appendCubic(pts[:], reverse)
	}
}

// tooBigForReliableFloatMath is the largest float extent for which chopping at extrema and clip values still computes
// reliably. Chosen by experiment.
func tooBigForReliableFloatMath(r Rect) bool {
	const limit = 1 << 22
	return r.Left < -limit || r.Top < -limit || r.Right > limit || r.Bottom > limit
}

// ClipCubic clips the cubic srcPts (4 points) against clip, returning true if anything was emitted.
func (e *EdgeClipper) ClipCubic(srcPts []Point, clip Rect) bool {
	e.currPoint = 0
	e.currVerb = 0

	bounds := BoundsOrEmpty(srcPts[:4])
	// check if we're clipped out vertically
	if bounds.Bottom > clip.Top && bounds.Top < clip.Bottom {
		if tooBigForReliableFloatMath(bounds) {
			// can't safely clip the cubic, so we give up and draw a line (which we can safely clip)
			return e.ClipLine(srcPts[0], srcPts[3], clip)
		}
		var monoY [10]Point
		countY := ChopCubicAtYExtrema(srcPts, monoY[:])
		for y := 0; y <= countY; y++ {
			var monoX [10]Point
			countX := ChopCubicAtXExtrema(monoY[y*3:y*3+4], monoX[:])
			for x := 0; x <= countX; x++ {
				e.clipMonoCubic(monoX[x*3:x*3+4], clip)
			}
		}
	}

	e.verbStop = e.currVerb
	e.currPoint = 0
	e.currVerb = 0
	return e.verbStop != 0
}

func (e *EdgeClipper) appendLine(p0, p1 Point) {
	e.verbs[e.currVerb] = ClipVerbLine
	e.currVerb++
	e.points[e.currPoint] = p0
	e.points[e.currPoint+1] = p1
	e.currPoint += 2
}

func (e *EdgeClipper) appendVLine(x, y0, y1 float32, reverse bool) {
	e.verbs[e.currVerb] = ClipVerbLine
	e.currVerb++
	if reverse {
		y0, y1 = y1, y0
	}
	e.points[e.currPoint] = Point{X: x, Y: y0}
	e.points[e.currPoint+1] = Point{X: x, Y: y1}
	e.currPoint += 2
}

func (e *EdgeClipper) appendQuad(pts []Point, reverse bool) {
	e.verbs[e.currVerb] = ClipVerbQuad
	e.currVerb++
	if reverse {
		e.points[e.currPoint] = pts[2]
		e.points[e.currPoint+2] = pts[0]
	} else {
		e.points[e.currPoint] = pts[0]
		e.points[e.currPoint+2] = pts[2]
	}
	e.points[e.currPoint+1] = pts[1]
	e.currPoint += 3
}

func (e *EdgeClipper) appendCubic(pts []Point, reverse bool) {
	e.verbs[e.currVerb] = ClipVerbCubic
	e.currVerb++
	if reverse {
		for i := 0; i < 4; i++ {
			e.points[e.currPoint+i] = pts[3-i]
		}
	} else {
		copy(e.points[e.currPoint:e.currPoint+4], pts[:4])
	}
	e.currPoint += 4
}

// Next returns the next clipped segment, filling pts (which must hold at least 4 points) with 2, 3 or 4 points for a
// line, quad or cubic. It returns ok == false when iteration is complete.
func (e *EdgeClipper) Next(pts []Point) (verb ClipVerb, ok bool) {
	if e.currVerb >= e.verbStop {
		return 0, false
	}
	verb = e.verbs[e.currVerb]
	e.currVerb++
	switch verb {
	case ClipVerbLine:
		copy(pts[:2], e.points[e.currPoint:e.currPoint+2])
		e.currPoint += 2
	case ClipVerbQuad:
		copy(pts[:3], e.points[e.currPoint:e.currPoint+3])
		e.currPoint += 3
	case ClipVerbCubic:
		copy(pts[:4], e.points[e.currPoint:e.currPoint+4])
		e.currPoint += 4
	}
	return verb, true
}
