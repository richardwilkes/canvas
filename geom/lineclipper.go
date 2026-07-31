// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Line-segment clipping against a rect.

package geom

// Line-clipper limits: clipping can turn one line into at most LineClipperMaxPoints-1 segments, stored as
// LineClipperMaxPoints sequential points.
const (
	LineClipperMaxPoints              = 4
	LineClipperMaxClippedLineSegments = LineClipperMaxPoints - 1
)

// pinUnsorted pins value to the (possibly unsorted) limits.
func pinUnsorted(value, limit0, limit1 float32) float32 {
	if limit1 < limit0 {
		limit0, limit1 = limit1, limit0
	}
	switch {
	case value < limit0:
		return limit0
	case value > limit1:
		return limit1
	}
	return value
}

// sectWithHorizontal returns the X coordinate of the intersection of the line src with the horizontal line at y.
func sectWithHorizontal(src *[2]Point, y float32) float32 {
	dy := src[1].Y - src[0].Y
	if ScalarNearlyZero(dy) {
		return (src[0].X + src[1].X) / 2
	}
	// need the extra precision so we don't compute a value that exceeds our original limits
	x0 := float64(src[0].X)
	y0 := float64(src[0].Y)
	x1 := float64(src[1].X)
	y1 := float64(src[1].Y)
	result := x0 + (float64(y)-y0)*(x1-x0)/(y1-y0)
	// The computed X value might still exceed [X0..X1] due to quantum flux when the doubles were added and subtracted,
	// so we have to pin the answer :(
	return float32(pinUnsortedDouble(result, x0, x1))
}

func pinUnsortedDouble(value, limit0, limit1 float64) float64 {
	if limit1 < limit0 {
		limit0, limit1 = limit1, limit0
	}
	switch {
	case value < limit0:
		return limit0
	case value > limit1:
		return limit1
	}
	return value
}

// sectWithVertical returns the Y coordinate of the intersection of the line src with the vertical line at x.
func sectWithVertical(src *[2]Point, x float32) float32 {
	dx := src[1].X - src[0].X
	if ScalarNearlyZero(dx) {
		return (src[0].Y + src[1].Y) / 2
	}
	// need the extra precision so we don't compute a value that exceeds our original limits
	x0 := float64(src[0].X)
	y0 := float64(src[0].Y)
	x1 := float64(src[1].X)
	y1 := float64(src[1].Y)
	return float32(y0 + (float64(x)-x0)*(y1-y0)/(x1-x0))
}

// sectClampWithVertical is sectWithVertical, clamped: the caller expects the result to be between src[0].Y and src[1].Y
// (unsorted), but float/double numerics might compute a value slightly outside of that, so clamp afterwards.
func sectClampWithVertical(src *[2]Point, x float32) float32 {
	return pinUnsorted(sectWithVertical(src, x), src[0].Y, src[1].Y)
}

// nestedLT reports a < b, or a == b when the interval is degenerate (dim zero).
func nestedLT(a, b, dim float32) bool {
	return a <= b && (a < b || dim > 0)
}

// containsNoEmptyCheck returns true if outer contains inner, even if inner is empty (Rect.Contains always returns false
// for an empty inner).
func containsNoEmptyCheck(outer, inner Rect) bool {
	return outer.Left <= inner.Left && outer.Top <= inner.Top &&
		outer.Right >= inner.Right && outer.Bottom >= inner.Bottom
}

// IntersectLine intersects the segment src against clip. If a non-empty segment results, it is stored in dst and true
// is returned; otherwise dst is unspecified. clip must be sorted (Left <= Right and Top <= Bottom); see the note on
// ClipLine.
func IntersectLine(src *[2]Point, clip Rect, dst *[2]Point) bool {
	// The sorted bounds of the two points.
	bounds := Rect{
		Left:   min32(src[0].X, src[1].X),
		Top:    min32(src[0].Y, src[1].Y),
		Right:  max32(src[0].X, src[1].X),
		Bottom: max32(src[0].Y, src[1].Y),
	}
	if containsNoEmptyCheck(clip, bounds) {
		if src != dst {
			*dst = *src
		}
		return true
	}
	// check for no overlap, and only permit coincident edges if the line and the edge are colinear
	if nestedLT(bounds.Right, clip.Left, bounds.Width()) ||
		nestedLT(clip.Right, bounds.Left, bounds.Width()) ||
		nestedLT(bounds.Bottom, clip.Top, bounds.Height()) ||
		nestedLT(clip.Bottom, bounds.Top, bounds.Height()) {
		return false
	}

	var index0, index1 int
	if src[0].Y < src[1].Y {
		index0 = 0
		index1 = 1
	} else {
		index0 = 1
		index1 = 0
	}

	tmp := *src

	// now compute Y intersections
	if tmp[index0].Y < clip.Top {
		tmp[index0] = Point{X: sectWithHorizontal(src, clip.Top), Y: clip.Top}
	}
	if tmp[index1].Y > clip.Bottom {
		tmp[index1] = Point{X: sectWithHorizontal(src, clip.Bottom), Y: clip.Bottom}
	}

	if tmp[0].X < tmp[1].X {
		index0 = 0
		index1 = 1
	} else {
		index0 = 1
		index1 = 0
	}

	// check for quick-reject in X again, now that we may have been chopped
	if tmp[index1].X <= clip.Left || tmp[index0].X >= clip.Right {
		// usually we will return false, but we don't if the line is vertical and coincident with the clip.
		if tmp[0].X != tmp[1].X || tmp[0].X < clip.Left || tmp[0].X > clip.Right {
			return false
		}
	}

	if tmp[index0].X < clip.Left {
		tmp[index0] = Point{X: clip.Left, Y: sectWithVertical(&tmp, clip.Left)}
	}
	if tmp[index1].X > clip.Right {
		tmp[index1] = Point{X: clip.Right, Y: sectWithVertical(&tmp, clip.Right)}
	}
	*dst = tmp
	return true
}

// ClipLine clips the line pts[0]..pts[1] against clip, ignoring portions completely above or below. Portions to the
// left or right become vertical segments aligned to the clip edge. Returns the number of resulting segments, whose
// endpoints are stored sequentially in lines: segment i runs lines[i]..lines[i+1].
//
// canCullToTheRight suppresses output only for a segment lying *wholly* at or to the right of clip.Right, which is
// dropped entirely (0 segments). It does not suppress the right-edge vertical segment emitted for a line that merely
// extends past the right edge while still crossing the clip — that piece is emitted either way.
//
// clip must be sorted (Left <= Right and Top <= Bottom). The edge tests compare against Left/Top/Right/Bottom
// directly, so a reversed rect such as RectLTRB(100, 0, 0, 100) silently emits a vertical "left edge" line at X=100 —
// outside the rect's own right edge — instead of rejecting the segment. Call Rect.Sorted first if the rect may be
// reversed.
func ClipLine(pts *[2]Point, clip Rect, lines *[LineClipperMaxPoints]Point, canCullToTheRight bool) int {
	var index0, index1 int
	if pts[0].Y < pts[1].Y {
		index0 = 0
		index1 = 1
	} else {
		index0 = 1
		index1 = 0
	}

	// Check if we're completely clipped out in Y (above or below)
	if pts[index1].Y <= clip.Top { // we're above the clip
		return 0
	}
	if pts[index0].Y >= clip.Bottom { // we're below the clip
		return 0
	}

	// Chop in Y to produce a single segment, stored in tmp[0..1]
	tmp := *pts

	// now compute intersections
	if pts[index0].Y < clip.Top {
		tmp[index0] = Point{X: sectWithHorizontal(pts, clip.Top), Y: clip.Top}
	}
	if tmp[index1].Y > clip.Bottom {
		tmp[index1] = Point{X: sectWithHorizontal(pts, clip.Bottom), Y: clip.Bottom}
	}

	// Chop it into 1..3 segments that are wholly within the clip in X.

	// temp storage for up to 3 segments
	var resultStorage [LineClipperMaxPoints]Point
	var result []Point // points to our results, either tmp or resultStorage
	lineCount := 1
	var reverse bool

	if pts[0].X < pts[1].X {
		index0 = 0
		index1 = 1
		reverse = false
	} else {
		index0 = 1
		index1 = 0
		reverse = true
	}

	switch {
	case tmp[index1].X <= clip.Left: // wholly to the left
		tmp[0].X = clip.Left
		tmp[1].X = clip.Left
		result = tmp[:]
		reverse = false
	case tmp[index0].X >= clip.Right: // wholly to the right
		if canCullToTheRight {
			return 0
		}
		tmp[0].X = clip.Right
		tmp[1].X = clip.Right
		result = tmp[:]
		reverse = false
	default:
		result = resultStorage[:]
		r := 0

		if tmp[index0].X < clip.Left {
			resultStorage[r] = Point{X: clip.Left, Y: tmp[index0].Y}
			r++
			resultStorage[r] = Point{X: clip.Left, Y: sectClampWithVertical(&tmp, clip.Left)}
		} else {
			resultStorage[r] = tmp[index0]
		}
		r++

		if tmp[index1].X > clip.Right {
			resultStorage[r] = Point{X: clip.Right, Y: sectClampWithVertical(&tmp, clip.Right)}
			r++
			resultStorage[r] = Point{X: clip.Right, Y: tmp[index1].Y}
		} else {
			resultStorage[r] = tmp[index1]
		}

		lineCount = r
	}

	// Now copy the results into the caller's lines[] parameter
	if reverse {
		// copy the pts in reverse order to maintain winding order
		for i := 0; i <= lineCount; i++ {
			lines[lineCount-i] = result[i]
		}
	} else {
		copy(lines[:lineCount+1], result[:lineCount+1])
	}
	return lineCount
}
