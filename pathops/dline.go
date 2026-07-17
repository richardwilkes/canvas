// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A double-precision line segment plus the point-on-line queries (ptAtT, exactPoint, nearPoint) and the axis-aligned
// exact/near helpers the horizontal/vertical intersection lanes use.

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// dLine is a segment defined by two double-precision endpoints.
type dLine struct {
	pts [2]dPoint
}

// set widens the float32 host points to double precision.
func (l *dLine) set(pts [2]geom.Point) {
	l.pts[0].set(pts[0])
	l.pts[1].set(pts[1])
}

// ptAtT evaluates the line at parameter t via linear interpolation between the two endpoints.
func (l dLine) ptAtT(t float64) dPoint {
	if t == 0 {
		return l.pts[0]
	}
	if t == 1 {
		return l.pts[1]
	}
	oneT := 1 - t
	return dPoint{
		x: oneT*l.pts[0].x + t*l.pts[1].x,
		y: oneT*l.pts[0].y + t*l.pts[1].y,
	}
}

// nearRay projects a perpendicular from xy onto the line and reports whether the resulting distance is within a ULP
// tolerance of the line's largest ordinal magnitude.
func (l dLine) nearRay(xy dPoint) bool {
	length := l.pts[1].sub(l.pts[0]) // the x/y magnitudes of the line
	denom := length.x*length.x + length.y*length.y
	ab0 := xy.sub(l.pts[0])
	numer := length.x*ab0.x + ab0.y*length.y
	t := numer / denom
	realPt := l.ptAtT(t)
	dist := realPt.distance(xy)
	// find the ordinal in the original line with the largest unsigned exponent
	tiniest := min(min(min(l.pts[0].x, l.pts[0].y), l.pts[1].x), l.pts[1].y)
	largest := max(max(max(l.pts[0].x, l.pts[0].y), l.pts[1].x), l.pts[1].y)
	largest = max(largest, -tiniest)
	return roughlyEqualUlps(largest, largest+dist)
}

// exactPoint returns 0 or 1 if xy is exactly an endpoint, else -1.
func (l dLine) exactPoint(xy dPoint) float64 {
	if xy.equals(l.pts[0]) { // do cheapest test first
		return 0
	}
	if xy.equals(l.pts[1]) {
		return 1
	}
	return -1
}

// nearPoint returns the t of xy projected onto the line if it lies within the segment and within ULP tolerance of it,
// else -1. When unequal is non-nil it is set to whether the point differs from the line at float32 precision.
func (l dLine) nearPoint(xy dPoint, unequal *bool) float64 {
	if !almostBetweenUlps(l.pts[0].x, xy.x, l.pts[1].x) ||
		!almostBetweenUlps(l.pts[0].y, xy.y, l.pts[1].y) {
		return -1
	}
	// project a perpendicular ray from the point to the line; find the T on the line
	length := l.pts[1].sub(l.pts[0]) // the x/y magnitudes of the line
	denom := length.x*length.x + length.y*length.y
	ab0 := xy.sub(l.pts[0])
	numer := length.x*ab0.x + ab0.y*length.y
	if !between(0, numer, denom) {
		return -1
	}
	if denom == 0 {
		return 0
	}
	t := numer / denom
	realPt := l.ptAtT(t)
	dist := realPt.distance(xy)
	// find the ordinal in the original line with the largest unsigned exponent
	tiniest := math.Min(math.Min(math.Min(l.pts[0].x, l.pts[0].y), l.pts[1].x), l.pts[1].y)
	largest := math.Max(math.Max(math.Max(l.pts[0].x, l.pts[0].y), l.pts[1].x), l.pts[1].y)
	largest = math.Max(largest, -tiniest)
	if !almostEqualUlpsPin(largest, largest+dist) {
		return -1
	}
	if unequal != nil {
		*unequal = float32(largest) != float32(largest+dist)
	}
	t = pinT(t) // a looser pin breaks skpwww_lptemp_com_3
	return t
}

// exactPointH returns 0 or 1 if xy exactly matches the left/right endpoint of the horizontal segment at height y, else
// -1.
func exactPointH(xy dPoint, left, right, y float64) float64 {
	if xy.y == y {
		if xy.x == left {
			return 0
		}
		if xy.x == right {
			return 1
		}
	}
	return -1
}

// nearPointH returns the t of xy projected onto the horizontal segment [left,right] at height y, if it lies within the
// segment and within ULP tolerance of it, else -1.
func nearPointH(xy dPoint, left, right, y float64) float64 {
	if !almostBequalUlps(xy.y, y) {
		return -1
	}
	if !almostBetweenUlps(left, xy.x, right) {
		return -1
	}
	t := (xy.x - left) / (right - left)
	t = pinT(t)
	realPtX := (1-t)*left + t*right
	distU := dVector{x: xy.y - y, y: xy.x - realPtX}
	distSq := distU.x*distU.x + distU.y*distU.y
	dist := math.Sqrt(distSq)
	tiniest := math.Min(math.Min(y, left), right)
	largest := math.Max(math.Max(y, left), right)
	largest = math.Max(largest, -tiniest)
	if !almostEqualUlps(largest, largest+dist) {
		return -1
	}
	return t
}

// exactPointV returns 0 or 1 if xy exactly matches the top/bottom endpoint of the vertical segment at x, else -1.
func exactPointV(xy dPoint, top, bottom, x float64) float64 {
	if xy.x == x {
		if xy.y == top {
			return 0
		}
		if xy.y == bottom {
			return 1
		}
	}
	return -1
}

// nearPointV returns the t of xy projected onto the vertical segment [top,bottom] at x, if it lies within the segment
// and within ULP tolerance of it, else -1.
func nearPointV(xy dPoint, top, bottom, x float64) float64 {
	if !almostBequalUlps(xy.x, x) {
		return -1
	}
	if !almostBetweenUlps(top, xy.y, bottom) {
		return -1
	}
	t := (xy.y - top) / (bottom - top)
	t = pinT(t)
	realPtY := (1-t)*top + t*bottom
	distU := dVector{x: xy.x - x, y: xy.y - realPtY}
	distSq := distU.x*distU.x + distU.y*distU.y
	dist := math.Sqrt(distSq)
	tiniest := math.Min(math.Min(x, top), bottom)
	largest := math.Max(math.Max(x, top), bottom)
	largest = math.Max(largest, -tiniest)
	if !almostEqualUlps(largest, largest+dist) {
		return -1
	}
	return t
}
