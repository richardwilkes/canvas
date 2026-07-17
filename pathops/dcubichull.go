// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The convex hull of a cubic's four control points, used by the cubic hull-intersection quick reject that the cubic
// solver relies on. The hull may have 3 or 4 points; the returned order describes it (a cubic degenerating to a point
// or line is not considered here).

package pathops

import "math"

// otherTwo returns, given two of the four indices {0,1,2,3}, the XOR mask that yields the other two.
func otherTwo(one, two int) int {
	return (1 >> (3 - (one ^ two))) ^ 3
}

// cubicRotate rotates the cubic so the line from control point 'zero' to 'index' aligns with an axis. Returns false
// when those two points coincide.
func cubicRotate(cubic dCubic, zero, index int) (dCubic, bool) {
	var rotPath dCubic
	dy := cubic.pts[index].y - cubic.pts[zero].y
	dx := cubic.pts[index].x - cubic.pts[zero].x
	if approximatelyZero(dy) {
		if approximatelyZero(dx) {
			return rotPath, false
		}
		rotPath = cubic
		if dy != 0 {
			rotPath.pts[index].y = cubic.pts[zero].y
			mask := otherTwo(index, zero)
			side1 := index ^ mask
			side2 := zero ^ mask
			if approximatelyEqual(cubic.pts[side1].y, cubic.pts[zero].y) {
				rotPath.pts[side1].y = cubic.pts[zero].y
			}
			if approximatelyEqual(cubic.pts[side2].y, cubic.pts[zero].y) {
				rotPath.pts[side2].y = cubic.pts[zero].y
			}
		}
		return rotPath, true
	}
	for i := 0; i < 4; i++ {
		rotPath.pts[i].x = cubic.pts[i].x*dx + cubic.pts[i].y*dy
		rotPath.pts[i].y = cubic.pts[i].y*dx - cubic.pts[i].x*dy
	}
	return rotPath, true
}

// cubicHullSide classifies x's sign: 0 if negative, 1 if zero, 2 if positive.
func cubicHullSide(x float64) int {
	return b2i(x > 0) + b2i(x >= 0)
}

// convexHull fills order with the (3 or 4) hull indices in order and returns the count.
func (c dCubic) convexHull(order *[4]int) int {
	// find top point
	yMin := 0
	for index := 1; index < 4; index++ {
		if c.pts[yMin].y > c.pts[index].y ||
			(c.pts[yMin].y == c.pts[index].y && c.pts[yMin].x > c.pts[index].x) {
			yMin = index
		}
	}
	order[0] = yMin
	midX := -1
	backupYMin := -1
	for pass := 0; pass < 2; pass++ {
		for index := 0; index < 4; index++ {
			if index == yMin {
				continue
			}
			// rotate line from (yMin, index) to axis; see if the remaining two points are both above or below and use
			// this to find mid
			mask := otherTwo(yMin, index)
			side1 := yMin ^ mask
			side2 := index ^ mask
			rotPath, ok := cubicRotate(c, yMin, index) // ! if cbc[yMin]==cbc[idx]
			if !ok {
				order[1] = side1
				order[2] = side2
				return 3
			}
			sides := cubicHullSide(rotPath.pts[side1].y - rotPath.pts[yMin].y)
			sides ^= cubicHullSide(rotPath.pts[side2].y - rotPath.pts[yMin].y)
			switch sides {
			case 2: // '2' means one remaining point <0, one >0
				if midX >= 0 {
					// one of the control points is equal to an end point
					order[0] = 0
					order[1] = 3
					if c.pts[1].equals(c.pts[0]) || c.pts[1].equals(c.pts[3]) {
						order[2] = 2
						return 3
					}
					if c.pts[2].equals(c.pts[0]) || c.pts[2].equals(c.pts[3]) {
						order[2] = 1
						return 3
					}
					// one of the control points may be very nearly but not exactly equal
					dist10 := c.pts[1].distanceSquared(c.pts[0])
					dist13 := c.pts[1].distanceSquared(c.pts[3])
					dist20 := c.pts[2].distanceSquared(c.pts[0])
					dist23 := c.pts[2].distanceSquared(c.pts[3])
					smallest1distSq := math.Min(dist10, dist13)
					smallest2distSq := math.Min(dist20, dist23)
					if approximatelyZero(math.Min(smallest1distSq, smallest2distSq)) {
						if smallest1distSq < smallest2distSq {
							order[2] = 2
						} else {
							order[2] = 1
						}
						return 3
					}
				}
				midX = index
			case 0: // '0' means both to one side or the other
				backupYMin = index
			}
		}
		if midX >= 0 {
			break
		}
		if backupYMin < 0 {
			break
		}
		yMin = backupYMin
		backupYMin = -1
	}
	if midX < 0 {
		midX = yMin ^ 3 // choose any other point
	}
	mask := otherTwo(yMin, midX)
	least := yMin ^ mask
	most := midX ^ mask
	order[0] = yMin
	order[1] = least
	// see if mid value is on same side of line (least, most) as yMin
	midPath, ok := cubicRotate(c, least, most) // ! if cbc[least]==cbc[most]
	if !ok {
		order[2] = midX
		return 3
	}
	midSides := cubicHullSide(midPath.pts[yMin].y - midPath.pts[least].y)
	midSides ^= cubicHullSide(midPath.pts[midX].y - midPath.pts[least].y)
	if midSides != 2 { // if mid point is not between
		order[2] = most
		return 3 // result is a triangle
	}
	order[2] = midX
	order[3] = most
	return 4 // result is a quadrilateral
}
