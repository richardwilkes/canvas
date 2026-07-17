// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Detecting whether a contour is, when filled, identical to a rectangle. The AA scan converter's fat-anti-rect fast
// path consumes this; the canvas and GPU backend may add more consumers later.

package path

import "github.com/richardwilkes/canvas/geom"

// rectContour holds the result of a successful isRectContour detection.
type rectContour struct {
	pointsConsumed int
	verbsConsumed  int
	rect           geom.Rect
	direction      geom.PathDirection
	isClosed       bool
}

// rectMakeDir encodes an axis-aligned direction 0..3 such that opposite directions differ by 2 (xor == 2) and
// perpendicular ones by 1 in the low bit.
func rectMakeDir(dx, dy float32) int8 {
	d := int8(0)
	if dx != 0 {
		d |= 1
	}
	if dx > 0 || dy > 0 {
		d |= 2
	}
	return d
}

// isRectContour detects whether the leading contour of the given verb/point stream is, when filled, identical to a
// rectangle. With allowPartial false it inspects the whole stream (any extra geometry disqualifies the rect); with
// allowPartial true it stops after the first closed contour (recording how much it consumed, for nested-rect detection
// later).
func isRectContour(points []geom.Point, verbs []Verb, allowPartial bool) (rectContour, bool) {
	var rc rectContour
	if len(points) < 4 {
		return rc, false
	}

	currVerb := 0
	verbCnt := len(verbs)

	corners := 0
	var closeXY geom.Point   // used to determine if final line falls on a diagonal
	var lineStart geom.Point // used to construct line from previous point
	firstPtIdx := -1         // first point in the rect (last of first moves)
	lastPtIdx := -1          // last point in the rect (last of lines or first if closed)
	var firstCorner, thirdCorner geom.Point
	pi := 0
	savePtsIdx := -1                          // used to allow the caller to iterate through a pair of rects
	directions := [5]int8{-1, -1, -1, -1, -1} // -1 to 3; -1 is uninitialized
	closedOrMoved := false
	autoClose := false
	insertClose := false

	for currVerb < verbCnt && (!allowPartial || !autoClose) {
		verb := verbs[currVerb]
		if insertClose {
			verb = VerbClose
		}
		skipVerbAdvance := false
		switch verb {
		case VerbClose, VerbLine:
			if verb == VerbClose {
				savePtsIdx = pi
				autoClose = true
				insertClose = false
			}
			var lineEnd geom.Point
			if verb == VerbClose {
				lineEnd = points[firstPtIdx]
			} else {
				lastPtIdx = pi
				lineEnd = points[pi]
				pi++
			}
			lineDelta := lineEnd.Sub(lineStart)
			if lineDelta.X != 0 && lineDelta.Y != 0 {
				return rc, false // diagonal
			}
			if !lineDelta.IsFinite() {
				return rc, false // path contains infinity or NaN
			}
			if lineStart == lineEnd {
				break // single point on side OK
			}
			nextDirection := rectMakeDir(lineDelta.X, lineDelta.Y) // 0 to 3
			if corners == 0 {
				directions[0] = nextDirection
				corners = 1
				closedOrMoved = false
				lineStart = lineEnd
				break
			}
			if closedOrMoved {
				return rc, false // closed followed by a line
			}
			if autoClose && nextDirection == directions[0] {
				break // colinear with first
			}
			closedOrMoved = autoClose
			if directions[corners-1] == nextDirection {
				if corners == 3 && verb == VerbLine {
					thirdCorner = lineEnd
				}
				lineStart = lineEnd
				break // colinear segment
			}
			directions[corners] = nextDirection
			corners++
			// opposite lines must point in opposite directions; xoring them should equal 2
			switch corners {
			case 2:
				firstCorner = lineStart
			case 3:
				if directions[0]^directions[2] != 2 {
					return rc, false
				}
				thirdCorner = lineEnd
			case 4:
				if directions[1]^directions[3] != 2 {
					return rc, false
				}
			default:
				return rc, false // too many direction changes
			}
			lineStart = lineEnd
		case VerbQuad, VerbConic, VerbCubic:
			return rc, false // quadratic, cubic not allowed
		case VerbMove:
			if allowPartial && !autoClose && directions[0] >= 0 {
				insertClose = true
				skipVerbAdvance = true // try move again after the inserted close
				break
			}
			if corners == 0 {
				firstPtIdx = pi
			} else {
				closeXY = points[firstPtIdx].Sub(points[lastPtIdx])
				if closeXY.X != 0 && closeXY.Y != 0 {
					return rc, false // we're diagonal, abort
				}
			}
			lineStart = points[pi]
			pi++
			closedOrMoved = true
		default:
		}
		if !skipVerbAdvance {
			currVerb++
		}
	}

	// Success if 4 corners and first point equals last
	if corners < 3 || corners > 4 {
		return rc, false
	}
	// check if close generates diagonal
	closeXY = points[firstPtIdx].Sub(points[lastPtIdx])
	if closeXY.X != 0 && closeXY.Y != 0 {
		return rc, false
	}

	rc.rect = geom.RectLTRB(
		min(firstCorner.X, thirdCorner.X), min(firstCorner.Y, thirdCorner.Y),
		max(firstCorner.X, thirdCorner.X), max(firstCorner.Y, thirdCorner.Y),
	)
	rc.isClosed = autoClose
	if directions[0] == (directions[1]+1)&3 {
		rc.direction = geom.DirectionCW
	} else {
		rc.direction = geom.DirectionCCW
	}
	if savePtsIdx >= 0 {
		rc.pointsConsumed = savePtsIdx
	}
	rc.verbsConsumed = currVerb
	return rc, true
}

// IsRect reports whether the path, when filled, is identical to a rectangle, and returns that rectangle.
func (p *Path) IsRect() (geom.Rect, bool) {
	rc, ok := isRectContour(p.points, p.verbs, false)
	if !ok {
		return geom.Rect{}, false
	}
	return rc.rect, true
}

// IsRectWithDirection is like IsRect, but also reports whether the rect contour is closed and its winding direction.
func (p *Path) IsRectWithDirection() (rect geom.Rect, isClosed bool, dir geom.PathDirection, ok bool) {
	rc, ok := isRectContour(p.points, p.verbs, false)
	if !ok {
		return geom.Rect{}, false, geom.DirectionCW, false
	}
	return rc.rect, rc.isClosed, rc.direction, true
}

// IsNestedFillRects reports whether the path, when filled, is two nested rectangles (as stroking a rect produces),
// storing them outer-first in rects.
func (p *Path) IsNestedFillRects(rects *[2]geom.Rect) bool {
	return p.IsNestedFillRectsWithDirections(rects, nil)
}

// IsNestedFillRectsWithDirections is IsNestedFillRects, with dirs (when non-nil) receiving each rect contour's winding
// direction, outer-first.
func (p *Path) IsNestedFillRectsWithDirections(rects *[2]geom.Rect, dirs *[2]geom.PathDirection) bool {
	rc, ok := isRectContour(p.points, p.verbs, true)
	if !ok {
		return false
	}
	first := rc.rect
	rc2, ok := isRectContour(p.points[rc.pointsConsumed:], p.verbs[rc.verbsConsumed:], false)
	if !ok {
		return false
	}
	second := rc2.rect
	switch {
	case first.ContainsRect(second):
		if rects != nil {
			rects[0] = first
			rects[1] = second
		}
		if dirs != nil {
			dirs[0] = rc.direction
			dirs[1] = rc2.direction
		}
		return true
	case second.ContainsRect(first):
		if rects != nil {
			rects[0] = second
			rects[1] = first
		}
		if dirs != nil {
			dirs[0] = rc2.direction
			dirs[1] = rc.direction
		}
		return true
	default:
		return false
	}
}
