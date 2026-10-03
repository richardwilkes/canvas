// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Computing the winding direction of the outermost contour (the one holding the global y-max).

package path

import "github.com/richardwilkes/canvas/geom"

// FirstDirection is the winding direction (CW/CCW) established by the outermost contour of a path.
type FirstDirection uint8

// FirstDirection values.
const (
	FirstDirectionCW FirstDirection = iota
	FirstDirectionCCW
	FirstDirectionUnknown
)

// convexityToFirstDirection converts a cached convex Convexity value into the corresponding FirstDirection.
func convexityToFirstDirection(c Convexity) FirstDirection {
	switch c {
	case ConvexityConvexCW:
		return FirstDirectionCW
	case ConvexityConvexCCW:
		return FirstDirectionCCW
	default:
		return FirstDirectionUnknown
	}
}

// contourIter walks the point runs of each contour in a path (the moveTo point included).
type contourIter struct {
	points  []geom.Point
	verbs   []Verb
	ptStart int
	ptCount int
	verbIdx int
	done    bool
}

func newContourIter(p *Path) contourIter {
	it := contourIter{points: p.points, verbs: p.verbs}
	it.next()
	return it
}

func (it *contourIter) pts() []geom.Point {
	return it.points[it.ptStart : it.ptStart+it.ptCount]
}

func (it *contourIter) next() {
	if it.verbIdx >= len(it.verbs) {
		it.done = true
	}
	if it.done {
		return
	}

	// skip pts of prev contour
	it.ptStart += it.ptCount

	ptCount := 1 // moveTo
	i := it.verbIdx + 1
	for ; i < len(it.verbs); i++ {
		v := it.verbs[i]
		if v == VerbMove {
			break
		}
		ptCount += v.pointsForVerb()
	}
	it.ptCount = ptCount
	it.verbIdx = i
}

// crossProd returns the cross product of (p1 - p0) and (p2 - p0), lazily promoted to double precision when the float32
// subtracts underflow to a zero cross.
func crossProd(p0, p1, p2 geom.Point) float32 {
	cross := p1.Sub(p0).Cross(p2.Sub(p0))
	if cross == 0 {
		p0x := float64(p0.X)
		p0y := float64(p0.Y)
		p1x := float64(p1.X)
		p1y := float64(p1.Y)
		p2x := float64(p2.X)
		p2y := float64(p2.Y)
		cross = geom.DoubleToScalar((p1x-p0x)*(p2y-p0y) - (p1y-p0y)*(p2x-p0x))
	}
	return cross
}

// findMaxY returns the index of the first point with the maximum Y coordinate.
func findMaxY(pts []geom.Point) int {
	maxY := pts[0].Y
	firstIndex := 0
	for i := 1; i < len(pts); i++ {
		if pts[i].Y > maxY {
			maxY = pts[i].Y
			firstIndex = i
		}
	}
	return firstIndex
}

// findDiffPt walks from index by inc (mod n) to the first point that differs from pts[index], or back to index if all
// points coincide.
func findDiffPt(pts []geom.Point, index, n, inc int) int {
	i := index
	for {
		i = (i + inc) % n
		if i == index {
			break
		}
		if pts[index] != pts[i] {
			break
		}
	}
	return i
}

// findMinMaxXAtY returns, starting at index and moving forward, the xmin and xmax indices of the contiguous points that
// share pts[index]'s Y.
func findMinMaxXAtY(pts []geom.Point, index, n int) (minIndex, maxIndex int) {
	y := pts[index].Y
	minX := pts[index].X
	maxX := minX
	minIndex = index
	maxIndex = index
	for i := index + 1; i < n; i++ {
		if pts[i].Y != y {
			break
		}
		x := pts[i].X
		if x < minX {
			minX = x
			minIndex = i
		} else if x > maxX {
			maxX = x
			maxIndex = i
		}
	}
	return minIndex, maxIndex
}

// crossToDir converts the sign of a cross product into a winding direction.
func crossToDir(cross float32) FirstDirection {
	if cross > 0 {
		return FirstDirectionCW
	}
	return FirstDirectionCCW
}

// ComputeFirstDirection answers from the path's cached convexity when it is convex (possibly Unknown — if computing
// convexity failed to find a direction, Unknown is the right answer); otherwise the winding of the contour holding the
// global y-max decides.
func (p *Path) ComputeFirstDirection() FirstDirection {
	if p.convexity.IsConvex() {
		return convexityToFirstDirection(p.convexity)
	}
	return p.ComputeFirstDirectionRaw()
}

// ComputeFirstDirectionRaw computes the winding direction from the geometry alone, without consulting the convexity
// cache (the stroker's stroke-and-fill lane calls this form).
func (p *Path) ComputeFirstDirectionRaw() FirstDirection {
	// Keep the cross product of the contour holding the global y-max. The first contour alone could be a hole (e.g. in
	// 'o'), correctly wound the opposite way.
	ymax := p.Bounds().Top
	ymaxCross := float32(0)

	for it := newContourIter(p); !it.done; it.next() {
		n := it.ptCount
		if n < 3 {
			continue
		}

		pts := it.pts()
		cross := float32(0)
		index := findMaxY(pts)
		if pts[index].Y < ymax {
			continue
		}

		// With several points at the y-max, the order of their x-min and x-max gives the direction.
		if pts[(index+1)%n].Y == pts[index].Y {
			minIndex, maxIndex := findMinMaxXAtY(pts, index, n)
			if minIndex == maxIndex {
				cross = tryCrossProd(pts, index, n)
				if cross == 0 {
					continue
				}
			} else {
				// Only the sign of the index difference matters.
				cross = float32(minIndex - maxIndex)
			}
		} else {
			cross = tryCrossProd(pts, index, n)
			if cross == 0 {
				continue
			}
		}

		if cross != 0 {
			ymax = pts[index].Y
			ymaxCross = cross
		}
	}

	if ymaxCross != 0 {
		return crossToDir(ymaxCross)
	}
	return FirstDirectionUnknown
}

// tryCrossProd finds prev/next indices forming non-degenerate vectors from pts[index] and takes the cross product,
// falling back to the X-spread when the points are horizontal. Returns 0 when the contour is completely degenerate,
// signaling the caller to skip to the next contour.
func tryCrossProd(pts []geom.Point, index, n int) float32 {
	// Step by n-1 rather than -1 to keep the % operand non-negative.
	prev := findDiffPt(pts, index, n, n-1)
	if prev == index {
		return 0
	}
	next := findDiffPt(pts, index, n, 1)
	cross := crossProd(pts[prev], pts[index], pts[next])
	// Ideally this would keep walking away from the degeneracy until the points diverge.
	if cross == 0 && pts[prev].Y == pts[index].Y && pts[next].Y == pts[index].Y {
		cross = pts[index].X - pts[next].X
	}
	return cross
}
