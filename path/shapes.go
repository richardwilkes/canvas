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
	"github.com/richardwilkes/canvas/geom"
)

// sqrt2Over2 is sqrt(2)/2, the conic weight for a quarter circle.
const sqrt2Over2 = 0.707106781

// pointIterator cycles through N fixed points starting at startIndex, in the given direction.
type pointIterator struct {
	pts     [8]geom.Point
	n       uint
	current uint
	advance uint
}

func (it *pointIterator) init(n uint, dir geom.PathDirection, startIndex uint) {
	it.n = n
	it.current = startIndex % n
	if dir == geom.DirectionCW {
		it.advance = 1
	} else {
		it.advance = n - 1
	}
}

func (it *pointIterator) cur() geom.Point {
	return it.pts[it.current]
}

func (it *pointIterator) next() geom.Point {
	it.current = (it.current + it.advance) % it.n
	return it.cur()
}

// rectPointIterator returns a pointIterator over r's four corners: L-T, R-T, R-B, L-B.
func rectPointIterator(r geom.Rect, dir geom.PathDirection, startIndex uint) pointIterator {
	var it pointIterator
	it.pts[0] = geom.Pt(r.Left, r.Top)
	it.pts[1] = geom.Pt(r.Right, r.Top)
	it.pts[2] = geom.Pt(r.Right, r.Bottom)
	it.pts[3] = geom.Pt(r.Left, r.Bottom)
	it.init(4, dir, startIndex)
	return it
}

// ovalPointIterator returns a pointIterator over oval's four edge midpoints: top, right, bottom, left.
func ovalPointIterator(oval geom.Rect, dir geom.PathDirection, startIndex uint) pointIterator {
	cx := oval.CenterX()
	cy := oval.CenterY()
	var it pointIterator
	it.pts[0] = geom.Pt(cx, oval.Top)
	it.pts[1] = geom.Pt(oval.Right, cy)
	it.pts[2] = geom.Pt(cx, oval.Bottom)
	it.pts[3] = geom.Pt(oval.Left, cy)
	it.init(4, dir, startIndex)
	return it
}

// rrectPointIterator returns a pointIterator over rrect's eight corner-arc endpoints, for the uniform-radii RRect
// shape.
func rrectPointIterator(rrect geom.RRect, dir geom.PathDirection, startIndex uint) pointIterator {
	bounds := rrect.Rect
	l := bounds.Left
	t := bounds.Top
	r := bounds.Right
	b := bounds.Bottom
	rx := rrect.RadiusX
	ry := rrect.RadiusY
	var it pointIterator
	it.pts[0] = geom.Pt(l+rx, t)
	it.pts[1] = geom.Pt(r-rx, t)
	it.pts[2] = geom.Pt(r, t+ry)
	it.pts[3] = geom.Pt(r, b-ry)
	it.pts[4] = geom.Pt(r-rx, b)
	it.pts[5] = geom.Pt(l+rx, b)
	it.pts[6] = geom.Pt(l, b-ry)
	it.pts[7] = geom.Pt(l, t+ry)
	it.init(8, dir, startIndex)
	return it
}

// isEffectivelyEmpty reports whether the path has zero verbs, or a single moveTo (which would be overwritten when
// adding another contour).
func (p *Path) isEffectivelyEmpty() bool {
	return len(p.verbs) <= 1
}

// finishSimpleShape records the convexity after adding a simple shape: one added to an effectively empty path leaves
// the path convex in the shape's direction; otherwise the convexity is unknown.
func (p *Path) finishSimpleShape(wasEffectivelyEmpty bool, dir geom.PathDirection) {
	if wasEffectivelyEmpty {
		p.convexity = directionToConvexity(dir)
	} else {
		p.convexity = ConvexityUnknown
	}
}

// AddRect adds a closed rect contour starting at the upper-left corner.
func (p *Path) AddRect(rect geom.Rect, dir geom.PathDirection) *Path {
	return p.AddRectWithStart(rect, dir, 0)
}

// AddRectWithStart adds a closed rect contour; the verbs added are Move, Line, Line, Line, Close; startIndex selects
// the initial corner (0 upper-left, 1 upper-right, 2 lower-right, 3 lower-left).
func (p *Path) AddRectWithStart(rect geom.Rect, dir geom.PathDirection, startIndex uint) *Path {
	wasEmpty := p.isEffectivelyEmpty()

	iter := rectPointIterator(rect, dir, startIndex)
	p.MoveToPt(iter.cur())
	p.LineToPt(iter.next())
	p.LineToPt(iter.next())
	p.LineToPt(iter.next())
	p.Close()

	p.finishSimpleShape(wasEmpty, dir)
	return p
}

// AddOval adds a closed oval contour, starting at point index 1 (the right edge midpoint).
func (p *Path) AddOval(oval geom.Rect, dir geom.PathDirection) *Path {
	// legacy start index: 1
	return p.AddOvalWithStart(oval, dir, 1)
}

// AddOvalWithStart adds a closed oval contour built from four conics, starting at startPointIndex.
func (p *Path) AddOvalWithStart(oval geom.Rect, dir geom.PathDirection, startPointIndex uint) *Path {
	// If addOval() is called after previous moveTo(), this path is still marked as an oval. This is used to fit into
	// WebKit's calling sequences. We can't simply check isEmpty() in this case, as additional moveTo() would mark the
	// path non empty.
	isOval := p.hasOnlyMoveTos()

	wasEmpty := p.isEffectivelyEmpty()

	ovalIter := ovalPointIterator(oval, dir, startPointIndex)
	rectStart := startPointIndex
	if dir != geom.DirectionCW {
		rectStart++
	}
	rectIter := rectPointIterator(oval, dir, rectStart)

	p.MoveToPt(ovalIter.cur())
	for i := 0; i < 4; i++ {
		p.ConicToPt(rectIter.next(), ovalIter.next(), sqrt2Over2)
	}
	p.Close()

	p.finishSimpleShape(wasEmpty, dir)

	if isOval {
		p.isa = isAOval
	}
	return p
}

// AddCircle adds a closed oval contour inscribed in the square around (x, y) with radius r. No effect when the radius
// is zero or negative.
func (p *Path) AddCircle(x, y, r float32, dir geom.PathDirection) *Path {
	if r > 0 {
		p.AddOval(geom.RectLTRB(x-r, y-r, x+r, y+r), dir)
	}
	return p
}

// AddRRect adds a closed round-rect contour, using the legacy start indices (6 for CW, 7 for CCW).
func (p *Path) AddRRect(rrect geom.RRect, dir geom.PathDirection) *Path {
	if dir == geom.DirectionCW {
		return p.AddRRectWithStart(rrect, dir, 6)
	}
	return p.AddRRectWithStart(rrect, dir, 7)
}

// AddRRectWithStart adds a closed contour of alternating lines and conics for a round rect. Degenerate round rects
// collapse to AddRect or AddOval.
func (p *Path) AddRRectWithStart(rrect geom.RRect, dir geom.PathDirection, startIndex uint) *Path {
	isRRect := p.hasOnlyMoveTos()
	bounds := rrect.Rect

	switch rrect.Type {
	case geom.RRectRect, geom.RRectEmpty:
		// degenerate(rect) => radii points are collapsing
		return p.AddRectWithStart(bounds, dir, (startIndex+1)/2)
	case geom.RRectOval:
		// degenerate(oval) => line points are collapsing
		return p.AddOvalWithStart(bounds, dir, startIndex/2)
	default:
	}

	wasEmpty := p.isEffectivelyEmpty()

	// we start with a conic on odd indices when moving CW vs. even indices when moving CCW
	startsWithConic := (startIndex&1 == 1) == (dir == geom.DirectionCW)

	rrectIter := rrectPointIterator(rrect, dir, startIndex)
	// Corner iterator indices follow the collapsed radii model, adjusted such that the start pt is "behind" the radii
	// start pt.
	rectStartIndex := startIndex / 2
	if dir != geom.DirectionCW {
		rectStartIndex++
	}
	rectIter := rectPointIterator(bounds, dir, rectStartIndex)

	p.MoveToPt(rrectIter.cur())
	if startsWithConic {
		for i := 0; i < 3; i++ {
			p.ConicToPt(rectIter.next(), rrectIter.next(), sqrt2Over2)
			p.LineToPt(rrectIter.next())
		}
		p.ConicToPt(rectIter.next(), rrectIter.next(), sqrt2Over2)
		// final line is accomplished by close()
	} else {
		for i := 0; i < 4; i++ {
			p.LineToPt(rrectIter.next())
			p.ConicToPt(rectIter.next(), rrectIter.next(), sqrt2Over2)
		}
	}
	p.Close()

	p.finishSimpleShape(wasEmpty, dir)

	if isRRect {
		p.isa = isARRect
	}
	return p
}

// AddRoundRect adds a closed round-rect contour with corner radii (rx, ry). Negative radii are a no-op; zero radii (or
// oversized ones, after clamping) degenerate per geom.MakeRRect's rules.
func (p *Path) AddRoundRect(rect geom.Rect, rx, ry float32, dir geom.PathDirection) *Path {
	if rx < 0 || ry < 0 {
		return p
	}
	return p.AddRRect(geom.MakeRRect(rect, rx, ry), dir)
}

// AddPoly adds a contour with a moveTo to pts[0] and lines to the remaining points, optionally closed. Unlike MoveTo,
// the leading move verb is always appended (never replaces a trailing moveTo).
func (p *Path) AddPoly(pts []geom.Point, closePath bool) *Path {
	if len(pts) == 0 {
		return p
	}

	p.setLastMoveToIndex(len(p.points))

	p.growForVerb(VerbMove, 0)
	p.points = append(p.points, pts[0])
	if len(pts) > 1 {
		for i := 1; i < len(pts); i++ {
			p.growForVerb(VerbLine, 0)
			p.points = append(p.points, pts[i])
		}
	}

	if closePath {
		p.growForVerb(VerbClose, 0)
		if idx := p.lastMoveToIndex(); idx >= 0 {
			p.setLastMoveToIndex(^idx)
		}
	}

	return p.dirtyAfterEdit()
}

//////////////////////////////////////////////////////////////////////////////
// addPath

// AddPathOffset appends src's contours translated by (dx, dy), per mode.
func (p *Path) AddPathOffset(src *Path, dx, dy float32, mode AddPathMode) *Path {
	var matrix geom.Matrix
	matrix.SetTranslate(dx, dy)
	return p.AddPathMatrix(src, &matrix, mode)
}

// AddPath appends src's contours unchanged, per mode.
func (p *Path) AddPath(src *Path, mode AddPathMode) *Path {
	identity := geom.IdentityMatrix()
	return p.AddPathMatrix(src, &identity, mode)
}

// AddPathMatrix appends src's contours (transformed by matrix) as new contours, or extends the last contour when mode
// is AddPathExtend.
func (p *Path) AddPathMatrix(src *Path, matrix *geom.Matrix, mode AddPathMode) *Path {
	if src.IsEmpty() {
		return p
	}

	canReplaceThis := (mode == AddPathAppend && p.isEffectivelyEmpty()) || len(p.verbs) == 0
	if canReplaceThis && matrix.IsIdentity() {
		fillType := p.fillType
		p.copyFrom(src)
		p.fillType = fillType
		return p
	}

	// Detect if we're trying to add ourself
	if p == src {
		src = src.Clone()
	}

	if mode == AddPathAppend && !matrix.HasPerspective() {
		if srcIdx := src.lastMoveToIndex(); srcIdx >= 0 {
			p.setLastMoveToIndex(srcIdx + len(p.points))
		} else {
			p.setLastMoveToIndex(srcIdx - len(p.points))
		}
		p.segmentMask |= src.segmentMask
		p.boundsValid = false
		p.isa = isAGeneral
		p.verbs = append(p.verbs, src.verbs...)
		oldPtCount := len(p.points)
		p.points = append(p.points, src.points...)
		matrix.MapPointsInPlace(p.points[oldPtCount:])
		p.conicWeights = append(p.conicWeights, src.conicWeights...)
		return p.dirtyAfterEdit()
	}

	firstVerb := true
	iter := newRawIter(src)
	for {
		verb, pts, w, ok := iter.next()
		if !ok {
			break
		}
		switch verb {
		case VerbMove:
			mapped := matrix.MapPoint(pts[0])
			if firstVerb && mode == AddPathExtend && !p.IsEmpty() {
				p.injectMoveToIfNeeded() // In case last contour is closed
				lastPt, has := p.LastPt()
				// don't add lineTo if it is degenerate
				if !has || lastPt != mapped {
					p.LineToPt(mapped)
				}
			} else {
				p.MoveToPt(mapped)
			}
		case VerbLine:
			p.LineToPt(matrix.MapPoint(pts[1]))
		case VerbQuad:
			p.QuadToPt(matrix.MapPoint(pts[1]), matrix.MapPoint(pts[2]))
		case VerbConic:
			p.ConicToPt(matrix.MapPoint(pts[1]), matrix.MapPoint(pts[2]), w)
		case VerbCubic:
			p.CubicToPt(matrix.MapPoint(pts[1]), matrix.MapPoint(pts[2]), matrix.MapPoint(pts[3]))
		case VerbClose:
			p.Close()
		}
		firstVerb = false
	}
	return p
}

// ReverseAddPath appends src's contours in reverse order (each contour's verbs reversed).
func (p *Path) ReverseAddPath(src *Path) *Path {
	// Detect if we're trying to add ourself
	if p == src {
		src = src.Clone()
	}

	verbs := src.verbs
	pts := len(src.points)
	weights := len(src.conicWeights)

	needMove := true
	needClose := false
	for v := len(verbs) - 1; v >= 0; v-- {
		verb := verbs[v]
		n := verb.pointsForVerb()

		if needMove {
			pts--
			p.MoveToPt(src.points[pts])
			needMove = false
		}
		pts -= n
		switch verb {
		case VerbMove:
			if needClose {
				p.Close()
				needClose = false
			}
			needMove = true
			pts++ // so we see the point in "if needMove" above
		case VerbLine:
			p.LineToPt(src.points[pts])
		case VerbQuad:
			p.QuadToPt(src.points[pts+1], src.points[pts])
		case VerbConic:
			weights--
			p.ConicToPt(src.points[pts+1], src.points[pts], src.conicWeights[weights])
		case VerbCubic:
			p.CubicToPt(src.points[pts+2], src.points[pts+1], src.points[pts])
		case VerbClose:
			needClose = true
		}
	}
	return p
}

// ReversePathTo appends src's last contour reversed, ignoring src's final point (the caller is expected to already be
// there), and without emitting src's initial moveTo. If src has multiple contours, only the last is reversed.
func (p *Path) ReversePathTo(src *Path) *Path {
	if len(src.verbs) == 0 {
		return p
	}
	ptIdx := len(src.points) - 1
	weightIdx := len(src.conicWeights)
	for i := len(src.verbs) - 1; i >= 0; i-- {
		v := src.verbs[i]
		ptIdx -= v.pointsForVerb()
		switch v {
		case VerbMove:
			// if the path has multiple contours, stop after reversing the last
			return p
		case VerbLine:
			p.LineToPt(src.points[ptIdx])
		case VerbQuad:
			p.QuadToPt(src.points[ptIdx+1], src.points[ptIdx])
		case VerbConic:
			weightIdx--
			p.ConicToPt(src.points[ptIdx+1], src.points[ptIdx], src.conicWeights[weightIdx])
		case VerbCubic:
			p.CubicToPt(src.points[ptIdx+2], src.points[ptIdx+1], src.points[ptIdx])
		case VerbClose:
		}
	}
	return p
}
