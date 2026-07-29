// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The intersection-phase walker: addIntersectTs takes two contours (or one contour against itself), enumerates every
// pair of segments whose bounds overlap, computes the intersection(s) via the six curve/curve and the line/curve
// solvers, and threads the resulting t values into the op data model — linking coincident points across the two
// segments' pt-t loops and recording coincident runs in the opCoincidence tracker.
//
// Debug-only assertions and tracing are omitted. Pathops computes in double precision, so t values flow as float64; the
// intersection point is compared/stored at float32 (geom.Point) precision. The driver (addIntersections) runs the
// double loop shared by the boolean-op and simplify entry points; it assumes the contour list is already bounds-sorted
// (sortContourList, common.go, runs first).

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// addIntersections is the "find all intersections between segments" loop shared by the boolean-op and simplify entry
// points: for each contour, intersect it against itself and every following contour, stopping the inner advance as soon
// as addIntersectTs reports the remaining contours are entirely below (the bounds-sort invariant). Always returns true
// today (a false addIntersectTs return only prunes the inner walk).
func addIntersections(contourList *opContourHead, coincidence *opCoincidence) bool {
	current := contourList.listHead()
	for current != nil {
		next := current
		for addIntersectTs(current, next, coincidence) {
			next = next.next
			if next == nil {
				break
			}
		}
		current = current.next
	}
	return true
}

// addIntersectTs intersects every segment of test against every segment of next. Returns false when test lies entirely
// above next (the caller stops advancing next), true otherwise.
func addIntersectTs(test, next *opContour, coincidence *opCoincidence) bool {
	if test != next {
		if almostLessUlps(float64(test.bounds.bottom), float64(next.bounds.top)) {
			return false
		}
		// OPTIMIZATION: outset contour bounds a smidgen instead?
		if !boundsIntersects(test.bounds, next.bounds) {
			return true
		}
	}
	var wt intersectionHelper
	wt.init(test)
	for {
		var wn intersectionHelper
		wn.init(next)
		// When intersecting a contour with itself, only look at segments following wt's.
		if test != next || wn.startAfter(&wt) {
			for {
				if boundsIntersects(wt.bounds(), wn.bounds()) {
					addIntersectSegments(&wt, &wn, coincidence)
				}
				if !wn.advance() {
					break
				}
			}
		}
		if !wt.advance() {
			break
		}
	}
	return true
}

// addIntersectSegments computes the intersection(s) of the two segments the cursors point at and threads the results
// into the data model. This is the body of the inner double loop in addIntersectTs.
func addIntersectSegments(wt, wn *intersectionHelper, coincidence *opCoincidence) {
	ts := newIntersections()
	swap := false
	switch wt.segmentType() {
	case horizontalLineSegment:
		swap = true
		switch wn.segmentType() {
		case horizontalLineSegment, verticalLineSegment, lineSegment:
			ts.horizontalLine(dLineOf(wn), wt.left(), wt.right(), wt.y(), wt.xFlipped())
		case quadSegment:
			ts.horizontalQuad(dQuadOf(wn), wt.left(), wt.right(), wt.y(), wt.xFlipped())
		case conicSegment:
			ts.horizontalConic(dConicOf(wn), wt.left(), wt.right(), wt.y(), wt.xFlipped())
		case cubicSegment:
			ts.horizontalCubic(dCubicOf(wn), wt.left(), wt.right(), wt.y(), wt.xFlipped())
		}
	case verticalLineSegment:
		swap = true
		switch wn.segmentType() {
		case horizontalLineSegment, verticalLineSegment, lineSegment:
			ts.verticalLine(dLineOf(wn), wt.top(), wt.bottom(), wt.x(), wt.yFlipped())
		case quadSegment:
			ts.verticalQuad(dQuadOf(wn), wt.top(), wt.bottom(), wt.x(), wt.yFlipped())
		case conicSegment:
			ts.verticalConic(dConicOf(wn), wt.top(), wt.bottom(), wt.x(), wt.yFlipped())
		case cubicSegment:
			ts.verticalCubic(dCubicOf(wn), wt.top(), wt.bottom(), wt.x(), wt.yFlipped())
		}
	case lineSegment:
		switch wn.segmentType() {
		case horizontalLineSegment:
			ts.horizontalLine(dLineOf(wt), wn.left(), wn.right(), wn.y(), wn.xFlipped())
		case verticalLineSegment:
			ts.verticalLine(dLineOf(wt), wn.top(), wn.bottom(), wn.x(), wn.yFlipped())
		case lineSegment:
			ts.intersectLineLine(dLineOf(wt), dLineOf(wn))
		case quadSegment:
			swap = true
			ts.intersectQuadLine(dQuadOf(wn), dLineOf(wt))
		case conicSegment:
			swap = true
			ts.intersectConicLine(dConicOf(wn), dLineOf(wt))
		case cubicSegment:
			swap = true
			ts.intersectCubicLine(dCubicOf(wn), dLineOf(wt))
		}
	case quadSegment:
		switch wn.segmentType() {
		case horizontalLineSegment:
			ts.horizontalQuad(dQuadOf(wt), wn.left(), wn.right(), wn.y(), wn.xFlipped())
		case verticalLineSegment:
			ts.verticalQuad(dQuadOf(wt), wn.top(), wn.bottom(), wn.x(), wn.yFlipped())
		case lineSegment:
			ts.intersectQuadLine(dQuadOf(wt), dLineOf(wn))
		case quadSegment:
			ts.intersectQuadQuad(dQuadOf(wt), dQuadOf(wn))
		case conicSegment:
			swap = true
			ts.intersectConicQuad(dConicOf(wn), dQuadOf(wt))
		case cubicSegment:
			swap = true
			ts.intersectCubicQuad(dCubicOf(wn), dQuadOf(wt))
		}
	case conicSegment:
		switch wn.segmentType() {
		case horizontalLineSegment:
			ts.horizontalConic(dConicOf(wt), wn.left(), wn.right(), wn.y(), wn.xFlipped())
		case verticalLineSegment:
			ts.verticalConic(dConicOf(wt), wn.top(), wn.bottom(), wn.x(), wn.yFlipped())
		case lineSegment:
			ts.intersectConicLine(dConicOf(wt), dLineOf(wn))
		case quadSegment:
			ts.intersectConicQuad(dConicOf(wt), dQuadOf(wn))
		case conicSegment:
			ts.intersectConicConic(dConicOf(wt), dConicOf(wn))
		case cubicSegment:
			swap = true
			ts.intersectCubicConic(dCubicOf(wn), dConicOf(wt))
		}
	case cubicSegment:
		switch wn.segmentType() {
		case horizontalLineSegment:
			ts.horizontalCubic(dCubicOf(wt), wn.left(), wn.right(), wn.y(), wn.xFlipped())
		case verticalLineSegment:
			ts.verticalCubic(dCubicOf(wt), wn.top(), wn.bottom(), wn.x(), wn.yFlipped())
		case lineSegment:
			ts.intersectCubicLine(dCubicOf(wt), dLineOf(wn))
		case quadSegment:
			ts.intersectCubicQuad(dCubicOf(wt), dQuadOf(wn))
		case conicSegment:
			ts.intersectCubicConic(dCubicOf(wt), dConicOf(wn))
		case cubicSegment:
			ts.intersectCubicCubic(dCubicOf(wt), dCubicOf(wn))
		}
	}
	addIntersectResults(ts, wt, wn, swap, coincidence)
}

// addIntersectResults threads the computed intersection set into the two segments' data model: each entry adds a t to
// both segments, links the resulting pt-t records into a shared loop, and records coincident pairs. This is the
// per-point loop that runs at the tail of addIntersectTs.
func addIntersectResults(ts *intersections, wt, wn *intersectionHelper, swap bool, coincidence *opCoincidence) {
	swapIdx := b2i(swap)
	coinIndex := -1
	var coinPtT [2]*opPtT
	pts := ts.usedCount()
	for pt := 0; pt < pts; pt++ {
		// If t value is used to compute pt in addT, error may creep in and rect intersections may result in non-rects.
		// If the pt from the intersection is integral, pass it in; else recompute from t.
		iPt := ts.pt(pt).asPoint()
		var testTAt, nextTAt *opPtT
		if isIntegral(iPt) {
			testTAt = wt.segment.addTPt(ts.tVal(swapIdx, pt), iPt)
			nextTAt = wn.segment.addTPt(ts.tVal(1-swapIdx, pt), iPt)
		} else {
			testTAt = wt.segment.addT(ts.tVal(swapIdx, pt))
			nextTAt = wn.segment.addT(ts.tVal(1-swapIdx, pt))
		}
		if testTAt == nil || nextTAt == nil {
			// A malformed insert (a t outside [0,1] reaching the head or the tail) yields no pt-t to thread. Drop this
			// point, and any coincident run it was going to close, rather than dereferencing nil.
			coinIndex = -1
			continue
		}
		if !testTAt.containsPtT(nextTAt) {
			oppPrev := testTAt.oppPrev(nextTAt) // nil if the pair already shares a pt-t loop
			if oppPrev != nil {
				testTAt.span.mergeMatches(nextTAt.span)
				testTAt.addOpp(nextTAt, oppPrev)
			}
		}
		if !ts.isCoincident(pt) {
			continue
		}
		if coinIndex < 0 {
			coinPtT[0] = testTAt
			coinPtT[1] = nextTAt
			coinIndex = pt
			continue
		}
		if coinPtT[0].span == testTAt.span {
			coinIndex = -1
			continue
		}
		if coinPtT[1].span == nextTAt.span {
			coinIndex = -1 // coincidence span collapsed
			continue
		}
		if swap {
			coinPtT[0], coinPtT[1] = coinPtT[1], coinPtT[0]
			testTAt, nextTAt = nextTAt, testTAt
		}
		if coinPtT[0].span.deleted() {
			coinIndex = -1
			continue
		}
		if testTAt.span.deleted() {
			coinIndex = -1
			continue
		}
		coincidence.add(coinPtT[0], testTAt, coinPtT[1], nextTAt)
		coinIndex = -1
	}
}

// boundsIntersects reports whether rectangles a and b overlap, using a ULP-tolerant comparison on each edge so bounds
// that touch within floating-point rounding still count as intersecting.
func boundsIntersects(a, b pathOpsBounds) bool {
	return almostLessOrEqualUlps(float64(a.left), float64(b.right)) &&
		almostLessOrEqualUlps(float64(b.left), float64(a.right)) &&
		almostLessOrEqualUlps(float64(a.top), float64(b.bottom)) &&
		almostLessOrEqualUlps(float64(b.top), float64(a.bottom))
}

// isIntegral reports whether both coordinates are whole numbers.
func isIntegral(p geom.Point) bool {
	return float64(p.X) == math.Floor(float64(p.X)) && float64(p.Y) == math.Floor(float64(p.Y))
}

// dLineOf/dQuadOf/dConicOf/dCubicOf build the double-precision curve a cursor's segment describes.
func dLineOf(h *intersectionHelper) dLine {
	var l dLine
	l.set([2]geom.Point{h.segment.pts[0], h.segment.pts[1]})
	return l
}

func dQuadOf(h *intersectionHelper) dQuad {
	var q dQuad
	q.set([3]geom.Point{h.segment.pts[0], h.segment.pts[1], h.segment.pts[2]})
	return q
}

func dConicOf(h *intersectionHelper) dConic {
	var c dConic
	c.set([3]geom.Point{h.segment.pts[0], h.segment.pts[1], h.segment.pts[2]}, h.weight())
	return c
}

func dCubicOf(h *intersectionHelper) dCubic {
	var c dCubic
	c.set([4]geom.Point{h.segment.pts[0], h.segment.pts[1], h.segment.pts[2], h.segment.pts[3]})
	return c
}
