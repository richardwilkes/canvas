// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The tSect curve/curve intersection solver: the iterative binary-subdivision engine that finds the intersections of
// two curves by repeatedly splitting whichever curve's remaining span has the largest bounding box, trimming the halves
// that cannot intersect the opposite curve, detecting coincident runs, and finally gathering the surviving
// near-touching span ends. It sits on the span layer (tspan.go: tSpan/ tCoincident) and the tCurve wrapper (tcurve.go),
// and is the first end-to-end consumer of the curve/line intersection lanes.
//
// Allocation is plain Go allocation, with a per-sect free list of deleted spans kept deliberately (rather than left
// entirely to the GC) because the algorithm relies on deterministic span reuse. There is no debug validate()/dump
// machinery. Pathops computes in double precision throughout.
//
// intersect(quad, quad) (intersectQuadQuad) and the conic/cubic entry points (via the tConic/tCubic wrappers) all feed
// addIntersections, part of the engine backing Op/Simplify/Builder.

package pathops

import (
	"math"
	"sort"
)

// coincidentSpanCount is COINCIDENT_SPAN_COUNT: the number of continuous spans on both sects that makes the solver
// suspect coincidence.
const coincidentSpanCount = 9

// The zeroOneSet bit flags EndsEqual/BinarySearch use to track which curve endpoints already produced an intersection.
const (
	zeroS1Set = 1
	oneS1Set  = 2
	zeroS2Set = 4
	oneS2Set  = 8
)

// tSect is the binary-subdivision state for one curve of an intersecting pair — its list of active spans (head), its
// coincident and deleted (free) span lists, and the removed-end bookkeeping.
type tSect struct {
	curve         tCurve
	head          *tSpan
	coincident    *tSpan
	deleted       *tSpan // free list, chained through tSpan.next
	activeCount   int
	removedStartT bool
	removedEndT   bool
	hung          bool
}

// newTSect creates a tSect for c, allocating the first span covering the whole curve.
func newTSect(c tCurve) *tSect {
	s := &tSect{curve: c}
	s.resetRemovedEnds()
	s.head = s.addOne()
	s.head.init(c)
	return s
}

// pointLast returns the curve's final control point.
func (s *tSect) pointLast() dPoint { return s.curve.pointAt(s.curve.pointLast()) }

// resetRemovedEnds clears the removed-start/removed-end tracking flags.
func (s *tSect) resetRemovedEnds() {
	s.removedStartT = false
	s.removedEndT = false
}

// addOne returns a fresh active span, reusing one from the deleted free list if available.
func (s *tSect) addOne() *tSpan {
	var result *tSpan
	if s.deleted != nil {
		result = s.deleted
		s.deleted = result.next
	} else {
		result = newTSpan(s.curve)
	}
	result.reset()
	result.hasPerp = false
	result.deleted = false
	s.activeCount++
	return result
}

// addFollowing inserts a fresh span filling the gap after prior (or before the head, if prior is nil).
func (s *tSect) addFollowing(prior *tSpan) *tSpan {
	result := s.addOne()
	if prior != nil {
		result.startT = prior.endT
	} else {
		result.startT = 0
	}
	var next *tSpan
	if prior != nil {
		next = prior.next
	} else {
		next = s.head
	}
	if next != nil {
		result.endT = next.startT
	} else {
		result.endT = 1
	}
	result.prev = prior
	result.next = next
	if prior != nil {
		prior.next = result
	} else {
		s.head = result
	}
	if next != nil {
		next.prev = result
	}
	result.resetBounds(s.curve)
	return result
}

// addForPerp ensures a span covering t exists (splitting/creating as needed) so the perpendicular from span can be
// tracked, and cross-bounds the two spans.
func (s *tSect) addForPerp(span *tSpan, t float64) {
	if !span.hasOppT(t) {
		var priorSpan *tSpan
		opp := s.spanAtT(t, &priorSpan)
		if opp == nil {
			opp = s.addFollowing(priorSpan)
		}
		opp.addBounded(span)
		span.addBounded(opp)
	}
}

// addSplitAt splits span at t into a new span, re-bounding both halves.
func (s *tSect) addSplitAt(span *tSpan, t float64) *tSpan {
	result := s.addOne()
	result.splitAt(span, t)
	result.initBounds(s.curve)
	span.initBounds(s.curve)
	return result
}

// binarySearchCoin walks outward from tStart to find the T on this curve whose perpendicular lands on sect2's curve
// inside an existing span, converging the coincident-run limit.
func (s *tSect) binarySearchCoin(sect2 *tSect, tStart, tStep float64, resultT, oppT *float64, oppFirst **tSpan) bool {
	work := newTSpan(s.curve)
	work.startT = tStart
	work.endT = tStart
	result := tStart
	last := s.curve.ptAtT(tStart)
	var oppPt dPoint
	flip := false
	contained := false
	down := tStep < 0
	opp := sect2.curve
	for {
		tStep *= 0.5
		work.startT += tStep
		if flip {
			tStep = -tStep
			flip = false
		}
		work.initBounds(s.curve)
		if work.collapsed {
			return false
		}
		if last.approximatelyEqual(work.pointFirst()) {
			break
		}
		last = work.pointFirst()
		work.coinStart.setPerp(s.curve, work.startT, last, opp)
		if work.coinStart.isMatch() {
			oppTTest := work.coinStart.perpT()
			if sect2.head.contains(oppTTest) {
				*oppT = oppTTest
				oppPt = work.coinStart.perpPoint()
				contained = true
				if down && result <= work.startT || !down && result >= work.startT {
					*oppFirst = nil // signal caller to fail
					return false
				}
				result = work.startT
				continue
			}
		}
		tStep = -tStep
		flip = true
	}
	if !contained {
		return false
	}
	if last.approximatelyEqual(s.curve.pointAt(0)) {
		result = 0
	} else if last.approximatelyEqual(s.pointLast()) {
		result = 1
	}
	if oppPt.approximatelyEqual(opp.pointAt(0)) {
		*oppT = 0
	} else if oppPt.approximatelyEqual(sect2.pointLast()) {
		*oppT = 1
	}
	*resultT = result
	return true
}

// boundsMax returns the active span with the largest bounds (preferring non-collapsed).
func (s *tSect) boundsMax() *tSpan {
	test := s.head
	largest := s.head
	lCollapsed := largest.collapsed
	safetyNet := 10000
	for test = test.next; test != nil; test = test.next {
		safetyNet--
		if safetyNet == 0 {
			s.hung = true
			return nil
		}
		tCollapsed := test.collapsed
		if (lCollapsed && !tCollapsed) || (lCollapsed == tCollapsed && largest.boundsMax < test.boundsMax) {
			largest = test
			lCollapsed = test.collapsed
		}
	}
	return largest
}

// coincidentCheck computes perpendiculars and extracts the coincident sub-runs over each run of consecutive spans long
// enough to suspect coincidence.
func (s *tSect) coincidentCheck(sect2 *tSect) bool {
	first := s.head
	if first == nil {
		return false
	}
	var last, next *tSpan
	for first != nil {
		consecutive := s.countConsecutiveSpans(first, &last)
		next = last.next
		if consecutive >= coincidentSpanCount {
			s.computePerpendiculars(sect2, first, last)
			// check to see if a range of points are on the curve
			coinStart := first
			for {
				if !s.extractCoincident(sect2, coinStart, last, &coinStart) {
					return false
				}
				if coinStart == nil || last.deleted {
					break
				}
			}
			if s.head == nil || sect2.head == nil {
				break
			}
			if next == nil || next.deleted {
				break
			}
		}
		first = next
	}
	return true
}

// coincidentForce collapses both curves to a single forced coincident span over [start1s, start1e] when the iterative
// search will not otherwise converge.
func (s *tSect) coincidentForce(sect2 *tSect, start1s, start1e float64) {
	first := s.head
	last := s.tail()
	oppFirst := sect2.head
	oppLast := sect2.tail()
	if last == nil || oppLast == nil {
		return
	}
	deleteEmptySpans := s.updateBounded(first, last, oppFirst)
	if sect2.updateBounded(oppFirst, oppLast, first) {
		deleteEmptySpans = true
	}
	s.removeSpanRange(first, last)
	sect2.removeSpanRange(oppFirst, oppLast)
	first.startT = start1s
	first.endT = start1e
	first.resetBounds(s.curve)
	first.coinStart.setPerp(s.curve, start1s, s.curve.pointAt(0), sect2.curve)
	first.coinEnd.setPerp(s.curve, start1e, s.pointLast(), sect2.curve)
	oppMatched := first.coinStart.perpT() < first.coinEnd.perpT()
	oppStartT := 0.0
	if first.coinStart.perpT() != -1 {
		oppStartT = math.Max(0, first.coinStart.perpT())
	}
	oppEndT := 1.0
	if first.coinEnd.perpT() != -1 {
		oppEndT = math.Min(1, first.coinEnd.perpT())
	}
	if !oppMatched {
		oppStartT, oppEndT = oppEndT, oppStartT
	}
	oppFirst.startT = oppStartT
	oppFirst.endT = oppEndT
	oppFirst.resetBounds(sect2.curve)
	s.removeCoincident(first, false)
	sect2.removeCoincident(oppFirst, true)
	if deleteEmptySpans {
		s.deleteEmptySpans()
		sect2.deleteEmptySpans()
	}
}

// coincidentHasT reports whether t falls within any already-recorded coincident span.
func (s *tSect) coincidentHasT(t float64) bool {
	for test := s.coincident; test != nil; test = test.next {
		if between(test.startT, t, test.endT) {
			return true
		}
	}
	return false
}

// collapsed returns the number of active spans that have collapsed to a point.
func (s *tSect) collapsed() int {
	result := 0
	for test := s.head; test != nil; test = test.next {
		if test.collapsed {
			result++
		}
	}
	return result
}

// computePerpendiculars drops perpendiculars from each span's ends onto the opposite curve, propagating the shared end
// perpendicular between adjacent spans.
func (s *tSect) computePerpendiculars(sect2 *tSect, first, last *tSpan) {
	if last == nil {
		return
	}
	opp := sect2.curve
	work := first
	var prior *tSpan
	for {
		if !work.hasPerp && !work.collapsed {
			if prior != nil {
				work.coinStart = prior.coinEnd
			} else {
				work.coinStart.setPerp(s.curve, work.startT, work.pointFirst(), opp)
			}
			if work.coinStart.isMatch() {
				perpT := work.coinStart.perpT()
				if sect2.coincidentHasT(perpT) {
					work.coinStart.init()
				} else {
					sect2.addForPerp(work, perpT)
				}
			}
			work.coinEnd.setPerp(s.curve, work.endT, work.pointLast(), opp)
			if work.coinEnd.isMatch() {
				perpT := work.coinEnd.perpT()
				if sect2.coincidentHasT(perpT) {
					work.coinEnd.init()
				} else {
					sect2.addForPerp(work, perpT)
				}
			}
			work.hasPerp = true
		}
		if work == last {
			break
		}
		prior = work
		work = work.next
	}
}

// countConsecutiveSpans returns the length of the run of touching spans starting at first; lastPtr receives the run's
// final span.
func (s *tSect) countConsecutiveSpans(first *tSpan, lastPtr **tSpan) int {
	consecutive := 1
	last := first
	for {
		next := last.next
		if next == nil {
			break
		}
		if next.startT > last.endT {
			break
		}
		consecutive++
		last = next
	}
	*lastPtr = last
	return consecutive
}

// hasBounded reports whether any active span has span among its bounded (cross-linked) spans.
func (s *tSect) hasBounded(span *tSpan) bool {
	test := s.head
	if test == nil {
		return false
	}
	for test != nil {
		if test.findOppSpan(span) != nil {
			return true
		}
		test = test.next
	}
	return false
}

// deleteEmptySpans drops every span whose bounded list is empty.
func (s *tSect) deleteEmptySpans() bool {
	next := s.head
	safetyHatch := 1000
	for next != nil {
		test := next
		next = test.next
		if test.bounded == nil {
			if !s.removeSpan(test) {
				return false
			}
		}
		safetyHatch--
		if safetyHatch < 0 {
			return false
		}
	}
	return true
}

// extractCoincident reduces a coincident run to a single entry per sect, splitting at the run limits found via
// binarySearchCoin. result receives the span to resume from.
func (s *tSect) extractCoincident(sect2 *tSect, first, last *tSpan, result **tSpan) bool {
	first = s.findCoincidentRun(first, &last)
	if first == nil || last == nil {
		*result = nil
		return true
	}
	// march outwards to find limit of coincidence from here to previous and next spans
	startT := first.startT
	var oppStartT, oppEndT float64
	prev := first.prev
	oppFirst := first.findOppT(first.coinStart.perpT())
	oppMatched := first.coinStart.perpT() < first.coinEnd.perpT()
	var coinStart float64
	var cutFirst *tSpan
	if prev != nil && prev.endT == startT &&
		s.binarySearchCoin(sect2, startT, prev.startT-startT, &coinStart, &oppStartT, &oppFirst) &&
		prev.startT < coinStart && coinStart < startT {
		cutFirst = prev.oppT(oppStartT)
	}
	if cutFirst != nil {
		oppFirst = cutFirst
		first = s.addSplitAt(prev, coinStart)
		first.markCoincident()
		prev.coinEnd.markCoincident()
		if oppFirst.startT < oppStartT && oppStartT < oppFirst.endT {
			oppHalf := sect2.addSplitAt(oppFirst, oppStartT)
			if oppMatched {
				oppFirst.coinEnd.markCoincident()
				oppHalf.markCoincident()
				oppFirst = oppHalf
			} else {
				oppFirst.markCoincident()
				oppHalf.coinStart.markCoincident()
			}
		}
	} else if oppFirst == nil {
		return false
	}
	// The end of the coincident run is not searched for the way its start is just above: if the run continues past
	// last.endT into last.next, this simply takes the perpendicular at last.endT, so the run is truncated to the span
	// boundary. That asymmetry is real and fires on every coincident pair probed -- intersecting a cubic with an exact
	// subdivision of itself over [0.4, 0.8] reports the run ending at t=0.78125 (a subdivision boundary) rather than at
	// 0.8 -- but the truncation is benign: the leftover tail is resolved as an ordinary intersection, so the output
	// geometry is correct and the only cost is a redundant chop of the result at that boundary.
	//
	// The mirror of the prev/cutFirst branch above is not a drop-in fix. coincidentCheck captures next = last.next
	// before calling and uses it as its loop cursor, so splitting or consuming that span leaves the caller pointing at
	// deleted state; implementing the obvious symmetric search regressed pairs that are correct today, including ones
	// whose runs this truncates. Extending the run here needs the caller's cursor contract sorted out first, and a
	// failing case to justify it -- the truncation alone is not one.
	oppLast := last.findOppT(last.coinEnd.perpT())
	if !oppMatched {
		// oppStartT/oppEndT are not swapped alongside: both are recomputed from the merged run's perpendiculars below
		// before their next read.
		oppFirst, oppLast = oppLast, oppFirst
	}
	if oppFirst == nil {
		*result = nil
		return true
	}
	if oppLast == nil {
		*result = nil
		return true
	}
	// reduce coincident runs to single entries
	deleteEmptySpans := s.updateBounded(first, last, oppFirst)
	if sect2.updateBounded(oppFirst, oppLast, first) {
		deleteEmptySpans = true
	}
	s.removeSpanRange(first, last)
	sect2.removeSpanRange(oppFirst, oppLast)
	first.endT = last.endT
	first.resetBounds(s.curve)
	first.coinStart.setPerp(s.curve, first.startT, first.pointFirst(), sect2.curve)
	first.coinEnd.setPerp(s.curve, first.endT, first.pointLast(), sect2.curve)
	oppStartT = first.coinStart.perpT()
	oppEndT = first.coinEnd.perpT()
	if between(0, oppStartT, 1) && between(0, oppEndT, 1) {
		if !oppMatched {
			oppStartT, oppEndT = oppEndT, oppStartT
		}
		oppFirst.startT = oppStartT
		oppFirst.endT = oppEndT
		oppFirst.resetBounds(sect2.curve)
	}
	last = first.next
	if !s.removeCoincident(first, false) {
		return false
	}
	if !sect2.removeCoincident(oppFirst, true) {
		return false
	}
	if deleteEmptySpans {
		if !s.deleteEmptySpans() || !sect2.deleteEmptySpans() {
			*result = nil
			return false
		}
	}
	if last != nil && !last.deleted && s.head != nil && sect2.head != nil {
		*result = last
	} else {
		*result = nil
	}
	return true
}

// findCoincidentRun returns the first..last sub-range of fully coincident spans.
func (s *tSect) findCoincidentRun(first *tSpan, lastPtr **tSpan) *tSpan {
	work := first
	var lastCandidate *tSpan
	first = nil
	// find the first fully coincident span (a span whose start matches but whose end does not ends the search)
	for !work.coinStart.isMatch() || work.coinEnd.isMatch() {
		switch {
		case work.coinStart.isMatch():
			lastCandidate = work
			if first == nil {
				first = work
			}
		case first != nil && work.collapsed:
			*lastPtr = lastCandidate
			return first
		default:
			lastCandidate = nil
		}
		if work == *lastPtr {
			return first
		}
		work = work.next
		if work == nil {
			return nil
		}
	}
	if lastCandidate != nil {
		*lastPtr = lastCandidate
	}
	return first
}

// intersects classifies how span and oppSpan overlap, collapsing to a point when the hulls meet at a single end.
// oppResult receives the opposite span's classification (1 or 2).
func (s *tSect) intersects(span *tSpan, opp *tSect, oppSpan *tSpan, oppResult *int) int {
	var spanStart, oppStart bool
	hullResult := span.hullsIntersect(oppSpan, &spanStart, &oppStart)
	if hullResult >= 0 {
		if hullResult == 2 { // hulls have one point in common
			if span.bounded == nil || span.bounded.next == nil {
				if spanStart {
					span.endT = span.startT
				} else {
					span.startT = span.endT
				}
			} else {
				hullResult = 1
			}
			if oppSpan.bounded == nil || oppSpan.bounded.next == nil {
				if oppSpan.bounded != nil && oppSpan.bounded.bounded != span {
					return 0
				}
				if oppStart {
					oppSpan.endT = oppSpan.startT
				} else {
					oppSpan.startT = oppSpan.endT
				}
				*oppResult = 2
			} else {
				*oppResult = 1
			}
		} else {
			*oppResult = 1
		}
		return hullResult
	}
	if span.isLine && oppSpan.isLine {
		i := newIntersections()
		sects := s.linesIntersect(span, opp, oppSpan, i)
		if sects == 2 {
			*oppResult = 1
			return 1
		}
		if sects == 0 {
			return -1
		}
		s.removedEndCheck(span)
		spanT := i.tVal(0, 0)
		span.startT = spanT
		span.endT = spanT
		opp.removedEndCheck(oppSpan)
		oppT := i.tVal(1, 0)
		oppSpan.startT = oppT
		oppSpan.endT = oppT
		*oppResult = 2
		return 2
	}
	if span.isLinear || oppSpan.isLinear {
		*oppResult = b2i(span.linearsIntersect(oppSpan))
		return *oppResult
	}
	*oppResult = 1
	return 1
}

// isParallel reports whether thisLine runs parallel to a conic opp (the perpendiculars from both endpoints re-hit the
// endpoints). Non-conic curves are never parallel here.
func isParallel(thisLine dLine, opp tCurve) bool {
	// The non-conic early-out is a deliberate upstream decision, not an unfinished edit: the parallel test below is
	// only trusted for conics, and running it on quads/cubics was disabled because it broke too much to keep.
	//
	// Do not treat a green suite as license to remove it. Deleting this guard was measured here: the test then runs on
	// 3191 non-conic calls and answers true on 2382 of them, each one flipping linesIntersect from its closest-pair
	// search to an outright "coincident" (return 2) -- and the whole suite plus the oracle gates still pass. That says
	// the corpus does not discriminate the two behaviors, not that enabling it is safe. The oracle is frozen and cannot
	// adjudicate the change, so the guard stays until a failing case argues otherwise.
	if !opp.isConic() {
		return false
	}
	finds := 0
	var thisPerp dLine
	thisPerp.pts[0].x = thisLine.pts[1].x + (thisLine.pts[1].y - thisLine.pts[0].y)
	thisPerp.pts[0].y = thisLine.pts[1].y + (thisLine.pts[0].x - thisLine.pts[1].x)
	thisPerp.pts[1] = thisLine.pts[1]
	perpRayI := newIntersections()
	perpRayI.intersectRayTCurve(opp, thisPerp)
	for pIndex := 0; pIndex < perpRayI.usedCount(); pIndex++ {
		finds += b2i(perpRayI.pt(pIndex).approximatelyEqual(thisPerp.pts[1]))
	}
	thisPerp.pts[1].x = thisLine.pts[0].x + (thisLine.pts[1].y - thisLine.pts[0].y)
	thisPerp.pts[1].y = thisLine.pts[0].y + (thisLine.pts[0].x - thisLine.pts[1].x)
	thisPerp.pts[0] = thisLine.pts[0]
	perpRayI.intersectRayTCurve(opp, thisPerp)
	for pIndex := 0; pIndex < perpRayI.usedCount(); pIndex++ {
		finds += b2i(perpRayI.pt(pIndex).approximatelyEqual(thisPerp.pts[0]))
	}
	return finds >= 2
}

// linesIntersect is the tangent-line convergence lane for two near-linear spans. Returns 0 (no intersection), 1 (found
// one, recorded in i), or 2 (coincident lines).
func (s *tSect) linesIntersect(span *tSpan, opp *tSect, oppSpan *tSpan, i *intersections) int {
	thisRayI := newIntersections()
	oppRayI := newIntersections()
	thisLine := dLine{pts: [2]dPoint{span.pointFirst(), span.pointLast()}}
	oppLine := dLine{pts: [2]dPoint{oppSpan.pointFirst(), oppSpan.pointLast()}}
	loopCount := 0
	bestDistSq := math.MaxFloat64
	if thisRayI.intersectRayTCurve(opp.curve, thisLine) == 0 {
		return 0
	}
	if oppRayI.intersectRayTCurve(s.curve, oppLine) == 0 {
		return 0
	}
	// if the ends of each line intersect the opposite curve, the lines are coincident
	if thisRayI.usedCount() > 1 {
		ptMatches := 0
		for tIndex := 0; tIndex < thisRayI.usedCount(); tIndex++ {
			for lIndex := 0; lIndex < len(thisLine.pts); lIndex++ {
				ptMatches += b2i(thisRayI.pt(tIndex).approximatelyEqual(thisLine.pts[lIndex]))
			}
		}
		if ptMatches == 2 || isParallel(thisLine, opp.curve) {
			return 2
		}
	}
	if oppRayI.usedCount() > 1 {
		ptMatches := 0
		for oIndex := 0; oIndex < oppRayI.usedCount(); oIndex++ {
			for lIndex := 0; lIndex < len(oppLine.pts); lIndex++ {
				ptMatches += b2i(oppRayI.pt(oIndex).approximatelyEqual(oppLine.pts[lIndex]))
			}
		}
		if ptMatches == 2 || isParallel(oppLine, s.curve) {
			return 2
		}
	}
	for {
		// pick the closest pair of points
		closest := math.MaxFloat64
		closeIndex := 0
		oppCloseIndex := 0
		for index := 0; index < oppRayI.usedCount(); index++ {
			if !roughlyBetween(span.startT, oppRayI.tVal(0, index), span.endT) {
				continue
			}
			for oIndex := 0; oIndex < thisRayI.usedCount(); oIndex++ {
				if !roughlyBetween(oppSpan.startT, thisRayI.tVal(0, oIndex), oppSpan.endT) {
					continue
				}
				distSq := thisRayI.pt(index).distanceSquared(oppRayI.pt(oIndex))
				if closest > distSq {
					closest = distSq
					closeIndex = index
					oppCloseIndex = oIndex
				}
			}
		}
		if closest == math.MaxFloat64 {
			break
		}
		oppIPt := thisRayI.pt(oppCloseIndex)
		iPt := oppRayI.pt(closeIndex)
		if between(span.startT, oppRayI.tVal(0, closeIndex), span.endT) &&
			between(oppSpan.startT, thisRayI.tVal(0, oppCloseIndex), oppSpan.endT) &&
			oppIPt.approximatelyEqual(iPt) {
			i.merge(oppRayI, closeIndex, thisRayI, oppCloseIndex)
			return i.usedCount()
		}
		distSq := oppIPt.distanceSquared(iPt)
		loopCount++
		if bestDistSq < distSq || loopCount > 5 {
			return 0
		}
		bestDistSq = distSq
		oppStart := oppRayI.tVal(0, closeIndex)
		thisLine.pts[0] = s.curve.ptAtT(oppStart)
		thisLine.pts[1] = thisLine.pts[0].plusVector(s.curve.dxdyAtT(oppStart))
		if thisRayI.intersectRayTCurve(opp.curve, thisLine) == 0 {
			break
		}
		start := thisRayI.tVal(0, oppCloseIndex)
		oppLine.pts[0] = opp.curve.ptAtT(start)
		oppLine.pts[1] = oppLine.pts[0].plusVector(opp.curve.dxdyAtT(start))
		if oppRayI.intersectRayTCurve(s.curve, oppLine) == 0 {
			break
		}
	}
	// convergence may fail if the curves are nearly coincident
	oCoinS := newTCoincident()
	oCoinE := newTCoincident()
	oCoinS.setPerp(opp.curve, oppSpan.startT, oppSpan.pointFirst(), s.curve)
	oCoinE.setPerp(opp.curve, oppSpan.endT, oppSpan.pointLast(), s.curve)
	tStart := oCoinS.perpT()
	tEnd := oCoinE.perpT()
	doSwap := tStart > tEnd
	if doSwap {
		tStart, tEnd = tEnd, tStart
	}
	tStart = math.Max(tStart, span.startT)
	tEnd = math.Min(tEnd, span.endT)
	if tStart > tEnd {
		return 0
	}
	var perpS, perpE dVector
	switch {
	case tStart == span.startT:
		coinS := newTCoincident()
		coinS.setPerp(s.curve, span.startT, span.pointFirst(), opp.curve)
		perpS = span.pointFirst().sub(coinS.perpPoint())
	case doSwap:
		perpS = oCoinE.perpPoint().sub(oppSpan.pointLast())
	default:
		perpS = oCoinS.perpPoint().sub(oppSpan.pointFirst())
	}
	switch {
	case tEnd == span.endT:
		coinE := newTCoincident()
		coinE.setPerp(s.curve, span.endT, span.pointLast(), opp.curve)
		perpE = span.pointLast().sub(coinE.perpPoint())
	case doSwap:
		perpE = oCoinS.perpPoint().sub(oppSpan.pointFirst())
	default:
		perpE = oCoinE.perpPoint().sub(oppSpan.pointLast())
	}
	if perpS.dot(perpE) >= 0 {
		return 0
	}
	coinW := newTCoincident()
	workT := tStart
	tStep := tEnd - tStart
	var workPt dPoint
	for {
		tStep *= 0.5
		if preciselyZero(tStep) {
			return 0
		}
		workT += tStep
		workPt = s.curve.ptAtT(workT)
		coinW.setPerp(s.curve, workT, workPt, opp.curve)
		perpT := coinW.perpT()
		var skip bool
		if coinW.isMatch() {
			skip = !between(oppSpan.startT, perpT, oppSpan.endT)
		} else {
			skip = perpT < 0
		}
		if skip {
			continue
		}
		perpW := workPt.sub(coinW.perpPoint())
		if (perpS.dot(perpW) >= 0) == (tStep < 0) {
			tStep = -tStep
		}
		if workPt.approximatelyEqual(coinW.perpPoint()) {
			break
		}
	}
	oppTTest := coinW.perpT()
	if !opp.head.contains(oppTTest) {
		return 0
	}
	i.setMax(1)
	i.insert(workT, oppTTest, workPt)
	return 1
}

// markSpanGone pushes span onto the deleted free list.
func (s *tSect) markSpanGone(span *tSpan) bool {
	s.activeCount--
	if s.activeCount < 0 {
		return false
	}
	span.next = s.deleted
	s.deleted = span
	span.deleted = true
	return true
}

// mergeCoincidence joins adjacent coincident spans whose midpoint is also coincident.
func (s *tSect) mergeCoincidence(sect2 *tSect) {
	smallLimit := 0.0
	for {
		// find the smallest unprocessed span
		var smaller *tSpan
		for test := s.coincident; test != nil; test = test.next {
			if test.startT < smallLimit {
				continue
			}
			if smaller != nil && smaller.endT < test.startT {
				continue
			}
			smaller = test
		}
		if smaller == nil {
			return
		}
		smallLimit = smaller.endT
		// find next larger span
		var prior, larger, largerPrior *tSpan
		test := s.coincident
		for test != nil {
			if test.startT >= smaller.endT {
				if larger == nil || larger.startT >= test.startT {
					largerPrior = prior
					larger = test
				}
			}
			prior = test
			test = test.next
		}
		if larger == nil {
			continue
		}
		// check middle t value to see if it is coincident as well
		midT := (smaller.endT + larger.startT) / 2
		midPt := s.curve.ptAtT(midT)
		coin := newTCoincident()
		coin.setPerp(s.curve, midT, midPt, sect2.curve)
		if coin.isMatch() {
			smaller.endT = larger.endT
			smaller.coinEnd = larger.coinEnd
			if largerPrior != nil {
				largerPrior.next = larger.next
			} else {
				s.coincident = larger.next
			}
		}
	}
}

// recoverCollapsed re-inserts collapsed deleted spans into the active list in T order (they carry an intersection at a
// single point).
func (s *tSect) recoverCollapsed() {
	deleted := s.deleted
	for deleted != nil {
		delNext := deleted.next
		if deleted.collapsed {
			if s.head == nil || s.head.endT > deleted.startT {
				deleted.next = s.head
				s.head = deleted
			} else {
				prev := s.head
				for prev.next != nil && prev.next.endT <= deleted.startT {
					prev = prev.next
				}
				deleted.next = prev.next
				prev.next = deleted
			}
		}
		deleted = delNext
	}
}

// removeAllBut drops every bounded span of span except keep.
func (s *tSect) removeAllBut(keep, span *tSpan, opp *tSect) {
	testBounded := span.bounded
	for testBounded != nil {
		bounded := testBounded.bounded
		next := testBounded.next
		// may have been deleted when opp did 'remove all but'
		if bounded != keep && !bounded.deleted {
			span.removeBounded(bounded)
			if bounded.removeBounded(span) {
				opp.removeSpan(bounded)
			}
		}
		testBounded = next
	}
}

// removeByPerpendicular drops spans whose start/end perpendiculars point the same way (so the opposite curve does not
// cross between them).
func (s *tSect) removeByPerpendicular(opp *tSect) bool {
	test := s.head
	for test != nil {
		next := test.next
		if test.coinStart.perpT() >= 0 && test.coinEnd.perpT() >= 0 {
			startV := test.coinStart.perpPoint().sub(test.pointFirst())
			endV := test.coinEnd.perpPoint().sub(test.pointLast())
			if startV.dot(endV) > 0 {
				if !s.removeSpans(test, opp) {
					return false
				}
			}
		}
		test = next
	}
	return true
}

// removeCoincident unlinks span from the active list and moves it to the coincident list (if it starts a coincident
// run) or the deleted list.
func (s *tSect) removeCoincident(span *tSpan, isBetween bool) bool {
	if !s.unlinkSpan(span) {
		return false
	}
	if isBetween || between(0, span.coinStart.perpT(), 1) {
		s.activeCount--
		span.next = s.coincident
		s.coincident = span
	} else {
		s.markSpanGone(span)
	}
	return true
}

// removedEndCheck records that an endpoint T (0 or 1) got removed.
func (s *tSect) removedEndCheck(span *tSpan) {
	if span.startT == 0 {
		s.removedStartT = true
	}
	if span.endT == 1 {
		s.removedEndT = true
	}
}

// removeSpan unlinks span from the active list and pushes it onto the deleted free list.
func (s *tSect) removeSpan(span *tSpan) bool {
	s.removedEndCheck(span)
	if !s.unlinkSpan(span) {
		return false
	}
	return s.markSpanGone(span)
}

// removeSpanRange deletes every span strictly between first and last.
func (s *tSect) removeSpanRange(first, last *tSpan) {
	if first == last {
		return
	}
	fin := last.next
	next := first.next
	for span := next; span != nil && span != fin; span = next {
		next = span.next
		s.markSpanGone(span)
	}
	if fin != nil {
		fin.prev = first
	}
	first.next = fin
}

// removeSpans unbinds span from all its bounded spans, removing whichever become empty.
func (s *tSect) removeSpans(span *tSpan, opp *tSect) bool {
	bounded := span.bounded
	for bounded != nil {
		spanBounded := bounded.bounded
		next := bounded.next
		if span.removeBounded(spanBounded) { // shuffles last into position 0
			s.removeSpan(span)
		}
		if spanBounded.removeBounded(span) {
			opp.removeSpan(spanBounded)
		}
		if span.deleted && opp.hasBounded(span) {
			return false
		}
		bounded = next
	}
	return true
}

// spanAtT returns the span whose range contains t, with priorSpan set to the span before.
func (s *tSect) spanAtT(t float64, priorSpan **tSpan) *tSpan {
	test := s.head
	var prev *tSpan
	for test != nil && test.endT < t {
		prev = test
		test = test.next
	}
	*priorSpan = prev
	if test != nil && test.startT <= t {
		return test
	}
	return nil
}

// tail returns the active span with the largest endT.
func (s *tSect) tail() *tSpan {
	result := s.head
	next := s.head
	safetyNet := 100000
	for next != nil {
		next = next.next
		if next == nil {
			break
		}
		safetyNet--
		if safetyNet == 0 {
			return nil
		}
		if next.endT > result.endT {
			result = next
		}
	}
	return result
}

// trim re-classifies each of span's bounded spans after a split and prunes the pairs that no longer intersect.
func (s *tSect) trim(span *tSpan, opp *tSect) bool {
	if !span.initBounds(s.curve) {
		return false
	}
	testBounded := span.bounded
	for testBounded != nil {
		test := testBounded.bounded
		next := testBounded.next
		var oppSects int
		sects := s.intersects(span, opp, test, &oppSects)
		if sects >= 1 {
			if oppSects == 2 {
				test.initBounds(opp.curve)
				opp.removeAllBut(span, test, s)
			}
			if sects == 2 {
				span.initBounds(s.curve)
				s.removeAllBut(test, span, opp)
				return true
			}
		} else {
			if span.removeBounded(test) {
				s.removeSpan(span)
			}
			if test.removeBounded(span) {
				opp.removeSpan(test)
			}
		}
		testBounded = next
	}
	return true
}

// unlinkSpan splices span out of the active doubly-linked list.
func (s *tSect) unlinkSpan(span *tSpan) bool {
	prev := span.prev
	next := span.next
	if prev != nil {
		prev.next = next
		if next != nil {
			next.prev = prev
			if next.startT > next.endT {
				return false
			}
		}
	} else {
		s.head = next
		if next != nil {
			next.prev = nil
		}
	}
	return true
}

// updateBounded collapses first..last's bounded lists to a single bound on oppFirst.
func (s *tSect) updateBounded(first, last, oppFirst *tSpan) bool {
	fin := last.next
	deleteSpan := false
	test := first
	for {
		if test.removeAllBounded() {
			deleteSpan = true
		}
		test = test.next
		if test == fin || test == nil {
			break
		}
	}
	first.bounded = nil
	first.addBounded(oppFirst)
	return deleteSpan
}

// tSectEndsEqual records the endpoint coincidences of the two curves and returns the zeroOneSet bitfield describing
// which endpoints matched.
func tSectEndsEqual(sect1, sect2 *tSect, in *intersections) int {
	zeroOneSet := 0
	s1c0 := sect1.curve.pointAt(0)
	s2c0 := sect2.curve.pointAt(0)
	s1last := sect1.pointLast()
	s2last := sect2.pointLast()
	if s1c0.equals(s2c0) {
		zeroOneSet |= zeroS1Set | zeroS2Set
		in.insert(0, 0, s1c0)
	}
	if s1c0.equals(s2last) {
		zeroOneSet |= zeroS1Set | oneS2Set
		in.insert(0, 1, s1c0)
	}
	if s1last.equals(s2c0) {
		zeroOneSet |= oneS1Set | zeroS2Set
		in.insert(1, 0, s1last)
	}
	if s1last.equals(s2last) {
		zeroOneSet |= oneS1Set | oneS2Set
		in.insert(1, 1, s1last)
	}
	// check for zero
	if zeroOneSet&(zeroS1Set|zeroS2Set) == 0 && s1c0.approximatelyEqual(s2c0) {
		zeroOneSet |= zeroS1Set | zeroS2Set
		in.insertNear(0, 0, s1c0)
	}
	if zeroOneSet&(zeroS1Set|oneS2Set) == 0 && s1c0.approximatelyEqual(s2last) {
		zeroOneSet |= zeroS1Set | oneS2Set
		in.insertNear(0, 1, s1c0)
	}
	// check for one
	if zeroOneSet&(oneS1Set|zeroS2Set) == 0 && s1last.approximatelyEqual(s2c0) {
		zeroOneSet |= oneS1Set | zeroS2Set
		in.insertNear(1, 0, s1last)
	}
	if zeroOneSet&(oneS1Set|oneS2Set) == 0 && s1last.approximatelyEqual(s2last) {
		zeroOneSet |= oneS1Set | oneS2Set
		in.insertNear(1, 1, s1last)
	}
	return zeroOneSet
}

// closestRecord is a candidate intersection at a pair of near-touching span endpoints.
type closestRecord struct {
	c1Span   *tSpan
	c2Span   *tSpan
	c1StartT float64
	c1EndT   float64
	c2StartT float64
	c2EndT   float64
	closest  float64
	c1Index  int
	c2Index  int
}

// addIntersection records this record's matched endpoint pair into in.
func (r *closestRecord) addIntersection(in *intersections) {
	r1t := r.c1Span.startT
	if r.c1Index != 0 {
		r1t = r.c1Span.endT
	}
	r2t := r.c2Span.startT
	if r.c2Index != 0 {
		r2t = r.c2Span.endT
	}
	in.insert(r1t, r2t, r.c1Span.part.pointAt(r.c1Index))
}

// findEnd keeps the closest matching endpoint pair between span1 and span2's given endpoints.
func (r *closestRecord) findEnd(span1, span2 *tSpan, c1Index, c2Index int) {
	c1 := span1.part
	c2 := span2.part
	if !c1.pointAt(c1Index).approximatelyEqual(c2.pointAt(c2Index)) {
		return
	}
	dist := c1.pointAt(c1Index).distanceSquared(c2.pointAt(c2Index))
	if r.closest < dist {
		return
	}
	r.c1Span = span1
	r.c2Span = span2
	r.c1StartT = span1.startT
	r.c1EndT = span1.endT
	r.c2StartT = span2.startT
	r.c2EndT = span2.endT
	r.c1Index = c1Index
	r.c2Index = c2Index
	r.closest = dist
}

// matesWith reports whether r and mate refer to the same or adjacent spans, meaning they should merge.
func (r *closestRecord) matesWith(mate *closestRecord) bool {
	return r.c1Span == mate.c1Span || r.c1Span.endT == mate.c1Span.startT ||
		r.c1Span.startT == mate.c1Span.endT ||
		r.c2Span == mate.c2Span ||
		r.c2Span.endT == mate.c2Span.startT ||
		r.c2Span.startT == mate.c2Span.endT
}

// merge overwrites r's fields with mate's.
func (r *closestRecord) merge(mate *closestRecord) {
	r.c1Span = mate.c1Span
	r.c2Span = mate.c2Span
	r.closest = mate.closest
	r.c1Index = mate.c1Index
	r.c2Index = mate.c2Index
}

// reset marks r as having no candidate match yet.
func (r *closestRecord) reset() {
	r.closest = math.MaxFloat32
}

// update widens r's t ranges to also cover mate's.
func (r *closestRecord) update(mate *closestRecord) {
	r.c1StartT = math.Min(r.c1StartT, mate.c1StartT)
	r.c1EndT = math.Max(r.c1EndT, mate.c1EndT)
	r.c2StartT = math.Min(r.c2StartT, mate.c2StartT)
	r.c2EndT = math.Max(r.c2EndT, mate.c2EndT)
}

// closestSect is the set of distinct closest-endpoint records gathered at the end of the search. The slot at index used
// is always the working (not-yet-committed) record.
type closestSect struct {
	closest []closestRecord
	used    int
}

// newClosestSect creates a closestSect with one blank working record.
func newClosestSect() *closestSect {
	s := &closestSect{}
	var blank closestRecord
	blank.reset()
	s.closest = append(s.closest, blank)
	return s
}

// find scores span1/span2's four endpoint pairs, merging into an existing record if they mate, else keeping a new
// record. Returns true if a new distinct record was added.
func (s *closestSect) find(span1, span2 *tSpan) bool {
	rec := &s.closest[s.used]
	rec.findEnd(span1, span2, 0, 0)
	rec.findEnd(span1, span2, 0, span2.part.pointLast())
	rec.findEnd(span1, span2, span1.part.pointLast(), 0)
	rec.findEnd(span1, span2, span1.part.pointLast(), span2.part.pointLast())
	if rec.closest == math.MaxFloat32 {
		return false
	}
	for index := 0; index < s.used; index++ {
		test := &s.closest[index]
		if test.matesWith(rec) {
			if test.closest > rec.closest {
				test.merge(rec)
			}
			test.update(rec)
			rec.reset()
			return false
		}
	}
	s.used++
	var blank closestRecord
	blank.reset()
	s.closest = append(s.closest, blank)
	return true
}

// finish emits the gathered records into in, in ascending closeness order.
func (s *closestSect) finish(in *intersections) {
	ptrs := make([]*closestRecord, 0, s.used)
	for index := 0; index < s.used; index++ {
		ptrs = append(ptrs, &s.closest[index])
	}
	sort.SliceStable(ptrs, func(i, j int) bool { return ptrs[i].closest < ptrs[j].closest })
	for _, test := range ptrs {
		test.addIntersection(in)
	}
}

// tSectBinarySearch is the top-level solver driving two sects to their intersection set.
func tSectBinarySearch(sect1, sect2 *tSect, in *intersections) {
	in.reset()
	in.setMax(sect1.curve.maxIntersections() + 4) // give extra for slop
	span1 := sect1.head
	span2 := sect2.head
	var oppSect int
	sect := sect1.intersects(span1, sect2, span2, &oppSect)
	if sect == 0 {
		return
	}
	if sect == 2 && oppSect == 2 {
		tSectEndsEqual(sect1, sect2, in)
		return
	}
	span1.addBounded(span2)
	span2.addBounded(span1)
	const maxCoinLoopCount = 8
	coinLoopCount := maxCoinLoopCount
	var start1s, start1e float64
	for {
		// find the largest bounds
		largest1 := sect1.boundsMax()
		if largest1 == nil {
			if sect1.hung {
				return
			}
			break
		}
		largest2 := sect2.boundsMax()
		// split it
		if largest2 == nil || (largest1.boundsMax > largest2.boundsMax ||
			(!largest1.collapsed && largest2.collapsed)) {
			if sect2.hung {
				return
			}
			if largest1.collapsed {
				break
			}
			sect1.resetRemovedEnds()
			sect2.resetRemovedEnds()
			// trim parts that don't intersect the opposite
			half1 := sect1.addOne()
			if !half1.split(largest1) {
				break
			}
			if !sect1.trim(largest1, sect2) {
				return
			}
			if !sect1.trim(half1, sect2) {
				return
			}
		} else {
			if largest2.collapsed {
				break
			}
			sect1.resetRemovedEnds()
			sect2.resetRemovedEnds()
			// trim parts that don't intersect the opposite
			half2 := sect2.addOne()
			if !half2.split(largest2) {
				break
			}
			if !sect2.trim(largest2, sect1) {
				return
			}
			if !sect2.trim(half2, sect1) {
				return
			}
		}
		// if there are 9 or more continuous spans on both sects, suspect coincidence
		if sect1.activeCount >= coincidentSpanCount && sect2.activeCount >= coincidentSpanCount {
			if coinLoopCount == maxCoinLoopCount {
				start1s = sect1.head.startT
				start1e = sect1.tail().endT
			}
			if !sect1.coincidentCheck(sect2) {
				return
			}
			coinLoopCount--
			if coinLoopCount == 0 && sect1.head != nil && sect2.head != nil {
				// All known working cases resolve in two tries. cubicConicTests[0] gets stuck in a loop; the force
				// converges it.
				sect1.coincidentForce(sect2, start1s, start1e)
			}
		}
		if sect1.activeCount >= coincidentSpanCount && sect2.activeCount >= coincidentSpanCount {
			if sect1.head == nil {
				return
			}
			sect1.computePerpendiculars(sect2, sect1.head, sect1.tail())
			if sect2.head == nil {
				return
			}
			sect2.computePerpendiculars(sect1, sect2.head, sect2.tail())
			if !sect1.removeByPerpendicular(sect2) {
				return
			}
			if sect1.collapsed() > sect1.curve.maxIntersections() {
				break
			}
		}
		if sect1.head == nil || sect2.head == nil {
			break
		}
	}
	coincident := sect1.coincident
	if coincident != nil {
		// if there is more than one coincident span, check loosely to see if they should be joined
		if coincident.next != nil {
			sect1.mergeCoincidence(sect2)
			coincident = sect1.coincident
		}
		for coincident != nil {
			if !coincident.coinStart.isMatch() {
				coincident = coincident.next
				continue
			}
			if !coincident.coinEnd.isMatch() {
				coincident = coincident.next
				continue
			}
			perpT := coincident.coinStart.perpT()
			if perpT < 0 {
				return
			}
			index := in.insertCoincident(coincident.startT, perpT, coincident.pointFirst())
			if in.insertCoincident(coincident.endT, coincident.coinEnd.perpT(),
				coincident.pointLast()) < 0 && index >= 0 {
				in.clearCoincidence(index)
			}
			coincident = coincident.next
		}
	}
	zeroOneSet := tSectEndsEqual(sect1, sect2, in)
	// if the final iteration contains an end (0 or 1), intersect its perpendicular with the opposite curve
	if sect1.removedStartT && zeroOneSet&zeroS1Set == 0 {
		perp := newTCoincident()
		perp.setPerp(sect1.curve, 0, sect1.curve.pointAt(0), sect2.curve)
		if perp.isMatch() {
			in.insert(0, perp.perpT(), perp.perpPoint())
		}
	}
	if sect1.removedEndT && zeroOneSet&oneS1Set == 0 {
		perp := newTCoincident()
		perp.setPerp(sect1.curve, 1, sect1.pointLast(), sect2.curve)
		if perp.isMatch() {
			in.insert(1, perp.perpT(), perp.perpPoint())
		}
	}
	if sect2.removedStartT && zeroOneSet&zeroS2Set == 0 {
		perp := newTCoincident()
		perp.setPerp(sect2.curve, 0, sect2.curve.pointAt(0), sect1.curve)
		if perp.isMatch() {
			in.insert(perp.perpT(), 0, perp.perpPoint())
		}
	}
	if sect2.removedEndT && zeroOneSet&oneS2Set == 0 {
		perp := newTCoincident()
		perp.setPerp(sect2.curve, 1, sect2.pointLast(), sect1.curve)
		if perp.isMatch() {
			in.insert(perp.perpT(), 1, perp.perpPoint())
		}
	}
	if sect1.head == nil || sect2.head == nil {
		return
	}
	sect1.recoverCollapsed()
	sect2.recoverCollapsed()
	result1 := sect1.head
	// check heads and tails for zero and ones and insert them if we haven't already done so
	head1 := result1
	if zeroOneSet&zeroS1Set == 0 && approximatelyLessThanZero(head1.startT) {
		start1 := sect1.curve.pointAt(0)
		if head1.isBounded() {
			t := head1.closestBoundedT(start1)
			if sect2.curve.ptAtT(t).approximatelyEqual(start1) {
				in.insert(0, t, start1)
			}
		}
	}
	head2 := sect2.head
	if zeroOneSet&zeroS2Set == 0 && approximatelyLessThanZero(head2.startT) {
		start2 := sect2.curve.pointAt(0)
		if head2.isBounded() {
			t := head2.closestBoundedT(start2)
			if sect1.curve.ptAtT(t).approximatelyEqual(start2) {
				in.insert(t, 0, start2)
			}
		}
	}
	if zeroOneSet&oneS1Set == 0 {
		tail1 := sect1.tail()
		if tail1 == nil {
			return
		}
		if approximatelyGreaterThanOne(tail1.endT) {
			end1 := sect1.pointLast()
			if tail1.isBounded() {
				t := tail1.closestBoundedT(end1)
				if sect2.curve.ptAtT(t).approximatelyEqual(end1) {
					in.insert(1, t, end1)
				}
			}
		}
	}
	if zeroOneSet&oneS2Set == 0 {
		tail2 := sect2.tail()
		if tail2 == nil {
			return
		}
		if approximatelyGreaterThanOne(tail2.endT) {
			end2 := sect2.pointLast()
			if tail2.isBounded() {
				t := tail2.closestBoundedT(end2)
				if sect1.curve.ptAtT(t).approximatelyEqual(end2) {
					in.insert(t, 1, end2)
				}
			}
		}
	}
	closest := newClosestSect()
	for result1 != nil {
		for result1 != nil && result1.coinStart.isMatch() && result1.coinEnd.isMatch() {
			result1 = result1.next
		}
		if result1 == nil {
			break
		}
		for result2 := sect2.head; result2 != nil; result2 = result2.next {
			closest.find(result1, result2)
		}
		result1 = result1.next
	}
	closest.finish(in)
	// if there is more than one intersection and it isn't already coincident, check
	last := in.usedCount() - 1
	for index := 0; index < last; {
		if in.isCoincident(index) && in.isCoincident(index+1) {
			index++
			continue
		}
		midT := (in.tVal(0, index) + in.tVal(0, index+1)) / 2
		midPt := sect1.curve.ptAtT(midT)
		// intersect perpendicular with opposite curve
		perp := newTCoincident()
		perp.setPerp(sect1.curve, midT, midPt, sect2.curve)
		if !perp.isMatch() {
			index++
			continue
		}
		switch {
		case in.isCoincident(index):
			in.removeOne(index)
			last--
		case in.isCoincident(index + 1):
			in.removeOne(index + 1)
			last--
		default:
			in.setCoincident(index)
			index++
		}
		in.setCoincident(index)
	}
}

// intersectQuadQuad is the quad/quad entry point into the tSect solver.
func (in *intersections) intersectQuadQuad(q1, q2 dQuad) int {
	quad1 := &tQuad{quad: q1}
	quad2 := &tQuad{quad: q2}
	sect1 := newTSect(quad1)
	sect2 := newTSect(quad2)
	tSectBinarySearch(sect1, sect2, in)
	return in.usedCount()
}

// intersectConicQuad is the conic/quad entry point into the tSect solver.
func (in *intersections) intersectConicQuad(c dConic, q dQuad) int {
	conic := &tConic{conic: c}
	quad := &tQuad{quad: q}
	sect1 := newTSect(conic)
	sect2 := newTSect(quad)
	tSectBinarySearch(sect1, sect2, in)
	return in.usedCount()
}

// intersectConicConic is the conic/conic entry point into the tSect solver.
func (in *intersections) intersectConicConic(c1, c2 dConic) int {
	conic1 := &tConic{conic: c1}
	conic2 := &tConic{conic: c2}
	sect1 := newTSect(conic1)
	sect2 := newTSect(conic2)
	tSectBinarySearch(sect1, sect2, in)
	return in.usedCount()
}

// intersectCubicQuad is the cubic/quad entry point into the tSect solver.
func (in *intersections) intersectCubicQuad(c dCubic, q dQuad) int {
	cubic := &tCubic{cubic: c}
	quad := &tQuad{quad: q}
	sect1 := newTSect(cubic)
	sect2 := newTSect(quad)
	tSectBinarySearch(sect1, sect2, in)
	return in.usedCount()
}

// intersectCubicConic is the cubic/conic entry point into the tSect solver.
func (in *intersections) intersectCubicConic(cu dCubic, co dConic) int {
	cubic := &tCubic{cubic: cu}
	conic := &tConic{conic: co}
	sect1 := newTSect(cubic)
	sect2 := newTSect(conic)
	tSectBinarySearch(sect1, sect2, in)
	return in.usedCount()
}

// intersectCubicCubic is the cubic/cubic entry point into the tSect solver.
func (in *intersections) intersectCubicCubic(c1, c2 dCubic) int {
	cubic1 := &tCubic{cubic: c1}
	cubic2 := &tCubic{cubic: c2}
	sect1 := newTSect(cubic1)
	sect2 := newTSect(cubic2)
	tSectBinarySearch(sect1, sect2, in)
	return in.usedCount()
}
