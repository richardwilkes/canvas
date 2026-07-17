// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-segment coincidence-maintenance phase: the trio handleCoincidence (common.go) interleaves with the
// coincidence resolution machinery — moveMultiples (merge multi-intersection spans onto the segments that missed them),
// moveNearby (collapse tiny gaps so nearby t values and points hang off one span), and missingCoincidence (find
// coincident runs that pairwise intersection missed) — plus their helpers spansNearby/testForCoincidence and the
// clearAll/clearOne/clearVisited span cleanup they call. Every t value is a float64, matching the precision the
// boolean-op math needs throughout this package. The walking phase (activeOp/findNext*) arrives with the op/simplify
// drivers.

package pathops

import "math"

// clearAll zeros every span's winding, marks them done, and drops the segment from the coincidence tracker.
func (s *opSegment) clearAll() {
	span := &s.head
	for {
		s.clearOne(span)
		next := span.next.upCastable()
		if next == nil {
			break
		}
		span = next
	}
	s.globalState().coincidence.releaseSegment(s)
}

// clearOne zeros one span's winding and marks it done.
func (s *opSegment) clearOne(span *opSpan) {
	span.setWindValue(0)
	span.setOppValue(0)
	s.markDone(span)
}

// clearVisited resets the visited flag on every opposite segment reachable from the given span's pt-t loops.
func clearVisited(span *opSpanBase) {
	for {
		stopPtT := &span.ptT
		for ptT := stopPtT.next; ptT != stopPtT; ptT = ptT.next {
			ptT.segment().resetVisited()
		}
		if span.final() {
			return
		}
		span = span.upCast().next
	}
}

// missingCoincidence looks for pairs of undetected coincident curves. Even though pairs of curves correctly detect
// coincident runs, a run may be missed if the coincidence is a product of multiple intersections. Assumes the segments
// going in have their visited flag clear.
func (s *opSegment) missingCoincidence() bool {
	if s.done() {
		return false
	}
	var prior *opSpan
	spanBase := &s.head.opSpanBase
	result := false
	safetyNet := 1000
	for {
		spanStopPtT := &spanBase.ptT
		for ptT := spanStopPtT.next; ptT != spanStopPtT; ptT = ptT.next {
			safetyNet--
			if safetyNet == 0 {
				return false
			}
			if ptT.deleted {
				continue
			}
			opp := ptT.span.segment
			if opp.done() {
				continue
			}
			// when opp is encountered the 1st time, continue; on 2nd encounter, look for coincidence
			if !opp.visited() {
				continue
			}
			if spanBase == &s.head.opSpanBase {
				continue
			}
			if ptT.segment() == s {
				continue
			}
			span := spanBase.upCastable()
			// Assumption: an opposite segment already recorded as coincident here needs no further coincidence
			// detection. Unproven, and inherited as a hedge rather than a known break -- but it is the documented
			// suspect if a span ever turns up with coincidence that this walk failed to find.
			if span != nil && span.containsCoincidenceSeg(opp) {
				continue
			}
			if spanBase.containsCoinEndSegment(opp) {
				continue
			}
			// find prior span containing opp segment
			var priorPtT *opPtT
			var priorOpp *opSegment
			priorTest := spanBase.prev
			for priorOpp == nil && priorTest != nil {
				priorStopPtT := &priorTest.ptT
				priorPtT = priorStopPtT.next
				for priorPtT != priorStopPtT {
					if !priorPtT.deleted && priorPtT.span.segment == opp {
						prior = priorTest
						priorOpp = opp
						break
					}
					priorPtT = priorPtT.next
				}
				priorTest = priorTest.prev
			}
			if priorOpp == nil {
				continue
			}
			if priorPtT == ptT {
				continue
			}
			oppStart := prior.ptTPtr()
			oppEnd := spanBase.ptTPtr()
			startPtT, endPtT := priorPtT, ptT
			if startPtT.t > endPtT.t {
				startPtT, endPtT = endPtT, startPtT
				oppStart, oppEnd = oppEnd, oppStart
			}
			coincidences := s.globalState().coincidence
			rootPriorPtT := startPtT.span.ptTPtr()
			rootPtT := endPtT.span.ptTPtr()
			rootOppStart := oppStart.span.ptTPtr()
			rootOppEnd := oppEnd.span.ptTPtr()
			if !coincidences.containsPtTs(rootPriorPtT, rootPtT, rootOppStart, rootOppEnd) {
				if s.testForCoincidence(rootPriorPtT, rootPtT, &prior.opSpanBase, spanBase, opp) {
					// mark coincidence
					if !coincidences.extend(rootPriorPtT, rootPtT, rootOppStart, rootOppEnd) {
						coincidences.add(rootPriorPtT, rootPtT, rootOppStart, rootOppEnd)
					}
					result = true
				}
			}
		}
		if spanBase.final() {
			break
		}
		spanBase = spanBase.upCast().next
	}
	clearVisited(&s.head.opSpanBase)
	return result
}

// moveMultiples merges other segments' spans, where a span has more than one intersection, so they all carry the same
// intersection count.
func (s *opSegment) moveMultiples() bool {
	for test := &s.head.opSpanBase; ; {
		if !s.moveMultiplesForSpan(test) {
			return false
		}
		if test.final() {
			return true
		}
		test = test.upCast().next
	}
}

// moveMultiplesForSpan runs moveMultiples's per-span body for test, merging a matching opposite span if one is found.
// It returns false only on the safety-hatch trip.
func (s *opSegment) moveMultiplesForSpan(test *opSpanBase) bool {
	addCount := test.spanAddsCount()
	if addCount <= 1 {
		return true
	}
	startPtT := &test.ptT
	testPtT := startPtT
	safetyHatch := 1000
	for {
		safetyHatch--
		if safetyHatch == 0 {
			return false
		}
		oppSpan := testPtT.span
		if oppSpan.spanAddsCount() != addCount && !oppSpan.deleted() {
			oppSegment := oppSpan.segment
			if oppSegment != s {
				// find range of spans to consider merging
				oppFirst := oppSpan
				oppPrev := oppSpan
				for {
					p := oppPrev.prev
					if p == nil {
						break
					}
					oppPrev = &p.opSpanBase
					if !roughlyEqual(oppPrev.t(), oppSpan.t()) {
						break
					}
					if oppPrev.spanAddsCount() == addCount || oppPrev.deleted() {
						continue
					}
					oppFirst = oppPrev
				}
				oppLast := oppSpan
				oppNext := oppSpan
				for !oppNext.final() {

					oppNext = oppNext.upCast().next
					if !roughlyEqual(oppNext.t(), oppSpan.t()) {
						break
					}
					if oppNext.spanAddsCount() == addCount || oppNext.deleted() {
						continue
					}
					oppLast = oppNext
				}
				if oppFirst != oppLast {
					for oppTest := oppFirst; ; {
						if oppTest != oppSpan && s.moveMultiplesMerge(oppTest, oppSpan, startPtT) {
							return true // foundMatch -> checkNextSpan: done with this test span
						}
						if oppTest == oppLast {
							break
						}
						oppTest = oppTest.upCast().next
					}
				}
			}
		}
		testPtT = testPtT.next
		if testPtT == startPtT {
			return true
		}
	}
}

// moveMultiplesMerge checks whether the candidate span oppTest holds a pt-t whose segment is in startPtT's loop (but is
// not this segment); if so it merges oppTest into oppSpan and reports the match, otherwise it reports no match.
func (s *opSegment) moveMultiplesMerge(oppTest, oppSpan *opSpanBase, startPtT *opPtT) bool {
	oppStartPtT := &oppTest.ptT
	// Only the first pt-t after the head is ever examined: every outcome for it returns, so what reads as a walk of
	// the loop never advances past oppStartPtT.next (kept as an if to make that explicit).
	if oppPtT := oppStartPtT.next; oppPtT != oppStartPtT {
		oppPtTSegment := oppPtT.segment()
		if oppPtTSegment == s {
			return false
		}
		matched := false
		for matchPtT := startPtT; ; {
			if matchPtT.segment() == oppPtTSegment {
				matched = true
				break
			}
			matchPtT = matchPtT.next
			if matchPtT == startPtT {
				break
			}
		}
		if !matched {
			return false
		}
		// merge oppTest and oppSpan
		oppTest.mergeMatches(oppSpan)
		oppTest.addOpp(oppSpan)
		return true
	}
	return false
}

// spansNearby checks whether adjacent spans have points close by. It reports whether a near pairing was found (found)
// alongside an ok flag that is false only when the escape hatch tripped.
func (s *opSegment) spansNearby(refSpan, checkSpan *opSpanBase) (found, ok bool) {
	refHead := &refSpan.ptT
	checkHead := &checkSpan.ptT
	// if the first pt pair from adjacent spans are far apart, assume that all are far enough apart
	if !dPointWayRoughlyEqual(refHead.pt, checkHead.pt) {
		return false, true
	}
	// check only unique points
	distSqBest := float32(math.MaxFloat32) // largest representable float32, as a "no best yet" sentinel
	var refBest, checkBest *opPtT
	ref := refHead
	for {
		if !ref.deleted {
			wrapped := false
			for ref.ptAlreadySeen(refHead) {
				ref = ref.next
				if ref == refHead {
					wrapped = true
					break
				}
			}
			if wrapped {
				break // doneCheckingDistance
			}
			if !s.spansNearbyCheck(ref, checkHead, &distSqBest, &refBest, &checkBest) {
				return false, false
			}
		}
		ref = ref.next
		if ref == refHead {
			break
		}
	}
	found = checkBest != nil && refBest.segment().match(refBest, checkBest.segment(), checkBest.t, checkBest.pt)
	return found, true
}

// spansNearbyCheck runs spansNearby's inner check loop for one ref pt-t: it walks the checkHead loop, updating the
// running best (distSqBest/refBest/checkBest) with the closest non-disjoint pairing. It returns false when the escape
// hatch trips (spansNearby then aborts).
func (s *opSegment) spansNearbyCheck(ref, checkHead *opPtT, distSqBest *float32, refBest, checkBest **opPtT) bool {
	check := checkHead
	refSeg := ref.segment()
	escapeHatch := 100 // defend against infinite loops
	for {
		if !check.deleted {
			wrapped := false
			for check.ptAlreadySeen(checkHead) {
				check = check.next
				if check == checkHead {
					wrapped = true
					break
				}
			}
			if wrapped {
				return true // nextRef
			}
			distSq := ref.pt.DistanceToSqd(check.pt)
			if *distSqBest > distSq &&
				(refSeg != check.segment() || !refSeg.ptsDisjoint(ref.t, ref.pt, check.t, check.pt)) {
				*distSqBest = distSq
				*refBest = ref
				*checkBest = check
			}
			escapeHatch--
			if escapeHatch <= 0 {
				return false
			}
		}
		check = check.next
		if check == checkHead {
			return true
		}
	}
}

// moveNearby moves nearby t values and points so they all hang off the same span. Alignment happens later.
func (s *opSegment) moveNearby() bool {
	// release undeleted spans pointing to this seg that are linked to the primary span
	spanBase := &s.head.opSpanBase
	escapeHatch := 9999 // the largest count for a regular test is 50; for a fuzzer, 500
	for {
		headPtT := &spanBase.ptT
		for ptT := headPtT.next; ptT != headPtT; ptT = ptT.next {
			escapeHatch--
			if escapeHatch == 0 {
				return false
			}
			test := ptT.span
			if ptT.segment() == s && !ptT.deleted && test != spanBase && &test.ptT == ptT {
				if test.final() {
					if spanBase == &s.head.opSpanBase {
						s.clearAll()
						return true
					}
					spanBase.upCast().release(ptT)
				} else if test.prev != nil {
					test.upCast().release(headPtT)
				}
				break
			}
		}
		spanBase = spanBase.upCast().next
		if spanBase.final() {
			break
		}
	}
	// This loop looks for adjacent spans which are near by
	spanBase = &s.head.opSpanBase
	for {
		test := spanBase.upCast().next
		found, ok := s.spansNearby(spanBase, test)
		if !ok {
			return false
		}
		if found {
			if test.final() {
				if spanBase.prev != nil {
					test.merge(spanBase.upCast())
				} else {
					s.clearAll()
					return true
				}
			} else {
				spanBase.merge(test.upCast())
			}
		}
		spanBase = test
		if spanBase.final() {
			break
		}
	}
	return true
}

// testForCoincidence probes whether the range [prior, spanBase] on this segment is coincident with the matching range
// on opp, by projecting the perpendicular through the mid-point of this segment's sub-curve and seeing whether it meets
// opp at (nearly) the same point.
func (s *opSegment) testForCoincidence(priorPtT, ptT *opPtT, prior, spanBase *opSpanBase, opp *opSegment) bool {
	// average t, find mid pt
	midT := (prior.t() + spanBase.t()) / 2
	midPt := s.ptAtT(midT)
	coincident := true
	// if the mid pt is not near either end pt, project perpendicular through opp seg
	if !dPointsApproximatelyEqual(priorPtT.pt, midPt) && !dPointsApproximatelyEqual(ptT.pt, midPt) {
		if priorPtT.span == ptT.span {
			return false
		}
		coincident = false
		i := newIntersections()
		var curvePart dCurve
		s.subDivide(prior, spanBase, &curvePart)
		dxdy := curveDDSlopeAtT(s.verb, curvePart, 0.5)
		partMidPt := curveDDPointAtT(s.verb, curvePart, 0.5)
		ray := dLine{pts: [2]dPoint{
			{x: float64(midPt.X), y: float64(midPt.Y)},
			{x: partMidPt.x + dxdy.y, y: partMidPt.y - dxdy.x},
		}}
		var oppPart dCurve
		opp.subDivide(priorPtT.span, ptT.span, &oppPart)
		curveDIntersectRay(opp.verb, oppPart, ray, i)
		// measure distance and see if it's small enough to denote coincidence
		var midD dPoint
		midD.set(midPt)
		for index := 0; index < i.usedCount(); index++ {
			if !between(0, i.tVal(0, index), 1) {
				continue
			}
			oppPt := i.pt(index)
			if oppPt.approximatelyDEqual(midD) {
				// the coincidence can occur at almost any angle
				coincident = true
			}
		}
	}
	return coincident
}
