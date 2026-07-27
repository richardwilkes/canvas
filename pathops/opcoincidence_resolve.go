// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The coincidence resolution machinery: the phase run by handleCoincidence (common.go) after intersections are found.
// It matches the pt-t records of each coincident run (addExpanded), pairs on-curve points and endpoints missed by
// intersection (addEndMovedSpans, addMissing), loosely expands the runs (expand), marks the coincident spans (mark),
// transfers winding across coincident edges (apply), and constructs new pairs where different receivers overlap
// (findOverlaps). The recording/traversal surface (add, release, fixUp, markCollapsed, isEmpty, coincidenceOrdered)
// lives in opcoincidence.go.
//
// correctOneEnd takes get/set closures rather than a member pointer, since Go has no equivalent. ordered returns a
// (result, ok) pair instead of using an out-parameter for the "not fully processed" case. Every t value is a float64,
// matching the precision the boolean-op math needs throughout this package.
//
// This machinery runs inside handleCoincidence (common.go), which the Op/Simplify drivers invoke.

package pathops

// ---- coincidentSpans resolution methods ----

// correctOneEnd resets one end to the pt-t referenced by the previous-next span, so span ends agree with the segment's
// spans that define them.
func (cs *coincidentSpans) correctOneEnd(getEnd func() *opPtT, setEnd func(*opPtT)) {
	origPtT := getEnd()
	origSpan := origPtT.span
	prev := origSpan.prev
	var testPtT *opPtT
	if prev != nil {
		testPtT = prev.next.ptTPtr()
	} else {
		testPtT = origSpan.upCast().next.prev.ptTPtr()
	}
	if origPtT != testPtT {
		setEnd(testPtT)
	}
}

// correctEnds makes all four of the run's span ends agree with the defining spans.
func (cs *coincidentSpans) correctEnds() {
	cs.correctOneEnd(func() *opPtT { return cs.coinPtTStart }, cs.setCoinPtTStart)
	cs.correctOneEnd(func() *opPtT { return cs.coinPtTEnd }, cs.setCoinPtTEnd)
	cs.correctOneEnd(func() *opPtT { return cs.oppPtTStart }, cs.setOppPtTStart)
	cs.correctOneEnd(func() *opPtT { return cs.oppPtTEnd }, cs.setOppPtTEnd)
}

// expand grows the run by checking the spans adjacent to each end for coincidence. Returns true if either end moved.
func (cs *coincidentSpans) expand() bool {
	expanded := false
	segment := cs.coinPtTStart.segment()
	oppSegment := cs.oppPtTStart.segment()
	for {
		start := cs.coinPtTStart.span.upCast()
		prev := start.prev
		if prev == nil {
			break
		}
		oppPtT := prev.containsSegment(oppSegment)
		if oppPtT == nil {
			break
		}
		midT := (prev.t() + start.t()) / 2
		if !segment.isClose(midT, oppSegment) {
			break
		}
		cs.setStarts(prev.ptTPtr(), oppPtT)
		expanded = true
	}
	for {
		end := cs.coinPtTEnd.span
		var next *opSpanBase
		if !end.final() {
			next = end.upCast().next
		}
		if next != nil && next.deleted() {
			break
		}
		if next == nil {
			break
		}
		oppPtT := next.containsSegment(oppSegment)
		if oppPtT == nil {
			break
		}
		midT := (end.t() + next.t()) / 2
		if !segment.isClose(midT, oppSegment) {
			break
		}
		cs.setEnds(next.ptTPtr(), oppPtT)
		expanded = true
	}
	return expanded
}

// extend increases the run's range to include the given ends.
func (cs *coincidentSpans) extend(coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd *opPtT) bool {
	result := false
	var startMoved bool
	if cs.flipped() {
		startMoved = cs.oppPtTStart.t < oppPtTStart.t
	} else {
		startMoved = cs.oppPtTStart.t > oppPtTStart.t
	}
	if cs.coinPtTStart.t > coinPtTStart.t || startMoved {
		cs.setStarts(coinPtTStart, oppPtTStart)
		result = true
	}
	var endMoved bool
	if cs.flipped() {
		endMoved = cs.oppPtTEnd.t > oppPtTEnd.t
	} else {
		endMoved = cs.oppPtTEnd.t < oppPtTEnd.t
	}
	if cs.coinPtTEnd.t < coinPtTEnd.t || endMoved {
		cs.setEnds(coinPtTEnd, oppPtTEnd)
		result = true
	}
	return result
}

// contains reports whether both points fall inside this run.
func (cs *coincidentSpans) contains(s, e *opPtT) bool {
	if s.t > e.t {
		s, e = e, s
	}
	if s.segment() == cs.coinPtTStart.segment() {
		return cs.coinPtTStart.t <= s.t && e.t <= cs.coinPtTEnd.t
	}
	oppTs := cs.oppPtTStart.t
	oppTe := cs.oppPtTEnd.t
	if oppTs > oppTe {
		oppTs, oppTe = oppTe, oppTs
	}
	return oppTs <= s.t && e.t <= oppTe
}

// ordered reports whether the run is ordered: a run is unordered if the paired t values on the main and opposite curves
// do not ascend/descend together. Returns (result, ok); ok is false if the run isn't fully processed (a span in the
// range has no opposite match).
func (cs *coincidentSpans) ordered() (result, ok bool) {
	start := cs.coinPtTStart.span
	end := cs.coinPtTEnd.span
	next := start.upCast().next
	if next == end {
		return true, true
	}
	flipped := cs.flipped()
	oppSeg := cs.oppPtTStart.segment()
	oppLastT := cs.oppPtTStart.t
	for {
		opp := next.containsSegment(oppSeg)
		if opp == nil {
			return false, false
		}
		if (oppLastT > opp.t) != flipped {
			return false, true
		}
		oppLastT = opp.t
		if next == end {
			break
		}
		if next.upCastable() == nil {
			return false, true
		}
		next = next.upCast().next
	}
	return true, true
}

// ---- opCoincidence resolution methods ----

// extend reports whether an existing pair overlaps the addition; if so it extends that pair in place.
func (co *opCoincidence) extend(coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd *opPtT) bool {
	test := co.head
	if test == nil {
		return false
	}
	coinSeg := coinPtTStart.segment()
	oppSeg := oppPtTStart.segment()
	if !coincidenceOrderedSeg(coinSeg, oppSeg) {
		coinSeg, oppSeg = oppSeg, coinSeg
		coinPtTStart, oppPtTStart = oppPtTStart, coinPtTStart
		coinPtTEnd, oppPtTEnd = oppPtTEnd, coinPtTEnd
		if coinPtTStart.t > coinPtTEnd.t {
			coinPtTStart, coinPtTEnd = coinPtTEnd, coinPtTStart
			oppPtTStart, oppPtTEnd = oppPtTEnd, oppPtTStart
		}
	}
	oppMinT := min(oppPtTStart.t, oppPtTEnd.t)
	for ; test != nil; test = test.next {
		if coinSeg != test.coinPtTStart.segment() {
			continue
		}
		if oppSeg != test.oppPtTStart.segment() {
			continue
		}
		oTestMinT := min(test.oppPtTStart.t, test.oppPtTEnd.t)
		oTestMaxT := max(test.oppPtTStart.t, test.oppPtTEnd.t)
		if (test.coinPtTStart.t <= coinPtTEnd.t && coinPtTStart.t <= test.coinPtTEnd.t) ||
			(oTestMinT <= oTestMaxT && oppMinT <= oTestMaxT) {
			test.extend(coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd)
			return true
		}
	}
	return false
}

// addEndMovedSpansBase handles the case where the implied line from a coincident endpoint to its non-endpoint match may
// intersect another curve near the endpoint; it probes the pt-t list adjacent to the base/testSpan loop for a new
// coincident pair. Returns false on abort.
func (co *opCoincidence) addEndMovedSpansBase(base *opSpan, testSpan *opSpanBase) bool {
	testPtT := testSpan.ptTPtr()
	stopPtT := testPtT
	baseSeg := base.segment
	escapeHatch := 100000 // 100x the debug loop limit; a fuzz-generated test too complex to succeed
	for {
		testPtT = testPtT.next
		if testPtT == stopPtT {
			break
		}
		escapeHatch--
		if escapeHatch <= 0 {
			return false
		}
		testSeg := testPtT.segment()
		if testPtT.deleted {
			continue
		}
		if testSeg == baseSeg {
			continue
		}
		if testPtT.span.ptTPtr() != testPtT {
			continue
		}
		if co.containsSegOppT(baseSeg, testSeg, testPtT.t) {
			continue
		}
		// intersect the perpendicular at base->ptT() with testPtT's segment
		dxdy := baseSeg.dSlopeAtT(base.t())
		pt := base.pt()
		ray := dLine{pts: [2]dPoint{
			{x: float64(pt.X), y: float64(pt.Y)},
			{x: float64(pt.X) + dxdy.y, y: float64(pt.Y) - dxdy.x},
		}}
		i := newIntersections()
		curveIntersectRay(testSeg.verb, testSeg.pts, testSeg.weight, ray, i)
		for index := 0; index < i.usedCount(); index++ {
			t := i.tVal(0, index)
			if !between(0, t, 1) {
				continue
			}
			oppPt := i.pt(index)
			if !oppPt.approximatelyEqualPt(pt) {
				continue
			}
			oppStart := testSeg.addT(t)
			if oppStart == testPtT {
				continue
			}
			oppStart.span.addOpp(&base.opSpanBase)
			if oppStart.deleted {
				continue
			}
			coinSeg := base.segment
			oppSeg := oppStart.segment()
			var coinTs, coinTe, oppTs, oppTe float64
			if coincidenceOrderedSeg(coinSeg, oppSeg) {
				coinTs = base.t()
				coinTe = testSpan.t()
				oppTs = oppStart.t
				oppTe = testPtT.t
			} else {
				coinSeg, oppSeg = oppSeg, coinSeg
				coinTs = oppStart.t
				coinTe = testPtT.t
				oppTs = base.t()
				oppTe = testSpan.t()
			}
			if coinTs > coinTe {
				coinTs, coinTe = coinTe, coinTs
				oppTs, oppTe = oppTe, oppTs
			}
			var added bool
			if !co.addOrOverlap(coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, &added) {
				return false
			}
		}
	}
	return true
}

// addEndMovedSpansPtT probes the spans on either side of ptT for a new coincident pair.
func (co *opCoincidence) addEndMovedSpansPtT(ptT *opPtT) bool {
	if ptT.span.upCastable() == nil {
		return false
	}
	base := ptT.span.upCast()
	prev := base.prev
	if prev == nil {
		return false
	}
	if !prev.isCanceled() {
		if !co.addEndMovedSpansBase(base, &prev.opSpanBase) {
			return false
		}
	}
	if !base.isCanceled() {
		if !co.addEndMovedSpansBase(base, base.next) {
			return false
		}
	}
	return true
}

// addEndMovedSpans looks, for each recorded run whose endpoints don't coincide, for an undetected coincident pair
// implied by the moved endpoint. Moves head to top while it walks so the walker sees the unperturbed old list, then
// restoreHead re-splices.
func (co *opCoincidence) addEndMovedSpans() bool {
	span := co.head
	if span == nil {
		return true
	}
	co.top = span
	co.head = nil
	for ; span != nil; span = span.next {
		if span.coinPtTStart.pt != span.oppPtTStart.pt {
			if span.coinPtTStart.t == 1 {
				return false
			}
			onEnd := span.coinPtTStart.t == 0
			oOnEnd := zeroOrOne(span.oppPtTStart.t)
			if onEnd {
				if !oOnEnd { // if both are on end, any nearby intersect was already found
					if !co.addEndMovedSpansPtT(span.oppPtTStart) {
						return false
					}
				}
			} else if oOnEnd {
				if !co.addEndMovedSpansPtT(span.coinPtTStart) {
					return false
				}
			}
		}
		if span.coinPtTEnd.pt != span.oppPtTEnd.pt {
			onEnd := span.coinPtTEnd.t == 1
			oOnEnd := zeroOrOne(span.oppPtTEnd.t)
			if onEnd {
				if !oOnEnd {
					if !co.addEndMovedSpansPtT(span.oppPtTEnd) {
						return false
					}
				}
			} else if oOnEnd {
				if !co.addEndMovedSpansPtT(span.coinPtTEnd) {
					return false
				}
			}
		}
	}
	co.restoreHead()
	return true
}

// addExpanded matches the spans for each coincident pair; if they don't match, it adds the missing t to the segment and
// loops it into the opposite span. Returns false on abort.
func (co *opCoincidence) addExpanded() bool {
	coin := co.head
	if coin == nil {
		return true
	}
	for ; coin != nil; coin = coin.next {
		startPtT := coin.coinPtTStart
		oStartPtT := coin.oppPtTStart
		priorT := startPtT.t
		oPriorT := oStartPtT.t
		if !startPtT.containsPtT(oStartPtT) {
			return false
		}
		start := startPtT.span
		oStart := oStartPtT.span
		end := coin.coinPtTEnd.span
		oEnd := coin.oppPtTEnd.span
		if oEnd.deleted() {
			return false
		}
		if start.upCastable() == nil {
			return false
		}
		test := start.upCast().next
		if !coin.flipped() && oStart.upCastable() == nil {
			return false
		}
		var oTest *opSpanBase
		if coin.flipped() {
			if oStart.prev != nil {
				oTest = &oStart.prev.opSpanBase
			}
		} else {
			oTest = oStart.upCast().next
		}
		if oTest == nil {
			return false
		}
		seg := start.segment
		oSeg := oStart.segment
		for test != end || oTest != oEnd {
			containedOpp := test.ptT.containsSeg(oSeg)
			containedThis := oTest.ptT.containsSeg(seg)
			if containedOpp == nil || containedThis == nil {
				// choose the ends, or the first common pt-t list shared by both
				var nextT, oNextT float64
				switch {
				case containedOpp != nil:
					nextT = test.t()
					oNextT = containedOpp.t
				case containedThis != nil:
					nextT = containedThis.t
					oNextT = oTest.t()
				default:
					// iterate until a pt-t list is found that contains the other
					walk := test
					var walkOpp *opPtT
					for {
						if walk.upCastable() == nil {
							return false
						}
						walk = walk.upCast().next
						walkOpp = walk.ptT.containsSeg(oSeg)
						if walkOpp != nil || walk == coin.coinPtTEnd.span {
							break
						}
					}
					if walkOpp == nil {
						return false
					}
					nextT = walk.t()
					oNextT = walkOpp.t
				}
				// use t ranges to guess which one is missing
				startRange := nextT - priorT
				if startRange == 0 {
					return false
				}
				startPart := (test.t() - priorT) / startRange
				oStartRange := oNextT - oPriorT
				if oStartRange == 0 {
					return false
				}
				oStartPart := (oTest.t() - oPriorT) / oStartRange
				if startPart == oStartPart {
					return false
				}
				var addToOpp bool
				if containedOpp == nil && containedThis == nil {
					addToOpp = startPart < oStartPart
				} else {
					addToOpp = containedThis != nil
				}
				startOver := false
				var success bool
				if addToOpp {
					success = oSeg.addExpanded(oPriorT+oStartRange*startPart, test, &startOver)
				} else {
					success = seg.addExpanded(priorT+startRange*oStartPart, oTest, &startOver)
				}
				if !success {
					return false
				}
				if startOver {
					test = start
					oTest = oStart
				}
				end = coin.coinPtTEnd.span
				oEnd = coin.oppPtTEnd.span
			}
			if test != end {
				if test.upCastable() == nil {
					return false
				}
				priorT = test.t()
				test = test.upCast().next
			}
			if oTest != oEnd {
				oPriorT = oTest.t()
				if coin.flipped() {
					if oTest.prev != nil {
						oTest = &oTest.prev.opSpanBase
					} else {
						oTest = nil
					}
				} else {
					if oTest.upCastable() == nil {
						return false
					}
					oTest = oTest.upCast().next
				}
				if oTest == nil {
					return false
				}
			}
		}
	}
	return true
}

// trange maps t on overS's span into coinSeg's t range, using the smallest known pt-t interval that captures t. Returns
// 1 when no bracketing interval exists.
func trange(overS *opPtT, t float64, coinSeg *opSegment) float64 {
	work := overS.span
	var foundStart, foundEnd, coinStart, coinEnd *opPtT
	for {
		contained := work.containsSegment(coinSeg)
		if contained == nil {
			if work.final() {
				break
			}
			work = work.upCast().next
			continue
		}
		if work.t() <= t {
			coinStart = contained
			foundStart = work.ptTPtr()
		}
		if work.t() >= t {
			coinEnd = contained
			foundEnd = work.ptTPtr()
			break
		}
		work = work.upCast().next
	}
	if coinStart == nil || coinEnd == nil {
		return 1
	}
	denom := foundEnd.t - foundStart.t
	sRatio := 1.0
	if denom != 0 {
		sRatio = (t - foundStart.t) / denom
	}
	return coinStart.t + (coinEnd.t-coinStart.t)*sRatio
}

// checkOverlap reports whether the span overlaps existing runs and the coincident list needs adjusting; the
// partially-overlapping runs are appended to overlaps.
func (co *opCoincidence) checkOverlap(check *coincidentSpans, coinSeg, oppSeg *opSegment, coinTs, coinTe, oppTs, oppTe float64, overlaps *[]*coincidentSpans) bool {
	if !coincidenceOrderedSeg(coinSeg, oppSeg) {
		if oppTs < oppTe {
			return co.checkOverlap(check, oppSeg, coinSeg, oppTs, oppTe, coinTs, coinTe, overlaps)
		}
		return co.checkOverlap(check, oppSeg, coinSeg, oppTe, oppTs, coinTe, coinTs, overlaps)
	}
	swapOpp := oppTs > oppTe
	if swapOpp {
		oppTs, oppTe = oppTe, oppTs
	}
	for ; check != nil; check = check.next {
		if check.coinPtTStart.segment() != coinSeg {
			continue
		}
		if check.oppPtTStart.segment() != oppSeg {
			continue
		}
		checkTs := check.coinPtTStart.t
		checkTe := check.coinPtTEnd.t
		coinOutside := coinTe < checkTs || coinTs > checkTe
		oCheckTs := check.oppPtTStart.t
		oCheckTe := check.oppPtTEnd.t
		if swapOpp {
			if oCheckTs <= oCheckTe {
				return false
			}
			oCheckTs, oCheckTe = oCheckTe, oCheckTs
		}
		oppOutside := oppTe < oCheckTs || oppTs > oCheckTe
		if coinOutside && oppOutside {
			continue
		}
		coinInside := coinTe <= checkTe && coinTs >= checkTs
		oppInside := oppTe <= oCheckTe && oppTs >= oCheckTs
		if coinInside && oppInside { // already included, do nothing
			return false
		}
		*overlaps = append(*overlaps, check) // partial overlap, extend existing entry
	}
	return true
}

// addIfMissing maps the (already ordered) tStart/tEnd range from over1s/over2s onto coinSeg and oppSeg and, unless the
// mapped range collapses, adds or overlaps the pair. It returns false only for a collapsed range: addOrOverlap's own
// result is deliberately discarded, because from here a false means "there was nothing to add", not "abort". The sole
// caller, addMissing, treats a false as a fatal abort of the whole operation, so propagating one would turn every
// nothing-to-add into a failed Op/Simplify. Upstream discards it the same way (an explicit `(void)` cast on the call in
// SkOpCoincidence::addIfMissing); see addOrOverlap's own comment for the two callers' opposing conventions.
func (co *opCoincidence) addIfMissing(over1s, over2s *opPtT, tStart, tEnd float64, coinSeg, oppSeg *opSegment, added *bool) bool {
	coinTs := trange(over1s, tStart, coinSeg)
	coinTe := trange(over1s, tEnd, coinSeg)
	result := coinSeg.collapsed(coinTs, coinTe)
	if result != spanCollapsedNo {
		return result == spanCollapsedYes
	}
	oppTs := trange(over2s, tStart, oppSeg)
	oppTe := trange(over2s, tEnd, oppSeg)
	result = oppSeg.collapsed(oppTs, oppTe)
	if result != spanCollapsedNo {
		return result == spanCollapsedYes
	}
	if coinTs > coinTe {
		coinTs, coinTe = coinTe, coinTs
		oppTs, oppTe = oppTe, oppTs
	}
	// The discarded result is intentional; see the doc comment. *added still reports whether anything was recorded.
	_ = co.addOrOverlap(coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, added)
	return true
}

// pickPtT returns a when non-nil, else b.
func pickPtT(a, b *opPtT) *opPtT {
	if a != nil {
		return a
	}
	return b
}

// coincidentRunIsReal reports whether coinSeg over [coinTs, coinTe] actually lies along oppSeg, rather than merely
// meeting it at the run's two ends. Both callers infer a run from matched endpoints alone, which is wrong wherever a
// curve meets the opposite segment twice without coinciding between -- a chord and the arc it subtends being the
// everyday case -- so the interior is sampled before the run is believed.
func coincidentRunIsReal(coinSeg, oppSeg *opSegment, coinTs, coinTe float64) bool {
	for _, part := range [3]float64{0.25, 0.5, 0.75} {
		if !coinSeg.isClose(coinTs+(coinTe-coinTs)*part, oppSeg) {
			return false
		}
	}
	return true
}

// addOrOverlap adds a coincident pair, or extends an overlapping one. Its two callers read a false return oppositely:
// from addEndMovedSpans it propagates out to an abort, while addIfMissing discards it as "there was nothing to add".
// Note that several of the false returns below fire after addT has already placed spans and span.addOpp has already
// spliced pt-t loops, so a false does not imply the span graph was left untouched -- that is upstream's behavior too,
// and the reason the abort/no-abort split is the caller's decision rather than this function's.
func (co *opCoincidence) addOrOverlap(coinSeg, oppSeg *opSegment, coinTs, coinTe, oppTs, oppTe float64, added *bool) bool {
	var overlaps []*coincidentSpans
	if co.top == nil {
		return false
	}
	if !coincidentRunIsReal(coinSeg, oppSeg, coinTs, coinTe) {
		return true // nothing to add; a false return here would abort the caller
	}
	if !co.checkOverlap(co.top, coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, &overlaps) {
		return true
	}
	if co.head != nil && !co.checkOverlap(co.head, coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, &overlaps) {
		return true
	}
	var overlap *coincidentSpans
	if len(overlaps) > 0 {
		overlap = overlaps[0]
	}
	for index := 1; index < len(overlaps); index++ { // combine overlaps before continuing
		test := overlaps[index]
		if overlap.coinPtTStart.t > test.coinPtTStart.t {
			overlap.setCoinPtTStart(test.coinPtTStart)
		}
		if overlap.coinPtTEnd.t < test.coinPtTEnd.t {
			overlap.setCoinPtTEnd(test.coinPtTEnd)
		}
		var startMoved bool
		if overlap.flipped() {
			startMoved = overlap.oppPtTStart.t < test.oppPtTStart.t
		} else {
			startMoved = overlap.oppPtTStart.t > test.oppPtTStart.t
		}
		if startMoved {
			overlap.setOppPtTStart(test.oppPtTStart)
		}
		var endMoved bool
		if overlap.flipped() {
			endMoved = overlap.oppPtTEnd.t > test.oppPtTEnd.t
		} else {
			endMoved = overlap.oppPtTEnd.t < test.oppPtTEnd.t
		}
		if endMoved {
			overlap.setOppPtTEnd(test.oppPtTEnd)
		}
		if co.head == nil || !co.release(co.head, test) {
			co.release(co.top, test)
		}
	}
	cs := coinSeg.existing(coinTs, oppSeg)
	ce := coinSeg.existing(coinTe, oppSeg)
	if overlap != nil && cs != nil && ce != nil && overlap.contains(cs, ce) {
		return true
	}
	if cs == ce && cs != nil {
		return false
	}
	os := oppSeg.existing(oppTs, coinSeg)
	oe := oppSeg.existing(oppTe, coinSeg)
	if overlap != nil && os != nil && oe != nil && overlap.contains(os, oe) {
		return true
	}
	if cs != nil && cs.deleted {
		return false
	}
	if os != nil && os.deleted {
		return false
	}
	if ce != nil && ce.deleted {
		return false
	}
	if oe != nil && oe.deleted {
		return false
	}
	var csExisting, ceExisting *opPtT
	if cs == nil {
		csExisting = coinSeg.existing(coinTs, nil)
	}
	if ce == nil {
		ceExisting = coinSeg.existing(coinTe, nil)
	}
	if csExisting != nil && csExisting == ceExisting {
		return false
	}
	if ceExisting != nil && (ceExisting == cs || ceExisting.containsPtT(pickPtT(csExisting, cs))) {
		return false
	}
	var osExisting, oeExisting *opPtT
	if os == nil {
		osExisting = oppSeg.existing(oppTs, nil)
	}
	if oe == nil {
		oeExisting = oppSeg.existing(oppTe, nil)
	}
	if osExisting != nil && osExisting == oeExisting {
		return false
	}
	if osExisting != nil && (osExisting == oe || osExisting.containsPtT(pickPtT(oeExisting, oe))) {
		return false
	}
	if oeExisting != nil && (oeExisting == os || oeExisting.containsPtT(pickPtT(osExisting, os))) {
		return false
	}
	if cs == nil || os == nil {
		csWritable := cs
		if csWritable == nil {
			csWritable = coinSeg.addT(coinTs)
		}
		if csWritable == ce {
			return true
		}
		osWritable := os
		if osWritable == nil {
			osWritable = oppSeg.addT(oppTs)
		}
		if csWritable == nil || osWritable == nil {
			return false
		}
		csWritable.span.addOpp(osWritable.span)
		cs = csWritable
		os = osWritable.active()
		if os == nil {
			return false
		}
		if (ce != nil && ce.deleted) || (oe != nil && oe.deleted) {
			return false
		}
	}
	if ce == nil || oe == nil {
		ceWritable := ce
		if ceWritable == nil {
			ceWritable = coinSeg.addT(coinTe)
		}
		oeWritable := oe
		if oeWritable == nil {
			oeWritable = oppSeg.addT(oppTe)
		}
		if ceWritable == nil || oeWritable == nil {
			return false
		}
		if !ceWritable.span.addOpp(oeWritable.span) {
			return false
		}
		ce = ceWritable
		oe = oeWritable
	}
	if cs.deleted || os.deleted || ce.deleted || oe.deleted {
		return false
	}
	if cs.containsPtT(ce) || os.containsPtT(oe) {
		return false
	}
	result := true
	if overlap != nil {
		if overlap.coinPtTStart.segment() == coinSeg {
			result = overlap.extend(cs, ce, os, oe)
		} else {
			if os.t > oe.t {
				cs, ce = ce, cs
				os, oe = oe, os
			}
			result = overlap.extend(os, oe, cs, ce)
		}
	} else {
		co.add(cs, ce, os, oe)
	}
	if result {
		*added = true
	}
	return true
}

// addMissing detects overlaps of different coincident runs sharing a segment (coincidence present in A-B and A-C but
// missing in B-C). Sets *added and returns false on abort.
func (co *opCoincidence) addMissing(added *bool) bool {
	outer := co.head
	*added = false
	if outer == nil {
		return true
	}
	co.top = outer
	co.head = nil
	for ; outer != nil; outer = outer.next {
		// addIfMissing can modify the list being walked; top holds the saved head so the walker iterates the old data
		// unperturbed, adding freely to head, then restoreHead re-splices at the end.
		ocs := outer.coinPtTStart
		if ocs.deleted {
			return false
		}
		outerCoin := ocs.segment()
		if outerCoin.done() {
			return false
		}
		oos := outer.oppPtTStart
		if oos.deleted {
			return true
		}
		outerOpp := oos.segment()
		inner := outer
		for {
			inner = inner.next
			if inner == nil {
				break
			}
			var overS, overE float64
			var ok bool
			ics := inner.coinPtTStart
			if ics.deleted {
				return false
			}
			innerCoin := ics.segment()
			if innerCoin.done() {
				return false
			}
			ios := inner.oppPtTStart
			if ios.deleted {
				return false
			}
			innerOpp := ios.segment()
			switch {
			case outerCoin == innerCoin:
				oce := outer.coinPtTEnd
				if oce.deleted {
					return true
				}
				ice := inner.coinPtTEnd
				if ice.deleted {
					return false
				}
				if outerOpp != innerOpp {
					if overS, overE, ok = co.overlap(ocs, oce, ics, ice); ok {
						if !co.addIfMissing(ocs.starter(oce), ics.starter(ice), overS, overE,
							outerOpp, innerOpp, added) {
							return false
						}
					}
				}
			case outerCoin == innerOpp:
				oce := outer.coinPtTEnd
				if oce.deleted {
					return false
				}
				ioe := inner.oppPtTEnd
				if ioe.deleted {
					return false
				}
				if outerOpp != innerCoin {
					if overS, overE, ok = co.overlap(ocs, oce, ios, ioe); ok {
						if !co.addIfMissing(ocs.starter(oce), ios.starter(ioe), overS, overE,
							outerOpp, innerCoin, added) {
							return false
						}
					}
				}
			case outerOpp == innerCoin:
				ooe := outer.oppPtTEnd
				if ooe.deleted {
					return false
				}
				ice := inner.coinPtTEnd
				if ice.deleted {
					return false
				}
				if overS, overE, ok = co.overlap(oos, ooe, ics, ice); ok {
					if !co.addIfMissing(oos.starter(ooe), ics.starter(ice), overS, overE,
						outerCoin, innerOpp, added) {
						return false
					}
				}
			case outerOpp == innerOpp:
				ooe := outer.oppPtTEnd
				if ooe.deleted {
					return false
				}
				ioe := inner.oppPtTEnd
				if ioe.deleted {
					return true
				}
				if overS, overE, ok = co.overlap(oos, ooe, ios, ioe); ok {
					if !co.addIfMissing(oos.starter(ooe), ios.starter(ioe), overS, overE,
						outerCoin, innerCoin, added) {
						return false
					}
				}
			}
		}
	}
	co.restoreHead()
	return true
}

// addOverlap constructs the pair resolving the mutual span of two coincident runs that overlap on a shared segment,
// choosing the run with nonzero wind value.
func (co *opCoincidence) addOverlap(seg1, seg1o, seg2, seg2o *opSegment, overS, overE *opPtT) bool {
	s1 := overS.find(seg1)
	e1 := overE.find(seg1)
	if s1 == nil || e1 == nil {
		return false
	}
	if s1.starter(e1).span.upCast().windValue == 0 {
		s1 = overS.find(seg1o)
		e1 = overE.find(seg1o)
		if s1 == nil || e1 == nil {
			return false
		}
		if s1.starter(e1).span.upCast().windValue == 0 {
			return true
		}
	}
	s2 := overS.find(seg2)
	e2 := overE.find(seg2)
	if s2 == nil || e2 == nil {
		return false
	}
	if s2.starter(e2).span.upCast().windValue == 0 {
		s2 = overS.find(seg2o)
		e2 = overE.find(seg2o)
		if s2 == nil || e2 == nil {
			return false
		}
		if s2.starter(e2).span.upCast().windValue == 0 {
			return true
		}
	}
	if s1.segment() == s2.segment() {
		return true
	}
	if s1.t > e1.t {
		s1, e1 = e1, s1
		s2, e2 = e2, s2
	}
	co.add(s1, e1, s2, e2)
	return true
}

// containsSegOppT reports whether some recorded run pairs seg with opp and its opposite-side t range brackets oppT.
func (co *opCoincidence) containsSegOppT(seg, opp *opSegment, oppT float64) bool {
	return co.containsCoinSegOppT(co.head, seg, opp, oppT) ||
		co.containsCoinSegOppT(co.top, seg, opp, oppT)
}

func (co *opCoincidence) containsCoinSegOppT(coin *coincidentSpans, seg, opp *opSegment, oppT float64) bool {
	for ; coin != nil; coin = coin.next {
		if coin.coinPtTStart.segment() == seg && coin.oppPtTStart.segment() == opp &&
			between(coin.oppPtTStart.t, oppT, coin.oppPtTEnd.t) {
			return true
		}
		if coin.oppPtTStart.segment() == seg && coin.coinPtTStart.segment() == opp &&
			between(coin.coinPtTStart.t, oppT, coin.coinPtTEnd.t) {
			return true
		}
	}
	return false
}

// containsPtTs reports whether the given coincident/opposite pt-t pair falls inside an existing run.
func (co *opCoincidence) containsPtTs(coinPtTStart, coinPtTEnd, oppPtTStart, oppPtTEnd *opPtT) bool {
	test := co.head
	if test == nil {
		return false
	}
	coinSeg := coinPtTStart.segment()
	oppSeg := oppPtTStart.segment()
	if !coincidenceOrderedSeg(coinSeg, oppSeg) {
		coinSeg, oppSeg = oppSeg, coinSeg
		coinPtTStart, oppPtTStart = oppPtTStart, coinPtTStart
		coinPtTEnd, oppPtTEnd = oppPtTEnd, coinPtTEnd
		if coinPtTStart.t > coinPtTEnd.t {
			coinPtTStart, coinPtTEnd = coinPtTEnd, coinPtTStart
			oppPtTStart, oppPtTEnd = oppPtTEnd, oppPtTStart
		}
	}
	oppMinT := min(oppPtTStart.t, oppPtTEnd.t)
	oppMaxT := max(oppPtTStart.t, oppPtTEnd.t)
	for ; test != nil; test = test.next {
		if coinSeg != test.coinPtTStart.segment() {
			continue
		}
		if coinPtTStart.t < test.coinPtTStart.t {
			continue
		}
		if coinPtTEnd.t > test.coinPtTEnd.t {
			continue
		}
		if oppSeg != test.oppPtTStart.segment() {
			continue
		}
		if oppMinT < min(test.oppPtTStart.t, test.oppPtTEnd.t) {
			continue
		}
		if oppMaxT > max(test.oppPtTStart.t, test.oppPtTEnd.t) {
			continue
		}
		return true
	}
	return false
}

// correctEnds calls correctEnds on every recorded coincident run.
func (co *opCoincidence) correctEnds() {
	for coin := co.head; coin != nil; coin = coin.next {
		coin.correctEnds()
	}
}

// apply walks each coincident span set in parallel, moving winding from one to the other and marking spans done once
// their winding drops to zero. Returns false on abort.
func (co *opCoincidence) apply() bool {
	coin := co.head
	if coin == nil {
		return true
	}
	for ; coin != nil; coin = coin.next {
		startSpan := coin.coinPtTStart.span
		if startSpan.upCastable() == nil {
			return false
		}
		start := startSpan.upCast()
		if start.deleted() {
			continue
		}
		end := coin.coinPtTEnd.span
		if start != start.starter(end) {
			return false
		}
		flipped := coin.flipped()
		var oStartBase *opSpanBase
		if flipped {
			oStartBase = coin.oppPtTEnd.span
		} else {
			oStartBase = coin.oppPtTStart.span
		}
		if oStartBase.upCastable() == nil {
			return false
		}
		oStart := oStartBase.upCast()
		if oStart.deleted() {
			continue
		}
		var oEnd *opSpanBase
		if flipped {
			oEnd = coin.oppPtTStart.span
		} else {
			oEnd = coin.oppPtTEnd.span
		}
		segment := start.segment
		oSegment := oStart.segment
		operandSwap := segment.operand() != oSegment.operand()
		if flipped {
			if oEnd.deleted() {
				continue
			}
			for {
				oNext := oStart.next
				if oNext == oEnd {
					break
				}
				if oNext.upCastable() == nil {
					return false
				}
				oStart = oNext.upCast()
			}
		}
		for {
			windValue := start.windValue
			oppValue := start.oppValue
			oWindValue := oStart.windValue
			oOppValue := oStart.oppValue
			// winding values add or subtract by direction and wind type; same/opposite values sum by operand
			windDiff := oWindValue
			oWindDiff := windValue
			if operandSwap {
				windDiff = oOppValue
				oWindDiff = oppValue
			}
			if !flipped {
				windDiff = -windDiff
				oWindDiff = -oWindDiff
			}
			addToStart := windValue != 0 && (windValue > windDiff ||
				(windValue == windDiff && oWindValue <= oWindDiff))
			if addToStart {
				if start.done {
					addToStart = false
				}
			} else {
				if oStart.done {
					addToStart = true
				}
			}
			if addToStart {
				if operandSwap {
					oWindValue, oOppValue = oOppValue, oWindValue
				}
				if flipped {
					windValue -= oWindValue
					oppValue -= oOppValue
				} else {
					windValue += oWindValue
					oppValue += oOppValue
				}
				if segment.isXor() {
					windValue &= 1
				}
				if segment.oppXor() {
					oppValue &= 1
				}
				oWindValue = 0
				oOppValue = 0
			} else {
				if operandSwap {
					windValue, oppValue = oppValue, windValue
				}
				if flipped {
					oWindValue -= windValue
					oOppValue -= oppValue
				} else {
					oWindValue += windValue
					oOppValue += oppValue
				}
				if oSegment.isXor() {
					oWindValue &= 1
				}
				if oSegment.oppXor() {
					oOppValue &= 1
				}
				windValue = 0
				oppValue = 0
			}
			if windValue <= -1 {
				return false
			}
			start.setWindValue(windValue)
			start.setOppValue(oppValue)
			if oWindValue <= -1 {
				return false
			}
			oStart.setWindValue(oWindValue)
			oStart.setOppValue(oOppValue)
			if windValue == 0 && oppValue == 0 {
				segment.markDone(start)
			}
			if oWindValue == 0 && oOppValue == 0 {
				oSegment.markDone(oStart)
			}
			next := start.next
			var oNext *opSpanBase
			if flipped {
				if oStart.prev != nil {
					oNext = &oStart.prev.opSpanBase
				}
			} else {
				oNext = oStart.next
			}
			if next == end {
				break
			}
			if next.upCastable() == nil {
				return false
			}
			start = next.upCast()
			// if the opposite ran out too soon, just reuse the last span
			if oNext == nil || oNext.upCastable() == nil {
				oNext = &oStart.opSpanBase
			}
			oStart = oNext.upCast()
		}
	}
	return true
}

// restoreHead appends top to the tail of head, then drops runs that reference a segment that has since collapsed
// (become done).
func (co *opCoincidence) restoreHead() {
	if co.head == nil {
		co.head = co.top
	} else {
		tail := co.head
		for tail.next != nil {
			tail = tail.next
		}
		tail.next = co.top
	}
	co.top = nil
	var prev *coincidentSpans
	for cur := co.head; cur != nil; {
		if cur.coinPtTStart.segment().done() || cur.oppPtTStart.segment().done() {
			if prev == nil {
				co.head = cur.next
			} else {
				prev.next = cur.next
			}
			cur = cur.next
			continue
		}
		prev = cur
		cur = cur.next
	}
}

// expand loosely expands each run; when two runs expand to become identical, it releases the duplicate. Returns true if
// any run expanded.
func (co *opCoincidence) expand() bool {
	coin := co.head
	if coin == nil {
		return false
	}
	expanded := false
	for ; coin != nil; coin = coin.next {
		if coin.expand() {
			// check whether multiple spans expanded so they are now identical
			for test := co.head; test != nil; test = test.next {
				if coin == test {
					continue
				}
				if coin.coinPtTStart == test.coinPtTStart && coin.oppPtTStart == test.oppPtTStart {
					co.release(co.head, test)
					break
				}
			}
			expanded = true
		}
	}
	return expanded
}

// findOverlaps adds a new pair (into overlaps) resolving the mutual span for each pair of runs whose receivers differ
// but whose other segments overlap. Returns false on abort.
func (co *opCoincidence) findOverlaps(overlaps *opCoincidence) bool {
	overlaps.head = nil
	overlaps.top = nil
	for outer := co.head; outer != nil; outer = outer.next {
		outerCoin := outer.coinPtTStart.segment()
		outerOpp := outer.oppPtTStart.segment()
		for inner := outer.next; inner != nil; inner = inner.next {
			innerCoin := inner.coinPtTStart.segment()
			if outerCoin == innerCoin {
				continue // both winners are the same segment, so there's no additional overlap
			}
			innerOpp := inner.oppPtTStart.segment()
			var overlapS, overlapE *opPtT
			ok := false
			if outerOpp == innerCoin {
				overlapS, overlapE, ok = opPtTOverlaps(outer.oppPtTStart, outer.oppPtTEnd,
					inner.coinPtTStart, inner.coinPtTEnd)
			}
			if !ok && outerCoin == innerOpp {
				overlapS, overlapE, ok = opPtTOverlaps(outer.coinPtTStart, outer.coinPtTEnd,
					inner.oppPtTStart, inner.oppPtTEnd)
			}
			if !ok && outerOpp == innerOpp {
				overlapS, overlapE, ok = opPtTOverlaps(outer.oppPtTStart, outer.oppPtTEnd,
					inner.oppPtTStart, inner.oppPtTEnd)
			}
			if ok {
				if !overlaps.addOverlap(outerCoin, outerOpp, innerCoin, innerOpp, overlapS, overlapE) {
					return false
				}
			}
		}
	}
	return true
}

// mark sets up the coincidence links in the segments where a coincidence crosses multiple spans. Returns false on
// abort.
func (co *opCoincidence) mark() bool {
	coin := co.head
	if coin == nil {
		return true
	}
	for ; coin != nil; coin = coin.next {
		startBase := coin.coinPtTStart.span
		if startBase.upCastable() == nil {
			return false
		}
		start := startBase.upCast()
		if start.deleted() {
			return false
		}
		end := coin.coinPtTEnd.span
		oStart := coin.oppPtTStart.span
		oEnd := coin.oppPtTEnd.span
		if oEnd.deleted() {
			return false
		}
		flipped := coin.flipped()
		if flipped {
			oStart, oEnd = oEnd, oStart
		}
		// coin and opp spans may not match up; mark the ends, then let the interior get marked as many times as the
		// spans allow
		if oStart.upCastable() == nil {
			return false
		}
		start.insertCoincidenceSpan(oStart.upCast())
		end.insertCoinEnd(oEnd)
		segment := start.segment
		oSegment := oStart.segment
		ordered, ok := coin.ordered()
		if !ok {
			return false
		}
		next := &start.opSpanBase
		for {
			next = next.upCast().next
			if next == end {
				break
			}
			if next.upCastable() == nil {
				return false
			}
			if !next.upCast().insertCoincidenceSeg(oSegment, flipped, ordered) {
				return false
			}
		}
		oNext := oStart
		for {
			oNext = oNext.upCast().next
			if oNext == oEnd {
				break
			}
			if oNext.upCastable() == nil {
				return false
			}
			if !oNext.upCast().insertCoincidenceSeg(segment, flipped, ordered) {
				return false
			}
		}
	}
	return true
}

// overlap returns the shared t range of two runs on the same segment, as (overS, overE, ok) with ok true when the range
// is non-empty.
func (co *opCoincidence) overlap(coin1s, coin1e, coin2s, coin2e *opPtT) (overS, overE float64, ok bool) {
	overS = max(min(coin1s.t, coin1e.t), min(coin2s.t, coin2e.t))
	overE = min(max(coin1s.t, coin1e.t), max(coin2s.t, coin2e.t))
	return overS, overE, overS < overE
}

// releaseSegment drops every run that references the deleted segment.
func (co *opCoincidence) releaseSegment(deleted *opSegment) {
	coin := co.head
	if coin == nil {
		return
	}
	for ; coin != nil; coin = coin.next {
		if coin.coinPtTStart.segment() == deleted || coin.coinPtTEnd.segment() == deleted ||
			coin.oppPtTStart.segment() == deleted || coin.oppPtTEnd.segment() == deleted {
			co.release(co.head, coin)
		}
	}
}
