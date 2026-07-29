// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The point-and-t records threaded through a segment: opPtT, opSpanBase, and opSpan. Provides the structures and their
// construction/addT surface (the pt-t loop machinery, the span linked list, containment queries) plus the
// intersection-time methods that fold spans together and carry the walk's bookkeeping (addOpp, merge, mergeMatches,
// release, setWindSum/setOppSum). The angle loops those spans point at are built in opsegment.go.

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

const (
	skMinS32 = -math.MaxInt32 // the sentinel used to mark a winding sum as not-yet-set
	skMaxS32 = math.MaxInt32  // the largest representable 32-bit winding sum
	// skNaN32 is the sentinel for "not a number" in 32-bit winding arithmetic. It must stay distinct from skMinS32:
	// computeSum returns skNaN32 only for a degenerate angle loop, while a span whose winding was never transferred
	// keeps skMinS32 and is still sortable (updateWinding re-seeds it with a ray cast).
	skNaN32 = math.MinInt32
)

// opPtT is a point/t value on a segment, linked into a loop of coincident points across segments (or aliases on the
// same curve).
type opPtT struct {
	span       *opSpanBase
	next       *opPtT     // intersection on the opposite curve or alias on this curve
	t          float64    // the curve parameter
	pt         geom.Point // cache of the point value at t
	deleted    bool
	coincident bool // set if at some point a coincident span pointed here
}

// init sets up the pt-t record as a fresh, single-element loop (next points to itself).
func (p *opPtT) init(span *opSpanBase, t float64, pt geom.Point) {
	p.t = t
	p.pt = pt
	p.span = span
	p.next = p
	p.deleted = false
	p.coincident = false
}

// active returns this pt-t if not deleted, else the first non-deleted alias on the same span.
func (p *opPtT) active() *opPtT {
	if !p.deleted {
		return p
	}
	ptT := p
	stop := ptT
	for ptT = ptT.next; ptT != stop; ptT = ptT.next {
		if ptT.span == p.span && !ptT.deleted {
			return ptT
		}
	}
	return nil // should never return deleted; caller must abort
}

// containsPtT reports whether check appears elsewhere in this pt-t's loop.
func (p *opPtT) containsPtT(check *opPtT) bool {
	for ptT := p.next; ptT != p; ptT = ptT.next {
		if ptT == check {
			return true
		}
	}
	return false
}

// containsSegT reports whether the loop contains a pt-t on segment at parameter t.
func (p *opPtT) containsSegT(segment *opSegment, t float64) bool {
	for ptT := p.next; ptT != p; ptT = ptT.next {
		if ptT.t == t && ptT.segment() == segment {
			return true
		}
	}
	return false
}

// containsSeg returns the pt-t in the loop belonging to segment, if any.
func (p *opPtT) containsSeg(check *opSegment) *opPtT {
	for ptT := p.next; ptT != p; ptT = ptT.next {
		if ptT.segment() == check && !ptT.deleted {
			return ptT
		}
	}
	return nil
}

// find returns the first non-deleted pt-t on segment (including this one).
func (p *opPtT) find(segment *opSegment) *opPtT {
	ptT := p
	stop := ptT
	for {
		if ptT.segment() == segment && !ptT.deleted {
			return ptT
		}
		ptT = ptT.next
		if stop == ptT {
			break
		}
	}
	return nil
}

// addOpp splices opp into this loop, moving opp's prior chain onto this pt-t's old next.
func (p *opPtT) addOpp(opp, oppPrev *opPtT) {
	oldNext := p.next
	p.next = opp
	oppPrev.next = oldNext
}

// insert splices span into the loop immediately after this pt-t.
func (p *opPtT) insert(span *opPtT) {
	span.next = p.next
	p.next = span
}

// oppPrev returns the pt-t immediately before opp in its loop, or nil if this pt-t is already in that loop.
func (p *opPtT) oppPrev(opp *opPtT) *opPtT {
	oppPrev := opp.next
	if oppPrev == p {
		return nil
	}
	for oppPrev.next != opp {
		oppPrev = oppPrev.next
		if oppPrev == p {
			return nil
		}
	}
	return oppPrev
}

// opPtTOverlaps reports whether the [s1, e1] and [s2, e2] t-ranges overlap, returning the pt-t pair bounding the
// overlap (ok is false when the ranges don't overlap or collapse to a single point).
func opPtTOverlaps(s1, e1, s2, e2 *opPtT) (sOut, eOut *opPtT, ok bool) {
	start1 := e1
	if s1.t < e1.t {
		start1 = s1
	}
	start2 := e2
	if s2.t < e2.t {
		start2 = s2
	}
	switch {
	case between(s1.t, start2.t, e1.t):
		sOut = start2
	case between(s2.t, start1.t, e2.t):
		sOut = start1
	}
	end1 := s1
	if s1.t < e1.t {
		end1 = e1
	}
	end2 := s2
	if s2.t < e2.t {
		end2 = e2
	}
	switch {
	case between(s1.t, end2.t, e1.t):
		eOut = end2
	case between(s2.t, end1.t, e2.t):
		eOut = end1
	}
	if sOut == eOut {
		return sOut, eOut, false
	}
	return sOut, eOut, sOut != nil && eOut != nil
}

// ptAlreadySeen reports whether check's point already appears earlier in the loop walk starting at p.
func (p *opPtT) ptAlreadySeen(check *opPtT) bool {
	for p != check {
		if p.pt == check.pt {
			return true
		}
		check = check.next
	}
	return false
}

// prev returns the pt-t whose next pointer is this one (the predecessor in the loop).
func (p *opPtT) prev() *opPtT {
	result := p
	next := p
	for {
		next = next.next
		if next == p {
			break
		}
		result = next
	}
	return result
}

// segment returns the segment owning this pt-t's span.
func (p *opPtT) segment() *opSegment { return p.span.segment }

// setCoincident flags this pt-t as pointed to by a coincident span.
func (p *opPtT) setCoincident() { p.coincident = true }

// setDeleted flags this pt-t as removed from active use.
func (p *opPtT) setDeleted() { p.deleted = true }

// starter returns whichever of this pt-t and end has the smaller t.
func (p *opPtT) starter(end *opPtT) *opPtT {
	if p.t < end.t {
		return p
	}
	return end
}

// opSpanBase is the shared base of a span (and of the terminal tail span at t==1). The up field is a Go artifice
// replacing an upcast to opSpan; it points to the enclosing opSpan, or nil for a tail base.
type opSpanBase struct {
	segment    *opSegment
	coinEnd    *opSpanBase // coincident spans that end here (may point to itself)
	fromAngleP *opAngle    // exposed via fromAngle()
	prev       *opSpan
	up         *opSpan // Go upcast back-pointer (nil for a tail base)
	ptT        opPtT   // the pt-t list associated with the start of this span
	spanAdds   int
	chased     bool
}

// initBase sets up the shared span-base fields, starting the pt-t and coincident-end loops as single-element loops.
func (b *opSpanBase) initBase(segment *opSegment, prev *opSpan, t float64, pt geom.Point) {
	b.segment = segment
	b.ptT.init(b, t, pt)
	b.coinEnd = b
	b.fromAngleP = nil
	b.prev = prev
	b.spanAdds = 0
	b.chased = false
}

func (b *opSpanBase) t() float64                  { return b.ptT.t }
func (b *opSpanBase) pt() geom.Point              { return b.ptT.pt }
func (b *opSpanBase) final() bool                 { return b.ptT.t == 1 }
func (b *opSpanBase) globalState() *opGlobalState { return b.segment.contour.state }
func (b *opSpanBase) bumpSpanAdds()               { b.spanAdds++ }
func (b *opSpanBase) spanAddsCount() int          { return b.spanAdds }
func (b *opSpanBase) setPrev(prev *opSpan)        { b.prev = prev }
func (b *opSpanBase) setFromAngle(angle *opAngle) { b.fromAngleP = angle }
func (b *opSpanBase) fromAngle() *opAngle         { return b.fromAngleP }
func (b *opSpanBase) setChased(chased bool)       { b.chased = chased }

// deleted reports whether the span's primary pt-t has been removed from active use.
func (b *opSpanBase) deleted() bool { return b.ptT.deleted }

// spanCollapsed is the result of testing whether a [s, e] t-range on a span's segment has collapsed onto a single point
// in the pt-t loop.
type spanCollapsed int

const (
	spanCollapsedNo spanCollapsed = iota
	spanCollapsedYes
	spanCollapsedError
)

// collapsed walks this span's pt-t loop, tracking the min/max t seen on the same segment; if the whole [s, e] range
// falls inside a single already-visited pt-t loop, it reports spanCollapsedYes.
func (b *opSpanBase) collapsed(s, e float64) spanCollapsed {
	start := &b.ptT
	var startNext *opPtT
	walk := start
	minT := walk.t
	maxT := minT
	segment := b.segment
	safetyNet := 100000
	for {
		walk = walk.next
		if walk == start {
			break
		}
		safetyNet--
		if safetyNet == 0 {
			return spanCollapsedError
		}
		if walk == startNext {
			return spanCollapsedError
		}
		if walk.segment() != segment {
			continue
		}
		minT = min(minT, walk.t)
		maxT = max(maxT, walk.t)
		if between(minT, s, maxT) && between(minT, e, maxT) {
			return spanCollapsedYes
		}
		startNext = start.next
	}
	return spanCollapsedNo
}

// ptTPtr returns the address of the span's primary pt-t record.
func (b *opSpanBase) ptTPtr() *opPtT { return &b.ptT }

// upCast returns the enclosing opSpan; valid only when !final().
func (b *opSpanBase) upCast() *opSpan { return b.up }

// upCastable returns the enclosing opSpan, or nil if this is the terminal (final) span.
func (b *opSpanBase) upCastable() *opSpan {
	if b.final() {
		return nil
	}
	return b.up
}

// simple reports whether the pt-t loop has exactly two entries (no other segment shares this point).
func (b *opSpanBase) simple() bool { return b.ptT.next.next == &b.ptT }

// step returns 1 if this span's t is less than end's, else -1 — the direction to walk from b toward end.
func (b *opSpanBase) step(end *opSpanBase) int {
	if b.t() < end.t() {
		return 1
	}
	return -1
}

// starter returns whichever of b and end has the smaller t, upcast to its enclosing opSpan.
func (b *opSpanBase) starter(end *opSpanBase) *opSpan {
	if b.t() < end.t() {
		return b.up
	}
	return end.up
}

// containsSpan reports whether span's pt-t appears in this span's pt-t loop.
func (b *opSpanBase) containsSpan(span *opSpanBase) bool {
	check := &span.ptT
	for walk := b.ptT.next; walk != &b.ptT; walk = walk.next {
		if walk == check {
			return true
		}
	}
	return false
}

// containsSegment returns the non-deleted pt-t in this span's loop that is the primary record of segment, if any.
func (b *opSpanBase) containsSegment(segment *opSegment) *opPtT {
	for walk := b.ptT.next; walk != &b.ptT; walk = walk.next {
		if walk.deleted {
			continue
		}
		if walk.segment() == segment && &walk.span.ptT == walk {
			return walk
		}
	}
	return nil
}

// containsCoinEndSpan reports whether coin appears in this span's coincident-end loop.
func (b *opSpanBase) containsCoinEndSpan(coin *opSpanBase) bool {
	for next := b.coinEnd; next != b; next = next.coinEnd {
		if next == coin {
			return true
		}
	}
	return false
}

// containsCoinEndSegment reports whether segment appears in this span's coincident-end loop.
func (b *opSpanBase) containsCoinEndSegment(segment *opSegment) bool {
	for next := b.coinEnd; next != b; next = next.coinEnd {
		if next.segment == segment {
			return true
		}
	}
	return false
}

// insertCoinEnd splices coin into this span's coincident-end loop, if it isn't already present.
func (b *opSpanBase) insertCoinEnd(coin *opSpanBase) {
	if b.containsCoinEndSpan(coin) {
		return
	}
	coin.coinEnd, b.coinEnd = b.coinEnd, coin.coinEnd
}

// addOpp splices opp's pt-t loop into this one, merging duplicate spans first. Returns false only when mergeMatches
// trips its safety hatch. Part of the intersection-time span surface consumed by moveNearby and the coincidence walker;
// addT-driven wiring uses the ptT-level addOpp directly.
func (b *opSpanBase) addOpp(opp *opSpanBase) bool {
	oppPrev := b.ptT.oppPrev(&opp.ptT)
	if oppPrev == nil {
		return true
	}
	if !b.mergeMatches(opp) {
		return false
	}
	b.ptT.addOpp(&opp.ptT, oppPrev)
	b.checkForCollapsedCoincidence()
	return true
}

// merge folds span's pt-t records into this span's loop, dropping duplicates. This does not compute the best t or pt;
// it merely moves all data into a single list. Part of the deferred intersection-time surface (consumed by opSegment's
// moveNearby in a later slice).
func (b *opSpanBase) merge(span *opSpan) {
	spanPtT := &span.ptT
	span.release(&b.ptT)
	if b.containsSpan(&span.opSpanBase) {
		return // merge is already in the ptT loop
	}
	remainder := spanPtT.next
	b.ptT.insert(spanPtT)
	for remainder != spanPtT {
		nextR := remainder.next
		found := false
		for compare := spanPtT.next; compare != spanPtT; {
			nextC := compare.next
			if nextC.span == remainder.span && nextC.t == remainder.t {
				found = true
				break
			}
			compare = nextC
		}
		if !found {
			spanPtT.insert(remainder)
		}
		remainder = nextR
	}
	b.spanAdds += span.spanAdds
}

// checkForCollapsedCoincidence handles the case where an insert put both ends of a coincident run in the same span: for
// each coincident pt-t in the loop it asks the coincidence tracker to mark it collapsed, then releases the deleted
// records.
func (b *opSpanBase) checkForCollapsedCoincidence() {
	// The tracker is nil for runs that never build one (fixWinding's global state, for one), matching the guard in
	// opSpan.release.
	coins := b.globalState().coincidence
	if coins == nil || coins.isEmpty() {
		return
	}
	head := &b.ptT
	for test := head; ; {
		if test.coincident {
			coins.markCollapsed(test)
		}
		test = test.next
		if test == head {
			break
		}
	}
	coins.releaseDeleted()
}

// mergeMatches handles the case where the pt-t loop references the same segment more than once and each is directly
// pointed to by a span in that segment: it merges them, keeping the points but removing the redundant spans so the
// segment has only one span per pt-t loop element. Returns false only on the safety-hatch trip.
func (b *opSpanBase) mergeMatches(opp *opSpanBase) bool {
	test := &b.ptT
	stop := test
	safetyHatch := 1000000
	for {
		safetyHatch--
		if safetyHatch == 0 {
			return false
		}
		testNext := test.next
		if !test.deleted {
			testBase := test.span
			segment := test.segment()
			if !segment.done() {
				inner := &opp.ptT
				innerStop := inner
				for {
					if inner.segment() == segment && !inner.deleted {
						innerBase := inner.span
						switch {
						case !zeroOrOne(inner.t):
							innerBase.upCast().release(test)
						case !zeroOrOne(test.t):
							testBase.upCast().release(inner)
						default:
							segment.markAllDone() // mark segment as collapsed
							test.setDeleted()
							inner.setDeleted()
						}
						break
					}
					inner = inner.next
					if inner == innerStop {
						break
					}
				}
			}
		}
		test = testNext
		if test == stop {
			break
		}
	}
	b.checkForCollapsedCoincidence()
	return true
}

// opSpan is a non-terminal span (t in [0, 1)) carrying winding/coincidence bookkeeping.
type opSpan struct {
	coincident *opSpan     // spans coincident with this one (may point to itself)
	toAngleP   *opAngle    // exposed via toAngle()
	next       *opSpanBase // next intersection point
	opSpanBase
	windSum      int
	oppSum       int
	windValue    int
	oppValue     int
	topTTry      int
	done         bool
	alreadyAdded bool
}

// init sets up a non-terminal span with default winding state (sums unset, wind value 1, opp value 0).
func (s *opSpan) init(segment *opSegment, prev *opSpan, t float64, pt geom.Point) {
	s.up = s
	s.initBase(segment, prev, t, pt)
	s.coincident = s
	s.toAngleP = nil
	s.windSum = skMinS32
	s.oppSum = skMinS32
	s.windValue = 1
	s.oppValue = 0
	s.topTTry = 0
	s.chased = false
	s.done = false
	segment.bumpCount()
	s.alreadyAdded = false
}

func (s *opSpan) setNext(next *opSpanBase) { s.next = next }
func (s *opSpan) setDone(done bool)        { s.done = done }
func (s *opSpan) markAdded()               { s.alreadyAdded = true }
func (s *opSpan) toAngle() *opAngle        { return s.toAngleP }

// setToAngle records the angle leading out of this span. Only the angle-loop build and sort (calcAngles/sortAngles in
// opsegment.go) call it; the coincidence machinery never assigns a toAngle.
func (s *opSpan) setToAngle(angle *opAngle) { s.toAngleP = angle }

// setWindSum assigns the winding sum, ignoring the assignment if the sum was already set to a different value. Such a
// conflict means the angle sort produced an inconsistent walk, but there is no abort path: the first value assigned
// wins and the run continues, so Op/Simplify can return an inconsistently wound result rather than reporting failure.
func (s *opSpan) setWindSum(windSum int) {
	if s.windSum != skMinS32 && s.windSum != windSum {
		return
	}
	s.windSum = windSum
}

// setOppSum assigns the opposite-operand winding sum, ignoring the assignment on a conflict for the same reason (and
// with the same consequence) as setWindSum.
func (s *opSpan) setOppSum(oppSum int) {
	if s.oppSum != skMinS32 && s.oppSum != oppSum {
		return
	}
	s.oppSum = oppSum
}

// isCanceled reports whether both the wind value and opposite-operand value are zero.
func (s *opSpan) isCanceled() bool { return s.windValue == 0 && s.oppValue == 0 }

// isCoincident reports whether this span has been linked into a coincident-span loop.
func (s *opSpan) isCoincident() bool { return s.coincident != s }

// containsCoincidenceSpan reports whether coin appears in this span's coincident loop.
func (s *opSpan) containsCoincidenceSpan(coin *opSpan) bool {
	for next := s.coincident; next != s; next = next.coincident {
		if next == coin {
			return true
		}
	}
	return false
}

// containsCoincidenceSeg reports whether segment appears in this span's coincident loop.
func (s *opSpan) containsCoincidenceSeg(segment *opSegment) bool {
	next := s.coincident
	for {
		if next.segment == segment {
			return true
		}
		next = next.coincident
		if next == s {
			return false
		}
	}
}

// setWindValue sets the wind value.
func (s *opSpan) setWindValue(windValue int) { s.windValue = windValue }

// setOppValue sets the opposite-operand wind value.
func (s *opSpan) setOppValue(oppValue int) { s.oppValue = oppValue }

// insertCoincidenceSpan splices coin into this span's coincident loop, if it isn't already present.
func (s *opSpan) insertCoincidenceSpan(coin *opSpan) {
	if s.containsCoincidenceSpan(coin) {
		return
	}
	coin.coincident, s.coincident = s.coincident, coin.coincident
}

// insertCoincidenceSeg finds the span on segment sharing this pt-t loop and splices it into the coincident loop.
// Returns false on a malformed list.
func (s *opSpan) insertCoincidenceSeg(segment *opSegment, flipped, ordered bool) bool {
	if s.containsCoincidenceSeg(segment) {
		return true
	}
	next := &s.ptT
	for {
		next = next.next
		if next == &s.ptT {
			break
		}
		if next.segment() != segment {
			continue
		}
		base := next.span
		var span *opSpan
		switch {
		case !ordered:
			spanEndPtT := s.next.containsSegment(segment)
			if spanEndPtT == nil {
				return false
			}
			spanEnd := spanEndPtT.span
			start := base.ptTPtr().starter(spanEnd.ptTPtr())
			if start.span.upCastable() == nil {
				return false
			}
			span = start.span.upCast()
		case flipped:
			span = base.prev
			if span == nil {
				return false
			}
		default:
			if base.upCastable() == nil {
				return false
			}
			span = base.upCast()
		}
		s.insertCoincidenceSpan(span)
		return true
	}
	return true
}

// release unlinks this span from the segment's list, fixes up any coincidence referencing its pt-t, marks the pt-t
// deleted, and repoints every pt-t alias that named this span at the kept span instead. Part of the intersection-time
// surface used by mergeMatches and merge.
func (s *opSpan) release(kept *opPtT) {
	prev := s.prev
	next := s.next
	prev.setNext(next)
	next.setPrev(prev)
	s.segment.release(s)
	if coincidence := s.globalState().coincidence; coincidence != nil {
		coincidence.fixUp(&s.ptT, kept)
	}
	s.ptT.setDeleted()
	stopPtT := &s.ptT
	keptSpan := kept.span
	for testPtT := stopPtT; ; {
		if &s.opSpanBase == testPtT.span {
			testPtT.span = keptSpan
		}
		testPtT = testPtT.next
		if testPtT == stopPtT {
			break
		}
	}
}
