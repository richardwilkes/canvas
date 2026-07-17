// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-segment "walking" phase: the methods the bridgeOp/bridgeWinding/bridgeXor output-tracing drivers use to walk
// the resolved segment graph, decide which edge is active for the pending operation, and emit it. Covered here: the
// active-edge decision tables (kUnaryActiveEdge/kActiveEdge), activeAngle(+Inner/Other), activeOp/activeWinding (both
// overloads: activeOp/activeOpWindings and activeWinding/activeWindingSum), setUpWinding (the unary singular form),
// addCurveTo (emit an edge into the path writer), findNextOp/findNextWinding/findNextXor, markAndChaseDone, undoneSpan,
// joinEnds, isSimple, and doneAngle. The winding/coincidence/angle machinery these consume lives in winding.go,
// opangle.go, opcoincidence*.go, and opsegment_move.go. The chase worklist is a *[]*opSpanBase; the active-edge lookup
// uses the PathOp enum directly.

package pathops

import "github.com/richardwilkes/canvas/path"

// kUnaryActiveEdge[from][to] reports whether a simplify (winding) edge is active given whether the max and sum windings
// are non-zero.
var kUnaryActiveEdge = [2][2]bool{
	// from=0 from=1
	{false, true}, {true, false},
}

// kActiveEdge[op][miFrom][miTo][suFrom][suTo] reports whether a binary-op edge is active given the minuend and
// subtrahend windings' from/to non-zero state. Only the first four ops (difference/intersect/union/xor) appear;
// reverse-difference is remapped to difference before the walk.
var kActiveEdge = [4][2][2][2][2]bool{
	// difference (mi - su)
	{
		{{{false, false}, {false, false}}, {{true, false}, {true, false}}},
		{{{true, true}, {false, false}}, {{false, true}, {true, false}}},
	},
	// intersect (mi & su)
	{
		{{{false, false}, {false, false}}, {{false, true}, {false, true}}},
		{{{false, false}, {true, true}}, {{false, true}, {true, false}}},
	},
	// union (mi | su)
	{
		{{{false, true}, {true, false}}, {{true, true}, {false, false}}},
		{{{true, false}, {true, false}}, {{false, false}, {false, false}}},
	},
	// xor (mi ^ su)
	{
		{{{false, true}, {true, false}}, {{true, false}, {false, true}}},
		{{{true, false}, {false, true}}, {{false, true}, {true, false}}},
	},
}

// activeAngle returns the active angle sweeping from start, first checking this segment's own spans, then the opposite
// segment sharing start's pt-t.
func (s *opSegment) activeAngle(start *opSpanBase, startPtr, endPtr **opSpanBase, done *bool) *opAngle {
	if result := s.activeAngleInner(start, startPtr, endPtr, done); result != nil {
		return result
	}
	return s.activeAngleOther(start, startPtr, endPtr, done)
}

// activeAngleInner looks at the span leaving start (upSpan) and the span entering it (downSpan); it returns the
// corresponding angle when the span is active and windSum is resolved, clearing *done when an active span still lacks
// its windSum.
func (s *opSegment) activeAngleInner(start *opSpanBase, startPtr, endPtr **opSpanBase, done *bool) *opAngle {
	if upSpan := start.upCastable(); upSpan != nil {
		if upSpan.windValue != 0 || upSpan.oppValue != 0 {
			next := upSpan.next
			if *endPtr == nil {
				*startPtr = start
				*endPtr = next
			}
			if !upSpan.done {
				if upSpan.windSum != skMinS32 {
					return s.spanToAngle(start, next)
				}
				*done = false
			}
		}
	}
	// edge leading into junction
	if downSpan := start.prev; downSpan != nil {
		if downSpan.windValue != 0 || downSpan.oppValue != 0 {
			if *endPtr == nil {
				*startPtr = start
				*endPtr = &downSpan.opSpanBase
			}
			if !downSpan.done {
				if downSpan.windSum != skMinS32 {
					return s.spanToAngle(start, &downSpan.opSpanBase)
				}
				*done = false
			}
		}
	}
	return nil
}

// activeAngleOther hops to the opposite segment sharing start's pt-t and tries activeAngleInner there.
func (s *opSegment) activeAngleOther(start *opSpanBase, startPtr, endPtr **opSpanBase, done *bool) *opAngle {
	oPtT := start.ptT.next
	other := oPtT.segment()
	oSpan := oPtT.span
	return other.activeAngleInner(oSpan, startPtr, endPtr, done)
}

// activeOp seeds the minuend/subtrahend windings from the running sums and defers to activeOpWindings.
func (s *opSegment) activeOp(start, end *opSpanBase, xorMiMask, xorSuMask int, op PathOp) bool {
	sumMiWinding := s.updateWinding(end, start)
	sumSuWinding := s.updateOppWinding(end, start)
	if s.operand() {
		sumMiWinding, sumSuWinding = sumSuWinding, sumMiWinding
	}
	return s.activeOpWindings(xorMiMask, xorSuMask, start, end, op, &sumMiWinding, &sumSuWinding)
}

// activeOpWindings derives the four in/out edge flags and looks them up in kActiveEdge.
func (s *opSegment) activeOpWindings(xorMiMask, xorSuMask int, start, end *opSpanBase, op PathOp, sumMiWinding, sumSuWinding *int) bool {
	var maxWinding, sumWinding, oppMaxWinding, oppSumWinding int
	s.setUpWindingsBinary(start, end, sumMiWinding, sumSuWinding, &maxWinding, &sumWinding, &oppMaxWinding,
		&oppSumWinding)
	var miFrom, miTo, suFrom, suTo bool
	if s.operand() {
		miFrom = oppMaxWinding&xorMiMask != 0
		miTo = oppSumWinding&xorMiMask != 0
		suFrom = maxWinding&xorSuMask != 0
		suTo = sumWinding&xorSuMask != 0
	} else {
		miFrom = maxWinding&xorMiMask != 0
		miTo = sumWinding&xorMiMask != 0
		suFrom = oppMaxWinding&xorSuMask != 0
		suTo = oppSumWinding&xorSuMask != 0
	}
	return kActiveEdge[op][b2i(miFrom)][b2i(miTo)][b2i(suFrom)][b2i(suTo)]
}

// activeWinding computes the running sum winding for the start/end span and defers to activeWindingSum.
func (s *opSegment) activeWinding(start, end *opSpanBase) bool {
	sumWinding := s.updateWinding(end, start)
	return s.activeWindingSum(start, end, &sumWinding)
}

// activeWindingSum reports whether the unary (simplify) edge from start to end is active, given the caller's running
// sum winding.
func (s *opSegment) activeWindingSum(start, end *opSpanBase, sumWinding *int) bool {
	var maxWinding int
	s.setUpWinding(start, end, &maxWinding, sumWinding)
	from := maxWinding != 0
	to := *sumWinding != 0
	return kUnaryActiveEdge[b2i(from)][b2i(to)]
}

// setUpWinding computes the unary singular winding setup: the max winding before the span and the sum winding after it.
func (s *opSegment) setUpWinding(start, end *opSpanBase, maxWinding, sumWinding *int) {
	deltaSum := spanSign(start, end)
	*maxWinding = *sumWinding
	if *sumWinding == skMinS32 {
		return
	}
	*sumWinding -= deltaSum
}

// addCurveTo emits the edge from start to end into the writer, marking the starter span added. Returns false on a
// duplicate add or a degenerate deferred line.
func (s *opSegment) addCurveTo(start, end *opSpanBase, writer *pathWriter) bool {
	spanStart := start.starter(end)
	if spanStart.alreadyAdded {
		return false
	}
	spanStart.markAdded()
	var curvePart dCurveSweep
	start.segment.subDivide(start, end, &curvePart.curve)
	curvePart.setCurveHullSweep(s.verb)
	verb := s.verb
	if !curvePart.isCurve() {
		verb = path.VerbLine
	}
	writer.deferredMove(&start.ptT)
	switch verb {
	case path.VerbLine:
		if !writer.deferredLine(&end.ptT) {
			return false
		}
	case path.VerbQuad:
		writer.quadTo(curvePart.curve.pts[1].asPoint(), &end.ptT)
	case path.VerbConic:
		writer.conicTo(curvePart.curve.pts[1].asPoint(), &end.ptT, curvePart.curve.weight)
	case path.VerbCubic:
		writer.cubicTo(curvePart.curve.pts[1].asPoint(), curvePart.curve.pts[2].asPoint(), &end.ptT)
	}
	return true
}

// findNextOp advances the trace to the next active edge for a boolean op, marking passed-over edges done and appending
// their chase spans. Advances *nextStart/*nextEnd/*simple.
func (s *opSegment) findNextOp(chase *[]*opSpanBase, nextStart, nextEnd **opSpanBase, unsortable, simple *bool, op PathOp, xorMiMask, xorSuMask int) *opSegment {
	start := *nextStart
	end := *nextEnd
	step := start.step(end)
	other := s.isSimple(nextStart, &step) // advances nextStart
	*simple = other != nil
	if other != nil {
		// mark the smaller of startIndex, endIndex done, and all adjacent spans with the same T value
		startSpan := start.starter(end)
		if startSpan.done {
			return nil
		}
		s.markDone(startSpan)
		if step > 0 {
			*nextEnd = (*nextStart).upCast().next
		} else {
			*nextEnd = &(*nextStart).prev.opSpanBase
		}
		return other
	}
	var endNear *opSpanBase
	if step > 0 {
		endNear = (*nextStart).upCast().next
	} else {
		endNear = &(*nextStart).prev.opSpanBase
	}
	// more than one viable candidate -- measure angles to find best
	calcWinding := s.computeSum(start, endNear, includeBinaryOpp)
	if calcWinding == skNaN32 {
		*unsortable = true
		s.markDone(start.starter(end))
		return nil
	}
	angle := s.spanToAngle(end, start)
	if angle.unorderable {
		*unsortable = true
		s.markDone(start.starter(end))
		return nil
	}
	sumMiWinding := s.updateWinding(end, start)
	if sumMiWinding == skMinS32 {
		*unsortable = true
		s.markDone(start.starter(end))
		return nil
	}
	sumSuWinding := s.updateOppWinding(end, start)
	if s.operand() {
		sumMiWinding, sumSuWinding = sumSuWinding, sumMiWinding
	}
	nextAngle := angle.next
	var foundAngle *opAngle
	foundDone := false
	var nextSegment *opSegment
	activeCount := 0
	for {
		nextSegment = nextAngle.segment()
		activeAngle := nextSegment.activeOpWindings(xorMiMask, xorSuMask, nextAngle.start, nextAngle.end, op,
			&sumMiWinding, &sumSuWinding)
		if activeAngle {
			activeCount++
			if foundAngle == nil || (foundDone && activeCount&1 != 0) {
				foundAngle = nextAngle
				foundDone = nextSegment.doneAngle(nextAngle)
			}
		}
		if !nextSegment.done() {
			if !activeAngle {
				nextSegment.markAndChaseDone(nextAngle.start, nextAngle.end, nil)
			}
			if last := nextAngle.lastMarked; last != nil {
				*chase = append(*chase, last)
			}
		}
		nextAngle = nextAngle.next
		if nextAngle == angle {
			break
		}
	}
	start.segment.markDone(start.starter(end))
	if foundAngle == nil {
		return nil
	}
	*nextStart = foundAngle.start
	*nextEnd = foundAngle.end
	return foundAngle.segment()
}

// findNextWinding is the simplify (unary winding) counterpart of findNextOp.
func (s *opSegment) findNextWinding(chase *[]*opSpanBase, nextStart, nextEnd **opSpanBase, unsortable *bool) *opSegment {
	start := *nextStart
	end := *nextEnd
	step := start.step(end)
	other := s.isSimple(nextStart, &step) // advances nextStart
	if other != nil {
		startSpan := start.starter(end)
		if startSpan.done {
			return nil
		}
		s.markDone(startSpan)
		if step > 0 {
			*nextEnd = (*nextStart).upCast().next
		} else {
			*nextEnd = &(*nextStart).prev.opSpanBase
		}
		return other
	}
	var endNear *opSpanBase
	if step > 0 {
		endNear = (*nextStart).upCast().next
	} else {
		endNear = &(*nextStart).prev.opSpanBase
	}
	calcWinding := s.computeSum(start, endNear, includeUnaryWinding)
	if calcWinding == skNaN32 {
		*unsortable = true
		s.markDone(start.starter(end))
		return nil
	}
	angle := s.spanToAngle(end, start)
	if angle.unorderable {
		*unsortable = true
		s.markDone(start.starter(end))
		return nil
	}
	sumWinding := s.updateWinding(end, start)
	nextAngle := angle.next
	var foundAngle *opAngle
	foundDone := false
	var nextSegment *opSegment
	activeCount := 0
	for {
		nextSegment = nextAngle.segment()
		activeAngle := nextSegment.activeWindingSum(nextAngle.start, nextAngle.end, &sumWinding)
		if activeAngle {
			activeCount++
			if foundAngle == nil || (foundDone && activeCount&1 != 0) {
				foundAngle = nextAngle
				foundDone = nextSegment.doneAngle(nextAngle)
			}
		}
		if !nextSegment.done() {
			if !activeAngle {
				nextSegment.markAndChaseDone(nextAngle.start, nextAngle.end, nil)
			}
			if last := nextAngle.lastMarked; last != nil {
				*chase = append(*chase, last)
			}
		}
		nextAngle = nextAngle.next
		if nextAngle == angle {
			break
		}
	}
	start.segment.markDone(start.starter(end))
	if foundAngle == nil {
		return nil
	}
	*nextStart = foundAngle.start
	*nextEnd = foundAngle.end
	return foundAngle.segment()
}

// findNextXor is the even-odd (xor) counterpart of findNextOp; no winding computation, just the angle order.
func (s *opSegment) findNextXor(nextStart, nextEnd **opSpanBase, unsortable *bool) *opSegment {
	start := *nextStart
	end := *nextEnd
	step := start.step(end)
	other := s.isSimple(nextStart, &step) // advances nextStart
	if other != nil {
		startSpan := start.starter(end)
		if startSpan.done {
			return nil
		}
		s.markDone(startSpan)
		if step > 0 {
			*nextEnd = (*nextStart).upCast().next
		} else {
			*nextEnd = &(*nextStart).prev.opSpanBase
		}
		return other
	}
	angle := s.spanToAngle(end, start)
	if angle == nil || angle.unorderable {
		*unsortable = true
		s.markDone(start.starter(end))
		return nil
	}
	nextAngle := angle.next
	var foundAngle *opAngle
	foundDone := false
	var nextSegment *opSegment
	activeCount := 0
	for {
		if nextAngle == nil {
			return nil
		}
		nextSegment = nextAngle.segment()
		activeCount++
		if foundAngle == nil || (foundDone && activeCount&1 != 0) {
			foundAngle = nextAngle
			foundDone = nextSegment.doneAngle(nextAngle)
			if !foundDone {
				break
			}
		}
		nextAngle = nextAngle.next
		if nextAngle == angle {
			break
		}
	}
	start.segment.markDone(start.starter(end))
	if foundAngle == nil {
		return nil
	}
	*nextStart = foundAngle.start
	*nextEnd = foundAngle.end
	return foundAngle.segment()
}

// markAndChaseDone marks the starter span done and chases the aliased-span run marking each done, stopping at an
// already-done or already-visited span. When found is non-nil it receives the last chased span (nil if the run looped).
func (s *opSegment) markAndChaseDone(start, end *opSpanBase, found **opSpanBase) bool {
	step := start.step(end)
	minSpan := start.starter(end)
	s.markDone(minSpan)
	var last *opSpanBase
	other := s
	var priorDone, lastDone *opSpan
	safetyNet := 1000
	for {
		other = other.nextChase(&start, &step, &minSpan, &last)
		if other == nil {
			break
		}
		safetyNet--
		if safetyNet == 0 {
			return false
		}
		if other.done() {
			break
		}
		if lastDone == minSpan || priorDone == minSpan {
			if found != nil {
				*found = nil
			}
			return true
		}
		other.markDone(minSpan)
		priorDone = lastDone
		lastDone = minSpan
	}
	if found != nil {
		*found = last
	}
	return true
}

// undoneSpan returns the first span still needing processing, or nil.
func (s *opSegment) undoneSpan() *opSpan {
	span := &s.head
	for {
		next := span.next
		if !span.done {
			return span
		}
		if next.final() {
			return nil
		}
		span = next.upCast()
	}
}

// joinEnds links this segment's tail pt-t to start's head pt-t.
func (s *opSegment) joinEnds(start *opSegment) {
	s.tail.ptT.addOpp(&start.head.ptT, &start.head.ptT)
}

// isSimple is nextChase with no min/last tracking.
func (s *opSegment) isSimple(end **opSpanBase, step *int) *opSegment {
	return s.nextChase(end, step, nil, nil)
}

// doneAngle reports whether the angle's starter span is done.
func (s *opSegment) doneAngle(angle *opAngle) bool {
	return angle.start.starter(angle.end).done
}
