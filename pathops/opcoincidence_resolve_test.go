// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the coincidence resolution machinery (opcoincidence_resolve.go) and its supporting span/segment methods:
// the targeted helpers at the unit level, plus end-to-end from real coincident paths through the handleCoincidence
// coincidence sub-sequence (addExpanded/correctEnds/addEndMovedSpans/addMissing/expand/ mark/apply/findOverlaps). The
// moveMultiples/moveNearby/calcAngles steps that handleCoincidence interleaves arrive with the walking-phase slice;
// these tests exercise the coincidence-only calls.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// TestOpSegmentExisting checks opSegment.existing: it returns the exact pt-t at a t, nil where no span exists, and the
// opp-aware form only matches when the span shares a pt-t loop with the opposite segment.
func TestOpSegmentExisting(t *testing.T) {
	head, state := newTestContourHead()
	segA := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	segB := head.addLine([]geom.Point{pt(5, -5), pt(5, 5)})
	segC := head.addLine([]geom.Point{pt(5, 20), pt(5, 30)})
	newOpCoincidence(state)
	aMid := segA.addT(0.5) // (5,0)
	bMid := segB.addT(0.5) // (5,0)
	aMid.span.addOpp(bMid.span)

	if got := segA.existing(0.5, nil); got != aMid {
		t.Fatalf("existing(0.5, nil) = %v, want the t=0.5 span", got)
	}
	if got := segA.existing(0.25, nil); got != nil {
		t.Fatalf("existing(0.25, nil) = %v, want nil (no span there)", got)
	}
	if got := segA.existing(0.5, segB); got != aMid {
		t.Fatal("existing(0.5, segB) should find the pt-t linked to segB")
	}
	if got := segA.existing(0.5, segC); got != nil {
		t.Fatal("existing(0.5, segC) should be nil (no shared loop with segC)")
	}
}

// TestOpSpanBaseCollapsed checks opSpanBase.collapsed: spanCollapsedNo on a fresh span, spanCollapsedYes when a t-range
// falls inside a single pt-t loop that spans it.
func TestOpSpanBaseCollapsed(t *testing.T) {
	head, _ := newTestContourHead()
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	if got := seg.head.collapsed(0.2, 0.8); got != spanCollapsedNo {
		t.Fatalf("fresh span collapsed = %v, want kNo", got)
	}
	a := seg.addT(0.25)
	b := seg.addT(0.75)
	a.span.merge(b.span.upCast()) // fold the t=0.75 pt-t into the t=0.25 span's loop
	if got := a.span.collapsed(0.4, 0.6); got != spanCollapsedYes {
		t.Fatalf("collapsed(0.4,0.6) over [0.25,0.75] loop = %v, want kYes", got)
	}
	if got := a.span.collapsed(0.4, 0.9); got != spanCollapsedNo {
		t.Fatalf("collapsed(0.4,0.9) exceeding [0.25,0.75] = %v, want kNo", got)
	}
}

// TestOpSegmentIsClose checks opSegment.isClose: the perpendicular at t hits a coincident opposite segment at the same
// point, but misses a far one.
func TestOpSegmentIsClose(t *testing.T) {
	head, _ := newTestContourHead()
	base := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	same := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	far := head.addLine([]geom.Point{pt(0, 100), pt(10, 100)})
	if !base.isClose(0.5, same) {
		t.Fatal("isClose should be true for a coincident opposite segment")
	}
	if base.isClose(0.5, far) {
		t.Fatal("isClose should be false for a far opposite segment")
	}
}

// TestOpSegmentIsXor checks opSegment.isXor reflects the contour's even-odd flag.
func TestOpSegmentIsXor(t *testing.T) {
	head, state := newTestContourHead()
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	if seg.isXor() {
		t.Fatal("default contour should be nonzero (isXor false)")
	}
	head.setXor(true)
	if !seg.isXor() {
		t.Fatal("isXor should follow the contour's xor flag")
	}
	_ = state
}

// TestOpSpanInsertCoincidence checks that opSpan.insertCoincidence splices a mutual link between two spans.
func TestOpSpanInsertCoincidence(t *testing.T) {
	_, segA, segB := coincidentPair(t)
	aHead := &segA.head
	bHead := &segB.head
	if aHead.isCoincident() {
		t.Fatal("span should not start coincident")
	}
	aHead.insertCoincidenceSpan(bHead)
	if !aHead.isCoincident() || !aHead.containsCoincidenceSpan(bHead) {
		t.Fatal("aHead should reference bHead after insert")
	}
	if !bHead.containsCoincidenceSpan(aHead) {
		t.Fatal("the coincidence link should be mutual")
	}
	aHead.insertCoincidenceSpan(bHead) // idempotent
	if !aHead.containsCoincidenceSpan(bHead) {
		t.Fatal("double insert should keep the link")
	}
}

// TestOpCoincidenceOverlap checks the shared t range computation.
func TestOpCoincidenceOverlap(t *testing.T) {
	head, state := newTestContourHead()
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	co := newOpCoincidence(state)
	a := seg.addT(0.2)
	b := seg.addT(0.8)
	// [head(0)..b(0.8)] overlapped with [a(0.2)..tail(1)] on the same segment -> [0.2, 0.8].
	overS, overE, ok := co.overlap(&seg.head.ptT, b, a, &seg.tail.ptT)
	if !ok || overS != 0.2 || overE != 0.8 {
		t.Fatalf("overlap = (%g, %g, %v), want (0.2, 0.8, true)", overS, overE, ok)
	}
	// Disjoint ranges [head..a] and [b..tail] -> empty.
	if _, _, ok = co.overlap(&seg.head.ptT, a, b, &seg.tail.ptT); ok {
		t.Fatal("disjoint ranges should report no overlap")
	}
}

// TestCoincidentSpansContainsAndOrdered checks the run-level contains and ordered helpers on a full-edge run.
func TestCoincidentSpansContainsAndOrdered(t *testing.T) {
	co, segA, segB := coincidentPair(t)
	co.add(&segA.head.ptT, &segA.tail.ptT, &segB.head.ptT, &segB.tail.ptT)
	run := co.head
	if !run.contains(&segA.head.ptT, &segA.tail.ptT) {
		t.Fatal("run should contain both of segA's endpoints")
	}
	res, ok := run.ordered()
	if !ok || !res {
		t.Fatalf("ordered() = (%v, %v), want (true, true) for a single-span run", res, ok)
	}
	if run.flipped() {
		t.Fatal("a forward (0..1 / 0..1) run should not be flipped")
	}
}

// TestCoincidentSpansExtend checks that extend grows a run's range when given wider ends.
func TestCoincidentSpansExtend(t *testing.T) {
	co, segA, segB := coincidentPair(t)
	aStart := segA.addT(0.25)
	aEnd := segA.addT(0.75)
	bStart := segB.addT(0.25)
	bEnd := segB.addT(0.75)
	co.add(aStart, aEnd, bStart, bEnd) // run over the inner [0.25, 0.75]
	run := co.head
	if !run.extend(&segA.head.ptT, &segA.tail.ptT, &segB.head.ptT, &segB.tail.ptT) {
		t.Fatal("extend to the full [0,1] range should report a change")
	}
	if run.coinPtTStart.t != 0 || run.coinPtTEnd.t != 1 {
		t.Fatalf("extended run range = [%g, %g], want [0, 1]", run.coinPtTStart.t, run.coinPtTEnd.t)
	}
	if run.extend(&segA.head.ptT, &segA.tail.ptT, &segB.head.ptT, &segB.tail.ptT) {
		t.Fatal("re-extending to the same range should report no change")
	}
}

// TestCoincidenceMarkAndApply drives correctEnds -> mark -> apply over two triangles that share their base edge
// (0,0)->(10,0). mark links the coincident base spans; apply transfers the winding so the opposite base span drops to
// zero winding and is marked done, while the coin base span keeps winding 1 / opp 1.
func TestCoincidenceMarkAndApply(t *testing.T) {
	above := path.New()
	above.MoveTo(0, 0).LineTo(10, 0).LineTo(5, 5).Close()
	below := path.New()
	below.MoveTo(0, 0).LineTo(10, 0).LineTo(5, -5).Close()
	head, state := buildOpModel(t, above, below)
	co := newOpCoincidence(state)
	cs := realContours(head)
	addIntersectTs(cs[0], cs[1], co)

	run := co.head
	if run == nil {
		t.Fatal("expected one coincident run for the shared base edge")
	}
	coinSpan := run.coinPtTStart.span.upCast()
	oppSpan := run.oppPtTStart.span.upCast()

	co.correctEnds()
	if !co.mark() {
		t.Fatal("mark() returned false")
	}
	if !coinSpan.isCoincident() || !coinSpan.containsCoincidenceSpan(oppSpan) {
		t.Fatal("mark should link the coin base span to the opp base span")
	}

	if !co.apply() {
		t.Fatal("apply() returned false")
	}
	if !oppSpan.done {
		t.Fatal("apply should mark the opposite base span done")
	}
	if oppSpan.windValue != 0 || oppSpan.oppValue != 0 {
		t.Fatalf("opp base span winding = (%d, %d), want (0, 0)", oppSpan.windValue, oppSpan.oppValue)
	}
	if coinSpan.windValue != 1 || coinSpan.oppValue != 1 {
		t.Fatalf("coin base span winding = (%d, %d), want (1, 1)", coinSpan.windValue, coinSpan.oppValue)
	}
}

// TestCoincidenceHandleSequenceIdenticalRects runs the full HandleCoincidence coincidence sub-sequence over two
// identical rectangles (every edge coincides). The sequence must complete and leave every span with a valid
// (non-negative) winding value.
func TestCoincidenceHandleSequenceIdenticalRects(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()
	b := path.New()
	b.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()
	head, state := buildOpModel(t, a, b)
	co := newOpCoincidence(state)
	addIntersections(head, co)

	if coincidenceCount(co) == 0 {
		t.Fatal("identical rectangles should record coincident runs")
	}
	runCoincidenceResolution(t, co)

	for _, s := range allSpans(head) {
		if s.windValue < 0 {
			t.Fatalf("span at %v has negative windValue %d after apply", s.pt(), s.windValue)
		}
	}
}

// runCoincidenceResolution executes the coincidence-only portion of handleCoincidence, failing the test if any step
// aborts. It skips the interleaved moveMultiples/moveNearby/missingCoincidence steps (walking-phase slice, not yet
// implemented here).
func runCoincidenceResolution(t *testing.T, co *opCoincidence) {
	t.Helper()
	if !co.addExpanded() {
		t.Fatal("addExpanded (1) returned false")
	}
	co.correctEnds()
	if !co.addEndMovedSpans() {
		t.Fatal("addEndMovedSpans returned false")
	}
	for safety := 3; ; safety-- {
		var added bool
		if !co.addMissing(&added) {
			t.Fatal("addMissing returned false")
		}
		if !added {
			break
		}
		if safety == 1 {
			break
		}
	}
	if co.expand() {
		var added bool
		if !co.addMissing(&added) {
			t.Fatal("addMissing (post-expand) returned false")
		}
		if !co.addExpanded() {
			t.Fatal("addExpanded (post-expand) returned false")
		}
	}
	if !co.addExpanded() {
		t.Fatal("addExpanded (2) returned false")
	}
	if !co.mark() {
		t.Fatal("mark returned false")
	}
	co.expand()
	overlaps := &opCoincidence{globalState: co.globalState}
	for safety := 3; safety > 0; safety-- {
		pairs := co
		if !overlaps.isEmpty() {
			pairs = overlaps
		}
		if !pairs.apply() {
			t.Fatal("apply returned false")
		}
		if !pairs.findOverlaps(overlaps) {
			t.Fatal("findOverlaps returned false")
		}
		if overlaps.isEmpty() {
			break
		}
	}
}

// TestAddOrOverlapEndOutOfRange drives addOrOverlap down the ce/oe branch with a coinTe that lies outside [0,1], so
// coinSeg.addT(coinTe) returns nil. The segment pair is a short edge (segA, t in [0,1] over x in [0,10]) coincident with
// a longer collinear one (segB, x in [-5,15]): the interior samples coincidentRunIsReal takes at 0.25/0.5/0.75 of
// [0.5, 1.2] still land on segB, so the run is believed, yet coinTe=1.2 extrapolates past segA's tail and addT yields
// nil. The mirror nil check must make addOrOverlap return false rather than nil-panic dereferencing ceWritable.span.
func TestAddOrOverlapEndOutOfRange(t *testing.T) {
	head, state := newTestContourHead()
	segA := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	segB := head.addLine([]geom.Point{pt(-5, 0), pt(15, 0)})
	co := newOpCoincidence(state)

	// Populate co.top with an unrelated run so the required non-nil precondition holds; checkOverlap skips it because
	// its segments differ from segA/segB.
	segC := head.addLine([]geom.Point{pt(0, 100), pt(10, 100)})
	segD := head.addLine([]geom.Point{pt(0, 100), pt(10, 100)})
	co.add(&segC.head.ptT, &segC.tail.ptT, &segD.head.ptT, &segD.tail.ptT)
	co.top = co.head
	co.head = nil

	// coinTe=1.2 on segA extrapolates to (12,0), which addT can't place (past the tail) -> nil; oppTe=0.85 on segB maps
	// to the same (12,0) and does resolve. Interior samples of [0.5,1.2] stay within segB, so the run is real.
	added := false
	if got := co.addOrOverlap(segA, segB, 0.5, 1.2, 0.5, 0.85, &added); got {
		t.Fatalf("addOrOverlap with an out-of-range coinTe = %v, want false", got)
	}
	if added {
		t.Fatal("addOrOverlap should not report an addition when the end span can't be placed")
	}
}

// allSpans returns every non-terminal span across the model's real contours.
func allSpans(head *opContourHead) []*opSpan {
	var out []*opSpan
	for _, c := range realContours(head) {
		for _, s := range segsOf(c) {
			for base := &s.head.opSpanBase; base != &s.tail; base = base.upCast().next {
				out = append(out, base.upCast())
			}
		}
	}
	return out
}

// TestAddIfMissingSwallowsAddOrOverlapFailure pins the deliberate discard in addIfMissing. addOrOverlap returns false
// for "nothing to add" as well as for genuine trouble, and its two callers read that oppositely: addEndMovedSpans
// aborts on false, addIfMissing must not, because addMissing turns any false from addIfMissing into a fatal abort of
// the whole Op/Simplify. Upstream discards the result the same way (`(void) this->addOrOverlap(...)` in
// SkOpCoincidence::addIfMissing), so propagating it here would fail operations Skia completes.
//
// The setup wires an "over" segment to two collinear segments through shared pt-t loops so trange maps a real range
// onto each, then leaves co.top nil -- addOrOverlap's first check, and the one place a false cannot mean anything but
// trouble. addIfMissing must still report success.
func TestAddIfMissingSwallowsAddOrOverlapFailure(t *testing.T) {
	head, state := newTestContourHead()
	segO := head.addLine([]geom.Point{pt(0, 0), pt(20, 0)})
	segA := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	segB := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})

	// addOpp reaches back into the global state's coincidence, so it has to exist before the wiring below.
	co := newOpCoincidence(state)
	if co.top != nil {
		t.Fatal("a fresh opCoincidence should have a nil top")
	}

	// Join segO's t=0 and t=0.5 spans (points (0,0) and (10,0)) to both collinear segments' endpoints, so trange can
	// map a t on segO onto segA and segB.
	oMid := segO.addT(0.5)
	if oMid == nil {
		t.Fatal("addT(0.5) on the over segment returned nil")
	}
	for _, seg := range []*opSegment{segA, segB} {
		if !segO.head.addOpp(&seg.head.opSpanBase) {
			t.Fatal("addOpp for the run's start returned false")
		}
		if !oMid.span.addOpp(&seg.tail) {
			t.Fatal("addOpp for the run's end returned false")
		}
	}

	// tStart=0.25 and tEnd=0.4 on segO map to 0.5 and 0.8 on segA and segB. Confirm addOrOverlap really does decline
	// this call, so the assertion below is about addIfMissing swallowing a false rather than never seeing one.
	added := false
	if co.addOrOverlap(segA, segB, 0.5, 0.8, 0.5, 0.8, &added) {
		t.Fatal("addOrOverlap with a nil top should return false; the test no longer covers what it claims")
	}

	added = false
	if !co.addIfMissing(&segO.head.ptT, &segO.head.ptT, 0.25, 0.4, segA, segB, &added) {
		t.Fatal("addIfMissing must return true when addOrOverlap declines; a false aborts the entire operation")
	}
	if added {
		t.Fatal("addIfMissing should not report an addition when addOrOverlap declined")
	}
}

// TestAddMissingSurvivesDecliningAddIfMissing is the caller-side half: addMissing must run to completion over a model
// whose coincident runs give addIfMissing nothing to add. A false from addIfMissing would abort at the first such run.
func TestAddMissingSurvivesDecliningAddIfMissing(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()
	b := path.New()
	b.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()
	head, state := buildOpModel(t, a, b)
	co := newOpCoincidence(state)
	addIntersections(head, co)
	if coincidenceCount(co) == 0 {
		t.Fatal("identical rectangles should record coincident runs")
	}

	var added bool
	if !co.addMissing(&added) {
		t.Fatal("addMissing aborted")
	}
}
