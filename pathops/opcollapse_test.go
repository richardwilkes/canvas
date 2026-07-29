// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Direct unit tests for the span/segment/coincidence collapse surface (opSpan.release,
// opSegment.markAllDone/markDone/release, opCoincidence.release/fixUp). These are reachable from addIntersectTs via
// mergeMatches/checkForCollapsedCoincidence but only fire on pathological concurrent geometry that the op-driver test
// corpus exercises; the tests here drive them directly at the unit level.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// TestOpSegmentMarkAllDone builds a segment with two interior spans and verifies the done accounting.
func TestOpSegmentMarkAllDone(t *testing.T) {
	head, _ := newTestContourHead()
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	seg.addT(0.25)
	seg.addT(0.5)
	if seg.count != 3 {
		t.Fatalf("span count = %d, want 3 (head + two interior)", seg.count)
	}
	if seg.done() {
		t.Fatal("segment should not be done before marking")
	}
	seg.markAllDone()
	if !seg.done() || seg.doneCount != 3 {
		t.Fatalf("after markAllDone: done=%v doneCount=%d, want true/3", seg.done(), seg.doneCount)
	}
	// markDone on an already-done span is a no-op (no double count).
	seg.markDone(&seg.head)
	if seg.doneCount != 3 {
		t.Fatalf("markDone on a done span changed doneCount to %d", seg.doneCount)
	}
}

// TestCheckForCollapsedCoincidenceWithoutTracker covers a global state that never built a coincidence tracker —
// fixWinding constructs one that way — so the field is nil for the whole run. checkForCollapsedCoincidence must fall
// out the way opSpan.release does rather than dereferencing it.
func TestCheckForCollapsedCoincidenceWithoutTracker(t *testing.T) {
	head, state := newTestContourHead()
	if state.coincidence != nil {
		t.Fatal("precondition: a global state with no newOpCoincidence call should have a nil tracker")
	}
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	mid := seg.addT(0.5)
	mid.coincident = true // the loop body would ask the nil tracker to mark this collapsed
	if seg.count != 2 {
		t.Fatalf("span count = %d, want 2 (head + one interior)", seg.count)
	}
	seg.head.checkForCollapsedCoincidence()
	if seg.count != 2 {
		t.Fatalf("span count = %d, want 2 (the call should have changed nothing)", seg.count)
	}
	// opSpan.release reads the same field and already guards it; the two readers must agree about it being nil.
	mid.span.upCast().release(&seg.head.ptT)
	if seg.count != 1 {
		t.Fatalf("span count after release = %d, want 1", seg.count)
	}
}

// TestOpSpanRelease releases an interior span and checks the list is re-linked, the counts drop, and every pt-t that
// named the released span is repointed at the kept span.
func TestOpSpanRelease(t *testing.T) {
	head, _ := newTestContourHead()
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	a := seg.addT(0.25) // interior span at 0.25
	b := seg.addT(0.5)  // interior span at 0.5
	if seg.count != 3 {
		t.Fatalf("span count = %d, want 3", seg.count)
	}
	aSpan := a.span.upCast()
	aSpan.release(b) // keep b's pt-t
	if seg.count != 2 {
		t.Fatalf("count after release = %d, want 2", seg.count)
	}
	// The 0.25 span is unlinked: head now points straight at the 0.5 span, and back.
	if seg.head.next != b.span {
		t.Fatal("head.next should be the 0.5 span after releasing 0.25")
	}
	if b.span.prev != &seg.head {
		t.Fatal("0.5 span.prev should be head after releasing 0.25")
	}
	if !a.deleted {
		t.Fatal("released span's pt-t should be marked deleted")
	}
	// a named the released span; release repoints it at the kept span (b's).
	if a.span != b.span {
		t.Fatal("released pt-t should be repointed at the kept span")
	}
	if ts := spanTs(seg); len(ts) != 3 { // head, 0.5, tail
		t.Fatalf("remaining spans = %v, want head/0.5/tail", ts)
	}
}

// TestOpCoincidenceReleaseAndFixUp records a coincident run over two identical segments and exercises the release
// (unlink) and both fixUp branches (repoint vs. release).
func TestOpCoincidenceReleaseAndFixUp(t *testing.T) {
	// release: a recorded run can be unlinked, emptying the list.
	t.Run("release", func(t *testing.T) {
		co, segA, segB := coincidentPair(t)
		co.add(&segA.head.ptT, &segA.tail.ptT, &segB.head.ptT, &segB.tail.ptT)
		if co.isEmpty() {
			t.Fatal("add should have recorded a run")
		}
		run := co.head
		if !co.release(co.head, run) {
			t.Fatal("release should find and remove the run")
		}
		if !co.isEmpty() {
			t.Fatal("list should be empty after releasing its only run")
		}
	})
	// fixUp repoint: deleting the coin start with a kept span distinct from the coin end repoints the start.
	t.Run("fixup-repoint", func(t *testing.T) {
		co, segA, segB := coincidentPair(t)
		co.add(&segA.head.ptT, &segA.tail.ptT, &segB.head.ptT, &segB.tail.ptT)
		run := co.head
		co.fixUp(&segA.head.ptT, &segB.tail.ptT) // kept span (segB.tail) != coin end (segA.tail)
		if co.isEmpty() {
			t.Fatal("fixUp should have repointed, not released, the run")
		}
		if run.coinPtTStart != &segB.tail.ptT {
			t.Fatal("coin start should be repointed at the kept pt-t")
		}
	})
	// fixUp release: deleting the coin start when the kept span equals the coin end drops the run.
	t.Run("fixup-release", func(t *testing.T) {
		co, segA, segB := coincidentPair(t)
		co.add(&segA.head.ptT, &segA.tail.ptT, &segB.head.ptT, &segB.tail.ptT)
		co.fixUp(&segA.head.ptT, &segA.tail.ptT) // kept span == coin end -> release
		if !co.isEmpty() {
			t.Fatal("fixUp should release the run when the kept span is its other end")
		}
	})
}

// TestOpSpanBaseAddOpp exercises the span-level addOpp: two crossing segments' coincident-point spans are linked into a
// shared pt-t loop. (AddIntersectTs uses the pt-t-level addOpp; this covers the span form its moveNearby consumer will
// use.)
func TestOpSpanBaseAddOpp(t *testing.T) {
	head, state := newTestContourHead()
	segA := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	segB := head.addLine([]geom.Point{pt(5, -5), pt(5, 5)})
	newOpCoincidence(state) // addOpp -> checkForCollapsedCoincidence needs a tracker present
	aMid := segA.addT(0.5)  // (5,0)
	bMid := segB.addT(0.5)  // (5,0)
	if aMid.containsSeg(segB) != nil {
		t.Fatal("spans should start unlinked")
	}
	if !aMid.span.addOpp(bMid.span) {
		t.Fatal("addOpp returned false")
	}
	if got := aMid.containsSeg(segB); got != bMid {
		t.Fatal("segA's crossing pt-t should link to segB's after addOpp")
	}
	if got := bMid.containsSeg(segA); got != aMid {
		t.Fatal("the link should be mutual (segB -> segA)")
	}
}

// TestOpSpanBaseMerge folds one interior span's pt-t records into another's loop, unlinking the merged span.
func TestOpSpanBaseMerge(t *testing.T) {
	head, _ := newTestContourHead()
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	a := seg.addT(0.25)
	b := seg.addT(0.5)
	if seg.count != 3 {
		t.Fatalf("span count = %d, want 3", seg.count)
	}
	a.span.merge(b.span.upCast()) // fold the 0.5 span into the 0.25 span's loop
	if seg.count != 2 {
		t.Fatalf("count after merge = %d, want 2", seg.count)
	}
	if seg.head.next != a.span {
		t.Fatal("head.next should still be the 0.25 span")
	}
	if a.span.upCast().next != &seg.tail {
		t.Fatal("the merged span should be unlinked (0.25 span now precedes the tail)")
	}
	if !a.containsPtT(b) {
		t.Fatal("the merged span's pt-t should be in the target loop")
	}
	if !b.deleted {
		t.Fatal("the merged span's pt-t should be marked deleted")
	}
}

// coincidentPair builds two identical line segments in one contour plus a coincidence tracker on their state.
func coincidentPair(t *testing.T) (coin *opCoincidence, seg1, seg2 *opSegment) {
	t.Helper()
	head, state := newTestContourHead()
	segA := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	segB := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	return newOpCoincidence(state), segA, segB
}
