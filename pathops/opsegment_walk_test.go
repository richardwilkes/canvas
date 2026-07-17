// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the opSegment walking phase (opsegment_walk.go) and pathWriter (pathwriter.go). The small helpers (the
// active-edge tables, undoneSpan, joinEnds) are exercised directly; findNextWinding/findNextXor, addCurveTo, and the
// whole pathWriter contour assembler are exercised end-to-end through a test-local bridgeWinding/bridgeXor (a
// transcription of the simplify.go drivers, minus the sortContourList re-heading and the findChase multi-piece chase).
// That restricts these end-to-end tests to single, non-self-intersecting contours, whose trace never populates the
// chase; the resulting output is checked for area equivalence to the input.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/path"
)

// traceSimplify is a test-local bridgeWinding/bridgeXor for a single non-self-intersecting contour: it seeds the
// winding, walks the active edges emitting each into the writer, and finishes the contour. It fails the test if the
// trace would need findChase (i.e. the chase list becomes non-empty), which only happens at real junctions.
func traceSimplify(t *testing.T, head *opContourHead, ft path.FillType, xor bool) *path.Path {
	t.Helper()
	writer := newPathWriter(ft)
	for {
		span := findSortableTop(head)
		if span == nil {
			break
		}
		current := span.segment
		start := span.next
		end := &span.opSpanBase
		var chase []*opSpanBase
		unsortable := false
		for {
			active := false
			if xor {
				active = true // bridgeXor has no activeWinding gate; FindUndone already picked a live span
			} else {
				active = current.activeWinding(start, end)
			}
			if active {
				for unsortable || !current.done() {

					nextStart, nextEnd := start, end
					var next *opSegment
					if xor {
						next = current.findNextXor(&nextStart, &nextEnd, &unsortable)
					} else {
						next = current.findNextWinding(&chase, &nextStart, &nextEnd, &unsortable)
					}
					if next == nil {
						break
					}
					if !current.addCurveTo(start, end, writer) {
						t.Fatal("addCurveTo returned false")
					}
					current = next
					start = nextStart
					end = nextEnd
					if writer.isClosed() || (unsortable && start.starter(end).done) {
						break
					}
				}
				if !xor && current.activeWinding(start, end) && !writer.isClosed() {
					spanStart := start.starter(end)
					if !spanStart.done {
						if !current.addCurveTo(start, end, writer) {
							t.Fatal("addCurveTo (tail) returned false")
						}
						current.markDone(spanStart)
					}
				}
				if xor && !writer.isClosed() {
					if !start.starter(end).done {
						t.Fatal("bridgeXor: open contour with an undone starter")
					}
				}
				writer.finishContour()
			} else {
				var last *opSpanBase
				if !current.markAndChaseDone(start, end, &last) {
					t.Fatal("markAndChaseDone returned false")
				}
				if last != nil && !last.chased {
					last.setChased(true)
					chase = append(chase, last)
				}
			}
			if len(chase) != 0 {
				t.Fatal("trace needs FindChase (junction present); use the full driver slice")
			}
			break // FindChase(empty) == nil
		}
	}
	writer.assemble()
	return writer.nativePath()
}

// prepModel builds a single-path op model and runs intersections + coincidence handling (calc/sort angles), mirroring
// the pipeline the bridge drivers run after SortContourList.
func prepModel(t *testing.T, p *path.Path) *opContourHead {
	t.Helper()
	head, state := buildOpModel(t, p)
	co := newOpCoincidence(state)
	addIntersections(head, co)
	runHandleCoincidence(t, head, co)
	return head
}

// TestActiveEdgeTables spot-checks the kUnaryActiveEdge / kActiveEdge decision tables against hand values.
func TestActiveEdgeTables(t *testing.T) {
	// kUnaryActiveEdge = {{F, T}, {T, F}}
	want := [2][2]bool{{false, true}, {true, false}}
	if kUnaryActiveEdge != want {
		t.Fatalf("kUnaryActiveEdge = %v, want %v", kUnaryActiveEdge, want)
	}
	// kActiveEdge[union][miFrom=1][miTo=0][suFrom=0][suTo=0] == T (union row: {{{{F,T},{T,F}},{{T,T},{F,F}}},
	// {{{T,F},{T,F}},{{F,F},{F,F}}}} -> [1][0][0][0] = T).
	if !kActiveEdge[Union][1][0][0][0] {
		t.Fatal("kActiveEdge[union][1][0][0][0] should be true")
	}
	// kActiveEdge[intersect][0][0][0][0] == F.
	if kActiveEdge[Intersect][0][0][0][0] {
		t.Fatal("kActiveEdge[intersect][0][0][0][0] should be false")
	}
	// kActiveEdge[difference][0][1][0][0] == T (difference: [0][1] = {{T,F},{T,F}} -> [0][0]=T).
	if !kActiveEdge[Difference][0][1][0][0] {
		t.Fatal("kActiveEdge[difference][0][1][0][0] should be true")
	}
}

// TestUndoneSpan checks undoneSpan before and after the single span of a rect edge is marked.
func TestUndoneSpan(t *testing.T) {
	head := buildContours(t, rectPath(0, 0, 10, 10, path.FillEvenOdd))
	seg := &realContours(head)[0].head
	span := seg.undoneSpan()
	if span == nil {
		t.Fatal("a fresh edge should report an undone span")
	}
	if span != &seg.head {
		t.Fatal("undoneSpan should return the head span of a single-span edge")
	}
	seg.markDone(&seg.head)
	if seg.undoneSpan() != nil {
		t.Fatal("after markDone, undoneSpan should be nil")
	}
}

// TestJoinEnds checks that joinEnds links this segment's tail pt-t to start's head pt-t.
func TestJoinEnds(t *testing.T) {
	head := buildContours(t, rectPath(0, 0, 10, 10, path.FillEvenOdd))
	segs := segsOf(realContours(head)[0])
	a, b := segs[0], segs[1]
	a.joinEnds(b)
	found := false
	for p := a.tail.ptT.next; p != &a.tail.ptT; p = p.next {
		if p == &b.head.ptT {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("joinEnds should splice start's head pt-t into this segment's tail pt-t loop")
	}
}

// TestBridgeWindingSquare traces a single square through the winding driver and checks the output area.
func TestBridgeWindingSquare(t *testing.T) {
	in := rectPath(0, 0, 10, 10, path.FillEvenOdd)
	head := prepModel(t, in)
	out := traceSimplify(t, head, path.FillEvenOdd, false)
	checkArea(t, "winding square", out, 100, 0.01)
}

// TestBridgeWindingLShape traces a non-convex (but non-self-intersecting) L through the winding driver.
func TestBridgeWindingLShape(t *testing.T) {
	in := polyPath(path.FillEvenOdd,
		pt(0, 0), pt(20, 0), pt(20, 10), pt(10, 10), pt(10, 20), pt(0, 20))
	head := prepModel(t, in)
	out := traceSimplify(t, head, path.FillEvenOdd, false)
	checkArea(t, "winding L", out, 300, 0.01)
}

// TestBridgeXorSquare traces a single square through the xor (even-odd) driver.
func TestBridgeXorSquare(t *testing.T) {
	in := rectPath(0, 0, 10, 10, path.FillEvenOdd)
	head := prepModel(t, in)
	out := traceSimplify(t, head, path.FillEvenOdd, true)
	checkArea(t, "xor square", out, 100, 0.01)
}

// TestDoneAngle checks that doneAngle reports the angle's starter span done flag.
func TestDoneAngle(t *testing.T) {
	head := buildContours(t, rectPath(0, 0, 10, 10, path.FillEvenOdd))
	seg := &realContours(head)[0].head
	// A single-span edge: head (t==0) -> tail (t==1). Build the angle over it directly (a simple edge has no sorted
	// angle, so doneAngle is tested on its own).
	angle := &opAngle{start: &seg.head.opSpanBase, end: seg.head.next}
	if seg.doneAngle(angle) != seg.head.done {
		t.Fatal("doneAngle should mirror the starter span's done flag")
	}
	seg.markDone(&seg.head)
	if !seg.doneAngle(angle) {
		t.Fatal("doneAngle should be true after the starter is marked done")
	}
}

// TestMarkAndChaseDone marks a span done on an intersected segment of two overlapping rects and confirms the starter
// span is flagged done.
func TestMarkAndChaseDone(t *testing.T) {
	a := rectPath(0, 0, 10, 10, path.FillEvenOdd)
	b := rectPath(5, 5, 15, 15, path.FillEvenOdd)
	head, state := buildOpModel(t, a, b)
	co := newOpCoincidence(state)
	addIntersections(head, co)
	runHandleCoincidence(t, head, co)
	var seg *opSegment
	for _, c := range realContours(head) {
		for _, s := range segsOf(c) {
			if s.count > 1 { // an edge that intersection split carries multiple spans
				seg = s
				break
			}
		}
		if seg != nil {
			break
		}
	}
	if seg == nil {
		t.Fatal("expected an intersected (multi-span) segment")
	}
	start := seg.head.next
	end := &seg.head.opSpanBase
	var found *opSpanBase
	if !seg.markAndChaseDone(start, end, &found) {
		t.Fatal("markAndChaseDone returned false")
	}
	if !start.starter(end).done {
		t.Fatal("markAndChaseDone should mark the starter span done")
	}
}
