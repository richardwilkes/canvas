// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// newTestContourHead builds a global state plus a contour head whose first contour is initialized, matching the state
// the edge builder establishes before it starts adding segments.
func newTestContourHead() (*opContourHead, *opGlobalState) {
	head := &opContourHead{}
	state := newOpGlobalState(head)
	head.init(state, false, false)
	return head, state
}

func pt(x, y float32) geom.Point { return geom.Point{X: x, Y: y} }

// spanTs walks a segment's span list from head to tail and returns the t values in order.
func spanTs(s *opSegment) []float64 {
	var ts []float64
	base := &s.head.opSpanBase
	for {
		ts = append(ts, base.t())
		if base == &s.tail {
			break
		}
		base = base.upCast().next
	}
	return ts
}

func TestOpSegmentAddLine(t *testing.T) {
	head, state := newTestContourHead()
	pts := []geom.Point{pt(10, 20), pt(30, 60)}
	seg := head.addLine(pts)
	if head.count != 1 {
		t.Fatalf("contour count = %d, want 1", head.count)
	}
	if seg.verb != path.VerbLine {
		t.Fatalf("verb = %d, want line", seg.verb)
	}
	if seg.count != 1 {
		t.Fatalf("span count = %d, want 1", seg.count)
	}
	if seg.globalState() != state {
		t.Fatal("segment global state mismatch")
	}
	if got := seg.bounds; got != (pathOpsBounds{left: 10, top: 20, right: 30, bottom: 60}) {
		t.Fatalf("bounds = %+v, want {10 20 30 60}", got)
	}
	if seg.head.t() != 0 || seg.tail.t() != 1 {
		t.Fatalf("head/tail t = %v/%v, want 0/1", seg.head.t(), seg.tail.t())
	}
	if seg.head.pt() != pt(10, 20) || seg.tail.pt() != pt(30, 60) {
		t.Fatalf("head/tail pt = %v/%v", seg.head.pt(), seg.tail.pt())
	}
	// The head is an upcastable span; the tail is not.
	if seg.head.upCastable() == nil {
		t.Fatal("head should be upcastable")
	}
	if seg.tail.upCastable() != nil {
		t.Fatal("tail should not be upcastable")
	}
}

func TestOpSegmentAddT(t *testing.T) {
	head, _ := newTestContourHead()
	seg := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})

	mid := seg.addT(0.5)
	if mid == nil || mid.pt != pt(5, 0) {
		t.Fatalf("addT(0.5) = %+v, want pt (5,0)", mid)
	}
	if seg.count != 2 {
		t.Fatalf("span count after one insert = %d, want 2", seg.count)
	}
	// Re-adding the same t returns the same pt-t and inserts nothing.
	if again := seg.addT(0.5); again != mid {
		t.Fatal("addT(0.5) again should return the same pt-t")
	}
	if seg.count != 2 {
		t.Fatalf("span count after duplicate = %d, want 2", seg.count)
	}
	// Out-of-order inserts land sorted.
	seg.addT(0.25)
	seg.addT(0.75)
	if got := spanTs(seg); !equalFloats(got, []float64{0, 0.25, 0.5, 0.75, 1}) {
		t.Fatalf("span ts = %v, want [0 0.25 0.5 0.75 1]", got)
	}
	if seg.count != 4 {
		t.Fatalf("span count = %d, want 4", seg.count)
	}
	// The endpoints resolve to the head and tail pt-ts.
	if seg.addT(0) != &seg.head.ptT {
		t.Fatal("addT(0) should return the head pt-t")
	}
	if seg.addT(1) != &seg.tail.ptT {
		t.Fatal("addT(1) should return the tail pt-t")
	}
	// Out-of-range t values fail.
	if seg.addT(1.5) != nil {
		t.Fatal("addT(1.5) should be nil")
	}
	if seg.addT(-0.5) != nil {
		t.Fatal("addT(-0.5) should be nil")
	}
	if seg.count != 4 {
		t.Fatalf("span count after failed adds = %d, want 4", seg.count)
	}
}

func TestOpSegmentAddTOnCurve(t *testing.T) {
	head, _ := newTestContourHead()
	// A quad; the inserted point must be the curve's point at t, not a linear interpolation.
	pts := []geom.Point{pt(0, 0), pt(10, 20), pt(20, 0)}
	seg := head.appendSegment().addQuad(pts, &head.opContour)
	ptT := seg.addT(0.5)
	if ptT == nil {
		t.Fatal("addT(0.5) on quad returned nil")
	}
	// Quad at t=0.5 of (0,0),(10,20),(20,0) is (10,10).
	if ptT.pt != pt(10, 10) {
		t.Fatalf("quad pt at 0.5 = %v, want (10,10)", ptT.pt)
	}
	if ptT.pt != seg.ptAtT(0.5) {
		t.Fatalf("addT point %v disagrees with ptAtT %v", ptT.pt, seg.ptAtT(0.5))
	}
}

func TestOpSegmentBounds(t *testing.T) {
	head, _ := newTestContourHead()

	// Monotonic quad: bounds are the control-point box.
	q := head.appendSegment().addQuad([]geom.Point{pt(0, 0), pt(10, 0), pt(10, 10)}, &head.opContour)
	if q.bounds != (pathOpsBounds{left: 0, top: 0, right: 10, bottom: 10}) {
		t.Fatalf("monotonic quad bounds = %+v, want {0 0 10 10}", q.bounds)
	}

	// Quad with a y extremum at t=0.5 (curve peaks at y=10, below the control point at y=20).
	q2 := head.appendSegment().addQuad([]geom.Point{pt(0, 0), pt(10, 20), pt(20, 0)}, &head.opContour)
	if q2.bounds != (pathOpsBounds{left: 0, top: 0, right: 20, bottom: 10}) {
		t.Fatalf("extremum quad bounds = %+v, want {0 0 20 10}", q2.bounds)
	}

	// Monotonic cubic: control-point box.
	c := head.appendSegment().addCubic([]geom.Point{pt(0, 0), pt(3, 4), pt(6, 8), pt(9, 12)}, &head.opContour)
	if c.bounds != (pathOpsBounds{left: 0, top: 0, right: 9, bottom: 12}) {
		t.Fatalf("cubic bounds = %+v, want {0 0 9 12}", c.bounds)
	}
}

func TestOpSegmentOrientation(t *testing.T) {
	head, _ := newTestContourHead()
	hLine := head.appendSegment().addLine([]geom.Point{pt(0, 5), pt(10, 5)}, &head.opContour)
	if !hLine.isHorizontal() || hLine.isVertical() {
		t.Fatal("horizontal line misclassified")
	}
	vLine := head.appendSegment().addLine([]geom.Point{pt(5, 0), pt(5, 10)}, &head.opContour)
	if !vLine.isVertical() || vLine.isHorizontal() {
		t.Fatal("vertical line misclassified")
	}
	if hLine.lastPt() != pt(10, 5) {
		t.Fatalf("lastPt = %v, want (10,5)", hLine.lastPt())
	}
}

func TestOpContourMultiSegment(t *testing.T) {
	head, _ := newTestContourHead()
	head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	head.addQuad([]geom.Point{pt(10, 0), pt(15, 5), pt(10, 10)})
	head.addConic([]geom.Point{pt(10, 10), pt(0, 10), pt(0, 0)}, 0.7)
	head.addCubic([]geom.Point{pt(0, 0), pt(-5, -5), pt(-5, 5), pt(0, 0)})
	if head.count != 4 {
		t.Fatalf("contour count = %d, want 4", head.count)
	}
	// Walk the forward segment list and collect verbs.
	var verbs []path.Verb
	for seg := head.first(); seg != nil; seg = seg.next {
		verbs = append(verbs, seg.verb)
		if seg.contour != &head.opContour {
			t.Fatal("segment contour back-pointer mismatch")
		}
	}
	want := []path.Verb{path.VerbLine, path.VerbQuad, path.VerbConic, path.VerbCubic}
	if len(verbs) != len(want) {
		t.Fatalf("verbs = %v, want %v", verbs, want)
	}
	for i := range want {
		if verbs[i] != want[i] {
			t.Fatalf("verbs = %v, want %v", verbs, want)
		}
	}
	// complete() computes the union of the segment bounds.
	head.complete()
	union := head.first().bounds
	for seg := head.first().next; seg != nil; seg = seg.next {
		union.addBounds(seg.bounds)
	}
	if head.bounds != union {
		t.Fatalf("contour bounds = %+v, want union %+v", head.bounds, union)
	}
}

func TestOpSegmentAddTPtsDisjoint(t *testing.T) {
	head, _ := newTestContourHead()
	// A quad whose point at t=0.5 is (10,10); the head point at t=0 is (0,0).
	seg := head.appendSegment().addQuad([]geom.Point{pt(0, 0), pt(10, 20), pt(20, 0)}, &head.opContour)
	// Adding t=0.5 with an explicit pt that coincides in coordinates with the head (0,0) must NOT merge with the head:
	// ptsDisjoint sees the curve wander far away between the two t values, so a new span is added.
	ptT := seg.addTPt(0.5, pt(0, 0))
	if ptT == nil {
		t.Fatal("addTPt(0.5, (0,0)) returned nil")
	}
	if ptT == &seg.head.ptT {
		t.Fatal("disjoint point should not have merged with the head")
	}
	if seg.count != 2 {
		t.Fatalf("span count = %d, want 2 (a new span inserted)", seg.count)
	}
	if ptT.t != 0.5 || ptT.pt != pt(0, 0) {
		t.Fatalf("inserted pt-t = {t:%v pt:%v}, want {0.5 (0,0)}", ptT.t, ptT.pt)
	}
}

func TestOpContourBuilderCurves(t *testing.T) {
	head, _ := newTestContourHead()
	b := newOpContourBuilder(&head.opContour)
	// A pending line flushes when a curve arrives; the curve is then added.
	b.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	b.addQuad([]geom.Point{pt(10, 0), pt(15, 5), pt(20, 0)})
	if head.count != 2 {
		t.Fatalf("line+quad; contour count = %d, want 2", head.count)
	}
	// addCurve dispatches by verb into stable-storage copies.
	b.addCurve(path.VerbConic, []geom.Point{pt(20, 0), pt(25, 5), pt(30, 0)}, 0.5)
	b.addCurve(path.VerbCubic, []geom.Point{pt(30, 0), pt(33, 3), pt(36, -3), pt(40, 0)}, 1)
	b.addCurve(path.VerbLine, []geom.Point{pt(40, 0), pt(50, 0)}, 1)
	b.flush()
	if head.count != 5 {
		t.Fatalf("after conic+cubic+line; contour count = %d, want 5", head.count)
	}
	var verbs []path.Verb
	for seg := head.first(); seg != nil; seg = seg.next {
		verbs = append(verbs, seg.verb)
	}
	want := []path.Verb{path.VerbLine, path.VerbQuad, path.VerbConic, path.VerbCubic, path.VerbLine}
	for i := range want {
		if i >= len(verbs) || verbs[i] != want[i] {
			t.Fatalf("verbs = %v, want %v", verbs, want)
		}
	}
}

func TestOpContourBuilderLineElimination(t *testing.T) {
	head, _ := newTestContourHead()
	b := newOpContourBuilder(&head.opContour)

	// A line followed by its exact reverse cancels: no segment is emitted.
	b.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	b.addLine([]geom.Point{pt(10, 0), pt(0, 0)})
	b.flush()
	if head.count != 0 {
		t.Fatalf("reversed pair should cancel; contour count = %d, want 0", head.count)
	}

	// Two distinct lines: the first flushes when the second arrives, the second on the final flush.
	b.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	b.addLine([]geom.Point{pt(10, 0), pt(10, 10)})
	b.flush()
	if head.count != 2 {
		t.Fatalf("distinct lines; contour count = %d, want 2", head.count)
	}
}

func TestOpContourHeadAppendRemove(t *testing.T) {
	head, state := newTestContourHead()
	c2 := head.appendContour()
	c2.init(state, false, false)
	c3 := head.appendContour()
	c3.init(state, false, false)
	if head.next != c2 || c2.next != c3 || c3.next != nil {
		t.Fatal("appendContour did not link contours in order")
	}
	head.remove(c3)
	if c2.next != nil {
		t.Fatal("remove(c3) should unlink c3 from c2")
	}
	head.remove(c2)
	if head.next != nil {
		t.Fatal("remove(c2) should unlink c2 from head")
	}
	// Removing the head itself is a no-op.
	head.remove(&head.opContour)
}

func equalFloats(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
