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

// buildOpModel runs the edge builder over one or more paths (the first is the primary, the rest operands), returning
// the contour head and its global state so the caller can attach a coincidence tracker.
func buildOpModel(t *testing.T, paths ...*path.Path) (*opContourHead, *opGlobalState) {
	t.Helper()
	head := &opContourHead{}
	state := newOpGlobalState(head)
	b := newOpEdgeBuilder(paths[0], head, state)
	for _, p := range paths[1:] {
		b.addOperand(p)
	}
	if !b.finish() {
		t.Fatal("edge builder finish() returned false")
	}
	return head, state
}

// segsOf returns a contour's segments in forward order.
func segsOf(c *opContour) []*opSegment {
	var out []*opSegment
	for s := c.first(); s != nil; s = s.next {
		out = append(out, s)
	}
	return out
}

// spanBaseAtPt returns the span whose primary pt-t point equals pt, or nil. The head/tail always match their endpoints;
// interior spans are inserted by intersection.
func spanBaseAtPt(s *opSegment, target geom.Point) *opSpanBase {
	for base := &s.head.opSpanBase; ; {
		if base.pt() == target {
			return base
		}
		if base == &s.tail {
			return nil
		}
		base = base.upCast().next
	}
}

// interiorSpanCount returns the number of spans strictly between the head and the tail (i.e. spans added by
// intersection).
func interiorSpanCount(s *opSegment) int {
	n := 0
	for base := s.head.next; base != &s.tail; base = base.upCast().next {
		n++
	}
	return n
}

// coincidenceCount returns the number of coincident runs recorded across both lists.
func coincidenceCount(co *opCoincidence) int {
	n := 0
	for c := co.head; c != nil; c = c.next {
		n++
	}
	for c := co.top; c != nil; c = c.next {
		n++
	}
	return n
}

// TestAddIntersectSelfCrossing intersects a bow-tie contour with itself: the two diagonals cross at (5,5). The crossing
// must add an interior span to each diagonal and link their pt-t records into a shared loop.
func TestAddIntersectSelfCrossing(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).LineTo(10, 10).LineTo(10, 0).LineTo(0, 10).Close()
	head, state := buildOpModel(t, p)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)

	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	segs := segsOf(cs[0])
	if len(segs) != 4 {
		t.Fatalf("segment count = %d, want 4", len(segs))
	}
	// seg0 = (0,0)->(10,10); seg2 = (10,0)->(0,10). They cross at (5,5).
	e0, e2 := segs[0], segs[2]
	cross := pt(5, 5)
	e0Cross := spanBaseAtPt(e0, cross)
	if e0Cross == nil {
		t.Fatal("seg0 has no span at the crossing (5,5)")
	}
	if e0Cross == &e0.head.opSpanBase || e0Cross == &e0.tail {
		t.Fatal("seg0 crossing span should be interior, not an endpoint")
	}
	e2Cross := spanBaseAtPt(e2, cross)
	if e2Cross == nil {
		t.Fatal("seg2 has no span at the crossing (5,5)")
	}
	// The two crossing pt-t records share a loop: from seg0's crossing, a pt-t on seg2 is reachable.
	linked := e0Cross.ptT.containsSeg(e2)
	if linked == nil {
		t.Fatal("seg0 crossing pt-t is not linked to seg2")
	}
	if linked.pt != cross {
		t.Fatalf("linked pt-t on seg2 = %v, want %v", linked.pt, cross)
	}
	if linked.span != e2Cross {
		t.Fatal("linked pt-t should belong to seg2's crossing span")
	}
	// No coincident runs for a transverse crossing.
	if !coincidence.isEmpty() {
		t.Fatalf("coincidence should be empty, got %d runs", coincidenceCount(coincidence))
	}
}

// TestAddIntersectTwoRects intersects two overlapping rectangles whose edges cross transversely at (20,10) and (10,20).
// Each crossing links one edge of each rectangle; no edges coincide.
func TestAddIntersectTwoRects(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(20, 0).LineTo(20, 20).LineTo(0, 20).Close()
	b := path.New()
	b.MoveTo(10, 10).LineTo(30, 10).LineTo(30, 30).LineTo(10, 30).Close()
	head, state := buildOpModel(t, a, b)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)

	cs := realContours(head)
	if len(cs) != 2 {
		t.Fatalf("contour count = %d, want 2", len(cs))
	}
	aSegs := segsOf(cs[0]) // top, right, bottom, left
	bSegs := segsOf(cs[1])
	// A.right (x=20) crosses B.top (y=10) at (20,10).
	if link := checkLinkedCrossing(t, aSegs[1], bSegs[0], pt(20, 10)); link == nil {
		t.Fatal("A.right x B.top crossing at (20,10) not linked")
	}
	// A.bottom (y=20) crosses B.left (x=10) at (10,20).
	if link := checkLinkedCrossing(t, aSegs[2], bSegs[3], pt(10, 20)); link == nil {
		t.Fatal("A.bottom x B.left crossing at (10,20) not linked")
	}
	if !coincidence.isEmpty() {
		t.Fatalf("coincidence should be empty, got %d runs", coincidenceCount(coincidence))
	}
}

// checkLinkedCrossing asserts that both segments carry an interior span at the crossing point and that their pt-t
// records share a loop; it returns the linked pt-t on segB (nil if the assertion setup failed).
func checkLinkedCrossing(t *testing.T, segA, segB *opSegment, cross geom.Point) *opPtT {
	t.Helper()
	aSpan := spanBaseAtPt(segA, cross)
	bSpan := spanBaseAtPt(segB, cross)
	if aSpan == nil || bSpan == nil {
		t.Fatalf("missing crossing span at %v (a=%v b=%v)", cross, aSpan, bSpan)
	}
	if aSpan == &segA.head.opSpanBase || aSpan == &segA.tail {
		t.Fatalf("crossing at %v on segA should be interior", cross)
	}
	link := aSpan.ptT.containsSeg(segB)
	if link == nil || link.span != bSpan {
		return nil
	}
	return link
}

// TestAddIntersectCoincidentEdge intersects two triangles that share their base edge (0,0)->(10,0). The shared edge is
// fully coincident, so exactly one coincident run is recorded.
func TestAddIntersectCoincidentEdge(t *testing.T) {
	above := path.New()
	above.MoveTo(0, 0).LineTo(10, 0).LineTo(5, 5).Close()
	below := path.New()
	below.MoveTo(0, 0).LineTo(10, 0).LineTo(5, -5).Close()
	head, state := buildOpModel(t, above, below)
	coincidence := newOpCoincidence(state)

	cs := realContours(head)
	if len(cs) != 2 {
		t.Fatalf("contour count = %d, want 2", len(cs))
	}
	// Intersect the two contours directly (bounds order is irrelevant for a direct pair).
	addIntersectTs(cs[0], cs[1], coincidence)

	if got := coincidenceCount(coincidence); got != 1 {
		t.Fatalf("coincident run count = %d, want 1", got)
	}
	run := coincidence.head
	if run == nil {
		t.Fatal("coincident run missing from head list")
	}
	// The recorded run must span the shared base edge's endpoints on both segments.
	endpoints := map[geom.Point]bool{pt(0, 0): true, pt(10, 0): true}
	if !endpoints[run.coinPtTStart.pt] || !endpoints[run.coinPtTEnd.pt] {
		t.Fatalf("coin run endpoints = %v..%v, want the base-edge corners", run.coinPtTStart.pt, run.coinPtTEnd.pt)
	}
	if !endpoints[run.oppPtTStart.pt] || !endpoints[run.oppPtTEnd.pt] {
		t.Fatalf("opp run endpoints = %v..%v, want the base-edge corners", run.oppPtTStart.pt, run.oppPtTEnd.pt)
	}
	// The tracked pt-t records must be flagged coincident.
	if !run.coinPtTStart.coincident || !run.oppPtTStart.coincident {
		t.Fatal("coincident run endpoints not flagged coincident")
	}
}

// TestAddIntersectQuadLine intersects a quadratic arc with a rectangle's edge: the arc y=40t(1-t) peaks at (10,10) and
// crosses the y=5 edge twice at non-integral t, exercising the curve dispatch (dQuadOf + horizontalQuad) and the
// recompute-from-t addT path.
func TestAddIntersectQuadLine(t *testing.T) {
	arc := path.New()
	arc.MoveTo(0, 0).QuadTo(10, 20, 20, 0).Close()
	box := path.New()
	box.MoveTo(-5, 5).LineTo(25, 5).LineTo(25, 25).LineTo(-5, 25).Close()
	head, state := buildOpModel(t, arc, box)
	coincidence := newOpCoincidence(state)

	cs := realContours(head)
	if len(cs) != 2 {
		t.Fatalf("contour count = %d, want 2", len(cs))
	}
	// The edge builder splits the arc at its max-curvature peak, so cs[0] carries two quad halves plus the closing
	// line; each half crosses the y=5 edge once. Assert against the box edge, which is not split.
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbQuad, path.VerbQuad, path.VerbLine) {
		t.Fatalf("arc contour verbs = %v, want quad, quad, line", v)
	}
	boxBottom := segsOf(cs[1])[0] // (-5,5)->(25,5), the y=5 edge
	addIntersectTs(cs[0], cs[1], coincidence)

	if n := interiorSpanCount(boxBottom); n != 2 {
		t.Fatalf("box-edge interior span count = %d, want 2 (two arc crossings)", n)
	}
	linkedQuads := 0
	for base := boxBottom.head.next; base != &boxBottom.tail; base = base.upCast().next {
		if base.pt().Y != 5 {
			t.Fatalf("box-edge interior span at %v is not on the y=5 edge", base.pt())
		}
		// Each crossing must link back to one of the quad halves (a Quad-verb segment).
		found := false
		for ptT := base.ptT.next; ptT != &base.ptT; ptT = ptT.next {
			if ptT.segment().verb == path.VerbQuad {
				found = true
				break
			}
		}
		if found {
			linkedQuads++
		}
	}
	if linkedQuads != 2 {
		t.Fatalf("%d of the box-edge crossings link to a quad half, want 2", linkedQuads)
	}
	if !coincidence.isEmpty() {
		t.Fatalf("coincidence should be empty, got %d runs", coincidenceCount(coincidence))
	}
}

// TestAddIntersectDisjoint intersects two far-apart contours: nothing is added and no run is recorded. The bounds
// early-out also returns false so the driver stops advancing.
func TestAddIntersectDisjoint(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(10, 0).LineTo(5, 8).Close()
	b := path.New()
	b.MoveTo(0, 100).LineTo(10, 100).LineTo(5, 108).Close()
	head, state := buildOpModel(t, a, b)
	coincidence := newOpCoincidence(state)

	cs := realContours(head)
	// a is entirely above b, so the pairwise early-out returns false.
	if addIntersectTs(cs[0], cs[1], coincidence) {
		t.Fatal("disjoint contours should report the vertical-separation early-out (false)")
	}
	for _, c := range cs {
		for _, s := range segsOf(c) {
			if n := interiorSpanCount(s); n != 0 {
				t.Fatalf("disjoint contour segment gained %d interior spans", n)
			}
		}
	}
	if !coincidence.isEmpty() {
		t.Fatal("coincidence should be empty for disjoint contours")
	}
}

// TestAddIntersectDriverMatchesPair verifies the addIntersections driver produces the same self-intersection structure
// as a direct addIntersectTs(contour, contour) call.
func TestAddIntersectDriverMatchesPair(t *testing.T) {
	build := func() (*opContourHead, *opCoincidence) {
		p := path.New()
		p.MoveTo(0, 0).LineTo(10, 10).LineTo(10, 0).LineTo(0, 10).Close()
		head, state := buildOpModel(t, p)
		return head, newOpCoincidence(state)
	}
	viaDriver, driverCoin := build()
	addIntersections(viaDriver, driverCoin)

	viaPair, pairCoin := build()
	c := realContours(viaPair)[0]
	addIntersectTs(c, c, pairCoin)

	dSeg := segsOf(realContours(viaDriver)[0])
	pSeg := segsOf(realContours(viaPair)[0])
	for i := range dSeg {
		if d, p := spanTs(dSeg[i]), spanTs(pSeg[i]); len(d) != len(p) {
			t.Fatalf("segment %d span count: driver %d vs pair %d", i, len(d), len(p))
		}
	}
	if a, b := coincidenceCount(driverCoin), coincidenceCount(pairCoin); a != b {
		t.Fatalf("coincidence count: driver %d vs pair %d", a, b)
	}
}
