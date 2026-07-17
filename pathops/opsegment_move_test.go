// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the per-segment coincidence-maintenance phase (opsegment_move.go): moveMultiples, moveNearby,
// missingCoincidence, spansNearby, testForCoincidence, and the clearAll/clearOne/clearVisited span cleanup. The small
// helpers are exercised directly; the trio is exercised both standalone (right after addIntersections, reproducing the
// head of handleCoincidence) and through a full-sequence driver, with a structural model-validity invariant checked
// after each run.

package pathops

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// validateModel walks every real contour's segments head->tail, asserting the span list threads correctly (monotonic
// non-decreasing t, consistent next/prev linkage, reaches the tail) and that the walked opSpan count matches the
// segment's fCount (which moveNearby may have lowered by releasing spans).
func validateModel(t *testing.T, head *opContourHead) {
	t.Helper()
	for _, c := range realContours(head) {
		for _, s := range segsOf(c) {
			base := &s.head.opSpanBase
			prevT := math.Inf(-1)
			n := 0
			for {
				if base.t() < prevT {
					t.Fatalf("segment spans out of t order: %v after %v", base.t(), prevT)
				}
				prevT = base.t()
				if base == &s.tail {
					break
				}
				n++
				sp := base.upCast()
				nxt := sp.next
				if nxt == nil {
					t.Fatal("span has nil next before reaching the tail")
				}
				if nxt.prev != sp {
					t.Fatal("next span's prev does not point back")
				}
				base = nxt
			}
			if n != s.count {
				t.Fatalf("walked %d spans but segment fCount is %d", n, s.count)
			}
		}
	}
}

// TestWayRoughlyEqual checks dPointWayRoughlyEqual's clear cases: identical, far apart, and within the rough epsilon of
// a large magnitude.
func TestWayRoughlyEqual(t *testing.T) {
	p := geom.Point{X: 3, Y: 4}
	if !dPointWayRoughlyEqual(p, p) {
		t.Fatal("identical points should be way-roughly-equal")
	}
	if dPointWayRoughlyEqual(geom.Point{X: 0, Y: 0}, geom.Point{X: 100, Y: 100}) {
		t.Fatal("far apart points should not be way-roughly-equal")
	}
	// magnitude 1000 -> rough threshold ~= 1000*roughEpsilon ~= 7.6e-3; a 1e-3 gap is comfortably below it.
	near := geom.Point{X: 1000, Y: 1000.001}
	if !dPointWayRoughlyEqual(geom.Point{X: 1000, Y: 1000}, near) {
		t.Fatal("points within rough epsilon should be way-roughly-equal")
	}
	if dPointWayRoughlyEqual(geom.Point{X: 1000, Y: 1000}, geom.Point{X: 1000, Y: 1000.5}) {
		t.Fatal("a 0.5 gap at magnitude 1000 exceeds the rough threshold")
	}
}

// TestSegmentVisitedToggle checks opSegment.visited's first-false-then-true behavior and resetVisited.
func TestSegmentVisitedToggle(t *testing.T) {
	var s opSegment
	if s.visited() {
		t.Fatal("first visited() call should return false")
	}
	if !s.visited() {
		t.Fatal("second visited() call should return true")
	}
	s.resetVisited()
	if s.visited() {
		t.Fatal("after resetVisited, visited() should return false again")
	}
}

// reachableOppSegs collects the opposite segments clearVisited would reset from a span's pt-t loops (mirroring its
// walk), so a test can mark them and confirm they are cleared.
func reachableOppSegs(span *opSpanBase) []*opSegment {
	var out []*opSegment
	for {
		stop := &span.ptT
		for ptT := stop.next; ptT != stop; ptT = ptT.next {
			out = append(out, ptT.segment())
		}
		if span.final() {
			return out
		}
		span = span.upCast().next
	}
}

// TestClearVisited marks every opposite segment reachable from a crossing segment's spans, then confirms clearVisited
// resets them all.
func TestClearVisited(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()
	b := path.New()
	b.MoveTo(5, 5).LineTo(15, 5).LineTo(15, 15).LineTo(5, 15).Close()
	head, state := buildOpModel(t, a, b)
	addIntersections(head, newOpCoincidence(state))

	// pick a segment with an interior (crossing) span so its pt-t loops reference opposite segments
	var target *opSegment
	for _, c := range realContours(head) {
		for _, s := range segsOf(c) {
			if interiorSpanCount(s) > 0 {
				target = s
				break
			}
		}
		if target != nil {
			break
		}
	}
	if target == nil {
		t.Fatal("expected a crossing segment with interior spans")
	}
	opps := reachableOppSegs(&target.head.opSpanBase)
	if len(opps) == 0 {
		t.Fatal("crossing segment should reach opposite segments")
	}
	for _, opp := range opps {
		opp.visitedFlag = true
	}
	clearVisited(&target.head.opSpanBase)
	for _, opp := range opps {
		if opp.visitedFlag {
			t.Fatal("clearVisited left an opposite segment visited")
		}
	}
}

// TestCurveDDDispatch checks the dCurve-based dispatch helpers (curveDDPointAtT/curveDDSlopeAtT/ curveDIntersectRay)
// against their already-verified point-array siblings (curveDPointAtT/curveDSlopeAtT/ curveIntersectRay) across all
// four verbs.
func TestCurveDDDispatch(t *testing.T) {
	cases := []struct {
		pts  []geom.Point
		w    float32
		verb path.Verb
	}{
		{verb: path.VerbLine, pts: []geom.Point{pt(0, 0), pt(10, 4)}, w: 1},
		{verb: path.VerbQuad, pts: []geom.Point{pt(0, 0), pt(5, 10), pt(10, 0)}, w: 1},
		{verb: path.VerbConic, pts: []geom.Point{pt(0, 0), pt(5, 10), pt(10, 0)}, w: 0.7},
		{verb: path.VerbCubic, pts: []geom.Point{pt(0, 0), pt(3, 9), pt(7, 9), pt(10, 0)}, w: 1},
	}
	ray := dLine{pts: [2]dPoint{{x: 5, y: -1}, {x: 5, y: 11}}} // vertical ray at x=5
	for _, c := range cases {
		var dc dCurve
		for i, p := range c.pts {
			dc.pts[i].set(p)
		}
		dc.weight = c.w
		for _, tt := range []float64{0, 0.25, 0.5, 0.75, 1} {
			if got, want := curveDDPointAtT(c.verb, dc, tt), curveDPointAtT(c.verb, c.pts, c.w, tt); got != want {
				t.Fatalf("%v curveDDPointAtT(%v) = %v, want %v", c.verb, tt, got, want)
			}
			if got, want := curveDDSlopeAtT(c.verb, dc, tt), curveDSlopeAtT(c.verb, c.pts, c.w, tt); got != want {
				t.Fatalf("%v curveDDSlopeAtT(%v) = %v, want %v", c.verb, tt, got, want)
			}
		}
		gotI := newIntersections()
		curveDIntersectRay(c.verb, dc, ray, gotI)
		wantI := newIntersections()
		curveIntersectRay(c.verb, c.pts, c.w, ray, wantI)
		if gotI.usedCount() != wantI.usedCount() {
			t.Fatalf("%v curveDIntersectRay used = %d, want %d", c.verb, gotI.usedCount(), wantI.usedCount())
		}
		if gotI.usedCount() == 0 {
			t.Fatalf("%v: ray should cross the curve at least once", c.verb)
		}
		for i := 0; i < gotI.usedCount(); i++ {
			if gotI.pt(i) != wantI.pt(i) || gotI.tVal(0, i) != wantI.tVal(0, i) {
				t.Fatalf("%v curveDIntersectRay[%d] = (%v,%v), want (%v,%v)", c.verb, i,
					gotI.tVal(0, i), gotI.pt(i), wantI.tVal(0, i), wantI.pt(i))
			}
		}
	}
}

// TestTestForCoincidence drives opSegment.testForCoincidence directly: a perpendicular from the mid-point of 'this'
// meets a coincident (collinear, overlapping) opp at the same point (coincident=true) but misses a parallel offset opp
// (coincident=false).
func TestTestForCoincidence(t *testing.T) {
	head, _ := newTestContourHead()
	this := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	opp := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)}) // identical / coincident
	if !this.testForCoincidence(opp.head.ptTPtr(), opp.tail.ptTPtr(),
		&this.head.opSpanBase, &this.tail, opp) {
		t.Fatal("collinear identical segments should test coincident")
	}

	head2, _ := newTestContourHead()
	this2 := head2.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	opp2 := head2.addLine([]geom.Point{pt(0, 5), pt(10, 5)}) // parallel, offset by 5
	if this2.testForCoincidence(opp2.head.ptTPtr(), opp2.tail.ptTPtr(),
		&this2.head.opSpanBase, &this2.tail, opp2) {
		t.Fatal("parallel offset segments should not test coincident")
	}
}

// TestClearOneClearAll builds two triangles sharing their base edge, then exercises clearOne on the base segment's head
// span and clearAll over the whole base segment (which also releases its coincident runs).
func TestClearOneClearAll(t *testing.T) {
	above := path.New()
	above.MoveTo(0, 0).LineTo(10, 0).LineTo(5, 5).Close()
	below := path.New()
	below.MoveTo(0, 0).LineTo(10, 0).LineTo(5, -5).Close()
	head, state := buildOpModel(t, above, below)
	co := newOpCoincidence(state)
	cs := realContours(head)
	addIntersectTs(cs[0], cs[1], co)
	if coincidenceCount(co) == 0 {
		t.Fatal("shared base edge should record a coincident run")
	}

	base := segsOf(cs[0])[0] // the (0,0)->(10,0) base edge

	// clearOne zeroes one span's winding and marks it done.
	span := &base.head
	before := base.doneCount
	base.clearOne(span)
	if span.windValue != 0 || span.oppValue != 0 || !span.done {
		t.Fatalf("clearOne left span in state (wind %d, opp %d, done %v)", span.windValue, span.oppValue, span.done)
	}
	if base.doneCount != before+1 {
		t.Fatalf("clearOne should bump doneCount by 1 (got %d, want %d)", base.doneCount, before+1)
	}

	// clearAll marks every span done and drops the runs referencing this segment.
	preRuns := coincidenceCount(co)
	base.clearAll()
	if !base.done() {
		t.Fatal("clearAll should leave the segment done")
	}
	for sp := &base.head; ; {
		if sp.windValue != 0 || sp.oppValue != 0 || !sp.done {
			t.Fatalf("clearAll left a span un-cleared (wind %d, opp %d, done %v)",
				sp.windValue, sp.oppValue, sp.done)
		}
		next := sp.next.upCastable()
		if next == nil {
			break
		}
		sp = next
	}
	if coincidenceCount(co) >= preRuns {
		t.Fatalf("clearAll should release the base segment's coincident run (runs %d -> %d)",
			preRuns, coincidenceCount(co))
	}
}

// TestSpansNearbyFarApart checks the WayRoughlyEqual short-circuit: a segment's head and tail (far apart) report
// found=false with ok=true and no escape.
func TestSpansNearbyFarApart(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).LineTo(100, 0).LineTo(100, 100).Close()
	head, _ := buildOpModel(t, p)
	s := segsOf(realContours(head)[0])[0] // (0,0)->(100,0), endpoints 100 apart
	found, ok := s.spansNearby(&s.head.opSpanBase, &s.tail)
	if !ok {
		t.Fatal("spansNearby should not trip the escape hatch here")
	}
	if found {
		t.Fatal("far-apart span endpoints should not be found near")
	}
}

// TestSpansNearbyFound drives spansNearby's found path (and its spansNearbyCheck inner loop) directly: two segments
// that share a head point report a near pairing.
func TestSpansNearbyFound(t *testing.T) {
	head, _ := newTestContourHead()
	s := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	other := head.addLine([]geom.Point{pt(0, 0), pt(5, 8)}) // shares the (0,0) head point
	found, ok := s.spansNearby(&s.head.opSpanBase, &other.head.opSpanBase)
	if !ok {
		t.Fatal("spansNearby escape hatch tripped unexpectedly")
	}
	if !found {
		t.Fatal("segments sharing a head point should be found near")
	}
}

// TestSpansNearbyCheckDirect drives spansNearbyCheck for one ref point against a coincident check point, confirming it
// records the zero-distance best pairing.
func TestSpansNearbyCheckDirect(t *testing.T) {
	head, _ := newTestContourHead()
	s := head.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	other := head.addLine([]geom.Point{pt(0, 0), pt(3, 7)})
	distSqBest := float32(math.MaxFloat32)
	var refBest, checkBest *opPtT
	if !s.spansNearbyCheck(s.head.ptTPtr(), other.head.ptTPtr(), &distSqBest, &refBest, &checkBest) {
		t.Fatal("spansNearbyCheck escape hatch tripped unexpectedly")
	}
	if checkBest == nil || refBest == nil || distSqBest != 0 {
		t.Fatalf("expected a zero-distance best pairing, got dist=%v refBest=%v checkBest=%v",
			distSqBest, refBest, checkBest)
	}
}

// TestMoveMultiplesMergeGuards covers moveMultiplesMerge's two no-merge branches: a candidate that only holds this
// segment (oppPtTSegment == this), and a candidate whose segment is absent from the start loop.
func TestMoveMultiplesMergeGuards(t *testing.T) {
	// Branch 1: oppTest's loop contains only 'this' -> return false.
	h1, s1 := newTestContourHead()
	newOpCoincidence(s1) // addOpp -> checkForCollapsedCoincidence needs a live tracker
	self := h1.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	cand := h1.addLine([]geom.Point{pt(0, 0), pt(0, 10)}) // shares (0,0) with self
	cand.head.addOpp(&self.head.opSpanBase)               // link self into cand's head pt-t loop
	if self.moveMultiplesMerge(&cand.head.opSpanBase, &self.head.opSpanBase, self.head.ptTPtr()) {
		t.Fatal("a candidate holding only this segment must not merge")
	}

	// Branch 2: oppTest holds a segment X absent from the start loop -> return false.
	h2, s2 := newTestContourHead()
	newOpCoincidence(s2)
	this := h2.addLine([]geom.Point{pt(0, 0), pt(10, 0)})
	oppSeg := h2.addLine([]geom.Point{pt(0, 0), pt(0, 10)})
	x := h2.addLine([]geom.Point{pt(0, 0), pt(-8, 4)})
	oppSeg.head.addOpp(&x.head.opSpanBase) // oppSeg's loop now holds X (not this)
	// start loop is 'this' head alone; X is not in it.
	if this.moveMultiplesMerge(&oppSeg.head.opSpanBase, &oppSeg.head.opSpanBase, this.head.ptTPtr()) {
		t.Fatal("a candidate whose segment is absent from the start loop must not merge")
	}
}

// TestMissingCoincidenceBowTieNone checks that a self-crossing bow-tie (no coincident edges) reports no missing
// coincidence and leaves every segment's visited flag clear.
func TestMissingCoincidenceBowTieNone(t *testing.T) {
	bow := path.New()
	bow.MoveTo(0, 0).LineTo(10, 10).LineTo(10, 0).LineTo(0, 10).Close()
	head, state := buildOpModel(t, bow)
	co := newOpCoincidence(state)
	addIntersections(head, co)

	found := false
	for _, c := range realContours(head) {
		if c.missingCoincidence() {
			found = true
		}
	}
	if found {
		t.Fatal("a bow-tie has no undetected coincident run")
	}
	for _, c := range realContours(head) {
		for _, s := range segsOf(c) {
			if s.visitedFlag {
				t.Fatal("missingCoincidence should clear every visited flag before returning")
			}
		}
	}
	validateModel(t, head)
}

// opPair is one corpus entry: two paths (the second may be nil for a single self-intersecting path).
type opPair struct {
	a    *path.Path
	b    *path.Path
	name string
}

// moveCorpus builds a small set of path pairs that exercise the coincidence-maintenance trio.
func moveCorpus() []opPair {
	rect := func(l, top, r, b float32) *path.Path {
		p := path.New()
		p.MoveTo(l, top).LineTo(r, top).LineTo(r, b).LineTo(l, b).Close()
		return p
	}
	tri := func(x0, y0, x1, y1, x2, y2 float32) *path.Path {
		p := path.New()
		p.MoveTo(x0, y0).LineTo(x1, y1).LineTo(x2, y2).Close()
		return p
	}
	oval := func(l, top, r, b float32) *path.Path {
		return path.New().AddOval(geom.Rect{Left: l, Top: top, Right: r, Bottom: b}, geom.DirectionCW)
	}
	bow := path.New()
	bow.MoveTo(0, 0).LineTo(10, 10).LineTo(10, 0).LineTo(0, 10).Close()
	return []opPair{
		{name: "identical-rects", a: rect(0, 0, 10, 10), b: rect(0, 0, 10, 10)},
		{name: "overlapping-rects", a: rect(0, 0, 10, 10), b: rect(5, 5, 15, 15)},
		{name: "nested-rects", a: rect(0, 0, 20, 20), b: rect(5, 5, 15, 15)},
		{name: "shared-base-tris", a: tri(0, 0, 10, 0, 5, 5), b: tri(0, 0, 10, 0, 5, -5)},
		{name: "corner-touch-rects", a: rect(0, 0, 10, 10), b: rect(10, 10, 20, 20)},
		// partial collinear overlap: bottom/top edges coincide only on [5,10] -> partial coincident runs
		{name: "partial-overlap-rects", a: rect(0, 0, 10, 10), b: rect(5, 0, 15, 10)},
		// T-junction: B's bottom edge (y=10) lies on A's top edge over [3,7] -> a segment split by two vertices that
		// the opposite edge does not share
		{name: "t-junction-rects", a: rect(0, 0, 10, 10), b: rect(3, 10, 7, 20)},
		// curves: two overlapping ovals exercise the quad/conic subDivide + testForCoincidence lanes
		{name: "overlapping-ovals", a: oval(0, 0, 12, 8), b: oval(6, 0, 18, 8)},
		{name: "identical-ovals", a: oval(0, 0, 10, 10), b: oval(0, 0, 10, 10)},
		{name: "bow-tie", a: bow, b: nil},
	}
}

// buildPair builds the op model for a corpus entry with a coincidence tracker and the intersections added.
func buildPair(t *testing.T, pair opPair) (*opContourHead, *opCoincidence) {
	t.Helper()
	var head *opContourHead
	var state *opGlobalState
	if pair.b != nil {
		head, state = buildOpModel(t, pair.a, pair.b)
	} else {
		head, state = buildOpModel(t, pair.a)
	}
	co := newOpCoincidence(state)
	addIntersections(head, co)
	return head, co
}

// TestMoveStepsPreserveModel runs the standalone head of HandleCoincidence (addExpanded, then move_multiples, then
// move_nearby) over the corpus and asserts each step succeeds and the model stays structurally valid.
func TestMoveStepsPreserveModel(t *testing.T) {
	for _, pair := range moveCorpus() {
		t.Run(pair.name, func(t *testing.T) {
			head, co := buildPair(t, pair)
			if !co.addExpanded() {
				t.Fatal("addExpanded returned false")
			}
			for _, c := range realContours(head) {
				if !c.moveMultiples() {
					t.Fatal("moveMultiples returned false")
				}
			}
			for _, c := range realContours(head) {
				if !c.moveNearby() {
					t.Fatal("moveNearby returned false")
				}
			}
			validateModel(t, head)
		})
	}
}

// TestHandleCoincidenceFullSequence runs the complete handleCoincidence sequence (including the
// moveMultiples/moveNearby/missingCoincidence steps and the final angle sort) over the corpus, asserting it completes,
// leaves a valid model, and leaves every span with a non-negative winding.
func TestHandleCoincidenceFullSequence(t *testing.T) {
	for _, pair := range moveCorpus() {
		t.Run(pair.name, func(t *testing.T) {
			head, co := buildPair(t, pair)
			runHandleCoincidence(t, head, co)
			validateModel(t, head)
			for _, s := range allSpans(head) {
				if s.windValue < 0 {
					t.Fatalf("span at %v has negative windValue %d", s.pt(), s.windValue)
				}
			}
		})
	}
}

// runHandleCoincidence runs handleCoincidence over the built model (the same driver Op/Simplify use), failing the test
// if it aborts. Tests call it after buildOpModel + addIntersections to bring the model to the fully-resolved,
// angle-sorted state the output tracers consume.
func runHandleCoincidence(t *testing.T, head *opContourHead, co *opCoincidence) {
	t.Helper()
	if !handleCoincidence(head, co) {
		t.Fatal("handleCoincidence returned false")
	}
}
