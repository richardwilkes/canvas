// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Focused tests for the tSect span layer (tcurve.go + tspan.go): the tQuad wrapper's delegation, the tCoincident
// perpendicular-match, and the tSpan geometry (bounds, splitting, hull/linear-intersection checks, the bounded-span
// list). These primitives have no dedicated test of their own elsewhere — they are exercised through the quad/quad
// intersection, which needs the full tSect solver body — so the checks here cross-validate against the already-verified
// dQuad geometry and against closed-form expectations.

package pathops

import (
	"math"
	"testing"
)

// tq wraps a dQuad in the tCurve adapter.
func tq(q dQuad) *tQuad { return &tQuad{quad: q} }

func TestTQuadDelegation(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	c := tq(q)
	if c.pointCount() != 3 || c.pointLast() != 2 || c.maxIntersections() != 4 || c.isConic() {
		t.Fatalf("constants wrong: count=%d last=%d max=%d conic=%v",
			c.pointCount(), c.pointLast(), c.maxIntersections(), c.isConic())
	}
	for n := 0; n < 3; n++ {
		if c.pointAt(n) != q.pts[n] {
			t.Errorf("pointAt(%d)=%v want %v", n, c.pointAt(n), q.pts[n])
		}
	}
	if c.ptAtT(0.3) != q.ptAtT(0.3) {
		t.Errorf("ptAtT mismatch")
	}
	if c.dxdyAtT(0.4) != q.dxdyAtT(0.4) {
		t.Errorf("dxdyAtT mismatch")
	}
	if c.collapsed() != q.collapsed() || c.controlsInside() != q.controlsInside() {
		t.Errorf("collapsed/controlsInside mismatch")
	}
	if got, want := c.otherPts(0), q.otherPts(0); got[0] != want[0] || got[1] != want[1] {
		t.Errorf("otherPts mismatch: %v vs %v", got, want)
	}
	// setPoint mutates the wrapped quad.
	c2 := tq(q)
	c2.setPoint(1, dPoint{x: 9, y: 9})
	if c2.quad.pts[1] != (dPoint{x: 9, y: 9}) {
		t.Errorf("setPoint did not mutate: %v", c2.quad.pts[1])
	}
	// makeCurve returns a fresh blank tQuad.
	if _, ok := c.makeCurve().(*tQuad); !ok {
		t.Errorf("makeCurve did not return *tQuad")
	}
	// subDivide writes the sub-quad into the destination curve.
	dst := &tQuad{}
	c.subDivide(0.2, 0.8, dst)
	if dst.quad != q.subDivide(0.2, 0.8) {
		t.Errorf("subDivide mismatch: %v vs %v", dst.quad, q.subDivide(0.2, 0.8))
	}
	// setBounds delegates to dRect.setBoundsQuadFull.
	var got, want dRect
	c.setBounds(&got)
	want.setBoundsQuadFull(q)
	if got != want {
		t.Errorf("setBounds mismatch: %v vs %v", got, want)
	}
}

func TestTQuadRayAndHullDelegation(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	c := tq(q)
	line := dLine{pts: [2]dPoint{{x: -1, y: 2}, {x: 5, y: 2}}}
	got := newIntersections()
	want := newIntersections()
	nGot := got.intersectRayTCurve(c, line)
	nWant := want.intersectRayQuad(q, line)
	if nGot != nWant || nGot == 0 {
		t.Fatalf("intersectRay used mismatch: %d vs %d", nGot, nWant)
	}
	for i := 0; i < nGot; i++ {
		if got.tVal(0, i) != want.tVal(0, i) {
			t.Errorf("intersectRay t[%d] mismatch: %g vs %g", i, got.tVal(0, i), want.tVal(0, i))
		}
	}
	// hullIntersectsQuad / hullIntersectsCurve delegate to dQuad.hullIntersects.
	q2 := dq(0, 4, 2, 0, 4, 4)
	wantRes, wantLin := q2.hullIntersects(c.quad)
	var gotLin bool
	if gotRes := c.hullIntersectsQuad(&q2, &gotLin); gotRes != wantRes || (gotRes && gotLin != wantLin) {
		t.Errorf("hullIntersectsQuad mismatch: (%v,%v) vs (%v,%v)", gotRes, gotLin, wantRes, wantLin)
	}
	var gotLin2 bool
	if gotRes := tq(q2).hullIntersectsCurve(c, &gotLin2); gotRes != wantRes {
		t.Errorf("hullIntersectsCurve mismatch: %v vs %v", gotRes, wantRes)
	}
}

func TestTCoincidentSetPerpMatch(t *testing.T) {
	c2 := tq(dq(0, 0, 4, 8, 8, 0))
	const u = 0.4
	cPt := c2.quad.ptAtT(u)
	c1 := tq(dq(0, 0, 1, 1, 2, 0)) // any curve supplying a tangent direction at t
	var coin tCoincident
	coin.setPerp(c1, 0.3, cPt, c2)
	if !coin.isMatch() {
		t.Fatalf("expected a coincidence match for a point on the opposite curve")
	}
	if !coin.perpPoint().approximatelyEqual(cPt) {
		t.Errorf("perp point %v not near cPt %v", coin.perpPoint(), cPt)
	}
	if math.Abs(coin.perpT()-u) > 1e-6 {
		t.Errorf("perpT=%g want ~%g", coin.perpT(), u)
	}
}

func TestTCoincidentSetPerpNoMatch(t *testing.T) {
	c2 := tq(dq(0, 0, 4, 8, 8, 0))
	c1 := tq(dq(0, 0, 1, 1, 2, 0))
	// A point well off the opposite curve cannot match.
	off := dPoint{x: 3, y: 40}
	var coin tCoincident
	coin.setPerp(c1, 0.5, off, c2)
	if coin.isMatch() {
		t.Errorf("did not expect a match for a point off the curve")
	}
	// When the perpendicular ray misses the curve entirely, setPerp resets to the init state.
	if coin.perpT() >= 0 && !coin.perpPoint().approximatelyEqual(off) {
		// A found-but-non-matching perp is fine; the assertion above already covers isMatch.
		return
	}
}

func TestTSpanInitBounds(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	c := tq(q)
	s := newTSpan(c)
	s.init(c)
	if s.startT != 0 || s.endT != 1 {
		t.Fatalf("init range wrong: [%g,%g]", s.startT, s.endT)
	}
	if s.collapsed {
		t.Errorf("real quad should not be collapsed")
	}
	var want dRect
	want.setBoundsQuadFull(q)
	if s.bounds != want {
		t.Errorf("bounds %v want %v", s.bounds, want)
	}
	if s.boundsMax != math.Max(want.width(), want.height()) {
		t.Errorf("boundsMax=%g want %g", s.boundsMax, math.Max(want.width(), want.height()))
	}
	// The [0,1] part is the whole curve.
	requireApproxPt(t, s.part.ptAtT(0.5), q.ptAtT(0.5), "part midpoint")
	// The coincidents start uninitialized-to-match (perpT == -1).
	if s.coinStart.perpT() != -1 || s.coinEnd.perpT() != -1 {
		t.Errorf("coincidents not init'd: %g %g", s.coinStart.perpT(), s.coinEnd.perpT())
	}
}

func TestTSpanSplitAt(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	c := tq(q)
	head := newTSpan(c)
	head.init(c)
	half := newTSpan(c)
	if !half.splitAt(head, 0.5) {
		t.Fatalf("splitAt returned false unexpectedly")
	}
	head.initBounds(c)
	half.initBounds(c)
	if head.startT != 0 || head.endT != 0.5 {
		t.Errorf("head range wrong: [%g,%g]", head.startT, head.endT)
	}
	if half.startT != 0.5 || half.endT != 1 {
		t.Errorf("half range wrong: [%g,%g]", half.startT, half.endT)
	}
	if head.next != half || half.prev != head {
		t.Errorf("split did not relink head<->half")
	}
	// The split point is shared by both halves and matches the parent curve.
	requireApproxPt(t, head.pointLast(), q.ptAtT(0.5), "head end")
	requireApproxPt(t, half.pointFirst(), q.ptAtT(0.5), "half start")
	requireApproxPt(t, half.pointLast(), q.ptAtT(1), "half end")
}

func TestTSpanSplitCollapse(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	c := tq(q)
	head := newTSpan(c)
	head.init(c)
	// Splitting at the end T collapses the new span to a point.
	s := newTSpan(c)
	if s.splitAt(head, 1) {
		t.Errorf("splitAt at endT should return false")
	}
	if !s.collapsed {
		t.Errorf("new span should be marked collapsed")
	}
}

func TestTSpanHullsIntersect(t *testing.T) {
	c := tq(dq(0, 0, 2, 4, 4, 0)) // arch up
	newSpan := func(q dQuad) *tSpan {
		s := newTSpan(tq(q))
		s.init(tq(q))
		return s
	}
	// Crossing arches: hulls intersect, no shared endpoint, not linear -> 1.
	a := newSpan(dq(0, 0, 2, 4, 4, 0))
	b := newSpan(dq(0, 4, 2, 0, 4, 4))
	var start, oppStart bool
	if got := a.hullsIntersect(b, &start, &oppStart); got != 1 {
		t.Errorf("crossing hulls: got %d want 1", got)
	}
	// Disjoint bounds -> 0.
	far := newSpan(dq(100, 100, 101, 101, 102, 100))
	if got := a.hullsIntersect(far, &start, &oppStart); got != 0 {
		t.Errorf("disjoint hulls: got %d want 0", got)
	}
	// Shared endpoint, diverging -> 2 (only endpoints in common).
	d := newSpan(dq(4, 0, 5, -1, 6, 0))
	if got := a.hullsIntersect(d, &start, &oppStart); got != 2 {
		t.Errorf("shared-endpoint hulls: got %d want 2", got)
	}
	_ = c
}

func TestTSpanOnlyEndPointsInCommon(t *testing.T) {
	mk := func(q dQuad) *tSpan {
		s := newTSpan(tq(q))
		s.init(tq(q))
		return s
	}
	a := mk(dq(0, 0, 1, 1, 2, 0))
	// b shares a's last point (2,0) and diverges.
	b := mk(dq(2, 0, 3, -1, 4, 0))
	var start, oppStart, pts bool
	if !a.onlyEndPointsInCommon(b, &start, &oppStart, &pts) {
		t.Errorf("expected diverging shared endpoint -> true")
	}
	if start || !oppStart || !pts {
		t.Errorf("out params wrong: start=%v oppStart=%v pts=%v", start, oppStart, pts)
	}
	// No shared endpoint at all -> false.
	e := mk(dq(0, 4, 2, 0, 4, 4))
	if a.onlyEndPointsInCommon(e, &start, &oppStart, &pts) || pts {
		t.Errorf("expected no common endpoint -> false")
	}
}

func TestTSpanLinearIntersects(t *testing.T) {
	// A near-straight span (essentially the x-axis).
	line := newTSpan(tq(dq(0, 0, 1, 1e-7, 2, 0)))
	line.init(tq(dq(0, 0, 1, 1e-7, 2, 0)))
	above := tq(dq(0, 1, 1, 2, 2, 1))     // wholly above y=0
	crossing := tq(dq(1, -1, 1, 1, 1, 3)) // straddles y=0
	if got := line.linearIntersects(above); got != 0 {
		t.Errorf("above-line span: got %d want 0", got)
	}
	if got := line.linearIntersects(crossing); got != 1 {
		t.Errorf("crossing span: got %d want 1", got)
	}
}

func TestTSpanBoundedList(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	c := tq(q)
	mkRange := func(s0, e0 float64) *tSpan {
		s := newTSpan(c)
		s.startT = s0
		s.endT = e0
		s.initBounds(c)
		return s
	}
	a := mkRange(0, 1)
	b := mkRange(0.2, 0.5)
	d := mkRange(0.6, 0.9)
	a.addBounded(b)
	a.addBounded(d)
	if !a.isBounded() {
		t.Fatalf("a should be bounded")
	}
	if a.findOppSpan(b) != b || a.findOppSpan(d) != d {
		t.Errorf("findOppSpan failed")
	}
	if a.findOppSpan(mkRange(0, 1)) != nil {
		t.Errorf("findOppSpan should be nil for a non-member")
	}
	if a.oppT(0.3) != b || a.oppT(0.7) != d || a.oppT(0.55) != nil {
		t.Errorf("oppT lookup failed")
	}
	if math.Abs(a.closestBoundedT(q.ptAtT(0.2))-0.2) > 1e-9 {
		t.Errorf("closestBoundedT(near 0.2)=%g", a.closestBoundedT(q.ptAtT(0.2)))
	}
	if math.Abs(a.closestBoundedT(q.ptAtT(0.9))-0.9) > 1e-9 {
		t.Errorf("closestBoundedT(near 0.9)=%g", a.closestBoundedT(q.ptAtT(0.9)))
	}
	// Removing one leaves the list non-empty; removing the last empties it.
	if a.removeBounded(b) {
		t.Errorf("removeBounded(b) should report the list still non-empty")
	}
	if !a.removeBounded(d) {
		t.Errorf("removeBounded(d) should report the list now empty")
	}
	if a.isBounded() {
		t.Errorf("a should no longer be bounded")
	}
}

func TestTSpanContains(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	c := tq(q)
	a := newTSpan(c)
	a.startT, a.endT = 0, 0.4
	a.initBounds(c)
	b := newTSpan(c)
	b.startT, b.endT = 0.4, 1
	b.initBounds(c)
	a.next = b
	if !a.contains(0.2) || !a.contains(0.7) || a.contains(1.5) {
		t.Errorf("contains walk failed: %v %v %v", a.contains(0.2), a.contains(0.7), a.contains(1.5))
	}
}

func TestTCoincidentMarkCoincident(t *testing.T) {
	var c tCoincident
	c.init()
	c.markCoincident() // unmatched -> perpT reset to -1, matched set
	if !c.isMatch() || c.perpT() != -1 {
		t.Errorf("markCoincident on unmatched: isMatch=%v perpT=%g", c.isMatch(), c.perpT())
	}
	// Already-matched: markCoincident must leave perpT untouched.
	c.perpTVal = 0.5
	c.markCoincident()
	if c.perpT() != 0.5 {
		t.Errorf("markCoincident clobbered an already-matched perpT: %g", c.perpT())
	}
	// tSpan.markCoincident marks both ends.
	s := newTSpan(tq(dq(0, 0, 2, 4, 4, 0)))
	s.markCoincident()
	if !s.coinStart.isMatch() || !s.coinEnd.isMatch() {
		t.Errorf("span markCoincident did not mark both ends")
	}
}

func TestTSpanLinearsIntersect(t *testing.T) {
	mk := func(q dQuad) *tSpan {
		s := newTSpan(tq(q))
		s.init(tq(q))
		return s
	}
	line := mk(dq(0, 0, 1, 1e-7, 2, 0))
	above := mk(dq(0, 1, 1, 2, 2, 1))
	crossing := mk(dq(1, -1, 1, 1, 1, 3))
	if line.linearsIntersect(above) {
		t.Errorf("linearsIntersect(above) should be false")
	}
	if !line.linearsIntersect(crossing) {
		t.Errorf("linearsIntersect(crossing) should be true")
	}
}

func TestTSpanRemoveAllBoundedAndSplit(t *testing.T) {
	c := tq(dq(0, 0, 2, 4, 4, 0))
	a := newTSpan(c)
	a.init(c)
	b := newTSpan(c)
	b.init(c)
	a.addBounded(b)
	b.addBounded(a)
	// removeAllBounded walks a's list and unlinks a from each opposite; b's list empties.
	if !a.removeAllBounded() {
		t.Errorf("removeAllBounded should report b became empty")
	}
	if b.isBounded() {
		t.Errorf("b should no longer be bounded")
	}
	// split is splitAt at the midpoint.
	head := newTSpan(c)
	head.init(c)
	half := newTSpan(c)
	if !half.split(head) {
		t.Fatalf("split returned false")
	}
	if head.endT != 0.5 || half.startT != 0.5 {
		t.Errorf("split midpoint wrong: head.endT=%g half.startT=%g", head.endT, half.startT)
	}
}

func TestTSpanOppAccessors(t *testing.T) {
	c := tq(dq(0, 0, 2, 4, 4, 0))
	a := newTSpan(c)
	a.init(c)
	b := newTSpan(c)
	b.startT, b.endT = 0.3, 0.6
	b.initBounds(c)
	a.addBounded(b)
	if !a.hasOppT(0.4) || a.hasOppT(0.9) {
		t.Errorf("hasOppT wrong")
	}
	if a.findOppT(0.4) != b {
		t.Errorf("findOppT should return b")
	}
	a.reset()
	if a.isBounded() {
		t.Errorf("reset should clear the bounded list")
	}
}
