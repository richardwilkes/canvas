// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Focused tests for the curve-vs-curve dQuad members that the quad/quad curve-pair solver consumes: subDivide (both
// forms), chopAt, otherPts, dxdyAtT, hullIntersects, quadSetABC, align, collapsed, controlsInside, plus the supporting
// dPointMid and the line/line intersectRay. These primitives have no dedicated unit test elsewhere (they are exercised
// through quad/quad intersection, which needs the full curve-pair solver); the checks here cross-validate against the
// already-oracle-verified ptAtT and against closed-form geometry.

package pathops

import (
	"fmt"
	"math"
	"testing"
)

// dq builds a dQuad directly from double-precision coordinates (no float32 rounding).
func dq(x0, y0, x1, y1, x2, y2 float64) dQuad {
	return dQuad{pts: [3]dPoint{{x: x0, y: y0}, {x: x1, y: y1}, {x: x2, y: y2}}}
}

func requireApproxPt(t *testing.T, got, want dPoint, ctx string) {
	t.Helper()
	if !got.approximatelyEqual(want) {
		t.Errorf("%s: got (%g,%g), want (%g,%g)", ctx, got.x, got.y, want.x, want.y)
	}
}

func TestQuadSubDivideIdentity(t *testing.T) {
	q := dq(1, 1, 0, 2, 3, 3)
	sub := q.subDivide(0, 1)
	// The (0,1) range must return the exact same control points (the fast-path early return).
	if sub != q {
		t.Fatalf("subDivide(0,1) changed the quad: got %+v want %+v", sub, q)
	}
}

func TestQuadSubDivideMatchesPtAtT(t *testing.T) {
	quads := []dQuad{
		dq(1, 1, 0, 2, 3, 3),
		dq(0, 0, 2, 4, 4, 0),
		dq(-37.3484879, 10.0192947, -36.4966316, 13.2140198, -38.1506348, 16.0788383),
	}
	ranges := [][2]float64{{0, 1}, {0, 0.5}, {0.25, 0.75}, {0.3, 1}, {0.6, 0.9}}
	for qi, q := range quads {
		for _, r := range ranges {
			t1, t2 := r[0], r[1]
			sub := q.subDivide(t1, t2)
			for _, s := range []float64{0, 0.25, 0.5, 0.75, 1} {
				mapped := t1 + s*(t2-t1)
				requireApproxPt(t, sub.ptAtT(s), q.ptAtT(mapped),
					// context
					fmt.Sprintf("subDivide q%d range[%g,%g] s=%g", qi, t1, t2, s))
			}
		}
	}
}

func TestQuadChopAt(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	const t0 = 0.375
	pair := q.chopAt(t0)
	first := pair.first()
	second := pair.second()
	// Endpoints come through exactly (interpQuadCoordsChop returns p0 and p2 directly).
	if first.pts[0] != q.pts[0] {
		t.Errorf("chopAt first[0]=%+v want %+v", first.pts[0], q.pts[0])
	}
	if second.pts[2] != q.pts[2] {
		t.Errorf("chopAt second[2]=%+v want %+v", second.pts[2], q.pts[2])
	}
	// The split point is shared.
	if first.pts[2] != second.pts[0] {
		t.Errorf("chopAt split point mismatch: first[2]=%+v second[0]=%+v", first.pts[2], second.pts[0])
	}
	// Each half reparameterizes the original.
	for _, s := range []float64{0, 0.25, 0.5, 0.75, 1} {
		requireApproxPt(t, first.ptAtT(s), q.ptAtT(s*t0), fmt.Sprintf("chop first s=%g", s))
		requireApproxPt(t, second.ptAtT(s), q.ptAtT(t0+s*(1-t0)), fmt.Sprintf("chop second s=%g", s))
	}
}

func TestQuadOtherPts(t *testing.T) {
	q := dq(10, 11, 20, 21, 30, 31) // distinct points so the index mapping is unambiguous
	// From the documented bit-twiddle table: the two points that are NOT oddMan.
	cases := []struct {
		oddMan int
		want   [2]dPoint
	}{
		{oddMan: 0, want: [2]dPoint{q.pts[1], q.pts[2]}},
		{oddMan: 1, want: [2]dPoint{q.pts[0], q.pts[2]}},
		{oddMan: 2, want: [2]dPoint{q.pts[1], q.pts[0]}},
	}
	for _, c := range cases {
		got := q.otherPts(c.oddMan)
		if got != c.want {
			t.Errorf("otherPts(%d)=%+v want %+v", c.oddMan, got, c.want)
		}
	}
}

func TestQuadDxDyAtT(t *testing.T) {
	q := dq(0, 0, 2, 4, 4, 0)
	for _, tv := range []float64{0, 0.2, 0.5, 0.8, 1} {
		got := q.dxdyAtT(tv)
		// Analytic Quad'(t)/2 = (1-t)(P1-P0) + t(P2-P1).
		p1p0 := q.pts[1].sub(q.pts[0])
		p2p1 := q.pts[2].sub(q.pts[1])
		wantX := (1-tv)*p1p0.x + tv*p2p1.x
		wantY := (1-tv)*p1p0.y + tv*p2p1.y
		if math.Abs(got.x-wantX) > 1e-9 || math.Abs(got.y-wantY) > 1e-9 {
			t.Errorf("dxdyAtT(%g)=(%g,%g) want (%g,%g)", tv, got.x, got.y, wantX, wantY)
		}
	}
	// Degenerate endpoint tangent: with P0==P1, the closed form is {0,0} at t=0, so the fallback returns P2-P0.
	qd := dq(1, 1, 1, 1, 4, 5)
	got := qd.dxdyAtT(0)
	want := qd.pts[2].sub(qd.pts[0])
	if got != want {
		t.Errorf("degenerate dxdyAtT(0)=%+v want %+v", got, want)
	}
}

func TestQuadHullIntersectsSeparated(t *testing.T) {
	// Two identical small arches, one shifted far to the right: their hulls are disjoint, so the conservative reject
	// must return false (at most an endpoint intersection).
	q1 := dq(0, 0, 1, 1, 2, 0)
	q2 := dq(10, 0, 11, 1, 12, 0)
	if result, _ := q1.hullIntersects(q2); result {
		t.Errorf("separated hulls: hullIntersects returned true, want false")
	}
}

func TestQuadHullIntersectsCrossing(t *testing.T) {
	// An up-arch and a down-arch that genuinely cross: the reject must return true (it may never reject a real
	// crossing) and neither quad is a line, so isLinear is false.
	q1 := dq(0, 0, 2, 4, 4, 0)
	q2 := dq(0, 2, 2, -2, 4, 2)
	result, isLinear := q1.hullIntersects(q2)
	if !result {
		t.Errorf("crossing hulls: hullIntersects returned false, want true")
	}
	if isLinear {
		t.Errorf("crossing hulls: isLinear=true, want false")
	}
}

func TestQuadHullIntersectsLinear(t *testing.T) {
	// A collinear "quad" (its hull is a line) against an arch that does not share endpoints: result is true and
	// isLinear is reported.
	q1 := dq(0, 0, 1, 1, 2, 2)
	q2 := dq(0, 3, 1, 3, 2, 3)
	result, isLinear := q1.hullIntersects(q2)
	if !result || !isLinear {
		t.Errorf("linear hull: got (result=%v,isLinear=%v), want (true,true)", result, isLinear)
	}
}

func TestQuadSetABC(t *testing.T) {
	// SetABC converts the (A,B,C) Bernstein-ish control values of one coordinate into plain-polynomial a,b,c so that
	// a*t*t + b*t + c == A*t*t + 2*B*t*(1-t) + C*(1-t)*(1-t).
	A, B, C := 3.0, -1.5, 4.25
	a, b, c := quadSetABC(A, B, C)
	for _, tv := range []float64{0, 0.1, 0.5, 0.9, 1} {
		got := a*tv*tv + b*tv + c
		want := A*tv*tv + 2*B*tv*(1-tv) + C*(1-tv)*(1-tv)
		if math.Abs(got-want) > 1e-12*(1+math.Abs(want)) {
			t.Errorf("SetABC at t=%g: got %g want %g", tv, got, want)
		}
	}
}

func TestQuadSubDivideCtrl(t *testing.T) {
	// Recover the middle control point of a sub-quad from its endpoints: subDivideCtrl(a,c,t1,t2) must reproduce
	// subDivide(t1,t2)'s control point. This also exercises intersectRayLine.
	quads := []dQuad{
		dq(0, 0, 2, 4, 4, 0),
		dq(1, 1, 0, 2, 3, 3),
		dq(-52.8062439, 14.1493912, -53.6638947, 10.948595, -52.0070419, 8.07883835),
	}
	ranges := [][2]float64{{0, 1}, {0.2, 0.8}, {0, 0.6}, {0.4, 1}}
	for qi, q := range quads {
		for _, r := range ranges {
			t1, t2 := r[0], r[1]
			sub := q.subDivide(t1, t2)
			b := q.subDivideCtrl(sub.pts[0], sub.pts[2], t1, t2)
			requireApproxPt(t, b, sub.pts[1], fmt.Sprintf("subDivideCtrl q%d range[%g,%g]", qi, t1, t2))
		}
	}
}

func TestQuadAlign(t *testing.T) {
	// pts[0] shares its x with the middle control point, so align snaps the dst x to it, leaving y alone.
	qx := dq(5, 3, 5, 8, 9, 1)
	dst := dPoint{x: 5.0000001, y: 2}
	qx.align(0, &dst)
	if dst.x != 5 || dst.y != 2 {
		t.Errorf("align x-branch: got (%g,%g) want (5,2)", dst.x, dst.y)
	}
	// pts[0] shares its y with the middle control point, so align snaps the dst y, leaving x alone.
	qy := dq(5, 3, 8, 3, 1, 9)
	dst = dPoint{x: 2, y: 3.0000001}
	qy.align(0, &dst)
	if dst.x != 2 || dst.y != 3 {
		t.Errorf("align y-branch: got (%g,%g) want (2,3)", dst.x, dst.y)
	}
}

func TestQuadCollapsedControlsInside(t *testing.T) {
	if !dq(2, 2, 2, 2, 2, 2).collapsed() {
		t.Errorf("all-equal quad: collapsed()=false, want true")
	}
	if dq(0, 0, 2, 4, 4, 0).collapsed() {
		t.Errorf("normal quad: collapsed()=true, want false")
	}
	if !dq(0, 0, 1, 0, 2, 0).controlsInside() {
		t.Errorf("control between endpoints: controlsInside()=false, want true")
	}
	if dq(0, 0, -1, 0, 2, 0).controlsInside() {
		t.Errorf("control behind endpoint: controlsInside()=true, want false")
	}
}

func TestIntersectRayLineCrossing(t *testing.T) {
	in := newIntersections()
	a := dLine{pts: [2]dPoint{{x: 0, y: 0}, {x: 2, y: 2}}}
	b := dLine{pts: [2]dPoint{{x: 0, y: 2}, {x: 2, y: 0}}}
	if used := in.intersectRayLine(a, b); used != 1 {
		t.Fatalf("crossing rays: used=%d want 1", used)
	}
	if math.Abs(in.ts[0][0]-0.5) > 1e-12 || math.Abs(in.ts[1][0]-0.5) > 1e-12 {
		t.Errorf("crossing rays: t=(%g,%g) want (0.5,0.5)", in.ts[0][0], in.ts[1][0])
	}
	requireApproxPt(t, in.pt(0), dPoint{x: 1, y: 1}, "crossing rays point")
}

func TestIntersectRayLineCoincident(t *testing.T) {
	// Two overlapping segments of the same line: the ray form has no single answer, so it returns the two endpoints
	// with a fixed choice of T values (t on a = 0, t on b = 1).
	in := newIntersections()
	a := dLine{pts: [2]dPoint{{x: 0, y: 0}, {x: 2, y: 2}}}
	b := dLine{pts: [2]dPoint{{x: 1, y: 1}, {x: 3, y: 3}}}
	if used := in.intersectRayLine(a, b); used != 2 {
		t.Fatalf("coincident rays: used=%d want 2", used)
	}
	if in.ts[0][0] != 0 || in.ts[1][0] != 1 || in.ts[1][1] != 1 {
		t.Errorf("coincident rays: t=(%g;%g,%g) want (0;1,1)", in.ts[0][0], in.ts[1][0], in.ts[1][1])
	}
}

func TestIntersectRayLineParallel(t *testing.T) {
	in := newIntersections()
	a := dLine{pts: [2]dPoint{{x: 0, y: 0}, {x: 2, y: 0}}}
	b := dLine{pts: [2]dPoint{{x: 0, y: 1}, {x: 2, y: 1}}}
	if used := in.intersectRayLine(a, b); used != 0 {
		t.Errorf("parallel distinct rays: used=%d want 0", used)
	}
}

func TestDPointMid(t *testing.T) {
	got := dPointMid(dPoint{x: 2, y: 4}, dPoint{x: 6, y: 10})
	if got != (dPoint{x: 4, y: 7}) {
		t.Errorf("dPointMid=%+v want (4,7)", got)
	}
}
