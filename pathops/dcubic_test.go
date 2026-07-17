// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Focused unit tests for dCubic's curve-vs-curve members used by the cubic solver (convexHull, chopAt, subDivide,
// dxdyAtT, otherPts, collapsed/controlsInside). The end-to-end cubic intersection drivers live in tsectcurve_test.go.
// cu4 is defined in dcubiclineintersect_test.go.

package pathops

import "testing"

// mkCubic builds a dCubic from double-precision points (debugSet semantics).
func mkCubic(pts [4]dPoint) dCubic {
	var c dCubic
	c.debugSet(pts)
	return c
}

// TestCubicConvexHullSquare verifies the four-point hull of a convex-quadrilateral cubic.
func TestCubicConvexHullSquare(t *testing.T) {
	c := mkCubic(cu4(0, 0, 0, 1, 1, 1, 1, 0))
	var order [4]int
	if got := c.convexHull(&order); got != 4 {
		t.Fatalf("convexHull count=%d, want 4", got)
	}
	seen := [4]bool{}
	for _, o := range order {
		if o < 0 || o > 3 || seen[o] {
			t.Fatalf("convexHull order not a permutation: %v", order)
		}
		seen[o] = true
	}
}

// TestCubicConvexHullTriangle verifies a three-point hull when a control point lies inside the triangle of the other
// three.
func TestCubicConvexHullTriangle(t *testing.T) {
	// (0,0),(4,0),(0,4) form a triangle; (1,1) is strictly inside it.
	c := mkCubic(cu4(0, 0, 4, 0, 0, 4, 1, 1))
	var order [4]int
	if got := c.convexHull(&order); got != 3 {
		t.Fatalf("convexHull count=%d, want 3", got)
	}
}

// TestCubicSubDivideInvariant verifies the subdivision invariant: sub over [t1,t2] at s equals the parent at t1 +
// s*(t2-t1). This exercises both the chopAt lanes (t1==0 / t2==1) and the general lane.
func TestCubicSubDivideInvariant(t *testing.T) {
	cubics := []dCubic{
		mkCubic(cu4(0, 0, 0, 1, 1, 1, 1, 0)),
		mkCubic(cu4(0, 1, 4, 5, 1, 0, 5, 3)),
		mkCubic(cu4(2, 3, 0, 4, 3, 2, 5, 3)),
		mkCubic(cu4(945.08099365234375, 747.1619873046875, 982.5679931640625, 747.1619873046875,
			1013.6290283203125, 719.656005859375, 1019.1910400390625, 683.72601318359375)),
	}
	ranges := [][2]float64{{0, 1}, {0, 0.5}, {0.5, 1}, {0.25, 0.75}, {0.3, 0.6}}
	samples := []float64{0, 0.25, 0.5, 0.75, 1}
	for ci, c := range cubics {
		for _, r := range ranges {
			t1, t2 := r[0], r[1]
			sub := c.subDivide(t1, t2)
			for _, s := range samples {
				got := sub.ptAtT(s)
				want := c.ptAtT(t1 + s*(t2-t1))
				if !got.approximatelyEqual(want) {
					t.Fatalf("cubic[%d] subDivide(%g,%g) at s=%g: got (%g,%g) want (%g,%g)",
						ci, t1, t2, s, got.x, got.y, want.x, want.y)
				}
			}
		}
	}
}

// TestCubicChopAtInvariant verifies chopAt(t): first over [0,t], second over [t,1].
func TestCubicChopAtInvariant(t *testing.T) {
	c := mkCubic(cu4(0, 1, 4, 5, 1, 0, 5, 3))
	for _, chopT := range []float64{0.5, 0.3, 0.9} {
		pair := c.chopAt(chopT)
		first := pair.first()
		second := pair.second()
		for _, s := range []float64{0, 0.5, 1} {
			gotFirst := first.ptAtT(s)
			wantFirst := c.ptAtT(s * chopT)
			if !gotFirst.approximatelyEqual(wantFirst) {
				t.Fatalf("chopAt(%g) first at s=%g: got (%g,%g) want (%g,%g)",
					chopT, s, gotFirst.x, gotFirst.y, wantFirst.x, wantFirst.y)
			}
			gotSecond := second.ptAtT(s)
			wantSecond := c.ptAtT(chopT + s*(1-chopT))
			if !gotSecond.approximatelyEqual(wantSecond) {
				t.Fatalf("chopAt(%g) second at s=%g: got (%g,%g) want (%g,%g)",
					chopT, s, gotSecond.x, gotSecond.y, wantSecond.x, wantSecond.y)
			}
		}
	}
}

// TestCubicDxdyAtTEnds verifies the endpoint tangents: dxdyAtT(0)=3*(p1-p0), dxdyAtT(1)=3*(p3-p2).
func TestCubicDxdyAtTEnds(t *testing.T) {
	c := mkCubic(cu4(1, 2, 3, 7, 5, 1, 6, 4))
	d0 := c.dxdyAtT(0)
	if d0.x != 3*(c.pts[1].x-c.pts[0].x) || d0.y != 3*(c.pts[1].y-c.pts[0].y) {
		t.Fatalf("dxdyAtT(0)=(%g,%g)", d0.x, d0.y)
	}
	d1 := c.dxdyAtT(1)
	if d1.x != 3*(c.pts[3].x-c.pts[2].x) || d1.y != 3*(c.pts[3].y-c.pts[2].y) {
		t.Fatalf("dxdyAtT(1)=(%g,%g)", d1.x, d1.y)
	}
}

// TestCubicOtherPts verifies the endpoint-odd-man point selection.
func TestCubicOtherPts(t *testing.T) {
	c := mkCubic(cu4(1, 2, 3, 4, 5, 6, 7, 8))
	if o := c.otherPts(0); o != [3]dPoint{c.pts[1], c.pts[2], c.pts[3]} {
		t.Fatalf("otherPts(0)=%v", o)
	}
	if o := c.otherPts(3); o != [3]dPoint{c.pts[0], c.pts[1], c.pts[2]} {
		t.Fatalf("otherPts(3)=%v", o)
	}
}

// TestCubicCollapsedControls verifies collapsed and controlsInside.
func TestCubicCollapsedControls(t *testing.T) {
	if !mkCubic(cu4(3, 3, 3, 3, 3, 3, 3, 3)).collapsed() {
		t.Fatal("coincident cubic should be collapsed")
	}
	if mkCubic(cu4(0, 0, 0, 1, 1, 1, 1, 0)).collapsed() {
		t.Fatal("square cubic should not be collapsed")
	}
	// A cubic whose control points project between the endpoints along the chord.
	if !mkCubic(cu4(0, 0, 1, 1, 2, 1, 3, 0)).controlsInside() {
		t.Fatal("expected controlsInside true")
	}
}
