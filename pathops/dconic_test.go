// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Focused unit tests for dConic's curve-vs-curve members used by the conic solver (dxdyAtT, conicFindExtrema,
// subDivide, the conic bounds). The end-to-end conic/quad and conic/conic intersection drivers live in
// tsectcurve_test.go. cp3 is defined in dconiclineintersect_test.go.

package pathops

import (
	"math"
	"testing"
)

// mkConic builds a dConic from double-precision points and a float32 weight (debugSet semantics).
func mkConic(pts [3]dPoint, w float32) dConic {
	var c dConic
	c.debugSet(pts, w)
	return c
}

// TestConicSubDivide verifies the sub-conic over [t1,t2] pins its endpoints to the parent arc exactly (s=0→t1, s=1→t2)
// and that subDivide(0,1) is the identity. A rational conic's interior parameterization under subdivision is a Möbius
// (not linear) reparameterization, so interior samples don't line up with a linear parent-T map; the interior arc
// geometry is exercised thoroughly by the conic intersection drivers.
func TestConicSubDivide(t *testing.T) {
	conics := []dConic{
		mkConic(cp3(0, 0, 1, 2, 2, 0), 1),
		mkConic(cp3(0, 0, 0, 1, 1, 1), 0.5),
		mkConic(cp3(-4, 1, -4, 5, 0, 5), 0.707106769),
		mkConic(cp3(306.588013, -227.983994, 212.464996, -262.242004, 95.5512009, 58.9763985), 0.707107008),
	}
	ranges := [][2]float64{{0, 1}, {0, 0.5}, {0.5, 1}, {0.25, 0.75}, {0.1, 0.9}}
	for ci, c := range conics {
		if id := c.subDivide(0, 1); id != c {
			t.Fatalf("conic[%d] subDivide(0,1) not identity: %+v", ci, id)
		}
		for _, r := range ranges {
			t1, t2 := r[0], r[1]
			sub := c.subDivide(t1, t2)
			if got, want := sub.ptAtT(0), c.ptAtT(t1); !got.approximatelyEqual(want) {
				t.Fatalf("conic[%d] subDivide(%g,%g).ptAtT(0)=(%g,%g) want parent(%g)=(%g,%g)",
					ci, t1, t2, got.x, got.y, t1, want.x, want.y)
			}
			if got, want := sub.ptAtT(1), c.ptAtT(t2); !got.approximatelyEqual(want) {
				t.Fatalf("conic[%d] subDivide(%g,%g).ptAtT(1)=(%g,%g) want parent(%g)=(%g,%g)",
					ci, t1, t2, got.x, got.y, t2, want.x, want.y)
			}
		}
	}
}

// TestConicDxdyAtTEnds verifies the endpoint tangents: dxdyAtT(0) = w*(p1-p0), dxdyAtT(1) = w*(p2-p1).
func TestConicDxdyAtTEnds(t *testing.T) {
	c := mkConic(cp3(0, 0, 1, 2, 2, 0), 0.75)
	w := float64(c.weight)
	d0 := c.dxdyAtT(0)
	if d0.x != w*(c.pts.pts[1].x-c.pts.pts[0].x) || d0.y != w*(c.pts.pts[1].y-c.pts.pts[0].y) {
		t.Fatalf("dxdyAtT(0)=(%g,%g)", d0.x, d0.y)
	}
	d1 := c.dxdyAtT(1)
	if d1.x != w*(c.pts.pts[2].x-c.pts.pts[1].x) || d1.y != w*(c.pts.pts[2].y-c.pts.pts[1].y) {
		t.Fatalf("dxdyAtT(1)=(%g,%g)", d1.x, d1.y)
	}
}

// TestConicFindExtrema verifies that at the reported y-extremum the tangent is horizontal.
func TestConicFindExtrema(t *testing.T) {
	c := mkConic(cp3(0, 0, 1, 2, 2, 0), 1)
	var tv float64
	if got := conicFindExtrema(c.pts.pts[0].y, c.pts.pts[1].y, c.pts.pts[2].y, c.weight, &tv); got != 1 {
		t.Fatalf("conicFindExtrema roots=%d, want 1", got)
	}
	if tv != 0.5 {
		t.Fatalf("extremum t=%g, want 0.5", tv)
	}
	if dy := c.dxdyAtT(tv).y; math.Abs(dy) > 1e-9 {
		t.Fatalf("dxdyAtT(%g).y=%g, want ~0", tv, dy)
	}
}

// TestConicSetBounds verifies setBoundsConic over an arch conic: x monotonic (endpoints), y-extremum apex.
func TestConicSetBounds(t *testing.T) {
	c := mkConic(cp3(0, 0, 1, 2, 2, 0), 1)
	var r dRect
	r.setBoundsConicFull(c)
	apex := c.ptAtT(0.5)
	if r.left != 0 || r.right != 2 || r.top != 0 {
		t.Fatalf("bounds L=%g R=%g T=%g", r.left, r.right, r.top)
	}
	if math.Abs(r.bottom-apex.y) > 1e-9 {
		t.Fatalf("bottom=%g, want apex y=%g", r.bottom, apex.y)
	}
}
