// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import (
	"math"
	"math/rand"
	"testing"
)

// evalQuadRef evaluates a quad Bezier in float64 via de Casteljau.
func evalQuadRef(p0, p1, p2 Point, t float64) (x, y float64) {
	lerp := func(a, b float64) float64 { return a + (b-a)*t }
	x01 := lerp(float64(p0.X), float64(p1.X))
	y01 := lerp(float64(p0.Y), float64(p1.Y))
	x12 := lerp(float64(p1.X), float64(p2.X))
	y12 := lerp(float64(p1.Y), float64(p2.Y))
	return lerp(x01, x12), lerp(y01, y12)
}

// evalCubicRef evaluates a cubic Bezier in float64 via de Casteljau.
func evalCubicRef(p0, p1, p2, p3 Point, t float64) (x, y float64) {
	lerp := func(a, b float64) float64 { return a + (b-a)*t }
	ax, ay := lerp(float64(p0.X), float64(p1.X)), lerp(float64(p0.Y), float64(p1.Y))
	bx, by := lerp(float64(p1.X), float64(p2.X)), lerp(float64(p1.Y), float64(p2.Y))
	cx, cy := lerp(float64(p2.X), float64(p3.X)), lerp(float64(p2.Y), float64(p3.Y))
	abx, aby := lerp(ax, bx), lerp(ay, by)
	bcx, bcy := lerp(bx, cx), lerp(by, cy)
	return lerp(abx, bcx), lerp(aby, bcy)
}

func TestEvalQuadAgainstReference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for iter := 0; iter < 100; iter++ {
		q := []Point{
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
		}
		for _, tt := range []float32{0, 0.25, 0.5, 0.75, 1} {
			got := EvalQuadAt(q, tt)
			wx, wy := evalQuadRef(q[0], q[1], q[2], float64(tt))
			if math.Abs(float64(got.X)-wx) > 1e-3 || math.Abs(float64(got.Y)-wy) > 1e-3 {
				t.Fatalf("EvalQuadAt(%v)=%+v want (%g,%g)", tt, got, wx, wy)
			}
		}
	}
}

func TestChopQuadAt(t *testing.T) {
	q := []Point{{X: 0, Y: 0}, {X: 50, Y: 100}, {X: 100, Y: 0}}
	dst := make([]Point, 5)
	ChopQuadAt(q, dst, 0.5)
	// Endpoints preserved, split point equals evaluation at t.
	if dst[0] != q[0] || dst[4] != q[2] {
		t.Error("chop endpoints wrong")
	}
	at := EvalQuadAt(q, 0.5)
	if math.Abs(float64(dst[2].X-at.X)) > 1e-4 || math.Abs(float64(dst[2].Y-at.Y)) > 1e-4 {
		t.Errorf("chop split %+v != eval %+v", dst[2], at)
	}
	// The two halves evaluate to points on the original curve.
	left := dst[:3]
	got := EvalQuadAt(left, 1)
	if math.Abs(float64(got.X-at.X)) > 1e-4 {
		t.Error("left half end mismatch")
	}
}

func TestEvalCubicAgainstReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	for iter := 0; iter < 100; iter++ {
		c := []Point{
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
		}
		for _, tt := range []float32{0, 0.1, 0.5, 0.9, 1} {
			got := EvalCubicPosAt(c, tt)
			wx, wy := evalCubicRef(c[0], c[1], c[2], c[3], float64(tt))
			if math.Abs(float64(got.X)-wx) > 2e-3 || math.Abs(float64(got.Y)-wy) > 2e-3 {
				t.Fatalf("EvalCubicPosAt(%v)=%+v want (%g,%g)", tt, got, wx, wy)
			}
		}
	}
}

func TestChopCubicAt(t *testing.T) {
	c := []Point{{X: 0, Y: 0}, {X: 0, Y: 100}, {X: 100, Y: 100}, {X: 100, Y: 0}}
	dst := make([]Point, 7)
	ChopCubicAt(c, dst, 0.25)
	if dst[0] != c[0] || dst[6] != c[3] {
		t.Error("chop endpoints wrong")
	}
	at := EvalCubicPosAt(c, 0.25)
	if math.Abs(float64(dst[3].X-at.X)) > 1e-3 || math.Abs(float64(dst[3].Y-at.Y)) > 1e-3 {
		t.Errorf("chop split %+v != eval %+v", dst[3], at)
	}
	// t == 1 special case.
	ChopCubicAt(c, dst, 1)
	if dst[3] != c[3] || dst[4] != c[3] || dst[5] != c[3] || dst[6] != c[3] {
		t.Error("t=1 chop should replicate the end point")
	}
	// Double chop matches sequential single chops on the shared point.
	dst10 := make([]Point, 10)
	ChopCubicAt2(c, dst10, 0.25, 0.75)
	if dst10[0] != c[0] || dst10[9] != c[3] {
		t.Error("double chop endpoints wrong")
	}
	at75 := EvalCubicPosAt(c, 0.75)
	if math.Abs(float64(dst10[6].X-at75.X)) > 1e-3 || math.Abs(float64(dst10[6].Y-at75.Y)) > 1e-3 {
		t.Errorf("double chop second split %+v != eval %+v", dst10[6], at75)
	}
}

func TestFindUnitQuadRoots(t *testing.T) {
	var roots [2]float32
	// (t - 0.25)(t - 0.75) = t^2 - t + 0.1875
	n := FindUnitQuadRoots(1, -1, 0.1875, &roots)
	if n != 2 || math.Abs(float64(roots[0])-0.25) > 1e-6 || math.Abs(float64(roots[1])-0.75) > 1e-6 {
		t.Errorf("roots = %v (n=%d)", roots, n)
	}
	// Roots outside (0,1) are rejected: (t-2)(t-3).
	n = FindUnitQuadRoots(1, -5, 6, &roots)
	if n != 0 {
		t.Errorf("out-of-range roots accepted: n=%d %v", n, roots)
	}
	// Linear case A == 0: t = 0.5.
	n = FindUnitQuadRoots(0, 2, -1, &roots)
	if n != 1 || roots[0] != 0.5 {
		t.Errorf("linear root = %v (n=%d)", roots, n)
	}
	// Double root at 0.5 collapses to one: (t-0.5)^2 = t^2 - t + 0.25.
	n = FindUnitQuadRoots(1, -1, 0.25, &roots)
	if n != 1 {
		t.Errorf("double root n = %d, want 1", n)
	}
	// Negative discriminant.
	n = FindUnitQuadRoots(1, 0, 1, &roots)
	if n != 0 {
		t.Errorf("imaginary roots accepted: n=%d", n)
	}
}

func TestFindExtrema(t *testing.T) {
	// Symmetric quad: extremum at t=0.5.
	if tv, ok := FindQuadExtrema(0, 100, 0); !ok || tv != 0.5 {
		t.Errorf("quad extrema = %v ok=%v", tv, ok)
	}
	// Monotonic: none.
	if _, ok := FindQuadExtrema(0, 50, 100); ok {
		t.Error("monotonic quad should have no extrema")
	}
	// Symmetric cubic y = (0, 100, 100, 0): dy/dt = 0 at t=0.5.
	var tv [2]float32
	n := FindCubicExtrema(0, 100, 100, 0, &tv)
	if n != 1 || math.Abs(float64(tv[0])-0.5) > 1e-6 {
		t.Errorf("cubic extrema = %v (n=%d)", tv, n)
	}
	// S-shaped cubic (0, 200, -100, 100) has two extrema.
	n = FindCubicExtrema(0, 200, -100, 100, &tv)
	if n != 2 || !(tv[0] > 0 && tv[0] < tv[1] && tv[1] < 1) {
		t.Errorf("s-cubic extrema = %v (n=%d)", tv, n)
	}
}

func TestConicEval(t *testing.T) {
	// Weight 1 must match the plain quad.
	q := []Point{{X: 0, Y: 0}, {X: 50, Y: 100}, {X: 100, Y: 0}}
	c := MakeConic(q[0], q[1], q[2], 1)
	for _, tt := range []float32{0, 0.3, 0.5, 0.8, 1} {
		got := c.EvalAt(tt)
		want := EvalQuadAt(q, tt)
		if math.Abs(float64(got.X-want.X)) > 1e-4 || math.Abs(float64(got.Y-want.Y)) > 1e-4 {
			t.Fatalf("conic w=1 EvalAt(%v)=%+v, quad=%+v", tt, got, want)
		}
	}
	// A 90-degree circular arc: (1,0)..(1,1)..(0,1) with w = sqrt(2)/2. Every point must be on the unit circle.
	arc := MakeConic(Pt(1, 0), Pt(1, 1), Pt(0, 1), ScalarRoot2Over2)
	for _, tt := range []float32{0, 0.25, 0.5, 0.75, 1} {
		p := arc.EvalAt(tt)
		radius := math.Hypot(float64(p.X), float64(p.Y))
		if math.Abs(radius-1) > 1e-6 {
			t.Fatalf("arc point %+v at t=%v has radius %g", p, tt, radius)
		}
	}
	// Midpoint of the arc is at 45 degrees.
	mid := arc.EvalAt(0.5)
	if math.Abs(float64(mid.X)-math.Sqrt2/2) > 1e-6 || math.Abs(float64(mid.Y)-math.Sqrt2/2) > 1e-6 {
		t.Errorf("arc midpoint = %+v", mid)
	}
}

func TestConicChop(t *testing.T) {
	arc := MakeConic(Pt(1, 0), Pt(1, 1), Pt(0, 1), ScalarRoot2Over2)
	var dst [2]Conic
	arc.Chop(&dst)
	// New weight is sqrt(0.5 + w/2).
	wantW := ScalarSqrt(0.5 + ScalarRoot2Over2*0.5)
	if dst[0].W != wantW || dst[1].W != wantW {
		t.Errorf("chop weights = %v %v, want %v", dst[0].W, dst[1].W, wantW)
	}
	// Shared point is the arc midpoint.
	mid := arc.EvalAt(0.5)
	if math.Abs(float64(dst[0].Pts[2].X-mid.X)) > 1e-6 || math.Abs(float64(dst[0].Pts[2].Y-mid.Y)) > 1e-6 {
		t.Errorf("chop midpoint %+v != eval %+v", dst[0].Pts[2], mid)
	}
	// Both halves stay on the unit circle.
	for _, half := range dst {
		for _, tt := range []float32{0.25, 0.5, 0.75} {
			p := half.EvalAt(tt)
			if r := math.Hypot(float64(p.X), float64(p.Y)); math.Abs(r-1) > 1e-5 {
				t.Fatalf("half-arc point %+v radius %g", p, r)
			}
		}
	}
	// ChopAt(t) agrees with EvalAt for the split point.
	if !arc.ChopAt(0.3, &dst) {
		t.Fatal("ChopAt failed")
	}
	at := arc.EvalAt(0.3)
	if math.Abs(float64(dst[0].Pts[2].X-at.X)) > 1e-5 || math.Abs(float64(dst[0].Pts[2].Y-at.Y)) > 1e-5 {
		t.Errorf("ChopAt split %+v != eval %+v", dst[0].Pts[2], at)
	}
}

func TestConicChopIntoQuads(t *testing.T) {
	arc := MakeConic(Pt(1, 0), Pt(1, 1), Pt(0, 1), ScalarRoot2Over2)
	// A loose tolerance keeps the whole conic as a single quad.
	if pow2 := arc.ComputeQuadPOW2(0.25); pow2 != 0 {
		t.Errorf("pow2 at tol 0.25 = %d, want 0", pow2)
	}
	// A tight tolerance forces subdivision.
	pow2 := arc.ComputeQuadPOW2(0.001)
	if pow2 < 2 || pow2 > MaxConicToQuadPOW2 {
		t.Fatalf("pow2 at tol 0.001 = %d", pow2)
	}
	pts := make([]Point, 1+2*(1<<MaxConicToQuadPOW2))
	n := arc.ChopIntoQuadsPOW2(pts, pow2)
	if n != 1<<pow2 {
		t.Fatalf("quad count = %d, want %d", n, 1<<pow2)
	}
	// Endpoints preserved.
	last := pts[2*n]
	if pts[0] != arc.Pts[0] || last != arc.Pts[2] {
		t.Errorf("quad chain endpoints %+v..%+v", pts[0], last)
	}
	// On-curve points (even indices) lie on the unit circle; control points (odd indices) sit slightly outside it,
	// bounded by the hull.
	for i := 0; i <= 2*n; i++ {
		r := math.Hypot(float64(pts[i].X), float64(pts[i].Y))
		if i%2 == 0 {
			if math.Abs(r-1) > 0.001 {
				t.Errorf("on-curve point %d = %+v radius %g", i, pts[i], r)
			}
		} else if r < 1 || r > 1.05 {
			t.Errorf("control point %d = %+v radius %g", i, pts[i], r)
		}
	}
	// The midpoint of each quad must approximate the conic within the requested tolerance (compare against the circle:
	// |radius - 1| <= tol with slack for float math).
	for q := 0; q < n; q++ {
		quad := pts[2*q : 2*q+3]
		mid := EvalQuadAt(quad, 0.5)
		if r := math.Hypot(float64(mid.X), float64(mid.Y)); math.Abs(r-1) > 0.002 {
			t.Errorf("quad %d midpoint radius %g", q, r)
		}
	}
}

func TestConicExtremaAndTightBounds(t *testing.T) {
	// Half circle from (1,0) through (0,1) to (-1,0) built from two 90-degree conics; use the single 90-degree arc for
	// extrema: x extremum should not exist in (0,1)... use an arc from (0,-1) to (0,1) through (1,0): x has an extremum
	// at t=0.5 where x=1.
	arc := MakeConic(Pt(0, -1), Pt(2, 0), Pt(0, 1), ScalarRoot2Over2)
	// This is not a unit-circle arc (control point 2,0 with w=root2/2 gives x max sqrt(2)-ish); just verify extrema
	// location and bounds consistency.
	tv, ok := arc.FindXExtrema()
	if !ok || math.Abs(float64(tv)-0.5) > 1e-5 {
		t.Errorf("x extrema = %v ok=%v", tv, ok)
	}
	bounds := arc.ComputeTightBounds()
	// Bounds must contain both endpoints and the extremum point.
	ext := arc.EvalAt(tv)
	if !(bounds.Left <= 0 && bounds.Right >= ext.X && bounds.Top <= -1 && bounds.Bottom >= 1) {
		t.Errorf("tight bounds %+v does not cover arc (ext %+v)", bounds, ext)
	}
	fast := arc.ComputeFastBounds()
	if !fast.ContainsRect(bounds) {
		t.Errorf("fast bounds %+v should contain tight bounds %+v", fast, bounds)
	}
	// ChopAtXExtrema flattens the middle at the extremum.
	var dst [2]Conic
	if arc.ChopAtXExtrema(&dst) {
		if dst[0].Pts[2].X != dst[0].Pts[1].X || dst[1].Pts[0].X != dst[1].Pts[1].X {
			t.Error("ChopAtXExtrema did not flatten the middle")
		}
	} else {
		t.Error("ChopAtXExtrema failed")
	}
}

func TestBuildUnitArc(t *testing.T) {
	var conics [MaxConicsForArc]Conic
	// Quarter turn from (1,0) to (0,1), clockwise (y-down: CW goes toward +y).
	n := BuildUnitArc(Pt(1, 0), Pt(0, 1), RotationCW, nil, &conics)
	if n != 1 {
		t.Fatalf("quarter arc conic count = %d", n)
	}
	end := conics[n-1].Pts[2]
	if math.Abs(float64(end.X)) > 1e-6 || math.Abs(float64(end.Y)-1) > 1e-6 {
		t.Errorf("arc end = %+v, want (0,1)", end)
	}
	// All sample points on the unit circle.
	for i := 0; i < n; i++ {
		for _, tt := range []float32{0.25, 0.5, 0.75} {
			p := conics[i].EvalAt(tt)
			if r := math.Hypot(float64(p.X), float64(p.Y)); math.Abs(r-1) > 1e-5 {
				t.Fatalf("arc point %+v radius %g", p, r)
			}
		}
	}
	// Full 270-degree sweep uses 3 quadrants + remainder.
	n = BuildUnitArc(Pt(1, 0), Pt(0, -1), RotationCW, nil, &conics)
	if n != 3 {
		t.Errorf("270-degree arc conic count = %d, want 3", n)
	}
	// Coincident start/stop with CW direction yields zero conics.
	n = BuildUnitArc(Pt(1, 0), Pt(1, 0), RotationCW, nil, &conics)
	if n != 0 {
		t.Errorf("degenerate arc count = %d", n)
	}
	// User matrix is applied: scale by 2 doubles the radius.
	sm := ScaleMatrix(2, 2)
	n = BuildUnitArc(Pt(1, 0), Pt(0, 1), RotationCW, &sm, &conics)
	if n != 1 {
		t.Fatalf("scaled arc count = %d", n)
	}
	p := conics[0].EvalAt(0.5)
	if r := math.Hypot(float64(p.X), float64(p.Y)); math.Abs(r-2) > 1e-5 {
		t.Errorf("scaled arc radius = %g, want 2", r)
	}
}
