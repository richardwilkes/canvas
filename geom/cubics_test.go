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
	"testing"
)

func TestQuadRootsReal(t *testing.T) {
	var roots [2]float64
	// (x-2)(x-3) = x^2 -5x + 6
	n := QuadRootsReal(1, -5, 6, &roots)
	if n != 2 {
		t.Fatalf("expected 2 roots, got %d", n)
	}
	got := [2]float64{math.Min(roots[0], roots[1]), math.Max(roots[0], roots[1])}
	if math.Abs(got[0]-2) > 1e-12 || math.Abs(got[1]-3) > 1e-12 {
		t.Errorf("roots = %v", got)
	}
	// double root
	n = QuadRootsReal(1, -2, 1, &roots)
	if n != 1 || math.Abs(roots[0]-1) > 1e-12 {
		t.Errorf("double root: n=%d roots=%v", n, roots)
	}
	// linear
	n = QuadRootsReal(0, 2, -4, &roots)
	if n != 1 || roots[0] != 2 {
		t.Errorf("linear: n=%d roots=%v", n, roots)
	}
	// no real roots
	if n = QuadRootsReal(1, 0, 1, &roots); n != 0 {
		t.Errorf("x^2+1: n=%d", n)
	}
}

func TestCubicRootsRealAndValidT(t *testing.T) {
	var roots [3]float64
	// (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
	n := CubicRootsReal(1, -6, 11, -6, &roots)
	if n != 3 {
		t.Fatalf("expected 3 roots, got %d (%v)", n, roots)
	}
	found := [3]bool{}
	for i := 0; i < n; i++ {
		for w := 1; w <= 3; w++ {
			if math.Abs(roots[i]-float64(w)) < 1e-9 {
				found[w-1] = true
			}
		}
	}
	if !found[0] || !found[1] || !found[2] {
		t.Errorf("missing roots: %v", roots)
	}

	// Valid-T pinning: (x-0.5)(x^2+1) has one real root at 0.5.
	n = CubicRootsValidT(1, -0.5, 1, -0.5, &roots)
	if n != 1 || math.Abs(roots[0]-0.5) > 1e-9 {
		t.Errorf("validT: n=%d roots=%v", n, roots)
	}
	// Roots outside [0,1] are dropped.
	n = CubicRootsValidT(1, -6, 11, -6, &roots)
	if n != 1 || roots[0] != 1 { // root at 1 kept (pinned), 2 and 3 dropped
		t.Errorf("validT drop: n=%d roots=%v", n, roots)
	}
}

func TestCubicBinarySearchRootsValidT(t *testing.T) {
	var roots [3]float64
	// f(t) = t^3 - 0.5^3 has a root at 0.5.
	n := CubicBinarySearchRootsValidT(1, 0, 0, -0.125, &roots)
	if n < 1 {
		t.Fatal("no roots found")
	}
	ok := false
	for i := 0; i < n; i++ {
		if math.Abs(roots[i]-0.5) < 1e-4 {
			ok = true
		}
	}
	if !ok {
		t.Errorf("roots = %v, want one near 0.5", roots[:n])
	}
}

func TestChopMonoCubicAtY(t *testing.T) {
	src := []Point{{X: 0, Y: 0}, {X: 10, Y: 5}, {X: 20, Y: 10}, {X: 30, Y: 15}}
	var dst [7]Point
	if !ChopMonoCubicAtY(src, dst[:], 7.5) {
		t.Fatal("chop failed")
	}
	if dst[0] != src[0] || dst[6] != src[3] {
		t.Error("endpoints not preserved")
	}
	if ScalarAbs(dst[3].Y-7.5) > 1e-4 {
		t.Errorf("chop point Y = %v, want 7.5", dst[3].Y)
	}
	// Both halves must remain monotonic in Y.
	for i := 1; i < 4; i++ {
		if dst[i].Y < dst[i-1].Y || dst[i+3].Y < dst[i+2].Y {
			t.Errorf("non-monotonic result: %v", dst)
		}
	}
}

func TestChopMonoCubicAtX(t *testing.T) {
	src := []Point{{X: 0, Y: 0}, {X: 5, Y: 10}, {X: 10, Y: 20}, {X: 15, Y: 30}}
	var dst [7]Point
	if !ChopMonoCubicAtX(src, dst[:], 3) {
		t.Fatal("chop failed")
	}
	if ScalarAbs(dst[3].X-3) > 1e-4 {
		t.Errorf("chop point X = %v, want 3", dst[3].X)
	}
}

func TestChopQuadAtXExtrema(t *testing.T) {
	// X goes 0 → 10 → 0: one X extremum.
	src := []Point{{X: 0, Y: 0}, {X: 10, Y: 5}, {X: 0, Y: 10}}
	var dst [5]Point
	if n := ChopQuadAtXExtrema(src, dst[:]); n != 1 {
		t.Fatalf("n = %d, want 1", n)
	}
	// The shared point's neighbors must have exactly its X (flattened).
	if dst[1].X != dst[2].X || dst[3].X != dst[2].X {
		t.Errorf("extrema not flattened: %v", dst)
	}
	// Monotonic X in each half.
	if dst[1].X < dst[0].X || dst[2].X < dst[1].X {
		t.Errorf("first half not monotonic: %v", dst[:3])
	}
	if dst[3].X > dst[2].X || dst[4].X > dst[3].X {
		t.Errorf("second half not monotonic: %v", dst[2:])
	}
}

func TestChopCubicAtXExtrema(t *testing.T) {
	// X: 0 → 20 → -10 → 10 has two X extrema.
	src := []Point{{X: 0, Y: 0}, {X: 20, Y: 3}, {X: -10, Y: 6}, {X: 10, Y: 9}}
	var dst [10]Point
	n := ChopCubicAtXExtrema(src, dst[:])
	if n != 2 {
		t.Fatalf("n = %d, want 2", n)
	}
	if dst[2].X != dst[3].X || dst[4].X != dst[3].X || dst[5].X != dst[6].X || dst[7].X != dst[6].X {
		t.Errorf("extrema not flattened: %v", dst)
	}
}
