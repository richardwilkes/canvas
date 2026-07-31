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

// FindQuadMaxCurvature: for a symmetric quad the max curvature is at t=0.5; for degenerate control placement it pins to
// the [0,1] ends.
func TestFindQuadMaxCurvature(t *testing.T) {
	sym := []Point{{X: 0, Y: 0}, {X: 50, Y: 100}, {X: 100, Y: 0}}
	if got := FindQuadMaxCurvature(sym); got != 0.5 {
		t.Fatalf("symmetric quad max curvature %v want 0.5", got)
	}
	// control point coincides with the start: numer <= 0 → 0
	deg := []Point{{X: 0, Y: 0}, {X: 0, Y: 0}, {X: 100, Y: 100}}
	if got := FindQuadMaxCurvature(deg); got != 0 {
		t.Fatalf("degenerate quad max curvature %v want 0", got)
	}
}

// FindCubicInflections: an S-curve has one inflection; verify the cross product of first and second derivatives
// vanishes there.
func TestFindCubicInflections(t *testing.T) {
	cubic := []Point{{X: 0, Y: 0}, {X: 30, Y: 60}, {X: 70, Y: -60}, {X: 100, Y: 0}}
	var ts [2]float32
	n := FindCubicInflections(cubic, &ts)
	if n != 1 {
		t.Fatalf("inflection count %d want 1", n)
	}
	d1 := evalCubicDerivative(cubic, ts[0])
	d2 := EvalCubicCurvatureAt(cubic, ts[0])
	cross := float64(d1.X)*float64(d2.Y) - float64(d1.Y)*float64(d2.X)
	if math.Abs(cross) > 1e-2*float64(d1.Length())*float64(d2.Length()) {
		t.Fatalf("curvature at inflection t=%v not ~0 (cross %v)", ts[0], cross)
	}
	line := []Point{{X: 0, Y: 0}, {X: 10, Y: 10}, {X: 20, Y: 20}, {X: 30, Y: 30}}
	if n = FindCubicInflections(line, &ts); n != 0 {
		t.Fatalf("collinear cubic inflections %d want 0", n)
	}
}

// FindCubicCusp: the classic crossing-control-segments cusp is found; smooth cubics and coincident end/control cases
// return -1.
func TestFindCubicCusp(t *testing.T) {
	cusp := []Point{{X: 20, Y: 20}, {X: 80, Y: 80}, {X: 20, Y: 80}, {X: 80, Y: 20}}
	ct := FindCubicCusp(cusp)
	if ct <= 0 || ct >= 1 {
		t.Fatalf("cusp t %v want in (0,1)", ct)
	}
	// the derivative at the cusp is (near) zero relative to the cubic's size
	d := evalCubicDerivative(cusp, ct)
	if d.LengthSqd() >= calcCubicPrecision(cusp) {
		t.Fatalf("derivative at cusp t=%v not near zero: %v", ct, d)
	}
	smooth := []Point{{X: 0, Y: 0}, {X: 30, Y: 60}, {X: 70, Y: 60}, {X: 100, Y: 0}}
	if got := FindCubicCusp(smooth); got != -1 {
		t.Fatalf("smooth cubic cusp %v want -1", got)
	}
	coincident := []Point{{X: 0, Y: 0}, {X: 0, Y: 0}, {X: 70, Y: 60}, {X: 100, Y: 0}}
	if got := FindCubicCusp(coincident); got != -1 {
		t.Fatalf("coincident-control cubic cusp %v want -1 (skipped by design)", got)
	}
}

// ComputeResScaleForStroking: max column length of the upper 2x2; degenerate/non-finite → 1.
func TestComputeResScaleForStroking(t *testing.T) {
	var m Matrix
	m.SetScale(3, 2)
	if got := m.ComputeResScaleForStroking(); got != 3 {
		t.Fatalf("scale res %v want 3", got)
	}
	m.SetAll(0, 0, 5, 0, 0, 9, 0, 0, 1) // zero upper 2x2
	if got := m.ComputeResScaleForStroking(); got != 1 {
		t.Fatalf("degenerate res %v want 1", got)
	}
	m.SetAll(float32(math.Inf(1)), 0, 0, 0, 1, 0, 0, 0, 1)
	if got := m.ComputeResScaleForStroking(); got != 1 {
		t.Fatalf("non-finite res %v want 1", got)
	}
	var rot Matrix
	rot.SetRotatePivot(90, 0, 0)
	got := rot.ComputeResScaleForStroking()
	if math.Abs(float64(got)-1) > 1e-6 {
		t.Fatalf("rotation res %v want ~1", got)
	}
}

// ChopQuadAtMaxCurvature: a symmetric quad chops at t=0.5 into two quads sharing the on-curve midpoint; a colinear quad
// (no interior max-curvature t) copies through unchopped.
func TestChopQuadAtMaxCurvature(t *testing.T) {
	src := []Point{{X: 0, Y: 0}, {X: 10, Y: 10}, {X: 20, Y: 0}}
	var dst [5]Point
	if got := ChopQuadAtMaxCurvature(src, dst[:]); got != 2 {
		t.Fatalf("symmetric quad chop count %v want 2", got)
	}
	if dst[0] != src[0] || dst[4] != src[2] {
		t.Fatalf("chopped endpoints %v %v want %v %v", dst[0], dst[4], src[0], src[2])
	}
	if math.Abs(float64(dst[2].X)-10) > 1e-4 || math.Abs(float64(dst[2].Y)-5) > 1e-4 {
		t.Fatalf("shared midpoint %v want (10, 5)", dst[2])
	}
	colinear := []Point{{X: 0, Y: 0}, {X: 10, Y: 10}, {X: 20, Y: 20}}
	if got := ChopQuadAtMaxCurvature(colinear, dst[:]); got != 1 {
		t.Fatalf("colinear quad chop count %v want 1", got)
	}
	if dst[0] != colinear[0] || dst[1] != colinear[1] || dst[2] != colinear[2] {
		t.Fatalf("colinear quad not copied through: %v", dst[:3])
	}
}

// ChopCubicAtInflections: an S-curve has one inflection (2 cubics with connected endpoints); a single-bulge cubic has
// none (copied through).
func TestChopCubicAtInflections(t *testing.T) {
	sCurve := []Point{{X: 0, Y: 0}, {X: 10, Y: 20}, {X: 20, Y: -20}, {X: 30, Y: 0}}
	var dst [10]Point
	if got := ChopCubicAtInflections(sCurve, dst[:]); got != 2 {
		t.Fatalf("s-curve inflection chop count %v want 2", got)
	}
	if dst[0] != sCurve[0] || dst[6] != sCurve[3] {
		t.Fatalf("chopped endpoints %v %v want %v %v", dst[0], dst[6], sCurve[0], sCurve[3])
	}
	bulge := []Point{{X: 0, Y: 0}, {X: 10, Y: -10}, {X: 20, Y: -10}, {X: 30, Y: 0}}
	if got := ChopCubicAtInflections(bulge, dst[:]); got != 1 {
		t.Fatalf("bulge cubic chop count %v want 1", got)
	}
	for i := range 4 {
		if dst[i] != bulge[i] {
			t.Fatalf("bulge cubic not copied through: %v", dst[:4])
		}
	}
}

func TestFindQuadMaxCurvatureNaN(t *testing.T) {
	// Documented contract: the result is in [0, 1] for finite input, but a NaN coordinate makes numer and denom NaN,
	// which fails every comparison and falls through to return NaN. Matches Skia, whose assert admits NaN.
	nan := float32(math.NaN())
	for _, src := range [][]Point{
		{{X: nan, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 0}},
		{{X: 0, Y: 0}, {X: nan, Y: 1}, {X: 2, Y: 0}},
		{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: nan}},
	} {
		if got := FindQuadMaxCurvature(src); !math.IsNaN(float64(got)) {
			t.Errorf("FindQuadMaxCurvature(%v) = %v, want NaN", src, got)
		}
	}
	// Finite input stays pinned to the unit range.
	rng := rand.New(rand.NewSource(5))
	for range 50000 {
		src := []Point{
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
			{X: rng.Float32()*200 - 100, Y: rng.Float32()*200 - 100},
		}
		if got := FindQuadMaxCurvature(src); got < 0 || got > 1 {
			t.Fatalf("FindQuadMaxCurvature(%v) = %v, want within [0, 1]", src, got)
		}
	}
}

func TestCalcCubicPrecisionScalesAsSquare(t *testing.T) {
	// Documented contract: the tolerance is proportional to the *square* of the cubic's dimensions, so scaling the
	// control polygon by k scales the result by k^2. A caller comparing it against an unsquared magnitude is wrong.
	src := []Point{{X: 0, Y: 0}, {X: 10, Y: 20}, {X: 30, Y: 5}, {X: 40, Y: 25}}
	base := calcCubicPrecision(src)
	if base <= 0 {
		t.Fatalf("base = %v, want positive", base)
	}
	for _, k := range []float32{2, 3, 10} {
		scaled := make([]Point, len(src))
		for i, p := range src {
			scaled[i] = Point{X: p.X * k, Y: p.Y * k}
		}
		got := calcCubicPrecision(scaled)
		want := base * k * k
		if diff := math.Abs(float64(got - want)); diff > float64(want)*1e-5 {
			t.Errorf("calcCubicPrecision scaled by %v = %v, want %v (k^2 growth, not k)", k, got, want)
		}
	}
}
