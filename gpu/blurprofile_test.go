// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gpu

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func TestNextPow2(t *testing.T) {
	cases := map[int]int{
		-5: 1, 0: 1, 1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 16: 16, 17: 32, 32: 32, 33: 64, 200: 256,
	}
	for in, want := range cases {
		if got := nextPow2(in); got != want {
			t.Errorf("nextPow2(%d) = %d, want %d", in, got, want)
		}
	}
}

func TestUnitScalarClampToByte(t *testing.T) {
	cases := []struct {
		in   float32
		want uint8
	}{
		{in: -1, want: 0},
		{in: 0, want: 0},
		{in: 0.5, want: 128},
		{in: 1, want: 255},
		{in: 2, want: 255},
		// (U8)(x*255 + 0.5) truncates the biased value.
		{in: 0.999, want: 255}, // 254.7 + 0.5 = 255.2 -> 255
		{in: 0.001, want: 0},   // 0.255 + 0.5 = 0.755 -> 0
	}
	for _, c := range cases {
		if got := unitScalarClampToByte(c.in); got != c.want {
			t.Errorf("unitScalarClampToByte(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestComputeIntegralTableWidth(t *testing.T) {
	cases := []struct {
		sixSigma float32
		want     int
	}{
		{sixSigma: 0, want: 32},    // minWidth 0 -> nextPow2 1 -> max(1,32) = 32
		{sixSigma: 6, want: 32},    // minWidth 12 -> nextPow2 16 -> 32
		{sixSigma: 16, want: 32},   // minWidth 32 -> nextPow2 32 -> 32
		{sixSigma: 17, want: 64},   // minWidth 34 -> nextPow2 64 -> 64
		{sixSigma: 30, want: 64},   // minWidth 60 -> nextPow2 64 -> 64
		{sixSigma: 100, want: 256}, // minWidth 200 -> nextPow2 256
	}
	for _, c := range cases {
		if got := ComputeIntegralTableWidth(c.sixSigma); got != c.want {
			t.Errorf("ComputeIntegralTableWidth(%v) = %d, want %d", c.sixSigma, got, c.want)
		}
	}
	if got := ComputeIntegralTableWidth(float32(math.NaN())); got != 0 {
		t.Errorf("ComputeIntegralTableWidth(NaN) = %d, want 0", got)
	}
	if got := ComputeIntegralTableWidth(float32(math.Inf(1))); got != 0 {
		t.Errorf("ComputeIntegralTableWidth(+Inf) = %d, want 0", got)
	}
}

func TestCreateIntegralTable(t *testing.T) {
	if CreateIntegralTable(0) != nil {
		t.Fatal("CreateIntegralTable(0) should be nil")
	}
	const width = 64
	table := CreateIntegralTable(width)
	if len(table) != width {
		t.Fatalf("len = %d, want %d", len(table), width)
	}
	if table[0] != 255 {
		t.Errorf("table[0] = %d, want 255", table[0])
	}
	if table[width-1] != 0 {
		t.Errorf("table[%d] = %d, want 0", width-1, table[width-1])
	}
	// Monotonically non-increasing (the integral of the normal distribution runs one way).
	for i := 1; i < width; i++ {
		if table[i] > table[i-1] {
			t.Fatalf("table not monotone at %d: %d > %d", i, table[i], table[i-1])
		}
	}
	// The value where the integral argument crosses 0 (x = (i+0.5)/width ~ 0.5) is ~0.5*255.
	mid := table[width/2]
	if mid < 120 || mid > 135 {
		t.Errorf("mid value %d not near 127/128", mid)
	}
	// Cross-check one entry against the closed-form formula.
	i := 20
	x := (float32(i) + 0.5) / width
	x = (-6*x + 3) * geom.ScalarRoot2Over2
	integral := 0.5 * (float32(math.Erf(float64(x))) + 1)
	want := byte(geom.RoundToInt(255 * integral))
	if table[i] != want {
		t.Errorf("table[%d] = %d, want %d", i, table[i], want)
	}
}

func TestCreateHalfPlaneProfile(t *testing.T) {
	if CreateHalfPlaneProfile(15) != nil {
		t.Fatal("odd width should be nil")
	}
	const width = 512
	p := CreateHalfPlaneProfile(width)
	if len(p) != width {
		t.Fatalf("len = %d, want %d", len(p), width)
	}
	if p[width-1] != 0 {
		t.Errorf("tail = %d, want 0", p[width-1])
	}
	// The half-plane coverage ramps from ~full at the left to 0 at the right: non-increasing.
	for i := 1; i < width; i++ {
		if p[i] > p[i-1] {
			t.Fatalf("profile not monotone at %d: %d > %d", i, p[i], p[i-1])
		}
	}
	if p[0] < 250 {
		t.Errorf("left edge %d, want near full coverage", p[0])
	}
}

func TestCreateCircleProfile(t *testing.T) {
	const width = 512
	// A moderately large circle relative to sigma: coverage full at center, 0 at the tail.
	p := CreateCircleProfile(4, 40, width)
	if len(p) != width {
		t.Fatalf("len = %d, want %d", len(p), width)
	}
	if p[width-1] != 0 {
		t.Errorf("tail = %d, want 0", p[width-1])
	}
	// Non-increasing from center outward.
	for i := 1; i < width; i++ {
		if p[i] > p[i-1] {
			t.Fatalf("profile not monotone at %d: %d > %d", i, p[i], p[i-1])
		}
	}
	if p[0] < 250 {
		t.Errorf("center %d, want near full coverage", p[0])
	}
}

func TestComputeBlurredRRectParams(t *testing.T) {
	// A 100x100 rrect with 20px circular corners and a small blur: nine-patchable.
	dev := geom.MakeRRect(geom.RectLTRB(0, 0, 100, 100), 20, 20)
	if dev.Type != geom.RRectSimple {
		t.Fatalf("expected a simple rrect, got type %d", dev.Type)
	}
	rrectToDraw, dims, ok := ComputeBlurredRRectParams(dev, dev, 3, 3)
	if !ok {
		t.Fatal("expected nine-patchable")
	}
	// devBlurRadius = 3*ceil(3 - 1/6) = 9; newRR = 20+20+18+1 = 59; dims = 59+18 = 77 (odd).
	if dims.Width != 77 || dims.Height != 77 {
		t.Errorf("dims = %dx%d, want 77x77", dims.Width, dims.Height)
	}
	if dims.Width%2 == 0 || dims.Height%2 == 0 {
		t.Errorf("nine-patch dims must be odd, got %dx%d", dims.Width, dims.Height)
	}
	if rrectToDraw.Type != geom.RRectSimple {
		t.Errorf("rrectToDraw type = %d, want simple", rrectToDraw.Type)
	}
	if rrectToDraw.RadiusX != 20 || rrectToDraw.RadiusY != 20 {
		t.Errorf("rrectToDraw radii = (%v,%v), want (20,20)", rrectToDraw.RadiusX, rrectToDraw.RadiusY)
	}
	// newRect = XYWH(9, 9, 59, 59).
	if r := rrectToDraw.Rect; r.Left != 9 || r.Top != 9 || r.Right != 68 || r.Bottom != 68 {
		t.Errorf("rrectToDraw.Rect = %+v, want (9,9,68,68)", r)
	}

	// Too much blur relative to the corners/extent: not nine-patchable (blurred corners overlap).
	small := geom.MakeRRect(geom.RectLTRB(0, 0, 40, 40), 18, 18)
	if _, _, ok = ComputeBlurredRRectParams(small, small, 5, 5); ok {
		t.Error("expected not nine-patchable for a small rrect under a large blur")
	}
}

func TestCreateRRectBlurMask(t *testing.T) {
	dev := geom.MakeRRect(geom.RectLTRB(0, 0, 100, 100), 20, 20)
	const sigma = 3
	rrectToDraw, dims, ok := ComputeBlurredRRectParams(dev, dev, sigma, sigma)
	if !ok {
		t.Fatal("expected nine-patchable")
	}
	w, h := int(dims.Width), int(dims.Height)
	mask := CreateRRectBlurMask(rrectToDraw, dims, sigma)
	if len(mask) != w*h {
		t.Fatalf("mask len = %d, want %d", len(mask), w*h)
	}
	at := func(x, y int) int { return int(mask[y*w+x]) }

	// Fully symmetric about both axes (the upper-left quadrant is computed then mirrored).
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if v, m := at(x, y), at(w-1-x, y); v != m {
				t.Fatalf("not x-symmetric at (%d,%d): %d vs %d", x, y, v, m)
			}
			if v, m := at(x, y), at(x, h-1-y); v != m {
				t.Fatalf("not y-symmetric at (%d,%d): %d vs %d", x, y, v, m)
			}
		}
	}

	// Center is deep inside the rrect: opaque.
	if c := at(w/2, h/2); c < 250 {
		t.Errorf("center coverage %d, want ~255", c)
	}
	// The mask's own corner is well outside the blurred rrect: transparent.
	if c := at(0, 0); c > 4 {
		t.Errorf("mask corner coverage %d, want ~0", c)
	}
	// Along the center column, coverage rises monotonically from the top edge to the center (crossing the blurred top
	// edge), confirming a Gaussian falloff rather than a hard edge.
	cx := w / 2
	for y := 1; y <= h/2; y++ {
		if at(cx, y) < at(cx, y-1) {
			t.Fatalf("center column not monotone into the interior at y=%d: %d < %d", y, at(cx, y),
				at(cx, y-1))
		}
	}
}
