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

func TestRoundToIntDoublePath(t *testing.T) {
	// These cases catch a subtle rounding bug: naive floorf(x + 0.5f) computed in float32 gives answers one too high
	// because x+0.5 is not representable and rounds up.
	cases := []struct {
		in   float32
		want int32
	}{
		{in: 0.49999997, want: 0},
		{in: 16777215, want: 16777215}, // 2^24 - 1: float32 16777215 + 0.5 would round to 2^24 in float math
		{in: 16777216, want: 16777216}, // 2^24
		{in: 0.5, want: 1},
		{in: -0.5, want: 0}, // floor(-0.5 + 0.5) = 0; rounds half toward +inf
		{in: -1.5, want: -1},
		{in: 2.5, want: 3},
		{in: -2.5, want: -2},
		{in: 0, want: 0},
	}
	for _, c := range cases {
		if got := RoundToInt(c.in); got != c.want {
			t.Errorf("RoundToInt(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestRoundToIntSaturation(t *testing.T) {
	// Rounding saturates to maxS32FitsInFloat (2147483520), not MaxInt32, and maps NaN to the positive bound.
	nan := float32(math.NaN())
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	if got := RoundToInt(nan); got != maxS32FitsInFloat {
		t.Errorf("RoundToInt(NaN) = %d, want maxS32FitsInFloat", got)
	}
	if got := RoundToInt(posInf); got != maxS32FitsInFloat {
		t.Errorf("RoundToInt(+inf) = %d, want maxS32FitsInFloat", got)
	}
	if got := RoundToInt(negInf); got != minS32FitsInFloat {
		t.Errorf("RoundToInt(-inf) = %d, want minS32FitsInFloat", got)
	}
	if got := RoundToInt(3e9); got != maxS32FitsInFloat {
		t.Errorf("RoundToInt(3e9) = %d, want maxS32FitsInFloat", got)
	}
	if got := RoundToInt(-3e9); got != minS32FitsInFloat {
		t.Errorf("RoundToInt(-3e9) = %d, want minS32FitsInFloat", got)
	}
	if got := RoundToInt(float32(math.MaxInt32)); got != maxS32FitsInFloat {
		t.Errorf("RoundToInt(float32(MaxInt32)) = %d, want maxS32FitsInFloat", got)
	}
	if got := RoundToInt(float32(math.MinInt32)); got != minS32FitsInFloat {
		t.Errorf("RoundToInt(float32(MinInt32)) = %d, want minS32FitsInFloat", got)
	}
	if got := FloorToInt(nan); got != maxS32FitsInFloat {
		t.Errorf("FloorToInt(NaN) = %d, want maxS32FitsInFloat (Skia NaN convention)", got)
	}
	if got := CeilToInt(negInf); got != minS32FitsInFloat {
		t.Errorf("CeilToInt(-inf) = %d, want minS32FitsInFloat", got)
	}
}

func TestIsFinite(t *testing.T) {
	if !IsFinite(1, -2, 0, 3.4e38) {
		t.Error("finite values reported non-finite")
	}
	if IsFinite(float32(math.NaN())) {
		t.Error("NaN reported finite")
	}
	if IsFinite(1, float32(math.Inf(1))) {
		t.Error("+inf reported finite")
	}
	if IsFinite(float32(math.Inf(-1)), 5) {
		t.Error("-inf reported finite")
	}
	if !IsFinite() {
		t.Error("empty argument list should be finite")
	}
}

func TestScalarNearlyZero(t *testing.T) {
	if !ScalarNearlyZero(0) || !ScalarNearlyZero(1.0/4096) || ScalarNearlyZero(1.1/4096) {
		t.Error("ScalarNearlyZero boundary behavior wrong")
	}
}

func TestSinCosSnapToZero(t *testing.T) {
	// 180 degrees: sin should snap to exactly 0.
	rad := DegreesToRadians(180)
	if got := ScalarSinSnapToZero(rad); got != 0 {
		t.Errorf("sin(pi) did not snap to zero: %g", got)
	}
	// 90 degrees: cos should snap to exactly 0.
	rad = DegreesToRadians(90)
	if got := ScalarCosSnapToZero(rad); got != 0 {
		t.Errorf("cos(pi/2) did not snap to zero: %g", got)
	}
	if got := ScalarSinSnapToZero(rad); got != 1 {
		t.Errorf("sin(pi/2) = %g, want 1", got)
	}
}

func TestScalarAs2sCompliment(t *testing.T) {
	if scalarAs2sCompliment(0) != 0 {
		t.Error("0 should map to 0")
	}
	if scalarAs2sCompliment(float32(math.Copysign(0, -1))) != 0 {
		t.Error("-0.0 should map to 0")
	}
	one := scalarAs2sCompliment(1)
	if one != scalar1Int {
		t.Errorf("1.0 mapped to %#x, want %#x", one, scalar1Int)
	}
	if scalarAs2sCompliment(-1) != -scalar1Int {
		t.Error("-1.0 should be the negation of 1.0's mapping")
	}
}
