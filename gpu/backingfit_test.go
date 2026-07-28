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

// TestApproxSizeAdjust covers the rounding rules and, above 2^30, that the search for the ceiling power of 2 terminates
// (the int32 shift overflows to a negative value and then to 0, which used to spin forever) while still returning a size
// at least as large as the request.
func TestApproxSizeAdjust(t *testing.T) {
	const maxPow2 = int32(1) << 30
	for _, c := range []struct {
		value int32
		want  int32
	}{
		{value: -5, want: 16}, // below the minimum
		{value: 0, want: 16},  // below the minimum
		{value: 16, want: 16}, // already a power of 2
		{value: 17, want: 32}, // rounds up to a power of 2 below magicTol
		{value: 1024, want: 1024},
		{value: 1025, want: 1536}, // above magicTol: the midpoint between 1024 and 2048
		{value: 1600, want: 2048}, // above the midpoint: the ceiling power of 2
		{value: maxPow2, want: maxPow2},
		{value: maxPow2 + 1, want: maxPow2 + maxPow2>>1},          // the midpoint is still representable
		{value: maxPow2 + maxPow2>>1, want: maxPow2 + maxPow2>>1}, // exactly the midpoint
		{value: maxPow2 + maxPow2>>1 + 1, want: math.MaxInt32},    // the ceiling power of 2 is not representable
		{value: math.MaxInt32, want: math.MaxInt32},
	} {
		if got := approxSizeAdjust(c.value); got != c.want {
			t.Errorf("approxSizeAdjust(%d) = %d, want %d", c.value, got, c.want)
		}
	}
	// The approx size may never come out below the request, at any magnitude.
	for _, v := range []int32{1, 15, 33, 1025, maxPow2 - 1, maxPow2 + 1, 2000000000, math.MaxInt32} {
		if got := approxSizeAdjust(v); got < v {
			t.Errorf("approxSizeAdjust(%d) = %d, want >= %d", v, got, v)
		}
	}
	if got := GetApproxSize(geom.ISize{Width: 1025, Height: math.MaxInt32}); got !=
		(geom.ISize{Width: 1536, Height: math.MaxInt32}) {
		t.Errorf("GetApproxSize = %v, want {1536 %d}", got, int32(math.MaxInt32))
	}
}
