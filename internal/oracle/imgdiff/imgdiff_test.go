// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imgdiff

import (
	"image/color"
	"testing"
)

// TestHeatColorBandsAreInclusiveUpperBounds pins the ramp to what Heatmap's doc says. Every band's threshold is its
// inclusive upper bound, so delta 64 is the last orange value and red starts at 65 — the doc used to claim red meant
// "delta ≥ 64", which is off by one at exactly the boundary a reader would check.
func TestHeatColorBandsAreInclusiveUpperBounds(t *testing.T) {
	var (
		black  = color.NRGBA{A: 0xFF}
		blue   = color.NRGBA{B: 0xFF, G: 0x60, A: 0xFF}
		cyan   = color.NRGBA{R: 0x30, G: 0xC0, B: 0xFF, A: 0xFF}
		yellow = color.NRGBA{R: 0xFF, G: 0xD0, A: 0xFF}
		orange = color.NRGBA{R: 0xFF, G: 0x80, A: 0xFF}
		red    = color.NRGBA{R: 0xFF, A: 0xFF}
	)
	for _, tc := range []struct {
		name  string
		want  color.NRGBA
		delta uint8
	}{
		{name: "identical", delta: 0, want: black},
		{name: "smallest visible", delta: 1, want: blue},
		{name: "top of the blue band", delta: 2, want: blue},
		{name: "first cyan", delta: 3, want: cyan},
		{name: "top of the cyan band", delta: 8, want: cyan},
		{name: "top of the yellow band", delta: 32, want: yellow},
		{name: "first orange", delta: 33, want: orange},
		{name: "top of the orange band", delta: 64, want: orange},
		{name: "first red", delta: 65, want: red},
		{name: "saturated", delta: 255, want: red},
	} {
		if got := heatColor(tc.delta); got != tc.want {
			t.Errorf("%s: heatColor(%d) = %v, want %v", tc.name, tc.delta, got, tc.want)
		}
	}
}

// TestCompareCounts checks the three counters a golden gate reads: pixels differing at all, pixels differing beyond the
// profile's channel tolerance, and the largest channel delta anywhere.
func TestCompareCounts(t *testing.T) {
	// 3x1 premul RGBA: identical, off by 1 on one channel, off by 5 on one channel.
	a := []byte{10, 10, 10, 0xFF, 20, 20, 20, 0xFF, 30, 30, 30, 0xFF}
	b := []byte{10, 10, 10, 0xFF, 20, 21, 20, 0xFF, 35, 30, 30, 0xFF}
	res, err := Compare(a, b, 3, 1, Exact1)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if res.AnyDiffPixels != 2 {
		t.Errorf("AnyDiffPixels = %d, want 2", res.AnyDiffPixels)
	}
	if res.DiffPixels != 1 {
		t.Errorf("DiffPixels = %d, want 1 (only the delta-5 pixel exceeds exact1)", res.DiffPixels)
	}
	if res.MaxDelta != 5 {
		t.Errorf("MaxDelta = %d, want 5", res.MaxDelta)
	}
	if res.Pass() {
		t.Errorf("a delta-5 pixel passed under %s: %s", res.Profile.Name, res)
	}

	// The ±1 wobble alone stays inside exact1, which is the whole point of the profile.
	res, err = Compare(a[:8], b[:8], 2, 1, Exact1)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if !res.Pass() || res.DiffPixels != 0 || res.AnyDiffPixels != 1 || res.MaxDelta != 1 {
		t.Errorf("a ±1 wobble did not pass cleanly under exact1: %s", res)
	}
	if res, err = Compare(a[:8], b[:8], 2, 1, Exact); err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if res.Pass() {
		t.Errorf("a ±1 wobble passed under exact: %s", res)
	}
}

// TestCompareRejectsMismatchedSizes verifies the buffer-length guard, which is what keeps a damaged golden from being
// silently compared against a differently sized render.
func TestCompareRejectsMismatchedSizes(t *testing.T) {
	if _, err := Compare(make([]byte, 16), make([]byte, 12), 2, 2, Exact); err == nil {
		t.Fatal("Compare with mismatched buffer sizes returned no error")
	}
}
