// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the op pipeline's bounds arithmetic.

package gl

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// TestRectsOverlap pins the overlap predicate, including its invalid-rect contract: no comparison involving a NaN may
// conclude "disjoint", so canReorder stays conservative and never hoists a draw past one it may cover.
func TestRectsOverlap(t *testing.T) {
	a := geom.Rect{Left: 0, Top: 0, Right: 10, Bottom: 10}
	nan := float32(math.NaN())
	for _, tc := range []struct {
		name string
		b    geom.Rect
		want bool
	}{
		{name: "overlapping", b: geom.Rect{Left: 5, Top: 5, Right: 15, Bottom: 15}, want: true},
		{name: "contained", b: geom.Rect{Left: 2, Top: 2, Right: 4, Bottom: 4}, want: true},
		{name: "disjoint", b: geom.Rect{Left: 20, Top: 20, Right: 30, Bottom: 30}, want: false},
		// A shared edge is not an overlap: the test is strict (rectsTouchOrOverlap is the inclusive form).
		{name: "shared edge", b: geom.Rect{Left: 10, Top: 0, Right: 20, Bottom: 10}, want: false},
		// Rects may be infinitely small, so a zero-area rect strictly inside another still overlaps it.
		{name: "empty inside", b: geom.Rect{Left: 5, Top: 5, Right: 5, Bottom: 5}, want: true},
		{name: "empty outside", b: geom.Rect{Left: 20, Top: 20, Right: 20, Bottom: 20}, want: false},
		// A NaN horizontal extent cannot prove horizontal disjointness, and the vertical extents overlap.
		{name: "NaN left", b: geom.Rect{Left: nan, Top: 5, Right: 15, Bottom: 15}, want: true},
		{name: "NaN right", b: geom.Rect{Left: -5, Top: 5, Right: nan, Bottom: 15}, want: true},
		{name: "all NaN", b: geom.Rect{Left: nan, Top: nan, Right: nan, Bottom: nan}, want: true},
		// A NaN on one axis does not poison an axis that is provably disjoint on its own.
		{name: "NaN left, below", b: geom.Rect{Left: nan, Top: 20, Right: 30, Bottom: 30}, want: false},
	} {
		if got := rectsOverlap(a, tc.b); got != tc.want {
			t.Errorf("%s: rectsOverlap(%v, %v) = %v, want %v", tc.name, a, tc.b, got, tc.want)
		}
		// The predicate is symmetric, and canReorder is its negation.
		if got := rectsOverlap(tc.b, a); got != tc.want {
			t.Errorf("%s: rectsOverlap is asymmetric", tc.name)
		}
		if got := canReorder(a, tc.b); got != !tc.want {
			t.Errorf("%s: canReorder = %v, want %v", tc.name, got, !tc.want)
		}
	}
}
