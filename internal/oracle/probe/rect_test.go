// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This probe checks geom.Rect against Skia's Rect intersect/join/round results, replayed from the frozen reference
// fixtures (see ref.go). It once drove Skia's C++ internals live through the skiac shim, which confined it to
// darwin/linux — mingw g++ cannot resolve Skia's MSVC-mangled C++ exports. Replay needs no linker, so it now runs
// everywhere.

package probe

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// TestRectIntersectProbe: geom.Rect.Intersect vs Skia's Rect::intersect — flag and mutated receiver, including the
// asymmetric NaN quirk the port deliberately preserves.
func TestRectIntersectProbe(t *testing.T) {
	rects := rectCorpus()
	for i, a := range rects {
		for j, b := range rects {
			wantRect, wantOK := refRectIntersect(a, b)
			got := a
			gotOK := got.Intersect(b)
			if gotOK != wantOK {
				t.Errorf("Intersect(rects[%d], rects[%d]) ok: go %v, c %v (a=%v b=%v)", i, j, gotOK, wantOK, a, b)
				continue
			}
			if gotOK && !rectEq(got, wantRect) {
				t.Errorf("Intersect(rects[%d], rects[%d]): go %v, c %v", i, j, got, wantRect)
			}
		}
	}
}

// TestRectJoinProbe: geom.Rect.Join vs Skia's Rect::join over all corpus pairs.
func TestRectJoinProbe(t *testing.T) {
	rects := rectCorpus()
	for i, a := range rects {
		for j, b := range rects {
			want := refRectJoin(a, b)
			got := a
			got.Join(b)
			if !rectEq(got, want) {
				t.Errorf("Join(rects[%d], rects[%d]): go %v, c %v (a=%v b=%v)", i, j, got, want, a, b)
			}
		}
	}
}

// TestRectSetBoundsProbe: geom.Rect.SetBounds vs Skia's Rect::setBoundsCheck (poison-accumulator finite check).
func TestRectSetBoundsProbe(t *testing.T) {
	pts := pointCorpus()
	// Prefix slices of the corpus, plus slices starting at a non-finite point.
	cases := [][]geom.Point{nil, pts[:1], pts[:2], pts[:7], pts, pts[38:], pts[40:44]}
	for i, c := range cases {
		wantRect, wantOK := refRectSetBounds(c)
		var got geom.Rect
		gotOK := got.SetBounds(c)
		if gotOK != wantOK || !rectEq(got, wantRect) {
			t.Errorf("SetBounds(case %d, %d pts): go %v/%v, c %v/%v", i, len(c), got, gotOK, wantRect, wantOK)
		}
	}
}

// TestRectRoundProbe: Round/RoundOut/RoundIn vs the Skia's Rect versions (saturated double rounding).
func TestRectRoundProbe(t *testing.T) {
	for i, r := range rectCorpus() {
		if got, want := r.Round(), refRectRound(r); got != want {
			t.Errorf("Round(rects[%d]=%v): go %v, c %v", i, r, got, want)
		}
		if got, want := r.RoundOut(), refRectRoundOut(r); got != want {
			t.Errorf("RoundOut(rects[%d]=%v): go %v, c %v", i, r, got, want)
		}
		if got, want := r.RoundIn(), refRectRoundIn(r); got != want {
			t.Errorf("RoundIn(rects[%d]=%v): go %v, c %v", i, r, got, want)
		}
	}
}

// TestScalarRoundToIntProbe: geom.RoundToInt vs Skia's ScalarRoundToInt over boundary and special values.
func TestScalarRoundToIntProbe(t *testing.T) {
	values := []float32{
		0, 0.5, -0.5, 0.49999997, -0.49999997, 1.5, -1.5, 2.5, -2.5,
		16777215, 16777215.5, -16777215, 8388608.5,
		2147483520, 2147483648, -2147483648, -2147483904,
		3e38, -3e38, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
		1e-40, -1e-40, 0.9999999,
	}
	for _, v := range values {
		if got, want := geom.RoundToInt(v), refScalarRoundToInt(v); got != want {
			t.Errorf("RoundToInt(%g): go %d, c %d", v, got, want)
		}
	}
}
