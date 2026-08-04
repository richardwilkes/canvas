// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This probe checks the library's analytic AA edge walkers against Skia's Analytic*'s, replayed from the frozen
// reference fixtures (see ref_test.go).

package probe

import (
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// TestAnalyticEdgeSegmentsProbe compares the raw analytic quad/cubic edge segment streams against Skia's,
// bit-exactly (all the math past the float->FDot6 conversion is integer).
//
// The streams are dumped at raster.SkiaSnapAccuracy, Skia's quarter-scanline endpoint snap. The port deliberately
// snaps endpoints finer by default — Skia's grid drops a partial row thinner than 1/8 of a scanline outright, which is
// GitHub issue #2 — so holding that one constant at Skia's value is what keeps this a differential test of the ported
// curve math (polynomial setup, subdivision heuristic, forward differences, slope derivation) rather than a test that
// fails wholesale on a difference already known and intended. The shipping grid is covered by the raster package's own
// tests and by the self-captured goldens; nothing here can validate it, because Skia has no answer to compare against.
func TestAnalyticEdgeSegmentsProbe(t *testing.T) {
	rng := rand.New(rand.NewSource(0xedbe))
	// The float32() forces the multiply to round before the subtract. Without it the compiler is free to fuse the two
	// into a single FMA — which arm64 does and amd64 does not — so this "fixed seed" corpus would be a *different*
	// corpus per platform (measured: 948 of 5000 coordinates differ by 1 ULP), and the fixtures, keyed by input, would
	// miss everywhere they were not captured. Rounding explicitly pins the corpus to the spec-literal value every
	// platform can reproduce. Corpora built as (x*2-1)*s need no such care: x*2 is exact, so fusing cannot change it.
	coord := func() float32 { return float32(rng.Float32()*300) - 20 }
	for iter := 0; iter < 500; iter++ {
		if iter%2 == 0 {
			var pts [3]geom.Point
			for i := range pts {
				pts[i] = geom.Pt(coord(), coord())
			}
			// The builder only feeds Y-monotonic quads; sort control Ys to approximate that.
			if pts[0].Y > pts[2].Y {
				pts[0], pts[2] = pts[2], pts[0]
			}
			if pts[1].Y < pts[0].Y {
				pts[1].Y = pts[0].Y
			}
			if pts[1].Y > pts[2].Y {
				pts[1].Y = pts[2].Y
			}
			want := refAnalyticQuadSegments(pts)
			got := raster.AnalyticQuadSegments(pts[:], raster.SkiaSnapAccuracy)
			compareSegs(t, iter, "quad", got, want)
		} else {
			var pts [4]geom.Point
			for i := range pts {
				pts[i] = geom.Pt(coord(), coord())
			}
			want := refAnalyticCubicSegments(pts)
			got := raster.AnalyticCubicSegments(pts[:], raster.SkiaSnapAccuracy)
			compareSegs(t, iter, "cubic", got, want)
		}
	}
}

func compareSegs(t *testing.T, iter int, kind string, got []raster.AnalyticEdgeSegment, want [][7]int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s[%d]: segment count go %d, c %d", kind, iter, len(got), len(want))
		return
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s[%d] seg %d: go %v, c %v", kind, iter, i, got[i], want[i])
			return
		}
	}
}
