// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This probe checks geom.Matrix against Skia's Matrix concat/invert/map results, replayed from the frozen reference
// fixtures (see ref.go). It once drove Skia's C++ internals live through the skiac shim, which confined it to
// darwin/linux — mingw g++ cannot resolve Skia's MSVC-mangled C++ exports. Replay needs no linker, so it now runs
// everywhere.

package probe

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// TestMatrixSetConcatProbe: geom.Matrix.SetConcat vs Skia's Matrix::setConcat over all corpus pairs. Affine×affine
// concat goes through Skia's double-precision muladdmul and must be bit-exact; when either input has perspective, Skia
// switches to the all-float rowcol3 and the contraction tolerance applies.
func TestMatrixSetConcatProbe(t *testing.T) {
	corpus := matrixCorpus()
	for i := range corpus {
		for j := range corpus {
			a, b := &corpus[i], &corpus[j]
			want := refMatrixSetConcat(a, b)
			var got geom.Matrix
			got.SetConcat(a, b)
			gv, wv := got.As9(), want.As9()
			if !a.HasPerspective() && !b.HasPerspective() {
				if !matEq(got, want) {
					t.Errorf("SetConcat(corpus[%d], corpus[%d]) affine:\n  go %v\n  c  %v", i, j, gv, wv)
				}
				continue
			}
			a9, b9 := a.As9(), b.As9()
			for k := range gv {
				_, mag := refRowCol3(a9, b9, k/3, k%3)
				if !agree(gv[k], wv[k], mag) {
					t.Errorf("SetConcat(corpus[%d], corpus[%d]) elem %d: go %v, c %v (mag %g)",
						i, j, k, gv[k], wv[k], mag)
				}
			}
		}
	}
}

// TestMatrixInvertProbe: geom.Matrix.Invert vs Skia's Matrix::invert. The affine path computes cofactors in double
// (dcross) and must be bit-exact; the perspective path's scross is float (measured ≤2 ULP here), so it gets a small
// fixed ULP allowance via agree() with the result's own magnitude.
func TestMatrixInvertProbe(t *testing.T) {
	for i, m := range matrixCorpus() {
		want, wantOK := refMatrixInvert(&m)
		got, gotOK := m.Invert()
		if gotOK != wantOK {
			t.Errorf("Invert(corpus[%d]) ok: go %v, c %v (m=%v)", i, gotOK, wantOK, m.As9())
			continue
		}
		if !gotOK {
			continue
		}
		if !m.HasPerspective() {
			if !matEq(got, want) {
				t.Errorf("Invert(corpus[%d]) affine:\n  go %v\n  c  %v\n  m=%v", i, got.As9(), want.As9(), m.As9())
			}
			continue
		}
		gv, wv := got.As9(), want.As9()
		for k := range gv {
			mag := math.Abs(float64(wv[k]))
			if !agree(gv[k], wv[k], mag) {
				t.Errorf("Invert(corpus[%d]) elem %d: go %v, c %v", i, k, gv[k], wv[k])
			}
		}
	}
}

// TestMatrixMapPointsProbe: geom.Matrix.MapPoints vs Skia's Matrix::mapPoints. All the C proc variants are float SIMD
// code, so the contraction tolerance applies throughout (identity/translate classes still end up bit-equal in
// practice).
func TestMatrixMapPointsProbe(t *testing.T) {
	pts := pointCorpus()
	for i, m := range matrixCorpus() {
		want := refMatrixMapPoints(&m, pts)
		got := make([]geom.Point, len(pts))
		m.MapPoints(got, pts)
		for k := range pts {
			_, _, magX, magY := refMapPoint(&m, pts[k])
			if !agree(got[k].X, want[k].X, magX) || !agree(got[k].Y, want[k].Y, magY) {
				t.Errorf("MapPoints(corpus[%d], pts[%d]=%v): go %v, c %v (mag %g/%g m=%v)",
					i, k, pts[k], got[k], want[k], magX, magY, m.As9())
			}
		}
	}
}

// TestMatrixMapRectProbe: mapRect vs Skia's Matrix::mapRect. Non-perspective matrices go through geom.Matrix.MapRect;
// perspective matrices go through path.MapRect, which mirrors Skia's Matrix::mapRect's route through a rect-path
// transform, including the w = 1/16384 half-plane clip via the ported edge clipper. Edges are compared with the maximum
// corner tolerance for the axis; results with a corner near the clip plane additionally use their own magnitude, since
// clipping projects coordinates up to ~16384× larger and the clip pipeline's float noise scales with them. The
// rect-stays-rect flag must match exactly (it derives from the exact type-mask classification, not float arithmetic).
func TestMatrixMapRectProbe(t *testing.T) {
	rects := rectCorpus()
	for i, m := range matrixCorpus() {
		for k, r := range rects {
			want, wantStays := refMatrixMapRect(&m, r)
			var got geom.Rect
			var gotStays bool
			if m.HasPerspective() {
				got, gotStays = path.MapRect(&m, r)
			} else {
				got, gotStays = m.MapRect(r)
			}
			if gotStays != wantStays {
				t.Errorf("MapRect(corpus[%d], rects[%d]) staysRect: go %v, c %v", i, k, gotStays, wantStays)
			}
			var magX, magY, wMag float64
			nearW0 := false
			m9 := m.As9()
			for _, corner := range [4]geom.Point{
				{X: r.Left, Y: r.Top},
				{X: r.Right, Y: r.Top},
				{X: r.Left, Y: r.Bottom},
				{X: r.Right, Y: r.Bottom},
			} {
				if m.HasPerspective() {
					w := float64(m9[6])*float64(corner.X) + float64(m9[7])*float64(corner.Y) + float64(m9[8])
					if w < 1e-3 {
						nearW0 = true
					}
					wMag = math.Max(wMag, math.Abs(float64(m9[6]))*math.Abs(float64(corner.X))+
						math.Abs(float64(m9[7]))*math.Abs(float64(corner.Y))+math.Abs(float64(m9[8])))
				}
				_, _, mx, my := refMapPoint(&m, corner)
				magX = math.Max(magX, mx)
				magY = math.Max(magY, my)
			}
			if nearW0 {
				// The clip pins geometry to w = w0 = 1/16384. Computing w for a pinned point cancels terms of magnitude
				// wMag down to w0; when the float rounding noise of that cancellation (probeK·eps32·wMag) can reach
				// w0's own scale, the projected result is unconstrained — Skia's own output is platform-dependent there
				// (its contracted w can even land on exactly 0, which Persp_pts maps to the origin). Skip those;
				// otherwise compare with the tolerance amplified by the ~1/w0 projection of the residual noise.
				const w0 = 1.0 / 16384
				if probeK*eps32*wMag >= w0/4 {
					continue
				}
				for _, v := range [...]float32{got.Left, got.Right, want.Left, want.Right} {
					magX = math.Max(magX, math.Abs(float64(v)))
				}
				for _, v := range [...]float32{got.Top, got.Bottom, want.Top, want.Bottom} {
					magY = math.Max(magY, math.Abs(float64(v)))
				}
				magX *= 2048
				magY *= 2048
			}
			if !agree(got.Left, want.Left, magX) || !agree(got.Right, want.Right, magX) ||
				!agree(got.Top, want.Top, magY) || !agree(got.Bottom, want.Bottom, magY) {
				t.Errorf("MapRect(corpus[%d], rects[%d]=%v):\n  go %v\n  c  %v\n  m=%v (mag %g/%g)",
					i, k, r, got, want, m.As9(), magX, magY)
			}
		}
	}
}
