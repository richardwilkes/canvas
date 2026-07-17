// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Byte-exactness locks for the SWAR/specialized forms of the bitmap-processing kernels: each rewritten kernel is
// compared against the original scalar formula it replaced, which is kept here as the reference. The kernels feed the
// oracle-verified legacy image lane, so any drift — a single bit in a single channel — is a correctness bug, not a
// tolerance question.

package shaders

import (
	"math/rand/v2"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// refFilterAndScaleByAlpha is the pre-SWAR filter-and-scale-by-alpha kernel: the original uint32 lane-pair form,
// retained as the differential reference.
func refFilterAndScaleByAlpha(x, y, a00, a01, a10, a11, alphaScale uint32) uint32 {
	xy := x * y
	const mask = 0xFF00FF

	scale := 256 - 16*y - 16*x + xy
	lo := (a00 & mask) * scale
	hi := ((a00 >> 8) & mask) * scale

	scale = 16*x - xy
	lo += (a01 & mask) * scale
	hi += ((a01 >> 8) & mask) * scale

	scale = 16*y - xy
	lo += (a10 & mask) * scale
	hi += ((a10 >> 8) & mask) * scale

	lo += (a11 & mask) * xy
	hi += ((a11 >> 8) & mask) * xy

	if alphaScale < 256 {
		lo = ((lo >> 8) & mask) * alphaScale
		hi = ((hi >> 8) & mask) * alphaScale
	}

	return ((lo >> 8) & mask) | (hi & ^uint32(mask))
}

// TestFilterAndScaleByAlphaMatchesReference sweeps the bilerp kernel across every weight pair (the 4-bit subpixel
// weights are 0..15) with random pixels and alpha scales, requiring bit equality with the original lane-pair formula.
// filterNoAlpha must additionally equal the general kernel at alphaScale 256.
func TestFilterAndScaleByAlphaMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	alphaScales := []uint32{1, 17, 128, 255, 256}
	for x := uint32(0); x < 16; x++ {
		for y := uint32(0); y < 16; y++ {
			for range 64 {
				a00, a01, a10, a11 := rng.Uint32(), rng.Uint32(), rng.Uint32(), rng.Uint32()
				for _, as := range alphaScales {
					want := refFilterAndScaleByAlpha(x, y, a00, a01, a10, a11, as)
					if got := filterAndScaleByAlpha(x, y, a00, a01, a10, a11, as); got != want {
						t.Fatalf("filterAndScaleByAlpha(%d,%d, %08x,%08x,%08x,%08x, %d) = %08x, want %08x",
							x, y, a00, a01, a10, a11, as, got, want)
					}
				}
				want := refFilterAndScaleByAlpha(x, y, a00, a01, a10, a11, 256)
				if got := filterNoAlpha(x, y, a00, a01, a10, a11); got != want {
					t.Fatalf("filterNoAlpha(%d,%d, %08x,%08x,%08x,%08x) = %08x, want %08x",
						x, y, a00, a01, a10, a11, got, want)
				}
			}
		}
	}
	// Saturated corners: every channel extreme on all four taps.
	for x := uint32(0); x < 16; x++ {
		for y := uint32(0); y < 16; y++ {
			for _, px := range []uint32{0x00000000, 0xFFFFFFFF, 0xFF00FF00, 0x00FF00FF} {
				want := refFilterAndScaleByAlpha(x, y, px, px, px, px, 256)
				if got := filterNoAlpha(x, y, px, px, px, px); got != want {
					t.Fatalf("filterNoAlpha(%d,%d, uniform %08x) = %08x, want %08x", x, y, px, got, want)
				}
			}
		}
	}
}

// buildMatrixProcState builds a bilerp/nearest clamp-clamp bitmapProcState over a small pixmap with the given inverse
// matrix, failing the test when the setup would reject it (the sweeps below only use accepted configurations).
func buildMatrixProcState(t *testing.T, w, h int32, inv geom.Matrix, bilerp bool) *bitmapProcState {
	t.Helper()
	pm := raster.NewPixmap(w, h)
	rng := rand.New(rand.NewPCG(3, 4))
	for i := range pm.Pix {
		pm.Pix[i] = rng.Uint32()
	}
	s := &bitmapProcState{}
	filter := FilterNearest
	if bilerp {
		filter = FilterLinear
	}
	if !s.setup(*pm, &inv, 0xFF, SamplingOptions{Filter: filter}, TileClamp, TileClamp) {
		t.Fatalf("setup rejected w=%d h=%d inv=%v bilerp=%v", w, h, inv, bilerp)
	}
	return s
}

// TestClampScaleMatrixProcsMatchGeneral drives the clamp-clamp specializations and the general tile-proc forms over the
// same states — scales that stay inside the source, scales that clamp on both edges, subpixel translations, negative
// (mirrored) scales, and a saturating extreme — and requires the packed coordinate buffers to match word for word.
func TestClampScaleMatrixProcsMatchGeneral(t *testing.T) {
	scaleTrans := func(sx, sy, tx, ty float32) geom.Matrix {
		m := geom.ScaleMatrix(sx, sy)
		m.PostTranslate(tx, ty)
		return m
	}
	cases := []struct {
		name string
		w, h int32
		inv  geom.Matrix
	}{
		{name: "upscale-interior", w: 64, h: 64, inv: scaleTrans(64.0/224.0, 64.0/224.0, -4.57, -3.2)},
		{name: "upscale-edges", w: 64, h: 64, inv: scaleTrans(64.0/224.0, 64.0/224.0, -8, -8)},
		{name: "downscale", w: 64, h: 64, inv: scaleTrans(3.7, 3.7, -1.25, -0.5)},
		{name: "identityish", w: 64, h: 64, inv: scaleTrans(1, 1, -0.375, -0.125)},
		{name: "negative-scale", w: 64, h: 64, inv: scaleTrans(-0.8, -0.8, 70.2, 66.6)},
		{name: "tiny-source", w: 2, h: 2, inv: geom.ScaleMatrix(2.0/256.0, 2.0/256.0)},
		{name: "huge-scale-saturating", w: 64, h: 64, inv: scaleTrans(40000, 40000, -1e6, -2000)},
	}
	for _, tc := range cases {
		for _, bilerp := range []bool{true, false} {
			s := buildMatrixProcState(t, tc.w, tc.h, tc.inv, bilerp)
			if !s.invMatrix.IsScaleTranslate() {
				t.Fatalf("%s: expected scale-translate state", tc.name)
			}
			const count = 300
			var got, want [count + 2]uint32
			for _, y := range []int32{0, 5, 63} {
				if bilerp {
					filterScaleClamp(s, got[:], count, 3, y)
					filterScale(s, tileClampProc, extractLowBitsClampClamp, true, want[:], count, 3, y)
				} else {
					nofilterScaleClamp(s, got[:], count, 3, y)
					nofilterScale(s, tileClampProc, true, want[:], count, 3, y)
				}
				if got != want {
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("%s bilerp=%v y=%d: xy[%d] = %08x, want %08x", tc.name, bilerp, y, i,
								got[i], want[i])
						}
					}
				}
				// The specializations must also be what setup wires up for clamp-clamp scale states.
				var viaState [count + 2]uint32
				s.matrixProc(s, viaState[:], count, 3, y)
				if viaState != want {
					t.Fatalf("%s bilerp=%v y=%d: state-wired matrixProc diverges from general form", tc.name,
						bilerp, y)
				}
			}
		}
	}
}
