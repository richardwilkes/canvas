// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// quietBits turns arbitrary random bits into a value the shader pipeline can actually emit: IEEE arithmetic never
// produces signaling NaNs, and FMAXNM/FMINNM treat sNaN differently from the scalar comparisons (see
// shaders/stage_arm64.s), so the comparison domain is quiet-NaN-only.
func quietBits(bits uint32) float32 {
	const expMask, quietBit = 0x7F800000, 0x00400000
	if bits&expMask == expMask && bits&0x007FFFFF != 0 {
		bits |= quietBit
	}
	return math.Float32frombits(bits)
}

// TestSpanNEONMatchesScalar locks the NEON span kernels against their portable twins bit for bit over randomized spans
// of every length class (quads plus tails), with hostile float classes for the float kernels and full random pixels for
// the integer kernel. It calls the NEON wrappers directly rather than the dispatch variables, so it keeps testing the
// NEON lane under goexperiment.simd too, where init repoints the dispatch at the archsimd kernels.
func TestSpanNEONMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	hostile := []float32{
		0, float32(math.Copysign(0, -1)), 1, -1, 0.5, 0.5 / 255, 1.5 / 255, 254.5 / 255, 1.0000001,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
		math.SmallestNonzeroFloat32, 3.4e38, -3.4e38,
	}
	randColor := func() colorcore.PMColor4f {
		pick := func() float32 {
			if rng.IntN(3) == 0 {
				return hostile[rng.IntN(len(hostile))]
			}
			return quietBits(rng.Uint32())
		}
		return colorcore.PMColor4f{R: pick(), G: pick(), B: pick(), A: pick()}
	}

	t.Run("clampSpan01", func(t *testing.T) {
		for _, n := range []int{1, 2, 3, 4, 5, 7, 16, 33, 240} {
			for range 64 {
				buf1 := make([]colorcore.PMColor4f, n)
				for i := range buf1 {
					buf1[i] = randColor()
				}
				buf2 := append([]colorcore.PMColor4f(nil), buf1...)
				for i := range buf1 {
					buf1[i].R = clamp01f(buf1[i].R)
					buf1[i].G = clamp01f(buf1[i].G)
					buf1[i].B = clamp01f(buf1[i].B)
					buf1[i].A = clamp01f(buf1[i].A)
				}
				clampSpan01NEON(buf2)
				// clamp01f never emits NaN or -0 (NaN and negatives clamp to +0), so plain bit comparison of every
				// channel is exact.
				for i := range buf1 {
					for ch, pair := range [][2]float32{
						{buf2[i].R, buf1[i].R},
						{buf2[i].G, buf1[i].G},
						{buf2[i].B, buf1[i].B},
						{buf2[i].A, buf1[i].A},
					} {
						if math.Float32bits(pair[0]) != math.Float32bits(pair[1]) {
							t.Fatalf("clampSpan01 n=%d pixel %d ch %d = %08x, want %08x", n, i, ch,
								math.Float32bits(pair[0]), math.Float32bits(pair[1]))
						}
					}
				}
			}
		}
	})

	t.Run("storeSpanSrc", func(t *testing.T) {
		for _, n := range []int{1, 2, 3, 4, 5, 7, 16, 33, 240} {
			for range 64 {
				buf := make([]colorcore.PMColor4f, n)
				for i := range buf {
					buf[i] = randColor()
				}
				want := make([]uint32, n)
				for i := range want {
					want[i] = storeWord(pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A})
				}
				got := make([]uint32, n)
				storeSpanSrcNEON(buf, got)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("storeSpanSrc n=%d pixel %d (%+v) = %08x, want %08x", n, i, buf[i],
							got[i], want[i])
					}
				}
			}
		}
	})

	t.Run("blitMaskOpaqueRow", func(t *testing.T) {
		for _, n := range []int{1, 2, 3, 4, 5, 7, 16, 33, 240} {
			for range 64 {
				aa := make([]uint8, n)
				d1 := make([]uint32, n)
				for i := range aa {
					aa[i] = uint8(rng.Uint32())
					d1[i] = rng.Uint32()
				}
				aa[0] = 0
				if n > 1 {
					aa[1] = 255
				}
				d2 := append([]uint32(nil), d1...)
				pm := rng.Uint32() | 0xFF000000
				blitMaskOpaqueRowGeneric(d1, aa, pm)
				blitMaskOpaqueRowNEON(d2, aa, pm)
				for i := range d1 {
					if d1[i] != d2[i] {
						t.Fatalf("blitMaskOpaqueRow n=%d pixel %d: pm %08x m %02x -> %08x, want %08x",
							n, i, pm, aa[i], d2[i], d1[i])
					}
				}
			}
		}
	})

	t.Run("pmSrcOverRow", func(t *testing.T) {
		for _, n := range []int{1, 2, 3, 4, 5, 7, 16, 33, 240} {
			for range 64 {
				src := make([]uint32, n)
				d1 := make([]uint32, n)
				for i := range src {
					src[i] = rng.Uint32()
					d1[i] = rng.Uint32()
				}
				d2 := append([]uint32(nil), d1...)
				pmSrcOverRowGeneric(d1, src)
				pmSrcOverRowNEON(d2, src)
				for i := range d1 {
					if d1[i] != d2[i] {
						t.Fatalf("pmSrcOverRow n=%d pixel %d: src %08x -> %08x, want %08x", n, i,
							src[i], d2[i], d1[i])
					}
				}
			}
		}
	})
}
