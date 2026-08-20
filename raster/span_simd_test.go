// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package raster

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// simdHostileFloats seeds channel values with every float class the shader pipeline can hand the span kernels, so the
// vector/scalar comparison exercises NaN, infinities, signed zero, denormals, and the values that straddle toUnorm's
// clamp and round-to-nearest-even boundaries. (A file-local copy of the shaders package's list, extended with this
// package's own boundary cases; the two packages do not share test helpers.)
var simdHostileFloats = []float32{
	0, float32(math.Copysign(0, -1)), 1, -1, 0.5, 0.5 / 255, 1.5 / 255, 254.5 / 255, 0.9999999, 1.0000001,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 1e-38, -1e-38,
	3.4e38, -3.4e38, 1e-45, 255, 256, -255, 0.1, 1.0 / 3.0, 2.0 / 3.0,
}

// simdQuietFloat turns arbitrary random bits into a value the pipeline could actually hold: every span kernel input is
// the result of IEEE arithmetic, which never emits a *signaling* NaN, so sNaN bit patterns are quieted. This matters
// because a hardware min/max would quiet an sNaN operand where Go's comparison-based minf/maxf selects the other
// operand — a divergence that is unreachable in production but constructible from raw bits.
func simdQuietFloat(bits uint32) float32 {
	const expMask, quietBit = 0x7F800000, 0x00400000
	if bits&expMask == expMask && bits&0x007FFFFF != 0 {
		bits |= quietBit
	}
	return math.Float32frombits(bits)
}

// TestSpanSIMDMatchesScalar drives every simd span kernel and its portable twin over identical randomized rows and
// requires bitwise identity. Row lengths run 1..67 so every chunk/tail split is covered, including the lengths that are
// not multiples of the four-pixel vector chunk. The two integer kernels additionally get exhaustive subtests over their
// complete input domains, which are small enough to enumerate.
func TestSpanSIMDMatchesScalar(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd kernels require; dispatch stays on the default forms")
	}
	rng := rand.New(rand.NewPCG(21, 22))
	pick := func() float32 {
		if rng.IntN(3) == 0 {
			return simdHostileFloats[rng.IntN(len(simdHostileFloats))]
		}
		return simdQuietFloat(rng.Uint32())
	}
	randSpan := func(n int) []colorcore.PMColor4f {
		buf := make([]colorcore.PMColor4f, n)
		for i := range buf {
			buf[i] = colorcore.PMColor4f{R: pick(), G: pick(), B: pick(), A: pick()}
		}
		return buf
	}

	t.Run("clampSpan01", func(t *testing.T) {
		for range 4096 {
			n := 1 + rng.IntN(67)
			want := randSpan(n)
			got := append([]colorcore.PMColor4f(nil), want...)
			clampSpan01Generic(want)
			clampSpan01SIMD(got)
			// clamp01f never emits NaN or -0 (NaN and negatives clamp to +0), so plain bit comparison is exact.
			for i := range want {
				for ch, pair := range [][2]float32{
					{got[i].R, want[i].R},
					{got[i].G, want[i].G},
					{got[i].B, want[i].B},
					{got[i].A, want[i].A},
				} {
					if math.Float32bits(pair[0]) != math.Float32bits(pair[1]) {
						t.Fatalf("clampSpan01 n=%d pixel %d ch %d = %08x, want %08x", n, i, ch,
							math.Float32bits(pair[0]), math.Float32bits(pair[1]))
					}
				}
			}
		}
	})

	t.Run("storeSpanSrc", func(t *testing.T) {
		for range 4096 {
			n := 1 + rng.IntN(67)
			buf := randSpan(n)
			want := make([]uint32, n)
			storeSpanSrcGeneric(buf, want)
			got := make([]uint32, n)
			storeSpanSrcSIMD(buf, got)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("storeSpanSrc n=%d pixel %d (%+v) = %08x, want %08x", n, i, buf[i], got[i], want[i])
				}
			}
		}
	})

	t.Run("pmSrcOverRow", func(t *testing.T) {
		for range 4096 {
			n := 1 + rng.IntN(67)
			src := make([]uint32, n)
			want := make([]uint32, n)
			for i := range src {
				src[i] = rng.Uint32()
				want[i] = rng.Uint32()
			}
			got := append([]uint32(nil), want...)
			pmSrcOverRowGeneric(want, src)
			pmSrcOverRowSIMD(got, src)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("pmSrcOverRow n=%d pixel %d: src %08x -> %08x, want %08x", n, i, src[i], got[i], want[i])
				}
			}
		}
	})

	t.Run("blitMaskOpaqueRow", func(t *testing.T) {
		for range 4096 {
			n := 1 + rng.IntN(67)
			aa := make([]uint8, n)
			want := make([]uint32, n)
			for i := range aa {
				aa[i] = uint8(rng.Uint32())
				want[i] = rng.Uint32()
			}
			aa[0] = 0
			if n > 1 {
				aa[1] = 255
			}
			got := append([]uint32(nil), want...)
			pm := rng.Uint32() | 0xFF000000
			blitMaskOpaqueRowGeneric(want, aa, pm)
			blitMaskOpaqueRowSIMD(got, aa, pm)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("blitMaskOpaqueRow n=%d pixel %d: pm %08x m %02x -> %08x, want %08x", n, i, pm, aa[i],
						got[i], want[i])
				}
			}
		}
	})

	// The exhaustive integer subtests below enumerate complete domains rather than sampling them: the kernels' only
	// freedom is per-byte integer arithmetic, so covering every (byte, scale) pair proves bit-exactness outright.

	t.Run("pmSrcOverRowExhaustive", func(t *testing.T) {
		// Every (dstByte, srcAlpha) pair: row i carries dst byte i in all four channels, and the outer loop walks the
		// source alpha, hence every 255-srcA scale. The source color bytes vary across channels so the saturating add
		// sees a spread of addends at each (dst, scale).
		src := make([]uint32, 256)
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		for a := range 256 {
			for i := range 256 {
				src[i] = uint32(a)<<24 | uint32(i)<<16 | uint32(255-i)<<8 | uint32((i*7)&0xFF)
				want[i] = uint32(i) * 0x01010101
				got[i] = want[i]
			}
			pmSrcOverRowGeneric(want, src)
			pmSrcOverRowSIMD(got, src)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("pmSrcOverRow srcA=%d dst byte=%d: %08x, want %08x", a, i, got[i], want[i])
				}
			}
		}
		// Every (srcByte, dstByte) pair through the saturating add: srcA == 0 makes the scale 255, and
		// mulDiv255Round(d, 255) == d, so the row's result is exactly satAdd8(src_b, dst_b) per channel.
		for s := range 256 {
			for i := range 256 {
				src[i] = uint32(s) * 0x00010101
				want[i] = uint32(i) * 0x01010101
				got[i] = want[i]
			}
			pmSrcOverRowGeneric(want, src)
			pmSrcOverRowSIMD(got, src)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("pmSrcOverRow satAdd src byte=%d dst byte=%d: %08x, want %08x", s, i, got[i], want[i])
				}
			}
		}
	})

	t.Run("blitMaskOpaqueRowExhaustive", func(t *testing.T) {
		// Every (pmByte, devByte, coverage) triple: the color's three color channels sweep pm, the row sweeps the
		// device byte, and the two outer loops sweep the color byte and the coverage.
		aa := make([]uint8, 256)
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		for p := range 256 {
			pm := uint32(p)*0x01010101&0x00FFFFFF | 0xFF000000
			for m := range 256 {
				for i := range 256 {
					aa[i] = uint8(m)
					want[i] = uint32(i) * 0x01010101
					got[i] = want[i]
				}
				blitMaskOpaqueRowGeneric(want, aa, pm)
				blitMaskOpaqueRowSIMD(got, aa, pm)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("blitMaskOpaqueRow pm=%08x m=%d dev byte=%d: %08x, want %08x", pm, m, i, got[i],
							want[i])
					}
				}
			}
		}
	})
}

// TestSpanSIMDWiring locks that where the simd kernels are the preferred lane (amd64 with AVX2+FMA; arm64 declines in
// favor of the faster NEON assembly, see span_simd_arm64.go), the goexperiment.simd build's init actually repointed
// the dispatch variables at them, so a refactor cannot silently fall back to the default forms.
func TestSpanSIMDWiring(t *testing.T) {
	if !simdKernelsPreferred() {
		t.Skip("the simd span kernels are not the preferred lane on this hardware; dispatch keeps the default forms")
	}
	for name, pair := range map[string][2]any{
		"clampSpan01":       {clampSpan01Fn, spanClampFn(clampSpan01SIMD)},
		"storeSpanSrc":      {storeSpanSrcFn, spanStoreFn(storeSpanSrcSIMD)},
		"pmSrcOverRow":      {pmSrcOverRowFn, spanRowFn(pmSrcOverRowSIMD)},
		"blitMaskOpaqueRow": {blitMaskOpaqueRowFn, spanMaskFn(blitMaskOpaqueRowSIMD)},
	} {
		if reflect.ValueOf(pair[0]).Pointer() != reflect.ValueOf(pair[1]).Pointer() {
			t.Fatalf("%s: dispatch fn is not the simd kernel", name)
		}
	}
}
