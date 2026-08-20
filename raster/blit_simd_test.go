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
	"math/rand/v2"
	"reflect"
	"testing"
)

// blitSIMDRowLen draws a row length for the randomized subtests: mostly lengths that cover every chunk/tail split of
// the four-pixel vector body, and one in four long enough to exercise the fill kernels' unrolled bodies and their
// remainders.
func blitSIMDRowLen(rng *rand.Rand) int {
	if rng.IntN(4) == 0 {
		return 1 + rng.IntN(300)
	}
	return 1 + rng.IntN(67)
}

// blitSIMDWords returns n random device words.
func blitSIMDWords(rng *rand.Rand, n int) []uint32 {
	w := make([]uint32, n)
	for i := range w {
		w[i] = rng.Uint32()
	}
	return w
}

// TestBlitSIMDMatchesScalar drives every simd blit row and its portable twin over identical inputs and requires bitwise
// identity. The randomized subtests cover every chunk/tail split; the exhaustive subtests that follow enumerate the
// complete integer domains wherever they are small enough to, which proves bit-exactness outright rather than sampling
// for it.
func TestBlitSIMDMatchesScalar(t *testing.T) {
	if !simdBlitSupported() {
		t.Skip("CPU lacks the features the simd blit rows require; dispatch stays on the portable forms")
	}
	rng := rand.New(rand.NewPCG(31, 32))

	t.Run("fillWords", func(t *testing.T) {
		for range 2048 {
			n := blitSIMDRowLen(rng)
			v := rng.Uint32()
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			fillWordsGeneric(want, v)
			fillWordsSIMD(got, v)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("fillWords n=%d index %d = %08x, want %08x", n, i, got[i], want[i])
				}
			}
		}
	})

	t.Run("fillBytes", func(t *testing.T) {
		for range 2048 {
			n := blitSIMDRowLen(rng)
			v := uint8(rng.Uint32())
			want := make([]uint8, n)
			for i := range want {
				want[i] = uint8(rng.Uint32())
			}
			got := append([]uint8(nil), want...)
			fillBytesGeneric(want, v)
			fillBytesSIMD(got, v)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("fillBytes n=%d index %d = %02x, want %02x", n, i, got[i], want[i])
				}
			}
		}
	})

	t.Run("color32Row", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			// A premultiplied color and its matching 256-based inverse scale, the pairing color32 dispatches with.
			a := 1 + rng.Uint32()%254
			color := a<<24 | (rng.Uint32()%(a+1))<<16 | (rng.Uint32()%(a+1))<<8 | rng.Uint32()%(a+1)
			invA := alpha255To256(255 - a)
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			color32RowGeneric(want, color, invA)
			color32RowSIMD(got, color, invA)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("color32Row n=%d pixel %d color %08x invA %d = %08x, want %08x", n, i, color, invA,
						got[i], want[i])
				}
			}
		}
	})

	t.Run("blitMaskTranslucentRow", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			srcA := rng.Uint32() % 256
			pm := srcA<<24 | (rng.Uint32()%(srcA+1))<<16 | (rng.Uint32()%(srcA+1))<<8 | rng.Uint32()%(srcA+1)
			aa := make([]uint8, n)
			for i := range aa {
				aa[i] = uint8(rng.Uint32())
			}
			aa[0] = 0
			if n > 1 {
				aa[1] = 255
			}
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			blitMaskTranslucentRowGeneric(want, aa, pm, srcA)
			blitMaskTranslucentRowSIMD(got, aa, pm, srcA)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("blitMaskTranslucentRow n=%d pixel %d pm %08x srcA %d m %02x = %08x, want %08x", n, i, pm,
						srcA, aa[i], got[i], want[i])
				}
			}
		}
	})

	t.Run("interp256Row", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			scale := rng.Uint32() % 257
			src := blitSIMDWords(rng, n)
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			interp256RowGeneric(want, src, scale)
			interp256RowSIMD(got, src, scale)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("interp256Row n=%d pixel %d scale %d = %08x, want %08x", n, i, scale, got[i], want[i])
				}
			}
		}
	})

	t.Run("premulRow", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			src := blitSIMDWords(rng, n)
			want := make([]uint32, n)
			got := make([]uint32, n)
			premulRowGeneric(want, src)
			premulRowSIMD(got, src)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("premulRow n=%d pixel %d src %08x = %08x, want %08x", n, i, src[i], got[i], want[i])
				}
			}
		}
	})

	t.Run("pmBlendRow", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			scale := rng.Uint32() % 257
			src := blitSIMDWords(rng, n)
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			pmBlendRowGeneric(want, src, scale)
			pmBlendRowSIMD(got, src, scale)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("pmBlendRow n=%d pixel %d src %08x scale %d = %08x, want %08x", n, i, src[i], scale,
						got[i], want[i])
				}
			}
		}
	})

	t.Run("blitRowLCD16", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			srcA, srcR, srcG, srcB := rng.Uint32()%256, rng.Uint32()%256, rng.Uint32()%256, rng.Uint32()%256
			mask := make([]uint16, n)
			for i := range mask {
				mask[i] = uint16(rng.Uint32())
			}
			mask[0] = 0
			if n > 1 {
				mask[1] = 0xFFFF
			}
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			blitRowLCD16Generic(want, mask, srcA, srcR, srcG, srcB)
			blitRowLCD16SIMD(got, mask, srcA, srcR, srcG, srcB)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("blitRowLCD16 n=%d pixel %d src %02x%02x%02x%02x mask %04x = %08x, want %08x", n, i, srcA,
						srcR, srcG, srcB, mask[i], got[i], want[i])
				}
			}
		}
	})

	t.Run("blitRowLCD16Opaque", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			srcR, srcG, srcB := rng.Uint32()%256, rng.Uint32()%256, rng.Uint32()%256
			// Half the rounds pass the device word of the color itself (the only way the blitter calls it) and half
			// pass an unrelated word, which is what makes the 0xFFFF-mask shortcut observable.
			opaqueDst := srcR | srcG<<8 | srcB<<16 | 0xFF<<24
			if rng.IntN(2) == 0 {
				opaqueDst = rng.Uint32()
			}
			mask := make([]uint16, n)
			for i := range mask {
				mask[i] = uint16(rng.Uint32())
			}
			mask[0] = 0xFFFF
			if n > 1 {
				mask[1] = 0
			}
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			blitRowLCD16OpaqueGeneric(want, mask, srcR, srcG, srcB, opaqueDst)
			blitRowLCD16OpaqueSIMD(got, mask, srcR, srcG, srcB, opaqueDst)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("blitRowLCD16Opaque n=%d pixel %d src %02x%02x%02x opaque %08x mask %04x = %08x, want %08x",
						n, i, srcR, srcG, srcB, opaqueDst, mask[i], got[i], want[i])
				}
			}
		}
	})

	t.Run("blendRowLCD16Opaque", func(t *testing.T) {
		for range 4096 {
			n := blitSIMDRowLen(rng)
			src := blitSIMDWords(rng, n)
			mask := make([]uint16, n)
			for i := range mask {
				mask[i] = uint16(rng.Uint32())
			}
			mask[0] = 0
			if n > 1 {
				mask[1] = 0xFFFF
			}
			want := blitSIMDWords(rng, n)
			got := append([]uint32(nil), want...)
			blendRowLCD16OpaqueGeneric(want, mask, src)
			blendRowLCD16OpaqueSIMD(got, mask, src)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("blendRowLCD16Opaque n=%d pixel %d src %08x mask %04x = %08x, want %08x", n, i, src[i],
						mask[i], got[i], want[i])
				}
			}
		}
	})

	// The exhaustive subtests below enumerate complete domains. Each builds a 256-pixel row whose pixels sweep one
	// byte of the domain and walks the remaining dimensions in its loops.

	t.Run("color32RowExhaustive", func(t *testing.T) {
		// Every (dstByte, invA) pair, at four color words that put every channel of the trailing add near its top.
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		for _, color := range []uint32{0, 0xFFFFFFFF, 0x01FE01FE, 0xFE01FE01} {
			for invA := range uint32(257) {
				for i := range want {
					want[i] = uint32(i) * 0x01010101
					got[i] = want[i]
				}
				color32RowGeneric(want, color, invA)
				color32RowSIMD(got, color, invA)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("color32Row color=%08x invA=%d dst byte=%d: %08x, want %08x", color, invA, i, got[i],
							want[i])
					}
				}
			}
		}
	})

	t.Run("blitMaskTranslucentRowExhaustive", func(t *testing.T) {
		// Every (devByte, coverage) pair at every source alpha, with the color's channels spread so each 16-bit lane
		// carries a different product.
		aa := make([]uint8, 256)
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		for srcA := range uint32(256) {
			pm := srcA<<24 | srcA<<16 | (srcA/2)<<8 | srcA/3
			for m := range 256 {
				for i := range want {
					aa[i] = uint8(m)
					want[i] = uint32(i) * 0x01010101
					got[i] = want[i]
				}
				blitMaskTranslucentRowGeneric(want, aa, pm, srcA)
				blitMaskTranslucentRowSIMD(got, aa, pm, srcA)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("blitMaskTranslucentRow pm=%08x srcA=%d m=%d dev byte=%d: %08x, want %08x", pm, srcA,
							m, i, got[i], want[i])
					}
				}
			}
		}
		// Every (pmByte, coverage) pair against a fixed spread of device bytes, so the color side sweeps too.
		for p := range 256 {
			pm := uint32(p) * 0x01010101
			for m := range 256 {
				for i := range want {
					aa[i] = uint8(m)
					want[i] = uint32(i) * 0x01010101
					got[i] = want[i]
				}
				blitMaskTranslucentRowGeneric(want, aa, pm, 255)
				blitMaskTranslucentRowSIMD(got, aa, pm, 255)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("blitMaskTranslucentRow pm=%08x m=%d dev byte=%d: %08x, want %08x", pm, m, i, got[i],
							want[i])
					}
				}
			}
		}
	})

	t.Run("interp256RowExhaustive", func(t *testing.T) {
		// Every (srcByte, dstByte, scale) triple: the row sweeps the destination byte, the middle loop the source
		// byte, and the outer loop every 0..256 scale.
		src := make([]uint32, 256)
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		for scale := range uint32(257) {
			for s := range 256 {
				for i := range want {
					src[i] = uint32(s) * 0x01010101
					want[i] = uint32(i) * 0x01010101
					got[i] = want[i]
				}
				interp256RowGeneric(want, src, scale)
				interp256RowSIMD(got, src, scale)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("interp256Row scale=%d src byte=%d dst byte=%d: %08x, want %08x", scale, s, i, got[i],
							want[i])
					}
				}
			}
		}
	})

	t.Run("premulRowExhaustive", func(t *testing.T) {
		// Every (colorByte, alpha) pair, with the three color channels offset from one another so each lane of the
		// vector carries a different product at every step.
		src := make([]uint32, 256)
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		for a := range uint32(256) {
			for i := range src {
				src[i] = a<<24 | uint32(i)<<16 | uint32(255-i)<<8 | uint32((i*7)&0xFF)
			}
			premulRowGeneric(want, src)
			premulRowSIMD(got, src)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("premulRow src=%08x: %08x, want %08x", src[i], got[i], want[i])
				}
			}
		}
	})

	t.Run("pmBlendRowExhaustive", func(t *testing.T) {
		// Every (srcAlpha, dstByte) pair at every 0..256 paint scale, which is every destination scale the
		// alphaMulInv256 divide can produce, then every (srcByte, dstByte) pair at a spread of scales.
		src := make([]uint32, 256)
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		for scale := range uint32(257) {
			for a := range uint32(256) {
				for i := range want {
					src[i] = a<<24 | uint32(i)<<16 | uint32(255-i)<<8 | uint32((i*7)&0xFF)
					want[i] = uint32(i) * 0x01010101
					got[i] = want[i]
				}
				pmBlendRowGeneric(want, src, scale)
				pmBlendRowSIMD(got, src, scale)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("pmBlendRow scale=%d srcA=%d dst byte=%d: %08x, want %08x", scale, a, i, got[i],
							want[i])
					}
				}
			}
		}
		for _, scale := range []uint32{0, 1, 2, 127, 128, 129, 255, 256} {
			for s := range 256 {
				for i := range want {
					src[i] = uint32(s) * 0x01010101
					want[i] = uint32(i) * 0x01010101
					got[i] = want[i]
				}
				pmBlendRowGeneric(want, src, scale)
				pmBlendRowSIMD(got, src, scale)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("pmBlendRow scale=%d src byte=%d dst byte=%d: %08x, want %08x", scale, s, i, got[i],
							want[i])
					}
				}
			}
		}
	})

	t.Run("blitRowLCD16Exhaustive", func(t *testing.T) {
		// Every one of the 65536 mask values, in rows of 256, against a spread of destinations and colors. The
		// destination set mixes opaque and translucent alphas so both arms of the min/max coverage rule run.
		mask := make([]uint16, 256)
		src := make([]uint32, 256)
		want := make([]uint32, 256)
		got := make([]uint32, 256)
		type color struct{ a, r, g, b uint32 }
		colors := []color{
			{0, 0, 0, 0},
			{1, 1, 1, 1},
			{128, 0x10, 0x20, 0x30},
			{254, 0xFE, 0x7F, 0x01},
			{255, 0xFF, 0xFF, 0xFF},
			{255, 0, 0xFF, 0},
		}
		dsts := []uint32{0xFF000000, 0xFFFFFFFF, 0xFF123456, 0x80404040, 0x00000000, 0xCC663311}
		for base := 0; base < 65536; base += 256 {
			for i := range mask {
				mask[i] = uint16(base + i)
			}
			for _, c := range colors {
				for _, d := range dsts {
					for i := range want {
						want[i] = d
						got[i] = d
					}
					blitRowLCD16Generic(want, mask, c.a, c.r, c.g, c.b)
					blitRowLCD16SIMD(got, mask, c.a, c.r, c.g, c.b)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("blitRowLCD16 src=%v dst=%08x mask=%04x: %08x, want %08x", c, d, mask[i], got[i],
								want[i])
						}
					}
					opaqueDst := c.r | c.g<<8 | c.b<<16 | 0xFF<<24
					for i := range want {
						want[i] = d
						got[i] = d
					}
					blitRowLCD16OpaqueGeneric(want, mask, c.r, c.g, c.b, opaqueDst)
					blitRowLCD16OpaqueSIMD(got, mask, c.r, c.g, c.b, opaqueDst)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("blitRowLCD16Opaque src=%v dst=%08x mask=%04x: %08x, want %08x", c, d, mask[i],
								got[i], want[i])
						}
					}
					// The shader-span row takes the color as a span of premultiplied words rather than as channel
					// arguments, so it rides the same mask sweep with the color splatted across the row.
					for i := range want {
						src[i] = opaqueDst
						want[i] = d
						got[i] = d
					}
					blendRowLCD16OpaqueGeneric(want, mask, src)
					blendRowLCD16OpaqueSIMD(got, mask, src)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("blendRowLCD16Opaque src=%08x dst=%08x mask=%04x: %08x, want %08x", opaqueDst, d,
								mask[i], got[i], want[i])
						}
					}
				}
			}
		}
	})
}

// TestBlitSIMDWiring locks that the goexperiment.simd build's init actually repointed every dispatch variable whose
// per-arch preference constant elects the simd kernel, so a refactor cannot silently fall back to the portable forms.
func TestBlitSIMDWiring(t *testing.T) {
	if !simdBlitSupported() {
		t.Skip("CPU lacks the features the simd blit rows require; dispatch stays on the portable forms")
	}
	for _, tc := range []struct {
		name   string
		prefer bool
		wired  any
		simd   any
	}{
		{"fillWords", preferSIMDFillWords, fillWordsFn, blitFillWordsFn(fillWordsSIMD)},
		{"fillBytes", preferSIMDFillBytes, fillBytesFn, blitFillBytesFn(fillBytesSIMD)},
		{"color32Row", preferSIMDColor32Row, color32RowFn, blitColorRowFn(color32RowSIMD)},
		{
			"blitMaskTranslucentRow", preferSIMDBlitMaskTranslucentRow, blitMaskTranslucentRowFn,
			blitMaskRowFn(blitMaskTranslucentRowSIMD),
		},
		{"interp256Row", preferSIMDInterp256Row, interp256RowFn, blitScaleRowFn(interp256RowSIMD)},
		{"premulRow", preferSIMDPremulRow, premulRowFn, spanRowFn(premulRowSIMD)},
		{"pmBlendRow", preferSIMDPMBlendRow, pmBlendRowFn, blitScaleRowFn(pmBlendRowSIMD)},
		{"blitRowLCD16", preferSIMDBlitRowLCD16, blitRowLCD16Fn, blitLCD16RowFn(blitRowLCD16SIMD)},
		{
			"blitRowLCD16Opaque", preferSIMDBlitRowLCD16Opaque, blitRowLCD16OpaqueFn,
			blitLCD16OpaqueRowFn(blitRowLCD16OpaqueSIMD),
		},
	} {
		if tc.prefer && reflect.ValueOf(tc.wired).Pointer() != reflect.ValueOf(tc.simd).Pointer() {
			t.Fatalf("%s: dispatch fn is not the simd kernel", tc.name)
		}
	}
}
