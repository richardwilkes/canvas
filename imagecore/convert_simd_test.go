// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package imagecore

import (
	"bytes"
	"math/rand/v2"
	"reflect"
	"testing"
)

// convertSIMDRowLen draws a row length for the randomized subtests: mostly lengths that cover every chunk/tail split of
// the four- and sixteen-pixel vector bodies, and one in four long enough to exercise the fill kernel's unrolled body
// and its remainder.
func convertSIMDRowLen(rng *rand.Rand) int {
	if rng.IntN(4) == 0 {
		return 1 + rng.IntN(300)
	}
	return 1 + rng.IntN(67)
}

// convertSIMDWords returns n random source words.
func convertSIMDWords(rng *rand.Rand, n int) []uint32 {
	w := make([]uint32, n)
	for i := range w {
		w[i] = rng.Uint32()
	}
	return w
}

// convertSIMDGuarded returns a destination buffer of n usable bytes followed by a 32-byte guard filled with a
// recognizable pattern, plus the usable prefix. A kernel that runs past the row it was handed corrupts the guard, which
// convertSIMDCheckGuard catches — the vector kernels store whole registers, so their last store is the one place an
// off-by-one could write outside the row.
func convertSIMDGuarded(n int) (buf, row []byte) {
	buf = make([]byte, n+32)
	for i := n; i < len(buf); i++ {
		buf[i] = 0x5A
	}
	return buf, buf[:n]
}

func convertSIMDCheckGuard(t *testing.T, buf []byte, n int) {
	t.Helper()
	for i := n; i < len(buf); i++ {
		if buf[i] != 0x5A {
			t.Fatalf("kernel wrote past the row at byte %d of %d", i, n)
		}
	}
}

// TestConvertSIMDMatchesScalar drives every simd conversion row and its portable twin over identical inputs and
// requires bitwise identity. The randomized subtests cover every chunk/tail split; the exhaustive subtests enumerate
// the complete integer domains — every (channel, alpha) pair for the two alpha lanes, every byte for the gather and
// the gray expansion — which proves bit-exactness outright rather than sampling for it.
func TestConvertSIMDMatchesScalar(t *testing.T) {
	if !simdConvertSupported() {
		t.Skip("CPU lacks the features the simd conversion rows require; dispatch stays on the portable forms")
	}
	rng := rand.New(rand.NewPCG(41, 42))

	randomRows := func(t *testing.T, name string, fn, ref convertWordRowFn) {
		t.Helper()
		for _, swap := range []bool{false, true} {
			for range 2048 {
				n := convertSIMDRowLen(rng)
				src := convertSIMDWords(rng, n)
				wantBuf, want := convertSIMDGuarded(4 * n)
				gotBuf, got := convertSIMDGuarded(4 * n)
				ref(want, src, swap)
				fn(got, src, swap)
				if !bytes.Equal(got, want) {
					t.Fatalf("%s swap=%v n=%d: got %x, want %x", name, swap, n, got, want)
				}
				convertSIMDCheckGuard(t, wantBuf, 4*n)
				convertSIMDCheckGuard(t, gotBuf, 4*n)
			}
		}
	}

	// exhaustiveRows enumerates every (channel value, alpha) pair against the two alpha kernels. Each row sweeps the
	// channel byte across its whole domain under one fixed alpha, in two channel layouts: the flat one, where all three
	// color channels carry the same value (so every (value, alpha) pair is seen by every channel), and a skewed one
	// where the three differ (so a kernel that leaked one channel's product into another's lane cannot pass). The row
	// is then re-run at the three lengths whose vector tails are non-empty.
	exhaustiveRows := func(t *testing.T, name string, fn, ref convertWordRowFn) {
		t.Helper()
		src := make([]uint32, 256)
		want := make([]byte, 4*256)
		got := make([]byte, 4*256)
		for _, skew := range []bool{false, true} {
			for a := range uint32(256) {
				for i := range src {
					r := uint32(i)
					g, b := r, r
					if skew {
						g = 255 - r
						b = r * 7 & 0xFF
					}
					src[i] = r | g<<8 | b<<16 | a<<24
				}
				for _, swap := range []bool{false, true} {
					for _, n := range []int{256, 255, 254, 253} {
						ref(want, src[:n], swap)
						fn(got, src[:n], swap)
						if !bytes.Equal(got[:4*n], want[:4*n]) {
							t.Fatalf("%s skew=%v a=%d swap=%v n=%d: got %x, want %x",
								name, skew, a, swap, n, got[:4*n], want[:4*n])
						}
					}
				}
			}
		}
	}

	t.Run("swizzleWordRow", func(t *testing.T) {
		randomRows(t, "swizzleWordRow", swizzleWordRowSIMD, swizzleWordRowGeneric)
	})

	t.Run("premulWordRow", func(t *testing.T) {
		randomRows(t, "premulWordRow", premulWordRowSIMD, premulWordRowGeneric)
	})

	t.Run("premulWordRowExhaustive", func(t *testing.T) {
		exhaustiveRows(t, "premulWordRow", premulWordRowSIMD, premulWordRowGeneric)
	})

	t.Run("unpremulWordRow", func(t *testing.T) {
		randomRows(t, "unpremulWordRow", unpremulWordRowSIMD, unpremulWordRowGeneric)
	})

	t.Run("unpremulWordRowExhaustive", func(t *testing.T) {
		exhaustiveRows(t, "unpremulWordRow", unpremulWordRowSIMD, unpremulWordRowGeneric)
	})

	t.Run("alphaFromWordsRow", func(t *testing.T) {
		for range 2048 {
			n := convertSIMDRowLen(rng)
			src := convertSIMDWords(rng, n)
			wantBuf, want := convertSIMDGuarded(n)
			gotBuf, got := convertSIMDGuarded(n)
			alphaFromWordsRowGeneric(want, src)
			alphaFromWordsRowSIMD(got, src)
			if !bytes.Equal(got, want) {
				t.Fatalf("alphaFromWordsRow n=%d: got %x, want %x", n, got, want)
			}
			convertSIMDCheckGuard(t, wantBuf, n)
			convertSIMDCheckGuard(t, gotBuf, n)
		}
	})

	t.Run("alphaFromWordsRowExhaustive", func(t *testing.T) {
		// Every alpha byte, at every position of a full sixteen-pixel group and of each of its tails.
		src := make([]uint32, 256)
		want := make([]byte, 256)
		got := make([]byte, 256)
		for i := range src {
			src[i] = uint32(i)<<24 | 0x00ABCDEF
		}
		for n := 1; n <= 256; n++ {
			alphaFromWordsRowGeneric(want, src[:n])
			alphaFromWordsRowSIMD(got, src[:n])
			if !bytes.Equal(got[:n], want[:n]) {
				t.Fatalf("alphaFromWordsRow n=%d: got %x, want %x", n, got[:n], want[:n])
			}
		}
	})

	t.Run("fillBytesRow", func(t *testing.T) {
		for range 2048 {
			n := convertSIMDRowLen(rng)
			v := byte(rng.Uint32())
			wantBuf, want := convertSIMDGuarded(n)
			gotBuf, got := convertSIMDGuarded(n)
			fillBytesRowGeneric(want, v)
			fillBytesRowSIMD(got, v)
			if !bytes.Equal(got, want) {
				t.Fatalf("fillBytesRow n=%d v=%d: got %x, want %x", n, v, got, want)
			}
			convertSIMDCheckGuard(t, wantBuf, n)
			convertSIMDCheckGuard(t, gotBuf, n)
		}
	})

	t.Run("grayToWordsRow", func(t *testing.T) {
		for range 2048 {
			n := convertSIMDRowLen(rng)
			src := make([]byte, n)
			for i := range src {
				src[i] = byte(rng.Uint32())
			}
			wantBuf, want := convertSIMDGuarded(4 * n)
			gotBuf, got := convertSIMDGuarded(4 * n)
			grayToWordsRowGeneric(want, src)
			grayToWordsRowSIMD(got, src)
			if !bytes.Equal(got, want) {
				t.Fatalf("grayToWordsRow n=%d: got %x, want %x", n, got, want)
			}
			convertSIMDCheckGuard(t, wantBuf, 4*n)
			convertSIMDCheckGuard(t, gotBuf, 4*n)
		}
	})

	t.Run("grayToWordsRowExhaustive", func(t *testing.T) {
		// Every gray byte, at every position of a full sixteen-pixel group and of each of its tails.
		src := make([]byte, 256)
		want := make([]byte, 4*256)
		got := make([]byte, 4*256)
		for i := range src {
			src[i] = byte(i)
		}
		for n := 1; n <= 256; n++ {
			grayToWordsRowGeneric(want, src[:n])
			grayToWordsRowSIMD(got, src[:n])
			if !bytes.Equal(got[:4*n], want[:4*n]) {
				t.Fatalf("grayToWordsRow n=%d: got %x, want %x", n, got[:4*n], want[:4*n])
			}
		}
	})
}

// TestConvertSIMDWiring locks that the goexperiment.simd build's init actually repointed every dispatch variable whose
// per-arch preference constant elects the simd kernel, so a refactor cannot silently fall back to the portable forms.
func TestConvertSIMDWiring(t *testing.T) {
	if !simdConvertSupported() {
		t.Skip("CPU lacks the features the simd conversion rows require; dispatch stays on the portable forms")
	}
	for _, tc := range []struct {
		wired  any
		simd   any
		name   string
		prefer bool
	}{
		{swizzleWordRowFn, convertWordRowFn(swizzleWordRowSIMD), "swizzleWordRow", preferSIMDSwizzleWordRow},
		{premulWordRowFn, convertWordRowFn(premulWordRowSIMD), "premulWordRow", preferSIMDPremulWordRow},
		{unpremulWordRowFn, convertWordRowFn(unpremulWordRowSIMD), "unpremulWordRow", preferSIMDUnpremulWordRow},
		{
			alphaFromWordsRowFn, alphaFromWordRowFn(alphaFromWordsRowSIMD),
			"alphaFromWordsRow", preferSIMDAlphaFromWordsRow,
		},
		{fillBytesRowFn, fillByteRowFn(fillBytesRowSIMD), "fillBytesRow", preferSIMDFillBytesRow},
		{grayToWordsRowFn, grayToWordRowFn(grayToWordsRowSIMD), "grayToWordsRow", preferSIMDGrayToWordsRow},
	} {
		if tc.prefer && reflect.ValueOf(tc.wired).Pointer() != reflect.ValueOf(tc.simd).Pointer() {
			t.Fatalf("%s: dispatch fn is not the simd kernel", tc.name)
		}
	}
}
