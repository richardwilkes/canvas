// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd

package imagecore

import "simd/archsimd"

// simdConvertSupported reports whether the CPU can run the simd conversion rows. The gate is AVX2 only: these kernels
// execute no FMA anywhere (the integer rows are all-integer, and the unpremultiply row is a divide, three multiplies,
// a round and a truncating convert — never a fused multiply-add), so the rule this package follows is the one raster
// and maskfilter follow, "gate on what the kernels actually execute". Requiring FMA would be inaccurate and would
// disable the path — and skip its equivalence tests — under Rosetta 2, which offers AVX2 without FMA. Everything they
// compile to is SSE4.1 or below (VPMULLD, VPMINUD, VPMULLW, VPSRLW/VPSLLW/VPSRLD/VPSLLD with immediate counts,
// VPMOVZXBW, VPMOVZXWD, VPSHUFB, VPALIGNR, VPCMPEQD/VPXOR for the "a != 0" mask, VPBLENDVB for the select, VDIVPS,
// VROUNDPS, VCVTDQ2PS, VCVTTPS2DQ) except the broadcasts, which archsimd emulates at AVX2. Unqualified CPUs keep the
// portable dispatch.
//
// Note what is deliberately *not* used here: Uint32x4.ConvertToFloat32 (VCVTUDQ2PS), Uint32x4.TruncToUint8/
// TruncToUint16 and Uint16x8.TruncToUint8/SaturateToUint8 are all AVX-512 in archsimd, so the byte values go through
// the signed convert (identical over 0..255) and the word→byte gather goes through VPSHUFB (see narrowWordAlphas).
func simdConvertSupported() bool { return archsimd.X86.AVX2() }

// Per-kernel dispatch preference: whether the simd conversion row is at least as fast as this build's default lane. On
// amd64 the only alternative is the portable scalar code in convert.go, which every one of these kernels beats.
// Measured on real amd64 hardware (Xeon W-2191B, darwin/amd64, benchstat n=10, 2026-08-20, via simd-bench.sh): -42%
// (alphaFromWords) to -83% (unpremul) on 256-pixel rows, geomean -73% — confirming and mostly widening the Rosetta 2
// estimates this landed with.
const (
	preferSIMDSwizzleWordRow    = true
	preferSIMDPremulWordRow     = true
	preferSIMDUnpremulWordRow   = true
	preferSIMDAlphaFromWordsRow = true
	preferSIMDFillBytesRow      = true
	preferSIMDGrayToWordsRow    = true
)

// The lane shifts go through the same hoisted-count helpers arm64 needs (see convert_simd_arm64.go for why it needs
// them). On amd64 there is nothing to hoist — the shift instructions take the count as an immediate — so the count is
// just the signed amount and the direction test folds away wherever the caller's count is the loop-invariant constant
// these kernels always pass.
type (
	shiftCount16 = int
	shiftCount32 = int
)

func rightShift16(n uint8) shiftCount16 { return -int(n) }
func leftShift16(n uint8) shiftCount16  { return int(n) }
func rightShift32(n uint8) shiftCount32 { return -int(n) }
func leftShift32(n uint8) shiftCount32  { return int(n) }

func shift16(x archsimd.Uint16x8, by shiftCount16) archsimd.Uint16x8 {
	if by < 0 {
		return x.ShiftAllRight(uint64(-by))
	}
	return x.ShiftAllLeft(uint64(by))
}

func shift32(x archsimd.Uint32x4, by shiftCount32) archsimd.Uint32x4 {
	if by < 0 {
		return x.ShiftAllRight(uint64(-by))
	}
	return x.ShiftAllLeft(uint64(by))
}

// wordAlphaNarrow is whatever loop-invariant state narrowWordAlphas needs on this arch. amd64's byte gather is VPSHUFB,
// which takes its index vector in a register, so the row builds the four indices once and threads them into the loop
// rather than rematerializing them per iteration.
type wordAlphaNarrow struct{ a, b, c, d archsimd.Int8x16 }

// wordAlphaIdx are the four VPSHUFB index patterns: each selects the top byte of its vector's four words into one
// quarter of the result and makes VPSHUFB zero the rest (the negative entries), so the four shuffles OR together into
// sixteen bytes in source order. They are plain data rather than package-level archsimd values on purpose — a
// package-level vector value would run AVX code during package init, on CPUs that failed simdConvertSupported and must
// never execute any of it — and keeping them package-level rather than composite literals inside newWordAlphaNarrow
// keeps the per-row setup to four loads instead of four stack copies.
var wordAlphaIdx = [4][16]int8{
	{3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, 3, 7, 11, 15, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 7, 11, 15},
}

// newWordAlphaNarrow builds the gather state once per row.
func newWordAlphaNarrow() wordAlphaNarrow {
	return wordAlphaNarrow{
		a: archsimd.LoadInt8x16Array(&wordAlphaIdx[0]),
		b: archsimd.LoadInt8x16Array(&wordAlphaIdx[1]),
		c: archsimd.LoadInt8x16Array(&wordAlphaIdx[2]),
		d: archsimd.LoadInt8x16Array(&wordAlphaIdx[3]),
	}
}

// narrowWordAlphas gathers the high byte of each of sixteen 32-bit lanes into sixteen consecutive bytes — the vector
// form of "byte(v >> 24)", since on a little-endian machine the top byte of a word is exactly that. The negative
// indices make VPSHUFB zero the lanes each vector does not own, so the OR of the four shuffles is exact. Lane order
// probed before use.
func narrowWordAlphas(a, b, c, d archsimd.Uint32x4, n wordAlphaNarrow) archsimd.Uint8x16 {
	return a.ReshapeToUint8s().PermuteOrZero(n.a).
		Or(b.ReshapeToUint8s().PermuteOrZero(n.b)).
		Or(c.ReshapeToUint8s().PermuteOrZero(n.c)).
		Or(d.ReshapeToUint8s().PermuteOrZero(n.d))
}
