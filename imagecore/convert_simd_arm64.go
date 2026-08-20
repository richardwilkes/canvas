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

// simdConvertSupported reports whether the CPU can run the simd conversion rows. Everything they compile to on arm64
// (the 128-bit loads/stores, the integer lane arithmetic, MUL, UMIN, UXTL, EXT, UZP2, USHL, the CMEQ/BSL
// compare-and-select pair, and on the float side SCVTF, FDIV, FRINTN, FCVTZS) is baseline NEON, present on every arm64
// CPU Go supports.
func simdConvertSupported() bool { return true }

// Per-kernel dispatch preference: whether the simd conversion row is at least as fast as this build's default lane.
// None of these rows has a hand-written NEON lane to lose to — the alternative is the portable scalar form in
// convert.go — and every one of them beats it, by -62% (swizzle) to -91% (fillBytes) on 256-pixel rows, geomean -78%
// (measured on an M4 Max, darwin/arm64, benchstat n=10, 2026-08-20), so every one is preferred here.
const (
	preferSIMDSwizzleWordRow    = true
	preferSIMDPremulWordRow     = true
	preferSIMDUnpremulWordRow   = true
	preferSIMDAlphaFromWordsRow = true
	preferSIMDFillBytesRow      = true
	preferSIMDGrayToWordsRow    = true
)

// The lane shifts go through a hoisted count because arm64 has no shift-by-immediate in this API:
// ShiftAllLeft/ShiftAllRight lower to a VDUP of the (negated) count plus VUSHL, and the compiler re-materializes that
// VDUP on every iteration instead of lifting it out of the loop (Go's SSA has no loop-invariant code motion) — three
// instructions per shift in a body that has half a dozen of them. Broadcasting the count once and issuing the bare
// VUSHL is exactly the instruction ShiftAllRight would have emitted, so the results are unchanged: a negative count is
// a logical right shift for an unsigned vector, and every count these kernels use (0, 8, 16, 24) is inside the lane
// width. This mirrors raster's span_simd_arm64.go; the two packages cannot share unexported helpers.
type (
	shiftCount16 = archsimd.Int16x8
	shiftCount32 = archsimd.Int32x4
)

func rightShift16(n uint8) shiftCount16 { return archsimd.BroadcastInt16x8(-int16(n)) }
func leftShift16(n uint8) shiftCount16  { return archsimd.BroadcastInt16x8(int16(n)) }
func rightShift32(n uint8) shiftCount32 { return archsimd.BroadcastInt32x4(-int32(n)) }
func leftShift32(n uint8) shiftCount32  { return archsimd.BroadcastInt32x4(int32(n)) }

func shift16(x archsimd.Uint16x8, by shiftCount16) archsimd.Uint16x8 { return x.Shift(by) }
func shift32(x archsimd.Uint32x4, by shiftCount32) archsimd.Uint32x4 { return x.Shift(by) }

// wordAlphaNarrow is whatever loop-invariant state narrowWordAlphas needs on this arch. arm64 gathers the bytes with
// VUZP2, which needs no shuffle table, so the value is empty and costs nothing to thread through the kernel.
type wordAlphaNarrow = struct{}

// newWordAlphaNarrow builds the (empty) gather state once per row.
func newWordAlphaNarrow() wordAlphaNarrow { return wordAlphaNarrow{} }

// narrowWordAlphas gathers the high byte of each of sixteen 32-bit lanes into sixteen consecutive bytes — the vector
// form of "byte(v >> 24)", since on a little-endian machine the top byte of a word is exactly that. VUZP2 takes the
// odd-indexed elements of the concatenation of its two operands, so one pass over the 16-bit view keeps each word's
// high half and a second pass over the byte view keeps that half's high byte, in source order. Lane order probed
// before use.
func narrowWordAlphas(a, b, c, d archsimd.Uint32x4, _ wordAlphaNarrow) archsimd.Uint8x16 {
	hiAB := a.ReshapeToUint16s().ConcatOdd(b.ReshapeToUint16s())
	hiCD := c.ReshapeToUint16s().ConcatOdd(d.ReshapeToUint16s())
	return hiAB.ReshapeToUint8s().ConcatOdd(hiCD.ReshapeToUint8s())
}
