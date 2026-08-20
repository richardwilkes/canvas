// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd

package maskfilter

import "simd/archsimd"

// simdKernelsSupported reports whether the CPU can run the simd kernels. Everything the mask blur kernels compile to
// on arm64 — VUMULL, VUZP1, VUZP2, VEXT, VADD, VDUP and the 64-bit lane moves — is baseline NEON, present on every
// arm64 CPU Go supports.
func simdKernelsSupported() bool { return true }

// a8Narrow is whatever loop-invariant state storeA8 needs on this arch. arm64 gathers the high bytes with a single
// VUZP2, which needs no shuffle table, so the value is empty and costs nothing to thread through the kernels.
type a8Narrow = struct{}

// newA8Narrow builds the (empty) narrowing state once per rect.
func newA8Narrow() a8Narrow { return a8Narrow{} }

// narrowHiBytes gathers the high byte of each of v's eight u16 lanes into the low eight bytes of the result — the
// vector form of store's "uint8(v[i] >> 8)", since on a little-endian machine the high byte of a u16 is exactly its
// value shifted right by 8. VUZP2 takes the odd-indexed bytes of the concatenation of its two operands; passing v for
// both puts the eight wanted bytes in each half and the caller reads only the low half.
func narrowHiBytes(v archsimd.Uint16x8, _ a8Narrow) archsimd.Uint8x16 {
	b := v.ReshapeToUint8s()
	return b.ConcatOdd(b)
}

// widenA8 is loadA8's "uint16(from[i]) << 8" for the low eight bytes of b. Zipping b against zero bytes does the widen
// and the shift in one VZIP1: the result's u16 lanes each read a zero low byte and one source byte as their high byte,
// which little-endian is the byte shifted left 8. The obvious VUXTL-then-shift spelling costs four more instructions
// per group here, because archsimd's arm64 ShiftAllLeft is a variable VUSHL whose shift-amount vector the compiler
// rebuilds every iteration (Go's SSA has no loop-invariant code motion). Lane order probed before use.
func widenA8(b archsimd.Uint8x16) archsimd.Uint16x8 {
	var zero archsimd.Uint8x16
	return zero.InterleaveLo(b).ReshapeToUint16s()
}

// mulhiV is the vector mulhi: uint16((uint32(a)*uint32(b))>>16) per lane. arm64 has no 16-bit high-half multiply, so
// the products are taken widened — VUMULL over each half of the register, exact in 32 bits — and VUZP2 gathers the
// odd u16 halves of the two product vectors, which little-endian are precisely the >>16 halves. Probed exhaustively
// against the scalar mulhi (all 65536 multiplicands against a dense structured set of multipliers, plus randomized
// pairs) before this was built on.
func mulhiV(a, b archsimd.Uint16x8) archsimd.Uint16x8 {
	lo := a.MulWidenLo(b).ReshapeToUint16s()
	hi := a.HiToLo().MulWidenLo(b.HiToLo()).ReshapeToUint16s()
	return lo.ConcatOdd(hi)
}
