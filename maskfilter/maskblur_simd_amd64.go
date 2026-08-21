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

// simdKernelsSupported reports whether the CPU can run the simd kernels. These kernels are all-integer, so unlike the
// shader stage kernels they need no FMA: every instruction they compile to is AVX (VPMULHUW, VPADDW, VPALIGNR,
// VPSHUFB, VPMOVZXBW, VPSLLW, VPINSRQ, VPEXTRQ) except the broadcasts, which archsimd emulates at AVX2. AVX2 (Haswell,
// 2013) is therefore the gate; unqualified CPUs keep the portable fp88 dispatch.
//
// Note what is deliberately *not* used here: Uint16x8.TruncToUint8 and Uint16x8.SaturateToUint8 are both AVX-512
// (VPMOVWB / VPMOVUSWB) in archsimd, so the u16 -> u8 narrowing goes through VPSHUFB instead (see narrowHiBytes).
func simdKernelsSupported() bool { return archsimd.X86.AVX2() }

// a8Narrow is whatever loop-invariant state storeA8 needs on this arch. amd64's byte gather is VPSHUFB, which takes its
// index vector in a register, so the driver builds the index once per rect and threads it down to the kernels rather
// than rematerializing it inside the row and column loops.
type a8Narrow = archsimd.Int8x16

// newA8Narrow builds the narrowing state once per rect: the VPSHUFB indices that select the high byte of each u16 lane.
// It must not be a package-level variable — a package-level archsimd value would run AVX code during package init, on
// CPUs that failed simdKernelsSupported and must never execute any of it.
func newA8Narrow() a8Narrow {
	idx := [16]int8{1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1}
	return archsimd.LoadInt8x16Array(&idx)
}

// narrowHiBytes gathers the high byte of each of v's eight u16 lanes into the low eight bytes of the result — the
// vector form of store's "uint8(v[i] >> 8)", since on a little-endian machine the high byte of a u16 is exactly its
// value shifted right by 8. The negative indices make VPSHUFB zero the upper half, which no caller reads.
func narrowHiBytes(v archsimd.Uint16x8, n a8Narrow) archsimd.Uint8x16 {
	return v.ReshapeToUint8s().PermuteOrZero(n)
}

// widenA8 is loadA8's "uint16(from[i]) << 8" for the low eight bytes of b: VPMOVZXBW then a VPSLLW whose shift amount
// the compiler folds into the instruction as an immediate, so there is nothing to hoist out of the caller's loop.
// (arm64 needs a different spelling; see the note there.)
func widenA8(b archsimd.Uint8x16) archsimd.Uint16x8 { return b.ExtendLo8ToUint16().ShiftAllLeft(8) }

// mulhiV is the vector mulhi: uint16((uint32(a)*uint32(b))>>16) per lane, which is exactly what VPMULHUW computes.
// Probed exhaustively against the scalar mulhi (all 65536 multiplicands against a dense structured set of multipliers,
// plus randomized pairs) before this was built on.
func mulhiV(a, b archsimd.Uint16x8) archsimd.Uint16x8 { return a.MulHigh(b) }
