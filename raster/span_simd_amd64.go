// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && amd64

package raster

import "simd/archsimd"

// simdKernelsSupported reports whether the CPU can run the simd span kernels. The gate is AVX2 only: the raster
// kernels use no FMA anywhere (the clamps are compare+select, toUnorm is Round then the truncating convert, and the
// row kernels are all-integer), so the rule this package and maskfilter follow is "gate on what the kernels actually
// execute" — requiring FMA here would be inaccurate and would disable the path (and skip its equivalence tests) under
// Rosetta 2, which offers AVX2 without FMA. Only the shaders package requires FMA, because its madf4 genuinely
// executes one. Unqualified CPUs keep the portable dispatch.
func simdKernelsSupported() bool {
	return archsimd.X86.AVX2()
}

// simdKernelsPreferred reports whether the simd span kernels are the fastest lane on this hardware, which is what the
// init-time dispatch gates on. On amd64 the only alternative is the portable scalar code, which the archsimd kernels
// beat by 2.2-4.6x, so preferred is exactly supported (on arm64 the NEON assembly wins instead — see
// span_simd_arm64.go).
func simdKernelsPreferred() bool { return simdKernelsSupported() }

// The lane shifts of the two integer kernels go through the same hoisted-count helpers arm64 needs (see
// span_simd_arm64.go for why it needs them). On amd64 there is nothing to hoist — VPSRLW/VPSLLD take the count as an
// immediate — so the count is just the signed amount and the direction test folds away wherever the caller's count is
// the loop-invariant constant these kernels always pass.
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
