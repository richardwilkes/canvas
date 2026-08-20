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

// simdBlitSupported reports whether the CPU can run the simd blit rows. Like the span kernels' gate the requirement is
// AVX2 only ("gate on what the kernels actually execute" — see span_simd_amd64.go): these rows are all-integer, and
// everything they compile to is SSE4.1 or below (VPMULLW, VPMULLD, VPSRLW/VPSLLW/VPSRAW with immediate counts,
// VPMOVZXBW, VPMOVZXWD, VPUNPCKLWD, VPMINUW/VPMAXUW, VPBLENDVB) except the broadcasts, which need AVX2. Requiring FMA
// would be inaccurate and would disable the path under Rosetta 2, which offers AVX2 without it (the same reasoning the
// fp88 blur kernels' gate follows). Unqualified CPUs keep the portable dispatch.
func simdBlitSupported() bool { return archsimd.X86.AVX2() }

// Per-kernel dispatch preference: whether the simd blit row is at least as fast as this build's default lane. On amd64
// the only alternative is the portable scalar/SWAR code, which every one of these kernels beats.
const (
	preferSIMDFillWords              = true
	preferSIMDFillBytes              = true
	preferSIMDColor32Row             = true
	preferSIMDBlitMaskTranslucentRow = true
	preferSIMDInterp256Row           = true
	preferSIMDPremulRow              = true
	preferSIMDPMBlendRow             = true
	preferSIMDBlitRowLCD16           = true
	preferSIMDBlitRowLCD16Opaque     = true
	preferSIMDBlendRowLCD16Opaque    = true
)

// shiftI16 is the signed counterpart of shift16: VPSRAW is the arithmetic right shift blend32's int32 >>5 performs, and
// the count is an immediate here, so the direction test folds away for the loop-invariant counts these kernels pass.
func shiftI16(x archsimd.Int16x8, by shiftCount16) archsimd.Int16x8 {
	if by < 0 {
		return x.ShiftAllRight(uint64(-by))
	}
	return x.ShiftAllLeft(uint64(by))
}
