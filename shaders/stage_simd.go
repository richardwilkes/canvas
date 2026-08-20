// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package shaders

import "simd/archsimd"

// The goexperiment.simd kernels. They are written against archsimd's 128-bit types (the widest shape arm64 and amd64
// share) plus the per-arch madf4 helper (stage_simd_arm64.go / stage_simd_amd64.go), because bit-exactness with the
// scalar stages pins every operation's semantics:
//
//   - madf is float32(math.FMA(float64...)) — a double FMA rounded to single. madf4 reproduces that two-rounding
//     sequence exactly; a single-precision vector FMA would diverge by ~1 ULP on rare inputs (see the stage_arm64.s
//     header, and the anchors in TestStageSIMDMatchesScalar's history).
//   - minf/maxf are comparison-based ("comparison false -> second operand"); vector Min/Max propagate NaN (FMIN), so
//     clamps must be built from compare+select instead.
//
// Like the NEON kernels, these process all stride lanes unconditionally; lanes at or beyond z.n are scratch no
// consumer reads. The substitution changes throughput only, never rendered bytes — locked by
// TestStageSIMDMatchesScalar.

// init swaps the dispatch variables to the simd kernels. On arm64 the required operations are baseline NEON; on amd64
// simdKernelsSupported gates on AVX2+FMA, falling back to the default dispatch (NEON via stage_arm64.go on arm64,
// scalar elsewhere) when the hardware does not qualify.
func init() {
	if !simdKernelsSupported() {
		return
	}
	matrix4x5StageFn = matrix4x5StageSIMD
}

// matrix4x5StageSIMD is the vector matrix_4x5 kernel. The 16 multiplicand coefficients are pre-widened once
// (broadcastCoef) and the 4 translate-column addends broadcast as singles; each output row is then the same nested
// madf chain the scalar stage computes, with every intermediate rounded to single before the next madf re-widens it —
// exactly like the scalar nesting.
func matrix4x5StageSIMD(z *lanes) {
	mat := z.ctx.(*[20]float32)
	c00, c01, c02, c03 := broadcastCoef(mat[0]), broadcastCoef(mat[1]), broadcastCoef(mat[2]), broadcastCoef(mat[3])
	c10, c11, c12, c13 := broadcastCoef(mat[5]), broadcastCoef(mat[6]), broadcastCoef(mat[7]), broadcastCoef(mat[8])
	c20, c21, c22, c23 := broadcastCoef(mat[10]), broadcastCoef(mat[11]), broadcastCoef(mat[12]), broadcastCoef(mat[13])
	c30, c31, c32, c33 := broadcastCoef(mat[15]), broadcastCoef(mat[16]), broadcastCoef(mat[17]), broadcastCoef(mat[18])
	c04 := archsimd.BroadcastFloat32x4(mat[4])
	c14 := archsimd.BroadcastFloat32x4(mat[9])
	c24 := archsimd.BroadcastFloat32x4(mat[14])
	c34 := archsimd.BroadcastFloat32x4(mat[19])
	for o := 0; o < stride; o += 4 {
		r := archsimd.LoadFloat32x4(z.r[o:])
		g := archsimd.LoadFloat32x4(z.g[o:])
		b := archsimd.LoadFloat32x4(z.b[o:])
		a := archsimd.LoadFloat32x4(z.a[o:])
		madf4(r, c00, madf4(g, c01, madf4(b, c02, madf4(a, c03, c04)))).Store(z.r[o:])
		madf4(r, c10, madf4(g, c11, madf4(b, c12, madf4(a, c13, c14)))).Store(z.g[o:])
		madf4(r, c20, madf4(g, c21, madf4(b, c22, madf4(a, c23, c24)))).Store(z.b[o:])
		madf4(r, c30, madf4(g, c31, madf4(b, c32, madf4(a, c33, c34)))).Store(z.a[o:])
	}
}
