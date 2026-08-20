// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && arm64

package shaders

import "simd/archsimd"

// simdKernelsSupported reports whether the CPU can run the simd kernels. Everything the kernels compile to on arm64
// (FMLA.D2, the converts, the loads/stores) is baseline NEON, present on every arm64 CPU Go supports.
func simdKernelsSupported() bool { return true }

// madfCoef is a loop-invariant madf multiplicand, pre-widened to double once so the chunk loop pays no per-iteration
// widen for it — the same hoist the NEON kernels perform (stage_arm64.s widens each coefficient with one FCVTL up
// front). On arm64 a 4-lane madf works in two Float64x2 halves.
type madfCoef = archsimd.Float64x2

// broadcastCoef pre-widens a scalar coefficient for madf4. float64(float32) is exact.
func broadcastCoef(c float32) madfCoef { return archsimd.BroadcastFloat64x2(float64(c)) }

// widen4 splits a Float32x4 into two Float64x2 halves (FCVTL; exact, like the scalar float64(float32) conversions
// inside madf).
func widen4(v archsimd.Float32x4) (lo, hi archsimd.Float64x2) {
	return v.ConvertLo2ToFloat64(), v.HiToLo().ConvertLo2ToFloat64()
}

// narrow4 rounds two Float64x2 halves to single and recombines them into a Float32x4. FCVTN leaves each half's pair in
// its low lanes; the 64-bit interleave stitches [lo0 lo1 hi0 hi1] back together (the FCVTN2 the assembler kernels use,
// spelled with the operations archsimd exposes).
func narrow4(lo, hi archsimd.Float64x2) archsimd.Float32x4 {
	l := lo.ConvertToFloat32().ToBits().ReshapeToUint64s()
	h := hi.ConvertToFloat32().ToBits().ReshapeToUint64s()
	return l.InterleaveLo(h).ReshapeToUint32s().BitsToFloat32()
}

// madf4 is the lanewise madf: float32(math.FMA(float64(f), float64(m), float64(a))). A single-precision FMLA is NOT
// equivalent (the 48-bit product makes the 53-bit double rounding non-innocuous; ~1-ULP divergences exist), so like
// the NEON kernels every madf site widens to double, performs one fused double FMA, and rounds to single — the
// identical two-rounding sequence the scalar code performs. Locked by TestStageSIMDMatchesScalar.
func madf4(f archsimd.Float32x4, m64 madfCoef, a archsimd.Float32x4) archsimd.Float32x4 {
	fLo, fHi := widen4(f)
	aLo, aHi := widen4(a)
	return narrow4(fLo.MulAdd(m64, aLo), fHi.MulAdd(m64, aHi))
}
