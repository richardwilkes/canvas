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

// Per-kernel dispatch preference: whether the simd kernel is at least as fast as this build's default lane, which on
// arm64 is the NEON assembly in stage_arm64.s. The equivalence tests gate on simdKernelsSupported and so lock every
// kernel bit-for-bit regardless of these; init consults them so the experiment build never dispatches a measured
// regression. Benchstat over 10 quiet runs on an M4 Max (stage_bench_test.go, both build modes): seed -36% and
// matrix4x5 -47% (its default is scalar; it has no NEON twin) are clear wins; clampX1, matrixScaleTranslate, and
// gradientEvenly are statistical ties, wired anyway to soak the lane that outlives the assembly; matrixTranslate
// (+23%), matrixAffine (+6%), and gradient2Stop (+38%) lose to the assembly — the archsimd spelling of their madf
// chains assembles FCVTL2/FCVTN2 idioms from several instructions where the assembly has one — so those three keep
// NEON. Revisit when the codegen improves.
//
// The stages ported after the assembly (everything from preferSIMDMaskApply down) have no NEON twin, so their default
// lane is the portable scalar loop and the vector kernel wins outright: on the same M4 Max, clamp_01 -80%, clamp_gamut
// -80%, apply_vector_mask -71%, bilinear_nx/ny -67% and px/py -53%, move_dst_src -64%, set_rgb -62%, premul -59%,
// unpremul -57%, accumulate -50%, scale_1_float -36%. move_src_dst is the lone exception at ±0% (p=0.31): its scalar
// form is four whole-array assignments the compiler already turns into 128-bit copies, so the vector spelling only adds
// a bounds-checked slice per quad — it stays on the default lane.
const (
	preferSIMDSeed                 = true
	preferSIMDClampX1              = true
	preferSIMDMatrixTranslate      = false
	preferSIMDMatrixScaleTranslate = true
	preferSIMDMatrixAffine         = false
	preferSIMDGradient2Stop        = false
	preferSIMDGradientEvenly       = true
	preferSIMDMatrix4x5            = true
	preferSIMDMaskApply            = true
	preferSIMDClamp01              = true
	preferSIMDClampGamut           = true
	preferSIMDPremul               = true
	preferSIMDUnpremul             = true
	preferSIMDScale1Float          = true
	preferSIMDSetRGB               = true
	preferSIMDMoveSrcDst           = false
	preferSIMDMoveDstSrc           = true
	preferSIMDBilinear             = true
	preferSIMDAccumulate           = true
)

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

// madf4v is madf4 for a multiplicand that varies per lane (the evenly-spaced gradient's gathered stop factors), so it
// cannot be hoisted and pre-widened. It is the identical widen / one fused double FMA / round-to-single sequence, with
// the multiplicand widened per call by the same exact FCVTL.
func madf4v(f, m, a archsimd.Float32x4) archsimd.Float32x4 {
	fLo, fHi := widen4(f)
	mLo, mHi := widen4(m)
	aLo, aHi := widen4(a)
	return narrow4(fLo.MulAdd(mLo, aLo), fHi.MulAdd(mHi, aHi))
}
