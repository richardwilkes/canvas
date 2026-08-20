// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && amd64

package shaders

import "simd/archsimd"

// simdKernelsSupported reports whether the CPU can run the simd kernels. The amd64 baseline (GOAMD64=v1, SSE2)
// includes neither the 256-bit doubles madf4 works in nor a fused multiply-add; requiring AVX2+FMA (Haswell, 2013)
// covers everything the kernels compile to. Unqualified CPUs keep the default scalar dispatch.
func simdKernelsSupported() bool {
	return archsimd.X86.AVX2() && archsimd.X86.FMA()
}

// Per-kernel dispatch preference: whether the simd kernel is at least as fast as this build's default lane. On amd64
// the only alternative is the portable scalar code, which every arithmetic kernel beats, so they are all preferred (on
// arm64 the NEON assembly wins three of them back — see stage_simd_arm64.go, where the stages ported after the assembly
// are measured against this same portable code and win by 36-80%). move_src_dst is the one exception on both arches:
// its scalar form is four whole-array assignments the compiler already copies 128 bits at a time, so the vector
// spelling only adds a bounds-checked slice per quad and measures as a tie.
const (
	preferSIMDSeed                 = true
	preferSIMDClampX1              = true
	preferSIMDMatrixTranslate      = true
	preferSIMDMatrixScaleTranslate = true
	preferSIMDMatrixAffine         = true
	preferSIMDGradient2Stop        = true
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
// widen for it. On amd64 a 4-lane madf works in one 256-bit Float64x4.
type madfCoef = archsimd.Float64x4

// broadcastCoef pre-widens a scalar coefficient for madf4. float64(float32) is exact.
func broadcastCoef(c float32) madfCoef { return archsimd.BroadcastFloat64x4(float64(c)) }

// madf4 is the lanewise madf: float32(math.FMA(float64(f), float64(m), float64(a))). A single-precision vector FMA is
// NOT equivalent (the 48-bit product makes the 53-bit double rounding non-innocuous; ~1-ULP divergences exist), so
// every madf site widens to double (VCVTPS2PD; exact), performs one fused double FMA, and rounds to single
// (VCVTPD2PS) — the identical two-rounding sequence the scalar code performs. Locked by TestStageSIMDMatchesScalar.
func madf4(f archsimd.Float32x4, m64 madfCoef, a archsimd.Float32x4) archsimd.Float32x4 {
	return f.ConvertToFloat64().MulAdd(m64, a.ConvertToFloat64()).ConvertToFloat32()
}

// madf4v is madf4 for a multiplicand that varies per lane (the evenly-spaced gradient's gathered stop factors), so it
// cannot be hoisted and pre-widened. It is the identical widen / one fused double FMA / round-to-single sequence, with
// the multiplicand widened per call by the same exact VCVTPS2PD.
func madf4v(f, m, a archsimd.Float32x4) archsimd.Float32x4 {
	return f.ConvertToFloat64().MulAdd(m.ConvertToFloat64(), a.ConvertToFloat64()).ConvertToFloat32()
}
