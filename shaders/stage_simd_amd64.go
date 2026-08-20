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
