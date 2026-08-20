// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd

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

// The image-filter kernel stages (filterkernels.go) ported afterwards, all of which likewise have no NEON twin, so
// their default lane is the portable scalar loop and the vector kernel wins outright — there is no measurement here
// that argues for the default lane. Benchstat on an M4 Max (stage_bench_test.go, both build modes, n=6, every
// comparison p=0.002): arith_blend -78%, morph_sparse_max_agg -74%, morph_max_agg -73%, matrix_conv_accum -71%
// (unpremul form) / -63% (convolve-alpha form), matrix_conv_coords -70%, normal_set_coords -69%, morph_plus_coords
// -67%, morph_take_plus and morph_sparse_agg_minus -66%, morph_return -62%, normal_filter -62%,
// matrix_conv_finalize -60%, morph_agg_init -52%; geomean -67%. The displacement and magnifier stages were measured
// afterwards, in their own pair of runs (n=6, p=0.002 both): displacement -73%, magnifier -44%. The magnifier is the
// weakest of the set because its per-lane branch is not hoistable, so the vector form always evaluates both the
// circular arm (a square root) and the linear one where the scalar takes just the arm each lane needs; it still wins
// by a wide margin.
//
// The morphology and convolution stages run many times per chunk — one plus-coords/take-plus/max-agg triple per
// morphology radius step (up to 14) and one coords/accum pair per convolution tap (25 for a 5x5 kernel) — so those
// percentages multiply through an image-filter draw rather than being paid once.
const (
	preferSIMDMorphAggInit        = true
	preferSIMDMorphPlusCoords     = true
	preferSIMDMorphTakePlus       = true
	preferSIMDMorphMaxAgg         = true
	preferSIMDMorphSparseAggMinus = true
	preferSIMDMorphSparseMaxAgg   = true
	preferSIMDMorphReturn         = true
	preferSIMDDisplacement        = true
	preferSIMDNormalSetCoords     = true
	preferSIMDNormalFilter        = true
	preferSIMDMagnifier           = true
	preferSIMDMatrixConvCoords    = true
	preferSIMDMatrixConvAccum     = true
	preferSIMDMatrixConvFinalize  = true
	preferSIMDArithBlend          = true
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

// exprMulAdd4 is the lanewise form of the *plain Go expression* "f*m + a" as this build's compiler lowers it — not the
// explicit madf, which is a different operation. Go permits an implementation to contract such an expression into a
// single fused operation, and the arm64 compiler does: the image-filter kernels' "sum += c*k", "sum*gain + bias" and
// arithmetic-blend chain all become FMADDS, a *single-precision* fused multiply-add. So the vector twin must fuse too.
// Spelling it as a separate Mul and Add instead diverges from the scalar stage on ~1.5% of hostile-float inputs
// (measured), and spelling it as madf4 diverges the other way — madf's two-rounding widen/FMA/narrow sequence is what
// float32(math.FMA(float64...)) means, and an explicit conversion is precisely what the compiler may not contract away.
func exprMulAdd4(f, m, a archsimd.Float32x4) archsimd.Float32x4 { return f.MulAdd(m, a) }

// exprMulSub4 is exprMulAdd4's subtractive shape: the plain Go expression "a - f*m" as this build's compiler lowers
// it. arm64 contracts that one too, into FMSUBS — verified by disassembling the magnifier stage, where "2 - edgeInset"
// re-derives the edgeInset product fused even though the separately rounded product is still computed for the
// neighboring comparison. archsimd exposes no fused multiply-subtract on Float32x4, so it is spelled as a negated
// multiplicand into MulAdd: FMSUB(f, m, a) is defined as a - f*m under one rounding, which is exactly FMA(-f, m, a)
// because negation is an exact sign flip and -(f*m) == (-f)*m for every float class (NaN payloads aside, and those are
// unobservable downstream). Spelling it as a separate Mul and Sub instead diverges from the scalar stage on hostile
// inputs — a negative control the magnifier subtest catches immediately.
func exprMulSub4(f, m, a archsimd.Float32x4) archsimd.Float32x4 { return f.Neg().MulAdd(m, a) }
