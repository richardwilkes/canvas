// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd

package filtercore

import "simd/archsimd"

// simdKernelsSupported reports whether the CPU can run the simd blur kernels. The gate is AVX2 only: these kernels
// execute no FMA anywhere — exprMulAdd4 below multiplies and adds separately, because that is what the scalar body
// compiles to at this module's GOAMD64=v1 baseline — so the rule this package follows is the one raster, imagecore and
// maskfilter follow, "gate on what the kernels actually execute". Requiring FMA would be inaccurate and would disable
// the path, and skip its equivalence tests, under Rosetta 2, which offers AVX2 without FMA. Everything they compile to
// is SSE4.1 or below (VMULPS, VADDPS, VCMPPS, VBLENDVPS, VCVTDQ2PS, VCVTTPS2DQ, VPMOVZXBW, VPMOVZXWD, VPADDD, VPSUBD,
// VPMULUDQ, VPSRLQ, VPSLLQ, VPOR, VPSHUFB and the 128-bit loads/stores) except the broadcasts, which archsimd emulates
// at AVX2. Unqualified CPUs keep the portable dispatch.
//
// Note what is deliberately *not* used here: Uint32x4.TruncToUint8 and SaturateToUint8 are AVX-512 in archsimd, so the
// word-to-byte gather goes through VPSHUFB (see packLowBytes), and the 0..255 widen goes through the signed convert
// (identical over that range) rather than Uint32x4.ConvertToFloat32, which is AVX-512 as well.
func simdKernelsSupported() bool { return archsimd.X86.AVX2() }

// Per-kernel dispatch preference: whether the simd kernel is at least as fast as this build's default lane. On amd64
// the only alternative is the portable scalar body in blurengine.go, which both kernels beat for the same reasons they
// beat it on arm64 — a tap drops from four loads plus four multiply-adds to one of each, and the three-box pipeline's
// twelve adds, twelve subtracts and four divides drop to three, three and one.
//
// Measured on real amd64 hardware (Xeon W-2191B, darwin/amd64, benchstat n=10, 2026-08-20, via simd-bench.sh): the
// Gaussian segment wins -73% at every window, the three-box segment -71% at both windows, and the end-to-end passes
// -73% (Gaussian) and -69% (three-box) — confirming and widening the Rosetta 2 estimates this landed with.
const (
	preferSIMDGaussianSegment = true
	preferSIMDThreeBoxSegment = true
)

// exprMulAdd4 is the lanewise form of the *plain Go expression* "f*m + a" as this build's compiler lowers it. Go
// permits an implementation to contract such an expression into a single fused operation; at this module's GOAMD64=v1
// baseline the amd64 compiler does not, emitting MULSS then ADDSS for both of gaussianBlurSegmentGeneric's float sites
// ("sum[c] += p.srcBuffer[s][c] * k" and "v := sum[c]*255 + 0.5"), so the vector twin multiplies and adds separately
// too — VMULPS and VADDPS are the same IEEE single-precision operations, lane for lane. (arm64 contracts both into
// FMADDS, which is why its exprMulAdd4 fuses and why the goldens are captured per platform.) Fusing here instead
// diverges from the scalar body on hostile-float inputs, which TestBlurEngineSIMDContractionNegativeControl drives
// explicitly — on a CPU with FMA, since VFMADD213PS is not in the AVX2-only set the kernels themselves need.
//
// Raising GOAMD64 changes this: at v3+ the compiler contracts these expressions into VFMADD231SS. That already moves
// the *default* build's rendered output off the captured goldens, so v3 is not a supported configuration for this
// module; should it become one, this helper and the scalar body have to be revisited together.
// TestBlurEngineSIMDMatchesScalar is the tripwire — it compares against whatever the scalar twin in the same binary
// does.
func exprMulAdd4(f, m, a archsimd.Float32x4) archsimd.Float32x4 { return f.Mul(m).Add(a) }

// lowByteIdx is packLowBytes' VPSHUFB index pattern: it selects the low byte of each of the four words into the first
// four bytes of the result and makes VPSHUFB zero the rest (the negative entries), which is all the caller reads. It is
// plain data rather than a package-level archsimd value on purpose — a package-level vector value would run AVX code
// during package init, on CPUs that failed simdKernelsSupported and must never execute any of it.
var lowByteIdx = [16]int8{0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}

// lowByteGather is whatever loop-invariant state packLowBytes needs on this arch. amd64's byte gather is VPSHUFB, which
// takes its index vector in a register, so the segment builds the index once and threads it into the loop rather than
// rematerializing it per sample.
type lowByteGather = archsimd.Int8x16

// newLowByteGather builds the gather state once per segment.
func newLowByteGather() lowByteGather { return archsimd.LoadInt8x16Array(&lowByteIdx) }

// packLowBytes gathers the low byte of each of four 32-bit lanes into one 8888 word, in lane order — the vector form of
// the scalar's "out |= uint32(uint8(v)) << (8*c)" loop, since on a little-endian machine the low byte of a lane is
// exactly uint8 of it. Lane order probed before use.
func packLowBytes(v archsimd.Uint32x4, g lowByteGather) uint32 {
	return v.ReshapeToUint8s().PermuteOrZero(g).ReshapeToUint32s().GetElem(0)
}

// divisorVec is the ScaledDividerU32 factor in whatever shape divideVec wants, splatted once per segment.
type divisorVec = archsimd.Uint32x4

// newDivisorVec splats the factor for divideVec.
func newDivisorVec(factor uint32) divisorVec { return archsimd.BroadcastUint32x4(factor) }

// divideVec is the vector form of threeBoxApproxPass.divide: uint32((uint64(v)*uint64(factor))>>32) per lane. amd64's
// widening 32-bit multiply is VPMULUDQ, which reads the *even* 32-bit lanes of both operands and produces two 64-bit
// products, so lanes 0 and 2 come from a direct multiply and lanes 1 and 3 from one taken after shifting the dividend
// down by 32 bits within each 64-bit half. The factor needs no matching shift: every lane of a splat holds the same
// value, so its even lanes are the right multiplier either way. Recombining takes the two products' high halves — the
// first shifted down into its low half, the second left in its high half — and ORs them, which lands the four results
// in lane order. Checked against the scalar over a randomized sweep of the full 32-bit dividend domain before this was
// built on.
func divideVec(v archsimd.Uint32x4, factor divisorVec) archsimd.Uint32x4 {
	even := v.MulWidenEven(factor)
	odd := v.ReshapeToUint64s().ShiftAllRight(32).ReshapeToUint32s().MulWidenEven(factor)
	return even.ShiftAllRight(32).Or(odd.ShiftAllRight(32).ShiftAllLeft(32)).ReshapeToUint32s()
}
