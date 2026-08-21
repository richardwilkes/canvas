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

// simdKernelsSupported reports whether the CPU can run the simd blur kernels. Everything they compile to on arm64
// (VFMLA, FMUL, VADD, VSUB, VUMULL, VUZP1, VUZP2, VUXTL, VDUP, SCVTF, FCVTZS, VFCMGT, VBIF and the 128-bit
// loads/stores) is baseline NEON, present on every arm64 CPU Go supports.
func simdKernelsSupported() bool { return true }

// Per-kernel dispatch preference: whether the simd kernel is at least as fast as this build's default lane, which for
// both blur passes is the portable scalar body in blurengine.go — neither has a NEON assembly twin. The equivalence
// test gates on simdKernelsSupported and so locks both kernels bit-for-bit regardless of these; init consults them so
// the experiment build never dispatches a measured regression.
//
// Benchstat on an M4 Max (blurengine_bench_test.go, both build modes, n=10, 2026-08-20, every comparison p=0.000):
// the Gaussian segment wins -58% at 3 taps, -63% at 7 and -70% at 13 — the whole reachable window range, from just
// above the no-blur cutoff to just under the three-box handover — and the three-box segment wins -82% at both a small
// and a large window (its per-sample cost does not depend on the window). Segment geomean -73%. The two passes carry
// essentially the entire cost of a blur, so the win lands on the draw: a whole 256x256 RGBA two-axis blur through
// evalBlurPasses is -71% at sigma 1.9 and -83% at sigma 8.
//
// The three-box margin is the wider one because the scalar body pays four separate loads, four adds and a 64-bit
// multiply per channel per sample against three circular buffers, all of which collapse into single 16-byte
// operations; the Gaussian margin grows with the tap count because the per-sample unpack and pack are fixed overhead
// that more taps amortize.
const (
	preferSIMDGaussianSegment = true
	preferSIMDThreeBoxSegment = true
)

// exprMulAdd4 is the lanewise form of the *plain Go expression* "f*m + a" as this build's compiler lowers it. Go
// permits an implementation to contract such an expression into a single fused operation, and the arm64 compiler does:
// both of gaussianBlurSegmentGeneric's float sites — "sum[c] += p.srcBuffer[s][c] * k" and "v := sum[c]*255 + 0.5" —
// disassemble to FMADDS, a *single-precision* fused multiply-add. So the vector twin must fuse too. Spelling it as a
// separate Mul and Add instead diverges from the scalar body on hostile-float inputs — 1274 of the 4096 hostile
// segments TestBlurEngineSIMDContractionNegativeControl drives, which is what makes the equivalence fuzz's corpus a
// real test of this choice rather than an assumption. Spelling it as a widen/double-FMA/narrow sequence would diverge
// the other way, since that is what float32(math.FMA(float64...)) means and an explicit conversion is precisely what
// the compiler may not contract away.
func exprMulAdd4(f, m, a archsimd.Float32x4) archsimd.Float32x4 { return f.MulAdd(m, a) }

// lowByteGather is whatever loop-invariant state packLowBytes needs on this arch. arm64 gathers the bytes with two
// VUZP1s, which need no shuffle table, so the value is empty and costs nothing to thread through the kernels.
type lowByteGather = struct{}

// newLowByteGather builds the (empty) gather state once per segment.
func newLowByteGather() lowByteGather { return lowByteGather{} }

// packLowBytes gathers the low byte of each of four 32-bit lanes into one 8888 word, in lane order — the vector form of
// the scalar's "out |= uint32(uint8(v)) << (8*c)" loop, since on a little-endian machine the low byte of a lane is
// exactly uint8 of it. VUZP1 takes the even-indexed elements of the concatenation of its two operands, so one pass over
// the 16-bit view keeps each word's low half and a second pass over the byte view keeps that half's low byte; passing
// the vector for both operands puts the four wanted bytes in each half of the result and only the low word is read.
// Lane order probed before use.
func packLowBytes(v archsimd.Uint32x4, _ lowByteGather) uint32 {
	lo16 := v.ReshapeToUint16s()
	lo8 := lo16.ConcatEven(lo16).ReshapeToUint8s()
	return lo8.ConcatEven(lo8).ReshapeToUint32s().GetElem(0)
}

// divisorVec is the ScaledDividerU32 factor in whatever shape divideVec wants, splatted once per segment.
type divisorVec = archsimd.Uint32x4

// newDivisorVec splats the factor for divideVec.
func newDivisorVec(factor uint32) divisorVec { return archsimd.BroadcastUint32x4(factor) }

// divideVec is the vector form of threeBoxApproxPass.divide: uint32((uint64(v)*uint64(factor))>>32) per lane. arm64 has
// no 32-bit high-half multiply, so the products are taken widened — VUMULL over each half of the register, exact in 64
// bits — and VUZP2 gathers the odd 32-bit halves of the two product vectors, which little-endian are precisely the >>32
// halves. VUMULL reads only the low two lanes of each operand, which is why the second product passes the factor
// unshifted: every lane of a splat holds the same value. Checked against the scalar over a randomized sweep of the full
// 32-bit dividend domain before this was built on.
func divideVec(v archsimd.Uint32x4, factor divisorVec) archsimd.Uint32x4 {
	lo := v.MulWidenLo(factor).ReshapeToUint32s()
	hi := v.HiToLo().MulWidenLo(factor).ReshapeToUint32s()
	return lo.ConcatOdd(hi)
}
