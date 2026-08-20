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

import (
	"math"
	"simd/archsimd"
	"unsafe"

	"github.com/richardwilkes/canvas/geom"
)

// The goexperiment.simd kernels. They are written against archsimd's 128-bit types (the widest shape arm64 and amd64
// share) plus the per-arch madf4/madf4v helpers (stage_simd_arm64.go / stage_simd_amd64.go), because bit-exactness with
// the scalar stages pins every operation's semantics:
//
//   - madf is float32(math.FMA(float64...)) — a double FMA rounded to single. madf4 reproduces that two-rounding
//     sequence exactly; a single-precision vector FMA would diverge by ~1 ULP on rare inputs (see the stage_arm64.s
//     header, and the anchors in TestStageSIMDMatchesScalar's history).
//   - minf/maxf are comparison-based ("comparison false -> second operand"); vector Min/Max propagate NaN (FMIN), so
//     clamps must be built from compare+select instead.
//   - Add/Sub/Mul are the same IEEE single ops the scalar stages perform, applied in the same operand order.
//   - Float32x4.ConvertToInt32 is the same truncate-toward-zero conversion the compiler emits for Go's int32(float32)
//     (FCVTZS on arm64, VCVTTPS2DQ on amd64), including each arch's implementation-defined out-of-range result, so the
//     evenly-spaced gradient's stop index matches the scalar one lane for lane.
//   - Div is the same IEEE single-precision divide as the scalar "/" (FDIV on arm64, VDIVPS on amd64), verified
//     bit-for-bit against it over the hostile float mix before unpremul was built on it.
//   - ToBits/And/BitsToFloat32 only reinterpret and mask bits, so the lane-mask stage's AND is exact by construction.
//
// Like the NEON kernels, these process all stride lanes unconditionally; lanes at or beyond z.n — in the register file
// and in the sampler scratch the image kernels write — are scratch no consumer reads. The substitution changes
// throughput only, never rendered bytes — locked by TestStageSIMDMatchesScalar.

// init swaps the dispatch variables to the simd kernels where they are the fastest lane. simdKernelsSupported is the
// hardware floor (arm64: baseline NEON; amd64: AVX2+FMA — unqualified CPUs keep the default dispatch entirely), and
// the per-arch preferSIMD* constants then decline any kernel the build's default lane beats: on arm64 the NEON
// assembly and the compiler's own array copies keep a few (see stage_simd_arm64.go). The equivalence tests gate on
// simdKernelsSupported alone, so declined kernels stay locked bit-for-bit.
func init() {
	if !simdKernelsSupported() {
		return
	}
	if preferSIMDSeed {
		seedStageFn = seedStageSIMD
	}
	if preferSIMDClampX1 {
		clampX1StageFn = clampX1StageSIMD
	}
	if preferSIMDMatrixTranslate {
		matrixTranslateStageFn = matrixTranslateStageSIMD
	}
	if preferSIMDMatrixScaleTranslate {
		matrixScaleTranslateStageFn = matrixScaleTranslateStageSIMD
	}
	if preferSIMDMatrixAffine {
		matrixAffineStageFn = matrixAffineStageSIMD
	}
	if preferSIMDGradient2Stop {
		gradient2StopStageFn = gradient2StopStageSIMD
	}
	if preferSIMDGradientEvenly {
		gradientEvenlyStageFn = gradientEvenlyStageSIMD
	}
	if preferSIMDMatrix4x5 {
		matrix4x5StageFn = matrix4x5StageSIMD
	}
	if preferSIMDMaskApply {
		maskApplyStageFn = maskApplyStageSIMD
	}
	if preferSIMDClamp01 {
		clamp01StageFn = clamp01StageSIMD
	}
	if preferSIMDClampGamut {
		clampGamutStageFn = clampGamutStageSIMD
	}
	if preferSIMDPremul {
		premulStageFn = premulStageSIMD
	}
	if preferSIMDUnpremul {
		unpremulStageFn = unpremulStageSIMD
	}
	if preferSIMDScale1Float {
		scale1FloatStageFn = scale1FloatStageSIMD
	}
	if preferSIMDSetRGB {
		setRGBStageFn = setRGBStageSIMD
	}
	if preferSIMDMoveSrcDst {
		moveSrcDstStageFn = moveSrcDstStageSIMD
	}
	if preferSIMDMoveDstSrc {
		moveDstSrcStageFn = moveDstSrcStageSIMD
	}
	if preferSIMDBilinear {
		bilinearNXStageFn = bilinearNXStageSIMD
		bilinearPXStageFn = bilinearPXStageSIMD
		bilinearNYStageFn = bilinearNYStageSIMD
		bilinearPYStageFn = bilinearPYStageSIMD
	}
	if preferSIMDAccumulate {
		accumulateStageFn = accumulateStageSIMD
	}
}

///////////////////////////////////////////////////////////////////////////////
// shared vector helpers

// maxf4 is the lanewise maxf: "a > b ? a : b". IfElse keeps x where the mask is true and takes y where it is false, so
// this is the comparison-based selection the scalar performs — a NaN lane compares false and yields b, and a -0/+0 pair
// yields b as well (neither compares greater). A vector Max would instead be FMAX/VMAXPS, which propagate (or, on
// amd64, prefer the second operand for) NaN and ignore zero signs, so it is not usable here.
func maxf4(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.IfElse(a.Greater(b), b) }

// minf4 is the lanewise minf: "a < b ? a : b", built from the same compare+select for the same reason as maxf4.
func minf4(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.IfElse(a.Less(b), b) }

// clamp014 is the lanewise clamp01: minf(maxf(v, 0), 1), in that nesting order — a NaN lane becomes +0 at the maxf and
// then survives the minf, exactly as the scalar clamp01 does.
func clamp014(v, zero, one archsimd.Float32x4) archsimd.Float32x4 {
	return minf4(maxf4(v, zero), one)
}

// transpose4 transposes the 4x4 matrix whose rows are r0..r3, returning its columns. It is the archsimd spelling of the
// ZIP1/ZIP2 pyramid gradEvenly16 uses to turn four gathered 32-byte stop records into per-channel quads: the 32-bit
// interleaves pair rows 0/1 and 2/3, then the 64-bit interleaves stitch those halves into columns. Only lane shuffling
// is involved, so no value is inspected or rounded; the 32-bit and 64-bit interleave primitives produce the same lane
// order on arm64 (ZIP1/ZIP2) and amd64 (PUNPCKLDQ/PUNPCKHDQ, PUNPCKLQDQ/PUNPCKHQDQ), verified by probe.
func transpose4(r0, r1, r2, r3 archsimd.Float32x4) (c0, c1, c2, c3 archsimd.Float32x4) {
	b0, b1, b2, b3 := r0.ToBits(), r1.ToBits(), r2.ToBits(), r3.ToBits()
	lo01 := b0.InterleaveLo(b1).ReshapeToUint64s()
	hi01 := b0.InterleaveHi(b1).ReshapeToUint64s()
	lo23 := b2.InterleaveLo(b3).ReshapeToUint64s()
	hi23 := b2.InterleaveHi(b3).ReshapeToUint64s()
	return lo01.InterleaveLo(lo23).ReshapeToUint32s().BitsToFloat32(),
		lo01.InterleaveHi(lo23).ReshapeToUint32s().BitsToFloat32(),
		hi01.InterleaveLo(hi23).ReshapeToUint32s().BitsToFloat32(),
		hi01.InterleaveHi(hi23).ReshapeToUint32s().BitsToFloat32()
}

// A gradStop is eight float32s with no padding, so it is exactly the two 16-byte quads {fr,fg,fb,fa} and {br,bg,bb,ba}
// the gather loads — the same layout the NEON kernel's paired VLD1 assumes. These zero-length arrays fail to compile if
// that ever stops holding.
var (
	_ [32 - unsafe.Sizeof(gradStop{})]byte
	_ [unsafe.Sizeof(gradStop{}) - 32]byte
)

// gradStopQuads reads one stop record as its factor and bias quads.
func gradStopQuads(s *gradStop) (factor, bias archsimd.Float32x4) {
	q := (*[2][4]float32)(unsafe.Pointer(s))
	return archsimd.LoadFloat32x4Array(&q[0]), archsimd.LoadFloat32x4Array(&q[1])
}

///////////////////////////////////////////////////////////////////////////////
// coordinate stages

// seedStageSIMD is the vector seed_shader kernel. The scalar stage computes fx + iota05[i] with a single-precision add;
// the vector form adds the same two operands in the same order (broadcast fx on the left, the iota quad on the right),
// and the g/b/a lanes are pure broadcasts of fy, 1 and 0.
func seedStageSIMD(z *lanes) {
	fx := archsimd.BroadcastFloat32x4(float32(uint32(z.dx)))
	fy := archsimd.BroadcastFloat32x4(float32(uint32(z.dy)) + 0.5)
	one := archsimd.BroadcastFloat32x4(1)
	zero := archsimd.BroadcastFloat32x4(0)
	for o := 0; o < stride; o += 4 {
		fx.Add(archsimd.LoadFloat32x4(iota05[o:])).Store(z.r[o:])
		fy.Store(z.g[o:])
		one.Store(z.b[o:])
		zero.Store(z.a[o:])
	}
}

// matrixTranslateStageSIMD is the vector matrix_translate kernel: the same single-precision adds the scalar stage
// performs, with the lane on the left and the translate on the right (matching "z.r[i] += tx").
func matrixTranslateStageSIMD(z *lanes) {
	c := z.ctx.(*matrixCtx)
	tx := archsimd.BroadcastFloat32x4(c.m[geom.MTransX])
	ty := archsimd.BroadcastFloat32x4(c.m[geom.MTransY])
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(z.r[o:]).Add(tx).Store(z.r[o:])
		archsimd.LoadFloat32x4(z.g[o:]).Add(ty).Store(z.g[o:])
	}
}

// matrixScaleTranslateStageSIMD is the vector matrix_scale_translate kernel: madf(lane, scale, translate) per axis. The
// two scale coefficients are pre-widened once (broadcastCoef) and the translates broadcast as singles, so each lane
// runs the identical widen / fused double FMA / round-to-single sequence the scalar madf performs.
func matrixScaleTranslateStageSIMD(z *lanes) {
	c := z.ctx.(*matrixCtx)
	sx, sy := broadcastCoef(c.m[geom.MScaleX]), broadcastCoef(c.m[geom.MScaleY])
	tx := archsimd.BroadcastFloat32x4(c.m[geom.MTransX])
	ty := archsimd.BroadcastFloat32x4(c.m[geom.MTransY])
	for o := 0; o < stride; o += 4 {
		madf4(archsimd.LoadFloat32x4(z.r[o:]), sx, tx).Store(z.r[o:])
		madf4(archsimd.LoadFloat32x4(z.g[o:]), sy, ty).Store(z.g[o:])
	}
}

// matrixAffineStageSIMD is the vector matrix_2x3 kernel: r' = madf(r, s0, madf(g, s1, s2)) and
// g' = madf(r, s3, madf(g, s4, s5)). madf4 returns a Float32x4, so the inner madf's result is rounded to single before
// the outer one re-widens it — exactly the scalar nesting, where each madf's float32 result feeds the next madf's
// float64() conversion. Both outputs are computed from the loaded r before either is stored, matching the scalar
// stage's "compute both, then assign" ordering.
func matrixAffineStageSIMD(z *lanes) {
	s := z.ctx.(*matrixCtx).m
	c0, c1 := broadcastCoef(s[0]), broadcastCoef(s[1])
	c3, c4 := broadcastCoef(s[3]), broadcastCoef(s[4])
	s2 := archsimd.BroadcastFloat32x4(s[2])
	s5 := archsimd.BroadcastFloat32x4(s[5])
	for o := 0; o < stride; o += 4 {
		r := archsimd.LoadFloat32x4(z.r[o:])
		g := archsimd.LoadFloat32x4(z.g[o:])
		nr := madf4(r, c0, madf4(g, c1, s2))
		ng := madf4(r, c3, madf4(g, c4, s5))
		nr.Store(z.r[o:])
		ng.Store(z.g[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// tile stage

// clampX1StageSIMD is the vector clamp_x_1 kernel. clamp014 reproduces clamp01's comparison-based minf/maxf pair, so a
// NaN lane clamps to +0 and the signed zeros land where the scalar puts them; a vector Min/Max pair would not.
func clampX1StageSIMD(z *lanes) {
	zero := archsimd.BroadcastFloat32x4(0)
	one := archsimd.BroadcastFloat32x4(1)
	for o := 0; o < stride; o += 4 {
		clamp014(archsimd.LoadFloat32x4(z.r[o:]), zero, one).Store(z.r[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// gradient fill stages

// gradient2StopStageSIMD is the vector two-stop gradient fill: {r,g,b,a} = madf(t, factor[ch], cl[ch]) with t the
// incoming r register. The four factors are pre-widened once and the four left-colors broadcast as singles; t is loaded
// before the r register is overwritten, as the scalar stage's local t is.
func gradient2StopStageSIMD(z *lanes) {
	c := z.ctx.(*gradientCtx)
	fr, fg := broadcastCoef(c.factor.R), broadcastCoef(c.factor.G)
	fb, fa := broadcastCoef(c.factor.B), broadcastCoef(c.factor.A)
	clR, clG := archsimd.BroadcastFloat32x4(c.cl.R), archsimd.BroadcastFloat32x4(c.cl.G)
	clB, clA := archsimd.BroadcastFloat32x4(c.cl.B), archsimd.BroadcastFloat32x4(c.cl.A)
	for o := 0; o < stride; o += 4 {
		t := archsimd.LoadFloat32x4(z.r[o:])
		madf4(t, fr, clR).Store(z.r[o:])
		madf4(t, fg, clG).Store(z.g[o:])
		madf4(t, fb, clB).Store(z.b[o:])
		madf4(t, fa, clA).Store(z.a[o:])
	}
}

// gradientEvenlyStageSIMD is the vector evenly-spaced gradient fill. Per lane the scalar stage computes
// idx = int32(t * gapCount) and then madf(t, stop[idx].f<ch>, stop[idx].b<ch>) per channel; the vector form computes
// the whole index quad at once — Mul is the same single-precision multiply and ConvertToInt32 is the same
// truncate-toward-zero conversion the compiler emits for int32(float32) on this arch — then clamps it into
// [0, len(stops)-1] with integer min/max.
//
// The clamp is a no-op for the live lanes, whose t a preceding tile stage has already put in [0,1]; it exists solely so
// the scratch tail (which the kernel processes unconditionally, and which may hold any float class) can never gather
// outside the stop array. That mirrors the SMAX/SMIN pair in gradEvenly16 and is why the equivalence subtest compares
// only lanes below z.n.
//
// There is no vector gather here (nor on NEON), so the four 32-byte records are loaded individually as factor/bias
// quads and transposed into per-channel quads; transpose4 only moves lanes. The final per-channel madf uses madf4v
// rather than madf4 because the multiplicand is gathered per lane rather than loop-invariant — same widen / fused
// double FMA / round-to-single sequence, just with the multiplicand widened per call.
func gradientEvenlyStageSIMD(z *lanes) {
	c := z.ctx.(*gradientCtx)
	gap := archsimd.BroadcastFloat32x4(c.gapCount)
	lowest := archsimd.BroadcastInt32x4(0)
	highest := archsimd.BroadcastInt32x4(int32(len(c.stops) - 1))
	var idx [4]int32
	for o := 0; o < stride; o += 4 {
		t := archsimd.LoadFloat32x4(z.r[o:])
		t.Mul(gap).ConvertToInt32().Max(lowest).Min(highest).StoreArray(&idx)
		f0, b0 := gradStopQuads(&c.stops[idx[0]])
		f1, b1 := gradStopQuads(&c.stops[idx[1]])
		f2, b2 := gradStopQuads(&c.stops[idx[2]])
		f3, b3 := gradStopQuads(&c.stops[idx[3]])
		fR, fG, fB, fA := transpose4(f0, f1, f2, f3)
		bR, bG, bB, bA := transpose4(b0, b1, b2, b3)
		madf4v(t, fR, bR).Store(z.r[o:])
		madf4v(t, fG, bG).Store(z.g[o:])
		madf4v(t, fB, bB).Store(z.b[o:])
		madf4v(t, fA, bA).Store(z.a[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// image sampler stages

// The four bilinear coordinate kernels are the vector bilinear_nx/px/ny/py stages: one axis coordinate offset by ±0.5
// and one weight lane, both plain single-precision adds/subtracts of the same operands in the same order the scalar
// stages use. They read the saved coordinate and fractional-offset scratch (written by bilinear_setup) and write the
// axis register plus the matching weight lane; like every kernel here they cover all stride lanes, and the samplerCtx
// tail beyond z.n is scratch whose only consumer — the accumulate stage — writes it into the dst registers' own
// scratch tail.

func bilinearNXStageSIMD(z *lanes) { // bilinear_nx: kScale = -1
	c := z.ctx.(*samplerCtx)
	half := archsimd.BroadcastFloat32x4(0.5)
	one := archsimd.BroadcastFloat32x4(1)
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(c.x[o:]).Sub(half).Store(z.r[o:])
		one.Sub(archsimd.LoadFloat32x4(c.fx[o:])).Store(c.scalex[o:])
	}
}

func bilinearPXStageSIMD(z *lanes) { // bilinear_px: kScale = +1
	c := z.ctx.(*samplerCtx)
	half := archsimd.BroadcastFloat32x4(0.5)
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(c.x[o:]).Add(half).Store(z.r[o:])
		archsimd.LoadFloat32x4(c.fx[o:]).Store(c.scalex[o:])
	}
}

func bilinearNYStageSIMD(z *lanes) { // bilinear_ny: kScale = -1
	c := z.ctx.(*samplerCtx)
	half := archsimd.BroadcastFloat32x4(0.5)
	one := archsimd.BroadcastFloat32x4(1)
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(c.y[o:]).Sub(half).Store(z.g[o:])
		one.Sub(archsimd.LoadFloat32x4(c.fy[o:])).Store(c.scaley[o:])
	}
}

func bilinearPYStageSIMD(z *lanes) { // bilinear_py: kScale = +1
	c := z.ctx.(*samplerCtx)
	half := archsimd.BroadcastFloat32x4(0.5)
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(c.y[o:]).Add(half).Store(z.g[o:])
		archsimd.LoadFloat32x4(c.fy[o:]).Store(c.scaley[o:])
	}
}

// accumulateStageSIMD is the vector accumulate kernel — the bilinear/bicubic inner loop, run once per tap (4 taps per
// pixel bilinear, 16 bicubic). The per-lane area weight is the same single-precision multiply the scalar performs, and
// each channel is then madf(scale, src, dst) with both multiplicands varying per lane, so madf4v supplies the identical
// widen / one fused double FMA / round-to-single sequence.
func accumulateStageSIMD(z *lanes) {
	c := z.ctx.(*samplerCtx)
	for o := 0; o < stride; o += 4 {
		scale := archsimd.LoadFloat32x4(c.scalex[o:]).Mul(archsimd.LoadFloat32x4(c.scaley[o:]))
		madf4v(scale, archsimd.LoadFloat32x4(z.r[o:]), archsimd.LoadFloat32x4(z.dr[o:])).Store(z.dr[o:])
		madf4v(scale, archsimd.LoadFloat32x4(z.g[o:]), archsimd.LoadFloat32x4(z.dg[o:])).Store(z.dg[o:])
		madf4v(scale, archsimd.LoadFloat32x4(z.b[o:]), archsimd.LoadFloat32x4(z.db[o:])).Store(z.db[o:])
		madf4v(scale, archsimd.LoadFloat32x4(z.a[o:]), archsimd.LoadFloat32x4(z.da[o:])).Store(z.da[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// mask stage

// maskApplyStageSIMD is the vector apply_vector_mask / check_decal_mask kernel: each channel's bits ANDed with the
// lane's 32-bit mask. ToBits and BitsToFloat32 only reinterpret the register, so the result is the scalar's
// Float32frombits(Float32bits(v) & mask) by construction, for every float class including NaN and the signed zeros.
func maskApplyStageSIMD(z *lanes) {
	mask := z.ctx.(*laneMask)
	for o := 0; o < stride; o += 4 {
		m := archsimd.LoadUint32x4(mask[o:])
		archsimd.LoadFloat32x4(z.r[o:]).ToBits().And(m).BitsToFloat32().Store(z.r[o:])
		archsimd.LoadFloat32x4(z.g[o:]).ToBits().And(m).BitsToFloat32().Store(z.g[o:])
		archsimd.LoadFloat32x4(z.b[o:]).ToBits().And(m).BitsToFloat32().Store(z.b[o:])
		archsimd.LoadFloat32x4(z.a[o:]).ToBits().And(m).BitsToFloat32().Store(z.a[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// color stages

// clamp01StageSIMD is the vector clamp_01 kernel: clamp014 on all four channels, which is clamp01's comparison-based
// minf(maxf(v, 0), 1) nesting — a NaN lane becomes +0 at the maxf and survives the minf, as the scalar does.
func clamp01StageSIMD(z *lanes) {
	zero := archsimd.BroadcastFloat32x4(0)
	one := archsimd.BroadcastFloat32x4(1)
	for o := 0; o < stride; o += 4 {
		clamp014(archsimd.LoadFloat32x4(z.r[o:]), zero, one).Store(z.r[o:])
		clamp014(archsimd.LoadFloat32x4(z.g[o:]), zero, one).Store(z.g[o:])
		clamp014(archsimd.LoadFloat32x4(z.b[o:]), zero, one).Store(z.b[o:])
		clamp014(archsimd.LoadFloat32x4(z.a[o:]), zero, one).Store(z.a[o:])
	}
}

// clampGamutStageSIMD is the vector clamp_gamut kernel: alpha clamped to [0,1] and stored, then rgb clamped to [0,a]
// against that already-clamped alpha — the same order and the same comparison-based minf/maxf pair as the scalar, so a
// NaN alpha clamps to +0 and then pins rgb to +0 too.
func clampGamutStageSIMD(z *lanes) {
	zero := archsimd.BroadcastFloat32x4(0)
	one := archsimd.BroadcastFloat32x4(1)
	for o := 0; o < stride; o += 4 {
		a := clamp014(archsimd.LoadFloat32x4(z.a[o:]), zero, one)
		a.Store(z.a[o:])
		minf4(maxf4(archsimd.LoadFloat32x4(z.r[o:]), zero), a).Store(z.r[o:])
		minf4(maxf4(archsimd.LoadFloat32x4(z.g[o:]), zero), a).Store(z.g[o:])
		minf4(maxf4(archsimd.LoadFloat32x4(z.b[o:]), zero), a).Store(z.b[o:])
	}
}

// premulStageSIMD is the vector premul kernel: rgb multiplied by the alpha register, the same single-precision multiply
// with the channel on the left and alpha on the right (matching "z.r[i] *= z.a[i]").
func premulStageSIMD(z *lanes) {
	for o := 0; o < stride; o += 4 {
		a := archsimd.LoadFloat32x4(z.a[o:])
		archsimd.LoadFloat32x4(z.r[o:]).Mul(a).Store(z.r[o:])
		archsimd.LoadFloat32x4(z.g[o:]).Mul(a).Store(z.g[o:])
		archsimd.LoadFloat32x4(z.b[o:]).Mul(a).Store(z.b[o:])
	}
}

// unpremulStageSIMD is the vector unpremul kernel: scale = 1/a where that reciprocal is finite, else +0. The scalar's
// guard is "s < inf", which is false for both NaN and +Inf and true for -Inf, so it is spelled here as the same
// comparison feeding a select (IfElse keeps s where the comparison holds and takes +0 where it does not) rather than
// as any NaN-aware min/max. a = +0 therefore yields scale +0 (the reciprocal overflows to +Inf and is rejected), a = -0
// yields -Inf, and a = NaN yields +0 — matching the scalar lane for lane, including the NaN the +0/±Inf multiply then
// produces for an infinite channel.
func unpremulStageSIMD(z *lanes) {
	one := archsimd.BroadcastFloat32x4(1)
	zero := archsimd.BroadcastFloat32x4(0)
	inf := archsimd.BroadcastFloat32x4(math.Float32frombits(0x7f800000))
	for o := 0; o < stride; o += 4 {
		s := one.Div(archsimd.LoadFloat32x4(z.a[o:]))
		scale := s.IfElse(s.Less(inf), zero)
		archsimd.LoadFloat32x4(z.r[o:]).Mul(scale).Store(z.r[o:])
		archsimd.LoadFloat32x4(z.g[o:]).Mul(scale).Store(z.g[o:])
		archsimd.LoadFloat32x4(z.b[o:]).Mul(scale).Store(z.b[o:])
	}
}

// scale1FloatStageSIMD is the vector scale_1_float kernel: all four channels multiplied by the broadcast constant, with
// the channel on the left as the scalar's "z.r[i] *= c" has it.
func scale1FloatStageSIMD(z *lanes) {
	c := archsimd.BroadcastFloat32x4(*z.ctx.(*float32))
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(z.r[o:]).Mul(c).Store(z.r[o:])
		archsimd.LoadFloat32x4(z.g[o:]).Mul(c).Store(z.g[o:])
		archsimd.LoadFloat32x4(z.b[o:]).Mul(c).Store(z.b[o:])
		archsimd.LoadFloat32x4(z.a[o:]).Mul(c).Store(z.a[o:])
	}
}

// setRGBStageSIMD is the vector set_rgb kernel: three broadcast stores, no arithmetic at all.
func setRGBStageSIMD(z *lanes) {
	g := z.ctx.(*gatherCtx)
	r := archsimd.BroadcastFloat32x4(g.setRGB[0])
	gg := archsimd.BroadcastFloat32x4(g.setRGB[1])
	b := archsimd.BroadcastFloat32x4(g.setRGB[2])
	for o := 0; o < stride; o += 4 {
		r.Store(z.r[o:])
		gg.Store(z.g[o:])
		b.Store(z.b[o:])
	}
}

// moveSrcDstStageSIMD is the vector move_src_dst kernel and moveDstSrcStageSIMD the vector move_dst_src kernel: pure
// register copies, moving whole quads instead of the compiler's array copy (move_src_dst) or per-lane loads and stores
// (move_dst_src). No value is inspected, so every float class copies verbatim.

func moveSrcDstStageSIMD(z *lanes) {
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(z.r[o:]).Store(z.dr[o:])
		archsimd.LoadFloat32x4(z.g[o:]).Store(z.dg[o:])
		archsimd.LoadFloat32x4(z.b[o:]).Store(z.db[o:])
		archsimd.LoadFloat32x4(z.a[o:]).Store(z.da[o:])
	}
}

func moveDstSrcStageSIMD(z *lanes) {
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(z.dr[o:]).Store(z.r[o:])
		archsimd.LoadFloat32x4(z.dg[o:]).Store(z.g[o:])
		archsimd.LoadFloat32x4(z.db[o:]).Store(z.b[o:])
		archsimd.LoadFloat32x4(z.da[o:]).Store(z.a[o:])
	}
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
