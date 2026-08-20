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
//   - Add/Mul are the same IEEE single ops the scalar stages perform, applied in the same operand order.
//   - Float32x4.ConvertToInt32 is the same truncate-toward-zero conversion the compiler emits for Go's int32(float32)
//     (FCVTZS on arm64, VCVTTPS2DQ on amd64), including each arch's implementation-defined out-of-range result, so the
//     evenly-spaced gradient's stop index matches the scalar one lane for lane.
//
// Like the NEON kernels, these process all stride lanes unconditionally; lanes at or beyond z.n are scratch no
// consumer reads. The substitution changes throughput only, never rendered bytes — locked by
// TestStageSIMDMatchesScalar.

// init swaps the dispatch variables to the simd kernels where they are the fastest lane. simdKernelsSupported is the
// hardware floor (arm64: baseline NEON; amd64: AVX2+FMA — unqualified CPUs keep the default dispatch entirely), and
// the per-arch preferSIMD* constants then decline any kernel the build's default lane beats: on arm64 the NEON
// assembly keeps three (see stage_simd_arm64.go), on amd64 the scalar default loses everywhere so all 8 wire in. The
// equivalence tests gate on simdKernelsSupported alone, so declined kernels stay locked bit-for-bit.
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
// color stage

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
