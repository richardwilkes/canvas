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

// The goexperiment.simd kernels for the image-filter runtime effects (filterkernels.go), written against the same
// 128-bit archsimd types and the same bit-exactness rules as stage_simd.go — read that file's header first; everything
// it says about madf4/madf4v, the comparison-based minf4/maxf4/clamp014, Div, and the scratch tail applies here
// unchanged. Two things are specific to this file:
//
//   - Many of these stages compute through *plain Go expressions* rather than through madf — the convolution
//     accumulator's "sum += c*k", the convolution finalize's "sum*gain + bias", the arithmetic chain
//     "k0*s*d + k1*s + k2*d + k3", displacement's "coords + scale*(sel - 0.5)", the magnifier's zoom — and Go permits
//     the compiler to contract such an expression into a fused multiply-add. arm64 does so and amd64 (at the module's
//     GOAMD64=v1 baseline) does not, so those sites go through the per-arch exprMulAdd4, which reproduces the arch's
//     choice. The magnifier carries the one *subtractive* shape, "2 - edgeInset" with edgeInset a product, which arm64
//     contracts into FMSUBS: exprMulSub4 is its per-arch twin. Every explicit madf site still goes through
//     madf4/madf4v as everywhere else.
//   - Several of these stages carry a loop-invariant branch (convolve-alpha, origin tap, displacement's channel
//     selects). The kernels hoist it out of the quad loop rather than re-testing it per lane, which is legal precisely
//     because it is loop-invariant: it is a property of the compiled stage instance, not of the data. The magnifier's
//     branch is the exception — it pivots on a per-lane value, so that kernel evaluates both arms and selects.
//
// Every one of these kernels processes all stride lanes. Their ctx scratch tolerates that: the coordinate buffers
// (storeCoordsStage), the tap alphas (saveAlphaStage) and the arithmetic dst hold (arithStoreRes0Stage) are all
// whole-array assignments, so their tails are defined; the morphology aggregate/plus buffers and the convolution
// sum/origin-alpha buffers are written by these same kernels across all lanes, and their only consumers are the
// finalize/return kernels here, which route the tail into the register file's own tail — scratch no consumer reads.

///////////////////////////////////////////////////////////////////////////////
// morphology kernels

// morphFlipInto stores flip * (r,g,b,a) into a four-channel scratch buffer — the "aggregate = flip * child.eval(...)"
// seed that the linear form performs into agg (morph_agg_init) and into plus (morph_take_plus), and the sparse form
// into agg. flip is on the left of the multiply, as the scalar stages have it.
func morphFlipInto(dst *[4][stride]float32, flip archsimd.Float32x4, z *lanes) {
	for o := 0; o < stride; o += 4 {
		flip.Mul(archsimd.LoadFloat32x4(z.r[o:])).Store(dst[0][o:])
		flip.Mul(archsimd.LoadFloat32x4(z.g[o:])).Store(dst[1][o:])
		flip.Mul(archsimd.LoadFloat32x4(z.b[o:])).Store(dst[2][o:])
		flip.Mul(archsimd.LoadFloat32x4(z.a[o:])).Store(dst[3][o:])
	}
}

// morphMinusCoords sets the child's sample coordinates to coord - delta, the -delta tap both the linear take-plus and
// the sparse agg-minus stages set up after saving their +delta tap. It runs as its own loop, after the save, because
// the save reads the r/g registers this then overwrites.
func morphMinusCoords(z *lanes, st *morphStep) {
	c := st.c
	dx := archsimd.BroadcastFloat32x4(st.dx)
	dy := archsimd.BroadcastFloat32x4(st.dy)
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(c.coords[0][o:]).Sub(dx).Store(z.r[o:])
		archsimd.LoadFloat32x4(c.coords[1][o:]).Sub(dy).Store(z.g[o:])
	}
}

// morphAggInitStageSIMD is the vector morph_agg_init kernel: agg = flip * child.
func morphAggInitStageSIMD(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	morphFlipInto(&c.agg, archsimd.BroadcastFloat32x4(c.flip), z)
}

// morphPlusCoordsStageSIMD is the vector morph_plus_coords kernel: the same single-precision adds of the saved
// coordinate and the step's delta the scalar stage performs, in the same operand order.
func morphPlusCoordsStageSIMD(z *lanes) {
	st := z.ctx.(*morphStep)
	c := st.c
	dx := archsimd.BroadcastFloat32x4(st.dx)
	dy := archsimd.BroadcastFloat32x4(st.dy)
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(c.coords[0][o:]).Add(dx).Store(z.r[o:])
		archsimd.LoadFloat32x4(c.coords[1][o:]).Add(dy).Store(z.g[o:])
	}
}

// morphTakePlusStageSIMD is the vector morph_take_plus kernel: hold the +delta tap, then set up the -delta tap.
func morphTakePlusStageSIMD(z *lanes) {
	st := z.ctx.(*morphStep)
	morphFlipInto(&st.c.plus, archsimd.BroadcastFloat32x4(st.c.flip), z)
	morphMinusCoords(z, st)
}

// morphMaxAggStageSIMD is the vector morph_max_agg kernel: agg = max(agg, max(plus, flip*minus)). maxf4 is the
// comparison-based maxf ("a > b ? a : b"), so a NaN tap yields the second operand exactly as the scalar does; a vector
// Max would propagate it instead. The nesting and the operand order are the scalar's.
func morphMaxAggStageSIMD(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	flip := archsimd.BroadcastFloat32x4(c.flip)
	minus := [4]*[stride]float32{&z.r, &z.g, &z.b, &z.a}
	for ch := range 4 {
		for o := 0; o < stride; o += 4 {
			m := flip.Mul(archsimd.LoadFloat32x4(minus[ch][o:]))
			agg := archsimd.LoadFloat32x4(c.agg[ch][o:])
			maxf4(agg, maxf4(archsimd.LoadFloat32x4(c.plus[ch][o:]), m)).Store(c.agg[ch][o:])
		}
	}
}

// morphSparseAggMinusStageSIMD is the vector sparse morph_agg_minus kernel: record the +offset tap straight into the
// aggregate (the sparse form holds no separate plus buffer), then set up the -offset tap.
func morphSparseAggMinusStageSIMD(z *lanes) {
	st := z.ctx.(*morphStep)
	morphFlipInto(&st.c.agg, archsimd.BroadcastFloat32x4(st.c.flip), z)
	morphMinusCoords(z, st)
}

// morphSparseMaxAggStageSIMD is the vector sparse morph_max_agg kernel: agg = max(flip*minus, agg) — the reversed
// argument order the sparse kernel uses, which decides which operand a NaN or a signed-zero pair yields, so it is
// preserved here too.
func morphSparseMaxAggStageSIMD(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	flip := archsimd.BroadcastFloat32x4(c.flip)
	minus := [4]*[stride]float32{&z.r, &z.g, &z.b, &z.a}
	for ch := range 4 {
		for o := 0; o < stride; o += 4 {
			m := flip.Mul(archsimd.LoadFloat32x4(minus[ch][o:]))
			maxf4(m, archsimd.LoadFloat32x4(c.agg[ch][o:])).Store(c.agg[ch][o:])
		}
	}
}

// morphReturnStageSIMD is the vector morph_return kernel: the registers become flip * aggregate.
func morphReturnStageSIMD(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	flip := archsimd.BroadcastFloat32x4(c.flip)
	for o := 0; o < stride; o += 4 {
		flip.Mul(archsimd.LoadFloat32x4(c.agg[0][o:])).Store(z.r[o:])
		flip.Mul(archsimd.LoadFloat32x4(c.agg[1][o:])).Store(z.g[o:])
		flip.Mul(archsimd.LoadFloat32x4(c.agg[2][o:])).Store(z.b[o:])
		flip.Mul(archsimd.LoadFloat32x4(c.agg[3][o:])).Store(z.a[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// displacement kernel

// displacementSelect resolves one of the displacement kernel's two channel selects into the register it reads and
// whether that register needs the unpremultiplying divide. Both are stage uniforms — the shader's one-hot channel
// indices — so the whole selection hoists out of the quad loop, leaving the lane work a divide, a subtract and a
// scale. sel == 3 is the alpha channel, which the scalar takes straight from the register: the unpremultiply divides
// only rgb. DisplacementMap rejects any index outside 0..3, and an out-of-range one would panic on the same channel
// table the scalar stage indexes.
func displacementSelect(z *lanes, sel int32) (reg *[stride]float32, unpremul bool) {
	ch := [4]*[stride]float32{&z.r, &z.g, &z.b, &z.a}
	return ch[sel], sel != 3
}

// displacementStageSIMD is the vector displacement kernel: coord + scale * (selected channel - 0.5). maxf4 supplies the
// comparison-based maxf(a, 0.0001) unpremultiply guard and Div is the same IEEE single-precision divide as the scalar
// "/"; the trailing "coords + scale*t" goes through exprMulAdd4 because that is what the scalar stage's plain
// expression compiles to on this arch. Both selected channels are loaded before either coordinate register is stored,
// which matters because the selects can name the very registers this writes.
func displacementStageSIMD(z *lanes) {
	c := z.ctx.(*displacementCtx)
	xReg, xUnpremul := displacementSelect(z, c.xSel)
	yReg, yUnpremul := displacementSelect(z, c.ySel)
	sx := archsimd.BroadcastFloat32x4(c.sx)
	sy := archsimd.BroadcastFloat32x4(c.sy)
	half := archsimd.BroadcastFloat32x4(0.5)
	eps := archsimd.BroadcastFloat32x4(0.0001)
	for o := 0; o < stride; o += 4 {
		inv := maxf4(archsimd.LoadFloat32x4(z.a[o:]), eps)
		xv := archsimd.LoadFloat32x4(xReg[o:])
		if xUnpremul {
			xv = xv.Div(inv)
		}
		yv := archsimd.LoadFloat32x4(yReg[o:])
		if yUnpremul {
			yv = yv.Div(inv)
		}
		nr := exprMulAdd4(sx, xv.Sub(half), archsimd.LoadFloat32x4(c.coords[0][o:]))
		ng := exprMulAdd4(sy, yv.Sub(half), archsimd.LoadFloat32x4(c.coords[1][o:]))
		nr.Store(z.r[o:])
		ng.Store(z.g[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// normal-map kernels

// normalSetCoordsStageSIMD is the vector normal_set_coords kernel, run once per Sobel tap: coord + tap offset clamped
// into the edge bounds. clamp(v, lo, hi) lowers to min(max(v, lo), hi) and both halves are the comparison-based
// minf4/maxf4, so the NaN and signed-zero cases land where the scalar puts them.
func normalSetCoordsStageSIMD(z *lanes) {
	tp := z.ctx.(*normalTapCtx)
	c := tp.c
	ddx := archsimd.BroadcastFloat32x4(tp.ddx)
	ddy := archsimd.BroadcastFloat32x4(tp.ddy)
	left := archsimd.BroadcastFloat32x4(c.l)
	right := archsimd.BroadcastFloat32x4(c.r)
	top := archsimd.BroadcastFloat32x4(c.t)
	bottom := archsimd.BroadcastFloat32x4(c.b)
	for o := 0; o < stride; o += 4 {
		minf4(maxf4(archsimd.LoadFloat32x4(c.coords[0][o:]).Add(ddx), left), right).Store(z.r[o:])
		minf4(maxf4(archsimd.LoadFloat32x4(c.coords[1][o:]).Add(ddy), top), bottom).Store(z.g[o:])
	}
}

// sobelDot4 is the vector $normal_filter Sobel dot: dot3 with the constant-folded kSobel weights (0.25, 0.5, 0.25)
// against three tap-alpha quads, i.e. madf(0.25, b0, madf(0.5, b1, 0.25*b2)). madf's product is commutative and both
// widenings are exact, so passing the per-lane alpha as madf4's f and the pre-widened constant as its m64 is the same
// two-rounding widen / fused double FMA / round-to-single sequence the scalar performs, with the constants hoisted.
func sobelDot4(b0, b1, b2, quarter archsimd.Float32x4, quarterC, halfC madfCoef) archsimd.Float32x4 {
	return madf4(b0, quarterC, madf4(b1, halfC, quarter.Mul(b2)))
}

// normalFilterStageSIMD is the vector $normal_filter kernel over the nine saved tap alphas. Every madf is explicit in
// the scalar source (through dot3), so every one goes through madf4/madf4v; the plain subtracts, multiplies and
// divides are the same IEEE single operations in the same order, and Float32x4.Sqrt is the same correctly rounded
// square root as sqrtf — the scalar's double-precision round trip is innocuous for sqrt, since float64's 53 bits
// exceed the 2*24+2 the double-rounding theorem requires (verified by probe over the hostile float mix).
//
// Both madf sites here happen to be shapes where a plain single-precision FMA would agree with madf anyway: the Sobel
// weights are exact powers of two, and the normalize dot's multiplicands are squares. Probed at 0 divergences in 4M
// samples each, against 2.3e-5 for a general madf — so a negative control on this stage cannot distinguish the two.
// madf4/madf4v is still what belongs here: it is what the scalar source says, and the equality is a property of these
// operands, not of the operation.
func normalFilterStageSIMD(z *lanes) {
	c := z.ctx.(*normalCtx)
	negDepth := archsimd.BroadcastFloat32x4(c.negSurfaceDepth)
	quarter := archsimd.BroadcastFloat32x4(0.25)
	one := archsimd.BroadcastFloat32x4(1)
	quarterC, halfC := broadcastCoef(0.25), broadcastCoef(0.5)
	for o := 0; o < stride; o += 4 {
		// Column-major taps: c0 is x=-1 (alphas 0..2), c1 is x=0 (3..5), c2 is x=+1 (6..8).
		c00 := archsimd.LoadFloat32x4(c.alpha[0][o:])
		c02 := archsimd.LoadFloat32x4(c.alpha[2][o:])
		c10 := archsimd.LoadFloat32x4(c.alpha[3][o:])
		c11 := archsimd.LoadFloat32x4(c.alpha[4][o:])
		c12 := archsimd.LoadFloat32x4(c.alpha[5][o:])
		c20 := archsimd.LoadFloat32x4(c.alpha[6][o:])
		c22 := archsimd.LoadFloat32x4(c.alpha[8][o:])
		nx := sobelDot4(c20, archsimd.LoadFloat32x4(c.alpha[7][o:]), c22, quarter, quarterC, halfC).
			Sub(sobelDot4(c00, archsimd.LoadFloat32x4(c.alpha[1][o:]), c02, quarter, quarterC, halfC))
		ny := sobelDot4(c02, c12, c22, quarter, quarterC, halfC).
			Sub(sobelDot4(c00, c10, c20, quarter, quarterC, halfC))
		vx := negDepth.Mul(nx)
		vy := negDepth.Mul(ny)
		// normalize(v) with v = (vx, vy, 1): v / vec(sqrt(dot(v, v))), and dot3 fuses as madf(vx, vx, madf(vy, vy,
		// 1*1)) — the trailing 1*1 is exactly 1. Both multiplicands vary per lane here, so madf4v widens them per call.
		length := madf4v(vx, vx, madf4v(vy, vy, one)).Sqrt()
		vx.Div(length).Store(z.r[o:])
		vy.Div(length).Store(z.g[o:])
		one.Div(length).Store(z.b[o:])
		c11.Store(z.a[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// magnifier kernel

// magnifierStageSIMD is the vector magnifier kernel: mix(coord, zoomCoord, weight*weight) with the rounded-inset
// weight.
//
// Its branch — the circular corner falloff versus the plain linear inset — pivots on the per-lane edge insets, not on
// a uniform, so unlike the convolution's hoisted branches this one evaluates both arms and selects. That is safe
// because neither arm can trap: the circular arm's square root of a negative sum is a NaN the select then discards.
// Less compares false for a NaN lane, so a NaN inset takes the linear arm exactly as the scalar's "eiX < 2 && eiY < 2"
// does.
//
// The two separately rounded edge insets are computed for the comparison and the linear arm, while the circular arm's
// "2 - edgeInset" goes through exprMulSub4 — which on arm64 re-derives that product *fused*, as the scalar's FMSUBS
// does, and so is not the same value the comparison saw. minf4/maxf4 (through clamp014) supply the comparison-based
// clamps, Sqrt is the same correctly rounded square root as sqrtf, and both explicit madf sites go through madf4v
// since every multiplicand varies per lane.
//
// Negative controls: unfusing either contraction site fails the magnifier subtest on the first few draws. The madf
// sites are the weak ones, as in the Sobel filter — the corner-distance madf's multiplicands are squares, where a
// plain single-precision FMA agrees anyway, and the mix madf is a general one that does differ but only around once
// per two million in-domain lane samples (bracketed by probe: clean at 65536 draws, caught at 131072). madf4v belongs
// at both regardless: it is what the scalar source says.
func magnifierStageSIMD(z *lanes) {
	sh := z.ctx.(*MagnifierShader)
	left := archsimd.BroadcastFloat32x4(sh.lensBounds.Left)
	top := archsimd.BroadcastFloat32x4(sh.lensBounds.Top)
	right := archsimd.BroadcastFloat32x4(sh.lensBounds.Right)
	bottom := archsimd.BroadcastFloat32x4(sh.lensBounds.Bottom)
	tx := archsimd.BroadcastFloat32x4(sh.zoomXform[0])
	ty := archsimd.BroadcastFloat32x4(sh.zoomXform[1])
	sx := archsimd.BroadcastFloat32x4(sh.zoomXform[2])
	sy := archsimd.BroadcastFloat32x4(sh.zoomXform[3])
	invX := archsimd.BroadcastFloat32x4(sh.invInset[0])
	invY := archsimd.BroadcastFloat32x4(sh.invInset[1])
	two := archsimd.BroadcastFloat32x4(2)
	zero := archsimd.BroadcastFloat32x4(0)
	one := archsimd.BroadcastFloat32x4(1)
	for o := 0; o < stride; o += 4 {
		x := archsimd.LoadFloat32x4(z.r[o:])
		y := archsimd.LoadFloat32x4(z.g[o:])
		zoomX := exprMulAdd4(sx, x, tx)
		zoomY := exprMulAdd4(sy, y, ty)
		// edgeInset = min(coord - lensBounds.xy, lensBounds.zw - coord) * invInset
		nearX := minf4(x.Sub(left), right.Sub(x))
		nearY := minf4(y.Sub(top), bottom.Sub(y))
		eiX := nearX.Mul(invX)
		eiY := nearY.Mul(invY)
		// Circular distortion weighted by distance to the inset corner: 2 - length(2 - edgeInset), with length(v2) =
		// sqrt(mad(x,x, y*y)).
		dx := exprMulSub4(nearX, invX, two)
		dy := exprMulSub4(nearY, invY, two)
		circular := two.Sub(madf4v(dx, dx, dy.Mul(dy)).Sqrt())
		weight := circular.IfElse(eiX.Less(two).And(eiY.Less(two)), minf4(eiX, eiY))
		weight = clamp014(weight, zero, one)
		w2 := weight.Mul(weight)
		// mix(coord, zoomCoord, w2) lowers to lerp: mad(to - from, t, from).
		nr := madf4v(zoomX.Sub(x), w2, x)
		ng := madf4v(zoomY.Sub(y), w2, y)
		nr.Store(z.r[o:])
		ng.Store(z.g[o:])
	}
}

///////////////////////////////////////////////////////////////////////////////
// matrix-convolution kernels

// matrixConvCoordsStageSIMD is the vector matrix_conv_coords kernel: (coord + kernelPos) - offset, add before subtract
// as the scalar has it — folding the two constants together first would round differently.
func matrixConvCoordsStageSIMD(z *lanes) {
	t := z.ctx.(*matrixConvTap)
	c := t.c
	fkx := archsimd.BroadcastFloat32x4(t.fkx)
	fky := archsimd.BroadcastFloat32x4(t.fky)
	ox := archsimd.BroadcastFloat32x4(c.ox)
	oy := archsimd.BroadcastFloat32x4(c.oy)
	for o := 0; o < stride; o += 4 {
		archsimd.LoadFloat32x4(c.coords[0][o:]).Add(fkx).Sub(ox).Store(z.r[o:])
		archsimd.LoadFloat32x4(c.coords[1][o:]).Add(fky).Sub(oy).Store(z.g[o:])
	}
}

// matrixConvAccumQuad folds one quad of a tap's channels (already unpremultiplied where the kernel requires it) into
// the accumulator: sum[ch] += ch*k, through exprMulAdd4 because that is what the scalar stage's "+=" compiles to on
// this arch.
func matrixConvAccumQuad(c *matrixConvCtx, o int, k, r, g, b, a archsimd.Float32x4) {
	exprMulAdd4(r, k, archsimd.LoadFloat32x4(c.sum[0][o:])).Store(c.sum[0][o:])
	exprMulAdd4(g, k, archsimd.LoadFloat32x4(c.sum[1][o:])).Store(c.sum[1][o:])
	exprMulAdd4(b, k, archsimd.LoadFloat32x4(c.sum[2][o:])).Store(c.sum[2][o:])
	exprMulAdd4(a, k, archsimd.LoadFloat32x4(c.sum[3][o:])).Store(c.sum[3][o:])
}

// matrixConvAccumStageSIMD is the vector matrix_conv_accum kernel — the convolution's inner loop, run once per kernel
// tap (25 times per chunk for a 5x5 kernel), which makes it the file's hottest stage.
//
// Both of the scalar loop's conditions are loop-invariant — convolveAlpha is a shader uniform and isOrigin is a
// property of this tap — so they are hoisted: the convolve-alpha form skips the unpremultiply entirely, and the other
// form tests the (register-resident) origin flag rather than re-deriving it. maxf4 supplies the comparison-based
// maxf(a, 0.0001) guard, and Div is the same IEEE single-precision divide as the scalar "/".
func matrixConvAccumStageSIMD(z *lanes) {
	t := z.ctx.(*matrixConvTap)
	c := t.c
	k := archsimd.BroadcastFloat32x4(t.k)
	if c.convolveAlpha {
		for o := 0; o < stride; o += 4 {
			matrixConvAccumQuad(c, o, k, archsimd.LoadFloat32x4(z.r[o:]), archsimd.LoadFloat32x4(z.g[o:]),
				archsimd.LoadFloat32x4(z.b[o:]), archsimd.LoadFloat32x4(z.a[o:]))
		}
		return
	}
	eps := archsimd.BroadcastFloat32x4(0.0001)
	isOrigin := t.isOrigin
	for o := 0; o < stride; o += 4 {
		a := archsimd.LoadFloat32x4(z.a[o:])
		if isOrigin {
			a.Store(c.origAlpha[o:])
		}
		inv := maxf4(a, eps)
		matrixConvAccumQuad(c, o, k, archsimd.LoadFloat32x4(z.r[o:]).Div(inv),
			archsimd.LoadFloat32x4(z.g[o:]).Div(inv), archsimd.LoadFloat32x4(z.b[o:]).Div(inv), a)
	}
}

// matrixConvFinalizeStageSIMD is the vector matrix_conv_finalize kernel: gain/bias, then a valid premultiplied color.
// The convolve-alpha branch is loop-invariant and hoisted. In the non-convolving form the accumulated alpha's
// gain/bias result is dead — the scalar computes it and then immediately overwrites it with the origin tap's alpha —
// so it is not computed here; nothing else observes it.
func matrixConvFinalizeStageSIMD(z *lanes) {
	c := z.ctx.(*matrixConvCtx)
	gain := archsimd.BroadcastFloat32x4(c.gain)
	bias := archsimd.BroadcastFloat32x4(c.bias)
	zero := archsimd.BroadcastFloat32x4(0)
	if c.convolveAlpha {
		one := archsimd.BroadcastFloat32x4(1)
		for o := 0; o < stride; o += 4 {
			ca := clamp014(exprMulAdd4(archsimd.LoadFloat32x4(c.sum[3][o:]), gain, bias), zero, one)
			matrixConvStoreRGB(z, o, exprMulAdd4(archsimd.LoadFloat32x4(c.sum[0][o:]), gain, bias),
				exprMulAdd4(archsimd.LoadFloat32x4(c.sum[1][o:]), gain, bias),
				exprMulAdd4(archsimd.LoadFloat32x4(c.sum[2][o:]), gain, bias), ca, zero)
			ca.Store(z.a[o:])
		}
		return
	}
	for o := 0; o < stride; o += 4 {
		ca := archsimd.LoadFloat32x4(c.origAlpha[o:])
		matrixConvStoreRGB(z, o, exprMulAdd4(archsimd.LoadFloat32x4(c.sum[0][o:]), gain, bias).Mul(ca),
			exprMulAdd4(archsimd.LoadFloat32x4(c.sum[1][o:]), gain, bias).Mul(ca),
			exprMulAdd4(archsimd.LoadFloat32x4(c.sum[2][o:]), gain, bias).Mul(ca), ca, zero)
		ca.Store(z.a[o:])
	}
}

// matrixConvStoreRGB writes the finalize kernel's rgb quads made valid premul with respect to the alpha:
// clamp(rgb, 0, a), spelled as the scalar's minf(maxf(v, 0), a) so a NaN or negative alpha pins the channel the same
// way.
func matrixConvStoreRGB(z *lanes, o int, cr, cg, cb, ca, zero archsimd.Float32x4) {
	minf4(maxf4(cr, zero), ca).Store(z.r[o:])
	minf4(maxf4(cg, zero), ca).Store(z.g[o:])
	minf4(maxf4(cb, zero), ca).Store(z.b[o:])
}

///////////////////////////////////////////////////////////////////////////////
// arithmetic blend kernel

// arithMix4 is one channel quad of the arithmetic equation, left-associative exactly as the scalar expression
// k0*src*dst + k1*src + k2*dst + k3 parses: the leading product is two plain multiplies, the two interior additions go
// through exprMulAdd4 (the arch contracts them, or not, and this mirrors that), and the trailing k3 is a plain add
// because its operand is not a product.
func arithMix4(sv, dv, k0, k1, k2, k3 archsimd.Float32x4) archsimd.Float32x4 {
	return exprMulAdd4(k2, dv, exprMulAdd4(k1, sv, k0.Mul(sv).Mul(dv))).Add(k3)
}

// arithBlendStageSIMD is the vector arithmetic-blend kernel: saturate the four channels, then re-establish a valid
// premul color with color.rgb = min(color.rgb, max(color.a, pmClamp)) — the comparison-based minf4/maxf4 again, and
// clamp014 for the saturate, so NaN saturates to +0 as the scalar's clamp01 does.
func arithBlendStageSIMD(z *lanes) {
	c := z.ctx.(*arithmeticCtx)
	k0 := archsimd.BroadcastFloat32x4(c.k[0])
	k1 := archsimd.BroadcastFloat32x4(c.k[1])
	k2 := archsimd.BroadcastFloat32x4(c.k[2])
	k3 := archsimd.BroadcastFloat32x4(c.k[3])
	pmClamp := archsimd.BroadcastFloat32x4(c.pmClamp)
	zero := archsimd.BroadcastFloat32x4(0)
	one := archsimd.BroadcastFloat32x4(1)
	for o := 0; o < stride; o += 4 {
		outR := clamp014(arithMix4(archsimd.LoadFloat32x4(z.r[o:]),
			archsimd.LoadFloat32x4(c.res0[0][o:]), k0, k1, k2, k3), zero, one)
		outG := clamp014(arithMix4(archsimd.LoadFloat32x4(z.g[o:]),
			archsimd.LoadFloat32x4(c.res0[1][o:]), k0, k1, k2, k3), zero, one)
		outB := clamp014(arithMix4(archsimd.LoadFloat32x4(z.b[o:]),
			archsimd.LoadFloat32x4(c.res0[2][o:]), k0, k1, k2, k3), zero, one)
		outA := clamp014(arithMix4(archsimd.LoadFloat32x4(z.a[o:]),
			archsimd.LoadFloat32x4(c.res0[3][o:]), k0, k1, k2, k3), zero, one)
		lim := maxf4(outA, pmClamp)
		minf4(outR, lim).Store(z.r[o:])
		minf4(outG, lim).Store(z.g[o:])
		minf4(outB, lim).Store(z.b[o:])
		outA.Store(z.a[o:])
	}
}
