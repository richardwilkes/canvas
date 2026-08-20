// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// The stage benchmarks drive the dispatch variables, so one binary measures whichever kernel the build wired in:
// scalar, NEON, or (under goexperiment.simd) the simd kernels. Compare builds with benchstat.

// benchLanes returns a full register file of plausible pipeline values: r/g device-ish coordinates, b/a a color-ish
// pair. The coordinate stages read r/g, the gradient stages read r as t, so this covers every kernel's inputs.
func benchLanes() lanes {
	var z lanes
	for i := range stride {
		z.r[i] = float32(i) * 0.05
		z.g[i] = 1 - float32(i)*0.03
		z.b[i] = float32(i&3) * 0.25
		z.a[i] = 1
	}
	z.n = stride
	return z
}

// benchMatrixCtx returns a matrix stage context whose nine coefficients are all non-trivial, so no kernel's madf chain
// degenerates into a value the hardware can shortcut. The 2x2 block is a contraction (both eigenvalues below 1), so the
// benchmark loop — which feeds each iteration's output back in as the next iteration's input — converges instead of
// running away to infinities and NaNs, keeping every measured iteration on ordinary normal floats.
func benchMatrixCtx() *matrixCtx {
	c := &matrixCtx{}
	c.m[geom.MScaleX], c.m[geom.MSkewX], c.m[geom.MTransX] = 0.5, 0.125, 0.25
	c.m[geom.MSkewY], c.m[geom.MScaleY], c.m[geom.MTransY] = -0.0625, 0.75, -0.125
	c.m[geom.MPersp0], c.m[geom.MPersp1], c.m[geom.MPersp2] = 0, 0, 1
	return c
}

func BenchmarkSeedStage(b *testing.B) {
	z := benchLanes()
	z.dx, z.dy = 137, 42
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		seedStageFn(&z)
	}
}

func BenchmarkClampX1Stage(b *testing.B) {
	z := benchLanes()
	// Straddle the clamp: some lanes below 0, some above 1, most inside.
	for i := range stride {
		z.r[i] = float32(i)*0.1 - 0.3
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		clampX1StageFn(&z)
	}
}

func BenchmarkMatrixTranslateStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchMatrixCtx()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		matrixTranslateStageFn(&z)
	}
}

func BenchmarkMatrixScaleTranslateStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchMatrixCtx()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		matrixScaleTranslateStageFn(&z)
	}
}

func BenchmarkMatrixAffineStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchMatrixCtx()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		matrixAffineStageFn(&z)
	}
}

func BenchmarkGradient2StopStage(b *testing.B) {
	z := benchLanes()
	z.ctx = &gradientCtx{
		cl:     colorcore.Color4f{R: 0.1, G: 0.2, B: 0.3, A: 1},
		factor: colorcore.Color4f{R: 0.8, G: -0.15, B: 0.55, A: -0.25},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		gradient2StopStageFn(&z)
	}
}

func BenchmarkGradientEvenlyStage(b *testing.B) {
	const stops = 8
	c := &gradientCtx{stops: make([]gradStop, stops+1), gapCount: stops - 1}
	for i := range c.stops {
		f := float32(i) * 0.1
		// The R channel is the identity ramp (madf(t, 1, 0) == t): the stage writes its result back over the r
		// register, which is also the t it reads, so an identity R keeps t at its initial spread for every iteration
		// of the benchmark loop instead of collapsing the lanes onto one stop — or, worse, drifting out of the [0,1]
		// domain the scalar stage's unclamped index requires. The other channels carry ordinary ramp coefficients.
		c.stops[i] = gradStop{
			fr: 1, fg: 0.25 - f, fb: f, fa: 0.125,
			br: 0, bg: 1 - f, bb: 0.5, ba: 0.75 + f,
		}
	}
	z := benchLanes()
	// t spread across the whole [0,1] domain so the lanes of a quad gather from different stop records.
	for i := range stride {
		z.r[i] = float32(i) / float32(stride-1)
	}
	z.ctx = c
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		gradientEvenlyStageFn(&z)
	}
}

// benchSamplerCtx returns a sampler context holding one bilinear tap's worth of plausible state: device-ish saved
// coordinates, fractional offsets spread across the unit interval, and the matching weight lanes (which the accumulate
// benchmark reads without running an axis stage first).
func benchSamplerCtx() *samplerCtx {
	c := &samplerCtx{}
	for i := range stride {
		c.x[i] = float32(i) + 0.25
		c.y[i] = 12 - float32(i)*0.5
		c.fx[i] = float32(i&7) * 0.125
		c.fy[i] = float32((i+5)&7) * 0.125
		c.scalex[i] = 1 - c.fx[i]
		c.scaley[i] = c.fy[i]
	}
	return c
}

func BenchmarkMatrix4x5Stage(b *testing.B) {
	var m [20]float32
	for i := range m {
		m[i] = float32(i)*0.1 - 0.7
	}
	z := benchLanes()
	z.ctx = &m
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		matrix4x5StageFn(&z)
	}
}

func BenchmarkClamp01Stage(b *testing.B) {
	z := benchLanes()
	// Straddle the clamp on every channel: some lanes below 0, some above 1, most inside. Like the clamp_x_1
	// benchmark, the first iteration pulls the lanes in and the rest measure the in-range cost.
	for i := range stride {
		z.r[i] = float32(i)*0.1 - 0.3
		z.g[i] = 0.9 - float32(i)*0.1
		z.b[i] = float32(i&3)*0.5 - 0.25
		z.a[i] = float32(i) * 0.08
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		clamp01StageFn(&z)
	}
}

func BenchmarkClampGamutStage(b *testing.B) {
	z := benchLanes()
	// Alpha straddles [0,1] and the color channels straddle [0,a], so both clamps bite on the first iteration; the
	// stage is idempotent afterward, which keeps the loop's feedback stable.
	for i := range stride {
		z.a[i] = float32(i)*0.1 - 0.2
		z.r[i] = float32(i)*0.09 - 0.1
		z.g[i] = 0.8 - float32(i)*0.07
		z.b[i] = float32(i&3) * 0.4
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		clampGamutStageFn(&z)
	}
}

func BenchmarkPremulStage(b *testing.B) {
	// benchLanes' alpha is 1, which is what keeps the loop's feedback stable: the multiplies all run, but the color
	// lanes stay put instead of decaying toward denormals over the benchmark's iterations.
	z := benchLanes()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		premulStageFn(&z)
	}
}

func BenchmarkUnpremulStage(b *testing.B) {
	z := benchLanes()
	// Zero color lanes make the stage idempotent (0*scale is 0 for every finite scale), so the loop's feedback stays
	// put while the reciprocal and its finiteness guard still run over a realistic alpha spread — including the fully
	// transparent lane whose 1/0 the guard must reject.
	for i := range stride {
		z.r[i], z.g[i], z.b[i] = 0, 0, 0
		z.a[i] = float32(i) / float32(stride-1)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		unpremulStageFn(&z)
	}
}

func BenchmarkScale1FloatStage(b *testing.B) {
	// The scale is 1 for the same reason premul's alpha is: any other factor drives the lanes to denormals (or to
	// infinity) within the benchmark's iteration count, measuring the hardware's slow paths instead of the stage.
	c := float32(1)
	z := benchLanes()
	z.ctx = &c
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		scale1FloatStageFn(&z)
	}
}

func BenchmarkMaskApplyStage(b *testing.B) {
	var mask laneMask
	for i := range mask {
		if i&3 != 3 {
			mask[i] = 0xFFFFFFFF
		}
	}
	z := benchLanes()
	z.ctx = &mask
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		maskApplyStageFn(&z)
	}
}

func BenchmarkSetRGBStage(b *testing.B) {
	z := benchLanes()
	z.ctx = &gatherCtx{setRGB: [3]float32{0.25, 0.5, 0.75}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		setRGBStageFn(&z)
	}
}

func BenchmarkMoveSrcDstStage(b *testing.B) {
	z := benchLanes()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		moveSrcDstStageFn(&z)
	}
}

func BenchmarkMoveDstSrcStage(b *testing.B) {
	z := benchLanes()
	for i := range stride {
		z.dr[i], z.dg[i], z.db[i], z.da[i] = z.r[i], z.g[i], z.b[i], z.a[i]
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		moveDstSrcStageFn(&z)
	}
}

// The four bilinear axis stages read the sampler context and write one coordinate register plus one weight lane, so
// nothing they write feeds back into anything they read: every iteration measures the same work.

func BenchmarkBilinearNXStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchSamplerCtx()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bilinearNXStageFn(&z)
	}
}

func BenchmarkBilinearPXStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchSamplerCtx()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bilinearPXStageFn(&z)
	}
}

func BenchmarkBilinearNYStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchSamplerCtx()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bilinearNYStageFn(&z)
	}
}

func BenchmarkBilinearPYStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchSamplerCtx()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bilinearPYStageFn(&z)
	}
}

func BenchmarkAccumulateStage(b *testing.B) {
	z := benchLanes()
	z.ctx = benchSamplerCtx()
	// The accumulators start at zero and the taps are ordinary [0,1] values, so the dst registers climb until the
	// per-iteration increment falls below their ulp (~2^24 increments in) and then hold there: no reset is needed and
	// no lane ever reaches an infinity or a denormal.
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		accumulateStageFn(&z)
	}
}
