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
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// TestStageSIMDMatchesScalar drives every simd stage kernel and its portable Go twin over identical randomized register
// files (hostile float classes included) and requires bitwise identity, exactly as TestStageNEONMatchesScalar does for
// the NEON kernels. Each subtest mirrors its NEON counterpart's setup so the two suites cover the same domains.
func TestStageSIMDMatchesScalar(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd kernels require; dispatch stays on the default forms")
	}
	rng := rand.New(rand.NewPCG(21, 22))

	t.Run("seed", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			z1.dx = int32(rng.Uint32())
			z1.dy = int32(rng.Uint32())
			z2 := z1
			seedStage(&z1)
			seedStageSIMD(&z2)
			eqLanes(t, "seed", &z2, &z1)
		}
	})

	t.Run("clampX1", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			z2 := z1
			clampX1Stage(&z1)
			clampX1StageSIMD(&z2)
			eqLanes(t, "clampX1", &z2, &z1)
		}
	})

	matrices := func() *matrixCtx {
		c := &matrixCtx{}
		for i := range c.m {
			if rng.IntN(4) == 0 {
				c.m[i] = hostileFloats[rng.IntN(len(hostileFloats))]
			} else {
				c.m[i] = quietFloat(rng.Uint32())
			}
		}
		return c
	}

	t.Run("matrixTranslate", func(t *testing.T) {
		for range 1024 {
			ctx := matrices()
			z1 := randLanes(rng)
			z1.ctx = ctx
			z2 := z1
			matrixTranslateStage(&z1)
			matrixTranslateStageSIMD(&z2)
			eqLanes(t, "matrixTranslate", &z2, &z1)
		}
	})

	t.Run("matrixScaleTranslate", func(t *testing.T) {
		for range 1024 {
			ctx := matrices()
			z1 := randLanes(rng)
			z1.ctx = ctx
			z2 := z1
			matrixScaleTranslateStage(&z1)
			matrixScaleTranslateStageSIMD(&z2)
			eqLanes(t, "matrixScaleTranslate", &z2, &z1)
		}
	})

	t.Run("matrixAffine", func(t *testing.T) {
		for range 1024 {
			ctx := matrices()
			z1 := randLanes(rng)
			z1.ctx = ctx
			z2 := z1
			matrixAffineStage(&z1)
			matrixAffineStageSIMD(&z2)
			eqLanes(t, "matrixAffine", &z2, &z1)
		}
	})

	t.Run("gradient2Stop", func(t *testing.T) {
		for range 1024 {
			ctx := &gradientCtx{
				cl: colorcore.Color4f{
					R: quietFloat(rng.Uint32()), G: quietFloat(rng.Uint32()),
					B: quietFloat(rng.Uint32()), A: quietFloat(rng.Uint32()),
				},
				factor: colorcore.Color4f{
					R: quietFloat(rng.Uint32()), G: quietFloat(rng.Uint32()),
					B: quietFloat(rng.Uint32()), A: quietFloat(rng.Uint32()),
				},
			}
			z1 := randLanes(rng)
			z1.ctx = ctx
			z2 := z1
			gradient2StopStage(&z1)
			gradient2StopStageSIMD(&z2)
			eqLanes(t, "gradient2Stop", &z2, &z1)
		}
	})

	t.Run("gradientEvenly", func(t *testing.T) {
		for range 1024 {
			nStops := 2 + rng.IntN(20)
			ctx := &gradientCtx{
				stops:    make([]gradStop, nStops+1),
				gapCount: float32(nStops - 1),
			}
			for i := range ctx.stops {
				ctx.stops[i] = gradStop{
					fr: rng.Float32(), fg: rng.Float32(), fb: rng.Float32(), fa: rng.Float32(),
					br: rng.Float32(), bg: rng.Float32(), bb: rng.Float32(), ba: rng.Float32(),
				}
			}
			z1 := randLanes(rng)
			// The evenly stage runs after a tile stage, so its defined domain is t in [0,1]; feed the valid lanes
			// in-domain values and leave hostile garbage in the scratch tail, which the simd kernel must tolerate
			// (clamped gathers) without perturbing the live lanes.
			n := 1 + rng.IntN(stride)
			for i := range n {
				z1.r[i] = rng.Float32()
			}
			z1.n = n
			ctx2 := &gradientCtx{stops: append([]gradStop(nil), ctx.stops...), gapCount: ctx.gapCount}
			z1.ctx = ctx
			z2 := z1
			z2.ctx = ctx2
			gradientEvenlyStage(&z1)
			gradientEvenlyStageSIMD(&z2)
			for i := range n {
				for ch, pair := range [][2]float32{
					{z2.r[i], z1.r[i]}, {z2.g[i], z1.g[i]}, {z2.b[i], z1.b[i]}, {z2.a[i], z1.a[i]},
				} {
					if math.Float32bits(pair[0]) != math.Float32bits(pair[1]) {
						t.Fatalf("gradientEvenly: lane %d ch %d = %08x, want %08x (n=%d stops=%d)",
							i, ch, math.Float32bits(pair[0]), math.Float32bits(pair[1]), n, nStops)
					}
				}
			}
		}
	})

	t.Run("matrix4x5", func(t *testing.T) {
		for range 1024 {
			var m [20]float32
			for i := range m {
				if rng.IntN(4) == 0 {
					m[i] = hostileFloats[rng.IntN(len(hostileFloats))]
				} else {
					m[i] = quietFloat(rng.Uint32())
				}
			}
			z1 := randLanes(rng)
			z1.ctx = &m
			z2 := z1
			matrix4x5Stage(&z1)
			matrix4x5StageSIMD(&z2)
			eqLanes(t, "matrix4x5", &z2, &z1)
		}
	})

	t.Run("clamp01", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			z2 := z1
			clamp01Stage(&z1)
			clamp01StageSIMD(&z2)
			eqLanes(t, "clamp01", &z2, &z1)
		}
	})

	t.Run("clampGamut", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			// Alpha drives the rgb clamp's upper bound, so give it the classes a premultiplied lane can actually
			// carry — including the ones that make the bound degenerate (0, negative, NaN).
			randAlphas(rng, &z1.a)
			z2 := z1
			clampGamutStage(&z1)
			clampGamutStageSIMD(&z2)
			eqLanes(t, "clampGamut", &z2, &z1)
		}
	})

	t.Run("premul", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			randAlphas(rng, &z1.a)
			z2 := z1
			premulStage(&z1)
			premulStageSIMD(&z2)
			eqLanes(t, "premul", &z2, &z1)
		}
	})

	t.Run("unpremul", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			// The 1/a guard is the whole stage: a = 0 overflows to +Inf and must be rejected (scale 0), a = -0 gives
			// -Inf and must be kept, a = NaN must be rejected, and a negative alpha must produce a negative scale.
			randAlphas(rng, &z1.a)
			z2 := z1
			unpremulStage(&z1)
			unpremulStageSIMD(&z2)
			eqLanes(t, "unpremul", &z2, &z1)
		}
	})

	t.Run("scale1Float", func(t *testing.T) {
		for range 1024 {
			c := quietFloat(rng.Uint32())
			if rng.IntN(4) == 0 {
				c = hostileFloats[rng.IntN(len(hostileFloats))]
			}
			z1 := randLanes(rng)
			z1.ctx = &c
			z2 := z1
			scale1FloatStage(&z1)
			scale1FloatStageSIMD(&z2)
			eqLanes(t, "scale1Float", &z2, &z1)
		}
	})

	t.Run("maskApply", func(t *testing.T) {
		for range 1024 {
			var mask laneMask
			for i := range mask {
				switch rng.IntN(4) {
				case 0:
					mask[i] = 0 // a masked-out lane
				case 1:
					mask[i] = rng.Uint32() // an arbitrary bit pattern the AND must reproduce exactly
				default:
					mask[i] = 0xFFFFFFFF // a kept lane
				}
			}
			z1 := randLanes(rng)
			z1.ctx = &mask
			z2 := z1
			maskApplyStage(&z1)
			maskApplyStageSIMD(&z2)
			eqLanes(t, "maskApply", &z2, &z1)
		}
	})

	t.Run("setRGB", func(t *testing.T) {
		for range 1024 {
			g := &gatherCtx{setRGB: [3]float32{
				quietFloat(rng.Uint32()), quietFloat(rng.Uint32()), quietFloat(rng.Uint32()),
			}}
			z1 := randLanes(rng)
			z1.ctx = g
			z2 := z1
			setRGBStage(&z1)
			setRGBStageSIMD(&z2)
			eqLanes(t, "setRGB", &z2, &z1)
		}
	})

	t.Run("moveSrcDst", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			z2 := z1
			moveSrcDstStage(&z1)
			moveSrcDstStageSIMD(&z2)
			eqLanes(t, "moveSrcDst", &z2, &z1)
			eqLanesDst(t, "moveSrcDst", &z2, &z1)
		}
	})

	t.Run("moveDstSrc", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			z2 := z1
			moveDstSrcStage(&z1)
			moveDstSrcStageSIMD(&z2)
			eqLanes(t, "moveDstSrc", &z2, &z1)
			eqLanesDst(t, "moveDstSrc", &z2, &z1)
		}
	})

	t.Run("bilinear", func(t *testing.T) {
		for _, tc := range []struct {
			scalar, simd stageFn
			name         string
			isX          bool
		}{
			{name: "nx", scalar: bilinearNXStage, simd: bilinearNXStageSIMD, isX: true},
			{name: "px", scalar: bilinearPXStage, simd: bilinearPXStageSIMD, isX: true},
			{name: "ny", scalar: bilinearNYStage, simd: bilinearNYStageSIMD},
			{name: "py", scalar: bilinearPYStage, simd: bilinearPYStageSIMD},
		} {
			for range 1024 {
				c1 := randSamplerCtx(rng)
				c2 := *c1 // the stage writes the weight lanes, so each side needs its own copy
				z1 := randLanes(rng)
				z1.ctx = c1
				z2 := z1
				z2.ctx = &c2
				tc.scalar(&z1)
				tc.simd(&z2)
				eqLanes(t, "bilinear_"+tc.name, &z2, &z1)
				if tc.isX {
					eqReg(t, "bilinear_"+tc.name, "scalex", &c2.scalex, &c1.scalex)
				} else {
					eqReg(t, "bilinear_"+tc.name, "scaley", &c2.scaley, &c1.scaley)
				}
			}
		}
	})

	t.Run("accumulate", func(t *testing.T) {
		for range 1024 {
			c := randSamplerCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = c
			z2 := z1
			accumulateStage(&z1)
			accumulateStageSIMD(&z2)
			eqLanes(t, "accumulate", &z2, &z1)
			eqLanesDst(t, "accumulate", &z2, &z1)
		}
	})
}

// randAlphas overwrites an alpha register with the classes the premul/unpremul/clamp_gamut kernels actually meet: the
// ordinary [0,1] coverage, exact zero, negatives, and (through the hostile mix) both signed zeros, the infinities and
// NaN. The unpremul guard is built directly on those corners — 1/+0 overflows to +Inf and must be rejected while
// 1/-0 = -Inf must be kept — so leaving them to the generic 1-in-3 hostile draw would under-cover them.
func randAlphas(rng *rand.Rand, dst *[stride]float32) {
	for i := range dst {
		switch rng.IntN(6) {
		case 0:
			dst[i] = hostileFloats[rng.IntN(len(hostileFloats))]
		case 1:
			dst[i] = 0
		case 2:
			dst[i] = -rng.Float32()
		case 3:
			dst[i] = float32(math.NaN())
		default:
			dst[i] = rng.Float32()
		}
	}
}

// randSamplerCtx returns a sampler context whose saved coordinates, fractional offsets and weight lanes all carry the
// hostile/random mix, so the bilinear and accumulate kernels are compared over every float class their inputs can hold.
func randSamplerCtx(rng *rand.Rand) *samplerCtx {
	c := &samplerCtx{}
	for _, reg := range []*[stride]float32{&c.x, &c.y, &c.fx, &c.fy, &c.scalex, &c.scaley} {
		randFloats(rng, reg)
	}
	return c
}

// eqLanesDst is eqLanes for the dst registers, which the move_src_dst, move_dst_src and accumulate kernels write.
func eqLanesDst(t *testing.T, name string, got, want *lanes) {
	t.Helper()
	eqReg(t, name, "dr", &got.dr, &want.dr)
	eqReg(t, name, "dg", &got.dg, &want.dg)
	eqReg(t, name, "db", &got.db, &want.db)
	eqReg(t, name, "da", &got.da, &want.da)
}

// TestStageSIMDWiring locks the goexperiment.simd build's dispatch to the per-arch preference table: on qualifying
// hardware init must have repointed every preferred kernel at its simd form, and must have left every declined kernel
// alone (on arm64 three stay on the faster NEON assembly — see stage_simd_arm64.go), so neither a silent fallback nor
// a silently wired regression can survive a refactor.
func TestStageSIMDWiring(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd kernels require; dispatch stays on the default forms")
	}
	for name, c := range map[string]struct {
		got       stageFn
		simd      stageFn
		preferred bool
	}{
		"seed":                 {seedStageFn, seedStageSIMD, preferSIMDSeed},
		"clampX1":              {clampX1StageFn, clampX1StageSIMD, preferSIMDClampX1},
		"matrixTranslate":      {matrixTranslateStageFn, matrixTranslateStageSIMD, preferSIMDMatrixTranslate},
		"matrixScaleTranslate": {matrixScaleTranslateStageFn, matrixScaleTranslateStageSIMD, preferSIMDMatrixScaleTranslate},
		"matrixAffine":         {matrixAffineStageFn, matrixAffineStageSIMD, preferSIMDMatrixAffine},
		"gradient2Stop":        {gradient2StopStageFn, gradient2StopStageSIMD, preferSIMDGradient2Stop},
		"gradientEvenly":       {gradientEvenlyStageFn, gradientEvenlyStageSIMD, preferSIMDGradientEvenly},
		"matrix4x5":            {matrix4x5StageFn, matrix4x5StageSIMD, preferSIMDMatrix4x5},
		"maskApply":            {maskApplyStageFn, maskApplyStageSIMD, preferSIMDMaskApply},
		"clamp01":              {clamp01StageFn, clamp01StageSIMD, preferSIMDClamp01},
		"clampGamut":           {clampGamutStageFn, clampGamutStageSIMD, preferSIMDClampGamut},
		"premul":               {premulStageFn, premulStageSIMD, preferSIMDPremul},
		"unpremul":             {unpremulStageFn, unpremulStageSIMD, preferSIMDUnpremul},
		"scale1Float":          {scale1FloatStageFn, scale1FloatStageSIMD, preferSIMDScale1Float},
		"setRGB":               {setRGBStageFn, setRGBStageSIMD, preferSIMDSetRGB},
		"moveSrcDst":           {moveSrcDstStageFn, moveSrcDstStageSIMD, preferSIMDMoveSrcDst},
		"moveDstSrc":           {moveDstSrcStageFn, moveDstSrcStageSIMD, preferSIMDMoveDstSrc},
		"bilinearNX":           {bilinearNXStageFn, bilinearNXStageSIMD, preferSIMDBilinear},
		"bilinearPX":           {bilinearPXStageFn, bilinearPXStageSIMD, preferSIMDBilinear},
		"bilinearNY":           {bilinearNYStageFn, bilinearNYStageSIMD, preferSIMDBilinear},
		"bilinearPY":           {bilinearPYStageFn, bilinearPYStageSIMD, preferSIMDBilinear},
		"accumulate":           {accumulateStageFn, accumulateStageSIMD, preferSIMDAccumulate},
	} {
		wired := reflect.ValueOf(c.got).Pointer() == reflect.ValueOf(c.simd).Pointer()
		if c.preferred && !wired {
			t.Fatalf("%s: dispatch fn is not the simd kernel", name)
		}
		if !c.preferred && wired {
			t.Fatalf("%s: dispatch fn is the simd kernel, but the default lane is preferred here", name)
		}
	}
}
