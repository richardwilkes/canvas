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
	"strconv"
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

	t.Run("morphAggInit", func(t *testing.T) {
		for range 1024 {
			c1 := randMorphologyCtx(rng)
			c2 := cloneMorphologyCtx(c1)
			z1 := randLanes(rng)
			z1.ctx = c1
			z2 := z1
			z2.ctx = c2
			morphAggInitStage(&z1)
			morphAggInitStageSIMD(&z2)
			eqLanes(t, "morphAggInit", &z2, &z1)
			eqScratch4(t, "morphAggInit", "agg", &c2.agg, &c1.agg)
		}
	})

	t.Run("morphPlusCoords", func(t *testing.T) {
		for range 1024 {
			c := randMorphologyCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = c.nextStep(randCoefFloat(rng), randCoefFloat(rng))
			z2 := z1
			morphPlusCoordsStage(&z1)
			morphPlusCoordsStageSIMD(&z2)
			eqLanes(t, "morphPlusCoords", &z2, &z1)
		}
	})

	t.Run("morphTakePlus", func(t *testing.T) {
		for range 1024 {
			c1 := randMorphologyCtx(rng)
			c2 := cloneMorphologyCtx(c1)
			dx, dy := randCoefFloat(rng), randCoefFloat(rng)
			z1 := randLanes(rng)
			z1.ctx = c1.nextStep(dx, dy)
			z2 := z1
			z2.ctx = c2.nextStep(dx, dy)
			morphTakePlusStage(&z1)
			morphTakePlusStageSIMD(&z2)
			eqLanes(t, "morphTakePlus", &z2, &z1)
			eqScratch4(t, "morphTakePlus", "plus", &c2.plus, &c1.plus)
		}
	})

	t.Run("morphMaxAgg", func(t *testing.T) {
		for range 1024 {
			c1 := randMorphologyCtx(rng)
			c2 := cloneMorphologyCtx(c1)
			z1 := randLanes(rng)
			z1.ctx = c1
			z2 := z1
			z2.ctx = c2
			morphMaxAggStage(&z1)
			morphMaxAggStageSIMD(&z2)
			eqLanes(t, "morphMaxAgg", &z2, &z1)
			eqScratch4(t, "morphMaxAgg", "agg", &c2.agg, &c1.agg)
		}
	})

	t.Run("morphSparseAggMinus", func(t *testing.T) {
		for range 1024 {
			c1 := randMorphologyCtx(rng)
			c2 := cloneMorphologyCtx(c1)
			dx, dy := randCoefFloat(rng), randCoefFloat(rng)
			z1 := randLanes(rng)
			z1.ctx = c1.nextStep(dx, dy)
			z2 := z1
			z2.ctx = c2.nextStep(dx, dy)
			morphSparseAggMinusStage(&z1)
			morphSparseAggMinusStageSIMD(&z2)
			eqLanes(t, "morphSparseAggMinus", &z2, &z1)
			eqScratch4(t, "morphSparseAggMinus", "agg", &c2.agg, &c1.agg)
		}
	})

	t.Run("morphSparseMaxAgg", func(t *testing.T) {
		for range 1024 {
			c1 := randMorphologyCtx(rng)
			c2 := cloneMorphologyCtx(c1)
			z1 := randLanes(rng)
			z1.ctx = c1
			z2 := z1
			z2.ctx = c2
			morphSparseMaxAggStage(&z1)
			morphSparseMaxAggStageSIMD(&z2)
			eqLanes(t, "morphSparseMaxAgg", &z2, &z1)
			eqScratch4(t, "morphSparseMaxAgg", "agg", &c2.agg, &c1.agg)
		}
	})

	t.Run("morphReturn", func(t *testing.T) {
		for range 1024 {
			c := randMorphologyCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = c
			z2 := z1
			morphReturnStage(&z1)
			morphReturnStageSIMD(&z2)
			eqLanes(t, "morphReturn", &z2, &z1)
		}
	})

	t.Run("normalSetCoords", func(t *testing.T) {
		for range 1024 {
			c := randNormalCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = &c.taps[rng.IntN(len(c.taps))]
			z2 := z1
			normalSetCoordsStage(&z1)
			normalSetCoordsStageSIMD(&z2)
			eqLanes(t, "normalSetCoords", &z2, &z1)
		}
	})

	t.Run("normalFilter", func(t *testing.T) {
		for range 1024 {
			c := randNormalCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = c
			z2 := z1
			normalFilterStage(&z1)
			normalFilterStageSIMD(&z2)
			eqLanes(t, "normalFilter", &z2, &z1)
		}
	})

	t.Run("matrixConvCoords", func(t *testing.T) {
		for range 1024 {
			c := randMatrixConvCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = randMatrixConvTap(rng, c)
			z2 := z1
			matrixConvCoordsStage(&z1)
			matrixConvCoordsStageSIMD(&z2)
			eqLanes(t, "matrixConvCoords", &z2, &z1)
		}
	})

	t.Run("matrixConvAccum", func(t *testing.T) {
		for range 1024 {
			c1 := randMatrixConvCtx(rng)
			c2 := cloneMatrixConvCtx(c1)
			t1 := randMatrixConvTap(rng, c1)
			t2 := c2.nextTap()
			t2.fkx, t2.fky, t2.k, t2.isOrigin = t1.fkx, t1.fky, t1.k, t1.isOrigin
			z1 := randLanes(rng)
			z1.ctx = t1
			z2 := z1
			z2.ctx = t2
			matrixConvAccumStage(&z1)
			matrixConvAccumStageSIMD(&z2)
			eqLanes(t, "matrixConvAccum", &z2, &z1)
			eqScratch4(t, "matrixConvAccum", "sum", &c2.sum, &c1.sum)
			eqReg(t, "matrixConvAccum", "origAlpha", &c2.origAlpha, &c1.origAlpha)
		}
	})

	t.Run("matrixConvFinalize", func(t *testing.T) {
		for range 1024 {
			c := randMatrixConvCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = c
			z2 := z1
			matrixConvFinalizeStage(&z1)
			matrixConvFinalizeStageSIMD(&z2)
			eqLanes(t, "matrixConvFinalize", &z2, &z1)
		}
	})

	t.Run("arithBlend", func(t *testing.T) {
		for range 1024 {
			c := randArithmeticCtx(rng)
			z1 := randLanes(rng)
			z1.ctx = c
			z2 := z1
			arithBlendStage(&z1)
			arithBlendStageSIMD(&z2)
			eqLanes(t, "arithBlend", &z2, &z1)
		}
	})
}

// randCoefFloat returns one stage-uniform scalar for the image-filter kernels: usually a plausible coefficient (a
// convolution weight, a gain or bias, an edge bound, a morphology tap delta), sometimes an arbitrary bit pattern or a
// hostile float class. The plausible draw is what the imagefilter callers actually produce; the other two cover the
// corners the kernels' comparisons, divides and NaN handling pivot on.
func randCoefFloat(rng *rand.Rand) float32 {
	switch rng.IntN(8) {
	case 0:
		return hostileFloats[rng.IntN(len(hostileFloats))]
	case 1:
		return quietFloat(rng.Uint32())
	default:
		return rng.Float32()*8 - 4
	}
}

// randMorphologyCtx returns a morphology context whose aggregate, +delta hold and saved coordinates carry the
// hostile/random mix, with the exactly ±1 flip NewLinearMorphology/NewSparseMorphology set (dilate/erode) — the sign
// decides which operand each maxf yields, so both are drawn.
func randMorphologyCtx(rng *rand.Rand) *morphologyCtx {
	c := &morphologyCtx{flip: 1}
	if rng.IntN(2) == 0 {
		c.flip = -1
	}
	for ch := range 4 {
		randFloats(rng, &c.agg[ch])
		randFloats(rng, &c.plus[ch])
	}
	randFloats(rng, &c.coords[0])
	randFloats(rng, &c.coords[1])
	return c
}

// cloneMorphologyCtx returns an independent copy of a morphology context (its scratch is all arrays, so a field-wise
// copy suffices) for the kernels that write agg or plus and so need one context per side. The steps slice is
// deliberately not copied: each side hands itself its own morphStep, whose back-pointer must reach its own context.
func cloneMorphologyCtx(c *morphologyCtx) *morphologyCtx {
	return &morphologyCtx{flip: c.flip, agg: c.agg, plus: c.plus, coords: c.coords}
}

// randNormalCtx returns a Sobel-normal context with its nine fixed taps wired in the column-major order
// NormalShader.appendStages builds them. Half the contexts are hostile — every scalar and every tap alpha drawn from
// the hostile/random mix — and half carry the domain the pipeline actually produces: alphas in [0,1], an ordered
// finite edge rect, and a negative surface depth. The in-domain half matters because the hostile half saturates the
// Sobel dot products into infinities and NaNs, where the normalize divide washes out the ~1-ULP differences that
// separate a correct madf from a plain single-precision FMA.
func randNormalCtx(rng *rand.Rand) *normalCtx {
	c := &normalCtx{}
	if rng.IntN(2) == 0 {
		c.l, c.t = -rng.Float32()*64, -rng.Float32()*64
		c.r, c.b = 64+rng.Float32()*64, 64+rng.Float32()*64
		c.negSurfaceDepth = -(0.5 + rng.Float32()*8)
		for i := range c.alpha {
			for j := range c.alpha[i] {
				c.alpha[i][j] = rng.Float32()
			}
		}
	} else {
		c.l, c.t = randCoefFloat(rng), randCoefFloat(rng)
		c.r, c.b = randCoefFloat(rng), randCoefFloat(rng)
		c.negSurfaceDepth = randCoefFloat(rng)
		for i := range c.alpha {
			randFloats(rng, &c.alpha[i])
		}
	}
	randFloats(rng, &c.coords[0])
	randFloats(rng, &c.coords[1])
	for i := range c.taps {
		c.taps[i].c = c
		c.taps[i].ddx = float32(i/3) - 1
		c.taps[i].ddy = float32(i%3) - 1
	}
	return c
}

// randMatrixConvCtx returns a matrix-convolution context shaped like the ones MatrixConvShader.appendStages builds: an
// offset inside a 1x1..5x5 kernel, a gain and a pre-scaled bias, both settings of convolveAlpha, and an accumulator,
// origin-alpha slot and saved coordinates carrying the hostile/random mix.
func randMatrixConvCtx(rng *rand.Rand) *matrixConvCtx {
	size := 1 + rng.IntN(5)
	c := &matrixConvCtx{
		ox:            float32(rng.IntN(size)),
		oy:            float32(rng.IntN(size)),
		gain:          randCoefFloat(rng),
		bias:          randCoefFloat(rng) / 255,
		convolveAlpha: rng.IntN(2) == 0,
	}
	for ch := range 4 {
		randFloats(rng, &c.sum[ch])
	}
	randFloats(rng, &c.origAlpha)
	randFloats(rng, &c.coords[0])
	randFloats(rng, &c.coords[1])
	return c
}

// cloneMatrixConvCtx returns an independent copy of a convolution context for the accumulator subtest, which writes
// sum and origAlpha and so needs one per side. As with the morphology clone the taps slice is left empty so each side
// hands itself a tap pointing back at its own context.
func cloneMatrixConvCtx(c *matrixConvCtx) *matrixConvCtx {
	return &matrixConvCtx{
		sum: c.sum, coords: c.coords, origAlpha: c.origAlpha, ox: c.ox, oy: c.oy,
		gain: c.gain, bias: c.bias, convolveAlpha: c.convolveAlpha,
	}
}

// randMatrixConvTap hands c one kernel tap: a position within the kernel and its coefficient. One draw in four forces
// the tap onto the kernel origin, which is where the non-convolving form records the alpha it later restores — far
// more often than random positions would hit it.
func randMatrixConvTap(rng *rand.Rand, c *matrixConvCtx) *matrixConvTap {
	t := c.nextTap()
	t.fkx, t.fky = float32(rng.IntN(5)), float32(rng.IntN(5))
	if rng.IntN(4) == 0 {
		t.fkx, t.fky = c.ox, c.oy
	}
	t.k = randCoefFloat(rng)
	t.isOrigin = t.fkx == c.ox && t.fky == c.oy
	return t
}

// randArithmeticCtx returns an arithmetic-blend context: the four coefficients, both settings of the premul clamp (1
// when the blend does not enforce premul, 0 when it does), and a dst hold carrying the hostile/random mix.
func randArithmeticCtx(rng *rand.Rand) *arithmeticCtx {
	c := &arithmeticCtx{pmClamp: 1}
	if rng.IntN(2) == 0 {
		c.pmClamp = 0
	}
	for i := range c.k {
		c.k[i] = randCoefFloat(rng)
	}
	for ch := range 4 {
		randFloats(rng, &c.res0[ch])
	}
	randFloats(rng, &c.coords[0])
	randFloats(rng, &c.coords[1])
	return c
}

// eqScratch4 is eqReg for a four-channel context scratch buffer — the morphology aggregate and +delta hold, and the
// convolution accumulator.
func eqScratch4(t *testing.T, name, which string, got, want *[4][stride]float32) {
	t.Helper()
	for ch := range 4 {
		eqReg(t, name, which+strconv.Itoa(ch), &got[ch], &want[ch])
	}
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
		"morphAggInit":         {morphAggInitStageFn, morphAggInitStageSIMD, preferSIMDMorphAggInit},
		"morphPlusCoords":      {morphPlusCoordsStageFn, morphPlusCoordsStageSIMD, preferSIMDMorphPlusCoords},
		"morphTakePlus":        {morphTakePlusStageFn, morphTakePlusStageSIMD, preferSIMDMorphTakePlus},
		"morphMaxAgg":          {morphMaxAggStageFn, morphMaxAggStageSIMD, preferSIMDMorphMaxAgg},
		"morphSparseAggMinus":  {morphSparseAggMinusStageFn, morphSparseAggMinusStageSIMD, preferSIMDMorphSparseAggMinus},
		"morphSparseMaxAgg":    {morphSparseMaxAggStageFn, morphSparseMaxAggStageSIMD, preferSIMDMorphSparseMaxAgg},
		"morphReturn":          {morphReturnStageFn, morphReturnStageSIMD, preferSIMDMorphReturn},
		"normalSetCoords":      {normalSetCoordsStageFn, normalSetCoordsStageSIMD, preferSIMDNormalSetCoords},
		"normalFilter":         {normalFilterStageFn, normalFilterStageSIMD, preferSIMDNormalFilter},
		"matrixConvCoords":     {matrixConvCoordsStageFn, matrixConvCoordsStageSIMD, preferSIMDMatrixConvCoords},
		"matrixConvAccum":      {matrixConvAccumStageFn, matrixConvAccumStageSIMD, preferSIMDMatrixConvAccum},
		"matrixConvFinalize":   {matrixConvFinalizeStageFn, matrixConvFinalizeStageSIMD, preferSIMDMatrixConvFinalize},
		"arithBlend":           {arithBlendStageFn, arithBlendStageSIMD, preferSIMDArithBlend},
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
