// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build arm64

package shaders

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// hostileFloats seeds lane values with every float class the pipeline can produce, so the NEON/scalar comparison
// exercises NaN propagation, infinities, signed zero, denormals, and rounding boundaries — bit-for-bit equality is
// required everywhere the scalar stage's output is defined.
var hostileFloats = []float32{
	0, float32(math.Copysign(0, -1)), 1, -1, 0.5, -0.5, 0.9999999, 1.0000001, 255, 256, -255,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 1e-38, -1e-38,
	3.4e38, -3.4e38, 1e-45, 0.1, 1.0 / 3.0, 2.0 / 3.0,
}

// quietFloat turns arbitrary random bits into a value the pipeline could actually hold: every stage input is the result
// of IEEE arithmetic (seed/matrix/FMA/tile math), which never emits a *signaling* NaN, so sNaN bit patterns are
// quieted. This matters because FMAXNM/FMINNM quiet an sNaN operand where Go's comparison-based minf/maxf would select
// the other operand — a divergence that is unreachable in production but constructible from raw bits.
func quietFloat(bits uint32) float32 {
	const expMask, quietBit = 0x7F800000, 0x00400000
	if bits&expMask == expMask && bits&0x007FFFFF != 0 {
		bits |= quietBit
	}
	return math.Float32frombits(bits)
}

// randLanes fills a lanes register file with a mix of hostile and random values for all 16 lanes.
func randLanes(rng *rand.Rand) lanes {
	var z lanes
	fill := func(dst *[stride]float32) {
		for i := range dst {
			if rng.IntN(3) == 0 {
				dst[i] = hostileFloats[rng.IntN(len(hostileFloats))]
			} else {
				dst[i] = quietFloat(rng.Uint32())
			}
		}
	}
	fill(&z.r)
	fill(&z.g)
	fill(&z.b)
	fill(&z.a)
	z.n = stride
	return z
}

// eqBits requires bit equality except between NaNs: FMLA and math.FMA may propagate a different operand's NaN *payload*
// when several operands are NaN, and payloads are unobservable downstream — every pipeline consumer (clamp01, toUnorm,
// the blend math) tests NaN-ness, never payload bits, so rendered bytes are identical either way.
func eqBits(g, w float32) bool {
	if math.Float32bits(g) == math.Float32bits(w) {
		return true
	}
	return math.IsNaN(float64(g)) && math.IsNaN(float64(w))
}

// eqLanes requires per-lane equality (per eqBits) of the full register file — all 16 lanes, since the NEON kernels
// write the scratch tail too and it must match the scalar loop run at n=16.
func eqLanes(t *testing.T, name string, got, want *lanes) {
	t.Helper()
	cmp := func(ch string, g, w *[stride]float32) {
		for i := range g {
			if !eqBits(g[i], w[i]) {
				t.Fatalf("%s: lane %s[%d] = %08x (%g), want %08x (%g)", name, ch, i,
					math.Float32bits(g[i]), g[i], math.Float32bits(w[i]), w[i])
			}
		}
	}
	cmp("r", &got.r, &want.r)
	cmp("g", &got.g, &want.g)
	cmp("b", &got.b, &want.b)
	cmp("a", &got.a, &want.a)
}

// TestStageNEONMatchesScalar drives every NEON stage kernel and its portable Go twin over identical randomized register
// files (hostile float classes included) and requires bitwise identity.
func TestStageNEONMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))

	t.Run("seed", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			z1.dx = int32(rng.Uint32())
			z1.dy = int32(rng.Uint32())
			z2 := z1
			seedStage(&z1)
			seedStageNEON(&z2)
			eqLanes(t, "seed", &z2, &z1)
		}
	})

	t.Run("clampX1", func(t *testing.T) {
		for range 1024 {
			z1 := randLanes(rng)
			z2 := z1
			clampX1Stage(&z1)
			clampX1StageNEON(&z2)
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
			matrixTranslateStageNEON(&z2)
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
			matrixScaleTranslateStageNEON(&z2)
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
			matrixAffineStageNEON(&z2)
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
			gradient2StopStageNEON(&z2)
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
			// in-domain values and leave hostile garbage in the scratch tail, which the NEON kernel must tolerate
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
			gradientEvenlyStageNEON(&z2)
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
}

// TestStageNEONWiring locks that the arm64 dispatch variables actually point at the NEON wrappers, so a refactor cannot
// silently fall back to the scalar stages.
func TestStageNEONWiring(t *testing.T) {
	for name, pair := range map[string][2]stageFn{
		"seed":                 {seedStageFn, seedStageNEON},
		"clampX1":              {clampX1StageFn, clampX1StageNEON},
		"matrixTranslate":      {matrixTranslateStageFn, matrixTranslateStageNEON},
		"matrixScaleTranslate": {matrixScaleTranslateStageFn, matrixScaleTranslateStageNEON},
		"matrixAffine":         {matrixAffineStageFn, matrixAffineStageNEON},
		"gradient2Stop":        {gradient2StopStageFn, gradient2StopStageNEON},
		"gradientEvenly":       {gradientEvenlyStageFn, gradientEvenlyStageNEON},
	} {
		if reflect.ValueOf(pair[0]).Pointer() != reflect.ValueOf(pair[1]).Pointer() {
			t.Fatalf("%s: dispatch fn is not the NEON wrapper", name)
		}
	}
}
