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
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// The fuzz harness (hostileFloats, quietFloat, randLanes, eqBits, eqLanes) lives in stage_equiv_test.go, shared with
// the goexperiment.simd equivalence suite.

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
