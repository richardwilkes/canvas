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
