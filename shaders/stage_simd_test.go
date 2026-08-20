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
	"math/rand/v2"
	"reflect"
	"testing"
)

// TestStageSIMDMatchesScalar drives every simd stage kernel and its portable Go twin over identical randomized register
// files (hostile float classes included) and requires bitwise identity, exactly as TestStageNEONMatchesScalar does for
// the NEON kernels.
func TestStageSIMDMatchesScalar(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd kernels require; dispatch stays on the default forms")
	}
	rng := rand.New(rand.NewPCG(21, 22))

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

// TestStageSIMDWiring locks that on qualifying hardware the goexperiment.simd build's init actually repointed the
// dispatch variables at the simd kernels, so a refactor cannot silently fall back to the default forms.
func TestStageSIMDWiring(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd kernels require; dispatch stays on the default forms")
	}
	for name, pair := range map[string][2]stageFn{
		"matrix4x5": {matrix4x5StageFn, matrix4x5StageSIMD},
	} {
		if reflect.ValueOf(pair[0]).Pointer() != reflect.ValueOf(pair[1]).Pointer() {
			t.Fatalf("%s: dispatch fn is not the simd kernel", name)
		}
	}
}
