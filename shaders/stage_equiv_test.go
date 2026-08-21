// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Built exactly when a consumer exists: the NEON equivalence suite (arm64) and the simd equivalence suite
// (goexperiment.simd). On any other build shape every helper here would be dead code.
//go:build arm64 || goexperiment.simd

package shaders

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The shared fuzz harness for the kernel-equivalence suites: the NEON comparison (stage_arm64_test.go) and the
// goexperiment.simd comparison (stage_simd_test.go) both drive vector kernels and their portable scalar twins over
// identical randomized register files and require bitwise identity.

// hostileFloats seeds lane values with every float class the pipeline can produce, so the vector/scalar comparison
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

// randLanes fills a lanes register file with a mix of hostile and random values for all 16 lanes. The dst registers are
// seeded too: the image-sampler kernels (accumulate, move_dst_src) read them, and for the kernels that do not they are
// simply carried through both sides of the comparison unchanged.
func randLanes(rng *rand.Rand) lanes {
	var z lanes
	for _, reg := range []*[stride]float32{&z.r, &z.g, &z.b, &z.a, &z.dr, &z.dg, &z.db, &z.da} {
		randFloats(rng, reg)
	}
	z.n = stride
	return z
}

// randFloats fills one 16-lane register or scratch array with the same hostile/random mix randLanes uses.
func randFloats(rng *rand.Rand, dst *[stride]float32) {
	for i := range dst {
		if rng.IntN(3) == 0 {
			dst[i] = hostileFloats[rng.IntN(len(hostileFloats))]
		} else {
			dst[i] = quietFloat(rng.Uint32())
		}
	}
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

// eqLanes requires per-lane equality (per eqBits) of the src register file — all 16 lanes, since the vector kernels
// write the scratch tail too and it must match the scalar loop run at n=16.
func eqLanes(t *testing.T, name string, got, want *lanes) {
	t.Helper()
	eqReg(t, name, "r", &got.r, &want.r)
	eqReg(t, name, "g", &got.g, &want.g)
	eqReg(t, name, "b", &got.b, &want.b)
	eqReg(t, name, "a", &got.a, &want.a)
}

// eqReg requires per-lane equality (per eqBits) of one 16-lane register or scratch array.
func eqReg(t *testing.T, name, ch string, got, want *[stride]float32) {
	t.Helper()
	for i := range got {
		if !eqBits(got[i], want[i]) {
			t.Fatalf("%s: lane %s[%d] = %08x (%g), want %08x (%g)", name, ch, i,
				math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}
