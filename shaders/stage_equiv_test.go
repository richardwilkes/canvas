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

// eqLanes requires per-lane equality (per eqBits) of the full register file — all 16 lanes, since the vector kernels
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
