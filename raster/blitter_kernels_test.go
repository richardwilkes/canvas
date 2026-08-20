// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Byte-exactness locks for the SWAR row kernels and the store-lane clamp elision.

package raster

import (
	"math"
	"math/rand/v2"
	"testing"
)

// refPMSrcOverRow is the pre-SWAR pmSrcOverRowGeneric: the per-channel scalar form over satAdd8 and mulDiv255Round32,
// retained as the differential reference.
func refPMSrcOverRow(dst, src []uint32) {
	for i, s := range src {
		d := dst[i]
		nalpha := 255 - (s >> 24)
		r := satAdd8(s&0xFF, mulDiv255Round32(d&0xFF, nalpha))
		g := satAdd8(s>>8&0xFF, mulDiv255Round32(d>>8&0xFF, nalpha))
		b := satAdd8(s>>16&0xFF, mulDiv255Round32(d>>16&0xFF, nalpha))
		a := satAdd8(s>>24, mulDiv255Round32(d>>24, nalpha))
		dst[i] = r | g<<8 | b<<16 | a<<24
	}
}

// TestPMSrcOverRowMatchesReference compares whichever pmSrcOverRow kernel the build wired (the SWAR form, the NEON
// kernel, or the simd kernel) with the scalar reference over random pixels plus structured extremes
// (transparent/opaque source, saturating channel sums), bit for bit.
func TestPMSrcOverRowMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const n = 1 << 16
	src := make([]uint32, n)
	dstGot := make([]uint32, n)
	dstWant := make([]uint32, n)
	for i := range src {
		src[i] = rng.Uint32()
		dstGot[i] = rng.Uint32()
	}
	// Structured extremes up front: alpha 0/255 sources over saturating destinations, per-channel boundary products for
	// the mulDiv255Round rounding.
	structured := []struct{ s, d uint32 }{
		{s: 0x00000000, d: 0xFFFFFFFF},
		{s: 0xFFFFFFFF, d: 0xFFFFFFFF},
		{s: 0x00FFFFFF, d: 0xFFFFFFFF}, // alpha 0, saturating additions
		{s: 0x80FF80FF, d: 0x7F017F01},
		{s: 0x01FFFFFF, d: 0xFFFFFFFF},
		{s: 0xFEFFFFFF, d: 0x01010101},
		{s: 0x7F808182, d: 0x83848586},
	}
	for i, sd := range structured {
		src[i] = sd.s
		dstGot[i] = sd.d
	}
	copy(dstWant, dstGot)
	pmSrcOverRowFn(dstGot, src)
	refPMSrcOverRow(dstWant, src)
	for i := range dstGot {
		if dstGot[i] != dstWant[i] {
			t.Fatalf("pmSrcOverRow: src %08x over dst -> %08x, want %08x", src[i], dstGot[i], dstWant[i])
		}
	}
}

// TestToUnormSubsumesClamp locks the equivalence the BlendSrc store lane relies on to skip shadeRow's clamp pass:
// toUnorm(clamp01f(v)) == toUnorm(v) for every float class the pipeline can emit — in-range values, out-of-range
// magnitudes, negative zero, infinities, NaN, denormals, and values that straddle the round-to-nearest-even boundaries.
func TestToUnormSubsumesClamp(t *testing.T) {
	fixed := []float32{
		0, float32(math.Copysign(0, -1)), 1, -1, 0.5, -0.5, 1.5, 2, 1000, -1000,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
		math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		math.MaxFloat32, -math.MaxFloat32,
		0.5 / 255, 1.5 / 255, 2.5 / 255, 253.5 / 255, 254.5 / 255, 0.99999994, 1.0000001,
	}
	for _, v := range fixed {
		if got, want := toUnorm(v), toUnorm(clamp01f(v)); got != want {
			t.Fatalf("toUnorm(%g) = %d but toUnorm(clamp01f) = %d", v, got, want)
		}
	}
	rng := rand.New(rand.NewPCG(7, 8))
	for range 1 << 20 {
		v := math.Float32frombits(rng.Uint32())
		if got, want := toUnorm(v), toUnorm(clamp01f(v)); got != want {
			t.Fatalf("toUnorm(%x=%g) = %d but toUnorm(clamp01f) = %d", math.Float32bits(v), v, got, want)
		}
	}
}
