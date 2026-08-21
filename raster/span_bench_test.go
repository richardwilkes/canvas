// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"math/rand/v2"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// The span benchmarks drive the dispatch variables, so one binary measures whichever kernel the build wired in:
// portable, NEON, or (under goexperiment.simd) the archsimd kernels. Compare builds with benchstat.

// benchSpanPixels is a typical span width — wide enough that the per-call setup is amortized the way it is in a real
// blit, and a multiple of four so the vector kernels run with an empty tail.
const benchSpanPixels = 256

// benchSpanColors returns a span of plausible premultiplied shader output: in-range channels with alpha dominating, a
// few values just outside [0, 1] so the clamps are not uniformly no-ops.
func benchSpanColors() []colorcore.PMColor4f {
	buf := make([]colorcore.PMColor4f, benchSpanPixels)
	for i := range buf {
		f := float32(i) / float32(benchSpanPixels-1)
		buf[i] = colorcore.PMColor4f{R: f * 1.02, G: 0.5 - f*0.5, B: f * f, A: 1 - f*0.25}
	}
	buf[0].R, buf[1].G, buf[2].B = -0.5, 1.5, -0.001
	return buf
}

// benchSpanWords returns a span of random 8888 device words, seeded deterministically so every run measures the same
// data. The alpha bytes span the whole range, so the src-over kernel sees every scale.
func benchSpanWords(seed uint64) []uint32 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	words := make([]uint32, benchSpanPixels)
	for i := range words {
		words[i] = rng.Uint32()
	}
	return words
}

func BenchmarkClampSpan01(b *testing.B) {
	buf := benchSpanColors()
	b.ReportAllocs()
	for b.Loop() {
		clampSpan01Fn(buf)
	}
}

func BenchmarkStoreSpanSrc(b *testing.B) {
	buf := benchSpanColors()
	span := make([]uint32, benchSpanPixels)
	b.ReportAllocs()
	for b.Loop() {
		storeSpanSrcFn(buf, span)
	}
}

func BenchmarkPMSrcOverRow(b *testing.B) {
	src := benchSpanWords(1)
	dst := benchSpanWords(2)
	b.ReportAllocs()
	for b.Loop() {
		pmSrcOverRowFn(dst, src)
	}
}

func BenchmarkBlitMaskOpaqueRow(b *testing.B) {
	dev := benchSpanWords(3)
	aa := make([]uint8, benchSpanPixels)
	rng := rand.New(rand.NewPCG(4, 5))
	for i := range aa {
		aa[i] = uint8(rng.Uint32())
	}
	b.ReportAllocs()
	for b.Loop() {
		blitMaskOpaqueRowFn(dev, aa, 0xFF3F7FBF)
	}
}
