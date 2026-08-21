// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imagecore

import (
	"math/rand/v2"
	"testing"
)

// The conversion-row benchmarks drive the dispatch variables, so one binary measures whichever kernel the build wired
// in: the portable form, or (under goexperiment.simd) the archsimd kernel. Compare builds with benchstat. Rows are
// benchConvertPixels wide, a multiple of the sixteen-pixel vector chunk so every kernel runs with an empty tail. The
// two word kernels whose R/B exchange changes their inner loop are measured both ways.

// benchConvertPixels is the benchmark row width, wide enough to amortize per-call setup and to make the row, not the
// dispatch, the thing being measured.
const benchConvertPixels = 256

// benchConvertWords returns a row of random source words, seeded deterministically so every run measures the same data.
func benchConvertWords(seed uint64) []uint32 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	w := make([]uint32, benchConvertPixels)
	for i := range w {
		w[i] = rng.Uint32()
	}
	return w
}

// benchConvertBytes returns a row of random bytes, seeded deterministically.
func benchConvertBytes(seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	b := make([]byte, benchConvertPixels)
	for i := range b {
		b[i] = byte(rng.Uint32())
	}
	return b
}

func BenchmarkSwizzleWordRow(b *testing.B) {
	src := benchConvertWords(61)
	dst := make([]byte, 4*benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		swizzleWordRowFn(dst, src, false)
	}
}

func BenchmarkSwizzleWordRowSwap(b *testing.B) {
	src := benchConvertWords(61)
	dst := make([]byte, 4*benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		swizzleWordRowFn(dst, src, true)
	}
}

func BenchmarkPremulWordRow(b *testing.B) {
	src := benchConvertWords(62)
	dst := make([]byte, 4*benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		premulWordRowFn(dst, src, false)
	}
}

func BenchmarkPremulWordRowSwap(b *testing.B) {
	src := benchConvertWords(62)
	dst := make([]byte, 4*benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		premulWordRowFn(dst, src, true)
	}
}

func BenchmarkUnpremulWordRow(b *testing.B) {
	src := benchConvertWords(63)
	dst := make([]byte, 4*benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		unpremulWordRowFn(dst, src, false)
	}
}

func BenchmarkUnpremulWordRowSwap(b *testing.B) {
	src := benchConvertWords(63)
	dst := make([]byte, 4*benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		unpremulWordRowFn(dst, src, true)
	}
}

func BenchmarkAlphaFromWordsRow(b *testing.B) {
	src := benchConvertWords(64)
	dst := make([]byte, benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		alphaFromWordsRowFn(dst, src)
	}
}

func BenchmarkFillBytesRow(b *testing.B) {
	dst := make([]byte, benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		fillBytesRowFn(dst, 0xFF)
	}
}

func BenchmarkGrayToWordsRow(b *testing.B) {
	src := benchConvertBytes(65)
	dst := make([]byte, 4*benchConvertPixels)
	b.ReportAllocs()
	for b.Loop() {
		grayToWordsRowFn(dst, src)
	}
}
