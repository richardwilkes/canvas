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
)

// The blit-row benchmarks drive the dispatch variables, so one binary measures whichever kernel the build wired in:
// the portable form, or (under goexperiment.simd) the archsimd kernel. Compare builds with benchstat. Rows are
// benchSpanPixels wide, the same width the span benchmarks use, and a multiple of four so the vector kernels run with
// an empty tail.

// benchBlitBytes returns a row of random coverage bytes, seeded deterministically so every run measures the same data.
func benchBlitBytes(seed uint64) []uint8 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	b := make([]uint8, benchSpanPixels)
	for i := range b {
		b[i] = uint8(rng.Uint32())
	}
	return b
}

// benchBlitMasks returns a row of random 565 subpixel coverage values.
func benchBlitMasks(seed uint64) []uint16 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	m := make([]uint16, benchSpanPixels)
	for i := range m {
		m[i] = uint16(rng.Uint32())
	}
	return m
}

func BenchmarkFillWords(b *testing.B) {
	dst := make([]uint32, benchSpanPixels)
	b.ReportAllocs()
	for b.Loop() {
		fillWordsFn(dst, 0xFF3F7FBF)
	}
}

func BenchmarkFillBytes(b *testing.B) {
	dst := make([]uint8, benchSpanPixels)
	b.ReportAllocs()
	for b.Loop() {
		fillBytesFn(dst, 0xFF)
	}
}

func BenchmarkColor32Row(b *testing.B) {
	dst := benchSpanWords(11)
	b.ReportAllocs()
	for b.Loop() {
		color32RowFn(dst, 0x80402010, alpha255To256(255-0x80))
	}
}

func BenchmarkBlitMaskTranslucentRow(b *testing.B) {
	dev := benchSpanWords(12)
	aa := benchBlitBytes(13)
	b.ReportAllocs()
	for b.Loop() {
		blitMaskTranslucentRowFn(dev, aa, 0x80402010, 0x80)
	}
}

func BenchmarkInterp256Row(b *testing.B) {
	src := benchSpanWords(14)
	dst := benchSpanWords(15)
	b.ReportAllocs()
	for b.Loop() {
		interp256RowFn(dst, src, 129)
	}
}

func BenchmarkPremulRow(b *testing.B) {
	src := benchSpanWords(16)
	dst := make([]uint32, benchSpanPixels)
	b.ReportAllocs()
	for b.Loop() {
		premulRowFn(dst, src)
	}
}

func BenchmarkPMBlendRow(b *testing.B) {
	src := benchSpanWords(17)
	dst := benchSpanWords(18)
	b.ReportAllocs()
	for b.Loop() {
		pmBlendRowFn(dst, src, 129)
	}
}

func BenchmarkBlitRowLCD16(b *testing.B) {
	dst := benchSpanWords(19)
	mask := benchBlitMasks(20)
	b.ReportAllocs()
	for b.Loop() {
		blitRowLCD16Fn(dst, mask, 0x80, 0x10, 0x20, 0x30)
	}
}

func BenchmarkBlitRowLCD16Opaque(b *testing.B) {
	dst := benchSpanWords(21)
	mask := benchBlitMasks(22)
	b.ReportAllocs()
	for b.Loop() {
		blitRowLCD16OpaqueFn(dst, mask, 0x10, 0x20, 0x30, 0xFF302010)
	}
}

func BenchmarkBlendRowLCD16Opaque(b *testing.B) {
	dst := benchSpanWords(23)
	src := benchSpanWords(24)
	mask := benchBlitMasks(25)
	b.ReportAllocs()
	for b.Loop() {
		blendRowLCD16OpaqueFn(dst, mask, src)
	}
}
