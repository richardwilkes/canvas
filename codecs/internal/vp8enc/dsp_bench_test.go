// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package vp8enc

import (
	"image"
	"math/rand/v2"
	"testing"
)

// The DSP benchmarks call the dispatch entry points, so one binary measures whichever kernel the build wired in: the
// portable form, or (under goexperiment.simd, on arm64 and amd64) the archsimd kernel. Compare builds with benchstat:
//
//	go test ./codecs/internal/vp8enc/ -run XXX -bench 'Bench' -count 10 >default.txt
//	GOEXPERIMENT=simd go test ./codecs/internal/vp8enc/ -run XXX -bench 'Bench' -count 10 >simd.txt
//	benchstat default.txt simd.txt
//
// The data is what the encoder actually hands these kernels: 4x4 and 16x16 sample blocks drawn from a photo-like
// source at the encoder's bps stride, and coefficient blocks that are the forward transform of those blocks, so the
// quantizer sees a realistic mix of zero, small and clipped levels rather than uniform noise.

// benchDSPBlocks is the number of distinct inputs each benchmark cycles through, enough that the caches see a working
// set rather than one hot line, and a power of two so the index wraps with a mask.
const benchDSPBlocks = 64

// benchDSPSamples returns a sample plane of blockCount 16x16 blocks at stride bps, filled with photo-like content.
func benchDSPSamples(seed uint64) []uint8 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	buf := make([]uint8, benchDSPBlocks*16*bps+16)
	for i := 0; i < len(buf)-16; i++ {
		// A slowly varying ramp with noise: neighboring samples correlate, as image samples do, so the transforms and
		// the quantizer see the coefficient distribution they would see in a real frame.
		buf[i] = uint8((i/7 + i/(3*bps) + rng.IntN(24)) & 0xFF)
	}
	return buf
}

// benchDSPCoeffs returns blockCount coefficient blocks: the forward transform of successive 4x4 sample blocks, which
// is exactly what quantizeBlock is handed.
func benchDSPCoeffs(seed uint64) [][16]int16 {
	src := benchDSPSamples(seed)
	ref := benchDSPSamples(seed + 100)
	out := make([][16]int16, benchDSPBlocks)
	for i := range out {
		fTransformGeneric(src[i*4*bps:], ref[i*4*bps:], out[i][:])
	}
	return out
}

func BenchmarkFTransform(b *testing.B) {
	src := benchDSPSamples(11)
	ref := benchDSPSamples(12)
	var out [16]int16
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		fTransform(src[(i&(benchDSPBlocks-1))*4*bps:], ref[(i&(benchDSPBlocks-1))*4*bps:], out[:])
		i++
	}
}

func BenchmarkITransformOne(b *testing.B) {
	ref := benchDSPSamples(13)
	dst := make([]uint8, 4*bps+16)
	coeffs := benchDSPCoeffs(14)
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		blk := coeffs[i&(benchDSPBlocks-1)]
		iTransformOne(ref[(i&(benchDSPBlocks-1))*4*bps:], blk[:], dst)
		i++
	}
}

func BenchmarkGetSSE16x16(b *testing.B) { benchGetSSE(b, 16, 16) }
func BenchmarkGetSSE16x8(b *testing.B)  { benchGetSSE(b, 16, 8) }
func BenchmarkGetSSE4x4(b *testing.B)   { benchGetSSE(b, 4, 4) }

func benchGetSSE(b *testing.B, w, h int) {
	x := benchDSPSamples(21)
	y := benchDSPSamples(22)
	total := int64(0)
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		total += getSSE(x[(i&(benchDSPBlocks-1))*16*bps:], y[(i&(benchDSPBlocks-1))*16*bps:], w, h)
		i++
	}
	if total == 0 {
		b.Fatal("no distortion measured")
	}
}

func BenchmarkQuantizeBlock(b *testing.B) {
	coeffs := benchDSPCoeffs(31)
	m := simdBenchMatrix()
	var in, out [16]int16
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		// The kernel writes the dequantized coefficients back over its input, so each iteration starts from a fresh
		// block; the 32-byte copy is charged to both lanes equally.
		in = coeffs[i&(benchDSPBlocks-1)]
		quantizeBlock(&in, &out, m)
		i++
	}
}

// isFlat is not a dispatch point (see the note on it in dsp.go), so these three measure the same scalar code in both
// builds. They are here because they are what settled that: the block counts are the three the encoder actually asks
// about, and an archsimd version of this lost at two of them.
func BenchmarkIsFlat1(b *testing.B)  { benchIsFlat(b, 1, flatnessLimitI4) }
func BenchmarkIsFlat8(b *testing.B)  { benchIsFlat(b, 8, flatnessLimitUV) }
func BenchmarkIsFlat16(b *testing.B) { benchIsFlat(b, 16, flatnessLimitI16) }

func benchIsFlat(b *testing.B, count, thresh int) {
	coeffs := benchDSPCoeffs(41)
	m := simdBenchMatrix()
	levels := make([][16]int16, benchDSPBlocks)
	for i := range levels {
		in := coeffs[i]
		quantizeBlockGeneric(&in, &levels[i], m)
	}
	flat := false
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		start := i & (benchDSPBlocks - 1)
		if start+count > benchDSPBlocks {
			start = benchDSPBlocks - count
		}
		flat = isFlat(levels[start:start+count], thresh)
		i++
	}
	_ = flat
}

// simdBenchMatrix is the luma-AC quantizer matrix at a mid quality, the plane and quality the encoder spends most of
// its quantizer time in.
func simdBenchMatrix() *matrix {
	var m matrix
	m.q[0] = dcTable[40]
	m.q[1] = acTable[40]
	m.expand(0)
	return &m
}

// BenchmarkEncodePhoto and its siblings measure the whole encoder, which is what the kernels are here to speed up.
func BenchmarkEncodePhoto(b *testing.B)    { benchEncode(b, photoLikeImage(512, 384, 17), 75) }
func BenchmarkEncodeGradient(b *testing.B) { benchEncode(b, gradientImage(512, 384), 75) }
func BenchmarkEncodeEdges(b *testing.B)    { benchEncode(b, sharpEdgesImage(512, 384), 75) }

func benchEncode(b *testing.B, img image.Image, quality float32) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Encode(img, quality); err != nil {
			b.Fatal(err)
		}
	}
}
