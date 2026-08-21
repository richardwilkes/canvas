// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package maskfilter

import (
	"math"
	"testing"
)

// The small-sigma blur benchmarks drive the dispatch variables, so one binary measures whichever driver the build
// wired in: the portable fp88 forms, or (under goexperiment.simd on qualifying hardware) the archsimd kernels. Compare
// builds with benchstat.

// benchBlurSigma is a typical drop-shadow sigma. It sits in the upper half of the small-sigma domain and yields
// radius 3 — the middle of the 1..4 the unrolled kernels cover — so neither the cheapest nor the most expensive
// unrolled form is what gets measured.
const benchBlurSigma = 1.5

// benchBlurGauss returns the radius and 0.16 fixed-point factors for a sigma, exactly as smallBlur derives them.
func benchBlurGauss(sigma float64) (radius int, factors [5]uint16) {
	f := newGaussFilter(sigma)
	for i := 0; i < f.n; i++ {
		factors[i] = uint16(math.Round(f.basis[i] * (1 << 16)))
	}
	return f.radius(), factors
}

// benchBlurMask returns a 64x64 A8 mask shaped like the shadow casters this path actually blurs: a solid core with a
// soft, ragged border. The kernels are branch-free integer code, so the content cannot change the timing — it exists
// so a profile taken on the benchmark shows plausible values.
func benchBlurMask(w, h int) []uint8 {
	m := make([]uint8, w*h)
	for y := range h {
		for x := range w {
			v := 255
			if d := min(min(x, w-1-x), min(y, h-1-y)); d < 3 {
				v = 32 + 74*d
			}
			m[y*w+x] = uint8(v)
		}
	}
	return m
}

// BenchmarkDirectBlurY measures the vertical pass over a 64x64 mask, with the same geometry smallBlur sets up: the
// destination is the mask outset by both radii and the pass writes into it at the horizontal radius offset.
func BenchmarkDirectBlurY(b *testing.B) {
	const srcW, srcH = 64, 64
	radius, gauss := benchBlurGauss(benchBlurSigma)
	src := benchBlurMask(srcW, srcH)
	dstW, dstH := srcW+2*radius, srcH+2*radius
	dst := make([]uint8, dstW*dstH)
	b.ReportAllocs()
	b.SetBytes(int64(srcW * srcH))
	for b.Loop() {
		directBlurYFn(radius, &gauss, src, srcW, srcW, srcH, dst[radius:], dstW)
	}
}

// BenchmarkDirectBlurX measures the horizontal pass over the same rect. smallBlur runs it in place over the vertical
// pass's output; here the source and destination are distinct buffers of identical shape so every iteration reads the
// same bytes — the arithmetic, the strides and the memory traffic are the same either way.
func BenchmarkDirectBlurX(b *testing.B) {
	const srcW, srcH = 64, 64
	radius, gauss := benchBlurGauss(benchBlurSigma)
	dstW, dstH := srcW+2*radius, srcH+2*radius
	src := benchBlurMask(dstW, dstH) // the vertical pass's output: the outset rect, already soft top and bottom
	dst := make([]uint8, dstW*dstH)
	b.ReportAllocs()
	b.SetBytes(int64(srcW * dstH))
	for b.Loop() {
		directBlurXFn(radius, &gauss, src[radius:], dstW, srcW, dst, dstW, dstW, dstH)
	}
}
