// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package filtercore

import (
	"math/rand/v2"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
)

// The blur-engine benchmarks drive the dispatch variables, so one binary measures whichever kernel the build wired in:
// the portable bodies, or (under goexperiment.simd on qualifying hardware) the archsimd kernels. Compare builds with
// benchstat.

// benchBlurRow is a realistic scanline length for a filtered layer.
const benchBlurRow = 256

// benchBlurWords returns a row of 8888 words shaped like the content these passes actually blur — a soft-edged
// translucent shape rather than noise — so a profile taken on the benchmark shows plausible values. Both kernels are
// branch-free over the sample, so the content cannot change the timing.
func benchBlurWords(n int) []uint32 {
	rng := rand.New(rand.NewPCG(5, 11))
	w := make([]uint32, n)
	for i := range w {
		a := uint32(255 * i / n)
		w[i] = a | (a/2+uint32(rng.IntN(8)))<<8 | (a/3)<<16 | a<<24
	}
	return w
}

// benchGaussianSegment measures one full-width Gaussian scanline at the given sigma. The three sigmas below cover the
// whole reachable window range: 3 taps just above the no-blur cutoff, 7 in the middle, and 13 just under the handover
// to the three-box pass.
func benchGaussianSegment(b *testing.B, sigma float32) {
	b.Helper()
	pass := makeGaussianPassMaker(sigma).makePass().(*gaussianPass)
	pass.startBlur()
	src := benchBlurWords(benchBlurRow)
	dst := make([]uint32, benchBlurRow)
	b.ReportAllocs()
	b.SetBytes(int64(benchBlurRow * 4))
	for b.Loop() {
		gaussianBlurSegmentFn(pass, benchBlurRow, src, 1, dst, 1)
	}
}

func BenchmarkGaussianSegmentW3(b *testing.B)  { benchGaussianSegment(b, 0.3) }
func BenchmarkGaussianSegmentW7(b *testing.B)  { benchGaussianSegment(b, 1) }
func BenchmarkGaussianSegmentW13(b *testing.B) { benchGaussianSegment(b, 1.9) }

// benchThreeBoxSegment measures one full-width three-box scanline. The pipeline's per-sample cost does not depend on
// the window — only the circular buffers' length does — so the two sigmas differ only in how much of the working set
// stays in L1.
func benchThreeBoxSegment(b *testing.B, sigma float32) {
	b.Helper()
	pass := makeThreeBoxPassMaker(sigma).makePass().(*threeBoxApproxPass)
	pass.startBlur()
	src := benchBlurWords(benchBlurRow)
	dst := make([]uint32, benchBlurRow)
	b.ReportAllocs()
	b.SetBytes(int64(benchBlurRow * 4))
	for b.Loop() {
		threeBoxBlurSegmentFn(pass, benchBlurRow, src, 1, dst, 1)
	}
}

func BenchmarkThreeBoxSegmentSmall(b *testing.B) { benchThreeBoxSegment(b, 4) }
func BenchmarkThreeBoxSegmentLarge(b *testing.B) { benchThreeBoxSegment(b, 40) }

// benchBlurPasses measures a whole two-axis blur of a 256x256 RGBA layer through the real driver, which is what an
// imagefilter blur scenario reaches: runPass' three-phase scanline walk, the inflated intermediate buffer and the
// in-place Y pass, plus the destination allocation each blur performs.
func benchBlurPasses(b *testing.B, sigma float32) {
	b.Helper()
	const size = 256
	src := imagecore.NewPixels(imagecore.MakeN32Premul(size, size))
	copy(src.Words, benchBlurWords(len(src.Words)))
	makeMaker := func(s float32) *passMaker {
		if maker := makeGaussianPassMaker(s); maker != nil {
			return maker
		}
		return makeThreeBoxPassMaker(s)
	}
	makerX, makerY := makeMaker(sigma), makeMaker(sigma)
	srcBounds := geom.IRect{Right: size, Bottom: size}
	dstBounds := srcBounds.Inset(-SigmaToRadius(sigma), -SigmaToRadius(sigma))
	b.ReportAllocs()
	b.SetBytes(int64(size * size * 4))
	for b.Loop() {
		if evalBlurPasses(makerX, makerY, src, srcBounds, dstBounds) == nil {
			b.Fatal("blur produced no image")
		}
	}
}

func BenchmarkBlurPassesGaussian(b *testing.B) { benchBlurPasses(b, 1.9) }
func BenchmarkBlurPassesThreeBox(b *testing.B) { benchBlurPasses(b, 8) }
