// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package maskfilter

import (
	"bytes"
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// smallBlurSigmaLo and smallBlurSigmaHi bracket the sigmas that actually reach the small-sigma direct convolution:
// boxBlur rejects anything at or below the 1/3 no-window cutoff, and blur() hands anything from 2 up to the sliding-box
// planGauss engine instead. Over that half-open range newGaussFilter produces radius 1 through 4 (roughly evenly
// split), which is exactly the set of unrolled kernels.
const (
	smallBlurSigmaLo = 1.0 / 3.0
	smallBlurSigmaHi = 2.0
)

// randGauss returns a radius and factor set for one axis. Half the cases use factors a real sigma produces, so the
// fuzz spends real time on the arithmetic the renderer performs; the other half uses arbitrary uint16 factors, which
// drive the 8.8 accumulators far past the ~65408 a normalized kernel can reach and so exercise 16-bit wraparound in
// both twins at once. The radius stays in 1..4 either way: that is what the reachable sigmas produce, and the vertical
// path does not accept anything else — blurYRadius indexes regs[2*radius-1], so a radius of 0 panics in the portable
// code long before any kernel sees it.
func randGauss(rng *rand.Rand) (int, [5]uint16) {
	var factors [5]uint16
	if rng.IntN(2) == 0 {
		f := newGaussFilter(smallBlurSigmaLo + rng.Float64()*(smallBlurSigmaHi-smallBlurSigmaLo))
		for i := 0; i < f.n; i++ {
			factors[i] = uint16(math.Round(f.basis[i] * (1 << 16)))
		}
		return f.radius(), factors
	}
	radius := 1 + rng.IntN(4)
	for i := range factors {
		factors[i] = uint16(rng.Uint32())
	}
	return radius, factors
}

// randBytes returns n bytes biased toward the extremes an A8 mask is full of (0 and 255) but covering everything in
// between, since the fixed-point truncations are most fragile at the ends of the range.
func randBytes(rng *rand.Rand, n int) []uint8 {
	b := make([]uint8, n)
	for i := range b {
		switch rng.IntN(4) {
		case 0:
			b[i] = 0
		case 1:
			b[i] = 255
		default:
			b[i] = uint8(rng.Uint32())
		}
	}
	return b
}

// TestMaskBlurSIMDMatchesScalar drives each simd kernel and its portable fp88 twin over identical rects and requires
// byte-identical output buffers — not merely equal pixels inside the blurred region, but equal whole buffers, so a
// kernel that wrote one byte too far or skipped an edge phase fails. The geometry mirrors smallBlur's exactly,
// including the radiusX offset into the destination and the horizontal pass running in place, and the widths and
// heights deliberately straddle multiples of 8 so every tail phase is hit.
func TestMaskBlurSIMDMatchesScalar(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd kernels require; dispatch stays on the portable forms")
	}
	rng := rand.New(rand.NewPCG(0x6d61736b, 0x626c7572))

	t.Run("directBlurY", func(t *testing.T) {
		for range 1024 {
			srcW, srcH := 1+rng.IntN(67), 1+rng.IntN(67)
			srcRB := srcW + rng.IntN(3) // masks may carry row padding
			src := randBytes(rng, srcH*srcRB)
			radiusY, gaussY := randGauss(rng)
			radiusX := rng.IntN(5)
			dstW, dstH := srcW+2*radiusX, srcH+2*radiusY
			// Pre-fill both destinations identically with garbage: bytes neither pass is supposed to touch must come
			// back unchanged in both, so a stray write is a mismatch rather than a coincidence.
			want := randBytes(rng, dstW*dstH)
			got := bytes.Clone(want)
			directBlurY(radiusY, &gaussY, src, srcRB, srcW, srcH, want[radiusX:], dstW)
			directBlurYSIMD(radiusY, &gaussY, src, srcRB, srcW, srcH, got[radiusX:], dstW)
			if !bytes.Equal(got, want) {
				t.Fatalf("directBlurY mismatch: radius=%d srcW=%d srcH=%d srcRB=%d radiusX=%d gauss=%v\n got %v\nwant %v",
					radiusY, srcW, srcH, srcRB, radiusX, gaussY, got, want)
			}
		}
	})

	t.Run("directBlurX", func(t *testing.T) {
		for i := range 1024 {
			srcW, srcH := 1+rng.IntN(67), 1+rng.IntN(67)
			radiusX, gaussX := randGauss(rng)
			if i%64 == 0 {
				// The horizontal path, unlike the vertical one, is well defined at radius 0, so a slice of the cases
				// covers the kernels' "no unrolled form for this radius, run the portable row whole" guard.
				radiusX = 0
			}
			radiusY := rng.IntN(5)
			dstW, dstH := srcW+2*radiusX, srcH+2*radiusY
			// smallBlur runs the horizontal pass in place over the vertical pass's output, reading the rect inset by
			// radiusX and writing the full width, so both twins get the same aliased buffer.
			want := randBytes(rng, dstW*dstH)
			got := bytes.Clone(want)
			directBlurX(radiusX, &gaussX, want[radiusX:], dstW, srcW, want, dstW, dstW, dstH)
			directBlurXSIMD(radiusX, &gaussX, got[radiusX:], dstW, srcW, got, dstW, dstW, dstH)
			if !bytes.Equal(got, want) {
				t.Fatalf("directBlurX mismatch: radius=%d srcW=%d dstW=%d dstH=%d gauss=%v\n got %v\nwant %v",
					radiusX, srcW, dstW, dstH, gaussX, got, want)
			}
		}
	})

	t.Run("smallBlur", func(t *testing.T) {
		// End to end through the dispatch variables, at the sigmas and geometry the renderer actually produces: the
		// same source blurred once with both drivers forced to the portable forms and once with them at the kernels.
		origX, origY := directBlurXFn, directBlurYFn
		t.Cleanup(func() { directBlurXFn, directBlurYFn = origX, origY })
		for range 1024 {
			srcW, srcH := 1+rng.IntN(67), 1+rng.IntN(67)
			src := &raster.Mask{
				Image:    randBytes(rng, srcW*srcH),
				Bounds:   geom.IRectXYWH(int32(rng.IntN(21)-10), int32(rng.IntN(21)-10), int32(srcW), int32(srcH)),
				RowBytes: int32(srcW),
			}
			sigmaX := smallBlurSigmaLo + rng.Float64()*(smallBlurSigmaHi-smallBlurSigmaLo)
			sigmaY := smallBlurSigmaLo + rng.Float64()*(smallBlurSigmaHi-smallBlurSigmaLo)
			var wantDst, gotDst raster.Mask
			directBlurXFn, directBlurYFn = directBlurX, directBlurY
			wantBorder := smallBlur(sigmaX, sigmaY, src, &wantDst)
			directBlurXFn, directBlurYFn = directBlurXSIMD, directBlurYSIMD
			gotBorder := smallBlur(sigmaX, sigmaY, src, &gotDst)
			if gotBorder != wantBorder || gotDst.Bounds != wantDst.Bounds || gotDst.RowBytes != wantDst.RowBytes ||
				!bytes.Equal(gotDst.Image, wantDst.Image) {
				t.Fatalf("smallBlur mismatch: sigma=(%v,%v) srcW=%d srcH=%d\n got %+v\nwant %+v",
					sigmaX, sigmaY, srcW, srcH, gotDst, wantDst)
			}
		}
	})
}

// TestMaskBlurSIMDWiring locks that on qualifying hardware the goexperiment.simd build's init actually repointed the
// dispatch variables at the simd kernels, so a refactor cannot silently fall back to the portable forms.
func TestMaskBlurSIMDWiring(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd kernels require; dispatch stays on the portable forms")
	}
	if reflect.ValueOf(directBlurXFn).Pointer() != reflect.ValueOf(directBlurXSIMD).Pointer() {
		t.Error("directBlurXFn is not the simd kernel")
	}
	if reflect.ValueOf(directBlurYFn).Pointer() != reflect.ValueOf(directBlurYSIMD).Pointer() {
		t.Error("directBlurYFn is not the simd kernel")
	}
}
