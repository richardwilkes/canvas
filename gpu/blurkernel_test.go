// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gpu

import (
	"math"
	"testing"
)

func TestBlurSigmaRadius(t *testing.T) {
	cases := []struct {
		sigma float32
		want  int32
	}{
		{sigma: 0, want: 0},
		{sigma: 0.02, want: 0},  // <= 0.03 is effectively identity
		{sigma: 0.03, want: 0},  // boundary
		{sigma: 0.031, want: 1}, // ceil(0.093) = 1
		{sigma: 1, want: 3},     // ceil(3)
		{sigma: 2, want: 6},     // ceil(6)
		{sigma: 4, want: 12},    // ceil(12)
		{sigma: 1.1, want: 4},   // ceil(3.3)
	}
	for _, c := range cases {
		if got := BlurSigmaRadius(c.sigma); got != c.want {
			t.Errorf("BlurSigmaRadius(%v) = %d, want %d", c.sigma, got, c.want)
		}
	}
}

func TestBlurBatchedKernelWidth(t *testing.T) {
	// Verifies the kernel-width bucket boundaries.
	cases := map[int32]int32{
		2: 4, 3: 4, 4: 4,
		5: 8, 8: 8,
		9: 12, 12: 12,
		13: 16, 16: 16,
		17: 20, 20: 20,
		21: 28, 28: 28,
	}
	for in, want := range cases {
		if got := BlurBatchedKernelWidth(in); got != want {
			t.Errorf("BlurBatchedKernelWidth(%d) = %d, want %d", in, got, want)
		}
	}
}

func TestCompute1DBlurKernelNormalizedSymmetric(t *testing.T) {
	for _, sigma := range []float32{0.3, 0.6, 1, 1.5, 2, 3, 4} {
		radius := BlurSigmaRadius(sigma)
		width := BlurKernelWidth(radius)
		kernel := make([]float32, width)
		Compute1DBlurKernel(sigma, radius, kernel)

		var sum float32
		for _, w := range kernel {
			if w < 0 {
				t.Fatalf("sigma %v: negative weight %v", sigma, w)
			}
			sum += w
		}
		if math.Abs(float64(sum)-1) > 1e-5 {
			t.Errorf("sigma %v: kernel sum = %v, want 1", sigma, sum)
		}
		// Symmetric about the center.
		for i := int32(0); i < radius; i++ {
			if d := kernel[i] - kernel[width-1-i]; math.Abs(float64(d)) > 1e-6 {
				t.Errorf("sigma %v: kernel not symmetric at %d (%v vs %v)", sigma, i,
					kernel[i], kernel[width-1-i])
			}
		}
		// The center weight is the largest.
		for i := int32(0); i < width; i++ {
			if kernel[i] > kernel[radius]+1e-6 {
				t.Errorf("sigma %v: weight[%d]=%v exceeds center %v", sigma, i, kernel[i],
					kernel[radius])
			}
		}
	}
}

// TestCompute1DBlurLinearKernelReconstructsFull verifies the linear (bilinear-folded) kernel reconstructs the full
// discrete Gaussian kernel: a tap at center-relative offset o with weight w contributes, via bilinear interpolation,
// w*(1-frac) and w*frac to the two straddled full-kernel taps. By construction this must equal Compute1DBlurKernel
// exactly.
func TestCompute1DBlurLinearKernelReconstructsFull(t *testing.T) {
	for _, sigma := range []float32{0.05, 0.3, 0.6, 1, 1.5, 2, 3, 4} {
		radius := BlurSigmaRadius(sigma)
		if radius == 0 {
			continue
		}
		width := BlurKernelWidth(radius)
		full := make([]float32, width)
		Compute1DBlurKernel(sigma, radius, full)

		oak := Compute1DBlurLinearKernel(sigma, radius)
		recon := make([]float32, width)
		var wsum float32
		for j := 0; j < MaxBlurSamples; j++ {
			offset := oak[2*j]
			weight := oak[2*j+1]
			wsum += weight
			if weight == 0 {
				continue
			}
			pos := float64(radius) + float64(offset)
			base := int(math.Floor(pos))
			frac := float32(pos - float64(base))
			if base >= 0 && base < int(width) {
				recon[base] += weight * (1 - frac)
			}
			if base+1 >= 0 && base+1 < int(width) {
				recon[base+1] += weight * frac
			}
		}
		if math.Abs(float64(wsum)-1) > 1e-4 {
			t.Errorf("sigma %v: linear kernel weights sum to %v, want 1", sigma, wsum)
		}
		for i := int32(0); i < width; i++ {
			if d := recon[i] - full[i]; math.Abs(float64(d)) > 1e-4 {
				t.Errorf("sigma %v: reconstructed[%d]=%v != full[%d]=%v", sigma, i, recon[i], i,
					full[i])
			}
		}
	}
}

// TestCompute2DBlurKernelSeparable verifies the 2D kernel equals the outer product of the two normalized 1D kernels
// (the Gaussian is separable).
func TestCompute2DBlurKernelSeparable(t *testing.T) {
	// Only combinations with kernelArea <= MaxBlurSamples reach convolveGaussian2D.
	cases := []struct{ sx, sy float32 }{{sx: 0.6, sy: 0.6}, {sx: 0.6, sy: 0.4}, {sx: 0.3, sy: 0.6}, {sx: 0.3, sy: 0.3}}
	for _, c := range cases {
		rx := BlurSigmaRadius(c.sx)
		ry := BlurSigmaRadius(c.sy)
		w := BlurKernelWidth(rx)
		h := BlurKernelWidth(ry)
		kernel := make([]float32, MaxBlurSamples)
		Compute2DBlurKernel(c.sx, c.sy, rx, ry, kernel)

		kx := make([]float32, w)
		ky := make([]float32, h)
		Compute1DBlurKernel(c.sx, rx, kx)
		Compute1DBlurKernel(c.sy, ry, ky)

		var sum float32
		for y := int32(0); y < h; y++ {
			for x := int32(0); x < w; x++ {
				got := kernel[y*w+x]
				want := kx[x] * ky[y]
				if d := got - want; math.Abs(float64(d)) > 1e-5 {
					t.Errorf("sigma (%v,%v): kernel[%d,%d]=%v, want %v", c.sx, c.sy, x, y, got, want)
				}
				sum += got
			}
		}
		if math.Abs(float64(sum)-1) > 1e-5 {
			t.Errorf("sigma (%v,%v): 2D kernel sum = %v, want 1", c.sx, c.sy, sum)
		}
		// The area beyond kernelArea must be zeroed.
		for i := int(w * h); i < MaxBlurSamples; i++ {
			if kernel[i] != 0 {
				t.Errorf("sigma (%v,%v): kernel[%d] = %v, want 0 (trailing)", c.sx, c.sy, i, kernel[i])
			}
		}
	}
}

func TestCompute2DBlurOffsets(t *testing.T) {
	var offsets [MaxBlurSamples * 2]float32
	Compute2DBlurOffsets(1, 1, offsets[:]) // 3x3 kernel
	// The first 9 (x,y) pairs walk y=-1..1, x=-1..1.
	want := [][2]float32{
		{-1, -1},
		{0, -1},
		{1, -1},
		{-1, 0},
		{0, 0},
		{1, 0},
		{-1, 1},
		{0, 1},
		{1, 1},
	}
	for i, wxy := range want {
		if offsets[2*i] != wxy[0] || offsets[2*i+1] != wxy[1] {
			t.Errorf("offset[%d] = (%v,%v), want (%v,%v)", i, offsets[2*i], offsets[2*i+1],
				wxy[0], wxy[1])
		}
	}
	// Trailing offsets repeat the last valid pair (1,1).
	for i := 9; i < MaxBlurSamples; i++ {
		if offsets[2*i] != 1 || offsets[2*i+1] != 1 {
			t.Errorf("trailing offset[%d] = (%v,%v), want (1,1)", i, offsets[2*i], offsets[2*i+1])
		}
	}
}

// TestMaxLinearBlurSigmaFitsSampleBudget pins the relationship MaxLinearBlurSigma's doc states: the largest sigma a
// single linear-filtered 1D pass handles needs radius 12 and hence 13 samples, comfortably inside the MaxBlurSamples
// budget the shader's uniform array is sized for.
func TestMaxLinearBlurSigmaFitsSampleBudget(t *testing.T) {
	radius := BlurSigmaRadius(MaxLinearBlurSigma)
	if radius != 12 {
		t.Errorf("BlurSigmaRadius(%v) = %d, want 12", float32(MaxLinearBlurSigma), radius)
	}
	width := BlurLinearKernelWidth(radius)
	if width != 13 {
		t.Errorf("BlurLinearKernelWidth(%d) = %d, want 13", radius, width)
	}
	if width > MaxBlurSamples {
		t.Errorf("linear kernel width %d exceeds the %d-sample budget", width, MaxBlurSamples)
	}
}
