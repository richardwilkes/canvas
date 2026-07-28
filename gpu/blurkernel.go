// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The backend-neutral blur-kernel math: the sigma→radius relation, the 1D/2D Gaussian weight tables, and the
// linear-sampling kernel that folds pairs of Gaussian taps into bilinear samples so a radius-N blur needs only N+1
// texture reads. This feeds the GPU blur pipeline.

package gpu

import "math"

// MaxBlurSamples is the largest number of samples a blur effect will read (a 1D linear kernel spans radius+1 samples, a
// 2D kernel spans the full area).
const MaxBlurSamples = 28

// MaxLinearBlurSigma is the largest sigma handled by a single 1D linear-filtered pass; larger sigmas must be
// downscaled before such a pass runs. It sits well under the MaxBlurSamples ceiling: this sigma needs radius 12 and so
// a linear kernel width of 13, not the 28 samples a single pass could afford. Upstream has kept the same value since
// before the linear-filtering optimization cut the per-tap cost, and this library follows it.
const MaxLinearBlurSigma = 4.0

// BlurIsEffectivelyIdentity reports whether a blur of this sigma is small enough to be a no-op.
func BlurIsEffectivelyIdentity(sigma float32) bool { return sigma <= 0.03 }

// BlurSigmaRadius returns the kernel radius needed to capture a Gaussian blur of the given sigma.
func BlurSigmaRadius(sigma float32) int32 {
	if BlurIsEffectivelyIdentity(sigma) {
		return 0
	}
	return int32(math.Ceil(float64(3 * sigma)))
}

// BlurKernelWidth returns the full 2N+1 kernel span for a given radius.
func BlurKernelWidth(radius int32) int32 { return 2*radius + 1 }

// BlurLinearKernelWidth returns the folded sample count for a given radius.
func BlurLinearKernelWidth(radius int32) int32 { return radius + 1 }

// Compute2DBlurKernel fills kernel with normalized separable Gaussian weights over the [-radius, radius] box in each
// axis, row-major (y*width + x), with any trailing slots zeroed. Setting a denominator to 1 when a radius is 0
// collapses that axis to the 1D distribution (so this also backs Compute1DBlurKernel).
func Compute2DBlurKernel(sigmaW, sigmaH float32, radiusW, radiusH int32, kernel []float32) {
	width := BlurKernelWidth(radiusW)
	height := BlurKernelWidth(radiusH)
	kernelSize := int(width * height)

	twoSigmaSqrdX := 2 * sigmaW * sigmaW
	twoSigmaSqrdY := 2 * sigmaH * sigmaH

	// Setting the denominator to 1 when the radius is 0 automatically converts the remaining math to the 1D Gaussian
	// distribution. When both radii are 0, it correctly computes a weight of 1.0.
	sigmaXDenom := float32(1)
	if radiusW > 0 {
		sigmaXDenom = 1 / twoSigmaSqrdX
	}
	sigmaYDenom := float32(1)
	if radiusH > 0 {
		sigmaYDenom = 1 / twoSigmaSqrdY
	}

	sum := float32(0)
	for x := int32(0); x < width; x++ {
		xTerm := float32(x - radiusW)
		xTerm = xTerm * xTerm * sigmaXDenom
		for y := int32(0); y < height; y++ {
			yTerm := float32(y - radiusH)
			// The constant term of the Gaussian is dropped here since we renormalize below.
			xyTerm := float32(math.Exp(float64(-(xTerm + yTerm*yTerm*sigmaYDenom))))
			kernel[y*width+x] = xyTerm
			sum += xyTerm
		}
	}
	scale := 1 / sum
	for i := 0; i < kernelSize; i++ {
		kernel[i] *= scale
	}
	for i := kernelSize; i < len(kernel); i++ {
		kernel[i] = 0
	}
}

// Compute1DBlurKernel fills kernel with the 1D Gaussian weights (the 2D kernel with a zero second-axis radius). kernel
// must hold at least KernelWidth(radius) entries.
func Compute1DBlurKernel(sigma float32, radius int32, kernel []float32) {
	Compute2DBlurKernel(sigma, 0, radius, 0, kernel)
}

// Compute1DBlurLinearKernel folds the full 2N+1 Gaussian kernel into radius+1 bilinear taps and interleaves them into
// an array of (offset, weight, offset, weight) quads sized MaxBlurSamples/2, the uniform layout the 1D blur shader
// consumes. Trailing quads carry a zero weight (and the last valid offset) so a shader that over-iterates reads a
// harmless, cache-friendly sample. sigma must be <= MaxLinearBlurSigma.
func Compute1DBlurLinearKernel(sigma float32, radius int32) [MaxBlurSamples * 2]float32 {
	// Given 2 adjacent gaussian points blended as Wi*Ci + Wj*Cj, the GPU mixes them during a linear sample as Ci*(1-x)
	// + Cj*x. Solving for W', x such that W'*(Ci*(1-x)+Cj*x) == Wi*Ci + Wj*Cj:
	//   W' = Wi + Wj ; x = Wj / (Wi + Wj).
	getNewWeight := func(wi, wj float32) (newW, offset float32) {
		return wi + wj, wj / (wi + wj)
	}

	// Compute the full standard kernel, then fold pairs of taps into the upper half and mirror.
	fullKernel := make([]float32, BlurKernelWidth(radius))
	Compute1DBlurKernel(sigma, radius, fullKernel)

	var kernel [MaxBlurSamples]float32
	var offsets [MaxBlurSamples]float32
	// Note that halfSize isn't just size/2 but radius+1: the size of the output array.
	halfSize := BlurLinearKernelWidth(radius)
	halfRadius := halfSize / 2
	lowIndex := halfRadius - 1

	index := radius
	if radius&1 != 0 {
		// N is odd: use two samples, halving the influence of the center texel per sample.
		kernel[halfRadius], offsets[halfRadius] = getNewWeight(fullKernel[index]*0.5, fullKernel[index+1])
		kernel[lowIndex] = kernel[halfRadius]
		offsets[lowIndex] = -offsets[halfRadius]
		index++
		lowIndex--
	} else {
		// N is even: sample the center texel directly.
		kernel[halfRadius] = fullKernel[index]
		offsets[halfRadius] = 0
	}
	index++

	// Every other pair gets one sample; mirror to the lower half.
	for i := halfRadius + 1; i < halfSize; i, lowIndex, index = i+1, lowIndex-1, index+2 {
		kernel[i], offsets[i] = getNewWeight(fullKernel[index], fullKernel[index+1])
		offsets[i] += float32(index - radius)
		kernel[lowIndex] = kernel[i]
		offsets[lowIndex] = -offsets[i]
	}

	// Zero the remaining weights, but copy the last valid offset forward for cache-friendly reads.
	for i := halfSize; i < MaxBlurSamples; i++ {
		kernel[i] = 0
		offsets[i] = offsets[halfSize-1]
	}

	// Interleave into the output as {offset[2i], kernel[2i], offset[2i+1], kernel[2i+1]} quads.
	var out [MaxBlurSamples * 2]float32
	for i := 0; i < MaxBlurSamples/2; i++ {
		out[4*i+0] = offsets[2*i]
		out[4*i+1] = kernel[2*i]
		out[4*i+2] = offsets[2*i+1]
		out[4*i+3] = kernel[2*i+1]
	}
	return out
}

// Compute2DBlurOffsets fills offsets with the (x, y) sample offsets over the kernel box, packed as two offsets per
// quad. offsets must hold MaxBlurSamples*2 floats; trailing slots repeat the last valid offset.
func Compute2DBlurOffsets(radiusW, radiusH int32, offsets []float32) {
	kernelArea := int(BlurKernelWidth(radiusW) * BlurKernelWidth(radiusH))
	i := 0
	for y := -radiusH; y <= radiusH; y++ {
		for x := -radiusW; x <= radiusW; x++ {
			offsets[2*i] = float32(x)
			offsets[2*i+1] = float32(y)
			i++
		}
	}
	lastValid := 2 * (kernelArea - 1)
	for ; i < MaxBlurSamples; i++ {
		offsets[2*i] = offsets[lastValid]
		offsets[2*i+1] = offsets[lastValid+1]
	}
}

// BlurBatchedKernelWidth buckets a kernel width into one of a small number of groups (≤4, ≤8, ≤12, ≤16, ≤20, ≤28) so
// distinct blur programs are shared across nearby kernel widths rather than compiling one variant per radius.
func BlurBatchedKernelWidth(width int32) int32 {
	switch {
	case width <= 4:
		return 4
	case width <= 8:
		return 8
	case width <= 12:
		return 12
	case width <= 16:
		return 16
	case width <= 20:
		return 20
	default:
		return 28
	}
}
