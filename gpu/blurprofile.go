// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The backend-neutral analytic-blur profile tables: the rect-blur integral table, the circle-blur / half-plane
// profiles, and the blurred-rrect nine-patch mask. These are uploaded as A8 textures and sampled from a runtime-effect
// shader so an axis-aligned rect, a circle, or a simple-circular rounded rect blurs analytically (no mask render +
// separable convolution).

package gpu

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// unitScalarClampToByte pins x to [0,1], then computes (U8)(x*255 + 0.5) — a truncation of the biased value, not a
// round-to-nearest.
func unitScalarClampToByte(x float32) uint8 {
	if x < 0 {
		x = 0
	} else if x > 1 {
		x = 1
	}
	return uint8(x*255 + 0.5)
}

// nextPow2 returns the smallest power of two >= value (>= 1 for value <= 1).
func nextPow2(value int) int {
	if value <= 1 {
		return 1
	}
	n := value - 1
	pow := 1
	for pow <= n {
		pow <<= 1
	}
	return pow
}

///////////////////////////////////////////////////////////////////////////////
//  Rect Blur

// CreateIntegralTable builds a width×1 A8 lookup table for the integral of the normal distribution. Entry i (mapped
// from [0,6*sigma] to [3*sigma,-3*sigma]) is the integral from -inf to (3*sigma - x). Returns nil when width <= 0.
func CreateIntegralTable(width int) []byte {
	if width <= 0 {
		return nil
	}
	table := make([]byte, width)
	table[0] = 255
	invWidth := 1.0 / float32(width)
	for i := 1; i < width-1; i++ {
		x := (float32(i) + 0.5) * invWidth
		x = (-6*x + 3) * geom.ScalarRoot2Over2
		integral := 0.5 * (float32(math.Erf(float64(x))) + 1)
		table[i] = byte(geom.RoundToInt(255 * integral))
	}
	table[width-1] = 0
	return table
}

// ComputeIntegralTableWidth returns the LUT width needed for CreateIntegralTable given 6*sigma. Returns 0 for a
// non-finite or overflowing sixSigma.
func ComputeIntegralTableWidth(sixSigma float32) int {
	if math.IsNaN(float64(sixSigma)) || math.IsInf(float64(sixSigma), 0) {
		return 0
	}
	// Avoid overflow, covers both multiplying by 2 and finding next power of 2.
	if sixSigma > float32(math.MaxInt32)/4+1 {
		return 0
	}
	// Two texels per dst pixel so the linear-interpolated texture lookup has no visible artifacts.
	minWidth := 2 * int(math.Ceil(float64(sixSigma)))
	// Bin by powers of 2 with a minimum so we get good profile reuse.
	if p := nextPow2(minWidth); p > 32 {
		return p
	}
	return 32
}

///////////////////////////////////////////////////////////////////////////////
//  Circle Blur

// makeUnnormalizedHalfKernel computes the right half of a Gaussian, sampled at half-pixel steps out from the center.
// Returns the sum of the (unnormalized) values.
func makeUnnormalizedHalfKernel(halfKernel []float32, halfKernelSize int, sigma float32) float32 {
	invSigma := 1.0 / sigma
	b := -0.5 * invSigma * invSigma
	var tot float32
	t := float32(0.5)
	for i := 0; i < halfKernelSize; i++ {
		value := float32(math.Exp(float64(t * t * b)))
		tot += value
		halfKernel[i] = value
		t++
	}
	return tot
}

// makeHalfKernelAndSummedTable computes a Gaussian half-kernel normalized to sum to 0.5 and its running-sum (summed
// area) table.
func makeHalfKernelAndSummedTable(halfKernel, summedHalfKernel []float32, halfKernelSize int, sigma float32) {
	tot := 2 * makeUnnormalizedHalfKernel(halfKernel, halfKernelSize, sigma)
	var sum float32
	for i := 0; i < halfKernelSize; i++ {
		halfKernel[i] /= tot
		sum += halfKernel[i]
		summedHalfKernel[i] = sum
	}
}

// applyKernelInY applies the 1D half-kernel vertically at points along the x axis to a circle centered at the origin
// with radius circleR.
func applyKernelInY(results []float32, numSteps int, firstX, circleR float32, halfKernelSize int, summedHalfKernelTable []float32) {
	x := firstX
	for i := 0; i < numSteps; i, x = i+1, x+1 {
		if x < -circleR || x > circleR {
			results[i] = 0
			continue
		}
		y := float32(math.Sqrt(float64(circleR*circleR - x*x)))
		// The summed table entry j reflects an offset of j + 0.5.
		y -= 0.5
		yInt := int(math.Floor(float64(y)))
		switch {
		case y < 0:
			results[i] = (y + 0.5) * summedHalfKernelTable[0]
		case yInt >= halfKernelSize-1:
			results[i] = 0.5
		default:
			yFrac := y - float32(yInt)
			results[i] = (1-yFrac)*summedHalfKernelTable[yInt] + yFrac*summedHalfKernelTable[yInt+1]
		}
	}
}

// evalAt applies a Gaussian at point (evalX, 0) to a circle of radius circleR, given a precomputed half-kernel and the
// y-evaluation table window starting at yKernelEvaluations.
func evalAt(evalX, circleR float32, halfKernel []float32, halfKernelSize int, yKernelEvaluations []float32) uint8 {
	var acc float32
	x := evalX - float32(halfKernelSize)
	for i := 0; i < halfKernelSize; i, x = i+1, x+1 {
		if x < -circleR || x > circleR {
			continue
		}
		acc += yKernelEvaluations[i] * halfKernel[halfKernelSize-i-1]
	}
	for i := 0; i < halfKernelSize; i, x = i+1, x+1 {
		if x < -circleR || x > circleR {
			continue
		}
		acc += yKernelEvaluations[i+halfKernelSize] * halfKernel[i]
	}
	// The half kernel was applied in y, so double (the circle is symmetric about the x axis).
	return unitScalarClampToByte(2 * acc)
}

// CreateCircleProfile builds a profileWidth×1 A8 profile of a blurred circle. Returns nil if the internal allocations
// would overflow.
func CreateCircleProfile(sigma, radius float32, profileWidth int) []byte {
	numSteps := profileWidth

	// The full kernel is 6 sigmas wide; round up to the next multiple of 2 then halve.
	halfKernelSize := int(math.Ceil(float64(6 * sigma)))
	halfKernelSize = ((halfKernelSize + 1) &^ 1) >> 1

	const kAllocLimit = math.MaxInt32 >> 2
	if numSteps > kAllocLimit || halfKernelSize > (kAllocLimit-numSteps) {
		return nil
	}

	// Number of x steps at which to apply the kernel in y to cover all the profile samples in x.
	numYSteps := numSteps + 2*halfKernelSize

	bulk := make([]float32, halfKernelSize+halfKernelSize+numYSteps)
	halfKernel := bulk[:halfKernelSize]
	summedKernel := bulk[halfKernelSize : 2*halfKernelSize]
	yEvals := bulk[2*halfKernelSize:]
	makeHalfKernelAndSummedTable(halfKernel, summedKernel, halfKernelSize, sigma)

	firstX := float32(-halfKernelSize) + 0.5
	applyKernelInY(yEvals, numYSteps, firstX, radius, halfKernelSize, summedKernel)

	profile := make([]byte, profileWidth)
	for i := 0; i < numSteps-1; i++ {
		evalX := float32(i) + 0.5
		profile[i] = evalAt(evalX, radius, halfKernel, halfKernelSize, yEvals[i:])
	}
	// Ensure the tail of the Gaussian goes to zero.
	profile[numSteps-1] = 0
	return profile
}

// CreateHalfPlaneProfile builds the profile for the extreme small-radius case, where a circle blur degenerates to
// convolving a Gaussian with a half-plane. profileWidth must be even.
func CreateHalfPlaneProfile(profileWidth int) []byte {
	if profileWidth&0x1 != 0 {
		return nil
	}
	profile := make([]byte, profileWidth)

	// The full kernel is 6 sigmas wide.
	sigma := float32(profileWidth) / 6
	halfKernelSize := profileWidth / 2

	halfKernel := make([]float32, halfKernelSize)

	// The half kernel should sum to 0.5.
	tot := 2 * makeUnnormalizedHalfKernel(halfKernel, halfKernelSize, sigma)
	var sum float32
	// Populate the profile from the right edge to the middle.
	for i := 0; i < halfKernelSize; i++ {
		halfKernel[halfKernelSize-i-1] /= tot
		sum += halfKernel[halfKernelSize-i-1]
		profile[profileWidth-i-1] = unitScalarClampToByte(sum)
	}
	// Populate the profile from the middle to the left edge (flip the half kernel, keep summing).
	for i := 0; i < halfKernelSize; i++ {
		sum += halfKernel[i]
		profile[halfKernelSize-i-1] = unitScalarClampToByte(sum)
	}
	// Ensure the tail of the Gaussian goes to zero.
	profile[profileWidth-1] = 0
	return profile
}

///////////////////////////////////////////////////////////////////////////////
//  RRect Blur

// ComputeBlurredRRectParams handles the uniform-radii rounded-rect subset: it derives the small 'rrectToDraw' to
// rasterize+blur into the nine-patch mask and the 'dimensions' of that mask, and reports whether the rrect is
// nine-patchable (the blur is small enough relative to the corner radii and the rrect's extent that the four blurred
// corners do not overlap in the middle). Placed here beside CreateRRectBlurMask, which consumes its output, because it
// is pure geometry. An overflow guard is omitted since reachable UI sizes cannot overflow int, and per-corner division
// arrays for the general (non-uniform-radii) case are not produced since they are unused here.
func ComputeBlurredRRectParams(srcRRect, devRRect geom.RRect, sigma, xformedSigma float32) (rrectToDraw geom.RRect, dimensions geom.ISize, ninePatchable bool) {
	// srcRRect and sigma feed only the (unused) source-space division arrays; kept for signature parity.
	_, _ = srcRRect, sigma
	devBlurRadius := 3 * int(geom.CeilToInt(xformedSigma-1.0/6.0))

	devOrig := devRRect.Rect
	// All four corners share (RadiusX, RadiusY) in the uniform subset, and devRRect is simple-circular here (RadiusX ≈
	// RadiusY), so the per-corner max()+ceil() collapse to one value per axis.
	devLeft := int(geom.CeilToInt(devRRect.RadiusX))
	devTop := int(geom.CeilToInt(devRRect.RadiusY))
	devRight := devLeft
	devBot := devTop

	// Conservative nine-patchability check: the blurred corners must not overlap in the middle.
	if devOrig.Left+float32(devLeft)+float32(devBlurRadius) >=
		devOrig.Right-float32(devRight)-float32(devBlurRadius) ||
		devOrig.Top+float32(devTop)+float32(devBlurRadius) >=
			devOrig.Bottom-float32(devBot)-float32(devBlurRadius) {
		return geom.RRect{}, geom.ISize{}, false
	}

	newRRWidth := devLeft + devRight + 2*devBlurRadius + 1
	newRRHeight := devTop + devBot + 2*devBlurRadius + 1
	dimensions = geom.ISize{
		Width:  int32(newRRWidth + 2*devBlurRadius),
		Height: int32(newRRHeight + 2*devBlurRadius),
	}

	newRect := geom.RectXYWH(float32(devBlurRadius), float32(devBlurRadius),
		float32(newRRWidth), float32(newRRHeight))
	rrectToDraw = geom.MakeRRect(newRect, geom.CeilToScalar(devRRect.RadiusX),
		geom.CeilToScalar(devRRect.RadiusY))
	return rrectToDraw, dimensions, true
}

// evalV evaluates the vertical blur at 'y' given 'top', the location of the top of the rrect for the column being
// sampled ('top' < 0 marks an empty column). 'integral' is the normal distribution integral table
// (CreateIntegralTable), sampled with linear interpolation.
func evalV(top float32, y int, integral []byte, integralSize int, sixSigma float32) uint8 {
	if top < 0 {
		return 0 // an empty column
	}
	fT := (top - float32(y) - 0.5) * (float32(integralSize) / sixSigma)
	if fT < 0 {
		return 255
	} else if fT >= float32(integralSize-1) {
		return 0
	}
	lower := int(fT)
	frac := fT - float32(lower)
	return uint8(float32(integral[lower])*(1.0-frac) + float32(integral[lower+1])*frac)
}

// evalH applies the gaussian 'kernel' horizontally at the (x, y) location, summing the vertical evaluations of the
// kernelSize columns centered on x.
func evalH(x, y int, topVec, kernel []float32, kernelSize int, integral []byte, integralSize int, sixSigma float32) uint8 {
	var accum float32
	xSampleLoc := x - kernelSize/2
	for i := 0; i < kernelSize; i++ {
		if xSampleLoc >= 0 && xSampleLoc < len(topVec) {
			accum += kernel[i] * float32(evalV(topVec[xSampleLoc], y, integral, integralSize, sixSigma))
		}
		xSampleLoc++
	}
	return uint8(accum + 0.5)
}

// CreateRRectBlurMask builds a blurred rounded-rectangle A8 mask. 'rrectToDraw' is the original (simple-circular) rrect
// centered within bounds defined by 'dimensions', which encompass the entire blurred rrect. The mask is computed for
// the upper-left quadrant and mirrored into the other three (the rrect is symmetric about both axes). Returns the A8
// pixels with rowBytes equal to dimensions.Width, or nil on failure. It is designed to closely match the equivalent GPU
// fill-in (draw rrect + Gaussian blur) so the two are interchangeable.
func CreateRRectBlurMask(rrectToDraw geom.RRect, dimensions geom.ISize, sigma float32) []byte {
	// BlurIsEffectivelyIdentity(sigma) is guaranteed false by the caller.
	radius := BlurSigmaRadius(sigma)
	kernelSize := int(BlurKernelWidth(radius))

	// RadiusX ≈ RadiusY for a simple-circular rrect, so the circular section math uses the single stored radius.
	radiiX := rrectToDraw.RadiusX

	width := int(dimensions.Width)
	height := int(dimensions.Height)
	halfWidthPlus1 := width/2 + 1
	halfHeightPlus1 := height/2 + 1

	kernel := make([]float32, kernelSize)
	Compute1DBlurKernel(sigma, radius, kernel)

	integral := CreateIntegralTable(ComputeIntegralTableWidth(6 * sigma))
	if len(integral) == 0 {
		return nil
	}
	integralSize := len(integral)
	sixSigma := 6 * sigma

	result := make([]byte, width*height)
	rect := rrectToDraw.Rect

	// topVec[x] is the y coordinate of the top edge of the rrect for column x (following the corner arc through the
	// circular section), or -1 for the empty columns outside the rect.
	topVec := make([]float32, width)
	for x := 0; x < width; x++ {
		fx := float32(x)
		switch {
		case fx < rect.Left || fx > rect.Right:
			topVec[x] = -1
		case fx+0.5 < rect.Left+radiiX: // in the circular section
			xDist := rect.Left + radiiX - fx - 0.5
			h := float32(math.Sqrt(float64(radiiX*radiiX - xDist*xDist)))
			topVec[x] = rect.Top + radiiX - h + 3*sigma
		default:
			topVec[x] = rect.Top + 3*sigma
		}
	}

	for y := 0; y < halfHeightPlus1; y++ {
		scanline := result[y*width : y*width+width]
		for x := 0; x < halfWidthPlus1; x++ {
			v := evalH(x, y, topVec, kernel, kernelSize, integral, integralSize, sixSigma)
			scanline[x] = v
			scanline[width-x-1] = v
		}
		// Mirror the finished row into the bottom half (a no-op copy for the middle row).
		copy(result[(height-y-1)*width:(height-y)*width], scanline)
	}
	return result
}
