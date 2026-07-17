// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Wang's formula gives the minimum number of evenly spaced (in the parametric sense) line segments that a bezier curve
// must be chopped into in order to guarantee all lines stay within a distance of "1/precision" pixels from the true
// curve. Its definition for a bezier curve of degree "n" is as follows:
//
//	maxLength = max([length(p[i+2] - 2p[i+1] + p[i]) for (0 <= i <= n-2)])
//	numParametricSegments = sqrt(maxLength * precision * n*(n - 1)/8)
//
// (Goldman, Ron. (2003). 5.6.3 Wang's Formula. "Pyramid Algorithms: A Dynamic Programming Approach to Curves and
// Surfaces for Geometric Modeling". Morgan Kaufmann Publishers.)

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// min32/max32 return the first operand when the comparison against a NaN operand is false, so NaN never silently wins.
// sqrt32 is a float32-rounded sqrt.
func min32(a, b float32) float32 {
	if b < a {
		return b
	}
	return a
}

func max32(a, b float32) float32 {
	if a < b {
		return b
	}
	return a
}

func sqrt32(x float32) float32 { return float32(math.Sqrt(float64(x))) }

// wangsLengthTermP2 returns the square of the value by which to multiply length in Wang's formula for a curve of the
// given degree.
func wangsLengthTermP2(degree int, precision float32) float32 {
	return (float32(degree*degree*(degree-1)*(degree-1)) / 64) * (precision * precision)
}

// wangsRoot4 returns sqrt(sqrt(x)).
func wangsRoot4(x float32) float32 {
	return float32(math.Sqrt(math.Sqrt(float64(x))))
}

// wangsNextLog2 returns, for finite positive x > 1, ceil(log2(x)); otherwise returns 0. For +/- NaN returns 0. For
// +infinity returns 128. For -infinity returns 0.
//
//	nextlog2((-inf..1]) -> 0
//	nextlog2((1..2]) -> 1
//	nextlog2((2..4]) -> 2
//	nextlog2((4..8]) -> 3
//	...
func wangsNextLog2(x float32) int {
	if x <= 1 {
		return 0
	}
	bits := math.Float32bits(x)
	const digitsAfterBinaryPoint = 23
	// The constant is a significand of all 1s -- 0b0'00000000'111'1111111111'111111111. So, if the significand of x is
	// all 0s (and therefore an integer power of two) this will not increment the exponent, but if it is just one ULP
	// above the power of two the carry will ripple into the exponent incrementing the exponent by 1.
	bits += (1 << digitsAfterBinaryPoint) - 1
	// Shift the exponent down, and adjust it by the exponent offset so that 2^0 is really 0 instead of 127. If x is NaN
	// then the addition above can carry a 1 into the sign bit, which the mask strips off. Infinity's all-1s exponent
	// with an all-0s significand is unchanged by the addition, so it stays all 1s (128 after the offset).
	exp := int((bits>>digitsAfterBinaryPoint)&0xff) - 127
	if exp > 0 {
		return exp
	}
	return 0
}

// wangsNextLog16 returns nextlog2(sqrt(sqrt(x))), computed as (wangsNextLog2(x) + 3) / 4.
func wangsNextLog16(x float32) int {
	return (wangsNextLog2(x) + 3) >> 2
}

// vectorXform is the upper-left 2x2 matrix of an affine transform, for applying to vectors (direction, not position):
//
//	vectorXform(p1 - p0) == M * float3(p1, 1) - M * float3(p0, 1)
type vectorXform struct {
	// First (c0x, c0y) and second (c1x, c1y) columns of the 2x2 matrix.
	c0x, c0y float32
	c1x, c1y float32
}

// identityVectorXform returns the identity transform.
func identityVectorXform() vectorXform {
	return vectorXform{c0x: 1, c0y: 0, c1x: 0, c1y: 1}
}

// makeVectorXform extracts the upper-left 2x2 (vector-transforming) portion of an affine matrix. The matrix must not
// have perspective.
func makeVectorXform(m *geom.Matrix) vectorXform {
	if m.HasPerspective() {
		panic("VectorXform requires a non-perspective matrix")
	}
	return vectorXform{
		c0x: m.Get(geom.MScaleX), c0y: m.Get(geom.MSkewY),
		c1x: m.Get(geom.MSkewX), c1y: m.Get(geom.MScaleY),
	}
}

// apply transforms v as a vector (direction only, no translation) by the 2x2 matrix.
func (x vectorXform) apply(v geom.Point) geom.Point {
	return geom.Point{X: x.c0x*v.X + x.c1x*v.Y, Y: x.c0y*v.X + x.c1y*v.Y}
}

// wangsQuadraticP4 returns Wang's formula raised to the 4th power, specialized for a quadratic curve.
func wangsQuadraticP4(precision float32, p0, p1, p2 geom.Point, xf vectorXform) float32 {
	v := geom.Point{X: -2*p1.X + p0.X + p2.X, Y: -2*p1.Y + p0.Y + p2.Y}
	v = xf.apply(v)
	return (v.X*v.X + v.Y*v.Y) * wangsLengthTermP2(2, precision)
}

// wangsQuadratic returns Wang's formula for a quadratic curve: the number of line segments needed.
func wangsQuadratic(precision float32, p0, p1, p2 geom.Point, xf vectorXform) float32 {
	return wangsRoot4(wangsQuadraticP4(precision, p0, p1, p2, xf))
}

// wangsCubicP4 returns Wang's formula raised to the 4th power, specialized for a cubic curve.
func wangsCubicP4(precision float32, p0, p1, p2, p3 geom.Point, xf vectorXform) float32 {
	v0 := xf.apply(geom.Point{X: -2*p1.X + p0.X + p2.X, Y: -2*p1.Y + p0.Y + p2.Y})
	v1 := xf.apply(geom.Point{X: -2*p2.X + p1.X + p3.X, Y: -2*p2.Y + p1.Y + p3.Y})
	m := v0.X*v0.X + v0.Y*v0.Y
	if n := v1.X*v1.X + v1.Y*v1.Y; m < n {
		m = n // Keep the larger of the two magnitudes.
	}
	return m * wangsLengthTermP2(3, precision)
}

// wangsCubic returns Wang's formula for a cubic curve: the number of line segments needed.
func wangsCubic(precision float32, p0, p1, p2, p3 geom.Point, xf vectorXform) float32 {
	return wangsRoot4(wangsCubicP4(precision, p0, p1, p2, p3, xf))
}

// wangsCubicLog2 returns log2 of Wang's formula for a cubic, rounded up.
func wangsCubicLog2(precision float32, p0, p1, p2, p3 geom.Point, xf vectorXform) int {
	return wangsNextLog16(wangsCubicP4(precision, p0, p1, p2, p3, xf))
}

// wangsWorstCaseCubicP4 returns the maximum number of line segments (raised to the 4th power) that a cubic with the
// given device-space bounding box size would ever need to be divided into.
func wangsWorstCaseCubicP4(precision, devWidth, devHeight float32) float32 {
	kk := wangsLengthTermP2(3, precision)
	return 4 * kk * (devWidth*devWidth + devHeight*devHeight)
}

// wangsWorstCaseCubic returns the worst-case number of line segments for a cubic with the given device-space bounding
// box size.
func wangsWorstCaseCubic(precision, devWidth, devHeight float32) float32 {
	return wangsRoot4(wangsWorstCaseCubicP4(precision, devWidth, devHeight))
}

// wangsConicP2 computes Wang's formula specialized for a conic curve, raised to the second power. Input points should
// be in projected space. This is not actually due to Wang, but is an analog from (Theorem 3, corollary 1): J. Zheng,
// T. Sederberg. "Estimating Tessellation Parameter Intervals for Rational Curves and Surfaces." ACM Transactions on
// Graphics 19(1). 2000.
func wangsConicP2(precision float32, p0, p1, p2 geom.Point, w float32, xf vectorXform) float32 {
	p0 = xf.apply(p0)
	p1 = xf.apply(p1)
	p2 = xf.apply(p2)

	// Compute the center of the bounding box in projected space.
	cx := 0.5 * (min32(min32(p0.X, p1.X), p2.X) + max32(max32(p0.X, p1.X), p2.X))
	cy := 0.5 * (min32(min32(p0.Y, p1.Y), p2.Y) + max32(max32(p0.Y, p1.Y), p2.Y))

	// Translate by -C. This improves translation-invariance of the formula, see Sec. 3.3 of the cited paper.
	p0.X -= cx
	p0.Y -= cy
	p1.X -= cx
	p1.Y -= cy
	p2.X -= cx
	p2.Y -= cy

	// Compute max length.
	maxLen := sqrt32(max32(p0.X*p0.X+p0.Y*p0.Y, max32(p1.X*p1.X+p1.Y*p1.Y, p2.X*p2.X+p2.Y*p2.Y)))

	// Compute forward differences.
	dpx := -2*w*p1.X + p0.X + p2.X
	dpy := -2*w*p1.Y + p0.Y + p2.Y
	dw := abs32(-2*w + 2)

	// Compute numerator and denominator for the parametric step size of the linearization. Here, the epsilon referenced
	// from the cited paper is 1/precision.
	rpMinus1 := max32(0, maxLen*precision-1)
	numer := sqrt32(dpx*dpx+dpy*dpy)*precision + rpMinus1*dw
	denom := 4 * min32(w, 1)

	// Number of segments = sqrt(numer / denom). This assumes the parametric interval of the curve being linearized is
	// [t0,t1] = [0, 1].
	return numer / denom
}

// wangsConic returns Wang's formula for a conic curve: the number of line segments needed.
func wangsConic(tolerance float32, p0, p1, p2 geom.Point, w float32, xf vectorXform) float32 {
	return sqrt32(wangsConicP2(tolerance, p0, p1, p2, w, xf))
}
