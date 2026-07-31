// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import "math"

// scalarCubeRoot computes pow(x, 0.3333333f), an approximate cube root.
func scalarCubeRoot(x float32) float32 {
	return float32(math.Pow(float64(x), float64(float32(0.3333333))))
}

// solveCubicPoly solves coeff[0]t³+coeff[1]t²+coeff[2]t+coeff[3] == 0, returning the roots pinned to [0, 1], distinct
// and increasing.
func solveCubicPoly(coeff *[4]float32, tValues *[3]float32) int {
	if ScalarNearlyZero(coeff[0]) { // we're just a quadratic
		var roots [2]float32
		n := FindUnitQuadRoots(coeff[1], coeff[2], coeff[3], &roots)
		tValues[0] = roots[0]
		tValues[1] = roots[1]
		return n
	}

	inva := ScalarInvert(coeff[0])
	a := coeff[1] * inva
	b := coeff[2] * inva
	c := coeff[3] * inva

	q := (a*a - b*3) / 9
	r := (2*a*a*a - 9*a*b + 27*c) / 54

	q3 := q * q * q
	r2MinusQ3 := r*r - q3
	adiv3 := a / 3

	if r2MinusQ3 < 0 { // we have 3 real roots
		// the divide/root can, due to finite precisions, be slightly outside of -1...1
		theta := float32(math.Acos(float64(pin32(r/float32(math.Sqrt(float64(q3))), -1, 1))))
		neg2RootQ := -2 * float32(math.Sqrt(float64(q)))

		tValues[0] = pin32(neg2RootQ*float32(math.Cos(float64(theta/3)))-adiv3, 0, 1)
		tValues[1] = pin32(neg2RootQ*float32(math.Cos(float64((theta+2*math.Pi)/3)))-adiv3, 0, 1)
		tValues[2] = pin32(neg2RootQ*float32(math.Cos(float64((theta-2*math.Pi)/3)))-adiv3, 0, 1)

		// now sort the roots (bubble_sort of 3)
		sort3(tValues)
		return collapsDuplicatesExact(tValues[:], 3)
	}
	// we have 1 real root
	a2 := float32(math.Abs(float64(r))) + float32(math.Sqrt(float64(r2MinusQ3)))
	a2 = scalarCubeRoot(a2)
	if r > 0 {
		a2 = -a2
	}
	if a2 != 0 {
		a2 += q / a2
	}
	tValues[0] = pin32(a2-adiv3, 0, 1)
	return 1
}

func sort3(v *[3]float32) {
	if v[0] > v[1] {
		v[0], v[1] = v[1], v[0]
	}
	if v[1] > v[2] {
		v[1], v[2] = v[2], v[1]
	}
	if v[0] > v[1] {
		v[0], v[1] = v[1], v[0]
	}
}

// collapsDuplicatesExact removes exact-duplicate adjacent entries from a sorted array.
func collapsDuplicatesExact(array []float32, count int) int {
	n := 0
	for i := 0; i < count; i++ {
		if n == 0 || array[i] != array[n-1] {
			array[n] = array[i]
			n++
		}
	}
	return n
}

// formulateF1DotF2 computes the F' dot F” coefficients over one coordinate axis (p0..p3 are that axis's values from the
// cubic's 4 control points).
func formulateF1DotF2(p0, p1, p2, p3 float32, coeff *[4]float32) {
	a := p1 - p0
	b := p2 - 2*p1 + p0
	c := p3 + 3*(p1-p2) - p0

	coeff[0] = c * c
	coeff[1] = 3 * b * c
	coeff[2] = 2*b*b + c*a
	coeff[3] = a * b
}

// FindCubicMaxCurvature returns the t values (up to 3) where the cubic's curvature is at a local maximum (F' dot F” ==
// 0).
func FindCubicMaxCurvature(src []Point, tValues *[3]float32) int {
	var coeffX, coeffY [4]float32
	formulateF1DotF2(src[0].X, src[1].X, src[2].X, src[3].X, &coeffX)
	formulateF1DotF2(src[0].Y, src[1].Y, src[2].Y, src[3].Y, &coeffY)
	for i := range coeffX {
		coeffX[i] += coeffY[i]
	}
	return solveCubicPoly(&coeffX, tValues)
}

// ChopCubicAtMaxCurvature chops the cubic at its max-curvature t values inside (0, 1), writing up to 4 cubics (13
// points) into dst and returning the cubic count.
func ChopCubicAtMaxCurvature(src, dst []Point) int {
	var roots [3]float32
	rootCount := FindCubicMaxCurvature(src, &roots)

	// Throw out values not inside 0..1.
	var tValues [3]float32
	count := 0
	for i := 0; i < rootCount; i++ {
		if 0 < roots[i] && roots[i] < 1 {
			tValues[count] = roots[i]
			count++
		}
	}

	if dst != nil {
		if count == 0 {
			copy(dst[:4], src[:4])
		} else {
			ChopCubicAtList(src, dst, tValues[:count])
		}
	}
	return count + 1
}

// FindQuadMaxCurvature returns the t where the quad's curvature is maximal (t = -(Ax Bx + Ay By) / (Bx^2 + By^2),
// pinned to the unit range). The result is in [0, 1] for finite input, but a NaN coordinate makes both numer and denom
// NaN, which fails every comparison below and falls through to return NaN — matching Skia, whose own assert is
// (0 <= t && t < 1) || SkScalarIsNaN(t). Callers that feed the result into further geometry must handle that, as
// gpu/gl/aahairlinepathrenderer.go does.
//
// The dot products are pinned to unfused evaluation because this function branches on numer <= 0 / numer >= denom, and
// FMA fusion would move mathematically-zero results across those cliffs differently per platform.
func FindQuadMaxCurvature(src []Point) float32 {
	ax := src[1].X - src[0].X
	ay := src[1].Y - src[0].Y
	bx := src[0].X - src[1].X - src[1].X + src[2].X
	by := src[0].Y - src[1].Y - src[1].Y + src[2].Y

	numer := -(float32(ax*bx) + float32(ay*by))
	denom := float32(bx*bx) + float32(by*by)
	if denom < 0 {
		numer = -numer
		denom = -denom
	}
	if numer <= 0 {
		return 0
	}
	if numer >= denom { // Also catches denom=0.
		return 1
	}
	return numer / denom
}

// ChopQuadAtMaxCurvature chops the quad at its max-curvature t if it lies inside (0, 1), writing up to 2 quads (5
// points) into dst and returning the quad count.
func ChopQuadAtMaxCurvature(src, dst []Point) int {
	t := FindQuadMaxCurvature(src)
	if t > 0 && t < 1 {
		ChopQuadAt(src, dst, t)
		return 2
	}
	copy(dst[:3], src[:3])
	return 1
}

// FindCubicInflections returns the t values (up to 2) where the cubic's curvature is zero. The quadratic's coefficients
// are cross products, pinned unfused like Point.Cross (their signs pick the root count).
func FindCubicInflections(src []Point, tValues *[2]float32) int {
	ax := src[1].X - src[0].X
	ay := src[1].Y - src[0].Y
	bx := src[2].X - 2*src[1].X + src[0].X
	by := src[2].Y - 2*src[1].Y + src[0].Y
	cx := src[3].X + 3*(src[1].X-src[2].X) - src[0].X
	cy := src[3].Y + 3*(src[1].Y-src[2].Y) - src[0].Y

	return FindUnitQuadRoots(float32(bx*cy)-float32(by*cx),
		float32(ax*cy)-float32(ay*cx),
		float32(ax*by)-float32(ay*bx),
		tValues)
}

// ChopCubicAtInflections chops the cubic at its inflection points, writing up to 3 cubics (10 points) into dst and
// returning the cubic count.
func ChopCubicAtInflections(src, dst []Point) int {
	var tValues [2]float32
	count := FindCubicInflections(src, &tValues)
	if dst != nil {
		if count == 0 {
			copy(dst[:4], src[:4])
		} else {
			ChopCubicAtList(src, dst, tValues[:count])
		}
	}
	return count + 1
}

// calcCubicPrecision returns a tolerance proportional to the *square* of the cubic's dimensions: it sums the squared
// lengths of the control polygon's segments. Compare it against a squared magnitude (LengthSqd/DistanceToSqd), never
// against an unsquared one.
func calcCubicPrecision(src []Point) float32 {
	return (src[1].DistanceToSqd(src[0]) + src[2].DistanceToSqd(src[1]) +
		src[3].DistanceToSqd(src[2])) * 1e-8
}

// onSameSide reports whether src[testIndex] and src[testIndex+1] are in the same half plane defined by the segment
// src[lineIndex]..src[lineIndex+1].
func onSameSide(src []Point, testIndex, lineIndex int) bool {
	origin := src[lineIndex]
	line := src[lineIndex+1].Sub(origin)
	var crosses [2]float32
	for index := 0; index < 2; index++ {
		testLine := src[testIndex+index].Sub(origin)
		crosses[index] = line.Cross(testLine)
	}
	return crosses[0]*crosses[1] >= 0
}

// FindCubicCusp returns the t of the cubic's cusp if there is one, else -1.
func FindCubicCusp(src []Point) float32 {
	// When the adjacent control point matches the end point, it behaves as if the cubic has a cusp: there's a point of
	// max curvature where the derivative goes to zero. Ideally, this would be where t is zero or one, but math error
	// makes not so. It is not uncommon to create cubics this way; skip them.
	if src[0] == src[1] {
		return -1
	}
	if src[2] == src[3] {
		return -1
	}
	// Cubics only have a cusp if the line segments formed by the control and end points cross. Detect crossing if line
	// ends are on opposite sides of plane formed by the other line.
	if onSameSide(src, 0, 2) || onSameSide(src, 2, 0) {
		return -1
	}
	// Cubics may have multiple points of maximum curvature, although at most only one is a cusp.
	var maxCurvature [3]float32
	roots := FindCubicMaxCurvature(src, &maxCurvature)
	for index := 0; index < roots; index++ {
		testT := maxCurvature[index]
		if testT <= 0 || testT >= 1 { // no need to consider max curvature on the end
			continue
		}
		// A cusp is at the max curvature, and also has a derivative close to zero. Choose the 'close to zero' meaning
		// by comparing the derivative length with the overall cubic size.
		dPt := evalCubicDerivative(src, testT)
		dPtMagnitude := dPt.LengthSqd()
		precision := calcCubicPrecision(src)
		if dPtMagnitude < precision {
			// All three max curvature t values may be close to the cusp; return the first one.
			return testT
		}
	}
	return -1
}
