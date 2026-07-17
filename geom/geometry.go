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

// This file implements the quad/cubic evaluation, chopping, and extrema-finding used by the path flattening and
// clipping machinery. Curves are represented as point slices: quads are [3]Point, cubics are [4]Point, and chops write
// the shared point once (a chop of a quad yields 5 points, of a cubic 7 points).

// ValidUnitDivide computes numer/denom and returns it with true only when the result is a valid parametric t in (0, 1).
// Zero, NaN, and out-of-range results return false.
func ValidUnitDivide(numer, denom float32) (float32, bool) {
	if numer < 0 {
		numer = -numer
		denom = -denom
	}
	if denom == 0 || numer == 0 || numer >= denom {
		return 0, false
	}
	r := numer / denom
	if ScalarIsNaN(r) {
		return 0, false
	}
	if r == 0 { // catch underflow if numer <<<< denom
		return 0, false
	}
	return r, true
}

// FindUnitQuadRoots stores the real roots of A*t^2 + B*t + C = 0 that lie in (0, 1) into roots, returning the count
// (0..2). Roots are sorted, and a double root is stored once.
func FindUnitQuadRoots(a, b, c float32, roots *[2]float32) int {
	if a == 0 {
		if r, ok := ValidUnitDivide(-c, b); ok {
			roots[0] = r
			return 1
		}
		return 0
	}
	// Use doubles so we do not overflow temporarily while computing the discriminant.
	dr := float64(b)*float64(b) - 4*float64(a)*float64(c)
	if dr < 0 {
		return 0
	}
	dr = math.Sqrt(dr)
	r := DoubleToScalar(dr)
	if !IsFinite(r) {
		return 0
	}
	var q float32
	if b < 0 {
		q = -(b - r) / 2
	} else {
		q = -(b + r) / 2
	}
	n := 0
	if t, ok := ValidUnitDivide(q, a); ok {
		roots[n] = t
		n++
	}
	if t, ok := ValidUnitDivide(c, q); ok {
		roots[n] = t
		n++
	}
	if n == 2 {
		if roots[0] > roots[1] {
			roots[0], roots[1] = roots[1], roots[0]
		} else if roots[0] == roots[1] {
			n = 1 // skip the double root
		}
	}
	return n
}

///////////////////////////////////////////////////////////////////////////////
// Quads

// quadCoeff holds a quad's polynomial coefficients: A = P2 - 2*P1 + P0, B = 2*(P1 - P0), C = P0; eval is (A*t + B)*t +
// C per component.
type quadCoeff struct {
	ax, ay float32
	bx, by float32
	cx, cy float32
}

func makeQuadCoeff(src []Point) quadCoeff {
	var q quadCoeff
	q.cx = src[0].X
	q.cy = src[0].Y
	q.bx = 2 * (src[1].X - src[0].X)
	q.by = 2 * (src[1].Y - src[0].Y)
	q.ax = src[2].X - 2*src[1].X + src[0].X
	q.ay = src[2].Y - 2*src[1].Y + src[0].Y
	return q
}

func (q quadCoeff) eval(t float32) Point {
	return Point{
		X: (q.ax*t+q.bx)*t + q.cx,
		Y: (q.ay*t+q.by)*t + q.cy,
	}
}

// EvalQuadAt returns the point on the quad src at parameter t.
func EvalQuadAt(src []Point, t float32) Point {
	return makeQuadCoeff(src).eval(t)
}

// EvalQuadTangentAt returns the tangent of the quad src at parameter t, including the degenerate endpoint handling.
func EvalQuadTangentAt(src []Point, t float32) Point {
	// The derivative is 2(b - a + (a - 2b + c)t), which is the zero vector when t is 0 or 1 and the control point
	// equals that end point; fall back to the end-to-end vector in that case.
	if (t == 0 && src[0] == src[1]) || (t == 1 && src[1] == src[2]) {
		return src[2].Sub(src[0])
	}
	bx := src[1].X - src[0].X
	by := src[1].Y - src[0].Y
	ax := src[2].X - src[1].X - bx
	ay := src[2].Y - src[1].Y - by
	tx := ax*t + bx
	ty := ay*t + by
	return Point{X: tx + tx, Y: ty + ty}
}

func interpScalar(v0, v1, t float32) float32 {
	return v0 + (v1-v0)*t
}

func interpPoint(v0, v1 Point, t float32) Point {
	return Point{X: interpScalar(v0.X, v1.X, t), Y: interpScalar(v0.Y, v1.Y, t)}
}

// ChopQuadAt splits the quad src at t into two quads sharing point dst[2]. dst must hold 5 points. Callers must ensure
// 0 < t < 1.
func ChopQuadAt(src, dst []Point, t float32) {
	p01 := interpPoint(src[0], src[1], t)
	p12 := interpPoint(src[1], src[2], t)
	dst[0] = src[0]
	dst[1] = p01
	dst[2] = interpPoint(p01, p12, t)
	dst[3] = p12
	dst[4] = src[2]
}

// ChopQuadAtHalf splits the quad src at its midpoint.
func ChopQuadAtHalf(src, dst []Point) {
	ChopQuadAt(src, dst, 0.5)
}

// FindQuadExtrema returns the t in (0, 1) where the quad with control values a, b, c has a derivative of zero.
func FindQuadExtrema(a, b, c float32) (float32, bool) {
	// The derivative is zero at t = (a - b) / (a - 2b + c).
	return ValidUnitDivide(a-b, a-b-b+c)
}

///////////////////////////////////////////////////////////////////////////////
// Cubics

// cubicCoeff holds a cubic's polynomial coefficients: A = P3 + 3*(P1 - P2) - P0, B = 3*(P2 - 2*P1 + P0), C = 3*(P1 -
// P0), D = P0; eval is ((A*t + B)*t + C)*t + D per component.
type cubicCoeff struct {
	ax, ay float32
	bx, by float32
	cx, cy float32
	dx, dy float32
}

func makeCubicCoeff(src []Point) cubicCoeff {
	var c cubicCoeff
	c.ax = src[3].X + 3*(src[1].X-src[2].X) - src[0].X
	c.ay = src[3].Y + 3*(src[1].Y-src[2].Y) - src[0].Y
	c.bx = 3 * (src[2].X - 2*src[1].X + src[0].X)
	c.by = 3 * (src[2].Y - 2*src[1].Y + src[0].Y)
	c.cx = 3 * (src[1].X - src[0].X)
	c.cy = 3 * (src[1].Y - src[0].Y)
	c.dx = src[0].X
	c.dy = src[0].Y
	return c
}

func (c cubicCoeff) eval(t float32) Point {
	return Point{
		X: ((c.ax*t+c.bx)*t+c.cx)*t + c.dx,
		Y: ((c.ay*t+c.by)*t+c.cy)*t + c.dy,
	}
}

// EvalCubicPosAt returns the position on the cubic src at t.
func EvalCubicPosAt(src []Point, t float32) Point {
	return makeCubicCoeff(src).eval(t)
}

// EvalCubicTangentAt returns the tangent of the cubic src at t, including the degenerate endpoint handling.
func EvalCubicTangentAt(src []Point, t float32) Point {
	if (t == 0 && src[0] == src[1]) || (t == 1 && src[2] == src[3]) {
		var tangent Point
		if t == 0 {
			tangent = src[2].Sub(src[0])
		} else {
			tangent = src[3].Sub(src[1])
		}
		if tangent.X == 0 && tangent.Y == 0 {
			tangent = src[3].Sub(src[0])
		}
		return tangent
	}
	return evalCubicDerivative(src, t)
}

// evalCubicDerivative evaluates the cubic's derivative as a quad, with coefficients A = P3 + 3*(P1 - P2) - P0, B =
// 2*(P2 - 2*P1 + P0), C = P1 - P0.
func evalCubicDerivative(src []Point, t float32) Point {
	var q quadCoeff
	q.ax = src[3].X + 3*(src[1].X-src[2].X) - src[0].X
	q.ay = src[3].Y + 3*(src[1].Y-src[2].Y) - src[0].Y
	q.bx = 2 * (src[2].X - 2*src[1].X + src[0].X)
	q.by = 2 * (src[2].Y - 2*src[1].Y + src[0].Y)
	q.cx = src[1].X - src[0].X
	q.cy = src[1].Y - src[0].Y
	return q.eval(t)
}

// EvalCubicCurvatureAt returns the second derivative of the cubic src at t.
func EvalCubicCurvatureAt(src []Point, t float32) Point {
	ax := src[3].X + 3*(src[1].X-src[2].X) - src[0].X
	ay := src[3].Y + 3*(src[1].Y-src[2].Y) - src[0].Y
	bx := src[2].X - 2*src[1].X + src[0].X
	by := src[2].Y - 2*src[1].Y + src[0].Y
	return Point{X: ax*t + bx, Y: ay*t + by}
}

// FindCubicExtrema returns the t values in (0, 1) where the cubic with control values a, b, c, d has a zero derivative.
// Returns the count (0..2).
func FindCubicExtrema(a, b, c, d float32, tValues *[2]float32) int {
	// Divide the derivative coefficients by 3 to simplify.
	aa := d - a + 3*(b-c)
	bb := 2 * (a - b - b + c)
	cc := b - a
	return FindUnitQuadRoots(aa, bb, cc, tValues)
}

// uncheckedMix computes (b - a)*t + a. It does not return exactly b when t == 1 (callers special-case that, as
// ChopCubicAt does), but gets better precision than a*(1 - t) + b*t for things like chopping cubics on exact cusp
// points.
func uncheckedMix(a, b Point, t float32) Point {
	return Point{X: (b.X-a.X)*t + a.X, Y: (b.Y-a.Y)*t + a.Y}
}

// ChopCubicAt splits the cubic at t into two cubics sharing dst[3]. dst must hold 7 points.
func ChopCubicAt(src, dst []Point, t float32) {
	if t == 1 {
		copy(dst[:4], src[:4])
		dst[4] = src[3]
		dst[5] = src[3]
		dst[6] = src[3]
		return
	}
	ab := uncheckedMix(src[0], src[1], t)
	bc := uncheckedMix(src[1], src[2], t)
	cd := uncheckedMix(src[2], src[3], t)
	abc := uncheckedMix(ab, bc, t)
	bcd := uncheckedMix(bc, cd, t)
	abcd := uncheckedMix(abc, bcd, t)
	dst[0] = src[0]
	dst[1] = ab
	dst[2] = abc
	dst[3] = abcd
	dst[4] = bcd
	dst[5] = cd
	dst[6] = src[3]
}

// ChopCubicAt2 splits the cubic at both t0 and t1, producing three cubics. dst must hold 10 points.
func ChopCubicAt2(src, dst []Point, t0, t1 float32) {
	if t1 == 1 {
		ChopCubicAt(src, dst, t0)
		dst[7] = src[3]
		dst[8] = src[3]
		dst[9] = src[3]
		return
	}
	// Both chops are computed from the original curve, matching what a SIMD dual evaluation would produce.
	ab0 := uncheckedMix(src[0], src[1], t0)
	bc0 := uncheckedMix(src[1], src[2], t0)
	cd0 := uncheckedMix(src[2], src[3], t0)
	abc0 := uncheckedMix(ab0, bc0, t0)
	bcd0 := uncheckedMix(bc0, cd0, t0)
	abcd0 := uncheckedMix(abc0, bcd0, t0)
	ab1 := uncheckedMix(src[0], src[1], t1)
	bc1 := uncheckedMix(src[1], src[2], t1)
	cd1 := uncheckedMix(src[2], src[3], t1)
	abc1 := uncheckedMix(ab1, bc1, t1)
	bcd1 := uncheckedMix(bc1, cd1, t1)
	abcd1 := uncheckedMix(abc1, bcd1, t1)
	// The middle segment's control points mix abc/bcd with the t values swapped between lane pairs: they are abc0..bcd0
	// evaluated at t1 and abc1..bcd1 evaluated at t0.
	mid0 := uncheckedMix(abc0, bcd0, t1)
	mid1 := uncheckedMix(abc1, bcd1, t0)
	dst[0] = src[0]
	dst[1] = ab0
	dst[2] = abc0
	dst[3] = abcd0
	dst[4] = mid0
	dst[5] = mid1
	dst[6] = abcd1
	dst[7] = bcd1
	dst[8] = cd1
	dst[9] = src[3]
}

// ChopCubicAtHalf splits the cubic src at its midpoint.
func ChopCubicAtHalf(src, dst []Point) {
	ChopCubicAt(src, dst, 0.5)
}

// pin32 clamps x to the inclusive range [lo, hi].
func pin32(x, lo, hi float32) float32 {
	return max32(lo, min32(x, hi))
}

// ChopCubicAtList chops the cubic at each of the sorted tValues in (0, 1), producing len(tValues)+1 cubics into dst
// (3*len(tValues)+4 points, each adjacent pair of cubics sharing a point).
func ChopCubicAtList(src, dst []Point, tValues []float32) {
	if len(tValues) == 0 {
		copy(dst[:4], src[:4])
		return
	}
	i := 0
	for ; i < len(tValues)-1; i += 2 {
		t0 := tValues[i]
		t1 := tValues[i+1]
		if i != 0 {
			lastT := tValues[i-1]
			t0 = pin32(IEEEFloatDivide(t0-lastT, 1-lastT), 0, 1)
			t1 = pin32(IEEEFloatDivide(t1-lastT, 1-lastT), 0, 1)
		}
		ChopCubicAt2(src, dst, t0, t1)
		src = dst[6:]
		dst = dst[6:]
	}
	if i < len(tValues) {
		t := tValues[i]
		if i != 0 {
			lastT := tValues[i-1]
			t = pin32(IEEEFloatDivide(t-lastT, 1-lastT), 0, 1)
		}
		ChopCubicAt(src, dst, t)
	}
}

///////////////////////////////////////////////////////////////////////////////
// Y-extrema chops (used by point-in-path winding computation)

// isNotMonotonic reports whether the sequence a, b, c is not monotonic.
func isNotMonotonic(a, b, c float32) bool {
	ab := a - b
	bc := b - c
	if ab < 0 {
		bc = -bc
	}
	return ab == 0 || bc < 0
}

// ChopQuadAtYExtrema chops the quad at its Y extrema if it is not monotonic in Y (flattening the shared point's Y
// exactly) and returns 1 (two quads in dst[0:5]); otherwise it copies src to dst[0:3] (forcing monotonicity if the
// unit divide underflowed) and returns 0. dst must hold 5 points.
func ChopQuadAtYExtrema(src, dst []Point) int {
	a := src[0].Y
	b := src[1].Y
	c := src[2].Y
	if isNotMonotonic(a, b, c) {
		if t, ok := ValidUnitDivide(a-b, a-b-b+c); ok {
			ChopQuadAt(src, dst, t)
			// Force the chopped point's neighbors to share its Y.
			dst[1].Y = dst[2].Y
			dst[3].Y = dst[2].Y
			return 1
		}
		// if we get here, we need to force dst to be monotonic, even though we couldn't compute a unit_divide value
		// (probably underflow).
		if ScalarAbs(a-b) < ScalarAbs(b-c) {
			b = a
		} else {
			b = c
		}
	}
	dst[0] = Point{X: src[0].X, Y: a}
	dst[1] = Point{X: src[1].X, Y: b}
	dst[2] = Point{X: src[2].X, Y: c}
	return 0
}

// ChopCubicAtYExtrema chops the cubic at its Y extrema (0, 1 or 2 of them), flattening each shared point's Y exactly,
// and returns the number of chops (dst holds roots+1 cubics, 3*roots+4 points). dst must hold 10 points.
func ChopCubicAtYExtrema(src, dst []Point) int {
	var tValues [2]float32
	roots := FindCubicExtrema(src[0].Y, src[1].Y, src[2].Y, src[3].Y, &tValues)
	ChopCubicAtList(src, dst, tValues[:roots])
	if roots > 0 {
		// Force the chopped point's neighbors to share its Y.
		dst[2].Y = dst[3].Y
		dst[4].Y = dst[3].Y
		if roots == 2 {
			dst[5].Y = dst[6].Y
			dst[7].Y = dst[6].Y
		}
	}
	return roots
}
