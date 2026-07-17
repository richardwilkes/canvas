// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Matrix decomposition helpers (decomposeScale, differentialAreaScale, minMaxScales) and homogeneous quad-containment
// tests (quadContainsRect/quadContainsRectMask) that filtercore consumes. These stay private to filtercore until
// another package needs them.

package filtercore

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

func absf32(v float32) float32  { return float32(math.Abs(float64(v))) }
func sqrtf32(v float32) float32 { return float32(math.Sqrt(float64(v))) }

func nearlyZero(v float32) bool { return geom.ScalarNearlyZero(v) }

func nearlyEqual(a, b, tolerance float32) bool { return absf32(a-b) <= tolerance }

func roundToScalar(v float32) float32 { return float32(math.Round(float64(v))) }

// ieeeFloatDivide is an ordinary float divide with no checks (named for the IEEE 754 semantics it relies on: division
// by zero produces +-Inf or NaN rather than panicking).
func ieeeFloatDivide(a, b float32) float32 { return a / b }

func pin32(v, lo, hi int32) int32 {
	// Clamp v to [lo, hi].
	return max(lo, min(v, hi))
}

// decomposeScale factors m as remaining * Scale(sx, sy).
func decomposeScale(m *geom.Matrix) (scale geom.Size, remaining geom.Matrix, ok bool) {
	if m.HasPerspective() {
		return geom.Size{}, geom.Matrix{}, false
	}
	sx := geom.Point{X: m.Get(geom.MScaleX), Y: m.Get(geom.MSkewY)}.Length()
	sy := geom.Point{X: m.Get(geom.MSkewX), Y: m.Get(geom.MScaleY)}.Length()
	if !geom.IsFinite(sx, sy) || nearlyZero(sx) || nearlyZero(sy) {
		return geom.Size{}, geom.Matrix{}, false
	}
	remaining = *m
	var inv geom.Matrix
	inv.SetScale(1/sx, 1/sy)
	remaining.PreConcat(&inv)
	return geom.Size{Width: sx, Height: sy}, remaining, true
}

// differentialAreaScale returns |det J| of the projected map m at p, or +inf at/behind the w=0 plane.
func differentialAreaScale(m *geom.Matrix, p geom.Point) float32 {
	// xyw = M * [p, 1]
	x := m.Get(geom.MScaleX)*p.X + m.Get(geom.MSkewX)*p.Y + m.Get(geom.MTransX)
	y := m.Get(geom.MSkewY)*p.X + m.Get(geom.MScaleY)*p.Y + m.Get(geom.MTransY)
	w := m.Get(geom.MPersp0)*p.X + m.Get(geom.MPersp1)*p.Y + m.Get(geom.MPersp2)
	if w < geom.ScalarNearlyZeroTol {
		// Reaching the discontinuity of xy/w and where the point would clip to w >= 0.
		return float32(math.Inf(1))
	}
	// J' = [x y w; m00 m10 m20; m01 m11 m21], |det J| = |det J' / w^3| (computed in doubles).
	a00, a01, a02 := float64(x), float64(y), float64(w)
	a10, a11, a12 := float64(m.Get(geom.MScaleX)), float64(m.Get(geom.MSkewY)), float64(m.Get(geom.MPersp0))
	a20, a21, a22 := float64(m.Get(geom.MSkewX)), float64(m.Get(geom.MScaleY)), float64(m.Get(geom.MPersp1))
	det := a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20)
	denom := 1.0 / float64(w) // 1/w
	denom = denom * denom * denom
	return absf32(float32(det * denom))
}

// minMaxScales returns m's minimum and maximum singular values (its extreme scale factors).
func minMaxScales(m *geom.Matrix) (minScale, maxScale float32, ok bool) {
	if m.HasPerspective() {
		return 0, 0, false
	}
	if m.IsIdentity() {
		return 1, 1, true
	}
	sdot := func(a, b, c, d float32) float32 { return a*b + c*d }
	scaleX := m.Get(geom.MScaleX)
	skewX := m.Get(geom.MSkewX)
	skewY := m.Get(geom.MSkewY)
	scaleY := m.Get(geom.MScaleY)
	if skewX == 0 && skewY == 0 {
		// Scale/translate-only: results are the absolute scale factors, sorted.
		r0 := absf32(scaleX)
		r1 := absf32(scaleY)
		if r0 > r1 {
			r0, r1 = r1, r0
		}
		return r0, r1, true
	}
	// Ignore the translation and compute the singular values of the 2x2: [a b; b c] = A^T*A.
	a := sdot(scaleX, scaleX, skewY, skewY)
	b := sdot(scaleX, skewX, scaleY, skewY)
	c := sdot(skewX, skewX, scaleY, scaleY)
	var r0, r1 float32
	bSqd := b * b
	if bSqd <= geom.ScalarNearlyZeroTol*geom.ScalarNearlyZeroTol {
		// The upper-left 2x2 is orthogonal: save some math.
		r0, r1 = a, c
		if r0 > r1 {
			r0, r1 = r1, r0
		}
	} else {
		aminusc := a - c
		apluscdiv2 := 0.5 * (a + c)
		x := 0.5 * sqrtf32(aminusc*aminusc+4*bSqd)
		r0 = apluscdiv2 - x
		r1 = apluscdiv2 + x
	}
	if !geom.IsFinite(r0) || !geom.IsFinite(r1) {
		return 0, 0, false
	}
	// Float inaccuracy in the quadratic can push nearly-zero eigenvalues negative; cap them.
	if r0 < 0 {
		r0 = 0
	}
	if r1 < 0 {
		r1 = 0
	}
	return sqrtf32(r0), sqrtf32(r1), true
}

// quadContainsRectMask reports, per edge, whether b's corners lie inside a's transformed quad, ordered T, R, B, L.
func quadContainsRectMask(m *geom.Matrix, a, b geom.Rect, tol float32) [4]bool {
	// With empty rectangles the calculated edges could give surprising results, so an empty 'a' is never "containing".
	if a.IsEmpty() {
		return [4]bool{}
	}
	// The 4 homogeneous coordinates of 'a' transformed by 'm' (Z=0, W=1); corners in CW order.
	ax := [4]float32{a.Left, a.Right, a.Right, a.Left}
	ay := [4]float32{a.Top, a.Top, a.Bottom, a.Bottom}
	var mx, my, mw [4]float32
	allWNeg := true
	for i := range 4 {
		mx[i] = m.Get(geom.MScaleX)*ax[i] + m.Get(geom.MSkewX)*ay[i] + m.Get(geom.MTransX)
		my[i] = m.Get(geom.MSkewY)*ax[i] + m.Get(geom.MScaleY)*ay[i] + m.Get(geom.MTransY)
		mw[i] = m.Get(geom.MPersp0)*ax[i] + m.Get(geom.MPersp1)*ay[i] + m.Get(geom.MPersp2)
		if mw[i] >= 0 {
			allWNeg = false
		}
	}
	if allWNeg {
		// All points of A mapped to w < 0: the edge equations would represent the convex hull of projected points when
		// A should be considered empty.
		return [4]bool{}
	}
	// Cross products of adjacent vertices give homogeneous lines for the quad's 4 sides.
	var lA, lB, lC [4]float32
	for i := range 4 {
		j := (i + 1) & 3
		lA[i] = my[i]*mw[j] - mw[i]*my[j]
		lB[i] = mw[i]*mx[j] - mx[i]*mw[j]
		lC[i] = mx[i]*my[j] - my[i]*mx[j]
	}
	// The corners were in CW order but may have become CCW; the sign corrects the edge normals to point inwards.
	sign := float32(1)
	if lA[0]*lB[1]-lB[0]*lA[1] < 0 {
		sign = -1
	}
	// Distance from b's corners (assumed W=1 post-projection) to each edge.
	bi := b.Inset(tol, tol)
	var out [4]bool
	for i := range 4 {
		d0 := sign * (lA[i]*bi.Left + lB[i]*bi.Top + lC[i])
		d1 := sign * (lA[i]*bi.Right + lB[i]*bi.Top + lC[i])
		d2 := sign * (lA[i]*bi.Right + lB[i]*bi.Bottom + lC[i])
		d3 := sign * (lA[i]*bi.Left + lB[i]*bi.Bottom + lC[i])
		out[i] = d0 >= 0 && d1 >= 0 && d2 >= 0 && d3 >= 0
	}
	return out
}

// quadContainsRect reports whether b lies entirely inside a's transformed quad.
func quadContainsRect(m *geom.Matrix, a, b geom.Rect, tol float32) bool {
	mask := quadContainsRectMask(m, a, b, tol)
	return mask[0] && mask[1] && mask[2] && mask[3]
}

// quadContainsIRect is quadContainsRect for integer rects.
func quadContainsIRect(m *geom.Matrix, a, b geom.IRect, tol float32) bool {
	return quadContainsRect(m, a.ToRect(), b.ToRect(), tol)
}
