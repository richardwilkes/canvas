// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Quadratic and cubic Bezier horizontal-line intersection helpers, used by the text-blob intercepts machinery that
// computes glyph path gaps.

package geom

import "math"

// pinTRange pins t to 0 or 1 if it is within half a float ulp of 0 or 1.
func pinTRange(t float64) float64 {
	// The ULPs around 0 are tiny compared to the ULPs around 1. Shift to 1 to use the same size ULPs.
	if float32(t+1.0) == 1.0 {
		return 0.0
	} else if float32(t) == 1.0 {
		return 1.0
	}
	return t
}

// QuadEvalAt evaluates A*t^2 + B*t + C via fused multiply-add.
func QuadEvalAt(a, b, c, t float64) float64 {
	return math.FMA(math.FMA(a, t, b), t, c)
}

// BezierQuadIntersectHorizontal returns the x-coordinates where the quadratic Bezier (p0, p1, p2) crosses y =
// yIntercept, appended to out (capacity 2).
func BezierQuadIntersectHorizontal(p0, p1, p2 Point, yIntercept float32, out []float32) []float32 {
	// Calculate A, B, C using doubles to reduce round-off error. The polynomial is generated in the form A*t^2 - 2*B*t
	// + C, so B is (p0 - p1) without the factor of 2.
	ax := float64(p0.X) - 2*float64(p1.X) + float64(p2.X)
	ay := float64(p0.Y) - 2*float64(p1.Y) + float64(p2.Y)
	bx := float64(p0.X) - float64(p1.X)
	by := float64(p0.Y) - float64(p1.Y)
	cx := float64(p0.X)
	cy := float64(p0.Y)

	_, r0, r1 := quadRoots(ay, by, cy-float64(yIntercept))

	t0 := pinTRange(r0)
	if 0 <= t0 && t0 <= 1 {
		out = append(out, float32(QuadEvalAt(ax, -2*bx, cx, t0)))
	}
	t1 := pinTRange(r1)
	if 0 <= t1 && t1 <= 1 && t1 != t0 {
		out = append(out, float32(QuadEvalAt(ax, -2*bx, cx, t1)))
	}
	return out
}

// BezierCubicIntersectHorizontal returns the x-coordinates where the cubic Bezier (p0..p3) crosses y = yIntercept,
// appended to out (capacity 3).
func BezierCubicIntersectHorizontal(p0, p1, p2, p3 Point, yIntercept float32, out []float32) []float32 {
	ax := -float64(p0.X) + 3*float64(p1.X) - 3*float64(p2.X) + float64(p3.X)
	ay := -float64(p0.Y) + 3*float64(p1.Y) - 3*float64(p2.Y) + float64(p3.Y)
	bx := 3*float64(p0.X) - 6*float64(p1.X) + 3*float64(p2.X)
	by := 3*float64(p0.Y) - 6*float64(p1.Y) + 3*float64(p2.Y)
	cx := -3*float64(p0.X) + 3*float64(p1.X)
	cy := -3*float64(p0.Y) + 3*float64(p1.Y)
	dx := float64(p0.X)
	dy := float64(p0.Y)

	var roots [3]float64
	n := CubicRootsReal(ay, by, cy, dy-float64(yIntercept), &roots)
	for _, t := range roots[:n] {
		pinned := pinTRange(t)
		if 0 <= pinned && pinned <= 1 {
			out = append(out, float32(CubicEvalAt(ax, bx, cx, dx, pinned)))
		}
	}
	return out
}
