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

// CubicType classifies a cubic per "Resolution Independent Curve Rendering using Programmable Graphics Hardware" (Loop
// & Blinn, 2005).
type CubicType int

const (
	// CubicSerpentine is an S-shaped cubic with two real inflection points.
	CubicSerpentine CubicType = iota
	// CubicLoop is a self-intersecting cubic.
	CubicLoop
	// CubicLocalCusp has a cusp at a finite parameter value with an inflection at t == infinity.
	CubicLocalCusp
	// CubicCuspAtInfinity has a cusp at t == infinity with a local inflection.
	CubicCuspAtInfinity
	// CubicQuadratic degenerates to a quadratic.
	CubicQuadratic
	// CubicLineOrPoint degenerates to a line or a point.
	CubicLineOrPoint
)

// calcDotCrossCubic computes p0 . (p1 x p2), assuming the third component of each point is 1.
func calcDotCrossCubic(p0, p1, p2 Point) float64 {
	xComp := float64(p0.X) * (float64(p1.Y) - float64(p2.Y))
	yComp := float64(p0.Y) * (float64(p2.X) - float64(p1.X))
	wComp := float64(p1.X)*float64(p2.Y) - float64(p1.Y)*float64(p2.X)
	return xComp + yComp + wComp
}

// previousInversePow2 returns a positive power of two that, when multiplied by n, shifts the exponent of n so its
// magnitude falls in [1, 2). Returns 2^1023 if abs(n) < 2^-1022 (including 0), and NaN if n is Inf or NaN.
func previousInversePow2(n float64) float64 {
	bits := math.Float64bits(n)
	bits = ((uint64(1023) * 2 << 52) + ((uint64(1) << 52) - 1)) - bits // exp = -exp
	bits &= uint64(0x7ff) << 52                                        // mantissa = 1.0, sign = 0
	return math.Float64frombits(bits)
}

// writeCubicInflectionRoots orients the implicit function so positive values are always on the "left" side of the
// curve, then orders t[0]/s[0] <= t[1]/s[1].
func writeCubicInflectionRoots(t0, s0, t1, s1 float64, t, s *[2]float64) {
	t[0] = t0
	s[0] = s0
	t[1] = -math.Copysign(t1, t1*s1)
	s[1] = -math.Abs(s1)
	if math.Copysign(s[1], s[0])*t[0] > -math.Abs(s[0])*t[1] {
		t[0], t[1] = t[1], t[0]
		s[0], s[1] = s[1], s[0]
	}
}

// ClassifyCubic categorizes the cubic P and returns the homogeneous inflection-function roots t[0..1]/s[0..1].
func ClassifyCubic(p [4]Point) (kind CubicType, tRoots, sRoots [2]float64) {
	var t, s [2]float64
	// Find the cubic's inflection function, I = [T^3 -3T^2 3T -1] dot D. (D0 is always 0 for integral cubics.)
	a1 := calcDotCrossCubic(p[0], p[3], p[2])
	a2 := calcDotCrossCubic(p[1], p[0], p[3])
	a3 := calcDotCrossCubic(p[2], p[1], p[0])
	d3 := 3 * a3
	d2 := d3 - a2
	d1 := d2 - a2 + a1
	// Shift the exponents in D so the largest magnitude falls in [1, 2). This protects against overflow while solving
	// for roots and KLM functionals.
	dmax := math.Max(math.Max(math.Abs(d1), math.Abs(d2)), math.Abs(d3))
	norm := previousInversePow2(dmax)
	d1 *= norm
	d2 *= norm
	d3 *= norm
	if d1 != 0 {
		discr := 3*d2*d2 - 4*d1*d3
		if discr > 0 { // serpentine
			q := 3*d2 + math.Copysign(math.Sqrt(3*discr), d2)
			writeCubicInflectionRoots(q, 6*d1, 2*d3, q, &t, &s)
			return CubicSerpentine, t, s
		}
		if discr < 0 { // loop
			q := d2 + math.Copysign(math.Sqrt(-discr), d2)
			writeCubicInflectionRoots(q, 2*d1, 2*(d2*d2-d3*d1), d1*q, &t, &s)
			return CubicLoop, t, s
		}
		// cusp
		writeCubicInflectionRoots(d2, 2*d1, d2, 2*d1, &t, &s)
		return CubicLocalCusp, t, s
	}
	if d2 != 0 { // cusp at T = infinity
		writeCubicInflectionRoots(d3, 3*d2, 1, 0, &t, &s) // t1 = infinity
		return CubicCuspAtInfinity, t, s
	}
	// degenerate
	writeCubicInflectionRoots(1, 0, 1, 0, &t, &s) // t0 = t1 = infinity
	if d3 != 0 {
		return CubicQuadratic, t, s
	}
	return CubicLineOrPoint, t, s
}
