// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Mid-tangent finders for quads and conics, plus the bisector helper they share. The quad case solves its linear
// equation inline; only the conic case needs the quadratic solver. The GPU stroke tessellator chops flat curves with
// 180-degree turnarounds at these mid-tangent points.

package geom

import "math"

// FindBisector returns the bisector of the two vectors. When a and b are more than 90 degrees apart the original
// vectors start canceling each other out (which eventually becomes unstable), so the bisector of their interior
// normals is used instead.
func FindBisector(a, b Point) Point {
	var v0, v1 Point
	switch {
	case a.Dot(b) >= 0:
		// a,b are within +/-90 degrees apart.
		v0, v1 = a, b
	case a.Cross(b) >= 0:
		// a,b are >90 degrees apart. Find the bisector of their interior normals instead.
		v0 = Point{X: -a.Y, Y: +a.X}
		v1 = Point{X: +b.Y, Y: -b.X}
	default:
		// a,b are <-90 degrees apart. Find the bisector of their interior normals instead.
		v0 = Point{X: +a.Y, Y: -a.X}
		v1 = Point{X: -b.Y, Y: +b.X}
	}
	// Return "normalize(v0) + normalize(v1)".
	invLen0 := 1 / ScalarSqrt(v0.X*v0.X+v0.Y*v0.Y)
	invLen1 := 1 / ScalarSqrt(v1.X*v1.X+v1.Y*v1.Y)
	return Point{X: v0.X*invLen0 + v1.X*invLen1, Y: v0.Y*invLen0 + v1.Y*invLen1}
}

// FindQuadMidTangent returns the T value whose tangent angle is halfway between the tangents at p0 and p2.
func FindQuadMidTangent(src []Point) float32 {
	// Tangents point in the direction of increasing T, so tan0 and -tan1 both point toward the midtangent. The bisector
	// of tan0 and -tan1 is orthogonal to the midtangent:
	//
	//	n dot midtangent = 0
	tan0 := Point{X: src[1].X - src[0].X, Y: src[1].Y - src[0].Y}
	tan1 := Point{X: src[2].X - src[1].X, Y: src[2].Y - src[1].Y}
	bisector := FindBisector(tan0, Point{X: -tan1.X, Y: -tan1.Y})

	// The midtangent can be found where (F' dot bisector) = 0:
	//
	//	0 = (F'(T) dot bisector) = |2*T 1| * |p0 - 2*p1 + p2| * |bisector.x|
	//	                                     |-2*p0 + 2*p1  |   |bisector.y|
	//
	//	                  = |2*T 1| * |tan1 - tan0| * |nx|
	//	                              |2*tan0     |   |ny|
	//
	//	                  = 2*T * ((tan1 - tan0) dot bisector) + (2*tan0 dot bisector)
	//
	//	T = (tan0 dot bisector) / ((tan0 - tan1) dot bisector)
	t := IEEEFloatDivide(tan0.Dot(bisector),
		Point{X: tan0.X - tan1.X, Y: tan0.Y - tan1.Y}.Dot(bisector))
	if !(t > 0 && t < 1) { // Use "!(positive_logic)" so T=nan will take this branch.
		t = 0.5 // The quadratic was a line or near-line. Just chop at .5.
	}
	return t
}

// solveQuadraticEquationForMidTangent picks the root of aT^2 + bT + c nearest T=.5, falling back on .5 when the curve
// is a flat line with no rotation or FP precision failed us.
func solveQuadraticEquationForMidTangent(a, b, c float32) float32 {
	discr := b*b - 4*a*c
	// Quadratic formula from Numerical Recipes in C.
	q := -0.5 * (b + float32(math.Copysign(math.Sqrt(float64(discr)), float64(b))))
	// The roots are q/a and c/q. Pick the midtangent closer to T=.5.
	halfNegQA := -0.5 * q * a
	var t float32
	if ScalarAbs(q*q+halfNegQA) < ScalarAbs(a*c+halfNegQA) {
		t = IEEEFloatDivide(q, a)
	} else {
		t = IEEEFloatDivide(c, q)
	}
	if !(t > 0 && t < 1) { // Use "!(positive_logic)" so T=NaN will take this branch.
		t = 0.5
	}
	return t
}

// FindMidTangent returns the T value whose tangent angle is halfway between the tangents at the conic's endpoints.
func (c *Conic) FindMidTangent() float32 {
	// Tangents point in the direction of increasing T, so tan0 and -tan1 both point toward the midtangent. The bisector
	// of tan0 and -tan1 is orthogonal to the midtangent:
	//
	//	bisector dot midtangent = 0
	tan0 := Point{X: c.Pts[1].X - c.Pts[0].X, Y: c.Pts[1].Y - c.Pts[0].Y}
	tan1 := Point{X: c.Pts[2].X - c.Pts[1].X, Y: c.Pts[2].Y - c.Pts[1].Y}
	bisector := FindBisector(tan0, Point{X: -tan1.X, Y: -tan1.Y})

	// Start by finding the tangent function's power basis coefficients. These define a tangent direction (scaled by
	// some uniform value) as:
	//
	//	                                              |T^2|
	//	Tangent_Direction(T) = dx,dy = |A  B  C| * |T  |
	//	                               |.  .  .|   |1  |
	//
	// The derivative of a conic has a cumbersome order-4 denominator. However, this isn't necessary if we are only
	// interested in a vector in the same *direction* as a given tangent line. Since the denominator scales dx and dy
	// uniformly, we can throw it out completely after evaluating the derivative with the standard quotient rule. This
	// leaves us with a simpler quadratic function that we use to find a tangent.
	e02 := Point{X: c.Pts[2].X - c.Pts[0].X, Y: c.Pts[2].Y - c.Pts[0].Y}
	e01 := Point{X: c.Pts[1].X - c.Pts[0].X, Y: c.Pts[1].Y - c.Pts[0].Y}
	pa := Point{X: e02.X * (c.W - 1), Y: e02.Y * (c.W - 1)}
	pb := Point{X: e02.X - e01.X*(c.W*2), Y: e02.Y - e01.Y*(c.W*2)}
	pc := Point{X: e01.X * c.W, Y: e01.Y * c.W}

	// Now solve for "bisector dot midtangent = 0":
	//
	//	                        |T^2|
	//	bisector * |A  B  C| * |T  | = 0
	//	           |.  .  .|   |1  |
	return solveQuadraticEquationForMidTangent(bisector.Dot(pa), bisector.Dot(pb),
		bisector.Dot(pc))
}
