// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The double-precision rational-quadratic (conic) primitive. Present are the members reachable from the conic/line
// intersection layer (set/debugSet, ptAtT, the weight, and the conic polynomial eval helpers) and the curve-vs-curve
// members the conic solver consumes (hullIntersects in all three forms, dxdyAtT, conicFindExtrema, subDivide,
// subDivideAC, collapsed, controlsInside, otherPts, the monotonic checks — most delegating to the underlying dQuad
// pts). The conic bounds live in drect.go (setBoundsConic).

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// dConic is a rational quadratic Bezier: three double-precision points plus a scalar weight. The weight is a float32
// and is promoted to double precision in the polynomial evaluations.
type dConic struct {
	pts    dQuad   // fPts
	weight float32 // fWeight
}

// set widens the float32 host points to double precision and stores the weight.
func (c *dConic) set(pts [3]geom.Point, weight float32) {
	c.pts.set(pts)
	c.weight = weight
}

// debugSet stores the double-precision points directly (no float32 rounding), as used by the test corpus.
func (c *dConic) debugSet(pts [3]dPoint, weight float32) {
	c.pts.pts = pts
	c.weight = weight
}

// conicEvalNumerator evaluates the numerator of the rational-quadratic parameterization at t for one coordinate:
// c0/c1/c2 are that coordinate's three control values. The weight promotes to double precision before multiplying, so
// the whole numerator is computed in double.
func conicEvalNumerator(c0, c1, c2 float64, w float32, t float64) float64 {
	src2w := c1 * float64(w)
	c := c0
	a := c2 - 2*src2w + c
	b := 2 * (src2w - c)
	return (a*t+b)*t + c
}

// conicEvalDenominator evaluates the denominator of the rational-quadratic parameterization at t. Note that 2 * (w - 1)
// is computed entirely in float32 before widening to double — the order of operations matters for bit-exact results.
func conicEvalDenominator(w float32, t float64) float64 {
	b := float64(2 * (w - 1))
	c := 1.0
	a := -b
	return (a*t+b)*t + c
}

// ptAtT evaluates the conic at parameter t, dividing the homogeneous numerator by the denominator (both computed in
// double precision).
func (c dConic) ptAtT(t float64) dPoint {
	if t == 0 {
		return c.pts.pts[0]
	}
	if t == 1 {
		return c.pts.pts[2]
	}
	denominator := conicEvalDenominator(c.weight, t)
	return dPoint{
		x: ieeeDoubleDivide(conicEvalNumerator(c.pts.pts[0].x, c.pts.pts[1].x, c.pts.pts[2].x, c.weight, t), denominator),
		y: ieeeDoubleDivide(conicEvalNumerator(c.pts.pts[0].y, c.pts.pts[1].y, c.pts.pts[2].y, c.weight, t), denominator),
	}
}

// collapsed reports whether the conic's control points have degenerated to a single point (delegates to the underlying
// quad).
func (c dConic) collapsed() bool { return c.pts.collapsed() }

// controlsInside reports whether the conic's control point lies inside the hull formed by its endpoints (delegates to
// the underlying quad).
func (c dConic) controlsInside() bool { return c.pts.controlsInside() }

// otherPts returns the two control points other than the one at index oddMan (delegates to the underlying quad).
func (c dConic) otherPts(oddMan int) [2]dPoint { return c.pts.otherPts(oddMan) }

// monotonicInX reports whether the conic's x-coordinate is monotonic along t (delegates to the underlying quad).
func (c dConic) monotonicInX() bool { return c.pts.monotonicInX() }

// monotonicInY reports whether the conic's y-coordinate is monotonic along t (delegates to the underlying quad).
func (c dConic) monotonicInY() bool { return c.pts.monotonicInY() }

// hullIntersectsQuad reports whether the convex hull of the conic's control points (its underlying quad) intersects the
// convex hull of the given quad.
func (c dConic) hullIntersectsQuad(quad dQuad) (sects, isLinear bool) {
	return c.pts.hullIntersects(quad)
}

// hullIntersectsConic reports whether the convex hulls of the two conics' control points intersect.
func (c dConic) hullIntersectsConic(conic dConic) (sects, isLinear bool) {
	return c.pts.hullIntersects(conic.pts)
}

// hullIntersectsCubic reports whether the conic's hull intersects the cubic's hull, dispatching to the cubic's own
// hull-intersection test.
func (c dConic) hullIntersectsCubic(cubic dCubic) (sects, isLinear bool) {
	return cubic.hullIntersectsConic(c)
}

// conicDerivCoeff computes the quadratic coefficients (c0,c1,c2) of the conic's derivative for one coordinate: p0/p1/p2
// are that coordinate's three control values. The weight promotes to double precision before the w*p10 / w*p20
// products.
func conicDerivCoeff(p0, p1, p2 float64, w float32) (c0, c1, c2 float64) {
	p20 := p2 - p0
	p10 := p1 - p0
	wp10 := float64(w) * p10
	c0 = float64(w)*p20 - p20
	c1 = p20 - 2*wp10
	c2 = wp10
	return c0, c1, c2
}

// conicEvalTan evaluates the tangent numerator of one coordinate at t.
func conicEvalTan(p0, p1, p2 float64, w float32, t float64) float64 {
	c0, c1, c2 := conicDerivCoeff(p0, p1, p2, w)
	return t*(t*c0+c1) + c2
}

// conicFindExtrema locates the extremum of one coordinate over t in [0,1]: p0..p2 are that coordinate's three control
// values. Returns 1 (and writes the t value) when a single valid extremum exists, else 0.
func conicFindExtrema(p0, p1, p2 float64, w float32, tValue *float64) int {
	c0, c1, c2 := conicDerivCoeff(p0, p1, p2, w)
	var tValues [2]float64
	// In extreme cases the root solver can return 2 valid roots; that case is treated as failure downstream, so only a
	// single root counts here.
	if quadRootsValidT(c0, c1, c2, tValues[:]) == 1 {
		*tValue = tValues[0]
		return 1
	}
	return 0
}

// dxdyAtT returns the (unnormalized) tangent vector at t.
func (c dConic) dxdyAtT(t float64) dVector {
	result := dVector{
		x: conicEvalTan(c.pts.pts[0].x, c.pts.pts[1].x, c.pts.pts[2].x, c.weight, t),
		y: conicEvalTan(c.pts.pts[0].y, c.pts.pts[1].y, c.pts.pts[2].y, c.weight, t),
	}
	if result.x == 0 && result.y == 0 {
		if zeroOrOne(t) {
			result = c.pts.pts[2].sub(c.pts.pts[0])
		}
		// else: this degenerate case (a zero tangent away from an endpoint) is left unresolved; result stays {0,0}.
	}
	return result
}

// subDivide returns the sub-conic over the parameter range [t1, t2]. The endpoints and the middle control point are
// recovered in homogeneous coordinates; the new weight is bz/sqrt(az*cz).
func (c dConic) subDivide(t1, t2 float64) dConic {
	var ax, ay, az float64
	switch {
	case t1 == 0:
		ax, ay, az = c.pts.pts[0].x, c.pts.pts[0].y, 1
	case t1 != 1:
		ax = conicEvalNumerator(c.pts.pts[0].x, c.pts.pts[1].x, c.pts.pts[2].x, c.weight, t1)
		ay = conicEvalNumerator(c.pts.pts[0].y, c.pts.pts[1].y, c.pts.pts[2].y, c.weight, t1)
		az = conicEvalDenominator(c.weight, t1)
	default:
		ax, ay, az = c.pts.pts[2].x, c.pts.pts[2].y, 1
	}
	midT := (t1 + t2) / 2
	dx := conicEvalNumerator(c.pts.pts[0].x, c.pts.pts[1].x, c.pts.pts[2].x, c.weight, midT)
	dy := conicEvalNumerator(c.pts.pts[0].y, c.pts.pts[1].y, c.pts.pts[2].y, c.weight, midT)
	dz := conicEvalDenominator(c.weight, midT)
	var cx, cy, cz float64
	switch {
	case t2 == 1:
		cx, cy, cz = c.pts.pts[2].x, c.pts.pts[2].y, 1
	case t2 != 0:
		cx = conicEvalNumerator(c.pts.pts[0].x, c.pts.pts[1].x, c.pts.pts[2].x, c.weight, t2)
		cy = conicEvalNumerator(c.pts.pts[0].y, c.pts.pts[1].y, c.pts.pts[2].y, c.weight, t2)
		cz = conicEvalDenominator(c.weight, t2)
	default:
		cx, cy, cz = c.pts.pts[0].x, c.pts.pts[0].y, 1
	}
	bx := 2*dx - (ax+cx)/2
	by := 2*dy - (ay+cy)/2
	bz := 2*dz - (az+cz)/2
	if bz == 0 {
		bz = 1 // if bz is 0, weight is 0, control point has no effect: any value will do
	}
	var dst dConic
	dst.pts.pts[0] = dPoint{x: ax / az, y: ay / az}
	dst.pts.pts[1] = dPoint{x: bx / bz, y: by / bz}
	dst.pts.pts[2] = dPoint{x: cx / cz, y: cy / cz}
	dst.weight = float32(bz / math.Sqrt(az*cz))
	return dst
}

// subDivideAC returns the middle control point and new weight of the sub-conic over [t1, t2]. The caller already has
// the endpoints (a and c) exactly, so they are neither recomputed nor returned; only the interior control point and the
// weight are.
func (c dConic) subDivideAC(t1, t2 float64) (mid dPoint, weight float32) {
	chopped := c.subDivide(t1, t2)
	return chopped.pts.pts[1], chopped.weight
}
