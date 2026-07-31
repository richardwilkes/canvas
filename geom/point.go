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

// Point is a 2D float32 coordinate. It doubles as a vector.
type Point struct {
	X float32
	Y float32
}

// Pt returns a Point.
func Pt(x, y float32) Point {
	return Point{X: x, Y: y}
}

// Add returns p + q.
func (p Point) Add(q Point) Point {
	return Point{X: p.X + q.X, Y: p.Y + q.Y}
}

// Sub returns p - q.
func (p Point) Sub(q Point) Point {
	return Point{X: p.X - q.X, Y: p.Y - q.Y}
}

// Scaled returns p with both components multiplied by scale.
func (p Point) Scaled(scale float32) Point {
	return Point{X: p.X * scale, Y: p.Y * scale}
}

// Negated returns -p.
func (p Point) Negated() Point {
	return Point{X: -p.X, Y: -p.Y}
}

// IsZero reports whether both components are zero.
func (p Point) IsZero() bool {
	return p.X == 0 && p.Y == 0
}

// IsFinite reports whether both components are finite.
func (p Point) IsFinite() bool {
	return IsFinite(p.X, p.Y)
}

// Dot returns the dot product p . q. The explicit float32 conversions force both products to round, forbidding the
// compiler from fusing a multiply into the add (Go fuses on arm64): callers branch on the sign of near-zero dot
// products, and fusion would move results across zero unpredictably per platform. The unfused evaluation is used
// everywhere so output is identical on every platform.
func (p Point) Dot(q Point) float32 {
	return float32(p.X*q.X) + float32(p.Y*q.Y)
}

// Cross returns the 2D cross product p x q. Unfused for the same reason as Dot — notably Cross(u, u) must be exactly
// zero or BuildUnitArc's coincident-vector check misfires (caught by the oracle probes).
func (p Point) Cross(q Point) float32 {
	return float32(p.X*q.Y) - float32(p.Y*q.X)
}

// Length returns the vector's length, computed in float32, falling back to double math when the squared magnitude
// overflows to infinity. The squared magnitude comes from LengthSqd rather than an inline p.X*p.X + p.Y*p.Y so it is
// the same unfused value Dot produces: the inline form fuses on arm64, which would make the length platform-dependent
// and let it disagree with LengthSqd() for the same point, feeding the near-zero tolerance test in
// Matrix.DecomposeScale and the scale > 0 test in ComputeResScaleForStroking.
func (p Point) Length() float32 {
	mag2 := p.LengthSqd()
	if IsFinite(mag2) {
		return float32(math.Sqrt(float64(mag2)))
	}
	xx := float64(p.X)
	yy := float64(p.Y)
	return DoubleToScalar(math.Sqrt(xx*xx + yy*yy))
}

// LengthSqd returns the vector's squared length.
func (p Point) LengthSqd() float32 {
	return p.Dot(p)
}

// DistanceTo returns the distance between p and q.
func (p Point) DistanceTo(q Point) float32 {
	return q.Sub(p).Length()
}

// setPointLength scales (x, y) to the requested length, using the double-precision path. Returns the zero point and
// false if the vector cannot be normalized (zero or non-finite input). When orig is non-nil, it receives the original
// length.
func setPointLength(pt *Point, x, y, length float32, orig *float32) bool {
	xx := float64(x)
	yy := float64(y)
	dmag := math.Sqrt(xx*xx + yy*yy)
	dscale := float64(length) / dmag
	// Scale in double and round once. Rounding dscale to float32 first and then multiplying costs an extra rounding
	// step, which lands 1 ulp away from the true double-precision result for a large fraction of inputs.
	x = DoubleToScalar(xx * dscale)
	y = DoubleToScalar(yy * dscale)
	if !IsFinite(x, y) || (x == 0 && y == 0) {
		*pt = Point{}
		return false
	}
	if orig != nil {
		*orig = DoubleToScalar(dmag)
	}
	*pt = Point{X: x, Y: y}
	return true
}

// SetLength scales p to the given length, returning false (and setting p to the origin) if p is zero-length or
// non-finite.
func (p *Point) SetLength(length float32) bool {
	return setPointLength(p, p.X, p.Y, length, nil)
}

// Normalize scales p to unit length, returning false (and setting p to the origin) on failure.
func (p *Point) Normalize() bool {
	return setPointLength(p, p.X, p.Y, 1, nil)
}

// NormalizeWithLength normalizes p and returns the original length, or 0 on failure.
func (p *Point) NormalizeWithLength() float32 {
	var mag float32
	if setPointLength(p, p.X, p.Y, 1, &mag) {
		return mag
	}
	return 0
}

// EqualsWithinTolerance reports whether both coordinate deltas are within the default nearly-zero tolerance.
func (p Point) EqualsWithinTolerance(q Point) bool {
	return ScalarNearlyZero(p.X-q.X) && ScalarNearlyZero(p.Y-q.Y)
}

// RotateCW rotates the vector a quarter turn clockwise in this package's y-down coordinate system: (x, y) -> (-y, x).
func (p Point) RotateCW() Point {
	return Point{X: -p.Y, Y: p.X}
}

// RotateCCW rotates the vector a quarter turn counter-clockwise: (x, y) -> (y, -x).
func (p Point) RotateCCW() Point {
	return Point{X: p.Y, Y: -p.X}
}

// PointSide identifies which side of a vector a point falls on.
type PointSide int8

// PointSide values.
const (
	PointSideLeft  PointSide = -1
	PointSideOn    PointSide = 0
	PointSideRight PointSide = 1
)

// MakeOrthog returns the vector rotated a quarter turn to the given side ((-y, x) for PointSideRight, (y, -x) for
// PointSideLeft).
func (p Point) MakeOrthog(side PointSide) Point {
	if side == PointSideRight {
		return Point{X: -p.Y, Y: p.X}
	}
	return Point{X: p.Y, Y: -p.X}
}

// DistanceToLineBetweenSqd returns the squared distance from p to the infinite line through a and b, or the squared
// distance to a when the line vector is degenerate.
func (p Point) DistanceToLineBetweenSqd(a, b Point) float32 {
	u := b.Sub(a)
	v := p.Sub(a)
	uLengthSqd := u.LengthSqd()
	det := u.Cross(v)
	temp := IEEEFloatDivide(det, uLengthSqd)
	temp *= det
	// It's possible we have a degenerate line vector, or we're so far away it looks degenerate. In this case, return
	// squared distance to point a.
	if !IsFinite(temp) {
		return v.LengthSqd()
	}
	return temp
}

// DistanceToLineBetween returns the distance from p to the infinite line through a and b.
func (p Point) DistanceToLineBetween(a, b Point) float32 {
	return ScalarSqrt(p.DistanceToLineBetweenSqd(a, b))
}

// IPoint is a 2D int32 coordinate.
type IPoint struct {
	X int32
	Y int32
}

// Point3 is a 3D float32 coordinate.
type Point3 struct {
	X float32
	Y float32
	Z float32
}

// CanNormalize reports whether the vector (dx, dy) is finite and not zero-length.
func CanNormalize(dx, dy float32) bool {
	return IsFinite(dx, dy) && (dx != 0 || dy != 0)
}

// SetNormalize sets p to the unit vector in the direction (x, y), returning false (and setting p to the origin) if that
// cannot be done.
func (p *Point) SetNormalize(x, y float32) bool {
	return setPointLength(p, x, y, 1, nil)
}

// DistanceToSqd returns the squared distance between p and a. Unfused like Dot so results are identical across
// platforms (the stroker's subdivision decisions compare these against tolerances).
func (p Point) DistanceToSqd(a Point) float32 {
	dx := p.X - a.X
	dy := p.Y - a.Y
	return float32(dx*dx) + float32(dy*dy)
}

// EqualsWithinToleranceTol reports whether both coordinate deltas are within the given tolerance.
func (p Point) EqualsWithinToleranceTol(q Point, tol float32) bool {
	return ScalarNearlyZeroTolerance(p.X-q.X, tol) && ScalarNearlyZeroTolerance(p.Y-q.Y, tol)
}
