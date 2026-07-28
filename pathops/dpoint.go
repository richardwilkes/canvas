// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Double-precision 2D vector and point primitives used throughout the pathops geometry, since the boolean-operation
// math needs more precision than the float32 host types (geom.Point) provide. Covers the members the line intersection
// layer reaches plus the curve-dependent free functions (dPointMid, dPointWayRoughlyEqual, …) the quad/cubic layers
// use.

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// dVector is a double-precision 2D vector.
type dVector struct {
	x, y float64
}

// cross returns the 2D cross product (scalar z-component) of v and a.
func (v dVector) cross(a dVector) float64 { return v.x*a.y - v.y*a.x }

// crossCheck returns the cross product of v and a, but treats nearly-coincident directions (within 16 ULPs in float32)
// as exactly parallel, returning 0 instead of a tiny nonzero value.
func (v dVector) crossCheck(a dVector) float64 {
	xy := v.x * a.y
	yx := v.y * a.x
	if almostEqualUlps(xy, yx) {
		return 0
	}
	return xy - yx
}

// crossNoNormalCheck is like crossCheck but its ULPs comparison is tolerant of denormalized operands.
func (v dVector) crossNoNormalCheck(a dVector) float64 {
	xy := v.x * a.y
	yx := v.y * a.x
	if almostEqualUlpsNoNormalCheck(xy, yx) {
		return 0
	}
	return xy - yx
}

// dot returns the dot product of v and a.
func (v dVector) dot(a dVector) float64 { return v.x*a.x + v.y*a.y }

// length returns the Euclidean length of v.
func (v dVector) length() float64 { return math.Sqrt(v.lengthSquared()) }

// lengthSquared returns the squared Euclidean length of v.
func (v dVector) lengthSquared() float64 { return v.x*v.x + v.y*v.y }

// normalize scales v to unit length using an ordinary IEEE divide, so a zero-length vector normalizes to NaN and an
// infinite-length vector normalizes to zero rather than trapping.
func (v *dVector) normalize() {
	inverseLength := ieeeDoubleDivide(1, v.length())
	v.x *= inverseLength
	v.y *= inverseLength
}

// isFinite reports whether both components of v are finite.
func (v dVector) isFinite() bool { return dIsFinite(v.x, v.y) }

// asVector narrows v to the float32 host vector type.
func (v dVector) asVector() geom.Point {
	return geom.Point{X: float32(v.x), Y: float32(v.y)}
}

// dPoint is a double-precision 2D point.
type dPoint struct {
	x, y float64
}

// sub returns the vector from b to the receiver.
func (p dPoint) sub(b dPoint) dVector { return dVector{x: p.x - b.x, y: p.y - b.y} }

// plusVector returns the point offset by v.
func (p dPoint) plusVector(v dVector) dPoint { return dPoint{x: p.x + v.x, y: p.y + v.y} }

// equals reports whether p and b have identical coordinates.
func (p dPoint) equals(b dPoint) bool { return p.x == b.x && p.y == b.y }

// set widens p from the float32 host point.
func (p *dPoint) set(pt geom.Point) {
	p.x = float64(pt.X)
	p.y = float64(pt.Y)
}

// asPoint narrows p to the float32 host point.
func (p dPoint) asPoint() geom.Point {
	return geom.Point{X: float32(p.x), Y: float32(p.y)}
}

// distance returns the Euclidean distance between p and a.
func (p dPoint) distance(a dPoint) float64 { return p.sub(a).length() }

// distanceSquared returns the squared Euclidean distance between p and a.
func (p dPoint) distanceSquared(a dPoint) float64 { return p.sub(a).lengthSquared() }

// dPointMid returns the midpoint of a and b.
func dPointMid(a, b dPoint) dPoint { return dPoint{x: (a.x + b.x) / 2, y: (a.y + b.y) / 2} }

// approximatelyEqual reports whether p and a are the same point, using a cascade of increasingly loose tests
// (component-wise approximate equality, then rough ULPs equality, then a final ULPs-based distance check) that accounts
// for the magnitude of the coordinates.
func (p dPoint) approximatelyEqual(a dPoint) bool {
	if approximatelyEqual(p.x, a.x) && approximatelyEqual(p.y, a.y) {
		return true
	}
	if !roughlyEqualUlps(p.x, a.x) || !roughlyEqualUlps(p.y, a.y) {
		return false
	}
	dist := p.distance(a)
	tiniest := math.Min(math.Min(math.Min(p.x, a.x), p.y), a.y)
	largest := math.Max(math.Max(math.Max(p.x, a.x), p.y), a.y)
	largest = math.Max(largest, -tiniest)
	return almostPequalUlps(largest, largest+dist)
}

// approximatelyDEqual is like approximatelyEqual, but its final distance check uses the looser 16-ULP almostDequalUlps
// rather than approximatelyEqual's 8-ULP almostPequalUlps.
func (p dPoint) approximatelyDEqual(a dPoint) bool {
	if approximatelyEqual(p.x, a.x) && approximatelyEqual(p.y, a.y) {
		return true
	}
	if !roughlyEqualUlps(p.x, a.x) || !roughlyEqualUlps(p.y, a.y) {
		return false
	}
	dist := p.distance(a)
	tiniest := math.Min(math.Min(math.Min(p.x, a.x), p.y), a.y)
	largest := math.Max(math.Max(math.Max(p.x, a.x), p.y), a.y)
	largest = math.Max(largest, -tiniest)
	return almostDequalUlps(largest, largest+dist)
}

// approximatelyEqualPt is approximatelyEqual for a float32 host point, widening a before comparing.
func (p dPoint) approximatelyEqualPt(a geom.Point) bool {
	var d dPoint
	d.set(a)
	return p.approximatelyEqual(d)
}

// approximatelyZero reports whether p is approximately the origin.
func (p dPoint) approximatelyZero() bool {
	return approximatelyZero(p.x) && approximatelyZero(p.y)
}

// dPointsApproximatelyEqual compares two float32 host points with the same approximate/rough/ULPs cascade
// approximatelyEqual uses, but with the min/max magnitude computed in float32 before the final ULPs distance check
// (matching the precision at which the host points actually live).
func dPointsApproximatelyEqual(a, b geom.Point) bool {
	if approximatelyEqual(float64(a.X), float64(b.X)) && approximatelyEqual(float64(a.Y), float64(b.Y)) {
		return true
	}
	if !roughlyEqualUlps(float64(a.X), float64(b.X)) || !roughlyEqualUlps(float64(a.Y), float64(b.Y)) {
		return false
	}
	var dA, dB dPoint
	dA.set(a)
	dB.set(b)
	dist := dA.distance(dB)
	tiniest := minF32(minF32(minF32(a.X, b.X), a.Y), b.Y)
	largest := maxF32(maxF32(maxF32(a.X, b.X), a.Y), b.Y)
	largest = maxF32(largest, -tiniest)
	return almostDequalUlps(float64(largest), float64(largest)+dist)
}

// dPointsRoughlyEqual is a looser (rough-ULPs) point equality than dPointsApproximatelyEqual, with the min/max
// magnitude computed in float32.
func dPointsRoughlyEqual(a, b geom.Point) bool {
	if !roughlyEqualUlps(float64(a.X), float64(b.X)) && !roughlyEqualUlps(float64(a.Y), float64(b.Y)) {
		return false
	}
	var dA, dB dPoint
	dA.set(a)
	dB.set(b)
	dist := dA.distance(dB)
	tiniest := minF32(minF32(minF32(a.X, b.X), a.Y), b.Y)
	largest := maxF32(maxF32(maxF32(a.X, b.X), a.Y), b.Y)
	largest = maxF32(largest, -tiniest)
	return roughlyEqualUlps(float64(largest), float64(largest)+dist)
}

// dPointWayRoughlyEqual is a very lightweight check that is only meaningful for inequality (a true result does not
// guarantee equality). The magnitude and difference are kept in float32 throughout, then compared at the rough epsilon.
func dPointWayRoughlyEqual(a, b geom.Point) bool {
	largestNumber := maxF32(absF32(a.X), maxF32(absF32(a.Y), maxF32(absF32(b.X), absF32(b.Y))))
	largestDiff := maxF32(absF32(a.X-b.X), absF32(a.Y-b.Y))
	return roughlyZeroWhenComparedTo(float64(largestDiff), float64(largestNumber))
}

// minF32/maxF32 are min/max over float32 (used where an intermediate value must be kept in float32 precision rather
// than widened to float64).
func minF32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxF32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// roughlyEqual reports whether p and a are the same point at the rough epsilon, falling back to a magnitude-scaled ULPs
// distance check when the component-wise rough comparison fails.
func (p dPoint) roughlyEqual(a dPoint) bool {
	if roughlyEqual(p.x, a.x) && roughlyEqual(p.y, a.y) {
		return true
	}
	dist := p.distance(a)
	tiniest := math.Min(math.Min(math.Min(p.x, a.x), p.y), a.y)
	largest := math.Max(math.Max(math.Max(p.x, a.x), p.y), a.y)
	largest = math.Max(largest, -tiniest)
	return roughlyEqualUlps(largest, largest+dist)
}
