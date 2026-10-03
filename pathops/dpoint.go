// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Double-precision 2D vector and point primitives used throughout the pathops geometry, since the boolean-operation
// math needs more precision than the float32 host type (geom.Point) provides.

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

type dVector struct {
	x, y float64
}

func (v dVector) cross(a dVector) float64 { return v.x*a.y - v.y*a.x }

// crossCheck returns the cross product of v and a, or 0 when its two terms are within 16 float32 ULPs of each other
// (nearly parallel directions).
func (v dVector) crossCheck(a dVector) float64 {
	xy := v.x * a.y
	yx := v.y * a.x
	if almostEqualUlps(xy, yx) {
		return 0
	}
	return xy - yx
}

// crossNoNormalCheck is like crossCheck but without the shortcut that treats two near-zero terms as equal.
func (v dVector) crossNoNormalCheck(a dVector) float64 {
	xy := v.x * a.y
	yx := v.y * a.x
	if almostEqualUlpsNoNormalCheck(xy, yx) {
		return 0
	}
	return xy - yx
}

func (v dVector) dot(a dVector) float64 { return v.x*a.x + v.y*a.y }

func (v dVector) length() float64 { return math.Sqrt(v.lengthSquared()) }

func (v dVector) lengthSquared() float64 { return v.x*v.x + v.y*v.y }

// normalize scales v to unit length. A zero-length vector becomes NaN and one whose length overflows to infinity
// becomes zero.
func (v *dVector) normalize() {
	inverseLength := ieeeDoubleDivide(1, v.length())
	v.x *= inverseLength
	v.y *= inverseLength
}

func (v dVector) isFinite() bool { return dIsFinite(v.x, v.y) }

func (v dVector) asVector() geom.Point {
	return geom.Point{X: float32(v.x), Y: float32(v.y)}
}

type dPoint struct {
	x, y float64
}

func (p dPoint) sub(b dPoint) dVector { return dVector{x: p.x - b.x, y: p.y - b.y} }

func (p dPoint) plusVector(v dVector) dPoint { return dPoint{x: p.x + v.x, y: p.y + v.y} }

func (p dPoint) equals(b dPoint) bool { return p.x == b.x && p.y == b.y }

func (p *dPoint) set(pt geom.Point) {
	p.x = float64(pt.X)
	p.y = float64(pt.Y)
}

func (p dPoint) asPoint() geom.Point {
	return geom.Point{X: float32(p.x), Y: float32(p.y)}
}

func (p dPoint) distance(a dPoint) float64 { return p.sub(a).length() }

func (p dPoint) distanceSquared(a dPoint) float64 { return p.sub(a).lengthSquared() }

func dPointMid(a, b dPoint) dPoint { return dPoint{x: (a.x + b.x) / 2, y: (a.y + b.y) / 2} }

// approximatelyEqual reports whether p and a are the same point: either both components are approximately equal, or
// both are roughly equal in ULPs and the distance between them is within 8 ULPs (almostPequalUlps) of the largest
// coordinate magnitude.
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

// approximatelyDEqual is approximatelyEqual with the distance check loosened to 16 ULPs (almostDequalUlps).
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

func (p dPoint) approximatelyEqualPt(a geom.Point) bool {
	var d dPoint
	d.set(a)
	return p.approximatelyEqual(d)
}

func (p dPoint) approximatelyZero() bool {
	return approximatelyZero(p.x) && approximatelyZero(p.y)
}

// dPointsApproximatelyEqual is approximatelyDEqual for float32 host points, with the min/max magnitude computed in
// float32 (the precision at which the host points live).
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
