// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import (
	"math"
	"math/rand"
	"testing"
)

// MakeOrthog: PointSideRight is (-y, x), PointSideLeft is (y, -x), matching RotateCW/RotateCCW.
func TestMakeOrthog(t *testing.T) {
	v := Point{X: 3, Y: 4}
	if got := v.MakeOrthog(PointSideRight); got != (Point{X: -4, Y: 3}) {
		t.Fatalf("right orthog %v want (-4, 3)", got)
	}
	if got := v.MakeOrthog(PointSideLeft); got != (Point{X: 4, Y: -3}) {
		t.Fatalf("left orthog %v want (4, -3)", got)
	}
	if v.MakeOrthog(PointSideRight) != v.RotateCW() || v.MakeOrthog(PointSideLeft) != v.RotateCCW() {
		t.Fatal("MakeOrthog disagrees with RotateCW/RotateCCW")
	}
}

// DistanceToLineBetween(Sqd): distance to the infinite line (not the segment), with the squared-distance-to-a fallback
// for degenerate line vectors.
func TestDistanceToLineBetween(t *testing.T) {
	a := Point{X: 0, Y: 0}
	b := Point{X: 10, Y: 0}
	// Beyond the segment's end but 3 above the infinite line: distance is still 3.
	p := Point{X: 25, Y: 3}
	if got := p.DistanceToLineBetweenSqd(a, b); math.Abs(float64(got)-9) > 1e-5 {
		t.Fatalf("line distance sqd %v want 9", got)
	}
	if got := p.DistanceToLineBetween(a, b); math.Abs(float64(got)-3) > 1e-6 {
		t.Fatalf("line distance %v want 3", got)
	}
	// Degenerate line vector: falls back to the squared distance to a.
	if got := p.DistanceToLineBetweenSqd(a, a); math.Abs(float64(got)-(25*25+9)) > 1e-2 {
		t.Fatalf("degenerate line distance sqd %v want %v", got, 25*25+9)
	}
}

// setPointLength is documented as "the double-precision path", which means the scale is applied in double and rounded
// to float32 exactly once. Rounding the scale to float32 before the multiply adds a second rounding step that lands 1
// ulp off the true result for a large fraction of inputs.
func TestSetPointLengthRoundsOnce(t *testing.T) {
	scaleRef := func(x, y, length float32) (float32, float32) {
		xx := float64(x)
		yy := float64(y)
		dscale := float64(length) / math.Sqrt(xx*xx+yy*yy)
		return float32(xx * dscale), float32(yy * dscale)
	}
	rng := rand.New(rand.NewSource(20260727)) //nolint:gosec // deterministic test corpus, not security-sensitive
	for i := range 200000 {
		x := float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(40)-20)))
		y := float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(40)-20)))
		length := float32(1)
		if i%4 == 0 {
			length = float32(rng.NormFloat64() * 16)
		}
		wantX, wantY := scaleRef(x, y, length)
		if !IsFinite(wantX, wantY) || (wantX == 0 && wantY == 0) {
			continue
		}
		p := Point{X: x, Y: y}
		if !p.SetLength(length) {
			t.Fatalf("SetLength(%v, %v, %v) failed", x, y, length)
		}
		if p.X != wantX || p.Y != wantY {
			t.Fatalf("SetLength(%v, %v, %v) = %v, want (%v, %v)", x, y, length, p, wantX, wantY)
		}
	}
	// Normalize/SetNormalize/NormalizeWithLength all share the same path.
	v := Point{X: 0.30000001192092896, Y: 0.4000000059604645}
	wantX, wantY := scaleRef(v.X, v.Y, 1)
	n := v
	if !n.Normalize() || n.X != wantX || n.Y != wantY {
		t.Fatalf("Normalize = %v, want (%v, %v)", n, wantX, wantY)
	}
	var sn Point
	if !sn.SetNormalize(v.X, v.Y) || sn != n {
		t.Fatalf("SetNormalize = %v, want %v", sn, n)
	}
	nl := v
	if got := nl.NormalizeWithLength(); nl != n || math.Abs(float64(got)-0.5) > 1e-7 {
		t.Fatalf("NormalizeWithLength = %v (len %v), want %v (len 0.5)", nl, got, n)
	}
}
