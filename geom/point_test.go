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
