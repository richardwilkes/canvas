// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// TestPoly2MapsUnitYSegment pins the half-transform poly2 actually builds: it maps (0,0) to p0 and (0,1) to p1 — the unit
// *y* segment, not the (0,0)-(1,0) x segment — with (1,0) landing on p0 plus the perpendicular of p1-p0. polyToPoly2
// composes two of these so the choice cancels; this test exists so the two halves are never "fixed" independently.
func TestPoly2MapsUnitYSegment(t *testing.T) {
	for _, tc := range []struct{ p0, p1 geom.Point }{
		{p0: geom.Point{}, p1: geom.Point{X: 1}},
		{p0: geom.Point{X: 3, Y: -2}, p1: geom.Point{X: 3, Y: 5}},
		{p0: geom.Point{X: -7, Y: 11}, p1: geom.Point{X: 2, Y: 4}},
	} {
		m := poly2(tc.p0, tc.p1)
		if got := m.MapXY(0, 0); got != tc.p0 {
			t.Errorf("poly2(%v,%v): (0,0) mapped to %v, want p0", tc.p0, tc.p1, got)
		}
		if got := m.MapXY(0, 1); got != tc.p1 {
			t.Errorf("poly2(%v,%v): (0,1) mapped to %v, want p1", tc.p0, tc.p1, got)
		}
		// The perpendicular: (1,0) goes to p0 + rot(p1-p0), which is p1 only for the degenerate zero-length pair.
		wantPerp := geom.Point{X: tc.p0.X + (tc.p1.Y - tc.p0.Y), Y: tc.p0.Y - (tc.p1.X - tc.p0.X)}
		if got := m.MapXY(1, 0); got != wantPerp {
			t.Errorf("poly2(%v,%v): (1,0) mapped to %v, want %v", tc.p0, tc.p1, got, wantPerp)
		}
	}
}

// TestPolyToPoly2MapsBothPoints locks polyToPoly2's documented contract — the composition of two poly2 halves really
// does send src0 to dst0 and src1 to dst1 — including the two pairs the conical code relies on (focalData.set's
// focal-point normalization and mapToUnitX).
func TestPolyToPoly2MapsBothPoints(t *testing.T) {
	unitX := geom.Point{X: 1}
	for _, tc := range []struct {
		name                   string
		src0, src1, dst0, dst1 geom.Point
	}{
		{name: "identity", src0: geom.Point{}, src1: unitX, dst0: geom.Point{}, dst1: unitX},
		{name: "focal-normalize", src0: geom.Point{X: 0.25}, src1: unitX, dst0: geom.Point{}, dst1: unitX},
		{
			name: "map-to-unit-x", src0: geom.Point{X: 10, Y: 20}, src1: geom.Point{X: 40, Y: 60},
			dst0: geom.Point{}, dst1: unitX,
		},
		{
			name: "rotate-and-scale", src0: geom.Point{X: -3, Y: 4}, src1: geom.Point{X: 5, Y: 4},
			dst0: geom.Point{X: 7, Y: 7}, dst1: geom.Point{X: 7, Y: -9},
		},
	} {
		m, ok := polyToPoly2(tc.src0, tc.src1, tc.dst0, tc.dst1)
		if !ok {
			t.Errorf("%s: polyToPoly2 failed", tc.name)
			continue
		}
		if got := m.MapPoint(tc.src0); !nearPoint(got, tc.dst0) {
			t.Errorf("%s: src0 mapped to %v, want %v", tc.name, got, tc.dst0)
		}
		if got := m.MapPoint(tc.src1); !nearPoint(got, tc.dst1) {
			t.Errorf("%s: src1 mapped to %v, want %v", tc.name, got, tc.dst1)
		}
	}
}

// TestPolyToPoly2RejectsDegenerateSrc locks that a zero-length src pair (whose poly2 half is singular) reports failure
// rather than returning a garbage matrix — the path focalData.set turns into a nil shader.
func TestPolyToPoly2RejectsDegenerateSrc(t *testing.T) {
	p := geom.Point{X: 4, Y: 9}
	if _, ok := polyToPoly2(p, p, geom.Point{}, geom.Point{X: 1}); ok {
		t.Fatal("expected polyToPoly2 to fail for a zero-length src pair")
	}
}

func nearPoint(a, b geom.Point) bool { return near(a.X, b.X, 1e-5) && near(a.Y, b.Y, 1e-5) }
