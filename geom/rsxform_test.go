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

// TestRSXformToQuadMatchesMatrix pins ToQuad/ToTriStrip against the documented slow path: mapping the corners of
// [0,0,w,h] through the xform's 3x3 matrix, for identity, rotate+scale, and anchored-rotation xforms.
func TestRSXformToQuadMatchesMatrix(t *testing.T) {
	xforms := []RSXform{
		MakeRSXform(1, 0, 0, 0),
		MakeRSXform(0, 1, 10, -3),
		MakeRSXform(0.75, -1.25, 100, 42.5),
		RSXformFromRadians(2, math.Pi/3, 20, 30, 8, 8),
		RSXformFromRadians(0.5, -math.Pi/7, -4, 9, 0, 16),
	}
	const w, h = 24, 16
	for i, x := range xforms {
		m := x.Matrix()
		want := [4]Point{
			m.MapXY(0, 0), m.MapXY(w, 0), m.MapXY(w, h), m.MapXY(0, h),
		}
		var quad [4]Point
		x.ToQuad(w, h, &quad)
		for c := range quad {
			if !nearlyEqualPoint(quad[c], want[c]) {
				t.Errorf("xform %d ToQuad[%d] = %v, want %v", i, c, quad[c], want[c])
			}
		}
		var strip [4]Point
		x.ToTriStrip(w, h, &strip)
		stripWant := [4]Point{want[0], want[3], want[1], want[2]}
		for c := range strip {
			if !nearlyEqualPoint(strip[c], stripWant[c]) {
				t.Errorf("xform %d ToTriStrip[%d] = %v, want %v", i, c, strip[c], stripWant[c])
			}
		}
	}
}

func nearlyEqualPoint(a, b Point) bool {
	const tol = 1e-4
	return ScalarAbs(a.X-b.X) <= tol && ScalarAbs(a.Y-b.Y) <= tol
}

// TestRSXformRectStaysRect pins the axis-alignment predicate.
func TestRSXformRectStaysRect(t *testing.T) {
	if !MakeRSXform(2, 0, 1, 1).RectStaysRect() {
		t.Error("scale-only xform should stay rect")
	}
	if !MakeRSXform(0, 1, 0, 0).RectStaysRect() {
		t.Error("90-degree xform should stay rect")
	}
	if MakeRSXform(1, 0.5, 0, 0).RectStaysRect() {
		t.Error("general rotation should not stay rect")
	}
}

// TestRSXformFromRadians pins the anchor-point algebra: the anchor maps to (tx, ty).
func TestRSXformFromRadians(t *testing.T) {
	const scale, radians, tx, ty, ax, ay = 1.5, 0.7, 33, -12, 5, 9
	x := RSXformFromRadians(scale, radians, tx, ty, ax, ay)
	m := x.Matrix()
	got := m.MapXY(ax, ay)
	if !nearlyEqualPoint(got, Point{X: tx, Y: ty}) {
		t.Errorf("anchor maps to %v, want (%v, %v)", got, float32(tx), float32(ty))
	}
}
