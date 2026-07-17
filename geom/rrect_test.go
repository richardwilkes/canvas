// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import "testing"

func TestRRectTransform(t *testing.T) {
	rr := MakeRRect(RectLTRB(10, 20, 110, 80), 12, 8)

	// Identity returns the rrect unchanged.
	ident := IdentityMatrix()
	got, ok := rr.Transform(&ident)
	if !ok || got != rr {
		t.Errorf("identity transform: ok=%v got=%+v", ok, got)
	}

	// Scale + translate scales the rect and radii per axis.
	var m Matrix
	m.SetScale(2, 3)
	m.PostTranslate(5, -7)
	got, ok = rr.Transform(&m)
	if !ok {
		t.Fatal("scale transform rejected")
	}
	wantRect := RectLTRB(10*2+5, 20*3-7, 110*2+5, 80*3-7)
	if got.Rect != wantRect || got.RadiusX != 24 || got.RadiusY != 24 {
		t.Errorf("scaled: %+v (rx=%v ry=%v), want rect %+v rx=24 ry=24", got.Rect, got.RadiusX,
			got.RadiusY, wantRect)
	}

	// A 90° rotation preserves axis alignment and swaps the radii.
	var rot Matrix
	rot.SetAll(0, -1, 0, 1, 0, 0, 0, 0, 1)
	got, ok = rr.Transform(&rot)
	if !ok {
		t.Fatal("90-degree rotation rejected")
	}
	if got.RadiusX != 8 || got.RadiusY != 12 {
		t.Errorf("rotated radii = (%v, %v), want (8, 12)", got.RadiusX, got.RadiusY)
	}

	// A general rotation is not axis-preserving.
	var rot30 Matrix
	rot30.SetRotate(30)
	if _, ok = rr.Transform(&rot30); ok {
		t.Error("30-degree rotation accepted")
	}

	// Rect- and oval-type rrects stay their types.
	rect := MakeRRect(RectLTRB(0, 0, 10, 10), 0, 0)
	if got, ok = rect.Transform(&m); !ok || got.Type != RRectRect {
		t.Errorf("rect transform: ok=%v type=%d", ok, got.Type)
	}
	oval := MakeRRect(RectLTRB(0, 0, 10, 10), 5, 5)
	if got, ok = oval.Transform(&m); !ok || got.Type != RRectOval {
		t.Errorf("oval transform: ok=%v type=%d", ok, got.Type)
	}
}

func TestMatrixMapRadius(t *testing.T) {
	var m Matrix
	m.SetScale(2, 8)
	// Geometric mean of the mapped axis lengths: sqrt(2r * 8r) = 4r.
	if got := m.MapRadius(3); got != 12 {
		t.Errorf("MapRadius(3) under scale(2,8) = %v, want 12", got)
	}
	ident := IdentityMatrix()
	if got := ident.MapRadius(5); got != 5 {
		t.Errorf("identity MapRadius(5) = %v", got)
	}
}
