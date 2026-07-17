// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

func TestOneContour(t *testing.T) {
	single := rectPath(0, 0, 10, 10, path.FillWinding)
	if !oneContour(single) {
		t.Error("a single rect should be one contour")
	}
	multi := path.New()
	multi.AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW)
	multi.AddRect(geom.RectLTRB(20, 20, 30, 30), geom.DirectionCW)
	if oneContour(multi) {
		t.Error("two rects should not be one contour")
	}
	if !oneContour(path.New()) {
		t.Error("an empty path should be one contour")
	}
}

func TestReversePath(t *testing.T) {
	// A CW rect reverses to the same closed region with the point order flipped.
	p := rectPath(0, 0, 10, 10, path.FillWinding)
	rev := reversePath(p)
	if rev.ComputeFirstDirection() != path.FirstDirectionCCW {
		t.Errorf("reversed CW rect direction = %v, want CCW", rev.ComputeFirstDirection())
	}
	// The reversal preserves the closed region (area 100 either way).
	rev.SetFillType(path.FillEvenOdd)
	checkArea(t, "reversed rect", rev, 100, 0.01)
	if !rev.Contains(5, 5) || rev.Contains(20, 20) {
		t.Error("reversed rect membership wrong")
	}
}

// TestFixWindingOneContour exercises fixWinding's one-contour fast path: a simplified even-odd single contour becomes
// winding fill oriented CCW (reversing a CW operand, leaving a CCW one), with the region preserved.
func TestFixWindingOneContour(t *testing.T) {
	for _, tc := range []struct {
		name string
		dir  geom.PathDirection
	}{
		{name: "cw", dir: geom.DirectionCW},
		{name: "ccw", dir: geom.DirectionCCW},
	} {
		p := path.New().AddRect(geom.RectLTRB(0, 0, 10, 10), tc.dir)
		p.SetFillType(path.FillEvenOdd)
		wound, ok := fixWinding(p)
		if !ok {
			t.Fatalf("%s: fixWinding failed", tc.name)
		}
		if wound.FillType() != path.FillWinding {
			t.Errorf("%s: fill = %d, want winding", tc.name, wound.FillType())
		}
		// fixWinding always leaves a one-contour path CCW (winding-positive outer orientation).
		if wound.ComputeFirstDirection() != path.FirstDirectionCCW {
			t.Errorf("%s: direction = %v, want CCW", tc.name, wound.ComputeFirstDirection())
		}
		checkArea(t, "fixWinding one-contour "+tc.name, wound, 100, 0.01)
		if !wound.Contains(5, 5) || wound.Contains(20, 20) {
			t.Errorf("%s: membership wrong", tc.name)
		}
	}

	// Inverse even-odd maps to inverse winding, keeping the inverse-ness.
	inv := path.New().AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW)
	inv.SetFillType(path.FillInverseEvenOdd)
	wound, ok := fixWinding(inv)
	if !ok {
		t.Fatal("inverse fixWinding failed")
	}
	if wound.FillType() != path.FillInverseWinding {
		t.Errorf("inverse fill = %d, want inverse winding", wound.FillType())
	}
}

// TestFixWindingMultiContour exercises fixWinding's edge-built path (more than one contour): a simplified even-odd
// annulus (outer square with a square hole) must convert to winding form that fills the same region — the ring filled,
// the hole empty. This is the reversal the fix-winding ray cast decides per contour.
func TestFixWindingMultiContour(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(0, 0, 20, 20), geom.DirectionCW)
	p.AddRect(geom.RectLTRB(5, 5, 15, 15), geom.DirectionCW) // same direction: even-odd makes it a hole
	p.SetFillType(path.FillEvenOdd)
	if oneContour(p) {
		t.Fatal("annulus should not be one contour (must take the edge-build path)")
	}
	// The even-odd source fills the ring but not the hole.
	if !p.Contains(2, 2) || p.Contains(10, 10) {
		t.Fatal("even-odd annulus setup wrong")
	}
	wound, ok := fixWinding(p)
	if !ok {
		t.Fatal("fixWinding failed on annulus")
	}
	if wound.FillType() != path.FillWinding {
		t.Errorf("fill = %d, want winding", wound.FillType())
	}
	// Under the winding rule the region is preserved: ring filled, hole empty, outside empty.
	for _, s := range []struct {
		x, y float32
		want bool
	}{
		{x: 2, y: 2, want: true}, {x: 10, y: 2, want: true}, {x: 2, y: 10, want: true}, // ring
		{x: 10, y: 10, want: false}, // hole
		{x: 25, y: 25, want: false}, // outside
	} {
		if got := wound.Contains(s.x, s.y); got != s.want {
			t.Errorf("wound.Contains(%v,%v) = %v, want %v", s.x, s.y, got, s.want)
		}
	}
	checkArea(t, "fixWinding annulus", wound, 300, 0.05) // 400 - 100
}

// TestResolveOverlappingUnion is the regression that motivates the all-union optimization: two overlapping union
// operands must reinforce (union), not cancel. A naive even-odd sum of the two simplified operands would leave the
// overlap empty; Builder.Resolve avoids that by fix-winding each operand before summing.
func TestResolveOverlappingUnion(t *testing.T) {
	// rect1 = [0,10]x[0,10], rect2 = [5,15]x[5,15]; overlap [5,10]x[5,10]. Union is an L, area 175.
	var b Builder
	b.Add(rectPath(0, 0, 10, 10, path.FillWinding), Union)
	b.Add(rectPath(5, 5, 15, 15, path.FillWinding), Union)
	result, ok := b.Resolve()
	if !ok {
		t.Fatal("overlapping union resolve failed")
	}
	if result.FillType() != path.FillEvenOdd {
		t.Errorf("fill = %d, want even-odd", result.FillType())
	}
	if n := contourCount(result); n != 1 {
		t.Errorf("contours = %d, want 1", n)
	}
	checkArea(t, "overlapping union", result, 175, 0.05)
	for _, s := range []struct {
		x, y float32
		want bool
	}{
		{x: 2, y: 2, want: true},    // rect1 only
		{x: 12, y: 12, want: true},  // rect2 only
		{x: 7, y: 7, want: true},    // overlap — must be filled (the reinforcement)
		{x: 12, y: 2, want: false},  // neither
		{x: 2, y: 12, want: false},  // neither
		{x: 20, y: 20, want: false}, // outside
	} {
		if got := result.Contains(s.x, s.y); got != s.want {
			t.Errorf("result.Contains(%v,%v) = %v, want %v (union reinforcement)", s.x, s.y, got, s.want)
		}
	}
}

// TestResolveUnionMixedDirections unions overlapping operands of opposite winding directions. The allUnion gate does
// not reverse operands (Simplify normalizes orientation), so the union region must still be correct.
func TestResolveUnionMixedDirections(t *testing.T) {
	var b Builder
	b.Add(rectPath(0, 0, 10, 10, path.FillWinding), Union) // CW via rectPath
	cc := path.New().AddRect(geom.RectLTRB(5, 5, 15, 15), geom.DirectionCCW)
	b.Add(cc, Union)
	result, ok := b.Resolve()
	if !ok {
		t.Fatal("mixed-direction union resolve failed")
	}
	checkArea(t, "mixed-direction union", result, 175, 0.05)
	if !result.Contains(7, 7) {
		t.Error("mixed-direction union overlap must be filled")
	}
}
