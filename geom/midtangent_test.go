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

// midTangentDirection returns normalize(tan0) + normalize(-tan1)-style bisector target: the direction the curve's
// tangent must have at the midtangent, i.e. the bisector of the normalized endpoint tangents.
func midTangentDirection(tan0, tan1 Point) Point {
	n0 := tan0
	n0.Normalize()
	n1 := tan1
	n1.Normalize()
	return Point{X: n0.X + n1.X, Y: n0.Y + n1.Y}
}

func TestFindQuadMidTangent(t *testing.T) {
	// A symmetric quad has its midtangent at exactly T=0.5.
	sym := []Point{{X: 0, Y: 0}, {X: 5, Y: 8}, {X: 10, Y: 0}}
	if got := FindQuadMidTangent(sym); got != 0.5 {
		t.Fatalf("symmetric quad midtangent = %v, want 0.5", got)
	}

	// An asymmetric quad: the tangent at the returned T must be parallel to the bisector of the endpoint tangents.
	quad := []Point{{X: 0, Y: 0}, {X: 8, Y: 2}, {X: 10, Y: 10}}
	tv := FindQuadMidTangent(quad)
	if !(tv > 0 && tv < 1) {
		t.Fatalf("asymmetric quad midtangent = %v, want in (0,1)", tv)
	}
	// Quad'(T) ∝ (1-T)*(p1-p0) + T*(p2-p1).
	tan := Point{
		X: (1-tv)*(quad[1].X-quad[0].X) + tv*(quad[2].X-quad[1].X),
		Y: (1-tv)*(quad[1].Y-quad[0].Y) + tv*(quad[2].Y-quad[1].Y),
	}
	tan.Normalize()
	want := midTangentDirection(
		Point{X: quad[1].X - quad[0].X, Y: quad[1].Y - quad[0].Y},
		Point{X: quad[2].X - quad[1].X, Y: quad[2].Y - quad[1].Y},
	)
	want.Normalize()
	if cross := tan.Cross(want); ScalarAbs(cross) > 1e-4 {
		t.Fatalf("midtangent tangent %v not parallel to bisector %v (cross %v)", tan, want, cross)
	}

	// A flat quad with a 180-degree turnaround: the cusp is at the apex of the (degenerate) parabola, T=0.5 by
	// symmetry, and EvalQuadAt there gives the turnaround point.
	flat := []Point{{X: 0, Y: 0}, {X: 5, Y: 0}, {X: 0, Y: 0}}
	tv = FindQuadMidTangent(flat)
	if tv != 0.5 {
		t.Fatalf("flat quad midtangent = %v, want 0.5", tv)
	}
	if cusp := EvalQuadAt(flat, tv); cusp != (Point{X: 2.5, Y: 0}) {
		t.Fatalf("flat quad cusp = %v, want (2.5, 0)", cusp)
	}

	// A line (no rotation) falls back to T=0.5.
	line := []Point{{X: 0, Y: 0}, {X: 5, Y: 5}, {X: 10, Y: 10}}
	if got := FindQuadMidTangent(line); got != 0.5 {
		t.Fatalf("line quad midtangent = %v, want 0.5", got)
	}
}

func TestConicFindMidTangent(t *testing.T) {
	// A quarter-circle conic is symmetric: midtangent at T=0.5, where the tangent is parallel to the chord's bisector
	// direction (-1, -1)...(1,1) family.
	w := float32(math.Sqrt2 / 2)
	c := MakeConic(Point{X: 0, Y: 1}, Point{X: 1, Y: 1}, Point{X: 1, Y: 0}, w)
	tv := c.FindMidTangent()
	if ScalarAbs(tv-0.5) > 1e-6 {
		t.Fatalf("quarter-circle conic midtangent = %v, want 0.5", tv)
	}
	// The point at T=.5 must be on the unit circle at 45 degrees.
	pt := c.EvalAt(tv)
	want := Point{X: float32(math.Sqrt2 / 2), Y: float32(math.Sqrt2 / 2)}
	if ScalarAbs(pt.X-want.X) > 1e-6 || ScalarAbs(pt.Y-want.Y) > 1e-6 {
		t.Fatalf("conic midtangent point = %v, want %v", pt, want)
	}

	// An asymmetric conic: tangent at T must be parallel to the endpoint-tangent bisector.
	c = MakeConic(Point{X: 0, Y: 0}, Point{X: 6, Y: 1}, Point{X: 8, Y: 9}, 1.3)
	tv = c.FindMidTangent()
	if !(tv > 0 && tv < 1) {
		t.Fatalf("asymmetric conic midtangent = %v, want in (0,1)", tv)
	}
	tan := c.EvalTangentAt(tv)
	tan.Normalize()
	want = midTangentDirection(
		Point{X: c.Pts[1].X - c.Pts[0].X, Y: c.Pts[1].Y - c.Pts[0].Y},
		Point{X: c.Pts[2].X - c.Pts[1].X, Y: c.Pts[2].Y - c.Pts[1].Y},
	)
	want.Normalize()
	if cross := tan.Cross(want); ScalarAbs(cross) > 1e-4 {
		t.Fatalf("conic midtangent tangent %v not parallel to bisector %v (cross %v)", tan, want,
			cross)
	}

	// A flat conic with a 180-degree turnaround: the cusp point is between the endpoints and the apex, with a
	// well-defined midtangent.
	c = MakeConic(Point{X: 0, Y: 0}, Point{X: 4, Y: 0}, Point{X: 0, Y: 0}, 0.75)
	tv = c.FindMidTangent()
	if ScalarAbs(tv-0.5) > 1e-6 {
		t.Fatalf("flat conic midtangent = %v, want 0.5", tv)
	}
}

func TestFindBisector(t *testing.T) {
	// Vectors within 90 degrees: the bisector of the originals.
	b := FindBisector(Point{X: 1, Y: 0}, Point{X: 0, Y: 1})
	b.Normalize()
	if ScalarAbs(b.X-b.Y) > 1e-6 || b.X <= 0 {
		t.Fatalf("bisector of +x,+y = %v, want 45 degrees", b)
	}

	// Vectors more than 90 degrees apart route through the interior normals but must still bisect: (1,0) and (-1,1) =>
	// the bisector direction is 67.5 degrees.
	b = FindBisector(Point{X: 1, Y: 0}, Point{X: -1, Y: 1})
	b.Normalize()
	wantAngle := (0.0 + 135.0) / 2 * math.Pi / 180
	got := math.Atan2(float64(b.Y), float64(b.X))
	if math.Abs(got-wantAngle) > 1e-5 {
		t.Fatalf("bisector angle = %v rad, want %v", got, wantAngle)
	}

	// Mirror case below -90 degrees.
	b = FindBisector(Point{X: 1, Y: 0}, Point{X: -1, Y: -1})
	b.Normalize()
	got = math.Atan2(float64(b.Y), float64(b.X))
	if math.Abs(got+wantAngle) > 1e-5 {
		t.Fatalf("bisector angle = %v rad, want %v", got, -wantAngle)
	}
}
