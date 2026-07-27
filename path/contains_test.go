// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// TestCheckOnCurveHorizontalIsHalfOpen pins the half-open X range checkOnCurve accepts for a horizontal segment:
// start.X is in, end.X is out, in either direction. The doc used to call the range "strict", which is wrong at both
// ends and hides the property TestCheckOnCurveSharedVertexCountedOnce depends on.
func TestCheckOnCurveHorizontalIsHalfOpen(t *testing.T) {
	for _, tc := range []struct {
		name       string
		start, end geom.Point
		x, y       float32
		want       bool
	}{
		{name: "left-to-right start", start: geom.Pt(10, 5), end: geom.Pt(20, 5), x: 10, y: 5, want: true},
		{name: "left-to-right interior", start: geom.Pt(10, 5), end: geom.Pt(20, 5), x: 15, y: 5, want: true},
		{name: "left-to-right end", start: geom.Pt(10, 5), end: geom.Pt(20, 5), x: 20, y: 5, want: false},
		{name: "left-to-right before", start: geom.Pt(10, 5), end: geom.Pt(20, 5), x: 9.5, y: 5, want: false},
		{name: "left-to-right past", start: geom.Pt(10, 5), end: geom.Pt(20, 5), x: 20.5, y: 5, want: false},
		{name: "right-to-left start", start: geom.Pt(20, 5), end: geom.Pt(10, 5), x: 20, y: 5, want: true},
		{name: "right-to-left interior", start: geom.Pt(20, 5), end: geom.Pt(10, 5), x: 15, y: 5, want: true},
		{name: "right-to-left end", start: geom.Pt(20, 5), end: geom.Pt(10, 5), x: 10, y: 5, want: false},
		// A degenerate horizontal segment is all end and no start, so it accepts nothing.
		{name: "degenerate", start: geom.Pt(10, 5), end: geom.Pt(10, 5), x: 10, y: 5, want: false},
		// The horizontal branch ignores y entirely; the caller has already range-checked it.
		{name: "y ignored", start: geom.Pt(10, 5), end: geom.Pt(20, 5), x: 15, y: 99, want: true},
		// Non-horizontal segments take the other branch: an exact match with start, nothing else.
		{name: "vertical start", start: geom.Pt(10, 5), end: geom.Pt(10, 15), x: 10, y: 5, want: true},
		{name: "vertical end", start: geom.Pt(10, 5), end: geom.Pt(10, 15), x: 10, y: 15, want: false},
		{name: "vertical interior", start: geom.Pt(10, 5), end: geom.Pt(10, 15), x: 10, y: 10, want: false},
	} {
		if got := checkOnCurve(tc.x, tc.y, tc.start, tc.end); got != tc.want {
			t.Errorf("%s: checkOnCurve(%v, %v, %v, %v) = %v, want %v", tc.name, tc.x, tc.y, tc.start, tc.end, got,
				tc.want)
		}
	}
}

// TestCheckOnCurveSharedVertexCountedOnce is the reason the range is half-open. Contains walks a contour's segments in
// order, so an interior vertex of a horizontal run is one segment's end and the next one's start; a closed range would
// credit it to onCurveCount twice and flip the parity Contains resolves ties with. The trailing vertex of the run is
// the price: it belongs to no segment's half-open range and so is not counted at all.
func TestCheckOnCurveSharedVertexCountedOnce(t *testing.T) {
	rightward := []geom.Point{geom.Pt(0, 5), geom.Pt(10, 5), geom.Pt(20, 5)}
	leftward := []geom.Point{geom.Pt(20, 5), geom.Pt(10, 5), geom.Pt(0, 5)}
	longer := []geom.Point{geom.Pt(0, 5), geom.Pt(10, 5), geom.Pt(20, 5), geom.Pt(30, 5)}
	for _, tc := range []struct {
		name  string
		chain []geom.Point
		x     float32
		want  int
	}{
		{name: "shared vertex rightward", chain: rightward, x: 10, want: 1},
		{name: "shared vertex leftward", chain: leftward, x: 10, want: 1},
		{name: "leading vertex", chain: rightward, x: 0, want: 1},
		{name: "trailing vertex", chain: rightward, x: 20, want: 0},
		{name: "interior of first", chain: rightward, x: 5, want: 1},
		{name: "three-segment shared", chain: longer, x: 20, want: 1},
	} {
		onCurveCount := 0
		for i := 0; i+1 < len(tc.chain); i++ {
			windingLine(tc.chain[i:i+2], tc.x, 5, &onCurveCount)
		}
		if onCurveCount != tc.want {
			t.Errorf("%s: onCurveCount at x=%v = %d, want %d", tc.name, tc.x, onCurveCount, tc.want)
		}
	}
}

// TestContainsSplitHorizontalEdgeVertex exercises the same property through Contains: a rect whose bottom edge carries
// an extra vertex must still report that vertex as contained, whichever way the contour is wound.
func TestContainsSplitHorizontalEdgeVertex(t *testing.T) {
	ccw := []geom.Point{geom.Pt(0, 0), geom.Pt(20, 0), geom.Pt(20, 10), geom.Pt(10, 10), geom.Pt(0, 10)}
	cw := []geom.Point{geom.Pt(0, 0), geom.Pt(0, 10), geom.Pt(10, 10), geom.Pt(20, 10), geom.Pt(20, 0)}
	for _, tc := range []struct {
		name string
		pts  []geom.Point
	}{
		{name: "ccw", pts: ccw},
		{name: "cw", pts: cw},
	} {
		for _, fill := range []FillType{FillWinding, FillEvenOdd} {
			p := New()
			p.SetFillType(fill)
			p.MoveToPt(tc.pts[0])
			for _, pt := range tc.pts[1:] {
				p.LineToPt(pt)
			}
			p.Close()
			if !p.Contains(10, 10) {
				t.Errorf("%s/%v: the split vertex (10, 10) should be contained", tc.name, fill)
			}
			if !p.Contains(5, 10) {
				t.Errorf("%s/%v: the edge point (5, 10) should be contained", tc.name, fill)
			}
			if !p.Contains(10, 5) {
				t.Errorf("%s/%v: the interior point (10, 5) should be contained", tc.name, fill)
			}
			if p.Contains(10, 11) {
				t.Errorf("%s/%v: the exterior point (10, 11) should not be contained", tc.name, fill)
			}
		}
	}
}
