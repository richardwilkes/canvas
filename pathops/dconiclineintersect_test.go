// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the conic/line intersection solver (TestPathOpsConicLineIntersection and
// TestPathOpsConicLineIntersectionOneOff), plus a focused reduceConic classification test. The conic is built from the
// double-precision test points directly (debugSet, no float32 rounding), while the reduceConic precondition narrows
// those points back to float32.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// cp3 builds a conic's three double-precision points from double literals, for use with debugSet.
func cp3(x0, y0, x1, y1, x2, y2 float64) [3]dPoint {
	return [3]dPoint{{x: x0, y: y0}, {x: x1, y: y1}, {x: x2, y: y2}}
}

// doConicLineIntersect dispatches to the vertical/horizontal/general lane based on the line's orientation, returning
// the result count and whether the axis-aligned lane flipped.
func doConicLineIntersect(in *intersections, conic dConic, line dLine) (int, bool) {
	flipped := false
	var result int
	switch {
	case line.pts[0].x == line.pts[1].x:
		top := line.pts[0].y
		bottom := line.pts[1].y
		flipped = top > bottom
		if flipped {
			top, bottom = bottom, top
		}
		result = in.verticalConic(conic, top, bottom, line.pts[0].x, flipped)
	case line.pts[0].y == line.pts[1].y:
		left := line.pts[0].x
		right := line.pts[1].x
		flipped = left > right
		if flipped {
			left, right = right, left
		}
		result = in.horizontalConic(conic, left, right, line.pts[0].y, flipped)
	default:
		in.intersectConicLine(conic, line)
		result = in.used
	}
	return result, flipped
}

type lineConicCase struct {
	conic    [3]dPoint
	weight   float32
	line     dLine
	result   int
	expected [2]dPoint
}

var lineConicTests = []lineConicCase{
	{
		conic: cp3(30.6499996, 25.6499996, 30.6499996, 20.6499996, 25.6499996, 20.6499996), weight: 0.707107008,
		line:     dl(25.6499996, 20.6499996, 45.6500015, 20.6499996),
		result:   1,
		expected: [2]dPoint{{x: 25.6499996, y: 20.6499996}, {x: 0, y: 0}},
	},
}

var conicLineOneOffs = []struct {
	conic  [3]dPoint
	weight float32
	line   dLine
}{
	{
		conic: cp3(30.6499996, 25.6499996, 30.6499996, 20.6499996, 25.6499996, 20.6499996), weight: 0.707107008,
		line: dl(25.6499996, 20.6499996, 45.6500015, 20.6499996),
	},
}

func TestPathOpsConicLineIntersectionOneOff(t *testing.T) {
	for index, tc := range conicLineOneOffs {
		var conic dConic
		conic.debugSet(tc.conic, tc.weight)
		in := newIntersections()
		result, _ := doConicLineIntersect(in, conic, tc.line)
		for inner := 0; inner < result; inner++ {
			conicT := in.ts[0][inner]
			conicXY := conic.ptAtT(conicT)
			lineT := in.ts[1][inner]
			lineXY := tc.line.ptAtT(lineT)
			if !conicXY.approximatelyEqual(lineXY) {
				t.Fatalf("oneOff %d: conic(%v)=%v != line(%v)=%v", index, conicT, conicXY, lineT, lineXY)
			}
		}
	}
}

func TestPathOpsConicLineIntersection(t *testing.T) {
	for index, tc := range lineConicTests {
		var conic dConic
		conic.debugSet(tc.conic, tc.weight)
		floatPts := [3]geom.Point{
			conic.pts.pts[0].asPoint(),
			conic.pts.pts[1].asPoint(),
			conic.pts.pts[2].asPoint(),
		}
		if order1 := reduceConic(floatPts, conic.weight); order1 != path.VerbConic {
			t.Fatalf("test %d: conic verb=%d (expected conic)", index, order1)
		}
		var reducer reduceOrder
		if order2 := reducer.reduceLine(tc.line); order2 < 2 {
			t.Fatalf("test %d: line order=%d (expected >= 2)", index, order2)
		}
		in := newIntersections()
		result, _ := doConicLineIntersect(in, conic, tc.line)
		if result != tc.result {
			t.Fatalf("test %d: result=%d, expected %d", index, result, tc.result)
		}
		if in.used <= 0 {
			continue
		}
		for pt := 0; pt < result; pt++ {
			tt1 := in.ts[0][pt]
			if tt1 < 0 || tt1 > 1 {
				t.Fatalf("test %d: conic T %v out of range", index, tt1)
			}
			t1 := conic.ptAtT(tt1)
			tt2 := in.ts[1][pt]
			if tt2 < 0 || tt2 > 1 {
				t.Fatalf("test %d: line T %v out of range", index, tt2)
			}
			t2 := tc.line.ptAtT(tt2)
			if !t1.approximatelyEqual(t2) {
				t.Fatalf("test %d,%d: conic %v != line %v", index, pt, t1, t2)
			}
			if !t1.approximatelyEqual(tc.expected[0]) &&
				(tc.result == 1 || !t1.approximatelyEqual(tc.expected[1])) {
				t.Fatalf("test %d: t1=%v not an expected point", index, t1)
			}
		}
	}
}

// TestPathOpsReduceConic exercises the reduceConic verb classifier across the reduce lanes: a genuine conic (weight !=
// 1) stays a conic, a genuine quad shape with weight 1 becomes a quad, a colinear control net collapses to a line, and
// a single-point net collapses to a move.
func TestPathOpsReduceConic(t *testing.T) {
	cases := []struct {
		pts  [3]geom.Point
		w    float32
		want path.Verb
	}{
		{pts: [3]geom.Point{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 3, Y: 1}}, w: 0.5, want: path.VerbConic}, // real conic
		{pts: [3]geom.Point{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 3, Y: 1}}, w: 1, want: path.VerbQuad},    // weight 1 -> quad
		{pts: [3]geom.Point{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 2}}, w: 0.7, want: path.VerbLine},  // colinear -> line
		{pts: [3]geom.Point{{X: 2, Y: 2}, {X: 2, Y: 2}, {X: 2, Y: 2}}, w: 0.7, want: path.VerbMove},  // single point -> move
	}
	for index, c := range cases {
		if got := reduceConic(c.pts, c.w); got != c.want {
			t.Fatalf("reduceConic case %d: got %d, want %d", index, got, c.want)
		}
	}
}
