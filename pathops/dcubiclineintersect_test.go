// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the cubic/line intersection solver (TestPathOpsCubicLineIntersection,
// TestPathOpsCubicLineIntersectionOneOff, and TestPathOpsFailCubicLineIntersection), plus focused reduceCubic
// classification and cubic-bounds tests. The cubic is built from the double-precision test points directly (debugSet,
// no float32 rounding); the few float-literal coordinates in the corpus are widened through float32 (f32d) to preserve
// their original float32 rounding.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// cu4 builds a cubic's four double-precision points from double literals, matching CubicPts/debugSet.
func cu4(x0, y0, x1, y1, x2, y2, x3, y3 float64) [4]dPoint {
	return [4]dPoint{{x: x0, y: y0}, {x: x1, y: y1}, {x: x2, y: y2}, {x: x3, y: y3}}
}

// f32d rounds a value through float32 before widening back to float64, matching a float32 literal stored into a
// double-precision field.
func f32d(x float64) float64 { return float64(float32(x)) }

// doCubicLineIntersect dispatches to the vertical/horizontal/general lane based on the line's orientation, returning
// the result count and whether the axis-aligned lane flipped.
func doCubicLineIntersect(in *intersections, cubic dCubic, line dLine) (int, bool) {
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
		result = in.verticalCubic(cubic, top, bottom, line.pts[0].x, flipped)
	case line.pts[0].y == line.pts[1].y:
		left := line.pts[0].x
		right := line.pts[1].x
		flipped = left > right
		if flipped {
			left, right = right, left
		}
		result = in.horizontalCubic(cubic, left, right, line.pts[0].y, flipped)
	default:
		in.intersectCubicLine(cubic, line)
		result = in.used
	}
	return result, flipped
}

type lineCubicCase struct {
	cubic [4]dPoint
	line  dLine
}

var lineCubicTests = []lineCubicCase{
	{cubic: cu4(0, 6, 1.0851458311080933, 4.3722810745239258, 1.5815209150314331, 3.038947582244873,
		1.9683018922805786, 1.9999997615814209), line: dl(3, 2, 1, 2)},
	{cubic: cu4(0.468027353, 4, 1.06734705, 1.33333337, 1.36700678, 0, 3, 0), line: dl(2, 1, 0, 1)},
	{
		cubic: cu4(-634.60540771484375, -481.262939453125, 266.2696533203125, -752.70867919921875,
			-751.8370361328125, -317.37921142578125, -969.7427978515625, 824.7255859375),
		line: dl(-287.9506133720805678, -557.1376476615772617, -285.9506133720805678, -557.1376476615772617),
	},
	{cubic: cu4(36.7184372, 0.888650894, 36.7184372, 0.888650894, 35.1233864, 0.554015458, 34.5114098,
		-0.115255356), line: dl(35.4531212, 0, 31.9375, 0)},
	{
		cubic: cu4(421, 378, 421, f32d(380.209137), f32d(418.761414), 382, 416, 382),
		line:  dl(320, 378, 421, f32d(378.000031)),
	},
	{
		cubic: cu4(416, 383, f32d(418.761414), 383, 421, f32d(380.761414), 421, 378),
		line:  dl(320, 378, 421, f32d(378.000031)),
	},
	{cubic: cu4(154, 715, 151.238571, 715, 149, 712.761414, 149, 710), line: dl(149, 675, 149, 710.001465)},
	{cubic: cu4(0, 1, 1, 6, 4, 1, 4, 3), line: dl(6, 1, 1, 4)},
	{cubic: cu4(0, 1, 2, 6, 4, 1, 5, 4), line: dl(6, 2, 1, 4)},
	{cubic: cu4(0, 4, 3, 4, 6, 2, 5, 2), line: dl(4, 3, 2, 6)},
	{cubic: cu4(1006.6951293945312, 291, 1023.263671875, 291, 1033.8402099609375, 304.43145751953125,
		1030.318359375, 321), line: dl(979.30487060546875, 561, 1036.695068359375, 291)},
	{cubic: cu4(259.30487060546875, 561, 242.73631286621094, 561, 232.15980529785156, 547.56854248046875,
		235.68154907226562, 531), line: dl(286.69512939453125, 291, 229.30485534667969, 561)},
	{cubic: cu4(1, 2, 2, 6, 2, 0, 1, 0), line: dl(1, 0, 1, 2)},
	{cubic: cu4(0, 0, 0, 1, 0, 1, 1, 1), line: dl(0, 1, 1, 0)},
}

var failLineCubicTests = []lineCubicCase{
	{
		cubic: cu4(37.5273438, -1.44140625, 37.8736992, -1.69921875, 38.1640625, -2.140625, 38.3984375, -2.765625),
		line:  dl(40.625, -5.7890625, 37.7109375, 1.3515625),
	},
}

// checkCubicOnBoth verifies every found intersection lies on both the cubic and the line.
func checkCubicOnBoth(t *testing.T, index int, cubic dCubic, line dLine, in *intersections, result int) {
	t.Helper()
	for pt := 0; pt < result; pt++ {
		tt1 := in.ts[0][pt]
		xy1 := cubic.ptAtT(tt1)
		tt2 := in.ts[1][pt]
		xy2 := line.ptAtT(tt2)
		if !xy1.approximatelyEqual(xy2) {
			t.Fatalf("test %d,%d: cubic(%v)=%v != line(%v)=%v", index, pt, tt1, xy1, tt2, xy2)
		}
	}
}

func TestPathOpsCubicLineIntersection(t *testing.T) {
	for index, tc := range lineCubicTests {
		var cubic dCubic
		cubic.debugSet(tc.cubic)
		var reduce1, reduce2 reduceOrder
		order1 := reduce1.reduceCubic(cubic, noQuadratics)
		order2 := reduce2.reduceLine(tc.line)
		if order1 < 4 {
			t.Fatalf("test %d: cubic order=%d (expected 4)", index, order1)
		}
		if order2 < 2 {
			t.Fatalf("test %d: line order=%d (expected 2)", index, order2)
		}
		in := newIntersections()
		result, _ := doCubicLineIntersect(in, cubic, tc.line)
		checkCubicOnBoth(t, index, cubic, tc.line, in, result)
	}
}

func TestPathOpsCubicLineIntersectionOneOff(t *testing.T) {
	index := 0
	tc := lineCubicTests[index]
	var cubic dCubic
	cubic.debugSet(tc.cubic)
	in := newIntersections()
	result, _ := doCubicLineIntersect(in, cubic, tc.line)
	checkCubicOnBoth(t, index, cubic, tc.line, in, result)

	in2 := newIntersections()
	in2.intersectCubicLine(cubic, tc.line)
	if in2.used != 1 {
		t.Fatalf("oneOff: used=%d, want 1", in2.used)
	}
}

func TestPathOpsFailCubicLineIntersection(t *testing.T) {
	for index, tc := range failLineCubicTests {
		var cubic dCubic
		cubic.debugSet(tc.cubic)
		var reduce1, reduce2 reduceOrder
		order1 := reduce1.reduceCubic(cubic, noQuadratics)
		order2 := reduce2.reduceLine(tc.line)
		if order1 < 4 {
			t.Fatalf("fail %d: cubic order=%d (expected 4)", index, order1)
		}
		if order2 < 2 {
			t.Fatalf("fail %d: line order=%d (expected 2)", index, order2)
		}
		in := newIntersections()
		roots := in.intersectCubicLine(cubic, tc.line)
		if roots != 0 {
			t.Fatalf("fail %d: roots=%d, want 0", index, roots)
		}
	}
}

// TestPathOpsReduceCubic exercises the reduceCubic verb classifier across the reduce lanes: a genuine cubic stays a
// cubic, an exact single-quadratic cubic collapses to a quad, a colinear control net collapses to a line, and a
// single-point net collapses to a move.
func TestPathOpsReduceCubic(t *testing.T) {
	cases := []struct {
		pts  [4]geom.Point
		want path.Verb
	}{
		{pts: [4]geom.Point{{X: 0, Y: 1}, {X: 1, Y: 6}, {X: 4, Y: 1}, {X: 4, Y: 3}}, want: path.VerbCubic}, // real cubic
		// quad (0,0),(2,4),(4,0) expressed as a cubic collapses back to a quad
		{
			pts:  [4]geom.Point{{X: 0, Y: 0}, {X: 4.0 / 3, Y: 8.0 / 3}, {X: 8.0 / 3, Y: 8.0 / 3}, {X: 4, Y: 0}},
			want: path.VerbQuad,
		},
		{pts: [4]geom.Point{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 2}, {X: 3, Y: 3}}, want: path.VerbLine}, // colinear
		{pts: [4]geom.Point{{X: 2, Y: 2}, {X: 2, Y: 2}, {X: 2, Y: 2}, {X: 2, Y: 2}}, want: path.VerbMove}, // one point
	}
	for index, c := range cases {
		if got := reduceCubicToVerb(c.pts); got != c.want {
			t.Fatalf("reduceCubicToVerb case %d: got %d, want %d", index, got, c.want)
		}
	}

	// Direct reduceCubic calls exercise the degenerate reduction lanes that reduceCubicToVerb's all-coincident
	// short-circuit and the intersection corpus never reach.
	direct := []struct {
		pts  [4]dPoint
		want int
	}{
		{pts: cu4(5, 0, 5, 1, 5, 2, 5, 3), want: 2}, // vertical line
		{pts: cu4(0, 5, 1, 5, 2, 5, 3, 5), want: 2}, // horizontal line
		{pts: cu4(2, 2, 2, 2, 2, 2, 2, 2), want: 1}, // all coincident -> point
		{pts: cu4(0, 1, 2, 6, 4, 1, 5, 4), want: 4}, // genuine cubic
	}
	for index, c := range direct {
		var cubic dCubic
		cubic.debugSet(c.pts)
		var r reduceOrder
		if got := r.reduceCubic(cubic, noQuadratics); got != c.want {
			t.Fatalf("reduceCubic direct case %d: got %d, want %d", index, got, c.want)
		}
	}
}

// TestPathOpsCubicBounds exercises setBoundsCubicFull: a symmetric arch cubic with its apex at (2,3), giving a
// {0,0,4,3} bounding box.
func TestPathOpsCubicBounds(t *testing.T) {
	var cubic dCubic
	cubic.debugSet(cu4(0, 0, 0, 4, 4, 4, 4, 0))
	var r dRect
	r.setBoundsCubicFull(cubic)
	if !r.valid() {
		t.Fatal("bounds not valid")
	}
	if r.left != 0 || r.top != 0 || r.right != 4 || r.bottom != 3 {
		t.Fatalf("bounds = {%v,%v,%v,%v}, want {0,0,4,3}", r.left, r.top, r.right, r.bottom)
	}
	if r.width() != 4 || r.height() != 3 {
		t.Fatalf("size = %vx%v, want 4x3", r.width(), r.height())
	}
}
