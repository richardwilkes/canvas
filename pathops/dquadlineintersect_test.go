// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the quad/line intersection corpus and one-off cases, plus focused tests for the supporting reduceOrder
// quad/line reduction, dRect quad bounds, and the verticalQuad flipped lane. The quad control points come from float32
// host points widened to double, while the line and the expected intersection points are double, matching the precision
// at which each type naturally lives in this package.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// qp3 builds a quad's three float32 host points from double literals, rounding each through float32.
func qp3(x0, y0, x1, y1, x2, y2 float64) [3]geom.Point {
	return [3]geom.Point{
		{X: float32(x0), Y: float32(y0)},
		{X: float32(x1), Y: float32(y1)},
		{X: float32(x2), Y: float32(y2)},
	}
}

// doQuadLineIntersect dispatches to the vertical/horizontal/general lane based on the line's orientation, returning the
// result count and whether the axis-aligned lane flipped.
func doQuadLineIntersect(in *intersections, quad dQuad, line dLine) (int, bool) {
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
		result = in.verticalQuad(quad, top, bottom, line.pts[0].x, flipped)
	case line.pts[0].y == line.pts[1].y:
		left := line.pts[0].x
		right := line.pts[1].x
		flipped = left > right
		if flipped {
			left, right = right, left
		}
		result = in.horizontalQuad(quad, left, right, line.pts[0].y, flipped)
	default:
		in.intersectQuadLine(quad, line)
		result = in.used
	}
	return result, flipped
}

type lineQuadCase struct {
	quad     [3]geom.Point
	line     dLine
	result   int
	expected [2]dPoint
}

var lineQuadTests = []lineQuadCase{
	{quad: qp3(1, 1, 2, 1, 0, 2), line: dl(0, 0, 1, 1), result: 1, expected: [2]dPoint{{x: 1, y: 1}, {x: 0, y: 0}}},
	{quad: qp3(0, 0, 1, 1, 3, 1), line: dl(0, 0, 3, 1), result: 2, expected: [2]dPoint{{x: 0, y: 0}, {x: 3, y: 1}}},
	{quad: qp3(2, 0, 1, 1, 2, 2), line: dl(0, 0, 0, 2), result: 0, expected: [2]dPoint{{x: 0, y: 0}, {x: 0, y: 0}}},
	{quad: qp3(4, 0, 0, 1, 4, 2), line: dl(3, 1, 4, 1), result: 0, expected: [2]dPoint{{x: 0, y: 0}, {x: 0, y: 0}}},
	{quad: qp3(0, 0, 0, 1, 1, 1), line: dl(0, 1, 1, 0), result: 1, expected: [2]dPoint{{x: .25, y: .75}, {x: 0, y: 0}}},
}

var quadLineOneOffs = []struct {
	quad [3]geom.Point
	line dLine
}{
	{quad: qp3(97.9337616, 100, 88, 112.94265, 88, 130), line: dl(88.919838, 120, 107.058823, 120)},
	{
		quad: qp3(447.96701049804687, 894.4381103515625, 448.007080078125, 894.4239501953125,
			448.0140380859375, 894.4215087890625),
		line: dl(490.43548583984375, 879.40740966796875, 405.59262084960937, 909.435546875),
	},
	{quad: qp3(142.589081, 102.283646, 149.821579, 100, 158, 100), line: dl(90, 230, 160, 60)},
	{
		quad: qp3(1101, 10, 1101, 8.3431453704833984, 1099.828857421875, 7.1711997985839844),
		line: dl(1099.828857421875, 7.1711711883544922, 1099.121337890625, 7.8786783218383789),
	},
	{
		quad: qp3(973, 507, 973, 508.24264526367187, 972.12158203125, 509.12161254882812),
		line: dl(930, 467, 973, 510),
	},
	{
		quad: qp3(369.848602, 145.680267, 382.360413, 121.298294, 406.207703, 121.298294),
		line: dl(406.207703, 121.298294, 348.781738, 123.864815),
	},
}

func TestPathOpsQuadLineIntersectionOneOff(t *testing.T) {
	for index, tc := range quadLineOneOffs {
		var quad dQuad
		quad.set(tc.quad)
		in := newIntersections()
		result, _ := doQuadLineIntersect(in, quad, tc.line)
		for inner := 0; inner < result; inner++ {
			quadT := in.ts[0][inner]
			quadXY := quad.ptAtT(quadT)
			lineT := in.ts[1][inner]
			lineXY := tc.line.ptAtT(lineT)
			if !quadXY.approximatelyEqual(lineXY) {
				t.Fatalf("oneOff %d: quad(%v)=%v != line(%v)=%v", index, quadT, quadXY, lineT, lineXY)
			}
		}
	}
}

func TestPathOpsQuadLineIntersection(t *testing.T) {
	for index, tc := range lineQuadTests {
		var quad dQuad
		quad.set(tc.quad)
		var reducer1, reducer2 reduceOrder
		if order1 := reducer1.reduceQuad(quad); order1 < 3 {
			t.Fatalf("test %d: quad order=%d (expected 3)", index, order1)
		}
		if order2 := reducer2.reduceLine(tc.line); order2 < 2 {
			t.Fatalf("test %d: line order=%d (expected 2)", index, order2)
		}
		in := newIntersections()
		result, _ := doQuadLineIntersect(in, quad, tc.line)
		if result != tc.result {
			t.Fatalf("test %d: result=%d, expected %d", index, result, tc.result)
		}
		if in.used <= 0 {
			continue
		}
		for pt := 0; pt < result; pt++ {
			tt1 := in.ts[0][pt]
			if tt1 < 0 || tt1 > 1 {
				t.Fatalf("test %d: quad T %v out of range", index, tt1)
			}
			t1 := quad.ptAtT(tt1)
			tt2 := in.ts[1][pt]
			if tt2 < 0 || tt2 > 1 {
				t.Fatalf("test %d: line T %v out of range", index, tt2)
			}
			t2 := tc.line.ptAtT(tt2)
			if !t1.approximatelyEqual(t2) {
				t.Fatalf("test %d,%d: quad %v != line %v", index, pt, t1, t2)
			}
			if !t1.approximatelyEqual(tc.expected[0]) &&
				(tc.result == 1 || !t1.approximatelyEqual(tc.expected[1])) {
				t.Fatalf("test %d: t1=%v not an expected point", index, t1)
			}
		}
	}
}

func TestPathOpsReduceOrderQuad(t *testing.T) {
	cases := []struct {
		quad [3]geom.Point
		want int
	}{
		{quad: qp3(0, 0, 1, 1, 3, 1), want: 3}, // a real quad
		{quad: qp3(2, 2, 2, 2, 2, 2), want: 1}, // degenerate: single point
		{quad: qp3(0, 0, 1, 1, 2, 2), want: 2}, // colinear -> line
		{quad: qp3(5, 0, 5, 1, 5, 2), want: 2}, // vertical line
		{quad: qp3(0, 5, 1, 5, 2, 5), want: 2}, // horizontal line
		{quad: qp3(3, 3, 5, 7, 3, 3), want: 1}, // start==end, control elsewhere -> degenerate point
	}
	for index, c := range cases {
		var quad dQuad
		quad.set(c.quad)
		var r reduceOrder
		if got := r.reduceQuad(quad); got != c.want {
			t.Fatalf("reduceQuad case %d: got %d, want %d", index, got, c.want)
		}
	}
	var r reduceOrder
	if got := r.reduceLine(dl(0, 0, 1, 1)); got != 2 {
		t.Fatalf("reduceLine proper: got %d, want 2", got)
	}
	if got := r.reduceLine(dl(3, 3, 3, 3)); got != 1 {
		t.Fatalf("reduceLine point: got %d, want 1", got)
	}
}

func TestPathOpsQuadBounds(t *testing.T) {
	var quad dQuad
	quad.set(qp3(0, 0, 2, 4, 4, 0)) // apex above the chord at (2,2)
	var r dRect
	r.setBoundsQuadFull(quad)
	if !r.valid() {
		t.Fatal("bounds not valid")
	}
	if r.left != 0 || r.top != 0 || r.right != 4 || r.bottom != 2 {
		t.Fatalf("bounds = {%v,%v,%v,%v}, want {0,0,4,2}", r.left, r.top, r.right, r.bottom)
	}
	if r.width() != 4 || r.height() != 2 {
		t.Fatalf("size = %vx%v, want 4x2", r.width(), r.height())
	}
}

// TestPathOpsQuadLineFlip verifies the verticalQuad flipped lane: the flipped result's line T is the complement of the
// unflipped one at the same quad crossing (exercising intersections.flip, which the standard corpus above does not
// reach).
func TestPathOpsQuadLineFlip(t *testing.T) {
	var quad dQuad
	quad.set(qp3(0, 0, 1, 1, 2, 0)) // crosses x=1 at its apex, quad T=0.5, point (1,0.5)

	unflipped := newIntersections()
	if got := unflipped.verticalQuad(quad, 0, 2, 1, false); got != 1 {
		t.Fatalf("unflipped result=%d, want 1", got)
	}
	flipped := newIntersections()
	if got := flipped.verticalQuad(quad, 0, 2, 1, true); got != 1 {
		t.Fatalf("flipped result=%d, want 1", got)
	}
	if !approximatelyEqual(unflipped.ts[0][0], 0.5) || !approximatelyEqual(flipped.ts[0][0], 0.5) {
		t.Fatalf("quad T: unflipped=%v flipped=%v, want 0.5", unflipped.ts[0][0], flipped.ts[0][0])
	}
	if !approximatelyEqual(unflipped.ts[1][0]+flipped.ts[1][0], 1) {
		t.Fatalf("line T not complementary: %v + %v", unflipped.ts[1][0], flipped.ts[1][0])
	}
}
