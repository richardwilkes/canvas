// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests over the intersecting, non-intersecting, and coincident line corpora, each verified through the line/line lane
// and (for axis-aligned inputs) the horizontal and vertical lanes.

package pathops

import (
	"math"
	"testing"
)

// dl builds a dLine from four coordinates.
func dl(x0, y0, x1, y1 float64) dLine {
	return dLine{pts: [2]dPoint{{x: x0, y: y0}, {x: x1, y: y1}}}
}

// f rounds x through float32 and back, producing the double value a float32 literal would have if widened to double
// (used for test coordinates that must match float32 precision exactly).
func f(x float64) float64 { return float64(float32(x)) }

var lineIntersectTests = [][2]dLine{
	{
		dl(0.00010360032320022583, 1.0172703415155411, 0.00014114845544099808, 1.0200891587883234),
		dl(0.00010259449481964111, 1.017270140349865, 0.00018215179443359375, 1.022890567779541),
	},
	{dl(30, 20, 30, 50), dl(24, 30, 36, 30)},
	{dl(323, 193, -317, 193), dl(0, 994, 0, 0)},
	{dl(90, 230, 160, 60), dl(60, 120, 260, 120)},
	{dl(90, 230, 160, 60), dl(181.176468, 120, 135.294128, 120)},
	{
		dl(f(181.1764678955078125), 120, f(186.3661956787109375), f(134.7042236328125)),
		dl(f(175.8309783935546875), f(141.5211334228515625), f(187.8782806396484375), f(133.7258148193359375)),
	},
	{dl(192, 4, 243, 4), dl(246, 4, 189, 4)},
	{dl(246, 4, 189, 4), dl(192, 4, 243, 4)},
	{dl(5, 0, 0, 5), dl(5, 4, 1, 4)},
	{dl(0, 0, 1, 0), dl(1, 0, 0, 0)},
	{dl(0, 0, 0, 0), dl(0, 0, 1, 0)},
	{dl(0, 1, 0, 1), dl(0, 0, 0, 2)},
	{dl(0, 0, 1, 0), dl(0, 0, 2, 0)},
	{dl(1, 1, 2, 2), dl(0, 0, 3, 3)},
	{
		dl(166.86950047022856, 112.69654129527828, 166.86948801592692, 112.69655741235339),
		dl(166.86960700313026, 112.6965477747386, 166.86925794355412, 112.69656471103423),
	},
}

var lineNoIntersect = [][2]dLine{
	{dl(float64(float32(2)-float32(1e-6)), 2, float64(float32(2)-float32(1e-6)), 4), dl(2, 1, 2, 3)},
	{dl(0, 0, 1, 0), dl(3, 0, 2, 0)},
	{dl(0, 0, 0, 0), dl(1, 0, 2, 0)},
	{dl(0, 1, 0, 1), dl(0, 3, 0, 2)},
	{dl(0, 0, 1, 0), dl(2, 0, 3, 0)},
	{dl(1, 1, 2, 2), dl(4, 4, 3, 3)},
}

var lineCoincidentTests = [][2]dLine{
	{dl(-1.48383003e-006, -83, 4.2268899e-014, -60), dl(9.5359502e-007, -60, 5.08227985e-015, -83)},
	{dl(10105, 2510, 10123, f(2509.98999)), dl(10105, f(2509.98999), 10123, 2510)},
	{dl(0, 482.5, -4.4408921e-016, 682.5), dl(0, 683, 0, 482)},
	{dl(1.77635684e-015, 312, -1.24344979e-014, 348), dl(0, 348, 0, 312)},
	{dl(979.304871, 561, 1036.69507, 291), dl(985.681519, 531, 982.159790, 547.568542)},
	{dl(232.159805, 547.568542, 235.681549, 531), dl(286.695129, 291, 229.304855, 561)},
	{
		dl(f(186.3661956787109375), f(134.7042236328125), f(187.8782806396484375), f(133.7258148193359375)),
		dl(f(175.8309783935546875), f(141.5211334228515625), f(187.8782806396484375), f(133.7258148193359375)),
	},
	{dl(235.681549, 531.000000, 280.318420, 321.000000), dl(286.695129, 291.000000, 229.304855, 561.000000)},
}

func checkResults(t *testing.T, line1, line2 dLine, ts *intersections, nearAllowed bool) {
	t.Helper()
	for i := 0; i < ts.usedCount(); i++ {
		result1 := line1.ptAtT(ts.tVal(0, i))
		result2 := line2.ptAtT(ts.tVal(1, i))
		if nearAllowed && result1.roughlyEqual(result2) {
			continue
		}
		if !result1.approximatelyEqual(result2) && !ts.nearlySameAt(i) {
			if ts.usedCount() == 1 {
				t.Fatalf("used==1 with mismatched intersection points")
			}
			result2 = line2.ptAtT(ts.tVal(1, i^1))
			if !result1.approximatelyEqual(result2) {
				t.Fatalf("results not approximately equal")
			}
			if !result1.approximatelyEqualPt(ts.pt(i).asPoint()) {
				t.Fatalf("result1 not approximately equal to intersection point")
			}
		}
	}
}

func testOneLine(t *testing.T, line1, line2 dLine, nearAllowed bool) {
	t.Helper()
	i := newIntersections()
	i.setAllowNear(nearAllowed)
	pts := i.intersectLineLine(line1, line2)
	if pts == 0 {
		t.Fatalf("expected intersection for %v / %v", line1, line2)
	}
	if pts != i.usedCount() {
		t.Fatalf("pts (%d) != used (%d)", pts, i.usedCount())
	}
	checkResults(t, line1, line2, i, nearAllowed)
	if line1.pts[0].equals(line1.pts[1]) || line2.pts[0].equals(line2.pts[1]) {
		return
	}
	if line1.pts[0].y == line1.pts[1].y {
		left := math.Min(line1.pts[0].x, line1.pts[1].x)
		right := math.Max(line1.pts[0].x, line1.pts[1].x)
		ts := newIntersections()
		ts.horizontalLine(line2, left, right, line1.pts[0].y, line1.pts[0].x != left)
		checkResults(t, line2, line1, ts, nearAllowed)
	}
	if line2.pts[0].y == line2.pts[1].y {
		left := math.Min(line2.pts[0].x, line2.pts[1].x)
		right := math.Max(line2.pts[0].x, line2.pts[1].x)
		ts := newIntersections()
		ts.horizontalLine(line1, left, right, line2.pts[0].y, line2.pts[0].x != left)
		checkResults(t, line1, line2, ts, nearAllowed)
	}
	if line1.pts[0].x == line1.pts[1].x {
		top := math.Min(line1.pts[0].y, line1.pts[1].y)
		bottom := math.Max(line1.pts[0].y, line1.pts[1].y)
		ts := newIntersections()
		ts.verticalLine(line2, top, bottom, line1.pts[0].x, line1.pts[0].y != top)
		checkResults(t, line2, line1, ts, nearAllowed)
	}
	if line2.pts[0].x == line2.pts[1].x {
		top := math.Min(line2.pts[0].y, line2.pts[1].y)
		bottom := math.Max(line2.pts[0].y, line2.pts[1].y)
		ts := newIntersections()
		ts.verticalLine(line1, top, bottom, line2.pts[0].x, line2.pts[0].y != top)
		checkResults(t, line1, line2, ts, nearAllowed)
	}
}

func testOneCoincidentLine(t *testing.T, line1, line2 dLine) {
	t.Helper()
	i := newIntersections()
	pts := i.intersectLineLine(line1, line2)
	if pts != 2 {
		t.Fatalf("expected 2 coincident points, got %d", pts)
	}
	if pts != i.usedCount() {
		t.Fatalf("pts (%d) != used (%d)", pts, i.usedCount())
	}
	checkResults(t, line1, line2, i, false)
	if line1.pts[0].equals(line1.pts[1]) || line2.pts[0].equals(line2.pts[1]) {
		return
	}
	if line1.pts[0].y == line1.pts[1].y {
		left := math.Min(line1.pts[0].x, line1.pts[1].x)
		right := math.Max(line1.pts[0].x, line1.pts[1].x)
		ts := newIntersections()
		ts.horizontalLine(line2, left, right, line1.pts[0].y, line1.pts[0].x != left)
		if ts.usedCount() != 2 {
			t.Fatalf("expected 2 from horizontal, got %d", ts.usedCount())
		}
		checkResults(t, line2, line1, ts, false)
	}
	if line2.pts[0].y == line2.pts[1].y {
		left := math.Min(line2.pts[0].x, line2.pts[1].x)
		right := math.Max(line2.pts[0].x, line2.pts[1].x)
		ts := newIntersections()
		ts.horizontalLine(line1, left, right, line2.pts[0].y, line2.pts[0].x != left)
		if ts.usedCount() != 2 {
			t.Fatalf("expected 2 from horizontal, got %d", ts.usedCount())
		}
		checkResults(t, line1, line2, ts, false)
	}
	if line1.pts[0].x == line1.pts[1].x {
		top := math.Min(line1.pts[0].y, line1.pts[1].y)
		bottom := math.Max(line1.pts[0].y, line1.pts[1].y)
		ts := newIntersections()
		ts.verticalLine(line2, top, bottom, line1.pts[0].x, line1.pts[0].y != top)
		if ts.usedCount() != 2 {
			t.Fatalf("expected 2 from vertical, got %d", ts.usedCount())
		}
		checkResults(t, line2, line1, ts, false)
	}
	if line2.pts[0].x == line2.pts[1].x {
		top := math.Min(line2.pts[0].y, line2.pts[1].y)
		bottom := math.Max(line2.pts[0].y, line2.pts[1].y)
		ts := newIntersections()
		ts.verticalLine(line1, top, bottom, line2.pts[0].x, line2.pts[0].y != top)
		if ts.usedCount() != 2 {
			t.Fatalf("expected 2 from vertical, got %d", ts.usedCount())
		}
		checkResults(t, line1, line2, ts, false)
	}
}

func TestPathOpsLineIntersection(t *testing.T) {
	for _, pair := range lineCoincidentTests {
		testOneCoincidentLine(t, pair[0], pair[1])
	}
	for _, pair := range lineIntersectTests {
		testOneLine(t, pair[0], pair[1], true)
	}
	for _, pair := range lineNoIntersect {
		ts := newIntersections()
		pts := ts.intersectLineLine(pair[0], pair[1])
		if pts != 0 {
			t.Fatalf("expected no intersection for %v / %v, got %d", pair[0], pair[1], pts)
		}
		if pts != ts.usedCount() {
			t.Fatalf("pts (%d) != used (%d)", pts, ts.usedCount())
		}
	}
}

func TestPathOpsLineIntersectionOneOff(t *testing.T) {
	testOneLine(t, lineIntersectTests[0][0], lineIntersectTests[0][1], true)
}

func TestPathOpsLineIntersectionExactOneOff(t *testing.T) {
	testOneLine(t, lineIntersectTests[0][0], lineIntersectTests[0][1], false)
}

func TestPathOpsLineIntersectionOneCoincident(t *testing.T) {
	testOneCoincidentLine(t, lineCoincidentTests[0][0], lineCoincidentTests[0][1])
}

// TestPathOpsIntersectRayLineCoincidentT verifies that the coincident-ray branch of intersectRayLine assigns the four
// T values mirroring Skia's fT[0][0]=fT[0][1]=0; fT[1][0]=fT[1][1]=1, and that pts[1] is derived from ts[0][1] rather
// than relying on it being left zero. The struct is deliberately preloaded with a nonzero ts[0][1] so a mis-transcribed
// assignment (writing ts[1][0] twice and leaving ts[0][1] stale) would corrupt pts[1].
func TestPathOpsIntersectRayLineCoincidentT(t *testing.T) {
	a := dl(0, 0, 4, 0)
	b := dl(3, 0, 7, 0) // same infinite line as a, offset along it, so the rays are coincident
	in := newIntersections()
	in.ts[0][1] = 0.75 // stale value that must be overwritten by the coincident branch
	used := in.intersectRayLine(a, b)
	if used != 2 {
		t.Fatalf("expected 2 coincident T values, got %d", used)
	}
	if in.tVal(0, 0) != 0 || in.tVal(0, 1) != 0 {
		t.Fatalf("ts[0] = {%v, %v}, want {0, 0}", in.tVal(0, 0), in.tVal(0, 1))
	}
	if in.tVal(1, 0) != 1 || in.tVal(1, 1) != 1 {
		t.Fatalf("ts[1] = {%v, %v}, want {1, 1}", in.tVal(1, 0), in.tVal(1, 1))
	}
	// pts[1] is computed from ts[0][1]; with the fix it must be a.ptAtT(0) == a.pts[0], not a.ptAtT(0.75).
	want := a.ptAtT(0)
	if !in.pt(1).equals(want) {
		t.Fatalf("pts[1] = %v, want %v (stale ts[0][1] leaked into computePoints)", in.pt(1), want)
	}
}
