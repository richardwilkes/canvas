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

// calcAndSortAngles reproduces the boolean-op driver's order of operations: build every segment's angles, then sort
// them all. Fails the test if any sortAngles trips.
func calcAndSortAngles(t *testing.T, head *opContourHead) {
	t.Helper()
	for _, c := range realContours(head) {
		for _, s := range segsOf(c) {
			s.calcAngles()
		}
	}
	for _, c := range realContours(head) {
		for _, s := range segsOf(c) {
			if !s.sortAngles() {
				t.Fatal("sortAngles returned false")
			}
		}
	}
}

// angleLoop returns the angles reachable by following next from start, once around the loop.
func angleLoop(start *opAngle) []*opAngle {
	var out []*opAngle
	a := start
	for a != nil {
		out = append(out, a)
		a = a.next
		if a == start {
			break
		}
	}
	return out
}

// cyclicallyAscending reports whether vals, read as a circular sequence, ascends everywhere except at a single wrap
// point — i.e. it is a rotation of a strictly ascending list.
func cyclicallyAscending(vals []int) bool {
	if len(vals) < 2 {
		return true
	}
	descents := 0
	for i := range vals {
		j := (i + 1) % len(vals)
		if vals[j] < vals[i] {
			descents++
		}
	}
	return descents == 1
}

// TestOpAngleFindSector checks the sedecimant classification against hand-computed sectors for the cardinal and
// diagonal sweep directions (line verb: no epsilon nudging).
func TestOpAngleFindSector(t *testing.T) {
	a := &opAngle{}
	cases := []struct {
		x, y float64
		want int
	}{
		{x: 1, y: 0, want: 31},
		{x: -1, y: 0, want: 15},
		{x: 0, y: 1, want: 23},
		{x: 0, y: -1, want: 7},
		{x: 5, y: 5, want: 27},
		{x: 5, y: -5, want: 3},
		{x: -5, y: 5, want: 19},
		{x: -5, y: -5, want: 11},
	}
	for _, c := range cases {
		if got := a.findSector(path.VerbLine, c.x, c.y); got != c.want {
			t.Errorf("findSector(%v,%v) = %d, want %d", c.x, c.y, got, c.want)
		}
	}
}

// TestSetCurveHullSweep checks the sweep vectors and is-curve flag for a line, a bending quad, and a well-behaved
// cubic.
func TestSetCurveHullSweep(t *testing.T) {
	line := dCurveSweep{curve: dCurve{pts: [4]dPoint{{x: 0, y: 0}, {x: 10, y: 0}, {}, {}}}}
	line.setCurveHullSweep(path.VerbLine)
	if line.isCurve() || !line.isOrdered() {
		t.Fatalf("line: isCurve=%v ordered=%v, want false/true", line.isCurve(), line.isOrdered())
	}
	if line.sweep[0] != (dVector{x: 10, y: 0}) || line.sweep[1] != (dVector{x: 10, y: 0}) {
		t.Fatalf("line sweeps = %v,%v, want (10,0),(10,0)", line.sweep[0], line.sweep[1])
	}

	quad := dCurveSweep{curve: dCurve{pts: [4]dPoint{{x: 0, y: 0}, {x: 10, y: 10}, {x: 20, y: 0}, {}}}}
	quad.setCurveHullSweep(path.VerbQuad)
	if !quad.isCurve() {
		t.Fatal("bending quad should report isCurve")
	}
	if quad.sweep[0] != (dVector{x: 10, y: 10}) || quad.sweep[1] != (dVector{x: 20, y: 0}) {
		t.Fatalf("quad sweeps = %v,%v, want (10,10),(20,0)", quad.sweep[0], quad.sweep[1])
	}

	// A degenerate (straight) quad has a zero cross product => not a curve.
	straight := dCurveSweep{curve: dCurve{pts: [4]dPoint{{x: 0, y: 0}, {x: 5, y: 0}, {x: 10, y: 0}, {}}}}
	straight.setCurveHullSweep(path.VerbQuad)
	if straight.isCurve() {
		t.Fatal("collinear quad should not report isCurve")
	}

	cubic := dCurveSweep{curve: dCurve{pts: [4]dPoint{{x: 0, y: 0}, {x: 10, y: 10}, {x: 20, y: 10}, {x: 30, y: 0}}}}
	cubic.setCurveHullSweep(path.VerbCubic)
	if !cubic.isCurve() || !cubic.isOrdered() {
		t.Fatalf("cubic: isCurve=%v ordered=%v, want true/true", cubic.isCurve(), cubic.isOrdered())
	}
}

// TestLineParametersDistance checks lineEndPoints/pointDistance/dx/dy on the horizontal line y=0.
func TestLineParametersDistance(t *testing.T) {
	var lp lineParameters
	lp.lineEndPoints(dLine{pts: [2]dPoint{{x: 0, y: 0}, {x: 10, y: 0}}})
	if lp.dx() != 10 || lp.dy() != 0 {
		t.Fatalf("dx/dy = %v/%v, want 10/0", lp.dx(), lp.dy())
	}
	if d := lp.pointDistance(dPoint{x: 5, y: 5}); d != 50 {
		t.Fatalf("pointDistance((5,5)) = %v, want 50", d)
	}
	if d := lp.pointDistance(dPoint{x: 5, y: 0}); d != 0 {
		t.Fatalf("pointDistance((5,0)) = %v, want 0", d)
	}
}

// TestOpSegmentSubDivide exercises both subDivide lanes on a monotonic quad segment: the endpoints-only fast path
// (start/end at t 0/1) and the recompute-from-t path.
func TestOpSegmentSubDivide(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).QuadTo(10, 0, 10, 10).Close() // monotonic quad, added whole by the edge builder
	head, _ := buildOpModel(t, p)
	segs := segsOf(realContours(head)[0])
	var quad *opSegment
	for _, s := range segs {
		if s.verb == path.VerbQuad {
			quad = s
			break
		}
	}
	if quad == nil {
		t.Fatal("no quad segment built")
	}
	// Fast path: whole curve, control point copied verbatim, returns false.
	var whole dCurve
	if quad.subDivide(&quad.head.opSpanBase, &quad.tail, &whole) {
		t.Fatal("whole-segment subDivide should take the endpoints-only fast path (false)")
	}
	if whole.pts[0] != (dPoint{x: 0, y: 0}) || whole.pts[1] != (dPoint{x: 10, y: 0}) || whole.pts[2] != (dPoint{x: 10, y: 10}) {
		t.Fatalf("whole subDivide = %v,%v,%v, want (0,0),(10,0),(10,10)", whole.pts[0], whole.pts[1], whole.pts[2])
	}
	// Recompute path: head..t=0.5 half. Point at t=0.5 is (7.5, 2.5).
	mid := quad.addT(0.5)
	if mid == nil {
		t.Fatal("addT(0.5) failed")
	}
	var half dCurve
	if !quad.subDivide(&quad.head.opSpanBase, mid.span, &half) {
		t.Fatal("partial subDivide should compute a curve (true)")
	}
	if half.pts[0] != (dPoint{x: 0, y: 0}) {
		t.Fatalf("half start = %v, want (0,0)", half.pts[0])
	}
	if !half.pts[2].approximatelyEqual(dPoint{x: 7.5, y: 2.5}) {
		t.Fatalf("half end = %v, want (7.5,2.5)", half.pts[2])
	}
}

// TestCalcSortAnglesBowtie is the end-to-end angle test: a self-crossing bow-tie whose two diagonals meet at (5,5).
// After calcAngles+sortAngles the four half-edges leaving (5,5) form one counterclockwise-sorted angle loop; their
// sweep sectors must be {3,11,19,27} in cyclic ascending (CCW) order, two per diagonal.
func TestCalcSortAnglesBowtie(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).LineTo(10, 10).LineTo(10, 0).LineTo(0, 10).Close()
	head, state := buildOpModel(t, p)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head)

	segs := segsOf(realContours(head)[0])
	seg0, seg2 := segs[0], segs[2] // the two diagonals
	cross := spanBaseAtPt(seg0, pt(5, 5))
	if cross == nil || cross == &seg0.head.opSpanBase || cross == &seg0.tail {
		t.Fatal("seg0 has no interior crossing span at (5,5)")
	}
	base := cross.upCast().toAngle()
	if base == nil {
		t.Fatal("crossing span carries no toAngle after sorting")
	}
	loop := angleLoop(base)
	if len(loop) != 4 {
		t.Fatalf("crossing angle loop count = %d, want 4", len(loop))
	}
	sectors := make([]int, len(loop))
	seg0Count, seg2Count := 0, 0
	for i, a := range loop {
		sectors[i] = a.sectorStart
		switch a.segment() {
		case seg0:
			seg0Count++
		case seg2:
			seg2Count++
		default:
			t.Fatalf("angle %d belongs to an unexpected segment", i)
		}
	}
	if seg0Count != 2 || seg2Count != 2 {
		t.Fatalf("angles per diagonal = %d/%d, want 2/2", seg0Count, seg2Count)
	}
	want := map[int]bool{3: true, 11: true, 19: true, 27: true}
	for _, s := range sectors {
		if !want[s] {
			t.Fatalf("unexpected sector %d in loop %v", s, sectors)
		}
		delete(want, s)
	}
	if len(want) != 0 {
		t.Fatalf("loop sectors %v missing %v", sectors, want)
	}
	if !cyclicallyAscending(sectors) {
		t.Fatalf("loop sectors %v are not in counterclockwise (cyclic ascending) order", sectors)
	}
}

// TestCalcSortAnglesTwoRects sorts the angles of two overlapping rectangles. Each of the two transverse edge crossings
// must yield a four-angle loop (both edges' from- and to-angles), verifying the sort over several independent
// intersection points.
func TestCalcSortAnglesTwoRects(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(20, 0).LineTo(20, 20).LineTo(0, 20).Close()
	b := path.New()
	b.MoveTo(10, 10).LineTo(30, 10).LineTo(30, 30).LineTo(10, 30).Close()
	head, state := buildOpModel(t, a, b)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head)

	cs := realContours(head)
	aSegs := segsOf(cs[0])
	for _, cross := range []struct {
		seg *opSegment
		at  geom.Point
	}{
		{seg: aSegs[1], at: pt(20, 10)}, // A.right x B.top
		{seg: aSegs[2], at: pt(10, 20)}, // A.bottom x B.left
	} {
		span := spanBaseAtPt(cross.seg, cross.at)
		if span == nil || span == &cross.seg.head.opSpanBase || span == &cross.seg.tail {
			t.Fatalf("no interior crossing span at %v", cross.at)
		}
		base := span.upCast().toAngle()
		if base == nil {
			base = span.fromAngle()
		}
		if base == nil {
			t.Fatalf("crossing at %v carries no angle after sorting", cross.at)
		}
		loop := angleLoop(base)
		if len(loop) != 4 {
			t.Fatalf("crossing at %v: angle loop count = %d, want 4", cross.at, len(loop))
		}
		if !cyclicallyAscending(sectorsOf(loop)) {
			t.Fatalf("crossing at %v: sectors %v not counterclockwise", cross.at, sectorsOf(loop))
		}
	}
}

// TestCalcSortAnglesQuadLine sorts the angles where a quadratic arc crosses a straight edge, exercising the
// curved-section angle path (setSpans side computation, convexHullOverlaps/endsIntersect). Each of the two crossings
// mixes a quad half with the line, and the sort must succeed with a four-angle loop.
func TestCalcSortAnglesQuadLine(t *testing.T) {
	arc := path.New()
	arc.MoveTo(0, 0).QuadTo(10, 20, 20, 0).Close()
	box := path.New()
	box.MoveTo(-5, 5).LineTo(25, 5).LineTo(25, 25).LineTo(-5, 25).Close()
	head, state := buildOpModel(t, arc, box)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head)

	boxBottom := segsOf(realContours(head)[1])[0] // the y=5 edge
	crossings := 0
	for base := boxBottom.head.next; base != &boxBottom.tail; base = base.upCast().next {
		crossings++
		angle := base.upCast().toAngle()
		if angle == nil {
			angle = base.fromAngle()
		}
		if angle == nil {
			t.Fatalf("box-edge crossing at %v carries no angle", base.pt())
		}
		loop := angleLoop(angle)
		if len(loop) != 4 {
			t.Fatalf("box-edge crossing at %v: loop count = %d, want 4", base.pt(), len(loop))
		}
		// Exactly one quad half and the line meet at each crossing (two from-/to-angles each).
		quadAngles, lineAngles := 0, 0
		for _, a := range loop {
			if a.segment().verb == path.VerbQuad {
				quadAngles++
			} else {
				lineAngles++
			}
		}
		if quadAngles != 2 || lineAngles != 2 {
			t.Fatalf("box-edge crossing at %v: quad/line angles = %d/%d, want 2/2", base.pt(), quadAngles, lineAngles)
		}
	}
	if crossings != 2 {
		t.Fatalf("box-edge crossing count = %d, want 2", crossings)
	}
}

// TestCalcSortAnglesTwoOvals sorts the angles where two overlapping circles cross. Because conic arcs span many
// sectors, the crossing angles' sector masks overlap, forcing the geometric comparison path (orderable ->
// convexHullOverlaps/endsIntersect) that the all-distinct-sector line crossings above skip. Two circles centered 12
// apart with radius 10 cross transversely at (6, +/-8).
func TestCalcSortAnglesTwoOvals(t *testing.T) {
	a := path.New().AddOval(geom.Rect{Left: -10, Top: -10, Right: 10, Bottom: 10}, geom.DirectionCW)
	b := path.New().AddOval(geom.Rect{Left: 2, Top: -10, Right: 22, Bottom: 10}, geom.DirectionCW)
	head, state := buildOpModel(t, a, b)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head) // asserts sortAngles success on every conic segment

	cs := realContours(head)
	if len(cs) != 2 {
		t.Fatalf("contour count = %d, want 2", len(cs))
	}
	cA, cB := cs[0], cs[1]
	mixedCrossings := 0
	for _, c := range cs {
		for _, s := range segsOf(c) {
			for base := s.head.next; base != &s.tail; base = base.upCast().next {
				angle := base.upCast().toAngle()
				if angle == nil {
					angle = base.fromAngle()
				}
				if angle == nil {
					continue
				}
				loop := angleLoop(angle)
				fromA, fromB := false, false
				for _, ang := range loop {
					switch ang.segment().contour {
					case cA:
						fromA = true
					case cB:
						fromB = true
					}
				}
				if fromA && fromB && len(loop) >= 4 {
					mixedCrossings++
				}
			}
		}
	}
	// Each of the two circle intersections is reachable from both crossing arcs' spans.
	if mixedCrossings < 2 {
		t.Fatalf("cross-contour angle loops (>=4 angles, both circles) = %d, want >=2", mixedCrossings)
	}
}

// TestCalcSortAnglesParallelQuads sorts the angles where two nearly parallel monotonic quads cross. Because the two
// arcs leave the crossing in similar directions their sector masks overlap, driving the curve/curve geometric
// comparison (orderable -> convexHullOverlaps) that the transverse crossings above bypass. The quads share one
// crossing; each closed shape adds a wide triangle so the contours are non-degenerate.
func TestCalcSortAnglesParallelQuads(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).QuadTo(20, 5, 40, 20).LineTo(40, 60).LineTo(0, 60).Close()
	b := path.New()
	b.MoveTo(0, 4).QuadTo(20, 6, 40, 16).LineTo(40, -40).LineTo(0, -40).Close()
	head, state := buildOpModel(t, a, b)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head) // asserts sortAngles success

	cs := realContours(head)
	if len(cs) != 2 {
		t.Fatalf("contour count = %d, want 2", len(cs))
	}
	cA, cB := cs[0], cs[1]
	mixed := 0
	for _, c := range cs {
		for _, s := range segsOf(c) {
			for base := s.head.next; base != &s.tail; base = base.upCast().next {
				angle := base.upCast().toAngle()
				if angle == nil {
					angle = base.fromAngle()
				}
				if angle == nil {
					continue
				}
				loop := angleLoop(angle)
				fromA, fromB := false, false
				for _, ang := range loop {
					switch ang.segment().contour {
					case cA:
						fromA = true
					case cB:
						fromB = true
					}
				}
				if fromA && fromB && len(loop) >= 4 {
					mixed++
				}
			}
		}
	}
	if mixed < 2 {
		t.Fatalf("cross-contour angle loops (>=4 angles, both quads) = %d, want >=2", mixed)
	}
}

// sectorsOf collects the sectorStart of each angle in the slice.
func sectorsOf(loop []*opAngle) []int {
	out := make([]int, len(loop))
	for i, a := range loop {
		out[i] = a.sectorStart
	}
	return out
}
