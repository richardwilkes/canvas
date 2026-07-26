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
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// TestWindingRayHelpers checks the opRayDir coordinate/axis helpers against hand values.
func TestWindingRayHelpers(t *testing.T) {
	if rayXYIndex(rayDirLeft) != 0 || rayXYIndex(rayDirTop) != 1 || rayXYIndex(rayDirRight) != 0 ||
		rayXYIndex(rayDirBottom) != 1 {
		t.Fatal("rayXYIndex wrong")
	}
	p := geom.Point{X: 3, Y: 4}
	if ptXY(p, rayDirLeft) != 3 || ptXY(p, rayDirTop) != 4 {
		t.Fatalf("ptXY = %v/%v, want 3/4", ptXY(p, rayDirLeft), ptXY(p, rayDirTop))
	}
	if ptYX(p, rayDirLeft) != 4 || ptYX(p, rayDirTop) != 3 {
		t.Fatalf("ptYX = %v/%v, want 4/3", ptYX(p, rayDirLeft), ptYX(p, rayDirTop))
	}
	v := dVector{x: 5, y: 6}
	if ptDxDy(v, rayDirLeft) != 5 || ptDxDy(v, rayDirTop) != 6 {
		t.Fatalf("ptDxDy = %v/%v, want 5/6", ptDxDy(v, rayDirLeft), ptDxDy(v, rayDirTop))
	}
	if ptDyDx(v, rayDirLeft) != 6 || ptDyDx(v, rayDirTop) != 5 {
		t.Fatalf("ptDyDx = %v/%v, want 6/5", ptDyDx(v, rayDirLeft), ptDyDx(v, rayDirTop))
	}
	b := pathOpsBounds{left: 1, top: 2, right: 3, bottom: 4}
	if rectSide(b, rayDirLeft) != 1 || rectSide(b, rayDirTop) != 2 || rectSide(b, rayDirRight) != 3 ||
		rectSide(b, rayDirBottom) != 4 {
		t.Fatal("rectSide wrong")
	}
	if !lessThan(rayDirLeft) || !lessThan(rayDirTop) || lessThan(rayDirRight) || lessThan(rayDirBottom) {
		t.Fatal("lessThan wrong")
	}
	// ccw_dxdy: for a +x,+y slope, kLeft is clockwise (false), kTop counterclockwise (true).
	if ccwDxDy(dVector{x: 1, y: 1}, rayDirLeft) {
		t.Fatal("ccwDxDy(+, kLeft) should be false")
	}
	if !ccwDxDy(dVector{x: 1, y: 1}, rayDirTop) {
		t.Fatal("ccwDxDy(+, kTop) should be true")
	}
	// sideways_overlap: for a horizontal ray, the point's Y must fall within [top, bottom].
	box := pathOpsBounds{left: 0, top: 0, right: 10, bottom: 10}
	if !sidewaysOverlap(box, geom.Point{X: 5, Y: 5}, rayDirLeft) {
		t.Fatal("interior point should overlap")
	}
	if sidewaysOverlap(box, geom.Point{X: 5, Y: 15}, rayDirLeft) {
		t.Fatal("point below the box should not overlap a horizontal ray")
	}
}

// TestGetTGuess checks the ray-origin t ladder and its direction offset against the first several tries.
func TestGetTGuess(t *testing.T) {
	cases := []struct {
		tTry    int
		wantT   float64
		wantOff int
	}{
		{tTry: 0, wantT: 0.5, wantOff: 0}, {tTry: 1, wantT: 0.5, wantOff: 1}, {tTry: 2, wantT: 0.25, wantOff: 0}, {tTry: 3, wantT: 0.25, wantOff: 1}, {tTry: 4, wantT: 0.375, wantOff: 0}, {tTry: 5, wantT: 0.375, wantOff: 1}, {tTry: 6, wantT: 0.625, wantOff: 0},
	}
	for _, c := range cases {
		var off int
		got := getTGuess(c.tTry, &off)
		if got != c.wantT || off != c.wantOff {
			t.Errorf("getTGuess(%d) = t %v off %d, want t %v off %d", c.tTry, got, off, c.wantT, c.wantOff)
		}
	}
}

// TestCurveIntercept checks the CurveIntercept dispatch for each verb: the returned roots must reproduce the axis
// intercept when the curve is evaluated at them.
func TestCurveIntercept(t *testing.T) {
	evalY := func(verb path.Verb, pts []geom.Point, w float32, tv float64) float64 {
		return curveDPointAtT(verb, pts, w, tv).y
	}
	evalX := func(verb path.Verb, pts []geom.Point, w float32, tv float64) float64 {
		return curveDPointAtT(verb, pts, w, tv).x
	}
	var roots [3]float64

	// Line, horizontal (xyIndex 0): Y = 5 on (0,0)-(10,10) => t = 0.5.
	line := []geom.Point{{X: 0, Y: 0}, {X: 10, Y: 10}}
	if n := curveIntercept(path.VerbLine, line, 1, 5, 0, roots[:]); n != 1 || math.Abs(roots[0]-0.5) > 1e-9 {
		t.Fatalf("line h intercept: n=%d roots[0]=%v, want 1/0.5", n, roots[0])
	}
	// Line, vertical (xyIndex 1): X = 5 => t = 0.5.
	if n := curveIntercept(path.VerbLine, line, 1, 5, 1, roots[:]); n != 1 || math.Abs(roots[0]-0.5) > 1e-9 {
		t.Fatalf("line v intercept: n=%d roots[0]=%v, want 1/0.5", n, roots[0])
	}
	// A horizontal line has no vertical crossing for a horizontal ray.
	flat := []geom.Point{{X: 0, Y: 0}, {X: 10, Y: 0}}
	if n := curveIntercept(path.VerbLine, flat, 1, 5, 0, roots[:]); n != 0 {
		t.Fatalf("horizontal line h intercept: n=%d, want 0", n)
	}

	// Quad (0,0),(10,10),(20,0): Y(t)=20t(1-t). Y=3.75 => t={0.25,0.75}.
	quad := []geom.Point{{X: 0, Y: 0}, {X: 10, Y: 10}, {X: 20, Y: 0}}
	n := curveIntercept(path.VerbQuad, quad, 1, 3.75, 0, roots[:])
	if n != 2 {
		t.Fatalf("quad h intercept: n=%d, want 2", n)
	}
	for i := 0; i < n; i++ {
		if math.Abs(evalY(path.VerbQuad, quad, 1, roots[i])-3.75) > 1e-6 {
			t.Fatalf("quad root %v does not reproduce Y=3.75 (got %v)", roots[i], evalY(path.VerbQuad, quad, 1, roots[i]))
		}
	}

	// Conic with weight 1 is a parabola identical to the quad above.
	conic := quad
	n = curveIntercept(path.VerbConic, conic, 1, 3.75, 0, roots[:])
	if n != 2 {
		t.Fatalf("conic h intercept: n=%d, want 2", n)
	}
	for i := 0; i < n; i++ {
		if math.Abs(evalY(path.VerbConic, conic, 1, roots[i])-3.75) > 1e-6 {
			t.Fatalf("conic root %v does not reproduce Y=3.75", roots[i])
		}
	}

	// Cubic (0,0),(10,30),(20,30),(30,0): Y(t)=90t(1-t). Vertical intercept X=15 => t=0.5 by symmetry.
	cubic := []geom.Point{{X: 0, Y: 0}, {X: 10, Y: 30}, {X: 20, Y: 30}, {X: 30, Y: 0}}
	n = curveIntercept(path.VerbCubic, cubic, 1, 15, 1, roots[:])
	if n < 1 {
		t.Fatalf("cubic v intercept: n=%d, want >=1", n)
	}
	for i := 0; i < n; i++ {
		if math.Abs(evalX(path.VerbCubic, cubic, 1, roots[i])-15) > 1e-6 {
			t.Fatalf("cubic root %v does not reproduce X=15 (got %v)", roots[i], evalX(path.VerbCubic, cubic, 1, roots[i]))
		}
	}
}

// TestUseInnerWinding checks useInnerWinding's truth table (equal magnitudes pick the negative; otherwise the smaller
// magnitude).
func TestUseInnerWinding(t *testing.T) {
	cases := []struct {
		outer, inner int
		want         bool
	}{
		{outer: 1, inner: 1, want: false},
		{outer: -1, inner: 1, want: true},
		{outer: 1, inner: -1, want: false},
		{outer: -1, inner: -1, want: true},
		{outer: 1, inner: 2, want: true},
		{outer: 2, inner: 1, want: false},
		{outer: 0, inner: 1, want: true},
		{outer: 1, inner: 0, want: false},
		{outer: -2, inner: 1, want: false},
		{outer: 1, inner: -2, want: true},
	}
	for _, c := range cases {
		if got := useInnerWinding(c.outer, c.inner); got != c.want {
			t.Errorf("useInnerWinding(%d,%d) = %v, want %v", c.outer, c.inner, got, c.want)
		}
	}
}

// TestSpanSignOppSign checks SpanSign/OppSign on a fresh (windValue 1, oppValue 0) square edge.
func TestSpanSignOppSign(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()
	head, _ := buildOpModel(t, p)
	seg := segsOf(realContours(head)[0])[0]
	head0 := &seg.head.opSpanBase
	tail0 := &seg.tail
	if got := spanSign(head0, tail0); got != -1 {
		t.Fatalf("spanSign(head,tail) = %d, want -1", got)
	}
	if got := spanSign(tail0, head0); got != 1 {
		t.Fatalf("spanSign(tail,head) = %d, want 1", got)
	}
	if got := oppSign(head0, tail0); got != 0 {
		t.Fatalf("oppSign(head,tail) = %d, want 0", got)
	}
}

// TestFindSortableTopSquare is the end-to-end ray-cast seed test: a single closed square. FindSortableTop projects a
// ray from a span, and markAndChaseWinding propagates the unit fill winding around the whole contour, so every span
// ends with windSum -1 (the closed CW square's fill) and no opposite winding.
func TestFindSortableTopSquare(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()
	head, state := buildOpModel(t, p)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head)

	top := findSortableTop(head)
	if top == nil {
		t.Fatal("findSortableTop returned nil")
	}
	if top.windSum == skMinS32 {
		t.Fatal("returned top span has no winding")
	}
	spanCount := 0
	for _, s := range segsOf(realContours(head)[0]) {
		for base := &s.head.opSpanBase; ; {
			if sp := base.upCastable(); sp != nil {
				spanCount++
				if sp.windSum != -1 {
					t.Fatalf("span at %v: windSum = %d, want -1", sp.pt(), sp.windSum)
				}
				if sp.oppSum != 0 {
					t.Fatalf("span at %v: oppSum = %d, want 0", sp.pt(), sp.oppSum)
				}
			}
			if base == &s.tail {
				break
			}
			base = base.upCast().next
		}
	}
	if spanCount != 4 {
		t.Fatalf("span count = %d, want 4", spanCount)
	}
}

// TestComputeWindSumTwoRects drives the binary (op) winding through computeWindSum on the interior crossing span where
// rectangle A's right edge enters rectangle B. The span sits inside B, so its opposite winding is nonzero.
func TestComputeWindSumTwoRects(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(20, 0).LineTo(20, 20).LineTo(0, 20).Close()
	b := path.New()
	b.MoveTo(10, 10).LineTo(30, 10).LineTo(30, 30).LineTo(10, 30).Close()
	head, state := buildOpModel(t, a, b)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head)

	aSegs := segsOf(realContours(head)[0])
	cross := spanBaseAtPt(aSegs[1], pt(20, 10))
	if cross == nil || cross == &aSegs[1].tail {
		t.Fatal("no interior crossing span at (20,10)")
	}
	sp := cross.upCast()
	got := sp.computeWindSum()
	if got == skMinS32 {
		t.Fatal("computeWindSum failed to seed a winding")
	}
	if got != -1 {
		t.Fatalf("computeWindSum = %d, want -1", got)
	}
	// The crossing sits inside rectangle B, so its opposite (operand) winding is nonzero.
	if sp.oppSum != -1 {
		t.Fatalf("cross oppSum = %d, want -1 (inside the other rectangle)", sp.oppSum)
	}
}

// TestComputeOneSumBowtie exercises the angle-loop transfer (the direct consumer of session 63's sorted angle loops):
// after seeding one angle's winding at the bow-tie crossing, ComputeOneSum transfers it onto the next angle in the
// counterclockwise loop.
func TestComputeOneSumBowtie(t *testing.T) {
	base := bowtieCrossAngle(t)
	base.starter().windSum = 1
	next := base.next
	if next.starter().windSum != skMinS32 {
		t.Fatal("next angle already has a winding before transfer")
	}
	if !computeOneSum(base, next, includeUnaryWinding) {
		t.Fatal("computeOneSum returned false")
	}
	if next.starter().windSum == skMinS32 {
		t.Fatal("computeOneSum did not transfer a winding to the next angle")
	}
	if next.starter().windSum != 1 {
		t.Fatalf("transferred windSum = %d, want 1", next.starter().windSum)
	}
}

// TestComputeOneSumBinary exercises the binary (op) angle-loop transfer: at the crossing where rectangle A's right edge
// meets rectangle B's top edge, ComputeOneSum with a binary include type transfers both the main and opposite winding
// onto the next angle (driving setUpWindingsBinary / markAngleBinary / updateOppWinding).
func TestComputeOneSumBinary(t *testing.T) {
	a := path.New()
	a.MoveTo(0, 0).LineTo(20, 0).LineTo(20, 20).LineTo(0, 20).Close()
	b := path.New()
	b.MoveTo(10, 10).LineTo(30, 10).LineTo(30, 30).LineTo(10, 30).Close()
	head, state := buildOpModel(t, a, b)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head)

	cross := spanBaseAtPt(segsOf(realContours(head)[0])[1], pt(20, 10))
	if cross == nil {
		t.Fatal("no crossing span at (20,10)")
	}
	base := cross.upCast().toAngle()
	if base == nil {
		base = cross.fromAngle()
	}
	if base == nil {
		t.Fatal("crossing carries no angle")
	}
	base.starter().windSum = 0
	base.starter().oppSum = 0
	next := base.next
	if next.starter().windSum != skMinS32 || next.starter().oppSum != skMinS32 {
		t.Fatal("next angle already has windings before transfer")
	}
	if !computeOneSum(base, next, includeBinarySingle) {
		t.Fatal("binary computeOneSum returned false")
	}
	if next.starter().windSum == skMinS32 || next.starter().oppSum == skMinS32 {
		t.Fatal("binary computeOneSum left the next angle unset")
	}
}

// TestComputeSumBowtie exercises the full computeSum loop orchestration: seeding one angle, it propagates the winding
// to every angle in the crossing's four-angle loop and reports the resulting sum.
func TestComputeSumBowtie(t *testing.T) {
	base := bowtieCrossAngle(t)
	loop := angleLoop(base)
	if len(loop) != 4 {
		t.Fatalf("bow-tie crossing loop size = %d, want 4", len(loop))
	}
	base.starter().windSum = 1
	// computeSum(start, end) resolves the junction angle via spanToAngle(end, start); base is base.start's toAngle, so
	// pass (base.end, base.start).
	got := base.segment().computeSum(base.end, base.start, includeUnaryWinding)
	if got != 1 {
		t.Fatalf("computeSum = %d, want 1", got)
	}
	for i, a := range loop {
		if a.starter().windSum == skMinS32 {
			t.Fatalf("angle %d still lacks a winding after computeSum", i)
		}
	}
}

// TestWindingSentinelsDistinct pins the three winding sentinels to their upstream Skia values. skMinS32 ("not yet
// computed") and skNaN32 ("degenerate angle loop") must not collide: computeSum returns either one, and its callers
// treat them differently.
func TestWindingSentinelsDistinct(t *testing.T) {
	if skMaxS32 != math.MaxInt32 {
		t.Fatalf("skMaxS32 = %d, want %d", skMaxS32, math.MaxInt32)
	}
	if skMinS32 != -math.MaxInt32 {
		t.Fatalf("skMinS32 = %d, want %d (Skia's SK_MinS32 == -SK_MaxS32)", skMinS32, -math.MaxInt32)
	}
	if skNaN32 != math.MinInt32 {
		t.Fatalf("skNaN32 = %d, want %d (Skia's SK_NaN32 == INT32_MIN)", skNaN32, math.MinInt32)
	}
	if skNaN32 == skMinS32 {
		t.Fatal("skNaN32 and skMinS32 must be distinct sentinels")
	}
}

// TestComputeSumUnsetIsNotNaN checks that a crossing whose angle loop carries no winding at all reports the
// "not yet computed" sentinel rather than the "degenerate loop" one: nothing transfers, but the loop is still sortable
// and updateWinding can re-seed it with a ray cast.
func TestComputeSumUnsetIsNotNaN(t *testing.T) {
	base := bowtieCrossAngle(t)
	for i, a := range angleLoop(base) {
		if a.starter().windSum != skMinS32 {
			t.Fatalf("angle %d already has a winding before computeSum", i)
		}
	}
	seg := base.segment()
	got := seg.computeSum(base.end, base.start, includeUnaryWinding)
	if got == skNaN32 {
		t.Fatal("computeSum reported a degenerate angle loop for a sortable, merely unseeded crossing")
	}
	if got != skMinS32 {
		t.Fatalf("computeSum = %d, want skMinS32 (%d)", got, skMinS32)
	}
	// The recovery the skNaN32 bail would have skipped: the ray cast seeds a real winding for the same span.
	if seeded := seg.updateWinding(base.start, base.end); seeded == skMinS32 {
		t.Fatal("updateWinding failed to re-seed the unset winding")
	}
}

// TestComputeSumDegenerateLoopIsNaN is the companion to TestComputeSumUnsetIsNotNaN: a span with no angle loop at all
// really is unsortable, and computeSum must say so with skNaN32.
func TestComputeSumDegenerateLoopIsNaN(t *testing.T) {
	head := buildContours(t, rectPath(0, 0, 10, 10, path.FillEvenOdd))
	seg := &realContours(head)[0].head // angles were never calculated, so spanToAngle yields nil
	if got := seg.computeSum(seg.head.next, &seg.head.opSpanBase, includeUnaryWinding); got != skNaN32 {
		t.Fatalf("computeSum on an angle-less span = %d, want skNaN32 (%d)", got, skNaN32)
	}
}

// TestFindNextWindingUnseededIsSortable walks the bow-tie crossing before any winding has been computed. computeSum
// returns skMinS32 there, which is not the unsortable sentinel, so the walk must proceed into updateWinding's ray cast
// instead of giving up and marking the span done.
func TestFindNextWindingUnseededIsSortable(t *testing.T) {
	base := bowtieCrossAngle(t)
	seg := base.segment()
	// Walk the crossing in the direction that has more than one candidate, so the trace has to measure angles (and
	// therefore call computeSum) rather than take isSimple's shortcut.
	probe := base.end
	probeStep := base.end.step(base.start)
	if seg.isSimple(&probe, &probeStep) != nil {
		t.Fatal("the crossing has a single candidate, so findNextWinding never reaches computeSum")
	}
	nextStart, nextEnd := base.end, base.start
	var chase []*opSpanBase
	unsortable := false
	next := seg.findNextWinding(&chase, &nextStart, &nextEnd, &unsortable)
	if unsortable {
		t.Fatal("findNextWinding marked an unseeded (but sortable) crossing unsortable")
	}
	if next == nil {
		t.Fatal("findNextWinding found no next segment at the bow-tie crossing")
	}
}

// TestNextChaseBackwardToHeadSpan walks nextChase backward into a pt-t whose opposite span is its segment's head
// (t==0, no prev). The chase has to stop there rather than dereference the missing predecessor.
func TestNextChaseBackwardToHeadSpan(t *testing.T) {
	head := buildContours(t, rectPath(0, 0, 10, 10, path.FillEvenOdd))
	segs := segsOf(realContours(head)[0])
	a, b := segs[0], segs[1]
	segs[len(segs)-1].joinEnds(a) // the contour join the walk phase makes: a's head pt-t loop holds the prior tail
	// Splice b's head pt-t (t==0) in as the first entry of a's head pt-t loop, the shape a coincident point landing on
	// both segments' start points leaves behind.
	a.head.ptT.insert(&b.head.ptT)
	start := &a.tail // t==1, so its prev is a's head (t==0)
	step := -1
	var minSpan *opSpan
	var last *opSpanBase
	if got := a.nextChase(&start, &step, &minSpan, &last); got != nil {
		t.Fatalf("nextChase should stop at a head-span pt-t, got segment %p", got)
	}
	if start != &a.tail || step != -1 || minSpan != nil || last != nil {
		t.Fatal("nextChase should leave its outputs untouched when the chase stops")
	}
}

// TestNextChaseBackwardToInteriorSpan is the companion to TestNextChaseBackwardToHeadSpan: when the opposite pt-t's
// span does have a predecessor, the backward chase continues onto that segment.
func TestNextChaseBackwardToInteriorSpan(t *testing.T) {
	head := buildContours(t, rectPath(0, 0, 10, 10, path.FillEvenOdd))
	segs := segsOf(realContours(head)[0])
	a := segs[0]
	// The contour join places the previous segment's tail (t==1, prev == its head) in a's head pt-t loop.
	segs[len(segs)-1].joinEnds(a)
	otherPtT := a.head.ptT.next
	other := otherPtT.segment()
	if otherPtT.span != &other.tail {
		t.Fatalf("expected a's head pt-t loop to start at the previous segment's tail, t = %v", otherPtT.t)
	}
	start := &a.tail
	step := -1
	var minSpan *opSpan
	var last *opSpanBase
	got := a.nextChase(&start, &step, &minSpan, &last)
	if got != other {
		t.Fatalf("nextChase = %p, want the joined segment %p", got, other)
	}
	if start != &other.tail || step != -1 || minSpan != &other.head || last != nil {
		t.Fatal("nextChase should hand back the joined segment's tail span and its starter")
	}
}

// bowtieCrossAngle builds the self-crossing bow-tie, intersects and sorts it, and returns the toAngle at the (5,5)
// crossing (a four-angle loop).
func bowtieCrossAngle(t *testing.T) *opAngle {
	t.Helper()
	p := path.New()
	p.MoveTo(0, 0).LineTo(10, 10).LineTo(10, 0).LineTo(0, 10).Close()
	head, state := buildOpModel(t, p)
	coincidence := newOpCoincidence(state)
	addIntersections(head, coincidence)
	calcAndSortAngles(t, head)
	cross := spanBaseAtPt(segsOf(realContours(head)[0])[0], pt(5, 5))
	if cross == nil {
		t.Fatal("bow-tie has no crossing span at (5,5)")
	}
	base := cross.upCast().toAngle()
	if base == nil {
		t.Fatal("crossing span carries no toAngle")
	}
	return base
}
