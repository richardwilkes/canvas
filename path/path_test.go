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
	"math"
	"sync"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func pts(vals ...float32) []geom.Point {
	out := make([]geom.Point, len(vals)/2)
	for i := range out {
		out[i] = geom.Pt(vals[i*2], vals[i*2+1])
	}
	return out
}

func checkVerbs(t *testing.T, p *Path, want ...Verb) {
	t.Helper()
	if len(p.verbs) != len(want) {
		t.Fatalf("verb count = %d, want %d (verbs %v)", len(p.verbs), len(want), p.verbs)
	}
	for i, v := range want {
		if p.verbs[i] != v {
			t.Fatalf("verb[%d] = %d, want %d", i, p.verbs[i], v)
		}
	}
}

func checkPoints(t *testing.T, p *Path, want []geom.Point) {
	t.Helper()
	if len(p.points) != len(want) {
		t.Fatalf("point count = %d, want %d (points %v)", len(p.points), len(want), p.points)
	}
	for i, pt := range want {
		if p.points[i] != pt {
			t.Fatalf("point[%d] = %v, want %v", i, p.points[i], pt)
		}
	}
}

func TestEmptyPath(t *testing.T) {
	p := New()
	if !p.IsEmpty() {
		t.Error("new path should be empty")
	}
	if p.FillType() != FillWinding {
		t.Error("default fill type should be winding")
	}
	if got := p.Bounds(); got != (geom.Rect{}) {
		t.Errorf("empty bounds = %v", got)
	}
	if !p.IsFinite() {
		t.Error("empty path is finite")
	}
	if _, ok := p.LastPt(); ok {
		t.Error("empty path has no last point")
	}
	if p.Point(0) != (geom.Point{}) {
		t.Error("out-of-range point should be (0,0)")
	}
	// Close on empty path adds nothing.
	p.Close()
	if !p.IsEmpty() {
		t.Error("close on empty path should add nothing")
	}
}

func TestMoveToReplacesTrailingMove(t *testing.T) {
	p := New()
	p.MoveTo(1, 1).MoveTo(2, 2).MoveTo(3, 3)
	checkVerbs(t, p, VerbMove)
	checkPoints(t, p, pts(3, 3))
	// A segment then another moveTo appends.
	p.LineTo(4, 4).MoveTo(5, 5)
	checkVerbs(t, p, VerbMove, VerbLine, VerbMove)
	checkPoints(t, p, pts(3, 3, 4, 4, 5, 5))
}

func TestInjectMoveTo(t *testing.T) {
	// lineTo on an empty path injects a moveTo (0,0).
	p := New()
	p.LineTo(10, 10)
	checkVerbs(t, p, VerbMove, VerbLine)
	checkPoints(t, p, pts(0, 0, 10, 10))

	// After close, the next segment injects a moveTo back to the contour start.
	p = New()
	p.MoveTo(5, 5).LineTo(10, 5).Close().LineTo(7, 7)
	checkVerbs(t, p, VerbMove, VerbLine, VerbClose, VerbMove, VerbLine)
	checkPoints(t, p, pts(5, 5, 10, 5, 5, 5, 7, 7))
}

func TestRMoveToAfterClose(t *testing.T) {
	p := New()
	p.MoveTo(10, 10).LineTo(20, 20).Close().RMoveTo(1, 2)
	// rMoveTo after close is relative to the previous contour's moveTo.
	checkVerbs(t, p, VerbMove, VerbLine, VerbClose, VerbMove)
	checkPoints(t, p, pts(10, 10, 20, 20, 11, 12))

	// rMoveTo mid-contour is relative to the last point.
	p = New()
	p.MoveTo(10, 10).LineTo(20, 20).RMoveTo(1, 2)
	checkPoints(t, p, pts(10, 10, 20, 20, 21, 22))
}

func TestRelativeOps(t *testing.T) {
	p := New()
	p.MoveTo(10, 20).RLineTo(5, 5).RQuadTo(1, 2, 3, 4).RConicTo(1, 1, 2, 2, 2).RCubicTo(1, 0, 2, 0, 3, 0)
	checkVerbs(t, p, VerbMove, VerbLine, VerbQuad, VerbConic, VerbCubic)
	checkPoints(t, p, pts(
		10, 20,
		15, 25,
		16, 27, 18, 29,
		19, 30, 20, 31,
		21, 31, 22, 31, 23, 31,
	))
	if p.conicWeights[0] != 2 {
		t.Errorf("conic weight = %v", p.conicWeights[0])
	}
}

func TestConicToDegenerateWeights(t *testing.T) {
	// w <= 0 becomes a single line to the end point.
	p := New()
	p.MoveTo(0, 0).ConicTo(5, 5, 10, 0, 0)
	checkVerbs(t, p, VerbMove, VerbLine)
	checkPoints(t, p, pts(0, 0, 10, 0))

	// NaN weight: !(w > 0) catches it, same as w <= 0.
	p = New()
	p.MoveTo(0, 0).ConicTo(5, 5, 10, 0, float32(math.NaN()))
	checkVerbs(t, p, VerbMove, VerbLine)

	// +Inf weight becomes two lines.
	p = New()
	p.MoveTo(0, 0).ConicTo(5, 5, 10, 0, float32(math.Inf(1)))
	checkVerbs(t, p, VerbMove, VerbLine, VerbLine)
	checkPoints(t, p, pts(0, 0, 5, 5, 10, 0))

	// w == 1 becomes a quad.
	p = New()
	p.MoveTo(0, 0).ConicTo(5, 5, 10, 0, 1)
	checkVerbs(t, p, VerbMove, VerbQuad)
	if len(p.conicWeights) != 0 {
		t.Error("quad should not record a weight")
	}
}

func TestCloseSemantics(t *testing.T) {
	p := New()
	p.MoveTo(0, 0).LineTo(1, 1).Close().Close() // second close is a no-op
	checkVerbs(t, p, VerbMove, VerbLine, VerbClose)

	// Close directly after a moveTo is recorded.
	p = New()
	p.MoveTo(0, 0).Close()
	checkVerbs(t, p, VerbMove, VerbClose)
}

func TestBounds(t *testing.T) {
	p := New()
	p.MoveTo(10, 10).LineTo(20, 5).LineTo(15, 25)
	want := geom.RectLTRB(10, 5, 20, 25)
	if got := p.Bounds(); got != want {
		t.Errorf("bounds = %v, want %v", got, want)
	}
	// Bounds include trailing moveTo points.
	p.MoveTo(100, 100)
	if got := p.Bounds(); got != geom.RectLTRB(10, 5, 100, 100) {
		t.Errorf("bounds with trailing move = %v", got)
	}
	// Non-finite points empty the bounds.
	p.LineTo(float32(math.Inf(1)), 0)
	if got := p.Bounds(); got != (geom.Rect{}) {
		t.Errorf("non-finite bounds = %v", got)
	}
	if p.IsFinite() {
		t.Error("path with inf point is not finite")
	}
}

func TestAddRectVerbsAndPoints(t *testing.T) {
	r := geom.RectLTRB(10, 20, 30, 40)
	p := New().AddRect(r, geom.DirectionCW)
	checkVerbs(t, p, VerbMove, VerbLine, VerbLine, VerbLine, VerbClose)
	checkPoints(t, p, pts(10, 20, 30, 20, 30, 40, 10, 40))
	if !p.IsConvex() {
		t.Error("rect is convex")
	}
	if p.GetConvexity() != ConvexityConvexCW {
		t.Errorf("rect convexity = %v", p.GetConvexity())
	}

	// CCW with start index 2 begins at the lower-right corner and winds the other way.
	p = New().AddRectWithStart(r, geom.DirectionCCW, 2)
	checkPoints(t, p, pts(30, 40, 30, 20, 10, 20, 10, 40))
	if p.GetConvexity() != ConvexityConvexCCW {
		t.Errorf("ccw rect convexity = %v", p.GetConvexity())
	}
}

func TestAddOvalLayout(t *testing.T) {
	r := geom.RectLTRB(0, 0, 20, 10)
	p := New().AddOval(r, geom.DirectionCW)
	checkVerbs(t, p, VerbMove, VerbConic, VerbConic, VerbConic, VerbConic, VerbClose)
	// Legacy start index 1: begins at (right, centerY).
	checkPoints(t, p, pts(
		20, 5,
		20, 10, 10, 10,
		0, 10, 0, 5,
		0, 0, 10, 0,
		20, 0, 20, 5,
	))
	for _, w := range p.conicWeights {
		if !geom.ScalarNearlyEqual(w, sqrt2Over2) {
			t.Errorf("oval conic weight = %v", w)
		}
	}
	if bounds, ok := p.IsOval(); !ok || bounds != r {
		t.Errorf("IsOval = %v, %v", bounds, ok)
	}
	// A second shape makes it not-an-oval.
	p.AddRect(r, geom.DirectionCW)
	if _, ok := p.IsOval(); ok {
		t.Error("two shapes should not be an oval")
	}
}

func TestAddRRectLayout(t *testing.T) {
	rr := geom.MakeRRect(geom.RectLTRB(0, 0, 100, 60), 10, 5)
	p := New().AddRRect(rr, geom.DirectionCW)
	// Legacy CW start index 6 is even => starts with a line.
	checkVerbs(t, p,
		VerbMove,
		VerbLine, VerbConic,
		VerbLine, VerbConic,
		VerbLine, VerbConic,
		VerbLine, VerbConic,
		VerbClose)
	// Start point: rrect point index 6 = (left, bottom - ry).
	if p.points[0] != geom.Pt(0, 55) {
		t.Errorf("rrect start = %v", p.points[0])
	}
	if !p.IsConvex() {
		t.Error("rrect is convex")
	}

	// Degenerate radii collapse to a rect.
	p = New().AddRoundRect(geom.RectLTRB(0, 0, 10, 10), 0, 0, geom.DirectionCW)
	checkVerbs(t, p, VerbMove, VerbLine, VerbLine, VerbLine, VerbClose)

	// Radii >= half dimensions collapse to an oval.
	p = New().AddRoundRect(geom.RectLTRB(0, 0, 10, 10), 5, 5, geom.DirectionCW)
	checkVerbs(t, p, VerbMove, VerbConic, VerbConic, VerbConic, VerbConic, VerbClose)

	// Negative radii are a no-op.
	p = New().AddRoundRect(geom.RectLTRB(0, 0, 10, 10), -1, 5, geom.DirectionCW)
	if !p.IsEmpty() {
		t.Error("negative radius round rect should be a no-op")
	}
}

func TestAddCircle(t *testing.T) {
	p := New().AddCircle(50, 50, 0, geom.DirectionCW)
	if !p.IsEmpty() {
		t.Error("zero radius circle should be a no-op")
	}
	p = New().AddCircle(50, 50, -3, geom.DirectionCW)
	if !p.IsEmpty() {
		t.Error("negative radius circle should be a no-op")
	}
	p = New().AddCircle(50, 50, 10, geom.DirectionCW)
	if got := p.Bounds(); got != geom.RectLTRB(40, 40, 60, 60) {
		t.Errorf("circle bounds = %v", got)
	}
}

func TestAddPoly(t *testing.T) {
	p := New().AddPoly(pts(0, 0, 10, 0, 10, 10), true)
	checkVerbs(t, p, VerbMove, VerbLine, VerbLine, VerbClose)
	checkPoints(t, p, pts(0, 0, 10, 0, 10, 10))

	// Unlike MoveTo, AddPoly always appends its move verb.
	p = New()
	p.MoveTo(5, 5)
	p.AddPoly(pts(0, 0, 1, 1), false)
	checkVerbs(t, p, VerbMove, VerbMove, VerbLine)
	checkPoints(t, p, pts(5, 5, 0, 0, 1, 1))

	// Empty input is a no-op.
	p = New().AddPoly(nil, true)
	if !p.IsEmpty() {
		t.Error("empty poly should be a no-op")
	}
}

func TestAddPathModes(t *testing.T) {
	src := New()
	src.MoveTo(10, 10).LineTo(20, 10).Close()

	// Append into an empty path replaces it (fill type preserved).
	dst := New()
	dst.SetFillType(FillEvenOdd)
	dst.AddPath(src, AddPathAppend)
	checkVerbs(t, dst, VerbMove, VerbLine, VerbClose)
	if dst.FillType() != FillEvenOdd {
		t.Error("fill type should be preserved on replace")
	}

	// Append with offset.
	dst = New()
	dst.MoveTo(0, 0).LineTo(1, 1)
	dst.AddPathOffset(src, 100, 200, AddPathAppend)
	checkVerbs(t, dst, VerbMove, VerbLine, VerbMove, VerbLine, VerbClose)
	checkPoints(t, dst, pts(0, 0, 1, 1, 110, 210, 120, 210))

	// Extend connects with a line when the endpoints differ.
	dst = New()
	dst.MoveTo(0, 0).LineTo(1, 1)
	dst.AddPath(src, AddPathExtend)
	checkVerbs(t, dst, VerbMove, VerbLine, VerbLine, VerbLine, VerbClose)
	checkPoints(t, dst, pts(0, 0, 1, 1, 10, 10, 20, 10))

	// Extend skips the degenerate connecting line when the endpoints match.
	dst = New()
	dst.MoveTo(0, 0).LineTo(10, 10)
	dst.AddPath(src, AddPathExtend)
	checkVerbs(t, dst, VerbMove, VerbLine, VerbLine, VerbClose)

	// Adding a path to itself works.
	self := New()
	self.MoveTo(0, 0).LineTo(5, 5)
	self.AddPath(self, AddPathAppend)
	checkVerbs(t, self, VerbMove, VerbLine, VerbMove, VerbLine)
	checkPoints(t, self, pts(0, 0, 5, 5, 0, 0, 5, 5))
}

func TestReverseAddPath(t *testing.T) {
	src := New()
	src.MoveTo(0, 0).LineTo(10, 0).QuadTo(15, 5, 10, 10).Close()

	dst := New()
	dst.ReverseAddPath(src)
	checkVerbs(t, dst, VerbMove, VerbQuad, VerbLine, VerbClose)
	checkPoints(t, dst, pts(
		10, 10,
		15, 5, 10, 0,
		0, 0,
	))
}

func TestReverseAddPathTwoContours(t *testing.T) {
	src := New()
	src.MoveTo(0, 0).LineTo(1, 0)
	src.MoveTo(10, 10).LineTo(11, 10)

	dst := New()
	dst.ReverseAddPath(src)
	checkVerbs(t, dst, VerbMove, VerbLine, VerbMove, VerbLine)
	checkPoints(t, dst, pts(11, 10, 10, 10, 1, 0, 0, 0))
}

func TestLastPtAndSetLastPt(t *testing.T) {
	p := New()
	p.MoveTo(1, 2).LineTo(3, 4)
	if pt, ok := p.LastPt(); !ok || pt != geom.Pt(3, 4) {
		t.Errorf("LastPt = %v, %v", pt, ok)
	}
	p.SetLastPt(9, 9)
	if pt, _ := p.LastPt(); pt != geom.Pt(9, 9) {
		t.Errorf("after SetLastPt = %v", pt)
	}
	// SetLastPt on an empty path is a moveTo.
	p = New()
	p.SetLastPt(1, 1)
	checkVerbs(t, p, VerbMove)
}

// TestSetLastPtInvalidatesCaches covers the caches that moving the last point has to dirty: the bounds, the cached
// convexity and the simple-shape (oval/rrect) identity.
func TestSetLastPtInvalidatesCaches(t *testing.T) {
	// A rect is convex; pulling its last corner into the middle makes it concave.
	p := New().AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW)
	if c := p.GetConvexity(); !c.IsConvex() {
		t.Fatalf("rect convexity = %v, want convex", c)
	}
	p.SetLastPt(8, 3) // inside the triangle formed by the other three corners, making that vertex reflex
	if c := p.GetConvexity(); c != ConvexityConcave {
		t.Errorf("convexity after SetLastPt = %v, want concave (stale cache)", c)
	}

	// The bounds must shrink to match the moved point.
	p = New().MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10)
	if got := p.Bounds(); got != geom.RectLTRB(0, 0, 10, 10) {
		t.Fatalf("bounds = %v", got)
	}
	p.SetLastPt(2, 3)
	if got := p.Bounds(); got != geom.RectLTRB(0, 0, 10, 3) {
		t.Errorf("bounds after SetLastPt = %v, want (0,0,10,3)", got)
	}

	// An oval that has had a point moved is no longer an oval.
	p = New().AddOval(geom.RectLTRB(0, 0, 10, 20), geom.DirectionCW)
	if _, ok := p.IsOval(); !ok {
		t.Fatal("AddOval did not record the oval identity")
	}
	p.SetLastPt(-5, -5)
	if _, ok := p.IsOval(); ok {
		t.Error("path is still reported as an oval after SetLastPt")
	}

	// A rrect that has had a point moved is no longer a rrect.
	p = New().AddRRect(geom.MakeRRect(geom.RectLTRB(0, 0, 10, 20), 3, 3), geom.DirectionCW)
	if p.isa != isARRect {
		t.Fatal("AddRRect did not record the rrect identity")
	}
	p.SetLastPt(-5, -5)
	if p.isa != isAGeneral {
		t.Error("path is still reported as a rrect after SetLastPt")
	}
}

// TestPrimedPathConcurrentReads exercises the concurrency contract on the Path doc: an immutable path is safe to share
// only once every lazy cache has been primed. Under -race this fails if any of the reads below still writes to the
// path, which is what a new lazy cache (or a missing entry in the documented priming sequence) would do.
func TestPrimedPathConcurrentReads(t *testing.T) {
	// Built vertex by vertex, so nothing pre-seeds the convexity the way AddRect does.
	p := New().MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 10).Close()

	// Bounds alone is not sufficient: it leaves the convexity and generation-ID caches unwritten.
	p.Bounds()
	if p.convexity != ConvexityUnknown {
		t.Fatal("Bounds unexpectedly primed the convexity cache")
	}
	if p.genID != 0 {
		t.Fatal("Bounds unexpectedly primed the generation ID")
	}

	// The full priming sequence named in the type doc.
	p.Bounds()
	p.IsFinite()
	p.GetConvexity()
	p.GenerationID()

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = p.Bounds()
			_ = p.IsFinite()
			_ = p.GetConvexity()
			_ = p.IsConvex()
			_ = p.GenerationID()
			_, _ = p.IsOval()
			_, _ = p.IsLine()
			_, _ = p.IsRect()
		}()
	}
	wg.Wait()
}

func TestIsLine(t *testing.T) {
	p := New()
	p.MoveTo(1, 2).LineTo(3, 4)
	line, ok := p.IsLine()
	if !ok || line[0] != geom.Pt(1, 2) || line[1] != geom.Pt(3, 4) {
		t.Errorf("IsLine = %v, %v", line, ok)
	}
	p.LineTo(5, 6)
	if _, ok = p.IsLine(); ok {
		t.Error("two-segment path is not a line")
	}
}

func TestGetPointsAndCounts(t *testing.T) {
	p := New()
	p.MoveTo(0, 0).QuadTo(1, 1, 2, 0).Close()
	if p.CountPoints() != 3 || p.CountVerbs() != 3 {
		t.Errorf("counts = %d, %d", p.CountPoints(), p.CountVerbs())
	}
	dst := make([]geom.Point, 2)
	if n := p.Points(dst); n != 3 {
		t.Errorf("Points returned %d", n)
	}
	if dst[0] != geom.Pt(0, 0) || dst[1] != geom.Pt(1, 1) {
		t.Errorf("copied points = %v", dst)
	}
	if p.SegmentMasks() != SegmentQuad {
		t.Errorf("segment masks = %d", p.SegmentMasks())
	}
}

func TestResetAndRewind(t *testing.T) {
	p := New()
	p.MoveTo(1, 1).LineTo(2, 2).Close()
	p.SetFillType(FillInverseEvenOdd)
	p.Reset()
	if !p.IsEmpty() || p.FillType() != FillWinding {
		t.Error("reset should empty the path and restore winding fill")
	}
	// After reset, lineTo injects (0,0) again.
	p.LineTo(3, 3)
	checkPoints(t, p, pts(0, 0, 3, 3))

	p = New()
	p.MoveTo(1, 1).LineTo(2, 2)
	p.Rewind()
	if !p.IsEmpty() {
		t.Error("rewind should empty the path")
	}
}

func TestCloneAndEqual(t *testing.T) {
	p := New()
	p.MoveTo(1, 1).ConicTo(2, 2, 3, 1, 0.5).Close()
	q := p.Clone()
	if !Equal(p, q) {
		t.Error("clone should be equal")
	}
	q.SetFillType(FillEvenOdd)
	if Equal(p, q) {
		t.Error("different fill types are not equal")
	}
	q.SetFillType(FillWinding)
	q.LineTo(5, 5)
	if Equal(p, q) {
		t.Error("different verbs are not equal")
	}
	// Mutating the clone must not affect the original.
	if p.CountVerbs() != 3 {
		t.Error("clone mutation leaked into original")
	}
}

// TestEqualUsesIEEECompare pins the comparison Equal documents: coordinates and conic weights go through the IEEE ==
// operator, not a bit comparison of their encodings, so -0 matches 0 and NaN matches nothing (other than through the
// pointer-identity shortcut).
func TestEqualUsesIEEECompare(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	if math.Float32bits(negZero) == math.Float32bits(0) {
		t.Fatal("precondition: -0 and 0 have different bit patterns")
	}
	p := New().MoveTo(negZero, negZero).LineTo(1, 2)
	q := New().MoveTo(0, 0).LineTo(1, 2)
	if !Equal(p, q) {
		t.Error("-0 and 0 coordinates should compare equal")
	}

	nan := float32(math.NaN())
	a := New().MoveTo(nan, 0).LineTo(1, 2)
	b := New().MoveTo(nan, 0).LineTo(1, 2)
	if Equal(a, b) {
		t.Error("NaN coordinates should never compare equal")
	}
	if !Equal(a, a) {
		t.Error("a path must be equal to itself even when it holds a NaN")
	}

	// Conic weights are compared the same way. ConicTo folds a NaN weight into a line, so the weight is poked in
	// directly to reach the weight comparison.
	c := New().MoveTo(0, 0).ConicTo(1, 1, 2, 0, 0.5)
	d := c.Clone()
	c.conicWeights[0] = nan
	d.conicWeights[0] = nan
	if Equal(c, d) {
		t.Error("NaN conic weights should never compare equal")
	}
}
