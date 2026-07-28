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
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// sampleTightBounds computes a reference tight bounds by dense sampling of every segment.
func sampleTightBounds(t *testing.T, p *Path) geom.Rect {
	t.Helper()
	minX, minY := float32(math.Inf(1)), float32(math.Inf(1))
	maxX, maxY := float32(math.Inf(-1)), float32(math.Inf(-1))
	add := func(pt geom.Point) {
		if pt.X < minX {
			minX = pt.X
		}
		if pt.Y < minY {
			minY = pt.Y
		}
		if pt.X > maxX {
			maxX = pt.X
		}
		if pt.Y > maxY {
			maxY = pt.Y
		}
	}
	iter := newRawIter(p)
	for {
		verb, pts, w, ok := iter.next()
		if !ok {
			break
		}
		const steps = 2048
		switch verb {
		case VerbMove:
			add(pts[0])
		case VerbLine:
			add(pts[0])
			add(pts[1])
		case VerbQuad:
			for i := 0; i <= steps; i++ {
				add(geom.EvalQuadAt(pts[:3], float32(i)/steps))
			}
		case VerbConic:
			conic := geom.MakeConic(pts[0], pts[1], pts[2], w)
			for i := 0; i <= steps; i++ {
				add(conic.EvalAt(float32(i) / steps))
			}
		case VerbCubic:
			for i := 0; i <= steps; i++ {
				add(geom.EvalCubicPosAt(pts[:4], float32(i)/steps))
			}
		case VerbClose:
		}
	}
	return geom.RectLTRB(minX, minY, maxX, maxY)
}

func rectsNearlyEqual(a, b geom.Rect, tol float32) bool {
	return geom.ScalarAbs(a.Left-b.Left) <= tol && geom.ScalarAbs(a.Top-b.Top) <= tol &&
		geom.ScalarAbs(a.Right-b.Right) <= tol && geom.ScalarAbs(a.Bottom-b.Bottom) <= tol
}

func TestTightBounds(t *testing.T) {
	// Lines only: tight bounds == bounds.
	p := New()
	p.MoveTo(0, 0).LineTo(10, 20).LineTo(-5, 8)
	if got := p.ComputeTightBounds(); got != p.Bounds() {
		t.Errorf("line-only tight bounds = %v, want %v", got, p.Bounds())
	}

	// A quad whose control point pushes the control bounds past the curve.
	p = New()
	p.MoveTo(0, 0).QuadTo(50, 100, 100, 0)
	got := p.ComputeTightBounds()
	want := sampleTightBounds(t, p)
	if !rectsNearlyEqual(got, want, 0.05) {
		t.Errorf("quad tight bounds = %v, want ~%v", got, want)
	}
	if got.Bottom >= p.Bounds().Bottom {
		t.Error("tight bounds should be tighter than control bounds")
	}

	// Cubic with interior extrema on both axes.
	p = New()
	p.MoveTo(10, 10).CubicTo(-30, 40, 150, -80, 20, 30)
	got = p.ComputeTightBounds()
	want = sampleTightBounds(t, p)
	if !rectsNearlyEqual(got, want, 0.1) {
		t.Errorf("cubic tight bounds = %v, want ~%v", got, want)
	}

	// Conic (oval): tight bounds of a full oval equal the oval rect.
	oval := geom.RectLTRB(5, 10, 45, 30)
	p = New().AddOval(oval, geom.DirectionCW)
	got = p.ComputeTightBounds()
	if !rectsNearlyEqual(got, oval, 0.001) {
		t.Errorf("oval tight bounds = %v, want %v", got, oval)
	}

	// Empty path.
	if got = New().ComputeTightBounds(); got != (geom.Rect{}) {
		t.Errorf("empty tight bounds = %v", got)
	}
}

func TestContainsRect(t *testing.T) {
	p := New().AddRect(geom.RectLTRB(10, 10, 30, 30), geom.DirectionCW)
	cases := []struct {
		x, y float32
		want bool
	}{
		{x: 20, y: 20, want: true},
		{x: 10, y: 20, want: true}, // points exactly on any edge count as contained (on-curve detection)
		{x: 30, y: 20, want: true},
		{x: 20, y: 10, want: true},
		{x: 20, y: 30, want: true},
		{x: 5, y: 20, want: false},
		{x: 35, y: 20, want: false},
		{x: 20, y: 5, want: false},
	}
	for _, c := range cases {
		if got := p.Contains(c.x, c.y); got != c.want {
			t.Errorf("Contains(%g, %g) = %v, want %v", c.x, c.y, got, c.want)
		}
	}
}

func TestContainsInverse(t *testing.T) {
	p := New().AddRect(geom.RectLTRB(10, 10, 30, 30), geom.DirectionCW)
	p.SetFillType(FillInverseWinding)
	if p.Contains(20, 20) {
		t.Error("inside should be false for inverse fill")
	}
	if !p.Contains(0, 0) {
		t.Error("outside should be true for inverse fill")
	}
	// Empty path with inverse fill contains everything.
	e := New()
	e.SetFillType(FillInverseWinding)
	if !e.Contains(123, 456) {
		t.Error("empty inverse path contains all points")
	}
}

func TestContainsDonut(t *testing.T) {
	// Outer CW circle with inner CCW circle: winding rule leaves the hole empty.
	p := New()
	p.AddCircle(50, 50, 40, geom.DirectionCW)
	p.AddCircle(50, 50, 10, geom.DirectionCCW)
	if !p.Contains(50, 75) {
		t.Error("ring interior should be contained")
	}
	if p.Contains(50, 50) {
		t.Error("hole should not be contained (winding, opposite directions)")
	}

	// Same direction: winding fills the hole, even-odd leaves it empty.
	p = New()
	p.AddCircle(50, 50, 40, geom.DirectionCW)
	p.AddCircle(50, 50, 10, geom.DirectionCW)
	if !p.Contains(50, 50) {
		t.Error("winding with same direction fills the hole")
	}
	p.SetFillType(FillEvenOdd)
	if p.Contains(50, 50) {
		t.Error("even-odd leaves the hole empty")
	}
	if !p.Contains(50, 75) {
		t.Error("even-odd ring interior")
	}
}

func TestContainsCurves(t *testing.T) {
	// Circle built from conics.
	p := New().AddCircle(50, 50, 30, geom.DirectionCW)
	if !p.Contains(50, 50) || !p.Contains(70, 50) {
		t.Error("circle should contain interior points")
	}
	if p.Contains(81, 50) || p.Contains(50, 81) || p.Contains(72, 72) {
		t.Error("circle should not contain exterior points")
	}

	// Unclosed contours are force-closed for hit testing.
	p = New()
	p.MoveTo(0, 0).QuadTo(50, 100, 100, 0)
	if !p.Contains(50, 25) {
		t.Error("point under the quad hump should be contained")
	}
	if p.Contains(50, 60) {
		t.Error("point above the quad should not be contained")
	}

	// Cubic region.
	p = New()
	p.MoveTo(0, 0).CubicTo(30, 90, 70, 90, 100, 0).Close()
	if !p.Contains(50, 30) {
		t.Error("cubic interior")
	}
	if p.Contains(50, 70) {
		t.Error("cubic exterior")
	}
}

func TestConvexity(t *testing.T) {
	// Point and line are (degenerately) convex.
	p := New()
	p.MoveTo(0, 0)
	if !p.IsConvex() {
		t.Error("single move is convex")
	}
	p = New()
	p.MoveTo(0, 0).LineTo(1, 1)
	if !p.IsConvex() {
		t.Error("single line is convex")
	}

	// Triangle CCW.
	p = New()
	p.MoveTo(0, 0).LineTo(10, 0).LineTo(5, -10).Close()
	if got := p.GetConvexity(); got != ConvexityConvexCCW {
		t.Errorf("triangle convexity = %v", got)
	}

	// Star (self-intersecting) is concave.
	p = New()
	p.MoveTo(50, 0).LineTo(80, 90).LineTo(0, 30).LineTo(100, 30).LineTo(20, 90).Close()
	if p.IsConvex() {
		t.Error("star is concave")
	}

	// L-shape is concave.
	p = New()
	p.MoveTo(0, 0).LineTo(20, 0).LineTo(20, 10).LineTo(10, 10).LineTo(10, 20).LineTo(0, 20).Close()
	if p.IsConvex() {
		t.Error("L-shape is concave")
	}

	// Two contours are never convex.
	p = New()
	p.AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW)
	p.AddRect(geom.RectLTRB(20, 20, 30, 30), geom.DirectionCW)
	if p.IsConvex() {
		t.Error("two contours are concave")
	}

	// Non-finite points are concave.
	p = New()
	p.MoveTo(0, 0).LineTo(float32(math.Inf(1)), 1).LineTo(1, 1).Close()
	if p.IsConvex() {
		t.Error("non-finite path reports concave")
	}
}

func TestTransformTranslateScale(t *testing.T) {
	p := New()
	p.MoveTo(1, 2).LineTo(3, 4).QuadTo(5, 6, 7, 8)
	p.Bounds() // prime the bounds cache to exercise the fast path

	var m geom.Matrix
	m.SetScaleTranslate(2, 3, 10, 20)
	dst := New()
	p.TransformTo(&m, dst)
	checkPoints(t, dst, pts(12, 26, 16, 32, 20, 38, 24, 44))
	if got, want := dst.Bounds(), geom.RectLTRB(12, 26, 24, 44); got != want {
		t.Errorf("transformed bounds = %v, want %v", got, want)
	}
	// Source unchanged.
	checkPoints(t, p, pts(1, 2, 3, 4, 5, 6, 7, 8))

	// In-place.
	p.Transform(&m)
	checkPoints(t, p, pts(12, 26, 16, 32, 20, 38, 24, 44))
}

func TestTransformPreservesOval(t *testing.T) {
	p := New().AddOval(geom.RectLTRB(0, 0, 20, 10), geom.DirectionCW)

	// Scale/translate keeps oval-ness.
	var m geom.Matrix
	m.SetScaleTranslate(2, 2, 5, 5)
	p.Transform(&m)
	if bounds, ok := p.IsOval(); !ok || bounds != geom.RectLTRB(5, 5, 45, 25) {
		t.Errorf("scaled oval = %v, %v", bounds, ok)
	}

	// 45-degree rotate does not stay rect, so the oval flag drops.
	q := New().AddOval(geom.RectLTRB(0, 0, 20, 10), geom.DirectionCW)
	var r geom.Matrix
	r.SetRotate(45)
	q.Transform(&r)
	if _, ok := q.IsOval(); ok {
		t.Error("rotated (45deg) oval should not report as oval")
	}
}

// TestTransformShapeIdentity covers the whole of what a transform carries for a simple shape: the identity itself (an
// oval stays an oval, a rrect stays a rrect and is never mistaken for one) whenever the matrix keeps rects as rects,
// and nothing more — no direction or start index is tracked, so a mirror needs no bookkeeping beyond the flag.
func TestTransformShapeIdentity(t *testing.T) {
	var mirror geom.Matrix
	mirror.SetScale(-1, 1)

	oval := New().AddOval(geom.RectLTRB(0, 0, 20, 10), geom.DirectionCW)
	oval.Transform(&mirror)
	if bounds, ok := oval.IsOval(); !ok || bounds != geom.RectLTRB(-20, 0, 0, 10) {
		t.Errorf("mirrored oval = %v, %v", bounds, ok)
	}

	rrect := New().AddRRect(geom.MakeRRect(geom.RectLTRB(0, 0, 20, 10), 3, 3), geom.DirectionCCW)
	rrect.Transform(&mirror)
	if rrect.isa != isARRect {
		t.Errorf("mirrored rrect identity = %v, want isARRect", rrect.isa)
	}
	if _, ok := rrect.IsOval(); ok {
		t.Error("a rrect must never report as an oval")
	}

	// A 45-degree rotate is not rect-stays-rect, so the rrect identity drops too.
	var r geom.Matrix
	r.SetRotate(45)
	rrect.Transform(&r)
	if rrect.isa != isAGeneral {
		t.Errorf("rotated rrect identity = %v, want isAGeneral", rrect.isa)
	}

	// TransformTo carries the identity to the destination without disturbing the source.
	src := New().AddOval(geom.RectLTRB(0, 0, 20, 10), geom.DirectionCW)
	dst := New().MoveTo(3, 3).LineTo(4, 4) // non-empty, and not a simple shape
	src.TransformTo(&mirror, dst)
	if _, ok := dst.IsOval(); !ok {
		t.Error("TransformTo should carry the oval identity to dst")
	}
	if _, ok := src.IsOval(); !ok {
		t.Error("TransformTo should leave the source's oval identity alone")
	}
}

func TestTransformConvexityDirectionFlip(t *testing.T) {
	p := New().AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW)
	if p.GetConvexity() != ConvexityConvexCW {
		t.Fatalf("precondition: rect is convex CW")
	}
	var m geom.Matrix
	m.SetScale(-1, 1) // mirror flips the direction
	p.Transform(&m)
	if got := p.convexity; got != ConvexityConvexCCW {
		t.Errorf("mirrored convexity = %v, want CCW", got)
	}
}

func TestTransformPerspective(t *testing.T) {
	p := New()
	p.MoveTo(10, 10).LineTo(100, 10).LineTo(100, 100).Close()

	var m geom.Matrix
	m.SetAll(1, 0, 0, 0, 1, 0, 0.001, 0.002, 1)
	dst := New()
	p.TransformTo(&m, dst)

	// Verbs: the close line gets materialized before the close verb.
	checkVerbs(t, dst, VerbMove, VerbLine, VerbLine, VerbLine, VerbClose)

	// Points must match the full perspective mapping of the source points.
	srcPts := pts(10, 10, 100, 10, 100, 100, 10, 10)
	for i, sp := range srcPts {
		w := 0.001*sp.X + 0.002*sp.Y + 1
		want := geom.Pt(sp.X/w, sp.Y/w)
		got := dst.points[i]
		if !geom.ScalarNearlyEqual(got.X, want.X) || !geom.ScalarNearlyEqual(got.Y, want.Y) {
			t.Errorf("persp point[%d] = %v, want %v", i, got, want)
		}
	}

	// A quad becomes a conic with the TransformW weight.
	q := New()
	q.MoveTo(0, 0).QuadTo(50, 50, 100, 0)
	qdst := New()
	q.TransformTo(&m, qdst)
	checkVerbs(t, qdst, VerbMove, VerbConic)
	wantW := geom.TransformW(pts(0, 0, 50, 50, 100, 0), 1, &m)
	if got := qdst.conicWeights[0]; got != wantW {
		t.Errorf("perspective conic weight = %v, want %v", got, wantW)
	}

	// A cubic is subdivided into four cubics.
	cu := New()
	cu.MoveTo(0, 0).CubicTo(10, 10, 20, 10, 30, 0)
	cdst := New()
	cu.TransformTo(&m, cdst)
	checkVerbs(t, cdst, VerbMove, VerbCubic, VerbCubic, VerbCubic, VerbCubic)
}

func TestTransformIdentity(t *testing.T) {
	p := New()
	p.MoveTo(1, 1).LineTo(2, 2)
	id := geom.IdentityMatrix()
	dst := New()
	p.TransformTo(&id, dst)
	if !Equal(p, dst) {
		t.Error("identity transform should copy")
	}
}

func TestArcToOval(t *testing.T) {
	// Quarter arc on a circle: all points on the circle radius.
	oval := geom.RectLTRB(0, 0, 100, 100)
	p := New()
	p.ArcToOval(oval, 0, 90, true)
	if p.IsEmpty() {
		t.Fatal("arc should produce segments")
	}
	// First point at angle 0 = (100, 50); last at angle 90 = (50, 100).
	if p.points[0] != geom.Pt(100, 50) {
		t.Errorf("arc start = %v", p.points[0])
	}
	last, _ := p.LastPt()
	if !geom.ScalarNearlyEqual(last.X, 50) || !geom.ScalarNearlyEqual(last.Y, 100) {
		t.Errorf("arc end = %v", last)
	}

	// Zero sweep at angle 0 is a lone point (moveTo or lineTo).
	p = New()
	p.MoveTo(0, 0)
	p.ArcToOval(oval, 0, 0, false)
	checkVerbs(t, p, VerbMove, VerbLine)
	if pt, _ := p.LastPt(); pt != geom.Pt(100, 50) {
		t.Errorf("lone point = %v", pt)
	}

	// Negative-dimension oval is a no-op.
	p = New()
	p.ArcToOval(geom.RectLTRB(10, 10, 0, 20), 0, 90, true)
	if !p.IsEmpty() {
		t.Error("negative oval should be a no-op")
	}
}

func TestArcToTangent(t *testing.T) {
	// Right-angle corner rounded with radius 10: line then conic ending on the tangent points.
	p := New()
	p.MoveTo(0, 0).ArcToTangent(100, 0, 100, 100, 10)
	checkVerbs(t, p, VerbMove, VerbLine, VerbConic)
	// Tangent points: (90, 0) and (100, 10).
	if got := p.points[1]; !geom.ScalarNearlyEqual(got.X, 90) || !geom.ScalarNearlyEqual(got.Y, 0) {
		t.Errorf("tangent line end = %v", got)
	}
	if got, _ := p.LastPt(); !geom.ScalarNearlyEqual(got.X, 100) || !geom.ScalarNearlyEqual(got.Y, 10) {
		t.Errorf("arc end = %v", got)
	}
	// 90-degree corner arc has weight cos(45deg).
	if w := p.conicWeights[0]; !geom.ScalarNearlyEqual(w, float32(math.Sqrt2/2)) {
		t.Errorf("tangent arc weight = %v", w)
	}

	// Zero radius reduces to a line to (x1, y1).
	p = New()
	p.MoveTo(0, 0).ArcToTangent(50, 0, 50, 50, 0)
	checkVerbs(t, p, VerbMove, VerbLine)

	// Collinear tangents reduce to a line to (x1, y1).
	p = New()
	p.MoveTo(0, 0).ArcToTangent(50, 0, 100, 0, 10)
	checkVerbs(t, p, VerbMove, VerbLine)
	if pt, _ := p.LastPt(); pt != geom.Pt(50, 0) {
		t.Errorf("collinear arcTo end = %v", pt)
	}
}

func TestArcToRotated(t *testing.T) {
	// Half circle from (0,0) to (100,0) with radius 50.
	p := New()
	p.MoveTo(0, 0).ArcToRotated(50, 50, 0, ArcSizeSmall, geom.DirectionCW, 100, 0)
	// End point is exact (SetLastPt patches it).
	if pt, _ := p.LastPt(); pt != geom.Pt(100, 0) {
		t.Errorf("svg arc end = %v", pt)
	}
	// DirectionCW is SVG sweep-flag 1: the arc from (0,0) to (100,0) bulges through negative y.
	b := p.Bounds()
	if !geom.ScalarNearlyEqualTolerance(b.Top, -50, 0.5) || b.Bottom > 0.5 {
		t.Errorf("svg arc bounds = %v", b)
	}

	// Zero radius is a line.
	p = New()
	p.MoveTo(0, 0).ArcToRotated(0, 50, 0, ArcSizeSmall, geom.DirectionCW, 100, 0)
	checkVerbs(t, p, VerbMove, VerbLine)

	// Identical endpoints are a line.
	p = New()
	p.MoveTo(10, 10).ArcToRotated(50, 50, 0, ArcSizeSmall, geom.DirectionCW, 10, 10)
	checkVerbs(t, p, VerbMove, VerbLine)

	// Radii too small get scaled up: arc from (0,0) to (100,0) with rx=ry=10 still reaches the end.
	p = New()
	p.MoveTo(0, 0).ArcToRotated(10, 10, 0, ArcSizeSmall, geom.DirectionCW, 100, 0)
	if pt, _ := p.LastPt(); pt != geom.Pt(100, 0) {
		t.Errorf("scaled-radii arc end = %v", pt)
	}
}

func TestAddArc(t *testing.T) {
	oval := geom.RectLTRB(0, 0, 100, 100)
	// Full sweep starting at a multiple of 90 becomes an oval.
	p := New().AddArc(oval, 0, 360)
	if _, ok := p.IsOval(); !ok {
		t.Error("360-degree arc should be an oval")
	}
	// Zero sweep is a no-op.
	p = New().AddArc(oval, 45, 0)
	if !p.IsEmpty() {
		t.Error("zero sweep should be a no-op")
	}
}

func TestIterForceClose(t *testing.T) {
	p := New()
	p.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).Close()

	iter := NewIter(p, false)
	var verbs []Verb
	for {
		verb, _, ok := iter.Next()
		if !ok {
			break
		}
		verbs = append(verbs, verb)
	}
	// Explicit close materializes the closing line: M, L, L, L(close line), Close.
	want := []Verb{VerbMove, VerbLine, VerbLine, VerbLine, VerbClose}
	if len(verbs) != len(want) {
		t.Fatalf("iter verbs = %v, want %v", verbs, want)
	}
	for i := range want {
		if verbs[i] != want[i] {
			t.Fatalf("iter verbs = %v, want %v", verbs, want)
		}
	}

	// forceClose on an unclosed contour appends the closing line and close.
	p = New()
	p.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10)
	iter = NewIter(p, true)
	verbs = verbs[:0]
	var lastLine [4]geom.Point
	for {
		verb, ptsArr, ok := iter.Next()
		if !ok {
			break
		}
		if verb == VerbLine {
			lastLine = ptsArr
		}
		verbs = append(verbs, verb)
	}
	if len(verbs) != 5 || verbs[3] != VerbLine || verbs[4] != VerbClose {
		t.Fatalf("force-close verbs = %v", verbs)
	}
	if lastLine[0] != geom.Pt(10, 10) || lastLine[1] != geom.Pt(0, 0) {
		t.Errorf("closing line = %v -> %v", lastLine[0], lastLine[1])
	}
}
