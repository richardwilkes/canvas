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

// buildContours runs the edge builder over p and returns the resulting contour head.
func buildContours(t *testing.T, p *path.Path) *opContourHead {
	t.Helper()
	head := &opContourHead{}
	state := newOpGlobalState(head)
	b := newOpEdgeBuilder(p, head, state)
	if !b.finish() {
		t.Fatal("edge builder finish() returned false")
	}
	return head
}

// realContours returns the built contours that carry at least one segment (the head is never unlinked, so an empty path
// leaves it in the list with a zero count).
func realContours(head *opContourHead) []*opContour {
	var out []*opContour
	for c := &head.opContour; c != nil; c = c.next {
		if c.count > 0 {
			out = append(out, c)
		}
	}
	return out
}

// segVerbs returns the verbs of a contour's segments in forward order.
func segVerbs(c *opContour) []path.Verb {
	var out []path.Verb
	for s := c.first(); s != nil; s = s.next {
		out = append(out, s.verb)
	}
	return out
}

func verbsEqual(a []path.Verb, b ...path.Verb) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestEdgeBuilderTriangle(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).LineTo(60, 0).LineTo(30, 50).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbLine, path.VerbLine, path.VerbLine) {
		t.Fatalf("verbs = %v, want three lines", v)
	}
	// Verify the three lines walk the triangle and its closing edge.
	wantStart := []geom.Point{pt(0, 0), pt(60, 0), pt(30, 50)}
	wantEnd := []geom.Point{pt(60, 0), pt(30, 50), pt(0, 0)}
	i := 0
	for s := cs[0].first(); s != nil; s = s.next {
		if s.head.pt() != wantStart[i] || s.tail.pt() != wantEnd[i] {
			t.Fatalf("segment %d = %v->%v, want %v->%v", i, s.head.pt(), s.tail.pt(), wantStart[i], wantEnd[i])
		}
		i++
	}
}

func TestEdgeBuilderRect(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(0, 0, 20, 10), geom.DirectionCW)
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbLine, path.VerbLine, path.VerbLine, path.VerbLine) {
		t.Fatalf("verbs = %v, want four lines", v)
	}
}

func TestEdgeBuilderOpenContourAutoCloses(t *testing.T) {
	// An open contour (no explicit close) is closed by preFetch, materializing the closing line.
	p := path.New()
	p.MoveTo(0, 0).LineTo(60, 0).LineTo(30, 50)
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbLine, path.VerbLine, path.VerbLine) {
		t.Fatalf("verbs = %v, want three lines (incl. the closing edge)", v)
	}
	// The final segment closes back to the contour start.
	var last *opSegment
	for s := cs[0].first(); s != nil; s = s.next {
		last = s
	}
	if last.tail.pt() != pt(0, 0) {
		t.Fatalf("closing segment ends at %v, want (0,0)", last.tail.pt())
	}
}

func TestEdgeBuilderExplicitCloseToStart(t *testing.T) {
	// The contour's final lineTo returns to the start point, so the close is degenerate and adds no extra closing line:
	// the three explicit edges are the whole contour.
	p := path.New()
	p.MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).LineTo(0, 0).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbLine, path.VerbLine, path.VerbLine) {
		t.Fatalf("verbs = %v, want three lines (no synthesized closing edge)", v)
	}
}

func TestEdgeBuilderTwoContours(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).LineTo(60, 0).LineTo(30, 50).Close()
	p.MoveTo(100, 100).LineTo(160, 100).LineTo(130, 150).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 2 {
		t.Fatalf("contour count = %d, want 2", len(cs))
	}
	for i, c := range cs {
		if v := segVerbs(c); !verbsEqual(v, path.VerbLine, path.VerbLine, path.VerbLine) {
			t.Fatalf("contour %d verbs = %v, want three lines", i, v)
		}
	}
	// The second contour starts where its moveTo placed it.
	if cs[1].first().head.pt() != pt(100, 100) {
		t.Fatalf("second contour starts at %v, want (100,100)", cs[1].first().head.pt())
	}
}

func TestEdgeBuilderMonotonicQuad(t *testing.T) {
	// A monotonic quad whose control tangent does not reverse is added whole (no max-curvature split).
	p := path.New()
	p.MoveTo(0, 0).QuadTo(10, 0, 10, 10).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbQuad, path.VerbLine) {
		t.Fatalf("verbs = %v, want quad then closing line", v)
	}
	q := cs[0].first()
	if q.pts[0] != pt(0, 0) || q.pts[1] != pt(10, 0) || q.pts[2] != pt(10, 10) {
		t.Fatalf("quad control points = %v, want (0,0)(10,0)(10,10)", q.pts[:3])
	}
}

func TestEdgeBuilderMonotonicCubic(t *testing.T) {
	// A cubic monotonic in x and y is not complex-broken; it is added whole.
	p := path.New()
	p.MoveTo(0, 0).CubicTo(10, 5, 20, 25, 30, 30).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbCubic, path.VerbLine) {
		t.Fatalf("verbs = %v, want cubic then closing line", v)
	}
}

func TestEdgeBuilderMonotonicConic(t *testing.T) {
	p := path.New()
	p.MoveTo(0, 0).ConicTo(10, 0, 10, 10, 0.7).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbConic, path.VerbLine) {
		t.Fatalf("verbs = %v, want conic then closing line", v)
	}
	if w := cs[0].first().weight; w != 0.7 {
		t.Fatalf("conic weight = %v, want 0.7", w)
	}
}

func TestEdgeBuilderDegenerateLineSkipped(t *testing.T) {
	// The zero-length lineTo is dropped by preFetch; the triangle survives intact.
	p := path.New()
	p.MoveTo(0, 0).LineTo(0, 0).LineTo(60, 0).LineTo(30, 50).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbLine, path.VerbLine, path.VerbLine) {
		t.Fatalf("verbs = %v, want three lines (degenerate dropped)", v)
	}
}

func TestEdgeBuilderFillMask(t *testing.T) {
	makeTri := func(ft path.FillType) *opContour {
		p := path.New()
		p.SetFillType(ft)
		p.MoveTo(0, 0).LineTo(60, 0).LineTo(30, 50).Close()
		return &buildContours(t, p).opContour
	}
	if makeTri(path.FillEvenOdd).xor != true {
		t.Fatal("even-odd fill should set the contour xor flag")
	}
	if makeTri(path.FillInverseEvenOdd).xor != true {
		t.Fatal("inverse-even-odd fill should set the contour xor flag")
	}
	if makeTri(path.FillWinding).xor != false {
		t.Fatal("winding fill should clear the contour xor flag")
	}
	if makeTri(path.FillInverseWinding).xor != false {
		t.Fatal("inverse-winding fill should clear the contour xor flag")
	}
}

func TestEdgeBuilderEmptyPath(t *testing.T) {
	head := buildContours(t, path.New())
	if head.count != 0 {
		t.Fatalf("empty path head count = %d, want 0", head.count)
	}
	if len(realContours(head)) != 0 {
		t.Fatalf("empty path produced %d real contours, want 0", len(realContours(head)))
	}
}

func TestEdgeBuilderQuadSplit(t *testing.T) {
	// A quad whose control tangent reverses is chopped at max curvature into two quads.
	p := path.New()
	p.MoveTo(0, 0).QuadTo(10, 20, 20, 0).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbQuad, path.VerbQuad, path.VerbLine) {
		t.Fatalf("verbs = %v, want two quad halves + closing line", v)
	}
}

func TestEdgeBuilderConicSplit(t *testing.T) {
	// A conic whose control tangent reverses is chopped at max curvature into two conics.
	p := path.New()
	p.MoveTo(0, 0).ConicTo(10, 20, 20, 0, 0.5).Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	if v := segVerbs(cs[0]); !verbsEqual(v, path.VerbConic, path.VerbConic, path.VerbLine) {
		t.Fatalf("verbs = %v, want two conic halves + closing line", v)
	}
}

func TestEdgeBuilderTwoOperands(t *testing.T) {
	// addOperand appends a second path; walk flips fOperand at the first-operand boundary, so the second path's contour
	// is marked as the operand.
	pa := path.New()
	pa.MoveTo(0, 0).LineTo(10, 0).LineTo(5, 8).Close()
	pb := path.New()
	pb.MoveTo(100, 0).LineTo(110, 0).LineTo(105, 8).Close()
	head := &opContourHead{}
	state := newOpGlobalState(head)
	b := newOpEdgeBuilder(pa, head, state)
	b.addOperand(pb)
	if !b.finish() {
		t.Fatal("edge builder finish() returned false")
	}
	cs := realContours(head)
	if len(cs) != 2 {
		t.Fatalf("contour count = %d, want 2", len(cs))
	}
	if cs[0].operand != false {
		t.Fatal("first contour should not be the operand")
	}
	if cs[1].operand != true {
		t.Fatal("second contour should be the operand")
	}
}

func TestEdgeBuilderComplexCubicSplits(t *testing.T) {
	// A self-intersecting (loop) cubic must be split by complexBreak before it enters the model, so the closed contour
	// holds two cubic pieces plus the closing line rather than one whole cubic.
	p := path.New()
	p.MoveTo(635.625, 614.687)
	p.CubicTo(171.625, 236.188, 1064.62, 135.688, 516.625, 570.187)
	p.Close()
	head := buildContours(t, p)
	cs := realContours(head)
	if len(cs) != 1 {
		t.Fatalf("contour count = %d, want 1", len(cs))
	}
	v := segVerbs(cs[0])
	if len(v) != 3 {
		t.Fatalf("verbs = %v, want three segments (two cubic halves + closing line)", v)
	}
	if v[0] != path.VerbCubic || v[1] != path.VerbCubic || v[2] != path.VerbLine {
		t.Fatalf("verbs = %v, want [cubic cubic line]", v)
	}
}

func TestComplexBreak(t *testing.T) {
	// A cubic monotonic in x and y needs no split.
	mono := [4]geom.Point{pt(0, 0), pt(10, 5), pt(20, 25), pt(30, 30)}
	var mt [3]float32
	if n := complexBreak(mono, mt[:]); n != 0 {
		t.Fatalf("monotonic cubic breaks = %d, want 0", n)
	}
	// A loop cubic whose self-intersection parameter lies inside [0,1] is split once, near the crossing.
	loop := [4]geom.Point{pt(635.625, 614.687), pt(171.625, 236.188), pt(1064.62, 135.688), pt(516.625, 570.187)}
	var lt [3]float32
	n := complexBreak(loop, lt[:])
	if n != 1 {
		t.Fatalf("loop cubic breaks = %d, want 1", n)
	}
	if lt[0] <= 0 || lt[0] >= 1 {
		t.Fatalf("loop split t = %v, want in (0,1)", lt[0])
	}
	// A serpentine (two inflections) and a cusp both drive the non-loop branch through findInflections /
	// findMaxCurvature / calcPrecision, and each yields a split point in (0,1).
	serp := [4]geom.Point{pt(285.625, 499.687), pt(411.625, 808.188), pt(1064.62, 135.688), pt(1042.63, 585.187)}
	var stv [3]float32
	if ns := complexBreak(serp, stv[:]); ns != 1 || stv[0] <= 0 || stv[0] >= 1 {
		t.Fatalf("serpentine breaks = %d, t = %v, want 1 split in (0,1)", ns, stv[0])
	}
	cusp := [4]geom.Point{pt(0, 0), pt(1, 1), pt(1, 0), pt(0, 1)}
	var ctv [3]float32
	if nc := complexBreak(cusp, ctv[:]); nc != 1 || ctv[0] <= 0 || ctv[0] >= 1 {
		t.Fatalf("cusp breaks = %d, t = %v, want 1 split in (0,1)", nc, ctv[0])
	}
}
