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

func collectEdges(p *Path) []EdgeIterResult {
	var out []EdgeIterResult
	it := NewEdgeIter(p)
	for {
		e, ok := it.Next()
		if !ok {
			return out
		}
		cp := e
		cp.Pts = append([]geom.Point(nil), e.Pts...)
		out = append(out, cp)
	}
}

func TestEdgeIterAutoClosesOpenContour(t *testing.T) {
	p := New()
	p.MoveTo(0, 0)
	p.LineTo(10, 0)
	p.LineTo(10, 10)
	edges := collectEdges(p)
	if len(edges) != 3 {
		t.Fatalf("got %d edges, want 3 (2 lines + close line)", len(edges))
	}
	closeLine := edges[2]
	if closeLine.Verb != VerbLine ||
		closeLine.Pts[0] != (geom.Point{X: 10, Y: 10}) ||
		closeLine.Pts[1] != (geom.Point{X: 0, Y: 0}) {
		t.Errorf("close line = %+v", closeLine)
	}
	if !edges[0].IsNewContour || edges[1].IsNewContour || edges[2].IsNewContour {
		t.Error("IsNewContour flags wrong")
	}
}

func TestEdgeIterMultipleContoursAndConics(t *testing.T) {
	p := New()
	p.MoveTo(0, 0)
	p.ConicTo(5, 0, 5, 5, 0.7)
	p.Close()
	p.MoveTo(20, 20)
	p.QuadTo(25, 20, 25, 25)
	edges := collectEdges(p)
	// contour 1: conic + close line; contour 2: quad + close line
	if len(edges) != 4 {
		t.Fatalf("got %d edges, want 4", len(edges))
	}
	if edges[0].Verb != VerbConic {
		t.Errorf("edge 0 verb = %v", edges[0].Verb)
	}
	if edges[1].Verb != VerbLine || edges[1].Pts[1] != (geom.Point{X: 0, Y: 0}) {
		t.Errorf("edge 1 = %+v", edges[1])
	}
	if !edges[2].IsNewContour || edges[2].Verb != VerbQuad {
		t.Errorf("edge 2 = %+v", edges[2])
	}
	if edges[3].Verb != VerbLine || edges[3].Pts[1] != (geom.Point{X: 20, Y: 20}) {
		t.Errorf("edge 3 = %+v", edges[3])
	}
}

func TestEdgeIterConicWeight(t *testing.T) {
	p := New()
	p.MoveTo(0, 0)
	p.ConicTo(5, 0, 5, 5, 0.75)
	it := NewEdgeIter(p)
	e, ok := it.Next()
	if !ok || e.Verb != VerbConic {
		t.Fatalf("expected conic, got %+v ok=%v", e, ok)
	}
	if it.ConicWeight() != 0.75 {
		t.Errorf("weight = %v", it.ConicWeight())
	}
}

func TestEdgeIterEmptyAndMoveOnly(t *testing.T) {
	p := New()
	if edges := collectEdges(p); len(edges) != 0 {
		t.Errorf("empty path: %d edges", len(edges))
	}
	p.MoveTo(1, 1)
	p.MoveTo(2, 2)
	if edges := collectEdges(p); len(edges) != 0 {
		t.Errorf("move-only path: %d edges", len(edges))
	}
}

func TestMapRectPerspectiveStraddling(t *testing.T) {
	// A perspective matrix whose w plane cuts through the rect: the faithful mapRect clips rather than projecting the
	// negative-w corners.
	var m geom.Matrix
	m.SetAll(
		1, 0, 0,
		0, 1, 0,
		0, -0.01, 0.05,
	) // w = 1 at y=... w crosses zero within y ∈ [0, 100]
	src := geom.RectLTRB(0, 0, 10, 100)
	got, stays := MapRect(&m, src)
	if stays {
		t.Error("perspective mapRect must report stays == false")
	}
	if got.IsEmpty() {
		t.Error("clipped mapRect should not be empty")
	}
	if !got.IsFinite() {
		t.Errorf("clipped mapRect must be finite, got %v", got)
	}
	// Non-perspective delegates to geom.
	var id geom.Matrix
	id.SetAll(2, 0, 1, 0, 2, 1, 0, 0, 1)
	gotAffine, _ := MapRect(&id, geom.RectLTRB(0, 0, 1, 1))
	if gotAffine != geom.RectLTRB(1, 1, 3, 3) {
		t.Errorf("affine MapRect = %v", gotAffine)
	}
}

func TestPerspectiveClipStraddlingProducesFinitePath(t *testing.T) {
	p := New()
	p.MoveTo(0, 0)
	p.LineTo(10, 0)
	p.LineTo(10, 100)
	p.LineTo(0, 100)
	p.Close()
	var m geom.Matrix
	m.SetAll(
		1, 0, 0,
		0, 1, 0,
		0, -0.01, 0.05,
	)
	dst := New()
	p.TransformTo(&m, dst)
	if !dst.IsFinite() {
		t.Errorf("transformed straddling path must be finite (clipped), bounds %v", dst.Bounds())
	}
	if dst.IsEmpty() {
		t.Error("straddling path should not be fully clipped out")
	}
	// Every surviving point must be on the visible side (w >= ~1/16384 pre-projection means bounded projected
	// coordinates; just check boundedness).
	b := dst.Bounds()
	const big = 1e7
	if b.Left < -big || b.Right > big || b.Top < -big || b.Bottom > big {
		t.Errorf("clipped path bounds unexpectedly large: %v", b)
	}
}
