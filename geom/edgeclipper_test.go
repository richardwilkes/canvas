// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import "testing"

// collectClipped drains the clipper into (verb, points) records.
type clippedSeg struct {
	pts  []Point
	verb ClipVerb
}

func drainClipper(e *EdgeClipper) []clippedSeg {
	var out []clippedSeg
	var pts [4]Point
	for {
		verb, ok := e.Next(pts[:])
		if !ok {
			return out
		}
		n := 0
		switch verb {
		case ClipVerbLine:
			n = 2
		case ClipVerbQuad:
			n = 3
		case ClipVerbCubic:
			n = 4
		}
		out = append(out, clippedSeg{verb: verb, pts: append([]Point(nil), pts[:n]...)})
	}
}

func TestClipLineInsideUnchanged(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	if !e.ClipLine(Point{X: 10, Y: 10}, Point{X: 90, Y: 90}, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) != 1 || segs[0].verb != ClipVerbLine {
		t.Fatalf("segs = %+v", segs)
	}
	if segs[0].pts[0] != (Point{X: 10, Y: 10}) || segs[0].pts[1] != (Point{X: 90, Y: 90}) {
		t.Errorf("pts = %v", segs[0].pts)
	}
}

func TestClipLineAboveRejected(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	if e.ClipLine(Point{X: 10, Y: -50}, Point{X: 90, Y: -1}, clip) {
		t.Fatal("expected no output for a line above the clip")
	}
}

func TestClipLineLeftBecomesVertical(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	if !e.ClipLine(Point{X: -50, Y: 10}, Point{X: -10, Y: 90}, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) != 1 {
		t.Fatalf("segs = %+v", segs)
	}
	p := segs[0].pts
	if p[0].X != 0 || p[1].X != 0 || p[0].Y != 10 || p[1].Y != 90 {
		t.Errorf("expected vertical line on left edge, got %v", p)
	}
}

func TestClipLineRightCulled(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(true)
	if e.ClipLine(Point{X: 110, Y: 10}, Point{X: 150, Y: 90}, clip) {
		t.Fatal("expected culled output with canCullToTheRight")
	}
	e = NewEdgeClipper(false)
	if !e.ClipLine(Point{X: 110, Y: 10}, Point{X: 150, Y: 90}, clip) {
		t.Fatal("expected vertical line without culling")
	}
	segs := drainClipper(e)
	if len(segs) != 1 || segs[0].pts[0].X != 100 || segs[0].pts[1].X != 100 {
		t.Errorf("segs = %+v", segs)
	}
}

func TestClipLineCrossingProducesSegments(t *testing.T) {
	clip := RectLTRB(0, 0, 10, 10)
	e := NewEdgeClipper(false)
	// Enters on the left, exits on the right.
	if !e.ClipLine(Point{X: -10, Y: 2}, Point{X: 20, Y: 8}, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) != 3 {
		t.Fatalf("expected 3 segments (vline, interior, vline), got %+v", segs)
	}
	// Segments must chain continuously.
	for i := 1; i < len(segs); i++ {
		last := segs[i-1].pts[len(segs[i-1].pts)-1]
		if segs[i].pts[0] != last {
			t.Errorf("discontinuity between segment %d and %d: %v vs %v", i-1, i, last, segs[i].pts[0])
		}
	}
}

func TestClipQuadInsideUnchanged(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	src := []Point{{X: 10, Y: 10}, {X: 50, Y: 90}, {X: 90, Y: 10}}
	if !e.ClipQuad(src, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	// The quad is chopped at its Y extrema (apex), so expect quads whose join point chains and whose endpoints match
	// the original.
	if segs[0].pts[0] != src[0] {
		t.Errorf("start = %v", segs[0].pts[0])
	}
	lastSeg := segs[len(segs)-1]
	if lastSeg.pts[len(lastSeg.pts)-1] != src[2] {
		t.Errorf("end = %v", lastSeg.pts[len(lastSeg.pts)-1])
	}
	for _, s := range segs {
		if s.verb != ClipVerbQuad {
			t.Errorf("expected only quads, got %+v", s)
		}
	}
}

func TestClipQuadChoppedInY(t *testing.T) {
	clip := RectLTRB(0, 20, 100, 100)
	e := NewEdgeClipper(false)
	src := []Point{{X: 10, Y: 0}, {X: 50, Y: 60}, {X: 90, Y: 0}}
	if !e.ClipQuad(src, clip) {
		t.Fatal("expected output")
	}
	for _, s := range drainClipper(e) {
		for _, p := range s.pts {
			if p.Y < clip.Top-0.001 || p.Y > clip.Bottom+0.001 {
				t.Errorf("point %v outside clip Y range", p)
			}
		}
	}
}

func TestClipCubicChoppedInXAndY(t *testing.T) {
	clip := RectLTRB(10, 10, 60, 60)
	e := NewEdgeClipper(false)
	src := []Point{{X: 0, Y: 0}, {X: 100, Y: 20}, {X: -40, Y: 60}, {X: 70, Y: 80}}
	if !e.ClipCubic(src, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) == 0 {
		t.Fatal("no segments")
	}
	for _, s := range segs {
		for _, p := range s.pts {
			if p.Y < clip.Top-0.01 || p.Y > clip.Bottom+0.01 {
				t.Errorf("point %v outside clip Y range", p)
			}
			// X may touch the clip edges exactly (vertical closure lines) but not exceed them.
			if p.X < clip.Left-0.01 || p.X > clip.Right+0.01 {
				t.Errorf("point %v outside clip X range", p)
			}
		}
	}
}

func TestClipCubicHugeFallsBackToLine(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	src := []Point{{X: -1e7, Y: -1e7}, {X: 5e6, Y: 2e6}, {X: -5e6, Y: 8e6}, {X: 1e7, Y: 1e7}}
	if !e.ClipCubic(src, clip) {
		t.Fatal("expected output")
	}
	for _, s := range drainClipper(e) {
		if s.verb != ClipVerbLine {
			t.Errorf("expected line-only fallback for huge cubic, got verb %d", s.verb)
		}
	}
}

func TestIntersectLine(t *testing.T) {
	clip := RectLTRB(0, 0, 10, 10)
	src := [2]Point{{X: -5, Y: 5}, {X: 15, Y: 5}}
	var dst [2]Point
	if !IntersectLine(&src, clip, &dst) {
		t.Fatal("expected intersection")
	}
	if dst[0] != (Point{X: 0, Y: 5}) || dst[1] != (Point{X: 10, Y: 5}) {
		t.Errorf("dst = %v", dst)
	}
	miss := [2]Point{{X: -5, Y: 15}, {X: 15, Y: 20}}
	if IntersectLine(&miss, clip, &dst) {
		t.Error("expected miss")
	}
}
