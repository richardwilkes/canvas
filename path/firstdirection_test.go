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

func TestComputeFirstDirection(t *testing.T) {
	cw := (&Path{}).AddRect(geom.RectLTRB(10, 10, 50, 40), geom.DirectionCW)
	if got := cw.ComputeFirstDirectionRaw(); got != FirstDirectionCW {
		t.Fatalf("CW rect raw direction %v", got)
	}
	ccw := (&Path{}).AddRect(geom.RectLTRB(10, 10, 50, 40), geom.DirectionCCW)
	if got := ccw.ComputeFirstDirectionRaw(); got != FirstDirectionCCW {
		t.Fatalf("CCW rect raw direction %v", got)
	}

	// A hole ('o'): the outer contour holds the global y-max and decides.
	o := (&Path{}).AddRect(geom.RectLTRB(0, 0, 100, 100), geom.DirectionCW)
	o.AddRect(geom.RectLTRB(30, 30, 70, 70), geom.DirectionCCW)
	if got := o.ComputeFirstDirectionRaw(); got != FirstDirectionCW {
		t.Fatalf("outer-contour direction %v want CW", got)
	}

	// Degenerate: all points coincide.
	deg := (&Path{}).MoveTo(5, 5).LineTo(5, 5).LineTo(5, 5).Close()
	if got := deg.ComputeFirstDirectionRaw(); got != FirstDirectionUnknown {
		t.Fatalf("degenerate direction %v want Unknown", got)
	}

	// The cached form consults convexity first and agrees.
	if got := cw.ComputeFirstDirection(); got != FirstDirectionCW {
		t.Fatalf("cached CW direction %v", got)
	}
}

func TestReversePathTo(t *testing.T) {
	src := (&Path{}).MoveTo(0, 0).LineTo(10, 0).QuadTo(20, 5, 10, 10).ConicTo(5, 15, 0, 10, 0.7)

	dst := (&Path{}).MoveTo(0, 10) // caller positions at src's last point
	dst.ReversePathTo(src)
	pt, ok := dst.LastPt()
	if !ok || pt != (geom.Point{}) {
		t.Fatalf("reverse should end at src's first point, got %v", pt)
	}
	// verbs after the move: conic, quad, line — src's segments reversed.
	wantVerbs := []Verb{VerbMove, VerbConic, VerbQuad, VerbLine}
	if dst.CountVerbs() != len(wantVerbs) {
		t.Fatalf("verb count %d want %d", dst.CountVerbs(), len(wantVerbs))
	}
	it := NewRawIter(dst)
	for i, want := range wantVerbs {
		verb, _, _, ok2 := it.Next()
		if !ok2 || verb != want {
			t.Fatalf("verb %d = %v want %v", i, verb, want)
		}
	}

	// Multi-contour: only the last contour is reversed.
	multi := (&Path{}).MoveTo(100, 100).LineTo(200, 100).MoveTo(0, 0).LineTo(10, 0)
	d2 := (&Path{}).MoveTo(10, 0)
	d2.ReversePathTo(multi)
	if pt2, _ := d2.LastPt(); pt2 != (geom.Point{}) {
		t.Fatalf("multi-contour reverse should stop at the last contour's start, got %v", pt2)
	}
	if d2.CountVerbs() != 2 { // move + one line
		t.Fatalf("multi-contour reverse verb count %d want 2", d2.CountVerbs())
	}
}

func TestIsRectWithDirection(t *testing.T) {
	closed := (&Path{}).AddRect(geom.RectLTRB(10, 20, 60, 50), geom.DirectionCCW)
	rect, isClosed, dir, ok := closed.IsRectWithDirection()
	if !ok || !isClosed || dir != geom.DirectionCCW || rect != geom.RectLTRB(10, 20, 60, 50) {
		t.Fatalf("closed rect: %v %v %v %v", rect, isClosed, dir, ok)
	}
	open := (&Path{}).MoveTo(10, 20).LineTo(60, 20).LineTo(60, 50).LineTo(10, 50)
	if _, isClosed2, _, ok2 := open.IsRectWithDirection(); ok2 && isClosed2 {
		t.Fatal("open rect contour must not report closed")
	}
	tri := (&Path{}).MoveTo(0, 0).LineTo(10, 0).LineTo(5, 8).Close()
	if _, _, _, ok3 := tri.IsRectWithDirection(); ok3 {
		t.Fatal("triangle is not a rect")
	}
}
