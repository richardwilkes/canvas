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

func TestIsNestedFillRects(t *testing.T) {
	// Two nested rects (what stroking a rect produces).
	p := New()
	p.AddRect(geom.RectLTRB(0, 0, 100, 80), geom.DirectionCW)
	p.AddRect(geom.RectLTRB(10, 10, 90, 70), geom.DirectionCCW)
	var rects [2]geom.Rect
	if !p.IsNestedFillRects(&rects) {
		t.Fatal("nested rects not detected")
	}
	if rects[0] != geom.RectLTRB(0, 0, 100, 80) || rects[1] != geom.RectLTRB(10, 10, 90, 70) {
		t.Errorf("rects = %+v", rects)
	}

	// Reversed order still reports outer first.
	p = New()
	p.AddRect(geom.RectLTRB(10, 10, 90, 70), geom.DirectionCW)
	p.AddRect(geom.RectLTRB(0, 0, 100, 80), geom.DirectionCW)
	if !p.IsNestedFillRects(&rects) {
		t.Fatal("reversed nested rects not detected")
	}
	if rects[0] != geom.RectLTRB(0, 0, 100, 80) {
		t.Errorf("outer-first ordering violated: %+v", rects)
	}

	// A single rect is not a nested pair.
	p = New()
	p.AddRect(geom.RectLTRB(0, 0, 100, 80), geom.DirectionCW)
	if p.IsNestedFillRects(&rects) {
		t.Error("single rect reported as nested")
	}

	// Overlapping (non-nested) rects fail.
	p = New()
	p.AddRect(geom.RectLTRB(0, 0, 60, 60), geom.DirectionCW)
	p.AddRect(geom.RectLTRB(30, 30, 100, 100), geom.DirectionCW)
	if p.IsNestedFillRects(&rects) {
		t.Error("overlapping rects reported as nested")
	}

	// A rect plus a triangle fails.
	p = New()
	p.AddRect(geom.RectLTRB(0, 0, 100, 80), geom.DirectionCW)
	p.MoveTo(20, 20)
	p.LineTo(40, 20)
	p.LineTo(30, 40)
	p.Close()
	if p.IsNestedFillRects(&rects) {
		t.Error("rect+triangle reported as nested")
	}
}

func TestIsRect(t *testing.T) {
	rect := geom.RectLTRB(10, 20, 110, 80)

	// AddRect in both directions
	for _, dir := range []geom.PathDirection{geom.DirectionCW, geom.DirectionCCW} {
		p := New()
		p.AddRect(rect, dir)
		got, ok := p.IsRect()
		if !ok || got != rect {
			t.Fatalf("AddRect dir %v: got %v ok %v", dir, got, ok)
		}
	}

	// Hand-built closed rect, and one with a redundant colinear midpoint
	p := New()
	p.MoveTo(10, 20)
	p.LineTo(60, 20)
	p.LineTo(110, 20)
	p.LineTo(110, 80)
	p.LineTo(10, 80)
	p.Close()
	if got, ok := p.IsRect(); !ok || got != rect {
		t.Fatalf("colinear rect: got %v ok %v", got, ok)
	}

	// Unclosed rect whose final implicit close edge is axis-aligned still counts
	p = New()
	p.MoveTo(10, 20)
	p.LineTo(110, 20)
	p.LineTo(110, 80)
	p.LineTo(10, 80)
	if got, ok := p.IsRect(); !ok || got != rect {
		t.Fatalf("open rect: got %v ok %v", got, ok)
	}

	// Non-rects
	for name, build := range map[string]func() *Path{
		"diagonal": func() *Path {
			q := New()
			q.MoveTo(0, 0)
			q.LineTo(10, 10)
			q.LineTo(0, 20)
			q.Close()
			return q
		},
		"curve": func() *Path {
			q := New()
			q.AddRoundRect(geom.RectLTRB(0, 0, 20, 20), 4, 4, geom.DirectionCW)
			return q
		},
		"two rects": func() *Path {
			q := New()
			q.AddRect(rect, geom.DirectionCW)
			q.AddRect(geom.RectLTRB(0, 0, 5, 5), geom.DirectionCW)
			return q
		},
		"too few points": func() *Path {
			q := New()
			q.MoveTo(0, 0)
			q.LineTo(10, 0)
			return q
		},
		"zigzag": func() *Path {
			q := New()
			q.MoveTo(0, 0)
			q.LineTo(10, 0)
			q.LineTo(10, 10)
			q.LineTo(20, 10)
			q.LineTo(20, 20)
			q.LineTo(0, 20)
			q.Close()
			return q
		},
	} {
		if got, ok := build().IsRect(); ok {
			t.Fatalf("%s: unexpectedly a rect: %v", name, got)
		}
	}
}
