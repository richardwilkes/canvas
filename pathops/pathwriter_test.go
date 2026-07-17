// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Focused unit tests for pathWriter (pathwriter.go). The line-contour lanes (deferredMove/deferredLine/
// finishContour/close/moveTo/lineTo/isClosed) are covered end-to-end by the bridge trace tests in
// opsegment_walk_test.go; here the curve lanes (quadTo/conicTo/cubicTo via update) and the matchedLast / changedSlopes
// predicates are exercised directly with synthetic pt-t nodes.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// selfPtT builds a standalone opPtT with a self-referential loop (contains(anything else) is false).
func selfPtT(x, y float32) *opPtT {
	p := &opPtT{pt: geom.Point{X: x, Y: y}}
	p.next = p
	return p
}

// TestPathWriterMatchedLast checks matchedLast's three cases.
func TestPathWriterMatchedLast(t *testing.T) {
	w := newPathWriter(path.FillWinding)
	a := selfPtT(1, 2)
	// with no deferred line, a non-nil node does not match; nil does (nil == defers[1], both nil), so isClosed on a
	// still-empty writer is true.
	if w.matchedLast(a) {
		t.Fatal("matchedLast(non-nil) should be false before any deferred line")
	}
	if !w.matchedLast(nil) {
		t.Fatal("matchedLast(nil) should be true when the deferred line is also nil (nil == nil)")
	}
	w.defers[1] = a
	if !w.matchedLast(a) {
		t.Fatal("matchedLast should be true for the deferred-line node itself")
	}
	if w.matchedLast(nil) {
		t.Fatal("matchedLast(nil) should be false")
	}
	// a node whose loop contains defers[1] also matches
	b := &opPtT{pt: geom.Point{X: 3, Y: 4}}
	b.next = a
	a.next = b // a<->b share a loop; b.contains(a) is true
	w.defers[1] = a
	if !w.matchedLast(b) {
		t.Fatal("matchedLast should be true when the test node's loop contains the deferred line")
	}
}

// TestPathWriterChangedSlopes checks the collinear (false) and bent (true) cases.
func TestPathWriterChangedSlopes(t *testing.T) {
	w := newPathWriter(path.FillWinding)
	w.defers[0] = selfPtT(0, 0)
	w.defers[1] = selfPtT(1, 0)
	if w.changedSlopes(selfPtT(2, 0)) {
		t.Fatal("collinear points should not report a slope change")
	}
	if !w.changedSlopes(selfPtT(1, 1)) {
		t.Fatal("a bend should report a slope change")
	}
	// when the last emitted line already matched, no slope change is reported
	w2 := newPathWriter(path.FillWinding)
	shared := selfPtT(5, 5)
	w2.defers[0] = shared
	w2.defers[1] = shared
	if w2.changedSlopes(selfPtT(9, 9)) {
		t.Fatal("changedSlopes should short-circuit false when the last point matched")
	}
}

// TestPathWriterCurves emits a closed contour with a quad and a cubic and checks the resulting verbs/points.
func TestPathWriterCurves(t *testing.T) {
	w := newPathWriter(path.FillEvenOdd)
	a := selfPtT(0, 0)
	b := selfPtT(10, 0)
	// aEnd shares a's pt-t loop so the contour closes back onto the move point
	aEnd := &opPtT{pt: geom.Point{X: 0, Y: 0}}
	aEnd.next = a
	a.next = aEnd
	ctrl := geom.Point{X: 5, Y: -5}
	c1 := geom.Point{X: 12, Y: 5}
	c2 := geom.Point{X: 8, Y: 8}

	w.deferredMove(a)
	w.quadTo(ctrl, b)
	w.cubicTo(c1, c2, aEnd)
	w.finishContour()

	out := w.nativePath()
	if out.FillType() != path.FillEvenOdd {
		t.Fatalf("fill type = %v, want even-odd", out.FillType())
	}
	var verbs []path.Verb
	var moveP, quadC, quadE, cub1, cub2, cubE geom.Point
	it := path.NewRawIter(out)
	for {
		verb, pts, _, ok := it.Next()
		if !ok {
			break
		}
		verbs = append(verbs, verb)
		switch verb {
		case path.VerbMove:
			moveP = pts[0]
		case path.VerbQuad:
			quadC, quadE = pts[1], pts[2]
		case path.VerbCubic:
			cub1, cub2, cubE = pts[1], pts[2], pts[3]
		}
	}
	wantVerbs := []path.Verb{path.VerbMove, path.VerbQuad, path.VerbCubic, path.VerbClose}
	if len(verbs) != len(wantVerbs) {
		t.Fatalf("verbs = %v, want %v", verbs, wantVerbs)
	}
	for i, v := range wantVerbs {
		if verbs[i] != v {
			t.Fatalf("verb[%d] = %v, want %v (all: %v)", i, verbs[i], v, verbs)
		}
	}
	if moveP != (geom.Point{X: 0, Y: 0}) {
		t.Fatalf("moveTo = %v, want (0,0)", moveP)
	}
	if quadC != ctrl || quadE != (geom.Point{X: 10, Y: 0}) {
		t.Fatalf("quadTo = ctrl %v end %v, want %v (10,0)", quadC, quadE, ctrl)
	}
	if cub1 != c1 || cub2 != c2 {
		t.Fatalf("cubicTo control = %v,%v want %v,%v", cub1, cub2, c1, c2)
	}
	// the cubic's end is the shared move point, so update() should have snapped it back to (0,0)
	if cubE != (geom.Point{X: 0, Y: 0}) {
		t.Fatalf("cubicTo end = %v, want (0,0) (snapped to the first point)", cubE)
	}
}

// TestPathWriterConic emits a closed conic-then-line contour and checks the weight is carried through.
func TestPathWriterConic(t *testing.T) {
	w := newPathWriter(path.FillWinding)
	a := selfPtT(0, 0)
	b := selfPtT(10, 10)
	// aEnd shares a's pt-t loop so the closing line lands back on the move point.
	aEnd := &opPtT{pt: geom.Point{X: 0, Y: 0}}
	aEnd.next = a
	a.next = aEnd
	w.deferredMove(a)
	w.conicTo(geom.Point{X: 10, Y: 0}, b, 0.5)
	w.deferredLine(aEnd)
	w.finishContour()
	out := w.nativePath()
	sawConic := false
	it := path.NewRawIter(out)
	for {
		verb, _, weight, ok := it.Next()
		if !ok {
			break
		}
		if verb == path.VerbConic {
			sawConic = true
			if weight != 0.5 {
				t.Fatalf("conic weight = %v, want 0.5", weight)
			}
		}
	}
	if !sawConic {
		t.Fatal("expected a conic verb in the output")
	}
}
