// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pdf

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

func TestClipStackWideOpen(t *testing.T) {
	cs := NewClipStack()
	if !cs.isWideOpen() {
		t.Error("a fresh clip stack should be wide open")
	}
	if cs.getTopmostGenID() != wideOpenGenID {
		t.Errorf("wide-open genID = %d, want %d", cs.getTopmostGenID(), wideOpenGenID)
	}
	if cs.isEmpty(geom.IRectWH(100, 100)) {
		t.Error("a wide-open clip is not empty over a non-empty device")
	}
	if b := cs.bounds(geom.IRectWH(100, 100)); b != geom.RectLTRB(0, 0, 100, 100) {
		t.Errorf("wide-open bounds = %v, want the whole device", b)
	}
}

func TestClipStackRectIntersect(t *testing.T) {
	cs := NewClipStack()
	m := geom.IdentityMatrix()
	cs.ClipRect(geom.RectLTRB(10, 10, 50, 50), &m, raster.ClipIntersect, false)

	if cs.isWideOpen() {
		t.Error("after a rect clip the stack is not wide open")
	}
	if got := cs.bounds(geom.IRectWH(100, 100)); got != geom.RectLTRB(10, 10, 50, 50) {
		t.Errorf("bounds after clip = %v, want {10,10,50,50}", got)
	}
	_, boundType, isRects := cs.getBounds()
	if boundType != normalBounds || !isRects {
		t.Errorf("expected a normal intersection-of-rects bound, got type=%d isRects=%v", boundType, isRects)
	}
	// A second intersecting rect stays an intersection-of-rects.
	cs.ClipRect(geom.RectLTRB(0, 0, 30, 30), &m, raster.ClipIntersect, false)
	if got := cs.bounds(geom.IRectWH(100, 100)); got != geom.RectLTRB(10, 10, 30, 30) {
		t.Errorf("bounds after second clip = %v, want {10,10,30,30}", got)
	}
}

func TestClipStackEmptyByDisjointIntersect(t *testing.T) {
	cs := NewClipStack()
	m := geom.IdentityMatrix()
	cs.ClipRect(geom.RectLTRB(10, 10, 50, 50), &m, raster.ClipIntersect, false)
	// Intersecting with a disjoint rect leaves nothing writeable (the rect-rect in-place merge detects it).
	cs.ClipRect(geom.RectLTRB(60, 60, 90, 90), &m, raster.ClipIntersect, false)
	if !cs.isEmpty(geom.IRectWH(100, 100)) {
		t.Error("intersect of disjoint rects should be empty")
	}
	// A difference-of-superset is NOT reported empty: the conservative difference bound keeps the prior rect
	// (combineBoundsDiff intentionally does not try to detect this cancellation).
	cs2 := NewClipStack()
	cs2.ClipRect(geom.RectLTRB(10, 10, 50, 50), &m, raster.ClipIntersect, false)
	cs2.ClipRect(geom.RectLTRB(0, 0, 100, 100), &m, raster.ClipDifference, false)
	if cs2.isEmpty(geom.IRectWH(100, 100)) {
		t.Error("difference-of-superset should stay non-empty (conservative bound)")
	}
}

func TestClipStackSaveRestore(t *testing.T) {
	cs := NewClipStack()
	m := geom.IdentityMatrix()
	cs.Save()
	cs.ClipRect(geom.RectLTRB(10, 10, 50, 50), &m, raster.ClipIntersect, false)
	if cs.isWideOpen() {
		t.Error("stack should be clipped after clipRect")
	}
	cs.Restore()
	if !cs.isWideOpen() {
		t.Error("restore should drop the clip added since save")
	}
}

func TestClipStackGenIDChanges(t *testing.T) {
	cs := NewClipStack()
	m := geom.IdentityMatrix()
	before := cs.getTopmostGenID()
	cs.ClipRect(geom.RectLTRB(10, 10, 50, 50), &m, raster.ClipIntersect, false)
	after := cs.getTopmostGenID()
	if after == before {
		t.Error("gen ID should change after adding a clip")
	}
	if after < 3 {
		t.Errorf("a real clip's gen ID should be >= 3, got %d", after)
	}
}

func TestClipStackQuickContains(t *testing.T) {
	cs := NewClipStack()
	m := geom.IdentityMatrix()
	cs.ClipRect(geom.RectLTRB(10, 10, 50, 50), &m, raster.ClipIntersect, false)
	if !cs.quickContains(geom.RectLTRB(20, 20, 40, 40)) {
		t.Error("a rect fully inside the clip should be quick-contained")
	}
	if cs.quickContains(geom.RectLTRB(0, 0, 100, 100)) {
		t.Error("a rect larger than the clip must not be quick-contained")
	}
}

func TestClipStackPathElementBounds(t *testing.T) {
	cs := NewClipStack()
	m := geom.IdentityMatrix()
	tri := (&path.Path{}).MoveTo(10, 10).LineTo(90, 20).LineTo(40, 80)
	tri.Close()
	cs.ClipPath(tri, &m, raster.ClipIntersect, true)
	if cs.isWideOpen() {
		t.Error("a path clip is not wide open")
	}
	// The finite bound is the path's control-point bounds.
	if got := cs.bounds(geom.IRectWH(100, 100)); got != geom.RectLTRB(10, 10, 90, 80) {
		t.Errorf("path clip bounds = %v, want the triangle's bounds", got)
	}
	if !cs.isAnyAA() {
		t.Error("an AA path clip should report an AA element")
	}
}

// TestClipStackReplaceRestoresToFrame verifies that a replace op discards prior clips in the frame.
func TestClipStackReplaceRestoresToFrame(t *testing.T) {
	cs := NewClipStack()
	m := geom.IdentityMatrix()
	cs.ClipRect(geom.RectLTRB(10, 10, 20, 20), &m, raster.ClipIntersect, false)
	cs.ReplaceClip(geom.RectLTRB(30, 30, 90, 90), false)
	if got := cs.bounds(geom.IRectWH(100, 100)); got != geom.RectLTRB(30, 30, 90, 90) {
		t.Errorf("after replace, bounds = %v, want the replacement rect", got)
	}
}
