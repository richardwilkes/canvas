// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Regression tests for the op-recording allocation reductions: the batchable geometry ops bootstrap their per-instance
// slice from an inline backing array (rectsArr/rrectsArr/instancesArr/ circlesArr/ellipsesArr), so a freshly
// constructed single-instance op does not heap-allocate a separate backing array — the single instance lives in the
// inline array until combining grows it past capacity one. These tests assert the invariant structurally (the slice's
// first element aliases the inline array) rather than via allocation counts, so they hold under the race detector too.
// They construct ops directly, needing no GL context. The remaining oval-family ops (ellipseOp, diEllipseOp,
// ellipticalRRectOp) share the identical one-line pattern and are covered functionally by the live render suite.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// drainPool empties a package-level op pool so the constructor under test is guaranteed a fresh (pool-miss) shell.
// The aliasing assertions below only hold for such a shell: a shell recycled by recycleKeepingBacking deliberately
// carries a preserved heap backing that bootstrapInstances prefers over the inline array — that reuse is the
// optimization, not a bug — and earlier tests in the process populate these pools with exactly such shells. Without
// the drain, these tests pass or fail based on test order and GC timing (sync.Pool gives no guarantee about which
// shell a borrow returns; see the same discipline note in opstaskpool_test.go). Tests in this package are serial, so
// nothing refills the pool between the drain and the borrow.
func drainPool[T any](p *opPool[T]) {
	for p.pool.Get() != nil {
	}
}

func TestCircularRRectOpUsesInlineInstanceArray(t *testing.T) {
	drainPool(&circularRRectOpPool)
	identity := geom.IdentityMatrix()
	o := newCircularRRectOp(NewPaint(), &identity, geom.Rect{Left: 5, Top: 5, Right: 45, Bottom: 45},
		8, 0, false)
	if len(o.rrects) != 1 {
		t.Fatalf("expected exactly one instance, got %d", len(o.rrects))
	}
	if &o.rrects[0] != &o.rrectsArr[0] {
		t.Fatal("rrects did not bootstrap from the inline rrectsArr — a separate backing array was allocated")
	}
}

func TestAAStrokeRectOpUsesInlineInstanceArray(t *testing.T) {
	drainPool(&aaStrokeRectOpPool)
	identity := geom.IdentityMatrix()
	op := NewAAStrokeRectOpFromDevRects(NewPaint(), &identity,
		geom.Rect{Left: 0, Top: 0, Right: 20, Bottom: 20},
		geom.Rect{Left: 2, Top: 2, Right: 18, Bottom: 18}, geom.Point{X: 1, Y: 1})
	if op == nil {
		t.Fatal("NewAAStrokeRectOpFromDevRects returned nil")
	}
	o := op.(*aaStrokeRectOp)
	if len(o.rects) != 1 {
		t.Fatalf("expected exactly one instance, got %d", len(o.rects))
	}
	if &o.rects[0] != &o.rectsArr[0] {
		t.Fatal("rects did not bootstrap from the inline rectsArr — a separate backing array was allocated")
	}
}

func TestFillRRectOpUsesInlineInstanceArray(t *testing.T) {
	drainPool(&fillRRectOpPool)
	identity := geom.IdentityMatrix()
	rrect := geom.MakeRRect(geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 40}, 5, 5)
	o := newFillRRectOp(nil, colorcore.PMColor4f{R: 1, A: 1}, &identity, rrect,
		LocalCoordsFromRect(rrect.Rect), 0)
	if len(o.instances) != 1 {
		t.Fatalf("expected exactly one instance, got %d", len(o.instances))
	}
	if &o.instances[0] != &o.instancesArr[0] {
		t.Fatal("instances did not bootstrap from the inline instancesArr — a separate backing " +
			"array was allocated")
	}
}
