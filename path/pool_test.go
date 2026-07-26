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

// TestRecycleRestoresFreshState covers Borrow's "same state as a freshly-constructed one" contract. The volatile flag
// is the one Rewind deliberately leaves alone (it describes the object, not its contents), so Recycle has to clear it:
// callers such as the GPU styled-shape path mark a borrowed path volatile, and that must not leak into the next
// borrower's unrelated geometry.
func TestRecycleRestoresFreshState(t *testing.T) {
	p := Borrow()
	p.AddOval(geom.RectLTRB(0, 0, 10, 20), geom.DirectionCW)
	p.SetFillType(FillInverseEvenOdd)
	p.SetVolatile(true)
	p.GetConvexity()
	p.GenerationID()
	p.Bounds()
	Recycle(p)

	// Compare the recycled object's fields directly rather than through the accessors, which would prime the lazy
	// caches of a path that now belongs to the pool.
	fresh := &Path{}
	if len(p.verbs) != 0 || len(p.points) != 0 || len(p.conicWeights) != 0 {
		t.Errorf("recycled path is not empty: %d verbs, %d points, %d weights",
			len(p.verbs), len(p.points), len(p.conicWeights))
	}
	if p.isVolatile != fresh.isVolatile {
		t.Errorf("volatile = %v, want %v", p.isVolatile, fresh.isVolatile)
	}
	if p.fillType != fresh.fillType {
		t.Errorf("fill type = %v, want %v", p.fillType, fresh.fillType)
	}
	if p.convexity != fresh.convexity {
		t.Errorf("convexity = %v, want %v", p.convexity, fresh.convexity)
	}
	if p.isa != fresh.isa {
		t.Errorf("shape identity = %v, want %v", p.isa, fresh.isa)
	}
	if p.segmentMask != fresh.segmentMask {
		t.Errorf("segment mask = %v, want %v", p.segmentMask, fresh.segmentMask)
	}
	if p.boundsValid != fresh.boundsValid {
		t.Errorf("boundsValid = %v, want %v", p.boundsValid, fresh.boundsValid)
	}
	if p.genID != fresh.genID {
		t.Errorf("generation ID = %v, want %v", p.genID, fresh.genID)
	}
	if p.lastMoveIdx != fresh.lastMoveIdx {
		t.Errorf("lastMoveIdx = %v, want %v", p.lastMoveIdx, fresh.lastMoveIdx)
	}
}

// TestBorrowNeverVolatile is the end-to-end form of the same contract: after a volatile path has been recycled, no
// path the pool hands out reports itself volatile.
func TestBorrowNeverVolatile(t *testing.T) {
	p := Borrow()
	p.MoveTo(0, 0).LineTo(4, 4).SetVolatile(true)
	Recycle(p)

	// sync.Pool is free to return a brand-new Path, so hold each borrowed path and keep pulling until the recycled
	// one comes back around.
	held := make([]*Path, 0, 64)
	sawRecycled := false
	for range cap(held) {
		q := Borrow()
		held = append(held, q)
		if q.IsVolatile() {
			t.Fatal("Borrow handed out a path still flagged volatile")
		}
		if q == p {
			sawRecycled = true
			break
		}
	}
	for _, q := range held {
		Recycle(q)
	}
	if !sawRecycled {
		t.Log("the pool never handed the recycled path back; only fresh paths were checked")
	}
}
