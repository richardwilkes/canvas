// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for the OpsTask opChains backing free list (opchainspool.go). Like the other pool tests (oppool_test.go,
// programinfopool_test.go) these need no GL context: they drive the borrow/recycle protocol directly and assert its
// safety invariant — recycle clears the backing (dropping op pointers and applied-clip coverage FPs so the pool pins
// nothing) while preserving the array's capacity for the next frame's task. The render-level cross-frame guard, that a
// reused backing never leaks a stale opChain into a later frame, is TestOpPoolCrossFrameStability in
// oppool_live_test.go, which drives the pool automatically because each frame's OpsTask borrows and (at onDelete)
// recycles its backing.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// TestRecycleOpChainsBoxClearsAndPreservesCapacity verifies recycleOpChainsBox (a) drops every reference the []opChain
// backing held — here an applied clip's coverage fragment processor, the pointer-carrying member that the inline-clip
// storage put into opChain — so the pool pins nothing, while (b) preserving the array (same backing, capacity kept,
// length reset to 0) so the next frame's task reuses it instead of re-growing from nil.
func TestRecycleOpChainsBoxClearsAndPreservesCapacity(t *testing.T) {
	box := borrowOpChainsBox()
	// Grow the backing past a single element onto the heap, as a multi-primitive frame does.
	box.chains = append(box.chains, opChain{}, opChain{}, opChain{})
	// Put a live reference into the backing (an applied clip carrying a coverage FP) so the clear is observable.
	box.chains[1].hasAppliedClip = true
	box.chains[1].appliedClip.coverageFP = MakeColorFP(colorcore.PMColor4f{A: 1})
	grownCap := cap(box.chains)
	if grownCap <= 1 {
		t.Fatalf("expected the backing to grow onto the heap; cap=%d", grownCap)
	}
	backing0 := &box.chains[:grownCap][0]

	recycleOpChainsBox(box)

	if len(box.chains) != 0 {
		t.Fatalf("backing length not reset to 0: %d", len(box.chains))
	}
	if cap(box.chains) != grownCap {
		t.Fatalf("backing capacity not preserved: got %d want %d", cap(box.chains), grownCap)
	}
	if &box.chains[:grownCap][0] != backing0 {
		t.Fatal("backing array was reallocated, not preserved")
	}
	for i, c := range box.chains[:grownCap] {
		if c.hasAppliedClip || c.appliedClip.coverageFP != nil {
			t.Fatalf("recycled backing still pins an applied-clip coverage FP at index %d", i)
		}
	}
}

// TestBorrowOpChainsBoxIsClean verifies a borrowed box always presents a length-0 backing, whether it is a fresh box or
// a reused one carrying a preserved (grown) backing, so NewOpsTask starts opChains from a clean array in both cases.
func TestBorrowOpChainsBoxIsClean(t *testing.T) {
	// Prime the pool with a recycled, grown box so a subsequent borrow can take the reuse path.
	box := borrowOpChainsBox()
	box.chains = append(box.chains, opChain{}, opChain{})
	recycleOpChainsBox(box)

	got := borrowOpChainsBox()
	if len(got.chains) != 0 {
		t.Fatalf("borrowed box backing not length 0: %d", len(got.chains))
	}
}
