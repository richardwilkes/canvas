// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GL-free guards for the persistent per-flush scratch objects. The DrawingManager now caches and reuses one
// OpFlushState (and its buffer pools) and one ResourceAllocator across flushes instead of heap-allocating them per
// flush; correct reuse depends on each being returned to its fresh-construction state before the next flush. These
// tests pin the two reset invariants that make that reuse safe. The render-level guard that a reused object cannot
// bleed stale state into a later frame is TestOpPoolCrossFrameStability (GL-live), which drives the persistent objects
// across many flushes on one context.

package gl

import "testing"

// TestResourceAllocatorBeginFlushClearsFailedInstantiation pins the one field that end-of-flush Reset intentionally
// does not clear (it is latched for the in-flush reorder retry): beginFlush must clear it so a persistent allocator's
// starting state matches a freshly constructed one's after a flush that hit an instantiation failure.
func TestResourceAllocatorBeginFlushClearsFailedInstantiation(t *testing.T) {
	// beginFlush touches only failedInstantiation, so a nil context is fine here.
	a := NewResourceAllocator(nil)

	// Reset deliberately leaves the latch set (the reorder retry reads it after Reset).
	a.failedInstantiation = true
	a.Reset()
	if !a.failedInstantiation {
		t.Fatal("Reset must not clear failedInstantiation — the in-flush reorder retry relies on it")
	}

	// beginFlush is what makes the reused allocator pristine for the next flush.
	a.beginFlush()
	if a.failedInstantiation {
		t.Fatal("beginFlush must clear failedInstantiation so a reused allocator starts clean")
	}
}

// TestOpFlushStateResetClearsPerFlushFields pins that Reset fully clears the per-op scratch fields the prepare/execute
// loops set, so no stale pointer survives into the next flush that reuses the persistent flush state. With no blocks
// allocated, Reset does not touch the (nil) gpu, so this runs without a GL context.
func TestOpFlushStateResetClearsPerFlushFields(t *testing.T) {
	s := NewOpFlushState(nil, nil, nil, nil, nil, nil, nil)

	// Simulate the residue an executing chain would leave if the normal clear-as-you-go flow were ever skipped; Reset
	// must not let any of it reach the next flush.
	s.opArgsSet = true
	s.opArgs.usesMSAASurface = true
	s.opArgs.colorLoadOp = 1
	pass := &OpsRenderPass{}
	s.opsRenderPass = pass
	arr := []*SurfaceProxy{nil}
	s.sampledProxyArray = &arr

	s.Reset()

	if s.opArgsSet {
		t.Error("Reset must clear opArgsSet")
	}
	if s.opArgs != (OpArgs{}) {
		t.Errorf("Reset must zero opArgs, got %+v", s.opArgs)
	}
	if s.opsRenderPass != nil {
		t.Error("Reset must clear opsRenderPass")
	}
	if s.sampledProxyArray != nil {
		t.Error("Reset must clear sampledProxyArray")
	}
}
