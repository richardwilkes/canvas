// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Guard for the OpsTask free list (opstaskpool.go), in the style of the other pool guards: a recycled task must
// come back indistinguishable from a zero-valued one on every live field — stale cross-frame state (bounds, clip
// generation, flags, dependency links) is exactly the class of bug whole-object pooling can introduce — while keeping
// only the documented carry-overs (slice capacities and the self-capturing visitor closures).

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func TestOpsTaskPoolRecycleResets(t *testing.T) {
	task := borrowOpsTask()
	// Dirty every category of state a frame can leave behind: base bookkeeping, dependency links, recording state, and
	// the cached visitors.
	task.initRenderTask(task)
	task.setFlag(taskFlagClosed | taskFlagDisowned)
	task.totalBounds = geom.Rect{Left: 1, Top: 2, Right: 3, Bottom: 4}
	task.clippedContentBounds = geom.IRect{Left: 5, Top: 6, Right: 7, Bottom: 8}
	task.lastClipStackGenID = 42
	task.usesMSAASurface = true
	task.dependencies = append(task.dependencies, task)
	task.dependents = append(task.dependents, task)
	task.targets = append(task.targets, nil)
	task.sampledProxies = append(task.sampledProxies, nil)
	_ = task.sampledDepVisitor(nil, nil)
	sampledFn := &task.sampledDepFn
	oldID := task.uniqueID
	clear(task.targets)
	task.targets = task.targets[:0] // Unref's contract before recycle.

	recycleOpsTask(task)
	// recycleOpsTask resets the task before handing it to the pool, so the reset is asserted on the pointer directly —
	// sync.Pool gives no guarantee about which shell a later borrow returns (earlier tests, -count reruns, and GCs all
	// perturb the per-P caches). Nothing else runs between the recycle and these reads, so the inspection cannot race a
	// reuse.
	got := task
	if got.flags != 0 || got.refCnt != 0 || got.totalBounds != (geom.Rect{}) ||
		got.clippedContentBounds != (geom.IRect{}) || got.lastClipStackGenID != 0 ||
		got.usesMSAASurface || got.arenas != nil || got.opChainsBox != nil ||
		got.visitDrawingMgr != nil || got.visitCaps != nil || got.visitAlloc != nil {
		t.Fatalf("recycled task carried stale state: %+v", got)
	}
	if len(got.dependencies) != 0 || len(got.dependents) != 0 || len(got.targets) != 0 ||
		len(got.sampledProxies) != 0 {
		t.Fatal("recycled task's slices were not truncated")
	}
	if cap(got.dependencies) == 0 || cap(got.targets) == 0 || cap(got.sampledProxies) == 0 {
		t.Fatal("recycled task lost its slice capacities")
	}
	if got.sampledDepFn == nil || &got.sampledDepFn != sampledFn {
		t.Fatal("recycled task lost its cached visitor closure")
	}
	// A reused task must get a fresh identity so stale task-keyed lookups can never match: borrow a shell (not
	// necessarily the one above) and run the constructor's init.
	reused := borrowOpsTask()
	reused.initRenderTask(reused)
	if reused.uniqueID == oldID {
		t.Fatal("reused task kept its old uniqueID")
	}
	reused.setFlag(taskFlagDisowned | taskFlagClosed)
	reused.Unref() // Return it to the pool for whoever runs next.
}
