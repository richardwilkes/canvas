// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Regression tests for insertTaskBeforeLast's reorder-blocker bookkeeping. reorderBlockerTaskIndices holds indices into
// dm.dag identifying the tasks that block reordering; sortTasks relies on those indices to partition the DAG into
// topo-sorted spans (the blocker task sits on the boundary between spans). When insertTaskBeforeLast swaps the new task
// in ahead of the current last task, and that last task is itself a reorder blocker, its recorded index must follow it
// from len(dag)-1 to len(dag). The predicate that bumps the index therefore has to compare against len(dm.dag)-1;
// comparing against len(dm.dag) (as upstream Skia's GrDrawingManager::insertTaskBeforeLast does) can never fire because
// a recorded index is always a valid task index in [0, len(dag)-1], leaving the index stale and mis-partitioning the
// spans. These tests construct the DAG directly with tiny stub tasks, so they need no GL context.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// stubRenderTask is a minimal concrete RenderTask used only to exercise the drawing-manager DAG bookkeeping. Its
// blocker flag is set at construction so BlocksReordering reflects the intent.
type stubRenderTask struct {
	RenderTaskBase
}

func newStubRenderTask(blocksReordering bool) *stubRenderTask {
	t := &stubRenderTask{}
	t.initRenderTask(t)
	if blocksReordering {
		t.setFlag(taskFlagBlocksReordering)
	}
	return t
}

func (t *stubRenderTask) Name() string { return "Stub" }
func (t *stubRenderTask) OnMakeClosed(*geom.IRect) ExpectedOutcome {
	return ExpectedOutcomeTargetUnchanged
}
func (t *stubRenderTask) OnIsUsed(*SurfaceProxy) bool             { return false }
func (t *stubRenderTask) GatherProxyIntervals(*ResourceAllocator) {}
func (t *stubRenderTask) OnExecute(*OpFlushState) bool            { return true }

// blockerInvariantHolds reports whether every recorded blocker index points at a task that actually blocks reordering,
// which is the invariant sortTasks depends on to place span boundaries correctly.
func blockerInvariantHolds(dm *DrawingManager) bool {
	for _, idx := range dm.reorderBlockerTaskIndices {
		if idx < 0 || idx >= len(dm.dag) || !dm.dag[idx].taskBase().BlocksReordering() {
			return false
		}
	}
	return true
}

// TestInsertTaskBeforeLastBumpsTrailingBlockerIndex verifies that inserting a task ahead of a trailing reorder blocker
// advances the blocker's recorded index so it keeps pointing at the blocker (not at the freshly inserted task).
func TestInsertTaskBeforeLastBumpsTrailingBlockerIndex(t *testing.T) {
	dm := &DrawingManager{}
	a := newStubRenderTask(false)
	blocker := newStubRenderTask(true)
	dm.appendTask(a)       // dag=[a]
	dm.appendTask(blocker) // dag=[a, blocker], reorderBlockerTaskIndices=[1]

	inserted := newStubRenderTask(false)
	dm.insertTaskBeforeLast(inserted) // want dag=[a, inserted, blocker], reorderBlockerTaskIndices=[2]

	if got := len(dm.dag); got != 3 {
		t.Fatalf("dag length = %d, want 3", got)
	}
	if dm.dag[2] != RenderTask(blocker) {
		t.Errorf("blocker not at the tail after insert; dag[2] is %p, blocker is %p", dm.dag[2], blocker)
	}
	if dm.dag[1] != RenderTask(inserted) {
		t.Errorf("inserted task not placed before the blocker; dag[1] is %p, inserted is %p", dm.dag[1], inserted)
	}
	if n := len(dm.reorderBlockerTaskIndices); n != 1 || dm.reorderBlockerTaskIndices[0] != 2 {
		t.Errorf("reorderBlockerTaskIndices = %v, want [2]", dm.reorderBlockerTaskIndices)
	}
	if !blockerInvariantHolds(dm) {
		t.Errorf("blocker index no longer points at a reorder-blocking task: %v", dm.reorderBlockerTaskIndices)
	}
	// sortTasks partitions on the blocker index; a stale index would place the boundary on the wrong task. It must at
	// least not panic here, and the resulting spans must respect the (corrected) boundary.
	dm.sortTasks()
	if a.index != 0 || inserted.index != 1 {
		t.Errorf("span before the blocker sorted wrong: a.index=%d inserted.index=%d, want 0 and 1", a.index,
			inserted.index)
	}
}

// TestInsertTaskBeforeLastLeavesNonTrailingBlockerIndex verifies the predicate does not over-fire: when the current
// last task is not a reorder blocker, an earlier blocker's index must stay put.
func TestInsertTaskBeforeLastLeavesNonTrailingBlockerIndex(t *testing.T) {
	dm := &DrawingManager{}
	blocker := newStubRenderTask(true)
	last := newStubRenderTask(false)
	dm.appendTask(blocker) // dag=[blocker], reorderBlockerTaskIndices=[0]
	dm.appendTask(last)    // dag=[blocker, last]

	inserted := newStubRenderTask(false)
	dm.insertTaskBeforeLast(inserted) // want dag=[blocker, inserted, last], reorderBlockerTaskIndices=[0]

	if dm.dag[0] != RenderTask(blocker) {
		t.Errorf("blocker moved unexpectedly; dag[0] is %p, blocker is %p", dm.dag[0], blocker)
	}
	if n := len(dm.reorderBlockerTaskIndices); n != 1 || dm.reorderBlockerTaskIndices[0] != 0 {
		t.Errorf("reorderBlockerTaskIndices = %v, want [0]", dm.reorderBlockerTaskIndices)
	}
	if !blockerInvariantHolds(dm) {
		t.Errorf("blocker index no longer points at a reorder-blocking task: %v", dm.reorderBlockerTaskIndices)
	}
}
