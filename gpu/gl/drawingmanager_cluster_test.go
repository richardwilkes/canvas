// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Regression tests for taskClusterVisit's multi/zero-target barrier handling. A task with anything other than exactly
// one target is treated as a full barrier for every one of its targets: the clustering pass must forget the last-seen
// cluster tail for all of them so no later same-target task can be reordered across the barrier. The barrier loop must
// therefore delete lastTaskMap[base.Target(j)] for each j, not lastTaskMap[base.Target(0)] repeatedly, which would
// leave the second and subsequent targets' cluster tails stale. These tests build the DAG directly with tiny stub
// tasks, so they need no GL context.

package gl

import "testing"

// newStubWithTargets builds a non-blocking stub render task whose targets are exactly the given proxies. It sets the
// targets field directly (rather than via addTarget) so the test needs no DrawingManager, ref-counting, or GL context.
func newStubWithTargets(targets ...*SurfaceProxy) *stubRenderTask {
	t := newStubRenderTask(false)
	t.targets = append(t.targets, targets...)
	return t
}

// llistOrder returns the tasks in a taskLList in head-to-tail order.
func llistOrder(l *taskLList) []RenderTask {
	var out []RenderTask
	for t := l.head; t != nil; t = t.taskBase().llNext {
		out = append(out, t)
	}
	return out
}

// TestTaskClusterVisitMultiTargetClearsAllTargets verifies that visiting a task with more than one target clears the
// cluster tail recorded for every one of its targets, not just the first. Before the fix the barrier loop deleted
// base.Target(0) on each iteration, so the second target's stale cluster tail survived, which would let a later
// same-target task cluster across the barrier.
func TestTaskClusterVisitMultiTargetClearsAllTargets(t *testing.T) {
	a := &SurfaceProxy{}
	b := &SurfaceProxy{}

	// Prime the cluster map so both a and b have a recorded cluster tail.
	ta := newStubWithTargets(a)
	tb := newStubWithTargets(b)
	lastTaskMap := map[*SurfaceProxy]RenderTask{a: ta, b: tb}

	barrier := newStubWithTargets(a, b) // two targets => full barrier for both a and b
	var llist taskLList
	if reordered := taskClusterVisit(barrier, &llist, lastTaskMap); reordered {
		t.Errorf("multi-target barrier must not report reordering, got true")
	}
	if _, ok := lastTaskMap[a]; ok {
		t.Errorf("first target's cluster tail was not cleared by the barrier")
	}
	if _, ok := lastTaskMap[b]; ok {
		t.Errorf("second target's cluster tail was not cleared by the barrier (Target(0) was deleted instead of Target(j))")
	}
}

// TestTaskClusterVisitZeroTargetIsSafe verifies the zero-target branch is well-behaved: the loop body never runs, so
// the map is untouched and no reordering is reported.
func TestTaskClusterVisitZeroTargetIsSafe(t *testing.T) {
	a := &SurfaceProxy{}
	lastTaskMap := map[*SurfaceProxy]RenderTask{a: newStubWithTargets(a)}
	barrier := newStubWithTargets() // zero targets
	var llist taskLList
	if reordered := taskClusterVisit(barrier, &llist, lastTaskMap); reordered {
		t.Errorf("zero-target barrier must not report reordering, got true")
	}
	if _, ok := lastTaskMap[a]; !ok {
		t.Errorf("zero-target barrier must not touch unrelated cluster tails")
	}
}

// TestClusterRenderTasksStillClustersWhenSafe guards against over-correction: a middle task whose target differs from a
// surrounding same-target pair must still be pulled aside so the pair clusters. This exercises the normal single-target
// path and must keep working after the barrier-loop fix.
func TestClusterRenderTasksStillClustersWhenSafe(t *testing.T) {
	a := &SurfaceProxy{}
	c := &SurfaceProxy{}
	t1 := newStubWithTargets(a)
	t2 := newStubWithTargets(c)
	t3 := newStubWithTargets(a)

	var llist taskLList
	if !clusterRenderTasks([]RenderTask{t1, t2, t3}, &llist) {
		t.Fatalf("expected clustering to reorder the differing middle task")
	}
	got := llistOrder(&llist)
	want := []RenderTask{t2, t1, t3}
	if len(got) != len(want) {
		t.Fatalf("cluster produced %d tasks, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("cluster order[%d] = %p, want %p", i, got[i], want[i])
		}
	}
}
