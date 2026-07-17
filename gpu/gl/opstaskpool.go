// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Free list for whole OpsTask objects. A fresh OpsTask is created for each surface every frame — the previous one is
// closed at flush and replaced — so the steady-state frame paid one OpsTask struct, its RenderTaskBase
// targets/dependency slice growths, and one heap closure per cached proxy visitor, every frame. Pooling the whole task
// eliminates that per-frame cost, and makes the task's cached self-capturing visitor closures
// (sampledDepFn/depFn/gatherFn) truly once-per-pooled-object instead of once-per-frame.
//
// Lifetime safety (same argument as opchainspool.go): a task is recycled only at the end of RenderTaskBase.Unref when
// refCnt reaches 0 — the death point at which, by this codebase's refcounting discipline, no owner may touch the task
// again. The task is provably disowned (Unref panics otherwise) and its ops/opChains backing were already recycled by
// the onDelete hook. recycleOpsTask rebuilds the task from the zero value (so any future field is reset by
// construction, never stale), preserving only: the capacity of the element-cleared
// targets/dependencies/dependents/sampledProxies/deferredProxies backings, and the three cached closures — which
// capture nothing but the (stable) task pointer itself and read their per-call state from fields that the zero-value
// reset just cleared. A fresh uniqueID is assigned on every borrow (initRenderTask), so anything stale keyed by task
// identity can never match a reused task.

package gl

import "sync"

var opsTaskPool = sync.Pool{New: func() any { return &OpsTask{} }}

// borrowOpsTask returns a reset OpsTask from the free list (or a zero-valued new one). NewOpsTask is the only caller
// and (re)initializes every live field.
func borrowOpsTask() *OpsTask { return opsTaskPool.Get().(*OpsTask) }

// recycleOpsTask resets t to its zero value — keeping the slice capacities and cached closures per the file comment —
// and returns it to the free list. Must only be called from the refCnt==0 path of RenderTaskBase.Unref.
func recycleOpsTask(t *OpsTask) {
	base := &t.RenderTaskBase
	targets := base.targets[:0] // Unref already dropped the target refs and cleared the elements.
	clear(base.dependencies)
	dependencies := base.dependencies[:0]
	clear(base.dependents)
	dependents := base.dependents[:0]
	clear(t.sampledProxies)
	sampledProxies := t.sampledProxies[:0]
	clear(t.deferredProxies)
	deferredProxies := t.deferredProxies[:0]
	sampledDepFn, depFn, gatherFn := t.sampledDepFn, t.depFn, t.gatherFn

	*t = OpsTask{}
	base.targets = targets
	base.dependencies = dependencies
	base.dependents = dependents
	t.sampledProxies = sampledProxies
	t.deferredProxies = deferredProxies
	t.sampledDepFn = sampledDepFn
	t.depFn = depFn
	t.gatherFn = gatherFn
	opsTaskPool.Put(t)
}
