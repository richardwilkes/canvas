// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GPU sync-object and submission machinery: inserting/testing/deleting fences (both the ARB sync-object and NV-fence
// lanes), the finished-callback queue, flush deferral, submission, and the OOM-aware error helpers. Timer queries are
// not implemented (they have no public entry point); finished callbacks are plain funcs.

package gl

// Fence wraps a backend fence: either a GLsync object or an NV fence id, per the caps fence type.
type Fence struct {
	sync    Sync
	nvFence uint32
}

// IsValid reports whether a fence was actually created.
func (s Fence) IsValid() bool { return s.sync != 0 || s.nvFence != 0 }

// InsertSync inserts a new GPU fence and returns it. Returns an invalid Fence when fences are unsupported.
func (g *Gpu) InsertSync() Fence {
	var sync Fence
	switch g.glCaps().FenceType() {
	case FenceNone:
		return sync
	case FenceNVFence:
		var fence uint32
		g.fns().GenFences(1, &fence)
		g.fns().SetFence(fence, ALL_COMPLETED)
		sync.nvFence = fence
	case FenceSyncObject:
		sync.sync = g.fns().FenceSync(SYNC_GPU_COMMANDS_COMPLETE, 0)
	}
	g.setNeedsFlush()
	return sync
}

// TestSync reports whether sync has signaled yet (non-blocking).
func (g *Gpu) TestSync(sync Fence) bool {
	switch g.glCaps().FenceType() {
	case FenceNone:
		panic("testing sync without sync support")
	case FenceNVFence:
		return g.fns().TestFence(sync.nvFence)
	case FenceSyncObject:
		result := g.fns().ClientWaitSync(sync.sync, 0, 0)
		return result == CONDITION_SATISFIED || result == ALREADY_SIGNALED
	}
	panic("unreachable fence type")
}

// DeleteSync releases the resources backing sync.
func (g *Gpu) DeleteSync(sync Fence) {
	switch g.glCaps().FenceType() {
	case FenceNone:
		panic("deleting sync without sync support")
	case FenceNVFence:
		g.fns().DeleteFences(1, &sync.nvFence)
	case FenceSyncObject:
		g.fns().DeleteSync(sync.sync)
	}
}

// finishCallback pairs a registered callback with the fence that must signal before it fires.
type finishCallback struct {
	callback func()
	sync     Fence
}

// AddFinishedCallback registers callback to fire once all work submitted so far completes on the GPU.
func (g *Gpu) AddFinishedCallback(callback func()) {
	if callback == nil {
		panic("nil finished callback")
	}
	g.finishCallbacks = append(g.finishCallbacks, finishCallback{callback: callback, sync: g.InsertSync()})
}

// CheckFinishedCallbacks calls the callbacks whose fences have signaled. Non-blocking; bails at the first unfinished
// sync since they signal in insertion order.
func (g *Gpu) CheckFinishedCallbacks() {
	for len(g.finishCallbacks) > 0 {
		fc := g.finishCallbacks[0]
		if fc.sync.IsValid() && !g.TestSync(fc.sync) {
			break
		}
		// Remove the entry before invoking it: the callback may re-enter (e.g. by calling FlushAndSubmit(sync)) and
		// must not see itself pending.
		g.finishCallbacks = g.finishCallbacks[1:]
		if fc.sync.IsValid() {
			g.DeleteSync(fc.sync)
		}
		fc.callback()
	}
}

// callAllFinishedCallbacks drains the callback list unconditionally; blocking is the caller's business. Fences are
// deleted when doDelete (i.e. the context is still valid).
func (g *Gpu) callAllFinishedCallbacks(doDelete bool) {
	for len(g.finishCallbacks) > 0 {
		fc := g.finishCallbacks[0]
		g.finishCallbacks = g.finishCallbacks[1:]
		if doDelete && fc.sync.IsValid() {
			g.DeleteSync(fc.sync)
		}
		fc.callback()
	}
}

func (g *Gpu) setNeedsFlush() { g.needsGLFlush = true }

// Flush calls glFlush if required by previous operations (or forced).
func (g *Gpu) Flush(force bool) {
	if g.needsGLFlush || force {
		g.fns().Flush()
		g.needsGLFlush = false
	}
}

// FinishOutstandingGpuWork blocks until all previously submitted GPU work completes.
func (g *Gpu) FinishOutstandingGpuWork() {
	g.fns().Finish()
}

// SubmitToGpu submits pending work to the GPU. With syncCPU it blocks until all GPU work is complete.
func (g *Gpu) SubmitToGpu(syncCPU bool) bool {
	if syncCPU || (len(g.finishCallbacks) > 0 && !g.glCaps().FenceSyncSupport()) {
		g.FinishOutstandingGpuWork()
		g.callAllFinishedCallbacks(true)
	} else {
		g.Flush(false)
		// See if any previously inserted finish procs are good to go.
		g.CheckFinishedCallbacks()
	}
	if !g.glCaps().SkipErrorChecks() {
		g.clearErrorsAndCheckForOOM()
	}
	return true
}

// clearErrorsAndCheckForOOM drains the GL error queue, recording whether an out-of-memory error occurred.
func (g *Gpu) clearErrorsAndCheckForOOM() {
	for g.getErrorAndCheckForOOM() != NO_ERROR {
	}
}

// getErrorAndCheckForOOM returns the next pending GL error, recording an out-of-memory occurrence.
func (g *Gpu) getErrorAndCheckForOOM() uint32 {
	err := g.fns().GetError()
	if err == OUT_OF_MEMORY {
		g.oomed = true
	}
	return err
}

// CheckAndResetOOMed reports whether GL reported out-of-memory since the last call, and resets the flag.
func (g *Gpu) CheckAndResetOOMed() bool {
	if g.oomed {
		g.oomed = false
		return true
	}
	return false
}
