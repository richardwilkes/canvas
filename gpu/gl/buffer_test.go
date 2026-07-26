// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for buffer creation, built on the recording Gpu from gpu_test.go.

package gl

import (
	"sync"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/richardwilkes/canvas/gpu"
)

// The out-of-memory procs are created once and shared, since purego callbacks are never freed and the process has a
// hard cap on them. pendingGLError is armed by the glBufferData proc and consumed by the glGetError proc; tests are not
// parallel, so a single package-level slot is enough (the same discipline recCounts et al. rely on).
var (
	oomProcsOnce      sync.Once
	oomBufferDataProc uintptr
	oomGetErrorProc   uintptr
	pendingGLError    uint32
)

// installBufferDataOOM makes glBufferData report GL_OUT_OF_MEMORY, the way a driver answers an allocation it cannot
// satisfy.
func installBufferDataOOM(g *Gpu) {
	oomProcsOnce.Do(func() {
		oomBufferDataProc = purego.NewCallback(func(_, _, _, _ uintptr) uintptr {
			if recCounts != nil {
				recCounts["glBufferData"]++
			}
			pendingGLError = OUT_OF_MEMORY
			return 0
		})
		oomGetErrorProc = purego.NewCallback(func() uintptr {
			err := pendingGLError
			pendingGLError = NO_ERROR
			return uintptr(err)
		})
	})
	pendingGLError = NO_ERROR
	f := g.fns()
	f.bufferData = oomBufferDataProc
	f.getError = oomGetErrorProc
}

// TestMakeBufferAllocFailureReleasesResource verifies that a buffer whose glBufferData fails is not left registered in
// the resource cache: it would otherwise sit there forever with its full requested size counted against the budget.
func TestMakeBufferAllocFailureReleasesResource(t *testing.T) {
	const size = 4096
	for _, tc := range []struct {
		name    string
		pattern gpu.AccessPattern
	}{
		{name: "dynamic", pattern: gpu.AccessPatternDynamic}, // scratch-keyed
		{name: "static", pattern: gpu.AccessPatternStatic},   // no scratch key
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := newRecordingGpu(t)
			cache := g.ResourceCache()
			baseCount := cache.ResourceCount()
			baseBytes := cache.ResourceBytes()
			baseBudgeted := cache.BudgetedResourceBytes()
			installBufferDataOOM(g)
			deleteBase := counts("glDeleteBuffers")

			if b := MakeBuffer(g, size, gpu.BufferTypeVertex, tc.pattern); b != nil {
				t.Fatal("MakeBuffer should return nil when glBufferData reports out of memory")
			}
			if !g.CheckAndResetOOMed() {
				t.Error("the out-of-memory error was not noticed")
			}
			if got := counts("glDeleteBuffers"); got != deleteBase+1 {
				t.Errorf("glDeleteBuffers calls = %d, want %d", got, deleteBase+1)
			}
			if got := cache.ResourceCount(); got != baseCount {
				t.Errorf("cache holds %d resources after the failed allocation, want %d", got, baseCount)
			}
			if got := cache.ResourceBytes(); got != baseBytes {
				t.Errorf("cache accounts %d bytes after the failed allocation, want %d", got, baseBytes)
			}
			if got := cache.BudgetedResourceBytes(); got != baseBudgeted {
				t.Errorf("budgeted bytes = %d after the failed allocation, want %d", got, baseBudgeted)
			}

			// The dead buffer must not be reachable as a scratch resource either.
			var key gpu.ScratchKey
			ComputeScratchKeyForDynamicBuffer(size, gpu.BufferTypeVertex, &key)
			if found := cache.FindAndRefScratchResource(&key); found != nil {
				t.Errorf("a failed buffer allocation was left available for scratch reuse: %s", found.ResourceType())
			}
		})
	}
}

// TestMakeBufferSuccessKeepsResource pins the other side of the failure path: a buffer that allocates successfully is
// returned live, registered, and budgeted, and only leaves the cache once the caller drops its ref.
func TestMakeBufferSuccessKeepsResource(t *testing.T) {
	const size = 4096
	g := newRecordingGpu(t)
	cache := g.ResourceCache()
	baseCount := cache.ResourceCount()
	baseBytes := cache.ResourceBytes()

	b := MakeBuffer(g, size, gpu.BufferTypeVertex, gpu.AccessPatternStatic)
	if b == nil {
		t.Fatal("MakeBuffer failed")
	}
	if b.WasDestroyed() {
		t.Error("a successfully created buffer must not be destroyed")
	}
	if got := cache.ResourceCount(); got != baseCount+1 {
		t.Errorf("cache holds %d resources, want %d", got, baseCount+1)
	}
	if got := cache.ResourceBytes(); got != baseBytes+size {
		t.Errorf("cache accounts %d bytes, want %d", got, baseBytes+size)
	}

	b.Unref()
	cache.PurgeUnlockedResources(gpu.PurgeAllResources)
	if got := cache.ResourceCount(); got != baseCount {
		t.Errorf("cache holds %d resources after release, want %d", got, baseCount)
	}
}
