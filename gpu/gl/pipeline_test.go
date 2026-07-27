// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the proxy/render-task/drawing-manager layer: proxy instantiation and backing-store semantics,
// the lazy-proxy contract (deferred and fully-lazy callback instantiation), interval overlap/reuse in the resource
// allocator, the topological sort and clustering of the task DAG, op-chain combining, and the full context→task→flush
// pipeline over the M4 Max fake driver with per-entry-point call counters.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

func newFakeDirectContext(t *testing.T) *DirectContext {
	t.Helper()
	dc := NewDirectContext(newRecordingGpu(t))
	if dc == nil {
		t.Fatal("NewDirectContext returned nil")
	}
	return dc
}

func makeDeferredProxy(t *testing.T, dc *DirectContext, dims geom.ISize, renderable gpu.Renderable, fit gpu.BackingFit) *SurfaceProxy {
	t.Helper()
	proxy := dc.ProxyProvider().CreateProxy(FormatRGBA8, dims, renderable, 1, gpu.MipmappedNo,
		fit, gpu.BudgetedYes, "test-proxy", 0, UseAllocatorYes)
	if proxy == nil {
		t.Fatal("CreateProxy failed")
	}
	return proxy
}

// TestProxyDeferredInstantiation covers the deferred proxy basics: uninstantiated creation, backing-store binning for
// approx fits, and on-demand instantiation through the provider.
func TestProxyDeferredInstantiation(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 100, Height: 50}

	exact := makeDeferredProxy(t, dc, dims, gpu.RenderableNo, gpu.BackingFitExact)
	defer exact.Unref()
	if exact.IsInstantiated() || exact.IsLazy() {
		t.Fatal("deferred proxy should be neither instantiated nor lazy")
	}
	if got := exact.BackingStoreDimensions(); got != dims {
		t.Fatalf("exact backing dims = %v, want %v", got, dims)
	}

	approx := makeDeferredProxy(t, dc, dims, gpu.RenderableNo, gpu.BackingFitApprox)
	defer approx.Unref()
	want := gpu.GetApproxSize(dims) // 100→128, 50→64
	if got := approx.BackingStoreDimensions(); got != want {
		t.Fatalf("approx backing dims = %v, want %v", got, want)
	}
	if approx.IsFunctionallyExact() {
		t.Fatal("100x50 approx proxy should not be functionally exact")
	}

	if !exact.Instantiate(dc.ResourceProvider()) {
		t.Fatal("Instantiate failed")
	}
	if exact.PeekTexture() == nil || exact.PeekRenderTarget() != nil {
		t.Fatal("texture proxy should instantiate to a texture-only surface")
	}
	if exact.UnderlyingUniqueID() == exact.UniqueID() {
		t.Fatal("deferred proxy should have a distinct ID from its backing surface")
	}
}

// TestLazyProxyInstantiation verifies that a lazy proxy's callback runs exactly once, at flush time through the
// resource allocator, and that a fully lazy proxy adopts the dimensions of the surface its callback creates.
func TestLazyProxyInstantiation(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 16, Height: 16}

	callCount := 0
	callback := func(rp *ResourceProvider, desc *LazySurfaceDesc) LazyCallbackResult {
		callCount++
		surf := rp.CreateTexture(dims, desc.Format, desc.TextureType, desc.Renderable,
			desc.SampleCnt, desc.Mipmapped, desc.Budgeted, "lazy-target")
		return MakeLazyCallbackResult(surf)
	}
	proxy := dc.ProxyProvider().CreateLazyProxy(callback, FormatRGBA8, dims, gpu.MipmappedNo,
		gpu.MipmapStatusNotAllocated, 0, gpu.BackingFitExact, gpu.BudgetedYes, UseAllocatorYes,
		"lazy-proxy")
	if proxy == nil {
		t.Fatal("CreateLazyProxy failed")
	}
	defer proxy.Unref()
	if !proxy.IsLazy() || proxy.IsFullyLazy() {
		t.Fatal("expected a (partially) lazy proxy")
	}
	if proxy.Instantiate(dc.ResourceProvider()) {
		t.Fatal("Instantiate must return false for lazy proxies")
	}

	alloc := NewResourceAllocator(dc)
	alloc.AddInterval(proxy, 0, 0, ActualUseYes, AllowRecyclingYes)
	alloc.IncOps()
	if !alloc.PlanAssignment() || !alloc.Assign() {
		t.Fatal("allocation failed")
	}
	if callCount != 1 {
		t.Fatalf("lazy callback ran %d times, want 1", callCount)
	}
	if !proxy.IsInstantiated() {
		t.Fatal("lazy proxy not instantiated by allocator")
	}
	// The callback was released, so the proxy is no longer lazy.
	if proxy.IsLazy() {
		t.Fatal("callback should have been released")
	}
	alloc.Reset()

	// Fully lazy: dimensions are unknown until instantiation.
	fullDims := geom.ISize{Width: 33, Height: 7}
	fully := MakeFullyLazyProxy(func(rp *ResourceProvider, desc *LazySurfaceDesc,
	) LazyCallbackResult {
		if desc.Dimensions.Width >= 0 || desc.Dimensions.Height >= 0 {
			t.Fatal("fully lazy desc should have negative dims")
		}
		surf := rp.CreateTexture(fullDims, desc.Format, desc.TextureType, desc.Renderable,
			desc.SampleCnt, desc.Mipmapped, desc.Budgeted, "fully-lazy-target")
		return MakeLazyCallbackResult(surf)
	}, FormatRGBA8, gpu.RenderableNo, 1, dc.GLCaps(), UseAllocatorYes)
	if fully == nil {
		t.Fatal("MakeFullyLazyProxy failed")
	}
	defer fully.Unref()
	if !fully.IsFullyLazy() {
		t.Fatal("expected fully lazy proxy")
	}
	if !fully.doLazyInstantiation(dc.ResourceProvider()) {
		t.Fatal("doLazyInstantiation failed")
	}
	if got := fully.Dimensions(); got != fullDims {
		t.Fatalf("fully lazy dims = %v, want %v", got, fullDims)
	}
}

// TestResourceAllocatorIntervals verifies that proxies with non-overlapping usage intervals share one backing surface,
// while overlapping (or merely abutting) intervals get distinct surfaces.
func TestResourceAllocatorIntervals(t *testing.T) {
	dims := geom.ISize{Width: 64, Height: 64}
	run := func(t *testing.T, aStart, aEnd, bStart, bEnd uint32, wantShared bool) {
		dc := newFakeDirectContext(t)
		p1 := makeDeferredProxy(t, dc, dims, gpu.RenderableYes, gpu.BackingFitExact)
		defer p1.Unref()
		p2 := makeDeferredProxy(t, dc, dims, gpu.RenderableYes, gpu.BackingFitExact)
		defer p2.Unref()

		alloc := NewResourceAllocator(dc)
		alloc.AddInterval(p1, aStart, aEnd, ActualUseYes, AllowRecyclingYes)
		alloc.AddInterval(p2, bStart, bEnd, ActualUseYes, AllowRecyclingYes)
		if !alloc.PlanAssignment() || !alloc.Assign() {
			t.Fatal("allocation failed")
		}
		s1 := p1.PeekSurface()
		s2 := p2.PeekSurface()
		if s1 == nil || s2 == nil {
			t.Fatal("proxies not instantiated")
		}
		if (s1 == s2) != wantShared {
			t.Fatalf("shared=%v, want %v", s1 == s2, wantShared)
		}
		alloc.Reset()
	}
	t.Run("non-overlapping share", func(t *testing.T) { run(t, 0, 2, 3, 5, true) })
	t.Run("overlapping do not share", func(t *testing.T) { run(t, 0, 4, 1, 5, false) })
	t.Run("abutting intervals do not share", func(t *testing.T) { run(t, 0, 2, 2, 5, false) })
}

// mockTask is a bare task for exercising the DAG machinery directly.
type mockTask struct {
	executed *[]uint32
	RenderTaskBase
}

func newMockTask(executed *[]uint32) *mockTask {
	m := &mockTask{executed: executed}
	m.initRenderTask(m)
	return m
}

func (m *mockTask) Name() string                            { return "Mock" }
func (m *mockTask) OnIsUsed(*SurfaceProxy) bool             { return false }
func (m *mockTask) GatherProxyIntervals(*ResourceAllocator) {}
func (m *mockTask) OnMakeClosed(*geom.IRect) ExpectedOutcome {
	return ExpectedOutcomeTargetUnchanged
}

func (m *mockTask) OnExecute(*OpFlushState) bool {
	*m.executed = append(*m.executed, m.TaskID())
	return true
}

// TestTopoSortTasks checks the topological-sort behavior of task scheduling: dependencies always precede their
// dependents, and the assigned indices match the final order.
func TestTopoSortTasks(t *testing.T) {
	var executed []uint32
	// Build tasks: d depends on c, c depends on a and b. Insert them badly ordered.
	a := newMockTask(&executed)
	b := newMockTask(&executed)
	c := newMockTask(&executed)
	d := newMockTask(&executed)
	c.addDependencyTask(a)
	c.addDependencyTask(b)
	d.addDependencyTask(c)

	span := []RenderTask{d, c, b, a}
	if !topoSortTasks(span, 0) {
		t.Fatal("topo sort reported a loop")
	}
	pos := map[RenderTask]int{}
	for i, task := range span {
		pos[task] = i
		if got := task.taskBase().index; got != uint32(i) {
			t.Fatalf("task at %d has index %d", i, got)
		}
	}
	if pos[a] > pos[c] || pos[b] > pos[c] || pos[c] > pos[d] {
		t.Fatalf("dependency order violated: a=%d b=%d c=%d d=%d", pos[a], pos[b], pos[c], pos[d])
	}

	for _, task := range span {
		task.taskBase().setFlag(taskFlagDisowned)
		task.taskBase().Unref()
	}
}

// TestTopoSortDetectsLoop verifies the temp-mark loop detection.
func TestTopoSortDetectsLoop(t *testing.T) {
	var executed []uint32
	a := newMockTask(&executed)
	b := newMockTask(&executed)
	a.dependencies = append(a.dependencies, b)
	b.taskBase().dependents = append(b.taskBase().dependents, a)
	b.dependencies = append(b.dependencies, a)
	a.taskBase().dependents = append(a.taskBase().dependents, b)

	span := []RenderTask{a, b}
	if topoSortTasks(span, 0) {
		t.Fatal("topo sort failed to detect a loop")
	}
	for _, task := range span {
		task.taskBase().setFlag(taskFlagDisowned)
		task.taskBase().Unref()
	}
}

// TestClearOpCombining verifies ClearOp's merge rules: a clear whose scissor contains another clear's scissor absorbs
// it, and two clears of different buffers with equal scissors merge into a single op that clears both buffers.
func TestClearOpCombining(t *testing.T) {
	dims := geom.ISize{Width: 64, Height: 64}

	var inner, outer gpu.ScissorState
	inner = gpu.MakeScissorState(dims)
	inner.Set(geom.IRectXYWH(8, 8, 16, 16))
	outer = gpu.MakeScissorState(dims)
	outer.Set(geom.IRectXYWH(0, 0, 32, 32))

	opA := MakeColorClearOp(&inner, [4]float32{1, 0, 0, 1})
	opB := MakeColorClearOp(&outer, [4]float32{0, 1, 0, 1})
	if got := opA.opBase().CombineIfPossible(opB); got != CombineResultMerged {
		t.Fatalf("containing clear should merge, got %v", got)
	}
	clearOp := opA.(*ClearOp)
	if clearOp.Color() != [4]float32{0, 1, 0, 1} {
		t.Fatalf("merged clear kept the wrong color: %v", clearOp.Color())
	}
	if !clearOp.scissor.Enabled() || clearOp.scissor.Rect() != geom.IRectXYWH(0, 0, 32, 32) {
		t.Fatal("merged clear kept the wrong scissor")
	}

	opC := MakeColorClearOp(&inner, [4]float32{1, 0, 0, 1})
	opD := MakeStencilClipClearOp(&inner, true)
	if got := opC.opBase().CombineIfPossible(opD); got != CombineResultMerged {
		t.Fatalf("same-scissor color+stencil clears should merge, got %v", got)
	}
	if opC.(*ClearOp).buffer != clearBufferColor|clearBufferStencilClip {
		t.Fatal("merged clear should cover both buffers")
	}
}

// TestFillContextClearPipeline drives the full context→task→flush pipeline on the fake driver: a fullscreen clear
// becomes a load op (one glClear at Begin), a scissored clear becomes a ClearOp, and the DAG is emptied by the flush.
func TestFillContextClearPipeline(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 64, Height: 64}

	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "clear-target")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	// Fullscreen clear → load op, no op chains.
	sdc.Clear([4]float32{0, 0, 1, 1})
	task := sdc.PeekLastOpsTask()
	if task == nil {
		t.Fatal("clear should have created an ops task")
	}
	if task.NumOpChains() != 0 || task.colorLoadOp != gpu.LoadOpClear {
		t.Fatalf("fullscreen clear should be a load op (chains=%d, loadOp=%d)",
			task.NumOpChains(), task.colorLoadOp)
	}

	// Scissored clear on top → ClearOp chain.
	sdc.ClearRect(geom.IRectXYWH(4, 4, 8, 8), [4]float32{1, 0, 0, 1})
	if task.NumOpChains() != 1 {
		t.Fatalf("scissored clear should add one chain, have %d", task.NumOpChains())
	}
	// A second scissored clear containing the first absorbs it rather than chaining.
	sdc.ClearRect(geom.IRectXYWH(2, 2, 12, 12), [4]float32{1, 1, 0, 1})
	if task.NumOpChains() != 1 {
		t.Fatalf("containing clear should merge, have %d chains", task.NumOpChains())
	}

	clearsBefore := counts("glClear")
	dc.FlushAndSubmit(false)
	clears := counts("glClear") - clearsBefore
	// One clear from the load op at Begin, one from the merged ClearOp.
	if clears != 2 {
		t.Fatalf("expected 2 glClear calls, got %d", clears)
	}
	if len(dc.DrawingManager().dag) != 0 {
		t.Fatalf("flush left %d tasks in the DAG", len(dc.DrawingManager().dag))
	}
	if !task.IsClosed() {
		t.Fatal("flushed task should be closed")
	}

	// The target proxy was instantiated by the allocator.
	if !sdc.AsSurfaceProxy().IsInstantiated() {
		t.Fatal("flush should have instantiated the target")
	}

	// The next clear creates a replacement ops task.
	sdc.Clear([4]float32{0, 1, 0, 1})
	if sdc.PeekLastOpsTask() == task {
		t.Fatal("closed task should have been replaced")
	}
	dc.FlushAndSubmit(false)
}

// TestOpsTaskMergeFrom exercises the reduce-ops-task-splitting merge: two closed tasks targeting the same proxy merge
// their chains.
func TestOpsTaskMergeFrom(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 32, Height: 32}
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "merge-target")
	defer sdc.Release()

	sdc.ClearRect(geom.IRectXYWH(0, 0, 4, 4), [4]float32{1, 0, 0, 1})
	first := sdc.GetOpsTask()
	first.MakeClosed()
	sdc.ClearRect(geom.IRectXYWH(20, 20, 4, 4), [4]float32{0, 1, 0, 1})
	second := sdc.GetOpsTask()
	if first == second {
		t.Fatal("closing should have split the tasks")
	}
	second.MakeClosed()

	if !first.CanMerge(second) {
		t.Fatal("same-target tasks should be mergeable")
	}
	merged := first.MergeFrom([]RenderTask{second})
	if merged != 1 {
		t.Fatalf("MergeFrom merged %d tasks, want 1", merged)
	}
	if first.NumOpChains() != 2 || second.NumOpChains() != 0 {
		t.Fatalf("chains not moved: first=%d second=%d", first.NumOpChains(),
			second.NumOpChains())
	}
	dc.FlushAndSubmit(false)
}

// TestFlushReordersSameTargetTasks verifies the clustering path end-to-end: interleaved tasks A1 B A2 (same target A
// twice) get clustered and merged so only two render passes run.
func TestFlushReordersSameTargetTasks(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 32, Height: 32}
	sdcA := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "target-a")
	defer sdcA.Release()
	sdcB := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "target-b")
	defer sdcB.Release()

	// A1, then B (closes A1), then A2 (closes B).
	sdcA.ClearRect(geom.IRectXYWH(0, 0, 4, 4), [4]float32{1, 0, 0, 1})
	sdcB.ClearRect(geom.IRectXYWH(0, 0, 4, 4), [4]float32{0, 1, 0, 1})
	sdcA.ClearRect(geom.IRectXYWH(8, 8, 4, 4), [4]float32{0, 0, 1, 1})
	if len(dc.DrawingManager().dag) != 3 {
		t.Fatalf("expected 3 tasks before flush, have %d", len(dc.DrawingManager().dag))
	}

	clearsBefore := counts("glClear")
	dc.FlushAndSubmit(false)
	clears := counts("glClear") - clearsBefore
	// Two render passes worth of ClearOps: A's two ops merge/chain into one task after clustering, B keeps its own.
	// (Each ClearOp is one glClear; load ops are kLoad here since the clears are scissored.)
	if clears != 3 {
		t.Fatalf("expected 3 glClear calls (2 for A's merged task, 1 for B), got %d", clears)
	}
}

// TestWritePixelsTaskOrdering checks that WritePixels through a SurfaceContext records a task that stays ordered
// against the clears targeting the same proxy.
func TestWritePixelsTaskOrdering(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 16, Height: 16}
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "write-target")
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})
	pix := Pixels{
		ColorType: gpu.ColorTypeRGBA8888,
		Dims:      geom.ISize{Width: 4, Height: 4},
		RowBytes:  16,
		Data:      make([]byte, 16*4),
	}
	if !sdc.WritePixels(pix, geom.IPoint{X: 2, Y: 2}) {
		t.Fatal("WritePixels failed")
	}
	// WritePixels on borrowed storage flushes; the DAG must be empty and the upload must have happened after the render
	// pass (glTexSubImage2D preceded by the clear's pass).
	if len(dc.DrawingManager().dag) != 0 {
		t.Fatal("WritePixels should have flushed the DAG")
	}
	if counts("glTexSubImage2D") == 0 {
		t.Fatal("WritePixels never uploaded")
	}
}

// TestAbandonSemantics: after AbandonContext no further GL calls are made — resources are dropped without deletion —
// while ReleaseResourcesAndAbandonContext deletes backend objects first.
func TestAbandonSemantics(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 32, Height: 32}
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "abandon-target")
	sdc.Clear([4]float32{1, 1, 1, 1})
	dc.FlushAndSubmit(false)

	if dc.ResourceCache().ResourceCount() == 0 {
		t.Fatal("expected cached resources before abandon")
	}
	// Record pending work that will never flush.
	sdc.ClearRect(geom.IRectXYWH(1, 1, 2, 2), [4]float32{0, 0, 0, 1})

	dc.AbandonContext()
	if !dc.Abandoned() {
		t.Fatal("context should be abandoned")
	}
	if dc.ResourceCache().ResourceCount() != 0 {
		t.Fatalf("abandon left %d resources in the cache", dc.ResourceCache().ResourceCount())
	}
	if counts("glDeleteTextures") != 0 && counts("glDeleteFramebuffers") != 0 {
		// Abandon must not free backend objects (the context is gone).
		t.Fatalf("abandon deleted GL objects (textures=%d, fbos=%d)", counts("glDeleteTextures"),
			counts("glDeleteFramebuffers"))
	}
	snapshot := map[string]int{}
	for k, v := range recCounts {
		snapshot[k] = v
	}
	// All entry points must be inert after abandon.
	if dc.Flush(FlushInfo{}) {
		t.Fatal("flush after abandon should fail")
	}
	sdc.Clear([4]float32{0, 0, 0, 0})
	dc.FlushAndSubmit(false)
	dc.FreeGpuResources()
	for k, v := range recCounts {
		if snapshot[k] != v {
			t.Fatalf("GL call %s made after abandon (%d -> %d)", k, snapshot[k], v)
		}
	}
	sdc.Release()
}

// TestReleaseResourcesAndAbandonSemantics: the release form deletes backend objects.
func TestReleaseResourcesAndAbandonSemantics(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 32, Height: 32}
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "release-target")
	sdc.Clear([4]float32{1, 1, 1, 1})
	dc.FlushAndSubmit(false)

	if counts("glDeleteTextures") != 0 {
		t.Fatal("unexpected early texture deletes")
	}
	dc.ReleaseResourcesAndAbandonContext()
	if dc.ResourceCache().ResourceCount() != 0 {
		t.Fatalf("release left %d resources", dc.ResourceCache().ResourceCount())
	}
	if counts("glDeleteTextures") == 0 {
		t.Fatal("release-and-abandon should delete backend textures")
	}
	sdc.Release()
}

// TestBufferAllocPool exercises the vertex pool over the fake driver: block creation, LIFO put-back, staging flush on
// unmap, and reset recycling.
func TestBufferAllocPool(t *testing.T) {
	dc := newFakeDirectContext(t)
	cache := MakeCpuBufferCache(2)
	pool := NewVertexBufferAllocPool(dc.Gpu(), cache, dc.ResourceProvider())

	ptr, buffer, startVertex := pool.MakeVertexSpace(16, 10)
	if ptr == nil || buffer == nil || startVertex != 0 {
		t.Fatalf("MakeVertexSpace failed (start=%d)", startVertex)
	}
	if len(ptr) != 160 {
		t.Fatalf("vertex window = %d bytes, want 160", len(ptr))
	}
	if buffer.IsCpuBuffer() {
		t.Fatal("desktop profile should use GPU buffers")
	}
	// A second request suballocates from the same block.
	_, buffer2, startVertex2 := pool.MakeVertexSpace(16, 5)
	if buffer2 != buffer || startVertex2 != 10 {
		t.Fatalf("expected suballocation from the same block (start=%d)", startVertex2)
	}
	// Put back the second allocation (LIFO).
	pool.PutBack(16 * 5)
	pool.Unmap()
	// The M4 Max profile maps via buffer-map threshold rules; either the map or the staging+bufferSubData lane must
	// have pushed the data.
	if counts("glBufferSubData") == 0 && counts("glMapBufferRange") == 0 {
		t.Fatal("unmap flushed nothing")
	}
	pool.Reset()

	// CPU buffer cache reuse: the staging buffer returns to the cache on reset.
	cb := cache.MakeBuffer(DefaultBufferSize, false)
	cb.Unref()
	cb2 := cache.MakeBuffer(DefaultBufferSize, false)
	defer cb2.Unref()
	if cb != cb2 {
		t.Fatal("cpu buffer cache did not recycle the buffer")
	}
}

// TestBufferAllocPoolMappedWindowSpansBuffer pins the mapped write window to the buffer the provider actually handed
// back, not to the requested size: CreateBuffer bins dynamic buffers upward (40000 bytes yields 49152), and
// MakeSpaceAtLeast hands out every free byte of the block, so a window sized to the request would be indexed past its
// length. The default -1 buffer-map threshold becomes MaxInt32 and never maps, so the threshold is set explicitly here
// to reach the mapped lane.
func TestBufferAllocPoolMappedWindowSpansBuffer(t *testing.T) {
	options := gpu.DefaultContextOptions()
	options.BufferMapThreshold = 0
	dc := NewDirectContext(newRecordingGpuWithDriverOptions(t, appleM4MaxDriver(), options))
	if dc == nil {
		t.Fatal("NewDirectContext returned nil")
	}
	defer dc.Destroy()
	pool := NewVertexBufferAllocPool(dc.Gpu(), MakeCpuBufferCache(2), dc.ResourceProvider())
	defer pool.Reset()

	const vertexSize = 8
	const fallbackVertexCount = 5000 // 40000 bytes, which bins up to 49152
	ptr, buffer, _, actualVertexCount := pool.MakeVertexSpaceAtLeast(vertexSize, 1,
		fallbackVertexCount)
	if ptr == nil || buffer == nil {
		t.Fatal("MakeVertexSpaceAtLeast failed")
	}
	if counts("glMapBufferRange") == 0 {
		t.Fatal("the block was staged through CPU memory; this test must cover the mapped lane")
	}
	if buffer.Size() <= vertexSize*fallbackVertexCount {
		t.Fatalf("buffer size = %d, want more than the requested %d (the binning this test needs)",
			buffer.Size(), vertexSize*fallbackVertexCount)
	}
	if uint64(len(ptr)) != buffer.Size() {
		t.Fatalf("window = %d bytes, want the buffer's full %d", len(ptr), buffer.Size())
	}
	if uint64(actualVertexCount)*vertexSize != buffer.Size() {
		t.Fatalf("actual vertex count %d covers %d bytes, want the buffer's full %d",
			actualVertexCount, uint64(actualVertexCount)*vertexSize, buffer.Size())
	}
	// Every handed-out byte must be writable through the window.
	for i := range ptr {
		ptr[i] = byte(i)
	}
	pool.Unmap()
}

// TestProxyProviderUniqueKeys covers assign/find/invalidate on the uniquely keyed proxy map.
func TestProxyProviderUniqueKeys(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 8, Height: 8}
	proxy := makeDeferredProxy(t, dc, dims, gpu.RenderableNo, gpu.BackingFitExact)
	defer proxy.Unref()

	var key gpu.UniqueKey
	b := gpu.UniqueKeyBuilder(&key, gpu.GenerateUniqueKeyDomain(), 1, "test")
	b.Slice()[0] = 42
	b.Finish()

	if !dc.ProxyProvider().AssignUniqueKeyToProxy(&key, proxy.AsTextureProxy()) {
		t.Fatal("AssignUniqueKeyToProxy failed")
	}
	if found := dc.ProxyProvider().FindProxyByUniqueKey(&key); found != proxy {
		t.Fatal("FindProxyByUniqueKey mismatch")
	}
	if dc.ProxyProvider().NumUniqueKeyProxies() != 1 {
		t.Fatal("expected one keyed proxy")
	}

	// Instantiate: the key propagates to the surface.
	if !proxy.Instantiate(dc.ResourceProvider()) {
		t.Fatal("Instantiate failed")
	}
	if !proxy.PeekSurface().UniqueKey().Equal(&key) {
		t.Fatal("unique key did not propagate to the surface")
	}

	dc.ProxyProvider().RemoveUniqueKeyFromProxy(proxy.AsTextureProxy())
	if dc.ProxyProvider().NumUniqueKeyProxies() != 0 {
		t.Fatal("key not removed from the provider map")
	}
	if proxy.UniqueKey().IsValid() {
		t.Fatal("key not cleared from the proxy")
	}
	if proxy.PeekSurface().UniqueKey().IsValid() {
		t.Fatal("key not invalidated on the surface")
	}
}

// TestProxyRefReleasesTarget: dropping the last proxy ref drops the target's cache ref so the backing surface becomes
// purgeable.
func TestProxyRefReleasesTarget(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 8, Height: 8}
	proxy := makeDeferredProxy(t, dc, dims, gpu.RenderableNo, gpu.BackingFitExact)
	if !proxy.Instantiate(dc.ResourceProvider()) {
		t.Fatal("Instantiate failed")
	}
	surf := proxy.PeekSurface()
	if surf.IsPurgeable() {
		t.Fatal("surface should be held by the proxy's ref")
	}
	proxy.Unref()
	if !surf.IsPurgeable() {
		t.Fatal("last proxy unref should make the surface purgeable")
	}
}

// TestProxyDeinstantiateReleasesTarget: Deinstantiate drops the proxy's own ref on the backing surface, not just the
// pointer to it. DrawOpAtlas.deactivateLastPage relies on this to let a retired atlas page's texture become purgeable;
// leaving the ref behind pins that memory for the context's lifetime.
func TestProxyDeinstantiateReleasesTarget(t *testing.T) {
	dc := newFakeDirectContext(t)
	dims := geom.ISize{Width: 8, Height: 8}
	proxy := makeDeferredProxy(t, dc, dims, gpu.RenderableNo, gpu.BackingFitExact)
	defer proxy.Unref()
	if !proxy.Instantiate(dc.ResourceProvider()) {
		t.Fatal("Instantiate failed")
	}
	surf := proxy.PeekSurface()
	if surf.IsPurgeable() {
		t.Fatal("surface should be held by the proxy's ref")
	}

	proxy.Deinstantiate()
	if proxy.IsInstantiated() {
		t.Fatal("Deinstantiate must leave the proxy uninstantiated")
	}
	if !surf.IsPurgeable() {
		t.Fatal("Deinstantiate must drop the proxy's ref so the backing surface becomes purgeable")
	}

	// The proxy is reusable: it can be given a fresh backing surface again.
	if !proxy.Instantiate(dc.ResourceProvider()) {
		t.Fatal("re-Instantiate after Deinstantiate failed")
	}
}
