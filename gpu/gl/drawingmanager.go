// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// DrawingManager plus the topological sort and render-task clustering helpers: the owner of the render-task DAG. It
// sorts the DAG topologically at flush (partitioned around reorder blockers), optionally clusters same-target tasks and
// merges them to reduce render passes (the reduce-ops-task-splitting mode, on by default), runs the resource allocator,
// and prepares/executes the tasks. Trims: DDL moves, wait tasks (semaphores are not publicly reachable), the async
// transfer-from task, buffer transfer/update tasks (nothing schedules them), and the write-pixels task (SurfaceContext
// writes directly on a direct context).

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// FlushInfo carries the callbacks for a flush: the semaphore fields are trimmed with the backend-semaphore surface (not
// publicly reachable).
type FlushInfo struct {
	// FinishedProc is called when the flushed work finishes executing on the GPU.
	FinishedProc func()
	// SubmittedProc is called when the flushed work is submitted (success reports whether submission succeeded).
	SubmittedProc func(success bool)
}

// DrawingManager owns the render-task DAG for a context.
type DrawingManager struct {
	lastRenderTasks      map[gpu.ResourceUniqueID]RenderTask
	activeOpsTask        *OpsTask
	softwarePathRenderer *SoftwarePathRenderer
	// cpuBufferCache is used by both the vertex and index pools, reusing memory across flushes.
	cpuBufferCache *CpuBufferCache
	// flushState and resourceAllocator are the per-flush scratch objects that would ideally be stack-allocated for a
	// flush (so a steady-state frame allocates neither), but cannot be — flushState is passed by pointer into
	// executeRenderTasks and the allocator into the reorder/gather/assign helpers, so both escape — so instead they are
	// cached here and reset-and-reused across flushes, exactly like cpuBufferCache/pathRendererChain above. flushState
	// owns the vertex/index BufferAllocPools whose blocks backing array is likewise retained across flushes. Both are
	// cleaned at the end of each flush (OpFlushState.Reset / ResourceAllocator.Reset), so the next flush reuses a
	// pristine object.
	flushState                *OpFlushState
	resourceAllocator         *ResourceAllocator
	pathRendererChain         *PathRendererChain
	context                   *DirectContext
	reorderBlockerTaskIndices []int
	// pendingSubmittedProcs are fired by SubmitToGpu.
	pendingSubmittedProcs []func(success bool)
	// dag holds task refs.
	dag []RenderTask
	// onFlushCBObjects holds the registered on-flush callback objects (the atlas manager, here).
	onFlushCBObjects []OnFlushCallbackObject
	// tokenTracker is the persistent draw/flush token counters the deferred-upload machinery sequences against.
	tokenTracker gpu.TokenTracker
	// Lazily allocated, with the chain options captured from the context options at construction.
	optionsForPathRendererChain PathRendererChainOptions
	flushing                    bool
	reduceOpsTaskSplitting      bool
}

// newDrawingManager creates a DrawingManager for context.
func newDrawingManager(context *DirectContext, reduceOpsTaskSplitting bool) *DrawingManager {
	chainOptions := PathRendererChainOptions{
		AllowPathMaskCaching: true,
		PathRenderers:        gpu.PathRenderersDefault,
	}
	if options := context.Gpu().ContextInfo().Options; options != nil {
		chainOptions.AllowPathMaskCaching = options.AllowPathMaskCaching
		chainOptions.PathRenderers = options.PathRenderers
	}
	return &DrawingManager{
		context:                     context,
		reduceOpsTaskSplitting:      reduceOpsTaskSplitting,
		lastRenderTasks:             make(map[gpu.ResourceUniqueID]RenderTask),
		optionsForPathRendererChain: chainOptions,
	}
}

// destroy tears down the drawing manager, closing and releasing all its tasks.
func (dm *DrawingManager) destroy() {
	dm.closeAllTasks()
	dm.removeRenderTasks()
	dm.pathRendererChain = nil
	dm.softwarePathRenderer = nil
}

// GetPathRenderer returns a renderer that can draw the shape per the chain's priority order, optionally falling back to
// the SW renderer when allowSW is true. stencilSupport (when non-nil) receives the selected renderer's stencil support
// for stencil draw types.
func (dm *DrawingManager) GetPathRenderer(args *CanDrawPathArgs, allowSW bool, drawType ChainDrawType, stencilSupport *StencilSupport) PathRenderer {
	if dm.pathRendererChain == nil {
		dm.pathRendererChain = NewPathRendererChain(dm.context, dm.optionsForPathRendererChain)
	}
	pr := dm.pathRendererChain.GetPathRenderer(args, drawType, stencilSupport)
	if pr == nil && allowSW {
		swPR := dm.GetSoftwarePathRenderer()
		if swPR.OnCanDrawPath(args) != CanDrawPathNo {
			pr = swPR
		}
	}
	return pr
}

// GetSoftwarePathRenderer returns (lazily creating) the SW path renderer. It is separate from the chain so that the
// chain lookup can skip it.
func (dm *DrawingManager) GetSoftwarePathRenderer() *SoftwarePathRenderer {
	if dm.softwarePathRenderer == nil {
		dm.softwarePathRenderer = NewSoftwarePathRenderer(dm.context.ProxyProvider(),
			dm.optionsForPathRendererChain.AllowPathMaskCaching)
	}
	return dm.softwarePathRenderer
}

// GetTessellationPathRenderer works like GetPathRenderer, but only returns the tessellation path renderer (or nil when
// it is not in the chain).
func (dm *DrawingManager) GetTessellationPathRenderer() PathRenderer {
	if dm.pathRendererChain == nil {
		dm.pathRendererChain = NewPathRendererChain(dm.context, dm.optionsForPathRendererChain)
	}
	// The chain returns a concrete *TessellationPathRenderer; returning it directly would wrap a nil pointer in a
	// non-nil PathRenderer interface when the renderer is absent, so the nil check must stay.
	if tess := dm.pathRendererChain.GetTessellationPathRenderer(); tess != nil { //nolint:revive // see above
		return tess
	}
	return nil
}

// Context returns the owning direct context.
func (dm *DrawingManager) Context() *DirectContext { return dm.context }

// TokenTracker exposes the drawing manager's token tracker (the flush state consults it directly; it also lets the
// atlas manager's client query flush tokens between flushes).
func (dm *DrawingManager) TokenTracker() *gpu.TokenTracker { return &dm.tokenTracker }

// AddOnFlushCallbackObject registers a callback to run before/after each flush.
func (dm *DrawingManager) AddOnFlushCallbackObject(onFlushCBObject OnFlushCallbackObject) {
	dm.onFlushCBObjects = append(dm.onFlushCBObjects, onFlushCBObject)
}

func (dm *DrawingManager) wasAbandoned() bool { return dm.context.Abandoned() }

// GetLastRenderTask returns the most recent render task that targeted proxy, or nil.
func (dm *DrawingManager) GetLastRenderTask(proxy *SurfaceProxy) RenderTask {
	return dm.lastRenderTasks[proxy.UniqueID()]
}

// GetLastOpsTask returns the most recent OpsTask that targeted proxy, or nil.
func (dm *DrawingManager) GetLastOpsTask(proxy *SurfaceProxy) *OpsTask {
	if task := dm.GetLastRenderTask(proxy); task != nil {
		return task.AsOpsTask()
	}
	return nil
}

// setLastRenderTask records task as the most recent render task targeting proxy.
func (dm *DrawingManager) setLastRenderTask(proxy *SurfaceProxy, task RenderTask) {
	if prior := dm.GetLastRenderTask(proxy); prior != nil && prior != task &&
		!prior.taskBase().IsClosed() {
		panic("replacing an open task as a proxy's last render task")
	}
	if task != nil {
		dm.lastRenderTasks[proxy.UniqueID()] = task
	} else {
		delete(dm.lastRenderTasks, proxy.UniqueID())
	}
}

// closeActiveOpsTask closes the currently active ops task, if any.
func (dm *DrawingManager) closeActiveOpsTask() {
	if dm.activeOpsTask != nil {
		// In the single-opsTask world, ops that refer to a far earlier render target would have glommed onto the end of
		// the single opsTask; here they need to appear in their own.
		dm.activeOpsTask.MakeClosed()
		dm.activeOpsTask = nil
	}
}

// closeAllTasks closes every task currently in the DAG.
func (dm *DrawingManager) closeAllTasks() {
	for _, task := range dm.dag {
		task.taskBase().MakeClosed()
	}
}

// appendTask appends task to the DAG; the DAG takes over the caller's task ref.
func (dm *DrawingManager) appendTask(task RenderTask) RenderTask {
	if task == nil {
		return nil
	}
	if task.taskBase().BlocksReordering() {
		dm.reorderBlockerTaskIndices = append(dm.reorderBlockerTaskIndices, len(dm.dag))
	}
	dm.dag = append(dm.dag, task)
	return task
}

// insertTaskBeforeLast inserts task into the DAG immediately before the current last task.
func (dm *DrawingManager) insertTaskBeforeLast(task RenderTask) RenderTask {
	if task == nil {
		return nil
	}
	if len(dm.dag) == 0 {
		dm.dag = append(dm.dag, task)
		return task
	}
	// If the current last task is a reorder blocker, the swap below moves it from index len(dm.dag)-1 to len(dm.dag), so
	// its recorded blocker index must bump to match.
	if n := len(dm.reorderBlockerTaskIndices); n > 0 &&
		dm.reorderBlockerTaskIndices[n-1] == len(dm.dag)-1 {
		dm.reorderBlockerTaskIndices[n-1]++
	}
	dm.dag = append(dm.dag, task)
	last := len(dm.dag) - 1
	dm.dag[last], dm.dag[last-1] = dm.dag[last-1], dm.dag[last]
	return task
}

// NewOpsTask creates and appends a new ops task targeting surfaceView. The DAG keeps its own ref; the returned task
// carries a ref owned by the caller.
func (dm *DrawingManager) NewOpsTask(surfaceView SurfaceProxyView, arenas *Arenas) *OpsTask {
	dm.closeActiveOpsTask()

	opsTask := NewOpsTask(dm, surfaceView, arenas)
	if dm.GetLastRenderTask(opsTask.Target(0)) != RenderTask(opsTask) {
		panic("new ops task is not its target's last render task")
	}
	dm.appendTask(opsTask)
	dm.activeOpsTask = opsTask

	opsTask.Ref()
	return opsTask
}

// newTextureResolveRenderTaskBefore creates a resolve task and adds it *before* the last task in the DAG (which will
// come to depend on it), leaving the active opsTask open. The caller must call AddProxy on the result.
func (dm *DrawingManager) newTextureResolveRenderTaskBefore() *TextureResolveRenderTask {
	task := newTextureResolveRenderTask()
	dm.insertTaskBeforeLast(task)
	return task
}

// NewTextureResolveRenderTask creates and appends a resolve task that resolves MSAA and/or regenerates mipmaps on the
// proxy at the end of the current task list.
func (dm *DrawingManager) NewTextureResolveRenderTask(proxy *SurfaceProxy, flags ResolveFlags, caps *Caps) {
	if !proxy.RequiresManualMSAAResolve() {
		return
	}
	lastTask := dm.GetLastRenderTask(proxy)
	if !proxy.AsRenderTargetProxy().IsMSAADirty() &&
		(lastTask == nil || lastTask.taskBase().IsClosed()) {
		return
	}

	dm.closeActiveOpsTask()

	resolveTask := newTextureResolveRenderTask()
	// AddProxy also adds all the needed dependencies.
	resolveTask.AddProxy(dm, proxy, flags, caps)

	dm.appendTask(resolveTask)
	resolveTask.MakeClosed()

	if dm.activeOpsTask != nil {
		panic("no active ops task expected after resolve task")
	}
}

// NewCopyRenderTask creates and appends a copy task. On success the returned task may later be marked skippable if the
// copy is deemed unnecessary (the DAG keeps its own ref; the caller does not own one).
func (dm *DrawingManager) NewCopyRenderTask(dst *SurfaceProxy, dstRect geom.IRect, src *SurfaceProxy, srcRect geom.IRect, filter gpu.FilterMode, origin gpu.SurfaceOrigin) *CopyRenderTask {
	if src.FramebufferOnly() {
		return nil
	}
	dm.closeActiveOpsTask()

	task := MakeCopyRenderTask(dm, dst, dstRect, src, srcRect, filter, origin)
	if task == nil {
		return nil
	}
	dm.appendTask(task)

	// Always kNo here: we are only copying from the base layer to another base layer, so the whole mip chain doesn't
	// need to be valid.
	task.AddDependency(dm, src, gpu.MipmappedNo, dm.context.GLCaps())
	task.MakeClosed()

	if dm.activeOpsTask != nil {
		panic("no active ops task expected after copy task")
	}
	return task
}

// NewWritePixelsTask creates and appends a pixel-upload task (the prefer-flushes-over-VRAM lane is ANGLE-only and
// trimmed).
func (dm *DrawingManager) NewWritePixelsTask(dst *SurfaceProxy, rect geom.IRect, srcColorType, dstColorType gpu.ColorType, levels []gpu.MipLevel) bool {
	dm.closeActiveOpsTask()
	task := MakeWritePixelsTask(dm, dst, rect, srcColorType, dstColorType, levels)
	dm.appendTask(task)
	task.MakeClosed()
	if dm.activeOpsTask != nil {
		panic("no active ops task expected after write-pixels task")
	}
	return true
}

// removeRenderTasks disowns and unrefs every task in the DAG, then clears the DAG for reuse.
func (dm *DrawingManager) removeRenderTasks() {
	for _, task := range dm.dag {
		base := task.taskBase()
		if base.refCnt > 1 {
			// Another holder (e.g. a surface context's ops task) survives the flush: truncate its ops so it starts
			// over.
			task.EndFlush(dm)
		}
		base.Disown(dm)
		base.Unref()
	}
	// Keep the dag and reorder-blocker backings for the next frame's appends; clear the retained elements so dead tasks
	// are not kept reachable.
	clear(dm.dag)
	dm.dag = dm.dag[:0]
	dm.reorderBlockerTaskIndices = dm.reorderBlockerTaskIndices[:0]
	clear(dm.lastRenderTasks)
}

// sortTasks topologically sorts the ranges between non-reorderable tasks.
func (dm *DrawingManager) sortTasks() {
	start := 0
	for i := 0; start < len(dm.dag); i++ {
		end := len(dm.dag)
		if i < len(dm.reorderBlockerTaskIndices) {
			end = dm.reorderBlockerTaskIndices[i]
		}
		span := dm.dag[start:end]
		if !topoSortTasks(span, uint32(start)) {
			panic("render task topo sort failed")
		}
		start = end + 1
	}
}

// topoSortTasks topologically sorts a span of the DAG. When task i depends on task j, j must appear before i. All
// dependencies of a task in the span must be in the span or already output. Returns false when a loop exists (leaving
// the span in an arbitrary state).
func topoSortTasks(span []RenderTask, offset uint32) bool {
	counter := offset
	succeeded := true

	// visit recursively visits a node and all the nodes it depends on.
	var visit func(task RenderTask) bool
	visit = func(task RenderTask) bool {
		base := task.taskBase()
		if base.isSetFlag(taskFlagTempMark) {
			return false // there is a loop
		}
		// Already-output nodes (and all their dependencies) are in the result.
		if base.isSetFlag(taskFlagWasOutput) {
			return true
		}
		ok := true
		// Output all the nodes this one depends on first.
		base.setFlag(taskFlagTempMark)
		for _, dep := range base.dependencies {
			if !visit(dep) {
				ok = false
			}
		}
		base.index = counter
		base.setFlag(taskFlagWasOutput)
		counter++
		base.resetFlag(taskFlagTempMark)
		return ok
	}

	for _, task := range span {
		if task.taskBase().isSetFlag(taskFlagWasOutput) {
			continue
		}
		if !visit(task) {
			succeeded = false
		}
	}
	if counter-offset != uint32(len(span)) {
		panic("topo sort output count mismatch")
	}

	// Reorder the array given the output order.
	for i := range span {
		for correctIndex := span[i].taskBase().index - offset; correctIndex != uint32(i); {
			span[i], span[correctIndex] = span[correctIndex], span[i]
			correctIndex = span[i].taskBase().index - offset
		}
	}
	return succeeded
}

// taskLList is an intrusive doubly-linked list of render tasks used by the clustering pass.
type taskLList struct {
	head RenderTask
	tail RenderTask
}

func (l *taskLList) addToTail(t RenderTask) {
	b := t.taskBase()
	b.llPrev = l.tail
	b.llNext = nil
	if l.tail != nil {
		l.tail.taskBase().llNext = t
	} else {
		l.head = t
	}
	l.tail = t
}

func (l *taskLList) remove(t RenderTask) {
	b := t.taskBase()
	if b.llPrev != nil {
		b.llPrev.taskBase().llNext = b.llNext
	} else {
		l.head = b.llNext
	}
	if b.llNext != nil {
		b.llNext.taskBase().llPrev = b.llPrev
	} else {
		l.tail = b.llPrev
	}
	b.llPrev = nil
	b.llNext = nil
}

func (l *taskLList) addBefore(t, existing RenderTask) {
	b := t.taskBase()
	e := existing.taskBase()
	b.llPrev = e.llPrev
	b.llNext = existing
	if e.llPrev != nil {
		e.llPrev.taskBase().llNext = t
	} else {
		l.head = t
	}
	e.llPrev = t
}

func (l *taskLList) concat(other *taskLList) {
	if other.head == nil {
		return
	}
	if l.tail == nil {
		*l = *other
	} else {
		l.tail.taskBase().llNext = other.head
		other.head.taskBase().llPrev = l.tail
		l.tail = other.tail
	}
	other.head = nil
	other.tail = nil
}

// dependsOnForCluster reports whether dependee is a formal dependency of depender or uses a surface depender targets.
func dependsOnForCluster(depender, dependee RenderTask) bool {
	db := depender.taskBase()
	// Check if depender writes to something dependee reads.
	for i := 0; i < db.NumTargets(); i++ {
		if dependee.taskBase().IsUsed(db.Target(i)) {
			return true
		}
	}
	// Check for a formal dependency.
	return db.DependsOn(dependee)
}

// taskClusterVisit visits one task in the clustering pass, moving it next to other tasks sharing its target when safe.
// Returns whether reordering occurred.
func taskClusterVisit(task RenderTask, llist *taskLList, lastTaskMap map[*SurfaceProxy]RenderTask) bool {
	base := task.taskBase()
	if base.NumTargets() != 1 {
		// Tasks with 0 or multiple targets are treated as full barriers for all their targets.
		for j := 0; j < base.NumTargets(); j++ {
			delete(lastTaskMap, base.Target(j))
		}
		return false
	}

	target := base.Target(0)
	clusterTail := lastTaskMap[target]
	lastTaskMap[target] = task

	if clusterTail == nil {
		return false // no cluster to extend
	}
	if clusterTail == llist.tail {
		return false // cluster is already tail
	}
	movedHead := clusterTail.taskBase().llNext

	// The "cluster" is the chain of tasks with the same target that we hope to extend by reordering; the "moved tasks"
	// are the ones between it and this task.
	clusterHead := clusterTail
	for {
		prev := clusterHead.taskBase().llPrev
		if prev == nil || prev.taskBase().NumTargets() != 1 || target != prev.taskBase().Target(0) {
			break
		}
		clusterHead = prev
	}

	// We can't reorder if any moved task depends on anything in the cluster.
	for moved := movedHead; moved != nil; moved = moved.taskBase().llNext {
		for passed := clusterHead; passed != movedHead; passed = passed.taskBase().llNext {
			if dependsOnForCluster(moved, passed) {
				return false
			}
		}
	}

	// Grab the moved tasks and pull them before clusterHead.
	for moved := movedHead; moved != nil; {
		nextMoved := moved.taskBase().llNext
		llist.remove(moved)
		llist.addBefore(moved, clusterHead)
		moved = nextMoved
	}
	return true
}

// clusterRenderTasks clusters same-target tasks together within input. Returns whether reordering occurred.
func clusterRenderTasks(input []RenderTask, llist *taskLList) bool {
	if len(input) < 3 {
		for _, t := range input {
			llist.addToTail(t)
		}
		return false
	}
	lastTaskMap := make(map[*SurfaceProxy]RenderTask)
	didReorder := false
	for _, t := range input {
		if taskClusterVisit(t, llist, lastTaskMap) {
			didReorder = true
		}
		llist.addToTail(t)
	}
	return didReorder
}

// reorderTasks attempts to reorder tasks to reduce render passes and check the memory budget of the resulting
// intervals. On true, the DAG has been updated to reflect the reordered tasks.
func (dm *DrawingManager) reorderTasks(resourceAllocator *ResourceAllocator) bool {
	if !dm.reduceOpsTaskSplitting {
		panic("reorderTasks without reduceOpsTaskSplitting")
	}
	// We separately cluster the ranges around non-reorderable tasks.
	clustered := false
	var llist taskLList
	start := 0
	for i := 0; start < len(dm.dag); i++ {
		end := len(dm.dag)
		if i < len(dm.reorderBlockerTaskIndices) {
			end = dm.reorderBlockerTaskIndices[i]
		}
		span := dm.dag[start:end]

		var subllist taskLList
		if clusterRenderTasks(span, &subllist) {
			clustered = true
		}
		if i < len(dm.reorderBlockerTaskIndices) {
			subllist.addToTail(dm.dag[dm.reorderBlockerTaskIndices[i]])
		}
		llist.concat(&subllist)
		start = end + 1
	}
	if !clustered {
		return false
	}

	for task := llist.head; task != nil; task = task.taskBase().llNext {
		task.GatherProxyIntervals(resourceAllocator)
	}
	if !resourceAllocator.PlanAssignment() {
		return false
	}
	if !resourceAllocator.MakeBudgetHeadroom() {
		return false
	}
	// Reorder the array to match the llist.
	i := 0
	for task := llist.head; task != nil; task = task.taskBase().llNext {
		dm.dag[i] = task
		i++
	}
	if i != len(dm.dag) {
		panic("cluster llist lost tasks")
	}

	newCount := 0
	for i = 0; i < len(dm.dag); i++ {
		task := dm.dag[i]
		if opsTask := task.AsOpsTask(); opsTask != nil {
			nextTasks := dm.dag[i+1:]
			removeCount := opsTask.MergeFrom(nextTasks)
			for _, removed := range nextTasks[:removeCount] {
				removed.taskBase().Disown(dm)
				removed.taskBase().Unref()
			}
			i += removeCount
		}
		dm.dag[newCount] = task
		newCount++
	}
	dm.dag = dm.dag[:newCount]
	return true
}

// executeRenderTasks prepares and executes every instantiated task in the DAG. Returns true if any render tasks were
// actually executed.
func (dm *DrawingManager) executeRenderTasks(flushState *OpFlushState) bool {
	anyRenderTasksExecuted := false

	for _, renderTask := range dm.dag {
		if !renderTask.taskBase().IsInstantiated() {
			continue
		}
		renderTask.taskBase().Prepare(flushState)
	}

	// Upload all data to the GPU.
	flushState.PreExecuteDraws()

	// A cap on the number of tasks executed before flushing to the GPU to relieve memory pressure.
	const maxRenderTasksBeforeFlush = 100
	numRenderTasksExecuted := 0

	// Execute the normal op lists.
	for _, renderTask := range dm.dag {
		if !renderTask.taskBase().IsInstantiated() {
			continue
		}
		if renderTask.taskBase().Execute(flushState) {
			anyRenderTasksExecuted = true
		}
		numRenderTasksExecuted++
		if numRenderTasksExecuted >= maxRenderTasksBeforeFlush {
			flushState.Gpu().SubmitToGpu(false)
			numRenderTasksExecuted = 0
		}
	}

	if flushState.OpsRenderPass() != nil {
		panic("render pass still active after execution")
	}
	if dm.tokenTracker.NextDrawToken() != dm.tokenTracker.NextFlushToken() {
		panic("draw/flush token imbalance after executing render tasks")
	}
	// We reset the flush state before the render tasks so that the last resources to be freed are those that are
	// written to in the render tasks: the most recently used resources are the last to be purged by the resource cache.
	flushState.Reset()

	return anyRenderTasksExecuted
}

// Flush executes and submits the pending work for the given proxies (empty for the whole context) (the
// surface-access/mutable-state parameters are trimmed with the backends that consume them).
func (dm *DrawingManager) Flush(proxies []*SurfaceProxy, info FlushInfo) bool {
	if dm.flushing || dm.wasAbandoned() {
		if info.SubmittedProc != nil {
			info.SubmittedProc(false)
		}
		if info.FinishedProc != nil {
			info.FinishedProc()
		}
		return false
	}

	// As of now we only short-circuit if we got an explicit list of surfaces to flush. A finished proc has to observe a
	// real submission, so it blocks the short-circuit; a submitted proc does not, since it is called below. (Upstream
	// also blocks on pending semaphores, which this trim has none of.)
	if len(proxies) > 0 && info.FinishedProc == nil {
		allUnused := true
		for _, proxy := range proxies {
			for _, task := range dm.dag {
				if task.taskBase().IsUsed(proxy) {
					allUnused = false
					break
				}
			}
			if !allUnused {
				break
			}
		}
		if allUnused {
			if info.SubmittedProc != nil {
				info.SubmittedProc(true)
			}
			return false
		}
	}

	g := dm.context.Gpu()
	dm.flushing = true

	resourceCache := dm.context.ResourceCache()

	// Semi-usually the render tasks are already closed at this point, but sometimes a flush is needed mid-draw. In that
	// case, the devices' ops tasks won't be closed but need to be flushed anyway; new ones will be created to replace
	// them if written to again.
	dm.closeAllTasks()
	dm.activeOpsTask = nil

	dm.sortTasks()

	if dm.cpuBufferCache == nil {
		// We cache more buffers when the backend is using client-side arrays; otherwise each pool only needs one
		// staging buffer at a time.
		maxCachedBuffers := 6
		if dm.context.Caps().PreferClientSideDynamicBuffers {
			maxCachedBuffers = 2
		}
		dm.cpuBufferCache = MakeCpuBufferCache(maxCachedBuffers)
	}

	// Create the flush state once and reuse it: it (and its vertex/index buffer pools) is reset after every flush, so
	// the retained object is pristine here. All constructor arguments are stable for the context's lifetime
	// (cpuBufferCache is set just above and never reassigned), so capturing them once is safe.
	if dm.flushState == nil {
		dm.flushState = NewOpFlushState(g, dm.context.ResourceProvider(), &dm.tokenTracker,
			dm.cpuBufferCache, dm.context.ThreadSafeCache(), dm.context.AtlasManager(),
			dm.context.TextStrikeCache())
	}
	flushState := dm.flushState

	// Prepare any onFlush op lists (e.g. atlases).
	preFlushSuccessful := true
	for _, onFlushCBObject := range dm.onFlushCBObjects {
		preFlushSuccessful = onFlushCBObject.PreFlush() && preFlushSuccessful
	}

	cachePurgeNeeded := false
	if preFlushSuccessful {
		usingReorderedDAG := false
		// Reuse the resource allocator across flushes (see the flushState comment). Its end-of-flush Reset (below)
		// clears the intervals, maps, and registers; beginFlush additionally clears the failedInstantiation latch that
		// Reset intentionally leaves set for the in-flush reorder retry, so the starting state matches a freshly
		// constructed allocator's exactly — the in-flush reorder/gather/assign logic is therefore unchanged.
		if dm.resourceAllocator == nil {
			dm.resourceAllocator = NewResourceAllocator(dm.context)
		}
		resourceAllocator := dm.resourceAllocator
		resourceAllocator.beginFlush()
		if dm.reduceOpsTaskSplitting {
			usingReorderedDAG = dm.reorderTasks(resourceAllocator)
			if !usingReorderedDAG {
				resourceAllocator.Reset()
			}
		}
		if !resourceAllocator.FailedInstantiation() {
			if !usingReorderedDAG {
				for _, task := range dm.dag {
					task.GatherProxyIntervals(resourceAllocator)
				}
				resourceAllocator.PlanAssignment()
			}
			resourceAllocator.Assign()
		}
		cachePurgeNeeded = !resourceAllocator.FailedInstantiation() &&
			dm.executeRenderTasks(flushState)
		// Release the register-held surface refs.
		resourceAllocator.Reset()
	}
	dm.removeRenderTasks()

	// Trimmed to the reachable pieces: finished procs ride the Gpu's finished-callback machinery; submitted procs fire
	// at the next submit.
	if info.FinishedProc != nil {
		g.AddFinishedCallback(info.FinishedProc)
	}
	if info.SubmittedProc != nil {
		dm.pendingSubmittedProcs = append(dm.pendingSubmittedProcs, info.SubmittedProc)
	}

	// Give the cache a chance to purge resources that become purgeable due to flushing.
	if cachePurgeNeeded {
		resourceCache.PurgeAsNeeded()
		cachePurgeNeeded = false
	}
	for _, onFlushCBObject := range dm.onFlushCBObjects {
		onFlushCBObject.PostFlush(dm.tokenTracker.NextFlushToken())
		cachePurgeNeeded = true
	}
	if cachePurgeNeeded {
		resourceCache.PurgeAsNeeded()
	}
	dm.flushing = false
	return true
}

// SubmitToGpu submits queued GPU work and fires any pending submitted-procs.
func (dm *DrawingManager) SubmitToGpu(syncCPU bool) bool {
	if dm.flushing || dm.wasAbandoned() {
		return false
	}
	submitted := dm.context.Gpu().SubmitToGpu(syncCPU)
	for _, proc := range dm.pendingSubmittedProcs {
		proc(submitted)
	}
	dm.pendingSubmittedProcs = nil
	return submitted
}

// FlushSurfaces flushes, then resolves MSAA and regenerates dirty mipmaps on the given proxies immediately (clients
// expect flushed surfaces' backing textures to be fully resolved upon return).
func (dm *DrawingManager) FlushSurfaces(proxies []*SurfaceProxy, info FlushInfo) bool {
	if dm.wasAbandoned() {
		if info.SubmittedProc != nil {
			info.SubmittedProc(false)
		}
		if info.FinishedProc != nil {
			info.FinishedProc()
		}
		return false
	}
	g := dm.context.Gpu()
	didFlush := dm.Flush(proxies, info)
	for _, proxy := range proxies {
		resolveAndMipmap(dm, g, proxy)
	}
	return didFlush
}

// resolveAndMipmap resolves a dirty MSAA target and/or regenerates dirty mipmaps for proxy.
func resolveAndMipmap(dm *DrawingManager, g *Gpu, proxy *SurfaceProxy) {
	if !proxy.IsInstantiated() {
		return
	}
	if proxy.RequiresManualMSAAResolve() {
		rtProxy := proxy.AsRenderTargetProxy()
		if rtProxy.IsMSAADirty() {
			g.ResolveRenderTarget(proxy.PeekRenderTarget(), rtProxy.MSAADirtyRect())
			dm.SubmitToGpu(false)
			rtProxy.MarkMSAAResolved()
		}
	}
	// If, after a flush, any of the proxies of interest have dirty mipmaps, regenerate them in case their backend
	// textures are being stolen.
	if textureProxy := proxy.AsTextureProxy(); textureProxy != nil {
		if textureProxy.MipmapsAreDirty() {
			g.RegenerateMipmapLevels(proxy.PeekTexture())
			textureProxy.MarkMipmapsClean()
		}
	}
}

// FreeGpuResources drops the on-flush callback objects that don't ask to be retained, and the path renderers (which may
// be holding onto resources).
func (dm *DrawingManager) FreeGpuResources() {
	kept := dm.onFlushCBObjects[:0]
	for _, onFlushCBObject := range dm.onFlushCBObjects {
		if onFlushCBObject.RetainOnFreeGpuResources() {
			kept = append(kept, onFlushCBObject)
		}
	}
	dm.onFlushCBObjects = kept
	// A path renderer may be holding onto resources.
	dm.pathRendererChain = nil
	dm.softwarePathRenderer = nil
}
