// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A render task targets surface proxies, participates in the drawing manager's DAG, and modifies its targets' contents
// on execute. This is expressed as a RenderTask interface plus an embedded RenderTaskBase (the Resource/Op pattern).
// Tasks hold refs on their target proxies via an explicit ref count whose final Unref performs the destructor work
// (drop target-proxy refs and the task's own cleanup). Texture-resolve scheduling goes through direct DrawingManager
// calls. The deferredProxies bookkeeping for the software path renderer's deferred texture uploads is tracked here, but
// the actual upload scheduling is not wired.

package gl

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// ExpectedOutcome reports whether a task's OnMakeClosed left its target proxy unchanged or dirty.
type ExpectedOutcome bool

// ExpectedOutcome values.
const (
	ExpectedOutcomeTargetUnchanged ExpectedOutcome = false
	ExpectedOutcomeTargetDirty     ExpectedOutcome = true
)

// RenderTaskBase flags.
const (
	taskFlagClosed           uint32 = 0x01 // this task can't accept any more dependencies
	taskFlagDisowned         uint32 = 0x02 // this task is disowned by its DrawingManager
	taskFlagSkippable        uint32 = 0x04 // this task is skippable
	taskFlagAtlas            uint32 = 0x08 // this task is atlas
	taskFlagBlocksReordering uint32 = 0x10 // no task can be reordered with respect to this task
	taskFlagWasOutput        uint32 = 0x20 // flag for topological sorting
	taskFlagTempMark         uint32 = 0x40 // flag for topological sorting
)

var nextRenderTaskID atomic.Uint32

// createRenderTaskID returns a fresh, never-zero, process-unique render task ID.
func createRenderTaskID() uint32 {
	for {
		if id := nextRenderTaskID.Add(1); id != 0 {
			return id
		}
	}
}

// RenderTask is the interface all render tasks implement.
type RenderTask interface {
	taskBase() *RenderTaskBase
	// Name identifies the task kind.
	Name() string
	// AsOpsTask returns the task as an *OpsTask, or nil if it isn't one (RenderTaskBase defaults to nil).
	AsOpsTask() *OpsTask
	// OnMakeClosed finalizes the task prior to execution; when returning target-dirty, it fills targetUpdateBounds with
	// the modified area (in native surface space, not extending beyond the proxy bounds).
	OnMakeClosed(targetUpdateBounds *geom.IRect) ExpectedOutcome
	// OnMakeSkippable lets the task react to being marked skippable (base default no-op).
	OnMakeSkippable()
	// OnIsUsed reports usage of proxies other than target proxies.
	OnIsUsed(*SurfaceProxy) bool
	// GatherProxyIntervals feeds proxy usage intervals to the resource allocator.
	GatherProxyIntervals(*ResourceAllocator)
	// OnPrepare prepares GPU data before execution (only OpsTask overrides; base default no-op).
	OnPrepare(*OpFlushState)
	// OnExecute issues the task's GPU work.
	OnExecute(*OpFlushState) bool
	// EndFlush runs once after a flush completes (base default no-op).
	EndFlush(*DrawingManager)
	// onDelete is the destructor hook run by the final Unref (e.g. OpsTask deletes its ops).
	onDelete()
}

// RenderTaskBase carries the common state shared by every concrete task and is embedded by each.
type RenderTaskBase struct {
	self   RenderTask
	llNext RenderTask
	// llPrev/llNext are the intrusive linked-list links used by the render-task clustering pass in the drawing manager.
	llPrev RenderTask
	// textureResolveTask: for performance, texture resolves are performed back-to-back, so one resolve task is made and
	// reused for each render task, added as a dependency at makeClosed.
	textureResolveTask *TextureResolveRenderTask
	// deferredProxies lists texture proxies whose contents are being prepared on a worker thread (software path
	// renderer masks).
	deferredProxies []*TextureProxy
	// targets hold proxy refs, dropped at destruction.
	targets []*SurfaceProxy
	// dependencies are the tasks whose output this task relies on; dependents rely on this task.
	dependencies []RenderTask
	dependents   []RenderTask
	// closedUpdateBounds is MakeClosed's scratch for the OnMakeClosed out-parameter. A local would escape to the heap
	// on every close (the callee is a virtual interface method), and the task itself is already heap-resident, so the
	// scratch rides here for free. Only valid during MakeClosed.
	closedUpdateBounds geom.IRect
	// index is the topological-sort output index.
	index    uint32
	flags    uint32
	uniqueID uint32
	refCnt   int32
}

func (b *RenderTaskBase) taskBase() *RenderTaskBase { return b }

// initRenderTask initializes the base state shared by every concrete task.
func (b *RenderTaskBase) initRenderTask(self RenderTask) {
	b.self = self
	b.refCnt = 1
	b.uniqueID = createRenderTaskID()
}

// Ref takes a reference on the task.
func (b *RenderTaskBase) Ref() { b.refCnt++ }

// Unref drops a reference; the final one performs the destructor work: the task must already be disowned, its
// target-proxy refs drop, and the concrete task's onDelete hook runs. A dead OpsTask is then returned to the free list
// (opstaskpool.go) — the targets backing keeps its capacity (elements cleared) so the pooled task's next addTarget
// append does not re-grow it.
func (b *RenderTaskBase) Unref() {
	b.refCnt--
	if b.refCnt > 0 {
		return
	}
	if b.refCnt < 0 {
		panic("RenderTask over-unref")
	}
	if !b.isSetFlag(taskFlagDisowned) {
		panic("render task destroyed without being disowned")
	}
	b.self.onDelete()
	for _, target := range b.targets {
		target.Unref()
	}
	clear(b.targets)
	b.targets = b.targets[:0]
	if t := b.self.AsOpsTask(); t != nil {
		recycleOpsTask(t)
	}
}

// onDelete is the base default destructor hook.
func (b *RenderTaskBase) onDelete() {}

// AsOpsTask is the base default.
func (b *RenderTaskBase) AsOpsTask() *OpsTask { return nil }

// OnMakeSkippable is the base default.
func (b *RenderTaskBase) OnMakeSkippable() {}

// OnPrepare is the base default.
func (b *RenderTaskBase) OnPrepare(*OpFlushState) {}

// EndFlush is the base default (a no-op).
func (b *RenderTaskBase) EndFlush(*DrawingManager) {}

func (b *RenderTaskBase) setFlag(flag uint32)        { b.flags |= flag }
func (b *RenderTaskBase) resetFlag(flag uint32)      { b.flags &^= flag }
func (b *RenderTaskBase) isSetFlag(flag uint32) bool { return b.flags&flag != 0 }

// TaskID returns the task's unique ID.
func (b *RenderTaskBase) TaskID() uint32 { return b.uniqueID }

// IsClosed reports whether the task has been closed.
func (b *RenderTaskBase) IsClosed() bool { return b.isSetFlag(taskFlagClosed) }

// IsSkippable reports whether the task has been marked skippable.
func (b *RenderTaskBase) IsSkippable() bool { return b.isSetFlag(taskFlagSkippable) }

// BlocksReordering reports whether no task can be reordered with respect to this one.
func (b *RenderTaskBase) BlocksReordering() bool { return b.isSetFlag(taskFlagBlocksReordering) }

// MakeSkippable marks the task skippable; purely an optimization — not all tasks actually skip their work.
func (b *RenderTaskBase) MakeSkippable() {
	if !b.IsClosed() {
		panic("makeSkippable on an open task")
	}
	if !b.IsSkippable() {
		b.setFlag(taskFlagSkippable)
		b.self.OnMakeSkippable()
	}
}

// Disown "disowns" the proxies this task modifies by telling the drawing manager to forget the proxy → last-render-task
// mappings.
func (b *RenderTaskBase) Disown(drawingMgr *DrawingManager) {
	if !b.IsClosed() {
		panic("disowning an open task")
	}
	if b.isSetFlag(taskFlagDisowned) {
		return
	}
	b.setFlag(taskFlagDisowned)
	for _, target := range b.targets {
		if b.self == drawingMgr.GetLastRenderTask(target) {
			drawingMgr.setLastRenderTask(target, nil)
		}
	}
}

// MakeClosed finalizes the task prior to execution, marking dependent proxy state (MSAA/mipmap dirty) as needed and
// closing any pending texture-resolve task.
func (b *RenderTaskBase) MakeClosed() {
	if b.IsClosed() {
		return
	}
	b.closedUpdateBounds = geom.IRect{}
	targetUpdateBounds := &b.closedUpdateBounds
	if b.self.OnMakeClosed(targetUpdateBounds) == ExpectedOutcomeTargetDirty {
		proxy := b.Target(0)
		if proxy.RequiresManualMSAAResolve() {
			rtProxy := proxy.AsRenderTargetProxy()
			if rtProxy == nil {
				panic("manual MSAA resolve without a render target proxy")
			}
			rtProxy.MarkMSAADirty(*targetUpdateBounds)
		}
		if textureProxy := proxy.AsTextureProxy(); textureProxy != nil &&
			textureProxy.Mipmapped() == gpu.MipmappedYes {
			textureProxy.MarkMipmapsDirty()
		}
	}
	if b.textureResolveTask != nil {
		b.addDependencyTask(b.textureResolveTask)
		b.textureResolveTask.MakeClosed()
		b.textureResolveTask = nil
	}
	b.setFlag(taskFlagClosed)
}

// Prepare runs the task's OnPrepare (deferred-proxy upload scheduling is not wired).
func (b *RenderTaskBase) Prepare(flushState *OpFlushState) {
	b.self.OnPrepare(flushState)
}

// Execute runs the task's OnExecute.
func (b *RenderTaskBase) Execute(flushState *OpFlushState) bool {
	return b.self.OnExecute(flushState)
}

// NumTargets returns the number of target proxies.
func (b *RenderTaskBase) NumTargets() int { return len(b.targets) }

// Target returns the i'th target proxy.
func (b *RenderTaskBase) Target(i int) *SurfaceProxy { return b.targets[i] }

// addTarget takes a ref on the proxy and updates the drawing manager's last-render-task association.
func (b *RenderTaskBase) addTarget(drawingMgr *DrawingManager, proxy *SurfaceProxy) {
	if proxy == nil {
		panic("nil task target")
	}
	if b.IsClosed() {
		panic("adding a target to a closed task")
	}
	drawingMgr.setLastRenderTask(proxy, b.self)
	proxy.IsUsedAsTaskTarget()
	proxy.Ref()
	b.targets = append(b.targets, proxy)
}

// addDependencyTask records dependedOn as a task this task depends on.
func (b *RenderTaskBase) addDependencyTask(dependedOn RenderTask) {
	if dependedOn.taskBase().DependsOn(b.self) {
		panic("dependency loop")
	}
	if b.DependsOn(dependedOn) {
		panic("duplicate dependency; caller should weed out duplicates")
	}
	b.dependencies = append(b.dependencies, dependedOn)
	dependedOn.taskBase().addDependent(b.self)
}

// AddDependenciesFromOtherTask makes this task rely on everything otherTask depends on.
func (b *RenderTaskBase) AddDependenciesFromOtherTask(otherTask RenderTask) {
	for _, task := range otherTask.taskBase().dependencies {
		if task == b.self {
			panic("task adding a dependency to itself")
		}
		if !b.DependsOn(task) {
			b.addDependencyTask(task)
		}
	}
}

// Dependencies returns the tasks this task depends on.
func (b *RenderTaskBase) Dependencies() []RenderTask { return b.dependencies }

// Dependents returns the tasks that depend on this task.
func (b *RenderTaskBase) Dependents() []RenderTask { return b.dependents }

// ReplaceDependency replaces a dependency on toReplace with one on replaceWith.
func (b *RenderTaskBase) ReplaceDependency(toReplace, replaceWith RenderTask) {
	for i, dep := range b.dependencies {
		if dep == toReplace {
			b.dependencies[i] = replaceWith
			rw := replaceWith.taskBase()
			rw.dependents = append(rw.dependents, b.self)
			break
		}
	}
}

// ReplaceDependent replaces a dependent task's reference from toReplace to replaceWith.
func (b *RenderTaskBase) ReplaceDependent(toReplace, replaceWith RenderTask) {
	for i, dep := range b.dependents {
		if dep == toReplace {
			b.dependents[i] = replaceWith
			rw := replaceWith.taskBase()
			rw.dependencies = append(rw.dependencies, b.self)
			break
		}
	}
}

// DependsOn reports whether this task depends on dependedOn.
func (b *RenderTaskBase) DependsOn(dependedOn RenderTask) bool {
	for _, dep := range b.dependencies {
		if dep == dependedOn {
			return true
		}
	}
	return false
}

func (b *RenderTaskBase) addDependent(dependent RenderTask) {
	b.dependents = append(b.dependents, dependent)
}

// AddDependency converts a proxy-based dependency into a task one: it finds the task that last rendered to dependedOn
// and adds a dependency on it, resolving MSAA/mipmap regeneration first if needed. mipmapped indicates the proxy will
// be sampled with mipmaps.
func (b *RenderTaskBase) AddDependency(drawingMgr *DrawingManager, dependedOn *SurfaceProxy, mipmapped gpu.Mipmapped, caps *Caps) {
	// If it is still receiving dependencies, this task shouldn't be closed.
	if b.IsClosed() {
		panic("adding a dependency to a closed task")
	}

	dependedOnTask := drawingMgr.GetLastRenderTask(dependedOn)

	if dependedOnTask == b.self {
		// Self-read — presumably for dst reads. We don't need to do anything in this case: the XferProcessor will
		// detect what is happening and insert a texture barrier.
		if mipmapped == gpu.MipmappedYes || dependedOn.RequiresManualMSAAResolve() {
			panic("invalid self-read")
		}
		return
	}

	alreadyDependent := false
	if dependedOnTask != nil {
		isResolveTask := b.textureResolveTask != nil &&
			RenderTask(b.textureResolveTask) == dependedOnTask
		if b.DependsOn(dependedOnTask) || isResolveTask {
			alreadyDependent = true
			dependedOnTask = nil // don't add duplicate dependencies
		} else if !dependedOnTask.taskBase().isSetFlag(taskFlagAtlas) {
			// We are closing 'dependedOnTask' here because its current contents are what this task depends on: a break
			// is needed so that usage of that state has a chance to execute.
			dependedOnTask.taskBase().MakeClosed()
		}
	}

	var resolveFlags ResolveFlags
	if dependedOn.RequiresManualMSAAResolve() {
		rtProxy := dependedOn.AsRenderTargetProxy()
		if rtProxy == nil {
			panic("manual MSAA resolve without a render target proxy")
		}
		if rtProxy.IsMSAADirty() {
			resolveFlags |= ResolveFlagMSAA
		}
	}

	textureProxy := dependedOn.AsTextureProxy()
	if mipmapped == gpu.MipmappedYes {
		if textureProxy == nil {
			panic("mipmapped sampling of a non-texture proxy")
		}
		// We might be given a non-mipmapped texture with a mipmap filter, in which case there is nothing to
		// regenerate.
		if textureProxy.Mipmapped() == gpu.MipmappedYes && textureProxy.MipmapsAreDirty() {
			resolveFlags |= ResolveFlagMipMaps
		}
	}

	// Does this proxy have msaa to resolve and/or mipmaps to regenerate?
	if resolveFlags != 0 {
		if b.textureResolveTask == nil {
			b.textureResolveTask = drawingMgr.newTextureResolveRenderTaskBefore()
		}
		b.textureResolveTask.AddProxy(drawingMgr, dependedOn, resolveFlags, caps)
		return
	}

	if textureProxy != nil && textureProxy.isDeferredUpload() {
		if !alreadyDependent {
			b.deferredProxies = append(b.deferredProxies, textureProxy)
		}
	}

	if dependedOnTask != nil {
		b.addDependencyTask(dependedOnTask)
	}
}

// IsUsed reports whether proxy is one of this task's targets, or is otherwise used by it.
func (b *RenderTaskBase) IsUsed(proxy *SurfaceProxy) bool {
	for _, target := range b.targets {
		if target == proxy {
			return true
		}
	}
	return b.self.OnIsUsed(proxy)
}

// IsInstantiated reports whether all targets (and their stencil, if required) are allocated and alive.
func (b *RenderTaskBase) IsInstantiated() bool {
	for _, proxy := range b.targets {
		if !proxy.IsInstantiated() {
			return false
		}
		if proxy.PeekSurface().WasDestroyed() {
			return false
		}
	}
	return true
}

// isDeferredUpload reports whether the proxy's contents are pending a worker-thread upload; always false, since no
// deferred uploader is wired.
func (t *TextureProxy) isDeferredUpload() bool { return false }
