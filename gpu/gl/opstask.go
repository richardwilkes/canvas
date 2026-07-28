// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The render task that records op chains against one render target view and replays them into an OpsRenderPass at
// flush. The op-combining machinery (backward merge in recordOp, forwardCombine, the opChain concat rules) is the
// perf-critical batching engine; the arena parameters are no-ops, since ops are ordinary allocations. There is no audit
// trail.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// Experimentally most combining occurs within the first 10 comparisons.
const (
	maxOpMergeDistance = 10
	maxOpChainDistance = 10
)

func canReorder(a, b geom.Rect) bool { return !rectsOverlap(a, b) }

// StencilContent describes what an OpsTask's render pass should assume about the stencil buffer's initial contents.
type StencilContent int32

// StencilContent values.
const (
	StencilContentDontCare StencilContent = iota
	// StencilContentUserBitsCleared: user bits cleared, clip bit don't-care (the clip bit is always pre-cleared by the
	// GPU backend).
	StencilContentUserBitsCleared
	StencilContentPreserved
)

// CanDiscardPreviousOps reports whether a fullscreen clear may discard the ops recorded so far.
type CanDiscardPreviousOps bool

// CanDiscardPreviousOps values.
const (
	CanDiscardPreviousOpsNo  CanDiscardPreviousOps = false
	CanDiscardPreviousOpsYes CanDiscardPreviousOps = true
)

// opList is an intrusively linked op chain.
type opList struct {
	head Op
	tail Op
}

func makeOpList(op Op) opList { return opList{head: op, tail: op} }

func (l *opList) empty() bool { return l.head == nil }

// popHead removes and returns the head op of the list.
func (l *opList) popHead() Op {
	if l.head == nil {
		panic("popHead of empty list")
	}
	temp := l.head
	l.head = temp.opBase().CutChain()
	if l.head == nil {
		if l.tail != temp {
			panic("list tail out of sync")
		}
		l.tail = nil
	}
	return temp
}

// removeOp removes and returns op from the list, which must be a member of it.
func (l *opList) removeOp(op Op) Op {
	prev := op.opBase().PrevInChain()
	if prev == nil {
		if op != l.head {
			panic("removeOp of non-head with no prev")
		}
		return l.popHead()
	}
	temp := prev.opBase().CutChain()
	if next := temp.opBase().CutChain(); next != nil {
		prev.opBase().ChainConcat(next)
	} else {
		if l.tail != op {
			panic("list tail out of sync")
		}
		l.tail = prev
	}
	return temp
}

// pushHead inserts op, which must not already be chained, at the head of the list.
func (l *opList) pushHead(op Op) {
	if op == nil || !op.opBase().IsChainHead() || !op.opBase().IsChainTail() {
		panic("pushHead requires an unchained op")
	}
	if l.head != nil {
		op.opBase().ChainConcat(l.head)
		l.head = op
	} else {
		l.head = op
		l.tail = op
	}
}

// pushTail appends op, which must be a chain tail, to the end of the list.
func (l *opList) pushTail(op Op) {
	if !op.opBase().IsChainTail() {
		panic("pushTail requires a chain-tail op")
	}
	l.tail.opBase().ChainConcat(op)
	l.tail = l.tail.opBase().NextInChain()
}

// opChain is a group of ops that have been merged or chained together because they share compatible state (processor
// analysis, destination proxy view, applied clip) and can be prepared and executed as a unit. The applied clip is
// stored inline by value (guarded by hasAppliedClip) rather than behind a pointer: the chain owns its clip for its
// whole lifetime, and keeping it in the opChain — which itself lives in the OpsTask's opChains slice — avoids a
// per-clipped-draw heap allocation for the retained copy. appliedClipPtr() hands out a pointer into the slice element
// for the consumers that need one; that pointer stays valid because opChains is never appended to while a chain's clip
// is read (recording finishes before flush, and the combine loops iterate by index without appending).
type opChain struct {
	list              opList
	dstProxyView      DstProxyView
	appliedClip       AppliedClip
	bounds            geom.Rect
	processorAnalysis ProcessorAnalysis
	hasAppliedClip    bool
}

func makeOpChain(op Op, processorAnalysis ProcessorAnalysis, appliedClip *AppliedClip, dstProxyView *DstProxyView) opChain {
	c := opChain{
		list:              makeOpList(op),
		processorAnalysis: processorAnalysis,
	}
	if appliedClip != nil {
		// Copy the caller's transient clip into the chain by value. makeOpChain never stores the appliedClip pointer
		// (it only dereferences it here), so escape analysis keeps the caller's AppliedClip — and the AddDrawOp clip
		// value parameter it points into — on the stack on the common unclipped path.
		c.appliedClip = *appliedClip
		c.hasAppliedClip = true
	}
	if processorAnalysis.RequiresDstTexture() {
		if dstProxyView == nil || dstProxyView.Proxy() == nil {
			panic("dst texture required but no dst proxy view")
		}
		c.dstProxyView = *dstProxyView
	}
	c.bounds = c.list.head.opBase().Bounds()
	return c
}

func (c *opChain) head() Op { return c.list.head }

// appliedClipPtr returns a pointer to the chain's inline applied clip, or nil when the chain has no clip. The pointer
// is only used transiently (op args during flush, tryConcat comparisons) and is never retained past a point where
// opChains might be reallocated, so pointing into the slice element is safe.
func (c *opChain) appliedClipPtr() *AppliedClip {
	if c.hasAppliedClip {
		return &c.appliedClip
	}
	return nil
}

func (c *opChain) shouldExecute() bool { return c.list.head != nil }

// visitProxies calls fn for every proxy read or written by the chain's ops, its dst proxy view (if any), and its
// applied clip (if any).
func (c *opChain) visitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if c.list.empty() {
		return
	}
	for op := c.list.head; op != nil; op = op.opBase().NextInChain() {
		op.VisitProxies(fn)
	}
	if c.dstProxyView.Proxy() != nil {
		fn(c.dstProxyView.Proxy(), gpu.MipmappedNo)
	}
	if c.hasAppliedClip {
		c.appliedClip.VisitProxies(fn)
	}
}

// deleteOps recycles all ops in the chain to their free lists and releases the chain's dst proxy view reference and
// applied clip.
func (c *opChain) deleteOps() {
	for !c.list.empty() {
		// The surviving ops of the chain are now dead: recycle each to its free list. deleteOps is the terminal delete
		// — the drawing manager calls it in EndFlush after OnExecute — so no op recycled here is referenced anywhere
		// afterward.
		c.list.popHead().recycle()
	}
	if c.dstProxyView.Proxy() != nil {
		releaseDstProxy(&c.dstProxyView)
		c.dstProxyView = DstProxyView{}
	}
	// Drop the inline clip's coverage-processor reference. deleteOps truncates opChains but keeps the backing capacity,
	// so without this the retained-capacity slot would pin the chain's FP tree across flushes until a later append
	// overwrites it.
	if c.hasAppliedClip {
		c.appliedClip = AppliedClip{}
		c.hasAppliedClip = false
	}
}

// doConcat concatenates two op chains, attempting to merge ops across them. Assumes the chains are known to be
// chainable. Returns the new chain.
func doConcat(chainA, chainB opList) opList {
	// We process ops in chain b from head to tail. We attempt to merge with nodes in a, starting at chain a's tail and
	// working toward the head. We produce one of the following outcomes:
	// 1) b's head is merged into an op in a.
	// 2) An op from chain a is merged into b's head. (b's head is then processed again.)
	// 3) b's head is popped from chain a and added at the tail of a.
	// After result 3 we don't want to attempt to merge the next head of b with the new tail of a, as we assume merges
	// were already attempted when chain b was created. So we track the original tail of a and start our iteration of a
	// there. We also track the bounds of the nodes appended to chain a that will be skipped for bounds testing. If the
	// original tail of a is merged into an op in b (case 2), the "original tail" advances towards the head of a.
	origATail := chainA.tail
	skipBounds := largestInvertedRect()
	for {
		numMergeChecks := 0
		merged := false
		noSkip := origATail == chainA.tail
		canBackwardMerge := noSkip || canReorder(chainB.head.opBase().Bounds(), skipBounds)
		forwardMergeBounds := skipBounds
		a := origATail
		for a != nil {
			canForwardMerge := a == chainA.tail ||
				canReorder(a.opBase().Bounds(), forwardMergeBounds)
			if canForwardMerge || canBackwardMerge {
				result := a.opBase().CombineIfPossible(chainB.head)
				if result == CombineResultCannotCombine {
					panic("chained ops must be able to combine")
				}
				merged = result == CombineResultMerged
			}
			if merged {
				if canBackwardMerge {
					// The merged op is dropped: its instances were copied into the surviving op by CombineIfPossible,
					// so it is dead — recycle it to its free list.
					chainB.popHead().recycle()
				} else {
					// We merged the contents of b's head into a. Replace b's head with a in chain b.
					if !canForwardMerge {
						panic("merge without forward-merge permission")
					}
					if a == origATail {
						origATail = a.opBase().PrevInChain()
					}
					detachedA := chainA.removeOp(a)
					// b's head was absorbed into a (which now replaces it at chain b's head), so b's old head is dead —
					// recycle it to its free list.
					chainB.popHead().recycle()
					chainB.pushHead(detachedA)
					if chainA.empty() {
						// We merged all the nodes in chain a to chain b.
						return chainB
					}
				}
				break
			}
			numMergeChecks++
			if numMergeChecks == maxOpMergeDistance {
				break
			}
			forwardMergeBounds = joinNonEmptyArgAllowingEmpty(forwardMergeBounds,
				a.opBase().Bounds())
			canBackwardMerge = canBackwardMerge &&
				canReorder(chainB.head.opBase().Bounds(), a.opBase().Bounds())
			a = a.opBase().PrevInChain()
		}
		// If we weren't able to merge b's head then pop it from chain b and make it the new tail of a.
		if !merged {
			chainA.pushTail(chainB.popHead())
			skipBounds = joinNonEmptyArgAllowingEmpty(skipBounds, chainA.tail.opBase().Bounds())
		}
		if chainB.empty() {
			return chainA
		}
	}
}

// joinNonEmptyArgAllowingEmpty is joinNonEmptyArg without the non-empty assertion: op bounds may legitimately be empty
// (zero-area draws), and the join simply proceeds in that case.
func joinNonEmptyArgAllowingEmpty(r, other geom.Rect) geom.Rect {
	if r.Left >= r.Right || r.Top >= r.Bottom {
		return other
	}
	return joinPossiblyEmptyRect(r, other)
}

// tryConcat attempts to concatenate the given list onto our own and merge ops across the chains. On success the
// provided list is emptied.
func (c *opChain) tryConcat(list *opList, processorAnalysis ProcessorAnalysis, dstProxyView *DstProxyView, appliedClip *AppliedClip, bounds geom.Rect) bool {
	if c.list.empty() || list.empty() {
		panic("tryConcat requires non-empty chains")
	}
	if c.processorAnalysis.RequiresDstTexture() != (c.dstProxyView.Proxy() != nil) ||
		processorAnalysis.RequiresDstTexture() != (dstProxyView.Proxy() != nil) {
		panic("dst-texture analysis out of sync with dst proxy views")
	}
	if c.list.head.opBase().ClassID() != list.head.opBase().ClassID() ||
		c.hasAppliedClip != (appliedClip != nil) ||
		(c.hasAppliedClip && !c.appliedClip.Equal(appliedClip)) ||
		(c.processorAnalysis.RequiresNonOverlappingDraws() !=
			processorAnalysis.RequiresNonOverlappingDraws()) ||
		(c.processorAnalysis.RequiresNonOverlappingDraws() &&
			// Non-overlapping draws are only required when the backend will either insert a barrier or read back a new
			// dst texture between draws. In either case, we can neither chain nor combine overlapping ops.
			rectsTouchOrOverlap(c.bounds, bounds)) ||
		(c.processorAnalysis.RequiresDstTexture() != processorAnalysis.RequiresDstTexture()) ||
		(c.processorAnalysis.RequiresDstTexture() && !c.dstProxyView.Equal(dstProxyView)) {
		return false
	}

	first := true
	for !list.empty() {
		switch c.list.tail.opBase().CombineIfPossible(list.head) {
		case CombineResultCannotCombine:
			// If an op supports chaining then chaining must be transitive, and if any two ops in two different chains
			// can merge then the two chains may also be chained together. Thus, this should only hit on the first
			// iteration.
			if !first {
				panic("chaining must be transitive")
			}
			return false
		case CombineResultMayChain:
			taken := *list
			*list = opList{}
			c.list = doConcat(c.list, taken)
			if !list.empty() {
				panic("doConcat must consume the list")
			}
		case CombineResultMerged:
			// The incoming op was merged into our tail; its instances were copied out, so it is dead — recycle it to
			// its free list.
			list.popHead().recycle()
		}
		first = false
	}

	// The new ops were successfully merged and/or chained onto our own.
	c.bounds = joinPossiblyEmptyRect(c.bounds, bounds)
	return true
}

// prependChain attempts to move the ops from the passed chain into this chain at the head. On success the passed chain
// is empty.
func (c *opChain) prependChain(that *opChain) bool {
	if !that.tryConcat(&c.list, c.processorAnalysis, &c.dstProxyView, c.appliedClipPtr(), c.bounds) {
		// Append failed.
		return false
	}
	// 'that' owns the combined chain. Move it into 'this'.
	if !c.list.empty() {
		panic("tryConcat should have consumed the list")
	}
	c.list = that.list
	that.list = opList{}
	c.bounds = that.bounds

	// Drop the reference 'that' held on the (shared) dst proxy.
	releaseDstProxy(&that.dstProxyView)
	that.dstProxyView.SetProxyView(SurfaceProxyView{})
	if that.hasAppliedClip && that.appliedClip.HasCoverageFragmentProcessor() {
		// Obliterates the processor.
		that.appliedClip.DetachCoverageFragmentProcessor()
	}
	return true
}

// appendOp attempts to add op to this chain by merging or adding to the tail. Returns the op back to the caller on
// failure, else nil.
func (c *opChain) appendOp(op Op, processorAnalysis ProcessorAnalysis, dstProxyView *DstProxyView, appliedClip *AppliedClip) Op {
	var noDstProxyView DstProxyView
	if dstProxyView == nil {
		dstProxyView = &noDstProxyView
	}
	if !op.opBase().IsChainHead() || !op.opBase().IsChainTail() {
		panic("appendOp requires an unchained op")
	}
	opBounds := op.opBase().Bounds()
	list := makeOpList(op)
	if !c.tryConcat(&list, processorAnalysis, dstProxyView, appliedClip, opBounds) {
		// Append failed, give the op back to the caller.
		return list.popHead()
	}
	if !list.empty() {
		panic("tryConcat should have consumed the list")
	}
	return nil
}

// OpsTask is the render task that records op chains against one render target view and replays them into an
// OpsRenderPass at flush.
type OpsTask struct {
	sampledDepFn func(*SurfaceProxy, gpu.Mipmapped)
	visitAlloc   *ResourceAllocator
	arenas       *Arenas
	// opChainsBox owns the opChains backing array between frames: NewOpsTask borrows it (starting opChains from its
	// backing) and onDelete returns it to the pool so the next frame's task reuses the grown array instead of
	// re-growing from nil (see opchainspool.go).
	opChainsBox *opChainsBox
	gatherFn    func(*SurfaceProxy, gpu.Mipmapped)
	visitCaps   *Caps
	depFn       func(*SurfaceProxy, gpu.Mipmapped)
	// Cached proxy-dependency visitor closures, created once per task and reused by every AddDrawOp/AddOp call, so
	// recording a draw allocates no per-call visitor. Because visitProxies is an interface method, a closure literal
	// passed to it would normally escape to the heap on every call; caching the closure on the task avoids that. The
	// per-call drawing manager and caps are stashed in the fields below before each visit — safe because recording is
	// single-threaded per context and visitProxies runs synchronously. sampledDepFn also registers the proxy as a
	// sampled texture; depFn only adds the task dependency (AddOp's non-draw ops are not sampled).
	visitDrawingMgr *DrawingManager
	opChains        []opChain
	sampledProxies  []*SurfaceProxy
	RenderTaskBase
	lastClipNumAnalyticElements int
	clippedContentBounds        geom.IRect
	lastDevClipBounds           geom.IRect
	loadClearColor              [4]float32
	totalBounds                 geom.Rect
	initialStencilContent       StencilContent
	renderPassXferBarriers      gpu.XferBarrierFlags
	lastClipStackGenID          uint32
	colorLoadOp                 gpu.LoadOp
	targetOrigin                gpu.SurfaceOrigin
	targetSwizzle               gpu.Swizzle
	cannotMergeBackward         bool
	mustPreserveStencil         bool
	usesMSAASurface             bool
}

// NewOpsTask creates an OpsTask targeting view; the task takes its own reference on the view's proxy and shares the
// arenas whose lifetime it maintains. The task object itself comes from the free list (opstaskpool.go, zero-valued on
// borrow), so a steady-state frame allocates neither the struct nor its visitor closures.
func NewOpsTask(drawingMgr *DrawingManager, view SurfaceProxyView, arenas *Arenas) *OpsTask {
	t := borrowOpsTask()
	t.usesMSAASurface = view.AsRenderTargetProxy().NumSamples() > 1
	t.targetSwizzle = view.Swizzle()
	t.targetOrigin = view.Origin()
	t.colorLoadOp = gpu.LoadOpLoad
	t.arenas = arenas
	// Start the opChains from a pooled backing so recordOp's append reuses a prior frame's grown array instead of
	// re-growing from nil. onDelete returns the box to the pool.
	t.opChainsBox = borrowOpChainsBox()
	t.opChains = t.opChainsBox.chains[:0]
	t.initRenderTask(t)
	t.addTarget(drawingMgr, view.Proxy())
	return t
}

// Name implements RenderTask.
func (t *OpsTask) Name() string { return "Ops" }

// AsOpsTask implements RenderTask.
func (t *OpsTask) AsOpsTask() *OpsTask { return t }

// onDelete implements the destructor hook, run by the final Unref (refCnt==0). It recycles the task's ops and then
// returns the opChains backing to the pool for the next frame's task. The task is provably dead here — disowned, and
// (after deleteOps) its ops recycled — so the backing is uniquely owned and safe to reuse (see opchainspool.go).
func (t *OpsTask) onDelete() {
	t.deleteOps()
	if t.opChainsBox != nil {
		// Hand the current array (grown past the borrowed backing if the frame added enough chains) to the box so the
		// larger capacity is preserved for the next borrow.
		t.opChainsBox.chains = t.opChains
		recycleOpChainsBox(t.opChainsBox)
		t.opChainsBox = nil
	}
	t.opChains = nil
}

// IsEmpty reports whether the task has no recorded op chains.
func (t *OpsTask) IsEmpty() bool { return len(t.opChains) == 0 }

// UsesMSAASurface reports whether any op recorded on this task requires an MSAA surface.
func (t *OpsTask) UsesMSAASurface() bool { return t.usesMSAASurface }

// RenderPassXferBarriers returns the transfer barrier flags required for this task's render pass.
func (t *OpsTask) RenderPassXferBarriers() gpu.XferBarrierFlags { return t.renderPassXferBarriers }

// isColorNoOp reports whether executing this task would have no visible effect on the color buffer.
func (t *OpsTask) isColorNoOp() bool {
	// TODO: a stored discard should also be grounds for skipping execution.
	return len(t.opChains) == 0 && t.colorLoadOp == gpu.LoadOpLoad
}

func (t *OpsTask) deleteOps() {
	for i := range t.opChains {
		t.opChains[i].deleteOps()
	}
	t.opChains = t.opChains[:0]
}

// AddSampledTexture records proxy as sampled by this task. Sampling implicitly requires the proxy be a texture.
func (t *OpsTask) AddSampledTexture(proxy *SurfaceProxy) {
	if proxy.AsTextureProxy() == nil {
		panic("sampled proxy must be a texture")
	}
	t.sampledProxies = append(t.sampledProxies, proxy)
}

// sampledDepVisitor returns the cached closure that registers a visited proxy as both a sampled texture and a task
// dependency, stashing the per-call drawing manager and caps for it to read.
func (t *OpsTask) sampledDepVisitor(drawingMgr *DrawingManager, caps *Caps) func(*SurfaceProxy, gpu.Mipmapped) {
	t.visitDrawingMgr = drawingMgr
	t.visitCaps = caps
	if t.sampledDepFn == nil {
		t.sampledDepFn = func(p *SurfaceProxy, mipmapped gpu.Mipmapped) {
			t.AddSampledTexture(p)
			t.AddDependency(t.visitDrawingMgr, p, mipmapped, t.visitCaps)
		}
	}
	return t.sampledDepFn
}

// depVisitor returns the cached closure that registers a visited proxy as a task dependency only (no sampled-texture
// registration), stashing the per-call drawing manager and caps for it to read.
func (t *OpsTask) depVisitor(drawingMgr *DrawingManager, caps *Caps) func(*SurfaceProxy, gpu.Mipmapped) {
	t.visitDrawingMgr = drawingMgr
	t.visitCaps = caps
	if t.depFn == nil {
		t.depFn = func(p *SurfaceProxy, mipmapped gpu.Mipmapped) {
			t.AddDependency(t.visitDrawingMgr, p, mipmapped, t.visitCaps)
		}
	}
	return t.depFn
}

// AddOp records a non-draw op onto the task, wiring up its proxy dependencies.
func (t *OpsTask) AddOp(drawingMgr *DrawingManager, op Op, caps *Caps) {
	op.VisitProxies(t.depVisitor(drawingMgr, caps))
	t.recordOp(op, false, EmptyProcessorSetAnalysis(), nil, nil)
}

// AddDrawOp records a draw op onto the task, wiring up its proxy dependencies (including the applied clip and
// destination-read proxy, if any) and any required transfer barriers.
func (t *OpsTask) AddDrawOp(drawingMgr *DrawingManager, op Op, usesMSAA bool, processorAnalysis ProcessorAnalysis, clip AppliedClip, dstProxyView *DstProxyView, caps *Caps) {
	addDependency := t.sampledDepVisitor(drawingMgr, caps)
	op.VisitProxies(addDependency)
	clip.VisitProxies(addDependency)
	if dstProxyView.Proxy() != nil {
		t.AddSampledTexture(dstProxyView.Proxy())
		if dstProxyView.DstSampleFlags()&gpu.DstSampleFlagRequiresTextureBarrier != 0 {
			t.renderPassXferBarriers |= gpu.XferBarrierFlagTexture
		}
		addDependency(dstProxyView.Proxy(), gpu.MipmappedNo)
	}
	if processorAnalysis.UsesNonCoherentHWBlending() {
		t.renderPassXferBarriers |= gpu.XferBarrierFlagBlend
	}
	var clipPtr *AppliedClip
	if clip.DoesClip() {
		clipPtr = &clip
	}
	t.recordOp(op, usesMSAA, processorAnalysis, clipPtr, dstProxyView)
}

// EndFlush implements RenderTask: empties the queued-up draws. The proxy lists are element-cleared but keep their
// backing capacity — the task either lives on (a surface context's retained opsTask, which will re-append next frame)
// or is about to be recycled through the free list, and both reuse the arrays.
func (t *OpsTask) EndFlush(drawingMgr *DrawingManager) {
	t.lastClipStackGenID = 0
	t.deleteOps()
	clear(t.deferredProxies)
	t.deferredProxies = t.deferredProxies[:0]
	clear(t.sampledProxies)
	t.sampledProxies = t.sampledProxies[:0]
	t.RenderTaskBase.EndFlush(drawingMgr)
}

// OnPrepare implements RenderTask: prepares every op chain that will execute this flush.
func (t *OpsTask) OnPrepare(flushState *OpFlushState) {
	if t.Target(0).PeekRenderTarget() == nil {
		panic("prepare without an instantiated render target")
	}
	if !t.IsClosed() {
		panic("prepare on an open task")
	}
	// Loop over the ops that haven't yet been prepared.
	if t.isColorNoOp() ||
		(t.clippedContentBounds.IsEmpty() && t.colorLoadOp != gpu.LoadOpDiscard) {
		return
	}
	flushState.SetSampledProxyArray(&t.sampledProxies)
	dstView := MakeSurfaceProxyView(t.Target(0), t.targetOrigin, t.targetSwizzle)
	for i := range t.opChains {
		chain := &t.opChains[i]
		if chain.shouldExecute() {
			opArgs := OpArgs{
				op:                     chain.head(),
				surfaceView:            dstView,
				usesMSAASurface:        t.usesMSAASurface,
				appliedClip:            chain.appliedClipPtr(),
				dstProxyView:           chain.dstProxyView,
				renderPassXferBarriers: t.renderPassXferBarriers,
				colorLoadOp:            t.colorLoadOp,
			}
			flushState.SetOpArgs(&opArgs)
			prepareOp(chain.head(), flushState)
			flushState.SetOpArgs(nil)
		}
	}
	flushState.SetSampledProxyArray(nil)
}

// OnExecute implements RenderTask: creates the render pass and draws all the generated geometry. Returns whether any
// commands were issued to the GPU.
func (t *OpsTask) OnExecute(flushState *OpFlushState) bool {
	if t.NumTargets() != 1 {
		panic("OpsTask must have exactly one target")
	}
	proxy := t.Target(0).AsRenderTargetProxy()
	if proxy == nil {
		panic("OpsTask target must be a render target")
	}
	defer proxy.ClearArenas()

	if t.isColorNoOp() || t.clippedContentBounds.IsEmpty() {
		return false
	}

	// Make sure load ops are not kClear if the GPU needs to use draws for clears.
	if t.colorLoadOp == gpu.LoadOpClear && flushState.Caps().PerformColorClearsAsDraws {
		panic("load-op clear with performColorClearsAsDraws")
	}
	caps := flushState.Caps()
	renderTarget := proxy.Proxy().PeekRenderTarget()
	if renderTarget == nil {
		panic("execute without an instantiated render target")
	}

	var stencil *Attachment
	if proxy.NeedsStencil() {
		if !proxy.CanUseStencil(caps) {
			panic("needsStencil on a target that can't use stencil")
		}
		if !flushState.ResourceProvider().AttachStencilAttachment(renderTarget,
			t.usesMSAASurface) {
			// Failed to attach a stencil buffer; rendering is skipped.
			return false
		}
		stencil = renderTarget.StencilAttachment(t.usesMSAASurface)
	}

	var stencilLoadOp gpu.LoadOp
	switch t.initialStencilContent {
	case StencilContentDontCare:
		stencilLoadOp = gpu.LoadOpDiscard
	case StencilContentUserBitsCleared:
		if caps.PerformStencilClearsAsDraws {
			panic("kUserBitsCleared with performStencilClearsAsDraws")
		}
		if stencil == nil {
			panic("kUserBitsCleared without a stencil attachment")
		}
		// The request is always honored with a real clear. Upstream skips it after the attachment's first clear, on the
		// stated invariant that a SurfaceDrawContext leaves the user stencil bits cleared once finished; that invariant
		// does not hold here. Stencil attachments are shared between render targets under a unique key carrying only
		// (caps, format, dims, usage, sampleCnt) and no render-target identity (ComputeSharedAttachmentUniqueKey), and
		// a finished SurfaceDrawContext leaves both user-bit residue (the clip mask helper's cover passes only zero
		// what they rasterize) and clip-bit residue (ClearStencilClip only ever clears within a scissor) behind. Under
		// DMSAA — where the shared, library-allocated MSAA stencil attachment is the one in play rather than a wrapped
		// FBO's own per-surface stencil — reloading that residue paints stray bands into a later render target that
		// carries no clip at all. Skipping the clear could save at most one glClear per SurfaceDrawContext anyway:
		// setNeedsStencil requests kUserBitsCleared once per context, and split tasks ask for kPreserved.
		stencilLoadOp = gpu.LoadOpClear
	case StencilContentPreserved:
		if stencil == nil {
			panic("kPreserved without a stencil attachment")
		}
		stencilLoadOp = gpu.LoadOpLoad
	}

	// If mustPreserveStencil is set we are executing a SurfaceDrawContext that split its opsTask.
	stencilStoreOp := gpu.StoreOpStore
	if caps.DiscardStencilValuesAfterRenderPass() && !t.mustPreserveStencil {
		stencilStoreOp = gpu.StoreOpDiscard
	}

	renderPass := flushState.Gpu().GetOpsRenderPass(flushState.ResourceProvider(),
		renderTarget, t.usesMSAASurface, stencil,
		t.targetOrigin, t.clippedContentBounds,
		LoadAndStoreInfo{
			LoadOp: t.colorLoadOp, StoreOp: gpu.StoreOpStore,
			ClearColor: t.loadClearColor,
		},
		StencilLoadAndStoreInfo{LoadOp: stencilLoadOp, StoreOp: stencilStoreOp},
		t.sampledProxies, t.renderPassXferBarriers)
	if renderPass == nil {
		return false
	}
	flushState.SetOpsRenderPass(renderPass)
	renderPass.Begin()

	dstView := MakeSurfaceProxyView(t.Target(0), t.targetOrigin, t.targetSwizzle)

	// Draw all the generated geometry.
	for i := range t.opChains {
		chain := &t.opChains[i]
		if !chain.shouldExecute() {
			continue
		}
		opArgs := OpArgs{
			op:                     chain.head(),
			surfaceView:            dstView,
			usesMSAASurface:        t.usesMSAASurface,
			appliedClip:            chain.appliedClipPtr(),
			dstProxyView:           chain.dstProxyView,
			renderPassXferBarriers: t.renderPassXferBarriers,
			colorLoadOp:            t.colorLoadOp,
		}
		flushState.SetOpArgs(&opArgs)
		executeOp(chain.head(), flushState, chain.bounds)
		flushState.SetOpArgs(nil)
	}

	renderPass.End()
	flushState.Gpu().Submit(renderPass)
	flushState.SetOpsRenderPass(nil)
	return true
}

// SetColorLoadOp sets how the color attachment should be loaded when this task's render pass begins. Must only be
// called if native color buffer clearing is enabled.
func (t *OpsTask) SetColorLoadOp(op gpu.LoadOp, color [4]float32) {
	t.colorLoadOp = op
	t.loadClearColor = color
	if op == gpu.LoadOpClear {
		proxy := t.Target(0)
		t.totalBounds = proxy.BackingStoreBoundsIRect().ToRect()
	}
}

// CanMerge reports whether the given opsTask can be appended at the end of this one.
func (t *OpsTask) CanMerge(opsTask *OpsTask) bool {
	return t.Target(0) == opsTask.Target(0) &&
		t.arenas == opsTask.arenas &&
		!opsTask.cannotMergeBackward
}

// MergeFrom merges as many opsTasks as possible from the head of tasks; they should all be render-pass compatible.
// Returns the number of tasks merged into this one.
func (t *OpsTask) MergeFrom(tasks []RenderTask) int {
	mergedCount := 0
	for _, task := range tasks {
		opsTask := task.AsOpsTask()
		if opsTask == nil || !t.CanMerge(opsTask) {
			break
		}
		if t.targetSwizzle != opsTask.targetSwizzle || t.targetOrigin != opsTask.targetOrigin {
			panic("mergeable tasks with mismatched views")
		}
		if opsTask.colorLoadOp == gpu.LoadOpClear {
			// TODO: go back to actually dropping ops tasks when merged with a color clear.
			return 0
		}
		mergedCount++
	}
	if mergedCount == 0 {
		return 0
	}

	mergingNodes := tasks[:mergedCount]
	for _, task := range mergingNodes {
		toMerge := task.AsOpsTask()
		t.clippedContentBounds.Join(toMerge.clippedContentBounds)
		t.totalBounds.Join(toMerge.totalBounds)
		t.renderPassXferBarriers |= toMerge.renderPassXferBarriers
		if t.initialStencilContent == StencilContentDontCare {
			// Propagate the first stencil content that isn't StencilContentDontCare: once it has any kind of initial
			// content that isn't don't-care, the initial contents of subsequent merged opsTasks don't matter. (This
			// works because the opsTasks all target the same render target in painter's order — the preserved case
			// happens automatically with a merge, and the cleared case is automatic because ops leave the stencil in a
			// cleared state when finished.)
			t.initialStencilContent = toMerge.initialStencilContent
		}
		t.usesMSAASurface = t.usesMSAASurface || toMerge.usesMSAASurface
	}

	t.lastClipStackGenID = 0
	for _, task := range mergingNodes {
		toMerge := task.AsOpsTask()
		for _, renderTask := range toMerge.Dependents() {
			renderTask.taskBase().ReplaceDependency(toMerge, t)
		}
		for _, renderTask := range toMerge.Dependencies() {
			renderTask.taskBase().ReplaceDependent(toMerge, t)
		}
		t.deferredProxies = append(t.deferredProxies, toMerge.deferredProxies...)
		t.sampledProxies = append(t.sampledProxies, toMerge.sampledProxies...)
		t.opChains = append(t.opChains, toMerge.opChains...)
		toMerge.deferredProxies = nil
		toMerge.sampledProxies = nil
		toMerge.opChains = nil
	}
	t.mustPreserveStencil = mergingNodes[len(mergingNodes)-1].AsOpsTask().mustPreserveStencil
	return mergedCount
}

// ResetForFullscreenClear does the bookkeeping for a fullscreen clear regardless of how the clear is implemented later.
// Returns true if the clear can be converted into a load op (barring device caps).
func (t *OpsTask) ResetForFullscreenClear(canDiscardPreviousOps CanDiscardPreviousOps) bool {
	if canDiscardPreviousOps == CanDiscardPreviousOpsYes || t.IsEmpty() {
		t.deleteOps()
		t.deferredProxies = nil
		t.sampledProxies = nil
		return true
	}
	// Could not empty the task, so an op must be added to handle the clear.
	return false
}

// Discard marks an empty task as fully discarded, updating the color & stencil load ops so nothing is loaded when it
// next executes. Calls to non-empty (in-progress) tasks are ignored.
func (t *OpsTask) Discard() {
	if t.IsEmpty() {
		t.colorLoadOp = gpu.LoadOpDiscard
		t.initialStencilContent = StencilContentDontCare
		t.totalBounds = geom.Rect{}
	}
}

// SetInitialStencilContent sets what this task's render pass should assume about the stencil buffer's initial contents.
func (t *OpsTask) SetInitialStencilContent(initialContent StencilContent) {
	t.initialStencilContent = initialContent
}

// SetMustPreserveStencil marks that this task's stencil contents must be preserved after execution rather than
// discarded.
func (t *OpsTask) SetMustPreserveStencil() { t.mustPreserveStencil = true }

// SetCannotMergeBackward marks that this task may not be merged backward into an earlier one. CanMerge and MergeFrom
// test the flag on the later of the two tasks, so it belongs on the newly created replacement task, not on the task
// being protected from it.
func (t *OpsTask) SetCannotMergeBackward() { t.cannotMergeBackward = true }

// OnMakeSkippable implements RenderTask: clears the task down to a color no-op so it can be skipped entirely.
func (t *OpsTask) OnMakeSkippable() {
	t.deleteOps()
	t.deferredProxies = nil
	t.colorLoadOp = gpu.LoadOpLoad
	if !t.isColorNoOp() {
		panic("skippable task should be a color no-op")
	}
}

// OnIsUsed implements RenderTask: reports whether proxyToCheck is among this task's sampled proxies.
func (t *OpsTask) OnIsUsed(proxyToCheck *SurfaceProxy) bool {
	for _, proxy := range t.sampledProxies {
		if proxy == proxyToCheck {
			return true
		}
	}
	return false
}

// GatherProxyIntervals implements RenderTask: registers this task's usage intervals for every proxy it reads or writes
// with the resource allocator.
func (t *OpsTask) GatherProxyIntervals(alloc *ResourceAllocator) {
	if !t.IsClosed() {
		panic("gathering intervals on an open task")
	}
	if t.isColorNoOp() {
		return
	}

	for _, deferred := range t.deferredProxies {
		if deferred.Proxy().IsInstantiated() {
			panic("deferred proxy is already instantiated")
		}
		// All the deferred proxies get a write usage at the very start of flushing, locking them out of reuse for the
		// entire flush until they are read (and then they can be recycled).
		alloc.AddInterval(deferred.Proxy(), 0, 0, ActualUseNo, AllowRecyclingYes)
	}

	targetProxy := t.Target(0)
	if len(t.opChains) > 0 {
		cur := alloc.CurOp()
		alloc.AddInterval(targetProxy, cur, cur+uint32(len(t.opChains))-1, ActualUseYes,
			AllowRecyclingYes)
	} else {
		// This can happen if there is a loadOp (e.g., a clear) but no other draws: we still need an interval for the
		// destination, so create a fake op# for the missing clear op.
		alloc.AddInterval(targetProxy, alloc.CurOp(), alloc.CurOp(), ActualUseYes,
			AllowRecyclingYes)
		alloc.IncOps()
	}

	// The gather closure is cached on the task like sampledDepFn/depFn (see sampledDepVisitor): capturing alloc
	// directly would heap-allocate a fresh closure every flush, so the per-call allocator is stashed in visitAlloc for
	// the cached closure to read (recording is single-threaded per context) and cleared after the loop.
	t.visitAlloc = alloc
	if t.gatherFn == nil {
		t.gatherFn = func(p *SurfaceProxy, _ gpu.Mipmapped) {
			t.visitAlloc.AddInterval(p, t.visitAlloc.CurOp(), t.visitAlloc.CurOp(), ActualUseYes,
				AllowRecyclingYes)
		}
	}
	for i := range t.opChains {
		t.opChains[i].visitProxies(t.gatherFn)
		// Even though the op may have been (re)moved we still need to increment the op count to keep all the math
		// consistent.
		alloc.IncOps()
	}
	t.visitAlloc = nil
}

// recordOp records op onto the task, attempting to merge or chain it into an existing op chain before starting a new
// one.
func (t *OpsTask) recordOp(op Op, usesMSAA bool, processorAnalysis ProcessorAnalysis, clip *AppliedClip, dstProxyView *DstProxyView) {
	proxy := t.Target(0)
	if processorAnalysis.RequiresDstTexture() != (dstProxyView != nil &&
		dstProxyView.Proxy() != nil) {
		panic("dst-texture analysis out of sync with dst proxy view")
	}
	if proxy == nil {
		panic("recordOp without a target")
	}
	if t.IsClosed() {
		panic("a closed OpsTask should never receive new ops")
	}
	// usesMSAA on a single-sample target is the DMSAA promotion: flush selects the render target's dynamic MSAA FBO
	// (GetOpsRenderPass → EnsureDynamicMSAAAttachment).

	if !op.opBase().Bounds().IsFinite() {
		return
	}

	t.usesMSAASurface = t.usesMSAASurface || usesMSAA

	// Account for this op's bounds before we attempt to combine. The caller should have already called
	// setClippedBounds() by now, if applicable.
	t.totalBounds.Join(op.opBase().Bounds())

	// Check if there is an op we can combine with by linearly searching back until we either 1) check every op, 2)
	// intersect with something, or 3) find a blocker.
	maxCandidates := min(maxOpChainDistance, len(t.opChains))
	if maxCandidates > 0 {
		i := 0
		for {
			candidate := &t.opChains[len(t.opChains)-1-i]
			op = candidate.appendOp(op, processorAnalysis, dstProxyView, clip)
			if op == nil {
				// Merged into the candidate (which requires an equal dst proxy view); the incoming view's reference is
				// released now that it's no longer needed.
				if dstProxyView != nil {
					releaseDstProxy(dstProxyView)
				}
				return
			}
			// Stop going backwards if we would cause a painter's order violation.
			if !canReorder(candidate.bounds, op.opBase().Bounds()) {
				break
			}
			i++
			if i == maxCandidates {
				break
			}
		}
	}
	// makeOpChain copies the caller's transient clip into the new chain by value — the chain owns its clip for its
	// whole lifetime, stored inline in the opChains slice — so no separate heap copy is needed here. Passing the clip
	// parameter straight through stays alloc-free on the common unclipped path: makeOpChain only dereferences the
	// pointer (never stores it), so escape analysis keeps the caller's AppliedClip and the AddDrawOp clip value
	// parameter on the stack.
	t.opChains = append(t.opChains, makeOpChain(op, processorAnalysis, clip, dstProxyView))
}

// forwardCombine walks the recorded op chains and attempts to prepend later chains onto earlier ones, merging and
// chaining ops that were recorded too far apart for recordOp's backward search to find.
func (t *OpsTask) forwardCombine() {
	if t.IsClosed() {
		panic("forwardCombine on a closed task")
	}
	for i := 0; i < len(t.opChains)-1; i++ {
		chain := &t.opChains[i]
		maxCandidateIdx := min(i+maxOpChainDistance, len(t.opChains)-1)
		j := i + 1
		for {
			candidate := &t.opChains[j]
			if candidate.prependChain(chain) {
				break
			}
			// Stop traversing if we would cause a painter's order violation.
			if !canReorder(chain.bounds, candidate.bounds) {
				break
			}
			j++
			if j > maxCandidateIdx {
				break
			}
		}
	}
}

// OnMakeClosed implements RenderTask: runs forwardCombine and computes the clipped content bounds before the task is
// sealed against further recording.
func (t *OpsTask) OnMakeClosed(targetUpdateBounds *geom.IRect) ExpectedOutcome {
	t.forwardCombine()
	if !t.isColorNoOp() {
		proxy := t.Target(0)
		// Use the entire backing store bounds since the GPU doesn't clip automatically to the logical dimensions.
		clippedContentBounds := proxy.BackingStoreBoundsIRect().ToRect()
		if clippedContentBounds.Intersect(t.totalBounds) {
			t.clippedContentBounds = clippedContentBounds.RoundOut()
			*targetUpdateBounds = makeNativeIRect(t.targetOrigin,
				proxy.BackingStoreDimensions().Height, t.clippedContentBounds)
			return ExpectedOutcomeTargetDirty
		}
	}
	return ExpectedOutcomeTargetUnchanged
}

// makeNativeIRect converts a top-down device rect into the surface's native coordinates (flipping for bottom-left
// origins).
func makeNativeIRect(origin gpu.SurfaceOrigin, surfaceHeight int32, devRect geom.IRect) geom.IRect {
	if origin == gpu.OriginBottomLeft {
		y := surfaceHeight - devRect.Top - devRect.Height()
		return geom.IRectXYWH(devRect.Left, y, devRect.Width(), devRect.Height())
	}
	return devRect
}

// NumOpChains returns the number of recorded op chains (test access).
func (t *OpsTask) NumOpChains() int { return len(t.opChains) }

// GetChainHead returns the head op of the op chain at index (test access).
func (t *OpsTask) GetChainHead(index int) Op { return t.opChains[index].head() }
