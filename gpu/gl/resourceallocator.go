// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The explicit distribution of GPU surfaces to proxies at flush. It is fed the usage intervals of the proxies (op
// indices come from the order of ops after the task DAG is linearized), plans an assignment of "registers" (future
// surfaces) with scratch reuse across disjoint intervals, optionally checks the budget, and finally instantiates and
// assigns real surfaces. Instantiation failure handling: read-only lazy proxies instantiate during AddInterval,
// fully-lazy ones during PlanAssignment, and everything else during Assign; the drawing manager drops the flush if
// anything fails. Intervals and registers use ordinary allocation (they are outside the recording-arena work), so Reset
// explicitly drops the register-held surface refs rather than relying on arena destruction.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/gpu"
)

// ActualUse records whether an AddInterval call represents an actual usage of the proxy. Deferred proxies attached to
// opsTasks get an extra-long interval for the upload that shouldn't count toward recyclability.
type ActualUse bool

// ActualUse values.
const (
	ActualUseNo  ActualUse = false
	ActualUseYes ActualUse = true
)

// AllowRecycling controls whether a proxy's backing surface may be recycled to another proxy (some render-pass
// configurations forbid it, so tasks pass this through explicitly).
type AllowRecycling bool

// AllowRecycling values.
const (
	AllowRecyclingNo  AllowRecycling = false
	AllowRecyclingYes AllowRecycling = true
)

// allocRegister is a future surface that will back each proxy assigned to it.
type allocRegister struct {
	originatingProxy *SurfaceProxy
	// existingSurface is queried from the resource cache (ref held); may be nil.
	existingSurface      *Surface
	scratchKey           gpu.ScratchKey
	accountedForInBudget bool
}

// canProxyUseScratch reports whether proxy may reuse a scratch resource; some cases always require a fresh texture.
func canProxyUseScratch(caps *Caps, proxy *SurfaceProxy) bool {
	return caps.ReuseScratchTextures || proxy.AsRenderTargetProxy() != nil
}

// newAllocRegister builds a register for originatingProxy, looking for an existing surface to reuse by scratch or
// unique key. An invalid scratch key is OK iff the proxy has a unique key.
func newAllocRegister(originatingProxy *SurfaceProxy, scratchKey gpu.ScratchKey, provider *ResourceProvider) *allocRegister {
	if originatingProxy.IsInstantiated() || originatingProxy.IsLazy() {
		panic("register for an instantiated or lazy proxy")
	}
	r := &allocRegister{originatingProxy: originatingProxy, scratchKey: scratchKey}
	if scratchKey.IsValid() {
		if canProxyUseScratch(provider.Caps(), originatingProxy) {
			r.existingSurface = provider.FindAndRefScratchTextureByKey(&r.scratchKey,
				"ResourceAllocatorRegister")
		}
	} else {
		if !r.uniqueKey().IsValid() {
			panic("register without scratch or unique key")
		}
		if res := provider.FindByUniqueKey(r.uniqueKey()); res != nil {
			r.existingSurface = res.(*Surface)
		}
	}
	return r
}

func (r *allocRegister) uniqueKey() *gpu.UniqueKey { return r.originatingProxy.UniqueKey() }

// isRecyclable reports whether this register can be used by other proxies after this one.
func (r *allocRegister) isRecyclable(caps *Caps, proxy *SurfaceProxy, knownUseCount int, allowRecycling AllowRecycling) bool {
	if allowRecycling == AllowRecyclingNo {
		return false
	}
	if !canProxyUseScratch(caps, proxy) {
		return false
	}
	if !r.scratchKey.IsValid() {
		return false // no scratch key, no free pool
	}
	if r.uniqueKey().IsValid() {
		return false // rely on the resource cache to hold onto uniquely-keyed surfaces
	}
	// If all the refs on the proxy are known to the resource allocator then no one should be holding onto it outside
	// the GPU backend.
	return int(proxy.RefCnt()) <= knownUseCount
}

// instantiateSurface resolves the register allocation to an actual surface, cached on the originating proxy when a
// register serves multiple proxies.
func (r *allocRegister) instantiateSurface(proxy *SurfaceProxy, resourceProvider *ResourceProvider) bool {
	if proxy.PeekSurface() != nil {
		panic("instantiating an already-backed proxy")
	}
	var newSurface *Surface
	if r.existingSurface == nil {
		if proxy == r.originatingProxy {
			newSurface = proxy.createSurface(resourceProvider)
		} else {
			newSurface = r.originatingProxy.PeekSurface()
			if newSurface != nil {
				newSurface.Ref()
			}
		}
	}
	if r.existingSurface == nil && newSurface == nil {
		return false
	}
	surface := newSurface
	if surface == nil {
		surface = r.existingSurface
		// The register keeps its own ref; the proxy gets an additional one.
		surface.Ref()
	}
	// Make the surface budgeted if this proxy is budgeted.
	if proxy.IsBudgeted() == gpu.BudgetedYes && surface.BudgetedType() != gpu.BudgetedTypeBudgeted {
		// This gets the job done but isn't quite correct: better would be matching budgeted proxies with budgeted
		// surfaces and unbudgeted with unbudgeted.
		surface.MakeBudgeted()
	}
	// Propagate the proxy unique key to the surface if we have one.
	if uniqueKey := proxy.UniqueKey(); uniqueKey.IsValid() {
		if !surface.UniqueKey().IsValid() {
			resourceProvider.AssignUniqueKeyToResource(uniqueKey, surface)
		}
		if !surface.UniqueKey().Equal(uniqueKey) {
			panic("surface unique key out of sync with proxy")
		}
	}
	proxy.assign(surface)
	return true
}

// allocInterval is a proxy's usage span [start, end] within the linearized op order.
type allocInterval struct {
	proxy          *SurfaceProxy
	next           *allocInterval
	register       *allocRegister
	uses           int
	start          uint32
	end            uint32
	allowRecycling AllowRecycling
}

func (i *allocInterval) extendEnd(newEnd uint32) {
	if newEnd > i.end {
		i.end = newEnd
	}
}

// intervalList is a singly linked list of intervals with head/tail pointers.
type intervalList struct {
	head *allocInterval
	tail *allocInterval
}

func (l *intervalList) empty() bool { return l.head == nil }

func (l *intervalList) popHead() *allocInterval {
	temp := l.head
	if temp != nil {
		l.head = temp.next
		if l.head == nil {
			l.tail = nil
		}
		temp.next = nil
	}
	return temp
}

func (l *intervalList) insertByIncreasingStart(intvl *allocInterval) {
	if intvl.next != nil {
		panic("interval already linked")
	}
	switch {
	case l.head == nil:
		l.head = intvl
		l.tail = intvl
	case intvl.start <= l.head.start:
		intvl.next = l.head
		l.head = intvl
	case l.tail.start <= intvl.start:
		l.tail.next = intvl
		l.tail = intvl
	default:
		prev := l.head
		next := prev.next
		for ; intvl.start > next.start; prev, next = next, next.next {
		}
		intvl.next = next
		prev.next = intvl
	}
}

func (l *intervalList) insertByIncreasingEnd(intvl *allocInterval) {
	if intvl.next != nil {
		panic("interval already linked")
	}
	switch {
	case l.head == nil:
		l.head = intvl
		l.tail = intvl
	case intvl.end <= l.head.end:
		intvl.next = l.head
		l.head = intvl
	case l.tail.end <= intvl.end:
		l.tail.next = intvl
		l.tail = intvl
	default:
		prev := l.head
		next := prev.next
		for ; intvl.end > next.end; prev, next = next, next.next {
		}
		intvl.next = next
		prev.next = intvl
	}
}

// ResourceAllocator assigns backing surfaces to proxies at flush time (see the package doc above).
type ResourceAllocator struct {
	dContext *DirectContext

	// freePool holds recently created/used registers keyed on their scratch keys, newest last (matching the
	// head-insertion order the resource cache uses).
	freePool map[string][]*allocRegister
	// intvlHash indexes all intervals by proxy ID for reuse detection.
	intvlHash map[gpu.ResourceUniqueID]*allocInterval

	intvlList      intervalList // all intervals, sorted by increasing start
	activeIntvls   intervalList // live intervals during assignment, sorted by increasing end
	finishedIntvls intervalList // completed intervals, sorted by increasing start

	// freeIntvls is the allocInterval free list: Reset drains finishedIntvls into it (each interval zeroed so no
	// proxy/register stays reachable) and AddInterval pops from it, so the steady-state frame reuses the previous
	// flush's intervals instead of allocating. Chained through the same next pointers the live lists use.
	freeIntvls *allocInterval

	uniqueKeyRegisters map[string]*allocRegister
	registers          []*allocRegister // every register created, for Reset's ref release

	numOps uint32

	planned             bool
	assigned            bool
	failedInstantiation bool
}

// NewResourceAllocator builds an allocator for the given context.
func NewResourceAllocator(dContext *DirectContext) *ResourceAllocator {
	return &ResourceAllocator{
		dContext:           dContext,
		freePool:           make(map[string][]*allocRegister),
		intvlHash:          make(map[gpu.ResourceUniqueID]*allocInterval),
		uniqueKeyRegisters: make(map[string]*allocRegister),
	}
}

// CurOp returns the current op index (the running count of AddInterval-relevant ops so far).
func (a *ResourceAllocator) CurOp() uint32 { return a.numOps }

// IncOps advances the current op index by one.
func (a *ResourceAllocator) IncOps() { a.numOps++ }

// FailedInstantiation reports whether some proxy failed to instantiate during planning or assignment.
func (a *ResourceAllocator) FailedInstantiation() bool { return a.failedInstantiation }

// AddInterval records a usage interval from start to end inclusive for proxy. An existing interval for the proxy is
// expanded to include the new range.
func (a *ResourceAllocator) AddInterval(proxy *SurfaceProxy, start, end uint32, actualUse ActualUse, allowRecycling AllowRecycling) {
	if start > end {
		panic("invalid interval")
	}
	if a.assigned {
		panic("adding intervals after assignment")
	}
	if proxy.CanSkipResourceAllocator() {
		return
	}

	// If a proxy is read-only it must refer to a texture with specific content that cannot be recycled: no register is
	// assigned and no other proxy can share its texture.
	if proxy.ReadOnly() {
		resourceProvider := a.dContext.ResourceProvider()
		if proxy.IsLazy() && !proxy.doLazyInstantiation(resourceProvider) {
			a.failedInstantiation = true
		} else if !proxy.IsInstantiated() {
			// Since no interval is added, this proxy won't be revisited in Assign: it must already be instantiated (or
			// have just been lazily instantiated above).
			panic("read-only proxy is not instantiated")
		}
		return
	}
	if intvl := a.intvlHash[proxy.UniqueID()]; intvl != nil {
		// Revise the interval for an existing use.
		if actualUse == ActualUseYes {
			intvl.uses++
		}
		if allowRecycling == AllowRecyclingNo {
			intvl.allowRecycling = AllowRecyclingNo
		}
		intvl.extendEnd(end)
		return
	}
	newIntvl := a.freeIntvls
	if newIntvl != nil {
		a.freeIntvls = newIntvl.next
		newIntvl.next = nil
	} else {
		newIntvl = &allocInterval{}
	}
	newIntvl.proxy = proxy
	newIntvl.start = start
	newIntvl.end = end
	newIntvl.allowRecycling = AllowRecyclingYes
	if actualUse == ActualUseYes {
		newIntvl.uses++
	}
	if allowRecycling == AllowRecyclingNo {
		newIntvl.allowRecycling = AllowRecyclingNo
	}
	a.intvlList.insertByIncreasingStart(newIntvl)
	a.intvlHash[proxy.UniqueID()] = newIntvl
}

// findOrCreateRegisterFor returns a register for proxy, first trying to reuse a recently allocated/used register from
// the free pool.
func (a *ResourceAllocator) findOrCreateRegisterFor(proxy *SurfaceProxy) *allocRegister {
	resourceProvider := a.dContext.ResourceProvider()
	// Handle uniquely keyed proxies.
	if uniqueKey := proxy.UniqueKey(); uniqueKey.IsValid() {
		if r := a.uniqueKeyRegisters[uniqueKey.MapKey()]; r != nil {
			return r
		}
		// No need for a scratch key; these don't go in the free pool.
		r := newAllocRegister(proxy, gpu.ScratchKey{}, resourceProvider)
		a.uniqueKeyRegisters[uniqueKey.MapKey()] = r
		a.registers = append(a.registers, r)
		return r
	}

	// Then look in the free pool.
	var scratchKey gpu.ScratchKey
	proxy.computeScratchKey(a.dContext.GLCaps(), &scratchKey)

	if list := a.freePool[scratchKey.MapKey()]; len(list) > 0 {
		r := list[len(list)-1]
		list = list[:len(list)-1]
		if len(list) == 0 {
			delete(a.freePool, scratchKey.MapKey())
		} else {
			a.freePool[scratchKey.MapKey()] = list
		}
		return r
	}
	r := newAllocRegister(proxy, scratchKey, resourceProvider)
	a.registers = append(a.registers, r)
	return r
}

// expire removes intervals that end before curIndex from the active list, returning their registers to the free pool if
// possible.
func (a *ResourceAllocator) expire(curIndex uint32) {
	for !a.activeIntvls.empty() && a.activeIntvls.head.end < curIndex {
		intvl := a.activeIntvls.popHead()
		if r := intvl.register; r != nil && r.isRecyclable(a.dContext.GLCaps(), intvl.proxy,
			intvl.uses, intvl.allowRecycling) {
			key := r.scratchKey.MapKey()
			a.freePool[key] = append(a.freePool[key], r)
		}
		a.finishedIntvls.insertByIncreasingStart(intvl)
	}
}

// PlanAssignment generates the internal allocation plan, instantiating fully-lazy proxies so their sizes are known.
// Returns false if a lazy proxy failed to instantiate.
func (a *ResourceAllocator) PlanAssignment() bool {
	clear(a.intvlHash) // we don't need the interval hash anymore

	if a.planned || a.assigned {
		panic("planAssignment called twice")
	}
	a.planned = true

	resourceProvider := a.dContext.ResourceProvider()
	for cur := a.intvlList.popHead(); cur != nil; cur = a.intvlList.popHead() {
		a.expire(cur.start)
		a.activeIntvls.insertByIncreasingEnd(cur)

		// Already-instantiated proxies and lazy proxies don't use registers.
		if cur.proxy.IsInstantiated() {
			continue
		}

		// Instantiate fully-lazy proxies immediately; ignore other lazy proxies at this stage.
		if cur.proxy.IsLazy() {
			if cur.proxy.IsFullyLazy() {
				a.failedInstantiation = !cur.proxy.doLazyInstantiation(resourceProvider)
				if a.failedInstantiation {
					break
				}
			}
			continue
		}

		r := a.findOrCreateRegisterFor(cur.proxy)
		if cur.proxy.PeekSurface() != nil {
			panic("uninstantiated proxy has a surface")
		}
		cur.register = r
	}

	// Expire all remaining intervals to drain the active interval list.
	a.expire(math.MaxUint32)
	return !a.failedInstantiation
}

// MakeBudgetHeadroom computes how much VRAM the plan requires for new resources and asks the cache to purge enough to
// fit; returns false when that is impossible.
func (a *ResourceAllocator) MakeBudgetHeadroom() bool {
	if !a.planned || a.failedInstantiation {
		panic("makeBudgetHeadroom requires a successful plan")
	}
	var additionalBytesNeeded uint64
	for cur := a.finishedIntvls.head; cur != nil; cur = cur.next {
		proxy := cur.proxy
		if proxy.IsBudgeted() == gpu.BudgetedNo || proxy.IsInstantiated() {
			continue
		}
		// N.B. fully-lazy proxies were already instantiated in PlanAssignment.
		if proxy.IsLazy() {
			additionalBytesNeeded += proxy.GpuMemorySize()
		} else {
			r := cur.register
			if r == nil {
				panic("planned interval without a register")
			}
			if !r.accountedForInBudget && r.existingSurface == nil {
				additionalBytesNeeded += proxy.GpuMemorySize()
			}
			r.accountedForInBudget = true
		}
	}
	return a.dContext.ResourceCache().PurgeToMakeHeadroom(additionalBytesNeeded)
}

// Reset clears the allocator's state for reuse, including the explicit release of register-held surface refs. The
// failed-instantiation flag is deliberately not reset (no recovery is attempted; callers must check it and bail).
func (a *ResourceAllocator) Reset() {
	a.planned = false
	a.assigned = false
	if !a.activeIntvls.empty() {
		panic("active intervals at reset")
	}
	for _, r := range a.registers {
		if r.existingSurface != nil {
			r.existingSurface.Unref()
			r.existingSurface = nil
		}
	}
	a.registers = nil
	// Drain the finished intervals into the free list for the next flush's AddInterval calls, zeroing each so the pool
	// pins no proxy or register.
	for intvl := a.finishedIntvls.popHead(); intvl != nil; intvl = a.finishedIntvls.popHead() {
		*intvl = allocInterval{next: a.freeIntvls}
		a.freeIntvls = intvl
	}
	a.finishedIntvls = intervalList{}
	a.intvlList = intervalList{}
	clear(a.intvlHash)
	clear(a.uniqueKeyRegisters)
	clear(a.freePool)
	a.numOps = 0
}

// beginFlush returns a reused allocator to the state a freshly constructed one has, so a single persistent allocator
// can be reused across flushes rather than allocated fresh per flush (see DrawingManager's flushState/resourceAllocator
// fields). The end-of-flush Reset already clears the intervals, maps, and registers; failedInstantiation is the one
// field Reset deliberately leaves latched (the in-flush reorder retry reads it after the retry's Reset), so clearing it
// here — and only here, at the start of a flush — makes the starting state identical to a fresh allocator's without
// altering that in-flush retry behavior.
func (a *ResourceAllocator) beginFlush() {
	a.failedInstantiation = false
}

// Assign instantiates and assigns surfaces to all proxies.
func (a *ResourceAllocator) Assign() bool {
	if a.failedInstantiation {
		return false
	}
	if !a.planned || a.assigned {
		panic("assign requires a plan and must run once")
	}
	a.assigned = true
	resourceProvider := a.dContext.ResourceProvider()
	// Each popped interval is dead once handled, so it goes straight onto the free list (zeroed; see the freeIntvls
	// comment). On a failed instantiation the remaining intervals drain unprocessed (Reset would reclaim them anyway).
	for cur := a.finishedIntvls.popHead(); cur != nil; cur = a.finishedIntvls.popHead() {
		if !a.failedInstantiation {
			switch {
			case cur.proxy.IsInstantiated():
			case cur.proxy.IsLazy():
				a.failedInstantiation = !cur.proxy.doLazyInstantiation(resourceProvider)
			default:
				r := cur.register
				if r == nil {
					panic("finished interval without a register")
				}
				a.failedInstantiation = !r.instantiateSurface(cur.proxy, resourceProvider)
			}
		}
		*cur = allocInterval{next: a.freeIntvls}
		a.freeIntvls = cur
	}
	return !a.failedInstantiation
}
