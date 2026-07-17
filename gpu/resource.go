// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ResourceBase and the Resource interface: the base of every object managed by the ResourceCache. Concrete resources
// (the GL object wrappers in gpu/gl) embed ResourceBase and implement the Resource hooks; RegisterWithCache wires the
// back-pointer that stands in for virtual dispatch. The ref count still exists and still drives cache state (purgeable
// vs not), but hitting zero refs simply drops the object to the garbage collector; a resource that has been released or
// abandoned tracks this with an explicit destroyed flag (deterministic backend-object free stays with the cache — the
// GC never frees GL objects).

package gpu

import (
	"math"
	"sync/atomic"
	"time"
)

// ResourceUniqueID identifies a Resource object for state-shadowing comparisons. Zero is invalid.
type ResourceUniqueID uint32

// IsInvalid reports whether id is the invalid (zero) ID.
func (id ResourceUniqueID) IsInvalid() bool { return id == 0 }

var nextResourceUniqueID atomic.Uint32

// CreateResourceUniqueID returns a fresh, process-unique ResourceUniqueID. It never returns 0.
func CreateResourceUniqueID() ResourceUniqueID {
	for {
		if id := nextResourceUniqueID.Add(1); id != 0 {
			return ResourceUniqueID(id)
		}
	}
}

// lastRemovedRef identifies which ref type just dropped to zero.
type lastRemovedRef int32

const (
	lastRemovedMainRef lastRemovedRef = iota
	lastRemovedCommandBufferUsage
)

// Resource is the interface every cache-managed GPU resource implements. ResourceBase provides defaults for
// ComputeScratchKey and OnSetLabel; concrete types override the hooks they need.
type Resource interface {
	// resourceBase gives the cache access to the embedded ResourceBase (promoted from it).
	resourceBase() *ResourceBase
	// OnRelease frees the object in the backend API.
	OnRelease()
	// OnAbandon drops internal handles to backend objects without freeing them — the underlying 3D context is no longer
	// valid, so no backend API calls may be made.
	OnAbandon()
	// OnGpuMemorySize computes the resource's approximate GPU memory use in bytes; the result is cached by
	// GpuMemorySize.
	OnGpuMemorySize() uint64
	// ComputeScratchKey populates key if instances should be recycled as scratch resources (the ResourceBase default
	// leaves the key invalid).
	ComputeScratchKey(key *ScratchKey)
	// OnSetLabel reacts to SetLabel.
	OnSetLabel()
	// ResourceType is a diagnostic name for the kind of resource.
	ResourceType() string
}

const invalidGpuMemorySize = math.MaxUint64

// ResourceBase carries the cache bookkeeping state shared by every concrete resource and is embedded by each of them.
type ResourceBase struct {
	timeWhenBecamePurgeable time.Time
	self                    Resource
	cache                   *ResourceCache
	label                   string
	uniqueKey               UniqueKey
	scratchKey              ScratchKey
	gpuMemorySize           uint64
	// cacheArrayIndex is an index into the purgeable heap when purgeable or the nonpurgeable array when not, maintained
	// by the cache.
	cacheArrayIndex int
	// timestamp reflects how recently the resource was accessed in the cache.
	timestamp             uint32
	commandBufferUsageCnt atomic.Int32
	uniqueID              ResourceUniqueID
	refCnt                atomic.Int32
	budgetedType          BudgetedType
	refsWrappedObjects    bool
	destroyed             bool
}

func (b *ResourceBase) resourceBase() *ResourceBase { return b }

// InitResource sets the initial resource state: ref count 1, a fresh unique ID, and the label. Constructors that touch
// GL state before registering (e.g. a buffer whose creation bind is recorded in the buffer state shadow under the
// unique ID) must call this first; the Register functions call it for everyone else.
func (b *ResourceBase) InitResource(label string) {
	b.refCnt.Store(1) // starts with one ref.
	b.cacheArrayIndex = -1
	b.gpuMemorySize = invalidGpuMemorySize
	b.budgetedType = BudgetedTypeUnbudgetedUncacheable
	b.uniqueID = CreateResourceUniqueID()
	b.label = label
}

// initResource wires the cache/self back-pointers, running InitResource first if the constructor has not already done
// so.
func (b *ResourceBase) initResource(cache *ResourceCache, self Resource, label string) {
	if b.uniqueID == 0 {
		b.InitResource(label)
	}
	b.self = self
	b.cache = cache
}

// RegisterWithCache must be called exactly once by every non-wrapped resource, after the concrete object is fully
// initialized.
func (b *ResourceBase) RegisterWithCache(cache *ResourceCache, self Resource, budgeted Budgeted, label string) {
	b.initResource(cache, self, label)
	if budgeted {
		b.budgetedType = BudgetedTypeBudgeted
	} else {
		b.budgetedType = BudgetedTypeUnbudgetedUncacheable
	}
	self.ComputeScratchKey(&b.scratchKey)
	cache.insertResource(self)
}

// RegisterWithCacheWrapped registers a resource wrapping an externally-owned backend object. Wrapped resources are
// never budgeted, but may be cached or uncached.
func (b *ResourceBase) RegisterWithCacheWrapped(cache *ResourceCache, self Resource, wrapType WrapCacheable, label string) {
	b.initResource(cache, self, label)
	if wrapType {
		b.budgetedType = BudgetedTypeUnbudgetedCacheable
	} else {
		b.budgetedType = BudgetedTypeUnbudgetedUncacheable
	}
	b.refsWrappedObjects = true
	cache.insertResource(self)
}

// Default hook implementations (overridden by concrete types that need something other than no-op behavior).

// ComputeScratchKey is the default no-op: by default resources are not recycled as scratch.
func (b *ResourceBase) ComputeScratchKey(*ScratchKey) {}

// OnSetLabel is the default no-op label hook.
func (b *ResourceBase) OnSetLabel() {}

// Ref adds a strong reference. Only the cache may take a resource from zero refs to one.
func (b *ResourceBase) Ref() {
	b.refCnt.Add(1)
}

// Unref drops a strong reference, notifying the cache once the count reaches zero.
func (b *ResourceBase) Unref() {
	if b.refCnt.Add(-1) == 0 {
		b.notifyWillBeZero(lastRemovedMainRef)
	}
}

// RefCommandBuffer records that a command buffer still references this resource.
func (b *ResourceBase) RefCommandBuffer() {
	b.commandBufferUsageCnt.Add(1)
}

// UnrefCommandBuffer releases one command-buffer reference, notifying the cache once none remain.
func (b *ResourceBase) UnrefCommandBuffer() {
	if b.commandBufferUsageCnt.Add(-1) == 0 {
		b.notifyWillBeZero(lastRemovedCommandBufferUsage)
	}
}

func (b *ResourceBase) notifyWillBeZero(removedRef lastRemovedRef) {
	// Once destroyed there is nothing further to notify; the object just becomes garbage.
	if b.destroyed {
		return
	}
	b.cache.notifyARefCntReachedZero(b.self, removedRef)
}

// Unique reports whether this is the only outstanding strong reference.
func (b *ResourceBase) Unique() bool { return b.refCnt.Load() == 1 }

func (b *ResourceBase) hasRef() bool { return b.refCnt.Load() > 0 }

func (b *ResourceBase) hasNoCommandBufferUsages() bool { return b.commandBufferUsageCnt.Load() == 0 }

func (b *ResourceBase) hasRefOrCommandBufferUsage() bool {
	return b.hasRef() || !b.hasNoCommandBufferUsages()
}

// WasDestroyed reports whether the resource has been released or abandoned (its creating context destroyed or lost).
func (b *ResourceBase) WasDestroyed() bool { return b.destroyed }

// GpuMemorySize returns the resource's approximate GPU memory use in bytes, computing and caching it on first call.
func (b *ResourceBase) GpuMemorySize() uint64 {
	if b.gpuMemorySize == invalidGpuMemorySize {
		b.gpuMemorySize = b.self.OnGpuMemorySize()
	}
	return b.gpuMemorySize
}

// UniqueID returns the resource's process-unique ID.
func (b *ResourceBase) UniqueID() ResourceUniqueID { return b.uniqueID }

// UniqueKey returns the resource's unique key (invalid if it has none).
func (b *ResourceBase) UniqueKey() *UniqueKey { return &b.uniqueKey }

// ScratchKey returns the resource's scratch key (invalid if it has none).
func (b *ResourceBase) ScratchKey() *ScratchKey { return &b.scratchKey }

// Label returns the resource's debug label.
func (b *ResourceBase) Label() string { return b.label }

// SetLabel updates the resource's debug label and notifies the concrete type.
func (b *ResourceBase) SetLabel(label string) {
	b.label = label
	b.self.OnSetLabel()
}

// BudgetedType returns the resource's current budgeted-type classification.
func (b *ResourceBase) BudgetedType() BudgetedType { return b.budgetedType }

// RefsWrappedObjects reports whether this resource wraps an externally-owned backend object.
func (b *ResourceBase) RefsWrappedObjects() bool { return b.refsWrappedObjects }

// IsPurgeable reports whether the cache may reclaim this resource. Resources in the unbudgeted-cacheable state are
// never purgeable while they hold a unique key.
func (b *ResourceBase) IsPurgeable() bool {
	return !b.hasRef() && b.hasNoCommandBufferUsages() &&
		(b.budgetedType != BudgetedTypeUnbudgetedCacheable || !b.uniqueKey.IsValid())
}

// release frees the backend object and removes the resource from the cache.
func (b *ResourceBase) release() {
	b.self.OnRelease()
	b.cache.removeResource(b.self)
	b.destroyed = true
	b.gpuMemorySize = 0
}

// abandon drops backend handles without freeing them.
func (b *ResourceBase) abandon() {
	if b.destroyed {
		return
	}
	b.self.OnAbandon()
	b.cache.removeResource(b.self)
	b.destroyed = true
	b.gpuMemorySize = 0
}

// SetUniqueKey assigns key to this resource via the cache. The caller must hold a ref. Uncached resources can never
// have a unique key unless they are wrapped.
func (b *ResourceBase) SetUniqueKey(key *UniqueKey) {
	if !b.hasRef() || !key.IsValid() {
		panic("SetUniqueKey requires a ref and a valid key")
	}
	if b.budgetedType != BudgetedTypeBudgeted && !b.refsWrappedObjects {
		return
	}
	if b.destroyed {
		return
	}
	b.cache.changeUniqueKey(b.self, key)
}

// RemoveUniqueKey drops this resource's unique key via the cache.
func (b *ResourceBase) RemoveUniqueKey() {
	if b.destroyed {
		return
	}
	b.cache.removeUniqueKey(b.self)
}

// RemoveScratchKey drops this resource's scratch key. Once removed, the resource will never again be usable as scratch.
func (b *ResourceBase) RemoveScratchKey() {
	if !b.destroyed && b.scratchKey.IsValid() {
		b.cache.willRemoveScratchKey(b.self)
		b.scratchKey.Reset()
	}
}

// MakeBudgeted moves an unbudgeted-uncacheable resource into the budgeted pool.
func (b *ResourceBase) MakeBudgeted() {
	if !b.destroyed && b.budgetedType == BudgetedTypeUnbudgetedUncacheable {
		b.budgetedType = BudgetedTypeBudgeted
		b.cache.didChangeBudgetStatus(b.self)
	}
}

// MakeUnbudgeted moves a budgeted resource out of the budget. Resources with unique keys cannot be made unbudgeted.
func (b *ResourceBase) MakeUnbudgeted() {
	if !b.destroyed && b.budgetedType == BudgetedTypeBudgeted && !b.uniqueKey.IsValid() {
		b.budgetedType = BudgetedTypeUnbudgetedUncacheable
		b.cache.didChangeBudgetStatus(b.self)
	}
}

// isScratch reports whether the resource is cached, holds a valid scratch key, and has no unique key.
func (b *ResourceBase) isScratch() bool {
	return !b.uniqueKey.IsValid() && b.scratchKey.IsValid() &&
		b.budgetedType == BudgetedTypeBudgeted
}

// isUsableAsScratch reports whether the resource can currently be handed out as scratch.
func (b *ResourceBase) isUsableAsScratch() bool {
	return b.isScratch() && !b.hasRef()
}
