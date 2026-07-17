// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ResourceCache: the budgeted LRU cache that manages the lifetime of every Resource. Purgeable resources sit in a
// timestamp-ordered priority queue (the purgeableQueue binary heap below); non-purgeable ones in an unordered array —
// both track their position through the resource's cacheArrayIndex. Scratch resources are found through a multimap-like
// structure that preserves head-insertion order (the most recently inserted resource for a key is returned first);
// uniquely keyed resources through a hash keyed on the full key bytes.
//
// This is a single-threaded cache: cross-thread unref and cross-thread key invalidation have no counterpart here. The
// ThreadSafeCacheHook interface lets a separate thread-safe cache layer plug in: abandonAll/releaseAll drop all of its
// refs, PurgeAsNeeded asks it to drop uniquely-held refs when still over budget after the first purge pass, and
// purgeUnlockedResources(PurgeAllResources) drops its uniquely-held (optionally stale) refs up front.

package gpu

import (
	"fmt"
	"sort"
	"time"
)

// ResourceCacheDefaultMaxSize is the default maximum number of bytes of GPU memory that budgeted resources may occupy.
const ResourceCacheDefaultMaxSize = 256 << 20

// PurgeResourceOptions selects which purgeable resources a purge call is allowed to reclaim.
type PurgeResourceOptions int32

// PurgeResourceOptions values.
const (
	// PurgeAllResources purges all purgeable resources.
	PurgeAllResources PurgeResourceOptions = iota
	// PurgeScratchResourcesOnly spares purgeable resources containing persistent data (those with unique keys).
	PurgeScratchResourcesOnly
)

// resourceCacheValidation enables the validate() consistency checks (tests only).
var resourceCacheValidation = false

// ResourceCache manages the lifetime of all Resource instances. Resources may have a scratch key (contents don't
// matter, allocations are interchangeable) and/or a unique key (exclusive use); a unique key always preempts the
// scratch key. A resource with neither key is deleted as soon as its last ref drops.
type ResourceCache struct {
	// newlyPurgeableResourceForValidation: this resource is allowed to be in the nonpurgeable array during validation
	// because it is in the midst of being converted to purgeable status.
	newlyPurgeableResourceForValidation Resource
	// threadSafeCache is the optional thread-safe cache hook; set via SetThreadSafeCache.
	threadSafeCache ThreadSafeCacheHook
	uniqueHash      map[string]Resource
	scratchMap      map[string][]Resource // all resources usable as scratch, newest last (head)
	// scratchBackings holds []Resource backing arrays reclaimed when a scratch-map key's list empties (the key is
	// deleted from the map on empty). scratchInsert reuses one when it (re)creates a key's list, so the steady-state
	// recycle of a scratch resource — e.g. a dynamic vertex/index buffer freed and re-acquired every frame — does not
	// re-grow its list from nil each frame. Always cleared before it is reused, so it pins no Resource, and bounded so
	// a burst of distinct keys cannot retain unbounded storage; this recovers the benefit a pooled-list-node
	// implementation would get for free.
	scratchBackings       [][]Resource
	nonpurgeableResources []Resource
	purgeableQueue        purgeableQueue
	scratchCount          int
	maxBytes              uint64
	count                 int
	bytes                 uint64
	budgetedCount         int
	budgetedBytes         uint64
	purgeableBytes        uint64
	// timestamp is assigned to a resource on insertion and on each cache hit, then incremented; the purgeable queue
	// orders by it, giving LRU purge order.
	timestamp uint32
}

// ThreadSafeCacheHook is the surface a separate thread-safe cache layer exposes to ResourceCache; an interface here
// because the cache implementation lives in the backend package.
type ThreadSafeCacheHook interface {
	// DropAllRefs drops every ref this hook holds.
	DropAllRefs()
	// DropUniqueRefs drops uniquely held refs until cache is under budget; a nil cache means drop all uniquely held
	// refs.
	DropUniqueRefs(cache *ResourceCache)
	// DropUniqueRefsOlderThan drops uniquely held refs that became stale before purgeTime.
	DropUniqueRefsOlderThan(purgeTime time.Time)
}

// SetThreadSafeCache installs the optional thread-safe cache hook.
func (c *ResourceCache) SetThreadSafeCache(h ThreadSafeCacheHook) { c.threadSafeCache = h }

// NewResourceCache creates an empty cache with the default budget.
func NewResourceCache() *ResourceCache {
	return &ResourceCache{
		maxBytes:   ResourceCacheDefaultMaxSize,
		scratchMap: make(map[string][]Resource),
		uniqueHash: make(map[string]Resource),
	}
}

// SetLimit changes the cache's byte budget, purging if the new limit is now exceeded.
func (c *ResourceCache) SetLimit(bytes uint64) {
	c.maxBytes = bytes
	c.PurgeAsNeeded()
}

// ResourceCount returns the total number of resources currently in the cache.
func (c *ResourceCache) ResourceCount() int {
	return c.purgeableQueue.count() + len(c.nonpurgeableResources)
}

// BudgetedResourceCount returns the number of budgeted resources currently in the cache.
func (c *ResourceCache) BudgetedResourceCount() int { return c.budgetedCount }

// ResourceBytes returns the total GPU memory used by all resources in the cache.
func (c *ResourceCache) ResourceBytes() uint64 { return c.bytes }

// PurgeableBytes returns the GPU memory used by resources currently eligible for purging.
func (c *ResourceCache) PurgeableBytes() uint64 { return c.purgeableBytes }

// BudgetedResourceBytes returns the GPU memory used by budgeted resources.
func (c *ResourceCache) BudgetedResourceBytes() uint64 { return c.budgetedBytes }

// MaxResourceBytes returns the cache's current byte budget.
func (c *ResourceCache) MaxResourceBytes() uint64 { return c.maxBytes }

// OverBudget reports whether budgeted resource usage currently exceeds the byte budget.
func (c *ResourceCache) OverBudget() bool { return c.budgetedBytes > c.maxBytes }

func (c *ResourceCache) wouldFit(bytes uint64) bool { return c.budgetedBytes+bytes <= c.maxBytes }

func (c *ResourceCache) insertResource(resource Resource) {
	b := resource.resourceBase()
	if c.isInCache(resource) {
		panic("resource already in cache")
	}
	if b.destroyed || b.IsPurgeable() {
		panic("inserting destroyed or purgeable resource")
	}

	// We must set the timestamp before adding to the array in case the timestamp wraps and we wind up iterating over
	// all the resources that already have timestamps.
	b.timestamp = c.getNextTimestamp()

	c.addToNonpurgeableArray(resource)

	size := b.GpuMemorySize()
	c.count++
	c.bytes += size
	if b.budgetedType == BudgetedTypeBudgeted {
		c.budgetedCount++
		c.budgetedBytes += size
	}
	c.PurgeAsNeeded()
}

func (c *ResourceCache) removeResource(resource Resource) {
	c.validate()
	b := resource.resourceBase()
	if !c.isInCache(resource) {
		panic("removing resource not in cache")
	}

	size := b.GpuMemorySize()
	if b.IsPurgeable() {
		c.purgeableQueue.remove(resource)
		c.purgeableBytes -= size
	} else {
		c.removeFromNonpurgeableArray(resource)
	}

	c.count--
	c.bytes -= size
	if b.budgetedType == BudgetedTypeBudgeted {
		c.budgetedCount--
		c.budgetedBytes -= size
	}

	if b.isUsableAsScratch() {
		c.scratchRemove(&b.scratchKey, resource)
	}
	if b.uniqueKey.IsValid() {
		delete(c.uniqueHash, b.uniqueKey.mapKey())
	}
	c.validate()
}

// AbandonAll drops the backend API objects of all resources without freeing them, and removes everything from the
// cache. Used when the underlying GPU context has been lost.
func (c *ResourceCache) AbandonAll() {
	c.validate()
	defer c.validate()

	for len(c.nonpurgeableResources) > 0 {
		back := c.nonpurgeableResources[len(c.nonpurgeableResources)-1]
		back.resourceBase().abandon()
	}
	for c.purgeableQueue.count() > 0 {
		top := c.purgeableQueue.peek()
		top.resourceBase().abandon()
	}
	if c.threadSafeCache != nil {
		c.threadSafeCache.DropAllRefs()
	}
	if c.scratchCount != 0 || len(c.uniqueHash) != 0 || c.count != 0 || c.ResourceCount() != 0 ||
		c.bytes != 0 || c.budgetedCount != 0 || c.budgetedBytes != 0 || c.purgeableBytes != 0 {
		panic("resource cache not empty after abandonAll")
	}
	// The teardown loop repopulated the reclaimed-backing free list via scratchRemove; drop it so an emptied cache
	// retains no storage.
	c.scratchBackings = nil
}

// ReleaseAll frees the backend API objects of all resources and removes everything from the cache.
func (c *ResourceCache) ReleaseAll() {
	c.validate()
	defer c.validate()

	// Drop the thread-safe cache's refs before freeing anything, so payloads it holds become purgeable and are released
	// by the loops below.
	if c.threadSafeCache != nil {
		c.threadSafeCache.DropAllRefs()
	}

	for len(c.nonpurgeableResources) > 0 {
		back := c.nonpurgeableResources[len(c.nonpurgeableResources)-1]
		c.releaseResource(back)
	}
	for c.purgeableQueue.count() > 0 {
		top := c.purgeableQueue.peek()
		c.releaseResource(top)
	}
	if c.scratchCount != 0 || len(c.uniqueHash) != 0 || c.count != 0 || c.ResourceCount() != 0 ||
		c.bytes != 0 || c.budgetedCount != 0 || c.budgetedBytes != 0 || c.purgeableBytes != 0 {
		panic("resource cache not empty after releaseAll")
	}
	// The teardown loop repopulated the reclaimed-backing free list via scratchRemove; drop it so an emptied cache
	// retains no storage.
	c.scratchBackings = nil
}

// releaseResource is the cache freeing a resource.
func (c *ResourceCache) releaseResource(resource Resource) {
	resource.resourceBase().release()
}

// RefResource adds a ref with proper cache tracking even if the resource currently has none (the surface-proxy path).
func (c *ResourceCache) RefResource(resource Resource) {
	b := resource.resourceBase()
	if b.cache != c {
		panic("resource belongs to another cache")
	}
	if b.hasRef() {
		b.Ref()
	} else {
		c.refAndMakeResourceMRU(resource)
	}
	c.validate()
}

// FindAndRefScratchResource looks up and returns a ref'd scratch resource matching scratchKey, or nil if none is
// cached.
func (c *ResourceCache) FindAndRefScratchResource(scratchKey *ScratchKey) Resource {
	if !scratchKey.IsValid() {
		panic("invalid scratch key")
	}
	key := scratchKey.mapKey()
	list := c.scratchMap[key]
	if len(list) == 0 {
		return nil
	}
	resource := list[len(list)-1] // the multimap head: most recently inserted
	c.scratchRemove(scratchKey, resource)
	c.refAndMakeResourceMRU(resource)
	c.validate()
	return resource
}

func (c *ResourceCache) scratchInsert(scratchKey *ScratchKey, resource Resource) {
	key := scratchKey.mapKey()
	list := c.scratchMap[key]
	if list == nil {
		// The key is absent (delete-on-empty removed its list, or it was never seen). Reuse a reclaimed backing array
		// if one is available so the steady-state recycle of a scratch resource does not re-grow its list from nil;
		// append below reuses the backing in place. A present key always has length >= 1 (delete-on-empty), so list ==
		// nil iff the key is absent.
		list = c.borrowScratchBacking()
	}
	c.scratchMap[key] = append(list, resource)
	c.scratchCount++
}

func (c *ResourceCache) scratchRemove(scratchKey *ScratchKey, resource Resource) {
	key := scratchKey.mapKey()
	list := c.scratchMap[key]
	for i := len(list) - 1; i >= 0; i-- {
		if list[i] == resource {
			list = append(list[:i], list[i+1:]...)
			if len(list) == 0 {
				delete(c.scratchMap, key)
				c.recycleScratchBacking(list)
			} else {
				c.scratchMap[key] = list
			}
			c.scratchCount--
			return
		}
	}
	panic("resource not in scratch map")
}

// scratchBackingsMax bounds the reclaimed-backing free list so a pathological burst of distinct scratch keys emptied
// without reuse cannot retain unbounded slice headers. Each backing is tiny (a handful of Resource slots) and always
// cleared, so this is purely a defensive cap; steady-state usage keeps only a couple of backings live.
const scratchBackingsMax = 32

// recycleScratchBacking reclaims an emptied scratch-map backing (length 0, capacity possibly nonzero) for reuse by a
// later scratchInsert. The full capacity is cleared first because append leaves the removed element(s) in the
// underlying array, so the backing would otherwise pin dead Resources — this gives the same leak-safety a
// pooled-list-node implementation would get from freeing its node.
func (c *ResourceCache) recycleScratchBacking(list []Resource) {
	if cap(list) == 0 || len(c.scratchBackings) >= scratchBackingsMax {
		return
	}
	list = list[:cap(list)]
	clear(list)
	c.scratchBackings = append(c.scratchBackings, list[:0])
}

// borrowScratchBacking returns a cleared, length-0 backing array reclaimed by a prior recycleScratchBacking, or nil
// when none is available (in which case scratchInsert's append allocates a fresh one, exactly as before).
func (c *ResourceCache) borrowScratchBacking() []Resource {
	n := len(c.scratchBackings)
	if n == 0 {
		return nil
	}
	list := c.scratchBackings[n-1]
	c.scratchBackings[n-1] = nil
	c.scratchBackings = c.scratchBackings[:n-1]
	return list
}

func (c *ResourceCache) willRemoveScratchKey(resource Resource) {
	b := resource.resourceBase()
	if !b.scratchKey.IsValid() {
		panic("expected valid scratch key")
	}
	if b.isUsableAsScratch() {
		c.scratchRemove(&b.scratchKey, resource)
	}
}

func (c *ResourceCache) removeUniqueKey(resource Resource) {
	b := resource.resourceBase()
	// Someone has a ref to this resource in order to have removed the key. When the ref count reaches zero we will get
	// a ref cnt notification and figure out what to do with it.
	if b.uniqueKey.IsValid() {
		if c.uniqueHash[b.uniqueKey.mapKey()] != resource {
			panic("unique hash inconsistent")
		}
		delete(c.uniqueHash, b.uniqueKey.mapKey())
	}
	b.uniqueKey.Reset()
	if b.isUsableAsScratch() {
		c.scratchInsert(&b.scratchKey, resource)
	}
	c.validate()
}

func (c *ResourceCache) changeUniqueKey(resource Resource, newKey *UniqueKey) {
	b := resource.resourceBase()
	if !c.isInCache(resource) {
		panic("resource not in cache")
	}

	// If another resource has the new key, remove its key then install the key on this resource.
	if newKey.IsValid() {
		if old := c.uniqueHash[newKey.mapKey()]; old != nil {
			ob := old.resourceBase()
			// If the old resource using the key is purgeable and is unreachable, then remove it.
			if !ob.scratchKey.IsValid() && ob.IsPurgeable() {
				c.releaseResource(old)
			} else {
				// removeUniqueKey expects an external owner of the resource.
				ob.Ref()
				c.removeUniqueKey(old)
				ob.Unref()
			}
		}
		if c.uniqueHash[newKey.mapKey()] != nil {
			panic("new unique key still in use")
		}

		// Remove the entry for this resource if it already has a unique key.
		if b.uniqueKey.IsValid() {
			if c.uniqueHash[b.uniqueKey.mapKey()] != resource {
				panic("unique hash inconsistent")
			}
			delete(c.uniqueHash, b.uniqueKey.mapKey())
		} else if b.isUsableAsScratch() {
			// The resource didn't have a valid unique key before, so it is switching sides. Remove it from the scratch
			// map before adding the new unique key.
			c.scratchRemove(&b.scratchKey, resource)
		}

		b.uniqueKey = *newKey
		c.uniqueHash[b.uniqueKey.mapKey()] = resource
	} else {
		c.removeUniqueKey(resource)
	}
	c.validate()
}

// FindAndRefUniqueResource looks up and returns a ref'd resource matching key, or nil if none is cached.
func (c *ResourceCache) FindAndRefUniqueResource(key *UniqueKey) Resource {
	resource := c.uniqueHash[key.mapKey()]
	if resource != nil {
		c.refAndMakeResourceMRU(resource)
	}
	return resource
}

// HasUniqueKey reports whether a resource with this unique key is currently cached.
func (c *ResourceCache) HasUniqueKey(key *UniqueKey) bool {
	return c.uniqueHash[key.mapKey()] != nil
}

func (c *ResourceCache) refAndMakeResourceMRU(resource Resource) {
	b := resource.resourceBase()
	if !c.isInCache(resource) {
		panic("resource not in cache")
	}
	if b.IsPurgeable() {
		// It's about to become unpurgeable.
		c.purgeableBytes -= b.GpuMemorySize()
		c.purgeableQueue.remove(resource)
		c.addToNonpurgeableArray(resource)
	}
	b.refCnt.Add(1) // only the cache may take a resource from zero refs to one.
	b.timestamp = c.getNextTimestamp()
	c.validate()
}

func (c *ResourceCache) notifyARefCntReachedZero(resource Resource, removedRef lastRemovedRef) {
	b := resource.resourceBase()
	if b.destroyed || !c.isInCache(resource) {
		panic("notification for resource not in cache")
	}
	// This resource should always be in the nonpurgeable array when this function is called. It will be moved to the
	// queue if it is newly purgeable.
	if c.nonpurgeableResources[b.cacheArrayIndex] != resource {
		panic("resource not in nonpurgeable array")
	}

	if removedRef == lastRemovedMainRef && b.isUsableAsScratch() {
		c.scratchInsert(&b.scratchKey, resource)
	}

	if b.hasRefOrCommandBufferUsage() {
		c.validate()
		return
	}

	// When the timestamp overflows validate() checks that resources in the nonpurgeable array are indeed not purgeable;
	// the movement to the purgeable queue happens just below, so mark the exception.
	if b.IsPurgeable() {
		c.newlyPurgeableResourceForValidation = resource
	}
	b.timestamp = c.getNextTimestamp()
	c.newlyPurgeableResourceForValidation = nil

	if !b.IsPurgeable() {
		c.validate()
		return
	}

	c.removeFromNonpurgeableArray(resource)
	c.purgeableQueue.insert(resource)
	b.timeWhenBecamePurgeable = time.Now()
	c.purgeableBytes += b.GpuMemorySize()

	hasUniqueKey := b.uniqueKey.IsValid()

	if b.budgetedType == BudgetedTypeBudgeted {
		// Purge the resource immediately if we're over budget. Also purge if the resource has neither a valid scratch
		// key nor a unique key.
		hasKey := b.scratchKey.IsValid() || hasUniqueKey
		if !c.OverBudget() && hasKey {
			return
		}
	} else {
		// We keep unbudgeted resources with a unique key in the purgeable queue of the cache so they can be reused
		// again by the image connected to the unique key.
		if hasUniqueKey && b.budgetedType == BudgetedTypeUnbudgetedCacheable {
			return
		}
		// Check whether this resource could still be used as a scratch resource.
		if !b.refsWrappedObjects && b.scratchKey.IsValid() {
			// We won't purge an existing resource to make room for this one.
			if c.wouldFit(b.GpuMemorySize()) {
				b.MakeBudgeted()
				return
			}
		}
	}

	beforeCount := c.ResourceCount()
	c.releaseResource(resource)
	// We should at least free this resource, perhaps dependent resources as well.
	if c.ResourceCount() >= beforeCount {
		panic("release freed no resources")
	}
	c.validate()
}

func (c *ResourceCache) didChangeBudgetStatus(resource Resource) {
	b := resource.resourceBase()
	if !c.isInCache(resource) {
		panic("resource not in cache")
	}

	size := b.GpuMemorySize()
	if b.budgetedType == BudgetedTypeBudgeted {
		c.budgetedCount++
		c.budgetedBytes += size
		if b.isUsableAsScratch() {
			c.scratchInsert(&b.scratchKey, resource)
		}
		c.PurgeAsNeeded()
	} else {
		if b.budgetedType == BudgetedTypeUnbudgetedCacheable {
			panic("cannot transition to unbudgeted-cacheable")
		}
		c.budgetedCount--
		c.budgetedBytes -= size
		if !b.hasRef() && !b.uniqueKey.IsValid() && b.scratchKey.IsValid() {
			c.scratchRemove(&b.scratchKey, resource)
		}
	}
	c.validate()
}

// PurgeAsNeeded purges resources to get back under budget.
func (c *ResourceCache) PurgeAsNeeded() {
	stillOverbudget := c.OverBudget()
	for stillOverbudget && c.purgeableQueue.count() > 0 {
		resource := c.purgeableQueue.peek()
		if !resource.resourceBase().IsPurgeable() {
			panic("unpurgeable resource in purgeable queue")
		}
		c.releaseResource(resource)
		stillOverbudget = c.OverBudget()
	}

	if stillOverbudget && c.threadSafeCache != nil {
		c.threadSafeCache.DropUniqueRefs(c)

		stillOverbudget = c.OverBudget()
		for stillOverbudget && c.purgeableQueue.count() > 0 {
			resource := c.purgeableQueue.peek()
			if !resource.resourceBase().IsPurgeable() {
				panic("unpurgeable resource in purgeable queue")
			}
			c.releaseResource(resource)
			stillOverbudget = c.OverBudget()
		}
	}
	c.validate()
}

// PurgeUnlockedResources purges unlocked resources per opts.
func (c *ResourceCache) PurgeUnlockedResources(opts PurgeResourceOptions) {
	c.purgeUnlockedResources(nil, opts)
}

// PurgeResourcesNotUsedSince purges unlocked resources that became purgeable before the passed point in time.
func (c *ResourceCache) PurgeResourcesNotUsedSince(purgeTime time.Time, opts PurgeResourceOptions) {
	c.purgeUnlockedResources(&purgeTime, opts)
}

func (c *ResourceCache) purgeUnlockedResources(purgeTime *time.Time, opts PurgeResourceOptions) {
	if opts == PurgeAllResources {
		if c.threadSafeCache != nil {
			if purgeTime != nil {
				c.threadSafeCache.DropUniqueRefsOlderThan(*purgeTime)
			} else {
				c.threadSafeCache.DropUniqueRefs(nil)
			}
		}
		// We could disable maintaining the heap property here, but it would add a lot of complexity. Moreover, this is
		// rarely called.
		for c.purgeableQueue.count() > 0 {
			resource := c.purgeableQueue.peek()
			if purgeTime != nil &&
				!resource.resourceBase().timeWhenBecamePurgeable.Before(*purgeTime) {
				// Resources were given both LRU timestamps and tagged with a frame number when they first became
				// purgeable. The LRU timestamp won't change again until the resource is made non-purgeable again. So,
				// at this point all the remaining resources in the timestamp-sorted queue will have a frame number >=
				// to this one.
				break
			}
			c.releaseResource(resource)
		}
	} else {
		// Early out if the very first item is too new to purge to avoid sorting the queue when nothing will be deleted.
		if purgeTime != nil && c.purgeableQueue.count() > 0 &&
			!c.purgeableQueue.peek().resourceBase().timeWhenBecamePurgeable.Before(*purgeTime) {
			return
		}

		c.purgeableQueue.sort()

		// Make a list of the scratch resources to delete, then delete them in a separate pass to avoid messing up the
		// sorted order of the queue.
		var scratchResources []Resource
		for i := 0; i < c.purgeableQueue.count(); i++ {
			resource := c.purgeableQueue.at(i)
			b := resource.resourceBase()
			if purgeTime != nil && !b.timeWhenBecamePurgeable.Before(*purgeTime) {
				// scratch or not, all later iterations will be too recently used to purge.
				break
			}
			if !b.uniqueKey.IsValid() {
				scratchResources = append(scratchResources, resource)
			}
		}
		for _, resource := range scratchResources {
			c.releaseResource(resource)
		}
	}
	c.validate()
}

// PurgeToMakeHeadroom purges just enough resources to create the desired budget headroom, if possible, returning true;
// otherwise does nothing and returns false.
func (c *ResourceCache) PurgeToMakeHeadroom(desiredHeadroomBytes uint64) bool {
	c.validate()
	defer c.validate()
	if desiredHeadroomBytes > c.maxBytes {
		return false
	}
	if c.wouldFit(desiredHeadroomBytes) {
		return true
	}
	c.purgeableQueue.sort()

	projectedBudget := c.budgetedBytes
	purgeCnt := 0
	for i := 0; i < c.purgeableQueue.count(); i++ {
		b := c.purgeableQueue.at(i).resourceBase()
		if b.budgetedType == BudgetedTypeBudgeted {
			projectedBudget -= b.GpuMemorySize()
		}
		if projectedBudget+desiredHeadroomBytes <= c.maxBytes {
			purgeCnt = i + 1
			break
		}
	}
	if purgeCnt == 0 {
		return false
	}

	// Success! Release the resources. Copy to an array first so we don't mess with the queue.
	resources := make([]Resource, purgeCnt)
	for i := 0; i < purgeCnt; i++ {
		resources[i] = c.purgeableQueue.at(i)
	}
	for _, resource := range resources {
		c.releaseResource(resource)
	}
	return true
}

// PurgeUnlockedResourcesBytes purges until bytesToPurge is reached or all unlocked resources are purged, optionally
// preferring scratch resources.
func (c *ResourceCache) PurgeUnlockedResourcesBytes(bytesToPurge uint64, preferScratchResources bool) {
	tmpByteBudget := uint64(0)
	if bytesToPurge < c.bytes {
		tmpByteBudget = c.bytes - bytesToPurge
	}
	stillOverbudget := tmpByteBudget < c.bytes

	if preferScratchResources && bytesToPurge < c.purgeableBytes {
		c.purgeableQueue.sort()

		var scratchResources []Resource
		scratchByteCount := uint64(0)
		for i := 0; i < c.purgeableQueue.count() && stillOverbudget; i++ {
			resource := c.purgeableQueue.at(i)
			b := resource.resourceBase()
			if !b.uniqueKey.IsValid() {
				scratchResources = append(scratchResources, resource)
				scratchByteCount += b.GpuMemorySize()
				stillOverbudget = tmpByteBudget < c.bytes-scratchByteCount
			}
		}
		for _, resource := range scratchResources {
			c.releaseResource(resource)
		}
		stillOverbudget = tmpByteBudget < c.bytes
		c.validate()
	}

	// Purge any remaining resources in LRU order.
	if stillOverbudget {
		cachedByteCount := c.maxBytes
		c.maxBytes = tmpByteBudget
		c.PurgeAsNeeded()
		c.maxBytes = cachedByteCount
	}
}

func (c *ResourceCache) addToNonpurgeableArray(resource Resource) {
	resource.resourceBase().cacheArrayIndex = len(c.nonpurgeableResources)
	c.nonpurgeableResources = append(c.nonpurgeableResources, resource)
}

func (c *ResourceCache) removeFromNonpurgeableArray(resource Resource) {
	b := resource.resourceBase()
	// Fill the hole we will create in the array with the tail object, adjust its index, and then pop the array.
	tail := c.nonpurgeableResources[len(c.nonpurgeableResources)-1]
	if c.nonpurgeableResources[b.cacheArrayIndex] != resource {
		panic("nonpurgeable array inconsistent")
	}
	c.nonpurgeableResources[b.cacheArrayIndex] = tail
	tail.resourceBase().cacheArrayIndex = b.cacheArrayIndex
	c.nonpurgeableResources = c.nonpurgeableResources[:len(c.nonpurgeableResources)-1]
	b.cacheArrayIndex = -1
}

func (c *ResourceCache) getNextTimestamp() uint32 {
	// If we wrap then all the existing resources will appear older than any resources that get a timestamp after the
	// wrap.
	if c.timestamp == 0 {
		count := c.ResourceCount()
		if count > 0 {
			// Reset all the timestamps: sort the resources by timestamp and then assign sequential timestamps beginning
			// with 0. O(n*lg(n)) but extremely rare.
			sortedPurgeableResources := make([]Resource, 0, c.purgeableQueue.count())
			for c.purgeableQueue.count() > 0 {
				sortedPurgeableResources = append(sortedPurgeableResources, c.purgeableQueue.peek())
				c.purgeableQueue.pop()
			}

			sort.SliceStable(c.nonpurgeableResources, func(i, j int) bool {
				return c.nonpurgeableResources[i].resourceBase().timestamp <
					c.nonpurgeableResources[j].resourceBase().timestamp
			})

			// Pick resources out of the purgeable and non-purgeable arrays based on lowest timestamp and assign new
			// timestamps.
			currP := 0
			currNP := 0
			for currP < len(sortedPurgeableResources) && currNP < len(c.nonpurgeableResources) {
				tsP := sortedPurgeableResources[currP].resourceBase().timestamp
				tsNP := c.nonpurgeableResources[currNP].resourceBase().timestamp
				if tsP == tsNP {
					panic("duplicate timestamps")
				}
				if tsP < tsNP {
					sortedPurgeableResources[currP].resourceBase().timestamp = c.timestamp
					c.timestamp++
					currP++
				} else {
					// Correct the index in the nonpurgeable array stored on the resource post-sort.
					c.nonpurgeableResources[currNP].resourceBase().cacheArrayIndex = currNP
					c.nonpurgeableResources[currNP].resourceBase().timestamp = c.timestamp
					c.timestamp++
					currNP++
				}
			}

			// The above loop ended when we hit the end of one array. Finish the other one.
			for currP < len(sortedPurgeableResources) {
				sortedPurgeableResources[currP].resourceBase().timestamp = c.timestamp
				c.timestamp++
				currP++
			}
			for currNP < len(c.nonpurgeableResources) {
				c.nonpurgeableResources[currNP].resourceBase().cacheArrayIndex = currNP
				c.nonpurgeableResources[currNP].resourceBase().timestamp = c.timestamp
				c.timestamp++
				currNP++
			}

			// Rebuild the queue.
			for _, resource := range sortedPurgeableResources {
				c.purgeableQueue.insert(resource)
			}

			c.validate()
			if count != c.ResourceCount() || c.timestamp != uint32(count) {
				panic("timestamp reset lost resources")
			}
		}
	}
	ts := c.timestamp
	c.timestamp++
	return ts
}

func (c *ResourceCache) isInCache(resource Resource) bool {
	index := resource.resourceBase().cacheArrayIndex
	if index < 0 {
		return false
	}
	if index < c.purgeableQueue.count() && c.purgeableQueue.at(index) == resource {
		return true
	}
	if index < len(c.nonpurgeableResources) && c.nonpurgeableResources[index] == resource {
		return true
	}
	panic("resource index should be -1 or the resource should be in the cache")
}

// validate checks the cache's internal bookkeeping for consistency. It runs only when enabled (tests); any
// inconsistency panics.
func (c *ResourceCache) validate() {
	if !resourceCacheValidation {
		return
	}

	var bytes, budgetedBytes, purgeableBytes uint64
	var budgetedCount, locked, scratch, content int

	check := func(resource Resource) {
		b := resource.resourceBase()
		bytes += b.GpuMemorySize()
		if !b.IsPurgeable() {
			locked++
		}
		switch {
		case b.isUsableAsScratch():
			assertf(!b.uniqueKey.IsValid(), "scratch resource with unique key")
			assertf(b.budgetedType == BudgetedTypeBudgeted, "scratch resource not budgeted")
			assertf(!b.hasRef(), "scratch resource with ref")
			assertf(!b.refsWrappedObjects, "scratch resource wraps backend objects")
			scratch++
			assertf(c.scratchMapHas(resource), "scratch resource missing from scratch map")
		case b.scratchKey.IsValid():
			assertf(b.budgetedType != BudgetedTypeBudgeted || b.uniqueKey.IsValid() || b.hasRef(),
				"budgeted scratch-keyed resource unaccounted for")
			assertf(!b.refsWrappedObjects, "wrapped resource with scratch key")
			assertf(!c.scratchMapHas(resource), "unusable resource in scratch map")
		}
		if b.uniqueKey.IsValid() {
			content++
			assertf(c.uniqueHash[b.uniqueKey.mapKey()] == resource, "unique hash mismatch")
			assertf(b.budgetedType == BudgetedTypeBudgeted || b.refsWrappedObjects,
				"unique key on unbudgeted, unwrapped resource")
		}
		if b.budgetedType == BudgetedTypeBudgeted {
			budgetedCount++
			budgetedBytes += b.GpuMemorySize()
		}
	}

	mapCount := 0
	for _, list := range c.scratchMap {
		for _, resource := range list {
			assertf(resource.resourceBase().isUsableAsScratch(), "unusable resource in scratch map")
			mapCount++
		}
	}
	assertf(mapCount == c.scratchCount, "scratch map count mismatch")

	for i, resource := range c.nonpurgeableResources {
		b := resource.resourceBase()
		assertf(!b.IsPurgeable() || c.newlyPurgeableResourceForValidation == resource,
			"purgeable resource in nonpurgeable array")
		assertf(b.cacheArrayIndex == i, "nonpurgeable index mismatch")
		assertf(!b.destroyed, "destroyed resource in cache")
		check(resource)
	}
	for i := 0; i < c.purgeableQueue.count(); i++ {
		resource := c.purgeableQueue.at(i)
		b := resource.resourceBase()
		assertf(b.IsPurgeable(), "unpurgeable resource in purgeable queue")
		assertf(b.cacheArrayIndex == i, "purgeable index mismatch")
		assertf(!b.destroyed, "destroyed resource in cache")
		check(resource)
		purgeableBytes += b.GpuMemorySize()
	}

	assertf(c.count == c.ResourceCount(), "count mismatch")
	assertf(c.budgetedCount <= c.count, "budgeted count exceeds count")
	assertf(c.budgetedBytes <= c.bytes, "budgeted bytes exceed bytes")
	assertf(bytes == c.bytes, "bytes mismatch")
	assertf(budgetedBytes == c.budgetedBytes, "budgeted bytes mismatch")
	assertf(budgetedCount == c.budgetedCount, "budgeted count mismatch")
	assertf(purgeableBytes == c.purgeableBytes, "purgeable bytes mismatch")
	assertf(content == len(c.uniqueHash), "unique hash count mismatch")
	assertf(scratch == c.scratchCount, "scratch count mismatch")
}

func (c *ResourceCache) scratchMapHas(resource Resource) bool {
	for _, r := range c.scratchMap[resource.resourceBase().scratchKey.mapKey()] {
		if r == resource {
			return true
		}
	}
	return false
}

func assertf(cond bool, format string, args ...any) {
	if !cond {
		panic(fmt.Sprintf(format, args...))
	}
}

// purgeableQueue is a binary min-heap on the resource timestamp that stores each element's heap position in its
// cacheArrayIndex so removal by value is O(lg n).
type purgeableQueue struct {
	items []Resource
}

func purgeCompare(a, b Resource) bool {
	return a.resourceBase().timestamp < b.resourceBase().timestamp
}

func (q *purgeableQueue) count() int { return len(q.items) }

func (q *purgeableQueue) at(i int) Resource { return q.items[i] }

func (q *purgeableQueue) peek() Resource { return q.items[0] }

func (q *purgeableQueue) setItem(i int, r Resource) {
	q.items[i] = r
	r.resourceBase().cacheArrayIndex = i
}

func (q *purgeableQueue) insert(r Resource) {
	q.items = append(q.items, r)
	r.resourceBase().cacheArrayIndex = len(q.items) - 1
	q.percolateUp(len(q.items) - 1)
}

func (q *purgeableQueue) pop() {
	q.removeAt(0)
}

func (q *purgeableQueue) remove(r Resource) {
	idx := r.resourceBase().cacheArrayIndex
	if idx < 0 || idx >= len(q.items) || q.items[idx] != r {
		panic("resource not in purgeable queue")
	}
	q.removeAt(idx)
	r.resourceBase().cacheArrayIndex = -1
}

func (q *purgeableQueue) removeAt(idx int) {
	last := len(q.items) - 1
	tail := q.items[last]
	q.items = q.items[:last]
	if idx == last {
		return
	}
	q.setItem(idx, tail)
	if !q.percolateUp(idx) {
		q.percolateDown(idx)
	}
}

func (q *purgeableQueue) percolateUp(idx int) bool {
	moved := false
	for idx > 0 {
		parent := (idx - 1) / 2
		if !purgeCompare(q.items[idx], q.items[parent]) {
			break
		}
		q.items[idx], q.items[parent] = q.items[parent], q.items[idx]
		q.items[idx].resourceBase().cacheArrayIndex = idx
		q.items[parent].resourceBase().cacheArrayIndex = parent
		idx = parent
		moved = true
	}
	return moved
}

func (q *purgeableQueue) percolateDown(idx int) {
	n := len(q.items)
	for {
		smallest := idx
		if l := 2*idx + 1; l < n && purgeCompare(q.items[l], q.items[smallest]) {
			smallest = l
		}
		if r := 2*idx + 2; r < n && purgeCompare(q.items[r], q.items[smallest]) {
			smallest = r
		}
		if smallest == idx {
			return
		}
		q.items[idx], q.items[smallest] = q.items[smallest], q.items[idx]
		q.items[idx].resourceBase().cacheArrayIndex = idx
		q.items[smallest].resourceBase().cacheArrayIndex = smallest
		idx = smallest
	}
}

// sort orders the array by timestamp (a sorted array satisfies the heap property, so the queue remains valid).
func (q *purgeableQueue) sort() {
	sort.SliceStable(q.items, purgeCompareIdx(q.items))
	for i, r := range q.items {
		r.resourceBase().cacheArrayIndex = i
	}
}

func purgeCompareIdx(items []Resource) func(i, j int) bool {
	return func(i, j int) bool { return purgeCompare(items[i], items[j]) }
}

// countScratchEntriesForKey returns how many resources are currently cached under key (test support).
func (c *ResourceCache) countScratchEntriesForKey(key *ScratchKey) int {
	return len(c.scratchMap[key.mapKey()])
}

// changeTimestamp overrides the cache's internal timestamp counter (test support).
func (c *ResourceCache) changeTimestamp(newTimestamp uint32) { c.timestamp = newTimestamp }
