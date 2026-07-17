// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Backend-independent tests for ResourceCache, driven by a mock resource type. The "number alive" counter decrements
// when a resource is released or abandoned by the cache, since a resource is only deleted once it is both removed from
// the cache and unreffed; the chained-unref of setUnrefWhenDestroyed fires on release/abandon for the same reason. The
// full validate() consistency checks run for every test via TestMain.

package gpu

import (
	"math/rand"
	"os"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	resourceCacheValidation = true
	os.Exit(m.Run())
}

const testResourceDefaultSize = 100

// simulatedProperty distinctly categorizes a scratch resource for scratch-key matching in tests.
type simulatedProperty uint32

const (
	simulatedPropertyA simulatedProperty = 0
	simulatedPropertyB simulatedProperty = 1
)

var testScratchKeyType = GenerateScratchKeyResourceType()

const testScratchKeyFieldCnt = 6

func computeTestScratchKey(property simulatedProperty, key *ScratchKey) {
	b := ScratchKeyBuilder(key, testScratchKeyType, testScratchKeyFieldCnt)
	s := b.Slice()
	for i := range s {
		s[i] = uint32(i) + uint32(property)
	}
	b.Finish()
}

func expectedTestScratchKeySize() int { return 4 * (testScratchKeyFieldCnt + keyMetaDataCount) }

type testResource struct {
	alive   *int
	toUnref *testResource
	ResourceBase
	size      uint64
	property  simulatedProperty
	isScratch bool
}

func (t *testResource) OnGpuMemorySize() uint64 { return t.size }

func (t *testResource) ResourceType() string { return "Test" }

func (t *testResource) ComputeScratchKey(key *ScratchKey) {
	if t.isScratch {
		computeTestScratchKey(t.property, key)
	}
}

func (t *testResource) died() {
	*t.alive--
	if t.toUnref != nil {
		t.toUnref.Unref()
		t.toUnref = nil
	}
}

func (t *testResource) OnRelease() { t.died() }
func (t *testResource) OnAbandon() { t.died() }

// setUnrefWhenDestroyed lets the caller transfer a ref on other (taken via Ref) that is released when this resource is
// destroyed or the link is replaced.
func (t *testResource) setUnrefWhenDestroyed(other *testResource) {
	if t.toUnref != nil {
		t.toUnref.Unref()
	}
	t.toUnref = other
}

// mock is a cache with a byte limit and an alive counter, used to drive the tests below.
type mock struct {
	cache *ResourceCache
	alive int
}

func newMock(maxBytes uint64) *mock {
	m := &mock{cache: NewResourceCache()}
	m.cache.SetLimit(maxBytes)
	return m
}

func (m *mock) newResource(budgeted Budgeted, size uint64) *testResource {
	r := &testResource{alive: &m.alive, size: size, property: simulatedPropertyA}
	m.alive++
	r.RegisterWithCache(m.cache, r, budgeted, "")
	return r
}

func (m *mock) newScratch(budgeted Budgeted, property simulatedProperty, size uint64) *testResource {
	r := &testResource{alive: &m.alive, size: size, property: property, isScratch: true}
	m.alive++
	r.RegisterWithCache(m.cache, r, budgeted, "")
	return r
}

func (m *mock) newWrapped(cacheable WrapCacheable, size uint64) *testResource {
	r := &testResource{alive: &m.alive, size: size, property: simulatedPropertyA}
	m.alive++
	r.RegisterWithCacheWrapped(m.cache, r, cacheable, "")
	return r
}

var uniqueKeyTestDomain = GenerateUniqueKeyDomain()

func makeUniqueKey(key *UniqueKey, data uint32) {
	b := UniqueKeyBuilder(key, uniqueKeyTestDomain, 1, "")
	b.Slice()[0] = data
	b.Finish()
}

func check(t *testing.T, cond bool, format string, args ...any) {
	t.Helper()
	if !cond {
		t.Errorf(format, args...)
	}
}

// TestResourceCacheNoKey verifies that keyless resources survive a purge while still referenced, but are deleted
// immediately upon Unref since they have no scratch or unique key to keep them in the cache.
func TestResourceCacheNoKey(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	// Create a bunch of resources with no keys.
	a := m.newResource(BudgetedYes, 11)
	b := m.newResource(BudgetedYes, 12)
	c := m.newResource(BudgetedYes, 13)
	d := m.newResource(BudgetedYes, 14)

	check(t, m.alive == 4, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 4, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 11+12+13+14, "bytes = %d", cache.ResourceBytes())

	// Should be safe to purge without deleting the resources since we still have refs.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 4, "alive = %d", m.alive)

	// Since the resources have neither unique nor scratch keys, delete immediately upon unref.
	a.Unref()
	check(t, m.alive == 3, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 3, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 12+13+14, "bytes = %d", cache.ResourceBytes())

	c.Unref()
	check(t, m.alive == 2, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 12+14, "bytes = %d", cache.ResourceBytes())

	d.Unref()
	check(t, m.alive == 1, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 12, "bytes = %d", cache.ResourceBytes())

	b.Unref()
	check(t, m.alive == 0, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 0, "bytes = %d", cache.ResourceBytes())
}

// TestResourceCachePurgeUnlocked verifies that PurgeUnlockedResources leaves referenced resources untouched, that
// PurgeScratchResourcesOnly removes only the resources without a unique key, and that a subsequent PurgeAllResources
// removes the remaining uniquely keyed ones.
func TestResourceCachePurgeUnlocked(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	// Create two resources w/ a unique key and two w/o but all of which have scratch keys.
	a := m.newScratch(BudgetedYes, simulatedPropertyA, 11)
	var uniqueKey UniqueKey
	makeUniqueKey(&uniqueKey, 0)
	b := m.newScratch(BudgetedYes, simulatedPropertyA, 12)
	b.SetUniqueKey(&uniqueKey)
	c := m.newScratch(BudgetedYes, simulatedPropertyA, 13)
	var uniqueKey2 UniqueKey
	makeUniqueKey(&uniqueKey2, 1)
	d := m.newScratch(BudgetedYes, simulatedPropertyA, 14)
	d.SetUniqueKey(&uniqueKey2)

	check(t, m.alive == 4, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 4, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 11+12+13+14, "bytes = %d", cache.ResourceBytes())

	// Should be safe to purge without deleting the resources since we still have refs.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 4, "alive = %d", m.alive)

	// Unref them all. Since they all have keys they should remain in the cache.
	a.Unref()
	b.Unref()
	c.Unref()
	d.Unref()
	check(t, m.alive == 4, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 4, "count = %d", cache.ResourceCount())

	// Purge only the two scratch resources.
	cache.PurgeUnlockedResources(PurgeScratchResourcesOnly)
	check(t, m.alive == 2, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 12+14, "bytes = %d", cache.ResourceBytes())

	// Purge the uniquely keyed resources.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 0, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 0, "bytes = %d", cache.ResourceBytes())
}

// TestResourceCachePurgeCommandBufferUsage verifies that a resource with an outstanding command-buffer usage ref
// survives PurgeUnlockedResources even after its regular ref is dropped, and is only purged once both the ref and the
// command-buffer usage are released.
func TestResourceCachePurgeCommandBufferUsage(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	a := m.newScratch(BudgetedYes, simulatedPropertyA, 11)
	b := m.newScratch(BudgetedYes, simulatedPropertyA, 12)
	check(t, m.alive == 2, "alive = %d", m.alive)

	cache.PurgeUnlockedResources(PurgeScratchResourcesOnly)
	check(t, m.alive == 2, "alive = %d", m.alive)

	// Add command buffer usages to all resources.
	a.RefCommandBuffer()
	b.RefCommandBuffer()
	cache.PurgeUnlockedResources(PurgeScratchResourcesOnly)
	check(t, m.alive == 2, "alive = %d", m.alive)

	// Unref the first resource: still held by its command buffer usage.
	a.Unref()
	check(t, m.alive == 2, "alive = %d", m.alive)
	cache.PurgeUnlockedResources(PurgeScratchResourcesOnly)
	check(t, m.alive == 2, "alive = %d", m.alive)

	// Remove command buffer usages.
	a.UnrefCommandBuffer()
	b.UnrefCommandBuffer()
	check(t, m.alive == 2, "alive = %d", m.alive)

	// This purge should remove the first resource: no refs or command buffer usages remain.
	cache.PurgeUnlockedResources(PurgeScratchResourcesOnly)
	check(t, m.alive == 1, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 12, "bytes = %d", cache.ResourceBytes())

	b.Unref()
	check(t, m.alive == 1, "alive = %d", m.alive)
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 0, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
}

// TestResourceCacheBudgeting verifies budgeted vs. unbudgeted and wrapped-cacheable vs. wrapped-uncacheable accounting:
// which resources count toward BudgetedResourceCount/Bytes, when unreffing frees a resource immediately vs. leaves it
// purgeable, and how removing a unique key on a wrapped resource affects its lifetime.
func TestResourceCacheBudgeting(t *testing.T) {
	m := newMock(300)
	cache := m.cache

	var uniqueKey UniqueKey
	makeUniqueKey(&uniqueKey, 0)

	// Create a scratch, a unique, and wrapped resources.
	scratch := m.newScratch(BudgetedYes, simulatedPropertyB, 10)
	unique := m.newResource(BudgetedYes, 11)
	unique.SetUniqueKey(&uniqueKey)
	wrappedCacheable := m.newWrapped(WrapCacheableYes, 12)
	wrappedUncacheable := m.newWrapped(WrapCacheableNo, 13)
	unbudgeted := m.newResource(BudgetedNo, 14)

	// Make sure we can add a unique key to the wrapped resources.
	var uniqueKey2, uniqueKey3 UniqueKey
	makeUniqueKey(&uniqueKey2, 1)
	makeUniqueKey(&uniqueKey3, 2)
	wrappedCacheable.SetUniqueKey(&uniqueKey2)
	wrappedUncacheable.SetUniqueKey(&uniqueKey3)
	wrappedCacheableViaKey := cache.FindAndRefUniqueResource(&uniqueKey2)
	check(t, wrappedCacheableViaKey != nil, "wrapped cacheable not found by key")
	wrappedUncacheableViaKey := cache.FindAndRefUniqueResource(&uniqueKey3)
	check(t, wrappedUncacheableViaKey != nil, "wrapped uncacheable not found by key")

	// Remove the extra refs we just added.
	wrappedCacheableViaKey.resourceBase().Unref()
	wrappedUncacheableViaKey.resourceBase().Unref()

	// Make sure sizes are as we expect.
	check(t, cache.ResourceCount() == 5, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 10+11+12+13+14, "bytes = %d", cache.ResourceBytes())
	check(t, cache.BudgetedResourceCount() == 2, "budgeted count = %d", cache.BudgetedResourceCount())
	check(t, cache.BudgetedResourceBytes() == 10+11, "budgeted bytes = %d", cache.BudgetedResourceBytes())
	check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())

	// Our refs mean that the resources are non purgeable.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, cache.ResourceCount() == 5, "count = %d", cache.ResourceCount())
	check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())

	// Unreffing the cacheable wrapped resource with a unique key shouldn't free it right away. However, unreffing the
	// uncacheable wrapped resource should free it.
	wrappedCacheable.Unref()
	wrappedUncacheable.Unref()
	check(t, cache.ResourceCount() == 4, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 10+11+12+14, "bytes = %d", cache.ResourceBytes())
	check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())

	// Now try freeing the budgeted resources first.
	wrappedUncacheable = m.newWrapped(WrapCacheableNo, testResourceDefaultSize)
	unique.Unref()
	check(t, cache.PurgeableBytes() == 11, "purgeable = %d", cache.PurgeableBytes())
	// This frees 'unique' but not wrappedCacheable, which requires its key removed to be freed.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, cache.ResourceCount() == 4, "count = %d", cache.ResourceCount())

	wrappedCacheableViaKey = cache.FindAndRefUniqueResource(&uniqueKey2)
	check(t, wrappedCacheableViaKey != nil, "wrapped cacheable not found by key")
	if wrappedCacheableViaKey != nil {
		wrappedCacheableViaKey.resourceBase().RemoveUniqueKey()
		wrappedCacheable.Unref()
	}
	// Removing the key on a wrapped cacheable resource should immediately delete it.
	check(t, cache.ResourceCount() == 3, "count = %d", cache.ResourceCount())

	wrappedCacheable = m.newWrapped(WrapCacheableYes, testResourceDefaultSize)
	check(t, cache.BudgetedResourceCount() == 1, "budgeted count = %d", cache.BudgetedResourceCount())
	check(t, cache.BudgetedResourceBytes() == 10, "budgeted bytes = %d", cache.BudgetedResourceBytes())
	check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())

	scratch.Unref()
	check(t, cache.PurgeableBytes() == 10, "purgeable = %d", cache.PurgeableBytes())
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, cache.ResourceCount() == 3, "count = %d", cache.ResourceCount())
	check(t, cache.BudgetedResourceCount() == 0, "budgeted count = %d", cache.BudgetedResourceCount())
	check(t, cache.BudgetedResourceBytes() == 0, "budgeted bytes = %d", cache.BudgetedResourceBytes())
	check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())

	// Unreffing the wrapped resources (with no unique key) should free them right away.
	wrappedUncacheable.Unref()
	wrappedCacheable.Unref()
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 14, "bytes = %d", cache.ResourceBytes())

	unbudgeted.Unref()
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 0, "bytes = %d", cache.ResourceBytes())
}

// TestResourceCacheUnbudgeted verifies that unbudgeted and wrapped resources are tracked in ResourceCount/ResourceBytes
// but excluded from BudgetedResourceCount/Bytes, and that inserting a large unbudgeted or wrapped resource does not
// evict any other resource.
func TestResourceCacheUnbudgeted(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	var uniqueKey UniqueKey
	makeUniqueKey(&uniqueKey, 0)

	// A large uncached or wrapped resource shouldn't evict anything.
	scratch := m.newScratch(BudgetedYes, simulatedPropertyB, 10)
	scratch.Unref()
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 10, "bytes = %d", cache.ResourceBytes())
	check(t, cache.BudgetedResourceCount() == 1, "budgeted count = %d", cache.BudgetedResourceCount())
	check(t, cache.BudgetedResourceBytes() == 10, "budgeted bytes = %d", cache.BudgetedResourceBytes())
	check(t, cache.PurgeableBytes() == 10, "purgeable = %d", cache.PurgeableBytes())

	unique := m.newResource(BudgetedYes, 11)
	unique.SetUniqueKey(&uniqueKey)
	unique.Unref()
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 21, "bytes = %d", cache.ResourceBytes())
	check(t, cache.PurgeableBytes() == 21, "purgeable = %d", cache.PurgeableBytes())

	large := 2 * cache.ResourceBytes()
	unbudgeted := m.newResource(BudgetedNo, large)
	check(t, cache.ResourceCount() == 3, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 21+large, "bytes = %d", cache.ResourceBytes())
	check(t, cache.BudgetedResourceCount() == 2, "budgeted count = %d", cache.BudgetedResourceCount())
	check(t, cache.BudgetedResourceBytes() == 21, "budgeted bytes = %d", cache.BudgetedResourceBytes())
	check(t, cache.PurgeableBytes() == 21, "purgeable = %d", cache.PurgeableBytes())

	unbudgeted.Unref()
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 21, "bytes = %d", cache.ResourceBytes())

	wrapped := m.newWrapped(WrapCacheableYes, large)
	check(t, cache.ResourceCount() == 3, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 21+large, "bytes = %d", cache.ResourceBytes())
	check(t, cache.BudgetedResourceCount() == 2, "budgeted count = %d", cache.BudgetedResourceCount())
	check(t, cache.BudgetedResourceBytes() == 21, "budgeted bytes = %d", cache.BudgetedResourceBytes())
	check(t, cache.PurgeableBytes() == 21, "purgeable = %d", cache.PurgeableBytes())

	wrapped.Unref()
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 21, "bytes = %d", cache.ResourceBytes())

	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 0, "bytes = %d", cache.ResourceBytes())
	check(t, cache.BudgetedResourceCount() == 0, "budgeted count = %d", cache.BudgetedResourceCount())
	check(t, cache.BudgetedResourceBytes() == 0, "budgeted bytes = %d", cache.BudgetedResourceBytes())
	check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())
}

// TestResourceCacheUnbudgetedToScratch verifies that an unbudgeted scratch resource is not discoverable via
// FindAndRefScratchResource while referenced, becomes discoverable and budgeted once unreferenced, and that
// MakeUnbudgeted or RemoveScratchKey correctly reverse or drop that scratch availability.
func TestResourceCacheUnbudgetedToScratch(t *testing.T) {
	m := newMock(300)
	cache := m.cache

	resource := m.newScratch(BudgetedNo, simulatedPropertyA, testResourceDefaultSize)
	var key ScratchKey
	computeTestScratchKey(simulatedPropertyA, &key)

	size := resource.GpuMemorySize()
	for i := 0; i < 2; i++ {
		// Since this resource is unbudgeted, it should not be reachable as scratch.
		check(t, resource.ScratchKey().Equal(&key), "scratch key mismatch")
		check(t, !resource.isScratchRes(), "should not be scratch")
		check(t, resource.BudgetedType() == BudgetedTypeUnbudgetedUncacheable, "budget type = %v",
			resource.BudgetedType())
		check(t, cache.FindAndRefScratchResource(&key) == nil, "should not find unbudgeted as scratch")
		check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
		check(t, cache.ResourceBytes() == size, "bytes = %d", cache.ResourceBytes())
		check(t, cache.BudgetedResourceCount() == 0, "budgeted count = %d", cache.BudgetedResourceCount())
		check(t, cache.BudgetedResourceBytes() == 0, "budgeted bytes = %d", cache.BudgetedResourceBytes())
		check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())

		// Once it is unrefed, it should become available as scratch.
		resource.Unref()
		check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
		check(t, cache.BudgetedResourceCount() == 1, "budgeted count = %d", cache.BudgetedResourceCount())
		check(t, cache.BudgetedResourceBytes() == size, "budgeted bytes = %d", cache.BudgetedResourceBytes())
		check(t, cache.PurgeableBytes() == size, "purgeable = %d", cache.PurgeableBytes())
		found := cache.FindAndRefScratchResource(&key)
		if found == nil {
			t.Fatal("should find as scratch after unref")
		}
		resource = found.(*testResource)
		check(t, resource.ScratchKey().Equal(&key), "scratch key mismatch")
		check(t, resource.isScratchRes(), "should be scratch")
		check(t, resource.BudgetedType() == BudgetedTypeBudgeted, "budget type = %v",
			resource.BudgetedType())

		if i == 0 {
			// If made unbudgeted, it should return to the original state: ref'ed and unbudgeted.
			resource.MakeUnbudgeted()
		} else {
			// The second time around, try removing the scratch key.
			resource.RemoveScratchKey()
			check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
			check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())
			check(t, !resource.ScratchKey().IsValid(), "scratch key should be removed")
			check(t, !resource.isScratchRes(), "should not be scratch")
			check(t, resource.BudgetedType() == BudgetedTypeBudgeted, "budget type = %v",
				resource.BudgetedType())

			// Now when it is unrefed it should die since it has no key.
			resource.Unref()
			check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
			check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())
		}
	}
}

// TestResourceCacheScratchBackingReuse verifies the reclaimed-backing free list: when a scratch-map key's list empties
// and the key is deleted from the map, its []Resource backing is reclaimed — cleared, so it pins no Resource — and
// reused by the next scratchInsert that (re)creates a key, so the steady-state recycle of a scratch resource (e.g. a
// dynamic buffer freed and re-acquired every frame) does not re-grow its list from nil each cycle. The map contents,
// scratch count, and eviction order are unchanged; validate() runs on every cache op via TestMain. Backing identity is
// asserted structurally (the reused list aliases the reclaimed array), and the steady-state cycle is checked to be
// allocation-free.
func TestResourceCacheScratchBackingReuse(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	resource := m.newScratch(BudgetedYes, simulatedPropertyA, 100)
	var key ScratchKey
	computeTestScratchKey(simulatedPropertyA, &key)
	mk := key.mapKey()

	// Its main ref dropping to zero moves it into the scratch map.
	resource.Unref()
	check(t, len(cache.scratchMap[mk]) == 1, "scratch list len = %d, want 1", len(cache.scratchMap[mk]))
	check(t, len(cache.scratchBackings) == 0, "no backing reclaimed yet, got %d",
		len(cache.scratchBackings))

	// Popping it empties and deletes the key, reclaiming its now-empty backing.
	found := cache.FindAndRefScratchResource(&key)
	if found == nil {
		t.Fatal("should find as scratch after unref")
	}
	if _, present := cache.scratchMap[mk]; present {
		t.Fatal("emptied key should be deleted from the scratch map")
	}
	if len(cache.scratchBackings) != 1 {
		t.Fatalf("emptied backing not reclaimed: scratchBackings = %d", len(cache.scratchBackings))
	}
	// Leak safety: the reclaimed backing pins no Resource across its full capacity (append leaves the removed element
	// behind; recycleScratchBacking clears it).
	reclaimed := cache.scratchBackings[0]
	full := reclaimed[:cap(reclaimed)]
	for i := range full {
		if full[i] != nil {
			t.Fatalf("reclaimed backing pins a Resource at index %d", i)
		}
	}
	reclaimedFirst := &reclaimed[:1][0]

	// Re-inserting reuses the reclaimed backing rather than allocating a new one.
	resource = found.(*testResource)
	resource.Unref()
	if len(cache.scratchBackings) != 0 {
		t.Fatalf("reclaimed backing not consumed on re-insert: scratchBackings = %d",
			len(cache.scratchBackings))
	}
	list := cache.scratchMap[mk]
	if len(list) != 1 {
		t.Fatalf("re-inserted list len = %d, want 1", len(list))
	}
	if &list[0] != reclaimedFirst {
		t.Fatal("scratchInsert did not reuse the reclaimed backing array")
	}

	// The steady-state find/reinsert cycle allocates nothing for the scratch list backing.
	if allocs := testing.AllocsPerRun(50, func() {
		got := cache.FindAndRefScratchResource(&key)
		got.(*testResource).Unref()
	}); allocs != 0 {
		t.Errorf("steady-state scratch recycle allocates %v/op, want 0", allocs)
	}
}

// TestResourceCacheDuplicateScratchKey verifies that two resources sharing the same scratch key are excluded from the
// scratch map while referenced, both become countable scratch entries once unreferenced, and a lookup with a different
// key never matches either.
func TestResourceCacheDuplicateScratchKey(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	a := m.newScratch(BudgetedYes, simulatedPropertyB, 11)
	b := m.newScratch(BudgetedYes, simulatedPropertyB, 12)

	// Check for negative case consistency.
	var scratchKey1 ScratchKey
	computeTestScratchKey(simulatedPropertyA, &scratchKey1)
	check(t, cache.FindAndRefScratchResource(&scratchKey1) == nil, "wrong-key find should fail")

	var scratchKey ScratchKey
	computeTestScratchKey(simulatedPropertyB, &scratchKey)

	// As long as there are outstanding refs on the resources they are not in the scratch map.
	check(t, cache.countScratchEntriesForKey(&scratchKey) == 0, "scratch entries = %d",
		cache.countScratchEntriesForKey(&scratchKey))
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, cache.ResourceBytes() == 11+12, "bytes = %d", cache.ResourceBytes())

	// Our refs mean that the resources are non purgeable.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 2, "alive = %d", m.alive)

	// Unref but don't purge.
	a.Unref()
	b.Unref()
	check(t, m.alive == 2, "alive = %d", m.alive)
	check(t, cache.countScratchEntriesForKey(&scratchKey) == 2, "scratch entries = %d",
		cache.countScratchEntriesForKey(&scratchKey))

	// Purge again. This time resources should be purgeable.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 0, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
	check(t, cache.countScratchEntriesForKey(&scratchKey) == 0, "scratch entries = %d",
		cache.countScratchEntriesForKey(&scratchKey))
}

// TestResourceCacheRemoveScratchKey verifies that RemoveScratchKey drops a resource from scratch lookup without
// destroying it immediately, and that the resource is then deleted on its next Unref since it is no longer reachable by
// any key.
func TestResourceCacheRemoveScratchKey(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	a := m.newScratch(BudgetedYes, simulatedPropertyB, testResourceDefaultSize)
	b := m.newScratch(BudgetedYes, simulatedPropertyB, testResourceDefaultSize)
	a.Unref()
	b.Unref()

	var scratchKey ScratchKey
	// Ensure that scratch key lookup is correct for the negative case.
	computeTestScratchKey(simulatedPropertyA, &scratchKey)
	check(t, cache.FindAndRefScratchResource(&scratchKey) == nil, "wrong-key find should fail")

	computeTestScratchKey(simulatedPropertyB, &scratchKey)
	check(t, m.alive == 2, "alive = %d", m.alive)
	check(t, cache.countScratchEntriesForKey(&scratchKey) == 2, "scratch entries")

	// Find the first resource and remove its scratch key.
	find := cache.FindAndRefScratchResource(&scratchKey)
	find.resourceBase().RemoveScratchKey()
	// It's still alive, but not cached by scratch key anymore.
	check(t, m.alive == 2, "alive = %d", m.alive)
	check(t, cache.countScratchEntriesForKey(&scratchKey) == 1, "scratch entries")
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())

	// The cache should immediately delete it when it's unrefed since it isn't accessible.
	find.resourceBase().Unref()
	check(t, m.alive == 1, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())

	// Repeat for the second resource.
	find = cache.FindAndRefScratchResource(&scratchKey)
	find.resourceBase().RemoveScratchKey()
	check(t, m.alive == 1, "alive = %d", m.alive)
	check(t, cache.countScratchEntriesForKey(&scratchKey) == 0, "scratch entries")

	// Should be able to call this multiple times with no problem.
	find.resourceBase().RemoveScratchKey()
	check(t, cache.countScratchEntriesForKey(&scratchKey) == 0, "scratch entries")

	find.resourceBase().Unref()
	check(t, m.alive == 0, "alive = %d", m.alive)
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
}

// TestResourceCacheScratchKeyConsistency verifies ScratchKey equality and copy semantics — copies compare equal and
// report the same size, distinct keys compare unequal — and that FindAndRefScratchResource lookups follow key equality
// rather than key identity, returning either of two resources that share a key.
func TestResourceCacheScratchKeyConsistency(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	a := m.newScratch(BudgetedYes, simulatedPropertyB, testResourceDefaultSize)
	b := m.newScratch(BudgetedYes, simulatedPropertyB, testResourceDefaultSize)
	a.Unref()
	b.Unref()

	// Ensure that scratch key comparison and assignment is consistent.
	var scratchKey1, scratchKey2, scratchKey ScratchKey
	computeTestScratchKey(simulatedPropertyA, &scratchKey1)
	computeTestScratchKey(simulatedPropertyB, &scratchKey2)
	check(t, scratchKey1.Size() == expectedTestScratchKeySize(), "key size = %d", scratchKey1.Size())
	check(t, !scratchKey1.Equal(&scratchKey2), "keys should differ")
	check(t, !scratchKey2.Equal(&scratchKey1), "keys should differ")
	scratchKey = scratchKey1
	check(t, scratchKey.Size() == expectedTestScratchKeySize(), "key size = %d", scratchKey.Size())
	check(t, scratchKey1.Equal(&scratchKey) && scratchKey.Equal(&scratchKey1), "copies should match")
	check(t, !scratchKey2.Equal(&scratchKey) && !scratchKey.Equal(&scratchKey2), "keys should differ")
	scratchKey = scratchKey2
	check(t, scratchKey2.Equal(&scratchKey) && scratchKey.Equal(&scratchKey2), "copies should match")

	// Ensure that scratch key lookup is correct for the negative case.
	computeTestScratchKey(simulatedPropertyA, &scratchKey)
	check(t, cache.FindAndRefScratchResource(&scratchKey) == nil, "wrong-key find should fail")

	// Find the first resource with a scratch key and a copy of a scratch key.
	computeTestScratchKey(simulatedPropertyB, &scratchKey)
	find := cache.FindAndRefScratchResource(&scratchKey)
	check(t, find != nil, "find failed")
	find.resourceBase().Unref()

	scratchKey2 = scratchKey
	find = cache.FindAndRefScratchResource(&scratchKey2)
	check(t, find != nil, "find failed")
	check(t, find == Resource(a) || find == Resource(b), "found something else")

	find2 := cache.FindAndRefScratchResource(&scratchKey2)
	check(t, find2 != nil, "second find failed")
	check(t, find2 == Resource(a) || find2 == Resource(b), "found something else")
	check(t, find2 != find, "found the same resource twice")
	find2.resourceBase().Unref()
	find.resourceBase().Unref()
}

// TestResourceCacheDuplicateUniqueKey verifies that setting the same UniqueKey on a second resource evicts the first
// resource's cache entry for that key (purging it once unreferenced), that redundantly re-setting a resource's own key
// is a no-op, that RemoveUniqueKey drops a resource from lookup, and that a key's custom data is preserved through
// registration and lookup.
func TestResourceCacheDuplicateUniqueKey(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	var key UniqueKey
	makeUniqueKey(&key, 0)

	// Create two resources that we will attempt to register with the same unique key.
	a := m.newResource(BudgetedYes, 11)
	a.SetUniqueKey(&key)
	check(t, cache.FindAndRefUniqueResource(&key) == Resource(a), "find != a")
	a.Unref()

	// Make sure that redundantly setting a's key works.
	a.SetUniqueKey(&key)
	check(t, cache.FindAndRefUniqueResource(&key) == Resource(a), "find != a")
	a.Unref()
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, m.alive == 1, "alive = %d", m.alive)

	// Create resource b and set the same key. It should replace a's unique key cache entry.
	b := m.newResource(BudgetedYes, 12)
	b.SetUniqueKey(&key)
	check(t, cache.FindAndRefUniqueResource(&key) == Resource(b), "find != b")
	b.Unref()

	// Still have two resources because a is still reffed.
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, m.alive == 2, "alive = %d", m.alive)

	a.Unref()
	// Now a should be gone.
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, m.alive == 1, "alive = %d", m.alive)

	// Now replace b with c, but make sure c can start with one unique key and change it to b's key. Also make b be
	// unreffed when replacement occurs.
	b.Unref()
	c := m.newResource(BudgetedYes, 13)
	var differentKey UniqueKey
	makeUniqueKey(&differentKey, 1)
	c.SetUniqueKey(&differentKey)
	check(t, cache.ResourceCount() == 2, "count = %d", cache.ResourceCount())
	check(t, m.alive == 2, "alive = %d", m.alive)
	// c replaces b and b should be immediately purged.
	c.SetUniqueKey(&key)
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, m.alive == 1, "alive = %d", m.alive)

	// c shouldn't be purged because it is ref'ed.
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())

	// Drop the ref on c, it should be kept alive because it has a unique key.
	c.Unref()
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, m.alive == 1, "alive = %d", m.alive)

	// Verify that we can find c, then remove its unique key. It should get purged immediately.
	check(t, cache.FindAndRefUniqueResource(&key) == Resource(c), "find != c")
	c.RemoveUniqueKey()
	c.Unref()
	check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
	check(t, m.alive == 0, "alive = %d", m.alive)

	// Verify that the custom data on a key survives the cache.
	{
		var key2 UniqueKey
		makeUniqueKey(&key2, 0)
		d := m.newResource(BudgetedYes, testResourceDefaultSize)
		key2.SetCustomData([]byte{4, 1, 3, 2})
		d.SetUniqueKey(&key2)
		d.Unref()
	}
	var key3 UniqueKey
	makeUniqueKey(&key3, 0)
	d2 := cache.FindAndRefUniqueResource(&key3)
	if d2 == nil {
		t.Fatal("find by fresh key failed")
	}
	got := d2.resourceBase().UniqueKey().CustomData()
	check(t, len(got) == 4 && got[0] == 4 && got[1] == 1 && got[2] == 3 && got[3] == 2,
		"custom data = %v", got)
	d2.resourceBase().Unref()
}

// TestResourceCacheChainedPurge verifies that two resources holding refs on each other via setUnrefWhenDestroyed form a
// cycle that keeps both alive across a purge, and that breaking the cycle lets a subsequent purge collect both.
func TestResourceCacheChainedPurge(t *testing.T) {
	m := newMock(30000)
	cache := m.cache

	var key1, key2 UniqueKey
	makeUniqueKey(&key1, 1)
	makeUniqueKey(&key2, 2)

	a := m.newResource(BudgetedYes, testResourceDefaultSize)
	b := m.newResource(BudgetedYes, testResourceDefaultSize)
	a.SetUniqueKey(&key1)
	b.SetUniqueKey(&key2)

	// Make a cycle: each holds a ref on the other, released when the counterpart is destroyed.
	b.Ref()
	a.setUnrefWhenDestroyed(b)
	a.Ref()
	b.setUnrefWhenDestroyed(a)

	check(t, m.alive == 2, "alive = %d", m.alive)

	a.Unref()
	b.Unref()

	check(t, m.alive == 2, "alive = %d", m.alive)
	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 2, "alive = %d", m.alive)

	// Break the cycle.
	a.setUnrefWhenDestroyed(nil)
	check(t, m.alive == 2, "alive = %d", m.alive)

	cache.PurgeUnlockedResources(PurgeAllResources)
	check(t, m.alive == 0, "alive = %d", m.alive)
}

// TestResourceCacheTimestampWrap verifies that purge-eligibility ordering by internal timestamp remains correct across
// many insertions even when the cache's timestamp counter wraps around uint32 max.
func TestResourceCacheTimestampWrap(t *testing.T) {
	const count = 50
	const lockedFreq = 8

	random := rand.New(rand.NewSource(12345))
	for i := 0; i < 2*count; i++ {
		m := newMock(0) // always over budget
		cache := m.cache

		// Pick a random number of resources to add before the timestamp will wrap.
		cache.changeTimestamp(^uint32(0) - uint32(random.Intn(count+1)))

		var shouldPurgeIdxs []int
		purgeableCnt := 0
		var resourcesToUnref []*testResource

		// Add resources, holding onto some at random so we have a mix of purgeable and not.
		for j := 0; j < count; j++ {
			var key UniqueKey
			makeUniqueKey(&key, uint32(j))
			r := m.newResource(BudgetedYes, testResourceDefaultSize)
			r.SetUniqueKey(&key)
			if random.Intn(lockedFreq) != 0 {
				// Make this purgeable.
				r.Unref()
				purgeableCnt++
				if purgeableCnt <= count {
					shouldPurgeIdxs = append(shouldPurgeIdxs, j)
				}
			} else {
				resourcesToUnref = append(resourcesToUnref, r)
			}
		}

		// Verify that the correct resources were purged.
		currShouldPurgeIdx := 0
		for j := 0; j < count; j++ {
			var key UniqueKey
			makeUniqueKey(&key, uint32(j))
			res := cache.FindAndRefUniqueResource(&key)
			if currShouldPurgeIdx < len(shouldPurgeIdxs) && shouldPurgeIdxs[currShouldPurgeIdx] == j {
				currShouldPurgeIdx++
				check(t, res == nil, "iteration %d: resource %d should have been purged", i, j)
			} else {
				check(t, res != nil, "iteration %d: resource %d should be present", i, j)
			}
			if res != nil {
				res.resourceBase().Unref()
			}
		}
		for _, r := range resourcesToUnref {
			r.Unref()
		}
	}
}

// nowish returns a time point strictly after every prior time observation, without needing an actual sleep between
// observations.
func nowish() time.Time {
	last := time.Now()
	for {
		if n := time.Now(); n.After(last) {
			return n
		}
	}
}

// TestResourceCacheTimePurge verifies that PurgeResourcesNotUsedSince purges resources in oldest-first order relative
// to recorded time points, correctly skips resources still referenced, and honors PurgeScratchResourcesOnly by leaving
// uniquely keyed resources untouched.
func TestResourceCacheTimePurge(t *testing.T) {
	m := newMock(1000000)
	cache := m.cache

	counts := []int{1, 10, 256}
	for _, cnt := range counts {
		timeStamps := make([]time.Time, cnt)

		// Insert resources and get time points between each addition.
		for i := 0; i < cnt; i++ {
			r := m.newResource(BudgetedYes, testResourceDefaultSize)
			var k UniqueKey
			makeUniqueKey(&k, uint32(i))
			r.SetUniqueKey(&k)
			r.Unref()
			timeStamps[i] = nowish()
		}

		// Purge based on the time points between resource additions. Each purge should remove the oldest resource.
		for i := 0; i < cnt; i++ {
			cache.PurgeResourcesNotUsedSince(timeStamps[i], PurgeAllResources)
			check(t, cache.ResourceCount() == cnt-i-1, "cnt %d: count = %d after purge %d", cnt,
				cache.ResourceCount(), i)
			for j := 0; j <= i; j++ {
				var k UniqueKey
				makeUniqueKey(&k, uint32(j))
				check(t, cache.FindAndRefUniqueResource(&k) == nil, "resource %d should be gone", j)
			}
		}
		check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
		cache.PurgeUnlockedResources(PurgeAllResources)

		// Similar, but leave refs on some resources to prevent them from being purged.
		{
			refedResources := make([]*testResource, 0, cnt/2)
			for i := 0; i < cnt; i++ {
				r := m.newResource(BudgetedYes, testResourceDefaultSize)
				var k UniqueKey
				makeUniqueKey(&k, uint32(i))
				r.SetUniqueKey(&k)
				// Leave a ref on every other resource, beginning with the first.
				if i&1 != 0 {
					refedResources = append(refedResources, r)
				} else {
					r.Unref()
				}
				timeStamps[i] = nowish()
			}
			for i := 0; i < cnt; i++ {
				// Should get a resource purged every other frame.
				cache.PurgeResourcesNotUsedSince(timeStamps[i], PurgeAllResources)
				check(t, cache.ResourceCount() == cnt-i/2-1, "count = %d", cache.ResourceCount())
			}
			// Unref all the resources that we kept refs on in the first loop.
			for i, r := range refedResources {
				r.Unref()
				cache.PurgeResourcesNotUsedSince(nowish(), PurgeAllResources)
				check(t, cache.ResourceCount() == cnt/2-i-1, "count = %d", cache.ResourceCount())
			}
			cache.PurgeUnlockedResources(PurgeAllResources)
		}
		check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())

		// Alternate adding scratch and uniquely keyed resources, then purge old scratch only.
		for i := 0; i < cnt; i++ {
			isScratch := i%2 == 0
			var r *testResource
			if isScratch {
				r = m.newScratch(BudgetedYes, simulatedPropertyA, testResourceDefaultSize)
			} else {
				r = m.newResource(BudgetedYes, testResourceDefaultSize)
				var k UniqueKey
				makeUniqueKey(&k, uint32(i))
				r.SetUniqueKey(&k)
			}
			r.Unref()
			timeStamps[i] = nowish()
		}
		for i := 0; i < cnt; i++ {
			// A resource is purged every other frame since uniquely keyed resources are not considered.
			cache.PurgeResourcesNotUsedSince(timeStamps[i], PurgeScratchResourcesOnly)
			check(t, cache.ResourceCount() == cnt-i/2-1, "count = %d", cache.ResourceCount())
		}
		cache.PurgeResourcesNotUsedSince(nowish(), PurgeAllResources)
		check(t, cache.ResourceCount() == 0, "count = %d", cache.ResourceCount())
	}
}

// TestResourceCachePartialPurge verifies PurgeUnlockedResourcesBytes: it purges only enough purgeable bytes to satisfy
// the requested amount, and the preferScratch flag controls whether scratch or uniquely keyed resources are evicted
// first.
func TestResourceCachePartialPurge(t *testing.T) {
	type testCase struct {
		name            string
		bytesToPurge    uint64
		preferScratch   bool
		expBudgetedCnt  int
		expBudgetedSize uint64
	}
	cases := []testCase{
		{name: "only scratch", bytesToPurge: 14, preferScratch: true, expBudgetedCnt: 3, expBudgetedSize: 33},
		{name: "partial scratch", bytesToPurge: 3, preferScratch: true, expBudgetedCnt: 4, expBudgetedSize: 47},
		{name: "all scratch", bytesToPurge: 50, preferScratch: true, expBudgetedCnt: 0, expBudgetedSize: 0},
		{name: "partial", bytesToPurge: 13, preferScratch: false, expBudgetedCnt: 3, expBudgetedSize: 37},
		{name: "all", bytesToPurge: 50, preferScratch: false, expBudgetedCnt: 0, expBudgetedSize: 0},
		{name: "none", bytesToPurge: 0, preferScratch: true, expBudgetedCnt: 5, expBudgetedSize: 60},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := newMock(100)
			cache := m.cache

			var key1, key2, key3 UniqueKey
			makeUniqueKey(&key1, 1)
			makeUniqueKey(&key2, 2)
			makeUniqueKey(&key3, 3)

			// Add three unique resources to the cache.
			unique1 := m.newResource(BudgetedYes, 10)
			unique2 := m.newResource(BudgetedYes, 11)
			unique3 := m.newResource(BudgetedYes, 12)
			unique1.SetUniqueKey(&key1)
			unique2.SetUniqueKey(&key2)
			unique3.SetUniqueKey(&key3)

			// Add two scratch resources to the cache.
			scratch1 := m.newScratch(BudgetedYes, simulatedPropertyA, 13)
			scratch2 := m.newScratch(BudgetedYes, simulatedPropertyB, 14)

			check(t, cache.BudgetedResourceCount() == 5, "budgeted count = %d",
				cache.BudgetedResourceCount())
			check(t, cache.BudgetedResourceBytes() == 60, "budgeted bytes = %d",
				cache.BudgetedResourceBytes())
			check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())

			// Add the resources to the purgeable queue (in a specific LRU order).
			unique1.Unref()
			scratch1.Unref()
			unique2.Unref()
			scratch2.Unref()
			unique3.Unref()

			check(t, cache.PurgeableBytes() == 60, "purgeable = %d", cache.PurgeableBytes())

			cache.PurgeUnlockedResourcesBytes(tc.bytesToPurge, tc.preferScratch)
			check(t, cache.BudgetedResourceCount() == tc.expBudgetedCnt, "budgeted count = %d",
				cache.BudgetedResourceCount())
			check(t, cache.BudgetedResourceBytes() == tc.expBudgetedSize, "budgeted bytes = %d",
				cache.BudgetedResourceBytes())

			// Ensure all are purged before the next case.
			cache.PurgeUnlockedResources(PurgeAllResources)
			check(t, cache.BudgetedResourceCount() == 0, "budgeted count = %d",
				cache.BudgetedResourceCount())
			check(t, cache.PurgeableBytes() == 0, "purgeable = %d", cache.PurgeableBytes())
		})
	}
}

// TestResourceCacheCustomData verifies that UniqueKey custom data round-trips through SetCustomData/CustomData and is
// preserved when a key is copied by value.
func TestResourceCacheCustomData(t *testing.T) {
	var key1, key2 UniqueKey
	makeUniqueKey(&key1, 1)
	makeUniqueKey(&key2, 2)
	key1.SetCustomData([]byte{4, 1, 3, 2})
	check(t, len(key1.CustomData()) == 4, "custom data missing")
	check(t, key2.CustomData() == nil, "unexpected custom data")

	// Copying a key preserves its custom data.
	key3 := key1
	check(t, len(key3.CustomData()) == 4, "copied custom data missing")
}

// TestResourceCacheAbandoned verifies that AbandonAll destroys a resource and that every public Resource method remains
// safe to call afterward without panicking.
func TestResourceCacheAbandoned(t *testing.T) {
	m := newMock(300)
	resource := m.newResource(BudgetedYes, testResourceDefaultSize)
	m.cache.AbandonAll()

	check(t, resource.WasDestroyed(), "resource should be destroyed")

	// Call the public methods on a resource in the abandoned state. They shouldn't crash.
	resource.UniqueID()
	resource.UniqueKey()
	resource.WasDestroyed()
	resource.GpuMemorySize()
	resource.ScratchKey()
	resource.BudgetedType()
	resource.MakeBudgeted()
	resource.MakeUnbudgeted()
	resource.RemoveScratchKey()
	var key UniqueKey
	makeUniqueKey(&key, 1)
	resource.SetUniqueKey(&key)
	resource.RemoveUniqueKey()
	resource.Unref()
}

// TestResourceCachePurgeToMakeHeadroom verifies PurgeToMakeHeadroom: it succeeds by purging just enough purgeable bytes
// to make room for the requested headroom, or fails and purges nothing when the unpurgeable resources alone already
// exceed the budget.
func TestResourceCachePurgeToMakeHeadroom(t *testing.T) {
	const texSize = 16 * 16 * 4
	for _, success := range []bool{true, false} {
		m := newMock(2 * texSize)
		cache := m.cache

		// One unpurgeable resource and one purgeable resource.
		locked := m.newResource(BudgetedYes, texSize)
		purgeable := m.newResource(BudgetedYes, texSize)
		var key UniqueKey
		makeUniqueKey(&key, 0)
		purgeable.SetUniqueKey(&key) // keyed so it stays in the cache when unreffed
		if success {
			purgeable.Unref()
		}

		expectedPurgeable := uint64(0)
		if success {
			expectedPurgeable = texSize
		}
		check(t, cache.PurgeableBytes() == expectedPurgeable, "purgeable = %d (success=%v)",
			cache.PurgeableBytes(), success)
		check(t, cache.PurgeToMakeHeadroom(texSize) == success, "headroom result (success=%v)", success)
		expectedBudgeted := uint64(texSize)
		if !success {
			expectedBudgeted = 2 * texSize
		}
		check(t, cache.BudgetedResourceBytes() == expectedBudgeted, "budgeted = %d (success=%v)",
			cache.BudgetedResourceBytes(), success)

		locked.Unref()
		if !success {
			purgeable.Unref()
		}
		cache.ReleaseAll()
	}
}

// TestResourceCacheReleaseAllWithRefs verifies releaseAll destroys resources that still hold refs (the
// gr_direct_context_release_resources_and_abandon path).
func TestResourceCacheReleaseAllWithRefs(t *testing.T) {
	m := newMock(30000)
	r := m.newResource(BudgetedYes, 16)
	m.cache.ReleaseAll()
	check(t, r.WasDestroyed(), "resource should be destroyed")
	check(t, m.cache.ResourceCount() == 0, "count = %d", m.cache.ResourceCount())
	r.Unref()
}

// TestResourceCacheOverBudgetPurge verifies LRU eviction order under budget pressure.
func TestResourceCacheOverBudgetPurge(t *testing.T) {
	m := newMock(300)
	cache := m.cache

	// Insert three keyed resources; unref them in a known order to establish LRU.
	var keys [3]UniqueKey
	var res [3]*testResource
	for i := range res {
		makeUniqueKey(&keys[i], uint32(i))
		res[i] = m.newResource(BudgetedYes, 100)
		res[i].SetUniqueKey(&keys[i])
	}
	res[1].Unref()
	res[0].Unref()
	res[2].Unref()
	check(t, cache.ResourceCount() == 3, "count = %d", cache.ResourceCount())

	// Shrink the budget so only one fits; eviction should follow purgeable order (1, then 0).
	cache.SetLimit(100)
	check(t, cache.ResourceCount() == 1, "count = %d", cache.ResourceCount())
	check(t, cache.FindAndRefUniqueResource(&keys[0]) == nil, "resource 0 should be evicted")
	check(t, cache.FindAndRefUniqueResource(&keys[1]) == nil, "resource 1 should be evicted")
	surviving := cache.FindAndRefUniqueResource(&keys[2])
	check(t, surviving == Resource(res[2]), "resource 2 should survive")
	if surviving != nil {
		surviving.resourceBase().Unref()
	}
	cache.PurgeUnlockedResources(PurgeAllResources)
}

// isScratchRes exposes the resource's internal scratch-registration state for test assertions.
func (t *testResource) isScratchRes() bool { return t.isScratch2() }

func (t *testResource) isScratch2() bool { return t.ResourceBase.isScratch() }
