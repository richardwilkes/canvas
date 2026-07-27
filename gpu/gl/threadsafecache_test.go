// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for ThreadSafeCache: find/add semantics, the is-newer-better replacement rule, uniquely-held-only
// dropping in LRU order, age-based dropping, and payload ref pinning.

package gl

import (
	"testing"
	"time"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

var tscTestDomain = gpu.GenerateUniqueKeyDomain()

func tscTestKey(id uint32, custom []byte) gpu.UniqueKey {
	var key gpu.UniqueKey
	builder := gpu.UniqueKeyBuilder(&key, tscTestDomain, 1, "test")
	builder.Slice()[0] = id
	builder.Finish()
	if custom != nil {
		key.SetCustomData(custom)
	}
	return key
}

func tscTestVerts(n int) *VertexData {
	return MakeVertexDataFromBytes(make([]byte, n*8), n, 8)
}

func TestThreadSafeCacheFindAdd(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(1, []byte{1})

	if v, _ := c.FindVertsWithData(&key); v != nil {
		t.Fatal("find on an empty cache must miss")
	}
	if c.Stats().Misses != 1 {
		t.Fatalf("misses = %d, want 1", c.Stats().Misses)
	}

	vd := tscTestVerts(3)
	vd.Ref() // the copy handed to the cache
	got, data := c.AddVertsWithData(&key, vd, func(_, _ []byte) bool { return false })
	if got != vd || len(data) != 1 || data[0] != 1 {
		t.Fatal("add must return the added payload and its custom data")
	}
	got.Unref()
	if c.NumEntries() != 1 || c.Stats().Adds != 1 {
		t.Fatalf("entries = %d, adds = %d", c.NumEntries(), c.Stats().Adds)
	}

	found, data2 := c.FindVertsWithData(&key)
	if found != vd || len(data2) != 1 {
		t.Fatal("find must return the cached payload")
	}
	found.Unref()

	// A second add with a losing challenger keeps the incumbent and consumes the caller's ref.
	loser := tscTestVerts(5)
	loser.Ref()
	got2, _ := c.AddVertsWithData(&key, loser, func(_, _ []byte) bool { return false })
	if got2 != vd {
		t.Fatal("losing challenger must not replace the incumbent")
	}
	got2.Unref()
	loser.Unref() // the caller's own ref
	if loser.refCnt != 0 {
		t.Fatalf("loser refCnt = %d, want 0", loser.refCnt)
	}

	// A winning challenger replaces the payload and the stored custom data.
	key2 := tscTestKey(1, []byte{2})
	winner := tscTestVerts(7)
	winner.Ref()
	got3, data3 := c.AddVertsWithData(&key2, winner, func(_, _ []byte) bool { return true })
	if got3 != winner || data3[0] != 2 {
		t.Fatal("winning challenger must replace the incumbent")
	}
	got3.Unref()
	if c.Stats().Replacements != 1 || c.NumEntries() != 1 {
		t.Fatalf("replacements = %d, entries = %d", c.Stats().Replacements, c.NumEntries())
	}
	// The incumbent was orphaned: only the original creator ref remains.
	vd.Unref()
	if vd.refCnt != 0 {
		t.Fatalf("orphaned incumbent refCnt = %d, want 0", vd.refCnt)
	}
}

func TestThreadSafeCacheDropUniqueRefs(t *testing.T) {
	c := NewThreadSafeCache()

	keyA := tscTestKey(1, []byte{0})
	keyB := tscTestKey(2, []byte{0})
	vdA := tscTestVerts(1)
	vdA.Ref()
	retA, _ := c.AddVertsWithData(&keyA, vdA, func(_, _ []byte) bool { return false })
	retA.Unref()
	vdB := tscTestVerts(2)
	vdB.Ref()
	retB, _ := c.AddVertsWithData(&keyB, vdB, func(_, _ []byte) bool { return false })
	retB.Unref()

	// vdA/vdB still carry the creators' refs: nothing is uniquely held.
	c.DropUniqueRefs(nil)
	if c.NumEntries() != 2 {
		t.Fatalf("entries = %d, want 2 (payloads pinned)", c.NumEntries())
	}

	// Release the creator ref on A only: A becomes uniquely held and is dropped; B survives.
	vdA.Unref()
	c.DropUniqueRefs(nil)
	if c.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1", c.NumEntries())
	}
	if v, _ := c.FindVertsWithData(&keyA); v != nil {
		t.Fatal("A must be gone")
	}
	if v, _ := c.FindVertsWithData(&keyB); v == nil {
		t.Fatal("B must survive while pinned")
	} else {
		v.Unref()
	}

	vdB.Unref()
	c.DropUniqueRefs(nil)
	if c.NumEntries() != 0 {
		t.Fatalf("entries = %d, want 0", c.NumEntries())
	}
}

func TestThreadSafeCacheDropUniqueRefsOlderThan(t *testing.T) {
	c := NewThreadSafeCache()

	keyOld := tscTestKey(1, []byte{0})
	old := tscTestVerts(1)
	old.Ref()
	r, _ := c.AddVertsWithData(&keyOld, old, func(_, _ []byte) bool { return false })
	r.Unref()
	old.Unref() // uniquely held by the cache

	keyNew := tscTestKey(2, []byte{0})
	fresh := tscTestVerts(2)
	fresh.Ref()
	r2, _ := c.AddVertsWithData(&keyNew, fresh, func(_, _ []byte) bool { return false })
	r2.Unref()
	fresh.Unref() // uniquely held too, but will be stamped newer than the cutoff

	// Assign explicit last-access times rather than racing time.Now() around the adds: on Windows the clock is coarse
	// enough that consecutive time.Now() calls return identical values, which made a wall-clock cutoff
	// nondeterministic. Explicit stamps keep this test hermetic without sleeping.
	base := time.Now()
	c.entries[keyOld.MapKey()].lastAccess = base.Add(-2 * time.Hour)
	c.entries[keyNew.MapKey()].lastAccess = base

	// An entry accessed exactly at the purge time must survive: dropping only applies to entries strictly older than
	// purgeTime.
	c.DropUniqueRefsOlderThan(base.Add(-2 * time.Hour))
	if c.NumEntries() != 2 {
		t.Fatalf("entries = %d, want 2 (entry accessed exactly at purgeTime must survive)", c.NumEntries())
	}

	c.DropUniqueRefsOlderThan(base.Add(-time.Hour))
	if c.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1", c.NumEntries())
	}
	if v, _ := c.FindVertsWithData(&keyOld); v != nil {
		t.Fatal("the older entry must be dropped")
	}
	if v, _ := c.FindVertsWithData(&keyNew); v == nil {
		t.Fatal("the newer entry must survive")
	} else {
		v.Unref()
	}
}

func TestThreadSafeCacheDropAllRefs(t *testing.T) {
	c := NewThreadSafeCache()
	for i := uint32(0); i < 4; i++ {
		key := tscTestKey(i, []byte{0})
		vd := tscTestVerts(int(i) + 1)
		vd.Ref()
		r, _ := c.AddVertsWithData(&key, vd, func(_, _ []byte) bool { return false })
		r.Unref()
		if i%2 == 0 {
			vd.Unref() // half uniquely held, half pinned — dropAllRefs takes everything
		}
	}
	c.DropAllRefs()
	if c.NumEntries() != 0 {
		t.Fatalf("entries = %d, want 0", c.NumEntries())
	}
}

func TestThreadSafeCacheRemove(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(9, []byte{0})
	vd := tscTestVerts(2)
	vd.Ref()
	r, _ := c.AddVertsWithData(&key, vd, func(_, _ []byte) bool { return false })
	r.Unref()
	c.Remove(&key)
	if c.NumEntries() != 0 {
		t.Fatalf("entries = %d, want 0", c.NumEntries())
	}
	vd.Unref()
	if vd.refCnt != 0 {
		t.Fatalf("refCnt = %d, want 0 after remove + creator unref", vd.refCnt)
	}
}

// tscTestView builds a bare deferred proxy view for the view-side cache tests. The proxy carries one ref (the
// constructor's), which Add adopts as the cache's ref. Its format/dimensions are irrelevant: the cache only stores the
// view and inspects the proxy's ref count.
func tscTestView() SurfaceProxyView {
	proxy := newDeferredSurfaceProxy(FormatUnknown, gpu.TextureType2D,
		geom.ISize{Width: 4, Height: 4}, gpu.BackingFitExact, gpu.BudgetedYes,
		gpu.SurfaceFlags(0), UseAllocatorYes, "tsc-test-view")
	return MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, gpu.SwizzleRGBA)
}

func TestThreadSafeCacheViewFindAdd(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(1, nil)

	if v := c.Find(&key); v.IsValid() {
		t.Fatal("find on an empty cache must miss")
	}
	if c.Has(&key) {
		t.Fatal("Has must be false on an empty cache")
	}
	if c.Stats().Misses != 1 {
		t.Fatalf("misses = %d, want 1", c.Stats().Misses)
	}

	view := tscTestView()
	proxy := view.Proxy()
	got := c.Add(&key, view)
	if got.Proxy() != proxy {
		t.Fatal("add must return the added view")
	}
	if proxy.RefCnt() != 1 {
		t.Fatalf("adopted proxy refCnt = %d, want 1 (the cache owns the single ref)", proxy.RefCnt())
	}
	if c.NumEntries() != 1 || c.Stats().Adds != 1 {
		t.Fatalf("entries = %d, adds = %d", c.NumEntries(), c.Stats().Adds)
	}
	if !c.Has(&key) {
		t.Fatal("Has must be true after add")
	}

	// Find returns the cached view without reffing it (the consumer is a transient FP).
	found, _ := c.FindWithData(&key)
	if found.Proxy() != proxy {
		t.Fatal("find must return the cached view")
	}
	if proxy.RefCnt() != 1 {
		t.Fatalf("find must not ref the transient view: refCnt = %d, want 1", proxy.RefCnt())
	}
	if c.Stats().Hits != 1 {
		t.Fatalf("hits = %d, want 1", c.Stats().Hits)
	}

	// A colliding add keeps the incumbent and drops the loser's adopted ref.
	loser := tscTestView()
	loserProxy := loser.Proxy()
	got2 := c.Add(&key, loser)
	if got2.Proxy() != proxy {
		t.Fatal("colliding add must return the incumbent")
	}
	if loserProxy.RefCnt() != 0 {
		t.Fatalf("colliding add must drop the loser's ref: refCnt = %d, want 0", loserProxy.RefCnt())
	}
	if c.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1 after a colliding add", c.NumEntries())
	}
}

// TestThreadSafeCacheViewWithData exercises the custom-data lanes (the general filter_mask cache lane stores the mask
// draw rect as custom data): add/find round-trip the stored key's data.
func TestThreadSafeCacheViewWithData(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(3, []byte{7, 8})
	view := tscTestView()
	proxy := view.Proxy()

	got, data := c.AddWithData(&key, view)
	if got.Proxy() != proxy || len(data) != 2 || data[0] != 7 || data[1] != 8 {
		t.Fatalf("AddWithData must return the view and its custom data, got data=%v", data)
	}
	found, data2 := c.FindWithData(&key)
	if found.Proxy() != proxy || len(data2) != 2 || data2[0] != 7 || data2[1] != 8 {
		t.Fatalf("FindWithData must return the cached view and custom data, got data=%v", data2)
	}
}

func TestThreadSafeCacheViewDropUniqueRefs(t *testing.T) {
	c := NewThreadSafeCache()
	keyA := tscTestKey(1, nil)
	keyB := tscTestKey(2, nil)

	viewA := tscTestView()
	proxyA := viewA.Proxy()
	c.Add(&keyA, viewA)
	viewB := tscTestView()
	proxyB := viewB.Proxy()
	c.Add(&keyB, viewB)

	// Simulate an in-flight op holding A: it is no longer uniquely held (refCnt 2).
	proxyA.Ref()

	c.DropUniqueRefs(nil)
	if c.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1 (A pinned, B dropped)", c.NumEntries())
	}
	if v := c.Find(&keyB); v.IsValid() {
		t.Fatal("B must be gone")
	}
	if proxyB.RefCnt() != 0 {
		t.Fatalf("dropped view proxy refCnt = %d, want 0", proxyB.RefCnt())
	}
	if v := c.Find(&keyA); !v.IsValid() {
		t.Fatal("A must survive while pinned")
	}

	// Release the simulated op ref: A becomes uniquely held and is dropped.
	proxyA.Unref()
	c.DropUniqueRefs(nil)
	if c.NumEntries() != 0 {
		t.Fatalf("entries = %d, want 0", c.NumEntries())
	}
	if proxyA.RefCnt() != 0 {
		t.Fatalf("dropped view proxy refCnt = %d, want 0", proxyA.RefCnt())
	}
}

func TestThreadSafeCacheViewRemoveAndDropAll(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(7, nil)
	view := tscTestView()
	proxy := view.Proxy()
	c.Add(&key, view)
	c.Remove(&key)
	if c.NumEntries() != 0 {
		t.Fatalf("entries = %d, want 0 after remove", c.NumEntries())
	}
	if proxy.RefCnt() != 0 {
		t.Fatalf("removed view proxy refCnt = %d, want 0", proxy.RefCnt())
	}

	// DropAllRefs takes everything regardless of unique-ness, dropping only the cache's ref.
	pinned := tscTestView()
	pinnedProxy := pinned.Proxy()
	key2 := tscTestKey(8, nil)
	c.Add(&key2, pinned)
	pinnedProxy.Ref() // simulate an in-flight holder
	c.DropAllRefs()
	if c.NumEntries() != 0 {
		t.Fatalf("entries = %d, want 0 after DropAllRefs", c.NumEntries())
	}
	if pinnedProxy.RefCnt() != 1 {
		t.Fatalf("pinned proxy refCnt = %d, want 1 (only the holder's ref remains)", pinnedProxy.RefCnt())
	}
}

// TestThreadSafeCacheViewSameProxyCollision covers the collision flavor where the incumbent entry already holds the
// very proxy being added: the caller's ref is still redundant (the cache owns the one the entry needs), so it has to be
// dropped, or the entry can never become uniquely held and DropUniqueRefs can never purge it.
func TestThreadSafeCacheViewSameProxyCollision(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(11, nil)
	view := tscTestView()
	proxy := view.Proxy()
	c.Add(&key, view) // the cache adopts the constructor's ref

	// A second caller arrives with its own ref on the same proxy.
	proxy.Ref()
	got := c.Add(&key, MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, gpu.SwizzleRGBA))
	if got.Proxy() != proxy {
		t.Fatal("colliding add must return the incumbent")
	}
	if proxy.RefCnt() != 1 {
		t.Fatalf("same-proxy collision must drop the caller's redundant ref: refCnt = %d, want 1",
			proxy.RefCnt())
	}
	if c.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1", c.NumEntries())
	}

	// With no leaked ref the entry is uniquely held, so it purges normally.
	c.DropUniqueRefs(nil)
	if c.NumEntries() != 0 {
		t.Fatalf("entries = %d, want 0 (a leaked ref would pin the entry forever)", c.NumEntries())
	}
	if proxy.RefCnt() != 0 {
		t.Fatalf("purged proxy refCnt = %d, want 0", proxy.RefCnt())
	}
}

// TestThreadSafeCacheViewOverVertDataCollision covers a view added over a key already held by a vertex-data entry.
// There is no incumbent view to hand back, so the vertData entry gives way (orphaning its payload exactly as the
// is-newer-better replacement does) and the caller gets a usable view rather than the zero one.
func TestThreadSafeCacheViewOverVertDataCollision(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(12, nil)
	vd := tscTestVerts(3)
	vd.Ref()
	r, _ := c.AddVertsWithData(&key, vd, func(_, _ []byte) bool { return false })
	r.Unref()

	view := tscTestView()
	proxy := view.Proxy()
	got, _ := c.AddWithData(&key, view)
	if !got.IsValid() || got.Proxy() != proxy {
		t.Fatal("a vertData collision must still hand back a usable view")
	}
	if proxy.RefCnt() != 1 {
		t.Fatalf("adopted proxy refCnt = %d, want 1 (the cache owns the single ref)", proxy.RefCnt())
	}
	if c.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1", c.NumEntries())
	}
	if v := c.Find(&key); !v.IsValid() || v.Proxy() != proxy {
		t.Fatal("the view must be the entry stored under the key")
	}
	if v, _ := c.FindVertsWithData(&key); v != nil {
		v.Unref()
		t.Fatal("the displaced vertData entry must be gone")
	}
	if vd.refCnt != 1 {
		t.Fatalf("displaced vertData refCnt = %d, want 1 (only the creator's ref remains)", vd.refCnt)
	}
}

// TestThreadSafeCacheMixedTags guards the tagged union: a view key and a vertData key coexist, and a cross-tag lookup
// misses (the payload accessors are tag-specific).
func TestThreadSafeCacheMixedTags(t *testing.T) {
	c := NewThreadSafeCache()
	viewKey := tscTestKey(1, nil)
	vertKey := tscTestKey(2, nil)

	c.Add(&viewKey, tscTestView())
	vd := tscTestVerts(3)
	vd.Ref()
	r, _ := c.AddVertsWithData(&vertKey, vd, func(_, _ []byte) bool { return false })
	r.Unref()

	// Cross-tag lookups miss.
	if v, _ := c.FindVertsWithData(&viewKey); v != nil {
		t.Fatal("FindVertsWithData on a view entry must miss")
	}
	if v := c.Find(&vertKey); v.IsValid() {
		t.Fatal("Find on a vertData entry must miss")
	}
	// Same-tag lookups still hit.
	if v := c.Find(&viewKey); !v.IsValid() {
		t.Fatal("Find on a view entry must hit")
	}
	if v, _ := c.FindVertsWithData(&vertKey); v == nil {
		t.Fatal("FindVertsWithData on a vertData entry must hit")
	} else {
		v.Unref()
	}
}

// TestAddVertsWithDataOverViewEntry pins the one exported add direction that used to be unguarded: a vertData add
// colliding with a view entry. isNewerBetter would have been handed a view entry's (non-tess) custom data and both the
// replace and the keep branch would then have nil-dereferenced e.vertData. The collision must instead drop the view
// entry — releasing its adopted proxy ref — and install the vertex data, mirroring internalAddView's handling of the
// opposite collision.
func TestAddVertsWithDataOverViewEntry(t *testing.T) {
	c := NewThreadSafeCache()
	key := tscTestKey(9, nil)

	view := tscTestView()
	viewProxy := view.Proxy()
	c.Add(&key, view)
	if viewProxy.RefCnt() != 1 {
		t.Fatalf("adopted proxy refCnt = %d, want 1", viewProxy.RefCnt())
	}

	vd := tscTestVerts(4)
	vd.Ref() // the copy handed to the cache
	// isNewerBetter must never be consulted for a cross-tag collision: there is no incumbent vertex data to compare.
	got, _ := c.AddVertsWithData(&key, vd, func(_, _ []byte) bool {
		t.Fatal("isNewerBetter must not be called on a view-entry collision")
		return false
	})
	if got != vd {
		t.Fatal("a view-entry collision must install the caller's vertex data")
	}
	got.Unref()
	if viewProxy.RefCnt() != 0 {
		t.Fatalf("dropped view proxy refCnt = %d, want 0", viewProxy.RefCnt())
	}
	if c.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1", c.NumEntries())
	}
	// The entry now answers the vertData lane and no longer answers the view lane.
	if v, _ := c.FindVertsWithData(&key); v == nil {
		t.Fatal("FindVertsWithData must hit after the replacement")
	} else {
		v.Unref()
	}
	if v := c.Find(&key); v.IsValid() {
		t.Fatal("Find must miss: the view entry was replaced")
	}
}
