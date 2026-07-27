// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ThreadSafeCache is the uniquely-keyed cache used for shareable utility data: the triangulating path renderer's cached
// vertex data and the analytic-blur profile/mask textures. Despite the name, this package is single-threaded per
// context, so there is no locking; the cache's observable contract is kept regardless: only this cache knows the unique
// key, entries hold refs on their payloads, uniquely-held entries are dropped LRU-to-MRU when the resource cache is
// over budget (prioritizing this cache's survival over plain resource-cache entries), and abandon/release/purge
// plumbing routes through the ResourceCache hooks.
//
// The entry is a tagged union over two payload kinds. VertexData carries an explicit ref count so "uniquely held" is
// observable; ops ref it while they hold it and unref when executed. A SurfaceProxyView entry holds one ref on its
// proxy — the cache adopts the caller's single construction ref in Add and drops it on eviction, so uniquelyHeld is
// `proxy.RefCnt() == 1`. The view consumers here (the analytic-blur FPs) always generate the profile/mask on the CPU,
// upload it as a lazy-upload texture, then add the resulting view, so no GPU-fill-in race resolution is needed on the
// add path. Because those views are lazy-upload proxies whose callbacks regenerate the data, and transient holders
// never ref proxies, a cached view becomes uniquely held as soon as its sampling op is recorded; a mid-recording budget
// purge merely re-uploads it at flush with identical output.

package gl

import (
	"time"

	"github.com/richardwilkes/canvas/gpu"
)

// VertexData is ref-counted vertex data that transparently transitions from CPU-side to GPU-side (the deferred
// GPU-buffer fill-in).
type VertexData struct {
	gpuBuffer   *Buffer
	vertices    []byte
	numVertices int
	vertexSize  uint64
	refCnt      int32
}

// MakeVertexDataFromBytes returns a new VertexData backed by CPU-side bytes, with refCnt 1.
func MakeVertexDataFromBytes(vertices []byte, vertexCount int, vertexSize uint64) *VertexData {
	return &VertexData{
		vertices: vertices, numVertices: vertexCount, vertexSize: vertexSize,
		refCnt: 1,
	}
}

// MakeVertexDataFromBuffer returns a new VertexData backed by a GPU buffer, with refCnt 1. The buffer ref is
// transferred to the VertexData.
func MakeVertexDataFromBuffer(buffer *Buffer, vertexCount int, vertexSize uint64) *VertexData {
	return &VertexData{
		gpuBuffer: buffer, numVertices: vertexCount, vertexSize: vertexSize,
		refCnt: 1,
	}
}

// Vertices returns the CPU-side vertex bytes, or nil once a GPU buffer has been set.
func (v *VertexData) Vertices() []byte { return v.vertices }

// Size returns the total size in bytes of the vertex data.
func (v *VertexData) Size() uint64 { return uint64(v.numVertices) * v.vertexSize }

// NumVertices returns the number of vertices.
func (v *VertexData) NumVertices() int { return v.numVertices }

// VertexSize returns the size in bytes of a single vertex.
func (v *VertexData) VertexSize() uint64 { return v.vertexSize }

// GpuBuffer returns the GPU buffer backing this data, or nil if it hasn't been uploaded yet.
func (v *VertexData) GpuBuffer() *Buffer { return v.gpuBuffer }

// SetGpuBuffer records that the vertex data has been uploaded to a GPU buffer, taking ownership of the buffer ref.
func (v *VertexData) SetGpuBuffer(buffer *Buffer) {
	if v.gpuBuffer != nil {
		panic("VertexData already has a gpu buffer")
	}
	v.gpuBuffer = buffer
}

// Ref adds a reference.
func (v *VertexData) Ref() { v.refCnt++ }

// Unref drops a reference; the last unref resets the data, freeing the GPU buffer ref.
func (v *VertexData) Unref() {
	v.refCnt--
	if v.refCnt == 0 {
		v.reset()
	}
}

func (v *VertexData) unique() bool { return v.refCnt == 1 }

// reset clears the vertex data and drops its GPU buffer ref, if any.
func (v *VertexData) reset() {
	v.vertices = nil
	v.numVertices = 0
	v.vertexSize = 0
	if v.gpuBuffer != nil {
		v.gpuBuffer.Unref()
		v.gpuBuffer = nil
	}
}

// tscTag distinguishes the entry's payload kind. Entries here are always created full and dropped whole, so there is no
// "empty" tag.
type tscTag int

const (
	tscTagVertData tscTag = iota // vertData is valid
	tscTagView                   // view is valid
)

// tscEntry is a tagged union over VertexData and SurfaceProxyView payloads.
type tscEntry struct {
	lastAccess time.Time
	vertData   *VertexData // valid when tag == tscTagVertData
	prev       *tscEntry   // the uniquely-keyed entry list; head is MRU
	next       *tscEntry
	view       SurfaceProxyView // valid when tag == tscTagView
	key        gpu.UniqueKey
	tag        tscTag
}

func (e *tscEntry) uniquelyHeld() bool {
	if e.tag == tscTagView {
		return e.view.Proxy().RefCnt() == 1
	}
	return e.vertData.unique()
}

// IsNewerBetter reports whether 'challenger' should replace 'incumbent' in the cache on a key collision. The arguments
// are the keys' custom data.
type IsNewerBetter func(incumbent, challenger []byte) bool

// ThreadSafeCacheStats exposes cache counters for tests to inspect entry counts and payload identity.
type ThreadSafeCacheStats struct {
	Hits         int // finds that returned an entry
	Misses       int // finds that returned nothing
	Adds         int // new entries created
	Replacements int // payloads replaced because the challenger was better
}

// ThreadSafeCache is a uniquely-keyed cache for shareable utility data such as cached vertex triangulations and
// analytic-blur mask textures.
type ThreadSafeCache struct {
	entries    map[string]*tscEntry
	head, tail *tscEntry
	stats      ThreadSafeCacheStats
}

// NewThreadSafeCache returns a new, empty ThreadSafeCache.
func NewThreadSafeCache() *ThreadSafeCache {
	return &ThreadSafeCache{entries: make(map[string]*tscEntry)}
}

// NumEntries returns the number of entries currently in the cache, for tests.
func (c *ThreadSafeCache) NumEntries() int { return len(c.entries) }

// Stats returns the test counters.
func (c *ThreadSafeCache) Stats() *ThreadSafeCacheStats { return &c.stats }

func (c *ThreadSafeCache) listRemove(e *tscEntry) {
	if e.prev != nil {
		e.prev.next = e.next
	} else {
		c.head = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else {
		c.tail = e.prev
	}
	e.prev = nil
	e.next = nil
}

func (c *ThreadSafeCache) listAddToHead(e *tscEntry) {
	e.prev = nil
	e.next = c.head
	if c.head != nil {
		c.head.prev = e
	}
	c.head = e
	if c.tail == nil {
		c.tail = e
	}
}

// makeExistingEntryMRU stamps e's last-access time and moves it to the head of the LRU list.
func (c *ThreadSafeCache) makeExistingEntryMRU(e *tscEntry) {
	e.lastAccess = time.Now()
	c.listRemove(e)
	c.listAddToHead(e)
}

// dropEntry removes an entry from the map and list and drops the cache's payload ref.
func (c *ThreadSafeCache) dropEntry(e *tscEntry) {
	delete(c.entries, e.key.MapKey())
	c.listRemove(e)
	if e.tag == tscTagView {
		e.view.Proxy().Unref()
		e.view.Reset()
		return
	}
	e.vertData.Unref()
	e.vertData = nil
}

// DropAllRefs drops every entry in the cache, regardless of whether it is uniquely held.
func (c *ThreadSafeCache) DropAllRefs() {
	for c.head != nil {
		c.dropEntry(c.head)
	}
}

// DropUniqueRefs drops uniquely held refs until the resource cache is under budget, LRU to MRU. A nil resourceCache
// means drop all uniquely held refs.
func (c *ThreadSafeCache) DropUniqueRefs(resourceCache *gpu.ResourceCache) {
	// Iterate from LRU to MRU.
	cur := c.tail
	var prev *tscEntry
	if cur != nil {
		prev = cur.prev
	}
	for cur != nil {
		if resourceCache != nil && !resourceCache.OverBudget() {
			return
		}
		if cur.uniquelyHeld() {
			c.dropEntry(cur)
		}
		cur = prev
		if cur != nil {
			prev = cur.prev
		} else {
			prev = nil
		}
	}
}

// DropUniqueRefsOlderThan drops uniquely held refs last accessed before purgeTime.
func (c *ThreadSafeCache) DropUniqueRefsOlderThan(purgeTime time.Time) {
	// Iterate from LRU to MRU.
	cur := c.tail
	var prev *tscEntry
	if cur != nil {
		prev = cur.prev
	}
	for cur != nil {
		if !cur.lastAccess.Before(purgeTime) {
			// This entry and all the remaining ones in the list are newer than purgeTime.
			return
		}
		if cur.uniquelyHeld() {
			c.dropEntry(cur)
		}
		cur = prev
		if cur != nil {
			prev = cur.prev
		} else {
			prev = nil
		}
	}
}

// FindVertsWithData returns a ref on the cached vertex data (nil on miss) plus the stored key's custom data.
func (c *ThreadSafeCache) FindVertsWithData(key *gpu.UniqueKey) (verts *VertexData, data []byte) {
	e := c.entries[key.MapKey()]
	if e == nil || e.tag != tscTagVertData {
		c.stats.Misses++
		return nil, nil
	}
	c.stats.Hits++
	c.makeExistingEntryMRU(e)
	e.vertData.Ref()
	return e.vertData, e.key.CustomData()
}

// AddVertsWithData adds vertData under key unless an entry already exists, in which case isNewerBetter decides whether
// the existing payload is replaced (orphaning prior uses but keeping the best version in the cache). Returns a ref on
// the cached vertex data plus the stored key's custom data; the caller's vertData ref is consumed.
func (c *ThreadSafeCache) AddVertsWithData(key *gpu.UniqueKey, vertData *VertexData, isNewerBetter IsNewerBetter) (verts *VertexData, data []byte) {
	e := c.entries[key.MapKey()]
	switch {
	case e == nil:
		e = &tscEntry{key: *key, tag: tscTagVertData, vertData: vertData}
		e.vertData.Ref() // the cache's ref
		c.entries[key.MapKey()] = e
		e.lastAccess = time.Now()
		c.listAddToHead(e)
		c.stats.Adds++
	case isNewerBetter(e.key.CustomData(), key.CustomData()):
		// This orphans any existing uses of the prior vertex data but ensures the best version is in the cache.
		e.vertData.Unref()
		e.key = *key
		e.vertData = vertData
		e.vertData.Ref()
		c.stats.Replacements++
	default:
		// The incumbent stays; drop the caller's payload ref and hand back the incumbent.
		vertData.Unref()
		e.vertData.Ref()
		return e.vertData, e.key.CustomData()
	}
	// The caller's ref is transferred to the returned value.
	return e.vertData, e.key.CustomData()
}

///////////////////////////////////////////////////////////////////////////////
// View side (SurfaceProxyView payloads)

// Has reports whether an entry exists for the key.
func (c *ThreadSafeCache) Has(key *gpu.UniqueKey) bool {
	_, ok := c.entries[key.MapKey()]
	return ok
}

// internalFindView returns the found view (invalid on miss) plus the stored key's custom data. Unlike the vertData lane
// the returned view is not reffed — its consumer is a transient texture-effect FP within the current recording, kept
// valid by GC and the cache's own ref (see the file comment).
func (c *ThreadSafeCache) internalFindView(key *gpu.UniqueKey) (v SurfaceProxyView, data []byte) {
	e := c.entries[key.MapKey()]
	if e == nil || e.tag != tscTagView {
		c.stats.Misses++
		return SurfaceProxyView{}, nil
	}
	c.stats.Hits++
	c.makeExistingEntryMRU(e)
	return e.view, e.key.CustomData()
}

// Find returns the cached view for key, or an invalid view on miss.
func (c *ThreadSafeCache) Find(key *gpu.UniqueKey) SurfaceProxyView {
	view, _ := c.internalFindView(key)
	return view
}

// FindWithData returns the cached view for key plus the stored key's custom data.
func (c *ThreadSafeCache) FindWithData(key *gpu.UniqueKey) (v SurfaceProxyView, data []byte) {
	return c.internalFindView(key)
}

// internalAddView adds view under key: on a new key the cache adopts the caller's single ref on the view's proxy
// (becoming the long-lived owner). A collision with an existing view entry keeps the incumbent and drops the caller's
// now-redundant ref — whether or not the two views name the same proxy, since either way the cache already holds the
// ref that entry needs, and a surviving extra one would keep it from ever being uniquely held (so unpurgeable by
// DropUniqueRefs). A collision with a vertData entry has no incumbent view to hand back, so that entry is dropped in
// favor of this one, which orphans any existing uses of the vertex data exactly as the is-newer-better replacement in
// AddVertsWithData does; the caller's ref is adopted as usual and the returned view is always usable. Returns the
// stored view plus the stored key's custom data. Neither collision path can occur in this package's single-threaded
// find-then-add flow, but both are handled anyway.
func (c *ThreadSafeCache) internalAddView(key *gpu.UniqueKey, view SurfaceProxyView) (v SurfaceProxyView, data []byte) {
	if e := c.entries[key.MapKey()]; e != nil {
		if e.tag == tscTagView {
			view.Proxy().Unref()
			return e.view, e.key.CustomData()
		}
		c.dropEntry(e)
	}
	e := &tscEntry{key: *key, tag: tscTagView, view: view}
	c.entries[key.MapKey()] = e
	e.lastAccess = time.Now()
	c.listAddToHead(e)
	c.stats.Adds++
	return e.view, e.key.CustomData()
}

// Add adds view under key: the cache adopts the caller's ref on the view (see internalAddView). Returns the stored
// view.
func (c *ThreadSafeCache) Add(key *gpu.UniqueKey, view SurfaceProxyView) SurfaceProxyView {
	v, _ := c.internalAddView(key, view)
	return v
}

// AddWithData adds view under key and returns the stored view plus the stored key's custom data.
func (c *ThreadSafeCache) AddWithData(key *gpu.UniqueKey, view SurfaceProxyView) (v SurfaceProxyView, data []byte) {
	return c.internalAddView(key, view)
}

// Remove drops the entry for key, if any.
func (c *ThreadSafeCache) Remove(key *gpu.UniqueKey) {
	if e := c.entries[key.MapKey()]; e != nil {
		c.dropEntry(e)
	}
}
