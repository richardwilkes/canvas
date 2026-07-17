// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The in-memory LRU cache of linked programs keyed on the program descriptor, with hit/miss metrics. The
// persistent-cache and precompile lanes are trimmed.

package gl

import (
	"container/list"

	"github.com/richardwilkes/canvas/gpu"
)

// programCacheMaxEntries is the default maximum number of linked programs the cache retains.
const programCacheMaxEntries = 256

// ProgramCacheResult classifies the outcome of a program-cache lookup.
type ProgramCacheResult int32

// ProgramCacheResult values.
const (
	ProgramCacheHit ProgramCacheResult = iota
	ProgramCacheMiss
	// ProgramCachePartial (a cache hit of a precompiled program) is unreachable with the persistent-cache trim but kept
	// for the enum's shape.
	ProgramCachePartial
)

// PipelineStats is the pipeline-builder metrics the GL backend tracks.
type PipelineStats struct {
	ShaderCompilations    int
	NumProgramCacheHits   int
	NumProgramCacheMisses int
}

// programCacheEntry is one LRU entry: a linked program and the key it was stored under.
type programCacheEntry struct {
	program *Program
	key     string
}

// ProgramCache is an LRU map from program descriptor bytes to linked programs.
type ProgramCache struct {
	gpu        *Gpu
	entries    map[string]*list.Element
	lru        *list.List // front = most recently used
	keyBuilder gpu.KeyBuilder
	// desc, keyBuilder, and keyScratch are reused across draws so a steady-state frame's per-draw program lookup
	// allocates nothing on a cache hit. The cache is touched only on the GL context thread, so no synchronization is
	// needed; the descriptor is not retained past the lookup (CreateProgram consumes it synchronously and Program
	// keeps no reference). keyBuilder is a value field (heap-resident with the cache) so building desc.key allocates
	// nothing even though its add methods pass it to interface calls the compiler treats as escaping.
	desc       ProgramDesc
	keyScratch []byte
	stats      PipelineStats
	maxEntries int
}

func newProgramCache(g *Gpu, runtimeProgramCacheSize int) *ProgramCache {
	return &ProgramCache{
		gpu:        g,
		entries:    make(map[string]*list.Element),
		lru:        list.New(),
		maxEntries: runtimeProgramCacheSize,
	}
}

// Stats returns the pipeline-builder metrics.
func (c *ProgramCache) Stats() PipelineStats { return c.stats }

// Count returns the number of cached programs.
func (c *ProgramCache) Count() int { return c.lru.Len() }

// Abandon marks every cached program as abandoned (its GL objects are already gone, e.g. after a lost context) and then
// resets the cache.
func (c *ProgramCache) Abandon() {
	for e := c.lru.Front(); e != nil; e = e.Next() {
		e.Value.(*programCacheEntry).program.Abandon()
	}
	c.Reset()
}

// Reset destroys every cached program (unless already abandoned) and empties the cache.
func (c *ProgramCache) Reset() {
	for e := c.lru.Front(); e != nil; e = e.Next() {
		e.Value.(*programCacheEntry).program.destroy(c.gpu)
	}
	c.entries = make(map[string]*list.Element)
	c.lru.Init()
}

// FindOrCreateProgram builds the descriptor for the program info, then returns the cached program or compiles a new
// one. Returns nil (and no cache entry) if building fails.
func (c *ProgramCache) FindOrCreateProgram(programInfo *ProgramInfo) *Program {
	// Reuse the cache's descriptor storage and key builder across draws (retains the key-word slice capacity; the
	// builder avoids the per-key heap allocation its escaping add methods would force).
	buildProgramDescReusing(&c.desc, &c.keyBuilder, programInfo, c.gpu.glCaps())
	if !c.desc.IsValid() {
		return nil
	}
	program, _ := c.findOrCreateProgramImpl(&c.desc, programInfo)
	return program
}

// findOrCreateProgramImpl looks up desc in the LRU cache, compiling and inserting a new program on a miss.
func (c *ProgramCache) findOrCreateProgramImpl(desc *ProgramDesc, programInfo *ProgramInfo) (*Program, ProgramCacheResult) {
	// Build the key's byte image into reused scratch and look it up with the compiler's m[string(bytes)] no-allocation
	// conversion, so a cache hit (the steady-state case) allocates nothing. Only a miss materializes the persistent
	// string key stored in the map entry.
	c.keyScratch = desc.AppendKeyBytes(c.keyScratch[:0])
	if e, ok := c.entries[string(c.keyScratch)]; ok {
		c.lru.MoveToFront(e)
		c.stats.NumProgramCacheHits++
		return e.Value.(*programCacheEntry).program, ProgramCacheHit
	}

	c.stats.NumProgramCacheMisses++
	program := CreateProgram(c.gpu, desc, programInfo)
	if program == nil {
		return nil, ProgramCacheMiss
	}
	key := string(c.keyScratch)
	c.entries[key] = c.lru.PushFront(&programCacheEntry{program: program, key: key})
	if c.lru.Len() > c.maxEntries {
		oldest := c.lru.Back()
		entry := oldest.Value.(*programCacheEntry)
		entry.program.destroy(c.gpu)
		delete(c.entries, entry.key)
		c.lru.Remove(oldest)
	}
	return program, ProgramCacheMiss
}
