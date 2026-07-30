// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// StrikeCache and Strike: the GPU-side strike cache mapping strike specs to Strikes, each of which owns the
// Glyph records (packed ID + atlas locator) for one scaler configuration. Strikes are keyed on the comparable
// font.StrikeSpec, the same way the CPU strike cache does (see font/strikecache.go) — including its rule that a spec
// carrying an unhashable effect skips the cache rather than panicking on the map key. The cache is only used from the
// single-threaded flush path, but stays lock-protected as a defensive measure.

package text

import (
	"sync"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/internal/memsize"
)

// Cache limits for the GPU-side strike cache.
const (
	defaultGpuFontCacheLimit      = 2 * 1024 * 1024
	defaultGpuFontCacheCountLimit = 2048
)

// What this cache charges its byte budget per item, derived from the types it retains rather than estimated.
var (
	// glyphRecordOverhead is what one Glyph record costs: the record itself plus its share of the strike's map.
	glyphRecordOverhead = memsize.Of[Glyph]() + memsize.MapEntry[font.PackedGlyphID, *Glyph]()
	// textStrikeOverhead is what a Strike costs before any glyph records are added: the Strike itself, its empty map,
	// and its own entry in the cache's map of strikes.
	textStrikeOverhead = memsize.Of[Strike]() + memsize.Map() + memsize.MapEntry[font.StrikeSpec, *Strike]()
)

// Strike manages the Glyph records for one strike. The font.Strike that generates masks may be purged while the
// Strike lives on; the spec is retained as the key to regenerate it.
type Strike struct {
	strikeCache *StrikeCache
	cache       map[font.PackedGlyphID]*Glyph
	prev        *Strike // LRU bookkeeping, managed under the cache's lock.
	next        *Strike
	spec        font.StrikeSpec
	memoryUsed  int
	removed     bool
}

// Spec returns the strike spec this Strike was created for.
func (t *Strike) Spec() *font.StrikeSpec { return &t.spec }

// GetGlyph returns the Glyph record for packedID, creating it if this is the first request.
func (t *Strike) GetGlyph(packedID font.PackedGlyphID) *Glyph {
	t.strikeCache.mu.Lock()
	defer t.strikeCache.mu.Unlock()
	g := t.cache[packedID]
	if g == nil {
		g = NewGlyph(packedID)
		t.cache[packedID] = g
		t.memoryUsed += glyphRecordOverhead
		if !t.removed {
			t.strikeCache.totalMemoryUsed += glyphRecordOverhead
		}
	}
	return g
}

// StrikeCache holds strikes indexed by spec that generate the individual glyph atlas records.
type StrikeCache struct {
	strikes         map[font.StrikeSpec]*Strike
	head            *Strike // LRU order: head = most recent
	tail            *Strike
	totalMemoryUsed int
	count           int
	sizeLimit       int
	countLimit      int
	mu              sync.Mutex
}

// NewStrikeCache returns an empty cache with the default budgets.
func NewStrikeCache() *StrikeCache {
	return &StrikeCache{
		strikes:    make(map[font.StrikeSpec]*Strike),
		sizeLimit:  defaultGpuFontCacheLimit,
		countLimit: defaultGpuFontCacheCountLimit,
	}
}

// FindOrCreateStrike returns the Strike for spec, creating it if not already cached. A cache hit does not move the
// strike to the LRU head. A spec the map cannot key (font.StrikeSpec.Keyable — an effect whose dynamic type Go refuses
// to hash) gets a fresh uncached strike: in neither the map nor the LRU list, and pre-marked removed so its glyph
// records stay out of the cache's accounting.
func (c *StrikeCache) FindOrCreateStrike(spec *font.StrikeSpec) *Strike {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !spec.Keyable() {
		return &Strike{
			strikeCache: c,
			spec:        *spec,
			cache:       make(map[font.PackedGlyphID]*Glyph),
			memoryUsed:  textStrikeOverhead,
			removed:     true,
		}
	}
	if cached := c.strikes[*spec]; cached != nil {
		return cached
	}
	strike := &Strike{
		strikeCache: c,
		spec:        *spec,
		cache:       make(map[font.PackedGlyphID]*Glyph),
		memoryUsed:  textStrikeOverhead,
	}
	c.strikes[*spec] = strike
	c.attachToHead(strike)
	c.totalMemoryUsed += strike.memoryUsed
	c.count++
	c.internalPurge(0)
	return strike
}

// FreeAll purges every cached strike.
func (c *StrikeCache) FreeAll() {
	c.mu.Lock()
	c.internalPurge(c.totalMemoryUsed)
	c.mu.Unlock()
}

// TotalMemoryUsed reports the cache's byte accounting (test hook).
func (c *StrikeCache) TotalMemoryUsed() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.totalMemoryUsed
}

// StrikeCount reports the number of cached strikes (test hook).
func (c *StrikeCache) StrikeCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.count
}

// internalPurge evicts LRU strikes until both the size and count budgets are satisfied, freeing at least
// minBytesNeeded. Caller holds c.mu.
func (c *StrikeCache) internalPurge(minBytesNeeded int) {
	bytesNeeded := 0
	if c.totalMemoryUsed > c.sizeLimit {
		bytesNeeded = c.totalMemoryUsed - c.sizeLimit
	}
	bytesNeeded = max(bytesNeeded, minBytesNeeded)
	if bytesNeeded > 0 {
		// no small purges!
		bytesNeeded = max(bytesNeeded, c.totalMemoryUsed>>2)
	}
	countNeeded := 0
	if c.count > c.countLimit {
		countNeeded = c.count - c.countLimit
		// no small purges!
		countNeeded = max(countNeeded, c.count>>2)
	}
	if countNeeded == 0 && bytesNeeded == 0 {
		return
	}

	bytesFreed := 0
	countFreed := 0
	// Start at the tail and proceed backwards deleting; the list is in LRU order, with unimportant entries at the tail.
	strike := c.tail
	for strike != nil && (bytesFreed < bytesNeeded || countFreed < countNeeded) {
		prev := strike.prev
		bytesFreed += strike.memoryUsed
		countFreed++
		c.internalRemoveStrike(strike)
		strike = prev
	}
}

// internalRemoveStrike evicts strike from the cache. Caller holds c.mu.
func (c *StrikeCache) internalRemoveStrike(strike *Strike) {
	if strike.removed {
		return
	}
	strike.removed = true
	c.totalMemoryUsed -= strike.memoryUsed
	c.count--
	delete(c.strikes, strike.spec)
	c.detach(strike)
}

// attachToHead inserts strike at the LRU head (most recently used). Caller holds c.mu.
func (c *StrikeCache) attachToHead(strike *Strike) {
	strike.prev = nil
	strike.next = c.head
	if c.head != nil {
		c.head.prev = strike
	}
	c.head = strike
	if c.tail == nil {
		c.tail = strike
	}
}

func (c *StrikeCache) detach(strike *Strike) {
	if strike.prev != nil {
		strike.prev.next = strike.next
	} else if c.head == strike {
		c.head = strike.next
	}
	if strike.next != nil {
		strike.next.prev = strike.prev
	} else if c.tail == strike {
		c.tail = strike.prev
	}
	strike.prev = nil
	strike.next = nil
}
