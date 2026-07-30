// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The global LRU cache of strikes (budget: 2 MiB / 2048 strikes by default) and the spec constructors the glyph-run
// painter and text APIs use. Strikes are keyed on the comparable ScalerRec plus effect identity (the effects are
// immutable objects, so pointer identity is a sound equality check). The rec's float fields are NaN-free by
// construction (ScalerRec.canonicalizeKeyFloats) — a NaN anywhere in the key would make every lookup miss and every
// delete a no-op, stranding the strike in the map forever. An effect that cannot be hashed at all (a value-receiver
// implementation over unhashable state) is not keyable and skips the cache entirely (StrikeSpec.Keyable).

package font

import (
	"hash/maphash"
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

// Default cache limits.
const (
	defaultFontCacheLimit      = 2 * 1024 * 1024
	defaultFontCacheCountLimit = 2048
)

// StrikeCache is an LRU list of strikes with byte and count budgets.
type StrikeCache struct {
	strikes         map[strikeKey]*Strike
	head            *Strike // LRU order: head = most recent
	tail            *Strike
	totalMemoryUsed int
	count           int
	sizeLimit       int
	countLimit      int
	mu              sync.Mutex
}

// strikeKey is the real (comparable) key type.
type strikeKey struct {
	effects ScalerEffects
	rec     ScalerRec
}

// globalStrikeCache is the process-wide default cache instance.
var globalStrikeCache = NewStrikeCache()

// GlobalStrikeCache returns the process-wide strike cache.
func GlobalStrikeCache() *StrikeCache { return globalStrikeCache }

// NewStrikeCache returns an empty cache with the default budgets.
func NewStrikeCache() *StrikeCache {
	return &StrikeCache{
		strikes:    make(map[strikeKey]*Strike),
		sizeLimit:  defaultFontCacheLimit,
		countLimit: defaultFontCacheCountLimit,
	}
}

// FindOrCreateStrike looks up the strike matching spec, creating and caching one if none exists yet. A spec the map
// cannot key (see StrikeSpec.Keyable) gets a fresh uncached strike.
func (c *StrikeCache) FindOrCreateStrike(spec *StrikeSpec) *Strike {
	if !spec.Keyable() {
		return c.newUncachedStrike(spec)
	}
	key := strikeKey{rec: spec.Rec, effects: spec.Effects}
	c.mu.Lock()
	s := c.strikes[key]
	if s != nil {
		c.moveToHead(s)
	} else {
		s = newStrike(c, &key, NewScalerContext(spec.Typeface, spec.Rec, spec.Effects))
		c.strikes[key] = s
		c.attachToHead(s)
		c.totalMemoryUsed += s.memoryUsed
		c.count++
	}
	c.internalPurge(0)
	c.mu.Unlock()
	return s
}

// newUncachedStrike builds a strike that lives outside the cache: in neither the map nor the LRU list, and pre-marked
// removed so the glyph bytes it generates never enter the cache's accounting (nothing can ever evict it to take them
// back out). Its key is left zero — it is never looked up or deleted by key.
func (c *StrikeCache) newUncachedStrike(spec *StrikeSpec) *Strike {
	s := newStrike(c, &strikeKey{}, NewScalerContext(spec.Typeface, spec.Rec, spec.Effects))
	s.removed = true
	return s
}

// PurgeAll evicts every cached strike (for tests and memory pressure hooks).
func (c *StrikeCache) PurgeAll() {
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

// internalPurge evicts LRU strikes until at least minBytesNeeded bytes and both budgets are satisfied. Caller holds
// c.mu.
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
	s := c.tail
	for s != nil && (bytesFreed < bytesNeeded || countFreed < countNeeded) {
		prev := s.prev
		bytesFreed += s.memoryUsed
		countFreed++
		c.removeStrike(s)
		s = prev
	}
}

// removeStrike evicts s from the cache. Caller holds c.mu.
func (c *StrikeCache) removeStrike(s *Strike) {
	if s.removed {
		return
	}
	s.removed = true
	c.totalMemoryUsed -= s.memoryUsed
	c.count--
	delete(c.strikes, s.keyVal)
	c.detach(s)
}

func (c *StrikeCache) attachToHead(s *Strike) {
	s.prev = nil
	s.next = c.head
	if c.head != nil {
		c.head.prev = s
	}
	c.head = s
	if c.tail == nil {
		c.tail = s
	}
}

func (c *StrikeCache) detach(s *Strike) {
	if s.prev != nil {
		s.prev.next = s.next
	} else if c.head == s {
		c.head = s.next
	}
	if s.next != nil {
		s.next.prev = s.prev
	} else if c.tail == s {
		c.tail = s.prev
	}
	s.prev = nil
	s.next = nil
}

func (c *StrikeCache) moveToHead(s *Strike) {
	if c.head == s {
		return
	}
	c.detach(s)
	c.attachToHead(s)
}

// StrikeSpec holds everything needed to find or create a strike.
type StrikeSpec struct {
	Effects  ScalerEffects
	Typeface *Typeface
	Rec      ScalerRec
}

// FindOrCreateStrike resolves the spec through the global cache.
func (spec *StrikeSpec) FindOrCreateStrike() *Strike {
	return GlobalStrikeCache().FindOrCreateStrike(spec)
}

// Keyable reports whether the spec (or any part of it) can be used as a map key, which every strike cache keyed on one
// must check first. The typeface pointer and the rec — a struct of scalars — always can, but stroke.PathEffect and
// maskfilter.MaskFilter are exported interfaces with exported methods, so a caller outside this module can implement
// them with a value receiver over a struct holding a slice, map, or func; Go panics with "hash of unhashable type" the
// moment such a value reaches a map key, which would take down the first text draw that used the effect. A spec that
// answers false gets an uncached strike instead: correct output, no caching, no panic. (Upstream Skia keys on a
// serialized SkDescriptor byte blob, which cannot be poisoned this way.)
func (spec *StrikeSpec) Keyable() bool {
	return effectKeyable(spec.Effects.PathEffect) && effectKeyable(spec.Effects.MaskFilter)
}

// effectKeySeed is the seed effectKeyable's hash probe runs against. Nothing consumes the hashes, so the seed's value
// is irrelevant; a fresh one just keeps the probe honest about being a hash.
var effectKeySeed = maphash.MakeSeed()

// effectKeyable reports whether effect can be hashed as part of a map key. The test is an actual hash attempt with the
// panic recovered, rather than a reflect.Type.Comparable test: a comparable type can still hold an unhashable value in
// an interface-typed field, and only hashing it reveals that. maphash.Comparable hashes — and panics — exactly as the
// map key would.
func effectKeyable(effect any) (ok bool) {
	if effect == nil {
		return true
	}
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	// The hash itself is not wanted, only the attempt: a value the map cannot key panics here instead of there.
	_ = maphash.Comparable(effectKeySeed, effect)
	return true
}

// MakeMaskSpec builds the strike spec for rendering a glyph mask. props carries the destination surface's pixel
// geometry for the LCD16 lane (nil means unknown geometry).
func MakeMaskSpec(f *Font, paint *ScalerPaint, deviceMatrix *geom.Matrix, props *DeviceProps) StrikeSpec {
	rec, effects := MakeRecAndEffects(f, paint, deviceMatrix, props)
	return StrikeSpec{Typeface: f.typeface, Rec: rec, Effects: effects}
}

// MakePathSpec builds the canonical-size path strike plus the strike-to-source scale. Path strikes never render LCD
// (setupForAsPaths rewrites EdgingSubpixelAntiAlias to EdgingAntiAlias), so no props parameter is needed. The strike
// always comes out sub-pixel-positioned, whatever the source font asked for: setupForAsPaths force-sets flagSubpixel,
// since the sub-pixel position is applied when the canonical-size path is transformed to the screen.
func MakePathSpec(f *Font, paint *ScalerPaint) (spec StrikeSpec, strikeToSourceScale float32) {
	pathFont := *f

	var pathPaint ScalerPaint
	if paint != nil {
		pathPaint = *paint
	}
	strikeToSourceScale = pathFont.setupForAsPaths(&pathPaint)

	identity := geom.IdentityMatrix()
	rec, effects := MakeRecAndEffects(&pathFont, &pathPaint, &identity, nil)
	return StrikeSpec{Typeface: f.typeface, Rec: rec, Effects: effects}, strikeToSourceScale
}

// MakeTransformMaskSpec builds the strike for masks that will be transformed to the screen (sub-pixel positioning is
// meaningless there, so it is disabled).
func MakeTransformMaskSpec(f *Font, paint *ScalerPaint, deviceMatrix *geom.Matrix, props *DeviceProps) StrikeSpec {
	sourceFont := *f
	sourceFont.SetSubpixel(false)
	rec, effects := MakeRecAndEffects(&sourceFont, paint, deviceMatrix, props)
	return StrikeSpec{Typeface: f.typeface, Rec: rec, Effects: effects}
}

// MakeSDFTMaskSpec builds the SDF strike spec for the E.1 distance-field text lane: the scaler generates its distance
// field directly as a rec format (rec.Format = MaskSDF) rather than through a mask-filter effect, so setting that one
// field is all that's needed beyond the identity-matrix mask spec. A8 coverage is always gamma-free, so no gamma
// adjustment is needed here (unlike the shaded distance-field draw itself). The caller (gpu/text's makeSDFTStrikeSpec)
// prepares the font/paint pair beforehand.
func MakeSDFTMaskSpec(f *Font, paint *ScalerPaint) StrikeSpec {
	identity := geom.IdentityMatrix()
	rec, effects := MakeRecAndEffects(f, paint, &identity, nil)
	rec.Format = MaskSDF
	// The field is always generated from anti-aliased coverage, whatever the run font's edging: aliased distance-field
	// text is the shader's monochrome step over the same field, not a hard-edged field (upstream's
	// SkStrikeSpec::MakeSDFT forces kAntiAlias for the same reason). The SDF font gpu/text derives normalizes the
	// edging already; this keeps a direct caller from degrading the field.
	rec.Flags &^= recFlagAliased
	return StrikeSpec{Typeface: f.typeface, Rec: rec, Effects: effects}
}

// MakeWithNoDeviceSpec builds a strike spec with the identity device matrix and unknown pixel geometry (paint may be
// nil).
func MakeWithNoDeviceSpec(f *Font, paint *ScalerPaint) StrikeSpec {
	identity := geom.IdentityMatrix()
	rec, effects := MakeRecAndEffects(f, paint, &identity, nil)
	return StrikeSpec{Typeface: f.typeface, Rec: rec, Effects: effects}
}

// ShouldDrawAsPathMatrix reports whether the font/paint combination should be measured/drawn as a path rather than
// through the glyph cache, under the given view matrix.
func ShouldDrawAsPathMatrix(paint *ScalerPaint, f *Font, viewMatrix *geom.Matrix) bool {
	// Hairline glyphs are fast enough, so we don't need to cache them.
	if paint != nil && paint.Style == stroke.PaintStyleStroke && paint.Width == 0 {
		return true
	}
	// We don't cache perspective.
	if viewMatrix.HasPerspective() {
		return true
	}
	var textMatrix geom.Matrix
	textMatrix.SetAll(f.size*f.scaleX, f.size*f.skewX, 0, 0, f.size, 0, 0, 0, 1)
	textMatrix.PostConcat(viewMatrix)

	// We have a self-imposed maximum, just to limit memory usage.
	const maxSizeSquared = 256 * 256
	distance := func(xIndex, yIndex int) float32 {
		return textMatrix.Get(xIndex)*textMatrix.Get(xIndex) + textMatrix.Get(yIndex)*textMatrix.Get(yIndex)
	}
	return distance(geom.MScaleX, geom.MSkewY) > maxSizeSquared ||
		distance(geom.MSkewX, geom.MScaleY) > maxSizeSquared
}
