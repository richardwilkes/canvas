// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-scaler glyph cache with lazily generated metrics, paths, and mask images, plus the accept/reject decisions
// the glyph-run painter consumes. Drawable glyphs are not implemented (no reachable scaler produces them); the GPU
// actions (ActionDirectMask/ActionMask) are dimension-only gates for the atlas lanes.

package font

import (
	"sync"
)

// GlyphAction records the disposition of a glyph for a given ActionType.
type GlyphAction uint8

// GlyphAction values.
const (
	GlyphActionUnset GlyphAction = iota
	GlyphActionAccept
	GlyphActionReject
	GlyphActionDrop
)

// ActionType identifies which consumer is asking for a glyph's disposition (the CPU subset).
type ActionType uint8

// ActionType values.
const (
	ActionDirectMaskCPU ActionType = iota
	ActionPath
	// ActionDirectMask is the GPU direct-mask gate (fits in the atlas at device scale). Dimension check only; the image
	// is generated at flush time (PrepareImage).
	ActionDirectMask
	// ActionMask is the GPU transformed-mask gate (fits in the atlas with room for the bilerp padding ring).
	ActionMask
	// ActionSDFT is the GPU distance-field gate — the glyph must carry an SDF mask (an SDF strike rec) and fit in the
	// atlas at its padded size.
	ActionSDFT
	actionTypeCount
)

// SideTooBigForAtlas is the largest glyph dimension (in pixels) that can fit in the GPU glyph atlas.
const SideTooBigForAtlas = 256

// glyphEntry pairs a glyph with its resolved actions (the per-action-type dispositions the CPU painter needs).
type glyphEntry struct {
	glyph   *Glyph
	actions [actionTypeCount]GlyphAction
}

// Strike is a glyph cache for one scaler configuration. All methods are safe for concurrent use.
type Strike struct {
	cache        *StrikeCache
	scaler       *ScalerContext
	glyphs       map[PackedGlyphID]*glyphEntry
	prev         *Strike // LRU bookkeeping, managed under the cache's lock.
	next         *Strike
	keyVal       strikeKey
	memoryUsed   int
	metrics      Metrics
	roundingSpec RoundingSpec
	mu           sync.Mutex
	removed      bool
}

// glyphOverhead approximates the per-glyph struct overhead for the memory budget.
const glyphOverhead = 88

func newStrike(cache *StrikeCache, key *strikeKey, scaler *ScalerContext) *Strike {
	return &Strike{
		cache:        cache,
		keyVal:       *key,
		scaler:       scaler,
		roundingSpec: NewRoundingSpec(scaler.rec.isSubpixel(), scaler.rec.computeAxisAlignmentForHText()),
		metrics:      scaler.getFontMetrics(),
		glyphs:       make(map[PackedGlyphID]*glyphEntry),
		memoryUsed:   1024, // baseline strike overhead
	}
}

// RoundingSpec returns the strike's position rounding rules.
func (s *Strike) RoundingSpec() RoundingSpec { return s.roundingSpec }

// FontMetrics returns the strike's font-wide metrics.
func (s *Strike) FontMetrics() Metrics { return s.metrics }

// entryFor returns (creating if needed) the glyph entry for packedID. Caller holds s.mu; the memory increase is
// accumulated into *increase.
func (s *Strike) entryFor(packedID PackedGlyphID, increase *int) *glyphEntry {
	e := s.glyphs[packedID]
	if e == nil {
		g := s.scaler.makeGlyph(packedID)
		e = &glyphEntry{glyph: g}
		if g.IsEmpty() {
			for i := range e.actions {
				e.actions[i] = GlyphActionDrop
			}
		}
		s.glyphs[packedID] = e
		*increase += glyphOverhead
	}
	return e
}

// DigestFor resolves the action for the given actionType, generating the path or image as required, and returns the
// glyph with its action.
func (s *Strike) DigestFor(actionType ActionType, packedID PackedGlyphID) (*Glyph, GlyphAction) {
	increase := 0
	s.mu.Lock()
	e := s.entryFor(packedID, &increase)
	if e.actions[actionType] == GlyphActionUnset {
		action := GlyphActionReject
		switch actionType {
		case ActionDirectMaskCPU:
			if s.prepareForImage(e.glyph, &increase) {
				action = GlyphActionAccept
			}
		case ActionPath:
			if s.prepareForPath(e.glyph, &increase) {
				action = GlyphActionAccept
			}
		case ActionDirectMask:
			// Fits in the atlas at device scale.
			if e.glyph.MaxDimension() <= SideTooBigForAtlas {
				action = GlyphActionAccept
			}
		case ActionMask:
			// Fits in the atlas with room left over for the bilerp padding.
			if e.glyph.MaxDimension() <= SideTooBigForAtlas-2 {
				action = GlyphActionAccept
			}
		case ActionSDFT:
			// Fits in the atlas and carries an SDF mask (the padding is baked into the SDF glyph's dimensions).
			if e.glyph.MaxDimension() <= SideTooBigForAtlas && e.glyph.Format == MaskSDF {
				action = GlyphActionAccept
			}
		}
		e.actions[actionType] = action
	}
	g, action := e.glyph, e.actions[actionType]
	s.mu.Unlock()
	s.updateMemoryUsage(increase)
	return g, action
}

// prepareForImage generates g's mask image if not already done. Caller holds s.mu.
func (s *Strike) prepareForImage(g *Glyph, increase *int) bool {
	if !g.imageDone {
		g.imageDone = true
		if size := g.ImageSize(); size > 0 {
			allocGlyphImage(g)
			s.scaler.getImage(g)
			*increase += size
		}
	}
	return g.HasImage()
}

// prepareForPath generates g's path if not already done. Caller holds s.mu.
func (s *Strike) prepareForPath(g *Glyph, increase *int) bool {
	if !g.pathDone {
		s.scaler.internalGetPath(g)
		*increase += approximatePathBytes(g.Path())
	}
	return g.Path() != nil
}

// PrepareImage resolves the glyph for packedID with its mask image generated (the lane the GPU atlas regeneration path
// reads through). The returned glyph may still have no image (empty or too-large glyphs).
func (s *Strike) PrepareImage(packedID PackedGlyphID) *Glyph {
	increase := 0
	s.mu.Lock()
	e := s.entryFor(packedID, &increase)
	s.prepareForImage(e.glyph, &increase)
	g := e.glyph
	s.mu.Unlock()
	s.updateMemoryUsage(increase)
	return g
}

// Metrics resolves metrics-only glyphs for the given IDs.
func (s *Strike) Metrics(glyphIDs []uint16, results []*Glyph) []*Glyph {
	increase := 0
	s.mu.Lock()
	results = results[:0]
	for _, gid := range glyphIDs {
		results = append(results, s.entryFor(PackGlyphID(gid), &increase).glyph)
	}
	s.mu.Unlock()
	s.updateMemoryUsage(increase)
	return results
}

// PreparePaths resolves glyphs with their paths.
func (s *Strike) PreparePaths(glyphIDs []uint16, results []*Glyph) []*Glyph {
	increase := 0
	s.mu.Lock()
	results = results[:0]
	for _, gid := range glyphIDs {
		e := s.entryFor(PackGlyphID(gid), &increase)
		s.prepareForPath(e.glyph, &increase)
		results = append(results, e.glyph)
	}
	s.mu.Unlock()
	s.updateMemoryUsage(increase)
	return results
}

// FindIntercepts returns the x-interval carved out of the [bounds[0], bounds[1]] horizontal band by the glyph's path,
// cached per band, scaled and offset into place. The results are appended to array (which may be nil to count only);
// count is updated.
func (s *Strike) FindIntercepts(bounds [2]float32, scale, xPos float32, g *Glyph, array []float32, count *int) []float32 {
	s.mu.Lock()
	defer s.mu.Unlock()

	offsetResults := func(interval [2]float32) {
		if array != nil {
			array = append(array, interval[0]*scale+xPos, interval[1]*scale+xPos)
		}
		*count += 2
	}

	for _, ic := range g.intercepts {
		if ic.bounds == bounds {
			if ic.interval[0] < ic.interval[1] {
				offsetResults(ic.interval)
			}
			return array
		}
	}

	ic := &glyphIntercept{
		bounds:   bounds,
		interval: [2]float32{scalarMax, scalarMin},
	}
	g.intercepts = append(g.intercepts, ic)
	p := g.Path()
	if p == nil {
		return array
	}
	pathBounds := p.Bounds()
	if pathBounds.Bottom < bounds[0] || bounds[1] < pathBounds.Top {
		return array
	}
	left, right := calculatePathGap(bounds[0], bounds[1], p)
	if left >= right {
		return array
	}
	ic.interval[0] = left
	ic.interval[1] = right
	offsetResults(ic.interval)
	return array
}

// updateMemoryUsage folds this strike's growth into the cache totals.
func (s *Strike) updateMemoryUsage(increase int) {
	if increase > 0 {
		s.cache.mu.Lock()
		s.memoryUsed += increase
		if !s.removed {
			s.cache.totalMemoryUsed += increase
		}
		s.cache.mu.Unlock()
	}
}

const (
	scalarMax = float32(3.402823466e+38)
	scalarMin = -scalarMax
)
