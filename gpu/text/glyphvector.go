// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GlyphVector delays the lookup of GPU Glyphs until the single-threaded flush. It holds the source font.Strike plus its
// spec — after PackedGlyphIDToGlyph the strike pointer is dropped (so the CPU strike cache can purge it and the GC can
// collect it) and the spec can regenerate it on demand. GlyphVariant carries both the packed ID and the resolved Glyph,
// since the regenerate loop needs both. regenerateAtlasForGL lives in gpu/gl and reaches the fields below through
// exported accessors.

package text

import (
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/gpu"
)

// GlyphVariant pairs a packed glyph ID with its resolved Glyph. Initially only PackedID is meaningful; Glyph is filled
// by PackedGlyphIDToGlyph.
type GlyphVariant struct {
	Glyph    *Glyph
	PackedID font.PackedGlyphID
}

// GlyphVector is a lazily-resolved run of glyphs awaiting atlas placement.
type GlyphVector struct {
	sourceStrike    *font.Strike // strong ref keeping the source strike alive; dropped after conversion
	textStrike      *Strike
	glyphs          []GlyphVariant
	bulkUseUpdater  gpu.BulkUsePlotUpdater
	spec            font.StrikeSpec
	atlasGeneration uint64
}

// MakeGlyphVector returns a GlyphVector for the given packed glyph IDs, resolved lazily against strike/spec.
func MakeGlyphVector(strike *font.Strike, spec *font.StrikeSpec, packedIDs []font.PackedGlyphID) GlyphVector {
	if len(packedIDs) == 0 {
		panic("empty glyph vector")
	}
	glyphs := make([]GlyphVariant, len(packedIDs))
	for i, id := range packedIDs {
		glyphs[i].PackedID = id
	}
	return GlyphVector{
		spec:            *spec,
		sourceStrike:    strike,
		glyphs:          glyphs,
		atlasGeneration: gpu.InvalidAtlasGeneration,
	}
}

// GlyphCount returns the number of glyphs in the vector.
func (v *GlyphVector) GlyphCount() int { return len(v.glyphs) }

// Variants exposes the resolved variant slice (the regenerate loop reads both fields).
func (v *GlyphVector) Variants() []GlyphVariant { return v.glyphs }

// Glyphs returns the resolved Glyph pointers (valid after PackedGlyphIDToGlyph).
func (v *GlyphVector) Glyphs() []*Glyph {
	out := make([]*Glyph, len(v.glyphs))
	for i := range v.glyphs {
		out[i] = v.glyphs[i].Glyph
	}
	return out
}

// Strike returns the GPU strike resolved by PackedGlyphIDToGlyph.
func (v *GlyphVector) Strike() *Strike { return v.textStrike }

// AtlasGeneration returns the generation the texture coordinates were computed against.
func (v *GlyphVector) AtlasGeneration() uint64 { return v.atlasGeneration }

// SetAtlasGeneration records the generation after a full regeneration.
func (v *GlyphVector) SetAtlasGeneration(gen uint64) { v.atlasGeneration = gen }

// BulkUseUpdater returns the plot updater tracking this vector's atlas plots.
func (v *GlyphVector) BulkUseUpdater() *gpu.BulkUsePlotUpdater { return &v.bulkUseUpdater }

// PackedGlyphIDToGlyph resolves this vector's GPU strike and Glyph records. Must run in single-threaded (flush) mode.
// On first use it resolves the GPU strike and the Glyph records, then drops the source-strike ref so the CPU strike can
// be purged if needed.
func (v *GlyphVector) PackedGlyphIDToGlyph(cache *StrikeCache) {
	if v.textStrike == nil {
		v.textStrike = cache.FindOrCreateStrike(&v.spec)
		// Get all the atlas locations for each glyph.
		for i := range v.glyphs {
			v.glyphs[i].Glyph = v.textStrike.GetGlyph(v.glyphs[i].PackedID)
		}
		// Drop the ref to the strike so that it can be purged if needed.
		v.sourceStrike = nil
	}
}
