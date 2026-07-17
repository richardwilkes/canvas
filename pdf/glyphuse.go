// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// glyphUse is the compact record of which glyph IDs of a font were actually drawn. The PDF font resource accumulates
// usage across every draw and, at emit time, the CID glyph-widths array and the ToUnicode CMap iterate the set glyphs.

package pdf

import "math/bits"

// glyphUse is a bit set over [firstNonZero, lastGlyph] plus a slot for glyph 0.
type glyphUse struct {
	bits         []uint64
	firstNonZero uint16
	lastGlyph    uint16
}

// newGlyphUse creates a glyphUse covering glyph 0 plus the range [firstNonZero, lastGlyph]; firstNonZero must be >= 1.
func newGlyphUse(firstNonZero, lastGlyph uint16) glyphUse {
	nbits := int(lastGlyph) - int(firstNonZero) + 2 // +1 for the range, +1 for glyph 0
	return glyphUse{
		bits:         make([]uint64, (nbits+63)/64),
		firstNonZero: firstNonZero,
		lastGlyph:    lastGlyph,
	}
}

// toCode maps a glyph ID to its bit index.
func (g *glyphUse) toCode(gid uint16) int {
	if gid == 0 || g.firstNonZero == 1 {
		return int(gid)
	}
	return int(gid) - int(g.firstNonZero) + 1
}

// set marks gid as used.
func (g *glyphUse) set(gid uint16) {
	code := g.toCode(gid)
	g.bits[code>>6] |= 1 << uint(code&63)
}

// has reports whether gid is marked used.
func (g *glyphUse) has(gid uint16) bool {
	code := g.toCode(gid)
	return g.bits[code>>6]&(1<<uint(code&63)) != 0
}

// forEach calls fn with every used glyph ID in ascending order. Bit index 0 maps to glyph 0; the rest carry the
// firstNonZero offset.
func (g *glyphUse) forEach(fn func(gid uint16)) {
	offset := 0
	if g.firstNonZero != 1 {
		offset = int(g.firstNonZero) - 1
	}
	for word := range g.bits {
		w := g.bits[word]
		for w != 0 {
			bit := bits.TrailingZeros64(w)
			w &= w - 1
			v := word*64 + bit
			if v == 0 || offset == 0 {
				fn(uint16(v))
			} else {
				fn(uint16(v + offset))
			}
		}
	}
}
