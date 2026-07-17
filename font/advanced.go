// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The PDF-embedding surface of the typeface: advanced font metrics, a glyph-to-Unicode map, and the raw font-program
// bytes needed to embed fonts. The PDF backend (pkg pdf) consumes these to embed fonts as CIDFontType2. The sfnt tables
// are read directly rather than through a font-rendering library.

package font

import (
	"github.com/richardwilkes/canvas/geom"

	"github.com/go-text/typesetting/font/opentype"
)

// AdvancedFontType identifies the encoding of the underlying font program.
type AdvancedFontType uint8

// AdvancedFontType values.
const (
	FontTypeType1 AdvancedFontType = iota
	FontTypeType1CID
	FontTypeCFF
	FontTypeTrueType
	FontTypeOther
)

// Advanced style flags (values match the PDF file format).
const (
	StyleFixedPitch uint32 = 0x00000001
	StyleSerif      uint32 = 0x00000002
	StyleScript     uint32 = 0x00000008
	StyleItalic     uint32 = 0x00000040
	StyleAllCaps    uint32 = 0x00010000
	StyleSmallCaps  uint32 = 0x00020000
	StyleForceBold  uint32 = 0x00040000
)

// Advanced font flags.
const (
	FontFlagVariable       uint32 = 1 << 0
	FontFlagNotEmbeddable  uint32 = 1 << 1
	FontFlagNotSubsettable uint32 = 1 << 2
	FontFlagAltDataFormat  uint32 = 1 << 3
)

// AdvancedMetrics holds the per-typeface information the PDF backend needs to embed a font. All the int16 fields except
// ItalicAngle are in font (design) units.
type AdvancedMetrics struct {
	PostScriptName string           // FontName / BaseFont in the PDF
	Style          uint32           // StyleFlags (fixed pitch, italic, ...)
	Type           AdvancedFontType // the underlying font-program encoding
	Flags          uint32           // FontFlags (variable, not embeddable, ...)
	ItalicAngle    int16            // counter-clockwise degrees from vertical
	Ascent         int16            // max height above baseline
	Descent        int16            // max depth below baseline (negative)
	StemV          int16            // thickness of the dominant vertical stem
	CapHeight      int16            // height of a flat capital
	BBox           geom.IRect       // bounding box of all glyphs (font units)
}

// OS/2 fsType embedding-permission bits (FT_FSTYPE_*), used by CanEmbed/CanSubset.
const (
	fsTypeRestrictedLicense uint16 = 0x0002
	fsTypeNoSubsetting      uint16 = 0x0100
	fsTypeBitmapOnly        uint16 = 0x0200
)

// GetAdvancedMetrics returns the raw sfnt values the PDF backend embeds. It returns nil for the empty typeface or an
// out-of-range glyph count. StemV and CapHeight may be zero here; the PDF backend fills reasonable guesses.
func (t *Typeface) GetAdvancedMetrics() *AdvancedMetrics {
	if t.face == nil || t.nGlyphs <= 0 {
		return nil
	}
	m := &AdvancedMetrics{PostScriptName: t.postScriptName, ItalicAngle: t.italicAngle}

	switch {
	case t.hasGlyf:
		m.Type = FontTypeTrueType
	case t.hasCFF:
		m.Type = FontTypeCFF
	default:
		m.Type = FontTypeOther
	}

	if !t.CanEmbed() {
		m.Flags |= FontFlagNotEmbeddable
	}
	if !t.CanSubset() {
		m.Flags |= FontFlagNotSubsettable
	}

	if t.fixedPitch {
		m.Style |= StyleFixedPitch
	}
	if t.style.Slant() != SlantUpright {
		m.Style |= StyleItalic
	}

	if t.hhea != nil {
		m.Ascent = t.hhea.Ascender
		m.Descent = t.hhea.Descender
	}
	// CapHeight comes from OS/2 sCapHeight (version 2+), zero otherwise (the PDF backend guesses then).
	m.CapHeight = t.sCapHgt
	// head bbox → LTRB(xMin, yMax, xMax, yMin) in font units.
	m.BBox = geom.IRectLTRB(int32(t.head.XMin), int32(t.head.YMax), int32(t.head.XMax), int32(t.head.YMin))
	return m
}

// CanEmbed reports whether the OS/2 fsType permits full embedding.
func (t *Typeface) CanEmbed() bool {
	return t.fsType&(fsTypeRestrictedLicense|fsTypeBitmapOnly) == 0
}

// CanSubset reports whether the OS/2 fsType permits subsetting.
func (t *Typeface) CanSubset() bool { return t.fsType&fsTypeNoSubsetting == 0 }

// FontData returns the raw font-program bytes and the collection (TTC) face index. The PDF backend embeds these
// directly as FontFile2. The bytes are the typeface's own storage; callers must not mutate them.
func (t *Typeface) FontData() (data []byte, collectionIndex int) { return t.data, t.collectionIndex }

// GlyphToUnicodeMap returns a slice indexed by glyph ID giving the first (smallest) Unicode code point that maps to
// that glyph through the font's cmap, or 0 when none does. The length is countGlyphs.
func (t *Typeface) GlyphToUnicodeMap() []int32 {
	buffer := make([]int32, t.nGlyphs)
	if t.face == nil || t.nGlyphs == 0 {
		return buffer
	}
	// The typeface is documented thread-safe; iterate the read-only Font cmap (like UnicharToGlyph), not the Face-level
	// lookup, which memoizes into an unsynchronized per-Face cache.
	iter := t.face.Cmap.Iter()
	for iter.Next() {
		r, gid := iter.Char()
		if int(gid) >= t.nGlyphs {
			continue
		}
		// Use the smallest character that maps to this glyph, matching FreeType's ascending
		// FT_Get_First_Char/FT_Get_Next_Char walk (https://crbug.com/359065).
		if buffer[gid] == 0 || (r != 0 && r < buffer[gid]) {
			buffer[gid] = r
		}
	}
	return buffer
}

// GlyphDesignBounds returns the control-box bounds of gid in font (design) units in the y-down glyph-rect convention
// (at unitsPerEm), for the PDF backend's per-glyph clip reject. An empty rect is returned for glyphs with no outline
// (e.g. spaces) and for out-of-range glyphs.
func (t *Typeface) GlyphDesignBounds(gid uint16) geom.Rect {
	if t.face == nil || int(gid) >= t.nGlyphs {
		return geom.Rect{}
	}
	ext, ok := t.faceGlyphExtents(opentype.GID(gid))
	if !ok || (ext.Width == 0 && ext.Height == 0) {
		return geom.Rect{}
	}
	// Extents are y-up (YBearing is the top, Height negative); flip y to the y-down device convention.
	return geom.RectLTRB(ext.XBearing, -ext.YBearing, ext.XBearing+ext.Width, -(ext.YBearing + ext.Height))
}

// DesignAdvance returns the horizontal advance of gid in font (design) units, the advance the PDF backend reads from a
// strike sized at unitsPerEm (linear metrics, no hinting). Out-of-range glyphs and the empty typeface return 0.
func (t *Typeface) DesignAdvance(gid uint16) float32 {
	if t.face == nil || int(gid) >= t.nGlyphs {
		return 0
	}
	return t.faceHAdvance(opentype.GID(gid))
}
