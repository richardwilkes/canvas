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
	"encoding/binary"
	"errors"
	"fmt"

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

// Byte offsets of the two PCLT fields the advanced metrics use, within the 54-byte table (version, fontNumber, pitch,
// xHeight, style, typeFamily, capHeight, symbolSet, typeface[16], characterComplement[8], fileName[6], strokeWeight,
// widthType, serifStyle, reserved).
const (
	pcltCapHeightOffset  = 16
	pcltSerifStyleOffset = 52
)

// PCLT serifStyle classes (the low 6 bits of the field), the ranges FreeType's advanced-metrics recipe maps to the
// serif and script style flags.
const (
	pcltSerifStyleMask = 0x3F
	pcltSerifFirst     = 2
	pcltSerifLast      = 6
	pcltScriptFirst    = 9
	pcltScriptLast     = 12
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

	if t.hasFvar {
		m.Flags |= FontFlagVariable
	}
	if !t.CanEmbed() {
		m.Flags |= FontFlagNotEmbeddable
	}
	if !t.CanSubset() {
		m.Flags |= FontFlagNotSubsettable
	}
	// Only an outline program is a candidate for embedding at all, so — as upstream does — the alt-data flag is reported
	// just for those: a container the sfnt reader accepts but that is not a plain sfnt file (WOFF, dfont) holds tables
	// that are individually compressed or offset within a wrapper, so its bytes are not a font program even though every
	// table parsed.
	if (m.Type == FontTypeTrueType || m.Type == FontTypeCFF) && !isStandardSfntData(t.data) {
		m.Flags |= FontFlagAltDataFormat
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
	// Cap height and the serif class come from PCLT when the font carries one, following FreeType's advanced-metrics
	// recipe; otherwise cap height is OS/2 sCapHeight (version 2+) and zero when that is absent too (the PDF backend
	// guesses then). Only PCLT classifies a face as serif or script, so a font without one emits neither flag.
	if t.hasPCLT {
		m.CapHeight = t.pcltCapHeight
		switch serifStyle := t.pcltSerifStyle & pcltSerifStyleMask; {
		case serifStyle >= pcltSerifFirst && serifStyle <= pcltSerifLast:
			m.Style |= StyleSerif
		case serifStyle >= pcltScriptFirst && serifStyle <= pcltScriptLast:
			m.Style |= StyleScript
		}
	} else {
		m.CapHeight = t.sCapHgt
	}
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

// FontData returns the raw bytes the typeface was parsed from and the collection (TTC) face index within them. For a
// collection these are the whole container, not this face's font program; use FontProgram for something embeddable. The
// bytes are the typeface's own storage; callers must not mutate them.
func (t *Typeface) FontData() (data []byte, collectionIndex int) { return t.data, t.collectionIndex }

// FontProgram returns a standalone sfnt font program for this face, which is what a PDF /FontFile2 must hold. A
// typeface parsed from a single-face file returns its own bytes; one parsed from a collection returns a freshly
// assembled font holding just this face's tables, since a 'ttcf' container is not a font program and its glyph IDs
// resolve against face 0 rather than this face. Returns nil and an error for a collection whose header or table
// directory is malformed. The single-face result is the typeface's own storage; callers must not mutate it.
func (t *Typeface) FontProgram() ([]byte, error) {
	return extractFontProgram(t.data, t.collectionIndex)
}

// ttcTag is the four-byte 'ttcf' signature that starts a TrueType/OpenType collection file.
const ttcTag = 0x74746366

// The other four-byte signatures a plain sfnt file may start with: 0x00010000 (TrueType), 'true' (Apple TrueType),
// 'typ1' (PostScript Type 1) and 'OTTO' (OpenType CFF).
const (
	trueTypeTag      = 0x00010000
	appleTrueTypeTag = 0x74727565
	postScript1Tag   = 0x74797031
	openTypeCFFTag   = 0x4F54544F
)

// headTag is the four-byte 'head' table tag.
const headTag = 0x68656164

// headCheckSumAdjustmentOffset is the byte offset of checkSumAdjustment within the head table.
const headCheckSumAdjustmentOffset = 8

// checkSumAdjustmentMagic is the constant the sum of an sfnt font's 32-bit words must add up to, per the OpenType head
// table specification.
const checkSumAdjustmentMagic = 0xB1B0AFBA

// maxDescribableTables is the largest table count an sfnt header's uint16 searchRange (16<<entrySelector) can express.
const maxDescribableTables = 1<<12 - 1

// isStandardSfntData reports whether data starts with one of the signatures of a plain sfnt file or collection, i.e.
// whether its bytes are a font program (or a container one can be extracted from). The sfnt reader also accepts WOFF and
// dfont, whose tables live inside a wrapper and may be individually compressed; those parse fine but their bytes are not
// embeddable, which is what FontFlagAltDataFormat reports.
func isStandardSfntData(data []byte) bool {
	if len(data) < 4 {
		return false
	}
	switch binary.BigEndian.Uint32(data) {
	case trueTypeTag, appleTrueTypeTag, postScript1Tag, openTypeCFFTag, ttcTag:
		return true
	default:
		return false
	}
}

// extractFontProgram returns the standalone sfnt font program for face index of data. Data that is not a collection is
// returned unchanged. A collection face is reassembled into its own font: the sfnt header, the face's table directory
// with every offset rebased onto the new file, the table data (4-byte aligned, as the specification requires), and a
// recomputed head.checkSumAdjustment. Table contents are copied verbatim, so the per-table checksums already in the
// directory stay valid (head's is defined with checkSumAdjustment treated as zero, so rewriting that field cannot
// invalidate it).
func extractFontProgram(data []byte, index int) ([]byte, error) {
	if len(data) < 12 || binary.BigEndian.Uint32(data) != ttcTag {
		return data, nil
	}
	numFonts := int64(binary.BigEndian.Uint32(data[8:]))
	if index < 0 || int64(index) >= numFonts {
		return nil, fmt.Errorf("font: collection index %d out of range (%d faces)", index, numFonts)
	}
	if 12+4*numFonts > int64(len(data)) {
		return nil, errors.New("font: truncated collection header")
	}
	dirOffset := int64(binary.BigEndian.Uint32(data[12+4*index:]))
	if dirOffset+12 > int64(len(data)) {
		return nil, fmt.Errorf("font: collection face %d has an out-of-range table directory", index)
	}
	numTables := int(binary.BigEndian.Uint16(data[dirOffset+4:]))
	if numTables == 0 {
		return nil, fmt.Errorf("font: collection face %d has no tables", index)
	}
	// The header's searchRange is a uint16 holding 16<<entrySelector, so it cannot describe a directory of 4096 or more
	// tables: 16*2^12 is 65536, which truncates to 0 and takes rangeShift with it. Such a directory is rejected rather
	// than emitted with a zeroed binary-search header a strict sfnt consumer would refuse.
	if numTables > maxDescribableTables {
		return nil, fmt.Errorf("font: collection face %d has %d tables, more than an sfnt header can describe",
			index, numTables)
	}
	if dirOffset+12+16*int64(numTables) > int64(len(data)) {
		return nil, fmt.Errorf("font: collection face %d has a truncated table directory", index)
	}

	// The directory is a fixed-size header the tables are appended after, so it can be sized and written up front and
	// then have each record's rebased offset filled in as that table is copied.
	out := make([]byte, 12+16*numTables)
	binary.BigEndian.PutUint32(out, binary.BigEndian.Uint32(data[dirOffset:])) // sfntVersion
	binary.BigEndian.PutUint16(out[4:], uint16(numTables))
	entrySelector := uint16(0)
	for 1<<(entrySelector+1) <= numTables {
		entrySelector++
	}
	searchRange := uint16(16) << entrySelector
	binary.BigEndian.PutUint16(out[6:], searchRange)
	binary.BigEndian.PutUint16(out[8:], entrySelector)
	binary.BigEndian.PutUint16(out[10:], uint16(numTables)*16-searchRange)

	headOffset := -1
	for i := range numTables {
		record := data[dirOffset+12+16*int64(i):]
		tag := binary.BigEndian.Uint32(record)
		offset := int64(binary.BigEndian.Uint32(record[8:]))
		length := int64(binary.BigEndian.Uint32(record[12:]))
		if offset+length > int64(len(data)) {
			return nil, fmt.Errorf("font: collection face %d has an out-of-range table", index)
		}
		for len(out)%4 != 0 { // tables must start on a 4-byte boundary
			out = append(out, 0)
		}
		tableOffset := len(out)
		out = append(out, data[offset:offset+length]...)
		// The appends may have moved the backing array, so re-slice the record only now.
		dst := out[12+16*i:]
		binary.BigEndian.PutUint32(dst, tag)
		binary.BigEndian.PutUint32(dst[4:], binary.BigEndian.Uint32(record[4:])) // checkSum
		binary.BigEndian.PutUint32(dst[8:], uint32(tableOffset))
		binary.BigEndian.PutUint32(dst[12:], uint32(length))
		if tag == headTag && length >= headCheckSumAdjustmentOffset+4 {
			headOffset = tableOffset
		}
	}
	for len(out)%4 != 0 { // the whole-font checksum below reads 32-bit words
		out = append(out, 0)
	}
	if headOffset >= 0 {
		adjustment := out[headOffset+headCheckSumAdjustmentOffset:]
		binary.BigEndian.PutUint32(adjustment, 0)
		var sum uint32
		for i := 0; i < len(out); i += 4 {
			sum += binary.BigEndian.Uint32(out[i:])
		}
		binary.BigEndian.PutUint32(adjustment, checkSumAdjustmentMagic-sum)
	}
	return out, nil
}

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
