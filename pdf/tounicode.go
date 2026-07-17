// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Builds the /ToUnicode CMap that maps a font's glyph IDs back to Unicode so PDF text can be copied/searched. The map
// is built from the typeface's reverse cmap (font.Typeface.GlyphToUnicodeMap); the extended (cluster-text) map a full
// clusterator would populate is always empty here because glyph runs carry no source text (unison does its own line
// layout, so no cluster text reaches the public surface), so the bfchar-ex section is dropped.

package pdf

import (
	"github.com/richardwilkes/canvas/stream"
)

// appendToUnicodeHeader writes the CMap's standard preamble (dict setup and codespace range).
func appendToUnicodeHeader(cmap stream.WStream, multibyte bool) {
	// 12 dict begin: an Adobe-suggested value that must not change (prevents old Readers from malfunctioning).
	writeText(cmap, "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n")
	// The /CIDSystemInfo must be consistent with the one in the CIDFont; the format differs so it can't be a shared
	// reference object.
	writeText(cmap, "/CIDSystemInfo\n<<  /Registry (Adobe)\n/Ordering (UCS)\n/Supplement 0\n>> def\n")
	writeText(cmap, "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n")
	if multibyte {
		writeText(cmap, "<0000> <FFFF>\n")
	} else {
		writeText(cmap, "<00> <FF>\n")
	}
	writeText(cmap, "endcodespacerange\n")
}

// appendCmapFooter writes the CMap's standard closing boilerplate.
func appendCmapFooter(cmap stream.WStream) {
	writeText(cmap, "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend")
}

type bfChar struct {
	glyphID uint16
	unicode int32
}

type bfRange struct {
	start   uint16
	end     uint16
	unicode int32
}

// writeCmapGlyph writes a glyph ID as one or two hex bytes, matching the font's byte width.
func writeCmapGlyph(cmap stream.WStream, multiByte bool, gid uint16) {
	if multiByte {
		writeUInt16BE(cmap, gid)
	} else {
		writeUInt8(cmap, uint8(gid))
	}
}

// appendBFCharSection writes the glyph→Unicode entries as beginbfchar blocks; each block holds at most 100 entries, the
// PDF spec's per-section limit.
func appendBFCharSection(bfchar []bfChar, multiByte bool, cmap stream.WStream) {
	for i := 0; i < len(bfchar); i += 100 {
		count := min(len(bfchar)-i, 100)
		writeDecAsText(cmap, count)
		writeText(cmap, " beginbfchar\n")
		for j := 0; j < count; j++ {
			writeText(cmap, "<")
			writeCmapGlyph(cmap, multiByte, bfchar[i+j].glyphID)
			writeText(cmap, "> <")
			writeUTF16beHex(cmap, bfchar[i+j].unicode)
			writeText(cmap, ">\n")
		}
		writeText(cmap, "endbfchar\n")
	}
}

// appendBFRangeSection writes the glyph→Unicode entries as beginbfrange blocks, each holding at most 100 entries.
func appendBFRangeSection(bfrange []bfRange, multiByte bool, cmap stream.WStream) {
	for i := 0; i < len(bfrange); i += 100 {
		count := min(len(bfrange)-i, 100)
		writeDecAsText(cmap, count)
		writeText(cmap, " beginbfrange\n")
		for j := 0; j < count; j++ {
			writeText(cmap, "<")
			writeCmapGlyph(cmap, multiByte, bfrange[i+j].start)
			writeText(cmap, "> <")
			writeCmapGlyph(cmap, multiByte, bfrange[i+j].end)
			writeText(cmap, "> <")
			writeUTF16beHex(cmap, bfrange[i+j].unicode)
			writeText(cmap, ">\n")
		}
		writeText(cmap, "endbfrange\n")
	}
}

// appendCmapSections generates the <bfchar> and <bfrange> tables from the reverse cmap, guaranteeing the two do not
// overlap and that all bfchar entries precede bfrange entries.
func appendCmapSections(glyphToUnicode []int32, subset *glyphUse, cmap stream.WStream, multiByteGlyphs bool, firstGlyphID, lastGlyphID uint16) {
	glyphOffset := 0
	if !multiByteGlyphs {
		glyphOffset = int(firstGlyphID) - 1
	}

	var bfcharEntries []bfChar
	var bfrangeEntries []bfRange

	currentRangeEntry := bfRange{}
	rangeEmpty := true
	limit := int(lastGlyphID) + 1 - glyphOffset

	for i := int(firstGlyphID) - glyphOffset; i < limit+1; i++ {
		gid := uint16(i + glyphOffset)
		inSubset := i < limit && (subset == nil || subset.has(gid))
		if !rangeEmpty {
			// The PDF spec requires bfrange entries not to change the high byte: <1035> <10FF> <2222> is ok, but <1035>
			// <1100> <2222> is not.
			inRange := i == int(currentRangeEntry.end)+1 &&
				i>>8 == int(currentRangeEntry.start)>>8 &&
				i < limit &&
				glyphToUnicode[gid] == currentRangeEntry.unicode+int32(i)-int32(currentRangeEntry.start)
			if !inSubset || !inRange {
				if currentRangeEntry.end > currentRangeEntry.start {
					bfrangeEntries = append(bfrangeEntries, currentRangeEntry)
				} else {
					bfcharEntries = append(bfcharEntries,
						bfChar{glyphID: currentRangeEntry.start, unicode: currentRangeEntry.unicode})
				}
				rangeEmpty = true
			}
		}
		if inSubset {
			currentRangeEntry.end = uint16(i)
			if rangeEmpty {
				currentRangeEntry.start = uint16(i)
				currentRangeEntry.unicode = glyphToUnicode[gid]
				rangeEmpty = false
			}
		}
	}

	// The spec requires all bfchar entries for a font to come before bfrange entries.
	appendBFCharSection(bfcharEntries, multiByteGlyphs, cmap)
	appendBFRangeSection(bfrangeEntries, multiByteGlyphs, cmap)
}

// makeToUnicodeCmap builds the complete /ToUnicode CMap stream bytes.
func makeToUnicodeCmap(glyphToUnicode []int32, subset *glyphUse, multiByteGlyphs bool, firstGlyphID, lastGlyphID uint16) []byte {
	cmap := stream.NewMemoryWStream()
	appendToUnicodeHeader(cmap, multiByteGlyphs)
	appendCmapSections(glyphToUnicode, subset, cmap, multiByteGlyphs, firstGlyphID, lastGlyphID)
	appendCmapFooter(cmap)
	return cmap.DetachAsData()
}
