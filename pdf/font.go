// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The CIDFontType2 embedding lane: the per-typeface PDF font resource, the document-level canonicalization of advanced
// metrics and reverse cmaps, the FontType decision, and emitFont (the Type0 + CIDFontType2 + FontFile2 + ToUnicode
// object graph). HarfBuzz/ICU are disabled, so — exactly as the cskia reference oracle in this configuration — the
// *full* font program is embedded (no subsetting) and the ToUnicode CMap is built purely from the cmap reverse mapping.
//
// Only glyf-outline (TrueType) faces embed here; CFF/variable/not-embeddable faces, mask filters, and perspective fall
// back to drawing glyphs as filled paths (drawGlyphRunAsPath in device.go) rather than a Type3 fallback, which is out
// of scope. Since glyph runs here carry no cluster text, the per-cluster grouping, the ActualText marked-content
// sequences, and the extended (Ex) ToUnicode map are all unreachable and dropped.

package pdf

import (
	"math"
	"sort"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
)

// kPdfSymbolic is PDF's symbolic-font flag bit in the FontDescriptor /Flags entry; it is always set since the embedded
// fonts here are always treated as symbolic.
const kPdfSymbolic = 4

// pdfFont holds the state for one embedded CIDFontType2 font resource: the typeface, the accumulated glyph usage, its
// reserved indirect reference, and the font-program type. One pdfFont exists per typeface per document — the fill,
// no-mask-filter case, where the path strike is sized at unitsPerEm and the paint carries no stroke/effect/mask filter.
type pdfFont struct {
	typeface   *font.Typeface
	glyphUsage glyphUse
	ref        IndirectReference
	fontType   font.AdvancedFontType
}

// multiByteGlyphs reports whether this font's encoding uses two-byte glyph indices.
func (f *pdfFont) multiByteGlyphs() bool {
	return f.fontType == font.FontTypeType1CID || f.fontType == font.FontTypeTrueType ||
		f.fontType == font.FontTypeCFF
}

func (f *pdfFont) firstGlyphID() uint16 { return f.glyphUsage.firstNonZero }
func (f *pdfFont) lastGlyphID() uint16  { return f.glyphUsage.lastGlyph }

// glyphToPDFFontEncoding maps a glyph ID to its encoded value in the font's /Encoding (identity for the multibyte
// lane).
func (f *pdfFont) glyphToPDFFontEncoding(gid uint16) uint16 {
	if f.multiByteGlyphs() || gid == 0 {
		return gid
	}
	return gid - f.firstGlyphID() + 1
}

// noteGlyphUsage records that gid was used, expanding the font's tracked glyph range.
func (f *pdfFont) noteGlyphUsage(gid uint16) { f.glyphUsage.set(gid) }

// getFontMetrics caches the typeface's advanced metrics per document, filling StemV/CapHeight guesses when the typeface
// doesn't supply them and prepending the always-present subset tag. Returns nil (and caches nil) for a bad typeface.
func (d *Document) getFontMetrics(tf *font.Typeface) *font.AdvancedMetrics {
	id := tf.UniqueID()
	if m, ok := d.typefaceMetrics[id]; ok {
		return m
	}
	// A font with no glyphs or more than can be indexed by a 16-bit CID is unusable.
	if count := tf.CountGlyphs(); count <= 0 || count > 1+0xFFFF {
		d.typefaceMetrics[id] = nil
		return nil
	}
	m := tf.GetAdvancedMetrics()
	if m == nil {
		d.typefaceMetrics[id] = nil
		return nil
	}
	if m.StemV == 0 || m.CapHeight == 0 {
		f := font.NewFont(tf, 1000, 1, 0) // glyph coordinate system
		f.SetHinting(font.HintingNone)
		if m.StemV == 0 {
			// A good guess for StemV: min width of i, I, !, 1.
			stemV := int16(math.MaxInt16)
			for _, c := range []rune{'i', 'I', '!', '1'} {
				stemV = minInt16(stemV, int16(geom.RoundToInt(glyphBoundsFor(f, tf, c).Width())))
			}
			m.StemV = stemV
		}
		if m.CapHeight == 0 {
			// A good guess for CapHeight: average the height of M and X.
			var capHeight float32
			for _, c := range []rune{'M', 'X'} {
				capHeight += glyphBoundsFor(f, tf, c).Height()
			}
			m.CapHeight = int16(geom.RoundToInt(capHeight / 2))
		}
	}
	// Subsetted fonts conventionally get a six-letter tag prefix (e.g. "ABCDEF+Arial"); the tag is prepended here too,
	// for output parity with subsetted fonts, even though the full font program is always embedded (no actual
	// subsetting).
	m.PostScriptName = d.nextFontSubsetTag() + m.PostScriptName
	d.typefaceMetrics[id] = m
	return m
}

// glyphBoundsFor returns the fill bounds of the glyph for rune c at the measuring font's size.
func glyphBoundsFor(f *font.Font, tf *font.Typeface, c rune) geom.Rect {
	var b [1]geom.Rect
	f.GlyphBounds([]uint16{tf.UnicharToGlyph(c)}, b[:])
	return b[0]
}

// getUnicodeMap caches the typeface's reverse cmap (glyph ID to Unicode) per document.
func (d *Document) getUnicodeMap(tf *font.Typeface) []int32 {
	id := tf.UniqueID()
	if m, ok := d.toUnicodeMaps[id]; ok {
		return m
	}
	m := tf.GlyphToUnicodeMap()
	d.toUnicodeMaps[id] = m
	return m
}

// nextFontSubsetTag returns the next subset tag: six uppercase letters followed by '+', unique per document.
func (d *Document) nextFontSubsetTag() string {
	this := d.nextFontSubsetTagValue
	d.nextFontSubsetTagValue = (d.nextFontSubsetTagValue + 1) % 308915776
	var b [7]byte
	for i := 0; i < 6; i++ {
		b[i] = 'A' + byte(this%26)
		this /= 26
	}
	b[6] = '+'
	return string(b[:])
}

// pdfFontType decides the font resource type, forcing the path-fallback case (FontTypeOther) for the encodings drawn as
// filled paths instead — variable, not-embeddable, alt-data (WOFF), CFF, and any mask-filtered draw.
func pdfFontType(hasMaskFilter bool, m *font.AdvancedMetrics) font.AdvancedFontType {
	if m.Flags&font.FontFlagVariable != 0 ||
		m.Flags&font.FontFlagAltDataFormat != 0 ||
		m.Flags&font.FontFlagNotEmbeddable != 0 ||
		m.Type == font.FontTypeCFF ||
		hasMaskFilter {
		return font.FontTypeOther
	}
	return m.Type
}

// getFontResource returns the CIDFontType2 resource for the multibyte lane: one per typeface, created lazily with a
// reserved reference and accumulating glyph usage over the whole document.
func (d *Document) getFontResource(tf *font.Typeface, fontType font.AdvancedFontType) *pdfFont {
	id := tf.UniqueID()
	if f, ok := d.fonts[id]; ok {
		return f
	}
	last := uint16(tf.CountGlyphs() - 1)
	f := &pdfFont{
		typeface:   tf,
		glyphUsage: newGlyphUse(1, last),
		ref:        d.ReserveRef(),
		fontType:   fontType,
	}
	f.noteGlyphUsage(0) // always include glyph 0
	d.fonts[id] = f
	return f
}

// fromFontUnits scales a font-unit value to the base-1000 glyph space PDF font metrics use.
func fromFontUnits(scaled float32, emSize uint16) float32 {
	if emSize == 1000 {
		return scaled
	}
	return scaled * 1000 / float32(emSize)
}

// scaleFromFontUnits is fromFontUnits for an integer input, returning a float32.
func scaleFromFontUnits(val int, emSize uint16) float32 { return fromFontUnits(float32(val), emSize) }

// populateCommonFontDescriptor fills in the FontDescriptor entries shared by every embedded font (FontName, Flags,
// Ascent/Descent, StemV, CapHeight, ItalicAngle, FontBBox, and MissingWidth).
func populateCommonFontDescriptor(descriptor *Dict, m *font.AdvancedMetrics, emSize uint16, defaultWidth int16) {
	descriptor.InsertName("FontName", m.PostScriptName)
	descriptor.InsertInt("Flags", int32(m.Style|kPdfSymbolic))
	descriptor.InsertScalar("Ascent", scaleFromFontUnits(int(m.Ascent), emSize))
	descriptor.InsertScalar("Descent", scaleFromFontUnits(int(m.Descent), emSize))
	descriptor.InsertScalar("StemV", scaleFromFontUnits(int(m.StemV), emSize))
	descriptor.InsertScalar("CapHeight", scaleFromFontUnits(int(m.CapHeight), emSize))
	descriptor.InsertInt("ItalicAngle", int32(m.ItalicAngle))
	descriptor.InsertObject("FontBBox", makeArrayScalars(
		scaleFromFontUnits(int(m.BBox.Left), emSize),
		scaleFromFontUnits(int(m.BBox.Bottom), emSize),
		scaleFromFontUnits(int(m.BBox.Right), emSize),
		scaleFromFontUnits(int(m.BBox.Top), emSize),
	))
	if defaultWidth > 0 {
		descriptor.InsertScalar("MissingWidth", scaleFromFontUnits(int(defaultWidth), emSize))
	}
}

// insertFontProgram inserts the /FontFile2 entry — the embedded font program stream, with its uncompressed size in
// /Length1 — into a FontDescriptor. A typeface whose asset couldn't be read yields no bytes; the entry is then omitted
// entirely rather than embedding an empty program with /Length1 0 that the CIDFontType2 descendant would still point
// at. Reports whether the program was embedded.
func insertFontProgram(doc *Document, descriptor *Dict, data []byte) bool {
	if len(data) == 0 {
		return false
	}
	streamDict := NewDict()
	streamDict.InsertInt("Length1", int32(len(data)))
	descriptor.InsertRef("FontFile2", doc.StreamOut(streamDict, data, true))
	return true
}

// emitFont emits the full object graph for one CIDFontType2 (TrueType) font: the FontDescriptor with the full FontFile2
// program (omitted when the font program is unavailable, as upstream does), the CIDFontType2 descendant (Identity
// CIDToGIDMap, /W widths, /DW default), and the Type0 wrapper (Identity-H, /ToUnicode). Called once per font at
// document close.
func emitFont(doc *Document, f *pdfFont) {
	tf := f.typeface
	m := doc.getFontMetrics(tf)
	if m == nil {
		return
	}
	emSize := uint16(geom.RoundToInt(float32(tf.UnitsPerEm())))

	descriptor := NewTypedDict("FontDescriptor")
	populateCommonFontDescriptor(descriptor, m, emSize, 0)

	data, _ := tf.FontData()
	insertFontProgram(doc, descriptor, data)

	newCIDFont := NewTypedDict("Font")
	newCIDFont.InsertRef("FontDescriptor", doc.Emit(descriptor))
	newCIDFont.InsertName("BaseFont", m.PostScriptName)
	newCIDFont.InsertName("Subtype", "CIDFontType2")
	newCIDFont.InsertName("CIDToGIDMap", "Identity")

	sysInfo := NewDict()
	sysInfo.InsertByteString("Registry", "Adobe")
	sysInfo.InsertByteString("Ordering", "Identity")
	sysInfo.InsertInt("Supplement", 0)
	newCIDFont.InsertObject("CIDSystemInfo", sysInfo)

	// Poppler enforces that /DW is an integer, so the widths array only considers integer advances.
	widths, defaultWidth := makeCIDGlyphWidthsArray(tf, &f.glyphUsage)
	if widths != nil && widths.Size() > 0 {
		newCIDFont.InsertObject("W", widths)
	}
	newCIDFont.InsertInt("DW", defaultWidth)

	fontDict := NewTypedDict("Font")
	fontDict.InsertName("Subtype", "Type0")
	fontDict.InsertName("BaseFont", m.PostScriptName)
	fontDict.InsertName("Encoding", "Identity-H")
	descendantFonts := NewArray()
	descendantFonts.AppendRef(doc.Emit(newCIDFont))
	fontDict.InsertObject("DescendantFonts", descendantFonts)

	glyphToUnicode := doc.getUnicodeMap(tf)
	toUnicode := makeToUnicodeCmap(glyphToUnicode, &f.glyphUsage, f.multiByteGlyphs(),
		f.firstGlyphID(), f.lastGlyphID())
	fontDict.InsertRef("ToUnicode", doc.StreamOut(nil, toUnicode, true))

	doc.EmitAt(fontDict, f.ref)
}

// emitFonts emits every used font, sorted by object number so the output is reproducible.
func (d *Document) emitFonts() {
	fonts := make([]*pdfFont, 0, len(d.fonts))
	for _, f := range d.fonts {
		fonts = append(fonts, f)
	}
	sort.Slice(fonts, func(i, j int) bool { return fonts[i].ref.value < fonts[j].ref.value })
	for _, f := range fonts {
		emitFont(d, f)
	}
}

func minInt16(a, b int16) int16 {
	if a < b {
		return a
	}
	return b
}
