// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package font

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

// TestAdvancedMetricsFlags covers the font flags the PDF backend branches on to decide between embedding the font
// program and drawing glyphs as filled paths. Both of the flags here mean "these bytes are not a font program": a
// variable font's program would embed the default instance's outlines under this face's metrics, and a WOFF's tables
// live inside a wrapper (individually compressible), so the container is not an sfnt at all.
func TestAdvancedMetricsFlags(t *testing.T) {
	base := readTestFont(t, "Roboto-Regular.ttf")
	plain, err := NewTypefaceFromData(base, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := plain.GetAdvancedMetrics().Flags; got != 0 {
		t.Errorf("the unmodified font's flags = %#x, want 0", got)
	}

	variable, err := NewTypefaceFromData(sfntWithTables(t, base, map[string][]byte{"fvar": synthFvarTable(1)}), 0)
	if err != nil {
		t.Fatal(err)
	}
	m := variable.GetAdvancedMetrics()
	if m.Flags&FontFlagVariable == 0 {
		t.Errorf("a font with an fvar axis has flags %#x, missing FontFlagVariable", m.Flags)
	}
	if m.Type != FontTypeTrueType {
		t.Errorf("type = %d, want %d (the fvar table must not change the outline format)", m.Type, FontTypeTrueType)
	}
	if got := pdfWouldEmbed(m); got {
		t.Error("a variable font would still be embedded as a font program")
	}

	woff, err := NewTypefaceFromData(woffWrap(t, base), 0)
	if err != nil {
		t.Fatal(err)
	}
	if m = woff.GetAdvancedMetrics(); m.Flags&FontFlagAltDataFormat == 0 {
		t.Errorf("a WOFF-wrapped font has flags %#x, missing FontFlagAltDataFormat", m.Flags)
	}
	if got := pdfWouldEmbed(m); got {
		t.Error("a WOFF-wrapped font would still be embedded as a font program")
	}
	// The same face's own bytes, unwrapped, stay embeddable — the flag tracks the container, not the tables.
	if got := plain.GetAdvancedMetrics().Flags & FontFlagAltDataFormat; got != 0 {
		t.Error("the plain sfnt was reported as an alternate data format")
	}
}

// pdfWouldEmbed mirrors the pdf backend's pdfFontType decision for the flags these tests set: a TrueType program is
// embedded as CIDFontType2 unless one of the "not a font program" flags is present. Duplicated rather than imported
// because pdf depends on font, not the other way around.
func pdfWouldEmbed(m *AdvancedMetrics) bool {
	return m.Type == FontTypeTrueType &&
		m.Flags&(FontFlagVariable|FontFlagAltDataFormat|FontFlagNotEmbeddable) == 0
}

// TestAdvancedMetricsPCLT covers the PCLT lane: it is the only source of the serif and script style flags, and the cap
// height it carries takes precedence over OS/2 sCapHeight, as FreeType's advanced-metrics recipe has it.
func TestAdvancedMetricsPCLT(t *testing.T) {
	base := readTestFont(t, "Roboto-Regular.ttf")
	plain, err := NewTypefaceFromData(base, 0)
	if err != nil {
		t.Fatal(err)
	}
	os2CapHeight := plain.GetAdvancedMetrics().CapHeight
	if os2CapHeight == 0 {
		t.Fatal("Roboto should report a nonzero OS/2 cap height; the precedence case below would be vacuous")
	}
	if got := plain.GetAdvancedMetrics().Style & (StyleSerif | StyleScript); got != 0 {
		t.Errorf("a font with no PCLT table has style bits %#x, want neither serif nor script", got)
	}
	const pcltCapHeight = 1234
	for _, c := range []struct {
		desc       string
		serifStyle uint8
		want       uint32
	}{
		{desc: "unknown", serifStyle: 0},
		{desc: "sans serif", serifStyle: 1},
		{desc: "first serif class", serifStyle: 2, want: StyleSerif},
		{desc: "last serif class", serifStyle: 6, want: StyleSerif},
		{desc: "between the two ranges", serifStyle: 7},
		{desc: "first script class", serifStyle: 9, want: StyleScript},
		{desc: "last script class", serifStyle: 12, want: StyleScript},
		{desc: "past the script classes", serifStyle: 13},
		// The class is the low 6 bits; the top two hold the vertical/horizontal stroke-position flags.
		{desc: "serif class with the high bits set", serifStyle: 0xC0 | 3, want: StyleSerif},
	} {
		data := sfntWithTables(t, base, map[string][]byte{
			"PCLT": synthPCLTTable(pcltCapHeight, c.serifStyle),
		})
		tf, err2 := NewTypefaceFromData(data, 0)
		if err2 != nil {
			t.Fatalf("%s: %v", c.desc, err2)
		}
		m := tf.GetAdvancedMetrics()
		if got := m.Style & (StyleSerif | StyleScript); got != c.want {
			t.Errorf("serifStyle %#x (%s): style bits = %#x, want %#x", c.serifStyle, c.desc, got, c.want)
		}
		if m.CapHeight != pcltCapHeight {
			t.Errorf("serifStyle %#x (%s): cap height = %d, want the PCLT value %d (OS/2 says %d)",
				c.serifStyle, c.desc, m.CapHeight, pcltCapHeight, os2CapHeight)
		}
	}
}

// TestGlyphToUnicodeMapBoundsTheCmapWalk pins the bound on the per-code-point cmap walk behind the PDF backend's
// glyph-to-Unicode map. It is the same unbounded iterator FaceRunesData walks (see maxCmapEntries), reached from
// Document.getUnicodeMap once per embedded font: typesetting's format-12/13 iterator takes a group's length from
// EndCharCode-StartCharCode in unsigned arithmetic, so one group declaring an EndCharCode of 0xFFFFFFFF asks for 4.29e9
// iterations, and a 12 KB cmap holding a thousand of them is hours of CPU inside a single PDF emission.
func TestGlyphToUnicodeMapBoundsTheCmapWalk(t *testing.T) {
	roboto := readTestFont(t, "Roboto-Regular.ttf")
	for _, c := range []struct {
		name   string
		groups [][3]uint32
		// The first (smallest) code point expected for glyphs 1 and 2; 0 means the glyph must be left unmapped.
		wantGlyph1, wantGlyph2 int32
	}{
		// The control: a well-formed cmap is walked to its end, so the bound costs a real font nothing.
		{name: "well-formed", groups: [][3]uint32{{'A', 'C', 1}, {'X', 'Z', 2}}, wantGlyph1: 'A', wantGlyph2: 'X'},
		{
			// A group sized to land exactly on the bound, followed by one only an unbounded walk could reach: glyph 2
			// stays unmapped, which is the bound observed without waiting for a malformed group to spin.
			name:       "stops at the bound",
			groups:     [][3]uint32{{0, maxCmapEntries - 1, 1}, {0x200000, 0x200002, 2}},
			wantGlyph1: 1, // code point 0 loses to 1: a zero entry is indistinguishable from "no code point".
		},
		// The shape that motivates the bound at all. Unbounded, these do not fail — they hang.
		{name: "unbounded end", groups: [][3]uint32{{0, 0xFFFFFFFF, 1}}, wantGlyph1: 1},
		{name: "end below start", groups: [][3]uint32{{'A', 'A' - 1, 1}}, wantGlyph1: 'A'},
	} {
		t.Run(c.name, func(t *testing.T) {
			data := sfntWithTables(t, roboto, map[string][]byte{"cmap": synthCmapFormat13(c.groups...)})
			tf, err := NewTypefaceFromData(data, 0)
			if err != nil {
				t.Fatal(err)
			}
			buffer := tf.GlyphToUnicodeMap()
			if len(buffer) != tf.CountGlyphs() {
				t.Fatalf("map has %d entries, want one per glyph (%d)", len(buffer), tf.CountGlyphs())
			}
			if buffer[1] != c.wantGlyph1 || buffer[2] != c.wantGlyph2 {
				t.Errorf("glyphs 1 and 2 map to %#x and %#x, want %#x and %#x", buffer[1], buffer[2],
					c.wantGlyph1, c.wantGlyph2)
			}
		})
	}
	// The bound is far above any real font's coverage, so the unpatched face is unaffected by it.
	tf, err := NewTypefaceFromData(roboto, 0)
	if err != nil {
		t.Fatal(err)
	}
	mapped := 0
	for _, r := range tf.GlyphToUnicodeMap() {
		if r != 0 {
			mapped++
		}
	}
	if mapped == 0 || mapped >= maxCmapEntries {
		t.Errorf("Roboto maps %d glyphs, want a nonzero count well below the %d bound", mapped, maxCmapEntries)
	}
}

// TestFontProgramSingleFace pins that a typeface parsed from a plain sfnt file embeds its own bytes verbatim: nothing
// is reassembled when there is no collection container to strip.
func TestFontProgramSingleFace(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	raw, _ := tf.FontData()
	program, err := tf.FontProgram()
	if err != nil {
		t.Fatalf("FontProgram: %v", err)
	}
	if !bytes.Equal(program, raw) {
		t.Errorf("single-face program differs from the parsed bytes (%d vs %d bytes)", len(program), len(raw))
	}
	if program, err = EmptyTypeface().FontProgram(); err != nil || program != nil {
		t.Errorf("the empty typeface's program = (%d bytes, %v), want (nil, nil)", len(program), err)
	}
}

// TestFontProgramExtractsCollectionFace covers the embedding lane's collection case: the face's own tables must come
// back as a standalone font, not the 'ttcf' container, and re-parsing that font must reproduce the face it was taken
// from — including its cmap, which is what would silently resolve against face 0 if the container were embedded.
func TestFontProgramExtractsCollectionFace(t *testing.T) {
	container, err := os.ReadFile("testdata/test.ttc")
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 2; index++ {
		tf := loadTypeface(t, "test.ttc", index)
		program, err2 := tf.FontProgram()
		if err2 != nil {
			t.Fatalf("index %d: FontProgram: %v", index, err2)
		}
		if raw, _ := tf.FontData(); bytes.Equal(program, raw) {
			t.Errorf("index %d: the whole collection container was returned as the font program", index)
		}
		if got := binary.BigEndian.Uint32(program); got != 0x00010000 {
			t.Errorf("index %d: sfnt version = %#x, want 0x00010000 (a TrueType font program)", index, got)
		}
		if len(program) >= len(container) {
			t.Errorf("index %d: program is %d bytes, not smaller than the %d-byte collection",
				index, len(program), len(container))
		}
		checkSfntStructure(t, program)

		// The extracted font must be the same face: same metrics, same cmap, same outlines. The two faces of test.ttc
		// disagree on all of those, so a wrong-face extraction cannot pass.
		face, err2 := NewTypefaceFromData(program, 0)
		if err2 != nil {
			t.Fatalf("index %d: re-parsing the extracted program: %v", index, err2)
		}
		if face.FamilyName() != tf.FamilyName() || face.Style() != tf.Style() ||
			face.UnitsPerEm() != tf.UnitsPerEm() || face.CountGlyphs() != tf.CountGlyphs() {
			t.Errorf("index %d: extracted face = (%q, %v, upem %d, %d glyphs), want (%q, %v, upem %d, %d glyphs)",
				index, face.FamilyName(), face.Style(), face.UnitsPerEm(), face.CountGlyphs(),
				tf.FamilyName(), tf.Style(), tf.UnitsPerEm(), tf.CountGlyphs())
		}
		want := tf.GlyphToUnicodeMap()
		got := face.GlyphToUnicodeMap()
		if len(got) != len(want) {
			t.Fatalf("index %d: extracted cmap has %d entries, want %d", index, len(got), len(want))
		}
		for gid := range want {
			if got[gid] != want[gid] {
				t.Errorf("index %d: gid %d maps to U+%04X, want U+%04X", index, gid, got[gid], want[gid])
			}
			id := uint16(gid)
			if face.GlyphDesignBounds(id) != tf.GlyphDesignBounds(id) ||
				face.DesignAdvance(id) != tf.DesignAdvance(id) {
				t.Errorf("index %d: gid %d = %v/%v, want %v/%v", index, gid,
					face.GlyphDesignBounds(id), face.DesignAdvance(id),
					tf.GlyphDesignBounds(id), tf.DesignAdvance(id))
			}
		}
		// The result is a single font, not a collection of one.
		if _, err2 = NewTypefaceFromData(program, 1); err2 == nil {
			t.Errorf("index %d: the extracted program still holds more than one face", index)
		}
	}
}

// checkSfntStructure verifies the assembled font's table directory: a search-range triple that matches numTables,
// in-bounds 4-byte-aligned tables, and a head.checkSumAdjustment that makes the whole font sum to the magic constant.
func checkSfntStructure(t *testing.T, program []byte) {
	t.Helper()
	numTables := int(binary.BigEndian.Uint16(program[4:]))
	if numTables == 0 {
		t.Fatal("assembled font has no tables")
	}
	entrySelector := uint16(0)
	for 1<<(entrySelector+1) <= numTables {
		entrySelector++
	}
	searchRange := uint16(16) << entrySelector
	if got := binary.BigEndian.Uint16(program[6:]); got != searchRange {
		t.Errorf("searchRange = %d, want %d", got, searchRange)
	}
	if got := binary.BigEndian.Uint16(program[8:]); got != entrySelector {
		t.Errorf("entrySelector = %d, want %d", got, entrySelector)
	}
	if got, want := binary.BigEndian.Uint16(program[10:]), uint16(numTables)*16-searchRange; got != want {
		t.Errorf("rangeShift = %d, want %d", got, want)
	}
	for i := 0; i < numTables; i++ {
		record := program[12+16*i:]
		offset := int(binary.BigEndian.Uint32(record[8:]))
		length := int(binary.BigEndian.Uint32(record[12:]))
		if offset%4 != 0 {
			t.Errorf("table %d starts at %d, which is not 4-byte aligned", i, offset)
		}
		if offset < 12+16*numTables || offset+length > len(program) {
			t.Errorf("table %d spans [%d,%d), outside the %d-byte font", i, offset, offset+length, len(program))
		}
	}
	if len(program)%4 != 0 {
		t.Fatalf("font length %d is not a multiple of 4", len(program))
	}
	var sum uint32
	for i := 0; i < len(program); i += 4 {
		sum += binary.BigEndian.Uint32(program[i:])
	}
	if sum != checkSumAdjustmentMagic {
		t.Errorf("whole-font checksum = %#x, want %#x (head.checkSumAdjustment is wrong)",
			sum, uint32(checkSumAdjustmentMagic))
	}
}

// synthCollectionWithTables builds a one-face 'ttcf' container whose table directory holds numTables in-bounds records,
// each naming an empty table. Nothing but the directory geometry is valid, which is all extractFontProgram reads.
func synthCollectionWithTables(numTables int) []byte {
	const headerSize = 16 // 'ttcf', version, numFonts, the single face offset
	data := make([]byte, headerSize+12+16*numTables)
	copy(data, "ttcf")
	binary.BigEndian.PutUint32(data[4:], 0x00010000) // version 1.0
	binary.BigEndian.PutUint32(data[8:], 1)          // numFonts
	binary.BigEndian.PutUint32(data[12:], headerSize)
	binary.BigEndian.PutUint32(data[headerSize:], 0x00010000) // sfntVersion
	binary.BigEndian.PutUint16(data[headerSize+4:], uint16(numTables))
	for i := range numTables {
		record := data[headerSize+12+16*i:]
		// Tags are sequential small integers so none of them is 'head' (which would take the checksum lane).
		binary.BigEndian.PutUint32(record, uint32(i)+1)
		binary.BigEndian.PutUint32(record[8:], uint32(len(data))) // offset: the end of the file, with length 0
	}
	return data
}

// TestExtractFontProgramTableCountLimit pins the sfnt header's uint16 ceiling on the table count. The largest directory
// searchRange can describe (4095 tables, 16<<11) is assembled with a correct binary-search triple; one table more is
// rejected, since 16<<12 is 65536 and would be written — along with rangeShift — as a zero a strict sfnt consumer may
// refuse. numTables comes straight from the file, so a crafted container can ask for any of these.
func TestExtractFontProgramTableCountLimit(t *testing.T) {
	program, err := extractFontProgram(synthCollectionWithTables(maxDescribableTables), 0)
	if err != nil {
		t.Fatalf("%d tables: %v", maxDescribableTables, err)
	}
	for _, c := range []struct {
		field  string
		offset int
		want   uint16
	}{
		{field: "numTables", offset: 4, want: 4095},
		{field: "searchRange", offset: 6, want: 32768}, // 16 << 11
		{field: "entrySelector", offset: 8, want: 11},  // floor(log2(4095))
		{field: "rangeShift", offset: 10, want: 32752}, // 4095*16 - 32768
	} {
		if got := binary.BigEndian.Uint16(program[c.offset:]); got != c.want {
			t.Errorf("%d tables: %s = %d, want %d", maxDescribableTables, c.field, got, c.want)
		}
	}
	if program, err = extractFontProgram(synthCollectionWithTables(maxDescribableTables+1), 0); err == nil {
		t.Errorf("%d tables: no error; the header carries searchRange %d and rangeShift %d",
			maxDescribableTables+1, binary.BigEndian.Uint16(program[6:]), binary.BigEndian.Uint16(program[10:]))
	}
}

// TestExtractFontProgramMalformed covers the damaged-container paths: each must report an error rather than hand back
// bytes that would be embedded as a font program.
func TestExtractFontProgramMalformed(t *testing.T) {
	container, err := os.ReadFile("testdata/test.ttc")
	if err != nil {
		t.Fatal(err)
	}
	dirOffset := int(binary.BigEndian.Uint32(container[12:]))

	truncatedHeader := append([]byte(nil), container[:16]...)
	binary.BigEndian.PutUint32(truncatedHeader[8:], 0xFF) // 255 faces, but no room for their offsets

	// Enough bytes for the directory's fixed header (so numTables is readable) but not for its records.
	shortDirectory := append([]byte(nil), container[:dirOffset+20]...)

	farDirectory := append([]byte(nil), container...)
	binary.BigEndian.PutUint32(farDirectory[12:], uint32(len(container)))

	noTables := append([]byte(nil), container...)
	binary.BigEndian.PutUint16(noTables[dirOffset+4:], 0)

	shortRecords := append([]byte(nil), container...)
	binary.BigEndian.PutUint16(shortRecords[dirOffset+4:], 0xFFF)

	hugeTable := append([]byte(nil), container...)
	binary.BigEndian.PutUint32(hugeTable[dirOffset+12+12:], 0xFFFFFF) // first record's length

	for _, c := range []struct {
		name  string
		data  []byte
		index int
	}{
		{name: "index past the last face", data: container, index: 2},
		{name: "negative index", data: container, index: -1},
		{name: "truncated collection header", data: truncatedHeader},
		{name: "table directory past the end", data: farDirectory},
		{name: "truncated table directory", data: shortDirectory},
		{name: "table directory with no tables", data: noTables},
		{name: "more table records than bytes", data: shortRecords},
		{name: "table extending past the end", data: hugeTable},
	} {
		program, err2 := extractFontProgram(c.data, c.index)
		if err2 == nil {
			t.Errorf("%s: no error, got %d bytes", c.name, len(program))
		}
		if program != nil {
			t.Errorf("%s: returned %d bytes alongside the error", c.name, len(program))
		}
	}

	// Data too short to even hold a collection header is not a collection; it comes back untouched for the parser to
	// reject.
	short := []byte("ttcf")
	if program, err2 := extractFontProgram(short, 0); err2 != nil || !bytes.Equal(program, short) {
		t.Errorf("short data = (%v, %v), want it returned verbatim", program, err2)
	}
}
