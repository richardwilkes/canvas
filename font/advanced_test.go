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
