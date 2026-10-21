// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

import (
	"encoding/binary"
	"maps"
	"path/filepath"
	"testing"
	"unicode/utf16"
)

// testNameRecord is one record of a name table built by nameTable. Windows and Unicode records are stored as UTF-16,
// Macintosh ones as bytes.
type testNameRecord struct {
	text                               string
	platform, encoding, language, name uint16
}

// nameTable builds a name table holding the given records.
func nameTable(records ...testNameRecord) []byte {
	table := binary.BigEndian.AppendUint16(nil, 0)
	table = binary.BigEndian.AppendUint16(table, uint16(len(records)))
	table = binary.BigEndian.AppendUint16(table, uint16(6+12*len(records)))
	var storage []byte
	for _, r := range records {
		start := len(storage)
		if r.platform == 1 {
			storage = append(storage, r.text...)
		} else {
			for _, unit := range utf16.Encode([]rune(r.text)) {
				storage = binary.BigEndian.AppendUint16(storage, unit)
			}
		}
		for _, v := range []uint16{r.platform, r.encoding, r.language, r.name, uint16(len(storage) - start), uint16(start)} {
			table = binary.BigEndian.AppendUint16(table, v)
		}
	}
	return append(table, storage...)
}

// TestHintReliantFamilyNames checks which name record decides, as FreeType picks one, and how a name is matched
// against the list.
func TestHintReliantFamilyNames(t *testing.T) {
	const (
		english = 0x409
		chinese = 0x404
	)
	wws := make([]byte, 64)
	binary.BigEndian.PutUint16(wws[62:], os2SelectionWWS)
	for _, c := range []struct {
		name    string
		os2     []byte
		records []testNameRecord
		want    bool
	}{
		{name: "an ordinary family", records: []testNameRecord{{"Arial", 3, 1, english, nameIDFamily}}},
		{name: "a listed family", want: true, records: []testNameRecord{{"MingLiU", 3, 1, english, nameIDFamily}}},
		{name: "a listed name inside a longer one", want: true, records: []testNameRecord{
			{"DFHei-Md-HK-BF", 3, 1, english, nameIDFamily},
		}},
		{name: "a subset tag in front", want: true, records: []testNameRecord{
			{"ABCDEF+DFKai-SB", 3, 1, english, nameIDFamily},
		}},
		{name: "a character outside ASCII where the list has '?'", want: true, records: []testNameRecord{
			{"HuaTianKaiTi体", 3, 1, english, nameIDFamily},
		}},
		{name: "the English Windows record over a later one", records: []testNameRecord{
			{"Arial", 3, 1, english, nameIDFamily},
			{"MingLiU", 3, 1, chinese, nameIDFamily},
		}},
		{name: "a Macintosh record over a Windows one that is not English", want: true, records: []testNameRecord{
			{"Arial", 3, 1, chinese, nameIDFamily},
			{"MingLiU", 1, 0, 0, nameIDFamily},
		}},
		{name: "the typographic family over the plain one", records: []testNameRecord{
			{"MingLiU", 3, 1, english, nameIDFamily},
			{"Arial", 3, 1, english, nameIDTypographicFamily},
		}},
		{name: "the WWS family first", want: true, records: []testNameRecord{
			{"Arial", 3, 1, english, nameIDTypographicFamily},
			{"MingLiU", 3, 1, english, nameIDWWSFamily},
		}},
		{name: "the WWS family ignored when the OS/2 bit says the others already are", os2: wws, records: []testNameRecord{
			{"Arial", 3, 1, english, nameIDTypographicFamily},
			{"MingLiU", 3, 1, english, nameIDWWSFamily},
		}},
	} {
		if got := hintReliantName(nameTable(c.records...), c.os2); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

// TestHintReliantTableIDs checks that a font is recognized by the checksums and lengths of its hinting tables alone,
// with the tables that are absent counting as empty.
func TestHintReliantTableIDs(t *testing.T) {
	if got, want := tableIDOf([]byte{0, 0, 0, 1, 0xFF}), (tableID{checksum: 0xFF000001, length: 5}); got != want {
		t.Errorf("table identity %+v, want %+v", got, want)
	}
	// A font with no control values whose programs sum to those of NEC fadpop7.ttf.
	want := hintReliantIDs[16]
	if want.cvt != (tableID{}) {
		t.Fatal("the list changed under the test")
	}
	withSum := func(id tableID) []byte {
		data := make([]byte, id.length)
		binary.BigEndian.PutUint32(data, id.checksum)
		return data
	}
	tables := map[string][]byte{"fpgm": withSum(want.fpgm), "prep": withSum(want.prep)}
	if !isHintReliant(lookup(tables)) {
		t.Error("a font with a listed font's hinting tables was not recognized")
	}
	tables["cvt "] = []byte{0, 1}
	if isHintReliant(lookup(tables)) {
		t.Error("a font with different control values was recognized")
	}
}

// TestHintReliantFontRunsWithoutBackwardCompatibility checks what recognizing a font is for: its glyph programs move
// points horizontally, which those of every other font are kept from doing under a smooth target.
func TestHintReliantFontRunsWithoutBackwardCompatibility(t *testing.T) {
	tables := sfntTableMap(t, filepath.Join("..", "..", "testdata", "Roboto-Regular.ttf"))
	plain, err := NewFont(lookup(tables))
	if err != nil {
		t.Fatal(err)
	}
	if plain.IsHintReliant() {
		t.Fatal("Roboto was taken for a font that needs its instructions")
	}
	renamed := maps.Clone(tables)
	renamed["name"] = nameTable(testNameRecord{"MingLiU", 3, 1, 0x409, nameIDFamily})
	reliant, err := NewFont(lookup(renamed))
	if err != nil {
		t.Fatal(err)
	}
	if !reliant.IsHintReliant() {
		t.Fatal("the renamed font was not recognized")
	}
	plainInstance, err := plain.NewInstance(13, TargetLCD)
	if err != nil {
		t.Fatal(err)
	}
	reliantInstance, err := reliant.NewInstance(13, TargetLCD)
	if err != nil {
		t.Fatal(err)
	}
	if !plainInstance.backwardCompatibility() || reliantInstance.backwardCompatibility() {
		t.Fatalf("backward compatibility: %v for the plain font and %v for the renamed one, want true and false",
			plainInstance.backwardCompatibility(), reliantInstance.backwardCompatibility())
	}
	// With the same programs, the two differ only in the horizontal moves the plain font's are denied.
	var a, b Outline
	movedX := 0
	for gid := range plain.numGlyphs {
		if !plainInstance.Outline(uint16(gid), &a) || !reliantInstance.Outline(uint16(gid), &b) {
			continue
		}
		for i := range a.Points {
			if a.Points[i].X != b.Points[i].X {
				movedX++
				break
			}
		}
	}
	if movedX < plain.numGlyphs/4 {
		t.Errorf("only %d of %d glyphs were hinted horizontally", movedX, plain.numGlyphs)
	}
}
