// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The cmap-only coverage probe: the font manager's character fallback trusts FaceCoversRune* in place of a full font
// parse, so its answer must be the parsed typeface's answer for every face and every code point.

package font

import (
	"os"
	"testing"
)

func TestFaceCoversRuneMatchesTypeface(t *testing.T) {
	faces := []struct {
		file  string
		index int
	}{
		{file: "Roboto-Regular.ttf"},
		{file: "DejaVuSans.subset.ttf"},
		{file: "test.ttc"},
		{file: "test.ttc", index: 1},
		{file: "colr.ttf"},
		{file: "cbdt.ttf"},
		{file: "sbix.ttf"},
		{file: "test_glyphs-glyf_colr_1.ttf"},
	}
	// A mix of mapped and unmapped code points, the U+FFFF .notdef sentinel three of these faces carry, a surrogate,
	// and both out-of-range ends.
	runes := []rune{-1, 0, '\r', ' ', '!', '0', 'A', 'H', 'a', 'x', 0x4E2D, 0xD800, 0x1F600, 0xFFFF, 0x10FFFF, 0x110000}
	var covered, uncovered int
	for _, f := range faces {
		path := "testdata/" + f.file
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		tf := loadTypeface(t, f.file, f.index)
		for _, r := range runes {
			want := tf.UnicharToGlyph(r) != 0
			if want {
				covered++
			} else {
				uncovered++
			}
			if got := FaceCoversRuneFile(path, f.index, r); got != want {
				t.Errorf("FaceCoversRuneFile(%s[%d], %#x) = %v, want %v", f.file, f.index, r, got, want)
			}
			if got := FaceCoversRuneData(data, f.index, r); got != want {
				t.Errorf("FaceCoversRuneData(%s[%d], %#x) = %v, want %v", f.file, f.index, r, got, want)
			}
		}
	}
	// Guard against a vacuous run: the corpus must exercise both answers.
	if covered == 0 || uncovered == 0 {
		t.Errorf("corpus produced %d covered / %d uncovered, want both nonzero", covered, uncovered)
	}
}

// TestFaceRunesDataBoundsTheCmapWalk pins the bound on the per-code-point cmap walk. typesetting's format-12/13
// iterator yields one code point at a time and takes a group's length from EndCharCode-StartCharCode in unsigned
// arithmetic, so a group declaring an EndCharCode of 0xFFFFFFFF — or one below its StartCharCode, which wraps to the
// same count — asks for 4.29e9 iterations and ~17 GB of appended runes. fontmgr.NewFromData calls this for every face
// of every supplied blob, so one malformed embedded font would otherwise kill manager construction.
func TestFaceRunesDataBoundsTheCmapWalk(t *testing.T) {
	roboto := readTestFont(t, "Roboto-Regular.ttf")
	for _, c := range []struct {
		name   string
		groups [][3]uint32
		want   int
	}{
		// The control: a well-formed group is walked to its end, so the bound costs a real font nothing.
		{name: "well-formed", groups: [][3]uint32{{'A', 'C', 1}}, want: 3},
		{name: "unbounded end", groups: [][3]uint32{{0, 0xFFFFFFFF, 1}}, want: maxCmapEntries},
		{name: "end below start", groups: [][3]uint32{{'A', 'A' - 1, 1}}, want: maxCmapEntries},
		// The bound is over the whole walk rather than per group, so a cmap packed with malformed groups costs no more
		// than one of them does.
		{
			name:   "many groups",
			groups: [][3]uint32{{0, 0xFFFFFFFF, 1}, {0, 0xFFFFFFFF, 2}, {0, 0xFFFFFFFF, 3}},
			want:   maxCmapEntries,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			data := sfntWithTables(t, roboto, map[string][]byte{"cmap": synthCmapFormat13(c.groups...)})
			runes := FaceRunesData(data, 0)
			if len(runes) != c.want {
				t.Fatalf("FaceRunesData returned %d runes, want %d", len(runes), c.want)
			}
			// Every group here maps every code point in it to a real glyph, so the answer has to be the contiguous run
			// starting at the first group's start: the walk stops at the bound rather than skipping ahead within a
			// group or reordering what it collects.
			for i, r := range runes {
				if want := rune(c.groups[0][0]) + rune(i); r != want {
					t.Fatalf("rune %d = %#x, want %#x", i, r, want)
				}
			}
		})
	}
	// The bound is far above any real font's coverage, so the unpatched face is unaffected by it.
	if got := len(FaceRunesData(roboto, 0)); got == 0 || got >= maxCmapEntries {
		t.Errorf("Roboto covers %d runes, want a nonzero count well below the %d bound", got, maxCmapEntries)
	}
}

func TestFaceCoversRuneUnknowable(t *testing.T) {
	// Anything that makes the answer unknowable reports false, the same treatment a face whose typeface fails to load
	// gets from the font manager.
	if FaceCoversRuneFile("testdata/does-not-exist.ttf", 0, 'A') {
		t.Error("missing file reported coverage")
	}
	for _, index := range []int{-1, 2, 99} {
		if FaceCoversRuneFile("testdata/test.ttc", index, 'A') {
			t.Errorf("collection index %d (of 2 faces) reported coverage", index)
		}
	}
	if FaceCoversRuneData(nil, 0, 'A') {
		t.Error("nil data reported coverage")
	}
	if FaceCoversRuneData([]byte("not a font at all"), 0, 'A') {
		t.Error("garbage data reported coverage")
	}
	// A real font truncated to its header parses as a collection whose tables are unreadable.
	data, err := os.ReadFile("testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	if FaceCoversRuneData(data[:12], 0, 'A') {
		t.Error("truncated data reported coverage")
	}
}
