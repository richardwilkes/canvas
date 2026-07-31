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
	"fmt"
	"os"
	"slices"
	"testing"
)

func TestFaceCoversRuneMatchesTypeface(t *testing.T) {
	// A mix of mapped and unmapped code points, the U+FFFF .notdef sentinel three of these faces carry, a surrogate,
	// and both out-of-range ends.
	runes := []rune{-1, 0, '\r', ' ', '!', '0', 'A', 'H', 'a', 'x', 0x4E2D, 0xD800, 0x1F600, 0xFFFF, 0x10FFFF, 0x110000}
	var covered, uncovered int
	for _, f := range faceCorpus {
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

// faceCorpus is the mixed set of test faces the cmap-probe cases run over: outline, subset, collection (both indices),
// COLRv0/v1, and the two bitmap formats.
var faceCorpus = []struct {
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

// TestFaceRunesDataMatchesFaceCoversRune pins FaceRunesData's documented equivalence with FaceCoversRuneData — it is
// "FaceCoversRuneData asked about every character at once" — in both directions and over the whole code space. Only the
// per-rune probe is checked elsewhere, yet fontmgr.NewFromData builds each face's whole rune set from this one instead,
// and that set is what the in-memory manager's character fallback filters candidates by: a returned rune the face does
// not really map makes the index over-claim coverage (the filter is what keeps the U+FFFF .notdef sentinel three of
// these faces carry out of it), and a missing one loses the face for that character before its cmap is ever consulted.
func TestFaceRunesDataMatchesFaceCoversRune(t *testing.T) {
	for _, f := range faceCorpus {
		t.Run(fmt.Sprintf("%s[%d]", f.file, f.index), func(t *testing.T) {
			data := readTestFont(t, f.file)
			runes := FaceRunesData(data, f.index)
			if len(runes) == 0 {
				t.Fatal("a face with a usable cmap reported no coverage at all")
			}
			// Nothing returned may be absent from the per-rune probe, and nothing may be listed twice — fontscan's
			// RuneSet would swallow a duplicate, but it means the walk revisited a code point.
			set := make(map[rune]bool, len(runes))
			for i, r := range runes {
				if set[r] {
					t.Errorf("rune %#x is listed twice (again at index %d)", r, i)
				}
				set[r] = true
				if !FaceCoversRuneData(data, f.index, r) {
					t.Errorf("FaceRunesData listed %#x, which FaceCoversRuneData rejects", r)
				}
			}
			// And nothing the face really maps may be missing. The probe answers exactly what the parsed typeface
			// answers (TestFaceCoversRuneMatchesTypeface), so the parsed face is the cheap ground truth for a sweep of
			// the whole code space.
			tf := loadTypeface(t, f.file, f.index)
			for r := rune(0); r <= 0x10FFFF; r++ {
				if want := tf.UnicharToGlyph(r) != 0; want != set[r] {
					t.Fatalf("rune %#x: FaceRunesData lists it = %v, but the face maps it = %v", r, set[r], want)
				}
			}
		})
	}
}

// TestFaceRunesDataFiltersUnmappedGlyphs pins the three filters that make the returned set exact where a scanned
// fontscan footprint over-claims: a cmap entry resolving to glyph 0 is not coverage, neither is one resolving past the
// 0xFFFF ceiling an sfnt glyph store can hold, and neither is one whose code point is outside Unicode — a cmap's code
// points are just 32-bit values, and fontscan.RuneSet.Add pages on uint16(r>>8), so 0x1000041 would fold onto page 0
// bit 0x41 and make fontmgr's in-memory index claim the face covers U+0041. All three are what the per-rune probe
// already answers, so dropping any would put runes in that index the face's own cmap check then rejects.
func TestFaceRunesDataFiltersUnmappedGlyphs(t *testing.T) {
	roboto := readTestFont(t, "Roboto-Regular.ttf")
	data := sfntWithTables(t, roboto, map[string][]byte{"cmap": synthCmapFormat13(
		[3]uint32{'A', 'C', 1},               // real glyphs: kept
		[3]uint32{'D', 'F', 0},               // .notdef: not coverage
		[3]uint32{'G', 'I', 0x10000},         // past the sfnt glyph-ID ceiling: not coverage
		[3]uint32{0x1000041, 0x1000041, 1},   // past the last code point: not coverage
		[3]uint32{0xFFFFFFF0, 0xFFFFFFF0, 1}, // and again where the rune would come out negative
	)})
	want := []rune{'A', 'B', 'C'}
	if got := FaceRunesData(data, 0); !slices.Equal(got, want) {
		t.Errorf("FaceRunesData = %#x, want %#x", got, want)
	}
	for _, r := range []rune{'D', 'E', 'F', 'G', 'H', 'I', 0x1000041, -16} {
		if FaceCoversRuneData(data, 0, r) {
			t.Errorf("FaceCoversRuneData(%#x) = true, so the two lanes disagree about the filter", r)
		}
	}
}

// TestFaceRunesDataUnknowable pins the nil contract: everything that makes the answer unknowable yields nil, matching
// FaceCoversRuneData's false. fontmgr.NewFromData walks every face of every supplied blob, so these are ordinary inputs
// for it rather than error cases.
func TestFaceRunesDataUnknowable(t *testing.T) {
	roboto := readTestFont(t, "Roboto-Regular.ttf")
	for _, c := range []struct {
		name  string
		data  []byte
		index int
	}{
		{name: "nil data"},
		{name: "garbage data", data: []byte("not a font at all")},
		{name: "truncated to the header", data: roboto[:12]},
		{name: "negative index", data: roboto, index: -1},
		{name: "index past the end", data: roboto, index: 1},
		{name: "collection index past the end", data: readTestFont(t, "test.ttc"), index: 2},
		{name: "no cmap table", data: sfntWithoutTables(t, roboto, "cmap")},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := FaceRunesData(c.data, c.index); got != nil {
				t.Errorf("FaceRunesData returned %d runes, want nil", len(got))
			}
		})
	}
}

// TestFaceRunesDataBoundsTheCmapWalk pins the bound on the per-code-point cmap walk. typesetting's format-12/13
// iterator yields one code point at a time and takes a group's length from EndCharCode-StartCharCode in unsigned
// arithmetic, so a group declaring an EndCharCode of 0xFFFFFFFF — or one below its StartCharCode, which wraps to the
// same count — asks for 4.29e9 iterations. fontmgr.NewFromData calls this for every face of every supplied blob, so one
// malformed embedded font would otherwise hang manager construction. The count of entries examined is what pins the
// bound: the runes returned cannot, since a malformed group's code points climb straight past U+10FFFF, where the
// code-point filter rejects every one of them, so an unbounded walk would return the same runes as this one.
func TestFaceRunesDataBoundsTheCmapWalk(t *testing.T) {
	roboto := readTestFont(t, "Roboto-Regular.ttf")
	for _, c := range []struct {
		name       string
		groups     [][3]uint32
		want       int
		wantWalked int
	}{
		// The control: a well-formed group is walked to its end, so the bound costs a real font nothing.
		{name: "well-formed", groups: [][3]uint32{{'A', 'C', 1}}, want: 3, wantWalked: 3},
		{
			name: "unbounded end", groups: [][3]uint32{{0, 0xFFFFFFFF, 1}},
			want: maxCmapEntries, wantWalked: maxCmapEntries,
		},
		{
			// Starting at 'A' pushes the walk's last 'A' entries past U+10FFFF, so the returned count falls short
			// of the entries examined by exactly that much.
			name: "end below start", groups: [][3]uint32{{'A', 'A' - 1, 1}},
			want: maxCmapEntries - 'A', wantWalked: maxCmapEntries,
		},
		// The bound is over the whole walk rather than per group, so a cmap packed with malformed groups costs no more
		// than one of them does.
		{
			name:       "many groups",
			groups:     [][3]uint32{{0, 0xFFFFFFFF, 1}, {0, 0xFFFFFFFF, 2}, {0, 0xFFFFFFFF, 3}},
			want:       maxCmapEntries,
			wantWalked: maxCmapEntries,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			data := sfntWithTables(t, roboto, map[string][]byte{"cmap": synthCmapFormat13(c.groups...)})
			runes, walked := faceRunesData(data, 0)
			if walked != c.wantWalked {
				t.Fatalf("the walk examined %d cmap entries, want %d", walked, c.wantWalked)
			}
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
