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
