// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package font

import "testing"

func TestCountUTF8(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{in: "", want: 0},
		{in: "abc", want: 3},
		{in: "é€\U0001F600", want: 3},
		{in: "aé", want: 2},
	}
	for _, c := range cases {
		if got := countUTF8([]byte(c.in)); got != c.want {
			t.Errorf("countUTF8(%q) = %d, want %d", c.in, got, c.want)
		}
	}
	bad := [][]byte{
		{0xC0, 0x80},       // overlong lead byte C0 never appears
		{0xF5, 0x80},       // > U+10FFFF lead
		{0x80},             // bare continuation
		{0xE0, 0x80},       // truncated 3-byte sequence
		{0x61, 0xC3},       // truncated at end
		{0xC3, 0x28},       // non-continuation after lead
		{0xFF},             // invalid byte
		{0x61, 0xE2, 0x82}, // truncated at end
	}
	for _, b := range bad {
		if got := countUTF8(b); got != -1 {
			t.Errorf("countUTF8(% x) = %d, want -1", b, got)
		}
	}
}

func TestUTF8ValidationIsStructuralOnly(t *testing.T) {
	// The file comment promises structural validation, not canonical-form validation: an overlong encoding, a
	// UTF-8-encoded surrogate, and a code point above U+10FFFF all pass the count functions and decode to the values
	// below (the mask-based decode folds an overlong form down to its short value). This matches upstream
	// SkUTF::CountUTF8/NextUTF8; the point of the test is that the comment and the code agree.
	cases := []struct {
		name string
		in   []byte
		want int32
	}{
		{name: "overlong 3-byte 'o'", in: []byte{0xE0, 0x81, 0xAF}, want: 0x6F},
		{name: "overlong 4-byte '/'", in: []byte{0xF0, 0x80, 0x80, 0xAF}, want: 0x2F},
		{name: "encoded high surrogate", in: []byte{0xED, 0xA0, 0x80}, want: 0xD800},
		{name: "above U+10FFFF", in: []byte{0xF4, 0x90, 0x80, 0x80}, want: 0x110000},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := countUTF8(c.in); got != 1 {
				t.Errorf("countUTF8(% x) = %d, want 1 (structural validation accepts it)", c.in, got)
			}
			if got := CountTextElements(c.in, TextEncodingUTF8); got != 1 {
				t.Errorf("CountTextElements(% x) = %d, want 1", c.in, got)
			}
			u, next := nextUTF8(c.in, 0)
			if u != c.want || next != len(c.in) {
				t.Errorf("nextUTF8(% x) = %#x,%d, want %#x,%d", c.in, u, next, c.want, len(c.in))
			}
		})
	}

	// The documented consequence: an overlong '/' maps to the same glyph as a real '/', because TextToGlyphs decodes
	// with nextUTF8 and hands the folded code point to the cmap.
	tf := loadColorTypeface(t, "Roboto-Regular.ttf")
	glyphs := make([]uint16, 1)
	if n := tf.TextToGlyphs([]byte{0xF0, 0x80, 0x80, 0xAF}, TextEncodingUTF8, glyphs); n != 1 {
		t.Fatalf("TextToGlyphs(overlong) = %d, want 1", n)
	}
	want := tf.UnicharToGlyph('/')
	if want == 0 {
		t.Fatal("'/' not mapped")
	}
	if glyphs[0] != want {
		t.Errorf("overlong '/' mapped to glyph %d, want the real '/' glyph %d", glyphs[0], want)
	}
}

func TestCountUTF16(t *testing.T) {
	if got := countUTF16([]byte{0x48, 0x00, 0x49, 0x00}); got != 2 {
		t.Errorf("BMP pair = %d, want 2", got)
	}
	// Surrogate pair (U+1F600 = D83D DE00 little-endian).
	if got := countUTF16([]byte{0x3D, 0xD8, 0x00, 0xDE}); got != 1 {
		t.Errorf("surrogate pair = %d, want 1", got)
	}
	bad := [][]byte{
		{0x48},                   // odd length
		{0x00, 0xDC},             // lone low surrogate
		{0x3D, 0xD8},             // truncated high surrogate
		{0x3D, 0xD8, 0x48, 0x00}, // high surrogate followed by non-low
	}
	for _, b := range bad {
		if got := countUTF16(b); got != -1 {
			t.Errorf("countUTF16(% x) = %d, want -1", b, got)
		}
	}
}

func TestNextUTF8(t *testing.T) {
	b := []byte("aé\U0001F600")
	u, i := nextUTF8(b, 0)
	if u != 'a' || i != 1 {
		t.Fatalf("first = %d,%d", u, i)
	}
	u, i = nextUTF8(b, i)
	if u != 0xE9 || i != 3 {
		t.Fatalf("second = %d,%d", u, i)
	}
	u, i = nextUTF8(b, i)
	if u != 0x1F600 || i != len(b) {
		t.Fatalf("third = %d,%d", u, i)
	}
	// Invalid input jumps to the end with -1, like next_fail.
	u, i = nextUTF8([]byte{0x61, 0xC0}, 1)
	if u != -1 || i != 2 {
		t.Fatalf("invalid = %d,%d", u, i)
	}
}

func TestNextUTF16(t *testing.T) {
	b := []byte{0x48, 0x00, 0x3D, 0xD8, 0x00, 0xDE}
	u, i := nextUTF16(b, 0)
	if u != 'H' || i != 2 {
		t.Fatalf("first = %d,%d", u, i)
	}
	u, i = nextUTF16(b, i)
	if u != 0x1F600 || i != 6 {
		t.Fatalf("second = %d,%d", u, i)
	}
	if u, _ = nextUTF16([]byte{0x00, 0xDC}, 0); u != -1 {
		t.Fatalf("lone low surrogate = %d", u)
	}
}

func TestCountTextElements(t *testing.T) {
	if got := CountTextElements([]byte("abc"), TextEncodingUTF8); got != 3 {
		t.Errorf("utf8 = %d", got)
	}
	if got := CountTextElements([]byte{1, 2, 3, 4, 5, 6, 7, 8}, TextEncodingUTF32); got != 2 {
		t.Errorf("utf32 = %d", got)
	}
	if got := CountTextElements([]byte{1, 2, 3, 4}, TextEncodingGlyphID); got != 2 {
		t.Errorf("glyphs = %d", got)
	}
	if got := CountTextElements([]byte{0xFF}, TextEncodingUTF8); got != -1 {
		t.Errorf("invalid utf8 = %d", got)
	}
}
