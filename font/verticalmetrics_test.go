// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The one ascent/descent/line-gap recipe (FreeType's sfnt_load_face) both the strike's font metrics and the PDF
// FontDescriptor read: a font laid out with one source and described with another puts an /Ascent and /Descent on the
// page that contradict the glyphs drawn under them.

package font

import (
	"encoding/binary"
	"testing"

	"github.com/go-text/typesetting/font/opentype/tables"
)

// Byte offsets within the hhea table of the three fields verticalMetrics reads.
const (
	hheaAscenderOffset  = 4
	hheaDescenderOffset = 6
	hheaLineGapOffset   = 8
)

// hheaWith returns a copy of data's hhea table carrying the given ascender, descender and line gap.
func hheaWith(t *testing.T, data []byte, ascender, descender, lineGap int16) []byte {
	t.Helper()
	out := append([]byte(nil), rawSfntTable(t, data, "hhea")...)
	if len(out) < hheaLineGapOffset+2 {
		t.Fatalf("hhea table is %d bytes, too short to hold the vertical metrics", len(out))
	}
	binary.BigEndian.PutUint16(out[hheaAscenderOffset:], uint16(ascender))
	binary.BigEndian.PutUint16(out[hheaDescenderOffset:], uint16(descender))
	binary.BigEndian.PutUint16(out[hheaLineGapOffset:], uint16(lineGap))
	return out
}

func TestVerticalMetricsRecipe(t *testing.T) {
	// Roboto's real values, which the end-to-end cases below run against: hhea says 1900/-500/0, the OS/2 typographic
	// fields say 1536/-512/102, and usWinAscent/usWinDescent say 1946/512.
	hhea := func(ascender, descender, lineGap int16) *tables.Hhea {
		return &tables.Hhea{Ascender: ascender, Descender: descender, LineGap: lineGap}
	}
	os2 := func(version, fsSelection uint16, typoAscender, typoDescender, typoLineGap int16) *tables.Os2 {
		return &tables.Os2{
			Version: version, FsSelection: fsSelection, STypoAscender: typoAscender,
			STypoDescender: typoDescender, STypoLineGap: typoLineGap,
		}
	}
	const useTypo = os2UseTypoMetricsBit
	for _, c := range []struct {
		hhea                     *tables.Hhea
		os2                      *tables.Os2
		name                     string
		winAscent, winDescent    uint16
		ascent, descent, lineGap int16
	}{
		{
			name: "hhea wins by default", hhea: hhea(1900, -500, 0), os2: os2(3, 0x40, 1536, -512, 102),
			winAscent: 1946, winDescent: 512, ascent: 1900, descent: -500, lineGap: 0,
		},
		{
			name: "USE_TYPO_METRICS wins over hhea", hhea: hhea(1900, -500, 0), os2: os2(3, 0x40|useTypo, 1536, -512, 102),
			winAscent: 1946, winDescent: 512, ascent: 1536, descent: -512, lineGap: 102,
		},
		{
			// 0xFFFF is FreeType's "no usable OS/2" sentinel, so nothing in that table may be consulted — not the bit,
			// and not the values it would have selected.
			name: "USE_TYPO_METRICS on the sentinel version is ignored", hhea: hhea(1900, -500, 0),
			os2: os2(0xFFFF, 0x40|useTypo, 1536, -512, 102), ascent: 1900, descent: -500, lineGap: 0,
		},
		{
			name: "a zero hhea falls back to the typographic values", hhea: hhea(0, 0, 0),
			os2: os2(3, 0x40, 1536, -512, 102), winAscent: 1946, winDescent: 512,
			ascent: 1536, descent: -512, lineGap: 102,
		},
		{
			name: "a zero hhea and no typographic values fall back to usWin", hhea: hhea(0, 0, 0),
			os2: os2(3, 0x40, 0, 0, 0), winAscent: 1946, winDescent: 512, ascent: 1946, descent: -512, lineGap: 0,
		},
		{
			// Only *both* being zero is the buggy-table signal; one alone is a face that really has no descent.
			name: "a zero hhea ascender alone still wins", hhea: hhea(0, -500, 7), os2: os2(3, 0x40, 1536, -512, 102),
			winAscent: 1946, winDescent: 512, ascent: 0, descent: -500, lineGap: 7,
		},
		{
			name: "no hhea at all falls back the same way", os2: os2(3, 0x40, 1536, -512, 102),
			winAscent: 1946, winDescent: 512, ascent: 1536, descent: -512, lineGap: 102,
		},
		{
			name: "no hhea and a sentinel OS/2 has no source at all", os2: os2(0xFFFF, 0, 1536, -512, 102),
			winAscent: 1946, winDescent: 512,
		},
		{name: "no hhea and no OS/2 at all"},
	} {
		t.Run(c.name, func(t *testing.T) {
			tf := &Typeface{hhea: c.hhea, os2: c.os2, usWinAscent: c.winAscent, usWinDescent: c.winDescent}
			ascent, descent, lineGap := tf.verticalMetrics()
			if ascent != c.ascent || descent != c.descent || lineGap != c.lineGap {
				t.Errorf("verticalMetrics = %d/%d/%d, want %d/%d/%d", ascent, descent, lineGap,
					c.ascent, c.descent, c.lineGap)
			}
		})
	}
}

// The two consumers must never disagree: whatever the recipe answers is both what a line of text is laid out with and
// what the PDF FontDescriptor declares.
func TestVerticalMetricsConsumersAgree(t *testing.T) {
	roboto := readTestFont(t, "Roboto-Regular.ttf")
	const size = 50
	const upem = 2048
	for _, c := range []struct {
		build           func(t *testing.T) []byte
		name            string
		ascent, descent int16
		lineGap         int16
	}{
		{
			name:  "as shipped, from hhea",
			build: func(_ *testing.T) []byte { return roboto },
			// hhea 1900/-500/0.
			ascent: 1900, descent: -500,
		},
		{
			name: "USE_TYPO_METRICS set",
			build: func(t *testing.T) []byte {
				return sfntWithTables(t, roboto, map[string][]byte{
					"OS/2": os2WithFsSelectionBits(t, roboto, os2UseTypoMetricsBit, true),
				})
			},
			// The OS/2 typographic values, 1536/-512/102.
			ascent: 1536, descent: -512, lineGap: 102,
		},
		{
			// A legal, real-world table state: the font ships an hhea that reports nothing, and FreeType (and so every
			// host that goes through it) reads the OS/2 typographic values instead. Reading hhea straight through here
			// lays every line of the run out on one baseline.
			name: "a zero hhea",
			build: func(t *testing.T) []byte {
				return sfntWithTables(t, roboto, map[string][]byte{"hhea": hheaWith(t, roboto, 0, 0, 0)})
			},
			ascent: 1536, descent: -512, lineGap: 102,
		},
		{
			name: "no hhea table at all",
			build: func(t *testing.T) []byte {
				return sfntWithoutTables(t, roboto, "hhea")
			},
			ascent: 1536, descent: -512, lineGap: 102,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			tf, err := NewTypefaceFromData(c.build(t), 0)
			if err != nil {
				t.Fatal(err)
			}
			m := tf.GetAdvancedMetrics()
			if m == nil {
				t.Fatal("no advanced metrics")
			}
			if m.Ascent != c.ascent || m.Descent != c.descent {
				t.Errorf("AdvancedMetrics ascent/descent = %d/%d, want %d/%d", m.Ascent, m.Descent, c.ascent,
					c.descent)
			}
			if tf.hhea == nil {
				// The strike's metrics decline to run at all with no hhea (they have no head bbox to report either), so
				// only the PDF side is checked for that case.
				return
			}
			var metrics Metrics
			spacing := NewFont(tf, size, 1, 0).Metrics(&metrics)
			// Device space is y-down and scaled by size/upem, so the font-unit values negate and scale into it.
			wantAscent := -float32(c.ascent) * size / upem
			wantDescent := -float32(c.descent) * size / upem
			wantLeading := float32(c.lineGap) * size / upem
			if metrics.Ascent != wantAscent || metrics.Descent != wantDescent || metrics.Leading != wantLeading {
				t.Errorf("Metrics ascent/descent/leading = %v/%v/%v, want %v/%v/%v", metrics.Ascent, metrics.Descent,
					metrics.Leading, wantAscent, wantDescent, wantLeading)
			}
			if want := wantDescent - wantAscent + wantLeading; spacing != want {
				t.Errorf("recommended line spacing = %v, want %v", spacing, want)
			}
			if spacing <= 0 {
				t.Errorf("line spacing %v stacks every line on the same baseline", spacing)
			}
		})
	}
}
