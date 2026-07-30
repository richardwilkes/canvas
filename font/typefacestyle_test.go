// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// computeStyle is the one place a face's weight/width/slant is derived from, and three consumers have to agree with it:
// Typeface.Style (what the PDF FontDescriptor's italic flag comes from), FaceInfo.Style (what the font manager ranks
// candidates by), and the macStyle fallback for a face with no usable OS/2 table. Every font in testdata is upright with
// a modern OS/2 table, so these cases are synthesized.

package font

import (
	"encoding/binary"
	"testing"
)

// OS/2 fsSelection bits the slant derivation reads.
const (
	fsSelectionItalic  = 0x001
	fsSelectionOblique = 0x200
)

// TestObliqueBitIsVersionIndependent pins fsSelection bit 9. The bit was first documented in OS/2 version 4, but
// FreeType (sfobjs.c, which folds it into its italic style flag) and Skia (SkFontHost_FreeType.cpp, which maps it to
// kOblique_Slant) both read it at every version, and a face that declares itself oblique means it whichever version
// number it also declares. Read as upright instead, the font manager ranks an oblique face as an upright candidate and
// the PDF FontDescriptor loses its italic flag.
func TestObliqueBitIsVersionIndependent(t *testing.T) {
	base := readTestFont(t, "Roboto-Regular.ttf")
	oblique := func(version uint16) []byte {
		return sfntWithTables(t, base, map[string][]byte{
			"OS/2": os2Patched(t, base, func(os2 []byte) {
				binary.BigEndian.PutUint16(os2[os2VersionOffset:], version)
				// Bit 0 stays clear so nothing but bit 9 can be what makes the face non-upright.
				fsSelection := binary.BigEndian.Uint16(os2[os2FsSelectionOffset:])
				binary.BigEndian.PutUint16(os2[os2FsSelectionOffset:],
					fsSelection&^fsSelectionItalic|fsSelectionOblique)
			}),
		})
	}
	for version := uint16(0); version <= 5; version++ {
		data := oblique(version)
		tf, err := NewTypefaceFromData(data, 0)
		if err != nil {
			t.Fatalf("version %d: %v", version, err)
		}
		if got := tf.Style().Slant(); got != SlantOblique {
			t.Errorf("version %d: Typeface slant = %v, want %v", version, got, SlantOblique)
		}
		if m := tf.GetAdvancedMetrics(); m.Style&StyleItalic == 0 {
			t.Errorf("version %d: FontDescriptor style bits %#x omit StyleItalic for an oblique face",
				version, m.Style)
		}
		// The font manager ranks by FaceInfo, so its view of the same bytes has to be the same view.
		info, err := DescribeFaceData(data, 0)
		if err != nil {
			t.Fatalf("version %d: DescribeFaceData: %v", version, err)
		}
		if info.Style != tf.Style() {
			t.Errorf("version %d: FaceInfo style %v disagrees with the typeface's %v", version, info.Style, tf.Style())
		}
	}

	// The 0xFFFF sentinel is not a version number: it says the face has no usable OS/2 table, so nothing in it — bit 9
	// included — is read, and the slant comes from head.macStyle instead.
	sentinel, err := NewTypefaceFromData(oblique(os2NoTableVersion), 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := sentinel.Style().Slant(); got != SlantUpright {
		t.Errorf("the sentinel version's slant = %v, want %v from head.macStyle", got, SlantUpright)
	}
}

// TestItalicAndObliqueBitsTogether pins the precedence between the two slant bits: a face setting both is oblique, as
// Skia has it (the oblique test runs after the italic one and overwrites its answer).
func TestItalicAndObliqueBitsTogether(t *testing.T) {
	base := readTestFont(t, "Roboto-Regular.ttf")
	for _, c := range []struct {
		desc string
		bits uint16
		want Slant
	}{
		{desc: "neither bit", want: SlantUpright},
		{desc: "italic only", bits: fsSelectionItalic, want: SlantItalic},
		{desc: "oblique only", bits: fsSelectionOblique, want: SlantOblique},
		{desc: "both", bits: fsSelectionItalic | fsSelectionOblique, want: SlantOblique},
	} {
		t.Run(c.desc, func(t *testing.T) {
			data := sfntWithTables(t, base, map[string][]byte{
				"OS/2": os2Patched(t, base, func(os2 []byte) {
					fsSelection := binary.BigEndian.Uint16(os2[os2FsSelectionOffset:])
					fsSelection &^= fsSelectionItalic | fsSelectionOblique
					binary.BigEndian.PutUint16(os2[os2FsSelectionOffset:], fsSelection|c.bits)
				}),
			})
			tf, err := NewTypefaceFromData(data, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got := tf.Style().Slant(); got != c.want {
				t.Errorf("slant = %v, want %v", got, c.want)
			}
		})
	}
}
