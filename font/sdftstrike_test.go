// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for the SDF strike lane (E.1): MakeSDFTMaskSpec produces MaskSDF glyphs whose bounds are the A8 bounds
// outset by SK_DistanceFieldPad, the kSDFT digest action accepts exactly the SDF-format in-atlas glyphs, and the
// generated image is a distance field consistent with the A8 coverage (inside > 128 where coverage is high, far pad
// below threshold).

package font

import (
	"os"
	"testing"
)

func loadSDFTestFont(t *testing.T, size float32) *Font {
	t.Helper()
	data, err := os.ReadFile("testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	return NewFont(tf, size, 1, 0)
}

// sdfGlyphID returns a glyph ID with a real outline ('A').
func sdfGlyphID(t *testing.T, f *Font) uint16 {
	t.Helper()
	var glyphs [1]uint16
	if f.TextToGlyphs([]byte("A"), TextEncodingUTF8, glyphs[:]) != 1 || glyphs[0] == 0 {
		t.Fatal("no glyph for 'A'")
	}
	return glyphs[0]
}

func TestSDFTMaskSpecGlyph(t *testing.T) {
	// The SDF strike font is what getSDFFont produces: kAntiAlias edging, no subpixel; use the DF mask size directly
	// (162, the large bucket).
	f := loadSDFTestFont(t, 162)
	f.SetEdging(EdgingAntiAlias)
	f.SetSubpixel(false)
	gid := sdfGlyphID(t, f)

	sdfSpec := MakeSDFTMaskSpec(f, nil)
	if sdfSpec.Rec.Format != MaskSDF {
		t.Fatalf("SDF spec format = %v, want MaskSDF", sdfSpec.Rec.Format)
	}
	a8Spec := MakeWithNoDeviceSpec(f, nil)
	if a8Spec.Rec.Format != MaskA8 {
		t.Fatalf("A8 spec format = %v, want MaskA8", a8Spec.Rec.Format)
	}
	// The SDF rec is a distinct strike key: same typeface/size but a different format.
	if sdfSpec.Rec == a8Spec.Rec {
		t.Fatal("SDF and A8 recs must differ (distinct strikes)")
	}

	sdfStrike := sdfSpec.FindOrCreateStrike()
	a8Strike := a8Spec.FindOrCreateStrike()

	sdfGlyph, action := sdfStrike.DigestFor(ActionSDFT, PackGlyphID(gid))
	if action != GlyphActionAccept {
		t.Fatalf("kSDFT action = %v, want accept", action)
	}
	a8Glyph, action := a8Strike.DigestFor(ActionDirectMask, PackGlyphID(gid))
	if action != GlyphActionAccept {
		t.Fatalf("A8 direct action = %v, want accept", action)
	}

	// Bounds: the A8 bounds outset by DistanceFieldPad on each side.
	if sdfGlyph.Format != MaskSDF {
		t.Errorf("glyph format = %v, want MaskSDF", sdfGlyph.Format)
	}
	if sdfGlyph.Left != a8Glyph.Left-DistanceFieldPad ||
		sdfGlyph.Top != a8Glyph.Top-DistanceFieldPad ||
		sdfGlyph.Width != a8Glyph.Width+2*DistanceFieldPad ||
		sdfGlyph.Height != a8Glyph.Height+2*DistanceFieldPad {
		t.Errorf("SDF bounds L%d T%d W%d H%d vs A8 L%d T%d W%d H%d: want pad %d",
			sdfGlyph.Left, sdfGlyph.Top, sdfGlyph.Width, sdfGlyph.Height,
			a8Glyph.Left, a8Glyph.Top, a8Glyph.Width, a8Glyph.Height, DistanceFieldPad)
	}

	// The image is a distance field: the pad corner is far outside (0) and the strongest-coverage texel of the A8 mask
	// reads inside (> 128) at the corresponding padded position.
	sdfGlyph = sdfStrike.PrepareImage(PackGlyphID(gid))
	if !sdfGlyph.HasImage() {
		t.Fatal("SDF glyph has no image")
	}
	if got := sdfGlyph.Image[0]; got != 0 {
		t.Errorf("pad corner = %d, want 0", got)
	}
	a8Glyph = a8Strike.PrepareImage(PackGlyphID(gid))
	bestX, bestY, bestV := 0, 0, uint8(0)
	for y := 0; y < int(a8Glyph.Height); y++ {
		for x := 0; x < int(a8Glyph.Width); x++ {
			if v := a8Glyph.Image[y*int(a8Glyph.Width)+x]; v > bestV {
				bestV = v
				bestX, bestY = x, y
			}
		}
	}
	if bestV < 200 {
		t.Fatalf("A8 mask has no strong texel (max %d)", bestV)
	}
	sdfW := int(sdfGlyph.Width)
	inside := sdfGlyph.Image[(bestY+DistanceFieldPad)*sdfW+bestX+DistanceFieldPad]
	if inside <= 128 {
		t.Errorf("SDF at strongest A8 texel = %d, want > 128 (inside)", inside)
	}

	// The digest gate rejects at atlas size: the padded max dimension governs.
	if sdfGlyph.MaxDimension() > SideTooBigForAtlas {
		t.Fatalf("test glyph unexpectedly larger than the atlas gate")
	}

	// An empty glyph (space) drops.
	var spaceGlyphs [1]uint16
	if f.TextToGlyphs([]byte(" "), TextEncodingUTF8, spaceGlyphs[:]) == 1 && spaceGlyphs[0] != 0 {
		_, action = sdfStrike.DigestFor(ActionSDFT, PackGlyphID(spaceGlyphs[0]))
		if action != GlyphActionDrop {
			t.Errorf("space kSDFT action = %v, want drop", action)
		}
	}
}

func TestSDFTActionRejectsNonSDFFormats(t *testing.T) {
	// A plain A8 strike's glyphs never accept the kSDFT action (maskFormat must be kSDF).
	f := loadSDFTestFont(t, 100)
	gid := sdfGlyphID(t, f)
	spec := MakeWithNoDeviceSpec(f, nil)
	strike := spec.FindOrCreateStrike()
	if _, action := strike.DigestFor(ActionSDFT, PackGlyphID(gid)); action != GlyphActionReject {
		t.Errorf("A8 glyph kSDFT action = %v, want reject", action)
	}
}
