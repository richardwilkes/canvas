// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the outline lane's glyph-data accessor: the lanes that set neverRequestPath cover COLR and PNG strikes
// only, so every other color format (SVG documents, sbix 'jpg '/'tif ' graphics, B&W strikes) reaches the outline lane
// and must render from the fallback outline the spec requires such glyphs to carry.

package font

import (
	"bytes"
	"encoding/binary"
	"os"
	"slices"
	"strconv"
	"testing"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"

	"github.com/richardwilkes/canvas/geom"
)

// svgTableFor builds an 'SVG ' table (version 0) whose single document-list record covers [first, last].
func svgTableFor(first, last uint16, doc string) []byte {
	const headerSize = 10      // version, svgDocumentListOffset, reserved
	list := make([]byte, 2+12) // numEntries plus one SVGDocumentRecord
	binary.BigEndian.PutUint16(list[0:], 1)
	binary.BigEndian.PutUint16(list[2:], first)
	binary.BigEndian.PutUint16(list[4:], last)
	binary.BigEndian.PutUint32(list[6:], uint32(len(list))) // the document follows the list
	binary.BigEndian.PutUint32(list[10:], uint32(len(doc)))
	out := make([]byte, headerSize)
	binary.BigEndian.PutUint32(out[2:], headerSize)
	out = append(out, list...)
	return append(out, doc...)
}

// glyphThroughMaskStrike resolves gid through a plain mask strike at size, preparing the lane action asks for.
func glyphThroughMaskStrike(t *testing.T, tf *Typeface, gid uint16, size float32,
	action ActionType,
) (*Glyph, GlyphAction) {
	t.Helper()
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(NewFont(tf, size, 1, 0), nil, &identity, nil)
	return spec.FindOrCreateStrike().DigestFor(action, PackGlyphID(gid))
}

func TestSVGGlyphRendersFromItsFallbackOutline(t *testing.T) {
	// An 'SVG ' entry has no lane of its own here, so the glyph takes the outline lane: non-empty bounds from the
	// outline extents, a mask filled from the outline, and a resolvable path. go-text's GlyphData prefers the SVG
	// document over the outline, so resolving through it leaves the allocated mask empty and Path() nil — the glyph
	// renders as nothing at all.
	data, err := os.ReadFile("testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	plain, err := NewTypefaceFromData(bytes.Clone(data), 0)
	if err != nil {
		t.Fatal(err)
	}
	gid := plain.UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("Roboto should map 'A'")
	}
	doc := `<?xml version="1.0" encoding="UTF-8"?>` +
		`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">` +
		`<g id="glyph` + strconv.Itoa(int(gid)) + `"><rect width="500" height="500" fill="red"/></g></svg>`
	withSVG, err := NewTypefaceFromData(
		sfntWithTables(t, data, map[string][]byte{"SVG ": svgTableFor(gid, gid, doc)}), 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, isOutline := withSVG.face.GlyphData(opentype.GID(gid)).(tsfont.GlyphOutline); isOutline {
		t.Fatal("the added SVG table did not take precedence in GlyphData; the test no longer covers the hazard")
	}

	want, wantAction := glyphThroughMaskStrike(t, plain, gid, 64, ActionDirectMaskCPU)
	got, gotAction := glyphThroughMaskStrike(t, withSVG, gid, 64, ActionDirectMaskCPU)
	if wantAction != GlyphActionAccept || gotAction != GlyphActionAccept {
		t.Fatalf("mask actions %v/%v, want accept for both", wantAction, gotAction)
	}
	if got.IRect() != want.IRect() {
		t.Errorf("bounds %v, want the outline lane's %v", got.IRect(), want.IRect())
	}
	if maskInk(got) == 0 {
		t.Error("the SVG glyph rendered a blank mask")
	}
	if !slices.Equal(got.Image, want.Image) {
		t.Error("the SVG glyph's mask differs from the same outline without the SVG table")
	}

	// The path lane must resolve too: ActionPath rejects a glyph whose path comes back nil.
	pg, pathAction := glyphThroughMaskStrike(t, withSVG, gid, 64, ActionPath)
	if pathAction != GlyphActionAccept || pg.Path() == nil {
		t.Errorf("path action %v, path nil %v; want accept with a path", pathAction, pg.Path() == nil)
	}
}
