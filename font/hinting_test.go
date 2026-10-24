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
	"slices"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// hintedTestGlyph returns the mask of 'H' from the test font at 16 pixels under the given hinting level and matrix.
func hintedTestGlyph(t *testing.T, cache *StrikeCache, level Hinting, matrix *geom.Matrix, skewX float32) *Glyph {
	t.Helper()
	f := loadMigratedTestFont(t, 16)
	f.SetHinting(level)
	f.SetSkewX(skewX)
	gid := f.Typeface().UnicharToGlyph('H')
	if gid == 0 {
		t.Fatal("test font does not map 'H'")
	}
	rec, effects := MakeRecAndEffects(f, nil, matrix, nil)
	spec := StrikeSpec{Typeface: f.Typeface(), Rec: rec, Effects: effects}
	g := cache.FindOrCreateStrike(&spec).PrepareImage(PackGlyphID(gid))
	if g == nil || g.IsEmpty() || g.Image == nil {
		t.Fatal("no glyph mask produced")
	}
	return g
}

// rowInk returns the summed coverage of each row of an A8 mask.
func rowInk(g *Glyph) []int {
	rows := make([]int, g.Height)
	rb := int(g.RowBytes())
	for y := range rows {
		for _, v := range g.Image[y*rb : y*rb+int(g.Width)] {
			rows[y] += int(v)
		}
	}
	return rows
}

// TestFullHinting checks that hinting puts the cap height of an 'H' on a row boundary without changing its advance, and
// that a rotated or slanted strike is left unhinted.
func TestFullHinting(t *testing.T) {
	cache := NewStrikeCache()
	identity := geom.IdentityMatrix()
	plain := hintedTestGlyph(t, cache, HintingNormal, &identity, 0)
	hinted := hintedTestGlyph(t, cache, HintingFull, &identity, 0)
	if plain.AdvanceX != hinted.AdvanceX {
		t.Errorf("hinting changed the advance from %v to %v", plain.AdvanceX, hinted.AdvanceX)
	}
	plainRows, hintedRows := rowInk(plain), rowInk(hinted)
	if len(hintedRows) < 4 || len(plainRows) < 4 {
		t.Fatalf("masks too small to compare: %d and %d rows", len(plainRows), len(hintedRows))
	}
	// Row 2 runs through the two stems in both masks.
	if top, stems := hintedRows[0], hintedRows[2]; top*10 < stems*9 {
		t.Errorf("hinted top row holds %d against %d in the stems: the cap height is not on a row boundary", top, stems)
	}
	if top, stems := plainRows[0], plainRows[2]; top*10 >= stems*9 {
		t.Errorf("unhinted top row holds %d against %d in the stems: the test glyph no longer shows the difference",
			top, stems)
	}

	same := func(a, b *Glyph) bool {
		return a.Width == b.Width && a.Height == b.Height && a.Left == b.Left && a.Top == b.Top &&
			slices.Equal(a.Image, b.Image)
	}
	rotated := geom.RotateDegMatrix(30)
	if !same(hintedTestGlyph(t, cache, HintingNormal, &rotated, 0), hintedTestGlyph(t, cache, HintingFull, &rotated, 0)) {
		t.Error("a rotated strike was hinted")
	}
	slantedPlain := hintedTestGlyph(t, cache, HintingNormal, &identity, -0.2)
	if !same(slantedPlain, hintedTestGlyph(t, cache, HintingFull, &identity, -0.2)) {
		t.Error("a slanted strike was hinted")
	}
}

// TestHintedContourReconstruction checks that every hinted glyph's path stays within its hinted control box, which a
// walk that dropped or invented a segment would break.
func TestHintedContourReconstruction(t *testing.T) {
	f := loadMigratedTestFont(t, 16)
	f.SetHinting(HintingFull)
	identity := geom.IdentityMatrix()
	rec, effects := MakeRecAndEffects(f, nil, &identity, nil)
	ctx := NewScalerContext(f.Typeface(), rec, effects)
	if ctx.hinter == nil {
		t.Fatal("the test font was not prepared for hinting")
	}
	checked := 0
	for gid := range uint16(f.Typeface().nGlyphs) {
		bounds, ok := ctx.hintedBounds(gid)
		if !ok {
			continue
		}
		p := ctx.hintedPath(gid)
		if p == nil {
			t.Fatalf("glyph %d has hinted bounds but no hinted path", gid)
		}
		got := p.Bounds()
		const slack = 1.0 / 64
		if got.Left < bounds.Left-slack || got.Top < bounds.Top-slack || got.Right > bounds.Right+slack ||
			got.Bottom > bounds.Bottom+slack {
			t.Errorf("glyph %d: path bounds %v outside the outline's %v", gid, got, bounds)
		}
		checked++
	}
	if checked < 100 {
		t.Errorf("only %d glyphs were hinted", checked)
	}
}
