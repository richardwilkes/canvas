// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Round-trip and accessor checks for the public font entry points.

package font

import (
	"os"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

func loadMigratedTestFont(t *testing.T, size float32) *Font {
	t.Helper()
	data, err := os.ReadFile("testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatalf("read test font: %v", err)
	}
	tf, err := NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatalf("parse test font: %v", err)
	}
	return NewFont(tf, size, 1, 0)
}

// TestFontFlagAndHintingRoundTrip covers the Font setter/getter pairs: force auto-hinting, subpixel, and the hinting
// level all round-trip.
func TestFontFlagAndHintingRoundTrip(t *testing.T) {
	f := loadMigratedTestFont(t, 20)
	if f.ForceAutoHinting() {
		t.Error("force auto-hinting should default off")
	}
	f.SetForceAutoHinting(true)
	if !f.ForceAutoHinting() {
		t.Error("SetForceAutoHinting(true) did not stick")
	}
	f.SetSubpixel(true)
	if !f.Subpixel() {
		t.Error("SetSubpixel(true) did not stick")
	}
	// Round-tripping the constructor default (HintingNormal) proves nothing about SetHinting, so pin the default first
	// and then walk every level away from it and back. A SetHinting that dropped its argument would hold at the default.
	if got := f.Hinting(); got != HintingNormal {
		t.Errorf("Hinting defaults to %v, want %v", got, HintingNormal)
	}
	identity := geom.IdentityMatrix()
	base, _ := MakeRecAndEffects(f, nil, &identity, nil)
	for _, h := range []Hinting{HintingNone, HintingSlight, HintingFull, HintingNormal} {
		f.SetHinting(h)
		if got := f.Hinting(); got != h {
			t.Errorf("SetHinting(%v) recorded %v", h, got)
		}
		// The recorded level must not reach the strike key: no lane honors hinting, so a level in the rec would only
		// split byte-identical masks across two strikes and charge the cache budget for both.
		if rec, _ := MakeRecAndEffects(f, nil, &identity, nil); rec != base {
			t.Errorf("SetHinting(%v) changed the scaler rec, which no lane reads the level from", h)
		}
	}
}

// TestHintingDoesNotFragmentStrikes is the consequence TestFontFlagAndHintingRoundTrip's rec comparison exists to
// prevent: since nothing honors the recorded level, two fonts differing only in hinting must resolve to the same strike
// (and therefore the same cached glyph masks) rather than to two strikes holding identical bytes.
func TestHintingDoesNotFragmentStrikes(t *testing.T) {
	cache := NewStrikeCache()
	base := loadMigratedTestFont(t, 20)
	strikeFor := func(h Hinting) *Strike {
		f := *base // same typeface, so only the hinting level differs
		f.SetHinting(h)
		spec := MakeWithNoDeviceSpec(&f, nil)
		return cache.FindOrCreateStrike(&spec)
	}
	normal := strikeFor(HintingNormal)
	none := strikeFor(HintingNone)
	if normal != none {
		t.Fatal("HintingNormal and HintingNone produced different strikes for identical rendering")
	}
	gid := base.Typeface().UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("test font does not map 'A'")
	}
	used := cache.TotalMemoryUsed()
	if normal.PrepareImage(PackGlyphID(gid)) == nil {
		t.Fatal("no glyph produced")
	}
	grew := cache.TotalMemoryUsed() - used
	if grew <= 0 {
		t.Fatalf("mask generation charged %d bytes to the cache, want a positive amount", grew)
	}
	// The second font's mask comes out of the shared strike, so it costs nothing more. Keying on hinting would charge
	// the same bytes again.
	if none.PrepareImage(PackGlyphID(gid)) == nil {
		t.Fatal("no glyph produced for the second hinting level")
	}
	if again := cache.TotalMemoryUsed() - used; again != grew {
		t.Errorf("the second hinting level added %d more bytes to the cache, want %d", again-grew, 0)
	}
}

// TestMeasureTextWithSpace measures a run containing a space: the space glyph has empty ink extents, so the scaler's
// empty-bounds lanes must contribute nothing to the union while the advance still accumulates.
func TestMeasureTextWithSpace(t *testing.T) {
	f := loadMigratedTestFont(t, 23)
	var withSpace, noSpace geom.Rect
	wsAdv := f.MeasureText([]byte("Ag Wq"), TextEncodingUTF8, &withSpace, nil)
	nsAdv := f.MeasureText([]byte("AgWq"), TextEncodingUTF8, &noSpace, nil)
	if wsAdv <= nsAdv {
		t.Errorf("advance with a space (%v) should exceed the advance without (%v)", wsAdv, nsAdv)
	}
	if withSpace.IsEmpty() {
		t.Error("bounds of a visible run must not be empty")
	}
	// A space-only run has a positive advance but no ink.
	var spaceBounds geom.Rect
	spaceAdv := f.MeasureText([]byte(" "), TextEncodingUTF8, &spaceBounds, nil)
	if spaceAdv <= 0 {
		t.Errorf("space advance = %v, want > 0", spaceAdv)
	}
	if !spaceBounds.IsEmpty() {
		t.Errorf("space ink bounds = %v, want empty", spaceBounds)
	}

	// A stroked paint reaches the scaler's styled (outline) bounds lane; the space's empty outline must still
	// contribute nothing while the visible glyph bounds widen.
	spec := stroke.PaintSpec{Style: stroke.PaintStyleStroke, Width: 4}
	var stroked geom.Rect
	f.MeasureText([]byte("Ag Wq"), TextEncodingUTF8, &stroked, &spec)
	if !(stroked.Left < withSpace.Left && stroked.Right > withSpace.Right) {
		t.Errorf("stroked bounds %v should be wider than unstroked %v", stroked, withSpace)
	}
}
