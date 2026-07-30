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
	for _, h := range []Hinting{HintingNone, HintingSlight, HintingFull, HintingNormal} {
		f.SetHinting(h)
		if got := f.Hinting(); got != h {
			t.Errorf("SetHinting(%v) recorded %v", h, got)
		}
		// The recorded level reaches the strike key, which is the only place it is consumed: a level that never left
		// the Font would collapse every hinting request onto one strike.
		if rec, _ := MakeRecAndEffects(f, nil, &identity, nil); rec.Hinting != h {
			t.Errorf("SetHinting(%v) put %v in the scaler rec", h, rec.Hinting)
		}
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
