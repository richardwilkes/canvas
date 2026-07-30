// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the guards shared by every glyph mask lane: the set of formats Glyph.Format actually carries,
// allocGlyphImage's empty/too-large gate, and saturateBounds' rounding invariant.

package font

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// TestGlyphFormatReachableSet pins what Glyph.Format documents: all four mask formats reach the field. A8 comes from
// the outline lane, LCD16 from subpixel-AA edging with known pixel geometry, SDF from an SDFT spec, and ARGB32 from the
// color lanes. Everything downstream of the field — bytesPerPixel, HasImage, RowBytes, Pixmap, Mask, DigestFor's
// ActionSDFT gate — branches on the full set, so a lane that stopped storing one would leave that doc stale.
func TestGlyphFormatReachableSet(t *testing.T) {
	identity := geom.IdentityMatrix()
	seen := make(map[MaskFormat]bool, 4)

	// A8: the plain outline lane at the identity device matrix, no pixel geometry.
	a8Font := loadSDFTestFont(t, 24)
	a8Font.SetEdging(EdgingAntiAlias)
	gid := a8Font.Typeface().UnicharToGlyph('H')
	if gid == 0 {
		t.Fatal("'H' not mapped")
	}
	a8Spec := MakeMaskSpec(a8Font, nil, &identity, nil)
	g, action := a8Spec.FindOrCreateStrike().DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if action != GlyphActionAccept {
		t.Fatalf("A8 action = %v, want accept", action)
	}
	seen[g.Format] = true

	// LCD16: subpixel-AA edging plus a horizontal-RGB device.
	lcdSpec := MakeMaskSpec(lcdTestFont(t, 24), &ScalerPaint{LumColor: colorcore.RGB(0, 0, 0)}, &identity,
		&DeviceProps{PixelGeometry: PixelGeometryRGBH})
	if g, action = lcdSpec.FindOrCreateStrike().DigestFor(ActionDirectMaskCPU, PackGlyphID(gid)); action !=
		GlyphActionAccept {
		t.Fatalf("LCD16 action = %v, want accept", action)
	}
	seen[g.Format] = true

	// SDF: the distance-field strike spec, whose rec format the glyph inherits.
	sdfFont := loadSDFTestFont(t, 162)
	sdfFont.SetEdging(EdgingAntiAlias)
	sdfFont.SetSubpixel(false)
	sdfSpec := MakeSDFTMaskSpec(sdfFont, nil)
	if g, action = sdfSpec.FindOrCreateStrike().DigestFor(ActionSDFT, PackGlyphID(sdfGlyphID(t,
		sdfFont))); action != GlyphActionAccept {
		t.Fatalf("SDF action = %v, want accept", action)
	}
	seen[g.Format] = true

	// ARGB32: a color lane overriding the rec's A8 format per glyph.
	if g, action = smileyGlyph(t, loadColorTypeface(t, "cbdt.ttf"), 64, nil); action != GlyphActionAccept {
		t.Fatalf("ARGB32 action = %v, want accept", action)
	}
	seen[g.Format] = true

	for _, want := range []struct {
		name   string
		format MaskFormat
	}{
		{name: "MaskA8", format: MaskA8},
		{name: "MaskARGB32", format: MaskARGB32},
		{name: "MaskLCD16", format: MaskLCD16},
		{name: "MaskSDF", format: MaskSDF},
	} {
		if !seen[want.format] {
			t.Errorf("no lane produced a %s glyph", want.name)
		}
	}
	if len(seen) != 4 {
		t.Errorf("stored formats = %v, want exactly the four documented ones", seen)
	}
}

// TestAllocGlyphImageHonorsImageSize pins allocGlyphImage as self-sufficient: it allocates exactly the plane its format
// names, sized to agree with ImageSize, and allocates nothing at all for empty or too-large glyphs. The ARGB32 and
// LCD16 lanes once skipped that gate and allocated Width*Height words unconditionally, which is up to 268 MB for an
// ARGB32 glyph just under the dimension cap — a glyph the rest of the code treats as image-less.
func TestAllocGlyphImageHonorsImageSize(t *testing.T) {
	for _, f := range []struct {
		name   string
		format MaskFormat
	}{
		{name: "A8", format: MaskA8},
		{name: "ARGB32", format: MaskARGB32},
		{name: "LCD16", format: MaskLCD16},
		{name: "SDF", format: MaskSDF},
	} {
		t.Run(f.name, func(t *testing.T) {
			for _, s := range []struct {
				name  string
				w     int32
				h     int32
				alloc bool
			}{
				{name: "ordinary", w: 3, h: 5, alloc: true},
				{name: "one pixel", w: 1, h: 1, alloc: true},
				{name: "zero width", w: 0, h: 5},
				{name: "zero height", w: 3, h: 0},
				{name: "at the width cap", w: maxGlyphWidth, h: 5},
				{name: "at the height cap", w: 3, h: maxGlyphHeight},
				{name: "past both caps", w: maxGlyphWidth + 1, h: maxGlyphHeight + 1},
			} {
				g := &Glyph{Width: s.w, Height: s.h, Format: f.format}
				allocGlyphImage(g)
				if got := g.HasImage(); got != s.alloc {
					t.Errorf("%s: HasImage = %v, want %v", s.name, got, s.alloc)
				}
				if !s.alloc {
					if g.Image != nil || g.Image32 != nil || g.Image16 != nil {
						t.Errorf("%s: allocated %d/%d/%d bytes/words for an image-less glyph", s.name, len(g.Image),
							len(g.Image32), len(g.Image16))
					}
					continue
				}
				// The allocated plane must hold exactly ImageSize bytes; the other two planes stay nil.
				pixels := int(s.w) * int(s.h)
				var bytes int
				switch f.format {
				case MaskARGB32:
					bytes = 4 * len(g.Image32)
					if len(g.Image32) != pixels || g.Image != nil || g.Image16 != nil {
						t.Errorf("%s: planes = %d/%d/%d, want 0/%d/0", s.name, len(g.Image), len(g.Image32),
							len(g.Image16), pixels)
					}
				case MaskLCD16:
					bytes = 2 * len(g.Image16)
					if len(g.Image16) != pixels || g.Image != nil || g.Image32 != nil {
						t.Errorf("%s: planes = %d/%d/%d, want 0/0/%d", s.name, len(g.Image), len(g.Image32),
							len(g.Image16), pixels)
					}
				default:
					bytes = len(g.Image)
					if len(g.Image) != pixels || g.Image32 != nil || g.Image16 != nil {
						t.Errorf("%s: planes = %d/%d/%d, want %d/0/0", s.name, len(g.Image), len(g.Image32),
							len(g.Image16), pixels)
					}
				}
				if bytes != g.ImageSize() {
					t.Errorf("%s: allocated %d bytes, ImageSize = %d", s.name, bytes, g.ImageSize())
				}
			}
		})
	}
}

// TestAllocGlyphImageDoesNotAllocateForTooLargeGlyph is the memory half of the guard: a too-large glyph must cost no
// allocation at all, in whichever plane its format would have used. A cap-width ARGB32 glyph is used because its
// unguarded allocation is the largest of the four.
func TestAllocGlyphImageDoesNotAllocateForTooLargeGlyph(t *testing.T) {
	var g Glyph
	for _, format := range []MaskFormat{MaskA8, MaskARGB32, MaskLCD16, MaskSDF} {
		if allocs := testing.AllocsPerRun(5, func() {
			g = Glyph{Width: maxGlyphWidth, Height: 4, Format: format}
			allocGlyphImage(&g)
		}); allocs != 0 {
			t.Errorf("format %d: %v allocations for a too-large glyph, want 0", format, allocs)
		}
	}
}

// TestSaturateBoundsNeverReEmpties pins the invariant behind saturateBounds having no post-rounding emptiness re-check:
// IsEmpty's negated comparisons already reject NaN edges and unsorted rects, and floor/ceil only widen whatever
// survives, so a rect that passes the gate always rounds out to a non-empty rect that contains it.
func TestSaturateBoundsNeverReEmpties(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	for _, c := range []struct {
		name string
		in   geom.Rect
	}{
		{name: "zero", in: geom.Rect{}},
		{name: "zero width", in: geom.RectLTRB(2, 1, 2, 4)},
		{name: "zero height", in: geom.RectLTRB(1, 2, 4, 2)},
		{name: "inverted", in: geom.RectLTRB(4, 4, 1, 1)},
		{name: "NaN left", in: geom.RectLTRB(nan, 1, 4, 4)},
		{name: "NaN top", in: geom.RectLTRB(1, nan, 4, 4)},
		{name: "NaN right", in: geom.RectLTRB(1, 1, nan, 4)},
		{name: "NaN bottom", in: geom.RectLTRB(1, 1, 4, nan)},
		{name: "all NaN", in: geom.RectLTRB(nan, nan, nan, nan)},
	} {
		if got := saturateBounds(c.in); got != (geom.Rect{}) {
			t.Errorf("%s: saturateBounds(%v) = %v, want the zero rect", c.name, c.in, got)
		}
	}

	sliverLeft := float32(1.25)
	sliverRight := math.Nextafter32(sliverLeft, 2)
	for _, c := range []struct {
		name string
		in   geom.Rect
	}{
		{name: "unit", in: geom.RectLTRB(0, 0, 1, 1)},
		{name: "fractional", in: geom.RectLTRB(1.2, -3.7, 4.1, -0.2)},
		{name: "inside one cell", in: geom.RectLTRB(-0.75, -0.75, -0.25, -0.25)},
		{name: "already integral", in: geom.RectLTRB(-4, -4, 4, 4)},
		{name: "one ULP wide", in: geom.RectLTRB(sliverLeft, sliverLeft, sliverRight, sliverRight)},
		{name: "beyond float32 integer precision", in: geom.RectLTRB(1e30, 1e30, 2e30, 2e30)},
		{name: "infinite", in: geom.RectLTRB(-inf, -inf, inf, inf)},
		{name: "half infinite", in: geom.RectLTRB(-1.5, -1.5, inf, inf)},
	} {
		got := saturateBounds(c.in)
		if got.IsEmpty() {
			t.Errorf("%s: saturateBounds(%v) = %v, want a non-empty rect", c.name, c.in, got)
			continue
		}
		if got != c.in.RoundOutRect() {
			t.Errorf("%s: saturateBounds(%v) = %v, want %v", c.name, c.in, got, c.in.RoundOutRect())
		}
		if got.Left > c.in.Left || got.Top > c.in.Top || got.Right < c.in.Right || got.Bottom < c.in.Bottom {
			t.Errorf("%s: saturateBounds(%v) = %v, which does not contain its input", c.name, c.in, got)
		}
	}

	// The same invariant over a sweep of sub-pixel slivers at every fractional offset: whatever cell a one-ULP-wide
	// rect lands in, rounding out widens it to that cell rather than collapsing it.
	for i := range 4096 {
		l := float32(i)*0.0173 - 35
		in := geom.RectLTRB(l, l, math.Nextafter32(l, 1e6), math.Nextafter32(l, 1e6))
		if in.IsEmpty() {
			t.Fatalf("sliver %d at %v is not a valid input", i, l)
		}
		if got := saturateBounds(in); got.IsEmpty() {
			t.Fatalf("sliver %d: saturateBounds(%v) = %v, want a non-empty rect", i, in, got)
		}
	}
}
