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
	"bytes"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stroke"
)

func TestPackedGlyphID(t *testing.T) {
	p := PackGlyphID(0x1234)
	if p.GlyphID() != 0x1234 {
		t.Fatalf("glyph ID round trip: got %#x", p.GlyphID())
	}
	if p.SubXOffset() != 0 || p.SubYOffset() != 0 {
		t.Fatalf("bare pack should have zero subpixel offsets")
	}

	// The X sub-pixel field quantizes the fractional position into quarters; the packing expects the rounding constant
	// to have been added already (the direct-mask-drawing path adds halfSampleFreq).
	mask := geom.IPoint{X: 3, Y: 0}
	cases := []struct {
		frac float32
		want float32
	}{
		{frac: 0.0, want: 0}, {frac: 0.24, want: 0}, {frac: 0.26, want: 0.25}, {frac: 0.51, want: 0.5}, {frac: 0.76, want: 0.75}, {frac: 0.99, want: 0.75},
	}
	for _, c := range cases {
		p = PackGlyphIDPoint(7, geom.Pt(10+c.frac, 3), mask)
		if p.GlyphID() != 7 {
			t.Fatalf("frac %v: glyph ID %d", c.frac, p.GlyphID())
		}
		if got := p.SubXOffset(); got != c.want {
			t.Errorf("frac %v: sub-x %v, want %v", c.frac, got, c.want)
		}
		if p.SubYOffset() != 0 {
			t.Errorf("frac %v: sub-y %v, want 0 (masked)", c.frac, p.SubYOffset())
		}
	}

	// Negative positions floor correctly (the [0,2) range trick).
	p = PackGlyphIDPoint(7, geom.Pt(-9.74, 0), mask) // fractional part 0.26
	if got := p.SubXOffset(); got != 0.25 {
		t.Errorf("negative pos: sub-x %v, want 0.25", got)
	}
}

func TestGlyphImageTooLarge(t *testing.T) {
	// imageTooLarge must gate both dimensions so an extreme height cannot drive an outsized mask allocation. Height is
	// otherwise saturated only to the 16-bit satUint16 ceiling (65535).
	cases := []struct {
		name          string
		width, height int32
		format        MaskFormat
		wantTooLarge  bool
		wantImageSize int
	}{
		{name: "small", width: 100, height: 100, format: MaskA8, wantTooLarge: false, wantImageSize: 100 * 100},
		{name: "wide-boundary", width: maxGlyphWidth, height: 10, format: MaskA8, wantTooLarge: true, wantImageSize: 0},
		{name: "wide-just-under", width: maxGlyphWidth - 1, height: 10, format: MaskA8, wantTooLarge: false, wantImageSize: (maxGlyphWidth - 1) * 10},
		{name: "tall-boundary", width: 10, height: maxGlyphHeight, format: MaskA8, wantTooLarge: true, wantImageSize: 0},
		{name: "tall-just-under", width: 10, height: maxGlyphHeight - 1, format: MaskA8, wantTooLarge: false, wantImageSize: 10 * (maxGlyphHeight - 1)},
		// Regression: an ~8191 wide × 65535 tall A8 glyph must be rejected rather than allocating ≈0.5 GB.
		{name: "extreme-height", width: maxGlyphWidth - 1, height: math.MaxUint16, format: MaskA8, wantTooLarge: true, wantImageSize: 0},
	}
	for _, c := range cases {
		g := &Glyph{Width: c.width, Height: c.height, Format: c.format}
		if got := g.imageTooLarge(); got != c.wantTooLarge {
			t.Errorf("%s: imageTooLarge()=%v, want %v", c.name, got, c.wantTooLarge)
		}
		if got := g.ImageSize(); got != c.wantImageSize {
			t.Errorf("%s: ImageSize()=%d, want %d", c.name, got, c.wantImageSize)
		}
	}
}

func TestRoundingSpec(t *testing.T) {
	spec := NewRoundingSpec(false, AxisAlignmentX)
	if spec.HalfAxisSampleFreq != geom.Pt(0.5, 0.5) {
		t.Errorf("non-subpixel: %v", spec.HalfAxisSampleFreq)
	}
	if spec.IgnorePositionFieldMask != (geom.IPoint{}) {
		t.Errorf("non-subpixel mask: %v", spec.IgnorePositionFieldMask)
	}
	spec = NewRoundingSpec(true, AxisAlignmentX)
	if spec.HalfAxisSampleFreq != geom.Pt(SubpixelRound, 0.5) {
		t.Errorf("subpixel X-aligned: %v", spec.HalfAxisSampleFreq)
	}
	if spec.IgnorePositionFieldMask != (geom.IPoint{X: 3, Y: 0}) {
		t.Errorf("subpixel X-aligned mask: %v", spec.IgnorePositionFieldMask)
	}
	spec = NewRoundingSpec(true, AxisAlignmentNone)
	if spec.IgnorePositionFieldMask != (geom.IPoint{X: 3, Y: 3 << packedSubPixelYShift}) {
		t.Errorf("subpixel unaligned mask: %v", spec.IgnorePositionFieldMask)
	}
}

// maskInk sums the coverage in a glyph mask.
func maskInk(g *Glyph) int {
	total := 0
	for _, v := range g.Image {
		total += int(v)
	}
	return total
}

// blankEdges counts the fully-zero rows at the top and bottom, and the fully-zero columns at the left and right, of an
// A8 glyph mask — how much empty margin the glyph's bounds carry on each side.
func blankEdges(g *Glyph) (top, bottom, left, right int) {
	w, h, rowBytes := int(g.Width), int(g.Height), int(g.RowBytes())
	rowInk := func(y int) (total int) {
		for x := range w {
			total += int(g.Image[y*rowBytes+x])
		}
		return total
	}
	colInk := func(x int) (total int) {
		for y := range h {
			total += int(g.Image[y*rowBytes+x])
		}
		return total
	}
	for top < h && rowInk(top) == 0 {
		top++
	}
	for bottom < h-top && rowInk(h-1-bottom) == 0 {
		bottom++
	}
	for left < w && colInk(left) == 0 {
		left++
	}
	for right < w-left && colInk(w-1-right) == 0 {
		right++
	}
	return top, bottom, left, right
}

func TestStrikeMaskGeneration(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 24, 1, 0)
	identity := geom.IdentityMatrix()

	spec := MakeMaskSpec(f, nil, &identity, nil)
	strike := spec.FindOrCreateStrike()

	gid := tf.UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("Roboto should map 'A'")
	}
	g, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if action != GlyphActionAccept {
		t.Fatalf("mask action = %d, want accept", action)
	}
	if g.IsEmpty() || g.Image == nil {
		t.Fatal("glyph mask should exist")
	}
	if maskInk(g) == 0 {
		t.Fatal("mask has no ink")
	}
	// The mask bounds must agree with the measuring strike's glyph bounds at identity.
	st, _ := makeCanonicalized(f, nil)
	want := st.glyphBounds(gid)
	got := g.IRect()
	if float32(got.Left) != want.Left || float32(got.Top) != want.Top ||
		float32(got.Right) != want.Right || float32(got.Bottom) != want.Bottom {
		t.Errorf("mask bounds %v, measuring bounds %v", got, want)
	}
	// The advance matches the measuring lane.
	if g.AdvanceX != st.glyphAdvance(gid) {
		t.Errorf("advance %v, measuring %v", g.AdvanceX, st.glyphAdvance(gid))
	}

	// The bounds are tight: the mask spans the glyph's outline bounds rounded out to whole pixels, so one extreme row
	// or column can still round to zero coverage — Roboto 'A' at 24 does exactly that at its apex row — but two blank
	// rows or columns at any one edge mean the bounds carry real slop (saturateGlyphBounds outsetting, say).
	top, bottom, left, right := blankEdges(g)
	for _, edge := range []struct {
		name  string
		blank int
	}{{"top", top}, {"bottom", bottom}, {"left", left}, {"right", right}} {
		if edge.blank > 1 {
			t.Errorf("%s edge has %d blank rows/columns (mask %dx%d); bounds are not tight",
				edge.name, edge.blank, g.Width, g.Height)
		}
	}

	// A space glyph is empty and drops.
	spaceGID := tf.UnicharToGlyph(' ')
	sg, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(spaceGID))
	if action != GlyphActionDrop || !sg.IsEmpty() {
		t.Errorf("space glyph: action %d empty %v; want drop/empty", action, sg.IsEmpty())
	}
	if sg.AdvanceX == 0 {
		t.Error("space should still carry an advance")
	}
}

// TestAtlasActionSizeGates drives the two atlas-bound mask actions across their bound: kDirectMask takes glyphs up to
// SideTooBigForAtlas, kMask stops two pixels short of it for the bilerp padding, and both must reject everything above.
// The text sizes bracket the crossing ('A' grows a bit under a pixel per point), so the walk sees each action flip —
// and sees kMask flip first, which is the whole point of the -2.
func TestAtlasActionSizeGates(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	gid := tf.UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("Roboto should map 'A'")
	}
	tests := []struct {
		name        string
		action      ActionType
		limit       int32
		firstReject float32
	}{
		{name: "kDirectMask", action: ActionDirectMask, limit: SideTooBigForAtlas},
		{name: "kMask", action: ActionMask, limit: SideTooBigForAtlas - 2},
	}
	for i, test := range tests {
		accepted, rejected := 0, 0
		for size := float32(320); size <= 400; size += 4 {
			spec := MakeWithNoDeviceSpec(NewFont(tf, size, 1, 0), nil)
			strike := spec.FindOrCreateStrike()
			g, action := strike.DigestFor(test.action, PackGlyphID(gid))
			want := GlyphActionReject
			if g.MaxDimension() <= test.limit {
				want = GlyphActionAccept
				accepted++
			} else {
				if rejected == 0 {
					tests[i].firstReject = size
				}
				rejected++
			}
			if action != want {
				t.Errorf("%s at size %v: %dx%d glyph action = %v, want %v",
					test.name, size, g.Width, g.Height, action, want)
			}
		}
		if accepted == 0 || rejected == 0 {
			t.Fatalf("%s: the size walk saw %d accepts and %d rejects; it must cross the atlas bound",
				test.name, accepted, rejected)
		}
	}
	// kMask's two pixels of bilerp padding must cost it a real size: it gives up before kDirectMask does.
	if tests[1].firstReject >= tests[0].firstReject {
		t.Errorf("kMask first rejects at size %v, kDirectMask at %v; kMask must give up first",
			tests[1].firstReject, tests[0].firstReject)
	}
}

func TestStrikeFontMetricsAreInDeviceSpace(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	gid := tf.UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("Roboto should map 'A'")
	}
	strikeFor := func(m geom.Matrix) *Strike {
		t.Helper()
		spec := MakeMaskSpec(NewFont(tf, 12, 1, 0), nil, &m, nil)
		return spec.FindOrCreateStrike()
	}
	advanceOf := func(s *Strike) float32 {
		t.Helper()
		g, _ := s.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
		return g.AdvanceX
	}

	// A strike reports one space: the device matrix that scales the advances and glyph bounds must scale the font
	// metrics with them. The scale applied is the single matrix's y scale, so a 2x device matrix doubles both.
	base := strikeFor(geom.IdentityMatrix())
	var scaled geom.Matrix
	scaled.SetScale(2, 2)
	doubled := strikeFor(scaled)
	baseMetrics, doubledMetrics := base.FontMetrics(), doubled.FontMetrics()
	if baseMetrics.Ascent >= 0 {
		t.Fatalf("base ascent %v, want negative (y-down)", baseMetrics.Ascent)
	}
	if got, want := advanceOf(doubled), 2*advanceOf(base); got != want {
		t.Fatalf("advance under 2x = %v, want %v", got, want)
	}
	for _, c := range []struct {
		name       string
		base, want float32
	}{
		{name: "Ascent", base: baseMetrics.Ascent, want: doubledMetrics.Ascent},
		{name: "Descent", base: baseMetrics.Descent, want: doubledMetrics.Descent},
		{name: "XHeight", base: baseMetrics.XHeight, want: doubledMetrics.XHeight},
		{name: "CapHeight", base: baseMetrics.CapHeight, want: doubledMetrics.CapHeight},
		{name: "XMax", base: baseMetrics.XMax, want: doubledMetrics.XMax},
		{name: "UnderlineThickness", base: baseMetrics.UnderlineThickness, want: doubledMetrics.UnderlineThickness},
	} {
		if 2*c.base != c.want {
			t.Errorf("%s: 2x device matrix gave %v, want %v (2 * %v)", c.name, c.want, 2*c.base, c.base)
		}
	}

	// Rotation is removed before the scale is taken, so a pure rotation leaves the metrics at their source-space
	// values.
	var rotated geom.Matrix
	rotated.SetRotate(90)
	if got := strikeFor(rotated).FontMetrics().Ascent; !geom.ScalarNearlyEqual(got, baseMetrics.Ascent) {
		t.Errorf("rotated ascent %v, want %v", got, baseMetrics.Ascent)
	}

	// A singular device matrix collapses the glyphs, and the metrics fall back to the measuring strike's size-1 recipe
	// rather than reporting a full-size ascent for zero-size glyphs.
	var singular geom.Matrix
	singular.SetScale(0, 0)
	if got := advanceOf(strikeFor(singular)); got != 0 {
		t.Errorf("singular advance %v, want 0", got)
	}
	got, want := strikeFor(singular).FontMetrics().Ascent, baseMetrics.Ascent/12
	if !geom.ScalarNearlyEqual(got, want) {
		t.Errorf("singular ascent %v, want %v (the size-1 fallback)", got, want)
	}
}

func TestStrikeAliasedEdgingMask(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	gid := tf.UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("Roboto should map 'A'")
	}
	identity := geom.IdentityMatrix()
	maskFor := func(edging Edging) *Glyph {
		t.Helper()
		f := NewFont(tf, 24, 1, 0)
		f.SetEdging(edging)
		spec := MakeMaskSpec(f, nil, &identity, nil)
		g, action := spec.FindOrCreateStrike().DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
		if action != GlyphActionAccept || g.Image == nil {
			t.Fatalf("edging %d: action %d, image nil=%v", edging, action, g.Image == nil)
		}
		return g
	}

	// EdgingAlias must reach the mask lane, not just the path and distance-field lanes: coverage comes out hard, so a
	// font renders with one edge treatment at every size (Font.HasSomeAntiAliasing, which the path lane consults,
	// reports false for it).
	aliased := maskFor(EdgingAlias)
	for i, v := range aliased.Image {
		if v != 0 && v != 255 {
			t.Fatalf("aliased mask has partial coverage %d at index %d", v, i)
		}
	}
	if maskInk(aliased) == 0 {
		t.Fatal("aliased mask has no ink")
	}

	// Anti-aliased edging still produces soft edges, and the two must be separate strikes: the rec flag is part of the
	// strike key, so one edging cannot serve a mask generated for the other.
	antiAliased := maskFor(EdgingAntiAlias)
	partial := 0
	for _, v := range antiAliased.Image {
		if v != 0 && v != 255 {
			partial++
		}
	}
	if partial == 0 {
		t.Error("anti-aliased mask has no partial coverage")
	}
	if aliased == antiAliased {
		t.Error("aliased and anti-aliased edging shared a glyph")
	}
	if aliased.IRect() != antiAliased.IRect() {
		t.Errorf("aliased bounds %v, anti-aliased %v; the edging must not move the bounds",
			aliased.IRect(), antiAliased.IRect())
	}

	// The SDF lane always measures anti-aliased coverage: aliased distance-field text is the shader's monochrome step
	// over the same field, so an aliased run font must not degrade the field it generates.
	f := NewFont(tf, 24, 1, 0)
	f.SetEdging(EdgingAlias)
	if spec := MakeSDFTMaskSpec(f, nil); spec.Rec.Flags&recFlagAliased != 0 {
		t.Error("the SDF rec must not carry the aliased flag")
	}
}

func TestStrikeSubpixelMasksDiffer(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 13, 1, 0)
	f.SetSubpixel(true)
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(f, nil, &identity, nil)
	strike := spec.FindOrCreateStrike()
	if !strike.scaler.rec.isSubpixel() {
		t.Fatal("rec should be subpixel")
	}

	gid := tf.UnicharToGlyph('l')
	g0, a0 := strike.DigestFor(ActionDirectMaskCPU, PackedGlyphID(uint32(gid)<<packedGlyphIDShift))
	g2, a2 := strike.DigestFor(ActionDirectMaskCPU, PackedGlyphID(uint32(gid)<<packedGlyphIDShift|2)) // sub-x = 0.5
	if a0 != GlyphActionAccept || a2 != GlyphActionAccept {
		t.Fatal("both subpixel variants should accept")
	}
	if g0 == g2 {
		t.Fatal("distinct packed IDs must produce distinct glyphs")
	}
	if g0.Left == g2.Left && bytes.Equal(g0.Image, g2.Image) {
		t.Error("subpixel variants should differ in bounds or coverage")
	}
}

func TestStrikeStrokedGlyph(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 24, 1, 0)
	identity := geom.IdentityMatrix()
	paint := &ScalerPaint{Style: stroke.PaintStyleStroke, Width: 2, MiterLimit: 4}
	spec := MakeMaskSpec(f, paint, &identity, nil)
	strike := spec.FindOrCreateStrike()

	gid := tf.UnicharToGlyph('O')
	g, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if action != GlyphActionAccept {
		t.Fatalf("stroked mask action = %d", action)
	}
	// Stroked 'O' has a hole in the ring: the center pixel must be transparent while ink exists.
	if maskInk(g) == 0 {
		t.Fatal("stroked glyph has no ink")
	}
	cx := g.Width / 2
	cy := g.Height / 2
	if g.Image[cy*g.RowBytes()+cx] != 0 {
		t.Error("stroke center should be empty")
	}
	// And its bounds outset the fill bounds.
	fillSpec := MakeMaskSpec(f, nil, &identity, nil)
	fillStrike := fillSpec.FindOrCreateStrike()
	fg, _ := fillStrike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if g.Left > fg.Left || g.Top > fg.Top || g.Width < fg.Width || g.Height < fg.Height {
		t.Errorf("stroke bounds %v should contain fill bounds %v", g.IRect(), fg.IRect())
	}
}

func TestStrikeMaskFilteredGlyph(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 24, 1, 0)
	identity := geom.IdentityMatrix()
	paint := &ScalerPaint{MaskFilter: maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)}
	spec := MakeMaskSpec(f, paint, &identity, nil)
	strike := spec.FindOrCreateStrike()

	gid := tf.UnicharToGlyph('A')
	g, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if action != GlyphActionAccept {
		t.Fatalf("blurred mask action = %d", action)
	}
	fillSpec := MakeMaskSpec(f, nil, &identity, nil)
	fillStrike := fillSpec.FindOrCreateStrike()
	fg, _ := fillStrike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if g.Width <= fg.Width || g.Height <= fg.Height {
		t.Errorf("blur bounds %v should outset fill bounds %v", g.IRect(), fg.IRect())
	}
	if maskInk(g) == 0 {
		t.Fatal("blurred glyph has no ink")
	}
	// Blur reaches the mask corners' vicinity: the top-left pixel should be faint, not fully opaque.
	if g.Image[0] == 0xFF {
		t.Error("blurred mask corner should not be fully opaque")
	}
}

func TestStrikePathAction(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 64, 1, 0)
	spec, scale := MakePathSpec(f, nil)
	if scale != 1 {
		t.Fatalf("64pt path spec scale = %v", scale)
	}
	strike := spec.FindOrCreateStrike()
	gid := tf.UnicharToGlyph('B')
	g, action := strike.DigestFor(ActionPath, PackGlyphID(gid))
	if action != GlyphActionAccept || g.Path() == nil {
		t.Fatalf("path action = %d, path %v", action, g.Path())
	}
	// The path strike measures at the canonical size regardless of the font size.
	f2 := NewFont(tf, 128, 1, 0)
	spec2, scale2 := MakePathSpec(f2, nil)
	if scale2 != 2 {
		t.Fatalf("128pt path spec scale = %v", scale2)
	}
	if spec2.Rec.TextSize != canonicalTextSizeForPaths {
		t.Fatalf("path rec size = %v", spec2.Rec.TextSize)
	}
	// Both fonts share the same strike (subpixel forced on by setupForAsPaths, size canonical).
	if spec.Rec != spec2.Rec {
		t.Error("path specs at different sizes should share a rec")
	}
}

func TestShouldDrawAsPathMatrix(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 100, 1, 0)
	identity := geom.IdentityMatrix()
	if ShouldDrawAsPathMatrix(nil, f, &identity) {
		t.Error("100pt identity should rasterize")
	}
	var big geom.Matrix
	big.SetScale(3, 3)
	if !ShouldDrawAsPathMatrix(nil, f, &big) {
		t.Error("300 effective pt should draw as path")
	}
	hairline := &ScalerPaint{Style: stroke.PaintStyleStroke, Width: 0}
	if !ShouldDrawAsPathMatrix(hairline, f, &identity) {
		t.Error("hairline stroke should draw as path")
	}
}

func TestStrikeCacheLRUAndBudget(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	cache := NewStrikeCache()
	cache.sizeLimit = 32 * 1024

	identity := geom.IdentityMatrix()
	var glyphs []uint16
	for gid := uint16(1); gid < 60; gid++ {
		glyphs = append(glyphs, gid)
	}
	results := make([]*Glyph, 0, len(glyphs))
	for size := float32(10); size <= 40; size++ {
		f := NewFont(tf, size, 1, 0)
		spec := MakeMaskSpec(f, nil, &identity, nil)
		s := cache.FindOrCreateStrike(&spec)
		for _, gid := range glyphs {
			s.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
		}
		_ = s.Metrics(glyphs, results)
	}
	if used := cache.TotalMemoryUsed(); used > 64*1024 {
		t.Errorf("cache did not purge: %d bytes used", used)
	}
	if n := cache.StrikeCount(); n >= 31 {
		t.Errorf("cache kept all %d strikes", n)
	}

	// A repeated lookup returns the same strike (and the cached glyph pointer). Use a fresh cache with the default
	// budget — under a tiny budget the purge may evict a just-created strike.
	roomy := NewStrikeCache()
	f := NewFont(tf, 40, 1, 0)
	spec := MakeMaskSpec(f, nil, &identity, nil)
	s1 := roomy.FindOrCreateStrike(&spec)
	g1, _ := s1.DigestFor(ActionDirectMaskCPU, PackGlyphID(5))
	s2 := roomy.FindOrCreateStrike(&spec)
	g2, _ := s2.DigestFor(ActionDirectMaskCPU, PackGlyphID(5))
	if s1 != s2 || g1 != g2 {
		t.Error("identical specs should share a strike and glyphs")
	}
}

// A NaN anywhere in the strike key would make the key not equal to itself: every lookup would miss, so each draw would
// build another strike, and removeStrike's delete would never match, stranding all of them (and every glyph mask they
// generated) in the map beyond the reach of either budget. MakeRecAndEffects canonicalizes NaN away instead.
func TestStrikeCacheNaNKeysDoNotLeak(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	nan := float32(math.NaN())
	identity := geom.IdentityMatrix()
	var nanScale geom.Matrix
	nanScale.SetScale(nan, 2)
	var nanSkew geom.Matrix
	nanSkew.SetAll(1, nan, 0, nan, 1, 0, 0, 0, 1)
	for _, tc := range []struct {
		font   *Font
		paint  *ScalerPaint
		device *geom.Matrix
		name   string
	}{
		{name: "size", font: NewFont(tf, nan, 1, 0), device: &identity},
		{name: "scaleX", font: NewFont(tf, 24, nan, 0), device: &identity},
		{name: "skewX", font: NewFont(tf, 24, 1, nan), device: &identity},
		{name: "deviceScale", font: NewFont(tf, 24, 1, 0), device: &nanScale},
		{name: "deviceSkew", font: NewFont(tf, 24, 1, 0), device: &nanSkew},
		{
			name:   "miterLimit",
			font:   NewFont(tf, 24, 1, 0),
			paint:  &ScalerPaint{Style: stroke.PaintStyleStroke, Width: 2, MiterLimit: nan},
			device: &identity,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec, effects := MakeRecAndEffects(tc.font, tc.paint, tc.device, nil)
			if same := rec; rec != same {
				t.Fatalf("rec does not equal its own copy, so it cannot key the cache: %+v", rec)
			}
			cache := NewStrikeCache()
			spec := StrikeSpec{Typeface: tc.font.Typeface(), Rec: rec, Effects: effects}
			s1 := cache.FindOrCreateStrike(&spec)
			s2 := cache.FindOrCreateStrike(&spec)
			if s1 != s2 {
				t.Error("identical specs built a second strike")
			}
			s1.DigestFor(ActionDirectMaskCPU, PackGlyphID(tf.UnicharToGlyph('H')))
			if n := len(cache.strikes); n != 1 || cache.count != 1 {
				t.Fatalf("cache holds %d map entries and counts %d strikes, want 1 and 1", n, cache.count)
			}
			cache.PurgeAll()
			if n := len(cache.strikes); n != 0 || cache.count != 0 || cache.totalMemoryUsed != 0 {
				t.Errorf("after PurgeAll: %d map entries, count %d, %d bytes; want 0, 0, 0", n, cache.count,
					cache.totalMemoryUsed)
			}
		})
	}
}

func TestScalerRecSingular(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 0, 1, 0)
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(f, nil, &identity, nil)
	strike := spec.FindOrCreateStrike()
	g, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(5))
	if action != GlyphActionDrop {
		t.Errorf("size-0 glyph action = %d, want drop", action)
	}
	if g.AdvanceX != 0 || !g.IsEmpty() {
		t.Error("size-0 glyphs must be zero")
	}
	// Rotated 90 degrees is not singular.
	var rot geom.Matrix
	rot.SetRotate(90)
	f24 := NewFont(tf, 24, 1, 0)
	rec, _ := MakeRecAndEffects(f24, nil, &rot, nil)
	if rec.singular() {
		t.Error("rotation must not be singular")
	}
	if rec.computeAxisAlignmentForHText() != AxisAlignmentY {
		t.Errorf("90-degree rotation should be Y-aligned, got %d", rec.computeAxisAlignmentForHText())
	}
}

// TestComputeScaleNonFiniteBranches pins which of computeScale's two branches handles each non-finite single matrix. A
// non-finite skew is never zero, so it always takes the Givens branch: the non-skewed branch below it only ever sees
// skews that are exactly zero, which is why its finite gate tests the scales alone. The NaN cases are unreachable
// through MakeRecAndEffects (canonicalizeKeyFloats scrubs them) but reachable inside the matrix concat, where an
// infinity times a zero yields one.
func TestComputeScaleNonFiniteBranches(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	for _, tc := range []struct {
		name string
		rec  ScalerRec
		// wantGivens is true when the rec's single matrix takes the skew/mirror (Givens rotation) branch.
		wantGivens bool
		wantOK     bool
		wantScale  geom.Point
	}{
		{
			name:      "plain scale",
			rec:       ScalerRec{TextSize: 24, PreScaleX: 1, Post2x2: [4]float32{1, 0, 0, 1}},
			wantOK:    true,
			wantScale: geom.Pt(24, 24),
		},
		{
			name:       "mirrored x",
			rec:        ScalerRec{TextSize: 24, PreScaleX: -1, Post2x2: [4]float32{1, 0, 0, 1}},
			wantGivens: true,
			wantOK:     true,
			wantScale:  geom.Pt(24, 24),
		},
		{
			name:       "infinite skew",
			rec:        ScalerRec{TextSize: 1, PreScaleX: 1, PreSkewX: inf, Post2x2: [4]float32{1, 0, 0, 1}},
			wantGivens: true,
		},
		{
			name:       "negative infinite skew",
			rec:        ScalerRec{TextSize: 1, PreScaleX: 1, PreSkewX: -inf, Post2x2: [4]float32{1, 0, 0, 1}},
			wantGivens: true,
		},
		{
			name:       "NaN skew",
			rec:        ScalerRec{TextSize: 1, PreScaleX: 1, PreSkewX: nan, Post2x2: [4]float32{1, 0, 0, 1}},
			wantGivens: true,
		},
		{
			name: "infinite x scale",
			rec:  ScalerRec{TextSize: 1, PreScaleX: inf, Post2x2: [4]float32{1, 0, 0, 1}},
		},
		{
			name: "NaN x scale",
			rec:  ScalerRec{TextSize: 1, PreScaleX: nan, Post2x2: [4]float32{1, 0, 0, 1}},
		},
		{
			name: "infinite y scale",
			rec:  ScalerRec{TextSize: 1, PreScaleX: 1, Post2x2: [4]float32{1, 0, 0, inf}},
		},
		{
			name: "scales below the tolerance",
			rec:  ScalerRec{TextSize: 1.0 / 8192, PreScaleX: 1, Post2x2: [4]float32{1, 0, 0, 1}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := tc.rec.singleMatrix()
			sx := float64(a.Get(geom.MScaleX))
			sy := float64(a.Get(geom.MScaleY))
			kx := float64(a.Get(geom.MSkewX))
			ky := float64(a.Get(geom.MSkewY))
			givens := kx != 0 || ky != 0 || sx < 0 || sy < 0
			if givens != tc.wantGivens {
				t.Fatalf("single matrix [%v %v %v %v] takes the Givens branch = %v, want %v", sx, kx, ky, sy, givens,
					tc.wantGivens)
			}
			if !givens && (!isFinite(kx) || !isFinite(ky)) {
				t.Fatalf("the non-skewed branch saw non-finite skews (%v, %v); its finite gate must cover them", kx, ky)
			}
			scale, ok := tc.rec.computeScale()
			if ok != tc.wantOK {
				t.Fatalf("computeScale ok = %v, want %v (scale %v)", ok, tc.wantOK, scale)
			}
			if ok && scale != tc.wantScale {
				t.Errorf("scale = %v, want %v", scale, tc.wantScale)
			}
			if tc.rec.singular() == tc.wantOK {
				t.Errorf("singular() = %v, want %v", tc.rec.singular(), !tc.wantOK)
			}
		})
	}
}

func TestGlyphRotatedMaskCoversInk(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 24, 1, 0)
	var rot geom.Matrix
	rot.SetRotate(30)
	spec := MakeMaskSpec(f, nil, &rot, nil)
	strike := spec.FindOrCreateStrike()
	gid := tf.UnicharToGlyph('M')
	g, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if action != GlyphActionAccept || maskInk(g) == 0 {
		t.Fatalf("rotated glyph should render (action %d)", action)
	}
	// The mask bounds must contain the rotated path's bounds.
	pg, _ := strike.DigestFor(ActionPath, PackGlyphID(gid))
	pb := pg.Path().Bounds()
	mb := g.IRect()
	if pb.Left < float32(mb.Left) || pb.Top < float32(mb.Top) ||
		pb.Right > float32(mb.Right) || pb.Bottom > float32(mb.Bottom) {
		t.Errorf("mask bounds %v do not cover path bounds %v", mb, pb)
	}
}

func TestTypefaceBoundsAndFontBounds(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	b := tf.Bounds()
	if b.IsEmpty() {
		t.Fatal("Roboto bounds should be non-empty")
	}
	// Roboto's head bbox: (-1509, -555, 2352, 2163)/2048, y-flipped.
	want := geom.RectLTRB(-1509.0/2048, -2163.0/2048, 2352.0/2048, 555.0/2048)
	if b != want {
		t.Errorf("bounds %v, want %v", b, want)
	}
	f := NewFont(tf, 20, 1, 0)
	fb := GetFontBounds(f)
	if fb.Width() <= b.Width()*19 || fb.Width() >= b.Width()*21 {
		t.Errorf("font bounds should scale by size: %v vs %v", fb, b)
	}
	if EmptyTypeface().Bounds() != (geom.Rect{}) {
		t.Error("empty typeface bounds should be empty")
	}
}

// unhashablePathEffect is a value-receiver stroke.PathEffect over unhashable state: a slice field makes its dynamic
// type one Go refuses to hash, which is all a caller outside this module needs to do by accident. FilterPath reports
// "did not filter", so the glyph still renders from its plain outline.
type unhashablePathEffect struct{ intervals []float32 }

func (unhashablePathEffect) FilterPath(_, _ *path.Path, _ *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	return false
}

func (unhashablePathEffect) AsPoints(_ *stroke.PointData, _ *path.Path, _ *stroke.Rec, _ *geom.Matrix,
	_ *geom.Rect,
) bool {
	return false
}

func (unhashablePathEffect) ComputeFastBounds(_ *geom.Rect) bool { return false }

// poisonedMaskFilter is comparable as a type — reflect.Type.Comparable reports true — yet unhashable as a value,
// because its interface-typed field holds a slice. Only an actual hash attempt can tell.
type poisonedMaskFilter struct{ payload any }

func (poisonedMaskFilter) FilterMask(_ *raster.Mask, _ *geom.Matrix) (*raster.Mask, geom.IPoint, bool) {
	return nil, geom.IPoint{}, false
}

func (poisonedMaskFilter) ComputeFastBounds(src geom.Rect) geom.Rect { return src }

func TestStrikeCacheUnhashableEffects(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	gid := tf.UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("Roboto should map 'A'")
	}
	identity := geom.IdentityMatrix()
	for _, c := range []struct {
		paint *ScalerPaint
		name  string
	}{
		{
			name:  "path effect",
			paint: &ScalerPaint{PathEffect: unhashablePathEffect{intervals: []float32{2, 2}}},
		},
		{
			name:  "mask filter",
			paint: &ScalerPaint{MaskFilter: poisonedMaskFilter{payload: []int{1}}},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := NewFont(tf, 24, 1, 0)
			spec := MakeMaskSpec(f, c.paint, &identity, nil)
			if spec.Keyable() {
				t.Fatal("the spec must not report itself keyable")
			}

			// The draw must go through — an effect the map cannot hash used to panic here with "hash of unhashable
			// type" — and the strike must work, just uncached.
			cache := NewStrikeCache()
			strike := cache.FindOrCreateStrike(&spec)
			g, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
			if action != GlyphActionAccept || maskInk(g) == 0 {
				t.Fatalf("action %v, ink %d", action, maskInk(g))
			}

			// Uncached: nothing entered the map or the byte accounting, so the strike cannot be looked up, evicted, or
			// leaked into a budget it will never be purged from.
			if got := cache.StrikeCount(); got != 0 {
				t.Errorf("cached strike count %d, want 0", got)
			}
			if got := cache.TotalMemoryUsed(); got != 0 {
				t.Errorf("cached bytes %d, want 0", got)
			}
			if again := cache.FindOrCreateStrike(&spec); again == strike {
				t.Error("an uncached spec must not resolve to the same strike twice")
			}
		})
	}

	// The hashable effects the module ships still key: an uncacheable spec must be the exception, not the rule.
	f := NewFont(tf, 24, 1, 0)
	shipped := MakeMaskSpec(f, &ScalerPaint{MaskFilter: maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)},
		&identity, nil)
	if !shipped.Keyable() {
		t.Error("a shipped mask filter must be keyable")
	}
	if plain := MakeMaskSpec(f, nil, &identity, nil); !plain.Keyable() {
		t.Error("a spec with no effects must be keyable")
	}
}

func TestStrikeConcurrentAccess(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	cache := NewStrikeCache()
	identity := geom.IdentityMatrix()
	done := make(chan bool)
	for w := 0; w < 8; w++ {
		go func(worker int) {
			defer func() { done <- true }()
			for i := 0; i < 40; i++ {
				f := NewFont(tf, float32(10+(worker+i)%6), 1, 0)
				spec := MakeMaskSpec(f, nil, &identity, nil)
				s := cache.FindOrCreateStrike(&spec)
				gid := uint16(1 + (worker*40+i)%60)
				s.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
				s.DigestFor(ActionPath, PackGlyphID(gid))
			}
		}(w)
	}
	for w := 0; w < 8; w++ {
		<-done
	}
}
