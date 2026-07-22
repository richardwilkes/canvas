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
	// to have been added already (prepare_for_direct_mask_drawing adds halfSampleFreq).
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

	// Rows at the extremes have ink (bounds are tight for the control-box recipe on 'A').
	top := g.Image[:g.Width]
	bottom := g.Image[(g.Height-1)*g.RowBytes():]
	if allZero(top) && allZero(bottom) {
		t.Error("expected ink near mask edges")
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

func allZero(b []uint8) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
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
