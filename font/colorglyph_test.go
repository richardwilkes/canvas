// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the color-glyph scaler lanes over colr/sbix/cbdt test fonts. All three map U+1F600 to the same smiley
// artwork: colr.ttf as five COLRv0 layers (palette 1 = black ring, palette 2 = the (255,204,0) face, three
// foreground-color layers for the eyes/mouth), sbix/cbdt as PNG strikes at 16/64/128 ppem whose face center is
// (255,204,0,255).

package font

import (
	"bytes"
	"encoding/binary"
	"image/png"
	"os"
	"slices"
	"testing"

	"github.com/go-text/typesetting/font/opentype"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
)

const smiley = 0x1F600

func loadColorTypeface(t *testing.T, name string) *Typeface {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	tf, err := NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	return tf
}

// smileyGlyph resolves U+1F600's glyph through a mask strike at the given size, preparing the image.
func smileyGlyph(t *testing.T, tf *Typeface, size float32, paint *ScalerPaint) (*Glyph, GlyphAction) {
	t.Helper()
	gid := tf.UnicharToGlyph(smiley)
	if gid == 0 {
		t.Fatal("U+1F600 not mapped")
	}
	f := NewFont(tf, size, 1, 0)
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(f, paint, &identity, nil)
	strike := spec.FindOrCreateStrike()
	return strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
}

// faceCenterWord is the premultiplied device word for the opaque face color (255, 204, 0).
var faceCenterWord = uint32(255) | uint32(204)<<8 | uint32(0)<<16 | uint32(255)<<24

func TestColorGlyphNeedsCurrentColor(t *testing.T) {
	if !loadColorTypeface(t, "colr.ttf").GlyphMaskNeedsCurrentColor() {
		t.Error("colr.ttf: COLR table must need the current color")
	}
	if loadColorTypeface(t, "sbix.ttf").GlyphMaskNeedsCurrentColor() {
		t.Error("sbix.ttf must not need the current color")
	}
	if loadColorTypeface(t, "Roboto-Regular.ttf").GlyphMaskNeedsCurrentColor() {
		t.Error("Roboto must not need the current color")
	}
}

func TestFaceColorPaintMalformedCOLR(t *testing.T) {
	// hasCOLR is set from the mere presence of nonempty COLR table bytes, but typesetting leaves face.COLR nil when
	// ParseCOLR fails on a malformed table. Simulate that state (bytes present, parse failed) with a non-COLR face and a
	// forced hasCOLR flag: the color lanes must report "no color glyph" instead of dereferencing a nil COLR table.
	tf := loadColorTypeface(t, "Roboto-Regular.ttf")
	if tf.face.COLR != nil {
		t.Fatal("Roboto unexpectedly carries a parsed COLR table")
	}
	tf.hasCOLR = true
	gid := tf.UnicharToGlyph('A')
	if gid == 0 {
		t.Fatal("'A' not mapped")
	}
	if paint, ok := tf.faceColorPaint(opentype.GID(gid)); ok || paint != nil {
		t.Errorf("faceColorPaint = (%v, %v), want (nil, false)", paint, ok)
	}
	if layers, ok := tf.faceColorV0Layers(opentype.GID(gid)); ok || layers != nil {
		t.Errorf("faceColorV0Layers = (%v, %v), want (nil, false)", layers, ok)
	}
	if tab := tf.colrTable(); tab != nil {
		t.Errorf("colrTable = %v, want nil", tab)
	}
}

func TestCOLRv0GlyphMetrics(t *testing.T) {
	tf := loadColorTypeface(t, "colr.ttf")
	g, action := smileyGlyph(t, tf, 50, nil)
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	if g.Format != MaskARGB32 {
		t.Fatalf("format %v, want ARGB32", g.Format)
	}
	// The layer-union bounding box: layer gid 9 spans x [24, 767], y-up [23, 765] font units (upem 1000). At size 50
	// that maps to [1.2, 38.35] x [-38.25, -1.15], rounded out.
	want := geom.IRectLTRB(1, -39, 39, -1)
	if g.IRect() != want {
		t.Errorf("bounds %v, want %v", g.IRect(), want)
	}
	// Advance 800/1000 * 50.
	if g.AdvanceX != 40 {
		t.Errorf("advance %v, want 40", g.AdvanceX)
	}
	// Color glyphs never produce paths (neverRequestPath).
	if g.Path() != nil {
		t.Error("color glyph must have no path")
	}
}

func TestCOLRv0GlyphImage(t *testing.T) {
	tf := loadColorTypeface(t, "colr.ttf")
	g, action := smileyGlyph(t, tf, 50, nil)
	if action != GlyphActionAccept || g.Image32 == nil {
		t.Fatalf("no image (action %v)", action)
	}
	// The face circle (palette 2: R=255 G=204 B=0, opaque) covers the glyph center.
	cx := int(g.Width) / 2
	cy := int(g.Height) / 2
	if got := g.Image32[cy*int(g.Width)+cx]; got != faceCenterWord {
		t.Errorf("center pixel %#08x, want %#08x", got, faceCenterWord)
	}
	// Corners lie outside the ring: transparent.
	if got := g.Image32[0]; got != 0 {
		t.Errorf("corner pixel %#08x, want transparent", got)
	}
	// Foreground-color layers (palette index 0xFFFF) default to black; some pixels must be opaque black.
	black := uint32(0xFF) << 24
	found := 0
	for _, w := range g.Image32 {
		if w == black {
			found++
		}
	}
	if found == 0 {
		t.Error("no foreground (black) pixels found")
	}
}

func TestCOLRv0ForegroundColor(t *testing.T) {
	tf := loadColorTypeface(t, "colr.ttf")
	red := colorcore.ARGB(0xFF, 0xFF, 0, 0)
	paint := &ScalerPaint{Color: red}
	g, action := smileyGlyph(t, tf, 50, paint)
	if action != GlyphActionAccept || g.Image32 == nil {
		t.Fatalf("no image (action %v)", action)
	}
	redWord := uint32(0xFF) | uint32(0xFF)<<24
	blackWord := uint32(0xFF) << 24
	reds, blacks := 0, 0
	for _, w := range g.Image32 {
		switch w {
		case redWord:
			reds++
		case blackWord:
			blacks++
		}
	}
	if reds == 0 {
		t.Error("foreground layers must paint with the paint color")
	}
	// The only black in this glyph should come from palette 1 (the ring), which stays black.
	if blacks == 0 {
		t.Error("palette layers must not change with the paint color")
	}

	// The foreground color is part of the strike key for COLR faces: black-paint and red-paint strikes must differ, and
	// their glyphs must differ.
	gBlack, _ := smileyGlyph(t, tf, 50, nil)
	if gBlack == g {
		t.Error("different foreground colors must resolve different strikes")
	}

	// A non-COLR face must not fragment strikes by color.
	sbix := loadColorTypeface(t, "sbix.ttf")
	g1, _ := smileyGlyph(t, sbix, 50, paint)
	g2, _ := smileyGlyph(t, sbix, 50, nil)
	if g1 != g2 {
		t.Error("sbix strikes must not key on the paint color")
	}
}

func TestBitmapGlyphMetrics(t *testing.T) {
	for _, name := range []string{"sbix.ttf", "cbdt.ttf"} {
		tf := loadColorTypeface(t, name)
		g, action := smileyGlyph(t, tf, 64, nil)
		if action != GlyphActionAccept {
			t.Fatalf("%s: action %v", name, action)
		}
		if g.Format != MaskARGB32 {
			t.Fatalf("%s: format %v, want ARGB32", name, g.Format)
		}
		// The 64-ppem strike is 52x52 px with extents 812.5 font units square: at size 64 the device bounds are exactly
		// [0, -52, 52, 0].
		want := geom.IRectLTRB(0, -52, 52, 0)
		if g.IRect() != want {
			t.Errorf("%s: bounds %v, want %v", name, g.IRect(), want)
		}
		if g.AdvanceX != 51.2 {
			t.Errorf("%s: advance %v, want 51.2", name, g.AdvanceX)
		}
		if g.Path() != nil {
			t.Errorf("%s: bitmap glyph must have no path", name)
		}
		// Path drawing rejects color glyphs, so huge-text draws fall back to the mask stages.
		f := NewFont(tf, 64, 1, 0)
		spec, _ := MakePathSpec(f, nil)
		strike := spec.FindOrCreateStrike()
		_, pathAction := strike.DigestFor(ActionPath, PackGlyphID(tf.UnicharToGlyph(smiley)))
		if pathAction != GlyphActionReject {
			t.Errorf("%s: path action %v, want reject", name, pathAction)
		}
	}
}

func TestBitmapGlyphImageNativeSize(t *testing.T) {
	// At the strike-native size the bitmap transform is the identity, so the mask must be a byte-exact premultiplied
	// copy of the strike PNG (the FT direct-copy lane; the shader's linear filter degrades to nearest at integer
	// translates).
	tf := loadColorTypeface(t, "sbix.ttf")
	g, action := smileyGlyph(t, tf, 64, nil)
	if action != GlyphActionAccept || g.Image32 == nil {
		t.Fatalf("no image (action %v)", action)
	}
	bm, _, ok := tf.faceBitmapGlyph(3, 64)
	if !ok {
		t.Fatal("no strike bitmap")
	}
	src, err := png.Decode(bytes.NewReader(bm.Data))
	if err != nil {
		t.Fatal(err)
	}
	b := src.Bounds()
	if int32(b.Dx()) != g.Width || int32(b.Dy()) != g.Height {
		t.Fatalf("dims %dx%d vs glyph %dx%d", b.Dx(), b.Dy(), g.Width, g.Height)
	}
	mismatches := 0
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			r, gr, bl, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA() // premultiplied 16-bit
			want := r>>8 | (gr>>8)<<8 | (bl>>8)<<16 | (a>>8)<<24
			got := g.Image32[y*int(g.Width)+x]
			if got != want {
				if delta(got, want) > 1 {
					mismatches++
				}
			}
		}
	}
	if mismatches != 0 {
		t.Errorf("%d pixels differ by more than 1 per channel", mismatches)
	}
	// And the well-known center color.
	cx, cy := int(g.Width)/2, int(g.Height)/2
	if got := g.Image32[cy*int(g.Width)+cx]; got != faceCenterWord {
		t.Errorf("center %#08x, want %#08x", got, faceCenterWord)
	}
}

// delta returns the max per-channel byte difference between two device words.
func delta(a, b uint32) int {
	m := 0
	for i := 0; i < 4; i++ {
		d := int(uint8(a>>(8*i))) - int(uint8(b>>(8*i)))
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func TestBitmapGlyphImageScaled(t *testing.T) {
	// Size 32 uses the 64-ppem strike scaled by 0.5: 26x26 with the face color at the center and transparent corners.
	tf := loadColorTypeface(t, "cbdt.ttf")
	g, action := smileyGlyph(t, tf, 32, nil)
	if action != GlyphActionAccept || g.Image32 == nil {
		t.Fatalf("no image (action %v)", action)
	}
	if g.Width != 26 || g.Height != 26 {
		t.Fatalf("dims %dx%d, want 26x26", g.Width, g.Height)
	}
	center := g.Image32[13*26+13]
	if delta(center, faceCenterWord) > 2 {
		t.Errorf("center %#08x, want ~%#08x", center, faceCenterWord)
	}
	if g.Image32[0] != 0 {
		t.Errorf("corner %#08x, want transparent", g.Image32[0])
	}
}

func TestStrikePpemFor(t *testing.T) {
	cases := []struct {
		scale float32
		want  uint16
	}{
		{scale: 0, want: 0},
		{scale: 0.001, want: 1},
		{scale: 12, want: 12},
		{scale: 64, want: 64},
		{scale: 64.2, want: 65},
		// Sub-1/64 fractions quantize away when converting to a 26.6 fixed-point ppem.
		{scale: 64.005, want: 64},
		{scale: 1e9, want: 65535},
	}
	for _, c := range cases {
		if got := strikePpemFor(c.scale); got != c.want {
			t.Errorf("strikePpemFor(%v) = %d, want %d", c.scale, got, c.want)
		}
	}
}

func TestColorGlyphBlurMaskFilter(t *testing.T) {
	// The box blur accepts ARGB32 sources: the alpha plane blurs into an A8 dst and the glyph's mask format flips to
	// match.
	tf := loadColorTypeface(t, "sbix.ttf")
	blur := maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)
	paint := &ScalerPaint{MaskFilter: blur}
	g, action := smileyGlyph(t, tf, 64, paint)
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	if g.Format != MaskA8 {
		t.Fatalf("format %v, want A8 after blur", g.Format)
	}
	if g.Image == nil || g.Image32 != nil {
		t.Fatal("blurred color glyph must fill the A8 plane")
	}
	unfiltered, _ := smileyGlyph(t, tf, 64, nil)
	if g.Width <= unfiltered.Width || g.Height <= unfiltered.Height {
		t.Errorf("blur must outset bounds: %dx%d vs %dx%d", g.Width, g.Height, unfiltered.Width, unfiltered.Height)
	}
	// The blurred silhouette's center must be fully opaque coverage (face is opaque there).
	cx := int(g.Width) / 2
	cy := int(g.Height) / 2
	if got := g.Image[cy*int(g.Width)+cx]; got != 0xFF {
		t.Errorf("center coverage %d, want 255", got)
	}
}

func TestColorGlyphNonBlurMaskFilter(t *testing.T) {
	// Table-based filters return false for kARGB32 sources, so the glyph renders unfiltered in color.
	tf := loadColorTypeface(t, "sbix.ttf")
	gamma := maskfilter.NewGamma(2.2)
	if maskfilter.AcceptsColorMask(gamma) {
		t.Fatal("gamma filter must not accept color masks")
	}
	paint := &ScalerPaint{MaskFilter: gamma}
	g, action := smileyGlyph(t, tf, 64, paint)
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	if g.Format != MaskARGB32 || g.Image32 == nil {
		t.Fatalf("format %v, want unfiltered ARGB32", g.Format)
	}
	unfiltered, _ := smileyGlyph(t, tf, 64, nil)
	if g.IRect() != unfiltered.IRect() {
		t.Errorf("bounds %v changed vs unfiltered %v", g.IRect(), unfiltered.IRect())
	}
	cx, cy := int(g.Width)/2, int(g.Height)/2
	if got := g.Image32[cy*int(g.Width)+cx]; got != faceCenterWord {
		t.Errorf("center %#08x, want %#08x", got, faceCenterWord)
	}
}

// retargetCOLRv0Layers rewrites every COLR v0 layer record in an sfnt to reference gid, leaving the base-glyph records
// (and every other table) alone. It builds the "a layer glyph is itself a color glyph" font the COLR lanes must handle:
// go-text's GlyphData prefers a glyph's COLR/bitmap/SVG entry over its outline, so a layer pointing at a COLR base
// glyph only renders through the raw-outline accessor.
func retargetCOLRv0Layers(t *testing.T, data []byte, gid uint16) []byte {
	t.Helper()
	out := bytes.Clone(data)
	numTables := int(binary.BigEndian.Uint16(out[4:]))
	colr := -1
	for i := range numTables {
		rec := 12 + i*16
		if string(out[rec:rec+4]) == "COLR" {
			colr = int(binary.BigEndian.Uint32(out[rec+8:]))
			break
		}
	}
	if colr < 0 {
		t.Fatal("no COLR table")
	}
	// COLR v0 header: version, numBaseGlyphRecords, baseGlyphRecordsOffset, layerRecordsOffset, numLayerRecords.
	layerRecords := colr + int(binary.BigEndian.Uint32(out[colr+8:]))
	numLayerRecords := int(binary.BigEndian.Uint16(out[colr+12:]))
	if numLayerRecords == 0 {
		t.Fatal("no COLR v0 layer records")
	}
	for i := range numLayerRecords { // LayerRecord: glyphID, paletteIndex
		binary.BigEndian.PutUint16(out[layerRecords+i*4:], gid)
	}
	return out
}

func TestCOLRv0LayerIsItselfAColorGlyph(t *testing.T) {
	// A COLRv0 layer whose glyph carries its own COLR entry must still be filled with that glyph's outline. Loading it
	// through the GlyphData preference order answers with the COLR entry instead, dropping the layer from both the
	// bounds union and the mask.
	const colorLayerGID = 2 // a COLR base glyph in colr.ttf that also has a real 'glyf' outline
	data, err := os.ReadFile("testdata/colr.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := NewTypefaceFromData(retargetCOLRv0Layers(t, data, colorLayerGID), 0)
	if err != nil {
		t.Fatal(err)
	}

	// The hazard itself: the COLR-preferring lookup has no outline for this glyph, the raw accessor does. The mapping
	// is mapDesignPoint for size 50 at upem 1000 with an identity device matrix.
	mapPt := func(x, y float32) geom.Point { return geom.Pt(x/1000*50, -y/1000*50) }
	if p := glyphOutlinePath(tf, colorLayerGID, mapPt); p != nil {
		t.Fatalf("gid %d unexpectedly resolves an outline through GlyphData", colorLayerGID)
	}
	raw := glyphRawOutlinePath(tf, colorLayerGID, mapPt)
	if raw == nil || raw.Bounds().IsEmpty() {
		t.Fatalf("gid %d has no raw outline to fill", colorLayerGID)
	}

	g, action := smileyGlyph(t, tf, 50, nil)
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	if g.Format != MaskARGB32 {
		t.Fatalf("format %v, want ARGB32", g.Format)
	}
	// Metrics: the layer union is the retargeted layer's control box, rounded out.
	want := saturateBounds(raw.Bounds()).RoundOut()
	if g.IRect() != want {
		t.Errorf("bounds %v, want %v (layer dropped from the bounds union?)", g.IRect(), want)
	}
	// Image: every layer paints the same outline, the last one opaque black (palette index 0xFFFF).
	if g.Image32 == nil {
		t.Fatal("no image")
	}
	painted := 0
	for _, w := range g.Image32 {
		if w != 0 {
			painted++
		}
	}
	if painted == 0 {
		t.Error("no pixels painted; the layer was dropped from the mask")
	}
}

func TestColorGlyphNoOpBlurMaskFilter(t *testing.T) {
	// A blur whose CTM-adjusted sigma falls under the no-blur cutoff (1/3) reports "filter did nothing" for both the
	// bounds pass and the image pass, so the glyph keeps its ARGB32 format and unfiltered bounds — and must keep the
	// unfiltered color mask rather than rendering fully transparent.
	tf := loadColorTypeface(t, "sbix.ttf")
	blur := maskfilter.NewBlur(maskfilter.BlurNormal, 0.25, false) // device-space sigma: the CTM cannot scale it up
	if blur == nil || !maskfilter.AcceptsColorMask(blur) {
		t.Fatal("want a color-accepting blur filter")
	}
	g, action := smileyGlyph(t, tf, 64, &ScalerPaint{MaskFilter: blur})
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	if g.Format != MaskARGB32 || g.Image32 == nil {
		t.Fatalf("format %v, want unfiltered ARGB32", g.Format)
	}
	unfiltered, _ := smileyGlyph(t, tf, 64, nil)
	if g.IRect() != unfiltered.IRect() {
		t.Errorf("bounds %v changed vs unfiltered %v", g.IRect(), unfiltered.IRect())
	}
	cx, cy := int(g.Width)/2, int(g.Height)/2
	if got := g.Image32[cy*int(g.Width)+cx]; got != faceCenterWord {
		t.Errorf("center %#08x, want %#08x", got, faceCenterWord)
	}
	if !slices.Equal(g.Image32, unfiltered.Image32) {
		t.Error("mask must be the unfiltered color mask")
	}
}

func TestColorGlyphMemoryAccounting(t *testing.T) {
	cache := NewStrikeCache()
	tf := loadColorTypeface(t, "sbix.ttf")
	f := NewFont(tf, 64, 1, 0)
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(f, nil, &identity, nil)
	strike := cache.FindOrCreateStrike(&spec)
	before := cache.TotalMemoryUsed()
	g, action := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(tf.UnicharToGlyph(smiley)))
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	growth := cache.TotalMemoryUsed() - before
	wantImage := int(g.Width) * int(g.Height) * 4
	if growth < wantImage {
		t.Errorf("cache grew %d bytes; ARGB32 image alone is %d", growth, wantImage)
	}
}

func TestBitmapFontMeasure(t *testing.T) {
	// Measure runs through the extents (largest strike at ppem 0): advances are linear hmtx values.
	tf := loadColorTypeface(t, "sbix.ttf")
	f := NewFont(tf, 64, 1, 0)
	gid := tf.UnicharToGlyph(smiley)
	glyphs := []byte{byte(gid), byte(gid >> 8)}
	var bounds geom.Rect
	width := f.MeasureText(glyphs, TextEncodingGlyphID, &bounds, nil)
	if width != 51.2 {
		t.Errorf("measured width %v, want 51.2", width)
	}
	if bounds.IsEmpty() {
		t.Error("measure bounds empty")
	}
}

func TestBitmapFacePpemDoesNotLeak(t *testing.T) {
	// go-text's GlyphExtents prefers ppem-scaled bitmap-strike extents, so the bitmap lane must leave the shared Face at
	// ppem 0. If it leaked a strike ppem, the design-unit extents readers (GlyphDesignBounds, glyphBounds, letterTop)
	// would return stale, cross-strike, ppem-scaled bounds for later glyphs. sbix.ttf carries strikes at 16/64/128 ppem
	// whose scaled-to-design-unit extents differ per strike, so a leak is observable.
	tf := loadColorTypeface(t, "sbix.ttf")
	gid := tf.UnicharToGlyph(smiley)
	if gid == 0 {
		t.Fatal("U+1F600 not mapped")
	}

	// Establish the resting (ppem 0) design bounds, then read them again after exercising every strike. They must not
	// change: the readers see design units, independent of whatever ppem the bitmap lane last requested.
	want := tf.GlyphDesignBounds(gid)
	if want.IsEmpty() {
		t.Fatal("design bounds empty")
	}
	for _, ppem := range []uint16{16, 64, 128} {
		if _, _, ok := tf.faceBitmapGlyph(opentype.GID(gid), ppem); !ok {
			t.Fatalf("no strike bitmap at ppem %d", ppem)
		}
		if x, y := tf.face.Ppem(); x != 0 || y != 0 {
			t.Errorf("ppem %d leaked: Face left at (%d, %d), want (0, 0)", ppem, x, y)
		}
		if got := tf.GlyphDesignBounds(gid); got != want {
			t.Errorf("after ppem %d: GlyphDesignBounds %+v, want stable %+v", ppem, got, want)
		}
	}

	// letterTop (x-height/cap-height synthesis) reads the same extents; it too must stay ppem-independent.
	st := &strike{t: tf, size: 100, scaleX: 1, frameWidth: -1}
	baseTop, ok := st.letterTop(rune(smiley), 1)
	if !ok {
		t.Fatal("letterTop missing")
	}
	tf.faceBitmapGlyph(opentype.GID(gid), 16)
	if got, _ := st.letterTop(rune(smiley), 1); got != baseTop {
		t.Errorf("letterTop leaked after ppem 16: %v, want %v", got, baseTop)
	}
}
