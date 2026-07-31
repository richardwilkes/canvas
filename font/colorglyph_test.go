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
	"compress/zlib"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"slices"
	"testing"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/font/opentype/tables"

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

// smileyGlyph resolves U+1F600's glyph through a mask strike at the given size, preparing the image. The strike comes
// from the process-wide cache, so the returned glyph is only good for its contents: see smileyGlyphIn for the cases
// that compare glyph pointers.
func smileyGlyph(t *testing.T, tf *Typeface, size float32, paint *ScalerPaint) (*Glyph, GlyphAction) {
	t.Helper()
	return smileyGlyphIn(t, GlobalStrikeCache(), tf, size, paint)
}

// smileyGlyphIn is smileyGlyph against a caller-supplied strike cache. Every case asserting *pointer identity* of two
// glyphs has to use one of its own: the process-wide cache is shared by the whole package (this file alone leaves a few
// hundred KB in its 2 MiB budget) and is never purged, so a later mask-heavy test pushing it over budget would let
// FindOrCreateStrike's purge evict the first of two strikes between the two calls and turn a stable identity check into
// an intermittent failure of a test that has nothing to do with the eviction.
func smileyGlyphIn(t *testing.T, cache *StrikeCache, tf *Typeface, size float32,
	paint *ScalerPaint,
) (*Glyph, GlyphAction) {
	t.Helper()
	gid := tf.UnicharToGlyph(smiley)
	if gid == 0 {
		t.Fatal("U+1F600 not mapped")
	}
	f := NewFont(tf, size, 1, 0)
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(f, paint, &identity, nil)
	return cache.FindOrCreateStrike(&spec).DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
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

func TestMeasuredBoundsCoverColorAndBitmapLanes(t *testing.T) {
	// The measuring lane must report the extent the glyph actually draws: the base outline's control box misses COLR
	// layers, a COLRv1 clip box, and a bitmap strike's quad entirely, so a caller sizing a surface or an invalidation
	// rect from MeasureText/GlyphBounds would clip the emoji.
	const size = 32
	for _, file := range []string{"colr.ttf", "sbix.ttf", "cbdt.ttf", "test_glyphs-glyf_colr_1.ttf"} {
		t.Run(file, func(t *testing.T) {
			tf := loadColorTypeface(t, file)
			f := NewFont(tf, size, 1, 0)
			identity := geom.IdentityMatrix()
			spec := MakeMaskSpec(f, nil, &identity, nil)
			strike := spec.FindOrCreateStrike()

			// What the drawing scaler produces at the identity device matrix is the reference: the measuring strike has
			// the same text matrix and no post 2x2, so the two must agree exactly, on every glyph of the font.
			nonEmpty := 0
			bounds := make([]geom.Rect, 1)
			for gid := range uint16(tf.nGlyphs) {
				g, _ := strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
				drawn := g.IRect()
				want := geom.Rect{}
				if !drawn.IsEmpty() {
					nonEmpty++
					want = geom.RectLTRB(float32(drawn.Left), float32(drawn.Top), float32(drawn.Right),
						float32(drawn.Bottom))
				}
				f.GlyphBounds([]uint16{gid}, bounds)
				if bounds[0] != want {
					t.Errorf("glyph %d: GlyphBounds %v, drawn %v", gid, bounds[0], want)
				}
				// MeasureText joins the same per-glyph bounds, so it covers the drawn extent too.
				var measured geom.Rect
				f.MeasureText([]byte{uint8(gid), uint8(gid >> 8)}, TextEncodingGlyphID, &measured, nil)
				if measured != want {
					t.Errorf("glyph %d: MeasureText bounds %v, drawn %v", gid, measured, want)
				}
			}
			if nonEmpty == 0 {
				t.Fatal("no glyph in the font drew anything")
			}
		})
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

// patchSfntTable returns a copy of an sfnt font with tag's table handed to fn for in-place editing. fn must not change
// the table's length, so every other table's directory offset stays valid.
func patchSfntTable(t *testing.T, data []byte, tag string, fn func(table []byte)) []byte {
	t.Helper()
	out := bytes.Clone(data)
	for i := range int(binary.BigEndian.Uint16(out[4:])) {
		rec := 12 + i*16
		if string(out[rec:rec+4]) != tag {
			continue
		}
		off := int(binary.BigEndian.Uint32(out[rec+8:]))
		fn(out[off : off+int(binary.BigEndian.Uint32(out[rec+12:]))])
		return out
	}
	t.Fatalf("%s table not found", tag)
	return nil
}

// patchCOLRv0Ranges rewrites every COLRv0 base-glyph record in colr.ttf to claim the given layer range, leaving the
// table parseable (the parser validates array extents, never the ranges records name inside them).
func patchCOLRv0Ranges(t *testing.T, data []byte, first, num uint16) []byte {
	t.Helper()
	return patchSfntTable(t, data, "COLR", func(tab []byte) {
		if v := binary.BigEndian.Uint16(tab); v != 0 {
			t.Fatalf("colr.ttf COLR version %d, want 0", v)
		}
		base := int(binary.BigEndian.Uint32(tab[4:]))
		for i := range int(binary.BigEndian.Uint16(tab[2:])) {
			binary.BigEndian.PutUint16(tab[base+i*6+2:], first)
			binary.BigEndian.PutUint16(tab[base+i*6+4:], num)
		}
	})
}

func TestFaceColorPaintOutOfRangeCOLRv0(t *testing.T) {
	// typesetting's COLRv0 lookup returns layerRecords[FirstLayerIndex : FirstLayerIndex+NumLayers] with no bounds check
	// and in wrapping uint16 arithmetic, and its parser never validates the range, so a crafted base-glyph record panics
	// inside COLR.Search during ordinary drawing or measuring. colr.ttf carries 15 layer records; both an oversized
	// range and one whose end wraps to zero must answer "no color glyph" and leave the glyph on the outline lane.
	data, err := os.ReadFile("testdata/colr.ttf")
	if err != nil {
		t.Fatal(err)
	}
	// What "left on the outline lane" means, spelled out: the same font with no COLR table at all renders the smiley
	// from its base glyf outline, and that is exactly what each patched font below has to produce. Checking only "not
	// an ARGB32 mask" is satisfied by dropping the glyph or emitting a blank mask.
	noCOLR, err := NewTypefaceFromData(sfntWithoutTables(t, data, "COLR"), 0)
	if err != nil {
		t.Fatal(err)
	}
	outline, outlineAction := smileyGlyph(t, noCOLR, 50, nil)
	if outlineAction != GlyphActionAccept || outline.Format != MaskA8 || maskInk(outline) == 0 {
		t.Fatalf("the COLR-less font renders the smiley as (action %d, format %d, ink %d); the cases below have no "+
			"outline-lane answer to compare against", outlineAction, outline.Format, maskInk(outline))
	}
	for _, c := range []struct {
		name  string
		first uint16
		num   uint16
	}{
		{name: "past the end", first: 0, num: 1000},
		{name: "start past the end", first: 1000, num: 1},
		{name: "end wraps to zero", first: 65535, num: 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			tf, err2 := NewTypefaceFromData(patchCOLRv0Ranges(t, data, c.first, c.num), 0)
			if err2 != nil {
				t.Fatal(err2)
			}
			if tf.face.COLR == nil {
				t.Fatal("the patched COLR table no longer parses; the case proves nothing")
			}
			gid := tf.UnicharToGlyph(smiley)
			if gid == 0 {
				t.Fatal("U+1F600 not mapped")
			}
			if paint, ok := tf.faceColorPaint(opentype.GID(gid)); ok || paint != nil {
				t.Errorf("faceColorPaint = (%v, %v), want (nil, false)", paint, ok)
			}
			if layers, ok := tf.faceColorV0Layers(opentype.GID(gid)); ok || layers != nil {
				t.Errorf("faceColorV0Layers = (%v, %v), want (nil, false)", layers, ok)
			}
			// The whole draw path, which is where the panic would land on untrusted data.
			g, action := smileyGlyph(t, tf, 50, nil)
			if action != outlineAction || g.Format != outline.Format || g.IRect() != outline.IRect() {
				t.Fatalf("out-of-range COLRv0 rendered (action %d, format %d, %v), want the outline lane's "+
					"(action %d, format %d, %v)", action, g.Format, g.IRect(), outlineAction, outline.Format,
					outline.IRect())
			}
			if !bytes.Equal(g.Image, outline.Image) {
				t.Errorf("out-of-range COLRv0 mask carries %d ink, want the outline lane's %d", maskInk(g),
					maskInk(outline))
			}
		})
	}

	// The unpatched font is unaffected: no record is flagged, and the color lane still answers.
	tf := loadColorTypeface(t, "colr.ttf")
	if tf.colr0BadGlyphs != nil {
		t.Errorf("colr.ttf flagged %d base glyphs; it is well formed", len(tf.colr0BadGlyphs))
	}
	if _, ok := tf.faceColorV0Layers(opentype.GID(tf.UnicharToGlyph(smiley))); !ok {
		t.Error("the unpatched font lost its COLRv0 layers")
	}
}

func TestCOLRv0EmptyLayerListStaysOnTheOutlineLane(t *testing.T) {
	// A base-glyph record declaring NumLayers == 0 is perfectly in range (0+0 is never past numLayerRecords), so the
	// load-time scan does not flag it and typesetting hands back an empty layer slice with ok=true. Claiming the glyph
	// for the color lane on the strength of that commits it to an ARGB32 mask with neverRequestPath and an empty
	// bounding box, so it measures as nothing and draws as nothing. FreeType's FT_Get_Color_Glyph_Layer returns 0 on
	// the first call and Skia falls through to the base outline, which is what the font's own glyf table is there for.
	data := readTestFont(t, "colr.ttf")
	tf, err := NewTypefaceFromData(patchCOLRv0Ranges(t, data, 0, 0), 0)
	if err != nil {
		t.Fatal(err)
	}
	if tf.colr0BadGlyphs != nil {
		t.Fatal("an empty layer range was flagged as out of range; the case proves nothing")
	}
	gid := tf.UnicharToGlyph(smiley)
	if gid == 0 {
		t.Fatal("U+1F600 not mapped")
	}
	// typesetting still reports the glyph as a color glyph, which is the whole problem.
	if layers, ok := tf.face.COLR.Search(gid); !ok {
		t.Fatal("the patched record no longer parses; the case proves nothing")
	} else if resolved, isV0 := layers.(tables.PaintColrLayersResolved); !isV0 || len(resolved) != 0 {
		t.Fatalf("COLR.Search returned %T with %d layers, want an empty v0 layer list", layers, len(resolved))
	}
	if paint, ok := tf.faceColorPaint(opentype.GID(gid)); ok || paint != nil {
		t.Errorf("faceColorPaint = (%v, %v), want (nil, false)", paint, ok)
	}
	if layers, ok := tf.faceColorV0Layers(opentype.GID(gid)); ok || layers != nil {
		t.Errorf("faceColorV0Layers = (%v, %v), want (nil, false)", layers, ok)
	}

	// The glyph is measured and drawn as the outline it still has, byte for byte as the same font with no COLR table
	// at all draws it.
	g, action := smileyGlyph(t, tf, 50, nil)
	if action != GlyphActionAccept {
		t.Fatalf("action %v, want accept", action)
	}
	if g.Format != MaskA8 {
		t.Errorf("format %v, want A8 (the outline lane)", g.Format)
	}
	outlineOnly, err := NewTypefaceFromData(sfntWithoutTables(t, data, "COLR", "CPAL"), 0)
	if err != nil {
		t.Fatal(err)
	}
	ref, action := smileyGlyph(t, outlineOnly, 50, nil)
	if action != GlyphActionAccept {
		t.Fatalf("the COLR-less reference was not accepted: action %v", action)
	}
	if g.IRect() != ref.IRect() {
		t.Errorf("bounds %v, want the base outline's %v", g.IRect(), ref.IRect())
	}
	ink := 0
	for i, v := range g.Image {
		if i >= len(ref.Image) || v != ref.Image[i] {
			t.Fatalf("mask differs from the base outline's at byte %d", i)
		}
		ink += int(v)
	}
	if ink == 0 {
		t.Error("the glyph drew nothing")
	}
}

func TestCOLR0OutOfRangeGlyphs(t *testing.T) {
	// A version 0 header (14 bytes) with one base-glyph record at offset 14 and numLayerRecords layer records at 20.
	header := func(numLayerRecords uint16, layerRecordsOffset uint32) []byte {
		raw := binary.BigEndian.AppendUint16(nil, 0) // version
		raw = binary.BigEndian.AppendUint16(raw, 1)  // numBaseGlyphRecords
		raw = binary.BigEndian.AppendUint32(raw, 14) // baseGlyphRecordsOffset
		raw = binary.BigEndian.AppendUint32(raw, layerRecordsOffset)
		return binary.BigEndian.AppendUint16(raw, numLayerRecords)
	}
	record := func(gid, first, num uint16) []byte {
		raw := binary.BigEndian.AppendUint16(nil, gid)
		raw = binary.BigEndian.AppendUint16(raw, first)
		return binary.BigEndian.AppendUint16(raw, num)
	}
	for _, c := range []struct {
		name string
		raw  []byte
		bad  bool
	}{
		{name: "in range", raw: append(header(4, 20), record(7, 1, 3)...)},
		{name: "empty range at the end", raw: append(header(4, 20), record(7, 4, 0)...)},
		{name: "past the end", raw: append(header(1, 20), record(7, 0, 10)...), bad: true},
		{name: "end wraps to zero", raw: append(header(1, 20), record(7, 65535, 1)...), bad: true},
		{name: "null layer array", raw: append(header(4, 0), record(7, 0, 1)...), bad: true},
		{name: "no base glyph array", raw: append(binary.BigEndian.AppendUint16(nil, 0), make([]byte, 12)...)},
		{name: "truncated header", raw: make([]byte, 13)},
		{name: "truncated record", raw: append(header(4, 20), 0, 7)},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := colr0OutOfRangeGlyphs(c.raw)
			if _, flagged := got[7]; flagged != c.bad {
				t.Errorf("colr0OutOfRangeGlyphs = %v, want glyph 7 flagged = %v", got, c.bad)
			}
			if !c.bad && got != nil {
				t.Errorf("a well-formed table must cost no map: got %v", got)
			}
		})
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
	// their glyphs must differ. Both of these are pointer comparisons, so they run against a cache of their own — see
	// smileyGlyphIn.
	cache := NewStrikeCache()
	gRed, _ := smileyGlyphIn(t, cache, tf, 50, paint)
	gBlack, _ := smileyGlyphIn(t, cache, tf, 50, nil)
	if gBlack == gRed {
		t.Error("different foreground colors must resolve different strikes")
	}

	// A non-COLR face must not fragment strikes by color.
	sbix := loadColorTypeface(t, "sbix.ttf")
	g1, _ := smileyGlyphIn(t, cache, sbix, 50, paint)
	g2, _ := smileyGlyphIn(t, cache, sbix, 50, nil)
	if g1 != g2 {
		t.Error("sbix strikes must not key on the paint color")
	}
	// Non-vacuity: both strikes are still in the cache, so the identity above is the key's doing and not an eviction's.
	if got := cache.StrikeCount(); got != 3 {
		t.Errorf("the local cache holds %d strikes, want 3 (two COLR, one sbix); an eviction would decide these "+
			"comparisons instead of the strike key", got)
	}
}

// TestBitmapGlyphMetrics covers the two bitmap strike formats. Each font is its own subtest and everything after the
// action and format checks reports rather than aborts, so a regression in one format cannot hide a simultaneous one in
// the other: run as a single loop with t.Fatalf throughout, an sbix bounds regression stopped the loop before cbdt was
// ever asked, and the CBDT failure only surfaced once the sbix one was fixed.
func TestBitmapGlyphMetrics(t *testing.T) {
	for _, name := range []string{"sbix.ttf", "cbdt.ttf"} {
		t.Run(name, func(t *testing.T) {
			tf := loadColorTypeface(t, name)
			g, action := smileyGlyph(t, tf, 64, nil)
			if action != GlyphActionAccept {
				t.Fatalf("action %v", action)
			}
			if g.Format != MaskARGB32 {
				t.Fatalf("format %v, want ARGB32", g.Format)
			}
			// The 64-ppem strike is 52x52 px with extents 812.5 font units square: at size 64 the device bounds are
			// exactly [0, -52, 52, 0].
			want := geom.IRectLTRB(0, -52, 52, 0)
			if g.IRect() != want {
				t.Errorf("bounds %v, want %v", g.IRect(), want)
			}
			if g.AdvanceX != 51.2 {
				t.Errorf("advance %v, want 51.2", g.AdvanceX)
			}
			if g.Path() != nil {
				t.Error("bitmap glyph must have no path")
			}
			// Path drawing rejects color glyphs, so huge-text draws fall back to the mask stages.
			f := NewFont(tf, 64, 1, 0)
			spec, _ := MakePathSpec(f, nil)
			strike := spec.FindOrCreateStrike()
			_, pathAction := strike.DigestFor(ActionPath, PackGlyphID(tf.UnicharToGlyph(smiley)))
			if pathAction != GlyphActionReject {
				t.Errorf("path action %v, want reject", pathAction)
			}
		})
	}
}

// cblcWithImageFormat returns a copy of a CBLC/EBLC-bearing font with every index subtable's imageFormat rewritten. It
// is an in-place two-byte edit per subtable, so the whole rest of the table — the strike list, the index offsets, the
// image data — is left exactly as it was.
func cblcWithImageFormat(t *testing.T, data []byte, format uint16) []byte {
	t.Helper()
	patched := 0
	out := patchSfntTable(t, data, "CBLC", func(tab []byte) {
		for size := range int(binary.BigEndian.Uint32(tab[4:])) { // numSizes, after the 4-byte version
			record := 8 + size*48 // the bitmapSize records follow the header
			array := int(binary.BigEndian.Uint32(tab[record:]))
			for i := range int(binary.BigEndian.Uint32(tab[record+8:])) { // numberOfIndexSubTables
				entry := array + i*8
				// indexSubHeader: indexFormat, then imageFormat.
				binary.BigEndian.PutUint16(tab[array+int(binary.BigEndian.Uint32(tab[entry+4:]))+2:], format)
				patched++
			}
		}
	})
	if patched == 0 {
		t.Fatal("the font has no index subtable to patch")
	}
	return out
}

// TestUndrawableStrikeIsMeasuredAsAnOutline covers the glyphs the bitmap lane refuses. go-text answers the extents
// lookup from a strike before it ever looks at 'glyf'/'CFF ', and the Face rests at ppem 0, where chooseStrike picks
// the largest strike — so a glyph carrying any strike at all is sized by that strike no matter which lane draws it.
// Every such glyph therefore has to be measured as the outline it will actually be filled with, or as nothing when it
// has none.
func TestUndrawableStrikeIsMeasuredAsAnOutline(t *testing.T) {
	t.Run("hybrid face", func(t *testing.T) {
		// An outline font given a strike in a graphic type this lane has no decoder for (an sbix 'jpg '). Its glyph must
		// come out exactly as the same face with no strike at all draws it, rather than with its outline squeezed into
		// the strike's ink box. Roboto is the host because the sbix fonts in testdata carry only the degenerate
		// placeholder contours an sbix fallback outline is usually built from, which draw no ink to compare.
		const size = 50
		square := image.NewRGBA(image.Rect(0, 0, 40, 40))
		draw.Draw(square, square.Bounds(), image.NewUniform(color.RGBA{R: 255, G: 204, A: 255}), image.Point{},
			draw.Src)
		var jpg, pngBytes bytes.Buffer
		if err := jpeg.Encode(&jpg, square, nil); err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(&pngBytes, square); err != nil {
			t.Fatal(err)
		}
		data := readTestFont(t, "Roboto-Regular.ttf")
		outlineOnly, err := NewTypefaceFromData(data, 0)
		if err != nil {
			t.Fatal(err)
		}
		gid := outlineOnly.UnicharToGlyph('H')
		if gid == 0 {
			t.Fatal("'H' not mapped")
		}
		withStrike := func(graphic string, raw []byte) *Typeface {
			t.Helper()
			strike := synthSbixTable(outlineOnly.nGlyphs, int(gid), 64, graphic, raw)
			tf, err2 := NewTypefaceFromData(sfntWithTables(t, data, map[string][]byte{"sbix": strike}), 0)
			if err2 != nil {
				t.Fatal(err2)
			}
			if !tf.hasBitmaps {
				t.Fatalf("the synthetic %q strike was not recognized", graphic)
			}
			return tf
		}
		glyphOf := func(tf *Typeface) *Glyph {
			t.Helper()
			f := NewFont(tf, size, 1, 0)
			identity := geom.IdentityMatrix()
			spec := MakeMaskSpec(f, nil, &identity, nil)
			g, action := spec.FindOrCreateStrike().DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
			if action != GlyphActionAccept {
				t.Fatalf("action %v, want accept", action)
			}
			return g
		}

		tf := withStrike("jpg ", jpg.Bytes())
		// The strike is readable — it is only undrawable — so the extents lookup really does answer from it.
		bm, ext, ok := tf.faceBitmapGlyph(opentype.GID(gid), 64)
		if !ok || bm.Format != tsfont.JPG {
			t.Fatalf("strike glyph = (format %v, ok %v), want a readable JPG", bm.Format, ok)
		}
		if ext.Width == 0 || ext.Height == 0 {
			t.Fatal("the strike reports no extents; the case proves nothing")
		}

		g := glyphOf(tf)
		if g.Format != MaskA8 {
			t.Errorf("format %v, want A8 (the outline lane)", g.Format)
		}
		ref := glyphOf(outlineOnly)
		if g.IRect() != ref.IRect() {
			t.Errorf("bounds %v, want the outline's %v", g.IRect(), ref.IRect())
		}
		ink := 0
		for i, v := range g.Image {
			if i >= len(ref.Image) || v != ref.Image[i] {
				t.Fatalf("mask differs from the strike-less face's at byte %d", i)
			}
			ink += int(v)
		}
		if ink == 0 {
			t.Error("the glyph drew nothing")
		}
		// Non-vacuity: the same strike in a format the lane *can* draw takes the bitmap lane and is sized by the
		// strike, so the two boxes really are distinguishable.
		if drawable := glyphOf(withStrike("png ", pngBytes.Bytes())); drawable.Format != MaskARGB32 ||
			drawable.IRect() == ref.IRect() {
			t.Fatalf("the drawable strike produced a %v glyph at %v; the outline is at %v", drawable.Format,
				drawable.IRect(), ref.IRect())
		}
	})

	t.Run("bitmap-only face", func(t *testing.T) {
		// cbdt.ttf is nothing but strikes — no glyf, no CFF — and EBDT/CBDT, unlike sbix and SVG, requires no fallback
		// outline. Relabel its strikes B&W (imageFormat 2, which this lane has no decoder for) and there is nothing left
		// to draw: the glyph must measure empty and drop, not report the strike's box and fill it with a nil outline,
		// which is a correctly sized, fully transparent mask.
		data := readTestFont(t, "cbdt.ttf")
		tf, err := NewTypefaceFromData(cblcWithImageFormat(t, data, 2), 0)
		if err != nil {
			t.Fatal(err)
		}
		gid := tf.UnicharToGlyph(smiley)
		if gid == 0 {
			t.Fatal("U+1F600 not mapped")
		}
		bm, ext, ok := tf.faceBitmapGlyph(opentype.GID(gid), 64)
		if !ok || bm.Format != tsfont.BlackAndWhite {
			t.Fatalf("strike glyph = (format %v, ok %v), want a readable black-and-white strike", bm.Format, ok)
		}
		if ext.Width == 0 || ext.Height == 0 {
			t.Fatal("the strike reports no extents; the case proves nothing")
		}
		g, action := smileyGlyph(t, tf, 64, nil)
		if !g.IsEmpty() {
			t.Errorf("bounds %v, want an empty glyph: the face has no outline to fill them with", g.IRect())
		}
		if action != GlyphActionDrop {
			t.Errorf("action %v, want drop", action)
		}
		// The advance is metrics, not artwork, so it survives — a run of these still lays out.
		if g.AdvanceX == 0 {
			t.Error("the glyph lost its advance")
		}
		// The control: as shipped, the same font's strikes decode and the glyph is a 52x52 color mask.
		if shipped, shippedAction := smileyGlyph(t, loadColorTypeface(t, "cbdt.ttf"), 64, nil); shippedAction !=
			GlyphActionAccept || shipped.Format != MaskARGB32 || shipped.IsEmpty() {
			t.Fatalf("the unpatched font no longer draws its strikes: action %v, format %v, bounds %v", shippedAction,
				shipped.Format, shipped.IRect())
		}
	})
}

func TestEmbeddedBitmapsFlagDoesNotGateTheBitmapLane(t *testing.T) {
	// The typeface.go/font.go comments say embedded bitmap strikes are always used when present and that no flag gates
	// them: the embeddedBitmaps request can neither enable nor suppress the lane, and it stays out of the scaler rec.
	// Both halves are pinned here, since a comment claiming "recorded but never honored — outline fonts only" would
	// describe the opposite behavior.
	for _, name := range []string{"sbix.ttf", "cbdt.ttf"} {
		t.Run(name, func(t *testing.T) {
			tf := loadColorTypeface(t, name)
			gid := tf.UnicharToGlyph(smiley)
			if gid == 0 {
				t.Fatal("U+1F600 not mapped")
			}
			// The two lookups are compared by pointer, so they run against a cache of their own (see smileyGlyphIn).
			cache := NewStrikeCache()
			glyphs := make([]*Glyph, 2)
			for i, on := range []bool{false, true} {
				f := NewFont(tf, 64, 1, 0)
				f.SetEmbeddedBitmaps(on)
				if f.EmbeddedBitmaps() != on {
					t.Fatalf("SetEmbeddedBitmaps(%v) did not stick", on)
				}
				identity := geom.IdentityMatrix()
				spec := MakeMaskSpec(f, nil, &identity, nil)
				g, action := cache.FindOrCreateStrike(&spec).DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
				if action != GlyphActionAccept {
					t.Fatalf("embeddedBitmaps=%v: action %v", on, action)
				}
				// The strike lane ran regardless of the request: an ARGB32 mask with pixels and no path.
				if g.Format != MaskARGB32 {
					t.Errorf("embeddedBitmaps=%v: format %v, want ARGB32", on, g.Format)
				}
				if g.Image32 == nil {
					t.Errorf("embeddedBitmaps=%v: no bitmap image", on)
				}
				if g.Path() != nil {
					t.Errorf("embeddedBitmaps=%v: bitmap glyph must have no path", on)
				}
				glyphs[i] = g
			}
			// The flag is out of the rec, so both requests land on the same strike and the same glyph.
			if glyphs[0] != glyphs[1] {
				t.Error("the embeddedBitmaps request must not fragment strikes")
			}
			// Non-vacuity: the one strike is still cached, so the identity above is the rec's doing rather than an
			// eviction having handed back a freshly built strike that happens to compare equal.
			if got := cache.StrikeCount(); got != 1 {
				t.Errorf("the local cache holds %d strikes, want 1", got)
			}
		})
	}
}

func TestBitmapGlyphImageNativeSize(t *testing.T) {
	// At the strike-native size the bitmap transform is the identity, so the mask must be a byte-exact premultiplied
	// copy of the strike PNG (the FT direct-copy lane; the shader's linear filter degrades to nearest at integer
	// translates). "Byte-exact" is only meaningful against the premultiply decodePremulPNG promises — MulDiv255Round —
	// so the expectation is computed with that same round-to-nearest formula over the PNG's unpremultiplied samples.
	// The obvious color.Color.RGBA() route premultiplies in 16 bits with truncation instead and disagrees by up to 1 on
	// the antialiased edge, and tolerating that difference here would hide a swap of MulDiv255Round for a truncating
	// x*a/255 (which TestDecodePremulPNGRoundsToNearest pins directly).
	tf := loadColorTypeface(t, "sbix.ttf")
	g, action := smileyGlyph(t, tf, 64, nil)
	if action != GlyphActionAccept || g.Image32 == nil {
		t.Fatalf("no image (action %v)", action)
	}
	gid := tf.UnicharToGlyph(smiley)
	if gid == 0 {
		t.Fatal("U+1F600 not mapped")
	}
	bm, _, ok := tf.faceBitmapGlyph(opentype.GID(gid), 64)
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
	nrgba, ok := src.(*image.NRGBA)
	if !ok {
		t.Fatalf("the strike PNG decoded to %T; the comparison needs its unpremultiplied samples", src)
	}
	mismatches := 0
	for y := range b.Dy() {
		for x := range b.Dx() {
			i := nrgba.PixOffset(b.Min.X+x, b.Min.Y+y)
			a := nrgba.Pix[i+3]
			want := uint32(colorcore.MulDiv255Round(nrgba.Pix[i], a)) |
				uint32(colorcore.MulDiv255Round(nrgba.Pix[i+1], a))<<8 |
				uint32(colorcore.MulDiv255Round(nrgba.Pix[i+2], a))<<16 |
				uint32(a)<<24
			if g.Image32[y*int(g.Width)+x] != want {
				mismatches++
			}
		}
	}
	if mismatches != 0 {
		t.Errorf("%d of %d pixels are not a byte-exact premultiplied copy of the strike PNG", mismatches,
			b.Dx()*b.Dy())
	}
	// And the well-known center color.
	cx, cy := int(g.Width)/2, int(g.Height)/2
	if got := g.Image32[cy*int(g.Width)+cx]; got != faceCenterWord {
		t.Errorf("center %#08x, want %#08x", got, faceCenterWord)
	}
}

func TestDecodePremulPNGRoundsToNearest(t *testing.T) {
	// decodePremulPNG documents a round-to-nearest premultiply (MulDiv255Round). The truncating x*a/255 a hand-rolled
	// premultiply falls into differs by 1 on a large share of partially transparent samples — the antialiased edge of
	// every bitmap strike — so the direct-copy lane's "byte-exact copy of the strike PNG" claim only means something
	// with the formula pinned exactly. The sweep below covers every alpha step against varied color samples, and the
	// truncation counter proves the assertion can tell the two formulas apart.
	const side = 16
	src := image.NewNRGBA(image.Rect(0, 0, side, side))
	for y := range side {
		for x := range side {
			src.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x * 17),
				G: uint8(255 - x*17),
				B: uint8(x*13 + y),
				A: uint8(y * 17),
			})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	img := decodePremulPNG(buf.Bytes(), side, side)
	if img == nil {
		t.Fatal("the strike did not decode")
	}
	pm := img.Image.Pixmap()
	if pm == nil {
		t.Fatal("the decoded strike has no pixels")
	}
	truncationWouldDiffer := 0
	for y := range side {
		for x := range side {
			c := src.NRGBAAt(x, y)
			want := uint32(colorcore.MulDiv255Round(c.R, c.A)) | uint32(colorcore.MulDiv255Round(c.G, c.A))<<8 |
				uint32(colorcore.MulDiv255Round(c.B, c.A))<<16 | uint32(c.A)<<24
			if got := pm.Pix[y*int(pm.RowPixels)+x]; got != want {
				t.Fatalf("(%d, %d) = %#08x, want %#08x (round-to-nearest premultiply)", x, y, got, want)
			}
			trunc := (uint32(c.R) * uint32(c.A) / 255) | (uint32(c.G)*uint32(c.A)/255)<<8 |
				(uint32(c.B)*uint32(c.A)/255)<<16 | uint32(c.A)<<24
			if trunc != want {
				truncationWouldDiffer++
			}
		}
	}
	if truncationWouldDiffer == 0 {
		t.Fatal("no sample distinguishes round-to-nearest from a truncating premultiply; the test proves nothing")
	}
}

// appendPNGChunk appends one PNG chunk (typeAndData is the 4-byte type followed by the chunk data) with its length and
// CRC.
func appendPNGChunk(dst, typeAndData []byte) []byte {
	dst = binary.BigEndian.AppendUint32(dst, uint32(len(typeAndData)-4))
	dst = append(dst, typeAndData...)
	return binary.BigEndian.AppendUint32(dst, crc32.ChecksumIEEE(typeAndData))
}

// craftedPNG builds a few-hundred-byte PNG whose 8-bit RGBA IHDR claims w x h and whose IDAT holds a valid but far too
// short zlib stream. png.Decode allocates the whole w*h*4 image the moment it starts the IDAT, well before it runs out
// of row data, so this is the shape a hostile bitmap strike takes.
func craftedPNG(t *testing.T, w, h uint32) []byte {
	t.Helper()
	ihdr := append([]byte("IHDR"), make([]byte, 0, 13)...)
	ihdr = binary.BigEndian.AppendUint32(ihdr, w)
	ihdr = binary.BigEndian.AppendUint32(ihdr, h)
	ihdr = append(ihdr, 8, 6, 0, 0, 0) // bit depth, color type (RGBA), compression, filter, interlace
	var z bytes.Buffer
	zw := zlib.NewWriter(&z)
	if _, err := zw.Write(make([]byte, 128)); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	out := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'}
	out = appendPNGChunk(out, ihdr)
	out = appendPNGChunk(out, append([]byte("IDAT"), z.Bytes()...))
	return appendPNGChunk(out, []byte("IEND"))
}

func TestDecodePremulPNGValidatesStrikeDimensions(t *testing.T) {
	// png.Decode allocates the full image from the IHDR before reading a single IDAT byte, so the strike's own
	// dimensions have to gate the decode. A real 4x3 strike decodes; the same bytes under mismatched strike metrics
	// (the CBDT/EBDT hazard, where the metrics are independent of the PNG) do not; and a header claiming 65535x65535 in
	// 33 bytes must be refused without allocating.
	var buf bytes.Buffer
	src := image.NewNRGBA(image.Rect(0, 0, 4, 3))
	src.Set(1, 1, color.NRGBA{R: 255, A: 128})
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	good := buf.Bytes()
	if img := decodePremulPNG(good, 4, 3); img == nil {
		t.Fatal("a strike whose metrics match its PNG must decode")
	}
	for _, c := range []struct {
		name string
		w, h int
	}{
		{name: "width disagrees", w: 5, h: 3},
		{name: "height disagrees", w: 4, h: 4},
		{name: "zero", w: 0, h: 0},
		{name: "negative", w: -4, h: -3},
		{name: "past the mask ceiling", w: maxGlyphWidth, h: maxGlyphHeight},
	} {
		if img := decodePremulPNG(good, c.w, c.h); img != nil {
			t.Errorf("%s: strike %dx%d decoded a 4x3 PNG", c.name, c.w, c.h)
		}
	}

	// The allocation guard itself: a few-hundred-byte strike claiming a size at the mask ceiling must cost nothing.
	// Unguarded, png.Decode reaches for maxGlyphWidth*maxGlyphHeight*4 (256MB) here, and 65535x65535 would be ~17GB.
	// What makes the refusal free is that it happens in strikeDimensionsUsable, from the metrics alone — no reader, no
	// header parse, nothing allocated — so that is what is asserted, rather than a runtime.MemStats delta: MemStats
	// counts every goroutine's allocations over the interval, so a background allocation of a megabyte would fail this
	// case for reasons that have nothing to do with the strike.
	huge := craftedPNG(t, maxGlyphWidth, maxGlyphHeight)
	if len(huge) > 1024 {
		t.Fatalf("the crafted strike is %d bytes; it is meant to be tiny", len(huge))
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(huge))
	if err != nil || int32(cfg.Width) != int32(maxGlyphWidth) || int32(cfg.Height) != int32(maxGlyphHeight) {
		t.Fatalf("crafted header did not parse at the mask ceiling: %v %v", cfg, err)
	}
	if strikeDimensionsUsable(maxGlyphWidth, maxGlyphHeight) {
		t.Error("a strike at the mask ceiling passed the dimension guard, so the decode would run before the refusal")
	}
	if img := decodePremulPNG(huge, maxGlyphWidth, maxGlyphHeight); img != nil {
		t.Error("a strike at the mask ceiling must be refused")
	}
	// Every shape the cases above refuse is refused by that guard rather than further in, so none of them reaches
	// png.Decode either.
	for _, c := range []struct {
		name string
		w, h int
	}{
		{name: "zero", w: 0, h: 0},
		{name: "negative", w: -4, h: -3},
		{name: "past the mask ceiling", w: maxGlyphWidth, h: maxGlyphHeight},
	} {
		if strikeDimensionsUsable(c.w, c.h) {
			t.Errorf("%s: strike %dx%d passed the dimension guard", c.name, c.w, c.h)
		}
	}
	// And a strike whose own dimensions are perfectly ordinary is refused by the header comparison instead — the
	// ordering this lane rests on, and the one the guard above short-circuits past. png.DecodeConfig reads the IHDR and
	// allocates nothing; without it, png.Decode would allocate the 67 Mpixel image the header claims and hand back a
	// picture the strike metrics never described.
	mismatched := craftedPNG(t, maxGlyphWidth-1, maxGlyphHeight-1)
	if !strikeDimensionsUsable(4, 3) {
		t.Fatal("a 4x3 strike no longer passes the dimension guard; the header comparison is not under test")
	}
	if img := decodePremulPNG(mismatched, 4, 3); img != nil {
		t.Error("a 4x3 strike decoded a PNG whose header claims 8191x8191")
	}
}

func TestDecodePremulPNGCapsTheStrikeArea(t *testing.T) {
	// The per-side ceiling leaves an 8191x8191 header — one step below it, and 67 Mpixel — perfectly acceptable, and on
	// the sbix path nothing else stands in its way: go-text has no independent metrics to check such a strike against,
	// so it takes GlyphBitmap.Width/Height from png.DecodeConfig on these very bytes and the equality check below
	// compares the header with itself. Decoding it would cost about 768 MB across the NRGBA image, the premultiplied
	// buffer, and the imagecore copy, out of a 33-byte strike.
	huge := craftedPNG(t, maxGlyphWidth-1, maxGlyphHeight-1)
	if len(huge) > 1024 {
		t.Fatalf("the crafted strike is %d bytes; it is meant to be tiny", len(huge))
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(huge))
	if err != nil {
		t.Fatalf("crafted header did not parse: %v", err)
	}
	if cfg.Width >= maxGlyphWidth || cfg.Height >= maxGlyphHeight {
		t.Fatalf("crafted header is %dx%d, which the per-side ceiling already refuses", cfg.Width, cfg.Height)
	}
	// The sbix lane's call, verbatim: the strike dimensions are the header's own. The refusal has to come from the
	// area cap in strikeDimensionsUsable, which runs before the data is touched at all; that is what keeps it free,
	// and it is a fact about the guard rather than about how much the process happened to allocate over an interval.
	if strikeDimensionsUsable(cfg.Width, cfg.Height) {
		t.Errorf("a %dx%d strike (%d pixels) passed the dimension guard, so the decode would run before the refusal",
			cfg.Width, cfg.Height, cfg.Width*cfg.Height)
	}
	if img := decodePremulPNG(huge, cfg.Width, cfg.Height); img != nil {
		t.Error("a strike of 67 Mpixel was decoded")
	}

	// The cap is on the area, not on either side, so it is the product that decides. A strike sitting exactly on it
	// decodes...
	const capWidth, capHeight = 1 << 12, maxStrikePixels >> 12
	var atTheCap bytes.Buffer
	if err = png.Encode(&atTheCap, image.NewNRGBA(image.Rect(0, 0, capWidth, capHeight))); err != nil {
		t.Fatal(err)
	}
	if img := decodePremulPNG(atTheCap.Bytes(), capWidth, capHeight); img == nil {
		t.Errorf("a %dx%d strike, exactly at the %d-pixel cap, was refused", capWidth, capHeight, maxStrikePixels)
	}
	// ...and one pixel-row past it does not, whatever shape it takes, and again from the guard alone.
	for _, c := range []struct {
		name string
		w, h int
	}{
		{name: "one row over", w: capWidth, h: capHeight + 1},
		{name: "wide and short", w: maxGlyphWidth - 1, h: maxStrikePixels/(maxGlyphWidth-1) + 1},
		{name: "tall and narrow", w: maxStrikePixels/(maxGlyphHeight-1) + 1, h: maxGlyphHeight - 1},
	} {
		crafted := craftedPNG(t, uint32(c.w), uint32(c.h))
		if strikeDimensionsUsable(c.w, c.h) {
			t.Errorf("%s: %dx%d (%d pixels) passed the dimension guard", c.name, c.w, c.h, c.w*c.h)
		}
		if img := decodePremulPNG(crafted, c.w, c.h); img != nil {
			t.Errorf("%s: %dx%d (%d pixels) decoded", c.name, c.w, c.h, c.w*c.h)
		}
	}

	// The cap is far above any strike a shipping color font carries, so the real fonts are unaffected: their glyphs
	// still decode and draw.
	for _, name := range []string{"sbix.ttf", "cbdt.ttf"} {
		g, action := smileyGlyph(t, loadColorTypeface(t, name), 128, nil)
		if action != GlyphActionAccept || g.Image32 == nil {
			t.Errorf("%s: the largest shipping strike no longer decodes (action %v)", name, action)
		}
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

	// The hazard itself: go-text's preference-ordered lookup answers with this glyph's COLR entry rather than its
	// outline, so glyphOutlinePath must go through the raw accessor to find anything to fill. The mapping is
	// mapDesignPoint for size 50 at upem 1000 with an identity device matrix.
	if _, isOutline := tf.face.GlyphData(opentype.GID(colorLayerGID)).(tsfont.GlyphOutline); isOutline {
		t.Fatalf("gid %d resolves an outline through GlyphData; it no longer exercises the preference order",
			colorLayerGID)
	}
	mapPt := func(x, y float32) geom.Point { return geom.Pt(x/1000*50, -y/1000*50) }
	raw := glyphOutlinePath(tf, colorLayerGID, mapPt)
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

// TestBitmapLaneKeepsItsOwnFace covers the split between the bitmap-strike lane's Face and the design-unit one. The
// lane is the only reader that wants a nonzero ppem, and typesetting's SetPpem throws away the Face's entire per-glyph
// extents cache; borrowing the shared Face meant setting the ppem and putting it back around every glyph, so each glyph
// paid two full nGlyphs-entry cache clears and then looked into a cache that was guaranteed empty. With a Face of its
// own the lane simply rests at the strike's ppem, so the cache survives from glyph to glyph — while the design-unit
// readers keep seeing a Face that never left ppem 0.
func TestBitmapLaneKeepsItsOwnFace(t *testing.T) {
	tf := loadColorTypeface(t, "sbix.ttf")
	gid := opentype.GID(tf.UnicharToGlyph(smiley))
	if gid == 0 {
		t.Fatal("U+1F600 not mapped")
	}

	// Warm the design-unit cache first: it must still be warm after the bitmap lane has run.
	if tf.GlyphDesignBounds(uint16(gid)).IsEmpty() {
		t.Fatal("design bounds empty")
	}
	const ppem = 64
	if _, _, ok := tf.faceBitmapGlyph(gid, ppem); !ok {
		t.Fatalf("no strike bitmap at ppem %d", ppem)
	}
	if tf.bitmapFace == nil {
		t.Fatal("the bitmap lane did not take a Face of its own")
	}
	if tf.bitmapFace == tf.face {
		t.Fatal("the bitmap lane is sharing the design-unit Face")
	}
	if x, y := tf.bitmapFace.Ppem(); x != ppem || y != ppem {
		t.Errorf("the lane's Face rests at ppem (%d, %d), want (%d, %d) so its extents cache survives the call",
			x, y, ppem, ppem)
	}
	if x, y := tf.face.Ppem(); x != 0 || y != 0 {
		t.Errorf("the design-unit Face was left at ppem (%d, %d), want (0, 0)", x, y)
	}

	// An extents cache hit returns the stored value; a miss re-derives it, which for an sbix strike means decoding the
	// PNG header and therefore allocating. Zero allocations is the cache being consulted rather than reset.
	var extents tsfont.GlyphExtents
	if allocs := testing.AllocsPerRun(20, func() { extents, _ = tf.bitmapFace.GlyphExtents(gid) }); allocs != 0 {
		t.Errorf("re-reading the lane's extents allocated %v times per call, want 0 (the cache is being reset)", allocs)
	}
	if extents == (tsfont.GlyphExtents{}) {
		t.Error("the lane reported empty extents")
	}
	// The same for the design-unit face: the bitmap lane must not have invalidated what was cached there either.
	if allocs := testing.AllocsPerRun(20, func() { extents, _ = tf.face.GlyphExtents(gid) }); allocs != 0 {
		t.Errorf("re-reading the design-unit extents allocated %v times per call, want 0", allocs)
	}

	// Another glyph at the same ppem must not disturb the ppem the lane rests at, and switching ppem must move only the
	// lane's Face.
	if _, _, ok := tf.faceBitmapGlyph(gid, 16); !ok {
		t.Fatal("no strike bitmap at ppem 16")
	}
	if x, y := tf.bitmapFace.Ppem(); x != 16 || y != 16 {
		t.Errorf("after a ppem 16 glyph the lane's Face is at (%d, %d), want (16, 16)", x, y)
	}
	if x, y := tf.face.Ppem(); x != 0 || y != 0 {
		t.Errorf("the ppem 16 glyph moved the design-unit Face to (%d, %d), want (0, 0)", x, y)
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
