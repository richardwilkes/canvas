// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the COLRv1 paint-graph interpreter (colorglyph_v1.go) over the googlefonts/color-fonts "COLRv1 Static Test
// Glyphs" conformance font (testdata/test_glyphs-glyf_colr_1.ttf). The codepoint groups come from that font's own
// generator: U+F0100… gradient stops, U+F0200… sweeps, U+F03/06/07/08/09xx transforms, U+F0500… extend modes, U+F0A00…
// composite modes, U+F0B00… foreground color, U+F0C00… clip boxes, U+F11xx/F12xx paint-graph cycles, U+F1400… nested
// PaintGlyphs. The font's upem is 1000 and the test artwork lives in the 0..1000 em square, so size 50 maps font units
// to device pixels at 1/20.

package font

import (
	"encoding/binary"
	"testing"

	"github.com/go-text/typesetting/font/opentype/tables"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

func loadCOLRv1Typeface(t *testing.T) *Typeface {
	t.Helper()
	return loadColorTypeface(t, "test_glyphs-glyf_colr_1.ttf")
}

// colrV1Glyph resolves ch's glyph through a mask strike at the given size, preparing the image.
func colrV1Glyph(t *testing.T, tf *Typeface, ch rune, size float32, paint *ScalerPaint) (*Glyph, GlyphAction) {
	t.Helper()
	gid := tf.UnicharToGlyph(ch)
	if gid == 0 {
		t.Fatalf("U+%X not mapped", ch)
	}
	f := NewFont(tf, size, 1, 0)
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(f, paint, &identity, nil)
	strike := spec.FindOrCreateStrike()
	return strike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
}

// mustRenderCOLRv1 renders ch and fails the test unless the strike accepted it and produced an ARGB32 image.
func mustRenderCOLRv1(t *testing.T, tf *Typeface, ch rune, paint *ScalerPaint) *Glyph {
	t.Helper()
	g, action := colrV1Glyph(t, tf, ch, 50, paint)
	if action != GlyphActionAccept {
		t.Fatalf("U+%X: action %v", ch, action)
	}
	if g.Format != MaskARGB32 || g.Image32 == nil {
		t.Fatalf("U+%X: format %v (image32 nil=%v), want ARGB32", ch, g.Format, g.Image32 == nil)
	}
	return g
}

// glyphInk summarizes a rendered mask: non-transparent pixel count, distinct device words, and the ink box in glyph
// coordinates (offset by Left/Top).
type glyphInk struct {
	distinct map[uint32]int
	count    int
	box      geom.IRect
}

func analyzeGlyphInk(g *Glyph) glyphInk {
	ink := glyphInk{distinct: map[uint32]int{}}
	first := true
	for y := int32(0); y < g.Height; y++ {
		for x := int32(0); x < g.Width; x++ {
			w := g.Image32[y*g.Width+x]
			if w == 0 {
				continue
			}
			ink.count++
			ink.distinct[w]++
			px := geom.IRectLTRB(x+g.Left, y+g.Top, x+g.Left+1, y+g.Top+1)
			if first {
				ink.box = px
				first = false
			} else {
				ink.box.Join(px)
			}
		}
	}
	return ink
}

// imagesEqual reports whether two glyphs rendered identical masks (same bounds, same words).
func imagesEqual(a, b *Glyph) bool {
	if a.IRect() != b.IRect() || len(a.Image32) != len(b.Image32) {
		return false
	}
	for i, w := range a.Image32 {
		if b.Image32[i] != w {
			return false
		}
	}
	return true
}

func TestCOLRv1NeedsCurrentColor(t *testing.T) {
	// The COLR table is present, so the foreground color must enter the scaler rec (the U+F0B00… group paints with
	// palette index 0xFFFF).
	if !loadCOLRv1Typeface(t).GlyphMaskNeedsCurrentColor() {
		t.Error("COLRv1 font must need the current color")
	}
}

func TestCOLRv1GlyphMetrics(t *testing.T) {
	tf := loadCOLRv1Typeface(t)

	// U+F0100 has a ClipList box (100, 250)..(900, 950) font units: at size 50 the device bounds are [5, 45] x [-47.5,
	// -12.5], rounded out. Advance is the linear hmtx 1000/1000*50.
	g, action := colrV1Glyph(t, tf, 0xf0100, 50, nil)
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	if g.Format != MaskARGB32 {
		t.Fatalf("format %v, want ARGB32", g.Format)
	}
	if want := geom.IRectLTRB(5, -48, 45, -12); g.IRect() != want {
		t.Errorf("clip-box bounds %v, want %v", g.IRect(), want)
	}
	if g.AdvanceX != 50 {
		t.Errorf("advance %v, want 50", g.AdvanceX)
	}
	if g.Path() != nil {
		t.Error("color glyph must have no path (neverRequestPath)")
	}

	// U+F0300 has no clip box: bounds come from the measuring traversal (computeColrV1GlyphBoundingBox), here a
	// composite of a rectangle with its scaled-around-center twin.
	g, action = colrV1Glyph(t, tf, 0xf0300, 50, nil)
	if action != GlyphActionAccept {
		t.Fatalf("action %v", action)
	}
	if want := geom.IRectLTRB(12, -44, 38, -6); g.IRect() != want {
		t.Errorf("traversal bounds %v, want %v", g.IRect(), want)
	}
}

func TestCOLRv1SolidFill(t *testing.T) {
	tf := loadCOLRv1Typeface(t)

	// U+F0B06: PaintGlyph over PaintSolid with the foreground palette index (0xFFFF) at alpha 1.0. With the default
	// black foreground the interior is opaque black premul.
	g := mustRenderCOLRv1(t, tf, 0xf0b06, nil)
	ink := analyzeGlyphInk(g)
	if ink.distinct[0xFF000000] == 0 {
		t.Errorf("no opaque black interior pixels: %v", ink.distinct)
	}

	// The same glyph with a red foreground paints premul red (device word R in the low byte).
	red := &ScalerPaint{Color: colorcore.ARGB(0xFF, 0xFF, 0, 0)}
	g = mustRenderCOLRv1(t, tf, 0xf0b06, red)
	ink = analyzeGlyphInk(g)
	if ink.distinct[0xFF0000FF] == 0 {
		t.Errorf("no opaque red interior pixels with red foreground: %v", ink.distinct)
	}

	// U+F0B07 scales the solid's alpha by 0.3 (F2Dot14 ≈ 0.29999 → 76/255): the interior word is the premultiplied
	// translucent foreground.
	g = mustRenderCOLRv1(t, tf, 0xf0b07, nil)
	ink = analyzeGlyphInk(g)
	if ink.distinct[0x4C000000] == 0 {
		t.Errorf("no alpha-scaled black interior pixels: %v", ink.distinct)
	}
	g = mustRenderCOLRv1(t, tf, 0xf0b07, red)
	ink = analyzeGlyphInk(g)
	if ink.distinct[0x4C00004C] == 0 {
		t.Errorf("no alpha-scaled premul red interior pixels: %v", ink.distinct)
	}
}

// gradientGroup asserts each glyph fills wantInk pixels with a real color ramp (≥ 10 distinct words) and that the
// variants (different extend modes / geometry) render pairwise-different masks.
func gradientGroup(t *testing.T, tf *Typeface, chs []rune, wantInk int) {
	t.Helper()
	glyphs := make([]*Glyph, len(chs))
	for i, ch := range chs {
		g := mustRenderCOLRv1(t, tf, ch, nil)
		ink := analyzeGlyphInk(g)
		if wantInk > 0 && ink.count != wantInk {
			t.Errorf("U+%X: ink %d, want %d", ch, ink.count, wantInk)
		}
		if wantInk == 0 && ink.count == 0 {
			t.Errorf("U+%X: no ink", ch)
		}
		if len(ink.distinct) < 10 {
			t.Errorf("U+%X: only %d distinct colors; gradient expected", ch, len(ink.distinct))
		}
		glyphs[i] = g
	}
	for i := range glyphs {
		for j := i + 1; j < len(glyphs); j++ {
			if imagesEqual(glyphs[i], glyphs[j]) {
				t.Errorf("U+%X and U+%X rendered identically", chs[i], chs[j])
			}
		}
	}
}

func TestCOLRv1LinearGradient(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// The extend-mode group's linear gradients (pad/repeat/reflect over the full 0..1000 square: 2500 device pixels at
	// size 50).
	gradientGroup(t, tf, []rune{0xf0500, 0xf0501, 0xf0502}, 2500)
	// The gradient-stops-repeat group (clipped to the ClipList box) and the skewed-P2 normal-form projection.
	gradientGroup(t, tf, []rune{0xf0100, 0xf0101}, 0)
	if g := mustRenderCOLRv1(t, tf, 0xf0d00, nil); len(analyzeGlyphInk(g).distinct) < 10 {
		t.Error("U+F0D00 (gradient_p2_skewed): gradient expected")
	}
}

func TestCOLRv1RadialGradient(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// Full-square radials (pad/repeat/reflect), then the variants over a partial-coverage outline.
	gradientGroup(t, tf, []rune{0xf0503, 0xf0504, 0xf0505}, 2500)
	gradientGroup(t, tf, []rune{0xf0506, 0xf0507, 0xf0508}, 0)
}

func TestCOLRv1SweepGradient(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// A static sweep from the sweep_varsweep group, and the foreground-color sweeps (palette index 0xFFFF stops mixed
	// with palette stops).
	gradientGroup(t, tf, []rune{0xf0200, 0xf0b04, 0xf0b05}, 0)
}

func TestCOLRv1Transforms(t *testing.T) {
	tf := loadCOLRv1Typeface(t)

	// Every transform group renders ink, and parameter variants differ from each other.
	for _, group := range [][]rune{
		{0xf0300, 0xf0301, 0xf0302, 0xf0303, 0xf0304, 0xf0305}, // scale (+ around-center/uniform variants)
		{0xf0600, 0xf0601, 0xf0602, 0xf0603},                   // rotate (+ around-center)
		{0xf0700, 0xf0701, 0xf0702, 0xf0703, 0xf0704, 0xf0705}, // skew (+ around-center)
		{0xf0800, 0xf0801, 0xf0802, 0xf0803},                   // affine 2x3
	} {
		glyphs := make([]*Glyph, len(group))
		for i, ch := range group {
			g := mustRenderCOLRv1(t, tf, ch, nil)
			if ink := analyzeGlyphInk(g); ink.count == 0 {
				t.Errorf("U+%X: no ink", ch)
			}
			glyphs[i] = g
		}
		for i := range glyphs {
			for j := i + 1; j < len(glyphs); j++ {
				if imagesEqual(glyphs[i], glyphs[j]) {
					t.Errorf("U+%X and U+%X rendered identically", group[i], group[j])
				}
			}
		}
	}

	// PaintTranslate's y-down conjugation, pinned by ink boxes: the base composite at U+F0900 (a rectangle over its
	// untranslated twin), then dy +100 font units moving ink up 5px in device space, dy -100 down, dx +100 right, dx
	// -100 left, (+200, +200) up-right, (-200, -200) down-left.
	wantBoxes := map[rune]geom.IRect{
		0xf0900: geom.IRectLTRB(12, -38, 38, -12),
		0xf0901: geom.IRectLTRB(12, -43, 38, -12),
		0xf0902: geom.IRectLTRB(12, -38, 38, -7),
		0xf0903: geom.IRectLTRB(12, -38, 43, -12),
		0xf0904: geom.IRectLTRB(7, -38, 38, -12),
		0xf0905: geom.IRectLTRB(12, -48, 48, -12),
		0xf0906: geom.IRectLTRB(2, -38, 38, -2),
	}
	for ch, want := range wantBoxes {
		g := mustRenderCOLRv1(t, tf, ch, nil)
		if ink := analyzeGlyphInk(g); ink.box != want {
			t.Errorf("U+%X: ink box %v, want %v", ch, ink.box, want)
		}
	}
}

func TestCOLRv1CompositeModes(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// The 28 composite-mode glyphs (U+F0A00..U+F0A1B, one per CompositeMode in enum order) composite the same two
	// shapes; every mode must render through the saveLayer machinery, and the modes must actually differ (a few pairs
	// legitimately coincide on this artwork).
	type key struct {
		pix    string
		bounds geom.IRect
	}
	distinct := map[key]bool{}
	withInk := 0
	for ch := rune(0xf0a00); ch <= 0xf0a1b; ch++ {
		g := mustRenderCOLRv1(t, tf, ch, nil)
		buf := make([]byte, len(g.Image32)*4)
		for i, w := range g.Image32 {
			buf[i*4] = byte(w)
			buf[i*4+1] = byte(w >> 8)
			buf[i*4+2] = byte(w >> 16)
			buf[i*4+3] = byte(w >> 24)
		}
		distinct[key{bounds: g.IRect(), pix: string(buf)}] = true
		if analyzeGlyphInk(g).count > 0 {
			withInk++
		}
	}
	if len(distinct) < 25 {
		t.Errorf("only %d distinct composite renders out of 28", len(distinct))
	}
	if withInk < 25 {
		t.Errorf("only %d composite renders produced ink", withInk)
	}
	// SRC_OVER (enum value 3 → U+F0A03) is the workhorse mode: pin its ink count.
	if ink := analyzeGlyphInk(mustRenderCOLRv1(t, tf, 0xf0a03, nil)); ink.count != 1028 {
		t.Errorf("SRC_OVER composite ink %d, want 1028", ink.count)
	}
}

func TestCOLRv1BlendModeMapping(t *testing.T) {
	// ToSkBlendMode's table, all 28 modes.
	want := map[tables.CompositeMode]raster.BlendMode{
		tables.CompositeClear:         raster.BlendClear,
		tables.CompositeSrc:           raster.BlendSrc,
		tables.CompositeDest:          raster.BlendDst,
		tables.CompositeSrcOver:       raster.BlendSrcOver,
		tables.CompositeDestOver:      raster.BlendDstOver,
		tables.CompositeSrcIn:         raster.BlendSrcIn,
		tables.CompositeDestIn:        raster.BlendDstIn,
		tables.CompositeSrcOut:        raster.BlendSrcOut,
		tables.CompositeDestOut:       raster.BlendDstOut,
		tables.CompositeSrcAtop:       raster.BlendSrcATop,
		tables.CompositeDestAtop:      raster.BlendDstATop,
		tables.CompositeXor:           raster.BlendXor,
		tables.CompositePlus:          raster.BlendPlus,
		tables.CompositeScreen:        raster.BlendScreen,
		tables.CompositeOverlay:       raster.BlendOverlay,
		tables.CompositeDarken:        raster.BlendDarken,
		tables.CompositeLighten:       raster.BlendLighten,
		tables.CompositeColorDodge:    raster.BlendColorDodge,
		tables.CompositeColorBurn:     raster.BlendColorBurn,
		tables.CompositeHardLight:     raster.BlendHardLight,
		tables.CompositeSoftLight:     raster.BlendSoftLight,
		tables.CompositeDifference:    raster.BlendDifference,
		tables.CompositeExclusion:     raster.BlendExclusion,
		tables.CompositeMultiply:      raster.BlendMultiply,
		tables.CompositeHslHue:        raster.BlendHue,
		tables.CompositeHslSaturation: raster.BlendSaturation,
		tables.CompositeHslColor:      raster.BlendColor,
		tables.CompositeHslLuminosity: raster.BlendLuminosity,
	}
	for mode, blend := range want {
		if got := colrBlendMode(mode); got != blend {
			t.Errorf("mode %d: blend %d, want %d", mode, got, blend)
		}
	}
	if got := colrBlendMode(tables.CompositeMode(0xFF)); got != raster.BlendDst {
		t.Errorf("unknown mode: blend %d, want kDst", got)
	}
}

func TestCOLRv1ClipBox(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// U+F0C00..03 clip the same full-square artwork to the four em-square quadrants; U+F0C04's box is the centered
	// half-size square. Both the metrics (bounds from the ClipList) and the render (clip-to-box in the walker) must
	// honor it: each quadrant fills its 25x25 device box completely.
	want := map[rune]geom.IRect{
		0xf0c00: geom.IRectLTRB(0, -50, 25, -25),
		0xf0c01: geom.IRectLTRB(0, -25, 25, 0),
		0xf0c02: geom.IRectLTRB(25, -25, 50, 0),
		0xf0c03: geom.IRectLTRB(25, -50, 50, -25),
		0xf0c04: geom.IRectLTRB(12, -38, 38, -12),
	}
	for ch, box := range want {
		g := mustRenderCOLRv1(t, tf, ch, nil)
		if g.IRect() != box {
			t.Errorf("U+%X: bounds %v, want %v", ch, g.IRect(), box)
		}
		ink := analyzeGlyphInk(g)
		if ch != 0xf0c04 && ink.count != 625 {
			t.Errorf("U+%X: ink %d, want the full 625-pixel quadrant", ch, ink.count)
		}
		if ink.box != box {
			t.Errorf("U+%X: ink box %v escapes the clip box %v", ch, ink.box, box)
		}
	}
}

func TestCOLRv1CycleGuard(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// U+F1100/U+F1101 are the spec's PaintColrGlyph cycles (5.7.11.1.9): each references the other. Their ClipList
	// boxes give them non-empty metrics, and the render must terminate producing no ink — the visited-set guard skips
	// the re-entered subtree.
	for _, ch := range []rune{0xf1100, 0xf1101} {
		g := mustRenderCOLRv1(t, tf, ch, nil)
		if ink := analyzeGlyphInk(g); ink.count != 0 {
			t.Errorf("U+%X: cycle rendered %d ink pixels, want none", ch, ink.count)
		}
	}
	// U+F1200 is the group's acyclic control: deep PaintColrGlyph nesting without a cycle must render.
	if ink := analyzeGlyphInk(mustRenderCOLRv1(t, tf, 0xf1200, nil)); ink.count == 0 {
		t.Error("U+F1200: acyclic PaintColrGlyph nesting rendered no ink")
	}
}

func TestCOLRv1NestedPaintGlyph(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// The paint_glyph_nested group: PaintGlyph clip chains under transforms.
	for _, ch := range []rune{0xf1400, 0xf1401, 0xf1402, 0xf1403} {
		if ink := analyzeGlyphInk(mustRenderCOLRv1(t, tf, ch, nil)); ink.count == 0 {
			t.Errorf("U+%X: no ink", ch)
		}
	}
}

func TestCOLRv1CompositeLayerBudget(t *testing.T) {
	// Every PaintComposite pushes two full-mask ARGB layers that stay live while its source subtree is walked, so the
	// depth cap alone would allow 2*colrV1MaxDepth of them at once over a mask that may itself approach
	// maxGlyphWidth x maxGlyphHeight. saveLayer must refuse once the budget is spent, leave the stacks untouched when
	// it does (so the caller's restoreToCount stays balanced), and hand the bytes back when a layer pops.
	const side = 64
	cv := newColrV1Canvas(raster.NewPixmap(side, side), side, side, geom.IdentityMatrix())
	if cv.layerBudget != colrV1MaxLayerBytes {
		t.Errorf("fresh canvas budget %d, want %d", cv.layerBudget, int64(colrV1MaxLayerBytes))
	}
	cv.layerBudget = 2 * cv.layerBytes() // room for exactly two layers
	for i := range 2 {
		if !cv.saveLayer(raster.BlendSrcOver) {
			t.Fatalf("saveLayer %d was refused inside the budget", i)
		}
	}
	saves, devs, modes := cv.saveCount(), len(cv.devs), len(cv.modes)
	if cv.saveLayer(raster.BlendSrcOver) {
		t.Fatal("saveLayer past the budget must be refused")
	}
	if cv.saveCount() != saves || len(cv.devs) != devs || len(cv.modes) != modes {
		t.Errorf("a refused saveLayer disturbed the stacks: saves %d->%d devs %d->%d modes %d->%d",
			saves, cv.saveCount(), devs, len(cv.devs), modes, len(cv.modes))
	}
	cv.restore()
	if !cv.saveLayer(raster.BlendSrcOver) {
		t.Error("popping a layer must return its bytes to the budget")
	}

	// The shipped budget is well clear of what a real color glyph needs: a mask at the atlas ceiling can still push a
	// full depth cap's worth of composites (two layers each).
	const big = SideTooBigForAtlas
	ordinary := newColrV1Canvas(raster.NewPixmap(big, big), big, big, geom.IdentityMatrix())
	for i := range 2 * colrV1MaxDepth {
		if !ordinary.saveLayer(raster.BlendSrcOver) {
			t.Fatalf("layer %d of a %dx%d mask was refused; the budget is too tight for ordinary glyphs", i, big, big)
		}
	}
}

func TestCOLRv1CompositeBudgetFailsTheSubtree(t *testing.T) {
	// End to end through the walker: a composite chain whose layers the budget cannot cover fails the traversal the
	// same way the depth cap does, and unwinds to a balanced canvas. PaintSolid with palette index 0xFFFF resolves to
	// the (zero) foreground color without touching the face, so no real typeface is needed.
	build := func(nesting int) tables.PaintTable {
		solid := tables.PaintSolid{PaletteIndex: 0xFFFF, Alpha: 1 << 14}
		var p tables.PaintTable = solid
		for range nesting {
			p = tables.PaintComposite{BackdropPaint: solid, SourcePaint: p}
		}
		return p
	}
	const side = 32
	for _, c := range []struct {
		nesting int
		want    bool
	}{{nesting: 2, want: true}, {nesting: 3, want: false}} {
		cv := newColrV1Canvas(raster.NewPixmap(side, side), side, side, geom.IdentityMatrix())
		cv.layerBudget = 5 * cv.layerBytes() // enough for two nested composites, one layer short of three
		w := &colrV1Walker{c: &ScalerContext{}, t: &Typeface{}, canvas: cv, visited: map[uint16]bool{}}
		if got := w.traverse(build(c.nesting)); got != c.want {
			t.Errorf("%d nested composites: traverse = %v, want %v", c.nesting, got, c.want)
		}
		cv.restoreToCount(0)
		if len(cv.devs) != 1 || len(cv.modes) != 0 {
			t.Errorf("%d nested composites: unwound to %d devices / %d modes, want 1/0",
				c.nesting, len(cv.devs), len(cv.modes))
		}
		if cv.layerBudget != 5*cv.layerBytes() {
			t.Errorf("%d nested composites: budget %d after unwinding, want %d",
				c.nesting, cv.layerBudget, 5*cv.layerBytes())
		}
	}
}

// synthCOLRv1LayerList builds the smallest parseable COLRv1 table with a LayerList: an empty v0 section, a one-entry
// LayerList, and a PaintSolid painting the (zero) foreground color. Everything the walker needs to resolve a
// PaintColrLayers node is present, and nothing in it touches a face.
func synthCOLRv1LayerList(t *testing.T) *tables.COLR1 {
	t.Helper()
	const layerListOffset = 34                   // the v0 header (14) plus the v1 offsets (20)
	raw := binary.BigEndian.AppendUint16(nil, 1) // version
	raw = binary.BigEndian.AppendUint16(raw, 0)  // numBaseGlyphRecords
	raw = binary.BigEndian.AppendUint32(raw, 0)  // baseGlyphRecordsOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, 0)  // layerRecordsOffset (null)
	raw = binary.BigEndian.AppendUint16(raw, 0)  // numLayerRecords
	raw = binary.BigEndian.AppendUint32(raw, 0)  // baseGlyphListOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, layerListOffset)
	raw = binary.BigEndian.AppendUint32(raw, 0)      // clipListOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, 0)      // varIndexMapOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, 0)      // itemVariationStoreOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, 1)      // LayerList: numLayers
	raw = binary.BigEndian.AppendUint32(raw, 8)      // LayerList: paint offset, from the LayerList start
	raw = append(raw, 2)                             // PaintSolid: format
	raw = binary.BigEndian.AppendUint16(raw, 0xFFFF) // PaintSolid: palette index (foreground)
	raw = binary.BigEndian.AppendUint16(raw, 1<<14)  // PaintSolid: alpha 1.0
	colr, err := tables.ParseCOLR(raw)
	if err != nil {
		t.Fatalf("the synthetic COLR table does not parse: %v", err)
	}
	return &colr
}

func TestCOLRv1LayerListRangeGuard(t *testing.T) {
	// tables.LayerList.Resolve computes FirstLayerIndex+NumLayers in wrapping uint32 arithmetic before its own bounds
	// check, so an index near 2^32 wraps the sum below the check and panics in the slice expression it then evaluates —
	// out of a font that parses cleanly, at glyph metrics/render time. Every range the layer list cannot satisfy must
	// instead fail the subtree, the same way the depth cap does.
	colr := synthCOLRv1LayerList(t)
	for _, c := range []struct {
		name  string
		paint tables.PaintColrLayers
		want  bool
	}{
		{name: "in range", paint: tables.PaintColrLayers{FirstLayerIndex: 0, NumLayers: 1}, want: true},
		{name: "empty range at the end", paint: tables.PaintColrLayers{FirstLayerIndex: 1, NumLayers: 0}, want: true},
		{name: "past the end", paint: tables.PaintColrLayers{FirstLayerIndex: 0, NumLayers: 2}},
		{name: "start past the end", paint: tables.PaintColrLayers{FirstLayerIndex: 4, NumLayers: 1}},
		{name: "end wraps to a valid index", paint: tables.PaintColrLayers{FirstLayerIndex: 0xFFFFFFFF, NumLayers: 2}},
		{name: "end wraps to zero", paint: tables.PaintColrLayers{FirstLayerIndex: 0xFFFFFFFF, NumLayers: 1}},
		{name: "index past the signed range", paint: tables.PaintColrLayers{FirstLayerIndex: 1 << 31, NumLayers: 0}},
	} {
		t.Run(c.name, func(t *testing.T) {
			// Bounds mode: PaintSolid contributes no bounds, so success is purely "the range resolved".
			var bounds geom.Rect
			w := &colrV1Walker{colr: colr, bounds: &bounds, visited: map[uint16]bool{}}
			w.ctm.SetIdentity()
			if got := w.traverse(c.paint); got != c.want {
				t.Errorf("bounds traverse = %v, want %v", got, c.want)
			}
			// Draw mode: the same guard on the other dispatcher. PaintSolid with palette index 0xFFFF resolves without
			// touching the face, so no real typeface is needed.
			const side = 8
			cv := newColrV1Canvas(raster.NewPixmap(side, side), side, side, geom.IdentityMatrix())
			w = &colrV1Walker{c: &ScalerContext{}, t: &Typeface{}, colr: colr, canvas: cv, visited: map[uint16]bool{}}
			if got := w.traverse(c.paint); got != c.want {
				t.Errorf("draw traverse = %v, want %v", got, c.want)
			}
		})
	}
}

func TestCOLRv1DepthGuard(t *testing.T) {
	// A synthetic non-cyclic chain deeper than colrV1MaxDepth must fail the traversal (the walker cannot recurse
	// without bound); a shallow chain succeeds. Bounds mode exercises the shared guard without needing a scaler
	// context: PaintTranslate/PaintSolid touch neither the face nor the palette.
	build := func(depth int) tables.PaintTable {
		var p tables.PaintTable = tables.PaintSolid{PaletteIndex: 0xFFFF, Alpha: 1 << 14}
		for range depth {
			p = tables.PaintTranslate{Paint: p}
		}
		return p
	}
	var bounds geom.Rect
	w := &colrV1Walker{bounds: &bounds, visited: map[uint16]bool{}}
	w.ctm.SetIdentity()
	if w.traverse(build(colrV1MaxDepth + 1)) {
		t.Error("traversal deeper than colrV1MaxDepth must fail")
	}
	if !w.traverse(build(8)) {
		t.Error("shallow traversal must succeed")
	}
	if w.depth != 0 {
		t.Errorf("depth counter leaked: %d", w.depth)
	}
}
