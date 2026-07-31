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
	"fmt"
	"slices"
	"testing"

	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/font/opentype/tables"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
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

// colrV1GradientPaintBuilder is one gradient paint format reduced to a call taking just the color line.
type colrV1GradientPaintBuilder struct {
	build func(stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool)
	name  string
}

// colrV1GradientPaintBuilders is the set of gradient paint formats, each reduced to a call taking just the color line,
// with geometry chosen to reach the shader (non-degenerate for linear, a real circle pair for radial, a 180° sector for
// sweep).
func colrV1GradientPaintBuilders(w *colrV1Walker) []colrV1GradientPaintBuilder {
	return []colrV1GradientPaintBuilder{
		{name: "linear", build: func(stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool) {
			return w.linearPaint(0, 0, 100, 0, 0, 100, tables.ExtendPad, stops, colors)
		}},
		{name: "radial", build: func(stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool) {
			return w.radialPaint(0, 0, 0, 0, 0, 100, tables.ExtendPad, stops, colors)
		}},
		// Both sweep orientations: a 0°→180° sector reverses the color line on the way to the shader's clockwise
		// convention, while 180°→0° (startAngle > endAngle, already reversed) leaves it in place, so only the pair
		// covers the color line's first stop landing at either end.
		{name: "sweep", build: func(stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool) {
			return w.sweepPaint(0, 0, -1<<14, 0, tables.ExtendPad, stops, colors)
		}},
		{name: "sweep reversed", build: func(stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool) {
			return w.sweepPaint(0, 0, 0, -1<<14, tables.ExtendPad, stops, colors)
		}},
	}
}

func TestCOLRv1GradientPaintColorIsOpaque(t *testing.T) {
	// A COLRv1 gradient carries its alpha in the color ramp (colrColors4b keeps each stop's alpha), so the paint color a
	// gradient paint node hands blitterFor must stay opaque: shaders.Compile appends a scale_1_float stage over the
	// whole shader output whenever the paint alpha is not 255, which would re-apply one stop's alpha across the entire
	// gradient. The color line below is the case that makes the difference visible — it fades in from a fully
	// transparent stop, so a paint color taken from the first stop would scale the gradient to nothing.
	w := &colrV1Walker{}
	for _, c := range colrV1GradientPaintBuilders(w) {
		t.Run(c.name, func(t *testing.T) {
			paint, ok := c.build([]float32{0, 1}, []colorcore.Color4f{{R: 1, A: 0}, {B: 1, A: 1}})
			if !ok {
				t.Fatal("the paint node failed")
			}
			if paint.shader == nil {
				t.Fatal("no gradient shader was built")
			}
			if a := paint.color.A(); a != 0xFF {
				t.Errorf("paint color alpha %d, want 255; it scales the whole gradient", a)
			}
		})
	}
}

func TestCOLRv1BlitterUsesPaintColor(t *testing.T) {
	// blitterFor compiles the shader against paint.color rather than a color of its own, which is observable through
	// the paint alpha: shaders.Compile scales the pipeline by it. The gradient below spans a 16-pixel row from a fully
	// transparent red to an opaque blue.
	colors := []colorcore.Color4f{{R: 1, A: 0}, {B: 1, A: 1}}
	shader := shaders.NewLinearGradient(geom.Pt(0, 0), geom.Pt(16, 0), colrColors4b(colors), []float32{0, 1},
		shaders.TileClamp, nil)
	if shader == nil {
		t.Fatal("no gradient shader was built")
	}
	render := func(paintColor colorcore.Color) []uint32 {
		pm := raster.NewPixmap(16, 1)
		cv := newColrV1Canvas(pm, 16, 1, geom.IdentityMatrix())
		cv.drawPaint(&colrV1Paint{shader: shader, color: paintColor})
		return slices.Clone(pm.Pix)
	}
	alphaAt := func(row []uint32, x int) int { return int(row[x] >> 24) }

	// The opaque paint color the gradient paint nodes carry leaves the ramp's own alpha intact: near-transparent at the
	// gradient's start (pixel 0 samples at its center, one half-pixel along the ramp, not at the stop itself) and
	// near-opaque at its end.
	opaque := render(colorcore.RGB(0, 0, 0))
	if a := alphaAt(opaque, 0); a > 0x10 {
		t.Errorf("opaque paint color: alpha %d at the transparent stop, want near 0", a)
	}
	if a := alphaAt(opaque, 15); a < 0xF0 {
		t.Errorf("opaque paint color: alpha %d at the opaque stop, want near 255", a)
	}

	// Halving the paint alpha halves the gradient's, which is what proves the paint color reaches shaders.Compile.
	half := render(colorcore.ARGB(0x80, 0, 0, 0))
	if want, got := alphaAt(opaque, 15)/2, alphaAt(half, 15); got < want-2 || got > want+2 {
		t.Errorf("half-alpha paint color: alpha %d at the opaque stop, want about %d", got, want)
	}

	// And the failure mode a non-opaque gradient paint color would produce: the gradient vanishes outright.
	for x, px := range render(colorcore.ARGB(0, 0, 0, 0)) {
		if px != 0 {
			t.Fatalf("transparent paint color: pixel %d is %08x, want the gradient fully scaled away", x, px)
		}
	}
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

// renderCOLRv1WithLayerBudget renders ch's paint graph the way renderCOLRv1 does, but with the canvas's live-layer
// budget preset to budget bytes, and reports the traversal result alongside the glyph. It is the only way to reach the
// budget without a mask over 16M pixels.
func renderCOLRv1WithLayerBudget(t *testing.T, tf *Typeface, ch rune, budget int64) (*Glyph, bool) {
	t.Helper()
	gid := tf.UnicharToGlyph(ch)
	if gid == 0 {
		t.Fatalf("U+%X not mapped", ch)
	}
	identity := geom.IdentityMatrix()
	spec := MakeMaskSpec(NewFont(tf, 50, 1, 0), nil, &identity, nil)
	c := NewScalerContext(spec.Typeface, spec.Rec, spec.Effects)
	g := c.makeGlyph(PackGlyphID(gid))
	if g.ImageSize() == 0 {
		t.Fatalf("U+%X: empty glyph", ch)
	}
	allocGlyphImage(g)
	pm := g.Pixmap()
	canvas := newColrV1Canvas(&pm, g.Width, g.Height, c.maskLocalTranslate(g))
	canvas.layerBudget = budget
	w := &colrV1Walker{
		c:       c,
		t:       c.typeface,
		colr:    c.typeface.colrTable(),
		canvas:  canvas,
		visited: make(map[uint16]bool),
	}
	return g, w.startGlyph(gid, true)
}

func TestCOLRv1CompositeDegradesWhenLayerBudgetExhausted(t *testing.T) {
	tf := loadCOLRv1Typeface(t)
	// A composite node whose layer the live-layer budget cannot cover must degrade to source-over, not fail the
	// subtree: the budget is reachable on ordinary artwork (one full-mask ARGB layer passes colrV1MaxLayerBytes at
	// w*h > 16M pixels), and failing there blanks the whole glyph while its metrics still claim full size.
	for _, ch := range []rune{0xf0a00 /* CLEAR */, 0xf0a03 /* SRC_OVER */} {
		g, ok := renderCOLRv1WithLayerBudget(t, tf, ch, 0)
		if !ok {
			t.Fatalf("U+%X: traversal failed with no layer budget", ch)
		}
		if ink := analyzeGlyphInk(g); ink.count == 0 {
			t.Errorf("U+%X: no ink with the layer budget exhausted", ch)
		}
	}

	// What the degradation costs, measured against the isolated render: for SRC_OVER — the mode it degrades to — only
	// the rounding of the layer composite it skipped (one per channel), while CLEAR loses its mode outright (the source
	// no longer erases the backdrop, so both shapes stay: ink 1028 against the isolated 192). Losing a mode on a mask
	// too large to isolate is the trade; blanking the glyph was the bug.
	for _, c := range []struct {
		ch    rune
		delta int
	}{{ch: 0xf0a03, delta: 1}, {ch: 0xf0a00, delta: 255}} {
		iso := mustRenderCOLRv1(t, tf, c.ch, nil)
		deg, _ := renderCOLRv1WithLayerBudget(t, tf, c.ch, 0)
		if iso.IRect() != deg.IRect() {
			t.Fatalf("U+%X: degraded bounds %v, isolated %v", c.ch, deg.IRect(), iso.IRect())
		}
		worst := 0
		for i, w := range iso.Image32 {
			for shift := 0; shift < 32; shift += 8 {
				d := int(uint8(w>>shift)) - int(uint8(deg.Image32[i]>>shift))
				worst = max(worst, d, -d)
			}
		}
		if worst != c.delta {
			t.Errorf("U+%X: worst channel delta vs the isolated render %d, want %d", c.ch, worst, c.delta)
		}
	}
}

func TestCOLRv1UnpaintableLayerSkipsOnlyThatLayer(t *testing.T) {
	// A fill the tables cannot produce — an out-of-range CPAL index, an empty color line — skips the one layer it
	// configures. Failing the node instead unwound through every enclosing PaintColrLayers loop, so a 20-layer glyph
	// with one bad stop in layer 5 abandoned layers 5 through 20 and cached the partial mask, while renderCOLRv0
	// genuinely continues past a layer with a bad palette index.
	const side = 4
	// Translucent, so each src-over layer moves the pixels and the layers that drew can be counted from the result.
	fg := colorcore.ARGB(0x80, 0xFF, 0x40, 0x20)
	newWalker := func(cv *colrV1Canvas, colr *tables.COLR1) *colrV1Walker {
		return &colrV1Walker{
			c: &ScalerContext{rec: ScalerRec{ForegroundColor: fg}}, t: &Typeface{},
			colr: colr, canvas: cv, visited: map[uint16]bool{},
		}
	}
	render := func(t *testing.T, records ...[]byte) []uint32 {
		t.Helper()
		pm := raster.NewPixmap(side, side)
		w := newWalker(newColrV1Canvas(pm, side, side, geom.IdentityMatrix()), synthCOLRv1Layers(t, records...))
		if !w.traverse(tables.PaintColrLayers{NumLayers: uint8(len(records))}) {
			t.Fatalf("the %d-layer traversal failed", len(records))
		}
		return slices.Clone(pm.Pix)
	}
	good := colrSolidRecordAt(0xFFFF)
	bad := colrSolidRecordAt(0)

	// The premise: the bad record really is unpaintable against a face with no CPAL, and the good one really paints.
	probe := newWalker(nil, nil)
	if _, ok := probe.solidPaint(0, 1<<14); ok {
		t.Fatal("palette index 0 resolves against an empty palette; the case is vacuous")
	}
	if paint, ok := probe.solidPaint(0xFFFF, 1<<14); !ok || paint.color != fg {
		t.Fatalf("the foreground index resolved to (%v, %v), want (%v, true)", paint.color, ok, fg)
	}

	one, three := render(t, good), render(t, good, good, good)
	if slices.Equal(one, three) {
		t.Fatal("one layer and three render alike; stacking is not observable and the case is vacuous")
	}
	if got := render(t, good, bad, good, good); !slices.Equal(got, three) {
		t.Errorf("a bad second layer of four rendered %v, want the three good layers' %v", got, three)
	}
	if got := render(t, bad, good, good, good); !slices.Equal(got, three) {
		t.Errorf("a bad first layer of four rendered %v, want the three good layers' %v", got, three)
	}
	if got := render(t, bad); !slices.Equal(got, make([]uint32, side*side)) {
		t.Errorf("a lone bad layer rendered %v, want an untouched mask", got)
	}

	// The PaintGlyph fill fast path configures its paint the same way and takes the same skip: nothing is drawn, and
	// the node still succeeds. It needs a real outline, so it runs against the conformance font.
	tf := loadCOLRv1Typeface(t)
	gid := tf.UnicharToGlyph(0xf0100)
	if gid == 0 {
		t.Fatal("U+F0100 not mapped")
	}
	const outOfRange = 0xFF00
	if _, ok := tf.PaletteColor(outOfRange, fg); ok {
		t.Fatalf("palette index %#x is in range for the conformance font; pick a higher one", outOfRange)
	}
	// The outline is in font units over the font's 0..1000 em square (with y negated into layer space), so the mask has
	// to be big enough to hold it: 1/20 of the em square, offset so the negated y lands on the mask.
	const emSide = 64
	glyphCTM := geom.ScaleMatrix(1.0/20, 1.0/20)
	glyphCTM.PostTranslate(0, emSide)
	drawnInk := func(paletteIndex uint16) int {
		pm := raster.NewPixmap(emSide, emSide)
		w := newWalker(newColrV1Canvas(pm, emSide, emSide, glyphCTM), nil)
		w.t = tf
		if !w.traverse(tables.PaintGlyph{
			GlyphID: gid,
			Paint:   tables.PaintSolid{PaletteIndex: paletteIndex, Alpha: 1 << 14},
		}) {
			t.Fatalf("the PaintGlyph fast path failed for palette index %#x", paletteIndex)
		}
		ink := 0
		for _, w := range pm.Pix {
			if w != 0 {
				ink++
			}
		}
		return ink
	}
	if ink := drawnInk(0xFFFF); ink == 0 {
		t.Fatal("the paintable PaintGlyph leaf drew nothing; the skip below proves nothing")
	}
	if ink := drawnInk(outOfRange); ink != 0 {
		t.Errorf("the unpaintable PaintGlyph leaf drew %d pixels, want none", ink)
	}
}

func TestCOLRv1BoundsAndDrawChargeTheSameGuards(t *testing.T) {
	// The bounds pass treated PaintGlyph as a leaf while the draw pass descends into its paint, so the two reached
	// different depths and node counts against the same caps. A graph the draw pass refuses then still measured at full
	// size, and the strike cached whatever the draw pass had painted before it gave up. Both passes must charge the
	// same budget for the same graph and reach the same verdict.
	tf := loadCOLRv1Typeface(t)
	gid := tf.UnicharToGlyph(0xf0100)
	if gid == 0 {
		t.Fatal("U+F0100 not mapped")
	}
	solid := tables.PaintSolid{PaletteIndex: 0xFFFF, Alpha: 1 << 14}
	// A translate chain under the outline: nothing but PaintGlyph's own descent puts it inside the depth cap's reach.
	nested := func(depth int) tables.PaintTable {
		var p tables.PaintTable = solid
		for range depth {
			p = tables.PaintTranslate{Dx: 4000, Dy: 4000, Paint: p}
		}
		return tables.PaintGlyph{GlyphID: gid, Paint: p}
	}
	measure := func(paint tables.PaintTable) (geom.Rect, int, bool) {
		var b geom.Rect
		w := &colrV1Walker{c: &ScalerContext{}, t: tf, bounds: &b, visited: map[uint16]bool{}}
		w.ctm.SetIdentity()
		ok := w.traverse(paint)
		return b, w.nodes, ok
	}
	draw := func(paint tables.PaintTable) (int, bool) {
		const side = 8
		cv := newColrV1Canvas(raster.NewPixmap(side, side), side, side, geom.IdentityMatrix())
		w := &colrV1Walker{c: &ScalerContext{}, t: tf, canvas: cv, visited: map[uint16]bool{}}
		ok := w.traverse(paint)
		return w.nodes, ok
	}
	for _, c := range []struct {
		paint tables.PaintTable
		name  string
		want  bool
	}{
		{name: "fill leaf", paint: tables.PaintGlyph{GlyphID: gid, Paint: solid}, want: true},
		{name: "shallow subtree", paint: nested(4), want: true},
		{name: "at the depth cap", paint: nested(colrV1MaxDepth - 3), want: true},
		{name: "past the depth cap", paint: nested(colrV1MaxDepth)},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, boundsNodes, boundsOK := measure(c.paint)
			drawNodes, drawOK := draw(c.paint)
			if boundsOK != c.want || drawOK != c.want {
				t.Errorf("bounds = %v, draw = %v, want both %v", boundsOK, drawOK, c.want)
			}
			if boundsNodes != drawNodes {
				t.Errorf("the bounds pass charged %d nodes, the draw pass %d", boundsNodes, drawNodes)
			}
		})
	}

	// Walking the subtree must not widen the measurement: the draw pass clips everything under a PaintGlyph to its
	// outline, so a nested PaintGlyph translated far outside it contributes nothing however far it moves.
	leaf, _, _ := measure(tables.PaintGlyph{GlyphID: gid, Paint: solid})
	if leaf.IsEmpty() {
		t.Fatal("the outline measured empty; the comparisons below prove nothing")
	}
	clipped, _, _ := measure(tables.PaintGlyph{
		GlyphID: gid,
		Paint: tables.PaintTranslate{
			Dx: 4000, Dy: 4000,
			Paint: tables.PaintGlyph{GlyphID: gid, Paint: solid},
		},
	})
	if clipped != leaf {
		t.Errorf("a nested clipped PaintGlyph measured %v, want the clipping outline's %v", clipped, leaf)
	}
}

func TestCOLRv1RefusedTraversalBlanksTheMask(t *testing.T) {
	// A traversal refused by a guard gives up partway, with the layers ahead of the refusal already painted. Those must
	// not survive into the strike: the metrics lane answers a ClipList-boxed glyph from its box without walking the
	// graph at all, so it never learns the draw pass gave up, and the fragment would be cached under full-size bounds.
	// Both glyphs below carry such a box. The node budget stands in for the guard — charging it up front reaches the
	// refusal without a font crafted to trip one.
	tf := loadCOLRv1Typeface(t)
	for _, ch := range []rune{0xf0c00, 0xf0a03} {
		t.Run(fmt.Sprintf("U+%X", ch), func(t *testing.T) {
			gid := tf.UnicharToGlyph(ch)
			if gid == 0 {
				t.Fatalf("U+%X not mapped", ch)
			}
			if _, _, _, _, ok := colrClipBox(tf.colrTable(), gid); !ok {
				t.Fatalf("U+%X has no ClipList box, so the metrics lane walks the graph and refuses with the draw "+
					"pass; pick a boxed glyph", ch)
			}
			newWalk := func(spentNodes int) (*Glyph, *colrV1Walker) {
				identity := geom.IdentityMatrix()
				spec := MakeMaskSpec(NewFont(tf, 50, 1, 0), nil, &identity, nil)
				c := NewScalerContext(spec.Typeface, spec.Rec, spec.Effects)
				g := c.makeGlyph(PackGlyphID(gid))
				if g.ImageSize() == 0 {
					t.Fatalf("U+%X: empty glyph", ch)
				}
				allocGlyphImage(g)
				pm := g.Pixmap()
				return g, &colrV1Walker{
					c: c, t: tf, colr: tf.colrTable(), visited: map[uint16]bool{},
					canvas: newColrV1Canvas(&pm, g.Width, g.Height, c.maskLocalTranslate(g)),
					nodes:  spentNodes,
				}
			}
			g, w := newWalk(0)
			if !w.startGlyph(gid, true) {
				t.Fatalf("U+%X: the graph does not traverse with a fresh budget", ch)
			}
			total := w.nodes
			if analyzeGlyphInk(g).count == 0 {
				t.Fatalf("U+%X drew nothing at all; the refusals below prove nothing", ch)
			}

			// Which budgets refuse the graph *after* it has painted: those are the ones where blanking is the whole
			// difference between caching a fragment and caching nothing. Discovering them beats pinning a constant, so
			// the test keeps its meaning if the conformance font's graph changes shape.
			var midPaint []int
			for k := 1; k <= total+1; k++ {
				partial, pw := newWalk(colrV1MaxNodes - k)
				if !pw.startGlyph(gid, true) && analyzeGlyphInk(partial).count != 0 {
					midPaint = append(midPaint, colrV1MaxNodes-k)
				}
			}
			if len(midPaint) == 0 {
				t.Fatalf("U+%X: no budget refuses the %d-node graph mid-paint; the case is vacuous", ch, total)
			}
			for _, spent := range midPaint {
				refused, rw := newWalk(spent)
				rw.renderTo(refused)
				if ink := analyzeGlyphInk(refused).count; ink != 0 {
					t.Errorf("U+%X with %d of %d nodes already spent: a refused traversal left %d ink pixels, "+
						"want a blank mask", ch, spent, colrV1MaxNodes, ink)
				}
			}
		})
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

// colrV1WithNullPaintOffset returns a copy of a COLR v1 font whose BaseGlyphList record for gid carries a null paint
// Offset32. typesetting's parseBaseGlyphPaintRecord skips a null offset ("ignore null offset") and still accepts the
// record, so COLR.Search answers ok=true with a nil PaintTable — a color-glyph record naming no artwork at all.
func colrV1WithNullPaintOffset(t *testing.T, data []byte, gid uint16) []byte {
	t.Helper()
	patched := false
	out := patchSfntTable(t, data, "COLR", func(tab []byte) {
		if v := binary.BigEndian.Uint16(tab); v != 1 {
			t.Fatalf("COLR version %d, want 1", v)
		}
		list := int(binary.BigEndian.Uint32(tab[14:])) // baseGlyphListOffset
		for i := range int(binary.BigEndian.Uint32(tab[list:])) {
			record := list + 4 + i*6
			if binary.BigEndian.Uint16(tab[record:]) == gid {
				binary.BigEndian.PutUint32(tab[record+2:], 0)
				patched = true
			}
		}
	})
	if !patched {
		t.Fatalf("no COLR v1 base-glyph record for glyph %d", gid)
	}
	return out
}

func TestCOLRv1NullPaintOffsetStaysOnTheOutlineLane(t *testing.T) {
	// A BaseGlyphList record with a null paint offset is still a record typesetting reports, so the glyph looks like a
	// color glyph while carrying no paint graph. Committing it to the v1 lane on that basis makes colrV1Bounds return an
	// empty rect and traverse(nil) fall to its default and refuse: the glyph measures as nothing, draws as nothing, and
	// never asks for the perfectly good glyf outline it also carries. FreeType rejects the record and Skia falls back to
	// that outline.
	const ch = 0xf0100
	data := readTestFont(t, "test_glyphs-glyf_colr_1.ttf")
	gid := loadCOLRv1Typeface(t).UnicharToGlyph(ch)
	if gid == 0 {
		t.Fatalf("U+%X not mapped", ch)
	}
	tf, err := NewTypefaceFromData(colrV1WithNullPaintOffset(t, data, gid), 0)
	if err != nil {
		t.Fatal(err)
	}
	// typesetting still claims the glyph, which is the whole problem.
	if paint, ok := tf.face.COLR.Search(gid); !ok {
		t.Fatal("the patched record no longer parses; the case proves nothing")
	} else if paint != nil {
		t.Fatalf("COLR.Search returned %T, want a nil paint", paint)
	}
	if paint, ok := tf.faceColorPaint(opentype.GID(gid)); ok || paint != nil {
		t.Errorf("faceColorPaint = (%v, %v), want (nil, false)", paint, ok)
	}

	// The glyph is measured and drawn as the outline it still has, byte for byte as the same font with no COLR table at
	// all draws it.
	g, action := colrV1Glyph(t, tf, ch, 50, nil)
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
	ref, action := colrV1Glyph(t, outlineOnly, ch, 50, nil)
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
	// Every other glyph of the font still renders through the v1 lane: only the patched record changed lanes.
	if other := mustRenderCOLRv1(t, tf, 0xf0300, nil); analyzeGlyphInk(other).count == 0 {
		t.Error("U+F0300 lost its color artwork")
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

func TestCOLRv1CompositeBudgetDegradesTheNode(t *testing.T) {
	// End to end through the walker: a composite chain that runs out of layer budget keeps traversing — the nodes past
	// the budget composite source-over instead of failing the subtree the way the depth and node caps do — and still
	// unwinds to a balanced canvas with every byte returned. PaintSolid with palette index 0xFFFF resolves to the
	// (zero) foreground color without touching the face, so no real typeface is needed.
	build := func(nesting int) tables.PaintTable {
		solid := tables.PaintSolid{PaletteIndex: 0xFFFF, Alpha: 1 << 14}
		var p tables.PaintTable = solid
		for range nesting {
			p = tables.PaintComposite{BackdropPaint: solid, SourcePaint: p}
		}
		return p
	}
	const side = 32
	for _, nesting := range []int{2, 3, colrV1MaxDepth / 2} {
		cv := newColrV1Canvas(raster.NewPixmap(side, side), side, side, geom.IdentityMatrix())
		cv.layerBudget = 5 * cv.layerBytes() // enough for two nested composites, one layer short of three
		w := &colrV1Walker{c: &ScalerContext{}, t: &Typeface{}, canvas: cv, visited: map[uint16]bool{}}
		if !w.traverse(build(nesting)) {
			t.Errorf("%d nested composites: traverse failed past the layer budget", nesting)
		}
		cv.restoreToCount(0)
		if len(cv.devs) != 1 || len(cv.modes) != 0 {
			t.Errorf("%d nested composites: unwound to %d devices / %d modes, want 1/0",
				nesting, len(cv.devs), len(cv.modes))
		}
		if cv.layerBudget != 5*cv.layerBytes() {
			t.Errorf("%d nested composites: budget %d after unwinding, want %d",
				nesting, cv.layerBudget, 5*cv.layerBytes())
		}
	}
}

// colrSolidRecord is a PaintSolid (format 2) painting the (zero) foreground color: palette index 0xFFFF resolves
// without a face, so a synthetic table built from it needs no real typeface.
func colrSolidRecord() []byte {
	rec := []byte{2}                                 // format
	rec = binary.BigEndian.AppendUint16(rec, 0xFFFF) // palette index (foreground)
	return binary.BigEndian.AppendUint16(rec, 1<<14) // alpha 1.0
}

// colrSolidRecordAt is colrSolidRecord with an explicit palette index. 0xFFFF is the foreground, which resolves against
// any face; every other index reads CPAL, so any of them is out of range for a face with no palette at all.
func colrSolidRecordAt(paletteIndex uint16) []byte {
	rec := binary.BigEndian.AppendUint16([]byte{2}, paletteIndex)
	return binary.BigEndian.AppendUint16(rec, 1<<14)
}

// colrLayersRecord is a PaintColrLayers (format 1) naming the LayerList range [first, first+num).
func colrLayersRecord(first uint32, num uint8) []byte {
	return binary.BigEndian.AppendUint32([]byte{1, num}, first)
}

// synthCOLRv1Layers builds the smallest parseable COLRv1 table holding the given paint records as its LayerList: an
// empty v0 section, no BaseGlyphList/ClipList/variation subtables, and the offset array computed for the records.
// Everything the walker needs to resolve a PaintColrLayers node is present, and nothing in it touches a face.
func synthCOLRv1Layers(t *testing.T, records ...[]byte) *tables.COLR1 {
	t.Helper()
	const layerListOffset = 34                   // the v0 header (14) plus the v1 offsets (20)
	raw := binary.BigEndian.AppendUint16(nil, 1) // version
	raw = binary.BigEndian.AppendUint16(raw, 0)  // numBaseGlyphRecords
	raw = binary.BigEndian.AppendUint32(raw, 0)  // baseGlyphRecordsOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, 0)  // layerRecordsOffset (null)
	raw = binary.BigEndian.AppendUint16(raw, 0)  // numLayerRecords
	raw = binary.BigEndian.AppendUint32(raw, 0)  // baseGlyphListOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, layerListOffset)
	raw = binary.BigEndian.AppendUint32(raw, 0) // clipListOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, 0) // varIndexMapOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, 0) // itemVariationStoreOffset (null)
	raw = binary.BigEndian.AppendUint32(raw, uint32(len(records)))
	offset := uint32(4 + 4*len(records)) // from the LayerList start: its count, then its offset array
	for _, rec := range records {
		raw = binary.BigEndian.AppendUint32(raw, offset)
		offset += uint32(len(rec))
	}
	for _, rec := range records {
		raw = append(raw, rec...)
	}
	colr, err := tables.ParseCOLR(raw)
	if err != nil {
		t.Fatalf("the synthetic COLR table does not parse: %v", err)
	}
	return &colr
}

// synthCOLRv1LayerList builds a synthetic table whose one-entry LayerList holds a PaintSolid.
func synthCOLRv1LayerList(t *testing.T) *tables.COLR1 {
	t.Helper()
	return synthCOLRv1Layers(t, colrSolidRecord())
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

func TestCOLRv1LayerListSelfReference(t *testing.T) {
	// PaintColrGlyph is not the paint graph's only back edge: tables.LayerList.Resolve is an index lookup performed at
	// walk time, so a PaintColrLayers stored in the LayerList can name a range containing itself. The visited set does
	// not cover that re-entry, so the traversal must terminate on the depth cap instead of recursing forever.
	colr := synthCOLRv1Layers(t, colrLayersRecord(0, 1)) // LayerList[0] resolves to itself as its only child
	var bounds geom.Rect
	w := &colrV1Walker{colr: colr, bounds: &bounds, visited: map[uint16]bool{}}
	w.ctm.SetIdentity()
	if w.traverse(tables.PaintColrLayers{FirstLayerIndex: 0, NumLayers: 1}) {
		t.Error("a self-referencing LayerList entry must fail the traversal")
	}
	if w.depth != 0 {
		t.Errorf("depth counter leaked: %d", w.depth)
	}
}

// colrFanOutLayers builds a LayerList of levels pairs, each entry of level k a PaintColrLayers naming both entries of
// level k+1, with a pair of PaintSolids as the last level. A PaintColrLayers naming level 0's pair therefore expands to
// 2^(levels+2)-1 nodes — the shape a shared, unmemoized reference gives an attacker — while nesting only levels+2 deep,
// well inside colrV1MaxDepth.
func colrFanOutLayers(t *testing.T, levels int) *tables.COLR1 {
	t.Helper()
	records := make([][]byte, 0, 2*levels+2)
	for k := range levels {
		next := uint32(2 * (k + 1))
		records = append(records, colrLayersRecord(next, 2), colrLayersRecord(next, 2))
	}
	records = append(records, colrSolidRecord(), colrSolidRecord())
	return synthCOLRv1Layers(t, records...)
}

func TestCOLRv1NodeBudget(t *testing.T) {
	// The depth cap bounds nesting, not work: a LayerList whose entries fan out to a shared next level doubles the node
	// count per level, so a few hundred bytes of font data describe a tree of 2^63 visits that never exceeds
	// colrV1MaxDepth. colrV1MaxNodes must bound the traversal instead, and must not be handed back as the walk unwinds.
	root := tables.PaintColrLayers{FirstLayerIndex: 0, NumLayers: 2}

	// 16 levels is 2^18-1 nodes: far past the budget, yet small enough that a walker without one still finishes here
	// (the whole point being that the shape scales to a hang with a handful more records).
	const levels = 16
	colr := colrFanOutLayers(t, levels)
	var bounds geom.Rect
	w := &colrV1Walker{colr: colr, bounds: &bounds, visited: map[uint16]bool{}}
	w.ctm.SetIdentity()
	if w.traverse(root) {
		t.Errorf("a %d-level fan-out must fail the traversal", levels)
	}
	if w.nodes != colrV1MaxNodes {
		t.Errorf("bounds traversal visited %d nodes, want the budget %d", w.nodes, colrV1MaxNodes)
	}
	// The budget is per traversal, not per subtree: a spent walker stays spent.
	if w.traverse(tables.PaintSolid{PaletteIndex: 0xFFFF, Alpha: 1 << 14}) {
		t.Error("a spent budget must keep failing")
	}

	// The same guard on the draw dispatcher. PaintSolid with palette index 0xFFFF resolves to the (zero) foreground
	// color without touching the face, so no real typeface is needed.
	const side = 8
	cv := newColrV1Canvas(raster.NewPixmap(side, side), side, side, geom.IdentityMatrix())
	w = &colrV1Walker{c: &ScalerContext{}, t: &Typeface{}, colr: colr, canvas: cv, visited: map[uint16]bool{}}
	if w.traverse(root) {
		t.Error("draw traverse of the fan-out must fail")
	}
	if w.nodes != colrV1MaxNodes {
		t.Errorf("draw traversal visited %d nodes, want the budget %d", w.nodes, colrV1MaxNodes)
	}
	cv.restoreToCount(0)
	if len(cv.devs) != 1 || len(cv.modes) != 0 {
		t.Errorf("unwound to %d devices / %d modes, want 1/0", len(cv.devs), len(cv.modes))
	}

	// The control: the same shape inside the budget still traverses. The root plus 7 doubling levels is 255 nodes.
	small := colrFanOutLayers(t, 6)
	bounds = geom.Rect{}
	w = &colrV1Walker{colr: small, bounds: &bounds, visited: map[uint16]bool{}}
	w.ctm.SetIdentity()
	if !w.traverse(root) {
		t.Error("a fan-out inside the budget must traverse")
	}
	if w.nodes != 255 {
		t.Errorf("the 6-level fan-out visited %d nodes, want 255", w.nodes)
	}
}

func TestCOLRv1NodeBudgetHeadroom(t *testing.T) {
	// The budget only ever fires on a crafted graph: every v1 glyph in the COLRv1 conformance font — the broadest paint
	// graphs anyone publishes, from the spec's own generator — must traverse with orders of magnitude to spare. A
	// bounds-mode walk needs neither a scaler context nor the root transform, so it can be driven per glyph directly.
	tf := loadCOLRv1Typeface(t)
	colr := tf.colrTable()
	if colr == nil {
		t.Fatal("the conformance font has no COLR table")
	}
	worst, worstGID, walked := 0, 0, 0
	for gid := range tf.CountGlyphs() {
		var bounds geom.Rect
		w := &colrV1Walker{t: tf, colr: colr, bounds: &bounds, visited: map[uint16]bool{}}
		w.ctm.SetIdentity()
		if !w.startGlyph(uint16(gid), false) {
			continue // no v1 paint for this glyph (or a v0 one)
		}
		walked++
		if w.nodes > worst {
			worst, worstGID = w.nodes, gid
		}
	}
	if walked == 0 {
		t.Fatal("no glyph in the conformance font traversed a v1 paint graph")
	}
	if worst > colrV1MaxNodes/16 {
		t.Errorf("glyph %d visits %d nodes, within 16x of the %d budget: the budget is too tight for real fonts",
			worstGID, worst, colrV1MaxNodes)
	}
	t.Logf("%d v1 glyphs walked; the busiest (glyph %d) visits %d of the %d node budget",
		walked, worstGID, worst, colrV1MaxNodes)
}

func TestCOLRv1RadialNegativeRadius(t *testing.T) {
	// COLR radii are UFWORD, but radialPaint's stop rescale computes startRadius + (endRadius-startRadius)*stops[0], so
	// an unsigned pair plus a stop offset outside [0, 1] — every value below is legal F2Dot14 — still yields a negative
	// radius. shaders.NewTwoPointConicalGradient rejects a negative radius by returning nil, which would fail the paint
	// node and drop the rest of the glyph's paint graph mid-draw, so the radius has to be resolved instead.
	w := &colrV1Walker{}
	for _, c := range []struct {
		name   string
		stops  []float32
		r0, r1 uint16
		extend tables.Extend
	}{
		{name: "clamp start negative", r0: 0, r1: 100, extend: tables.ExtendPad, stops: []float32{-1, 0}},
		{name: "clamp end negative", r0: 100, r1: 0, extend: tables.ExtendPad, stops: []float32{0, 1.5}},
		{name: "clamp both negative", r0: 0, r1: 100, extend: tables.ExtendPad, stops: []float32{-2, -1}},
		{name: "clamp cut between stops", r0: 200, r1: 0, extend: tables.ExtendPad, stops: []float32{0, 0.5, 1.5}},
		{name: "repeat end negative", r0: 100, r1: 0, extend: tables.ExtendRepeat, stops: []float32{0, 1.5}},
		{name: "reflect end negative", r0: 100, r1: 0, extend: tables.ExtendReflect, stops: []float32{0, 1.5}},
		{name: "repeat start negative", r0: 0, r1: 100, extend: tables.ExtendRepeat, stops: []float32{-1, 0}},
		{name: "reflect both negative", r0: 0, r1: 100, extend: tables.ExtendReflect, stops: []float32{-2, -1}},
	} {
		t.Run(c.name, func(t *testing.T) {
			colors := make([]colorcore.Color4f, len(c.stops))
			for i := range colors {
				colors[i] = colorcore.Color4f{R: float32(i) / float32(len(colors)), B: 1, A: 1}
			}
			paint, ok := w.radialPaint(10, 20, c.r0, 30, 40, c.r1, c.extend, append([]float32(nil), c.stops...), colors)
			if !ok {
				t.Fatal("the paint node failed instead of resolving the negative radius")
			}
			if paint.shader == nil {
				t.Error("no gradient shader was built")
			}
		})
	}
}

func TestCOLRv1ResolveNegativeRadius(t *testing.T) {
	// The resolution's contract, over the whole negative-radius space the rescale can produce: whatever comes back is
	// something the two-point conical shader accepts — finite centers, non-negative radii, and a color line whose stops
	// and colors still correspond.
	colors := []colorcore.Color4f{{R: 1, A: 1}, {G: 1, A: 1}, {B: 1, A: 1}}
	gradient := func(startRadius, endRadius float32) colrRadialGradient {
		return colrRadialGradient{
			stops:       []float32{0, 0.25, 1},
			colors:      append([]colorcore.Color4f(nil), colors...),
			start:       geom.Pt(10, -20),
			end:         geom.Pt(30, -40),
			startRadius: startRadius,
			endRadius:   endRadius,
		}
	}
	for _, tileMode := range []shaders.TileMode{shaders.TileClamp, shaders.TileRepeat, shaders.TileMirror} {
		for r0 := -300; r0 <= 300; r0 += 25 {
			for r1 := -300; r1 <= 300; r1 += 25 {
				if r0 == r1 || (r0 >= 0 && r1 >= 0) {
					continue // equal radii draw nothing; a non-negative pair never reaches the resolution
				}
				got, drawn := gradient(float32(r0), float32(r1)).resolveNegativeRadius(tileMode)
				if !drawn {
					t.Fatalf("mode %v r0=%d r1=%d: reported nothing to draw for unequal radii", tileMode, r0, r1)
				}
				if got.startRadius < 0 || got.endRadius < 0 {
					t.Errorf("mode %v r0=%d r1=%d: resolved to radii %g/%g, want both non-negative",
						tileMode, r0, r1, got.startRadius, got.endRadius)
				}
				if !got.start.IsFinite() || !got.end.IsFinite() {
					t.Errorf("mode %v r0=%d r1=%d: resolved to centers %v/%v, want both finite",
						tileMode, r0, r1, got.start, got.end)
				}
				if len(got.stops) != len(got.colors) || len(got.stops) == 0 {
					t.Errorf("mode %v r0=%d r1=%d: %d stops for %d colors",
						tileMode, r0, r1, len(got.stops), len(got.colors))
				}
				if shaders.NewTwoPointConicalGradient(got.start, got.startRadius, got.end, got.endRadius,
					colrColors4b(got.colors), got.stops, tileMode, nil) == nil {
					t.Errorf("mode %v r0=%d r1=%d: the shader rejected the resolved gradient", tileMode, r0, r1)
				}
			}
		}
	}
	// Equal negative radii are the empty gradient, and the zero denominator the resolution would otherwise divide by.
	if _, drawn := gradient(-50, -50).resolveNegativeRadius(shaders.TileClamp); drawn {
		t.Error("equal negative radii must report nothing to draw")
	}
}

func TestCOLRv1TruncateToStopInterpolating(t *testing.T) {
	// The clamp lane cuts the color line where the radius reaches zero and replaces the discarded side with one stop
	// carrying the interpolated color, so the colors that side held do not reappear under the clamp. A cut outside the
	// stop range, or landing on an end stop, has nothing to interpolate between and leaves the line alone.
	red := colorcore.Color4f{R: 1, A: 1}
	green := colorcore.Color4f{G: 1, A: 1}
	blue := colorcore.Color4f{B: 1, A: 1}
	in := []float32{0, 0.5, 1}
	for _, c := range []struct {
		name          string
		wantStops     []float32
		wantColors    []colorcore.Color4f
		cut           float32
		truncateStart bool
	}{
		{
			name: "start, mid first span", cut: 0.25, truncateStart: true, wantStops: []float32{0, 0.5, 1},
			wantColors: []colorcore.Color4f{{R: 0.5, G: 0.5, A: 1}, green, blue},
		},
		{
			name: "start, mid second span", cut: 0.75, truncateStart: true, wantStops: []float32{0, 1},
			wantColors: []colorcore.Color4f{{G: 0.5, B: 0.5, A: 1}, blue},
		},
		{
			name: "end, mid first span", cut: 0.25, truncateStart: false, wantStops: []float32{0, 1},
			wantColors: []colorcore.Color4f{red, {R: 0.5, G: 0.5, A: 1}},
		},
		{
			name: "end, mid second span", cut: 0.75, truncateStart: false, wantStops: []float32{0, 0.5, 1},
			wantColors: []colorcore.Color4f{red, green, {G: 0.5, B: 0.5, A: 1}},
		},
		{
			name: "past the end", cut: 1.5, truncateStart: false, wantStops: in,
			wantColors: []colorcore.Color4f{red, green, blue},
		},
		{
			name: "before the start", cut: -0.5, truncateStart: true, wantStops: in,
			wantColors: []colorcore.Color4f{red, green, blue},
		},
		{
			name: "on the first stop", cut: 0, truncateStart: true, wantStops: in,
			wantColors: []colorcore.Color4f{red, green, blue},
		},
		{
			name: "on the last stop", cut: 1, truncateStart: false, wantStops: in,
			wantColors: []colorcore.Color4f{red, green, blue},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			stops, colors := colrTruncateToStopInterpolating(c.cut, append([]float32(nil), in...),
				[]colorcore.Color4f{red, green, blue}, c.truncateStart)
			if !slices.Equal(stops, c.wantStops) {
				t.Errorf("stops = %v, want %v", stops, c.wantStops)
			}
			if !slices.Equal(colors, c.wantColors) {
				t.Errorf("colors = %v, want %v", colors, c.wantColors)
			}
		})
	}
}
