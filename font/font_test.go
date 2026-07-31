// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Expected values in this file were captured from a reference C oracle run over the same font files. The live oracle
// probe that re-verified them against that oracle is gone with the C library; these frozen values are the record.

package font

import (
	"math"
	"os"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

func loadTypeface(t *testing.T, name string, index int) *Typeface {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	tf, err := NewTypefaceFromData(data, index)
	if err != nil {
		t.Fatal(err)
	}
	return tf
}

func TestTypefaceQueries(t *testing.T) {
	cases := []struct {
		file       string
		family     string
		index      int
		weight     int
		width      int
		upem       int
		nGlyphs    int
		slant      Slant
		fixedPitch bool
	}{
		{file: "Roboto-Regular.ttf", family: "Roboto", weight: 400, width: 5, upem: 2048, nGlyphs: 238},
		{file: "DejaVuSans.subset.ttf", family: "DejaVu Sans", weight: 400, width: 5, upem: 2048, nGlyphs: 4},
		{file: "test.ttc", family: "Test", weight: 400, width: 5, upem: 2048, nGlyphs: 12},
		{file: "test.ttc", index: 1, family: "Test", weight: 700, width: 5, upem: 2048, nGlyphs: 12},
	}
	for _, c := range cases {
		tf := loadTypeface(t, c.file, c.index)
		if got := tf.FamilyName(); got != c.family {
			t.Errorf("%s[%d]: family = %q, want %q", c.file, c.index, got, c.family)
		}
		st := tf.Style()
		if st.Weight() != c.weight || st.Width() != c.width || st.Slant() != c.slant {
			t.Errorf("%s[%d]: style = (%d,%d,%d), want (%d,%d,%d)", c.file, c.index,
				st.Weight(), st.Width(), st.Slant(), c.weight, c.width, c.slant)
		}
		if got := tf.UnitsPerEm(); got != c.upem {
			t.Errorf("%s[%d]: upem = %d, want %d", c.file, c.index, got, c.upem)
		}
		if got := tf.IsFixedPitch(); got != c.fixedPitch {
			t.Errorf("%s[%d]: fixedPitch = %v", c.file, c.index, got)
		}
		if got := tf.CountGlyphs(); got != c.nGlyphs {
			t.Errorf("%s[%d]: countGlyphs = %d, want %d", c.file, c.index, got, c.nGlyphs)
		}
	}
}

func TestTypefaceFromDataErrors(t *testing.T) {
	data, err := os.ReadFile("testdata/test.ttc")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = NewTypefaceFromData(data, 2); err == nil {
		t.Error("index 2 of a 2-face collection should fail")
	}
	if _, err = NewTypefaceFromData(data, -1); err == nil {
		t.Error("negative index should fail")
	}
	if _, err = NewTypefaceFromData([]byte("not a font"), 0); err == nil {
		t.Error("garbage data should fail")
	}
}

func TestGlyphMapping(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 100, 1, 0)

	text := []byte("Hxign AVW.!? o")
	want := []uint16{44, 92, 77, 75, 82, 4, 37, 58, 59, 18, 5, 35, 4, 83}
	if n := f.TextToGlyphs(text, TextEncodingUTF8, nil); n != len(want) {
		t.Fatalf("count = %d, want %d", n, len(want))
	}
	glyphs := make([]uint16, len(want))
	f.TextToGlyphs(text, TextEncodingUTF8, glyphs)
	for i := range want {
		if glyphs[i] != want[i] {
			t.Fatalf("glyphs = %v, want %v", glyphs, want)
		}
	}
	if got := f.UnicharToGlyph(0x20AC); got != 159 { // euro sign
		t.Errorf("euro glyph = %d, want 159", got)
	}
	if got := f.UnicharToGlyph(0x1F600); got != 0 { // unmapped
		t.Errorf("unmapped glyph = %d, want 0", got)
	}
	if got := f.UnicharToGlyph(-1); got != 0 {
		t.Errorf("negative unichar glyph = %d, want 0", got)
	}

	// UTF-16 with a surrogate pair (H + U+1F600).
	utf16 := []byte{0x48, 0x00, 0x3D, 0xD8, 0x00, 0xDE}
	g16 := make([]uint16, 2)
	if n := f.TextToGlyphs(utf16, TextEncodingUTF16, g16); n != 2 || g16[0] != 44 || g16[1] != 0 {
		t.Errorf("utf16 = %d %v", n, g16)
	}
	// UTF-32.
	utf32 := []byte{0x48, 0, 0, 0, 0xAC, 0x20, 0, 0}
	g32 := make([]uint16, 2)
	if n := f.TextToGlyphs(utf32, TextEncodingUTF32, g32); n != 2 || g32[0] != 44 || g32[1] != 159 {
		t.Errorf("utf32 = %d %v", n, g32)
	}
	// Glyph IDs pass through.
	gid := []byte{44, 0, 159, 0}
	gg := make([]uint16, 2)
	if n := f.TextToGlyphs(gid, TextEncodingGlyphID, gg); n != 2 || gg[0] != 44 || gg[1] != 159 {
		t.Errorf("glyphid = %d %v", n, gg)
	}
	// Invalid UTF-8 reports -1 without writing.
	if n := f.TextToGlyphs([]byte{0x48, 0xC0, 0x20}, TextEncodingUTF8, gg); n != -1 {
		t.Errorf("invalid utf8 count = %d, want -1", n)
	}
	// Insufficient output space returns the count without writing.
	small := make([]uint16, 2)
	if n := f.TextToGlyphs(text, TextEncodingUTF8, small); n != len(want) || small[0] != 0 {
		t.Errorf("small buffer: n=%d small=%v", n, small)
	}
	// UnicharsToGlyphs.
	out := make([]uint16, 3)
	f.UnicharsToGlyphs([]int32{'H', 0x20AC, 0x1F600}, out)
	if out[0] != 44 || out[1] != 159 || out[2] != 0 {
		t.Errorf("unicharsToGlyphs = %v", out)
	}
}

func near(a, b, tol float32) bool {
	return float32(math.Abs(float64(a)-float64(b))) <= tol
}

func TestAdvancesAndPositions(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 100, 1, 0)

	glyphs := []uint16{44, 92, 75, 5} // H x g !
	wantWidths := []float32{71.28906, 49.560547, 56.103516, 25.732422}
	widths := make([]float32, len(glyphs))
	f.GlyphWidths(glyphs, widths)
	for i := range wantWidths {
		if !near(widths[i], wantWidths[i], 1e-4) {
			t.Errorf("width[%d] = %v, want %v", i, widths[i], wantWidths[i])
		}
	}

	xpos := make([]float32, len(glyphs))
	f.GetXPos(glyphs, xpos, 10.5)
	wantX := []float32{10.5, 81.78906, 131.34961, 187.45312}
	for i := range wantX {
		if !near(xpos[i], wantX[i], 1e-3) {
			t.Errorf("xpos[%d] = %v, want %v", i, xpos[i], wantX[i])
		}
	}

	// Out-of-range glyph IDs measure as zero (the failed-load behavior).
	f.GlyphWidths([]uint16{9999}, widths[:1])
	if widths[0] != 0 {
		t.Errorf("out-of-range width = %v", widths[0])
	}

	// scaleX and skewX: advances scale by scaleX; skew leaves advances unchanged.
	f2 := NewFont(tf, 50, 1.5, -0.25)
	f2.GlyphWidths(glyphs[:1], widths[:1])
	if !near(widths[0], 71.28906*0.75, 1e-3) {
		t.Errorf("scaled width = %v", widths[0])
	}
}

func TestMeasureText(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 100, 1, 0)
	text := []byte("Hxg!")

	var bounds geom.Rect
	w := f.MeasureText(text, TextEncodingUTF8, &bounds, nil)
	if !near(w, 202.68555, 1e-3) {
		t.Errorf("measure = %v", w)
	}
	// Oracle (path-lane, which shares the raw outline bounds): {8 -72 195.95312 21}.
	want := geom.RectLTRB(8, -72, 195.95312, 21)
	if !near(bounds.Left, want.Left, 1e-3) || !near(bounds.Top, want.Top, 1e-3) ||
		!near(bounds.Right, want.Right, 1e-3) || !near(bounds.Bottom, want.Bottom, 1e-3) {
		t.Errorf("bounds = %v, want %v", bounds, want)
	}
	// Without bounds the width is identical.
	if w2 := f.MeasureText(text, TextEncodingUTF8, nil, nil); w2 != w {
		t.Errorf("width without bounds = %v", w2)
	}
	// Empty text.
	bounds = geom.RectLTRB(1, 2, 3, 4)
	if w2 := f.MeasureText(nil, TextEncodingUTF8, &bounds, nil); w2 != 0 || !bounds.IsEmpty() {
		t.Errorf("empty text = %v %v", w2, bounds)
	}

	// Stroked bounds (oracle-verified): stroke width 2 outsets by 1, width 7.5 by 3.75 (pre-round).
	p := &stroke.PaintSpec{Style: stroke.PaintStyleStroke, Width: 2, MiterLimit: 4}
	w = f.MeasureText(text, TextEncodingUTF8, &bounds, p)
	if !near(w, 202.68555, 1e-3) {
		t.Errorf("stroked measure = %v", w)
	}
	wantStroke := geom.RectLTRB(7, -73, 196.95312, 22)
	if bounds != wantStroke {
		t.Errorf("stroked bounds = %v, want %v", bounds, wantStroke)
	}
	p.Width = 7.5
	f.MeasureText(text, TextEncodingUTF8, &bounds, p)
	wantStroke = geom.RectLTRB(4, -75, 199.95312, 25)
	if bounds != wantStroke {
		t.Errorf("stroked(7.5) bounds = %v, want %v", bounds, wantStroke)
	}
	// Stroke-and-fill with width 0 behaves as fill (a zero-width stroke collapses to nothing), through the path lane.
	p.Style = stroke.PaintStyleStrokeAndFill
	p.Width = 0
	f.MeasureText(text, TextEncodingUTF8, &bounds, p)
	if bounds != want {
		t.Errorf("strokefill(0) bounds = %v, want %v", bounds, want)
	}
}

func TestCanonicalization(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	// Size 300 exceeds the 256 gate: the strike measures at 64 and scales by 300/64.
	f := NewFont(tf, 300, 1, 0)
	widths := make([]float32, 1)
	f.GlyphWidths([]uint16{44}, widths)
	if !near(widths[0], 213.86719, 1e-3) { // linear: 71.28906 * 3
		t.Errorf("big width = %v", widths[0])
	}
	var bounds geom.Rect
	f.MeasureText([]byte("Hxign"), TextEncodingUTF8, &bounds, nil)
	// Vertical bounds quantize to the canonical strike: multiples of 300/64 = 4.6875 (the horizontal edges carry
	// fractional advance offsets, so only top/bottom sit exactly on the grid). Zero sits on every grid, so the step
	// counts are pinned rather than merely rounded against: an all-zero rect satisfies integrality on its own.
	if want := geom.RectLTRB(23.4375, -220.3125, 748.9746, 65.625); bounds != want {
		t.Errorf("size 300 bounds = %v, want %v", bounds, want)
	}
	checkGrid := func(what string, b geom.Rect, step, topSteps, bottomSteps float32) {
		t.Helper()
		if b.IsEmpty() {
			t.Fatalf("%s bounds are empty; there is no grid to be on", what)
		}
		for _, c := range []struct {
			edge string
			got  float32
			want float32
		}{
			{edge: "top", got: b.Top / step, want: topSteps},
			{edge: "bottom", got: b.Bottom / step, want: bottomSteps},
		} {
			if !near(c.got, c.want, 1e-4) {
				t.Errorf("%s %s edge sits at %v canonical steps, want %v", what, c.edge, c.got, c.want)
			}
			if c.want != float32(math.Round(float64(c.want))) {
				t.Errorf("%s %s expectation %v is not a whole step", what, c.edge, c.want)
			}
		}
	}
	checkGrid("size 300", bounds, 4.6875, -47, 14)
	// A hairline stroke paint also forces the canonical path lane, dropping the paint.
	fSmall := NewFont(tf, 100, 1, 0)
	p := &stroke.PaintSpec{Style: stroke.PaintStyleStroke, Width: 0, MiterLimit: 4}
	fSmall.MeasureText([]byte("Hxg!"), TextEncodingUTF8, &bounds, p)
	if want := geom.RectLTRB(7.8125, -71.875, 195.70312, 21.875); bounds != want {
		t.Errorf("hairline bounds = %v, want %v", bounds, want)
	}
	checkGrid("hairline", bounds, 100.0/64, -46, 14)
}

func TestFontMetrics(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 100, 1, 0)
	var m Metrics
	spacing := f.Metrics(&m)
	want := Metrics{
		Flags:              15,
		Top:                -105.615234,
		Ascent:             -92.77344,
		Descent:            24.414062,
		Bottom:             27.09961,
		Leading:            0,
		AvgCharWidth:       56.54297, // OS/2 xAvgCharWidth (the FreeType recipe; CoreText reports the bbox width)
		MaxCharWidth:       188.52539,
		XMin:               -73.68164,
		XMax:               114.84375,
		XHeight:            52.83203,
		CapHeight:          71.09375,
		UnderlineThickness: 4.8828125,
		UnderlinePosition:  7.3242188, // -post.underlinePosition (-150)/2048*100; the FT half-thickness bias cancels
		StrikeoutThickness: 4.9804688,
		StrikeoutPosition:  -25,
	}
	if m.Flags != want.Flags {
		t.Errorf("flags = %d, want %d", m.Flags, want.Flags)
	}
	fields := []struct {
		name string
		got  float32
		want float32
	}{
		{name: "Top", got: m.Top, want: want.Top},
		{name: "Ascent", got: m.Ascent, want: want.Ascent},
		{name: "Descent", got: m.Descent, want: want.Descent},
		{name: "Bottom", got: m.Bottom, want: want.Bottom},
		{name: "Leading", got: m.Leading, want: want.Leading},
		{name: "AvgCharWidth", got: m.AvgCharWidth, want: want.AvgCharWidth},
		{name: "MaxCharWidth", got: m.MaxCharWidth, want: want.MaxCharWidth},
		{name: "XMin", got: m.XMin, want: want.XMin},
		{name: "XMax", got: m.XMax, want: want.XMax},
		{name: "XHeight", got: m.XHeight, want: want.XHeight},
		{name: "CapHeight", got: m.CapHeight, want: want.CapHeight},
		{name: "UnderlineThickness", got: m.UnderlineThickness, want: want.UnderlineThickness},
		{name: "UnderlinePosition", got: m.UnderlinePosition, want: want.UnderlinePosition},
		{name: "StrikeoutThickness", got: m.StrikeoutThickness, want: want.StrikeoutThickness},
		{name: "StrikeoutPosition", got: m.StrikeoutPosition, want: want.StrikeoutPosition},
	}
	for _, fl := range fields {
		if !near(fl.got, fl.want, 1e-3) {
			t.Errorf("%s = %v, want %v", fl.name, fl.got, fl.want)
		}
	}
	if !near(spacing, 117.1875, 1e-3) {
		t.Errorf("spacing = %v", spacing)
	}
	// Metrics with a nil pointer still returns the spacing.
	if got := f.Metrics(nil); got != spacing {
		t.Errorf("nil metrics spacing = %v", got)
	}
}

func TestDegenerateAndEmpty(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	// Size 0: advances/bounds zero, metrics at scale 1 (the computeMatrices singular-matrix rule).
	f := NewFont(tf, 0, 1, 0)
	var m Metrics
	spacing := f.Metrics(&m)
	if !near(spacing, 1.171875, 1e-4) || !near(m.Ascent, -0.9277344, 1e-4) {
		t.Errorf("size-0 metrics: spacing=%v ascent=%v", spacing, m.Ascent)
	}
	var bounds geom.Rect
	if w := f.MeasureText([]byte("Hxg!"), TextEncodingUTF8, &bounds, nil); w != 0 || !bounds.IsEmpty() {
		t.Errorf("size-0 measure = %v %v", w, bounds)
	}
	// Negative sizes clamp to 0.
	fn := NewFont(tf, -10, 1, 0)
	widths := make([]float32, 1)
	fn.GlyphWidths([]uint16{44}, widths)
	if widths[0] != 0 {
		t.Errorf("negative-size width = %v", widths[0])
	}
	// The empty typeface: no glyphs, zero metrics, but text still counts.
	fe := NewFont(nil, 20, 1, 0)
	if fe.Typeface() != EmptyTypeface() {
		t.Error("nil typeface should become the empty typeface")
	}
	// Its metrics are all zero, and the four bounds fields are flagged as the non-answers they are (see
	// MetricsFlagBoundsInvalid).
	if got := fe.Metrics(&m); got != 0 || m != (Metrics{Flags: MetricsFlagBoundsInvalid}) {
		t.Errorf("empty metrics = %v %+v", got, m)
	}
	if n := fe.TextToGlyphs([]byte("Hxg!"), TextEncodingUTF8, nil); n != 4 {
		t.Errorf("empty count = %d", n)
	}
	if g := fe.UnicharToGlyph('x'); g != 0 {
		t.Errorf("empty glyph = %d", g)
	}
	if !EmptyTypeface().IsFixedPitch() {
		t.Error("empty typeface should be fixed pitch")
	}
	if EmptyTypeface().Style() != NormalStyle() {
		t.Error("empty typeface style should be normal")
	}
}

func TestNaNSizeClampsToZero(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	nan := float32(math.NaN())
	// Go's max propagates NaN, so validSize cannot be written with it.
	if got := NewFont(tf, nan, 1, 0).Size(); got != 0 {
		t.Errorf("NewFont(NaN).Size() = %v, want 0", got)
	}
	f := NewFont(tf, 24, 1, 0)
	f.SetSize(nan)
	if got := f.Size(); got != 0 {
		t.Errorf("SetSize(NaN) then Size() = %v, want 0", got)
	}
	// Finite sizes are untouched, and negatives still clamp.
	f.SetSize(24)
	if got := f.Size(); got != 24 {
		t.Errorf("SetSize(24) then Size() = %v, want 24", got)
	}
	f.SetSize(-10)
	if got := f.Size(); got != 0 {
		t.Errorf("SetSize(-10) then Size() = %v, want 0", got)
	}
}

func TestStylePacking(t *testing.T) {
	s := NewStyle(WeightBold, WidthCondensed, SlantItalic)
	if s.Weight() != 700 || s.Width() != 3 || s.Slant() != SlantItalic {
		t.Errorf("style = (%d,%d,%d)", s.Weight(), s.Width(), s.Slant())
	}
	// Pins.
	s = NewStyle(2000, 100, Slant(9))
	if s.Weight() != 1000 || s.Width() != 9 || s.Slant() != SlantOblique {
		t.Errorf("pinned = (%d,%d,%d)", s.Weight(), s.Width(), s.Slant())
	}
	s = NewStyle(-5, -5, Slant(-1))
	if s.Weight() != 0 || s.Width() != 1 || s.Slant() != SlantUpright {
		t.Errorf("pinned low = (%d,%d,%d)", s.Weight(), s.Width(), s.Slant())
	}
	if int32(NormalStyle()) != 400+5<<16 {
		t.Errorf("packing = %#x", int32(NormalStyle()))
	}
}

// TestNewFontDefaults pins the constructor's documented defaults, including the absence of a default size. NewFont's
// size argument is the size: nothing substitutes a nominal one for a zero, so a caller who passes zero expecting a
// documented default gets a font whose every measurement is zero, and that has to be visible here rather than promised
// away in a comment.
func TestNewFontDefaults(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 0, 1, 0)
	if f.Size() != 0 {
		t.Errorf("NewFont(…, 0, …).Size() = %v, want 0 — there is no default size", f.Size())
	}
	var bounds geom.Rect
	if width := f.MeasureText([]byte("Ag"), TextEncodingUTF8, &bounds, nil); width != 0 || !bounds.IsEmpty() {
		t.Errorf("zero-size MeasureText = %v with bounds %v, want 0 and empty", width, bounds)
	}
	widths := make([]float32, 1)
	f.GlyphWidths([]uint16{tf.UnicharToGlyph('A')}, widths)
	if widths[0] != 0 {
		t.Errorf("zero-size GlyphWidths = %v, want 0", widths[0])
	}
	// The defaults that do exist.
	if !f.BaselineSnap() {
		t.Error("baseline snapping should default on")
	}
	if got := f.Edging(); got != EdgingAntiAlias {
		t.Errorf("edging defaults to %d, want EdgingAntiAlias", got)
	}
	if got := f.Hinting(); got != HintingNormal {
		t.Errorf("hinting defaults to %d, want HintingNormal", got)
	}
	if f.ForceAutoHinting() || f.Subpixel() || f.LinearMetrics() || f.EmbeddedBitmaps() || f.Embolden() {
		t.Errorf("every other request should default off: flags = %#x", f.flags)
	}
}

// TestEdgingValidation pins SetEdging's fold of values outside the Edging set. Edging promises one edge treatment at
// every size, but the lanes read the recorded value through different tests: the mask lane sets recFlagAliased for
// exactly EdgingAlias, while the path and distance-field lanes take their anti-aliasing from HasSomeAntiAliasing, true
// for exactly the two AA values. A value in neither set therefore draws anti-aliased through the mask lane and
// hard-edged once the size crosses ShouldDrawAsPathMatrix into the path lane, which is the promise broken.
func TestEdgingValidation(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	f := NewFont(tf, 20, 1, 0)
	identity := geom.IdentityMatrix()
	aaRec, _ := MakeRecAndEffects(f, nil, &identity, nil) // the default edging, for the folded cases to land on
	for _, c := range []struct {
		name string
		set  Edging
		want Edging
	}{
		{name: "alias", set: EdgingAlias, want: EdgingAlias},
		{name: "anti-alias", set: EdgingAntiAlias, want: EdgingAntiAlias},
		{name: "subpixel anti-alias", set: EdgingSubpixelAntiAlias, want: EdgingSubpixelAntiAlias},
		{name: "one past the set", set: Edging(3), want: EdgingAntiAlias},
		{name: "far outside the set", set: Edging(255), want: EdgingAntiAlias},
	} {
		t.Run(c.name, func(t *testing.T) {
			f.SetEdging(c.set)
			if got := f.Edging(); got != c.want {
				t.Errorf("SetEdging(%d) recorded %d, want %d", c.set, got, c.want)
			}
			rec, _ := MakeRecAndEffects(f, nil, &identity, nil)
			if aliased := rec.Flags&recFlagAliased != 0; aliased == f.HasSomeAntiAliasing() {
				t.Errorf("the mask lane's aliased flag (%v) and HasSomeAntiAliasing (%v) agree, so this edging "+
					"renders one way through the mask lane and the other through the path lane", aliased,
					f.HasSomeAntiAliasing())
			}
			if c.set > EdgingSubpixelAntiAlias && rec != aaRec {
				t.Error("a folded edging must produce the default edging's rec, and therefore its strike")
			}
		})
	}
}

func TestLinearMetricsFlag(t *testing.T) {
	f := NewFont(loadTypeface(t, "Roboto-Regular.ttf", 0), 12, 1, 0)
	if f.LinearMetrics() {
		t.Error("linear metrics should default to off")
	}
	identity := geom.IdentityMatrix()
	rec, _ := MakeRecAndEffects(f, nil, &identity, nil)
	if rec.Flags&recFlagLinearMetrics != 0 {
		t.Error("rec carries the linear-metrics flag with the request off")
	}
	f.SetLinearMetrics(true)
	if !f.LinearMetrics() {
		t.Error("SetLinearMetrics(true) did not take")
	}
	// The request reaches the scaler rec, so it keys the strike.
	linear, _ := MakeRecAndEffects(f, nil, &identity, nil)
	if linear.Flags&recFlagLinearMetrics == 0 {
		t.Error("rec is missing the linear-metrics flag with the request on")
	}
	if linear == rec {
		t.Error("linear-metrics recs must not collapse onto the default rec")
	}
	// setupForAsPaths keeps it (path measurement is already unhinted and linear); only these two are cleared.
	f.SetForceAutoHinting(true)
	pathFont := *f
	pathFont.setupForAsPaths(nil)
	if !pathFont.LinearMetrics() || pathFont.ForceAutoHinting() {
		t.Errorf("setupForAsPaths flags = %#x", pathFont.flags)
	}
	f.SetLinearMetrics(false)
	if f.LinearMetrics() {
		t.Error("SetLinearMetrics(false) did not take")
	}
}

func TestEmboldenAndEmbeddedBitmapFlags(t *testing.T) {
	f := NewFont(loadTypeface(t, "Roboto-Regular.ttf", 0), 12, 1, 0)
	if f.Embolden() || f.EmbeddedBitmaps() {
		t.Errorf("embolden and embedded bitmaps should default to off: flags = %#x", f.flags)
	}
	identity := geom.IdentityMatrix()
	rec, _ := MakeRecAndEffects(f, nil, &identity, nil)
	f.SetEmbolden(true)
	f.SetEmbeddedBitmaps(true)
	if !f.Embolden() || !f.EmbeddedBitmaps() {
		t.Fatalf("the setters did not take: flags = %#x", f.flags)
	}
	// Neither request has a lane in the scaler (no synthetic-bold generator, and bitmap strikes are decoded whenever
	// the typeface carries them), so neither may key a strike: the rec must come out identical.
	if both, _ := MakeRecAndEffects(f, nil, &identity, nil); both != rec {
		t.Error("a request the scaler cannot honor reached the rec and would fragment the strike cache")
	}
	// Only the embedded-bitmap request is in setupForAsPaths's ignore mask (upstream's flagsToIgnore covers it and
	// force-auto-hinting, nothing else).
	pathFont := *f
	pathFont.setupForAsPaths(nil)
	if pathFont.EmbeddedBitmaps() {
		t.Error("setupForAsPaths kept the embedded-bitmap request")
	}
	if !pathFont.Embolden() {
		t.Error("setupForAsPaths cleared the embolden request")
	}
	f.SetEmbolden(false)
	f.SetEmbeddedBitmaps(false)
	if f.Embolden() || f.EmbeddedBitmaps() {
		t.Errorf("clearing the requests did not take: flags = %#x", f.flags)
	}
}

func TestBaselineSnapFlag(t *testing.T) {
	f := NewFont(loadTypeface(t, "Roboto-Regular.ttf", 0), 12, 1, 0)
	if !f.BaselineSnap() {
		t.Errorf("baseline snapping should default to on: flags = %#x", f.flags)
	}
	f.SetSubpixel(true) // the axis alignment only changes the rounding of a subpixel-positioned strike
	identity := geom.IdentityMatrix()
	snapped, _ := MakeRecAndEffects(f, nil, &identity, nil)
	if snapped.Flags&recFlagBaselineSnap == 0 {
		t.Error("rec is missing the baseline-snap flag with the request on")
	}
	if got := snapped.computeAxisAlignmentForHText(); got != AxisAlignmentX {
		t.Errorf("snapped axis alignment = %d, want AxisAlignmentX", got)
	}
	// setupForAsPaths leaves the request alone in both directions (its ignore mask covers only embedded bitmaps and
	// force-auto-hinting). Pinning it only from off would let flagBaselineSnap join flagsToIgnore and silently disable
	// baseline snapping for every path-drawn run, so the on case has to be checked while the request is still on.
	snapPathFont := *f
	snapPathFont.setupForAsPaths(nil)
	if !snapPathFont.BaselineSnap() {
		t.Error("setupForAsPaths cleared baseline snapping")
	}
	pathRec, _ := MakeRecAndEffects(&snapPathFont, nil, &identity, nil)
	if pathRec.Flags&recFlagBaselineSnap == 0 {
		t.Error("the path-lane rec is missing the baseline-snap flag with the request on")
	}

	f.SetBaselineSnap(false)
	if f.BaselineSnap() {
		t.Error("SetBaselineSnap(false) did not take")
	}
	off, _ := MakeRecAndEffects(f, nil, &identity, nil)
	if off.Flags&recFlagBaselineSnap != 0 {
		t.Error("rec carries the baseline-snap flag with the request off")
	}
	if off == snapped {
		t.Error("unsnapped recs must not collapse onto the snapped rec")
	}
	// The request is honored: with no snapping axis, the strike's rounding keeps sub-pixel bits on both axes instead of
	// rounding y to whole pixels.
	if got := off.computeAxisAlignmentForHText(); got != AxisAlignmentNone {
		t.Errorf("unsnapped axis alignment = %d, want AxisAlignmentNone", got)
	}
	snapRound := NewRoundingSpec(snapped.isSubpixel(), snapped.computeAxisAlignmentForHText())
	offRound := NewRoundingSpec(off.isSubpixel(), off.computeAxisAlignmentForHText())
	if snapRound.HalfAxisSampleFreq.Y != 0.5 || offRound.HalfAxisSampleFreq.Y != SubpixelRound {
		t.Errorf("y rounding: snapped = %v, unsnapped = %v", snapRound.HalfAxisSampleFreq.Y,
			offRound.HalfAxisSampleFreq.Y)
	}
	if snapRound.IgnorePositionFieldMask.Y != 0 || offRound.IgnorePositionFieldMask.Y == 0 {
		t.Errorf("y sub-pixel field mask: snapped = %#x, unsnapped = %#x", snapRound.IgnorePositionFieldMask.Y,
			offRound.IgnorePositionFieldMask.Y)
	}
	pathFont := *f
	pathFont.setupForAsPaths(nil)
	if pathFont.BaselineSnap() {
		t.Error("setupForAsPaths re-enabled baseline snapping")
	}
	f.SetBaselineSnap(true)
	if !f.BaselineSnap() {
		t.Error("SetBaselineSnap(true) did not take")
	}
}

func TestMetricsBoundsInvalidFlag(t *testing.T) {
	var m Metrics
	NewFont(loadTypeface(t, "Roboto-Regular.ttf", 0), 100, 1, 0).Metrics(&m)
	if m.Flags&MetricsFlagBoundsInvalid != 0 {
		t.Errorf("a font with a head bbox reported invalid bounds: flags = %#x", m.Flags)
	}
	// The empty typeface (and any font the metrics recipe cannot run on) populates no bounds at all, so the four bounds
	// fields are zero because nothing was read — not because the font reported an empty box.
	NewFont(nil, 20, 1, 0).Metrics(&m)
	if m.Flags&MetricsFlagBoundsInvalid == 0 {
		t.Errorf("the empty typeface reported valid bounds: flags = %#x", m.Flags)
	}
	// The other lane into that early return: a font with no hhea table, where ascent/descent/leading have no source.
	tf, err := NewTypefaceFromData(sfntWithoutTables(t, readTestFont(t, "Roboto-Regular.ttf"), "hhea"), 0)
	if err != nil {
		t.Fatal(err)
	}
	if tf.hhea != nil {
		t.Fatal("the hhea table survived the strip")
	}
	if spacing := NewFont(tf, 100, 1, 0).Metrics(&m); spacing != 0 || m != (Metrics{Flags: MetricsFlagBoundsInvalid}) {
		t.Errorf("hhea-less metrics = %v %+v", spacing, m)
	}
}

func TestUnderlineMetricsFlagsFollowThePostTable(t *testing.T) {
	const underlineFlags = MetricsFlagUnderlineThicknessIsValid | MetricsFlagUnderlinePositionIsValid
	data := readTestFont(t, "Roboto-Regular.ttf")
	withPost, err := NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	var m Metrics
	NewFont(withPost, 100, 1, 0).Metrics(&m)
	if m.Flags&underlineFlags != underlineFlags {
		t.Errorf("flags = %#x: a font with a post table must report both underline metrics valid", m.Flags)
	}
	if m.UnderlineThickness == 0 || m.UnderlinePosition == 0 {
		t.Fatalf("underline = %v/%v, want the post table's values", m.UnderlineThickness, m.UnderlinePosition)
	}
	// With no post table there is nowhere for the underline geometry to come from, so the flags a consumer honors must
	// not claim a zero-thickness underline on the baseline is what the font asked for.
	noPost, err := NewTypefaceFromData(sfntWithoutTables(t, data, "post"), 0)
	if err != nil {
		t.Fatal(err)
	}
	if noPost.post != nil {
		t.Fatal("the post table survived the strip")
	}
	var stripped Metrics
	NewFont(noPost, 100, 1, 0).Metrics(&stripped)
	if stripped.Flags&underlineFlags != 0 {
		t.Errorf("flags = %#x: a font with no post table reported valid underline metrics", stripped.Flags)
	}
	if stripped.UnderlineThickness != 0 || stripped.UnderlinePosition != 0 {
		t.Errorf("underline = %v/%v, want zeroes", stripped.UnderlineThickness, stripped.UnderlinePosition)
	}
	// Only the underline flags move: the strikeout pair still comes from OS/2, and the rest of the metrics stay put.
	const strikeoutFlags = MetricsFlagStrikeoutThicknessIsValid | MetricsFlagStrikeoutPositionIsValid
	if stripped.Flags != strikeoutFlags {
		t.Errorf("flags = %#x, want the strikeout pair alone (%#x)", stripped.Flags, strikeoutFlags)
	}
	if stripped.Ascent != m.Ascent || stripped.Top != m.Top || stripped.StrikeoutThickness != m.StrikeoutThickness {
		t.Errorf("stripping post changed unrelated metrics: %+v", stripped)
	}
}

func TestDrawTextPositionsShortOutput(t *testing.T) {
	f := NewFont(loadTypeface(t, "Roboto-Regular.ttf", 0), 20, 1, 0)
	glyphs := []uint16{44, 45, 46}
	full := make([]geom.Point, len(glyphs))
	DrawTextPositions(f, glyphs, geom.Pt(3, 7), full)
	if full[0] != geom.Pt(3, 7) || full[1].X <= full[0].X || full[2].X <= full[1].X || full[1].Y != 7 {
		t.Fatalf("positions = %v", full)
	}
	// A caller sizing out by capacity alone (or otherwise supplying a short slice) gets the origins that fit rather
	// than a panic.
	short := make([]geom.Point, 0, len(glyphs))
	DrawTextPositions(f, glyphs, geom.Pt(3, 7), short)
	partial := make([]geom.Point, 2)
	DrawTextPositions(f, glyphs, geom.Pt(3, 7), partial)
	if partial[0] != full[0] || partial[1] != full[1] {
		t.Errorf("partial = %v, want the first 2 of %v", partial, full)
	}
	// An over-long out is left alone past the glyph count.
	long := make([]geom.Point, len(glyphs)+2)
	DrawTextPositions(f, glyphs, geom.Pt(3, 7), long)
	if long[len(glyphs)] != (geom.Point{}) || long[len(glyphs)+1] != (geom.Point{}) {
		t.Errorf("long tail written: %v", long)
	}
}
