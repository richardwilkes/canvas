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
	"math"
	"slices"
	"testing"

	"github.com/go-text/typesetting/font/opentype"

	"github.com/richardwilkes/canvas/font/internal/tthint"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// hintedTestGlyph returns the mask of 'H' from the test font at 16 pixels under the given hinting level and matrix,
// with linear metrics.
func hintedTestGlyph(t *testing.T, cache *StrikeCache, level Hinting, matrix *geom.Matrix, skewX float32) *Glyph {
	t.Helper()
	return hintedTestGlyphWithMetrics(t, cache, level, matrix, skewX, true)
}

// hintedTestGlyphWithMetrics is hintedTestGlyph with the choice of linear metrics.
func hintedTestGlyphWithMetrics(t *testing.T, cache *StrikeCache, level Hinting, matrix *geom.Matrix, skewX float32,
	linear bool,
) *Glyph {
	t.Helper()
	f := loadMigratedTestFont(t, 16)
	f.SetHinting(level)
	f.SetSkewX(skewX)
	f.SetLinearMetrics(linear)
	gid := f.Typeface().UnicharToGlyph('H')
	if gid == 0 {
		t.Fatal("test font does not map 'H'")
	}
	rec, effects := MakeRecAndEffects(f, nil, matrix, nil)
	spec := StrikeSpec{Typeface: f.Typeface(), Rec: rec, Effects: effects}
	g := cache.FindOrCreateStrike(&spec).PrepareImage(PackGlyphID(gid))
	if g == nil || g.IsEmpty() || g.Image == nil {
		t.Fatal("no glyph mask produced")
	}
	return g
}

// rowInk returns the summed coverage of each row of an A8 mask.
func rowInk(g *Glyph) []int {
	rows := make([]int, g.Height)
	rb := int(g.RowBytes())
	for y := range rows {
		for _, v := range g.Image[y*rb : y*rb+int(g.Width)] {
			rows[y] += int(v)
		}
	}
	return rows
}

// TestFullHinting checks that hinting puts the cap height of an 'H' on a row boundary, without changing its advance
// when metrics are linear, and
// that a rotated or slanted strike is left unhinted.
func TestFullHinting(t *testing.T) {
	cache := NewStrikeCache()
	identity := geom.IdentityMatrix()
	plain := hintedTestGlyph(t, cache, HintingNormal, &identity, 0)
	hinted := hintedTestGlyph(t, cache, HintingFull, &identity, 0)
	if plain.AdvanceX != hinted.AdvanceX {
		t.Errorf("hinting changed the advance from %v to %v", plain.AdvanceX, hinted.AdvanceX)
	}
	plainRows, hintedRows := rowInk(plain), rowInk(hinted)
	if len(hintedRows) < 4 || len(plainRows) < 4 {
		t.Fatalf("masks too small to compare: %d and %d rows", len(plainRows), len(hintedRows))
	}
	// Row 2 runs through the two stems in both masks.
	if top, stems := hintedRows[0], hintedRows[2]; top*10 < stems*9 {
		t.Errorf("hinted top row holds %d against %d in the stems: the cap height is not on a row boundary", top, stems)
	}
	if top, stems := plainRows[0], plainRows[2]; top*10 >= stems*9 {
		t.Errorf("unhinted top row holds %d against %d in the stems: the test glyph no longer shows the difference",
			top, stems)
	}

	same := func(a, b *Glyph) bool {
		return a.Width == b.Width && a.Height == b.Height && a.Left == b.Left && a.Top == b.Top &&
			slices.Equal(a.Image, b.Image)
	}
	rotated := geom.RotateDegMatrix(30)
	if !same(hintedTestGlyph(t, cache, HintingNormal, &rotated, 0), hintedTestGlyph(t, cache, HintingFull, &rotated, 0)) {
		t.Error("a rotated strike was hinted")
	}
	slantedPlain := hintedTestGlyph(t, cache, HintingNormal, &identity, -0.2)
	if !same(slantedPlain, hintedTestGlyph(t, cache, HintingFull, &identity, -0.2)) {
		t.Error("a slanted strike was hinted")
	}
}

// TestHintedContourReconstruction checks that every hinted glyph has a path and that it stays within the glyph's hinted
// control box, which a path built from another glyph's points, or from points in the wrong space, would not. The walk
// that turns the points into segments is checked by TestAppendQuadraticContour.
func TestHintedContourReconstruction(t *testing.T) {
	f := loadMigratedTestFont(t, 16)
	f.SetHinting(HintingFull)
	identity := geom.IdentityMatrix()
	rec, effects := MakeRecAndEffects(f, nil, &identity, nil)
	ctx := NewScalerContext(f.Typeface(), rec, effects)
	if ctx.hinter == nil {
		t.Fatal("the test font was not prepared for hinting")
	}
	checked := 0
	for gid := range uint16(f.Typeface().nGlyphs) {
		bounds, ok := ctx.hintedBounds(gid)
		if !ok {
			continue
		}
		p := ctx.hintedPath(gid)
		if p == nil {
			t.Fatalf("glyph %d has hinted bounds but no hinted path", gid)
		}
		got := p.Bounds()
		const slack = 1.0 / 64
		if got.Left < bounds.Left-slack || got.Top < bounds.Top-slack || got.Right > bounds.Right+slack ||
			got.Bottom > bounds.Bottom+slack {
			t.Errorf("glyph %d: path bounds %v outside the outline's %v", gid, got, bounds)
		}
		checked++
	}
	if checked < 100 {
		t.Errorf("only %d glyphs were hinted", checked)
	}
}

// TestAppendQuadraticContour checks the walk from a TrueType contour to path segments, segment by segment, for each way
// a contour can begin: on the curve, off it with its last point on, and with no point on the curve at all.
func TestAppendQuadraticContour(t *testing.T) {
	pt := func(x, y int32) tthint.Point { return tthint.Point{X: x * 64, Y: -y * 64} }
	for _, c := range []struct {
		name    string
		points  []tthint.Point
		onCurve []bool
		want    []geom.Point // the move, then each segment's points
		verbs   int          // the move, the segments and the close
	}{
		{
			name:    "lines",
			points:  []tthint.Point{pt(0, 0), pt(4, 0), pt(4, 4)},
			onCurve: []bool{true, true, true},
			want:    []geom.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}},
			verbs:   4,
		},
		{
			name:    "curves that close on the starting point",
			points:  []tthint.Point{pt(0, 0), pt(4, 0), pt(4, 4), pt(0, 4)},
			onCurve: []bool{true, false, true, false},
			want:    []geom.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}, {X: 0, Y: 0}},
			verbs:   4,
		},
		{
			name:    "an implied point between two control points",
			points:  []tthint.Point{pt(0, 0), pt(4, 0), pt(4, 4), pt(0, 4)},
			onCurve: []bool{true, false, false, true},
			want:    []geom.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 2}, {X: 4, Y: 4}, {X: 0, Y: 4}},
			verbs:   4,
		},
		{
			name:    "starting off the curve, from the last point",
			points:  []tthint.Point{pt(4, 0), pt(4, 4), pt(0, 0)},
			onCurve: []bool{false, true, true},
			want:    []geom.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}},
			verbs:   3,
		},
		{
			name:    "no point on the curve",
			points:  []tthint.Point{pt(0, 0), pt(4, 0), pt(4, 4), pt(0, 4)},
			onCurve: []bool{false, false, false, false},
			want: []geom.Point{
				{X: 0, Y: 2},
				{X: 0, Y: 0},
				{X: 2, Y: 0},
				{X: 4, Y: 0},
				{X: 4, Y: 2},
				{X: 4, Y: 4},
				{X: 2, Y: 4},
				{X: 0, Y: 4},
				{X: 0, Y: 2},
			},
			verbs: 6,
		},
	} {
		p := path.New()
		appendQuadraticContour(p, c.points, c.onCurve)
		got := make([]geom.Point, p.CountPoints())
		p.Points(got)
		if !slices.Equal(got, c.want) {
			t.Errorf("%s: points %v, want %v", c.name, got, c.want)
		}
		if p.CountVerbs() != c.verbs {
			t.Errorf("%s: %d verbs, want %d", c.name, p.CountVerbs(), c.verbs)
		}
	}
}

// TestHintedStrikeKeyFollowsWhatIsHinted checks that asking for hinting marks a strike only where it will be hinted:
// a rotated strike, or a size the font turns its instructions off at, draws the same masks either way and must not be
// kept twice.
func TestHintedStrikeKeyFollowsWhatIsHinted(t *testing.T) {
	identity := geom.IdentityMatrix()
	rotated := geom.RotateDegMatrix(30)
	for _, c := range []struct {
		matrix *geom.Matrix
		name   string
		size   float32
		hinted bool
	}{
		{name: "upright", matrix: &identity, size: 16, hinted: true},
		{name: "rotated", matrix: &rotated, size: 16},
		{name: "a size the font does not hint at", matrix: &identity, size: 7},
	} {
		f := loadMigratedTestFont(t, c.size)
		plain, _ := MakeRecAndEffects(f, nil, c.matrix, nil)
		f.SetHinting(HintingFull)
		full, _ := MakeRecAndEffects(f, nil, c.matrix, nil)
		if got := full.Flags&recFlagHinted != 0; got != c.hinted {
			t.Errorf("%s: hinted flag %v, want %v", c.name, got, c.hinted)
		}
		if !c.hinted && full != plain {
			t.Errorf("%s: asking for hinting changed the strike key of a strike it does not change", c.name)
		}
	}
}

// TestMeasuredBoundsCoverHintedInk checks that the bounds a caller measures hold what is drawn: with HintingFull the
// mask comes from the hinted outline, whose edges hinting has moved off the design extents.
func TestMeasuredBoundsCoverHintedInk(t *testing.T) {
	cache := NewStrikeCache()
	identity := geom.IdentityMatrix()
	for _, size := range []float32{11, 12.5, 16} {
		f := loadMigratedTestFont(t, size)
		f.SetHinting(HintingFull)
		rec, effects := MakeRecAndEffects(f, nil, &identity, nil)
		if rec.Flags&recFlagHinted == 0 {
			t.Fatalf("size %v: the test font is not hinted", size)
		}
		strike := cache.FindOrCreateStrike(&StrikeSpec{Typeface: f.Typeface(), Rec: rec, Effects: effects})
		unhinted := loadMigratedTestFont(t, size)
		checked, moved := 0, 0
		for gid := range uint16(f.Typeface().nGlyphs) {
			g := strike.PrepareImage(PackGlyphID(gid))
			if g == nil || g.IsEmpty() {
				continue
			}
			var measured, design [1]geom.Rect
			f.GlyphBounds([]uint16{gid}, measured[:])
			unhinted.GlyphBounds([]uint16{gid}, design[:])
			ink := geom.RectLTRB(float32(g.Left), float32(g.Top), float32(g.Left+g.Width), float32(g.Top+g.Height))
			if b := measured[0]; ink.Left < b.Left || ink.Top < b.Top || ink.Right > b.Right || ink.Bottom > b.Bottom {
				t.Errorf("size %v, glyph %d: mask %v outside the measured bounds %v", size, gid, ink, b)
			}
			if measured[0] != design[0] {
				moved++
			}
			checked++
		}
		if checked < 100 || moved == 0 {
			t.Errorf("size %v: %d glyphs checked, %d with bounds hinting moved", size, checked, moved)
		}
	}
}

// TestHintedAdvances checks the advances of hinted glyphs without linear metrics, the default: each is a whole number
// of pixels, the one the hinter reports for the size the glyphs were fitted to, and measuring reports what is drawn.
// With linear metrics, or without hinting, the advance is the design one at the requested size.
func TestHintedAdvances(t *testing.T) {
	cache := NewStrikeCache()
	identity := geom.IdentityMatrix()
	// The test font asks for whole sizes, so at 12.5 its glyphs are fitted for 13.
	const size, fitted = 12.5, 13
	f := loadMigratedTestFont(t, size)
	tf := f.Typeface()
	gids := []uint16{tf.UnicharToGlyph('H'), tf.UnicharToGlyph('i'), tf.UnicharToGlyph('m'), tf.UnicharToGlyph(' ')}
	design := func(gid uint16, ppem float32) float32 {
		return tf.faceHAdvance(opentype.GID(gid)) / float32(tf.upem) * ppem
	}
	drawn := func(gid uint16) float32 {
		rec, effects := MakeRecAndEffects(f, nil, &identity, nil)
		strike := cache.FindOrCreateStrike(&StrikeSpec{Typeface: tf, Rec: rec, Effects: effects})
		return strike.Metrics([]uint16{gid}, nil)[0].AdvanceX
	}
	measured := func(gid uint16) float32 {
		var width [1]float32
		f.GlyphWidths([]uint16{gid}, width[:])
		// Text is laid out for drawing with the advance it is measured with.
		var positions [2]geom.Point
		DrawTextPositions(f, []uint16{gid, gid}, geom.Point{}, positions[:])
		if positions[1].X != width[0] {
			t.Errorf("glyph %d: positioned for drawing with an advance of %v, measured as %v", gid, positions[1].X, width[0])
		}
		return width[0]
	}
	for _, gid := range gids {
		f.SetHinting(HintingFull)
		f.SetLinearMetrics(false)
		want := float32(math.Round(float64(design(gid, fitted))))
		if got := drawn(gid); got != want {
			t.Errorf("glyph %d: hinted advance %v, want %v", gid, got, want)
		}
		if got := measured(gid); got != want {
			t.Errorf("glyph %d: measured hinted advance %v, want %v", gid, got, want)
		}
		linear := design(gid, size)
		f.SetLinearMetrics(true)
		if got := drawn(gid); got != linear {
			t.Errorf("glyph %d: advance with linear metrics %v, want %v", gid, got, linear)
		}
		if got := measured(gid); got != linear {
			t.Errorf("glyph %d: measured advance with linear metrics %v, want %v", gid, got, linear)
		}
		f.SetHinting(HintingNormal)
		f.SetLinearMetrics(false)
		if got := drawn(gid); got != linear {
			t.Errorf("glyph %d: unhinted advance %v, want %v", gid, got, linear)
		}
	}
}
