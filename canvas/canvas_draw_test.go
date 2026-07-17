// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package canvas

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

func at(p *raster.Pixmap, x, y int32) uint32 {
	return p.Pix[int(y)*int(p.RowPixels)+int(x)]
}

func newWhiteCanvasWH(w, h int32) (*Canvas, *raster.Pixmap) {
	pix := raster.NewPixmap(w, h)
	c := NewForPixmap(pix)
	c.Clear(0xFFFFFFFF)
	return c, pix
}

func pixmapsEqual(t *testing.T, a, b *raster.Pixmap, what string) {
	t.Helper()
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			t.Fatalf("%s: pixel (%d,%d): %08x vs %08x", what, int32(i)%a.Width, int32(i)/a.Width, a.Pix[i], b.Pix[i])
		}
	}
}

// TestDrawRectMatchesPathUnderRotation verifies the drawRect path-fallback lane (non-rect-stays-rect CTM) is identical
// to drawing the equivalent rect path.
func TestDrawRectMatchesPathUnderRotation(t *testing.T) {
	r := geom.RectLTRB(10, 10, 60, 40)
	for _, aa := range []bool{false, true} {
		viaRect, pixA := newWhiteCanvasWH(100, 80)
		viaPath, pixB := newWhiteCanvasWH(100, 80)

		var rot geom.Matrix
		rot.SetRotatePivot(30, 50, 40)
		viaRect.Concat(&rot)
		viaPath.Concat(&rot)

		paint := NewPaint()
		paint.Color = colorcore.Color(0xFF224466)
		paint.AntiAlias = aa

		viaRect.DrawRect(r, paint)
		p := &path.Path{}
		p.AddRect(r, geom.DirectionCW)
		viaPath.DrawPath(p, paint)

		pixmapsEqual(t, pixA, pixB, "rotated drawRect vs path")
	}
}

// TestDrawOvalMatchesPath verifies drawOval lowers to exactly the oval path fill.
func TestDrawOvalMatchesPath(t *testing.T) {
	oval := geom.RectLTRB(8.5, 6.25, 70.75, 50.5)
	viaOval, pixA := newWhiteCanvasWH(90, 60)
	viaPath, pixB := newWhiteCanvasWH(90, 60)

	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF663311)
	paint.AntiAlias = true

	viaOval.DrawOval(oval, paint)
	p := &path.Path{}
	p.AddOval(oval, geom.DirectionCW)
	viaPath.DrawPath(p, paint)

	pixmapsEqual(t, pixA, pixB, "drawOval vs oval path")
}

// TestDrawArcMatchesCreateDrawArcPath verifies the drawArc lowering.
func TestDrawArcMatchesCreateDrawArcPath(t *testing.T) {
	oval := geom.RectLTRB(10, 10, 70, 60)
	for _, useCenter := range []bool{false, true} {
		viaArc, pixA := newWhiteCanvasWH(90, 80)
		viaPath, pixB := newWhiteCanvasWH(90, 80)

		paint := NewPaint()
		paint.Color = colorcore.Color(0xFF157015)
		paint.AntiAlias = true

		viaArc.DrawArc(oval, 30, 200, useCenter, paint)
		p := path.CreateDrawArcPath(oval, 30, 200, useCenter, true)
		viaPath.DrawPath(p, paint)

		pixmapsEqual(t, pixA, pixB, "drawArc vs CreateDrawArcPath")
	}
}

// TestDrawRectHairline verifies zero-width stroke rects draw through the hairline converters.
func TestDrawRectHairline(t *testing.T) {
	c, pix := newWhiteCanvasWH(40, 30)
	paint := NewPaint()
	paint.Style = StyleStroke
	paint.Color = colorcore.Color(0xFF000000)
	c.DrawRect(geom.RectLTRB(5, 5, 30, 20), paint)

	if at(pix, 5, 5) != 0xFF000000 {
		t.Fatalf("hairline rect corner not drawn")
	}
	if at(pix, 15, 12) != 0xFFFFFFFF {
		t.Fatalf("hairline rect interior filled")
	}
}

// TestDrawRectStrokeFrame verifies integer-width miter stroke rects draw via FrameRect (interior untouched, band
// covered).
func TestDrawRectStrokeFrame(t *testing.T) {
	for _, aa := range []bool{false, true} {
		c, pix := newWhiteCanvasWH(50, 40)
		paint := NewPaint()
		paint.Style = StyleStroke
		paint.StrokeWidth = 4
		paint.AntiAlias = aa
		paint.Color = colorcore.Color(0xFF000000)
		c.DrawRect(geom.RectLTRB(10, 10, 40, 30), paint)

		if at(pix, 10, 10) != 0xFF000000 {
			t.Fatalf("aa=%v: stroke band not drawn at corner", aa)
		}
		if at(pix, 25, 20) != 0xFFFFFFFF {
			t.Fatalf("aa=%v: stroke interior filled", aa)
		}
		if at(pix, 10, 5) != 0xFFFFFFFF {
			t.Fatalf("aa=%v: pixels above the outer band drawn", aa)
		}
	}
}

// TestDrawThickStroke verifies non-hairline strokes draw through the stroker: pixels on the line's spine are covered,
// pixels far from the band stay white.
func TestDrawThickStroke(t *testing.T) {
	c, pix := newWhiteCanvasWH(40, 30)
	paint := NewPaint()
	paint.Style = StyleStroke
	paint.StrokeWidth = 5
	paint.Color = colorcore.Color(0xFF000000)
	p := &path.Path{}
	p.MoveTo(5, 5)
	p.LineTo(35, 25)
	c.DrawPath(p, paint)
	// the segment midpoint (20, 15) sits on the stroke's spine
	if at(pix, 20, 15) == 0xFFFFFFFF {
		t.Fatal("stroke spine pixel not drawn")
	}
	// the far corner is well outside the 2.5px band
	if at(pix, 37, 2) != 0xFFFFFFFF {
		t.Fatal("pixel far from the stroke band was drawn")
	}
}

// TestDrawPathHairlineStroke verifies zero-width strokes draw as hairlines.
func TestDrawPathHairlineStroke(t *testing.T) {
	c, pix := newWhiteCanvasWH(40, 30)
	paint := NewPaint()
	paint.Style = StyleStroke
	paint.Color = colorcore.Color(0xFF000000)
	p := &path.Path{}
	p.MoveTo(5, 15)
	p.LineTo(35, 15)
	c.DrawPath(p, paint)
	if at(pix, 20, 15) != 0xFF000000 {
		t.Fatalf("hairline stroke did not draw")
	}
}

// TestDrawThinAAStrokePromotedToHairline verifies the AA thin-stroke promotion modulates alpha.
func TestDrawThinAAStrokePromotedToHairline(t *testing.T) {
	c, pix := newWhiteCanvasWH(40, 30)
	paint := NewPaint()
	paint.Style = StyleStroke
	paint.StrokeWidth = 0.5
	paint.AntiAlias = true
	paint.Color = colorcore.Color(0xFF000000)
	p := &path.Path{}
	p.MoveTo(5, 15.5)
	p.LineTo(35, 15.5)
	c.DrawPath(p, paint)

	v := at(pix, 20, 15)
	if v == 0xFFFFFFFF {
		t.Fatalf("thin AA stroke drew nothing")
	}
	if v == 0xFF000000 {
		t.Fatalf("thin AA stroke drew at full alpha (no coverage modulation)")
	}
}

// TestDrawPointsModes exercises the three point modes.
func TestDrawPointsModes(t *testing.T) {
	pts := []geom.Point{{X: 5.4, Y: 5.6}, {X: 20.2, Y: 8.9}, {X: 12.7, Y: 20.1}}

	// Points mode, hairline: one pixel per point at floor coords.
	c, pix := newWhiteCanvasWH(30, 30)
	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF000000)
	c.DrawPoints(PointModePoints, pts, paint)
	for _, p := range pts {
		if at(pix, int32(p.X), int32(p.Y)) != 0xFF000000 {
			t.Fatalf("point (%v) not drawn", p)
		}
	}

	// Lines mode: only pts[0]-pts[1] (odd count truncates).
	c2, pix2 := newWhiteCanvasWH(30, 30)
	c2.DrawPoints(PointModeLines, pts, paint)
	drawn := 0
	for _, v := range pix2.Pix {
		if v != 0xFFFFFFFF {
			drawn++
		}
	}
	if drawn == 0 {
		t.Fatalf("lines mode drew nothing")
	}
	if at(pix2, 15, 16) != 0xFFFFFFFF {
		t.Fatalf("lines mode drew the second (truncated) segment")
	}

	// Polygon mode: both segments, not closed.
	c3, pix3 := newWhiteCanvasWH(30, 30)
	c3.DrawPoints(PointModePolygon, pts, paint)
	drawn3 := 0
	for _, v := range pix3.Pix {
		if v != 0xFFFFFFFF {
			drawn3++
		}
	}
	if drawn3 <= drawn {
		t.Fatalf("polygon mode should draw more than lines mode (%d vs %d)", drawn3, drawn)
	}
}

// TestDrawPointsSquareWidth verifies wide square-cap points draw as fixed-point squares.
func TestDrawPointsSquareWidth(t *testing.T) {
	c, pix := newWhiteCanvasWH(30, 30)
	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF000000)
	paint.StrokeWidth = 6
	paint.Cap = CapSquare
	c.DrawPoints(PointModePoints, []geom.Point{{X: 15, Y: 15}}, paint)

	if at(pix, 13, 13) != 0xFF000000 || at(pix, 17, 17) != 0xFF000000 {
		t.Fatalf("square point not filled")
	}
	if at(pix, 15, 10) != 0xFFFFFFFF {
		t.Fatalf("square point too large")
	}
}

// TestDrawPointsRoundCapCircle verifies wide round-cap points lower to circle paths.
func TestDrawPointsRoundCapCircle(t *testing.T) {
	viaPoints, pixA := newWhiteCanvasWH(30, 30)
	viaPath, pixB := newWhiteCanvasWH(30, 30)

	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF000000)
	paint.StrokeWidth = 10
	paint.Cap = CapRound
	paint.AntiAlias = true
	viaPoints.DrawPoints(PointModePoints, []geom.Point{{X: 15, Y: 15}}, paint)

	fill := NewPaint()
	fill.Color = colorcore.Color(0xFF000000)
	fill.AntiAlias = true
	circle := &path.Path{}
	circle.AddCircle(0, 0, 5, geom.DirectionCW)
	var pre geom.Matrix
	pre.SetTranslate(15, 15)
	circle.TransformTo(&pre, circle)
	viaPath.DrawPath(circle, fill)

	pixmapsEqual(t, pixA, pixB, "round point vs circle path")
}

///////////////////////////////////////////////////////////////////////////////
// layers

// TestSaveLayerAlphaUniformOverlap is the classic layer test: two overlapping opaque rects in an alpha layer composite
// as a uniform 50% wash (drawing them directly would double-blend the overlap).
func TestSaveLayerAlphaUniformOverlap(t *testing.T) {
	c, pix := newWhiteCanvasWH(60, 40)
	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF336699)

	c.SaveLayerAlpha(nil, 0x80)
	c.DrawRect(geom.RectLTRB(5, 5, 35, 25), paint)
	c.DrawRect(geom.RectLTRB(20, 10, 50, 30), paint)
	c.Restore()

	inFirst := at(pix, 10, 10)
	inOverlap := at(pix, 25, 15)
	inSecond := at(pix, 45, 25)
	if inFirst != inOverlap || inOverlap != inSecond {
		t.Fatalf("layer overlap not uniform: %08x %08x %08x", inFirst, inOverlap, inSecond)
	}
	if inFirst == 0xFFFFFFFF || inFirst == at(pix, 2, 2) && inFirst != 0xFFFFFFFF {
		t.Fatalf("layer did not composite: %08x", inFirst)
	}
}

// TestSaveLayerFullAlphaMatchesDirect verifies an alpha-255 layer is invisible in the output.
func TestSaveLayerFullAlphaMatchesDirect(t *testing.T) {
	layered, pixA := newWhiteCanvasWH(60, 40)
	direct, pixB := newWhiteCanvasWH(60, 40)

	paint := NewPaint()
	paint.Color = colorcore.Color(0x99336699)
	paint.AntiAlias = true

	layered.SaveLayerAlpha(nil, 255)
	layered.DrawRect(geom.RectLTRB(5.5, 5.25, 35.75, 25.5), paint)
	layered.Restore()

	direct.DrawRect(geom.RectLTRB(5.5, 5.25, 35.75, 25.5), paint)

	pixmapsEqual(t, pixA, pixB, "alpha-255 layer vs direct")
}

// TestSaveLayerBoundsClipContent verifies the layer bounds hint hard-clips layer content.
func TestSaveLayerBoundsClipContent(t *testing.T) {
	c, pix := newWhiteCanvasWH(60, 40)
	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF000000)

	bounds := geom.RectLTRB(10, 10, 30, 30)
	c.SaveLayerAlpha(&bounds, 0x80)
	c.DrawRect(geom.RectLTRB(0, 0, 60, 40), paint) // fills the whole layer
	c.Restore()

	if at(pix, 15, 15) == 0xFFFFFFFF {
		t.Fatalf("inside layer bounds not drawn")
	}
	if at(pix, 35, 15) != 0xFFFFFFFF {
		t.Fatalf("outside layer bounds drawn")
	}
}

// TestSaveLayerMatrixAndClipInsideLayer verifies the CTM continues into the layer and clips set after saveLayer
// restrict layer content; the pre-layer clip applies at restore.
func TestSaveLayerMatrixAndClipInsideLayer(t *testing.T) {
	layered, pixA := newWhiteCanvasWH(80, 60)
	direct, pixB := newWhiteCanvasWH(80, 60)

	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF993311)

	// pre-layer clip
	layered.ClipRect(geom.RectLTRB(0, 0, 50, 60), raster.ClipIntersect, false)
	direct.ClipRect(geom.RectLTRB(0, 0, 50, 60), raster.ClipIntersect, false)

	layered.Translate(10, 5)
	direct.Translate(10, 5)

	layered.SaveLayerAlpha(nil, 255)
	layered.ClipRect(geom.RectLTRB(5, 5, 40, 40), raster.ClipIntersect, false)
	layered.DrawRect(geom.RectLTRB(0, 0, 60, 50), paint)
	layered.Restore()

	direct.Save()
	direct.ClipRect(geom.RectLTRB(5, 5, 40, 40), raster.ClipIntersect, false)
	direct.DrawRect(geom.RectLTRB(0, 0, 60, 50), paint)
	direct.Restore()

	pixmapsEqual(t, pixA, pixB, "layer with inner clip vs direct")
}

// TestSaveLayerNothingToDrawPaint verifies a nothing-to-draw layer paint degenerates to save + empty clip: nothing
// inside draws, and the save count balances.
func TestSaveLayerNothingToDrawPaint(t *testing.T) {
	c, pix := newWhiteCanvasWH(40, 30)
	paint := NewPaint()
	paint.Color = colorcore.Color(0x00000000) // alpha 0, src-over: nothing to draw

	before := c.SaveCount()
	c.SaveLayer(nil, paint)
	if !c.IsClipEmpty() {
		t.Fatalf("clip should be empty inside a nothing-to-draw layer")
	}
	fill := NewPaint()
	fill.Color = colorcore.Color(0xFF000000)
	c.DrawPaint(fill)
	c.Restore()
	if c.SaveCount() != before {
		t.Fatalf("save count unbalanced: %d vs %d", c.SaveCount(), before)
	}
	for _, v := range pix.Pix {
		if v != 0xFFFFFFFF {
			t.Fatalf("nothing-to-draw layer leaked a draw")
		}
	}
}

// TestSaveLayerNested verifies nested layers with restoreToCount.
func TestSaveLayerNested(t *testing.T) {
	c, pix := newWhiteCanvasWH(60, 40)
	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF336699)

	innerPaint := NewPaint()
	innerPaint.Color = colorcore.Color(0xFFAA2211)

	count := c.Save()
	c.SaveLayerAlpha(nil, 0x80)
	c.DrawRect(geom.RectLTRB(5, 5, 55, 35), paint)
	c.SaveLayerAlpha(nil, 0x80)
	c.DrawRect(geom.RectLTRB(15, 15, 45, 30), innerPaint)
	c.RestoreToCount(count)

	if c.SaveCount() != 1 {
		t.Fatalf("save count after restoreToCount: %d", c.SaveCount())
	}
	outer := at(pix, 8, 8)
	inner := at(pix, 30, 20)
	if outer == 0xFFFFFFFF || inner == 0xFFFFFFFF {
		t.Fatalf("nested layers did not composite")
	}
	if outer == inner {
		t.Fatalf("inner (double-alpha'd) region should differ from outer: %08x", outer)
	}
}

// TestSaveLayerQuickRejectUsesLayerBounds verifies draws outside a bounded layer are quick-rejected but in-bounds
// content still lands.
func TestSaveLayerQuickRejectUsesLayerBounds(t *testing.T) {
	c, pix := newWhiteCanvasWH(80, 60)
	paint := NewPaint()
	paint.Color = colorcore.Color(0xFF000000)

	bounds := geom.RectLTRB(20, 20, 50, 40)
	c.SaveLayerAlpha(&bounds, 255)
	if !c.QuickReject(geom.RectLTRB(60, 45, 75, 55)) {
		t.Fatalf("rect outside layer bounds not quick-rejected")
	}
	if c.QuickReject(geom.RectLTRB(25, 25, 45, 35)) {
		t.Fatalf("rect inside layer bounds quick-rejected")
	}
	c.DrawRect(geom.RectLTRB(25, 25, 45, 35), paint)
	c.Restore()

	if at(pix, 30, 30) != 0xFF000000 {
		t.Fatalf("in-bounds layer content missing after restore")
	}
}

func TestDrawRectWithLinearGradient(t *testing.T) {
	c, pix := newWhiteCanvasWH(100, 40)
	paint := NewPaint()
	paint.Shader = shaders.NewLinearGradient(geom.Point{X: 0, Y: 0}, geom.Point{X: 100, Y: 0},
		[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, shaders.TileClamp, nil)
	c.DrawRect(geom.RectLTRB(0, 0, 100, 40), paint)

	// Word layout is R|G<<8|B<<16|A<<24; opaque srcover gradients strength-reduce to Src, so the stored word is the
	// rounded shader output: t = (x+0.5)/100.
	if got := at(pix, 0, 20); got != 0xFF0100FE {
		t.Fatalf("left edge: %08x", got)
	}
	if got := at(pix, 99, 20); got != 0xFFFE0001 {
		t.Fatalf("right edge: %08x", got)
	}
	mid := at(pix, 49, 20)
	r, b := mid&0xFF, mid>>16&0xFF
	if r+b != 0xFF || r < 0x7C || r > 0x83 {
		t.Fatalf("midpoint not a red/blue lerp: %08x", mid)
	}
}

func TestDrawWithShaderRespectsCTMAndAlpha(t *testing.T) {
	// A gradient under a CTM translate moves with the geometry, so it must match the same gradient pre-placed with an
	// equivalent local matrix.
	base, pixA := newWhiteCanvasWH(100, 20)
	moved, pixB := newWhiteCanvasWH(100, 20)
	grad := shaders.NewLinearGradient(geom.Point{X: 0, Y: 0}, geom.Point{X: 50, Y: 0},
		[]colorcore.Color{0xFF00FF00, 0xFFFF00FF}, nil, shaders.TileClamp, nil)

	basePaint := NewPaint()
	basePaint.Shader = shaders.NewWithLocalMatrix(grad, geom.TranslateMatrix(20, 0))
	base.DrawRect(geom.RectLTRB(20, 0, 70, 20), basePaint)

	movedPaint := NewPaint()
	movedPaint.Shader = grad
	var tr geom.Matrix
	tr.SetTranslate(20, 0)
	moved.Concat(&tr)
	moved.DrawRect(geom.RectLTRB(0, 0, 50, 20), movedPaint)
	pixmapsEqual(t, pixA, pixB, "shader local matrix vs CTM translate")

	// Paint alpha modulates the shader (scale_1_float): the stored color halves.
	half, pixC := newWhiteCanvasWH(4, 4)
	solid := shaders.NewColor(0xFFFF0000)
	p2 := NewPaint()
	p2.Shader = solid
	p2.Color = 0x80000000
	p2.BlendMode = raster.BlendSrc
	half.DrawRect(geom.RectLTRB(0, 0, 4, 4), p2)
	if got := pixC.Pix[0]; got != 0x80000080 { // R|G<<8|B<<16|A<<24: premul half red
		t.Fatalf("paint alpha over shader: %08x", got)
	}
}

func TestDrawWithDitheredGradient(t *testing.T) {
	// A dithered gradient must stay within ±1 of the undithered rounding and must not be uniform along a column band
	// where the undithered value is constant.
	plain, pixA := newWhiteCanvasWH(64, 8)
	dith, pixB := newWhiteCanvasWH(64, 8)
	grad := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 1024},
		[]colorcore.Color{0xFF000000, 0xFFFFFFFF}, nil, shaders.TileClamp, nil)
	for i, cv := range []*Canvas{plain, dith} {
		p := NewPaint()
		p.Shader = grad
		p.Dither = i == 1
		cv.DrawRect(geom.RectLTRB(0, 0, 64, 8), p)
	}
	varied := false
	for i := range pixA.Pix {
		a := int32(pixA.Pix[i] & 0xFF)
		b := int32(pixB.Pix[i] & 0xFF)
		d := a - b
		if d < -1 || d > 1 {
			t.Fatalf("dither moved pixel %d by %d", i, d)
		}
		if d != 0 {
			varied = true
		}
	}
	if !varied {
		t.Fatal("dither had no effect")
	}
}

func TestDrawWithColorFilterFoldsIntoColor(t *testing.T) {
	// A solid srcover paint with a matrix color filter takes the legacy lane with the filtered color: an
	// identity-with-translate matrix shifts red up by a known amount.
	shift := [20]float32{
		1, 0, 0, 0, 0.5,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
	}
	cv, pix := newWhiteCanvasWH(20, 20)
	p := NewPaint()
	p.Color = 0xFF002040
	p.ColorFilter = colorfilter.NewMatrix(&shift)
	cv.DrawRect(geom.RectLTRB(0, 0, 20, 20), p)
	got := at(pix, 10, 10)
	// r = 0 + 0.5 -> 128 (Sk4f_toL32 rounds 127.5+0.5 up), g/b unchanged, opaque.
	if got != 0xFF402080 { // RGBA word order: A=FF B=40 G=20 R=80
		t.Fatalf("filtered solid = %08x", got)
	}
}

func TestDrawWithColorFilterOnShader(t *testing.T) {
	// A color filter over a constant color shader routes through ColorFilterShader (never the constant collapse) and
	// must equal filtering the equivalent solid paint.
	inv := [20]float32{
		-1, 0, 0, 0, 1,
		0, -1, 0, 0, 1,
		0, 0, -1, 0, 1,
		0, 0, 0, 1, 0,
	}
	viaShader, pixA := newWhiteCanvasWH(20, 20)
	p := NewPaint()
	p.Shader = shaders.NewColor(0xFF3060C0)
	p.ColorFilter = colorfilter.NewMatrix(&inv)
	viaShader.DrawRect(geom.RectLTRB(0, 0, 20, 20), p)

	viaSolid, pixB := newWhiteCanvasWH(20, 20)
	q := NewPaint()
	q.Color = 0xFF3060C0
	q.ColorFilter = colorfilter.NewMatrix(&inv)
	viaSolid.DrawRect(geom.RectLTRB(0, 0, 20, 20), q)
	pixmapsEqual(t, pixA, pixB, "filtered shader vs filtered solid")
}

func TestDrawRectWithBlurMaskFilter(t *testing.T) {
	// A blurred fill must produce coverage outside the rect, keep the center saturated, and be horizontally symmetric
	// about the rect's axis.
	cv, pix := newWhiteCanvasWH(80, 60)
	p := NewPaint()
	p.Color = 0xFF000000
	p.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 4, true)
	cv.DrawRect(geom.RectLTRB(30, 20, 50, 40), p)

	// At sigma 4 the sliding window (23) is wider than the 20px rect, so even the center coverage stays just below
	// saturation.
	if got := at(pix, 40, 30); got&0xFF > 8 || got>>24 != 0xFF {
		t.Fatalf("center = %08x, want nearly opaque black", got)
	}
	if got := at(pix, 26, 30); got == 0xFFFFFFFF {
		t.Fatal("no blur spill outside the rect")
	}
	for x := int32(0); x < 40; x++ {
		if at(pix, x, 30) != at(pix, 79-x, 30) {
			// The rect spans [30,50) on an 80-wide surface, so row 30 is symmetric about x=39.5.
			if at(pix, x, 30) != at(pix, 79-x, 30) {
				t.Fatalf("asymmetric blur at x=%d", x)
			}
		}
	}
}

func TestDrawBlurredRRectMatchesPathLane(t *testing.T) {
	// The rrect nine-patch fast path must agree with the same geometry pushed through the generic filterPath lane (an
	// added degenerate contour defeats the rect/rrect detection without changing the fill).
	viaRRect, pixA := newWhiteCanvasWH(120, 90)
	p := NewPaint()
	p.Color = 0xFF204080
	p.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 3, true)
	viaRRect.DrawRoundRect(geom.RectLTRB(20, 15, 100, 75), 10, 10, p)

	viaPath, pixB := newWhiteCanvasWH(120, 90)
	rr := &path.Path{}
	rr.AddRRect(geom.MakeRRect(geom.RectLTRB(20, 15, 100, 75), 10, 10), geom.DirectionCW)
	rr.MoveTo(60, 45)
	rr.Close()
	viaPath.DrawPath(rr, p)

	// The nine-patch stretches a small blurred patch, so the stretched seams may differ by a hair from the full-mask
	// blur; require per-channel agreement within ±1.
	for i := range pixA.Pix {
		a := pixA.Pix[i]
		b := pixB.Pix[i]
		for shift := 0; shift < 32; shift += 8 {
			ca := int32(a>>shift) & 0xFF
			cb := int32(b>>shift) & 0xFF
			if d := ca - cb; d < -1 || d > 1 {
				t.Fatalf("nine-patch vs path lane: pixel (%d,%d) %08x vs %08x",
					int32(i)%pixA.Width, int32(i)/pixA.Width, a, b)
			}
		}
	}
}

func TestDitherSurvivesMaskFilter(t *testing.T) {
	// Dithering stays enabled with a mask filter even for a constant color, which switches the solid srcover lane to
	// the raster pipeline's constant collapse. Constant pipelines never actually dither, so the output must stay within
	// the ±1 legacy-vs-pipeline rounding band of the undithered draw.
	plain, pixA := newWhiteCanvasWH(60, 40)
	dith, pixB := newWhiteCanvasWH(60, 40)
	for i, cv := range []*Canvas{plain, dith} {
		p := NewPaint()
		p.Color = 0x80336699
		p.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)
		p.Dither = i == 1
		cv.DrawRect(geom.RectLTRB(15, 10, 45, 30), p)
	}
	for i := range pixA.Pix {
		a := pixA.Pix[i]
		b := pixB.Pix[i]
		for shift := 0; shift < 32; shift += 8 {
			ca := int32(a>>shift) & 0xFF
			cb := int32(b>>shift) & 0xFF
			if d := ca - cb; d < -1 || d > 1 {
				t.Fatalf("dither with mask filter: pixel (%d,%d) %08x vs %08x",
					int32(i)%pixA.Width, int32(i)/pixA.Width, a, b)
			}
		}
	}
}
