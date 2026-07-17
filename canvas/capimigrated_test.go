// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests.

package canvas

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stroke"
)

// TestCanvasSkewAndResetMatrix verifies Skew composes the same skew matrix SetSkew builds, and ResetMatrix returns the
// CTM to identity regardless of what preceded it.
func TestCanvasSkewAndResetMatrix(t *testing.T) {
	c, _ := newWhiteCanvas()
	c.Translate(3, 5)
	c.Skew(0.1, 0.2)

	want, _ := newWhiteCanvas()
	want.Translate(3, 5)
	var sk geom.Matrix
	sk.SetSkew(0.1, 0.2)
	want.Concat(&sk)
	got, ref := c.TotalMatrix(), want.TotalMatrix()
	if got.As9() != ref.As9() {
		t.Errorf("Skew matrix = %v, want %v", got.As9(), ref.As9())
	}

	c.Rotate(37)
	c.Scale(2, 0.5)
	c.ResetMatrix()
	identity := geom.IdentityMatrix()
	if m := c.TotalMatrix(); !m.CheapEqual(&identity) {
		t.Errorf("ResetMatrix gave %v, want identity", m.As9())
	}
}

// TestCanvasRotateRadiansConversion pins geom.RadiansToDegrees through the canvas rotate lane: rotating by pi/2 radians
// must build (approximately) the same matrix as rotating by 90 degrees.
func TestCanvasRotateRadiansConversion(t *testing.T) {
	c, _ := newWhiteCanvas()
	c.Rotate(geom.RadiansToDegrees(float32(math.Pi / 2)))
	var want geom.Matrix
	want.SetRotate(90)
	got := c.TotalMatrix()
	g, w := got.As9(), want.As9()
	for i := range g {
		if d := g[i] - w[i]; d > 1e-3 || d < -1e-3 {
			t.Errorf("rotate(RadiansToDegrees(pi/2)) matrix = %v, want ~rotate(90) %v", g, w)
			break
		}
	}
}

// TestCanvasLocalClipBoundsEmpty covers LocalClipBounds' empty-clip lane: intersecting with a rect off the device
// leaves an empty clip, and the local bounds collapse to the zero rect.
func TestCanvasLocalClipBoundsEmpty(t *testing.T) {
	c, _ := newWhiteCanvas()
	if b := c.LocalClipBounds(); b.IsEmpty() {
		t.Fatalf("fresh canvas local clip bounds %v should not be empty", b)
	}
	c.ClipRect(geom.RectLTRB(cvW+50, cvH+50, cvW+60, cvH+60), raster.ClipIntersect, false)
	if b := c.LocalClipBounds(); b != (geom.Rect{}) {
		t.Errorf("empty clip local bounds = %v, want the zero rect", b)
	}
}

// TestCanvasQuickRejectPath covers sk_canvas_quick_reject_path's semantics on the core API: an empty path is always
// rejected, a path outside the clip is rejected, and a path inside it is not.
func TestCanvasQuickRejectPath(t *testing.T) {
	c, _ := newWhiteCanvas()
	c.ClipRect(geom.RectLTRB(10, 12, 30, 28), raster.ClipIntersect, false)

	var empty path.Path
	if !c.QuickRejectPath(&empty) {
		t.Error("an empty path should be quick-rejected")
	}
	var outside path.Path
	outside.MoveTo(40, 40)
	outside.LineTo(60, 44)
	outside.LineTo(50, 60)
	outside.Close()
	if !c.QuickRejectPath(&outside) {
		t.Error("a path outside the clip should be quick-rejected")
	}
	var inside path.Path
	inside.MoveTo(12, 14)
	inside.LineTo(24, 16)
	inside.LineTo(16, 24)
	inside.Close()
	if c.QuickRejectPath(&inside) {
		t.Error("a path inside the clip should not be quick-rejected")
	}
}

// TestCanvasDrawPointMatchesDrawPoints verifies DrawPoint lowers to a one-point DrawPoints draw, and — with an
// anti-aliased non-round cap — exercises the square-cap point proc.
func TestCanvasDrawPointMatchesDrawPoints(t *testing.T) {
	sp := solidPaint(0xFF2E64C8, true)
	sp.Style = StyleStroke
	sp.StrokeWidth = 3 // default CapButt: the AA square proc

	c1, p1 := newWhiteCanvas()
	c1.DrawPoint(12.5, 19.5, sp)

	c2, p2 := newWhiteCanvas()
	pt := [1]geom.Point{{X: 12.5, Y: 19.5}}
	c2.DrawPoints(PointModePoints, pt[:], sp)

	drew := false
	for i := range p1.Pix {
		if p1.Pix[i] != p2.Pix[i] {
			t.Fatal("DrawPoint output differs from a one-point DrawPoints")
		}
		if p1.Pix[i] != 0xFFFFFFFF {
			drew = true
		}
	}
	if !drew {
		t.Fatal("DrawPoint drew nothing")
	}
}

// TestDrawImageNineCleansPaint verifies clean_paint_for_lattice on the nine-patch lane: the paint's mask filter and
// anti-alias flag are stripped, so a paint carrying them renders identically to one without.
func TestDrawImageNineCleansPaint(t *testing.T) {
	info, _ := imagecore.MakeInfo(6, 6, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeOpaque)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < 6; y++ {
		for x := int32(0); x < 6; x++ {
			col := uint32(0xFF0000FF)
			if x >= 2 && x < 4 && y >= 2 && y < 4 {
				col = 0xFF00FF00
			}
			p.Words[y*6+x] = col
		}
	}
	img := imagecore.FromPixels(p)
	center := geom.IRect{Left: 2, Top: 2, Right: 4, Bottom: 4}

	dirty := NewPaint()
	dirty.AntiAlias = true
	dirty.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)

	c1, p1 := newWhiteCanvas()
	c1.DrawImageNine(img, center, geom.RectWH(16, 16), shaders.FilterNearest, dirty)
	c2, p2 := newWhiteCanvas()
	c2.DrawImageNine(img, center, geom.RectWH(16, 16), shaders.FilterNearest, NewPaint())

	for i := range p1.Pix {
		if p1.Pix[i] != p2.Pix[i] {
			t.Fatal("nine-patch draw with a mask-filtered AA paint should render as if the paint were cleaned")
		}
	}
}

// TestPaintFillPathStroke exercises Paint.FillPath through to the stroker (the entry point behind unison's
// get-fill-path use): a stroked rectangle becomes a filled outline (true), while a zero-width hairline stroke reports
// false. StrokeSpec must agree with the paint's stroke fields.
func TestPaintFillPathStroke(t *testing.T) {
	var src path.Path
	src.AddRect(geom.Rect{Left: 0, Top: 0, Right: 20, Bottom: 10}, geom.DirectionCW)

	p := NewPaint()
	p.Style = StyleStroke
	p.StrokeWidth = 4

	spec := p.StrokeSpec()
	if spec.Width != 4 || spec.Style != stroke.PaintStyle(StyleStroke) {
		t.Fatalf("StrokeSpec = %+v, want width 4, stroke style", spec)
	}

	var dst path.Path
	if !p.FillPath(&src, &dst, nil, 1) {
		t.Fatal("stroking a width-4 rect should yield a fillable outline (true)")
	}
	if dst.IsEmpty() {
		t.Error("fill path is empty; expected the stroke outline")
	}

	// A zero-width stroke is a hairline: the contract returns false (draw dst as a hairline).
	p.StrokeWidth = 0
	var hair path.Path
	if p.FillPath(&src, &hair, nil, 1) {
		t.Error("a zero-width (hairline) stroke should report false")
	}
}

// TestDrawTextClippedGlyphSkipped covers paintMasks' per-glyph clip rejection: with a clip that excludes part of the
// text run, the clipped-out glyphs are skipped and only the in-clip ink lands.
func TestDrawTextClippedGlyphSkipped(t *testing.T) {
	f := loadTestFont(t, 24)
	p := solidPaint(0xFF2E64C8, true)

	// Unclipped: both glyphs ink.
	c1, p1 := newWhiteCanvas()
	c1.DrawSimpleText([]byte("Ag"), font.TextEncodingUTF8, 4, 28, f, p)
	full := inkBounds(p1)
	if full.IsEmpty() {
		t.Fatal("text draw produced no ink")
	}

	// Clip to a band left of the second glyph: its mask cannot intersect the clip and is skipped.
	c2, p2 := newWhiteCanvas()
	c2.ClipRect(geom.RectLTRB(0, 0, 12, cvH), raster.ClipIntersect, false)
	c2.DrawSimpleText([]byte("Ag"), font.TextEncodingUTF8, 4, 28, f, p)
	clipped := inkBounds(p2)
	if clipped.IsEmpty() {
		t.Fatal("clipped text draw produced no ink at all")
	}
	if clipped.Right > 12 {
		t.Errorf("ink escaped the clip: bounds %v", clipped)
	}
	if clipped.Right >= full.Right {
		t.Errorf("clip did not reduce the ink extent: clipped %v vs full %v", clipped, full)
	}
}
