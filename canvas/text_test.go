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
	"os"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/textblob"
)

func loadTestFont(t *testing.T, size float32) *font.Font {
	t.Helper()
	data, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	return font.NewFont(tf, size, 1, 0)
}

// inkBounds returns the bounding box of non-white pixels.
func inkBounds(pix *raster.Pixmap) geom.IRect {
	buf := pix.RGBA8888Bytes()
	var bounds geom.IRect
	first := true
	for y := int32(0); y < pix.Height; y++ {
		for x := int32(0); x < pix.Width; x++ {
			off := (y*pix.Width + x) * 4
			if buf[off] != 0xFF || buf[off+1] != 0xFF || buf[off+2] != 0xFF {
				r := geom.IRectLTRB(x, y, x+1, y+1)
				if first {
					bounds = r
					first = false
				} else {
					bounds.Join(r)
				}
			}
		}
	}
	return bounds
}

func newTextCanvas(w, h int32) (*Canvas, *raster.Pixmap) {
	pix := raster.NewPixmap(w, h)
	c := NewForPixmap(pix)
	c.Clear(colorcore.Color(0xFFFFFFFF))
	return c, pix
}

func TestDrawSimpleTextInk(t *testing.T) {
	f := loadTestFont(t, 20)
	c, pix := newTextCanvas(160, 40)
	paint := NewPaint()
	paint.AntiAlias = true
	c.DrawSimpleText([]byte("Handgloves"), font.TextEncodingUTF8, 8, 28, f, paint)

	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	// The ink must roughly match measureText: starts near x=8, spans the measured advance.
	var bounds geom.Rect
	width := f.MeasureText([]byte("Handgloves"), font.TextEncodingUTF8, &bounds, nil)
	if ink.Left < 8+int32(bounds.Left)-1 || ink.Left > 8+int32(bounds.Left)+2 {
		t.Errorf("ink starts at %d; measured left %v", ink.Left, 8+bounds.Left)
	}
	if float32(ink.Right) < 8+bounds.Right-2 || float32(ink.Right) > 8+bounds.Right+2 {
		t.Errorf("ink ends at %d; measured right %v (width %v)", ink.Right, 8+bounds.Right, width)
	}
	// Baseline at 28: ascenders above, 'g' descender below.
	if ink.Top >= 28 || ink.Bottom <= 28 {
		t.Errorf("ink rows [%d,%d) should straddle the baseline 28", ink.Top, ink.Bottom)
	}
}

func TestDrawSimpleTextEmptyAndInvalid(t *testing.T) {
	f := loadTestFont(t, 20)
	c, pix := newTextCanvas(60, 30)
	paint := NewPaint()
	c.DrawSimpleText(nil, font.TextEncodingUTF8, 5, 20, f, paint)
	c.DrawSimpleText([]byte{0xff}, font.TextEncodingUTF8, 5, 20, f, paint)
	c.DrawSimpleText([]byte("x"), font.TextEncodingUTF8, 5, 20, nil, paint)
	if !inkBounds(pix).IsEmpty() {
		t.Error("nothing should have drawn")
	}
}

func TestDrawSimpleTextRespectsClipAndCTM(t *testing.T) {
	f := loadTestFont(t, 20)
	c, pix := newTextCanvas(120, 60)
	paint := NewPaint()
	paint.AntiAlias = true

	// Unclipped, the ink spans rows ~[30, 50) (baseline at 15+30=45, 'gg' descenders below). Clip to [0, 40): the
	// visible ink must stop at row 40 while still starting near row 30.
	c.Save()
	c.ClipRect(geom.RectLTRB(0, 0, 120, 40), raster.ClipIntersect, false)
	c.Translate(0, 30)
	c.DrawSimpleText([]byte("Egg"), font.TextEncodingUTF8, 4, 15, f, paint)
	c.Restore()

	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	if ink.Bottom > 40 {
		t.Errorf("ink escaped the clip: %v", ink)
	}
	if ink.Top >= 40 || ink.Top < 28 {
		t.Errorf("unexpected ink top: %v", ink)
	}
}

func TestDrawSimpleTextScaledCTM(t *testing.T) {
	f := loadTestFont(t, 12)
	c1, pix1 := newTextCanvas(120, 40)
	paint := NewPaint()
	paint.AntiAlias = true
	c1.Scale(2, 2)
	c1.DrawSimpleText([]byte("Ape"), font.TextEncodingUTF8, 2, 14, f, paint)
	inkScaled := inkBounds(pix1)

	f24 := loadTestFont(t, 24)
	c2, pix2 := newTextCanvas(120, 40)
	c2.DrawSimpleText([]byte("Ape"), font.TextEncodingUTF8, 4, 28, f24, paint)
	inkBig := inkBounds(pix2)

	if inkScaled.IsEmpty() || inkBig.IsEmpty() {
		t.Fatal("both draws must produce ink")
	}
	// A 12pt font under a 2x CTM must rasterize like a 24pt font (strike keyed on the device matrix), so the ink boxes
	// agree within a pixel on every edge.
	for _, d := range []int32{
		inkScaled.Left - inkBig.Left, inkScaled.Top - inkBig.Top,
		inkScaled.Right - inkBig.Right, inkScaled.Bottom - inkBig.Bottom,
	} {
		if d < -1 || d > 1 {
			t.Errorf("scaled CTM ink %v vs 24pt ink %v", inkScaled, inkBig)
			break
		}
	}
}

func TestDrawSimpleTextHugeUsesPaths(t *testing.T) {
	// 300pt text exceeds the 256 strike cap and rides the path lane; it must still render.
	f := loadTestFont(t, 300)
	c, pix := newTextCanvas(240, 240)
	paint := NewPaint()
	paint.AntiAlias = true
	c.DrawSimpleText([]byte("R"), font.TextEncodingUTF8, 10, 220, f, paint)
	ink := inkBounds(pix)
	if ink.IsEmpty() || ink.Width() < 100 || ink.Height() < 150 {
		t.Fatalf("huge glyph did not render: %v", ink)
	}
}

func TestDrawSimpleTextRotated(t *testing.T) {
	f := loadTestFont(t, 20)
	c, pix := newTextCanvas(100, 100)
	paint := NewPaint()
	paint.AntiAlias = true
	c.Translate(50, 50)
	c.Rotate(45)
	c.DrawSimpleText([]byte("Wave"), font.TextEncodingUTF8, 0, 0, f, paint)
	if inkBounds(pix).IsEmpty() {
		t.Fatal("rotated text must render")
	}
}

func TestDrawTextBlobMultiRun(t *testing.T) {
	f := loadTestFont(t, 18)
	builder := textblob.NewBuilder()
	glyphs := make([]uint16, 3)
	f.TextToGlyphs([]byte("abc"), font.TextEncodingUTF8, glyphs)

	r1 := builder.AllocRun(f, 3, 0, 0, nil)
	copy(r1.Glyphs, glyphs)
	r2 := builder.AllocRunPosH(f, 3, 22, nil)
	copy(r2.Glyphs, glyphs)
	copy(r2.Pos, []float32{0, 14, 28})
	blob := builder.Make()

	c, pix := newTextCanvas(80, 60)
	paint := NewPaint()
	paint.AntiAlias = true
	c.DrawTextBlob(blob, 6, 16, paint)
	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	// Two rows of text: ink spans both the first baseline (16) region and the second (38).
	if ink.Top >= 16 || ink.Bottom <= 30 {
		t.Errorf("expected two text rows, ink %v", ink)
	}
}

func TestDrawTextBlobNil(t *testing.T) {
	c, pix := newTextCanvas(40, 40)
	c.DrawTextBlob(nil, 0, 0, NewPaint())
	if !inkBounds(pix).IsEmpty() {
		t.Error("nil blob must draw nothing")
	}
}

func TestDrawTextMaskFilterWidensInk(t *testing.T) {
	f := loadTestFont(t, 24)
	paint := NewPaint()
	paint.AntiAlias = true
	c1, pix1 := newTextCanvas(90, 50)
	c1.DrawSimpleText([]byte("Bo"), font.TextEncodingUTF8, 10, 35, f, paint)
	plain := inkBounds(pix1)

	blurred := NewPaint()
	blurred.AntiAlias = true
	blurred.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 3, true)
	c2, pix2 := newTextCanvas(90, 50)
	c2.DrawSimpleText([]byte("Bo"), font.TextEncodingUTF8, 10, 35, f, blurred)
	blur := inkBounds(pix2)

	if blur.Left >= plain.Left || blur.Top >= plain.Top || blur.Right <= plain.Right ||
		blur.Bottom <= plain.Bottom {
		t.Errorf("blurred ink %v should strictly contain plain ink %v", blur, plain)
	}
}

func TestDrawTextStroked(t *testing.T) {
	f := loadTestFont(t, 40)
	paint := NewPaint()
	paint.AntiAlias = true
	paint.Style = StyleStroke
	paint.StrokeWidth = 1.5
	c, pix := newTextCanvas(60, 60)
	c.DrawSimpleText([]byte("O"), font.TextEncodingUTF8, 10, 45, f, paint)
	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	// The center of the stroked 'O' is unpainted.
	buf := pix.RGBA8888Bytes()
	cx := (ink.Left + ink.Right) / 2
	cy := (ink.Top + ink.Bottom) / 2
	off := (cy*pix.Width + cx) * 4
	if buf[off] != 0xFF {
		t.Error("stroked O center should be white")
	}
}

func TestDrawTextSubpixelShiftsCoverage(t *testing.T) {
	f := loadTestFont(t, 16)
	f.SetSubpixel(true)
	paint := NewPaint()
	paint.AntiAlias = true

	c1, pix1 := newTextCanvas(60, 24)
	c1.DrawSimpleText([]byte("ll"), font.TextEncodingUTF8, 5, 18, f, paint)
	c2, pix2 := newTextCanvas(60, 24)
	c2.DrawSimpleText([]byte("ll"), font.TextEncodingUTF8, 5.5, 18, f, paint)

	b1 := pix1.RGBA8888Bytes()
	b2 := pix2.RGBA8888Bytes()
	same := true
	for i := range b1 {
		if b1[i] != b2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("subpixel positioning: a half-pixel shift must change coverage")
	}
}
