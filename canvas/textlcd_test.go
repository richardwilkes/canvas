// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// End-to-end CPU tests for LCD subpixel text through the full canvas text stack: fringing on a known geometry,
// byte-identical grayscale on unknown geometry, the layer and non-srcOver degrades, and the BGR/RGB order flip.

package canvas

import (
	"os"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

func lcdTestTypeface(t *testing.T) *font.Typeface {
	t.Helper()
	data, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	return tf
}

// renderLCDText draws "Hamburgefonstiv" onto a white 200x60 device with the given props and settings, returning the
// pixels.
func renderLCDText(t *testing.T, props font.DeviceProps, edging font.Edging, mutate func(*Canvas, *Paint) (*Paint, bool)) []uint32 {
	t.Helper()
	pix := raster.NewPixmap(200, 60)
	dev := NewBitmapDevice(pix)
	dev.SetDeviceProps(props)
	c := New(dev)
	c.Clear(colorcore.RGB(255, 255, 255))
	f := font.NewFont(lcdTestTypeface(t), 24, 1, 0)
	f.SetEdging(edging)
	paint := NewPaint()
	paint.Color = colorcore.RGB(0, 0, 0)
	restore := false
	if mutate != nil {
		paint, restore = mutate(c, paint)
	}
	c.DrawSimpleText([]byte("Hamburgefonstiv"), font.TextEncodingUTF8, 10, 40, f, paint)
	if restore {
		c.Restore()
	}
	out := make([]uint32, len(pix.Pix))
	copy(out, pix.Pix)
	return out
}

func fringePixels(pix []uint32) int {
	n := 0
	for _, w := range pix {
		r, g, b := w&0xFF, w>>8&0xFF, w>>16&0xFF
		if r != g || g != b {
			n++
		}
	}
	return n
}

func inkPixels(pix []uint32) int {
	n := 0
	for _, w := range pix {
		if w != 0xFFFFFFFF {
			n++
		}
	}
	return n
}

func pixEqual(a, b []uint32) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestLCDTextEndToEnd(t *testing.T) {
	rgbh := font.DeviceProps{PixelGeometry: font.PixelGeometryRGBH}
	bgrh := font.DeviceProps{PixelGeometry: font.PixelGeometryBGRH}
	unknown := font.DeviceProps{}

	lcd := renderLCDText(t, rgbh, font.EdgingSubpixelAntiAlias, nil)
	bgr := renderLCDText(t, bgrh, font.EdgingSubpixelAntiAlias, nil)
	gray := renderLCDText(t, unknown, font.EdgingSubpixelAntiAlias, nil)
	aa := renderLCDText(t, unknown, font.EdgingAntiAlias, nil)
	aaOnLCD := renderLCDText(t, rgbh, font.EdgingAntiAlias, nil)

	// LCD output carries color fringing; the same text has substantial ink either way.
	if f := fringePixels(lcd); f == 0 {
		t.Error("RGB_H subpixel text has no fringing")
	}
	if ink := inkPixels(lcd); ink < 500 {
		t.Errorf("LCD ink pixels = %d, implausibly low", ink)
	}

	// The BGR geometry swaps the fringe channel order: the two renderings differ but their per-pixel R/B-swapped words
	// match exactly (the FIR and pre-blend are symmetric under the swap).
	if pixEqual(lcd, bgr) {
		t.Error("RGB_H and BGR_H produced identical output")
	}
	swapped := make([]uint32, len(bgr))
	for i, w := range bgr {
		swapped[i] = w&0xFF00FF00 | w>>16&0xFF | w&0xFF<<16
	}
	if !pixEqual(lcd, swapped) {
		t.Error("BGR_H output is not the R/B mirror of RGB_H")
	}

	// Unknown geometry degrades subpixel edging to grayscale AA — byte-identical to kAntiAlias for unstyled glyphs (the
	// FreeType host renders plain A8 there; the 4x filter lane is styled-only).
	if !pixEqual(gray, aa) {
		t.Error("subpixel edging on unknown geometry != plain AA bytes")
	}
	if f := fringePixels(gray); f != 0 {
		t.Errorf("unknown-geometry text has %d fringed pixels, want 0", f)
	}

	// The default path is untouched by the geometry: plain AA on an RGB_H device is byte-identical.
	if !pixEqual(aaOnLCD, aa) {
		t.Error("plain AA bytes change with pixel geometry")
	}
}

func TestLCDTextDegrades(t *testing.T) {
	rgbh := font.DeviceProps{PixelGeometry: font.PixelGeometryRGBH}

	// Layers keep unknown pixel geometry, so text inside a layer on an LCD surface is grayscale.
	layer := renderLCDText(t, rgbh, font.EdgingSubpixelAntiAlias, func(c *Canvas, p *Paint) (*Paint, bool) {
		c.SaveLayer(nil, nil)
		return p, true
	})
	if f := fringePixels(layer); f != 0 {
		t.Errorf("layer text has %d fringed pixels, want 0", f)
	}

	// Non-srcOver paints fall back to the unknown-geometry props (the bitmap blitters can only draw LCD text in
	// srcOver): darken over white is visually srcover-like but must not fringe.
	darken := renderLCDText(t, rgbh, font.EdgingSubpixelAntiAlias, func(_ *Canvas, p *Paint) (*Paint, bool) {
		p.BlendMode = raster.BlendDarken
		return p, false
	})
	if f := fringePixels(darken); f != 0 {
		t.Errorf("non-srcOver text has %d fringed pixels, want 0", f)
	}

	// kUseDeviceIndependentFonts disables LCD.
	dif := font.DeviceProps{PixelGeometry: font.PixelGeometryRGBH, UseDeviceIndependentFonts: true}
	difPix := renderLCDText(t, dif, font.EdgingSubpixelAntiAlias, nil)
	if f := fringePixels(difPix); f != 0 {
		t.Errorf("device-independent-fonts text has %d fringed pixels, want 0", f)
	}

	// Blur mask filters keep the LCD16 rec but flip the glyph to a blurred A8 silhouette (the pre-blend is not applied
	// to filtered text): no fringing.
	blurred := renderLCDText(t, rgbh, font.EdgingSubpixelAntiAlias, func(_ *Canvas, p *Paint) (*Paint, bool) {
		p.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)
		return p, false
	})
	if f := fringePixels(blurred); f != 0 {
		t.Errorf("blurred text has %d fringed pixels, want 0", f)
	}
	if inkPixels(blurred) == 0 {
		t.Error("blurred text drew nothing")
	}
}

// TestLCDTextShaderPaint exercises the shader blitter LCD lanes: a gradient-shaded srcOver paint keeps LCD fringing on
// a known geometry.
func TestLCDTextShaderPaint(t *testing.T) {
	rgbh := font.DeviceProps{PixelGeometry: font.PixelGeometryRGBH}
	shaded := renderLCDText(t, rgbh, font.EdgingSubpixelAntiAlias, func(_ *Canvas, p *Paint) (*Paint, bool) {
		p.Shader = shaders.NewLinearGradient(geom.Point{X: 0, Y: 0}, geom.Point{X: 200, Y: 0},
			[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, shaders.TileClamp, nil)
		return p, false
	})
	if f := fringePixels(shaded); f == 0 {
		t.Error("shader-painted LCD text has no fringing")
	}
}
