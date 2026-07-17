// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pdf

import (
	"bytes"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// renderImageShaderPage draws a full-page rect filled with shader and returns the uncompressed PDF bytes, so the
// pattern/page content streams hold raw (assertable) operators.
func renderImageShaderPage(t *testing.T, w, h float32, s shaders.Shader) []byte {
	t.Helper()
	p := canvas.NewPaint()
	p.Shader = s
	data := renderPDFUncompressed(t, w, h, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, w, h), p)
	})
	validatePDF(t, data)
	return data
}

// countPatternType1 counts the tiling-pattern (PatternType 1) objects in the document. A tiling pattern carries a
// content stream, so it is counted via allObjectDicts (which captures stream dict headers), unlike the plain-dict
// shading patterns (PatternType 2) that dictObjects covers.
func countPatternType1(data []byte) int {
	n := 0
	for _, s := range allObjectDicts(data) {
		if strings.Contains(s, "/PatternType 1") {
			n++
		}
	}
	return n
}

func TestImageShaderClampTilingPattern(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 0x40, 0x80, 0xC0, 0xFF)
	s := shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, shaders.SamplingOptions{}, nil)
	data := renderImageShaderPage(t, 100, 100, s)

	// The page selects a /Pattern color space and paints with the pattern.
	if !bytes.Contains(data, []byte("/Pattern cs")) {
		t.Error("page content does not select a /Pattern color space")
	}
	if !bytes.Contains(data, []byte(" scn\n")) {
		t.Error("page content does not set the pattern with scn")
	}
	// A colored, constant-spacing tiling pattern (PatternType 1 / PaintType 1 / TilingType 1) is emitted.
	if !anyObjectContains(data, "/PatternType 1", "/PaintType 1", "/TilingType 1") {
		t.Errorf("no tiling pattern object; objects=%v", dictObjects(data))
	}
	// The pattern cell drew the image as an Image XObject and invoked it with "/Xn Do".
	if countSubstr(data, "/Subtype /Image") < 1 {
		t.Error("no Image XObject emitted for the image-shader pattern cell")
	}
	if !xObjectDoRE.Match(data) {
		t.Error("the pattern cell does not invoke the image XObject (no '/Xn Do')")
	}
}

func TestImageShaderRepeatTiling(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 0x10, 0x20, 0x30, 0xFF)
	s := shaders.NewImage(img, shaders.TileRepeat, shaders.TileRepeat, shaders.SamplingOptions{}, nil)
	data := renderImageShaderPage(t, 100, 100, s)

	// A repeat image shader tiles the 4x4 cell: XStep/YStep are the image dimensions, and PDF's own pattern tiling
	// provides the repetition (no clamp edge synthesis).
	if !anyObjectContains(data, "/PatternType 1", "/XStep 4", "/YStep 4") {
		t.Errorf("no 4x4-step tiling pattern; objects=%v", dictObjects(data))
	}
	if countSubstr(data, "/Subtype /Image") != 1 {
		t.Errorf("expected exactly one Image XObject for a repeat cell, got %d",
			countSubstr(data, "/Subtype /Image"))
	}
}

// TestImageShaderAlphaOnlyTintsCoverage exercises make_image_shader over an alpha-only (A8) source image. adjust_color
// keeps the paint color, so the pattern cell draws the A8 image tinted by that color; the draw routes through
// internalDrawImageRect's alpha-only lane, which renders the alpha as a luminosity SMask over a paint-color fill.
// Before that lane landed the pattern cell drew empty.
func TestImageShaderAlphaOnlyTintsCoverage(t *testing.T) {
	w, h := int32(4), int32(4)
	alphas := make([]byte, int(w)*int(h))
	for i := range alphas {
		alphas[i] = 0xC0
	}
	info := imagecore.ImageInfo{Width: w, Height: h, ColorType: imagecore.ColorTypeAlpha8, AlphaType: imagecore.AlphaTypePremul}
	img := imagecore.NewRasterData(info, alphas, int(w))
	if img == nil {
		t.Fatal("nil alpha8 image")
	}
	s := shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, shaders.SamplingOptions{}, nil)
	p := canvas.NewPaint()
	p.Color = colorcore.ARGB(255, 200, 30, 40)
	p.Shader = s
	data := renderPDFUncompressed(t, 40, 40, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 40, 40), p)
	})
	validatePDF(t, data)

	// A tiling pattern is emitted for the image shader (not an empty cell).
	if n := countPatternType1(data); n < 1 {
		t.Fatalf("no tiling pattern for the alpha-only image shader; objects=%v", dictObjects(data))
	}
	// The alpha becomes a luminosity soft mask, and the pattern cell fills with the paint color it masks (200/255,
	// 30/255, 40/255 -> ".7843 .1176 .1569 rg"). Together these prove the coverage rendered.
	if !anyObjectContains(data, "/S /Luminosity") {
		t.Errorf("alpha-only image shader cell has no luminosity SMask; objects=%v", dictObjects(data))
	}
	if !anyStreamContains(data, ".7843 .1176 .1569 rg") {
		t.Errorf("pattern cell does not fill the tinted coverage color; streams=%v", allStreamContents(data))
	}
}

func TestImageShaderPatternDedup(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 0x40, 0x80, 0xC0, 0xFF)
	s := shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, shaders.SamplingOptions{}, nil)
	p := canvas.NewPaint()
	p.Shader = s
	data := renderPDFUncompressed(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 40, 40), p)
		c.DrawRect(geom.RectLTRB(50, 50, 90, 90), p)
	})
	validatePDF(t, data)

	// The identical image shader drawn twice shares exactly one tiling pattern (fImageShaderMap).
	if n := countPatternType1(data); n != 1 {
		t.Errorf("expected 1 shared tiling pattern, got %d\nobjects=%v", n, dictObjects(data))
	}
	// It is selected once (the graphic-state machine avoids re-emitting an unchanged pattern) and reused by both fills.
	if c := bytes.Count(data, []byte("/Pattern cs")); c != 1 {
		t.Errorf("expected the shared pattern to be selected once, got %d", c)
	}
	if c := bytes.Count(data, []byte("\nf\n")) + bytes.Count(data, []byte(" f\n")); c < 2 {
		t.Errorf("expected two fills reusing the pattern, got %d", c)
	}
}

func TestFallbackShaderTilingPattern(t *testing.T) {
	// A blend shader is neither a gradient nor an image, so it takes make_fallback_shader: the whole shader is
	// rasterized into an N32 surface and wrapped as a clamp/clamp image-shader pattern.
	dst := shaders.NewColor(colorcore.ARGB(255, 200, 40, 40))
	src := shaders.NewColor(colorcore.ARGB(255, 40, 40, 200))
	blend := shaders.NewBlend(raster.BlendPlus, dst, src)
	if _, _, _, _, ok := shaders.AsImage(blend); ok {
		t.Fatal("test precondition failed: the blend shader reports as an image shader")
	}
	data := renderImageShaderPage(t, 16, 16, blend)

	if !bytes.Contains(data, []byte("/Pattern cs")) {
		t.Error("page content does not select a /Pattern color space for the fallback shader")
	}
	if !anyObjectContains(data, "/PatternType 1", "/PaintType 1", "/TilingType 1") {
		t.Errorf("no tiling pattern for the fallback shader; objects=%v", dictObjects(data))
	}
	// The rasterized shader is embedded as an Image XObject drawn into the pattern cell.
	if countSubstr(data, "/Subtype /Image") < 1 {
		t.Error("no Image XObject emitted for the fallback (rasterized) shader")
	}
	if !xObjectDoRE.Match(data) {
		t.Error("the fallback pattern cell does not invoke its image XObject")
	}
}

func TestPinInt(t *testing.T) {
	cases := []struct{ x, lo, hi, want int32 }{
		{x: 5, lo: 1, hi: 10, want: 5},
		{x: 0, lo: 1, hi: 10, want: 1},
		{x: 20, lo: 1, hi: 10, want: 10},
		{x: 1, lo: 1, hi: 10, want: 1},
		{x: 10, lo: 1, hi: 10, want: 10},
	}
	for _, c := range cases {
		if got := pinInt(c.x, c.lo, c.hi); got != c.want {
			t.Errorf("pinInt(%d,%d,%d) = %d, want %d", c.x, c.lo, c.hi, got, c.want)
		}
	}
}
