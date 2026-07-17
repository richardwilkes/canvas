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
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// streamWithAll reports whether any single (inflated) content stream contains every substring.
func streamWithAll(data []byte, subs ...string) bool {
	for _, s := range allStreamContents(data) {
		all := true
		for _, sub := range subs {
			if !strings.Contains(s, sub) {
				all = false
				break
			}
		}
		if all {
			return true
		}
	}
	return false
}

// TestDeviceColorFilterFoldsIntoColor exercises the paint-cleanup path: a color filter on a shaderless paint folds into
// the paint color (a Blend/Src filter replaces the color outright), so the fill emits that color directly and never a
// pattern.
func TestDeviceColorFilterFoldsIntoColor(t *testing.T) {
	p := canvas.NewPaint()
	p.Color = colorcore.ARGB(255, 200, 100, 50)
	p.ColorFilter = colorfilter.NewBlend(colorcore.ARGB(255, 0, 0, 255), raster.BlendSrc)
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(10, 10, 90, 90), p)
	})
	validatePDF(t, data)
	// The folded color is opaque blue (0 0 1); no color filter survives, no pattern is used.
	if !streamWithAll(data, "0 0 1 RG 0 0 1 rg") {
		t.Errorf("color filter did not fold into a blue fill color; streams=%v", allStreamContents(data))
	}
	if streamWithAll(data, "/Pattern cs") {
		t.Error("a shaderless color filter must fold into the color, not become a pattern")
	}
}

// TestDeviceColorFilterWithShaderMakesPattern exercises the shader branch of paint cleanup: a color filter wrapped
// around a shader becomes a color-filtering shader, which the PDF device rasterizes as a fallback tiling pattern (an
// image XObject inside a PatternType 1), never a native shading.
func TestDeviceColorFilterWithShaderMakesPattern(t *testing.T) {
	p := canvas.NewPaint()
	p.Color = colorcore.ARGB(255, 255, 255, 255)
	p.Shader = shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{colorcore.ARGB(255, 255, 0, 0), colorcore.ARGB(255, 0, 255, 0)}, nil, shaders.TileClamp, nil)
	p.ColorFilter = colorfilter.NewBlend(colorcore.ARGB(255, 0, 0, 255), raster.BlendModulate)
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 100, 100), p)
	})
	validatePDF(t, data)
	// The wrapped shader is not recognized as a gradient, so it rasterizes into a fallback tiling pattern.
	if !anyObjectContains(data, "/PatternType 1") {
		t.Errorf("color-filtered gradient did not rasterize to a tiling pattern; objects=%v", dictObjects(data))
	}
	if countSubstr(data, "/Subtype /Image") < 1 {
		t.Error("fallback pattern has no rasterized image XObject")
	}
	if !streamWithAll(data, "/Pattern cs") {
		t.Error("the fill does not paint with a pattern")
	}
}

// TestDevicePathMaskFilterLuminositySMask exercises internalDrawPathWithFilter: a mask-filtered fill draws the filtered
// coverage into a DeviceGray form XObject used as a luminosity soft mask over the path shape.
func TestDevicePathMaskFilterLuminositySMask(t *testing.T) {
	p := canvas.NewPaint()
	p.Color = colorcore.ARGB(255, 20, 40, 60)
	p.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 3, true)
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(30, 30, 70, 70), p)
	})
	validatePDF(t, data)
	if !anyObjectContains(data, "/SMask <<", "/S /Luminosity") {
		t.Errorf("no luminosity SMask ExtGState; objects=%v", dictObjects(data))
	}
	// The mask is carried by a DeviceGray transparency-group form XObject holding a DeviceGray coverage image.
	if !anyObjectContains(data, "/Subtype /Form", "/Group", "/CS /DeviceGray") {
		t.Errorf("no DeviceGray transparency-group form XObject; objects=%v", dictObjects(data))
	}
	if countSubstr(data, "/ColorSpace /DeviceGray") < 1 {
		t.Error("no DeviceGray mask coverage image")
	}
	// The soft mask is turned on for the fill, then back off (/SMask /None).
	if !anyObjectContains(data, "/SMask /None") {
		t.Errorf("no /SMask /None turn-off; objects=%v", dictObjects(data))
	}
	// Under the mask, a rect is filled.
	if !streamWithAll(data, " re\n", "f\n") {
		t.Error("no masked rect fill in the content")
	}
}

// TestDeviceImageColorFilter exercises internalDrawImageRect's color-filter lane: the filter is baked into a new image
// XObject (a Blend/Src filter turns an opaque image entirely blue), which is then drawn normally.
func TestDeviceImageColorFilter(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 200, 100, 50, 255)
	p := canvas.NewPaint()
	p.ColorFilter = colorfilter.NewBlend(colorcore.ARGB(255, 0, 0, 255), raster.BlendSrc)
	data := renderPDFUncompressed(t, 20, 20, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectWH(4, 4), geom.RectLTRB(0, 0, 8, 8),
			shaders.SamplingOptions{}, p, canvas.ConstraintFast)
	})
	validatePDF(t, data)
	if n := countSubstr(data, "/Subtype /Image"); n != 1 {
		t.Fatalf("expected 1 color-filtered image XObject, got %d", n)
	}
	mustContain(t, string(data), "/ColorSpace /DeviceRGB")
	// Every pixel became blue (0,0,255); the original (200,100,50) is gone.
	if !bytes.Contains(data, []byte{0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF}) {
		t.Error("color-filtered image stream is not blue")
	}
	if bytes.Contains(data, []byte{200, 100, 50}) {
		t.Error("original image color survived the filter")
	}
}

// TestDeviceImagePerspectiveRasterizes exercises internalDrawImageRect's perspective lane: a perspective CTM (which PDF
// can't express for an image) forces the image to be rasterized into a new axis-aligned image XObject sized to its
// physical bounds. The opaque 4x4 source rasterizes to a ~20x20 image whose perspective-AA edges are no longer fully
// opaque, so it also gains a DeviceGray SMask — both proof that the image was rasterized rather than passed through.
func TestDeviceImagePerspectiveRasterizes(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 200, 100, 50, 255)
	var persp geom.Matrix
	persp.SetAll(1, 0, 0, 0, 1, 0, 0.002, 0, 1)
	data := renderPDF(t, 40, 40, func(c *canvas.Canvas) {
		c.Concat(&persp)
		c.DrawImageRect(img, geom.RectWH(4, 4), geom.RectLTRB(0, 0, 20, 20),
			shaders.SamplingOptions{}, canvas.NewPaint(), canvas.ConstraintFast)
	})
	validatePDF(t, data)
	if n := countSubstr(data, "/Subtype /Image"); n < 1 {
		t.Fatalf("expected a rasterized image XObject, got %d", n)
	}
	// The embedded image is the rasterized physical-bounds result, not the 4x4 source.
	if anyObjectContains(data, "/Width 4\n") || anyObjectContains(data, "/Height 4\n") {
		t.Error("perspective image was not rasterized to its physical bounds")
	}
	if !anyStreamContains(data, " Do\n") {
		t.Error("no image XObject invocation")
	}
}

// TestDeviceAlphaOnlyImageLuminositySMask exercises internalDrawImageRect's alpha-only lane: an A8 image with no color
// filter becomes a luminosity soft mask (the alpha, as a DeviceGray form XObject) over a fill of the paint color.
func TestDeviceAlphaOnlyImageLuminositySMask(t *testing.T) {
	w, h := int32(4), int32(4)
	alphas := make([]byte, int(w)*int(h))
	for i := range alphas {
		alphas[i] = 0xA0
	}
	info := imagecore.ImageInfo{Width: w, Height: h, ColorType: imagecore.ColorTypeAlpha8, AlphaType: imagecore.AlphaTypePremul}
	img := imagecore.NewRasterData(info, alphas, int(w))
	if img == nil {
		t.Fatal("nil alpha8 image")
	}
	p := canvas.NewPaint()
	p.Color = colorcore.ARGB(255, 200, 30, 40)
	data := renderPDF(t, 20, 20, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectWH(4, 4), geom.RectLTRB(0, 0, 8, 8),
			shaders.SamplingOptions{}, p, canvas.ConstraintFast)
	})
	validatePDF(t, data)
	if !anyObjectContains(data, "/SMask <<", "/S /Luminosity") {
		t.Errorf("no luminosity SMask ExtGState; objects=%v", dictObjects(data))
	}
	if !anyObjectContains(data, "/Subtype /Form", "/Group", "/CS /DeviceGray") {
		t.Errorf("no DeviceGray transparency-group form XObject; objects=%v", dictObjects(data))
	}
	if countSubstr(data, "/ColorSpace /DeviceGray") < 1 {
		t.Error("no DeviceGray mask coverage image")
	}
	if !anyObjectContains(data, "/SMask /None") {
		t.Errorf("no /SMask /None turn-off; objects=%v", dictObjects(data))
	}
	// Under the mask, the paint color fills the device.
	if !streamWithAll(data, ".7843 .1176 .1569 rg") {
		t.Errorf("no paint-color fill under the alpha mask; streams=%v", allStreamContents(data))
	}
}
