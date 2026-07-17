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
	"regexp"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stream"
)

// renderPDFUncompressed draws through a device-backed page with FlateDecode disabled, so image streams hold raw
// (assertable) bytes and validatePDF's object-header scan cannot trip over compressed noise.
func renderPDFUncompressed(t *testing.T, w, h float32, draw func(c *canvas.Canvas)) []byte {
	t.Helper()
	md := DefaultMetadata()
	md.CompressionLevel = CompressionNone
	var buf stream.MemoryWStream
	doc := NewDocument(&buf, md)
	c := doc.BeginPageCanvas(w, h)
	draw(c)
	doc.EndPage()
	doc.Close()
	return append([]byte(nil), buf.Bytes()...)
}

// solidRGBAImage builds a WxH image whose every pixel is the given premul RGBA value.
func solidRGBAImage(t *testing.T, w, h int32, r, g, b, a byte) *imagecore.Image {
	t.Helper()
	data := make([]byte, int(w)*int(h)*4)
	for i := 0; i < len(data); i += 4 {
		data[i], data[i+1], data[i+2], data[i+3] = r, g, b, a
	}
	at := imagecore.AlphaTypePremul
	if a == 0xFF {
		at = imagecore.AlphaTypeOpaque
	}
	info := imagecore.ImageInfo{Width: w, Height: h, ColorType: imagecore.ColorTypeRGBA8888, AlphaType: at}
	img := imagecore.NewRasterData(info, data, int(w)*4)
	if img == nil {
		t.Fatal("solidRGBAImage: nil image")
	}
	return img
}

var xObjectDoRE = regexp.MustCompile(`/X\d+ Do`)

func countSubstr(data []byte, sub string) int { return bytes.Count(data, []byte(sub)) }

func TestDeviceDrawImageOpaque(t *testing.T) {
	img := solidRGBAImage(t, 4, 4, 0x40, 0x80, 0xC0, 0xFF)
	data := renderPDFUncompressed(t, 20, 20, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectWH(4, 4), geom.RectLTRB(2, 3, 10, 11),
			shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	})
	validatePDF(t, data)

	// Exactly one Image XObject, DeviceRGB, no soft mask.
	if n := countSubstr(data, "/Subtype /Image"); n != 1 {
		t.Fatalf("expected 1 image XObject, got %d", n)
	}
	mustContain(t, string(data), "/ColorSpace /DeviceRGB")
	mustContain(t, string(data), "/Width 4")
	mustContain(t, string(data), "/Height 4")
	mustContain(t, string(data), "/BitsPerComponent 8")
	if bytes.Contains(data, []byte("/SMask")) {
		t.Error("opaque image should have no /SMask")
	}
	// The raw color plane: 16 pixels of R,G,B = 40 80 C0.
	if !bytes.Contains(data, bytes.Repeat([]byte{0x40, 0x80, 0xC0}, 16)) {
		t.Error("image stream missing raw RGB plane")
	}

	// The content stream places and invokes the image XObject with the scaled/flipped matrix.
	content := pageContent(t, data)
	mustContain(t, content, "8 0 0 -8 2 11 cm\n")
	if !xObjectDoRE.MatchString(content) {
		t.Errorf("content missing /X<n> Do:\n%s", content)
	}
}

func TestDeviceDrawImageWithAlpha(t *testing.T) {
	// Premul (32,64,96,128): uniform alpha 128, so the image is not opaque and gets an SMask.
	img := solidRGBAImage(t, 4, 4, 32, 64, 96, 128)
	data := renderPDFUncompressed(t, 20, 20, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectWH(4, 4), geom.RectLTRB(0, 0, 8, 8),
			shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	})
	validatePDF(t, data)

	// Two Image XObjects: the DeviceRGB color image and its DeviceGray soft mask.
	if n := countSubstr(data, "/Subtype /Image"); n != 2 {
		t.Fatalf("expected 2 image XObjects (color + SMask), got %d", n)
	}
	mustContain(t, string(data), "/ColorSpace /DeviceRGB")
	mustContain(t, string(data), "/ColorSpace /DeviceGray")
	mustContain(t, string(data), "/SMask ")
	// The soft mask holds the alpha channel: 16 bytes of 0x80.
	if !bytes.Contains(data, bytes.Repeat([]byte{0x80}, 16)) {
		t.Error("SMask stream missing the alpha plane")
	}
}

func TestDeviceDrawImageCaching(t *testing.T) {
	img := solidRGBAImage(t, 2, 2, 0x11, 0x22, 0x33, 0xFF)
	data := renderPDFUncompressed(t, 40, 40, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectWH(2, 2), geom.RectLTRB(0, 0, 10, 10),
			shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
		c.DrawImageRect(img, geom.RectWH(2, 2), geom.RectLTRB(20, 20, 30, 30),
			shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	})
	validatePDF(t, data)

	// The same image drawn twice is serialized once (fPDFBitmapMap dedup)...
	if n := countSubstr(data, "/Subtype /Image"); n != 1 {
		t.Fatalf("expected 1 cached image XObject, got %d", n)
	}
	// ...and both draws reference the same XObject resource name.
	content := pageContent(t, data)
	names := xObjectDoRE.FindAllString(content, -1)
	if len(names) != 2 {
		t.Fatalf("expected 2 image draws, got %d: %q", len(names), names)
	}
	if names[0] != names[1] {
		t.Errorf("cached image draws use different resource names: %q vs %q", names[0], names[1])
	}
}

func TestDeviceDrawImageIntegralSubset(t *testing.T) {
	img := solidRGBAImage(t, 8, 8, 0x30, 0x60, 0x90, 0xFF)
	// An integral 2x2 sub-rect is subsetted (no clip needed).
	data := renderPDFUncompressed(t, 40, 40, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectLTRB(1, 1, 3, 3), geom.RectLTRB(0, 0, 20, 20),
			shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	})
	validatePDF(t, data)
	// The serialized image is the 2x2 subset, not the full 8x8.
	mustContain(t, string(data), "/Width 2")
	mustContain(t, string(data), "/Height 2")
	if bytes.Contains(data, []byte("/Width 8")) {
		t.Error("integral subset should serialize the 2x2 sub-image, not the full 8x8")
	}
	// An integral src rect needs no sub-pixel clip.
	if strings.Contains(pageContent(t, data), "W* n") {
		t.Error("integral src draw should not emit a sub-pixel clip")
	}
}

func TestDeviceDrawImageNonIntegralSrcClips(t *testing.T) {
	img := solidRGBAImage(t, 8, 8, 0x30, 0x60, 0x90, 0xFF)
	// A non-integral src rect adds a sub-pixel clip to the dst.
	data := renderPDFUncompressed(t, 40, 40, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectLTRB(0.5, 0.5, 6.5, 6.5), geom.RectLTRB(2, 2, 22, 22),
			shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	if !xObjectDoRE.MatchString(content) {
		t.Errorf("content missing image draw:\n%s", content)
	}
	// The sub-pixel clip is a device-space rect clip.
	mustContain(t, content, "W* n")
}

func TestDeviceDrawGray8Image(t *testing.T) {
	w, h := int32(4), int32(3)
	data := make([]byte, int(w)*int(h))
	for i := range data {
		data[i] = 0x55
	}
	info := imagecore.ImageInfo{Width: w, Height: h, ColorType: imagecore.ColorTypeGray8, AlphaType: imagecore.AlphaTypeOpaque}
	img := imagecore.NewRasterData(info, data, int(w))
	if img == nil {
		t.Fatal("nil gray8 image")
	}
	pdfData := renderPDFUncompressed(t, 20, 20, func(c *canvas.Canvas) {
		c.DrawImageRect(img, geom.RectWH(4, 3), geom.RectLTRB(0, 0, 8, 6),
			shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	})
	validatePDF(t, pdfData)
	// A gray-8 image serializes as a single-channel DeviceGray image (no SMask).
	if n := countSubstr(pdfData, "/Subtype /Image"); n != 1 {
		t.Fatalf("expected 1 image XObject, got %d", n)
	}
	mustContain(t, string(pdfData), "/ColorSpace /DeviceGray")
	if bytes.Contains(pdfData, []byte("/SMask")) {
		t.Error("opaque gray image should have no /SMask")
	}
	if !bytes.Contains(pdfData, bytes.Repeat([]byte{0x55}, int(w)*int(h))) {
		t.Error("gray image stream missing the raw gray plane")
	}
}

// TestSerializeAlpha8Image exercises do_deflated_image's alpha-only lane directly (the device draw path routes
// alpha-only images through the luminosity-SMask form-XObject lane instead — see
// TestDeviceAlphaOnlyImageLuminositySMask).
func TestSerializeAlpha8Image(t *testing.T) {
	w, h := int32(3), int32(2)
	alphas := make([]byte, int(w)*int(h))
	for i := range alphas {
		alphas[i] = 0x90
	}
	info := imagecore.ImageInfo{Width: w, Height: h, ColorType: imagecore.ColorTypeAlpha8, AlphaType: imagecore.AlphaTypePremul}
	img := imagecore.NewRasterData(info, alphas, int(w))
	if img == nil {
		t.Fatal("nil alpha8 image")
	}

	md := DefaultMetadata()
	md.CompressionLevel = CompressionNone
	var buf stream.MemoryWStream
	doc := NewDocument(&buf, md)
	doc.BeginPage(20, 20) // emit the header/info dict so serialized objects have somewhere to live
	ref := doc.serializeImageCached(newKeyedImage(img))
	if !ref.IsValid() {
		t.Fatal("invalid image ref")
	}
	// Caching: a second call reuses the same reference.
	if ref2 := doc.serializeImageCached(newKeyedImage(img)); ref2 != ref {
		t.Errorf("expected cached ref %v, got %v", ref, ref2)
	}
	out := buf.Bytes()
	// Color plane (all black) + a DeviceGray SMask carrying the alpha bytes.
	if n := bytes.Count(out, []byte("/Subtype /Image")); n != 2 {
		t.Fatalf("expected color + SMask images, got %d", n)
	}
	mustContain(t, string(out), "/SMask ")
	if !bytes.Contains(out, make([]byte, int(w)*int(h))) {
		t.Error("alpha8 color plane should be all black")
	}
	if !bytes.Contains(out, bytes.Repeat([]byte{0x90}, int(w)*int(h))) {
		t.Error("alpha8 SMask should carry the alpha bytes")
	}
}

// TestNeighborAvgColor checks get_neighbor_avg_color: a fully-transparent pixel takes the average color of its
// non-transparent neighbors so soft-mask resampling does not bleed black.
func TestNeighborAvgColor(t *testing.T) {
	// 3x3, center transparent (word 0), the 8 neighbors opaque (10,20,30).
	pm := pdfPixels{
		width: 3, height: 3, colorType: imagecore.ColorTypeRGBA8888,
		data: make([]byte, 3*3*4),
	}
	for i := 0; i < 9; i++ {
		if i == 4 {
			continue // center stays 0,0,0,0 (transparent)
		}
		pm.data[i*4], pm.data[i*4+1], pm.data[i*4+2], pm.data[i*4+3] = 10, 20, 30, 0xFF
	}
	plane := buildRGBPlane(pm)
	// The center pixel (index 4) becomes the neighbor average = (10,20,30).
	r, g, b := plane[4*3], plane[4*3+1], plane[4*3+2]
	if r != 10 || g != 20 || b != 30 {
		t.Errorf("transparent center = (%d,%d,%d), want (10,20,30)", r, g, b)
	}
}
