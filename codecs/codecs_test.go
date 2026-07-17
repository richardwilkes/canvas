// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package codecs

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"testing"

	"golang.org/x/image/bmp"

	"github.com/richardwilkes/canvas/imagecore"
)

func init() { Register() }

// testNRGBA builds a deterministic 8x6 image with translucency.
func testNRGBA() *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, 8, 6))
	for y := range 6 {
		for x := range 8 {
			m.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x * 32), G: uint8(y * 42), B: uint8((x + y) * 17), A: uint8(255 - x*16),
			})
		}
	}
	return m
}

func decodeVia(t *testing.T, data []byte) *imagecore.Image {
	t.Helper()
	im := imagecore.NewFromEncoded(data)
	if im == nil {
		t.Fatal("NewFromEncoded returned nil")
	}
	return im
}

func TestPNGRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, testNRGBA()); err != nil {
		t.Fatal(err)
	}
	im := decodeVia(t, buf.Bytes())
	if im.Width() != 8 || im.Height() != 6 {
		t.Fatalf("dims %dx%d", im.Width(), im.Height())
	}
	if im.ColorType() != imagecore.ColorTypeRGBA8888 || im.AlphaType() != imagecore.AlphaTypePremul {
		t.Fatalf("info %v %v", im.ColorType(), im.AlphaType())
	}
	p := im.PeekPixels(imagecore.CachingAllow)
	if p == nil {
		t.Fatal("decode failed")
	}
	// premul check with round-to-nearest: pixel (7,0): r=224 a=143 → (224*143+128)*257>>16
	want := (224*143 + 128 + (224*143+128)>>8) >> 8
	if got := p.Words[7] & 0xFF; got != uint32(want) {
		t.Fatalf("premul r = %d want %d", got, want)
	}

	// gray PNG → Gray8 opaque
	g := image.NewGray(image.Rect(0, 0, 3, 3))
	g.Pix[4] = 0x77
	buf.Reset()
	if err := png.Encode(&buf, g); err != nil {
		t.Fatal(err)
	}
	gim := decodeVia(t, buf.Bytes())
	if gim.ColorType() != imagecore.ColorTypeGray8 || gim.AlphaType() != imagecore.AlphaTypeOpaque {
		t.Fatalf("gray info %v %v", gim.ColorType(), gim.AlphaType())
	}
	if gp := gim.PeekPixels(imagecore.CachingAllow); gp.Bytes[4] != 0x77 {
		t.Fatal("gray pixel wrong")
	}
}

func TestEncodePNGRoundTrip(t *testing.T) {
	src := testNRGBA()
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	im := decodeVia(t, buf.Bytes())
	for _, level := range []int{0, 1, 6, 9} {
		enc := EncodePNG(im, level)
		if enc == nil {
			t.Fatalf("EncodePNG(%d) nil", level)
		}
		back := decodeVia(t, enc)
		bp := back.PeekPixels(imagecore.CachingAllow)
		ip := im.PeekPixels(imagecore.CachingAllow)
		for i := range ip.Words {
			// premul → unpremul → premul round trip may wiggle low bits by 1
			for shift := 0; shift < 32; shift += 8 {
				a := int(ip.Words[i]>>shift) & 0xFF
				b := int(bp.Words[i]>>shift) & 0xFF
				if d := a - b; d < -1 || d > 1 {
					t.Fatalf("level %d pixel %d differs: %08x vs %08x", level, i, ip.Words[i], bp.Words[i])
				}
			}
		}
	}
}

func TestJPEGAndEncode(t *testing.T) {
	src := testNRGBA()
	for i := range src.Pix {
		if i%4 == 3 {
			src.Pix[i] = 0xFF
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}
	im := decodeVia(t, buf.Bytes())
	if im.ColorType() != imagecore.ColorTypeRGBA8888 || im.AlphaType() != imagecore.AlphaTypeOpaque {
		t.Fatalf("info %v %v", im.ColorType(), im.AlphaType())
	}
	enc := EncodeJPEG(im, 90)
	if enc == nil || !bytes.HasPrefix(enc, []byte{0xFF, 0xD8}) {
		t.Fatal("EncodeJPEG failed")
	}
	if back := decodeVia(t, enc); back.Width() != 8 || back.Height() != 6 {
		t.Fatal("re-decode dims wrong")
	}
}

func TestJPEGEXIFOrientation(t *testing.T) {
	// Encode a 4x2 luminance gradient (chroma subsampling would smear color channels at this size), splice in an EXIF
	// APP1 with orientation 6 (rotate 90 CW).
	src := image.NewNRGBA(image.Rect(0, 0, 4, 2))
	for x := range 4 {
		bright := uint8(40 + 50*x)
		src.SetNRGBA(x, 0, color.NRGBA{R: bright, G: bright, B: bright, A: 255})
		src.SetNRGBA(x, 1, color.NRGBA{R: 20, G: 20, B: 20, A: 255})
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatal(err)
	}
	plain := buf.Bytes()

	// Minimal EXIF payload: "Exif\0\0" + TIFF LE header + one IFD entry (0x0112 SHORT 1 value 6).
	tiff := []byte{
		'I', 'I', 42, 0, 8, 0, 0, 0, // header, IFD at offset 8
		1, 0, // one entry
		0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, // orientation = 6
		0, 0, 0, 0, // next IFD
	}
	payload := append([]byte("Exif\x00\x00"), tiff...)
	app1 := append([]byte{0xFF, 0xE1, byte((len(payload) + 2) >> 8), byte(len(payload) + 2)}, payload...)
	withEXIF := append(append(append([]byte{}, plain[:2]...), app1...), plain[2:]...)

	if got := jpegOrigin(withEXIF); got != 6 {
		t.Fatalf("jpegOrigin = %d", got)
	}
	im := decodeVia(t, withEXIF)
	if im.Width() != 2 || im.Height() != 4 {
		t.Fatalf("oriented dims %dx%d", im.Width(), im.Height())
	}
	p := im.PeekPixels(imagecore.CachingAllow)
	if p == nil || p.Info.Width != 2 || p.Info.Height != 4 {
		t.Fatal("oriented decode failed")
	}
	// Rotate 90 CW: src(3,0) (brightest, ~190) lands at dst(1,3); src(0,0) (~40) at dst(1,0); row 1 (dark ~20) becomes
	// column 0. JPEG is lossy so accept slop.
	bright := p.Words[3*int(p.RowElems)+1] & 0xFF
	dim := p.Words[0*int(p.RowElems)+1] & 0xFF
	dark := p.Words[3*int(p.RowElems)+0] & 0xFF
	if bright < 150 || dim > 90 || dark > 90 {
		t.Fatalf("rotated pixels bright=%d dim=%d dark=%d", bright, dim, dark)
	}
}

func TestOrientPixelsMappings(t *testing.T) {
	info, _ := imagecore.MakeInfo(3, 2, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	src := imagecore.NewPixels(info)
	for i := range 6 {
		src.Words[i] = uint32(i + 1)
	}
	// origin 3 = rotate 180
	out := orientPixels(src, 3)
	if out.Words[0] != 6 || out.Words[5] != 1 {
		t.Fatalf("rot180: %v", out.Words)
	}
	// origin 6 = rotate 90 CW: dst is 2x3; src(0,0) → dst(1,0)
	out = orientPixels(src, 6)
	if out.Info.Width != 2 || out.Info.Height != 3 {
		t.Fatal("rot90 dims")
	}
	if out.Words[1] != 1 || out.Words[0] != 4 {
		t.Fatalf("rot90: %v", out.Words)
	}
}

func TestGIFFirstFrame(t *testing.T) {
	pal := color.Palette{color.NRGBA{A: 0}, color.NRGBA{R: 255, A: 255}, color.NRGBA{G: 255, A: 255}}
	frame := image.NewPaletted(image.Rect(0, 0, 4, 4), pal)
	for i := range frame.Pix {
		frame.Pix[i] = 1
	}
	frame.Pix[0] = 0 // transparent pixel
	var buf bytes.Buffer
	if err := gif.EncodeAll(&buf, &gif.GIF{Image: []*image.Paletted{frame}, Delay: []int{0}}); err != nil {
		t.Fatal(err)
	}
	im := decodeVia(t, buf.Bytes())
	if im.AlphaType() != imagecore.AlphaTypePremul {
		t.Fatalf("gif alpha %v", im.AlphaType())
	}
	p := im.PeekPixels(imagecore.CachingAllow)
	if p.Words[0] != 0 || p.Words[1]&0xFF != 255 {
		t.Fatalf("gif pixels %08x %08x", p.Words[0], p.Words[1])
	}
}

func TestBMP(t *testing.T) {
	src := testNRGBA()
	for i := range src.Pix {
		if i%4 == 3 {
			src.Pix[i] = 0xFF
		}
	}
	var buf bytes.Buffer
	if err := bmp.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	im := decodeVia(t, buf.Bytes())
	if im.Width() != 8 || im.Height() != 6 {
		t.Fatal("bmp dims")
	}
	p := im.PeekPixels(imagecore.CachingAllow)
	if p.Words[1]&0xFF != 32 {
		t.Fatalf("bmp pixel %08x", p.Words[1])
	}
}

func TestWebPEncodeDecode(t *testing.T) {
	src := testNRGBA()
	im0 := imagecore.NewRasterData(
		mustInfo(t, 8, 6, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeUnpremul),
		src.Pix, src.Stride,
	)
	if im0 == nil {
		t.Fatal("raster image nil")
	}
	enc := EncodeWebP(im0, 100, false)
	if enc == nil {
		t.Fatal("EncodeWebP nil")
	}
	if EncodeWebP(im0, 80, true) == nil {
		t.Fatal("lossy encode failed")
	}
	im := decodeVia(t, enc)
	if im.Width() != 8 || im.Height() != 6 {
		t.Fatal("webp dims")
	}
	if !webpHasAlpha(enc) {
		t.Fatal("alpha bit not sniffed")
	}
	if im.AlphaType() != imagecore.AlphaTypePremul {
		t.Fatalf("webp alpha %v", im.AlphaType())
	}
	// Lossless round trip: unpremul source → premul decode must equal round-to-nearest premul.
	p := im.PeekPixels(imagecore.CachingAllow)
	want := (224*143 + 128 + (224*143+128)>>8) >> 8
	if got := p.Words[7] & 0xFF; got != uint32(want) {
		t.Fatalf("webp premul r = %d want %d", got, want)
	}
}

func mustInfo(t *testing.T, w, h int32, ct imagecore.ColorType, at imagecore.AlphaType) imagecore.ImageInfo {
	t.Helper()
	info, ok := imagecore.MakeInfo(w, h, ct, at)
	if !ok {
		t.Fatal("bad info")
	}
	return info
}

func TestICO(t *testing.T) {
	// Synthesize a 2-entry ICO: a 2x2 32bpp DIB and a 4x4 32bpp DIB (larger wins).
	makeDIB := func(n int, argb uint32) []byte {
		var b bytes.Buffer
		hdr := make([]byte, 40)
		binary.LittleEndian.PutUint32(hdr, 40)
		binary.LittleEndian.PutUint32(hdr[4:], uint32(n))
		binary.LittleEndian.PutUint32(hdr[8:], uint32(2*n))
		binary.LittleEndian.PutUint16(hdr[12:], 1)
		binary.LittleEndian.PutUint16(hdr[14:], 32)
		b.Write(hdr)
		px := make([]byte, 4)
		binary.LittleEndian.PutUint32(px, argb)
		for range n * n {
			b.Write(px)
		}
		andStride := ((n + 31) / 32) * 4
		b.Write(make([]byte, andStride*n))
		return b.Bytes()
	}
	small := makeDIB(2, 0xFF0000FF) // opaque blue (B in byte 0)
	big := makeDIB(4, 0x80FF0000)   // half-alpha red

	var ico bytes.Buffer
	ico.Write([]byte{0, 0, 1, 0, 2, 0})
	off := 6 + 32
	for i, e := range [][]byte{small, big} {
		n := byte(2 + 2*i)
		ent := []byte{n, n, 0, 0, 1, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0}
		binary.LittleEndian.PutUint32(ent[8:], uint32(len(e)))
		binary.LittleEndian.PutUint32(ent[12:], uint32(off))
		ico.Write(ent)
		off += len(e)
	}
	ico.Write(small)
	ico.Write(big)

	im := decodeVia(t, ico.Bytes())
	if im.Width() != 4 || im.Height() != 4 {
		t.Fatalf("ico dims %dx%d", im.Width(), im.Height())
	}
	p := im.PeekPixels(imagecore.CachingAllow)
	if p == nil {
		t.Fatal("ico decode failed")
	}
	// half-alpha red premultiplied: r = (255*128+128)*257>>16 = 128
	if got := p.Words[0]; got != 0x80000080 {
		t.Fatalf("ico pixel %08x", got)
	}
}

func TestWBMP(t *testing.T) {
	// 10x2: row0 = 0b1010101010......, row1 all black
	data := []byte{0, 0, 10, 2, 0xAA, 0x80, 0x00, 0x00}
	im := decodeVia(t, data)
	if im.ColorType() != imagecore.ColorTypeGray8 || im.Width() != 10 || im.Height() != 2 {
		t.Fatalf("wbmp info %v %dx%d", im.ColorType(), im.Width(), im.Height())
	}
	p := im.PeekPixels(imagecore.CachingAllow)
	if p.Bytes[0] != 0xFF || p.Bytes[1] != 0 || p.Bytes[8] != 0xFF || p.Bytes[10] != 0 {
		t.Fatalf("wbmp bits % x", p.Bytes)
	}
}

func TestUnregisteredReturnsNil(t *testing.T) {
	if imagecore.NewFromEncoded([]byte{1, 2, 3, 4}) != nil {
		t.Fatal("garbage should not decode")
	}
}
