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
	"hash/crc32"
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

// exifAPP1 builds an APP1 segment holding a minimal EXIF payload: "Exif\0\0" + TIFF LE header + one IFD entry
// (0x0112 SHORT 1 value orientation).
func exifAPP1(orientation byte) []byte {
	tiff := []byte{
		'I', 'I', 42, 0, 8, 0, 0, 0, // header, IFD at offset 8
		1, 0, // one entry
		0x12, 0x01, 3, 0, 1, 0, 0, 0, orientation, 0, 0, 0,
		0, 0, 0, 0, // next IFD
	}
	payload := append([]byte("Exif\x00\x00"), tiff...)
	return append([]byte{0xFF, 0xE1, byte((len(payload) + 2) >> 8), byte(len(payload) + 2)}, payload...)
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

	app1 := exifAPP1(6) // orientation 6 = rotate 90 CW
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

func TestJPEGGrayEXIFOrientation(t *testing.T) {
	// A grayscale JPEG decodes to Gray8 (Bytes-backed). With a transposing orientation, DecodeInfo swaps the
	// dimensions; Decode must return pixels at those same swapped dimensions (and actually rotated), not the original.
	src := image.NewGray(image.Rect(0, 0, 4, 2))
	for x := range 4 {
		src.SetGray(x, 0, color.Gray{Y: uint8(40 + 50*x)})
		src.SetGray(x, 1, color.Gray{Y: 20})
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatal(err)
	}
	plain := buf.Bytes()

	app1 := exifAPP1(6) // orientation 6 = rotate 90 CW
	withEXIF := append(append(append([]byte{}, plain[:2]...), app1...), plain[2:]...)

	codec := jpegCodec()
	info, ok := codec.DecodeInfo(withEXIF)
	if !ok || info.ColorType != imagecore.ColorTypeGray8 {
		t.Fatalf("DecodeInfo gray=%v ok=%v", info.ColorType, ok)
	}
	if info.Width != 2 || info.Height != 4 {
		t.Fatalf("DecodeInfo dims %dx%d, want 2x4", info.Width, info.Height)
	}
	p, err := codec.Decode(withEXIF)
	if err != nil {
		t.Fatal(err)
	}
	if p.Bytes == nil {
		t.Fatal("expected Bytes-backed Gray8 pixels")
	}
	// Decode dims must match DecodeInfo dims (the mismatch this test guards against).
	if p.Info.Width != info.Width || p.Info.Height != info.Height {
		t.Fatalf("Decode dims %dx%d != DecodeInfo %dx%d", p.Info.Width, p.Info.Height, info.Width, info.Height)
	}
	// Rotate 90 CW: src(3,0) (brightest, ~190) → dst(1,3); src(0,0) (~40) → dst(1,0); row 1 (dark ~20) → column 0.
	bright := p.Bytes[3*int(p.RowElems)+1]
	dim := p.Bytes[0*int(p.RowElems)+1]
	dark := p.Bytes[3*int(p.RowElems)+0]
	if bright < 150 || dim > 90 || dark > 90 {
		t.Fatalf("rotated gray bright=%d dim=%d dark=%d", bright, dim, dark)
	}
}

func TestJPEGEXIFOrientationWithFillBytes(t *testing.T) {
	// ITU T.81 §B.1.1.2 permits any number of 0xFF fill bytes ahead of any marker, and image/jpeg accepts them, so the
	// segment walk must skip them rather than mistake a fill byte for the marker and give up.
	src := image.NewGray(image.Rect(0, 0, 4, 2))
	for x := range 4 {
		src.SetGray(x, 0, color.Gray{Y: uint8(40 + 50*x)})
		src.SetGray(x, 1, color.Gray{Y: 20})
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatal(err)
	}
	plain := buf.Bytes()

	// SOI, fill + a comment segment (so a fill run before a skipped segment is exercised too), fill + APP1 Exif.
	withFill := append([]byte{}, plain[:2]...)
	withFill = append(withFill, 0xFF, 0xFF, 0xFF, 0xFE, 0, 4, 'h', 'i', 0xFF, 0xFF)
	withFill = append(withFill, exifAPP1(6)...) // orientation 6 = rotate 90 CW
	withFill = append(withFill, plain[2:]...)

	if got := jpegOrigin(withFill); got != 6 {
		t.Fatalf("jpegOrigin = %d, want 6", got)
	}
	codec := jpegCodec()
	info, ok := codec.DecodeInfo(withFill)
	if !ok {
		t.Fatal("DecodeInfo failed")
	}
	if info.Width != 2 || info.Height != 4 {
		t.Fatalf("DecodeInfo dims %dx%d, want 2x4", info.Width, info.Height)
	}
	p, err := codec.Decode(withFill)
	if err != nil {
		t.Fatal(err)
	}
	if p.Info.Width != 2 || p.Info.Height != 4 {
		t.Fatalf("Decode dims %dx%d, want 2x4", p.Info.Width, p.Info.Height)
	}
	// Rotate 90 CW: src(3,0) (brightest, ~190) → dst(1,3); src(0,0) (~40) → dst(1,0); row 1 (dark ~20) → column 0.
	bright := p.Bytes[3*int(p.RowElems)+1]
	dim := p.Bytes[0*int(p.RowElems)+1]
	dark := p.Bytes[3*int(p.RowElems)+0]
	if bright < 150 || dim > 90 || dark > 90 {
		t.Fatalf("rotated gray bright=%d dim=%d dark=%d", bright, dim, dark)
	}
}

func TestJPEGOriginMalformed(t *testing.T) {
	// Truncated and fill-only tails must fall back to orientation 1 rather than panic or read out of bounds.
	for i, data := range [][]byte{
		{0xFF, 0xD8},
		{0xFF, 0xD8, 0xFF},
		{0xFF, 0xD8, 0xFF, 0xFF},
		{0xFF, 0xD8, 0xFF, 0xFF, 0xFF, 0xFF},
		{0xFF, 0xD8, 0xFF, 0xFF, 0xE1},
		{0xFF, 0xD8, 0xFF, 0xE1, 0x00},
		{0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x01}, // segLen < 2
		{0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF}, // segLen past the end
		{0xFF, 0xD8, 0xFF, 0xD0, 0xFF, 0xD9}, // standalone markers only
		{0xFF, 0xD8, 0x00, 0xE1, 0x00, 0x08}, // missing marker prefix
		{0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02}, // SOS: stop scanning
	} {
		if got := jpegOrigin(data); got != 1 {
			t.Fatalf("case %d: jpegOrigin = %d, want 1", i, got)
		}
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

func TestOrientPixelsGray8(t *testing.T) {
	// Gray8 pixels are Bytes-backed (Words == nil); orientation must still be applied.
	info, _ := imagecore.MakeInfo(3, 2, imagecore.ColorTypeGray8, imagecore.AlphaTypeOpaque)
	src := imagecore.NewPixels(info)
	if src.Words != nil || src.Bytes == nil {
		t.Fatal("expected Bytes-backed Gray8 pixels")
	}
	for i := range 6 {
		src.Bytes[i] = byte(i + 1)
	}
	// origin 3 = rotate 180: first and last swap ends.
	out := orientPixels(src, 3)
	if out.Words != nil || out.Bytes[0] != 6 || out.Bytes[5] != 1 {
		t.Fatalf("rot180: %v", out.Bytes)
	}
	// origin 6 = rotate 90 CW: dst is 2x3; src(0,0)=1 → dst(1,0), src(0,1)=4 → dst(0,0).
	out = orientPixels(src, 6)
	if out.Info.Width != 2 || out.Info.Height != 3 {
		t.Fatalf("rot90 dims %dx%d", out.Info.Width, out.Info.Height)
	}
	if out.Bytes[1] != 1 || out.Bytes[0] != 4 {
		t.Fatalf("rot90: %v", out.Bytes)
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

// makeV4BMP builds a 32bpp BMP with a BITMAPV4HEADER, the only 32bpp layout whose alpha channel x/image/bmp honors
// (bmp.Encode always writes the 40-byte BITMAPINFOHEADER, whose alpha is forced opaque on decode). px holds
// unpremultiplied RGBA in top-down row order.
func makeV4BMP(w, h int, px []color.NRGBA) []byte {
	const fileHeaderLen, v4InfoHeaderLen = 14, 108
	stride := 4 * w
	out := make([]byte, fileHeaderLen+v4InfoHeaderLen+stride*h)
	copy(out, "BM")
	binary.LittleEndian.PutUint32(out[2:], uint32(len(out)))
	binary.LittleEndian.PutUint32(out[10:], fileHeaderLen+v4InfoHeaderLen) // pixel offset
	binary.LittleEndian.PutUint32(out[14:], v4InfoHeaderLen)
	binary.LittleEndian.PutUint32(out[18:], uint32(w))
	binary.LittleEndian.PutUint32(out[22:], uint32(h))
	binary.LittleEndian.PutUint16(out[26:], 1)  // planes
	binary.LittleEndian.PutUint16(out[28:], 32) // bpp
	binary.LittleEndian.PutUint32(out[34:], uint32(stride*h))
	binary.LittleEndian.PutUint32(out[54:], 0x00FF0000) // red mask
	binary.LittleEndian.PutUint32(out[58:], 0x0000FF00) // green mask
	binary.LittleEndian.PutUint32(out[62:], 0x000000FF) // blue mask
	binary.LittleEndian.PutUint32(out[66:], 0xFF000000) // alpha mask
	for y := range h {
		row := out[fileHeaderLen+v4InfoHeaderLen+(h-1-y)*stride:] // bottom-up
		for x := range w {
			c := px[y*w+x]
			row[4*x], row[4*x+1], row[4*x+2], row[4*x+3] = c.B, c.G, c.R, c.A // BGRA
		}
	}
	return out
}

func TestBMPAlpha(t *testing.T) {
	// A 32bpp V4 BMP carries real transparency, so both the reported info and the decoded pixels must say premul.
	px := []color.NRGBA{
		{R: 10, G: 20, B: 30, A: 0},
		{R: 40, G: 50, B: 60, A: 0xFF},
		{R: 200, G: 100, B: 50, A: 128},
		{R: 1, G: 2, B: 3, A: 0xFF},
	}
	im := decodeVia(t, makeV4BMP(2, 2, px))
	if im.Width() != 2 || im.Height() != 2 {
		t.Fatalf("v4 dims %dx%d", im.Width(), im.Height())
	}
	if im.AlphaType() != imagecore.AlphaTypePremul {
		t.Fatalf("v4 alpha %v", im.AlphaType())
	}
	p := im.PeekPixels(imagecore.CachingAllow)
	if p.Info.AlphaType != im.AlphaType() || p.Info.ColorType != im.ColorType() {
		t.Fatalf("v4 decoded info %v/%v != reported %v/%v", p.Info.ColorType, p.Info.AlphaType, im.ColorType(),
			im.AlphaType())
	}
	if p.Words[0] != 0 {
		t.Fatalf("v4 transparent pixel %08x", p.Words[0])
	}
	if got := p.Words[2] >> 24; got != 128 {
		t.Fatalf("v4 translucent alpha %d", got)
	}
	if got := p.Words[2] & 0xFF; got != uint32((200*128+128+(200*128+128)>>8)>>8) { // premul red, round-to-nearest
		t.Fatalf("v4 translucent red %d", got)
	}

	// A 32bpp BITMAPINFOHEADER BMP (what bmp.Encode writes for a translucent image) has its alpha forced opaque by the
	// decoder, so the reported info must say opaque rather than trusting the *image.NRGBA it decodes to.
	var buf bytes.Buffer
	if err := bmp.Encode(&buf, testNRGBA()); err != nil {
		t.Fatal(err)
	}
	if bpp := binary.LittleEndian.Uint16(buf.Bytes()[28:]); bpp != 32 {
		t.Fatalf("expected a 32bpp encode, got %dbpp", bpp)
	}
	im = decodeVia(t, buf.Bytes())
	if im.AlphaType() != imagecore.AlphaTypeOpaque {
		t.Fatalf("v3 alpha %v", im.AlphaType())
	}
	p = im.PeekPixels(imagecore.CachingAllow)
	if p.Info.AlphaType != im.AlphaType() {
		t.Fatalf("v3 decoded alpha %v != reported %v", p.Info.AlphaType, im.AlphaType())
	}
	for i, w := range p.Words {
		if w>>24 != 0xFF {
			t.Fatalf("v3 pixel %d not opaque: %08x", i, w)
		}
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
	if !webpIsLossless(enc) {
		t.Fatal("lossless payload not sniffed")
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

func TestICOOverflowRejected(t *testing.T) {
	// A crafted ~62-byte ICO whose BITMAPINFOHEADER declares near-int32-max width/height with 32bpp used to overflow
	// xorStride*h, bypassing the short-pixel-data guard and reaching make([]px, w*h) with a ~2.3e18 length (panic or
	// memory exhaustion). The dimension cap in dibHeader must reject it before any of that arithmetic runs.
	dib := make([]byte, 40)
	binary.LittleEndian.PutUint32(dib, 40)
	binary.LittleEndian.PutUint32(dib[4:], 2147483647) // w
	binary.LittleEndian.PutUint32(dib[8:], 2147483646) // h2 (positive, even)
	binary.LittleEndian.PutUint16(dib[12:], 1)         // planes
	binary.LittleEndian.PutUint16(dib[14:], 32)        // bpp

	// dibHeader must reject the oversized dimensions outright.
	if _, _, _, ok := dibHeader(dib); ok {
		t.Fatal("dibHeader accepted oversized dimensions")
	}

	// decodeDIB must fail cleanly rather than panic when handed the same payload directly.
	info := makeDecodedInfo(2147483647, 1073741823, false, true)
	if _, err := decodeDIB(dib, info); err == nil {
		t.Fatal("decodeDIB accepted oversized DIB")
	}

	// The full decode path must return nil (not panic) for the crafted ICO.
	var ico bytes.Buffer
	ico.Write([]byte{0, 0, 1, 0, 1, 0}) // ICONDIR: 1 entry
	ent := []byte{0, 0, 0, 0, 1, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	binary.LittleEndian.PutUint32(ent[8:], uint32(len(dib))) // size
	binary.LittleEndian.PutUint32(ent[12:], 6+16)            // offset
	ico.Write(ent)
	ico.Write(dib)
	if im := imagecore.NewFromEncoded(ico.Bytes()); im != nil {
		t.Fatal("oversized ICO should not decode")
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

// insertPNGChunk splices a chunk into a stdlib-encoded PNG immediately ahead of the first IDAT, which is where tRNS
// belongs (after IHDR, and after PLTE for a paletted image). png.Encode never emits tRNS for the non-paletted color
// types, so hand-splicing is the only way to build one.
func insertPNGChunk(t *testing.T, data []byte, typ string, payload []byte) []byte {
	t.Helper()
	idat := bytes.Index(data, []byte("IDAT"))
	if idat < 4 {
		t.Fatal("no IDAT chunk found")
	}
	chunk := binary.BigEndian.AppendUint32(make([]byte, 0, 12+len(payload)), uint32(len(payload)))
	chunk = append(chunk, typ...)
	chunk = append(chunk, payload...)
	chunk = binary.BigEndian.AppendUint32(chunk, crc32.ChecksumIEEE(chunk[4:]))
	out := make([]byte, 0, len(data)+len(chunk))
	out = append(out, data[:idat-4]...) // the chunk's 4-byte length precedes its type
	out = append(out, chunk...)
	return append(out, data[idat-4:]...)
}

// checkPNGTRNS decodes data through the png codec and asserts both lanes agree on N32/premul and produce want.
func checkPNGTRNS(t *testing.T, data []byte, want []uint32) {
	t.Helper()
	codec := pngCodec()
	info, ok := codec.DecodeInfo(data)
	if !ok {
		t.Fatal("DecodeInfo failed")
	}
	if info.ColorType != imagecore.ColorTypeRGBA8888 || info.AlphaType != imagecore.AlphaTypePremul {
		t.Fatalf("DecodeInfo reported %v/%v, want RGBA8888/premul", info.ColorType, info.AlphaType)
	}
	p, err := codec.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	if p.Info.ColorType != info.ColorType || p.Info.AlphaType != info.AlphaType {
		t.Fatalf("Decode info %v/%v disagrees with DecodeInfo %v/%v", p.Info.ColorType, p.Info.AlphaType,
			info.ColorType, info.AlphaType)
	}
	for i, w := range want {
		if p.Words[i] != w {
			t.Fatalf("pixel %d = %08x, want %08x", i, p.Words[i], w)
		}
	}
	im := decodeVia(t, data)
	if im.IsOpaque() {
		t.Fatal("IsOpaque() is true for an image holding transparent pixels")
	}
}

// TestPNGGrayWithTRNS covers the tRNS chunk on a grayscale PNG: png.DecodeConfig stops at IHDR and reports
// color.GrayModel, but png.Decode honors tRNS and returns an *image.NRGBA. Reporting Gray8/opaque from the config
// alone ran every pixel through color.GrayModel.Convert, whose RGBA() premultiplies, so a transparent pixel decoded to
// opaque black and the alpha vanished.
func TestPNGGrayWithTRNS(t *testing.T) {
	g := image.NewGray(image.Rect(0, 0, 2, 1))
	g.Pix[0] = 0x00
	g.Pix[1] = 0xFF
	var buf bytes.Buffer
	if err := png.Encode(&buf, g); err != nil {
		t.Fatal(err)
	}
	// tRNS for 8-bit grayscale is one 16-bit sample; gray level 0 becomes fully transparent.
	data := insertPNGChunk(t, buf.Bytes(), "tRNS", []byte{0x00, 0x00})
	checkPNGTRNS(t, data, []uint32{0x00000000, 0xFFFFFFFF})
}

// TestPNGTruecolorWithTRNS covers the same defect on the truecolor lane, where DecodeConfig reports color.RGBAModel
// and png.Decode again returns an *image.NRGBA.
func TestPNGTruecolorWithTRNS(t *testing.T) {
	m := image.NewRGBA(image.Rect(0, 0, 2, 1))
	m.SetRGBA(0, 0, color.RGBA{R: 0x11, G: 0x22, B: 0x33, A: 0xFF})
	m.SetRGBA(1, 0, color.RGBA{R: 0x44, G: 0x55, B: 0x66, A: 0xFF})
	var buf bytes.Buffer
	if err := png.Encode(&buf, m); err != nil { // opaque, so png.Encode picks truecolor with no alpha channel
		t.Fatal(err)
	}
	// tRNS for 8-bit truecolor is three 16-bit samples; color 0x112233 becomes fully transparent.
	data := insertPNGChunk(t, buf.Bytes(), "tRNS", []byte{0, 0x11, 0, 0x22, 0, 0x33})
	checkPNGTRNS(t, data, []uint32{0x00000000, 0xFF665544})
}

// TestPNGPalettedTRNSUnaffected guards the other direction: DecodeConfig does parse tRNS for paletted PNGs, so the
// palette it reports already carries the alpha and an all-opaque tRNS must not force a premul alpha type.
func TestPNGPalettedTRNSUnaffected(t *testing.T) {
	pal := image.NewPaletted(image.Rect(0, 0, 2, 1), color.Palette{
		color.RGBA{R: 0xFF, A: 0xFF},
		color.RGBA{B: 0xFF, A: 0xFF},
	})
	pal.Pix[1] = 1
	var buf bytes.Buffer
	if err := png.Encode(&buf, pal); err != nil {
		t.Fatal(err)
	}
	data := insertPNGChunk(t, buf.Bytes(), "tRNS", []byte{0xFF, 0xFF}) // both entries fully opaque
	info, ok := pngCodec().DecodeInfo(data)
	if !ok {
		t.Fatal("DecodeInfo failed")
	}
	if info.ColorType != imagecore.ColorTypeRGBA8888 || info.AlphaType != imagecore.AlphaTypeOpaque {
		t.Fatalf("DecodeInfo reported %v/%v, want RGBA8888/opaque", info.ColorType, info.AlphaType)
	}
}

// TestICOPNGEntryWithTRNS covers the PNG-compressed ICO directory entry, which shares pngModelInfo with the PNG codec
// and inherited the same tRNS defect.
func TestICOPNGEntryWithTRNS(t *testing.T) {
	g := image.NewGray(image.Rect(0, 0, 2, 2))
	for i := range g.Pix {
		g.Pix[i] = 0xFF
	}
	g.Pix[0] = 0x00
	var buf bytes.Buffer
	if err := png.Encode(&buf, g); err != nil {
		t.Fatal(err)
	}
	sub := insertPNGChunk(t, buf.Bytes(), "tRNS", []byte{0x00, 0x00})

	var ico bytes.Buffer
	ico.Write([]byte{0, 0, 1, 0, 1, 0}) // ICONDIR: 1 entry
	ent := []byte{2, 2, 0, 0, 1, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	binary.LittleEndian.PutUint32(ent[8:], uint32(len(sub)))
	binary.LittleEndian.PutUint32(ent[12:], 6+16)
	ico.Write(ent)
	ico.Write(sub)

	im := decodeVia(t, ico.Bytes())
	if im.ColorType() != imagecore.ColorTypeRGBA8888 || im.IsOpaque() {
		t.Fatalf("ico png entry %v opaque=%v", im.ColorType(), im.IsOpaque())
	}
	if p := im.PeekPixels(imagecore.CachingAllow); p.Words[0] != 0 || p.Words[1] != 0xFFFFFFFF {
		t.Fatalf("ico png entry pixels %08x %08x", p.Words[0], p.Words[1])
	}
}

// TestGIFAnimationFirstFrameOnly pins that only frame 0 is decoded: the payload is truncated right after the first
// frame's data, which gif.DecodeAll rejects (it keeps reading for more frames) but gif.Decode accepts.
func TestGIFAnimationFirstFrameOnly(t *testing.T) {
	pal := color.Palette{color.NRGBA{A: 0}, color.NRGBA{R: 255, A: 255}, color.NRGBA{B: 255, A: 255}}
	frames := make([]*image.Paletted, 3)
	for i := range frames {
		f := image.NewPaletted(image.Rect(0, 0, 4, 4), pal)
		for j := range f.Pix {
			f.Pix[j] = uint8(1 + i%2)
		}
		f.Pix[0] = 0 // transparent
		frames[i] = f
	}
	var buf bytes.Buffer
	if err := gif.EncodeAll(&buf, &gif.GIF{Image: frames, Delay: []int{0, 0, 0}}); err != nil {
		t.Fatal(err)
	}
	full := buf.Bytes()

	// The whole animation still decodes to frame 0.
	im := decodeVia(t, full)
	p := im.PeekPixels(imagecore.CachingAllow)
	if p.Words[0] != 0 || p.Words[1] != 0xFF0000FF {
		t.Fatalf("frame 0 pixels %08x %08x", p.Words[0], p.Words[1])
	}

	// Cutting the stream short of frame 1's image descriptor leaves frame 0 intact and decodable.
	second := bytes.IndexByte(full[13:], 0x2C) // first image descriptor
	if second < 0 {
		t.Fatal("no image descriptor found")
	}
	third := bytes.IndexByte(full[13+second+1:], 0x2C)
	if third < 0 {
		t.Fatal("only one image descriptor found")
	}
	truncated := full[:13+second+1+third]
	if _, err := gif.DecodeAll(bytes.NewReader(truncated)); err == nil {
		t.Fatal("truncation is not past the first frame; the test no longer proves anything")
	}
	tim := decodeVia(t, truncated)
	tp := tim.PeekPixels(imagecore.CachingAllow)
	if tp.Words[0] != 0 || tp.Words[1] != 0xFF0000FF {
		t.Fatalf("truncated frame 0 pixels %08x %08x", tp.Words[0], tp.Words[1])
	}
}

// hostilePNG builds a minimal PNG whose IHDR declares w x h and which carries no image data at all. png.DecodeConfig
// parses it happily — Go rejects a PNG's dimensions only when the pixel count overflows an int — so a few dozen bytes
// are enough to hand a codec dimensions no allocation could ever satisfy.
func hostilePNG(w, h uint32) []byte {
	ihdr := binary.BigEndian.AppendUint32(make([]byte, 0, 13), w)
	ihdr = binary.BigEndian.AppendUint32(ihdr, h)
	ihdr = append(ihdr, 8, 6, 0, 0, 0) // depth 8, truecolor+alpha, deflate, adaptive filtering, no interlace
	out := append(make([]byte, 0, 33), pngMagic...)
	out = binary.BigEndian.AppendUint32(out, uint32(len(ihdr)))
	out = append(out, "IHDR"...)
	out = append(out, ihdr...)
	return binary.BigEndian.AppendUint32(out, crc32.ChecksumIEEE(out[len(out)-len(ihdr)-4:]))
}

// TestNewFromEncodedRejectsHostileDimensions pins the validation on the codec-reported info. A codec reports whatever
// the encoded header declares, so before that check a 33-byte PNG produced an *Image with Width == 1<<30 that panicked
// with "makeslice: len out of range" on every subsequent use — while NewRasterData given the same info returned nil.
// The ICO lane inherits the hole through its PNG directory entries: its own DIB decoder is bounded by
// maxICODimension, but the delegating png.DecodeConfig is not.
func TestNewFromEncodedRejectsHostileDimensions(t *testing.T) {
	const w, h = 1 << 30, 1 << 28
	data := hostilePNG(w, h)
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("fixture must be a PNG the stdlib config parser accepts, else it proves nothing: %v", err)
	}
	if cfg.Width != w || cfg.Height != h {
		t.Fatalf("fixture config = %dx%d, want %dx%d", cfg.Width, cfg.Height, w, h)
	}
	if im := imagecore.NewFromEncoded(data); im != nil {
		t.Fatalf("PNG declaring %dx%d produced an image reporting %dx%d", w, h, im.Width(), im.Height())
	}

	// The same PNG as an ICO directory entry. The entry must still be the one icoBest picks, so that the rejection
	// is NewFromEncoded's and not an accident of the entry being dropped.
	ico := []byte{0, 0, 1, 0, 1, 0} // reserved, type 1 (icon), one entry
	entry := []byte{0, 0, 0, 0, 1, 0, 32, 0}
	entry = binary.LittleEndian.AppendUint32(entry, uint32(len(data)))
	entry = binary.LittleEndian.AppendUint32(entry, 22) // 6-byte header + one 16-byte directory entry
	ico = append(append(ico, entry...), data...)
	if _, ok := icoCodec().DecodeInfo(ico); !ok {
		t.Fatal("fixture must reach the ICO codec's reported info, else it proves nothing")
	}
	if im := imagecore.NewFromEncoded(ico); im != nil {
		t.Fatalf("ICO wrapping the same PNG produced an image reporting %dx%d", im.Width(), im.Height())
	}
}

// TestEncodersRejectHostileInfoBeforeAllocating drives each encoder with an image whose reported info declares
// dimensions no buffer can hold. ReadPixels is the only thing that validates those dimensions, and it runs after the
// read-back buffer has already been sized from them, so without the guard ahead of the allocation every encoder here
// panics inside image.New{,N}RGBA.
func TestEncodersRejectHostileInfoBeforeAllocating(t *testing.T) {
	p := imagecore.NewPixels(mustInfo(t, 1, 1, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul))
	p.Info.Width = 1 << 30
	p.Info.Height = 1 << 30
	img := imagecore.FromPixels(p)
	for _, tc := range []struct {
		encode func() []byte
		name   string
	}{
		{name: "EncodePNG", encode: func() []byte { return EncodePNG(img, 6) }},
		{name: "EncodeJPEG", encode: func() []byte { return EncodeJPEG(img, 80) }},
		{name: "EncodeWebP lossless", encode: func() []byte { return EncodeWebP(img, 100, false) }},
		{name: "EncodeWebP lossy", encode: func() []byte { return EncodeWebP(img, 80, true) }},
	} {
		if got := tc.encode(); got != nil {
			t.Errorf("%s returned %d bytes for an image with out-of-range dimensions", tc.name, len(got))
		}
	}
}

// TestWebPLosslessAlphaIgnoresHeaderHint clears the VP8L header's alpha_is_used bit on a file that really does carry
// alpha. The bit is a hint the spec says "should not impact decoding", and x/image/webp does ignore it — it always
// decodes the ARGB plane — so trusting it stamped AlphaTypeOpaque on translucent pixels and the blend fast paths that
// branch on IsOpaque then rendered them with their transparency dropped.
func TestWebPLosslessAlphaIgnoresHeaderHint(t *testing.T) {
	src := testNRGBA() // alpha runs 255 down to 143 across x
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
	if enc[20] != 0x2F {
		t.Fatalf("fixture is not a bare VP8L stream (byte 20 = %#02x)", enc[20])
	}
	hdr := binary.LittleEndian.Uint32(enc[21:])
	if hdr>>28&1 == 0 {
		t.Fatal("encoder already left alpha_is_used clear; clearing it proves nothing")
	}
	binary.LittleEndian.PutUint32(enc[21:], hdr&^(1<<28))

	im := decodeVia(t, enc)
	if im.IsOpaque() {
		t.Fatal("IsOpaque() is true for a lossless WebP holding translucent pixels")
	}
	if im.AlphaType() != imagecore.AlphaTypePremul {
		t.Fatalf("alpha type %v, want premul", im.AlphaType())
	}
	// The pixels really do decode with alpha, so the reported info would have been a lie.
	p := im.PeekPixels(imagecore.CachingAllow)
	if a := p.Words[7] >> 24; a != 143 {
		t.Fatalf("decoded alpha of the last pixel of row 0 = %d, want 143", a)
	}
	if p.Info.AlphaType != im.AlphaType() {
		t.Fatalf("Decode reported %v but DecodeInfo reported %v", p.Info.AlphaType, im.AlphaType())
	}
}

// TestWebPLosslessOpaqueStaysOpaque is the other half of the pixel scan: an opaque lossless WebP must keep the opaque
// fast path rather than being labeled premul wholesale.
func TestWebPLosslessOpaqueStaysOpaque(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 8, 6))
	for i := range src.Pix {
		src.Pix[i] = 0xFF
	}
	im0 := imagecore.NewRasterData(
		mustInfo(t, 8, 6, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeUnpremul),
		src.Pix, src.Stride,
	)
	enc := EncodeWebP(im0, 100, false)
	if enc == nil {
		t.Fatal("EncodeWebP nil")
	}
	if !webpIsLossless(enc) {
		t.Fatal("fixture is not a lossless payload")
	}
	im := decodeVia(t, enc)
	if !im.IsOpaque() {
		t.Fatalf("opaque lossless WebP reported %v", im.AlphaType())
	}
}
