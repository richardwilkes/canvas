// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Encoder-level checks: nil-image tolerance and the WebP lossy alpha container.

package codecs

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/imagecore"
)

// TestEncodeNilImage keeps the C-parity nil tolerance the encoders themselves guarantee: encoding a nil image returns
// nil rather than panicking.
func TestEncodeNilImage(t *testing.T) {
	if EncodePNG(nil, 6) != nil {
		t.Error("EncodePNG(nil image): want nil")
	}
	if EncodeJPEG(nil, 85) != nil {
		t.Error("EncodeJPEG(nil image): want nil")
	}
	if EncodeWebP(nil, 0.9, false) != nil {
		t.Error("EncodeWebP(nil image): want nil")
	}
}

// translucentRampImage builds a w×h premul RGBA_8888 image whose alpha ramps with x (hitting 0, mid values and 255) and
// whose color stays premul-valid (channels <= alpha).
func translucentRampImage(w, h int32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			a := uint32(x * 255 / max(w-1, 1))
			r := a / 2
			g := uint32(min(int32(a), y*8))
			p.Words[y*p.RowElems+x] = r | g<<8 | a<<16 | a<<24
		}
	}
	return imagecore.FromPixels(p)
}

// TestEncodeWebPLossyAlphaContainer checks the lossy alpha lane end-to-end: a translucent image gets the extended (VP8X
// + ALPH) container — including the odd-length chunk padding rule — an opaque image stays in the simple VP8 container,
// and the losslessly-coded alpha plane survives the round trip exactly.
func TestEncodeWebPLossyAlphaContainer(t *testing.T) {
	img := gradOpaqueImage(64, 48)
	enc := EncodeWebP(img, 75, true)
	if enc == nil {
		t.Fatal("EncodeWebP(lossy, opaque) returned nil")
	}
	if !bytes.Equal(enc[12:16], []byte("VP8 ")) {
		t.Errorf("opaque lossy first chunk %q, want the simple VP8 container", enc[12:16])
	}

	timg := translucentRampImage(64, 48)
	enc = EncodeWebP(timg, 75, true)
	if enc == nil {
		t.Fatal("EncodeWebP(lossy, alpha) returned nil")
	}
	if !bytes.Equal(enc[12:16], []byte("VP8X")) {
		t.Errorf("alpha lossy first chunk %q, want VP8X", enc[12:16])
	}
	back := imagecore.NewFromEncoded(enc)
	if back == nil {
		t.Fatal("decoding the lossy+alpha WebP failed")
	}
	if back.Width() != 64 || back.Height() != 48 {
		t.Fatalf("decoded dims %dx%d, want 64x48", back.Width(), back.Height())
	}
	// The alpha plane is coded losslessly, so the decoded premul alpha bytes match the source exactly.
	wantPix := timg.PeekPixels(imagecore.CachingAllow)
	gotPix := back.PeekPixels(imagecore.CachingAllow)
	for y := int32(0); y < 48; y++ {
		for x := int32(0); x < 64; x++ {
			w := wantPix.Words[y*wantPix.RowElems+x] >> 24
			g := gotPix.Words[y*gotPix.RowElems+x] >> 24
			if w != g {
				t.Fatalf("alpha at (%d,%d): got %d, want %d (ALPH plane must round-trip losslessly)", x, y, g, w)
			}
		}
	}
}

// gradOpaqueImage builds a w×h opaque RGBA_8888 image whose red varies with x and green with y.
func gradOpaqueImage(w, h int32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			r := uint32(20+x*3) & 0xFF
			g := uint32(10+y*3) & 0xFF
			p.Words[y*p.RowElems+x] = r | g<<8 | 0x80<<16 | 0xFF<<24
		}
	}
	return imagecore.FromPixels(p)
}
