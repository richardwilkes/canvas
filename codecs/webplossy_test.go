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
	"math"
	"testing"

	"golang.org/x/image/webp"

	"github.com/richardwilkes/canvas/imagecore"
)

// lossyTestImage builds a deterministic photo-like w×h source; alphaFn (nil = opaque) supplies the alpha value per
// pixel.
func lossyTestImage(w, h int, alphaFn func(x, y int) uint8) *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			fx, fy := float64(x), float64(y)
			a := uint8(255)
			if alphaFn != nil {
				a = alphaFn(x, y)
			}
			m.SetNRGBA(x, y, color.NRGBA{
				R: uint8(128 + 90*math.Sin(fx/19)*math.Cos(fy/13)),
				G: uint8(128 + 80*math.Sin((fx+fy)/29)),
				B: uint8(128 + 70*math.Cos(fx/9)*math.Sin(fy/23)),
				A: a,
			})
		}
	}
	return m
}

func lossyImagecore(t *testing.T, src *image.NRGBA) *imagecore.Image {
	t.Helper()
	b := src.Bounds()
	im := imagecore.NewRasterData(
		mustInfo(t, int32(b.Dx()), int32(b.Dy()), imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeUnpremul),
		src.Pix, src.Stride,
	)
	if im == nil {
		t.Fatal("raster image nil")
	}
	return im
}

// rgbPSNR computes the PSNR over the RGB channels of two same-sized images, considering only pixels where keep (nil =
// all) is true.
func rgbPSNR(t *testing.T, want, got image.Image, keep func(x, y int) bool) float64 {
	t.Helper()
	b := want.Bounds()
	if got.Bounds().Dx() != b.Dx() || got.Bounds().Dy() != b.Dy() {
		t.Fatalf("dims %v vs %v", want.Bounds(), got.Bounds())
	}
	var sum, n float64
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			if keep != nil && !keep(x, y) {
				continue
			}
			wc := color.NRGBAModel.Convert(want.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			gc := color.NRGBAModel.Convert(got.At(got.Bounds().Min.X+x, got.Bounds().Min.Y+y)).(color.NRGBA)
			for _, d := range [3]int{int(wc.R) - int(gc.R), int(wc.G) - int(gc.G), int(wc.B) - int(gc.B)} {
				sum += float64(d * d)
			}
			n += 3
		}
	}
	if sum == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(255*255*n/sum)
}

// alphChunkHeader returns the first payload byte of the ALPH chunk (the method/filter/pre-processing header), failing
// if the chunk is absent.
func alphChunkHeader(t *testing.T, data []byte) byte {
	t.Helper()
	payload := riffChunk(data, "ALPH")
	if len(payload) < 1 {
		t.Fatal("no ALPH chunk")
	}
	return payload[0]
}

// TestWebPLossyOpaque checks that an opaque image encodes into the simple RIFF/VP8 container (no VP8X/ALPH) and
// round-trips through x/image/webp at a reasonable fidelity.
func TestWebPLossyOpaque(t *testing.T) {
	src := lossyTestImage(96, 80, nil)
	data := EncodeWebP(lossyImagecore(t, src), 75, true)
	if data == nil {
		t.Fatal("EncodeWebP(lossy) returned nil")
	}
	if !bytes.Equal(data[12:16], []byte("VP8 ")) {
		t.Fatalf("opaque image: first chunk %q, want the simple VP8 container", data[12:16])
	}
	if webpHasAlpha(data) {
		t.Error("opaque image: alpha sniffed")
	}
	back, err := webp.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := back.(*image.YCbCr); !ok {
		t.Errorf("decoded as %T, want *image.YCbCr", back)
	}
	// RGB PSNR (not luma-only), so 4:2:0 chroma subsampling bounds it well below the vp8enc Y-PSNR floors.
	if psnr := rgbPSNR(t, src, back, nil); psnr < 30 {
		t.Errorf("opaque q75 PSNR = %.1f dB, want >= 30", psnr)
	}
}

// TestWebPLossyAlpha checks the extended container: VP8X with the alpha flag, an ALPH chunk whose decoded plane is
// byte-exact (the alpha lane is lossless), and color that still round-trips where it is visible. The smooth ramp
// compresses (ALPH method 1, the VP8L lane); the noisy plane must trip the method-0 uncompressed fallback.
func TestWebPLossyAlpha(t *testing.T) {
	cases := []struct {
		alphaFn    func(x, y int) uint8
		name       string
		wantMethod byte
	}{
		{name: "smooth", alphaFn: func(x, _ int) uint8 { return uint8(x * 255 / 95) }, wantMethod: 1},
		// splitmix64-style avalanche per pixel: incompressible, so VP8L cannot beat the raw plane.
		{name: "noisy", alphaFn: func(x, y int) uint8 {
			z := uint64(y*96+x)*0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9
			z ^= z >> 30
			z *= 0xBF58476D1CE4E5B9
			z ^= z >> 27
			return uint8(z * 0x94D049BB133111EB >> 56)
		}, wantMethod: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 96, 80
			src := lossyTestImage(w, h, tc.alphaFn)
			data := EncodeWebP(lossyImagecore(t, src), 75, true)
			if data == nil {
				t.Fatal("EncodeWebP(lossy) returned nil")
			}
			if !bytes.Equal(data[12:16], []byte("VP8X")) {
				t.Fatalf("first chunk %q, want VP8X", data[12:16])
			}
			if !webpHasAlpha(data) {
				t.Error("alpha bit not sniffed from VP8X")
			}
			if wm1 := binary.LittleEndian.Uint32(data[24:]) & 0xFFFFFF; wm1 != w-1 {
				t.Errorf("VP8X canvas width-1 = %d, want %d", wm1, w-1)
			}
			if hdr := alphChunkHeader(t, data); hdr != tc.wantMethod {
				t.Errorf("ALPH header byte = %#x, want method %d (filter none)", hdr, tc.wantMethod)
			}
			back, err := webp.Decode(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			nycc, ok := back.(*image.NYCbCrA)
			if !ok {
				t.Fatalf("decoded as %T, want *image.NYCbCrA", back)
			}
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					if got, want := nycc.A[y*nycc.AStride+x], tc.alphaFn(x, y); got != want {
						t.Fatalf("alpha[%d,%d] = %d, want %d (plane must be lossless)", x, y, got, want)
					}
				}
			}
			// Color under high alpha must still be a faithful lossy round trip.
			opaqueish := func(x, y int) bool { return tc.alphaFn(x, y) >= 200 }
			if psnr := rgbPSNR(t, src, back, opaqueish); psnr < 30 {
				t.Errorf("q75 PSNR over alpha>=200 = %.1f dB, want >= 30", psnr)
			}
		})
	}
}

// TestWebPLossyQuality checks that the quality parameter is honored end-to-end: higher quality costs more bytes and
// recovers more fidelity, and out-of-range quality fails.
func TestWebPLossyQuality(t *testing.T) {
	src := lossyTestImage(96, 80, nil)
	im := lossyImagecore(t, src)
	var sizes [3]int
	var psnrs [3]float64
	for i, q := range [3]float32{10, 50, 90} {
		data := EncodeWebP(im, q, true)
		if data == nil {
			t.Fatalf("q=%v: nil", q)
		}
		back, err := webp.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("q=%v: %v", q, err)
		}
		sizes[i] = len(data)
		psnrs[i] = rgbPSNR(t, src, back, nil)
	}
	if sizes[0] >= sizes[1] || sizes[1] >= sizes[2] {
		t.Errorf("sizes not increasing with quality: %v", sizes)
	}
	if !(psnrs[0] < psnrs[1] && psnrs[1] < psnrs[2]) {
		t.Errorf("PSNR not increasing with quality: %v", psnrs)
	}
	for _, q := range []float32{-1, 100.5} {
		if EncodeWebP(im, q, true) != nil || EncodeWebP(im, q, false) != nil {
			t.Errorf("q=%v: want nil for out-of-range quality (WebPConfigPreset parity)", q)
		}
	}
}
