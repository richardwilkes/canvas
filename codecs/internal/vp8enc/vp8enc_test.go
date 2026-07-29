// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package vp8enc

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"golang.org/x/image/webp"
)

//------------------------------------------------------------------------------
// Test image generators

func gradientImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x * 255 / maxTest(w-1, 1)),
				G: uint8(y * 255 / maxTest(h-1, 1)),
				B: uint8((x + y) * 255 / maxTest(w+h-2, 1)),
				A: 255,
			})
		}
	}
	return img
}

func solidImage(w, h int, c color.NRGBA) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

// photoLikeImage layers smooth low-frequency waves with mild noise, approximating natural content.
func photoLikeImage(w, h int, seed int64) *image.NRGBA {
	rng := rand.New(rand.NewSource(seed))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			fx, fy := float64(x), float64(y)
			r := 128 + 90*math.Sin(fx/23)*math.Cos(fy/17)
			g := 128 + 80*math.Sin((fx+fy)/31)
			b := 128 + 70*math.Cos(fx/11)*math.Sin(fy/29)
			n := float64(rng.Intn(21) - 10)
			img.SetNRGBA(x, y, color.NRGBA{
				R: clamp255(int(r + n)),
				G: clamp255(int(g + n)),
				B: clamp255(int(b + n)),
				A: 255,
			})
		}
	}
	return img
}

func sharpEdgesImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.NRGBA{R: 20, G: 40, B: 200, A: 255}
			if (x/13+y/9)%2 == 0 {
				c = color.NRGBA{R: 240, G: 220, B: 30, A: 255}
			}
			if x == y {
				c = color.NRGBA{R: 255, G: 0, B: 0, A: 255}
			}
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

func noiseImage(w, h int, seed int64) *image.NRGBA {
	rng := rand.New(rand.NewSource(seed))
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(rng.Intn(256)),
				G: uint8(rng.Intn(256)),
				B: uint8(rng.Intn(256)),
				A: 255,
			})
		}
	}
	return img
}

func maxTest(a, b int) int {
	if a > b {
		return a
	}
	return b
}

//------------------------------------------------------------------------------
// Helpers

// yPSNR computes the PSNR between the encoder's own Y-plane input and the decoded luma. The luma plane is where the
// quality knob shows; chroma is subsampled and not comparable 1:1.
func yPSNR(t *testing.T, e *encoder, decoded *image.YCbCr) float64 {
	t.Helper()
	var sse, n float64
	for y := 0; y < e.height; y++ {
		for x := 0; x < e.width; x++ {
			d := float64(e.yPlane[y*e.yStride+x]) - float64(decoded.Y[decoded.YOffset(x, y)])
			sse += d * d
			n++
		}
	}
	if sse == 0 {
		return 99
	}
	return 10 * math.Log10(255*255*n/sse)
}

func decodeWebP(t *testing.T, data []byte) *image.YCbCr {
	t.Helper()
	img, err := webp.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("x/image/webp failed to decode the bitstream: %v", err)
	}
	ycc, ok := img.(*image.YCbCr)
	if !ok {
		t.Fatalf("decoded image is %T, want *image.YCbCr", img)
	}
	return ycc
}

//------------------------------------------------------------------------------
// End-to-end tests

// TestRoundTripDecodable encodes a spread of content and sizes (including odd dimensions that exercise edge
// macroblocks) and checks that x/image/webp decodes every bitstream with the right dimensions, and that the
// reconstruction invariant holds: the encoder's internal reconstruction must equal the decoder output exactly.
func TestRoundTripDecodable(t *testing.T) {
	sizes := [][2]int{{16, 16}, {33, 17}, {1, 1}, {15, 100}, {128, 128}, {161, 143}, {31, 32}}
	for _, size := range sizes {
		w, h := size[0], size[1]
		images := map[string]image.Image{
			"gradient": gradientImage(w, h),
			"solid":    solidImage(w, h, color.NRGBA{R: 120, G: 130, B: 140, A: 255}),
			"photo":    photoLikeImage(w, h, 5),
			"edges":    sharpEdgesImage(w, h),
			"noise":    noiseImage(w, h, 11),
		}
		for name, img := range images {
			for _, quality := range []float32{4, 25, 75} {
				e, err := encode(img, quality)
				if err != nil {
					t.Fatalf("%s %dx%d q%v: encode: %v", name, w, h, quality, err)
				}
				data, err := e.assemble()
				if err != nil {
					t.Fatalf("%s %dx%d q%v: assemble: %v", name, w, h, quality, err)
				}
				cfg, err := webp.DecodeConfig(bytes.NewReader(data))
				if err != nil {
					t.Fatalf("%s %dx%d q%v: DecodeConfig: %v", name, w, h, quality, err)
				}
				if cfg.Width != w || cfg.Height != h {
					t.Fatalf("%s %dx%d q%v: decoded dimensions %dx%d", name, w, h, quality, cfg.Width, cfg.Height)
				}
				decoded := decodeWebP(t, data)
				checkReconMatchesDecode(t, e, decoded, name, quality)
			}
		}
	}
}

// checkReconMatchesDecode asserts the encoder's reconstructed planes equal the decoder's output byte-for-byte over the
// visible area (the precise catch-all for reconstruction-loop bugs).
func checkReconMatchesDecode(t *testing.T, e *encoder, decoded *image.YCbCr, name string, quality float32) {
	t.Helper()
	for y := 0; y < e.height; y++ {
		for x := 0; x < e.width; x++ {
			got := decoded.Y[decoded.YOffset(x, y)]
			want := e.reconY[y*e.reconYStride+x]
			if got != want {
				t.Fatalf("%s %dx%d q%v: luma recon mismatch at (%d,%d): decoder %d encoder %d",
					name, e.width, e.height, quality, x, y, got, want)
			}
		}
	}
	uvW := (e.width + 1) >> 1
	uvH := (e.height + 1) >> 1
	for y := 0; y < uvH; y++ {
		for x := 0; x < uvW; x++ {
			off := decoded.COffset(2*x, 2*y)
			if got, want := decoded.Cb[off], e.reconU[y*e.reconUVStride+x]; got != want {
				t.Fatalf("%s %dx%d q%v: Cb recon mismatch at (%d,%d): decoder %d encoder %d",
					name, e.width, e.height, quality, x, y, got, want)
			}
			if got, want := decoded.Cr[off], e.reconV[y*e.reconUVStride+x]; got != want {
				t.Fatalf("%s %dx%d q%v: Cr recon mismatch at (%d,%d): decoder %d encoder %d",
					name, e.width, e.height, quality, x, y, got, want)
			}
		}
	}
}

// TestPSNRFloors asserts conservative quality floors at quality 75 on smooth/synthetic content.
func TestPSNRFloors(t *testing.T) {
	cases := []struct {
		img   image.Image
		name  string
		floor float64
	}{
		{name: "gradient-128", img: gradientImage(128, 128), floor: 36},
		{name: "solid-64", img: solidImage(64, 64, color.NRGBA{R: 200, G: 100, B: 50, A: 255}), floor: 45},
		{name: "photo-160x120", img: photoLikeImage(160, 120, 3), floor: 33},
		{name: "edges-96", img: sharpEdgesImage(96, 96), floor: 30},
	}
	for _, c := range cases {
		e, err := encode(c.img, 75)
		if err != nil {
			t.Fatalf("%s: encode: %v", c.name, err)
		}
		data, err := e.assemble()
		if err != nil {
			t.Fatalf("%s: assemble: %v", c.name, err)
		}
		decoded := decodeWebP(t, data)
		psnr := yPSNR(t, e, decoded)
		t.Logf("%s: q75 Y-PSNR %.2f dB, %d bytes", c.name, psnr, len(data))
		if psnr < c.floor {
			t.Errorf("%s: Y-PSNR %.2f dB below the %v dB floor", c.name, psnr, c.floor)
		}
	}
}

// TestQualityMonotonicity asserts that higher quality yields higher (or equal) PSNR and that file sizes grow with
// quality on non-trivial content.
func TestQualityMonotonicity(t *testing.T) {
	images := map[string]image.Image{
		"photo":    photoLikeImage(160, 128, 9),
		"gradient": gradientImage(144, 96),
		"edges":    sharpEdgesImage(112, 80),
	}
	qualities := []float32{10, 50, 90}
	for name, img := range images {
		var lastPSNR float64 = -1
		var lastSize int
		for _, q := range qualities {
			e, err := encode(img, q)
			if err != nil {
				t.Fatalf("%s q%v: encode: %v", name, q, err)
			}
			data, err := e.assemble()
			if err != nil {
				t.Fatalf("%s q%v: assemble: %v", name, q, err)
			}
			decoded := decodeWebP(t, data)
			psnr := yPSNR(t, e, decoded)
			t.Logf("%s q%v: Y-PSNR %.2f dB, %d bytes", name, q, psnr, len(data))
			if psnr < lastPSNR {
				t.Errorf("%s: PSNR decreased from %.2f to %.2f between qualities", name, lastPSNR, psnr)
			}
			if lastSize > 0 && len(data) < lastSize {
				t.Errorf("%s: size decreased from %d to %d between qualities", name, lastSize, len(data))
			}
			lastPSNR = psnr
			lastSize = len(data)
		}
	}
}

// TestRoundTripRealImages runs the round-trip and reconstruction invariants over checked-in rendered scenes (the
// oracle's raster goldens for this platform: antialiased paths, gradients, complex path content), when present. Only
// the realism of the content matters here — every assertion compares the encoder against its own reconstruction, never
// against the golden — so it costs nothing that those sets are per-platform and get recaptured when rendering changes.
// A platform with no raster set skips.
func TestRoundTripRealImages(t *testing.T) {
	dir := filepath.Join("..", "..", "..", "internal", "oracle", "goldens", "raster",
		runtime.GOOS+"_"+runtime.GOARCH)
	names := []string{"gradient-radial.png", "fill-rects-aa.png", "arcs.png", "path-star-selfintersect.png"}
	for _, name := range names {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Skipf("oracle goldens not available: %v", err)
		}
		img, err := png.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("%s: decode png: %v", name, err)
		}
		for _, q := range []float32{35, 75} {
			e, encErr := encode(img, q)
			if encErr != nil {
				t.Fatalf("%s q%v: encode: %v", name, q, encErr)
			}
			data, encErr := e.assemble()
			if encErr != nil {
				t.Fatalf("%s q%v: assemble: %v", name, q, encErr)
			}
			decoded := decodeWebP(t, data)
			checkReconMatchesDecode(t, e, decoded, name, q)
			if q == 75 {
				psnr := yPSNR(t, e, decoded)
				t.Logf("%s: q75 Y-PSNR %.2f dB, %d bytes", name, psnr, len(data))
				if psnr < 30 {
					t.Errorf("%s: q75 Y-PSNR %.2f dB below 30 dB", name, psnr)
				}
			}
		}
	}
}

// TestEncodePublicAPI exercises the exported entry point, including quality clamping, the dimension guards and
// non-NRGBA inputs.
func TestEncodePublicAPI(t *testing.T) {
	data, err := Encode(gradientImage(40, 24), 75)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !bytes.HasPrefix(data, []byte("RIFF")) || !bytes.Equal(data[8:16], []byte("WEBPVP8 ")) {
		t.Fatalf("container header malformed: %q", data[:16])
	}
	if len(data)%2 != 0 {
		t.Fatalf("RIFF payload must be even-sized, got %d", len(data))
	}
	decodeWebP(t, data)

	// Quality outside [0, 100] clamps rather than failing.
	if _, err = Encode(gradientImage(16, 16), -5); err != nil {
		t.Fatalf("Encode with quality -5: %v", err)
	}
	if _, err = Encode(gradientImage(16, 16), 200); err != nil {
		t.Fatalf("Encode with quality 200: %v", err)
	}

	// Dimension guards.
	if _, err = Encode(image.NewNRGBA(image.Rect(0, 0, 0, 0)), 75); err == nil {
		t.Fatal("Encode accepted an empty image")
	}
	if _, err = Encode(image.NewNRGBA(image.Rect(0, 0, 16384, 4)), 75); err == nil {
		t.Fatal("Encode accepted an over-limit width")
	}

	// A Gray image and a generic (non-fast-path) image both take the conversion lanes.
	gray := image.NewGray(image.Rect(0, 0, 33, 17))
	for i := range gray.Pix {
		gray.Pix[i] = uint8(i * 7)
	}
	if data, err = Encode(gray, 60); err != nil {
		t.Fatalf("Encode gray: %v", err)
	}
	decodeWebP(t, data)
	rgba64 := image.NewRGBA64(image.Rect(2, 3, 35, 20)) // non-zero origin as well
	for y := 3; y < 20; y++ {
		for x := 2; x < 35; x++ {
			rgba64.SetRGBA64(x, y, color.RGBA64{R: uint16(x * 1000), G: uint16(y * 2000), B: 30000, A: 65535})
		}
	}
	if data, err = Encode(rgba64, 60); err != nil {
		t.Fatalf("Encode rgba64: %v", err)
	}
	cfg, err := webp.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("DecodeConfig rgba64: %v", err)
	}
	if cfg.Width != 33 || cfg.Height != 17 {
		t.Fatalf("rgba64 decoded dimensions %dx%d, want 33x17", cfg.Width, cfg.Height)
	}
	decodeWebP(t, data)
}
