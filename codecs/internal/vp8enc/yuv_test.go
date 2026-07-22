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
	"image"
	"image/color"
	"testing"
)

// referenceFillRGB is the generic, always-correct path from fillRGB: it routes every pixel through
// color.NRGBAModel, un-premultiplying as needed. The image-type fast paths must agree with it exactly.
func referenceFillRGB(img image.Image, rgb []uint8, w, h int) {
	b := img.Bounds()
	for y := 0; y < h; y++ {
		d := rgb[3*w*y:]
		for x := 0; x < w; x++ {
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			d[3*x] = c.R
			d[3*x+1] = c.G
			d[3*x+2] = c.B
		}
	}
}

// TestFillRGBFastPathsMatchGeneric verifies that the *image.RGBA fast path un-premultiplies to agree with the
// generic color.NRGBAModel path for every alpha value, and that the *image.NRGBA and *image.Gray fast paths agree
// as well.
func TestFillRGBFastPathsMatchGeneric(t *testing.T) {
	const w, h = 4, 3

	// Build an RGBA image whose pixels span a range of premultiplied colors and alphas, including partial alpha
	// where a straight byte copy would produce the wrong (darkened) result.
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	// Use color.RGBA (premultiplied) values with r,g,b <= a as required by the premultiplied invariant.
	samples := []color.RGBA{
		{R: 255, G: 128, B: 64, A: 255}, // opaque
		{R: 0, G: 0, B: 0, A: 0},        // fully transparent
		{R: 64, G: 32, B: 16, A: 128},   // half alpha
		{R: 10, G: 5, B: 1, A: 20},      // low alpha
		{R: 200, G: 100, B: 50, A: 200},
		{R: 1, G: 1, B: 1, A: 1}, // extreme low alpha
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rgba.SetRGBA(x, y, samples[(y*w+x)%len(samples)])
		}
	}

	got := make([]uint8, 3*w*h)
	want := make([]uint8, 3*w*h)
	fillRGB(rgba, got, w, h)
	referenceFillRGB(rgba, want, w, h)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("RGBA fast path disagrees with generic path at byte %d: got %d, want %d", i, got[i], want[i])
		}
	}

	// NRGBA fast path.
	nrgba := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			nrgba.SetNRGBA(x, y, color.NRGBA{R: uint8(x * 60), G: uint8(y * 80), B: uint8(x*20 + y*10), A: uint8(x * 50)})
		}
	}
	fillRGB(nrgba, got, w, h)
	referenceFillRGB(nrgba, want, w, h)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("NRGBA fast path disagrees with generic path at byte %d: got %d, want %d", i, got[i], want[i])
		}
	}

	// Gray fast path.
	gray := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gray.SetGray(x, y, color.Gray{Y: uint8((x*7 + y*13) * 5)})
		}
	}
	fillRGB(gray, got, w, h)
	referenceFillRGB(gray, want, w, h)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Gray fast path disagrees with generic path at byte %d: got %d, want %d", i, got[i], want[i])
		}
	}
}
