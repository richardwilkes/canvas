// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context test for LCD subpixel text on the GPU: a real GL surface with RGB_H pixel geometry renders
// subpixel-edged text through the A565 atlas (RGB565 uploads + the LCD-coverage blend), and the readback must show
// channel fringing; the same draw with kAntiAlias edging (and any draw on a default-props surface) stays gray. Skips
// when no GL context is available.

package gl_test

import (
	"os"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/surface"
)

func TestLiveLCDTextFringing(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 160, 48

	data, err := os.ReadFile("../../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}

	render := func(geometry surface.PixelGeometry, edging font.Edging) []byte {
		props := surface.Props{PixelGeometry: geometry}
		s := gl.MakeRenderTargetSurface(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: w, Height: h},
			1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "lcd-live-test", &props)
		if s == nil {
			t.Fatal("MakeRenderTargetSurface returned nil")
		}
		c := s.Canvas()
		c.Clear(colorcore.RGB(255, 255, 255))
		f := font.NewFont(tf, 20, 1, 0)
		f.SetEdging(edging)
		paint := canvas.NewPaint()
		paint.Color = colorcore.RGB(0, 0, 0)
		c.DrawSimpleText([]byte("Hamburg"), font.TextEncodingUTF8, 8, 32, f, paint)
		buf := make([]byte, w*h*4)
		if !s.ReadPixels(rgba8888Premul(w, h), buf, w*4, 0, 0) {
			t.Fatal("ReadPixels failed")
		}
		return buf
	}

	fringe := func(buf []byte) int {
		n := 0
		for i := 0; i+3 < len(buf); i += 4 {
			if buf[i] != buf[i+1] || buf[i+1] != buf[i+2] {
				n++
			}
		}
		return n
	}
	ink := func(buf []byte) int {
		n := 0
		for i := 0; i+3 < len(buf); i += 4 {
			if buf[i] != 255 || buf[i+1] != 255 || buf[i+2] != 255 {
				n++
			}
		}
		return n
	}

	lcd := render(surface.PixelGeometryRGBH, font.EdgingSubpixelAntiAlias)
	if got := ink(lcd); got < 100 {
		t.Fatalf("LCD text ink pixels = %d, implausibly low", got)
	}
	if got := fringe(lcd); got == 0 {
		t.Fatal("GPU LCD text has no channel fringing")
	}

	gray := render(surface.PixelGeometryRGBH, font.EdgingAntiAlias)
	if got := fringe(gray); got != 0 {
		t.Fatalf("kAntiAlias text fringed %d pixels on the GPU", got)
	}
	unknown := render(surface.PixelGeometryUnknown, font.EdgingSubpixelAntiAlias)
	if got := fringe(unknown); got != 0 {
		t.Fatalf("unknown-geometry subpixel text fringed %d pixels on the GPU", got)
	}
}
