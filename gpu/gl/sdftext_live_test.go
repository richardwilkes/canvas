// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context test for distance-field text: a real GL surface renders large rotated text — which the SubRunControl
// routes through the SDFT lane — and the readback must show anti-aliased ink; redrawing under a different rotation
// (matrix-range reuse: same strike, no re-rasterization) must render too. Skips when no GL context is available.

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
)

func TestLiveSDFTextRotated(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 320, 320

	data, err := os.ReadFile("../../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}

	s := gl.MakeRenderTargetSurface(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: w, Height: h},
		1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "sdf-live-test", nil)
	if s == nil {
		t.Fatal("MakeRenderTargetSurface returned nil")
	}
	c := s.Canvas()

	// 200px text is inside the default SDFT window (162..glyphsAsPathsFontSize) on every OS.
	f := font.NewFont(tf, 200, 1, 0)
	paint := canvas.NewPaint()
	paint.Color = colorcore.RGB(0, 0, 0)
	paint.AntiAlias = true

	render := func(deg float32) []byte {
		c.Clear(colorcore.RGB(255, 255, 255))
		c.Save()
		var m geom.Matrix
		m.SetRotatePivot(deg, w/2, h/2)
		c.Concat(&m)
		c.DrawSimpleText([]byte("S"), font.TextEncodingUTF8, 90, 230, f, paint)
		c.Restore()
		buf := make([]byte, w*h*4)
		if !s.ReadPixels(rgba8888Premul(w, h), buf, w*4, 0, 0) {
			t.Fatal("ReadPixels failed")
		}
		return buf
	}

	stats := func(buf []byte) (ink, aa int) {
		for i := 0; i+3 < len(buf); i += 4 {
			v := buf[i]
			if v != 255 {
				ink++
				if v != 0 {
					aa++
				}
			}
		}
		return ink, aa
	}

	ink30, aa30 := stats(render(30))
	if ink30 < 1000 {
		t.Fatalf("rotated SDF text ink pixels = %d, implausibly low", ink30)
	}
	if aa30 < 100 {
		t.Fatalf("rotated SDF text AA pixels = %d — distance-field coverage must be smooth", aa30)
	}

	// A second rotation reuses the cached SDFT subrun (matrix range covers any rotation at the same scale) and must ink
	// comparably.
	ink60, _ := stats(render(-45))
	if ink60 < 1000 {
		t.Fatalf("re-rotated SDF text ink pixels = %d, implausibly low", ink60)
	}
	ratio := float64(ink60) / float64(ink30)
	if ratio < 0.5 || ratio > 2 {
		t.Errorf("rotation reuse ink ratio = %.2f (30°: %d, -45°: %d), implausible", ratio,
			ink30, ink60)
	}
}
