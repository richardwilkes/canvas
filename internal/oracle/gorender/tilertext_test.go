// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gorender_test

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// TestTilerTextSeamWindow covers the glyph lane of the draw tiler (canvas/drawtiler.go): a text run crossing the 8191px
// tile seam on a 10000x24 strip must render exactly as the same scene on a small untiled surface translated into view.
// Glyph rasterization is the port's own on both sides (same scaler, same masks; text output is deterministic per
// platform, which is also what lets the corpus's text scenes gate bit-exactly against the per-platform raster
// goldens), and glyph placement/subpixel quantization is invariant under integer translation, so
// the tiled strip and the untiled window must agree byte for byte; a seam misplacement or a wrong per-tile CTM shift in
// the glyph painter breaks whole glyphs. cgo-free: both renders go through gorender's raster backend.
func TestTilerTextSeamWindow(t *testing.T) {
	const stripW, stripH = 10000, 24
	scene := func(c scenario.Canvas) {
		c.Clear(colorcore.Color(0xFFFFFFFF))
		p := scenario.Fill(0xFF202020)
		// The run starts left of the seam and crosses it mid-word; a second, larger run puts a glyph directly astride
		// x=8191.
		c.DrawSimpleText("tiling across the seam at 8191 pixels", 8000, 10, 13, p)
		c.DrawSimpleText("Wm", 8182, 22, 15, p)
	}
	strip := gorender.RenderScenarioRaster(scenario.Scenario{
		Name: "tiler-text-strip", Width: stripW, Height: stripH, Draw: scene,
	})

	const winW = 512
	for _, x0 := range []int{8191 - 256, 7936, 8300} {
		win := gorender.RenderScenarioRaster(scenario.Scenario{
			Name: "tiler-text-window", Width: winW, Height: stripH,
			Draw: func(c scenario.Canvas) {
				c.Translate(float32(-x0), 0)
				scene(c)
			},
		})
		for y := 0; y < stripH; y++ {
			for x := 0; x < winW; x++ {
				si := (y*stripW + x0 + x) * 4
				wi := (y*winW + x) * 4
				for ch := 0; ch < 4; ch++ {
					if strip[si+ch] != win[wi+ch] {
						t.Fatalf("window x0=%d: pixel (%d,%d) channel %d: tiled %d != untiled %d",
							x0, x, y, ch, strip[si+ch], win[wi+ch])
					}
				}
			}
		}
		t.Logf("window x0=%d byte-identical", x0)
	}
}
