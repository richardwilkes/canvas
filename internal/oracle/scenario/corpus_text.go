// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package scenario

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// More text scenes, including color emoji. Text renders through the port's own rasterizer, deterministic per platform,
// so these scenarios gate in every lane against the per-platform golden sets like any other scenario.

func regText(name string, draw func(c Canvas)) {
	Register(Scenario{Name: name, Width: 256, Height: 256, Draw: draw})
}

func init() {
	regText("text-waterfall", func(c Canvas) {
		// Upright sizes across the direct-mask lane; the largest line sits inside the SDFT window.
		c.Clear(white)
		y := float32(0)
		for _, size := range []float32{11, 14, 18, 24, 32, 44, 60} {
			y += size + 6
			c.DrawSimpleText("Waft", 8, y, size, Fill(black))
		}
	})

	regText("text-styled", func(c Canvas) {
		c.Clear(white)
		// Stroked text.
		c.DrawSimpleText("Sk", 16, 96, 96, Stroke(blue, 2.5))
		// Gradient-shaded fill.
		p := Fill(black)
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 120}, {X: 0, Y: 230}},
			Colors: []colorcore.Color{red, yellow, blue},
			Tile:   TileClamp,
		}
		c.DrawSimpleText("Go", 16, 226, 110, p)
		// Translucent overlapping text.
		c.DrawSimpleText("aa", 140, 90, 72, Fill(colorcore.Color(0x902E9E44)))
		c.DrawSimpleText("aa", 156, 104, 72, Fill(colorcore.Color(0x90E0392E)))
	})

	regText("text-skewed", func(c Canvas) {
		// A skewed CTM keeps text out of the axis-aligned direct-mask fast path without leaving the mask lane entirely.
		c.Clear(white)
		c.Save()
		c.Skew(-0.25, 0)
		c.DrawSimpleText("skewed", 40, 80, 44, Fill(black))
		c.Restore()
		c.Save()
		c.Skew(0, 0.18)
		c.DrawSimpleText("vertical", 24, 170, 40, Fill(colorcore.Color(0xFF102040)))
		c.Restore()
	})

	regText("text-clipped-layer", func(c Canvas) {
		// Text through an AA clip and inside a translucent layer: glyph masks composited through the layer/clip
		// machinery rather than direct-to-surface.
		c.Clear(white)
		c.ClipPath(NewPathSpec().AddCircle(128, 128, 92, geom.DirectionCW), ClipIntersect, true)
		c.SaveLayerAlpha(nil, 150)
		c.DrawSimpleText("clip", 46, 120, 84, Fill(blue))
		c.DrawSimpleText("layer", 52, 196, 64, Fill(red))
		c.Restore()
	})

	regText("text-emoji-sbix", func(c Canvas) {
		// The sbix PNG-strike font at sizes bracketing its 16/64/128 ppem strikes (bitmap selection + resampling on
		// both sides).
		c.Clear(white)
		c.DrawSimpleTextFont("\U0001F600", 12, 76, 64, FontSbix, Fill(black))
		c.DrawSimpleTextFont("\U0001F600", 96, 130, 100, FontSbix, Fill(black))
		c.DrawSimpleTextFont("\U0001F600", 12, 232, 40, FontSbix, Fill(black))
	})

	regText("text-emoji-colr", func(c Canvas) {
		// The COLRv0/CPAL layered color glyph — vector color glyphs scale, so exercise a rotation too.
		c.Clear(white)
		c.DrawSimpleTextFont("\U0001F600", 16, 120, 110, FontCOLR, Fill(black))
		c.Save()
		c.Translate(150, 230)
		c.RotateRadians(-0.5)
		c.DrawSimpleTextFont("\U0001F600", 0, 0, 72, FontCOLR, Fill(black))
		c.Restore()
	})
}
