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

// Blend-mode coverage. blend-modes-basic samples 8 modes; these grids cover all 29 sk_blend_mode_t
// values (12 Porter-Duff, 13 separable, 4 HSL), grouped by family, each cell compositing a translucent source circle
// over a translucent destination rect inside its own clip cell (the blend-modes-basic cell recipe, kept so the grids
// read uniformly). A separate scenario blends over a gradient destination to catch coefficient errors an axis-aligned
// flat dst would mask, and one drives DrawColor's blend argument.

// The three grids' mode lists, split by family. Together they must name every BlendMode exactly once; see
// TestBlendGridsCoverEveryMode.
var (
	porterDuffBlendModes = []BlendMode{
		BlendClear, BlendSrc, BlendDst, BlendSrcOver,
		BlendDstOver, BlendSrcIn, BlendDstIn, BlendSrcOut,
		BlendDstOut, BlendSrcATop, BlendDstATop, BlendXor,
	}
	separableBlendModes = []BlendMode{
		BlendPlus, BlendModulate, BlendScreen, BlendOverlay,
		BlendDarken, BlendLighten, BlendColorDodge, BlendColorBurn,
		BlendHardLight, BlendSoftLight, BlendDifference, BlendExclusion,
		BlendMultiply,
	}
	hslBlendModes = []BlendMode{BlendHue, BlendSaturation, BlendColor, BlendLuminosity}
)

// blendGrid draws one mode per cell in a grid of 64x64 cells, 4 per row.
func blendGrid(c Canvas, modes []BlendMode) {
	c.Clear(white)
	for i, mode := range modes {
		x := float32(i%4) * 64
		y := float32(i/4) * 64
		c.Save()
		c.ClipRect(geom.RectLTRB(x, y, x+64, y+64), ClipIntersect, false)
		dst := Fill(colorcore.Color(0xD0E0C42E))
		c.DrawRect(geom.RectLTRB(x+4, y+6, x+48, y+46), dst)
		src := Fill(colorcore.Color(0xB02E5FE0))
		src.Blend = mode
		c.DrawCircle(x+40, y+40, 20, src)
		c.Restore()
	}
}

func init() {
	reg("blend-modes-porterduff", func(c Canvas) {
		blendGrid(c, porterDuffBlendModes)
	})

	reg("blend-modes-separable", func(c Canvas) {
		blendGrid(c, separableBlendModes)
	})

	reg("blend-modes-hsl", func(c Canvas) {
		// The four non-separable HSL modes get bigger cells (128x128) — their per-pixel luminosity/ saturation math is
		// where implementations drift, so give the gate more pixels.
		c.Clear(white)
		for i, mode := range hslBlendModes {
			x := float32(i%2) * 128
			y := float32(i/2) * 128
			c.Save()
			c.ClipRect(geom.RectLTRB(x, y, x+128, y+128), ClipIntersect, false)
			dst := NewPaint()
			dst.AA = true
			dst.Shader = &ShaderSpec{
				Kind:   ShaderLinearGradient,
				Pts:    [2]geom.Point{{X: x, Y: y}, {X: x + 128, Y: y + 128}},
				Colors: []colorcore.Color{red, yellow, teal},
				Tile:   TileClamp,
			}
			c.DrawRect(geom.RectLTRB(x+6, y+6, x+122, y+122), dst)
			src := Fill(colorcore.Color(0xE07A2EE0))
			src.Blend = mode
			c.DrawCircle(x+64, y+64, 52, src)
			c.Restore()
		}
	})

	reg("blend-modes-gradient-dst", func(c Canvas) {
		// Non-flat dst AND src: gradient dst under gradient-shaded src circles per mode.
		c.Clear(white)
		bg := NewPaint()
		bg.AA = true
		bg.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 0, Y: 256}},
			Colors: []colorcore.Color{colorcore.Color(0xFF303860), colorcore.Color(0xFFE8E0C8)},
			Tile:   TileClamp,
		}
		c.DrawPaint(bg)
		modes := []BlendMode{
			BlendMultiply, BlendScreen, BlendOverlay, BlendColorDodge,
			BlendColorBurn, BlendDifference, BlendExclusion, BlendSoftLight,
		}
		for i, mode := range modes {
			x := 36 + float32(i%4)*62
			y := 64 + float32(i/4)*128
			src := NewPaint()
			src.AA = true
			src.Blend = mode
			src.Shader = &ShaderSpec{
				Kind:   ShaderRadialGradient,
				Center: geom.Point{X: x, Y: y},
				Radius: 30,
				Colors: []colorcore.Color{colorcore.Color(0xF0E0392E), colorcore.Color(0x402E9E44)},
				Tile:   TileClamp,
			}
			c.DrawCircle(x, y, 30, src)
		}
	})

	reg("draw-color-blends", func(c Canvas) {
		c.Clear(white)
		// A patterned base so full-canvas DrawColor blends have structure to act on.
		for i := range 8 {
			x := float32(i) * 32
			c.DrawRect(geom.RectLTRB(x, 0, x+16, 256), Fill(colorcore.Color(0xFFB0B8C8)))
		}
		c.DrawCircle(128, 128, 90, Fill(yellow))
		c.DrawColor(colorcore.Color(0x662E5FE0), BlendMultiply)
		c.DrawColor(colorcore.Color(0x40E0392E), BlendPlus)
		c.DrawColor(colorcore.Color(0x30FFFFFF), BlendScreen)
	})
}
