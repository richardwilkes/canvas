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

// Corpus growth: color-filter coverage — one scenario per sk_colorfilter_new_* constructor the spec carries,
// applied over a shared colorful base scene so filter output has structure to act on.

// cfBase draws a colorful reference scene: a diagonal gradient background with three solid shapes.
func cfBase(c Canvas, filter *ColorFilterSpec) {
	c.Clear(white)
	bg := NewPaint()
	bg.AA = true
	bg.ColorFilter = filter
	bg.Shader = &ShaderSpec{
		Kind:   ShaderLinearGradient,
		Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 256, Y: 256}},
		Colors: []colorcore.Color{red, yellow, green, blue},
		Tile:   TileClamp,
	}
	c.DrawRect(geom.RectLTRB(8, 8, 248, 148), bg)
	shapes := []struct {
		color colorcore.Color
		cx    float32
	}{{color: colorcore.Color(0xFFE0392E), cx: 56}, {color: colorcore.Color(0x902E9E44), cx: 128}, {color: colorcore.Color(0xFF7A2EE0), cx: 200}}
	for _, s := range shapes {
		p := Fill(s.color)
		p.ColorFilter = filter
		c.DrawCircle(s.cx, 200, 40, p)
	}
}

func init() {
	reg("colorfilter-matrix-saturate", func(c Canvas) {
		var s float32 = 0.2
		m := [20]float32{
			0.213 + 0.787*s, 0.715 - 0.715*s, 0.072 - 0.072*s, 0, 0,
			0.213 - 0.213*s, 0.715 + 0.285*s, 0.072 - 0.072*s, 0, 0,
			0.213 - 0.213*s, 0.715 - 0.715*s, 0.072 + 0.928*s, 0, 0,
			0, 0, 0, 1, 0,
		}
		cfBase(c, &ColorFilterSpec{Kind: ColorFilterMatrix, Matrix: &m})
	})

	reg("colorfilter-matrix-swap", func(c Canvas) {
		// Channel rotation R<-G, G<-B, B<-R plus a bias on blue (exercises the +offset column).
		m := [20]float32{
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			1, 0, 0, 0, 0.15,
			0, 0, 0, 1, 0,
		}
		cfBase(c, &ColorFilterSpec{Kind: ColorFilterMatrix, Matrix: &m})
	})

	reg("colorfilter-blend-tint", func(c Canvas) {
		cfBase(c, &ColorFilterSpec{
			Kind: ColorFilterBlend, Color: colorcore.Color(0x802E5FE0),
			Blend: BlendMultiply,
		})
	})

	reg("colorfilter-lighting", func(c Canvas) {
		cfBase(c, &ColorFilterSpec{
			Kind: ColorFilterLighting, Mul: colorcore.Color(0xFF80C0FF),
			Add: colorcore.Color(0xFF201000),
		})
	})

	reg("colorfilter-luma", func(c Canvas) {
		// Luma turns luminance into alpha (RGB goes to 0) — draw over a patterned base so the resulting alpha ramp
		// shows.
		c.Clear(white)
		for i := range 8 {
			x := float32(i) * 32
			c.DrawRect(geom.RectLTRB(x, 0, x+16, 256), Fill(colorcore.Color(0xFFE0C42E)))
		}
		p := NewPaint()
		p.AA = true
		p.ColorFilter = &ColorFilterSpec{Kind: ColorFilterLuma}
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 20}, {X: 0, Y: 236}},
			Colors: []colorcore.Color{white, black},
			Tile:   TileClamp,
		}
		c.DrawRect(geom.RectLTRB(20, 20, 236, 236), p)
	})

	reg("colorfilter-highcontrast", func(c Canvas) {
		cfBase(c, &ColorFilterSpec{
			Kind: ColorFilterHighContrast, Grayscale: false,
			InvertStyle: InvertBrightness, Contrast: 0.3,
		})
	})

	reg("colorfilter-compose", func(c Canvas) {
		// Outer lighting filter over an inner saturation matrix.
		var s float32 // saturation 0
		m := [20]float32{
			0.213 + 0.787*s, 0.715 - 0.715*s, 0.072 - 0.072*s, 0, 0,
			0.213 - 0.213*s, 0.715 + 0.285*s, 0.072 - 0.072*s, 0, 0,
			0.213 - 0.213*s, 0.715 - 0.715*s, 0.072 + 0.928*s, 0, 0,
			0, 0, 0, 1, 0,
		}
		cfBase(c, &ColorFilterSpec{
			Kind:  ColorFilterCompose,
			Outer: &ColorFilterSpec{Kind: ColorFilterLighting, Mul: colorcore.Color(0xFFFFA060), Add: colorcore.Color(0xFF000020)},
			Inner: &ColorFilterSpec{Kind: ColorFilterMatrix, Matrix: &m},
		})
	})
}
