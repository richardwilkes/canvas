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
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// Corpus growth: saveLayer variants — bounded layers, nesting, and the paint-carrying SaveLayer whose restore
// composites through the paint's alpha / blend mode / color filter / image filter.

func init() {
	reg("layer-bounded", func(c Canvas) {
		c.Clear(white)
		c.DrawRect(geom.RectLTRB(0, 0, 256, 128), Fill(colorcore.Color(0xFFD8E0F0)))
		// Layer bounds smaller than the draws inside: content must clip to the layer.
		bounds := geom.RectLTRB(48, 48, 208, 208)
		c.SaveLayerAlpha(&bounds, 160)
		c.DrawCircle(60, 60, 60, Fill(red))
		c.DrawCircle(196, 196, 60, Fill(blue))
		c.DrawRect(geom.RectLTRB(0, 118, 256, 138), Fill(green)) // extends past bounds both sides
		c.Restore()
		c.DrawCircle(128, 128, 12, Fill(black)) // over the composited layer
	})

	// Stacked translucent-layer restores accumulate div255 rounding that differs between SIMD architectures, which is
	// exactly why the raster golden sets are per-platform: this scene gates bit-exactly against its own platform's set
	// in every lane.
	Register(Scenario{
		Name: "layer-nested-alpha", Width: 256, Height: 256,
		Draw: func(c Canvas) {
			c.Clear(white)
			c.SaveLayerAlpha(nil, 200)
			c.DrawCircle(96, 96, 70, Fill(red))
			c.SaveLayerAlpha(nil, 128)
			c.DrawCircle(160, 96, 70, Fill(blue))
			c.SaveLayerAlpha(nil, 90)
			c.DrawCircle(128, 170, 70, Fill(green))
			c.Restore()
			c.Restore()
			c.Restore()
		},
	})

	reg("layer-blend-multiply", func(c Canvas) {
		c.Clear(white)
		bg := NewPaint()
		bg.AA = true
		bg.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 256, Y: 0}},
			Colors: []colorcore.Color{yellow, white, teal},
			Tile:   TileClamp,
		}
		c.DrawPaint(bg)
		// The layer composites onto the gradient with Multiply on restore.
		lp := NewPaint()
		lp.Blend = BlendMultiply
		c.SaveLayer(nil, &lp)
		c.DrawCircle(100, 128, 80, Fill(colorcore.Color(0xFFE08040)))
		c.DrawCircle(156, 128, 80, Fill(colorcore.Color(0xFF4080E0)))
		c.Restore()
	})

	reg("layer-blend-screen-alpha", func(c Canvas) {
		c.Clear(colorcore.Color(0xFF283048))
		c.DrawCircle(128, 128, 100, Fill(colorcore.Color(0xFF603840)))
		// Alpha and blend mode together on the layer paint.
		lp := NewPaint()
		lp.Color = colorcore.Color(0xA0FFFFFF) // layer paint alpha 0xA0
		lp.Blend = BlendScreen
		c.SaveLayer(nil, &lp)
		c.DrawRect(geom.RectLTRB(30, 90, 226, 166), Fill(colorcore.Color(0xFF3070C0)))
		c.DrawCircle(128, 128, 46, Fill(colorcore.Color(0xFFE0C42E)))
		c.Restore()
	})

	reg("layer-colorfilter", func(c Canvas) {
		c.Clear(white)
		// Desaturating color-matrix filter applied to the whole layer on restore; half the scene sits outside the layer
		// as the saturated reference.
		c.DrawCircle(64, 64, 48, Fill(red))
		c.DrawCircle(64, 192, 48, Fill(green))
		var desat float32 = 0.25
		m := [20]float32{
			0.213 + 0.787*desat, 0.715 - 0.715*desat, 0.072 - 0.072*desat, 0, 0,
			0.213 - 0.213*desat, 0.715 + 0.285*desat, 0.072 - 0.072*desat, 0, 0,
			0.213 - 0.213*desat, 0.715 - 0.715*desat, 0.072 + 0.928*desat, 0, 0,
			0, 0, 0, 1, 0,
		}
		lp := NewPaint()
		lp.ColorFilter = &ColorFilterSpec{Kind: ColorFilterMatrix, Matrix: &m}
		bounds := geom.RectLTRB(128, 0, 256, 256)
		c.SaveLayer(&bounds, &lp)
		c.DrawCircle(192, 64, 48, Fill(red))
		c.DrawCircle(192, 192, 48, Fill(green))
		c.Restore()
	})

	reg("layer-imagefilter-blur", func(c Canvas) {
		c.Clear(white)
		c.DrawRect(geom.RectLTRB(0, 120, 256, 136), Fill(gray))
		// The layer's restore runs its content through a Gaussian blur; the black frame outside stays sharp.
		lp := NewPaint()
		lp.ImageFilter = &ImageFilterSpec{Kind: FilterBlur, SigmaX: 4, SigmaY: 4, Tile: TileDecal}
		bounds := geom.RectLTRB(32, 32, 224, 224)
		c.SaveLayer(&bounds, &lp)
		c.DrawCircle(96, 96, 44, Fill(red))
		c.DrawRect(geom.RectLTRB(120, 130, 210, 200), Fill(blue))
		c.Restore()
		c.DrawPath(NewPathSpec().AddRect(geom.RectLTRB(32, 32, 224, 224), geom.DirectionCW),
			Stroke(black, 3))
	})

	reg("layer-under-transform", func(c Canvas) {
		c.Clear(white)
		c.Translate(128, 128)
		c.RotateRadians(float32(math.Pi) / 7)
		lp := NewPaint()
		lp.Color = colorcore.Color(0x90000000)
		lp.Blend = BlendSrcOver
		bounds := geom.RectLTRB(-80, -80, 80, 80)
		c.SaveLayer(&bounds, &lp)
		// Drawn inside a rotated, bounded, translucent layer.
		c.DrawRect(geom.RectLTRB(-70, -70, 70, 70), Fill(teal))
		c.DrawCircle(0, 0, 40, Fill(yellow))
		c.Restore()
	})

	reg("layer-clip-interaction", func(c Canvas) {
		c.Clear(white)
		c.ClipPath(NewPathSpec().AddCircle(128, 128, 100, geom.DirectionCW), ClipIntersect, true)
		c.SaveLayerAlpha(nil, 140)
		// The layer inherits the circular clip; this paint floods only the clip.
		c.DrawPaint(Fill(purple))
		c.ClipRect(geom.RectLTRB(64, 64, 192, 192), ClipIntersect, false)
		c.DrawPaint(Fill(yellow))
		c.Restore()
	})
}
