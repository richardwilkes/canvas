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

// Corpus growth: image filters, one scenario per sk_imagefilter_new_* constructor the declarative spec reaches
// ("image filters per constructor"). Parameter ranges follow the GPU image-filter live differential scenes, which both
// GPU backends agreed on; those scenes were one-offs outside the corpus, and once these scenarios covered every
// constructor they reached — all 21 ImageFilterKind values, including the six straggler kernels (morphology,
// displacement, lighting, arithmetic, convolution, magnifier) — the one-offs were redundant and went with the rest of
// the C differentials. These gate in every lane against the self-captured golden sets.

const ifClear = colorcore.Color(0xFF202830)

func init() {
	reg("imagefilter-blur-tilemodes", func(c Canvas) {
		c.Clear(white)
		for i, tile := range []TileMode{TileClamp, TileRepeat, TileMirror, TileDecal} {
			x := 24 + float32(i%2)*120
			y := 24 + float32(i/2)*120
			p := Fill(blue)
			p.ImageFilter = &ImageFilterSpec{Kind: FilterBlur, SigmaX: 5, SigmaY: 5, Tile: tile}
			c.DrawRect(geom.RectLTRB(x, y, x+88, y+88), p)
		}
	})

	reg("imagefilter-blur-aniso", func(c Canvas) {
		c.Clear(white)
		p := Fill(red)
		p.ImageFilter = &ImageFilterSpec{Kind: FilterBlur, SigmaX: 12, SigmaY: 1, Tile: TileDecal}
		c.DrawCircle(128, 70, 42, p)
		p2 := Fill(green)
		p2.ImageFilter = &ImageFilterSpec{Kind: FilterBlur, SigmaX: 0.8, SigmaY: 10, Tile: TileDecal}
		c.DrawRect(geom.RectLTRB(84, 150, 172, 226), p2)
	})

	reg("imagefilter-dropshadow", func(c Canvas) {
		c.Clear(white)
		p := Fill(colorcore.Color(0xFF3060C0))
		p.ImageFilter = &ImageFilterSpec{
			Kind: FilterDropShadow, Dx: 10, Dy: 8, SigmaX: 4, SigmaY: 4,
			Color: colorcore.Color(0xB0000000),
		}
		c.DrawRoundRect(geom.RectLTRB(32, 32, 140, 120), 14, 14, p)
		p2 := Fill(colorcore.Color(0xFFE0392E))
		p2.ImageFilter = &ImageFilterSpec{
			Kind: FilterDropShadowOnly, Dx: -8, Dy: 12, SigmaX: 5, SigmaY: 5,
			Color: colorcore.Color(0xC02E9E44),
		} // shadow-only: the red circle itself must not appear
		c.DrawCircle(180, 180, 44, p2)
	})

	reg("imagefilter-offset", func(c Canvas) {
		c.Clear(white)
		c.DrawRect(geom.RectLTRB(40, 40, 120, 120), Fill(colorcore.Color(0xFFD0D4DC))) // where it would be
		p := Fill(purple)
		p.ImageFilter = &ImageFilterSpec{Kind: FilterOffset, Dx: 70, Dy: 90}
		c.DrawRect(geom.RectLTRB(40, 40, 120, 120), p)
	})

	reg("imagefilter-colorfilter", func(c Canvas) {
		c.Clear(white)
		// Color-filter image filter (invert-ish matrix) wrapped around a blur input.
		m := [20]float32{
			-1, 0, 0, 0, 1,
			0, -1, 0, 0, 1,
			0, 0, -1, 0, 1,
			0, 0, 0, 1, 0,
		}
		p := Fill(teal)
		p.ImageFilter = &ImageFilterSpec{
			Kind:        FilterColorFilter,
			ColorFilter: &ColorFilterSpec{Kind: ColorFilterMatrix, Matrix: &m},
			Input:       &ImageFilterSpec{Kind: FilterBlur, SigmaX: 3, SigmaY: 3, Tile: TileDecal},
		}
		c.DrawCircle(128, 128, 70, p)
	})

	reg("imagefilter-compose", func(c Canvas) {
		c.Clear(white)
		// outer(inner(src)): offset applied to the blurred source.
		p := Fill(red)
		p.ImageFilter = &ImageFilterSpec{
			Kind: FilterCompose,
			A:    &ImageFilterSpec{Kind: FilterOffset, Dx: 40, Dy: 30},
			B:    &ImageFilterSpec{Kind: FilterBlur, SigmaX: 4, SigmaY: 4, Tile: TileDecal},
		}
		c.DrawRect(geom.RectLTRB(40, 40, 150, 150), p)
	})

	reg("imagefilter-merge", func(c Canvas) {
		c.Clear(white)
		p := Fill(colorcore.Color(0xFF2E5FE0))
		p.ImageFilter = &ImageFilterSpec{
			Kind: FilterMerge,
			Inputs: []*ImageFilterSpec{
				{Kind: FilterOffset, Dx: -30, Dy: -20},
				{Kind: FilterBlur, SigmaX: 6, SigmaY: 6, Tile: TileDecal},
				{Kind: FilterOffset, Dx: 30, Dy: 20},
			},
		}
		c.DrawCircle(128, 128, 44, p)
	})

	reg("imagefilter-arithmetic", func(c Canvas) {
		c.Clear(ifClear)
		p := Fill(colorcore.Color(0xFF3060C0))
		p.AA = true
		p.ImageFilter = &ImageFilterSpec{
			Kind:      FilterArithmetic,
			K:         [4]float32{0.5, 0.5, 0.5, 0},
			EnforcePM: true,
			B:         &ImageFilterSpec{Kind: FilterBlur, SigmaX: 3, SigmaY: 3, Tile: TileDecal},
		}
		c.DrawRect(geom.RectLTRB(48, 48, 208, 208), p)
	})

	reg("imagefilter-morphology", func(c Canvas) {
		c.Clear(ifClear)
		p := Fill(colorcore.Color(0xFFFF8000))
		p.AA = true
		p.ImageFilter = &ImageFilterSpec{Kind: FilterDilate, RadiusX: 8, RadiusY: 4}
		c.DrawRect(geom.RectLTRB(36, 44, 104, 112), p)
		p2 := Fill(colorcore.Color(0xFF20A060))
		p2.AA = true
		p2.ImageFilter = &ImageFilterSpec{Kind: FilterErode, RadiusX: 6, RadiusY: 6}
		c.DrawRect(geom.RectLTRB(144, 44, 232, 132), p2)
		p3 := Fill(colorcore.Color(0xFF3060C0))
		p3.AA = true
		p3.ImageFilter = &ImageFilterSpec{Kind: FilterDilate, RadiusX: 3, RadiusY: 9}
		c.DrawCircle(128, 192, 34, p3)
	})

	reg("imagefilter-displacement", func(c Canvas) {
		c.Clear(ifClear)
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 48, Y: 48}, {X: 208, Y: 208}},
			Colors: []colorcore.Color{colorcore.Color(0xFFFF0000), colorcore.Color(0xFF00FF00), colorcore.Color(0xFF4040FF)},
			Tile:   TileClamp,
		}
		p.ImageFilter = &ImageFilterSpec{
			Kind: FilterDisplacement, XChannel: ChannelR,
			YChannel: ChannelG, Scale: 24,
		}
		c.DrawRect(geom.RectLTRB(48, 48, 208, 208), p)
	})

	reg("imagefilter-convolution", func(c Canvas) {
		c.Clear(ifClear)
		sharpen := []float32{
			0, -1, 0,
			-1, 5, -1,
			0, -1, 0,
		}
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 24, Y: 24}, {X: 116, Y: 116}},
			Colors: []colorcore.Color{colorcore.Color(0xFF202020), colorcore.Color(0xFFF0F0F0)},
			Tile:   TileClamp,
		}
		p.ImageFilter = &ImageFilterSpec{
			Kind: FilterConvolution, KernelW: 3, KernelH: 3,
			Kernel: sharpen, Gain: 1, Bias: 0, OffsetX: 1, OffsetY: 1, Tile: TileDecal,
		}
		c.DrawRect(geom.RectLTRB(24, 24, 116, 116), p)
		box := make([]float32, 25)
		for i := range box {
			box[i] = 1.0 / 25
		}
		p2 := Fill(colorcore.Color(0xFF20A060))
		p2.ImageFilter = &ImageFilterSpec{
			Kind: FilterConvolution, KernelW: 5, KernelH: 5,
			Kernel: box, Gain: 1, Bias: 0, OffsetX: 2, OffsetY: 2, Tile: TileDecal, ConvolveAlpha: true,
		}
		c.DrawRect(geom.RectLTRB(144, 132, 232, 220), p2)
	})

	reg("imagefilter-matrix-transform", func(c Canvas) {
		// The rect is 90x90 rather than the 110x110 it started as because the rotated linear-resample area used to be
		// held inside a cross-renderer tolerance budget: bilerp weight rounding drifted a couple of LSBs between the
		// C oracle's SIMD lanes and the port's, and at 110x110 that drifting area exceeded the raster differential's
		// 0.5% budget on the linux leg. No live budget constrains the geometry now — the C oracle is gone and every
		// lane gates against self-captured goldens ((near-)bit-exactly) — so the size is simply what the goldens hold.
		c.Clear(white)
		m := geom.IdentityMatrix()
		m.SetAll(0.86, -0.5, 60, 0.5, 0.86, -30, 0, 0, 1) // rotate 30 degrees + offset
		p := Fill(colorcore.Color(0xFF7A2EE0))
		p.ImageFilter = &ImageFilterSpec{Kind: FilterMatrixTransform, Matrix: &m, Linear: true}
		c.DrawRect(geom.RectLTRB(60, 60, 150, 150), p)
		c.DrawPath(NewPathSpec().AddRect(geom.RectLTRB(60, 60, 150, 150), geom.DirectionCW),
			Stroke(black, 2)) // where the unfiltered rect would be
	})

	reg("imagefilter-magnifier", func(c Canvas) {
		c.Clear(ifClear)
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 40, Y: 40}, {X: 216, Y: 40}},
			Colors: []colorcore.Color{colorcore.Color(0xFF000000), colorcore.Color(0xFFFFFFFF)},
			Tile:   TileClamp,
		}
		p.ImageFilter = &ImageFilterSpec{
			Kind: FilterMagnifier,
			Lens: geom.RectLTRB(40, 40, 216, 216), Zoom: 2, Inset: 12, Linear: true,
		}
		c.DrawRect(geom.RectLTRB(40, 40, 216, 216), p)
	})

	reg("imagefilter-lit-diffuse", func(c Canvas) {
		c.Clear(ifClear)
		p := Fill(colorcore.Color(0xFF3060C0))
		p.AA = true
		p.ImageFilter = &ImageFilterSpec{
			Kind:     FilterPointLitDiffuse,
			Location: Point3{X: 80, Y: 60, Z: 80}, LightColor: white, SurfaceScale: 4, KD: 1,
		}
		c.DrawRect(geom.RectLTRB(32, 32, 120, 120), p)
		p2 := Fill(colorcore.Color(0xFFC03028))
		p2.AA = true
		p2.ImageFilter = &ImageFilterSpec{
			Kind:      FilterDistantLitDiffuse,
			Direction: Point3{X: 0, Y: -1, Z: 1}, LightColor: white, SurfaceScale: 3, KD: 1.2,
		}
		c.DrawRoundRect(geom.RectLTRB(140, 140, 224, 224), 18, 18, p2)
	})

	reg("imagefilter-lit-specular", func(c Canvas) {
		c.Clear(ifClear)
		p := Fill(colorcore.Color(0xFF20A060))
		p.AA = true
		p.ImageFilter = &ImageFilterSpec{
			Kind:      FilterDistantLitSpecular,
			Direction: Point3{X: 0, Y: -1, Z: 1}, LightColor: white, SurfaceScale: 3, KS: 1, Shininess: 8,
		}
		c.DrawRoundRect(geom.RectLTRB(32, 32, 120, 120), 12, 12, p)
		p2 := Fill(colorcore.Color(0xFF3060C0))
		p2.AA = true
		p2.ImageFilter = &ImageFilterSpec{
			Kind:     FilterSpotLitSpecular,
			Location: Point3{X: 190, Y: 120, Z: 70}, Target: Point3{X: 180, Y: 182, Z: 0},
			SpecExp: 2, CutoffAngle: 40, LightColor: white, SurfaceScale: 4, KS: 0.9, Shininess: 6,
		}
		c.DrawCircle(180, 182, 42, p2)
	})

	reg("imagefilter-lit-spot-diffuse", func(c Canvas) {
		c.Clear(ifClear)
		p := Fill(colorcore.Color(0xFFB06020))
		p.AA = true
		p.ImageFilter = &ImageFilterSpec{
			Kind:     FilterSpotLitDiffuse,
			Location: Point3{X: 128, Y: 40, Z: 90}, Target: Point3{X: 128, Y: 128, Z: 0},
			SpecExp: 1.5, CutoffAngle: 45, LightColor: white, SurfaceScale: 4, KD: 1.1,
		}
		c.DrawRect(geom.RectLTRB(48, 48, 208, 208), p)
		p2 := Fill(colorcore.Color(0xFF6040C0))
		p2.AA = true
		p2.ImageFilter = &ImageFilterSpec{
			Kind:     FilterPointLitSpecular,
			Location: Point3{X: 128, Y: 128, Z: 60}, LightColor: white, SurfaceScale: 3, KS: 1, Shininess: 10,
		}
		c.DrawCircle(128, 128, 34, p2)
	})

	reg("imagefilter-crop-rect", func(c Canvas) {
		c.Clear(white)
		// The same blur with and without a crop rect: the crop must slice the blur output.
		crop := geom.RectLTRB(24, 24, 120, 120)
		p := Fill(colorcore.Color(0xFFE0392E))
		p.ImageFilter = &ImageFilterSpec{Kind: FilterBlur, SigmaX: 6, SigmaY: 6, Tile: TileDecal, Crop: &crop}
		c.DrawCircle(72, 72, 40, p)
		p2 := Fill(colorcore.Color(0xFF2E5FE0))
		p2.ImageFilter = &ImageFilterSpec{Kind: FilterBlur, SigmaX: 6, SigmaY: 6, Tile: TileDecal}
		c.DrawCircle(184, 184, 40, p2)
	})

	reg("imagefilter-tile", func(c Canvas) {
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 32, Y: 32}, {X: 80, Y: 80}},
			Colors: []colorcore.Color{red, yellow, blue},
			Tile:   TileClamp,
		}
		p.ImageFilter = &ImageFilterSpec{
			Kind:    FilterTile,
			SrcRect: geom.RectLTRB(32, 32, 80, 80), DstRect: geom.RectLTRB(32, 32, 224, 224),
		}
		c.DrawRect(geom.RectLTRB(32, 32, 80, 80), p)
	})
}
