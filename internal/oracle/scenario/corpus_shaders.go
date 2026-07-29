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

// Shader coverage beyond the linear/radial pair — sweep and two-point conical gradients, the color
// and blend shaders, tile-mode and stop-position edge cases, local matrices, and dither.

func init() {
	reg("gradient-sweep", func(c Canvas) {
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderSweepGradient,
			Center: geom.Point{X: 70, Y: 70},
			Colors: []colorcore.Color{red, yellow, green, blue, red},
			Tile:   TileClamp,
		}
		c.DrawCircle(70, 70, 58, p)
		// Partial sweep window with repeat outside it.
		p.Shader = &ShaderSpec{
			Kind:       ShaderSweepGradient,
			Center:     geom.Point{X: 186, Y: 70},
			StartAngle: 30,
			EndAngle:   150,
			Colors:     []colorcore.Color{purple, teal},
			Tile:       TileRepeat,
		}
		c.DrawCircle(186, 70, 58, p)
		// Partial sweep, mirrored, on a rect.
		p.Shader = &ShaderSpec{
			Kind:       ShaderSweepGradient,
			Center:     geom.Point{X: 128, Y: 196},
			StartAngle: -45,
			EndAngle:   45,
			Colors:     []colorcore.Color{black, yellow, black},
			Pos:        []float32{0, 0.35, 1},
			Tile:       TileMirror,
		}
		c.DrawRect(geom.RectLTRB(24, 140, 232, 252), p)
	})

	reg("gradient-conical", func(c Canvas) {
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		// Classic offset two-point conical (a "spotlight").
		p.Shader = &ShaderSpec{
			Kind:    ShaderTwoPointConicalGradient,
			Pts:     [2]geom.Point{{X: 96, Y: 96}, {X: 128, Y: 128}},
			Radius:  12,
			Radius2: 110,
			Colors:  []colorcore.Color{white, blue, black},
			Pos:     []float32{0, 0.6, 1},
			Tile:    TileClamp,
		}
		c.DrawRect(geom.RectLTRB(8, 8, 248, 128), p)
		// Non-contained circles (the strip case) with repeat.
		p.Shader = &ShaderSpec{
			Kind:    ShaderTwoPointConicalGradient,
			Pts:     [2]geom.Point{{X: 60, Y: 196}, {X: 196, Y: 196}},
			Radius:  28,
			Radius2: 44,
			Colors:  []colorcore.Color{red, yellow},
			Tile:    TileRepeat,
		}
		c.DrawRect(geom.RectLTRB(8, 140, 248, 252), p)
	})

	reg("gradient-decal", func(c Canvas) {
		// Decal tiling renders transparent outside the gradient span — over a checkered base so the transparent region
		// is visible.
		c.Clear(white)
		for row := range 8 {
			for col := range 8 {
				if (row+col)%2 == 0 {
					x, y := float32(col)*32, float32(row)*32
					c.DrawRect(geom.RectLTRB(x, y, x+32, y+32), Fill(colorcore.Color(0xFFD0D4DC)))
				}
			}
		}
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 64, Y: 0}, {X: 192, Y: 0}},
			Colors: []colorcore.Color{red, blue},
			Tile:   TileDecal,
		}
		c.DrawRect(geom.RectLTRB(16, 16, 240, 120), p)
		p.Shader = &ShaderSpec{
			Kind:   ShaderRadialGradient,
			Center: geom.Point{X: 128, Y: 192},
			Radius: 40,
			Colors: []colorcore.Color{green, purple},
			Tile:   TileDecal,
		}
		c.DrawRect(geom.RectLTRB(16, 136, 240, 248), p)
	})

	reg("gradient-local-matrix", func(c Canvas) {
		c.Clear(white)
		rot := geom.IdentityMatrix()
		rot.SetAll(0.7071, -0.7071, 96, 0.7071, 0.7071, -40, 0, 0, 1) // 45-degree rotation
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 64, Y: 0}},
			Colors: []colorcore.Color{teal, black},
			Tile:   TileMirror,
			Local:  &rot,
		}
		c.DrawRect(geom.RectLTRB(8, 8, 248, 120), p)
		squash := geom.IdentityMatrix()
		squash.SetAll(1.8, 0, 30, 0, 0.5, 100, 0, 0, 1) // anisotropic scale + offset
		p.Shader = &ShaderSpec{
			Kind:   ShaderRadialGradient,
			Center: geom.Point{X: 64, Y: 190},
			Radius: 40,
			Colors: []colorcore.Color{yellow, red, purple},
			Tile:   TileRepeat,
			Local:  &squash,
		}
		c.DrawRect(geom.RectLTRB(8, 136, 248, 248), p)
	})

	reg("gradient-many-stops", func(c Canvas) {
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind: ShaderLinearGradient,
			Pts:  [2]geom.Point{{X: 10, Y: 0}, {X: 246, Y: 0}},
			Colors: []colorcore.Color{
				red, yellow, green, teal, blue, purple,
				colorcore.Color(0xFFE06090), black, white,
			},
			Pos:  []float32{0, 0.08, 0.22, 0.35, 0.51, 0.62, 0.80, 0.93, 1},
			Tile: TileClamp,
		}
		c.DrawRect(geom.RectLTRB(10, 10, 246, 122), p)
		// Hard stops: duplicated positions make color bands with sharp seams.
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 10, Y: 0}, {X: 246, Y: 0}},
			Colors: []colorcore.Color{red, red, yellow, yellow, blue, blue},
			Pos:    []float32{0, 0.33, 0.33, 0.66, 0.66, 1},
			Tile:   TileClamp,
		}
		c.DrawRect(geom.RectLTRB(10, 134, 246, 246), p)
	})

	reg("gradient-200-stops", func(c Canvas) {
		// The textured colorizer: gradients past the GPU looping colorizer's 128-color limit, which the GL backend
		// renders through a 256×1 LUT texture (make_textured_colorizer) — the analytic colorizers cannot represent
		// them. The raster lane renders the same stops analytically, so this pins the raster and GPU lanes against
		// their own self-captured goldens at high stop counts. Stops are evenly spaced (the public surface's
		// nil-positions form); colors are a deterministic modular walk so both lanes build identical stop lists.
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		linearStops := make([]colorcore.Color, 200)
		for i := range linearStops {
			r := uint32(i*41) % 256
			g := uint32(i*97+64) % 256
			b := uint32(255 - (i*23)%256)
			linearStops[i] = colorcore.Color(0xFF000000 | r<<16 | g<<8 | b)
		}
		// The gradient span is narrower than the rect so the clamp borders (the outer stop colors) show on both sides.
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 48, Y: 0}, {X: 208, Y: 0}},
			Colors: linearStops,
			Tile:   TileClamp,
		}
		c.DrawRect(geom.RectLTRB(10, 10, 246, 122), p)
		// Radial with 210 stops, some translucent (the LUT bakes the premultiply), repeat tiling.
		radialStops := make([]colorcore.Color, 210)
		for i := range radialStops {
			r := uint32(255 - (i*67)%256)
			g := uint32(i*29) % 256
			b := uint32(i*131+32) % 256
			a := uint32(128 + (i*53)%128)
			radialStops[i] = colorcore.Color(a<<24 | r<<16 | g<<8 | b)
		}
		p.Shader = &ShaderSpec{
			Kind:   ShaderRadialGradient,
			Center: geom.Point{X: 128, Y: 190},
			Radius: 44,
			Colors: radialStops,
			Tile:   TileRepeat,
		}
		c.DrawRect(geom.RectLTRB(10, 134, 246, 246), p)
	})

	reg("gradient-alpha-stops", func(c Canvas) {
		// Gradients whose stops carry alpha, over a patterned base: interpolation happens in premul.
		c.Clear(white)
		for i := range 6 {
			x := float32(i) * 44
			c.DrawRect(geom.RectLTRB(x, 0, x+22, 256), Fill(colorcore.Color(0xFFC8CCD4)))
		}
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 12}, {X: 0, Y: 116}},
			Colors: []colorcore.Color{colorcore.Color(0xFFE0392E), colorcore.Color(0x00E0392E)},
			Tile:   TileClamp,
		}
		c.DrawRect(geom.RectLTRB(12, 12, 244, 116), p)
		p.Shader = &ShaderSpec{
			Kind:   ShaderRadialGradient,
			Center: geom.Point{X: 128, Y: 190},
			Radius: 66,
			Colors: []colorcore.Color{colorcore.Color(0x202E5FE0), colorcore.Color(0xF02E5FE0), colorcore.Color(0x302E9E44)},
			Pos:    []float32{0, 0.55, 1},
			Tile:   TileClamp,
		}
		c.DrawCircle(128, 190, 66, p)
	})

	reg("shader-color-blend", func(c Canvas) {
		c.Clear(white)
		// The single-color shader (paint alpha still modulates it), plus the blend-of-shaders combinator.
		p := NewPaint()
		p.AA = true
		p.Color = colorcore.Color(0x80FFFFFF) // alpha 0x80 modulates the shader
		p.Shader = &ShaderSpec{Kind: ShaderColor, Color: red}
		c.DrawRect(geom.RectLTRB(12, 12, 122, 122), p)
		p2 := NewPaint()
		p2.AA = true
		p2.Shader = &ShaderSpec{
			Kind:  ShaderBlend,
			Blend: BlendMultiply,
			Dst: &ShaderSpec{
				Kind:   ShaderLinearGradient,
				Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 256, Y: 0}},
				Colors: []colorcore.Color{white, blue},
				Tile:   TileClamp,
			},
			Src: &ShaderSpec{
				Kind:   ShaderLinearGradient,
				Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 0, Y: 256}},
				Colors: []colorcore.Color{white, red},
				Tile:   TileClamp,
			},
		}
		c.DrawRect(geom.RectLTRB(134, 12, 244, 122), p2)
		p3 := NewPaint()
		p3.AA = true
		p3.Shader = &ShaderSpec{
			Kind:  ShaderBlend,
			Blend: BlendDifference,
			Dst: &ShaderSpec{
				Kind:   ShaderSweepGradient,
				Center: geom.Point{X: 128, Y: 190},
				Colors: []colorcore.Color{black, white, black},
				Tile:   TileClamp,
			},
			Src: &ShaderSpec{
				Kind:   ShaderRadialGradient,
				Center: geom.Point{X: 128, Y: 190},
				Radius: 58,
				Colors: []colorcore.Color{yellow, purple},
				Tile:   TileMirror,
			},
		}
		c.DrawCircle(128, 190, 58, p3)
	})

	reg("gradient-dither", func(c Canvas) {
		// A deliberately narrow-range gradient with Dither on: both engines share Skia's ordered 4x4 dither lattice, so
		// the noise pattern itself is compared.
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		p.Dither = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 256, Y: 256}},
			Colors: []colorcore.Color{colorcore.Color(0xFF101418), colorcore.Color(0xFF202830)},
			Tile:   TileClamp,
		}
		c.DrawPaint(p)
		p.Dither = false
		c.DrawRect(geom.RectLTRB(64, 64, 192, 192), p) // undithered inset for contrast
	})

	reg("gradient-on-stroke", func(c Canvas) {
		// The ring is sized deliberately: tall enough (2*(44+6) rows) that the library takes the banded parallel
		// shaded-path-fill lane, so the golden gates keep exercising it. Banded output is not byte-exact against a
		// *serial* fill (AA seam pixels can differ by a coverage step; canvas/parallel_path_test.go), but the default
		// band split is a pure function of fill height (raster.bandCount), never of the machine, so the banded render
		// itself is deterministic per platform and gates bit-exactly against the self-captured raster goldens.
		// Shrinking below ~46 rows would drop out of the banded lane and stop exercising it.
		c.Clear(white)
		p := Stroke(black, 12)
		p.Shader = &ShaderSpec{
			Kind:   ShaderSweepGradient,
			Center: geom.Point{X: 128, Y: 148},
			Colors: []colorcore.Color{red, yellow, green, blue, red},
			Tile:   TileClamp,
		}
		c.DrawCircle(128, 148, 44, p)
		p2 := Stroke(black, 10)
		p2.Cap = CapRound
		p2.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 24, Y: 0}, {X: 232, Y: 0}},
			Colors: []colorcore.Color{purple, teal},
			Tile:   TileClamp,
		}
		c.DrawPath(NewPathSpec().MoveTo(24, 40).CubicTo(90, -10, 166, 90, 232, 40), p2)
	})
}
