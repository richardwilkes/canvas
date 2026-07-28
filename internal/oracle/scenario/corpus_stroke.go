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

// The stroking / dash / path-effect family. Dashing is the highest-value path-effect coverage (the
// one real apps use constantly); the rest exercises each sk_path_effect_create_* constructor once through both
// backends.

func dash(intervals []float32, phase float32) *PathEffectSpec {
	return &PathEffectSpec{Kind: PathEffectDash, Intervals: intervals, Phase: phase}
}

func init() {
	reg("dash-caps", func(c Canvas) {
		c.Clear(white)
		for i, cap := range []StrokeCap{CapButt, CapRound, CapSquare} {
			p := Stroke(blue, 12)
			p.Cap = cap
			p.PathEffect = dash([]float32{24, 16}, 0)
			y := 36 + float32(i)*40
			c.DrawLine(20, y, 236, y, p)
		}
		// The same caps on a hairline dash.
		for i, cap := range []StrokeCap{CapButt, CapRound, CapSquare} {
			p := Stroke(red, 0)
			p.Cap = cap
			p.PathEffect = dash([]float32{10, 6}, 0)
			y := 176 + float32(i)*24
			c.DrawLine(20, y, 236, y, p)
		}
	})

	reg("dash-phase", func(c Canvas) {
		c.Clear(white)
		for i := range 6 {
			p := Stroke(purple, 8)
			p.PathEffect = dash([]float32{30, 18}, float32(i)*8)
			y := 30 + float32(i)*40
			c.DrawLine(16, y, 240, y, p)
		}
	})

	reg("dash-closed-contours", func(c Canvas) {
		c.Clear(white)
		p := Stroke(green, 6)
		p.PathEffect = dash([]float32{18, 10}, 4)
		c.DrawPath(NewPathSpec().AddRect(geom.RectLTRB(24, 24, 120, 120), geom.DirectionCW), p)
		p2 := Stroke(red, 6)
		p2.PathEffect = dash([]float32{14, 8}, 0)
		c.DrawPath(NewPathSpec().AddCircle(184, 72, 52, geom.DirectionCW), p2)
		p3 := Stroke(blue, 4)
		p3.PathEffect = dash([]float32{12, 6, 4, 6}, 0) // 4-interval dash
		c.DrawPath(NewPathSpec().AddRoundRect(geom.RectLTRB(40, 150, 216, 236), 24, 24, geom.DirectionCW), p3)
	})

	reg("dash-curves", func(c Canvas) {
		c.Clear(white)
		p := Stroke(teal, 7)
		p.Cap = CapRound
		p.PathEffect = dash([]float32{20, 14}, 0)
		c.DrawPath(NewPathSpec().
			MoveTo(16, 70).CubicTo(80, -30, 176, 170, 240, 70), p)
		p2 := Stroke(black, 5)
		p2.PathEffect = dash([]float32{2, 6}, 0) // dot-like dash on a tight quad
		c.DrawPath(NewPathSpec().MoveTo(16, 160).QuadTo(128, 40, 240, 160), p2)
		p3 := Stroke(purple, 9)
		p3.Cap = CapSquare
		p3.PathEffect = dash([]float32{26, 20}, 12)
		c.DrawPath(NewPathSpec().MoveTo(16, 220).ConicTo(128, 140, 240, 220, 0.5), p3)
	})

	reg("dash-joins-zigzag", func(c Canvas) {
		c.Clear(white)
		zig := func() *PathSpec {
			return NewPathSpec().MoveTo(0, 50).LineTo(45, 0).LineTo(90, 50).LineTo(135, 0).LineTo(180, 50)
		}
		for i, join := range []StrokeJoin{JoinMiter, JoinRound, JoinBevel} {
			p := Stroke(red, 10)
			p.Join = join
			p.PathEffect = dash([]float32{34, 12}, 17) // phase puts dash gaps astride the corners
			c.Save()
			c.Translate(38, 20+float32(i)*78)
			c.DrawPath(zig(), p)
			c.Restore()
		}
	})

	reg("patheffect-corner", func(c Canvas) {
		c.Clear(white)
		box := func() *PathSpec {
			return NewPathSpec().MoveTo(0, 0).LineTo(90, 0).LineTo(90, 90).LineTo(0, 90).Close()
		}
		for i, radius := range []float32{8, 24, 60} {
			p := Stroke(blue, 6)
			p.PathEffect = &PathEffectSpec{Kind: PathEffectCorner, Radius: radius}
			c.Save()
			c.Translate(20+float32(i)*76, 24)
			c.Scale(0.8, 0.8)
			c.DrawPath(box(), p)
			c.Restore()
		}
		// Corner effect on a filled star: fills go through the same path transformation.
		star := NewPathSpec().MoveTo(128, 130).LineTo(180, 240).LineTo(70, 170).LineTo(186, 170).LineTo(76, 240).Close()
		pf := Fill(purple)
		pf.PathEffect = &PathEffectSpec{Kind: PathEffectCorner, Radius: 18}
		c.DrawPath(star, pf)
	})

	reg("patheffect-discrete", func(c Canvas) {
		// Skia's discrete effect is seeded (Skia's LCGRandom over seedAssist), so both engines produce the identical
		// jitter — this is a determinism check as much as a rendering one.
		c.Clear(white)
		for i := range 5 {
			p := Stroke(green, 3)
			p.PathEffect = &PathEffectSpec{
				Kind: PathEffectDiscrete, SegLength: 12,
				Deviation: 4 + float32(i), Seed: uint32(i),
			}
			y := 30 + float32(i)*36
			c.DrawLine(16, y, 240, y, p)
		}
		p := Stroke(black, 2)
		p.PathEffect = &PathEffectSpec{Kind: PathEffectDiscrete, SegLength: 8, Deviation: 6, Seed: 99}
		c.DrawPath(NewPathSpec().AddCircle(128, 210, 38, geom.DirectionCW), p)
	})

	reg("patheffect-trim", func(c Canvas) {
		c.Clear(white)
		loop := func() *PathSpec {
			return NewPathSpec().MoveTo(20, 60).CubicTo(90, -30, 166, 150, 236, 60)
		}
		p := Stroke(red, 8)
		p.Cap = CapRound
		p.PathEffect = &PathEffectSpec{Kind: PathEffectTrim, Start: 0.15, Stop: 0.70, Trim: TrimNormal}
		c.DrawPath(loop(), p)
		p2 := Stroke(blue, 8)
		p2.Cap = CapRound
		p2.PathEffect = &PathEffectSpec{Kind: PathEffectTrim, Start: 0.15, Stop: 0.70, Trim: TrimInverted}
		c.Save()
		c.Translate(0, 90)
		c.DrawPath(loop(), p2)
		c.Restore()
		p3 := Stroke(green, 6)
		p3.PathEffect = &PathEffectSpec{Kind: PathEffectTrim, Start: 0.25, Stop: 1, Trim: TrimNormal}
		c.DrawPath(NewPathSpec().AddCircle(128, 210, 36, geom.DirectionCW), p3)
	})

	reg("patheffect-1d-path", func(c Canvas) {
		c.Clear(white)
		stamp := NewPathSpec().AddCircle(0, 0, 6, geom.DirectionCW)
		for i, style := range []Path1DStyle{Path1DTranslate, Path1DRotate, Path1DMorph} {
			p := Fill(purple)
			p.PathEffect = &PathEffectSpec{
				Kind: PathEffect1DPath, Path: stamp, Advance: 22,
				Phase: 6, Style1D: style,
			}
			c.Save()
			c.Translate(0, float32(i)*80)
			c.DrawPath(NewPathSpec().MoveTo(20, 44).QuadTo(128, -16, 236, 44), p)
			c.Restore()
		}
	})

	reg("patheffect-2d", func(c Canvas) {
		c.Clear(white)
		// 2D line hatch over a filled circle region.
		hatch := geom.IdentityMatrix()
		hatch.SetAll(10, 0, 0, 0, 10, 0, 0, 0, 1)
		p := Fill(teal)
		p.PathEffect = &PathEffectSpec{Kind: PathEffect2DLine, Width: 2.5, Matrix: &hatch}
		c.DrawPath(NewPathSpec().AddCircle(70, 78, 56, geom.DirectionCW), p)
		// 2D path lattice: little squares tiling a rounded rect region.
		lattice := geom.IdentityMatrix()
		lattice.SetAll(18, 0, 4, 0, 18, 4, 0, 0, 1)
		tile := NewPathSpec().AddRect(geom.RectLTRB(0, 0, 9, 9), geom.DirectionCW)
		p2 := Fill(red)
		p2.PathEffect = &PathEffectSpec{Kind: PathEffect2DPath, Matrix: &lattice, Path: tile}
		c.DrawPath(NewPathSpec().AddRoundRect(geom.RectLTRB(140, 24, 240, 132), 18, 18, geom.DirectionCW), p2)
		// The same lattice under a rotated CTM.
		c.Save()
		c.Translate(128, 190)
		c.RotateRadians(0.4)
		c.DrawPath(NewPathSpec().AddRect(geom.RectLTRB(-90, -40, 90, 40), geom.DirectionCW), p2)
		c.Restore()
	})

	reg("patheffect-sum-compose", func(c Canvas) {
		c.Clear(white)
		wave := func() *PathSpec {
			return NewPathSpec().MoveTo(16, 50).CubicTo(90, -20, 166, 120, 240, 50)
		}
		// Sum: dashed stroke drawn on top of the corner-rounded stroke of the same path.
		p := Stroke(blue, 5)
		p.PathEffect = &PathEffectSpec{
			Kind: PathEffectSum,
			A:    dash([]float32{16, 10}, 0),
			B:    &PathEffectSpec{Kind: PathEffectCorner, Radius: 30},
		}
		c.DrawPath(wave(), p)
		// Compose: corner-round the path, then dash the result.
		zig := NewPathSpec().MoveTo(16, 0).LineTo(72, 60).LineTo(128, 0).LineTo(184, 60).LineTo(240, 0)
		p2 := Stroke(red, 5)
		p2.PathEffect = &PathEffectSpec{
			Kind: PathEffectCompose,
			A:    dash([]float32{14, 9}, 0),
			B:    &PathEffectSpec{Kind: PathEffectCorner, Radius: 22},
		}
		c.Save()
		c.Translate(0, 120)
		c.DrawPath(zig, p2)
		c.Restore()
	})

	reg("stroke-and-fill-shapes", func(c Canvas) {
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		p.Style = StyleStrokeAndFill
		p.StrokeWidth = 10
		p.Color = colorcore.Color(0x902E5FE0)
		c.DrawCircle(70, 70, 44, p)
		p.Color = colorcore.Color(0x90E0392E)
		p.Join = JoinRound
		c.DrawRect(geom.RectLTRB(150, 30, 230, 110), p)
		p.Color = colorcore.Color(0x902E9E44)
		c.DrawPath(NewPathSpec().MoveTo(40, 160).LineTo(120, 236).LineTo(40, 236).Close(), p)
		p.Color = colorcore.Color(0x907A2EE0)
		c.DrawRoundRect(geom.RectLTRB(150, 150, 236, 236), 20, 32, p)
	})

	reg("stroke-under-scale", func(c Canvas) {
		c.Clear(white)
		wave := func() *PathSpec {
			return NewPathSpec().MoveTo(0, 0).QuadTo(40, -34, 80, 0).QuadTo(120, 34, 160, 0)
		}
		c.Save()
		c.Translate(30, 50)
		c.Scale(1.2, 0.45) // anisotropic scale: stroke outline must be computed post-CTM
		c.DrawPath(wave(), Stroke(blue, 12))
		c.Restore()
		c.Save()
		c.Translate(30, 130)
		c.Scale(0.4, 1.6)
		c.DrawPath(wave(), Stroke(red, 12))
		c.Restore()
		c.Save()
		c.Translate(128, 210)
		c.RotateRadians(0.5)
		c.Scale(1.3, 1.3)
		c.DrawPath(NewPathSpec().AddRect(geom.RectLTRB(-50, -22, 50, 22), geom.DirectionCW), Stroke(green, 9))
		c.Restore()
	})

	reg("stroke-degenerate", func(c Canvas) {
		c.Clear(white)
		// Zero-length segments under each cap: butt draws nothing, round a dot, square a square.
		for i, cap := range []StrokeCap{CapButt, CapRound, CapSquare} {
			p := Stroke(black, 20)
			p.Cap = cap
			x := 52 + float32(i)*76
			c.DrawPath(NewPathSpec().MoveTo(x, 40).LineTo(x, 40), p)
		}
		// A closed tiny triangle vs a very wide stroke (stroke self-overlap).
		tri := NewPathSpec().MoveTo(70, 130).LineTo(90, 150).LineTo(70, 150).Close()
		c.DrawPath(tri, Stroke(red, 26))
		// Very long thin sliver rect stroked.
		c.DrawRect(geom.RectLTRB(140, 128, 236, 131), Stroke(blue, 5))
		// Stroke width exceeding the shape size.
		c.DrawCircle(80, 210, 8, Stroke(purple, 30))
		c.DrawRect(geom.RectLTRB(180, 195, 210, 225), Stroke(teal, 22))
	})
}
