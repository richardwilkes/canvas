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
	"github.com/richardwilkes/canvas/path"
)

// The seed corpus: geometry-only scenarios (no text, no images) sized 256x256, exercising the drawing operations the
// library must reproduce. Names are stable identifiers for goldens.

const (
	red    = colorcore.Color(0xFFE0392E)
	green  = colorcore.Color(0xFF2E9E44)
	blue   = colorcore.Color(0xFF2E5FE0)
	yellow = colorcore.Color(0xFFE0C42E)
	purple = colorcore.Color(0xFF7A2EE0)
	teal   = colorcore.Color(0xFF2EC9C9)
	white  = colorcore.Color(0xFFFFFFFF)
	black  = colorcore.Color(0xFF000000)
	gray   = colorcore.Color(0xFF808080)
)

func reg(name string, draw func(c Canvas)) {
	Register(Scenario{Name: name, Width: 256, Height: 256, Draw: draw})
}

func init() {
	reg("clear-color", func(c Canvas) {
		c.Clear(teal)
		c.DrawColor(colorcore.Color(0x80E0392E), BlendSrcOver) // translucent red over teal
	})

	reg("fill-rects-aliased", func(c Canvas) {
		c.Clear(white)
		p := Fill(red)
		p.AA = false
		c.DrawRect(geom.RectLTRB(10, 10, 120, 120), p)
		p.Color = blue
		c.DrawRect(geom.RectLTRB(60.5, 60.5, 200.25, 170.75), p) // fractional, aliased: hard rounding
		p.Color = colorcore.Color(0x8033CC55)
		c.DrawRect(geom.RectLTRB(100, 130, 240, 245), p) // translucent over both
	})

	reg("fill-rects-aa", func(c Canvas) {
		c.Clear(white)
		c.DrawRect(geom.RectLTRB(10.5, 10.25, 120.75, 120.5), Fill(red))
		c.DrawRect(geom.RectLTRB(60.125, 60.875, 200.5, 170.0625), Fill(colorcore.Color(0xA02E5FE0)))
		c.DrawRect(geom.RectLTRB(130, 180.5, 130.75, 250), Fill(black)) // sub-pixel-wide sliver
	})

	reg("fill-ovals", func(c Canvas) {
		c.Clear(white)
		c.DrawOval(geom.RectLTRB(10, 10, 246, 130), Fill(green))
		c.DrawCircle(128, 180, 60.5, Fill(colorcore.Color(0x90E0392E)))
		c.DrawOval(geom.RectLTRB(200, 150, 210, 250), Fill(blue)) // thin oval
	})

	reg("round-rects", func(c Canvas) {
		c.Clear(white)
		c.DrawRoundRect(geom.RectLTRB(10, 10, 246, 100), 20, 30, Fill(purple))
		c.DrawRoundRect(geom.RectLTRB(20, 120, 120, 240), 50, 60, Fill(red)) // radii clamp to oval
		c.DrawRoundRect(geom.RectLTRB(140, 120, 240, 240), 8, 8, Stroke(black, 4))
	})

	reg("arcs", func(c Canvas) {
		c.Clear(white)
		c.DrawArc(geom.RectLTRB(10, 10, 120, 120), 30, 270, false, Fill(teal))
		c.DrawArc(geom.RectLTRB(130, 10, 246, 120), 30, 270, true, Fill(red)) // wedge
		c.DrawArc(geom.RectLTRB(10, 140, 120, 245), -45, 200, false, Stroke(blue, 6))
		c.DrawArc(geom.RectLTRB(130, 140, 246, 245), 90, 359.9, true, Stroke(green, 3))
	})

	reg("line-caps", func(c Canvas) {
		c.Clear(white)
		for i, cap := range []StrokeCap{CapButt, CapRound, CapSquare} {
			p := Stroke(black, 16)
			p.Cap = cap
			y := 40 + float32(i)*40
			c.DrawLine(30, y, 226, y, p)
		}
		hair := Stroke(red, 0) // hairline
		c.DrawLine(30, 170.5, 226, 190.25, hair)
		thin := Stroke(blue, 0.5)
		c.DrawLine(30, 200.5, 226, 230.75, thin)
	})

	reg("stroke-joins", func(c Canvas) {
		c.Clear(white)
		zig := func() *PathSpec {
			return NewPathSpec().MoveTo(0, 40).LineTo(30, 0).LineTo(60, 40).LineTo(90, 5)
		}
		for i, join := range []StrokeJoin{JoinMiter, JoinRound, JoinBevel} {
			p := Stroke(blue, 12)
			p.Join = join
			c.Save()
			c.Translate(20, 20+float32(i)*75)
			c.DrawPath(zig(), p)
			c.Restore()
		}
		// Miter-limit sweep on a sharp point: limit below/above the needed ratio.
		sharp := NewPathSpec().MoveTo(140, 40).LineTo(200, 20).LineTo(140, 0)
		low := Stroke(red, 10)
		low.StrokeMiter = 1.5
		c.Save()
		c.Translate(20, 30)
		c.DrawPath(sharp, low)
		c.Restore()
		high := Stroke(green, 10)
		high.StrokeMiter = 10
		c.Save()
		c.Translate(20, 110)
		c.DrawPath(sharp, high)
		c.Restore()
	})

	reg("stroke-curves", func(c Canvas) {
		c.Clear(white)
		spec := NewPathSpec().
			MoveTo(20, 60).QuadTo(128, -40, 236, 60).
			MoveTo(20, 130).CubicTo(80, 40, 176, 220, 236, 130).
			MoveTo(20, 200).ConicTo(128, 120, 236, 200, 0.7071)
		c.DrawPath(spec, Stroke(purple, 8))
		c.DrawPath(NewPathSpec().MoveTo(20, 240).QuadTo(128, 180, 236, 240), Stroke(black, 0))
	})

	reg("path-fill-winding-vs-evenodd", func(c Canvas) {
		c.Clear(white)
		donut := func() *PathSpec {
			return NewPathSpec().
				AddCircle(64, 128, 55, geom.DirectionCW).
				AddCircle(64, 128, 25, geom.DirectionCW) // same direction: winding keeps it solid
		}
		c.DrawPath(donut(), Fill(red))
		d2 := donut().SetFill(path.FillEvenOdd)
		spec := NewPathSpec().AddPathOffset(d2, 128, 0, path.AddPathAppend).SetFill(path.FillEvenOdd)
		c.DrawPath(spec, Fill(blue))
	})

	reg("path-inverse-fill", func(c Canvas) {
		c.Clear(white)
		spec := NewPathSpec().AddCircle(128, 128, 70, geom.DirectionCW).SetFill(path.FillInverseWinding)
		c.DrawPath(spec, Fill(colorcore.Color(0xFF2E9E44)))
	})

	reg("path-star-selfintersect", func(c Canvas) {
		c.Clear(white)
		star := NewPathSpec()
		const n = 5
		for i := range n {
			angle := float64(i*2%n) * 2 * math.Pi / n
			x := 128 + 100*float32(math.Sin(angle))
			y := 128 - 100*float32(math.Cos(angle))
			if i == 0 {
				star.MoveTo(x, y)
			} else {
				star.LineTo(x, y)
			}
		}
		star.Close()
		c.DrawPath(star, Fill(yellow))
		even := *star
		even.SetFill(path.FillEvenOdd)
		c.Save()
		c.Scale(0.4, 0.4)
		c.DrawPath(&even, Fill(purple))
		c.Restore()
	})

	// The SVG equivalent of this path is "M30 100 A40 40 0 0 1 110 100 A40 40 0 0 1 190 100 Q210 130 190 160 L30 160
	// Z". SVG's sweep-flag 1 maps to DirectionCW (the flag is inverted relative to the enum) and its large-arc-flag 0
	// to ArcSizeSmall.
	reg("path-arcs-svg", func(c Canvas) {
		c.Clear(white)
		spec := NewPathSpec().MoveTo(30, 100).
			ArcToRotated(40, 40, 0, path.ArcSizeSmall, geom.DirectionCW, 110, 100).
			ArcToRotated(40, 40, 0, path.ArcSizeSmall, geom.DirectionCW, 190, 100).
			QuadTo(210, 130, 190, 160).
			LineTo(30, 160).
			Close()
		c.DrawPath(spec, Fill(teal))
		tangent := NewPathSpec().MoveTo(30, 200).ArcToTangent(128, 180, 226, 240, 30).LineTo(226, 240)
		c.DrawPath(tangent, Stroke(red, 4))
	})

	reg("transforms", func(c Canvas) {
		c.Clear(white)
		r := geom.RectLTRB(-40, -20, 40, 20)
		c.Save()
		c.Translate(64, 64)
		c.DrawRect(r, Fill(red))
		c.Restore()
		c.Save()
		c.Translate(192, 64)
		c.RotateRadians(float32(math.Pi) / 6)
		c.DrawRect(r, Fill(green))
		c.Restore()
		c.Save()
		c.Translate(64, 192)
		c.Scale(1.5, 0.75)
		c.DrawRect(r, Fill(blue))
		c.Restore()
		c.Save()
		c.Translate(192, 192)
		c.Skew(0.4, 0.1)
		c.DrawRect(r, Fill(purple))
		c.Restore()
	})

	reg("transform-perspective", func(c Canvas) {
		c.Clear(white)
		m := geom.IdentityMatrix()
		m.SetAll(1, 0.1, 10, 0.05, 1, 20, 0.0005, 0.0008, 1)
		c.Concat(&m)
		c.DrawRect(geom.RectLTRB(20, 20, 200, 200), Fill(colorcore.Color(0xC02E5FE0)))
		c.DrawCircle(110, 110, 60, Stroke(black, 5))
	})

	reg("clip-rect", func(c Canvas) {
		c.Clear(white)
		c.Save()
		c.ClipRect(geom.RectLTRB(40.5, 40.5, 215.5, 215.5), ClipIntersect, true)
		c.DrawCircle(128, 128, 110, Fill(red))
		c.Restore()
		c.Save()
		c.ClipRect(geom.RectLTRB(90, 90, 166, 166), ClipDifference, false)
		c.DrawCircle(128, 128, 55, Fill(blue))
		c.Restore()
	})

	reg("clip-path", func(c Canvas) {
		c.Clear(white)
		c.Save()
		c.ClipPath(NewPathSpec().AddCircle(128, 128, 90, geom.DirectionCW), ClipIntersect, true)
		for i := range 8 {
			y := float32(i) * 32
			c.DrawRect(geom.RectLTRB(0, y, 256, y+16), Fill(teal))
		}
		c.Restore()
	})

	reg("save-restore-nesting", func(c Canvas) {
		c.Clear(white)
		p := Fill(colorcore.Color(0x882E5FE0))
		r := geom.RectLTRB(-30, -30, 30, 30)
		c.Translate(128, 128)
		for range 6 {
			c.Save()
		}
		for range 6 {
			c.Restore()
			c.RotateRadians(float32(math.Pi) / 7)
			c.Scale(1.12, 1.12)
			c.DrawRect(r, p)
		}
	})

	reg("save-layer-alpha", func(c Canvas) {
		c.Clear(white)
		c.DrawRect(geom.RectLTRB(0, 100, 256, 156), Fill(gray))
		c.SaveLayerAlpha(nil, 128)
		c.DrawCircle(100, 128, 70, Fill(red))
		c.DrawCircle(156, 128, 70, Fill(blue)) // overlaps red inside the layer, then 50% composite
		c.Restore()
	})

	reg("blend-modes-basic", func(c Canvas) {
		c.Clear(white)
		modes := []BlendMode{
			BlendSrcOver, BlendMultiply, BlendScreen, BlendPlus,
			BlendXor, BlendDstOver, BlendDifference, BlendOverlay,
		}
		for i, mode := range modes {
			x := float32(i%4) * 64
			y := float32(i/4) * 128
			c.Save()
			c.ClipRect(geom.RectLTRB(x, y, x+64, y+128), ClipIntersect, false)
			dst := Fill(colorcore.Color(0xD0E0C42E))
			c.DrawRect(geom.RectLTRB(x+4, y+8, x+48, y+88), dst)
			src := Fill(colorcore.Color(0xB02E5FE0))
			src.Blend = mode
			c.DrawCircle(x+40, y+72, 34, src)
			c.Restore()
		}
	})

	reg("gradient-linear", func(c Canvas) {
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 20, Y: 20}, {X: 236, Y: 120}},
			Colors: []colorcore.Color{red, yellow, blue},
			Pos:    []float32{0, 0.3, 1},
			Tile:   TileClamp,
		}
		c.DrawRect(geom.RectLTRB(10, 10, 246, 130), p)
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 150}, {X: 64, Y: 182}},
			Colors: []colorcore.Color{green, purple},
			Tile:   TileMirror,
		}
		c.DrawRect(geom.RectLTRB(10, 140, 246, 246), p)
	})

	reg("gradient-radial", func(c Canvas) {
		c.Clear(white)
		p := NewPaint()
		p.AA = true
		p.Shader = &ShaderSpec{
			Kind:   ShaderRadialGradient,
			Center: geom.Point{X: 128, Y: 128},
			Radius: 100,
			Colors: []colorcore.Color{white, teal, black},
			Pos:    []float32{0, 0.7, 1},
			Tile:   TileClamp,
		}
		c.DrawCircle(128, 128, 100, p)
		p.Shader = &ShaderSpec{
			Kind:   ShaderRadialGradient,
			Center: geom.Point{X: 40, Y: 216},
			Radius: 25,
			Colors: []colorcore.Color{red, yellow},
			Tile:   TileRepeat,
		}
		c.DrawRect(geom.RectLTRB(0, 176, 256, 256), p)
	})

	reg("draw-points", func(c Canvas) {
		c.Clear(white)
		pts := []geom.Point{{X: 30, Y: 40}, {X: 90, Y: 20}, {X: 150, Y: 60}, {X: 226, Y: 30}}
		p := Stroke(red, 10)
		p.Cap = CapRound
		c.DrawPoints(PointModePoints, pts, p)
		pts2 := make([]geom.Point, len(pts))
		for i, pt := range pts {
			pts2[i] = geom.Point{X: pt.X, Y: pt.Y + 90}
		}
		p.Color = green
		p.Cap = CapButt
		c.DrawPoints(PointModeLines, pts2, p)
		pts3 := make([]geom.Point, len(pts))
		for i, pt := range pts {
			pts3[i] = geom.Point{X: pt.X, Y: pt.Y + 180}
		}
		p.Color = blue
		p.Join = JoinRound
		c.DrawPoints(PointModePolygon, pts3, p)
	})

	reg("draw-paint-shader", func(c Canvas) {
		p := NewPaint()
		p.Shader = &ShaderSpec{
			Kind:   ShaderLinearGradient,
			Pts:    [2]geom.Point{{X: 0, Y: 0}, {X: 256, Y: 256}},
			Colors: []colorcore.Color{purple, teal, yellow},
			Tile:   TileClamp,
		}
		c.DrawPaint(p)
	})
}
