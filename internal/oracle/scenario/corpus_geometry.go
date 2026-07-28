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

// Extra geometry coverage for the operations the corpus already draws — hairline AA/non-AA splits,
// rotated ovals and round rects (each leaves its axis-aligned specialized op), degenerate arcs, inverse fills under
// clips, and point-mode edge cases.

func init() {
	reg("hairlines", func(c Canvas) {
		c.Clear(white)
		// AA hairlines at shallow angles (coverage-critical), then the same aliased.
		for i := range 5 {
			y := 14 + float32(i)*10
			c.DrawLine(8, y, 248, y+3.5, Stroke(black, 0))
		}
		for i := range 5 {
			y := 78 + float32(i)*10
			p := Stroke(black, 0)
			p.AA = false
			c.DrawLine(8, y, 248, y+3.5, p)
		}
		// Hairline curves and a hairline closed shape.
		c.DrawPath(NewPathSpec().MoveTo(8, 150).CubicTo(70, 120, 180, 190, 248, 140), Stroke(red, 0))
		c.DrawPath(NewPathSpec().AddCircle(70, 205, 36, geom.DirectionCW), Stroke(blue, 0))
		hp := Stroke(green, 0)
		hp.AA = false
		c.DrawPath(NewPathSpec().AddCircle(180, 205, 36, geom.DirectionCW), hp)
	})

	reg("ovals-rotated", func(c Canvas) {
		c.Clear(white)
		// Rotated ovals leave the axis-aligned oval fast path on both backends.
		for i := range 4 {
			c.Save()
			c.Translate(64+float32(i%2)*128, 64+float32(i/2)*128)
			c.RotateRadians(float32(i+1) * 0.35)
			c.DrawOval(geom.RectLTRB(-52, -28, 52, 28), Fill([]colorcore.Color{red, green, blue, purple}[i]))
			c.Restore()
		}
		c.Save()
		c.Translate(128, 128)
		c.RotateRadians(0.8)
		c.DrawOval(geom.RectLTRB(-40, -18, 40, 18), Stroke(black, 5))
		c.Restore()
	})

	reg("rrects-varied", func(c Canvas) {
		c.Clear(white)
		// Radii combinations: near-oval, tiny radius, radius clamping, stroked, rotated.
		c.DrawRoundRect(geom.RectLTRB(12, 12, 120, 84), 50, 34, Fill(teal)) // radii clamp to half-extent
		c.DrawRoundRect(geom.RectLTRB(136, 12, 244, 84), 2, 2, Fill(red))
		c.DrawRoundRect(geom.RectLTRB(12, 100, 120, 156), 12, 26, Fill(purple))
		c.DrawRoundRect(geom.RectLTRB(136, 100, 244, 156), 18, 18, Stroke(blue, 7))
		c.Save()
		c.Translate(64, 210)
		c.RotateRadians(0.5)
		c.DrawRoundRect(geom.RectLTRB(-48, -28, 48, 28), 14, 14, Fill(green))
		c.Restore()
		c.Save()
		c.Translate(190, 210)
		c.Skew(0.3, 0)
		c.DrawRoundRect(geom.RectLTRB(-44, -26, 44, 26), 12, 20, Stroke(black, 4))
		c.Restore()
	})

	reg("arcs-degenerate", func(c Canvas) {
		c.Clear(white)
		// Tiny sweeps, >360 sweeps clamp, negative sweeps, near-zero ovals.
		c.DrawArc(geom.RectLTRB(12, 12, 116, 116), 0, 4, true, Fill(red))
		c.DrawArc(geom.RectLTRB(12, 12, 116, 116), 20, -120, true, Fill(blue))
		c.DrawArc(geom.RectLTRB(140, 12, 244, 116), 45, 720, false, Stroke(green, 5))
		c.DrawArc(geom.RectLTRB(140, 12, 244, 116), -30, -300, true, Stroke(black, 2))
		c.DrawArc(geom.RectLTRB(12, 140, 244, 156), 0, 270, false, Stroke(purple, 4)) // sliver oval
		c.DrawArc(geom.RectLTRB(60, 170, 200, 250), 90, 180, false, Fill(teal))       // no center, filled chord
	})

	reg("points-modes-caps", func(c Canvas) {
		c.Clear(white)
		pts := []geom.Point{{X: 30, Y: 30}, {X: 226, Y: 44}, {X: 60, Y: 70}, {X: 200, Y: 90}}
		// Points mode with square caps: each point becomes an axis-aligned square.
		p := Stroke(blue, 14)
		p.Cap = CapSquare
		c.DrawPoints(PointModePoints, pts, p)
		// Points mode, butt cap: Skia draws nothing for zero-length butt segments... except points mode promotes them
		// to squares? Both engines must agree — keep AA on.
		pts2 := make([]geom.Point, len(pts))
		for i, pt := range pts {
			pts2[i] = geom.Point{X: pt.X, Y: pt.Y + 80}
		}
		p2 := Stroke(red, 14)
		p2.Cap = CapButt
		c.DrawPoints(PointModePoints, pts2, p2)
		// Lines mode with an odd point count (last point ignored) and round caps.
		pts3 := []geom.Point{{X: 30, Y: 190}, {X: 120, Y: 210}, {X: 150, Y: 190}, {X: 226, Y: 230}, {X: 240, Y: 190}}
		p3 := Stroke(green, 9)
		p3.Cap = CapRound
		c.DrawPoints(PointModeLines, pts3, p3)
		// Polygon mode with a hairline.
		pts4 := []geom.Point{{X: 30, Y: 246}, {X: 90, Y: 236}, {X: 150, Y: 250}, {X: 226, Y: 240}}
		c.DrawPoints(PointModePolygon, pts4, Stroke(black, 0))
	})

	reg("path-conics-weights", func(c Canvas) {
		c.Clear(white)
		// Conic weight sweep: w<1 (elliptical), w=1 (parabolic), w>1 (hyperbolic).
		for i, w := range []float32{0.3, 0.7071, 1, 2, 8} {
			y := 24 + float32(i)*44
			c.DrawPath(NewPathSpec().MoveTo(16, y+30).ConicTo(128, y-24, 240, y+30, w), Stroke(blue, 4))
		}
	})

	reg("path-inverse-clipped", func(c Canvas) {
		c.Clear(white)
		// Inverse fills interacting with clips: the inverse region must respect the clip stack.
		c.Save()
		c.ClipRect(geom.RectLTRB(16, 16, 240, 120), ClipIntersect, true)
		spec := NewPathSpec().AddCircle(80, 68, 40, geom.DirectionCW).SetFill(path.FillInverseWinding)
		c.DrawPath(spec, Fill(teal))
		c.Restore()
		c.Save()
		c.ClipPath(NewPathSpec().AddCircle(128, 190, 58, geom.DirectionCW), ClipIntersect, true)
		spec2 := NewPathSpec().
			AddRect(geom.RectLTRB(100, 160, 156, 220), geom.DirectionCW).
			SetFill(path.FillInverseEvenOdd)
		c.DrawPath(spec2, Fill(colorcore.Color(0xC0E0392E)))
		c.Restore()
	})

	reg("transform-rotate-sweep", func(c Canvas) {
		c.Clear(white)
		// A dial of rotated copies: accumulates CTM precision over many small rotations.
		c.Translate(128, 128)
		p := Fill(colorcore.Color(0x702E5FE0))
		for range 24 {
			c.RotateRadians(float32(math.Pi) / 12)
			c.DrawRect(geom.RectLTRB(46, -7, 116, 7), p)
		}
		c.DrawCircle(0, 0, 30, Stroke(black, 3))
	})
}
