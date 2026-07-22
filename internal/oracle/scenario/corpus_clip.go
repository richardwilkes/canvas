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

// Corpus growth: the clip-torture family — deep clip stacks, AA/non-AA mixes, difference stacking, path/rect
// interactions, and clips under rotated/perspective CTMs.

func init() {
	reg("clip-nested-shrink", func(c Canvas) {
		c.Clear(white)
		// Ten nested intersect clips, alternating AA rects and AA circles, then one big fill.
		for i := range 10 {
			inset := float32(i) * 11.5
			if i%2 == 0 {
				c.ClipRect(geom.RectLTRB(4+inset, 4+inset, 252-inset, 252-inset), ClipIntersect, i%4 == 0)
			} else {
				c.ClipPath(NewPathSpec().AddCircle(128, 128, 172-inset*1.4, geom.DirectionCW),
					ClipIntersect, true)
			}
		}
		c.DrawPaint(Fill(teal))
	})

	reg("clip-difference-grid", func(c Canvas) {
		c.Clear(white)
		// Punch a 4x4 grid of holes out of the clip, then fill: only the grout should paint.
		for row := range 4 {
			for col := range 4 {
				x := 20 + float32(col)*56
				y := 20 + float32(row)*56
				c.ClipRect(geom.RectLTRB(x, y, x+44, y+44), ClipDifference, (row+col)%2 == 0)
			}
		}
		c.DrawPaint(Fill(purple))
	})

	reg("clip-path-evenodd", func(c Canvas) {
		c.Clear(white)
		rings := NewPathSpec().
			AddCircle(128, 128, 100, geom.DirectionCW).
			AddCircle(128, 128, 64, geom.DirectionCW).
			AddCircle(128, 128, 30, geom.DirectionCW).
			SetFill(path.FillEvenOdd)
		c.ClipPath(rings, ClipIntersect, true)
		for i := range 13 {
			y := float32(i) * 20
			c.DrawRect(geom.RectLTRB(0, y, 256, y+10), Fill(red))
		}
	})

	reg("clip-mixed-aa", func(c Canvas) {
		c.Clear(white)
		// An AA path clip intersected with a non-AA rect clip at fractional coordinates: the two edge disciplines must
		// compose, not blur into each other.
		c.ClipPath(NewPathSpec().AddCircle(128, 128, 105.3, geom.DirectionCW), ClipIntersect, true)
		c.ClipRect(geom.RectLTRB(40.6, 40.4, 215.5, 215.7), ClipIntersect, false)
		c.DrawPaint(Fill(green))
		// A second draw through one more AA difference clip.
		c.ClipPath(NewPathSpec().AddCircle(128, 128, 55, geom.DirectionCW), ClipDifference, true)
		c.DrawPaint(Fill(colorcore.Color(0x807A2EE0)))
	})

	reg("clip-rotated-rect", func(c Canvas) {
		c.Clear(white)
		// Set a rotated CTM, clip under it, then unwind the CTM manually (no Save/Restore, which would pop the clip
		// with the matrix). The rotated clip is baked into device space and persists while the bars draw axis-aligned.
		angle := float32(math.Pi) / 5
		c.Translate(128, 128)
		c.RotateRadians(angle)
		c.ClipRect(geom.RectLTRB(-90, -90, 90, 90), ClipIntersect, true)
		c.RotateRadians(-angle) // undo the rotation (transforms pre-concatenate, so reverse order)
		c.Translate(-128, -128) // undo the translation: CTM back to identity, clip still in device space
		for i := range 8 {
			x := float32(i) * 32
			c.DrawRect(geom.RectLTRB(x, 0, x+16, 256), Fill(blue))
		}
	})

	reg("clip-tiny-slivers", func(c Canvas) {
		c.Clear(white)
		// Sub-pixel-tall clip bands: AA coverage of the band vs non-AA rounding.
		for i := range 6 {
			y := 30 + float32(i)*36
			c.Save()
			c.ClipRect(geom.RectLTRB(16, y, 240, y+0.6+float32(i)*0.35), ClipIntersect, i%2 == 0)
			c.DrawPaint(Fill(black))
			c.Restore()
		}
		// A clip that collapses to empty, then a draw that must be fully culled.
		c.Save()
		c.ClipRect(geom.RectLTRB(40, 240, 90, 250), ClipIntersect, true)
		c.ClipRect(geom.RectLTRB(150, 240, 200, 250), ClipIntersect, true) // disjoint: empty
		c.DrawPaint(Fill(red))
		c.Restore()
		c.DrawRect(geom.RectLTRB(16, 246, 240, 252), Fill(teal)) // proves state recovered after empty clip
	})

	reg("clip-save-restore-interleave", func(c Canvas) {
		c.Clear(white)
		c.ClipRect(geom.RectLTRB(10, 10, 246, 246), ClipIntersect, false)
		c.Save()
		c.ClipPath(NewPathSpec().AddCircle(90, 90, 70, geom.DirectionCW), ClipIntersect, true)
		c.DrawPaint(Fill(colorcore.Color(0xFFE0C42E)))
		c.Save()
		c.ClipRect(geom.RectLTRB(60, 60, 130, 130), ClipDifference, false)
		c.DrawPaint(Fill(colorcore.Color(0xC0E0392E)))
		c.Restore()
		c.DrawCircle(90, 90, 20, Fill(black)) // difference clip gone, circle clip back
		c.Restore()
		c.DrawRect(geom.RectLTRB(170, 170, 260, 260), Fill(colorcore.Color(0xA02E5FE0))) // only outer clip
	})

	reg("clip-persp", func(c Canvas) {
		c.Clear(white)
		m := geom.IdentityMatrix()
		m.SetAll(1, 0.05, 8, 0.02, 1, 12, 0.0006, 0.0011, 1)
		c.Concat(&m)
		c.ClipPath(NewPathSpec().AddRoundRect(geom.RectLTRB(24, 24, 216, 216), 30, 30, geom.DirectionCW),
			ClipIntersect, true)
		for i := range 12 {
			x := float32(i) * 20
			c.DrawRect(geom.RectLTRB(x, 0, x+10, 256), Fill(teal))
			c.DrawRect(geom.RectLTRB(0, x, 256, x+10), Fill(colorcore.Color(0x80E0392E)))
		}
	})

	reg("clip-stroke-interaction", func(c Canvas) {
		c.Clear(white)
		// Wide strokes crossing an AA clip edge: the stroke geometry must be clipped, not the source path.
		c.ClipPath(NewPathSpec().AddCircle(128, 128, 92, geom.DirectionCW), ClipIntersect, true)
		for i := range 7 {
			y := 32 + float32(i)*32
			c.DrawLine(0, y, 256, y+14, Stroke(blue, 11))
		}
		c.ClipRect(geom.RectLTRB(96, 0, 160, 256), ClipDifference, false)
		c.DrawPath(NewPathSpec().AddCircle(128, 128, 60, geom.DirectionCW), Stroke(red, 13))
	})
}
