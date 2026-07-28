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

// The blur mask filter — the one mask filter with a dedicated GPU lane on both backends. Covers the
// four styles, rect/rrect/circle/path geometries (each takes a different specialized blur lane), stroked geometry, and
// the respectCTM flag.

func blurMF(style BlurStyle, sigma float32) *MaskFilterSpec {
	return &MaskFilterSpec{Style: style, Sigma: sigma, RespectCTM: true}
}

func init() {
	reg("maskblur-styles", func(c Canvas) {
		c.Clear(white)
		for i, style := range []BlurStyle{BlurNormal, BlurSolid, BlurOuter, BlurInner} {
			x := 24 + float32(i%2)*120
			y := 24 + float32(i/2)*120
			p := Fill(blue)
			p.MaskFilter = blurMF(style, 6)
			c.DrawRect(geom.RectLTRB(x, y, x+88, y+88), p)
		}
	})

	reg("maskblur-geometries", func(c Canvas) {
		c.Clear(white)
		p := Fill(red)
		p.MaskFilter = blurMF(BlurNormal, 4)
		c.DrawCircle(64, 64, 40, p)
		p2 := Fill(green)
		p2.MaskFilter = blurMF(BlurNormal, 4)
		c.DrawRoundRect(geom.RectLTRB(140, 24, 232, 104), 18, 18, p2)
		p3 := Fill(purple)
		p3.MaskFilter = blurMF(BlurNormal, 5)
		star := NewPathSpec().MoveTo(64, 140).LineTo(96, 236).LineTo(16, 176).LineTo(112, 176).LineTo(32, 236).Close()
		c.DrawPath(star, p3)
		p4 := Fill(teal)
		p4.MaskFilter = blurMF(BlurNormal, 3)
		c.DrawOval(geom.RectLTRB(140, 150, 240, 230), p4)
	})

	reg("maskblur-stroked", func(c Canvas) {
		c.Clear(white)
		p := Stroke(blue, 8)
		p.MaskFilter = blurMF(BlurNormal, 3.5)
		c.DrawCircle(80, 80, 50, p)
		p2 := Stroke(red, 6)
		p2.Cap = CapRound
		p2.MaskFilter = blurMF(BlurNormal, 2.5)
		c.DrawPath(NewPathSpec().MoveTo(150, 30).CubicTo(210, 60, 150, 110, 220, 140), p2)
		p3 := Stroke(black, 12)
		p3.MaskFilter = blurMF(BlurSolid, 4)
		c.DrawLine(30, 200, 226, 220, p3)
	})

	reg("maskblur-ctm", func(c Canvas) {
		c.Clear(white)
		// respectCTM=true: device sigma scales with the CTM; false: sigma is fixed in device space.
		c.Save()
		c.Translate(20, 20)
		c.Scale(2.4, 2.4)
		p := Fill(colorcore.Color(0xFF2E5FE0))
		p.MaskFilter = &MaskFilterSpec{Style: BlurNormal, Sigma: 3, RespectCTM: true}
		c.DrawRect(geom.RectLTRB(0, 0, 36, 36), p)
		c.Restore()
		c.Save()
		c.Translate(140, 20)
		c.Scale(2.4, 2.4)
		p2 := Fill(colorcore.Color(0xFFE0392E))
		p2.MaskFilter = &MaskFilterSpec{Style: BlurNormal, Sigma: 3, RespectCTM: false}
		c.DrawRect(geom.RectLTRB(0, 0, 36, 36), p2)
		c.Restore()
		// Under rotation.
		c.Save()
		c.Translate(128, 190)
		c.RotateRadians(0.6)
		p3 := Fill(colorcore.Color(0xFF2E9E44))
		p3.MaskFilter = blurMF(BlurNormal, 5)
		c.DrawRect(geom.RectLTRB(-70, -34, 70, 34), p3)
		c.Restore()
	})

	reg("maskblur-translucent-overlap", func(c Canvas) {
		c.Clear(white)
		// Translucent blurred shapes overlapping: blur output composites through SrcOver, so the coverage-to-alpha
		// conversion order matters.
		p := Fill(colorcore.Color(0x90E0392E))
		p.MaskFilter = blurMF(BlurNormal, 7)
		c.DrawCircle(104, 118, 62, p)
		p2 := Fill(colorcore.Color(0x902E5FE0))
		p2.MaskFilter = blurMF(BlurNormal, 7)
		c.DrawCircle(152, 138, 62, p2)
		p3 := Fill(colorcore.Color(0x882E9E44))
		p3.MaskFilter = blurMF(BlurNormal, 10)
		c.DrawRect(geom.RectLTRB(48, 48, 208, 208), p3)
	})
}
