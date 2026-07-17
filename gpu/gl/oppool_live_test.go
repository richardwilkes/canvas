// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Cross-frame render-level guard for the op free lists (oppool.go). Op shells are recycled at flush and reused by later
// frames, so a bug in the recycle/reset protocol would surface as one frame's stale state bleeding into a later frame
// that reused its shell — precisely the failure mode single-frame render tests cannot see. This test renders a
// reference scene, then repeatedly renders a different pooled-op-heavy scene (which fills the pools and recycles them)
// followed by the reference scene again, on the *same* context, and asserts every reference render is byte-identical to
// the first. Skips when no GL context is available.

package gl_test

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu/gl"
)

// drawOpPoolChurnScene draws a batch of primitives that all lower to pooled ops (circularRRectOp, aaStrokeRectOp,
// circleOp, ellipseOp), with per-iteration varying geometry/color so many distinct op shells are created, merged, and
// recycled — maximizing the chance that the following reference render reuses a shell that last held one of these ops.
func drawOpPoolChurnScene(c *canvas.Canvas) {
	bg := canvas.NewPaint()
	bg.Color = 0xFF101820
	c.DrawPaint(bg)

	fill := canvas.NewPaint()
	fill.AntiAlias = true
	border := canvas.NewPaint()
	border.AntiAlias = true
	border.Style = canvas.StyleStroke
	border.StrokeWidth = 2
	border.Color = 0xFF80C0FF

	for i := 0; i < 10; i++ {
		f := float32(i)
		fill.Color = colorcore.Color(0xFF003000 | uint32((i*23)&0xFF)<<16 | uint32((i*47)&0xFF)<<8)
		c.DrawRoundRect(geom.RectLTRB(4+f*2, 4+f*2, 58+f, 34+f*2), 6, 6, fill) // circularRRectOp
		c.DrawRect(geom.RectLTRB(6+f*3, 60, 46+f*3, 96), border)               // aaStrokeRectOp
		c.DrawCircle(14+f*10, 112, 6, fill)                                    // circleOp
		c.DrawOval(geom.RectLTRB(66+f, 6+f*2, 104+f, 24+f*2), fill)            // ellipse/circular op
	}
}

// TestOpPoolCrossFrameStability drives many recycle/reuse cycles on one context and requires the reference scene to
// render identically every time.
func TestOpPoolCrossFrameStability(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const side = 128
	img := newCheckerImage(t)
	sdc := newLiveDrawSDC(t, dc, side, side)
	defer sdc.Release()
	c := canvas.New(gl.NewDevice(sdc))

	render := func(scene func()) []byte {
		scene()
		dc.FlushAndSubmit(false)
		return readSDC(t, sdc)
	}

	ref := render(func() { sceneCanvas(c, img) })
	for i := 0; i < 8; i++ {
		render(func() { drawOpPoolChurnScene(c) })
		again := render(func() { sceneCanvas(c, img) })
		if !bytes.Equal(ref, again) {
			t.Fatalf("reference scene render #%d differs from the first after a churn frame in "+
				"between — a recycled op shell carried stale state into a reused frame", i+1)
		}
	}
}
