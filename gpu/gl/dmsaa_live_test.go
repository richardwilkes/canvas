// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context test for the pixel-level consequence of the shared stencil attachment: under dynamic MSAA every
// same-shaped render target is handed one shared stencil renderbuffer, so what a finished draw context left in it must
// never be able to reach the next one. Skips when no GL context is available.

package gl_test

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/patheffect"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/surface"
)

const (
	dmsaaResidueSize = 256
	// dmsaaResidueStencilBits is the stencil depth of the offscreen FBO's packed depth-stencil renderbuffer — the FBO
	// shape a windowing embedder hands the library.
	dmsaaResidueStencilBits = 8
)

// renderLiveDMSAAFrame renders draw into a fresh caller-owned FBO wrapped as a dynamic-MSAA surface — the production
// wrapped-FBO path — and returns the read-back RGBA8888 pixels. Each call builds and tears down its own render target,
// so consecutive calls are exactly the "one render after another in the same process" shape the shared stencil
// attachment spans.
func renderLiveDMSAAFrame(t *testing.T, env *glEnv, dc *gl.DirectContext, draw func(*canvas.Canvas)) []byte {
	t.Helper()
	const size = int32(dmsaaResidueSize)
	f := &env.intf.Functions

	var colorRB uint32
	f.GenRenderbuffers(1, &colorRB)
	f.BindRenderbuffer(gl.RENDERBUFFER, colorRB)
	f.RenderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, size, size)
	var fbo uint32
	f.GenFramebuffers(1, &fbo)
	f.BindFramebuffer(gl.FRAMEBUFFER, fbo)
	f.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRB)
	var dsRB uint32
	f.GenRenderbuffers(1, &dsRB)
	f.BindRenderbuffer(gl.RENDERBUFFER, dsRB)
	f.RenderbufferStorage(gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, size, size)
	f.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, dsRB)
	if status := f.CheckFramebufferStatus(gl.FRAMEBUFFER); status != gl.FRAMEBUFFER_COMPLETE {
		t.Fatalf("framebuffer incomplete: 0x%04X", status)
	}
	f.Viewport(0, 0, size, size)
	f.ClearColor(0, 0, 0, 0)
	f.ClearStencil(0)
	f.Clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
	// The FBO was created and cleared behind the context's back.
	dc.ResetContext(gl.AllBackendState)

	rt := gl.NewRenderTargetSurfaceFromBackendRenderTarget(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: size, Height: size}, gl.FormatFromEnum(gl.RGBA8), 0,
		dmsaaResidueStencilBits, fbo, gpu.OriginTopLeft,
		&surface.Props{Flags: surface.DynamicMSAAFlag, PixelGeometry: surface.PixelGeometryUnknown})
	if rt == nil {
		t.Fatal("NewRenderTargetSurfaceFromBackendRenderTarget returned nil")
	}
	if !rt.SDC().CanUseDynamicMSAA() {
		t.Skip("this GL stack cannot supply a dynamic MSAA attachment")
	}

	draw(rt.Canvas())
	dc.FlushAndSubmit(false)

	info, _ := imagecore.MakeInfo(size, size, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	buf := make([]byte, int(size)*int(size)*4)
	if !rt.ReadPixels(info, buf, int(size)*4, 0, 0) {
		t.Fatal("wrapped-FBO ReadPixels failed")
	}

	rt.SDC().Release()
	f.BindFramebuffer(gl.FRAMEBUFFER, 0)
	f.DeleteFramebuffers(1, &fbo)
	f.DeleteRenderbuffers(1, &colorRB)
	f.DeleteRenderbuffers(1, &dsRB)
	return buf
}

// drawDashedContours is the victim frame: dashed strokes produce many short contours, which the path renderer fills by
// stenciling the winding into the user stencil bits and then covering. The cover pass tests those bits, so it renders
// whatever residue the render pass started with as stray geometry.
func drawDashedContours(c *canvas.Canvas) {
	c.Clear(0xFFFFFFFF)
	rect := path.New()
	rect.AddRect(geom.RectLTRB(24, 24, 120, 120), geom.DirectionCW)
	c.DrawPath(rect, dashPaint(0xFF008000, 6, []float32{18, 10}, 4))
	circle := path.New()
	circle.AddCircle(184, 72, 52, geom.DirectionCW)
	c.DrawPath(circle, dashPaint(0xFFFF0000, 6, []float32{14, 8}, 0))
	rrect := path.New()
	rrect.AddRoundRect(geom.RectLTRB(40, 150, 216, 236), 24, 24, geom.DirectionCW)
	c.DrawPath(rrect, dashPaint(0xFF0000FF, 4, []float32{12, 6, 4, 6}, 0))
}

func dashPaint(color colorcore.Color, width float32, intervals []float32, phase float32) *canvas.Paint {
	p := canvas.NewPaint()
	p.Color = color
	p.Style = canvas.StyleStroke
	p.StrokeWidth = width
	p.AntiAlias = true
	p.PathEffect = patheffect.MakeDash(intervals, phase)
	return p
}

// drawEvenOddRingClip is the dirtying frame: an even-odd, multi-contour clip path is too complex for an analytic
// coverage FP, so on a dynamic-MSAA target it is rasterized into the shared stencil buffer — leaving both clip-bit and
// user-bit residue behind once the frame is done.
func drawEvenOddRingClip(c *canvas.Canvas) {
	c.Clear(0xFFFFFFFF)
	rings := path.New()
	rings.AddCircle(128, 128, 100, geom.DirectionCW)
	rings.AddCircle(128, 128, 64, geom.DirectionCW)
	rings.AddCircle(128, 128, 30, geom.DirectionCW)
	rings.SetFillType(path.FillEvenOdd)
	c.ClipPath(rings, raster.ClipIntersect, true)
	bar := canvas.NewPaint()
	bar.Color = 0xFFFF0000
	bar.AntiAlias = true
	for i := range 13 {
		y := float32(i) * 20
		c.DrawRect(geom.RectLTRB(0, y, 256, y+10), bar)
	}
}

// TestLiveDMSAAStencilResidueDoesNotCrossRenderTargets renders the same dashed-contour frame twice, with a
// stencil-clipped frame in between. All three render targets are handed the same shared stencil attachment (the
// sharing key is only dimensions, format, usage and sample count), so if the middle frame's leftovers survive into the
// third render pass the dashes come back with stray bands through them.
//
// Whether the residue is actually visible is stack-dependent — it reproduces on llvmpipe and not on every hardware
// driver — so a pass here is a real result on the stacks that show it, not proof on the ones that don't.
// TestDMSAASharedStencilClearedForEveryRenderTarget pins the load op itself, deterministically, everywhere.
func TestLiveDMSAAStencilResidueDoesNotCrossRenderTargets(t *testing.T) {
	env, dc := newLiveDirectContext(t)

	// The first user of the shared attachment: nothing can have dirtied it yet, so this is the reference.
	want := renderLiveDMSAAFrame(t, env, dc, drawDashedContours)
	renderLiveDMSAAFrame(t, env, dc, drawEvenOddRingClip)
	got := renderLiveDMSAAFrame(t, env, dc, drawDashedContours)

	differing := 0
	firstX, firstY := int32(-1), int32(-1)
	for y := range int32(dmsaaResidueSize) {
		for x := range int32(dmsaaResidueSize) {
			off := (int(y)*dmsaaResidueSize + int(x)) * 4
			if !bytes.Equal(got[off:off+4], want[off:off+4]) {
				differing++
				if firstX < 0 {
					firstX, firstY = x, y
				}
			}
		}
	}
	if differing != 0 {
		off := (int(firstY)*dmsaaResidueSize + int(firstX)) * 4
		t.Fatalf("%d pixels changed after an unrelated stencil-clipped frame on another render target "+
			"sharing the stencil attachment; first at (%d,%d): got %v, want %v",
			differing, firstX, firstY, got[off:off+4], want[off:off+4])
	}
}
