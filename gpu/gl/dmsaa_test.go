// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for dynamic MSAA: a surface created with the dynamic-MSAA surface-props flag promotes path draws to
// an MSAA render pass over a lazily created multisample FBO, resolved back into the single-sample target with
// glBlitFramebuffer. The recording driver pins the exact call sequence — MSAA FBO/renderbuffer creation, the
// draw-to-MSAA pass, the store resolve, and the load resolve of a second frame — so CI's software legs cover the lane
// without hardware. The default-off test proves a plain surface never touches any of it.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/surface"
)

// newDMSAATestSDC builds a 1-sample draw context over the recording driver, with or without the DMSAA surface-props
// flag.
func newDMSAATestSDC(t *testing.T, dc *DirectContext, dynamicMSAA bool) *SurfaceDrawContext {
	t.Helper()
	var props *surface.Props
	if dynamicMSAA {
		props = &surface.Props{Flags: surface.DynamicMSAAFlag}
	}
	sdc := MakeSurfaceDrawContextWithProps(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: 32, Height: 32}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "dmsaa-test", props)
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContextWithProps failed")
	}
	return sdc
}

// drawAAStar records one AA concave-star path draw, the canonical DMSAA trigger (paths always escalate to MSAA on a
// DMSAA surface).
func drawAAStar(sdc *SurfaceDrawContext) {
	p := path.New()
	p.MoveTo(16, 2)
	p.LineTo(20, 22)
	p.LineTo(4, 9)
	p.LineTo(28, 9)
	p.LineTo(12, 22)
	p.Close()
	shape := MakeShapePath(p)
	paint := NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.75, A: 1})
	identity := geom.IdentityMatrix()
	sdc.DrawShapeWithStyle(nil, paint, gpu.AAYes, &identity, &shape, nil)
}

// TestDMSAAPathDrawCallSequence pins the DMSAA GL call sequence over the recording driver.
func TestDMSAAPathDrawCallSequence(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDMSAATestSDC(t, dc, true)
	defer sdc.Release()
	if !sdc.CanUseDynamicMSAA() {
		t.Fatal("DMSAA surface props did not enable dynamic MSAA (caps should support it)")
	}

	// Frame 1: clear (a native load-op clear) + AA path. The path must promote the ops task to the MSAA surface.
	sdc.Clear([4]float32{0, 0, 0, 1})
	drawAAStar(sdc)
	if !sdc.GetOpsTask().UsesMSAASurface() {
		t.Fatal("AA path draw did not set usesMSAASurface on a DMSAA surface")
	}

	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)

	// The lazily created DMSAA attachment: one framebuffer for the MSAA pass plus the multisampled color renderbuffer,
	// both attached, and the multisampled stencil for the concave path's stencil passes.
	if got := counts("glGenFramebuffers"); got < 1 {
		t.Errorf("glGenFramebuffers = %d, want >= 1 (the dynamic MSAA FBO)", got)
	}
	if got := counts("glRenderbufferStorageMultisample"); got != 2 {
		t.Errorf("glRenderbufferStorageMultisample = %d, want 2 (MSAA color + MSAA stencil)", got)
	}
	if counts("glFramebufferRenderbuffer") < 2 {
		t.Errorf("glFramebufferRenderbuffer = %d, want >= 2 (color + stencil attach)",
			counts("glFramebufferRenderbuffer"))
	}
	// Store resolve only: the frame began with a load-op clear, so there is no load resolve.
	if got := counts("glBlitFramebuffer"); got != 1 {
		t.Errorf("glBlitFramebuffer = %d, want exactly 1 (the MSAA->single store resolve)", got)
	}
	if counts("glDrawArrays")+counts("glDrawElements")+counts("glDrawRangeElements")+
		counts("glDrawArraysInstanced")+counts("glDrawElementsInstanced") == 0 {
		t.Error("no draws issued for the DMSAA frame")
	}

	// Frame 2: the same path with no clear. The render pass now loads the color buffer, so the single-sample content is
	// first blitted into the MSAA attachment (load resolve), then resolved back (store resolve) — exactly two blits,
	// and no new FBO/renderbuffer creation (the attachment is retained on the render target).
	drawAAStar(sdc)
	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if got := counts("glBlitFramebuffer"); got != 2 {
		t.Errorf("glBlitFramebuffer = %d, want 2 (load resolve + store resolve)", got)
	}
	if got := counts("glGenFramebuffers"); got != 0 {
		t.Errorf("glGenFramebuffers = %d, want 0 (DMSAA FBO must be reused)", got)
	}
	if got := counts("glRenderbufferStorageMultisample"); got != 0 {
		t.Errorf("glRenderbufferStorageMultisample = %d, want 0 (attachments must be reused)", got)
	}
}

// TestDMSAADefaultOffTouchesNoMSAAState proves the default-off regression guarantee at the GL level: the identical draw
// sequence on a plain (no-props) 1-sample surface issues no MSAA renderbuffer allocations and no resolve blits, and
// never promotes the ops task.
func TestDMSAADefaultOffTouchesNoMSAAState(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDMSAATestSDC(t, dc, false)
	defer sdc.Release()
	if sdc.CanUseDynamicMSAA() {
		t.Fatal("plain surface unexpectedly enabled dynamic MSAA")
	}

	sdc.Clear([4]float32{0, 0, 0, 1})
	drawAAStar(sdc)
	if sdc.GetOpsTask().UsesMSAASurface() {
		t.Fatal("usesMSAASurface set on a 1-sample non-DMSAA surface")
	}
	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if got := counts("glRenderbufferStorageMultisample"); got != 0 {
		t.Errorf("glRenderbufferStorageMultisample = %d, want 0 with DMSAA off", got)
	}
	if got := counts("glBlitFramebuffer"); got != 0 {
		t.Errorf("glBlitFramebuffer = %d, want 0 with DMSAA off", got)
	}
}

// TestDMSAAStencilEscalation covers the "always trigger DMSAA when there is stencil" lane of addDrawOp: a draw that
// requires stencil (a stencil-clipped fill) promotes the task to the MSAA surface even when the op itself is not an
// MSAA op.
func TestDMSAAStencilEscalation(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDMSAATestSDC(t, dc, true)
	defer sdc.Release()

	// A non-AA rect fill carrying user stencil settings (the StencilRect lane ClipStack uses).
	paint := NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, A: 1})
	identity := geom.IdentityMatrix()
	sdc.StencilRect(nil, SetClipBitSettings(false), paint, gpu.AANo, &identity,
		geom.Rect{Right: 8, Bottom: 8}, nil)
	if !sdc.GetOpsTask().UsesMSAASurface() {
		t.Fatal("stencil-carrying draw did not escalate to the MSAA surface under DMSAA")
	}
}
