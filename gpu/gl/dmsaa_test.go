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
	"github.com/richardwilkes/canvas/raster"
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

// dmsaaStencilAttachment returns the MSAA-slot stencil attachment the draw context's render target is currently using
// (nil before one has been attached, which happens during the flush).
func dmsaaStencilAttachment(sdc *SurfaceDrawContext) *Attachment {
	proxy := sdc.AsRenderTargetProxy()
	if proxy == nil {
		return nil
	}
	rt := proxy.Proxy().PeekRenderTarget()
	if rt == nil {
		return nil
	}
	return rt.StencilAttachment(true)
}

// TestDMSAASharedStencilClearedForEveryRenderTarget: stencil attachments are shared between render targets under a
// unique key that carries no render-target identity, so the second render target to reach a given attachment inherits
// whatever the first left in it. Tracking "the initial clear already happened" on the attachment therefore downgrades
// the second target's kUserBitsCleared load op to a load of the first target's residue, which under DMSAA paints stray
// bands into a render that carries no clip at all. Every render target that asks for cleared stencil must get a real
// clear.
func TestDMSAASharedStencilClearedForEveryRenderTarget(t *testing.T) {
	dc := newShaderRecordingContext(t)
	dc.Gpu().ctx.Interface.Functions.clearStencil = counterProc("glClearStencil")

	// A stencil-carrying rect draw: it calls setNeedsStencil (so the ops task asks for kUserBitsCleared) without
	// recording a stencil-clip clear op, leaving the load-op clear as the only glClearStencil of the flush.
	flushStencilDraw := func(sdc *SurfaceDrawContext) (clears int, stencil *Attachment) {
		paint := NewPaint()
		paint.SetColor4f(colorcore.PMColor4f{R: 1, A: 1})
		identity := geom.IdentityMatrix()
		sdc.StencilRect(nil, SetClipBitSettings(false), paint, gpu.AANo, &identity,
			geom.Rect{Right: 8, Bottom: 8}, nil)
		recCounts = map[string]int{}
		dc.FlushAndSubmit(false)
		return counts("glClearStencil"), dmsaaStencilAttachment(sdc)
	}

	first := newDMSAATestSDC(t, dc, true)
	defer first.Release()
	firstClears, firstStencil := flushStencilDraw(first)
	if firstStencil == nil {
		t.Fatal("no MSAA stencil attachment after a stencil draw on a DMSAA surface")
	}
	if firstClears == 0 {
		t.Fatal("the first render target's kUserBitsCleared load op did not clear the stencil")
	}

	// A second, identically shaped DMSAA surface: the resource provider hands it the very same attachment, whose
	// contents are the first surface's leftovers.
	second := newDMSAATestSDC(t, dc, true)
	defer second.Release()
	secondClears, secondStencil := flushStencilDraw(second)
	if secondStencil != firstStencil {
		t.Fatal("the two render targets did not share a stencil attachment, so this test no longer covers the " +
			"sharing it exists to guard")
	}
	if secondClears == 0 {
		t.Fatal("the second render target loaded the shared stencil attachment instead of clearing it, " +
			"inheriting the first render target's residue")
	}
}

// newDMSAAClipTestSDC is newDMSAATestSDC at an explicit size, for the clip-lane tests below.
func newDMSAAClipTestSDC(t *testing.T, dc *DirectContext, w, h int32, dynamicMSAA bool) *SurfaceDrawContext {
	t.Helper()
	var props *surface.Props
	if dynamicMSAA {
		props = &surface.Props{Flags: surface.DynamicMSAAFlag}
	}
	sdc := MakeSurfaceDrawContextWithProps(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "dmsaa-clip-test", props)
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContextWithProps failed")
	}
	return sdc
}

// dmsaaConcaveClipPath is the concave AA path clip TestClipStackApplySWMask uses: too complex for an analytic coverage
// FP, so it lands on either the SW mask lane or the stencil lane.
func dmsaaConcaveClipPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(10, 10)
	p.LineTo(54, 10)
	p.LineTo(32, 30) // notch back toward the center
	p.LineTo(54, 54)
	p.LineTo(10, 54)
	p.Close()
	return p
}

// TestDMSAAClipUsesStencilNotSWMask: on a DMSAA surface the mask-requiring clip elements belong in the stencil buffer
// DMSAA is about to promote, not in a CPU-rasterized SW mask. The surface reports one sample, so a decision that looks
// only at NumSamples() sees "1 sample + AA mask" and wrongly takes the SW mask lane for every such draw — changing the
// coverage produced and paying a mask rasterization plus upload per draw.
func TestDMSAAClipUsesStencilNotSWMask(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDMSAAClipTestSDC(t, dc, 64, 64, true)
	defer sdc.Release()
	if !sdc.CanUseDynamicMSAA() {
		t.Fatal("DMSAA surface props did not enable dynamic MSAA")
	}
	if !sdc.AsRenderTargetProxy().CanUseStencil(sdc.Caps()) {
		t.Skip("the fake driver's caps cannot attach a stencil buffer")
	}

	// forceAA mirrors what NewDevice does for a DMSAA target: every element carries AAYes.
	cs := NewClipStack(deviceBounds64(), true)
	identity := geom.IdentityMatrix()
	cs.ClipPath(&identity, dmsaaConcaveClipPath(), gpu.AAYes, raster.ClipIntersect)

	op, bounds := applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64})
	out := MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeMSAA, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("effect = %v, want clipped", effect)
	}
	if !out.HasStencilClip() {
		t.Fatal("a DMSAA target must rasterize the remaining clip elements into the stencil buffer")
	}
	if len(cs.masks) != 0 {
		t.Fatalf("SW masks recorded = %d, want 0 on a DMSAA target", len(cs.masks))
	}

	// The same clip on a plain 1-sample surface still takes the SW mask lane: the DMSAA term must not disable the
	// software fallback where it is genuinely needed.
	plain := newDMSAAClipTestSDC(t, dc, 64, 64, false)
	defer plain.Release()
	csPlain := NewClipStack(deviceBounds64(), false)
	csPlain.ClipPath(&identity, dmsaaConcaveClipPath(), gpu.AAYes, raster.ClipIntersect)
	op, bounds = applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64})
	out = MakeAppliedClip(plain.Dimensions())
	if effect := csPlain.Apply(plain, op, gpu.AATypeCoverage, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("plain effect = %v, want clipped", effect)
	}
	if len(csPlain.masks) != 1 {
		t.Fatalf("plain SW masks = %d, want 1", len(csPlain.masks))
	}
	if out.HasStencilClip() {
		t.Fatal("a 1-sample non-DMSAA target must use the SW mask, not the stencil")
	}
}

// TestDMSAAStencilMaskHelperUsesMSAA: StencilMaskHelper.supportedAA reports the AA a stencil clip-mask draw may use.
// A DMSAA target reports one sample but promotes stencil draws onto an MSAA attachment, so its clip-mask draws must be
// issued AA; dropping the DMSAA term renders every stencil clip edge hard-aliased.
func TestDMSAAStencilMaskHelperUsesMSAA(t *testing.T) {
	dc := newShaderRecordingContext(t)

	dmsaa := newDMSAAClipTestSDC(t, dc, 32, 32, true)
	defer dmsaa.Release()
	if got := NewStencilMaskHelper(dmsaa).supportedAA(gpu.AANo); got != gpu.AAYes {
		t.Fatalf("DMSAA supportedAA = %v, want AAYes", got)
	}

	plain := newDMSAAClipTestSDC(t, dc, 32, 32, false)
	defer plain.Release()
	if got := NewStencilMaskHelper(plain).supportedAA(gpu.AAYes); got != gpu.AANo {
		t.Fatalf("1-sample supportedAA = %v, want AANo (no MSAA available)", got)
	}
}
