// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gorender

import (
	"errors"

	gocanvas "github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/gpu/gl/gltest"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/surface"
)

// offscreenStencilBits is the stencil bit depth of the wrapped FBO's DEPTH24_STENCIL8 attachment.
const offscreenStencilBits = 8

// GPUContext renders scenarios through the library's GL backend — the render context behind the GPU golden gates and the
// GPU lanes of `oracle bless`/`soak`. It owns a headless gltest GL context (the stand-in for the unison-provided
// context, exactly as gpu/gl's own live tests use it) plus a gl.DirectContext assembled over it via
// MakeNativeInterface + MakeGLDirectContext — the same assembly unison performs in production.
//
// It is thread-bound: gltest.New locks the calling goroutine to its OS thread (GL binds the context to the thread), and
// Dispose unlocks it. Every surface it renders and every readback must run on that goroutine — the
// one-context-per-thread contract. The gates read their reference side from the checked-in goldens, so one context is
// all they need; the constraint mattered when the C Skia oracle owned a second one and a differential had to render
// fully through one, dispose it, then the other.
type GPUContext struct {
	ctx  *gltest.Context
	intf *gl.Interface
	dc   *gl.DirectContext
}

// NewGPUContext creates the gltest GL context and the gl.DirectContext over it. It honors
// CANVAS_GLTEST_RENDERER=software the same way gltest does — the pin both `oracle bless` and the golden gates run
// under, so a golden comparison exercises the GL stack the goldens were captured on (see ../goldens/README.md). It
// returns an error — which callers treat as a skip — when no core-profile GL context is available (headless CI without
// GL, or a not-yet-implemented platform leg).
func NewGPUContext() (*GPUContext, error) {
	ctx, err := gltest.New()
	if err != nil {
		return nil, err
	}
	intf := gl.MakeNativeInterface()
	if intf == nil {
		ctx.Destroy()
		return nil, errors.New("gorender: MakeNativeInterface returned nil with a current context")
	}
	dc := gl.MakeGLDirectContext(intf, nil)
	if dc == nil {
		ctx.Destroy()
		return nil, errors.New("gorender: MakeGLDirectContext returned nil")
	}
	return &GPUContext{ctx: ctx, intf: intf, dc: dc}, nil
}

// RendererString returns the live context's GL_RENDERER string — the identity of the GL stack this context renders on,
// recorded into golden.Manifest.GLRenderer at capture time so a gate can detect that the stack moved. Like every other
// use of the context, it must be called on the goroutine that created it (the one-context-per-thread contract).
func (g *GPUContext) RendererString() string {
	return g.intf.Functions.GetStringGo(gl.RENDERER)
}

// VersionString returns the live context's GL_VERSION string, the companion to RendererString (same threading
// contract).
func (g *GPUContext) VersionString() string {
	return g.intf.Functions.GetStringGo(gl.VERSION)
}

// Dispose releases the GPU resources, abandons the direct context, and destroys the GL context (unlocking the OS
// thread).
func (g *GPUContext) Dispose() {
	if g.dc != nil {
		g.dc.ReleaseResourcesAndAbandonContext()
		g.dc = nil
	}
	if g.ctx != nil {
		g.ctx.Destroy()
		g.ctx = nil
	}
}

// RenderScenarioGPU renders one scenario through the library's GL backend into RGBA8888-premul pixels (tightly packed,
// top-left origin) — directly comparable with the self-captured GPU goldens in ../goldens/gpu, which `oracle bless`
// captures through this same path. It builds a SurfaceDrawContext, wraps it in a
// GL canvas device, replays the scenario through the shared sceneCanvas adapter, flushes, and reads the pixels back.
func RenderScenarioGPU(g *GPUContext, sc scenario.Scenario) []byte {
	return RenderSceneGPU(g, sc.Width, sc.Height, sc.Name, func(c *gocanvas.Canvas) {
		sc.Draw(sceneCanvas{c: c})
	})
}

// RenderSceneGPU renders an arbitrary draw callback through the library's GL backend into RGBA8888-premul pixels
// (tightly packed, top-left origin) — the one-off-scene form of RenderScenarioGPU for differentials whose content is
// not expressible in the declarative scenario corpus (e.g. paints carrying image filters).
func RenderSceneGPU(g *GPUContext, width, height int, label string, draw func(*gocanvas.Canvas)) []byte {
	dims := geom.ISize{Width: int32(width), Height: int32(height)}
	sdc := gl.MakeSurfaceDrawContext(g.dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, label)
	if sdc == nil {
		panic("gorender: MakeSurfaceDrawContext failed for " + label)
	}
	defer sdc.Release()

	c := gocanvas.New(gl.NewDevice(sdc))
	draw(c)
	g.dc.FlushAndSubmit(false)

	dst := gl.Pixels{
		ColorType: gpu.ColorTypeRGBA8888,
		Dims:      dims,
		RowBytes:  int(dims.Width) * 4,
		Data:      make([]byte, int(dims.Width)*int(dims.Height)*4),
	}
	if !sdc.ReadPixels(dst, geom.IPoint{}) {
		panic("gorender: ReadPixels failed for " + label)
	}
	return dst.Data
}

// createOffscreenFBO builds a caller-owned offscreen FBO — an RGBA8 color renderbuffer plus a packed DEPTH24_STENCIL8
// renderbuffer for the GL backend clip/path stenciling — the FBO shape a windowing embedder hands the library (and the
// same formats, attachments, and clear the removed C oracle's FBO helper used, so the archived goldens-skia renders
// were captured over an identical target). It leaves the FBO bound; the caller re-syncs the direct
// context's GL-state shadow before wrapping it. Returns the FBO name and its two renderbuffer names (for teardown), and
// ok=false if the FBO is incomplete.
func createOffscreenFBO(f *gl.Functions, w, h int32) (fbo, colorRB, dsRB uint32, ok bool) {
	f.GenRenderbuffers(1, &colorRB)
	f.BindRenderbuffer(gl.RENDERBUFFER, colorRB)
	f.RenderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, w, h)

	f.GenFramebuffers(1, &fbo)
	f.BindFramebuffer(gl.FRAMEBUFFER, fbo)
	f.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRB)

	f.GenRenderbuffers(1, &dsRB)
	f.BindRenderbuffer(gl.RENDERBUFFER, dsRB)
	f.RenderbufferStorage(gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, w, h)
	f.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, dsRB)

	if f.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE {
		f.BindFramebuffer(gl.FRAMEBUFFER, 0)
		f.DeleteRenderbuffers(1, &colorRB)
		f.DeleteRenderbuffers(1, &dsRB)
		f.DeleteFramebuffers(1, &fbo)
		return 0, 0, 0, false
	}
	// Clear to transparent black so an unrendered target reads back as zeros, matching the raster surface's zeroed
	// start.
	f.Viewport(0, 0, w, h)
	f.ClearColor(0, 0, 0, 0)
	f.ClearStencil(0)
	f.Clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
	return fbo, colorRB, dsRB, true
}

// RenderScenarioGPUWrappedFBO renders one scenario through the library's GL backend into a *caller-owned wrapped FBO* —
// the wrap-backend-render-target production path unison follows (it hands the library its window FBO), rather than the
// library-owned offscreen draw context RenderScenarioGPU uses. It drives the API unison drives —
// gl.NewRenderTargetSurfaceFromBackendRenderTarget (the Go equivalent of sk_surface_new_backend_render_target / Skia's
// Surfaces::WrapBackendRenderTarget) — so the wrapped-FBO surface path is gated end-to-end against the self-captured
// GPU goldens on the whole corpus rather than only CPU-checked in gpu/gl's live tests.
//
// It wraps at top-left origin: that isolates the wrapped-FBO surface-creation path — the thing this lane exists to
// gate — from the origin flip, so the result matches the RenderScenarioGPU owned-RT path (and therefore the
// owned-RT-captured goldens) scenario for scenario instead of carrying the origin-convention drift a
// bottom-left-vs-top-left comparison shows on aliased edges. The bottom-left flip unison uses in production is covered
// separately (against the asymmetric source image, where a flip would diverge) by the gpu/gl live tests.
func RenderScenarioGPUWrappedFBO(g *GPUContext, sc scenario.Scenario) []byte {
	return renderScenarioGPUWrappedFBOWithProps(g, sc, nil)
}

// RenderScenarioGPUDMSAA is RenderScenarioGPUWrappedFBO with the surface props carrying DynamicMSAAFlag (the library of
// SkSurfaceProps::kDynamicMSAA_Flag): the gpudmsaa lane's render path, which promotes path/stencil render passes to a
// dynamic 4x MSAA attachment over the wrapped render target. Its self-captured reference sets live in
// ../goldens/gpudmsaa; the lane needs its own sets because an MSAA resolve antialiases edges differently from
// coverage-AA.
func RenderScenarioGPUDMSAA(g *GPUContext, sc scenario.Scenario) []byte {
	return renderScenarioGPUWrappedFBOWithProps(g, sc,
		&surface.Props{Flags: surface.DynamicMSAAFlag, PixelGeometry: surface.PixelGeometryUnknown})
}

func renderScenarioGPUWrappedFBOWithProps(g *GPUContext, sc scenario.Scenario, props *surface.Props) []byte {
	w, h := int32(sc.Width), int32(sc.Height)
	f := &g.intf.Functions
	fbo, colorRB, dsRB, ok := createOffscreenFBO(f, w, h)
	if !ok {
		panic("gorender: offscreen FBO creation failed for scenario " + sc.Name)
	}
	// createOffscreenFBO bound the FBO behind the context's back, and GL reuses deleted FBO ids across scenarios, so
	// re-sync the direct context's GL-state shadow before it wraps and renders to the new FBO — the same reset a
	// production embedder performs after touching GL state the library cannot see.
	g.dc.ResetContext(gl.AllBackendState)

	rt := gl.NewRenderTargetSurfaceFromBackendRenderTarget(g.dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gl.FormatFromEnum(gl.RGBA8), 0, offscreenStencilBits, fbo,
		gpu.OriginTopLeft, props)
	if rt == nil {
		panic("gorender: NewRenderTargetSurfaceFromBackendRenderTarget returned nil for scenario " + sc.Name)
	}

	sc.Draw(sceneCanvas{c: rt.Canvas()})
	g.dc.FlushAndSubmit(false)

	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	buf := make([]byte, int(w)*int(h)*4)
	if !rt.ReadPixels(info, buf, int(w)*4, 0, 0) {
		panic("gorender: wrapped-FBO ReadPixels failed for scenario " + sc.Name)
	}

	// Release the draw context's GPU resources, then delete the caller-owned FBO (borrowed ownership, so the surface
	// never deletes it). SDC release precedes the FBO delete so any teardown GL targets a live FBO.
	rt.SDC().Release()
	f.BindFramebuffer(gl.FRAMEBUFFER, 0)
	f.DeleteFramebuffers(1, &fbo)
	f.DeleteRenderbuffers(1, &colorRB)
	f.DeleteRenderbuffers(1, &dsRB)
	// Teardown is the symmetric mutation to the setup above — an unbind plus three deletes performed behind the
	// context's back — so it needs the same re-sync. Without it the context's GL-state shadow still believes the
	// now-deleted FBO is bound, and the library skips redundant framebuffer binds: a later render on this same context
	// whose render target is handed the recycled FBO id would have its bind elided and draw into framebuffer 0.
	g.dc.ResetContext(gl.AllBackendState)
	return buf
}
