// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the resource layer: the Gpu runs against the Apple M4 Max caps profile from caps_test.go, with
// the object-creation entry points implemented by purego callbacks and the state-changing entry points replaced by
// per-name counters, so the tests can assert exactly which GL calls the state shadow allows through — the
// redundant-call elimination that is the state shadow's whole point — without a live context.

package gl

import (
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// recorder state shared by the package-level callbacks (tests are not parallel).
var (
	recCounts map[string]int
	recNextID uint32
	recMapBuf [1 << 16]byte
)

// counterProcs holds one counting callback per GL entry point name, created lazily and reused for the life of the test
// binary (purego callbacks are never freed).
var counterProcs = map[string]uintptr{}

func counterProc(name string) uintptr {
	if p, ok := counterProcs[name]; ok {
		return p
	}
	p := purego.NewCallback(func(_, _, _, _, _, _ uintptr) uintptr {
		if recCounts != nil {
			recCounts[name]++
		}
		return 0
	})
	counterProcs[name] = p
	return p
}

func genProc(name string) uintptr {
	if p, ok := counterProcs[name]; ok {
		return p
	}
	p := purego.NewCallback(func(n, ptr uintptr) uintptr {
		if recCounts != nil {
			recCounts[name]++
		}
		for i := 0; i < int(int32(n)); i++ {
			recNextID++
			*(*uint32)(ptrFromUintptr(ptr + uintptr(i)*4)) = recNextID
		}
		return 0
	})
	counterProcs[name] = p
	return p
}

func constProc(name string, value uintptr) uintptr {
	if p, ok := counterProcs[name]; ok {
		return p
	}
	p := purego.NewCallback(func(_, _, _, _, _, _ uintptr) uintptr {
		if recCounts != nil {
			recCounts[name]++
		}
		return value
	})
	counterProcs[name] = p
	return p
}

// newRecordingGpu builds a Gpu over the M4 Max fake-driver profile with recording hooks installed.
func newRecordingGpu(t *testing.T) *Gpu {
	t.Helper()
	return newRecordingGpuWithDriver(t, appleM4MaxDriver())
}

// newRecordingGpuWithDriver is newRecordingGpu over an arbitrary caps profile, for tests that need a capability the M4
// Max profile lacks (e.g. mesaLlvmpipeDriver's baseVertexBaseInstance).
func newRecordingGpuWithDriver(t *testing.T, d *capsFakeDriver) *Gpu {
	t.Helper()
	return newRecordingGpuWithDriverOptions(t, d, nil)
}

// newRecordingGpuWithDriverOptions additionally takes explicit context options, for tests that pin option-dependent
// behavior — e.g. the per-OS fGlyphsAsPathsFontSize (E.1) — without depending on the host GOOS's defaults.
func newRecordingGpuWithDriverOptions(t *testing.T, d *capsFakeDriver, options *gpu.ContextOptions) *Gpu {
	t.Helper()
	recCounts = map[string]int{}
	recNextID = 0
	ctxInfo := d.newContextInfo(t, options)
	f := &ctxInfo.Interface.Functions

	// Object factories and queries with real-ish behavior.
	f.genBuffers = genProc("glGenBuffers")
	f.genTextures = genProc("glGenTextures")
	f.genFramebuffers = genProc("glGenFramebuffers")
	f.genRenderbuffers = genProc("glGenRenderbuffers")
	f.genVertexArrays = genProc("glGenVertexArrays")
	f.genSamplers = genProc("glGenSamplers")
	f.checkFramebufferStatus = constProc("glCheckFramebufferStatus", FRAMEBUFFER_COMPLETE)
	f.fenceSync = constProc("glFenceSync", 0x1234)
	f.clientWaitSync = constProc("glClientWaitSync", ALREADY_SIGNALED)
	f.unmapBuffer = constProc("glUnmapBuffer", 1)
	if counterProcs["glMapBufferRange"] == 0 {
		counterProcs["glMapBufferRange"] = purego.NewCallback(func(_, _, _, _ uintptr) uintptr {
			if recCounts != nil {
				recCounts["glMapBufferRange"]++
			}
			return uintptr(unsafe.Pointer(&recMapBuf[0]))
		})
	}
	f.mapBufferRange = counterProcs["glMapBufferRange"]

	// State-changing entry points become pure counters.
	for _, hook := range []struct {
		field *uintptr
		name  string
	}{
		{field: &f.activeTexture, name: "glActiveTexture"},
		{field: &f.bindBuffer, name: "glBindBuffer"},
		{field: &f.bindTexture, name: "glBindTexture"},
		{field: &f.bindFramebuffer, name: "glBindFramebuffer"},
		{field: &f.bindRenderbuffer, name: "glBindRenderbuffer"},
		{field: &f.bindVertexArray, name: "glBindVertexArray"},
		{field: &f.bindSampler, name: "glBindSampler"},
		{field: &f.scissor, name: "glScissor"},
		{field: &f.viewport, name: "glViewport"},
		{field: &f.enable, name: "glEnable"},
		{field: &f.disable, name: "glDisable"},
		{field: &f.texParameteri, name: "glTexParameteri"},
		{field: &f.texStorage2D, name: "glTexStorage2D"},
		{field: &f.texSubImage2D, name: "glTexSubImage2D"},
		{field: &f.bufferData, name: "glBufferData"},
		{field: &f.bufferSubData, name: "glBufferSubData"},
		{field: &f.clear, name: "glClear"},
		{field: &f.clearColor, name: "glClearColor"},
		{field: &f.colorMask, name: "glColorMask"},
		{field: &f.deleteBuffers, name: "glDeleteBuffers"},
		{field: &f.deleteTextures, name: "glDeleteTextures"},
		{field: &f.deleteFramebuffers, name: "glDeleteFramebuffers"},
		{field: &f.deleteRenderbuffers, name: "glDeleteRenderbuffers"},
		{field: &f.deleteSync, name: "glDeleteSync"},
		{field: &f.deleteSamplers, name: "glDeleteSamplers"},
		{field: &f.framebufferTexture2D, name: "glFramebufferTexture2D"},
		{field: &f.framebufferRenderbuffer, name: "glFramebufferRenderbuffer"},
		{field: &f.blitFramebuffer, name: "glBlitFramebuffer"},
		{field: &f.renderbufferStorage, name: "glRenderbufferStorage"},
		{field: &f.renderbufferStorageMultisample, name: "glRenderbufferStorageMultisample"},
		{field: &f.readPixels, name: "glReadPixels"},
		{field: &f.finish, name: "glFinish"},
		{field: &f.flush, name: "glFlush"},
		{field: &f.samplerParameteri, name: "glSamplerParameteri"},
		{field: &f.generateMipmap, name: "glGenerateMipmap"},
		{field: &f.useProgram, name: "glUseProgram"},
		{field: &f.vertexAttribPointer, name: "glVertexAttribPointer"},
		{field: &f.vertexAttribIPointer, name: "glVertexAttribIPointer"},
		{field: &f.vertexAttribDivisor, name: "glVertexAttribDivisor"},
		{field: &f.enableVertexAttribArray, name: "glEnableVertexAttribArray"},
		{field: &f.disableVertexAttribArray, name: "glDisableVertexAttribArray"},
		{field: &f.drawArrays, name: "glDrawArrays"},
		{field: &f.drawArraysInstanced, name: "glDrawArraysInstanced"},
		{field: &f.drawArraysInstancedBaseInstance, name: "glDrawArraysInstancedBaseInstance"},
		{field: &f.drawElements, name: "glDrawElements"},
		{field: &f.drawElementsInstanced, name: "glDrawElementsInstanced"},
		{field: &f.drawElementsInstancedBaseVertexBaseInstance, name: "glDrawElementsInstancedBaseVertexBaseInstance"},
		{field: &f.drawRangeElements, name: "glDrawRangeElements"},
	} {
		*hook.field = counterProc(hook.name)
	}
	// Entry points on the RegisterFunc lane (glBlitFramebuffer, glClearColor, ...) were bound to their typed fn
	// pointers at assembly time, so rebind them over the counting procs.
	f.initRegistered()

	g := NewGpu(ctxInfo)
	if g == nil {
		t.Fatal("NewGpu returned nil")
	}
	// Consume the initial all-dirty reset (state tracking starts dirty and is lazily reconciled on first use) so tests
	// count only the GL traffic they cause.
	g.handleDirtyContext()
	recCounts = map[string]int{}
	return g
}

func counts(name string) int { return recCounts[name] }

// TestGpuBindBufferStateShadow verifies bindBuffer skips redundant binds and that index binds implicitly bind VAO 0.
func TestGpuBindBufferStateShadow(t *testing.T) {
	g := newRecordingGpu(t)

	b1 := g.CreateBuffer(1024, gpu.BufferTypeVertex, gpu.AccessPatternDynamic)
	if b1 == nil {
		t.Fatal("CreateBuffer failed")
	}
	base := counts("glBindBuffer")

	// b1 is already bound from creation: no new GL bind.
	g.BindBuffer(gpu.BufferTypeVertex, b1)
	g.BindBuffer(gpu.BufferTypeVertex, b1)
	if got := counts("glBindBuffer"); got != base {
		t.Errorf("redundant binds hit GL: %d calls (base %d)", got, base)
	}

	// A second buffer forces one bind; going back to b1 forces another.
	b2 := g.CreateBuffer(1024, gpu.BufferTypeVertex, gpu.AccessPatternDynamic)
	afterCreate := counts("glBindBuffer")
	g.BindBuffer(gpu.BufferTypeVertex, b1)
	if got := counts("glBindBuffer"); got != afterCreate+1 {
		t.Errorf("bind after switch = %d calls, want %d", got, afterCreate+1)
	}

	// Index binds implicitly bind VAO 0 through the tracked path.
	ib := g.CreateBuffer(512, gpu.BufferTypeIndex, gpu.AccessPatternStatic)
	vaoBase := counts("glBindVertexArray")
	g.BindBuffer(gpu.BufferTypeIndex, ib)
	if got := counts("glBindVertexArray"); got != vaoBase {
		t.Errorf("VAO 0 rebind should have been elided, got %d calls (base %d)", got, vaoBase)
	}

	b1.Unref()
	b2.Unref()
	ib.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}

// TestGpuScissorViewportShadow verifies the scissor/viewport shadows and their reset.
func TestGpuScissorViewportShadow(t *testing.T) {
	g := newRecordingGpu(t)

	g.flushScissorTest(true)
	g.flushScissorTest(true)
	if got := counts("glEnable"); got != 1 {
		t.Errorf("scissor enable calls = %d, want 1", got)
	}
	rect := geom.IRectXYWH(1, 2, 30, 40)
	g.flushScissorRect(rect, 100, gpu.OriginTopLeft)
	g.flushScissorRect(rect, 100, gpu.OriginTopLeft)
	if got := counts("glScissor"); got != 1 {
		t.Errorf("scissor calls = %d, want 1", got)
	}
	// A different origin produces a different native rect and must be flushed.
	g.flushScissorRect(rect, 100, gpu.OriginBottomLeft)
	if got := counts("glScissor"); got != 2 {
		t.Errorf("scissor calls = %d, want 2", got)
	}

	vp := geom.IRectWH(64, 64)
	g.flushViewport(vp, 64, gpu.OriginTopLeft)
	g.flushViewport(vp, 64, gpu.OriginTopLeft)
	if got := counts("glViewport"); got != 1 {
		t.Errorf("viewport calls = %d, want 1", got)
	}

	// External GL use dirties the view state; the next flush re-emits.
	g.ResetContext(ViewBackendState)
	g.handleDirtyContext()
	g.flushScissorTest(true)
	g.flushViewport(vp, 64, gpu.OriginTopLeft)
	if got := counts("glEnable"); got != 2 {
		t.Errorf("scissor enable after reset = %d, want 2", got)
	}
	if got := counts("glViewport"); got != 2 {
		t.Errorf("viewport after reset = %d, want 2", got)
	}
}

// TestGpuBlendStateShadow verifies flushBlendAndColorWrite's shadow.
func TestGpuBlendStateShadow(t *testing.T) {
	g := newRecordingGpu(t)
	f := &g.ctx.Interface.Functions
	f.blendEquation = counterProc("glBlendEquation")
	f.blendFunc = counterProc("glBlendFunc")

	srcOver := gpu.BlendInfo{
		Equation:    gpu.BlendEquationAdd,
		SrcBlend:    gpu.BlendCoeffOne,
		DstBlend:    gpu.BlendCoeffISA,
		WritesColor: true,
	}
	g.flushBlendAndColorWrite(srcOver, gpu.SwizzleRGBA)
	g.flushBlendAndColorWrite(srcOver, gpu.SwizzleRGBA)
	if got := counts("glBlendEquation"); got != 1 {
		t.Errorf("blend equation calls = %d, want 1", got)
	}
	if got := counts("glBlendFunc"); got != 1 {
		t.Errorf("blend func calls = %d, want 1", got)
	}
	if got := counts("glColorMask"); got != 1 {
		t.Errorf("color mask calls = %d, want 1", got)
	}

	// Disabling blending (src, one/zero) disables GL_BLEND once.
	disabled := gpu.MakeBlendInfo()
	g.flushBlendAndColorWrite(disabled, gpu.SwizzleRGBA)
	g.flushBlendAndColorWrite(disabled, gpu.SwizzleRGBA)
	if got := counts("glDisable"); got != 1 {
		t.Errorf("disable calls = %d, want 1", got)
	}
}

// TestGpuCreateTextureAndClearShadow drives CreateTexture + Clear through the fake driver.
func TestGpuCreateTextureAndClearShadow(t *testing.T) {
	g := newRecordingGpu(t)

	dims := geom.ISize{Width: 64, Height: 64}
	tex := g.CreateTexture(dims, FormatRGBA8, gpu.TextureType2D, gpu.RenderableYes, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "test-rt")
	if tex == nil {
		t.Fatal("CreateTexture failed")
	}
	if tex.AsTexture() == nil || tex.AsRenderTarget() == nil {
		t.Fatal("expected a texture render target")
	}
	// M4 Max exposes ARB_texture_storage: allocation must go through TexStorage2D.
	if got := counts("glTexStorage2D"); got != 1 {
		t.Errorf("texStorage calls = %d, want 1", got)
	}
	if tex.GpuMemorySize() != 64*64*4 {
		t.Errorf("gpuMemorySize = %d, want %d", tex.GpuMemorySize(), 64*64*4)
	}

	rt := tex.AsRenderTarget()
	scissor := gpu.MakeScissorState(dims)
	color := [4]float32{0.25, 0.5, 0.75, 1}
	g.Clear(&scissor, color, rt, false, gpu.OriginTopLeft)
	fbBinds := counts("glBindFramebuffer")
	clearColors := counts("glClearColor")
	g.Clear(&scissor, color, rt, false, gpu.OriginTopLeft)
	if got := counts("glBindFramebuffer"); got != fbBinds {
		t.Errorf("redundant FBO bind hit GL: %d calls (base %d)", got, fbBinds)
	}
	if got := counts("glClearColor"); got != clearColors {
		t.Errorf("redundant clear color hit GL: %d (base %d)", got, clearColors)
	}
	if got := counts("glClear"); got != 2 {
		t.Errorf("clear calls = %d, want 2", got)
	}

	// Binding a different render target then returning must rebind.
	tex2 := g.CreateTexture(dims, FormatRGBA8, gpu.TextureType2D, gpu.RenderableYes, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "test-rt-2")
	g.Clear(&scissor, color, tex2.AsRenderTarget(), false, gpu.OriginTopLeft)
	fbBinds = counts("glBindFramebuffer")
	g.Clear(&scissor, color, rt, false, gpu.OriginTopLeft)
	if got := counts("glBindFramebuffer"); got != fbBinds+1 {
		t.Errorf("render target switch binds = %d, want %d", got, fbBinds+1)
	}

	tex.Unref()
	tex2.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
	// Each TRT deletes one texture and one FBO.
	if got := counts("glDeleteTextures"); got != 2 {
		t.Errorf("delete texture calls = %d, want 2", got)
	}
	if got := counts("glDeleteFramebuffers"); got != 2 {
		t.Errorf("delete framebuffer calls = %d, want 2", got)
	}
}

// TestGpuBindTextureShadow verifies texture unit binding and sampler-object state tracking.
func TestGpuBindTextureShadow(t *testing.T) {
	g := newRecordingGpu(t)
	if !g.glCaps().UseSamplerObjects() {
		t.Fatal("expected sampler objects on the M4 profile")
	}

	dims := geom.ISize{Width: 16, Height: 16}
	surf := g.CreateTexture(dims, FormatRGBA8, gpu.TextureType2D, gpu.RenderableNo, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "tex")
	if surf == nil {
		t.Fatal("CreateTexture failed")
	}
	tex := surf.AsTexture()

	state := gpu.SamplerStateFilter(gpu.FilterModeLinear, gpu.MipmapModeNone)
	binds := counts("glBindTexture")
	g.BindTexture(0, state, tex)
	if got := counts("glBindTexture"); got != binds+1 {
		t.Errorf("first bind = %d calls, want %d", got, binds+1)
	}
	samplerBinds := counts("glBindSampler")
	genSamplers := counts("glGenSamplers")
	g.BindTexture(0, state, tex)
	if got := counts("glBindTexture"); got != binds+1 {
		t.Errorf("redundant texture bind hit GL")
	}
	if got := counts("glBindSampler"); got != samplerBinds {
		t.Errorf("redundant sampler bind hit GL")
	}
	// The same sampler state must reuse the cached sampler object (a new unit needs a new bind).
	g.BindTexture(1, state, tex)
	if got := counts("glGenSamplers"); got != genSamplers {
		t.Errorf("sampler object not reused: %d gens (base %d)", got, genSamplers)
	}

	// After a texture-binding reset, the bind must be re-emitted.
	preReset := counts("glBindTexture")
	g.ResetContext(TextureBindingBackendState)
	g.handleDirtyContext()
	g.BindTexture(0, state, tex)
	if got := counts("glBindTexture"); got != preReset+1 {
		t.Errorf("bind after reset = %d calls, want %d", got, preReset+1)
	}

	surf.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}

// TestGpuStencilProbeCaching verifies getCompatibleStencilIndex probes once per format.
func TestGpuStencilProbeCaching(t *testing.T) {
	g := newRecordingGpu(t)

	idx := g.getCompatibleStencilIndex(FormatRGBA8)
	if idx < 0 {
		t.Fatal("no compatible stencil format found")
	}
	genRB := counts("glGenRenderbuffers")
	if again := g.getCompatibleStencilIndex(FormatRGBA8); again != idx {
		t.Errorf("second probe = %d, want %d", again, idx)
	}
	if got := counts("glGenRenderbuffers"); got != genRB {
		t.Error("second probe re-ran the GL experiment")
	}

	// The stencil attachment factory uses the probed format.
	stencil := g.MakeStencilAttachment(FormatRGBA8, geom.ISize{Width: 32, Height: 32}, 1)
	if stencil == nil {
		t.Fatal("MakeStencilAttachment failed")
	}
	if got := stencil.GLFormat(); got != g.glCaps().StencilFormats()[idx] {
		t.Errorf("stencil format = %v, want %v", got, g.glCaps().StencilFormats()[idx])
	}
	stencil.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}

// TestGpuWrapBackendRenderTarget verifies the wrapped-FBO lane (gr_backendrendertarget_new_gl).
func TestGpuWrapBackendRenderTarget(t *testing.T) {
	g := newRecordingGpu(t)

	dims := geom.ISize{Width: 640, Height: 480}
	surf := g.WrapBackendRenderTarget(dims, FormatRGBA8, 1, 8, 0)
	if surf == nil {
		t.Fatal("WrapBackendRenderTarget failed")
	}
	rt := surf.AsRenderTarget()
	if rt == nil || surf.AsTexture() != nil {
		t.Fatal("expected a pure render target")
	}
	if !surf.GLRTFBOIDIs0() || !rt.IsFBO0(false) {
		t.Error("wrapped FBO 0 not flagged")
	}
	if got := rt.NumStencilBits(false); got != 8 {
		t.Errorf("stencil bits = %d, want 8", got)
	}
	if surf.BudgetedType() != gpu.BudgetedTypeUnbudgetedUncacheable || !surf.RefsWrappedObjects() {
		t.Error("wrapped RT should be unbudgeted/uncacheable and flagged wrapped")
	}
	if got := surf.GpuMemorySize(); got != 640*480*4 {
		t.Errorf("gpuMemorySize = %d, want %d", got, 640*480*4)
	}
	// The wrapped RT itself is unbudgeted, but its stand-in stencil Attachment registers budgeted regardless
	// (renderbuffer ID 0, so nothing is ever deleted).
	stencilBytes := uint64(640) * 480 * uint64(Format(FormatDEPTH24STENCIL8).BytesPerBlock())
	if got := g.ResourceCache().BudgetedResourceBytes(); got != stencilBytes {
		t.Errorf("budgeted bytes = %d, want the stencil stand-in's %d", got, stencilBytes)
	}

	// Unreffing a wrapped resource with no unique key frees it immediately, without deleting the caller's FBO. (The
	// internal stencil attachment is owned/budgeted and purges normally.)
	delFB := counts("glDeleteFramebuffers")
	surf.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
	if g.ResourceCache().ResourceCount() != 0 {
		t.Errorf("cache count = %d, want 0", g.ResourceCache().ResourceCount())
	}
	if got := counts("glDeleteFramebuffers"); got != delFB {
		t.Error("wrapped FBO was deleted")
	}
}

// TestGpuScratchTextureReuse verifies the texture scratch key round-trip through the cache.
func TestGpuScratchTextureReuse(t *testing.T) {
	g := newRecordingGpu(t)
	cache := g.ResourceCache()

	dims := geom.ISize{Width: 32, Height: 32}
	surf := g.CreateTexture(dims, FormatRGBA8, gpu.TextureType2D, gpu.RenderableNo, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "scratch-tex")
	if surf == nil {
		t.Fatal("CreateTexture failed")
	}
	id := surf.AsTexture().TextureID()
	surf.Unref()
	if cache.ResourceCount() != 1 {
		t.Fatalf("cache count = %d, want 1 (texture should stay as scratch)", cache.ResourceCount())
	}

	var key gpu.ScratchKey
	ComputeTextureScratchKey(g.glCaps(), FormatRGBA8, dims, gpu.RenderableNo, 1, gpu.MipmappedNo,
		&key)
	found := cache.FindAndRefScratchResource(&key)
	if found == nil {
		t.Fatal("scratch find failed")
	}
	reused, ok := found.(*Surface)
	if !ok || reused.AsTexture().TextureID() != id {
		t.Error("scratch find returned a different texture")
	}
	// A second request must not return the same (now reffed) resource.
	if second := cache.FindAndRefScratchResource(&key); second != nil {
		t.Error("reffed scratch resource returned again")
		second.(*Surface).Unref()
	}
	reused.Unref()
	cache.PurgeUnlockedResources(gpu.PurgeAllResources)
	if got := counts("glDeleteTextures"); got != 1 {
		t.Errorf("delete texture calls = %d, want 1", got)
	}
}

// TestGpuBufferLifecycle drives buffer update/map and the dynamic-buffer scratch key.
func TestGpuBufferLifecycle(t *testing.T) {
	g := newRecordingGpu(t)
	cache := g.ResourceCache()

	b := g.CreateBuffer(256, gpu.BufferTypeVertex, gpu.AccessPatternDynamic)
	if b == nil {
		t.Fatal("CreateBuffer failed")
	}
	// Non-preserving update invalidates via BufferData(null) (4.1 has no InvalidateBufferData).
	data := make([]byte, 64)
	if !b.UpdateData(data, 0, false) {
		t.Error("UpdateData failed")
	}
	if got := counts("glBufferSubData"); got != 1 {
		t.Errorf("bufferSubData calls = %d, want 1", got)
	}
	if got := counts("glBufferData"); got != 2 { // creation + invalidate
		t.Errorf("bufferData calls = %d, want 2", got)
	}
	if p := b.Map(); p == nil {
		t.Error("Map failed")
	} else {
		b.Unmap()
	}

	// Dynamic buffers are recycled by scratch key.
	b.Unref()
	var key gpu.ScratchKey
	ComputeScratchKeyForDynamicBuffer(256, gpu.BufferTypeVertex, &key)
	found := cache.FindAndRefScratchResource(&key)
	if found == nil {
		t.Fatal("dynamic buffer not found as scratch")
	}
	if found.(*Buffer) != b {
		t.Error("found a different buffer")
	}
	found.(*Buffer).Unref()

	// Static buffers must not get scratch keys.
	s := g.CreateBuffer(256, gpu.BufferTypeVertex, gpu.AccessPatternStatic)
	if s.ScratchKey().IsValid() {
		t.Error("static buffer has a scratch key")
	}
	s.Unref()
	cache.PurgeUnlockedResources(gpu.PurgeAllResources)
	if cache.ResourceCount() != 0 {
		t.Errorf("cache count = %d, want 0", cache.ResourceCount())
	}
}

// TestGpuSyncAndFinishedCallbacks drives the fence lanes and finished-proc list.
func TestGpuSyncAndFinishedCallbacks(t *testing.T) {
	g := newRecordingGpu(t)
	if g.glCaps().FenceType() != FenceSyncObject {
		t.Fatalf("fence type = %v, want sync objects", g.glCaps().FenceType())
	}

	sync := g.InsertSync()
	if !sync.IsValid() {
		t.Fatal("InsertSync failed")
	}
	if !g.TestSync(sync) {
		t.Error("TestSync should report signaled (fake driver always signals)")
	}
	g.DeleteSync(sync)
	if got := counts("glDeleteSync"); got != 1 {
		t.Errorf("deleteSync calls = %d, want 1", got)
	}

	fired := 0
	g.AddFinishedCallback(func() { fired++ })
	// InsertSync defers the flush to submit; SubmitToGpu without sync flushes and polls.
	if counts("glFlush") != 0 {
		t.Error("flush should be deferred until submit")
	}
	g.SubmitToGpu(false)
	if fired != 1 {
		t.Errorf("callback fired %d times, want 1", fired)
	}
	if counts("glFlush") != 1 {
		t.Errorf("flush calls = %d, want 1", counts("glFlush"))
	}

	// syncCPU forces a glFinish and drains callbacks.
	g.AddFinishedCallback(func() { fired++ })
	g.SubmitToGpu(true)
	if fired != 2 {
		t.Errorf("callback fired %d times, want 2", fired)
	}
	if counts("glFinish") != 1 {
		t.Errorf("finish calls = %d, want 1", counts("glFinish"))
	}
}

// TestGpuAbandonVsRelease verifies the two teardown lanes: abandon makes no GL deletes, release frees everything.
func TestGpuAbandonVsRelease(t *testing.T) {
	newSurfWithResources := func(t *testing.T) (*Gpu, *Surface, *Buffer) {
		t.Helper()
		g := newRecordingGpu(t)
		surf := g.CreateTexture(geom.ISize{Width: 8, Height: 8}, FormatRGBA8, gpu.TextureType2D,
			gpu.RenderableNo, 1, gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888,
			gpu.ColorTypeRGBA8888, nil, "t")
		buf := g.CreateBuffer(128, gpu.BufferTypeVertex, gpu.AccessPatternDynamic)
		if surf == nil || buf == nil {
			t.Fatal("resource creation failed")
		}
		return g, surf, buf
	}

	g, surf, buf := newSurfWithResources(t)
	g.AbandonContext()
	if !g.Abandoned() {
		t.Error("not abandoned")
	}
	if !surf.WasDestroyed() || !buf.WasDestroyed() {
		t.Error("resources should be destroyed after abandon")
	}
	if counts("glDeleteTextures") != 0 || counts("glDeleteBuffers") != 0 {
		t.Error("abandon must not touch GL")
	}
	if g.ResourceCache().ResourceCount() != 0 {
		t.Error("cache not empty after abandon")
	}
	surf.Unref()
	buf.Unref()

	g, surf, buf = newSurfWithResources(t)
	g.ReleaseResourcesAndAbandonContext()
	if !surf.WasDestroyed() || !buf.WasDestroyed() {
		t.Error("resources should be destroyed after release-and-abandon")
	}
	if counts("glDeleteTextures") != 1 || counts("glDeleteBuffers") != 1 {
		t.Errorf("release should free GL objects (textures %d, buffers %d)",
			counts("glDeleteTextures"), counts("glDeleteBuffers"))
	}
	surf.Unref()
	buf.Unref()
}

// TestGpuResetTextureBindings verifies the gr_direct_context_reset_gl_texture_bindings lane.
func TestGpuResetTextureBindings(t *testing.T) {
	g := newRecordingGpu(t)
	surf := g.CreateTexture(geom.ISize{Width: 8, Height: 8}, FormatRGBA8, gpu.TextureType2D,
		gpu.RenderableNo, 1, gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888,
		nil, "t")
	g.BindTexture(0, gpu.SamplerStateFilter(gpu.FilterModeNearest, gpu.MipmapModeNone),
		surf.AsTexture())

	binds := counts("glBindTexture")
	g.ResetTextureBindings()
	// Only modified targets are unbound: unit 0 2D (the draw bind) and the scratch unit 2D (creation bound there).
	if got := counts("glBindTexture"); got != binds+2 {
		t.Errorf("reset unbinds = %d, want %d", got-binds, 2)
	}
	surf.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}
