// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the resource layer: texture create/upload/readback round-trips, render-target clears through
// the state shadow, stencil attachment probing, buffer mapping, sync objects and finished callbacks, and scratch reuse
// — all on a real GL context provided the way unison will provide one (gltest). Skips when no GL context is available.

package gl_test

import (
	"testing"
	"unsafe"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
)

func newLiveGpu(t *testing.T) *gl.Gpu {
	t.Helper()
	ctxInfo := newLiveContextInfo(t)
	g := gl.NewGpu(ctxInfo)
	if g == nil {
		t.Fatal("NewGpu returned nil on a live context")
	}
	return g
}

func sweepGLErrors(t *testing.T, g *gl.Gpu, stage string) {
	t.Helper()
	if err := g.Functions().GetError(); err != gl.NO_ERROR {
		t.Fatalf("GL error 0x%04x after %s", err, stage)
	}
}

func testPattern(dims geom.ISize, seed byte) []byte {
	buf := make([]byte, int(dims.Width)*int(dims.Height)*4)
	for i := range buf {
		buf[i] = byte(i*7+13) + seed
	}
	// Force opaque alpha so the values are format-lossless everywhere.
	for i := 3; i < len(buf); i += 4 {
		buf[i] = 0xFF
	}
	return buf
}

// TestGpuLiveTextureUploadReadback creates a texture with initial contents, reads it back through the temporary-FBO
// lane, overwrites a sub-rect, and reads that back too.
func TestGpuLiveTextureUploadReadback(t *testing.T) {
	g := newLiveGpu(t)
	dims := geom.ISize{Width: 64, Height: 64}
	pixels := testPattern(dims, 0)
	texels := []gpu.MipLevel{{Pixels: pixels, RowBytes: int(dims.Width) * 4}}

	surf := g.CreateTexture(dims, gl.FormatRGBA8, gpu.TextureType2D, gpu.RenderableNo, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, texels, "live-tex")
	if surf == nil {
		t.Fatal("CreateTexture failed")
	}
	sweepGLErrors(t, g, "create+upload")

	got := make([]byte, len(pixels))
	if !g.ReadPixels(surf, geom.IRectSize(dims), gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888,
		got, int(dims.Width)*4) {
		t.Fatal("ReadPixels failed")
	}
	sweepGLErrors(t, g, "readback")
	for i := range got {
		if got[i] != pixels[i] {
			t.Fatalf("readback mismatch at byte %d: got %d, want %d", i, got[i], pixels[i])
		}
	}

	// Overwrite a sub-rect with WritePixels and confirm both regions.
	subDims := geom.ISize{Width: 16, Height: 16}
	subRect := geom.IRectXYWH(8, 4, subDims.Width, subDims.Height)
	subPixels := testPattern(subDims, 101)
	if !g.WritePixels(surf, subRect, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888,
		[]gpu.MipLevel{{Pixels: subPixels, RowBytes: int(subDims.Width) * 4}}, false) {
		t.Fatal("WritePixels failed")
	}
	sweepGLErrors(t, g, "sub-rect write")
	gotSub := make([]byte, len(subPixels))
	if !g.ReadPixels(surf, subRect, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, gotSub,
		int(subDims.Width)*4) {
		t.Fatal("sub-rect ReadPixels failed")
	}
	for i := range gotSub {
		if gotSub[i] != subPixels[i] {
			t.Fatalf("sub-rect mismatch at byte %d: got %d, want %d", i, gotSub[i], subPixels[i])
		}
	}
	// A pixel outside the sub-rect keeps its original value.
	var corner [4]byte
	if !g.ReadPixels(surf, geom.IRectXYWH(0, 0, 1, 1), gpu.ColorTypeRGBA8888,
		gpu.ColorTypeRGBA8888, corner[:], 4) {
		t.Fatal("corner ReadPixels failed")
	}
	for i := range corner {
		if corner[i] != pixels[i] {
			t.Fatalf("corner byte %d changed: got %d, want %d", i, corner[i], pixels[i])
		}
	}

	surf.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
	sweepGLErrors(t, g, "teardown")
}

func expectPixel(t *testing.T, buf []byte, dims geom.ISize, x, y int32, want [4]byte, stage string) {
	t.Helper()
	off := (int(y)*int(dims.Width) + int(x)) * 4
	for i := 0; i < 4; i++ {
		got := buf[off+i]
		diff := int(got) - int(want[i])
		if diff < -1 || diff > 1 {
			t.Fatalf("%s: pixel (%d,%d) channel %d = %d, want %d±1", stage, x, y, i, got, want[i])
		}
	}
}

// TestGpuLiveRenderTargetClear renders through the clear path (render target objects, FBO binding, scissor/clear-color
// state shadow) and verifies the results, including a scissored clear.
func TestGpuLiveRenderTargetClear(t *testing.T) {
	g := newLiveGpu(t)
	dims := geom.ISize{Width: 64, Height: 64}
	surf := g.CreateTexture(dims, gl.FormatRGBA8, gpu.TextureType2D, gpu.RenderableYes, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "live-rt")
	if surf == nil {
		t.Fatal("CreateTexture failed")
	}
	rt := surf.AsRenderTarget()
	if rt == nil {
		t.Fatal("expected render target")
	}

	scissor := gpu.MakeScissorState(dims)
	g.Clear(&scissor, [4]float32{1, 0.5, 0, 1}, rt, false, gpu.OriginTopLeft)
	sweepGLErrors(t, g, "full clear")

	buf := make([]byte, int(dims.Width)*int(dims.Height)*4)
	if !g.ReadPixels(surf, geom.IRectSize(dims), gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888,
		buf, int(dims.Width)*4) {
		t.Fatal("ReadPixels failed")
	}
	orange := [4]byte{255, 128, 0, 255}
	expectPixel(t, buf, dims, 0, 0, orange, "full clear")
	expectPixel(t, buf, dims, 63, 63, orange, "full clear")

	// A scissored clear only touches the scissor rect.
	if !scissor.Set(geom.IRectXYWH(8, 8, 16, 16)) {
		t.Fatal("scissor set failed")
	}
	g.Clear(&scissor, [4]float32{0, 0, 1, 1}, rt, false, gpu.OriginTopLeft)
	sweepGLErrors(t, g, "scissored clear")
	if !g.ReadPixels(surf, geom.IRectSize(dims), gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888,
		buf, int(dims.Width)*4) {
		t.Fatal("ReadPixels failed")
	}
	blue := [4]byte{0, 0, 255, 255}
	expectPixel(t, buf, dims, 8, 8, blue, "scissored clear inside")
	expectPixel(t, buf, dims, 23, 23, blue, "scissored clear inside")
	expectPixel(t, buf, dims, 7, 7, orange, "scissored clear outside")
	expectPixel(t, buf, dims, 24, 24, orange, "scissored clear outside")

	// Attach a stencil buffer and make sure the FBO stays usable (the deferred stencil bind runs on the next
	// flushRenderTarget).
	stencil := g.MakeStencilAttachment(gl.FormatRGBA8, dims, 1)
	if stencil == nil {
		t.Fatal("MakeStencilAttachment failed")
	}
	if bits := stencil.GLFormat().StencilBits(); bits < 8 {
		t.Errorf("stencil bits = %d, want >= 8", bits)
	}
	rt.AttachStencilAttachment(stencil, false)
	scissor.SetDisabled()
	g.Clear(&scissor, [4]float32{0, 1, 0, 1}, rt, false, gpu.OriginTopLeft)
	sweepGLErrors(t, g, "clear with stencil attached")
	if !g.ReadPixels(surf, geom.IRectXYWH(0, 0, 1, 1), gpu.ColorTypeRGBA8888,
		gpu.ColorTypeRGBA8888, buf[:4], 4) {
		t.Fatal("ReadPixels failed")
	}
	expectPixel(t, buf, geom.ISize{Width: 1, Height: 1}, 0, 0, [4]byte{0, 255, 0, 255},
		"clear with stencil")

	surf.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
	sweepGLErrors(t, g, "teardown")
}

// TestGpuLiveBufferMapUpdate exercises buffer creation, updateData, and map/unmap on real GL.
func TestGpuLiveBufferMapUpdate(t *testing.T) {
	g := newLiveGpu(t)
	b := g.CreateBuffer(1024, gpu.BufferTypeVertex, gpu.AccessPatternDynamic)
	if b == nil {
		t.Fatal("CreateBuffer failed")
	}
	data := make([]byte, 256)
	for i := range data {
		data[i] = byte(i)
	}
	if !b.UpdateData(data, 128, false) {
		t.Error("UpdateData failed")
	}
	sweepGLErrors(t, g, "updateData")
	if p := b.Map(); p == nil {
		t.Error("Map failed")
	} else {
		b.Unmap()
	}
	sweepGLErrors(t, g, "map/unmap")
	if !b.ClearToZero() {
		t.Error("ClearToZero failed")
	}
	sweepGLErrors(t, g, "clearToZero")
	b.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
	sweepGLErrors(t, g, "teardown")
}

// TestGpuLiveSyncAndFinishedCallbacks exercises fences and the finished-proc list on real GL.
func TestGpuLiveSyncAndFinishedCallbacks(t *testing.T) {
	g := newLiveGpu(t)
	if g.GLCaps().FenceType() == gl.FenceNone {
		t.Skip("no fence support")
	}
	sync := g.InsertSync()
	if !sync.IsValid() {
		t.Fatal("InsertSync failed")
	}
	g.FinishOutstandingGpuWork()
	if !g.TestSync(sync) {
		t.Error("sync not signaled after glFinish")
	}
	g.DeleteSync(sync)
	sweepGLErrors(t, g, "sync round trip")

	fired := false
	g.AddFinishedCallback(func() { fired = true })
	g.SubmitToGpu(true) // the flush_and_submit(syncCpu=true) semantics
	if !fired {
		t.Error("finished callback did not fire after synced submit")
	}
	sweepGLErrors(t, g, "finished callbacks")
}

// TestGpuLiveScratchReuseAndRelease verifies scratch reuse returns the same GL object and that release-and-abandon
// frees everything without GL errors.
func TestGpuLiveScratchReuseAndRelease(t *testing.T) {
	g := newLiveGpu(t)
	dims := geom.ISize{Width: 32, Height: 32}
	surf := g.CreateTexture(dims, gl.FormatRGBA8, gpu.TextureType2D, gpu.RenderableNo, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "scratch")
	if surf == nil {
		t.Fatal("CreateTexture failed")
	}
	texID := surf.AsTexture().TextureID()
	surf.Unref()

	var key gpu.ScratchKey
	gl.ComputeTextureScratchKey(g.GLCaps(), gl.FormatRGBA8, dims, gpu.RenderableNo, 1,
		gpu.MipmappedNo, &key)
	found := g.ResourceCache().FindAndRefScratchResource(&key)
	if found == nil {
		t.Fatal("scratch find failed")
	}
	reused := found.(*gl.Surface)
	if reused.AsTexture().TextureID() != texID {
		t.Errorf("scratch reuse returned texture %d, want %d", reused.AsTexture().TextureID(), texID)
	}
	reused.Unref()

	g.ReleaseResourcesAndAbandonContext()
	if g.ResourceCache().ResourceCount() != 0 {
		t.Errorf("cache count = %d after release", g.ResourceCache().ResourceCount())
	}
	if !reused.WasDestroyed() {
		t.Error("resource should be destroyed")
	}
	sweepGLErrors(t, g, "release-and-abandon")
}

// TestGpuLiveMipmapRegen uploads a base level and regenerates the mip chain.
func TestGpuLiveMipmapRegen(t *testing.T) {
	g := newLiveGpu(t)
	dims := geom.ISize{Width: 16, Height: 16}
	levels := gpu.ComputeMipLevelCount(dims.Width, dims.Height) + 1
	texels := make([]gpu.MipLevel, levels)
	texels[0] = gpu.MipLevel{Pixels: testPattern(dims, 3), RowBytes: int(dims.Width) * 4}

	surf := g.CreateTexture(dims, gl.FormatRGBA8, gpu.TextureType2D, gpu.RenderableNo, 1,
		gpu.BudgetedYes, levels, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, texels, "mips")
	if surf == nil {
		t.Fatal("CreateTexture failed")
	}
	tex := surf.AsTexture()
	if tex.Mipmapped() != gpu.MipmappedYes {
		t.Fatal("texture should be mipmapped")
	}
	if !tex.MipmapsAreDirty() {
		t.Fatal("mipmaps should start dirty")
	}
	if !g.RegenerateMipmapLevels(tex) {
		t.Fatal("RegenerateMipmapLevels failed")
	}
	if tex.MipmapsAreDirty() {
		t.Error("mipmaps still dirty after regeneration")
	}
	sweepGLErrors(t, g, "mipmap regen")

	// Content check: level 1 must be the 2x2 box average of the base level, whichever lane ran (glGenerateMipmap, or
	// the manual downsample draws on doManualMipmapping drivers). A small tolerance absorbs per-driver rounding.
	base := texels[0].Pixels
	half := dims.Width / 2
	fns := g.Functions()
	var fbo uint32
	fns.GenFramebuffers(1, &fbo)
	g.BindFramebuffer(gl.FRAMEBUFFER, fbo)
	fns.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex.TextureID(), 1)
	level1 := make([]byte, int(half)*int(half)*4)
	fns.PixelStorei(gl.PACK_ALIGNMENT, 1)
	fns.ReadPixels(0, 0, half, half, gl.RGBA, gl.UNSIGNED_BYTE, uintptr(unsafe.Pointer(&level1[0])))
	sweepGLErrors(t, g, "mip level 1 readback")
	const tolerance = 2
	for y := int32(0); y < half; y++ {
		for x := int32(0); x < half; x++ {
			for c := 0; c < 4; c++ {
				idx := func(px, py int32) int { return (int(py)*int(dims.Width) + int(px)) * 4 }
				want := (int(base[idx(2*x, 2*y)+c]) + int(base[idx(2*x+1, 2*y)+c]) +
					int(base[idx(2*x, 2*y+1)+c]) + int(base[idx(2*x+1, 2*y+1)+c]) + 2) / 4
				got := int(level1[(int(y)*int(half)+int(x))*4+c])
				if got < want-tolerance || got > want+tolerance {
					t.Fatalf("mip level 1 (%d,%d) channel %d = %d, want %d +/- %d", x, y, c, got,
						want, tolerance)
				}
			}
		}
	}
	fns.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, 0, 0)
	g.DeleteFramebuffer(fbo)

	surf.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}

// TestGpuLiveMipmapManualDrawLane forces the manual draw-based mipmap regeneration lane (createMipmapProgram +
// downsample draws) regardless of the driver's DoManualMipmapping caps, and verifies the produced level-1 is a correct
// 2x2 box average. The downsample draws read vertex positions from a_vertex; the box-average correctness confirms the
// fullscreen quad geometry flowed through the attribute the program binds to array location 0, which requires the
// attribute binding to be established before LinkProgram.
func TestGpuLiveMipmapManualDrawLane(t *testing.T) {
	g := newLiveGpu(t)
	if !g.GLCaps().IsFormatRenderable(gl.FormatRGBA8, 1) {
		t.Skip("RGBA8 not single-sample renderable; manual mipmap lane needs a renderable format")
	}
	dims := geom.ISize{Width: 16, Height: 16}
	levels := gpu.ComputeMipLevelCount(dims.Width, dims.Height) + 1
	texels := make([]gpu.MipLevel, levels)
	texels[0] = gpu.MipLevel{Pixels: testPattern(dims, 5), RowBytes: int(dims.Width) * 4}

	surf := g.CreateTexture(dims, gl.FormatRGBA8, gpu.TextureType2D, gpu.RenderableYes, 1,
		gpu.BudgetedYes, levels, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, texels, "manualmips")
	if surf == nil {
		t.Fatal("CreateTexture failed")
	}
	tex := surf.AsTexture()
	if !g.RegenerateMipmapLevelsByDrawsForTest(tex) {
		t.Fatal("manual mipmap draw lane failed")
	}
	sweepGLErrors(t, g, "manual mipmap draw lane")

	base := texels[0].Pixels
	half := dims.Width / 2
	fns := g.Functions()
	var fbo uint32
	fns.GenFramebuffers(1, &fbo)
	g.BindFramebuffer(gl.FRAMEBUFFER, fbo)
	fns.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex.TextureID(), 1)
	level1 := make([]byte, int(half)*int(half)*4)
	fns.PixelStorei(gl.PACK_ALIGNMENT, 1)
	fns.ReadPixels(0, 0, half, half, gl.RGBA, gl.UNSIGNED_BYTE, uintptr(unsafe.Pointer(&level1[0])))
	sweepGLErrors(t, g, "manual mip level 1 readback")
	const tolerance = 2
	for y := int32(0); y < half; y++ {
		for x := int32(0); x < half; x++ {
			for c := 0; c < 4; c++ {
				idx := func(px, py int32) int { return (int(py)*int(dims.Width) + int(px)) * 4 }
				want := (int(base[idx(2*x, 2*y)+c]) + int(base[idx(2*x+1, 2*y)+c]) +
					int(base[idx(2*x, 2*y+1)+c]) + int(base[idx(2*x+1, 2*y+1)+c]) + 2) / 4
				got := int(level1[(int(y)*int(half)+int(x))*4+c])
				if got < want-tolerance || got > want+tolerance {
					t.Fatalf("manual mip level 1 (%d,%d) channel %d = %d, want %d +/- %d", x, y, c,
						got, want, tolerance)
				}
			}
		}
	}
	fns.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, 0, 0)
	g.DeleteFramebuffer(fbo)

	surf.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}

// TestGpuLiveCapsDump logs the resource-layer-relevant caps for the running context, for parity with the checked-in
// driver dumps.
func TestGpuLiveCapsDump(t *testing.T) {
	g := newLiveGpu(t)
	caps := g.GLCaps()
	t.Logf("fenceType=%d mapBufferType=%d transferBufferType=%d invalidateBufferType=%d",
		caps.FenceType(), caps.MapBufferType(), caps.TransferBufferType(),
		caps.InvalidateBufferType())
}
