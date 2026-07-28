// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the proxy/render-task/drawing-manager layer: clear + readback scenarios running through the
// full DirectContext → SurfaceDrawContext → OpsTask → DrawingManager flush pipeline on a real GL context, including the
// wrapped-FBO path, MSAA resolve tasks, write-pixels task ordering,
// and copy tasks. Pixel expectations are computed on the CPU (clears and copies are exact); comparison against the
// oracle GPU goldens lives in internal/oracle/gorender (TestGoGPUvsSelfCapturedGolden), not here. Skips when no GL
// context is available.

package gl_test

import (
	"testing"
	"unsafe"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
)

func newLiveDirectContext(t testing.TB) (*glEnv, *gl.DirectContext) {
	t.Helper()
	env := newGLEnv(t)
	dc := gl.MakeGLDirectContext(env.intf, nil)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	return env, dc
}

// readSDC reads the full contents of a draw context as RGBA8888 bytes.
func readSDC(t *testing.T, sdc *gl.SurfaceDrawContext) []byte {
	t.Helper()
	dims := sdc.Dimensions()
	dst := gl.Pixels{
		ColorType: gpu.ColorTypeRGBA8888,
		Dims:      dims,
		RowBytes:  int(dims.Width) * 4,
		Data:      make([]byte, int(dims.Width)*int(dims.Height)*4),
	}
	if !sdc.ReadPixels(dst, geom.IPoint{}) {
		t.Fatal("ReadPixels failed")
	}
	return dst.Data
}

func checkPixel(t *testing.T, data []byte, rowBytes int, x, y int32, want [4]byte, tag string) {
	t.Helper()
	off := int(y)*rowBytes + int(x)*4
	got := [4]byte{data[off], data[off+1], data[off+2], data[off+3]}
	for c := 0; c < 4; c++ {
		d := int(got[c]) - int(want[c])
		if d < -1 || d > 1 {
			t.Fatalf("%s: pixel (%d,%d) = %v, want %v", tag, x, y, got, want)
		}
	}
}

// TestLiveClearReadback drives the offscreen path end-to-end: fullscreen clear (load op lane), scissored clear (ClearOp
// lane), flush via the task DAG, readback.
func TestLiveClearReadback(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	dims := geom.ISize{Width: 64, Height: 64}

	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "live-clear-target")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 1, 1})                                   // blue everywhere
	sdc.ClearRect(geom.IRectXYWH(8, 8, 16, 16), [4]float32{1, 0, 0, 1}) // red window

	data := readSDC(t, sdc)
	rb := int(dims.Width) * 4
	checkPixel(t, data, rb, 0, 0, [4]byte{0, 0, 255, 255}, "outside")
	checkPixel(t, data, rb, 63, 63, [4]byte{0, 0, 255, 255}, "outside far corner")
	checkPixel(t, data, rb, 8, 8, [4]byte{255, 0, 0, 255}, "inside top-left")
	checkPixel(t, data, rb, 23, 23, [4]byte{255, 0, 0, 255}, "inside bottom-right")
	checkPixel(t, data, rb, 24, 24, [4]byte{0, 0, 255, 255}, "just outside")
	sweepGLErrors(t, dc.Gpu(), "clear+readback")
	_ = env
}

// TestLiveWrappedFBOClearReadback wraps a caller-owned FBO into a proxy-backed draw context with a bottom-left origin,
// clears through the pipeline, and reads it back (exercising the flip lane).
func TestLiveWrappedFBOClearReadback(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	const size = 64
	fbo, _ := env.makeOffscreenFBO(t, size)
	dims := geom.ISize{Width: size, Height: size}

	sdc := gl.MakeSurfaceDrawContextFromBackendRenderTarget(dc, gpu.ColorTypeRGBA8888, dims,
		gl.FormatRGBA8, 1, 0, fbo, gpu.OriginBottomLeft, nil)
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContextFromBackendRenderTarget failed")
	}
	defer sdc.Release()
	if !sdc.AsSurfaceProxy().IsInstantiated() {
		t.Fatal("wrapped proxy should be instantiated")
	}
	if sdc.AsSurfaceProxy().UseAllocator() != gl.UseAllocatorNo {
		t.Fatal("wrapped proxies must not participate in allocation")
	}

	sdc.Clear([4]float32{0, 1, 0, 1})                                 // green everywhere
	sdc.ClearRect(geom.IRectXYWH(0, 0, 8, 8), [4]float32{1, 0, 1, 1}) // magenta at the canvas's top-left

	data := readSDC(t, sdc)
	rb := size * 4
	checkPixel(t, data, rb, 2, 2, [4]byte{255, 0, 255, 255}, "wrapped top-left")
	checkPixel(t, data, rb, 2, 12, [4]byte{0, 255, 0, 255}, "wrapped below window")
	checkPixel(t, data, rb, 60, 60, [4]byte{0, 255, 0, 255}, "wrapped bottom-right")
	sweepGLErrors(t, dc.Gpu(), "wrapped clear+readback")

	// Verify the flip actually landed in GL space: the canvas's top-left window must occupy the GL bottom rows of the
	// raw framebuffer.
	f := &env.intf.Functions
	f.BindFramebuffer(gl.FRAMEBUFFER, fbo)
	raw := make([]byte, size*size*4)
	f.ReadPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, uintptr(unsafe.Pointer(&raw[0])))
	dc.ResetContext(gl.AllBackendState) // we touched GL behind the context's back
	// glReadPixels rows run bottom-to-top: the canvas's top-left window sits in the high GL rows.
	checkPixel(t, raw, rb, 2, 58, [4]byte{255, 0, 255, 255}, "GL-space high rows")
	checkPixel(t, raw, rb, 2, 2, [4]byte{0, 255, 0, 255}, "GL-space low rows")
}

// TestLiveWritePixelsThenClear verifies write-pixels tasks stay ordered against later clears targeting the same proxy.
func TestLiveWritePixelsThenClear(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	dims := geom.ISize{Width: 16, Height: 16}
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "live-write-target")
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	// Upload an 8x8 white block at (0,0)...
	src := gl.Pixels{
		ColorType: gpu.ColorTypeRGBA8888,
		Dims:      geom.ISize{Width: 8, Height: 8},
		RowBytes:  8 * 4,
		Data:      make([]byte, 8*8*4),
	}
	for i := range src.Data {
		src.Data[i] = 0xFF
	}
	if !sdc.WritePixels(src, geom.IPoint{}) {
		t.Fatal("WritePixels failed")
	}
	// ...then clear its right half red. The clear is recorded after the write and must win.
	sdc.ClearRect(geom.IRectXYWH(4, 0, 4, 8), [4]float32{1, 0, 0, 1})

	data := readSDC(t, sdc)
	rb := int(dims.Width) * 4
	checkPixel(t, data, rb, 1, 1, [4]byte{255, 255, 255, 255}, "written block")
	checkPixel(t, data, rb, 5, 1, [4]byte{255, 0, 0, 255}, "clear over write")
	checkPixel(t, data, rb, 10, 10, [4]byte{0, 0, 0, 255}, "background")
	sweepGLErrors(t, dc.Gpu(), "write+clear+readback")
}

// TestLiveCopyTask runs a CopyRenderTask through the DAG: clear a source, copy a sub-rect into a fresh proxy, and read
// the copy back.
func TestLiveCopyTask(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	dims := geom.ISize{Width: 32, Height: 32}
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "live-copy-src")
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 1, 1})
	sdc.ClearRect(geom.IRectXYWH(0, 0, 8, 8), [4]float32{1, 1, 0, 1})

	copyProxy, copyTask := gl.CopySurfaceProxy(dc, sdc.AsSurfaceProxy(), gpu.OriginTopLeft,
		gpu.MipmappedNo, geom.IRectXYWH(0, 0, 16, 16), gpu.BackingFitExact, gpu.BudgetedYes,
		"live-copy-dst")
	if copyProxy == nil || copyTask == nil {
		t.Fatal("CopySurfaceProxy failed")
	}
	defer copyProxy.Unref()

	scCopy := gl.NewSurfaceContextForTesting(dc, copyProxy, gpu.OriginTopLeft,
		gpu.ColorTypeRGBA8888)
	defer scCopy.Release()

	dst := gl.Pixels{
		ColorType: gpu.ColorTypeRGBA8888,
		Dims:      geom.ISize{Width: 16, Height: 16},
		RowBytes:  16 * 4,
		Data:      make([]byte, 16*16*4),
	}
	if !scCopy.ReadPixels(dst, geom.IPoint{}) {
		t.Fatal("ReadPixels from copy failed")
	}
	checkPixel(t, dst.Data, 16*4, 2, 2, [4]byte{255, 255, 0, 255}, "copied window")
	checkPixel(t, dst.Data, 16*4, 12, 12, [4]byte{0, 0, 255, 255}, "copied background")
	sweepGLErrors(t, dc.Gpu(), "copy+readback")
}

// TestLiveMSAAResolveReadback drives the manual-MSAA-resolve machinery: a 4-sample draw context's clear must resolve
// through a TextureResolveRenderTask (or the flush-surfaces resolve) before readback.
func TestLiveMSAAResolveReadback(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	if dc.GLCaps().GetRenderTargetSampleCount(4, gl.FormatRGBA8) < 4 {
		t.Skip("no 4-sample RGBA8 support")
	}
	dims := geom.ISize{Width: 32, Height: 32}
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 4,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "live-msaa-target")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext (MSAA) failed")
	}
	defer sdc.Release()
	if !sdc.AsSurfaceProxy().RequiresManualMSAAResolve() {
		t.Fatal("MSAA texture render target should require manual resolve")
	}

	sdc.Clear([4]float32{1, 0, 1, 1})
	data := readSDC(t, sdc)
	rb := int(dims.Width) * 4
	checkPixel(t, data, rb, 0, 0, [4]byte{255, 0, 255, 255}, "msaa resolved")
	checkPixel(t, data, rb, 31, 31, [4]byte{255, 0, 255, 255}, "msaa resolved far")
	if sdc.AsSurfaceProxy().AsRenderTargetProxy().IsMSAADirty() {
		t.Fatal("readback should have resolved the MSAA dirty state")
	}
	sweepGLErrors(t, dc.Gpu(), "msaa clear+resolve+readback")
}

// TestLiveContextDestroy: teardown flushes, finishes, and releases every resource with no GL errors left behind.
func TestLiveContextDestroy(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	dims := geom.ISize{Width: 16, Height: 16}
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "live-destroy-target")
	sdc.Clear([4]float32{1, 0, 0, 1})
	dc.FlushAndSubmit(true)
	sdc.Release()

	dc.Destroy()
	if got := dc.ResourceCache().ResourceCount(); got != 0 {
		t.Fatalf("Destroy left %d resources", got)
	}
	sweepGLErrors(t, dc.Gpu(), "destroy")
}
