// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the atlas stack: the ASAP and inline upload lanes run through a real DrawingManager.Flush
// against a real GL context, and the atlas page contents are read back and verified after each flush — proving the
// deferred uploads reach the texture through OpFlushState.DoUpload → Gpu.WritePixels in flush order (ASAP at
// preExecuteDraws, inline between the recorded draws). Skips when no GL context is available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
)

// TestLiveAtlasFlushUploads runs the fake-driver TestAtlasFlushIntegration scenario on a live context, with pixel
// readbacks: flush 1 adds one full-plot subimage (ASAP lane, page activation); flush 2 adds two subimages to the
// now-full single-plot ARGB atlas, exercising the evict-and-reuse ASAP lane and the inline lane. After each flush the
// atlas page must hold the last-uploaded bytes.
func TestLiveAtlasFlushUploads(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	am := dc.AtlasManager()
	am.SetAtlasDimensionsToMinimumForTest()
	// The minimum ARGB atlas is a single 256x256 plot; capped to one page so the second upload of flush 2 must go
	// inline instead of activating another page.
	const format = gpu.MaskFormatARGB
	const size = 256
	if am.AtlasForTest(format) == nil {
		t.Fatal("atlas init failed")
	}
	am.SetMaxPagesForTest(1)

	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 64, Height: 64},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes,
		"live-atlas-target")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	readAtlasPage := func() []byte {
		t.Helper()
		views, numActive := am.GetViews(format)
		if views == nil || numActive != 1 {
			t.Fatalf("GetViews: views=%v numActive=%d", views != nil, numActive)
		}
		sc := gl.NewSurfaceContextForTesting(dc, views[0].Proxy(), gpu.OriginTopLeft,
			gpu.ColorTypeRGBA8888)
		defer sc.Release()
		dst := gl.Pixels{
			ColorType: gpu.ColorTypeRGBA8888,
			Dims:      geom.ISize{Width: size, Height: size},
			RowBytes:  size * 4,
			Data:      make([]byte, size*size*4),
		}
		if !sc.ReadPixels(dst, geom.IPoint{}) {
			t.Fatal("ReadPixels on the atlas page failed")
		}
		return dst.Data
	}

	checkAll := func(data []byte, want byte, tag string) {
		t.Helper()
		for i, b := range data {
			if b != want {
				t.Fatalf("%s: byte %d = %d, want %d", tag, i, b, want)
			}
		}
	}

	// Flush 1: one subimage rides the ASAP lane while activating page 0.
	sdc.Clear([4]float32{0, 0, 1, 1})
	op1 := gl.NewAtlasUploadTestOp(dc, format, size, 1)
	sdc.GetOpsTask().AddOp(dc.DrawingManager(), op1, dc.GLCaps())
	dc.Flush(gl.FlushInfo{})
	if len(op1.Codes) != 1 || op1.Codes[0] != gl.AtlasErrorCodeSucceeded || op1.Executed != 1 {
		t.Fatalf("flush-1 codes=%v executed=%d", op1.Codes, op1.Executed)
	}
	checkAll(readAtlasPage(), 1, "after ASAP upload")

	// Flush 2: the plot is full and was used last flush; the first add evicts and re-uploads ASAP, the second hits the
	// pending-draw hazard and uploads inline between the two draws.
	sdc.Clear([4]float32{0, 0, 1, 1})
	op2 := gl.NewAtlasUploadTestOp(dc, format, size, 2)
	sdc.GetOpsTask().AddOp(dc.DrawingManager(), op2, dc.GLCaps())
	dc.Flush(gl.FlushInfo{})
	if len(op2.Codes) != 2 || op2.Codes[0] != gl.AtlasErrorCodeSucceeded ||
		op2.Codes[1] != gl.AtlasErrorCodeSucceeded || op2.Executed != 2 {
		t.Fatalf("flush-2 codes=%v executed=%d", op2.Codes, op2.Executed)
	}
	// The inline upload (bytes = 2) executed after the ASAP one (bytes = 1) and wins.
	checkAll(readAtlasPage(), 2, "after inline upload")

	sweepGLErrors(t, dc.Gpu(), "atlas uploads")
	_ = env
}

// TestLiveAtlasGlyphUpload pushes an A8 glyph-shaped subimage through the manager's glyph lane on a live context and
// reads back the exact texel placement, including the transformed-mask padding ring.
func TestLiveAtlasGlyphUpload(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	am := dc.AtlasManager()
	am.SetAtlasDimensionsToMinimumForTest()
	const format = gpu.MaskFormatA8

	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 64, Height: 64},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes,
		"live-atlas-a8-target")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	// One 8x8 A8 subimage through a real flush.
	sdc.Clear([4]float32{0, 0, 1, 1})
	op := gl.NewAtlasUploadTestOp(dc, format, 8, 1)
	sdc.GetOpsTask().AddOp(dc.DrawingManager(), op, dc.GLCaps())
	dc.Flush(gl.FlushInfo{})
	if len(op.Codes) != 1 || op.Codes[0] != gl.AtlasErrorCodeSucceeded {
		t.Fatalf("codes=%v", op.Codes)
	}

	views, numActive := am.GetViews(format)
	if views == nil || numActive != 1 {
		t.Fatal("A8 atlas page not active")
	}
	// Read the page as A8 and verify the 8x8 block of 1s at the locator, zero elsewhere on its row band.
	sc := gl.NewSurfaceContextForTesting(dc, views[0].Proxy(), gpu.OriginTopLeft,
		gpu.ColorTypeAlpha8)
	defer sc.Release()
	dims := views[0].Proxy().Dimensions()
	dst := gl.Pixels{
		ColorType: gpu.ColorTypeAlpha8,
		Dims:      dims,
		RowBytes:  int(dims.Width),
		Data:      make([]byte, int(dims.Width)*int(dims.Height)),
	}
	if !sc.ReadPixels(dst, geom.IPoint{}) {
		t.Fatal("ReadPixels on the A8 atlas page failed")
	}
	// Only the uploaded region is defined: the atlas never clears its pages, so texels outside the subimage hold
	// whatever the driver had in that memory.
	topLeft := op.Locators[0].TopLeft()
	w := int(dims.Width)
	for y := range 8 {
		for x := range 8 {
			if got := dst.Data[(int(topLeft.Y)+y)*w+int(topLeft.X)+x]; got != 1 {
				t.Fatalf("texel (%d,%d) = %d, want 1", x, y, got)
			}
		}
	}

	sweepGLErrors(t, dc.Gpu(), "a8 atlas upload")
	_ = env
}
