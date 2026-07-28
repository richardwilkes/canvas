// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the path-renderer chain: renderer selection order, DefaultPathRenderer's stencil-and-cover
// fills (winding, even-odd, inverse) and hairline lanes compared against the Go CPU rasterizer (Go-GPU vs Go-CPU
// self-consistency), the SoftwarePathRenderer's unique-key mask cache (hit/miss semantics incl. the
// fractional-translate key bits and the volatile no-key lane), stencil path clips through StencilMaskHelper.DrawPath,
// and the MSAA lane. Skips when no GL context is available.

package gl_test

import (
	"testing"
	"unsafe"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// liveStarPath is a self-intersecting five-point star (concave — no dedicated op or analytic FP can take it, so it
// exercises the renderer chain).
func liveStarPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(50.3, 10.2)
	p.LineTo(70.1, 90.4)
	p.LineTo(10.2, 40.3)
	p.LineTo(90.4, 40.3)
	p.LineTo(30.1, 90.2)
	p.Close()
	return p
}

// liveBigConcavePath is a concave zig-zag with more than 10 verbs: past the AA tessellator's verb limit, but still
// keyed (via the source path's generation ID).
func liveBigConcavePath() *path.Path {
	p := &path.Path{}
	p.MoveTo(5, 5)
	for i := 1; i <= 12; i++ {
		y := float32(6 + (i%2)*70)
		p.LineTo(float32(5+i*9), y)
	}
	p.Close()
	return p
}

func TestLivePathRendererSelection(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 128, 128)
	defer sdc.Release()

	identity := geom.IdentityMatrix()
	bounds := geom.IRectSize(sdc.Dimensions())

	makeArgs := func(shape *gl.StyledShape, aaType gpu.AAType) *gl.CanDrawPathArgs {
		return &gl.CanDrawPathArgs{
			Caps:                   sdc.Caps(),
			Proxy:                  sdc.AsRenderTargetProxy(),
			ClipConservativeBounds: bounds,
			ViewMatrix:             &identity,
			Shape:                  shape,
			AAType:                 aaType,
		}
	}

	fillShape := gl.MakeStyledShapePath(liveStarPath(), gl.SimpleFillStyle(), gl.DoSimplifyYes)

	// Non-AA concave fill of a keyed shape: TriangulatingPathRenderer takes it (caching lane).
	pr := dc.DrawingManager().GetPathRenderer(makeArgs(&fillShape, gpu.AATypeNone), false,
		gl.ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "Triangulating" {
		t.Fatalf("non-AA fill selected %v, want Triangulating", pr)
	}
	// MSAA: same.
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&fillShape, gpu.AATypeMSAA), false,
		gl.ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "Triangulating" {
		t.Fatalf("MSAA fill selected %v, want Triangulating", pr)
	}

	// Coverage AA under the AA-tessellator verb limit: Triangulating's alpha-ramp lane.
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&fillShape, gpu.AATypeCoverage), false,
		gl.ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "Triangulating" {
		t.Fatalf("coverage-AA fill selected %v, want Triangulating", pr)
	}

	// A volatile path has no unstyled key: Triangulating's non-AA/MSAA lanes reject it (no caching benefit) and
	// TessellationPathRenderer, which doesn't care about keys, takes it.
	volStar := liveStarPath()
	volStar.SetVolatile(true)
	volShape := gl.MakeStyledShapePath(volStar, gl.SimpleFillStyle(), gl.DoSimplifyYes)
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&volShape, gpu.AATypeNone), false,
		gl.ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "Tessellation" {
		t.Fatalf("non-AA volatile fill selected %v, want Tessellation", pr)
	}

	// Coverage AA past the AA-tessellator verb limit (> 10 verbs): nothing in the chain takes it; the SW renderer
	// accepts as the fallback.
	big := liveBigConcavePath()
	bigShape := gl.MakeStyledShapePath(big, gl.SimpleFillStyle(), gl.DoSimplifyYes)
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&bigShape, gpu.AATypeCoverage), false,
		gl.ChainDrawTypeColor, nil)
	if pr != nil {
		t.Fatalf("coverage-AA big fill selected %s from the chain, want none", pr.Name())
	}
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&bigShape, gpu.AATypeCoverage), true,
		gl.ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "SW" {
		t.Fatalf("coverage-AA big fill with allowSW selected %v, want SW", pr)
	}

	// A coverage-AA style that still applies is rejected by the whole chain (the SDC applies the style and retries);
	// non-AA strokes are claimed by the tessellation renderer's stroke lane.
	fillStyle := gl.SimpleFillStyle()
	rec := fillStyle.Rec()
	rec.SetStrokeStyle(5, false)
	strokedShape := gl.MakeStyledShapePath(liveStarPath(), gl.MakeStyle(*rec), gl.DoSimplifyYes)
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&strokedShape, gpu.AATypeCoverage), true,
		gl.ChainDrawTypeColor, nil)
	if pr != nil {
		t.Fatalf("coverage-AA styled shape selected %s, want none until the style is applied",
			pr.Name())
	}
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&strokedShape, gpu.AATypeNone), false,
		gl.ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "Tessellation" {
		t.Fatalf("non-AA stroke selected %v, want Tessellation", pr)
	}

	// Hairline styles ride the tessellation renderer's stroke lane for non-AA.
	hairShape := gl.MakeStyledShapePath(liveStarPath(), gl.SimpleHairlineStyle(),
		gl.DoSimplifyYes)
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&hairShape, gpu.AATypeNone), false,
		gl.ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "Tessellation" {
		t.Fatalf("non-AA hairline selected %v, want Tessellation", pr)
	}

	// Stencil draw type: TessellationPathRenderer sits ahead of DefaultPathRenderer and reports two-pass (stencil-only)
	// support for concave fills and no-restriction for convex ones.
	var support gl.StencilSupport
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&fillShape, gpu.AATypeNone), false,
		gl.ChainDrawTypeStencil, &support)
	if pr == nil || pr.Name() != "Tessellation" || support != gl.StencilSupportStencilOnly {
		t.Fatalf("concave stencil selection = %v/%d, want Tessellation/kStencilOnly", pr, support)
	}
	convex := &path.Path{}
	convex.MoveTo(10, 10).LineTo(90, 20).LineTo(50, 80).Close()
	convexShape := gl.MakeStyledShapePath(convex, gl.SimpleFillStyle(), gl.DoSimplifyYes)
	pr = dc.DrawingManager().GetPathRenderer(makeArgs(&convexShape, gpu.AATypeNone), false,
		gl.ChainDrawTypeStencil, &support)
	if pr == nil || pr.Name() != "Tessellation" || support != gl.StencilSupportNoRestriction {
		t.Fatalf("convex stencil selection = %v/%d, want Tessellation/kNoRestriction", pr, support)
	}
}

// compareCPUGPU compares a GPU readback against a CPU pixmap, budgeting a small fraction of differing pixels for
// rasterization-rule differences along edges.
func compareCPUGPU(t *testing.T, gpuData []byte, cpuPix *raster.Pixmap, w, h, budget int, tag string) {
	t.Helper()
	cpuData := unsafe.Slice((*byte)(unsafe.Pointer(&cpuPix.Pix[0])), len(cpuPix.Pix)*4)
	rb := w * 4
	diff := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := y*rb + x*4
			for c := 0; c < 4; c++ {
				d := int(gpuData[off+c]) - int(cpuData[off+c])
				if d < -1 || d > 1 {
					diff++
					break
				}
			}
		}
	}
	if diff > budget {
		t.Fatalf("%s: %d pixels differ (budget %d)", tag, diff, budget)
	}
	t.Logf("%s: %d differing pixels (budget %d)", tag, diff, budget)
}

func TestLiveDefaultPathRendererFills(t *testing.T) {
	// Disable the tessellation renderer so volatile non-AA fills stay on DefaultPathRenderer.
	env := newGLEnv(t)
	opts := gpu.DefaultContextOptions()
	opts.PathRenderers &^= gpu.PathRenderersTessellation
	dc := gl.MakeGLDirectContext(env.intf, opts)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	const w, h = 128, 128

	fillTypes := []struct {
		name string
		ft   path.FillType
	}{
		{name: "winding", ft: path.FillWinding},
		{name: "evenodd", ft: path.FillEvenOdd},
		{name: "inverse-winding", ft: path.FillInverseWinding},
		{name: "inverse-evenodd", ft: path.FillInverseEvenOdd},
	}
	for _, tc := range fillTypes {
		scene := func(c *canvas.Canvas) {
			bg := canvas.NewPaint()
			bg.Color = 0xFFFFFFFF
			c.DrawPaint(bg)
			p := liveStarPath()
			p.SetFillType(tc.ft)
			// No unstyled key: with Tessellation disabled above, non-AA stays on DefaultPathRenderer.
			p.SetVolatile(true)
			red := canvas.NewPaint()
			red.Color = 0xFFDD2200
			red.AntiAlias = false
			c.DrawPath(p, red)
		}

		cpuPix := raster.NewPixmap(w, h)
		scene(canvas.NewForPixmap(cpuPix))

		sdc := newLiveDrawSDC(t, dc, w, h)
		dev := gl.NewDevice(sdc)
		scene(canvas.New(dev))
		gpuData := readSDC(t, sdc)
		sdc.Release()

		// Non-AA scan conversion differs only where edges land within rounding distance of pixel centers; budget a thin
		// sliver of the edge pixels.
		compareCPUGPU(t, gpuData, cpuPix, w, h, w*h/200, tc.name)

		rb := w * 4
		inStar := [4]byte{221, 34, 0, 255}
		outStar := [4]byte{255, 255, 255, 255}
		inside, spike, outside := inStar, inStar, outStar
		if tc.ft == path.FillEvenOdd || tc.ft == path.FillInverseEvenOdd {
			// The central pentagon has winding count two: covered by winding, not by even-odd.
			inside = outStar
		}
		if tc.ft.IsInverse() {
			flip := func(c [4]byte) [4]byte {
				if c == inStar {
					return outStar
				}
				return inStar
			}
			inside = flip(inside)
			spike = flip(spike)
			outside = flip(outside)
		}
		probeAt(t, gpuData, rb, 50, 48, inside, 1, tc.name+" center")
		probeAt(t, gpuData, rb, 50, 22, spike, 1, tc.name+" top spike")
		probeAt(t, gpuData, rb, 8, 100, outside, 1, tc.name+" outside")
	}
}

func TestLiveDefaultPathRendererHairline(t *testing.T) {
	// Disable the tessellation renderer so the hairline stays on DefaultPathRenderer's line lane (the tessellation
	// stroke lane claims hairlines; its own hairline coverage lives in TestLiveStrokeTessellationHairline).
	env := newGLEnv(t)
	opts := gpu.DefaultContextOptions()
	opts.PathRenderers &^= gpu.PathRenderersTessellation
	dc := gl.MakeGLDirectContext(env.intf, opts)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	const w, h = 96, 64
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	dev := gl.NewDevice(sdc)
	c := canvas.New(dev)
	p := &path.Path{}
	p.MoveTo(10, 20.5)
	p.LineTo(60, 20.5)
	pen := canvas.NewPaint()
	pen.Color = 0xFF0000CC
	pen.Style = canvas.StyleStroke
	pen.StrokeWidth = 0 // hairline
	pen.AntiAlias = false
	c.DrawPath(p, pen)

	data := readSDC(t, sdc)
	rb := w * 4
	// All colored pixels must sit in row 20, spanning roughly x in [10, 60).
	colored := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := y*rb + x*4
			if data[off] != 255 || data[off+1] != 255 || data[off+2] != 255 {
				if y != 20 {
					t.Fatalf("hairline pixel off-row at (%d,%d)", x, y)
				}
				if x < 9 || x > 60 {
					t.Fatalf("hairline pixel out of span at (%d,%d)", x, y)
				}
				colored++
			}
		}
	}
	if colored < 45 || colored > 52 {
		t.Fatalf("hairline covered %d pixels, want ~50", colored)
	}
}

func TestLiveSWPathMaskCaching(t *testing.T) {
	// Disable the triangulating renderer so coverage-AA concave fills reach the SW mask lane.
	env := newGLEnv(t)
	opts := gpu.DefaultContextOptions()
	opts.PathRenderers &^= gpu.PathRenderersTriangulating
	dc := gl.MakeGLDirectContext(env.intf, opts)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	const w, h = 128, 128
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()

	stats := dc.Stats()
	*stats = gl.ContextStats{}

	blue := colorcore.PMColor4f{R: 0, G: 0.2, B: 0.9, A: 1}
	src := liveStarPath() // small path: keyed from its data

	draw := func(m geom.Matrix) []byte {
		shape := gl.MakeStyledShapePath(src, gl.SimpleFillStyle(), gl.DoSimplifyNo)
		sdc.Clear([4]float32{1, 1, 1, 1})
		sdc.DrawShape(nil, livePaint(blue), gpu.AAYes, &m, &shape)
		return readSDC(t, sdc)
	}

	identity := geom.IdentityMatrix()
	data := draw(identity)
	if stats.NumPathMasksGenerated != 1 || stats.NumPathMasksCacheHits != 0 {
		t.Fatalf("after first draw: generated %d, hits %d", stats.NumPathMasksGenerated,
			stats.NumPathMasksCacheHits)
	}
	probeAt(t, data, w*4, 50, 48, [4]byte{0, 51, 230, 255}, 2, "sw mask interior")

	// Identical draw: cache hit.
	data = draw(identity)
	if stats.NumPathMasksGenerated != 1 || stats.NumPathMasksCacheHits != 1 {
		t.Fatalf("after repeat draw: generated %d, hits %d", stats.NumPathMasksGenerated,
			stats.NumPathMasksCacheHits)
	}
	probeAt(t, data, w*4, 50, 48, [4]byte{0, 51, 230, 255}, 2, "cached mask interior")

	// Integer translation: the fractional-translate key bits are unchanged — still a hit.
	data = draw(geom.TranslateMatrix(6, 9))
	if stats.NumPathMasksGenerated != 1 || stats.NumPathMasksCacheHits != 2 {
		t.Fatalf("after integer translate: generated %d, hits %d",
			stats.NumPathMasksGenerated, stats.NumPathMasksCacheHits)
	}
	probeAt(t, data, w*4, 56, 57, [4]byte{0, 51, 230, 255}, 2, "translated mask interior")

	// Fractional translation: new subpixel phase, new mask.
	_ = draw(geom.TranslateMatrix(0.37, 0))
	if stats.NumPathMasksGenerated != 2 || stats.NumPathMasksCacheHits != 2 {
		t.Fatalf("after fractional translate: generated %d, hits %d",
			stats.NumPathMasksGenerated, stats.NumPathMasksCacheHits)
	}

	// Scaling changes the 2x2: new mask.
	scale := geom.ScaleMatrix(1.25, 1.25)
	_ = draw(scale)
	if stats.NumPathMasksGenerated != 3 {
		t.Fatalf("after scale: generated %d", stats.NumPathMasksGenerated)
	}

	// Volatile paths have no unstyled key and are never cached.
	vol := liveStarPath()
	vol.SetVolatile(true)
	drawVol := func() {
		shape := gl.MakeStyledShapePath(vol, gl.SimpleFillStyle(), gl.DoSimplifyNo)
		sdc.DrawShape(nil, livePaint(blue), gpu.AAYes, &identity, &shape)
		_ = readSDC(t, sdc)
	}
	drawVol()
	drawVol()
	if stats.NumPathMasksGenerated != 5 || stats.NumPathMasksCacheHits != 2 {
		t.Fatalf("after volatile draws: generated %d, hits %d", stats.NumPathMasksGenerated,
			stats.NumPathMasksCacheHits)
	}

	// Large paths (> 10 verbs) key on the source path's generation ID: reusing the same source hits, rebuilding the
	// same geometry misses.
	big := &path.Path{}
	big.MoveTo(5, 5)
	for i := 1; i <= 12; i++ {
		y := float32(6 + (i%2)*70)
		big.LineTo(float32(5+i*9), y)
	}
	big.Close()
	drawBig := func(p *path.Path) {
		shape := gl.MakeStyledShapePath(p, gl.SimpleFillStyle(), gl.DoSimplifyNo)
		sdc.DrawShape(nil, livePaint(blue), gpu.AAYes, &identity, &shape)
		_ = readSDC(t, sdc)
	}
	drawBig(big)
	drawBig(big)
	if stats.NumPathMasksGenerated != 6 || stats.NumPathMasksCacheHits != 3 {
		t.Fatalf("after gen-ID draws: generated %d, hits %d", stats.NumPathMasksGenerated,
			stats.NumPathMasksCacheHits)
	}
}

func TestLiveStencilPathClip(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()

	dev := gl.NewDevice(sdc)
	c := canvas.New(dev)

	bg := canvas.NewPaint()
	bg.Color = 0xFFFFFFFF
	c.DrawPaint(bg)

	// A non-AA concave path clip cannot be expressed as an analytic FP; with AA off it takes the stencil lane
	// (StencilMaskHelper.DrawPath through the renderer chain).
	c.Save()
	c.ClipPath(liveStarPath(), raster.ClipIntersect, false)
	red := canvas.NewPaint()
	red.Color = 0xFFCC2200
	c.DrawPaint(red)
	c.Restore()

	data := readSDC(t, sdc)
	rb := w * 4
	probeAt(t, data, rb, 50, 48, [4]byte{204, 34, 0, 255}, 1, "clip interior (pentagon)")
	probeAt(t, data, rb, 50, 22, [4]byte{204, 34, 0, 255}, 1, "clip interior (spike)")
	probeAt(t, data, rb, 8, 100, [4]byte{255, 255, 255, 255}, 1, "clip exterior")
	probeAt(t, data, rb, 50, 95, [4]byte{255, 255, 255, 255}, 1, "clip exterior below")

	// Difference op takes the two-pass user-to-clip lane.
	c.Save()
	c.ClipPath(liveStarPath(), raster.ClipDifference, false)
	blue := canvas.NewPaint()
	blue.Color = 0xFF0022CC
	c.DrawPaint(blue)
	c.Restore()

	data = readSDC(t, sdc)
	probeAt(t, data, rb, 50, 48, [4]byte{204, 34, 0, 255}, 1, "difference keeps interior")
	probeAt(t, data, rb, 8, 100, [4]byte{0, 34, 204, 255}, 1, "difference covers exterior")
}

func TestLiveDefaultPathRendererMSAA(t *testing.T) {
	// Disable the tessellation renderer so the volatile MSAA fill stays on DefaultPathRenderer.
	env := newGLEnv(t)
	opts := gpu.DefaultContextOptions()
	opts.PathRenderers &^= gpu.PathRenderersTessellation
	dc := gl.MakeGLDirectContext(env.intf, opts)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	const w, h = 128, 128
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gpu.BackingFitExact, 4, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "live-path-msaa")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()
	if sdc.NumSamples() <= 1 {
		t.Skip("driver does not support MSAA render targets")
	}

	sdc.Clear([4]float32{1, 1, 1, 1})
	identity := geom.IdentityMatrix()
	star := liveStarPath()
	star.SetVolatile(true) // no unstyled key: the MSAA lane stays on DefaultPathRenderer
	shape := gl.MakeStyledShapePath(star, gl.SimpleFillStyle(), gl.DoSimplifyNo)
	sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0.8, G: 0.1, B: 0, A: 1}), gpu.AAYes,
		&identity, &shape)

	data := readSDC(t, sdc)
	rb := w * 4
	probeAt(t, data, rb, 50, 48, [4]byte{204, 26, 0, 255}, 2, "msaa star interior")
	probeAt(t, data, rb, 8, 100, [4]byte{255, 255, 255, 255}, 1, "msaa star exterior")

	// Somewhere along the left spike edge there must be a genuinely partial pixel after the MSAA resolve.
	partial := false
	for y := int32(50); y < 80 && !partial; y++ {
		for x := int32(10); x < 50; x++ {
			off := int(y)*rb + int(x)*4
			g := data[off+1]
			if g > 40 && g < 215 {
				partial = true
				break
			}
		}
	}
	if !partial {
		t.Fatal("no partially covered pixel found along the star edge under MSAA")
	}
}
