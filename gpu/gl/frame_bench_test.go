// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GPU-frame benchmarks: render a representative UI-like frame through the GPU canvas device — gl.Device → canvas draw
// ops → OpsTask recording → DrawingManager flush → GL draw calls — and report the two first-class GPU outputs:
// allocations per frame (the GC-pressure gate on the recording layer) and the batching metric (primitives recorded vs
// GL draw calls issued, via DirectContext.NumGLDraws). These complement the CPU-side canvas/bench_test.go suite. The
// benchmarks skip when no GL context is available, exactly like the gpu/gl live tests. Run e.g.:
//
//	go test ./gpu/gl/ -run '^$' -bench BenchmarkGLFrame -benchmem
//
// A warm-up frame precedes the timed loop so first-frame shader compiles and texture uploads are excluded and the
// reported allocs/frame and GLdraws/frame reflect the steady state.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/surface"
)

// benchPanels draws a batching-friendly UI-like frame: an opaque background fill, then a grid of filled rounded-rect
// "panels" each with a stroked border. The panel fills lower to instanced FillRRectOp and the borders to StrokeRectOp,
// both of which batch across the grid, so a well-batched frame issues far fewer GL draw calls than the 2*rows*cols
// primitives recorded — the batching metric the GLdraws/frame report exposes. The paints are provided by the caller
// (built once, outside the timed loop) so the measured allocations are the device/op/flush pipeline's, not per-frame
// paint construction; the per-cell fill color is mutated in place to exercise the per-instance color attribute.
func benchPanels(c *canvas.Canvas, bg, fill, border *canvas.Paint, side float32, rows, cols int) {
	c.DrawPaint(bg)
	const pad = 6
	cw := (side - pad) / float32(cols)
	ch := (side - pad) / float32(rows)
	for r := 0; r < rows; r++ {
		for k := 0; k < cols; k++ {
			x := pad + float32(k)*cw
			y := pad + float32(r)*ch
			rect := geom.RectLTRB(x, y, x+cw-pad, y+ch-pad)
			fill.Color = colorcore.Color(0xFF000000 | uint32((r*37+k*53)&0xFF)<<16 |
				uint32((r*91+k*17)&0xFF)<<8 | uint32((r*13+k*71)&0xFF))
			c.DrawRoundRect(rect, 5, 5, fill)
			c.DrawRect(rect, border)
		}
	}
}

func newFramePanelPaints() (bg, fill, border *canvas.Paint) {
	bg = canvas.NewPaint()
	bg.Color = 0xFF202830
	fill = canvas.NewPaint()
	fill.AntiAlias = true
	border = canvas.NewPaint()
	border.AntiAlias = true
	border.Style = canvas.StyleStroke
	border.StrokeWidth = 2
	border.Color = 0xFF3060A0
	return bg, fill, border
}

// BenchmarkGLFramePanels measures the batching-friendly rrect/stroke grid frame. The GLdraws/frame metric shows how few
// GL draw calls the grid batches into; prims/frame is the number of canvas primitives recorded, so prims/frame ÷
// GLdraws/frame is the batching factor.
func BenchmarkGLFramePanels(b *testing.B) {
	_, dc := newLiveDirectContext(b)
	const side = 256
	const rows, cols = 6, 6
	sdc := newLiveDrawSDC(b, dc, side, side)
	defer sdc.Release()
	c := canvas.New(gl.NewDevice(sdc))
	bg, fill, border := newFramePanelPaints()

	// Warm the program/resource caches with one full frame so the measurement is steady state.
	benchPanels(c, bg, fill, border, side, rows, cols)
	dc.FlushAndSubmit(false)
	if dc.NumGLDraws() == 0 {
		b.Fatal("warm-up frame issued no GL draws — the frame did not render")
	}

	dc.ResetGLDrawStats()
	dc.Gpu().Functions().ResetCallCounts()
	pcBefore := dc.Gpu().ProgramCacheStats()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPanels(c, bg, fill, border, side, rows, cols)
		dc.FlushAndSubmit(false)
	}
	b.StopTimer()
	reportFrameMetrics(b, dc, pcBefore, 2*rows*cols+1)
}

// BenchmarkGLFrameScene measures the heavier mixed frame reused from the device scene test (sceneCanvas): translucent
// rrect fill, AA stroked rect, filled circle, a concave star (the software-mask path lane), a scaled image, a stroked
// line, and a saveLayer-alpha group. It exposes the per-frame allocations of the layer/SW-mask/image lanes that the
// panel frame does not touch.
func BenchmarkGLFrameScene(b *testing.B) {
	_, dc := newLiveDirectContext(b)
	const side = 128
	img := newCheckerImage(b)
	sdc := newLiveDrawSDC(b, dc, side, side)
	defer sdc.Release()
	c := canvas.New(gl.NewDevice(sdc))

	sceneCanvas(c, img)
	dc.FlushAndSubmit(false)
	if dc.NumGLDraws() == 0 {
		b.Fatal("warm-up frame issued no GL draws — the frame did not render")
	}

	dc.ResetGLDrawStats()
	dc.Gpu().Functions().ResetCallCounts()
	pcBefore := dc.Gpu().ProgramCacheStats()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sceneCanvas(c, img)
		dc.FlushAndSubmit(false)
	}
	b.StopTimer()
	reportFrameMetrics(b, dc, pcBefore, -1)
}

// benchPathFrame draws an AA-path-heavy frame: a background fill plus a grid of concave AA curved stars (five points
// whose edges are quadratics, a self-intersecting path), the content class DMSAA exists for — under coverage AA every
// star pays linearization + triangulation per frame, while under DMSAA the paths ride multisampled stencil-and-cover
// passes.
func benchPathFrame(c *canvas.Canvas, bg, fill *canvas.Paint, side float32, rows, cols int) {
	c.DrawPaint(bg)
	cw := side / float32(cols)
	ch := side / float32(rows)
	for r := 0; r < rows; r++ {
		for k := 0; k < cols; k++ {
			x := float32(k) * cw
			y := float32(r) * ch
			p := path.New()
			p.MoveTo(x+cw*0.50, y+ch*0.06)
			p.QuadTo(x+cw*0.60, y+ch*0.40, x+cw*0.65, y+ch*0.70)
			p.QuadTo(x+cw*0.35, y+ch*0.45, x+cw*0.10, y+ch*0.30)
			p.QuadTo(x+cw*0.50, y+ch*0.25, x+cw*0.90, y+ch*0.30)
			p.QuadTo(x+cw*0.60, y+ch*0.45, x+cw*0.35, y+ch*0.70)
			p.Close()
			fill.Color = colorcore.Color(0xFF000000 | uint32((r*37+k*53)&0xFF)<<16 |
				uint32((r*91+k*17)&0xFF)<<8 | uint32((r*13+k*71)&0xFF))
			c.DrawPath(p, fill)
		}
	}
}

// benchGLFramePaths is the shared body of the DMSAA on/off path-frame pair, measuring what DMSAA buys: identical
// content, the only difference being surface.DynamicMSAAFlag on the surface props.
func benchGLFramePaths(b *testing.B, dynamicMSAA bool) {
	_, dc := newLiveDirectContext(b)
	const side = 256
	const rows, cols = 6, 6
	var props *surface.Props
	if dynamicMSAA {
		props = &surface.Props{Flags: surface.DynamicMSAAFlag}
	}
	sdc := gl.MakeSurfaceDrawContextWithProps(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: side, Height: side}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "bench-paths", props)
	if sdc == nil {
		b.Fatal("MakeSurfaceDrawContextWithProps failed")
	}
	if dynamicMSAA && !sdc.CanUseDynamicMSAA() {
		b.Fatal("DMSAA not supported on this context")
	}
	defer sdc.Release()
	c := canvas.New(gl.NewDevice(sdc))
	bg := canvas.NewPaint()
	bg.Color = 0xFF202830
	fill := canvas.NewPaint()
	fill.AntiAlias = true

	benchPathFrame(c, bg, fill, side, rows, cols)
	dc.FlushAndSubmit(false)
	if dc.NumGLDraws() == 0 {
		b.Fatal("warm-up frame issued no GL draws — the frame did not render")
	}

	dc.ResetGLDrawStats()
	dc.Gpu().Functions().ResetCallCounts()
	pcBefore := dc.Gpu().ProgramCacheStats()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPathFrame(c, bg, fill, side, rows, cols)
		dc.FlushAndSubmit(false)
	}
	b.StopTimer()
	reportFrameMetrics(b, dc, pcBefore, rows*cols+1)
}

// BenchmarkGLFramePathsAA measures the AA-path-heavy frame with DMSAA off (coverage AA).
func BenchmarkGLFramePathsAA(b *testing.B) { benchGLFramePaths(b, false) }

// BenchmarkGLFramePathsDMSAA measures the same frame with DMSAA on.
func BenchmarkGLFramePathsDMSAA(b *testing.B) { benchGLFramePaths(b, true) }

// reportFrameMetrics attaches the GPU frame's first-class outputs: GL draw calls issued per frame (the batching
// metric), total GL calls issued per frame (the B.1 FFI-pressure metric — every call pays purego.SyscallN's marshaling
// allocations, so this is the number state shadowing drives down) and, when prims >= 0, the primitives recorded per
// frame (so the batching factor is prims÷draws), plus the steady-state program-cache hit rate (should be ~1.0 after
// warm-up — a lower value means the cache is thrashing). The caller must reset both stats (ResetGLDrawStats,
// ResetCallCounts) right before the timed loop.
func reportFrameMetrics(b *testing.B, dc *gl.DirectContext, pcBefore gl.PipelineStats, prims int) {
	b.Helper()
	frames := float64(b.N)
	b.ReportMetric(float64(dc.NumGLDraws())/frames, "GLdraws/frame")
	b.ReportMetric(float64(dc.Gpu().Functions().TotalCallCount())/frames, "GLcalls/frame")
	if prims >= 0 {
		b.ReportMetric(float64(prims), "prims/frame")
	}
	pc := dc.Gpu().ProgramCacheStats()
	hits := pc.NumProgramCacheHits - pcBefore.NumProgramCacheHits
	misses := pc.NumProgramCacheMisses - pcBefore.NumProgramCacheMisses
	if total := hits + misses; total > 0 {
		b.ReportMetric(float64(hits)/float64(total), "progcache-hit")
	}
}
