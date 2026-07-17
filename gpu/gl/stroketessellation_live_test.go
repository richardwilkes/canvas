// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the stroke tessellation stack: StrokeTessellateOp's parametric/radial edge shader across
// joins, caps, curves with cusps, closed contours, the hairline lane, dynamic stroke/color merging, the
// stencil-then-fill lane for translucent strokes, the explicit-curve-type lane, and the MSAA lane. Strokes are matched
// against the Go CPU stroker + rasterizer (Go-GPU vs Go-CPU self-consistency). Skips when no GL context is available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// liveStrokeCurvyPath is a volatile open path mixing a line, quad, and cubic.
func liveStrokeCurvyPath(dx, dy float32) *path.Path {
	p := &path.Path{}
	p.MoveTo(15+dx, 100+dy)
	p.LineTo(30+dx, 40+dy)
	p.QuadTo(60+dx, 10+dy, 90+dx, 45+dy)
	p.CubicTo(110+dx, 70+dy, 70+dx, 90+dy, 100+dx, 110+dy)
	p.SetVolatile(true)
	return p
}

// runStrokeScene renders the scene through the CPU rasterizer and the live GPU device and compares them within the
// given differing-pixel budget.
func runStrokeScene(t *testing.T, dc *gl.DirectContext, w, h, budget int, tag string, scene func(c *canvas.Canvas)) {
	t.Helper()
	cpuPix := raster.NewPixmap(int32(w), int32(h))
	scene(canvas.NewForPixmap(cpuPix))

	sdc := newLiveDrawSDC(t, dc, int32(w), int32(h))
	dev := gl.NewDevice(sdc)
	scene(canvas.New(dev))
	gpuData := readSDC(t, sdc)
	sdc.Release()

	compareCPUGPU(t, gpuData, cpuPix, w, h, budget, tag)
}

func strokeScenePaint(color colorcore.Color, width float32, capType canvas.StrokeCap, join canvas.StrokeJoin) *canvas.Paint {
	pen := canvas.NewPaint()
	pen.Color = color
	pen.Style = canvas.StyleStroke
	pen.StrokeWidth = width
	pen.Cap = capType
	pen.Join = join
	pen.AntiAlias = false // non-AA selects TessellationPathRenderer's stroke lane
	return pen
}

func TestLiveStrokeTessellationJoinsAndCaps(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	cases := []struct {
		name string
		cap  canvas.StrokeCap
		join canvas.StrokeJoin
	}{
		{name: "round-round", cap: canvas.CapRound, join: canvas.JoinRound},
		{name: "butt-miter", cap: canvas.CapButt, join: canvas.JoinMiter},
		{name: "square-bevel", cap: canvas.CapSquare, join: canvas.JoinBevel},
	}
	for _, tc := range cases {
		scene := func(c *canvas.Canvas) {
			bg := canvas.NewPaint()
			bg.Color = 0xFFFFFFFF
			c.DrawPaint(bg)
			c.DrawPath(liveStrokeCurvyPath(0, 0),
				strokeScenePaint(0xFFDD2200, 11, tc.cap, tc.join))
			// A zig-zag exercising sharp joins.
			zig := &path.Path{}
			zig.MoveTo(15, 15).LineTo(45, 25).LineTo(20, 45).LineTo(60, 60)
			zig.SetVolatile(true)
			c.DrawPath(zig, strokeScenePaint(0xFF2044CC, 7, tc.cap, tc.join))
		}
		// Non-AA rasterization-rule differences plus the 1/4px edge linearization stay confined to pixels along the
		// stroke boundaries.
		runStrokeScene(t, dc, w, h, w*h/40, tc.name, scene)
	}
}

func TestLiveStrokeTessellationClosedContour(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		tri := &path.Path{}
		tri.MoveTo(25, 100).LineTo(64, 20).QuadTo(80, 40, 103, 100).Close()
		tri.SetVolatile(true)
		c.DrawPath(tri, strokeScenePaint(0xFF11AA33, 9, canvas.CapButt, canvas.JoinRound))
	}
	runStrokeScene(t, dc, w, h, w*h/40, "closed contour", scene)
}

func TestLiveStrokeTessellationCusp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		// A flat cubic with a 180-degree turnaround: the tessellator chops at the cusp and covers it with a circle
		// patch.
		cusp := &path.Path{}
		cusp.MoveTo(20, 64)
		cusp.CubicTo(120, 64, 8, 64, 108, 64)
		cusp.SetVolatile(true)
		c.DrawPath(cusp, strokeScenePaint(0xFFDD2200, 10, canvas.CapButt, canvas.JoinBevel))
		// A flat quad with a 180-degree turnaround.
		qc := &path.Path{}
		qc.MoveTo(20, 100)
		qc.QuadTo(120, 100, 20, 100)
		qc.SetVolatile(true)
		c.DrawPath(qc, strokeScenePaint(0xFF2044CC, 8, canvas.CapRound, canvas.JoinRound))
	}
	runStrokeScene(t, dc, w, h, w*h/40, "cusps", scene)
}

func TestLiveStrokeTessellationHairline(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 96, 64
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	dev := gl.NewDevice(sdc)
	c := canvas.New(dev)
	p := &path.Path{}
	p.MoveTo(10, 20.5)
	p.LineTo(60, 20.5)
	p.SetVolatile(true)
	pen := strokeScenePaint(0xFF0000CC, 0 /* hairline */, canvas.CapButt, canvas.JoinMiter)
	c.DrawPath(p, pen)

	// The hairline lane transforms before tessellation and sweeps a half-pixel radius: all colored pixels must sit in
	// row 20, spanning roughly x in [10, 60).
	data := readSDC(t, sdc)
	rb := w * 4
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

func TestLiveStrokeTessellationDynamicMerge(t *testing.T) {
	// Strokes with differing widths and colors in one scene merge into a single instanced op with dynamic stroke params
	// and colors; the rendering must still match the CPU.
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		seg1 := &path.Path{}
		seg1.MoveTo(15, 20).LineTo(60, 30).QuadTo(90, 40, 110, 25)
		seg1.SetVolatile(true)
		c.DrawPath(seg1, strokeScenePaint(0xFFDD2200, 5, canvas.CapButt, canvas.JoinBevel))
		seg2 := &path.Path{}
		seg2.MoveTo(15, 60).LineTo(60, 75).QuadTo(90, 85, 110, 65)
		seg2.SetVolatile(true)
		c.DrawPath(seg2, strokeScenePaint(0xFF2044CC, 11, canvas.CapButt, canvas.JoinBevel))
	}
	runStrokeScene(t, dc, w, h, w*h/40, "dynamic merge", scene)
}

func TestLiveStrokeTessellationTranslucentStencil(t *testing.T) {
	// A translucent stroke takes the stencil-then-fill lane: self-intersections and the overlapping quad-strip edges
	// must not double blend.
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		loop := &path.Path{}
		loop.MoveTo(20, 30)
		loop.CubicTo(110, 30, 20, 90, 100, 95)
		loop.SetVolatile(true)
		c.DrawPath(loop, strokeScenePaint(0x80DD2200, 13, canvas.CapRound, canvas.JoinRound))
	}
	runStrokeScene(t, dc, w, h, w*h/40, "translucent stencil", scene)
}

func TestLiveStrokeTessellationExplicitCurveType(t *testing.T) {
	// Clearing InfinitySupport on a fresh context forces the stroke tessellator to write the kExplicitCurveType attrib
	// and the shader to branch on it instead of isinf() — the GL 3.2/GLSL-1.50 production lane.
	env := newGLEnv(t)
	dc := gl.MakeGLDirectContext(env.intf, nil)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	dc.GLCaps().ShaderCaps.InfinitySupport = false

	const w, h = 128, 128
	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		c.DrawPath(liveStrokeCurvyPath(0, 0),
			strokeScenePaint(0xFFDD2200, 9, canvas.CapRound, canvas.JoinRound))
		// A conic (arc) so the explicit conic curve type is exercised too.
		arc := &path.Path{}
		arc.MoveTo(20, 115)
		arc.ConicTo(64, 80, 108, 115, 0.7)
		arc.SetVolatile(true)
		c.DrawPath(arc, strokeScenePaint(0xFF2044CC, 7, canvas.CapButt, canvas.JoinMiter))
	}
	runStrokeScene(t, dc, w, h, w*h/40, "explicit curve type", scene)
}

func TestLiveStrokeTessellationMSAA(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gpu.BackingFitExact, 4, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "live-stroke-msaa")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()
	if sdc.NumSamples() <= 1 {
		t.Skip("driver does not support MSAA render targets")
	}

	sdc.Clear([4]float32{1, 1, 1, 1})
	dev := gl.NewDevice(sdc)
	c := canvas.New(dev)
	pen := strokeScenePaint(0xFFCC1A00, 12, canvas.CapRound, canvas.JoinRound)
	pen.AntiAlias = true // MSAA target + AA selects the MSAA stroke lane
	diag := &path.Path{}
	diag.MoveTo(20, 100).LineTo(100, 30)
	diag.SetVolatile(true)
	c.DrawPath(diag, pen)

	data := readSDC(t, sdc)
	rb := w * 4
	probeAt(t, data, rb, 60, 65, [4]byte{204, 26, 0, 255}, 2, "msaa stroke interior")
	probeAt(t, data, rb, 10, 10, [4]byte{255, 255, 255, 255}, 1, "msaa stroke exterior")

	// Somewhere along the stroke edge there must be a genuinely partial pixel after the MSAA resolve.
	partial := false
	for y := 20; y < 110 && !partial; y++ {
		for x := 10; x < 110; x++ {
			off := y*rb + x*4
			g := data[off+1]
			if g > 40 && g < 215 {
				partial = true
				break
			}
		}
	}
	if !partial {
		t.Fatal("no partially covered pixel found along the stroke edge under MSAA")
	}
}
