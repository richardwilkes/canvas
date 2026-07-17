// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the fill tessellation stack: PathTessellateOp's convex single-pass lane, PathStencilCoverOp's
// Redbook stencil-then-cover fills across all four fill types (including the inverse bbox lane),
// PathInnerTriangulateOp's three-pass lane on a large target, per-patch colors on a merged op, and the MSAA lane. All
// fills are matched against the Go CPU rasterizer (Go-GPU vs Go-CPU self-consistency). Skips when no GL context is
// available.

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

// liveTessConvexBlob is a volatile convex path with a curve: not a rect/oval/rrect, so the dedicated draw ops pass and
// the renderer chain (Tessellation for non-AA) takes it.
func liveTessConvexBlob(dx, dy float32) *path.Path {
	p := &path.Path{}
	p.MoveTo(15+dx, 35+dy)
	p.QuadTo(55+dx, -5+dy, 95+dx, 35+dy)
	p.LineTo(80+dx, 100+dy)
	p.LineTo(30+dx, 100+dy)
	p.Close()
	p.SetVolatile(true)
	return p
}

// liveTessCurvyStar is a volatile concave path mixing lines, quads and a cubic, scaled to the given size.
func liveTessCurvyStar(s float32) *path.Path {
	p := &path.Path{}
	p.MoveTo(0.50*s, 0.05*s)
	p.QuadTo(0.60*s, 0.35*s, 0.90*s, 0.38*s)
	p.LineTo(0.66*s, 0.56*s)
	p.CubicTo(0.80*s, 0.72*s, 0.72*s, 0.94*s, 0.62*s, 0.78*s)
	p.LineTo(0.50*s, 0.62*s)
	p.QuadTo(0.34*s, 0.90*s, 0.22*s, 0.90*s)
	p.LineTo(0.32*s, 0.58*s)
	p.LineTo(0.08*s, 0.36*s)
	p.QuadTo(0.36*s, 0.34*s, 0.50*s, 0.05*s)
	p.Close()
	p.SetVolatile(true)
	return p
}

func TestLiveTessellationConvexFill(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	blob := liveTessConvexBlob(10, 10)
	if !blob.IsConvex() {
		t.Fatal("test blob must be convex")
	}
	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		red := canvas.NewPaint()
		red.Color = 0xFFDD2200
		red.AntiAlias = false // volatile + non-AA selects TessellationPathRenderer
		c.DrawPath(blob, red)
	}

	cpuPix := raster.NewPixmap(w, h)
	scene(canvas.NewForPixmap(cpuPix))

	sdc := newLiveDrawSDC(t, dc, w, h)
	dev := gl.NewDevice(sdc)
	scene(canvas.New(dev))
	gpuData := readSDC(t, sdc)
	sdc.Release()

	// Non-AA rasterization-rule differences plus the 1/4px curve linearization stay confined to edge pixels.
	compareCPUGPU(t, gpuData, cpuPix, w, h, w*h/150, "tessellated convex blob")
}

func TestLiveTessellationStencilCoverFills(t *testing.T) {
	_, dc := newLiveDirectContext(t)
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
			p := liveTessCurvyStar(w)
			p.SetFillType(tc.ft)
			red := canvas.NewPaint()
			red.Color = 0xFFDD2200
			red.AntiAlias = false // volatile concave + non-AA selects PathStencilCoverOp
			c.DrawPath(p, red)
		}

		cpuPix := raster.NewPixmap(w, h)
		scene(canvas.NewForPixmap(cpuPix))

		sdc := newLiveDrawSDC(t, dc, w, h)
		dev := gl.NewDevice(sdc)
		scene(canvas.New(dev))
		gpuData := readSDC(t, sdc)
		sdc.Release()

		compareCPUGPU(t, gpuData, cpuPix, w, h, w*h/150, "stencil-cover "+tc.name)
	}
}

func TestLiveTessellationInnerTriangulate(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	// A large target: with few verbs and a big clipped draw area, makeNonConvexFillOp's heuristic picks
	// PathInnerTriangulateOp (asserted hermetically in TestNonConvexFillOpSelection).
	const w, h = 512, 512

	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		p := liveTessCurvyStar(w)
		blue := canvas.NewPaint()
		blue.Color = 0xFF2044CC
		blue.AntiAlias = false
		c.DrawPath(p, blue)
	}

	cpuPix := raster.NewPixmap(w, h)
	scene(canvas.NewForPixmap(cpuPix))

	sdc := newLiveDrawSDC(t, dc, w, h)
	dev := gl.NewDevice(sdc)
	scene(canvas.New(dev))
	gpuData := readSDC(t, sdc)
	sdc.Release()

	compareCPUGPU(t, gpuData, cpuPix, w, h, w*h/150, "inner-triangulate star")
}

func TestLiveTessellationMergedColors(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 224, 224
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	// Two convex volatile draws with identical trivial paints and different colors merge into one PathTessellateOp
	// (asserted hermetically in TestPathTessellateOpBatching); the merged op must write each path's own color through
	// the per-patch color attribute.
	identity := geom.IdentityMatrix()
	shape1 := gl.MakeStyledShapePath(liveTessConvexBlob(0, 0), gl.SimpleFillStyle(),
		gl.DoSimplifyNo)
	sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0.8, G: 0.1, B: 0, A: 1}), gpu.AANo,
		&identity, &shape1)
	shape2 := gl.MakeStyledShapePath(liveTessConvexBlob(105, 105), gl.SimpleFillStyle(),
		gl.DoSimplifyNo)
	sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0.2, B: 0.9, A: 1}), gpu.AANo,
		&identity, &shape2)

	data := readSDC(t, sdc)
	rb := w * 4
	probeAt(t, data, rb, 55, 60, [4]byte{204, 26, 0, 255}, 1, "merged blob 1 interior")
	probeAt(t, data, rb, 160, 165, [4]byte{0, 51, 230, 255}, 1, "merged blob 2 interior")
	probeAt(t, data, rb, 10, 215, [4]byte{255, 255, 255, 255}, 1, "outside both blobs")
}

func TestLiveTessellationExplicitCurveType(t *testing.T) {
	// The live desktop contexts report GLSL >= 3.30 (infinity support), so the explicit curve-type lane — what
	// production GL 3.2/GLSL 1.50 contexts take — would go uncompiled. Clearing InfinitySupport on a fresh context
	// forces the tessellators to write the kExplicitCurveType attrib and every tessellation shader to branch on it
	// instead of isinf(), across all three fill ops.
	env := newGLEnv(t)
	dc := gl.MakeGLDirectContext(env.intf, nil)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	dc.GLCaps().ShaderCaps.InfinitySupport = false

	const w, h = 512, 512
	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		red := canvas.NewPaint()
		red.Color = 0xFFDD2200
		red.AntiAlias = false
		// Full-size star: the inner-triangulate lane (HullShader + curve tessellator).
		c.DrawPath(liveTessCurvyStar(w), red)
		blue := canvas.NewPaint()
		blue.Color = 0xFF2044CC
		blue.AntiAlias = false
		// Convex blob: the PathTessellateOp wedge lane (MiddleOutShader with fan points).
		c.DrawPath(liveTessConvexBlob(390, 20), blue)
		green := canvas.NewPaint()
		green.Color = 0xFF11AA33
		green.AntiAlias = false
		// A small star: clipped fragment work below the triangulation threshold, so the stencil-cover lane
		// (MiddleOutShader wedges + BoundingBoxShader).
		small := liveTessCurvyStar(100)
		tm := geom.TranslateMatrix(20, 390)
		small.Transform(&tm)
		small.SetVolatile(true)
		c.DrawPath(small, green)
	}

	cpuPix := raster.NewPixmap(w, h)
	scene(canvas.NewForPixmap(cpuPix))

	sdc := newLiveDrawSDC(t, dc, w, h)
	dev := gl.NewDevice(sdc)
	scene(canvas.New(dev))
	gpuData := readSDC(t, sdc)
	sdc.Release()

	compareCPUGPU(t, gpuData, cpuPix, w, h, w*h/150, "explicit curve-type lanes")
}

func TestLiveTessellationMSAA(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gpu.BackingFitExact, 4, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "live-tess-msaa")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()
	if sdc.NumSamples() <= 1 {
		t.Skip("driver does not support MSAA render targets")
	}

	sdc.Clear([4]float32{1, 1, 1, 1})
	identity := geom.IdentityMatrix()
	star := liveTessCurvyStar(w) // volatile: the MSAA fill selects TessellationPathRenderer
	shape := gl.MakeStyledShapePath(star, gl.SimpleFillStyle(), gl.DoSimplifyNo)
	sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0.8, G: 0.1, B: 0, A: 1}), gpu.AAYes,
		&identity, &shape)

	data := readSDC(t, sdc)
	rb := w * 4
	probeAt(t, data, rb, 52, 40, [4]byte{204, 26, 0, 255}, 2, "msaa star interior")
	probeAt(t, data, rb, 8, 120, [4]byte{255, 255, 255, 255}, 1, "msaa star exterior")

	// Somewhere along the star's edges there must be a genuinely partial pixel after the MSAA resolve.
	partial := false
	for y := int32(30); y < 100 && !partial; y++ {
		for x := int32(10); x < 118; x++ {
			off := int(y)*rb + int(x)*4
			g := data[off+1]
			if g > 40 && g < 215 {
				partial = true
				break
			}
		}
	}
	if !partial {
		t.Fatal("no partially covered pixel found along the star edges under MSAA")
	}
}
