// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the AA path renderers: AAConvexPathRenderer's coverage-AA convex fills (line and curved
// boundaries), AALinearizingConvexPathRenderer's convex strokes / stroke-and-fills, and AAHairLinePathRenderer's
// line/quad/conic/cubic hairlines including the perspective lane and record-time batching, all matched against the Go
// CPU rasterizer for self-consistency. The AA coverage models legitimately differ along edges (the GPU renderers use
// distance approximations, the CPU rasterizer analytic area coverage), so scenes with solid interiors compare with an
// edge-band pixel budget, and hairline scenes compare structurally (ink-set size and overlap). Skips when no GL context
// is available.

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

// livePentagonPath is a convex pentagon that no dedicated draw op takes.
func livePentagonPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(64, 10)
	p.LineTo(115, 47)
	p.LineTo(96, 110)
	p.LineTo(32, 110)
	p.LineTo(13, 47)
	p.Close()
	return p
}

// liveConvexCurvedPath is a convex rounded diamond built from quads (curved boundary segments exercise the AAConvex
// quad-segment lane).
func liveConvexCurvedPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(64, 12)
	p.QuadTo(108, 20, 116, 64)
	p.QuadTo(108, 108, 64, 116)
	p.QuadTo(20, 108, 12, 64)
	p.QuadTo(20, 20, 64, 12)
	p.Close()
	return p
}

// renderAAScene renders the scene through the CPU rasterizer and the live GPU device, returning both RGBA byte buffers.
func renderAAScene(t *testing.T, dc *gl.DirectContext, w, h int, scene func(c *canvas.Canvas)) (gpuData []byte, cpuPix *raster.Pixmap) {
	t.Helper()
	cpuPix = raster.NewPixmap(int32(w), int32(h))
	scene(canvas.NewForPixmap(cpuPix))

	sdc := newLiveDrawSDC(t, dc, int32(w), int32(h))
	dev := gl.NewDevice(sdc)
	scene(canvas.New(dev))
	gpuData = readSDC(t, sdc)
	sdc.Release()
	return gpuData, cpuPix
}

// pixmapBytes exposes a pixmap's RGBA bytes.
func pixmapBytes(cpuPix *raster.Pixmap) []byte {
	out := make([]byte, len(cpuPix.Pix)*4)
	for i, px := range cpuPix.Pix {
		out[4*i+0] = byte(px)
		out[4*i+1] = byte(px >> 8)
		out[4*i+2] = byte(px >> 16)
		out[4*i+3] = byte(px >> 24)
	}
	return out
}

// compareInk structurally compares two AA renderings over a white background: the sets of inked pixels (differing from
// white) must have similar sizes, and each side's strongly-inked pixels must be inked (within a 1px dilation) on the
// other side. This is robust to the legitimate per-pixel coverage-model differences of the AA renderers while catching
// missing, misplaced, or malformed geometry.
func compareInk(t *testing.T, gpuData, cpuData []byte, w, h int, tag string) {
	t.Helper()
	const inkThreshold = 8 // channel delta from white to count as ink
	ink := func(data []byte, x, y int) (isInk, isStrong bool) {
		off := (y*w + x) * 4
		maxDelta := 0
		for c := 0; c < 4; c++ {
			d := 255 - int(data[off+c])
			if d < 0 {
				d = -d
			}
			if d > maxDelta {
				maxDelta = d
			}
		}
		return maxDelta > inkThreshold, maxDelta > 128
	}
	dilatedInk := func(data []byte, x, y int) bool {
		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				nx, ny := x+dx, y+dy
				if nx < 0 || ny < 0 || nx >= w || ny >= h {
					continue
				}
				if isInk, _ := ink(data, nx, ny); isInk {
					return true
				}
			}
		}
		return false
	}

	gpuInk, cpuInk := 0, 0
	gpuStrongMissed, cpuStrongMissed := 0, 0
	gpuStrong, cpuStrong := 0, 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gi, gs := ink(gpuData, x, y)
			ci, cs := ink(cpuData, x, y)
			if gi {
				gpuInk++
			}
			if ci {
				cpuInk++
			}
			if gs {
				gpuStrong++
				if !dilatedInk(cpuData, x, y) {
					gpuStrongMissed++
				}
			}
			if cs {
				cpuStrong++
				if !dilatedInk(gpuData, x, y) {
					cpuStrongMissed++
				}
			}
		}
	}
	if cpuInk == 0 {
		t.Fatalf("%s: CPU reference rendered nothing", tag)
	}
	if gpuInk < cpuInk/2 || gpuInk > cpuInk*2 {
		t.Fatalf("%s: ink-set sizes diverge (gpu %d, cpu %d)", tag, gpuInk, cpuInk)
	}
	if gpuStrongMissed > gpuStrong/20 || cpuStrongMissed > cpuStrong/20 {
		t.Fatalf("%s: strong ink not mutually covered (gpu missed %d/%d, cpu missed %d/%d)",
			tag, gpuStrongMissed, gpuStrong, cpuStrongMissed, cpuStrong)
	}
	t.Logf("%s: gpu ink %d, cpu ink %d, strong misses %d/%d", tag, gpuInk, cpuInk,
		gpuStrongMissed, cpuStrongMissed)
}

// expectAAPixel asserts an exact RGBA value at (x, y).
func expectAAPixel(t *testing.T, data []byte, w, x, y int, want [4]byte, tag string) {
	t.Helper()
	off := (y*w + x) * 4
	for c := 0; c < 4; c++ {
		if data[off+c] != want[c] {
			t.Fatalf("%s: pixel (%d,%d) = %v want %v", tag, x, y, data[off:off+4], want)
		}
	}
}

func aaSelectionName(t *testing.T, dc *gl.DirectContext, sdc *gl.SurfaceDrawContext, shape *gl.StyledShape, viewMatrix *geom.Matrix) string {
	t.Helper()
	args := &gl.CanDrawPathArgs{
		Caps:                   sdc.Caps(),
		Proxy:                  sdc.AsRenderTargetProxy(),
		ClipConservativeBounds: geom.IRectSize(sdc.Dimensions()),
		ViewMatrix:             viewMatrix,
		Shape:                  shape,
		AAType:                 gpu.AATypeCoverage,
	}
	pr := dc.DrawingManager().GetPathRenderer(args, false, gl.ChainDrawTypeColor, nil)
	if pr == nil {
		return "<none>"
	}
	return pr.Name()
}

func TestLiveAAConvexFills(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	// Selection: convex coverage-AA fills go to AAConvex.
	sdc := newLiveDrawSDC(t, dc, w, h)
	identity := geom.IdentityMatrix()
	shape := gl.MakeStyledShapePath(livePentagonPath(), gl.SimpleFillStyle(), gl.DoSimplifyYes)
	if got := aaSelectionName(t, dc, sdc, &shape, &identity); got != "AAConvex" {
		t.Fatalf("convex coverage fill selects %q want AAConvex", got)
	}
	sdc.Release()

	cases := []struct {
		make func() *path.Path
		name string
		inX  int
		inY  int
	}{
		{name: "pentagon", make: livePentagonPath, inX: 64, inY: 60},
		{name: "curved", make: liveConvexCurvedPath, inX: 64, inY: 64},
	}
	for _, tc := range cases {
		scene := func(c *canvas.Canvas) {
			bg := canvas.NewPaint()
			bg.Color = 0xFFFFFFFF
			c.DrawPaint(bg)
			red := canvas.NewPaint()
			red.Color = 0xFFDD2200
			red.AntiAlias = true
			c.DrawPath(tc.make(), red)
		}
		gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
		// Interiors are solid and must match exactly; the AA edge band may differ pixel by pixel between the
		// distance-based GPU coverage and the CPU's analytic coverage.
		expectAAPixel(t, gpuData, w, tc.inX, tc.inY, [4]byte{0xDD, 0x22, 0x00, 0xFF},
			"aaconvex "+tc.name+" interior")
		expectAAPixel(t, gpuData, w, 2, 2, [4]byte{0xFF, 0xFF, 0xFF, 0xFF},
			"aaconvex "+tc.name+" exterior")
		compareCPUGPU(t, gpuData, cpuPix, w, h, 1400, "aaconvex "+tc.name)
	}
}

func TestLiveAAConvexBatching(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	identity := geom.IdentityMatrix()
	for range 2 {
		shape := gl.MakeStyledShapePath(livePentagonPath(), gl.SimpleFillStyle(),
			gl.DoSimplifyYes)
		sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0.5, A: 0.5}), gpu.AAYes,
			&identity, &shape)
	}
	if n := sdc.GetOpsTask().NumOpChains(); n != 1 {
		t.Fatalf("chain count %d want 1 (AAConvex draws should merge)", n)
	}
	// The merged op must still render correctly end to end.
	data := readSDC(t, sdc)
	off := (60*w + 64) * 4
	if data[off+3] != 0xFF {
		t.Fatalf("merged AAConvex draw interior alpha %d want 255", data[off+3])
	}
}

func TestLiveAALinearizingStrokes(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	cases := []struct {
		name string
		join canvas.StrokeJoin
	}{
		{name: "miter", join: canvas.JoinMiter},
		{name: "bevel", join: canvas.JoinBevel},
	}
	for _, tc := range cases {
		scene := func(c *canvas.Canvas) {
			bg := canvas.NewPaint()
			bg.Color = 0xFFFFFFFF
			c.DrawPaint(bg)
			pen := canvas.NewPaint()
			pen.Color = 0xFF0044CC
			pen.Style = canvas.StyleStroke
			pen.StrokeWidth = 5
			pen.Join = tc.join
			pen.AntiAlias = true
			c.DrawPath(livePentagonPath(), pen)
		}
		gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
		compareInk(t, gpuData, pixmapBytes(cpuPix), w, h, "aalinear stroke "+tc.name)
		// The stroke band center on the top-right edge is solid: (89,28)'s pixel center is the midpoint of the
		// (64,10)-(115,47) edge, so the 5px-wide band covers it completely and it must be pure stroke color.
		expectAAPixel(t, gpuData, w, 89, 28, [4]byte{0x00, 0x44, 0xCC, 0xFF},
			"aalinear stroke "+tc.name+" band")
		// The pentagon's unstroked interior stays background white.
		expectAAPixel(t, gpuData, w, 64, 64, [4]byte{0xFF, 0xFF, 0xFF, 0xFF},
			"aalinear stroke "+tc.name+" hole")
	}
}

func TestLiveAALinearizingStrokeAndFill(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		pen := canvas.NewPaint()
		pen.Color = 0xFF008833
		pen.Style = canvas.StyleStrokeAndFill
		pen.StrokeWidth = 4
		pen.Join = canvas.JoinBevel
		pen.AntiAlias = true
		c.DrawPath(livePentagonPath(), pen)
	}
	gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
	// Interior is solid fill.
	expectAAPixel(t, gpuData, w, 64, 60, [4]byte{0x00, 0x88, 0x33, 0xFF},
		"aalinear stroke-and-fill interior")
	compareCPUGPU(t, gpuData, cpuPix, w, h, 1600, "aalinear stroke-and-fill")
}

func TestLiveAAHairlines(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	// Selection: coverage-AA hairlines go to AAHairline.
	sdc := newLiveDrawSDC(t, dc, w, h)
	identity := geom.IdentityMatrix()
	hairShape := gl.MakeStyledShapePath(liveStarPath(), gl.SimpleHairlineStyle(),
		gl.DoSimplifyYes)
	if got := aaSelectionName(t, dc, sdc, &hairShape, &identity); got != "AAHairline" {
		t.Fatalf("coverage hairline selects %q want AAHairline", got)
	}
	sdc.Release()

	cases := []struct {
		make func() *path.Path
		name string
	}{
		{name: "lines", make: liveStarPath},
		{name: "quad", make: func() *path.Path {
			p := &path.Path{}
			p.MoveTo(10, 100)
			p.QuadTo(64, 8, 118, 100)
			return p
		}},
		{name: "cubic", make: func() *path.Path {
			p := &path.Path{}
			p.MoveTo(10, 64)
			p.CubicTo(40, 8, 88, 120, 118, 64)
			return p
		}},
		{name: "conic", make: func() *path.Path {
			p := &path.Path{}
			p.MoveTo(10, 100)
			p.ConicTo(64, 8, 118, 100, 0.7)
			return p
		}},
	}
	for _, tc := range cases {
		scene := func(c *canvas.Canvas) {
			bg := canvas.NewPaint()
			bg.Color = 0xFFFFFFFF
			c.DrawPaint(bg)
			pen := canvas.NewPaint()
			pen.Color = 0xFF112288
			pen.Style = canvas.StyleStroke
			pen.StrokeWidth = 0 // hairline
			pen.AntiAlias = true
			c.DrawPath(tc.make(), pen)
		}
		gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
		compareInk(t, gpuData, pixmapBytes(cpuPix), w, h, "aahairline "+tc.name)
	}
}

func TestLiveAAHairlinePerspective(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	var persp geom.Matrix
	persp.SetIdentity()
	persp.Set(6, 0.0008) // mild perspective
	persp.Set(7, 0.0004)

	scene := func(c *canvas.Canvas) {
		bg := canvas.NewPaint()
		bg.Color = 0xFFFFFFFF
		c.DrawPaint(bg)
		c.Concat(&persp)
		pen := canvas.NewPaint()
		pen.Color = 0xFF884400
		pen.Style = canvas.StyleStroke
		pen.StrokeWidth = 0
		pen.AntiAlias = true
		p := &path.Path{}
		p.MoveTo(12, 100)
		p.QuadTo(64, 10, 116, 100)
		p.LineTo(12, 100)
		p.Close()
		c.DrawPath(p, pen)
	}
	gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
	compareInk(t, gpuData, pixmapBytes(cpuPix), w, h, "aahairline perspective")
}

func TestLiveAAHairlineBatching(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	identity := geom.IdentityMatrix()
	for range 2 {
		shape := gl.MakeStyledShapePath(liveStarPath(), gl.SimpleHairlineStyle(),
			gl.DoSimplifyYes)
		sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0.2, G: 0.2, B: 0.6, A: 1}),
			gpu.AAYes, &identity, &shape)
	}
	if n := sdc.GetOpsTask().NumOpChains(); n != 1 {
		t.Fatalf("chain count %d want 1 (same-color hairline draws should merge)", n)
	}
	data := readSDC(t, sdc)
	// The star must have rendered: some ink along its top vertex column.
	inked := 0
	for y := 0; y < h; y++ {
		off := (y*w + 50) * 4
		if data[off] != 0xFF || data[off+1] != 0xFF {
			inked++
		}
	}
	if inked == 0 {
		t.Fatal("merged hairline draw rendered nothing")
	}
}
