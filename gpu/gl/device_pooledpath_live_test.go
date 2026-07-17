// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context regression test for the GPU device's pooled per-draw working paths. The device lowers styled shapes,
// hairline points/polygons, stroked lines, and mask-filtered shapes through transient device-space paths that are now
// drawn from path.Borrow() and returned with path.Recycle() (the pooled-temporaries tier, extended from the CPU raster
// device to the GPU device). Correctness hinges on each borrowed path being cloned (MakeShapePath /
// MakeStyledShapePath) or consumed synchronously before it is recycled, so a leak would let a later draw's Borrow()
// reuse storage a still-live op referenced. This test renders a scene that interleaves several distinct styled draws
// through the device and compares it to the CPU canvas (the Go-GPU vs Go-CPU self-consistency lane): a pooled-path leak
// would corrupt a shape's geometry and blow far past the AA-edge budget. Skips when no GL context is available.

package gl_test

import (
	"testing"
	"unsafe"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/patheffect"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stroke"
)

// styledPooledScene draws the device lanes that lower through pooled transient paths: dashed (path-effect) stroked
// rect/oval/rrect, hairline points, a stroked polygon, a round-cap stroked line, and a blurred (mask-filter) filled
// path. Every shape occupies a distinct region so that a pooled-path leak from one draw corrupts a visibly different
// shape.
func styledPooledScene(c *canvas.Canvas) {
	white := canvas.NewPaint()
	white.Color = 0xFFFFFFFF
	c.DrawPaint(white)

	// Dashed stroked rect/oval/rrect: DrawRect/DrawOval/DrawRRect route through the styled lane (paint.PathEffect !=
	// nil), building a pooled transient path (AddRect/AddOval/AddRRect) and devolving the dash on the CPU into a second
	// pooled path inside drawStyledShapeSlow.
	dash := canvas.NewPaint()
	dash.Color = 0xFFCC2200
	dash.AntiAlias = true
	dash.Style = canvas.StyleStroke
	dash.StrokeWidth = 3
	dash.PathEffect = patheffect.MakeDash([]float32{6, 4}, 0)
	c.DrawRect(geom.RectLTRB(8, 8, 56, 34), dash)
	c.DrawOval(geom.RectLTRB(72, 8, 120, 40), dash)
	c.DrawRoundRect(geom.RectLTRB(8, 42, 56, 70), 6, 6, dash)

	// Hairline points (width 0): the DrawPoints degenerate-segment lane builds a pooled path.
	hair := canvas.NewPaint()
	hair.Color = 0xFF004488
	hair.AntiAlias = true
	hair.Style = canvas.StyleStroke
	hair.StrokeWidth = 0
	pts := []geom.Point{
		{X: 72, Y: 48},
		{X: 82, Y: 52},
		{X: 92, Y: 48},
		{X: 102, Y: 52},
		{X: 112, Y: 48},
	}
	c.DrawPoints(canvas.PointModePoints, pts, hair)

	// Stroked polygon: the DrawPoints polygon lane builds a pooled path.
	poly := canvas.NewPaint()
	poly.Color = 0xFF117722
	poly.AntiAlias = true
	poly.Style = canvas.StyleStroke
	poly.StrokeWidth = 2
	tri := []geom.Point{{X: 74, Y: 60}, {X: 116, Y: 60}, {X: 95, Y: 86}}
	c.DrawPoints(canvas.PointModePolygon, tri, poly)

	// Round-cap stroked line: the non-fast drawLine lane builds a pooled path.
	lp := canvas.NewPaint()
	lp.Color = 0xFF008080
	lp.AntiAlias = true
	lp.Style = canvas.StyleStroke
	lp.StrokeWidth = 5
	lp.Cap = canvas.StrokeCap(stroke.CapRound)
	c.DrawLine(12, 80, 52, 118, lp)

	// Blurred filled triangle path (mask filter): DrawPath routes through the styled lane's mask-filter branch, cloning
	// the transient before swCreateFilteredMask rasterizes its own pooled device path. A general
	// (non-rect/rrect/circle) shape takes the SW mask lane, which shares maskfilter.DrawToMask/FilterMask with the CPU
	// canvas, so the two match closely.
	tri2 := &path.Path{}
	tri2.MoveTo(74, 96)
	tri2.LineTo(118, 96)
	tri2.LineTo(96, 122)
	tri2.Close()
	blur := canvas.NewPaint()
	blur.Color = 0xFF663399
	blur.AntiAlias = true
	blur.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)
	c.DrawPath(tri2, blur)
}

// TestLiveGLDevicePooledPaths renders the styled-pooled-path scene through the GPU device and the CPU canvas and
// asserts Go-GPU vs Go-CPU self-consistency. It guards the GPU-device path.Borrow adoption: a use-after-recycle would
// corrupt a shape and exceed the budget.
func TestLiveGLDevicePooledPaths(t *testing.T) {
	_, dc := newLiveDirectContext(t)

	const w, h = 128, 128

	// CPU reference.
	cpuPix := raster.NewPixmap(w, h)
	cpuCanvas := canvas.NewForPixmap(cpuPix)
	styledPooledScene(cpuCanvas)

	// GPU device — the same scene through gl.Device.
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	dev := gl.NewDevice(sdc)
	gpuCanvas := canvas.New(dev)
	styledPooledScene(gpuCanvas)

	gpuData := readSDC(t, sdc)
	cpuData := unsafe.Slice((*byte)(unsafe.Pointer(&cpuPix.Pix[0])), len(cpuPix.Pix)*4)

	// Go-GPU vs Go-CPU self-consistency via the over-tolerance fraction. This verifies the device's styled / points /
	// line / mask-filter lanes — which now lower through pooled transient paths — render the same scene as the CPU
	// canvas. Interiors must agree tightly; AA edge pixels (dashes, hairlines, and the sharp stroked-polygon corners
	// have many) legitimately swing the full range where one renderer covers a boundary pixel and the other does not,
	// so a hard per-pixel delta cap is inappropriate here. A pooled-path leak corrupts a shape's geometry — its whole
	// area disagrees with the CPU reference (well beyond the thin AA fringe) — so it trips the budget. Measured worst
	// case ~0.9%; the budget leaves headroom for CI's software-GL AA differences.
	const tol = 12
	budget := w * h * 4 / 100 // 4% — above the AA fringe, below a corrupted shape's footprint
	over := 0
	worst := 0
	rb := w * 4
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := y*rb + x*4
			pixOver := false
			for c := range 4 {
				d := int(gpuData[off+c]) - int(cpuData[off+c])
				if d < 0 {
					d = -d
				}
				if d > tol {
					pixOver = true
				}
				if d > worst {
					worst = d
				}
			}
			if pixOver {
				over++
			}
		}
	}
	if over > budget {
		t.Fatalf("%d pixels beyond ±%d (budget %d, worst delta %d) — likely a pooled-path "+
			"corruption", over, tol, budget, worst)
	}
	t.Logf("styled pooled-path scene: %d/%d pixels beyond ±%d (budget %d), worst delta %d",
		over, w*h, tol, budget, worst)
}
