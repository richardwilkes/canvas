// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-frame allocation gate.
//
// This is what remains of the old paired driver matrix. That matrix stated its gates as ratios against the cgo Skia
// baseline rendered alongside the port in one process (GPU frame time <=1.5x, CPU raster <=3x); with the C library gone
// there is nothing to divide by, and absolute wall-clock gates are not worth stating on shared-tenancy CI runners — the
// ratio was precisely what canceled runner speed variance. The allocation count does not have that problem: it is
// deterministic, it is the axis the port actually regresses on (FFI marshaling, lost pooling), and it is what caught
// the per-frame allocation regressions. So timing is dropped and allocations are kept, as a plain test rather than a
// benchmark whose output something else has to parse.
//
// Cgo-free, so it runs on every leg with a GL stack rather than only where Skia builds. Skips without a GL context,
// like the other live GPU tests.
package gorender_test

import (
	"testing"

	gocanvas "github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/gorender"
)

// The panel-frame geometry, identical to gpu/gl/frame_bench_test.go's BenchmarkGLFramePanels — a batching-friendly UI
// frame (background fill plus a grid of filled rounded-rect panels with stroked borders), the "unison workload" proxy
// the GA gates were stated against.
const (
	framePanelSide = 256
	framePanelRows = 6
	framePanelCols = 6
	framePanelPad  = 6
)

// frameAllocBudget caps allocations per rendered frame. The frame records at 1 alloc on darwin/arm64 (the panels frame
// went 238 -> 1), and the fixed-arity FFI lane removes the per-call marshaling allocs on all three OSes, so 10 leaves
// ~10x absolute headroom while still tripping on a single GL call falling back to the SyscallN lane or a lost pooling
// tier. Carried over from the retired benchratio gate this replaces.
const frameAllocBudget = 10

// panelColor is the deterministic per-cell fill color from gpu/gl/frame_bench_test.go's benchPanels.
func panelColor(r, k int) uint32 {
	return 0xFF000000 | uint32((r*37+k*53)&0xFF)<<16 | uint32((r*91+k*17)&0xFF)<<8 | uint32((r*13+k*71)&0xFF)
}

func benchPanels(c *gocanvas.Canvas, bg, fill, border *gocanvas.Paint, side float32, rows, cols int) {
	c.DrawPaint(bg)
	const pad = framePanelPad
	cw := (side - pad) / float32(cols)
	ch := (side - pad) / float32(rows)
	for r := 0; r < rows; r++ {
		for k := 0; k < cols; k++ {
			x := pad + float32(k)*cw
			y := pad + float32(r)*ch
			rect := geom.RectLTRB(x, y, x+cw-pad, y+ch-pad)
			fill.Color = colorcore.Color(panelColor(r, k))
			c.DrawRoundRect(rect, 5, 5, fill)
			c.DrawRect(rect, border)
		}
	}
}

// newFramePanelPaints builds the paints once, outside the measured frame (matching benchPanels).
func newFramePanelPaints() (bg, fill, border *gocanvas.Paint) {
	bg = gocanvas.NewPaint()
	bg.Color = 0xFF202830
	fill = gocanvas.NewPaint()
	fill.AntiAlias = true
	border = gocanvas.NewPaint()
	border.AntiAlias = true
	border.Style = gocanvas.StyleStroke
	border.StrokeWidth = 2
	border.Color = 0xFF3060A0
	return bg, fill, border
}

// TestGLFramePanelsAllocs holds the port's steady-state per-frame allocations under frameAllocBudget.
//
// Rendered inline on the test goroutine: gorender.NewGPUContext locks the OS thread and the GL context is current only
// on the goroutine that created it, so there are no subtests.
func TestGLFramePanelsAllocs(t *testing.T) {
	g, err := gorender.NewGPUContext()
	if err != nil {
		t.Skipf("no Go GPU context available: %v", err)
	}
	defer g.Dispose()
	target := g.NewFrameTarget(framePanelSide, framePanelSide)
	defer target.Release()
	c := target.Canvas()
	bg, fill, border := newFramePanelPaints()

	frame := func() {
		benchPanels(c, bg, fill, border, framePanelSide, framePanelRows, framePanelCols)
		target.Flush()
	}

	// Warm the program/resource caches with one full frame: first-frame compilation and cache fills allocate heavily
	// and are not what this gates, so measuring without a warm-up would gate on start-up cost.
	frame()
	if target.DirectContext().NumGLDraws() == 0 {
		t.Fatal("warm-up frame issued no GL draws — the frame did not render")
	}

	allocs := testing.AllocsPerRun(20, frame)
	t.Logf("panels frame: %.1f allocs/frame (budget %d)", allocs, frameAllocBudget)
	if allocs > frameAllocBudget {
		t.Errorf("panels frame allocates %.1f/frame, over the budget of %d — a GL call has likely fallen back "+
			"to the SyscallN lane or a pooling tier was lost", allocs, frameAllocBudget)
	}
}
