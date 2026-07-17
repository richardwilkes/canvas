// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gorender

import (
	gocanvas "github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
)

// FrameTarget is a persistent GL render target for frame-loop measurement: one SurfaceDrawContext wrapped in a GL
// canvas device, reused across many frames — unlike RenderScenarioGPU, which creates a fresh surface per call for the
// once-per-scenario golden render. Built for the retired paired C-vs-Go frame-time baseline, it now backs the
// per-frame allocation gate (framealloc_test.go), mirroring the persistent surface a real render loop (unison) reuses
// across frames so the measurement is the steady-state per-frame recording/batching/flush cost, not surface creation.
type FrameTarget struct {
	g   *GPUContext
	sdc *gl.SurfaceDrawContext
	c   *gocanvas.Canvas
}

// NewFrameTarget creates a w x h RGBA8888-premul top-left offscreen draw context and a GL canvas over it — the same
// assembly RenderScenarioGPU performs, but retained on the returned target so callers render many frames into the one
// surface. It panics if the draw context could not be created (a benchmark harness treats that as fatal).
func (g *GPUContext) NewFrameTarget(w, h int) *FrameTarget {
	dims := geom.ISize{Width: int32(w), Height: int32(h)}
	sdc := gl.MakeSurfaceDrawContext(g.dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "frame-bench")
	if sdc == nil {
		panic("gorender: MakeSurfaceDrawContext failed for frame target")
	}
	return &FrameTarget{g: g, sdc: sdc, c: gocanvas.New(gl.NewDevice(sdc))}
}

// Canvas returns the target's persistent GL canvas (owned by the target).
func (t *FrameTarget) Canvas() *gocanvas.Canvas { return t.c }

// DirectContext returns the owning GL direct context, for the frame benchmark's warm-up GL-draw-count check and any
// resource-cache / batching stats.
func (t *FrameTarget) DirectContext() *gl.DirectContext { return t.g.dc }

// Flush flushes and submits the frame's recorded GPU work without reading anything back (syncCPU=false), matching the
// Go-side gpu/gl frame benchmark.
func (t *FrameTarget) Flush() { t.g.dc.FlushAndSubmit(false) }

// Release frees the draw context's GPU resources. The owning GPUContext must still be Disposed by the caller.
func (t *FrameTarget) Release() { t.sdc.Release() }
