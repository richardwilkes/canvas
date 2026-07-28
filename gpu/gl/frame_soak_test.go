// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GPU frame race/leak soak (GA hardening: race/leak soak over 10k frames): render many panel frames through the GPU
// backend on one persistent context and surface — a steady-state render loop, exactly what BenchmarkGLFramePanels times
// — and assert the GPU resource cache reaches a steady state and stays there across the soak: no unbounded growth in
// resource count or bytes, no budget overflow, and zero program-cache misses (no per-frame recompilation). A real leak
// (an un-recycled buffer/texture/program per frame, or a pool that fails to reset) grows ~linearly with the frame count
// and blows past the tolerances; the per-frame Go allocation ceiling is the separate 238 allocs/frame gate the
// benchmark reports.
//
// This is the GPU half of the GA soak and is cgo-free (built on a gltest context, with no cgo dependency), so it also
// runs under `-race`, where it exercises the op / program-info / pipeline / geometry-processor / scratch-backing pools
// and the persistent flush state across thousands of recycle cycles on one context. Skips when no GL context is
// available.

package gl_test

import (
	"os"
	"strconv"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/gpu/gl"
)

func TestGLFrameLeakSoak(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	defer dc.Destroy()
	const side = 256
	const rows, cols = 6, 6
	sdc := newLiveDrawSDC(t, dc, side, side)
	defer sdc.Release()
	c := canvas.New(gl.NewDevice(sdc))
	bg, fill, border := newFramePanelPaints()

	// A leak grows linearly with the frame count, so a few thousand frames is a decisive signal while staying fast
	// under -race (each frame is ~50us un-raced). -short cuts it for quick local runs; CANVAS_SOAK_FRAMES overrides for
	// the pre-release long soak (10k+ under -race).
	frames := 8000
	if testing.Short() {
		frames = 500
	}
	if v := os.Getenv("CANVAS_SOAK_FRAMES"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			t.Fatalf("bad CANVAS_SOAK_FRAMES %q: %v", v, err)
		}
		frames = n
	}

	// Warm to steady state: fill the program cache and establish the scratch buffers and render target so the baseline
	// footprint is the recycled steady state, not the first-frame ramp.
	const warmup = 16
	for i := 0; i < warmup; i++ {
		benchPanels(c, bg, fill, border, side, rows, cols)
		dc.FlushAndSubmit(false)
	}
	rc := dc.ResourceCache()
	baseCount := rc.ResourceCount()
	baseBytes := rc.ResourceBytes()
	if baseCount == 0 {
		t.Fatal("warm-up produced no GPU resources — the frame did not render")
	}
	dc.ResetGLDrawStats()
	pcBefore := dc.Gpu().ProgramCacheStats()

	for i := 0; i < frames; i++ {
		benchPanels(c, bg, fill, border, side, rows, cols)
		dc.FlushAndSubmit(false)
	}

	// Leak signal 1: the steady-state resource footprint must not grow across the soak. The small tolerance absorbs
	// cache bookkeeping (a scratch buffer binned to a different size, a purge that runs a frame later); a real
	// per-frame resource leak grows by ~frames, orders of magnitude past this.
	endCount := rc.ResourceCount()
	endBytes := rc.ResourceBytes()
	t.Logf("soak: %d frames, %.2f GL draws/frame; resources %d->%d (%d->%d bytes)",
		frames, float64(dc.NumGLDraws())/float64(frames), baseCount, endCount, baseBytes, endBytes)
	if endCount > baseCount+2 {
		t.Errorf("resource count grew across %d frames: %d -> %d (leak?)", frames, baseCount, endCount)
	}
	if endBytes > baseBytes+baseBytes/8+4096 {
		t.Errorf("resource bytes grew across %d frames: %d -> %d (leak?)", frames, baseBytes, endBytes)
	}

	// Leak signal 2: the budgeted cache must not have overflowed its limit.
	if rc.OverBudget() {
		t.Errorf("resource cache over budget after soak: %d > %d budgeted bytes",
			rc.BudgetedResourceBytes(), rc.MaxResourceBytes())
	}

	// Leak signal 3: zero program-cache misses during the soak. After warm-up every frame reuses the cached programs; a
	// miss means a program was dropped and recompiled (a program-cache leak/thrash).
	pc := dc.Gpu().ProgramCacheStats()
	if misses := pc.NumProgramCacheMisses - pcBefore.NumProgramCacheMisses; misses != 0 {
		t.Errorf("program cache took %d misses during the soak (recompilation thrash?)", misses)
	}
}
