// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live profile of GL calls per frame: renders one steady-state panels frame and one scene frame and logs the
// per-entry-point call counts, sorted by count — the "which GL entry points dominate" view that drives the
// state-shadowing work. Run with -v to see the tables:
//
//  go test ./gpu/gl/ -run TestFrameGLCallProfile -v

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/gpu/gl"
)

func TestFrameGLCallProfile(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	fns := dc.Gpu().Functions()

	// Panels frame (matches BenchmarkGLFramePanels).
	const side = 256
	const rows, cols = 6, 6
	sdc := newLiveDrawSDC(t, dc, side, side)
	defer sdc.Release()
	c := canvas.New(gl.NewDevice(sdc))
	bg, fill, border := newFramePanelPaints()
	benchPanels(c, bg, fill, border, side, rows, cols) // Warm-up frame.
	dc.FlushAndSubmit(false)
	fns.ResetCallCounts()
	benchPanels(c, bg, fill, border, side, rows, cols)
	dc.FlushAndSubmit(false)
	logCallCounts(t, "panels", fns)

	// Scene frame (matches BenchmarkGLFrameScene).
	fns.ResetCallCounts()
	img := newCheckerImage(t)
	sdc2 := newLiveDrawSDC(t, dc, 128, 128)
	defer sdc2.Release()
	c2 := canvas.New(gl.NewDevice(sdc2))
	sceneCanvas(c2, img) // Warm-up frame.
	dc.FlushAndSubmit(false)
	fns.ResetCallCounts()
	sceneCanvas(c2, img)
	dc.FlushAndSubmit(false)
	logCallCounts(t, "scene", fns)
}

func logCallCounts(t *testing.T, frame string, fns *gl.Functions) {
	t.Helper()
	counts := fns.CallCounts()
	if len(counts) == 0 {
		t.Fatalf("%s frame issued no GL calls", frame)
	}
	t.Logf("%s frame: %d GL calls", frame, fns.TotalCallCount())
	for _, cc := range counts {
		t.Logf("  %6d  %s", cc.Count, cc.Name)
	}
}
