// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gorender_test

import (
	"runtime"
	"testing"

	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// mixedPathScenarios are the scenarios TestWrappedFBOTeardownResyncsContext interleaves: enough distinct render-target
// sizes and enough drawing to make the render-target and FBO churn that recycles GL names, and each one paints most of
// its canvas so a render that lands in the wrong framebuffer reads back as an obvious blank rather than a near-match.
var mixedPathScenarios = []string{"clip-rotated-rect", "clip-nested-shrink", "hairlines", "ovals-rotated"}

// TestWrappedFBOTeardownResyncsContext gates mixing the wrapped-FBO and library-owned render paths on one GPUContext.
//
// createOffscreenFBO binds an FBO behind the direct context's back, so the setup re-syncs the context's GL-state shadow
// with ResetContext(AllBackendState). The teardown performs the symmetric mutation — an unbind plus three deletes,
// again behind the context's back — and used to skip that re-sync, leaving the shadow describing GL state that no
// longer exists (the deleted FBO recorded as the bound draw framebuffer, the torn-down wrapped target recorded as the
// bound render target) while GL itself has framebuffer 0 bound. The library skips redundant render-target binds, so a
// later render on the same context that the shadow believes is already bound would have its bind elided and draw into
// framebuffer 0.
//
// Nothing in the harness mixed the two paths on one context before this test, which is why the asymmetry survived: the
// gates each build their own context and stay on one path. The check therefore interleaves them deliberately —
// owned-RT baselines for several scenarios, then a wrapped-FBO render followed immediately by the same owned-RT render
// for each — and requires every post-teardown render to reproduce its baseline within the ±1 LSB envelope the GPU
// gates use. A render whose bind was elided loses its drawing entirely, far outside that envelope.
//
// It is a guard over the mixed-path contract, not a reproduction of a live failure: the elision in
// Gpu.flushRenderTarget keys on the render target's monotonically increasing unique ID rather than on the FBO name,
// and the library invalidates that ID whenever it creates an FBO, so a freshly created render target rebinds
// regardless and this test passes with the teardown re-sync removed. It exists so that mixing the paths — which
// nothing did before — stays gated, and so the re-sync cannot be dropped without something exercising what it
// protects.
//
// It runs inline on the caller's goroutine (NewGPUContext locks the OS thread and the context is current only there)
// and pins the software renderer for the same reason the golden gates do.
func TestWrappedFBOTeardownResyncsContext(t *testing.T) {
	t.Setenv("CANVAS_GLTEST_RENDERER", "software")
	g, err := gorender.NewGPUContext()
	if err != nil {
		if runtime.GOOS == "darwin" {
			t.Fatalf("no Go GPU context available with the software-renderer pin requested: %v", err)
		}
		t.Skipf("no Go GPU context available: %v", err)
	}
	defer g.Dispose()

	scenarios := make([]scenario.Scenario, 0, len(mixedPathScenarios))
	baselines := make([][]byte, 0, len(mixedPathScenarios))
	for _, name := range mixedPathScenarios {
		sc := findScenario(t, name)
		scenarios = append(scenarios, sc)
		baselines = append(baselines, gorender.RenderScenarioGPU(g, sc))
	}
	for i, sc := range scenarios {
		gorender.RenderScenarioGPUWrappedFBO(g, sc) // creates, wraps, and tears down a caller-owned FBO
		after := gorender.RenderScenarioGPU(g, sc)
		res, cmpErr := imgdiff.Compare(baselines[i], after, sc.Width, sc.Height, imgdiff.Exact1)
		if cmpErr != nil {
			t.Fatalf("%s: comparing the owned-RT renders around a wrapped-FBO render: %v", sc.Name, cmpErr)
		}
		if !res.Pass() {
			t.Fatalf("%s: the owned-RT render after a wrapped-FBO render diverged from the one before it: %s — the "+
				"wrapped-FBO teardown left the direct context's GL-state shadow describing torn-down state",
				sc.Name, res)
		}
	}
}
