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
	"path/filepath"
	"runtime"
	"testing"

	"github.com/richardwilkes/canvas/internal/oracle/golden"
	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// TestGoGPUDMSAAvsSelfCapturedGolden is the dynamic-MSAA lane's pixel gate: the corpus rendered through the port's GL
// backend with the dynamic-MSAA surface-props flag set (the production wrapped-FBO path), diffed against the
// self-captured per-platform DMSAA goldens under the exact1 profile (see checkGPUGoldens for why exact1 and why the
// software-renderer pin).
//
// DMSAA output cannot be compared against the plain GPU goldens: promoting path draws to a 4x MSAA attachment resolves
// every antialiased edge differently from coverage-AA. The lane needs its own reference, which is why a separate
// golden set exists.
//
// darwin_arm64 deliberately has no DMSAA golden set: Apple's software renderer on arm64 renders MSAA-quantized edges
// in one of two bit-exact flavors per GL session (structural whole-pixel edge shifts, far beyond the ±1 envelope),
// proven driver-internal, so a captured set would intermittently fail this gate through no fault of the code. The
// missing-set skip below covers it; the lane gates on the llvmpipe and darwin_amd64 legs, whose stacks render it
// deterministically.
//
// The skip for a missing set must stay ahead of the renderer pin and context creation: with no set there is nothing to
// gate, and the darwin fail-not-skip contract below only applies once a set exists. Having requested the pin, a
// context-creation failure on darwin is a hard failure, not a skip: the usual err-means-no-GL-stack skip would let the
// gate silently vanish if a runner's driver stopped exposing the pinned pixel format, and a gate that quietly stops
// running is worse than a red leg that says why. On other platforms the variable is a no-op (see context_linux.go /
// context_windows.go) and a missing GL stack remains the legitimate, graceful skip it is everywhere else.
//
// Rendered inline on the test goroutine: NewGPUContext locks the OS thread and the GL context is current only on the
// goroutine that created it, so there are no t.Run subtests.
func TestGoGPUDMSAAvsSelfCapturedGolden(t *testing.T) {
	dir := filepath.Join("..", "goldens", "gpudmsaa", runtime.GOOS+"_"+runtime.GOARCH)
	manifest, err := golden.ReadManifest(dir)
	if err != nil {
		t.Skipf("no self-captured DMSAA goldens for %s_%s (%v); darwin_arm64 has none by design "+
			"(bimodal driver MSAA quantization) — other GL-capable platforms gate this lane",
			runtime.GOOS, runtime.GOARCH, err)
	}

	goldenNames := make(map[string]bool, len(manifest.Entries))
	for _, e := range manifest.Entries {
		goldenNames[e.Name] = true
	}
	for _, sc := range scenario.All() {
		if !goldenNames[sc.Name] {
			t.Errorf("scenario %q has no self-captured DMSAA golden in %s", sc.Name, dir)
		}
		delete(goldenNames, sc.Name)
	}
	for name := range goldenNames {
		t.Errorf("self-captured DMSAA golden %q has no corresponding corpus scenario (stale golden in %s)", name, dir)
	}

	t.Setenv("CANVAS_GLTEST_RENDERER", "software")
	g, err := gorender.NewGPUContext()
	if err != nil {
		if runtime.GOOS == "darwin" {
			t.Fatalf("no Go GPU context available with the software-renderer pin requested: %v", err)
		}
		t.Skipf("no Go GPU context available: %v", err)
	}
	defer g.Dispose()
	if got := g.RendererString(); got != manifest.GLRenderer {
		t.Fatalf("GL stack mismatch: goldens in %s were captured on %q, this context is %q — the GL stack "+
			"moved; re-capture deliberately if the change is intentional", dir, manifest.GLRenderer, got)
	}

	gated, exact, wobbled := 0, 0, 0
	for _, sc := range scenario.All() {
		goPix := gorender.RenderScenarioGPUDMSAA(g, sc)
		wantPix, w, h, readErr := golden.ReadPNG(filepath.Join(dir, sc.Name+".png"))
		if readErr != nil {
			t.Errorf("%s: reading golden: %v", sc.Name, readErr)
			continue
		}
		if w != sc.Width || h != sc.Height {
			t.Errorf("%s: golden is %dx%d, scenario is %dx%d", sc.Name, w, h, sc.Width, sc.Height)
			continue
		}
		res, readErr := imgdiff.Compare(wantPix, goPix, sc.Width, sc.Height, imgdiff.Exact1)
		if readErr != nil {
			t.Errorf("%s: %v", sc.Name, readErr)
			continue
		}
		if gorender.DriverBimodal(manifest.GLRenderer, sc.Name) {
			t.Logf("dmsaa %-32s bimodal on this renderer (reported, not gated): %s", sc.Name, res)
			continue
		}
		gated++
		switch {
		case res.MaxDelta == 0:
			exact++
		case res.Pass():
			wobbled++
		}
		if !res.Pass() {
			t.Errorf("dmsaa %-32s diverged from the self-captured golden beyond the ±1 envelope: %s",
				sc.Name, res)
		} else {
			t.Logf("dmsaa %-32s PASS: %s", sc.Name, res)
		}
	}
	t.Logf("dmsaa: gated %d scenarios (%d bit-exact, %d within the ±1 envelope; %d bimodal on this renderer)",
		gated, exact, wobbled, len(scenario.All())-gated)
}
