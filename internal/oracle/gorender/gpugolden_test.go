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

// gpuGoldenDir is the per-platform self-captured GPU golden directory (relative to this package, since `go test` runs
// with the package directory as the working directory). The goldens are the library's own output, captured by `oracle
// bless` on the platform's software GL stack.
func gpuGoldenDir() string {
	return filepath.Join("..", "goldens", "gpu", runtime.GOOS+"_"+runtime.GOARCH)
}

// The log labels of the two GPU golden lanes; labelWrappedFBO additionally selects the wrappedFBOKnifeEdge exclusions.
const (
	labelOwnedRT    = "owned-rt"
	labelWrappedFBO = "wrapped-fbo"
)

// wrappedFBOKnifeEdge lists scenarios whose wrapped-FBO render deterministically differs from the owned-RT render (and
// therefore from the owned-RT-captured golden) beyond the ±1 envelope on a specific GL stack, keyed by the manifest's
// GL_RENDERER string. The two lanes draw identically; on the listed stack the destination's backing type (caller-owned
// renderbuffer FBO vs library-owned texture RT) nudges rasterization at a knife-edge pixel. Measured on linux_amd64
// llvmpipe: text-sdf-rotated differs by exactly 1 px at max delta 5, byte-identical across runs, while the owned-RT
// lane matches the golden bit-exactly and every other stack keeps the lanes within the envelope. The entries are keyed
// to the exact renderer string deliberately: a driver bump trips the GL-stack mismatch diagnostic first, forcing
// re-evaluation with the recapture. Listed scenarios are reported, not gated, in the wrapped-FBO lane only.
var wrappedFBOKnifeEdge = map[string]map[string]bool{
	"llvmpipe (LLVM 15.0.7, 256 bits)": {"text-sdf-rotated": true},
}

// checkGPUGoldens renders the whole corpus through render and diffs each scenario against the self-captured GPU golden
// under the exact1 profile, gating every scenario. label names the Go-side surface path in the log.
//
// The comparison is exact modulo ±1 LSB (imgdiff.Exact1), not bit-exact: software GL rasterizers wobble ±1
// intermittently between GL sessions, proven driver-internal (see the oracle soak command's doc comment), and the
// blessed capture is one representative of that envelope. Real rendering breaks measure ≥32 LSB, so the envelope costs
// no detection power.
//
// The gate pins CANVAS_GLTEST_RENDERER=software via t.Setenv: under exact1, comparing across GL stacks is meaningless
// (hardware and software rasterizers differ structurally at AA edges), so the gate must render on the same stack the
// goldens were captured on. On darwin the pin selects kCGLRendererGenericFloatID as a hard CGLChoosePixelFormat
// attribute (gpu/gl/gltest/context_darwin.go); having requested the pin, a context-creation failure on darwin is a
// hard failure, not a skip — a gate that quietly stops running is worse than a red leg that says why. On other
// platforms the variable is a no-op and a missing GL stack remains the legitimate, graceful skip. As a second guard,
// the live context's GL_RENDERER string must equal the manifest's: a mismatch means the GL stack moved underneath the
// goldens (a runner-image driver bump, or a local run on the wrong stack) and fails with a diagnosis instead of a
// wall of pixel differences.
//
// It renders inline on the caller's goroutine: NewGPUContext locks the OS thread and the GL context is current only on
// the goroutine that created it, so there are no t.Run subtests (a subtest runs on a different goroutine where the
// context would not be current).
func checkGPUGoldens(t *testing.T, label string, render func(*gorender.GPUContext, scenario.Scenario) []byte) {
	t.Helper()
	dir := gpuGoldenDir()
	manifest, err := golden.ReadManifest(dir)
	if err != nil {
		t.Skipf("no self-captured GPU goldens for %s_%s (%v); capture a set with the capture-goldens "+
			"workflow to gate this platform", runtime.GOOS, runtime.GOARCH, err)
	}

	// The checked-in goldens must cover exactly the corpus: a scenario added without regenerating goldens (or a stale
	// golden left after a scenario is removed) is a drift bug the differential would otherwise miss.
	goldenNames := make(map[string]bool, len(manifest.Entries))
	for _, e := range manifest.Entries {
		goldenNames[e.Name] = true
	}
	for _, sc := range scenario.All() {
		if !goldenNames[sc.Name] {
			t.Errorf("scenario %q has no self-captured GPU golden in %s", sc.Name, dir)
		}
		delete(goldenNames, sc.Name)
	}
	for name := range goldenNames {
		t.Errorf("self-captured GPU golden %q has no corresponding corpus scenario (stale golden in %s)", name, dir)
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
		goPix := render(g, sc)
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
			t.Logf("%s %-32s bimodal on this renderer (reported, not gated): %s", label, sc.Name, res)
			continue
		}
		if label == labelWrappedFBO && wrappedFBOKnifeEdge[manifest.GLRenderer][sc.Name] {
			t.Logf("%s %-32s backing-type knife edge on this renderer (reported, not gated): %s",
				label, sc.Name, res)
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
			t.Errorf("%s %-32s diverged from the self-captured golden beyond the ±1 envelope: %s",
				label, sc.Name, res)
		} else {
			t.Logf("%s %-32s PASS: %s", label, sc.Name, res)
		}
	}
	t.Logf("%s: gated %d scenarios (%d bit-exact, %d within the ±1 envelope; %d reported-not-gated on this renderer)",
		label, gated, exact, wobbled, len(scenario.All())-gated)
}

// TestGoGPUvsSelfCapturedGolden is the GPU pixel gate over the library-owned offscreen render target: the corpus
// rendered through the library's GL backend, diffed against the self-captured per-platform goldens under exact1. Any
// machine whose GL stack matches the manifest verifies the library's GPU output against the committed reference.
func TestGoGPUvsSelfCapturedGolden(t *testing.T) {
	checkGPUGoldens(t, labelOwnedRT, gorender.RenderScenarioGPU)
}

// TestGoGPUWrappedFBOvsSelfCapturedGolden is the same gate over the *caller-owned wrapped FBO* —
// gl.NewRenderTargetSurfaceFromBackendRenderTarget, the production surface path unison drives when it hands the library
// its window FBO — rather than the library-owned offscreen target the test above uses. The two lanes are separate gates
// because the surface-creation path differs even though the drawing does not; keeping both is what verifies the
// wrapped-FBO path end-to-end on the whole corpus rather than only on gpu/gl's CPU-checked live tests.
//
// Both lanes gate against the same goldens, which bless captures through the owned-RT path:
// RenderScenarioGPUWrappedFBO wraps at top-left origin, so its output matches the owned-RT path scenario for scenario
// (verified within the same ±1 envelope the gate applies, except the per-stack knife-edge scenarios in
// wrappedFBOKnifeEdge; the live cgo differential historically measured the two lanes within one bit-exact scenario of
// each other on every platform and driver tested).
func TestGoGPUWrappedFBOvsSelfCapturedGolden(t *testing.T) {
	checkGPUGoldens(t, labelWrappedFBO, gorender.RenderScenarioGPUWrappedFBO)
}
