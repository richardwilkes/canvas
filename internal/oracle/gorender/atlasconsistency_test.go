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
	"testing"

	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// TestAtlasGPUvsCPUSelfConsistency renders the four atlas scenarios through both of the port's own backends and
// compares them under the GPU profile. The two Go lanes lower drawAtlas differently — the CPU lane per sprite (rect
// fill under a concatenated CTM), the GPU lane through the faithful DrawAtlasOp vertex stream — so this cross-check is
// the standing guard on the per-sprite equivalence claim, independent of the C oracle. Cgo-free; skips without a GL
// context.
//
// Rendered inline on the test goroutine: gorender.NewGPUContext locks the OS thread and the GL context is current only
// on the goroutine that created it.
func TestAtlasGPUvsCPUSelfConsistency(t *testing.T) {
	g, err := gorender.NewGPUContext()
	if err != nil {
		t.Skipf("no Go GPU context available: %v", err)
	}
	defer g.Dispose()

	for _, name := range []string{"atlas-basic", "atlas-colors-modulate", "atlas-blend", "atlas-cull"} {
		sc, ok := scenario.ByName(name)
		if !ok {
			t.Fatalf("scenario %q not registered", name)
		}
		cpuPix := gorender.RenderScenarioRaster(sc)
		gpuPix := gorender.RenderScenarioGPU(g, sc)
		res, compErr := imgdiff.Compare(cpuPix, gpuPix, sc.Width, sc.Height, imgdiff.GPU)
		if compErr != nil {
			t.Errorf("%s: %v", name, compErr)
			continue
		}
		if !res.Pass() {
			t.Errorf("%-24s Go-GPU vs Go-CPU over GPU threshold: %s", name, res)
		} else {
			t.Logf("%-24s PASS: %s", name, res)
		}
	}
}
