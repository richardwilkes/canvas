// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package main

import (
	"os"
	"testing"
)

// TestPinSoftwareRendererCoversEveryGLLane guards the capture-vs-gate pin agreement: both GPU gates pin
// CANVAS_GLTEST_RENDERER=software (gorender's gpugolden_test.go and gpudmsaagolden_test.go), so both GPU lanes must
// capture under the same pin. When only gpudmsaa pinned it, `oracle bless -lane gpu` on a machine with hardware GL
// captured a set the gate could never render, caught only by the gate's GL_RENDERER manifest guard after a full
// two-pass capture.
func TestPinSoftwareRendererCoversEveryGLLane(t *testing.T) {
	for _, lane := range []string{laneGPU, laneGPUDMSAA} {
		t.Setenv("CANVAS_GLTEST_RENDERER", "")
		if err := pinSoftwareRenderer(lane); err != nil {
			t.Fatalf("pinSoftwareRenderer(%q): %v", lane, err)
		}
		if got := os.Getenv("CANVAS_GLTEST_RENDERER"); got != "software" {
			t.Fatalf("lane %s: CANVAS_GLTEST_RENDERER = %q, want \"software\" — a capture on this lane would render "+
				"on a stack its golden gate never uses", lane, got)
		}
	}
	// The raster lane is pure Go with no GL exposure; it must not touch the variable at all.
	t.Setenv("CANVAS_GLTEST_RENDERER", "left-alone")
	if err := pinSoftwareRenderer(laneRaster); err != nil {
		t.Fatalf("pinSoftwareRenderer(%q): %v", laneRaster, err)
	}
	if got := os.Getenv("CANVAS_GLTEST_RENDERER"); got != "left-alone" {
		t.Fatalf("the raster lane changed CANVAS_GLTEST_RENDERER to %q", got)
	}
}
