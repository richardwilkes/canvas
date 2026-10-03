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
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

func findScenario(t *testing.T, name string) scenario.Scenario {
	t.Helper()
	for _, sc := range scenario.All() {
		if sc.Name == name {
			return sc
		}
	}
	t.Fatalf("scenario %q not found in corpus", name)
	return scenario.Scenario{}
}

// TestClipRotatedRectPersists guards the clip-rotated-rect scenario's setup: the rotated clip must be baked into device
// space and persist while the axis-aligned blue bars draw. Wrapping the clip in a Save/Restore pair pops it together
// with the CTM, so the bars render unclipped and the scenario tests no clipping at all.
//
// The check renders through the raster backend (the raster gate's own path, no GL needed) and samples pixels chosen so
// the buggy and fixed outputs disagree:
//   - (5,5) lies on the first blue bar (x in [0,16]) but well outside the rotated clip square. Unclipped (bug) it is
//     blue; clipped (fixed) it is the white background. This is the regression sentinel.
//   - (136,128) lies on a bar (x in [128,144]) and inside the clip, so it must stay blue in both — proving the fix did
//     not simply clip everything away.
func TestClipRotatedRectPersists(t *testing.T) {
	sc := findScenario(t, "clip-rotated-rect")
	pix := gorender.RenderScenarioRaster(sc)

	at := func(x, y int) (r, g, b, a byte) {
		i := (y*sc.Width + x) * 4
		return pix[i], pix[i+1], pix[i+2], pix[i+3]
	}
	// blue = 0xFF2E5FE0 → R,G,B,A bytes; white = 0xFFFFFFFF.
	const (
		blueR, blueG, blueB = 0x2E, 0x5F, 0xE0
	)
	isBlue := func(x, y int) bool {
		r, g, b, a := at(x, y)
		return r == blueR && g == blueG && b == blueB && a == 0xFF
	}
	isWhite := func(x, y int) bool {
		r, g, b, a := at(x, y)
		return r == 0xFF && g == 0xFF && b == 0xFF && a == 0xFF
	}

	if !isWhite(5, 5) {
		r, g, b, a := at(5, 5)
		t.Errorf("pixel (5,5) is on bar 0 but outside the rotated clip: want white background, got RGBA %#02x %#02x "+
			"%#02x %#02x — the rotated clip did not persist", r, g, b, a)
	}
	if !isBlue(136, 128) {
		r, g, b, a := at(136, 128)
		t.Errorf("pixel (136,128) is on a bar inside the rotated clip: want blue, got RGBA %#02x %#02x %#02x %#02x",
			r, g, b, a)
	}
}
