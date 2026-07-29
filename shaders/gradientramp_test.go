// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// TestEvalGradientRampSingleStop covers the searched fill stage's smallest input: one stop with an explicit position.
// The stop-trimming code used to hard-code the last stop index to 1 here, walking one stop past the end of the color and
// position slices.
func TestEvalGradientRampSingleStop(t *testing.T) {
	c := colorcore.Color4f{R: 0.25, G: 0.5, B: 0.75, A: 1}
	dst := make([]colorcore.PMColor4f, 4)
	EvalGradientRamp([]colorcore.Color4f{c}, []float32{0.5}, false, dst)
	want := colorcore.PMColor4f(c)
	for i, got := range dst {
		if got != want {
			t.Errorf("texel %d: got %+v, want %+v", i, got, want)
		}
	}

	// The same ramp premultiplied: a single translucent stop stays constant across the ramp.
	c = colorcore.Color4f{R: 1, G: 0.5, B: 0, A: 0.5}
	EvalGradientRamp([]colorcore.Color4f{c}, []float32{0}, true, dst)
	want = colorcore.PMColor4f{R: 0.5, G: 0.25, B: 0, A: 0.5}
	for i, got := range dst {
		if geom.ScalarAbs(got.R-want.R) > 1e-6 || geom.ScalarAbs(got.G-want.G) > 1e-6 ||
			geom.ScalarAbs(got.B-want.B) > 1e-6 || geom.ScalarAbs(got.A-want.A) > 1e-6 {
			t.Errorf("premul texel %d: got %+v, want %+v", i, got, want)
		}
	}
}

// TestEvalGradientRampTwoStops pins the two-stop searched-stage result the single-stop fix must not disturb: the ramp
// still interpolates linearly between the stops at their explicit positions.
func TestEvalGradientRampTwoStops(t *testing.T) {
	colors := []colorcore.Color4f{{A: 1}, {R: 1, A: 1}}
	dst := make([]colorcore.PMColor4f, 4)
	EvalGradientRamp(colors, []float32{0, 1}, false, dst)
	for i, got := range dst {
		wantR := (float32(i) + 0.5) / float32(len(dst))
		if geom.ScalarAbs(got.R-wantR) > 1e-6 || got.G != 0 || got.B != 0 || got.A != 1 {
			t.Errorf("texel %d: got %+v, want r=%v", i, got, wantR)
		}
	}
}
