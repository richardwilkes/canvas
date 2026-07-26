// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/raster"
)

// cpuBlendMismatchesGPU lists the modes whose CPU evaluation drifts from the GPU's: the four non-separable (HSL) modes,
// which are computed as scalar floats rather than vectors, plus soft-light and color-burn.
var cpuBlendMismatchesGPU = map[raster.BlendMode]bool{
	raster.BlendSoftLight:  true,
	raster.BlendColorBurn:  true,
	raster.BlendHue:        true,
	raster.BlendSaturation: true,
	raster.BlendColor:      true,
	raster.BlendLuminosity: true,
}

// TestDoesCPUBlendImplMatchGPU pins the sense of the predicate: it returns true for the modes the CPU reproduces
// closely enough to fold, i.e. everything except the HSL modes, soft-light and color-burn.
func TestDoesCPUBlendImplMatchGPU(t *testing.T) {
	for mode := raster.BlendClear; mode <= raster.BlendLuminosity; mode++ {
		want := !cpuBlendMismatchesGPU[mode]
		if got := doesCPUBlendImplMatchGPU(mode); got != want {
			t.Errorf("doesCPUBlendImplMatchGPU(%d) = %v, want %v", mode, got, want)
		}
	}
}

// TestBlendFPConstantOutputFolding checks that a matching mode is what enables constant folding (and that the folded
// value is the CPU blend), while a mismatching mode or a non-constant child suppresses it.
func TestBlendFPConstantOutputFolding(t *testing.T) {
	src := colorcore.PMColor4f{R: 0.75, G: 0.25, B: 0.5, A: 1}
	dst := colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.125, A: 0.5}
	for mode := raster.BlendClear; mode <= raster.BlendLuminosity; mode++ {
		want := !cpuBlendMismatchesGPU[mode]
		fp := BlendFP(MakeColorFP(src), MakeColorFP(dst), mode)
		if got := fp.fpBase().HasConstantOutputForConstantInput(); got != want {
			t.Errorf("mode %d: constant-output flag = %v, want %v", mode, got, want)
			continue
		}
		if !want {
			continue
		}
		wantColor := raster.BlendHighp4f(mode, src, dst)
		if got := fpConstantOutputForConstantInput(fp, colorcore.PMColor4f{}); got != wantColor {
			t.Errorf("mode %d: folded output = %v, want %v", mode, got, wantColor)
		}
	}

	// A child without a constant output blocks folding even for a matching mode.
	fp := BlendFP(newTestCoordsFP(), MakeColorFP(dst), raster.BlendMultiply)
	if fp.fpBase().HasConstantOutputForConstantInput() {
		t.Error("a non-constant src child must suppress constant folding")
	}
	fp = BlendFP(MakeColorFP(src), newTestCoordsFP(), raster.BlendMultiply)
	if fp.fpBase().HasConstantOutputForConstantInput() {
		t.Error("a non-constant dst child must suppress constant folding")
	}

	// Nil children read the input color, which is still constant, so folding stays available.
	if !BlendFP(nil, nil, raster.BlendMultiply).fpBase().HasConstantOutputForConstantInput() {
		t.Error("nil children must not suppress constant folding")
	}
}
