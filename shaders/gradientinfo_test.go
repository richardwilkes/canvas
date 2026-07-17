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

var gradStops = []colorcore.Color{colorcore.ARGB(255, 255, 0, 0), colorcore.ARGB(255, 0, 0, 255)}

func TestAsGradientLinear(t *testing.T) {
	g := NewLinearGradient(geom.Point{X: 1, Y: 2}, geom.Point{X: 3, Y: 4}, gradStops, nil, TileClamp, nil)
	grad, ok := AsGradientShader(g)
	if !ok {
		t.Fatal("linear gradient not reported as a gradient")
	}
	typ, info := grad.AsGradient()
	if typ != GradientLinear {
		t.Errorf("type = %v, want GradientLinear", typ)
	}
	if info.Point[0] != (geom.Point{X: 1, Y: 2}) || info.Point[1] != (geom.Point{X: 3, Y: 4}) {
		t.Errorf("endpoints = %v, %v", info.Point[0], info.Point[1])
	}
	if len(info.Colors) != 2 || len(info.ColorOffsets) != 2 {
		t.Fatalf("colors/offsets = %d/%d, want 2/2", len(info.Colors), len(info.ColorOffsets))
	}
	// Uniform stops materialize to 0 and 1.
	if info.ColorOffsets[0] != 0 || info.ColorOffsets[1] != 1 {
		t.Errorf("offsets = %v, want [0 1]", info.ColorOffsets)
	}
	if info.Colors[0] != (colorcore.Color4f{R: 1, A: 1}) || info.Colors[1] != (colorcore.Color4f{B: 1, A: 1}) {
		t.Errorf("colors = %v", info.Colors)
	}
	if info.TileMode != TileClamp || info.GradientFlags != 0 {
		t.Errorf("tileMode=%v flags=%d", info.TileMode, info.GradientFlags)
	}
}

func TestAsGradientRadial(t *testing.T) {
	g := NewRadialGradient(geom.Point{X: 5, Y: 6}, 7, gradStops, nil, TileRepeat, nil)
	grad, _ := AsGradientShader(g)
	typ, info := grad.AsGradient()
	if typ != GradientRadial {
		t.Errorf("type = %v, want GradientRadial", typ)
	}
	if info.Point[0] != (geom.Point{X: 5, Y: 6}) || info.Radius[0] != 7 {
		t.Errorf("center/radius = %v/%v", info.Point[0], info.Radius[0])
	}
	if info.TileMode != TileRepeat {
		t.Errorf("tileMode = %v", info.TileMode)
	}
}

func TestAsGradientSweep(t *testing.T) {
	g := NewSweepGradient(geom.Point{X: 8, Y: 9}, gradStops, nil, TileClamp, 0, 180, nil)
	grad, _ := AsGradientShader(g)
	typ, info := grad.AsGradient()
	if typ != GradientSweep {
		t.Errorf("type = %v, want GradientSweep", typ)
	}
	// fPoint[0] is the center; fPoint[1] carries {tScale, tBias}. For 0..180deg: t0=0, t1=0.5, so tScale = 1/(t1-t0) =
	// 2 and tBias = -t0 = 0.
	if info.Point[0] != (geom.Point{X: 8, Y: 9}) {
		t.Errorf("center = %v", info.Point[0])
	}
	if info.Point[1].X != 2 || info.Point[1].Y != 0 {
		t.Errorf("tScale/tBias = %v/%v, want 2/0", info.Point[1].X, info.Point[1].Y)
	}
}

func TestAsGradientConical(t *testing.T) {
	g := NewTwoPointConicalGradient(geom.Point{X: 1, Y: 1}, 2, geom.Point{X: 9, Y: 9}, 4, gradStops, nil,
		TileClamp, nil)
	grad, ok := AsGradientShader(g)
	if !ok {
		t.Fatal("conical gradient not reported as a gradient")
	}
	typ, info := grad.AsGradient()
	if typ != GradientConical {
		t.Errorf("type = %v, want GradientConical", typ)
	}
	if info.Point[0] != (geom.Point{X: 1, Y: 1}) || info.Point[1] != (geom.Point{X: 9, Y: 9}) {
		t.Errorf("centers = %v, %v", info.Point[0], info.Point[1])
	}
	if info.Radius[0] != 2 || info.Radius[1] != 4 {
		t.Errorf("radii = %v, %v", info.Radius[0], info.Radius[1])
	}
}

func TestAsGradientSeesThroughLocalMatrix(t *testing.T) {
	lm := geom.TranslateMatrix(10, 20)
	g := NewLinearGradient(geom.Point{}, geom.Point{X: 1}, gradStops, nil, TileClamp, &lm)
	if _, isLM := g.(*LocalMatrixShader); !isLM {
		t.Fatal("expected the local matrix to wrap the gradient")
	}
	grad, ok := AsGradientShader(g)
	if !ok {
		t.Fatal("AsGradientShader did not see through the local-matrix wrapper")
	}
	if typ, _ := grad.AsGradient(); typ != GradientLinear {
		t.Errorf("unwrapped type = %v, want GradientLinear", typ)
	}
}

func TestAsGradientNonGradient(t *testing.T) {
	if _, ok := AsGradientShader(NewColor(colorcore.ARGB(255, 1, 2, 3))); ok {
		t.Error("a color shader must not report as a gradient")
	}
}

func TestAsGradientExplicitOffsets(t *testing.T) {
	// Non-uniform stops are reported verbatim (0, 0.25, 1).
	colors := []colorcore.Color{
		colorcore.ARGB(255, 255, 0, 0),
		colorcore.ARGB(255, 0, 255, 0),
		colorcore.ARGB(255, 0, 0, 255),
	}
	pos := []float32{0, 0.25, 1}
	g := NewLinearGradient(geom.Point{}, geom.Point{X: 1}, colors, pos, TileClamp, nil)
	grad, _ := AsGradientShader(g)
	_, info := grad.AsGradient()
	if len(info.ColorOffsets) != 3 {
		t.Fatalf("offsets = %v", info.ColorOffsets)
	}
	if info.ColorOffsets[0] != 0 || info.ColorOffsets[1] != 0.25 || info.ColorOffsets[2] != 1 {
		t.Errorf("offsets = %v, want [0 .25 1]", info.ColorOffsets)
	}
}
