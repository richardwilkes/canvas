// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests.

package shaders

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// TestGradientDegenerateReturnsNil verifies the gradient constructors' null contract: no colors or a negative radius
// yield nil, valid inputs yield shaders.
func TestGradientDegenerateReturnsNil(t *testing.T) {
	colors := []colorcore.Color{0xFFFFFFFF, 0xFF000000}
	if s := NewLinearGradient(geom.Point{}, geom.Point{X: 10}, nil, nil, TileClamp, nil); s != nil {
		t.Error("linear gradient with no colors: want nil")
	}
	if s := NewLinearGradient(geom.Point{}, geom.Point{X: 10}, colors, nil, TileClamp, nil); s == nil {
		t.Error("valid linear gradient: want non-nil")
	}
	if s := NewRadialGradient(geom.Point{X: 5, Y: 5}, -1, colors, nil, TileClamp, nil); s != nil {
		t.Error("radial gradient with negative radius: want nil")
	}
	if s := NewRadialGradient(geom.Point{X: 5, Y: 5}, 8, colors, nil, TileClamp, nil); s == nil {
		t.Error("valid radial gradient: want non-nil")
	}
}

// TestWithLocalMatrixIdentity verifies NewWithLocalMatrix's identity short-circuit — wrapping with the identity returns
// the shader unchanged — while a real matrix wraps and changes sampling.
func TestWithLocalMatrixIdentity(t *testing.T) {
	colors := []colorcore.Color{0xFFFF0000, 0xFF0000FF}
	base := NewLinearGradient(geom.Point{}, geom.Point{X: 12}, colors, nil, TileRepeat, nil)
	if base == nil {
		t.Fatal("base gradient is nil")
	}
	if got := NewWithLocalMatrix(base, geom.IdentityMatrix()); got != base {
		t.Error("with-local-matrix(identity) should return the same shader")
	}

	var m geom.Matrix
	m.SetScaleTranslate(2.5, 1, 0, 0)
	wrapped := NewWithLocalMatrix(base, m)
	if wrapped == nil || wrapped == base {
		t.Fatal("with-local-matrix(scale) should wrap the shader")
	}
	// The scale must change the sampled color at a probe point: x=10 samples t=10/12 bare but t=4/12 wrapped (local
	// matrix maps device x into a 2.5x smaller gradient coordinate).
	bare := shadeAt(t, base, geom.IdentityMatrix(), 10, 0)
	scaled := shadeAt(t, wrapped, geom.IdentityMatrix(), 10, 0)
	if near(bare.R, scaled.R, 1e-3) && near(bare.B, scaled.B, 1e-3) {
		t.Errorf("local matrix did not change sampling: bare %+v vs scaled %+v", bare, scaled)
	}
}

// TestPerlinNoiseTraitsAndStitch verifies the perlin-noise shader's traits (never opaque, never constant) and the
// stitched-tile lane: a tile size snaps the base frequencies and still produces deterministic, non-constant noise.
func TestPerlinNoiseTraitsAndStitch(t *testing.T) {
	s := NewFractalNoise(0.2, 0.15, 3, 1, 8, 8)
	if s == nil {
		t.Fatal("fractal noise shader is nil")
	}
	pn, ok := s.(*PerlinNoiseShader)
	if !ok {
		t.Fatalf("fractal noise shader is %T, want *PerlinNoiseShader", s)
	}
	if pn.IsOpaque() {
		t.Error("perlin noise must never report opaque")
	}
	if pn.IsConstant() {
		t.Error("perlin noise must never report constant")
	}

	// Rendering through the stitched lane is deterministic and non-constant.
	a := shadeAt(t, s, geom.IdentityMatrix(), 1, 1)
	b := shadeAt(t, s, geom.IdentityMatrix(), 5, 3)
	c := shadeAt(t, s, geom.IdentityMatrix(), 1, 1)
	if a != c {
		t.Error("perlin noise is not deterministic for the same pixel")
	}
	if a == b {
		t.Error("perlin noise produced identical colors at distinct probes (suspiciously constant)")
	}
}
