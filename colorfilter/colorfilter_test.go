// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package colorfilter

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// filter runs a filter over one premultiplied color via the pipeline, mirroring onFilterColor4f.
func filter(t *testing.T, cf shaders.ColorFilter, c colorcore.PMColor4f) colorcore.PMColor4f {
	t.Helper()
	out, ok := shaders.FilterColor4f(cf, c)
	if !ok {
		t.Fatal("filter failed to append stages")
	}
	return out
}

func near(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func nearColor(a, b colorcore.PMColor4f, tol float32) bool {
	return near(a.R, b.R, tol) && near(a.G, b.G, tol) && near(a.B, b.B, tol) && near(a.A, b.A, tol)
}

func TestMatrixFilter(t *testing.T) {
	identity := [20]float32{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
	}
	cf := NewMatrix(&identity)
	if cf == nil {
		t.Fatal("identity matrix rejected")
	}
	if !cf.IsAlphaUnchanged() {
		t.Error("identity matrix should not change alpha")
	}
	in := colorcore.Color4f{R: 0.8, G: 0.4, B: 0.2, A: 0.5}.Premul()
	if out := filter(t, cf, in); !nearColor(out, in, 1e-6) {
		t.Errorf("identity matrix changed the color: %+v -> %+v", in, out)
	}

	// A translate row in unpremul space: r' = r + 0.1 unpremul, then re-premultiplied.
	translate := identity
	translate[4] = 0.1
	cf = NewMatrix(&translate)
	out := filter(t, cf, in)
	want := colorcore.Color4f{R: 0.8 + 0.1, G: 0.4, B: 0.2, A: 0.5}.Premul()
	if !nearColor(out, want, 1e-6) {
		t.Errorf("translate row: got %+v, want %+v", out, want)
	}

	// Clamp: a big scale pins at 1 in unpremul space.
	scale := identity
	scale[0] = 10
	cf = NewMatrix(&scale)
	out = filter(t, cf, in)
	want = colorcore.Color4f{R: 1, G: 0.4, B: 0.2, A: 0.5}.Premul()
	if !nearColor(out, want, 1e-6) {
		t.Errorf("clamped scale: got %+v, want %+v", out, want)
	}

	// The alpha-row detector.
	alphaRow := identity
	alphaRow[15] = 0.5
	if af := NewMatrix(&alphaRow); af.IsAlphaUnchanged() {
		t.Error("alpha-consuming matrix reported alpha unchanged")
	}

	// Non-finite input rejected.
	bad := identity
	bad[7] = float32(math.Inf(1))
	if NewMatrix(&bad) != nil {
		t.Error("non-finite matrix accepted")
	}
}

func TestBlendFilterCollapseRules(t *testing.T) {
	// kDst is a no-op.
	if NewBlend(0x80336699, raster.BlendDst) != nil {
		t.Error("kDst should collapse to nil")
	}
	// srcover with alpha 0 → dst → nil.
	if NewBlend(0x00336699, raster.BlendSrcOver) != nil {
		t.Error("srcover alpha 0 should collapse to nil")
	}
	// dstin with alpha 1 → nil.
	if NewBlend(0xFF336699, raster.BlendDstIn) != nil {
		t.Error("dstin alpha 1 should collapse to nil")
	}
	// alpha 0 noop set.
	for _, mode := range []raster.BlendMode{
		raster.BlendDstOver, raster.BlendDstOut,
		raster.BlendSrcATop, raster.BlendXor, raster.BlendDarken,
	} {
		if NewBlend(0x00FFFFFF, mode) != nil {
			t.Errorf("mode %d with alpha 0 should collapse to nil", mode)
		}
	}
	// srcover with alpha 1 → src.
	cf := NewBlend(0xFF336699, raster.BlendSrcOver)
	bf, ok := cf.(*blendFilter)
	if !ok {
		t.Fatal("expected blendFilter")
	}
	if bf.Mode() != raster.BlendSrc {
		t.Errorf("opaque srcover should reduce to src, got %d", bf.Mode())
	}
	// kClear → (src, transparent).
	cf = NewBlend(0x80336699, raster.BlendClear)
	bf = cf.(*blendFilter)
	if bf.Mode() != raster.BlendSrc || bf.Color() != (colorcore.Color4f{}) {
		t.Errorf("clear should become src with transparent black, got %d %+v", bf.Mode(), bf.Color())
	}
	// Out-of-range mode rejected.
	if NewBlend(0x80336699, raster.BlendLuminosity+1) != nil {
		t.Error("invalid mode accepted")
	}
	// IsAlphaUnchanged for srcatop.
	if !NewBlend(0x80336699, raster.BlendSrcATop).IsAlphaUnchanged() {
		t.Error("srcatop should leave alpha unchanged")
	}
}

func TestBlendFilterMath(t *testing.T) {
	// src mode replaces the color entirely (premultiplied constant).
	cf := NewBlend(0x80FF0000, raster.BlendSrc)
	out := filter(t, cf, colorcore.Color4f{R: 0.1, G: 0.9, B: 0.3, A: 1}.Premul())
	want := colorcore.Color4fFromColor(0x80FF0000).Premul()
	if !nearColor(out, want, 1e-6) {
		t.Errorf("src blend: got %+v, want %+v", out, want)
	}

	// multiply of an opaque half-gray over an opaque full-red: r = s*d.
	cf = NewBlend(0xFF808080, raster.BlendMultiply)
	out = filter(t, cf, colorcore.Color4f{R: 1, G: 0.5, B: 0, A: 1}.Premul())
	g := float32(0x80) / 255
	want = colorcore.PMColor4f{R: g * 1, G: g * 0.5, B: 0, A: 1}
	if !nearColor(out, want, 1e-6) {
		t.Errorf("multiply blend: got %+v, want %+v", out, want)
	}
}

func TestComposeFilter(t *testing.T) {
	half := [20]float32{
		0.5, 0, 0, 0, 0,
		0, 0.5, 0, 0, 0,
		0, 0, 0.5, 0, 0,
		0, 0, 0, 1, 0,
	}
	plus := [20]float32{
		1, 0, 0, 0, 0.25,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
	}
	inner := NewMatrix(&half)
	outer := NewMatrix(&plus)
	if NewCompose(nil, inner) != inner {
		t.Error("nil outer should return inner")
	}
	if NewCompose(outer, nil) != outer {
		t.Error("nil inner should return outer")
	}
	cf := NewCompose(outer, inner)
	if !cf.IsAlphaUnchanged() {
		t.Error("compose of alpha-preserving filters should preserve alpha")
	}
	// inner first: r = 1*0.5 + 0.25 = 0.75 (order matters: outer(inner(c))).
	out := filter(t, cf, colorcore.PMColor4f{R: 1, G: 0, B: 0, A: 1})
	if !near(out.R, 0.75, 1e-6) {
		t.Errorf("compose order wrong: r = %v, want 0.75", out.R)
	}
}

func TestLightingFilter(t *testing.T) {
	// Black add collapses to a modulate blend filter.
	cf := NewLighting(0xFF804020, 0xFF000000)
	bf, ok := cf.(*blendFilter)
	if !ok {
		t.Fatalf("black add should collapse to a blend filter, got %T", cf)
	}
	if bf.Mode() != raster.BlendModulate {
		t.Errorf("collapse mode = %d, want modulate", bf.Mode())
	}
	// Non-black add builds a matrix: out = mul*c + add on RGB.
	cf = NewLighting(0xFF800000, 0xFF002000)
	out := filter(t, cf, colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	wantR := float32(0x80) * 0.00392156862745
	wantG := float32(0x20) * 0.00392156862745
	if !near(out.R, wantR, 1e-6) || !near(out.G, 1*0+wantG, 1e-6) || !near(out.B, 0, 1e-6) {
		t.Errorf("lighting matrix: got %+v, want r=%v g=%v b=0", out, wantR, wantG)
	}
}

func TestLumaFilter(t *testing.T) {
	cf := NewLuma()
	if cf.IsAlphaUnchanged() {
		t.Error("luma replaces alpha")
	}
	// Operates on the premultiplied color: rgb zero, alpha = dot(coeff, premul rgb).
	in := colorcore.PMColor4f{R: 0.5, G: 0.25, B: 1, A: 1}
	out := filter(t, cf, in)
	want := 0.2126*0.5 + 0.7152*0.25 + 0.0722*1
	if out.R != 0 || out.G != 0 || out.B != 0 || !near(out.A, float32(want), 1e-6) {
		t.Errorf("luma: got %+v, want a=%v", out, want)
	}
}

func TestHighContrastFilter(t *testing.T) {
	// Invalid configs rejected.
	if NewHighContrast(HighContrastConfig{Contrast: 2}) != nil {
		t.Error("contrast > 1 accepted")
	}
	if NewHighContrast(HighContrastConfig{InvertStyle: 3}) != nil {
		t.Error("invalid invert style accepted")
	}

	// A no-op config (no grayscale, no invert, contrast 0) round-trips through the working format: unpremul → linearize
	// → encode → premul must return the input within the transfer function's approximation error.
	cf := NewHighContrast(HighContrastConfig{})
	if !cf.IsAlphaUnchanged() {
		t.Error("high contrast preserves alpha")
	}
	in := colorcore.Color4f{R: 0.8, G: 0.4, B: 0.1, A: 0.75}.Premul()
	out := filter(t, cf, in)
	if !nearColor(out, in, 2.0/255) {
		t.Errorf("no-op config: got %+v, want ~%+v", out, in)
	}

	// Brightness inversion of opaque white in linear space is black.
	cf = NewHighContrast(HighContrastConfig{InvertStyle: InvertBrightness})
	out = filter(t, cf, colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	if !nearColor(out, colorcore.PMColor4f{A: 1}, 1e-3) {
		t.Errorf("invert white: got %+v, want black", out)
	}

	// Grayscale maps pure green to the luma constant (in linear space; g=1 is a fixed point of the transfer function so
	// the result encodes back to the same value).
	cf = NewHighContrast(HighContrastConfig{Grayscale: true})
	out = filter(t, cf, colorcore.PMColor4f{G: 1, A: 1})
	if !near(out.R, out.G, 1e-6) || !near(out.G, out.B, 1e-6) {
		t.Errorf("grayscale output not gray: %+v", out)
	}
}

func TestSRGBTransferFunctionInverse(t *testing.T) {
	// The derived inverse constants have closed forms: check them.
	if !near(srgbInvTF.G, 1/2.4, 1e-6) {
		t.Errorf("inv G = %v", srgbInvTF.G)
	}
	if !near(srgbInvTF.C, 12.92, 1e-4) {
		t.Errorf("inv C = %v", srgbInvTF.C)
	}
	if !near(srgbInvTF.D, float32(0.04045/12.92), 1e-6) {
		t.Errorf("inv D = %v", srgbInvTF.D)
	}
	// The invariant the inversion preserves: inv(srgb(1)) == 1.
	if got := colorcore.EvalSkcmsTF(&srgbInvTF, colorcore.EvalSkcmsTF(&srgbTF, 1)); got != 1 {
		t.Errorf("inv(srgb(1)) = %v, want exactly 1", got)
	}
	// Round trips stay within a half step of 1/255 across the range.
	for i := 0; i <= 255; i++ {
		x := float32(i) / 255
		rt := colorcore.EvalSkcmsTF(&srgbInvTF, colorcore.EvalSkcmsTF(&srgbTF, x))
		if !near(rt, x, 0.5/255) {
			t.Errorf("round trip %v -> %v", x, rt)
		}
	}
}
