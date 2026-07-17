// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the FP catalog: gradient/color-filter/dither/texture-effect swatches rendered through the
// paint-to-FP conversion plus the fill-rect op and compared against the CPU shader pipeline (the Go-GPU vs Go-CPU
// self-consistency lane), plus the advanced blend modes through the dst-copy lane compared against the raster blend
// kernels. Skips when no GL context is available.

package gl_test

import (
	"fmt"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// cpuShadePixel evaluates the CPU shader pipeline at one device pixel center under ctm.
func cpuShadePixel(t *testing.T, shader shaders.Shader, ctm geom.Matrix, paint colorcore.Color, x, y int32) colorcore.PMColor4f {
	t.Helper()
	p := shaders.Compile(shader, ctm, paint)
	if p == nil {
		t.Fatal("CPU pipeline compile failed")
	}
	var out [1]colorcore.PMColor4f
	p.ShadeSpan(x, y, out[:])
	return out[0]
}

// checkPixelTol compares a readback pixel against a premul float expectation within tol bytes.
func checkPixelTol(t *testing.T, data []byte, rowBytes, x, y int32, want colorcore.PMColor4f, tol int, what string) {
	t.Helper()
	got := data[y*rowBytes+x*4 : y*rowBytes+x*4+4]
	wb := pmToBytes(want)
	for c := 0; c < 4; c++ {
		d := int(got[c]) - int(wb[c])
		if d < -tol || d > tol {
			t.Fatalf("%s: pixel (%d,%d) = %v, want %v ±%d", what, x, y, got, wb, tol)
		}
	}
}

// TestLiveGradientSwatches renders each gradient family through the FP catalog and compares interior pixels against the
// CPU pipeline.
func TestLiveGradientSwatches(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 32
	ctm := geom.IdentityMatrix()

	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0x800000FF}
	positioned := []colorcore.Color{0xFF112233, 0xFFDDCCBB, 0xFF44FF66, 0xFF9900AA, 0xFF00FFFF}
	positions := []float32{0, 0.2, 0.5, 0.8, 1}

	cases := []struct {
		shader shaders.Shader
		name   string
	}{
		{name: "linear-clamp", shader: shaders.NewLinearGradient(geom.Point{X: 2}, geom.Point{X: 30}, colors,
			nil, shaders.TileClamp, nil)},
		{name: "linear-repeat-5stop", shader: shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 11},
			positioned, positions, shaders.TileRepeat, nil)},
		{name: "linear-mirror", shader: shaders.NewLinearGradient(geom.Point{X: 4}, geom.Point{X: 14}, colors,
			nil, shaders.TileMirror, nil)},
		{name: "radial", shader: shaders.NewRadialGradient(geom.Point{X: 16, Y: 16}, 14, colors, nil,
			shaders.TileClamp, nil)},
		{name: "sweep", shader: shaders.NewSweepGradient(geom.Point{X: 16, Y: 16}, positioned, positions,
			shaders.TileClamp, 0, 360, nil)},
		{name: "conical-focal", shader: shaders.NewTwoPointConicalGradient(geom.Point{X: 10, Y: 16}, 2,
			geom.Point{X: 22, Y: 16}, 12, colors, nil, shaders.TileClamp, nil)},
		{name: "conical-strip", shader: shaders.NewTwoPointConicalGradient(geom.Point{X: 8, Y: 16}, 6,
			geom.Point{X: 24, Y: 16}, 6, colors, nil, shaders.TileClamp, nil)},
	}
	samples := [][2]int32{{3, 3}, {9, 16}, {16, 16}, {23, 8}, {28, 28}}
	for _, tc := range cases {
		if tc.shader == nil {
			t.Fatalf("%s: shader construction failed", tc.name)
		}
		sdc := newLiveSDC(t, dc, size)
		sdc.Clear([4]float32{0, 0, 0, 0})
		gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
			Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
			BlendMode: raster.BlendSrc,
			Shader:    tc.shader,
		}, ctm)
		if !ok {
			t.Fatalf("%s: MakePaint failed", tc.name)
		}
		sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: size, Bottom: size})
		data := readSDC(t, sdc)
		for _, s := range samples {
			want := cpuShadePixel(t, tc.shader, ctm, 0xFFFFFFFF, s[0], s[1])
			// The CPU pipeline evaluates lerp-by-search; the GPU colorizers evaluate scale*t+bias (and the
			// sweep/conical lanes differ in atan/rsqrt precision): allow ±2/255.
			checkPixelTol(t, data, size*4, s[0], s[1], want, 2, tc.name)
		}
		sweepGLErrors(t, dc.Gpu(), tc.name)
		sdc.Release()
	}
}

// TestLiveGradientSwatchesUnderCTM re-renders gradient swatches under non-identity CTMs and compares device pixels
// against the CPU pipeline compiled with the same CTM. The FP's sample coords are already local-space, so
// ApplyForFragmentProcessor must invert only the pending local matrix; a regression that folds the CTM back into that
// inversion shifts every GPU gradient the moment the canvas is translated/scaled/rotated — which the identity-CTM
// swatches above can never see (with identity CTM the two inversions coincide). The local-matrix case exercises the
// pending local matrix and the CTM together, so dropping the inversion entirely also fails.
func TestLiveGradientSwatchesUnderCTM(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 32

	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0x800000FF}
	var lm geom.Matrix
	lm.SetScaleTranslate(0.5, 0.5, 6, 2)
	shaderCases := []struct {
		shader shaders.Shader
		name   string
	}{
		{name: "linear", shader: shaders.NewLinearGradient(geom.Point{X: 2}, geom.Point{X: 30}, colors, nil,
			shaders.TileClamp, nil)},
		{name: "linear-local-matrix", shader: shaders.NewLinearGradient(geom.Point{X: 2}, geom.Point{X: 30},
			colors, nil, shaders.TileClamp, &lm)},
		{name: "radial", shader: shaders.NewRadialGradient(geom.Point{X: 16, Y: 16}, 14, colors, nil,
			shaders.TileClamp, nil)},
		{name: "sweep", shader: shaders.NewSweepGradient(geom.Point{X: 16, Y: 16}, colors, nil,
			shaders.TileClamp, 0, 360, nil)},
		{name: "conical-focal", shader: shaders.NewTwoPointConicalGradient(geom.Point{X: 10, Y: 16}, 2,
			geom.Point{X: 22, Y: 16}, 12, colors, nil, shaders.TileClamp, nil)},
	}

	var scaleTranslate, rotated geom.Matrix
	scaleTranslate.SetScaleTranslate(0.75, 1.25, 5, -3)
	rotated.SetRotatePivot(30, 16, 16)
	ctms := []struct {
		name string
		ctm  geom.Matrix
	}{
		{name: "scale-translate", ctm: scaleTranslate},
		{name: "rotated", ctm: rotated},
	}

	// A local-space rect large enough that every device pixel is interior under both CTMs.
	drawRect := geom.Rect{Left: -2 * size, Top: -2 * size, Right: 2 * size, Bottom: 2 * size}
	samples := [][2]int32{{3, 3}, {9, 16}, {16, 16}, {23, 8}, {28, 28}}
	for _, mc := range ctms {
		for _, tc := range shaderCases {
			if tc.shader == nil {
				t.Fatalf("%s: shader construction failed", tc.name)
			}
			sdc := newLiveSDC(t, dc, size)
			sdc.Clear([4]float32{0, 0, 0, 0})
			gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
				Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
				BlendMode: raster.BlendSrc,
				Shader:    tc.shader,
			}, mc.ctm)
			if !ok {
				t.Fatalf("%s/%s: MakePaint failed", mc.name, tc.name)
			}
			sdc.FillRectWithPaint(gpuPaint, &mc.ctm, drawRect)
			data := readSDC(t, sdc)
			for _, s := range samples {
				want := cpuShadePixel(t, tc.shader, mc.ctm, 0xFFFFFFFF, s[0], s[1])
				// Same colorizer/precision spread as the identity-CTM swatches: allow ±2/255.
				checkPixelTol(t, data, size*4, s[0], s[1], want, 2,
					fmt.Sprintf("%s/%s (%d,%d)", mc.name, tc.name, s[0], s[1]))
			}
			sweepGLErrors(t, dc.Gpu(), mc.name+"/"+tc.name)
			sdc.Release()
		}
	}
}

// TestLiveAdvancedBlendSwatches renders the 15 advanced blend modes over a translucent dst through the dst-copy lane
// and compares against the CPU blend kernels.
func TestLiveAdvancedBlendSwatches(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env

	dstColor := colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.125, A: 0.5}
	src := colorcore.Color4f{R: 0.4, G: 0.1, B: 0.6, A: 0.75} // unpremul paint color
	srcPM := colorcore.PMColor4f{R: src.R * src.A, G: src.G * src.A, B: src.B * src.A, A: src.A}
	ctm := geom.IdentityMatrix()

	for mode := raster.BlendOverlay; mode <= raster.BlendLuminosity; mode++ {
		sdc := newLiveSDC(t, dc, 16)
		sdc.Clear([4]float32{dstColor.R, dstColor.G, dstColor.B, dstColor.A})
		// The clear's float->UNORM8 conversion is driver-dependent: most drivers round to nearest, but Windows AMD
		// (FireGL 27.20) truncates, landing 0.25/0.5/0.125 one code value below the rounded results. Read the cleared
		// dst back and state both checks against what the driver actually produced instead of assuming a rounding mode.
		pre := readSDC(t, sdc)
		const po = 8*(16*4) + 8*4
		preDst := colorcore.PMColor4f{
			R: float32(pre[po]) / 255, G: float32(pre[po+1]) / 255,
			B: float32(pre[po+2]) / 255, A: float32(pre[po+3]) / 255,
		}

		gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
			Color:     src,
			BlendMode: mode,
		}, ctm)
		if !ok {
			t.Fatalf("mode %d: conversion failed", mode)
		}
		sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Left: 4, Top: 4, Right: 12, Bottom: 12})

		data := readSDC(t, sdc)
		want := raster.BlendHighp4f(mode, srcPM, preDst)
		// Advanced blend math runs in the shader against a dst copy; the guarded divides and pow() in soft-light
		// introduce a little more rounding than the coefficient modes.
		checkPixelTol(t, data, 16*4, 8, 8, want, 2, fmt.Sprintf("mode %d inside", mode))
		checkPixelTol(t, data, 16*4, 1, 1, preDst, 0, fmt.Sprintf("mode %d outside", mode))
		sweepGLErrors(t, dc.Gpu(), "advanced blend swatch")
		sdc.Release()
	}
}

// TestLiveColorFilterSwatches renders a gradient through matrix/luma/high-contrast/compose color filters and compares
// against the CPU ColorFilterShader lane.
func TestLiveColorFilterSwatches(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 16
	ctm := geom.IdentityMatrix()

	base := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: size},
		[]colorcore.Color{0xFFFF4020, 0x9020C0FF}, nil, shaders.TileClamp, nil)

	var mat [20]float32
	// A saturating channel mix with translates.
	mat[0], mat[1], mat[4] = 0.5, 0.5, 0.1
	mat[6], mat[9] = 0.8, -0.05
	mat[12], mat[10] = 0.6, 0.4
	mat[18] = 1

	filters := []struct {
		cf   shaders.ColorFilter
		name string
	}{
		{name: "matrix", cf: colorfilter.NewMatrix(&mat)},
		{name: "luma", cf: colorfilter.NewLuma()},
		{name: "lighting", cf: colorfilter.NewLighting(0xFFC08040, 0xFF102030)},
		{name: "compose", cf: colorfilter.NewCompose(colorfilter.NewBlend(0x8040A060, raster.BlendScreen),
			colorfilter.NewLuma())},
		{name: "high-contrast", cf: colorfilter.NewHighContrast(colorfilter.HighContrastConfig{
			Grayscale: true, InvertStyle: colorfilter.InvertLightness, Contrast: 0.3,
		})},
	}
	for _, tc := range filters {
		// The CPU expectation goes through the same descriptor via the ColorFilterShader lane.
		cpuShader := shaders.NewWithColorFilter(base, tc.cf)
		sdc := newLiveSDC(t, dc, size)
		sdc.Clear([4]float32{0, 0, 0, 0})
		gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
			Color:       colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
			BlendMode:   raster.BlendSrc,
			Shader:      base,
			ColorFilter: tc.cf,
		}, ctm)
		if !ok {
			t.Fatalf("%s: conversion failed", tc.name)
		}
		sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: size, Bottom: size})
		data := readSDC(t, sdc)
		for _, x := range []int32{1, 5, 8, 12, 14} {
			want := cpuShadePixel(t, cpuShader, ctm, 0xFFFFFFFF, x, 8)
			// High contrast stacks two transfer-function evaluations (approx_powf on the CPU, pow() on the GPU): allow
			// ±3; the linear filters stay within ±2.
			tol := 2
			if tc.name == "high-contrast" {
				tol = 3
			}
			checkPixelTol(t, data, size*4, x, 8, want, tol, tc.name)
		}
		sweepGLErrors(t, dc.Gpu(), tc.name)
		sdc.Release()
	}
}

// TestLiveDitherSwatch renders a shallow ramp with dithering and compares each pixel against the CPU pipeline with the
// dither stage appended — the 8x8 table lookup must reproduce the CPU stage's ordered-dither pattern.
func TestLiveDitherSwatch(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 16
	ctm := geom.IdentityMatrix()

	ramp := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: size},
		[]colorcore.Color{0xFF404040, 0xFF444444}, nil, shaders.TileClamp, nil)

	sdc := newLiveSDC(t, dc, size)
	sdc.Clear([4]float32{0, 0, 0, 0})
	gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
		Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
		BlendMode: raster.BlendSrc,
		Shader:    ramp,
		Dither:    true,
	}, ctm)
	if !ok {
		t.Fatal("conversion failed")
	}
	sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: size, Bottom: size})
	data := readSDC(t, sdc)

	differs := false
	for y := int32(0); y < size; y++ {
		for x := int32(0); x < size; x++ {
			p := shaders.Compile(ramp, geom.IdentityMatrix(), 0xFFFFFFFF)
			p.AppendDither(1.0 / 255)
			var out [1]colorcore.PMColor4f
			p.ShadeSpan(x, y, out[:])
			checkPixelTol(t, data, size*4, x, y, out[0], 1, "dither")
			if data[y*size*4+x*4] != data[0] {
				differs = true
			}
		}
	}
	if !differs {
		t.Fatal("dither should produce a non-uniform byte pattern")
	}
	sweepGLErrors(t, dc.Gpu(), "dither")
}

// TestLiveTextureEffectSwatch uploads a 2x2 texture and samples it with nearest filtering over an 8x8 rect (each texel
// covering a 4x4 block).
func TestLiveTextureEffectSwatch(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 8
	ctm := geom.IdentityMatrix()

	pixels := []byte{
		255, 0, 0, 255, 0, 255, 0, 255, // red, green
		0, 0, 255, 255, 255, 255, 255, 255, // blue, white
	}
	var key gpu.UniqueKey
	builder := gpu.UniqueKeyBuilder(&key, gpu.GenerateUniqueKeyDomain(), 1, "TestTexture")
	builder.Slice()[0] = 42
	builder.Finish()
	view := gl.FindOrCreatePixelsProxyView(dc, &key, geom.ISize{Width: 2, Height: 2},
		gpu.ColorTypeRGBA8888, pixels, 8, "live-te-test")
	if !view.IsValid() {
		t.Fatal("FindOrCreatePixelsProxyView failed")
	}

	// Map the 8x8 local space onto the 2x2 texel space.
	matrix := geom.Matrix{}
	matrix.SetScale(2.0/size, 2.0/size)
	sampler := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, gpu.FilterModeNearest,
		gpu.MipmapModeNone)
	te := gl.MakeTextureEffect(view, gpu.AlphaTypePremul, matrix, sampler, dc.GLCaps(),
		[4]float32{})

	sdc := newLiveSDC(t, dc, size)
	sdc.Clear([4]float32{0, 0, 0, 0})
	paint := gl.NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	paint.SetPorterDuffXPFactory(raster.BlendSrc)
	paint.SetColorFragmentProcessor(te)
	sdc.FillRectWithPaint(paint, &ctm, geom.Rect{Right: size, Bottom: size})

	data := readSDC(t, sdc)
	checkPixel(t, data, size*4, 1, 1, [4]byte{255, 0, 0, 255}, "top-left texel")
	checkPixel(t, data, size*4, 6, 1, [4]byte{0, 255, 0, 255}, "top-right texel")
	checkPixel(t, data, size*4, 1, 6, [4]byte{0, 0, 255, 255}, "bottom-left texel")
	checkPixel(t, data, size*4, 6, 6, [4]byte{255, 255, 255, 255}, "bottom-right texel")
	sweepGLErrors(t, dc.Gpu(), "texture effect")
}
