// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the FP catalog: texture-effect sampling analysis, gradient colorizer selection, color-filter FP
// constant folding vs the CPU descriptors, custom-XP selection, the paint-to-FP conversion lanes, and a full dst-copy
// draw through the recording driver.

package gl

import (
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/surface"
)

func newTestTextureProxyView(t *testing.T, dc *DirectContext, w, h int32) SurfaceProxyView {
	t.Helper()
	caps := dc.GLCaps()
	format := caps.GetDefaultBackendFormat(gpu.ColorTypeRGBA8888, false)
	proxy := dc.ProxyProvider().CreateProxy(format, geom.ISize{Width: w, Height: h},
		gpu.RenderableNo, 1, gpu.MipmappedNo, gpu.BackingFitExact, gpu.BudgetedYes,
		"fp-test-texture", 0, UseAllocatorYes)
	if proxy == nil {
		t.Fatal("CreateProxy failed")
	}
	return MakeSurfaceProxyView(proxy, gpu.OriginTopLeft,
		caps.ReadSwizzle(format, gpu.ColorTypeRGBA8888))
}

func TestTextureEffectShaderModes(t *testing.T) {
	// GetShaderMode table.
	cases := []struct {
		wrap   gpu.WrapMode
		filter gpu.FilterMode
		mm     gpu.MipmapMode
		want   TextureEffectShaderMode
	}{
		{wrap: gpu.WrapModeClamp, filter: gpu.FilterModeLinear, mm: gpu.MipmapModeNone, want: ShaderModeClamp},
		{
			wrap: gpu.WrapModeMirrorRepeat, filter: gpu.FilterModeNearest, mm: gpu.MipmapModeNone,
			want: ShaderModeMirrorRepeat,
		},
		{
			wrap: gpu.WrapModeRepeat, filter: gpu.FilterModeNearest, mm: gpu.MipmapModeNone,
			want: ShaderModeRepeatNearestNone,
		},
		{wrap: gpu.WrapModeRepeat, filter: gpu.FilterModeLinear, mm: gpu.MipmapModeNone, want: ShaderModeRepeatLinearNone},
		{
			wrap: gpu.WrapModeRepeat, filter: gpu.FilterModeNearest, mm: gpu.MipmapModeLinear,
			want: ShaderModeRepeatNearestMipmap,
		},
		{
			wrap: gpu.WrapModeRepeat, filter: gpu.FilterModeLinear, mm: gpu.MipmapModeNearest,
			want: ShaderModeRepeatLinearMipmap,
		},
		{
			wrap: gpu.WrapModeClampToBorder, filter: gpu.FilterModeNearest, mm: gpu.MipmapModeNone,
			want: ShaderModeClampToBorderNearest,
		},
		{
			wrap: gpu.WrapModeClampToBorder, filter: gpu.FilterModeLinear, mm: gpu.MipmapModeNone,
			want: ShaderModeClampToBorderFilter,
		},
	}
	for _, tc := range cases {
		if got := getTextureEffectShaderMode(tc.wrap, tc.filter, tc.mm); got != tc.want {
			t.Fatalf("GetShaderMode(%d,%d,%d) = %d, want %d", tc.wrap, tc.filter, tc.mm, got,
				tc.want)
		}
	}
}

func TestTextureEffectSamplingAnalysis(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	view := newTestTextureProxyView(t, dc, 8, 8)
	caps := dc.GLCaps()

	// A full-texture repeat wrap on 2D with NPOT tiling support runs entirely in HW.
	sampler := gpu.MakeSamplerState(gpu.WrapModeRepeat, gpu.WrapModeRepeat, gpu.FilterModeLinear,
		gpu.MipmapModeNone)
	fullRect := geom.Rect{Right: 8, Bottom: 8}
	sampling := makeTextureEffectSampling(view.Proxy(), sampler, fullRect, nil, [4]float32{},
		false, &caps.Caps, geom.Point{X: 0.5, Y: 0.5})
	if sampling.shaderModes[0] != ShaderModeNone || sampling.shaderModes[1] != ShaderModeNone {
		t.Fatalf("full-texture repeat should be HW: %+v", sampling.shaderModes)
	}
	if sampling.hwSampler.WrapModeX != gpu.WrapModeRepeat {
		t.Fatalf("HW wrap should stay repeat, got %d", sampling.hwSampler.WrapModeX)
	}

	// A subset forces the wrap into the shader, with the linear clamp inset by half a texel.
	subset := geom.Rect{Left: 2, Top: 2, Right: 6, Bottom: 6}
	sampling = makeTextureEffectSampling(view.Proxy(), sampler, subset, nil, [4]float32{}, false,
		&caps.Caps, geom.Point{X: 0.5, Y: 0.5})
	if sampling.shaderModes[0] != ShaderModeRepeatLinearNone ||
		sampling.shaderModes[1] != ShaderModeRepeatLinearNone {
		t.Fatalf("subset repeat should be shader-based: %+v", sampling.shaderModes)
	}
	if sampling.hwSampler.WrapModeX != gpu.WrapModeClamp {
		t.Fatal("shader-based wrap must use HW clamp")
	}
	if sampling.shaderSubset != subset {
		t.Fatalf("shader subset = %+v, want %+v", sampling.shaderSubset, subset)
	}
	want := subset.Inset(TextureEffectLinearInset, TextureEffectLinearInset)
	if sampling.shaderClamp != want {
		t.Fatalf("shader clamp = %+v, want %+v", sampling.shaderClamp, want)
	}

	// Decal (clamp-to-border with a transparent border) has HW support in the desktop caps, but a non-zero border color
	// forces the shader lane.
	decal := gpu.MakeSamplerState(gpu.WrapModeClampToBorder, gpu.WrapModeClampToBorder,
		gpu.FilterModeNearest, gpu.MipmapModeNone)
	sampling = makeTextureEffectSampling(view.Proxy(), decal, fullRect, nil, [4]float32{}, false,
		&caps.Caps, geom.Point{X: 0.5, Y: 0.5})
	if caps.ClampToBorderSupport {
		if sampling.shaderModes[0] != ShaderModeNone {
			t.Fatalf("zero-border decal should be HW, got %d", sampling.shaderModes[0])
		}
	}
	sampling = makeTextureEffectSampling(view.Proxy(), decal, fullRect, nil,
		[4]float32{1, 0, 0, 1}, false, &caps.Caps, geom.Point{X: 0.5, Y: 0.5})
	if sampling.shaderModes[0] != ShaderModeClampToBorderNearest {
		t.Fatalf("colored-border decal should be shader-based, got %d", sampling.shaderModes[0])
	}

	view.Proxy().Unref()
}

// gradientColorizerOf digs the colorizer child out of a top-level gradient FP.
func gradientColorizerOf(t *testing.T, fp FragmentProcessor) FragmentProcessor {
	t.Helper()
	if fp == nil {
		t.Fatal("gradient FP is nil")
	}
	switch fp.(type) {
	case *clampedGradientFP, *tiledGradientFP:
	default:
		t.Fatalf("unexpected top-level gradient FP %T", fp)
	}
	colorizer := fp.fpBase().ChildProcessor(0)
	// Unwrap the interpolated-to-dst premul wrapper when present.
	if cx, ok := colorizer.(*colorXformFP); ok {
		colorizer = cx.ChildProcessor(0)
	}
	return colorizer
}

func TestGradientColorizerSelection(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	args := &FPArgs{Ctx: dc, Caps: dc.GLCaps(), DstColorType: gpu.ColorTypeRGBA8888}
	p0 := geom.Point{}
	p1 := geom.Point{X: 10}
	mRec := shaders.NewMatrixRec(geom.IdentityMatrix())

	makeFP := func(colors []colorcore.Color, pos []float32, tm shaders.TileMode) FragmentProcessor {
		shader := shaders.NewLinearGradient(p0, p1, colors, pos, tm, nil)
		if shader == nil {
			t.Fatal("NewLinearGradient failed")
		}
		return MakeShaderFP(shader, args, &mRec)
	}

	// Two stops → single interval.
	fp := makeFP([]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, shaders.TileClamp)
	if _, ok := gradientColorizerOf(t, fp).(*singleIntervalColorizerFP); !ok {
		t.Fatalf("2-stop colorizer is %T, want singleInterval", gradientColorizerOf(t, fp))
	}
	if _, ok := fp.(*clampedGradientFP); !ok {
		t.Fatalf("clamp tile mode should use the clamped gradient, got %T", fp)
	}

	// Three stops → dual interval.
	fp = makeFP([]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.25, 1},
		shaders.TileRepeat)
	if _, ok := gradientColorizerOf(t, fp).(*dualIntervalColorizerFP); !ok {
		t.Fatalf("3-stop colorizer is %T, want dualInterval", gradientColorizerOf(t, fp))
	}
	if tiled, ok := fp.(*tiledGradientFP); !ok || tiled.mirror {
		t.Fatalf("repeat tile mode should use the tiled gradient (mirror=false), got %T", fp)
	}

	// Six evenly spaced stops → looping colorizer with intervals rounded up to 8.
	fp = makeFP([]colorcore.Color{
		0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00, 0xFF00FFFF,
		0x80FF00FF,
	}, nil, shaders.TileMirror)
	looping, ok := gradientColorizerOf(t, fp).(*loopingColorizerFP)
	if !ok {
		t.Fatalf("6-stop colorizer is %T, want looping", gradientColorizerOf(t, fp))
	}
	if looping.intervalCount != 8 {
		t.Fatalf("interval count = %d, want 8 (5 rounded up)", looping.intervalCount)
	}
	// The translucent stop means the colorizer gets the premul wrapper.
	if _, ok = fp.fpBase().ChildProcessor(0).(*colorXformFP); !ok {
		t.Fatal("translucent stops must premultiply via the color-xform wrapper")
	}

	// Hard stops at both ends are stripped, leaving a single interval; the borders keep the stripped outer colors.
	fp = makeFP([]colorcore.Color{0xFF111111, 0xFFFF0000, 0xFF0000FF, 0xFF222222},
		[]float32{0, 0, 1, 1}, shaders.TileClamp)
	if _, ok = gradientColorizerOf(t, fp).(*singleIntervalColorizerFP); !ok {
		t.Fatalf("hard-stop-trimmed colorizer is %T, want singleInterval",
			gradientColorizerOf(t, fp))
	}
	clamped := fp.(*clampedGradientFP)
	wantLeft := colorcore.PMColor4f{
		R: float32(0x11) / 255, G: float32(0x11) / 255,
		B: float32(0x11) / 255, A: 1,
	}
	if clamped.leftBorderColor != wantLeft {
		t.Fatalf("left border = %+v, want %+v", clamped.leftBorderColor, wantLeft)
	}

	// Past the looping limit the textured colorizer takes over: the colorizer is the matrix-effect-wrapped texture
	// effect sampling the 256×1 LUT with clamp/linear.
	many := make([]colorcore.Color, 130)
	for i := range many {
		many[i] = colorcore.Color(0xFF000000 | uint32(i))
	}
	fp = makeFP(many, nil, shaders.TileClamp)
	if fp == nil {
		t.Fatal("130-stop gradient must convert via the textured colorizer")
	}
	me, ok := gradientColorizerOf(t, fp).(*matrixEffect)
	if !ok {
		t.Fatalf("130-stop colorizer is %T, want the matrix-effect-wrapped texture effect",
			gradientColorizerOf(t, fp))
	}
	te, ok := me.fpBase().ChildProcessor(0).(*TextureEffect)
	if !ok {
		t.Fatalf("matrix effect child is %T, want TextureEffect", me.fpBase().ChildProcessor(0))
	}
	if dims := te.View().Proxy().Dimensions(); dims.Width != gradientTextureSize || dims.Height != 1 {
		t.Fatalf("LUT texture is %dx%d, want %dx1", dims.Width, dims.Height, gradientTextureSize)
	}
	sampler := te.SamplerState()
	if sampler.Filter != gpu.FilterModeLinear || sampler.WrapModeX != gpu.WrapModeClamp {
		t.Fatalf("LUT sampler = %+v, want clamp/linear", sampler)
	}
	// An opaque >128-stop gradient bakes the interpolated-to-dst result directly, so no colorXform premul wrapper
	// appears even though the colorizer is not the analytic lane.
	if _, ok = fp.fpBase().ChildProcessor(0).(*colorXformFP); ok {
		t.Fatal("textured colorizer must not get the ColorXformFP premul wrapper")
	}

	// The second conversion of the same gradient hits the per-context bitmap cache.
	if entries := len(dc.gradientRampCache.entries); entries != 1 {
		t.Fatalf("gradient ramp cache holds %d entries, want 1", entries)
	}
	makeFP(many, nil, shaders.TileClamp)
	if entries := len(dc.gradientRampCache.entries); entries != 1 {
		t.Fatalf("re-converting the same gradient must hit the cache; %d entries", entries)
	}

	// The gradient program key changes with the colorizer: the textured lane keys differently from an analytic
	// (looping) colorizer gradient.
	loopingFP := makeFP([]colorcore.Color{
		0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00, 0xFF00FFFF,
		0xFFFF00FF,
	}, nil, shaders.TileClamp)
	var texturedKey, loopingKey []uint32
	kb := gpu.NewKeyBuilder(&texturedKey)
	fpAddToKey(fp, dc.Caps().ShaderCaps, kb)
	kb.Flush()
	kb = gpu.NewKeyBuilder(&loopingKey)
	fpAddToKey(loopingFP, dc.Caps().ShaderCaps, kb)
	kb.Flush()
	if len(texturedKey) == len(loopingKey) {
		same := true
		for i := range texturedKey {
			if texturedKey[i] != loopingKey[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatal("textured-colorizer gradient must key differently from the looping-colorizer gradient")
		}
	}
}

// TestFakeTexturedGradientDrawGLSL runs a >128-stop gradient draw through the recording driver and checks the generated
// GLSL structurally: the texture-effect child (the LUT sampler) appears in place of the looping colorizer's uniform
// arrays and search loop.
func TestFakeTexturedGradientDrawGLSL(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 16, Height: 16},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes,
		"textured-gradient-draw")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	many := make([]colorcore.Color, 200)
	for i := range many {
		many[i] = colorcore.Color(0xFF000000 | uint32(i*97)&0xFFFFFF)
	}
	shader := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 16}, many, nil,
		shaders.TileClamp, nil)
	ctm := geom.IdentityMatrix()
	gpuPaint, ok := MakePaint(sdc, &PaintParams{
		Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
		BlendMode: raster.BlendSrcOver,
		Shader:    shader,
	}, ctm)
	if !ok {
		t.Fatal("conversion failed — the textured colorizer fallback must not skip the draw")
	}
	sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: 16, Bottom: 16})
	dc.FlushAndSubmit(false)

	var fragment string
	for _, src := range capturedShaderSource {
		if strings.Contains(src, "fsColorOut") {
			fragment = src
		}
	}
	if fragment == "" {
		t.Fatal("no fragment shader captured")
	}
	for _, wantPiece := range []string{
		"sampler2D uTextureSampler_", // the LUT texture-effect child
		"leftBorderColor",            // still the clamped-gradient top level
	} {
		if !strings.Contains(fragment, wantPiece) {
			t.Fatalf("fragment shader missing %q:\n%s", wantPiece, fragment)
		}
	}
	for _, absentPiece := range []string{
		"uthresholds", // the looping colorizer's uniforms/loop must be gone
		"for (int loop",
	} {
		if strings.Contains(fragment, absentPiece) {
			t.Fatalf("fragment shader unexpectedly contains %q (analytic colorizer leaked in):\n%s",
				absentPiece, fragment)
		}
	}
	// The 256×1 LUT upload itself must have been recorded.
	if recCounts["glTexImage2D"] == 0 && recCounts["glTexSubImage2D"] == 0 &&
		recCounts["glTexStorage2D"] == 0 {
		t.Fatalf("no LUT texture upload recorded: %v", recCounts)
	}
}

func TestColorFilterFPConstantFoldMatchesCPU(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	args := &FPArgs{Ctx: dc, Caps: dc.GLCaps(), DstColorType: gpu.ColorTypeRGBA8888}

	var swapRB [20]float32
	swapRB[2] = 1  // R = B
	swapRB[6] = 1  // G = G
	swapRB[10] = 1 // B = B? no: row2 col0
	swapRB[10] = 0
	swapRB[8] = 0.25 // B row gets a translate via matrix col
	swapRB[10] = 1
	swapRB[18] = 1 // A = A
	filters := []struct {
		cf   shaders.ColorFilter
		name string
	}{
		{name: "matrix", cf: colorfilter.NewMatrix(&swapRB)},
		{name: "blend-multiply", cf: colorfilter.NewBlend(0x80336699, raster.BlendMultiply)},
		{name: "luma", cf: colorfilter.NewLuma()},
		{name: "lighting", cf: colorfilter.NewLighting(0xFF808080, 0xFF102030)},
		{name: "compose", cf: colorfilter.NewCompose(colorfilter.NewLuma(),
			colorfilter.NewBlend(0xFF33AA55, raster.BlendScreen))},
		{name: "high-contrast", cf: colorfilter.NewHighContrast(colorfilter.HighContrastConfig{
			Grayscale: false, InvertStyle: colorfilter.InvertBrightness, Contrast: 0.2,
		})},
	}
	inputs := []colorcore.PMColor4f{
		{R: 0.25, G: 0.5, B: 0.75, A: 1},
		{R: 0.2, G: 0.1, B: 0.4, A: 0.5},
	}
	for _, tc := range filters {
		for _, in := range inputs {
			fp, ok := MakeColorFilterFP(args, tc.cf, MakeColorFP(in))
			if !ok || fp == nil {
				t.Fatalf("%s: MakeColorFilterFP failed", tc.name)
			}
			if !fp.fpBase().HasConstantOutputForConstantInput() {
				t.Fatalf("%s: FP chain should fold constants", tc.name)
			}
			got := fpConstantOutputForConstantInput(fp, colorcore.PMColor4f{})
			want, cpuOK := shaders.FilterColor4f(tc.cf, in)
			if !cpuOK {
				t.Fatalf("%s: CPU FilterColor4f failed", tc.name)
			}
			// filterColor4f is unclamped; the FP lanes clamp where the corresponding effects clamp. Both land in range
			// for these inputs. The CPU pipeline uses an approximate power function while the FP folding uses scalar
			// color-space math, so allow a small tolerance.
			const tol = 2.0 / 255
			for i, pair := range [][2]float32{
				{got.R, want.R},
				{got.G, want.G},
				{got.B, want.B},
				{got.A, want.A},
			} {
				d := pair[0] - pair[1]
				if d < -tol || d > tol {
					t.Fatalf("%s input %+v channel %d: fp=%v cpu=%v", tc.name, in, i, got, want)
				}
			}
		}
	}

	// The descriptor constructor already collapses the dst-blend noop to nil; the FP-level early-out is exercised
	// directly.
	if colorfilter.NewBlend(0xFF112233, raster.BlendDst) != nil {
		t.Fatal("kDst blend filter must collapse to nil at construction")
	}
	in := MakeColorFP(inputs[0])
	fp, ok := makeBlendColorFilterFP(colorcore.Color4f{R: 1, A: 1}, raster.BlendDst, in)
	if !ok || fp != in {
		t.Fatal("kDst blend lane must return the input FP as-is")
	}
}

func TestCustomXPSelection(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	caps := &dc.GLCaps().Caps

	factory := XPFactoryFromBlendMode(raster.BlendOverlay)
	if factory == nil {
		t.Fatal("overlay must have an XP factory")
	}

	// The recording caps profile has no advanced-blend support: the dst-read lane is selected and the shared analysis
	// adds the dst-texture requirement.
	if caps.AdvancedBlendEquationSupport() {
		t.Skip("test caps profile unexpectedly supports advanced blends")
	}
	props := XPFactoryGetAnalysisProperties(factory, AnalysisColorUnknown(),
		AnalysisCoverageSingleChannel, caps, gpu.ClampTypeAuto)
	if props&XPAnalysisReadsDstInShader == 0 || props&XPAnalysisRequiresDstTexture == 0 {
		t.Fatalf("overlay without advanced blend must read a dst texture: %x", props)
	}
	xp := XPFactoryMakeXferProcessor(factory, AnalysisColorUnknown(),
		AnalysisCoverageSingleChannel, caps, gpu.ClampTypeAuto)
	custom := xp.(*customXP)
	if !custom.xpBase().WillReadDstColor() || custom.hasHWBlendEquation() {
		t.Fatal("expected the dst-read custom XP")
	}
	if custom.XferBarrierType(caps) != XferBarrierNone {
		t.Fatal("dst-read custom XP needs no barrier")
	}

	// With (non-coherent) advanced-blend caps the HW lane is selected, keyed on the equation, with a blend barrier and
	// non-overlapping draws.
	hwCaps := *caps
	hwCaps.BlendEquationSupport = gpu.AdvancedBlendEquationSupport
	props = XPFactoryGetAnalysisProperties(factory, AnalysisColorUnknown(),
		AnalysisCoverageSingleChannel, &hwCaps, gpu.ClampTypeAuto)
	if props&XPAnalysisUsesNonCoherentHWBlending == 0 ||
		props&XPAnalysisRequiresNonOverlappingDraws == 0 {
		t.Fatalf("non-coherent HW advanced blend properties wrong: %x", props)
	}
	xp = XPFactoryMakeXferProcessor(factory, AnalysisColorUnknown(),
		AnalysisCoverageSingleChannel, &hwCaps, gpu.ClampTypeAuto)
	custom = xp.(*customXP)
	if custom.xpBase().WillReadDstColor() || !custom.hasHWBlendEquation() {
		t.Fatal("expected the HW custom XP")
	}
	if custom.hwBlendEquation != gpu.BlendEquationOverlay {
		t.Fatalf("equation = %d, want overlay", custom.hwBlendEquation)
	}
	if custom.XferBarrierType(&hwCaps) != XferBarrierBlend {
		t.Fatal("non-coherent advanced blend needs a blend barrier")
	}
	info := xpGetBlendInfo(custom)
	if info.Equation != gpu.BlendEquationOverlay {
		t.Fatalf("blend info equation = %d, want overlay", info.Equation)
	}

	// LCD coverage vetoes the HW lane.
	xp = XPFactoryMakeXferProcessor(factory, AnalysisColorUnknown(), AnalysisCoverageLCD,
		&hwCaps, gpu.ClampTypeAuto)
	if !xp.(*customXP).xpBase().WillReadDstColor() {
		t.Fatal("LCD coverage must fall back to the dst-read lane")
	}
}

func TestMakePaintLanes(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 8, Height: 8},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "skgr-test")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()
	ctm := geom.IdentityMatrix()

	// Solid color with a color filter folds into the paint color (no FP).
	gpuPaint, ok := MakePaint(sdc, &PaintParams{
		Color:       colorcore.Color4f{R: 1, G: 0.5, B: 0, A: 1},
		BlendMode:   raster.BlendSrcOver,
		ColorFilter: colorfilter.NewLuma(),
	}, ctm)
	if !ok {
		t.Fatal("solid conversion failed")
	}
	if gpuPaint.HasColorFragmentProcessor() {
		t.Fatal("solid color + filter must fold into the paint color")
	}
	wantLuma := clamp01(0.2126*1 + 0.7152*0.5)
	if got := gpuPaint.Color4f(); geom.ScalarAbs(got.A-wantLuma) > 1e-6 || got.R != 0 {
		t.Fatalf("folded luma color = %+v, want a=%v", got, wantLuma)
	}

	// A translucent paint with a shader goes through ApplyPaintAlpha with the unpremul color.
	shader := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 8},
		[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, shaders.TileClamp, nil)
	gpuPaint, ok = MakePaint(sdc, &PaintParams{
		Color:     colorcore.Color4f{R: 0.5, G: 0.25, B: 1, A: 0.5},
		BlendMode: raster.BlendSrcOver,
		Shader:    shader,
	}, ctm)
	if !ok {
		t.Fatal("shader conversion failed")
	}
	if !gpuPaint.HasColorFragmentProcessor() {
		t.Fatal("shader paint must carry a color FP")
	}
	if _, isApply := gpuPaint.ColorFragmentProcessor().(*applyPaintAlphaFP); !isApply {
		t.Fatalf("translucent shader paint FP is %T, want ApplyPaintAlpha",
			gpuPaint.ColorFragmentProcessor())
	}
	if got := gpuPaint.Color4f(); got.R != 0.5 || got.A != 0.5 {
		t.Fatalf("paint color must stay unpremul for ApplyPaintAlpha: %+v", got)
	}

	// Dither wraps non-constant shaders on 8888 when requested.
	gpuPaint, ok = MakePaint(sdc, &PaintParams{
		Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
		BlendMode: raster.BlendSrcOver,
		Shader:    shader,
		Dither:    true,
	}, ctm)
	if !ok {
		t.Fatal("dither conversion failed")
	}
	if _, isDither := gpuPaint.ColorFragmentProcessor().(*ditherFP); !isDither {
		t.Fatalf("dithered shader paint FP is %T, want ditherFP",
			gpuPaint.ColorFragmentProcessor())
	}

	// A constant paint (no shader) never dithers.
	gpuPaint, ok = MakePaint(sdc, &PaintParams{
		Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
		BlendMode: raster.BlendSrcOver,
		Dither:    true,
	}, ctm)
	if !ok || gpuPaint.HasColorFragmentProcessor() {
		t.Fatal("constant paints must not dither")
	}

	// An advanced blend mode routes to the custom XP factory.
	gpuPaint, ok = MakePaint(sdc, &PaintParams{
		Color:     colorcore.Color4f{R: 1, G: 0, B: 0, A: 1},
		BlendMode: raster.BlendHue,
	}, ctm)
	if !ok {
		t.Fatal("advanced-mode conversion failed")
	}
	if _, isCustom := gpuPaint.XPFactory().(*customXPFactory); !isCustom {
		t.Fatalf("advanced mode factory is %T, want customXPFactory", gpuPaint.XPFactory())
	}
}

// TestAlwaysDitherSurfaceProp checks that surface.AlwaysDitherFlag on the draw context's props forces the dither FP on
// for a paint whose own dither flag is off, and that a paint with no color FP still has nothing to wrap.
func TestAlwaysDitherSurfaceProp(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	makeSDC := func(props *surface.Props) *SurfaceDrawContext {
		sdc := MakeSurfaceDrawContextWithProps(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 8, Height: 8},
			gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "dither-props-test",
			props)
		if sdc == nil {
			t.Fatal("MakeSurfaceDrawContextWithProps failed")
		}
		t.Cleanup(sdc.Release)
		return sdc
	}
	ctm := geom.IdentityMatrix()
	shader := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 8},
		[]colorcore.Color{0xFF101010, 0xFF181818}, nil, shaders.TileClamp, nil)
	white := colorcore.Color4f{R: 1, G: 1, B: 1, A: 1}

	// Control: with default props, a non-dithering paint is not wrapped.
	gpuPaint, ok := MakePaint(makeSDC(nil), &PaintParams{
		Color:     white,
		BlendMode: raster.BlendSrcOver,
		Shader:    shader,
	}, ctm)
	if !ok {
		t.Fatal("default-props conversion failed")
	}
	if _, isDither := gpuPaint.ColorFragmentProcessor().(*ditherFP); isDither {
		t.Fatal("paint without the dither flag must not dither on a default-props surface")
	}

	// The same paint on an always-dither surface is wrapped in the dither FP.
	alwaysDither := makeSDC(&surface.Props{Flags: surface.AlwaysDitherFlag})
	gpuPaint, ok = MakePaint(alwaysDither, &PaintParams{
		Color:     white,
		BlendMode: raster.BlendSrcOver,
		Shader:    shader,
	}, ctm)
	if !ok {
		t.Fatal("always-dither conversion failed")
	}
	if _, isDither := gpuPaint.ColorFragmentProcessor().(*ditherFP); !isDither {
		t.Fatalf("always-dither surface FP is %T, want ditherFP", gpuPaint.ColorFragmentProcessor())
	}

	// A plain color paint produces no color FP at all, so there is nothing for the prop to wrap.
	gpuPaint, ok = MakePaint(alwaysDither, &PaintParams{Color: white, BlendMode: raster.BlendSrcOver}, ctm)
	if !ok || gpuPaint.HasColorFragmentProcessor() {
		t.Fatal("a color-only paint must not gain an FP from the always-dither prop")
	}
}

// TestFakeGradientDitherDrawGLSL runs a full dithered-gradient draw through the recording driver and checks the
// generated GLSL structurally: the looping colorizer's uniforms and loop, the dither table sampler + RT-flip lowering
// of gl_FragCoord, and the texture-effect sampler.
func TestFakeGradientDitherDrawGLSL(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 16, Height: 16},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes,
		"gradient-draw")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	shader := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 16},
		[]colorcore.Color{0xFF101010, 0xFF181818, 0xFF202020, 0xFF282828, 0xFF303030, 0xFF383838},
		nil, shaders.TileClamp, nil)
	ctm := geom.IdentityMatrix()
	gpuPaint, ok := MakePaint(sdc, &PaintParams{
		Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
		BlendMode: raster.BlendSrcOver,
		Shader:    shader,
		Dither:    true,
	}, ctm)
	if !ok {
		t.Fatal("conversion failed")
	}
	sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: 16, Bottom: 16})
	dc.FlushAndSubmit(false)

	var fragment string
	for _, src := range capturedShaderSource {
		if strings.Contains(src, "fsColorOut") {
			fragment = src
		}
	}
	if fragment == "" {
		t.Fatal("no fragment shader captured")
	}
	for _, wantPiece := range []string{
		"uthresholds", // looping colorizer uniform arrays (u-prefixed + mangled)
		"uscale",
		"ubias",
		"for (int loop = 0; loop < 1; ++loop)", // 8 intervals → 2 chunks → 1 loop iteration
		"urange",                               // dither range uniform
		// The dither table's identity matrix effect under a FragCoord sample lifts to a device-position varying
		// (collectTransforms' SampleUsageFragCoord lane), and the A8 read swizzle lowers through the constant-swizzle
		// constructor.
		"vTransformedCoords_",
		"sampler2D uTextureSampler_",                    // the dither LUT sampler
		"vec4(0.0, 0.0, 0.0, (texture(uTextureSampler_", // 000r swizzle lowering
		"clamp(color.rgb + value * ",                    // the dither clamp-to-alpha
	} {
		if !strings.Contains(fragment, wantPiece) {
			t.Fatalf("fragment shader missing %q:\n%s", wantPiece, fragment)
		}
	}
}

// TestFakeAdvancedBlendDstCopyDraw runs an overlay draw through the recording driver: without advanced-blend or
// texture-barrier support the dst is copied and the shader blends against it.
func TestFakeAdvancedBlendDstCopyDraw(t *testing.T) {
	dc := newShaderRecordingContext(t)
	defer dc.Destroy()
	if dc.GLCaps().TextureBarrierSupport || dc.GLCaps().AdvancedBlendEquationSupport() {
		t.Skip("test caps profile has barrier/advanced-blend support")
	}
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 16, Height: 16},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "dst-copy")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()
	sdc.Clear([4]float32{0.25, 0.5, 0.125, 0.5})

	ctm := geom.IdentityMatrix()
	gpuPaint, ok := MakePaint(sdc, &PaintParams{
		Color:     colorcore.Color4f{R: 0.4, G: 0.1, B: 0.6, A: 0.75},
		BlendMode: raster.BlendOverlay,
	}, ctm)
	if !ok {
		t.Fatal("conversion failed")
	}
	sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Left: 4, Top: 4, Right: 12, Bottom: 12})
	dc.FlushAndSubmit(false)

	var fragment string
	for _, src := range capturedShaderSource {
		if strings.Contains(src, "fsColorOut") {
			fragment = src
		}
	}
	if fragment == "" {
		t.Fatal("no fragment shader captured")
	}
	for _, wantPiece := range []string{
		"_dstColor",         // the dst-read global
		"DstTextureSampler", // the dst copy sampler
		"blend_overlay",     // the advanced blend body
		"_blend_overlay_component",
	} {
		if !strings.Contains(fragment, wantPiece) {
			t.Fatalf("fragment shader missing %q:\n%s", wantPiece, fragment)
		}
	}
	// The dst copy itself must have been recorded (copyTexSubImage or blit).
	if recCounts["glCopyTexSubImage2D"] == 0 && recCounts["glBlitFramebuffer"] == 0 {
		t.Fatalf("no dst copy recorded: %v", recCounts)
	}
}
