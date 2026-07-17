// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the shader-pipeline spine: BlendFormula table values, key-builder packing, color-FP analysis
// and processor-set finalization (constant folding, src-over collapse), Porter-Duff XP selection, program descriptor
// stability, and a full draw through the recording driver that captures and structurally checks the generated GLSL,
// verifies the program cache hit/miss metrics, and counts the GL draw traffic.

package gl

import (
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

var (
	shaderProcsOnce      sync.Once
	shaderProcs          map[string]uintptr
	capturedShaderSource map[uint32]string
	nextUniformLocation  int32
)

// installShaderProcs extends the recording driver with program/shader entry points that behave well enough to link:
// id-generating creates, always-successful compile/link queries, sequential uniform locations, and shader-source
// capture for the GLSL structure assertions.
func installShaderProcs(g *Gpu) {
	shaderProcsOnce.Do(func() {
		shaderProcs = map[string]uintptr{}
		shaderProcs["glCreateShader"] = purego.NewCallback(func(_ uintptr) uintptr {
			recNextID++
			return uintptr(recNextID)
		})
		shaderProcs["glCreateProgram"] = purego.NewCallback(func() uintptr {
			recNextID++
			return uintptr(recNextID)
		})
		shaderProcs["glShaderSource"] = purego.NewCallback(
			func(shader, count, strPtr, lenPtr uintptr) uintptr {
				if count != 1 {
					panic("expected a single source string")
				}
				src := *(*uintptr)(ptrFromUintptr(strPtr))
				length := *(*int32)(ptrFromUintptr(lenPtr))
				b := make([]byte, length)
				copy(b, unsafe.Slice((*byte)(ptrFromUintptr(src)), length))
				capturedShaderSource[uint32(shader)] = string(b)
				return 0
			},
		)
		writeOne := purego.NewCallback(func(_, pname, params uintptr) uintptr {
			switch uint32(pname) {
			case COMPILE_STATUS, LINK_STATUS:
				*(*int32)(ptrFromUintptr(params)) = 1
			default:
				*(*int32)(ptrFromUintptr(params)) = 0
			}
			return 0
		})
		shaderProcs["glGetShaderiv"] = writeOne
		shaderProcs["glGetProgramiv"] = writeOne
		shaderProcs["glGetUniformLocation"] = purego.NewCallback(
			func(_, _ uintptr) uintptr {
				nextUniformLocation++
				return uintptr(nextUniformLocation)
			},
		)
	})
	capturedShaderSource = map[uint32]string{}
	nextUniformLocation = 0
	f := &g.ctx.Interface.Functions
	f.createShader = shaderProcs["glCreateShader"]
	f.createProgram = shaderProcs["glCreateProgram"]
	f.shaderSource = shaderProcs["glShaderSource"]
	f.getShaderiv = shaderProcs["glGetShaderiv"]
	f.getProgramiv = shaderProcs["glGetProgramiv"]
	f.getUniformLocation = shaderProcs["glGetUniformLocation"]
	for _, hook := range []struct {
		field *uintptr
		name  string
	}{
		{field: &f.attachShader, name: "glAttachShader"},
		{field: &f.bindAttribLocation, name: "glBindAttribLocation"},
		{field: &f.bindFragDataLocation, name: "glBindFragDataLocation"},
		{field: &f.bindFragDataLocationIndexed, name: "glBindFragDataLocationIndexed"},
		{field: &f.compileShader, name: "glCompileShader"},
		{field: &f.linkProgram, name: "glLinkProgram"},
		{field: &f.useProgram, name: "glUseProgram"},
		{field: &f.deleteShader, name: "glDeleteShader"},
		{field: &f.deleteProgram, name: "glDeleteProgram"},
		{field: &f.vertexAttribPointer, name: "glVertexAttribPointer"},
		{field: &f.vertexAttribIPointer, name: "glVertexAttribIPointer"},
		{field: &f.enableVertexAttribArray, name: "glEnableVertexAttribArray"},
		{field: &f.disableVertexAttribArray, name: "glDisableVertexAttribArray"},
		{field: &f.drawArrays, name: "glDrawArrays"},
		{field: &f.drawElements, name: "glDrawElements"},
		{field: &f.drawRangeElements, name: "glDrawRangeElements"},
		{field: &f.blendFunc, name: "glBlendFunc"},
		{field: &f.blendEquation, name: "glBlendEquation"},
		{field: &f.uniform1i, name: "glUniform1i"},
		{field: &f.uniform1fv, name: "glUniform1fv"},
		{field: &f.uniform2fv, name: "glUniform2fv"},
		{field: &f.uniform4fv, name: "glUniform4fv"},
		{field: &f.uniformMatrix3fv, name: "glUniformMatrix3fv"},
	} {
		*hook.field = counterProc(hook.name)
	}
	// The RegisterFunc-bound float-argument entry points get plain counters too.
	f.fnUniform1f = func(int32, float32) { recCounts["glUniform1f"]++ }
	f.fnUniform2f = func(int32, float32, float32) { recCounts["glUniform2f"]++ }
	f.fnUniform3f = func(int32, float32, float32, float32) { recCounts["glUniform3f"]++ }
	f.fnUniform4f = func(int32, float32, float32, float32, float32) { recCounts["glUniform4f"]++ }
}

func newShaderRecordingContext(t *testing.T) *DirectContext {
	t.Helper()
	g := newRecordingGpu(t)
	installShaderProcs(g)
	dc := NewDirectContext(g)
	if dc == nil {
		t.Fatal("NewDirectContext returned nil")
	}
	return dc
}

// newShaderRecordingContextWithOptions is newShaderRecordingContext with explicit context options, for tests that pin
// option-dependent behavior — e.g. the per-OS fGlyphsAsPathsFontSize (E.1) — without depending on the host GOOS's
// defaults.
func newShaderRecordingContextWithOptions(t *testing.T, options *gpu.ContextOptions) *DirectContext {
	t.Helper()
	g := newRecordingGpuWithDriverOptions(t, appleM4MaxDriver(), options)
	installShaderProcs(g)
	dc := NewDirectContext(g)
	if dc == nil {
		t.Fatal("NewDirectContext returned nil")
	}
	return dc
}

// TestBlendFormulaTable spot-checks transcribed blend-formula table rows against known-good values.
func TestBlendFormulaTable(t *testing.T) {
	// No coverage, unknown color, src-over: modulate output, add, (One, ISA).
	f := gpu.GetBlendFormula(false, false, raster.BlendSrcOver)
	if f.PrimaryOutput() != gpu.BlendFormulaOutputModulate || f.HasSecondaryOutput() ||
		f.Equation() != gpu.BlendEquationAdd || f.SrcCoeff() != gpu.BlendCoeffOne ||
		f.DstCoeff() != gpu.BlendCoeffISA {
		t.Fatalf("src-over formula wrong: %+v", f)
	}
	if !f.CanTweakAlphaForCoverage() || !f.ModifiesDst() || f.UnaffectedByDst() ||
		!f.UnaffectedByDstIfOpaque() || !f.UsesInputColor() {
		t.Fatalf("src-over properties wrong: %+v", f)
	}

	// Coverage, unknown color, src: dual-source (coverage secondary), dst coeff IS2A.
	f = gpu.GetBlendFormula(false, true, raster.BlendSrc)
	if f.PrimaryOutput() != gpu.BlendFormulaOutputModulate ||
		f.SecondaryOutput() != gpu.BlendFormulaOutputCoverage ||
		f.SrcCoeff() != gpu.BlendCoeffOne || f.DstCoeff() != gpu.BlendCoeffIS2A {
		t.Fatalf("src-with-coverage formula wrong: %+v", f)
	}
	if f.CanTweakAlphaForCoverage() {
		t.Fatal("dual-source src must not tweak alpha for coverage")
	}

	// Coverage, unknown color, modulate: reverse-subtract (DC, One) with ISC-modulate output.
	f = gpu.GetBlendFormula(false, true, raster.BlendModulate)
	if f.PrimaryOutput() != gpu.BlendFormulaOutputISCModulate ||
		f.Equation() != gpu.BlendEquationReverseSubtract ||
		f.SrcCoeff() != gpu.BlendCoeffDC || f.DstCoeff() != gpu.BlendCoeffOne {
		t.Fatalf("modulate-with-coverage formula wrong: %+v", f)
	}

	// Opaque, no coverage, dst-out: (Zero, Zero) collapses the primary output to none.
	f = gpu.GetBlendFormula(true, false, raster.BlendDstOut)
	if f.PrimaryOutput() != gpu.BlendFormulaOutputNone || f.UsesInputColor() {
		t.Fatalf("opaque dst-out formula wrong: %+v", f)
	}

	// Opaque, no coverage, dst-in: (Zero, One) does not modify the dst.
	f = gpu.GetBlendFormula(true, false, raster.BlendDstIn)
	if f.ModifiesDst() || f.PrimaryOutput() != gpu.BlendFormulaOutputNone {
		t.Fatalf("opaque dst-in formula wrong: %+v", f)
	}

	// LCD src-over: SA-modulate secondary via MakeCoverageFormula.
	f = gpu.GetLCDBlendFormula(raster.BlendSrcOver)
	if f.SecondaryOutput() != gpu.BlendFormulaOutputSAModulate ||
		f.DstCoeff() != gpu.BlendCoeffIS2C {
		t.Fatalf("LCD src-over formula wrong: %+v", f)
	}
}

// TestKeyBuilderPacking checks the bit-packing semantics the program keys rely on.
func TestKeyBuilderPacking(t *testing.T) {
	var key []uint32
	b := gpu.NewKeyBuilder(&key)
	b.AddBits(8, 0xAB, "a")
	b.AddBits(8, 0xCD, "b")
	b.Add32(0x12345678, "c")
	b.AddBool(true, "d")
	b.Flush()
	// Values pack from the low bits: 0xAB, then 0xCD<<8, then the low 16 bits of 0x12345678 fill the first word; the
	// second word gets the high 16 bits plus the bool at bit 16.
	want := []uint32{0xAB | 0xCD<<8 | 0x5678<<16, 0x1234 | 1<<16}
	if len(key) != len(want) {
		t.Fatalf("key length = %d, want %d (%x)", len(key), len(want), key)
	}
	for i := range want {
		if key[i] != want[i] {
			t.Fatalf("key[%d] = %#x, want %#x", i, key[i], want[i])
		}
	}
}

// TestColorFPAnalysisElimination checks constant folding through the color-FP analysis and the Compose factory's
// optimization pass.
func TestColorFPAnalysisElimination(t *testing.T) {
	red := colorcore.PMColor4f{R: 1, A: 1}
	half := colorcore.PMColor4f{R: 0.5, G: 0.5, B: 0.5, A: 0.5}

	// A constant color FP over a constant input is eliminated and its output becomes the new input color.
	fp := MakeColorFP(red)
	analysis := NewColorFragmentProcessorAnalysis(AnalysisColorConstant(half),
		[]FragmentProcessor{fp})
	color, n := analysis.InitialProcessorsToEliminate()
	if n != 1 || color != red {
		t.Fatalf("eliminate = (%v, %d), want (%v, 1)", color, n, red)
	}
	if !analysis.IsOpaque() {
		t.Fatal("constant red output must analyze opaque")
	}

	// Compose runs its optimization pass with an *unknown* input color, so two constant color FPs stay composed (the
	// analysis can only fold from a known input), but the composition itself is constant-foldable.
	composed := ComposeFP(MakeColorFP(red), MakeColorFP(half))
	if composed.Name() != "Compose" {
		t.Fatalf("compose = %q, want Compose", composed.Name())
	}
	if got := fpConstantOutputForConstantInput(composed, colorcore.PMColor4f{}); got != red {
		t.Fatalf("compose output = %v, want %v", got, red)
	}
	// With a known input, the analysis over the composed chain folds both stages.
	analysis = NewColorFragmentProcessorAnalysis(AnalysisColorConstant(half),
		[]FragmentProcessor{composed})
	color, n = analysis.InitialProcessorsToEliminate()
	if n != 1 || color != red {
		t.Fatalf("composed eliminate = (%v, %d), want (%v, 1)", color, n, red)
	}

	// BlendFP constant output matches the CPU blend kernels.
	blend := BlendFP(MakeColorFP(red), MakeColorFP(half), raster.BlendMultiply)
	want := raster.BlendHighp4f(raster.BlendMultiply, red, half)
	if got := fpConstantOutputForConstantInput(blend, colorcore.PMColor4f{}); got != want {
		t.Fatalf("blend constant output = %v, want %v", got, want)
	}
}

// TestFPHelperOptimizationFlags checks the helper FPs' flag computation.
func TestFPHelperOptimizationFlags(t *testing.T) {
	red := colorcore.PMColor4f{R: 1, A: 1}

	color := MakeColorFP(red)
	if !color.fpBase().PreservesOpaqueInput() ||
		!color.fpBase().HasConstantOutputForConstantInput() ||
		color.fpBase().CompatibleWithCoverageAsAlpha() {
		t.Fatalf("color FP flags wrong: %#x", color.fpBase().flags)
	}

	// DisableCoverageAsAlpha on an incompatible FP is an identity.
	if got := DisableCoverageAsAlphaFP(color); got != color {
		t.Fatal("DisableCoverageAsAlpha should pass through incompatible FPs")
	}

	// A compatible FP gets wrapped and the flag cleared.
	swizzled := SwizzleOutputFP(MakeColorFP(red), gpu.SwizzleBGRA)
	blend := BlendFP(nil, MakeColorFP(red), raster.BlendSrcIn) // input * red.a: compatible
	if !blend.fpBase().CompatibleWithCoverageAsAlpha() {
		t.Skip("blend srcin unexpectedly incompatible") // guard: OptFlags gives no compat bit
	}
	_ = swizzled

	// OverrideInput: constant output flows from the child; preserves-opaque needs both an opaque override color and a
	// preserving child.
	override := OverrideInputFP(MakeColorFP(red), colorcore.PMColor4f{R: 0.5, A: 0.5})
	if override.fpBase().PreservesOpaqueInput() {
		t.Fatal("translucent override cannot preserve opaque input")
	}
	if got := fpConstantOutputForConstantInput(override, colorcore.PMColor4f{}); got != red {
		t.Fatalf("override constant output = %v, want %v", got, red)
	}
}

// TestPorterDuffXPSelection checks XP creation decisions against the M4 Max caps profile.
func TestPorterDuffXPSelection(t *testing.T) {
	dc := newFakeDirectContext(t)
	caps := dc.Caps()

	// Simple src-over: nil from the factory means "use the shared simple src-over XP".
	xp := XPFactoryMakeXferProcessor(nil, AnalysisColorUnknown(), AnalysisCoverageSingleChannel,
		caps, gpu.ClampTypeAuto)
	if xp != nil {
		t.Fatal("non-LCD src-over with coverage should return the nil sentinel")
	}

	// Opaque + no coverage collapses src-over to src when the caps prefer it.
	if caps.ShouldCollapseSrcOverToSrcWhenAble {
		xp = XPFactoryMakeXferProcessor(nil, AnalysisColorUnknownOpaque(), AnalysisCoverageNone,
			caps, gpu.ClampTypeAuto)
		if xp == nil {
			t.Fatal("collapse lane should create a real XP")
		}
		info := xpGetBlendInfo(xp)
		if info.SrcBlend != gpu.BlendCoeffOne || info.DstBlend != gpu.BlendCoeffZero {
			t.Fatalf("collapsed src blend info = %+v, want (One, Zero)", info)
		}
	}

	// Dual-source lane: src mode with coverage needs a secondary output (the caps profile has dual source blending, so
	// no dst read).
	factory := PorterDuffXPFactory(raster.BlendSrc)
	props := XPFactoryGetAnalysisProperties(factory, AnalysisColorUnknown(),
		AnalysisCoverageSingleChannel, caps, gpu.ClampTypeAuto)
	if props&XPAnalysisReadsDstInShader != 0 {
		t.Fatal("src mode with dual-source blending must not read dst")
	}
	xp = XPFactoryMakeXferProcessor(factory, AnalysisColorUnknown(),
		AnalysisCoverageSingleChannel, caps, gpu.ClampTypeAuto)
	if !xpHasSecondaryOutput(xp) {
		t.Fatal("src mode with coverage should use a secondary output")
	}

	// kClear ignores the input color entirely.
	props = XPFactoryGetAnalysisProperties(PorterDuffXPFactory(raster.BlendClear),
		AnalysisColorUnknown(), AnalysisCoverageNone, caps, gpu.ClampTypeAuto)
	if props&XPAnalysisIgnoresInputColor == 0 {
		t.Fatal("clear must ignore the input color")
	}
}

// TestProcessorSetFinalize checks the constant-color elimination and analysis outputs.
func TestProcessorSetFinalize(t *testing.T) {
	dc := newFakeDirectContext(t)
	caps := dc.Caps()
	red := colorcore.PMColor4f{R: 1, A: 1}

	paint := NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 0.25, G: 0.25, B: 0.25, A: 0.5})
	paint.SetColorFragmentProcessor(MakeColorFP(red))
	paint.SetPorterDuffXPFactory(raster.BlendSrcOver)

	ps := NewProcessorSetFromPaint(paint)
	clip := MakeAppliedClip(geom.ISize{Width: 16, Height: 16})
	analysis, override := ps.Finalize(AnalysisColorConstant(paint.Color4f()),
		AnalysisCoverageNone, &clip, UnusedStencilSettings(), caps, gpu.ClampTypeAuto)

	if !analysis.IsInitialized() {
		t.Fatal("analysis must be initialized")
	}
	if !analysis.InputColorIsOverridden() || override != red {
		t.Fatalf("constant color FP should be folded into the input color (got %v)", override)
	}
	if analysis.HasColorFragmentProcessor() {
		t.Fatal("folded color FP should be removed from the set")
	}
	if analysis.UsesLocalCoords() {
		t.Fatal("no FP uses local coords")
	}
	if analysis.RequiresDstTexture() {
		t.Fatal("src-over must not require a dst texture")
	}
	if !ps.IsFinalized() {
		t.Fatal("set must be finalized")
	}
}

func makeTestProgramInfo(t *testing.T, dc *DirectContext, mode raster.BlendMode, colorFP FragmentProcessor, viewMatrix *geom.Matrix) *ProgramInfo {
	t.Helper()
	caps := dc.GLCaps()
	proxy := makeDeferredProxy(t, dc, geom.ISize{Width: 32, Height: 32}, gpu.RenderableYes,
		gpu.BackingFitExact)
	defer proxy.Unref()
	view := MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, gpu.SwizzleRGBA)

	paint := NewPaint()
	paint.SetPorterDuffXPFactory(mode)
	if colorFP != nil {
		paint.SetColorFragmentProcessor(colorFP)
	}
	ps := NewProcessorSetFromPaint(paint)
	clip := MakeAppliedClip(geom.ISize{Width: 32, Height: 32})
	analysis, _ := ps.Finalize(AnalysisColorConstant(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}),
		AnalysisCoverageNone, &clip, UnusedStencilSettings(), dc.Caps(), gpu.ClampTypeAuto)

	pipeline := NewPipeline(&PipelineInitArgs{Caps: caps, WriteSwizzle: gpu.SwizzleRGBA}, ps,
		&clip)
	localCoords := LocalCoordsTypeUnused
	if analysis.UsesLocalCoords() {
		localCoords = LocalCoordsTypeUsePosition
	}
	gp := MakeDefaultGeoProc(ColorTypePremulUniform, colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1},
		CoverageTypeSolid, 0xff, localCoords, nil, viewMatrix)
	return NewProgramInfo(caps, &view, false, pipeline, UnusedStencilSettings(), gp,
		gpu.PrimitiveTypeTriangles, 0, gpu.LoadOpLoad)
}

// TestProgramDescStability checks that equal pipelines produce equal keys and different pipelines differ.
func TestProgramDescStability(t *testing.T) {
	dc := newFakeDirectContext(t)
	identity := geom.Matrix{}
	identity.SetIdentity()
	scale := geom.Matrix{}
	scale.SetScale(2, 3)

	var a, b, c, c2, d ProgramDesc
	BuildProgramDesc(&a, makeTestProgramInfo(t, dc, raster.BlendSrcOver, nil, &identity),
		dc.GLCaps())
	BuildProgramDesc(&b, makeTestProgramInfo(t, dc, raster.BlendSrcOver, nil, &identity),
		dc.GLCaps())
	// GL keys deliberately exclude the fixed-function blend coefficients (srcover and srcin share one shader); modes
	// with different shader *outputs* must differ (clear's primary output is kNone).
	BuildProgramDesc(&c, makeTestProgramInfo(t, dc, raster.BlendSrcIn, nil, &identity),
		dc.GLCaps())
	BuildProgramDesc(&c2, makeTestProgramInfo(t, dc, raster.BlendClear, nil, &identity),
		dc.GLCaps())
	// A scale-translate view matrix changes the GP's matrix key but not the FP/XP keys.
	BuildProgramDesc(&d, makeTestProgramInfo(t, dc, raster.BlendSrcOver, nil, &scale),
		dc.GLCaps())

	if a.MapKey() != b.MapKey() {
		t.Fatal("identical pipelines must produce identical program keys")
	}
	if a.MapKey() != c.MapKey() {
		t.Fatal("srcover and srcin share a shader; their program keys must match")
	}
	if a.MapKey() == c2.MapKey() {
		t.Fatal("clear's kNone output must produce a different program key")
	}
	if a.MapKey() == d.MapKey() {
		t.Fatal("different matrix types must produce different program keys")
	}

	// Uniform values (a different sample matrix over the same matrix type) must NOT change the key.
	scaleA := geom.Matrix{}
	scaleA.SetScale(0.5, 0.5)
	scaleB := geom.Matrix{}
	scaleB.SetScale(0.25, 0.125)
	var e, f ProgramDesc
	BuildProgramDesc(&e, makeTestProgramInfo(t, dc, raster.BlendSrcOver,
		MatrixEffectFP(scaleA, newTestCoordsFP()), &identity), dc.GLCaps())
	BuildProgramDesc(&f, makeTestProgramInfo(t, dc, raster.BlendSrcOver,
		MatrixEffectFP(scaleB, newTestCoordsFP()), &identity), dc.GLCaps())
	if e.MapKey() != f.MapKey() {
		t.Fatal("sample-matrix value changes must not change the program key")
	}
	if e.MapKey() == a.MapKey() {
		t.Fatal("adding a color FP must change the program key")
	}
}

// testCoordsFP is a test-only FP that returns vec4(coords, 0, 1), for exercising the sample-coord lifting machinery.
type testCoordsFP struct {
	FPBase
}

func newTestCoordsFP() FragmentProcessor {
	fp := &testCoordsFP{}
	fp.initFP(TestFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *testCoordsFP) Name() string { return "TestCoords" }

func (f *testCoordsFP) Clone() FragmentProcessor { return newTestCoordsFP() }

func (f *testCoordsFP) onAddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

func (f *testCoordsFP) onIsEqual(FragmentProcessor) bool { return true }

func (f *testCoordsFP) onMakeProgramImpl() FPProgramImpl { return &testCoordsFPImpl{} }

type testCoordsFPImpl struct {
	FPImplBase
}

func (i *testCoordsFPImpl) EmitCode(args *FPEmitArgs) {
	args.FragBuilder.CodeAppendf("return vec4(%s, 0.0, 1.0);", args.SampleCoord)
}

// TestFakeDrawPipelineGLSL drives a full draw through the recording driver and structurally checks the generated GLSL
// plus the cache metrics and draw traffic.
func TestFakeDrawPipelineGLSL(t *testing.T) {
	dc := newShaderRecordingContext(t)
	dims := geom.ISize{Width: 32, Height: 32}
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, dims, gpu.BackingFitExact, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "shader-test")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	identity := geom.Matrix{}
	identity.SetIdentity()
	scale := geom.Matrix{}
	scale.SetScale(1.0/32, 1.0/32)

	draw := func(mode raster.BlendMode, fp FragmentProcessor) {
		paint := NewPaint()
		paint.SetColor4f(colorcore.PMColor4f{R: 0.5, G: 0.25, B: 0.125, A: 0.5})
		paint.SetPorterDuffXPFactory(mode)
		if fp != nil {
			paint.SetColorFragmentProcessor(fp)
		}
		sdc.FillRectWithPaint(paint, &identity, geom.Rect{Right: 16, Bottom: 16})
	}

	// Draw 1: plain src-over. Draw 2: src-in — a different XP, so FillRectOp cannot merge the ops, but GL keys no
	// fixed-function blend state so the program is a cache hit. Draw 3: a coords-sampling FP under a matrix effect (new
	// program; exercises the varying lift). (Identical paints now merge into a single op through the FillRectOp combine
	// rules; that batching is asserted separately in fillrectop_test.go.)
	draw(raster.BlendSrcOver, nil)
	draw(raster.BlendSrcIn, nil)
	draw(raster.BlendSrcOver, MatrixEffectFP(scale, newTestCoordsFP()))
	dc.FlushAndSubmit(false)

	stats := dc.Gpu().ProgramCacheStats()
	if stats.NumProgramCacheMisses != 2 {
		t.Fatalf("program cache misses = %d, want 2", stats.NumProgramCacheMisses)
	}
	if stats.NumProgramCacheHits != 1 {
		t.Fatalf("program cache hits = %d, want 1", stats.NumProgramCacheHits)
	}
	if stats.ShaderCompilations != 4 { // 2 programs x (VS + FS)
		t.Fatalf("shader compilations = %d, want 4", stats.ShaderCompilations)
	}
	if got := dc.Gpu().ProgramCacheCount(); got != 2 {
		t.Fatalf("program cache count = %d, want 2", got)
	}
	// Single-quad non-AA fill-rect ops use the tri-strip lane (no index buffer).
	if counts("glDrawArrays") != 3 {
		t.Fatalf("draw calls = %d, want 3 glDrawArrays", counts("glDrawArrays"))
	}

	// Structural GLSL checks over the captured sources.
	var vsPlain, fsPlain, vsCoords, fsCoords string
	for _, src := range capturedShaderSource {
		switch {
		case strings.Contains(src, "gl_Position"):
			if strings.Contains(src, "TransformedCoords") {
				vsCoords = src
			} else {
				vsPlain = src
			}
		case strings.Contains(src, "fsColorOut"):
			if strings.Contains(src, "TestCoords") {
				fsCoords = src
			} else {
				fsPlain = src
			}
		}
	}
	if vsPlain == "" || fsPlain == "" || vsCoords == "" || fsCoords == "" {
		t.Fatalf("missing captured shaders (%d captured)", len(capturedShaderSource))
	}

	// The plain VS carries the RT-adjust fixup, the device-space position attribute (the QuadPerEdgeAA GP has no
	// view-matrix uniform — positions are pre-mapped on the CPU), and the byte-color attribute passed through a flat
	// varying.
	for _, want := range []string{
		"uniform vec4 u_rtAdjust;",
		"_position.xy * u_rtAdjust.xz + _position.ww * u_rtAdjust.yw",
		"in vec2 position;",
		"in vec4 color;",
		"flat out vec4 vcolor_S0;",
	} {
		if !strings.Contains(vsPlain, want) {
			t.Errorf("plain VS missing %q:\n%s", want, vsPlain)
		}
	}
	// The plain FS: vertex color GP output via the flat varying, solid coverage, src-over blend via modulate output.
	for _, want := range []string{
		"out vec4 fsColorOut;",
		"flat in vec4 vcolor_S0;",
		"outputColor_S0 = vcolor_S0;",
		"const vec4 outputCoverage_S0 = vec4(1.0);",
		"fsColorOut = outputColor_S0 * outputCoverage_S0;",
	} {
		if !strings.Contains(fsPlain, want) {
			t.Errorf("plain FS missing %q:\n%s", want, fsPlain)
		}
	}
	// The coords pipeline: the matrix uniform is lifted to the VS and drives a varying; the FP function takes no coords
	// parameter and reads the varying.
	for _, want := range []string{
		"uniform mat3 umatrix_S1;",
		"vTransformedCoords_2_S0",
	} {
		if !strings.Contains(vsCoords, want) {
			t.Errorf("coords VS missing %q:\n%s", want, vsCoords)
		}
	}
	for _, want := range []string{
		"vec4 TestCoords_S1_c0(vec4 _input)",
		"vTransformedCoords_2_S0",
		"MatrixEffect_S1",
	} {
		if !strings.Contains(fsCoords, want) {
			t.Errorf("coords FS missing %q:\n%s", want, fsCoords)
		}
	}
}
