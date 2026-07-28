// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Gradient shader assembly over the shader descriptors: builds a fragment processor for each gradient family (linear,
// radial, sweep, conical) by combining a layout FP (maps a fragment to a scalar t along the gradient) with a colorizer
// FP (maps t to a color) and a tile-mode top-level effect (clamp/repeat/mirror/decal). Each runtime effect is
// hand-written as a GLSL-emitting FP with its own uniforms, specializations, children, and optimization flags. Trims:
// the unrolled-binary colorizer is unreachable (desktop GLSL always has non-constant array indexing) and the
// low-precision interval limit never trips (FloatIs32Bits is always true on the desktop trim). Gradients past the
// looping colorizer's 128-color limit fall back to the textured colorizer — the 256×1 LUT texture baked by
// gradientbitmapcache.go — and the draw is skipped only when the LUT texture cannot be created.

package gl

import (
	"math/bits"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/shaders"
)

// Colorizer size limits.
const (
	maxLoopingColorCount    = 128
	maxLoopingIntervalCount = maxLoopingColorCount / 2
)

//////////////////////////////////////////////////////////////////////////////
// Layout FPs. Each produces t in the x channel and a validity flag in y (negative y rejects the fragment in the
// top-level clamped/tiled effect).

// linearLayoutFP is the linear-gradient layout effect: maps a fragment's local x coordinate directly to t.
type linearLayoutFP struct{ FPBase }

func makeLinearLayoutFP() FragmentProcessor {
	fp := &linearLayoutFP{}
	// The linear gradient never rejects a pixel so it doesn't change opacity.
	fp.initFP(GLSLFPClassID, FPPreservesOpaqueInput)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *linearLayoutFP) Name() string                     { return "LinearLayout" }
func (f *linearLayoutFP) Clone() FragmentProcessor         { return makeLinearLayoutFP() }
func (f *linearLayoutFP) onIsEqual(FragmentProcessor) bool { return true }

func (f *linearLayoutFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPLinearLayout), "runtimeFPKind")
}

func (f *linearLayoutFP) onMakeProgramImpl() FPProgramImpl { return &linearLayoutImpl{} }

type linearLayoutImpl struct{ FPImplBase }

func (i *linearLayoutImpl) EmitCode(args *FPEmitArgs) {
	// A tiny delta is added to t so a hard stop at a pixel-center column resolves consistently to the color on its
	// "right" (crbug.com/938592).
	args.FragBuilder.CodeAppendf("return vec4(%s.x + 0.00001, 1.0, 0.0, 0.0);", args.SampleCoord)
}

// radialLayoutFP is the radial-gradient layout effect: maps a fragment's distance from the origin to t.
type radialLayoutFP struct{ FPBase }

func makeRadialLayoutFP() FragmentProcessor {
	fp := &radialLayoutFP{}
	// The radial gradient never rejects a pixel so it doesn't change opacity.
	fp.initFP(GLSLFPClassID, FPPreservesOpaqueInput)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *radialLayoutFP) Name() string                     { return "RadialLayout" }
func (f *radialLayoutFP) Clone() FragmentProcessor         { return makeRadialLayoutFP() }
func (f *radialLayoutFP) onIsEqual(FragmentProcessor) bool { return true }

func (f *radialLayoutFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPRadialLayout), "runtimeFPKind")
}

func (f *radialLayoutFP) onMakeProgramImpl() FPProgramImpl { return &radialLayoutImpl{} }

type radialLayoutImpl struct{ FPImplBase }

func (i *radialLayoutImpl) EmitCode(args *FPEmitArgs) {
	args.FragBuilder.CodeAppendf("return vec4(length(%s), 1.0, 0.0, 0.0);", args.SampleCoord)
}

// sweepLayoutFP is the sweep-gradient layout effect: maps a fragment's angle around the origin to t. A mobile-driver
// atan2 workaround is outside the desktop trim, so the non-workaround branch is always emitted (keyed as constant
// false).
type sweepLayoutFP struct {
	FPBase
	bias, scale float32
}

func makeSweepLayoutFP(bias, scale float32) FragmentProcessor {
	fp := &sweepLayoutFP{bias: bias, scale: scale}
	// The sweep gradient never rejects a pixel so it doesn't change opacity.
	fp.initFP(GLSLFPClassID, FPPreservesOpaqueInput)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *sweepLayoutFP) Name() string             { return "SweepLayout" }
func (f *sweepLayoutFP) Clone() FragmentProcessor { return makeSweepLayoutFP(f.bias, f.scale) }

func (f *sweepLayoutFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*sweepLayoutFP)
	return ok && f.bias == o.bias && f.scale == o.scale
}

func (f *sweepLayoutFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPSweepLayout), "runtimeFPKind")
	b.AddBool(false, "useAtanWorkaround")
}

func (f *sweepLayoutFP) onMakeProgramImpl() FPProgramImpl { return &sweepLayoutImpl{} }

type sweepLayoutImpl struct {
	FPImplBase
	biasUni, scaleUni UniformHandle
}

func (i *sweepLayoutImpl) EmitCode(args *FPEmitArgs) {
	var biasName, scaleName string
	i.biasUni, biasName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf,
		"bias")
	i.scaleUni, scaleName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "scale")
	fb := args.FragBuilder
	// Hardcode pi/2 for the angle when x == 0, to avoid undefined behavior in that case.
	fb.CodeAppendf("float angle = (%[1]s.x != 0.0) ? atan(-%[1]s.y, -%[1]s.x)"+
		" : sign(%[1]s.y) * -1.5707963267949;", args.SampleCoord)
	// 0.1591549430918 is 1/(2*pi), used since atan returns values [-pi, pi].
	fb.CodeAppendf("float t = (angle * 0.1591549430918 + 0.5 + %s) * %s;", biasName, scaleName)
	fb.CodeAppend("return vec4(t, 1.0, 0.0, 0.0);")
}

func (i *sweepLayoutImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*sweepLayoutFP)
	pdman.Set1f(i.biasUni, f.bias)
	pdman.Set1f(i.scaleUni, f.scale)
}

// conicalStripLayoutFP is the two-point-conical-gradient layout effect for the strip case (equal radii).
type conicalStripLayoutFP struct {
	FPBase
	r02 float32 // r0 squared
}

func makeConicalStripLayoutFP(r02 float32) FragmentProcessor {
	fp := &conicalStripLayoutFP{r02: r02}
	// The 2-point conical gradient can reject a pixel, so it changes opacity: no opt flags.
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *conicalStripLayoutFP) Name() string             { return "TwoPointConicalStripLayout" }
func (f *conicalStripLayoutFP) Clone() FragmentProcessor { return makeConicalStripLayoutFP(f.r02) }

func (f *conicalStripLayoutFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*conicalStripLayoutFP)
	return ok && f.r02 == o.r02
}

func (f *conicalStripLayoutFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPConicalStripLayout), "runtimeFPKind")
}

func (f *conicalStripLayoutFP) onMakeProgramImpl() FPProgramImpl { return &conicalStripImpl{} }

type conicalStripImpl struct {
	FPImplBase
	r02Uni UniformHandle
}

func (i *conicalStripImpl) EmitCode(args *FPEmitArgs) {
	var r02Name string
	i.r02Uni, r02Name = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf,
		"r0_2")
	fb := args.FragBuilder
	fb.CodeAppend("float v = 1.0;") // validation flag, negative means discard later
	fb.CodeAppendf("float t = %s - %s.y * %s.y;", r02Name, args.SampleCoord, args.SampleCoord)
	fb.CodeAppendf("if (t >= 0.0) { t = %s.x + sqrt(t); } else { v = -1.0; }", args.SampleCoord)
	fb.CodeAppend("return vec4(t, v, 0.0, 0.0);")
}

func (i *conicalStripImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	pdman.Set1f(i.r02Uni, fp.(*conicalStripLayoutFP).r02)
}

// conicalRadialLayoutFP is the two-point-conical-gradient layout effect for the concentric-circles case.
type conicalRadialLayoutFP struct {
	FPBase
	r0          float32
	lengthScale float32
}

func makeConicalRadialLayoutFP(r0, lengthScale float32) FragmentProcessor {
	fp := &conicalRadialLayoutFP{r0: r0, lengthScale: lengthScale}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *conicalRadialLayoutFP) Name() string { return "TwoPointConicalRadialLayout" }

func (f *conicalRadialLayoutFP) Clone() FragmentProcessor {
	return makeConicalRadialLayoutFP(f.r0, f.lengthScale)
}

func (f *conicalRadialLayoutFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*conicalRadialLayoutFP)
	return ok && f.r0 == o.r0 && f.lengthScale == o.lengthScale
}

func (f *conicalRadialLayoutFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPConicalRadialLayout), "runtimeFPKind")
}

func (f *conicalRadialLayoutFP) onMakeProgramImpl() FPProgramImpl { return &conicalRadialImpl{} }

type conicalRadialImpl struct {
	FPImplBase
	r0Uni, lengthScaleUni UniformHandle
}

func (i *conicalRadialImpl) EmitCode(args *FPEmitArgs) {
	var r0Name, lengthScaleName string
	i.r0Uni, r0Name = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf,
		"r0")
	i.lengthScaleUni, lengthScaleName = args.UniformHandler.AddUniform(args.FP,
		ShaderFlagFragment, GLSLTypeHalf, "lengthScale")
	fb := args.FragBuilder
	fb.CodeAppend("float v = 1.0;") // validation flag, negative means discard later
	fb.CodeAppendf("float t = length(%s) * %s - %s;", args.SampleCoord, lengthScaleName, r0Name)
	fb.CodeAppend("return vec4(t, v, 0.0, 0.0);")
}

func (i *conicalRadialImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*conicalRadialLayoutFP)
	pdman.Set1f(i.r0Uni, f.r0)
	pdman.Set1f(i.lengthScaleUni, f.lengthScale)
}

// conicalFocalLayoutFP is the two-point-conical-gradient layout effect for the general focal-point case. The five
// booleans are specialized: baked into the shader text and the key rather than passed as uniforms.
type conicalFocalLayoutFP struct {
	FPBase
	isRadiusIncreasing bool
	isFocalOnCircle    bool
	isWellBehaved      bool
	isSwapped          bool
	isNativelyFocal    bool
	invR1              float32
	fx                 float32
}

func makeConicalFocalLayoutFP(isRadiusIncreasing, isFocalOnCircle, isWellBehaved, isSwapped, isNativelyFocal bool, invR1, fx float32) FragmentProcessor {
	fp := &conicalFocalLayoutFP{
		isRadiusIncreasing: isRadiusIncreasing,
		isFocalOnCircle:    isFocalOnCircle,
		isWellBehaved:      isWellBehaved,
		isSwapped:          isSwapped,
		isNativelyFocal:    isNativelyFocal,
		invR1:              invR1,
		fx:                 fx,
	}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *conicalFocalLayoutFP) Name() string { return "TwoPointConicalFocalLayout" }

func (f *conicalFocalLayoutFP) Clone() FragmentProcessor {
	return makeConicalFocalLayoutFP(f.isRadiusIncreasing, f.isFocalOnCircle, f.isWellBehaved,
		f.isSwapped, f.isNativelyFocal, f.invR1, f.fx)
}

func (f *conicalFocalLayoutFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*conicalFocalLayoutFP)
	return ok && f.isRadiusIncreasing == o.isRadiusIncreasing &&
		f.isFocalOnCircle == o.isFocalOnCircle && f.isWellBehaved == o.isWellBehaved &&
		f.isSwapped == o.isSwapped && f.isNativelyFocal == o.isNativelyFocal &&
		f.invR1 == o.invR1 && f.fx == o.fx
}

func (f *conicalFocalLayoutFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPConicalFocalLayout), "runtimeFPKind")
	b.AddBool(f.isRadiusIncreasing, "isRadiusIncreasing")
	b.AddBool(f.isFocalOnCircle, "isFocalOnCircle")
	b.AddBool(f.isWellBehaved, "isWellBehaved")
	b.AddBool(f.isSwapped, "isSwapped")
	b.AddBool(f.isNativelyFocal, "isNativelyFocal")
}

func (f *conicalFocalLayoutFP) onMakeProgramImpl() FPProgramImpl { return &conicalFocalImpl{} }

type conicalFocalImpl struct {
	FPImplBase
	invR1Uni, fxUni UniformHandle
}

func (i *conicalFocalImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*conicalFocalLayoutFP)
	var invR1Name, fxName string
	i.invR1Uni, invR1Name = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "invR1")
	i.fxUni, fxName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf,
		"fx")
	fb := args.FragBuilder
	coord := args.SampleCoord

	fb.CodeAppend("float t = -1.0;")
	fb.CodeAppend("float v = 1.0;") // validation flag, negative means discard later
	fb.CodeAppend("float x_t = -1.0;")
	switch {
	case fp.isFocalOnCircle:
		fb.CodeAppendf("x_t = dot(%s, %s) / %s.x;", coord, coord, coord)
	case fp.isWellBehaved:
		fb.CodeAppendf("x_t = length(%s) - %s.x * %s;", coord, coord, invR1Name)
	default:
		fb.CodeAppendf("float temp = %s.x * %s.x - %s.y * %s.y;", coord, coord, coord, coord)
		// Only sqrt if temp >= 0 (GPU may break on sqrt of a negative float).
		fb.CodeAppend("if (temp >= 0.0) {")
		if fp.isSwapped || !fp.isRadiusIncreasing {
			fb.CodeAppendf("x_t = -sqrt(temp) - %s.x * %s;", coord, invR1Name)
		} else {
			fb.CodeAppendf("x_t = sqrt(temp) - %s.x * %s;", coord, invR1Name)
		}
		fb.CodeAppend("}")
	}
	// The final calculation of t from x_t has lots of static optimizations, but only do them when x_t is positive
	// (which can be assumed if isWellBehaved).
	if !fp.isWellBehaved {
		// t is still calculated even though it will be ignored later, to avoid a branch.
		fb.CodeAppend("if (x_t <= 0.0) { v = -1.0; }")
	}
	if fp.isRadiusIncreasing {
		if fp.isNativelyFocal {
			fb.CodeAppend("t = x_t;")
		} else {
			fb.CodeAppendf("t = x_t + %s;", fxName)
		}
	} else {
		if fp.isNativelyFocal {
			fb.CodeAppend("t = -x_t;")
		} else {
			fb.CodeAppendf("t = -x_t + %s;", fxName)
		}
	}
	if fp.isSwapped {
		fb.CodeAppend("t = 1.0 - t;")
	}
	fb.CodeAppend("return vec4(t, v, 0.0, 0.0);")
}

func (i *conicalFocalImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*conicalFocalLayoutFP)
	pdman.Set1f(i.invR1Uni, f.invR1)
	pdman.Set1f(i.fxUni, f.fx)
}

//////////////////////////////////////////////////////////////////////////////
// Colorizers.

// singleIntervalColorizerFP is the colorizer for a two-color gradient: a single lerp across the whole [0, 1] range.
type singleIntervalColorizerFP struct {
	FPBase
	start, end colorcore.Color4f
}

func makeSingleIntervalColorizer(start, end colorcore.Color4f) FragmentProcessor {
	fp := &singleIntervalColorizerFP{start: start, end: end}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *singleIntervalColorizerFP) Name() string { return "SingleIntervalColorizer" }

func (f *singleIntervalColorizerFP) Clone() FragmentProcessor {
	return makeSingleIntervalColorizer(f.start, f.end)
}

func (f *singleIntervalColorizerFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*singleIntervalColorizerFP)
	return ok && f.start == o.start && f.end == o.end
}

func (f *singleIntervalColorizerFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPSingleIntervalColorizer), "runtimeFPKind")
}

func (f *singleIntervalColorizerFP) onMakeProgramImpl() FPProgramImpl {
	return &singleIntervalImpl{}
}

type singleIntervalImpl struct {
	FPImplBase
	startUni, endUni UniformHandle
}

func (i *singleIntervalImpl) EmitCode(args *FPEmitArgs) {
	var startName, endName string
	i.startUni, startName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "start")
	i.endUni, endName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf4,
		"end")
	// Clamping and/or wrapping was already handled by the parent shader, so the output color is a simple lerp.
	args.FragBuilder.CodeAppendf("return mix(%s, %s, %s.x);", startName, endName,
		args.SampleCoord)
}

func (i *singleIntervalImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*singleIntervalColorizerFP)
	pdman.Set4f(i.startUni, f.start.R, f.start.G, f.start.B, f.start.A)
	pdman.Set4f(i.endUni, f.end.R, f.end.G, f.end.B, f.end.A)
}

// dualIntervalColorizerFP is the colorizer for a gradient with exactly two color intervals split at threshold:
// precomputed scale/bias per interval.
type dualIntervalColorizerFP struct {
	FPBase
	scale     [8]float32 // two float4s
	bias      [8]float32
	threshold float32
}

func makeDualIntervalColorizer(c0, c1, c2, c3 colorcore.Color4f, threshold float32) FragmentProcessor {
	fp := &dualIntervalColorizerFP{threshold: threshold}
	// Derive scale and biases from the 4 colors and threshold.
	for i := 0; i < 4; i++ {
		v0 := colorComponent(c0, i)
		v1 := colorComponent(c1, i)
		v2 := colorComponent(c2, i)
		v3 := colorComponent(c3, i)
		fp.scale[i] = (v1 - v0) / threshold
		fp.scale[4+i] = (v3 - v2) / (1 - threshold)
		fp.bias[i] = v0
		fp.bias[4+i] = v2 - threshold*fp.scale[4+i]
	}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func colorComponent(c colorcore.Color4f, i int) float32 {
	switch i {
	case 0:
		return c.R
	case 1:
		return c.G
	case 2:
		return c.B
	default:
		return c.A
	}
}

func (f *dualIntervalColorizerFP) Name() string { return "DualIntervalColorizer" }

func (f *dualIntervalColorizerFP) Clone() FragmentProcessor {
	clone := &dualIntervalColorizerFP{scale: f.scale, bias: f.bias, threshold: f.threshold}
	clone.initFP(GLSLFPClassID, 0)
	clone.setUsesSampleCoordsDirectly()
	return clone
}

func (f *dualIntervalColorizerFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*dualIntervalColorizerFP)
	return ok && f.scale == o.scale && f.bias == o.bias && f.threshold == o.threshold
}

func (f *dualIntervalColorizerFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPDualIntervalColorizer), "runtimeFPKind")
}

func (f *dualIntervalColorizerFP) onMakeProgramImpl() FPProgramImpl { return &dualIntervalImpl{} }

type dualIntervalImpl struct {
	FPImplBase
	scaleUni, biasUni, thresholdUni UniformHandle
}

func (i *dualIntervalImpl) EmitCode(args *FPEmitArgs) {
	var scaleName, biasName, thresholdName string
	i.scaleUni, scaleName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "scale", 2)
	i.biasUni, biasName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "bias", 2)
	i.thresholdUni, thresholdName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "threshold")
	fb := args.FragBuilder
	fb.CodeAppendf("float t = %s.x;", args.SampleCoord)
	fb.CodeAppend("vec4 s, b;")
	fb.CodeAppendf("if (t < %s) { s = %s[0]; b = %s[0]; } else { s = %s[1]; b = %s[1]; }",
		thresholdName, scaleName, biasName, scaleName, biasName)
	fb.CodeAppend("return t * s + b;")
}

func (i *dualIntervalImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*dualIntervalColorizerFP)
	pdman.Set4fv(i.scaleUni, 2, f.scale[:])
	pdman.Set4fv(i.biasUni, 2, f.bias[:])
	pdman.Set1f(i.thresholdUni, f.threshold)
}

// loopingColorizerFP is the colorizer for gradients with more than a few stops: a fixed-trip-count binary search over
// uniform arrays of interval scale/bias/thresholds. The interval count is baked into the shader text (it sizes the
// arrays and the loop), so it participates in the key.
type loopingColorizerFP struct {
	thresholds []float32 // 4 * chunks
	scale      []float32 // 4 * intervalCount
	bias       []float32 // 4 * intervalCount
	FPBase
	intervalCount int
}

func makeLoopingColorizer(intervalCount int, scale, bias, thresholds []float32) FragmentProcessor {
	if intervalCount < 1 || intervalCount > maxLoopingIntervalCount || intervalCount%4 != 0 {
		panic("invalid looping colorizer interval count")
	}
	fp := &loopingColorizerFP{
		intervalCount: intervalCount,
		thresholds:    thresholds,
		scale:         scale,
		bias:          bias,
	}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *loopingColorizerFP) Name() string { return "LoopingBinaryColorizer" }

func (f *loopingColorizerFP) Clone() FragmentProcessor {
	return makeLoopingColorizer(f.intervalCount, f.scale, f.bias, f.thresholds)
}

func (f *loopingColorizerFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*loopingColorizerFP)
	if !ok || f.intervalCount != o.intervalCount {
		return false
	}
	return floatsEqual(f.thresholds, o.thresholds) && floatsEqual(f.scale, o.scale) &&
		floatsEqual(f.bias, o.bias)
}

func floatsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (f *loopingColorizerFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPLoopingColorizer), "runtimeFPKind")
	b.Add32(uint32(f.intervalCount), "intervalCount")
}

func (f *loopingColorizerFP) onMakeProgramImpl() FPProgramImpl { return &loopingColorizerImpl{} }

type loopingColorizerImpl struct {
	FPImplBase
	thresholdsUni, scaleUni, biasUni UniformHandle
}

// skNextLog2 returns ceil(log2(v)) for positive v, or 0 for v <= 1.
func skNextLog2(v int) int {
	if v <= 1 {
		return 0
	}
	return bits.Len(uint(v - 1))
}

func (i *loopingColorizerImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*loopingColorizerFP)
	intervalChunks := fp.intervalCount / 4
	loopCount := skNextLog2(intervalChunks)

	var thresholdsName, scaleName, biasName string
	i.thresholdsUni, thresholdsName = args.UniformHandler.AddUniformArray(args.FP,
		ShaderFlagFragment, GLSLTypeFloat4, "thresholds", intervalChunks)
	i.scaleUni, scaleName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "scale", fp.intervalCount)
	i.biasUni, biasName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "bias", fp.intervalCount)

	fb := args.FragBuilder
	fb.CodeAppendf("float t = %s.x;", args.SampleCoord)
	// Choose a chunk from thresholds via binary search in a fixed-trip-count loop.
	fb.CodeAppend("int low = 0;")
	fb.CodeAppendf("int high = %d;", intervalChunks-1)
	fb.CodeAppendf("int chunk = %d;", (intervalChunks-1)/2)
	fb.CodeAppendf("for (int loop = 0; loop < %d; ++loop) {", loopCount)
	fb.CodeAppendf("if (t < %s[chunk].w) { high = chunk; } else { low = chunk + 1; }",
		thresholdsName)
	fb.CodeAppend("chunk = (low + high) / 2;")
	fb.CodeAppend("}")
	// Choose the final position via explicit 4-way binary search.
	fb.CodeAppend("int pos;")
	fb.CodeAppendf("if (t < %s[chunk].y) { pos = (t < %s[chunk].x) ? 0 : 1; }"+
		" else { pos = (t < %s[chunk].z) ? 2 : 3; }",
		thresholdsName, thresholdsName, thresholdsName)
	if loopCount > 0 {
		fb.CodeAppend("pos += 4 * chunk;")
	}
	fb.CodeAppendf("return t * %s[pos] + %s[pos];", scaleName, biasName)
}

func (i *loopingColorizerImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*loopingColorizerFP)
	pdman.Set4fv(i.thresholdsUni, int32(len(f.thresholds)/4), f.thresholds)
	pdman.Set4fv(i.scaleUni, int32(f.intervalCount), f.scale)
	pdman.Set4fv(i.biasUni, int32(f.intervalCount), f.bias)
}

//////////////////////////////////////////////////////////////////////////////
// Top-level tile-mode effects.

// clampedGradientFP is the clamp/decal top-level tile-mode effect: clamps the layout coordinate with explicit border
// colors (the clamp tile mode uses the outer stop colors; decal uses transparent black).
type clampedGradientFP struct {
	FPBase
	leftBorderColor        colorcore.PMColor4f
	rightBorderColor       colorcore.PMColor4f
	layoutPreservesOpacity bool // specialized
}

func makeClampedGradientFP(colorizer, gradLayout FragmentProcessor, leftBorderColor, rightBorderColor colorcore.PMColor4f, colorsAreOpaque bool) FragmentProcessor {
	layoutPreservesOpacity := gradLayout.fpBase().PreservesOpaqueInput()
	// If the layout does not preserve opacity, remove the opaque optimization, but otherwise respect the provided color
	// opacity state (which accounts for the border colors).
	flags := FPCompatibleWithCoverageAsAlpha
	if colorsAreOpaque && layoutPreservesOpacity {
		flags |= FPPreservesOpaqueInput
	}
	fp := &clampedGradientFP{
		leftBorderColor:        leftBorderColor,
		rightBorderColor:       rightBorderColor,
		layoutPreservesOpacity: layoutPreservesOpacity,
	}
	fp.initFP(GLSLFPClassID, flags)
	// Both children's optimization flags are ignored here, so they are not merged into this FP's flags.
	registerChildOf(fp, colorizer, ExplicitSampleUsage())
	registerChildOf(fp, gradLayout, PassThroughSampleUsage())
	return fp
}

func (f *clampedGradientFP) Name() string { return "ClampedGradient" }

func (f *clampedGradientFP) Clone() FragmentProcessor {
	clone := &clampedGradientFP{
		leftBorderColor:        f.leftBorderColor,
		rightBorderColor:       f.rightBorderColor,
		layoutPreservesOpacity: f.layoutPreservesOpacity,
	}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *clampedGradientFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*clampedGradientFP)
	return ok && f.leftBorderColor == o.leftBorderColor &&
		f.rightBorderColor == o.rightBorderColor &&
		f.layoutPreservesOpacity == o.layoutPreservesOpacity
}

func (f *clampedGradientFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPClampedGradient), "runtimeFPKind")
	b.AddBool(f.layoutPreservesOpacity, "layoutPreservesOpacity")
}

func (f *clampedGradientFP) onMakeProgramImpl() FPProgramImpl { return &clampedGradientImpl{} }

type clampedGradientImpl struct {
	FPImplBase
	leftUni, rightUni UniformHandle
}

func (i *clampedGradientImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*clampedGradientFP)
	var leftName, rightName string
	i.leftUni, leftName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "leftBorderColor")
	i.rightUni, rightName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "rightBorderColor")
	fb := args.FragBuilder
	fb.CodeAppendf("vec4 t = %s;", i.InvokeChild(1 /* gradLayout */, args))
	fb.CodeAppend("vec4 outColor;")
	if !fp.layoutPreservesOpacity {
		// The layout has rejected this fragment (the branch is eliminated at "compile time" when the layout preserves
		// opacity, since layoutPreservesOpacity is baked into the shader text).
		fb.CodeAppend("if (t.y < 0.0) { outColor = vec4(0.0); } else")
	}
	// If t.x is below 0, use the left border color without invoking the child processor; above 1, the right border
	// color. Otherwise t is in the [0, 1] range assumed by the colorizer.
	fb.CodeAppendf(" if (t.x < 0.0) { outColor = %s; }", leftName)
	fb.CodeAppendf(" else if (t.x > 1.0) { outColor = %s; }", rightName)
	// Always sample from (x, 0), discarding y, since the layout FP can use y as a side-channel.
	fb.CodeAppendf(" else { outColor = %s; }",
		i.InvokeChildWithCoords(0 /* colorizer */, "", args, "vec2(t.x, 0.0)"))
	fb.CodeAppend("return outColor;")
}

func (i *clampedGradientImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*clampedGradientFP)
	pdman.Set4f(i.leftUni, f.leftBorderColor.R, f.leftBorderColor.G, f.leftBorderColor.B,
		f.leftBorderColor.A)
	pdman.Set4f(i.rightUni, f.rightBorderColor.R, f.rightBorderColor.G, f.rightBorderColor.B,
		f.rightBorderColor.A)
}

// tiledGradientFP is the repeat/mirror top-level tile-mode effect: repeat or mirror tiling of the layout coordinate.
type tiledGradientFP struct {
	FPBase
	mirror                 bool // specialized
	layoutPreservesOpacity bool // specialized
	useFloorAbsWorkaround  bool // specialized
}

func makeTiledGradientFP(caps *gpu.ShaderCaps, colorizer, gradLayout FragmentProcessor, mirror, colorsAreOpaque bool) FragmentProcessor {
	layoutPreservesOpacity := gradLayout.fpBase().PreservesOpaqueInput()
	flags := FPCompatibleWithCoverageAsAlpha
	if colorsAreOpaque && layoutPreservesOpacity {
		flags |= FPPreservesOpaqueInput
	}
	fp := &tiledGradientFP{
		mirror:                 mirror,
		layoutPreservesOpacity: layoutPreservesOpacity,
		useFloorAbsWorkaround:  caps.MustDoOpBetweenFloorAndAbs,
	}
	fp.initFP(GLSLFPClassID, flags)
	registerChildOf(fp, colorizer, ExplicitSampleUsage())
	registerChildOf(fp, gradLayout, PassThroughSampleUsage())
	return fp
}

func (f *tiledGradientFP) Name() string { return "TiledGradient" }

func (f *tiledGradientFP) Clone() FragmentProcessor {
	clone := &tiledGradientFP{
		mirror:                 f.mirror,
		layoutPreservesOpacity: f.layoutPreservesOpacity,
		useFloorAbsWorkaround:  f.useFloorAbsWorkaround,
	}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *tiledGradientFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*tiledGradientFP)
	return ok && f.mirror == o.mirror && f.layoutPreservesOpacity == o.layoutPreservesOpacity &&
		f.useFloorAbsWorkaround == o.useFloorAbsWorkaround
}

func (f *tiledGradientFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPTiledGradient), "runtimeFPKind")
	b.AddBool(f.mirror, "mirror")
	b.AddBool(f.layoutPreservesOpacity, "layoutPreservesOpacity")
	b.AddBool(f.useFloorAbsWorkaround, "useFloorAbsWorkaround")
}

func (f *tiledGradientFP) onMakeProgramImpl() FPProgramImpl { return &tiledGradientImpl{} }

type tiledGradientImpl struct {
	FPImplBase
}

func (i *tiledGradientImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*tiledGradientFP)
	fb := args.FragBuilder
	fb.CodeAppendf("vec4 t = %s;", i.InvokeChild(1 /* gradLayout */, args))
	if !fp.layoutPreservesOpacity {
		fb.CodeAppend("if (t.y < 0.0) { return vec4(0.0); }")
	}
	if fp.mirror {
		fb.CodeAppend("float t_1 = t.x - 1.0;")
		fb.CodeAppend("float tiled_t = t_1 - 2.0 * floor(t_1 * 0.5) - 1.0;")
		if fp.useFloorAbsWorkaround {
			// The expected value of tiled_t is in [-1, 1], so this clamp has no effect other than to break up the floor
			// and abs calls so the compiler can't merge them.
			fb.CodeAppend("tiled_t = clamp(tiled_t, -1.0, 1.0);")
		}
		fb.CodeAppend("t.x = abs(tiled_t);")
	} else {
		// Simple repeat mode.
		fb.CodeAppend("t.x = fract(t.x);")
	}
	// Always sample from (x, 0), discarding y, since the layout FP can use y as a side-channel.
	fb.CodeAppendf("vec4 outColor = %s;",
		i.InvokeChildWithCoords(0 /* colorizer */, "", args, "vec2(t.x, 0.0)"))
	fb.CodeAppend("return outColor;")
}

//////////////////////////////////////////////////////////////////////////////
// Colorizer selection and top-level assembly.

// buildGradientIntervals converts {colors, positions} into {scales, biases, thresholds} (flattened float4s). Returns
// the interval count, or 0 when the gradient needs more intervals than outputLength.
func buildGradientIntervals(colors []colorcore.Color4f, positions []float32, outputLength int) (scales, biases, thresholds []float32, intervalCount int) {
	scales = make([]float32, 0, 4*outputLength)
	biases = make([]float32, 0, 4*outputLength)
	thresholds = make([]float32, 0, outputLength)
	for i := 0; i < len(colors)-1; i++ {
		if intervalCount >= outputLength {
			// Already reached the output limit with color stops remaining: this gradient cannot be represented without
			// more intervals.
			return nil, nil, nil, 0
		}
		t0 := positions[i]
		t1 := positions[i+1]
		dt := t1 - t0
		// If the interval is empty, skip to the next one (this automatically creates distinct hard-stop intervals and
		// protects against unreachable leading hard stops).
		if geom.ScalarNearlyZero(dt) {
			continue
		}
		for c := 0; c < 4; c++ {
			scale := (colorComponent(colors[i+1], c) - colorComponent(colors[i], c)) / dt
			scales = append(scales, scale)
			biases = append(biases, colorComponent(colors[i], c)-t0*scale)
		}
		thresholds = append(thresholds, t1)
		intervalCount++
	}
	return scales, biases, thresholds, intervalCount
}

// makeLoopingBinaryColorizer builds the looping-binary colorizer for a gradient with more stops than the
// single/dual-interval colorizers handle.
func makeLoopingBinaryColorizer(colors []colorcore.Color4f, positions []float32) FragmentProcessor {
	if len(colors) > maxLoopingColorCount {
		// Definitely cannot represent this gradient configuration.
		return nil
	}
	scales, biases, thresholds, intervalCount := buildGradientIntervals(colors, positions,
		maxLoopingIntervalCount)
	if intervalCount <= 0 {
		return nil
	}
	// Round up the number of intervals to the next power of two (at least 4): fewer unique shaders at the cost of a
	// handful of wasted uniforms.
	roundedSize := 4
	for roundedSize < intervalCount {
		roundedSize *= 2
	}
	// roundedSize is always a multiple of 4, so this also leaves the thresholds padded to a whole number of float4
	// chunks, which is how the colorizer uploads them.
	for ; intervalCount < roundedSize; intervalCount++ {
		thresholds = append(thresholds, thresholds[intervalCount-1])
		scales = append(scales, scales[4*(intervalCount-1):4*intervalCount]...)
		biases = append(biases, biases[4*(intervalCount-1):4*intervalCount]...)
	}
	return makeLoopingColorizer(intervalCount, scales, biases, thresholds)
}

// makeUniformColorizer analyzes the stops and chooses an analytic colorizer, or returns nil when the gradient is too
// complex (the caller then falls back to the textured colorizer).
func makeUniformColorizer(colors []colorcore.Color4f, positions []float32, caps *gpu.ShaderCaps) FragmentProcessor {
	count := len(colors)

	// Hard stops at the ends are handled by the clamped border colors; strip them so the remaining stop analysis is
	// simpler.
	if geom.ScalarNearlyEqual(positions[0], positions[1]) { // bottom hard stop
		colors = colors[1:]
		positions = positions[1:]
		count--
	}
	if geom.ScalarNearlyEqual(positions[count-2], positions[count-1]) { // top hard stop
		count--
		colors = colors[:count]
		positions = positions[:count]
	}

	// Two remaining colors means a single interval from 0 to 1 (possibly originally a 3-4 color gradient with hard
	// stops at the ends).
	if count == 2 {
		return makeSingleIntervalColorizer(colors[0], colors[1])
	}

	// intervalsExceedPrecisionLimit is a half-float-hardware concern; FloatIs32Bits is always true on the desktop trim,
	// but the check is retained for completeness.
	if !caps.FloatIs32Bits {
		const lowPrecisionIntervalLimit = 0.01
		for i := 0; i < count-1; i++ {
			dt := geom.ScalarAbs(positions[i] - positions[i+1])
			if dt <= lowPrecisionIntervalLimit && dt > geom.ScalarNearlyZeroTol {
				return nil
			}
		}
	}

	limit := maxLoopingColorCount
	if !caps.NonconstantArrayIndexSupport {
		// The unrolled-binary colorizer lane; unreachable with desktop GLSL.
		panic("unrolled colorizer lane is unreachable on the desktop trim")
	}
	if count <= limit {
		// The dual-interval colorizer uses the same principles as the binary-search colorizer, but is limited to
		// exactly 2 intervals.
		if count == 3 {
			// Must be a dual interval gradient with the middle stop shared.
			return makeDualIntervalColorizer(colors[0], colors[1], colors[1], colors[2],
				positions[1])
		}
		if count == 4 && geom.ScalarNearlyEqual(positions[1], positions[2]) {
			// Two separate intervals that join at the same threshold position.
			return makeDualIntervalColorizer(colors[0], colors[1], colors[2], colors[3],
				positions[1])
		}
		if colorizer := makeLoopingBinaryColorizer(colors, positions); colorizer != nil {
			return colorizer
		}
	}
	return nil
}

// gradientDescriptor is the subset of the gradientBase accessors the FP assembly needs.
type gradientDescriptor interface {
	Colors() []colorcore.Color4f
	Positions() []float32
	TileMode() shaders.TileMode
	PtsToUnit() geom.Matrix
}

// premulColor4f converts an (interpolation-space, unpremul) stop color to premul, mirroring the interpolated-to-dst
// premul step applied to border colors on the CPU.
func premulColor4f(c colorcore.Color4f) colorcore.PMColor4f {
	return colorcore.PMColor4f{R: c.R * c.A, G: c.G * c.A, B: c.B * c.A, A: c.A}
}

// makeGradientFP combines the layout with a colorizer and the tile-mode top-level effect. overrideMatrix (used by the
// conical radial variant) replaces the descriptor's gradient matrix when non-nil.
func makeGradientFP(shader gradientDescriptor, args *FPArgs, mRec *shaders.MatrixRec, layout FragmentProcessor, overrideMatrix *geom.Matrix) FragmentProcessor {
	// No shader is possible if a layout couldn't be created.
	if layout == nil {
		return nil
	}

	m := shader.PtsToUnit()
	if overrideMatrix != nil {
		m = *overrideMatrix
	}
	total, ok := mRec.ApplyForFragmentProcessor(&m)
	if !ok {
		return nil
	}
	layout = MatrixEffectFP(total, layout)

	// There is no color-space conversion: the descriptor's fixed stops are already the interpolation-space colors
	// (legacy unpremul-sRGB lerp). Positions are explicit, or synthesized when the descriptor is evenly spaced.
	colors := shader.Colors()
	positions := shader.Positions()
	if positions == nil {
		positions = make([]float32, len(colors))
		denom := float32(len(colors) - 1)
		for i := range positions {
			positions[i] = float32(i) / denom
		}
	}

	allOpaque := true
	for i := range colors {
		if allOpaque && !geom.ScalarNearlyEqual(colors[i].A, 1.0) {
			allOpaque = false
		}
	}

	// All gradients are colorized the same way, regardless of layout.
	colorizer := makeUniformColorizer(colors, positions, args.Caps.ShaderCaps)
	if colorizer != nil {
		// Wrap the colorizer in the conversion from interpolation space to destination, which in an sRGB-only pipeline
		// is exactly the final premultiplication (a no-op when every stop is opaque).
		if !allOpaque {
			colorizer = ColorXformFP(colorizer, XformStepPremul)
		}
	} else {
		// The gradient is too complex for the analytic colorizers (more than 128 fixed stops): rasterize it into a
		// 256×1 LUT texture instead. The LUT bake directly encodes the interpolated-to-dst result — including the final
		// premultiplication — so the ColorXformFP wrapper above is skipped. The draw is skipped only when texture
		// creation fails.
		colorizer = makeTexturedColorizer(colors, positions, allOpaque, args)
		if colorizer == nil {
			return nil
		}
	}

	switch shader.TileMode() {
	case shaders.TileRepeat:
		return makeTiledGradientFP(args.Caps.ShaderCaps, colorizer, layout,
			false /* mirror */, allOpaque)
	case shaders.TileMirror:
		return makeTiledGradientFP(args.Caps.ShaderCaps, colorizer, layout,
			true /* mirror */, allOpaque)
	case shaders.TileClamp:
		// The border colors are the first and last stop colors (finished converting to the destination space:
		// premultiplied); the stop fixing upstream guarantees these correspond to t=0 and t=1 even with hard stops.
		return makeClampedGradientFP(colorizer, layout, premulColor4f(colors[0]),
			premulColor4f(colors[len(colors)-1]), allOpaque)
	case shaders.TileDecal:
		// Even if the gradient colors are opaque, the decal borders are transparent, so that optimization is disabled.
		return makeClampedGradientFP(colorizer, layout, colorcore.PMColor4f{},
			colorcore.PMColor4f{}, false /* colorsAreOpaque */)
	}
	panic("unknown tile mode")
}

//////////////////////////////////////////////////////////////////////////////
// Per-family conversions: build the FP tree for each gradient shader type.

func makeLinearGradientFP(shader *shaders.LinearGradient, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	return makeGradientFP(shader, args, mRec, makeLinearLayoutFP(), nil)
}

func makeRadialGradientFP(shader *shaders.RadialGradient, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	return makeGradientFP(shader, args, mRec, makeRadialLayoutFP(), nil)
}

func makeSweepGradientFP(shader *shaders.SweepGradient, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	return makeGradientFP(shader, args, mRec, makeSweepLayoutFP(shader.TBias(), shader.TScale()),
		nil)
}

func makeConicalGradientFP(shader *shaders.ConicalGradient, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	var layout FragmentProcessor
	var overrideMatrix *geom.Matrix
	switch shader.Kind() {
	case shaders.ConicalKindStrip:
		r0 := shader.Radius1() / shader.CenterX1()
		layout = makeConicalStripLayoutFP(r0 * r0)
	case shaders.ConicalKindRadial:
		dr := shader.DiffRadius()
		r0 := shader.Radius1() / dr
		lengthScale := float32(1)
		if dr < 0 {
			lengthScale = -1
		}
		layout = makeConicalRadialLayoutFP(r0, lengthScale)
		// The GPU radial matrix is different from the original: |diffRadius| maps to 1, so the final gradient matrix is
		// computed manually here.
		m := geom.TranslateMatrix(-shader.Center1().X, -shader.Center1().Y)
		m.PostScale(1/dr, 1/dr)
		overrideMatrix = &m
	case shaders.ConicalKindFocal:
		isRadiusIncreasing := (1 - shader.FocalX()) > 0
		layout = makeConicalFocalLayoutFP(isRadiusIncreasing, shader.FocalOnCircle(),
			shader.FocalWellBehaved(), shader.FocalIsSwapped(), shader.FocalNativelyFocal(),
			1/shader.FocalR1(), shader.FocalX())
	}
	return makeGradientFP(shader, args, mRec, layout, overrideMatrix)
}
