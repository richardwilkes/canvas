// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Small helper fragment processors used throughout the pipeline: MakeColorFP produces a constant color, OverrideInputFP
// feeds a fixed color to a child instead of the input, DisableCoverageAsAlphaFP clears the coverage-as-alpha
// optimization, ComposeFP chains two processors in series, SwizzleOutputFP permutes a child's output channels,
// MulInputByChildAlphaFP multiplies the input by a child's alpha, and ApplyPaintAlphaFP defers alpha application until
// after a child runs. Each is a small GLSL-emitting fragment processor whose optimization flags are computed from its
// own semantics and merged with any child's flags (pass-through children merge, constant-output when the effect
// supports it).

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

//////////////////////////////////////////////////////////////////////////////
// MakeColor ("color_fp"): always returns the stored color.

type colorFP struct {
	FPBase
	color colorcore.PMColor4f
}

// MakeColorFP returns a fragment processor that always outputs the given color, ignoring its input.
func MakeColorFP(color colorcore.PMColor4f) FragmentProcessor {
	flags := FPConstantOutputForConstantInput
	if pmIsOpaque(color) {
		flags |= FPPreservesOpaqueInput
	}
	fp := &colorFP{color: color}
	fp.initFP(GLSLFPClassID, flags)
	return fp
}

func (f *colorFP) Name() string { return "color_fp" }

func (f *colorFP) Clone() FragmentProcessor { return MakeColorFP(f.color) }

func (f *colorFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	// The program key is a runtimeFPKind tag plus specialized uniforms; the color is a regular uniform. The tag
	// distinguishes these hand-written effects from one another even though they share a single class ID.
	b.Add32(uint32(runtimeFPColor), "runtimeFPKind")
}

func (f *colorFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*colorFP)
	return ok && f.color == o.color
}

func (f *colorFP) constantOutputForConstantInput(colorcore.PMColor4f) colorcore.PMColor4f {
	return f.color
}

func (f *colorFP) onMakeProgramImpl() FPProgramImpl { return &colorFPImpl{} }

type colorFPImpl struct {
	FPImplBase
	colorUni  UniformHandle
	lastColor colorcore.PMColor4f
	hasLast   bool
}

func (i *colorFPImpl) EmitCode(args *FPEmitArgs) {
	colorName := ""
	i.colorUni, colorName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "color")
	args.FragBuilder.CodeAppendf("return %s;", colorName)
}

func (i *colorFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	c := fp.(*colorFP).color
	if !i.hasLast || i.lastColor != c {
		pdman.Set4f(i.colorUni, c.R, c.G, c.B, c.A)
		i.lastColor = c
		i.hasLast = true
	}
}

// runtimeFPKind separates these hand-written fragment-processor effects from one another in program keys; each distinct
// effect gets a distinct value.
type runtimeFPKind uint32

const (
	runtimeFPColor runtimeFPKind = iota
	runtimeFPOverrideInput
	runtimeFPDisableCoverageAsAlpha
	runtimeFPApplyPaintAlpha
	runtimeFPLinearLayout
	runtimeFPRadialLayout
	runtimeFPSweepLayout
	runtimeFPConicalStripLayout
	runtimeFPConicalRadialLayout
	runtimeFPConicalFocalLayout
	runtimeFPSingleIntervalColorizer
	runtimeFPDualIntervalColorizer
	runtimeFPLoopingColorizer
	runtimeFPClampedGradient
	runtimeFPTiledGradient
	runtimeFPColorMatrix
	runtimeFPLuma
	runtimeFPHighContrast
	runtimeFPDither
	runtimeFPClampOutput
	runtimeFPRectClip
	runtimeFPCircleClip
	runtimeFPEllipseClip
	runtimeFPBlur1D
	runtimeFPBlur2D
	runtimeFPCircleBlur
	runtimeFPRectBlur
	runtimeFPRRectBlur
	runtimeFPFilterDecal
	runtimeFPLinearMorphology
	runtimeFPSparseMorphology
	runtimeFPDisplacement
	runtimeFPNormal
	runtimeFPLighting
	runtimeFPMagnifier
	runtimeFPMatrixConv
	runtimeFPArithmetic
)

//////////////////////////////////////////////////////////////////////////////
// OverrideInput: feeds a constant color to the child instead of the input color.

type overrideInputFP struct {
	FPBase
	color colorcore.PMColor4f
}

// OverrideInputFP returns a fragment processor that feeds child a fixed color instead of the input color.
func OverrideInputFP(child FragmentProcessor, color colorcore.PMColor4f) FragmentProcessor {
	if child == nil {
		return nil
	}
	// Declares PreservesOpaqueInput (for opaque colors) plus constant-output support, then intersects with the child's
	// flags as the child is added.
	declared := FPConstantOutputForConstantInput
	if pmIsOpaque(color) {
		declared |= FPPreservesOpaqueInput
	}
	fp := &overrideInputFP{color: color}
	fp.initFP(GLSLFPClassID, declared)
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *overrideInputFP) Name() string { return "OverrideInput" }

func (f *overrideInputFP) Clone() FragmentProcessor {
	return OverrideInputFP(f.ChildProcessor(0).Clone(), f.color)
}

func (f *overrideInputFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPOverrideInput), "runtimeFPKind")
}

func (f *overrideInputFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*overrideInputFP)
	return ok && f.color == o.color
}

func (f *overrideInputFP) constantOutputForConstantInput(colorcore.PMColor4f) colorcore.PMColor4f {
	return fpConstantOutputForConstantInput(f.ChildProcessor(0), f.color)
}

func (f *overrideInputFP) onMakeProgramImpl() FPProgramImpl { return &overrideInputFPImpl{} }

type overrideInputFPImpl struct {
	FPImplBase
	colorUni UniformHandle
}

func (i *overrideInputFPImpl) EmitCode(args *FPEmitArgs) {
	colorName := ""
	i.colorUni, colorName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "color")
	args.FragBuilder.CodeAppendf("return %s;",
		i.InvokeChildWithColor(0, colorName, args))
}

func (i *overrideInputFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	c := fp.(*overrideInputFP).color
	pdman.Set4f(i.colorUni, c.R, c.G, c.B, c.A)
}

//////////////////////////////////////////////////////////////////////////////
// DisableCoverageAsAlpha: returns the child's color but clears the coverage-as-alpha optimization.

type disableCoverageAsAlphaFP struct {
	FPBase
}

// DisableCoverageAsAlphaFP wraps child so it no longer allows the coverage-as-alpha optimization, unless child already
// disallows it (or is nil), in which case child is returned unchanged.
func DisableCoverageAsAlphaFP(child FragmentProcessor) FragmentProcessor {
	if child == nil || !child.fpBase().CompatibleWithCoverageAsAlpha() {
		return child
	}
	declared := FPPreservesOpaqueInput | FPConstantOutputForConstantInput
	fp := &disableCoverageAsAlphaFP{}
	fp.initFP(GLSLFPClassID, declared)
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *disableCoverageAsAlphaFP) Name() string { return "DisableCoverageAsAlpha" }

func (f *disableCoverageAsAlphaFP) Clone() FragmentProcessor {
	return DisableCoverageAsAlphaFP(f.ChildProcessor(0).Clone())
}

func (f *disableCoverageAsAlphaFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPDisableCoverageAsAlpha), "runtimeFPKind")
}

func (f *disableCoverageAsAlphaFP) onIsEqual(other FragmentProcessor) bool {
	_, ok := other.(*disableCoverageAsAlphaFP)
	return ok
}

func (f *disableCoverageAsAlphaFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	return fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
}

func (f *disableCoverageAsAlphaFP) onMakeProgramImpl() FPProgramImpl {
	return &disableCoverageAsAlphaFPImpl{}
}

type disableCoverageAsAlphaFPImpl struct {
	FPImplBase
}

func (i *disableCoverageAsAlphaFPImpl) EmitCode(args *FPEmitArgs) {
	args.FragBuilder.CodeAppendf("return %s;", i.InvokeChild(0, args))
}

//////////////////////////////////////////////////////////////////////////////
// ApplyPaintAlpha: invokes the child with an opaque version of the input color, then applies the input alpha to the
// result.

type applyPaintAlphaFP struct {
	FPBase
}

// ApplyPaintAlphaFP returns a fragment processor that invokes child with the input color made opaque, then multiplies
// the result by the input's alpha.
func ApplyPaintAlphaFP(child FragmentProcessor) FragmentProcessor {
	if child == nil {
		panic("ApplyPaintAlpha requires a child")
	}
	declared := FPPreservesOpaqueInput | FPCompatibleWithCoverageAsAlpha |
		FPConstantOutputForConstantInput
	fp := &applyPaintAlphaFP{}
	fp.initFP(GLSLFPClassID, declared)
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *applyPaintAlphaFP) Name() string { return "ApplyPaintAlpha" }

func (f *applyPaintAlphaFP) Clone() FragmentProcessor {
	return ApplyPaintAlphaFP(f.ChildProcessor(0).Clone())
}

func (f *applyPaintAlphaFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPApplyPaintAlpha), "runtimeFPKind")
}

func (f *applyPaintAlphaFP) onIsEqual(other FragmentProcessor) bool {
	_, ok := other.(*applyPaintAlphaFP)
	return ok
}

func (f *applyPaintAlphaFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	opaque := colorcore.PMColor4f{R: input.R, G: input.G, B: input.B, A: 1}
	out := fpConstantOutputForConstantInput(f.ChildProcessor(0), opaque)
	return colorcore.PMColor4f{
		R: out.R * input.A, G: out.G * input.A, B: out.B * input.A, A: out.A * input.A,
	}
}

func (f *applyPaintAlphaFP) onMakeProgramImpl() FPProgramImpl { return &applyPaintAlphaFPImpl{} }

type applyPaintAlphaFPImpl struct {
	FPImplBase
}

func (i *applyPaintAlphaFPImpl) EmitCode(args *FPEmitArgs) {
	args.FragBuilder.CodeAppendf("vec4 _paintColor = %s;", args.InputColor)
	args.FragBuilder.CodeAppendf("return %s * _paintColor.a;",
		i.InvokeChildWithColor(0, "vec4(_paintColor.rgb, 1.0)", args))
}

//////////////////////////////////////////////////////////////////////////////
// Compose ("SeriesFragmentProcessor"): f(g(x)).

type composeFP struct {
	FPBase
}

// ComposeFP composes two fragment processors f and g into f(g(x)) — running them in series (g, then f) with no blending
// step. Either may be nil, in which case the other is returned unchanged.
func ComposeFP(f, g FragmentProcessor) FragmentProcessor {
	// Allow either of the composed functions to be nil.
	if f == nil {
		return g
	}
	if g == nil {
		return f
	}

	// Upstream runs a constant-folding analysis here, but its input color is unknown by construction, so no leading
	// processor can ever be eliminated; the composition is always emitted as requested.
	return newComposeFP(f, g)
}

func newComposeFP(f, g FragmentProcessor) FragmentProcessor {
	fp := &composeFP{}
	fp.initFP(SeriesFragmentProcessorClassID,
		f.fpBase().optimizationFlags()&g.fpBase().optimizationFlags())
	registerChildOf(fp, f, PassThroughSampleUsage())
	registerChildOf(fp, g, PassThroughSampleUsage())
	return fp
}

func (f *composeFP) Name() string { return "Compose" }

func (f *composeFP) Clone() FragmentProcessor {
	clone := &composeFP{}
	clone.initFP(SeriesFragmentProcessorClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *composeFP) onAddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

func (f *composeFP) onIsEqual(FragmentProcessor) bool { return true }

func (f *composeFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	color := input
	color = fpConstantOutputForConstantInput(f.ChildProcessor(1), color)
	color = fpConstantOutputForConstantInput(f.ChildProcessor(0), color)
	return color
}

func (f *composeFP) onMakeProgramImpl() FPProgramImpl { return &composeFPImpl{} }

type composeFPImpl struct {
	FPImplBase
}

func (i *composeFPImpl) EmitCode(args *FPEmitArgs) {
	result := i.InvokeChild(1, args)                 // g(x)
	result = i.InvokeChildWithColor(0, result, args) // f(g(x))
	args.FragBuilder.CodeAppendf("return %s;", result)
}

//////////////////////////////////////////////////////////////////////////////
// SwizzleOutput: invokes the child, then swizzles the output.

type swizzleFP struct {
	FPBase
	swizzle gpu.Swizzle
}

// SwizzleOutputFP returns a fragment processor that invokes child and then swizzles its output channels; returns child
// unchanged for the identity swizzle.
func SwizzleOutputFP(child FragmentProcessor, swizzle gpu.Swizzle) FragmentProcessor {
	if child == nil {
		return nil
	}
	if swizzle == gpu.SwizzleRGBA {
		return child
	}
	fp := &swizzleFP{swizzle: swizzle}
	fp.initFP(SwizzleFragmentProcessorClassID, fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *swizzleFP) Name() string { return "Swizzle" }

func (f *swizzleFP) Clone() FragmentProcessor {
	return SwizzleOutputFP(f.ChildProcessor(0).Clone(), f.swizzle)
}

func (f *swizzleFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(f.swizzle), "swizzle")
}

func (f *swizzleFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*swizzleFP)
	return ok && f.swizzle == o.swizzle
}

func (f *swizzleFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	out := fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
	v := f.swizzle.ApplyTo([4]float32{out.R, out.G, out.B, out.A})
	return colorcore.PMColor4f{R: v[0], G: v[1], B: v[2], A: v[3]}
}

func (f *swizzleFP) onMakeProgramImpl() FPProgramImpl { return &swizzleFPImpl{} }

type swizzleFPImpl struct {
	FPImplBase
}

func (i *swizzleFPImpl) EmitCode(args *FPEmitArgs) {
	childColor := i.InvokeChild(0, args)
	swizzle := args.FP.(*swizzleFP).swizzle
	args.FragBuilder.CodeAppendf("return %s;", glslSwizzleExpr(childColor, swizzle))
}

//////////////////////////////////////////////////////////////////////////////

// MulInputByChildAlphaFP returns a fragment processor computing output = input * child.a.
func MulInputByChildAlphaFP(child FragmentProcessor) FragmentProcessor {
	if child == nil {
		return nil
	}
	return BlendFP(nil, child, raster.BlendSrcIn)
}
