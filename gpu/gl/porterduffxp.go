// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The coefficient-mode transfer processors driven by gpu.BlendFormula, the shader fallback (shaderPDXP) for dst-read
// blending, the LCD constant-color trick XP, and the color-write-disabled XP used by stencil passes.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

//////////////////////////////////////////////////////////////////////////////
// PorterDuffXferProcessor: fixed-function blending from a BlendFormula.

type porterDuffXP struct {
	XPBase
	blendFormula gpu.BlendFormula
}

func newPorterDuffXP(formula gpu.BlendFormula, coverage AnalysisCoverage) *porterDuffXP {
	xp := &porterDuffXP{blendFormula: formula}
	xp.initXPFull(PorterDuffXferProcessorClassID, false /* willReadDstColor */, coverage)
	return xp
}

func (x *porterDuffXP) Name() string { return "Porter Duff" }

func (x *porterDuffXP) MakeProgramImpl() XPProgramImpl { return &porterDuffXPImpl{} }

func (x *porterDuffXP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	if gpu.BlendFormulaOutputTypeLast >= 8 {
		panic("output type must fit in 3 bits")
	}
	b.Add32(uint32(x.blendFormula.PrimaryOutput())|uint32(x.blendFormula.SecondaryOutput())<<3,
		"blendFormulaOutputs")
}

func (x *porterDuffXP) onHasSecondaryOutput() bool { return x.blendFormula.HasSecondaryOutput() }

func (x *porterDuffXP) onGetBlendInfo() gpu.BlendInfo {
	blendInfo := gpu.MakeBlendInfo()
	blendInfo.Equation = x.blendFormula.Equation()
	blendInfo.SrcBlend = x.blendFormula.SrcCoeff()
	blendInfo.DstBlend = x.blendFormula.DstCoeff()
	blendInfo.WritesColor = x.blendFormula.ModifiesDst()
	return blendInfo
}

func (x *porterDuffXP) onIsEqual(other XferProcessor) bool {
	o, ok := other.(*porterDuffXP)
	return ok && x.blendFormula == o.blendFormula
}

type porterDuffXPImpl struct {
	XPImplBase
}

// appendColorOutput emits the fragment-shader assignment for output given outputType.
func appendColorOutput(fragBuilder *FragmentShaderBuilder, outputType gpu.BlendFormulaOutputType, output, inColor, inCoverage string) {
	if inCoverage == "" || inColor == "" {
		panic("color and coverage expressions are required")
	}
	switch outputType {
	case gpu.BlendFormulaOutputNone:
		fragBuilder.CodeAppendf("%s = vec4(0.0);", output)
	case gpu.BlendFormulaOutputCoverage:
		fragBuilder.CodeAppendf("%s = %s;", output, inCoverage)
	case gpu.BlendFormulaOutputModulate:
		fragBuilder.CodeAppendf("%s = %s * %s;", output, inColor, inCoverage)
	case gpu.BlendFormulaOutputSAModulate:
		fragBuilder.CodeAppendf("%s = %s.a * %s;", output, inColor, inCoverage)
	case gpu.BlendFormulaOutputISAModulate:
		fragBuilder.CodeAppendf("%s = (1.0 - %s.a) * %s;", output, inColor, inCoverage)
	case gpu.BlendFormulaOutputISCModulate:
		fragBuilder.CodeAppendf("%s = (vec4(1.0) - %s) * %s;", output, inColor, inCoverage)
	default:
		panic("unsupported output type")
	}
}

func (i *porterDuffXPImpl) emitOutputsForBlendState(args *XPEmitArgs) {
	xp := args.XP.(*porterDuffXP)
	if xp.blendFormula.HasSecondaryOutput() {
		appendColorOutput(args.FragBuilder, xp.blendFormula.SecondaryOutput(),
			args.OutputSecondary, args.InputColor, args.InputCoverage)
	}
	appendColorOutput(args.FragBuilder, xp.blendFormula.PrimaryOutput(),
		args.OutputPrimary, args.InputColor, args.InputCoverage)
}

//////////////////////////////////////////////////////////////////////////////
// ShaderPDXferProcessor: dst-read shader blending for a coefficient mode.

type shaderPDXP struct {
	XPBase
	mode raster.BlendMode
}

func newShaderPDXP(mode raster.BlendMode, coverage AnalysisCoverage) *shaderPDXP {
	xp := &shaderPDXP{mode: mode}
	xp.initXPFull(ShaderPDXferProcessorClassID, true /* willReadDstColor */, coverage)
	return xp
}

func (x *shaderPDXP) Name() string { return "Porter Duff Shader" }

func (x *shaderPDXP) MakeProgramImpl() XPProgramImpl { return &shaderPDXPImpl{} }

func (x *shaderPDXP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(GLSLBlendKey(x.mode)), "blendKey")
}

func (x *shaderPDXP) onIsEqual(other XferProcessor) bool {
	o, ok := other.(*shaderPDXP)
	return ok && x.mode == o.mode
}

type shaderPDXPImpl struct {
	XPImplBase
	blendUniform UniformHandle
}

func (i *shaderPDXPImpl) emitBlendCodeForDstRead(fragBuilder *FragmentShaderBuilder, uniformHandler *UniformHandler, srcColor, srcCoverage, dstColor, outColor, outColorSecondary string, xp XferProcessor) {
	pdxp := xp.(*shaderPDXP)
	blendExpr := GLSLBlendExpression(xp, uniformHandler, fragBuilder, &i.blendUniform,
		srcColor, dstColor, pdxp.mode)
	fragBuilder.CodeAppendf("%s = %s;", outColor, blendExpr)

	// Apply coverage.
	defaultCoverageModulation(fragBuilder, srcCoverage, dstColor, outColor, outColorSecondary, xp)
}

func (i *shaderPDXPImpl) onSetData(pdman *ProgramDataManager, xp XferProcessor) {
	if i.blendUniform.IsValid() {
		GLSLSetBlendModeUniformData(pdman, i.blendUniform, xp.(*shaderPDXP).mode)
	}
}

//////////////////////////////////////////////////////////////////////////////
// PDLCDXferProcessor: the constant-color src-over trick for LCD text without dual-source blending or in-shader dst
// reads.

type pdLCDXP struct {
	XPBase
	blendConstant colorcore.PMColor4f
	alpha         float32
}

// makePDLCDXP builds the constant-color LCD trick transfer processor; returns nil when the trick does not apply.
func makePDLCDXP(mode raster.BlendMode, color ProcessorAnalysisColor) XferProcessor {
	if mode != raster.BlendSrcOver {
		return nil
	}
	blendConstantPM, ok := color.IsConstant()
	if !ok {
		return nil
	}
	blendConstantUPM := blendConstantPM.Unpremul()
	alpha := blendConstantUPM.A
	xp := &pdLCDXP{
		blendConstant: colorcore.PMColor4f{
			R: blendConstantUPM.R, G: blendConstantUPM.G, B: blendConstantUPM.B, A: 1,
		},
		alpha: alpha,
	}
	xp.initXPFull(PDLCDXferProcessorClassID, false /* willReadDstColor */, AnalysisCoverageLCD)
	return xp
}

func (x *pdLCDXP) Name() string { return "Porter Duff LCD" }

func (x *pdLCDXP) MakeProgramImpl() XPProgramImpl {
	return &pdLCDXPImpl{lastAlpha: float32(math.NaN())}
}

func (x *pdLCDXP) onAddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

func (x *pdLCDXP) onGetBlendInfo() gpu.BlendInfo {
	blendInfo := gpu.MakeBlendInfo()
	blendInfo.SrcBlend = gpu.BlendCoeffConstC
	// The trick needs constColor*src + (1 - src)*dst: src carries the per-channel coverage times alpha, so the dst has
	// to be attenuated by coverage (one minus src color), not by the text color in the blend constant.
	blendInfo.DstBlend = gpu.BlendCoeffISC
	blendInfo.BlendConstant = [4]float32{
		x.blendConstant.R, x.blendConstant.G, x.blendConstant.B, x.blendConstant.A,
	}
	return blendInfo
}

func (x *pdLCDXP) onIsEqual(other XferProcessor) bool {
	o, ok := other.(*pdLCDXP)
	return ok && x.blendConstant == o.blendConstant && x.alpha == o.alpha
}

type pdLCDXPImpl struct {
	XPImplBase
	alphaUniform UniformHandle
	lastAlpha    float32
}

func (i *pdLCDXPImpl) emitOutputsForBlendState(args *XPEmitArgs) {
	var alpha string
	i.alphaUniform, alpha = args.UniformHandler.AddUniform(nil, ShaderFlagFragment, GLSLTypeHalf,
		"alpha")
	// Force the primary output to be alpha * coverage: there are no color stages (or this XP would not have been
	// created) and the RGB channels of the op's input color are baked into the blend constant.
	if args.InputCoverage == "" {
		panic("PDLCD requires coverage")
	}
	args.FragBuilder.CodeAppendf("%s = %s * %s;", args.OutputPrimary, alpha, args.InputCoverage)
}

func (i *pdLCDXPImpl) onSetData(pdman *ProgramDataManager, xp XferProcessor) {
	alpha := xp.(*pdLCDXP).alpha
	if i.lastAlpha != alpha {
		pdman.Set1f(i.alphaUniform, alpha)
		i.lastAlpha = alpha
	}
}

//////////////////////////////////////////////////////////////////////////////
// PorterDuffXPFactory.

type porterDuffXPFactory struct {
	mode raster.BlendMode
}

// The static factory singletons, one per coefficient blend mode.
var porterDuffXPFactories = func() [int(raster.BlendScreen) + 1]*porterDuffXPFactory {
	var fs [int(raster.BlendScreen) + 1]*porterDuffXPFactory
	for i := range fs {
		fs[i] = &porterDuffXPFactory{mode: raster.BlendMode(i)}
	}
	return fs
}()

// PorterDuffXPFactory returns the shared XPFactory singleton for mode.
func PorterDuffXPFactory(mode raster.BlendMode) XPFactory {
	if mode > raster.BlendScreen {
		panic("not a coefficient blend mode")
	}
	return porterDuffXPFactories[mode]
}

func (f *porterDuffXPFactory) makeXferProcessor(color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, clampType gpu.ClampType) XferProcessor {
	isLCD := coverage == AnalysisCoverageLCD
	// See the comment in makeSrcOverXferProcessor about color.isOpaque here.
	if isLCD && f.mode == raster.BlendSrcOver && analysisColorIsConstant(color) &&
		!caps.ShaderCaps.DualSourceBlendingSupport && !caps.ShaderCaps.DstReadInShaderSupport {
		// Without dual-source blending or in-shader dst reads we use the blend-constant trick for SrcOver LCD text
		// instead of doing a dst copy.
		return makePDLCDXP(f.mode, color)
	}
	var blendFormula gpu.BlendFormula
	switch {
	case isLCD:
		blendFormula = gpu.GetLCDBlendFormula(f.mode)
	case f.mode == raster.BlendSrcOver && color.IsOpaque() &&
		coverage == AnalysisCoverageNone && caps.ShouldCollapseSrcOverToSrcWhenAble:
		blendFormula = gpu.GetBlendFormula(true, false, raster.BlendSrc)
	default:
		blendFormula = gpu.GetBlendFormula(color.IsOpaque(), coverage != AnalysisCoverageNone,
			f.mode)
	}

	// The Plus blend mode always saturates its output, so it requires shader-based blending when pixels aren't
	// guaranteed to automatically be normalized (i.e. any floating point config).
	if (blendFormula.HasSecondaryOutput() && !caps.ShaderCaps.DualSourceBlendingSupport) ||
		(isLCD && f.mode != raster.BlendSrcOver) ||
		(clampType != gpu.ClampTypeAuto && f.mode == raster.BlendPlus) {
		return newShaderPDXP(f.mode, coverage)
	}
	return newPorterDuffXP(blendFormula, coverage)
}

func (f *porterDuffXPFactory) analysisProperties(color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, clampType gpu.ClampType) uint32 {
	return pdAnalysisProperties(color, coverage, caps, clampType, f.mode)
}

// analysisColorIsConstant is a small readability helper for the two-value IsConstant form.
func analysisColorIsConstant(color ProcessorAnalysisColor) bool {
	_, ok := color.IsConstant()
	return ok
}

// pdAnalysisProperties computes the analysis property bits for a coefficient-mode blend.
func pdAnalysisProperties(color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, clampType gpu.ClampType, mode raster.BlendMode) uint32 {
	var props uint32
	hasCoverage := coverage != AnalysisCoverageNone
	isLCD := coverage == AnalysisCoverageLCD
	var formula gpu.BlendFormula
	if isLCD {
		formula = gpu.GetLCDBlendFormula(mode)
	} else {
		formula = gpu.GetBlendFormula(color.IsOpaque(), hasCoverage, mode)
	}

	if formula.CanTweakAlphaForCoverage() && !isLCD {
		props |= XPAnalysisCompatibleWithCoverageAsAlpha
	}

	if isLCD {
		// See the comment in makeSrcOverXferProcessor about color.isOpaque here.
		if mode == raster.BlendSrcOver && analysisColorIsConstant(color) &&
			!caps.ShaderCaps.DualSourceBlendingSupport &&
			!caps.ShaderCaps.DstReadInShaderSupport {
			props |= XPAnalysisIgnoresInputColor
		} else if mode != raster.BlendSrcOver ||
			(formula.HasSecondaryOutput() && !caps.ShaderCaps.DualSourceBlendingSupport) {
			// For LCD blending, if the color is not opaque we must read the dst in shader even with dual source
			// blending. The opaqueness check must be done after blending, so for simplicity only src-over avoids the
			// dst-read path (the check itself is disabled). We also fall into the dst-read case for src-over without
			// dual source blending.
			props |= XPAnalysisReadsDstInShader
		}
	} else {
		// With dual-source blending we never need the destination color in the shader.
		if !caps.ShaderCaps.DualSourceBlendingSupport && formula.HasSecondaryOutput() {
			props |= XPAnalysisReadsDstInShader
		}
	}

	if clampType != gpu.ClampTypeAuto && mode == raster.BlendPlus {
		props |= XPAnalysisReadsDstInShader
	}

	if !formula.ModifiesDst() || !formula.UsesInputColor() {
		props |= XPAnalysisIgnoresInputColor
	}
	if formula.UnaffectedByDst() ||
		(formula.UnaffectedByDstIfOpaque() && color.IsOpaque() && !hasCoverage) {
		props |= XPAnalysisUnaffectedByDstValue
	}
	return props
}

// simpleSrcOverXP is the shared global XP for the common non-coverage, non-opaque src-over case.
var simpleSrcOverXP = newPorterDuffXP(
	gpu.GetBlendFormula(false /* isOpaque */, false /* hasCoverage */, raster.BlendSrcOver),
	AnalysisCoverageSingleChannel,
)

// SimpleSrcOverXP returns the shared src-over transfer processor used when no special-casing applies.
func SimpleSrcOverXP() XferProcessor { return simpleSrcOverXP }

// makeSrcOverXferProcessor builds a specialized transfer processor for the common src-over case, or returns nil to mean
// "use SimpleSrcOverXP" — Pipeline's nil xferProcessor field keeps that same meaning.
func makeSrcOverXferProcessor(color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps) XferProcessor {
	if coverage != AnalysisCoverageLCD {
		if color.IsOpaque() && coverage == AnalysisCoverageNone &&
			caps.ShouldCollapseSrcOverToSrcWhenAble {
			return newPorterDuffXP(gpu.GetBlendFormula(true, false, raster.BlendSrc), coverage)
		}
		// nil means "use the shared simple src-over XP".
		return nil
	}

	// LCD: currently the stack requires that the dst is opaque or that opaqueness doesn't matter, so the src opaqueness
	// check is disabled.
	if analysisColorIsConstant(color) && !caps.ShaderCaps.DualSourceBlendingSupport &&
		!caps.ShaderCaps.DstReadInShaderSupport {
		return makePDLCDXP(raster.BlendSrcOver, color)
	}

	blendFormula := gpu.GetLCDBlendFormula(raster.BlendSrcOver)
	if blendFormula.HasSecondaryOutput() && !caps.ShaderCaps.DualSourceBlendingSupport {
		return newShaderPDXP(raster.BlendSrcOver, coverage)
	}
	return newPorterDuffXP(blendFormula, coverage)
}

// MakeNoCoverageXP builds a transfer processor for mode assuming no coverage and a non-opaque source.
func MakeNoCoverageXP(mode raster.BlendMode) XferProcessor {
	return newPorterDuffXP(gpu.GetBlendFormula(false, false, mode), AnalysisCoverageNone)
}

// srcOverAnalysisProperties computes the analysis property bits for the src-over blend mode.
func srcOverAnalysisProperties(color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, clampType gpu.ClampType) uint32 {
	return pdAnalysisProperties(color, coverage, caps, clampType, raster.BlendSrcOver)
}

//////////////////////////////////////////////////////////////////////////////
// DisableColorXP: disables color writing (color and coverage are ignored, no blending occurs). Useful for stenciling.

type disableColorXP struct {
	XPBase
}

var disableColorXPSingleton = func() *disableColorXP {
	xp := &disableColorXP{}
	xp.initXP(DisableColorXPClassID)
	return xp
}()

func (x *disableColorXP) Name() string { return "Disable Color" }

func (x *disableColorXP) MakeProgramImpl() XPProgramImpl { return &disableColorXPImpl{} }

func (x *disableColorXP) onAddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

func (x *disableColorXP) onGetBlendInfo() gpu.BlendInfo {
	blendInfo := gpu.MakeBlendInfo()
	blendInfo.WritesColor = false
	return blendInfo
}

func (x *disableColorXP) onIsEqual(XferProcessor) bool { return true }

type disableColorXPImpl struct {
	XPImplBase
}

func (i *disableColorXPImpl) emitOutputsForBlendState(args *XPEmitArgs) {
	if args.ShaderCaps.MustWriteToFragColor {
		// Mobile-driver workaround; unreachable with the desktop trim but kept for fidelity.
		args.FragBuilder.CodeAppendf("%s = vec4(0);", args.OutputPrimary)
	}
}

func (i *disableColorXPImpl) emitWriteSwizzle(*FragmentShaderBuilder, gpu.Swizzle, string, string) {
	// Don't write any swizzling, so the final shader does not output a color.
}

//////////////////////////////////////////////////////////////////////////////

type disableColorXPFactory struct{}

var disableColorXPFactorySingleton = &disableColorXPFactory{}

// DisableColorXPFactory returns the shared XPFactory that disables color writes (used for stencil-only passes).
func DisableColorXPFactory() XPFactory { return disableColorXPFactorySingleton }

func (f *disableColorXPFactory) makeXferProcessor(ProcessorAnalysisColor, AnalysisCoverage, *gpu.Caps, gpu.ClampType) XferProcessor {
	return disableColorXPSingleton
}

func (f *disableColorXPFactory) analysisProperties(ProcessorAnalysisColor, AnalysisCoverage, *gpu.Caps, gpu.ClampType) uint32 {
	return XPAnalysisCompatibleWithCoverageAsAlpha | XPAnalysisIgnoresInputColor
}
