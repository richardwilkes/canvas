// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The XP for the "advanced" blend modes (everything past the coefficient modes). When the driver exposes
// KHR/NV_blend_equation_advanced the blend happens in fixed function (with a blend barrier when the non-coherent
// variant is all that's available); otherwise the blend math runs in the shader against a dst read — on macOS GL 4.1
// (no advanced blend, no texture barrier) that dst read is always a dst copy, making the copy lane the primary path
// there.

package gl

import (
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// customXfermodeIsSupportedMode reports whether mode is one of the advanced blend modes.
func customXfermodeIsSupportedMode(mode raster.BlendMode) bool {
	return mode > raster.BlendScreen && mode <= raster.BlendLuminosity
}

// hwBlendEquation maps a blend mode to the corresponding advanced hardware blend equation.
func hwBlendEquation(mode raster.BlendMode) gpu.BlendEquation {
	const eqOffset = int(gpu.BlendEquationOverlay) - int(raster.BlendOverlay)
	return gpu.BlendEquation(int(mode) + eqOffset)
}

// canUseHWBlendEquation reports whether the advanced blend equation can be applied in fixed function hardware for the
// given coverage type and caps.
func canUseHWBlendEquation(equation gpu.BlendEquation, coverage AnalysisCoverage, caps *gpu.Caps) bool {
	if !caps.AdvancedBlendEquationSupport() {
		return false
	}
	if coverage == AnalysisCoverageLCD {
		return false // LCD coverage must be applied after the blend equation.
	}
	if caps.IsAdvancedBlendEquationDisabled(equation) {
		return false
	}
	return true
}

// customXP is the transfer processor for the advanced blend modes.
type customXP struct {
	XPBase
	mode            raster.BlendMode
	hwBlendEquation gpu.BlendEquation // BlendEquationIllegal when blending in the shader
}

func newCustomXPHW(mode raster.BlendMode, equation gpu.BlendEquation) *customXP {
	xp := &customXP{mode: mode, hwBlendEquation: equation}
	xp.initXP(CustomXPClassID)
	return xp
}

func newCustomXPDstRead(mode raster.BlendMode, coverage AnalysisCoverage) *customXP {
	xp := &customXP{mode: mode, hwBlendEquation: gpu.BlendEquationIllegal}
	xp.initXPFull(CustomXPClassID, true /* willReadDstColor */, coverage)
	return xp
}

func (x *customXP) Name() string { return "Custom Xfermode" }

func (x *customXP) hasHWBlendEquation() bool {
	return x.hwBlendEquation != gpu.BlendEquationIllegal
}

func (x *customXP) onAddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	if x.hasHWBlendEquation() {
		if caps.AdvBlendEqInteraction <= 0 {
			// 0 would collide with "!hasHWBlendEquation" keys.
			panic("HW blend equation without an interaction mode")
		}
		b.AddBool(true, "has hardware blend equation")
		b.Add32(uint32(caps.AdvBlendEqInteraction), "advBlendEqInteraction")
	} else {
		b.AddBool(false, "has hardware blend equation")
		b.Add32(uint32(GLSLBlendKey(x.mode)), "blendKey")
	}
}

func (x *customXP) onIsEqual(other XferProcessor) bool {
	o := other.(*customXP)
	return x.mode == o.mode && x.hwBlendEquation == o.hwBlendEquation
}

func (x *customXP) XferBarrierType(caps *gpu.Caps) XferBarrierType {
	if x.hasHWBlendEquation() && !caps.AdvancedCoherentBlendEquationSupport() {
		return XferBarrierBlend
	}
	return XferBarrierNone
}

func (x *customXP) onGetBlendInfo() gpu.BlendInfo {
	info := gpu.MakeBlendInfo()
	if x.hasHWBlendEquation() {
		info.Equation = x.hwBlendEquation
	}
	return info
}

func (x *customXP) MakeProgramImpl() XPProgramImpl {
	if x.xpBase().WillReadDstColor() == x.hasHWBlendEquation() {
		panic("custom XP must either read dst or use a HW blend equation")
	}
	return &customXPImpl{}
}

type customXPImpl struct {
	XPImplBase
	blendUniform UniformHandle
}

func (i *customXPImpl) emitOutputsForBlendState(args *XPEmitArgs) {
	xp := args.XP.(*customXP)
	if !xp.hasHWBlendEquation() {
		panic("blend-state emission requires a HW blend equation")
	}
	args.FragBuilder.EnableAdvancedBlendEquationIfNeeded(xp.hwBlendEquation)
	// Apply coverage by multiplying it into the src color before blending. This will "just work" automatically (see
	// analysisProperties).
	args.FragBuilder.CodeAppendf("%s = %s * %s;", args.OutputPrimary, args.InputCoverage,
		args.InputColor)
}

func (i *customXPImpl) emitBlendCodeForDstRead(fragBuilder *FragmentShaderBuilder, uniformHandler *UniformHandler, srcColor, srcCoverage, dstColor, outColor, outColorSecondary string, proc XferProcessor) {
	xp := proc.(*customXP)
	if xp.hasHWBlendEquation() {
		panic("dst-read emission with a HW blend equation")
	}
	blendExpr := GLSLBlendExpression(xp, uniformHandler, fragBuilder, &i.blendUniform, srcColor,
		dstColor, xp.mode)
	fragBuilder.CodeAppendf("%s = %s;", outColor, blendExpr)
	// Apply coverage.
	defaultCoverageModulation(fragBuilder, srcCoverage, dstColor, outColor, outColorSecondary,
		xp)
}

func (i *customXPImpl) onSetData(pdman *ProgramDataManager, proc XferProcessor) {
	if i.blendUniform.IsValid() {
		GLSLSetBlendModeUniformData(pdman, i.blendUniform, proc.(*customXP).mode)
	}
}

//////////////////////////////////////////////////////////////////////////////

// customXPFactory is the XP factory for one advanced blend mode, one singleton per mode.
type customXPFactory struct {
	mode            raster.BlendMode
	hwBlendEquation gpu.BlendEquation
}

func (f *customXPFactory) makeXferProcessor(_ ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, _ gpu.ClampType) XferProcessor {
	if !customXfermodeIsSupportedMode(f.mode) {
		panic("unsupported custom xfermode")
	}
	if canUseHWBlendEquation(f.hwBlendEquation, coverage, caps) {
		return newCustomXPHW(f.mode, f.hwBlendEquation)
	}
	return newCustomXPDstRead(f.mode, coverage)
}

// analysisProperties reports this factory's XP analysis flags: the advanced blends' general SVG blend equation has
// X=Y=Z=1, which can modulate alpha for coverage, so every lane is compatible with coverage-as-alpha.
func (f *customXPFactory) analysisProperties(_ ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, _ gpu.ClampType) uint32 {
	if canUseHWBlendEquation(f.hwBlendEquation, coverage, caps) {
		if caps.BlendEquationSupport == gpu.AdvancedCoherentBlendEquationSupport {
			return XPAnalysisCompatibleWithCoverageAsAlpha
		}
		return XPAnalysisCompatibleWithCoverageAsAlpha |
			XPAnalysisRequiresNonOverlappingDraws | XPAnalysisUsesNonCoherentHWBlending
	}
	return XPAnalysisCompatibleWithCoverageAsAlpha | XPAnalysisReadsDstInShader
}

// customXPFactories holds the per-mode singletons.
var customXPFactories = func() map[raster.BlendMode]*customXPFactory {
	m := make(map[raster.BlendMode]*customXPFactory)
	for mode := raster.BlendOverlay; mode <= raster.BlendLuminosity; mode++ {
		m[mode] = &customXPFactory{mode: mode, hwBlendEquation: hwBlendEquation(mode)}
	}
	return m
}()

// CustomXfermodeGet returns the XP factory for an advanced blend mode, or nil for unsupported modes.
func CustomXfermodeGet(mode raster.BlendMode) XPFactory {
	f, ok := customXPFactories[mode]
	if !ok {
		if customXfermodeIsSupportedMode(mode) {
			panic("supported mode missing a factory")
		}
		return nil
	}
	return f
}
