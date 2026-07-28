// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// XferProcessor is the processor responsible for the blend between src and dst and for applying coverage — via
// fixed-function blend state (possibly with a dual-source secondary output) or, when the mode requires it, by reading
// the dst in the shader. XP factories are stateless singletons compared by identity.

package gl

import (
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// XferBarrierType identifies what kind of GPU barrier a transfer processor requires before its draw, if any.
type XferBarrierType int32

// XferBarrierType values.
const (
	XferBarrierNone    XferBarrierType = iota // no barrier is required
	XferBarrierTexture                        // shader reads and renders to the same texture
	XferBarrierBlend                          // required by certain blend extensions
)

// XferProcessor computes the final src/dst blend and applies coverage for a draw.
type XferProcessor interface {
	Processor

	// MakeProgramImpl builds the shader-emitting implementation for this processor.
	MakeProgramImpl() XPProgramImpl
	// XferBarrierType reports what GPU barrier, if any, this processor's draw requires.
	XferBarrierType(caps *gpu.Caps) XferBarrierType

	// xpBase gives shared code access to the embedded base.
	xpBase() *XPBase

	// onAddToKey adds this processor's key contribution to b.
	onAddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder)
	// onHasSecondaryOutput reports whether this processor emits a dual-source secondary output.
	onHasSecondaryOutput() bool
	// onGetBlendInfo returns the fixed-function blend state for this processor. Unlike a mutate- in-place approach,
	// this returns the value directly so the result does not escape to the heap through this call on every executing
	// op.
	onGetBlendInfo() gpu.BlendInfo
	// onIsEqual reports whether this processor is equal to other; only called when ClassIDs match.
	onIsEqual(other XferProcessor) bool
}

// XPBase holds the shared state and default method implementations for XferProcessor implementations to embed.
type XPBase struct {
	processorBase
	willReadDstColor bool
	isLCD            bool
}

// initXP initializes the base with just a class ID (no dst read, no LCD coverage).
func (b *XPBase) initXP(classID ClassID) {
	b.classID = classID
}

// initXPFull initializes the base with a class ID and the dst-read/coverage configuration.
func (b *XPBase) initXPFull(classID ClassID, willReadDstColor bool, coverage AnalysisCoverage) {
	b.classID = classID
	b.willReadDstColor = willReadDstColor
	b.isLCD = coverage == AnalysisCoverageLCD
}

func (b *XPBase) xpBase() *XPBase { return b }

// WillReadDstColor reports whether this processor's shader needs to read the destination color.
func (b *XPBase) WillReadDstColor() bool { return b.willReadDstColor }

// IsLCD reports whether this processor was built with LCD text coverage.
func (b *XPBase) IsLCD() bool { return b.isLCD }

// XferBarrierType is the base default (none).
func (b *XPBase) XferBarrierType(*gpu.Caps) XferBarrierType { return XferBarrierNone }

// onHasSecondaryOutput is the base default (no secondary output).
func (b *XPBase) onHasSecondaryOutput() bool { return false }

// onGetBlendInfo is the base default (no blending).
func (b *XPBase) onGetBlendInfo() gpu.BlendInfo { return gpu.MakeBlendInfo() }

// xpHasSecondaryOutput reports whether xp emits a dual-source secondary output, which is never the case when the shader
// reads the dst color directly instead.
func xpHasSecondaryOutput(xp XferProcessor) bool {
	if !xp.xpBase().WillReadDstColor() {
		return xp.onHasSecondaryOutput()
	}
	return false
}

// xpGetBlendInfo returns xp's fixed-function blend state, or the no-blend default when xp reads the dst color in the
// shader instead.
func xpGetBlendInfo(xp XferProcessor) gpu.BlendInfo {
	if !xp.xpBase().WillReadDstColor() {
		return xp.onGetBlendInfo()
	}
	return gpu.MakeBlendInfo()
}

// xpAddToKey adds xp's key contribution (including the shared willReadDstColor/isLCD bits) to b.
func xpAddToKey(xp XferProcessor, caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBool(xp.xpBase().WillReadDstColor(), "willReadDstColor")
	b.AddBool(xp.xpBase().IsLCD(), "isLCD")
	xp.onAddToKey(caps, b)
}

// xpIsEqual reports whether a and b are equal transfer processors.
func xpIsEqual(a, b XferProcessor) bool {
	if a.ClassID() != b.ClassID() {
		return false
	}
	if a.xpBase().willReadDstColor != b.xpBase().willReadDstColor {
		return false
	}
	if a.xpBase().isLCD != b.xpBase().isLCD {
		return false
	}
	return a.onIsEqual(b)
}

//////////////////////////////////////////////////////////////////////////////

// XPFactory analysis properties.
const (
	// XPAnalysisReadsDstInShader: the fragment shader will require the destination color.
	XPAnalysisReadsDstInShader uint32 = 0x1
	// XPAnalysisCompatibleWithCoverageAsAlpha: the op may apply coverage as alpha and still blend correctly.
	XPAnalysisCompatibleWithCoverageAsAlpha uint32 = 0x2
	// XPAnalysisIgnoresInputColor: the color input to the XferProcessor will be ignored.
	XPAnalysisIgnoresInputColor uint32 = 0x4
	// XPAnalysisRequiresDstTexture: the destination color will be provided to the fragment processor using a texture
	// (additional information about XPAnalysisReadsDstInShader).
	XPAnalysisRequiresDstTexture uint32 = 0x10
	// XPAnalysisRequiresNonOverlappingDraws: each pixel can only be touched once during a draw (e.g. because of a dst
	// texture or an xfer barrier).
	XPAnalysisRequiresNonOverlappingDraws uint32 = 0x20
	// XPAnalysisUsesNonCoherentHWBlending: the draw uses fixed-function non-coherent advanced blends.
	XPAnalysisUsesNonCoherentHWBlending uint32 = 0x40
	// XPAnalysisUnaffectedByDstValue: the existing dst value has no effect on the final output.
	XPAnalysisUnaffectedByDstValue uint32 = 0x80
)

// XPFactory is a stateless immutable singleton, compared by identity, that builds transfer processors for a given blend
// mode.
type XPFactory interface {
	// makeXferProcessor builds the transfer processor for the given paint/coverage analysis.
	makeXferProcessor(color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps,
		clampType gpu.ClampType) XferProcessor
	// analysisProperties returns this factory's XPAnalysis* bits for the given paint/coverage analysis. It should not
	// return kRequiresDstTexture; that is inferred by the shared XPFactoryGetAnalysisProperties.
	analysisProperties(color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps,
		clampType gpu.ClampType) uint32
}

// XPFactoryFromBlendMode maps a blend mode to its factory singleton: coefficient modes map to the Porter-Duff factory
// singletons, the advanced modes to the custom-xfermode singletons.
func XPFactoryFromBlendMode(mode raster.BlendMode) XPFactory {
	if mode <= raster.BlendScreen {
		return PorterDuffXPFactory(mode)
	}
	return CustomXfermodeGet(mode)
}

// XPFactoryGetAnalysisProperties returns factory's analysis properties, with the shared dst-texture inference applied
// (a nil factory means src-over).
func XPFactoryGetAnalysisProperties(factory XPFactory, color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, clampType gpu.ClampType) uint32 {
	var result uint32
	if factory != nil {
		result = factory.analysisProperties(color, coverage, caps, clampType)
	} else {
		result = srcOverAnalysisProperties(color, coverage, caps, clampType)
	}
	if coverage == AnalysisCoverageNone {
		result |= XPAnalysisCompatibleWithCoverageAsAlpha
	}
	if result&XPAnalysisRequiresDstTexture != 0 {
		panic("analysisProperties must not set kRequiresDstTexture")
	}
	if result&XPAnalysisReadsDstInShader != 0 && !caps.ShaderCaps.DstReadInShaderSupport {
		result |= XPAnalysisRequiresDstTexture | XPAnalysisRequiresNonOverlappingDraws
	}
	return result
}

// XPFactoryMakeXferProcessor builds the transfer processor for factory (a nil factory means src-over). A nil return
// means "use the shared simple src-over XP".
func XPFactoryMakeXferProcessor(factory XPFactory, color ProcessorAnalysisColor, coverage AnalysisCoverage, caps *gpu.Caps, clampType gpu.ClampType) XferProcessor {
	if factory != nil {
		return factory.makeXferProcessor(color, coverage, caps, clampType)
	}
	return makeSrcOverXferProcessor(color, coverage, caps)
}

//////////////////////////////////////////////////////////////////////////////

// XPProgramImpl is the shader-emitting counterpart of an XferProcessor.
type XPProgramImpl interface {
	// emitOutputsForBlendState emits blending and coverage via fixed-function state; only implemented by XPs that don't
	// read the dst.
	emitOutputsForBlendState(args *XPEmitArgs)
	// emitBlendCodeForDstRead emits blend logic only; the base applies coverage.
	emitBlendCodeForDstRead(fragBuilder *FragmentShaderBuilder, uniformHandler *UniformHandler,
		srcColor, srcCoverage, dstColor, outColor, outColorSecondary string, xp XferProcessor)
	// emitWriteSwizzle emits the write-swizzle adjustment for the fragment shader outputs.
	emitWriteSwizzle(fragBuilder *FragmentShaderBuilder, swizzle gpu.Swizzle, outColor,
		outColorSecondary string)
	// onSetData uploads this program's uniform values for xp.
	onSetData(pdman *ProgramDataManager, xp XferProcessor)
}

// XPImplBase provides the default implementations for XPProgramImpl methods a subtype doesn't override.
type XPImplBase struct{}

func (XPImplBase) emitOutputsForBlendState(*XPEmitArgs) {
	panic("emitOutputsForBlendState not implemented")
}

func (XPImplBase) emitBlendCodeForDstRead(*FragmentShaderBuilder, *UniformHandler, string, string, string, string, string, XferProcessor) {
	panic("emitBlendCodeForDstRead not implemented")
}

func (XPImplBase) emitWriteSwizzle(fragBuilder *FragmentShaderBuilder, swizzle gpu.Swizzle, outColor, outColorSecondary string) {
	if swizzle != gpu.SwizzleRGBA {
		fragBuilder.CodeAppendf("%s = %s;", outColor, glslSwizzleExpr(outColor, swizzle))
		if outColorSecondary != "" {
			fragBuilder.CodeAppendf("%s = %s;", outColorSecondary,
				glslSwizzleExpr(outColorSecondary, swizzle))
		}
	}
}

func (XPImplBase) onSetData(*ProgramDataManager, XferProcessor) {}

// XPEmitArgs bundles the inputs XPProgramImpl.emitOutputsForBlendState/emitBlendCodeForDstRead need to emit shader
// code.
type XPEmitArgs struct {
	FragBuilder             *FragmentShaderBuilder
	UniformHandler          *UniformHandler
	ShaderCaps              *gpu.ShaderCaps
	XP                      XferProcessor
	InputColor              string
	InputCoverage           string
	OutputPrimary           string
	OutputSecondary         string
	DstTextureSamplerHandle SamplerHandle
	DstTextureOrigin        gpu.SurfaceOrigin
	WriteSwizzle            gpu.Swizzle
}

// xpImplEmitCode emits the shader code for impl, choosing between the fixed-function blend-state path and the dst-read
// path depending on whether args.XP reads the dst color.
func xpImplEmitCode(impl XPProgramImpl, args *XPEmitArgs) {
	if !args.XP.xpBase().WillReadDstColor() {
		adjustForLCDCoverage(args.FragBuilder, args.InputCoverage, args.XP)
		impl.emitOutputsForBlendState(args)
	} else {
		fragBuilder := args.FragBuilder
		dstColor := fragBuilder.DstColor()

		needsLocalOutColor := false

		if args.DstTextureSamplerHandle.IsValid() {
			if args.InputCoverage != "" {
				// A safety check against floating point precision errors — compare the RGB channels of coverage with <=
				// 0 and discard. This also helps batching text-draws that read from a dst copy.
				fragBuilder.CodeAppendf(
					"if (all(lessThanEqual(%s.rgb, vec3(0)))) {"+
						"    discard;"+
						"}", args.InputCoverage,
				)
			}
		} else {
			needsLocalOutColor = args.ShaderCaps.RequiresLocalOutputColorForFBFetch
		}

		outColor := "_localColorOut"
		if !needsLocalOutColor {
			outColor = args.OutputPrimary
		} else {
			fragBuilder.CodeAppendf("vec4 %s;", outColor)
		}

		impl.emitBlendCodeForDstRead(fragBuilder, args.UniformHandler, args.InputColor,
			args.InputCoverage, dstColor, outColor, args.OutputSecondary, args.XP)
		if needsLocalOutColor {
			fragBuilder.CodeAppendf("%s = %s;", args.OutputPrimary, outColor)
		}
	}

	// Swizzle the fragment shader outputs if necessary.
	impl.emitWriteSwizzle(args.FragBuilder, args.WriteSwizzle, args.OutputPrimary,
		args.OutputSecondary)
}

// adjustForLCDCoverage collapses per-channel LCD coverage to a single alpha value. Only called for LCD coverage without
// in-shader blending.
func adjustForLCDCoverage(fragBuilder *FragmentShaderBuilder, srcCoverage string, xp XferProcessor) {
	if srcCoverage != "" && xp.xpBase().IsLCD() {
		fragBuilder.CodeAppendf("%s.a = max(max(%s.r, %s.g), %s.b);",
			srcCoverage, srcCoverage, srcCoverage, srcCoverage)
	}
}

// defaultCoverageModulation emits the default per-fragment coverage blend: lerps outColor toward dstColor by (1 -
// coverage).
func defaultCoverageModulation(fragBuilder *FragmentShaderBuilder, srcCoverage, dstColor, outColor, outColorSecondary string, xp XferProcessor) {
	if srcCoverage != "" {
		if xp.xpBase().IsLCD() {
			fragBuilder.CodeAppendf("vec3 lerpRGB = mix(%s.aaa, %s.aaa, %s.rgb);",
				dstColor, outColor, srcCoverage)
		}
		fragBuilder.CodeAppendf("%s = %s * %s + (vec4(1.0) - %s) * %s;",
			outColor, srcCoverage, outColor, srcCoverage, dstColor)
		if xp.xpBase().IsLCD() {
			fragBuilder.CodeAppendf("%s.a = max(max(lerpRGB.r, lerpRGB.b), lerpRGB.g);", outColor)
		}
	}
	_ = outColorSecondary
}
