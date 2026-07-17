// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The mesh-draw-op support layer: the DrawOp interface below, a set of static helper functions for mesh-draw ops, and
// SimpleMeshDrawOpHelper, which reduces the boilerplate for ops that construct a single pipeline for a uniform
// primitive color and a paint. The stencil-carrying variant is folded into one helper carrying the stencil settings
// (the no-stencil form passes UnusedStencilSettings); NewProcessorSetFromPaint builds the processor set for non-trivial
// paints; prePrepare lanes are dropped with DDLs.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// ClipResult reports the effect ClipToShape had on an op's clip.
type ClipResult int32

// ClipResult values.
const (
	// ClipResultFail: no clip was applied.
	ClipResultFail ClipResult = iota
	// ClipResultClippedGeometrically: the clip was applied to the op's actual geometry. The clip stack is free to
	// disable the scissor test.
	ClipResultClippedGeometrically
	// ClipResultClippedInShader: the clip was applied via shader coverage. The clip stack will still use a scissor test
	// in order to reduce overdraw of transparent pixels.
	ClipResultClippedInShader
	// ClipResultClippedOut: the op can be thrown out entirely.
	ClipResultClippedOut
)

// DrawOp extends Op with the additional methods a mesh-drawing op must implement.
type DrawOp interface {
	Op
	// Finalize must be called exactly once, after the applied clip has been computed and before program information is
	// created, to finalize the processor set.
	Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis
	// UsesMSAA is called before setting up the AppliedClip and before Finalize.
	UsesMSAA() bool
	// UsesStencil is called after Finalize.
	UsesStencil() bool
	// ClipToShape is called while the clip is being computed, before Finalize and before any attempts to combine with
	// other ops, giving the op a chance to clip its own geometry (only FillRRectOp implements it; everything else
	// fails).
	ClipToShape(op raster.ClipOp, clipMatrix *geom.Matrix, shape *Shape, aa gpu.AA) ClipResult
}

// CanUpgradeAAOnMerge reports whether two ops with the given AA types can merge, one upgrading its AA type to match the
// other.
func CanUpgradeAAOnMerge(aa1, aa2 gpu.AAType) bool {
	return (aa1 == gpu.AATypeNone && aa2 == gpu.AATypeCoverage) ||
		(aa1 == gpu.AATypeCoverage && aa2 == gpu.AATypeNone)
}

// CombinedQuadCountWillOverflow reports whether merging would push the combined quad count past the index-buffer-sized
// limit for the resulting AA type.
func CombinedQuadCountWillOverflow(aaType gpu.AAType, willBeUpgradedToAA bool, combinedQuadCount int) bool {
	willBeAA := aaType == gpu.AATypeCoverage || willBeUpgradedToAA
	if willBeAA {
		return combinedQuadCount > MaxNumAAQuads
	}
	return combinedQuadCount > MaxNumNonAAQuads
}

// SimpleMeshDrawOpHelper holds optional processor-set ownership plus the AA type, pipeline input flags, and user
// stencil settings shared by simple mesh draw ops.
type SimpleMeshDrawOpHelper struct {
	processors                    *ProcessorSet
	stencilSettings               *UserStencilSettings
	aaType                        gpu.AAType
	pipelineFlags                 uint8
	usesLocalCoords               bool
	compatibleWithCoverageAsAlpha bool
	didAnalysis                   bool
}

// InitSimpleMeshDrawOpHelper initializes the helper. processors is nil for trivial paints; stencilSettings nil means
// unused.
func (h *SimpleMeshDrawOpHelper) InitSimpleMeshDrawOpHelper(processors *ProcessorSet, aaType gpu.AAType, stencilSettings *UserStencilSettings, inputFlags uint8) {
	if stencilSettings == nil {
		stencilSettings = UnusedStencilSettings()
	}
	h.processors = processors
	h.stencilSettings = stencilSettings
	h.pipelineFlags = inputFlags
	h.aaType = aaType
}

// IsTrivial reports whether the helper has no processor set (a plain solid-color draw).
func (h *SimpleMeshDrawOpHelper) IsTrivial() bool { return h.processors == nil }

// UsesLocalCoords reports whether the finalized processor set reads local coordinates. Must not be called before
// FinalizeProcessors.
func (h *SimpleMeshDrawOpHelper) UsesLocalCoords() bool {
	if !h.didAnalysis {
		panic("usesLocalCoords before analysis")
	}
	return h.usesLocalCoords
}

// CompatibleWithCoverageAsAlpha reports whether the finalized processor set is compatible with treating its coverage as
// a straight alpha multiply.
func (h *SimpleMeshDrawOpHelper) CompatibleWithCoverageAsAlpha() bool {
	return h.compatibleWithCoverageAsAlpha
}

// VisitProxies calls fn for every proxy the processor set references.
func (h *SimpleMeshDrawOpHelper) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if h.processors != nil {
		h.processors.VisitProxies(fn)
	}
}

// AAType returns the helper's AA type.
func (h *SimpleMeshDrawOpHelper) AAType() gpu.AAType { return h.aaType }

// SetAAType overrides the helper's AA type.
func (h *SimpleMeshDrawOpHelper) SetAAType(aaType gpu.AAType) { h.aaType = aaType }

// UsesMSAA reports whether the AA type requires hardware MSAA.
func (h *SimpleMeshDrawOpHelper) UsesMSAA() bool { return h.aaType.IsHW() }

// UsesStencil reports whether the helper carries real (not unused) stencil settings.
func (h *SimpleMeshDrawOpHelper) UsesStencil() bool {
	return h.stencilSettings != UnusedStencilSettings()
}

// StencilSettings returns the helper's user stencil settings.
func (h *SimpleMeshDrawOpHelper) StencilSettings() *UserStencilSettings {
	return h.stencilSettings
}

// PipelineFlags returns the pipeline input flags the helper was constructed with.
func (h *SimpleMeshDrawOpHelper) PipelineFlags() uint8 { return h.pipelineFlags }

// IsCompatible reports whether two helpers can be merged into one draw. ignoreAAType should be true if the op already
// knows the AA settings are acceptable.
func (h *SimpleMeshDrawOpHelper) IsCompatible(that *SimpleMeshDrawOpHelper, ignoreAAType bool) bool {
	if (h.processors != nil) != (that.processors != nil) {
		return false
	}
	if h.processors != nil && !h.processors.Equal(that.processors) {
		return false
	}
	if ignoreAAType {
		// If we're ignoring AA it should be because we already know they are the same or that they are different but
		// compatible (i.e., one is AA and the other is none).
		if h.aaType != that.aaType && !CanUpgradeAAOnMerge(h.aaType, that.aaType) {
			panic("ignoreAAType with incompatible AA types")
		}
	}
	result := h.pipelineFlags == that.pipelineFlags &&
		(ignoreAAType || h.aaType == that.aaType)
	if result {
		if h.compatibleWithCoverageAsAlpha != that.compatibleWithCoverageAsAlpha {
			panic("compatible helpers must agree on coverage-as-alpha")
		}
		if h.usesLocalCoords != that.usesLocalCoords {
			panic("compatible helpers must agree on local coords")
		}
	}
	// The WithStencil extension: stencil settings must match.
	return result && h.stencilSettings == that.stencilSettings
}

// FinalizeProcessors finalizes the processor set and determines whether the destination must be provided to the
// fragment shader as a texture for blending.
func (h *SimpleMeshDrawOpHelper) FinalizeProcessors(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType, geometryCoverage AnalysisCoverage, geometryColor *ProcessorAnalysisColor) ProcessorAnalysis {
	h.didAnalysis = true
	var analysis ProcessorAnalysis
	if h.processors != nil {
		coverage := geometryCoverage
		if coverage == AnalysisCoverageNone && clip != nil &&
			clip.HasCoverageFragmentProcessor() {
			coverage = AnalysisCoverageSingleChannel
		}
		var overrideColor colorcore.PMColor4f
		analysis, overrideColor = h.processors.Finalize(*geometryColor, coverage, clip,
			h.stencilSettings, caps, clampType)
		if analysis.InputColorIsOverridden() {
			*geometryColor = AnalysisColorConstant(overrideColor)
		}
	} else {
		analysis = EmptyProcessorSetAnalysis()
	}
	h.usesLocalCoords = analysis.UsesLocalCoords()
	h.compatibleWithCoverageAsAlpha = analysis.IsCompatibleWithCoverageAsAlpha()
	return analysis
}

// FinalizeProcessorsWithColor is the variant used by ops with a constant-color geometry processor output: after return,
// if geometryColor changed the op must override its geometry processor's color output with the new color. wideColor may
// be nil.
func (h *SimpleMeshDrawOpHelper) FinalizeProcessorsWithColor(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType, geometryCoverage AnalysisCoverage, geometryColor *colorcore.PMColor4f, wideColor *bool) ProcessorAnalysis {
	color := AnalysisColorConstant(*geometryColor)
	result := h.FinalizeProcessors(caps, clip, clampType, geometryCoverage, &color)
	if c, isConstant := color.IsConstant(); isConstant {
		*geometryColor = c
	}
	if wideColor != nil {
		*wideColor = !pmFitsInBytes(*geometryColor)
	}
	return result
}

// detachProcessorSet transfers ownership of the helper's processor set to the caller, leaving the helper with none (or
// a fresh empty set if it had none to begin with).
func (h *SimpleMeshDrawOpHelper) detachProcessorSet() *ProcessorSet {
	if h.processors != nil {
		p := h.processors
		h.processors = nil
		return p
	}
	return MakeEmptyProcessorSet()
}

// CreatePipeline builds a pipeline from the helper's detached processor set. Used by ops that share one pipeline across
// several program infos (AAHairlineOp); most ops use CreateProgramInfoWithStencil instead.
func (h *SimpleMeshDrawOpHelper) CreatePipeline(caps *Caps, writeSwizzle gpu.Swizzle, appliedClip *AppliedClip, dstProxyView *DstProxyView) *Pipeline {
	return NewPipeline(&PipelineInitArgs{
		InputFlags:   h.pipelineFlags,
		Caps:         caps,
		DstProxyView: *dstProxyView,
		WriteSwizzle: writeSwizzle,
	}, h.detachProcessorSet(), appliedClip)
}

// CreateProgramInfoWithStencil builds the pipeline from the helper's detached processor set and wraps it in a
// ProgramInfo.
func (h *SimpleMeshDrawOpHelper) CreateProgramInfoWithStencil(caps *Caps, writeView *SurfaceProxyView, usesMSAASurface bool, appliedClip *AppliedClip, dstProxyView *DstProxyView, gp GeometryProcessor, primType gpu.PrimitiveType, renderPassXferBarriers gpu.XferBarrierFlags, colorLoadOp gpu.LoadOp) *ProgramInfo {
	pipeline := NewPipeline(&PipelineInitArgs{
		InputFlags:   h.pipelineFlags,
		Caps:         caps,
		DstProxyView: *dstProxyView,
		WriteSwizzle: writeView.Swizzle(),
	}, h.detachProcessorSet(), appliedClip)
	return NewProgramInfo(caps, writeView, usesMSAASurface, pipeline, h.stencilSettings, gp,
		primType, renderPassXferBarriers, colorLoadOp)
}
