// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The paint's processors while an op is being finalized — analysis runs over the color/coverage FPs and the XP factory,
// may eliminate a constant-folded color FP, and produces the XP.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// analysisInputColorType classifies how an analyzed processor set treats its input color.
type analysisInputColorType uint8

const (
	analysisInputColorOriginal analysisInputColorType = iota
	analysisInputColorOverridden
	analysisInputColorIgnored
)

// ProcessorAnalysis is the result of finalizing a ProcessorSet: what the analysis learned about the color/coverage
// processors and transfer processor, as plain fields.
type ProcessorAnalysis struct {
	usesLocalCoords               bool
	compatibleWithCoverageAsAlpha bool
	requiresDstTexture            bool
	requiresNonOverlappingDraws   bool
	hasColorFragmentProcessor     bool
	isInitialized                 bool
	usesNonCoherentHWBlending     bool
	unaffectedByDstValue          bool
	inputColorType                analysisInputColorType
}

// EmptyProcessorSetAnalysis returns the analysis for a processor set with no processors at all.
func EmptyProcessorSetAnalysis() ProcessorAnalysis {
	return ProcessorAnalysis{
		compatibleWithCoverageAsAlpha: true,
		isInitialized:                 true,
	}
}

// IsInitialized reports whether the analysis has been populated by Finalize.
func (a ProcessorAnalysis) IsInitialized() bool { return a.isInitialized }

// UsesLocalCoords reports whether any analyzed processor samples local (pre-transform) coordinates.
func (a ProcessorAnalysis) UsesLocalCoords() bool { return a.usesLocalCoords }

// RequiresDstTexture reports whether the transfer processor needs a copy of the destination.
func (a ProcessorAnalysis) RequiresDstTexture() bool { return a.requiresDstTexture }

// RequiresNonOverlappingDraws reports whether overlapping ops must be kept from chaining or combining: a barrier or a
// fresh destination-texture readback is required between draws.
func (a ProcessorAnalysis) RequiresNonOverlappingDraws() bool {
	return a.requiresNonOverlappingDraws
}

// IsCompatibleWithCoverageAsAlpha reports whether coverage can be folded in as a plain alpha multiplier rather than
// needing full coverage-aware blending.
func (a ProcessorAnalysis) IsCompatibleWithCoverageAsAlpha() bool {
	return a.compatibleWithCoverageAsAlpha
}

// HasColorFragmentProcessor reports whether the analyzed set still carries a color FP after elimination.
func (a ProcessorAnalysis) HasColorFragmentProcessor() bool {
	return a.hasColorFragmentProcessor
}

// InputColorIsIgnored reports whether the transfer processor ignores the input color entirely.
func (a ProcessorAnalysis) InputColorIsIgnored() bool {
	return a.inputColorType == analysisInputColorIgnored
}

// InputColorIsOverridden reports whether analysis replaced the input color with a constant-folded value (see Finalize's
// returned color).
func (a ProcessorAnalysis) InputColorIsOverridden() bool {
	return a.inputColorType == analysisInputColorOverridden
}

// UsesNonCoherentHWBlending reports whether the transfer processor relies on non-coherent hardware blending extensions.
func (a ProcessorAnalysis) UsesNonCoherentHWBlending() bool { return a.usesNonCoherentHWBlending }

// UnaffectedByDstValue reports whether the blend result is independent of the destination pixel.
func (a ProcessorAnalysis) UnaffectedByDstValue() bool { return a.unaffectedByDstValue }

// ProcessorSet holds a draw's color and coverage fragment processors plus its transfer processor factory. Before
// finalize the transfer side holds the factory; after finalize it holds the processor (possibly nil, meaning the shared
// simple src-over transfer processor).
type ProcessorSet struct {
	colorFragmentProcessor    FragmentProcessor
	coverageFragmentProcessor FragmentProcessor
	xpFactory                 XPFactory
	xferProcessor             XferProcessor
	finalized                 bool
}

// NewProcessorSetFromPaint moves the paint's processors and transfer-processor factory into a new ProcessorSet. The
// paint must not be reused.
func NewProcessorSetFromPaint(paint *Paint) *ProcessorSet {
	s := &ProcessorSet{
		colorFragmentProcessor:    paint.colorFragmentProcessor,
		coverageFragmentProcessor: paint.coverageFragmentProcessor,
		xpFactory:                 paint.xpFactory,
	}
	paint.colorFragmentProcessor = nil
	paint.coverageFragmentProcessor = nil
	return s
}

// NewProcessorSetFromBlendMode builds a processor set with no color or coverage FP, using the transfer-processor
// factory for the given blend mode.
func NewProcessorSetFromBlendMode(mode raster.BlendMode) *ProcessorSet {
	return &ProcessorSet{xpFactory: XPFactoryFromBlendMode(mode)}
}

// NewProcessorSetFromColorFP builds a processor set whose only content is the given color FP.
func NewProcessorSetFromColorFP(colorFP FragmentProcessor) *ProcessorSet {
	if colorFP == nil {
		panic("color FP required")
	}
	return &ProcessorSet{colorFragmentProcessor: colorFP}
}

// MakeEmptyProcessorSet returns an already-finalized processor set with no processors at all.
func MakeEmptyProcessorSet() *ProcessorSet {
	return &ProcessorSet{finalized: true}
}

// HasColorFragmentProcessor reports whether the set carries a color FP.
func (s *ProcessorSet) HasColorFragmentProcessor() bool {
	return s.colorFragmentProcessor != nil
}

// HasCoverageFragmentProcessor reports whether the set carries a coverage FP.
func (s *ProcessorSet) HasCoverageFragmentProcessor() bool {
	return s.coverageFragmentProcessor != nil
}

// ColorFragmentProcessor returns the set's color FP, or nil.
func (s *ProcessorSet) ColorFragmentProcessor() FragmentProcessor {
	return s.colorFragmentProcessor
}

// CoverageFragmentProcessor returns the set's coverage FP, or nil.
func (s *ProcessorSet) CoverageFragmentProcessor() FragmentProcessor {
	return s.coverageFragmentProcessor
}

// XferProcessor returns the finalized transfer processor (only legal after Finalize; nil means simple src-over).
func (s *ProcessorSet) XferProcessor() XferProcessor {
	if !s.finalized {
		panic("processor set not finalized")
	}
	return s.xferProcessor
}

// IsFinalized reports whether Finalize has been called.
func (s *ProcessorSet) IsFinalized() bool { return s.finalized }

// detachColorFragmentProcessor removes and returns the set's color FP, leaving it unset.
func (s *ProcessorSet) detachColorFragmentProcessor() FragmentProcessor {
	fp := s.colorFragmentProcessor
	s.colorFragmentProcessor = nil
	return fp
}

// detachCoverageFragmentProcessor removes and returns the set's coverage FP, leaving it unset.
func (s *ProcessorSet) detachCoverageFragmentProcessor() FragmentProcessor {
	fp := s.coverageFragmentProcessor
	s.coverageFragmentProcessor = nil
	return fp
}

// Equal reports whether two finalized processor sets are equivalent (only legal on finalized sets).
func (s *ProcessorSet) Equal(that *ProcessorSet) bool {
	if !s.finalized || !that.finalized {
		panic("comparisons are only legal on finalized processor sets")
	}
	if s.HasColorFragmentProcessor() != that.HasColorFragmentProcessor() ||
		s.HasCoverageFragmentProcessor() != that.HasCoverageFragmentProcessor() {
		return false
	}
	if s.HasColorFragmentProcessor() &&
		!fpIsEqual(s.colorFragmentProcessor, that.colorFragmentProcessor) {
		return false
	}
	if s.HasCoverageFragmentProcessor() &&
		!fpIsEqual(s.coverageFragmentProcessor, that.coverageFragmentProcessor) {
		return false
	}
	// Most of the time both of these are nil.
	if s.xferProcessor == nil && that.xferProcessor == nil {
		return true
	}
	thisXP := s.xferProcessor
	if thisXP == nil {
		thisXP = SimpleSrcOverXP()
	}
	thatXP := that.xferProcessor
	if thatXP == nil {
		thatXP = SimpleSrcOverXP()
	}
	return xpIsEqual(thisXP, thatXP)
}

// VisitProxies calls f for every surface proxy referenced by the set's color and coverage FPs.
func (s *ProcessorSet) VisitProxies(f func(*SurfaceProxy, gpu.Mipmapped)) {
	if s.HasColorFragmentProcessor() {
		fpVisitProxies(s.colorFragmentProcessor, f)
	}
	if s.HasCoverageFragmentProcessor() {
		fpVisitProxies(s.coverageFragmentProcessor, f)
	}
}

// Finalize analyzes the processors given an op's input color and coverage plus the clip, may replace the color FP chain
// with an overridden input color, and creates the transfer processor. Must be called exactly once before the set is
// used to construct a Pipeline. The returned color is only meaningful when the analysis reports InputColorIsOverridden.
func (s *ProcessorSet) Finalize(colorInput ProcessorAnalysisColor, coverageInput AnalysisCoverage, clip *AppliedClip, userStencil *UserStencilSettings, caps *gpu.Caps, clampType gpu.ClampType) (ProcessorAnalysis, colorcore.PMColor4f) {
	if s.finalized {
		panic("processor set already finalized")
	}
	_ = userStencil // consulted only by DMSAA lanes; kept for signature fidelity

	var analysis ProcessorAnalysis
	analysis.compatibleWithCoverageAsAlpha = coverageInput != AnalysisCoverageLCD

	var colorFPs []FragmentProcessor
	if s.colorFragmentProcessor != nil {
		colorFPs = []FragmentProcessor{s.colorFragmentProcessor}
	}
	colorAnalysis := NewColorFragmentProcessorAnalysis(colorInput, colorFPs)
	hasCoverageFP := s.HasCoverageFragmentProcessor()
	coverageUsesLocalCoords := false
	if hasCoverageFP {
		if !s.coverageFragmentProcessor.fpBase().CompatibleWithCoverageAsAlpha() {
			analysis.compatibleWithCoverageAsAlpha = false
		}
		if s.coverageFragmentProcessor.fpBase().UsesSampleCoords() {
			coverageUsesLocalCoords = true
		}
	}
	if clip != nil && clip.HasCoverageFragmentProcessor() {
		hasCoverageFP = true
		clipFP := clip.CoverageFragmentProcessor()
		if !clipFP.fpBase().CompatibleWithCoverageAsAlpha() {
			analysis.compatibleWithCoverageAsAlpha = false
		}
		if clipFP.fpBase().UsesSampleCoords() {
			coverageUsesLocalCoords = true
		}
	}
	overrideInputColor, colorFPsToEliminate := colorAnalysis.InitialProcessorsToEliminate()
	if colorFPsToEliminate > 0 {
		analysis.inputColorType = analysisInputColorOverridden
	} else {
		analysis.inputColorType = analysisInputColorOriginal
	}

	var outputCoverage AnalysisCoverage
	switch {
	case coverageInput == AnalysisCoverageLCD:
		outputCoverage = AnalysisCoverageLCD
	case hasCoverageFP || coverageInput == AnalysisCoverageSingleChannel:
		outputCoverage = AnalysisCoverageSingleChannel
	default:
		outputCoverage = AnalysisCoverageNone
	}

	props := XPFactoryGetAnalysisProperties(s.xpFactory, colorAnalysis.OutputColor(),
		outputCoverage, caps, clampType)
	analysis.requiresDstTexture = props&XPAnalysisRequiresDstTexture != 0 ||
		colorAnalysis.RequiresDstTexture(caps)
	analysis.compatibleWithCoverageAsAlpha = analysis.compatibleWithCoverageAsAlpha &&
		props&XPAnalysisCompatibleWithCoverageAsAlpha != 0
	analysis.requiresNonOverlappingDraws = props&XPAnalysisRequiresNonOverlappingDraws != 0 ||
		analysis.requiresDstTexture
	analysis.usesNonCoherentHWBlending = props&XPAnalysisUsesNonCoherentHWBlending != 0
	analysis.unaffectedByDstValue = props&XPAnalysisUnaffectedByDstValue != 0
	if props&XPAnalysisIgnoresInputColor != 0 {
		if s.HasColorFragmentProcessor() {
			colorFPsToEliminate = 1
		} else {
			colorFPsToEliminate = 0
		}
		analysis.inputColorType = analysisInputColorIgnored
		analysis.usesLocalCoords = coverageUsesLocalCoords
	} else {
		analysis.compatibleWithCoverageAsAlpha = analysis.compatibleWithCoverageAsAlpha &&
			colorAnalysis.AllProcessorsCompatibleWithCoverageAsAlpha()
		analysis.usesLocalCoords = coverageUsesLocalCoords || colorAnalysis.UsesLocalCoords()
	}
	if colorFPsToEliminate > 0 {
		if colorFPsToEliminate != 1 {
			panic("can only eliminate the single color FP")
		}
		s.colorFragmentProcessor = nil
	}
	analysis.hasColorFragmentProcessor = s.HasColorFragmentProcessor()

	s.xferProcessor = XPFactoryMakeXferProcessor(s.xpFactory, colorAnalysis.OutputColor(),
		outputCoverage, caps, clampType)
	s.xpFactory = nil
	s.finalized = true
	analysis.isInitialized = true

	// Sanity check: requiresNonOverlappingDraws must equal (dst texture or xfer barrier).
	hasXferBarrier := s.xferProcessor != nil &&
		s.xferProcessor.XferBarrierType(caps) != XferBarrierNone
	if analysis.requiresNonOverlappingDraws != (analysis.requiresDstTexture || hasXferBarrier) {
		panic("requiresNonOverlappingDraws inconsistent with dst texture / xfer barrier")
	}
	return analysis, overrideInputColor
}
