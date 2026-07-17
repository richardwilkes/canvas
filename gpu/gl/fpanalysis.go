// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The color/coverage invariants that flow through the FP chain ahead of pipeline construction so blending can be
// optimized and constant-folded FPs eliminated.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
)

// pmIsOpaque reports whether a normalized premultiplied color has full alpha.
func pmIsOpaque(c colorcore.PMColor4f) bool { return c.A == 1 }

// ProcessorAnalysisColor tracks what is statically known about a color flowing through the FP chain: unknown,
// unknown-but-opaque, or a known constant value.
type ProcessorAnalysisColor struct {
	color   colorcore.PMColor4f
	isKnown bool
	opaque  bool
}

// AnalysisColorUnknown returns a color analysis with no known value and no opacity guarantee.
func AnalysisColorUnknown() ProcessorAnalysisColor { return ProcessorAnalysisColor{} }

// AnalysisColorUnknownOpaque returns a color analysis with no known value but a guarantee of full opacity.
func AnalysisColorUnknownOpaque() ProcessorAnalysisColor {
	return ProcessorAnalysisColor{opaque: true}
}

// AnalysisColorConstant returns a color analysis for a known constant color.
func AnalysisColorConstant(color colorcore.PMColor4f) ProcessorAnalysisColor {
	return ProcessorAnalysisColor{color: color, isKnown: true, opaque: pmIsOpaque(color)}
}

// IsUnknown reports whether neither a constant value nor opacity is known.
func (c ProcessorAnalysisColor) IsUnknown() bool { return !c.isKnown && !c.opaque }

// IsOpaque reports whether the color is known to be fully opaque.
func (c ProcessorAnalysisColor) IsOpaque() bool { return c.opaque }

// IsConstant returns the known constant color and whether one is known.
func (c ProcessorAnalysisColor) IsConstant() (colorcore.PMColor4f, bool) {
	return c.color, c.isKnown
}

// CombineAnalysisColors merges two color analyses for values combined earlier in the chain: the result is a known
// constant only if both inputs are the same known constant, opaque if both inputs are opaque, and unknown otherwise.
func CombineAnalysisColors(a, b ProcessorAnalysisColor) ProcessorAnalysisColor {
	aColor, aConst := a.IsConstant()
	bColor, bConst := b.IsConstant()
	switch {
	case aConst && bConst && aColor == bColor:
		return AnalysisColorConstant(aColor)
	case a.IsOpaque() && b.IsOpaque():
		return AnalysisColorUnknownOpaque()
	default:
		return AnalysisColorUnknown()
	}
}

// Equal reports whether c and that carry the same known/opaque/constant state.
func (c ProcessorAnalysisColor) Equal(that ProcessorAnalysisColor) bool {
	if c.isKnown != that.isKnown || c.opaque != that.opaque {
		return false
	}
	if c.isKnown {
		return c.color == that.color
	}
	return true
}

// AnalysisCoverage classifies how many channels of coverage a draw carries.
type AnalysisCoverage int32

// AnalysisCoverage values.
const (
	AnalysisCoverageNone AnalysisCoverage = iota
	AnalysisCoverageSingleChannel
	AnalysisCoverageLCD
)

// ColorFragmentProcessorAnalysis holds invariant data gathered from the (at most one) color fragment processor.
type ColorFragmentProcessorAnalysis struct {
	lastKnownOutputColor          colorcore.PMColor4f
	processorsToEliminate         int
	isOpaque                      bool
	compatibleWithCoverageAsAlpha bool
	usesLocalCoords               bool
	willReadDstColor              bool
	outputColorKnown              bool
}

// NewColorFragmentProcessorAnalysis builds a ColorFragmentProcessorAnalysis over the given input color and FP list (nil
// entries are not allowed; pass a shorter slice).
func NewColorFragmentProcessorAnalysis(input ProcessorAnalysisColor, fps []FragmentProcessor) *ColorFragmentProcessorAnalysis {
	a := &ColorFragmentProcessorAnalysis{
		compatibleWithCoverageAsAlpha: true,
		isOpaque:                      input.IsOpaque(),
	}
	a.lastKnownOutputColor, a.outputColorKnown = input.IsConstant()
	for _, fp := range fps {
		if a.outputColorKnown {
			if out, ok := fpHasConstantOutputForConstantInput(fp, a.lastKnownOutputColor); ok {
				a.processorsToEliminate++
				a.lastKnownOutputColor = out
				a.isOpaque = pmIsOpaque(out)
				// Reset the flags since the earlier fragment processors are being eliminated.
				a.compatibleWithCoverageAsAlpha = true
				a.usesLocalCoords = false
				a.willReadDstColor = false
				continue
			}
		}

		a.outputColorKnown = false
		if a.isOpaque && !fp.fpBase().PreservesOpaqueInput() {
			a.isOpaque = false
		}
		if a.compatibleWithCoverageAsAlpha && !fp.fpBase().CompatibleWithCoverageAsAlpha() {
			a.compatibleWithCoverageAsAlpha = false
		}
		if fp.fpBase().UsesSampleCoords() {
			a.usesLocalCoords = true
		}
		if fp.fpBase().WillReadDstColor() {
			a.willReadDstColor = true
		}
	}
	return a
}

// IsOpaque reports whether the analyzed chain's output is known to be fully opaque.
func (a *ColorFragmentProcessorAnalysis) IsOpaque() bool { return a.isOpaque }

// AllProcessorsCompatibleWithCoverageAsAlpha reports whether every processor in the analyzed chain tolerates coverage
// being folded into the alpha channel.
func (a *ColorFragmentProcessorAnalysis) AllProcessorsCompatibleWithCoverageAsAlpha() bool {
	return a.compatibleWithCoverageAsAlpha
}

// UsesLocalCoords reports whether any processor in the analyzed chain samples local coordinates.
func (a *ColorFragmentProcessorAnalysis) UsesLocalCoords() bool { return a.usesLocalCoords }

// WillReadDstColor reports whether any processor in the analyzed chain reads the destination color.
func (a *ColorFragmentProcessorAnalysis) WillReadDstColor() bool { return a.willReadDstColor }

// RequiresDstTexture reports whether the analyzed chain's destination-color read must be served from a texture copy
// rather than directly in the shader.
func (a *ColorFragmentProcessorAnalysis) RequiresDstTexture(caps *gpu.Caps) bool {
	return a.willReadDstColor && !caps.ShaderCaps.DstReadInShaderSupport
}

// InitialProcessorsToEliminate returns the constant-folded output color and how many leading processors it renders
// unnecessary.
func (a *ColorFragmentProcessorAnalysis) InitialProcessorsToEliminate() (color colorcore.PMColor4f, count int) {
	return a.lastKnownOutputColor, a.processorsToEliminate
}

// OutputColor returns the analyzed chain's output as a ProcessorAnalysisColor.
func (a *ColorFragmentProcessorAnalysis) OutputColor() ProcessorAnalysisColor {
	if a.outputColorKnown {
		return AnalysisColorConstant(a.lastKnownOutputColor)
	}
	if a.isOpaque {
		return AnalysisColorUnknownOpaque()
	}
	return AnalysisColorUnknown()
}
