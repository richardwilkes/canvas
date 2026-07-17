// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The blend-formula tables: for each Porter-Duff blend mode, with and without coverage and with and without an opaque
// input color, the shader outputs and hardware blend state that implement it. Property deduction (does it modify the
// destination, can alpha be tweaked for coverage, etc.) runs once at table-construction time.

package gpu

import "github.com/richardwilkes/canvas/raster"

// BlendCoeffRefsSrc reports whether c depends on the source color or alpha. It exists as a free function alongside the
// BlendCoeff.RefsSrc method for use in the formula tables below.
func BlendCoeffRefsSrc(c BlendCoeff) bool { return c.RefsSrc() }

// BlendCoeffsUseSrcColor reports whether this src/dst coefficient pair reads the source color.
func BlendCoeffsUseSrcColor(srcCoeff, dstCoeff BlendCoeff) bool {
	return srcCoeff != BlendCoeffZero || dstCoeff.RefsSrc()
}

// BlendCoeffsUseDstColor reports whether this src/dst coefficient pair reads the destination color, given whether the
// source color is known to be opaque.
func BlendCoeffsUseDstColor(srcCoeff, dstCoeff BlendCoeff, srcColorIsOpaque bool) bool {
	return srcCoeff.RefsDst() ||
		(dstCoeff != BlendCoeffZero && (dstCoeff != BlendCoeffISA || !srcColorIsOpaque))
}

// BlendModifiesDst reports whether this equation/coefficient combination can change the destination value (false means
// it is a no-op blend).
func BlendModifiesDst(equation BlendEquation, srcCoeff, dstCoeff BlendCoeff) bool {
	return (equation != BlendEquationAdd && equation != BlendEquationReverseSubtract) ||
		srcCoeff != BlendCoeffZero || dstCoeff != BlendCoeffOne
}

// BlendAllowsCoverageAsAlpha reports whether coverage can be folded into the alpha channel of the source color rather
// than needing a separate coverage multiply.
func BlendAllowsCoverageAsAlpha(equation BlendEquation, srcCoeff, dstCoeff BlendCoeff) bool {
	return equation.IsAdvanced() ||
		!BlendModifiesDst(equation, srcCoeff, dstCoeff) ||
		((equation == BlendEquationAdd || equation == BlendEquationReverseSubtract) &&
			!srcCoeff.RefsSrc() &&
			(dstCoeff == BlendCoeffOne || dstCoeff == BlendCoeffISC || dstCoeff == BlendCoeffISA))
}

// BlendFormulaOutputType identifies the value a shader writes to its primary or secondary output. These are all
// modulated by coverage; the multiplies are ignored when not using coverage.
type BlendFormulaOutputType uint8

// BlendFormulaOutputType values.
const (
	BlendFormulaOutputNone        BlendFormulaOutputType = iota // 0
	BlendFormulaOutputCoverage                                  // inputCoverage
	BlendFormulaOutputModulate                                  // inputColor * inputCoverage
	BlendFormulaOutputSAModulate                                // inputColor.a * inputCoverage
	BlendFormulaOutputISAModulate                               // (1 - inputColor.a) * inputCoverage
	BlendFormulaOutputISCModulate                               // (1 - inputColor) * inputCoverage

	// BlendFormulaOutputTypeLast is the last valid output type.
	BlendFormulaOutputTypeLast = BlendFormulaOutputISCModulate
)

// BlendFormula property bits.
const (
	blendFormulaModifiesDst = 1 << iota
	blendFormulaUnaffectedByDst
	blendFormulaUnaffectedByDstIfOpaque
	blendFormulaUsesInputColor
	blendFormulaCanTweakAlphaForCoverage
)

// BlendFormula is the shader outputs and hardware blend state that implement a Porter-Duff blend mode with coverage.
type BlendFormula struct {
	primaryOutputType   BlendFormulaOutputType
	secondaryOutputType BlendFormulaOutputType
	blendEquation       BlendEquation
	srcCoeff            BlendCoeff
	dstCoeff            BlendCoeff
	props               uint8
}

// MakeBlendFormula builds a BlendFormula, deducing its property bits from the given outputs, equation, and
// coefficients.
func MakeBlendFormula(primaryOut, secondaryOut BlendFormulaOutputType, equation BlendEquation, srcCoeff, dstCoeff BlendCoeff) BlendFormula {
	// The provided formula must already be optimized: these checks validate that invariant.
	if (primaryOut == BlendFormulaOutputNone) == BlendCoeffsUseSrcColor(srcCoeff, dstCoeff) {
		panic("primary output inconsistent with coefficients")
	}
	if srcCoeff.RefsSrc2() {
		panic("src coefficient references src2")
	}
	if (secondaryOut == BlendFormulaOutputNone) == dstCoeff.RefsSrc2() {
		panic("secondary output inconsistent with dst coefficient")
	}
	if primaryOut == secondaryOut && primaryOut != BlendFormulaOutputNone {
		panic("primary and secondary outputs must differ")
	}
	if primaryOut == BlendFormulaOutputNone && secondaryOut != BlendFormulaOutputNone {
		panic("secondary output without primary output")
	}

	var props uint8
	if BlendModifiesDst(equation, srcCoeff, dstCoeff) {
		props |= blendFormulaModifiesDst
	}
	if !BlendCoeffsUseDstColor(srcCoeff, dstCoeff, false /* srcColorIsOpaque */) {
		props |= blendFormulaUnaffectedByDst
	}
	if !BlendCoeffsUseDstColor(srcCoeff, dstCoeff, true /* srcColorIsOpaque */) {
		props |= blendFormulaUnaffectedByDstIfOpaque
	}
	if (primaryOut >= BlendFormulaOutputModulate && BlendCoeffsUseSrcColor(srcCoeff, dstCoeff)) ||
		(secondaryOut >= BlendFormulaOutputModulate && dstCoeff.RefsSrc2()) {
		props |= blendFormulaUsesInputColor
	}
	if (primaryOut == BlendFormulaOutputModulate || primaryOut == BlendFormulaOutputNone) &&
		secondaryOut == BlendFormulaOutputNone &&
		BlendAllowsCoverageAsAlpha(equation, srcCoeff, dstCoeff) {
		props |= blendFormulaCanTweakAlphaForCoverage
	}

	return BlendFormula{
		primaryOutputType:   primaryOut,
		secondaryOutputType: secondaryOut,
		blendEquation:       equation,
		srcCoeff:            srcCoeff,
		dstCoeff:            dstCoeff,
		props:               props,
	}
}

// HasSecondaryOutput reports whether the formula writes a secondary shader output (dual source blending).
func (f BlendFormula) HasSecondaryOutput() bool {
	return f.secondaryOutputType != BlendFormulaOutputNone
}

// ModifiesDst reports whether the formula can change the destination value.
func (f BlendFormula) ModifiesDst() bool { return f.props&blendFormulaModifiesDst != 0 }

// UnaffectedByDst reports whether the result is independent of the destination's current value.
func (f BlendFormula) UnaffectedByDst() bool { return f.props&blendFormulaUnaffectedByDst != 0 }

// UnaffectedByDstIfOpaque reports whether the result is independent of the destination's current value when the source
// color is opaque. The formula is not always fully optimized for opaque input (e.g. opaque src-over), so this weaker
// variant covers those cases.
func (f BlendFormula) UnaffectedByDstIfOpaque() bool {
	return f.props&blendFormulaUnaffectedByDstIfOpaque != 0
}

// UsesInputColor reports whether the formula reads the shader's input color.
func (f BlendFormula) UsesInputColor() bool { return f.props&blendFormulaUsesInputColor != 0 }

// CanTweakAlphaForCoverage reports whether coverage can be folded into the source alpha instead of needing a separate
// coverage multiply.
func (f BlendFormula) CanTweakAlphaForCoverage() bool {
	return f.props&blendFormulaCanTweakAlphaForCoverage != 0
}

// Equation returns the hardware blend equation.
func (f BlendFormula) Equation() BlendEquation { return f.blendEquation }

// SrcCoeff returns the hardware source blend coefficient.
func (f BlendFormula) SrcCoeff() BlendCoeff { return f.srcCoeff }

// DstCoeff returns the hardware destination blend coefficient.
func (f BlendFormula) DstCoeff() BlendCoeff { return f.dstCoeff }

// PrimaryOutput returns the value the shader writes to its primary output.
func (f BlendFormula) PrimaryOutput() BlendFormulaOutputType { return f.primaryOutputType }

// SecondaryOutput returns the value the shader writes to its secondary output.
func (f BlendFormula) SecondaryOutput() BlendFormulaOutputType { return f.secondaryOutputType }

// makeCoeffFormula builds the standard Porter-Duff formula for use when there is no coverage, or the blend mode can
// tweak alpha for coverage instead of needing a separate multiply.
func makeCoeffFormula(srcCoeff, dstCoeff BlendCoeff) BlendFormula {
	// When the coeffs are (Zero, Zero) or (Zero, One) we set the primary output to none.
	if srcCoeff == BlendCoeffZero && (dstCoeff == BlendCoeffZero || dstCoeff == BlendCoeffOne) {
		return MakeBlendFormula(BlendFormulaOutputNone, BlendFormulaOutputNone,
			BlendEquationAdd, BlendCoeffZero, dstCoeff)
	}
	return MakeBlendFormula(BlendFormulaOutputModulate, BlendFormulaOutputNone,
		BlendEquationAdd, srcCoeff, dstCoeff)
}

// makeSAModulateFormula builds a formula for the LCD dst-out case.
func makeSAModulateFormula(srcCoeff, dstCoeff BlendCoeff) BlendFormula {
	return MakeBlendFormula(BlendFormulaOutputSAModulate, BlendFormulaOutputNone,
		BlendEquationAdd, srcCoeff, dstCoeff)
}

// makeCoverageFormula builds a formula that outputs [f * (1 - dstCoeff)] as the secondary color and replaces the
// hardware dst coefficient with IS2C.
func makeCoverageFormula(oneMinusDstCoeffModulateOutput BlendFormulaOutputType, srcCoeff BlendCoeff) BlendFormula {
	return MakeBlendFormula(BlendFormulaOutputModulate, oneMinusDstCoeffModulateOutput,
		BlendEquationAdd, srcCoeff, BlendCoeffIS2C)
}

// makeCoverageSrcCoeffZeroFormula builds a formula that outputs [f * (1 - dstCoeff)] as the primary color with a
// reverse-subtract hardware blend equation and coefficients of (DC, One).
func makeCoverageSrcCoeffZeroFormula(oneMinusDstCoeffModulateOutput BlendFormulaOutputType) BlendFormula {
	return MakeBlendFormula(oneMinusDstCoeffModulateOutput, BlendFormulaOutputNone,
		BlendEquationReverseSubtract, BlendCoeffDC, BlendCoeffOne)
}

// makeCoverageDstCoeffZeroFormula builds a formula that outputs [f] as the secondary color and replaces the hardware
// dst coefficient with IS2A.
func makeCoverageDstCoeffZeroFormula(srcCoeff BlendCoeff) BlendFormula {
	return MakeBlendFormula(BlendFormulaOutputModulate, BlendFormulaOutputCoverage,
		BlendEquationAdd, srcCoeff, BlendCoeffIS2A)
}

// blendTable holds, for each combination of [isOpaque][hasCoverage][blend mode], the formula that implements that
// Porter-Duff blend mode. RGB (LCD) coverage is not supported here.
var blendTable = [2][2][int(raster.BlendScreen) + 1]BlendFormula{
	{ // No coverage, input color unknown.
		{
			/* clear */ makeCoeffFormula(BlendCoeffZero, BlendCoeffZero),
			/* src */ makeCoeffFormula(BlendCoeffOne, BlendCoeffZero),
			/* dst */ makeCoeffFormula(BlendCoeffZero, BlendCoeffOne),
			/* src-over */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISA),
			/* dst-over */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffOne),
			/* src-in */ makeCoeffFormula(BlendCoeffDA, BlendCoeffZero),
			/* dst-in */ makeCoeffFormula(BlendCoeffZero, BlendCoeffSA),
			/* src-out */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffZero),
			/* dst-out */ makeCoeffFormula(BlendCoeffZero, BlendCoeffISA),
			/* src-atop */ makeCoeffFormula(BlendCoeffDA, BlendCoeffISA),
			/* dst-atop */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffSA),
			/* xor */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffISA),
			/* plus */ makeCoeffFormula(BlendCoeffOne, BlendCoeffOne),
			/* modulate */ makeCoeffFormula(BlendCoeffZero, BlendCoeffSC),
			/* screen */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISC),
		},
		{ // Has coverage, input color unknown.
			/* clear */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputCoverage),
			/* src */ makeCoverageDstCoeffZeroFormula(BlendCoeffOne),
			/* dst */ makeCoeffFormula(BlendCoeffZero, BlendCoeffOne),
			/* src-over */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISA),
			/* dst-over */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffOne),
			/* src-in */ makeCoverageDstCoeffZeroFormula(BlendCoeffDA),
			/* dst-in */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputISAModulate),
			/* src-out */ makeCoverageDstCoeffZeroFormula(BlendCoeffIDA),
			/* dst-out */ makeCoeffFormula(BlendCoeffZero, BlendCoeffISA),
			/* src-atop */ makeCoeffFormula(BlendCoeffDA, BlendCoeffISA),
			/* dst-atop */ makeCoverageFormula(BlendFormulaOutputISAModulate, BlendCoeffIDA),
			/* xor */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffISA),
			/* plus */ makeCoeffFormula(BlendCoeffOne, BlendCoeffOne),
			/* modulate */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputISCModulate),
			/* screen */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISC),
		},
	},
	{ // No coverage, input color opaque.
		{
			/* clear */ makeCoeffFormula(BlendCoeffZero, BlendCoeffZero),
			/* src */ makeCoeffFormula(BlendCoeffOne, BlendCoeffZero),
			/* dst */ makeCoeffFormula(BlendCoeffZero, BlendCoeffOne),
			// src-over is intentionally not converted to src mode here even though the source is opaque.
			/* src-over */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISA),
			/* dst-over */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffOne),
			/* src-in */ makeCoeffFormula(BlendCoeffDA, BlendCoeffZero),
			/* dst-in */ makeCoeffFormula(BlendCoeffZero, BlendCoeffOne),
			/* src-out */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffZero),
			/* dst-out */ makeCoeffFormula(BlendCoeffZero, BlendCoeffZero),
			/* src-atop */ makeCoeffFormula(BlendCoeffDA, BlendCoeffZero),
			/* dst-atop */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffOne),
			/* xor */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffZero),
			/* plus */ makeCoeffFormula(BlendCoeffOne, BlendCoeffOne),
			/* modulate */ makeCoeffFormula(BlendCoeffZero, BlendCoeffSC),
			/* screen */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISC),
		},
		{ // Has coverage, input color opaque.
			/* clear */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputCoverage),
			/* src */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISA),
			/* dst */ makeCoeffFormula(BlendCoeffZero, BlendCoeffOne),
			/* src-over */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISA),
			/* dst-over */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffOne),
			/* src-in */ makeCoeffFormula(BlendCoeffDA, BlendCoeffISA),
			/* dst-in */ makeCoeffFormula(BlendCoeffZero, BlendCoeffOne),
			/* src-out */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffISA),
			/* dst-out */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputCoverage),
			/* src-atop */ makeCoeffFormula(BlendCoeffDA, BlendCoeffISA),
			/* dst-atop */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffOne),
			/* xor */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffISA),
			/* plus */ makeCoeffFormula(BlendCoeffOne, BlendCoeffOne),
			/* modulate */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputISCModulate),
			/* screen */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISC),
		},
	},
}

// lcdBlendTable holds, per blend mode, the formula used for LCD (per-channel RGB) coverage.
var lcdBlendTable = [int(raster.BlendScreen) + 1]BlendFormula{
	/* clear */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputCoverage),
	/* src */ makeCoverageFormula(BlendFormulaOutputCoverage, BlendCoeffOne),
	/* dst */ makeCoeffFormula(BlendCoeffZero, BlendCoeffOne),
	/* src-over */ makeCoverageFormula(BlendFormulaOutputSAModulate, BlendCoeffOne),
	/* dst-over */ makeCoeffFormula(BlendCoeffIDA, BlendCoeffOne),
	/* src-in */ makeCoverageFormula(BlendFormulaOutputCoverage, BlendCoeffDA),
	/* dst-in */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputISAModulate),
	/* src-out */ makeCoverageFormula(BlendFormulaOutputCoverage, BlendCoeffIDA),
	/* dst-out */ makeSAModulateFormula(BlendCoeffZero, BlendCoeffISC),
	/* src-atop */ makeCoverageFormula(BlendFormulaOutputSAModulate, BlendCoeffDA),
	/* dst-atop */ makeCoverageFormula(BlendFormulaOutputISAModulate, BlendCoeffIDA),
	/* xor */ makeCoverageFormula(BlendFormulaOutputSAModulate, BlendCoeffIDA),
	/* plus */ makeCoeffFormula(BlendCoeffOne, BlendCoeffOne),
	/* modulate */ makeCoverageSrcCoeffZeroFormula(BlendFormulaOutputISCModulate),
	/* screen */ makeCoeffFormula(BlendCoeffOne, BlendCoeffISC),
}

// GetBlendFormula returns the blend formula for a coefficient-based blend mode, given whether the input color is known
// to be opaque and whether coverage is present.
func GetBlendFormula(isOpaque, hasCoverage bool, mode raster.BlendMode) BlendFormula {
	if mode > raster.BlendScreen {
		panic("blend mode is not a coefficient mode")
	}
	return blendTable[b2i(isOpaque)][b2i(hasCoverage)][mode]
}

// GetLCDBlendFormula returns the blend formula for a coefficient-based blend mode used with LCD (per-channel RGB)
// coverage.
func GetLCDBlendFormula(mode raster.BlendMode) BlendFormula {
	if mode > raster.BlendScreen {
		panic("blend mode is not a coefficient mode")
	}
	return lcdBlendTable[mode]
}

func b2i(v bool) int {
	if v {
		return 1
	}
	return 0
}
