// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU blend equations and coefficients used to configure a hardware blend state, plus the helpers that classify and
// validate them. The advanced-blend disable flags in Caps are bit-indexed by these values, so the ordering below must
// stay fixed.

package gpu

// BlendEquation identifies a hardware blend equation: a basic Porter-Duff combination or one of the advanced (SVG/PDF)
// blend modes.
type BlendEquation uint8

// BlendEquation values, ordered so BlendEquationFirstAdvanced marks the start of the advanced set.
const (
	// Basic blend equations.
	BlendEquationAdd             BlendEquation = iota // Cs*S + Cd*D
	BlendEquationSubtract                             // Cs*S - Cd*D
	BlendEquationReverseSubtract                      // Cd*D - Cs*S

	// Advanced blend equations, described in the SVG and PDF specs.
	BlendEquationScreen
	BlendEquationOverlay
	BlendEquationDarken
	BlendEquationLighten
	BlendEquationColorDodge
	BlendEquationColorBurn
	BlendEquationHardLight
	BlendEquationSoftLight
	BlendEquationDifference
	BlendEquationExclusion
	BlendEquationMultiply
	BlendEquationHSLHue
	BlendEquationHSLSaturation
	BlendEquationHSLColor
	BlendEquationHSLLuminosity

	BlendEquationIllegal

	// BlendEquationFirstAdvanced is the first advanced equation.
	BlendEquationFirstAdvanced = BlendEquationScreen
	// BlendEquationCount is the number of valid BlendEquation values.
	BlendEquationCount = int(BlendEquationIllegal) + 1
)

// IsAdvanced reports whether this is one of the advanced (SVG/PDF) blend equations rather than a basic Porter-Duff one.
func (e BlendEquation) IsAdvanced() bool { return e >= BlendEquationFirstAdvanced }

// BlendCoeff is a coefficient for alpha-blending: what to multiply a color term by.
type BlendCoeff uint8

// BlendCoeff values.
const (
	BlendCoeffZero    BlendCoeff = iota // 0
	BlendCoeffOne                       // 1
	BlendCoeffSC                        // src color
	BlendCoeffISC                       // one minus src color
	BlendCoeffDC                        // dst color
	BlendCoeffIDC                       // one minus dst color
	BlendCoeffSA                        // src alpha
	BlendCoeffISA                       // one minus src alpha
	BlendCoeffDA                        // dst alpha
	BlendCoeffIDA                       // one minus dst alpha
	BlendCoeffConstC                    // constant color
	BlendCoeffIConstC                   // one minus constant color
	BlendCoeffS2C
	BlendCoeffIS2C
	BlendCoeffS2A
	BlendCoeffIS2A

	BlendCoeffIllegal

	// BlendCoeffCount is the number of valid BlendCoeff values.
	BlendCoeffCount = int(BlendCoeffIllegal) + 1
)

// RefsSrc reports whether this coefficient depends on the source color or alpha.
func (c BlendCoeff) RefsSrc() bool {
	return c == BlendCoeffSC || c == BlendCoeffISC || c == BlendCoeffSA || c == BlendCoeffISA
}

// RefsDst reports whether this coefficient depends on the destination color or alpha.
func (c BlendCoeff) RefsDst() bool {
	return c == BlendCoeffDC || c == BlendCoeffIDC || c == BlendCoeffDA || c == BlendCoeffIDA
}

// RefsSrc2 reports whether this coefficient depends on the secondary source color or alpha (dual source blending).
func (c BlendCoeff) RefsSrc2() bool {
	return c == BlendCoeffS2C || c == BlendCoeffIS2C || c == BlendCoeffS2A || c == BlendCoeffIS2A
}

// RefsConstant reports whether this coefficient depends on the blend constant color.
func (c BlendCoeff) RefsConstant() bool {
	return c == BlendCoeffConstC || c == BlendCoeffIConstC
}

// BlendShouldDisable reports whether this equation/coefficient combination is a no-op blend (dst = src) that can be
// implemented by disabling blending entirely.
func BlendShouldDisable(equation BlendEquation, srcCoeff, dstCoeff BlendCoeff) bool {
	return (equation == BlendEquationAdd || equation == BlendEquationSubtract) &&
		srcCoeff == BlendCoeffOne && dstCoeff == BlendCoeffZero
}

// BlendInfo describes a resolved hardware blend state. BlendConstant is a premultiplied RGBA color.
type BlendInfo struct {
	BlendConstant [4]float32
	Equation      BlendEquation
	SrcBlend      BlendCoeff
	DstBlend      BlendCoeff
	WritesColor   bool
}

// MakeBlendInfo returns the default blend state: add equation, src-one/dst-zero coefficients (equivalent to no
// blending), a transparent constant, and color writes enabled.
func MakeBlendInfo() BlendInfo {
	return BlendInfo{
		Equation:    BlendEquationAdd,
		SrcBlend:    BlendCoeffOne,
		DstBlend:    BlendCoeffZero,
		WritesColor: true,
	}
}
