// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ShaderCaps: the shader-related capabilities the GLSL emitters and analysis code consult, merged into a single flat
// struct since there is no separate shader-compiler layer here. ES-only extension-string fields (shader derivatives,
// secondary output, noperspective interpolation, sample variables) and external-texture fields are dropped; on desktop
// GL those extensions are either core or unreachable.

package gpu

// GLSLGeneration identifies a GLSL language version, desktop versions only. Ordering matters so >= comparisons select
// "at least this generation".
type GLSLGeneration int32

// GLSLGeneration values.
const (
	// GLSL110 is desktop GLSL 1.10.
	GLSL110 GLSLGeneration = iota
	// GLSL130 is desktop GLSL 1.30.
	GLSL130
	// GLSL140 is desktop GLSL 1.40.
	GLSL140
	// GLSL150 is desktop GLSL 1.50.
	GLSL150
	// GLSL330 is desktop GLSL 3.30.
	GLSL330
	// GLSL400 is desktop GLSL 4.00.
	GLSL400
	// GLSL420 is desktop GLSL 4.20.
	GLSL420
)

// AdvBlendEqInteraction describes how GLSL must interact with advanced blend equations. Some drivers require special
// layout qualifiers in the fragment shader to use them.
type AdvBlendEqInteraction int32

// AdvBlendEqInteraction values.
const (
	// NotSupportedAdvBlendEqInteraction: no *_blend_equation_advanced extension.
	NotSupportedAdvBlendEqInteraction AdvBlendEqInteraction = iota
	// AutomaticAdvBlendEqInteraction: no interaction required.
	AutomaticAdvBlendEqInteraction
	// GeneralEnableAdvBlendEqInteraction: layout(blend_support_all_equations) out.
	GeneralEnableAdvBlendEqInteraction
)

// ShaderCaps holds the shader-related capabilities. Fields keep their intended default values via NewShaderCaps; the Go
// zero value is not meaningful.
type ShaderCaps struct {
	// VersionDeclString is the "#version ...\n" line to emit at the top of generated shaders.
	VersionDeclString string
	// FBFetchColorName is the variable to read the framebuffer color from when FBFetchSupport is set.
	FBFetchColorName string
	// FBFetchExtensionString names the extension to enable in shaders that use framebuffer fetch.
	FBFetchExtensionString string

	GLSLGeneration        GLSLGeneration
	AdvBlendEqInteraction AdvBlendEqInteraction
	MaxFragmentSamplers   int

	DualSourceBlendingSupport bool
	ShaderDerivativeSupport   bool
	// ExplicitTextureLodSupport enables sampleGrad and sampleLod functions that don't rely on implicit derivatives.
	ExplicitTextureLodSupport bool
	// IntegerSupport indicates true 32-bit integer support, with unsigned types and bitwise operations.
	IntegerSupport                    bool
	NonsquareMatrixSupport            bool
	InverseHyperbolicSupport          bool
	FBFetchSupport                    bool
	FBFetchNeedsCustomOutput          bool
	UsesPrecisionModifiers            bool
	FlatInterpolationSupport          bool
	NoPerspectiveInterpolationSupport bool
	SampleMaskSupport                 bool
	FloatIs32Bits                     bool
	// InfinitySupport: isinf() is defined, and floating point infinities are handled per IEEE.
	InfinitySupport bool
	// BuiltinFMASupport and BuiltinDeterminantSupport tell the GLSL emitters when to generate polyfills.
	BuiltinFMASupport         bool
	BuiltinDeterminantSupport bool

	// Additional shader-backend capability fields.
	DstReadInShaderSupport  bool
	PreferFlatInterpolation bool
	VertexIDSupport         bool
	// NonconstantArrayIndexSupport is true if `expr` in `myArray[expr]` can be any integer expression. If false, it
	// must be a constant-index-expression per the OpenGL ES2 spec, Appendix A.5.
	NonconstantArrayIndexSupport bool
	// BitManipulationSupport covers frexp(), ldexp(), findMSB(), findLSB().
	BitManipulationSupport  bool
	HalfIs32Bits            bool
	HasLowFragmentPrecision bool
	// ReducedShaderMode uses a reduced set of rendering algorithms or less optimal effects in order to reduce the
	// number of unique shaders generated.
	ReducedShaderMode bool

	// Driver-workaround fields consulted by the GLSL emitters. Only the ones settable through desktop code paths
	// survive the trim.
	CanUseMinAndAbsTogether                     bool
	CanUseFractForNegativeValues                bool
	MustForceNegatedAtanParamToFloat            bool
	MustForceNegatedLdexpParamToMultiply        bool
	MustDoOpBetweenFloorAndAbs                  bool
	MustGuardDivisionEvenAfterExplicitZeroCheck bool
	CanUseFragCoord                             bool
	AddAndTrueToLoopCondition                   bool
	UnfoldShortCircuitAsTernary                 bool
	EmulateAbsIntFunction                       bool
	RewriteDoWhileLoops                         bool
	RewriteSwitchStatements                     bool
	RemovePowWithConstantExponent               bool
	RequiresLocalOutputColorForFBFetch          bool
	MustObfuscateUniformColor                   bool
	MustWriteToFragColor                        bool
	AvoidDfDxForGradientsWhenPossible           bool
	RewriteMatrixVectorMultiply                 bool
	RewriteMatrixComparisons                    bool
	RemoveConstFromFunctionParameters           bool
	PerlinNoiseRoundingFix                      bool
	CanUseVoidInSequenceExpressions             bool
	VectorClampMinMaxSupport                    bool
}

// NewShaderCaps returns a ShaderCaps with the standard field defaults.
func NewShaderCaps() *ShaderCaps {
	return &ShaderCaps{
		GLSLGeneration:                  GLSL330,
		FloatIs32Bits:                   true,
		BuiltinFMASupport:               true,
		BuiltinDeterminantSupport:       true,
		CanUseVoidInSequenceExpressions: true,
		CanUseMinAndAbsTogether:         true,
		CanUseFractForNegativeValues:    true,
		CanUseFragCoord:                 true,
		VectorClampMinMaxSupport:        true,
	}
}

// MustEnableAdvBlendEqs reports whether the fragment shader must declare a layout qualifier to use advanced blend
// equations.
func (sc *ShaderCaps) MustEnableAdvBlendEqs() bool {
	return sc.AdvBlendEqInteraction >= GeneralEnableAdvBlendEqInteraction
}

// MustDeclareFragmentShaderOutput reports whether the fragment shader must explicitly declare its output variable
// rather than relying on the legacy built-in.
func (sc *ShaderCaps) MustDeclareFragmentShaderOutput() bool {
	return sc.GLSLGeneration > GLSL110
}

// SupportsDistanceFieldText reports whether the shader backend can support distance-field text rendering.
func (sc *ShaderCaps) SupportsDistanceFieldText() bool { return sc.ShaderDerivativeSupport }

// ApplyOptionsOverrides applies context-option overrides on top of the detected shader caps.
func (sc *ShaderCaps) ApplyOptionsOverrides(options *ContextOptions) {
	if options.ReducedShaderVariations {
		sc.ReducedShaderMode = true
	}
}
