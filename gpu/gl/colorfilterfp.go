// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The color-filter FP conversions over the colorfilter descriptors. The reachable set: matrix (RGBA domain, clamped),
// blend-mode, compose, luma, and high contrast (whose lighting collapses to blend/matrix in the descriptor
// constructors). With no shader compiler, the runtime effects are hand-written GLSL-emitting FPs with the same uniforms
// and specializations; the working-format sandwich is the colorXformFP pair.

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

//////////////////////////////////////////////////////////////////////////////
// ColorMatrix.

type colorMatrixFP struct {
	FPBase
	m              [16]float32 // row-major 4x4 (the 4x5 matrix without the translate column)
	v              [4]float32  // the translate column
	unpremulInput  bool        // specialized
	clampRGBOutput bool        // specialized
	premulOutput   bool        // specialized
}

// ColorMatrixFP builds a fragment processor applying a row-major 4x5 color matrix.
func ColorMatrixFP(child FragmentProcessor, matrix *[20]float32, unpremulInput, clampRGBOutput, premulOutput bool) FragmentProcessor {
	fp := &colorMatrixFP{
		m: [16]float32{
			matrix[0], matrix[1], matrix[2], matrix[3],
			matrix[5], matrix[6], matrix[7], matrix[8],
			matrix[10], matrix[11], matrix[12], matrix[13],
			matrix[15], matrix[16], matrix[17], matrix[18],
		},
		v:              [4]float32{matrix[4], matrix[9], matrix[14], matrix[19]},
		unpremulInput:  unpremulInput,
		clampRGBOutput: clampRGBOutput,
		premulOutput:   premulOutput,
	}
	// The effect supports constant output.
	fp.initFP(GLSLFPClassID, FPConstantOutputForConstantInput)
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *colorMatrixFP) Name() string { return "ColorMatrix" }

func (f *colorMatrixFP) Clone() FragmentProcessor {
	clone := &colorMatrixFP{
		m:              f.m,
		v:              f.v,
		unpremulInput:  f.unpremulInput,
		clampRGBOutput: f.clampRGBOutput,
		premulOutput:   f.premulOutput,
	}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *colorMatrixFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*colorMatrixFP)
	return ok && f.m == o.m && f.v == o.v && f.unpremulInput == o.unpremulInput &&
		f.clampRGBOutput == o.clampRGBOutput && f.premulOutput == o.premulOutput
}

func (f *colorMatrixFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPColorMatrix), "runtimeFPKind")
	b.AddBool(f.unpremulInput, "unpremulInput")
	b.AddBool(f.clampRGBOutput, "clampRGBOutput")
	b.AddBool(f.premulOutput, "premulOutput")
}

func clamp01(v float32) float32 {
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

func (f *colorMatrixFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	c := fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
	v := [4]float32{c.R, c.G, c.B, c.A}
	if f.unpremulInput {
		// This must reproduce what EmitCode emits (rgb / max(a, 0.0001)) exactly, since the FP declares
		// FPConstantOutputForConstantInput and a folded draw must render the same as the unfolded one. A plain 1/a with a
		// zero-alpha special case diverges for a premul input with 0 < a < 1e-4.
		a := max32(v[3], 1e-4)
		v[0] /= a
		v[1] /= a
		v[2] /= a
	}
	var out [4]float32
	for row := range 4 {
		out[row] = f.m[row*4]*v[0] + f.m[row*4+1]*v[1] + f.m[row*4+2]*v[2] + f.m[row*4+3]*v[3] +
			f.v[row]
	}
	if f.clampRGBOutput {
		for i := range out {
			out[i] = clamp01(out[i])
		}
	} else {
		out[3] = clamp01(out[3])
	}
	if f.premulOutput {
		out[0] *= out[3]
		out[1] *= out[3]
		out[2] *= out[3]
	}
	return colorcore.PMColor4f{R: out[0], G: out[1], B: out[2], A: out[3]}
}

func (f *colorMatrixFP) onMakeProgramImpl() FPProgramImpl { return &colorMatrixImpl{} }

type colorMatrixImpl struct {
	FPImplBase
	mUni, vUni UniformHandle
}

func (i *colorMatrixImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*colorMatrixFP)
	var mName, vName string
	i.mUni, mName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf4x4,
		"m")
	i.vUni, vName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf4, "v")
	fb := args.FragBuilder
	fb.CodeAppendf("vec4 color = %s;", i.InvokeChild(0, args))
	if fp.unpremulInput {
		fb.CodeAppend("color = vec4(color.rgb / max(color.a, 0.0001), color.a);")
	}
	fb.CodeAppendf("color = %s * color + %s;", mName, vName)
	if fp.clampRGBOutput {
		fb.CodeAppend("color = clamp(color, 0.0, 1.0);")
	} else {
		fb.CodeAppend("color.a = clamp(color.a, 0.0, 1.0);")
	}
	if fp.premulOutput {
		fb.CodeAppend("color.rgb *= color.a;")
	}
	fb.CodeAppend("return color;")
}

func (i *colorMatrixImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*colorMatrixFP)
	// Row-major → the column-major mat4 GL expects.
	var cm [16]float32
	for r := range 4 {
		for c := range 4 {
			cm[c*4+r] = f.m[r*4+c]
		}
	}
	pdman.SetMatrix4f(i.mUni, &cm)
	pdman.Set4fv(i.vUni, 1, f.v[:])
}

//////////////////////////////////////////////////////////////////////////////
// Luma.

type lumaFP struct {
	FPBase
}

func makeLumaFP(child FragmentProcessor) FragmentProcessor {
	fp := &lumaFP{}
	fp.initFP(GLSLFPClassID, FPConstantOutputForConstantInput)
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *lumaFP) Name() string { return "luma_fp" }

func (f *lumaFP) Clone() FragmentProcessor {
	clone := &lumaFP{}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *lumaFP) onIsEqual(other FragmentProcessor) bool {
	_, ok := other.(*lumaFP)
	return ok
}

func (f *lumaFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPLuma), "runtimeFPKind")
}

func (f *lumaFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	c := fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
	return colorcore.PMColor4f{A: clamp01(0.2126*c.R + 0.7152*c.G + 0.0722*c.B)}
}

func (f *lumaFP) onMakeProgramImpl() FPProgramImpl { return &lumaFPImpl{} }

type lumaFPImpl struct {
	FPImplBase
}

func (i *lumaFPImpl) EmitCode(args *FPEmitArgs) {
	// Luma: return saturate(dot(half3(0.2126, 0.7152, 0.0722), color.rgb)).000r.
	args.FragBuilder.CodeAppendf("vec4 color = %s;", i.InvokeChild(0, args))
	args.FragBuilder.CodeAppend("return vec4(0.0, 0.0, 0.0, " +
		"clamp(dot(vec3(0.2126, 0.7152, 0.0722), color.rgb), 0.0, 1.0));")
}

//////////////////////////////////////////////////////////////////////////////
// High contrast, inside its working-format sandwich.

type highContrastFP struct {
	FPBase
	grayscale   bool // specialized
	invertStyle colorfilter.HighContrastInvertStyle
	contrast    float32
}

func makeHighContrastFP(child FragmentProcessor, grayscale bool, invertStyle colorfilter.HighContrastInvertStyle, contrast float32) FragmentProcessor {
	fp := &highContrastFP{grayscale: grayscale, invertStyle: invertStyle, contrast: contrast}
	fp.initFP(GLSLFPClassID, FPConstantOutputForConstantInput)
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *highContrastFP) Name() string { return "high_contrast_fp" }

func (f *highContrastFP) Clone() FragmentProcessor {
	clone := &highContrastFP{
		grayscale:   f.grayscale,
		invertStyle: f.invertStyle,
		contrast:    f.contrast,
	}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *highContrastFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*highContrastFP)
	return ok && f.grayscale == o.grayscale && f.invertStyle == o.invertStyle &&
		f.contrast == o.contrast
}

func (f *highContrastFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPHighContrast), "runtimeFPKind")
	b.AddBool(f.grayscale, "grayscale")
	b.AddBits(2, uint32(f.invertStyle), "invertStyle")
}

func (f *highContrastFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	c := fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
	r, g, b := colorfilter.EvalHighContrast(f.grayscale, f.invertStyle, f.contrast, c.R, c.G, c.B)
	return colorcore.PMColor4f{R: r, G: g, B: b, A: c.A}
}

func (f *highContrastFP) onMakeProgramImpl() FPProgramImpl { return &highContrastImpl{} }

type highContrastImpl struct {
	FPImplBase
	contrastUni UniformHandle
}

func (i *highContrastImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*highContrastFP)
	fb := args.FragBuilder
	var contrastName string
	i.contrastUni, contrastName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "contrast")

	fb.CodeAppendf("vec4 inColor = %s;", i.InvokeChild(0, args))
	fb.CodeAppend("vec3 color = inColor.rgb;")
	if fp.grayscale {
		fb.CodeAppend("color = vec3(dot(vec3(0.2126, 0.7152, 0.0722), color));")
	}
	switch fp.invertStyle {
	case colorfilter.InvertBrightness:
		fb.CodeAppend("color = 1.0 - color;")
	case colorfilter.InvertLightness:
		rgbToHsl := i.emitHighContrastRGBToHSL(fb)
		hslToRgb := i.emitHslToRgb(fb)
		fb.CodeAppendf("color = %s(color);", rgbToHsl)
		fb.CodeAppend("color.b = 1.0 - color.b;")
		fb.CodeAppendf("color = %s(color);", hslToRgb)
	default:
	}
	fb.CodeAppendf("color = clamp(mix(vec3(0.5), color, %s), 0.0, 1.0);", contrastName)
	fb.CodeAppend("return vec4(color, inColor.a);")
}

// emitHighContrastRGBToHSL emits the RGB-to-HSL helper function used by the high-contrast filter.
func (i *highContrastImpl) emitHighContrastRGBToHSL(fb *FragmentShaderBuilder) string {
	name := fb.GetMangledFunctionName("high_contrast_rgb_to_hsl")
	fb.EmitFunction(GLSLTypeHalf3, name, []ShaderVar{NewShaderVar("c", GLSLTypeHalf3)},
		"float mx = max(max(c.r, c.g), c.b);\n"+
			"float mn = min(min(c.r, c.g), c.b);\n"+
			"float d = mx - mn;\n"+
			"float invd = 1.0 / d;\n"+
			"float g_lt_b = c.g < c.b ? 6.0 : 0.0;\n"+
			// We'd prefer `mx == c.r`, but on some GPUs max(x,y) is not always equal to either x or y, so use the long
			// form.
			"float h = (1.0/6.0) * (mx == mn ? 0.0 :\n"+
			"    c.r >= c.g && c.r >= c.b ? invd * (c.g - c.b) + g_lt_b :\n"+
			"    c.g >= c.b               ? invd * (c.b - c.r) + 2.0\n"+
			"                             : invd * (c.r - c.g) + 4.0);\n"+
			"float sum = mx + mn;\n"+
			"float l = sum * 0.5;\n"+
			"float s = mx == mn ? 0.0 : d / (l > 0.5 ? 2.0 - sum : sum);\n"+
			"return vec3(h, s, l);")
	return name
}

// emitHslToRgb emits the HSL-to-RGB helper function used by the high-contrast filter.
func (i *highContrastImpl) emitHslToRgb(fb *FragmentShaderBuilder) string {
	name := fb.GetMangledFunctionName("hsl_to_rgb")
	fb.EmitFunction(GLSLTypeHalf3, name, []ShaderVar{NewShaderVar("hsl", GLSLTypeHalf3)},
		"float C = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;\n"+
			"vec3 p = hsl.xxx + vec3(0.0, 2.0/3.0, 1.0/3.0);\n"+
			"vec3 q = clamp(abs(fract(p) * 6.0 - 3.0) - 1.0, 0.0, 1.0);\n"+
			"return (q - 0.5) * C + hsl.z;")
	return name
}

func (i *highContrastImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	pdman.Set1f(i.contrastUni, fp.(*highContrastFP).contrast)
}

//////////////////////////////////////////////////////////////////////////////
// Dispatcher.

// MakeColorFilterFP converts a color-filter descriptor to an FP chain over inputFP. On failure (nil filter or an
// unreachable configuration) it returns (nil, false) and the caller must treat inputFP as consumed.
func MakeColorFilterFP(args *FPArgs, cf shaders.ColorFilter, inputFP FragmentProcessor) (FragmentProcessor, bool) {
	if cf == nil {
		return nil, false
	}
	if m, ok := colorfilter.AsMatrix(cf); ok {
		// The reachable form always unpremultiplies the input and clamps the RGB output.
		return ColorMatrixFP(inputFP, &m, true /* unpremulInput */, true, /* clampRGBOutput */
			true /* premulOutput */), true
	}
	if color, mode, ok := colorfilter.AsBlend(cf); ok {
		return makeBlendColorFilterFP(color, mode, inputFP)
	}
	if outer, inner, ok := colorfilter.AsCompose(cf); ok {
		// Clone the input before we know we need it, so the original FP can be returned if either internal filter
		// fails.
		var inputClone FragmentProcessor
		if inputFP != nil {
			inputClone = inputFP.Clone()
		}
		innerFP, innerOK := MakeColorFilterFP(args, inner, inputFP)
		if !innerOK {
			return inputClone, false
		}
		outerFP, outerOK := MakeColorFilterFP(args, outer, innerFP)
		if !outerOK {
			return inputClone, false
		}
		return outerFP, true
	}
	if colorfilter.AsLuma(cf) {
		return makeLumaFP(inputFP), true
	}
	if grayscale, invertStyle, contrast, ok := colorfilter.AsHighContrast(cf); ok {
		// The working-format sandwich: dst (premul sRGB) → working (unpremul linear sRGB) → filter → back.
		fp := ColorXformFP(inputFP, XformStepUnpremul|XformStepLinearize)
		fp = makeHighContrastFP(fp, grayscale, invertStyle, contrast)
		return ColorXformFP(fp, XformStepEncode|XformStepPremul), true
	}
	panic(fmt.Sprintf("unknown color filter type %T", cf))
}

// makeBlendColorFilterFP builds the FP chain for a blend-mode color filter.
func makeBlendColorFilterFP(color colorcore.Color4f, mode raster.BlendMode, inputFP FragmentProcessor) (FragmentProcessor, bool) {
	if mode == raster.BlendDst {
		// If the blend mode is "dest," the blend color won't factor into it at all; return the input FP as-is.
		return inputFP, true
	}
	// map_color: sRGB → dst is the identity; premultiply.
	pm := colorcore.PMColor4f{
		R: color.R * color.A, G: color.G * color.A, B: color.B * color.A,
		A: color.A,
	}
	return BlendFP(MakeColorFP(pm), inputFP, mode), true
}
