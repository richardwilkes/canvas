// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file builds the vertex and fragment shader sources emitted for a GPU program: _position is a local vec4 whose
// value is written to gl_Position through an RT-adjust fixup appended to main; fsColorOut becomes the declared color
// output; shared shader-math helper functions the processors rely on (proj, the blend_* family) are emitted on demand
// into the definitions section via small registries on the fragment builder.

package gl

import (
	"fmt"
	"strings"

	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// Shader string sections, emitted in this order when a shader is finalized.
const (
	sectionExtensions = iota
	sectionDefinitions
	sectionPrecisionQualifier
	sectionLayoutQualifiers
	sectionUniforms
	sectionInputs
	sectionOutputs
	sectionFunctions
	sectionMain
	sectionCode
)

// ShaderBuilder accumulates the GLSL text for one shader stage under construction, split into named sections
// (extensions, uniforms, functions, code, ...) that are concatenated in order at finalize.
type ShaderBuilder struct {
	programBuilder     *ProgramBuilder
	onFinalize         func()
	shaderStrings      [][]byte
	compiled           []byte
	codeIndex          int
	tmpVariableCounter int
	finalized          bool
}

func (s *ShaderBuilder) initShaderBuilder(program *ProgramBuilder, onFinalize func()) {
	s.programBuilder = program
	s.shaderStrings = make([][]byte, sectionCode+1)
	s.codeIndex = sectionCode
	s.onFinalize = onFinalize
	s.main().append("void main() {")
}

// shaderSection is a tiny convenience wrapper over the string sections.
type shaderSection struct {
	s   *ShaderBuilder
	idx int
}

func (sec shaderSection) append(str string) {
	sec.s.shaderStrings[sec.idx] = append(sec.s.shaderStrings[sec.idx], str...)
}

func (sec shaderSection) appendf(format string, args ...any) {
	sec.s.shaderStrings[sec.idx] = fmt.Appendf(sec.s.shaderStrings[sec.idx], format, args...)
}

func (s *ShaderBuilder) extensions() shaderSection {
	return shaderSection{s: s, idx: sectionExtensions}
}

func (s *ShaderBuilder) definitions() shaderSection {
	return shaderSection{s: s, idx: sectionDefinitions}
}

func (s *ShaderBuilder) layoutQualifiers() shaderSection {
	return shaderSection{s: s, idx: sectionLayoutQualifiers}
}

func (s *ShaderBuilder) outputs() shaderSection { return shaderSection{s: s, idx: sectionOutputs} }

func (s *ShaderBuilder) functions() shaderSection { return shaderSection{s: s, idx: sectionFunctions} }
func (s *ShaderBuilder) main() shaderSection      { return shaderSection{s: s, idx: sectionMain} }
func (s *ShaderBuilder) code() shaderSection      { return shaderSection{s: s, idx: s.codeIndex} }

// CodeAppend appends str to the current code section.
func (s *ShaderBuilder) CodeAppend(str string) { s.code().append(str) }

// CodeAppendf appends a formatted string to the current code section.
func (s *ShaderBuilder) CodeAppendf(format string, args ...any) {
	s.code().appendf(format, args...)
}

// DefinitionAppend appends str to the definitions section.
func (s *ShaderBuilder) DefinitionAppend(str string) { s.definitions().append(str) }

// DeclareGlobal appends a global variable declaration to the definitions section.
func (s *ShaderBuilder) DeclareGlobal(v *ShaderVar) {
	v.AppendDecl(&s.shaderStrings[sectionDefinitions])
	s.definitions().append(";")
}

// DeclAppend appends a local variable declaration to the current code section.
func (s *ShaderBuilder) DeclAppend(v ShaderVar) {
	v.AppendDecl(&s.shaderStrings[s.codeIndex])
	s.CodeAppend(";")
}

// NewTmpVarName returns a fresh, unique temporary variable name built from suffix.
func (s *ShaderBuilder) NewTmpVarName(suffix string) string {
	idx := s.tmpVariableCounter
	s.tmpVariableCounter++
	return fmt.Sprintf("_tmp_%d_%s", idx, suffix)
}

// GetMangledFunctionName returns baseName mangled with the current program-builder stage suffix, so functions from
// different processors in the same shader don't collide.
func (s *ShaderBuilder) GetMangledFunctionName(baseName string) string {
	return s.programBuilder.nameVariable(0, baseName, true)
}

// appendFunctionDecl appends a function declaration's return type, name, and parameter list (with no trailing body or
// semicolon) to the functions section.
func (s *ShaderBuilder) appendFunctionDecl(returnType GLSLType, mangledName string, args []ShaderVar) {
	s.functions().appendf("%s %s(", returnType.GLSLTypeString(), mangledName)
	for i := range args {
		if i > 0 {
			s.functions().append(", ")
		}
		args[i].AppendDecl(&s.shaderStrings[sectionFunctions])
	}
	s.functions().append(")")
}

// EmitFunction emits a complete function definition (declaration plus body) into the functions section.
func (s *ShaderBuilder) EmitFunction(returnType GLSLType, mangledName string, args []ShaderVar, body string) {
	s.appendFunctionDecl(returnType, mangledName, args)
	s.functions().appendf(" {\n%s}\n\n", body)
}

// EmitFunctionPrototype emits a function prototype (declaration only, no body) into the functions section.
func (s *ShaderBuilder) EmitFunctionPrototype(returnType GLSLType, mangledName string, args []ShaderVar) {
	s.appendFunctionDecl(returnType, mangledName, args)
	s.functions().append(";\n")
}

// InsertFunction copies a complete GLSL function definition verbatim into the functions section.
func (s *ShaderBuilder) InsertFunction(functionDefinition string) {
	s.functions().append(functionDefinition)
}

// nextStage opens a new code section, so a subsequent processor's emitted code is isolated from the current one's.
func (s *ShaderBuilder) nextStage() {
	s.shaderStrings = append(s.shaderStrings, nil)
	s.codeIndex++
}

// deleteStage discards the current code section, restoring the previous one (used after a fragment processor's code has
// been moved into a function body).
func (s *ShaderBuilder) deleteStage() {
	s.shaderStrings = s.shaderStrings[:len(s.shaderStrings)-1]
	s.codeIndex--
}

// stageCode returns the current stage's accumulated code (used when moving a stage's body into a function).
func (s *ShaderBuilder) stageCode() string { return string(s.shaderStrings[s.codeIndex]) }

// finalize concatenates the shader's sections in order into the final compiled GLSL source.
func (s *ShaderBuilder) finalize(visibility ShaderFlags) {
	if s.finalized {
		panic("shader already finalized")
	}
	// The version declaration must come first in a GLSL shader.
	header := s.programBuilder.gpu.Caps().ShaderCaps.VersionDeclString
	s.compiled = append(s.compiled, header...)
	s.programBuilder.appendUniformDecls(visibility, &s.shaderStrings[sectionUniforms])
	s.onFinalize()
	// Append the 'footer' to code.
	s.code().append("}")

	for i := 0; i <= s.codeIndex; i++ {
		s.compiled = append(s.compiled, s.shaderStrings[i]...)
	}
	s.finalized = true
}

//////////////////////////////////////////////////////////////////////////////

// VertexShaderBuilder builds the vertex shader stage: attribute inputs, the geometry processor's emitted code, and the
// final device-position write.
type VertexShaderBuilder struct {
	ShaderBuilder
	usedPosition bool
}

func newVertexShaderBuilder(program *ProgramBuilder) *VertexShaderBuilder {
	v := &VertexShaderBuilder{}
	v.initShaderBuilder(program, v.vsOnFinalize)
	return v
}

// EmitNormalizedPosition writes devPos (a float2 or float3 device position) into the local _position variable, snapping
// to pixel centers first if the pipeline requests it. The RT-adjust fixup that converts _position into gl_Position is
// appended separately, at finalize.
func (v *VertexShaderBuilder) EmitNormalizedPosition(devPos string, devPosType GLSLType) {
	v.usedPosition = true
	switch {
	case v.programBuilder.snapVerticesToPixelCenters():
		if devPosType == GLSLTypeFloat3 {
			v.CodeAppendf("{vec2 _posTmp = %s.xy / %s.z;", devPos, devPos)
		} else {
			if devPosType != GLSLTypeFloat2 {
				panic("unexpected position type")
			}
			v.CodeAppendf("{vec2 _posTmp = %s;", devPos)
		}
		v.CodeAppend("_posTmp = floor(_posTmp) + vec2(0.5);" +
			"_position = vec4(_posTmp.xy, 0.0, 1.0);}")
	case devPosType == GLSLTypeFloat3:
		v.CodeAppendf("_position = vec4(%s.xy, 0.0, %s.z);", devPos, devPos)
	default:
		if devPosType != GLSLTypeFloat2 {
			panic("unexpected position type")
		}
		v.CodeAppendf("_position = vec4(%s.xy, 0.0, 1.0);", devPos)
	}
}

func (v *VertexShaderBuilder) vsOnFinalize() {
	if v.programBuilder.hasPointSize() {
		v.CodeAppend("gl_PointSize = 1.0;")
	}
	if v.usedPosition {
		// Declare the local _position at the top of main and append the RT-adjust fixup:
		//   _position = float4(_position.xy * rtAdjust.xz + _position.ww * rtAdjust.yw, 0, _position.w);
		// followed by the write to gl_Position.
		v.main().append("vec4 _position;")
		rtAdjust := v.programBuilder.uniformHandler.GetUniformCStr(
			v.programBuilder.uniformHandles.RTAdjustmentUni,
		)
		v.CodeAppendf("gl_Position = vec4(_position.xy * %s.xz + _position.ww * %s.yw, "+
			"0.0, _position.w);", rtAdjust, rtAdjust)
	}
	v.programBuilder.varyingHandler.getVertexDecls(
		&v.shaderStrings[sectionInputs], &v.shaderStrings[sectionOutputs],
	)
}

//////////////////////////////////////////////////////////////////////////////

// FragmentShaderBuilder builds the fragment shader stage: the fragment processor tree, the transfer processor, and the
// declared color output(s). It combines what would otherwise be separate fragment-processor and transfer-processor
// builder interfaces into one concrete type.
type FragmentShaderBuilder struct {
	blendFuncsEmitted map[string]bool
	// fragOutputs holds the explicitly declared fragment outputs.
	fragOutputs []ShaderVar
	ShaderBuilder
	projEmitted        bool
	advBlendEqEnabled  bool
	hasSecondaryOutput bool
}

// declaredColorOutputName is the name of the declared primary fragment color output.
const declaredColorOutputName = "fsColorOut"

// declaredSecondaryColorOutputName is the name of the declared secondary (dual-source blending) fragment color output.
const declaredSecondaryColorOutputName = "fsSecondaryColorOut"

// fsDstColorName is the name of the local variable holding the sampled destination color.
const fsDstColorName = "_dstColor"

func newFragmentShaderBuilder(program *ProgramBuilder) *FragmentShaderBuilder {
	f := &FragmentShaderBuilder{blendFuncsEmitted: make(map[string]bool)}
	f.initShaderBuilder(program, f.fsOnFinalize)
	return f
}

// DstColor returns the expression naming the sampled destination color, for shaders that read the existing destination
// pixel (e.g. blend-mode compositing).
func (f *FragmentShaderBuilder) DstColor() string {
	shaderCaps := f.programBuilder.gpu.Caps().ShaderCaps
	if shaderCaps.FBFetchSupport {
		// EXT_shader_framebuffer_fetch lanes are ES-only; unreachable with the desktop trim.
		panic("framebuffer fetch is unreachable on desktop GL")
	}
	return fsDstColorName
}

// HasSecondaryOutput reports whether dual-source (secondary) blending output has been enabled.
func (f *FragmentShaderBuilder) HasSecondaryOutput() bool { return f.hasSecondaryOutput }

// EnableSecondaryOutput turns on the dual-source blending secondary color output, declaring it if the target requires
// explicit fragment output declarations.
func (f *FragmentShaderBuilder) EnableSecondaryOutput() {
	if f.hasSecondaryOutput {
		panic("secondary output already enabled")
	}
	f.hasSecondaryOutput = true
	// Dual-source blending is core in GL 3.3; no extension enable is needed on desktop. The secondary output must be
	// declared whenever custom outputs are in use.
	if f.programBuilder.gpu.Caps().ShaderCaps.MustDeclareFragmentShaderOutput() {
		v := NewShaderVarMod(declaredSecondaryColorOutputName, GLSLTypeHalf4, TypeModifierOut)
		f.programBuilder.finalizeFragmentSecondaryColor(&v)
		f.fragOutputs = append(f.fragOutputs, v)
	}
}

// PrimaryColorOutputName returns the name of the declared primary color output.
func (f *FragmentShaderBuilder) PrimaryColorOutputName() string { return declaredColorOutputName }

// SecondaryColorOutputName returns the name of the declared secondary color output, or "" if dual-source blending isn't
// enabled.
func (f *FragmentShaderBuilder) SecondaryColorOutputName() string {
	if f.hasSecondaryOutput {
		if !f.programBuilder.gpu.Caps().ShaderCaps.MustDeclareFragmentShaderOutput() {
			panic("GLSL 1.10 secondary outputs are unreachable on desktop")
		}
		return declaredSecondaryColorOutputName
	}
	return ""
}

func (f *FragmentShaderBuilder) fsOnFinalize() {
	// Declare the primary color output followed by any explicitly added outputs (secondary color).
	if f.programBuilder.gpu.Caps().ShaderCaps.MustDeclareFragmentShaderOutput() {
		v := NewShaderVarMod(declaredColorOutputName, GLSLTypeHalf4, TypeModifierOut)
		out := make([]byte, 0, 64+len(f.shaderStrings[sectionOutputs]))
		v.AppendDecl(&out)
		out = append(out, ";\n"...)
		f.shaderStrings[sectionOutputs] = append(out, f.shaderStrings[sectionOutputs]...)
	}
	for i := range f.fragOutputs {
		f.fragOutputs[i].AppendDecl(&f.shaderStrings[sectionOutputs])
		f.outputs().append(";\n")
	}
	f.programBuilder.varyingHandler.getFragDecls(&f.shaderStrings[sectionInputs])
}

// ensureBlendHelper emits one of the sksl_gpu.sksl helper functions (registry key) once per shader, emitting its own
// dependencies first.
func (f *FragmentShaderBuilder) ensureBlendHelper(key string) {
	if f.blendFuncsEmitted[key] {
		return
	}
	f.blendFuncsEmitted[key] = true
	if key == glslGuardedDivideFn {
		// Both overloads emit together. The epsilon folds in the MustGuardDivisionEvenAfterExplicitZeroCheck caps bit
		// (an Adreno workaround; false on every desktop-trim driver, but the caps bit is honored for faithfulness).
		eps := "0.0"
		if f.programBuilder.shaderCaps().MustGuardDivisionEvenAfterExplicitZeroCheck {
			eps = "0.00000001"
		}
		f.definitions().appendf("float _guarded_divide(float n, float d) { return n / (d + %s); }\n", eps)
		f.definitions().appendf("vec3 _guarded_divide(vec3 n, float d) { return n / (d + %s); }\n", eps)
		return
	}
	h, ok := glslBlendHelpers[key]
	if !ok {
		panic("unknown blend helper " + key)
	}
	for _, dep := range h.helpers {
		f.ensureBlendHelper(dep)
	}
	for _, dep := range h.modes {
		f.ensureBlendFunction(dep)
	}
	name := key
	switch key {
	case "blend_darken_mode":
		name = glslBlendDarkenFn
	case "blend_overlay_flip":
		name = glslBlendOverlayFn
	}
	retType := "vec4"
	switch key {
	case glslBlendOverlayComponentFn, "_color_dodge_component", "_color_burn_component",
		"_soft_light_component", "_blend_color_luminance", "_blend_color_saturation":
		retType = "float"
	case "_blend_set_color_luminance", "_blend_set_color_saturation":
		retType = "vec3"
	}
	f.definitions().appendf("%s %s(%s) {\n%s\n}\n", retType, name, h.params, h.body)
}

// ensureBlendFunction emits the dedicated GLSL function for a blend mode once per shader and returns its name (helper
// functions it calls are emitted first).
func (f *FragmentShaderBuilder) ensureBlendFunction(mode raster.BlendMode) string {
	name := glslBlendFuncName(mode)
	if !f.blendFuncsEmitted[name] {
		f.blendFuncsEmitted[name] = true
		helpers, modes := glslBlendFuncDeps(mode)
		for _, dep := range helpers {
			f.ensureBlendHelper(dep)
		}
		for _, dep := range modes {
			f.ensureBlendFunction(dep)
		}
		f.definitions().appendf("vec4 %s(vec4 src, vec4 dst) { %s }\n", name,
			glslBlendFuncBody(mode))
	}
	return name
}

// ensureSharedBlendFunction emits one of the shared uniform-driven blend helpers (blend_porter_duff,
// blend_overlay(flip,...), blend_darken(mode,...), blend_hslc) once per shader and returns its emitted name.
func (f *FragmentShaderBuilder) ensureSharedBlendFunction(fn string) string {
	switch fn {
	case "blend_porter_duff":
		if !f.blendFuncsEmitted[fn] {
			f.blendFuncsEmitted[fn] = true
			f.definitions().appendf("vec4 blend_porter_duff(vec4 blendOp, vec4 src, vec4 dst) "+
				"{\n%s\n}\n", glslPorterDuffBody)
		}
		return fn
	case glslBlendOverlayFn:
		f.ensureBlendHelper("blend_overlay_flip")
		return fn
	case glslBlendDarkenFn:
		f.ensureBlendHelper("blend_darken_mode")
		return fn
	case glslBlendHSLCFn:
		f.ensureBlendHelper(glslBlendHSLCFn)
		return fn
	default:
		panic("unknown shared blend function " + fn)
	}
}

// EnableAdvancedBlendEquationIfNeeded enables the given advanced blend equation for this shader: when the driver
// requires an explicit opt-in, requires the KHR_blend_equation_advanced extension and declares the "all equations"
// layout qualifier.
func (f *FragmentShaderBuilder) EnableAdvancedBlendEquationIfNeeded(equation gpu.BlendEquation) {
	if !equation.IsAdvanced() {
		panic("not an advanced blend equation")
	}
	if f.programBuilder.shaderCaps().AdvBlendEqInteraction ==
		gpu.GeneralEnableAdvBlendEqInteraction && !f.advBlendEqEnabled {
		f.advBlendEqEnabled = true
		f.extensions().append("#extension GL_KHR_blend_equation_advanced : require\n")
		f.layoutQualifiers().append("layout(blend_support_all_equations) out;\n")
	}
}

// rtFlipUniformName is the uniform name that gl_FragCoord and dFdy usages lower through to correct for the render
// target's y-orientation.
const rtFlipUniformName = "u_rtFlip"

// fragCoordName is the sentinel coordinate expression for sampling FragCoord-usage children; InvokeChild lowers it
// through FragmentPosition, so it never reaches the emitted shader text.
const fragCoordName = "_fragCoord.xy"

// FragmentPosition returns gl_FragCoord.xy with the RT-flip uniform applied to y, registering the uniform on first use.
func (f *FragmentShaderBuilder) FragmentPosition() string {
	pb := f.programBuilder
	if !pb.uniformHandles.RTFlipUni.IsValid() {
		pb.addRTFlipUniform(rtFlipUniformName)
	}
	return fmt.Sprintf("vec2(gl_FragCoord.x, %s.x + %s.y * gl_FragCoord.y)", rtFlipUniformName,
		rtFlipUniformName)
}

// glslSwizzleExpr applies a color-channel swizzle to a GLSL expression. Swizzles may contain the constants 0 and 1
// ("000r", "rgb1"), which raw GLSL cannot express as component selectors, so this groups consecutive component/constant
// runs into a constructor (vec4(0.0, 0.0, 0.0, (expr).r)). Every swizzle in the caps tables has at most one component
// run, so the base expression is evaluated exactly once.
func glslSwizzleExpr(expr string, swz gpu.Swizzle) string {
	if swz == gpu.SwizzleRGBA {
		return expr
	}
	s := swz.String()
	hasConstant := false
	for i := 0; i < len(s); i++ {
		if s[i] == '0' || s[i] == '1' {
			hasConstant = true
			break
		}
	}
	if !hasConstant {
		return expr + "." + s
	}
	componentRuns := 0
	var parts []string
	for i := 0; i < len(s); {
		switch s[i] {
		case '0', '1':
			j := i
			for j < len(s) && s[j] == s[i] {
				j++
			}
			lit := "0.0"
			if s[i] == '1' {
				lit = "1.0"
			}
			for ; i < j; i++ {
				parts = append(parts, lit)
			}
		default:
			j := i
			for j < len(s) && s[j] != '0' && s[j] != '1' {
				j++
			}
			parts = append(parts, fmt.Sprintf("(%s).%s", expr, s[i:j]))
			componentRuns++
			i = j
		}
	}
	if componentRuns > 1 {
		// Would evaluate expr more than once; no reachable swizzle does this.
		panic("swizzle with multiple component runs: " + s)
	}
	return fmt.Sprintf("vec4(%s)", strings.Join(parts, ", "))
}

// AppendTextureLookup returns the expression that samples the texture at coord, with the sampler's read swizzle
// applied.
func (f *FragmentShaderBuilder) AppendTextureLookup(handler *UniformHandler, handle SamplerHandle, coord string) string {
	return glslSwizzleExpr(fmt.Sprintf("texture(%s, %s)", handler.SamplerVariable(handle), coord),
		handler.SamplerSwizzle(handle))
}

// ensureProjFunction emits the shared proj() helper (perspective divide) once per shader.
func (f *FragmentShaderBuilder) ensureProjFunction() {
	if !f.projEmitted {
		f.projEmitted = true
		f.definitions().append("vec2 proj(vec3 p) { return p.xy / p.z; }\n")
	}
}
