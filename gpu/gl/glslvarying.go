// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file manages varyings: values written by the vertex shader and read by the fragment shader. Scope is always
// vertex-to-fragment, since there is no geometry shader stage.

package gl

// Varying is a value written by the vertex shader and read by the fragment shader, together with its resolved GLSL
// variable names on each side.
type Varying struct {
	vsOut   string
	fsIn    string
	varType GLSLType
}

// NewVarying creates a varying of the given type; it must be registered with a VaryingHandler before its names can be
// used.
func NewVarying(t GLSLType) Varying { return Varying{varType: t} }

// Type returns the varying's GLSL type.
func (v *Varying) Type() GLSLType { return v.varType }

// VsOut returns the vertex-shader-side variable name.
func (v *Varying) VsOut() string {
	if v.vsOut == "" {
		panic("varying not registered")
	}
	return v.vsOut
}

// FsIn returns the fragment-shader-side variable name.
func (v *Varying) FsIn() string {
	if v.fsIn == "" {
		panic("varying not registered")
	}
	return v.fsIn
}

// VsOutVar returns the varying as an "out" shader variable declaration for the vertex shader.
func (v *Varying) VsOutVar() ShaderVar {
	return NewShaderVarMod(v.VsOut(), v.varType, TypeModifierOut)
}

// FsInVar returns the varying as an "in" shader variable declaration for the fragment shader.
func (v *Varying) FsInVar() ShaderVar {
	return NewShaderVarMod(v.FsIn(), v.varType, TypeModifierIn)
}

// VaryingInterpolation selects how a varying is interpolated across a primitive.
type VaryingInterpolation int32

// VaryingInterpolation values.
const (
	InterpolationInterpolated VaryingInterpolation = iota
	InterpolationCanBeFlat                         // use "flat" if it will be faster
	InterpolationMustBeFlat                        // use "flat" even if it is known to be slow
)

// varyingInfo records one registered varying's vertex-shader name, type, stage visibility, and whether it uses flat
// interpolation.
type varyingInfo struct {
	vsOut      string
	visibility ShaderFlags
	varType    GLSLType
	isFlat     bool
}

// VaryingHandler tracks the varyings and attributes declared for a program, and emits their vertex/fragment shader
// declarations.
type VaryingHandler struct {
	programBuilder               *ProgramBuilder
	defaultInterpolationModifier string
	varyings                     []varyingInfo
	vertexInputs                 []ShaderVar
	vertexOutputs                []ShaderVar
	fragInputs                   []ShaderVar
	fragOutputs                  []ShaderVar
}

func newVaryingHandler(program *ProgramBuilder) *VaryingHandler {
	return &VaryingHandler{programBuilder: program}
}

// useFlatInterpolation resolves an interpolation request against the driver's capabilities, panicking if flat
// interpolation is required but unsupported.
func (h *VaryingHandler) useFlatInterpolation(interpolation VaryingInterpolation) bool {
	caps := h.programBuilder.gpu.Caps().ShaderCaps
	switch interpolation {
	case InterpolationInterpolated:
		return false
	case InterpolationCanBeFlat:
		return caps.PreferFlatInterpolation
	case InterpolationMustBeFlat:
		if !caps.FlatInterpolationSupport {
			panic("flat interpolation required but unsupported")
		}
		return true
	}
	panic("invalid interpolation")
}

// AddVarying registers varying with default (smooth) interpolation.
func (h *VaryingHandler) AddVarying(name string, varying *Varying) {
	h.AddVaryingWithInterpolation(name, varying, InterpolationInterpolated)
}

// AddVaryingWithInterpolation registers varying under a mangled version of name, resolving its interpolation mode and
// assigning its vertex/fragment-shader variable names.
func (h *VaryingHandler) AddVaryingWithInterpolation(name string, varying *Varying, interpolation VaryingInterpolation) {
	if !varying.varType.IsFloatType() && interpolation != InterpolationMustBeFlat {
		panic("integer varyings must use flat interpolation")
	}
	if varying.varType == GLSLTypeVoid {
		panic("varying requires a type")
	}
	v := varyingInfo{
		varType:    varying.varType,
		isFlat:     h.useFlatInterpolation(interpolation),
		vsOut:      h.programBuilder.nameVariable('v', name, true),
		visibility: ShaderFlagVertex | ShaderFlagFragment,
	}
	varying.vsOut = v.vsOut
	varying.fsIn = v.vsOut
	h.varyings = append(h.varyings, v)
}

// AddPassThroughAttribute passes a vertex attribute through to 'output' in the fragment shader (which must already be
// declared there).
func (h *VaryingHandler) AddPassThroughAttribute(vsVar ShaderVar, output string, interpolation VaryingInterpolation) {
	if vsVar.Type() == GLSLTypeVoid {
		panic("attribute requires a type")
	}
	v := NewVarying(vsVar.Type())
	h.AddVaryingWithInterpolation(vsVar.Name(), &v, interpolation)
	h.programBuilder.vs.CodeAppendf("%s = %s;", v.VsOut(), vsVar.Name())
	h.programBuilder.fs.CodeAppendf("%s = %s;", output, v.FsIn())
}

// EmitAttributes registers a geometry processor's vertex and instance attributes as vertex-shader inputs.
func (h *VaryingHandler) EmitAttributes(gp GeometryProcessor) {
	for _, attr := range gp.gpBase().vertexAttributes.All() {
		h.addAttribute(attr.AsShaderVar())
	}
	for _, attr := range gp.gpBase().instanceAttributes.All() {
		h.addAttribute(attr.AsShaderVar())
	}
}

// addAttribute records v as a vertex-shader input, skipping it if already added.
func (h *VaryingHandler) addAttribute(v ShaderVar) {
	if v.TypeModifier() != TypeModifierIn {
		panic("attributes must be inputs")
	}
	for i := range h.vertexInputs {
		// If the attribute was already added, don't add it again.
		if h.vertexInputs[i].Name() == v.Name() {
			return
		}
	}
	h.vertexInputs = append(h.vertexInputs, v)
}

// SetNoPerspective requests "noperspective" interpolation for varyings by default, when the driver supports it. On
// desktop core profiles this is available from GLSL 1.30 without an extension.
func (h *VaryingHandler) SetNoPerspective() {
	if !h.programBuilder.gpu.Caps().ShaderCaps.NoPerspectiveInterpolationSupport {
		return
	}
	h.defaultInterpolationModifier = "noperspective"
}

// finish resolves each registered varying's interpolation modifier and builds its vertex-output and fragment-input
// shader variable declarations. Named finish rather than finalize to avoid clashing with the shader builders' finalize.
func (h *VaryingHandler) finish() {
	for i := range h.varyings {
		v := &h.varyings[i]
		modifier := h.defaultInterpolationModifier
		if v.isFlat {
			modifier = "flat"
		}
		if v.visibility&ShaderFlagVertex != 0 {
			h.vertexOutputs = append(h.vertexOutputs, NewShaderVarExtras(v.vsOut, v.varType,
				TypeModifierOut, ShaderVarNonArray, "", modifier))
		}
		if v.visibility&ShaderFlagFragment != 0 {
			h.fragInputs = append(h.fragInputs, NewShaderVarExtras(v.vsOut, v.varType,
				TypeModifierIn, ShaderVarNonArray, "", modifier))
		}
	}
}

// appendDecls appends each variable's declaration, semicolon-terminated, to out.
func (h *VaryingHandler) appendDecls(vars []ShaderVar, out *[]byte) {
	for i := range vars {
		vars[i].AppendDecl(out)
		*out = append(*out, ";"...)
	}
}

// getVertexDecls appends the vertex shader's input and output variable declarations to inputDecls and outputDecls
// respectively.
func (h *VaryingHandler) getVertexDecls(inputDecls, outputDecls *[]byte) {
	h.appendDecls(h.vertexInputs, inputDecls)
	h.appendDecls(h.vertexOutputs, outputDecls)
}

// getFragDecls appends the fragment shader's input and output variable declarations to inputDecls and outputDecls
// respectively.
func (h *VaryingHandler) getFragDecls(inputDecls, outputDecls *[]byte) {
	h.appendDecls(h.fragInputs, inputDecls)
	h.appendDecls(h.fragOutputs, outputDecls)
}
