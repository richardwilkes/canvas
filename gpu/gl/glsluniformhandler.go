// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The uniform handler for shader programs: uniforms are declared with mangled names, bound (or queried) to locations at
// link, and identified afterwards by handle index.

package gl

import "github.com/richardwilkes/canvas/gpu"

// UniformHandle identifies a uniform previously added to a UniformHandler. The zero value is invalid; indices are
// stored +1.
type UniformHandle int32

// IsValid reports whether h refers to a real uniform.
func (h UniformHandle) IsValid() bool { return h > 0 }

func (h UniformHandle) toIndex() int {
	if !h.IsValid() {
		panic("invalid uniform handle")
	}
	return int(h) - 1
}

func uniformHandleFromIndex(idx int) UniformHandle { return UniformHandle(idx + 1) }

// SamplerHandle identifies a sampler previously added to a UniformHandler, with the same +1 convention as
// UniformHandle.
type SamplerHandle int32

// IsValid reports whether h refers to a real sampler.
func (h SamplerHandle) IsValid() bool { return h > 0 }

func (h SamplerHandle) toIndex() int {
	if !h.IsValid() {
		panic("invalid sampler handle")
	}
	return int(h) - 1
}

// unusedUniformLocation marks a uniform whose GL location has not yet been resolved.
const unusedUniformLocation = -1

// UniformInfo records one declared uniform's shader variable, owning processor, pre-mangling name, visibility, and
// resolved GL location.
type UniformInfo struct {
	Owner      Processor
	RawName    string
	Variable   ShaderVar
	Visibility ShaderFlags
	Location   int32
}

// UniformHandler is the merged uniform handler.
type UniformHandler struct {
	programBuilder  *ProgramBuilder
	uniforms        []UniformInfo
	samplers        []UniformInfo
	samplerSwizzles []gpu.Swizzle
}

func newUniformHandler(program *ProgramBuilder) *UniformHandler {
	return &UniformHandler{programBuilder: program}
}

// NumUniforms returns the number of uniforms added so far.
func (u *UniformHandler) NumUniforms() int { return len(u.uniforms) }

// Uniform returns the uniform at idx.
func (u *UniformHandler) Uniform(idx int) *UniformInfo { return &u.uniforms[idx] }

// GetUniformVariable returns the shader variable for handle h.
func (u *UniformHandler) GetUniformVariable(h UniformHandle) *ShaderVar {
	return &u.uniforms[h.toIndex()].Variable
}

// GetUniformCStr returns the mangled shader-visible name for handle h.
func (u *UniformHandler) GetUniformCStr(h UniformHandle) string {
	return u.GetUniformVariable(h).Name()
}

// AddUniform adds a (non-array) uniform visible in one or more shader stages and returns its handle and final (mangled)
// name.
func (u *UniformHandler) AddUniform(owner Processor, visibility ShaderFlags, t GLSLType, name string) (handle UniformHandle, mangledName string) {
	if t.IsCombinedSamplerType() {
		panic("use AddSampler for sampler types")
	}
	return u.AddUniformArray(owner, visibility, t, name, 0)
}

// AddUniformArray adds an array uniform visible in one or more shader stages and returns its handle and final (mangled)
// name.
func (u *UniformHandler) AddUniformArray(owner Processor, visibility ShaderFlags, t GLSLType, name string, arrayCount int) (handle UniformHandle, mangledName string) {
	if t.IsCombinedSamplerType() {
		panic("use AddSampler for sampler types")
	}
	return u.internalAddUniformArray(owner, visibility, t, name, true, arrayCount)
}

// internalAddUniformArray records a new uniform (scalar when arrayCount is 0) and returns its handle and resolved
// shader-visible name.
func (u *UniformHandler) internalAddUniformArray(owner Processor, visibility ShaderFlags, t GLSLType, name string, mangleName bool, arrayCount int) (handle UniformHandle, mangledName string) {
	if name == "" {
		panic("uniform name required")
	}
	if visibility == 0 {
		panic("uniform visibility required")
	}

	// Names that already carry the 'u' prefix (uniform view matrices, the program-wide RT-adjust and RT-flip uniforms)
	// keep their names as-is; other names get a 'u' prefix.
	prefix := byte('u')
	if name[0] == 'u' {
		prefix = 0
	}
	resolvedName := u.programBuilder.nameVariable(prefix, name, mangleName)

	u.uniforms = append(u.uniforms, UniformInfo{
		Variable:   NewShaderVarArray(resolvedName, t, TypeModifierUniform, arrayCount),
		Visibility: visibility,
		Owner:      owner,
		RawName:    name,
		Location:   unusedUniformLocation,
	})
	h := uniformHandleFromIndex(len(u.uniforms) - 1)
	return h, resolvedName
}

// AddSampler adds a sampler uniform for the given texture target and returns its handle and mangled name.
func (u *UniformHandler) AddSampler(target uint32, swizzle gpu.Swizzle, name string) (handle SamplerHandle, mangledName string) {
	if name == "" {
		panic("sampler name required")
	}
	mangleName := u.programBuilder.nameVariable('u', name, true)

	u.samplers = append(u.samplers, UniformInfo{
		Variable: NewShaderVarMod(mangleName, CombinedSamplerTypeForTextureType(target),
			TypeModifierUniform),
		Visibility: ShaderFlagFragment,
		RawName:    name,
		Location:   unusedUniformLocation,
	})
	u.samplerSwizzles = append(u.samplerSwizzles, swizzle)
	return SamplerHandle(len(u.samplers)), mangleName
}

// SamplerVariable returns the mangled shader-visible name for sampler handle h.
func (u *UniformHandler) SamplerVariable(h SamplerHandle) string {
	return u.samplers[h.toIndex()].Variable.Name()
}

// SamplerSwizzle returns the read swizzle recorded for sampler handle h.
func (u *UniformHandler) SamplerSwizzle(h SamplerHandle) gpu.Swizzle {
	return u.samplerSwizzles[h.toIndex()]
}

// getUniformMapping looks up a uniform added by 'owner' with the given (pre-mangling) name; returns a void variable
// when absent.
func (u *UniformHandler) getUniformMapping(owner Processor, rawName string) ShaderVar {
	for i := u.NumUniforms() - 1; i >= 0; i-- {
		uni := &u.uniforms[i]
		if uni.Owner == owner && uni.RawName == rawName {
			return uni.Variable
		}
	}
	return ShaderVar{varType: GLSLTypeVoid}
}

// liftUniformToVertexShader looks up a uniform added by 'owner' with the given (pre-mangling) name and marks it visible
// to the vertex stage too, so a fragment-only uniform can be sampled from a lifted vertex-shader computation.
func (u *UniformHandler) liftUniformToVertexShader(owner Processor, rawName string) ShaderVar {
	for i := u.NumUniforms() - 1; i >= 0; i-- {
		uni := &u.uniforms[i]
		if uni.Owner == owner && uni.RawName == rawName {
			uni.Visibility |= ShaderFlagVertex
			return uni.Variable
		}
	}
	return ShaderVar{varType: GLSLTypeVoid}
}

// appendUniformDecls writes GLSL declarations for every uniform and sampler visible in the given shader stage(s) to
// out.
func (u *UniformHandler) appendUniformDecls(visibility ShaderFlags, out *[]byte) {
	for i := range u.uniforms {
		if u.uniforms[i].Visibility&visibility != 0 {
			u.uniforms[i].Variable.AppendDecl(out)
			*out = append(*out, ";"...)
		}
	}
	for i := range u.samplers {
		if u.samplers[i].Visibility&visibility != 0 {
			u.samplers[i].Variable.AppendDecl(out)
			*out = append(*out, ";\n"...)
		}
	}
}

// bindUniformLocations is a no-op that keeps the call-site shape: desktop GL has no bindUniformLocation extension in
// the trimmed caps matrix, so uniform locations are always resolved after link instead.
func (u *UniformHandler) bindUniformLocations(programID uint32, caps *Caps) {
	_ = programID
	_ = caps
}

// getUniformLocations resolves the GL location of every uniform and sampler after program link.
func (u *UniformHandler) getUniformLocations(programID uint32, g *Gpu) {
	for i := range u.uniforms {
		u.uniforms[i].Location = g.fns().GetUniformLocation(programID,
			glStr(u.uniforms[i].Variable.Name()))
	}
	for i := range u.samplers {
		u.samplers[i].Location = g.fns().GetUniformLocation(programID,
			glStr(u.samplers[i].Variable.Name()))
	}
}

// BuiltinUniformHandles holds the handles for the uniforms every program may need regardless of its fragment-processor
// DAG.
type BuiltinUniformHandles struct {
	// RTAdjustmentUni adjusts from device space to normalized device coordinates.
	RTAdjustmentUni UniformHandle
	// RTFlipUni is used for dFdy and gl_FragCoord with bottom-left origins.
	RTFlipUni UniformHandle
	// DstTextureCoordsUni holds the destination texture origin and scale for dst reads.
	DstTextureCoordsUni UniformHandle
}
