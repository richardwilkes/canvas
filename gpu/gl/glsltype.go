// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GLSLType enumerates the abstract shader value types used throughout the shader-building code, and ShaderVar
// represents a single shader variable declaration. This package emits GLSL directly, so GLSLTypeString returns the
// *GLSL* spelling of each type — half maps to float and half4 to vec4, matching a desktop GLSL compile with no
// precision qualifiers.

package gl

import "strconv"

// GLSLType is an abstract shader value type, translated to a concrete GLSL type string by GLSLTypeString.
type GLSLType int8

// GLSLType values.
const (
	GLSLTypeVoid GLSLType = iota
	GLSLTypeBool
	GLSLTypeBool2
	GLSLTypeBool3
	GLSLTypeBool4
	GLSLTypeShort
	GLSLTypeShort2
	GLSLTypeShort3
	GLSLTypeShort4
	GLSLTypeUShort
	GLSLTypeUShort2
	GLSLTypeUShort3
	GLSLTypeUShort4
	GLSLTypeFloat
	GLSLTypeFloat2
	GLSLTypeFloat3
	GLSLTypeFloat4
	GLSLTypeFloat2x2
	GLSLTypeFloat3x3
	GLSLTypeFloat4x4
	GLSLTypeHalf
	GLSLTypeHalf2
	GLSLTypeHalf3
	GLSLTypeHalf4
	GLSLTypeHalf2x2
	GLSLTypeHalf3x3
	GLSLTypeHalf4x4
	GLSLTypeInt
	GLSLTypeInt2
	GLSLTypeInt3
	GLSLTypeInt4
	GLSLTypeUInt
	GLSLTypeUInt2
	GLSLTypeUInt3
	GLSLTypeUInt4
	GLSLTypeTexture2DSampler
	GLSLTypeTextureExternalSampler
	GLSLTypeTexture2DRectSampler
	GLSLTypeTexture2D
	GLSLTypeSampler
	GLSLTypeInput

	// GLSLTypeCount is the number of defined GLSLType values.
	GLSLTypeCount = int(GLSLTypeInput) + 1
)

// GLSLTypeString returns the GLSL spelling of the type (half types become float types on desktop, since there's no
// separate half precision in desktop GLSL).
func (t GLSLType) GLSLTypeString() string {
	switch t {
	case GLSLTypeVoid:
		return "void"
	case GLSLTypeBool:
		return "bool"
	case GLSLTypeBool2:
		return "bvec2"
	case GLSLTypeBool3:
		return "bvec3"
	case GLSLTypeBool4:
		return "bvec4"
	case GLSLTypeShort, GLSLTypeInt:
		return "int"
	case GLSLTypeShort2, GLSLTypeInt2:
		return "ivec2"
	case GLSLTypeShort3, GLSLTypeInt3:
		return "ivec3"
	case GLSLTypeShort4, GLSLTypeInt4:
		return "ivec4"
	case GLSLTypeUShort, GLSLTypeUInt:
		return "uint"
	case GLSLTypeUShort2, GLSLTypeUInt2:
		return "uvec2"
	case GLSLTypeUShort3, GLSLTypeUInt3:
		return "uvec3"
	case GLSLTypeUShort4, GLSLTypeUInt4:
		return "uvec4"
	case GLSLTypeFloat, GLSLTypeHalf:
		return "float"
	case GLSLTypeFloat2, GLSLTypeHalf2:
		return "vec2"
	case GLSLTypeFloat3, GLSLTypeHalf3:
		return "vec3"
	case GLSLTypeFloat4, GLSLTypeHalf4:
		return "vec4"
	case GLSLTypeFloat2x2, GLSLTypeHalf2x2:
		return "mat2"
	case GLSLTypeFloat3x3, GLSLTypeHalf3x3:
		return "mat3"
	case GLSLTypeFloat4x4, GLSLTypeHalf4x4:
		return "mat4"
	case GLSLTypeTexture2DSampler:
		return "sampler2D"
	case GLSLTypeTexture2DRectSampler:
		return "sampler2DRect"
	default:
		panic("unsupported GLSLType in GLSL emission")
	}
}

// IsFloatType reports whether t is one of the float or half scalar/vector/matrix types.
func (t GLSLType) IsFloatType() bool {
	switch t {
	case GLSLTypeFloat, GLSLTypeFloat2, GLSLTypeFloat3, GLSLTypeFloat4,
		GLSLTypeFloat2x2, GLSLTypeFloat3x3, GLSLTypeFloat4x4,
		GLSLTypeHalf, GLSLTypeHalf2, GLSLTypeHalf3, GLSLTypeHalf4,
		GLSLTypeHalf2x2, GLSLTypeHalf3x3, GLSLTypeHalf4x4:
		return true
	default:
		return false
	}
}

// VecLength returns the number of components in a float vector (or 1 for a scalar), -1 otherwise.
func (t GLSLType) VecLength() int {
	switch t {
	case GLSLTypeFloat, GLSLTypeHalf:
		return 1
	case GLSLTypeFloat2, GLSLTypeHalf2:
		return 2
	case GLSLTypeFloat3, GLSLTypeHalf3:
		return 3
	case GLSLTypeFloat4, GLSLTypeHalf4:
		return 4
	default:
		return -1
	}
}

// IsCombinedSamplerType reports whether t is a combined texture-and-sampler type.
func (t GLSLType) IsCombinedSamplerType() bool {
	switch t {
	case GLSLTypeTexture2DSampler, GLSLTypeTextureExternalSampler, GLSLTypeTexture2DRectSampler:
		return true
	default:
		return false
	}
}

// CombinedSamplerTypeForTextureType returns the combined sampler type for the given GL texture target.
func CombinedSamplerTypeForTextureType(target uint32) GLSLType {
	switch target {
	case TEXTURE_2D:
		return GLSLTypeTexture2DSampler
	case TEXTURE_RECTANGLE:
		return GLSLTypeTexture2DRectSampler
	default:
		panic("unexpected texture target")
	}
}

// ShaderFlags is a bitmask of shader stages.
type ShaderFlags uint32

// ShaderFlags values.
const (
	ShaderFlagNone     ShaderFlags = 0
	ShaderFlagVertex   ShaderFlags = 1 << 0
	ShaderFlagFragment ShaderFlags = 1 << 1
)

// ShaderVarTypeModifier is the storage qualifier (in/out/inout/uniform) applied to a shader variable declaration.
type ShaderVarTypeModifier int8

// ShaderVarTypeModifier values.
const (
	TypeModifierNone ShaderVarTypeModifier = iota
	TypeModifierOut
	TypeModifierIn
	TypeModifierInOut
	TypeModifierUniform
)

// ShaderVarNonArray is the arrayCount value denoting a non-array shader variable.
const ShaderVarNonArray = 0

// ShaderVar is a variable declaration in a shader: a name, type, optional array count, storage qualifier, and extra
// layout/modifier text.
type ShaderVar struct {
	name         string
	layoutQuals  string
	extraMods    string
	varType      GLSLType
	typeModifier ShaderVarTypeModifier
	arrayCount   int
}

// NewShaderVar creates a shader variable declaration with just a name and type.
func NewShaderVar(name string, t GLSLType) ShaderVar {
	return ShaderVar{name: name, varType: t}
}

// NewShaderVarMod creates a shader variable declaration with a name, type, and storage qualifier.
func NewShaderVarMod(name string, t GLSLType, mod ShaderVarTypeModifier) ShaderVar {
	return ShaderVar{name: name, varType: t, typeModifier: mod}
}

// NewShaderVarArray creates an array shader variable declaration.
func NewShaderVarArray(name string, t GLSLType, mod ShaderVarTypeModifier, arrayCount int) ShaderVar {
	return ShaderVar{name: name, varType: t, typeModifier: mod, arrayCount: arrayCount}
}

// NewShaderVarExtras creates a shader variable declaration with layout qualifiers and extra modifier text.
func NewShaderVarExtras(name string, t GLSLType, mod ShaderVarTypeModifier, arrayCount int, layoutQuals, extraMods string) ShaderVar {
	return ShaderVar{
		name: name, varType: t, typeModifier: mod, arrayCount: arrayCount,
		layoutQuals: layoutQuals, extraMods: extraMods,
	}
}

// Set replaces the variable's type and name.
func (v *ShaderVar) Set(t GLSLType, name string) {
	v.varType = t
	v.name = name
}

// Name returns the variable's name.
func (v *ShaderVar) Name() string { return v.name }

// Type returns the variable's type.
func (v *ShaderVar) Type() GLSLType { return v.varType }

// TypeModifier returns the variable's storage qualifier.
func (v *ShaderVar) TypeModifier() ShaderVarTypeModifier { return v.typeModifier }

// ArrayCount returns the variable's array length, or ShaderVarNonArray if it isn't an array.
func (v *ShaderVar) ArrayCount() int { return v.arrayCount }

// IsArray reports whether the variable is an array.
func (v *ShaderVar) IsArray() bool { return v.arrayCount != ShaderVarNonArray }

// AppendDecl appends the variable's GLSL declaration to out (layout quals, extra modifiers, type modifier, type, name,
// and array suffix).
func (v *ShaderVar) AppendDecl(out *[]byte) {
	if v.layoutQuals != "" {
		*out = append(*out, "layout("...)
		*out = append(*out, v.layoutQuals...)
		*out = append(*out, ") "...)
	}
	if v.extraMods != "" {
		*out = append(*out, v.extraMods...)
		*out = append(*out, ' ')
	}
	switch v.typeModifier {
	case TypeModifierNone:
	case TypeModifierOut:
		*out = append(*out, "out "...)
	case TypeModifierIn:
		*out = append(*out, "in "...)
	case TypeModifierInOut:
		*out = append(*out, "inout "...)
	case TypeModifierUniform:
		*out = append(*out, "uniform "...)
	}
	*out = append(*out, v.varType.GLSLTypeString()...)
	*out = append(*out, ' ')
	*out = append(*out, v.name...)
	if v.IsArray() {
		*out = append(*out, '[')
		*out = append(*out, strconv.Itoa(v.arrayCount)...)
		*out = append(*out, ']')
	}
}
