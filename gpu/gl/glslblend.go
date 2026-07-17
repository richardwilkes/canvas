// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file provides the GLSL blend-mode functions (coefficient and advanced Porter-Duff blends) as raw GLSL source,
// since they have no built-in equivalent. The fragment builder owns a registry (see glslBlendHelpers and
// ensureBlendHelper/ensureBlendFunction in glslshaderbuilder.go) that emits each needed function's GLSL definition once
// per shader (helper functions first). Private helper names use a '_' prefix since GLSL identifiers cannot contain '$'.

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/raster"
)

// glslBlendFuncName returns the name of the dedicated GLSL function that implements mode.
func glslBlendFuncName(mode raster.BlendMode) string {
	switch mode {
	case raster.BlendClear:
		return "blend_clear"
	case raster.BlendSrc:
		return "blend_src"
	case raster.BlendDst:
		return "blend_dst"
	case raster.BlendSrcOver:
		return "blend_src_over"
	case raster.BlendDstOver:
		return "blend_dst_over"
	case raster.BlendSrcIn:
		return "blend_src_in"
	case raster.BlendDstIn:
		return "blend_dst_in"
	case raster.BlendSrcOut:
		return "blend_src_out"
	case raster.BlendDstOut:
		return "blend_dst_out"
	case raster.BlendSrcATop:
		return "blend_src_atop"
	case raster.BlendDstATop:
		return "blend_dst_atop"
	case raster.BlendXor:
		return "blend_xor"
	case raster.BlendPlus:
		return "blend_plus"
	case raster.BlendModulate:
		return "blend_modulate"
	case raster.BlendScreen:
		return "blend_screen"
	case raster.BlendOverlay:
		return glslBlendOverlayFn
	case raster.BlendDarken:
		return glslBlendDarkenFn
	case raster.BlendLighten:
		return "blend_lighten"
	case raster.BlendColorDodge:
		return "blend_color_dodge"
	case raster.BlendColorBurn:
		return "blend_color_burn"
	case raster.BlendHardLight:
		return "blend_hard_light"
	case raster.BlendSoftLight:
		return "blend_soft_light"
	case raster.BlendDifference:
		return "blend_difference"
	case raster.BlendExclusion:
		return "blend_exclusion"
	case raster.BlendMultiply:
		return "blend_multiply"
	case raster.BlendHue:
		return "blend_hue"
	case raster.BlendSaturation:
		return "blend_saturation"
	case raster.BlendColor:
		return "blend_color"
	case raster.BlendLuminosity:
		return "blend_luminosity"
	}
	panic("unknown blend mode")
}

// glslBlendFuncBody returns the GLSL body for a blend function implementing the standard Porter-Duff/blend-mode formula
// for mode (src and dst are vec4 parameters). glslBlendFuncDeps lists the helper functions each body calls.
func glslBlendFuncBody(mode raster.BlendMode) string {
	switch mode {
	case raster.BlendClear:
		return "return vec4(0.0);"
	case raster.BlendSrc:
		return "return src;"
	case raster.BlendDst:
		return "return dst;"
	case raster.BlendSrcOver:
		return "return src + (1.0 - src.a)*dst;"
	case raster.BlendDstOver:
		return "return (1.0 - dst.a)*src + dst;"
	case raster.BlendSrcIn:
		return "return src*dst.a;"
	case raster.BlendDstIn:
		return "return dst*src.a;"
	case raster.BlendSrcOut:
		return "return (1.0 - dst.a)*src;"
	case raster.BlendDstOut:
		return "return (1.0 - src.a)*dst;"
	case raster.BlendSrcATop:
		return "return dst.a*src + (1.0 - src.a)*dst;"
	case raster.BlendDstATop:
		return "return (1.0 - dst.a)*src + src.a*dst;"
	case raster.BlendXor:
		return "return (1.0 - dst.a)*src + (1.0 - src.a)*dst;"
	case raster.BlendPlus:
		return "return min(src + dst, 1.0);"
	case raster.BlendModulate:
		return "return src*dst;"
	case raster.BlendScreen:
		return "return src + (1.0 - src)*dst;"
	case raster.BlendOverlay:
		return "vec4 result = vec4(_blend_overlay_component(src.ra, dst.ra),\n" +
			"                   _blend_overlay_component(src.ga, dst.ga),\n" +
			"                   _blend_overlay_component(src.ba, dst.ba),\n" +
			"                   src.a + (1.0 - src.a)*dst.a);\n" +
			"result.rgb += dst.rgb*(1.0 - src.a) + src.rgb*(1.0 - dst.a);\n" +
			"return result;"
	case raster.BlendDarken:
		return "return blend_darken_mode(1.0, src, dst);"
	case raster.BlendLighten:
		return "vec4 result = blend_src_over(src, dst);\n" +
			"result.rgb = max(result.rgb, (1.0 - dst.a)*src.rgb + dst.rgb);\n" +
			"return result;"
	case raster.BlendColorDodge:
		return "return vec4(_color_dodge_component(src.ra, dst.ra),\n" +
			"            _color_dodge_component(src.ga, dst.ga),\n" +
			"            _color_dodge_component(src.ba, dst.ba),\n" +
			"            src.a + (1.0 - src.a)*dst.a);"
	case raster.BlendColorBurn:
		return "return vec4(_color_burn_component(src.ra, dst.ra),\n" +
			"            _color_burn_component(src.ga, dst.ga),\n" +
			"            _color_burn_component(src.ba, dst.ba),\n" +
			"            src.a + (1.0 - src.a)*dst.a);"
	case raster.BlendHardLight:
		return "return blend_overlay(dst, src);"
	case raster.BlendSoftLight:
		return "return (dst.a == 0.0) ? src : vec4(_soft_light_component(src.ra, dst.ra),\n" +
			"                                    _soft_light_component(src.ga, dst.ga),\n" +
			"                                    _soft_light_component(src.ba, dst.ba),\n" +
			"                                    src.a + (1.0 - src.a)*dst.a);"
	case raster.BlendDifference:
		return "return vec4(src.rgb + dst.rgb - 2.0*min(src.rgb*dst.a, dst.rgb*src.a),\n" +
			"            src.a + (1.0 - src.a)*dst.a);"
	case raster.BlendExclusion:
		return "return vec4(dst.rgb + src.rgb - 2.0*dst.rgb*src.rgb, src.a + (1.0 - src.a)*dst.a);"
	case raster.BlendMultiply:
		return "return vec4((1.0 - src.a)*dst.rgb + (1.0 - dst.a)*src.rgb + src.rgb*dst.rgb,\n" +
			"            src.a + (1.0 - src.a)*dst.a);"
	case raster.BlendHue:
		return "return blend_hslc(vec2(0.0, 1.0), src, dst);"
	case raster.BlendSaturation:
		return "return blend_hslc(vec2(1.0), src, dst);"
	case raster.BlendColor:
		return "return blend_hslc(vec2(0.0), src, dst);"
	case raster.BlendLuminosity:
		return "return blend_hslc(vec2(1.0, 0.0), src, dst);"
	}
	panic("unknown blend mode")
}

// glslBlendFuncDeps returns the helper functions (and blend-mode functions) a blend function body calls, in emission
// order.
func glslBlendFuncDeps(mode raster.BlendMode) (helpers []string, modes []raster.BlendMode) {
	switch mode {
	case raster.BlendOverlay:
		return []string{glslBlendOverlayComponentFn}, nil
	case raster.BlendDarken:
		return []string{"blend_darken_mode"}, nil
	case raster.BlendLighten:
		return nil, []raster.BlendMode{raster.BlendSrcOver}
	case raster.BlendColorDodge:
		return []string{glslGuardedDivideFn, "_color_dodge_component"}, nil
	case raster.BlendColorBurn:
		return []string{glslGuardedDivideFn, "_color_burn_component"}, nil
	case raster.BlendHardLight:
		return nil, []raster.BlendMode{raster.BlendOverlay}
	case raster.BlendSoftLight:
		return []string{glslGuardedDivideFn, "_soft_light_component"}, nil
	case raster.BlendHue, raster.BlendSaturation, raster.BlendColor, raster.BlendLuminosity:
		return []string{glslBlendHSLCFn}, nil
	default:
		return nil, nil
	}
}

// glslMinNormalHalf is 1.0 / (1 << 14), the smallest positive normal half-precision float: a denormal-half guard used
// before half-precision divides.
const glslMinNormalHalf = "0.00006103515625"

// glslBlendHelper describes one private helper function (each named with a '_' prefix) or shared multi-mode blend
// function used by the blend-mode GLSL bodies.
type glslBlendHelper struct {
	params  string
	body    string
	helpers []string           // helper functions this one calls
	modes   []raster.BlendMode // blend-mode functions this one calls
}

// GLSL blend helper-function names and the shared component-helper parameter list, used by both the emitter and the
// blend-mode tables.
const (
	glslBlendOverlayFn          = "blend_overlay"
	glslBlendDarkenFn           = "blend_darken"
	glslBlendHSLCFn             = "blend_hslc"
	glslBlendOverlayComponentFn = "_blend_overlay_component"
	glslGuardedDivideFn         = "_guarded_divide"
	glslBlendComponentParams    = "vec2 s, vec2 d"
)

// glslOpaqueWhite is the default input/dest color expression when none is supplied.
const glslOpaqueWhite = "vec4(1)"

// glslBlendHelpers is the registry of helper-function definitions (glslGuardedDivideFn is handled directly by the
// emitter — its two overloads and caps-dependent epsilon don't fit the table).
var glslBlendHelpers = map[string]glslBlendHelper{
	glslBlendOverlayComponentFn: {
		params: glslBlendComponentParams,
		body: "return (2.0*d.x <= d.y) ? 2.0*s.x*d.x\n" +
			"                        : s.y*d.y - 2.0*(d.y - d.x)*(s.y - s.x);",
	},
	"_color_dodge_component": {
		params: glslBlendComponentParams,
		body: "float dxScale = d.x == 0.0 ? 0.0 : 1.0;\n" +
			"float delta = dxScale * min(d.y, abs(s.y - s.x) >= " + glslMinNormalHalf + "\n" +
			"                                        ? _guarded_divide(d.x*s.y, s.y - s.x)\n" +
			"                                        : d.y);\n" +
			"return delta*s.y + s.x*(1.0 - d.y) + d.x*(1.0 - s.y);",
		helpers: []string{glslGuardedDivideFn},
	},
	"_color_burn_component": {
		params: glslBlendComponentParams,
		body: "float dyTerm = d.y == d.x ? d.y : 0.0;\n" +
			"float delta = abs(s.x) >= " + glslMinNormalHalf + "\n" +
			"                    ? d.y - min(d.y, _guarded_divide((d.y - d.x)*s.y, s.x))\n" +
			"                    : dyTerm;\n" +
			"return delta*s.y + s.x*(1.0 - d.y) + d.x*(1.0 - s.y);",
		helpers: []string{glslGuardedDivideFn},
	},
	"_soft_light_component": {
		params: glslBlendComponentParams,
		body: "if (2.0*s.x <= s.y) {\n" +
			"    return _guarded_divide(d.x*d.x*(s.y - 2.0*s.x), d.y) + (1.0 - d.y)*s.x + " +
			"d.x*(-s.y + 2.0*s.x + 1.0);\n" +
			"} else if (4.0 * d.x <= d.y) {\n" +
			"    float DSqd = d.x*d.x;\n" +
			"    float DCub = DSqd*d.x;\n" +
			"    float DaSqd = d.y*d.y;\n" +
			"    float DaCub = DaSqd*d.y;\n" +
			"    return _guarded_divide(DaSqd*(s.x - d.x*(3.0*s.y - 6.0*s.x - 1.0)) + " +
			"12.0*d.y*DSqd*(s.y - 2.0*s.x)\n" +
			"                           - 16.0*DCub * (s.y - 2.0*s.x) - DaCub*s.x, DaSqd);\n" +
			"} else {\n" +
			"    return d.x*(s.y - 2.0*s.x + 1.0) + s.x - sqrt(d.y*d.x)*(s.y - 2.0*s.x) - d.y*s.x;\n" +
			"}",
		helpers: []string{glslGuardedDivideFn},
	},
	"_blend_color_luminance": {
		params: "vec3 color",
		body:   "return dot(vec3(0.3, 0.59, 0.11), color);",
	},
	"_blend_set_color_luminance": {
		params: "vec3 hueSatColor, float alpha, vec3 lumColor",
		body: "float lum = _blend_color_luminance(lumColor);\n" +
			"vec3 result = lum - _blend_color_luminance(hueSatColor) + hueSatColor;\n" +
			"float minComp = min(min(result.r, result.g), result.b);\n" +
			"float maxComp = max(max(result.r, result.g), result.b);\n" +
			"if (minComp < 0.0 && lum != minComp) {\n" +
			"    result = lum + (result - lum) * _guarded_divide(lum, (lum - minComp) + " +
			glslMinNormalHalf + ");\n" +
			"}\n" +
			"if (maxComp > alpha && maxComp != lum) {\n" +
			"    result = lum +\n" +
			"             _guarded_divide((result - lum) * (alpha - lum), (maxComp - lum) + " +
			glslMinNormalHalf + ");\n" +
			"}\n" +
			"return result;",
		helpers: []string{glslGuardedDivideFn, "_blend_color_luminance"},
	},
	"_blend_color_saturation": {
		params: "vec3 color",
		body:   "return max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);",
	},
	"_blend_set_color_saturation": {
		params: "vec3 color, vec3 satColor",
		body: "float mn = min(min(color.r, color.g), color.b);\n" +
			"float mx = max(max(color.r, color.g), color.b);\n" +
			"return (mx > mn) ? ((color - mn) * _blend_color_saturation(satColor)) / (mx - mn)\n" +
			"                 : vec3(0.0);",
		helpers: []string{"_blend_color_saturation"},
	},
	// blend_darken_mode is the shared uniform-driven blend_darken(float mode, vec4 src, vec4 dst) helper (darken: 1,
	// lighten: -1); GLSL has overloading, but a distinct name keeps this form separable from the two-argument mode
	// function in dumps.
	"blend_darken_mode": {
		params: "float mode, vec4 src, vec4 dst",
		body: "vec4 a = blend_src_over(src, dst);\n" +
			"vec3 b = (1.0 - dst.a) * src.rgb + dst.rgb;\n" +
			"a.rgb = mode * min(a.rgb * mode, b.rgb * mode);\n" +
			"return a;",
		modes: []raster.BlendMode{raster.BlendSrcOver},
	},
	"blend_overlay_flip": {
		params: "float flip, vec4 a, vec4 b",
		body:   "return blend_overlay(bool(flip) ? b : a, bool(flip) ? a : b);",
		modes:  []raster.BlendMode{raster.BlendOverlay},
	},
	glslBlendHSLCFn: {
		params: "vec2 flipSat, vec4 src, vec4 dst",
		body: "float alpha = dst.a * src.a;\n" +
			"vec3 sda = src.rgb * dst.a;\n" +
			"vec3 dsa = dst.rgb * src.a;\n" +
			"vec3 l = bool(flipSat.x) ? dsa : sda;\n" +
			"vec3 r = bool(flipSat.x) ? sda : dsa;\n" +
			"if (bool(flipSat.y)) {\n" +
			"    l = _blend_set_color_saturation(l, r);\n" +
			"    r = dsa;\n" +
			"}\n" +
			"return vec4(_blend_set_color_luminance(l, alpha, r) + dst.rgb - dsa + src.rgb - sda,\n" +
			"            src.a + dst.a - alpha);",
		helpers: []string{"_blend_set_color_saturation", "_blend_set_color_luminance"},
	},
}

// glslPorterDuffBody is the body of blend_porter_duff: the multi-purpose function that performs any of the twelve
// coefficient blends given a (C, C, S, S) blendOp vector.
const glslPorterDuffBody = "vec2 coeff = blendOp.xy + blendOp.zw * vec2(dst.a, src.a);\n" +
	"return src * coeff.x + dst * coeff.y;"

// getPorterDuffBlendConstants returns the (C, C, S, S) blendOp vector for one of the twelve coefficient blend modes,
// for use with blend_porter_duff, or nil if mode isn't a coefficient blend.
func getPorterDuffBlendConstants(mode raster.BlendMode) []float32 {
	switch mode {
	case raster.BlendClear:
		return []float32{0, 0, 0, 0}
	case raster.BlendSrc:
		return []float32{1, 0, 0, 0}
	case raster.BlendDst:
		return []float32{0, 1, 0, 0}
	case raster.BlendSrcOver:
		return []float32{1, 1, 0, -1}
	case raster.BlendDstOver:
		return []float32{1, 1, -1, 0}
	case raster.BlendSrcIn:
		return []float32{0, 0, 1, 0}
	case raster.BlendDstIn:
		return []float32{0, 0, 0, 1}
	case raster.BlendSrcOut:
		return []float32{1, 0, -1, 0}
	case raster.BlendDstOut:
		return []float32{0, 1, 0, -1}
	case raster.BlendSrcATop:
		return []float32{0, 1, 1, -1}
	case raster.BlendDstATop:
		return []float32{1, 0, -1, 1}
	case raster.BlendXor:
		return []float32{1, 1, -1, -1}
	default:
		return nil
	}
}

// reducedBlendModeInfo names the shared GLSL blend function a mode reduces to, plus the uniform data (if any) that
// parameterizes it for that mode.
type reducedBlendModeInfo struct {
	function    string
	uniformData []float32
}

// getReducedBlendModeInfo reduces mode to a shared, uniform-driven blend function where possible (so many modes can
// share one compiled function), falling back to a dedicated per-mode function otherwise. This switch must be kept in
// sync with GLSLBlendKey.
func getReducedBlendModeInfo(mode raster.BlendMode) reducedBlendModeInfo {
	switch mode {
	// Clear/src/dst are intentionally omitted; using the dedicated blend functions is preferable, since that gives an
	// opportunity to eliminate the src/dst entirely.
	case raster.BlendSrcOver, raster.BlendDstOver, raster.BlendSrcIn, raster.BlendDstIn,
		raster.BlendSrcOut, raster.BlendDstOut, raster.BlendSrcATop, raster.BlendDstATop,
		raster.BlendXor:
		return reducedBlendModeInfo{function: "blend_porter_duff", uniformData: getPorterDuffBlendConstants(mode)}

	case raster.BlendHue:
		return reducedBlendModeInfo{function: glslBlendHSLCFn, uniformData: []float32{0, 1}}
	case raster.BlendSaturation:
		return reducedBlendModeInfo{function: glslBlendHSLCFn, uniformData: []float32{1, 1}}
	case raster.BlendColor:
		return reducedBlendModeInfo{function: glslBlendHSLCFn, uniformData: []float32{0, 0}}
	case raster.BlendLuminosity:
		return reducedBlendModeInfo{function: glslBlendHSLCFn, uniformData: []float32{1, 0}}

	case raster.BlendOverlay:
		return reducedBlendModeInfo{function: glslBlendOverlayFn, uniformData: []float32{0}}
	case raster.BlendHardLight:
		return reducedBlendModeInfo{function: glslBlendOverlayFn, uniformData: []float32{1}}

	case raster.BlendDarken:
		return reducedBlendModeInfo{function: glslBlendDarkenFn, uniformData: []float32{1}}
	case raster.BlendLighten:
		return reducedBlendModeInfo{function: glslBlendDarkenFn, uniformData: []float32{-1}}

	default:
		return reducedBlendModeInfo{function: glslBlendFuncName(mode), uniformData: nil}
	}
}

// GLSLBlendKey returns the shader program cache key contribution for mode: a negative sentinel identifying a shared
// blend function, or the mode value itself when a dedicated per-mode function is used. This switch must be kept in sync
// with getReducedBlendModeInfo.
func GLSLBlendKey(mode raster.BlendMode) int32 {
	switch mode {
	case raster.BlendSrcOver, raster.BlendDstOver, raster.BlendSrcIn, raster.BlendDstIn,
		raster.BlendSrcOut, raster.BlendDstOut, raster.BlendSrcATop, raster.BlendDstATop,
		raster.BlendXor:
		return -1 // blend_porter_duff

	case raster.BlendHue, raster.BlendSaturation, raster.BlendLuminosity, raster.BlendColor:
		return -2 // blend_hslc

	case raster.BlendOverlay, raster.BlendHardLight:
		return -3 // blend_overlay

	case raster.BlendDarken, raster.BlendLighten:
		return -4 // blend_darken

	default:
		return int32(mode) // uses a dedicated blend function
	}
}

// GLSLBlendExpression returns a GLSL expression that blends srcColor and dstColor per mode, adding a "blend" uniform
// when the mode's function is uniform-driven.
func GLSLBlendExpression(processor Processor, uniformHandler *UniformHandler, fragBuilder *FragmentShaderBuilder, blendUniform *UniformHandle, srcColor, dstColor string, mode raster.BlendMode) string {
	info := getReducedBlendModeInfo(mode)
	if len(info.uniformData) == 0 {
		fn := fragBuilder.ensureBlendFunction(mode)
		return fmt.Sprintf("%s(%s, %s)", fn, srcColor, dstColor)
	}

	glslType := GLSLType(int(GLSLTypeHalf) + len(info.uniformData) - 1)
	if glslType < GLSLTypeHalf || glslType > GLSLTypeHalf4 {
		panic("unexpected blend uniform type")
	}

	var blendUniName string
	*blendUniform, blendUniName = uniformHandler.AddUniform(processor, ShaderFlagFragment,
		glslType, "blend")
	fn := fragBuilder.ensureSharedBlendFunction(info.function)
	return fmt.Sprintf("%s(%s, %s, %s)", fn, blendUniName, srcColor, dstColor)
}

// GLSLSetBlendModeUniformData uploads the uniform data (if any) that parameterizes mode's shared blend function,
// matching the uniform added by GLSLBlendExpression.
func GLSLSetBlendModeUniformData(pdman *ProgramDataManager, blendUniform UniformHandle, mode raster.BlendMode) {
	info := getReducedBlendModeInfo(mode)
	switch len(info.uniformData) {
	case 0:
		// No uniform data necessary.
	case 1:
		pdman.Set1f(blendUniform, info.uniformData[0])
	case 2:
		pdman.Set2f(blendUniform, info.uniformData[0], info.uniformData[1])
	case 3:
		pdman.Set3f(blendUniform, info.uniformData[0], info.uniformData[1], info.uniformData[2])
	case 4:
		pdman.Set4f(blendUniform, info.uniformData[0], info.uniformData[1], info.uniformData[2],
			info.uniformData[3])
	}
}
