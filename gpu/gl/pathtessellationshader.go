// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The common base for shaders in the GPU tessellator: pipeline/program helpers and Wang's-formula GLSL library, the
// standard Redbook stencil settings, and the two path-fill shaders — the simple triangle shader (inner fans and
// bounding boxes) and the fixed-count "middle-out" instanced curve shader. Shader bodies are emitted directly as GLSL,
// with fma() (GLSL 4.00+) written as mul-add so one body serves GLSL 1.50 contexts too (a tolerance-only difference).

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
)

//////////////////////////////////////////////////////////////////////////////
// Tessellation shader shared infrastructure

// tessProgramArgs bundles the flush-state fields the tessellation program/pipeline builders need.
type tessProgramArgs struct {
	writeView        *SurfaceProxyView
	dstProxyView     *DstProxyView
	caps             *Caps
	xferBarrierFlags gpu.XferBarrierFlags
	colorLoadOp      gpu.LoadOp
	usesMSAASurface  bool
}

// makeTessProgramArgs gathers the tessProgramArgs fields from the flush state, matching the argument lists at the ops'
// onPrepare call sites.
func makeTessProgramArgs(state *OpFlushState) tessProgramArgs {
	args := state.OpArgs()
	return tessProgramArgs{
		writeView:        args.WriteView(),
		usesMSAASurface:  args.UsesMSAASurface(),
		dstProxyView:     args.DstProxyView(),
		xferBarrierFlags: args.RenderPassBarriers(),
		colorLoadOp:      args.ColorLoadOp(),
		caps:             state.Caps(),
	}
}

// tessellationShaderBase holds the state shared by every tessellation geometry processor: the primitive type, view
// matrix, and paint color.
type tessellationShaderBase struct {
	GPBase
	primitiveType gpu.PrimitiveType
	viewMatrix    geom.Matrix
	color         colorcore.PMColor4f
}

func (t *tessellationShaderBase) initTessellationShader(classID ClassID, primitiveType gpu.PrimitiveType, viewMatrix *geom.Matrix, color colorcore.PMColor4f) {
	t.initGP(classID)
	t.primitiveType = primitiveType
	t.viewMatrix = *viewMatrix
	t.color = color
}

// tessMakePipeline builds the pipeline for a tessellation draw. The processors must already be finalized; the applied
// clip is consumed.
func tessMakePipeline(args *tessProgramArgs, processors *ProcessorSet, appliedClip *AppliedClip) *Pipeline {
	return NewPipeline(&PipelineInitArgs{
		Caps:         args.caps,
		DstProxyView: *args.dstProxyView,
		WriteSwizzle: args.writeView.Swizzle(),
	}, processors, appliedClip)
}

// tessMakeProgram builds the program info for a tessellation draw.
func tessMakeProgram(args *tessProgramArgs, shader GeometryProcessor, primitiveType gpu.PrimitiveType, pipeline *Pipeline, stencil *UserStencilSettings) *ProgramInfo {
	return NewProgramInfo(args.caps, args.writeView, args.usesMSAASurface, pipeline, stencil,
		shader, primitiveType, args.xferBarrierFlags, args.colorLoadOp)
}

// tessWangsFormulaGLSL is the shared Wang's-formula GLSL library, used to estimate how many segments a curve needs at a
// given tolerance (fma becomes mul-add per the file comment).
const tessWangsFormulaGLSL = `
float wangs_formula_max_fdiff_p2(vec2 p0, vec2 p1, vec2 p2, vec2 p3, mat2 matrix_) {
    vec2 d0 = matrix_ * ((-2.0*p1 + p2) + p0);
    vec2 d1 = matrix_ * ((-2.0*p2 + p3) + p1);
    return max(dot(d0,d0), dot(d1,d1));
}
float wangs_formula_cubic(float precision_, vec2 p0, vec2 p1, vec2 p2, vec2 p3, mat2 matrix_) {
    float m = wangs_formula_max_fdiff_p2(p0, p1, p2, p3, matrix_);
    return max(ceil(sqrt(0.75 * precision_ * sqrt(m))), 1.0);
}
float wangs_formula_cubic_log2(float precision_, vec2 p0, vec2 p1, vec2 p2, vec2 p3,
                               mat2 matrix_) {
    float m = wangs_formula_max_fdiff_p2(p0, p1, p2, p3, matrix_);
    return ceil(log2(max(0.5625 * precision_ * precision_ * m, 1.0)) * .25);
}
float wangs_formula_conic_p2(float precision_, vec2 p0, vec2 p1, vec2 p2, float w) {
    vec2 C_ = (min(min(p0, p1), p2) + max(max(p0, p1), p2)) * 0.5;
    p0 -= C_;
    p1 -= C_;
    p2 -= C_;
    float m = sqrt(max(max(dot(p0,p0), dot(p1,p1)), dot(p2,p2)));
    vec2 dp = ((-2.0*w)*p1 + p0) + p2;
    float dw = abs((-2.0*w) + 2.0);
    float rp_minus_1 = max(0.0, (m*precision_) - 1.0);
    float numer = length(dp) * precision_ + rp_minus_1 * dw;
    float denom = 4.0 * min(w, 1.0);
    return numer/denom;
}
float wangs_formula_conic(float precision_, vec2 p0, vec2 p1, vec2 p2, float w) {
    float n2 = wangs_formula_conic_p2(precision_, p0, p1, p2, w);
    return max(ceil(sqrt(n2)), 1.0);
}
float wangs_formula_conic_log2(float precision_, vec2 p0, vec2 p1, vec2 p2, float w) {
    float n2 = wangs_formula_conic_p2(precision_, p0, p1, p2, w);
    return ceil(log2(max(n2, 1.0)) * .5);
}
`

//////////////////////////////////////////////////////////////////////////////
// Redbook stencil settings shared by the tessellation path-fill shaders

// Increments clockwise triangles and decrements counterclockwise ones; used for "winding" fill.
var tessIncrDecrStencil = MakeUserStencilSettingsSeparate(
	UserStencilFace{
		Ref: 0x0000, Test: UserStencilTestAlwaysIfInClip, TestMask: 0xffff,
		PassOp: UserStencilOpIncWrap, FailOp: UserStencilOpKeep, WriteMask: 0xffff,
	},
	UserStencilFace{
		Ref: 0x0000, Test: UserStencilTestAlwaysIfInClip, TestMask: 0xffff,
		PassOp: UserStencilOpDecWrap, FailOp: UserStencilOpKeep, WriteMask: 0xffff,
	},
)

// Inverts the bottom stencil bit; used for "even/odd" fill.
var tessInvertStencil = MakeUserStencilSettings(0x0000, UserStencilTestAlwaysIfInClip, 0xffff,
	UserStencilOpInvert, UserStencilOpKeep, 0x0001)

// Allows non-zero stencil values to pass and write a color, and resets the stencil value back to zero; discards
// immediately on stencil values of zero. (No need to check the clip because the previous stencil pass will have only
// written to samples already inside the clip.)
var tessTestAndResetStencil = MakeUserStencilSettings(0x0000, UserStencilTestNotEqual, 0xffff,
	UserStencilOpZero, UserStencilOpKeep, 0xffff)

var tessTestAndResetStencilInverted = MakeUserStencilSettings(0x0000, UserStencilTestEqual,
	0xffff, UserStencilOpKeep, UserStencilOpZero, 0xffff)

// tessStencilPathSettings returns the stencil settings to use for a standard Redbook "stencil" pass, selected by fill
// rule: winding and inverse-winding take the nonzero rule; even-odd fills take the invert rule.
func tessStencilPathSettings(fillType path.FillType) *UserStencilSettings {
	if fillType == path.FillWinding || fillType == path.FillInverseWinding {
		return &tessIncrDecrStencil
	}
	return &tessInvertStencil
}

// tessTestAndResetStencilSettings returns the stencil settings for the "test and reset" cover pass, selected by whether
// the fill is inverse.
func tessTestAndResetStencilSettings(isInverseFill bool) *UserStencilSettings {
	if isInverseFill {
		return &tessTestAndResetStencilInverted
	}
	return &tessTestAndResetStencil
}

// tessMakeStencilOnlyPipeline builds a pipeline that does not write to the color buffer. hardClip carries the applied
// clip's scissor/stencil-clip state.
func tessMakeStencilOnlyPipeline(args *tessProgramArgs, hardClip *AppliedClip, pipelineFlags uint8) *Pipeline {
	return NewPipelineHardClip(&PipelineInitArgs{
		InputFlags:   pipelineFlags,
		Caps:         args.caps,
		WriteSwizzle: gpu.SwizzleRGBA, // the default write swizzle
	}, disableColorXPSingleton, hardClip)
}

//////////////////////////////////////////////////////////////////////////////
// pathTessShaderImpl — the shared uniform matrix + color plumbing.

// pathTessShader is implemented by the tessellation fill shaders so the shared program impl can reach their common
// state.
type pathTessShader interface {
	GeometryProcessor
	tessBase() *tessellationShaderBase
	shaderAttribs() PatchAttribs
}

// pathTessShaderImpl manages the affineMatrix/translate vertex uniforms and the color (uniform or varying), delegating
// the vertex body to emitVertexCode.
type pathTessShaderImpl struct {
	GPImplBase
	// emitVertexCode emits the shader-specific vertex body; each concrete shader supplies its own.
	emitVertexCode      func(args *GPEmitArgs, impl *pathTessShaderImpl, gpArgs *GPArgs)
	varyingColorName    string
	affineMatrixUniform UniformHandle
	translateUniform    UniformHandle
	colorUniform        UniformHandle
}

// onEmitCode implements GPProgramImpl: emits the shared affineMatrix/translate uniform plumbing, then delegates the
// vertex body to emitVertexCode.
func (i *pathTessShaderImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	shader := args.GeomProc.(pathTessShader)
	args.VaryingHandler.EmitAttributes(shader)

	// Vertex shader.
	var affineMatrix, translate string
	i.affineMatrixUniform, affineMatrix = args.UniformHandler.AddUniform(nil, ShaderFlagVertex,
		GLSLTypeFloat4, "affineMatrix")
	i.translateUniform, translate = args.UniformHandler.AddUniform(nil, ShaderFlagVertex,
		GLSLTypeFloat2, "translate")
	args.VertBuilder.CodeAppendf("mat2 AFFINE_MATRIX = mat2(%s.xy, %s.zw);", affineMatrix,
		affineMatrix)
	args.VertBuilder.CodeAppendf("vec2 TRANSLATE = %s;", translate)
	i.emitVertexCode(args, i, gpArgs)

	// Fragment shader.
	if shader.shaderAttribs()&PatchAttribColor == 0 {
		var color string
		i.colorUniform, color = args.UniformHandler.AddUniform(nil, ShaderFlagFragment,
			GLSLTypeHalf4, "color")
		args.FragBuilder.CodeAppendf("vec4 %s = %s;", args.OutputColor, color)
	} else {
		args.FragBuilder.CodeAppendf("vec4 %s = %s;", args.OutputColor, i.varyingColorName)
	}
	args.FragBuilder.CodeAppendf("const vec4 %s = vec4(1);", args.OutputCoverage)
}

// SetData implements GPProgramImpl: uploads the view matrix and (when not attribute-carried) the color uniform.
func (i *pathTessShaderImpl) SetData(pdman *ProgramDataManager, _ *gpu.ShaderCaps, geomProc GeometryProcessor) {
	shader := geomProc.(pathTessShader)
	base := shader.tessBase()
	m := &base.viewMatrix
	pdman.Set4f(i.affineMatrixUniform, m.Get(geom.MScaleX), m.Get(geom.MSkewY),
		m.Get(geom.MSkewX), m.Get(geom.MScaleY))
	pdman.Set2f(i.translateUniform, m.Get(geom.MTransX), m.Get(geom.MTransY))

	if shader.shaderAttribs()&PatchAttribColor == 0 {
		pdman.Set4f(i.colorUniform, base.color.R, base.color.G, base.color.B, base.color.A)
	}
}

//////////////////////////////////////////////////////////////////////////////
// SimpleTriangleShader

// simpleTriangleShader draws a simple array of triangles.
type simpleTriangleShader struct {
	attrs [1]Attribute
	tessellationShaderBase
}

// makeSimpleTriangleShader returns a shader for drawing a plain triangle array (inner fans and bounding boxes).
func makeSimpleTriangleShader(viewMatrix *geom.Matrix, color colorcore.PMColor4f) *simpleTriangleShader {
	s := &simpleTriangleShader{}
	s.initTessellationShader(TessellateSimpleTriangleShaderClassID, gpu.PrimitiveTypeTriangles,
		viewMatrix, color)
	s.attrs[0] = MakeAttribute("inputPoint", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	s.setVertexAttributesWithImplicitOffsets(s.attrs[:])
	return s
}

func (s *simpleTriangleShader) Name() string { return "tessellate_SimpleTriangleShader" }

func (s *simpleTriangleShader) tessBase() *tessellationShaderBase {
	return &s.tessellationShaderBase
}

func (s *simpleTriangleShader) shaderAttribs() PatchAttribs { return PatchAttribsNone }

// AddToKey implements GeometryProcessor (nothing beyond the class/attribute key).
func (s *simpleTriangleShader) AddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

// MakeProgramImpl implements GeometryProcessor.
func (s *simpleTriangleShader) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &pathTessShaderImpl{
		emitVertexCode: func(args *GPEmitArgs, _ *pathTessShaderImpl, gpArgs *GPArgs) {
			args.VertBuilder.CodeAppend(
				"vec2 localcoord = inputPoint;" +
					"vec2 vertexpos = AFFINE_MATRIX * localcoord + TRANSLATE;",
			)
			gpArgs.LocalCoordVar.Set(GLSLTypeFloat2, "localcoord")
			gpArgs.PositionVar.Set(GLSLTypeFloat2, "vertexpos")
		},
	}
}

//////////////////////////////////////////////////////////////////////////////
// MiddleOutShader

// middleOutShader uses instanced draws to triangulate standalone closed curves with a "middle-out" topology. Middle-out
// draws a triangle with vertices at T=[0, 1/2, 1] and then recurses breadth first:
//
//	depth=0: T=[0, 1/2, 1]
//	depth=1: T=[0, 1/4, 2/4], T=[2/4, 3/4, 1]
//	depth=2: T=[0, 1/8, 2/8], T=[2/8, 3/8, 4/8], T=[4/8, 5/8, 6/8], T=[6/8, 7/8, 1]
//	...
//
// The shader determines how many segments are required to render each individual curve smoothly, and emits empty
// triangles at any vertices whose IDs are higher than necessary. It is the caller's responsibility to draw enough
// vertices per instance for the most complex curve in the batch to render smoothly (i.e.,
// numCurveTrianglesAtResolveLevel() * 3).
type middleOutShader struct {
	instanceAttrs [5]Attribute // p01, p23, fanPointAttrib, colorAttrib, curveType
	vertexAttrs   [1]Attribute
	tessellationShaderBase
	attribs PatchAttribs
}

// makeMiddleOutShader returns a middle-out curve shader. The explicit curve type is used when, and only when, there
// isn't infinity support (otherwise the GPU can infer curve type from infinity).
func makeMiddleOutShader(shaderCaps *gpu.ShaderCaps, viewMatrix *geom.Matrix, color colorcore.PMColor4f, attribs PatchAttribs) *middleOutShader {
	if shaderCaps.InfinitySupport == (attribs&PatchAttribExplicitCurveType != 0) {
		panic("explicit curve type must be used exactly when infinity is unsupported")
	}
	s := &middleOutShader{attribs: attribs}
	s.initTessellationShader(TessellateMiddleOutShaderClassID, gpu.PrimitiveTypeTriangles,
		viewMatrix, color)
	n := 0
	s.instanceAttrs[n] = MakeAttribute("p01", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	n++
	s.instanceAttrs[n] = MakeAttribute("p23", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	n++
	if attribs&PatchAttribFanPoint != 0 {
		s.instanceAttrs[n] = MakeAttribute("fanPointAttrib", gpu.VertexAttribTypeFloat2,
			GLSLTypeFloat2)
		n++
	}
	if attribs&PatchAttribColor != 0 {
		s.instanceAttrs[n] = MakeColorAttribute("colorAttrib",
			attribs&PatchAttribWideColorIfEnabled != 0)
		n++
	}
	if attribs&PatchAttribExplicitCurveType != 0 {
		// A conic curve is written out with p3=[w,Infinity], but GPUs that don't support infinity can't detect this. On
		// those platforms we write out an extra float with each patch that explicitly tells the shader what type of
		// curve it is.
		s.instanceAttrs[n] = MakeAttribute("curveType", gpu.VertexAttribTypeFloat, GLSLTypeFloat)
		n++
	}
	s.setInstanceAttributesWithImplicitOffsets(s.instanceAttrs[:n])
	if s.InstanceStride() != patchStride(attribs) {
		panic("instance attributes out of sync with the patch stride")
	}
	s.vertexAttrs[0] = MakeAttribute("resolveLevel_and_idx", gpu.VertexAttribTypeFloat2,
		GLSLTypeFloat2)
	s.setVertexAttributesWithImplicitOffsets(s.vertexAttrs[:])
	return s
}

func (s *middleOutShader) Name() string { return "tessellate_MiddleOutShader" }

func (s *middleOutShader) tessBase() *tessellationShaderBase { return &s.tessellationShaderBase }

func (s *middleOutShader) shaderAttribs() PatchAttribs { return s.attribs }

// AddToKey implements GeometryProcessor. When color is in a uniform, it's always wide, so we need to ignore
// kWideColorIfEnabled; when color is in an attrib, its wideness is accounted for as part of the attribute key. Either
// way, we get the correct key by ignoring the flag.
func (s *middleOutShader) AddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(s.attribs&^PatchAttribWideColorIfEnabled), "middleOutShaderAttribs")
}

// MakeProgramImpl implements GeometryProcessor.
func (s *middleOutShader) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &pathTessShaderImpl{
		emitVertexCode: middleOutShaderEmitVertexCode,
	}
}

// middleOutShaderEmitVertexCode emits the vertex-stage GLSL for middleOutShader.
func middleOutShaderEmitVertexCode(args *GPEmitArgs, impl *pathTessShaderImpl, gpArgs *GPArgs) {
	shader := args.GeomProc.(*middleOutShader)
	v := args.VertBuilder
	v.DefinitionAppend(fmt.Sprintf("const float PRECISION = %g;", tessPrecision))
	v.DefinitionAppend(fmt.Sprintf("const float MAX_FIXED_RESOLVE_LEVEL = %g;",
		float32(tessMaxResolveLevel)))
	v.DefinitionAppend(fmt.Sprintf("const float MAX_FIXED_SEGMENTS = %g;",
		float32(tessMaxParametricSegments)))
	v.InsertFunction(tessWangsFormulaGLSL)
	if shader.attribs&PatchAttribExplicitCurveType != 0 {
		v.InsertFunction(fmt.Sprintf(
			"bool is_conic_curve() { return curveType != %g; }", tessCubicCurveType,
		))
		v.InsertFunction(fmt.Sprintf(
			"bool is_triangular_conic_curve() { return curveType == %g; }",
			tessTriangularConicCurveType,
		))
	} else {
		if !args.ShaderCaps.InfinitySupport {
			panic("curve-type inference requires shader infinity support")
		}
		v.InsertFunction(
			"bool is_conic_curve() { return isinf(p23.w); }" +
				"bool is_triangular_conic_curve() { return isinf(p23.z); }",
		)
	}
	if args.ShaderCaps.BitManipulationSupport {
		v.InsertFunction(
			"float ldexp_portable(float x, float p) { return ldexp(x, int(p)); }",
		)
	} else {
		v.InsertFunction(
			"float ldexp_portable(float x, float p) { return x * exp2(p); }",
		)
	}
	v.CodeAppend(
		"float resolveLevel = resolveLevel_and_idx.x;" +
			"float idxInResolveLevel = resolveLevel_and_idx.y;" +
			"vec2 localcoord;",
	)
	if shader.attribs&PatchAttribFanPoint != 0 {
		v.CodeAppend(
			// A negative resolve level means this is the fan point.
			"if (resolveLevel < 0.0) {" +
				"localcoord = fanPointAttrib;" +
				"} else ",
		) // Fall through to the next if(). The trailing space is important.
	}
	v.CodeAppend(
		"if (is_triangular_conic_curve()) {" +
			// This patch is an exact triangle.
			"localcoord = (resolveLevel != 0.0)      ? p01.zw " +
			": (idxInResolveLevel != 0.0) ? p23.xy " +
			": p01.xy;" +
			"} else {" +
			"vec2 p0=p01.xy, p1=p01.zw, p2=p23.xy, p3=p23.zw;" +
			"float w = -1.0;" + // w < 0 means treat the instance as an integral cubic.
			"float maxResolveLevel;" +
			"if (is_conic_curve()) {" +
			// Conics are 3 points, with the weight in p3.
			"w = p3.x;" +
			"maxResolveLevel = wangs_formula_conic_log2(PRECISION, AFFINE_MATRIX * p0," +
			"AFFINE_MATRIX * p1," +
			"AFFINE_MATRIX * p2, w);" +
			"p1 *= w;" + // Unproject p1.
			"p3 = p2;" + // Duplicate the endpoint for shared code that also runs on cubics.
			"} else {" +
			// The patch is an integral cubic.
			"maxResolveLevel = wangs_formula_cubic_log2(PRECISION, p0, p1, p2, p3," +
			"AFFINE_MATRIX);" +
			"}" +
			"if (resolveLevel > maxResolveLevel) {" +
			// This vertex is at a higher resolve level than we need. Demote to a lower resolveLevel, which will produce
			// a degenerate triangle.
			"idxInResolveLevel = floor(ldexp_portable(idxInResolveLevel," +
			"maxResolveLevel - resolveLevel));" +
			"resolveLevel = maxResolveLevel;" +
			"}" +
			// Promote our location to a discrete position in the maximum fixed resolve level. This is extra paranoia to
			// ensure we get the exact same fp32 coordinates for colocated points from different resolve levels (e.g.,
			// the vertices T=3/4 and T=6/8 should be exactly colocated).
			"float fixedVertexID = floor(.5 + ldexp_portable(" +
			"idxInResolveLevel, MAX_FIXED_RESOLVE_LEVEL - resolveLevel));" +
			"if (0.0 < fixedVertexID && fixedVertexID < MAX_FIXED_SEGMENTS) {" +
			"float T = fixedVertexID * (1.0 / MAX_FIXED_SEGMENTS);" +

			// Evaluate at T. Use De Casteljau's for its accuracy and stability.
			"vec2 ab = mix(p0, p1, T);" +
			"vec2 bc = mix(p1, p2, T);" +
			"vec2 cd = mix(p2, p3, T);" +
			"vec2 abc = mix(ab, bc, T);" +
			"vec2 bcd = mix(bc, cd, T);" +
			"vec2 abcd = mix(abc, bcd, T);" +

			// Evaluate the conic weight at T.
			"float u = mix(1.0, w, T);" +
			"float v = w + 1.0 - u;" + // == mix(w, 1, T)
			"float uv = mix(u, v, T);" +

			"localcoord = (w < 0.0) ? abcd : abc/uv;" + /*cubic : conic*/
			"} else {" +
			"localcoord = (fixedVertexID == 0.0) ? p0.xy : p3.xy;" +
			"}" +
			"}" +
			"vec2 vertexpos = AFFINE_MATRIX * localcoord + TRANSLATE;",
	)
	gpArgs.LocalCoordVar.Set(GLSLTypeFloat2, "localcoord")
	gpArgs.PositionVar.Set(GLSLTypeFloat2, "vertexpos")
	if shader.attribs&PatchAttribColor != 0 {
		colorVarying := NewVarying(GLSLTypeHalf4)
		args.VaryingHandler.AddVaryingWithInterpolation("color", &colorVarying,
			InterpolationCanBeFlat)
		v.CodeAppendf("%s = colorAttrib;", colorVarying.VsOut())
		impl.varyingColorName = colorVarying.FsIn()
	}
}
