// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// strokeTessellationShader tessellates a batch of stroke patches directly to the canvas. Tessellated stroking works by
// creating stroke-width, orthogonal edges at set locations along the curve and then connecting them with a quad strip.
// These orthogonal edges come from two different sets: "parametric edges" and "radial edges". Parametric edges are
// spaced evenly in the parametric sense, and radial edges divide the curve's _rotation_ into even steps. The
// tessellation shader evaluates both sets of edges and sorts them into a single quad strip. With this combined set of
// edges we can stroke any curve, regardless of curvature.
//
// The shader bodies are emitted directly as GLSL: float2/float4 become vec2/vec4, float2x2 becomes mat2, the vertex-ID
// builtin becomes gl_VertexID, cross_length_2d is inserted as a determinant helper, and fma() is written as mul-add so
// one body serves GLSL 1.50 contexts (a tolerance-only difference). The binary-search counter is named "exp_" so it
// cannot shadow the GLSL builtin. The no-gl_VertexID edgeID vertex attribute is trimmed (desktop core profiles always
// have gl_VertexID).

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/stroke"
)

// vec2 robust_normalize_diff(vec2 a, vec2 b): the normalized difference between a and b, i.e. normalize(a - b), with
// care taken for if a and/or b have large coordinates.
const strokeRobustNormalizeDiffFn = `
vec2 robust_normalize_diff(vec2 a, vec2 b) {
    vec2 diff = a - b;
    if (diff == vec2(0.0)) {
        return vec2(0.0);
    } else {
        float invMag = 1.0 / max(abs(diff.x), abs(diff.y));
        return normalize(invMag * diff);
    }
}`

// float cosine_between_unit_vectors(vec2 a, vec2 b): the cosine of the angle between a and b, assuming a and b are unit
// vectors already. Guaranteed to be between [-1, 1]. (Since a and b are assumed to be normalized, the cosine is equal
// to the dot product, although we clamp that to ensure it falls within the expected range.)
const strokeCosineBetweenUnitVectorsFn = `
float cosine_between_unit_vectors(vec2 a, vec2 b) {
    return clamp(dot(a, b), -1.0, 1.0);
}`

// float miter_extent(float cosTheta, float miterLimit): extends the middle radius to either the miter point, or the
// bevel edge if we surpassed the miter limit and need to revert to a bevel join. (fma(cosTheta, .5, .5) is written as
// mul-add.)
const strokeMiterExtentFn = `
float miter_extent(float cosTheta, float miterLimit) {
    float x = cosTheta * .5 + .5;
    return (x * miterLimit * miterLimit >= 1.0) ? inversesqrt(x) : sqrt(x);
}`

// float num_radial_segments_per_radian(float approxDevStrokeRadius): the number of radial segments required for each
// radian of rotation, in order for the curve to appear "smooth" as defined by the approximate device-space stroke
// radius.
const strokeNumRadialSegmentsPerRadianFn = `
float num_radial_segments_per_radian(float approxDevStrokeRadius) {
    return .5 / acos(max(1.0 - (1.0 / PRECISION) / approxDevStrokeRadius, -1.0));
}`

// unchecked_mix: unlike mix(), this does not return b when t==1. But it otherwise seems to get better precision than
// "a*(1 - t) + b*t" for things like chopping cubics on exact cusp points. We override this result anyway when t==1 so
// it shouldn't be a problem. (fma is written as mul-add.)
const strokeUncheckedMixFn = `
float unchecked_mix(float a, float b, float T) {
    return (b - a) * T + a;
}
vec2 unchecked_mix(vec2 a, vec2 b, float T) {
    return (b - a) * vec2(T) + a;
}
vec4 unchecked_mix(vec4 a, vec4 b, vec4 T) {
    return (b - a) * T + a;
}`

// strokeCrossLength2DFn defines the cross_length_2d helper that the stroke body references.
const strokeCrossLength2DFn = `
float cross_length_2d(vec2 a, vec2 b) { return determinant(mat2(a, b)); }`

// strokeTessellationShader is the geometry processor for tessellated stroke patches.
type strokeTessellationShader struct {
	attribs []Attribute
	tessellationShaderBase
	stroke       stroke.Rec
	patchAttribs PatchAttribs
}

// newStrokeTessellationShader creates a stroke tessellation shader. viewMatrix is applied to the geometry post
// tessellation; it cannot have perspective.
func newStrokeTessellationShader(shaderCaps *gpu.ShaderCaps, attribs PatchAttribs, viewMatrix *geom.Matrix, rec *stroke.Rec, color colorcore.PMColor4f) *strokeTessellationShader {
	s := &strokeTessellationShader{
		patchAttribs: attribs | PatchAttribJoinControlPoint,
		stroke:       *rec,
	}
	s.initTessellationShader(TessellateStrokeTessellationShaderClassID,
		gpu.PrimitiveTypeTriangleStrip, viewMatrix, color)
	// We should use explicit curve type when, and only when, there isn't infinity support. Otherwise the GPU can infer
	// curve type based on infinity.
	if shaderCaps.InfinitySupport == (attribs&PatchAttribExplicitCurveType != 0) {
		panic("explicit curve type must be used exactly when infinity is unsupported")
	}
	// pts 0..3 define the stroke as a cubic bezier. If p3.y is infinity, then it's a conic with w=p3.x.
	//
	// An empty stroke (p0==p1==p2==p3) is a special case that denotes a circle, or 180-degree point stroke.
	s.attribs = append(s.attribs,
		MakeAttribute("pts01Attr", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4),
		MakeAttribute("pts23Attr", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4),
		// argsAttr contains the lastControlPoint for setting up the join.
		MakeAttribute("argsAttr", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2))
	if s.patchAttribs&PatchAttribStrokeParams != 0 {
		s.attribs = append(s.attribs,
			MakeAttribute("dynamicStrokeAttr", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2))
	}
	if s.patchAttribs&PatchAttribColor != 0 {
		s.attribs = append(s.attribs, MakeColorAttribute("dynamicColorAttr",
			s.patchAttribs&PatchAttribWideColorIfEnabled != 0))
	}
	if s.patchAttribs&PatchAttribExplicitCurveType != 0 {
		// A conic curve is written out with p3=[w,Infinity], but GPUs that don't support infinity can't detect this. On
		// these platforms we write out an extra float with each patch that explicitly tells the shader what type of
		// curve it is.
		s.attribs = append(s.attribs,
			MakeAttribute("curveTypeAttr", gpu.VertexAttribTypeFloat, GLSLTypeFloat))
	}
	s.setInstanceAttributesWithImplicitOffsets(s.attribs)
	if s.InstanceStride() != 4*8+patchAttribsStride(s.patchAttribs) {
		panic("stroke shader instance attributes out of sync with the patch stride")
	}
	// An "edgeID" vertex attribute would be needed here if gl_VertexID were unsupported; desktop core profiles always
	// have gl_VertexID.
	return s
}

func (s *strokeTessellationShader) Name() string { return "StrokeTessellationShader" }

func (s *strokeTessellationShader) hasDynamicStroke() bool {
	return s.patchAttribs&PatchAttribStrokeParams != 0
}

func (s *strokeTessellationShader) hasDynamicColor() bool {
	return s.patchAttribs&PatchAttribColor != 0
}

func (s *strokeTessellationShader) hasExplicitCurveType() bool {
	return s.patchAttribs&PatchAttribExplicitCurveType != 0
}

// AddToKey implements GeometryProcessor. Attribs get worked into the key automatically as part of the attribute key.
// When color is in a uniform, it's always wide, so the wide-color flag doesn't need to be considered here.
func (s *strokeTessellationShader) AddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	keyNeedsJoin := s.patchAttribs&PatchAttribStrokeParams == 0
	if s.stroke.Join()>>2 != 0 {
		panic("stroke join must fit in two bits")
	}
	key := uint32(s.patchAttribs &^ PatchAttribColor)
	joinBits := uint32(0)
	if keyNeedsJoin {
		joinBits = uint32(s.stroke.Join())
	}
	key = (key << 2) | joinBits
	key <<= 1
	if s.stroke.IsHairlineStyle() {
		key |= 1
	}
	b.Add32(key, "strokeTessellationShaderKey")
}

// MakeProgramImpl implements GeometryProcessor.
func (s *strokeTessellationShader) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &strokeTessellationShaderImpl{}
}

// strokeTessellationShaderImpl emits shader code for the parametric/radial stroke tessellation algorithm described
// above.
type strokeTessellationShaderImpl struct {
	GPImplBase
	dynamicColorName       string
	tessControlArgsUniform UniformHandle
	translateUniform       UniformHandle
	affineMatrixUniform    UniformHandle
	colorUniform           UniformHandle
}

// onEmitCode implements GPProgramImpl.
func (i *strokeTessellationShaderImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	shader := args.GeomProc.(*strokeTessellationShader)
	joinType := shader.stroke.Join()
	args.VaryingHandler.EmitAttributes(shader)

	v := args.VertBuilder
	v.DefinitionAppend("const float PI = 3.141592653589793238;")
	v.DefinitionAppend(fmt.Sprintf("const float PRECISION = %g;", tessPrecision))
	// There is an artificial maximum number of edges (compared to the max limit calculated based on the number of
	// radial segments per radian, Wang's formula, and join type). With vertex ID support the limit is what can be
	// represented in a uint16 (the no-vertex-ID fallback buffer limit is trimmed).
	v.DefinitionAppend(fmt.Sprintf("const float NUM_TOTAL_EDGES = %g;",
		float32(fixedCountStrokesMaxEdges)))

	// Helper functions.
	if shader.hasDynamicStroke() {
		v.InsertFunction(strokeNumRadialSegmentsPerRadianFn)
	}
	v.InsertFunction(strokeRobustNormalizeDiffFn)
	v.InsertFunction(strokeCosineBetweenUnitVectorsFn)
	v.InsertFunction(strokeMiterExtentFn)
	v.InsertFunction(strokeUncheckedMixFn)
	v.InsertFunction(strokeCrossLength2DFn)
	v.InsertFunction(tessWangsFormulaGLSL)

	// Tessellation control uniforms and/or dynamic attributes.
	if !shader.hasDynamicStroke() {
		// [NUM_RADIAL_SEGMENTS_PER_RADIAN, JOIN_TYPE, STROKE_RADIUS]
		var tessArgsName string
		i.tessControlArgsUniform, tessArgsName = args.UniformHandler.AddUniform(nil,
			ShaderFlagVertex, GLSLTypeFloat3, "tessControlArgs")
		v.CodeAppendf(
			"float NUM_RADIAL_SEGMENTS_PER_RADIAN = %s.x;"+
				"float JOIN_TYPE = %s.y;"+
				"float STROKE_RADIUS = %s.z;", tessArgsName, tessArgsName, tessArgsName,
		)
	} else {
		// The shader does not currently support dynamic hairlines, so this case only needs to configure
		// NUM_RADIAL_SEGMENTS_PER_RADIAN based on the fixed maxScale and per-instance stroke radius attribute that's
		// defined in local space.
		if shader.stroke.IsHairlineStyle() {
			panic("dynamic hairlines are not supported")
		}
		var maxScaleName string
		i.tessControlArgsUniform, maxScaleName = args.UniformHandler.AddUniform(nil,
			ShaderFlagVertex, GLSLTypeFloat, "maxScale")
		v.CodeAppendf(
			"float STROKE_RADIUS = dynamicStrokeAttr.x;"+
				"float JOIN_TYPE = dynamicStrokeAttr.y;"+
				"float NUM_RADIAL_SEGMENTS_PER_RADIAN = num_radial_segments_per_radian("+
				"%s * STROKE_RADIUS);", maxScaleName,
		)
	}

	if shader.hasDynamicColor() {
		// Create a varying for color to get passed in through.
		dynamicColor := NewVarying(GLSLTypeHalf4)
		args.VaryingHandler.AddVarying("dynamicColor", &dynamicColor)
		v.CodeAppendf("%s = dynamicColorAttr;", dynamicColor.VsOut())
		i.dynamicColorName = dynamicColor.FsIn()
	}

	// View matrix uniforms.
	var translateName, affineMatrixName string
	i.affineMatrixUniform, affineMatrixName = args.UniformHandler.AddUniform(nil,
		ShaderFlagVertex, GLSLTypeFloat4, "affineMatrix")
	i.translateUniform, translateName = args.UniformHandler.AddUniform(nil, ShaderFlagVertex,
		GLSLTypeFloat2, "translate")
	v.CodeAppendf("mat2 AFFINE_MATRIX = mat2(%s.xy, %s.zw);\n", affineMatrixName,
		affineMatrixName)
	v.CodeAppendf("vec2 TRANSLATE = %s;\n", translateName)

	if shader.hasExplicitCurveType() {
		v.InsertFunction(fmt.Sprintf(
			"bool is_conic_curve() { return curveTypeAttr != %g; }", tessCubicCurveType,
		))
	} else {
		v.InsertFunction("bool is_conic_curve() { return isinf(pts23Attr.w); }")
	}

	// Tessellation code.
	v.CodeAppend(
		"vec2 p0=pts01Attr.xy, p1=pts01Attr.zw, p2=pts23Attr.xy, p3=pts23Attr.zw;" +
			"vec2 lastControlPoint = argsAttr.xy;" +
			"float w = -1.0;" + // w<0 means the curve is an integral cubic.
			"if (is_conic_curve()) {" +
			// Conics are 3 points, with the weight in p3.
			"w = p3.x;" +
			"p3 = p2;" + // Setting p3 equal to p2 works for the remaining rotational logic.
			"}",
	)

	// Emit code to call Wang's formula to determine parametric segments. We do this before transforming points for
	// hairlines so that it is consistent with how the CPU tested the control points for chopping.
	v.CodeAppend(
		// Find how many parametric segments this stroke requires.
		"float numParametricSegments;" +
			"if (w < 0.0) {" +
			"if (p0 == p1 && p2 == p3) {" +
			"numParametricSegments = 1.0;" + // a line
			"} else {" +
			"numParametricSegments = wangs_formula_cubic(PRECISION, p0, p1, p2, p3," +
			"AFFINE_MATRIX);" +
			"}" +
			"} else {" +
			"numParametricSegments = wangs_formula_conic(PRECISION," +
			"AFFINE_MATRIX * p0," +
			"AFFINE_MATRIX * p1," +
			"AFFINE_MATRIX * p2, w);" +
			"}",
	)

	if shader.stroke.IsHairlineStyle() {
		// Hairline case. Transform the points before tessellation. We can still hold off on the translate until the
		// end; we just need to perform the scale and skew right now.
		v.CodeAppend(
			"p0 = AFFINE_MATRIX * p0;" +
				"p1 = AFFINE_MATRIX * p1;" +
				"p2 = AFFINE_MATRIX * p2;" +
				"p3 = AFFINE_MATRIX * p3;" +
				"lastControlPoint = AFFINE_MATRIX * lastControlPoint;",
		)
	}

	v.CodeAppend(
		// Find the starting and ending tangents.
		"vec2 tan0 = robust_normalize_diff((p0 == p1) ? ((p1 == p2) ? p3 : p2) : p1, p0);" +
			"vec2 tan1 = robust_normalize_diff(p3, (p3 == p2) ? ((p2 == p1) ? p0 : p1) : p2);" +
			"if (tan0 == vec2(0.0)) {" +
			// The stroke is a point. This special case tells us to draw a stroke-width circle as a 180 degree point
			// stroke instead.
			"tan0 = vec2(1.0, 0.0);" +
			"tan1 = vec2(-1.0, 0.0);" +
			"}",
	)

	// gl_VertexID is always available on desktop core profiles (the no-vertex-ID edgeID vertex attrib is trimmed).
	v.CodeAppend(
		"float edgeID = float(gl_VertexID >> 1);" +
			"if ((gl_VertexID & 1) != 0) {" +
			"edgeID = -edgeID;" +
			"}",
	)

	// Potential optimization: (shader.hasDynamicStroke() && shader.hasRoundJoins())?
	if joinType == stroke.JoinRound || shader.hasDynamicStroke() {
		v.CodeAppend(
			// Determine how many edges to give to the round join. We emit the first and final edges of the join twice:
			// once full width and once restricted to half width. This guarantees perfect seaming by matching the
			// vertices from the join as well as from the strokes on either side.
			"vec2 prevTan = robust_normalize_diff(p0, lastControlPoint);" +
				"float joinRads = acos(cosine_between_unit_vectors(prevTan, tan0));" +
				"float numRadialSegmentsInJoin = max(ceil(joinRads *" +
				"NUM_RADIAL_SEGMENTS_PER_RADIAN), 1.0);" +
				// +2 because we emit the beginning and ending edges twice (see above comment).
				"float numEdgesInJoin = numRadialSegmentsInJoin + 2.0;" +
				// The stroke section needs at least two edges. Don't assign more to the join than "NUM_TOTAL_EDGES -
				// 2". (This is only relevant when the ideal max edge count calculated on the CPU had to be limited to
				// NUM_TOTAL_EDGES in the draw call.)
				"numEdgesInJoin = min(numEdgesInJoin, NUM_TOTAL_EDGES - 2.0);",
		)
		if shader.hasDynamicStroke() {
			v.CodeAppend(
				"if (JOIN_TYPE >= 0.0) {" + // Is the join not a round type?
					// Bevel and miter joins get 1 and 2 segments respectively. +2 because we emit the beginning and
					// ending edges twice (see above).
					"numEdgesInJoin = sign(JOIN_TYPE) + 1.0 + 2.0;" +
					"}",
			)
		}
	} else {
		v.CodeAppendf("float numEdgesInJoin = %d.0;", tessNumFixedEdgesInJoinType(joinType))
	}

	v.CodeAppend(
		// Find which direction the curve turns. NOTE: Since the curve is not allowed to inflect, we can just check
		// F'(.5) x F''(.5). NOTE: F'(.5) x F''(.5) has the same sign as (P2 - P0) x (P3 - P1).
		"float turn = cross_length_2d(p2 - p0, p3 - p1);" +
			"float combinedEdgeID = abs(edgeID) - numEdgesInJoin;" +
			"if (combinedEdgeID < 0.0) {" +
			"tan1 = tan0;" +
			// Don't let tan0 become zero. The code as-is isn't built to handle that case. tan0=0 means the join is
			// disabled, and to disable it with the existing code we can leave tan0 equal to tan1.
			"if (lastControlPoint != p0) {" +
			"tan0 = robust_normalize_diff(p0, lastControlPoint);" +
			"}" +
			"turn = cross_length_2d(tan0, tan1);" +
			"}" +

			// Calculate the curve's starting angle and rotation.
			"float cosTheta = cosine_between_unit_vectors(tan0, tan1);" +
			"float rotation = acos(cosTheta);" +
			"if (turn < 0.0) {" +
			// Adjust the sign of rotation to match the direction the curve turns.
			"rotation = -rotation;" +
			"}" +

			"float numRadialSegments;" +
			"float strokeOutset = sign(edgeID);" +
			"if (combinedEdgeID < 0.0) {" +
			// We belong to the preceding join. The first and final edges get duplicated, so we only have
			// "numEdgesInJoin - 2" segments.
			"numRadialSegments = numEdgesInJoin - 2.0;" +
			"numParametricSegments = 1.0;" + // Joins don't have parametric segments.
			"p3 = p2 = p1 = p0;" + // Colocate all points on the junction point.
			// Shift combinedEdgeID to the range [-1, numRadialSegments]. This duplicates the first edge and lands one
			// edge at the very end of the join. (The duplicated final edge will actually come from the section of our
			// strip that belongs to the stroke.)
			"combinedEdgeID += numRadialSegments + 1.0;" +
			// We normally restrict the join on one side of the junction, but if the tangents are nearly equivalent this
			// could theoretically result in bad seaming and/or cracks on the side we don't put it on. If the tangents
			// are nearly equivalent then we leave the join double-sided.
			"float sinEpsilon = 1e-2;" + // ~= sin(180deg / 3000)
			"bool tangentsNearlyParallel =" +
			"(abs(turn) * inversesqrt(dot(tan0, tan0) * dot(tan1, tan1))) < sinEpsilon;" +
			"if (!tangentsNearlyParallel || dot(tan0, tan1) < 0.0) {" +
			// There are two edges colocated at the beginning. Leave the first one double sided for seaming with the
			// previous stroke. (The double sided edge at the end will actually come from the section of our strip that
			// belongs to the stroke.)
			"if (combinedEdgeID >= 0.0) {" +
			"strokeOutset = (turn < 0.0) ? min(strokeOutset, 0.0) : max(strokeOutset, 0.0);" +
			"}" +
			"}" +
			"combinedEdgeID = max(combinedEdgeID, 0.0);" +
			"} else {" +
			// We belong to the stroke. Unless NUM_RADIAL_SEGMENTS_PER_RADIAN is incredibly high, clamping to
			// maxCombinedSegments will be a no-op because the draw call was invoked with sufficient vertices to cover
			// the worst case scenario of 180 degree rotation.
			"float maxCombinedSegments = NUM_TOTAL_EDGES - numEdgesInJoin - 1.0;" +
			"numRadialSegments = max(ceil(abs(rotation) * NUM_RADIAL_SEGMENTS_PER_RADIAN), 1.0);" +
			"numRadialSegments = min(numRadialSegments, maxCombinedSegments);" +
			"numParametricSegments = min(numParametricSegments," +
			"maxCombinedSegments - numRadialSegments + 1.0);" +
			"}" +

			// Additional parameters for the tessellation code.
			"float radsPerSegment = rotation / numRadialSegments;" +
			"float numCombinedSegments = numParametricSegments + numRadialSegments - 1.0;" +
			"bool isFinalEdge = (combinedEdgeID >= numCombinedSegments);" +
			"if (combinedEdgeID > numCombinedSegments) {" +
			"strokeOutset = 0.0;" + // The strip has more edges than we need. Drop this one.
			"}",
	)

	if joinType == stroke.JoinMiter || shader.hasDynamicStroke() {
		miterCondition := "true"
		if shader.hasDynamicStroke() {
			miterCondition = "JOIN_TYPE > 0.0" // Is the join a miter type?
		}
		v.CodeAppendf(
			// Edge #2 extends to the miter point.
			"if (abs(edgeID) == 2.0 && %s) {"+
				"strokeOutset *= miter_extent(cosTheta, JOIN_TYPE);"+ // miterLimit
				"}", miterCondition,
		)
	}

	i.emitTessellationCode(shader, v, gpArgs)

	i.emitFragmentCode(shader, args)
}

// emitTessellationCode emits code that calculates the vertex position and any other inputs to the fragment shader. The
// caller is responsible to define the following symbols before calling this method:
//
//	// Functions.
//	vec2 unchecked_mix(vec2, vec2, float);
//	float unchecked_mix(float, float, float);
//
//	// Values provided by either uniforms or attribs.
//	vec2 p0, p1, p2, p3;
//	float w;
//	float STROKE_RADIUS;
//	mat2 AFFINE_MATRIX;
//	vec2 TRANSLATE;
//
//	// Values calculated by the caller.
//	float combinedEdgeID;
//	bool isFinalEdge;
//	float numParametricSegments;
//	float radsPerSegment;
//	vec2 tan0; // Must be pre-normalized
//	vec2 tan1; // Must be pre-normalized
//	float strokeOutset;
func (i *strokeTessellationShaderImpl) emitTessellationCode(shader *strokeTessellationShader, v *VertexShaderBuilder, gpArgs *GPArgs) {
	v.CodeAppendf(
		"vec2 tangent, strokeCoord;"+
			"if (combinedEdgeID != 0.0 && !isFinalEdge) {"+
			// Compute the location and tangent direction of the stroke edge with the integral id "combinedEdgeID",
			// where combinedEdgeID is the sorted-order index of parametric and radial edges. Start by finding the
			// tangent function's power basis coefficients. These define a tangent direction (scaled by some uniform
			// value) as:
			//
			//	                                            |T^2|
			//	Tangent_Direction(T) = dx,dy = |A  2B  C| * |T  |
			//	                               |.   .  .|   |1  |
			"vec2 A, B, C = p1 - p0;"+
			"vec2 D = p3 - p0;"+
			"if (w >= 0.0) {"+
			// P0..P2 represent a conic and P3==P2. The derivative of a conic has a cumbersome order-4 denominator.
			// However, this isn't necessary if we are only interested in a vector in the same *direction* as a given
			// tangent line. Since the denominator scales dx and dy uniformly, we can throw it out completely after
			// evaluating the derivative with the standard quotient rule. This leaves us with a simpler quadratic
			// function that we use to find a tangent.
			"C *= w;"+
			"B = .5*D - C;"+
			"A = (w - 1.0) * D;"+
			"p1 *= w;"+
			"} else {"+
			"vec2 E = p2 - p1;"+
			"B = E - C;"+
			"A = vec2(-3.0) * E + D;"+ // fma(float2(-3), E, D)
			"}"+
			// A, B and C are deliberately left unnormalized. Upstream (crbug.com/800804, skbug.com/40042642) considered
			// normalizing their exponents here to prevent fp32 overflow, since C_ below scales with the square of
			// numParametricSegments: |C| * n^2 passes FLT_MAX once the local-space control points grow large enough.
			// The overflow is real, but it is unreachable in any regime where this lane is still accurate. Measured by
			// stroking a cubic whose coordinates are scaled by k under a compensating 1/k matrix (k an exact power of
			// two, so the device-space geometry is unchanged and any divergence is float32 blowup):
			//
			//	k >= 2^28  (coords ~3e10): this lane already stops matching its own k=1 rendering.
			//	k >= 2^60  (coords ~1e18): the CPU rasterizer diverges too; squares of coordinates overflow.
			//	k >= 2^115 (coords ~1e34): |C| * n^2 overflows -- the case this note is about.
			//
			// Normalizing the exponents would therefore repair the last link in a chain that broke ~87 binary orders
			// earlier; it cannot make any such input render correctly, and it would add per-vertex work to every
			// stroke. If a stroke ever does produce NaN/Inf geometry at plausible coordinates, this is not the
			// suspect -- the divergence at 2^28 is.

			// Now find the coefficients that give a tangent direction from a parametric edge ID:
			//
			//	                                                               |parametricEdgeID^2|
			//	Tangent_Direction(parametricEdgeID) = dx,dy = |A  B_  C_| * |parametricEdgeID  |
			//	                                              |.   .   .|   |1                 |
			"vec2 B_ = B * (numParametricSegments * 2.0);"+
			"vec2 C_ = C * (numParametricSegments * numParametricSegments);"+

			// Run a binary search to determine the highest parametric edge that is located on or before the
			// combinedEdgeID. A combined ID is determined by the sum of complete parametric and radial segments behind
			// it. i.e., find the highest parametric edge where:
			//
			//	parametricEdgeID + floor(numRadialSegmentsAtParametricT) <= combinedEdgeID
			"float lastParametricEdgeID = 0.0;"+
			"float maxParametricEdgeID = min(numParametricSegments - 1.0, combinedEdgeID);"+
			"float negAbsRadsPerSegment = -abs(radsPerSegment);"+
			"float maxRotation0 = (1.0 + combinedEdgeID) * abs(radsPerSegment);"+
			"for (int exp_ = %d - 1; exp_ >= 0; --exp_) {"+
			// Test the parametric edge at lastParametricEdgeID + 2^exp_.
			"float testParametricID = lastParametricEdgeID + exp2(float(exp_));"+
			"if (testParametricID <= maxParametricEdgeID) {"+
			"vec2 testTan = testParametricID * A + B_;"+ // fma(float2(id), A, B_)
			"testTan = testParametricID * testTan + C_;"+ // fma(float2(id), testTan, C_)
			"float cosRotation = dot(normalize(testTan), tan0);"+
			"float maxRotation = testParametricID * negAbsRadsPerSegment + maxRotation0;"+
			"maxRotation = min(maxRotation, PI);"+
			// Is rotation <= maxRotation? (i.e., is the number of complete radial segments behind testT, +
			// testParametricID <= combinedEdgeID?)
			"if (cosRotation >= cos(maxRotation)) {"+
			// testParametricID is on or before the combinedEdgeID. Keep it!
			"lastParametricEdgeID = testParametricID;"+
			"}"+
			"}"+
			"}"+

			// Find the T value of the parametric edge at lastParametricEdgeID.
			"float parametricT = lastParametricEdgeID / numParametricSegments;"+

			// Now that we've identified the highest parametric edge on or before the combinedEdgeID, the highest radial
			// edge is easy:
			"float lastRadialEdgeID = combinedEdgeID - lastParametricEdgeID;"+

			// Find the angle of tan0, i.e. the angle between tan0 and the positive x axis.
			"float angle0 = acos(clamp(tan0.x, -1.0, 1.0));"+
			"angle0 = tan0.y >= 0.0 ? angle0 : -angle0;"+

			// Find the tangent vector on the edge at lastRadialEdgeID. By construction it is already normalized.
			"float radialAngle = lastRadialEdgeID * radsPerSegment + angle0;"+ // fma
			"tangent = vec2(cos(radialAngle), sin(radialAngle));"+
			"vec2 norm = vec2(-tangent.y, tangent.x);"+

			// Find the T value where the tangent is orthogonal to norm. This is a quadratic:
			//
			//	dot(norm, Tangent_Direction(T)) == 0
			//
			//	                    |T^2|
			//	norm * |A  2B  C| * |T  | == 0
			//	       |.   .  .|   |1  |
			"float a=dot(norm,A), b_over_2=dot(norm,B), c=dot(norm,C);"+
			"float discr_over_4 = max(b_over_2*b_over_2 - a*c, 0.0);"+
			"float q = sqrt(discr_over_4);"+
			"if (b_over_2 > 0.0) {"+
			"q = -q;"+
			"}"+
			"q -= b_over_2;"+

			// Roots are q/a and c/q. Since each curve section does not inflect or rotate more than 180 degrees, there
			// can only be one tangent orthogonal to "norm" inside 0..1. Pick the root nearest .5.
			"float _5qa = -.5*q*a;"+
			"vec2 root = (abs(q*q + _5qa) < abs(a*c + _5qa)) ? vec2(q,a) : vec2(c,q);"+ // fma x2
			"float radialT = (root.t != 0.0) ? root.s / root.t : 0.0;"+
			"radialT = clamp(radialT, 0.0, 1.0);"+

			"if (lastRadialEdgeID == 0.0) {"+
			// The root finder above can become unstable when lastRadialEdgeID == 0 (e.g., if there are roots at exactly
			// 0 and 1 both). radialT should always == 0 in this case.
			"radialT = 0.0;"+
			"}"+

			// Now that we've identified the T values of the last parametric and radial edges, our final T value for
			// combinedEdgeID is whichever is larger.
			"float T = max(parametricT, radialT);"+

			// Evaluate the cubic at T. Use De Casteljau's for its accuracy and stability.
			"vec2 ab = unchecked_mix(p0, p1, T);"+
			"vec2 bc = unchecked_mix(p1, p2, T);"+
			"vec2 cd = unchecked_mix(p2, p3, T);"+
			"vec2 abc = unchecked_mix(ab, bc, T);"+
			"vec2 bcd = unchecked_mix(bc, cd, T);"+
			"vec2 abcd = unchecked_mix(abc, bcd, T);"+

			// Evaluate the conic weight at T.
			"float u = unchecked_mix(1.0, w, T);"+
			"float v = w + 1.0 - u;"+ // == mix(w, 1, T)
			"float uv = unchecked_mix(u, v, T);"+

			// If we went with T=parametricT, then update the tangent. Otherwise leave it at the radial tangent found
			// previously. (In the event that parametricT == radialT, we keep the radial tangent.)
			"if (T != radialT) {"+
			// We must re-normalize here because the tangent is determined by the curve coefficients.
			"tangent = w >= 0.0 ? robust_normalize_diff(bc*u, ab*v)"+
			": robust_normalize_diff(bcd, abc);"+
			"}"+

			"strokeCoord = (w >= 0.0) ? abc/uv : abcd;"+
			"} else {"+
			// Edges at the beginning and end of the strip use exact endpoints and tangents. This ensures crack-free
			// seaming between instances.
			"tangent = (combinedEdgeID == 0.0) ? tan0 : tan1;"+
			"strokeCoord = (combinedEdgeID == 0.0) ? p0 : p3;"+
			"}", tessMaxResolveLevel, /* Parametric/radial sort loop count. */
	)

	v.CodeAppend(
		// At this point 'tangent' is normalized, so the orthogonal vector is also normalized.
		"vec2 ortho = vec2(tangent.y, -tangent.x);" +
			"strokeCoord += ortho * (STROKE_RADIUS * strokeOutset);",
	)

	if !shader.stroke.IsHairlineStyle() {
		// Normal case. Do the transform after tessellation.
		v.CodeAppend("vec2 devCoord = AFFINE_MATRIX * strokeCoord + TRANSLATE;")
		gpArgs.PositionVar.Set(GLSLTypeFloat2, "devCoord")
		gpArgs.LocalCoordVar.Set(GLSLTypeFloat2, "strokeCoord")
	} else {
		// Hairline case. The scale and skew already happened before tessellation.
		v.CodeAppend(
			"vec2 devCoord = strokeCoord + TRANSLATE;" +
				"vec2 localCoord = inverse(AFFINE_MATRIX) * strokeCoord;",
		)
		gpArgs.PositionVar.Set(GLSLTypeFloat2, "devCoord")
		gpArgs.LocalCoordVar.Set(GLSLTypeFloat2, "localCoord")
	}
}

// emitFragmentCode emits the fragment shader body that outputs the stroke's color and coverage.
func (i *strokeTessellationShaderImpl) emitFragmentCode(shader *strokeTessellationShader, args *GPEmitArgs) {
	if !shader.hasDynamicColor() {
		// The fragment shader just outputs a uniform color.
		var colorUniformName string
		i.colorUniform, colorUniformName = args.UniformHandler.AddUniform(nil,
			ShaderFlagFragment, GLSLTypeHalf4, "color")
		args.FragBuilder.CodeAppendf("vec4 %s = %s;", args.OutputColor, colorUniformName)
	} else {
		args.FragBuilder.CodeAppendf("vec4 %s = %s;", args.OutputColor, i.dynamicColorName)
	}
	args.FragBuilder.CodeAppendf("const vec4 %s = vec4(1.0);", args.OutputCoverage)
}

// SetData implements GPProgramImpl.
func (i *strokeTessellationShaderImpl) SetData(pdman *ProgramDataManager, _ *gpu.ShaderCaps, geomProc GeometryProcessor) {
	shader := geomProc.(*strokeTessellationShader)
	rec := &shader.stroke

	// getMaxScale() returns -1 if it can't compute a scale factor (e.g. perspective); taking the absolute value
	// automatically converts that to an identity scale factor for our purposes.
	maxScale := geom.ScalarAbs(shader.viewMatrix.MaxScale())
	if !shader.hasDynamicStroke() {
		// Set up the tessellation control uniforms. In the hairline case we transform prior to tessellation, so it will
		// be defined in device space units instead of local units.
		strokeRadius := 0.5 * rec.Width()
		scale := maxScale
		if rec.IsHairlineStyle() {
			strokeRadius = 0.5
			scale = 1
		}
		numRadialSegmentsPerRadian := tessCalcNumRadialSegmentsPerRadian(scale * strokeRadius)
		pdman.Set3f(i.tessControlArgsUniform,
			numRadialSegmentsPerRadian, // NUM_RADIAL_SEGMENTS_PER_RADIAN
			tessGetJoinType(rec),       // JOIN_TYPE
			strokeRadius)               // STROKE_RADIUS
	} else {
		if rec.IsHairlineStyle() {
			panic("dynamic hairlines are not supported")
		}
		pdman.Set1f(i.tessControlArgsUniform, maxScale)
	}

	// Set up the view matrix, if any.
	m := &shader.viewMatrix
	pdman.Set2f(i.translateUniform, m.Get(geom.MTransX), m.Get(geom.MTransY))
	pdman.Set4f(i.affineMatrixUniform, m.Get(geom.MScaleX), m.Get(geom.MSkewY),
		m.Get(geom.MSkewX), m.Get(geom.MScaleY))

	if !shader.hasDynamicColor() {
		pdman.Set4f(i.colorUniform, shader.color.R, shader.color.G, shader.color.B,
			shader.color.A)
	}
}
