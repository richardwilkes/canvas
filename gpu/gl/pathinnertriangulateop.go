// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The inner-fan path-fill op: a 3-pass twist on the standard Redbook "stencil then cover" algorithm:
//
//  1. Tessellate the path's outer curves into the stencil buffer.
//  2. Triangulate the path's inner fan and fill it with a stencil test against the curves.
//  3. Draw convex hulls around each curve that fill in remaining samples.
//
// In practice, a path's inner fan takes up a large majority of its pixels, so from a GPU load perspective this op is
// effectively as fast as a single-pass algorithm. Trims: the DDL-only onPrePrepare lane is dropped; the no-gl_VertexID
// strip-order vertex buffer is unreachable on desktop core profiles.

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
)

//////////////////////////////////////////////////////////////////////////////
// innerFanTriangulator

// innerFanTriangulator triangulates the inner polygon(s) of a path (i.e., the triangle fan for a Redbook rendering
// method). When combined with the outer curves and breadcrumb triangles, these produce a complete path. If a breadcrumb
// collector is not provided, pathToPolys fails upon self intersection.
type innerFanTriangulator struct {
	tri *triangulator
}

func newInnerFanTriangulator(p *path.Path) *innerFanTriangulator {
	tri := &triangulator{path: p}
	tri.preserveCollinearVertices = true
	tri.collectBreadcrumbTriangles = true
	return &innerFanTriangulator{tri: tri}
}

// pathToPolys triangulates the path's inner fan into a linked list of polygons.
func (f *innerFanTriangulator) pathToPolys(breadcrumbList *breadcrumbTriangleList, isLinear *bool) *triPoly {
	polys, success := f.tri.pathToPolys(0, geom.Rect{}, isLinear)
	if !success {
		return nil
	}
	breadcrumbList.concat(&f.tri.breadcrumbList)
	return polys
}

// polysToTriangles emits polys as triangles through vertexAlloc, returning the vertex count written.
func (f *innerFanTriangulator) polysToTriangles(polys *triPoly, vertexAlloc eagerVertexAllocator, breadcrumbList *breadcrumbTriangleList) int {
	vertexCount := f.tri.polysToTriangles(polys, vertexAlloc)
	breadcrumbList.concat(&f.tri.breadcrumbList)
	return vertexCount
}

//////////////////////////////////////////////////////////////////////////////
// HullShader

// hullShader fills an array of convex hulls surrounding 4-point cubic or conic instances. This shader is used for the
// "cover" pass after the curves have been fully stenciled.
type hullShader struct {
	instanceAttrs [3]Attribute // p01, p23, curveType
	tessellationShaderBase
	infinitySupport bool
}

func makeHullShader(viewMatrix *geom.Matrix, color colorcore.PMColor4f, shaderCaps *gpu.ShaderCaps) *hullShader {
	s := &hullShader{infinitySupport: shaderCaps.InfinitySupport}
	s.initTessellationShader(TessellateHullShaderClassID, gpu.PrimitiveTypeTriangleStrip,
		viewMatrix, color)
	n := 0
	s.instanceAttrs[n] = MakeAttribute("p01", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	n++
	s.instanceAttrs[n] = MakeAttribute("p23", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	n++
	if !shaderCaps.InfinitySupport {
		// A conic curve is written out with p3=[w,Infinity], but GPUs that don't support infinity can't detect this. On
		// these platforms we also write out an extra float with each patch that explicitly tells the shader what type
		// of curve it is.
		s.instanceAttrs[n] = MakeAttribute("curveType", gpu.VertexAttribTypeFloat, GLSLTypeFloat)
		n++
	}
	s.setInstanceAttributesWithImplicitOffsets(s.instanceAttrs[:n])
	return s
}

func (s *hullShader) Name() string { return "tessellate_HullShader" }

func (s *hullShader) tessBase() *tessellationShaderBase { return &s.tessellationShaderBase }

func (s *hullShader) shaderAttribs() PatchAttribs { return PatchAttribsNone }

// AddToKey implements GeometryProcessor (nothing beyond the class/attribute key).
func (s *hullShader) AddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

// MakeProgramImpl implements GeometryProcessor.
func (s *hullShader) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &pathTessShaderImpl{emitVertexCode: hullShaderEmitVertexCode}
}

// hullShaderEmitVertexCode emits the vertex shader that computes each hull instance's convex-hull vertex position.
func hullShaderEmitVertexCode(args *GPEmitArgs, _ *pathTessShaderImpl, gpArgs *GPArgs) {
	shader := args.GeomProc.(*hullShader)
	v := args.VertBuilder
	v.InsertFunction(
		"float cross_length_2d(vec2 a, vec2 b) { return determinant(mat2(a, b)); }",
	)
	if shader.infinitySupport {
		v.InsertFunction(
			"bool is_conic_curve() { return isinf(p23.w); }" +
				"bool is_non_triangular_conic_curve() {" +
				// We consider a conic non-triangular as long as its weight isn't infinity. NOTE: "isinf == false" works
				// on Mac Radeon GLSL; "!isinf" can get the wrong answer.
				"return isinf(p23.z) == false;" +
				"}",
		)
	} else {
		v.InsertFunction(fmt.Sprintf(
			"bool is_conic_curve() { return curveType != %g; }", tessCubicCurveType,
		))
		v.InsertFunction(fmt.Sprintf(
			"bool is_non_triangular_conic_curve() { return curveType == %g; }",
			tessConicCurveType,
		))
	}
	v.CodeAppend(
		"vec2 p0=p01.xy, p1=p01.zw, p2=p23.xy, p3=p23.zw;" +
			"if (is_conic_curve()) {" +
			// Conics are 3 points, with the weight in p3.
			"float w = p3.x;" +
			"p3 = p2;" + // Duplicate the endpoint for shared code that also runs on cubics.
			"if (is_non_triangular_conic_curve()) {" +
			// Convert the points to a trapeziodal hull that circumscribes the conic.
			"vec2 p1w = p1 * w;" +
			"float T_ = .51;" + // Bias outward a bit to ensure we cover the outermost samples.
			"vec2 c1 = mix(p0, p1w, T_);" +
			"vec2 c2 = mix(p2, p1w, T_);" +
			"float iw = 1.0 / mix(1.0, w, T_);" +
			"p2 = c2 * iw;" +
			"p1 = c1 * iw;" +
			"}" +
			"}" +

			// Translate the points to v0..3 where v0=0.
			"vec2 v1 = p1 - p0;" +
			"vec2 v2 = p2 - p0;" +
			"vec2 v3 = p3 - p0;" +

			// Reorder the points so v2 bisects v1 and v3.
			"if (sign(cross_length_2d(v2, v1)) == sign(cross_length_2d(v2, v3))) {" +
			"vec2 tmp = p2;" +
			"if (sign(cross_length_2d(v1, v2)) != sign(cross_length_2d(v1, v3))) {" +
			"p2 = p1;" + // swap(p2, p1)
			"p1 = tmp;" +
			"} else {" +
			"p2 = p3;" + // swap(p2, p3)
			"p3 = tmp;" +
			"}" +
			"}",
	)

	// The vertex ID comes in fan order. Convert to strip order (gl_VertexID is always available on desktop core
	// profiles).
	v.CodeAppend(
		"int vertexidx = gl_VertexID;" +
			"vertexidx ^= vertexidx >> 1;",
	)

	v.CodeAppend(
		// Find the "turn direction" of each corner and net turn direction.
		"float vertexdir = 0.0;" +
			"float netdir = 0.0;" +
			"vec2 prev, next;" +
			"float dir;" +
			"vec2 localcoord;" +
			"vec2 nextcoord;",
	)

	for i := 0; i < 4; i++ {
		v.CodeAppendf("prev = p%d - p%d;", i, (i+3)%4)
		v.CodeAppendf("next = p%d - p%d;", (i+1)%4, i)
		v.CodeAppendf(
			"dir = sign(cross_length_2d(prev, next));"+
				"if (vertexidx == %d) {"+
				"vertexdir = dir;"+
				"localcoord = p%d;"+
				"nextcoord = p%d;"+
				"}"+
				"netdir += dir;", i, i, (i+1)%4,
		)
	}

	v.CodeAppend(
		// Remove the non-convex vertex, if any.
		"if (vertexdir != sign(netdir)) {" +
			"localcoord = nextcoord;" +
			"}" +

			"vec2 vertexpos = AFFINE_MATRIX * localcoord + TRANSLATE;",
	)
	gpArgs.LocalCoordVar.Set(GLSLTypeFloat2, "localcoord")
	gpArgs.PositionVar.Set(GLSLTypeFloat2, "vertexpos")
}

//////////////////////////////////////////////////////////////////////////////
// PathInnerTriangulateOp

var pathInnerTriangulateOpClassID = GenOpClassID()

// pathInnerTriangulateOp fills a path using the inner-fan stencil-then-cover algorithm described at the top of this
// file.
type pathInnerTriangulateOp struct {
	drawOpNoClipToShape
	// This buffer gets created by the fan triangulator during onPrepare.
	fanBuffer AnyBuffer
	// Triangulates the inner fan.
	fanTriangulator *innerFanTriangulator
	// This pipeline is shared by all programs that do filling.
	pipelineForFills *Pipeline
	path             *path.Path
	// Pass 3: draw convex hulls around each curve.
	coverHullsProgram *ProgramInfo
	// Pass 1: tessellate the outer curves into the stencil buffer.
	stencilCurvesProgram *ProgramInfo
	processors           *ProcessorSet
	// Tessellates the outer curves.
	tessellator    *pathCurveTessellator
	fanPolys       *triPoly
	fanBreadcrumbs breadcrumbTriangleList
	// Pass 2: fill the path's inner fan with a stencil test against the curves. (In extenuating circumstances this
	// might require two separate draws.)
	fanPrograms []*ProgramInfo
	OpBase
	baseFanVertex  int
	fanVertexCount int
	viewMatrix     geom.Matrix
	color          colorcore.PMColor4f
	aaType         gpu.AAType
	pathFlags      FillPathFlags
}

// newPathInnerTriangulateOp builds a pathInnerTriangulateOp. The paint is consumed; the path must not be inverse
// filled.
func newPathInnerTriangulateOp(viewMatrix *geom.Matrix, p *path.Path, paint *Paint, aaType gpu.AAType, pathFlags FillPathFlags, drawBounds geom.Rect) *pathInnerTriangulateOp {
	if p.IsInverseFillType() {
		panic("PathInnerTriangulateOp cannot draw inverse fills")
	}
	o := &pathInnerTriangulateOp{
		pathFlags:  pathFlags,
		viewMatrix: *viewMatrix,
		path:       p,
		aaType:     aaType,
		color:      paint.Color4f(),
	}
	o.InitOp(pathInnerTriangulateOpClassID, o)
	o.processors = NewProcessorSetFromPaint(paint)
	o.SetBounds(drawBounds, HasAABloat(false), IsHairline(false))
	return o
}

// Name implements Op.
func (o *pathInnerTriangulateOp) Name() string { return "PathInnerTriangulateOp" }

// VisitProxies implements Op.
func (o *pathInnerTriangulateOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.pipelineForFills != nil {
		o.pipelineForFills.VisitProxies(fn)
	} else {
		o.processors.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *pathInnerTriangulateOp) UsesMSAA() bool { return o.aaType != gpu.AATypeNone }

// UsesStencil implements DrawOp (fixedFunctionFlags always sets kUsesStencil).
func (o *pathInnerTriangulateOp) UsesStencil() bool { return true }

// Finalize implements DrawOp.
func (o *pathInnerTriangulateOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	color := AnalysisColorConstant(o.color)
	analysis, overrideColor := o.processors.Finalize(color, AnalysisCoverageNone, clip, nil,
		caps, clampType)
	if analysis.InputColorIsOverridden() {
		o.color = overrideColor
	}
	return analysis
}

// OnCombineIfPossible implements Op (PathInnerTriangulateOp does not combine).
func (o *pathInnerTriangulateOp) OnCombineIfPossible(Op) CombineResult {
	return CombineResultCannotCombine
}

// pushFanStencilProgram appends a program that stencils the inner fan with pipelineForStencils.
func (o *pathInnerTriangulateOp) pushFanStencilProgram(args *tessProgramArgs, pipelineForStencils *Pipeline, stencil *UserStencilSettings) {
	if pipelineForStencils == nil {
		panic("fan stencil program requires a stencil pipeline")
	}
	shader := makeSimpleTriangleShader(&o.viewMatrix, colorcore.PMColor4f{})
	o.fanPrograms = append(o.fanPrograms, tessMakeProgram(args, shader, shader.primitiveType,
		pipelineForStencils, stencil))
}

// pushFanFillProgram appends a program that fills the inner fan directly into pipelineForFills.
func (o *pathInnerTriangulateOp) pushFanFillProgram(args *tessProgramArgs, stencil *UserStencilSettings) {
	if o.pipelineForFills == nil {
		panic("fan fill program requires the fill pipeline")
	}
	shader := makeSimpleTriangleShader(&o.viewMatrix, o.color)
	o.fanPrograms = append(o.fanPrograms, tessMakeProgram(args, shader, shader.primitiveType,
		o.pipelineForFills, stencil))
}

// Twists on the standard Redbook stencil settings that allow us to fill the inner polygon directly to the final render
// target. By the time these programs execute, the outer curves will already be stenciled in, so if the stencil value
// is zero then it means the sample in question is not affected by any curves and we can fill it in directly; if the
// stencil value is nonzero, we don't fill and instead continue the standard Redbook counting process.
var innerTriangulateFillOrIncrDecrStencil = MakeUserStencilSettingsSeparate(
	UserStencilFace{
		Ref: 0x0000, Test: UserStencilTestEqual, TestMask: 0xffff,
		PassOp: UserStencilOpKeep, FailOp: UserStencilOpIncWrap, WriteMask: 0xffff,
	},
	UserStencilFace{
		Ref: 0x0000, Test: UserStencilTestEqual, TestMask: 0xffff,
		PassOp: UserStencilOpKeep, FailOp: UserStencilOpDecWrap, WriteMask: 0xffff,
	},
)

var innerTriangulateFillOrInvertStencil = MakeUserStencilSettings(0x0000, UserStencilTestEqual,
	0xffff,
	UserStencilOpKeep,
	// "Zero" instead of "Invert" because the fan only touches any given pixel once.
	UserStencilOpZero,
	0xffff)

// The two-pass variant used when there is a stencil clip: the stencil test isn't expressive enough to do the above
// tests and also check the clip bit in a single pass.
var innerTriangulateFillIfZeroAndInClip = MakeUserStencilSettings(0x0000,
	UserStencilTestEqualIfInClip, 0xffff, UserStencilOpKeep, UserStencilOpKeep, 0xffff)

var innerTriangulateIncrDecrStencilIfNonzero = MakeUserStencilSettingsSeparate(
	// No need to check the clip because the previous stencil pass will have only written to samples already inside the
	// clip.
	UserStencilFace{
		Ref: 0x0000, Test: UserStencilTestNotEqual, TestMask: 0xffff,
		PassOp: UserStencilOpIncWrap, FailOp: UserStencilOpKeep, WriteMask: 0xffff,
	},
	UserStencilFace{
		Ref: 0x0000, Test: UserStencilTestNotEqual, TestMask: 0xffff,
		PassOp: UserStencilOpDecWrap, FailOp: UserStencilOpKeep, WriteMask: 0xffff,
	},
)

var innerTriangulateInvertStencilIfNonZero = MakeUserStencilSettings(0x0000,
	UserStencilTestNotEqual, 0xffff,
	// "Zero" instead of "Invert" because the fan only touches any given pixel once.
	UserStencilOpZero,
	UserStencilOpKeep,
	0xffff)

// prePreparePrograms builds the stencil/fan/cover programs for this op. The applied clip is consumed.
func (o *pathInnerTriangulateOp) prePreparePrograms(args *tessProgramArgs, appliedClip *AppliedClip) {
	if o.fanTriangulator != nil || o.fanPolys != nil || o.pipelineForFills != nil ||
		o.tessellator != nil || o.stencilCurvesProgram != nil || len(o.fanPrograms) != 0 ||
		o.coverHullsProgram != nil {
		panic("prePreparePrograms called twice")
	}
	if o.path.CountVerbs() <= 0 {
		return
	}

	// If using wireframe, we have to fall back on a standard Redbook "stencil then cover" algorithm instead of
	// bypassing the stencil buffer to fill the fan directly.
	forceRedbookStencilPass := o.pathFlags&(FillPathFlagStencilOnly|FillPathFlagWireframe) != 0
	doFill := o.pathFlags&FillPathFlagStencilOnly == 0

	var isLinear bool
	o.fanTriangulator = newInnerFanTriangulator(o.path)
	o.fanPolys = o.fanTriangulator.pathToPolys(&o.fanBreadcrumbs, &isLinear)

	// Create a pipeline for stencil passes if needed.
	var pipelineForStencils *Pipeline
	if forceRedbookStencilPass || !isLinear { // Curves always get stenciled.
		pipelineFlags := PipelineInputFlagNone
		if o.pathFlags&FillPathFlagWireframe != 0 {
			pipelineFlags = PipelineWireframe
		}
		pipelineForStencils = tessMakeStencilOnlyPipeline(args, appliedClip, pipelineFlags)
	}

	// Create a pipeline for fill passes if needed.
	if doFill {
		o.pipelineForFills = tessMakePipeline(args, o.processors, appliedClip)
	}

	// Pass 1: tessellate the outer curves into the stencil buffer.
	if !isLinear {
		o.tessellator = newPathCurveTessellator(args.caps.ShaderCaps.InfinitySupport,
			PatchAttribsNone)
		tessShader := makeMiddleOutShader(args.caps.ShaderCaps, &o.viewMatrix,
			colorcore.PMColor4f{}, o.tessellator.patchAttribs())
		o.stencilCurvesProgram = tessMakeProgram(args, tessShader, tessShader.primitiveType,
			pipelineForStencils, tessStencilPathSettings(o.path.FillType()))
	}

	// Pass 2: fill the path's inner fan with a stencil test against the curves.
	if o.fanPolys != nil {
		switch {
		case forceRedbookStencilPass:
			// Use a standard Redbook "stencil then cover" algorithm instead of bypassing the stencil buffer to fill the
			// fan directly.
			o.pushFanStencilProgram(args, pipelineForStencils,
				tessStencilPathSettings(o.path.FillType()))
			if doFill {
				o.pushFanFillProgram(args, tessTestAndResetStencilSettings(false))
			}
		case isLinear:
			// There are no outer curves! Ignore stencil and fill the path directly.
			if pipelineForStencils != nil {
				panic("linear paths must not have a stencil pipeline here")
			}
			o.pushFanFillProgram(args, UnusedStencilSettings())
		case !o.pipelineForFills.HasStencilClip():
			var stencil *UserStencilSettings
			if o.path.FillType() == path.FillWinding {
				stencil = &innerTriangulateFillOrIncrDecrStencil
			} else {
				stencil = &innerTriangulateFillOrInvertStencil
			}
			o.pushFanFillProgram(args, stencil)
		default:
			// Pass 2a: directly fill fan samples whose stencil values (from curves) are zero.
			o.pushFanFillProgram(args, &innerTriangulateFillIfZeroAndInClip)
			// Pass 2b: Redbook counting on fan samples whose stencil values (from curves) != 0.
			var stencil *UserStencilSettings
			if o.path.FillType() == path.FillWinding {
				stencil = &innerTriangulateIncrDecrStencilIfNonzero
			} else {
				stencil = &innerTriangulateInvertStencilIfNonZero
			}
			o.pushFanStencilProgram(args, pipelineForStencils, stencil)
		}
	}

	// Pass 3: draw convex hulls around each curve.
	if doFill && !isLinear {
		// By the time this program executes, every pixel will be filled in except the ones touched by curves. We issue
		// a final cover pass over the curves by drawing their convex hulls. This will fill in any remaining samples and
		// reset the stencil values back to zero.
		if o.tessellator == nil {
			panic("curves require a tessellator")
		}
		shader := makeHullShader(&o.viewMatrix, o.color, args.caps.ShaderCaps)
		o.coverHullsProgram = tessMakeProgram(args, shader, shader.primitiveType,
			o.pipelineForFills, tessTestAndResetStencilSettings(false))
	}
}

// OnPrepare implements Op.
func (o *pathInnerTriangulateOp) OnPrepare(state *OpFlushState) {
	if o.fanTriangulator == nil {
		args := makeTessProgramArgs(state)
		o.prePreparePrograms(&args, state.OpArgs().AppliedClip())
		if o.fanTriangulator == nil {
			return
		}
	}

	if o.fanPolys != nil {
		alloc := newEagerDynamicVertexAllocator(state)
		o.fanVertexCount = o.fanTriangulator.polysToTriangles(o.fanPolys, alloc,
			&o.fanBreadcrumbs)
		o.fanBuffer = alloc.buffer
		o.baseFanVertex = alloc.firstVertex
	}

	if o.tessellator != nil {
		tessShader := o.stencilCurvesProgram.GeomProc().(*middleOutShader)
		o.tessellator.prepareWithTriangles(state, &tessShader.viewMatrix, &o.fanBreadcrumbs,
			&pathDrawList{
				pathMatrix: geom.IdentityMatrix(), path: o.path,
				color: colorcore.PMColor4f{},
			},
			o.path.CountVerbs())
	}
}

// OnExecute implements Op.
func (o *pathInnerTriangulateOp) OnExecute(state *OpFlushState, _chainBounds geom.Rect) {
	renderPass := state.OpsRenderPass()
	scissor := func(info *ProgramInfo) {
		if info.Pipeline().IsScissorTestEnabled() {
			renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
		}
	}

	if o.stencilCurvesProgram != nil {
		if o.tessellator == nil {
			panic("stencil curves without a tessellator")
		}
		if renderPass.BindPipeline(o.stencilCurvesProgram, o.Bounds()) {
			scissor(o.stencilCurvesProgram)
			o.tessellator.draw(state)
		}
	}

	// Allocation of the fan vertex buffer may have failed but we already pushed back fan programs.
	if o.fanBuffer != nil {
		for _, fanProgram := range o.fanPrograms {
			if !renderPass.BindPipeline(fanProgram, o.Bounds()) {
				continue
			}
			scissor(fanProgram)
			renderPass.BindTextures(fanProgram.GeomProc(), nil, fanProgram.Pipeline())
			renderPass.BindBuffers(nil, nil, o.fanBuffer)
			renderPass.Draw(o.fanVertexCount, o.baseFanVertex)
		}
	}

	if o.coverHullsProgram != nil {
		if o.tessellator == nil {
			panic("cover hulls without a tessellator")
		}
		if renderPass.BindPipeline(o.coverHullsProgram, o.Bounds()) {
			scissor(o.coverHullsProgram)
			renderPass.BindTextures(o.coverHullsProgram.GeomProc(), nil, o.pipelineForFills)
			o.tessellator.drawHullInstances(state)
		}
	}
}
