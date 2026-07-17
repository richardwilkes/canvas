// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// pathStencilCoverOp draws paths using a standard Redbook "stencil then cover" method — the path is stenciled through
// the fixed-count tessellator (large complex paths additionally stencil their inner fans with a dedicated triangle
// shader in middle-out topology), then a bounding box is drawn over the path that fills its stencil coverage into the
// color buffer. This op does not apply analytic AA, so it requires MSAA if AA is desired. There is no atlas-only
// draw-list constructor: the atlas path renderer stays trimmed, as an optional DMSAA atlasing optimization. The
// no-gl_VertexID unit-quad vertex buffer fallback is unreachable on desktop core profiles.

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
)

//////////////////////////////////////////////////////////////////////////////
// BoundingBoxShader

// boundingBoxShader fills a path's bounding box, with subpixel outset to avoid possible T-junctions with extreme edges
// of the path. NOTE: the emitted geometry may not be axis-aligned, depending on the view matrix. There is no per-vertex
// unitCoord attribute lane: desktop core profiles always have gl_VertexID, so the vertex position is derived from that
// instead.
type boundingBoxShader struct {
	instanceAttrs [3]Attribute
	GPBase
	color colorcore.PMColor4f
}

// bboxInstanceStride is boundingBoxShader's instance stride: matrix2d (float4) + translate (float2) + pathBounds
// (float4).
const bboxInstanceStride = 16 + 8 + 16

func makeBoundingBoxShader(color colorcore.PMColor4f) *boundingBoxShader {
	s := &boundingBoxShader{color: color}
	s.initGP(TessellateBoundingBoxShaderClassID)
	s.instanceAttrs[0] = MakeAttribute("matrix2d", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	s.instanceAttrs[1] = MakeAttribute("translate", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	s.instanceAttrs[2] = MakeAttribute("pathBounds", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	s.setInstanceAttributesWithImplicitOffsets(s.instanceAttrs[:])
	if s.InstanceStride() != bboxInstanceStride {
		panic("bounding box instance stride mismatch")
	}
	return s
}

func (s *boundingBoxShader) Name() string { return "tessellate_BoundingBoxShader" }

// AddToKey implements GeometryProcessor (nothing beyond the class/attribute key).
func (s *boundingBoxShader) AddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

// MakeProgramImpl implements GeometryProcessor.
func (s *boundingBoxShader) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &boundingBoxShaderImpl{}
}

type boundingBoxShaderImpl struct {
	GPImplBase
	colorUniform UniformHandle
}

func (i *boundingBoxShaderImpl) SetData(pdman *ProgramDataManager, _ *gpu.ShaderCaps, geomProc GeometryProcessor) {
	color := geomProc.(*boundingBoxShader).color
	pdman.Set4f(i.colorUniform, color.R, color.G, color.B, color.A)
}

func (i *boundingBoxShaderImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	args.VaryingHandler.EmitAttributes(args.GeomProc)

	// Vertex shader (gl_VertexID is always available on desktop core profiles).
	args.VertBuilder.CodeAppend("vec2 unitCoord = vec2(gl_VertexID & 1, gl_VertexID >> 1);")
	args.VertBuilder.CodeAppend(
		// Bloat the bounding box by 1/4px to be certain we will reset every stencil value.
		"mat2 M_ = inverse(mat2(matrix2d.xy, matrix2d.zw));" +
			"vec2 bloat = vec2(abs(M_[0]) + abs(M_[1])) * .25;" +

			// Find the vertex position.
			"vec2 localcoord = mix(pathBounds.xy - bloat, pathBounds.zw + bloat, unitCoord);" +
			"vec2 vertexpos = mat2(matrix2d.xy, matrix2d.zw) * localcoord + translate;",
	)
	gpArgs.LocalCoordVar.Set(GLSLTypeFloat2, "localcoord")
	gpArgs.PositionVar.Set(GLSLTypeFloat2, "vertexpos")

	// Fragment shader.
	var color string
	i.colorUniform, color = args.UniformHandler.AddUniform(nil, ShaderFlagFragment, GLSLTypeHalf4, "color")
	args.FragBuilder.CodeAppendf("vec4 %s = %s;", args.OutputColor, color)
	args.FragBuilder.CodeAppendf("const vec4 %s = vec4(1);", args.OutputCoverage)
}

//////////////////////////////////////////////////////////////////////////////
// PathStencilCoverOp

var pathStencilCoverOpClassID = GenOpClassID()

// pathStencilCoverOp draws one or more paths with the stencil-then-cover method described at the top of this file. If
// the path is inverse filled, drawBounds must be the entire backing-store dimensions of the render target.
type pathStencilCoverOp struct {
	drawOpNoClipToShape
	// Decided during prePreparePrograms.
	tessellator pathTessellator
	bboxBuffer  AnyBuffer
	// Filled during onPrepare.
	fanBuffer          AnyBuffer
	stencilPathProgram *ProgramInfo
	stencilFanProgram  *ProgramInfo
	pathDrawList       *pathDrawList
	coverBBoxProgram   *ProgramInfo
	processors         *ProcessorSet
	curveTessellator   *pathCurveTessellator // non-nil when tessellator is the curve variant
	OpBase
	pathCount                int
	totalCombinedPathVerbCnt int
	fanBaseVertex            int
	fanVertexCount           int
	bboxBaseInstance         int
	color                    colorcore.PMColor4f
	aaType                   gpu.AAType
	pathFlags                FillPathFlags
}

// newPathStencilCoverOp creates a pathStencilCoverOp for a single path. The paint is consumed.
func newPathStencilCoverOp(viewMatrix *geom.Matrix, p *path.Path, paint *Paint, aaType gpu.AAType, pathFlags FillPathFlags, drawBounds geom.Rect) *pathStencilCoverOp {
	o := &pathStencilCoverOp{
		totalCombinedPathVerbCnt: p.CountVerbs(),
		pathCount:                1,
		pathFlags:                pathFlags,
		aaType:                   aaType,
		color:                    paint.Color4f(),
	}
	o.InitOp(pathStencilCoverOpClassID, o)
	o.pathDrawList = &pathDrawList{
		pathMatrix: *viewMatrix, path: p,
		color: colorcore.PMColor4f{},
	}
	o.processors = NewProcessorSetFromPaint(paint)
	o.SetBounds(drawBounds, HasAABloat(false), IsHairline(false))
	return o
}

// pathFillType returns the fill type shared by all paths in the draw list (they are required to all have the same fill
// type).
func (o *pathStencilCoverOp) pathFillType() path.FillType {
	return o.pathDrawList.path.FillType()
}

// Name implements Op.
func (o *pathStencilCoverOp) Name() string { return "PathStencilCoverOp" }

// VisitProxies implements Op.
func (o *pathStencilCoverOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.coverBBoxProgram != nil {
		o.coverBBoxProgram.VisitFPProxies(fn)
	} else {
		o.processors.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp (fixedFunctionFlags' kUsesHWAA bit).
func (o *pathStencilCoverOp) UsesMSAA() bool { return o.aaType != gpu.AATypeNone }

// UsesStencil implements DrawOp (fixedFunctionFlags always sets kUsesStencil).
func (o *pathStencilCoverOp) UsesStencil() bool { return true }

// Finalize implements DrawOp: runs the processor set's finalization pass to compute the op's coverage/color analysis.
func (o *pathStencilCoverOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	color := AnalysisColorConstant(o.color)
	analysis, overrideColor := o.processors.Finalize(color, AnalysisCoverageNone, clip, nil,
		caps, clampType)
	if analysis.InputColorIsOverridden() {
		o.color = overrideColor
	}
	return analysis
}

// OnCombineIfPossible implements Op (PathStencilCoverOp does not combine).
func (o *pathStencilCoverOp) OnCombineIfPossible(Op) CombineResult {
	return CombineResultCannotCombine
}

// prePreparePrograms chooses the rendering method and creates the corresponding tessellator and stencil/cover programs.
// The applied clip is consumed.
func (o *pathStencilCoverOp) prePreparePrograms(args *tessProgramArgs, appliedClip *AppliedClip) {
	if o.tessellator != nil {
		panic("prePreparePrograms called twice")
	}

	// We transform paths on the CPU. This allows for better batching.
	shaderMatrix := geom.IdentityMatrix()
	pipelineFlags := PipelineInputFlagNone
	if o.pathFlags&FillPathFlagWireframe != 0 {
		pipelineFlags = PipelineWireframe
	}
	stencilPipeline := tessMakeStencilOnlyPipeline(args, appliedClip, pipelineFlags)
	stencilSettings := tessStencilPathSettings(o.pathFillType())

	if o.totalCombinedPathVerbCnt > 50 &&
		o.Bounds().Height()*o.Bounds().Width() > 256*256 {
		// Large complex paths do better with a dedicated triangle shader for the inner fan. This takes less PCI bus
		// bandwidth (6 floats per triangle instead of 8) and allows us to make sure it has an efficient middle-out
		// topology.
		shader := makeSimpleTriangleShader(&shaderMatrix, colorcore.PMColor4f{})
		o.stencilFanProgram = tessMakeProgram(args, shader, shader.primitiveType,
			stencilPipeline, stencilSettings)
		o.curveTessellator = newPathCurveTessellator(args.caps.ShaderCaps.InfinitySupport,
			PatchAttribsNone)
		o.tessellator = o.curveTessellator
	} else {
		o.tessellator = newPathWedgeTessellator(args.caps.ShaderCaps.InfinitySupport,
			PatchAttribsNone)
	}
	tessShader := makeMiddleOutShader(args.caps.ShaderCaps, &shaderMatrix, colorcore.PMColor4f{},
		o.tessellator.patchAttribs())
	o.stencilPathProgram = tessMakeProgram(args, tessShader, tessShader.primitiveType,
		stencilPipeline, stencilSettings)

	if o.pathFlags&FillPathFlagStencilOnly == 0 {
		// Create a program that draws a bounding box over the path and fills its stencil coverage into the color
		// buffer.
		bboxShader := makeBoundingBoxShader(o.color)
		bboxPipeline := tessMakePipeline(args, o.processors, appliedClip)
		bboxStencil := tessTestAndResetStencilSettings(o.pathFillType().IsInverse())
		o.coverBBoxProgram = NewProgramInfo(args.caps, args.writeView, args.usesMSAASurface,
			bboxPipeline, bboxStencil, bboxShader, gpu.PrimitiveTypeTriangleStrip,
			args.xferBarrierFlags, args.colorLoadOp)
	}
}

// OnPrepare implements Op: builds the programs (if not already built), the inner-fan vertex buffer, and the
// bounding-box instance buffer.
func (o *pathStencilCoverOp) OnPrepare(state *OpFlushState) {
	if o.tessellator == nil {
		args := makeTessProgramArgs(state)
		o.prePreparePrograms(&args, state.OpArgs().AppliedClip())
		if o.tessellator == nil {
			return
		}
	}

	if o.stencilFanProgram != nil {
		// The inner fan isn't built into the tessellator. Generate a standard Redbook fan with a middle-out topology.
		// Path fans might have an extra edge from an implicit kClose at the end, but they also always begin with kMove,
		// so the max possible number of edges in a single path is equal to the number of verbs; therefore the max
		// number of combined fan edges in a path list is the number of combined verbs from the paths in the list. (A
		// single n-sided polygon is fanned by n-2 triangles; multiple polygons with a combined edge count of n are
		// fanned by strictly fewer triangles.)
		maxTrianglesInFans := max(o.totalCombinedPathVerbCnt-2, 0)
		if maxTrianglesInFans > 0 {
			alloc := newEagerDynamicVertexAllocator(state)
			if buf := alloc.lockWriter(8, maxTrianglesInFans*3); buf != nil {
				fanTriangleCount := 0
				off := 0
				for draw := o.pathDrawList; draw != nil; draw = draw.next {
					m := makeTessAffineMatrix(&draw.pathMatrix)
					pathMiddleOutFanTriangles(draw.path, func(p0, p1, p2 geom.Point) {
						for _, pt := range [3]geom.Point{p0, p1, p2} {
							mapped := m.mapPoint(pt)
							binary.LittleEndian.PutUint32(buf[off:],
								math.Float32bits(mapped.X))
							binary.LittleEndian.PutUint32(buf[off+4:],
								math.Float32bits(mapped.Y))
							off += 8
						}
						fanTriangleCount++
					})
				}
				if fanTriangleCount > maxTrianglesInFans {
					panic("fan triangle count exceeds the worst-case bound")
				}
				o.fanVertexCount = fanTriangleCount * 3
				alloc.unlock(o.fanVertexCount)
				o.fanBuffer = alloc.buffer
				o.fanBaseVertex = alloc.firstVertex
			}
		}
	}

	tessShader := o.stencilPathProgram.GeomProc().(*middleOutShader)
	o.tessellator.prepare(state, &tessShader.viewMatrix, o.pathDrawList,
		o.totalCombinedPathVerbCnt)

	if o.coverBBoxProgram != nil {
		instanceStride := o.coverBBoxProgram.GeomProc().gpBase().InstanceStride()
		vertexData, buffer, baseInstance := state.MakeVertexSpace(uint64(instanceStride),
			o.pathCount)
		if vertexData == nil {
			o.coverBBoxProgram = nil
		} else {
			o.bboxBuffer = buffer
			o.bboxBaseInstance = baseInstance
			off := 0
			putF32 := func(v float32) {
				binary.LittleEndian.PutUint32(vertexData[off:], math.Float32bits(v))
				off += 4
			}
			pathCount := 0
			for draw := o.pathDrawList; draw != nil; draw = draw.next {
				putF32(draw.pathMatrix.Get(geom.MScaleX))
				putF32(draw.pathMatrix.Get(geom.MSkewY))
				putF32(draw.pathMatrix.Get(geom.MSkewX))
				putF32(draw.pathMatrix.Get(geom.MScaleY))
				putF32(draw.pathMatrix.Get(geom.MTransX))
				putF32(draw.pathMatrix.Get(geom.MTransY))
				var bounds geom.Rect
				if draw.path.IsInverseFillType() {
					// Fill the entire backing store to make sure we clear every stencil value back to 0. If there is a
					// scissor it will have already clipped the stencil draw.
					rtBounds := state.OpArgs().RTProxy().Proxy().BackingStoreBoundsIRect().ToRect()
					// Map the render target bounds through the inverse matrix.
					if inv, ok := draw.pathMatrix.Invert(); ok {
						bounds, _ = inv.MapRect(rtBounds)
					} else {
						bounds = draw.path.Bounds()
					}
				} else {
					bounds = draw.path.Bounds()
				}
				putF32(bounds.Left)
				putF32(bounds.Top)
				putF32(bounds.Right)
				putF32(bounds.Bottom)
				pathCount++
			}
			if pathCount != o.pathCount || off != instanceStride*o.pathCount {
				panic("bbox instance data out of sync")
			}
		}
	}
}

// OnExecute implements Op: issues the inner-fan stencil draw (if any), the tessellated stencil draw, and the
// bounding-box cover draw (if not in stencil-only mode).
func (o *pathStencilCoverOp) OnExecute(state *OpFlushState, _chainBounds geom.Rect) {
	if o.tessellator == nil {
		return
	}
	renderPass := state.OpsRenderPass()
	scissor := func(info *ProgramInfo) {
		if info.Pipeline().IsScissorTestEnabled() {
			renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
		}
	}

	// Stencil the inner fan, if any.
	if o.fanVertexCount > 0 {
		if o.stencilFanProgram == nil || o.fanBuffer == nil {
			panic("fan vertices without a fan program or buffer")
		}
		if renderPass.BindPipeline(o.stencilFanProgram, o.Bounds()) {
			scissor(o.stencilFanProgram)
			renderPass.BindBuffers(nil, nil, o.fanBuffer)
			renderPass.Draw(o.fanVertexCount, o.fanBaseVertex)
		}
	}

	// Stencil the rest of the path.
	if o.stencilPathProgram == nil {
		panic("missing stencil path program")
	}
	if renderPass.BindPipeline(o.stencilPathProgram, o.Bounds()) {
		scissor(o.stencilPathProgram)
		o.tessellator.draw(state)
	}

	// Fill in the bounding box (if not in stencil-only mode).
	if o.coverBBoxProgram != nil {
		if renderPass.BindPipeline(o.coverBBoxProgram, o.Bounds()) {
			scissor(o.coverBBoxProgram)
			renderPass.BindTextures(o.coverBBoxProgram.GeomProc(), nil,
				o.coverBBoxProgram.Pipeline())
			renderPass.BindBuffers(nil, o.bboxBuffer, nil)
			renderPass.DrawInstanced(o.pathCount, o.bboxBaseInstance, 4, 0)
		}
	}
}
