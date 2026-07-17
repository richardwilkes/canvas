// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// pathTessellateOp tessellates a path directly to the color buffer, using a single render pass. This currently only
// works for convex paths.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
)

var pathTessellateOpClassID = GenOpClassID()

// pathTessellateOp tessellates a convex path directly to the color buffer in a single render pass.
type pathTessellateOp struct {
	drawOpNoClipToShape
	stencil      *UserStencilSettings
	pathDrawList *pathDrawList
	pathDrawTail **pathDrawList
	processors   *ProcessorSet
	// Decided during prepareTessellator.
	tessellator         *pathWedgeTessellator
	tessellationProgram *ProgramInfo
	OpBase
	totalCombinedPathVerbCnt int
	shaderMatrix             geom.Matrix
	aaType                   gpu.AAType
	patchAttribs             PatchAttribs
}

// newPathTessellateOp creates a pathTessellateOp. The paint is consumed; the path must not be inverse filled.
func newPathTessellateOp(aaType gpu.AAType, stencil *UserStencilSettings, viewMatrix *geom.Matrix, p *path.Path, paint *Paint, drawBounds geom.Rect) *pathTessellateOp {
	if p.IsInverseFillType() {
		panic("PathTessellateOp cannot draw inverse fills")
	}
	o := &pathTessellateOp{
		aaType:                   aaType,
		stencil:                  stencil,
		totalCombinedPathVerbCnt: p.CountVerbs(),
		shaderMatrix:             *viewMatrix,
	}
	o.InitOp(pathTessellateOpClassID, o)
	o.pathDrawList = &pathDrawList{
		pathMatrix: geom.IdentityMatrix(), path: p,
		color: paint.Color4f(),
	}
	o.pathDrawTail = &o.pathDrawList.next
	o.processors = NewProcessorSetFromPaint(paint)
	if !pmFitsInBytes(o.headDraw().color) {
		o.patchAttribs |= PatchAttribWideColorIfEnabled
	}
	o.SetBounds(drawBounds, HasAABloat(false), IsHairline(false))
	return o
}

func (o *pathTessellateOp) headDraw() *pathDrawList { return o.pathDrawList }

// Name implements Op.
func (o *pathTessellateOp) Name() string { return "PathTessellateOp" }

// VisitProxies implements Op.
func (o *pathTessellateOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.tessellationProgram != nil {
		o.tessellationProgram.VisitFPProxies(fn)
	} else {
		o.processors.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *pathTessellateOp) UsesMSAA() bool { return o.aaType == gpu.AATypeMSAA }

// UsesStencil implements DrawOp.
func (o *pathTessellateOp) UsesStencil() bool { return !o.stencil.IsUnused() }

// Finalize implements DrawOp: runs the processor set's finalization pass to compute the op's coverage/color analysis,
// moving the transform to the CPU when local coords aren't needed.
func (o *pathTessellateOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	color := AnalysisColorConstant(o.headDraw().color)
	analysis, overrideColor := o.processors.Finalize(color, AnalysisCoverageNone, clip, nil,
		caps, clampType)
	if analysis.InputColorIsOverridden() {
		o.headDraw().color = overrideColor
	}
	if !analysis.UsesLocalCoords() {
		// Since we don't need local coords, we can transform on CPU instead of in the shader. This gives us better
		// batching potential.
		o.headDraw().pathMatrix = o.shaderMatrix
		o.shaderMatrix = geom.IdentityMatrix()
	}
	return analysis
}

// OnCombineIfPossible implements Op: merges op into this one when they share aaType, stencil settings, processors, and
// shader matrix, concatenating their path draw lists.
func (o *pathTessellateOp) OnCombineIfPossible(t Op) CombineResult {
	op, ok := t.(*pathTessellateOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if o.aaType != op.aaType || o.stencil != op.stencil ||
		!o.processors.Equal(op.processors) || !o.shaderMatrix.CheapEqual(&op.shaderMatrix) {
		return CombineResultCannotCombine
	}

	o.totalCombinedPathVerbCnt += op.totalCombinedPathVerbCnt
	o.patchAttribs |= op.patchAttribs

	if o.patchAttribs&PatchAttribColor == 0 && o.headDraw().color != op.headDraw().color {
		// Color is no longer uniform. Move it into patch attribs.
		o.patchAttribs |= PatchAttribColor
	}

	*o.pathDrawTail = op.pathDrawList
	o.pathDrawTail = op.pathDrawTail
	return CombineResultMerged
}

// prepareTessellator builds the tessellator and its program. The applied clip is consumed.
func (o *pathTessellateOp) prepareTessellator(args *tessProgramArgs, appliedClip *AppliedClip) {
	if o.tessellator != nil || o.tessellationProgram != nil {
		panic("prepareTessellator called twice")
	}
	pipeline := tessMakePipeline(args, o.processors, appliedClip)
	o.tessellator = newPathWedgeTessellator(args.caps.ShaderCaps.InfinitySupport,
		o.patchAttribs)
	tessShader := makeMiddleOutShader(args.caps.ShaderCaps, &o.shaderMatrix,
		o.headDraw().color, o.tessellator.patchAttribs())
	o.tessellationProgram = tessMakeProgram(args, tessShader, tessShader.primitiveType,
		pipeline, o.stencil)
}

// OnPrepare implements Op: builds the tessellator (if not already built) and its vertex data.
func (o *pathTessellateOp) OnPrepare(state *OpFlushState) {
	if o.tessellator == nil {
		args := makeTessProgramArgs(state)
		o.prepareTessellator(&args, state.OpArgs().AppliedClip())
	}
	o.tessellator.prepare(state, &o.shaderMatrix, o.pathDrawList, o.totalCombinedPathVerbCnt)
}

// OnExecute implements Op: binds the tessellation program and issues the draw.
func (o *pathTessellateOp) OnExecute(state *OpFlushState, _chainBounds geom.Rect) {
	if o.tessellator == nil || o.tessellationProgram == nil {
		return
	}
	renderPass := state.OpsRenderPass()
	if !renderPass.BindPipeline(o.tessellationProgram, o.Bounds()) {
		return
	}
	if o.tessellationProgram.Pipeline().IsScissorTestEnabled() {
		renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
	}
	renderPass.BindTextures(o.tessellationProgram.GeomProc(), nil,
		o.tessellationProgram.Pipeline())
	o.tessellator.draw(state)
}
