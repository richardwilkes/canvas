// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// strokeTessellateOp renders strokes by linearizing them into sorted "parametric" and "radial" edges (see
// strokeTessellationShader).

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

var strokeTessellateOpClassID = GenOpClassID()

// strokeMarkStencil marks every stencil value as "1".
var strokeMarkStencil = MakeUserStencilSettings(
	0x0001,
	UserStencilTestLessIfInClip, // Match strokeTestAndResetStencil.
	0x0000,                      // Always fail.
	UserStencilOpZero,
	UserStencilOpReplace,
	0xffff,
)

// strokeTestAndResetStencil passes if the stencil value is nonzero, and resets the stencil value to zero on pass. This
// is formulated to match strokeMarkStencil everywhere except the ref and compare mask, which would allow the same
// pipeline for both stencil and fill if dynamic stencil state were supported.
var strokeTestAndResetStencil = MakeUserStencilSettings(
	0x0000,
	UserStencilTestLessIfInClip, // i.e., "not equal to zero, if in clip".
	0x0001,
	UserStencilOpZero,
	UserStencilOpReplace,
	0xffff,
)

// strokeTessellateOp is the draw op that tessellates and renders one or more merged strokes.
type strokeTessellateOp struct {
	drawOpNoClipToShape
	pathStrokeList     pathStrokeList
	tessellator        *strokeTessellator
	fillProgram        *ProgramInfo
	stencilProgram     *ProgramInfo // Only used if the stroke has transparency.
	pathStrokeTail     **pathStrokeList
	processors         *ProcessorSet
	tessellationShader *strokeTessellationShader
	OpBase
	totalCombinedVerbCnt int
	viewMatrix           geom.Matrix
	patchAttribs         PatchAttribs
	aaType               gpu.AAType
	needsStencil         bool
}

// newStrokeTessellateOp creates a stroke tessellate op for path stroked with rec. The paint is consumed.
func newStrokeTessellateOp(aaType gpu.AAType, viewMatrix *geom.Matrix, p *path.Path, rec *stroke.Rec, paint *Paint) *strokeTessellateOp {
	o := &strokeTessellateOp{
		aaType:     aaType,
		viewMatrix: *viewMatrix,
		pathStrokeList: pathStrokeList{
			path:   p,
			stroke: *rec,
			color:  paint.Color4f(),
		},
		totalCombinedVerbCnt: p.CountVerbs(),
	}
	o.InitOp(strokeTessellateOpClassID, o)
	o.pathStrokeTail = &o.pathStrokeList.next
	o.processors = NewProcessorSetFromPaint(paint)
	if !pmFitsInBytes(o.headColor()) {
		o.patchAttribs |= PatchAttribWideColorIfEnabled
	}
	devBounds := p.Bounds()
	if !o.headStroke().IsHairlineStyle() {
		// Non-hairlines inflate in local path space (pre-transform).
		r := rec.InflationRadius()
		devBounds = devBounds.Outset(r, r)
	}
	devBounds, _ = viewMatrix.MapRect(devBounds)
	if o.headStroke().IsHairlineStyle() {
		// Hairlines inflate in device space (post-transform).
		r := stroke.GetInflationRadius(rec.Join(), rec.Miter(), rec.Cap(), 1)
		devBounds = devBounds.Outset(r, r)
	}
	o.SetBounds(devBounds, HasAABloat(false), IsHairline(false))
	return o
}

func (o *strokeTessellateOp) headStroke() *stroke.Rec        { return &o.pathStrokeList.stroke }
func (o *strokeTessellateOp) headColor() colorcore.PMColor4f { return o.pathStrokeList.color }

// Name implements Op.
func (o *strokeTessellateOp) Name() string { return "StrokeTessellateOp" }

// VisitProxies implements Op.
func (o *strokeTessellateOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	switch {
	case o.fillProgram != nil:
		o.fillProgram.VisitFPProxies(fn)
	case o.stencilProgram != nil:
		o.stencilProgram.VisitFPProxies(fn)
	default:
		o.processors.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *strokeTessellateOp) UsesMSAA() bool { return o.aaType == gpu.AATypeMSAA }

// UsesStencil implements DrawOp. This must be called after Finalize (needsStencil can change in Finalize).
func (o *strokeTessellateOp) UsesStencil() bool {
	if !o.processors.IsFinalized() {
		panic("UsesStencil called before Finalize")
	}
	return o.needsStencil
}

// Finalize implements DrawOp. Make sure the finalize happens before combining: we might change needsStencil here.
func (o *strokeTessellateOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	if o.pathStrokeList.next != nil {
		panic("Finalize must run before combining")
	}
	if !caps.ShaderCaps.InfinitySupport {
		// The GPU can't infer curve type based on infinity, so we need to send in an attrib explicitly stating the
		// curve type.
		o.patchAttribs |= PatchAttribExplicitCurveType
	}
	analysis, overrideColor := o.processors.Finalize(AnalysisColorConstant(o.headColor()),
		AnalysisCoverageNone, clip, UnusedStencilSettings(), caps, clampType)
	if analysis.InputColorIsOverridden() {
		o.pathStrokeList.color = overrideColor
	}
	o.needsStencil = !analysis.UnaffectedByDstValue()
	return analysis
}

// OnCombineIfPossible implements Op.
func (o *strokeTessellateOp) OnCombineIfPossible(t Op) CombineResult {
	op, ok := t.(*strokeTessellateOp)
	if !ok {
		return CombineResultCannotCombine
	}

	// This must be called after Finalize(). needsStencil can change in Finalize().
	if !o.processors.IsFinalized() || !op.processors.IsFinalized() {
		panic("OnCombineIfPossible called before Finalize")
	}

	if o.needsStencil ||
		op.needsStencil ||
		!o.viewMatrix.CheapEqual(&op.viewMatrix) ||
		o.aaType != op.aaType ||
		!o.processors.Equal(op.processors) ||
		o.headStroke().IsHairlineStyle() != op.headStroke().IsHairlineStyle() {
		return CombineResultCannotCombine
	}

	combinedAttribs := o.patchAttribs | op.patchAttribs
	if combinedAttribs&PatchAttribStrokeParams == 0 &&
		!strokesHaveEqualParams(o.headStroke(), op.headStroke()) {
		// The paths have different stroke properties. We will need to enable dynamic stroke if we still decide to
		// combine them.
		if o.headStroke().IsHairlineStyle() {
			return CombineResultCannotCombine // Dynamic hairlines aren't supported.
		}
		combinedAttribs |= PatchAttribStrokeParams
	}
	if combinedAttribs&PatchAttribColor == 0 && o.headColor() != op.headColor() {
		// The paths have different colors. We will need to enable dynamic color if we still decide to combine them.
		combinedAttribs |= PatchAttribColor
	}

	// Don't actually enable new dynamic state on ops that already have lots of verbs.
	const dynamicStatesMask = PatchAttribStrokeParams | PatchAttribColor
	neededDynamicStates := combinedAttribs & dynamicStatesMask
	if neededDynamicStates != PatchAttribsNone {
		if !o.shouldUseDynamicStates(neededDynamicStates) ||
			!op.shouldUseDynamicStates(neededDynamicStates) {
			return CombineResultCannotCombine
		}
	}

	o.patchAttribs = combinedAttribs

	// Concat the op's pathStrokeList. Since the head element is allocated inside the op, we need to copy it.
	headCopy := op.pathStrokeList
	*o.pathStrokeTail = &headCopy
	if op.pathStrokeTail == &op.pathStrokeList.next {
		o.pathStrokeTail = &headCopy.next
	} else {
		o.pathStrokeTail = op.pathStrokeTail
	}

	o.totalCombinedVerbCnt += op.totalCombinedVerbCnt
	return CombineResultMerged
}

// shouldUseDynamicStates reports whether it is a good tradeoff to use the dynamic states flagged in the given bitfield.
// Dynamic states improve batching, but if they aren't already enabled, they come at the cost of having to write out
// more data with each patch or instance.
func (o *strokeTessellateOp) shouldUseDynamicStates(neededDynamicStates PatchAttribs) bool {
	// Use the dynamic states if either (1) they are all already enabled anyway, or (2) we don't have many verbs.
	const maxVerbsToEnableDynamicState = 50
	anyStateDisabled := ^o.patchAttribs&neededDynamicStates != 0
	allStatesEnabled := !anyStateDisabled
	return allStatesEnabled || o.totalCombinedVerbCnt <= maxVerbsToEnableDynamicState
}

// prepareTessellator creates the tessellator and the stencil/fill program(s) we will use with it. The applied clip is
// consumed.
func (o *strokeTessellateOp) prepareTessellator(args *tessProgramArgs, appliedClip *AppliedClip) {
	if o.tessellator != nil || o.fillProgram != nil || o.stencilProgram != nil {
		panic("prepareTessellator called twice")
	}

	pipeline := tessMakePipeline(args, o.processors, appliedClip)

	o.tessellator = newStrokeTessellator(o.patchAttribs)
	o.tessellationShader = newStrokeTessellationShader(args.caps.ShaderCaps, o.patchAttribs,
		&o.viewMatrix, o.headStroke(), o.headColor())

	fillStencil := UnusedStencilSettings()
	if o.needsStencil {
		o.stencilProgram = tessMakeProgram(args, o.tessellationShader,
			o.tessellationShader.primitiveType, pipeline, &strokeMarkStencil)
		fillStencil = &strokeTestAndResetStencil
		// TODO: currently if we have a texture barrier for a dst read it will get put in before both the stencil draw
		// and the fill draw. In reality we only really need the barrier once, to guard the reads of the color buffer in
		// the fill from the previous writes.
	}

	o.fillProgram = tessMakeProgram(args, o.tessellationShader,
		o.tessellationShader.primitiveType, pipeline, fillStencil)
}

// OnPrepare implements Op.
func (o *strokeTessellateOp) OnPrepare(state *OpFlushState) {
	if o.tessellator == nil {
		args := makeTessProgramArgs(state)
		o.prepareTessellator(&args, state.OpArgs().AppliedClip())
	}
	o.tessellator.prepare(state, &o.viewMatrix, &o.pathStrokeList, o.totalCombinedVerbCnt)
}

// OnExecute implements Op.
func (o *strokeTessellateOp) OnExecute(state *OpFlushState, _chainBounds geom.Rect) {
	renderPass := state.OpsRenderPass()
	bindAndDraw := func(program *ProgramInfo) {
		if !renderPass.BindPipeline(program, o.Bounds()) {
			return
		}
		if program.Pipeline().IsScissorTestEnabled() {
			renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
		}
		renderPass.BindTextures(program.GeomProc(), nil, program.Pipeline())
		o.tessellator.draw(state)
	}
	if o.stencilProgram != nil {
		bindAndDraw(o.stencilProgram)
	}
	if o.fillProgram != nil {
		bindAndDraw(o.fillProgram)
	}
}
