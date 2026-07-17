// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The batched per-edge-AA quad fill op — unison's most common draw. Each op accumulates (device quad, local quad,
// color, edge flags) entries in a QuadBuffer; the OpsTask combine machinery merges compatible ops (including
// none↔coverage AA upgrades), and prepare tessellates every quad into one vertex buffer drawn with a single indexed
// draw. This replaces the earlier interim fill-rect op. The DDL prePrepare lane is dropped; the arena parameters wait
// on the pooling work.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

var fillRectOpClassID = GenOpClassID()

// fillRectOpPool is the free list of fillRectOp shells (see oppool.go); fillRectQuadBufferPool holds their QuadBuffers,
// recycled/borrowed alongside (B.3).
var (
	fillRectOpPool         opPool[fillRectOp]
	fillRectQuadBufferPool opPool[QuadBuffer[colorAndAA]]
)

// colorAndAA is the per-quad metadata for a fillRectOp.
type colorAndAA struct {
	color   colorcore.PMColor4f
	aaFlags gpu.QuadAAFlags
}

// fillRectOp draws one or more batched per-edge-AA quad fills.
type fillRectOp struct {
	// Prepared state.
	vertexBuffer AnyBuffer
	quads        *QuadBuffer[colorAndAA]
	programInfo  *ProgramInfo
	indexBuffer  *Buffer
	helper       SimpleMeshDrawOpHelper
	OpBase
	baseVertex int
	colorType  QuadColorType
}

// NewFillRectOp builds the op from a paint, AA type, and DrawQuad (which it may modify). The paint is consumed. stencil
// may be nil (unused); inputFlags are Pipeline input flags.
func NewFillRectOp(paint *Paint, aaType gpu.AAType, quad *DrawQuad, stencil *UserStencilSettings, inputFlags uint8) DrawOp {
	// Clean up deviations between aaType and edgeAA.
	aaType, quad.EdgeFlags = ResolveAAType(aaType, quad.EdgeFlags, &quad.Device)

	// Only non-trivial paints allocate a processor set.
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}
	return newFillRectOp(processors, color, aaType, quad, stencil, inputFlags)
}

// newFillRectOp creates a fillRectOp. Incongruities between aaType and edge flags must be resolved before calling.
func newFillRectOp(processors *ProcessorSet, paintColor colorcore.PMColor4f, aaType gpu.AAType, quad *DrawQuad, stencil *UserStencilSettings, inputFlags uint8) *fillRectOp {
	o := fillRectOpPool.borrow()
	o.InitOp(fillRectOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, aaType, stencil, inputFlags)
	o.quads = borrowQuadBuffer(&fillRectQuadBufferPool, 1, !o.helper.IsTrivial())

	// Set bounds before clipping so we don't have to worry about unioning the bounds of the two potential quads
	// (Quad.Bounds() is perspective-safe).
	hairline := WillUseHairline(&quad.Device, aaType, quad.EdgeFlags)
	o.SetBounds(quad.Device.Bounds(), HasAABloat(aaType == gpu.AATypeCoverage),
		IsHairline(hairline))

	var extra DrawQuad
	// Always crop to W>0 to remain consistent with Quad.Bounds().
	count := ClipToW0(quad, &extra)
	if count == 0 {
		// We can't discard the op at this point, but disable AA flags so it won't go through inset/outset processing.
		quad.EdgeFlags = gpu.QuadAAFlagsNone
		count = 1
	}

	// Conservatively keep track of the local coordinates; it may be that the paint doesn't need them after analysis is
	// finished. If the paint is known to be solid up front they can be skipped entirely.
	var local *Quad
	if !o.helper.IsTrivial() {
		local = &quad.Local
	}
	o.quads.Append(&quad.Device, colorAndAA{color: paintColor, aaFlags: quad.EdgeFlags}, local)
	if count > 1 {
		if !o.helper.IsTrivial() {
			local = &extra.Local
		}
		o.quads.Append(&extra.Device, colorAndAA{color: paintColor, aaFlags: extra.EdgeFlags},
			local)
	}
	return o
}

// Name implements Op.
func (o *fillRectOp) Name() string { return "FillRectOp" }

// recycle implements Op: returns this dead op — and its ProgramInfo/Pipeline pair — to their free lists.
// recycleProgramInfo runs first (nil-safe) so it reads o.programInfo before the pool zeroes the shell.
func (o *fillRectOp) recycle() {
	recycleProgramInfo(o.programInfo)
	if o.quads != nil {
		recycleQuadBuffer(&fillRectQuadBufferPool, o.quads)
	}
	fillRectOpPool.recycle(o)
}

// ClipToShape implements DrawOp (fillRectOp does not clip its own geometry).
func (o *fillRectOp) ClipToShape(raster.ClipOp, *geom.Matrix, *Shape, gpu.AA) ClipResult {
	return ClipResultFail
}

// NumQuads is test access to the batched quad count.
func (o *fillRectOp) NumQuads() int { return o.quads.Count() }

// VisitProxies implements Op.
func (o *fillRectOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *fillRectOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *fillRectOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *fillRectOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	// Initialize the aggregate color analysis with the first quad's color (which always exists).
	iter := o.quads.Metadata()
	if !iter.Next() {
		panic("fill-rect op must have at least one quad")
	}
	quadColors := AnalysisColorConstant(iter.Get().color)
	// Then combine the colors of any additional quads (e.g. from a quad set).
	for iter.Next() {
		quadColors = CombineAnalysisColors(quadColors, AnalysisColorConstant(iter.Get().color))
		if quadColors.IsUnknown() {
			// No point in accumulating additional starting colors; combining cannot make it less unknown.
			break
		}
	}

	// If the AA type is coverage, it will be a single value per pixel; if it's not coverage AA then the coverage is
	// always 1.0, so specify kNone for more optimal blending.
	coverage := AnalysisCoverageNone
	if o.helper.AAType() == gpu.AATypeCoverage {
		coverage = AnalysisCoverageSingleChannel
	}
	result := o.helper.FinalizeProcessors(caps, clip, clampType, coverage, &quadColors)

	// If there is a constant color after analysis, all quads should be set to the same color (even if they started out
	// with different colors).
	iter = o.quads.Metadata()
	if colorOverride, isConstant := quadColors.IsConstant(); isConstant {
		o.colorType = MinQuadColorType(colorOverride)
		for iter.Next() {
			iter.Get().color = colorOverride
		}
	} else {
		// Otherwise compute the color type needed as the max over all quads.
		o.colorType = QuadColorTypeNone
		for iter.Next() {
			o.colorType = max(o.colorType, MinQuadColorType(iter.Get().color))
		}
	}
	// Most shaders' FPs multiply their calculated color by the paint color or alpha. We want to use ColorType none to
	// optimize out that multiply. However, if there are no color FPs then we would really be writing a special shader
	// for white rectangles and not saving any multiplies, so use bytes (which also works around an ANGLE driver issue).
	if o.colorType == QuadColorTypeNone && !result.HasColorFragmentProcessor() {
		o.colorType = QuadColorTypeByte
	}
	return result
}

// vertexSpec computes the vertex layout for this op's current quads/AA/color state.
func (o *fillRectOp) vertexSpec() VertexSpec {
	indexBufferOption := CalcIndexBufferOption(o.helper.AAType(), o.quads.Count())
	return NewVertexSpec(o.quads.DeviceQuadType(), o.colorType, o.quads.LocalQuadType(),
		o.helper.UsesLocalCoords(), QuadSubsetNo, o.helper.AAType(),
		o.helper.CompatibleWithCoverageAsAlpha(), indexBufferOption)
}

// createProgramInfo builds the program info for this op's draw.
func (o *fillRectOp) createProgramInfo(state *OpFlushState) {
	spec := o.vertexSpec()
	gp := MakeQuadPerEdgeAAProcessor(&spec)
	if gp.gpBase().VertexStride() != spec.VertexSize() {
		panic("GP vertex stride out of sync with the vertex spec")
	}
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		spec.PrimitiveType(), args.RenderPassBarriers(), args.ColorLoadOp())
}

// tessellate writes every quad's vertex data into dst per spec.
func (o *fillRectOp) tessellate(spec *VertexSpec, dst []byte) {
	tess := NewQuadTessellator(*spec, dst)
	iter := o.quads.Iterator()
	for iter.Next() {
		// All entries should have local coords, or no entries should have local coords, matching !helper.isTrivial()
		// (which is more conservative than helper.usesLocalCoords).
		if iter.IsLocalValid() == o.helper.IsTrivial() {
			panic("local coords out of sync with the helper's triviality")
		}
		info := iter.Metadata()
		tess.Append(iter.DeviceQuad(), iter.LocalQuad(), info.color, geom.Rect{}, info.aaFlags)
	}
}

// OnPrepare implements Op.
func (o *fillRectOp) OnPrepare(state *OpFlushState) {
	spec := o.vertexSpec()

	// Make sure that if the op thought it was a solid color, the vertex spec does not use local coords.
	if o.helper.IsTrivial() && o.helper.UsesLocalCoords() {
		panic("trivial helper cannot use local coords")
	}

	totalNumVertices := o.quads.Count() * spec.VerticesPerQuad()

	// Fill the allocated vertex data.
	vdata, vertexBuffer, baseVertex := state.MakeVertexSpace(uint64(spec.VertexSize()),
		totalNumVertices)
	if vdata == nil {
		// Could not allocate vertices.
		return
	}
	o.vertexBuffer = vertexBuffer
	o.baseVertex = baseVertex

	o.tessellate(&spec, vdata)

	if spec.NeedsIndexBuffer() {
		o.indexBuffer = GetQuadIndexBuffer(state, spec.IndexBufferOption())
	}
}

// OnExecute implements Op.
func (o *fillRectOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.vertexBuffer == nil {
		return
	}
	spec := o.vertexSpec()
	if spec.NeedsIndexBuffer() && o.indexBuffer == nil {
		return
	}

	if o.programInfo == nil {
		o.createProgramInfo(state)
	}

	// Bind the pipeline and any scissor clip.
	renderPass := state.OpsRenderPass()
	if !renderPass.BindPipeline(o.programInfo, chainBounds) {
		return
	}
	if o.programInfo.Pipeline().IsScissorTestEnabled() {
		renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
	}
	renderPass.BindBuffers(o.indexBuffer, nil, o.vertexBuffer)
	renderPass.BindTextures(o.programInfo.GeomProc(), nil, o.programInfo.Pipeline())
	IssueQuadDraw(state.Caps(), renderPass, &spec, 0, o.quads.Count(), o.baseVertex)
}

// OnCombineIfPossible implements Op.
func (o *fillRectOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*fillRectOp)
	if !ok {
		return CombineResultCannotCombine
	}

	upgradeToCoverageAAOnMerge := false
	if o.helper.AAType() != that.helper.AAType() {
		if !CanUpgradeAAOnMerge(o.helper.AAType(), that.helper.AAType()) {
			return CombineResultCannotCombine
		}
		upgradeToCoverageAAOnMerge = true
	}

	if CombinedQuadCountWillOverflow(o.helper.AAType(), upgradeToCoverageAAOnMerge,
		o.quads.Count()+that.quads.Count()) {
		return CombineResultCannotCombine
	}

	// Unlike most users of the draw op helper, this op can merge none-aa and coverage-aa draw ops together, so pass
	// true as the last argument.
	if !o.helper.IsCompatible(&that.helper, true) {
		return CombineResultCannotCombine
	}

	// If the paints were compatible, the trivial/solid-color state should be the same.
	if o.helper.IsTrivial() != that.helper.IsTrivial() {
		panic("compatible helpers must agree on triviality")
	}

	// If the processor sets are compatible, the two ops are always compatible; it just needs to adjust the state of the
	// op to be the more general quad and aa types of the two ops and then concatenate the per-quad data.
	o.colorType = max(o.colorType, that.colorType)

	// The helper stores the aa type, but isCompatible(with true arg) allows the two ops' aa types to be none and
	// coverage, in which case this op's aa type must be lifted to coverage so that quads with no aa edges can be
	// batched with quads that have some/all edges aa'ed.
	if upgradeToCoverageAAOnMerge {
		o.helper.SetAAType(gpu.AATypeCoverage)
	}

	o.quads.Concat(that.quads)
	return CombineResultMerged
}

// canAddQuads reports whether numQuads more quads at aaType would fit within the index-buffer limit.
func (o *fillRectOp) canAddQuads(numQuads int, aaType gpu.AAType) bool {
	// The new quad's aa type should be the same as the first quad's or none, except when the first quad's aa type was
	// already downgraded to none, in which case the stored type must be lifted back up to the requested type.
	quadCount := o.quads.Count() + numQuads
	if aaType != o.helper.AAType() && aaType != gpu.AATypeNone {
		indexBufferOption := CalcIndexBufferOption(aaType, quadCount)
		if quadCount > QuadLimit(indexBufferOption) {
			// Promoting to the new aaType would've caused an overflow of the index buffer limit.
			return false
		}
		// The original quad was downgraded to non-aa; lift back up to this quad's required type.
		if o.helper.AAType() != gpu.AATypeNone {
			panic("expected a downgraded-to-none aa type")
		}
		o.helper.SetAAType(aaType)
	} else {
		indexBufferOption := CalcIndexBufferOption(o.helper.AAType(), quadCount)
		if quadCount > QuadLimit(indexBufferOption) {
			return false // This op can't grow any more.
		}
	}
	return true
}

// addQuad works similar to OnCombineIfPossible, but adds a quad assuming its op would have been compatible; since it
// avoids the op list management, it must update the op's bounds.
func (o *fillRectOp) addQuad(quad *DrawQuad, color colorcore.PMColor4f, aaType gpu.AAType) bool {
	newBounds := joinPossiblyEmptyRect(o.Bounds(), quad.Device.Bounds())

	var extra DrawQuad
	count := 1
	if quad.EdgeFlags != gpu.QuadAAFlagsNone {
		count = ClipToW0(quad, &extra)
	}
	switch {
	case count == 0:
		// Just skip the append (trivial success).
		return true
	case !o.canAddQuads(count, aaType):
		// Not enough room in the index buffer for the AA type.
		return false
	default:
		// Can actually add the 1 or 2 quads representing the draw.
		var local *Quad
		if !o.helper.IsTrivial() {
			local = &quad.Local
		}
		o.quads.Append(&quad.Device, colorAndAA{color: color, aaFlags: quad.EdgeFlags}, local)
		if count > 1 {
			if !o.helper.IsTrivial() {
				local = &extra.Local
			}
			o.quads.Append(&extra.Device, colorAndAA{color: color, aaFlags: extra.EdgeFlags},
				local)
		}
		// Update the bounds.
		o.SetBounds(newBounds, HasAABloat(o.helper.AAType() == gpu.AATypeCoverage),
			IsHairline(false))
		return true
	}
}

// MakeNonAARectFillOp builds a non-AA fillRectOp for a single rect.
func MakeNonAARectFillOp(paint *Paint, view *geom.Matrix, rect geom.Rect, stencil *UserStencilSettings) DrawOp {
	quad := DrawQuad{
		Device: MakeQuadFromRect(rect, view),
		Local:  MakeQuadFromRectNoTransform(rect), EdgeFlags: gpu.QuadAAFlagsNone,
	}
	return NewFillRectOp(paint, gpu.AATypeNone, &quad, stencil, PipelineInputFlagNone)
}

// QuadSetEntry is one rect in a batched quad-set fill.
type QuadSetEntry struct {
	Rect        geom.Rect
	Color       colorcore.PMColor4f // Overrides any color on the paint.
	LocalMatrix geom.Matrix
	AAFlags     gpu.QuadAAFlags
}

// makeFillRectOpSet creates an op for the first quad in the set and accumulates as many of the remaining quads as fit
// (similar to onCombineIfPossible without creating extra ops). Returns the op and the number of consumed entries.
func makeFillRectOpSet(paint *Paint, aaType gpu.AAType, viewMatrix *geom.Matrix, quads []QuadSetEntry, stencilSettings *UserStencilSettings) (drawOp DrawOp, consumed int) {
	if len(quads) == 0 {
		panic("quad set must not be empty")
	}
	quad := DrawQuad{
		Device:    MakeQuadFromRect(quads[0].Rect, viewMatrix),
		Local:     MakeQuadFromRect(quads[0].Rect, &quads[0].LocalMatrix),
		EdgeFlags: quads[0].AAFlags,
	}
	paint.SetColor4f(quads[0].Color)
	op := NewFillRectOp(paint, aaType, &quad, stencilSettings, PipelineInputFlagNone)
	fillRects := op.(*fillRectOp)

	numConsumed := 1
	for i := 1; i < len(quads); i++ {
		quad = DrawQuad{
			Device:    MakeQuadFromRect(quads[i].Rect, viewMatrix),
			Local:     MakeQuadFromRect(quads[i].Rect, &quads[i].LocalMatrix),
			EdgeFlags: quads[i].AAFlags,
		}

		resolvedAA, edgeFlags := ResolveAAType(aaType, quads[i].AAFlags, &quad.Device)
		quad.EdgeFlags = edgeFlags

		if !fillRects.addQuad(&quad, quads[i].Color, resolvedAA) {
			break
		}
		numConsumed++
	}
	return op, numConsumed
}

// AddFillRectOps chunks the quad set into as few ops as the index-buffer limits allow and records them on the draw
// context.
func AddFillRectOps(sdc *SurfaceDrawContext, clip Clip, paint *Paint, aaType gpu.AAType, viewMatrix *geom.Matrix, quads []QuadSetEntry, stencilSettings *UserStencilSettings) {
	offset := 0
	for offset < len(quads) {
		op, numConsumed := makeFillRectOpSet(ClonePaint(paint), aaType, viewMatrix,
			quads[offset:], stencilSettings)
		offset += numConsumed
		sdc.AddDrawOp(clip, op)
	}
}
