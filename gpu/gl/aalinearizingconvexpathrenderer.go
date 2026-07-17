// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Coverage-AA convex fills and miter/bevel strokes rendered by flattening the path with aaConvexTessellator into
// per-vertex-coverage triangles drawn through the default geometry processor. Vertex/index storage grows via slice
// appends; a draw splits whenever it would overflow the 2^16 index range.

package gl

import (
	"encoding/binary"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// aaLinearizingMaxStrokeWidth is the stroke-width limit above which this renderer bails: the thicker the stroke, the
// harder it is to produce high-quality results using tessellation. For the time being, we simply drop back to software
// rendering above this stroke width.
const aaLinearizingMaxStrokeWidth = float32(20.0)

// aaFlatteningExtractVerts writes the tessellator's result vertices and indices (position, color, optional local
// coords, coverage per vertex).
func aaFlatteningExtractVerts(tess *aaConvexTessellator, localCoordsMatrix *geom.Matrix, verts *quadVertexWriter, color colorcore.PMColor4f, wideColor bool, firstIndex uint16, idxs []uint16) {
	for i := 0; i < tess.numPts(); i++ {
		pt := tess.point(i)
		verts.putF32(pt.X)
		verts.putF32(pt.Y)
		verts.putColor(color, wideColor)
		if localCoordsMatrix != nil {
			lc := localCoordsMatrix.MapPoint(pt)
			verts.putF32(lc.X)
			verts.putF32(lc.Y)
		}
		verts.putF32(tess.coverage(i))
	}

	for i := 0; i < tess.numIndices(); i++ {
		idxs[i] = uint16(tess.index(i)) + firstIndex
	}
}

// createLinesOnlyGP builds the default geometry processor configured for per-vertex coverage.
func createLinesOnlyGP(tweakAlphaForCoverage, usesLocalCoords, wideColor bool) GeometryProcessor {
	coverageType := CoverageTypeAttribute
	if tweakAlphaForCoverage {
		coverageType = CoverageTypeAttributeTweakAlpha
	}
	localCoordsType := LocalCoordsTypeUnused
	if usesLocalCoords {
		localCoordsType = LocalCoordsTypeHasExplicit
	}
	colorType := ColorTypePremulAttribute
	if wideColor {
		colorType = ColorTypePremulWideAttribute
	}
	identity := geom.IdentityMatrix()
	return MakeDefaultGeoProc(colorType, colorcore.PMColor4f{}, coverageType, 0xff,
		localCoordsType, nil, &identity)
}

//////////////////////////////////////////////////////////////////////////////
// AAFlatteningConvexPathOp

var aaFlatteningConvexPathOpClassID = GenOpClassID()

// aaFlatteningConvexPathData holds one path's per-draw state within a batched aaFlatteningConvexPathOp.
type aaFlatteningConvexPathData struct {
	path        *path.Path
	viewMatrix  geom.Matrix
	color       colorcore.PMColor4f
	strokeWidth float32
	miterLimit  float32
	style       stroke.Style
	join        stroke.Join
}

// aaFlatteningConvexPathOp draws one or more coverage-AA convex fills or miter/bevel strokes, batched together when
// possible.
type aaFlatteningConvexPathOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	paths       []aaFlatteningConvexPathData
	meshes      []*simpleMesh
	OpBase
	wideColor bool
}

// newAAFlatteningConvexPathOp creates an op to fill or stroke p with paint under viewMatrix. The paint is consumed.
func newAAFlatteningConvexPathOp(paint *Paint, viewMatrix *geom.Matrix, p *path.Path, strokeWidth float32, style stroke.Style, join stroke.Join, miterLimit float32, stencilSettings *UserStencilSettings) DrawOp {
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := &aaFlatteningConvexPathOp{}
	o.InitOp(aaFlatteningConvexPathOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, stencilSettings,
		PipelineInputFlagNone)
	o.paths = append(o.paths, aaFlatteningConvexPathData{
		viewMatrix:  *viewMatrix,
		path:        p,
		color:       color,
		strokeWidth: strokeWidth,
		miterLimit:  miterLimit,
		style:       style,
		join:        join,
	})

	// Compute bounds.
	bounds := p.Bounds()
	w := strokeWidth
	if w > 0 {
		w /= 2
		maxScale := viewMatrix.MaxScale()
		// We should not have a perspective matrix, thus we should have a valid scale.
		if maxScale == -1 {
			panic("stroke bounds require a non-perspective matrix")
		}
		if join == stroke.JoinMiter && w*maxScale > 1 {
			w *= miterLimit
		}
		bounds = bounds.Outset(w, w)
	}
	devBounds, _ := viewMatrix.MapRect(bounds)
	o.SetBounds(devBounds, HasAABloat(true), IsHairline(false))
	return o
}

// Name implements Op.
func (o *aaFlatteningConvexPathOp) Name() string { return "AAFlatteningConvexPathOp" }

// VisitProxies implements Op.
func (o *aaFlatteningConvexPathOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *aaFlatteningConvexPathOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *aaFlatteningConvexPathOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *aaFlatteningConvexPathOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.paths[len(o.paths)-1].color, &o.wideColor)
}

// createProgramInfo builds the program info for this op's draw, if not already built.
func (o *aaFlatteningConvexPathOp) createProgramInfo(state *OpFlushState) {
	gp := createLinesOnlyGP(o.helper.CompatibleWithCoverageAsAlpha(),
		o.helper.UsesLocalCoords(), o.wideColor)
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// recordDraw copies the accumulated vertex/index data into GPU-visible buffers and appends the resulting mesh.
func (o *aaFlatteningConvexPathOp) recordDraw(state *OpFlushState, vertexCount int, vertexStride uint64, vertices []byte, indexCount int, indices []uint16) {
	if vertexCount == 0 || indexCount == 0 {
		return
	}
	verts, vertexBuffer, firstVertex := state.MakeVertexSpace(vertexStride, vertexCount)
	if verts == nil {
		return
	}
	copy(verts, vertices[:uint64(vertexCount)*vertexStride])

	idxData, indexBuffer, firstIndex := state.MakeIndexSpace(indexCount)
	if idxData == nil {
		return
	}
	for i := 0; i < indexCount; i++ {
		binary.LittleEndian.PutUint16(idxData[2*i:], indices[i])
	}
	mesh := &simpleMesh{}
	mesh.setIndexed(indexBuffer, indexCount, firstIndex, 0, uint16(vertexCount-1),
		vertexBuffer, firstVertex)
	o.meshes = append(o.meshes, mesh)
}

// OnPrepare implements Op.
func (o *aaFlatteningConvexPathOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	vertexStride := uint64(o.programInfo.GeomProc().gpBase().VertexStride())

	vertexCount := 0
	indexCount := 0
	var vertices []byte
	var indices []uint16
	for i := range o.paths {
		args := &o.paths[i]
		tess := newAAConvexTessellator(args.style, args.strokeWidth, args.join,
			args.miterLimit)

		if !tess.tessellate(&args.viewMatrix, args.path) {
			continue
		}

		currentVertices := tess.numPts()
		if vertexCount+currentVertices > int(^uint16(0)) {
			// If we added the current instance, we would overflow the indices we can store in a uint16. Draw what we've
			// got so far and reset.
			o.recordDraw(state, vertexCount, vertexStride, vertices, indexCount, indices)
			vertexCount = 0
			indexCount = 0
			vertices = vertices[:0]
			indices = indices[:0]
		}
		currentIndices := tess.numIndices()

		need := (uint64(vertexCount) + uint64(currentVertices)) * vertexStride
		if uint64(len(vertices)) < need {
			vertices = append(vertices, make([]byte, need-uint64(len(vertices)))...)
		}
		if len(indices) < indexCount+currentIndices {
			indices = append(indices, make([]uint16, indexCount+currentIndices-len(indices))...)
		}

		var localCoordsMatrix *geom.Matrix
		if o.helper.UsesLocalCoords() {
			var ivm geom.Matrix
			if inv, ok := args.viewMatrix.Invert(); ok {
				ivm = inv
			}
			localCoordsMatrix = &ivm
		}

		writer := &quadVertexWriter{buf: vertices, offset: vertexCount * int(vertexStride)}
		aaFlatteningExtractVerts(tess, localCoordsMatrix, writer,
			args.color, o.wideColor, uint16(vertexCount), indices[indexCount:])
		vertexCount += currentVertices
		indexCount += currentIndices
	}
	o.recordDraw(state, vertexCount, vertexStride, vertices, indexCount, indices)
}

// OnExecute implements Op.
func (o *aaFlatteningConvexPathOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.programInfo == nil || len(o.meshes) == 0 {
		return
	}
	renderPass := state.OpsRenderPass()
	if !renderPass.BindPipeline(o.programInfo, chainBounds) {
		return
	}
	if o.programInfo.Pipeline().IsScissorTestEnabled() {
		renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
	}
	renderPass.BindTextures(o.programInfo.GeomProc(), nil, o.programInfo.Pipeline())
	for _, mesh := range o.meshes {
		mesh.draw(renderPass)
	}
}

// OnCombineIfPossible implements Op.
func (o *aaFlatteningConvexPathOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*aaFlatteningConvexPathOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}

	o.paths = append(o.paths, that.paths...)
	o.wideColor = o.wideColor || that.wideColor
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// AALinearizingConvexPathRenderer

// AALinearizingConvexPathRenderer is a PathRenderer for coverage-AA convex fills and miter/bevel strokes, rendered via
// flattening/tessellation.
type AALinearizingConvexPathRenderer struct{}

// NewAALinearizingConvexPathRenderer returns the renderer (it is stateless).
func NewAALinearizingConvexPathRenderer() *AALinearizingConvexPathRenderer {
	return &AALinearizingConvexPathRenderer{}
}

// Name implements PathRenderer.
func (r *AALinearizingConvexPathRenderer) Name() string { return "AALinear" }

// OnGetStencilSupport implements PathRenderer (the base-class kNoSupport default).
func (r *AALinearizingConvexPathRenderer) OnGetStencilSupport(*StyledShape) StencilSupport {
	return StencilSupportNone
}

// OnCanDrawPath implements PathRenderer.
func (r *AALinearizingConvexPathRenderer) OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath {
	if args.AAType != gpu.AATypeCoverage {
		return CanDrawPathNo
	}
	if !args.Shape.KnownToBeConvex() {
		return CanDrawPathNo
	}
	if args.Shape.Style().HasPathEffect() {
		return CanDrawPathNo
	}
	if args.Shape.InverseFilled() {
		return CanDrawPathNo
	}
	if args.Shape.Bounds().Width() <= 0 && args.Shape.Bounds().Height() <= 0 {
		// Stroked zero length lines should draw, but this PR doesn't handle that case.
		return CanDrawPathNo
	}
	strokeRec := args.Shape.Style().Rec()

	if strokeRec.Style() == stroke.StyleStroke || strokeRec.Style() == stroke.StyleStrokeAndFill {
		if !args.ViewMatrix.IsSimilarity() {
			return CanDrawPathNo
		}
		strokeWidth := args.ViewMatrix.MaxScale() * strokeRec.Width()
		if strokeWidth < 1 && strokeRec.Style() == stroke.StyleStroke {
			return CanDrawPathNo
		}
		if (strokeWidth > aaLinearizingMaxStrokeWidth && !args.Shape.IsRect()) ||
			!args.Shape.KnownToBeClosed() ||
			strokeRec.Join() == stroke.JoinRound {
			return CanDrawPathNo
		}
		return CanDrawPathYes
	}
	if strokeRec.Style() != stroke.StyleFill {
		return CanDrawPathNo
	}
	// This can almost handle perspective. It would need to use 3 component explicit local coords when there are FPs
	// that require them. This is difficult to test because AAConvexPathRenderer takes almost all filled paths that
	// could get here. So just avoid perspective fills.
	if args.ViewMatrix.HasPerspective() {
		return CanDrawPathNo
	}
	return CanDrawPathYes
}

// OnDrawPath implements PathRenderer.
func (r *AALinearizingConvexPathRenderer) OnDrawPath(args *DrawPathArgs) bool {
	if args.SurfaceDrawContext.NumSamples() > 1 {
		panic("AALinearizingConvexPathRenderer requires a single-sample target")
	}
	if args.Shape.IsEmpty() {
		panic("AALinearizingConvexPathRenderer requires a non-empty shape")
	}
	if args.Shape.Style().HasPathEffect() {
		panic("AALinearizingConvexPathRenderer cannot handle path effects")
	}

	p := args.Shape.AsPath()
	fill := args.Shape.Style().IsSimpleFill()
	strokeRec := args.Shape.Style().Rec()
	strokeWidth := float32(-1)
	join := stroke.JoinMiter
	if !fill {
		strokeWidth = strokeRec.Width()
		join = strokeRec.Join()
	}
	miterLimit := strokeRec.Miter()

	op := newAAFlatteningConvexPathOp(args.Paint, args.ViewMatrix, p, strokeWidth,
		strokeRec.Style(), join, miterLimit, args.UserStencilSettings)
	args.SurfaceDrawContext.AddDrawOp(args.Clip, op)
	return true
}

// OnStencilPath implements PathRenderer (never reached: kNoSupport).
func (r *AALinearizingConvexPathRenderer) OnStencilPath(*StencilPathArgs) {
	panic("AALinearizingConvexPathRenderer cannot stencil paths")
}
