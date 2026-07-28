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

// aaFlatteningMaxVertices is the most vertices one draw can address: the indices are uint16 and recordDraw hands
// vertexCount-1 to the mesh as its maximum index value.
const aaFlatteningMaxVertices = int(^uint16(0))

// aaFlatteningPutVert writes one of the tessellator's vertices (position, color, optional local coords, coverage). It is
// the single definition of the vertex layout createLinesOnlyGP declares, shared by the batched and split lanes.
func aaFlatteningPutVert(tess *aaConvexTessellator, i int, localCoordsMatrix *geom.Matrix, verts *quadVertexWriter, color colorcore.PMColor4f, wideColor bool) {
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

// aaFlatteningExtractVerts writes the tessellator's result vertices and indices, biasing each index by the vertices
// already accumulated in the batch. The caller guarantees firstIndex+numPts stays inside the uint16 index range.
func aaFlatteningExtractVerts(tess *aaConvexTessellator, localCoordsMatrix *geom.Matrix, verts *quadVertexWriter, color colorcore.PMColor4f, wideColor bool, firstIndex uint16, idxs []uint16) {
	for i := 0; i < tess.numPts(); i++ {
		aaFlatteningPutVert(tess, i, localCoordsMatrix, verts, color, wideColor)
	}

	for i := 0; i < tess.numIndices(); i++ {
		idxs[i] = uint16(tess.index(i)) + firstIndex
	}
}

// aaFlatteningBatch is the vertex/index staging buffer OnPrepare accumulates draws into. The buffers (and the split
// lane's remap scratch) are kept across draws so a long batch or a split path reuses one allocation.
type aaFlatteningBatch struct {
	vertices     []byte
	indices      []uint16
	remap        []int32
	vertexCount  int
	indexCount   int
	vertexStride uint64
}

// reserve grows the buffers so extraVerts more vertices and extraIndices more indices fit past what is accumulated.
func (b *aaFlatteningBatch) reserve(extraVerts, extraIndices int) {
	need := (uint64(b.vertexCount) + uint64(extraVerts)) * b.vertexStride
	if uint64(len(b.vertices)) < need {
		b.vertices = append(b.vertices, make([]byte, need-uint64(len(b.vertices)))...)
	}
	if len(b.indices) < b.indexCount+extraIndices {
		b.indices = append(b.indices, make([]uint16, b.indexCount+extraIndices-len(b.indices))...)
	}
}

// writer returns a vertex writer positioned just past the accumulated vertices. Only valid until the next reserve.
func (b *aaFlatteningBatch) writer() *quadVertexWriter {
	return &quadVertexWriter{buf: b.vertices, offset: b.vertexCount * int(b.vertexStride)}
}

// aaFlatteningBuildSplitChunk stages the longest run of tess's triangles starting at firstTri that one draw can address,
// copying in only the vertices that run references and remapping their indices into the chunk's own range (a vertex
// shared by two chunks is written to both). The batch must be empty; returns the number of triangles staged. This is how
// a tessellation with more than aaFlatteningMaxVertices vertices is drawn: the batched lane's uint16 indices cannot
// reach past that, and emitting them anyway wraps every index and renders scrambled triangles.
func aaFlatteningBuildSplitChunk(b *aaFlatteningBatch, tess *aaConvexTessellator, firstTri int, localCoordsMatrix *geom.Matrix, color colorcore.PMColor4f, wideColor bool) int {
	// Each triangle adds at most three vertices, so a run this long always fits without checking mid-triangle.
	const maxTris = aaFlatteningMaxVertices / 3
	if b.vertexCount != 0 || b.indexCount != 0 {
		panic("split chunks must be staged into an empty batch")
	}
	tris := min(maxTris, tess.numIndices()/3-firstTri)
	if tris <= 0 {
		return 0
	}
	if cap(b.remap) < tess.numPts() {
		b.remap = make([]int32, tess.numPts())
	}
	remap := b.remap[:tess.numPts()]
	for i := range remap {
		remap[i] = -1
	}
	b.reserve(3*tris, 3*tris)
	verts := b.writer()
	for i := 3 * firstTri; i < 3*(firstTri+tris); i++ {
		src := tess.index(i)
		if remap[src] < 0 {
			remap[src] = int32(b.vertexCount)
			aaFlatteningPutVert(tess, src, localCoordsMatrix, verts, color, wideColor)
			b.vertexCount++
		}
		b.indices[b.indexCount] = uint16(remap[src])
		b.indexCount++
	}
	return tris
}

// localCoordsInverse returns the inverse of viewMatrix, for use as the local-coordinates matrix. A non-invertible view
// matrix falls back to the identity matrix: geom.Matrix's zero value is the all-zeros matrix, which would map every
// local coordinate to (0,0).
func localCoordsInverse(viewMatrix *geom.Matrix) geom.Matrix {
	if inv, ok := viewMatrix.Invert(); ok {
		return inv
	}
	return geom.IdentityMatrix()
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

// recordBatch records whatever the batch has accumulated as a single draw and empties it.
func (o *aaFlatteningConvexPathOp) recordBatch(state *OpFlushState, b *aaFlatteningBatch) {
	o.recordDraw(state, b.vertexCount, b.vertexStride, b.vertices, b.indexCount, b.indices)
	b.vertexCount = 0
	b.indexCount = 0
}

// recordSplitDraws records a tessellation that a single draw's uint16 indices cannot address as a series of
// self-contained draws (see aaFlatteningBuildSplitChunk).
func (o *aaFlatteningConvexPathOp) recordSplitDraws(state *OpFlushState, b *aaFlatteningBatch, tess *aaConvexTessellator, localCoordsMatrix *geom.Matrix, color colorcore.PMColor4f) {
	numTris := tess.numIndices() / 3
	for firstTri := 0; firstTri < numTris; {
		tris := aaFlatteningBuildSplitChunk(b, tess, firstTri, localCoordsMatrix, color, o.wideColor)
		if tris == 0 {
			break
		}
		firstTri += tris
		o.recordBatch(state, b)
	}
}

// OnPrepare implements Op.
func (o *aaFlatteningConvexPathOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	batch := aaFlatteningBatch{
		vertexStride: uint64(o.programInfo.GeomProc().gpBase().VertexStride()),
	}
	for i := range o.paths {
		args := &o.paths[i]
		tess := newAAConvexTessellator(args.style, args.strokeWidth, args.join,
			args.miterLimit)

		if !tess.tessellate(&args.viewMatrix, args.path) {
			continue
		}

		var localCoordsMatrix *geom.Matrix
		if o.helper.UsesLocalCoords() {
			ivm := localCoordsInverse(&args.viewMatrix)
			localCoordsMatrix = &ivm
		}

		currentVertices := tess.numPts()
		if currentVertices > aaFlatteningMaxVertices {
			// This one path's vertices cannot all be addressed by a uint16 index, so no bias into a shared batch can make
			// it fit. Flush what came before (keeping the draw order) and give the path its own split draws.
			o.recordBatch(state, &batch)
			o.recordSplitDraws(state, &batch, tess, localCoordsMatrix, args.color)
			continue
		}
		if batch.vertexCount+currentVertices > aaFlatteningMaxVertices {
			// If we added the current instance, we would overflow the indices we can store in a uint16. Draw what we've
			// got so far and reset.
			o.recordBatch(state, &batch)
		}
		currentIndices := tess.numIndices()
		batch.reserve(currentVertices, currentIndices)
		aaFlatteningExtractVerts(tess, localCoordsMatrix, batch.writer(),
			args.color, o.wideColor, uint16(batch.vertexCount), batch.indices[batch.indexCount:])
		batch.vertexCount += currentVertices
		batch.indexCount += currentIndices
	}
	o.recordBatch(state, &batch)
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
