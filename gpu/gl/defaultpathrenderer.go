// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// DefaultPathRenderer and the path stencil settings it draws with: the always-available non-AA/MSAA path renderer that
// flattens paths on the CPU into triangle fans or line primitives and resolves interior/exterior through the stencil
// buffer (single-pass for convex/hairline shapes, stencil-then-cover otherwise).

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

//////////////////////////////////////////////////////////////////////////////
// Path stencil settings

// Even/odd.
var eoStencilPass = MakeUserStencilSettings(0xffff, UserStencilTestAlwaysIfInClip, 0xffff,
	UserStencilOpInvert, UserStencilOpKeep, 0xffff)

// Ok not to check clip b/c stencil pass only wrote inside clip.
var eoColorPass = MakeUserStencilSettings(0x0000, UserStencilTestNotEqual, 0xffff,
	UserStencilOpZero, UserStencilOpZero, 0xffff)

// Have to check clip b/c outside clip will always be zero.
var invEOColorPass = MakeUserStencilSettings(0x0000, UserStencilTestEqualIfInClip, 0xffff,
	UserStencilOpZero, UserStencilOpZero, 0xffff)

// Winding.
var windStencilPass = MakeUserStencilSettingsSeparate(
	UserStencilFace{
		Ref: 0xffff, Test: UserStencilTestAlwaysIfInClip, TestMask: 0xffff,
		PassOp: UserStencilOpIncWrap, FailOp: UserStencilOpKeep, WriteMask: 0xffff,
	},
	UserStencilFace{
		Ref: 0xffff, Test: UserStencilTestAlwaysIfInClip, TestMask: 0xffff,
		PassOp: UserStencilOpDecWrap, FailOp: UserStencilOpKeep, WriteMask: 0xffff,
	},
)

// "0 < stencil" is equivalent to "0 != stencil".
var windColorPass = MakeUserStencilSettings(0x0000, UserStencilTestLessIfInClip, 0xffff,
	UserStencilOpZero, UserStencilOpZero, 0xffff)

var invWindColorPass = MakeUserStencilSettings(0x0000, UserStencilTestEqualIfInClip, 0xffff,
	UserStencilOpZero, UserStencilOpZero, 0xffff)

// Sometimes the default path renderer can draw a path directly to the stencil buffer without having to first resolve
// the interior/exterior.
var directToStencil = MakeUserStencilSettings(0x0000, UserStencilTestAlwaysIfInClip, 0xffff,
	UserStencilOpZero, UserStencilOpIncMaybeClamp, 0xffff)

//////////////////////////////////////////////////////////////////////////////
// Helpers for drawPath

// singlePassShape reports whether shape can be drawn in a single pass: inverse fills are always two-pass; this renderer
// only accepts simple fill paths or strokes that are hairline-equivalent, and hairlines are always single pass while
// filled paths are single pass if convex.
func singlePassShape(shape *StyledShape) bool {
	if shape.InverseFilled() {
		return false
	}
	if shape.Style().IsSimpleFill() {
		return shape.KnownToBeConvex()
	}
	return true
}

// isStrokeHairlineOrEquivalent reports whether style draws as a hairline or a stroke thin enough to be treated as one,
// for the path-effect-free Style.
func isStrokeHairlineOrEquivalent(style *Style, matrix *geom.Matrix) (coverage float32, ok bool) {
	rec := style.Rec()
	if rec.IsHairlineStyle() {
		return 1, true
	}
	if rec.Style() != stroke.StyleStroke {
		return 0, false
	}
	return glDrawTreatAAStrokeAsHairline(rec.Width(), matrix)
}

// glFastLen is the approximate vector length used by glDrawTreatAAStrokeAsHairline (the max+half-min approximation
// matches the CPU version in canvas/draw.go).
func glFastLen(pt geom.Point) float32 {
	x := geom.ScalarAbs(pt.X)
	y := geom.ScalarAbs(pt.Y)
	if x < y {
		x, y = y, x
	}
	return x + y/2
}

// glDrawTreatAAStrokeAsHairline reports whether an AA stroke of the given width is thin enough, under matrix, to be
// drawn as a modulated hairline instead.
func glDrawTreatAAStrokeAsHairline(strokeWidth float32, matrix *geom.Matrix) (float32, bool) {
	// We need to try to fake a thick-stroke with a modulated hairline.
	if matrix.HasPerspective() {
		return 0, false
	}
	src := [2]geom.Point{{X: strokeWidth, Y: 0}, {X: 0, Y: strokeWidth}}
	var dst [2]geom.Point
	matrix.MapVectors(dst[:], src[:])
	len0 := glFastLen(dst[0])
	len1 := glFastLen(dst[1])
	if len0 <= 1 && len1 <= 1 {
		return (len0 + len1) / 2, true
	}
	return 0, false
}

//////////////////////////////////////////////////////////////////////////////
// PathGeoBuilder

// pathGeoBuilder accumulates flattened path geometry into pooled vertex/index chunks, emitting a mesh per chunk.
type pathGeoBuilder struct {
	vertexBuffer      AnyBuffer
	indexBuffer       AnyBuffer
	target            *OpFlushState
	meshes            *[]*simpleMesh
	vertexData        []byte
	indexData         []byte
	firstVertex       int
	curVert           int // vertices written to the current chunk
	verticesInChunk   int
	firstIndex        int
	indicesInChunk    int
	curIdx            int // indices written to the current chunk
	subpathStartPoint geom.Point
	subpathIndexStart uint16
	primitiveType     gpu.PrimitiveType
	valid             bool
}

const pathGeoVertexStride = 8 // sizeof(geom.Point)

func newPathGeoBuilder(primitiveType gpu.PrimitiveType, target *OpFlushState, meshes *[]*simpleMesh) *pathGeoBuilder {
	b := &pathGeoBuilder{
		primitiveType: primitiveType, target: target, meshes: meshes,
		valid: true,
	}
	b.allocNewBuffers()
	return b
}

// Derived properties.
func (b *pathGeoBuilder) isIndexed() bool {
	return b.primitiveType == gpu.PrimitiveTypeLines || b.primitiveType == gpu.PrimitiveTypeTriangles
}

func (b *pathGeoBuilder) isHairline() bool {
	return b.primitiveType == gpu.PrimitiveTypeLines ||
		b.primitiveType == gpu.PrimitiveTypeLineStrip
}

func (b *pathGeoBuilder) indexScale() int {
	switch b.primitiveType {
	case gpu.PrimitiveTypeLines:
		return 2
	case gpu.PrimitiveTypeTriangles:
		return 3
	default:
		return 0
	}
}

func (b *pathGeoBuilder) currentIndex() uint16 { return uint16(b.curVert) }

func (b *pathGeoBuilder) putPoint(pt geom.Point) {
	off := b.curVert * pathGeoVertexStride
	binary.LittleEndian.PutUint32(b.vertexData[off:], math.Float32bits(pt.X))
	binary.LittleEndian.PutUint32(b.vertexData[off+4:], math.Float32bits(pt.Y))
	b.curVert++
}

func (b *pathGeoBuilder) putIndex(idx uint16) {
	binary.LittleEndian.PutUint16(b.indexData[b.curIdx*2:], idx)
	b.curIdx++
}

// allocNewBuffers ensures we always get enough verts for a worst-case quad/cubic, plus leftover points from the
// previous mesh piece (up to two verts to continue fanning).
func (b *pathGeoBuilder) allocNewBuffers() {
	if !b.valid {
		panic("allocNewBuffers on invalid builder")
	}
	const minVerticesPerChunk = pathUtilsMaxPointsPerCurve + 2
	const fallbackVerticesPerChunk = 16384

	b.vertexData, b.vertexBuffer, b.firstVertex, b.verticesInChunk = b.target.MakeVertexSpaceAtLeast(pathGeoVertexStride, minVerticesPerChunk,
		fallbackVerticesPerChunk)
	if b.vertexData == nil {
		b.curVert = 0
		b.curIdx = 0
		b.indexData = nil
		b.subpathIndexStart = 0
		b.valid = false
		return
	}

	if b.isIndexed() {
		// Similar to above: ensure we get enough indices for one worst-case quad/cubic. No extra indices are needed for
		// stitching.
		minIndicesPerChunk := pathUtilsMaxPointsPerCurve * b.indexScale()
		fallbackIndicesPerChunk := fallbackVerticesPerChunk * b.indexScale()
		b.indexData, b.indexBuffer, b.firstIndex, b.indicesInChunk = b.target.MakeIndexSpaceAtLeast(minIndicesPerChunk, fallbackIndicesPerChunk)
		if b.indexData == nil {
			b.vertexData = nil
			b.valid = false
			return
		}
	}

	b.curVert = 0
	b.curIdx = 0
	b.subpathIndexStart = 0
}

// appendContourEdgeIndices appends indices for the next contour edge: when drawing lines we're appending line segments
// along the contour; when applying the other fill rules we're drawing triangle fans around the start of the current
// (sub)path.
func (b *pathGeoBuilder) appendContourEdgeIndices(edgeV0Idx uint16) {
	if !b.isHairline() {
		b.putIndex(b.subpathIndexStart)
	}
	b.putIndex(edgeV0Idx)
	b.putIndex(edgeV0Idx + 1)
}

// createMeshAndPutBackReserve emits a single mesh with all accumulated vertex/index data.
func (b *pathGeoBuilder) createMeshAndPutBackReserve() {
	if !b.valid {
		return
	}
	vertexCount := b.curVert
	indexCount := b.curIdx

	var mesh *simpleMesh
	haveData := vertexCount > 0
	if b.isIndexed() {
		haveData = indexCount > 0
	}
	if haveData {
		mesh = &simpleMesh{}
		if !b.isIndexed() {
			mesh.vertexBuffer = b.vertexBuffer
			mesh.vertexCount = vertexCount
			mesh.baseVertex = b.firstVertex
			mesh.kind = simpleMeshNonIndexed
		} else {
			mesh.setIndexed(b.indexBuffer, indexCount, b.firstIndex, 0, uint16(vertexCount-1),
				b.vertexBuffer, b.firstVertex)
		}
	}

	b.target.PutBackIndices(b.indicesInChunk - indexCount)
	b.target.PutBackVertices(b.verticesInChunk-vertexCount, pathGeoVertexStride)
	b.verticesInChunk = 0
	b.indicesInChunk = 0

	if mesh != nil {
		*b.meshes = append(*b.meshes, mesh)
	}
}

// ensureSpace guarantees room for the next chunk of geometry, flushing and reallocating if needed. lastPoint (when
// non-nil) is re-appended to weld the new mesh to the previous one.
func (b *pathGeoBuilder) ensureSpace(vertsNeeded, indicesNeeded int, lastPoint *geom.Point) bool {
	if !b.valid {
		return false
	}
	if b.curVert+vertsNeeded > b.verticesInChunk || b.curIdx+indicesNeeded > b.indicesInChunk {
		// We are about to run out of space (possibly). Draw the mesh we've accumulated, put back any unused space, and
		// get new buffers.
		b.createMeshAndPutBackReserve()
		b.allocNewBuffers()
		if !b.valid {
			return false
		}
		// On moves we don't need to copy over any points to the new buffer (lastPoint is nil).
		if lastPoint != nil {
			// Append copies of the points we saved so the two meshes will weld properly.
			if !b.isHairline() {
				b.putPoint(b.subpathStartPoint)
			}
			b.putPoint(*lastPoint)
		}
	}
	return true
}

// moveTo begins a new subpath at p.
func (b *pathGeoBuilder) moveTo(p geom.Point) {
	if !b.ensureSpace(1, 0, nil) {
		return
	}
	if !b.isHairline() {
		b.subpathIndexStart = b.currentIndex()
		b.subpathStartPoint = p
	}
	b.putPoint(p)
}

// addLine appends a line segment ending at pts[1].
func (b *pathGeoBuilder) addLine(pts *[4]geom.Point) {
	if !b.ensureSpace(1, b.indexScale(), &pts[0]) {
		return
	}
	if b.isIndexed() {
		prevIdx := b.currentIndex() - 1
		b.appendContourEdgeIndices(prevIdx)
	}
	b.putPoint(pts[1])
}

// addQuad flattens and appends a quadratic curve.
func (b *pathGeoBuilder) addQuad(p0, p1, p2 geom.Point, srcSpaceTolSqd, srcSpaceTol float32) {
	if !b.ensureSpace(pathUtilsMaxPointsPerCurve, pathUtilsMaxPointsPerCurve*b.indexScale(),
		&p0) {
		return
	}
	// The first pt of the quad is the pt we ended on in the previous step.
	firstQPtIdx := b.currentIndex() - 1
	var scratch [pathUtilsMaxPointsPerCurve]geom.Point
	numPts := generateQuadraticPoints(p0, p1, p2, srcSpaceTolSqd, scratch[:],
		quadraticPointCount([3]geom.Point{p0, p1, p2}, srcSpaceTol))
	for i := 0; i < numPts; i++ {
		b.putPoint(scratch[i])
	}
	if b.isIndexed() {
		for i := 0; i < numPts; i++ {
			b.appendContourEdgeIndices(firstQPtIdx + uint16(i))
		}
	}
}

// addConic flattens and appends a conic (via quad subdivision).
func (b *pathGeoBuilder) addConic(weight float32, pts *[4]geom.Point, srcSpaceTolSqd, srcSpaceTol float32) {
	conic := geom.MakeConic(pts[0], pts[1], pts[2], weight)
	pow2 := conic.ComputeQuadPOW2(srcSpaceTol)
	quadStorage := make([]geom.Point, 1+2*(1<<pow2))
	count := conic.ChopIntoQuadsPOW2(quadStorage, pow2)
	for i := 0; i < count; i++ {
		q := quadStorage[i*2:]
		b.addQuad(q[0], q[1], q[2], srcSpaceTolSqd, srcSpaceTol)
	}
}

// addCubic flattens and appends a cubic curve.
func (b *pathGeoBuilder) addCubic(p0, p1, p2, p3 geom.Point, srcSpaceTolSqd, srcSpaceTol float32) {
	if !b.ensureSpace(pathUtilsMaxPointsPerCurve, pathUtilsMaxPointsPerCurve*b.indexScale(),
		&p0) {
		return
	}
	// The first pt of the cubic is the pt we ended on in the previous step.
	firstCPtIdx := b.currentIndex() - 1
	var scratch [pathUtilsMaxPointsPerCurve]geom.Point
	numPts := generateCubicPoints(p0, p1, p2, p3, srcSpaceTolSqd, scratch[:],
		cubicPointCount([4]geom.Point{p0, p1, p2, p3}, srcSpaceTol))
	for i := 0; i < numPts; i++ {
		b.putPoint(scratch[i])
	}
	if b.isIndexed() {
		for i := 0; i < numPts; i++ {
			b.appendContourEdgeIndices(firstCPtIdx + uint16(i))
		}
	}
}

// addPath flattens and appends every verb in p.
func (b *pathGeoBuilder) addPath(p *path.Path, srcSpaceTol float32) {
	srcSpaceTolSqd := srcSpaceTol * srcSpaceTol
	it := path.NewIter(p, false)
	for {
		verb, pts, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			b.moveTo(pts[0])
		case path.VerbLine:
			b.addLine(&pts)
		case path.VerbConic:
			b.addConic(it.ConicWeight(), &pts, srcSpaceTolSqd, srcSpaceTol)
		case path.VerbQuad:
			b.addQuad(pts[0], pts[1], pts[2], srcSpaceTolSqd, srcSpaceTol)
		case path.VerbCubic:
			b.addCubic(pts[0], pts[1], pts[2], pts[3], srcSpaceTolSqd, srcSpaceTol)
		case path.VerbClose:
		}
	}
}

// pathHasMultipleSubpaths reports whether p contains more than one moveTo-started contour.
func pathHasMultipleSubpaths(p *path.Path) bool {
	first := true
	it := path.NewIter(p, false)
	for {
		verb, _, ok := it.Next()
		if !ok {
			return false
		}
		if verb == path.VerbMove && !first {
			return true
		}
		first = false
	}
}

//////////////////////////////////////////////////////////////////////////////
// DefaultPathOp

var defaultPathOpClassID = GenOpClassID()

type defaultPathData struct {
	path      *path.Path
	tolerance float32
}

type defaultPathOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	paths       []defaultPathData
	meshes      []*simpleMesh
	OpBase
	viewMatrix geom.Matrix
	color      colorcore.PMColor4f
	coverage   uint8
	isHairline bool
}

// newDefaultPathOp creates an op to fill/stroke p. The paint is consumed.
func newDefaultPathOp(paint *Paint, p *path.Path, tolerance float32, coverage uint8, viewMatrix *geom.Matrix, isHairline bool, aaType gpu.AAType, devBounds geom.Rect, stencilSettings *UserStencilSettings) DrawOp {
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := &defaultPathOp{}
	o.InitOp(defaultPathOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, aaType, stencilSettings,
		PipelineInputFlagNone)
	o.color = color
	o.coverage = coverage
	o.viewMatrix = *viewMatrix
	o.isHairline = isHairline
	o.paths = append(o.paths, defaultPathData{path: p, tolerance: tolerance})

	o.SetBounds(devBounds, HasAABloat(aaType != gpu.AATypeNone), IsHairline(isHairline))
	return o
}

// Name implements Op.
func (o *defaultPathOp) Name() string { return "DefaultPathOp" }

// VisitProxies implements Op.
func (o *defaultPathOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *defaultPathOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *defaultPathOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *defaultPathOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	gpCoverage := AnalysisCoverageSingleChannel
	if o.coverage == 0xFF {
		gpCoverage = AnalysisCoverageNone
	}
	// This op uses uniform (not vertex) color, so doesn't need to track wide color.
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType, gpCoverage, &o.color, nil)
}

// primType returns the GPU primitive type this op draws.
func (o *defaultPathOp) primType() gpu.PrimitiveType {
	if o.isHairline {
		// We avoid indices when we have a single hairline contour.
		if len(o.paths) > 1 || pathHasMultipleSubpaths(o.paths[0].path) {
			return gpu.PrimitiveTypeLines
		}
		return gpu.PrimitiveTypeLineStrip
	}
	return gpu.PrimitiveTypeTriangles
}

// createProgramInfo builds the program info for this op's draw.
func (o *defaultPathOp) createProgramInfo(state *OpFlushState) {
	localCoordsType := LocalCoordsTypeUnused
	if o.helper.UsesLocalCoords() {
		localCoordsType = LocalCoordsTypeUsePosition
	}
	gp := MakeDefaultGeoProc(ColorTypePremulUniform, o.color, CoverageTypeUniform, o.coverage,
		localCoordsType, nil, &o.viewMatrix)
	if gp.gpBase().VertexStride() != pathGeoVertexStride {
		panic("unexpected vertex stride")
	}
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp, o.primType(),
		args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op.
func (o *defaultPathOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
	}
	builder := newPathGeoBuilder(o.primType(), state, &o.meshes)
	for i := range o.paths {
		builder.addPath(o.paths[i].path, o.paths[i].tolerance)
	}
	builder.createMeshAndPutBackReserve()
}

// OnExecute implements Op.
func (o *defaultPathOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
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
func (o *defaultPathOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*defaultPathOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}
	if o.color != that.color {
		return CombineResultCannotCombine
	}
	if o.coverage != that.coverage {
		return CombineResultCannotCombine
	}
	if !o.viewMatrix.CheapEqual(&that.viewMatrix) {
		return CombineResultCannotCombine
	}
	if o.isHairline != that.isHairline {
		return CombineResultCannotCombine
	}
	o.paths = append(o.paths, that.paths...)
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// DefaultPathRenderer

// DefaultPathRenderer is the always-available fallback path renderer using CPU flattening plus the stencil buffer.
type DefaultPathRenderer struct{}

// NewDefaultPathRenderer returns the renderer (it is stateless).
func NewDefaultPathRenderer() *DefaultPathRenderer { return &DefaultPathRenderer{} }

// Name implements PathRenderer.
func (r *DefaultPathRenderer) Name() string { return "Default" }

// OnGetStencilSupport implements PathRenderer.
func (r *DefaultPathRenderer) OnGetStencilSupport(shape *StyledShape) StencilSupport {
	if singlePassShape(shape) {
		return StencilSupportNoRestriction
	}
	return StencilSupportStencilOnly
}

// OnCanDrawPath implements PathRenderer.
func (r *DefaultPathRenderer) OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath {
	_, isHairline := isStrokeHairlineOrEquivalent(args.Shape.Style(), args.ViewMatrix)
	// If we aren't a single-pass shape or hairline, we require stencil buffers.
	if (!singlePassShape(args.Shape) && !isHairline) && !args.Proxy.CanUseStencil(args.Caps) {
		return CanDrawPathNo
	}
	// If antialiasing is required, we only support MSAA.
	if args.AAType != gpu.AATypeNone && args.AAType != gpu.AATypeMSAA {
		return CanDrawPathNo
	}
	// This can draw any path with any simple fill style.
	if !args.Shape.Style().IsSimpleFill() && !isHairline {
		return CanDrawPathNo
	}
	// Don't try to draw hairlines with DefaultPathRenderer if avoidLineDraws is true.
	if args.Caps.AvoidLineDraws && isHairline {
		return CanDrawPathNo
	}
	// Don't use this path renderer if we exceed the max verb count.
	if args.Shape.VerbCount() > pathRendererMaxVerbs {
		return CanDrawPathNo
	}
	// This is the fallback renderer for when a path is too complicated for the others to draw.
	return CanDrawPathAsBackup
}

// internalDrawPath does the actual stencil-then-cover (or single-pass) drawing. The paint is consumed.
func (r *DefaultPathRenderer) internalDrawPath(sdc *SurfaceDrawContext, paint *Paint, aaType gpu.AAType, userStencilSettings *UserStencilSettings, clip Clip, viewMatrix *geom.Matrix, shape *StyledShape, stencilOnly bool) bool {
	if aaType == gpu.AATypeCoverage {
		panic("DefaultPathRenderer cannot do coverage AA")
	}
	p := shape.AsPath()

	newCoverage := uint8(0xff)
	isHairline := false
	if hairlineCoverage, ok := isStrokeHairlineOrEquivalent(shape.Style(), viewMatrix); ok {
		newCoverage = uint8(hairlineCoverage*0xff + 0.5)
		isHairline = true
	} else if !shape.Style().IsSimpleFill() {
		panic("non-hairline shapes must be simple fills here")
	}

	var passes [2]*UserStencilSettings
	passCount := 0
	reverse := false
	lastPassIsBounds := false

	switch {
	case isHairline:
		passCount = 1
		if stencilOnly {
			passes[0] = &directToStencil
		} else {
			passes[0] = userStencilSettings
		}
		lastPassIsBounds = false
	case singlePassShape(shape):
		passCount = 1
		if stencilOnly {
			passes[0] = &directToStencil
		} else {
			passes[0] = userStencilSettings
		}
		lastPassIsBounds = false
	default:
		switch p.FillType() {
		case path.FillInverseEvenOdd, path.FillEvenOdd:
			reverse = p.FillType() == path.FillInverseEvenOdd
			passes[0] = &eoStencilPass
			if stencilOnly {
				passCount = 1
				lastPassIsBounds = false
			} else {
				passCount = 2
				lastPassIsBounds = true
				if reverse {
					passes[1] = &invEOColorPass
				} else {
					passes[1] = &eoColorPass
				}
			}
		case path.FillInverseWinding, path.FillWinding:
			reverse = p.FillType() == path.FillInverseWinding
			passes[0] = &windStencilPass
			passCount = 2
			if stencilOnly {
				lastPassIsBounds = false
				passCount--
			} else {
				lastPassIsBounds = true
				if reverse {
					passes[passCount-1] = &invWindColorPass
				} else {
					passes[passCount-1] = &windColorPass
				}
			}
		default:
			return false
		}
	}

	srcSpaceTol := scaleToleranceToSrc(pathUtilsDefaultTolerance, viewMatrix, p.Bounds())

	devBounds := getPathDevBounds(p, sdc.AsRenderTargetProxy().Proxy().BackingStoreDimensions(),
		viewMatrix)

	for pass := 0; pass < passCount; pass++ {
		if lastPassIsBounds && pass == passCount-1 {
			var bounds geom.Rect
			localMatrix := geom.IdentityMatrix()
			if reverse {
				// Draw over the dev bounds (which will be the whole dst surface for inv fill).
				bounds = devBounds
				// mapRect through a persp matrix may not be correct.
				vmi, invertible := viewMatrix.Invert()
				switch {
				case !viewMatrix.HasPerspective() && invertible:
					bounds, _ = vmi.MapRect(bounds)
				case invertible:
					localMatrix = vmi
				default:
					return false
				}
			} else {
				bounds = p.Bounds()
			}
			viewM := viewMatrix
			identity := geom.IdentityMatrix()
			if reverse && viewMatrix.HasPerspective() {
				viewM = &identity
			}
			// This is a non-coverage-aa rect op since we assert aaType != kCoverage at the start.
			sdc.StencilRect(clip, passes[pass], paint, gpu.AA(aaType == gpu.AATypeMSAA), viewM,
				bounds, &localMatrix)
		} else {
			stencilPass := stencilOnly || passCount > 1
			var op DrawOp
			if stencilPass {
				stencilPaint := NewPaint()
				stencilPaint.SetXPFactory(DisableColorXPFactory())
				op = newDefaultPathOp(stencilPaint, p, srcSpaceTol, newCoverage, viewMatrix,
					isHairline, aaType, devBounds, passes[pass])
			} else {
				op = newDefaultPathOp(paint, p, srcSpaceTol, newCoverage, viewMatrix,
					isHairline, aaType, devBounds, passes[pass])
			}
			sdc.AddDrawOp(clip, op)
		}
	}
	return true
}

// OnDrawPath implements PathRenderer.
func (r *DefaultPathRenderer) OnDrawPath(args *DrawPathArgs) bool {
	aaType := gpu.AATypeNone
	if args.AAType != gpu.AATypeNone {
		aaType = gpu.AATypeMSAA
	}
	return r.internalDrawPath(args.SurfaceDrawContext, args.Paint, aaType,
		args.UserStencilSettings, args.Clip, args.ViewMatrix, args.Shape, false)
}

// OnStencilPath implements PathRenderer.
func (r *DefaultPathRenderer) OnStencilPath(args *StencilPathArgs) {
	if args.Shape.InverseFilled() {
		panic("stencilPath cannot handle inverse fills")
	}
	paint := NewPaint()
	paint.SetXPFactory(DisableColorXPFactory())

	aaType := gpu.AATypeNone
	if args.DoStencilMSAA == gpu.AAYes {
		aaType = gpu.AATypeMSAA
	}
	r.internalDrawPath(args.SurfaceDrawContext, paint, aaType, UnusedStencilSettings(),
		args.Clip, args.ViewMatrix, args.Shape, true)
}
