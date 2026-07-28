// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The oval-drawing ops: CircleOp (with the arc clip planes and round caps), EllipseOp, DIEllipseOp, CircularRRectOp,
// EllipticalRRectOp, and the factory functions the SurfaceDrawContext draw entry points dispatch through. The
// mesh/pattern-helper/quad-helper machinery shared mesh-draw ops rely on reduces to the simpleMesh struct below. Trims:
// styling reduces to stroke.Rec (path effects devolve to paths at the device), so MakeCircleOp's dashed lanes and
// a butt-cap-dashed-circle variant are unreachable.

package gl

import (
	"encoding/binary"
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stroke"
)

// circleStaysCircle reports whether m maps circles to circles (rather than ellipses).
func circleStaysCircle(m *geom.Matrix) bool { return m.IsSimilarity() }

//////////////////////////////////////////////////////////////////////////////
// A single mesh draw, recorded at prepare time and replayed at execute time.

// simpleMesh holds the buffers and draw parameters recorded at prepare time and replayed at execute time.
type simpleMesh struct {
	indexBuffer  AnyBuffer
	vertexBuffer AnyBuffer
	baseVertex   int
	baseIndex    int
	indexCount   int
	vertexCount  int
	// Patterned draws (see setIndexedPatterned).
	patternIndexCount  int
	patternRepeatCount int
	maxPatternReps     int
	patternVertexCount int
	minIndexValue      uint16
	maxIndexValue      uint16
	kind               simpleMeshKind
}

type simpleMeshKind int8

const (
	simpleMeshInvalid simpleMeshKind = iota
	simpleMeshNonIndexed
	simpleMeshIndexed
	simpleMeshPatterned
)

func (m *simpleMesh) isValid() bool { return m.kind != simpleMeshInvalid }

// setIndexed configures the mesh for a single indexed draw.
func (m *simpleMesh) setIndexed(indexBuffer AnyBuffer, indexCount, baseIndex int, minIndexValue, maxIndexValue uint16, vertexBuffer AnyBuffer, baseVertex int) {
	m.indexBuffer = indexBuffer
	m.vertexBuffer = vertexBuffer
	m.indexCount = indexCount
	m.baseIndex = baseIndex
	m.minIndexValue = minIndexValue
	m.maxIndexValue = maxIndexValue
	m.baseVertex = baseVertex
	m.kind = simpleMeshIndexed
}

// setIndexedPatterned configures the mesh for a repeated indexed draw over a patterned index buffer.
func (m *simpleMesh) setIndexedPatterned(indexBuffer AnyBuffer, indexCount, vertexCount, patternRepeatCount, maxPatternRepetitionsInIndexBuffer int, vertexBuffer AnyBuffer, baseVertex int) {
	m.indexBuffer = indexBuffer
	m.vertexBuffer = vertexBuffer
	m.patternIndexCount = indexCount
	m.patternVertexCount = vertexCount
	m.patternRepeatCount = patternRepeatCount
	m.maxPatternReps = maxPatternRepetitionsInIndexBuffer
	m.baseVertex = baseVertex
	m.kind = simpleMeshPatterned
}

// draw replays the mesh on the render pass.
func (m *simpleMesh) draw(renderPass *OpsRenderPass) {
	renderPass.BindBuffers(m.indexBuffer, nil, m.vertexBuffer)
	switch m.kind {
	case simpleMeshIndexed:
		renderPass.DrawIndexed(m.indexCount, m.baseIndex, m.minIndexValue, m.maxIndexValue,
			m.baseVertex)
	case simpleMeshPatterned:
		renderPass.DrawIndexPattern(m.patternIndexCount, m.patternRepeatCount,
			m.maxPatternReps, m.patternVertexCount, m.baseVertex)
	case simpleMeshNonIndexed:
		renderPass.Draw(m.vertexCount, m.baseVertex)
	case simpleMeshInvalid:
		panic("draw of invalid mesh")
	}
}

// makePatternHelper allocates vertex space for repeatCount repetitions and records the patterned mesh. Returns the
// vertex bytes (nil on failure).
func makePatternHelper(state *OpFlushState, vertexStride uint64, indexBuffer *Buffer, verticesPerRepetition, indicesPerRepetition, repeatCount, maxRepetitions int, mesh *simpleMesh) []byte {
	if indexBuffer == nil {
		return nil
	}
	vertexCount := verticesPerRepetition * repeatCount
	vertices, vertexBuffer, firstVertex := state.MakeVertexSpace(vertexStride, vertexCount)
	if vertices == nil {
		return nil
	}
	mesh.setIndexedPatterned(indexBuffer, indicesPerRepetition, verticesPerRepetition,
		repeatCount, maxRepetitions, vertexBuffer, firstVertex)
	return vertices
}

// makeQuadHelper allocates vertex space for quadsToDraw non-AA quads and records the patterned mesh, using the shared
// non-AA quad index buffer.
func makeQuadHelper(state *OpFlushState, vertexStride uint64, quadsToDraw int, mesh *simpleMesh) []byte {
	return makePatternHelper(state, vertexStride, state.ResourceProvider().RefNonAAQuadIndexBuffer(),
		NumVertsPerNonAAQuad, NumIndicesPerNonAAQuad, quadsToDraw, MaxNumNonAAQuads, mesh)
}

// findOrCreatePatternedIndexBuffer returns the cached index buffer for the given pattern key, creating it if it doesn't
// already exist. The returned buffer is borrowed, not owned, matching RefNonAAQuadIndexBuffer: the creation ref is
// retained for the context's lifetime, so exactly one ref exists no matter how many ops ask for it. Callers must not
// unref it.
func (rp *ResourceProvider) findOrCreatePatternedIndexBuffer(pattern []uint16, patternSize, reps, vertCount int, key *gpu.UniqueKey) *Buffer {
	if buffer, ok := rp.FindByUniqueKey(key).(*Buffer); ok && buffer != nil {
		// FindByUniqueKey hands back a caller-owned ref. The creation ref below keeps the buffer alive, so release this
		// one immediately rather than accumulating one permanent ref per prepared op.
		buffer.Unref()
		return buffer
	}
	return rp.createPatternedIndexBuffer(pattern, patternSize, reps, vertCount, key)
}

// putU16Indices writes uint16 indices into a MakeIndexSpace allocation.
func putU16Indices(dst []byte, offset int, indices []uint16, indexAdjust int) int {
	for _, v := range indices {
		binary.LittleEndian.PutUint16(dst[offset:], uint16(int(v)+indexAdjust))
		offset += 2
	}
	return offset
}

// drawOpNoClipToShape provides DrawOp's default ClipToShape for ops that don't clip their own geometry.
type drawOpNoClipToShape struct{}

// ClipToShape implements DrawOp with the default failure.
func (drawOpNoClipToShape) ClipToShape(raster.ClipOp, *geom.Matrix, *Shape, gpu.AA) ClipResult {
	return ClipResultFail
}

//////////////////////////////////////////////////////////////////////////////
// Circle geometry constants.

// In the case of a normal fill, we draw geometry for the circle as an octagon.
var fillCircleIndices = []uint16{
	// enter the octagon
	0, 1, 8, 1, 2, 8,
	2, 3, 8, 3, 4, 8,
	4, 5, 8, 5, 6, 8,
	6, 7, 8, 7, 0, 8,
}

// For stroked circles, we use two nested octagons.
var strokeCircleIndices = []uint16{
	// enter the octagon
	0, 1, 9, 0, 9, 8,
	1, 2, 10, 1, 10, 9,
	2, 3, 11, 2, 11, 10,
	3, 4, 12, 3, 12, 11,
	4, 5, 13, 4, 13, 12,
	5, 6, 14, 5, 14, 13,
	6, 7, 15, 6, 15, 14,
	7, 0, 8, 7, 8, 15,
}

// Normalized geometry for octagons that circumscribe and lie on a circle:

const octOffset = 0.41421356237 // sqrt(2) - 1

var octagonOuter = [8]geom.Point{
	{X: -octOffset, Y: -1},
	{X: octOffset, Y: -1},
	{X: 1, Y: -octOffset},
	{X: 1, Y: octOffset},
	{X: octOffset, Y: 1},
	{X: -octOffset, Y: 1},
	{X: -1, Y: octOffset},
	{X: -1, Y: -octOffset},
}

// Cosine and sine of pi/8.
const (
	cosPi8 = 0.923579533
	sinPi8 = 0.382683432
)

var octagonInner = [8]geom.Point{
	{X: -sinPi8, Y: -cosPi8},
	{X: sinPi8, Y: -cosPi8},
	{X: cosPi8, Y: -sinPi8},
	{X: cosPi8, Y: sinPi8},
	{X: sinPi8, Y: cosPi8},
	{X: -sinPi8, Y: cosPi8},
	{X: -cosPi8, Y: sinPi8},
	{X: -cosPi8, Y: -sinPi8},
}

const (
	vertsPerStrokeCircle = 16
	vertsPerFillCircle   = 9
)

func circleTypeToVertCount(stroked bool) int {
	if stroked {
		return vertsPerStrokeCircle
	}
	return vertsPerFillCircle
}

func circleTypeToIndexCount(stroked bool) int {
	if stroked {
		return len(strokeCircleIndices)
	}
	return len(fillCircleIndices)
}

func circleTypeToIndices(stroked bool) []uint16 {
	if stroked {
		return strokeCircleIndices
	}
	return fillCircleIndices
}

//////////////////////////////////////////////////////////////////////////////
// CircleOp

var circleOpClassID = GenOpClassID()

// circleOpPool is the free list of circleOp shells (see oppool.go).
var circleOpPool opPool[circleOp]

// ArcParams holds optional extra params to render a partial arc rather than a full circle.
type ArcParams struct {
	StartAngleRadians float32
	SweepAngleRadians float32
	UseCenter         bool
}

// circleGeom holds one circle instance's per-draw geometry and color.
type circleGeom struct {
	color           colorcore.PMColor4f
	innerRadius     float32
	outerRadius     float32
	clipPlane       [3]float32
	isectPlane      [3]float32
	unionPlane      [3]float32
	roundCapCenters [2]geom.Point
	devBounds       geom.Rect
	stroked         bool
}

// circleOp draws one or more circles (or circular arcs), batched together when compatible.
type circleOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	circles     []circleGeom
	OpBase
	mesh       simpleMesh
	vertCount  int
	indexCount int
	// circlesArr backs circles for the common single-instance case (see the aaStrokeRectOp.rectsArr note): the
	// constructor always appends exactly one geom, so a fresh op needs no separate backing array; combining grows past
	// it into the heap as normal.
	circlesArr                   [1]circleGeom
	viewMatrixIfUsingLocalCoords geom.Matrix
	clipPlane                    bool
	roundCaps                    bool
	wideColor                    bool
	clipPlaneUnion               bool
	clipPlaneIsect               bool
	allFill                      bool
}

// NewCircleOp returns a DrawOp that fills or strokes a circle, or nil if the combination of arcParams and strokeRec
// isn't supported. The paint is consumed; arcParams may be nil.
func NewCircleOp(paint *Paint, viewMatrix *geom.Matrix, center geom.Point, radius float32, strokeRec *stroke.Rec, arcParams *ArcParams) DrawOp {
	if !circleStaysCircle(viewMatrix) {
		panic("circle op requires a similarity matrix")
	}
	recStyle := strokeRec.Style()
	if arcParams != nil {
		// Arc support depends on the style.
		switch recStyle {
		case stroke.StyleStrokeAndFill:
			// This produces a strange result that this op doesn't implement.
			return nil
		case stroke.StyleFill:
			// This supports all fills.
		case stroke.StyleStroke:
			// Strokes that don't use the center point are supported with butt and round caps.
			if arcParams.UseCenter || strokeRec.Cap() == stroke.CapSquare {
				return nil
			}
		case stroke.StyleHairline:
			// Hairline only supports butt cap. Round caps could be emulated by slightly extending the angle range if we
			// ever cared to.
			if arcParams.UseCenter || strokeRec.Cap() != stroke.CapButt {
				return nil
			}
		}
	}
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}
	return newCircleOp(processors, color, viewMatrix, center, radius, strokeRec, arcParams)
}

// newCircleOp is the CircleOp constructor.
func newCircleOp(processors *ProcessorSet, color colorcore.PMColor4f, viewMatrix *geom.Matrix, center geom.Point, radius float32, strokeRec *stroke.Rec, arcParams *ArcParams) *circleOp {
	o := circleOpPool.borrow()
	o.circles = bootstrapInstances(o.circles, o.circlesArr[:0])
	o.InitOp(circleOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, nil,
		PipelineInputFlagNone)
	recStyle := strokeRec.Style()

	center = viewMatrix.MapPoint(center)
	radius = viewMatrix.MapRadius(radius)
	strokeWidth := viewMatrix.MapRadius(strokeRec.Width())

	isStrokeOnly := recStyle == stroke.StyleStroke || recStyle == stroke.StyleHairline
	hasStroke := isStrokeOnly || recStyle == stroke.StyleStrokeAndFill

	innerRadius := float32(-0.5)
	outerRadius := radius
	halfWidth := float32(0)
	if hasStroke {
		if geom.ScalarNearlyZero(strokeWidth) {
			halfWidth = 0.5
		} else {
			halfWidth = strokeWidth / 2
		}
		outerRadius += halfWidth
		if isStrokeOnly {
			innerRadius = radius - halfWidth
		}
	}

	// The radii are outset for two reasons. First, it allows the shader to simply perform simpler computation because
	// the computed alpha is zero, rather than 50%, at the radius. Second, the outer radius is used to compute the verts
	// of the bounding box that is rendered and the outset ensures the box will cover all pixels partially covered by
	// the circle.
	outerRadius += 0.5
	innerRadius -= 0.5
	stroked := isStrokeOnly && innerRadius > 0
	o.viewMatrixIfUsingLocalCoords = *viewMatrix

	// This makes every point fully inside the intersection plane.
	unusedIsectPlane := [3]float32{0, 0, 1}
	// This makes every point fully outside the union plane.
	unusedUnionPlane := [3]float32{0, 0, 0}
	unusedRoundCaps := [2]geom.Point{{X: 1e10, Y: 1e10}, {X: 1e10, Y: 1e10}}
	devBounds := geom.Rect{
		Left: center.X - outerRadius, Top: center.Y - outerRadius,
		Right: center.X + outerRadius, Bottom: center.Y + outerRadius,
	}
	if arcParams != nil {
		// The shader operates in a space where the circle is translated to be centered at the origin. Here we compute
		// points on the unit circle at the starting and ending angles.
		sinStart, cosStart := math.Sincos(float64(arcParams.StartAngleRadians))
		startPoint := geom.Point{X: float32(cosStart), Y: float32(sinStart)}
		endAngle := float64(arcParams.StartAngleRadians + arcParams.SweepAngleRadians)
		sinEnd, cosEnd := math.Sincos(endAngle)
		stopPoint := geom.Point{X: float32(cosEnd), Y: float32(sinEnd)}

		// Adjust the start and end points based on the view matrix (to handle rotated arcs).
		startPoint = viewMatrix.MapVector(startPoint)
		stopPoint = viewMatrix.MapVector(stopPoint)
		startPoint.Normalize()
		stopPoint.Normalize()

		// We know the matrix is a similarity here. Detect mirroring which will affect how we should orient the clip
		// planes for arcs.
		upperLeftDet := viewMatrix.Get(geom.MScaleX)*viewMatrix.Get(geom.MScaleY) -
			viewMatrix.Get(geom.MSkewX)*viewMatrix.Get(geom.MSkewY)
		if upperLeftDet < 0 {
			startPoint, stopPoint = stopPoint, startPoint
		}

		o.roundCaps = hasStroke && strokeRec.Width() > 0 && strokeRec.Cap() == stroke.CapRound
		var roundCaps [2]geom.Point
		if o.roundCaps {
			// Compute the cap center points in the normalized space.
			midRadius := (innerRadius + outerRadius) / (2 * outerRadius)
			roundCaps[0] = geom.Point{X: startPoint.X * midRadius, Y: startPoint.Y * midRadius}
			roundCaps[1] = geom.Point{X: stopPoint.X * midRadius, Y: stopPoint.Y * midRadius}
		} else {
			roundCaps = unusedRoundCaps
		}

		// Like a fill without useCenter, butt-cap stroke can be implemented by clipping against radial lines. We treat
		// round caps the same way, but tack coverage of circles at the centers of the butts. However, in both cases we
		// have to be careful about the half-circle case: the two radial lines are equal and so that edge gets clipped
		// twice. Since the shared edge goes through the center we fall back on the !useCenter case.
		absSweep := geom.ScalarAbs(arcParams.SweepAngleRadians)
		useCenter := (arcParams.UseCenter || isStrokeOnly) &&
			!geom.ScalarNearlyEqual(absSweep, math.Pi)
		if useCenter {
			norm0 := geom.Point{X: startPoint.Y, Y: -startPoint.X}
			norm1 := geom.Point{X: stopPoint.Y, Y: -stopPoint.X}
			// This ensures that norm0 is always the clockwise plane, and norm1 is CCW.
			if arcParams.SweepAngleRadians < 0 {
				norm0, norm1 = norm1, norm0
			}
			norm0 = geom.Point{X: -norm0.X, Y: -norm0.Y}
			o.clipPlane = true
			if absSweep > math.Pi {
				o.circles = append(o.circles, circleGeom{
					color:           color,
					innerRadius:     innerRadius,
					outerRadius:     outerRadius,
					clipPlane:       [3]float32{norm0.X, norm0.Y, 0.5},
					isectPlane:      unusedIsectPlane,
					unionPlane:      [3]float32{norm1.X, norm1.Y, 0.5},
					roundCapCenters: roundCaps,
					devBounds:       devBounds,
					stroked:         stroked,
				})
				o.clipPlaneIsect = false
				o.clipPlaneUnion = true
			} else {
				o.circles = append(o.circles, circleGeom{
					color:           color,
					innerRadius:     innerRadius,
					outerRadius:     outerRadius,
					clipPlane:       [3]float32{norm0.X, norm0.Y, 0.5},
					isectPlane:      [3]float32{norm1.X, norm1.Y, 0.5},
					unionPlane:      unusedUnionPlane,
					roundCapCenters: roundCaps,
					devBounds:       devBounds,
					stroked:         stroked,
				})
				o.clipPlaneIsect = true
				o.clipPlaneUnion = false
			}
		} else {
			// We clip to a secant of the original circle.
			startPoint = geom.Point{X: startPoint.X * radius, Y: startPoint.Y * radius}
			stopPoint = geom.Point{X: stopPoint.X * radius, Y: stopPoint.Y * radius}
			norm := geom.Point{X: startPoint.Y - stopPoint.Y, Y: stopPoint.X - startPoint.X}
			norm.Normalize()
			if arcParams.SweepAngleRadians > 0 {
				norm = geom.Point{X: -norm.X, Y: -norm.Y}
			}
			d := -(norm.X*startPoint.X + norm.Y*startPoint.Y) + 0.5
			o.circles = append(o.circles, circleGeom{
				color:           color,
				innerRadius:     innerRadius,
				outerRadius:     outerRadius,
				clipPlane:       [3]float32{norm.X, norm.Y, d},
				isectPlane:      unusedIsectPlane,
				unionPlane:      unusedUnionPlane,
				roundCapCenters: roundCaps,
				devBounds:       devBounds,
				stroked:         stroked,
			})
			o.clipPlane = true
			o.clipPlaneIsect = false
			o.clipPlaneUnion = false
		}
	} else {
		o.circles = append(o.circles, circleGeom{
			color:           color,
			innerRadius:     innerRadius,
			outerRadius:     outerRadius,
			clipPlane:       unusedIsectPlane,
			isectPlane:      unusedIsectPlane,
			unionPlane:      unusedUnionPlane,
			roundCapCenters: unusedRoundCaps,
			devBounds:       devBounds,
			stroked:         stroked,
		})
	}
	// Use the original radius and stroke radius for the bounds so that it does not include the AA bloat.
	radius += halfWidth
	o.SetBounds(geom.Rect{
		Left: center.X - radius, Top: center.Y - radius,
		Right: center.X + radius, Bottom: center.Y + radius,
	}, HasAABloat(true),
		IsHairline(false))
	o.vertCount = circleTypeToVertCount(stroked)
	o.indexCount = circleTypeToIndexCount(stroked)
	o.allFill = !stroked
	return o
}

// Name implements Op.
func (o *circleOp) Name() string { return "CircleOp" }

// recycle implements Op: returns this dead op to its free list.
func (o *circleOp) recycle() {
	recycleProgramInfo(o.programInfo)
	recycleKeepingBacking(&circleOpPool, o,
		func(o *circleOp) []circleGeom { return o.circles },
		func(o *circleOp, s []circleGeom) { o.circles = s })
}

// VisitProxies implements Op.
func (o *circleOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *circleOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *circleOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *circleOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.circles[0].color, &o.wideColor)
}

// createProgramInfo builds the geometry processor and program info for this circle op.
func (o *circleOp) createProgramInfo(state *OpFlushState) {
	args := state.OpArgs()
	if args.UsesMSAASurface() {
		panic("circle op is coverage-AA only")
	}
	localMatrix, ok := o.viewMatrixIfUsingLocalCoords.Invert()
	if !ok {
		return
	}
	gp := makeCircleGeometryProcessor(!o.allFill, o.clipPlane, o.clipPlaneIsect,
		o.clipPlaneUnion, o.roundCaps, o.wideColor, &localMatrix)
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// writeCircleExtras writes the optional per-vertex plane/cap attributes.
func (o *circleOp) writeCircleExtras(w *quadVertexWriter, circle *circleGeom) {
	if o.clipPlane {
		w.putF32(circle.clipPlane[0])
		w.putF32(circle.clipPlane[1])
		w.putF32(circle.clipPlane[2])
	}
	if o.clipPlaneIsect {
		w.putF32(circle.isectPlane[0])
		w.putF32(circle.isectPlane[1])
		w.putF32(circle.isectPlane[2])
	}
	if o.clipPlaneUnion {
		w.putF32(circle.unionPlane[0])
		w.putF32(circle.unionPlane[1])
		w.putF32(circle.unionPlane[2])
	}
	if o.roundCaps {
		w.putF32(circle.roundCapCenters[0].X)
		w.putF32(circle.roundCapCenters[0].Y)
		w.putF32(circle.roundCapCenters[1].X)
		w.putF32(circle.roundCapCenters[1].Y)
	}
}

// OnPrepare implements Op: it builds the vertex and index buffers for the batched circles.
func (o *circleOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	stride := o.programInfo.GeomProc().gpBase().VertexStride()
	vertices, vertexBuffer, firstVertex := state.MakeVertexSpace(uint64(stride), o.vertCount)
	if vertices == nil {
		return
	}
	indices, indexBuffer, firstIndex := state.MakeIndexSpace(o.indexCount)
	if indices == nil {
		return
	}

	w := quadVertexWriter{buf: vertices}
	idxOffset := 0
	currStartVertex := 0
	for c := range o.circles {
		circle := &o.circles[c]
		innerRadius := circle.innerRadius
		outerRadius := circle.outerRadius
		bounds := circle.devBounds

		// The inner radius in the vertex data must be specified in normalized space.
		innerRadius /= outerRadius

		center := geom.Point{
			X: (bounds.Left + bounds.Right) / 2,
			Y: (bounds.Top + bounds.Bottom) / 2,
		}
		halfWidth := bounds.Width() / 2

		geoClipPlane := geom.Point{}
		offsetClipDist := float32(1)
		if !circle.stroked && o.clipPlane && o.clipPlaneIsect &&
			circle.clipPlane[0]*circle.isectPlane[0]+
				circle.clipPlane[1]*circle.isectPlane[1] < 0 {
			// Acute arc. Clip the vertices to the perpendicular half-plane. We've constructed the clip plane to be
			// clockwise, and the isect plane to be CCW, so we can rotate them each 90 degrees to point "out", then
			// average them. We back off by 1/2 pixel so the AA can extend just past the center of the circle.
			geoClipPlane = geom.Point{
				X: circle.clipPlane[1] - circle.isectPlane[1],
				Y: circle.isectPlane[0] - circle.clipPlane[0],
			}
			if !geoClipPlane.Normalize() {
				panic("degenerate geo clip plane")
			}
			offsetClipDist = 0.5 / halfWidth
		}

		for i := range 8 {
			// This clips the normalized offset to the half-plane we computed above. Then we compute the vertex position
			// from this.
			dist := min(octagonOuter[i].X*geoClipPlane.X+octagonOuter[i].Y*geoClipPlane.Y+
				offsetClipDist, 0)
			offset := geom.Point{
				X: octagonOuter[i].X - geoClipPlane.X*dist,
				Y: octagonOuter[i].Y - geoClipPlane.Y*dist,
			}
			w.putF32(center.X + offset.X*halfWidth)
			w.putF32(center.Y + offset.Y*halfWidth)
			w.putColor(circle.color, o.wideColor)
			w.putF32(offset.X)
			w.putF32(offset.Y)
			w.putF32(outerRadius)
			w.putF32(innerRadius)
			o.writeCircleExtras(&w, circle)
		}

		if circle.stroked {
			// Compute the inner ring.
			for i := range 8 {
				w.putF32(center.X + octagonInner[i].X*circle.innerRadius)
				w.putF32(center.Y + octagonInner[i].Y*circle.innerRadius)
				w.putColor(circle.color, o.wideColor)
				w.putF32(octagonInner[i].X * innerRadius)
				w.putF32(octagonInner[i].Y * innerRadius)
				w.putF32(outerRadius)
				w.putF32(innerRadius)
				o.writeCircleExtras(&w, circle)
			}
		} else {
			// Filled.
			w.putF32(center.X)
			w.putF32(center.Y)
			w.putColor(circle.color, o.wideColor)
			w.putF32(0)
			w.putF32(0)
			w.putF32(outerRadius)
			w.putF32(innerRadius)
			o.writeCircleExtras(&w, circle)
		}

		idxOffset = putU16Indices(indices, idxOffset, circleTypeToIndices(circle.stroked),
			currStartVertex)
		currStartVertex += circleTypeToVertCount(circle.stroked)
	}

	o.mesh.setIndexed(indexBuffer, o.indexCount, firstIndex, 0, uint16(o.vertCount-1),
		vertexBuffer, firstVertex)
}

// OnExecute implements Op: it binds the program and mesh and issues the draw for the batched circles.
func (o *circleOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.programInfo == nil || !o.mesh.isValid() {
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
	o.mesh.draw(renderPass)
}

// OnCombineIfPossible implements Op: it merges o with another circleOp when their processors, bounds, and vertex budget
// are compatible.
func (o *circleOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*circleOp)
	if !ok {
		return CombineResultCannotCombine
	}

	// Can only represent 65535 unique vertices with 16-bit indices.
	if o.vertCount+that.vertCount > 65536 {
		return CombineResultCannotCombine
	}

	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}

	if o.helper.UsesLocalCoords() &&
		!o.viewMatrixIfUsingLocalCoords.CheapEqual(&that.viewMatrixIfUsingLocalCoords) {
		return CombineResultCannotCombine
	}

	// Because we've set up the ops that don't use the planes with noop values we can just accumulate used planes by
	// later ops.
	o.clipPlane = o.clipPlane || that.clipPlane
	o.clipPlaneIsect = o.clipPlaneIsect || that.clipPlaneIsect
	o.clipPlaneUnion = o.clipPlaneUnion || that.clipPlaneUnion
	o.roundCaps = o.roundCaps || that.roundCaps
	o.wideColor = o.wideColor || that.wideColor

	o.circles = append(o.circles, that.circles...)
	o.vertCount += that.vertCount
	o.indexCount += that.indexCount
	o.allFill = o.allFill && that.allFill
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// EllipseOp

var ellipseOpClassID = GenOpClassID()

// ellipseOpPool is the free list of ellipseOp shells (see oppool.go).
var ellipseOpPool opPool[ellipseOp]

// ellipseGeom holds one ellipse instance's per-draw geometry and color.
type ellipseGeom struct {
	color        colorcore.PMColor4f
	xRadius      float32
	yRadius      float32
	innerXRadius float32
	innerYRadius float32
	devBounds    geom.Rect
}

// ellipseOp draws one or more ellipses, batched together when compatible.
type ellipseOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	ellipses    []ellipseGeom
	OpBase
	mesh simpleMesh
	// ellipsesArr backs ellipses for the common single-instance case (see aaStrokeRectOp.rectsArr).
	ellipsesArr                  [1]ellipseGeom
	viewMatrixIfUsingLocalCoords geom.Matrix
	stroked                      bool
	wideColor                    bool
	useScale                     bool
}

// NewEllipseOp returns a DrawOp that fills or strokes an axis-aligned ellipse under a scale+translate matrix, or nil if
// the ellipse/stroke combination isn't supported. The paint is consumed.
func NewEllipseOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, ellipse geom.Rect, strokeRec *stroke.Rec) DrawOp {
	// Do any matrix crunching before we reset the draw state for device coords.
	center := viewMatrix.MapPoint(geom.Point{
		X: (ellipse.Left + ellipse.Right) / 2,
		Y: (ellipse.Top + ellipse.Bottom) / 2,
	})
	ellipseXRadius := ellipse.Width() / 2
	ellipseYRadius := ellipse.Height() / 2
	xRadius := geom.ScalarAbs(viewMatrix.Get(geom.MScaleX)*ellipseXRadius +
		viewMatrix.Get(geom.MSkewX)*ellipseYRadius)
	yRadius := geom.ScalarAbs(viewMatrix.Get(geom.MSkewY)*ellipseXRadius +
		viewMatrix.Get(geom.MScaleY)*ellipseYRadius)

	// Do (potentially) anisotropic mapping of stroke.
	strokeWidth := strokeRec.Width()
	scaledStrokeX := geom.ScalarAbs(strokeWidth * (viewMatrix.Get(geom.MScaleX) +
		viewMatrix.Get(geom.MSkewY)))
	scaledStrokeY := geom.ScalarAbs(strokeWidth * (viewMatrix.Get(geom.MSkewX) +
		viewMatrix.Get(geom.MScaleY)))

	style := strokeRec.Style()
	isStrokeOnly := style == stroke.StyleStroke || style == stroke.StyleHairline
	hasStroke := isStrokeOnly || style == stroke.StyleStrokeAndFill

	innerXRadius := float32(0)
	innerYRadius := float32(0)
	if hasStroke {
		strokeLen := float32(math.Hypot(float64(scaledStrokeX), float64(scaledStrokeY)))
		if geom.ScalarNearlyZero(strokeLen) {
			scaledStrokeX = 0.5
			scaledStrokeY = 0.5
		} else {
			scaledStrokeX /= 2
			scaledStrokeY /= 2
		}

		// We only handle thick strokes for near-circular ellipses.
		if float32(math.Hypot(float64(scaledStrokeX), float64(scaledStrokeY))) > 0.5 &&
			(0.5*xRadius > yRadius || 0.5*yRadius > xRadius) {
			return nil
		}

		// We don't handle it if the curvature of the stroke is less than the curvature of the ellipse.
		if scaledStrokeX*(xRadius*yRadius) < (scaledStrokeY*scaledStrokeY)*xRadius ||
			scaledStrokeY*(xRadius*xRadius) < (scaledStrokeX*scaledStrokeX)*yRadius {
			return nil
		}

		// This is legit only if scale & translation (which should be the case at the moment).
		if isStrokeOnly {
			innerXRadius = xRadius - scaledStrokeX
			innerYRadius = yRadius - scaledStrokeY
		}

		xRadius += scaledStrokeX
		yRadius += scaledStrokeY
	}

	// For large ovals with low precision floats, we fall back to the path renderer. To compute the AA at the edge we
	// divide by the gradient, which is clamped to a minimum value to avoid divides by zero. With large ovals and low
	// precision this leads to blurring at the edge of the oval.
	const maxOvalRadius = 16384
	if !caps.ShaderCaps.FloatIs32Bits && (xRadius >= maxOvalRadius || yRadius >= maxOvalRadius) {
		return nil
	}

	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := ellipseOpPool.borrow()
	o.ellipses = bootstrapInstances(o.ellipses, o.ellipsesArr[:0])
	o.InitOp(ellipseOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, nil,
		PipelineInputFlagNone)
	o.ellipses = append(o.ellipses, ellipseGeom{
		color:        color,
		xRadius:      xRadius,
		yRadius:      yRadius,
		innerXRadius: innerXRadius,
		innerYRadius: innerYRadius,
		devBounds: geom.Rect{
			Left: center.X - xRadius, Top: center.Y - yRadius,
			Right: center.X + xRadius, Bottom: center.Y + yRadius,
		},
	})
	o.SetBounds(o.ellipses[0].devBounds, HasAABloat(true), IsHairline(false))
	o.stroked = isStrokeOnly && innerXRadius > 0 && innerYRadius > 0
	o.viewMatrixIfUsingLocalCoords = *viewMatrix
	return o
}

// Name implements Op.
func (o *ellipseOp) Name() string { return "EllipseOp" }

// recycle implements Op: returns this dead op to its free list.
func (o *ellipseOp) recycle() {
	recycleProgramInfo(o.programInfo)
	recycleKeepingBacking(&ellipseOpPool, o,
		func(o *ellipseOp) []ellipseGeom { return o.ellipses },
		func(o *ellipseOp, s []ellipseGeom) { o.ellipses = s })
}

// VisitProxies implements Op.
func (o *ellipseOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *ellipseOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *ellipseOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *ellipseOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	o.useScale = !caps.ShaderCaps.FloatIs32Bits && !caps.ShaderCaps.HasLowFragmentPrecision
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.ellipses[0].color, &o.wideColor)
}

// createProgramInfo builds the geometry processor and program info for this ellipse op.
func (o *ellipseOp) createProgramInfo(state *OpFlushState) {
	localMatrix, ok := o.viewMatrixIfUsingLocalCoords.Invert()
	if !ok {
		return
	}
	gp := makeEllipseGeometryProcessor(o.stroked, o.wideColor, o.useScale, &localMatrix)
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op: it builds the vertex buffer for the batched ellipses.
func (o *ellipseOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	stride := o.programInfo.GeomProc().gpBase().VertexStride()
	vertices := makeQuadHelper(state, uint64(stride), len(o.ellipses), &o.mesh)
	if vertices == nil {
		return
	}
	w := quadVertexWriter{buf: vertices}

	// On MSAA, bloat enough to guarantee any pixel that might be touched by the ellipse has full sample coverage.
	aaBloat := float32(0.5)
	if state.OpArgs().UsesMSAASurface() {
		aaBloat = math.Sqrt2
	}

	for e := range o.ellipses {
		ellipse := &o.ellipses[e]
		xRadius := ellipse.xRadius
		yRadius := ellipse.yRadius

		// Compute the reciprocals of the radii here to save time in the shader.
		invRadii := [4]float32{
			1 / xRadius,
			1 / yRadius,
			1 / ellipse.innerXRadius,
			1 / ellipse.innerYRadius,
		}
		xMaxOffset := xRadius + aaBloat
		yMaxOffset := yRadius + aaBloat

		if !o.stroked {
			// For filled ellipses we map a unit circle in the vertex attributes rather than computing an ellipse and
			// modifying that distance, so we normalize to 1.
			xMaxOffset /= xRadius
			yMaxOffset /= yRadius
		}

		bounds := ellipse.devBounds.Outset(aaBloat, aaBloat)
		// Tri-strip corner order: (l,t), (l,b), (r,t), (r,b); the offsets mirror origin_centered_tri_strip(xMaxOffset,
		// yMaxOffset).
		for corner := range 4 {
			x := bounds.Left
			offX := -xMaxOffset
			if corner >= 2 {
				x = bounds.Right
				offX = xMaxOffset
			}
			y := bounds.Top
			offY := -yMaxOffset
			if corner&1 != 0 {
				y = bounds.Bottom
				offY = yMaxOffset
			}
			w.putF32(x)
			w.putF32(y)
			w.putColor(ellipse.color, o.wideColor)
			w.putF32(offX)
			w.putF32(offY)
			if o.useScale {
				w.putF32(max(xRadius, yRadius))
			}
			w.putF32(invRadii[0])
			w.putF32(invRadii[1])
			w.putF32(invRadii[2])
			w.putF32(invRadii[3])
		}
	}
}

// OnExecute implements Op: it binds the program and mesh and issues the draw for the batched ellipses.
func (o *ellipseOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.programInfo == nil || !o.mesh.isValid() {
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
	o.mesh.draw(renderPass)
}

// OnCombineIfPossible implements Op: it merges o with another ellipseOp when their processors, stroke style, and (if
// using local coords) view matrix match.
func (o *ellipseOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*ellipseOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}
	if o.stroked != that.stroked {
		return CombineResultCannotCombine
	}
	if o.helper.UsesLocalCoords() &&
		!o.viewMatrixIfUsingLocalCoords.CheapEqual(&that.viewMatrixIfUsingLocalCoords) {
		return CombineResultCannotCombine
	}
	o.ellipses = append(o.ellipses, that.ellipses...)
	o.wideColor = o.wideColor || that.wideColor
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// DIEllipseOp

var diEllipseOpClassID = GenOpClassID()

// diEllipseOpPool is the free list of diEllipseOp shells (see oppool.go).
var diEllipseOpPool opPool[diEllipseOp]

// diEllipseGeom holds one device-independent ellipse instance's per-draw geometry and color.
type diEllipseGeom struct {
	viewMatrix   geom.Matrix
	color        colorcore.PMColor4f
	xRadius      float32
	yRadius      float32
	innerXRadius float32
	innerYRadius float32
	geoDx        float32
	geoDy        float32
	style        diEllipseStyle
	bounds       geom.Rect
}

// diEllipseOp draws one or more ellipses under an arbitrary (non-scale/translate) view matrix, batched together when
// compatible.
type diEllipseOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	ellipses    []diEllipseGeom
	OpBase
	mesh simpleMesh
	// ellipsesArr backs ellipses for the common single-instance case (see aaStrokeRectOp.rectsArr).
	ellipsesArr [1]diEllipseGeom
	wideColor   bool
	useScale    bool
}

// NewDIEllipseOp returns a DrawOp that fills or strokes an ellipse under an arbitrary view matrix, or nil if the
// ellipse/stroke combination isn't supported. The paint is consumed.
func NewDIEllipseOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, ellipse geom.Rect, strokeRec *stroke.Rec) DrawOp {
	center := geom.Point{
		X: (ellipse.Left + ellipse.Right) / 2,
		Y: (ellipse.Top + ellipse.Bottom) / 2,
	}
	xRadius := ellipse.Width() / 2
	yRadius := ellipse.Height() / 2

	recStyle := strokeRec.Style()
	var style diEllipseStyle
	switch recStyle {
	case stroke.StyleStroke:
		style = diEllipseStyleStroke
	case stroke.StyleHairline:
		style = diEllipseStyleHairline
	default:
		style = diEllipseStyleFill
	}

	innerXRadius := float32(0)
	innerYRadius := float32(0)
	if recStyle != stroke.StyleFill && recStyle != stroke.StyleHairline {
		strokeWidth := strokeRec.Width()
		if geom.ScalarNearlyZero(strokeWidth) {
			strokeWidth = 0.5
		} else {
			strokeWidth /= 2
		}

		// We only handle thick strokes for near-circular ellipses.
		if strokeWidth > 0.5 && (0.5*xRadius > yRadius || 0.5*yRadius > xRadius) {
			return nil
		}

		// We don't handle it if the curvature of the stroke is less than the curvature of the ellipse.
		if strokeWidth*(yRadius*yRadius) < (strokeWidth*strokeWidth)*xRadius {
			return nil
		}
		if strokeWidth*(xRadius*xRadius) < (strokeWidth*strokeWidth)*yRadius {
			return nil
		}

		// Set the inner radius (if needed).
		if recStyle == stroke.StyleStroke {
			innerXRadius = xRadius - strokeWidth
			innerYRadius = yRadius - strokeWidth
		}

		xRadius += strokeWidth
		yRadius += strokeWidth
	}

	const maxOvalRadius = 16384
	if !caps.ShaderCaps.FloatIs32Bits && (xRadius >= maxOvalRadius || yRadius >= maxOvalRadius) {
		return nil
	}

	if style == diEllipseStyleStroke && (innerXRadius <= 0 || innerYRadius <= 0) {
		style = diEllipseStyleFill
	}

	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := diEllipseOpPool.borrow()
	o.ellipses = bootstrapInstances(o.ellipses, o.ellipsesArr[:0])
	o.InitOp(diEllipseOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, nil,
		PipelineInputFlagNone)

	// This expands the outer rect so that after CTM we end up with a half-pixel border.
	a := viewMatrix.Get(geom.MScaleX)
	b := viewMatrix.Get(geom.MSkewX)
	c := viewMatrix.Get(geom.MSkewY)
	d := viewMatrix.Get(geom.MScaleY)
	geoDx := 1 / float32(math.Sqrt(float64(a*a+c*c)))
	geoDy := 1 / float32(math.Sqrt(float64(b*b+d*d)))

	o.ellipses = append(o.ellipses, diEllipseGeom{
		viewMatrix:   *viewMatrix,
		color:        color,
		xRadius:      xRadius,
		yRadius:      yRadius,
		innerXRadius: innerXRadius,
		innerYRadius: innerYRadius,
		geoDx:        geoDx,
		geoDy:        geoDy,
		style:        style,
		bounds: geom.Rect{
			Left: center.X - xRadius, Top: center.Y - yRadius,
			Right: center.X + xRadius, Bottom: center.Y + yRadius,
		},
	})
	devBounds, _ := viewMatrix.MapRect(o.ellipses[0].bounds)
	o.SetBounds(devBounds, HasAABloat(true), IsHairline(false))
	return o
}

func (o *diEllipseOp) viewMatrix() *geom.Matrix { return &o.ellipses[0].viewMatrix }
func (o *diEllipseOp) style() diEllipseStyle    { return o.ellipses[0].style }

// Name implements Op.
func (o *diEllipseOp) Name() string { return "DIEllipseOp" }

// recycle implements Op: returns this dead op to its free list.
func (o *diEllipseOp) recycle() {
	recycleProgramInfo(o.programInfo)
	recycleKeepingBacking(&diEllipseOpPool, o,
		func(o *diEllipseOp) []diEllipseGeom { return o.ellipses },
		func(o *diEllipseOp, s []diEllipseGeom) { o.ellipses = s })
}

// VisitProxies implements Op.
func (o *diEllipseOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *diEllipseOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *diEllipseOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *diEllipseOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	o.useScale = !caps.ShaderCaps.FloatIs32Bits && !caps.ShaderCaps.HasLowFragmentPrecision
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.ellipses[0].color, &o.wideColor)
}

// createProgramInfo builds the geometry processor and program info for this device-independent ellipse op.
func (o *diEllipseOp) createProgramInfo(state *OpFlushState) {
	gp := makeDIEllipseGeometryProcessor(o.wideColor, o.useScale, o.viewMatrix(), o.style())
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op: it builds the vertex buffer for the batched device-independent ellipses.
func (o *diEllipseOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	stride := o.programInfo.GeomProc().gpBase().VertexStride()
	vertices := makeQuadHelper(state, uint64(stride), len(o.ellipses), &o.mesh)
	if vertices == nil {
		return
	}
	w := quadVertexWriter{buf: vertices}

	// On MSAA, bloat enough to guarantee any pixel that might be touched by the ellipse has full sample coverage.
	aaBloat := float32(0.5)
	if state.OpArgs().UsesMSAASurface() {
		aaBloat = math.Sqrt2
	}

	for e := range o.ellipses {
		ellipse := &o.ellipses[e]
		xRadius := ellipse.xRadius
		yRadius := ellipse.yRadius

		drawBounds := ellipse.bounds.Outset(ellipse.geoDx*aaBloat, ellipse.geoDy*aaBloat)

		// Normalize the "outer radius" coordinates within drawBounds so that the outer edge occurs at x^2 + y^2 == 1.
		outerCoordX := drawBounds.Width() / (xRadius * 2)
		outerCoordY := drawBounds.Height() / (yRadius * 2)

		// By default, constructed so that the inner coord is (0, 0) for all points.
		innerCoordX := float32(0)
		innerCoordY := float32(0)

		// ... unless we're stroked. Then normalize the "inner radius" coordinates within drawBounds so that the inner
		// edge occurs at x2^2 + y2^2 == 1.
		if o.style() == diEllipseStyleStroke {
			innerCoordX = drawBounds.Width() / (ellipse.innerXRadius * 2)
			innerCoordY = drawBounds.Height() / (ellipse.innerYRadius * 2)
		}

		for corner := range 4 {
			x := drawBounds.Left
			outerX := -outerCoordX
			innerX := -innerCoordX
			if corner >= 2 {
				x = drawBounds.Right
				outerX = outerCoordX
				innerX = innerCoordX
			}
			y := drawBounds.Top
			outerY := -outerCoordY
			innerY := -innerCoordY
			if corner&1 != 0 {
				y = drawBounds.Bottom
				outerY = outerCoordY
				innerY = innerCoordY
			}
			w.putF32(x)
			w.putF32(y)
			w.putColor(ellipse.color, o.wideColor)
			w.putF32(outerX)
			w.putF32(outerY)
			if o.useScale {
				w.putF32(max(xRadius, yRadius))
			}
			w.putF32(innerX)
			w.putF32(innerY)
		}
	}
}

// OnExecute implements Op: it binds the program and mesh and issues the draw for the batched device-independent
// ellipses.
func (o *diEllipseOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.programInfo == nil || !o.mesh.isValid() {
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
	o.mesh.draw(renderPass)
}

// OnCombineIfPossible implements Op: it merges o with another diEllipseOp when their processors, stroke style, and view
// matrix match.
func (o *diEllipseOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*diEllipseOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}
	if o.style() != that.style() {
		return CombineResultCannotCombine
	}
	if !o.viewMatrix().CheapEqual(that.viewMatrix()) {
		return CombineResultCannotCombine
	}
	o.ellipses = append(o.ellipses, that.ellipses...)
	o.wideColor = o.wideColor || that.wideColor
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// RRect geometry constants.

// In the case of a normal fill or a stroke, we draw the roundrect as a 9-patch; for strokes we don't draw the center
// quad. For circular roundrects, in the overstroke case (stroke width greater than twice the corner radius), additional
// geometry marks out the rectangle in the center; the shared vertices are duplicated so we can set a different outer
// radius for the fill calculation.
var overstrokeRRectIndices = []uint16{
	// overstroke quads we place this at the beginning so that we can skip these indices when rendering normally
	16, 17, 19, 16, 19, 18,
	19, 17, 23, 19, 23, 21,
	21, 23, 22, 21, 22, 20,
	22, 16, 18, 22, 18, 20,

	// corners
	0, 1, 5, 0, 5, 4,
	2, 3, 7, 2, 7, 6,
	8, 9, 13, 8, 13, 12,
	10, 11, 15, 10, 15, 14,

	// edges
	1, 2, 6, 1, 6, 5,
	4, 5, 9, 4, 9, 8,
	6, 7, 11, 6, 11, 10,
	9, 10, 14, 9, 14, 13,

	// center we place this at the end so that we can ignore these indices when not rendering as filled
	5, 6, 10, 5, 10, 9,
}

// Fill and standard stroke indices skip the overstroke "ring".
var standardRRectIndices = overstrokeRRectIndices[6*4:]

// Overstroke count is the array size minus the center indices; fill count skips the overstroke indices and includes the
// center; stroke count is the fill count minus the center indices.
var (
	indicesPerOverstrokeRRect = len(overstrokeRRectIndices) - 6
	indicesPerFillRRect       = indicesPerOverstrokeRRect - 6*4 + 6
	indicesPerStrokeRRect     = indicesPerFillRRect - 6
)

const (
	vertsPerStandardRRect   = 16
	vertsPerOverstrokeRRect = 24
)

// rrectGeomType distinguishes the different rounded-rect corner/stroke configurations a rrect op can draw.
type rrectGeomType int8

const (
	rrectGeomFill rrectGeomType = iota
	rrectGeomStroke
	rrectGeomOverstroke
)

func rrectTypeToVertCount(t rrectGeomType) int {
	if t == rrectGeomOverstroke {
		return vertsPerOverstrokeRRect
	}
	return vertsPerStandardRRect
}

func rrectTypeToIndexCount(t rrectGeomType) int {
	switch t {
	case rrectGeomFill:
		return indicesPerFillRRect
	case rrectGeomStroke:
		return indicesPerStrokeRRect
	default:
		return indicesPerOverstrokeRRect
	}
}

func rrectTypeToIndices(t rrectGeomType) []uint16 {
	switch t {
	case rrectGeomFill:
		// The full nine-patch, including the center quad (the last 6 indices).
		return standardRRectIndices[:indicesPerFillRRect]
	case rrectGeomStroke:
		// Strokes skip the center quad, so drop the trailing 6 indices; this must match the index-buffer size from
		// rrectTypeToIndexCount or OnPrepare overruns the allocation.
		return standardRRectIndices[:indicesPerStrokeRRect]
	default:
		return overstrokeRRectIndices[:indicesPerOverstrokeRRect]
	}
}

const numRRectsInIndexBuffer = 256

var (
	rrectIndexBufferKeyOnce sync.Once
	strokeRRectIndexKey     gpu.UniqueKey
	fillRRectOnlyIndexKey   gpu.UniqueKey
)

// getRRectIndexBuffer returns the cached shared index buffer for the given rrect geometry type, creating and keying it
// on first use.
func getRRectIndexBuffer(t rrectGeomType, rp *ResourceProvider) *Buffer {
	rrectIndexBufferKeyOnce.Do(func() {
		builder := gpu.UniqueKeyBuilder(&strokeRRectIndexKey, gpu.GenerateUniqueKeyDomain(), 1,
			"StrokeRRectOnlyIndexBuffer")
		builder.Slice()[0] = 1
		builder.Finish()
		builder = gpu.UniqueKeyBuilder(&fillRRectOnlyIndexKey, gpu.GenerateUniqueKeyDomain(), 1,
			"RRectOnlyIndexBuffer")
		builder.Slice()[0] = 1
		builder.Finish()
	})
	switch t {
	case rrectGeomFill:
		return rp.findOrCreatePatternedIndexBuffer(standardRRectIndices, indicesPerFillRRect,
			numRRectsInIndexBuffer, vertsPerStandardRRect, &fillRRectOnlyIndexKey)
	case rrectGeomStroke:
		return rp.findOrCreatePatternedIndexBuffer(standardRRectIndices, indicesPerStrokeRRect,
			numRRectsInIndexBuffer, vertsPerStandardRRect, &strokeRRectIndexKey)
	default:
		panic("no patterned index buffer for overstroke rrects")
	}
}

//////////////////////////////////////////////////////////////////////////////
// CircularRRectOp

var circularRRectOpClassID = GenOpClassID()

// circularRRectOpPool is the free list of circularRRectOp shells (see oppool.go).
var circularRRectOpPool opPool[circularRRectOp]

// circularRRectGeom holds one circular-cornered rounded-rect instance's per-draw geometry and color.
type circularRRectGeom struct {
	color       colorcore.PMColor4f
	innerRadius float32
	outerRadius float32
	devBounds   geom.Rect
	typ         rrectGeomType
}

// circularRRectOp draws one or more rounded rects whose corners are circular (equal x/y radius), batched together when
// compatible.
type circularRRectOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	rrects      []circularRRectGeom
	OpBase
	mesh       simpleMesh
	vertCount  int
	indexCount int
	// rrectsArr backs rrects for the common single-instance case so a freshly constructed op does not heap-allocate a
	// separate backing array (see the aaStrokeRectOp.rectsArr note). Combining grows past it into the heap as normal.
	rrectsArr                    [1]circularRRectGeom
	viewMatrixIfUsingLocalCoords geom.Matrix
	allFill                      bool
	wideColor                    bool
}

// newCircularRRectOp is the CircularRRectOp constructor (Make is trivial). A devStrokeWidth <= 0 indicates fill only;
// otherwise strokeOnly indicates whether the rrect is only stroked or stroked and filled. The paint is consumed.
func newCircularRRectOp(paint *Paint, viewMatrix *geom.Matrix, devRect geom.Rect, devRadius, devStrokeWidth float32, strokeOnly bool) *circularRRectOp {
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := circularRRectOpPool.borrow()
	o.viewMatrixIfUsingLocalCoords = *viewMatrix
	o.rrects = bootstrapInstances(o.rrects, o.rrectsArr[:0])
	o.InitOp(circularRRectOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, nil,
		PipelineInputFlagNone)

	bounds := devRect
	if devStrokeWidth <= 0 && strokeOnly {
		panic("stroke-only requires a positive stroke width")
	}
	innerRadius := float32(0)
	outerRadius := devRadius
	halfWidth := float32(0)
	typ := rrectGeomFill
	if devStrokeWidth > 0 {
		if geom.ScalarNearlyZero(devStrokeWidth) {
			halfWidth = 0.5
		} else {
			halfWidth = devStrokeWidth / 2
		}

		if strokeOnly {
			// Outset stroke by 1/4 pixel.
			devStrokeWidth += 0.25
			// If the stroke is greater than the width or height, this is still a fill; otherwise we compute stroke
			// params.
			if devStrokeWidth <= devRect.Width() && devStrokeWidth <= devRect.Height() {
				innerRadius = devRadius - halfWidth
				if innerRadius >= 0 {
					typ = rrectGeomStroke
				} else {
					typ = rrectGeomOverstroke
				}
			}
		}
		outerRadius += halfWidth
		bounds = bounds.Outset(halfWidth, halfWidth)
	}

	// The radii are outset for two reasons. First, it allows the shader to simply perform simpler computation because
	// the computed alpha is zero, rather than 50%, at the radius. Second, the outer radius is used to compute the verts
	// of the bounding box that is rendered and the outset ensures the box will cover all pixels partially covered by
	// the rrect corners.
	outerRadius += 0.5
	innerRadius -= 0.5

	o.SetBounds(bounds, HasAABloat(true), IsHairline(false))

	// Expand the rect for aa to generate correct vertices.
	bounds = bounds.Outset(0.5, 0.5)

	o.rrects = append(o.rrects, circularRRectGeom{
		color:       color,
		innerRadius: innerRadius,
		outerRadius: outerRadius,
		devBounds:   bounds,
		typ:         typ,
	})
	o.vertCount = rrectTypeToVertCount(typ)
	o.indexCount = rrectTypeToIndexCount(typ)
	o.allFill = typ == rrectGeomFill
	return o
}

// Name implements Op.
func (o *circularRRectOp) Name() string { return "CircularRRectOp" }

// recycle implements Op: returns this dead op to its free list.
func (o *circularRRectOp) recycle() {
	recycleProgramInfo(o.programInfo)
	recycleKeepingBacking(&circularRRectOpPool, o,
		func(o *circularRRectOp) []circularRRectGeom { return o.rrects },
		func(o *circularRRectOp, s []circularRRectGeom) { o.rrects = s })
}

// VisitProxies implements Op.
func (o *circularRRectOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *circularRRectOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *circularRRectOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *circularRRectOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.rrects[0].color, &o.wideColor)
}

// fillInOverstrokeVerts writes the extra ring of vertices needed when a stroke is wide enough that its inner edge
// overlaps itself.
func fillInOverstrokeVerts(w *quadVertexWriter, bounds geom.Rect, smInset, bigInset, xOffset, outerRadius, innerRadius float32, color colorcore.PMColor4f, wideColor bool) {
	if smInset >= bigInset {
		panic("overstroke insets out of order")
	}
	put := func(x, y, ox float32) {
		w.putF32(x)
		w.putF32(y)
		w.putColor(color, wideColor)
		w.putF32(ox)
		w.putF32(0)
		w.putF32(outerRadius)
		w.putF32(innerRadius)
	}
	// TL
	put(bounds.Left+smInset, bounds.Top+smInset, xOffset)
	// TR
	put(bounds.Right-smInset, bounds.Top+smInset, xOffset)
	put(bounds.Left+bigInset, bounds.Top+bigInset, 0)
	put(bounds.Right-bigInset, bounds.Top+bigInset, 0)
	put(bounds.Left+bigInset, bounds.Bottom-bigInset, 0)
	put(bounds.Right-bigInset, bounds.Bottom-bigInset, 0)
	// BL
	put(bounds.Left+smInset, bounds.Bottom-smInset, xOffset)
	// BR
	put(bounds.Right-smInset, bounds.Bottom-smInset, xOffset)
}

// createProgramInfo builds the geometry processor and program info for this circular rrect op.
func (o *circularRRectOp) createProgramInfo(state *OpFlushState) {
	args := state.OpArgs()
	if args.UsesMSAASurface() {
		panic("circular rrect op is coverage-AA only")
	}
	// Invert the view matrix as a local matrix (if any other processors require coords).
	localMatrix, ok := o.viewMatrixIfUsingLocalCoords.Invert()
	if !ok {
		return
	}
	gp := makeCircleGeometryProcessor(!o.allFill, false, false, false, false, o.wideColor,
		&localMatrix)
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op: it builds the vertex and index buffers for the batched circular rrects.
func (o *circularRRectOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	stride := o.programInfo.GeomProc().gpBase().VertexStride()
	vertices, vertexBuffer, firstVertex := state.MakeVertexSpace(uint64(stride), o.vertCount)
	if vertices == nil {
		return
	}
	indices, indexBuffer, firstIndex := state.MakeIndexSpace(o.indexCount)
	if indices == nil {
		return
	}

	w := quadVertexWriter{buf: vertices}
	idxOffset := 0
	currStartVertex := 0
	for r := range o.rrects {
		rrect := &o.rrects[r]
		outerRadius := rrect.outerRadius
		bounds := rrect.devBounds

		yCoords := [4]float32{
			bounds.Top, bounds.Top + outerRadius,
			bounds.Bottom - outerRadius, bounds.Bottom,
		}
		yOuterRadii := [4]float32{-1, 0, 0, 1}
		// The inner radius in the vertex data must be specified in normalized space. For fills, specifying
		// -1/outerRadius guarantees an alpha of 1.0 at the inner radius.
		var innerRadius float32
		if rrect.typ != rrectGeomFill {
			innerRadius = rrect.innerRadius / rrect.outerRadius
		} else {
			innerRadius = -1.0 / rrect.outerRadius
		}
		for i := range 4 {
			put := func(x, ox float32) {
				w.putF32(x)
				w.putF32(yCoords[i])
				w.putColor(rrect.color, o.wideColor)
				w.putF32(ox)
				w.putF32(yOuterRadii[i])
				w.putF32(outerRadius)
				w.putF32(innerRadius)
			}
			put(bounds.Left, -1)
			put(bounds.Left+outerRadius, 0)
			put(bounds.Right-outerRadius, 0)
			put(bounds.Right, 1)
		}
		// Add the additional vertices for overstroked rrects. Effectively this is an additional stroked rrect, with its
		// outer radius = outerRadius - innerRadius, and inner radius = 0. This will give us correct AA in the center
		// and the correct distance to the outer edge. Also, the outer offset is a constant vector pointing to the
		// right, which guarantees that the distance value along the outer rectangle is constant.
		if rrect.typ == rrectGeomOverstroke {
			if rrect.innerRadius > 0 {
				panic("overstroke with positive inner radius")
			}
			overstrokeOuterRadius := outerRadius - rrect.innerRadius
			// This is the normalized distance from the outer rectangle of this geometry to the outer edge.
			maxOffset := -rrect.innerRadius / overstrokeOuterRadius
			fillInOverstrokeVerts(&w, bounds, outerRadius, overstrokeOuterRadius, maxOffset,
				overstrokeOuterRadius, 0, rrect.color, o.wideColor)
		}

		idxOffset = putU16Indices(indices, idxOffset, rrectTypeToIndices(rrect.typ),
			currStartVertex)
		currStartVertex += rrectTypeToVertCount(rrect.typ)
	}

	o.mesh.setIndexed(indexBuffer, o.indexCount, firstIndex, 0, uint16(o.vertCount-1),
		vertexBuffer, firstVertex)
}

// OnExecute implements Op: it binds the program and mesh and issues the draw for the batched circular rrects.
func (o *circularRRectOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.programInfo == nil || !o.mesh.isValid() {
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
	o.mesh.draw(renderPass)
}

// OnCombineIfPossible implements Op: it merges o with another circularRRectOp when their processors, bounds, and vertex
// budget are compatible.
func (o *circularRRectOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*circularRRectOp)
	if !ok {
		return CombineResultCannotCombine
	}

	// Cannot combine if the net number of indices would overflow int32, or if the net number of vertices would overflow
	// uint16 (since the index values are 16-bit that point into the vertex buffer).
	if o.indexCount > math.MaxInt32-that.indexCount ||
		o.vertCount > math.MaxUint16-that.vertCount {
		return CombineResultCannotCombine
	}

	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}

	if o.helper.UsesLocalCoords() &&
		!o.viewMatrixIfUsingLocalCoords.CheapEqual(&that.viewMatrixIfUsingLocalCoords) {
		return CombineResultCannotCombine
	}

	o.rrects = append(o.rrects, that.rrects...)
	o.vertCount += that.vertCount
	o.indexCount += that.indexCount
	o.allFill = o.allFill && that.allFill
	o.wideColor = o.wideColor || that.wideColor
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// EllipticalRRectOp

var ellipticalRRectOpClassID = GenOpClassID()

// ellipticalRRectOpPool is the free list of ellipticalRRectOp shells (see oppool.go).
var ellipticalRRectOpPool opPool[ellipticalRRectOp]

// ellipticalRRectGeom holds one elliptical-cornered rounded-rect instance's per-draw geometry and color.
type ellipticalRRectGeom struct {
	color        colorcore.PMColor4f
	xRadius      float32
	yRadius      float32
	innerXRadius float32
	innerYRadius float32
	devBounds    geom.Rect
}

// ellipticalRRectOp draws one or more rounded rects with elliptical corners, batched together when compatible.
type ellipticalRRectOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	rrects      []ellipticalRRectGeom
	OpBase
	mesh simpleMesh
	// rrectsArr backs rrects for the common single-instance case (see aaStrokeRectOp.rectsArr).
	rrectsArr                    [1]ellipticalRRectGeom
	viewMatrixIfUsingLocalCoords geom.Matrix
	stroked                      bool
	wideColor                    bool
	useScale                     bool
}

// newEllipticalRRectOp returns an op that fills or strokes an elliptical-cornered rounded rect. If devStrokeWidths
// values are <= 0 then fill only; otherwise strokeOnly indicates whether the rrect is only stroked or stroked and
// filled. The paint is consumed.
func newEllipticalRRectOp(paint *Paint, viewMatrix *geom.Matrix, devRect geom.Rect, devXRadius, devYRadius float32, devStrokeWidths geom.Point, strokeOnly bool) DrawOp {
	if (devXRadius < 0.5 || devYRadius < 0.5) && !strokeOnly {
		panic("elliptical rrect radii must be >= 0.5 for fills")
	}
	if (devStrokeWidths.X > 0) != (devStrokeWidths.Y > 0) {
		panic("stroke widths must agree in sign")
	}
	if strokeOnly && devStrokeWidths.X <= 0 {
		panic("stroke-only requires positive stroke widths")
	}
	if devStrokeWidths.X > 0 {
		strokeLen := float32(math.Hypot(float64(devStrokeWidths.X), float64(devStrokeWidths.Y)))
		if geom.ScalarNearlyZero(strokeLen) {
			devStrokeWidths = geom.Point{X: 0.5, Y: 0.5}
		} else {
			devStrokeWidths = geom.Point{X: devStrokeWidths.X / 2, Y: devStrokeWidths.Y / 2}
		}

		// We only handle thick strokes for near-circular ellipses.
		if float32(math.Hypot(float64(devStrokeWidths.X), float64(devStrokeWidths.Y))) > 0.5 &&
			(0.5*devXRadius > devYRadius || 0.5*devYRadius > devXRadius) {
			return nil
		}

		// We don't handle it if the curvature of the stroke is less than the curvature of the ellipse.
		if devStrokeWidths.X*(devYRadius*devYRadius) <
			(devStrokeWidths.Y*devStrokeWidths.Y)*devXRadius {
			return nil
		}
		if devStrokeWidths.Y*(devXRadius*devXRadius) <
			(devStrokeWidths.X*devStrokeWidths.X)*devYRadius {
			return nil
		}
	}

	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := ellipticalRRectOpPool.borrow()
	o.rrects = bootstrapInstances(o.rrects, o.rrectsArr[:0])
	o.InitOp(ellipticalRRectOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, nil,
		PipelineInputFlagNone)

	innerXRadius := float32(0)
	innerYRadius := float32(0)
	bounds := devRect
	stroked := false
	if devStrokeWidths.X > 0 {
		// This is legit only if scale & translation (which should be the case at the moment).
		if strokeOnly {
			innerXRadius = devXRadius - devStrokeWidths.X
			innerYRadius = devYRadius - devStrokeWidths.Y
			stroked = innerXRadius >= 0 && innerYRadius >= 0
		}
		devXRadius += devStrokeWidths.X
		devYRadius += devStrokeWidths.Y
		bounds = bounds.Outset(devStrokeWidths.X, devStrokeWidths.Y)
	}

	o.stroked = stroked
	o.viewMatrixIfUsingLocalCoords = *viewMatrix
	o.SetBounds(bounds, HasAABloat(true), IsHairline(false))
	o.rrects = append(o.rrects, ellipticalRRectGeom{
		color:        color,
		xRadius:      devXRadius,
		yRadius:      devYRadius,
		innerXRadius: innerXRadius,
		innerYRadius: innerYRadius,
		devBounds:    bounds,
	})
	return o
}

// Name implements Op.
func (o *ellipticalRRectOp) Name() string { return "EllipticalRRectOp" }

// recycle implements Op: returns this dead op to its free list.
func (o *ellipticalRRectOp) recycle() {
	recycleProgramInfo(o.programInfo)
	recycleKeepingBacking(&ellipticalRRectOpPool, o,
		func(o *ellipticalRRectOp) []ellipticalRRectGeom { return o.rrects },
		func(o *ellipticalRRectOp, s []ellipticalRRectGeom) { o.rrects = s })
}

// VisitProxies implements Op.
func (o *ellipticalRRectOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *ellipticalRRectOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *ellipticalRRectOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *ellipticalRRectOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	o.useScale = !caps.ShaderCaps.FloatIs32Bits
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.rrects[0].color, &o.wideColor)
}

// createProgramInfo builds the geometry processor and program info for this elliptical rrect op.
func (o *ellipticalRRectOp) createProgramInfo(state *OpFlushState) {
	localMatrix, ok := o.viewMatrixIfUsingLocalCoords.Invert()
	if !ok {
		return
	}
	gp := makeEllipseGeometryProcessor(o.stroked, o.wideColor, o.useScale, &localMatrix)
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op: it builds the vertex buffer for the batched elliptical rrects using the shared rrect index
// buffer.
func (o *ellipticalRRectOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	// Drop out the middle quad if we're stroked.
	indicesPerInstance := indicesPerFillRRect
	geomType := rrectGeomFill
	if o.stroked {
		indicesPerInstance = indicesPerStrokeRRect
		geomType = rrectGeomStroke
	}
	indexBuffer := getRRectIndexBuffer(geomType, state.ResourceProvider())
	if indexBuffer == nil {
		return
	}
	stride := o.programInfo.GeomProc().gpBase().VertexStride()
	vertices := makePatternHelper(state, uint64(stride), indexBuffer, vertsPerStandardRRect,
		indicesPerInstance, len(o.rrects), numRRectsInIndexBuffer, &o.mesh)
	if vertices == nil {
		return
	}
	w := quadVertexWriter{buf: vertices}

	for r := range o.rrects {
		rrect := &o.rrects[r]
		// Compute the reciprocals of the radii here to save time in the shader.
		reciprocalRadii := [4]float32{
			1 / rrect.xRadius,
			1 / rrect.yRadius,
			1 / rrect.innerXRadius,
			1 / rrect.innerYRadius,
		}

		// If the stroke width is exactly double the radius, the inner radii will be zero. Pin to a large value, to
		// avoid infinities in the shader.
		reciprocalRadii[2] = min(reciprocalRadii[2], 1e6)
		reciprocalRadii[3] = min(reciprocalRadii[3], 1e6)

		// On MSAA, bloat enough to guarantee any pixel that might be touched by the rrect has full sample coverage.
		aaBloat := float32(0.5)
		if state.OpArgs().UsesMSAASurface() {
			aaBloat = math.Sqrt2
		}

		// Extend out the radii to antialias.
		xOuterRadius := rrect.xRadius + aaBloat
		yOuterRadius := rrect.yRadius + aaBloat

		xMaxOffset := xOuterRadius
		yMaxOffset := yOuterRadius
		if !o.stroked {
			// For filled rrects we map a unit circle in the vertex attributes rather than computing an ellipse and
			// modifying that distance, so we normalize to 1.
			xMaxOffset /= rrect.xRadius
			yMaxOffset /= rrect.yRadius
		}

		bounds := rrect.devBounds.Outset(aaBloat, aaBloat)

		yCoords := [4]float32{
			bounds.Top, bounds.Top + yOuterRadius,
			bounds.Bottom - yOuterRadius, bounds.Bottom,
		}
		yOuterOffsets := [4]float32{
			yMaxOffset,
			geom.ScalarNearlyZeroTol, // we're using inversesqrt() in the shader, so can't be
			geom.ScalarNearlyZeroTol, // exactly 0
			yMaxOffset,
		}

		for i := range 4 {
			put := func(x, ox float32) {
				w.putF32(x)
				w.putF32(yCoords[i])
				w.putColor(rrect.color, o.wideColor)
				w.putF32(ox)
				w.putF32(yOuterOffsets[i])
				if o.useScale {
					w.putF32(max(rrect.xRadius, rrect.yRadius))
				}
				w.putF32(reciprocalRadii[0])
				w.putF32(reciprocalRadii[1])
				w.putF32(reciprocalRadii[2])
				w.putF32(reciprocalRadii[3])
			}
			put(bounds.Left, xMaxOffset)
			put(bounds.Left+xOuterRadius, geom.ScalarNearlyZeroTol)
			put(bounds.Right-xOuterRadius, geom.ScalarNearlyZeroTol)
			put(bounds.Right, xMaxOffset)
		}
	}
}

// OnExecute implements Op: it binds the program and mesh and issues the draw for the batched elliptical rrects.
func (o *ellipticalRRectOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.programInfo == nil || !o.mesh.isValid() {
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
	o.mesh.draw(renderPass)
}

// OnCombineIfPossible implements Op: it merges o with another ellipticalRRectOp when their processors, stroke style,
// and (if using local coords) view matrix match.
func (o *ellipticalRRectOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*ellipticalRRectOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}
	if o.stroked != that.stroked {
		return CombineResultCannotCombine
	}
	if o.helper.UsesLocalCoords() &&
		!o.viewMatrixIfUsingLocalCoords.CheapEqual(&that.viewMatrixIfUsingLocalCoords) {
		return CombineResultCannotCombine
	}
	o.rrects = append(o.rrects, that.rrects...)
	o.wideColor = o.wideColor || that.wideColor
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// Factory functions: entry points that pick the right op for a given oval/rrect draw.

// MakeCircularRRectOp builds a draw op for a round rect with equal x/y corner radii. The paint is consumed. Returns nil
// when the op cannot represent the draw.
func MakeCircularRRectOp(paint *Paint, viewMatrix *geom.Matrix, rrect geom.RRect, strokeRec *stroke.Rec) DrawOp {
	if !viewMatrix.RectStaysRect() || !viewMatrix.IsSimilarity() {
		panic("circular rrect op requires a rect-stays-rect similarity matrix")
	}
	if rrect.Type != geom.RRectSimple {
		panic("circular rrect op requires a simple rrect")
	}
	if rrect.RadiusX != rrect.RadiusY {
		panic("circular rrect op requires circular corners")
	}

	// RRect ops only handle simple, but not too simple, rrects. Do any matrix crunching before we reset the draw state
	// for device coords.
	bounds, _ := viewMatrix.MapRect(rrect.Rect)

	radius := rrect.RadiusX
	scaledRadius := geom.ScalarAbs(radius * (viewMatrix.Get(geom.MScaleX) +
		viewMatrix.Get(geom.MSkewY)))

	// Do mapping of stroke. Use -1 to indicate fill-only draws.
	scaledStroke := float32(-1)
	strokeWidth := strokeRec.Width()
	style := strokeRec.Style()

	isStrokeOnly := style == stroke.StyleStroke || style == stroke.StyleHairline
	hasStroke := isStrokeOnly || style == stroke.StyleStrokeAndFill

	if hasStroke {
		if style == stroke.StyleHairline {
			scaledStroke = 1
		} else {
			scaledStroke = geom.ScalarAbs(strokeWidth * (viewMatrix.Get(geom.MScaleX) +
				viewMatrix.Get(geom.MSkewY)))
		}
	}

	// The way the effect interpolates the offset-to-ellipse/circle-center attribute only works on the interior of the
	// rrect if the radii are >= 0.5. Otherwise, the inner rect of the nine-patch will have fractional coverage. This
	// only matters when the interior is actually filled. We could consider falling back to rect rendering here, since a
	// tiny radius is indistinguishable from a square corner.
	if !isStrokeOnly && scaledRadius < 0.5 {
		return nil
	}

	return newCircularRRectOp(paint, viewMatrix, bounds, scaledRadius, scaledStroke,
		isStrokeOnly)
}

// makeRRectOpInternal builds the DrawOp for a single stroked or filled rrect, choosing between the circular and
// elliptical corner op implementations.
func makeRRectOpInternal(paint *Paint, viewMatrix *geom.Matrix, rrect geom.RRect, strokeRec *stroke.Rec) DrawOp {
	if !viewMatrix.RectStaysRect() {
		panic("rrect op requires a rect-stays-rect matrix")
	}
	if rrect.Type != geom.RRectSimple {
		panic("rrect op requires a simple rrect")
	}

	// RRect ops only handle simple, but not too simple, rrects. Do any matrix crunching before we reset the draw state
	// for device coords.
	bounds, _ := viewMatrix.MapRect(rrect.Rect)

	xRadius := geom.ScalarAbs(viewMatrix.Get(geom.MScaleX)*rrect.RadiusX +
		viewMatrix.Get(geom.MSkewY)*rrect.RadiusY)
	yRadius := geom.ScalarAbs(viewMatrix.Get(geom.MSkewX)*rrect.RadiusX +
		viewMatrix.Get(geom.MScaleY)*rrect.RadiusY)

	style := strokeRec.Style()

	// Do (potentially) anisotropic mapping of stroke. Use -1s to indicate fill-only draws.
	scaledStroke := geom.Point{X: -1, Y: -1}
	strokeWidth := strokeRec.Width()

	isStrokeOnly := style == stroke.StyleStroke || style == stroke.StyleHairline
	hasStroke := isStrokeOnly || style == stroke.StyleStrokeAndFill

	if hasStroke {
		if style == stroke.StyleHairline {
			scaledStroke = geom.Point{X: 1, Y: 1}
		} else {
			scaledStroke.X = geom.ScalarAbs(strokeWidth * (viewMatrix.Get(geom.MScaleX) +
				viewMatrix.Get(geom.MSkewY)))
			scaledStroke.Y = geom.ScalarAbs(strokeWidth * (viewMatrix.Get(geom.MSkewX) +
				viewMatrix.Get(geom.MScaleY)))
		}

		// If half of the stroke width is greater than the radius, we don't handle that right now.
		if 0.5*scaledStroke.X > xRadius || 0.5*scaledStroke.Y > yRadius {
			return nil
		}
	}

	// The matrix may have a rotation by an odd multiple of 90 degrees.
	if viewMatrix.Get(geom.MScaleX) == 0 {
		xRadius, yRadius = yRadius, xRadius
		scaledStroke.X, scaledStroke.Y = scaledStroke.Y, scaledStroke.X
	}

	// The way the effect interpolates the offset-to-ellipse/circle-center attribute only works on the interior of the
	// rrect if the radii are >= 0.5 (see MakeCircularRRectOp).
	if !isStrokeOnly && (xRadius < 0.5 || yRadius < 0.5) {
		return nil
	}

	// If the corners are circles, use the circle renderer.
	return newEllipticalRRectOp(paint, viewMatrix, bounds, xRadius, yRadius, scaledStroke,
		isStrokeOnly)
}

// MakeRRectOp builds a draw op for a round rect, dispatching to the oval op for a full ellipse or to the
// circular/elliptical rrect op otherwise. The paint is consumed.
func MakeRRectOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, rrect geom.RRect, strokeRec *stroke.Rec) DrawOp {
	if rrect.Type == geom.RRectOval {
		return MakeOvalOp(caps, paint, viewMatrix, rrect.Rect, strokeRec)
	}

	if !viewMatrix.RectStaysRect() || rrect.Type != geom.RRectSimple {
		return nil
	}

	return makeRRectOpInternal(paint, viewMatrix, rrect, strokeRec)
}

// MakeCircleOp builds a draw op for a circle. Dashed strokes are unreachable here since this library's style carries no
// path effects. The paint is consumed.
func MakeCircleOp(paint *Paint, viewMatrix *geom.Matrix, oval geom.Rect, strokeRec *stroke.Rec) DrawOp {
	width := oval.Width()
	if width <= geom.ScalarNearlyZeroTol || !geom.ScalarNearlyEqual(width, oval.Height()) ||
		!circleStaysCircle(viewMatrix) {
		panic("circle op preconditions not met")
	}
	center := geom.Point{X: (oval.Left + oval.Right) / 2, Y: (oval.Top + oval.Bottom) / 2}
	return NewCircleOp(paint, viewMatrix, center, width/2, strokeRec, nil)
}

// MakeOvalOp builds a draw op for an axis-aligned oval, preferring the device-space ellipse op and falling back to the
// device-independent one when shader derivatives are available. The paint is consumed.
func MakeOvalOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, oval geom.Rect, strokeRec *stroke.Rec) DrawOp {
	// Prefer the device space ellipse op for batchability.
	if viewMatrix.RectStaysRect() {
		return NewEllipseOp(caps, paint, viewMatrix, oval, strokeRec)
	}

	// Otherwise, if we have shader derivative support, render as device-independent.
	if caps.ShaderCaps.ShaderDerivativeSupport {
		a := viewMatrix.Get(geom.MScaleX)
		b := viewMatrix.Get(geom.MSkewX)
		c := viewMatrix.Get(geom.MSkewY)
		d := viewMatrix.Get(geom.MScaleY)
		// Check for a near-degenerate matrix.
		if a*a+c*c > geom.ScalarNearlyZeroTol && b*b+d*d > geom.ScalarNearlyZeroTol {
			return NewDIEllipseOp(caps, paint, viewMatrix, oval, strokeRec)
		}
	}

	return nil
}

// MakeArcOp builds a draw op for a circular arc. The paint is consumed; angles are in degrees.
func MakeArcOp(paint *Paint, viewMatrix *geom.Matrix, oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, strokeRec *stroke.Rec) DrawOp {
	if oval.IsEmpty() || sweepAngle == 0 {
		panic("arc op preconditions not met")
	}
	width := oval.Width()
	if geom.ScalarAbs(sweepAngle) >= 360 {
		return nil
	}
	if !geom.ScalarNearlyEqual(width, oval.Height()) || !circleStaysCircle(viewMatrix) {
		return nil
	}
	center := geom.Point{X: (oval.Left + oval.Right) / 2, Y: (oval.Top + oval.Bottom) / 2}
	arcParams := ArcParams{
		StartAngleRadians: startAngle * math.Pi / 180,
		SweepAngleRadians: sweepAngle * math.Pi / 180,
		UseCenter:         useCenter,
	}
	return NewCircleOp(paint, viewMatrix, center, width/2, strokeRec, &arcParams)
}
