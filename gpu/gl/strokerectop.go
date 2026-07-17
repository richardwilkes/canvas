// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Stroke-rect drawing ops: nonAAStrokeRectOp (a tri-strip "picture frame", or a line strip for hairlines) and
// aaStrokeRectOp (four nested device-space rects with coverage ramps, mitered or beveled), plus the
// MakeStrokeRectOp/MakeNestedStrokeRectOp entry points drawRect and the nested-rect shape reduction dispatch through.

package gl

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/stroke"
)

// allowedStroke reports whether strokeRec's style can be drawn by this op family: this emits line primitives for
// hairlines, so only support hairlines if allowed by caps. Otherwise we support all hairlines, bevels, and miters, but
// not round joins. Also, check whether the miter limit makes a miter join effectively beveled; if so, it is only
// supported when using an AA stroke.
func allowedStroke(caps *Caps, strokeRec *stroke.Rec, aa gpu.AA) (isMiter, ok bool) {
	if style := strokeRec.Style(); style != stroke.StyleStroke && style != stroke.StyleHairline {
		panic("stroke rect requires stroke or hairline style")
	}
	if caps.AvoidLineDraws && strokeRec.IsHairlineStyle() {
		return false, false
	}
	// For hairlines, make bevel and round joins appear the same as mitered ones.
	if strokeRec.Width() == 0 {
		return true, true
	}
	switch strokeRec.Join() {
	case stroke.JoinBevel:
		return false, aa == gpu.AAYes // bevel only supported with AA
	case stroke.JoinMiter:
		isMiter = strokeRec.Miter() >= math.Sqrt2
		// Supported under non-AA only if it remains mitered.
		return isMiter, aa == gpu.AAYes || isMiter
	default:
		return false, false
	}
}

//////////////////////////////////////////////////////////////////////////////
// Non-AA stroking

var nonAAStrokeRectOpClassID = GenOpClassID()

// initNonAAStrokeRectStrip builds a triangle strip that strokes the rect. There are 8 unique vertices, but we repeat
// the last 2 to close up.
func initNonAAStrokeRectStrip(verts *[10]geom.Point, rect geom.Rect, width float32) {
	rad := width / 2

	verts[0] = geom.Point{X: rect.Left + rad, Y: rect.Top + rad}
	verts[1] = geom.Point{X: rect.Left - rad, Y: rect.Top - rad}
	verts[2] = geom.Point{X: rect.Right - rad, Y: rect.Top + rad}
	verts[3] = geom.Point{X: rect.Right + rad, Y: rect.Top - rad}
	verts[4] = geom.Point{X: rect.Right - rad, Y: rect.Bottom - rad}
	verts[5] = geom.Point{X: rect.Right + rad, Y: rect.Bottom + rad}
	verts[6] = geom.Point{X: rect.Left + rad, Y: rect.Bottom - rad}
	verts[7] = geom.Point{X: rect.Left - rad, Y: rect.Bottom + rad}
	verts[8] = verts[0]
	verts[9] = verts[1]

	if 2*rad >= rect.Width() {
		cx := (rect.Left + rect.Right) / 2
		verts[0].X = cx
		verts[2].X = cx
		verts[4].X = cx
		verts[6].X = cx
		verts[8].X = cx
	}
	if 2*rad >= rect.Height() {
		cy := (rect.Top + rect.Bottom) / 2
		verts[0].Y = cy
		verts[2].Y = cy
		verts[4].Y = cy
		verts[6].Y = cy
		verts[8].Y = cy
	}
}

const (
	vertsPerHairlineRect = 5
	vertsPerStrokeRect   = 10
)

// nonAAStrokeRectOp draws a stroked rect with no antialiasing (or as a hairline).
type nonAAStrokeRectOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	OpBase
	mesh        simpleMesh
	viewMatrix  geom.Matrix
	color       colorcore.PMColor4f
	rect        geom.Rect
	strokeWidth float32
}

// NewNonAAStrokeRectOp builds a non-antialiased stroked-rect draw op. The paint is consumed.
func NewNonAAStrokeRectOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, rect geom.Rect, strokeRec *stroke.Rec, aaType gpu.AAType) DrawOp {
	if _, ok := allowedStroke(caps, strokeRec, gpu.AANo); !ok {
		return nil
	}
	inputFlags := PipelineInputFlagNone
	// Depending on sub-pixel coordinates and the particular GPU, we may lose a corner of hairline rects. We jam all the
	// vertices to pixel centers to avoid this, but not when MSAA is enabled because it can cause ugly artifacts.
	if strokeRec.Style() == stroke.StyleHairline && aaType != gpu.AATypeMSAA {
		inputFlags |= PipelineSnapVerticesToPixelCenters
	}

	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := &nonAAStrokeRectOp{}
	o.InitOp(nonAAStrokeRectOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, aaType, nil, inputFlags)
	o.color = color
	o.viewMatrix = *viewMatrix
	o.rect = rect
	// Sort the rect for hairlines.
	o.rect.Sort()
	o.strokeWidth = strokeRec.Width()

	rad := o.strokeWidth / 2
	bounds := rect.Outset(rad, rad)

	// If our caller snaps to pixel centers then we have to round out the bounds.
	if inputFlags&PipelineSnapVerticesToPixelCenters != 0 {
		if o.strokeWidth != 0 && aaType != gpu.AATypeNone {
			panic("snapping requires hairline or non-AA")
		}
		bounds, _ = viewMatrix.MapRect(bounds)
		// We want to be consistent with how we snap non-aa lines. To match what we do in the vertex shader builder, we
		// first floor all the vertex values and then add half a pixel to force us to pixel centers.
		bounds = geom.Rect{
			Left:   geom.FloorToScalar(bounds.Left),
			Top:    geom.FloorToScalar(bounds.Top),
			Right:  geom.FloorToScalar(bounds.Right),
			Bottom: geom.FloorToScalar(bounds.Bottom),
		}
		bounds = bounds.Offset(0.5, 0.5)
		o.SetBounds(bounds, HasAABloat(false), IsHairline(false))
	} else {
		devBounds, _ := viewMatrix.MapRect(bounds)
		o.SetBounds(devBounds, HasAABloat(aaType != gpu.AATypeNone),
			IsHairline(o.strokeWidth == 0))
	}
	return o
}

// Name implements Op.
func (o *nonAAStrokeRectOp) Name() string { return "NonAAStrokeRectOp" }

// VisitProxies implements Op.
func (o *nonAAStrokeRectOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *nonAAStrokeRectOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *nonAAStrokeRectOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *nonAAStrokeRectOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	// This op uses uniform (not vertex) color, so doesn't need to track wide color.
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType, AnalysisCoverageNone,
		&o.color, nil)
}

// createProgramInfo builds the op's program info.
func (o *nonAAStrokeRectOp) createProgramInfo(state *OpFlushState) {
	localCoordsType := LocalCoordsTypeUnused
	if o.helper.UsesLocalCoords() {
		localCoordsType = LocalCoordsTypeUsePosition
	}
	gp := MakeDefaultGeoProc(ColorTypePremulUniform, o.color, CoverageTypeSolid, 0xff,
		localCoordsType, nil, &o.viewMatrix)

	primType := gpu.PrimitiveTypeLineStrip
	if o.strokeWidth > 0 {
		primType = gpu.PrimitiveTypeTriangleStrip
	}

	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp, primType,
		args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op: builds the vertex buffer for the stroked rect.
func (o *nonAAStrokeRectOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
	}

	stride := o.programInfo.GeomProc().gpBase().VertexStride()
	vertexCount := vertsPerHairlineRect
	if o.strokeWidth > 0 {
		vertexCount = vertsPerStrokeRect
	}

	vertices, vertexBuffer, firstVertex := state.MakeVertexSpace(uint64(stride), vertexCount)
	if vertices == nil {
		return
	}
	w := quadVertexWriter{buf: vertices}

	if o.strokeWidth > 0 {
		var verts [10]geom.Point
		initNonAAStrokeRectStrip(&verts, o.rect, o.strokeWidth)
		for _, v := range verts {
			w.putF32(v.X)
			w.putF32(v.Y)
		}
	} else {
		// Hairline.
		pts := [5]geom.Point{
			{X: o.rect.Left, Y: o.rect.Top},
			{X: o.rect.Right, Y: o.rect.Top},
			{X: o.rect.Right, Y: o.rect.Bottom},
			{X: o.rect.Left, Y: o.rect.Bottom},
			{X: o.rect.Left, Y: o.rect.Top},
		}
		for _, v := range pts {
			w.putF32(v.X)
			w.putF32(v.Y)
		}
	}

	o.mesh.vertexCount = vertexCount
	o.mesh.baseVertex = firstVertex
	o.mesh.vertexBuffer = vertexBuffer
	o.mesh.kind = simpleMeshNonIndexed
}

// OnExecute implements Op.
func (o *nonAAStrokeRectOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if !o.mesh.isValid() {
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

//////////////////////////////////////////////////////////////////////////////
// AA stroking

var aaStrokeRectOpClassID = GenOpClassID()

// aaStrokeRectOpPool is the free list of aaStrokeRectOp shells (see oppool.go).
var aaStrokeRectOpPool opPool[aaStrokeRectOp]

// strokeDevHalfSizeSupported reports whether the given half-stroke size is drawable: since the horizontal and vertical
// strokes share internal corners, the coverage value at those corners needs to be equal for both, which holds when the
// widths are (nearly) equal or both >= 1.
func strokeDevHalfSizeSupported(devHalfStrokeSize geom.Point) bool {
	return geom.ScalarNearlyEqual(devHalfStrokeSize.X, devHalfStrokeSize.Y) ||
		min(devHalfStrokeSize.X, devHalfStrokeSize.Y) >= 0.5
}

// computeAARects computes the nested outside/inside device-space rects an AA stroke draws. The results are one
// geometric answer consumed whole by the single caller; a result struct would only add indirection.
func computeAARects(caps *Caps, viewMatrix *geom.Matrix, rect geom.Rect, strokeWidth float32, miterStroke bool) (devOutside, devOutsideAssist, devInside geom.Rect, isDegenerate bool, devHalfStrokeSize geom.Point, ok bool) { //nolint:gocritic // see above
	var devStrokeSize geom.Point
	if strokeWidth > 0 {
		devStrokeSize = viewMatrix.MapVector(geom.Point{X: strokeWidth, Y: strokeWidth})
		devStrokeSize = geom.Point{
			X: geom.ScalarAbs(devStrokeSize.X),
			Y: geom.ScalarAbs(devStrokeSize.Y),
		}
	} else {
		devStrokeSize = geom.Point{X: 1, Y: 1}
	}

	dx := devStrokeSize.X
	dy := devStrokeSize.Y
	rx := dx / 2
	ry := dy / 2

	devHalfStrokeSize = geom.Point{X: rx, Y: ry}

	devRect, _ := viewMatrix.MapRect(rect)

	// Clip our draw rect 1 full stroke width plus bloat outside the viewport. This avoids interpolation precision
	// issues with very large coordinates.
	m := float32(caps.MaxRenderTargetSize)
	visibilityBounds := geom.Rect{Right: m, Bottom: m}.Outset(dx+1, dy+1)
	if !devRect.Intersect(visibilityBounds) {
		return devOutside, devOutsideAssist, devInside, false, devHalfStrokeSize, false
	}

	devOutside = devRect
	devOutsideAssist = devRect
	devInside = devRect

	devOutside = devOutside.Outset(rx, ry)
	devInside = devInside.Inset(rx, ry)

	// If we have a degenerate stroking rect (i.e. the stroke is larger than the inner rect) then we make a degenerate
	// inside rect to avoid double hitting. We will also jam all of the points together when we render these rects.
	spare := min(devRect.Width()-dx, devRect.Height()-dy)
	isDegenerate = spare <= 0
	if isDegenerate {
		cx := (devRect.Left + devRect.Right) / 2
		cy := (devRect.Top + devRect.Bottom) / 2
		devInside = geom.Rect{Left: cx, Top: cy, Right: cx, Bottom: cy}
	}

	// For bevel-stroke, use 2 rects (devOutside and devOutsideAssist) to draw the outside of the octagon. Because there
	// are 8 vertices on the outer edge, while the vertex number of the inner edge is 4, the same as miter-stroke.
	if !miterStroke {
		devOutside = devOutside.Inset(0, ry)
		devOutsideAssist = devOutsideAssist.Outset(0, ry)
	}

	return devOutside, devOutsideAssist, devInside, isDegenerate, devHalfStrokeSize, true
}

// createAAStrokeRectGP builds the geometry processor for an AA stroke-rect draw.
func createAAStrokeRectGP(usesMSAASurface, tweakAlphaForCoverage bool, viewMatrix *geom.Matrix, usesLocalCoords, wideColor bool) GeometryProcessor {
	// When MSAA is enabled, we have to extend our AA bloats and interpolate coverage values outside 0..1. We tell the
	// gp in this case that coverage is an unclamped attribute so it will call saturate(coverage) in the fragment
	// shader.
	coverageType := CoverageTypeSolid
	switch {
	case usesMSAASurface:
		coverageType = CoverageTypeAttributeUnclamped
	case !tweakAlphaForCoverage:
		coverageType = CoverageTypeAttribute
	}
	localCoordsType := LocalCoordsTypeUnused
	if usesLocalCoords {
		localCoordsType = LocalCoordsTypeUsePosition
	}
	colorType := ColorTypePremulAttribute
	if wideColor {
		colorType = ColorTypePremulWideAttribute
	}
	return MakeDefaultGeoProcForDeviceSpace(colorType, colorcore.PMColor4f{}, coverageType, 0xff,
		localCoordsType, nil, viewMatrix)
}

// aaStrokeRectInfo is one instance's geometry and color for a combined AA stroke-rect draw.
type aaStrokeRectInfo struct {
	color             colorcore.PMColor4f
	devOutside        geom.Rect
	devOutsideAssist  geom.Rect
	devInside         geom.Rect
	devHalfStrokeSize geom.Point
	degenerate        bool
}

const (
	miterIndexCnt              = 3 * 24
	miterVertexCnt             = 16
	numMiterRectsInIndexBuffer = 256

	bevelIndexCnt              = 48 + 36 + 24
	bevelVertexCnt             = 24
	numBevelRectsInIndexBuffer = 256
)

var (
	strokeRectIndexBufferKeyOnce sync.Once
	miterIndexBufferKey          gpu.UniqueKey
	bevelIndexBufferKey          gpu.UniqueKey
)

var miterStrokeRectIndices = []uint16{
	0 + 0, 1 + 0, 5 + 0, 5 + 0, 4 + 0, 0 + 0,
	1 + 0, 2 + 0, 6 + 0, 6 + 0, 5 + 0, 1 + 0,
	2 + 0, 3 + 0, 7 + 0, 7 + 0, 6 + 0, 2 + 0,
	3 + 0, 0 + 0, 4 + 0, 4 + 0, 7 + 0, 3 + 0,

	0 + 4, 1 + 4, 5 + 4, 5 + 4, 4 + 4, 0 + 4,
	1 + 4, 2 + 4, 6 + 4, 6 + 4, 5 + 4, 1 + 4,
	2 + 4, 3 + 4, 7 + 4, 7 + 4, 6 + 4, 2 + 4,
	3 + 4, 0 + 4, 4 + 4, 4 + 4, 7 + 4, 3 + 4,

	0 + 8, 1 + 8, 5 + 8, 5 + 8, 4 + 8, 0 + 8,
	1 + 8, 2 + 8, 6 + 8, 6 + 8, 5 + 8, 1 + 8,
	2 + 8, 3 + 8, 7 + 8, 7 + 8, 6 + 8, 2 + 8,
	3 + 8, 0 + 8, 4 + 8, 4 + 8, 7 + 8, 3 + 8,
}

// The bevel-stroke rect index layout: outer AA line 0~3 and 4~7, outer edge 8~11 and 12~15, inner edge 16~19, inner AA
// line 20~23.
var bevelStrokeRectIndices = []uint16{
	// Draw outer AA, from outer AA line to outer edge, shift is 0.
	0 + 0, 1 + 0, 9 + 0, 9 + 0, 8 + 0, 0 + 0,
	1 + 0, 5 + 0, 13 + 0, 13 + 0, 9 + 0, 1 + 0,
	5 + 0, 6 + 0, 14 + 0, 14 + 0, 13 + 0, 5 + 0,
	6 + 0, 2 + 0, 10 + 0, 10 + 0, 14 + 0, 6 + 0,
	2 + 0, 3 + 0, 11 + 0, 11 + 0, 10 + 0, 2 + 0,
	3 + 0, 7 + 0, 15 + 0, 15 + 0, 11 + 0, 3 + 0,
	7 + 0, 4 + 0, 12 + 0, 12 + 0, 15 + 0, 7 + 0,
	4 + 0, 0 + 0, 8 + 0, 8 + 0, 12 + 0, 4 + 0,

	// Draw the stroke, from outer edge to inner edge, shift is 8.
	0 + 8, 1 + 8, 9 + 8, 9 + 8, 8 + 8, 0 + 8,
	1 + 8, 5 + 8, 9 + 8,
	5 + 8, 6 + 8, 10 + 8, 10 + 8, 9 + 8, 5 + 8,
	6 + 8, 2 + 8, 10 + 8,
	2 + 8, 3 + 8, 11 + 8, 11 + 8, 10 + 8, 2 + 8,
	3 + 8, 7 + 8, 11 + 8,
	7 + 8, 4 + 8, 8 + 8, 8 + 8, 11 + 8, 7 + 8,
	4 + 8, 0 + 8, 8 + 8,

	// Draw the inner AA, from inner edge to inner AA line, shift is 16.
	0 + 16, 1 + 16, 5 + 16, 5 + 16, 4 + 16, 0 + 16,
	1 + 16, 2 + 16, 6 + 16, 6 + 16, 5 + 16, 1 + 16,
	2 + 16, 3 + 16, 7 + 16, 7 + 16, 6 + 16, 2 + 16,
	3 + 16, 0 + 16, 4 + 16, 4 + 16, 7 + 16, 3 + 16,
}

// getStrokeRectIndexBuffer returns the shared, cached index buffer for miter or bevel stroke rects.
func getStrokeRectIndexBuffer(rp *ResourceProvider, miterStroke bool) *Buffer {
	strokeRectIndexBufferKeyOnce.Do(func() {
		builder := gpu.UniqueKeyBuilder(&miterIndexBufferKey, gpu.GenerateUniqueKeyDomain(), 1,
			"MiterStrokeRectIndexBuffer")
		builder.Slice()[0] = 1
		builder.Finish()
		builder = gpu.UniqueKeyBuilder(&bevelIndexBufferKey, gpu.GenerateUniqueKeyDomain(), 1,
			"BevelStrokeRectIndexBuffer")
		builder.Slice()[0] = 1
		builder.Finish()
	})
	if miterStroke {
		return rp.findOrCreatePatternedIndexBuffer(miterStrokeRectIndices, miterIndexCnt,
			numMiterRectsInIndexBuffer, miterVertexCnt, &miterIndexBufferKey)
	}
	return rp.findOrCreatePatternedIndexBuffer(bevelStrokeRectIndices, bevelIndexCnt,
		numBevelRectsInIndexBuffer, bevelVertexCnt, &bevelIndexBufferKey)
}

// aaStrokeRectOp draws one or more antialiased stroked rects, batched together.
type aaStrokeRectOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	rects       []aaStrokeRectInfo
	OpBase
	mesh simpleMesh
	// rectsArr backs rects for the common single-instance case so a freshly constructed op does not heap-allocate a
	// separate backing array (rects is bootstrapped to rectsArr[:0]); combining more instances in via
	// OnCombineIfPossible grows past it into the heap as normal. Safe because ops are only ever referenced by pointer —
	// never value-copied — so the slice never outlives or aliases the wrong array.
	rectsArr    [1]aaStrokeRectInfo
	viewMatrix  geom.Matrix
	miterStroke bool
	wideColor   bool
}

// NewAAStrokeRectOpFromDevRects builds an AA stroke-rect draw op from already-computed device-space outside/inside
// rects. The paint is consumed.
func NewAAStrokeRectOpFromDevRects(paint *Paint, viewMatrix *geom.Matrix, devOutside, devInside geom.Rect, devHalfStrokeSize geom.Point) DrawOp {
	if !viewMatrix.RectStaysRect() {
		// The AA op only supports axis-aligned rectangles.
		return nil
	}
	if !strokeDevHalfSizeSupported(devHalfStrokeSize) {
		return nil
	}
	if devOutside.IsEmpty() || devInside.IsEmpty() {
		panic("AA stroke rect requires non-empty rects")
	}

	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := aaStrokeRectOpPool.borrow()
	o.viewMatrix = *viewMatrix
	o.rects = bootstrapInstances(o.rects, o.rectsArr[:0])
	o.InitOp(aaStrokeRectOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, nil,
		PipelineInputFlagNone)
	o.rects = append(o.rects, aaStrokeRectInfo{
		color:             color,
		devOutside:        devOutside,
		devOutsideAssist:  devOutside,
		devInside:         devInside,
		devHalfStrokeSize: devHalfStrokeSize,
	})
	o.SetBounds(devOutside, HasAABloat(true), IsHairline(false))
	o.miterStroke = true
	return o
}

// NewAAStrokeRectOp builds an AA stroke-rect draw op from a rect and stroke description. The paint is consumed.
func NewAAStrokeRectOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, rect geom.Rect, strokeRec *stroke.Rec) DrawOp {
	if !viewMatrix.RectStaysRect() {
		// The AA op only supports axis-aligned rectangles.
		return nil
	}
	isMiter, ok := allowedStroke(caps, strokeRec, gpu.AAYes)
	if !ok {
		return nil
	}
	var info aaStrokeRectInfo
	info.devOutside, info.devOutsideAssist, info.devInside, info.degenerate,
		info.devHalfStrokeSize, ok = computeAARects(caps, viewMatrix, rect, strokeRec.Width(),
		isMiter)
	if !ok {
		return nil
	}
	if !strokeDevHalfSizeSupported(info.devHalfStrokeSize) {
		return nil
	}

	info.color = paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := aaStrokeRectOpPool.borrow()
	o.viewMatrix = *viewMatrix
	o.rects = bootstrapInstances(o.rects, o.rectsArr[:0])
	o.InitOp(aaStrokeRectOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, nil,
		PipelineInputFlagNone)
	o.miterStroke = isMiter
	o.rects = append(o.rects, info)
	if isMiter {
		o.SetBounds(info.devOutside, HasAABloat(true), IsHairline(false))
	} else {
		// The outer polygon of the bevel stroke is an octagon specified by the points of a pair of overlapping
		// rectangles where one is wide and the other is narrow.
		bounds := joinPossiblyEmptyRect(info.devOutside, info.devOutsideAssist)
		o.SetBounds(bounds, HasAABloat(true), IsHairline(false))
	}
	return o
}

// Name implements Op.
func (o *aaStrokeRectOp) Name() string { return "AAStrokeRect" }

// recycle implements Op: returns this dead op to its free list.
func (o *aaStrokeRectOp) recycle() {
	recycleProgramInfo(o.programInfo)
	recycleKeepingBacking(&aaStrokeRectOpPool, o,
		func(o *aaStrokeRectOp) []aaStrokeRectInfo { return o.rects },
		func(o *aaStrokeRectOp, s []aaStrokeRectInfo) { o.rects = s })
}

// VisitProxies implements Op.
func (o *aaStrokeRectOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *aaStrokeRectOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *aaStrokeRectOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *aaStrokeRectOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.rects[len(o.rects)-1].color, &o.wideColor)
}

// compatibleWithCoverageAsAlpha reports whether coverage can be folded in as alpha: when MSAA is enabled, we have to
// extend our AA bloats and interpolate coverage values outside 0..1, which makes us incompatible with coverage as
// alpha.
func (o *aaStrokeRectOp) compatibleWithCoverageAsAlpha(usesMSAASurface bool) bool {
	return !usesMSAASurface && o.helper.CompatibleWithCoverageAsAlpha()
}

// createProgramInfo builds the op's program info.
func (o *aaStrokeRectOp) createProgramInfo(state *OpFlushState) {
	args := state.OpArgs()
	gp := createAAStrokeRectGP(args.UsesMSAASurface(),
		o.compatibleWithCoverageAsAlpha(args.UsesMSAASurface()), &o.viewMatrix,
		o.helper.UsesLocalCoords(), o.wideColor)
	if gp == nil {
		return
	}
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op: builds the vertex buffer for the batched AA stroke rects.
func (o *aaStrokeRectOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	innerVertexNum := 4
	outerVertexNum := 4
	if !o.miterStroke {
		outerVertexNum = 8
	}
	verticesPerInstance := (outerVertexNum + innerVertexNum) * 2
	indicesPerInstance := miterIndexCnt
	maxRects := numMiterRectsInIndexBuffer
	if !o.miterStroke {
		indicesPerInstance = bevelIndexCnt
		maxRects = numBevelRectsInIndexBuffer
	}

	indexBuffer := getStrokeRectIndexBuffer(state.ResourceProvider(), o.miterStroke)
	if indexBuffer == nil {
		return
	}
	stride := o.programInfo.GeomProc().gpBase().VertexStride()
	vertices := makePatternHelper(state, uint64(stride), indexBuffer, verticesPerInstance,
		indicesPerInstance, len(o.rects), maxRects, &o.mesh)
	if vertices == nil {
		return
	}
	w := quadVertexWriter{buf: vertices}

	for i := range o.rects {
		info := &o.rects[i]
		o.generateAAStrokeRectGeometry(&w, info, state.OpArgs().UsesMSAASurface())
	}
}

// OnExecute implements Op.
func (o *aaStrokeRectOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
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

// OnCombineIfPossible implements Op: merges compatible AA stroke-rect ops into one batched draw.
func (o *aaStrokeRectOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*aaStrokeRectOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}
	if o.miterStroke != that.miterStroke {
		return CombineResultCannotCombine
	}
	// We apply the view matrix to the rect points on the cpu. However, if the pipeline uses local coords then we won't
	// be able to combine.
	if o.helper.UsesLocalCoords() && !o.viewMatrix.CheapEqual(&that.viewMatrix) {
		return CombineResultCannotCombine
	}
	o.rects = append(o.rects, that.rects...)
	o.wideColor = o.wideColor || that.wideColor
	return CombineResultMerged
}

// writeAAStrokeQuad writes one 4-vertex tri-fan quad: per-vertex position (from the tri-fan corners of rect), color,
// and optional coverage.
func writeAAStrokeQuad(w *quadVertexWriter, rect geom.Rect, color colorcore.PMColor4f, wideColor bool, coverage float32, writeCoverage bool) {
	// TriFan corner order: (l,t), (l,b), (r,b), (r,t).
	corners := [4][2]float32{
		{rect.Left, rect.Top},
		{rect.Left, rect.Bottom},
		{rect.Right, rect.Bottom},
		{rect.Right, rect.Top},
	}
	for _, c := range corners {
		w.putF32(c[0])
		w.putF32(c[1])
		w.putColor(color, wideColor)
		if writeCoverage {
			w.putF32(coverage)
		}
	}
}

// generateAAStrokeRectGeometry writes vertices for four nested rectangles — two ramps from 0 to full coverage, one on
// the exterior of the stroke and the other on the interior.
func (o *aaStrokeRectOp) generateAAStrokeRectGeometry(w *quadVertexWriter, info *aaStrokeRectInfo, usesMSAASurface bool) {
	color := info.color
	devOutside := info.devOutside
	devOutsideAssist := info.devOutsideAssist
	devInside := info.devInside
	devHalfStrokeSize := info.devHalfStrokeSize

	if !strokeDevHalfSizeSupported(devHalfStrokeSize) {
		panic("unsupported dev half stroke size")
	}

	tweakAlphaForCoverage := o.compatibleWithCoverageAsAlpha(usesMSAASurface)
	writeCoverage := !tweakAlphaForCoverage

	// How much do we inset toward the inside of the strokes?
	inset := min(0.5, min(devHalfStrokeSize.X, devHalfStrokeSize.Y))
	innerCoverage := float32(1)
	if inset < 0.5 {
		// The stroke is subpixel, so reduce the coverage to simulate the narrower strokes.
		innerCoverage = 2 * inset / (inset + 0.5)
	}

	// How much do we outset away from the outside of the strokes? We always want to keep the AA picture frame one pixel
	// wide.
	outset := 1 - inset
	outerCoverage := float32(0)

	// How much do we outset away from the interior side of the stroke (toward the center)?
	interiorOutset := outset
	interiorCoverage := outerCoverage

	if usesMSAASurface {
		// Since we're using MSAA, extend our outsets to ensure any pixel with partial coverage has a full sample mask.
		const msaaExtraBloat = float32(math.Sqrt2 - 0.5)
		outset += msaaExtraBloat
		outerCoverage -= msaaExtraBloat

		insetExtraBloat := min(inset+msaaExtraBloat,
			min(devHalfStrokeSize.X, devHalfStrokeSize.Y)) - inset
		inset += insetExtraBloat
		innerCoverage += insetExtraBloat

		interiorExtraBloat := min(interiorOutset+msaaExtraBloat,
			min(devInside.Width(), devInside.Height())/2) - interiorOutset
		interiorOutset += interiorExtraBloat
		interiorCoverage -= interiorExtraBloat
	}

	scaleColor := func(c colorcore.PMColor4f, s float32) colorcore.PMColor4f {
		return colorcore.PMColor4f{R: c.R * s, G: c.G * s, B: c.B * s, A: c.A * s}
	}
	innerColor := color
	outerColor := colorcore.PMColor4f{}
	if tweakAlphaForCoverage {
		innerColor = scaleColor(color, innerCoverage)
	} else {
		outerColor = color
	}

	// Exterior outset rect (away from stroke).
	writeAAStrokeQuad(w, devOutside.Inset(-outset, -outset), outerColor, o.wideColor,
		outerCoverage, writeCoverage)

	if !o.miterStroke {
		// Second exterior outset.
		writeAAStrokeQuad(w, devOutsideAssist.Inset(-outset, -outset), outerColor, o.wideColor,
			outerCoverage, writeCoverage)
	}

	// Exterior inset rect (toward stroke).
	writeAAStrokeQuad(w, devOutside.Inset(inset, inset), innerColor, o.wideColor, innerCoverage,
		writeCoverage)

	if !o.miterStroke {
		// Second exterior inset.
		writeAAStrokeQuad(w, devOutsideAssist.Inset(inset, inset), innerColor, o.wideColor,
			innerCoverage, writeCoverage)
	}

	if !info.degenerate {
		// Interior inset rect (toward stroke).
		writeAAStrokeQuad(w, devInside.Inset(-inset, -inset), innerColor, o.wideColor,
			innerCoverage, writeCoverage)

		// Interior outset rect (away from stroke, toward the center of the rect).
		interiorAABoundary := devInside.Inset(interiorOutset, interiorOutset)
		coverageBackset := float32(0) // Adds back coverage when the interior AA edges cross.
		if interiorAABoundary.Left > interiorAABoundary.Right {
			coverageBackset = (interiorAABoundary.Left - interiorAABoundary.Right) /
				(interiorOutset * 2)
			cx := (interiorAABoundary.Left + interiorAABoundary.Right) / 2
			interiorAABoundary.Left = cx
			interiorAABoundary.Right = cx
		}
		if interiorAABoundary.Top > interiorAABoundary.Bottom {
			coverageBackset = max(
				(interiorAABoundary.Top-interiorAABoundary.Bottom)/(interiorOutset*2),
				coverageBackset,
			)
			cy := (interiorAABoundary.Top + interiorAABoundary.Bottom) / 2
			interiorAABoundary.Top = cy
			interiorAABoundary.Bottom = cy
		}
		if coverageBackset > 0 {
			// The interior edges crossed. Lerp back toward innerCoverage, which is what this op will draw in the
			// degenerate case. This gives a smooth transition into the degenerate case.
			interiorCoverage += interiorCoverage*(1-coverageBackset) +
				innerCoverage*coverageBackset
		}
		interiorColor := color
		if tweakAlphaForCoverage {
			interiorColor = scaleColor(color, interiorCoverage)
		}
		writeAAStrokeQuad(w, interiorAABoundary, interiorColor, o.wideColor, interiorCoverage,
			writeCoverage)
	} else {
		// When the interior rect has become degenerate we smoosh to a single point.
		if devInside.Left != devInside.Right || devInside.Top != devInside.Bottom {
			panic("degenerate interior must be a point")
		}
		writeAAStrokeQuad(w, devInside, innerColor, o.wideColor, innerCoverage, writeCoverage)
		// ... unless we are degenerate, in which case we must apply the scaled coverage.
		writeAAStrokeQuad(w, devInside, innerColor, o.wideColor, innerCoverage, writeCoverage)
	}
}

//////////////////////////////////////////////////////////////////////////////
// Entry points.

// MakeStrokeRectOp builds a stroked-rect draw op, dispatching to the AA or non-AA form. The paint is consumed; returns
// nil when the stroke is unsupported.
func MakeStrokeRectOp(caps *Caps, paint *Paint, aaType gpu.AAType, viewMatrix *geom.Matrix, rect geom.Rect, strokeRec *stroke.Rec) DrawOp {
	if aaType == gpu.AATypeCoverage {
		return NewAAStrokeRectOp(caps, paint, viewMatrix, rect, strokeRec)
	}
	return NewNonAAStrokeRectOp(caps, paint, viewMatrix, rect, strokeRec, aaType)
}

// MakeNestedStrokeRectOp fills the area between the two rects (both already in the same coordinate space) with AA. The
// paint is consumed.
func MakeNestedStrokeRectOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, rects [2]geom.Rect) DrawOp {
	if !viewMatrix.RectStaysRect() {
		panic("nested stroke rects require a rect-stays-rect matrix")
	}
	if rects[0].IsEmpty() || rects[1].IsEmpty() {
		panic("nested stroke rects must be non-empty")
	}

	devOutside, _ := viewMatrix.MapRect(rects[0])
	devInside, _ := viewMatrix.MapRect(rects[1])
	dx := devOutside.Right - devInside.Right
	dy := devOutside.Bottom - devInside.Bottom

	// Clips our draw rects 1 full pixel outside the viewport. This avoids interpolation precision issues with very
	// large coordinates.
	m := float32(caps.MaxRenderTargetSize)
	visibilityBounds := geom.Rect{Right: m, Bottom: m}.Outset(1, 1)

	if !devOutside.Intersect(visibilityBounds.Outset(dx, dy)) {
		return nil
	}

	if devInside.IsEmpty() || !devInside.Intersect(visibilityBounds) {
		if devOutside.IsEmpty() {
			return nil
		}
		quad := DrawQuad{
			Device: MakeQuadFromRect(rects[0], viewMatrix),
			Local:  MakeQuadFromRectNoTransform(rects[0]), EdgeFlags: gpu.QuadAAFlagsAll,
		}
		return NewFillRectOp(paint, gpu.AATypeCoverage, &quad, nil, PipelineInputFlagNone)
	}

	return NewAAStrokeRectOpFromDevRects(paint, viewMatrix, devOutside, devInside,
		geom.Point{X: dx * 0.5, Y: dy * 0.5})
}
