// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The two path tessellators (curve and wedge): they prepare GPU data for, and then draw, a path's tessellated geometry
// via fixed-count instanced draws over the static fixed-count vertex/index buffers. The curve tessellator emits an
// array of "outer curve" patches — each an independent 4-point curve (cubic, or conic with w in the last control point)
// — leaving the inner fan to be filled by some external source of geometry (the inner-fan triangulation or a bounding
// box). The wedge tessellator additionally fans every patch around a per-contour midpoint so the wedges alone define
// the complete path once stenciled.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
)

// pathDrawList is a linked list of (pathMatrix, path, color) draws that merged ops accumulate.
type pathDrawList struct {
	path       *path.Path
	next       *pathDrawList
	pathMatrix geom.Matrix
	color      colorcore.PMColor4f
}

// pathTessellator is the interface shared by the curve and wedge tessellators: prepare must be called before draw, and
// the caller is responsible for binding its desired pipeline ahead of draw.
type pathTessellator interface {
	patchAttribs() PatchAttribs
	prepare(target *OpFlushState, shaderMatrix *geom.Matrix, drawList *pathDrawList,
		totalCombinedPathVerbCnt int)
	draw(flushState *OpFlushState)
}

// pathTessellatorBase carries state shared by both tessellator implementations.
type pathTessellatorBase struct {
	fixedVertexBuffer *Buffer
	fixedIndexBuffer  *Buffer
	vertexChunks      []vertexChunk
	// maxVertexCount is the max number of vertices that must be drawn to account for the accumulated tessellation
	// levels of the written patches.
	maxVertexCount int
	attribs        PatchAttribs
}

func (b *pathTessellatorBase) patchAttribs() PatchAttribs { return b.attribs }

// initPathTessellator initializes the shared tessellator state: without shader infinity support the explicit-curve-type
// attrib is required.
func (b *pathTessellatorBase) initPathTessellator(infinitySupport bool, attribs PatchAttribs) {
	b.attribs = attribs
	if !infinitySupport {
		b.attribs |= PatchAttribExplicitCurveType
	}
}

// drawFixedCount issues the instanced draws over the patch chunks (both tessellators' draw()).
func (b *pathTessellatorBase) drawFixedCount(flushState *OpFlushState) {
	if b.fixedVertexBuffer == nil || b.fixedIndexBuffer == nil {
		return
	}
	renderPass := flushState.OpsRenderPass()
	for i := range b.vertexChunks {
		chunk := &b.vertexChunks[i]
		renderPass.BindBuffers(b.fixedIndexBuffer, chunk.buffer, b.fixedVertexBuffer)
		// The max vertex count is the logical number of vertices that the GPU needs to emit per instance, so with
		// drawIndexedInstanced it's provided as the index count.
		renderPass.DrawIndexedInstanced(b.maxVertexCount, 0, chunk.count, chunk.base, 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// PathCurveTessellator

// pathCurveTessellator draws an array of "outer curve" patches. Each patch is an independent 4-point curve,
// representing either a cubic or a conic. Quadratics are converted to cubics and triangles are converted to conics with
// w=Inf.
type pathCurveTessellator struct {
	pathTessellatorBase
}

// newPathCurveTessellator returns a curve tessellator ready for prepare/draw.
func newPathCurveTessellator(infinitySupport bool, attribs PatchAttribs) *pathCurveTessellator {
	t := &pathCurveTessellator{}
	t.initPathTessellator(infinitySupport, attribs)
	return t
}

// writeCurvePatches walks each path in drawList and writes one tessellation patch per quadratic, conic, or cubic verb
// (lines are skipped; they contribute no curve patches).
func writeCurvePatches(patchWriter *patchWriter, shaderMatrix *geom.Matrix, drawList *pathDrawList) {
	patchWriter.setShaderTransform(makeVectorXform(shaderMatrix), 1)
	for draw := drawList; draw != nil; draw = draw.next {
		m := makeTessAffineMatrix(&draw.pathMatrix)
		if patchWriter.attribs&PatchAttribColor != 0 {
			patchWriter.updateColorAttrib(draw.color)
		}
		it := path.NewRawIter(draw.path)
		for {
			verb, pts, w, ok := it.Next()
			if !ok {
				break
			}
			switch verb {
			case path.VerbQuad:
				patchWriter.writeQuadratic(m.mapPoint(pts[0]), m.mapPoint(pts[1]),
					m.mapPoint(pts[2]))
			case path.VerbConic:
				patchWriter.writeConic(m.mapPoint(pts[0]), m.mapPoint(pts[1]),
					m.mapPoint(pts[2]), w)
			case path.VerbCubic:
				patchWriter.writeCubic(m.mapPoint(pts[0]), m.mapPoint(pts[1]),
					m.mapPoint(pts[2]), m.mapPoint(pts[3]))
			default:
			}
		}
	}
}

// prepareWithTriangles writes out extra space-filling triangles to connect the curve patches with any external source
// of geometry (e.g. an inner triangulation that handles winding explicitly) before the curve patches.
func (t *pathCurveTessellator) prepareWithTriangles(target *OpFlushState, shaderMatrix *geom.Matrix, extraTriangles *breadcrumbTriangleList, drawList *pathDrawList, totalCombinedPathVerbCnt int) {
	patchPreallocCount := fixedCountCurvesPreallocCount(totalCombinedPathVerbCnt)
	if extraTriangles != nil {
		patchPreallocCount += extraTriangles.count
	}

	if patchPreallocCount != 0 {
		worstCase := newLinearTolerances()
		alloc := newVertexChunkPatchAllocator(uint64(patchStride(t.attribs)), &worstCase, target,
			&t.vertexChunks, patchPreallocCount)
		writer := newPatchWriter(t.attribs, alloc, true /*addTrianglesWhenChopping*/, true /*discardFlatCurves*/)

		if extraTriangles != nil {
			breadcrumbCount := 0
			for tri := extraTriangles.head; tri != nil; tri = tri.next {
				breadcrumbCount++
				p0, p1, p2 := tri.pts[0], tri.pts[1], tri.pts[2]
				if (p0.X == p1.X && p1.X == p2.X) || (p0.Y == p1.Y && p1.Y == p2.Y) {
					// Cull completely horizontal or vertical triangles. The triangulator can't always get these
					// breadcrumb edges right when they run parallel to the sweep direction because their winding is
					// undefined by its current definition. This seemed safe, but if there is a view matrix it will
					// introduce T-junctions.
					continue
				}
				writer.writeTriangle(p0, p1, p2)
			}
			if breadcrumbCount != extraTriangles.count {
				panic("breadcrumb count out of sync with the list")
			}
		}

		writeCurvePatches(writer, shaderMatrix, drawList)
		alloc.close()
		t.maxVertexCount = fixedCountCurvesVertexCountForTolerances(&worstCase)
	}

	fixedCountStaticBuffers()
	rp := target.ResourceProvider()
	t.fixedVertexBuffer = rp.FindOrMakeStaticBuffer(gpu.BufferTypeVertex,
		uint64(len(fixedCountCurveVertexBytes)), fixedCountCurveVertexBytes,
		&fixedCountCurveVertexKey)
	t.fixedIndexBuffer = rp.FindOrMakeStaticBuffer(gpu.BufferTypeIndex,
		uint64(len(fixedCountCurveIndexBytes)), fixedCountCurveIndexBytes,
		&fixedCountCurveIndexKey)
}

// prepare implements pathTessellator.
func (t *pathCurveTessellator) prepare(target *OpFlushState, shaderMatrix *geom.Matrix, drawList *pathDrawList, totalCombinedPathVerbCnt int) {
	t.prepareWithTriangles(target, shaderMatrix, nil, drawList, totalCombinedPathVerbCnt)
}

// draw implements pathTessellator.
func (t *pathCurveTessellator) draw(flushState *OpFlushState) { t.drawFixedCount(flushState) }

// drawHullInstances draws a 4-point instance for each patch, used for drawing convex hulls over each cubic with the
// hull shader. The caller is responsible for binding its desired pipeline ahead of time. Desktop core profiles always
// have a built-in vertex index, so no explicit vertex buffer is needed here.
func (t *pathCurveTessellator) drawHullInstances(flushState *OpFlushState) {
	renderPass := flushState.OpsRenderPass()
	for i := range t.vertexChunks {
		chunk := &t.vertexChunks[i]
		renderPass.BindBuffers(nil, chunk.buffer, nil)
		renderPass.DrawInstanced(chunk.count, chunk.base, 4, 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// PathWedgeTessellator

// pathWedgeTessellator prepares an array of "wedge" patches. A wedge is an independent, 5-point closed contour
// consisting of 4 control points plus an anchor point fanning from the center of the curve's resident contour. Once
// stenciled, these wedges alone define the complete path.
type pathWedgeTessellator struct {
	pathTessellatorBase
}

// newPathWedgeTessellator returns a wedge tessellator ready for prepare/draw.
func newPathWedgeTessellator(infinitySupport bool, attribs PatchAttribs) *pathWedgeTessellator {
	t := &pathWedgeTessellator{}
	t.initPathTessellator(infinitySupport, attribs|PatchAttribFanPoint)
	return t
}

// writeWedgePatches walks each contour in drawList and writes a fan-shaped wedge patch per verb, converting lines to
// degenerate cubics and closing each contour implicitly back to its start point.
func writeWedgePatches(patchWriter *patchWriter, shaderMatrix *geom.Matrix, drawList *pathDrawList) {
	patchWriter.setShaderTransform(makeVectorXform(shaderMatrix), 1)
	for draw := drawList; draw != nil; draw = draw.next {
		m := makeTessAffineMatrix(&draw.pathMatrix)
		if patchWriter.attribs&PatchAttribColor != 0 {
			patchWriter.updateColorAttrib(draw.color)
		}
		parser := newMidpointContourParser(draw.path)
		for parser.parseNextContour() {
			patchWriter.updateFanPointAttrib(m.mapPoint(parser.currentMidpoint()))
			var lastPoint, startPoint geom.Point
			for _, rec := range parser.currentContour() {
				switch rec.verb {
				case path.VerbMove:
					startPoint = rec.pts[0]
					lastPoint = rec.pts[0]
				case path.VerbLine:
					// Explicitly convert the line to an equivalent cubic with four distinct control points because it
					// fans better and avoids double-hitting pixels.
					patchWriter.writeLine(m.mapPoint(rec.pts[0]), m.mapPoint(rec.pts[1]))
					lastPoint = rec.pts[1]
				case path.VerbQuad:
					patchWriter.writeQuadratic(m.mapPoint(rec.pts[0]), m.mapPoint(rec.pts[1]),
						m.mapPoint(rec.pts[2]))
					lastPoint = rec.pts[2]
				case path.VerbConic:
					patchWriter.writeConic(m.mapPoint(rec.pts[0]), m.mapPoint(rec.pts[1]),
						m.mapPoint(rec.pts[2]), rec.weight)
					lastPoint = rec.pts[2]
				case path.VerbCubic:
					patchWriter.writeCubic(m.mapPoint(rec.pts[0]), m.mapPoint(rec.pts[1]),
						m.mapPoint(rec.pts[2]), m.mapPoint(rec.pts[3]))
					lastPoint = rec.pts[3]
				case path.VerbClose:
					// Ignore. We can assume an implicit close at the end.
				}
			}
			if lastPoint != startPoint {
				patchWriter.writeLine(m.mapPoint(lastPoint), m.mapPoint(startPoint))
			}
		}
	}
}

// prepare implements pathTessellator.
func (t *pathWedgeTessellator) prepare(target *OpFlushState, shaderMatrix *geom.Matrix, drawList *pathDrawList, totalCombinedPathVerbCnt int) {
	if patchPreallocCount := fixedCountWedgesPreallocCount(totalCombinedPathVerbCnt); patchPreallocCount != 0 {
		worstCase := newLinearTolerances()
		alloc := newVertexChunkPatchAllocator(uint64(patchStride(t.attribs)), &worstCase, target,
			&t.vertexChunks, patchPreallocCount)
		writer := newPatchWriter(t.attribs, alloc, false /*addTrianglesWhenChopping*/, false /*discardFlatCurves*/)
		writeWedgePatches(writer, shaderMatrix, drawList)
		alloc.close()
		t.maxVertexCount = fixedCountWedgesVertexCountForTolerances(&worstCase)
	}

	fixedCountStaticBuffers()
	rp := target.ResourceProvider()
	t.fixedVertexBuffer = rp.FindOrMakeStaticBuffer(gpu.BufferTypeVertex,
		uint64(len(fixedCountWedgeVertexBytes)), fixedCountWedgeVertexBytes,
		&fixedCountWedgeVertexKey)
	t.fixedIndexBuffer = rp.FindOrMakeStaticBuffer(gpu.BufferTypeIndex,
		uint64(len(fixedCountWedgeIndexBytes)), fixedCountWedgeIndexBytes,
		&fixedCountWedgeIndexKey)
}

// draw implements pathTessellator.
func (t *pathWedgeTessellator) draw(flushState *OpFlushState) { t.drawFixedCount(flushState) }
