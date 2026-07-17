// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// strokeTessellator prepares GPU data for, and then draws, a stroke's tessellated geometry. Renders strokes as
// fixed-count triangle strip instances; any extra triangles not needed by the instance are emitted as degenerate
// triangles. A fallback vertex buffer for platforms without gl_VertexID is not needed since desktop core profiles
// always have it.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// pathStrokeList is a linked list of (path, stroke, color) draws that merged ops accumulate.
type pathStrokeList struct {
	path   *path.Path
	next   *pathStrokeList
	stroke stroke.Rec
	color  colorcore.PMColor4f
}

// strokeTessellator prepares and draws the tessellated geometry for one or more merged strokes.
type strokeTessellator struct {
	vertexChunks []vertexChunk
	vertexCount  int
	attribs      PatchAttribs
}

// newStrokeTessellator creates a stroke tessellator for the given patch attribute layout.
func newStrokeTessellator(attribs PatchAttribs) *strokeTessellator {
	return &strokeTessellator{attribs: attribs | PatchAttribJoinControlPoint}
}

// writeFixedCountPatches walks each path's stroke geometry and writes the corresponding patches (lines, quadratics,
// conics, cubics, cusp circles) via patchWriter.
func writeFixedCountPatches(patchWriter *patchWriter, shaderMatrix *geom.Matrix, pathStrokes *pathStrokeList) {
	// The vector xform approximates how the control points are transformed by the shader to more accurately compute how
	// many *parametric* segments are needed. getMaxScale() returns -1 if it can't compute a scale factor (e.g.
	// perspective); taking the absolute value automatically converts that to an identity scale factor for our purposes.
	patchWriter.setShaderTransform(makeVectorXform(shaderMatrix),
		geom.ScalarAbs(shaderMatrix.MaxScale()))
	if patchWriter.attribs&PatchAttribStrokeParams == 0 {
		// Strokes are static. Calculate tolerances once.
		patchWriter.updateUniformStrokeParams(makeStrokeParams(&pathStrokes.stroke))
	}

	for pathStroke := pathStrokes; pathStroke != nil; pathStroke = pathStroke.next {
		if patchWriter.attribs&PatchAttribStrokeParams != 0 {
			// Strokes are dynamic. Calculate tolerances every time.
			patchWriter.updateStrokeParamsAttrib(makeStrokeParams(&pathStroke.stroke))
		}
		if patchWriter.attribs&PatchAttribColor != 0 {
			patchWriter.updateColorAttrib(pathStroke.color)
		}

		strokeIter := newStrokeIterator(pathStroke.path, &pathStroke.stroke, shaderMatrix)
		for strokeIter.next() {
			p := strokeIter.pts()
			switch strokeIter.verb() {
			case strokeIterVerbContourFinished:
				patchWriter.writeDeferredStrokePatch()
			case strokeIterVerbCircle:
				// Round cap or else an empty stroke that is specified to be drawn as a circle.
				patchWriter.writeCircle(p[0])
				// A regular move invalidates the previous control point; the stroke iterator tells us a new value to
				// use.
				patchWriter.updateJoinControlPointAttrib(p[0])
			case strokeIterVerbMoveWithinContour:
				patchWriter.updateJoinControlPointAttrib(p[0])
			case strokeIterVerbLine:
				patchWriter.writeLine(p[0], p[1])
			case strokeIterVerbQuad:
				quad := [3]geom.Point{p[0], p[1], p[2]}
				if conicHasCusp(&quad) {
					// The cusp is always at the midtangent.
					cusp := geom.EvalQuadAt(quad[:], geom.FindQuadMidTangent(quad[:]))
					patchWriter.writeCircle(cusp)
					// A quad can only have a cusp if it's flat with a 180-degree turnaround.
					patchWriter.writeLine(quad[0], cusp)
					patchWriter.writeLine(cusp, quad[2])
				} else {
					patchWriter.writeQuadratic(quad[0], quad[1], quad[2])
				}
			case strokeIterVerbConic:
				pts := [3]geom.Point{p[0], p[1], p[2]}
				if conicHasCusp(&pts) {
					// The cusp is always at the midtangent.
					conic := geom.MakeConic(pts[0], pts[1], pts[2], strokeIter.w())
					cusp := conic.EvalAt(conic.FindMidTangent())
					patchWriter.writeCircle(cusp)
					// A conic can only have a cusp if it's flat with a 180-degree turnaround.
					patchWriter.writeLine(pts[0], cusp)
					patchWriter.writeLine(cusp, pts[2])
				} else {
					patchWriter.writeConic(pts[0], pts[1], pts[2], strokeIter.w())
				}
			case strokeIterVerbCubic:
				cubic := [4]geom.Point{p[0], p[1], p[2], p[3]}
				var t [2]float32
				var areCusps bool
				numChops := findCubicConvex180Chops(&cubic, &t, &areCusps)
				switch numChops {
				case 0:
					patchWriter.writeCubic(cubic[0], cubic[1], cubic[2], cubic[3])
				case 1:
					var chops [7]geom.Point
					geom.ChopCubicAt(cubic[:], chops[:], t[0])
					if areCusps {
						patchWriter.writeCircle(chops[3])
						// In a perfect world, these 3 points would be equal after chopping on a cusp.
						chops[2] = chops[3]
						chops[4] = chops[3]
					}
					patchWriter.writeCubic(chops[0], chops[1], chops[2], chops[3])
					patchWriter.writeCubic(chops[3], chops[4], chops[5], chops[6])
				default:
					if numChops != 2 {
						panic("findCubicConvex180Chops returned more than 2 chops")
					}
					var chops [10]geom.Point
					geom.ChopCubicAt2(cubic[:], chops[:], t[0], t[1])
					if areCusps {
						patchWriter.writeCircle(chops[3])
						patchWriter.writeCircle(chops[6])
						// Two cusps are only possible if it's a flat line with two 180-degree turnarounds.
						patchWriter.writeLine(chops[0], chops[3])
						patchWriter.writeLine(chops[3], chops[6])
						patchWriter.writeLine(chops[6], chops[9])
					} else {
						patchWriter.writeCubic(chops[0], chops[1], chops[2], chops[3])
						patchWriter.writeCubic(chops[3], chops[4], chops[5], chops[6])
						patchWriter.writeCubic(chops[6], chops[7], chops[8], chops[9])
					}
				}
			}
		}
	}

	// Flush any last recorded contour.
	patchWriter.writeDeferredStrokePatch()
}

// prepare is called before draw; it prepares the GPU buffers containing the geometry to tessellate.
func (t *strokeTessellator) prepare(target *OpFlushState, shaderMatrix *geom.Matrix, pathStrokes *pathStrokeList, totalCombinedStrokeVerbCnt int) {
	preallocCount := fixedCountStrokesPreallocCount(totalCombinedStrokeVerbCnt)
	worstCase := newLinearTolerances()
	alloc := newVertexChunkPatchAllocator(uint64(patchStride(t.attribs)), &worstCase, target,
		&t.vertexChunks, preallocCount)
	writer := newStrokePatchWriter(t.attribs, alloc)
	writeFixedCountPatches(writer, shaderMatrix, pathStrokes)
	alloc.close()
	t.vertexCount = fixedCountStrokesVertexCountForTolerances(&worstCase)

	// A static fallback vertex buffer holding the edge IDs would be bound here if gl_VertexID were unsupported; desktop
	// core profiles always have gl_VertexID, so that lane is trimmed.
}

// draw issues draw calls for the tessellated stroke. The caller is responsible for creating and binding a pipeline that
// uses this class's shader before calling draw.
func (t *strokeTessellator) draw(flushState *OpFlushState) {
	if len(t.vertexChunks) == 0 || t.vertexCount <= 0 {
		return
	}
	renderPass := flushState.OpsRenderPass()
	for i := range t.vertexChunks {
		chunk := &t.vertexChunks[i]
		renderPass.BindBuffers(nil, chunk.buffer, nil)
		renderPass.DrawInstanced(chunk.count, chunk.base, t.vertexCount, 0)
	}
}
