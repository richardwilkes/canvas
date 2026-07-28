// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// patchWriter emits tessellation patches for the three fill/stroke configurations this port supports:
//
//   - CurveWriter: Optional color/wide-color-if-enabled/explicit-curve-type + AddTrianglesWhenChopping +
//     DiscardFlatCurves
//   - WedgeWriter: Required fan point + Optional color/wide-color-if-enabled/explicit-curve-type
//   - StrokeWriter: Required join control point + Optional stroke-params/color/
//     wide-color-if-enabled/explicit-curve-type + ReplicateLineEndPoints + TrackJoinControlPoints
//
// Each configuration is selected by constructor-time feature flags plus the runtime PatchAttribs. A "patch" is 4
// control points (8 floats) defining the curve's geometry — quads are converted to equivalent cubics on the CPU during
// writing, conics store {w, inf} in their last control point, triangles store {inf, inf} — followed by the enabled
// attrib values in PatchAttribs order. TrackJoinControlPoints defers each contour's first patch to CPU-side storage
// until writeDeferredStrokePatch() supplies the now-known join control point; the stroke iterator this port uses
// manages caps itself, so only the no-cap deferral lane below is reachable (a cap-producing deferral lane exists in
// other tessellators this port doesn't implement). All point math is plain element-wise float32 arithmetic (mix(a,b,T)
// == (b-a)*T + a throughout).

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// patchAllocator is the PatchWriter's PatchAllocator contract: append returns space for a single patch worth of data
// (nil on allocation failure); the provided LinearTolerances value represents the tolerances for the curve that will be
// written to the returned space.
type patchAllocator interface {
	append(tolerances *linearTolerances) []byte
}

// vertexChunkPatchAllocator is an adapter around vertexChunkBuilder that accumulates the worst-case linearTolerances
// from each append.
type vertexChunkPatchAllocator struct {
	worstCaseTolerances *linearTolerances
	builder             *vertexChunkBuilder
}

func newVertexChunkPatchAllocator(stride uint64, worstCaseTolerances *linearTolerances, target *OpFlushState, chunks *[]vertexChunk, minVerticesPerChunk int) *vertexChunkPatchAllocator {
	return &vertexChunkPatchAllocator{
		worstCaseTolerances: worstCaseTolerances,
		builder:             newVertexChunkBuilder(target, chunks, stride, minVerticesPerChunk),
	}
}

func (a *vertexChunkPatchAllocator) append(tolerances *linearTolerances) []byte {
	a.worstCaseTolerances.accumulate(tolerances)
	return a.builder.appendVertices(1)
}

func (a *vertexChunkPatchAllocator) close() { a.builder.close() }

// deferredPatchStorage holds state and deferred patch data when TrackJoinControlPoints is used. The signs of curveType
// and nP4 encode some special states:
//
//   - If both are negative, then nothing has been written that needs to be deferred.
//   - If nP4 == 0, the caller has explicitly managed the join and nothing is deferred.
//   - If nP4 < 0 and curveType >= 0, then a degenerate verb has been recorded so caps should be written when the
//     deferred patch is ended (a lane this port's stroke iterator never takes, since it produces caps itself).
//   - If nP4 > 0 and curveType >= 0, then this is a full deferred patch.
//
// Upstream also tracks the contour's first and last control points here, to synthesize caps and closes. This port's
// stroke iterator produces its own caps and closes, so neither is tracked.
type deferredPatchStorage struct {
	data      []byte  // an entire patch, except with an undefined join control point
	curveType float32 // the explicit curve type passed to writePatch
	nP4       float32 // the parametric segment value to restore on linearTolerances
}

func (d *deferredPatchStorage) disableDeferral() {
	// Do nothing if we already have a deferred patch recorded.
	d.nP4 = max32(d.nP4, 0)
}

func (d *deferredPatchStorage) mustDefer() bool { return d.nP4 < 0 }

func (d *deferredPatchStorage) hasVerb() bool { return d.curveType >= 0 }

func (d *deferredPatchStorage) hasPending() bool {
	// nP4 == 0 shouldn't happen naturally from Wang's formula and is used to disable deferring patches when the join is
	// defined manually via updateJoinControlPointAttrib.
	return d.nP4 > 0
}

func (d *deferredPatchStorage) reset() {
	d.curveType = -1
	d.nP4 = -1
}

// patchWriter emits tessellation patches for the fill and stroke configurations.
type patchWriter struct {
	alloc         patchAllocator
	deferredPatch deferredPatchStorage // only used if trackJoinControlPoints
	// Tracks the linear tolerances for the most recently written patches.
	tolerances linearTolerances
	stride     int
	// The 2x2 approximation of the local-to-device transform that will affect subsequently recorded curves (when fully
	// transformed in the vertex shader).
	approxTransform vectorXform
	colorWide       [4]float32
	fanPoint        geom.Point
	// Attribute state written after the 4 control points of a patch.
	join     geom.Point
	strokePs strokeParams
	// A maximum scale factor extracted from the current approximate transform (stroking only).
	maxScale               float32
	attribs                PatchAttribs
	colorBytes             [4]byte
	replicateLineEndPoints bool
	trackJoinControlPoints bool
	discardFlatCurves      bool
	// Feature traits, fixed at construction per configuration.
	addTrianglesWhenChopping bool
}

// newPatchWriter builds a patchWriter for a fill configuration, with the feature traits fixed by the two flags.
func newPatchWriter(attribs PatchAttribs, alloc patchAllocator, addTrianglesWhenChopping, discardFlatCurves bool) *patchWriter {
	if attribs&(PatchAttribJoinControlPoint|PatchAttribStrokeParams|PatchAttribPaintDepth|
		PatchAttribSsboIndex) != 0 {
		panic("stroke/Graphite attribs are not supported by the fill patch writer")
	}
	return &patchWriter{
		attribs:                  attribs,
		stride:                   patchStride(attribs),
		addTrianglesWhenChopping: addTrianglesWhenChopping,
		discardFlatCurves:        discardFlatCurves,
		approxTransform:          identityVectorXform(),
		maxScale:                 1,
		tolerances:               newLinearTolerances(),
		alloc:                    alloc,
	}
}

// newStrokePatchWriter builds a patchWriter for the stroke configuration: required join control point + optional
// stroke-params/color/wide-color-if-enabled/explicit-curve-type + ReplicateLineEndPoints + TrackJoinControlPoints.
func newStrokePatchWriter(attribs PatchAttribs, alloc patchAllocator) *patchWriter {
	if attribs&PatchAttribJoinControlPoint == 0 {
		panic("the stroke patch writer requires the join control point attrib")
	}
	if attribs&(PatchAttribFanPoint|PatchAttribPaintDepth|PatchAttribSsboIndex) != 0 {
		panic("fan-point/Graphite attribs are not supported by the stroke patch writer")
	}
	w := &patchWriter{
		attribs:                attribs,
		stride:                 patchStride(attribs),
		trackJoinControlPoints: true,
		replicateLineEndPoints: true,
		approxTransform:        identityVectorXform(),
		maxScale:               1,
		tolerances:             newLinearTolerances(),
		alloc:                  alloc,
	}
	w.deferredPatch.data = make([]byte, w.stride)
	w.deferredPatch.reset()
	return w
}

// setShaderTransform records the approximate local-to-device transform used to estimate tessellation tolerances. The
// max scale factor should be derived from the same matrix as xform; it's only used in stroking calculations, so can be
// ignored for filling.
func (w *patchWriter) setShaderTransform(xform vectorXform, maxScale float32) {
	w.approxTransform = xform
	w.maxScale = maxScale
}

// updateFanPointAttrib sets the point that wedges fan around.
func (w *patchWriter) updateFanPointAttrib(fanPoint geom.Point) {
	if w.attribs&PatchAttribFanPoint == 0 {
		panic("fan point attrib not enabled")
	}
	w.fanPoint = fanPoint
}

// writeDeferredStrokePatch completes a contour of a stroke by rewriting the deferred patch with now-available join
// control point information, then resets the deferral state. This port's stroke iterator manages caps on its own, so
// only joins are ever produced here (a cap-producing variant exists in other tessellators this port doesn't implement).
func (w *patchWriter) writeDeferredStrokePatch() {
	if !w.trackJoinControlPoints {
		panic("writeDeferredStrokePatch requires join tracking")
	}
	if w.deferredPatch.hasPending() {
		if !w.deferredPatch.hasVerb() {
			panic("a pending deferred patch must have recorded its verb")
		}
		// When the contour is closed, there are no caps. We just have to update the join attribute to reflect the last
		// patch and write the deferred patch. Overwrite the join control point with the updated value, which is the
		// first attribute after the 4 control points.
		patchPutPoint(w.deferredPatch.data, 4*8, w.join)
		// Assuming that the stroke parameters aren't changing within a contour, we only have to set the parametric
		// segments in order to recover the linearTolerances state at the time the deferred patch was recorded.
		w.tolerances.setParametricSegments(w.deferredPatch.nP4)
		if buf := w.alloc.append(&w.tolerances); buf != nil {
			copy(buf, w.deferredPatch.data)
		}
	}
	w.deferredPatch.reset()
}

// updateJoinControlPointAttrib sets the join control point directly; the join is then valid, so the next real patch
// need not be deferred.
func (w *patchWriter) updateJoinControlPointAttrib(lastControlPoint geom.Point) {
	if w.attribs&PatchAttribJoinControlPoint == 0 {
		panic("join control point attrib not enabled")
	}
	w.join = lastControlPoint
	w.deferredPatch.disableDeferral()
}

// updateStrokeParamsAttrib sets the stroke params written out with each patch (dynamic strokes).
func (w *patchWriter) updateStrokeParamsAttrib(params strokeParams) {
	if w.attribs&PatchAttribStrokeParams == 0 {
		panic("stroke params attrib not enabled")
	}
	w.strokePs = params
	w.tolerances.setStroke(params, w.maxScale)
}

// updateUniformStrokeParams updates tolerances to account for stroke params that are stored as uniforms instead of
// dynamic instance attributes.
func (w *patchWriter) updateUniformStrokeParams(params strokeParams) {
	if w.attribs&PatchAttribStrokeParams != 0 {
		panic("uniform stroke params require the attrib to be disabled")
	}
	w.tolerances.setStroke(params, w.maxScale)
}

// updateColorAttrib sets the color written out with each patch. The optional-wide variant stores both forms and writes
// per the runtime wide flag (VertexColor).
func (w *patchWriter) updateColorAttrib(color colorcore.PMColor4f) {
	if w.attribs&PatchAttribColor == 0 {
		panic("color attrib not enabled")
	}
	w.colorWide = [4]float32{color.R, color.G, color.B, color.A}
	toByte := func(v float32) byte {
		// Round to nearest by adding 0.5 before truncating, clamping to the byte range first.
		f := v*255 + 0.5
		if f < 0 {
			f = 0
		} else if f > 255 {
			f = 255
		}
		return byte(f)
	}
	w.colorBytes = [4]byte{toByte(color.R), toByte(color.G), toByte(color.B), toByte(color.A)}
}

//////////////////////////////////////////////////////////////////////////////
// Patch emission internals.

func patchPutF32(buf []byte, off int, v float32) int {
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(v))
	return off + 4
}

func patchPutPoint(buf []byte, off int, p geom.Point) int {
	off = patchPutF32(buf, off, p.X)
	return patchPutF32(buf, off, p.Y)
}

// emitPatchAttribs writes the enabled attribs after a patch's 4 control points (the paint-depth/ ssbo-index attribs are
// never enabled in this port). The join is a parameter because writeCircle supplies its own location instead of the
// tracked join.
func (w *patchWriter) emitPatchAttribs(buf []byte, off int, join geom.Point, explicitCurveType float32) {
	if w.attribs&PatchAttribJoinControlPoint != 0 {
		off = patchPutPoint(buf, off, join)
	}
	if w.attribs&PatchAttribFanPoint != 0 {
		off = patchPutPoint(buf, off, w.fanPoint)
	}
	if w.attribs&PatchAttribStrokeParams != 0 {
		off = patchPutF32(buf, off, w.strokePs.radius)
		off = patchPutF32(buf, off, w.strokePs.joinType)
	}
	if w.attribs&PatchAttribColor != 0 {
		if w.attribs&PatchAttribWideColorIfEnabled != 0 {
			for _, v := range w.colorWide {
				off = patchPutF32(buf, off, v)
			}
		} else {
			copy(buf[off:], w.colorBytes[:])
			off += 4
		}
	}
	if w.attribs&PatchAttribExplicitCurveType != 0 {
		off = patchPutF32(buf, off, explicitCurveType)
	}
	if off != w.stride {
		panic("patch attribs out of sync with the patch stride")
	}
}

// appendPatch returns the destination for a single patch's data — the deferred-patch CPU storage when join tracking has
// not yet seen a join for this contour, or space from the allocator otherwise (nil on allocation failure).
func (w *patchWriter) appendPatch(explicitCurveType float32) []byte {
	if w.trackJoinControlPoints && w.deferredPatch.mustDefer() {
		// Save the computed parametric segment tolerance value so that we can pass that to the allocator when flushing
		// the deferred patch.
		w.deferredPatch.curveType = explicitCurveType
		w.deferredPatch.nP4 = w.tolerances.numParametricSegmentsP4
		if w.deferredPatch.mustDefer() || !w.deferredPatch.hasPending() {
			panic("recording a deferred patch must mark it pending")
		}
		return w.deferredPatch.data
	}
	return w.alloc.append(&w.tolerances)
}

// writePatch writes a single 4-control-point patch (skipping degenerate patches), then updates the join control point
// for the next patch when join tracking is enabled.
func (w *patchWriter) writePatch(p0, p1, p2, p3 geom.Point, explicitCurveType float32) {
	if w.trackJoinControlPoints {
		// Store this even if we don't write a patch, to remember we should add caps (a lane this port's stroke iterator
		// never takes, but kept for structural fidelity with the deferral state machine).
		w.deferredPatch.curveType = explicitCurveType
	}

	last := p3
	if explicitCurveType != tessCubicCurveType {
		last = p2
	}
	// all(p0 == p1 & p1 == p2 & p2 == last): degenerate patch, so skip writing. In a regular fill or stroke this won't
	// affect rendering.
	if p0.X == p1.X && p1.X == p2.X && p2.X == last.X &&
		p0.Y == p1.Y && p1.Y == p2.Y && p2.Y == last.Y {
		return
	}
	if buf := w.appendPatch(explicitCurveType); buf != nil {
		// NOTE: w.join will be undefined if we're writing to a deferred patch. If that's the case, correct data will
		// overwrite it when the contour is closed (this is fine since a deferred patch writes to CPU memory instead of
		// directly to the GPU buffer).
		off := patchPutPoint(buf, 0, p0)
		off = patchPutPoint(buf, off, p1)
		off = patchPutPoint(buf, off, p2)
		off = patchPutPoint(buf, off, p3)
		w.emitPatchAttribs(buf, off, w.join, explicitCurveType)

		// Automatically update the join control point for the next patch.
		if w.trackJoinControlPoints {
			// Points are ordered in reverse order to get the outgoing tangent control point.
			w.join = patchTangentPoint(last, p2, p1, p0)
		}
	}
}

// patchTangentPoint returns the first point that isn't colocated with p0 (the callers pass the control points in
// reverse order to find an outgoing tangent).
func patchTangentPoint(p0, p1, p2, p3 geom.Point) geom.Point {
	switch {
	case p0 != p1:
		return p1
	case p2 != p1:
		return p2
	default:
		return p3
	}
}

// writeCircle writes a circle used for round caps and joins in stroking, encoded as a cubic with identical control
// points and an empty join. This does not use writePatch because it uses its own location as the join attribute value
// instead of the tracked join, and never defers.
func (w *patchWriter) writeCircle(p geom.Point) {
	w.tolerances.setParametricSegments(1)
	if buf := w.alloc.append(&w.tolerances); buf != nil {
		off := 0
		for range 4 { // p0,p1,p2,p3 = p -> 4 copies
			off = patchPutPoint(buf, off, p)
		}
		w.emitPatchAttribs(buf, off, p, tessCubicCurveType)
	}
}

// writeCubicPatch writes a single cubic patch with the explicit cubic curve type.
func (w *patchWriter) writeCubicPatch(p0, p1, p2, p3 geom.Point) {
	w.writePatch(p0, p1, p2, p3, tessCubicCurveType)
}

// writeQuadPatch writes a quadratic as its equivalent cubic via mix((p0,p2), p1, 2/3) on each control point.
func (w *patchWriter) writeQuadPatch(p0, p1, p2 geom.Point) {
	const twoThirds = 2.0 / 3.0
	c1 := geom.Point{X: (p1.X-p0.X)*twoThirds + p0.X, Y: (p1.Y-p0.Y)*twoThirds + p0.Y}
	c2 := geom.Point{X: (p1.X-p2.X)*twoThirds + p2.X, Y: (p1.Y-p2.Y)*twoThirds + p2.Y}
	w.writeCubicPatch(p0, c1, c2, p2)
}

// writeConicPatch writes a conic patch, encoding the weight as {w, inf} in the last control point.
func (w *patchWriter) writeConicPatch(p0, p1, p2 geom.Point, weight float32) {
	w.writePatch(p0, p1, p2, geom.Point{X: weight, Y: float32(math.Inf(1))},
		tessConicCurveType)
}

// accountForCurve records n^4 and returns 0 to signal no chopping, or clamps to the max allowed segmentation for a
// patch and returns the required number of chops to achieve visual correctness.
func (w *patchWriter) accountForCurve(n4 float32) int {
	if n4 <= tessMaxParametricSegmentsP4 {
		// Don't let n go below 1; that causes problems when we take its log (particularly reaching 0, which can happen
		// for control points that are incredibly close together).
		w.tolerances.setParametricSegments(max32(1, n4))
		return 0
	}
	w.tolerances.setParametricSegments(tessMaxParametricSegmentsP4)
	return int(geom.CeilToInt(wangsRoot4(min32(n4, tessMaxSegmentsPerCurveP4) /
		tessMaxParametricSegmentsP4)))
}

// patchMix computes (b - a)*T + a. This does not return b when T==1, but otherwise seems to get better precision than
// "a*(1 - t) + b*t" for things like chopping cubics on exact cusp points. The responsibility falls on the caller to
// check that T != 1 before calling.
func patchMix(a, b geom.Point, t float32) geom.Point {
	return geom.Point{X: (b.X-a.X)*t + a.X, Y: (b.Y-a.Y)*t + a.Y}
}

//////////////////////////////////////////////////////////////////////////////
// Public write entry points.

// writeCubic writes a cubic curve with its four control points, chopping it into multiple patches if a single patch
// cannot represent it within tolerance.
func (w *patchWriter) writeCubic(p0, p1, p2, p3 geom.Point) {
	n4 := wangsCubicP4(tessPrecision, p0, p1, p2, p3, w.approxTransform)
	if w.discardFlatCurves && n4 <= 1 {
		// This cubic only needs one segment (e.g. a line), but we're not filling space with fans or stroking, so
		// nothing actually needs to be drawn.
		return
	}
	if numPatches := w.accountForCurve(n4); numPatches != 0 {
		w.chopAndWriteCubics(p0, p1, p2, p3, numPatches)
	} else {
		w.writeCubicPatch(p0, p1, p2, p3)
	}
}

// writeConic writes a conic with three control points and a weight w, with the last coord of the last control point
// signaling a conic by being set to infinity.
func (w *patchWriter) writeConic(p0, p1, p2 geom.Point, weight float32) {
	n2 := wangsConicP2(tessPrecision, p0, p1, p2, weight, w.approxTransform)
	if w.discardFlatCurves && n2 <= 1 {
		return
	}
	if numPatches := w.accountForCurve(n2 * n2); numPatches != 0 {
		w.chopAndWriteConics(p0, p1, p2, weight, numPatches)
	} else {
		w.writeConicPatch(p0, p1, p2, weight)
	}
}

// writeQuadratic writes a quadratic, automatically converting its three control points into an equivalent cubic.
func (w *patchWriter) writeQuadratic(p0, p1, p2 geom.Point) {
	n4 := wangsQuadraticP4(tessPrecision, p0, p1, p2, w.approxTransform)
	if w.discardFlatCurves && n4 <= 1 {
		return
	}
	if numPatches := w.accountForCurve(n4); numPatches != 0 {
		w.chopAndWriteQuads(p0, p1, p2, numPatches)
	} else {
		w.writeQuadPatch(p0, p1, p2)
	}
}

// writeLine writes a line, converted into an equivalent cubic. With ReplicateLineEndPoints the cubic is {a, a, b, b} —
// visually still a line, but 't' does not move linearly over it, so Wang's formula is more pessimistic; shaders avoid
// evaluating Wang's formula when a patch has control points in this arrangement (this bypasses numerical stability
// issues in Wang's formula for very large control point coordinates). Otherwise the truly linear cubic {a, 2/3a + 1/3b,
// 1/3a + 2/3b, b} is written; in exact math that structure should have Wang's formula return 0, but due to floating
// point math this isn't always the case, so shaders need some way to restrict the number of parametric segments if
// Wang's formula numerically blows up.
func (w *patchWriter) writeLine(p0, p1 geom.Point) {
	if w.discardFlatCurves {
		return
	}
	// No chopping needed; a line only ever requires one segment (the minimum already).
	w.tolerances.setParametricSegments(1)
	if w.replicateLineEndPoints {
		w.writeCubicPatch(p0, p0, p1, p1)
		return
	}
	const third = 1.0 / 3.0
	c1 := geom.Point{X: (p1.X-p0.X)*third + p0.X, Y: (p1.Y-p0.Y)*third + p0.Y}
	c2 := geom.Point{X: (p0.X-p1.X)*third + p1.X, Y: (p0.Y-p1.Y)*third + p1.Y}
	w.writeCubicPatch(p0, c1, c2, p1)
}

// writeTriangle writes a triangle encoded as a conic with w=Inf, using a distinct explicit curve type for when inf
// isn't supported in shaders.
func (w *patchWriter) writeTriangle(p0, p1, p2 geom.Point) {
	// No chopping needed; the max supported segment count always supports 2 lines (which form a triangle when
	// implicitly closed).
	const triangleSegmentsP4 = 2 * 2 * 2 * 2
	w.tolerances.setParametricSegments(triangleSegmentsP4)
	inf := float32(math.Inf(1))
	w.writePatch(p0, p1, p2, geom.Point{X: inf, Y: inf}, tessTriangularConicCurveType)
}

//////////////////////////////////////////////////////////////////////////////
// Chopping.

// chopAndWriteQuads chops into numPatches parametrically uniform quads. It is assumed that numPatches is calculated
// such that the resulting curves require the maximum number of segments to draw appropriately (since the original
// presumably needed even more).
func (w *patchWriter) chopAndWriteQuads(p0, p1, p2 geom.Point, numPatches int) {
	var triangulator *middleOutPolygonTriangulator
	emit := func(a, b, c geom.Point) { w.writeTriangle(a, b, c) }
	if w.addTrianglesWhenChopping {
		triangulator = newMiddleOutPolygonTriangulator(numPatches, p0)
	}
	for ; numPatches >= 3; numPatches -= 2 {
		// Chop into 3 quads at T = {1/numPatches, 2/numPatches} (the float4 lanes hold {T1, T1, T2, T2}).
		t1 := 1 / float32(numPatches)
		t2 := 2 / float32(numPatches)
		ab1, ab2 := patchMix(p0, p1, t1), patchMix(p0, p1, t2)
		bc1, bc2 := patchMix(p1, p2, t1), patchMix(p1, p2, t2)
		abc1, abc2 := patchMix(ab1, bc1, t1), patchMix(ab2, bc2, t2)
		// p1 & p2 of the cubic representation of the middle quad: middle = mix(ab, bc, mix(T, T.zwxy(), 2/3)).
		const twoThirds = 2.0 / 3.0
		m1 := (t2-t1)*twoThirds + t1
		m2 := (t1-t2)*twoThirds + t2
		mid1 := patchMix(ab1, bc1, m1)
		mid2 := patchMix(ab2, bc2, m2)

		w.writeQuadPatch(p0, ab1, abc1) // Write the 1st quad.
		if w.addTrianglesWhenChopping {
			w.writeTriangle(p0, abc1, abc2)
		}
		w.writeCubicPatch(abc1, mid1, mid2, abc2) // Write the 2nd quad (already a cubic).
		if w.addTrianglesWhenChopping {
			triangulator.pushVertex(abc2, emit)
		}
		p0, p1 = abc2, bc2 // Save the 3rd quad.
	}
	if numPatches == 2 {
		// Chop into 2 quads.
		ab := geom.Point{X: (p0.X + p1.X) * 0.5, Y: (p0.Y + p1.Y) * 0.5}
		bc := geom.Point{X: (p1.X + p2.X) * 0.5, Y: (p1.Y + p2.Y) * 0.5}
		abc := geom.Point{X: (ab.X + bc.X) * 0.5, Y: (ab.Y + bc.Y) * 0.5}

		w.writeQuadPatch(p0, ab, abc) // Write the 1st quad.
		if w.addTrianglesWhenChopping {
			w.writeTriangle(p0, abc, p2)
		}
		w.writeQuadPatch(abc, bc, p2) // Write the 2nd quad.
	} else {
		if numPatches != 1 {
			panic("chopAndWriteQuads bookkeeping error")
		}
		w.writeQuadPatch(p0, p1, p2) // Write the single remaining quad.
	}
	if w.addTrianglesWhenChopping {
		triangulator.pushVertex(p2, emit)
		triangulator.close(emit)
	}
}

// chopAndWriteConics chops the conic into numPatches patches in homogeneous (unprojected) space.
func (w *patchWriter) chopAndWriteConics(p0, p1, p2 geom.Point, weight float32, numPatches int) {
	var triangulator *middleOutPolygonTriangulator
	emit := func(a, b, c geom.Point) { w.writeTriangle(a, b, c) }
	if w.addTrianglesWhenChopping {
		triangulator = newMiddleOutPolygonTriangulator(numPatches, p0)
	}
	// Load the conic in 3d homogeneous (unprojected) space (the fourth float4 lane is unused).
	h0 := [3]float32{p0.X, p0.Y, 1}
	h1 := [3]float32{p1.X * weight, p1.Y * weight, weight}
	h2 := [3]float32{p2.X, p2.Y, 1}
	mix3 := func(a, b [3]float32, t float32) [3]float32 {
		return [3]float32{(b[0]-a[0])*t + a[0], (b[1]-a[1])*t + a[1], (b[2]-a[2])*t + a[2]}
	}
	for ; numPatches >= 2; numPatches-- {
		// Chop in homogeneous space.
		t := 1 / float32(numPatches)
		ab := mix3(h0, h1, t)
		bc := mix3(h1, h2, t)
		abc := mix3(ab, bc, t)

		// Project and write the 1st conic.
		midpoint := geom.Point{X: abc[0] / abc[2], Y: abc[1] / abc[2]}
		w.writeConicPatch(
			geom.Point{X: h0[0] / h0[2], Y: h0[1] / h0[2]},
			geom.Point{X: ab[0] / ab[2], Y: ab[1] / ab[2]},
			midpoint,
			ab[2]/sqrt32(h0[2]*abc[2]),
		)
		if w.addTrianglesWhenChopping {
			triangulator.pushVertex(midpoint, emit)
		}
		h0, h1 = abc, bc // Save the 2nd conic (in homogeneous space).
	}
	// Project and write the remaining conic.
	if numPatches != 1 {
		panic("chopAndWriteConics bookkeeping error")
	}
	last := geom.Point{X: h2[0], Y: h2[1]} // h2.w == 1
	w.writeConicPatch(
		geom.Point{X: h0[0] / h0[2], Y: h0[1] / h0[2]},
		geom.Point{X: h1[0] / h1[2], Y: h1[1] / h1[2]},
		last,
		h1[2]/sqrt32(h0[2]),
	)
	if w.addTrianglesWhenChopping {
		triangulator.pushVertex(last, emit)
		triangulator.close(emit)
	}
}

// chopAndWriteCubics chops the cubic into numPatches parametrically uniform cubic patches.
func (w *patchWriter) chopAndWriteCubics(p0, p1, p2, p3 geom.Point, numPatches int) {
	var triangulator *middleOutPolygonTriangulator
	emit := func(a, b, c geom.Point) { w.writeTriangle(a, b, c) }
	if w.addTrianglesWhenChopping {
		triangulator = newMiddleOutPolygonTriangulator(numPatches, p0)
	}
	for ; numPatches >= 3; numPatches -= 2 {
		// Chop into 3 cubics at T = {1/numPatches, 2/numPatches}.
		t1 := 1 / float32(numPatches)
		t2 := 2 / float32(numPatches)
		ab1, ab2 := patchMix(p0, p1, t1), patchMix(p0, p1, t2)
		bc1, bc2 := patchMix(p1, p2, t1), patchMix(p1, p2, t2)
		cd1, cd2 := patchMix(p2, p3, t1), patchMix(p2, p3, t2)
		abc1, abc2 := patchMix(ab1, bc1, t1), patchMix(ab2, bc2, t2)
		bcd1, bcd2 := patchMix(bc1, cd1, t1), patchMix(bc2, cd2, t2)
		abcd1, abcd2 := patchMix(abc1, bcd1, t1), patchMix(abc2, bcd2, t2)
		// p1 & p2 of the middle cubic: middle = mix(abc, bcd, T.zwxy()).
		mid1 := patchMix(abc1, bcd1, t2)
		mid2 := patchMix(abc2, bcd2, t1)

		w.writeCubicPatch(p0, ab1, abc1, abcd1) // Write the 1st cubic.
		if w.addTrianglesWhenChopping {
			w.writeTriangle(p0, abcd1, abcd2)
		}
		w.writeCubicPatch(abcd1, mid1, mid2, abcd2) // Write the 2nd cubic.
		if w.addTrianglesWhenChopping {
			triangulator.pushVertex(abcd2, emit)
		}
		p0, p1, p2 = abcd2, bcd2, cd2 // Save the 3rd cubic.
	}
	if numPatches == 2 {
		// Chop into 2 cubics.
		ab := geom.Point{X: (p0.X + p1.X) * 0.5, Y: (p0.Y + p1.Y) * 0.5}
		bc := geom.Point{X: (p1.X + p2.X) * 0.5, Y: (p1.Y + p2.Y) * 0.5}
		cd := geom.Point{X: (p2.X + p3.X) * 0.5, Y: (p2.Y + p3.Y) * 0.5}
		abc := geom.Point{X: (ab.X + bc.X) * 0.5, Y: (ab.Y + bc.Y) * 0.5}
		bcd := geom.Point{X: (bc.X + cd.X) * 0.5, Y: (bc.Y + cd.Y) * 0.5}
		abcd := geom.Point{X: (abc.X + bcd.X) * 0.5, Y: (abc.Y + bcd.Y) * 0.5}

		w.writeCubicPatch(p0, ab, abc, abcd) // Write the 1st cubic.
		if w.addTrianglesWhenChopping {
			w.writeTriangle(p0, abcd, p3)
		}
		w.writeCubicPatch(abcd, bcd, cd, p3) // Write the 2nd cubic.
	} else {
		if numPatches != 1 {
			panic("chopAndWriteCubics bookkeeping error")
		}
		w.writeCubicPatch(p0, p1, p2, p3) // Write the single remaining cubic.
	}
	if w.addTrianglesWhenChopping {
		triangulator.pushVertex(p3, emit)
		triangulator.close(emit)
	}
}
