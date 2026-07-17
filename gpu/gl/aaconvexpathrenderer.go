// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Coverage-AA convex fills rendered by decomposing the path's boundary into line/quad segments, fanning the interior
// from the center of mass, and computing edge coverage in the fragment shader from the signed distance to each segment
// (lines via a degenerate quad, quads via the Loop-Blinn u^2 - v canonical form in the quadEdgeEffect geometry
// processor).

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
)

// aaConvexClose is the vertex-fusing tolerance, in device space.
const aaConvexClose = float32(1) / 16

const aaConvexCloseSqd = aaConvexClose * aaConvexClose

// aaConvexSegment is a line (one point) or quad (two points) piece of the path boundary. norms are the normals to the
// edge ending at each point; mid is the normalized outward bisector where the previous segment meets this one.
type aaConvexSegment struct {
	// typ is 0 for a line, 1 for a quad (countPoints/endPt assume these values).
	typ   int
	pts   [2]geom.Point
	norms [2]geom.Point
	mid   geom.Point
}

func (s *aaConvexSegment) countPoints() int { return s.typ + 1 }

func (s *aaConvexSegment) endPt() geom.Point { return s.pts[s.typ] }

func (s *aaConvexSegment) endNorm() geom.Point { return s.norms[s.typ] }

// aaConvexCenterOfMass computes the polygon's center of mass, used as the fan point for the interior triangulation. If
// the polygon has no area, it instead returns the average of its points.
func aaConvexCenterOfMass(segments []aaConvexSegment) (c geom.Point, ok bool) {
	area := float32(0)
	center := geom.Point{}
	count := len(segments)
	if count <= 0 {
		return c, false
	}
	p0 := geom.Point{}
	if count > 2 {
		// We translate the polygon so that the first point is at the origin. This avoids some precision issues with
		// small area polygons far away from the origin.
		p0 = segments[0].endPt()
		// The first and last iteration of the below loop would compute zeros since the starting / ending point is
		// (0,0). So instead we start at i=1 and make the last iteration i=count-2.
		pj := segments[1].endPt().Sub(p0)
		for i := 1; i < count-1; i++ {
			pi := pj
			pj = segments[i+1].endPt().Sub(p0)

			t := pi.Cross(pj)
			area += t
			center.X += (pi.X + pj.X) * t
			center.Y += (pi.Y + pj.Y) * t
		}
	}

	// If the poly has no area then we instead return the average of its points.
	if geom.ScalarNearlyZero(area) {
		var avg geom.Point
		for i := 0; i < count; i++ {
			pt := segments[i].endPt()
			avg.X += pt.X
			avg.Y += pt.Y
		}
		denom := 1 / float32(count)
		c = avg.Scaled(denom)
	} else {
		area *= 3
		area = geom.ScalarInvert(area)
		// Undo the translate of p0 to the origin.
		c = center.Scaled(area).Add(p0)
	}
	return c, !math.IsNaN(float64(c.X)) && !math.IsNaN(float64(c.Y)) && c.IsFinite()
}

// aaConvexComputeVectors computes the fan point and, for every segment, the outward-facing normals and the bisector
// where it meets the previous segment; it also tallies the vertex and index counts the triangulation will need.
func aaConvexComputeVectors(segments []aaConvexSegment, fanPt *geom.Point, dir path.FirstDirection, vCount, iCount *int) bool {
	c, ok := aaConvexCenterOfMass(segments)
	if !ok {
		return false
	}
	*fanPt = c
	count := len(segments)

	// Make the normals point towards the outside.
	normSide := geom.PointSideLeft
	if dir == path.FirstDirectionCCW {
		normSide = geom.PointSideRight
	}

	var vCount64, iCount64 int64
	// Compute normals at all points.
	for a := 0; a < count; a++ {
		sega := &segments[a]
		b := (a + 1) % count
		segb := &segments[b]

		prevPt := sega.endPt()
		n := segb.countPoints()
		for p := 0; p < n; p++ {
			norm := segb.pts[p].Sub(prevPt)
			norm.Normalize()
			segb.norms[p] = norm.MakeOrthog(normSide)
			prevPt = segb.pts[p]
		}
		if segb.typ == 0 {
			vCount64 += 5
			iCount64 += 9
		} else {
			vCount64 += 6
			iCount64 += 12
		}
	}

	// Compute mid-vectors where segments meet.
	for a := 0; a < count; a++ {
		sega := &segments[a]
		b := (a + 1) % count
		segb := &segments[b]
		mid := segb.norms[0].Add(sega.endNorm())
		mid.Normalize()
		segb.mid = mid
		// Corner wedges.
		vCount64 += 4
		iCount64 += 6
	}
	if vCount64 > math.MaxInt32 || iCount64 > math.MaxInt32 {
		return false
	}
	*vCount = int(vCount64)
	*iCount = int(iCount64)
	return true
}

// aaConvexDegenerateStage is the progress of the degenerate-path test as points are fed to it.
type aaConvexDegenerateStage uint8

const (
	aaConvexDegenerateInitial aaConvexDegenerateStage = iota
	aaConvexDegeneratePoint
	aaConvexDegenerateLine
	aaConvexNonDegenerate
)

// aaConvexDegenerateData incrementally tests whether a path's points are all coincident or all collinear (both of which
// yield a zero-area shape that this renderer must reject).
type aaConvexDegenerateData struct {
	stage      aaConvexDegenerateStage
	firstPoint geom.Point
	lineNormal geom.Point
	lineC      float32
}

func (d *aaConvexDegenerateData) isDegenerate() bool { return d.stage != aaConvexNonDegenerate }

// update feeds the next path point into the degenerate-path test.
func (d *aaConvexDegenerateData) update(pt geom.Point) {
	switch d.stage {
	case aaConvexDegenerateInitial:
		d.firstPoint = pt
		d.stage = aaConvexDegeneratePoint
	case aaConvexDegeneratePoint:
		if pt.DistanceToSqd(d.firstPoint) > aaConvexCloseSqd {
			normal := pt.Sub(d.firstPoint)
			normal.Normalize()
			d.lineNormal = normal.MakeOrthog(geom.PointSideLeft)
			d.lineC = -d.lineNormal.Dot(d.firstPoint)
			d.stage = aaConvexDegenerateLine
		}
	case aaConvexDegenerateLine:
		if geom.ScalarAbs(d.lineNormal.Dot(pt)+d.lineC) > aaConvexClose {
			d.stage = aaConvexNonDegenerate
		}
	case aaConvexNonDegenerate:
	}
}

// oppositeFirstDirection returns the reverse of dir, or Unknown if dir is Unknown.
func oppositeFirstDirection(dir path.FirstDirection) path.FirstDirection {
	switch dir {
	case path.FirstDirectionCW:
		return path.FirstDirectionCCW
	case path.FirstDirectionCCW:
		return path.FirstDirectionCW
	default:
		return path.FirstDirectionUnknown
	}
}

// aaConvexGetDirection returns the path's winding direction as seen after applying m, accounting for m reversing
// orientation.
func aaConvexGetDirection(p *path.Path, m *geom.Matrix) (path.FirstDirection, bool) {
	// At this point, we've already returned true from canDraw(), which checked that the path's direction could be
	// determined, so this should just be fetching the cached direction. However, if perspective is involved, we're
	// operating on a transformed path, which may no longer have a computable direction.
	dir := p.ComputeFirstDirection()
	if dir == path.FirstDirectionUnknown {
		return dir, false
	}

	// Check whether m reverses the orientation.
	if m.HasPerspective() {
		panic("aaConvexGetDirection requires a non-perspective matrix")
	}
	det2x2 := m.Get(0)*m.Get(4) - m.Get(1)*m.Get(3)
	if det2x2 < 0 {
		dir = oppositeFirstDirection(dir)
	}
	return dir, true
}

// allPointsEq reports whether every point in pts equals the first one.
func allPointsEq(pts []geom.Point) bool {
	for i := 1; i < len(pts); i++ {
		if pts[0] != pts[i] {
			return false
		}
	}
	return true
}

// aaConvexAddLineToSegment appends a line segment ending at pt.
func aaConvexAddLineToSegment(pt geom.Point, segments *[]aaConvexSegment) {
	*segments = append(*segments, aaConvexSegment{typ: 0, pts: [2]geom.Point{pt}})
}

// aaConvexAddQuadSegment appends a quad segment, collapsing it to a line segment if the control point is close enough
// to the chord between the endpoints.
func aaConvexAddQuadSegment(pts *[3]geom.Point, segments *[]aaConvexSegment) {
	if distanceToLineSegmentBetweenSqd(pts[1], pts[0], pts[2]) < aaConvexCloseSqd {
		if pts[0] != pts[2] {
			aaConvexAddLineToSegment(pts[2], segments)
		}
	} else {
		*segments = append(*segments, aaConvexSegment{
			typ: 1,
			pts: [2]geom.Point{pts[1], pts[2]},
		})
	}
}

// aaConvexAddCubicSegments approximates a cubic with a sequence of quads (constrained to the cubic's tangents) and
// appends each as a quad segment.
func aaConvexAddCubicSegments(pts *[4]geom.Point, dir path.FirstDirection, segments *[]aaConvexSegment) {
	quads := make([]geom.Point, 0, 15)
	convertCubicToQuadsConstrainToTangents(pts, 1, dir, &quads)
	for q := 0; q+2 < len(quads); q += 3 {
		var quad [3]geom.Point
		copy(quad[:], quads[q:q+3])
		aaConvexAddQuadSegment(&quad, segments)
	}
}

// aaConvexGetSegments walks the transformed path, converting each verb into line/quad segments and tracking whether the
// path degenerates to a point or a line (in which case it returns false).
func aaConvexGetSegments(p *path.Path, m *geom.Matrix, segments *[]aaConvexSegment, fanPt *geom.Point, vCount, iCount *int) bool {
	// This renderer over-emphasizes very thin path regions. We use the distance to the path from the sample to compute
	// coverage. Every pixel intersected by the path will be hit and the maximum distance is sqrt(2)/2. We don't notice
	// that the sample may be close to a very thin area of the path and thus should be very light. This is particularly
	// egregious for degenerate line paths. We detect paths that are very close to a line (zero area) and draw nothing.
	var degenerateData aaConvexDegenerateData
	dir, ok := aaConvexGetDirection(p, m)
	if !ok {
		return false
	}

	it := path.NewIter(p, true)
	for {
		verb, pts, iterOK := it.Next()
		if !iterOK {
			break
		}
		switch verb {
		case path.VerbMove:
			degenerateData.update(m.MapPoint(pts[0]))
		case path.VerbLine:
			if !allPointsEq(pts[:2]) {
				dst := m.MapPoint(pts[1])
				degenerateData.update(dst)
				aaConvexAddLineToSegment(dst, segments)
			}
		case path.VerbQuad:
			if !allPointsEq(pts[:3]) {
				var dst [3]geom.Point
				m.MapPoints(dst[:], pts[:3])
				degenerateData.update(dst[1])
				degenerateData.update(dst[2])
				aaConvexAddQuadSegment(&dst, segments)
			}
		case path.VerbConic:
			if !allPointsEq(pts[:3]) {
				var dst [3]geom.Point
				m.MapPoints(dst[:], pts[:3])
				// Approximate the conic with quads at tolerance 0.25.
				conic := geom.MakeConic(dst[0], dst[1], dst[2], it.ConicWeight())
				pow2 := conic.ComputeQuadPOW2(0.25)
				quadPts := make([]geom.Point, 1+2*(1<<pow2))
				count := conic.ChopIntoQuadsPOW2(quadPts, pow2)
				for i := 0; i < count; i++ {
					degenerateData.update(quadPts[2*i+1])
					degenerateData.update(quadPts[2*i+2])
					var quad [3]geom.Point
					copy(quad[:], quadPts[2*i:2*i+3])
					aaConvexAddQuadSegment(&quad, segments)
				}
			}
		case path.VerbCubic:
			if !allPointsEq(pts[:4]) {
				var dst [4]geom.Point
				m.MapPoints(dst[:], pts[:4])
				degenerateData.update(dst[1])
				degenerateData.update(dst[2])
				degenerateData.update(dst[3])
				aaConvexAddCubicSegments(&dst, dir, segments)
			}
		case path.VerbClose:
		}
	}

	if degenerateData.isDegenerate() {
		return false
	}
	return aaConvexComputeVectors(*segments, fanPt, dir, vCount, iCount)
}

// aaConvexDraw tracks the vertex/index count of one draw call within a batch (a batch may split into multiple draws
// when the index range would overflow a uint16).
type aaConvexDraw struct {
	vertexCnt int
	indexCnt  int
}

// aaConvexStableLargeNegativeValue is a negative value that is very large so it won't affect results if interpolated
// with, but not so large a negative that it affects numerical precision on less powerful GPUs.
const aaConvexStableLargeNegativeValue = -geom.ScalarMax / 1000000

// aaConvexCreateVertices builds the triangulation for the segments' fan and edge coverage. verts writes the interleaved
// vertex data (position, color, UV, D0, D1); idxs receives draw-relative index values.
func aaConvexCreateVertices(segments []aaConvexSegment, fanPt geom.Point, color colorcore.PMColor4f, wideColor bool, draws *[]aaConvexDraw, verts *quadVertexWriter, idxs []uint16) {
	*draws = append(*draws, aaConvexDraw{})
	draw := &(*draws)[len(*draws)-1]
	idxBase := 0 // The running offset of the current draw's indices within idxs.

	putVertex := func(pos, uv geom.Point, d0, d1 float32) {
		verts.putF32(pos.X)
		verts.putF32(pos.Y)
		verts.putColor(color, wideColor)
		verts.putF32(uv.X)
		verts.putF32(uv.Y)
		verts.putF32(d0)
		verts.putF32(d1)
	}

	count := len(segments)
	for a := 0; a < count; a++ {
		sega := &segments[a]
		b := (a + 1) % count
		segb := &segments[b]

		// Check whether adding the verts for this segment to the current draw would cause index values to overflow.
		vCount := 4
		if segb.typ == 0 {
			vCount += 5
		} else {
			vCount += 6
		}
		if draw.vertexCnt+vCount > (1 << 16) {
			idxBase += draw.indexCnt
			*draws = append(*draws, aaConvexDraw{})
			draw = &(*draws)[len(*draws)-1]
		}
		v := &draw.vertexCnt
		i := &draw.indexCnt

		const negOne = float32(-1)

		// These tris are inset in the 1 unit arc around the corner.
		p0 := sega.endPt()
		// Position, Color, UV, D0, D1.
		putVertex(p0, geom.Point{}, negOne, negOne)
		putVertex(p0.Add(sega.endNorm()), geom.Point{Y: -1}, negOne, negOne)
		putVertex(p0.Add(segb.mid), geom.Point{Y: -1}, negOne, negOne)
		putVertex(p0.Add(segb.norms[0]), geom.Point{Y: -1}, negOne, negOne)

		idxs[idxBase+*i+0] = uint16(*v + 0)
		idxs[idxBase+*i+1] = uint16(*v + 2)
		idxs[idxBase+*i+2] = uint16(*v + 1)
		idxs[idxBase+*i+3] = uint16(*v + 0)
		idxs[idxBase+*i+4] = uint16(*v + 3)
		idxs[idxBase+*i+5] = uint16(*v + 2)

		*v += 4
		*i += 6

		if segb.typ == 0 {
			// We draw the line edge as a degenerate quad (u is 0, v is the signed distance to the edge).
			v1Pos := sega.endPt()
			v2Pos := segb.pts[0]
			dist := fanPt.DistanceToLineBetween(v1Pos, v2Pos)

			putVertex(fanPt, geom.Point{Y: dist}, negOne, negOne)
			putVertex(v1Pos, geom.Point{}, negOne, negOne)
			putVertex(v2Pos, geom.Point{}, negOne, negOne)
			putVertex(v1Pos.Add(segb.norms[0]), geom.Point{Y: -1}, negOne, negOne)
			putVertex(v2Pos.Add(segb.norms[0]), geom.Point{Y: -1}, negOne, negOne)

			idxs[idxBase+*i+0] = uint16(*v + 3)
			idxs[idxBase+*i+1] = uint16(*v + 1)
			idxs[idxBase+*i+2] = uint16(*v + 2)

			idxs[idxBase+*i+3] = uint16(*v + 4)
			idxs[idxBase+*i+4] = uint16(*v + 3)
			idxs[idxBase+*i+5] = uint16(*v + 2)

			*i += 6

			// Draw the interior fan if it exists. TODO: detect and combine colinear segments. This will ensure we catch
			// every case with no interior, and that the resulting shared edge uses the same endpoints.
			if count >= 3 {
				idxs[idxBase+*i+0] = uint16(*v + 0)
				idxs[idxBase+*i+1] = uint16(*v + 2)
				idxs[idxBase+*i+2] = uint16(*v + 1)

				*i += 3
			}

			*v += 5
		} else {
			qpts := [3]geom.Point{sega.endPt(), segb.pts[0], segb.pts[1]}

			c0 := segb.norms[0].Dot(qpts[0])
			c1 := segb.norms[1].Dot(qpts[2])

			midVec := segb.norms[0].Add(segb.norms[1])
			midVec.Normalize()

			pos := [6]geom.Point{
				fanPt,
				qpts[0],
				qpts[2],
				qpts[0].Add(segb.norms[0]),
				qpts[2].Add(segb.norms[1]),
				qpts[1].Add(midVec),
			}
			var uv [6]geom.Point
			var toUV quadUVMatrix
			toUV.set(&qpts)
			toUV.apply(pos[:], uv[:])

			putVertex(pos[0], uv[0], -segb.norms[0].Dot(fanPt)+c0, -segb.norms[1].Dot(fanPt)+c1)
			putVertex(pos[1], uv[1], 0, -segb.norms[1].Dot(qpts[0])+c1)
			putVertex(pos[2], uv[2], -segb.norms[0].Dot(qpts[2])+c0, 0)
			putVertex(pos[3], uv[3], aaConvexStableLargeNegativeValue,
				aaConvexStableLargeNegativeValue)
			putVertex(pos[4], uv[4], aaConvexStableLargeNegativeValue,
				aaConvexStableLargeNegativeValue)
			putVertex(pos[5], uv[5], aaConvexStableLargeNegativeValue,
				aaConvexStableLargeNegativeValue)

			idxs[idxBase+*i+0] = uint16(*v + 3)
			idxs[idxBase+*i+1] = uint16(*v + 1)
			idxs[idxBase+*i+2] = uint16(*v + 2)
			idxs[idxBase+*i+3] = uint16(*v + 4)
			idxs[idxBase+*i+4] = uint16(*v + 3)
			idxs[idxBase+*i+5] = uint16(*v + 2)

			idxs[idxBase+*i+6] = uint16(*v + 5)
			idxs[idxBase+*i+7] = uint16(*v + 3)
			idxs[idxBase+*i+8] = uint16(*v + 4)

			*i += 9

			// Draw the interior fan if it exists (see the TODO above).
			if count >= 3 {
				idxs[idxBase+*i+0] = uint16(*v + 0)
				idxs[idxBase+*i+1] = uint16(*v + 2)
				idxs[idxBase+*i+2] = uint16(*v + 1)

				*i += 3
			}

			*v += 6
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// QuadEdgeEffect

// quadEdgeEffect is a geometry processor for a quadratic specified by 0 = u^2 - v canonical coords, u and v being the
// first two components of the vertex attribute. Coverage is based on signed distance with negative being inside,
// positive outside. The edge is specified in window space (y-down). If either the third or fourth component of the
// interpolated vertex coord is > 0 then the pixel is considered outside the edge; this is used to attempt to trim to a
// portion of the infinite quad. Requires shader derivative instruction support.
type quadEdgeEffect struct {
	attrs [3]Attribute // inPosition, inColor, inQuadEdge
	GPBase
	localMatrix     geom.Matrix
	usesLocalCoords bool
}

// makeQuadEdgeEffect builds a quadEdgeEffect geometry processor.
func makeQuadEdgeEffect(localMatrix *geom.Matrix, usesLocalCoords, wideColor bool) GeometryProcessor {
	gp := &quadEdgeEffect{localMatrix: *localMatrix, usesLocalCoords: usesLocalCoords}
	gp.initGP(QuadEdgeEffectClassID)
	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	gp.attrs[1] = MakeColorAttribute("inColor", wideColor)
	// float4 is used for extra precision headroom on GPUs that need it for the quadedge attributes.
	gp.attrs[2] = MakeAttribute("inQuadEdge", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])
	return gp
}

func (g *quadEdgeEffect) Name() string { return "QuadEdge" }

// AddToKey adds this geometry processor's shader-affecting state to the program key.
func (g *quadEdgeEffect) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBool(g.usesLocalCoords, "usesLocalCoords")
	b.AddBits(matrixKeyBits, ComputeMatrixKey(caps, &g.localMatrix), "localMatrixType")
}

func (g *quadEdgeEffect) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &quadEdgeEffectImpl{}
}

// quadEdgeEffectImpl is the shader implementation for quadEdgeEffect.
type quadEdgeEffectImpl struct {
	GPImplBase
	localMatrixPrev    geom.Matrix
	localMatrixPrevSet bool
	localMatrixUniform UniformHandle
}

func (i *quadEdgeEffectImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	qe := geomProc.(*quadEdgeEffect)
	SetTransform(pdman, caps, i.localMatrixUniform, &qe.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)
}

func (i *quadEdgeEffectImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	qe := args.GeomProc.(*quadEdgeEffect)
	vertBuilder := args.VertBuilder
	fragBuilder := args.FragBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler

	// Emit attributes.
	varyingHandler.EmitAttributes(qe)

	v := NewVarying(GLSLTypeFloat4)
	varyingHandler.AddVarying("QuadEdge", &v)
	vertBuilder.CodeAppendf("%s = %s;", v.VsOut(), qe.attrs[2].Name())

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	varyingHandler.AddPassThroughAttribute(qe.attrs[1].AsShaderVar(), args.OutputColor,
		InterpolationInterpolated)

	// Set up position.
	WriteOutputPosition(vertBuilder, gpArgs, qe.attrs[0].Name())
	if qe.usesLocalCoords {
		WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
			qe.attrs[0].AsShaderVar(), &qe.localMatrix, &i.localMatrixUniform)
	}

	fragBuilder.CodeAppend("float edgeAlpha;")

	// Keep the derivative instructions outside the conditional.
	fragBuilder.CodeAppendf("vec2 duvdx = vec2(dFdx(%s.xy));", v.FsIn())
	fragBuilder.CodeAppendf("vec2 duvdy = vec2(dFdy(%s.xy));", v.FsIn())
	fragBuilder.CodeAppendf("if (%s.z > 0.0 && %s.w > 0.0) {", v.FsIn(), v.FsIn())
	// Today we know z and w are in device space. We could use derivatives.
	fragBuilder.CodeAppendf("edgeAlpha = min(min(%s.z, %s.w) + 0.5, 1.0);", v.FsIn(), v.FsIn())
	fragBuilder.CodeAppend("} else {")
	fragBuilder.CodeAppendf("vec2 gF = vec2(2.0*%s.x*duvdx.x - duvdx.y,"+
		"                 2.0*%s.x*duvdy.x - duvdy.y);", v.FsIn(), v.FsIn())
	fragBuilder.CodeAppendf("edgeAlpha = %s.x*%s.x - %s.y;", v.FsIn(), v.FsIn(), v.FsIn())
	fragBuilder.CodeAppend("edgeAlpha = clamp(0.5 - edgeAlpha / length(gF), 0.0, 1.0);}")

	fragBuilder.CodeAppendf("vec4 %s = vec4(edgeAlpha);", args.OutputCoverage)
}

//////////////////////////////////////////////////////////////////////////////
// AAConvexPathOp

var aaConvexPathOpClassID = GenOpClassID()

// aaConvexPathData holds one path's per-draw state within a batched aaConvexPathOp.
type aaConvexPathData struct {
	path       *path.Path
	viewMatrix geom.Matrix
	color      colorcore.PMColor4f
}

// aaConvexPathOp draws one or more coverage-AA convex path fills, batched together when possible.
type aaConvexPathOp struct {
	drawOpNoClipToShape
	programInfo *ProgramInfo
	helper      SimpleMeshDrawOpHelper
	paths       []aaConvexPathData
	meshes      []*simpleMesh
	OpBase
	wideColor bool
}

// newAAConvexPathOp creates an op to fill p with paint under viewMatrix. The paint is consumed.
func newAAConvexPathOp(paint *Paint, viewMatrix *geom.Matrix, p *path.Path, stencilSettings *UserStencilSettings) DrawOp {
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := &aaConvexPathOp{}
	o.InitOp(aaConvexPathOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, stencilSettings,
		PipelineInputFlagNone)
	o.paths = append(o.paths, aaConvexPathData{viewMatrix: *viewMatrix, path: p, color: color})
	devBounds, _ := viewMatrix.MapRect(p.Bounds())
	o.SetBounds(devBounds, HasAABloat(true), IsHairline(false))
	return o
}

// Name implements Op.
func (o *aaConvexPathOp) Name() string { return "AAConvexPathOp" }

// VisitProxies implements Op.
func (o *aaConvexPathOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *aaConvexPathOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *aaConvexPathOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *aaConvexPathOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.paths[len(o.paths)-1].color, &o.wideColor)
}

// createProgramInfo builds the program info for this op's draw, if not already built.
func (o *aaConvexPathOp) createProgramInfo(state *OpFlushState) {
	invert := geom.IdentityMatrix()
	if o.helper.UsesLocalCoords() {
		inv, ok := o.paths[len(o.paths)-1].viewMatrix.Invert()
		if !ok {
			return
		}
		invert = inv
	}

	quadProcessor := makeQuadEdgeEffect(&invert, o.helper.UsesLocalCoords(), o.wideColor)

	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), quadProcessor,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op.
func (o *aaConvexPathOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
		if o.programInfo == nil {
			return
		}
	}

	vertexStride := uint64(o.programInfo.GeomProc().gpBase().VertexStride())

	// TODO: generate all segments for all paths and use one vertex buffer.
	for i := range o.paths {
		args := &o.paths[i]

		// A perspective transform needs subdivision, so transform the path itself in that case. Otherwise, we apply the
		// view matrix when copying to the segment representation.
		viewMatrix := &args.viewMatrix
		pathPtr := args.path
		if viewMatrix.HasPerspective() {
			tmpPath := pathPtr.Clone()
			tmpPath.Transform(viewMatrix)
			tmpPath.SetVolatile(true)
			pathPtr = tmpPath
			identity := geom.IdentityMatrix()
			viewMatrix = &identity
		}

		var vertexCount, indexCount int
		segments := make([]aaConvexSegment, 0, 512/64)
		var fanPt geom.Point

		if !aaConvexGetSegments(pathPtr, viewMatrix, &segments, &fanPt, &vertexCount,
			&indexCount) {
			continue
		}

		vertData, vertexBuffer, firstVertex := state.MakeVertexSpace(vertexStride, vertexCount)
		if vertData == nil {
			return
		}
		idxData, indexBuffer, firstIndex := state.MakeIndexSpace(indexCount)
		if idxData == nil {
			return
		}

		var draws []aaConvexDraw
		idxs := make([]uint16, indexCount)
		writer := &quadVertexWriter{buf: vertData}
		aaConvexCreateVertices(segments, fanPt, args.color, o.wideColor, &draws, writer, idxs)
		for j := range idxs {
			binary.LittleEndian.PutUint16(idxData[2*j:], idxs[j])
		}

		for j := range draws {
			draw := &draws[j]
			mesh := &simpleMesh{}
			mesh.setIndexed(indexBuffer, draw.indexCnt, firstIndex, 0,
				uint16(draw.vertexCnt-1), vertexBuffer, firstVertex)
			firstIndex += draw.indexCnt
			firstVertex += draw.vertexCnt
			o.meshes = append(o.meshes, mesh)
		}
	}
}

// OnExecute implements Op.
func (o *aaConvexPathOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
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
func (o *aaConvexPathOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*aaConvexPathOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}
	if o.helper.UsesLocalCoords() &&
		!o.paths[0].viewMatrix.CheapEqual(&that.paths[0].viewMatrix) {
		return CombineResultCannotCombine
	}

	o.paths = append(o.paths, that.paths...)
	o.wideColor = o.wideColor || that.wideColor
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// AAConvexPathRenderer

// AAConvexPathRenderer is a PathRenderer for coverage-AA convex fills.
type AAConvexPathRenderer struct{}

// NewAAConvexPathRenderer returns the renderer (it is stateless).
func NewAAConvexPathRenderer() *AAConvexPathRenderer { return &AAConvexPathRenderer{} }

// Name implements PathRenderer.
func (r *AAConvexPathRenderer) Name() string { return "AAConvex" }

// OnGetStencilSupport implements PathRenderer (the base-class kNoSupport default).
func (r *AAConvexPathRenderer) OnGetStencilSupport(*StyledShape) StencilSupport {
	return StencilSupportNone
}

// OnCanDrawPath implements PathRenderer.
func (r *AAConvexPathRenderer) OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath {
	// This check requires convexity and known direction, since the direction is used to build the geometry segments.
	// Degenerate convex paths will fall through to some other path renderer.
	if args.Caps.ShaderCaps.ShaderDerivativeSupport &&
		args.AAType == gpu.AATypeCoverage && args.Shape.Style().IsSimpleFill() &&
		!args.Shape.InverseFilled() && args.Shape.KnownToBeConvex() &&
		args.Shape.KnownDirection() {
		return CanDrawPathYes
	}
	return CanDrawPathNo
}

// OnDrawPath implements PathRenderer.
func (r *AAConvexPathRenderer) OnDrawPath(args *DrawPathArgs) bool {
	if args.SurfaceDrawContext.NumSamples() > 1 {
		panic("AAConvexPathRenderer requires a single-sample target")
	}
	if args.Shape.IsEmpty() {
		panic("AAConvexPathRenderer requires a non-empty shape")
	}

	p := args.Shape.AsPath()
	op := newAAConvexPathOp(args.Paint, args.ViewMatrix, p, args.UserStencilSettings)
	args.SurfaceDrawContext.AddDrawOp(args.Clip, op)
	return true
}

// OnStencilPath implements PathRenderer (never reached: kNoSupport).
func (r *AAConvexPathRenderer) OnStencilPath(*StencilPathArgs) {
	panic("AAConvexPathRenderer cannot stencil paths")
}
