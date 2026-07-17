// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Coverage-AA hairlines (and hairline-equivalent strokes) rendered by gathering the path into device-space line
// segments, quads, and conics; lines become 6-vertex bloated segments with per-vertex coverage through the default
// geometry processor, while quads and conics become 5-sided bounding polys shaded by the quad/conic distance
// approximations (bezierfx.go). The program characterization (which of the line/quad/conic programs are needed) is
// computed at prepare time and consumed at execute time; overflow of the int32 line/quad/conic counts is guarded with
// the same maxLines/maxQuadsAndConics bail-outs.

package gl

import (
	"encoding/binary"
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// Quadratics are rendered as 5-sided polys in order to bound the AA stroke around the center-curve. See the comments in
// aaHairlineQuadIdxPattern and bloatQuad. Quadratics and conics share an index buffer.

// Each quadratic is rendered as a five sided polygon. This poly bounds the quadratic's bounding triangle but has been
// expanded so that the 1-pixel wide area around the curve is inside the poly. If a,b,c are the original control points
// then the poly a0,b0,c0,c1,a1 that is rendered would look like this:
//
//	         b0
//	         b
//
//	a0              c0
//	 a            c
//	  a1       c1
//
// Each is drawn as three triangles ((a0,a1,b0), (b0,c1,c0), (a1,c1,b0)) specified by these 9 indices.
var aaHairlineQuadIdxPattern = []uint16{
	0, 1, 2,
	2, 4, 3,
	1, 4, 2,
}

const (
	aaHairlineIdxsPerQuad       = 9
	aaHairlineQuadNumVertices   = 5
	aaHairlineQuadsNumInIdxBuf  = 256
	aaHairlineIdxsPerLineSeg    = 18
	aaHairlineLineSegNumVerts   = 6
	aaHairlineLineSegsNumInIdxB = 256
)

// Each line segment is rendered as two quads and two triangles. p0 and p1 have alpha = 1 while all other points have
// alpha = 0. The four external points are offset 1 pixel perpendicular to the line and half a pixel parallel to the
// line.
//
//	p4                  p5
//	     p0         p1
//	p2                  p3
//
// Each is drawn as six triangles specified by these 18 indices.
var aaHairlineLineSegIdxPattern = []uint16{
	0, 1, 3,
	0, 3, 2,
	0, 4, 5,
	0, 5, 1,
	0, 2, 4,
	1, 5, 3,
}

// The static unique keys for the two patterned index buffers, generated once.
var (
	aaHairlineKeysOnce            sync.Once
	aaHairlineQuadsIndexBufferKey gpu.UniqueKey
	aaHairlineLinesIndexBufferKey gpu.UniqueKey
)

func aaHairlineIndexBufferKeys() {
	aaHairlineKeysOnce.Do(func() {
		makeKey := func(key *gpu.UniqueKey, label string) {
			builder := gpu.UniqueKeyBuilder(key, gpu.GenerateUniqueKeyDomain(), 1, label)
			builder.Slice()[0] = 1
			builder.Finish()
		}
		makeKey(&aaHairlineQuadsIndexBufferKey, "AAHairlineQuadsIndexBuffer")
		makeKey(&aaHairlineLinesIndexBufferKey, "AAHairlineLinesIndexBuffer")
	})
}

// getQuadsIndexBuffer returns the shared patterned index buffer for bloated quads/conics.
func getQuadsIndexBuffer(rp *ResourceProvider) *Buffer {
	aaHairlineIndexBufferKeys()
	return rp.findOrCreatePatternedIndexBuffer(aaHairlineQuadIdxPattern, aaHairlineIdxsPerQuad,
		aaHairlineQuadsNumInIdxBuf, aaHairlineQuadNumVertices, &aaHairlineQuadsIndexBufferKey)
}

// getLinesIndexBuffer returns the shared patterned index buffer for bloated line segments.
func getLinesIndexBuffer(rp *ResourceProvider) *Buffer {
	aaHairlineIndexBufferKeys()
	return rp.findOrCreatePatternedIndexBuffer(aaHairlineLineSegIdxPattern,
		aaHairlineIdxsPerLineSeg, aaHairlineLineSegsNumInIdxB, aaHairlineLineSegNumVerts,
		&aaHairlineLinesIndexBufferKey)
}

// getFloatExp returns the (biased) binary exponent of x.
func getFloatExp(x float32) int {
	return int((math.Float32bits(x)&0x7f800000)>>23) - 127
}

// splitConic uses the max curvature function for quads to estimate where to chop the conic. If the max curvature is not
// found along the curve segment it will return 1 and dst[0] is the original conic. If it returns 2 the dst[0] and
// dst[1] are the two new conics.
func splitConic(src []geom.Point, dst *[2]geom.Conic, weight float32) int {
	t := geom.FindQuadMaxCurvature(src)
	// FindQuadMaxCurvature() returns either a value in [0, 1) or NaN. However, passing NaN to ChopAt will assert.
	// Checking to see if t is in (0,1) will also cover the NaN case since NaN comparisons are always false, so we'll
	// drop down into the else block in that case.
	if 0 < t && t < 1 {
		if dst != nil {
			conic := geom.MakeConic(src[0], src[1], src[2], weight)
			if !conic.ChopAt(t, dst) {
				dst[0] = geom.MakeConic(src[0], src[1], src[2], weight)
				return 1
			}
		}
		return 2
	}
	if dst != nil {
		dst[0] = geom.MakeConic(src[0], src[1], src[2], weight)
	}
	return 1
}

// chopConic calls splitConic on the entire conic and then once more on each subsection. Most cases will result in
// either 1 conic (chop point is not within t range) or 3 conics (split once and then one subsection is split again).
func chopConic(src []geom.Point, dst *[4]geom.Conic, weight float32) int {
	var dstTemp [2]geom.Conic
	conicCnt := splitConic(src, &dstTemp, weight)
	if conicCnt == 2 {
		var first, second [2]geom.Conic
		conicCnt2 := splitConic(dstTemp[0].Pts[:], &first, dstTemp[0].W)
		dst[0] = first[0]
		dst[1] = first[1]
		cnt := conicCnt2 + splitConic(dstTemp[1].Pts[:], &second, dstTemp[1].W)
		dst[conicCnt2] = second[0]
		dst[conicCnt2+1] = second[1]
		conicCnt = cnt
	} else {
		dst[0] = dstTemp[0]
	}
	return conicCnt
}

// isDegenQuadOrConic reports whether the quad/conic is degenerate or close to it (in which case the path is
// approximated with lines), along with the squared distance of the middle control point from the baseline.
func isDegenQuadOrConic(p []geom.Point) (dsqd float32, degenerate bool) {
	degenerateToLineTol := pathUtilsDefaultTolerance
	degenerateToLineTolSqd := degenerateToLineTol * degenerateToLineTol

	if p[0].DistanceToSqd(p[1]) < degenerateToLineTolSqd ||
		p[1].DistanceToSqd(p[2]) < degenerateToLineTolSqd {
		return 0, true
	}

	dsqd = p[1].DistanceToLineBetweenSqd(p[0], p[2])
	if dsqd < degenerateToLineTolSqd {
		return dsqd, true
	}

	if p[2].DistanceToLineBetweenSqd(p[1], p[0]) < degenerateToLineTolSqd {
		return dsqd, true
	}
	return dsqd, false
}

// numQuadSubdivs computes how many times to subdivide a quad to avoid huge overfill; returns -1 if the quad should be
// drawn as lines instead.
func numQuadSubdivs(p []geom.Point) int {
	dsqd, degenerate := isDegenQuadOrConic(p)
	if degenerate {
		return -1
	}

	// Tolerance of triangle height in pixels; tuned on windows Quadro FX 380 / Z600. Trade off of fill vs cpu time on
	// verts.
	const subdivTol = float32(175)

	if dsqd <= subdivTol*subdivTol {
		return 0
	}
	const maxSub = 4
	// Subdividing the quad reduces d by 4. So we want x = log4(d/tol) = log4(d*d/tol*tol)/2 = log2(d*d/tol*tol). +1
	// since we're ignoring the mantissa contribution.
	log := getFloatExp(dsqd/(subdivTol*subdivTol)) + 1
	return min(max(0, log), maxSub)
}

// gatherLinesAndQuads generates the lines and quads to be rendered. Lines are always recorded in device space; we will
// do a device space bloat to account for the 1 pixel thickness. Quads are recorded in device space unless m contains
// perspective, in which case they are in source space. We do this because we will subdivide large quads to reduce
// over-fill; this subdivision has to be performed before applying the perspective matrix. Returns the total quad count
// (after subdivision).
//
// path.Iter materializes a contour's closing line itself as a VerbLine record (followed by VerbClose carrying the
// contour start), so the VerbClose case here only needs to emit caps.
func gatherLinesAndQuads(p *path.Path, m *geom.Matrix, devClipBounds geom.IRect, capLength float32, convertConicsToQuads bool, lines, quads, conics *[]geom.Point, quadSubdivCnts *[]int, conicWeights *[]float32) int {
	totalQuadCount := 0
	persp := m.HasPerspective()

	// Whenever a degenerate, zero-length contour is encountered, this code will insert a 'capLength' x-aligned line
	// segment. Since this is rendering hairlines it is hoped this will suffice for AA square & circle capping.
	verbsInContour := 0 // does not count moves
	seenZeroLengthVerb := false
	var zeroVerbPt geom.Point

	safeIBounds := func(pts []geom.Point) geom.IRect {
		return geom.BoundsOrEmpty(pts).Outset(1, 1).RoundOut()
	}

	// Adds a quad that has already been chopped to the list and checks for quads that are close to lines. Also does a
	// bounding box check. It takes points that are in source space and device space; the source points are only
	// required if the view matrix has perspective.
	addChoppedQuad := func(srcPts, devPts []geom.Point, isContourStart bool) {
		if srcPts == nil && persp {
			panic("perspective requires source-space points")
		}
		if devClipBounds.Intersects(safeIBounds(devPts[:3])) {
			subdiv := numQuadSubdivs(devPts)
			if subdiv == -1 {
				*lines = append(*lines, devPts[0], devPts[1], devPts[1], devPts[2])
				pts := (*lines)[len(*lines)-4:]
				if isContourStart && pts[0] == pts[1] && pts[2] == pts[3] {
					seenZeroLengthVerb = true
					zeroVerbPt = pts[0]
				}
			} else {
				// When in perspective keep quads in source space.
				qPts := devPts
				if persp {
					qPts = srcPts
				}
				*quads = append(*quads, qPts[0], qPts[1], qPts[2])
				*quadSubdivCnts = append(*quadSubdivCnts, subdiv)
				totalQuadCount += 1 << subdiv
			}
		}
	}

	// Common code for handling lines.
	handleLineVerb := func(pathPts []geom.Point) {
		var devPts [2]geom.Point
		m.MapPoints(devPts[:], pathPts[:2])
		if devClipBounds.Intersects(safeIBounds(devPts[:])) {
			*lines = append(*lines, devPts[0], devPts[1])
			pts := (*lines)[len(*lines)-2:]
			if verbsInContour == 0 && pts[0] == pts[1] {
				seenZeroLengthVerb = true
				zeroVerbPt = pts[0]
			}
		}
		verbsInContour++
	}

	// Applies the view matrix to quad source points and calls the above helper.
	addSrcChoppedQuad := func(srcSpaceQuadPts []geom.Point, isContourStart bool) {
		var devPts [3]geom.Point
		m.MapPoints(devPts[:], srcSpaceQuadPts[:3])
		addChoppedQuad(srcSpaceQuadPts, devPts[:], isContourStart)
	}

	it := path.NewIter(p, false)
	for {
		verb, pathPts, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbConic:
			if convertConicsToQuads {
				weight := it.ConicWeight()
				quadPts, count := path.ConicToQuads(pathPts[:3], weight, 0.25)
				for i := 0; i < count; i++ {
					addSrcChoppedQuad(quadPts[2*i:2*i+3], verbsInContour == 0 && i == 0)
				}
			} else {
				var dst [4]geom.Conic
				// We chop the conics to create tighter clipping to hide error that appears near max curvature of very
				// thin conics. Thin hyperbolas with high weight still show error.
				conicCnt := chopConic(pathPts[:3], &dst, it.ConicWeight())
				for i := 0; i < conicCnt; i++ {
					var devPts [3]geom.Point
					chopPnts := dst[i].Pts[:]
					m.MapPoints(devPts[:], chopPnts[:3])
					if devClipBounds.Intersects(safeIBounds(devPts[:])) {
						if _, degen := isDegenQuadOrConic(devPts[:]); degen {
							*lines = append(*lines, devPts[0], devPts[1], devPts[1], devPts[2])
							pts := (*lines)[len(*lines)-4:]
							if verbsInContour == 0 && i == 0 && pts[0] == pts[1] &&
								pts[2] == pts[3] {
								seenZeroLengthVerb = true
								zeroVerbPt = pts[0]
							}
						} else {
							// When in perspective keep conics in source space.
							cPts := devPts[:]
							if persp {
								cPts = chopPnts
							}
							*conics = append(*conics, cPts[0], cPts[1], cPts[2])
							*conicWeights = append(*conicWeights, dst[i].W)
						}
					}
				}
			}
			verbsInContour++
		case path.VerbMove:
			// New contour (and last one was unclosed). If it was just a zero length drawing operation, and we're
			// supposed to draw caps, then add a tiny line.
			if seenZeroLengthVerb && verbsInContour == 1 && capLength > 0 {
				*lines = append(*lines,
					geom.Point{X: zeroVerbPt.X - capLength, Y: zeroVerbPt.Y},
					geom.Point{X: zeroVerbPt.X + capLength, Y: zeroVerbPt.Y})
			}
			verbsInContour = 0
			seenZeroLengthVerb = false
		case path.VerbLine:
			handleLineVerb(pathPts[:2])
		case path.VerbQuad:
			var choppedPts [5]geom.Point
			// Chopping the quad helps when the quad is either degenerate or nearly degenerate. When it is degenerate it
			// allows the approximation with lines to work since the chop point (if there is one) will be at the
			// parabola's vertex. In the nearly degenerate the QuadUVMatrix computed for the points is almost singular
			// which can cause rendering artifacts.
			n := geom.ChopQuadAtMaxCurvature(pathPts[:3], choppedPts[:])
			for i := 0; i < n; i++ {
				addSrcChoppedQuad(choppedPts[i*2:i*2+3], verbsInContour == 0 && i == 0)
			}
			verbsInContour++
		case path.VerbCubic:
			var devPts [4]geom.Point
			m.MapPoints(devPts[:], pathPts[:4])
			if devClipBounds.Intersects(safeIBounds(devPts[:])) {
				var q []geom.Point
				// We convert cubics to quadratics (for now). In perspective we have to do the conversion in source
				// space.
				src := pathPts
				if persp {
					tolScale := scaleToleranceToSrc(1, m, p.Bounds())
					convertCubicToQuads(&src, tolScale, &q)
				} else {
					convertCubicToQuads(&devPts, 1, &q)
				}
				for i := 0; i+2 < len(q); i += 3 {
					if persp {
						addSrcChoppedQuad(q[i:i+3], verbsInContour == 0 && i == 0)
					} else {
						addChoppedQuad(nil, q[i:i+3], verbsInContour == 0 && i == 0)
					}
				}
			}
			verbsInContour++
		case path.VerbClose:
			// The closing line (when the contour's last point differs from its start) has already been emitted by the
			// iterator as a VerbLine. Contour is closed, so we don't need to grow the starting line, unless it's *just*
			// a zero length subpath (SVG spec 11.4, 'stroke').
			if capLength > 0 {
				if seenZeroLengthVerb && verbsInContour == 1 {
					*lines = append(*lines,
						geom.Point{X: zeroVerbPt.X - capLength, Y: zeroVerbPt.Y},
						geom.Point{X: zeroVerbPt.X + capLength, Y: zeroVerbPt.Y})
				} else if verbsInContour == 0 {
					// Contour was (moveTo, close). Add a line.
					var devPts [2]geom.Point
					m.MapPoints(devPts[:1], pathPts[:1])
					devPts[1] = devPts[0]
					if devClipBounds.Intersects(safeIBounds(devPts[:])) {
						*lines = append(*lines,
							geom.Point{X: devPts[0].X - capLength, Y: devPts[0].Y},
							geom.Point{X: devPts[1].X + capLength, Y: devPts[1].Y})
					}
				}
			}
		}
	}
	if seenZeroLengthVerb && verbsInContour == 1 && capLength > 0 {
		// Path ended with a dangling (moveTo, line|quad|etc). If the final verb is degenerate, we need to draw a line.
		*lines = append(*lines,
			geom.Point{X: zeroVerbPt.X - capLength, Y: zeroVerbPt.Y},
			geom.Point{X: zeroVerbPt.X + capLength, Y: zeroVerbPt.Y})
	}
	return totalQuadCount
}

// aaHairlineLineVertex is a line-segment vertex: position + coverage (12 bytes serialized).
type aaHairlineLineVertex struct {
	pos      geom.Point
	coverage float32
}

// aaHairlineBezierVertex is a quad/conic vertex: position + the conic KLM / quad UV union (24 bytes serialized).
type aaHairlineBezierVertex struct {
	pos geom.Point
	// coeffs is fConic.fKLM[0..2] + padding for conics, or fQuadCoord (u, v) in the first two entries for quads.
	coeffs [4]float32
}

// intersectLines finds the intersection of two lines, each given as a point and normal.
func intersectLines(ptA, normA, ptB, normB geom.Point) geom.Point {
	lineAW := -normA.Dot(ptA)
	lineBW := -normB.Dot(ptB)

	wInv := normA.X*normB.Y - normA.Y*normB.X
	wInv = geom.IEEEFloatDivide(1, wInv)
	if !geom.IsFinite(wInv) {
		// Lines are parallel, pick the point in between.
		return ptA.Add(ptB).Scaled(0.5).Add(normA)
	}
	return geom.Point{
		X: (normA.Y*lineBW - lineAW*normB.Y) * wInv,
		Y: (lineAW*normB.X - normA.X*lineBW) * wInv,
	}
}

// setUVQuad computes each vertex's UV coordinate relative to qpts. qpts should be in source space, not device coords,
// when we have perspective.
func setUVQuad(qpts *[3]geom.Point, verts []aaHairlineBezierVertex) {
	var devToUV quadUVMatrix
	devToUV.set(qpts)
	pos := make([]geom.Point, len(verts))
	uv := make([]geom.Point, len(verts))
	for i := range verts {
		pos[i] = verts[i].pos
	}
	devToUV.apply(pos, uv)
	for i := range verts {
		verts[i].coeffs[0] = uv[i].X
		verts[i].coeffs[1] = uv[i].Y
	}
}

// bloatQuad makes a new poly where the quad's endpoints a and c are replaced by 1-pixel wide edges orthogonal to edges
// ab and bc. Returns false when the quad is degenerate (and should not be added).
func bloatQuad(qpts *[3]geom.Point, toDevice, toSrc *geom.Matrix, verts []aaHairlineBezierVertex) bool {
	if (toDevice == nil) != (toSrc == nil) {
		panic("bloatQuad requires both or neither of toDevice/toSrc")
	}
	// Original quad is specified by tri a,b,c.
	a := qpts[0]
	b := qpts[1]
	c := qpts[2]

	if toDevice != nil {
		a = toDevice.MapPoint(a)
		b = toDevice.MapPoint(b)
		c = toDevice.MapPoint(c)
	}
	// Make a new poly where we replace a and c by a 1-pixel wide edges orthogonal to edges ab and bc:
	//
	//   before       |        after
	//                |              b0
	//         b      |
	//                |
	//                |     a0            c0
	// a c | a1 c1
	//
	// edges a0->b0 and b0->c0 are parallel to original edges a->b and b->c, respectively.
	ab := b.Sub(a)
	ac := c.Sub(a)
	cb := b.Sub(c)

	// After the transform (or due to floating point math) we might have a line, try to do something reasonable.
	abNormalized := ab.Normalize()
	cbNormalized := cb.Normalize()

	if !abNormalized {
		if !cbNormalized {
			return false // quad is degenerate so we won't add it
		}
		ab = cb
	}
	if !cbNormalized {
		cb = ab
	}

	abN := ab.MakeOrthog(geom.PointSideLeft)
	if abN.Dot(ac) > 0 {
		abN = abN.Negated()
	}
	cbN := cb.MakeOrthog(geom.PointSideLeft)
	if cbN.Dot(ac) < 0 {
		cbN = cbN.Negated()
	}

	verts[0].pos = a.Add(abN) // a0
	verts[1].pos = a.Sub(abN) // a1

	if toDevice != nil && ac.LengthSqd() <= geom.ScalarNearlyZeroTol*geom.ScalarNearlyZeroTol {
		c = b
	}
	verts[3].pos = c.Add(cbN) // c0
	verts[4].pos = c.Sub(cbN) // c1

	verts[2].pos = intersectLines(verts[0].pos, abN, verts[3].pos, cbN) // b0

	if toSrc != nil {
		for i := range verts {
			verts[i].pos = toSrc.MapPoint(verts[i].pos)
		}
	}
	return true
}

// setConicCoeffs computes each vertex's implicit-form coefficients. Equations based off of Loop-Blinn Quadratic GPU
// Rendering. Input parametric:
//
//	P(t) = (P0*(1-t)^2 + 2*w*P1*t*(1-t) + P2*t^2) / (1-t)^2 + 2*w*t*(1-t) + t^2)
//
// Output implicit: f(x, y, w) = f(P) = K^2 - LM, with K = dot(k, P), L = dot(l, P), M = dot(m, P); k, l, m are
// calculated by getConicKLM.
func setConicCoeffs(p *[3]geom.Point, verts []aaHairlineBezierVertex, weight float32) {
	klm := getConicKLM(p, weight)
	for i := range verts {
		x := verts[i].pos.X
		y := verts[i].pos.Y
		verts[i].coeffs[0] = klm[0]*x + klm[1]*y + klm[2]
		verts[i].coeffs[1] = klm[3]*x + klm[4]*y + klm[5]
		verts[i].coeffs[2] = klm[6]*x + klm[7]*y + klm[8]
	}
}

// addConics bloats a conic's control triangle and writes its coverage/coefficient vertices.
func addConics(p []geom.Point, weight float32, toDevice, toSrc *geom.Matrix, verts []aaHairlineBezierVertex, vertIdx *int) {
	quad := [3]geom.Point{p[0], p[1], p[2]}
	if bloatQuad(&quad, toDevice, toSrc, verts[*vertIdx:*vertIdx+aaHairlineQuadNumVertices]) {
		setConicCoeffs(&quad, verts[*vertIdx:*vertIdx+aaHairlineQuadNumVertices], weight)
		*vertIdx += aaHairlineQuadNumVertices
	}
}

// addQuads subdivides a quad subdiv times, bloating and writing vertices for each piece.
func addQuads(p []geom.Point, subdiv int, toDevice, toSrc *geom.Matrix, verts []aaHairlineBezierVertex, vertIdx *int) {
	if subdiv < 0 {
		panic("subdiv must be non-negative")
	}
	// Temporary vertex storage to avoid reading the vertex buffer.
	var outVerts [aaHairlineQuadNumVertices]aaHairlineBezierVertex

	// Storage for the chopped quad: pts 0,1,2 are the first quad, and 2,3,4 the second quad.
	var choppedQuadPts [5]geom.Point
	// Start off with our original curve in the second quad slot.
	copy(choppedQuadPts[2:], p[:3])

	stepCount := 1 << subdiv
	for stepCount > 1 {
		// The general idea is:
		// * chop the quad using pts 2,3,4 as the input
		// * write out verts using pts 0,1,2
		// * now 2,3,4 is the remainder of the curve, chop again until all subdivisions are done
		h := 1 / float32(stepCount)
		geom.ChopQuadAt(choppedQuadPts[2:], choppedQuadPts[:], h)

		first := [3]geom.Point{choppedQuadPts[0], choppedQuadPts[1], choppedQuadPts[2]}
		if bloatQuad(&first, toDevice, toSrc, outVerts[:]) {
			setUVQuad(&first, outVerts[:])
			copy(verts[*vertIdx:*vertIdx+aaHairlineQuadNumVertices], outVerts[:])
			*vertIdx += aaHairlineQuadNumVertices
		}
		stepCount--
	}

	// Finish up, write out the final quad.
	last := [3]geom.Point{choppedQuadPts[2], choppedQuadPts[3], choppedQuadPts[4]}
	if bloatQuad(&last, toDevice, toSrc, outVerts[:]) {
		setUVQuad(&last, outVerts[:])
		copy(verts[*vertIdx:*vertIdx+aaHairlineQuadNumVertices], outVerts[:])
		*vertIdx += aaHairlineQuadNumVertices
	}
}

// addLine bloats a line segment into a 6-vertex quad pair with per-vertex coverage.
func addLine(p []geom.Point, toSrc *geom.Matrix, coverage uint8, verts []aaHairlineLineVertex, vertIdx *int) {
	a := p[0]
	b := p[1]

	vec := b.Sub(a)
	lengthSqd := vec.LengthSqd()

	out := verts[*vertIdx : *vertIdx+aaHairlineLineSegNumVerts]
	if vec.SetLength(0.5) {
		// Create a vector orthogonal to 'vec' and of unit length.
		ortho := geom.Point{X: 2 * vec.Y, Y: -2 * vec.X}

		floatCoverage := float32(coverage) * (1.0 / 255)

		if lengthSqd >= 1 {
			// Relative to points a and b: the inner vertices are inset half a pixel along the line a,b.
			out[0].pos = a.Add(vec)
			out[0].coverage = floatCoverage
			out[1].pos = b.Sub(vec)
			out[1].coverage = floatCoverage
		} else {
			// The inner vertices are inset a distance of length(a,b) from the outer edge of geometry. For the "a" inset
			// this is the same as insetting from b by half a pixel. The coverage is then modulated by the length. This
			// gives us the correct coverage for rects shorter than a pixel as they get translated subpixel amounts
			// inside of a pixel.
			length := geom.ScalarSqrt(lengthSqd)
			out[0].pos = b.Sub(vec)
			out[0].coverage = floatCoverage * length
			out[1].pos = a.Add(vec)
			out[1].coverage = floatCoverage * length
		}
		// Relative to points a and b: the outer vertices are outset half a pixel along the line a,b and then a whole
		// pixel orthogonally.
		out[2].pos = a.Sub(vec).Add(ortho)
		out[2].coverage = 0
		out[3].pos = b.Add(vec).Add(ortho)
		out[3].coverage = 0
		out[4].pos = a.Sub(vec).Sub(ortho)
		out[4].coverage = 0
		out[5].pos = b.Add(vec).Sub(ortho)
		out[5].coverage = 0

		if toSrc != nil {
			for i := range out {
				out[i].pos = toSrc.MapPoint(out[i].pos)
			}
		}
	} else {
		// Just make it degenerate and likely offscreen.
		for i := range out {
			out[i].pos = geom.Point{X: geom.ScalarMax, Y: geom.ScalarMax}
		}
	}

	*vertIdx += aaHairlineLineSegNumVerts
}

//////////////////////////////////////////////////////////////////////////////
// AAHairlineOp

// aaHairlineProgram is a bitmask of which of the line/quad/conic programs a draw needs.
type aaHairlineProgram uint8

const (
	aaHairlineProgramNone  aaHairlineProgram = 0
	aaHairlineProgramLine  aaHairlineProgram = 0x1
	aaHairlineProgramQuad  aaHairlineProgram = 0x2
	aaHairlineProgramConic aaHairlineProgram = 0x4
)

var aaHairlineOpClassID = GenOpClassID()

// aaHairlinePathData holds one path's per-draw state within a batched aaHairlineOp.
type aaHairlinePathData struct {
	path          *path.Path
	viewMatrix    geom.Matrix
	devClipBounds geom.IRect
	capLength     float32
}

// aaHairlineOp draws one or more coverage-AA hairlines, batched together when possible.
type aaHairlineOp struct {
	drawOpNoClipToShape
	meshes       [3]*simpleMesh
	programInfos [3]*ProgramInfo
	helper       SimpleMeshDrawOpHelper
	paths        []aaHairlinePathData
	OpBase
	color            colorcore.PMColor4f
	coverage         uint8
	characterization aaHairlineProgram // holds a mask of required programs
}

// newAAHairlineOp creates an op to draw p as a hairline (or hairline-equivalent stroke) with paint under viewMatrix.
// The paint is consumed.
func newAAHairlineOp(paint *Paint, viewMatrix *geom.Matrix, p *path.Path, style *Style, devClipBounds geom.IRect, stencilSettings *UserStencilSettings) DrawOp {
	newCoverage := uint8(0xff)
	hairlineCoverage, isHairline := isStrokeHairlineOrEquivalent(style, viewMatrix)
	if isHairline {
		newCoverage = uint8(geom.RoundToInt(hairlineCoverage * 0xff))
	}

	strokeRec := style.Rec()
	capLength := float32(0)
	if strokeRec.Cap() != stroke.CapButt {
		capLength = hairlineCoverage * 0.5
	}

	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := &aaHairlineOp{}
	o.InitOp(aaHairlineOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, gpu.AATypeCoverage, stencilSettings,
		PipelineInputFlagNone)
	o.color = color
	o.coverage = newCoverage
	o.paths = append(o.paths, aaHairlinePathData{
		viewMatrix:    *viewMatrix,
		path:          p,
		devClipBounds: devClipBounds,
		capLength:     capLength,
	})
	devBounds, _ := viewMatrix.MapRect(p.Bounds())
	o.SetBounds(devBounds, HasAABloat(true), IsHairline(true))
	return o
}

// Name implements Op.
func (o *aaHairlineOp) Name() string { return "AAHairlineOp" }

// VisitProxies implements Op.
func (o *aaHairlineOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	visited := false
	for i := 0; i < 3; i++ {
		if o.programInfos[i] != nil {
			o.programInfos[i].VisitFPProxies(fn)
			visited = true
		}
	}
	if !visited {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *aaHairlineOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *aaHairlineOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *aaHairlineOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	// This op uses uniform (not vertex) color, so doesn't need to track wide color.
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.color, nil)
}

func (o *aaHairlineOp) viewMatrix() *geom.Matrix { return &o.paths[0].viewMatrix }

// makeLineProgramInfo builds the program info for the line-segment draw, if not already built.
func (o *aaHairlineOp) makeLineProgramInfo(state *OpFlushState, pipeline *Pipeline, geometryProcessorViewM, geometryProcessorLocalM *geom.Matrix) {
	if o.programInfos[0] != nil {
		return
	}
	localCoordsType := LocalCoordsTypeUnused
	if o.helper.UsesLocalCoords() {
		localCoordsType = LocalCoordsTypeUsePosition
	}
	lineGP := MakeDefaultGeoProc(ColorTypePremulUniform, o.color, CoverageTypeAttribute, 0xff,
		localCoordsType, geometryProcessorLocalM, geometryProcessorViewM)
	if lineGP.gpBase().VertexStride() != 12 {
		panic("unexpected line vertex stride")
	}
	args := state.OpArgs()
	o.programInfos[0] = NewProgramInfo(state.Caps(), args.WriteView(), args.UsesMSAASurface(),
		pipeline, o.helper.StencilSettings(), lineGP, gpu.PrimitiveTypeTriangles,
		args.RenderPassBarriers(), args.ColorLoadOp())
}

// makeQuadProgramInfo builds the program info for the quad draw, if not already built.
func (o *aaHairlineOp) makeQuadProgramInfo(state *OpFlushState, pipeline *Pipeline, geometryProcessorViewM, geometryProcessorLocalM *geom.Matrix) {
	if o.programInfos[1] != nil {
		return
	}
	quadGP := makeQuadEffect(o.color, geometryProcessorViewM, state.Caps(),
		geometryProcessorLocalM, o.helper.UsesLocalCoords(), o.coverage)
	if quadGP.gpBase().VertexStride() != 24 {
		panic("unexpected quad vertex stride")
	}
	args := state.OpArgs()
	o.programInfos[1] = NewProgramInfo(state.Caps(), args.WriteView(), args.UsesMSAASurface(),
		pipeline, o.helper.StencilSettings(), quadGP, gpu.PrimitiveTypeTriangles,
		args.RenderPassBarriers(), args.ColorLoadOp())
}

// makeConicProgramInfo builds the program info for the conic draw, if not already built.
func (o *aaHairlineOp) makeConicProgramInfo(state *OpFlushState, pipeline *Pipeline, geometryProcessorViewM, geometryProcessorLocalM *geom.Matrix) {
	if o.programInfos[2] != nil {
		return
	}
	conicGP := makeConicEffect(o.color, geometryProcessorViewM, state.Caps(),
		geometryProcessorLocalM, o.helper.UsesLocalCoords(), o.coverage)
	if conicGP.gpBase().VertexStride() != 24 {
		panic("unexpected conic vertex stride")
	}
	args := state.OpArgs()
	o.programInfos[2] = NewProgramInfo(state.Caps(), args.WriteView(), args.UsesMSAASurface(),
		pipeline, o.helper.StencilSettings(), conicGP, gpu.PrimitiveTypeTriangles,
		args.RenderPassBarriers(), args.ColorLoadOp())
}

// createProgramInfos builds program info for each program (line/quad/conic) the characterization computed by OnPrepare
// determined is needed.
func (o *aaHairlineOp) createProgramInfos(state *OpFlushState) {
	// Setup the viewmatrix and localmatrix for the geometry processor.
	inverse, ok := o.viewMatrix().Invert()
	if !ok {
		return
	}

	// We will transform to identity space if the viewmatrix does not have perspective.
	identity := geom.IdentityMatrix()
	geometryProcessorViewM := &identity
	geometryProcessorLocalM := &inverse
	if o.viewMatrix().HasPerspective() {
		geometryProcessorViewM = o.viewMatrix()
		geometryProcessorLocalM = &identity
	}

	args := state.OpArgs()
	pipeline := o.helper.CreatePipeline(state.Caps(), args.WriteView().Swizzle(),
		args.AppliedClip(), args.DstProxyView())

	if o.characterization&aaHairlineProgramLine != 0 {
		o.makeLineProgramInfo(state, pipeline, geometryProcessorViewM, geometryProcessorLocalM)
	}
	if o.characterization&aaHairlineProgramQuad != 0 {
		o.makeQuadProgramInfo(state, pipeline, geometryProcessorViewM, geometryProcessorLocalM)
	}
	if o.characterization&aaHairlineProgramConic != 0 {
		o.makeConicProgramInfo(state, pipeline, geometryProcessorViewM, geometryProcessorLocalM)
	}
}

// OnPrepare implements Op.
func (o *aaHairlineOp) OnPrepare(state *OpFlushState) {
	// Setup the viewmatrix and localmatrix for the geometry processor.
	inverse, ok := o.viewMatrix().Invert()
	if !ok {
		return
	}

	// We will transform to identity space if the viewmatrix does not have perspective.
	var toDevice, toSrc *geom.Matrix
	if o.viewMatrix().HasPerspective() {
		toDevice = o.viewMatrix()
		toSrc = &inverse
	}

	actualPrograms := aaHairlineProgramNone

	var lines, quads, conics []geom.Point
	var qSubdivs []int
	var cWeights []float32
	quadCount := 0

	convertConicsToQuads := !state.Caps().ShaderCaps.FloatIs32Bits
	for i := range o.paths {
		args := &o.paths[i]
		quadCount += gatherLinesAndQuads(args.path, &args.viewMatrix, args.devClipBounds,
			args.capLength, convertConicsToQuads, &lines, &quads, &conics, &qSubdivs,
			&cWeights)
	}

	lineCount := len(lines) / 2
	conicCount := len(conics) / 3
	quadAndConicCount := conicCount + quadCount

	const maxLines = math.MaxInt32 / aaHairlineLineSegNumVerts
	const maxQuadsAndConics = math.MaxInt32 / aaHairlineQuadNumVertices
	if lineCount > maxLines || quadAndConicCount > maxQuadsAndConics {
		return
	}

	// Do lines first.
	if lineCount != 0 {
		actualPrograms |= aaHairlineProgramLine

		linesIndexBuffer := getLinesIndexBuffer(state.ResourceProvider())

		mesh := &simpleMesh{}
		vertData := makePatternHelper(state, 12, linesIndexBuffer, aaHairlineLineSegNumVerts,
			aaHairlineIdxsPerLineSeg, lineCount, aaHairlineLineSegsNumInIdxB, mesh)
		if vertData == nil {
			return
		}

		verts := make([]aaHairlineLineVertex, lineCount*aaHairlineLineSegNumVerts)
		vertIdx := 0
		for i := 0; i < lineCount; i++ {
			addLine(lines[2*i:2*i+2], toSrc, o.coverage, verts, &vertIdx)
		}
		for i, v := range verts {
			off := i * 12
			binary.LittleEndian.PutUint32(vertData[off:], math.Float32bits(v.pos.X))
			binary.LittleEndian.PutUint32(vertData[off+4:], math.Float32bits(v.pos.Y))
			binary.LittleEndian.PutUint32(vertData[off+8:], math.Float32bits(v.coverage))
		}

		o.meshes[0] = mesh
	}

	if quadCount != 0 || conicCount != 0 {
		quadsIndexBuffer := getQuadsIndexBuffer(state.ResourceProvider())

		vertexCount := aaHairlineQuadNumVertices * quadAndConicCount
		vertData, vertexBuffer, firstVertex := state.MakeVertexSpace(24, vertexCount)
		if vertData == nil || quadsIndexBuffer == nil {
			return
		}

		// Setup vertices.
		bezVerts := make([]aaHairlineBezierVertex, vertexCount)
		vertIdx := 0

		unsubdivQuadCnt := len(quads) / 3
		for i := 0; i < unsubdivQuadCnt; i++ {
			if qSubdivs[i] < 0 {
				panic("negative subdivision count")
			}
			if !quads[3*i].IsFinite() || !quads[3*i+1].IsFinite() || !quads[3*i+2].IsFinite() {
				return
			}
			addQuads(quads[3*i:3*i+3], qSubdivs[i], toDevice, toSrc, bezVerts, &vertIdx)
		}

		// Start Conics.
		for i := 0; i < conicCount; i++ {
			addConics(conics[3*i:3*i+3], cWeights[i], toDevice, toSrc, bezVerts, &vertIdx)
		}

		for i, v := range bezVerts {
			off := i * 24
			binary.LittleEndian.PutUint32(vertData[off:], math.Float32bits(v.pos.X))
			binary.LittleEndian.PutUint32(vertData[off+4:], math.Float32bits(v.pos.Y))
			for j, f := range v.coeffs {
				binary.LittleEndian.PutUint32(vertData[off+8+4*j:], math.Float32bits(f))
			}
		}

		if quadCount > 0 {
			actualPrograms |= aaHairlineProgramQuad

			mesh := &simpleMesh{}
			mesh.setIndexedPatterned(quadsIndexBuffer, aaHairlineIdxsPerQuad,
				aaHairlineQuadNumVertices, quadCount, aaHairlineQuadsNumInIdxBuf,
				vertexBuffer, firstVertex)
			o.meshes[1] = mesh
			firstVertex += quadCount * aaHairlineQuadNumVertices
		}

		if conicCount > 0 {
			actualPrograms |= aaHairlineProgramConic

			mesh := &simpleMesh{}
			mesh.setIndexedPatterned(quadsIndexBuffer, aaHairlineIdxsPerQuad,
				aaHairlineQuadNumVertices, conicCount, aaHairlineQuadsNumInIdxBuf,
				vertexBuffer, firstVertex)
			o.meshes[2] = mesh
		}
	}

	o.characterization = actualPrograms
}

// OnExecute implements Op.
func (o *aaHairlineOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	o.createProgramInfos(state)

	renderPass := state.OpsRenderPass()
	for i := 0; i < 3; i++ {
		if o.programInfos[i] != nil && o.meshes[i] != nil {
			if !renderPass.BindPipeline(o.programInfos[i], chainBounds) {
				continue
			}
			if o.programInfos[i].Pipeline().IsScissorTestEnabled() {
				renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
			}
			renderPass.BindTextures(o.programInfos[i].GeomProc(), nil,
				o.programInfos[i].Pipeline())
			o.meshes[i].draw(renderPass)
		}
	}
}

// OnCombineIfPossible implements Op.
func (o *aaHairlineOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*aaHairlineOp)
	if !ok {
		return CombineResultCannotCombine
	}

	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}

	if o.viewMatrix().HasPerspective() != that.viewMatrix().HasPerspective() {
		return CombineResultCannotCombine
	}

	// We go to identity if we don't have perspective.
	if o.viewMatrix().HasPerspective() && !o.viewMatrix().CheapEqual(that.viewMatrix()) {
		return CombineResultCannotCombine
	}

	// TODO: we can actually combine hairlines if they are the same color in a kind of bulk method but we haven't
	// implemented this yet; investigate going to vertex color and coverage?
	if o.coverage != that.coverage {
		return CombineResultCannotCombine
	}

	if o.color != that.color {
		return CombineResultCannotCombine
	}

	if o.helper.UsesLocalCoords() && !o.viewMatrix().CheapEqual(that.viewMatrix()) {
		return CombineResultCannotCombine
	}

	o.paths = append(o.paths, that.paths...)
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// AAHairLinePathRenderer

// AAHairLinePathRenderer is a PathRenderer for coverage-AA hairlines.
type AAHairLinePathRenderer struct{}

// NewAAHairLinePathRenderer returns the renderer (it is stateless).
func NewAAHairLinePathRenderer() *AAHairLinePathRenderer { return &AAHairLinePathRenderer{} }

// Name implements PathRenderer.
func (r *AAHairLinePathRenderer) Name() string { return "AAHairline" }

// OnGetStencilSupport implements PathRenderer (the base-class kNoSupport default).
func (r *AAHairLinePathRenderer) OnGetStencilSupport(*StyledShape) StencilSupport {
	return StencilSupportNone
}

// OnCanDrawPath implements PathRenderer.
func (r *AAHairLinePathRenderer) OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath {
	if args.AAType != gpu.AATypeCoverage {
		return CanDrawPathNo
	}

	if _, isHairline := isStrokeHairlineOrEquivalent(args.Shape.Style(),
		args.ViewMatrix); !isHairline {
		return CanDrawPathNo
	}

	// We don't currently handle dashing in this class though perhaps we should.
	if args.Shape.Style().HasPathEffect() {
		return CanDrawPathNo
	}

	if args.Shape.SegmentMask() == path.SegmentLine ||
		args.Caps.ShaderCaps.ShaderDerivativeSupport {
		return CanDrawPathYes
	}

	return CanDrawPathNo
}

// OnDrawPath implements PathRenderer.
func (r *AAHairLinePathRenderer) OnDrawPath(args *DrawPathArgs) bool {
	if args.SurfaceDrawContext.NumSamples() > 1 {
		panic("AAHairLinePathRenderer requires a single-sample target")
	}

	p := args.Shape.AsPath()
	op := newAAHairlineOp(args.Paint, args.ViewMatrix, p, args.Shape.Style(),
		args.ClipConservativeBounds, args.UserStencilSettings)
	args.SurfaceDrawContext.AddDrawOp(args.Clip, op)
	return true
}

// OnStencilPath implements PathRenderer (never reached: kNoSupport).
func (r *AAHairLinePathRenderer) OnStencilPath(*StencilPathArgs) {
	panic("AAHairLinePathRenderer cannot stencil paths")
}
