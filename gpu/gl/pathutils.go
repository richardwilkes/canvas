// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Path-geometry helpers the path renderers consume: device-tolerance scaling to source space, recursive quad/cubic
// point generation whose worst-case counts come from Wang's formula (the scalar log2 forms), the quadUVMatrix xy→uv
// transform, the conic KLM coefficients (Loop-Blinn), and the cubic→quad conversions (plain and tangent-constrained)
// used by the AA hairline/convex renderers.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// pathUtilsDefaultTolerance is the default device-space flattening tolerance for path tessellation.
const pathUtilsDefaultTolerance float32 = 0.25

// pathUtilsMaxPointsPerCurve caps the number of points a single curve can flatten to.
const pathUtilsMaxPointsPerCurve = 1 << 10

// pathUtilsMinCurveTol is the smallest source-space tolerance scaleToleranceToSrc will return.
const pathUtilsMinCurveTol float32 = 0.0001

// scaleToleranceToSrc converts a device-space flattening tolerance to source space, bounding how much viewM can scale
// when mapping to screen coordinates.
func scaleToleranceToSrc(devTol float32, viewM *geom.Matrix, pathBounds geom.Rect) float32 {
	stretch := viewM.MaxScale()
	if stretch < 0 {
		// Take the worst case mapRadius among the four corners (less than perfect).
		for i := 0; i < 4; i++ {
			var mat geom.Matrix
			x := pathBounds.Right
			if i%2 != 0 {
				x = pathBounds.Left
			}
			y := pathBounds.Bottom
			if i < 2 {
				y = pathBounds.Top
			}
			mat.SetTranslate(x, y)
			mat.PostConcat(viewM)
			stretch = max(stretch, mat.MapRadius(1))
		}
	}
	var srcTol float32
	if stretch <= 0 {
		// We have degenerate bounds or some degenerate matrix. Thus we set the tolerance to be the max of the path
		// bounds width and height.
		srcTol = max(pathBounds.Width(), pathBounds.Height())
	} else {
		srcTol = devTol / stretch
	}
	if srcTol < pathUtilsMinCurveTol {
		srcTol = pathUtilsMinCurveTol
	}
	return srcTol
}

// maxBezierVertices returns 1 << min(chopCount, 10).
func maxBezierVertices(chopCount int) int {
	const maxChopsPerCurve = 10
	return 1 << min(chopCount, maxChopsPerCurve)
}

// toleranceToWangsPrecision converts a max-distance tolerance (the max distance a linear segment may be from the real
// curve) to a Wang's-formula precision: Wang's formula guarantees segments stay within 1/precision of the true curve,
// so precision = 1/srcTol.
func toleranceToWangsPrecision(srcTol float32) float32 { return 1 / srcTol }

// quadraticPointCount returns the number of points needed to flatten the quadratic within tol.
func quadraticPointCount(pts [3]geom.Point, tol float32) int {
	return maxBezierVertices(wangsNextLog16(wangsQuadraticP4(toleranceToWangsPrecision(tol),
		pts[0], pts[1], pts[2], identityVectorXform())))
}

// cubicPointCount returns the number of points needed to flatten the cubic within tol.
func cubicPointCount(pts [4]geom.Point, tol float32) int {
	return maxBezierVertices(wangsNextLog16(wangsCubicP4(toleranceToWangsPrecision(tol),
		pts[0], pts[1], pts[2], pts[3], identityVectorXform())))
}

// distanceToLineSegmentBetweenSqd returns the squared distance from pt to the segment a-b.
func distanceToLineSegmentBetweenSqd(pt, a, b geom.Point) float32 {
	// The projection falls in one of three zones along the segment; uDotV and vLengthSqd decide which without a divide.
	ux := b.X - a.X
	uy := b.Y - a.Y
	vx := pt.X - a.X
	vy := pt.Y - a.Y
	uLengthSqd := ux*ux + uy*uy
	uDotV := ux*vx + uy*vy
	switch {
	case uDotV <= 0:
		return vx*vx + vy*vy
	case uDotV > uLengthSqd:
		wx := pt.X - b.X
		wy := pt.Y - b.Y
		return wx*wx + wy*wy
	default:
		det := ux*vy - uy*vx
		temp := det / uLengthSqd * det
		// Temp should be non-negative but it is possible that it is negative due to floating point error; return 0 in
		// that case.
		if !(temp >= 0) {
			return 0
		}
		return temp
	}
}

// generateQuadraticPoints appends up to pointsLeft points approximating the quadratic to dst, returning the count
// written.
func generateQuadraticPoints(p0, p1, p2 geom.Point, tolSqd float32, dst []geom.Point, pointsLeft int) int {
	if pointsLeft < 2 || distanceToLineSegmentBetweenSqd(p1, p0, p2) < tolSqd {
		dst[0] = p2
		return 1
	}

	q0 := geom.Point{X: (p0.X + p1.X) / 2, Y: (p0.Y + p1.Y) / 2}
	q1 := geom.Point{X: (p1.X + p2.X) / 2, Y: (p1.Y + p2.Y) / 2}
	r := geom.Point{X: (q0.X + q1.X) / 2, Y: (q0.Y + q1.Y) / 2}

	pointsLeft >>= 1
	a := generateQuadraticPoints(p0, q0, r, tolSqd, dst, pointsLeft)
	b := generateQuadraticPoints(r, q1, p2, tolSqd, dst[a:], pointsLeft)
	return a + b
}

// generateCubicPoints appends up to pointsLeft points approximating the cubic to dst, returning the count written.
func generateCubicPoints(p0, p1, p2, p3 geom.Point, tolSqd float32, dst []geom.Point, pointsLeft int) int {
	if pointsLeft < 2 ||
		(distanceToLineSegmentBetweenSqd(p1, p0, p3) < tolSqd &&
			distanceToLineSegmentBetweenSqd(p2, p0, p3) < tolSqd) {
		dst[0] = p3
		return 1
	}

	q0 := geom.Point{X: (p0.X + p1.X) / 2, Y: (p0.Y + p1.Y) / 2}
	q1 := geom.Point{X: (p1.X + p2.X) / 2, Y: (p1.Y + p2.Y) / 2}
	q2 := geom.Point{X: (p2.X + p3.X) / 2, Y: (p2.Y + p3.Y) / 2}
	r0 := geom.Point{X: (q0.X + q1.X) / 2, Y: (q0.Y + q1.Y) / 2}
	r1 := geom.Point{X: (q1.X + q2.X) / 2, Y: (q1.Y + q2.Y) / 2}
	s := geom.Point{X: (r0.X + r1.X) / 2, Y: (r0.Y + r1.Y) / 2}

	pointsLeft >>= 1
	a := generateCubicPoints(p0, q0, r0, s, tolSqd, dst, pointsLeft)
	b := generateCubicPoints(s, r1, q2, p3, tolSqd, dst[a:], pointsLeft)
	return a + b
}

//////////////////////////////////////////////////////////////////////////////
// QuadUVMatrix

// quadUVMatrix is an affine transform M such that M * xy_pt = uv_pt, where the quad's three control points map to (0,
// 0), (1/2, 0), and (1, 1) — the canonical space in which the curve is the parabola u^2 - v = 0. Stored as the first
// six entries of a row-major 3x3 (the third row is 0 0 1).
type quadUVMatrix struct {
	m [6]float32
}

// set computes the matrix for the given quadratic's three control points.
func (q *quadUVMatrix) set(qPts *[3]geom.Point) {
	// We know M * control_pts = [0 1/2 1]
	//                           [0  0   1]
	//                           [1  1   1]
	// We invert the control pt matrix and post concat to both sides to get M. Using the known form of the control point
	// matrix and the result, we can optimize and improve precision.
	x0 := float64(qPts[0].X)
	y0 := float64(qPts[0].Y)
	x1 := float64(qPts[1].X)
	y1 := float64(qPts[1].Y)
	x2 := float64(qPts[2].X)
	y2 := float64(qPts[2].Y)

	// Pre-calculate some adjugate matrix factors for determinant.
	a2 := x1*y2 - x2*y1
	a5 := x2*y0 - x0*y2
	a8 := x0*y1 - x1*y0
	det := a2 + a5 + a8

	if math.IsNaN(det) || math.IsInf(det, 0) ||
		geom.ScalarNearlyZeroTolerance(float32(det),
			geom.ScalarNearlyZeroTol*geom.ScalarNearlyZeroTol) {
		// The quad is degenerate. Hopefully this is rare. Find the pts that are farthest apart to compute a line
		// (unless it is really a pt).
		maxD := qPts[0].DistanceToSqd(qPts[1])
		maxEdge := 0
		d := qPts[1].DistanceToSqd(qPts[2])
		if d > maxD {
			maxD = d
			maxEdge = 1
		}
		d = qPts[2].DistanceToSqd(qPts[0])
		if d > maxD {
			maxEdge = 2
		}
		// We could have a tolerance here, not sure if it would improve anything.
		if maxD > 0 {
			// Set the matrix to give (u = 0, v = distance_to_line).
			lineVec := qPts[(maxEdge+1)%3].Sub(qPts[maxEdge])
			// When looking from the point 0 down the line we want positive distances to be to the left. This matches
			// the non-degenerate case.
			lineVec = lineVec.MakeOrthog(geom.PointSideLeft)
			// First row.
			q.m[0] = 0
			q.m[1] = 0
			q.m[2] = 0
			// Second row.
			q.m[3] = lineVec.X
			q.m[4] = lineVec.Y
			q.m[5] = -lineVec.Dot(qPts[maxEdge])
		} else {
			// It's a point. It should cover zero area. Just set the matrix such that (u, v) will always be far away
			// from the quad.
			q.m[0] = 0
			q.m[1] = 0
			q.m[2] = 100
			q.m[3] = 0
			q.m[4] = 0
			q.m[5] = 100
		}
	} else {
		scale := 1.0 / det

		// Compute adjugate matrix.
		a3 := y2 - y0
		a4 := x0 - x2
		a6 := y0 - y1
		a7 := x1 - x0

		// This performs the uv_pts*adjugate(control_pts) multiply, then does the scale by 1/det afterwards to improve
		// precision.
		q.m[0] = float32((0.5*a3 + a6) * scale)
		q.m[1] = float32((0.5*a4 + a7) * scale)
		q.m[2] = float32((0.5*a5 + a8) * scale)
		q.m[3] = float32(a6 * scale)
		q.m[4] = float32(a7 * scale)
		q.m[5] = float32(a8 * scale)
	}
}

// apply computes the UV coordinate for each position, writing to the parallel uv slice.
func (q *quadUVMatrix) apply(pos, uv []geom.Point) {
	sx := q.m[0]
	kx := q.m[1]
	tx := q.m[2]
	ky := q.m[3]
	sy := q.m[4]
	ty := q.m[5]
	for i := range pos {
		uv[i] = geom.Point{
			X: sx*pos[i].X + kx*pos[i].Y + tx,
			Y: ky*pos[i].X + sy*pos[i].Y + ty,
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// Conic KLM

// getConicKLM returns the (k, l, m) line functionals of the Loop-Blinn implicit form f(x, y) = k^2 - l*m for the conic
// through p with the given weight, returned as the rows of a row-major 3x3 (k = row 0, l = row 1, m = row 2), scaled so
// the max absolute coefficient is 10.
//
//	k = (y2 - y0, x0 - x2, x2*y0 - x0*y2)
//	l = (y1 - y0, x0 - x1, x1*y0 - x0*y1) * 2*w
//	m = (y2 - y1, x1 - x2, x2*y1 - x1*y2) * 2*w
func getConicKLM(p *[3]geom.Point, weight float32) [9]float32 {
	var klm [9]float32
	w2 := 2 * weight
	klm[0] = p[2].Y - p[0].Y
	klm[1] = p[0].X - p[2].X
	klm[2] = p[2].X*p[0].Y - p[0].X*p[2].Y

	klm[3] = w2 * (p[1].Y - p[0].Y)
	klm[4] = w2 * (p[0].X - p[1].X)
	klm[5] = w2 * (p[1].X*p[0].Y - p[0].X*p[1].Y)

	klm[6] = w2 * (p[2].Y - p[1].Y)
	klm[7] = w2 * (p[1].X - p[2].X)
	klm[8] = w2 * (p[2].X*p[1].Y - p[1].X*p[2].Y)

	// Scale the max absolute value of coeffs to 10.
	scale := float32(0)
	for i := 0; i < 9; i++ {
		scale = max(scale, geom.ScalarAbs(klm[i]))
	}
	scale = 10 / scale
	for i := 0; i < 9; i++ {
		klm[i] *= scale
	}
	return klm
}

//////////////////////////////////////////////////////////////////////////////
// Cubic → quads

// isPointWithinCubicTangents reports whether p lies within the cubic's tangent lines. a is the first control point of
// the cubic, ab the vector from a to the second control point, dc the vector from the fourth to the third control
// point, d the fourth control point, and p the candidate quadratic control point. Assumes the cubic doesn't inflect and
// is simple.
func isPointWithinCubicTangents(a, ab, dc, d geom.Point, dir path.FirstDirection, p geom.Point) bool {
	ap := p.Sub(a)
	apXab := ap.Cross(ab)
	if dir == path.FirstDirectionCW {
		if apXab > 0 {
			return false
		}
	} else {
		if apXab < 0 {
			return false
		}
	}

	dp := p.Sub(d)
	dpXdc := dp.Cross(dc)
	if dir == path.FirstDirectionCW {
		if dpXdc < 0 {
			return false
		}
	} else {
		if dpXdc > 0 {
			return false
		}
	}
	return true
}

// cubicToQuadsLengthScale and cubicToQuadsMaxSubdivs bound the cubic-to-quads control-point extrapolation and recursion
// depth.
const (
	cubicToQuadsLengthScale = float32(3) / 2
	cubicToQuadsMaxSubdivs  = 10
)

// convertNoninflectCubicToQuads approximates a cubic with no inflection points as a sequence of quads appended to quads
// as triples of points.
func convertNoninflectCubicToQuads(p *[4]geom.Point, toleranceSqd float32, quads *[]geom.Point, sublevel int, preserveFirstTangent, preserveLastTangent bool) {
	// Notation: Point a is always p[0]. Point b is p[1] unless p[1] == p[0], in which case it is p[2]. Point d is
	// always p[3]. Point c is p[2] unless p[2] == p[3], in which case it is p[1].
	ab := p[1].Sub(p[0])
	dc := p[2].Sub(p[3])

	if ab.LengthSqd() < geom.ScalarNearlyZeroTol {
		if dc.LengthSqd() < geom.ScalarNearlyZeroTol {
			*quads = append(*quads, p[0], p[0], p[3])
			return
		}
		ab = p[2].Sub(p[0])
	}
	if dc.LengthSqd() < geom.ScalarNearlyZeroTol {
		dc = p[1].Sub(p[3])
	}

	ab = ab.Scaled(cubicToQuadsLengthScale)
	dc = dc.Scaled(cubicToQuadsLengthScale)

	// c0 and c1 are extrapolations along vectors ab and dc.
	c0 := p[0].Add(ab)
	c1 := p[3].Add(dc)

	dSqd := float32(0)
	if sublevel <= cubicToQuadsMaxSubdivs {
		dSqd = c0.DistanceToSqd(c1)
	}
	if dSqd < toleranceSqd {
		var newC geom.Point
		switch {
		case preserveFirstTangent == preserveLastTangent:
			// We used to force a split when both tangents need to be preserved and c0 != c1. This introduced a large
			// performance regression for tiny paths for no noticeable quality improvement. However, we aren't quite
			// fulfilling our contract of guaranteeing the two tangent vectors and this could introduce a missed pixel
			// in AAHairlinePathRenderer.
			newC = c0.Add(c1).Scaled(0.5)
		case preserveFirstTangent:
			newC = c0
		default:
			newC = c1
		}
		*quads = append(*quads, p[0], newC, p[3])
		return
	}
	var choppedPts [7]geom.Point
	geom.ChopCubicAtHalf(p[:], choppedPts[:])
	var first, second [4]geom.Point
	copy(first[:], choppedPts[0:4])
	copy(second[:], choppedPts[3:7])
	convertNoninflectCubicToQuads(&first, toleranceSqd, quads, sublevel+1,
		preserveFirstTangent, false)
	convertNoninflectCubicToQuads(&second, toleranceSqd, quads, sublevel+1, false,
		preserveLastTangent)
}

// convertNoninflectCubicToQuadsWithConstraint is convertNoninflectCubicToQuads constrained to keep every generated quad
// control point within the cubic's tangent lines.
func convertNoninflectCubicToQuadsWithConstraint(p *[4]geom.Point, toleranceSqd float32, dir path.FirstDirection, quads *[]geom.Point, sublevel int) {
	// Notation: Point a is always p[0]. Point b is p[1] unless p[1] == p[0], in which case it is p[2]. Point d is
	// always p[3]. Point c is p[2] unless p[2] == p[3], in which case it is p[1].
	ab := p[1].Sub(p[0])
	dc := p[2].Sub(p[3])

	if ab.LengthSqd() < geom.ScalarNearlyZeroTol {
		if dc.LengthSqd() < geom.ScalarNearlyZeroTol {
			*quads = append(*quads, p[0], p[0], p[3])
			return
		}
		ab = p[2].Sub(p[0])
	}
	if dc.LengthSqd() < geom.ScalarNearlyZeroTol {
		dc = p[1].Sub(p[3])
	}

	// When the ab and cd tangents are degenerate or nearly parallel with vector from d to a the constraint that the
	// quad point falls between the tangents becomes hard to enforce and we are likely to hit the max subdivision count.
	// However, in this case the cubic is approaching a line and the accuracy of the quad point isn't so important. We
	// check if the two middle cubic control points are very close to the baseline vector. If so then we just pick
	// quadratic points on the control polygon.
	da := p[0].Sub(p[3])
	doQuads := dc.LengthSqd() < geom.ScalarNearlyZeroTol ||
		ab.LengthSqd() < geom.ScalarNearlyZeroTol
	if !doQuads {
		invDALengthSqd := da.LengthSqd()
		if invDALengthSqd > geom.ScalarNearlyZeroTol {
			invDALengthSqd = geom.ScalarInvert(invDALengthSqd)
			// cross(ab, da)^2/length(da)^2 == sqd distance from b to line from d to a. Same goes for point c using
			// vector cd.
			detABSqd := ab.Cross(da)
			detABSqd *= detABSqd
			detDCSqd := dc.Cross(da)
			detDCSqd *= detDCSqd
			if detABSqd*invDALengthSqd < toleranceSqd &&
				detDCSqd*invDALengthSqd < toleranceSqd {
				doQuads = true
			}
		}
	}
	if doQuads {
		b := p[0].Add(ab)
		c := p[3].Add(dc)
		mid := b.Add(c).Scaled(0.5)
		// Insert two quadratics to cover the case when ab points away from d and/or dc points away from a.
		if da.Dot(dc) < 0 || ab.Dot(da) > 0 {
			*quads = append(*quads, p[0], b, mid, mid, c, p[3])
		} else {
			*quads = append(*quads, p[0], mid, p[3])
		}
		return
	}

	ab = ab.Scaled(cubicToQuadsLengthScale)
	dc = dc.Scaled(cubicToQuadsLengthScale)

	// c0 and c1 are extrapolations along vectors ab and dc.
	c0 := p[0].Add(ab)
	c1 := p[3].Add(dc)

	dSqd := float32(0)
	if sublevel <= cubicToQuadsMaxSubdivs {
		dSqd = c0.DistanceToSqd(c1)
	}
	if dSqd < toleranceSqd {
		cAvg := c0.Add(c1).Scaled(0.5)
		subdivide := false

		if !isPointWithinCubicTangents(p[0], ab, dc, p[3], dir, cAvg) {
			// Choose a new cAvg that is the intersection of the two tangent lines.
			ab = ab.MakeOrthog(geom.PointSideLeft)
			z0 := -ab.Dot(p[0])
			dc = dc.MakeOrthog(geom.PointSideLeft)
			z1 := -dc.Dot(p[3])
			cAvg.X = ab.Y*z1 - z0*dc.Y
			cAvg.Y = z0*dc.X - ab.X*z1
			z := ab.X*dc.Y - ab.Y*dc.X
			z = geom.IEEEFloatDivide(1, z)
			cAvg.X *= z
			cAvg.Y *= z
			if sublevel <= cubicToQuadsMaxSubdivs {
				d0Sqd := c0.DistanceToSqd(cAvg)
				d1Sqd := c1.DistanceToSqd(cAvg)
				// We need to subdivide if d0 + d1 > tolerance but we have the sqd values. We know the distances and
				// tolerance can't be negative. (d0 + d1)^2 > toleranceSqd d0Sqd + 2*d0*d1 + d1Sqd > toleranceSqd
				d0d1 := geom.ScalarSqrt(d0Sqd * d1Sqd)
				subdivide = 2*d0d1+d0Sqd+d1Sqd > toleranceSqd
			}
		}
		if !subdivide {
			*quads = append(*quads, p[0], cAvg, p[3])
			return
		}
	}
	var choppedPts [7]geom.Point
	geom.ChopCubicAtHalf(p[:], choppedPts[:])
	var first, second [4]geom.Point
	copy(first[:], choppedPts[0:4])
	copy(second[:], choppedPts[3:7])
	convertNoninflectCubicToQuadsWithConstraint(&first, toleranceSqd, dir, quads, sublevel+1)
	convertNoninflectCubicToQuadsWithConstraint(&second, toleranceSqd, dir, quads, sublevel+1)
}

// convertCubicToQuads approximates the cubic with a sequence of quads appended to quads as triples of points.
func convertCubicToQuads(p *[4]geom.Point, tolScale float32, quads *[]geom.Point) {
	if !p[0].IsFinite() || !p[1].IsFinite() || !p[2].IsFinite() || !p[3].IsFinite() {
		return
	}
	if !geom.IsFinite(tolScale) {
		return
	}
	var chopped [10]geom.Point
	count := geom.ChopCubicAtInflections(p[:], chopped[:])

	tolSqd := tolScale * tolScale

	for i := 0; i < count; i++ {
		var cubic [4]geom.Point
		copy(cubic[:], chopped[3*i:3*i+4])
		convertNoninflectCubicToQuads(&cubic, tolSqd, quads, 0, true, true)
	}
}

// convertCubicToQuadsConstrainToTangents is convertCubicToQuads, but the resulting quads are constrained to stay within
// the cubic's tangent lines (dir is the cubic's contour winding direction). Used for bounding the AA convex fill
// geometry.
func convertCubicToQuadsConstrainToTangents(p *[4]geom.Point, tolScale float32, dir path.FirstDirection, quads *[]geom.Point) {
	if !p[0].IsFinite() || !p[1].IsFinite() || !p[2].IsFinite() || !p[3].IsFinite() {
		return
	}
	if !geom.IsFinite(tolScale) {
		return
	}
	var chopped [10]geom.Point
	count := geom.ChopCubicAtInflections(p[:], chopped[:])

	tolSqd := tolScale * tolScale

	for i := 0; i < count; i++ {
		var cubic [4]geom.Point
		copy(cubic[:], chopped[3*i:3*i+4])
		convertNoninflectCubicToQuadsWithConstraint(&cubic, tolSqd, dir, quads, 0)
	}
}
