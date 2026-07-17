// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The shared fixed-count tessellation support layer: the tessellation constants and patch-attrib layout, the
// pre-chopper that bounds per-curve segment counts (flattening fully offscreen curves to lines), the convex-180 cubic
// chopper the stroke tessellator consumes, the tolerance accumulator that converts Wang's-formula results into
// fixed-count vertex counts, and the per-contour midpoint parser used by wedge tessellation. SIMD float2/float4 lanes
// reduce to element-wise float32 expressions.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// tessPrecision bounds linearized segments to within 1/4th of a pixel of the true curve.
const tessPrecision float32 = 4

// tessMaxResolveLevel is the maximum number of subdivisions of a Bezier curve that can be represented in the
// fixed-count vertex and index buffers. If rendering a curve that requires more subdivisions, it must be chopped.
const tessMaxResolveLevel = 5

// tessMaxParametricSegments (and its powers) is the maximum number of parametric segments (linear sections) that a
// curve can be split into.
const (
	tessMaxParametricSegments   = 1 << tessMaxResolveLevel
	tessMaxParametricSegmentsP2 = tessMaxParametricSegments * tessMaxParametricSegments
	tessMaxParametricSegmentsP4 = tessMaxParametricSegmentsP2 * tessMaxParametricSegmentsP2
)

// tessMaxSegmentsPerCurve (and its powers): don't tessellate paths that might have an individual curve that requires
// more than 1024 segments — call preChopPathCurves first. Standard chopping, when Wang's formula is between
// tessMaxParametricSegments and tessMaxSegmentsPerCurve, is handled automatically by patchWriter; it differs from
// preChopPathCurves in that it does no culling of offscreen chopped curves.
const (
	tessMaxSegmentsPerCurve   float32 = 1024
	tessMaxSegmentsPerCurveP2         = tessMaxSegmentsPerCurve * tessMaxSegmentsPerCurve
	tessMaxSegmentsPerCurveP4         = tessMaxSegmentsPerCurveP2 * tessMaxSegmentsPerCurveP2
)

// numCurveTrianglesAtResolveLevel returns how many triangles are in a curve with 2^resolveLevel line segments.
//
//	resolveLevel=0 -> 0 line segments -> 0 triangles
//	resolveLevel=1 -> 2 line segments -> 1 triangle
//	resolveLevel=2 -> 4 line segments -> 3 triangles
//	...
func numCurveTrianglesAtResolveLevel(resolveLevel int) int {
	return (1 << resolveLevel) - 1
}

// PatchAttribs are optional attribs that are included in tessellation patches, following the control points and in the
// same order as they appear here. PatchAttribPaintDepth and PatchAttribSsboIndex are unused by this backend; they keep
// their bit values for key fidelity.
type PatchAttribs uint32

// PatchAttribs values.
const (
	PatchAttribsNone              PatchAttribs = 0
	PatchAttribJoinControlPoint   PatchAttribs = 1 << 0 // [float2] strokes; defines tangent direction
	PatchAttribFanPoint           PatchAttribs = 1 << 1 // [float2] wedges; the center the wedges fan around
	PatchAttribStrokeParams       PatchAttribs = 1 << 2 // [float2] strokes with differing widths/joins
	PatchAttribColor              PatchAttribs = 1 << 3 // [ubyte4 or float4] per-patch colors
	PatchAttribPaintDepth         PatchAttribs = 1 << 4 // unused by this backend
	PatchAttribExplicitCurveType  PatchAttribs = 1 << 5 // [float] GPU can't infer curve type from infinity
	PatchAttribWideColorIfEnabled PatchAttribs = 1 << 6 // flag: kColor is float4 wide color
	PatchAttribSsboIndex          PatchAttribs = 1 << 7 // unused by this backend
)

// Explicit curve types: when PatchAttribExplicitCurveType is set, these are the values that tell the GPU what type of
// curve is being drawn.
const (
	tessCubicCurveType           float32 = 0
	tessConicCurveType           float32 = 1
	tessTriangularConicCurveType float32 = 2 // conic curve with w=Inf
)

// patchAttribsStride returns the packed size in bytes of the attribs portion of tessellation patches (or instances) in
// GPU buffers.
func patchAttribsStride(attribs PatchAttribs) int {
	stride := 0
	if attribs&PatchAttribJoinControlPoint != 0 {
		stride += 8
	}
	if attribs&PatchAttribFanPoint != 0 {
		stride += 8
	}
	if attribs&PatchAttribStrokeParams != 0 {
		stride += 8
	}
	if attribs&PatchAttribColor != 0 {
		if attribs&PatchAttribWideColorIfEnabled != 0 {
			stride += 16
		} else {
			stride += 4
		}
	}
	if attribs&PatchAttribPaintDepth != 0 {
		stride += 4
	}
	if attribs&PatchAttribExplicitCurveType != 0 {
		stride += 4
	}
	if attribs&PatchAttribSsboIndex != 0 {
		stride += 8
	}
	return stride
}

// patchStride returns the byte size of one patch: 4 control points plus the attribs.
func patchStride(attribs PatchAttribs) int {
	return 4*8 + patchAttribsStride(attribs)
}

//////////////////////////////////////////////////////////////////////////////
// findCubicConvex180Chops / conicHasCusp

// findCubicConvex180Chops finds 0, 1, or 2 T values at which to chop the given curve in order to guarantee the
// resulting cubics are convex and rotate no more than 180 degrees.
//
//   - If the cubic is "serpentine", then the T values are any inflection points in [0 < T < 1].
//   - If the cubic is linear, then the T values are any 180-degree cusp points in [0 < T < 1].
//   - Otherwise the T value is the point at which rotation reaches 180 degrees, iff in [0 < T < 1].
//
// areCusps is set to true if the chop point occurred at a cusp (within tolerance), or if the chop point(s) occurred at
// 180-degree turnaround points on a degenerate flat line.
func findCubicConvex180Chops(pts *[4]geom.Point, t *[2]float32, areCusps *bool) int {
	// If a chop falls within a distance of "kEpsilon" from 0 or 1, throw it out. Tangents become unstable when we chop
	// too close to the boundary. This works out because the tessellation shaders don't allow more than 2^10 parametric
	// segments, and they snap the beginning and ending edges at 0 and 1. So if we overstep an inflection or point of
	// 180-degree rotation by a fraction of a tessellation segment, it just gets snapped.
	const epsilon float32 = 1.0 / (1 << 11)
	// Floating-point representation of "1 - 2*kEpsilon".
	const ieeeOneMinus2Epsilon uint32 = (127 << 23) - 2*(1<<(24-11))

	p0, p1, p2, p3 := pts[0], pts[1], pts[2], pts[3]

	// Find the cubic's power basis coefficients. These define the bezier curve as:
	//
	//	                              |T^3|
	//	Cubic(T) = x,y = |A  3B  3C| *|T^2| + P0
	//	                 |.   .   .|  |T  |
	//
	// And the tangent direction (scaled by a uniform 1/3) will be:
	//
	//	                                          |T^2|
	//	Tangent_Direction(T) = dx,dy = |A  2B  C| *|T  |
	//	                               |.   .  .|  |1  |
	cx, cy := p1.X-p0.X, p1.Y-p0.Y
	dx, dy := p2.X-p1.X, p2.Y-p1.Y
	ex, ey := p3.X-p0.X, p3.Y-p0.Y
	bx, by := dx-cx, dy-cy
	ax, ay := -3*dx+ex, -3*dy+ey

	cross := func(x0, y0, x1, y1 float32) float32 { return x0*y1 - y0*x1 }
	dot := func(x0, y0, x1, y1 float32) float32 { return x0*x1 + y0*y1 }

	// Now find the cubic's inflection function. There are inflections where F' x F'' == 0. We formulate this as a
	// quadratic equation: F' x F'' == aT^2 + bT + c == 0. See:
	// https://www.microsoft.com/en-us/research/wp-content/uploads/2005/01/p1000-loop.pdf NOTE: We only need the roots,
	// so a uniform scale factor does not affect the solution.
	a := cross(ax, ay, bx, by)
	b := cross(ax, ay, cx, cy)
	c := cross(bx, by, cx, cy)
	bOverMinus2 := -0.5 * b
	discrOver4 := bOverMinus2*bOverMinus2 - a*c

	// If -cuspThreshold <= discrOver4 <= cuspThreshold, the two roots are within kEpsilon of one another (in parametric
	// space). This is close enough for our purposes to consider them a single cusp.
	cuspThreshold := a * (epsilon / 2)
	cuspThreshold *= cuspThreshold

	if discrOver4 < -cuspThreshold {
		// The curve does not inflect or cusp. This means it might rotate more than 180 degrees instead. Chop where
		// rotation == 180 deg. (This is the 2nd root where the tangent is parallel to tan0.)
		//
		//	Tangent_Direction(T) x tan0 == 0
		//	(AT^2 x tan0) + (2BT x tan0) + (C x tan0) == 0
		//	(A x C)T^2 + (2B x C)T + (C x C) == 0  [[because tan0 == P1 - P0 == C]]
		//	bT^2 + 2cT + 0 == 0  [[because A x C == b, B x C == c]]
		//	T = [0, -2c/b]
		//
		// NOTE: if C == 0, then C != tan0. But this is fine because the curve is definitely convex-180 if any points
		// are colocated, and T[0] will equal NaN which returns 0 chops.
		*areCusps = false
		root := c / bOverMinus2 // plain float division
		// Is "root" inside the range [kEpsilon, 1 - kEpsilon)?
		if math.Float32bits(root-epsilon) < ieeeOneMinus2Epsilon {
			t[0] = root
			return 1
		}
		return 0
	}

	*areCusps = discrOver4 <= cuspThreshold
	if *areCusps {
		// The two roots are close enough that we can consider them a single cusp.
		if a != 0 || bOverMinus2 != 0 || c != 0 {
			// Pick the average of both roots.
			root := bOverMinus2 / a // plain float division
			// Is "root" inside the range [kEpsilon, 1 - kEpsilon)?
			if math.Float32bits(root-epsilon) < ieeeOneMinus2Epsilon {
				t[0] = root
				return 1
			}
			return 0
		}

		// The curve is a flat line. The standard inflection function doesn't detect cusps from flat lines. Find cusps
		// by searching instead for points where the tangent is perpendicular to tan0. This will find any cusp point.
		//
		//	dot(tan0, Tangent_Direction(T)) == 0
		//
		//	                    |T^2|
		//	tan0 * |A  2B  C| * |T  | == 0
		//	       |.   .  .|   |1  |
		tan0x, tan0y := cx, cy
		if cx == 0 {
			tan0x = p2.X - p0.X // selects C when nonzero, else p2 - p0, per lane.
		}
		if cy == 0 {
			tan0y = p2.Y - p0.Y
		}
		a = dot(tan0x, tan0y, ax, ay)
		bOverMinus2 = -dot(tan0x, tan0y, bx, by)
		c = dot(tan0x, tan0y, cx, cy)
		discrOver4 = bOverMinus2*bOverMinus2 - a*c
		if discrOver4 < -cuspThreshold {
			// With the updated discriminant, this line actually wouldn't have cusps (e.g., it never turns back on
			// itself).
			return 0
		}
		discrOver4 = max32(discrOver4, 0)
	}

	// Solve our quadratic equation to find where to chop. See the quadratic formula from Numerical Recipes in C.
	q := sqrt32(discrOver4)
	q = float32(math.Copysign(float64(q), float64(bOverMinus2)))
	q += bOverMinus2
	root0 := q / a
	root1 := c / q

	inside0 := root0 > epsilon && root0 < 1-epsilon
	inside1 := root1 > epsilon && root1 < 1-epsilon
	if inside0 {
		if inside1 && root0 != root1 {
			if root0 > root1 {
				root0, root1 = root1, root0 // Sort.
			}
			t[0] = root0
			t[1] = root1
			return 2
		}
		t[0] = root0
		return 1
	}
	if inside1 {
		t[0] = root1
		return 1
	}
	return 0
}

// conicHasCusp reports whether the given conic (or quadratic) has a cusp point. The w value is not necessary in
// determining this. If there is a cusp, it can be found at the midtangent.
func conicHasCusp(p *[3]geom.Point) bool {
	ax, ay := p[1].X-p[0].X, p[1].Y-p[0].Y
	bx, by := p[2].X-p[1].X, p[2].Y-p[1].Y
	// A conic of any class can only have a cusp if it is a degenerate flat line with a 180 degree turnaround. To detect
	// this, the beginning and ending tangents must be parallel (a.cross(b) == 0) and pointing in opposite directions
	// (a.dot(b) < 0).
	return ax*by-ay*bx == 0 && ax*bx+ay*by < 0
}

//////////////////////////////////////////////////////////////////////////////
// Stroke parameters

// tessGetJoinType encodes all of a join's information in a single float value:
//
//	Negative => Round Join
//	Zero     => Bevel Join
//	Positive => Miter join, and the value is also the miter limit
func tessGetJoinType(rec *stroke.Rec) float32 {
	switch rec.Join() {
	case stroke.JoinRound:
		return -1
	case stroke.JoinBevel:
		return 0
	case stroke.JoinMiter:
		return rec.Miter()
	}
	panic("unreachable join type")
}

// strokeParams is written out with each patch/instance when PatchAttribStrokeParams is enabled.
type strokeParams struct {
	radius   float32
	joinType float32 // see tessGetJoinType
}

// makeStrokeParams derives strokeParams from a stroke description.
func makeStrokeParams(rec *stroke.Rec) strokeParams {
	return strokeParams{radius: rec.Width() * 0.5, joinType: tessGetJoinType(rec)}
}

// tessNumFixedEdgesInJoin returns the fixed number of edges that are always emitted with the given join type. If the
// join is round, the caller needs to account for the additional radial edges on their own.
func tessNumFixedEdgesInJoin(params strokeParams) int {
	if params.joinType > 0 { // miter
		return 4
	}
	return 3 // round or bevel
}

// tessNumFixedEdgesInJoinType returns the fixed number of edges for a join type alone (no stroke params). Each join
// always emits two colocated edges at the beginning (a full-width edge to seam with the preceding stroke and a
// half-width edge to begin the join), an extra edge in the middle for miter joins (round joins' variable radial edges
// are the caller's responsibility), and a half-width edge at the end that will be colocated with the first edge of the
// stroke.
func tessNumFixedEdgesInJoinType(join stroke.Join) int {
	if join == stroke.JoinMiter {
		return 4
	}
	return 3 // round or bevel
}

// strokesHaveEqualParams reports whether two strokes can share the same (uniform or per-patch) StrokeParams.
func strokesHaveEqualParams(a, b *stroke.Rec) bool {
	return a.Width() == b.Width() && a.Join() == b.Join() &&
		(a.Join() != stroke.JoinMiter || a.Miter() == b.Miter())
}

// tessCalcNumRadialSegmentsPerRadian returns the number of radial segments the tessellator adds for each radian of
// rotation in local path space (uniform steps in tangent angle).
func tessCalcNumRadialSegmentsPerRadian(approxDevStrokeRadius float32) float32 {
	cosTheta := 1 - (1/tessPrecision)/approxDevStrokeRadius
	return 0.5 / float32(math.Acos(float64(max32(cosTheta, -1))))
}

//////////////////////////////////////////////////////////////////////////////
// linearTolerances

// linearTolerances stores state to approximate the final device-space transform applied to curves, and uses that to
// calculate segmentation levels for both the parametric curves and radial components (when stroking). These tolerances
// determine the worst-case number of parametric and radial segments required to accurately linearize curves. The
// tolerance values are estimated in local path space via the 2x2 vector transform that approximates the scale/skew of
// the full local-to-device transform applied in the vertex shader.
type linearTolerances struct {
	// numParametricSegmentsP4 is used for both fills and strokes; always at least one segment.
	numParametricSegmentsP4 float32
	// numRadialSegmentsPerRadian is used for strokes, adding additional segments along the curve to account for its
	// rotation.
	numRadialSegmentsPerRadian float32
	// edgesInJoins is used for strokes, tracking the number of additional vertices required to handle joins based on
	// the join type and stroke width.
	edgesInJoins int
}

// newLinearTolerances returns tolerances with the default field values.
func newLinearTolerances() linearTolerances {
	return linearTolerances{numParametricSegmentsP4: 1}
}

// requiredResolveLevel returns a fast log2 of the minimum required number of segments per the tracked Wang's-formula
// calculations. log16(n^4) == log2(n).
func (t *linearTolerances) requiredResolveLevel() int {
	return wangsNextLog16(t.numParametricSegmentsP4)
}

// requiredStrokeEdges returns the total number of edges a stroke instance needs, combining join and stroke segments.
func (t *linearTolerances) requiredStrokeEdges() int {
	// The maximum rotation we can have in a stroke is 180 degrees (pi radians). NOTE: this is also sufficient to handle
	// circular caps because the shader sweeps a stroke-width line 180 degrees about the center point.
	maxRadialSegmentsInStroke := max(int(geom.CeilToInt(t.numRadialSegmentsPerRadian*math.Pi)), 1)

	maxParametricSegmentsInStroke := int(geom.CeilToInt(wangsRoot4(t.numParametricSegmentsP4)))

	// The first and last edges in a stroke are shared by both the parametric and radial sets of edges, so the total
	// number of edges is:
	//
	//	numCombinedEdges = numParametricEdges + numRadialEdges - 2
	//
	// It's important to differentiate between the number of edges and segments in a strip:
	//
	//	numSegments = numEdges - 1
	//
	// So the total number of combined edges in the stroke is:
	//
	//	numEdgesInStroke = numParametricSegments + 1 + numRadialSegments + 1 - 2
	//	                 = numParametricSegments + numRadialSegments
	maxEdgesInStroke := maxRadialSegmentsInStroke + maxParametricSegmentsInStroke

	// Each triangle strip has two sections: it starts with a join then transitions to a stroke. The number of edges in
	// an instance is the sum of edges from the join and stroke sections both. NOTE: the final join edge and the first
	// stroke edge are co-located, however we still need to emit both because the join's edge is half-width and the
	// stroke is full-width.
	return t.edgesInJoins + maxEdgesInStroke
}

// setParametricSegments records the parametric segment count (already raised to the 4th power).
func (t *linearTolerances) setParametricSegments(n4 float32) {
	t.numParametricSegmentsP4 = n4
}

// setStroke records the radial-segment and join-edge tolerances for a stroke at the given max device scale.
func (t *linearTolerances) setStroke(params strokeParams, maxScale float32) {
	var approxDeviceStrokeRadius float32
	if params.radius == 0 {
		// Hairlines are always 1 px wide.
		approxDeviceStrokeRadius = 0.5
	} else {
		// Approximate max scale * local stroke width / 2.
		approxDeviceStrokeRadius = params.radius * maxScale
	}
	t.numRadialSegmentsPerRadian = tessCalcNumRadialSegmentsPerRadian(approxDeviceStrokeRadius)
	t.edgesInJoins = tessNumFixedEdgesInJoin(params)
	if params.joinType < 0 && t.numRadialSegmentsPerRadian > 0 {
		// For round joins we need to count the radial edges on our own. Account for a worst-case join of 180 degrees
		// (pi radians).
		t.edgesInJoins += int(geom.CeilToInt(t.numRadialSegmentsPerRadian*math.Pi)) - 1
	}
}

// accumulate widens t to cover the tolerances of other, keeping the maximum of each field.
func (t *linearTolerances) accumulate(other *linearTolerances) {
	if other.numParametricSegmentsP4 > t.numParametricSegmentsP4 {
		t.numParametricSegmentsP4 = other.numParametricSegmentsP4
	}
	if other.numRadialSegmentsPerRadian > t.numRadialSegmentsPerRadian {
		t.numRadialSegmentsPerRadian = other.numRadialSegmentsPerRadian
	}
	if other.edgesInJoins > t.edgesInJoins {
		t.edgesInJoins = other.edgesInJoins
	}
}

//////////////////////////////////////////////////////////////////////////////
// cullTest

// cullTest determines whether the given local-space points will be contained in the cull bounds post transform. For the
// versions that take >1 point, it returns whether any region of their device-space bounding box will be in the cull
// bounds.
//
// NOTE: the stored "matrix" is not a normal matrix: M*p maps to the float4 [x, y, -x, -y] in device space, to aid in
// quick bounds calculations. It also carries no translation; the translation is unapplied from the cull bounds ahead of
// time.
type cullTest struct {
	// matX/matY map path coordinates to the float4 [x, y, -x, -y] in device space.
	matX [4]float32
	matY [4]float32
	// cullBounds is [l, t, -r, -b] with the matrix translation pre-subtracted.
	cullBounds [4]float32
}

// makeCullTest builds a cullTest for devCullBounds under matrix m. The matrix must not have perspective.
func makeCullTest(devCullBounds geom.Rect, m *geom.Matrix) cullTest {
	if m.HasPerspective() {
		panic("CullTest requires a non-perspective matrix")
	}
	sx, ky := m.Get(geom.MScaleX), m.Get(geom.MSkewY)
	kx, sy := m.Get(geom.MSkewX), m.Get(geom.MScaleY)
	tx, ty := m.Get(geom.MTransX), m.Get(geom.MTransY)
	return cullTest{
		matX:       [4]float32{sx, ky, -sx, -ky},
		matY:       [4]float32{kx, sy, -kx, -sy},
		cullBounds: [4]float32{devCullBounds.Left - tx, devCullBounds.Top - ty, tx - devCullBounds.Right, ty - devCullBounds.Bottom},
	}
}

// mapToDev computes matX*p.X + matY*p.Y: [x, y, -x, -y] in device space.
func (c *cullTest) mapToDev(p geom.Point) [4]float32 {
	return [4]float32{
		c.matX[0]*p.X + c.matY[0]*p.Y,
		c.matX[1]*p.X + c.matY[1]*p.Y,
		c.matX[2]*p.X + c.matY[2]*p.Y,
		c.matX[3]*p.X + c.matY[3]*p.Y,
	}
}

// areVisible3 reports whether any region of the bounding box of M * p0..2 will be in the viewport.
func (c *cullTest) areVisible3(p *[3]geom.Point) bool {
	val0 := c.mapToDev(p[0])
	val1 := c.mapToDev(p[1])
	val2 := c.mapToDev(p[2])
	for i := range val0 {
		// After the maxes: val0 = [r, b, -l, -t] of the device-space bounding box of p0..2.
		val0[i] = max32(max32(val0[i], val1[i]), val2[i])
		// Does cullBounds intersect the bounding box? i.e., l0 < r1 && t0 < b1 && r0 > l1 && b0 > t1.
		if !(c.cullBounds[i] < val0[i]) {
			return false
		}
	}
	return true
}

// areVisible4 is areVisible3 for 4 points.
func (c *cullTest) areVisible4(p *[4]geom.Point) bool {
	val0 := c.mapToDev(p[0])
	val1 := c.mapToDev(p[1])
	val2 := c.mapToDev(p[2])
	val3 := c.mapToDev(p[3])
	for i := range val0 {
		val0[i] = max32(max32(val0[i], val1[i]), max32(val2[i], val3[i]))
		if !(c.cullBounds[i] < val0[i]) {
			return false
		}
	}
	return true
}

//////////////////////////////////////////////////////////////////////////////
// tessAffineMatrix

// tessAffineMatrix applies an affine 2d transformation to points. Per-point float32 math reproduces the same result
// regardless of how many points are mapped at once.
type tessAffineMatrix struct {
	scaleX, scaleY float32
	skewX, skewY   float32
	transX, transY float32
}

// makeTessAffineMatrix builds a tessAffineMatrix from m. The matrix must not have perspective.
func makeTessAffineMatrix(m *geom.Matrix) tessAffineMatrix {
	if m.HasPerspective() {
		panic("AffineMatrix requires a non-perspective matrix")
	}
	return tessAffineMatrix{
		scaleX: m.Get(geom.MScaleX), scaleY: m.Get(geom.MScaleY),
		skewX: m.Get(geom.MSkewX), skewY: m.Get(geom.MSkewY),
		transX: m.Get(geom.MTransX), transY: m.Get(geom.MTransY),
	}
}

// mapPoint applies the affine transform to p.
func (m *tessAffineMatrix) mapPoint(p geom.Point) geom.Point {
	return geom.Point{
		X: m.scaleX*p.X + (m.skewX*p.Y + m.transX),
		Y: m.scaleY*p.Y + (m.skewY*p.X + m.transY),
	}
}

//////////////////////////////////////////////////////////////////////////////
// midpointContourParser

// tessContourRecord is one decoded path record for midpointContourParser's contour iteration.
type tessContourRecord struct {
	verb   path.Verb
	pts    [4]geom.Point
	weight float32
}

// midpointContourParser parses out each contour in a path and tracks the midpoint (the mean of each verb's on-curve
// endpoint, plus the start point again when the contour closes implicitly). It decodes the records through path.RawIter
// up front and works over index ranges.
type midpointContourParser struct {
	records        []tessContourRecord
	idx            int // start of the not-yet-parsed remainder
	contourStart   int
	contourEnd     int
	midpoint       geom.Point
	midpointWeight int
}

// newMidpointContourParser decodes the path.
func newMidpointContourParser(p *path.Path) *midpointContourParser {
	m := &midpointContourParser{}
	it := path.NewRawIter(p)
	for {
		verb, pts, w, ok := it.Next()
		if !ok {
			break
		}
		m.records = append(m.records, tessContourRecord{verb: verb, pts: pts, weight: w})
	}
	return m
}

// parseNextContour advances to the next contour that has geometry, accumulating its midpoint. Returns false when there
// are no more contours.
func (m *midpointContourParser) parseNextContour() bool {
	hasGeometry := false
	var first, last geom.Point
	m.contourStart = m.idx
	for ; m.idx < len(m.records); m.idx++ {
		rec := &m.records[m.idx]
		switch rec.verb {
		case path.VerbMove:
			if !hasGeometry {
				m.midpoint = geom.Point{}
				m.midpointWeight = 0
				m.contourStart = m.idx
				first = rec.pts[0]
				last = rec.pts[0]
				continue
			}
			if first != last {
				// There's an implicit close at the end. Add the start point to our mean.
				m.midpoint = m.midpoint.Add(first)
				m.midpointWeight++
			}
			m.contourEnd = m.idx
			return true
		case path.VerbClose:
			continue
		case path.VerbLine:
			last = rec.pts[1]
		case path.VerbQuad, path.VerbConic:
			last = rec.pts[2]
		case path.VerbCubic:
			last = rec.pts[3]
		}
		m.midpoint = m.midpoint.Add(last)
		m.midpointWeight++
		hasGeometry = true
	}
	m.contourEnd = m.idx
	if hasGeometry && first != last {
		// There's an implicit close at the end. Add the start point to our mean.
		m.midpoint = m.midpoint.Add(first)
		m.midpointWeight++
	}
	return hasGeometry
}

// currentContour returns the records of the contour parseNextContour returned.
func (m *midpointContourParser) currentContour() []tessContourRecord {
	return m.records[m.contourStart:m.contourEnd]
}

// currentMidpoint returns the midpoint of the current contour.
func (m *midpointContourParser) currentMidpoint() geom.Point {
	inv := 1 / float32(m.midpointWeight)
	return geom.Point{X: m.midpoint.X * inv, Y: m.midpoint.Y * inv}
}

//////////////////////////////////////////////////////////////////////////////
// preChopPathCurves

// tessMaxChopsPerCurve only protects us against getting stuck in infinite recursion due to fp32 precision issues.
// Mathematically, every curve should reduce to manageable visible sections in O(log N) chops, where N is the magnitude
// of its control points. But, to define a protective upper bound: a cubic can enter or exit the viewport as many as 6
// times, so we may need to refine the curve (via binary-search chopping at T=.5) up to 6 times; and chopping a cubic at
// T=.5 may only reduce its length by 1/8 (.5^3), so we may require up to 6 chops in order to reduce the length by 1/2.
const tessMaxChopsPerCurve = 128 /*magnitude of +fp32_max - -fp32_max*/ *
	6 /*max chops to reduce the length by half*/ * 6 /*max viewport boundary crosses*/

// pathChopper writes a new path, chopping as necessary so no verbs require more segments than tessMaxSegmentsPerCurve.
// Curves completely outside the viewport are flattened into lines. The output path is marked volatile since it's
// transient tessellation input.
type pathChopper struct {
	builder *path.Path
	// Stack-based recursion (instead of the runtime stack).
	pointStack            []geom.Point
	weightStack           []float32
	cullTest              cullTest
	vectorXform           vectorXform
	tessellationPrecision float32
}

func newPathChopper(tessellationPrecision float32, m *geom.Matrix, viewport geom.Rect) *pathChopper {
	b := &path.Path{}
	b.SetVolatile(true)
	return &pathChopper{
		tessellationPrecision: tessellationPrecision,
		cullTest:              makeCullTest(viewport, m),
		vectorXform:           makeVectorXform(m),
		builder:               b,
	}
}

func (c *pathChopper) detachPath(ft path.FillType) *path.Path {
	c.builder.SetFillType(ft)
	return c.builder
}

func (c *pathChopper) moveTo(p geom.Point) { c.builder.MoveToPt(p) }
func (c *pathChopper) lineTo(p geom.Point) { c.builder.LineToPt(p) }
func (c *pathChopper) closeContour()       { c.builder.Close() }

// quadTo appends a quadratic curve, chopping it as needed.
func (c *pathChopper) quadTo(quad *[3]geom.Point) {
	c.pointStack = append(c.pointStack, quad[0], quad[1], quad[2])
	numChops := 0
	for len(c.pointStack) > 0 {
		p := [3]geom.Point(c.pointStack[len(c.pointStack)-3:])
		if !c.cullTest.areVisible3(&p) {
			c.builder.LineToPt(p[2])
		} else {
			n4 := wangsQuadraticP4(c.tessellationPrecision, p[0], p[1], p[2], c.vectorXform)
			if n4 > tessMaxSegmentsPerCurveP4 && numChops < tessMaxChopsPerCurve {
				var chops [5]geom.Point
				geom.ChopQuadAtHalf(p[:], chops[:])
				c.pointStack = c.pointStack[:len(c.pointStack)-3]
				c.pointStack = append(c.pointStack, chops[2], chops[3], chops[4], chops[0], chops[1], chops[2])
				numChops++
				continue
			}
			c.builder.QuadToPt(p[1], p[2])
		}
		c.pointStack = c.pointStack[:len(c.pointStack)-3]
	}
}

// conicTo appends a conic curve, chopping it as needed.
func (c *pathChopper) conicTo(conic *[3]geom.Point, weight float32) {
	c.pointStack = append(c.pointStack, conic[0], conic[1], conic[2])
	c.weightStack = append(c.weightStack, weight)
	numChops := 0
	for len(c.pointStack) > 0 {
		p := [3]geom.Point(c.pointStack[len(c.pointStack)-3:])
		w := c.weightStack[len(c.weightStack)-1]
		if !c.cullTest.areVisible3(&p) {
			c.builder.LineToPt(p[2])
		} else {
			n2 := wangsConicP2(c.tessellationPrecision, p[0], p[1], p[2], w, c.vectorXform)
			if n2 > tessMaxSegmentsPerCurveP2 && numChops < tessMaxChopsPerCurve {
				conicSrc := geom.MakeConic(p[0], p[1], p[2], w)
				var chops [2]geom.Conic
				if !conicSrc.ChopAt(0.5, &chops) {
					// Flatten to a line when the chop cannot produce finite values. Pop the stack entry so the
					// flattened line is emitted exactly once rather than re-testing the identical conic forever (a
					// latent infinite loop, reachable only for near-fp32-max control points that still straddle the
					// viewport).
					c.builder.LineToPt(p[2])
					c.pointStack = c.pointStack[:len(c.pointStack)-3]
					c.weightStack = c.weightStack[:len(c.weightStack)-1]
					continue
				}
				c.pointStack = c.pointStack[:len(c.pointStack)-3]
				c.weightStack = c.weightStack[:len(c.weightStack)-1]
				c.pointStack = append(c.pointStack, chops[1].Pts[0], chops[1].Pts[1], chops[1].Pts[2])
				c.weightStack = append(c.weightStack, chops[1].W)
				c.pointStack = append(c.pointStack, chops[0].Pts[0], chops[0].Pts[1], chops[0].Pts[2])
				c.weightStack = append(c.weightStack, chops[0].W)
				numChops++
				continue
			}
			c.builder.ConicToPt(p[1], p[2], w)
		}
		c.pointStack = c.pointStack[:len(c.pointStack)-3]
		c.weightStack = c.weightStack[:len(c.weightStack)-1]
	}
}

// cubicTo appends a cubic curve, chopping it as needed.
func (c *pathChopper) cubicTo(cubic *[4]geom.Point) {
	c.pointStack = append(c.pointStack, cubic[0], cubic[1], cubic[2], cubic[3])
	numChops := 0
	for len(c.pointStack) > 0 {
		p := [4]geom.Point(c.pointStack[len(c.pointStack)-4:])
		if !c.cullTest.areVisible4(&p) {
			c.builder.LineToPt(p[3])
		} else {
			n4 := wangsCubicP4(c.tessellationPrecision, p[0], p[1], p[2], p[3], c.vectorXform)
			if n4 > tessMaxSegmentsPerCurveP4 && numChops < tessMaxChopsPerCurve {
				var chops [7]geom.Point
				geom.ChopCubicAtHalf(p[:], chops[:])
				c.pointStack = c.pointStack[:len(c.pointStack)-4]
				c.pointStack = append(c.pointStack,
					chops[3], chops[4], chops[5], chops[6], chops[0], chops[1], chops[2], chops[3])
				numChops++
				continue
			}
			c.builder.CubicToPt(p[1], p[2], p[3])
		}
		c.pointStack = c.pointStack[:len(c.pointStack)-4]
	}
}

// preChopPathCurves returns a new path, equivalent to 'p' within the given viewport, whose verbs can all be drawn with
// tessMaxSegmentsPerCurve tessellation segments or fewer while staying within 1/tessellationPrecision pixels of the
// true curve. Curves and chops that fall completely outside the viewport are flattened into lines.
func preChopPathCurves(tessellationPrecision float32, p *path.Path, m *geom.Matrix, viewport geom.Rect) *path.Path {
	// If the viewport is exceptionally large, we could end up blowing out memory with an unbounded number of chops.
	// Therefore, we require that the viewport is manageable enough that a fully contained curve can be tessellated in
	// tessMaxSegmentsPerCurve or fewer. (Any larger and that amount of pixels wouldn't fit in memory anyway.)
	if wangsWorstCaseCubic(tessellationPrecision, viewport.Width(),
		viewport.Height()) > tessMaxSegmentsPerCurve {
		panic("preChopPathCurves viewport is too large")
	}
	chopper := newPathChopper(tessellationPrecision, m, viewport)
	it := path.NewRawIter(p)
	for {
		verb, pts, w, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			chopper.moveTo(pts[0])
		case path.VerbLine:
			chopper.lineTo(pts[1])
		case path.VerbQuad:
			quad := [3]geom.Point{pts[0], pts[1], pts[2]}
			chopper.quadTo(&quad)
		case path.VerbConic:
			conic := [3]geom.Point{pts[0], pts[1], pts[2]}
			chopper.conicTo(&conic, w)
		case path.VerbCubic:
			cubic := [4]geom.Point{pts[0], pts[1], pts[2], pts[3]}
			chopper.cubicTo(&cubic)
		case path.VerbClose:
			chopper.closeContour()
		}
	}
	// Must preserve the input path's fill type.
	return chopper.detachPath(p.FillType())
}
