// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// AA-type resolution, w=0 plane clipping, cropping quads to rects, and the TessellationHelper that insets/outsets quad
// edges for per-edge coverage AA — the geometric heart of the per-edge-AA quad system. SIMD-style vector math is
// expressed with the [4]float32 helpers from quad.go plus per-lane loops.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

const (
	// quadTolerance is the general tolerance used for denominators, checking div-by-0.
	quadTolerance = 1e-9
	// quadDistTolerance is the increased slop when comparing signed distances / lengths.
	quadDistTolerance    = 1e-2
	quadDist2Tolerance   = quadDistTolerance * quadDistTolerance
	quadInvDistTolerance = 1 / quadDistTolerance
	// quadW0PlaneDistance is the minimum w considered "in front of" the viewer for clipping.
	quadW0PlaneDistance = 1.0 / (1 << 14)
)

// f4 arithmetic helpers (per-lane float32, matching skvx semantics).

func f4Add(a, b [4]float32) [4]float32 {
	return [4]float32{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}
}

func f4Sub(a, b [4]float32) [4]float32 {
	return [4]float32{a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]}
}

func f4Mul(a, b [4]float32) [4]float32 {
	return [4]float32{a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]}
}

func f4Div(a, b [4]float32) [4]float32 {
	return [4]float32{a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]}
}

func f4Scale(a [4]float32, s float32) [4]float32 {
	return [4]float32{a[0] * s, a[1] * s, a[2] * s, a[3] * s}
}

func f4Neg(a [4]float32) [4]float32 { return [4]float32{-a[0], -a[1], -a[2], -a[3]} }

func f4Abs(a [4]float32) [4]float32 {
	return [4]float32{
		float32(math.Abs(float64(a[0]))), float32(math.Abs(float64(a[1]))),
		float32(math.Abs(float64(a[2]))), float32(math.Abs(float64(a[3]))),
	}
}

func f4MaxV(a, b [4]float32) [4]float32 {
	return [4]float32{max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])}
}

func f4MinLane(a [4]float32) float32 { return min(min(a[0], a[1]), min(a[2], a[3])) }
func f4MaxLane(a [4]float32) float32 { return max(max(a[0], a[1]), max(a[2], a[3])) }

func f4LT(a, b [4]float32) [4]bool {
	return [4]bool{a[0] < b[0], a[1] < b[1], a[2] < b[2], a[3] < b[3]}
}

func f4GT(a, b [4]float32) [4]bool {
	return [4]bool{a[0] > b[0], a[1] > b[1], a[2] > b[2], a[3] > b[3]}
}

func f4GE(a, b [4]float32) [4]bool {
	return [4]bool{a[0] >= b[0], a[1] >= b[1], a[2] >= b[2], a[3] >= b[3]}
}

func f4NE(a, b [4]float32) [4]bool {
	return [4]bool{a[0] != b[0], a[1] != b[1], a[2] != b[2], a[3] != b[3]}
}

func f4Shuffle(v [4]float32, i0, i1, i2, i3 int) [4]float32 {
	return [4]float32{v[i0], v[i1], v[i2], v[i3]}
}

func maskShuffle(v [4]bool, i0, i1, i2, i3 int) [4]bool {
	return [4]bool{v[i0], v[i1], v[i2], v[i3]}
}

// correctBadEdges replaces zero-length 'bad' edge vectors with the reversed opposite edge vector. e3 may be nil if only
// 2D edges need correction.
func correctBadEdges(bad [4]bool, e1, e2, e3 *[4]float32) {
	if maskAny(bad) {
		// Want opposite edges, L B T R -> R T B L but with flipped sign to preserve winding.
		*e1 = f4IfThenElse(bad, f4Neg(nextDiag(*e1)), *e1)
		*e2 = f4IfThenElse(bad, f4Neg(nextDiag(*e2)), *e2)
		if e3 != nil {
			*e3 = f4IfThenElse(bad, f4Neg(nextDiag(*e3)), *e3)
		}
	}
}

// correctBadCoords replaces 'bad' coordinates by rotating CCW to get the next point. c3 may be nil for 2D points.
func correctBadCoords(bad [4]bool, c1, c2, c3 *[4]float32) {
	if maskAny(bad) {
		*c1 = f4IfThenElse(bad, nextCCW(*c1), *c1)
		*c2 = f4IfThenElse(bad, nextCCW(*c2), *c2)
		if c3 != nil {
			*c3 = f4IfThenElse(bad, nextCCW(*c3), *c3)
		}
	}
}

// interpolateLocal interpolates local coordinates for a cropped vertex. Since the local quad may not be an axis-aligned
// rect, this uses the opposite vertex for each interpolation and calculates new ws in addition to xs, ys.
func interpolateLocal(alpha float32, v0, v1, v2, v3 int, lx, ly, lw *[4]float32) {
	beta := 1 - alpha
	lx[v0] = alpha*lx[v0] + beta*lx[v2]
	ly[v0] = alpha*ly[v0] + beta*ly[v2]
	lw[v0] = alpha*lw[v0] + beta*lw[v2]

	lx[v1] = alpha*lx[v1] + beta*lx[v3]
	ly[v1] = alpha*ly[v1] + beta*ly[v3]
	lw[v1] = alpha*lw[v1] + beta*lw[v3]
}

// cropRectEdge crops the edge from v0 to v1 based on clipDevRect. v2 is opposite of v0, v3 is opposite of v1. It does
// not modify coordinates if there's no intersection along the edge.
func cropRectEdge(clipDevRect geom.Rect, v0, v1, v2, v3 int, x, y, lx, ly, lw *[4]float32) bool {
	if geom.ScalarNearlyEqual(x[v0], x[v1]) {
		// A vertical edge.
		if x[v0] < clipDevRect.Left && x[v2] >= clipDevRect.Left {
			// Overlapping with the left edge of clipDevRect.
			if lx != nil {
				alpha := (x[v2] - clipDevRect.Left) / (x[v2] - x[v0])
				interpolateLocal(alpha, v0, v1, v2, v3, lx, ly, lw)
			}
			x[v0] = clipDevRect.Left
			x[v1] = clipDevRect.Left
			return true
		} else if x[v0] > clipDevRect.Right && x[v2] <= clipDevRect.Right {
			// Overlapping with the right edge of clipDevRect.
			if lx != nil {
				alpha := (clipDevRect.Right - x[v2]) / (x[v0] - x[v2])
				interpolateLocal(alpha, v0, v1, v2, v3, lx, ly, lw)
			}
			x[v0] = clipDevRect.Right
			x[v1] = clipDevRect.Right
			return true
		}
	} else {
		// A horizontal edge.
		if y[v0] < clipDevRect.Top && y[v2] >= clipDevRect.Top {
			if lx != nil {
				alpha := (y[v2] - clipDevRect.Top) / (y[v2] - y[v0])
				interpolateLocal(alpha, v0, v1, v2, v3, lx, ly, lw)
			}
			y[v0] = clipDevRect.Top
			y[v1] = clipDevRect.Top
			return true
		} else if y[v0] > clipDevRect.Bottom && y[v2] <= clipDevRect.Bottom {
			if lx != nil {
				alpha := (clipDevRect.Bottom - y[v2]) / (y[v0] - y[v2])
				interpolateLocal(alpha, v0, v1, v2, v3, lx, ly, lw)
			}
			y[v0] = clipDevRect.Bottom
			y[v1] = clipDevRect.Bottom
			return true
		}
	}
	// No overlap so don't crop it.
	return false
}

// cropRect updates x and y to intersect with clipDevRect. lx, ly, and lw may be nil to skip local calculations. Returns
// a bit mask of the edges that were clipped.
func cropRect(clipDevRect geom.Rect, x, y, lx, ly, lw *[4]float32) gpu.QuadAAFlags {
	clipEdgeFlags := gpu.QuadAAFlagsNone
	// The quad's left edge may not align with the rect's notion of left due to 90 degree rotations or mirrors, so
	// process the logical edges of the quad against all 4 sides of clipDevRect.
	if cropRectEdge(clipDevRect, 0, 1, 2, 3, x, y, lx, ly, lw) {
		clipEdgeFlags |= gpu.QuadAAFlagLeft
	}
	if cropRectEdge(clipDevRect, 0, 2, 1, 3, x, y, lx, ly, lw) {
		clipEdgeFlags |= gpu.QuadAAFlagTop
	}
	if cropRectEdge(clipDevRect, 2, 3, 0, 1, x, y, lx, ly, lw) {
		clipEdgeFlags |= gpu.QuadAAFlagRight
	}
	if cropRectEdge(clipDevRect, 1, 3, 0, 2, x, y, lx, ly, lw) {
		clipEdgeFlags |= gpu.QuadAAFlagBottom
	}
	return clipEdgeFlags
}

// cropSimpleRect assumes both the device coordinates and optional local coordinates geometrically match the TL, BL, TR,
// BR vertex ordering.
func cropSimpleRect(clipDevRect geom.Rect, x, y, lx, ly *[4]float32) gpu.QuadAAFlags {
	clipEdgeFlags := gpu.QuadAAFlagsNone

	// Update local coordinates proportionately to how much the device rect edge was clipped.
	var dx, dy float32
	if lx != nil {
		dx = (lx[2] - lx[0]) / (x[2] - x[0])
	}
	if ly != nil {
		dy = (ly[1] - ly[0]) / (y[1] - y[0])
	}
	if clipDevRect.Left > x[0] {
		if lx != nil {
			lx[0] += (clipDevRect.Left - x[0]) * dx
			lx[1] = lx[0]
		}
		x[0] = clipDevRect.Left
		x[1] = clipDevRect.Left
		clipEdgeFlags |= gpu.QuadAAFlagLeft
	}
	if clipDevRect.Top > y[0] {
		if ly != nil {
			ly[0] += (clipDevRect.Top - y[0]) * dy
			ly[2] = ly[0]
		}
		y[0] = clipDevRect.Top
		y[2] = clipDevRect.Top
		clipEdgeFlags |= gpu.QuadAAFlagTop
	}
	if clipDevRect.Right < x[2] {
		if lx != nil {
			lx[2] -= (x[2] - clipDevRect.Right) * dx
			lx[3] = lx[2]
		}
		x[2] = clipDevRect.Right
		x[3] = clipDevRect.Right
		clipEdgeFlags |= gpu.QuadAAFlagRight
	}
	if clipDevRect.Bottom < y[1] {
		if ly != nil {
			ly[1] -= (y[1] - clipDevRect.Bottom) * dy
			ly[3] = ly[1]
		}
		y[1] = clipDevRect.Bottom
		y[3] = clipDevRect.Bottom
		clipEdgeFlags |= gpu.QuadAAFlagBottom
	}
	return clipEdgeFlags
}

// isSimpleRect reports whether q is an axis-aligned rect with the standard TL/BL/TR/BR vertex order; consistent with
// Quad.AsRect but requires fewer operations.
func isSimpleRect(q *Quad) bool {
	if q.QuadType() != QuadTypeAxisAligned {
		return false
	}
	// v0 at the geometric top-left is unique, so we only need to compare x[0] < x[2] for left and y[0] < y[1] for top,
	// but add a little padding to protect against numerical precision on R90 and R270 transforms tricking this check.
	return (q.X(0)+geom.ScalarNearlyZeroTol) < q.X(2) && (q.Y(0)+geom.ScalarNearlyZeroTol) < q.Y(1)
}

// barycentricCoords calculates barycentric coordinates for each point in (testX, testY) in the triangle formed by
// (x0,y0) - (x1,y1) - (x2,y2).
func barycentricCoords(x0, y0, x1, y1, x2, y2 float32, testX, testY [4]float32, u, v, w *[4]float32) bool {
	// The 32-bit calculations can have catastrophic cancellation if the device-space coordinates are really big;
	// preserve some precision by shrinking the coordinate space if the bounds are large.
	const coordLimit = 1e7 // Big but somewhat arbitrary, fixes crbug:10141204.
	scaleX := max(max(x0, x1), x2) - min(min(x0, x1), x2)
	scaleY := max(max(y0, y1), y2) - min(min(y0, y1), y2)
	if scaleX > coordLimit {
		scaleX = coordLimit / scaleX
		x0 *= scaleX
		x1 *= scaleX
		x2 *= scaleX
	} else {
		scaleX = 1
	}
	if scaleY > coordLimit {
		scaleY = coordLimit / scaleY
		y0 *= scaleY
		y1 *= scaleY
		y2 *= scaleY
	} else {
		scaleY = 1
	}

	v0x := x2 - x0
	v0y := y2 - y0
	v1x := x1 - x0
	v1y := y1 - y0

	dot00 := v0x*v0x + v0y*v0y
	dot01 := v0x*v1x + v0y*v1y
	dot11 := v1x*v1x + v1y*v1y

	// Not yet 1/d; first check d != 0 with a healthy tolerance (worst case is we end up not cropping something we could
	// have).
	invDenom := dot00*dot11 - dot01*dot01
	const emptyTriTolerance = 1.0 / (1 << 5)
	if geom.ScalarNearlyZeroTolerance(invDenom, emptyTriTolerance) {
		// Degenerate/empty triangle: the UVW calculations would report every test point inside.
		return false
	}
	invDenom = 1 / invDenom

	for i := range 4 {
		v2x := scaleX*testX[i] - x0
		v2y := scaleY*testY[i] - y0
		dot02 := v0x*v2x + v0y*v2y
		dot12 := v1x*v2x + v1y*v2y
		// These are relative to the vertices, so there's no need to undo the scale factor.
		u[i] = (dot11*dot02 - dot01*dot12) * invDenom
		v[i] = (dot00*dot12 - dot01*dot02) * invDenom
		w[i] = 1 - u[i] - v[i]
	}
	return true
}

func insideTriangle(u, v, w [4]float32) [4]bool {
	var r [4]bool
	for i := range 4 {
		r[i] = u[i] >= 0 && u[i] <= 1 && v[i] >= 0 && v[i] <= 1 && w[i] >= 0 && w[i] <= 1
	}
	return r
}

// projectedBounds computes the device-space bounding rect of the quad after perspective division, clipping any vertices
// behind the w=0 plane to the plane first.
func (q *Quad) projectedBounds() geom.Rect {
	xs := q.X4f()
	ys := q.Y4f()
	ws := q.W4f()
	clipW := f4LT(ws, f4Splat(quadW0PlaneDistance))
	if maskAny(clipW) {
		x2d := f4Div(xs, ws)
		y2d := f4Div(ys, ws)
		// Bounds of just the projected points in front of w = epsilon.
		posInf := float32(math.Inf(1))
		negInf := float32(math.Inf(-1))
		frontBounds := geom.Rect{
			Left:   f4MinLane(f4IfThenElse(clipW, f4Splat(posInf), x2d)),
			Top:    f4MinLane(f4IfThenElse(clipW, f4Splat(posInf), y2d)),
			Right:  f4MaxLane(f4IfThenElse(clipW, f4Splat(negInf), x2d)),
			Bottom: f4MaxLane(f4IfThenElse(clipW, f4Splat(negInf), y2d)),
		}
		// Calculate clipped coordinates by following CCW edges, only keeping points where the w actually changes sign
		// between the vertices.
		t := f4Div(f4Sub(f4Splat(quadW0PlaneDistance), ws), f4Sub(nextCCW(ws), ws))
		for i := range 4 {
			x2d[i] = (t[i]*nextCCW(xs)[i] + (1-t[i])*xs[i]) / quadW0PlaneDistance
			y2d[i] = (t[i]*nextCCW(ys)[i] + (1-t[i])*ys[i]) / quadW0PlaneDistance
		}
		// True if (w < e) xor (ccw(w) < e), i.e. crosses the w = epsilon plane.
		clipW = maskXor(clipW, f4LT(nextCCW(ws), f4Splat(quadW0PlaneDistance)))
		return geom.Rect{
			Left:   f4MinLane(f4IfThenElse(clipW, x2d, f4Splat(frontBounds.Left))),
			Top:    f4MinLane(f4IfThenElse(clipW, y2d, f4Splat(frontBounds.Top))),
			Right:  f4MaxLane(f4IfThenElse(clipW, x2d, f4Splat(frontBounds.Right))),
			Bottom: f4MaxLane(f4IfThenElse(clipW, y2d, f4Splat(frontBounds.Bottom))),
		}
	}
	// Nothing is behind the viewer, so the projection is straightforward and valid.
	x2d := f4Div(xs, ws)
	y2d := f4Div(ys, ws)
	return geom.Rect{
		Left: f4MinLane(x2d), Top: f4MinLane(y2d), Right: f4MaxLane(x2d),
		Bottom: f4MaxLane(y2d),
	}
}

//////////////////////////////////////////////////////////////////////////////

// ResolveAAType resolves disagreements between the overall requested AA type and the per-edge quad AA flags.
func ResolveAAType(requestedAAType gpu.AAType, requestedEdgeFlags gpu.QuadAAFlags, quad *Quad) (gpu.AAType, gpu.QuadAAFlags) {
	outAAType := requestedAAType
	outEdgeFlags := requestedEdgeFlags

	switch requestedAAType {
	case gpu.AATypeCoverage:
		// When the aa type is coverage, disable AA if the edge configuration doesn't actually need it.
		if requestedEdgeFlags == gpu.QuadAAFlagsNone {
			outAAType = gpu.AATypeNone
		} else if quad.QuadType() == QuadTypeAxisAligned &&
			!quad.AAHasEffectOnRect(requestedEdgeFlags) {
			// For coverage AA, if the quad is a rect and AA-enabled edges line up with pixel boundaries, then overall
			// AA and per-edge AA can be completely disabled.
			outAAType = gpu.AATypeNone
			outEdgeFlags = gpu.QuadAAFlagsNone
		}
	case gpu.AATypeNone:
		outEdgeFlags = gpu.QuadAAFlagsNone
	case gpu.AATypeMSAA:
		outEdgeFlags = gpu.QuadAAFlagsAll
	}
	return outAAType, outEdgeFlags
}

// ClipToW0 clips the device vertices of 'quad' to be in front of the W = 0 plane (within epsilon), updating local
// coordinates to match. Returns the number of clipped quads that need to be drawn: 0 if entirely behind the plane, 1 if
// unclipped or 2-3 vertices were clipped, or 2 if one vertex was clipped (a pentagon spanned by 'quad' and
// 'extraVertices').
func ClipToW0(quad, extraVertices *DrawQuad) int {
	if quad.Device.QuadType() < QuadTypePerspective {
		// W implicitly 1s for each vertex, so nothing to do but draw unmodified 'quad'.
		return 1
	}

	validW := f4GE(quad.Device.W4f(), f4Splat(quadW0PlaneDistance))
	if maskAll(validW) {
		// Nothing to clip, can proceed normally drawing just 'quad'.
		return 1
	} else if !maskAny(validW) {
		// Everything is clipped, so draw nothing.
		return 0
	}

	// The clipped local coordinates will most likely not remain rectilinear.
	localType := quad.Local.QuadType()
	if localType < QuadTypeGeneral {
		localType = QuadTypeGeneral
	}

	// If we got here, there are 1, 2, or 3 points behind the w = 0 plane. If 2 or 3 points are clipped we can define a
	// new quad that covers the clipped shape directly. If there's 1 clipped out, the new geometry is a pentagon.
	var v tessVertices
	v.reset(&quad.Device, &quad.Local)

	clipCount := 0
	for i := range 4 {
		if !validW[i] {
			clipCount++
		}
	}

	t := f4Div(f4Sub(f4Splat(quadW0PlaneDistance), v.w), f4Sub(nextCCW(v.w), v.w))

	var clip tessVertices
	for i := range 4 {
		omt := 1 - t[i]
		clip.x[i] = t[i]*nextCCW(v.x)[i] + omt*v.x[i]
		clip.y[i] = t[i]*nextCCW(v.y)[i] + omt*v.y[i]
		clip.u[i] = t[i]*nextCCW(v.u)[i] + omt*v.u[i]
		clip.v[i] = t[i]*nextCCW(v.v)[i] + omt*v.v[i]
		clip.r[i] = t[i]*nextCCW(v.r)[i] + omt*v.r[i]
	}
	clip.w = f4Splat(quadW0PlaneDistance)

	ccwValid := f4GE(nextCCW(v.w), f4Splat(quadW0PlaneDistance))
	cwValid := f4GE(nextCW(v.w), f4Splat(quadW0PlaneDistance))

	if clipCount != 1 {
		// Simplest case: replace behind-w0 points with their clipped points by following the CCW or CW edge, depending
		// on which edge crosses the plane. (When 3 vertices are clipped, this results in a degenerate quad with one
		// replicated vertex, which is preferable to inserting a 3rd vertex on the w = 0 intersection line because two
		// parallel edges make inset/outset math unstable for large quads.)
		neither := maskAnd(maskNot(ccwValid), maskNot(cwValid))
		blend := func(orig, c [4]float32) [4]float32 {
			return f4IfThenElse(validW, orig,
				f4IfThenElse(neither, nextCCW(c), f4IfThenElse(ccwValid, c, nextCW(c))))
		}
		v.x = blend(v.x, clip.x)
		v.y = blend(v.y, clip.y)
		v.w = f4IfThenElse(validW, v.w, clip.w)
		v.u = blend(v.u, clip.u)
		v.v = blend(v.v, clip.v)
		v.r = blend(v.r, clip.r)

		v.asQuads(&quad.Device, QuadTypePerspective, &quad.Local, localType)
		return 1
	}

	// The clipped geometry is a pentagon, represented as two quads connected by a new non-AA edge. Use the midpoint
	// along one of the unclipped edges as a split vertex.
	var mid tessVertices
	mid.x = f4Scale(f4Add(v.x, nextCCW(v.x)), 0.5)
	mid.y = f4Scale(f4Add(v.y, nextCCW(v.y)), 0.5)
	mid.w = f4Scale(f4Add(v.w, nextCCW(v.w)), 0.5)
	mid.u = f4Scale(f4Add(v.u, nextCCW(v.u)), 0.5)
	mid.v = f4Scale(f4Add(v.v, nextCCW(v.v)), 0.5)
	mid.r = f4Scale(f4Add(v.r, nextCCW(v.r)), 0.5)

	// A quad formed by the 2 clipped points, the inserted mid point, and the good vertex that is CCW rotated from the
	// clipped vertex.
	var v2 tessVertices
	v2.uvrCount = v.uvrCount
	invalidOrNoCCW := maskOr(maskNot(validW), maskNot(ccwValid))
	blend2 := func(c, m, orig [4]float32) [4]float32 {
		return f4IfThenElse(invalidOrNoCCW, c, f4IfThenElse(cwValid, nextCW(m), orig))
	}
	v2.x = blend2(clip.x, mid.x, v.x)
	v2.y = blend2(clip.y, mid.y, v.y)
	v2.w = blend2(clip.w, mid.w, v.w)
	v2.u = blend2(clip.u, mid.u, v.u)
	v2.v = blend2(clip.v, mid.v, v.v)
	v2.r = blend2(clip.r, mid.r, v.r)
	// The non-AA edge for this quad is the opposite of the clipped vertex's edge.
	var v2EdgeFlag gpu.QuadAAFlags
	switch {
	case !validW[0]:
		v2EdgeFlag = gpu.QuadAAFlagRight // left clipped -> right
	case !validW[1]:
		v2EdgeFlag = gpu.QuadAAFlagTop // bottom clipped -> top
	case !validW[2]:
		v2EdgeFlag = gpu.QuadAAFlagBottom // top clipped -> bottom
	default:
		v2EdgeFlag = gpu.QuadAAFlagLeft // right clipped -> left
	}
	extraVertices.EdgeFlags = quad.EdgeFlags &^ v2EdgeFlag

	// A quad formed by the remaining two good vertices, one clipped point, and the inserted mid point.
	notValid := maskNot(validW)
	notCW := maskNot(cwValid)
	v.x = f4IfThenElse(notValid, nextCW(clip.x), f4IfThenElse(notCW, mid.x, v.x))
	v.y = f4IfThenElse(notValid, nextCW(clip.y), f4IfThenElse(notCW, mid.y, v.y))
	v.w = f4IfThenElse(notValid, clip.w, f4IfThenElse(notCW, mid.w, v.w))
	v.u = f4IfThenElse(notValid, nextCW(clip.u), f4IfThenElse(notCW, mid.u, v.u))
	v.v = f4IfThenElse(notValid, nextCW(clip.v), f4IfThenElse(notCW, mid.v, v.v))
	v.r = f4IfThenElse(notValid, clip.r, f4IfThenElse(notCW, mid.r, v.r))
	// The non-AA edge for this quad is the clipped vertex's edge.
	var v1EdgeFlag gpu.QuadAAFlags
	switch {
	case !validW[0]:
		v1EdgeFlag = gpu.QuadAAFlagLeft
	case !validW[1]:
		v1EdgeFlag = gpu.QuadAAFlagBottom
	case !validW[2]:
		v1EdgeFlag = gpu.QuadAAFlagTop
	default:
		v1EdgeFlag = gpu.QuadAAFlagRight
	}

	v.asQuads(&quad.Device, QuadTypePerspective, &quad.Local, localType)
	quad.EdgeFlags &^= v1EdgeFlag

	v2.asQuads(&extraVertices.Device, QuadTypePerspective, &extraVertices.Local, localType)
	// Caller must draw both 'quad' and 'extraVertices' to cover the clipped geometry.
	return 2
}

// CropToRect crops the quad to the provided device-space axis-aligned rectangle. Returns true if the intersection of
// the (projected) quad and cropRect results in a quadrilateral; otherwise the quad may have been updated to be a
// smaller quad of the same type with a visually identical intersection. The quad's coordinates must be finite.
func CropToRect(crop geom.Rect, cropAA gpu.AA, quad *DrawQuad, computeLocal bool) bool {
	if !quad.Device.IsFinite() {
		panic("CropToRect requires finite quad coordinates")
	}

	if quad.Device.QuadType() == QuadTypeAxisAligned {
		// cropRect and cropSimpleRect keep the rectangles as rectangles, so the intersection of the crop and quad can
		// be calculated exactly. Some care must be taken if the quad is axis-aligned but does not satisfy asRect() due
		// to flips, etc.
		var clippedEdges gpu.QuadAAFlags
		if computeLocal {
			if isSimpleRect(&quad.Device) && isSimpleRect(&quad.Local) {
				clippedEdges = cropSimpleRect(crop, quad.Device.Xs(), quad.Device.Ys(),
					quad.Local.Xs(), quad.Local.Ys())
			} else {
				clippedEdges = cropRect(crop, quad.Device.Xs(), quad.Device.Ys(),
					quad.Local.Xs(), quad.Local.Ys(), quad.Local.Ws())
			}
		} else {
			if isSimpleRect(&quad.Device) {
				clippedEdges = cropSimpleRect(crop, quad.Device.Xs(), quad.Device.Ys(),
					nil, nil)
			} else {
				clippedEdges = cropRect(crop, quad.Device.Xs(), quad.Device.Ys(),
					nil, nil, nil)
			}
		}

		// Apply the clipped edge updates to the original edge flags.
		if cropAA == gpu.AAYes {
			// Turn on all edges that were clipped.
			quad.EdgeFlags |= clippedEdges
		} else {
			// Turn off all edges that were clipped.
			quad.EdgeFlags &^= clippedEdges
		}
		return true
	}

	if computeLocal || quad.Device.QuadType() == QuadTypePerspective {
		// TODO: calculate cropped local coordinates when not axis-aligned; the perspective lane is disabled for
		// numerical robustness (crbug.com/1204347).
		return false
	}

	devX := quad.Device.X4f()
	devY := quad.Device.Y4f()

	clipX := [4]float32{crop.Left, crop.Left, crop.Right, crop.Right}
	clipY := [4]float32{crop.Top, crop.Bottom, crop.Top, crop.Bottom}

	// Calculate barycentric coordinates for the 4 rect corners in the 2 triangles that the quad is tessellated into
	// when drawn.
	var u1, v1, w1, u2, v2, w2 [4]float32
	if !barycentricCoords(devX[0], devY[0], devX[1], devY[1], devX[2], devY[2], clipX, clipY,
		&u1, &v1, &w1) ||
		!barycentricCoords(devX[1], devY[1], devX[3], devY[3], devX[2], devY[2], clipX, clipY,
			&u2, &v2, &w2) {
		// Bad triangles, skip cropping.
		return false
	}

	// The crop rect is completely inside this quad if each corner is in at least one of the two triangles.
	inTri1 := insideTriangle(u1, v1, w1)
	inTri2 := insideTriangle(u2, v2, w2)
	if maskAll(maskOr(inTri1, inTri2)) {
		// We can crop to exactly the clip rect: the draw discards any perspective and the type becomes axis-aligned.
		*quad.Device.Xs() = clipX
		*quad.Device.Ys() = clipY
		quad.Device.SetQuadType(QuadTypeAxisAligned)

		// Update the edge flags to match the clip setting since all 4 edges have been clipped.
		if cropAA == gpu.AAYes {
			quad.EdgeFlags = gpu.QuadAAFlagsAll
		} else {
			quad.EdgeFlags = gpu.QuadAAFlagsNone
		}
		return true
	}

	// TODO: use TessellationHelper's inset/outset math to move edges to the closest clip corner they are outside of.
	return false
}

// WillUseHairline reports whether the quad will be drawn using the hairline (subpixel-width) lane given the requested
// AA type and edge flags.
func WillUseHairline(quad *Quad, aaType gpu.AAType, edgeFlags gpu.QuadAAFlags) bool {
	if aaType != gpu.AATypeCoverage || edgeFlags != gpu.QuadAAFlagsAll {
		// Non-aa or msaa don't do any outsetting so they will not be hairlined; mixed edge flags could be hairlined in
		// theory, but applying hairline bloat would extend beyond the original tiled shape.
		return false
	}

	if quad.QuadType() == QuadTypeAxisAligned {
		// Fast path that avoids computing edge properties via TessellationHelper. Taking the absolute value of the
		// diagonals always produces the minimum of width or height given that this is axis-aligned, regardless of
		// mirror or 90/180-degree rotations.
		d := min(float32(math.Abs(float64(quad.X(3)-quad.X(0)))),
			float32(math.Abs(float64(quad.Y(3)-quad.Y(0)))))
		return d < 1
	}
	var helper TessellationHelper
	helper.Reset(quad, nil)
	return helper.IsSubpixel()
}

// OutsetQuad moves quad's edges outward by edgeDistances (ordered L, B, T, R) in place.
func OutsetQuad(edgeDistances [4]float32, quad *Quad) {
	var outsetter TessellationHelper
	outsetter.Reset(quad, nil)
	outsetter.Outset(edgeDistances, quad, nil)
}

//////////////////////////////////////////////////////////////////////////////
// TessellationHelper

// tessEdgeVectors holds cached calculations pertaining to the edge vectors of the input quad, projected into 2D device
// coordinates.
type tessEdgeVectors struct {
	// Projected corners (x/w and y/w).
	x2d, y2d [4]float32
	// Normalized edge vectors of the device space quad, ordered L, B, T, R (i.e. next_ccw(x) - x).
	dx, dy [4]float32
	// Reciprocal of the edge lengths.
	invLengths [4]float32
	// Theta represents the angle formed by the two edges connected at each corner.
	cosTheta    [4]float32
	invSinTheta [4]float32
}

func (e *tessEdgeVectors) reset(xs, ys, ws [4]float32, quadType QuadType) {
	// Calculate all projected edge vector values for this quad.
	if quadType == QuadTypePerspective {
		for i := range 4 {
			iw := 1 / ws[i]
			e.x2d[i] = xs[i] * iw
			e.y2d[i] = ys[i] * iw
		}
	} else {
		e.x2d = xs
		e.y2d = ys
	}

	e.dx = f4Sub(nextCCW(e.x2d), e.x2d)
	e.dy = f4Sub(nextCCW(e.y2d), e.y2d)
	for i := range 4 {
		e.invLengths[i] = 1 / float32(math.Sqrt(float64(e.dx[i]*e.dx[i]+e.dy[i]*e.dy[i])))
	}

	// Normalize edge vectors.
	e.dx = f4Mul(e.dx, e.invLengths)
	e.dy = f4Mul(e.dy, e.invLengths)

	// Calculate angles between vectors.
	if quadType <= QuadTypeRectilinear {
		e.cosTheta = f4Splat(0)
		e.invSinTheta = f4Splat(1)
	} else {
		e.cosTheta = f4Add(f4Mul(e.dx, nextCW(e.dx)), f4Mul(e.dy, nextCW(e.dy)))
		// NOTE: if cosTheta is close to 1, the inset/outset math avoids the fast paths that rely on invSinTheta, since
		// it approaches infinity.
		for i := range 4 {
			e.invSinTheta[i] = 1 / float32(math.Sqrt(float64(1-e.cosTheta[i]*e.cosTheta[i])))
		}
	}
}

// tessEdgeEquations holds the quad's edge line equations a*x + b*y + c = 0; a positive distance is inside the quad;
// ordered LBTR.
type tessEdgeEquations struct {
	a, b, c [4]float32
}

func (eq *tessEdgeEquations) reset(ev *tessEdgeVectors) {
	dx := ev.dx
	dy := ev.dy
	// Correct for bad edges by copying adjacent edge information into the bad component.
	correctBadEdges(f4GE(ev.invLengths, f4Splat(quadInvDistTolerance)), &dx, &dy, nil)

	var c [4]float32
	for i := range 4 {
		c[i] = dx[i]*ev.y2d[i] - dy[i]*ev.x2d[i]
	}
	// Make sure the normals point into the shape.
	cwX := nextCW(ev.x2d)
	cwY := nextCW(ev.y2d)
	flip := false
	for i := range 4 {
		if dy[i]*cwX[i]+(-dx[i]*cwY[i]+c[i]) < -quadDistTolerance {
			flip = true
			break
		}
	}
	if flip {
		eq.a = f4Neg(dy)
		eq.b = dx
		eq.c = f4Neg(c)
	} else {
		eq.a = dy
		eq.b = f4Neg(dx)
		eq.c = c
	}
}

// estimateCoverage estimates the per-vertex fractional pixel coverage of the inset points x2d,y2d relative to the
// original quad's edges.
func (eq *tessEdgeEquations) estimateCoverage(x2d, y2d [4]float32) [4]float32 {
	// Calculate the distance of the 4 inset points to the 4 edges.
	var d0, d1, d2, d3 [4]float32
	for i := range 4 {
		d0[i] = eq.a[0]*x2d[i] + (eq.b[0]*y2d[i] + eq.c[0])
		d1[i] = eq.a[1]*x2d[i] + (eq.b[1]*y2d[i] + eq.c[1])
		d2[i] = eq.a[2]*x2d[i] + (eq.b[2]*y2d[i] + eq.c[2])
		d3[i] = eq.a[3]*x2d[i] + (eq.b[3]*y2d[i] + eq.c[3])
	}

	// For each point, pretend that there's a rectangle that touches e0 and e3 on the horizontal axis and e1/e2 on the
	// vertical axis; pin each dimension to [0, 1] and approximate the coverage at each point as their product.
	var cov [4]float32
	for i := range 4 {
		w := max(0, min(1, d0[i]+d3[i]))
		h := max(0, min(1, d1[i]+d2[i]))
		cov[i] = w * h
	}
	return cov
}

// isSubpixel reports whether all 4 vertex-to-opposite-edge distances are less than 1px, in which case the inset
// geometry would be a point or line and quad rendering switches to hairline mode.
func (eq *tessEdgeEquations) isSubpixel(x2d, y2d [4]float32) bool {
	a1 := f4Shuffle(eq.a, 1, 2, 1, 2)
	b1 := f4Shuffle(eq.b, 1, 2, 1, 2)
	c1 := f4Shuffle(eq.c, 1, 2, 1, 2)
	a2 := f4Shuffle(eq.a, 3, 3, 0, 0)
	b2 := f4Shuffle(eq.b, 3, 3, 0, 0)
	c2 := f4Shuffle(eq.c, 3, 3, 0, 0)
	for i := range 4 {
		d := min(x2d[i]*a1[i]+y2d[i]*b1[i]+c1[i], x2d[i]*a2[i]+y2d[i]*b2[i]+c2[i])
		if d >= 1 {
			return false
		}
	}
	return true
}

// computeDegenerateQuad outsets or insets x2d/y2d in place for very small interiors, near-parallel edges, or very short
// edges. Returns the number of effective vertices in the degenerate quad.
func (eq *tessEdgeEquations) computeDegenerateQuad(signedEdgeDistances [4]float32, x2d, y2d *[4]float32, aaMask *[4]bool) int {
	// If the original points form a line in the 2D projection then give up on antialiasing.
	for i := range 4 {
		allNear := true
		for j := range 4 {
			d := x2d[j]*eq.a[i] + y2d[j]*eq.b[i] + eq.c[i]
			if float32(math.Abs(float64(d))) >= quadDistTolerance {
				allNear = false
				break
			}
		}
		if allNear {
			*aaMask = [4]bool{}
			return 4
		}
	}

	*aaMask = f4NE(signedEdgeDistances, f4Splat(0))

	// Move the edge by the signed edge adjustment.
	oc := f4Add(eq.c, signedEdgeDistances)

	// There are 6 points that we care about to determine the final shape of the polygon: the intersections between
	// (e0,e2), (e1,e0), (e2,e3), (e3,e1) (the 4 corners), and (e1,e2), (e0,e3) (the intersections of opposite edges).
	var denom, px, py [4]float32
	cwA := nextCW(eq.a)
	cwB := nextCW(eq.b)
	cwOC := nextCW(oc)
	for i := range 4 {
		denom[i] = eq.a[i]*cwB[i] - eq.b[i]*cwA[i]
		px[i] = (eq.b[i]*cwOC[i] - oc[i]*cwB[i]) / denom[i]
		py[i] = (oc[i]*cwA[i] - eq.a[i]*cwOC[i]) / denom[i]
	}
	correctBadCoords(f4LT(f4Abs(denom), f4Splat(quadTolerance)), &px, &py, nil)

	// Signed distances from these 4 corners to the other two edges that did not define the intersection: p(0) vs e3,e1;
	// p(1) vs e3,e2; p(2) vs e0,e1; p(3) vs e0,e2.
	a1 := f4Shuffle(eq.a, 3, 3, 0, 0)
	b1 := f4Shuffle(eq.b, 3, 3, 0, 0)
	oc1 := f4Shuffle(oc, 3, 3, 0, 0)
	a2 := f4Shuffle(eq.a, 1, 2, 1, 2)
	b2 := f4Shuffle(eq.b, 1, 2, 1, 2)
	oc2 := f4Shuffle(oc, 1, 2, 1, 2)
	var dists1, dists2 [4]float32
	for i := range 4 {
		dists1[i] = px[i]*a1[i] + py[i]*b1[i] + oc1[i]
		dists2[i] = px[i]*a2[i] + py[i]*b2[i] + oc2[i]
	}

	// If all the distances are >= 0, the 4 corners form a valid quadrilateral. If any point is on the wrong side of
	// both edges, the interior has collapsed to a point. If all four points are only on the wrong side of 1 edge, one
	// edge has crossed over another and we use a line. Otherwise use a triangle.
	d1v0 := f4LT(dists1, f4Splat(quadDistTolerance))
	d2v0 := f4LT(dists2, f4Splat(quadDistTolerance))
	d1And2 := maskAnd(d1v0, d2v0)
	d1Or2 := maskOr(d1v0, d2v0)

	switch {
	case !maskAny(d1Or2):
		// Not degenerate: use all 4 corners as-is with full coverage.
		*x2d = px
		*y2d = py
		return 4
	case maskAny(d1And2):
		// A point failed against two edges, so reduce the shape to a single point (the center of the original quad to
		// ensure it is contained in the intended geometry). It cannot cover a pixel, so update the coverage.
		cx := 0.25 * (x2d[0] + x2d[1] + x2d[2] + x2d[3])
		cy := 0.25 * (y2d[0] + y2d[1] + y2d[2] + y2d[3])
		*x2d = f4Splat(cx)
		*y2d = f4Splat(cy)
		anyAA := maskAny(*aaMask)
		*aaMask = [4]bool{anyAA, anyAA, anyAA, anyAA}
		return 1
	case maskAll(d1Or2):
		// Degenerates to a line. Compare p[2] and p[3] to edge 0: if they are on the wrong side, edges 0 and 3 crossed,
		// otherwise edges 1 and 2 crossed.
		if dists1[2] < quadDistTolerance && dists1[3] < quadDistTolerance {
			// Edges 0 and 3 crossed: the line is from the averages of (p0,p2) and (p1,p3).
			*x2d = f4Scale(f4Add(f4Shuffle(px, 0, 1, 0, 1), f4Shuffle(px, 2, 3, 2, 3)), 0.5)
			*y2d = f4Scale(f4Add(f4Shuffle(py, 0, 1, 0, 1), f4Shuffle(py, 2, 3, 2, 3)), 0.5)
			// Both 2D points on the shared edge moved, so moveTo() must be able to move both 3D points along the shared
			// edge: ensure both have AA.
			*aaMask = maskOr(*aaMask, [4]bool{true, false, false, true})
		} else {
			// Edges 1 and 2 crossed: the line is from the averages of (p0,p1) and (p2,p3).
			*x2d = f4Scale(f4Add(f4Shuffle(px, 0, 0, 2, 2), f4Shuffle(px, 1, 1, 3, 3)), 0.5)
			*y2d = f4Scale(f4Add(f4Shuffle(py, 0, 0, 2, 2), f4Shuffle(py, 1, 1, 3, 3)), 0.5)
			*aaMask = maskOr(*aaMask, [4]bool{false, true, true, false})
		}
		return 2
	default:
		// This turns into a triangle. Replace corners as needed with the intersections between (e0,e3) and (e1,e2).
		// Because of the tolerance we can have cases where the intersection lies far outside the quad; in that case
		// replace the point with the average of itself and the point calculated using the edge equation it failed.
		eDenom := [2]float32{
			eq.a[0]*eq.b[3] - eq.b[0]*eq.a[3],
			eq.a[1]*eq.b[2] - eq.b[1]*eq.a[2],
		}
		ex := [2]float32{
			(eq.b[0]*oc[3] - oc[0]*eq.b[3]) / eDenom[0],
			(eq.b[1]*oc[2] - oc[1]*eq.b[2]) / eDenom[1],
		}
		ey := [2]float32{
			(oc[0]*eq.a[3] - eq.a[0]*oc[3]) / eDenom[0],
			(oc[1]*eq.a[2] - eq.a[1]*oc[2]) / eDenom[1],
		}

		avgX := f4Scale(f4Add(f4Shuffle(px, 0, 1, 0, 2), f4Shuffle(px, 2, 3, 1, 3)), 0.5)
		avgY := f4Scale(f4Add(f4Shuffle(py, 0, 1, 0, 2), f4Shuffle(py, 2, 3, 1, 3)), 0.5)
		for i := range 4 {
			// It can't be the case that d1v0[i] and d2v0[i] are both true (that was the point case above).
			switch {
			case dists1[i] < -quadDistTolerance &&
				float32(math.Abs(float64(eDenom[0]))) > quadTolerance:
				px[i] = ex[0]
				py[i] = ey[0]
			case d1v0[i]:
				px[i] = avgX[i%2]
				py[i] = avgY[i%2]
			case dists2[i] < -quadDistTolerance &&
				float32(math.Abs(float64(eDenom[1]))) > quadTolerance:
				px[i] = ex[1]
				py[i] = ey[1]
			case d2v0[i]:
				px[i] = avgX[i/2+2]
				py[i] = avgY[i/2+2]
			}
		}

		// If we replace a vertex with an intersection then it will not fall along the edges that intersect at the
		// original vertex, so provide an additional degree of freedom by turning AA on for both edges if either edge is
		// AA at each point.
		*aaMask = maskOr(*aaMask, maskOr(maskAnd(d1Or2, nextCWMask(*aaMask)),
			maskAnd(nextCCWMask(d1Or2), nextCCWMask(*aaMask))))
		*x2d = px
		*y2d = py
		return 3
	}
}

// tessOutsetRequest describes a requested inset/outset by edge distance, along with whether the non-degenerate fast
// paths for inset and outset are usable.
type tessOutsetRequest struct {
	// Positive edge distances to move each edge of the quad (the shortest perpendicular distance between the original
	// edge and the inset or outset edge).
	edgeDistances [4]float32
	// True if the new corners cannot be calculated by simply adding scaled edge vectors.
	insetDegenerate  bool
	outsetDegenerate bool
}

func (o *tessOutsetRequest) reset(ev *tessEdgeVectors, quadType QuadType, edgeDistances [4]float32) {
	o.edgeDistances = edgeDistances

	// Based on the edge distances, determine if it's acceptable to use invSinTheta to calculate the inset or outset
	// geometry.
	switch {
	case quadType <= QuadTypeRectilinear:
		// Since it's rectangular, the width (edge[1] or edge[2]) collapses if subtracting (dist[0] + dist[3]) makes the
		// new width negative; same for height (edge[0] or edge[3]) and (dist[1] + dist[2]). Outsetting will never be
		// degenerate here.
		o.outsetDegenerate = false
		widthChange := edgeDistances[0] + edgeDistances[3]
		heightChange := edgeDistances[1] + edgeDistances[2]
		// 1/len > 1/(edge sum) implies len - edge sum < 0.
		o.insetDegenerate = (widthChange > 0 && ev.invLengths[1] > 1/widthChange) ||
			(heightChange > 0 && ev.invLengths[0] > 1/heightChange)
	case maskAny(f4GE(ev.invLengths, f4Splat(quadInvDistTolerance))):
		// An edge is effectively length 0, so we're dealing with a triangle, which must always go through the
		// degenerate code path.
		o.outsetDegenerate = true
		o.insetDegenerate = true
	default:
		// If possible, the corners will move +/- edgeDistances * 1/sin(theta). The entire request is degenerate if
		// 1/sin(theta) -> infinity (or cos(theta) -> 1).
		if maskAny(f4GE(f4Abs(ev.cosTheta), f4Splat(0.9))) {
			o.outsetDegenerate = true
			o.insetDegenerate = true
		} else {
			// With an edge-centric view, an edge's length changes by edgeDistance * cos(pi - theta) / sin(theta) for
			// each of its corners (the second corner uses the ccw theta value); it also changes when its adjacent edges
			// move, by edgeDistance / sin(theta) (or cos(theta) for the other edge).
			//
			// cos(pi - theta) = -cos(theta)
			var halfTanTheta, edgeAdjust [4]float32
			for i := range 4 {
				halfTanTheta[i] = -ev.cosTheta[i] * ev.invSinTheta[i]
			}
			ccwHTT := nextCCW(halfTanTheta)
			ccwDist := nextCCW(edgeDistances)
			ccwIST := nextCCW(ev.invSinTheta)
			cwDist := nextCW(edgeDistances)
			for i := range 4 {
				edgeAdjust[i] = edgeDistances[i]*(halfTanTheta[i]+ccwHTT[i]) +
					ccwDist[i]*ccwIST[i] + cwDist[i]*ev.invSinTheta[i]
			}

			// If either outsetting (plus edgeAdjust) or insetting (minus edgeAdjust) make the edge lengths negative,
			// then it's degenerate.
			var threshold [4]float32
			for i := range 4 {
				threshold[i] = 0.1 - 1/ev.invLengths[i]
			}
			o.outsetDegenerate = maskAny(f4LT(edgeAdjust, threshold))
			o.insetDegenerate = maskAny(f4GT(edgeAdjust, f4Neg(threshold)))
		}
	}
}

// tessVertices holds a quad's device (and optional local) coordinates during tessellation.
type tessVertices struct {
	// X, Y, and W coordinates in device space. If not perspective, w should be set to 1.
	x, y, w [4]float32
	// U, V, and R coordinates representing the local quad; ignored depending on uvrCount (0, 2, or 3).
	u, v, r  [4]float32
	uvrCount int
}

func (t *tessVertices) reset(deviceQuad, localQuad *Quad) {
	t.x = deviceQuad.X4f()
	t.y = deviceQuad.Y4f()
	t.w = deviceQuad.W4f()
	if localQuad != nil {
		t.u = localQuad.X4f()
		t.v = localQuad.Y4f()
		t.r = localQuad.W4f()
		if localQuad.HasPerspective() {
			t.uvrCount = 3
		} else {
			t.uvrCount = 2
		}
	} else {
		t.uvrCount = 0
	}
}

func (t *tessVertices) asQuads(deviceOut *Quad, deviceType QuadType, localOut *Quad, localType QuadType) {
	*deviceOut.Xs() = t.x
	*deviceOut.Ys() = t.y
	if deviceType == QuadTypePerspective {
		*deviceOut.Ws() = t.w
	}
	deviceOut.SetQuadType(deviceType) // Sets ws == 1 when the device type is not perspective.

	if t.uvrCount > 0 {
		*localOut.Xs() = t.u
		*localOut.Ys() = t.v
		if t.uvrCount == 3 {
			*localOut.Ws() = t.r
		}
		localOut.SetQuadType(localType)
	}
}

// moveAlong updates the device and optional local coordinates by moving the corners along their edge vectors such that
// the new edges have moved signedEdgeDistances from their original lines. Should only be called when the edge vectors'
// invSinTheta data is numerically sound.
func (t *tessVertices) moveAlong(ev *tessEdgeVectors, signedEdgeDistances [4]float32) {
	// When the projected device quad is not degenerate, the vertex corners can move cornerOutsetLen along their edge
	// and their cw-rotated edge. The vertex's edge points inwards and the cw-rotated edge points outwards, hence the
	// minus-sign. The edge distances are rotated compared to the corner outsets and (dx, dy), since if the edge is "on"
	// both its corners need to be moved along their other edge vectors.
	signedOutsets := f4Mul(f4Neg(ev.invSinTheta), nextCW(signedEdgeDistances))
	signedOutsetsCW := f4Mul(ev.invSinTheta, signedEdgeDistances)

	cwDX := nextCW(ev.dx)
	cwDY := nextCW(ev.dy)
	for i := range 4 {
		t.x[i] += signedOutsetsCW[i]*cwDX[i] + signedOutsets[i]*ev.dx[i]
		t.y[i] += signedOutsetsCW[i]*cwDY[i] + signedOutsets[i]*ev.dy[i]
	}
	if t.uvrCount > 0 {
		// Extend the texture coords by the same proportion as the positions.
		signedOutsets = f4Mul(signedOutsets, ev.invLengths)
		signedOutsetsCW = f4Mul(signedOutsetsCW, nextCW(ev.invLengths))
		du := f4Sub(nextCCW(t.u), t.u)
		dv := f4Sub(nextCCW(t.v), t.v)
		cwDU := nextCW(du)
		cwDV := nextCW(dv)
		for i := range 4 {
			t.u[i] += signedOutsetsCW[i]*cwDU[i] + signedOutsets[i]*du[i]
			t.v[i] += signedOutsetsCW[i]*cwDV[i] + signedOutsets[i]*dv[i]
		}
		if t.uvrCount == 3 {
			dr := f4Sub(nextCCW(t.r), t.r)
			cwDR := nextCW(dr)
			for i := range 4 {
				t.r[i] += signedOutsetsCW[i]*cwDR[i] + signedOutsets[i]*dr[i]
			}
		}
	}
}

// moveTo updates the device coordinates by deriving (x,y,w) that project to (x2d, y2d), with optional local coordinates
// updated to match the new vertices. 'mask' is used to ensure that only unmasked unprojected edge vectors are used when
// computing device and local coordinates.
func (t *tessVertices) moveTo(x2d, y2d [4]float32, mask [4]bool) {
	// Left to right, in device space, for each point.
	e1x := f4Sub(f4Shuffle(t.x, 2, 3, 2, 3), f4Shuffle(t.x, 0, 1, 0, 1))
	e1y := f4Sub(f4Shuffle(t.y, 2, 3, 2, 3), f4Shuffle(t.y, 0, 1, 0, 1))
	e1w := f4Sub(f4Shuffle(t.w, 2, 3, 2, 3), f4Shuffle(t.w, 0, 1, 0, 1))
	var e1Bad [4]bool
	for i := range 4 {
		e1Bad[i] = e1x[i]*e1x[i]+e1y[i]*e1y[i] < quadDist2Tolerance
	}
	correctBadEdges(e1Bad, &e1x, &e1y, &e1w)

	// Top to bottom, in device space, for each point.
	e2x := f4Sub(f4Shuffle(t.x, 1, 1, 3, 3), f4Shuffle(t.x, 0, 0, 2, 2))
	e2y := f4Sub(f4Shuffle(t.y, 1, 1, 3, 3), f4Shuffle(t.y, 0, 0, 2, 2))
	e2w := f4Sub(f4Shuffle(t.w, 1, 1, 3, 3), f4Shuffle(t.w, 0, 0, 2, 2))
	var e2Bad [4]bool
	for i := range 4 {
		e2Bad[i] = e2x[i]*e2x[i]+e2y[i]*e2y[i] < quadDist2Tolerance
	}
	correctBadEdges(e2Bad, &e2x, &e2y, &e2w)

	// Can only move along e1 and e2 to reach the new 2D point, so we have x2d = (x + a*e1x + b*e2x) / (w + a*e1w +
	// b*e2w) and y2d = (y + a*e1y + b*e2y) / (w + a*e1w + b*e2w), for some a, b. This can be rewritten as a*c1x + b*c2x
	// + c3x = 0; a*c1y + b*c2y + c3y = 0, where:
	var c1x, c1y, c2x, c2y, c3x, c3y [4]float32
	for i := range 4 {
		c1x[i] = e1w[i]*x2d[i] - e1x[i]
		c1y[i] = e1w[i]*y2d[i] - e1y[i]
		c2x[i] = e2w[i]*x2d[i] - e2x[i]
		c2y[i] = e2w[i]*y2d[i] - e2y[i]
		c3x[i] = t.w[i]*x2d[i] - t.x[i]
		c3y[i] = t.w[i]*y2d[i] - t.y[i]
	}

	// Solve for a and b.
	var a, b, denom [4]float32
	if maskAll(mask) {
		// When every edge is outset/inset, each corner can use both edge vectors.
		for i := range 4 {
			denom[i] = c1x[i]*c2y[i] - c2x[i]*c1y[i]
			a[i] = (c2x[i]*c3y[i] - c3x[i]*c2y[i]) / denom[i]
			b[i] = (c3x[i]*c1y[i] - c1x[i]*c3y[i]) / denom[i]
		}
	} else {
		// Force a or b to be 0 if that edge cannot be used due to non-AA.
		aMask := maskShuffle(mask, 0, 0, 3, 3)
		bMask := maskShuffle(mask, 2, 1, 2, 1)

		// When aMask[i]&bMask[i], a[i], b[i], denom[i] match the all-mask case. When aMask[i]&!bMask[i], b[i] = 0, a[i]
		// = -c3x/c1x or -c3y/c1y, using the better denom. When !aMask[i]&bMask[i], a[i] = 0, b[i] = -c3x/c2x or
		// -c3y/c2y. When !aMask[i]&!bMask[i], a[i] = 0 and b[i] = 0.
		for i := range 4 {
			useC1x := float32(math.Abs(float64(c1x[i]))) > float32(math.Abs(float64(c1y[i])))
			useC2x := float32(math.Abs(float64(c2x[i]))) > float32(math.Abs(float64(c2y[i])))

			switch {
			case aMask[i] && bMask[i]:
				denom[i] = c1x[i]*c2y[i] - c2x[i]*c1y[i]
				a[i] = (c2x[i]*c3y[i] - c3x[i]*c2y[i]) / denom[i]
				b[i] = (c3x[i]*c1y[i] - c1x[i]*c3y[i]) / denom[i]
			case aMask[i]: // A & !B
				if useC1x {
					denom[i] = c1x[i]
					a[i] = -c3x[i] / denom[i]
				} else {
					denom[i] = c1y[i]
					a[i] = -c3y[i] / denom[i]
				}
				b[i] = 0
			case bMask[i]: // !A & B
				if useC2x {
					denom[i] = c2x[i]
					b[i] = -c3x[i] / denom[i]
				} else {
					denom[i] = c2y[i]
					b[i] = -c3y[i] / denom[i]
				}
				a[i] = 0
			default: // !A & !B
				denom[i] = 1
				a[i] = 0
				b[i] = 0
			}
		}
	}

	for i := range 4 {
		t.x[i] += a[i]*e1x[i] + b[i]*e2x[i]
		t.y[i] += a[i]*e1y[i] + b[i]*e2y[i]
		t.w[i] += a[i]*e1w[i] + b[i]*e2w[i]
	}

	// If w has gone negative, flip the point to the other side of w=0. This only happens if the edge was approaching a
	// vanishing point and it was physically impossible to outset 1/2px in screen space without going behind the viewer
	// and being mirrored. Scaling by -1 preserves the computed screen space position but moves the 3D point off of the
	// original quad.
	if maskAny(f4LT(t.w, f4Splat(0))) {
		for i := range 4 {
			if t.w[i] < 0 {
				t.x[i] = -t.x[i]
				t.y[i] = -t.y[i]
				t.w[i] = -t.w[i]
			}
		}
	}

	badDenom := f4LT(f4Abs(denom), f4Splat(quadTolerance))
	correctBadCoords(badDenom, &t.x, &t.y, &t.w)

	if t.uvrCount > 0 {
		// Calculate R here so it can be corrected with U and V in case it's needed later.
		e1u := f4Sub(f4Shuffle(t.u, 2, 3, 2, 3), f4Shuffle(t.u, 0, 1, 0, 1))
		e1v := f4Sub(f4Shuffle(t.v, 2, 3, 2, 3), f4Shuffle(t.v, 0, 1, 0, 1))
		e1r := f4Sub(f4Shuffle(t.r, 2, 3, 2, 3), f4Shuffle(t.r, 0, 1, 0, 1))
		correctBadEdges(e1Bad, &e1u, &e1v, &e1r)

		e2u := f4Sub(f4Shuffle(t.u, 1, 1, 3, 3), f4Shuffle(t.u, 0, 0, 2, 2))
		e2v := f4Sub(f4Shuffle(t.v, 1, 1, 3, 3), f4Shuffle(t.v, 0, 0, 2, 2))
		e2r := f4Sub(f4Shuffle(t.r, 1, 1, 3, 3), f4Shuffle(t.r, 0, 0, 2, 2))
		correctBadEdges(e2Bad, &e2u, &e2v, &e2r)

		for i := range 4 {
			t.u[i] += a[i]*e1u[i] + b[i]*e2u[i]
			t.v[i] += a[i]*e1v[i] + b[i]*e2v[i]
		}
		if t.uvrCount == 3 {
			for i := range 4 {
				t.r[i] += a[i]*e1r[i] + b[i]*e2r[i]
			}
			correctBadCoords(badDenom, &t.u, &t.v, &t.r)
		} else {
			correctBadCoords(badDenom, &t.u, &t.v, nil)
		}
	}
}

// TessellationHelper computes inset/outset geometry and edge equations for per-edge coverage AA rendering of a quad.
type TessellationHelper struct {
	original    tessVertices
	edgeVectors tessEdgeVectors
	deviceType  QuadType
	localType   QuadType

	// Lazily computed as needed.
	outsetRequest tessOutsetRequest
	edgeEquations tessEdgeEquations

	verticesValid      bool
	outsetRequestValid bool
	edgeEquationsValid bool
}

// Reset sets the original device and (optional) local coordinates that are inset or outset by the requested edge
// distances. This assumes all device coordinates have been clipped to W > 0.
func (h *TessellationHelper) Reset(deviceQuad, localQuad *Quad) {
	h.deviceType = deviceQuad.QuadType()
	if localQuad != nil {
		h.localType = localQuad.QuadType()
	} else {
		h.localType = QuadTypeAxisAligned
	}

	h.outsetRequestValid = false
	h.edgeEquationsValid = false

	h.original.reset(deviceQuad, localQuad)
	h.edgeVectors.reset(h.original.x, h.original.y, h.original.w, h.deviceType)

	h.verticesValid = true
}

// Inset calculates a new quadrilateral with edges parallel to the original but moved inwards by edgeDistances (ordered
// L, B, T, R). Returns the per-vertex coverage (1.0 unless the inset geometry collapsed).
func (h *TessellationHelper) Inset(edgeDistances [4]float32, deviceInset, localInset *Quad) [4]float32 {
	if !h.verticesValid {
		panic("Reset must be called before Inset")
	}
	inset := h.original
	request := h.getOutsetRequest(edgeDistances)
	vertexCount := 4
	if request.insetDegenerate {
		vertexCount = h.adjustDegenerateVertices(f4Neg(request.edgeDistances), &inset)
	} else {
		h.adjustVertices(f4Neg(request.edgeDistances), &inset)
	}

	inset.asQuads(deviceInset, h.deviceType, localInset, h.localType)
	if vertexCount < 3 {
		// The interior has less than a full pixel's area, so estimate reduced coverage using the distance of the
		// inset's projected corners to the original edges.
		return h.getEdgeEquations().estimateCoverage(f4Div(inset.x, inset.w),
			f4Div(inset.y, inset.w))
	}
	return f4Splat(1)
}

// Outset calculates a new quadrilateral with edges parallel to the original but moved outwards by edgeDistances
// (ordered L, B, T, R).
func (h *TessellationHelper) Outset(edgeDistances [4]float32, deviceOutset, localOutset *Quad) {
	if !h.verticesValid {
		panic("Reset must be called before Outset")
	}
	outset := h.original
	request := h.getOutsetRequest(edgeDistances)
	if request.outsetDegenerate {
		h.adjustDegenerateVertices(request.edgeDistances, &outset)
	} else {
		h.adjustVertices(request.edgeDistances, &outset)
	}
	outset.asQuads(deviceOutset, h.deviceType, localOutset, h.localType)
}

// GetEdgeEquations returns the edge equation coefficients of the original device-space quad (ax + by + c = 0, positive
// distance inside, ordered LBTR).
func (h *TessellationHelper) GetEdgeEquations() (a, b, c [4]float32) {
	if !h.verticesValid {
		panic("Reset must be called before GetEdgeEquations")
	}
	eq := h.getEdgeEquations()
	return eq.a, eq.b, eq.c
}

// GetEdgeLengths returns the original quad's edge lengths (ordered LBTR).
func (h *TessellationHelper) GetEdgeLengths() [4]float32 {
	if !h.verticesValid {
		panic("Reset must be called before GetEdgeLengths")
	}
	var lengths [4]float32
	for i := range 4 {
		lengths[i] = 1 / h.edgeVectors.invLengths[i]
	}
	return lengths
}

// IsSubpixel reports whether the original device-space quad has vertices closer than 1px to its opposing edges.
func (h *TessellationHelper) IsSubpixel() bool {
	if !h.verticesValid {
		panic("Reset must be called before IsSubpixel")
	}
	if h.deviceType <= QuadTypeRectilinear {
		// Check the edge lengths: if the shortest is less than 1px it's degenerate, which is the same as the max
		// 1/length being greater than 1px.
		return maskAny(f4GT(h.edgeVectors.invLengths, f4Splat(1)))
	}
	// Compute edge equations and then the distance from each vertex to the opposite edges.
	return h.getEdgeEquations().isSubpixel(h.edgeVectors.x2d, h.edgeVectors.y2d)
}

func (h *TessellationHelper) getOutsetRequest(edgeDistances [4]float32) *tessOutsetRequest {
	// Much of the code assumes that we start from positive distances and apply it unmodified to create an outset;
	// knowing that it's outset simplifies degeneracy checking.
	for i := range 4 {
		if edgeDistances[i] < 0 {
			panic("edge distances must be non-negative")
		}
	}
	// Rebuild the outset request if invalid or if the edge distances have changed.
	if !h.outsetRequestValid || edgeDistances != h.outsetRequest.edgeDistances {
		h.outsetRequest.reset(&h.edgeVectors, h.deviceType, edgeDistances)
		h.outsetRequestValid = true
	}
	return &h.outsetRequest
}

func (h *TessellationHelper) getEdgeEquations() *tessEdgeEquations {
	if !h.edgeEquationsValid {
		h.edgeEquations.reset(&h.edgeVectors)
		h.edgeEquationsValid = true
	}
	return &h.edgeEquations
}

func (h *TessellationHelper) adjustVertices(signedEdgeDistances [4]float32, vertices *tessVertices) {
	if h.deviceType < QuadTypePerspective {
		// For non-perspective, non-degenerate quads, moveAlong is correct and most efficient.
		vertices.moveAlong(&h.edgeVectors, signedEdgeDistances)
	} else {
		// For perspective, non-degenerate quads, use moveAlong for the projected points and then reconstruct Ws with
		// moveTo.
		projected := tessVertices{x: h.edgeVectors.x2d, y: h.edgeVectors.y2d, w: f4Splat(1)}
		projected.moveAlong(&h.edgeVectors, signedEdgeDistances)
		vertices.moveTo(projected.x, projected.y, f4NE(signedEdgeDistances, f4Splat(0)))
	}
}

func (h *TessellationHelper) adjustDegenerateVertices(signedEdgeDistances [4]float32, vertices *tessVertices) int {
	if h.deviceType <= QuadTypeRectilinear {
		// For rectilinear, degenerate quads, can use moveAlong if the edge distances are adjusted to not cross over
		// each other (the only way rectilinear can degenerate is insets).
		var halfLengths [4]float32
		cwInvLengths := nextCW(h.edgeVectors.invLengths)
		for i := range 4 {
			halfLengths[i] = -0.5 / cwInvLengths[i] // Negate to inset.
		}
		crossedEdges := f4GT(halfLengths, signedEdgeDistances)
		safeInsets := f4IfThenElse(crossedEdges, halfLengths, signedEdgeDistances)
		vertices.moveAlong(&h.edgeVectors, safeInsets)

		// A degenerate rectilinear quad is either a point (both width and height crossed) or a line.
		if maskAll(crossedEdges) {
			return 1
		}
		return 2
	}
	// A degenerate non-rectangular shape must go through the slowest path (which automatically handles perspective).
	x2d := h.edgeVectors.x2d
	y2d := h.edgeVectors.y2d
	var aaMask [4]bool
	vertexCount := h.getEdgeEquations().computeDegenerateQuad(signedEdgeDistances, &x2d, &y2d,
		&aaMask)
	vertices.moveTo(x2d, y2d, aaMask)
	return vertexCount
}
