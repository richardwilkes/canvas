// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The tCurve abstraction and its tQuad/tConic/tCubic wrappers: the polymorphic curve the tSect binary-subdivision
// curve/curve solver operates on. Allocation is plain Go allocation (the GC manages these CPU-side objects), so
// makeCurve() takes no arena argument.

package pathops

// Quad point/intersection counts.
const (
	quadPointCount       = 3
	quadPointLast        = quadPointCount - 1
	quadMaxIntersections = 4
)

// Conic point/intersection counts.
const (
	conicPointCount       = 3
	conicPointLast        = conicPointCount - 1
	conicMaxIntersections = 4
)

// Cubic point/intersection counts.
const (
	cubicPointCount       = 4
	cubicPointLast        = cubicPointCount - 1
	cubicMaxIntersections = 9
)

// tCurve is the abstract curve the tSect solver subdivides. Every member either delegates to the wrapped
// double-precision curve or participates in the curve-vs-curve double dispatch (hullIntersects / intersectRay).
type tCurve interface {
	// pointAt returns the nth control point.
	pointAt(n int) dPoint
	// setPoint overwrites the nth control point.
	setPoint(n int, pt dPoint)
	collapsed() bool
	controlsInside() bool
	dxdyAtT(t float64) dVector
	// hullIntersectsQuad reports whether q's convex hull intersects this curve's.
	hullIntersectsQuad(q *dQuad, isLinear *bool) bool
	// hullIntersectsConic reports whether c's convex hull intersects this curve's.
	hullIntersectsConic(c *dConic, isLinear *bool) bool
	// hullIntersectsCubic reports whether c's convex hull intersects this curve's.
	hullIntersectsCubic(c *dCubic, isLinear *bool) bool
	// hullIntersectsCurve is the curve-vs-curve double dispatch: each concrete curve calls the opposite curve's
	// matching hullIntersects<T> overload.
	hullIntersectsCurve(other tCurve, isLinear *bool) bool
	// intersectRay intersects this curve with line, recording the crossings in i.
	intersectRay(i *intersections, line dLine) int
	isConic() bool
	// makeCurve returns a fresh blank curve of the same concrete type.
	makeCurve() tCurve
	maxIntersections() int
	// otherPts returns the pointCount()-1 control points other than oddMan.
	otherPts(oddMan int) []dPoint
	pointCount() int
	pointLast() int
	ptAtT(t float64) dPoint
	// setBounds grows r to this curve's control-point-mapped extrema bounds.
	setBounds(r *dRect)
	// subDivide writes the sub-curve over [t1,t2] into dst (which is the same concrete type as the receiver).
	subDivide(t1, t2 float64, dst tCurve)
}

// intersectRayTCurve is the double dispatch into the concrete curve's ray intersection.
func (in *intersections) intersectRayTCurve(c tCurve, line dLine) int {
	return c.intersectRay(in, line)
}

// ---- tQuad ----

// tQuad is the tCurve wrapper around a double-precision quad.
type tQuad struct {
	quad dQuad
}

func (c *tQuad) pointAt(n int) dPoint      { return c.quad.pts[n] }
func (c *tQuad) setPoint(n int, pt dPoint) { c.quad.pts[n] = pt }
func (c *tQuad) collapsed() bool           { return c.quad.collapsed() }
func (c *tQuad) controlsInside() bool      { return c.quad.controlsInside() }
func (c *tQuad) dxdyAtT(t float64) dVector { return c.quad.dxdyAtT(t) }

// hullIntersectsQuad reports whether q's convex hull intersects this quad's hull.
func (c *tQuad) hullIntersectsQuad(q *dQuad, isLinear *bool) bool {
	result, linear := q.hullIntersects(c.quad)
	*isLinear = linear
	return result
}

// hullIntersectsConic reports whether cn's convex hull intersects this quad's hull.
func (c *tQuad) hullIntersectsConic(cn *dConic, isLinear *bool) bool {
	result, linear := cn.hullIntersectsQuad(c.quad)
	*isLinear = linear
	return result
}

// hullIntersectsCubic reports whether cu's convex hull intersects this quad's hull.
func (c *tQuad) hullIntersectsCubic(cu *dCubic, isLinear *bool) bool {
	result, linear := cu.hullIntersectsQuad(c.quad)
	*isLinear = linear
	return result
}

// hullIntersectsCurve dispatches to the opposite curve's hullIntersectsQuad overload.
func (c *tQuad) hullIntersectsCurve(other tCurve, isLinear *bool) bool {
	return other.hullIntersectsQuad(&c.quad, isLinear)
}

// intersectRay intersects the wrapped quad with line, recording the crossings in i.
func (c *tQuad) intersectRay(i *intersections, line dLine) int {
	return i.intersectRayQuad(c.quad, line)
}

func (c *tQuad) isConic() bool         { return false }
func (c *tQuad) makeCurve() tCurve     { return &tQuad{} }
func (c *tQuad) maxIntersections() int { return quadMaxIntersections }

// otherPts returns the two control points other than oddMan.
func (c *tQuad) otherPts(oddMan int) []dPoint {
	pts := c.quad.otherPts(oddMan)
	return pts[:]
}

func (c *tQuad) pointCount() int        { return quadPointCount }
func (c *tQuad) pointLast() int         { return quadPointLast }
func (c *tQuad) ptAtT(t float64) dPoint { return c.quad.ptAtT(t) }
func (c *tQuad) setBounds(r *dRect)     { r.setBoundsQuadFull(c.quad) }

// subDivide writes the sub-curve over [t1,t2] into dst's wrapped quad.
func (c *tQuad) subDivide(t1, t2 float64, dst tCurve) {
	dst.(*tQuad).quad = c.quad.subDivide(t1, t2)
}

// ---- tConic ----

// tConic is the tCurve wrapper around a double-precision conic.
type tConic struct {
	conic dConic
}

func (c *tConic) pointAt(n int) dPoint      { return c.conic.pts.pts[n] }
func (c *tConic) setPoint(n int, pt dPoint) { c.conic.pts.pts[n] = pt }
func (c *tConic) collapsed() bool           { return c.conic.collapsed() }
func (c *tConic) controlsInside() bool      { return c.conic.controlsInside() }
func (c *tConic) dxdyAtT(t float64) dVector { return c.conic.dxdyAtT(t) }

// hullIntersectsQuad reports whether q's convex hull intersects this conic's hull.
func (c *tConic) hullIntersectsQuad(q *dQuad, isLinear *bool) bool {
	result, linear := q.hullIntersectsConic(c.conic)
	*isLinear = linear
	return result
}

// hullIntersectsConic reports whether cn's convex hull intersects this conic's hull.
func (c *tConic) hullIntersectsConic(cn *dConic, isLinear *bool) bool {
	result, linear := cn.hullIntersectsConic(c.conic)
	*isLinear = linear
	return result
}

// hullIntersectsCubic reports whether cu's convex hull intersects this conic's hull.
func (c *tConic) hullIntersectsCubic(cu *dCubic, isLinear *bool) bool {
	result, linear := cu.hullIntersectsConic(c.conic)
	*isLinear = linear
	return result
}

// hullIntersectsCurve dispatches to the opposite curve's hullIntersectsConic overload.
func (c *tConic) hullIntersectsCurve(other tCurve, isLinear *bool) bool {
	return other.hullIntersectsConic(&c.conic, isLinear)
}

// intersectRay intersects the wrapped conic with line, recording the crossings in i.
func (c *tConic) intersectRay(i *intersections, line dLine) int {
	return i.intersectRayConic(c.conic, line)
}

func (c *tConic) isConic() bool         { return true }
func (c *tConic) makeCurve() tCurve     { return &tConic{} }
func (c *tConic) maxIntersections() int { return conicMaxIntersections }

func (c *tConic) otherPts(oddMan int) []dPoint {
	pts := c.conic.otherPts(oddMan)
	return pts[:]
}

func (c *tConic) pointCount() int        { return conicPointCount }
func (c *tConic) pointLast() int         { return conicPointLast }
func (c *tConic) ptAtT(t float64) dPoint { return c.conic.ptAtT(t) }
func (c *tConic) setBounds(r *dRect)     { r.setBoundsConicFull(c.conic) }

// subDivide writes the sub-curve over [t1,t2] into dst's wrapped conic.
func (c *tConic) subDivide(t1, t2 float64, dst tCurve) {
	dst.(*tConic).conic = c.conic.subDivide(t1, t2)
}

// ---- tCubic ----

// tCubic is the tCurve wrapper around a double-precision cubic.
type tCubic struct {
	cubic dCubic
}

func (c *tCubic) pointAt(n int) dPoint      { return c.cubic.pts[n] }
func (c *tCubic) setPoint(n int, pt dPoint) { c.cubic.pts[n] = pt }
func (c *tCubic) collapsed() bool           { return c.cubic.collapsed() }
func (c *tCubic) controlsInside() bool      { return c.cubic.controlsInside() }
func (c *tCubic) dxdyAtT(t float64) dVector { return c.cubic.dxdyAtT(t) }

// hullIntersectsQuad reports whether q's convex hull intersects this cubic's hull.
func (c *tCubic) hullIntersectsQuad(q *dQuad, isLinear *bool) bool {
	result, linear := q.hullIntersectsCubic(c.cubic)
	*isLinear = linear
	return result
}

// hullIntersectsConic reports whether cn's convex hull intersects this cubic's hull.
func (c *tCubic) hullIntersectsConic(cn *dConic, isLinear *bool) bool {
	result, linear := cn.hullIntersectsCubic(c.cubic)
	*isLinear = linear
	return result
}

// hullIntersectsCubic reports whether cu's convex hull intersects this cubic's hull.
func (c *tCubic) hullIntersectsCubic(cu *dCubic, isLinear *bool) bool {
	result, linear := cu.hullIntersectsCubic(c.cubic)
	*isLinear = linear
	return result
}

// hullIntersectsCurve dispatches to the opposite curve's hullIntersectsCubic overload.
func (c *tCubic) hullIntersectsCurve(other tCurve, isLinear *bool) bool {
	return other.hullIntersectsCubic(&c.cubic, isLinear)
}

// intersectRay intersects the wrapped cubic with line, recording the crossings in i.
func (c *tCubic) intersectRay(i *intersections, line dLine) int {
	return i.intersectRayCubic(c.cubic, line)
}

func (c *tCubic) isConic() bool         { return false }
func (c *tCubic) makeCurve() tCurve     { return &tCubic{} }
func (c *tCubic) maxIntersections() int { return cubicMaxIntersections }

func (c *tCubic) otherPts(oddMan int) []dPoint {
	pts := c.cubic.otherPts(oddMan)
	return pts[:]
}

func (c *tCubic) pointCount() int        { return cubicPointCount }
func (c *tCubic) pointLast() int         { return cubicPointLast }
func (c *tCubic) ptAtT(t float64) dPoint { return c.cubic.ptAtT(t) }
func (c *tCubic) setBounds(r *dRect)     { r.setBoundsCubicFull(c.cubic) }

// subDivide writes the sub-curve over [t1,t2] into dst's wrapped cubic.
func (c *tCubic) subDivide(t1, t2 float64, dst tCurve) {
	dst.(*tCubic).cubic = c.cubic.subDivide(t1, t2)
}
