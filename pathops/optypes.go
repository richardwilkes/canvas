// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The operation-time foundation types: opGlobalState (the state shared across one pathops run), pathOpsBounds (an
// axis-aligned box with non-empty degenerate-line bounds), the opVerbToPoints helper, and the verb-indexed curve
// evaluation dispatch. These back the op data model (opContour/opSegment/opSpan) behind the Op/Simplify engine.
//
// There is no arena allocator here (Go's GC manages the span/segment/contour graph) and no debug-only ID counters.
// Pathops computes in double precision throughout; float32 is used only where a value is stored/compared in the host
// geometry's native float precision.

package pathops

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// opPhase is the phase the pathops walker is in.
type opPhase int

const (
	opPhaseNoChange opPhase = iota
	opPhaseIntersecting
	opPhaseWalking
	opPhaseFixWinding
)

const maxWindingTries = 10 // cap on retries when fixing up inconsistent winding sums

// opGlobalState is the state shared by every contour/segment/span in one pathops run.
type opGlobalState struct {
	coincidence     *opCoincidence
	contourHead     *opContourHead
	nested          int
	allocatedOpSpan bool
	phase           opPhase
}

// newOpGlobalState creates the shared state for one pathops run; the walk starts in the intersecting phase.
func newOpGlobalState(head *opContourHead) *opGlobalState {
	return &opGlobalState{contourHead: head, phase: opPhaseIntersecting}
}

func (g *opGlobalState) setAllocatedOpSpan()   { g.allocatedOpSpan = true }
func (g *opGlobalState) resetAllocatedOpSpan() { g.allocatedOpSpan = false }

func (g *opGlobalState) bumpNested()  { g.nested++ }
func (g *opGlobalState) clearNested() { g.nested = 0 }

func (g *opGlobalState) setContourHead(head *opContourHead) { g.contourHead = head }

// setCoincidence records the coincidence tracker used by this run.
func (g *opGlobalState) setCoincidence(coincidence *opCoincidence) { g.coincidence = coincidence }

// setPhase updates the walker's phase; opPhaseNoChange leaves the phase untouched.
func (g *opGlobalState) setPhase(phase opPhase) {
	if phase == opPhaseNoChange {
		return
	}
	g.phase = phase
}

// opVerbToPoints returns the number of points a verb consumes past its start point (line 1, quad/conic 2, cubic 3).
func opVerbToPoints(v path.Verb) int {
	switch v {
	case path.VerbLine:
		return 1
	case path.VerbQuad, path.VerbConic:
		return 2
	case path.VerbCubic:
		return 3
	default:
		return 0
	}
}

// pathOpsBounds is an axis-aligned box that, unlike a plain rectangle, does not treat the bounds of a horizontal or
// vertical line as empty. Stored in float32 to match the host geometry's point precision.
type pathOpsBounds struct {
	left, top, right, bottom float32
}

// setLTRB sets the four edges directly.
func (b *pathOpsBounds) setLTRB(l, t, r, bottom float32) {
	b.left, b.top, b.right, b.bottom = l, t, r, bottom
}

// setBoundsPoints computes the min/max box of the points. Used for the line segment, whose two points are finite and
// distinct.
func (b *pathOpsBounds) setBoundsPoints(pts []geom.Point) {
	if len(pts) == 0 {
		*b = pathOpsBounds{}
		return
	}
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts[1:] {
		minX = min(minX, p.X)
		minY = min(minY, p.Y)
		maxX = max(maxX, p.X)
		maxY = max(maxY, p.Y)
	}
	b.left, b.top, b.right, b.bottom = minX, minY, maxX, maxY
}

// addBounds grows the box to include o, without any "line is empty" special case.
func (b *pathOpsBounds) addBounds(o pathOpsBounds) {
	if o.left < b.left {
		b.left = o.left
	}
	if o.top < b.top {
		b.top = o.top
	}
	if o.right > b.right {
		b.right = o.right
	}
	if o.bottom > b.bottom {
		b.bottom = o.bottom
	}
}

// setFromDRect stores a double-precision bounding box as float32.
func (b *pathOpsBounds) setFromDRect(r dRect) {
	b.setLTRB(float32(r.left), float32(r.top), float32(r.right), float32(r.bottom))
}

// curveDPointAtT returns the double-precision point on the verb's curve at t.
func curveDPointAtT(verb path.Verb, pts []geom.Point, weight float32, t float64) dPoint {
	switch verb {
	case path.VerbLine:
		var l dLine
		l.set([2]geom.Point{pts[0], pts[1]})
		return l.ptAtT(t)
	case path.VerbQuad:
		var q dQuad
		q.set([3]geom.Point{pts[0], pts[1], pts[2]})
		return q.ptAtT(t)
	case path.VerbConic:
		var c dConic
		c.set([3]geom.Point{pts[0], pts[1], pts[2]}, weight)
		return c.ptAtT(t)
	case path.VerbCubic:
		var c dCubic
		c.set([4]geom.Point{pts[0], pts[1], pts[2], pts[3]})
		return c.ptAtT(t)
	default:
		panic("pathops: curveDPointAtT: bad verb")
	}
}

// curvePointAtT returns the float32 point on the verb's curve at t.
func curvePointAtT(verb path.Verb, pts []geom.Point, weight float32, t float64) geom.Point {
	return curveDPointAtT(verb, pts, weight, t).asPoint()
}

// curveIntersectRay intersects the verb's curve with the ray, recording the crossings in i.
func curveIntersectRay(verb path.Verb, pts []geom.Point, weight float32, ray dLine, i *intersections) {
	switch verb {
	case path.VerbLine:
		var l dLine
		l.set([2]geom.Point{pts[0], pts[1]})
		i.intersectRayLine(l, ray)
	case path.VerbQuad:
		var q dQuad
		q.set([3]geom.Point{pts[0], pts[1], pts[2]})
		i.intersectRayQuad(q, ray)
	case path.VerbConic:
		var c dConic
		c.set([3]geom.Point{pts[0], pts[1], pts[2]}, weight)
		i.intersectRayConic(c, ray)
	case path.VerbCubic:
		var c dCubic
		c.set([4]geom.Point{pts[0], pts[1], pts[2], pts[3]})
		i.intersectRayCubic(c, ray)
	default:
		panic("pathops: curveIntersectRay: bad verb")
	}
}

// curveDDPointAtT returns the double-precision point on the operation-time dCurve at t (the dCurve-based sibling of
// curveDPointAtT).
func curveDDPointAtT(verb path.Verb, c dCurve, t float64) dPoint {
	switch verb {
	case path.VerbLine:
		return c.line().ptAtT(t)
	case path.VerbQuad:
		return c.quad().ptAtT(t)
	case path.VerbConic:
		return c.conic().ptAtT(t)
	case path.VerbCubic:
		return c.cubic().ptAtT(t)
	default:
		panic("pathops: curveDDPointAtT: bad verb")
	}
}

// curveDDSlopeAtT returns the double-precision tangent on the dCurve at t.
func curveDDSlopeAtT(verb path.Verb, c dCurve, t float64) dVector {
	switch verb {
	case path.VerbLine:
		l := c.line()
		return l.pts[1].sub(l.pts[0])
	case path.VerbQuad:
		return c.quad().dxdyAtT(t)
	case path.VerbConic:
		return c.conic().dxdyAtT(t)
	case path.VerbCubic:
		return c.cubic().dxdyAtT(t)
	default:
		panic("pathops: curveDDSlopeAtT: bad verb")
	}
}

// curveDIntersectRay intersects the dCurve with the ray, recording the crossings in i (the dCurve-based sibling of
// curveIntersectRay).
func curveDIntersectRay(verb path.Verb, c dCurve, ray dLine, i *intersections) {
	switch verb {
	case path.VerbLine:
		i.intersectRayLine(c.line(), ray)
	case path.VerbQuad:
		i.intersectRayQuad(c.quad(), ray)
	case path.VerbConic:
		i.intersectRayConic(c.conic(), ray)
	case path.VerbCubic:
		i.intersectRayCubic(c.cubic(), ray)
	default:
		panic("pathops: curveDIntersectRay: bad verb")
	}
}

// curveDSlopeAtT returns the double-precision tangent vector at t.
func curveDSlopeAtT(verb path.Verb, pts []geom.Point, weight float32, t float64) dVector {
	switch verb {
	case path.VerbLine:
		var l dLine
		l.set([2]geom.Point{pts[0], pts[1]})
		return l.pts[1].sub(l.pts[0])
	case path.VerbQuad:
		var q dQuad
		q.set([3]geom.Point{pts[0], pts[1], pts[2]})
		return q.dxdyAtT(t)
	case path.VerbConic:
		var c dConic
		c.set([3]geom.Point{pts[0], pts[1], pts[2]}, weight)
		return c.dxdyAtT(t)
	case path.VerbCubic:
		var c dCubic
		c.set([4]geom.Point{pts[0], pts[1], pts[2], pts[3]})
		return c.dxdyAtT(t)
	default:
		panic("pathops: curveDSlopeAtT: bad verb")
	}
}
