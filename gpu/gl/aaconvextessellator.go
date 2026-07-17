// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Flattens a convex path into a polygon and builds an antialiased triangulation by outsetting an outer ring (zero
// coverage) and insetting rings toward the interior (full coverage), sliding vertices along inward bisectors until they
// fuse. Consumed by AALinearizingConvexPathRenderer, which supports fill, stroke (miter/bevel), and stroke-and-fill
// styles.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// aaTessAntialiasingRadius is the device-space distance which we inset/outset points in order to create the soft
// antialiased edge.
const aaTessAntialiasingRadius = 0.5

// The tolerance for fusing vertices and eliminating colinear lines (in device space).
const (
	aaTessClose    = float32(1) / 16
	aaTessCloseSqd = aaTessClose * aaTessClose
)

// Tessellation tolerance values, in device space pixels.
const (
	aaTessQuadTolerance     = float32(0.2)
	aaTessCubicTolerance    = float32(0.2)
	aaTessQuadToleranceSqd  = aaTessQuadTolerance * aaTessQuadTolerance
	aaTessCubicToleranceSqd = aaTessCubicTolerance * aaTessCubicTolerance
	aaTessConicTolerance    = float32(0.25)
)

// aaTessRoundCapThreshold is the dot product below which we use a round cap between curve segments.
const aaTessRoundCapThreshold = float32(0.8)

// aaTessCurveConnectionThreshold is the dot product above which we consider two adjacent curves to be part of the
// "same" curve.
const aaTessCurveConnectionThreshold = float32(0.8)

// aaTessIntersect finds where the line through p0 in direction n0 crosses the line through p1 in direction n1,
// returning the parametric t along the first line (or ok=false if parallel).
func aaTessIntersect(p0, n0, p1, n1 geom.Point) (t float32, ok bool) {
	v := p1.Sub(p0)
	perpDot := n0.X*n1.Y - n0.Y*n1.X
	if geom.ScalarNearlyZero(perpDot) {
		return 0, false
	}
	t = (v.X*n1.Y - v.Y*n1.X) / perpDot
	return t, geom.IsFinite(t)
}

// aaTessPerpIntersect is a special case of aaTessIntersect where we have the vector perpendicular to the second line
// rather than the vector parallel to it.
func aaTessPerpIntersect(p0, n0, p1, perp geom.Point) (t float32, ok bool) {
	v := p1.Sub(p0)
	perpDot := n0.Dot(perp)
	if geom.ScalarNearlyZero(perpDot) {
		return 0, false
	}
	t = v.Dot(perp) / perpDot
	return t, geom.IsFinite(t)
}

// aaTessDuplicatePt reports whether p0 and p1 are close enough to be treated as the same point.
func aaTessDuplicatePt(p0, p1 geom.Point) bool {
	return p0.DistanceToSqd(p1) < aaTessCloseSqd
}

// aaTessPointsAreColinearAndBIsMiddle reports whether b lies close enough to the line through a and c, and between
// them, that it can be dropped as a near-collinear point.
func aaTessPointsAreColinearAndBIsMiddle(a, b, c geom.Point, accumError *float32) bool {
	// First check distance from b to the infinite line through a, c.
	aToC := c.Sub(a)
	n := geom.Point{X: aToC.Y, Y: -aToC.X}
	n.Normalize()

	distBToLineAC := geom.ScalarAbs(n.Dot(b) - n.Dot(a))
	if *accumError+distBToLineAC >= aaTessClose || aToC.Dot(b.Sub(a)) <= 0 ||
		aToC.Dot(c.Sub(b)) <= 0 {
		// Too far from the line or not between the line segment from a to c.
		return false
	}
	// Accumulate the distance from b to |ac| that goes "away" when this near-colinear point is removed to simplify the
	// path.
	*accumError += distBToLineAC
	return true
}

// aaTessCurveState tracks whether a given point is within a curve. A point is inside a curve only if it is an interior
// point within a quad, cubic, or conic, or if it is the endpoint of a quad, cubic, or conic with another curve meeting
// it at (more or less) the same angle.
type aaTessCurveState uint8

const (
	// aaTessCurveSharp: point is a sharp vertex.
	aaTessCurveSharp aaTessCurveState = iota
	// aaTessCurveIndeterminate: endpoint of a curve with the other side's curvature not yet determined.
	aaTessCurveIndeterminate
	// aaTessCurveCurve: point is in the interior of a curve.
	aaTessCurveCurve
)

// aaTessCandidatePoint is one candidate vertex for the next ring being built.
type aaTessCandidatePoint struct {
	pt             geom.Point
	originatingIdx int
	origEdgeID     int
	needsToBeNew   bool
}

// aaTessCandidateVerts holds the vertices for the next ring while they are being generated. Its main function is to
// de-dup the points.
type aaTessCandidateVerts struct {
	pts []aaTessCandidatePoint
}

func (c *aaTessCandidateVerts) setReserve(numPts int) {
	if cap(c.pts) < numPts {
		newPts := make([]aaTessCandidatePoint, len(c.pts), numPts)
		copy(newPts, c.pts)
		c.pts = newPts
	}
}

func (c *aaTessCandidateVerts) rewind() { c.pts = c.pts[:0] }

func (c *aaTessCandidateVerts) numPts() int { return len(c.pts) }

func (c *aaTessCandidateVerts) lastPoint() geom.Point { return c.pts[len(c.pts)-1].pt }

func (c *aaTessCandidateVerts) firstPoint() geom.Point { return c.pts[0].pt }

func (c *aaTessCandidateVerts) point(index int) geom.Point { return c.pts[index].pt }

func (c *aaTessCandidateVerts) originatingIdx(index int) int { return c.pts[index].originatingIdx }

func (c *aaTessCandidateVerts) origEdge(index int) int { return c.pts[index].origEdgeID }

func (c *aaTessCandidateVerts) needsToBeNew(index int) bool { return c.pts[index].needsToBeNew }

func (c *aaTessCandidateVerts) addNewPt(newPt geom.Point, originatingIdx, origEdge int, needsToBeNew bool) int {
	c.pts = append(c.pts, aaTessCandidatePoint{
		pt: newPt, originatingIdx: originatingIdx,
		origEdgeID: origEdge, needsToBeNew: needsToBeNew,
	})
	return len(c.pts) - 1
}

func (c *aaTessCandidateVerts) fuseWithPrior(origEdgeID int) int {
	last := &c.pts[len(c.pts)-1]
	last.origEdgeID = origEdgeID
	last.originatingIdx = -1
	last.needsToBeNew = true
	return len(c.pts) - 1
}

func (c *aaTessCandidateVerts) fuseWithNext() int {
	c.pts[0].originatingIdx = -1
	c.pts[0].needsToBeNew = true
	return 0
}

func (c *aaTessCandidateVerts) fuseWithBoth() int {
	if len(c.pts) > 1 {
		c.pts = c.pts[:len(c.pts)-1]
	}
	c.pts[0].originatingIdx = -1
	c.pts[0].needsToBeNew = true
	return 0
}

// aaTessRingPoint is one point of a ring, with its outward normal and inward bisector.
type aaTessRingPoint struct {
	norm       geom.Point
	bisector   geom.Point
	index      int
	origEdgeID int
}

// aaTessRing is a set of indices into the global pool that together define a single polygon inset.
type aaTessRing struct {
	pts []aaTessRingPoint
}

func (r *aaTessRing) setReserve(numPts int) {
	if cap(r.pts) < numPts {
		newPts := make([]aaTessRingPoint, len(r.pts), numPts)
		copy(newPts, r.pts)
		r.pts = newPts
	}
}

func (r *aaTessRing) rewind() { r.pts = r.pts[:0] }

func (r *aaTessRing) numPts() int { return len(r.pts) }

func (r *aaTessRing) addIdx(index, origEdgeID int) {
	r.pts = append(r.pts, aaTessRingPoint{index: index, origEdgeID: origEdgeID})
}

// makeOriginalRing upgrades this ring so that it can behave like an originating ring.
func (r *aaTessRing) makeOriginalRing() {
	for i := range r.pts {
		r.pts[i].origEdgeID = r.pts[i].index
	}
}

// initFromTess computes this ring's normals and bisectors from tess. It should be called after all the indices have
// been added (via addIdx).
func (r *aaTessRing) initFromTess(tess *aaConvexTessellator) {
	r.computeNormals(tess)
	r.computeBisectors(tess)
}

// initFrom sets this ring's per-point normals and bisectors directly from precomputed slices.
func (r *aaTessRing) initFrom(norms, bisectors []geom.Point) {
	for i := range r.pts {
		r.pts[i].norm = norms[i]
		r.pts[i].bisector = bisectors[i]
	}
}

func (r *aaTessRing) norm(index int) geom.Point { return r.pts[index].norm }

func (r *aaTessRing) bisector(index int) geom.Point { return r.pts[index].bisector }

func (r *aaTessRing) index(index int) int { return r.pts[index].index }

func (r *aaTessRing) origEdgeID(index int) int { return r.pts[index].origEdgeID }

// computeNormals computes the outward facing normal at each vertex.
func (r *aaTessRing) computeNormals(tess *aaConvexTessellator) {
	for cur := range r.pts {
		next := (cur + 1) % len(r.pts)
		norm := tess.point(r.pts[next].index).Sub(tess.point(r.pts[cur].index))
		norm.Normalize()
		r.pts[cur].norm = norm.MakeOrthog(tess.side)
	}
}

func (r *aaTessRing) computeBisectors(tess *aaConvexTessellator) {
	prev := len(r.pts) - 1
	for cur := 0; cur < len(r.pts); prev, cur = cur, cur+1 {
		bisector := r.pts[cur].norm.Add(r.pts[prev].norm)
		if !bisector.Normalize() {
			bisector = r.pts[cur].norm.MakeOrthog(-tess.side).
				Add(r.pts[prev].norm.MakeOrthog(tess.side))
			if !bisector.Normalize() {
				panic("aaTessRing bisector normalization failed")
			}
		} else {
			bisector = bisector.Negated() // make the bisector face in
		}
		r.pts[cur].bisector = bisector
	}
}

// aaConvexTessellator holds the global pool of points and the triangulation that connects them, and drives the
// tessellation process. The outward facing normals of the original polygon are stored (in norms) to service
// computeDepthFromEdge requests.
type aaConvexTessellator struct {
	rings          [2]aaTessRing
	initialRing    aaTessRing
	candidateVerts aaTessCandidateVerts
	curveState     []aaTessCurveState
	// The outward facing normals for the original polygon.
	norms []geom.Point
	// The inward facing bisector at each point in the original polygon. Only needed for exterior ring creation and then
	// handed off to the initial ring.
	bisectors   []geom.Point
	pointBuffer []geom.Point
	movable     []bool
	// The triangulation of the points.
	indices   []int
	coverages []float32
	// pts, coverages, movable & curveState always have the same number of elements. Movable points are those that can
	// be slid further along their bisector.
	pts []geom.Point
	// The stroke width is only used for stroke or stroke-and-fill styles.
	strokeWidth float32
	miterLimit  float32
	// Accumulated error when removing near colinear points to prevent an overly greedy simplification.
	accumLinearError float32
	style            stroke.Style
	join             stroke.Join
	side             geom.PointSide // winding of the original polygon
}

// newAAConvexTessellator creates a tessellator configured with the given style, stroke width, join, and miter limit.
func newAAConvexTessellator(style stroke.Style, strokeWidth float32, join stroke.Join, miterLimit float32) *aaConvexTessellator {
	return &aaConvexTessellator{
		side:        geom.PointSideOn,
		strokeWidth: strokeWidth,
		style:       style,
		join:        join,
		miterLimit:  miterLimit,
	}
}

// The next five should only be called after tessellate to extract the result.
func (t *aaConvexTessellator) numPts() int { return len(t.pts) }

func (t *aaConvexTessellator) numIndices() int { return len(t.indices) }

func (t *aaConvexTessellator) lastPoint() geom.Point { return t.pts[len(t.pts)-1] }

func (t *aaConvexTessellator) point(index int) geom.Point { return t.pts[index] }

func (t *aaConvexTessellator) index(index int) int { return t.indices[index] }

func (t *aaConvexTessellator) coverage(index int) float32 { return t.coverages[index] }

// addPt appends a point to the global pool. Movable points are those that can be slid along their bisector; a point is
// immovable if it is part of the original polygon or it results from the fusing of two bisectors.
func (t *aaConvexTessellator) addPt(pt geom.Point, coverage float32, movable bool, curve aaTessCurveState) int {
	if !pt.IsFinite() {
		panic("addPt requires a finite point")
	}
	t.validate()
	index := len(t.pts)
	t.pts = append(t.pts, pt)
	t.coverages = append(t.coverages, coverage)
	t.movable = append(t.movable, movable)
	t.curveState = append(t.curveState, curve)
	t.validate()
	return index
}

func (t *aaConvexTessellator) popLastPt() {
	t.validate()
	t.pts = t.pts[:len(t.pts)-1]
	t.coverages = t.coverages[:len(t.coverages)-1]
	t.movable = t.movable[:len(t.movable)-1]
	t.curveState = t.curveState[:len(t.curveState)-1]
	t.validate()
}

// popFirstPtShuffle removes the first point by moving the last element into slot 0.
func (t *aaConvexTessellator) popFirstPtShuffle() {
	t.validate()
	last := len(t.pts) - 1
	t.pts[0] = t.pts[last]
	t.pts = t.pts[:last]
	t.coverages[0] = t.coverages[last]
	t.coverages = t.coverages[:last]
	t.movable[0] = t.movable[last]
	t.movable = t.movable[:last]
	t.curveState[0] = t.curveState[last]
	t.curveState = t.curveState[:last]
	t.validate()
}

func (t *aaConvexTessellator) updatePt(index int, pt geom.Point, coverage float32) {
	t.validate()
	if !t.movable[index] {
		panic("updatePt on an immovable point")
	}
	t.pts[index] = pt
	t.coverages[index] = coverage
}

func (t *aaConvexTessellator) addTri(i0, i1, i2 int) {
	if i0 == i1 || i1 == i2 || i2 == i0 {
		return
	}
	t.indices = append(t.indices, i0, i1, i2)
}

func (t *aaConvexTessellator) reservePts(count int) {
	grow := func(n int) {
		if cap(t.pts) < n {
			newPts := make([]geom.Point, len(t.pts), n)
			copy(newPts, t.pts)
			t.pts = newPts
			newCov := make([]float32, len(t.coverages), n)
			copy(newCov, t.coverages)
			t.coverages = newCov
			newMov := make([]bool, len(t.movable), n)
			copy(newMov, t.movable)
			t.movable = newMov
		}
	}
	grow(count)
}

func (t *aaConvexTessellator) computeNormals() {
	normalToVector := func(v geom.Point) geom.Point {
		n := v.MakeOrthog(t.side)
		if !n.Normalize() {
			panic("computeNormals normalization failed")
		}
		return n
	}

	// Check the cross product of the final trio.
	t.norms = append(t.norms[:0], make([]geom.Point, len(t.pts))...)
	t.norms[0] = t.pts[1].Sub(t.pts[0])
	t.norms[len(t.norms)-1] = t.pts[0].Sub(t.pts[len(t.pts)-1])
	cross := t.norms[0].Cross(t.norms[len(t.norms)-1])
	if cross > 0 {
		t.side = geom.PointSideRight
	} else {
		t.side = geom.PointSideLeft
	}
	t.norms[0] = normalToVector(t.norms[0])
	for cur := 1; cur < len(t.norms)-1; cur++ {
		t.norms[cur] = normalToVector(t.pts[cur+1].Sub(t.pts[cur]))
	}
	t.norms[len(t.norms)-1] = normalToVector(t.norms[len(t.norms)-1])
}

func (t *aaConvexTessellator) computeBisectors() {
	t.bisectors = append(t.bisectors[:0], make([]geom.Point, len(t.norms))...)

	prev := len(t.bisectors) - 1
	for cur := 0; cur < len(t.bisectors); prev, cur = cur, cur+1 {
		bisector := t.norms[cur].Add(t.norms[prev])
		if !bisector.Normalize() {
			bisector = t.norms[cur].MakeOrthog(-t.side).
				Add(t.norms[prev].MakeOrthog(t.side))
			if !bisector.Normalize() {
				panic("computeBisectors normalization failed")
			}
		} else {
			bisector = bisector.Negated() // make the bisector face in
		}
		t.bisectors[cur] = bisector
		if t.curveState[prev] == aaTessCurveIndeterminate {
			if t.curveState[cur] == aaTessCurveSharp {
				t.curveState[prev] = aaTessCurveSharp
			} else {
				if geom.ScalarAbs(t.norms[cur].Dot(t.norms[prev])) >
					aaTessCurveConnectionThreshold {
					t.curveState[prev] = aaTessCurveCurve
					t.curveState[cur] = aaTessCurveCurve
				} else {
					t.curveState[prev] = aaTessCurveSharp
					t.curveState[cur] = aaTessCurveSharp
				}
			}
		}
	}
}

// createInsetRings creates as many rings as we need to (up to a predefined limit) to reach the specified target depth.
// If we are in fill mode, the final ring will automatically be fanned.
func (t *aaConvexTessellator) createInsetRings(previousRing *aaTessRing, initialDepth, initialCoverage, targetDepth, targetCoverage float32) (finalRing *aaTessRing, done bool) {
	const maxNumRings = 8

	if previousRing.numPts() < 3 {
		return nil, false
	}
	currentRing := previousRing
	var i int
	for i = 0; i < maxNumRings; i++ {
		nextRing := t.getNextRing(currentRing)
		if nextRing == currentRing {
			panic("getNextRing returned the current ring")
		}
		ringDone := t.createInsetRing(currentRing, nextRing, initialDepth, initialCoverage,
			targetDepth, targetCoverage, i == 0)
		currentRing = nextRing
		if ringDone {
			break
		}
		currentRing.initFromTess(t)
	}

	if i == maxNumRings {
		// Bail if we've exceeded the amount of time we want to throw at this.
		t.terminate(currentRing)
		return nil, false
	}
	done = currentRing.numPts() >= 3
	if done {
		currentRing.initFromTess(t)
	}
	return currentRing, done
}

// tessellate runs the tessellation. The general idea is to, conceptually, start with the original polygon and slide the
// vertices along the bisectors until the first intersection. At that point two of the edges collapse and the process
// repeats on the new polygon. The polygon state is captured in the Ring class while the tessellator controls the
// iteration. The CandidateVerts holds the formative points for the next ring.
func (t *aaConvexTessellator) tessellate(m *geom.Matrix, p *path.Path) bool {
	if !t.extractFromPath(m, p) {
		return false
	}

	coverage := float32(1)
	scaleFactor := float32(0)

	if t.style == stroke.StyleStrokeAndFill {
		if !m.IsSimilarity() {
			panic("stroke-and-fill requires a similarity matrix")
		}
		scaleFactor = m.MaxScale() // x and y scale are the same
		effectiveStrokeWidth := scaleFactor * t.strokeWidth
		var outerStrokeAndAARing aaTessRing
		t.createOuterRing(&t.initialRing,
			effectiveStrokeWidth/2+aaTessAntialiasingRadius, 0, &outerStrokeAndAARing)

		// Discard all the triangles added between the originating ring and the new outer ring.
		t.indices = t.indices[:0]

		outerStrokeAndAARing.initFromTess(t)

		outerStrokeAndAARing.makeOriginalRing()

		// Add the outer stroke ring's normals to the originating ring's normals so it can also act as an originating
		// ring.
		t.norms = append(t.norms, make([]geom.Point, outerStrokeAndAARing.numPts())...)
		for i := 0; i < outerStrokeAndAARing.numPts(); i++ {
			if outerStrokeAndAARing.index(i) >= len(t.norms) {
				panic("outer ring index out of range")
			}
			t.norms[outerStrokeAndAARing.index(i)] = outerStrokeAndAARing.norm(i)
		}

		// The bisectors are only needed for the computation of the outer ring.
		t.bisectors = t.bisectors[:0]

		t.createInsetRings(&outerStrokeAndAARing, 0, 0, 2*aaTessAntialiasingRadius, 1)

		t.validate()
		return true
	}

	if t.style == stroke.StyleStroke {
		if t.strokeWidth < 0 {
			panic("stroke requires a non-negative width")
		}
		if !m.IsSimilarity() {
			panic("stroke requires a similarity matrix")
		}
		scaleFactor = m.MaxScale() // x and y scale are the same
		effectiveStrokeWidth := scaleFactor * t.strokeWidth
		var outerStrokeRing aaTessRing
		t.createOuterRing(&t.initialRing, effectiveStrokeWidth/2-aaTessAntialiasingRadius,
			coverage, &outerStrokeRing)
		outerStrokeRing.initFromTess(t)
		var outerAARing aaTessRing
		t.createOuterRing(&outerStrokeRing, aaTessAntialiasingRadius*2, 0, &outerAARing)
	} else {
		var outerAARing aaTessRing
		t.createOuterRing(&t.initialRing, aaTessAntialiasingRadius, 0, &outerAARing)
	}

	// The bisectors are only needed for the computation of the outer ring.
	t.bisectors = t.bisectors[:0]
	if t.style == stroke.StyleStroke && t.initialRing.numPts() > 2 {
		if t.strokeWidth < 0 {
			panic("stroke requires a non-negative width")
		}
		effectiveStrokeWidth := scaleFactor * t.strokeWidth
		strokeDepth := effectiveStrokeWidth/2 - aaTessAntialiasingRadius
		if insetStrokeRing, ok := t.createInsetRings(&t.initialRing, 0, coverage, strokeDepth,
			coverage); ok {
			t.createInsetRings(insetStrokeRing, strokeDepth, coverage,
				strokeDepth+aaTessAntialiasingRadius*2, 0)
		}
	} else {
		t.createInsetRings(&t.initialRing, 0, 0.5, aaTessAntialiasingRadius, 1)
	}

	t.validate()
	return true
}

func (t *aaConvexTessellator) computeDepthFromEdge(edgeIdx int, p geom.Point) float32 {
	if edgeIdx >= len(t.norms) {
		panic("edge index out of range")
	}
	v := p.Sub(t.pts[edgeIdx])
	return -t.norms[edgeIdx].Dot(v)
}

// computePtAlongBisector finds a point that is 'desiredDepth' away from the 'edgeIdx'-th edge and lies along the
// 'bisector' from the 'startIdx'-th point.
func (t *aaConvexTessellator) computePtAlongBisector(startIdx int, bisector geom.Point, edgeIdx int, desiredDepth float32) (geom.Point, bool) {
	norm := t.norms[edgeIdx]

	// First find the point where the edge and the bisector intersect.
	var newP geom.Point
	tt, ok := aaTessPerpIntersect(t.pts[startIdx], bisector, t.pts[edgeIdx], norm)
	if !ok {
		return geom.Point{}, false
	}
	switch {
	case geom.ScalarNearlyEqual(tt, 0):
		// The start point was one of the original ring points.
		if startIdx >= len(t.pts) {
			panic("start index out of range")
		}
		newP = t.pts[startIdx]
	case tt < 0:
		newP = bisector.Scaled(tt).Add(t.pts[startIdx])
	default:
		return geom.Point{}, false
	}

	// Then offset along the bisector from that point the correct distance.
	dot := bisector.Dot(norm)
	tt = -desiredDepth / dot
	return bisector.Scaled(tt).Add(newP), true
}

// extractFromPath returns false on failure/degenerate path.
func (t *aaConvexTessellator) extractFromPath(m *geom.Matrix, p *path.Path) bool {
	bounds, _ := m.MapRect(p.Bounds())
	if !bounds.IsFinite() {
		// We could do something smarter here like clip the path based on the bounds of the dst. We'd have to be careful
		// about strokes to ensure we don't draw something wrong.
		return false
	}

	// Outer ring: 3*numPts; middle ring: numPts; presumptive inner ring: numPts.
	t.reservePts(5 * p.CountPoints())
	// Outer ring: 12*numPts; middle ring: 0; presumptive inner ring: 6*numPts + 6.
	if cap(t.indices) < 18*p.CountPoints()+6 {
		newIndices := make([]int, len(t.indices), 18*p.CountPoints()+6)
		copy(newIndices, t.indices)
		t.indices = newIndices
	}

	// Reset the accumulated error for all the future lineTo() calls when iterating over the path.
	t.accumLinearError = 0
	it := path.NewEdgeIter(p)
	for {
		e, ok := it.Next()
		if !ok {
			break
		}
		switch e.Verb {
		case path.VerbLine:
			if !allPointsEq(e.Pts[:2]) {
				t.lineToTransformed(m, e.Pts[1], aaTessCurveSharp)
			}
		case path.VerbQuad:
			if !allPointsEq(e.Pts[:3]) {
				t.quadToTransformed(m, e.Pts)
			}
		case path.VerbCubic:
			if !allPointsEq(e.Pts[:4]) {
				t.cubicTo(m, e.Pts)
			}
		case path.VerbConic:
			if !allPointsEq(e.Pts[:3]) {
				t.conicTo(m, e.Pts, it.ConicWeight())
			}
		default:
			panic("unknown edge type")
		}
	}

	if t.numPts() < 2 {
		return false
	}

	// Check if last point is a duplicate of the first point. If so, remove it.
	if aaTessDuplicatePt(t.pts[t.numPts()-1], t.pts[0]) {
		t.popLastPt()
	}

	// Remove any lingering colinear points where the path wraps around.
	t.accumLinearError = 0
	noRemovalsToDo := false
	for !noRemovalsToDo && t.numPts() >= 3 {
		switch {
		case aaTessPointsAreColinearAndBIsMiddle(t.pts[len(t.pts)-2], t.pts[len(t.pts)-1],
			t.pts[0], &t.accumLinearError):
			t.popLastPt()
		case aaTessPointsAreColinearAndBIsMiddle(t.pts[len(t.pts)-1], t.pts[0], t.pts[1],
			&t.accumLinearError):
			t.popFirstPtShuffle()
		default:
			noRemovalsToDo = true
		}
	}

	// Compute the normals and bisectors.
	if len(t.norms) != 0 {
		panic("norms must be empty at extraction")
	}
	switch {
	case t.numPts() >= 3:
		t.computeNormals()
		t.computeBisectors()
	case t.numPts() == 2:
		// We've got two points, so we're degenerate.
		if t.style == stroke.StyleFill {
			// It's a fill, so we don't need to worry about degenerate paths.
			return false
		}
		// For stroking, we still need to process the degenerate path, so fix it up.
		t.side = geom.PointSideLeft

		norm := t.pts[1].Sub(t.pts[0]).MakeOrthog(t.side)
		norm.Normalize()
		t.norms = append(t.norms, norm, norm.Negated())
		// We won't actually use the bisectors, so just push zeroes.
		t.bisectors = append(t.bisectors, geom.Point{}, geom.Point{})
	default:
		return false
	}

	t.candidateVerts.setReserve(t.numPts())
	t.initialRing.setReserve(t.numPts())
	for i := 0; i < t.numPts(); i++ {
		t.initialRing.addIdx(i, i)
	}
	t.initialRing.initFrom(t.norms, t.bisectors)

	t.validate()
	return true
}

// getNextRing flip-flops back and forth between rings[0] and rings[1].
func (t *aaConvexTessellator) getNextRing(lastRing *aaTessRing) *aaTessRing {
	nextRing := 0
	if lastRing == &t.rings[0] {
		nextRing = 1
	}
	t.rings[nextRing].setReserve(t.initialRing.numPts())
	t.rings[nextRing].rewind()
	return &t.rings[nextRing]
}

// fanRing fans out from point 0.
func (t *aaConvexTessellator) fanRing(ring *aaTessRing) {
	startIdx := ring.index(0)
	for cur := ring.numPts() - 2; cur >= 0; cur-- {
		t.addTri(startIdx, ring.index(cur), ring.index(cur+1))
	}
}

func (t *aaConvexTessellator) createOuterRing(previousRing *aaTessRing, outset, coverage float32, nextRing *aaTessRing) {
	numPts := previousRing.numPts()
	if numPts == 0 {
		return
	}

	prev := numPts - 1
	lastPerpIdx := -1
	firstPerpIdx := -1

	outsetSq := outset * outset
	miterLimitSq := outset * t.miterLimit
	miterLimitSq *= miterLimitSq
	for cur := 0; cur < numPts; cur++ {
		originalIdx := previousRing.index(cur)
		// For each vertex of the original polygon we add at least two points to the outset polygon — one extending
		// perpendicular to each impinging edge. Connecting these two points yields a bevel join. We need one additional
		// point for a mitered join, and a round join requires one or more points depending upon curvature.

		// The perpendicular point for the last edge.
		normal1 := previousRing.norm(prev)
		perp1 := normal1.Scaled(outset).Add(t.point(originalIdx))

		// The perpendicular point for the next edge.
		normal2 := previousRing.norm(cur)
		perp2 := normal2.Scaled(outset).Add(t.pts[originalIdx])

		curve := t.curveState[originalIdx]

		// We know it isn't a duplicate of the prior point (since it and this one are just perpendicular offsets from
		// the non-merged polygon points).
		perp1Idx := t.addPt(perp1, coverage, false, curve)
		nextRing.addIdx(perp1Idx, originalIdx)

		var perp2Idx int
		// For very shallow angles all the corner points could fuse.
		if aaTessDuplicatePt(perp2, t.point(perp1Idx)) {
			perp2Idx = perp1Idx
		} else {
			perp2Idx = t.addPt(perp2, coverage, false, curve)
		}

		if perp2Idx != perp1Idx {
			if curve == aaTessCurveCurve {
				// Bevel or round depending upon curvature.
				dotProd := normal1.Dot(normal2)
				if dotProd < aaTessRoundCapThreshold {
					// Currently we "round" by creating a single extra point, which produces good results for common
					// cases. For thick strokes with high curvature, we will need to add more points; for the time being
					// we simply fall back to software rendering for thick strokes.
					miter := previousRing.bisector(cur)
					miter.SetLength(-outset)
					miter = miter.Add(t.pts[originalIdx])

					// For very shallow angles all the corner points could fuse.
					if !aaTessDuplicatePt(miter, t.point(perp1Idx)) {
						miterIdx := t.addPt(miter, coverage, false, aaTessCurveSharp)
						nextRing.addIdx(miterIdx, originalIdx)
						// The two triangles for the corner.
						t.addTri(originalIdx, perp1Idx, miterIdx)
						t.addTri(originalIdx, miterIdx, perp2Idx)
					}
				} else {
					t.addTri(originalIdx, perp1Idx, perp2Idx)
				}
			} else {
				switch t.join {
				case stroke.JoinMiter:
					// The bisector outset point.
					miter := previousRing.bisector(cur)
					dotProd := normal1.Dot(normal2)
					// The max is because this could go slightly negative if precision causes us to become slightly
					// concave.
					sinHalfAngleSq := max((1+dotProd)/2, 0)
					lengthSq := geom.IEEEFloatDivide(outsetSq, sinHalfAngleSq)
					if lengthSq > miterLimitSq {
						// Just bevel it.
						t.addTri(originalIdx, perp1Idx, perp2Idx)
						break
					}
					miter.SetLength(-geom.ScalarSqrt(lengthSq))
					miter = miter.Add(t.pts[originalIdx])

					// For very shallow angles all the corner points could fuse.
					if !aaTessDuplicatePt(miter, t.point(perp1Idx)) {
						miterIdx := t.addPt(miter, coverage, false, aaTessCurveSharp)
						nextRing.addIdx(miterIdx, originalIdx)
						// The two triangles for the corner.
						t.addTri(originalIdx, perp1Idx, miterIdx)
						t.addTri(originalIdx, miterIdx, perp2Idx)
					} else {
						// Ignore the miter point as it's so close to perp1/perp2 and simply bevel.
						t.addTri(originalIdx, perp1Idx, perp2Idx)
					}
				case stroke.JoinBevel:
					t.addTri(originalIdx, perp1Idx, perp2Idx)
				default:
					// kRound_Join is unsupported for now. AALinearizingConvexPathRenderer is only willing to draw
					// mitered or beveled, so we should never get here.
					panic("unsupported round join in outer ring creation")
				}
			}

			nextRing.addIdx(perp2Idx, originalIdx)
		}

		if cur == 0 {
			// Store the index of the first perpendicular point to finish up.
			firstPerpIdx = perp1Idx
			if lastPerpIdx != -1 {
				panic("lastPerpIdx already set")
			}
		} else {
			// The triangles for the previous edge.
			prevIdx := previousRing.index(prev)
			t.addTri(prevIdx, perp1Idx, originalIdx)
			t.addTri(prevIdx, lastPerpIdx, perp1Idx)
		}

		// Track the last perpendicular outset point so we can construct the trailing edge triangles.
		lastPerpIdx = perp2Idx
		prev = cur
	}

	// Pick up the final edge rect.
	lastIdx := previousRing.index(numPts - 1)
	t.addTri(lastIdx, firstPerpIdx, previousRing.index(0))
	t.addTri(lastIdx, lastPerpIdx, firstPerpIdx)

	t.validate()
}

// terminate handles the case where something went wrong in the creation of the next ring. If we're filling the shape,
// just go ahead and fan it.
func (t *aaConvexTessellator) terminate(ring *aaTessRing) {
	if t.style != stroke.StyleStroke && ring.numPts() > 0 {
		t.fanRing(ring)
	}
}

// aaTessComputeCoverage linearly interpolates the coverage at depth between the coverage values known at initialDepth
// and targetDepth, clamped to [0, 1].
func aaTessComputeCoverage(depth, initialDepth, initialCoverage, targetDepth, targetCoverage float32) float32 {
	if geom.ScalarNearlyEqual(initialDepth, targetDepth) {
		return targetCoverage
	}
	result := (depth-initialDepth)/(targetDepth-initialDepth)*
		(targetCoverage-initialCoverage) + initialCoverage
	return min(max(result, 0), 1)
}

// createInsetRing returns true when processing is complete.
func (t *aaConvexTessellator) createInsetRing(lastRing, nextRing *aaTessRing, initialDepth, initialCoverage, targetDepth, targetCoverage float32, forceNew bool) bool {
	done := false

	t.candidateVerts.rewind()

	// Loop through all the points in the ring and find the intersection with the smallest depth.
	minDist := geom.ScalarMax
	minT := float32(0)
	minEdgeIdx := -1

	for cur := 0; cur < lastRing.numPts(); cur++ {
		next := (cur + 1) % lastRing.numPts()

		tt, ok := aaTessIntersect(t.point(lastRing.index(cur)), lastRing.bisector(cur),
			t.point(lastRing.index(next)), lastRing.bisector(next))
		// The bisectors may be parallel (!ok) or the previous ring may have become slightly concave due to accumulated
		// error (t <= 0).
		if !ok || tt <= 0 {
			continue
		}
		dist := -tt * lastRing.norm(cur).Dot(lastRing.bisector(cur))

		if minDist > dist {
			minDist = dist
			minT = tt
			minEdgeIdx = cur
		}
	}

	if minEdgeIdx == -1 {
		return false
	}
	newPt := lastRing.bisector(minEdgeIdx).Scaled(minT).
		Add(t.point(lastRing.index(minEdgeIdx)))

	depth := t.computeDepthFromEdge(lastRing.origEdgeID(minEdgeIdx), newPt)
	if depth >= targetDepth {
		// None of the bisectors intersect before reaching the desired depth. Just step them all to the desired depth.
		depth = targetDepth
		done = true
	}

	// 'dst' stores where each point in the last ring maps to/transforms into in the next ring.
	dst := make([]int, lastRing.numPts())

	// Create the first point (who compares with no one).
	newPt, ok := t.computePtAlongBisector(lastRing.index(0), lastRing.bisector(0),
		lastRing.origEdgeID(0), depth)
	if !ok {
		t.terminate(lastRing)
		return true
	}
	dst[0] = t.candidateVerts.addNewPt(newPt, lastRing.index(0), lastRing.origEdgeID(0),
		!t.movable[lastRing.index(0)])

	// Handle the middle points (who only compare with the prior point).
	for cur := 1; cur < lastRing.numPts()-1; cur++ {
		newPt, ok = t.computePtAlongBisector(lastRing.index(cur), lastRing.bisector(cur),
			lastRing.origEdgeID(cur), depth)
		if !ok {
			t.terminate(lastRing)
			return true
		}
		if !aaTessDuplicatePt(newPt, t.candidateVerts.lastPoint()) {
			dst[cur] = t.candidateVerts.addNewPt(newPt, lastRing.index(cur),
				lastRing.origEdgeID(cur), !t.movable[lastRing.index(cur)])
		} else {
			dst[cur] = t.candidateVerts.fuseWithPrior(lastRing.origEdgeID(cur))
		}
	}

	// Check on the last point (handling the wrap around).
	cur := lastRing.numPts() - 1
	newPt, ok = t.computePtAlongBisector(lastRing.index(cur), lastRing.bisector(cur),
		lastRing.origEdgeID(cur), depth)
	if !ok {
		t.terminate(lastRing)
		return true
	}
	dupPrev := aaTessDuplicatePt(newPt, t.candidateVerts.lastPoint())
	dupNext := aaTessDuplicatePt(newPt, t.candidateVerts.firstPoint())

	switch {
	case !dupPrev && !dupNext:
		dst[cur] = t.candidateVerts.addNewPt(newPt, lastRing.index(cur),
			lastRing.origEdgeID(cur), !t.movable[lastRing.index(cur)])
	case dupPrev && !dupNext:
		dst[cur] = t.candidateVerts.fuseWithPrior(lastRing.origEdgeID(cur))
	case !dupPrev && dupNext:
		dst[cur] = t.candidateVerts.fuseWithNext()
	default:
		dupPrevVsNext := aaTessDuplicatePt(t.candidateVerts.firstPoint(),
			t.candidateVerts.lastPoint())
		if !dupPrevVsNext {
			dst[cur] = t.candidateVerts.fuseWithPrior(lastRing.origEdgeID(cur))
		} else {
			fused := t.candidateVerts.fuseWithBoth()
			dst[cur] = fused
			targetIdx := dst[cur-1]
			for i := cur - 1; i >= 0 && dst[i] == targetIdx; i-- {
				dst[i] = fused
			}
		}
	}

	// Fold the new ring's points into the global pool.
	for i := 0; i < t.candidateVerts.numPts(); i++ {
		var newIdx int
		if t.candidateVerts.needsToBeNew(i) || forceNew {
			// If the originating index is still valid then this point wasn't fused (and is thus movable).
			coverage := aaTessComputeCoverage(depth, initialDepth, initialCoverage,
				targetDepth, targetCoverage)
			newIdx = t.addPt(t.candidateVerts.point(i), coverage,
				t.candidateVerts.originatingIdx(i) != -1, aaTessCurveSharp)
		} else {
			if t.candidateVerts.originatingIdx(i) == -1 {
				panic("fused candidate must be new")
			}
			t.updatePt(t.candidateVerts.originatingIdx(i), t.candidateVerts.point(i), targetCoverage)
			newIdx = t.candidateVerts.originatingIdx(i)
		}

		nextRing.addIdx(newIdx, t.candidateVerts.origEdge(i))
	}

	// 'dst' currently has indices into the ring. Remap these to be indices into the global pool since the triangulation
	// operates in that space.
	for i := range dst {
		dst[i] = nextRing.index(dst[i])
	}

	for i := 0; i < lastRing.numPts(); i++ {
		next := (i + 1) % lastRing.numPts()

		t.addTri(lastRing.index(i), lastRing.index(next), dst[next])
		t.addTri(lastRing.index(i), dst[next], dst[i])
	}

	if done && t.style != stroke.StyleStroke {
		// Fill or stroke-and-fill.
		t.fanRing(nextRing)
	}

	if nextRing.numPts() < 3 {
		done = true
	}
	return done
}

func (t *aaConvexTessellator) validate() {
	if len(t.pts) != len(t.movable) {
		panic("pts/movable length mismatch")
	}
	if len(t.pts) != len(t.coverages) {
		panic("pts/coverages length mismatch")
	}
	if len(t.pts) != len(t.curveState) {
		panic("pts/curveState length mismatch")
	}
	if len(t.indices)%3 != 0 {
		panic("indices not a multiple of 3")
	}
	if len(t.bisectors) != 0 && len(t.bisectors) != len(t.norms) {
		panic("bisectors/norms length mismatch")
	}
}

func (t *aaConvexTessellator) lineTo(p geom.Point, curve aaTessCurveState) {
	if t.numPts() > 0 && aaTessDuplicatePt(p, t.lastPoint()) {
		return
	}

	if t.numPts() >= 2 && aaTessPointsAreColinearAndBIsMiddle(t.pts[len(t.pts)-2],
		t.pts[len(t.pts)-1], p, &t.accumLinearError) {
		// The old last point is on the line from the second to last to the new point.
		t.popLastPt()
		// Double-check that the new last point is not a duplicate of the new point. In an ideal world this wouldn't be
		// necessary (since it's only possible for non-convex paths), but floating point precision issues mean it can
		// actually happen on paths that were determined to be convex.
		if aaTessDuplicatePt(p, t.lastPoint()) {
			return
		}
	} else {
		t.accumLinearError = 0
	}
	initialRingCoverage := float32(1)
	if t.style == stroke.StyleFill {
		initialRingCoverage = 0.5
	}
	t.addPt(p, initialRingCoverage, false, curve)
}

func (t *aaConvexTessellator) lineToTransformed(m *geom.Matrix, p geom.Point, curve aaTessCurveState) {
	t.lineTo(m.MapPoint(p), curve)
}

func (t *aaConvexTessellator) quadTo(pts []geom.Point) {
	maxCount := quadraticPointCount([3]geom.Point{pts[0], pts[1], pts[2]}, aaTessQuadTolerance)
	if cap(t.pointBuffer) < maxCount {
		t.pointBuffer = make([]geom.Point, maxCount)
	}
	target := t.pointBuffer[:maxCount]
	count := generateQuadraticPoints(pts[0], pts[1], pts[2], aaTessQuadToleranceSqd, target,
		maxCount)
	for i := 0; i < count-1; i++ {
		t.lineTo(target[i], aaTessCurveCurve)
	}
	lastState := aaTessCurveIndeterminate
	if count == 1 {
		lastState = aaTessCurveSharp
	}
	t.lineTo(target[count-1], lastState)
}

func (t *aaConvexTessellator) quadToTransformed(m *geom.Matrix, srcPts []geom.Point) {
	var pts [3]geom.Point
	m.MapPoints(pts[:], srcPts[:3])
	t.quadTo(pts[:])
}

func (t *aaConvexTessellator) cubicTo(m *geom.Matrix, srcPts []geom.Point) {
	var pts [4]geom.Point
	m.MapPoints(pts[:], srcPts[:4])
	maxCount := cubicPointCount(pts, aaTessCubicTolerance)
	if cap(t.pointBuffer) < maxCount {
		t.pointBuffer = make([]geom.Point, maxCount)
	}
	target := t.pointBuffer[:maxCount]
	count := generateCubicPoints(pts[0], pts[1], pts[2], pts[3], aaTessCubicToleranceSqd,
		target, maxCount)
	for i := 0; i < count-1; i++ {
		t.lineTo(target[i], aaTessCurveCurve)
	}
	lastState := aaTessCurveIndeterminate
	if count == 1 {
		lastState = aaTessCurveSharp
	}
	t.lineTo(target[count-1], lastState)
}

func (t *aaConvexTessellator) conicTo(m *geom.Matrix, srcPts []geom.Point, w float32) {
	var pts [3]geom.Point
	m.MapPoints(pts[:], srcPts[:3])
	quads, count := path.ConicToQuads(pts[:], w, aaTessConicTolerance)
	lastPoint := quads[0]
	for i := 0; i < count; i++ {
		var quadPts [3]geom.Point
		quadPts[0] = lastPoint
		quadPts[1] = quads[1+2*i]
		if i == count-1 {
			quadPts[2] = pts[2]
		} else {
			quadPts[2] = quads[2+2*i]
		}
		t.quadTo(quadPts[:])
		lastPoint = quadPts[2]
	}
}
