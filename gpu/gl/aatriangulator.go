// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Triangulates a path in device space with a mesh of alpha ramps for antialiasing. After the base triangulator's stages
// 1-5 produce polygons, the boundaries are extracted per the fill rule (5b), simplified to remove "pointy" vertices
// that would invert on stroking (5c), and stroked half a pixel inward and outward along their normals (5d), building a
// new mesh whose interior vertices carry alpha 255 and exterior vertices alpha 0; overlap regions are collapsed with an
// event queue, and the new mesh is retessellated. The event queue is a container/heap min-heap with the comparator
// inverted; equal-alpha event order is unspecified.

package gl

import (
	"container/heap"
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// aaCosMiterAngle corresponds to an angle of ~14 degrees.
const aaCosMiterAngle = 0.97

// ssVertex is a vertex in the straight-skeleton event graph used to collapse overlap regions.
type ssVertex struct {
	vertex     *triVertex
	prev, next *ssEdge
}

// ssEdge is an edge in the straight-skeleton event graph, tracking the underlying triEdge and any pending collapse
// event.
type ssEdge struct {
	edge       *triEdge
	event      *aaEvent
	prev, next *ssVertex
}

// aaEvent is a pending edge-collapse event: the point and alpha at which two ssEdges meet.
type aaEvent struct {
	edge  *ssEdge
	point geom.Point
	alpha uint8
}

// aaEventList is a priority queue of pending collapse events, ordered so pop returns the highest- or lowest-alpha event
// depending on lessThan.
type aaEventList struct {
	events []*aaEvent
	// lessThan selects max-alpha-first ordering when true, min-alpha-first when false.
	lessThan bool
}

func (l *aaEventList) Len() int { return len(l.events) }

func (l *aaEventList) Less(i, j int) bool {
	if l.lessThan {
		return l.events[j].alpha < l.events[i].alpha
	}
	return l.events[j].alpha > l.events[i].alpha
}

func (l *aaEventList) Swap(i, j int) { l.events[i], l.events[j] = l.events[j], l.events[i] }

func (l *aaEventList) Push(x any) { l.events = append(l.events, x.(*aaEvent)) }

func (l *aaEventList) Pop() any {
	old := l.events
	n := len(old)
	e := old[n-1]
	l.events = old[:n-1]
	return e
}

// aaTriangulator triangulates a path into an antialiased mesh of alpha ramps.
type aaTriangulator struct {
	outerMesh triVertexList
	triangulator
}

// triPathToAATriangles triangulates p into antialiased triangles, written via vertexAllocator, and returns the vertex
// count.
func triPathToAATriangles(p *path.Path, tolerance float32, clipBounds geom.Rect, vertexAllocator eagerVertexAllocator) int {
	aa := &aaTriangulator{}
	aa.path = p
	aa.roundVerticesToQuarterPixel = true
	aa.emitCoverage = true
	aa.tessellateOverride = aa.aaTessellate
	var isLinear bool
	polys, success := aa.pathToPolys(tolerance, clipBounds, &isLinear)
	if !success {
		return 0
	}
	return aa.polysToAATriangles(polys, vertexAllocator)
}

// makeEvent computes and queues the collapse event (if any) for e from its neighboring vertices' bisectors.
func (aa *aaTriangulator) makeEvent(e *ssEdge, events *aaEventList) {
	prev := e.prev.vertex
	next := e.next.vertex
	if prev == next || prev.partner == nil || next.partner == nil {
		return
	}
	bisector1 := makeTriEdge(prev, prev.partner, 1, triEdgeTypeConnector)
	bisector2 := makeTriEdge(next, next.partner, 1, triEdgeTypeConnector)
	var p geom.Point
	var alpha uint8
	if bisector1.intersect(bisector2, &p, &alpha) {
		e.event = &aaEvent{edge: e, point: p, alpha: alpha}
		heap.Push(events, e.event)
	}
}

// makeEventAt computes and queues the collapse event (if any) for edge given a specific vertex v and destination dest,
// re-deriving edge's line through dest first.
func (aa *aaTriangulator) makeEventAt(edge *ssEdge, v, dest *triVertex, events *aaEventList, c triComparator) {
	if v.partner == nil {
		return
	}
	top := edge.edge.top
	bottom := edge.edge.bottom
	if top == nil || bottom == nil {
		return
	}
	line := edge.edge.line
	line.c = -(float64(dest.point.X)*line.a + float64(dest.point.Y)*line.b)
	bisector := makeTriEdge(v, v.partner, 1, triEdgeTypeConnector)
	var p geom.Point
	alpha := dest.alpha
	if line.intersect(bisector.line, &p) && !c.sweepLt(p, top.point) &&
		c.sweepLt(p, bottom.point) {
		edge.event = &aaEvent{edge: edge, point: p, alpha: alpha}
		heap.Push(events, edge.event)
	}
}

// connectPartners joins each connected inner/outer vertex pair with a zero-winding connector edge.
func (aa *aaTriangulator) connectPartners(mesh *triVertexList, c triComparator) {
	for outer := mesh.head; outer != nil; outer = outer.next {
		if inner := outer.partner; inner != nil {
			if (inner.prev != nil || inner.next != nil) &&
				(outer.prev != nil || outer.next != nil) {
				// Connector edges get zero winding, since they are only structural (ensuring no 0-0-0 alpha triangles
				// are produced) and should not affect the poly winding number.
				aa.makeConnectingEdge(outer, inner, triEdgeTypeConnector, c, 0)
				inner.partner = nil
				outer.partner = nil
			}
		}
	}
}

// removeNonBoundaryEdges disconnects edges whose fill state matches on both sides, leaving only the boundary edges
// connected.
func (aa *aaTriangulator) removeNonBoundaryEdges(mesh *triVertexList) {
	var activeEdges triEdgeList
	for v := mesh.head; v != nil; v = v.next {
		if !v.isConnected() {
			continue
		}
		leftEnclosingEdge, _ := triFindEnclosingEdges(v, &activeEdges)
		prevFilled := leftEnclosingEdge != nil && aa.applyFillType(leftEnclosingEdge.winding)
		for e := v.firstEdgeAbove; e != nil; {
			next := e.nextEdgeAbove
			activeEdges.remove(e)
			filled := aa.applyFillType(e.winding)
			if filled == prevFilled {
				e.disconnect()
			}
			prevFilled = filled
			e = next
		}
		prev := leftEnclosingEdge
		for e := v.firstEdgeBelow; e != nil; e = e.nextEdgeBelow {
			if prev != nil {
				e.winding += prev.winding
			}
			activeEdges.insert(e, prev)
			prev = e
		}
	}
}

// aaGetEdgeNormal returns the normal to the edge, not necessarily unit length.
func aaGetEdgeNormal(e *triEdge) geom.Point {
	return geom.Point{X: float32(e.line.a), Y: float32(e.line.b)}
}

// simplifyBoundary (stage 5c) detects and removes "pointy" vertices whose edge normals point in opposite directions and
// whose adjacent vertices are less than a quarter pixel from an edge — these are guaranteed to invert on stroking.
func (aa *aaTriangulator) simplifyBoundary(boundary *triEdgeList, c triComparator) {
	prevEdge := boundary.tail
	prevNormal := aaGetEdgeNormal(prevEdge)
	for e := boundary.head; e != nil; {
		var prev, next *triVertex
		if prevEdge.winding == 1 {
			prev = prevEdge.top
		} else {
			prev = prevEdge.bottom
		}
		if e.winding == 1 {
			next = e.bottom
		} else {
			next = e.top
		}
		distPrev := e.dist(prev.point)
		distNext := prevEdge.dist(next.point)
		normal := aaGetEdgeNormal(e)
		const quarterPixelSq = 0.25 * 0.25
		switch {
		case prev == next:
			boundary.remove(prevEdge)
			boundary.remove(e)
			prevEdge = boundary.tail
			e = boundary.head
			if prevEdge != nil {
				prevNormal = aaGetEdgeNormal(prevEdge)
			}
		case float64(prevNormal.Dot(normal)) < 0 &&
			(distPrev*distPrev <= quarterPixelSq || distNext*distNext <= quarterPixelSq):
			join := aa.makeEdge(prev, next, triEdgeTypeInner, c)
			if prev.point != next.point {
				join.line.normalize()
				join.line = join.line.scaled(float64(join.winding))
			}
			boundary.insert(join, e)
			boundary.remove(prevEdge)
			boundary.remove(e)
			if join.left != nil && join.right != nil {
				prevEdge = join.left
				e = join
			} else {
				prevEdge = boundary.tail
				e = boundary.head
			}
			prevNormal = aaGetEdgeNormal(prevEdge)
		default:
			prevEdge = e
			prevNormal = normal
			e = e.right
		}
	}
}

// connectSSEdge connects v to dest in the straight-skeleton graph, either via a synthetic connector edge or by
// repointing v's partner.
func (aa *aaTriangulator) connectSSEdge(v, dest *triVertex, c triComparator) {
	if v == dest {
		return
	}
	if v.synthetic {
		aa.makeConnectingEdge(v, dest, triEdgeTypeConnector, c, 0)
	} else if v.partner != nil {
		v.partner.partner = dest
		v.partner = nil
	}
}

// apply resolves this collapse event: it creates the new merged vertex, reconnects the surrounding edges, and queues
// any new events the merge produces.
func (ev *aaEvent) apply(mesh *triVertexList, c triComparator, events *aaEventList, aa *aaTriangulator) {
	if ev.edge == nil {
		return
	}
	prev := ev.edge.prev.vertex
	next := ev.edge.next.vertex
	prevEdge := ev.edge.prev.prev
	nextEdge := ev.edge.next.next
	if prevEdge == nil || nextEdge == nil || prevEdge.edge == nil || nextEdge.edge == nil {
		return
	}
	dest := aa.makeSortedVertex(ev.point, ev.alpha, mesh, prev, c)
	dest.synthetic = true
	ssv := &ssVertex{vertex: dest}
	ev.edge.edge = nil

	aa.connectSSEdge(prev, dest, c)
	aa.connectSSEdge(next, dest, c)

	prevEdge.next = ssv
	nextEdge.prev = ssv
	ssv.prev = prevEdge
	ssv.next = nextEdge
	if prevEdge.edge == nil || nextEdge.edge == nil {
		return
	}
	if prevEdge.event != nil {
		prevEdge.event.edge = nil
	}
	if nextEdge.event != nil {
		nextEdge.event.edge = nil
	}
	if prevEdge.prev == nextEdge.next {
		aa.connectSSEdge(prevEdge.prev.vertex, dest, c)
		prevEdge.edge = nil
		nextEdge.edge = nil
	} else {
		aa.computeBisector(prevEdge.edge, nextEdge.edge, dest)
		if dest.partner != nil {
			aa.makeEvent(prevEdge, events)
			aa.makeEvent(nextEdge, events)
		} else {
			aa.makeEventAt(prevEdge, prevEdge.prev.vertex, dest, events, c)
			aa.makeEventAt(nextEdge, nextEdge.next.vertex, dest, events, c)
		}
	}
}

// aaIsOverlapEdge reports whether e's winding indicates an overlap region for its edge type.
func aaIsOverlapEdge(e *triEdge) bool {
	switch e.edgeType {
	case triEdgeTypeOuter:
		return e.winding != 0 && e.winding != 1
	case triEdgeTypeInner:
		return e.winding != 0 && e.winding != -2
	default:
		return false
	}
}

// collapseOverlapRegions is a stripped-down version of tessellate() which computes edges joining two filled regions
// (overlap regions) and collapses them.
func (aa *aaTriangulator) collapseOverlapRegions(mesh *triVertexList, c triComparator, lessThan bool) bool {
	var activeEdges triEdgeList
	events := &aaEventList{lessThan: lessThan}
	ssVertices := make(map[*triVertex]*ssVertex)
	var ssEdges []*ssEdge
	for v := mesh.head; v != nil; v = v.next {
		if !v.isConnected() {
			continue
		}
		leftEnclosingEdge, _ := triFindEnclosingEdges(v, &activeEdges)
		for e := v.lastEdgeAbove; e != nil && e != leftEnclosingEdge; {
			prev := e.prevEdgeAbove
			if prev == nil {
				prev = leftEnclosingEdge
			}
			activeEdges.remove(e)
			leftOverlap := prev != nil && aaIsOverlapEdge(prev)
			rightOverlap := aaIsOverlapEdge(e)
			isOuterBoundary := e.edgeType == triEdgeTypeOuter &&
				(prev == nil || prev.winding == 0 || e.winding == 0)
			if prev != nil {
				e.winding -= prev.winding
			}
			switch {
			case leftOverlap && rightOverlap:
				e.disconnect()
			case leftOverlap || rightOverlap:
				var prevVertex, nextVertex *triVertex
				if e.winding < 0 {
					prevVertex = e.bottom
					nextVertex = e.top
				} else {
					prevVertex = e.top
					nextVertex = e.bottom
				}
				ssPrev := ssVertices[prevVertex]
				if ssPrev == nil {
					ssPrev = &ssVertex{vertex: prevVertex}
					ssVertices[prevVertex] = ssPrev
				}
				ssNext := ssVertices[nextVertex]
				if ssNext == nil {
					ssNext = &ssVertex{vertex: nextVertex}
					ssVertices[nextVertex] = ssNext
				}
				ssEdge := &ssEdge{edge: e, prev: ssPrev, next: ssNext}
				ssEdges = append(ssEdges, ssEdge)
				ssPrev.next = ssEdge
				ssNext.prev = ssEdge
				aa.makeEvent(ssEdge, events)
				if !isOuterBoundary {
					e.disconnect()
				} else {
					// Ensure winding values match the expected scale for the edge type: during merging of collinear
					// edges in overlap regions windings are summed and can exceed the expected +/-1 for outer and +/-2
					// for inner that fills properly during subsequent polygon generation.
					mag := 1.0
					if e.edgeType == triEdgeTypeInner {
						mag = 2.0
					}
					e.winding = int(math.Copysign(mag, float64(e.winding)))
				}
			}
			e = prev
		}
		prev := leftEnclosingEdge
		for e := v.firstEdgeBelow; e != nil; e = e.nextEdgeBelow {
			if prev != nil {
				e.winding += prev.winding
			}
			activeEdges.insert(e, prev)
			prev = e
		}
	}
	complexMesh := events.Len() > 0

	for events.Len() > 0 {
		event := heap.Pop(events).(*aaEvent)
		event.apply(mesh, c, events, aa)
	}
	for _, edge := range ssEdges {
		if e := edge.edge; e != nil {
			aa.makeConnectingEdge(edge.prev.vertex, edge.next.vertex, e.edgeType, c, 0)
		}
	}
	return complexMesh
}

// aaInversion reports whether the winding implied by the sweep order of prev/next disagrees with origEdge's recorded
// winding, meaning the stroked boundary has locally inverted.
func aaInversion(prev, next *triVertex, origEdge *triEdge, c triComparator) bool {
	if prev == nil || next == nil {
		return true
	}
	winding := -1
	if c.sweepLt(prev.point, next.point) {
		winding = 1
	}
	return winding != origEdge.winding
}

// strokeBoundary (stage 5d) displaces edges by half a pixel inward and outward along their normals, intersect to find
// new vertices (alpha 0 on the exterior, 255 on the interior), and build a new antialiased mesh from those vertices.
func (aa *aaTriangulator) strokeBoundary(boundary *triEdgeList, innerMesh *triVertexList, c triComparator) {
	// A boundary with fewer than 3 edges is degenerate.
	if boundary.head == nil || boundary.head.right == nil || boundary.head.right.right == nil {
		return
	}
	prevEdge := boundary.tail
	var prevV *triVertex
	if prevEdge.winding > 0 {
		prevV = prevEdge.top
	} else {
		prevV = prevEdge.bottom
	}
	prevNormal := aaGetEdgeNormal(prevEdge)
	const radius = 0.5
	prevInner := prevEdge.line
	prevInner.c -= radius
	prevOuter := prevEdge.line
	prevOuter.c += radius
	var innerVertices, outerVertices triVertexList
	innerInversion := true
	outerInversion := true
	for e := boundary.head; e != nil; e = e.right {
		var v *triVertex
		if e.winding > 0 {
			v = e.top
		} else {
			v = e.bottom
		}
		normal := aaGetEdgeNormal(e)
		inner := e.line
		inner.c -= radius
		outer := e.line
		outer.c += radius
		var innerPoint, outerPoint geom.Point
		if !prevEdge.line.nearParallel(e.line) && prevInner.intersect(inner, &innerPoint) &&
			prevOuter.intersect(outer, &outerPoint) {
			cosAngle := normal.Dot(prevNormal)
			if cosAngle < -aaCosMiterAngle {
				var nextV *triVertex
				if e.winding > 0 {
					nextV = e.bottom
				} else {
					nextV = e.top
				}

				// This is a pointy vertex whose angle is smaller than the threshold; miter it.
				bisector := makeTriLine(innerPoint, outerPoint)
				tangent := makeTriLine(v.point,
					v.point.Add(geom.Point{X: float32(bisector.a), Y: float32(bisector.b)}))
				if tangent.a == 0 && tangent.b == 0 {
					continue
				}
				tangent.normalize()
				innerTangent := tangent
				outerTangent := tangent
				innerTangent.c -= 0.5
				outerTangent.c += 0.5
				var innerPoint1, innerPoint2, outerPoint1, outerPoint2 geom.Point
				if prevNormal.Cross(normal) > 0 {
					// Miter inner points.
					if !innerTangent.intersect(prevInner, &innerPoint1) ||
						!innerTangent.intersect(inner, &innerPoint2) ||
						!outerTangent.intersect(bisector, &outerPoint) {
						continue
					}
					prevTangent := makeTriLine(prevV.point,
						prevV.point.Add(geom.Point{X: float32(prevOuter.a), Y: float32(prevOuter.b)}))
					nextTangent := makeTriLine(nextV.point,
						nextV.point.Add(geom.Point{X: float32(outer.a), Y: float32(outer.b)}))
					if prevTangent.dist(outerPoint) > 0 {
						bisector.intersect(prevTangent, &outerPoint)
					}
					if nextTangent.dist(outerPoint) < 0 {
						bisector.intersect(nextTangent, &outerPoint)
					}
					outerPoint1 = outerPoint
					outerPoint2 = outerPoint
				} else {
					// Miter outer points.
					if !outerTangent.intersect(prevOuter, &outerPoint1) ||
						!outerTangent.intersect(outer, &outerPoint2) {
						continue
					}
					prevTangent := makeTriLine(prevV.point,
						prevV.point.Add(geom.Point{X: float32(prevInner.a), Y: float32(prevInner.b)}))
					nextTangent := makeTriLine(nextV.point,
						nextV.point.Add(geom.Point{X: float32(inner.a), Y: float32(inner.b)}))
					if prevTangent.dist(innerPoint) > 0 {
						bisector.intersect(prevTangent, &innerPoint)
					}
					if nextTangent.dist(innerPoint) < 0 {
						bisector.intersect(nextTangent, &innerPoint)
					}
					innerPoint1 = innerPoint
					innerPoint2 = innerPoint
				}
				if !innerPoint1.IsFinite() || !innerPoint2.IsFinite() ||
					!outerPoint1.IsFinite() || !outerPoint2.IsFinite() {
					continue
				}
				innerVertex1 := &triVertex{point: innerPoint1, alpha: 255}
				innerVertex2 := &triVertex{point: innerPoint2, alpha: 255}
				outerVertex1 := &triVertex{point: outerPoint1, alpha: 0}
				outerVertex2 := &triVertex{point: outerPoint2, alpha: 0}
				innerVertex1.partner = outerVertex1
				innerVertex2.partner = outerVertex2
				outerVertex1.partner = innerVertex1
				outerVertex2.partner = innerVertex2
				if !aaInversion(innerVertices.tail, innerVertex1, prevEdge, c) {
					innerInversion = false
				}
				if !aaInversion(outerVertices.tail, outerVertex1, prevEdge, c) {
					outerInversion = false
				}
				innerVertices.append(innerVertex1)
				innerVertices.append(innerVertex2)
				outerVertices.append(outerVertex1)
				outerVertices.append(outerVertex2)
			} else {
				innerVertex := &triVertex{point: innerPoint, alpha: 255}
				outerVertex := &triVertex{point: outerPoint, alpha: 0}
				innerVertex.partner = outerVertex
				outerVertex.partner = innerVertex
				if !aaInversion(innerVertices.tail, innerVertex, prevEdge, c) {
					innerInversion = false
				}
				if !aaInversion(outerVertices.tail, outerVertex, prevEdge, c) {
					outerInversion = false
				}
				innerVertices.append(innerVertex)
				outerVertices.append(outerVertex)
			}
		}
		prevInner = inner
		prevOuter = outer
		prevV = v
		prevEdge = e
		prevNormal = normal
	}
	if !aaInversion(innerVertices.tail, innerVertices.head, prevEdge, c) {
		innerInversion = false
	}
	if !aaInversion(outerVertices.tail, outerVertices.head, prevEdge, c) {
		outerInversion = false
	}
	// Outer edges get 1 winding, and inner edges get -2 winding. This ensures that the interior is always filled (1 +
	// -2 = -1 for normal cases, 1 + 2 = 3 for thin features where the interior inverts). For total inversion cases the
	// shape has reversed handedness, so invert the winding so it is detected during collapseOverlapRegions().
	innerWinding := -2
	if innerInversion {
		innerWinding = 2
	}
	outerWinding := 1
	if outerInversion {
		outerWinding = -1
	}
	for v := innerVertices.head; v != nil && v.next != nil; v = v.next {
		aa.makeConnectingEdge(v, v.next, triEdgeTypeInner, c, innerWinding)
	}
	aa.makeConnectingEdge(innerVertices.tail, innerVertices.head, triEdgeTypeInner, c,
		innerWinding)
	for v := outerVertices.head; v != nil && v.next != nil; v = v.next {
		aa.makeConnectingEdge(v, v.next, triEdgeTypeOuter, c, outerWinding)
	}
	aa.makeConnectingEdge(outerVertices.tail, outerVertices.head, triEdgeTypeOuter, c,
		outerWinding)
	innerMesh.appendList(&innerVertices)
	aa.outerMesh.appendList(&outerVertices)
}

// extractBoundary walks connected edges starting from e, collecting them into boundary and disconnecting each as it's
// consumed, until the walk returns to its starting vertex.
func (aa *aaTriangulator) extractBoundary(boundary *triEdgeList, e *triEdge) {
	down := aa.applyFillType(e.winding)
	var start *triVertex
	if down {
		start = e.top
	} else {
		start = e.bottom
	}
	for {
		if down {
			e.winding = 1
		} else {
			e.winding = -1
		}
		var next *triEdge
		e.line.normalize()
		e.line = e.line.scaled(float64(e.winding))
		boundary.insertBetween(e, boundary.tail, nil)
		if down {
			// Find the outgoing edge in clockwise order.
			switch {
			case e.nextEdgeAbove != nil:
				next = e.nextEdgeAbove
				down = false
			case e.bottom.lastEdgeBelow != nil:
				next = e.bottom.lastEdgeBelow
				down = true
			case e.prevEdgeAbove != nil:
				next = e.prevEdgeAbove
				down = false
			}
		} else {
			// Find the outgoing edge in counter-clockwise order.
			switch {
			case e.prevEdgeBelow != nil:
				next = e.prevEdgeBelow
				down = true
			case e.top.firstEdgeAbove != nil:
				next = e.top.firstEdgeAbove
				down = false
			case e.nextEdgeBelow != nil:
				next = e.nextEdgeBelow
				down = true
			}
		}
		e.disconnect()
		e = next
		if e == nil {
			break
		}
		var cur *triVertex
		if down {
			cur = e.top
		} else {
			cur = e.bottom
		}
		if cur == start {
			break
		}
	}
}

// extractBoundaries (stage 5b) extracts boundaries from the mesh, simplify and stroke them into a new mesh.
func (aa *aaTriangulator) extractBoundaries(inMesh, innerVertices *triVertexList, c triComparator) {
	aa.removeNonBoundaryEdges(inMesh)
	for v := inMesh.head; v != nil; v = v.next {
		for v.firstEdgeBelow != nil {
			var boundary triEdgeList
			aa.extractBoundary(&boundary, v.firstEdgeBelow)
			aa.simplifyBoundary(&boundary, c)
			aa.strokeBoundary(&boundary, innerVertices, c)
		}
	}
}

// aaTessellate is the antialiasing tessellation pass: it extracts and strokes boundaries, merges coincident vertices,
// simplifies, collapses overlap regions, and re-tessellates if the mesh turned out to be complex.
func (aa *aaTriangulator) aaTessellate(mesh *triVertexList, c triComparator) (*triPoly, bool) {
	var innerMesh triVertexList
	aa.extractBoundaries(mesh, &innerMesh, c)
	triSortMesh(&innerMesh, c)
	triSortMesh(&aa.outerMesh, c)
	aa.mergeCoincidentVertices(&innerMesh, c)
	wasComplex := aa.mergeCoincidentVertices(&aa.outerMesh, c)
	result := aa.simplify(&innerMesh, c)
	if result == triSimplifyFailed {
		return nil, false
	}
	wasComplex = result == triSimplifyFoundSelfIntersection || wasComplex
	result = aa.simplify(&aa.outerMesh, c)
	if result == triSimplifyFailed {
		return nil, false
	}
	wasComplex = result == triSimplifyFoundSelfIntersection || wasComplex
	wasComplex = aa.collapseOverlapRegions(&innerMesh, c, true) || wasComplex
	wasComplex = aa.collapseOverlapRegions(&aa.outerMesh, c, false) || wasComplex
	if wasComplex {
		var aaMesh triVertexList
		aa.connectPartners(&aa.outerMesh, c)
		aa.connectPartners(&innerMesh, c)
		triSortedMerge(&innerMesh, &aa.outerMesh, &aaMesh, c)
		aa.mergeCoincidentVertices(&aaMesh, c)
		result = aa.simplify(&aaMesh, c)
		if result == triSimplifyFailed {
			return nil, false
		}
		aa.outerMesh.head = nil
		aa.outerMesh.tail = nil
		return aa.tessellate(&aaMesh)
	}
	return aa.tessellate(&innerMesh)
}

// polysToAATriangles emits stage 6's fill triangles plus the outer-mesh alpha-ramp quads.
func (aa *aaTriangulator) polysToAATriangles(polys *triPoly, vertexAllocator eagerVertexAllocator) int {
	count64 := triCountPoints(polys, path.FillWinding)
	// Count the points from the outer mesh.
	for v := aa.outerMesh.head; v != nil; v = v.next {
		for e := v.firstEdgeBelow; e != nil; e = e.nextEdgeBelow {
			count64 += 6
		}
	}
	if count64 == 0 || count64 > math.MaxInt32 {
		return 0
	}
	count := int(count64)

	const vertexStride = 8 + 4 // sizeof(point) + sizeof(float)
	verts := vertexAllocator.lockWriter(vertexStride, count)
	if verts == nil {
		return 0
	}

	w := triVertexWriter{data: verts, emitCoverage: true}
	aa.polysToTrianglesInto(polys, path.FillWinding, &w)
	// Emit the triangles from the outer mesh.
	for v := aa.outerMesh.head; v != nil; v = v.next {
		for e := v.firstEdgeBelow; e != nil; e = e.nextEdgeBelow {
			v0 := e.top
			v1 := e.bottom
			v2 := e.bottom.partner
			v3 := e.top.partner
			aa.emitTriangle(v0, v1, v2, 0 /*winding*/, &w)
			aa.emitTriangle(v0, v2, v3, 0 /*winding*/, &w)
		}
	}

	actualCount := w.off / vertexStride
	if actualCount > count {
		panic("AA triangulator emitted more vertices than counted")
	}
	vertexAllocator.unlock(actualCount)
	return actualCount
}
