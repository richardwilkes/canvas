// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The triangulator: converts a path into a set of non-overlapping triangles in six stages — (1) linearize the contours,
// (2) build a mesh of edges, (3) merge-sort the vertices in sweep order, (4) simplify the mesh by inserting vertices at
// edge intersections (Bentley-Ottmann with topology repair for float error), (5) tessellate the simplified mesh into
// monotone polygons (Fournier & Montuno), and (6) ear-clip the monotone polygons into a vertex buffer. The
// breadcrumb-triangle machinery (used by the inner-fan mode of the tessellation renderers) lives alongside it. Debug
// logging/wireframe lanes are not implemented.

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// triSide selects which side of a monotone polygon an edge belongs to.
type triSide uint8

const (
	triSideLeft triSide = iota
	triSideRight
)

// triEdgeType classifies an edge's role in AA mesh construction.
type triEdgeType uint8

const (
	triEdgeTypeInner triEdgeType = iota
	triEdgeTypeOuter
	triEdgeTypeConnector
)

// triDirection selects the sweep axis used to sort vertices.
type triDirection uint8

const (
	triDirectionVertical triDirection = iota
	triDirectionHorizontal
)

// triComparator orders points along the sweep direction.
type triComparator struct {
	direction triDirection
}

func triSweepLtHoriz(a, b geom.Point) bool {
	return a.X < b.X || (a.X == b.X && a.Y > b.Y)
}

func triSweepLtVert(a, b geom.Point) bool {
	return a.Y < b.Y || (a.Y == b.Y && a.X < b.X)
}

func (c triComparator) sweepLt(a, b geom.Point) bool {
	if c.direction == triDirectionHorizontal {
		return triSweepLtHoriz(a, b)
	}
	return triSweepLtVert(a, b)
}

// triVertex is one point in the mesh, doubling as a contour-list and sweep-order list node.
type triVertex struct {
	prev               *triVertex // Linked list of contours, then Y-sorted vertices.
	next               *triVertex
	firstEdgeAbove     *triEdge // Linked list of edges above this vertex.
	lastEdgeAbove      *triEdge
	firstEdgeBelow     *triEdge // Linked list of edges below this vertex.
	lastEdgeBelow      *triEdge
	leftEnclosingEdge  *triEdge   // Nearest edge in the AEL left of this vertex.
	rightEnclosingEdge *triEdge   // Nearest edge in the AEL right of this vertex.
	partner            *triVertex // Corresponding inner or outer vertex (for AA).
	point              geom.Point // Vertex position
	alpha              uint8
	synthetic          bool // Is this a synthetic vertex?
}

func (v *triVertex) isConnected() bool { return v.firstEdgeAbove != nil || v.firstEdgeBelow != nil }

// triVertexList is a doubly linked list of vertices, head/tail only.
type triVertexList struct {
	head, tail *triVertex
}

func (l *triVertexList) insert(v, prev, next *triVertex) {
	v.prev = prev
	v.next = next
	if prev != nil {
		prev.next = v
	} else {
		l.head = v
	}
	if next != nil {
		next.prev = v
	} else {
		l.tail = v
	}
}

func (l *triVertexList) append(v *triVertex)  { l.insert(v, l.tail, nil) }
func (l *triVertexList) prepend(v *triVertex) { l.insert(v, nil, l.head) }

func (l *triVertexList) appendList(list *triVertexList) {
	if list.head == nil {
		return
	}
	if l.tail != nil {
		l.tail.next = list.head
		list.head.prev = l.tail
	} else {
		l.head = list.head
	}
	l.tail = list.tail
}

func (l *triVertexList) remove(v *triVertex) {
	if v.prev != nil {
		v.prev.next = v.next
	} else {
		l.head = v.next
	}
	if v.next != nil {
		v.next.prev = v.prev
	} else {
		l.tail = v.prev
	}
	v.prev = nil
	v.next = nil
}

// triLine is a * x + b * y + c = 0 for all points (x, y) on the line. The coefficients are doubles to avoid
// catastrophic cancellation in the isLeftOf/isRightOf checks (correct in float for the degree-2 polynomial).
type triLine struct {
	a, b, c float64
}

func makeTriLine(p, q geom.Point) triLine {
	return triLine{
		a: float64(q.Y) - float64(p.Y),                           // a = dY
		b: float64(p.X) - float64(q.X),                           // b = -dX
		c: float64(p.Y)*float64(q.X) - float64(p.X)*float64(q.Y), // c = cross(q, p)
	}
}

func (l triLine) dist(p geom.Point) float64 {
	return l.a*float64(p.X) + l.b*float64(p.Y) + l.c
}

func (l triLine) scaled(v float64) triLine { return triLine{a: l.a * v, b: l.b * v, c: l.c * v} }

func (l triLine) magSq() float64 { return l.a*l.a + l.b*l.b }

func (l *triLine) normalize() {
	length := math.Sqrt(l.magSq())
	if length == 0 {
		return
	}
	scale := 1.0 / length
	l.a *= scale
	l.b *= scale
	l.c *= scale
}

func (l triLine) nearParallel(o triLine) bool {
	return math.Abs(o.a-l.a) < 0.00001 && math.Abs(o.b-l.b) < 0.00001
}

// triRoundQuarterPixel rounds to nearest quarter-pixel (used for screenspace tessellation).
func triRoundQuarterPixel(p *geom.Point) {
	p.X = geom.RoundToScalar(p.X*4.0) * 0.25
	p.Y = geom.RoundToScalar(p.Y*4.0) * 0.25
}

// doubleToClampedScalar clamps large values to what is finitely representable when cast back to a float, and flushes
// near-zero values (below 16 * FLT_MIN) to zero to protect against denormals and ill-conditioned intermediates.
func doubleToClampedScalar(d float64) float32 {
	const maxLimit = float64(math.MaxFloat32)
	const nearZeroLimit = 16 * float64(1.17549435082228750797e-38) // 16 * FLT_MIN (normal)
	if math.Abs(d) < nearZeroLimit {
		d = 0
	}
	return float32(math.Max(-maxLimit, math.Min(d, maxLimit)))
}

// intersect computes the intersection of two (infinite) lines.
func (l triLine) intersect(other triLine, point *geom.Point) bool {
	denom := l.a*other.b - l.b*other.a
	if denom == 0 {
		return false
	}
	scale := 1.0 / denom
	point.X = doubleToClampedScalar((l.b*other.c - other.b*l.c) * scale)
	point.Y = doubleToClampedScalar((other.a*l.c - l.a*other.c) * scale)
	triRoundQuarterPixel(point)
	return point.IsFinite()
}

// triIlogb returns the base-2 exponent of x, saturated to 0 for |x| < 1 (the callers only care about very large
// coordinates).
func triIlogb(x float32) int {
	if geom.ScalarAbs(x) < 1 {
		return 0
	}
	return math.Ilogb(float64(x))
}

// edgeLineNeedsRecursion reports whether the edge's vertices differ by many orders of magnitude, meaning the computed
// line equation has significant error in its distance and intersection tests, so long edges are recursively subdivided
// (a binary search) for a more accurate intersection test.
func edgeLineNeedsRecursion(p0, p1 geom.Point) bool {
	expDiffX := triIlogb(p0.X) - triIlogb(p1.X)
	if expDiffX < 0 {
		expDiffX = -expDiffX
	}
	expDiffY := triIlogb(p0.Y) - triIlogb(p1.Y)
	if expDiffY < 0 {
		expDiffY = -expDiffY
	}
	// Differ by more than 2^20, or roughly a factor of one million.
	return expDiffX > 20 || expDiffY > 20
}

// recursiveEdgeIntersect finds the intersection of segments [u0,u1] and [v0,v1], recursively subdividing when needed
// for accuracy (see edgeLineNeedsRecursion).
func recursiveEdgeIntersect(u triLine, u0, u1 geom.Point, v triLine, v0, v1 geom.Point, p *geom.Point, s, t *float64) bool {
	// First check if the bounding boxes of [u0,u1] and [v0,v1] intersect. If they do not, the two segments cannot
	// intersect in their domain (even if the lines themselves might). A geom.Rect intersection test is deliberately not
	// used: the vertices are unsorted, and horizontal/vertical lines appear as empty rects, which would never
	// "intersect".
	if math.Min(float64(u0.X), float64(u1.X)) > math.Max(float64(v0.X), float64(v1.X)) ||
		math.Max(float64(u0.X), float64(u1.X)) < math.Min(float64(v0.X), float64(v1.X)) ||
		math.Min(float64(u0.Y), float64(u1.Y)) > math.Max(float64(v0.Y), float64(v1.Y)) ||
		math.Max(float64(u0.Y), float64(u1.Y)) < math.Min(float64(v0.Y), float64(v1.Y)) {
		return false
	}

	// Compute the intersection based on the current segment vertices; if an intersection is found but the vertices
	// differ too much in magnitude, recurse using the midpoint of the segment to reject false positives. False
	// negatives are not currently avoided.
	denom := u.a*v.b - u.b*v.a
	if denom == 0 {
		return false
	}
	dx := float64(v0.X) - float64(u0.X)
	dy := float64(v0.Y) - float64(u0.Y)
	sNumer := dy*v.b + dx*v.a
	tNumer := dy*u.b + dx*u.a
	// If (sNumer / denom) or (tNumer / denom) is not in [0..1], exit early: this saves the divide below unless
	// absolutely necessary.
	if denom > 0 {
		if sNumer < 0 || sNumer > denom || tNumer < 0 || tNumer > denom {
			return false
		}
	} else if sNumer > 0 || sNumer < denom || tNumer > 0 || tNumer < denom {
		return false
	}

	*s = sNumer / denom
	*t = tNumer / denom

	uNeedsSplit := edgeLineNeedsRecursion(u0, u1)
	vNeedsSplit := edgeLineNeedsRecursion(v0, v1)
	if !uNeedsSplit && !vNeedsSplit {
		p.X = doubleToClampedScalar(float64(u0.X) - (*s)*u.b)
		p.Y = doubleToClampedScalar(float64(u0.Y) + (*s)*u.a)
		return true
	}
	sScale, sShift := 1.0, 0.0
	tScale, tShift := 1.0, 0.0

	if uNeedsSplit {
		uM := geom.Point{
			X: float32(0.5*float64(u0.X) + 0.5*float64(u1.X)),
			Y: float32(0.5*float64(u0.Y) + 0.5*float64(u1.Y)),
		}
		sScale = 0.5
		if *s >= 0.5 {
			u0 = uM
			sShift = 0.5
		} else {
			u1 = uM
		}
	}
	if vNeedsSplit {
		vM := geom.Point{
			X: float32(0.5*float64(v0.X) + 0.5*float64(v1.X)),
			Y: float32(0.5*float64(v0.Y) + 0.5*float64(v1.Y)),
		}
		tScale = 0.5
		if *t >= 0.5 {
			v0 = vM
			tShift = 0.5
		} else {
			v1 = vM
		}
	}

	// Just recompute both lines, even if only one was split; this is already a slow path.
	if recursiveEdgeIntersect(makeTriLine(u0, u1), u0, u1, makeTriLine(v0, v1), v0, v1, p, s, t) {
		// Adjust s and t back to the full range.
		*s = sScale*(*s) + sShift
		*t = tScale*(*t) + tShift
		return true
	}
	// False positive.
	return false
}

// triEdge joins a top vertex to a bottom vertex (in sweep order).
type triEdge struct {
	prevEdgeBelow   *triEdge // The linked list in the top vertex's "edges below".
	leftPoly        *triPoly // The poly to the left of this edge, if any.
	bottom          *triVertex
	nextEdgeBelow   *triEdge
	left            *triEdge // The linked list of edges in the active edge list.
	right           *triEdge
	prevEdgeAbove   *triEdge // The linked list in the bottom vertex's "edges above".
	nextEdgeAbove   *triEdge
	top             *triVertex
	rightPolyNext   *triEdge
	rightPolyPrev   *triEdge
	rightPoly       *triPoly // The poly to the right of this edge, if any.
	leftPolyPrev    *triEdge
	leftPolyNext    *triEdge
	line            triLine
	winding         int // 1 == edge goes downward; -1 = edge goes upward.
	edgeType        triEdgeType
	usedInLeftPoly  bool
	usedInRightPoly bool
}

func makeTriEdge(top, bottom *triVertex, winding int, edgeType triEdgeType) *triEdge {
	return &triEdge{
		winding:  winding,
		top:      top,
		bottom:   bottom,
		edgeType: edgeType,
		line:     makeTriLine(top.point, bottom.point),
	}
}

// dist coerces points coincident with the vertices to dist = 0, since converting from a double intersection point back
// to float storage might construct a point no longer on the ideal line.
func (e *triEdge) dist(p geom.Point) float64 {
	if p == e.top.point || p == e.bottom.point {
		return 0
	}
	return e.line.dist(p)
}

func (e *triEdge) isRightOf(v *triVertex) bool { return e.dist(v.point) < 0 }
func (e *triEdge) isLeftOf(v *triVertex) bool  { return e.dist(v.point) > 0 }

func (e *triEdge) recompute() { e.line = makeTriLine(e.top.point, e.bottom.point) }

func (e *triEdge) hasTopAndBottom() bool { return e.top != nil && e.bottom != nil }

// intersect finds the intersection point of e and other, and (if requested) the interpolated alpha at that point.
func (e *triEdge) intersect(other *triEdge, p *geom.Point, alpha *uint8) bool {
	if e.top == other.top || e.bottom == other.bottom || e.top == other.bottom ||
		e.bottom == other.top {
		// If the two edges share a vertex by construction, they have already been split and are no longer considered
		// "intersecting".
		return false
	}
	var s, t float64 // needed to interpolate vertex alpha
	if !recursiveEdgeIntersect(e.line, e.top.point, e.bottom.point, other.line, other.top.point,
		other.bottom.point, p, &s, &t) {
		return false
	}
	if alpha != nil {
		switch {
		case e.edgeType == triEdgeTypeInner || other.edgeType == triEdgeTypeInner:
			// If the intersection is on any interior edge it needs to stay fully opaque, or later triangulation could
			// leech transparency into the inner fill region.
			*alpha = 255
		case e.edgeType == triEdgeTypeOuter && other.edgeType == triEdgeTypeOuter:
			// Trivially fully transparent: by construction it is on the outer edge.
			*alpha = 0
		default:
			// Two connectors crossing, or a connector crossing an outer edge: take the max interpolated alpha (double
			// math truncated to byte).
			*alpha = uint8(math.Max((1.0-s)*float64(e.top.alpha)+s*float64(e.bottom.alpha),
				(1.0-t)*float64(other.top.alpha)+t*float64(other.bottom.alpha)))
		}
	}
	return true
}

// insertAbove inserts this edge into v's edges-above list.
func (e *triEdge) insertAbove(v *triVertex, c triComparator) {
	if e.top.point == e.bottom.point || c.sweepLt(e.bottom.point, e.top.point) {
		return
	}
	var prev *triEdge
	next := v.firstEdgeAbove
	for next != nil {
		if next.isRightOf(e.top) {
			break
		}
		prev = next
		next = next.nextEdgeAbove
	}
	e.prevEdgeAbove = prev
	e.nextEdgeAbove = next
	if prev != nil {
		prev.nextEdgeAbove = e
	} else {
		v.firstEdgeAbove = e
	}
	if next != nil {
		next.prevEdgeAbove = e
	} else {
		v.lastEdgeAbove = e
	}
}

// insertBelow inserts this edge into v's edges-below list.
func (e *triEdge) insertBelow(v *triVertex, c triComparator) {
	if e.top.point == e.bottom.point || c.sweepLt(e.bottom.point, e.top.point) {
		return
	}
	var prev *triEdge
	next := v.firstEdgeBelow
	for next != nil {
		if next.isRightOf(e.bottom) {
			break
		}
		prev = next
		next = next.nextEdgeBelow
	}
	e.prevEdgeBelow = prev
	e.nextEdgeBelow = next
	if prev != nil {
		prev.nextEdgeBelow = e
	} else {
		v.firstEdgeBelow = e
	}
	if next != nil {
		next.prevEdgeBelow = e
	} else {
		v.lastEdgeBelow = e
	}
}

// removeEdgeAbove unlinks e from its bottom vertex's edges-above list.
func removeEdgeAbove(e *triEdge) {
	v := e.bottom
	if e.prevEdgeAbove != nil {
		e.prevEdgeAbove.nextEdgeAbove = e.nextEdgeAbove
	} else {
		v.firstEdgeAbove = e.nextEdgeAbove
	}
	if e.nextEdgeAbove != nil {
		e.nextEdgeAbove.prevEdgeAbove = e.prevEdgeAbove
	} else {
		v.lastEdgeAbove = e.prevEdgeAbove
	}
	e.prevEdgeAbove = nil
	e.nextEdgeAbove = nil
}

// removeEdgeBelow unlinks e from its top vertex's edges-below list.
func removeEdgeBelow(e *triEdge) {
	v := e.top
	if e.prevEdgeBelow != nil {
		e.prevEdgeBelow.nextEdgeBelow = e.nextEdgeBelow
	} else {
		v.firstEdgeBelow = e.nextEdgeBelow
	}
	if e.nextEdgeBelow != nil {
		e.nextEdgeBelow.prevEdgeBelow = e.prevEdgeBelow
	} else {
		v.lastEdgeBelow = e.prevEdgeBelow
	}
	e.prevEdgeBelow = nil
	e.nextEdgeBelow = nil
}

func (e *triEdge) disconnect() {
	removeEdgeAbove(e)
	removeEdgeBelow(e)
}

// triEdgeList is the active edge list: edges currently crossing the sweep line.
type triEdgeList struct {
	head, tail *triEdge
}

func (l *triEdgeList) insertBetween(edge, prev, next *triEdge) {
	edge.left = prev
	edge.right = next
	if prev != nil {
		prev.right = edge
	} else {
		l.head = edge
	}
	if next != nil {
		next.left = edge
	} else {
		l.tail = edge
	}
}

// insert adds edge to the list after prev (or at the head, if prev is nil). Returns false if the edge is already
// present.
func (l *triEdgeList) insert(edge, prev *triEdge) bool {
	if l.contains(edge) {
		return false
	}
	var next *triEdge
	if prev != nil {
		next = prev.right
	} else {
		next = l.head
	}
	l.insertBetween(edge, prev, next)
	return true
}

func (l *triEdgeList) remove(edge *triEdge) bool {
	if !l.contains(edge) {
		return false
	}
	if edge.left != nil {
		edge.left.right = edge.right
	} else {
		l.head = edge.right
	}
	if edge.right != nil {
		edge.right.left = edge.left
	} else {
		l.tail = edge.left
	}
	edge.left = nil
	edge.right = nil
	return true
}

func (l *triEdgeList) contains(edge *triEdge) bool {
	return edge.left != nil || edge.right != nil || l.head == edge
}

// triMonotonePoly is one monotone polygon within a triPoly, built from a chain of edges on one side.
type triMonotonePoly struct {
	firstEdge *triEdge
	lastEdge  *triEdge
	prev      *triMonotonePoly
	next      *triMonotonePoly
	winding   int
	side      triSide
}

// addEdge appends edge to this monotone poly's chain on its configured side.
func (m *triMonotonePoly) addEdge(edge *triEdge) {
	if m.side == triSideRight {
		edge.rightPolyPrev = m.lastEdge
		edge.rightPolyNext = nil
		if m.lastEdge != nil {
			m.lastEdge.rightPolyNext = edge
		} else {
			m.firstEdge = edge
		}
		m.lastEdge = edge
		edge.usedInRightPoly = true
	} else {
		edge.leftPolyPrev = m.lastEdge
		edge.leftPolyNext = nil
		if m.lastEdge != nil {
			m.lastEdge.leftPolyNext = edge
		} else {
			m.firstEdge = edge
		}
		m.lastEdge = edge
		edge.usedInLeftPoly = true
	}
}

// triPoly is a polygon built during tessellation, as a chain of monotone-poly sections.
type triPoly struct {
	firstVertex *triVertex
	head        *triMonotonePoly
	tail        *triMonotonePoly
	next        *triPoly
	partner     *triPoly
	winding     int
	count       int
}

func (p *triPoly) lastVertex() *triVertex {
	if p.tail != nil {
		return p.tail.lastEdge.bottom
	}
	return p.firstVertex
}

// addEdge appends e to this poly on the given side, growing or splitting its monotone-poly chain as needed.
func (p *triPoly) addEdge(e *triEdge, side triSide, tri *triangulator) *triPoly {
	partner := p.partner
	poly := p
	if side == triSideRight {
		if e.usedInRightPoly {
			return p
		}
	} else if e.usedInLeftPoly {
		return p
	}
	if partner != nil {
		p.partner = nil
		partner.partner = nil
	}
	switch {
	case p.tail == nil:
		p.head = tri.allocateMonotonePoly(e, side, p.winding)
		p.tail = p.head
		p.count += 2
	case e.bottom == p.tail.lastEdge.bottom:
		return poly
	case side == p.tail.side:
		p.tail.addEdge(e)
		p.count++
	default:
		e = tri.allocateEdge(p.tail.lastEdge.bottom, e.bottom, 1, triEdgeTypeInner)
		p.tail.addEdge(e)
		p.count++
		if partner != nil {
			partner.addEdge(e, side, tri)
			poly = partner
		} else {
			m := tri.allocateMonotonePoly(e, side, p.winding)
			m.prev = p.tail
			p.tail.next = m
			p.tail = m
		}
	}
	return poly
}

// triVertexWriter sequentially emits little-endian positions (and optional coverage) into locked vertex space.
type triVertexWriter struct {
	data         []byte
	off          int
	emitCoverage bool
}

func (w *triVertexWriter) emitVertex(v *triVertex) {
	binary.LittleEndian.PutUint32(w.data[w.off:], math.Float32bits(v.point.X))
	binary.LittleEndian.PutUint32(w.data[w.off+4:], math.Float32bits(v.point.Y))
	w.off += 8
	if w.emitCoverage {
		// Normalize the byte alpha to [0,1].
		binary.LittleEndian.PutUint32(w.data[w.off:], math.Float32bits(float32(v.alpha)*(1.0/255)))
		w.off += 4
	}
}

func (w *triVertexWriter) emitTriangle(v0, v1, v2 *triVertex) {
	w.emitVertex(v0)
	w.emitVertex(v1)
	w.emitVertex(v2)
}

// triApplyFillType reports whether winding should be filled under fillType.
func triApplyFillType(fillType path.FillType, winding int) bool {
	switch fillType {
	case path.FillWinding:
		return winding != 0
	case path.FillEvenOdd:
		return winding&1 != 0
	case path.FillInverseWinding:
		return winding == 1
	case path.FillInverseEvenOdd:
		return winding&1 == 1
	default:
		panic("unknown fill type")
	}
}

func triApplyFillTypePoly(fillType path.FillType, poly *triPoly) bool {
	return poly != nil && triApplyFillType(fillType, poly.winding)
}

// breadcrumbTriangleList holds the razor-thin triangles that glue T-junctions between a path's outer curves and its
// inner polygon triangulation (used by the inner-fan mode of the tessellation renderers).
type breadcrumbTriangleList struct {
	head, tail *breadcrumbNode
	count      int
}

type breadcrumbNode struct {
	next *breadcrumbNode
	pts  [3]geom.Point
}

func (l *breadcrumbTriangleList) append(a, b, c geom.Point, winding int) {
	if a == b || a == c || b == c || winding == 0 {
		return
	}
	if winding < 0 {
		a, b = b, a
		winding = -winding
	}
	for i := 0; i < winding; i++ {
		n := &breadcrumbNode{pts: [3]geom.Point{a, b, c}}
		if l.tail != nil {
			l.tail.next = n
		} else {
			l.head = n
		}
		l.tail = n
	}
	l.count += winding
}

func (l *breadcrumbTriangleList) concat(list *breadcrumbTriangleList) {
	if list.head != nil {
		if l.tail != nil {
			l.tail.next = list.head
		} else {
			l.head = list.head
		}
		l.tail = list.tail
		l.count += list.count
		list.head = nil
		list.tail = nil
		list.count = 0
	}
}

// triangulator converts a path into non-overlapping triangles (see the file comment above).
type triangulator struct {
	path *path.Path
	// tessellateOverride lets a specialized triangulator (e.g. an AA-aware one) replace the base tessellate()
	// implementation; nil selects the base implementation.
	tessellateOverride       func(vertices *triVertexList, c triComparator) (*triPoly, bool)
	breadcrumbList           breadcrumbTriangleList
	numMonotonePolys         int
	numEdges                 int
	mergeCollinearStackCount int
	// Internal control knobs.
	roundVerticesToQuarterPixel bool
	emitCoverage                bool
	preserveCollinearVertices   bool
	collectBreadcrumbTriangles  bool
}

// triSimplifyResult is the outcome of a mesh-simplification pass.
type triSimplifyResult uint8

const (
	triSimplifyFailed triSimplifyResult = iota
	triSimplifyAlreadySimple
	triSimplifyFoundSelfIntersection
)

// triBoolFail is a tri-state boolean that also records failure.
type triBoolFail uint8

const (
	triBoolFalse triBoolFail = iota
	triBoolTrue
	triBoolFailed
)

// triPathToTriangles triangulates p into non-overlapping triangles. Returns the emitted vertex count.
func triPathToTriangles(p *path.Path, tolerance float32, clipBounds geom.Rect, vertexAllocator eagerVertexAllocator, isLinear *bool) int {
	if !p.IsFinite() {
		return 0
	}
	tri := &triangulator{path: p}
	polys, success := tri.pathToPolys(tolerance, clipBounds, isLinear)
	if !success {
		return 0
	}
	return tri.polysToTriangles(polys, vertexAllocator)
}

// emitMonotonePoly ear-clips one monotone polygon (stage 6).
func (tri *triangulator) emitMonotonePoly(monotonePoly *triMonotonePoly, w *triVertexWriter) {
	e := monotonePoly.firstEdge
	var vertices triVertexList
	vertices.append(e.top)
	count := 1
	for e != nil {
		if monotonePoly.side == triSideRight {
			vertices.append(e.bottom)
			e = e.rightPolyNext
		} else {
			vertices.prepend(e.bottom)
			e = e.leftPolyNext
		}
		count++
	}
	first := vertices.head
	v := first.next
	for v != vertices.tail {
		prev := v.prev
		curr := v
		next := v.next
		if count == 3 {
			tri.emitTriangle(prev, curr, next, monotonePoly.winding, w)
			return
		}
		ax := float64(curr.point.X) - float64(prev.point.X)
		ay := float64(curr.point.Y) - float64(prev.point.Y)
		bx := float64(next.point.X) - float64(curr.point.X)
		by := float64(next.point.Y) - float64(curr.point.Y)
		if ax*by-ay*bx >= 0 {
			tri.emitTriangle(prev, curr, next, monotonePoly.winding, w)
			v.prev.next = v.next
			v.next.prev = v.prev
			count--
			if v.prev == first {
				v = v.next
			} else {
				v = v.prev
			}
		} else {
			v = v.next
		}
	}
}

// emitTriangle writes one triangle (winding-corrected), and records a breadcrumb triangle for any extra winding count
// beyond +/-1.
func (tri *triangulator) emitTriangle(prev, curr, next *triVertex, winding int, w *triVertexWriter) {
	if winding > 0 {
		// Ensure our triangles always wind in the same direction as if the path had been triangulated as a simple fan
		// (a la red book).
		prev, next = next, prev
	}
	if tri.collectBreadcrumbTriangles && (winding > 1 || winding < -1) &&
		tri.path.FillType() == path.FillWinding {
		// The first winding count will come from the actual triangle we emit. The remaining counts come from the
		// breadcrumb triangle.
		abs := winding
		if abs < 0 {
			abs = -abs
		}
		tri.breadcrumbList.append(prev.point, curr.point, next.point, abs-1)
	}
	w.emitTriangle(prev, curr, next)
}

// emitPoly ear-clips every monotone section of poly.
func (tri *triangulator) emitPoly(poly *triPoly, w *triVertexWriter) {
	if poly.count < 3 {
		return
	}
	for m := poly.head; m != nil; m = m.next {
		tri.emitMonotonePoly(m, w)
	}
}

func triCoincident(a, b geom.Point) bool { return a == b }

// makePoly allocates a new poly starting at v and prepends it to the *head list.
func (tri *triangulator) makePoly(head **triPoly, v *triVertex, winding int) *triPoly {
	poly := &triPoly{firstVertex: v, winding: winding}
	poly.next = *head
	*head = poly
	return poly
}

// appendPointToContour appends p as a new fully-opaque vertex onto contour.
func (tri *triangulator) appendPointToContour(p geom.Point, contour *triVertexList) {
	contour.append(&triVertex{point: p, alpha: 255})
}

// triQuadCoeff is a quadratic Bezier in power-basis form, for appendQuadraticToContour.
type triQuadCoeff struct {
	a, b, c geom.Point
}

func makeTriQuadCoeff(pts *[3]geom.Point) triQuadCoeff {
	var q triQuadCoeff
	q.c = pts[0]
	q.b = pts[1].Sub(pts[0]).Scaled(2)
	q.a = pts[2].Sub(pts[1]).Sub(pts[1].Sub(pts[0])) // P2 - 2*P1 + P0
	return q
}

func (q triQuadCoeff) eval(t float32) geom.Point {
	// (A*t + B)*t + C
	return geom.Point{
		X: (q.a.X*t+q.b.X)*t + q.c.X,
		Y: (q.a.Y*t+q.b.Y)*t + q.c.Y,
	}
}

// triQuadErrorAt estimates a quadratic's flatness error at parameter t over a step of size u.
func triQuadErrorAt(pts *[3]geom.Point, t, u float32) float32 {
	quad := makeTriQuadCoeff(pts)
	p0 := quad.eval(t - 0.5*u)
	mid := quad.eval(t)
	p1 := quad.eval(t + 0.5*u)
	if !p0.IsFinite() || !mid.IsFinite() || !p1.IsFinite() {
		return 0
	}
	return distanceToLineSegmentBetweenSqd(mid, p0, p1)
}

// appendQuadraticToContour linearizes a quadratic Bezier into contour to within toleranceSqd.
func (tri *triangulator) appendQuadraticToContour(pts *[3]geom.Point, toleranceSqd float32, contour *triVertexList) {
	quad := makeTriQuadCoeff(pts)
	aa := quad.a.X*quad.a.X + quad.a.Y*quad.a.Y // fA * fA, lanes summed below
	denom := 2.0 * aa
	ab := quad.a.X*quad.b.X + quad.a.Y*quad.b.Y
	t := float32(0)
	if denom != 0 {
		t = -ab / denom
	}
	nPoints := 1
	u := float32(1)
	// Test possible subdivision values only at the point of maximum curvature. If it passes the flatness metric there,
	// it will pass everywhere.
	for nPoints < pathUtilsMaxPointsPerCurve {
		u = 1.0 / float32(nPoints)
		if triQuadErrorAt(pts, t, u) < toleranceSqd {
			break
		}
		nPoints++
	}
	for j := 1; j <= nPoints; j++ {
		tri.appendPointToContour(quad.eval(float32(j)*u), contour)
	}
}

// generateCubicPoints recursively linearizes a cubic Bezier into contour to within tolSqd (distinct from the
// generateCubicPoints in pathutils.go, which fills a fixed-size slice).
func (tri *triangulator) generateCubicPoints(p0, p1, p2, p3 geom.Point, tolSqd float32, contour *triVertexList, pointsLeft int) {
	d1 := distanceToLineSegmentBetweenSqd(p1, p0, p3)
	d2 := distanceToLineSegmentBetweenSqd(p2, p0, p3)
	if pointsLeft < 2 || (d1 < tolSqd && d2 < tolSqd) || !geom.IsFinite(d1, d2) {
		tri.appendPointToContour(p3, contour)
		return
	}
	q := [3]geom.Point{
		{X: scalarAve(p0.X, p1.X), Y: scalarAve(p0.Y, p1.Y)},
		{X: scalarAve(p1.X, p2.X), Y: scalarAve(p1.Y, p2.Y)},
		{X: scalarAve(p2.X, p3.X), Y: scalarAve(p2.Y, p3.Y)},
	}
	r := [2]geom.Point{
		{X: scalarAve(q[0].X, q[1].X), Y: scalarAve(q[0].Y, q[1].Y)},
		{X: scalarAve(q[1].X, q[2].X), Y: scalarAve(q[1].Y, q[2].Y)},
	}
	s := geom.Point{X: scalarAve(r[0].X, r[1].X), Y: scalarAve(r[0].Y, r[1].Y)}
	pointsLeft >>= 1
	tri.generateCubicPoints(p0, q[0], r[0], s, tolSqd, contour, pointsLeft)
	tri.generateCubicPoints(s, r[1], q[2], p3, tolSqd, contour, pointsLeft)
}

// pathToContours linearizes the path into contours (stage 1).
func (tri *triangulator) pathToContours(tolerance float32, clipBounds geom.Rect, contours []triVertexList, isLinear *bool) {
	toleranceSqd := tolerance * tolerance
	*isLinear = true
	contourIdx := 0
	if tri.path.IsInverseFillType() {
		quad := clipBounds.ToQuad()
		for i := 3; i >= 0; i-- {
			tri.appendPointToContour(quad[i], &contours[0])
		}
		contourIdx++
	}
	it := path.NewIter(tri.path, false)
	for {
		verb, pts, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbConic:
			*isLinear = false
			if toleranceSqd == 0 {
				tri.appendPointToContour(pts[2], &contours[contourIdx])
				break
			}
			// toleranceSqd is passed as the conic-to-quads conversion tolerance here.
			conic := geom.MakeConic(pts[0], pts[1], pts[2], it.ConicWeight())
			pow2 := conic.ComputeQuadPOW2(toleranceSqd)
			quadStorage := make([]geom.Point, 1+2*(1<<pow2))
			count := conic.ChopIntoQuadsPOW2(quadStorage, pow2)
			for i := 0; i < count; i++ {
				quadPts := [3]geom.Point{quadStorage[i*2], quadStorage[i*2+1], quadStorage[i*2+2]}
				tri.appendQuadraticToContour(&quadPts, toleranceSqd, &contours[contourIdx])
			}
		case path.VerbMove:
			if contours[contourIdx].head != nil {
				contourIdx++
			}
			tri.appendPointToContour(pts[0], &contours[contourIdx])
		case path.VerbLine:
			tri.appendPointToContour(pts[1], &contours[contourIdx])
		case path.VerbQuad:
			*isLinear = false
			if toleranceSqd == 0 {
				tri.appendPointToContour(pts[2], &contours[contourIdx])
				break
			}
			quadPts := [3]geom.Point{pts[0], pts[1], pts[2]}
			tri.appendQuadraticToContour(&quadPts, toleranceSqd, &contours[contourIdx])
		case path.VerbCubic:
			*isLinear = false
			if toleranceSqd == 0 {
				tri.appendPointToContour(pts[3], &contours[contourIdx])
				break
			}
			pointsLeft := cubicPointCount([4]geom.Point{pts[0], pts[1], pts[2], pts[3]}, tolerance)
			tri.generateCubicPoints(pts[0], pts[1], pts[2], pts[3], toleranceSqd,
				&contours[contourIdx], pointsLeft)
		case path.VerbClose:
		}
	}
}

func (tri *triangulator) applyFillType(winding int) bool {
	return triApplyFillType(tri.path.FillType(), winding)
}

func (tri *triangulator) allocateMonotonePoly(edge *triEdge, side triSide, winding int) *triMonotonePoly {
	tri.numMonotonePolys++
	m := &triMonotonePoly{side: side, winding: winding}
	m.addEdge(edge)
	return m
}

func (tri *triangulator) allocateEdge(top, bottom *triVertex, winding int, edgeType triEdgeType) *triEdge {
	tri.numEdges++
	return makeTriEdge(top, bottom, winding, edgeType)
}

// makeEdge builds a new edge between prev and next, in sweep-order (top/bottom, winding).
func (tri *triangulator) makeEdge(prev, next *triVertex, edgeType triEdgeType, c triComparator) *triEdge {
	winding := -1
	if c.sweepLt(prev.point, next.point) {
		winding = 1
	}
	top := prev
	bottom := next
	if winding < 0 {
		top, bottom = next, prev
	}
	return tri.allocateEdge(top, bottom, winding, edgeType)
}

// triFindEnclosingEdges returns the active edges immediately left and right of v.
func triFindEnclosingEdges(v *triVertex, edges *triEdgeList) (left, right *triEdge) {
	if v.firstEdgeAbove != nil && v.lastEdgeAbove != nil {
		return v.firstEdgeAbove.left, v.lastEdgeAbove.right
	}
	var next *triEdge
	prev := edges.tail
	for prev != nil {
		if prev.isLeftOf(v) {
			break
		}
		next = prev
		prev = prev.left
	}
	return prev, next
}

// triRewind removes and re-inserts active edges from the current vertex back to dst, restoring active-edge-list order
// after topology repairs. A broken walk (nil prev) is treated as the same failure lane used for other inconsistencies.
func triRewind(activeEdges *triEdgeList, current **triVertex, dst *triVertex, c triComparator) bool {
	if current == nil || *current == dst || c.sweepLt((*current).point, dst.point) {
		return true
	}
	v := *current
	for v != dst {
		v = v.prev
		if v == nil {
			return false
		}
		for e := v.firstEdgeBelow; e != nil; e = e.nextEdgeBelow {
			if !activeEdges.remove(e) {
				return false
			}
		}
		leftEdge := v.leftEnclosingEdge
		for e := v.firstEdgeAbove; e != nil; e = e.nextEdgeAbove {
			if !activeEdges.insert(e, leftEdge) {
				return false
			}
			leftEdge = e
			top := e.top
			if top == nil ||
				(top.leftEnclosingEdge != nil && !top.leftEnclosingEdge.hasTopAndBottom()) ||
				(top.rightEnclosingEdge != nil && !top.rightEnclosingEdge.hasTopAndBottom()) {
				return false
			}
			if c.sweepLt(top.point, dst.point) &&
				((top.leftEnclosingEdge != nil && !top.leftEnclosingEdge.isLeftOf(top)) ||
					(top.rightEnclosingEdge != nil && !top.rightEnclosingEdge.isRightOf(top))) {
				dst = top
			}
		}
	}
	*current = v
	return true
}

// triRewindIfNecessary rewinds the active edge list if edge's ordering relative to its neighbors has become
// inconsistent.
func triRewindIfNecessary(edge *triEdge, activeEdges *triEdgeList, current **triVertex, c triComparator) bool {
	if activeEdges == nil || current == nil {
		return true
	}
	if edge == nil {
		return false
	}
	top := edge.top
	bottom := edge.bottom
	if edge.left != nil {
		leftTop := edge.left.top
		leftBottom := edge.left.bottom
		if leftTop != nil && leftBottom != nil {
			switch {
			case c.sweepLt(leftTop.point, top.point) && !edge.left.isLeftOf(top):
				if !triRewind(activeEdges, current, leftTop, c) {
					return false
				}
			case c.sweepLt(top.point, leftTop.point) && !edge.isRightOf(leftTop):
				if !triRewind(activeEdges, current, top, c) {
					return false
				}
			case c.sweepLt(bottom.point, leftBottom.point) && !edge.left.isLeftOf(bottom):
				if !triRewind(activeEdges, current, leftTop, c) {
					return false
				}
			case c.sweepLt(leftBottom.point, bottom.point) && !edge.isRightOf(leftBottom):
				if !triRewind(activeEdges, current, top, c) {
					return false
				}
			}
		}
	}
	if edge.right != nil {
		rightTop := edge.right.top
		rightBottom := edge.right.bottom
		if rightTop != nil && rightBottom != nil {
			switch {
			case c.sweepLt(rightTop.point, top.point) && !edge.right.isRightOf(top):
				if !triRewind(activeEdges, current, rightTop, c) {
					return false
				}
			case c.sweepLt(top.point, rightTop.point) && !edge.isLeftOf(rightTop):
				if !triRewind(activeEdges, current, top, c) {
					return false
				}
			case c.sweepLt(bottom.point, rightBottom.point) && !edge.right.isRightOf(bottom):
				if !triRewind(activeEdges, current, rightTop, c) {
					return false
				}
			case c.sweepLt(rightBottom.point, bottom.point) && !edge.isLeftOf(rightBottom):
				if !triRewind(activeEdges, current, top, c) {
					return false
				}
			}
		}
	}
	return true
}

// setTop moves edge's top vertex to v, recording a breadcrumb triangle for the removed span, then repairs topology.
func (tri *triangulator) setTop(edge *triEdge, v *triVertex, activeEdges *triEdgeList, current **triVertex, c triComparator) bool {
	removeEdgeBelow(edge)
	if tri.collectBreadcrumbTriangles {
		tri.breadcrumbList.append(edge.top.point, edge.bottom.point, v.point, edge.winding)
	}
	edge.top = v
	edge.recompute()
	edge.insertBelow(v, c)
	if !triRewindIfNecessary(edge, activeEdges, current, c) {
		return false
	}
	return tri.mergeCollinearEdges(edge, activeEdges, current, c)
}

// setBottom moves edge's bottom vertex to v, recording a breadcrumb triangle for the removed span, then repairs
// topology.
func (tri *triangulator) setBottom(edge *triEdge, v *triVertex, activeEdges *triEdgeList, current **triVertex, c triComparator) bool {
	removeEdgeAbove(edge)
	if tri.collectBreadcrumbTriangles {
		tri.breadcrumbList.append(edge.top.point, edge.bottom.point, v.point, edge.winding)
	}
	edge.bottom = v
	edge.recompute()
	edge.insertAbove(v, c)
	if !triRewindIfNecessary(edge, activeEdges, current, c) {
		return false
	}
	return tri.mergeCollinearEdges(edge, activeEdges, current, c)
}

// mergeEdgesAbove merges two adjacent, collinear edges. Coincident endpoints absorb one edge's winding into the other
// and fully disconnect the "zombie" edge (also removing it from the active edge list to prevent state corruption);
// overlapping endpoints shorten one edge in place.
func (tri *triangulator) mergeEdgesAbove(edge, other *triEdge, activeEdges *triEdgeList, current **triVertex, c triComparator) bool {
	if edge == nil || other == nil {
		return false
	}
	switch {
	case triCoincident(edge.top.point, other.top.point):
		if !triRewind(activeEdges, current, edge.top, c) {
			return false
		}
		other.winding += edge.winding
		edge.disconnect()
		if activeEdges != nil {
			activeEdges.remove(edge)
		}
		edge.top = nil
		edge.bottom = nil
	case c.sweepLt(edge.top.point, other.top.point):
		if !triRewind(activeEdges, current, edge.top, c) {
			return false
		}
		other.winding += edge.winding
		if !tri.setBottom(edge, other.top, activeEdges, current, c) {
			return false
		}
	default:
		if !triRewind(activeEdges, current, other.top, c) {
			return false
		}
		edge.winding += other.winding
		if !tri.setBottom(other, edge.top, activeEdges, current, c) {
			return false
		}
	}
	return true
}

// mergeEdgesBelow is mergeEdgesAbove's counterpart for shared bottom vertices.
func (tri *triangulator) mergeEdgesBelow(edge, other *triEdge, activeEdges *triEdgeList, current **triVertex, c triComparator) bool {
	if edge == nil || other == nil {
		return false
	}
	switch {
	case triCoincident(edge.bottom.point, other.bottom.point):
		if !triRewind(activeEdges, current, edge.top, c) {
			return false
		}
		other.winding += edge.winding
		edge.disconnect()
		if activeEdges != nil {
			activeEdges.remove(edge)
		}
		edge.top = nil
		edge.bottom = nil
	case c.sweepLt(edge.bottom.point, other.bottom.point):
		if !triRewind(activeEdges, current, other.top, c) {
			return false
		}
		edge.winding += other.winding
		if !tri.setTop(other, edge.bottom, activeEdges, current, c) {
			return false
		}
	default:
		if !triRewind(activeEdges, current, edge.top, c) {
			return false
		}
		other.winding += edge.winding
		if !tri.setTop(edge, other.bottom, activeEdges, current, c) {
			return false
		}
	}
	return true
}

func triTopCollinear(left, right *triEdge) bool {
	if left == nil || right == nil {
		return false
	}
	return left.top.point == right.top.point || !left.isLeftOf(right.top) ||
		!right.isRightOf(left.top)
}

func triBottomCollinear(left, right *triEdge) bool {
	if left == nil || right == nil {
		return false
	}
	return left.bottom.point == right.bottom.point || !left.isLeftOf(right.bottom) ||
		!right.isRightOf(left.bottom)
}

// triMaxMergeCollinearCalls is how deep a stack of mergeCollinearEdges() calls is accepted.
const triMaxMergeCollinearCalls = 64

// mergeCollinearEdges merges edge with its active-list neighbor if they are (nearly) collinear at the top or bottom.
func (tri *triangulator) mergeCollinearEdges(edge *triEdge, activeEdges *triEdgeList, current **triVertex, c triComparator) bool {
	tri.mergeCollinearStackCount++
	if tri.mergeCollinearStackCount > triMaxMergeCollinearCalls {
		return false
	}
	for {
		switch {
		case triTopCollinear(edge.prevEdgeAbove, edge):
			if !tri.mergeEdgesAbove(edge.prevEdgeAbove, edge, activeEdges, current, c) {
				return false
			}
		case triTopCollinear(edge, edge.nextEdgeAbove):
			if !tri.mergeEdgesAbove(edge.nextEdgeAbove, edge, activeEdges, current, c) {
				return false
			}
		case triBottomCollinear(edge.prevEdgeBelow, edge):
			if !tri.mergeEdgesBelow(edge.prevEdgeBelow, edge, activeEdges, current, c) {
				return false
			}
		case triBottomCollinear(edge, edge.nextEdgeBelow):
			if !tri.mergeEdgesBelow(edge.nextEdgeBelow, edge, activeEdges, current, c) {
				return false
			}
		default:
			return true
		}
	}
}

// splitEdge splits edge at v into two edges, repairing winding and topology.
func (tri *triangulator) splitEdge(edge *triEdge, v *triVertex, activeEdges *triEdgeList, current **triVertex, c triComparator) triBoolFail {
	if edge.top == nil || edge.bottom == nil || v == edge.top || v == edge.bottom {
		return triBoolFalse
	}
	var top, bottom *triVertex
	winding := edge.winding
	// Theoretically (and ideally) the edge between p0 and p1 is being split by v, with v "between" the segment end
	// points per c (p0 < v < p1). If v was clamped/rounded this may not hold.
	switch {
	case c.sweepLt(v.point, edge.top.point):
		// Actually "v < p0 < p1": update 'edge' to be v->p1 and add v->p0, flipping the winding on the new edge so it
		// winds as if it were p0->v.
		top = v
		bottom = edge.top
		winding *= -1
		if !tri.setTop(edge, v, activeEdges, current, c) {
			return triBoolFailed
		}
	case c.sweepLt(edge.bottom.point, v.point):
		// Actually "p0 < p1 < v": update 'edge' to be p0->v and add p1->v, flipping the winding on the new edge so it
		// winds as if it were v->p1.
		top = edge.bottom
		bottom = v
		winding *= -1
		if !tri.setBottom(edge, v, activeEdges, current, c) {
			return triBoolFailed
		}
	default:
		// The ideal case, "p0 < v < p1": update 'edge' to be p0->v and add v->p1. The original winding is valid for
		// both edges.
		top = v
		bottom = edge.bottom
		if !tri.setBottom(edge, v, activeEdges, current, c) {
			return triBoolFailed
		}
	}
	newEdge := tri.allocateEdge(top, bottom, winding, edge.edgeType)
	newEdge.insertBelow(top, c)
	newEdge.insertAbove(bottom, c)
	tri.mergeCollinearStackCount = 0
	if !tri.mergeCollinearEdges(newEdge, activeEdges, current, c) {
		return triBoolFailed
	}
	return triBoolTrue
}

// intersectEdgePair applies an explicit topology correction when the sided-ness checks (the source of ground truth)
// suggest an intersection that triEdge.intersect lacked the precision to find.
func (tri *triangulator) intersectEdgePair(left, right *triEdge, activeEdges *triEdgeList, current **triVertex, c triComparator) triBoolFail {
	if left.top == nil || left.bottom == nil || right.top == nil || right.bottom == nil {
		return triBoolFalse
	}
	if left.top == right.top || left.bottom == right.bottom {
		return triBoolFalse
	}

	var split *triEdge
	var splitAt *triVertex
	if c.sweepLt(left.top.point, right.top.point) {
		if !left.isLeftOf(right.top) {
			split = left
			splitAt = right.top
		}
	} else if !right.isRightOf(left.top) {
		split = right
		splitAt = left.top
	}
	if c.sweepLt(right.bottom.point, left.bottom.point) {
		if !left.isLeftOf(right.bottom) {
			split = left
			splitAt = right.bottom
		}
	} else if !right.isRightOf(left.bottom) {
		split = right
		splitAt = left.bottom
	}

	if split == nil {
		return triBoolFalse
	}

	// Rewind to the top of the edge that is "moving", since this topology correction can change the geometry of the
	// split edge.
	if !triRewind(activeEdges, current, split.top, c) {
		return triBoolFailed
	}
	return tri.splitEdge(split, splitAt, activeEdges, current, c)
}

// makeConnectingEdge builds and links a new edge between prev and next, scaling its winding.
func (tri *triangulator) makeConnectingEdge(prev, next *triVertex, edgeType triEdgeType, c triComparator, windingScale int) *triEdge {
	if prev == nil || next == nil || prev.point == next.point {
		return nil
	}
	edge := tri.makeEdge(prev, next, edgeType, c)
	edge.insertBelow(edge.top, c)
	edge.insertAbove(edge.bottom, c)
	edge.winding *= windingScale
	tri.mergeCollinearStackCount = 0
	tri.mergeCollinearEdges(edge, nil, nil, c)
	return edge
}

// mergeVertices merges src into dst, moving all of src's edges over and removing src from mesh.
func (tri *triangulator) mergeVertices(src, dst *triVertex, mesh *triVertexList, c triComparator) {
	if src.alpha > dst.alpha {
		dst.alpha = src.alpha
	}
	if src.partner != nil {
		src.partner.partner = dst
	}
	// setBottom()/setTop() call mergeCollinearEdges(), which can recurse, so the stack count is cleared before each
	// call.
	for src.firstEdgeAbove != nil {
		edge := src.firstEdgeAbove
		tri.mergeCollinearStackCount = 0
		_ = tri.setBottom(edge, dst, nil, nil, c)
	}
	for src.firstEdgeBelow != nil {
		edge := src.firstEdgeBelow
		tri.mergeCollinearStackCount = 0
		_ = tri.setTop(edge, dst, nil, nil, c)
	}
	mesh.remove(src)
	dst.synthetic = true
}

// makeSortedVertex builds a new vertex at p and inserts it into mesh in sweep order, searching outward from reference.
func (tri *triangulator) makeSortedVertex(p geom.Point, alpha uint8, mesh *triVertexList, reference *triVertex, c triComparator) *triVertex {
	prevV := reference
	for prevV != nil && c.sweepLt(p, prevV.point) {
		prevV = prevV.prev
	}
	var nextV *triVertex
	if prevV != nil {
		nextV = prevV.next
	} else {
		nextV = mesh.head
	}
	for nextV != nil && c.sweepLt(nextV.point, p) {
		prevV = nextV
		nextV = nextV.next
	}
	var v *triVertex
	switch {
	case prevV != nil && triCoincident(prevV.point, p):
		v = prevV
	case nextV != nil && triCoincident(nextV.point, p):
		v = nextV
	default:
		v = &triVertex{point: p, alpha: alpha}
		mesh.insert(v, prevV, nextV)
	}
	return v
}

// triPin clamps x to [lo, hi], with NaN pinning to lo.
func triPin(x, lo, hi float32) float32 {
	m := x
	if hi < m {
		m = hi
	}
	if !(lo < m) {
		m = lo
	}
	return m
}

// triClamp pins x and y independently to the bounding box formed by the corners of min and max (min/max per the
// ordering imposed by c).
func triClamp(p, minPt, maxPt geom.Point, c triComparator) geom.Point {
	if c.direction == triDirectionHorizontal {
		// With horizontal sorting min.x <= max.x, but there is no relation between the Y components unless min.x ==
		// max.x.
		loY, hiY := minPt.Y, maxPt.Y
		if hiY < loY {
			loY, hiY = hiY, loY
		}
		return geom.Point{X: triPin(p.X, minPt.X, maxPt.X), Y: triPin(p.Y, loY, hiY)}
	}
	// With vertical sorting Y's relation is known but not necessarily X's.
	loX, hiX := minPt.X, maxPt.X
	if hiX < loX {
		loX, hiX = hiX, loX
	}
	return geom.Point{X: triPin(p.X, loX, hiX), Y: triPin(p.Y, minPt.Y, maxPt.Y)}
}

// computeBisector computes v's AA partner vertex along the bisector of edge1 and edge2 (edge-AA only).
func (tri *triangulator) computeBisector(edge1, edge2 *triEdge, v *triVertex) {
	if !tri.emitCoverage {
		panic("computeBisector is edge-AA only")
	}
	line1 := edge1.line
	line2 := edge2.line
	line1.normalize()
	line2.normalize()
	cosAngle := line1.a*line2.a + line1.b*line2.b
	if cosAngle > 0.999 {
		return
	}
	if edge1.winding > 0 {
		line1.c--
	} else {
		line1.c++
	}
	if edge2.winding > 0 {
		line2.c--
	} else {
		line2.c++
	}
	var p geom.Point
	if line1.intersect(line2, &p) {
		alpha := uint8(0)
		if edge1.edgeType == triEdgeTypeOuter {
			alpha = 255
		}
		v.partner = &triVertex{point: p, alpha: alpha}
	}
}

// checkForIntersection finds and resolves an intersection between left and right, splitting both edges at a new (or
// existing) vertex.
func (tri *triangulator) checkForIntersection(left, right *triEdge, activeEdges *triEdgeList, current **triVertex, mesh *triVertexList, c triComparator) triBoolFail {
	if left == nil || right == nil {
		return triBoolFalse
	}
	var p geom.Point
	var alpha uint8
	// If intersect is going to be called, there must be tops and bottoms.
	if left.top == nil || left.bottom == nil || right.top == nil || right.bottom == nil {
		return triBoolFailed
	}
	if left.intersect(right, &p, &alpha) && p.IsFinite() {
		var v *triVertex
		top := *current
		// If the intersection point is above the current vertex, rewind to the vertex above the intersection.
		for top != nil && c.sweepLt(p, top.point) {
			top = top.prev
		}

		// Always clamp the intersection to lie between the vertices of each segment, since in theory that is where the
		// intersection is, but in reality floating point error may have computed an intersection beyond a vertex's
		// component(s).
		p = triClamp(p, left.top.point, left.bottom.point, c)
		p = triClamp(p, right.top.point, right.bottom.point, c)

		switch {
		case triCoincident(p, left.top.point):
			v = left.top
		case triCoincident(p, left.bottom.point):
			v = left.bottom
		case triCoincident(p, right.top.point):
			v = right.top
		case triCoincident(p, right.bottom.point):
			v = right.bottom
		default:
			v = tri.makeSortedVertex(p, alpha, mesh, top, c)
			if left.top.partner != nil {
				v.synthetic = true
				tri.computeBisector(left, right, v)
			}
		}
		rewindTo := top
		if rewindTo == nil {
			rewindTo = v
		}
		if !triRewind(activeEdges, current, rewindTo, c) {
			return triBoolFailed
		}
		if tri.splitEdge(left, v, activeEdges, current, c) == triBoolFailed {
			return triBoolFailed
		}
		if tri.splitEdge(right, v, activeEdges, current, c) == triBoolFailed {
			return triBoolFailed
		}
		if alpha > v.alpha {
			v.alpha = alpha
		}
		return triBoolTrue
	}
	return tri.intersectEdgePair(left, right, activeEdges, current, c)
}

// sanitizeContours clamps/rounds vertex coordinates and removes degenerate (coincident, non-finite, or collinear)
// vertices from each contour.
func (tri *triangulator) sanitizeContours(contours []triVertexList) {
	for ci := range contours {
		contour := &contours[ci]
		if contour.head == nil {
			continue
		}
		prev := contour.tail
		prev.point.X = doubleToClampedScalar(float64(prev.point.X))
		prev.point.Y = doubleToClampedScalar(float64(prev.point.Y))
		if tri.roundVerticesToQuarterPixel {
			triRoundQuarterPixel(&prev.point)
		}
		for v := contour.head; v != nil; {
			v.point.X = doubleToClampedScalar(float64(v.point.X))
			v.point.Y = doubleToClampedScalar(float64(v.point.Y))
			if tri.roundVerticesToQuarterPixel {
				triRoundQuarterPixel(&v.point)
			}
			next := v.next
			nextWrap := next
			if nextWrap == nil {
				nextWrap = contour.head
			}
			switch {
			case triCoincident(prev.point, v.point):
				contour.remove(v)
			case !v.point.IsFinite():
				contour.remove(v)
			case !tri.preserveCollinearVertices &&
				makeTriLine(prev.point, nextWrap.point).dist(v.point) == 0.0:
				contour.remove(v)
			default:
				prev = v
			}
			v = next
		}
	}
}

// mergeCoincidentVertices merges any vertex in mesh that has become coincident with its sweep-order predecessor.
func (tri *triangulator) mergeCoincidentVertices(mesh *triVertexList, c triComparator) bool {
	if mesh.head == nil {
		return false
	}
	merged := false
	for v := mesh.head.next; v != nil; {
		next := v.next
		if c.sweepLt(v.point, v.prev.point) {
			v.point = v.prev.point
		}
		if triCoincident(v.prev.point, v.point) {
			tri.mergeVertices(v, v.prev, mesh, c)
			merged = true
		}
		v = next
	}
	return merged
}

// buildEdges converts contours to a mesh of edges (stage 2).
func (tri *triangulator) buildEdges(contours []triVertexList, mesh *triVertexList, c triComparator) {
	for ci := range contours {
		contour := &contours[ci]
		prev := contour.tail
		for v := contour.head; v != nil; {
			next := v.next
			tri.makeConnectingEdge(prev, v, triEdgeTypeInner, c, 1)
			mesh.append(v)
			prev = v
			v = next
		}
	}
}

// triSortedMerge merges two sweep-ordered vertex lists (front, back) into result.
func triSortedMerge(front, back, result *triVertexList, c triComparator) {
	a := front.head
	b := back.head
	for a != nil && b != nil {
		if c.sweepLt(a.point, b.point) {
			front.remove(a)
			result.append(a)
			a = front.head
		} else {
			back.remove(b)
			result.append(b)
			b = back.head
		}
	}
	result.appendList(front)
	result.appendList(back)
}

// triMergeSort sorts the vertices in sweep order (stage 3).
func triMergeSort(vertices *triVertexList, c triComparator) {
	slow := vertices.head
	if slow == nil {
		return
	}
	fast := slow.next
	if fast == nil {
		return
	}
	for {
		fast = fast.next
		if fast != nil {
			fast = fast.next
			slow = slow.next
		}
		if fast == nil {
			break
		}
	}
	front := triVertexList{head: vertices.head, tail: slow}
	back := triVertexList{head: slow.next, tail: vertices.tail}
	front.tail.next = nil
	back.head.prev = nil

	triMergeSort(&front, c)
	triMergeSort(&back, c)

	vertices.head = nil
	vertices.tail = nil
	triSortedMerge(&front, &back, vertices, c)
}

// triSortMesh sorts the mesh's vertices in sweep order.
func triSortMesh(vertices *triVertexList, c triComparator) {
	if vertices == nil || vertices.head == nil {
		return
	}
	triMergeSort(vertices, c)
}

// simplify inserts vertices at intersecting edges (stage 4).
func (tri *triangulator) simplify(mesh *triVertexList, c triComparator) triSimplifyResult {
	initialNumEdges := tri.numEdges
	initialNumVertices := 0
	for v := mesh.head; v != nil; v = v.next {
		initialNumVertices++
	}
	numSelfIntersections := 0

	var activeEdges triEdgeList
	result := triSimplifyAlreadySimple
	numVisitedVertices := 0
	for v := mesh.head; v != nil; v = v.next {
		numVisitedVertices++
		if !v.isConnected() {
			continue
		}

		// The max increase observed across a broad corpus of test inputs, using only this triangulator and the software
		// path renderer, is 17x.
		if tri.numEdges > 170*initialNumEdges {
			return triSimplifyFailed
		}
		if numVisitedVertices > 170*initialNumVertices {
			return triSimplifyFailed
		}

		var leftEnclosingEdge, rightEnclosingEdge *triEdge
		for restart := true; restart; {
			restart = false
			leftEnclosingEdge, rightEnclosingEdge = triFindEnclosingEdges(v, &activeEdges)
			v.leftEnclosingEdge = leftEnclosingEdge
			v.rightEnclosingEdge = rightEnclosingEdge
			if v.firstEdgeBelow != nil {
				for edge := v.firstEdgeBelow; edge != nil; edge = edge.nextEdgeBelow {
					l := tri.checkForIntersection(leftEnclosingEdge, edge, &activeEdges, &v, mesh, c)
					if l == triBoolFailed {
						return triSimplifyFailed
					}
					if l == triBoolFalse {
						r := tri.checkForIntersection(edge, rightEnclosingEdge, &activeEdges, &v,
							mesh, c)
						if r == triBoolFailed {
							return triSimplifyFailed
						}
						if r == triBoolFalse {
							// Neither l nor r is true.
							continue
						}
					}

					// Either l or r is true.
					result = triSimplifyFoundSelfIntersection
					restart = true
					numSelfIntersections++
					break
				}
			} else {
				bf := tri.checkForIntersection(leftEnclosingEdge, rightEnclosingEdge,
					&activeEdges, &v, mesh, c)
				if bf == triBoolFailed {
					return triSimplifyFailed
				}
				if bf == triBoolTrue {
					result = triSimplifyFoundSelfIntersection
					restart = true
					numSelfIntersections++
				}
			}

			// In pathological cases a path can intersect itself millions of times. After 500,000 self-intersections are
			// found, reject the path.
			if numSelfIntersections > 500000 {
				return triSimplifyFailed
			}
		}
		for e := v.firstEdgeAbove; e != nil; e = e.nextEdgeAbove {
			if !activeEdges.remove(e) {
				return triSimplifyFailed
			}
		}
		leftEdge := leftEnclosingEdge
		for e := v.firstEdgeBelow; e != nil; e = e.nextEdgeBelow {
			activeEdges.insert(e, leftEdge)
			leftEdge = e
		}
	}
	if activeEdges.head != nil || activeEdges.tail != nil {
		panic("active edges remain after simplify")
	}
	return result
}

// tessellate builds monotone polygons from the simplified mesh (stage 5).
func (tri *triangulator) tessellate(vertices *triVertexList) (*triPoly, bool) {
	var activeEdges triEdgeList
	var polys *triPoly
	for v := vertices.head; v != nil; v = v.next {
		if !v.isConnected() {
			continue
		}
		leftEnclosingEdge, rightEnclosingEdge := triFindEnclosingEdges(v, &activeEdges)
		var leftPoly, rightPoly *triPoly
		if v.firstEdgeAbove != nil {
			leftPoly = v.firstEdgeAbove.leftPoly
			rightPoly = v.lastEdgeAbove.rightPoly
		} else {
			if leftEnclosingEdge != nil {
				leftPoly = leftEnclosingEdge.rightPoly
			}
			if rightEnclosingEdge != nil {
				rightPoly = rightEnclosingEdge.leftPoly
			}
		}
		if v.firstEdgeAbove != nil {
			if leftPoly != nil {
				leftPoly = leftPoly.addEdge(v.firstEdgeAbove, triSideRight, tri)
			}
			if rightPoly != nil {
				rightPoly = rightPoly.addEdge(v.lastEdgeAbove, triSideLeft, tri)
			}
			for e := v.firstEdgeAbove; e != v.lastEdgeAbove; e = e.nextEdgeAbove {
				rightEdge := e.nextEdgeAbove
				activeEdges.remove(e)
				if e.rightPoly != nil {
					e.rightPoly.addEdge(e, triSideLeft, tri)
				}
				if rightEdge.leftPoly != nil && rightEdge.leftPoly != e.rightPoly {
					rightEdge.leftPoly.addEdge(e, triSideRight, tri)
				}
			}
			activeEdges.remove(v.lastEdgeAbove)
			if v.firstEdgeBelow == nil {
				if leftPoly != nil && rightPoly != nil && leftPoly != rightPoly {
					rightPoly.partner = leftPoly
					leftPoly.partner = rightPoly
				}
			}
		}
		if v.firstEdgeBelow != nil {
			if v.firstEdgeAbove == nil {
				if leftPoly != nil && rightPoly != nil {
					if leftPoly == rightPoly {
						if leftPoly.tail != nil && leftPoly.tail.side == triSideLeft {
							leftPoly = tri.makePoly(&polys, leftPoly.lastVertex(),
								leftPoly.winding)
							leftEnclosingEdge.rightPoly = leftPoly
						} else {
							rightPoly = tri.makePoly(&polys, rightPoly.lastVertex(),
								rightPoly.winding)
							rightEnclosingEdge.leftPoly = rightPoly
						}
					}
					join := tri.allocateEdge(leftPoly.lastVertex(), v, 1, triEdgeTypeInner)
					leftPoly = leftPoly.addEdge(join, triSideRight, tri)
					rightPoly = rightPoly.addEdge(join, triSideLeft, tri)
				}
			}
			leftEdge := v.firstEdgeBelow
			leftEdge.leftPoly = leftPoly
			activeEdges.insert(leftEdge, leftEnclosingEdge)
			for rightEdge := leftEdge.nextEdgeBelow; rightEdge != nil; rightEdge = rightEdge.nextEdgeBelow {
				activeEdges.insert(rightEdge, leftEdge)
				winding := 0
				if leftEdge.leftPoly != nil {
					winding = leftEdge.leftPoly.winding
				}
				winding += leftEdge.winding
				if winding != 0 {
					poly := tri.makePoly(&polys, v, winding)
					leftEdge.rightPoly = poly
					rightEdge.leftPoly = poly
				}
				leftEdge = rightEdge
			}
			v.lastEdgeBelow.rightPoly = rightPoly
		}
	}
	return polys, true
}

// callTessellate dispatches to tessellateOverride if set, else the base tessellate().
func (tri *triangulator) callTessellate(vertices *triVertexList, c triComparator) (*triPoly, bool) {
	if tri.tessellateOverride != nil {
		return tri.tessellateOverride(vertices, c)
	}
	return tri.tessellate(vertices)
}

// contoursToMesh drives stage 2: sanitize the contours, then build a mesh of edges.
func (tri *triangulator) contoursToMesh(contours []triVertexList, mesh *triVertexList, c triComparator) {
	tri.sanitizeContours(contours)
	tri.buildEdges(contours, mesh, c)
}

// contoursToPolys drives stages 2-5: mesh construction, sort, coincident-vertex merge, simplify, and tessellate.
func (tri *triangulator) contoursToPolys(contours []triVertexList) (*triPoly, bool) {
	pathBounds := tri.path.Bounds()
	c := triComparator{direction: triDirectionVertical}
	if pathBounds.Width() > pathBounds.Height() {
		c.direction = triDirectionHorizontal
	}
	var mesh triVertexList
	tri.contoursToMesh(contours, &mesh, c)
	triSortMesh(&mesh, c)
	tri.mergeCoincidentVertices(&mesh, c)
	result := tri.simplify(&mesh, c)
	if result == triSimplifyFailed {
		return nil, false
	}
	return tri.callTessellate(&mesh, c)
}

// triGetContourCount returns the number of contours in p. It could theoretically be more aggressive about not counting
// empty contours, but it must match the exact number of contour linked lists the tessellator creates later on.
func triGetContourCount(p *path.Path) int {
	contourCnt := 1
	hasPoints := false
	it := path.NewIter(p, false)
	first := true
	for {
		verb, _, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			if !first {
				contourCnt++
			}
			hasPoints = true
		case path.VerbLine, path.VerbConic, path.VerbQuad, path.VerbCubic:
			hasPoints = true
		case path.VerbClose:
		}
		first = false
	}
	if !hasPoints {
		return 0
	}
	return contourCnt
}

// pathToPolys runs the full pipeline (stages 1-5) from the source path to a list of polys.
func (tri *triangulator) pathToPolys(tolerance float32, clipBounds geom.Rect, isLinear *bool) (*triPoly, bool) {
	contourCnt := triGetContourCount(tri.path)
	if contourCnt <= 0 {
		*isLinear = true
		return nil, true
	}
	if tri.path.FillType().IsInverse() {
		contourCnt++
	}
	contours := make([]triVertexList, contourCnt)
	tri.pathToContours(tolerance, clipBounds, contours, isLinear)
	return tri.contoursToPolys(contours)
}

// triCountPoints returns the total emitted vertex count across all polys under overrideFillType.
func triCountPoints(polys *triPoly, overrideFillType path.FillType) int64 {
	var count int64
	for poly := polys; poly != nil; poly = poly.next {
		if triApplyFillTypePoly(overrideFillType, poly) && poly.count >= 3 {
			count += int64(poly.count-2) * 3
		}
	}
	return count
}

// polysToTrianglesInto ear-clips every poly (under overrideFillType) directly into w (stage 6).
func (tri *triangulator) polysToTrianglesInto(polys *triPoly, overrideFillType path.FillType, w *triVertexWriter) {
	for poly := polys; poly != nil; poly = poly.next {
		if triApplyFillTypePoly(overrideFillType, poly) {
			tri.emitPoly(poly, w)
		}
	}
}

// polysToTriangles allocates vertex space via vertexAllocator and ear-clips every poly into it. Returns the emitted
// vertex count.
func (tri *triangulator) polysToTriangles(polys *triPoly, vertexAllocator eagerVertexAllocator) int {
	count64 := triCountPoints(polys, tri.path.FillType())
	if count64 == 0 || count64 > math.MaxInt32 {
		return 0
	}
	count := int(count64)

	vertexStride := uint64(8) // two float32s per point
	if tri.emitCoverage {
		vertexStride += 4
	}
	verts := vertexAllocator.lockWriter(vertexStride, count)
	if verts == nil {
		return 0
	}

	w := triVertexWriter{data: verts, emitCoverage: tri.emitCoverage}
	tri.polysToTrianglesInto(polys, tri.path.FillType(), &w)

	actualCount := w.off / int(vertexStride)
	if actualCount > count {
		panic("triangulator emitted more vertices than counted")
	}
	vertexAllocator.unlock(actualCount)
	return actualCount
}

// scalarAve returns the average of a and b.
func scalarAve(a, b float32) float32 { return (a + b) / 2 }
