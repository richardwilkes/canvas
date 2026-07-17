// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Generates a middle-out triangulation of a polygon. Conceptually, middle-out emits one large triangle with vertices on
// both endpoints and a middle point, then recurses on both sides of the new triangle:
//
//	void emit_middle_out_triangulation(int startIdx, int endIdx) {
//	    if (startIdx + 1 == endIdx) {
//	        return;
//	    }
//	    int middleIdx = startIdx + nextPow2(endIdx - startIdx) / 2;
//	    emit_middle_out_triangulation(startIdx, middleIdx);            // Recurse on the left.
//	    emit_triangle(vertices[startIdx], vertices[middleIdx], vertices[endIdx - 1]);
//	    emit_middle_out_triangulation(middleIdx, endIdx);              // Recurse on the right.
//	}
//
// Middle-out produces drastically less work for the rasterizer as compared to a linear triangle strip or fan. The
// triangulator does not know or store all the vertices at once: the caller pushes each vertex in linear order and an
// O(log N) stack determines the correct triangulation. Popped triangles are reported through an emit callback invoked
// in stack-pop order, before the stack is updated with the new vertex.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// middleOutStackVertex is one entry on the middle-out triangulation stack.
type middleOutStackVertex struct {
	point geom.Point
	// vertexIdxDelta is how many polygon vertices away this vertex is from the previous vertex on the stack.
	// vertexStack[0].vertexIdxDelta is always 0.
	vertexIdxDelta int
}

// middleOutPolygonTriangulator performs the middle-out triangulation. maxPushVertexCalls is an upper bound on the
// number of times the caller will call pushVertex; the caller must not exceed it.
type middleOutPolygonTriangulator struct {
	vertexStack []middleOutStackVertex
	top         int // index of the current top vertex
}

// newMiddleOutPolygonTriangulator creates a triangulator seeded with startPoint.
func newMiddleOutPolygonTriangulator(maxPushVertexCalls int, startPoint geom.Point) *middleOutPolygonTriangulator {
	if maxPushVertexCalls < 0 {
		panic("negative maxPushVertexCalls")
	}
	// Determine the deepest the stack can ever go (skNextLog2 is shared with gradientfp.go).
	maxStackDepth := skNextLog2(maxPushVertexCalls) + 1
	m := &middleOutPolygonTriangulator{
		vertexStack: make([]middleOutStackVertex, maxStackDepth+1),
	}
	// The stack will always contain a starting point. This is an implicit moveTo(0, 0) initially, but will be
	// overridden if moveTo() gets called before adding geometry.
	m.vertexStack[0] = middleOutStackVertex{point: startPoint, vertexIdxDelta: 0}
	m.top = 0
	return m
}

// pushVertex emits (in stack-pop order) the triangles that pop off the stack, then pushes pt onto the vertex stack.
//
// The topology wants triangles that have the same vertexIdxDelta on both sides: e.g., a run of 9 points should be
// triangulated as:
//
//	[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]  // vertexIdxDelta == 1
//	[0, 2, 4], [4, 6, 8]                        // vertexIdxDelta == 2
//	[0, 4, 8]                                   // vertexIdxDelta == 4
func (m *middleOutPolygonTriangulator) pushVertex(pt geom.Point, emit func(p0, p1, p2 geom.Point)) {
	// Find as many new triangles as we can pop off the stack that have equal-delta sides. (This is a stack-based
	// implementation of the recursive example method from the class comment.)
	endVertex := m.top
	vertexIdxDelta := 1
	for m.vertexStack[endVertex].vertexIdxDelta == vertexIdxDelta {
		endVertex--
		vertexIdxDelta *= 2
	}

	// Iterates from the current top down to endVertex, yielding the popped triangles as {stack[v-1].point,
	// stack[v].point, pt}.
	for v := m.top; v != endVertex; v-- {
		emit(m.vertexStack[v-1].point, m.vertexStack[v].point, pt)
	}

	// Once the above triangles are popped, push pt to the top of the stack.
	m.top = endVertex + 1
	if m.top >= len(m.vertexStack) {
		panic("middle-out stack overflow: maxPushVertexCalls too small")
	}
	m.vertexStack[m.top] = middleOutStackVertex{point: pt, vertexIdxDelta: vertexIdxDelta}
}

// closeAndMove emits the remaining triangles, then resets the vertex stack with newStartPoint.
func (m *middleOutPolygonTriangulator) closeAndMove(newStartPoint geom.Point, emit func(p0, p1, p2 geom.Point)) {
	// Add an implicit line back to the starting point.
	startPt := m.vertexStack[0].point

	// Triangulate the rest of the polygon. Since we simply have to finish now, we can't be picky anymore about getting
	// a pure middle-out topology.
	endVertex := min(m.top, 1)

	for v := m.top; v != endVertex; v-- {
		emit(m.vertexStack[v-1].point, m.vertexStack[v].point, startPt)
	}

	// Reset the vertex stack with newStartPoint.
	m.top = 0
	m.vertexStack[0] = middleOutStackVertex{point: newStartPoint, vertexIdxDelta: 0}
}

// close is closeAndMove with the same starting point as before.
func (m *middleOutPolygonTriangulator) close(emit func(p0, p1, p2 geom.Point)) {
	m.closeAndMove(m.vertexStack[0].point, emit)
}

// pathMiddleOutFanTriangles transforms and pushes a path's inner-fan vertices onto a middleOutPolygonTriangulator,
// emitting every triangle through the callback.
func pathMiddleOutFanTriangles(p *path.Path, emit func(p0, p1, p2 geom.Point)) {
	m := newMiddleOutPolygonTriangulator(p.CountVerbs(), geom.Point{})
	it := path.NewRawIter(p)
	for {
		verb, pts, _, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			m.closeAndMove(pts[0], emit)
		case path.VerbLine:
			m.pushVertex(pts[1], emit)
		case path.VerbQuad, path.VerbConic:
			m.pushVertex(pts[2], emit)
		case path.VerbCubic:
			m.pushVertex(pts[3], emit)
		case path.VerbClose:
			m.close(emit)
		}
	}
	m.close(emit)
}
