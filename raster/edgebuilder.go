// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The edge builders convert a path into the edge list a scan converter walks. edgeBuilder feeds the non-AA converter;
// analyticEdgeBuilder feeds the analytic-AA converter. The two builders share the same driver functions through the
// edgeSink interface.

package raster

import (
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// The edge builders are pooled across fills (and across the concurrent bands of FillPathParallel, which is why a
// sync.Pool rather than a single shared instance): each fill Gets a builder, whose arenas and list slice are reset and
// reused, so steady-state curve/stroke fills allocate no edges at all. The edges never outlive the fill (they are
// built, sorted, walked, and discarded within fillPath / aaaFillPath), so returning the builder on function exit is
// safe.
var (
	basicBuilderPool    = sync.Pool{New: func() any { return new(edgeBuilder) }}
	analyticBuilderPool = sync.Pool{New: func() any { return new(analyticEdgeBuilder) }}
)

func getBasicEdgeBuilder() *edgeBuilder { return basicBuilderPool.Get().(*edgeBuilder) }

func putBasicEdgeBuilder(b *edgeBuilder) {
	b.reset()
	basicBuilderPool.Put(b)
}

func getAnalyticEdgeBuilder() *analyticEdgeBuilder {
	return analyticBuilderPool.Get().(*analyticEdgeBuilder)
}

func putAnalyticEdgeBuilder(b *analyticEdgeBuilder) {
	b.reset()
	analyticBuilderPool.Put(b)
}

// combineResult reports how a new vertical edge combined with the previous one in the edge list.
type combineResult uint8

const (
	combineNo combineResult = iota
	combinePartial
	combineTotal
)

// conicTol is the conic-to-quad tolerance used by the edge builders.
const conicTol = 0.25

// edgeSink receives the per-segment callbacks from the shared build drivers.
type edgeSink interface {
	addLine(pts []geom.Point)
	addQuad(pts []geom.Point)
	addCubic(pts []geom.Point)
}

// recoverClip converts an integer clip rect to a float rect.
func recoverClip(src geom.IRect) geom.Rect {
	return geom.RectLTRB(float32(src.Left), float32(src.Top), float32(src.Right), float32(src.Bottom))
}

// buildPolyEdges is the all-lines fast path for building edges. s.iter is reused across builds to avoid a per-build
// *EdgeIter allocation.
func buildPolyEdges(b edgeSink, p *path.Path, iclip *geom.IRect, canCullToTheRight bool, s *edgeScratch) {
	iter := &s.iter
	iter.Reset(p)
	if iclip != nil {
		clip := recoverClip(*iclip)
		for {
			e, ok := iter.Next()
			if !ok {
				break
			}
			// Clipping can turn one line into up to geom.LineClipperMaxPoints-1 segments, since portions clipped out on
			// the left/right become vertical segments.
			var lines [geom.LineClipperMaxPoints]geom.Point
			pts := [2]geom.Point{e.Pts[0], e.Pts[1]}
			lineCount := geom.ClipLine(&pts, clip, &lines, canCullToTheRight)
			for i := 0; i < lineCount; i++ {
				b.addLine(lines[i : i+2])
			}
		}
	} else {
		for {
			e, ok := iter.Next()
			if !ok {
				break
			}
			b.addLine(e.Pts)
		}
	}
}

// edgeScratch holds the reusable per-build temporaries the edge builders would otherwise allocate: iter (the path edge
// iterator, one *EdgeIter per build otherwise), clip (ClipPath's iterator + clipper + conic->quad scratch, reused
// across clipped builds), chop (>= 10 points) for the clipper output and Y-extrema chopping, and conic for the
// conic->quad approximation, sized to the worst case (1 + 2*(1<<MaxConicToQuadPOW2)). Both builders embed one and pass
// it to the build drivers; because these buffers flow into the edgeSink interface calls (which forces them to escape),
// a reused builder field turns per-segment/per-build heap allocations into one-time, pooled ones.
type edgeScratch struct {
	iter  path.EdgeIter
	clip  path.EdgeClipScratch
	chop  [10]geom.Point
	conic [1 + 2*(1<<geom.MaxConicToQuadPOW2)]geom.Point
}

// release drops the *path.Path references iter and clip's iterator retain from the last build, so an idle pooled
// builder pins no caller path. The buffers themselves are kept; the build drivers Reset the iterators before use.
func (s *edgeScratch) release() {
	s.iter.Release()
	s.clip.Release()
}

// buildCurveEdges is the general path, handling curves. Returns false if clipped output produced non-finite points (the
// caller then reports zero edges). s is a caller-owned (reused) scratch buffer for the per-segment temporaries.
func buildCurveEdges(b edgeSink, p *path.Path, iclip *geom.IRect, canCullToTheRight bool, s *edgeScratch) bool {
	if iclip != nil {
		clip := recoverClip(*iclip)
		isFinite := true

		s.clip.ClipPath(p, clip, canCullToTheRight, func(clipper *geom.EdgeClipper, _ bool) {
			pts := s.chop[:4]
			for {
				verb, more := clipper.Next(pts)
				if !more {
					break
				}
				var count int
				switch verb {
				case geom.ClipVerbLine:
					count = 2
				case geom.ClipVerbQuad:
					count = 3
				case geom.ClipVerbCubic:
					count = 4
				}
				for i := 0; i < count; i++ {
					if !pts[i].IsFinite() {
						isFinite = false
						return
					}
				}
				switch verb {
				case geom.ClipVerbLine:
					b.addLine(pts[:2])
				case geom.ClipVerbQuad:
					b.addQuad(pts[:3])
				case geom.ClipVerbCubic:
					b.addCubic(pts[:4])
				}
			}
		})
		return isFinite
	}

	iter := &s.iter
	iter.Reset(p)
	handleQuad := func(pts []geom.Point) {
		monoX := s.chop[:5]
		n := geom.ChopQuadAtYExtrema(pts, monoX)
		for i := 0; i <= n; i++ {
			b.addQuad(monoX[i*2 : i*2+3])
		}
	}
	for {
		e, ok := iter.Next()
		if !ok {
			break
		}
		switch e.Verb {
		case path.VerbLine:
			b.addLine(e.Pts)
		case path.VerbQuad:
			handleQuad(e.Pts)
		case path.VerbConic:
			quadPts, count := path.ConicToQuadsInto(e.Pts, iter.ConicWeight(), conicTol, s.conic[:])
			for i := 0; i < count; i++ {
				handleQuad(quadPts[i*2 : i*2+3])
			}
		case path.VerbCubic:
			monoY := s.chop[:10]
			n := geom.ChopCubicAtYExtrema(e.Pts, monoY)
			for i := 0; i <= n; i++ {
				b.addCubic(monoY[i*3 : i*3+4])
			}
		default:
		}
	}
	return true
}

///////////////////////////////////////////////////////////////////////////////
// Basic (non-AA) edge builder

// edgeBuilder converts a path into the edge list the non-AA scan converter walks. Edges are bump-allocated from
// per-type arenas rather than one heap object per segment; list and the arenas are rewound by reset so a pooled builder
// reallocates nothing in steady state.
type edgeBuilder struct {
	inv  inverseBlitter
	list []*Edge
	// head/tail/inv home fillPath's sentinel edges and inverse-blitter wrapper: their addresses enter the edge linked
	// list / blitter chain, which would force a per-fill heap allocation as locals. They live and die within one
	// fillPath, like the arenas.
	head    Edge
	tail    Edge
	lines   edgeArena[Edge]
	quads   edgeArena[QuadraticEdge]
	cubics  edgeArena[CubicEdge]
	scratch edgeScratch
}

// reset rewinds the builder for reuse from basicBuilderPool.
func (b *edgeBuilder) reset() {
	b.list = b.list[:0]
	b.lines.reset()
	b.quads.reset()
	b.cubics.reset()
	// Drop the sentinel links, the inverse blitter's wrapped blitter and the scratch iterators' path so the idle pooled
	// builder pins none of arena edges, a caller's blitter or a caller's path.
	b.head = Edge{}
	b.tail = Edge{}
	b.inv = inverseBlitter{}
	b.scratch.release()
}

// isVertical reports whether edge is vertical; only edges that were originally lines are considered vertical, to avoid
// numerical issues (crbug.com/1154864).
func isVertical(edge *Edge) bool {
	return edge.DxDy == 0 && edge.EdgeType == EdgeTypeLine
}

// combineVertical merges/cancels a new vertical line edge against the previous one when they abut or overlap in Y at
// the same X.
func (b *edgeBuilder) combineVertical(edge, last *Edge) combineResult {
	if last.EdgeType != EdgeTypeLine || last.DxDy != 0 || edge.X != last.X {
		return combineNo
	}
	if edge.Winding == last.Winding {
		if edge.LastY+1 == last.FirstY {
			last.FirstY = edge.FirstY
			return combinePartial
		}
		if edge.FirstY == last.LastY+1 {
			last.LastY = edge.LastY
			return combinePartial
		}
		return combineNo
	}
	if edge.FirstY == last.FirstY {
		if edge.LastY == last.LastY {
			return combineTotal
		}
		if edge.LastY < last.LastY {
			last.FirstY = edge.LastY + 1
			return combinePartial
		}
		last.FirstY = last.LastY + 1
		last.LastY = edge.LastY
		last.Winding = edge.Winding
		return combinePartial
	}
	if edge.LastY == last.LastY {
		if edge.FirstY > last.FirstY {
			last.LastY = edge.FirstY - 1
			return combinePartial
		}
		last.LastY = last.FirstY - 1
		last.FirstY = edge.FirstY
		last.Winding = edge.Winding
		return combinePartial
	}
	return combineNo
}

func (b *edgeBuilder) addLine(pts []geom.Point) {
	edge := b.lines.alloc()
	if edge.SetLine(pts[0], pts[1]) {
		combine := combineNo
		if isVertical(edge) && len(b.list) > 0 {
			combine = b.combineVertical(edge, b.list[len(b.list)-1])
		}
		switch combine {
		case combineTotal:
			b.list = b.list[:len(b.list)-1]
		case combinePartial:
		case combineNo:
			b.list = append(b.list, edge)
		}
	}
}

func (b *edgeBuilder) addQuad(pts []geom.Point) {
	edge := b.quads.alloc()
	if edge.SetQuadratic(pts) {
		b.list = append(b.list, &edge.Edge)
	}
}

func (b *edgeBuilder) addCubic(pts []geom.Point) {
	edge := b.cubics.alloc()
	if edge.SetCubic(pts) {
		b.list = append(b.list, &edge.Edge)
	}
}

// buildEdges builds the edge list for p. shiftedClip, when non-nil, is the clip.
func (b *edgeBuilder) buildEdges(p *path.Path, shiftedClip *geom.IRect) int {
	// If we're convex, then we need both edges, even if the right edge is past the clip.
	canCullToTheRight := !p.IsConvex()

	// We can use the buildPoly optimization if all the segments are lines.
	if p.SegmentMasks() == path.SegmentLine {
		buildPolyEdges(b, p, shiftedClip, canCullToTheRight, &b.scratch)
		return len(b.list)
	}
	if !buildCurveEdges(b, p, shiftedClip, canCullToTheRight, &b.scratch) {
		return 0
	}
	return len(b.list)
}

///////////////////////////////////////////////////////////////////////////////
// Analytic (AA) edge builder

// analyticEdgeBuilder converts a path into the edge list the analytic-AA scan converter walks. As with the basic
// builder, edges are bump-allocated from per-type arenas rather than one heap object per segment, and the builder is
// pooled (analyticBuilderPool) so a stroked/curve fill's hundreds of edges reallocate nothing in steady state.
type analyticEdgeBuilder struct {
	list []*AnalyticEdge
	// head and tail are the sorted-list boundary sentinels aaaFillPath threads through the edges. They live on the
	// pooled builder rather than as aaaFillPath locals because the list's real edges (also on this builder) point back
	// at them, which would otherwise force the locals to escape to the heap per fill. reset zeroes them so each reused
	// fill sees the same clean state a fresh `var` would.
	head    AnalyticEdge
	tail    AnalyticEdge
	lines   edgeArena[AnalyticEdge]
	quads   edgeArena[AnalyticQuadraticEdge]
	cubics  edgeArena[AnalyticCubicEdge]
	scratch edgeScratch
}

// reset rewinds the builder for reuse from analyticBuilderPool.
func (b *analyticEdgeBuilder) reset() {
	b.list = b.list[:0]
	b.lines.reset()
	b.quads.reset()
	b.cubics.reset()
	b.head = AnalyticEdge{}
	b.tail = AnalyticEdge{}
	b.scratch.release()
}

// isVerticalAnalytic reports whether an analytic edge is vertical.
func isVerticalAnalytic(edge *AnalyticEdge) bool {
	return edge.DX == 0 && edge.EdgeType == EdgeTypeLine
}

// approximatelyEqualFixed reports whether two Fixed values are within a small tolerance of each other.
func approximatelyEqualFixed(a, b Fixed) bool {
	return FixedAbs(a-b) < 0x100
}

// combineVertical merges/cancels a new vertical analytic edge against the previous one when they abut or overlap in Y
// at the same X.
func (b *analyticEdgeBuilder) combineVertical(edge, last *AnalyticEdge) combineResult {
	if last.EdgeType != EdgeTypeLine || last.DX != 0 || edge.X != last.X {
		return combineNo
	}
	if edge.Winding == last.Winding {
		if edge.LowerY == last.UpperY {
			last.UpperY = edge.UpperY
			last.Y = last.UpperY
			return combinePartial
		}
		if approximatelyEqualFixed(edge.UpperY, last.LowerY) {
			last.LowerY = edge.LowerY
			return combinePartial
		}
		return combineNo
	}
	if approximatelyEqualFixed(edge.UpperY, last.UpperY) {
		if approximatelyEqualFixed(edge.LowerY, last.LowerY) {
			return combineTotal
		}
		if edge.LowerY < last.LowerY {
			last.UpperY = edge.LowerY
			last.Y = last.UpperY
			return combinePartial
		}
		last.UpperY = last.LowerY
		last.Y = last.UpperY
		last.LowerY = edge.LowerY
		last.Winding = edge.Winding
		return combinePartial
	}
	if approximatelyEqualFixed(edge.LowerY, last.LowerY) {
		if edge.UpperY > last.UpperY {
			last.LowerY = edge.UpperY
			return combinePartial
		}
		last.LowerY = last.UpperY
		last.UpperY = edge.UpperY
		last.Y = last.UpperY
		last.Winding = edge.Winding
		return combinePartial
	}
	return combineNo
}

func (b *analyticEdgeBuilder) addLine(pts []geom.Point) {
	edge := b.lines.alloc()
	if edge.SetLine(pts[0], pts[1]) {
		combine := combineNo
		if isVerticalAnalytic(edge) && len(b.list) > 0 {
			combine = b.combineVertical(edge, b.list[len(b.list)-1])
		}
		switch combine {
		case combineTotal:
			b.list = b.list[:len(b.list)-1]
		case combinePartial:
		case combineNo:
			b.list = append(b.list, edge)
		}
	}
}

func (b *analyticEdgeBuilder) addQuad(pts []geom.Point) {
	edge := b.quads.alloc()
	if edge.SetQuadratic(pts) {
		b.list = append(b.list, &edge.AnalyticEdge)
	}
}

func (b *analyticEdgeBuilder) addCubic(pts []geom.Point) {
	edge := b.cubics.alloc()
	if edge.SetCubic(pts) {
		b.list = append(b.list, &edge.AnalyticEdge)
	}
}

// buildEdges builds the analytic edge list for p.
func (b *analyticEdgeBuilder) buildEdges(p *path.Path, clip *geom.IRect) int {
	canCullToTheRight := !p.IsConvex()

	if p.SegmentMasks() == path.SegmentLine {
		buildPolyEdges(b, p, clip, canCullToTheRight, &b.scratch)
		return len(b.list)
	}
	if !buildCurveEdges(b, p, clip, canCullToTheRight, &b.scratch) {
		return 0
	}
	return len(b.list)
}
