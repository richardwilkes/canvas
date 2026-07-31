// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The edge-level iteration and clipping the scan converter and clippers consume, where every contour is implicitly
// closed (a fill's close line is materialized) and moves/closes disappear.

package path

import "github.com/richardwilkes/canvas/geom"

// EdgeIterResult is one edge from an EdgeIter: Pts holds the segment's points (2 for a line, 3 for a quad/conic, 4 for
// a cubic) beginning with the segment's start point. Pts aliases either the path's point storage or the iterator's
// scratch space; callers must not mutate or retain it across Next calls.
type EdgeIterResult struct {
	Pts          []geom.Point
	Verb         Verb
	IsNewContour bool
}

// EdgeIter iterates a path's edges, materializing the implicit close line of every contour (whether or not it was
// explicitly closed) and never emitting moves or closes.
type EdgeIter struct {
	path             *Path
	verbIdx          int
	ptIdx            int
	moveToIdx        int
	conicIdx         int
	scratch          [2]geom.Point
	needsCloseLine   bool
	nextIsNewContour bool
}

// NewEdgeIter returns an EdgeIter over p.
func NewEdgeIter(p *Path) *EdgeIter {
	it := &EdgeIter{}
	it.Reset(p)
	return it
}

// Reset reinitializes the iterator to iterate p from the beginning, reusing the receiver's storage. A hot caller (the
// scan converter's edge builder) holds one EdgeIter and Resets it per build instead of allocating a fresh *EdgeIter
// each time.
func (it *EdgeIter) Reset(p *Path) {
	*it = EdgeIter{path: p, conicIdx: -1}
}

// Release drops the iterator's reference to its path so an idle pooled owner does not pin the path (and its point,
// verb and conic-weight storage) against collection. The iterator must be Reset before it is used again.
func (it *EdgeIter) Release() {
	*it = EdgeIter{conicIdx: -1}
}

// ConicWeight returns the weight of the conic most recently returned by Next.
func (it *EdgeIter) ConicWeight() float32 {
	return it.path.conicWeights[it.conicIdx]
}

func (it *EdgeIter) closeLine() EdgeIterResult {
	it.scratch[0] = it.path.points[it.ptIdx-1]
	it.scratch[1] = it.path.points[it.moveToIdx]
	it.needsCloseLine = false
	it.nextIsNewContour = true
	return EdgeIterResult{Pts: it.scratch[:], Verb: VerbLine, IsNewContour: false}
}

// Next returns the next edge; ok is false when iteration is complete.
func (it *EdgeIter) Next() (result EdgeIterResult, ok bool) {
	for {
		if it.verbIdx == len(it.path.verbs) {
			if it.needsCloseLine {
				return it.closeLine(), true
			}
			return EdgeIterResult{}, false
		}
		verb := it.path.verbs[it.verbIdx]
		it.verbIdx++
		switch verb {
		case VerbMove:
			if it.needsCloseLine {
				res := it.closeLine()
				it.moveToIdx = it.ptIdx
				it.ptIdx++
				return res, true
			}
			it.moveToIdx = it.ptIdx
			it.ptIdx++
			it.nextIsNewContour = true
		case VerbClose:
			if it.needsCloseLine {
				return it.closeLine(), true
			}
		default:
			// Actual edge.
			v := int(verb)
			ptsCount := (v + 2) / 2
			cwsCount := (v & (v - 1)) / 2
			it.needsCloseLine = true
			it.ptIdx += ptsCount
			it.conicIdx += cwsCount
			isNewContour := it.nextIsNewContour
			it.nextIsNewContour = false
			return EdgeIterResult{
				Pts:          it.path.points[it.ptIdx-(ptsCount+1) : it.ptIdx],
				Verb:         verb,
				IsNewContour: isNewContour,
			}, true
		}
	}
}

// ConicToQuads approximates the conic with quads at the given tolerance, returning the shared quad points (1 + 2*count
// points) and the quad count.
func ConicToQuads(pts []geom.Point, weight, tol float32) (quadPts []geom.Point, quadCount int) {
	return ConicToQuadsInto(pts, weight, tol, make([]geom.Point, geom.MaxConicToQuadPointCount))
}

// ConicToQuadsInto is ConicToQuads writing into a caller-provided buffer instead of allocating one, so hot callers (the
// scan converter's edge builder) can reuse a single buffer across conics. dst must have capacity for the worst case,
// geom.MaxConicToQuadPointCount points; the returned slice is a prefix of dst.
func ConicToQuadsInto(pts []geom.Point, weight, tol float32, dst []geom.Point) (quadPts []geom.Point, quadCount int) {
	conic := geom.MakeConic(pts[0], pts[1], pts[2], weight)
	pow2 := conic.ComputeQuadPOW2(tol)
	storage := dst[:1+2*(1<<pow2)]
	count := conic.ChopIntoQuadsPOW2(storage, pow2)
	return storage, count
}

// kConicTol is the conic-to-quad tolerance the scan converter and edge clipper use.
const kConicTol = 0.25

// EdgeClipScratch bundles ClipPath's reusable temporaries — the edge iterator, the edge clipper, and the conic->quad
// approximation buffer — so a hot caller (the scan converter's edge builder) can clip many paths without the per-call
// *EdgeIter / *EdgeClipper allocations and the per-conic ConicToQuads allocation. The zero value is ready to use.
type EdgeClipScratch struct {
	iter    EdgeIter
	clipper geom.EdgeClipper
	conic   [geom.MaxConicToQuadPointCount]geom.Point
}

// Release drops the scratch's reference to the path it last clipped so an idle pooled owner does not pin it against
// collection. The scratch stays ready to use: ClipPath resets everything it needs.
func (s *EdgeClipScratch) Release() {
	s.iter.Release()
}

// ClipPath clips each segment of p against clip, reusing the receiver's temporaries across calls.
func (s *EdgeClipScratch) ClipPath(p *Path, clip geom.Rect, canCullToTheRight bool, consume func(clipper *geom.EdgeClipper, newContour bool)) {
	iter := &s.iter
	iter.Reset(p)
	clipper := &s.clipper
	clipper.Reset(canCullToTheRight)

	for {
		e, ok := iter.Next()
		if !ok {
			break
		}
		switch e.Verb {
		case VerbLine:
			if clipper.ClipLine(e.Pts[0], e.Pts[1], clip) {
				consume(clipper, e.IsNewContour)
			}
		case VerbQuad:
			if clipper.ClipQuad(e.Pts, clip) {
				consume(clipper, e.IsNewContour)
			}
		case VerbConic:
			quadPts, count := ConicToQuadsInto(e.Pts, iter.ConicWeight(), kConicTol, s.conic[:])
			for i := 0; i < count; i++ {
				if clipper.ClipQuad(quadPts[i*2:i*2+3], clip) {
					consume(clipper, e.IsNewContour)
				}
			}
		case VerbCubic:
			if clipper.ClipCubic(e.Pts, clip) {
				consume(clipper, e.IsNewContour)
			}
		default:
		}
	}
}

// ClipPath clips each segment of the path against clip (conics are first approximated by quads at the 0.25 tolerance),
// passing each clipped segment batch (in a clipper ready for iteration) to consume along with whether the segment began
// a new contour. Callers that clip many paths should hold an EdgeClipScratch and call its ClipPath method to avoid
// per-call allocations.
func ClipPath(p *Path, clip geom.Rect, canCullToTheRight bool, consume func(clipper *geom.EdgeClipper, newContour bool)) {
	var s EdgeClipScratch
	s.ClipPath(p, clip, canCullToTheRight, consume)
}
