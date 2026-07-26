// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// rawIter iterates the raw verb stream: no cleanup, no injected verbs. For line/quad/conic/cubic records, pts[0] is the
// segment's start point (the previous point in the stream); a move record carries its point in pts[0]; a close record
// carries no points.
type rawIter struct {
	p         *Path
	verbIdx   int
	ptIdx     int
	weightIdx int
}

func newRawIter(p *Path) rawIter {
	return rawIter{p: p}
}

// next returns the next verb record, with the conic weight when verb == VerbConic (1 otherwise).
func (it *rawIter) next() (verb Verb, pts [4]geom.Point, weight float32, ok bool) {
	if it.verbIdx >= len(it.p.verbs) {
		return 0, pts, 0, false
	}
	verb = it.p.verbs[it.verbIdx]
	it.verbIdx++
	weight = 1
	switch verb {
	case VerbMove:
		pts[0] = it.p.points[it.ptIdx]
		it.ptIdx++
	case VerbLine:
		pts[0] = it.p.points[it.ptIdx-1]
		pts[1] = it.p.points[it.ptIdx]
		it.ptIdx++
	case VerbQuad, VerbConic:
		pts[0] = it.p.points[it.ptIdx-1]
		pts[1] = it.p.points[it.ptIdx]
		pts[2] = it.p.points[it.ptIdx+1]
		it.ptIdx += 2
		if verb == VerbConic {
			weight = it.p.conicWeights[it.weightIdx]
			it.weightIdx++
		}
	case VerbCubic:
		pts[0] = it.p.points[it.ptIdx-1]
		pts[1] = it.p.points[it.ptIdx]
		pts[2] = it.p.points[it.ptIdx+1]
		pts[3] = it.p.points[it.ptIdx+2]
		it.ptIdx += 3
	case VerbClose:
	}
	return verb, pts, weight, true
}

// Iter iterates the path's verbs with cleanup — segment start points are supplied, trailing lone moveTos are dropped,
// and when forceClose is set each contour is returned closed, with the closing line segment materialized when the
// contour does not end at its start point.
type Iter struct {
	p          *Path
	verbIdx    int
	ptIdx      int
	weightIdx  int
	moveTo     geom.Point
	lastPt     geom.Point
	forceClose bool
	needClose  bool
	closeLine  bool
}

// NewIter returns an Iter over path, closing each contour when forceClose is true.
func NewIter(p *Path, forceClose bool) *Iter {
	it := &Iter{}
	it.SetPath(p, forceClose)
	return it
}

// SetPath resets the iterator to the start of the given path.
func (it *Iter) SetPath(p *Path, forceClose bool) {
	*it = Iter{p: p, weightIdx: -1, forceClose: forceClose}
}

// ConicWeight returns the weight of the conic most recently returned by Next.
func (it *Iter) ConicWeight() float32 {
	return it.p.conicWeights[it.weightIdx]
}

// IsClosedContour reports whether the contour the iterator is currently positioned in ends with a close verb (always
// true when iterating with forceClose).
func (it *Iter) IsClosedContour() bool {
	verbs := it.p.verbs
	i := it.verbIdx
	if len(verbs) == 0 || i >= len(verbs) {
		return false
	}
	if it.forceClose {
		return true
	}
	if verbs[i] == VerbMove {
		i++ // skip the initial moveto
	}
	for i < len(verbs) {
		v := verbs[i]
		i++
		if v == VerbMove {
			break // the next contour started, so this one has no close
		}
		if v == VerbClose {
			return true
		}
	}
	return false
}

// autoClose synthesizes the closing verb (a line back to moveTo, or a bare close if already there).
func (it *Iter) autoClose(pts *[4]geom.Point) Verb {
	if it.lastPt != it.moveTo {
		// A special case: if both points are NaN, an equality compare returns false, but the iterator expects that they
		// are treated as the same.
		if math.IsNaN(float64(it.lastPt.X)) || math.IsNaN(float64(it.lastPt.Y)) ||
			math.IsNaN(float64(it.moveTo.X)) || math.IsNaN(float64(it.moveTo.Y)) {
			return VerbClose
		}
		pts[0] = it.lastPt
		pts[1] = it.moveTo
		it.lastPt = it.moveTo
		it.closeLine = true
		return VerbLine
	}
	pts[0] = it.moveTo
	return VerbClose
}

// Next returns the next verb and its points (with the segment start point in pts[0]), or ok == false when done. For a
// conic, call ConicWeight for its weight.
func (it *Iter) Next() (verb Verb, pts [4]geom.Point, ok bool) {
	if it.verbIdx >= len(it.p.verbs) {
		// Close the curve if requested and if there is some curve to close.
		if it.needClose {
			if it.autoClose(&pts) == VerbLine {
				return VerbLine, pts, true
			}
			it.needClose = false
			return VerbClose, pts, true
		}
		return 0, pts, false
	}
	verb = it.p.verbs[it.verbIdx]
	it.verbIdx++
	points := it.p.points
	switch verb {
	case VerbMove:
		if it.needClose {
			it.verbIdx-- // move back one verb
			verb = it.autoClose(&pts)
			if verb == VerbClose {
				it.needClose = false
			}
			return verb, pts, true
		}
		if it.verbIdx >= len(it.p.verbs) { // might be a trailing moveto
			return 0, pts, false
		}
		it.moveTo = points[it.ptIdx]
		pts[0] = it.moveTo
		it.ptIdx++
		it.lastPt = it.moveTo
		it.needClose = it.forceClose
	case VerbLine:
		pts[0] = it.lastPt
		pts[1] = points[it.ptIdx]
		it.lastPt = points[it.ptIdx]
		it.closeLine = false
		it.ptIdx++
	case VerbConic, VerbQuad:
		if verb == VerbConic {
			it.weightIdx++
		}
		pts[0] = it.lastPt
		pts[1] = points[it.ptIdx]
		pts[2] = points[it.ptIdx+1]
		it.lastPt = points[it.ptIdx+1]
		it.ptIdx += 2
	case VerbCubic:
		pts[0] = it.lastPt
		pts[1] = points[it.ptIdx]
		pts[2] = points[it.ptIdx+1]
		pts[3] = points[it.ptIdx+2]
		it.lastPt = points[it.ptIdx+2]
		it.ptIdx += 3
	case VerbClose:
		verb = it.autoClose(&pts)
		if verb == VerbLine {
			it.verbIdx-- // move back one verb
		} else {
			it.needClose = false
		}
		it.lastPt = it.moveTo
	}
	return verb, pts, true
}

// RawIter iterates the raw verb stream: no cleanup, no injected verbs. For line/quad/conic/cubic records, pts[0] is the
// segment's start point; close records carry no points. PeekVerb exposes the next verb without consuming it, which the
// hairline converter uses for cap extension decisions.
type RawIter struct {
	it rawIter
}

// NewRawIter returns a raw iterator over p.
func NewRawIter(p *Path) *RawIter {
	return &RawIter{it: newRawIter(p)}
}

// Next returns the next verb record, with the conic weight when verb == VerbConic (1 otherwise).
func (r *RawIter) Next() (verb Verb, pts [4]geom.Point, weight float32, ok bool) {
	return r.it.next()
}

// PeekVerb returns the next verb without consuming it; ok is false at the end of the stream.
func (r *RawIter) PeekVerb() (verb Verb, ok bool) {
	if r.it.verbIdx >= len(r.it.p.verbs) {
		return 0, false
	}
	return r.it.p.verbs[r.it.verbIdx], true
}
