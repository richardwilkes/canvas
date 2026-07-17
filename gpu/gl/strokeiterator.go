// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// strokeIterator iterates over the stroke geometry defined by a path and stroke. It automatically converts closes and
// square caps to lines, and round caps to circles, so the user doesn't have to worry about it. At each location it
// provides a verb and "prevVerb" so there is context about the preceding join. It decodes the path's records through
// path.RawIter up front (supplying the close verbs' current point, which the raw iterator omits) and queues point
// slices, allocating fresh slices for the geometry that gets defined implicitly by the path (close lines and square-cap
// lines).
//
// Usage:
//
//	iter := newStrokeIterator(path, stroke, viewMatrix)
//	for iter.next() { // Call next() first.
//	    iter.verb(); iter.pts(); iter.w(); iter.prevVerb(); iter.prevPts()
//	}

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// strokeIterVerb identifies the kind of stroke segment or control event produced by strokeIterator. The geometric
// values coincide with path.Verb's so casts stay value-preserving.
type strokeIterVerb uint8

// strokeIterVerb values.
const (
	// Verbs that describe stroke geometry.
	strokeIterVerbLine                  = strokeIterVerb(path.VerbLine)
	strokeIterVerbQuad                  = strokeIterVerb(path.VerbQuad)
	strokeIterVerbConic                 = strokeIterVerb(path.VerbConic)
	strokeIterVerbCubic                 = strokeIterVerb(path.VerbCubic)
	strokeIterVerbCircle strokeIterVerb = iota + 1 // stroke-width circle as a 180-degree point stroke

	// Helper verbs that notify callers to update their own iteration state.
	strokeIterVerbMoveWithinContour
	strokeIterVerbContourFinished
)

// strokeIterVerbIsGeometric reports whether v describes actual stroke geometry, as opposed to a helper verb that only
// notifies the caller of contour-iteration state.
func strokeIterVerbIsGeometric(v strokeIterVerb) bool { return v < strokeIterVerbMoveWithinContour }

// strokeIterVerbPtCount returns the number of points for the geometric verbs (the only callers pass
// line/quad/conic/cubic).
func strokeIterVerbPtCount(v strokeIterVerb) int {
	switch v {
	case strokeIterVerbLine:
		return 2
	case strokeIterVerbQuad, strokeIterVerbConic:
		return 3
	case strokeIterVerbCubic:
		return 4
	default:
		panic("strokeIterVerbPtCount requires a line/quad/conic/cubic verb")
	}
}

const strokeIterQueueBufferCount = 8

// strokeIterator is the iterator state described in the package comment above.
type strokeIterator struct {
	stroke *stroke.Rec
	// Info from the original path and stroke.
	viewMatrix             *geom.Matrix // For hairlines.
	ptsQueue               [strokeIterQueueBufferCount][]geom.Point
	lastDegenerateStrokePt []geom.Point // nil when unset
	records                []tessContourRecord
	firstPtsInContour      []geom.Point
	recordIdx              int
	queueFrontIdx          int
	queueCount             int
	wQueue                 [strokeIterQueueBufferCount]float32
	firstWInContour        float32
	// The queue is implemented as a roll-over array with a floating front index.
	verbs [strokeIterQueueBufferCount]strokeIterVerb
	// Info for the current contour we are iterating.
	firstVerbInContour strokeIterVerb
}

// newStrokeIterator creates a stroke iterator over p's stroke geometry.
func newStrokeIterator(p *path.Path, rec *stroke.Rec, viewMatrix *geom.Matrix) *strokeIterator {
	s := &strokeIterator{viewMatrix: viewMatrix, stroke: rec}
	// Decode the records up front. The raw iterator supplies the segment start point for curve verbs but no points for
	// closes; track the current point so close records carry it in pts[0].
	it := path.NewRawIter(p)
	var lastPt geom.Point
	for {
		verb, pts, w, ok := it.Next()
		if !ok {
			break
		}
		rec := tessContourRecord{verb: verb, pts: pts, weight: w}
		switch verb {
		case path.VerbMove:
			lastPt = pts[0]
		case path.VerbLine:
			lastPt = pts[1]
		case path.VerbQuad, path.VerbConic:
			lastPt = pts[2]
		case path.VerbCubic:
			lastPt = pts[3]
		case path.VerbClose:
			rec.pts[0] = lastPt
		}
		s.records = append(s.records, rec)
	}
	return s
}

// next must be called first; it loads the next pair of "prev" and "current" strokes. Returns false when iteration is
// complete.
func (s *strokeIterator) next() bool {
	if s.queueCount != 0 {
		if s.queueCount < 2 {
			panic("stroke iterator queue underflow")
		}
		s.popFront()
		if s.queueCount >= 2 {
			return true
		}
		if s.atVerb(0) == strokeIterVerbContourFinished {
			// Don't let "kContourFinished" be prevVerb at the start of the next contour.
			s.queueCount = 0
		}
	}
	for ; s.recordIdx < len(s.records); s.recordIdx++ {
		rec := &s.records[s.recordIdx]
		switch rec.verb {
		case path.VerbMove:
			if !s.finishOpenContour() {
				continue
			}
		case path.VerbCubic, path.VerbConic, path.VerbQuad, path.VerbLine:
			// A verb is degenerate when all of its points beyond the first are equal to it ("p3 == p2 && p2 == p1 && p1
			// == p0" for cubics, and so on down).
			n := strokeIterVerbPtCount(strokeIterVerb(rec.verb))
			degenerate := true
			for i := n - 1; i >= 1; i-- {
				if rec.pts[i] != rec.pts[i-1] {
					degenerate = false
					break
				}
			}
			if degenerate {
				s.lastDegenerateStrokePt = rec.pts[:1]
				continue
			}
			s.enqueue(strokeIterVerb(rec.verb), rec.pts[:n], rec.weight)
			if s.queueCount == 1 {
				// Defer the first verb until the end when we know what it's joined to.
				s.firstVerbInContour = strokeIterVerb(rec.verb)
				s.firstPtsInContour = rec.pts[:n]
				s.firstWInContour = rec.weight
				continue
			}
		case path.VerbClose:
			if s.queueCount == 0 {
				s.lastDegenerateStrokePt = rec.pts[:1]
				continue
			}
			if rec.pts[0] != s.firstPtsInContour[0] {
				// Draw a line back to the contour's starting point.
				s.enqueue(strokeIterVerbLine,
					[]geom.Point{rec.pts[0], s.firstPtsInContour[0]}, 1)
			}
			// Repeat the first verb, this time as the "current" stroke instead of the prev.
			s.enqueue(s.firstVerbInContour, s.firstPtsInContour, s.firstWInContour)
			s.enqueue(strokeIterVerbContourFinished, nil, 1)
			s.lastDegenerateStrokePt = nil
		}
		if s.queueCount < 2 {
			panic("stroke iterator expected a queued pair")
		}
		s.recordIdx++
		return true
	}
	return s.finishOpenContour()
}

func (s *strokeIterator) prevVerb() strokeIterVerb { return s.atVerb(0) }

func (s *strokeIterator) verb() strokeIterVerb { return s.atVerb(1) }
func (s *strokeIterator) pts() []geom.Point    { return s.atPts(1) }
func (s *strokeIterator) w() float32           { return s.atW(1) }

func (s *strokeIterator) atVerb(i int) strokeIterVerb {
	if i < 0 || i >= s.queueCount {
		panic("stroke iterator verb index out of range")
	}
	return s.verbs[(s.queueFrontIdx+i)&(strokeIterQueueBufferCount-1)]
}

func (s *strokeIterator) backVerb() strokeIterVerb { return s.atVerb(s.queueCount - 1) }

func (s *strokeIterator) atPts(i int) []geom.Point {
	if i < 0 || i >= s.queueCount {
		panic("stroke iterator pts index out of range")
	}
	return s.ptsQueue[(s.queueFrontIdx+i)&(strokeIterQueueBufferCount-1)]
}

func (s *strokeIterator) backPts() []geom.Point { return s.atPts(s.queueCount - 1) }

func (s *strokeIterator) atW(i int) float32 {
	if i < 0 || i >= s.queueCount {
		panic("stroke iterator w index out of range")
	}
	return s.wQueue[(s.queueFrontIdx+i)&(strokeIterQueueBufferCount-1)]
}

func (s *strokeIterator) enqueue(verb strokeIterVerb, pts []geom.Point, w float32) {
	if s.queueCount >= strokeIterQueueBufferCount {
		panic("stroke iterator queue overflow")
	}
	i := (s.queueFrontIdx + s.queueCount) & (strokeIterQueueBufferCount - 1)
	s.verbs[i] = verb
	s.ptsQueue[i] = pts
	s.wQueue[i] = w
	s.queueCount++
}

func (s *strokeIterator) popFront() {
	if s.queueCount <= 0 {
		panic("stroke iterator queue underflow")
	}
	s.queueFrontIdx++
	s.queueCount--
}

// finishOpenContour finishes the current contour without closing it, enqueueing any necessary caps as well as the
// contour's first stroke that we deferred at the beginning. Returns false and makes no changes if the current contour
// was already finished.
func (s *strokeIterator) finishOpenContour() bool {
	switch {
	case s.queueCount != 0:
		switch bv := s.backVerb(); bv {
		case strokeIterVerbLine, strokeIterVerbQuad, strokeIterVerbConic, strokeIterVerbCubic:
		default:
			panic("stroke iterator back verb must be geometric")
		}
		switch s.stroke.Cap() {
		case stroke.CapButt:
			// There are no caps, but inject a "move" so the first stroke doesn't get joined with the end of the contour
			// when it's processed.
			s.enqueue(strokeIterVerbMoveWithinContour, s.firstPtsInContour, s.firstWInContour)
		case stroke.CapRound:
			// The "kCircle" verb serves as our barrier to prevent the first stroke from getting joined with the end of
			// the contour. We just need to make sure that the first point of the contour goes last.
			backIdx := strokeIterVerbPtCount(s.backVerb()) - 1
			s.enqueue(strokeIterVerbCircle, s.backPts()[backIdx:backIdx+1], 1)
			s.enqueue(strokeIterVerbCircle, s.firstPtsInContour, s.firstWInContour)
		case stroke.CapSquare:
			endingCapPts, beginningCapPts := s.fillSquareCapPoints()
			// Append the ending cap onto the current contour.
			s.enqueue(strokeIterVerbLine, endingCapPts, 1)
			// Move to the beginning cap and append it right before (and joined to) the first stroke (that we will add
			// below).
			s.enqueue(strokeIterVerbMoveWithinContour, beginningCapPts, 1)
			s.enqueue(strokeIterVerbLine, beginningCapPts, 1)
		}
	case s.lastDegenerateStrokePt != nil:
		// queueCount == 0 means this subpath is zero length. Generate caps on its location:
		//
		//	"Any zero length subpath ... shall be stroked if the 'stroke-linecap' property has a
		//	value of round or square producing respectively a circle or a square."
		//
		//	(https://www.w3.org/TR/SVG11/painting.html#StrokeProperties)
		switch s.stroke.Cap() {
		case stroke.CapButt:
			// Zero-length contour with butt caps. There are no caps and no first stroke to generate.
			return false
		case stroke.CapRound:
			s.enqueue(strokeIterVerbCircle, s.lastDegenerateStrokePt, 1)
			// Setting the "first" stroke as the circle causes it to be added again below, this time as the "current"
			// stroke.
			s.firstVerbInContour = strokeIterVerbCircle
			s.firstPtsInContour = s.lastDegenerateStrokePt
			s.firstWInContour = 1
		case stroke.CapSquare:
			var outset geom.Point
			if !s.stroke.IsHairlineStyle() {
				// Implement degenerate square caps as a stroke-width square in path space.
				outset = geom.Point{X: s.stroke.Width() * 0.5, Y: 0}
			} else {
				// If the stroke is hairline, draw a 1x1 device-space square instead. This is equivalent to using:
				//
				//	outset = inverse(viewMatrix).mapVector(.5, 0)
				//
				// And since the matrix cannot have perspective, we only need to invert the upper 2x2 of the viewMatrix
				// to achieve this.
				if s.viewMatrix.HasPerspective() {
					panic("stroke iterator requires a non-perspective view matrix")
				}
				a, b := s.viewMatrix.Get(geom.MScaleX), s.viewMatrix.Get(geom.MSkewX)
				c, d := s.viewMatrix.Get(geom.MSkewY), s.viewMatrix.Get(geom.MScaleY)
				det := a*d - b*c
				if det > 0 {
					// outset = inverse(|a b|) * |.5|
					//                  |c d|    | 0|
					//
					//     == 1/det * | d -b| * |.5|
					//                |-c  a|   | 0|
					//
					//     == | d| * .5/det
					//        |-c|
					outset = geom.Point{X: d * (0.5 / det), Y: -c * (0.5 / det)}
				} else {
					outset = geom.Point{X: 1, Y: 0}
				}
			}
			p := s.lastDegenerateStrokePt[0]
			endingCapPts := []geom.Point{
				{X: p.X - outset.X, Y: p.Y - outset.Y},
				{X: p.X + outset.X, Y: p.Y + outset.Y},
			}
			// Add the square first as the "prev" join.
			s.enqueue(strokeIterVerbLine, endingCapPts, 1)
			s.enqueue(strokeIterVerbMoveWithinContour, endingCapPts, 1)
			// Setting the "first" stroke as the square causes it to be added again below, this time as the "current"
			// stroke.
			s.firstVerbInContour = strokeIterVerbLine
			s.firstPtsInContour = endingCapPts
			s.firstWInContour = 1
		}
	default:
		// This contour had no lines, beziers, or "close" verbs. There are no caps and no first stroke to generate.
		return false
	}

	// Repeat the first verb, this time as the "current" stroke instead of the prev.
	s.enqueue(s.firstVerbInContour, s.firstPtsInContour, s.firstWInContour)
	s.enqueue(strokeIterVerbContourFinished, nil, 1)
	s.lastDegenerateStrokePt = nil
	return true
}

// fillSquareCapPoints finds the endpoints for the two extra line verbs used to implement square caps.
func (s *strokeIterator) fillSquareCapPoints() (endingCapPts, beginningCapPts []geom.Point) {
	// Find the endpoints of the cap at the end of the contour.
	var lastTangent geom.Point
	lastPts := s.backPts()
	lastVerb := s.backVerb()
	// Take the last non-zero tangent, walking backward through the control points.
	switch lastVerb {
	case strokeIterVerbCubic:
		lastTangent = geom.Point{X: lastPts[3].X - lastPts[2].X, Y: lastPts[3].Y - lastPts[2].Y}
		if lastTangent != (geom.Point{}) {
			break
		}
		fallthrough
	case strokeIterVerbConic, strokeIterVerbQuad:
		lastTangent = geom.Point{X: lastPts[2].X - lastPts[1].X, Y: lastPts[2].Y - lastPts[1].Y}
		if lastTangent != (geom.Point{}) {
			break
		}
		fallthrough
	case strokeIterVerbLine:
		lastTangent = geom.Point{X: lastPts[1].X - lastPts[0].X, Y: lastPts[1].Y - lastPts[0].Y}
		if lastTangent == (geom.Point{}) {
			panic("stroke iterator cap tangent must be non-zero")
		}
	default:
		panic("unreachable stroke iterator cap verb")
	}
	if !s.stroke.IsHairlineStyle() {
		// Extend the cap by 1/2 stroke width.
		scale := (0.5 * s.stroke.Width()) / lastTangent.Length()
		lastTangent = geom.Point{X: lastTangent.X * scale, Y: lastTangent.Y * scale}
	} else {
		// Extend the cap by what will be 1/2 pixel after transformation.
		scale := 0.5 / s.viewMatrix.MapVector(lastTangent).Length()
		lastTangent = geom.Point{X: lastTangent.X * scale, Y: lastTangent.Y * scale}
	}
	lastPoint := lastPts[strokeIterVerbPtCount(lastVerb)-1]
	endingCapPts = []geom.Point{
		lastPoint,
		{X: lastPoint.X + lastTangent.X, Y: lastPoint.Y + lastTangent.Y},
	}

	// Find the endpoints of the cap at the beginning of the contour.
	first := s.firstPtsInContour
	firstTangent := geom.Point{X: first[1].X - first[0].X, Y: first[1].Y - first[0].Y}
	if firstTangent == (geom.Point{}) {
		firstTangent = geom.Point{X: first[2].X - first[0].X, Y: first[2].Y - first[0].Y}
		if firstTangent == (geom.Point{}) {
			firstTangent = geom.Point{X: first[3].X - first[0].X, Y: first[3].Y - first[0].Y}
			if firstTangent == (geom.Point{}) {
				panic("stroke iterator first tangent must be non-zero")
			}
		}
	}
	if !s.stroke.IsHairlineStyle() {
		// Set the cap back by 1/2 stroke width.
		scale := (-0.5 * s.stroke.Width()) / firstTangent.Length()
		firstTangent = geom.Point{X: firstTangent.X * scale, Y: firstTangent.Y * scale}
	} else {
		// Set the cap back by what will be 1/2 pixel after transformation.
		scale := -0.5 / s.viewMatrix.MapVector(firstTangent).Length()
		firstTangent = geom.Point{X: firstTangent.X * scale, Y: firstTangent.Y * scale}
	}
	beginningCapPts = []geom.Point{
		{X: first[0].X + firstTangent.X, Y: first[0].Y + firstTangent.Y},
		first[0],
	}
	return endingCapPts, beginningCapPts
}
