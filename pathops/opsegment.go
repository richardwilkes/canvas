// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The segment construction surface: a monotonic curve of a contour holding the span list threaded from head (t==0) to
// tail (t==1). This slice builds the data model and inserts intersection t values: init,
// addLine/addQuad/addConic/addCubic, addT, insert, subDivide, the geometry accessors, the point-matching helpers, and
// the angle-loop build/sort (calcAngles, sortAngles). The walking/winding/coincidence machinery (activeOp, findNext*,
// moveNearby, …) lives with its consumers.

package pathops

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// opSegment is a monotonic curve of a contour: the span list runs from head (t==0) to tail (t==1). head/tail are stored
// by value; pts is a slice into the point array owned by the edge builder's arena that may be tweaked in place.
type opSegment struct {
	tail        opSpanBase    // fTail: t==1
	contour     *opContour    // fContour
	next        *opSegment    // fNext: forward-only list the contour walks
	prev        *opSegment    // fPrev
	pts         []geom.Point  // fPts
	head        opSpan        // fHead: t==0
	count       int           // fCount: number of spans (one for a non-intersecting segment)
	doneCount   int           // fDoneCount
	bounds      pathOpsBounds // fBounds: tight bounds
	weight      float32       // fWeight
	verb        path.Verb     // fVerb
	visitedFlag bool          // fVisited: used by missing-coincidence check (see visited/resetVisited)
}

// visited reports whether the segment was already visited, setting the flag on the first call. missingCoincidence uses
// it to act only on the second encounter of an opposite segment.
func (s *opSegment) visited() bool {
	if !s.visitedFlag {
		s.visitedFlag = true
		return false
	}
	return true
}

// resetVisited clears the visited flag.
func (s *opSegment) resetVisited() { s.visitedFlag = false }

func (s *opSegment) globalState() *opGlobalState { return s.contour.state }
func (s *opSegment) operand() bool               { return s.contour.operand }
func (s *opSegment) oppXor() bool                { return s.contour.oppXor }
func (s *opSegment) bumpCount()                  { s.count++ }
func (s *opSegment) setNext(next *opSegment)     { s.next = next }
func (s *opSegment) setPrev(prev *opSegment)     { s.prev = prev }

// done reports whether every span in this segment has been processed.
func (s *opSegment) done() bool { return s.doneCount == s.count }

// markDone flags one span done, bumping the done count.
func (s *opSegment) markDone(span *opSpan) {
	if span.done {
		return
	}
	span.setDone(true)
	s.doneCount++
}

// markAllDone flags every span done, marking the segment collapsed.
func (s *opSegment) markAllDone() {
	span := &s.head
	for span != nil {
		s.markDone(span)
		next := span.next
		if next == nil {
			break
		}
		span = next.upCastable()
	}
}

// release accounts for a span removed from the list, adjusting the done and total span counts.
func (s *opSegment) release(span *opSpan) {
	if span.done {
		s.doneCount--
	}
	s.count--
}

// isHorizontal reports whether the segment's bounds have zero height.
func (s *opSegment) isHorizontal() bool { return s.bounds.top == s.bounds.bottom }

// isVertical reports whether the segment's bounds have zero width.
func (s *opSegment) isVertical() bool { return s.bounds.left == s.bounds.right }

// lastPt returns the segment's final point, at t==1.
func (s *opSegment) lastPt() geom.Point { return s.pts[opVerbToPoints(s.verb)] }

// ptAtT returns the point on the curve at parameter t.
func (s *opSegment) ptAtT(t float64) geom.Point { return curvePointAtT(s.verb, s.pts, s.weight, t) }

// dPtAtT returns the double-precision point on the curve at parameter t.
func (s *opSegment) dPtAtT(t float64) dPoint { return curveDPointAtT(s.verb, s.pts, s.weight, t) }

// dSlopeAtT returns the curve's tangent vector at parameter t, computed in double precision.
func (s *opSegment) dSlopeAtT(t float64) dVector { return curveDSlopeAtT(s.verb, s.pts, s.weight, t) }

// init sets up the segment's head (t==0) and tail (t==1) spans from the given points.
func (s *opSegment) init(pts []geom.Point, weight float32, contour *opContour, verb path.Verb) {
	s.contour = contour
	s.next = nil
	s.pts = pts
	s.weight = weight
	s.verb = verb
	s.count = 0
	s.doneCount = 0
	s.visitedFlag = false
	zeroSpan := &s.head
	zeroSpan.init(s, nil, 0, pts[0])
	oneSpan := &s.tail
	zeroSpan.setNext(oneSpan)
	oneSpan.initBase(s, zeroSpan, 1, pts[opVerbToPoints(verb)])
}

// addLine initializes the segment as a line and computes its bounds.
func (s *opSegment) addLine(pts []geom.Point, parent *opContour) *opSegment {
	s.init(pts, 1, parent, path.VerbLine)
	s.bounds.setBoundsPoints(pts[:2])
	return s
}

// addQuad initializes the segment as a quadratic and computes its tight bounds.
func (s *opSegment) addQuad(pts []geom.Point, parent *opContour) *opSegment {
	s.init(pts, 1, parent, path.VerbQuad)
	var q dQuad
	q.set([3]geom.Point{pts[0], pts[1], pts[2]})
	var r dRect
	r.setBoundsQuadFull(q)
	s.bounds.setFromDRect(r)
	return s
}

// addConic initializes the segment as a conic and computes its tight bounds.
func (s *opSegment) addConic(pts []geom.Point, weight float32, parent *opContour) *opSegment {
	s.init(pts, weight, parent, path.VerbConic)
	var c dConic
	c.set([3]geom.Point{pts[0], pts[1], pts[2]}, weight)
	var r dRect
	r.setBoundsConicFull(c)
	s.bounds.setFromDRect(r)
	return s
}

// addCubic initializes the segment as a cubic and computes its tight bounds.
func (s *opSegment) addCubic(pts []geom.Point, parent *opContour) *opSegment {
	s.init(pts, 1, parent, path.VerbCubic)
	var c dCubic
	c.set([4]geom.Point{pts[0], pts[1], pts[2], pts[3]})
	var r dRect
	r.setBoundsCubicFull(c)
	s.bounds.setFromDRect(r)
	return s
}

// insert allocates a new span between prev and prev's next.
func (s *opSegment) insert(prev *opSpan) *opSpan {
	s.globalState().setAllocatedOpSpan()
	result := &opSpan{}
	next := prev.next
	result.setPrev(prev)
	prev.setNext(&result.opSpanBase)
	result.setNext(next)
	if next != nil {
		next.setPrev(result)
	}
	return result
}

// addTPt returns the existing pt-t at t (or a matching point on a curve), else inserts a new span in sorted t order.
// Returns nil on a malformed insert.
func (s *opSegment) addTPt(t float64, pt geom.Point) *opPtT {
	spanBase := &s.head.opSpanBase
	for {
		result := &spanBase.ptT
		if t == result.t || (!zeroOrOne(t) && s.match(result, s, t, pt)) {
			spanBase.bumpSpanAdds()
			return result
		}
		if t < result.t {
			prev := spanBase.prev
			if prev == nil {
				return nil
			}
			span := s.insert(prev)
			span.init(s, prev, t, pt)
			span.bumpSpanAdds()
			return &span.ptT
		}
		if spanBase == &s.tail {
			return nil
		}
		spanBase = spanBase.upCast().next
	}
}

// addT computes the point at t and adds it via addTPt.
func (s *opSegment) addT(t float64) *opPtT { return s.addTPt(t, s.ptAtT(t)) }

// match reports whether testT/testPt coincides with base's pt-t.
func (s *opSegment) match(base *opPtT, testParent *opSegment, testT float64, testPt geom.Point) bool {
	if s == testParent {
		if preciselyEqual(base.t, testT) {
			return true
		}
	}
	if !dPointsApproximatelyEqual(testPt, base.pt) {
		return false
	}
	return s != testParent || !s.ptsDisjoint(base.t, base.pt, testT, testPt)
}

// ptsDisjoint reports whether a curve loops back so that two very different t values share nearly the same point.
// Computed in float32 precision, matching the curve's native point type.
func (s *opSegment) ptsDisjoint(t1 float64, pt1 geom.Point, t2 float64, pt2 geom.Point) bool {
	if s.verb == path.VerbLine {
		return false
	}
	midT := (t1 + t2) / 2
	midPt := s.ptAtT(midT)
	seDistSq := max(pt1.DistanceToSqd(pt2)*2, float32(fltEpsilon)*2)
	return midPt.DistanceToSqd(pt1) > seDistSq || midPt.DistanceToSqd(pt2) > seDistSq
}

// subDivide fills edge with the curve running between the start and end spans (endpoints from the spans' points,
// interior control points recovered from the segment's own geometry). Returns true when a non-line curve was computed.
// Consumed by opAngle.setSpans.
func (s *opSegment) subDivide(start, end *opSpanBase, edge *dCurve) bool {
	startPtT := &start.ptT
	endPtT := &end.ptT
	edge.pts[0].set(startPtT.pt)
	points := opVerbToPoints(s.verb)
	edge.pts[points].set(endPtT.pt)
	if s.verb == path.VerbLine {
		return false
	}
	startT := startPtT.t
	endT := endPtT.t
	if (startT == 0 || endT == 0) && (startT == 1 || endT == 1) {
		// don't compute midpoints if we already have them
		switch s.verb {
		case path.VerbQuad:
			edge.pts[1].set(s.pts[1])
			return false
		case path.VerbConic:
			edge.pts[1].set(s.pts[1])
			edge.weight = s.weight
			return false
		default: // cubic
			if startT == 0 {
				edge.pts[1].set(s.pts[1])
				edge.pts[2].set(s.pts[2])
				return false
			}
			edge.pts[1].set(s.pts[2])
			edge.pts[2].set(s.pts[1])
			return false
		}
	}
	switch s.verb {
	case path.VerbQuad:
		var q dQuad
		q.set([3]geom.Point{s.pts[0], s.pts[1], s.pts[2]})
		edge.pts[1] = q.subDivideCtrl(edge.pts[0], edge.pts[2], startT, endT)
	case path.VerbConic:
		var c dConic
		c.set([3]geom.Point{s.pts[0], s.pts[1], s.pts[2]}, s.weight)
		mid, w := c.subDivideAC(startT, endT)
		edge.pts[1] = mid
		edge.weight = w
	default: // cubic
		var c dCubic
		c.set([4]geom.Point{s.pts[0], s.pts[1], s.pts[2], s.pts[3]})
		mid := c.subDivideAD(edge.pts[0], edge.pts[3], startT, endT)
		edge.pts[1] = mid[0]
		edge.pts[2] = mid[1]
	}
	return true
}

// addStartSpan builds the angle sweeping forward from the head.
func (s *opSegment) addStartSpan() *opAngle {
	angle := &opAngle{}
	angle.set(&s.head.opSpanBase, s.head.next)
	s.head.setToAngle(angle)
	return angle
}

// addEndSpan builds the angle sweeping back from the tail.
func (s *opSegment) addEndSpan() *opAngle {
	angle := &opAngle{}
	angle.set(&s.tail, &s.tail.prev.opSpanBase)
	s.tail.setFromAngle(angle)
	return angle
}

// calcAngles builds a from-angle and to-angle for every active span so the intersection-point sort (sortAngles) can
// order the segments meeting at each pt-t.
func (s *opSegment) calcAngles() {
	activePrior := !s.head.isCanceled()
	if activePrior && !s.head.simple() {
		s.addStartSpan()
	}
	prior := &s.head
	spanBase := s.head.next
	for spanBase != &s.tail {
		if activePrior {
			priorAngle := &opAngle{}
			priorAngle.set(spanBase, &prior.opSpanBase)
			spanBase.setFromAngle(priorAngle)
		}
		span := spanBase.upCast()
		active := !span.isCanceled()
		next := span.next
		if active {
			angle := &opAngle{}
			angle.set(&span.opSpanBase, next)
			span.setToAngle(angle)
		}
		activePrior = active
		prior = span
		spanBase = next
	}
	if activePrior && !s.tail.simple() {
		s.addEndSpan()
	}
}

// sortAngles merges, at each span, the local from/to angles with every angle that shares the span's pt-t loop into a
// single sorted angle loop. Returns false on the safety-net trip or an insert failure.
func (s *opSegment) sortAngles() bool {
	span := &s.head.opSpanBase
	for {
		fromAngle := span.fromAngle()
		var toAngle *opAngle
		if !span.final() {
			toAngle = span.upCast().toAngle()
		}
		if fromAngle != nil || toAngle != nil {
			baseAngle := fromAngle
			if fromAngle != nil && toAngle != nil {
				if !fromAngle.insert(toAngle) {
					return false
				}
			} else if fromAngle == nil {
				baseAngle = toAngle
			}
			ptT := span.ptTPtr()
			stopPtT := ptT
			safetyNet := 1000
			for {
				safetyNet--
				if safetyNet == 0 {
					return false
				}
				oSpan := ptT.span
				if oSpan != span {
					oAngle := oSpan.fromAngle()
					if oAngle != nil && !oAngle.loopContains(baseAngle) {
						baseAngle.insert(oAngle)
					}
					if !oSpan.final() {
						oAngle = oSpan.upCast().toAngle()
						if oAngle != nil && !oAngle.loopContains(baseAngle) {
							baseAngle.insert(oAngle)
						}
					}
				}
				ptT = ptT.next
				if ptT == stopPtT {
					break
				}
			}
			if baseAngle.loopCount() == 1 {
				span.setFromAngle(nil)
				if toAngle != nil {
					span.upCast().setToAngle(nil)
				}
				baseAngle = nil
			}
			_ = baseAngle
		}
		if span.final() {
			break
		}
		span = span.upCast().next
	}
	return true
}

// containsT reports whether newT is aliased somewhere in this segment's pt-t loops (true only after intersections have
// added aliases).
func (s *opSegment) containsT(newT float64) bool {
	spanBase := &s.head.opSpanBase
	for {
		if spanBase.ptT.containsSegT(s, newT) {
			return true
		}
		if spanBase == &s.tail {
			break
		}
		spanBase = spanBase.upCast().next
	}
	return false
}

// isXor reports whether the segment's contour uses XOR fill semantics.
func (s *opSegment) isXor() bool { return s.contour.xor }

// existing returns the pt-t already present at t (matching this segment's point and, when opp is non-nil, sharing a
// pt-t loop entry on opp), else nil.
func (s *opSegment) existing(t float64, opp *opSegment) *opPtT {
	test := &s.head.opSpanBase
	pt := s.ptAtT(t)
	var testPtT *opPtT
	for {
		testPtT = &test.ptT
		if testPtT.t == t {
			break
		}
		if !s.match(testPtT, s, t, pt) {
			if t < testPtT.t {
				return nil
			}
		} else {
			if opp == nil {
				return testPtT
			}
			matched := false
			for loop := testPtT.next; loop != testPtT; loop = loop.next {
				if loop.segment() == s && loop.t == t && loop.pt == pt {
					matched = true
					break
				}
			}
			if !matched {
				return nil
			}
			break
		}
		next := test.upCast()
		if next == nil {
			return nil // defensive: the tail is reached only for malformed input
		}
		test = next.next
	}
	if opp != nil && test.containsSegment(opp) == nil {
		return nil
	}
	return testPtT
}

// collapsed reports whether any span reports the [s, e] range collapsed.
func (s *opSegment) collapsed(startT, endT float64) spanCollapsed {
	span := &s.head.opSpanBase
	for {
		result := span.collapsed(startT, endT)
		if result != spanCollapsedNo {
			return result
		}
		next := span.upCastable()
		if next == nil {
			break
		}
		span = next.next
	}
	return spanCollapsedNo
}

// isClose reports whether the perpendicular to this segment at t meets opp at (nearly) the same point — used to decide
// whether an adjacent span extends a coincident run.
func (s *opSegment) isClose(t float64, opp *opSegment) bool {
	cPt := s.dPtAtT(t)
	dxdy := s.dSlopeAtT(t)
	perp := dLine{pts: [2]dPoint{cPt, {x: cPt.x + dxdy.y, y: cPt.y - dxdy.x}}}
	i := newIntersections()
	curveIntersectRay(opp.verb, opp.pts, opp.weight, perp, i)
	used := i.usedCount()
	for index := 0; index < used; index++ {
		if cPt.roughlyEqual(i.pt(index)) {
			return true
		}
	}
	return false
}

// addExpanded breaks this segment at newT (unless already present) and loops the new pt-t into test's opposite span, so
// the coincident part does not change the angle of the remainder. startOver is set when a new span had to be allocated.
// Returns false on a malformed insert.
func (s *opSegment) addExpanded(newT float64, test *opSpanBase, startOver *bool) bool {
	if s.containsT(newT) {
		return true
	}
	s.globalState().resetAllocatedOpSpan()
	if !between(0, newT, 1) {
		return false
	}
	newPtT := s.addT(newT)
	if s.globalState().allocatedOpSpan {
		*startOver = true
	}
	if newPtT == nil {
		return false
	}
	newPtT.pt = s.ptAtT(newT)
	oppPrev := test.ptT.oppPrev(newPtT)
	if oppPrev != nil {
		test.mergeMatches(newPtT.span)
		test.ptT.addOpp(newPtT, oppPrev)
		test.checkForCollapsedCoincidence()
	}
	return true
}
