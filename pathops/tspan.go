// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The span layer the tSect curve/curve solver sits on: tCoincident (a point's perpendicular-projection match onto the
// opposite curve) and tSpan (one [startT,endT] sub-range of a curve during binary subdivision, carrying the subdivided
// curve part, its bounds, and its doubly-linked list of "bounded" opposite spans it may still intersect). These consume
// the tCurve wrapper (tcurve.go) and are consumed in turn by the tSect solver body (tsect.go). Allocation is plain Go
// allocation with no debug validate()/dump machinery. Pathops computes in double precision throughout.

package pathops

import "math"

// tCoincident is the perpendicular dropped from a point on one curve onto the opposite curve, recording whether it
// lands back on the originating point (a coincidence) and, if so, the opposite curve's T and point there.
type tCoincident struct {
	perpPt   dPoint
	perpTVal float64 // perpendicular intersection T on the opposite curve
	matched  bool
}

// newTCoincident returns a tCoincident with no match recorded yet.
func newTCoincident() tCoincident {
	var c tCoincident
	c.init()
	return c
}

// init resets c to record no match.
func (c *tCoincident) init() {
	c.perpTVal = -1
	c.matched = false
	c.perpPt = dPoint{x: math.NaN(), y: math.NaN()}
}

// isMatch reports whether the perpendicular landed back on the originating point.
func (c *tCoincident) isMatch() bool { return c.matched }

// markCoincident flags c as matched, clearing any stale perpendicular T if it wasn't already matched.
func (c *tCoincident) markCoincident() {
	if !c.matched {
		c.perpTVal = -1
	}
	c.matched = true
}

// perpPoint returns the point where the perpendicular crossed the opposite curve.
func (c *tCoincident) perpPoint() dPoint { return c.perpPt }

// perpT returns the opposite curve's T where the perpendicular crossed it.
func (c *tCoincident) perpT() float64 { return c.perpTVal }

// setPerp drops a perpendicular from cPt (oriented by c1's tangent at t) and ray-intersects the opposite curve c2,
// keeping the closest crossing and recording whether it lands back on cPt.
func (c *tCoincident) setPerp(c1 tCurve, t float64, cPt dPoint, c2 tCurve) {
	dxdy := c1.dxdyAtT(t)
	perp := dLine{pts: [2]dPoint{cPt, {x: cPt.x + dxdy.y, y: cPt.y - dxdy.x}}}
	i := newIntersections()
	used := i.intersectRayTCurve(c2, perp)
	// only keep closest
	if used == 0 || used == 3 {
		c.init()
		return
	}
	c.perpTVal = i.tVal(0, 0)
	c.perpPt = i.pt(0)
	if used == 2 {
		distSq := c.perpPt.sub(cPt).lengthSquared()
		dist2Sq := i.pt(1).sub(cPt).lengthSquared()
		if dist2Sq < distSq {
			c.perpTVal = i.tVal(0, 1)
			c.perpPt = i.pt(1)
		}
	}
	c.matched = cPt.approximatelyEqual(c.perpPt)
}

// tSpanBounded is a node in a span's singly-linked list of opposite spans it may still intersect.
type tSpanBounded struct {
	bounded *tSpan
	next    *tSpanBounded
}

// tSpan is one [startT,endT] sub-range of a curve during binary subdivision.
type tSpan struct {
	part      tCurve
	prev      *tSpan
	next      *tSpan
	bounded   *tSpanBounded
	bounds    dRect
	coinEnd   tCoincident
	coinStart tCoincident
	startT    float64
	endT      float64
	boundsMax float64
	collapsed bool
	hasPerp   bool
	isLinear  bool
	isLine    bool
	deleted   bool
}

// newTSpan creates a tSpan wrapping a fresh blank curve of the same concrete type as curve, with coinStart and coinEnd
// initialized to record no match (rather than relying on Go's all-zero value).
func newTSpan(curve tCurve) *tSpan {
	return &tSpan{
		part:      curve.makeCurve(),
		coinStart: newTCoincident(),
		coinEnd:   newTCoincident(),
	}
}

// pointFirst returns the span's first control point.
func (s *tSpan) pointFirst() dPoint { return s.part.pointAt(0) }

// pointLast returns the span's last control point.
func (s *tSpan) pointLast() dPoint { return s.part.pointAt(s.part.pointLast()) }

// isBounded reports whether this span has any bounded (cross-linked) opposite spans.
func (s *tSpan) isBounded() bool { return s.bounded != nil }

// contains reports whether t lands in this span or any span after it.
func (s *tSpan) contains(t float64) bool {
	for work := s; work != nil; work = work.next {
		if between(work.startT, t, work.endT) {
			return true
		}
	}
	return false
}

// addBounded prepends span to this span's bounded list.
func (s *tSpan) addBounded(span *tSpan) {
	s.bounded = &tSpanBounded{bounded: span, next: s.bounded}
}

// closestBoundedT returns the T (start or end of some bounded span) whose point is nearest pt.
func (s *tSpan) closestBoundedT(pt dPoint) float64 {
	result := -1.0
	closest := math.MaxFloat64
	for testBounded := s.bounded; testBounded != nil; testBounded = testBounded.next {
		test := testBounded.bounded
		startDist := test.pointFirst().distanceSquared(pt)
		if closest > startDist {
			closest = startDist
			result = test.startT
		}
		endDist := test.pointLast().distanceSquared(pt)
		if closest > endDist {
			closest = endDist
			result = test.endT
		}
	}
	return result
}

// findOppSpan returns opp if it appears in this span's bounded list, else nil.
func (s *tSpan) findOppSpan(opp *tSpan) *tSpan {
	for bounded := s.bounded; bounded != nil; bounded = bounded.next {
		if opp == bounded.bounded {
			return bounded.bounded
		}
	}
	return nil
}

// oppT returns the bounded span containing t.
func (s *tSpan) oppT(t float64) *tSpan {
	for bounded := s.bounded; bounded != nil; bounded = bounded.next {
		test := bounded.bounded
		if between(test.startT, t, test.endT) {
			return test
		}
	}
	return nil
}

// hasOppT reports whether some bounded span contains t.
func (s *tSpan) hasOppT(t float64) bool { return s.oppT(t) != nil }

// findOppT returns the bounded span containing t, if any.
func (s *tSpan) findOppT(t float64) *tSpan { return s.oppT(t) }

// markCoincident flags both of this span's coincidence records as matched.
func (s *tSpan) markCoincident() {
	s.coinStart.markCoincident()
	s.coinEnd.markCoincident()
}

// reset clears the span's bounded list.
func (s *tSpan) reset() { s.bounded = nil }

// resetBounds clears the linear/line flags and recomputes the span's bounds from curve.
func (s *tSpan) resetBounds(curve tCurve) {
	s.isLinear = false
	s.isLine = false
	s.initBounds(curve)
}

// init sets up a span covering the whole curve c ([0,1]), with no links and no bounded spans.
func (s *tSpan) init(c tCurve) {
	s.prev = nil
	s.next = nil
	s.startT = 0
	s.endT = 1
	s.bounded = nil
	s.resetBounds(c)
}

// initBounds subdivides c over [startT,endT] into part, then recomputes the bounds and collapse/max caches. Returns
// whether the resulting bounds are valid.
func (s *tSpan) initBounds(c tCurve) bool {
	if math.IsNaN(s.startT) || math.IsNaN(s.endT) {
		return false
	}
	c.subDivide(s.startT, s.endT, s.part)
	s.part.setBounds(&s.bounds) // setBounds fully (re)initializes s.bounds
	s.coinStart.init()
	s.coinEnd.init()
	s.boundsMax = math.Max(s.bounds.width(), s.bounds.height())
	s.collapsed = s.part.collapsed()
	s.hasPerp = false
	s.deleted = false
	return s.bounds.valid()
}

// linearIntersects treats this (near-linear) span's extreme points as a line and tests which side of it q2's points
// fall on. Returns 0 (all one side), 1 (crosses or exactly on), or 3 (approximately on).
func (s *tSpan) linearIntersects(q2 tCurve) int {
	// the outside points are usually the extremes
	start, end := 0, s.part.pointLast()
	if !s.part.controlsInside() {
		dist := 0.0 // if there's any question, compute distance to find best outsiders
		for outer := 0; outer < s.part.pointCount()-1; outer++ {
			for inner := outer + 1; inner < s.part.pointCount(); inner++ {
				test := s.part.pointAt(outer).sub(s.part.pointAt(inner)).lengthSquared()
				if dist > test {
					continue
				}
				dist = test
				start = outer
				end = inner
			}
		}
	}
	// see if q2 is on one side of the line formed by the extreme points
	origX := s.part.pointAt(start).x
	origY := s.part.pointAt(start).y
	adj := s.part.pointAt(end).x - origX
	opp := s.part.pointAt(end).y - origY
	maxPart := math.Max(math.Abs(adj), math.Abs(opp))
	sign := 0.0 // initialization to shut up warning in release build
	for n := 0; n < q2.pointCount(); n++ {
		qn := q2.pointAt(n)
		dx := qn.y - origY
		dy := qn.x - origX
		maxVal := math.Max(maxPart, math.Max(math.Abs(dx), math.Abs(dy)))
		test := (qn.y-origY)*adj - (qn.x-origX)*opp
		if preciselyZeroWhenComparedTo(test, maxVal) {
			return 1
		}
		if approximatelyZeroWhenComparedTo(test, maxVal) {
			return 3
		}
		if n == 0 {
			sign = test
			continue
		}
		if test*sign < 0 {
			return 1
		}
	}
	return 0
}

// linearsIntersect reports whether this and span's near-linear extreme-point chords cross.
func (s *tSpan) linearsIntersect(span *tSpan) bool {
	result := s.linearIntersects(span.part)
	if result <= 1 {
		return result != 0
	}
	result = span.linearIntersects(s.part)
	return result != 0
}

// onlyEndPointsInCommon reports whether this and opp share exactly one end point and diverge from it (every "other"
// control-point pair pointing away from the shared base).
func (s *tSpan) onlyEndPointsInCommon(opp *tSpan, start, oppStart, ptsInCommon *bool) bool {
	switch {
	case opp.pointFirst().equals(s.pointFirst()):
		*start, *oppStart = true, true
	case opp.pointFirst().equals(s.pointLast()):
		*start, *oppStart = false, true
	case opp.pointLast().equals(s.pointFirst()):
		*start, *oppStart = true, false
	case opp.pointLast().equals(s.pointLast()):
		*start, *oppStart = false, false
	default:
		*ptsInCommon = false
		return false
	}
	*ptsInCommon = true
	baseIndex := 0
	if !*start {
		baseIndex = s.part.pointLast()
	}
	otherPts := s.part.otherPts(baseIndex)
	oppBaseIndex := 0
	if !*oppStart {
		oppBaseIndex = opp.part.pointLast()
	}
	oppOtherPts := opp.part.otherPts(oppBaseIndex)
	base := s.part.pointAt(baseIndex)
	for o1 := 0; o1 < s.part.pointCount()-1; o1++ {
		v1 := otherPts[o1].sub(base)
		for o2 := 0; o2 < opp.part.pointCount()-1; o2++ {
			v2 := oppOtherPts[o2].sub(base)
			if v2.dot(v1) >= 0 {
				return false
			}
		}
	}
	return true
}

// hullCheck classifies how this span's convex hull relates to opp's. Returns 0 if no hull intersection, 1 if the hulls
// intersect, 2 if they only share a common endpoint, and -1 if linear and further checking is required.
func (s *tSpan) hullCheck(opp *tSpan, start, oppStart *bool) int {
	if s.isLinear {
		return -1
	}
	var ptsInCommon bool
	if s.onlyEndPointsInCommon(opp, start, oppStart, &ptsInCommon) {
		return 2
	}
	var linear bool
	if s.part.hullIntersectsCurve(opp.part, &linear) {
		if !linear { // check set true if linear
			return 1
		}
		s.isLinear = true
		s.isLine = s.part.controlsInside()
		if ptsInCommon {
			return 1
		}
		return -1
	}
	// hull is not linear; check set true if intersected at the end points
	return b2i(ptsInCommon) << 1 // 0 or 2
}

// hullsIntersect classifies how this span's and opp's convex hulls relate, checking bounds first as a cheap rejection.
func (s *tSpan) hullsIntersect(opp *tSpan, start, oppStart *bool) int {
	if !s.bounds.intersects(opp.bounds) {
		return 0
	}
	if hullSect := s.hullCheck(opp, start, oppStart); hullSect >= 0 {
		return hullSect
	}
	if hullSect := opp.hullCheck(s, oppStart, start); hullSect >= 0 {
		return hullSect
	}
	return -1
}

// removeAllBounded unlinks this span from every span in its bounded list, reporting whether this span should now be
// deleted.
func (s *tSpan) removeAllBounded() bool {
	deleteSpan := false
	for bounded := s.bounded; bounded != nil; bounded = bounded.next {
		opp := bounded.bounded
		if opp.removeBounded(s) {
			deleteSpan = true
		}
	}
	return deleteSpan
}

// removeBounded unlinks opp from this span's bounded list (and drops the coincidence caches if the perpendiculars no
// longer bracket a bounded span). Returns whether this span should now be deleted (its bounded list became empty).
func (s *tSpan) removeBounded(opp *tSpan) bool {
	if s.hasPerp {
		foundStart := false
		foundEnd := false
		for bounded := s.bounded; bounded != nil; bounded = bounded.next {
			test := bounded.bounded
			if opp != test {
				if between(test.startT, s.coinStart.perpT(), test.endT) {
					foundStart = true
				}
				if between(test.startT, s.coinEnd.perpT(), test.endT) {
					foundEnd = true
				}
			}
		}
		if !foundStart || !foundEnd {
			s.hasPerp = false
			s.coinStart.init()
			s.coinEnd.init()
		}
	}
	var prev *tSpanBounded
	for bounded := s.bounded; bounded != nil; {
		boundedNext := bounded.next
		if opp == bounded.bounded {
			if prev != nil {
				prev.next = boundedNext
				return false
			}
			s.bounded = boundedNext
			return s.bounded == nil
		}
		prev = bounded
		bounded = boundedNext
	}
	return false
}

// split splits work at its midpoint into this span.
func (s *tSpan) split(work *tSpan) bool {
	return s.splitAt(work, (work.startT+work.endT)*0.5)
}

// splitAt makes this span the [t, work.endT] half and shrinks work to [work.startT, t], relinking the list and copying
// work's bounded list to both halves. Returns false if either half collapsed to a point.
func (s *tSpan) splitAt(work *tSpan, t float64) bool {
	s.startT = t
	s.endT = work.endT
	if s.startT == s.endT {
		s.collapsed = true
		return false
	}
	work.endT = t
	if work.startT == work.endT {
		work.collapsed = true
		return false
	}
	s.prev = work
	s.next = work.next
	s.isLinear = work.isLinear
	s.isLine = work.isLine
	work.next = s
	if s.next != nil {
		s.next.prev = s
	}
	bounded := work.bounded
	s.bounded = nil
	for bounded != nil {
		s.addBounded(bounded.bounded)
		bounded = bounded.next
	}
	for bounded = s.bounded; bounded != nil; bounded = bounded.next {
		bounded.bounded.addBounded(s)
	}
	return true
}
