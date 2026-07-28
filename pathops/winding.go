// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The pathops winding-sum assignment. Two mechanisms cooperate to give every span a winding sum so the op/simplify
// drivers can decide which edges bound the result:
//
//   - The ray-cast seed: opSpan.sortableTop projects the most-perpendicular ray from a prospective top span, gathers
//     every curve it crosses (opContour/opSegment.rayCheck over the curveIntercept dispatch), sorts the hits, and
//     accumulates the winding directly. findSortableTop drives it across the contour list. opSpan.computeWindSum
//     retries it up to maxWindingTries times.
//   - The angle-loop transfer (opSegment.computeSum/computeOneSum): once one span at a junction has a winding, the
//     sorted angle loop built by calcAngles/sortAngles lets the winding transfer counterclockwise to its neighbors via
//     markAndChaseWinding.
//
// This file also provides the opSegment winding helpers both mechanisms share (markAndChaseWinding, markWinding,
// nextChase, updateWinding family, setUpWindings, markAngle, useInnerWinding, spanSign/oppSign, spanToAngle).
//
// Pathops computes in double precision; the winding accumulators are plain int. Sorting uses an ascending
// slices.SortFunc — ties that would matter cause sortableTop to report unsortable regardless, so the exact tie order
// does not change the outcome.

package pathops

import (
	"cmp"
	"math"
	"slices"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// opRayDir is the axis and sense of a projected winding-seed ray.
type opRayDir int

const (
	rayDirLeft opRayDir = iota
	rayDirTop
	rayDirRight
	rayDirBottom
)

// rayXYIndex returns 0 for a horizontal ray (rayDirLeft/rayDirRight), 1 for a vertical ray (rayDirTop/ rayDirBottom).
func rayXYIndex(dir opRayDir) int { return int(dir) & 1 }

// ptXY returns the point coordinate the ray travels along.
func ptXY(pt geom.Point, dir opRayDir) float32 {
	if rayXYIndex(dir) != 0 {
		return pt.Y
	}
	return pt.X
}

// ptYX returns the point coordinate perpendicular to the ray (the axis-intercept fed to curveIntercept).
func ptYX(pt geom.Point, dir opRayDir) float32 {
	if rayXYIndex(dir) != 0 {
		return pt.X
	}
	return pt.Y
}

// ptDxDy returns the slope component along the ray.
func ptDxDy(v dVector, dir opRayDir) float64 {
	if rayXYIndex(dir) != 0 {
		return v.y
	}
	return v.x
}

// ptDyDx returns the slope component perpendicular to the ray.
func ptDyDx(v dVector, dir opRayDir) float64 {
	if rayXYIndex(dir) != 0 {
		return v.x
	}
	return v.y
}

// rectSide returns the bounds edge in the ray's direction (left/top/right/bottom by enum order).
func rectSide(b pathOpsBounds, dir opRayDir) float32 {
	switch dir {
	case rayDirLeft:
		return b.left
	case rayDirTop:
		return b.top
	case rayDirRight:
		return b.right
	default: // rayDirBottom
		return b.bottom
	}
}

// sidewaysOverlap reports whether pt's perpendicular coordinate lies within the bounds span perpendicular to the ray.
func sidewaysOverlap(b pathOpsBounds, pt geom.Point, dir opRayDir) bool {
	if rayXYIndex(dir) == 0 {
		// horizontal ray: check Y within [top, bottom]
		return approximatelyBetween(float64(b.top), float64(pt.Y), float64(b.bottom))
	}
	// vertical ray: check X within [left, right]
	return approximatelyBetween(float64(b.left), float64(pt.X), float64(b.right))
}

// lessThan reports whether rayDirLeft/rayDirTop compare with '<' (true) versus rayDirRight/rayDirBottom with '>'
// (false).
func lessThan(dir opRayDir) bool { return int(dir)&2 == 0 }

// ccwDxDy reports whether the slope, in the ray's frame, sweeps counterclockwise.
func ccwDxDy(v dVector, dir opRayDir) bool {
	vPartPos := ptDyDx(v, dir) > 0
	leftBottom := (int(dir)+1)&2 != 0
	return vPartPos == leftBottom
}

// getTGuess returns the tTry'th ray origin t (0.5, then a bisection ladder) plus the direction offset (0/1) that
// rotates rayDirLeft->rayDirTop / rayDirRight->rayDirBottom on odd tries.
func getTGuess(tTry int, dirOffset *int) float64 {
	t := 0.5
	*dirOffset = tTry & 1
	tBase := tTry >> 1
	tBits := 0
	for {
		tTry >>= 1
		if tTry == 0 {
			break
		}
		t /= 2
		tBits++
	}
	if tBits != 0 {
		tIndex := (tBase - 1) & ((1 << tBits) - 1)
		t += t * 2 * float64(tIndex)
	}
	return t
}

// opRayHit is one curve crossing found by the ray (span is always a non-terminal span).
type opRayHit struct {
	next  *opRayHit
	span  *opSpan
	pt    geom.Point
	t     float64
	slope dVector
	valid bool
}

// makeTestBase seeds the base hit from span at parametric fraction t and returns the initial ray direction (rayDirLeft
// if the slope is more vertical, else rayDirTop).
func (h *opRayHit) makeTestBase(span *opSpan, t float64) opRayDir {
	h.next = nil
	h.span = span
	h.t = span.t()*(1-t) + span.next.t()*t
	segment := span.segment
	h.slope = segment.dSlopeAtT(h.t)
	h.pt = segment.ptAtT(h.t)
	h.valid = true
	if math.Abs(h.slope.x) < math.Abs(h.slope.y) {
		return rayDirLeft
	}
	return rayDirTop
}

// curveIntercept computes the roots at which the verb's curve meets the axis line intercept (horizontal for xyIndex 0,
// vertical for 1). Returns the root count written into roots.
func curveIntercept(verb path.Verb, pts []geom.Point, weight float32, intercept float64, xyIndex int, roots []float64) int {
	horizontal := xyIndex == 0
	switch verb {
	case path.VerbLine:
		return lineIntercept(pts, intercept, horizontal, roots)
	case path.VerbQuad:
		return quadIntercept(pts, intercept, horizontal, roots)
	case path.VerbConic:
		return conicIntercept(pts, weight, intercept, horizontal, roots)
	case path.VerbCubic:
		return cubicIntercept(pts, intercept, horizontal, roots)
	default:
		return 0
	}
}

// lineIntercept computes the single root at which the line meets the horizontal or vertical intercept.
func lineIntercept(pts []geom.Point, intercept float64, horizontal bool, roots []float64) int {
	var l dLine
	l.set([2]geom.Point{pts[0], pts[1]})
	if horizontal {
		if pts[0].Y == pts[1].Y {
			return 0
		}
		roots[0] = horizontalIntercept(l, intercept)
	} else {
		if pts[0].X == pts[1].X {
			return 0
		}
		roots[0] = verticalIntercept(l, intercept)
	}
	if between(0, roots[0], 1) {
		return 1
	}
	return 0
}

// quadIntercept computes the roots at which the quad meets the horizontal or vertical intercept, via a
// lineQuadraticIntersections built from just the quad.
func quadIntercept(pts []geom.Point, intercept float64, horizontal bool, roots []float64) int {
	var q dQuad
	q.set([3]geom.Point{pts[0], pts[1], pts[2]})
	lq := &lineQuadraticIntersections{quad: q}
	if horizontal {
		return lq.horizontalIntersectRoots(intercept, roots)
	}
	return lq.verticalIntersectRoots(intercept, roots)
}

// conicIntercept computes the roots at which the conic meets the horizontal or vertical intercept.
func conicIntercept(pts []geom.Point, weight float32, intercept float64, horizontal bool, roots []float64) int {
	var c dConic
	c.set([3]geom.Point{pts[0], pts[1], pts[2]}, weight)
	lc := &lineConicIntersections{conic: c}
	if horizontal {
		return lc.horizontalIntersectRoots(intercept, roots)
	}
	return lc.verticalIntersectRoots(intercept, roots)
}

// cubicIntercept computes the roots at which the cubic meets the horizontal or vertical intercept.
func cubicIntercept(pts []geom.Point, intercept float64, horizontal bool, roots []float64) int {
	var c dCubic
	c.set([4]geom.Point{pts[0], pts[1], pts[2], pts[3]})
	if horizontal {
		return cubicHorizontalIntersect(c, intercept, roots)
	}
	return cubicVerticalIntersect(c, intercept, roots)
}

// windingSpanAtT returns the span whose [t, next.t) interval contains tHit, or nil when tHit lands on a span boundary
// or past the end.
func (s *opSegment) windingSpanAtT(tHit float64) *opSpan {
	span := &s.head
	for {
		next := span.next
		if approximatelyEqual(tHit, next.t()) {
			return nil
		}
		if tHit < next.t() {
			return span
		}
		if next.final() {
			return nil
		}
		span = next.upCast()
	}
}

// rayCheck gathers the crossings of this segment with the ray, adding a hit for each that survives the near-end /
// tangent / winding filters.
func (s *opSegment) rayCheck(base *opRayHit, dir opRayDir, hits **opRayHit) {
	if !sidewaysOverlap(s.bounds, base.pt, dir) {
		return
	}
	baseXY := ptXY(base.pt, dir)
	boundsXY := rectSide(s.bounds, dir)
	checkLessThan := lessThan(dir)
	if !approximatelyEqual(float64(baseXY), float64(boundsXY)) && (baseXY < boundsXY) == checkLessThan {
		return
	}
	var tVals [3]float64
	baseYX := ptYX(base.pt, dir)
	roots := curveIntercept(s.verb, s.pts, s.weight, float64(baseYX), rayXYIndex(dir), tVals[:])
	for index := 0; index < roots; index++ {
		t := tVals[index]
		if base.span.segment == s && approximatelyEqual(base.t, t) {
			continue
		}
		var slope dVector
		var pt geom.Point
		valid := false
		switch {
		case approximatelyZero(t):
			pt = s.pts[0]
		case approximatelyEqual(t, 1):
			pt = s.pts[opVerbToPoints(s.verb)]
		default:
			pt = s.ptAtT(t)
			if dPointsApproximatelyEqual(pt, base.pt) {
				if base.span.segment == s {
					continue
				}
			} else {
				ptXYv := ptXY(pt, dir)
				if !approximatelyEqual(float64(baseXY), float64(ptXYv)) && (baseXY < ptXYv) == checkLessThan {
					continue
				}
				slope = s.dSlopeAtT(t)
				if s.verb == path.VerbCubic && base.span.segment == s && roughlyEqual(base.t, t) &&
					dPointsRoughlyEqual(pt, base.pt) {
					continue
				}
				if math.Abs(ptDyDx(slope, dir)*10000) > math.Abs(ptDxDy(slope, dir)) {
					valid = true
				}
			}
		}
		span := s.windingSpanAtT(t)
		if span == nil {
			valid = false
		} else if span.windValue == 0 && span.oppValue == 0 {
			continue
		}
		newHit := &opRayHit{
			next:  *hits,
			pt:    pt,
			slope: slope,
			span:  span,
			t:     t,
			valid: valid,
		}
		*hits = newHit
	}
}

// rayCheck skips the contour if its extreme is past the ray base, else probes every segment.
func (c *opContour) rayCheck(base *opRayHit, dir opRayDir, hits **opRayHit) {
	baseXY := ptXY(base.pt, dir)
	boundsXY := rectSide(c.bounds, dir)
	checkLessThan := lessThan(dir)
	if !approximatelyEqual(float64(baseXY), float64(boundsXY)) && (baseXY < boundsXY) == checkLessThan {
		return
	}
	for testSegment := &c.head; testSegment != nil; testSegment = testSegment.next {
		testSegment.rayCheck(base, dir, hits)
	}
}

// sortableTop casts the tTry'th ray from this span, sorts the crossings along it, and accumulates the winding,
// assigning windSum/oppSum to each crossed span. Returns false if the ray is degenerate or the crossings are ambiguous
// (the caller then retries with a different ray).
func (s *opSpan) sortableTop(contourHead *opContourHead) bool {
	var dirOffset int
	t := getTGuess(s.topTTry, &dirOffset)
	s.topTTry++
	var hitBase opRayHit
	dir := hitBase.makeTestBase(s, t)
	if hitBase.slope.x == 0 && hitBase.slope.y == 0 {
		return false
	}
	hitHead := &hitBase
	dir = opRayDir(int(dir) + dirOffset)
	if hitBase.span != nil && hitBase.span.segment.verb > path.VerbLine && ptDyDx(hitBase.slope, dir) == 0 {
		return false
	}
	for contour := contourHead.listHead(); contour != nil; contour = contour.next {
		if contour.count == 0 {
			continue
		}
		contour.rayCheck(&hitBase, dir, &hitHead)
	}
	// gather hits into a slice
	var sorted []*opRayHit
	for hit := hitHead; hit != nil; hit = hit.next {
		sorted = append(sorted, hit)
	}
	count := len(sorted)
	xyIdx := rayXYIndex(dir)
	lt := lessThan(dir)
	slices.SortFunc(sorted, func(a, b *opRayHit) int {
		var av, bv float32
		if xyIdx != 0 {
			av, bv = a.pt.Y, b.pt.Y
		} else {
			av, bv = a.pt.X, b.pt.X
		}
		if !lt {
			av, bv = bv, av
		}
		return cmp.Compare(av, bv)
	})
	// verify windings
	var last *geom.Point
	wind := 0
	oppWind := 0
	for index := 0; index < count; index++ {
		hit := sorted[index]
		if !hit.valid {
			return false
		}
		ccw := ccwDxDy(hit.slope, dir)
		span := hit.span
		if span == nil {
			return false
		}
		hitSegment := span.segment
		if span.windValue == 0 && span.oppValue == 0 {
			continue
		}
		if last != nil && dPointsApproximatelyEqual(*last, hit.pt) {
			return false
		}
		if index < count-1 {
			nextPt := sorted[index+1].pt
			if dPointsApproximatelyEqual(nextPt, hit.pt) {
				return false
			}
		}
		operand := hitSegment.operand()
		if operand {
			wind, oppWind = oppWind, wind
		}
		lastWind := wind
		lastOpp := oppWind
		var windValue, oppValue int
		if ccw {
			windValue = -span.windValue
			oppValue = -span.oppValue
		} else {
			windValue = span.windValue
			oppValue = span.oppValue
		}
		wind += windValue
		oppWind += oppValue
		sumSet := false
		var windSum int
		if useInnerWinding(lastWind, wind) {
			windSum = wind
		} else {
			windSum = lastWind
		}
		if span.windSum == skMinS32 {
			span.setWindSum(windSum)
			sumSet = true
		}
		var oppSum int
		if useInnerWinding(lastOpp, oppWind) {
			oppSum = oppWind
		} else {
			oppSum = lastOpp
		}
		if span.oppSum == skMinS32 {
			span.setOppSum(oppSum)
		}
		if sumSet {
			if s.globalState().phase == opPhaseFixWinding {
				hitSegment.contour.setCcw(b2i(ccw))
			} else {
				hitSegment.markAndChaseWindingBinary(&span.opSpanBase, span.next, windSum, oppSum, nil)
				hitSegment.markAndChaseWindingBinary(span.next, &span.opSpanBase, windSum, oppSum, nil)
			}
		}
		if operand {
			wind, oppWind = oppWind, wind
		}
		pt := hit.pt
		last = &pt
		s.globalState().bumpNested()
	}
	return true
}

// findSortableTop returns the first undone span already carrying a winding, else the first for which sortableTop
// succeeds.
func (s *opSegment) findSortableTop(contourHead *opContourHead) *opSpan {
	span := &s.head
	for {
		next := span.next
		if !span.done {
			if span.windSum != skMinS32 {
				return span
			}
			if span.sortableTop(contourHead) {
				return span
			}
		}
		if next.final() {
			return nil
		}
		span = next.upCast()
	}
}

// findSortableTop searches every segment in the contour, marking the contour done when every segment is.
func (c *opContour) findSortableTop(contourHead *opContourHead) *opSpan {
	allDone := true
	if c.count != 0 {
		for testSegment := &c.head; testSegment != nil; testSegment = testSegment.next {
			if testSegment.done() {
				continue
			}
			allDone = false
			if result := testSegment.findSortableTop(contourHead); result != nil {
				return result
			}
		}
	}
	if allDone {
		c.done = true
	}
	return nil
}

// findSortableTop retries the whole contour sweep up to maxWindingTries times, since a ray from one span may seed the
// winding another span needs.
func findSortableTop(contourHead *opContourHead) *opSpan {
	for index := 0; index < maxWindingTries; index++ {
		for contour := contourHead.listHead(); contour != nil; contour = contour.next {
			if contour.done {
				continue
			}
			if result := contour.findSortableTop(contourHead); result != nil {
				return result
			}
		}
	}
	return nil
}

// computeWindSum retries sortableTop until it succeeds or the try budget is exhausted, then reports the resulting sum.
func (s *opSpan) computeWindSum() int {
	contourHead := s.globalState().contourHead
	windTry := 0
	for !s.sortableTop(contourHead) {
		windTry++
		if windTry >= maxWindingTries {
			break
		}
	}
	return s.windSum
}

// ---- opSegment winding helpers ----

// intAbs returns the absolute value of x.
func intAbs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// spanSign returns the signed wind value of the span between start and end, oriented by their t order.
func spanSign(start, end *opSpanBase) int {
	if start.t() < end.t() {
		return -start.upCast().windValue
	}
	return end.upCast().windValue
}

// oppSign returns the signed opposite-operand wind value of the span between start and end, oriented by their t order.
func oppSign(start, end *opSpanBase) int {
	if start.t() < end.t() {
		return -start.upCast().oppValue
	}
	return end.upCast().oppValue
}

// useInnerWinding reports whether the inner (smaller-magnitude, or negative-on-tie) winding should be preferred over
// the outer one.
func useInnerWinding(outerWinding, innerWinding int) bool {
	absOut := intAbs(outerWinding)
	absIn := intAbs(innerWinding)
	if absOut == absIn {
		return outerWinding < 0
	}
	return absOut < absIn
}

// windSum returns the wind sum of angle's starter span.
func (s *opSegment) windSum(angle *opAngle) int {
	minSpan := angle.start.starter(angle.end)
	return minSpan.windSum
}

// spanToAngle returns the angle sweeping from start toward end.
func (s *opSegment) spanToAngle(start, end *opSpanBase) *opAngle {
	if start.t() < end.t() {
		return start.upCast().toAngle()
	}
	return start.fromAngle()
}

// setLast stashes endSpan into last and returns nil (stops the nextChase walk).
func setLast(last **opSpanBase, endSpan *opSpanBase) *opSegment {
	if last != nil {
		*last = endSpan
	}
	return nil
}

// nextChase advances to the segment continuing the winding run, updating start, step and the minimum span; returns nil
// (recording last) when the run ends.
func (s *opSegment) nextChase(startPtr **opSpanBase, stepPtr *int, minPtr **opSpan, last **opSpanBase) *opSegment {
	origStart := *startPtr
	step := *stepPtr
	var endSpan *opSpanBase
	if step > 0 {
		endSpan = origStart.upCast().next
	} else {
		endSpan = &origStart.prev.opSpanBase
	}
	var angle *opAngle
	if step > 0 {
		angle = endSpan.fromAngle()
	} else {
		angle = endSpan.upCast().toAngle()
	}
	var foundSpan, otherEnd *opSpanBase
	var other *opSegment
	if angle == nil {
		if endSpan.t() != 0 && endSpan.t() != 1 {
			return nil
		}
		otherPtT := endSpan.ptT.next
		other = otherPtT.segment()
		foundSpan = otherPtT.span
		// Either direction may run off the end of the opposite segment's span list: forward when the opposite pt-t is
		// that segment's tail (t==1, not up-castable), backward when it is that segment's head (t==0, no prev). Both
		// leave otherEnd nil for the check below to stop the chase.
		if step > 0 {
			if up := foundSpan.upCastable(); up != nil {
				otherEnd = up.next
			}
		} else if prev := foundSpan.prev; prev != nil {
			otherEnd = &prev.opSpanBase
		}
	} else {
		if angle.loopCount() > 2 {
			return setLast(last, endSpan)
		}
		next := angle.next
		if next == nil {
			return nil
		}
		other = next.segment()
		endSpan = next.start
		foundSpan = next.start
		otherEnd = next.end
	}
	if otherEnd == nil {
		return nil
	}
	foundStep := foundSpan.step(otherEnd)
	if *stepPtr != foundStep {
		return setLast(last, endSpan)
	}
	var origMin *opSpan
	if step < 0 {
		origMin = origStart.prev
	} else {
		origMin = origStart.upCast()
	}
	foundMin := foundSpan.starter(otherEnd)
	if foundMin.windValue != origMin.windValue || foundMin.oppValue != origMin.oppValue {
		return setLast(last, endSpan)
	}
	*startPtr = foundSpan
	*stepPtr = foundStep
	if minPtr != nil {
		*minPtr = foundMin
	}
	return other
}

// markWindingUnary sets span's wind sum to winding, unless the span is already done.
func (s *opSegment) markWindingUnary(span *opSpan, winding int) bool {
	if span.done {
		return false
	}
	span.setWindSum(winding)
	return true
}

// markWindingBinary sets span's wind sum and opposite-operand wind sum, unless the span is already done.
func (s *opSegment) markWindingBinary(span *opSpan, winding, oppWinding int) bool {
	if span.done {
		return false
	}
	span.setWindSum(winding)
	span.setOppSum(oppWinding)
	return true
}

// markAndChaseWindingUnary marks the winding on the start/end span, then chases nextChase to propagate it along the run
// of spans sharing the same winding.
func (s *opSegment) markAndChaseWindingUnary(start, end *opSpanBase, winding int, lastPtr **opSpanBase) bool {
	spanStart := start.starter(end)
	step := start.step(end)
	success := s.markWindingUnary(spanStart, winding)
	var last *opSpanBase
	other := s
	safetyNet := 1000
	for {
		other = other.nextChase(&start, &step, &spanStart, &last)
		if other == nil {
			break
		}
		safetyNet--
		if safetyNet == 0 {
			return false
		}
		if spanStart.windSum != skMinS32 {
			break
		}
		other.markWindingUnary(spanStart, winding)
	}
	if lastPtr != nil {
		*lastPtr = last
	}
	return success
}

// markAndChaseWindingBinary marks the winding and opposite-operand winding on the start/end span, then chases nextChase
// to propagate them along the run of spans sharing the same winding.
func (s *opSegment) markAndChaseWindingBinary(start, end *opSpanBase, winding, oppWinding int, lastPtr **opSpanBase) bool {
	spanStart := start.starter(end)
	step := start.step(end)
	success := s.markWindingBinary(spanStart, winding, oppWinding)
	var last *opSpanBase
	other := s
	safetyNet := 1000
	for {
		other = other.nextChase(&start, &step, &spanStart, &last)
		if other == nil {
			break
		}
		safetyNet--
		if safetyNet == 0 {
			return false
		}
		if spanStart.windSum != skMinS32 {
			if s.operand() == other.operand() {
				if spanStart.windSum != winding || spanStart.oppSum != oppWinding {
					// The already-assigned sums disagree with the ones being chased, so the angle sort produced an
					// inconsistent walk. There is no abort path for that: stop chasing and report success, leaving
					// whichever sums were assigned first in place.
					return true
				}
			} else {
				if spanStart.windSum != oppWinding {
					return false
				}
				if spanStart.oppSum != winding {
					return false
				}
			}
			break
		}
		if s.operand() == other.operand() {
			other.markWindingBinary(spanStart, winding, oppWinding)
		} else {
			other.markWindingBinary(spanStart, oppWinding, winding)
		}
	}
	if lastPtr != nil {
		*lastPtr = last
	}
	return success
}

// markAngleUnary picks the inner winding of maxWinding/sumWinding and marks it along angle's span run.
func (s *opSegment) markAngleUnary(maxWinding, sumWinding int, angle *opAngle, result **opSpanBase) bool {
	if useInnerWinding(maxWinding, sumWinding) {
		maxWinding = sumWinding
	}
	return s.markAndChaseWindingUnary(angle.start, angle.end, maxWinding, result)
}

// markAngleBinary picks the inner winding of each of the primary and opposite-operand winding pairs and marks them
// along angle's span run.
func (s *opSegment) markAngleBinary(maxWinding, sumWinding, oppMaxWinding, oppSumWinding int, angle *opAngle, result **opSpanBase) bool {
	if useInnerWinding(maxWinding, sumWinding) {
		maxWinding = sumWinding
	}
	if oppMaxWinding != oppSumWinding && useInnerWinding(oppMaxWinding, oppSumWinding) {
		oppMaxWinding = oppSumWinding
	}
	return s.markAndChaseWindingBinary(angle.start, angle.end, maxWinding, oppMaxWinding, result)
}

// setUpWindingsUnary derives maxWinding and sumWinding for the start/end span from the running total sumMiWinding, then
// updates sumMiWinding by the span's signed contribution.
func (s *opSegment) setUpWindingsUnary(start, end *opSpanBase, sumMiWinding, maxWinding, sumWinding *int) {
	deltaSum := spanSign(start, end)
	*maxWinding = *sumMiWinding
	*sumMiWinding -= deltaSum
	*sumWinding = *sumMiWinding
}

// setUpWindingsBinary derives maxWinding/sumWinding and oppMaxWinding/oppSumWinding for the start/end span from the
// running totals sumMiWinding and sumSuWinding, swapping which running total is "this operand's" versus "the opposite
// operand's" depending on s.operand().
func (s *opSegment) setUpWindingsBinary(start, end *opSpanBase, sumMiWinding, sumSuWinding, maxWinding, sumWinding, oppMaxWinding, oppSumWinding *int) {
	deltaSum := spanSign(start, end)
	oppDeltaSum := oppSign(start, end)
	if s.operand() {
		*maxWinding = *sumSuWinding
		*sumSuWinding -= deltaSum
		*sumWinding = *sumSuWinding
		*oppMaxWinding = *sumMiWinding
		*sumMiWinding -= oppDeltaSum
		*oppSumWinding = *sumMiWinding
	} else {
		*maxWinding = *sumMiWinding
		*sumMiWinding -= deltaSum
		*sumWinding = *sumMiWinding
		*oppMaxWinding = *sumSuWinding
		*sumSuWinding -= oppDeltaSum
		*oppSumWinding = *sumSuWinding
	}
}

// updateOppWinding returns the opposite-operand winding at the start/end span, adjusted by the span's signed
// opposite-operand contribution when that keeps the inner winding.
func (s *opSegment) updateOppWinding(start, end *opSpanBase) int {
	lesser := start.starter(end)
	oppWinding := lesser.oppSum
	oppSpanWinding := oppSign(start, end)
	if oppSpanWinding != 0 && useInnerWinding(oppWinding-oppSpanWinding, oppWinding) && oppWinding != skMaxS32 {
		oppWinding -= oppSpanWinding
	}
	return oppWinding
}

// updateOppWindingAngle returns updateOppWinding(angle.end, angle.start).
func (s *opSegment) updateOppWindingAngle(angle *opAngle) int {
	return s.updateOppWinding(angle.end, angle.start)
}

// updateOppWindingReverse returns updateOppWinding(angle.start, angle.end).
func (s *opSegment) updateOppWindingReverse(angle *opAngle) int {
	return s.updateOppWinding(angle.start, angle.end)
}

// updateWinding computes the span's winding, seeding it via a ray cast (computeWindSum) if it has none yet.
func (s *opSegment) updateWinding(start, end *opSpanBase) int {
	lesser := start.starter(end)
	winding := lesser.windSum
	if winding == skMinS32 {
		winding = lesser.computeWindSum()
	}
	if winding == skMinS32 {
		return winding
	}
	spanWinding := spanSign(start, end)
	if winding != 0 && useInnerWinding(winding-spanWinding, winding) && winding != skMaxS32 {
		winding -= spanWinding
	}
	return winding
}

// updateWindingAngle returns updateWinding(angle.end, angle.start).
func (s *opSegment) updateWindingAngle(angle *opAngle) int {
	return s.updateWinding(angle.end, angle.start)
}

// updateWindingReverse returns updateWinding(angle.start, angle.end).
func (s *opSegment) updateWindingReverse(angle *opAngle) int {
	return s.updateWinding(angle.start, angle.end)
}

// computeOneSum transfers the winding from baseAngle's segment onto nextAngle (counterclockwise sweep) and marks
// nextAngle's spans.
func computeOneSum(baseAngle, nextAngle *opAngle, includeTyp includeType) bool {
	baseSegment := baseAngle.segment()
	sumMiWinding := baseSegment.updateWindingReverse(baseAngle)
	var sumSuWinding int
	binary := includeTyp >= includeBinarySingle
	if binary {
		sumSuWinding = baseSegment.updateOppWindingReverse(baseAngle)
		if baseSegment.operand() {
			sumMiWinding, sumSuWinding = sumSuWinding, sumMiWinding
		}
	}
	nextSegment := nextAngle.segment()
	var maxWinding, sumWinding int
	var last *opSpanBase
	if binary {
		var oppMaxWinding, oppSumWinding int
		nextSegment.setUpWindingsBinary(nextAngle.start, nextAngle.end, &sumMiWinding, &sumSuWinding,
			&maxWinding, &sumWinding, &oppMaxWinding, &oppSumWinding)
		if !nextSegment.markAngleBinary(maxWinding, sumWinding, oppMaxWinding, oppSumWinding, nextAngle, &last) {
			return false
		}
	} else {
		nextSegment.setUpWindingsUnary(nextAngle.start, nextAngle.end, &sumMiWinding, &maxWinding, &sumWinding)
		if !nextSegment.markAngleUnary(maxWinding, sumWinding, nextAngle, &last) {
			return false
		}
	}
	nextAngle.setLastMarked(last)
	return true
}

// computeOneSumReverse is the clockwise-sweep transfer used by the reverse pass of computeSum.
func computeOneSumReverse(baseAngle, nextAngle *opAngle, includeTyp includeType) bool {
	baseSegment := baseAngle.segment()
	sumMiWinding := baseSegment.updateWindingAngle(baseAngle)
	var sumSuWinding int
	binary := includeTyp >= includeBinarySingle
	if binary {
		sumSuWinding = baseSegment.updateOppWindingAngle(baseAngle)
		if baseSegment.operand() {
			sumMiWinding, sumSuWinding = sumSuWinding, sumMiWinding
		}
	}
	nextSegment := nextAngle.segment()
	var maxWinding, sumWinding int
	var last *opSpanBase
	if binary {
		var oppMaxWinding, oppSumWinding int
		nextSegment.setUpWindingsBinary(nextAngle.end, nextAngle.start, &sumMiWinding, &sumSuWinding,
			&maxWinding, &sumWinding, &oppMaxWinding, &oppSumWinding)
		if !nextSegment.markAngleBinary(maxWinding, sumWinding, oppMaxWinding, oppSumWinding, nextAngle, &last) {
			return false
		}
	} else {
		nextSegment.setUpWindingsUnary(nextAngle.end, nextAngle.start, &sumMiWinding, &maxWinding, &sumWinding)
		if !nextSegment.markAngleUnary(maxWinding, sumWinding, nextAngle, &last) {
			return false
		}
	}
	nextAngle.setLastMarked(last)
	return true
}

// computeSum transfers winding sums counterclockwise (then, if needed, clockwise) between adjacent orderable angles in
// the span's already-sorted (or unorderable) angle loop. Returns the resulting windSum of the start->end span (skNaN32
// when the loop is degenerate).
func (s *opSegment) computeSum(start, end *opSpanBase, includeTyp includeType) int {
	firstAngle := s.spanToAngle(end, start)
	if firstAngle == nil || firstAngle.next == nil {
		return skNaN32
	}
	var baseAngle *opAngle
	tryReverse := false
	// look for counterclockwise transfers
	angle := firstAngle.previous()
	next := angle.next
	firstAngle = next
	for {
		prior := angle
		angle = next
		next = angle.next
		switch {
		case prior.unorderable || angle.unorderable || next.unorderable:
			baseAngle = nil
		case angle.starter().windSum != skMinS32:
			baseAngle = angle
			tryReverse = true
		case baseAngle != nil:
			computeOneSum(baseAngle, angle, includeTyp)
			if angle.starter().windSum != skMinS32 {
				baseAngle = angle
			} else {
				baseAngle = nil
			}
		}
		if next == firstAngle {
			break
		}
	}
	if baseAngle != nil && firstAngle.starter().windSum == skMinS32 {
		firstAngle = baseAngle
		tryReverse = true
	}
	if tryReverse {
		baseAngle = nil
		prior := firstAngle
		for {
			angle = prior
			prior = angle.previous()
			next = angle.next
			switch {
			case prior.unorderable || angle.unorderable || next.unorderable:
				baseAngle = nil
			case angle.starter().windSum != skMinS32:
				baseAngle = angle
			case baseAngle != nil:
				computeOneSumReverse(baseAngle, angle, includeTyp)
				if angle.starter().windSum != skMinS32 {
					baseAngle = angle
				} else {
					baseAngle = nil
				}
			}
			if prior == firstAngle {
				break
			}
		}
	}
	return start.starter(end).windSum
}
