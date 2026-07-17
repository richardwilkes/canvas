// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The sortable angle formed by a segment span, used to order every segment meeting at an intersection point
// counterclockwise so winding can be assigned. Angles are sorted counterclockwise; the smallest has a positive x and
// the smallest positive y, the largest a positive x and a zero y. The type carries the curve (offset to a shared
// origin), its sweep vectors, a 16-sector compass classification of that sweep, and the pairwise comparison machinery
// (after/orderable/endsIntersect/convexHullOverlaps/checkParallel) the sorted insert relies on.
//
// All geometry here is computed in float64 for the precision the boolean-op math needs; the sector mask is a plain
// 32-bit field. Sorting the inflection-test parameter values uses an ascending slices.Sort.

package pathops

import (
	"math"
	"slices"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

/*             quarter angle values for sector

31   x > 0, y == 0              horizontal line (to the right)
0    x > 0, y == epsilon        quad/cubic horizontal tangent eventually going +y
1    x > 0, y > 0, x > y        nearer horizontal angle
2                  x + e == y   quad/cubic 45 going horiz
3    x > 0, y > 0, x == y       45 angle
4                  x == y + e   quad/cubic 45 going vert
5    x > 0, y > 0, x < y        nearer vertical angle
6    x == epsilon, y > 0        quad/cubic vertical tangent eventually going +x
7    x == 0, y > 0              vertical line (to the top)
*/

// includeType classifies a winding computation: whether it is unary (simplify) or binary (a two-path op), and whether
// the fill is even-odd (xor). Values >= includeBinarySingle are binary.
type includeType int

const (
	includeUnaryWinding includeType = iota
	includeUnaryXor
	includeBinarySingle
	includeBinaryOpp
)

// opAngle is one segment's sweep at an intersection point, sortable against every other angle meeting there.
type opAngle struct {
	computedEnd       *opSpanBase    // the end span as last computed (may differ from end mid-computation)
	next              *opAngle       // the next angle in this intersection's sorted counterclockwise loop
	lastMarked        *opSpanBase    // the last span visited while marking windings through this loop
	start             *opSpanBase    // the span this angle sweeps from
	end               *opSpanBase    // the span this angle sweeps to
	part              dCurveSweep    // the curve from start to end, offset to a shared origin as needed
	originalCurvePart dCurve         // the curve from start to end, before any sector-length adjustment
	tangentHalf       lineParameters // used only to sort a pair of lines or line-like sections
	sectorEnd         int            // sweep end, in 32nds of a circle
	side              float64        // signed distance used to order curved sections (sign only, not scaled)
	sectorStart       int            // sweep start, in 32nds of a circle
	sectorMask        uint32         // bitmask of the 32 sedecimant sectors this sweep spans
	unorderable       bool           // true when this angle cannot be placed in the sorted loop
	computeSectorFlag bool           // true when the sector classification was deferred and still needs work
	computedSector    bool           // true once computeSector has resolved the deferred sector
	checkCoincidence  bool           // true when this angle's span pair should be checked for coincidence
	tangentsAmbiguous bool           // true when the tangent comparison could not order this angle unambiguously
}

// segment returns the segment that owns this angle's starting span.
func (a *opAngle) segment() *opSegment { return a.start.segment }

// midT returns the midpoint of the angle's span parameter range: (start.t()+end.t())/2.
func (a *opAngle) midT() float64 { return (a.start.t() + a.end.t()) / 2 }

// starter returns whichever of start/end comes first in t-order (see opSpanBase.starter).
func (a *opAngle) starter() *opSpan { return a.start.starter(a.end) }

// setLastMarked records the last span visited while marking windings through this angle's loop.
func (a *opAngle) setLastMarked(last *opSpanBase) { a.lastMarked = last }

// set installs a new start/end span pair for this angle, resetting its cached state and recomputing the curve section
// and sector classification.
func (a *opAngle) set(start, end *opSpanBase) {
	a.start = start
	a.computedEnd = end
	a.end = end
	a.next = nil
	a.computeSectorFlag = false
	a.computedSector = false
	a.checkCoincidence = false
	a.tangentsAmbiguous = false
	a.setSpans()
	a.setSector()
}

// setSpans builds the curve part, its sweep, and the side sign used to order curved sections.
func (a *opAngle) setSpans() {
	a.unorderable = false
	a.lastMarked = nil
	if a.start == nil {
		a.unorderable = true
		return
	}
	segment := a.start.segment
	pts := segment.pts
	segment.subDivide(a.start, a.end, &a.part.curve) // set at least the line part if not more
	a.originalCurvePart = a.part.curve
	verb := segment.verb
	a.part.setCurveHullSweep(verb)
	if verb != path.VerbLine && !a.part.isCurve() {
		a.part.curve.pts[1] = a.part.curve.pts[opVerbToPoints(verb)]
		a.originalCurvePart.pts[1] = a.part.curve.pts[1]
		var lineHalf dLine
		lineHalf.pts[0].set(a.part.curve.pts[0].asPoint())
		lineHalf.pts[1].set(a.part.curve.pts[1].asPoint())
		a.tangentHalf.lineEndPoints(lineHalf)
		a.side = 0
	}
	switch verb {
	case path.VerbLine:
		idx := 0
		if a.start.t() < a.end.t() {
			idx = 1
		}
		cP1 := pts[idx]
		var lineHalf dLine
		lineHalf.pts[0].set(a.start.pt())
		lineHalf.pts[1].set(cP1)
		a.tangentHalf.lineEndPoints(lineHalf)
		a.side = 0
	case path.VerbQuad, path.VerbConic:
		var tangentPart lineParameters
		tangentPart.quadEndPoints(a.part.curve.quad())
		a.side = -tangentPart.pointDistance(a.part.curve.pts[2]) // not normalized -- compare sign only
	case path.VerbCubic:
		var tangentPart lineParameters
		tangentPart.cubicPart(a.part.curve.cubic())
		a.side = -tangentPart.pointDistance(a.part.curve.pts[3])
		var testTs [4]float64
		var cub dCubic
		cub.set([4]geom.Point{pts[0], pts[1], pts[2], pts[3]})
		testCount := cub.findInflections(testTs[:]) // parameter values where curvature sign changes
		startT := a.start.t()
		endT := a.end.t()
		limitT := endT
		for index := 0; index < testCount; index++ {
			if !between(startT, testTs[index], limitT) {
				testTs[index] = -1
			}
		}
		testTs[testCount] = startT
		testCount++
		testTs[testCount] = endT
		testCount++
		slices.Sort(testTs[:testCount])
		bestSide := 0.0
		testCases := (testCount << 1) - 1
		index := 0
		for testTs[index] < 0 {
			index++
		}
		index <<= 1
		for ; index < testCases; index++ {
			testIndex := index >> 1
			testT := testTs[testIndex]
			if index&1 != 0 {
				testT = (testT + testTs[testIndex+1]) / 2
			}
			pt := curveDPointAtT(path.VerbCubic, pts, segment.weight, testT)
			var testPart lineParameters
			testPart.cubicEndPoints(a.part.curve.cubic())
			testSide := testPart.pointDistance(pt)
			if math.Abs(bestSide) < math.Abs(testSide) {
				bestSide = testSide
			}
		}
		a.side = -bestSide // compare sign only
	}
}

// setSector classifies the sweep into the 16-part (32-value) sedecimant space and builds the sector bit mask, deferring
// when the section is too short to classify.
func (a *opAngle) setSector() {
	if a.start == nil {
		a.unorderable = true
		return
	}
	verb := a.start.segment.verb
	a.sectorStart = a.findSector(verb, a.part.sweep[0].x, a.part.sweep[0].y)
	if a.sectorStart < 0 {
		a.deferSector()
		return
	}
	if !a.part.isCurve() { // if it's a line or line-like, both sectors are the same
		a.sectorEnd = a.sectorStart
		a.sectorMask = 1 << uint(a.sectorStart)
		return
	}
	a.sectorEnd = a.findSector(verb, a.part.sweep[1].x, a.part.sweep[1].y)
	if a.sectorEnd < 0 {
		a.deferSector()
		return
	}
	if a.sectorEnd == a.sectorStart && (a.sectorStart&3) != 3 { // no span => can't be an exact angle
		a.sectorMask = 1 << uint(a.sectorStart)
		return
	}
	crossesZero := a.checkCrossesZero()
	start := min(a.sectorStart, a.sectorEnd)
	curveBendsCCW := (a.sectorStart == start) != crossesZero
	// bump the start and end of the sector span if they are on exact compass points
	if (a.sectorStart & 3) == 3 {
		if curveBendsCCW {
			a.sectorStart = (a.sectorStart + 1) & 0x1f
		} else {
			a.sectorStart = (a.sectorStart + 31) & 0x1f
		}
	}
	if (a.sectorEnd & 3) == 3 {
		if curveBendsCCW {
			a.sectorEnd = (a.sectorEnd + 31) & 0x1f
		} else {
			a.sectorEnd = (a.sectorEnd + 1) & 0x1f
		}
	}
	crossesZero = a.checkCrossesZero()
	start = min(a.sectorStart, a.sectorEnd)
	end := max(a.sectorStart, a.sectorEnd)
	if !crossesZero {
		a.sectorMask = ^uint32(0) >> uint(31-end+start) << uint(start)
	} else {
		a.sectorMask = ^uint32(0)>>uint(31-start) | (^uint32(0) << uint(end))
	}
}

// deferSector marks the sector unknown until the segment length can be established (computeSector).
func (a *opAngle) deferSector() {
	a.sectorStart = -1
	a.sectorEnd = -1
	a.sectorMask = 0
	a.computeSectorFlag = true
}

/*      y<0 y==0 y>0  x<0 x==0 x>0 xy<0 xy==0 xy>0 */

// findSector maps a sweep vector into a 0..31 sector (odd values are exact compass points; -1 defers).
func (a *opAngle) findSector(verb path.Verb, x, y float64) int {
	absX := math.Abs(x)
	absY := math.Abs(y)
	var xy float64
	if verb == path.VerbLine || !almostEqualUlps(absX, absY) {
		xy = absX - absY
	}
	// sixteen sections: a space divided into 16 "sedecimant" sections
	sedecimant := [3][3][3]int{
		//       y<0           y==0           y>0
		//   x<0 x==0 x>0  x<0 x==0 x>0  x<0 x==0 x>0
		{{4, 3, 2}, {7, -1, 15}, {10, 11, 12}},  // abs(x) <  abs(y)
		{{5, -1, 1}, {-1, -1, -1}, {9, -1, 13}}, // abs(x) == abs(y)
		{{6, 3, 0}, {7, -1, 15}, {8, 11, 14}},   // abs(x) >  abs(y)
	}
	i0 := b2i(xy >= 0) + b2i(xy > 0)
	i1 := b2i(y >= 0) + b2i(y > 0)
	i2 := b2i(x >= 0) + b2i(x > 0)
	return sedecimant[i0][i1][i2]*2 + 1
}

// checkCrossesZero reports whether the sector span between sectorStart and sectorEnd wraps past sector 0 (i.e. spans
// more than half the compass).
func (a *opAngle) checkCrossesZero() bool {
	start := min(a.sectorStart, a.sectorEnd)
	end := max(a.sectorStart, a.sectorEnd)
	return end-start > 16
}

// oppositePlanes reports whether rh's sector start is at least 8 sedecimants (a quarter turn) away from this angle's.
func (a *opAngle) oppositePlanes(rh *opAngle) bool {
	startSpan := rh.sectorStart - a.sectorStart
	if startSpan < 0 {
		startSpan = -startSpan
	}
	return startSpan >= 8
}

// distEndRatio returns the segment's longest control-point-to-control-point distance divided by dist, used to gauge how
// significant a small perpendicular offset is relative to the curve's overall scale.
func (a *opAngle) distEndRatio(dist float64) float64 {
	longest := 0.0
	segment := a.segment()
	ptCount := opVerbToPoints(segment.verb)
	pts := segment.pts
	for idx1 := 0; idx1 <= ptCount-1; idx1++ {
		for idx2 := idx1 + 1; idx2 <= ptCount; idx2++ {
			if idx1 == idx2 {
				continue
			}
			v := dVector{x: float64(pts[idx2].X - pts[idx1].X), y: float64(pts[idx2].Y - pts[idx1].Y)}
			lenSq := v.lengthSquared()
			longest = max(longest, lenSq)
		}
	}
	return math.Sqrt(longest) / dist
}

// tangentsDiverge reports whether the two control tangents point far enough apart to order by them alone; sets
// tangentsAmbiguous in the borderline band.
func (a *opAngle) tangentsDiverge(rh *opAngle, s0xt0 float64) bool {
	if s0xt0 == 0 {
		return false
	}
	sweep := a.part.sweep
	tweep := rh.part.sweep
	s0dt0 := sweep[0].dot(tweep[0])
	if s0dt0 == 0 {
		return true
	}
	m := s0xt0 / s0dt0
	sDist := sweep[0].length() * m
	tDist := tweep[0].length() * m
	useS := math.Abs(sDist) < math.Abs(tDist)
	var mFactor float64
	if useS {
		mFactor = math.Abs(a.distEndRatio(sDist))
	} else {
		mFactor = math.Abs(rh.distEndRatio(tDist))
	}
	a.tangentsAmbiguous = mFactor >= 50 && mFactor < 200
	return mFactor < 50 // empirically found limit
}

// checkParallel is the last-resort ordering for tangent-parallel curves.
func (a *opAngle) checkParallel(rh *opAngle) bool {
	var scratch [2]dVector
	var sweep, tweep dVector
	if a.part.isOrdered() {
		sweep = a.part.sweep[0]
	} else {
		scratch[0] = a.part.curve.pts[1].sub(a.part.curve.pts[0])
		sweep = scratch[0]
	}
	if rh.part.isOrdered() {
		tweep = rh.part.sweep[0]
	} else {
		scratch[1] = rh.part.curve.pts[1].sub(rh.part.curve.pts[0])
		tweep = scratch[1]
	}
	s0xt0 := sweep.crossCheck(tweep)
	if a.tangentsDiverge(rh, s0xt0) {
		return s0xt0 < 0
	}
	// compute the perpendicular to the endpoints and see where it intersects the opposite curve; if the intersections
	// are within the t range, do a cross check on those
	var inside bool
	if !a.end.containsSpan(rh.end) {
		if a.endToSide(rh, &inside) {
			return inside
		}
		if rh.endToSide(a, &inside) {
			return !inside
		}
	}
	if a.midToSide(rh, &inside) {
		return inside
	}
	if rh.midToSide(a, &inside) {
		return !inside
	}
	// compute the cross check from the mid T values (last resort)
	m0 := a.segment().dPtAtT(a.midT()).sub(a.part.curve.pts[0])
	m1 := rh.segment().dPtAtT(rh.midT()).sub(rh.part.curve.pts[0])
	m0xm1 := m0.crossCheck(m1)
	if m0xm1 == 0 {
		a.unorderable = true
		rh.unorderable = true
		return true
	}
	return m0xm1 < 0
}

// computeSector lengthens a too-short section until it is long enough to classify, unless doing so would run into an
// adjacent angle.
func (a *opAngle) computeSector() bool {
	if a.computedSector {
		return !a.unorderable
	}
	a.computedSector = true
	stepUp := a.start.t() < a.end.t()
	checkEnd := a.end
	if checkEnd.final() && stepUp {
		a.unorderable = true
		return false
	}
	matched := false
	for checkEnd != nil {
		// advance end
		other := checkEnd.segment
		oSpan := &other.head.opSpanBase
		for {
			if oSpan.segment == a.segment() && oSpan != checkEnd &&
				approximatelyEqual(oSpan.t(), checkEnd.t()) {
				matched = true
				break
			}
			if oSpan.final() {
				break
			}
			oSpan = oSpan.upCast().next
		}
		if matched {
			break
		}
		if stepUp {
			if !checkEnd.final() {
				checkEnd = checkEnd.upCast().next
			} else {
				checkEnd = nil
			}
		} else {
			if checkEnd.prev != nil {
				checkEnd = &checkEnd.prev.opSpanBase
			} else {
				checkEnd = nil
			}
		}
	}
	// recomputeSector
	var computedEnd *opSpanBase
	if stepUp {
		if checkEnd != nil {
			computedEnd = &checkEnd.prev.opSpanBase
		} else {
			computedEnd = &a.end.segment.head.opSpanBase
		}
	} else {
		if checkEnd != nil {
			computedEnd = checkEnd.upCast().next
		} else {
			computedEnd = &a.end.segment.tail
		}
	}
	if checkEnd == a.end || computedEnd == a.end || computedEnd == a.start {
		a.unorderable = true
		return false
	}
	if stepUp != (a.start.t() < computedEnd.t()) {
		a.unorderable = true
		return false
	}
	saveEnd := a.end
	a.computedEnd = computedEnd
	a.end = computedEnd
	a.setSpans()
	a.setSector()
	a.end = saveEnd
	return !a.unorderable
}

// convexHullOverlaps orders two curved sweeps by convex-hull containment; -1 if their hulls overlap (need
// endsIntersect).
func (a *opAngle) convexHullOverlaps(rh *opAngle) int {
	sweep := a.part.sweep
	tweep := rh.part.sweep
	s0xs1 := sweep[0].crossCheck(sweep[1])
	s0xt0 := sweep[0].crossCheck(tweep[0])
	s1xt0 := sweep[1].crossCheck(tweep[0])
	var tBetweenS bool
	if s0xs1 > 0 {
		tBetweenS = s0xt0 > 0 && s1xt0 < 0
	} else {
		tBetweenS = s0xt0 < 0 && s1xt0 > 0
	}
	s0xt1 := sweep[0].crossCheck(tweep[1])
	s1xt1 := sweep[1].crossCheck(tweep[1])
	if s0xs1 > 0 {
		tBetweenS = tBetweenS || (s0xt1 > 0 && s1xt1 < 0)
	} else {
		tBetweenS = tBetweenS || (s0xt1 < 0 && s1xt1 > 0)
	}
	t0xt1 := tweep[0].crossCheck(tweep[1])
	if tBetweenS {
		return -1
	}
	if (s0xt0 == 0 && s1xt1 == 0) || (s1xt0 == 0 && s0xt1 == 0) { // s0 to s1 equals t0 to t1
		return -1
	}
	var sBetweenT bool
	if t0xt1 > 0 {
		sBetweenT = s0xt0 < 0 && s0xt1 > 0
	} else {
		sBetweenT = s0xt0 > 0 && s0xt1 < 0
	}
	if t0xt1 > 0 {
		sBetweenT = sBetweenT || (s1xt0 < 0 && s1xt1 > 0)
	} else {
		sBetweenT = sBetweenT || (s1xt0 > 0 && s1xt1 < 0)
	}
	if sBetweenT {
		return -1
	}
	// if all of the sweeps are in the same half plane, then the order of any pair is enough
	if s0xt0 >= 0 && s0xt1 >= 0 && s1xt0 >= 0 && s1xt1 >= 0 {
		return 0
	}
	if s0xt0 <= 0 && s0xt1 <= 0 && s1xt0 <= 0 && s1xt1 <= 0 {
		return 1
	}
	// if the outside sweeps are greater than 180 degrees, use the midpoint direction
	m0 := a.segment().dPtAtT(a.midT()).sub(a.part.curve.pts[0])
	m1 := rh.segment().dPtAtT(rh.midT()).sub(rh.part.curve.pts[0])
	m0xm1 := m0.crossCheck(m1)
	if s0xt0 > 0 && m0xm1 > 0 {
		return 0
	}
	if s0xt0 < 0 && m0xm1 < 0 {
		return 1
	}
	if a.tangentsDiverge(rh, s0xt0) {
		return b2i(s0xt0 < 0)
	}
	return b2i(m0xm1 < 0)
}

// endsIntersect orders two sections whose hulls overlap by projecting each endpoint ray onto the opposite curve and
// comparing where it lands.
func (a *opAngle) endsIntersect(rh *opAngle) bool {
	lVerb := a.segment().verb
	rVerb := rh.segment().verb
	lPts := opVerbToPoints(lVerb)
	rPts := opVerbToPoints(rVerb)
	rays := [2]dLine{
		{pts: [2]dPoint{a.part.curve.pts[0], rh.part.curve.pts[rPts]}},
		{pts: [2]dPoint{a.part.curve.pts[0], a.part.curve.pts[lPts]}},
	}
	if a.end.containsSpan(rh.end) {
		return a.checkParallel(rh)
	}
	smallTs := [2]float64{-1, -1}
	limited := [2]bool{false, false}
	for index := 0; index < 2; index++ {
		cVerb := lVerb
		if index != 0 {
			cVerb = rVerb
		}
		// if the curve is a line, then the line and the ray intersect only at their crossing
		if cVerb == path.VerbLine {
			continue
		}
		segment := a.segment()
		tStart := a.start.t()
		tEnd := a.computedEnd.t()
		if index != 0 {
			segment = rh.segment()
			tStart = rh.start.t()
			tEnd = rh.computedEnd.t()
		}
		i := newIntersections()
		curveIntersectRay(cVerb, segment.pts, segment.weight, rays[index], i)
		testAscends := tStart < tEnd
		t := 1.0
		if testAscends {
			t = 0
		}
		for idx2 := 0; idx2 < i.usedCount(); idx2++ {
			testT := i.tVal(0, idx2)
			if !approximatelyBetweenOrderable(tStart, testT, tEnd) {
				continue
			}
			if approximatelyEqualOrderable(tStart, testT) {
				continue
			}
			if testAscends {
				t = max(t, testT)
			} else {
				t = min(t, testT)
			}
			smallTs[index] = t
			limited[index] = approximatelyEqualOrderable(t, tEnd)
		}
	}
	sRayLonger := false
	var sCept dVector
	sCeptT := -1.0
	sIndex := -1
	useIntersect := false
	for index := 0; index < 2; index++ {
		if smallTs[index] < 0 {
			continue
		}
		segment := a.segment()
		if index != 0 {
			segment = rh.segment()
		}
		dPt := segment.dPtAtT(smallTs[index])
		cept := dPt.sub(rays[index].pts[0])
		// If this point is on the curve, it should have been detected earlier by ordinary curve intersection. For lines
		// the point could be close to its end, but shouldn't be near the start.
		sel := rPts
		if index != 0 {
			sel = lPts
		}
		if sel == 1 {
			total := rays[index].pts[1].sub(rays[index].pts[0])
			if cept.lengthSquared()*2 < total.lengthSquared() {
				continue
			}
		}
		end := rays[index].pts[1].sub(rays[index].pts[0])
		if cept.x*end.x < 0 || cept.y*end.y < 0 {
			continue
		}
		rayDist := cept.length()
		endDist := end.length()
		rayLonger := rayDist > endDist
		if limited[0] && limited[1] && rayLonger {
			useIntersect = true
			sRayLonger = rayLonger
			sCept = cept
			sCeptT = smallTs[index]
			sIndex = index
			break
		}
		delta := math.Abs(rayDist - endDist)
		minX := math.Inf(1)
		minY := math.Inf(1)
		maxX := math.Inf(-1)
		maxY := math.Inf(-1)
		curve := a.part.curve
		ptCount := lPts
		if index != 0 {
			curve = rh.part.curve
			ptCount = rPts
		}
		for idx2 := 0; idx2 <= ptCount; idx2++ {
			minX = min(minX, curve.pts[idx2].x)
			minY = min(minY, curve.pts[idx2].y)
			maxX = max(maxX, curve.pts[idx2].x)
			maxY = max(maxY, curve.pts[idx2].y)
		}
		maxWidth := max(maxX-minX, maxY-minY)
		delta = ieeeDoubleDivide(delta, maxWidth)
		// Empirically tuned thresholds: if translating the curve causes a control point to cross the opposite curve's
		// hull, treat the curves as parallel.
		if delta < 4e-3 && delta > 1e-3 && !useIntersect && a.part.isCurve() &&
			rh.part.isCurve() && !a.originalCurvePart.pts[0].equals(a.part.curve.pts[0]) {
			origin := rh.originalCurvePart.pts[0]
			count := opVerbToPoints(rh.segment().verb)
			line := rh.originalCurvePart.pts[count].sub(origin)
			originalSide := rh.lineOnOneSidePt(origin, line, a, true)
			if originalSide >= 0 {
				translatedSide := rh.lineOnOneSidePt(origin, line, a, false)
				if originalSide != translatedSide {
					continue
				}
			}
		}
		if delta > 1e-3 {
			useIntersect = !useIntersect
			if useIntersect {
				sRayLonger = rayLonger
				sCept = cept
				sCeptT = smallTs[index]
				sIndex = index
			}
		}
	}
	if useIntersect {
		curve := a.part.curve
		segment := a.segment()
		tStart := a.start.t()
		if sIndex != 0 {
			curve = rh.part.curve
			segment = rh.segment()
			tStart = rh.start.t()
		}
		mid := segment.dPtAtT(tStart + (sCeptT-tStart)/2).sub(curve.pts[0])
		septDir := mid.crossCheck(sCept)
		if septDir == 0 {
			return a.checkParallel(rh)
		}
		return (sRayLonger != (sIndex == 0)) != (septDir < 0)
	}
	return a.checkParallel(rh)
}

// endToSide projects a perpendicular from this end onto the opposite curve and reports which side it lands on.
func (a *opAngle) endToSide(rh *opAngle, inside *bool) bool {
	segment := a.segment()
	verb := segment.verb
	var rayEnd dLine
	rayEnd.pts[0].set(a.end.pt())
	rayEnd.pts[1] = rayEnd.pts[0]
	slopeAtEnd := curveDSlopeAtT(verb, segment.pts, segment.weight, a.end.t())
	rayEnd.pts[1].x += slopeAtEnd.y
	rayEnd.pts[1].y -= slopeAtEnd.x
	iEnd := newIntersections()
	oppSegment := rh.segment()
	oppVerb := oppSegment.verb
	curveIntersectRay(oppVerb, oppSegment.pts, oppSegment.weight, rayEnd, iEnd)
	closestEnd, endDist := iEnd.closestTo(rh.start.t(), rh.end.t(), rayEnd.pts[0])
	if closestEnd < 0 {
		return false
	}
	if endDist == 0 {
		return false
	}
	var start dPoint
	start.set(a.start.pt())
	// This max-extent-of-the-hull computation recurs several times in this file; it isn't factored out
	minX := math.Inf(1)
	minY := math.Inf(1)
	maxX := math.Inf(-1)
	maxY := math.Inf(-1)
	curve := rh.part.curve
	oppPts := opVerbToPoints(oppVerb)
	for idx2 := 0; idx2 <= oppPts; idx2++ {
		minX = min(minX, curve.pts[idx2].x)
		minY = min(minY, curve.pts[idx2].y)
		maxX = max(maxX, curve.pts[idx2].x)
		maxY = max(maxY, curve.pts[idx2].y)
	}
	maxWidth := max(maxX-minX, maxY-minY)
	endDist = ieeeDoubleDivide(endDist, maxWidth)
	if !(endDist >= 5e-12) { // empirically found; the ! also catches NaN
		return false
	}
	endPt := rayEnd.pts[0]
	oppPt := iEnd.pt(closestEnd)
	vLeft := endPt.sub(start)
	vRight := oppPt.sub(start)
	dir := vLeft.crossNoNormalCheck(vRight)
	if dir == 0 {
		return false
	}
	*inside = dir < 0
	return true
}

// midToSide projects a perpendicular from the section's midpoint chord onto both curves and compares the sides.
func (a *opAngle) midToSide(rh *opAngle, inside *bool) bool {
	segment := a.segment()
	verb := segment.verb
	startPt := a.start.pt()
	endPt := a.end.pt()
	var dStartPt dPoint
	dStartPt.set(startPt)
	var rayMid dLine
	rayMid.pts[0].x = float64((startPt.X + endPt.X) / 2)
	rayMid.pts[0].y = float64((startPt.Y + endPt.Y) / 2)
	rayMid.pts[1].x = rayMid.pts[0].x + float64(endPt.Y-startPt.Y)
	rayMid.pts[1].y = rayMid.pts[0].y - float64(endPt.X-startPt.X)
	iMid := newIntersections()
	curveIntersectRay(verb, segment.pts, segment.weight, rayMid, iMid)
	iOutside := iMid.mostOutside(a.start.t(), a.end.t(), dStartPt)
	if iOutside < 0 {
		return false
	}
	oppSegment := rh.segment()
	oppVerb := oppSegment.verb
	oppMid := newIntersections()
	curveIntersectRay(oppVerb, oppSegment.pts, oppSegment.weight, rayMid, oppMid)
	oppOutside := oppMid.mostOutside(rh.start.t(), rh.end.t(), dStartPt)
	if oppOutside < 0 {
		return false
	}
	iSide := iMid.pt(iOutside).sub(dStartPt)
	oppSide := oppMid.pt(oppOutside).sub(dStartPt)
	dir := iSide.crossCheck(oppSide)
	if dir == 0 {
		return false
	}
	*inside = dir < 0
	return true
}

// lineOnOneSidePt reports whether the opposite curve's control points all fall on one side of the line through origin
// with direction line. -1 not on one side, 0 CW, 1 CCW, -2 ambiguous.
func (a *opAngle) lineOnOneSidePt(origin dPoint, line dVector, test *opAngle, useOriginal bool) int {
	var crosses [3]float64
	testVerb := test.segment().verb
	iMax := opVerbToPoints(testVerb)
	testCurve := test.part.curve
	if useOriginal {
		testCurve = test.originalCurvePart
	}
	for index := 1; index <= iMax; index++ {
		xy1 := line.x * (testCurve.pts[index].y - origin.y)
		xy2 := line.y * (testCurve.pts[index].x - origin.x)
		if almostBequalUlps(xy1, xy2) {
			crosses[index-1] = 0
		} else {
			crosses[index-1] = xy1 - xy2
		}
	}
	if crosses[0]*crosses[1] < 0 {
		return -1
	}
	if testVerb == path.VerbCubic {
		if crosses[0]*crosses[2] < 0 || crosses[1]*crosses[2] < 0 {
			return -1
		}
	}
	if crosses[0] != 0 {
		return b2i(crosses[0] < 0)
	}
	if crosses[1] != 0 {
		return b2i(crosses[1] < 0)
	}
	if testVerb == path.VerbCubic && crosses[2] != 0 {
		return b2i(crosses[2] < 0)
	}
	return -2
}

// lineOnOneSide is lineOnOneSidePt using this angle's own line, flagging the angle unorderable on ambiguity.
func (a *opAngle) lineOnOneSide(test *opAngle, useOriginal bool) int {
	origin := a.part.curve.pts[0]
	line := a.part.curve.pts[1].sub(origin)
	result := a.lineOnOneSidePt(origin, line, test, useOriginal)
	if result == -2 {
		a.unorderable = true
		result = -1
	}
	return result
}

// linesOnOriginalSide is an experimental original-data ordering for two lines sharing a point (returns 2 for 180
// degrees apart).
func (a *opAngle) linesOnOriginalSide(test *opAngle) int {
	origin := a.originalCurvePart.pts[0]
	line := a.originalCurvePart.pts[1].sub(origin)
	var dots [2]float64
	var crosses [2]float64
	testCurve := test.originalCurvePart
	for index := 0; index < 2; index++ {
		testLine := testCurve.pts[index].sub(origin)
		xy1 := line.x * testLine.y
		xy2 := line.y * testLine.x
		dots[index] = line.x*testLine.x + line.y*testLine.y
		if almostBequalUlps(xy1, xy2) {
			crosses[index] = 0
		} else {
			crosses[index] = xy1 - xy2
		}
	}
	if crosses[0]*crosses[1] < 0 {
		return -1
	}
	if crosses[0] != 0 {
		return b2i(crosses[0] < 0)
	}
	if crosses[1] != 0 {
		return b2i(crosses[1] < 0)
	}
	if (dots[0] == 0 && dots[1] < 0) || (dots[0] < 0 && dots[1] == 0) {
		return 2 // 180 degrees apart
	}
	a.unorderable = true
	return -1
}

// alignmentSameSide reverses the previously computed order when translating a line curve to the shared origin flips
// which side of a compared line a control point is on.
func (a *opAngle) alignmentSameSide(test *opAngle, order *int) {
	if *order < 0 {
		return
	}
	if a.part.isCurve() {
		return // enabling for curves causes existing tests to fail
	}
	if test.part.isCurve() {
		return
	}
	xOrigin := test.part.curve.pts[0]
	oOrigin := test.originalCurvePart.pts[0]
	if xOrigin.equals(oOrigin) {
		return
	}
	iMax := opVerbToPoints(a.segment().verb)
	xLine := test.part.curve.pts[1].sub(xOrigin)
	oLine := test.originalCurvePart.pts[1].sub(oOrigin)
	for index := 1; index <= iMax; index++ {
		testPt := a.part.curve.pts[index]
		xCross := oLine.crossCheck(testPt.sub(xOrigin))
		oCross := xLine.crossCheck(testPt.sub(oOrigin))
		if oCross*xCross < 0 {
			*order ^= 1
			break
		}
	}
}

// orderable orders this angle vs rh; 0 == this < rh, 1 == this > rh, -1 == unorderable.
func (a *opAngle) orderable(rh *opAngle) int {
	var result int
	if !a.part.isCurve() {
		if !rh.part.isCurve() {
			leftX := a.tangentHalf.dx()
			leftY := a.tangentHalf.dy()
			rightX := rh.tangentHalf.dx()
			rightY := rh.tangentHalf.dy()
			xRy := leftX * rightY
			rxY := rightX * leftY
			if xRy == rxY {
				if leftX*rightX < 0 || leftY*rightY < 0 {
					return 1 // exactly 180 degrees apart
				}
				a.unorderable = true
				rh.unorderable = true
				return -1
			}
			if xRy < rxY {
				return 1
			}
			return 0
		}
		if result = a.lineOnOneSide(rh, false); result >= 0 {
			return result
		}
		if a.unorderable || approximatelyZero(rh.side) {
			a.unorderable = true
			rh.unorderable = true
			return -1
		}
	} else if !rh.part.isCurve() {
		if result = rh.lineOnOneSide(a, false); result >= 0 {
			if result != 0 {
				return 0
			}
			return 1
		}
		if rh.unorderable || approximatelyZero(a.side) {
			a.unorderable = true
			rh.unorderable = true
			return -1
		}
	} else if result = a.convexHullOverlaps(rh); result >= 0 {
		return result
	}
	if a.endsIntersect(rh) {
		return 1
	}
	return 0
}

// after reports whether lh < this < rh, where lh is test and rh is test's next angle in the sorted loop.
func (a *opAngle) after(test *opAngle) bool {
	lh := test
	rh := lh.next
	a.part.curve = a.originalCurvePart
	// Adjust lh and rh to share the same origin (intersection float error can offset them).
	lh.part.curve = lh.originalCurvePart
	lh.part.curve.pts[0] = a.part.curve.pts[0]
	rh.part.curve = rh.originalCurvePart
	rh.part.curve.pts[0] = a.part.curve.pts[0]
	if lh.computeSectorFlag && !lh.computeSector() {
		return true
	}
	if a.computeSectorFlag && !a.computeSector() {
		return true
	}
	if rh.computeSectorFlag && !rh.computeSector() {
		return true
	}
	ltrOverlap := (lh.sectorMask|rh.sectorMask)&a.sectorMask != 0
	lrOverlap := lh.sectorMask&rh.sectorMask != 0
	var lrOrder int // set to -1 if either order works
	if !lrOverlap { // no lh/rh sector overlap
		if !ltrOverlap { // no lh/this/rh sector overlap
			return ((lh.sectorEnd > rh.sectorStart) != (a.sectorStart > lh.sectorEnd)) !=
				(a.sectorStart > rh.sectorStart)
		}
		lrGap := (rh.sectorStart - lh.sectorStart + 32) & 0x1f
		// A tiny change can move the start +/- 4; order is determinable only outside the 12..20 gap band.
		switch {
		case lrGap > 20:
			lrOrder = 0
		case lrGap > 11:
			lrOrder = -1
		default:
			lrOrder = 1
		}
	} else {
		lrOrder = lh.orderable(rh)
		if !ltrOverlap && lrOrder >= 0 {
			return lrOrder == 0
		}
	}
	var ltOrder int
	if lh.sectorMask&a.sectorMask != 0 {
		ltOrder = lh.orderable(a)
	} else {
		ltGap := (a.sectorStart - lh.sectorStart + 32) & 0x1f
		switch {
		case ltGap > 20:
			ltOrder = 0
		case ltGap > 11:
			ltOrder = -1
		default:
			ltOrder = 1
		}
	}
	var trOrder int
	if rh.sectorMask&a.sectorMask != 0 {
		trOrder = a.orderable(rh)
	} else {
		trGap := (rh.sectorStart - a.sectorStart + 32) & 0x1f
		switch {
		case trGap > 20:
			trOrder = 0
		case trGap > 11:
			trOrder = -1
		default:
			trOrder = 1
		}
	}
	a.alignmentSameSide(lh, &ltOrder)
	a.alignmentSameSide(rh, &trOrder)
	if lrOrder >= 0 && ltOrder >= 0 && trOrder >= 0 {
		if lrOrder != 0 {
			return ltOrder&trOrder != 0
		}
		return ltOrder|trOrder != 0
	}
	// Not enough information to sort directly: pull the pairs of angles into opposite planes.
	switch {
	case ltOrder == 0 && lrOrder == 0:
		// trOrder < 0
		return lh.oppositePlanes(a)
	case ltOrder == 1 && trOrder == 0:
		// lrOrder < 0
		return a.oppositePlanes(rh)
	case lrOrder == 1 && trOrder == 1:
		// ltOrder < 0
		return lh.oppositePlanes(rh)
	}
	// If a pair couldn't be ordered, try the raw (untranslated) line data.
	if a.unorderable || lh.unorderable || rh.unorderable {
		// limit to lines; should work with curves, but wait for a failing test to verify
		if !a.part.isCurve() && !lh.part.isCurve() && !rh.part.isCurve() {
			// if two share a point, check if the third has both points in the same half plane
			ltShare := lh.originalCurvePart.pts[0].equals(a.originalCurvePart.pts[0])
			lrShare := lh.originalCurvePart.pts[0].equals(rh.originalCurvePart.pts[0])
			trShare := a.originalCurvePart.pts[0].equals(rh.originalCurvePart.pts[0])
			// if only one pair is the same, the third point touches neither of the pair
			if b2i(ltShare)+b2i(lrShare)+b2i(trShare) == 1 {
				switch {
				case lrShare:
					ltOOrder := lh.linesOnOriginalSide(a)
					rtOOrder := rh.linesOnOriginalSide(a)
					if (rtOOrder ^ ltOOrder) == 1 {
						return ltOOrder != 0
					}
				case trShare:
					tlOOrder := a.linesOnOriginalSide(lh)
					rlOOrder := rh.linesOnOriginalSide(lh)
					if (tlOOrder ^ rlOOrder) == 1 {
						return rlOOrder != 0
					}
				default: // ltShare
					trOOrder := rh.linesOnOriginalSide(a)
					lrOOrder := lh.linesOnOriginalSide(rh)
					// result must be 0 and 1 or 1 and 0 to be valid
					if (lrOOrder ^ trOOrder) == 1 {
						return trOOrder != 0
					}
				}
			}
		}
	}
	if lrOrder < 0 {
		if ltOrder < 0 {
			return trOrder != 0
		}
		return ltOrder != 0
	}
	return lrOrder == 0
}

// previous returns the angle immediately before this one in the sorted counterclockwise loop.
func (a *opAngle) previous() *opAngle {
	last := a.next
	for {
		next := last.next
		if next == a {
			return last
		}
		last = next
	}
}

// loopContains reports whether angle's start segment/t pair appears in this loop.
func (a *opAngle) loopContains(angle *opAngle) bool {
	if a.next == nil {
		return false
	}
	first := a
	loop := a
	tSegment := angle.start.segment
	tStart := angle.start.t()
	tEnd := angle.end.t()
	for {
		lSegment := loop.start.segment
		if lSegment == tSegment {
			lStart := loop.start.t()
			if lStart == tEnd {
				lEnd := loop.end.t()
				if lEnd == tStart {
					return true
				}
			}
		}
		loop = loop.next
		if loop == first {
			break
		}
	}
	return false
}

// loopCount returns the number of angles in this angle's sorted loop.
func (a *opAngle) loopCount() int {
	count := 0
	first := a
	next := a
	for {
		next = next.next
		count++
		if next == nil || next == first {
			break
		}
	}
	return count
}

// insert places angle into this angle's sorted counterclockwise loop.
func (a *opAngle) insert(angle *opAngle) bool {
	if angle.next != nil {
		switch {
		case a.loopCount() >= angle.loopCount():
			if !a.merge(angle) {
				return true
			}
		case a.next != nil:
			if !angle.merge(a) {
				return true
			}
		default:
			angle.insert(a)
		}
		return true
	}
	singleton := a.next == nil
	if singleton {
		a.next = a
	}
	next := a.next
	if next.next == a {
		if singleton || angle.after(a) {
			a.next = angle
			angle.next = next
		} else {
			next.next = angle
			angle.next = a
		}
		return true
	}
	last := a
	flipAmbiguity := false
	for {
		if angle.after(last) != (angle.tangentsAmbiguous && flipAmbiguity) {
			last.next = angle
			angle.next = next
			break
		}
		last = next
		if last == a {
			if flipAmbiguity {
				return false
			}
			// We're in a loop. If a sort was ambiguous, flip it to end the loop.
			flipAmbiguity = true
		}
		next = next.next
	}
	return true
}

// merge folds angle's loop into this loop one element at a time.
func (a *opAngle) merge(angle *opAngle) bool {
	working := angle
	for {
		if a == working {
			return false
		}
		working = working.next
		if working == angle {
			break
		}
	}
	for {
		next := working.next
		working.next = nil
		a.insert(working)
		working = next
		if working == angle {
			break
		}
	}
	return true
}
