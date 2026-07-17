// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The path stroker: convert a path plus stroke geometry (width, cap, join, miter limit) into the path that, when
// filled, is the stroke. Curve offsetting uses a quadratic-approximation approach: each curve segment's parallel is
// approximated by quads found by intersecting end tangents, recursively subdivided until within 1/resScale of the true
// offset.

package stroke

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// Recursive limit indices: cubicStroke indexes by foundTangents (false = tangent, true = cubic).
const (
	tangentRecursiveLimit = iota
	cubicRecursiveLimit
	conicRecursiveLimit
	quadRecursiveLimit
)

// kRecursiveLimits caps the curve-subdivision recursion depth: quads with extreme widths (e.g. (0,1) (1,6) (0,3)
// width=5e7) recurse to point of failure; these limits are 3x those seen in practice, except for cubics, where 24 is
// close to the peak of the recursion-depth histogram.
var kRecursiveLimits = [4]int{5 * 3, 24, 11 * 3, 11 * 3}

// degenerateVector reports whether v is too small to normalize.
func degenerateVector(v geom.Point) bool {
	return !geom.CanNormalize(v.X, v.Y)
}

// setNormalUnitNormal computes, from two points: unitNormal = normalize((after - before) * scale) rotated CCW, normal =
// unitNormal * radius.
func setNormalUnitNormal(before, after geom.Point, scale, radius float32) (normal, unitNormal geom.Point, ok bool) {
	if !unitNormal.SetNormalize((after.X-before.X)*scale, (after.Y-before.Y)*scale) {
		return normal, unitNormal, false
	}
	unitNormal = unitNormal.RotateCCW()
	normal = unitNormal.Scaled(radius)
	return normal, unitNormal, true
}

// setNormalUnitNormalVec is the vector form of setNormalUnitNormal.
func setNormalUnitNormalVec(vec geom.Point, radius float32) (normal, unitNormal geom.Point, ok bool) {
	if !unitNormal.SetNormalize(vec.X, vec.Y) {
		return normal, unitNormal, false
	}
	unitNormal = unitNormal.RotateCCW()
	normal = unitNormal.Scaled(radius)
	return normal, unitNormal, true
}

///////////////////////////////////////////////////////////////////////////////

// quadConstruct is the state of the quad stroke under construction.
type quadConstruct struct {
	quad             [3]geom.Point // the stroked quad parallel to the original curve
	tangentStart     geom.Point    // tangent vector at quad[0]
	tangentEnd       geom.Point    // tangent vector at quad[2]
	startT           float32       // a segment of the original curve
	midT             float32       //              "
	endT             float32       //              "
	startSet         bool          // state to share common points across structs
	endSet           bool          //              "
	oppositeTangents bool          // set if coincident tangents have opposite directions
}

// init returns false if start and end are too close to have a unique middle.
func (q *quadConstruct) init(start, end float32) bool {
	q.startT = start
	q.midT = (start + end) * 0.5
	q.endT = end
	q.startSet = false
	q.endSet = false
	return q.startT < q.midT && q.midT < q.endT
}

func (q *quadConstruct) initWithStart(parent *quadConstruct) bool {
	if !q.init(parent.startT, parent.midT) {
		return false
	}
	q.quad[0] = parent.quad[0]
	q.tangentStart = parent.tangentStart
	q.startSet = true
	return true
}

func (q *quadConstruct) initWithEnd(parent *quadConstruct) bool {
	if !q.init(parent.midT, parent.endT) {
		return false
	}
	q.quad[2] = parent.quad[2]
	q.tangentEnd = parent.tangentEnd
	q.endSet = true
	return true
}

// isZeroLengthSincePoint reports whether all points from startPtIndex on coincide.
func isZeroLengthSincePoint(p *path.Path, startPtIndex int) bool {
	count := p.CountPoints() - startPtIndex
	if count < 2 {
		return true
	}
	first := p.Point(startPtIndex)
	for index := 1; index < count; index++ {
		if first != p.Point(startPtIndex+index) {
			return false
		}
	}
	return true
}

// strokeType selects which side of the stroke is being built: sign-opposite values flip the perpendicular axis.
type strokeType int8

const (
	strokeTypeOuter strokeType = 1
	strokeTypeInner strokeType = -1
)

// resultType reports how a quad-vs-curve comparison resolved.
type resultType uint8

const (
	resultSplit      resultType = iota // the caller should split the quad stroke in two
	resultDegenerate                   // the caller should add a line
	resultQuad                         // the caller should (continue to try to) add a quad stroke
)

// reductionType classifies how degenerate a curve segment is for stroking purposes.
type reductionType uint8

const (
	reductionPoint       reductionType = iota // all curve points are practically identical
	reductionLine                             // the control point is on the line between the ends
	reductionQuad                             // the control point is outside the line between the ends
	reductionDegenerate                       // the control point is on the line but outside the ends
	reductionDegenerate2                      // two control points are on the line but outside ends (cubic)
	reductionDegenerate3                      // three areas of max curvature found (for cubic)
)

// intersectRayType selects what intersectRay computes in addition to the result type.
type intersectRayType uint8

const (
	ctrlPtRayType intersectRayType = iota
	resultTypeRayType
)

// pathStroker holds the working state for stroking one path: accumulated inner/outer contours plus the join/cap procs
// and per-stroke geometry (radius, miter limit, resolution scale).
type pathStroker struct {
	capper                     capProc
	cusper                     *path.Path
	outer                      *path.Path // our working answer
	inner                      *path.Path // temp
	joiner                     joinProc
	firstOuterPtIndexInContour int
	recursionDepth             int // track stack depth to abort if numerics run amok
	segmentCount               int
	firstNormal                geom.Point
	firstUnitNormal            geom.Point
	prevUnitNormal             geom.Point
	firstPt                    geom.Point // on original path
	prevPt                     geom.Point
	firstOuterPt               geom.Point
	prevNormal                 geom.Point
	invMiterLimit              float32
	invResScaleSquared         float32
	invResScale                float32
	resScale                   float32
	radius                     float32
	capType                    Cap
	canIgnoreCenter            bool
	prevIsLine                 bool
	strokeType                 strokeType
	foundTangents              bool // do less work until tangents meet (cubic)
	joinCompleted              bool // previous join was not degenerate
}

// pathStrokerPool retains pathStroker structs — and, crucially, their inner/outer/cusper working paths with their
// accumulated slice capacity — across strokes, so that repeated strokes do not re-allocate the stroke geometry storage.
var pathStrokerPool = sync.Pool{New: func() any {
	return &pathStroker{inner: &path.Path{}, outer: &path.Path{}, cusper: &path.Path{}}
}}

// acquirePathStroker returns a pathStroker from the pool, reinitialized for a new stroke while retaining the pooled
// struct's inner/outer/cusper working paths (rewound, so their storage is reused). Pair every acquire with
// releasePathStroker.
func acquirePathStroker(radius, miterLimit float32, strokeCap Cap, join Join, resScale float32, canIgnoreCenter bool) *pathStroker {
	s := pathStrokerPool.Get().(*pathStroker)
	inner, outer, cusper := s.inner, s.outer, s.cusper
	inner.Rewind()
	outer.Rewind()
	cusper.Rewind()
	// Reset every field (the struct comes back from the pool with stale state), keeping only the retained working-path
	// storage.
	*s = pathStroker{
		radius:                     radius,
		resScale:                   resScale,
		canIgnoreCenter:            canIgnoreCenter,
		inner:                      inner,
		outer:                      outer,
		cusper:                     cusper,
		segmentCount:               -1,
		firstOuterPtIndexInContour: 0,
	}

	// invMiterLimit is only consulted when join is miter_join; it stays at its zero value otherwise.
	if join == JoinMiter {
		if miterLimit <= 1 {
			join = JoinBevel
		} else {
			s.invMiterLimit = geom.ScalarInvert(miterLimit)
		}
	}
	s.capType = strokeCap
	s.capper = capFactory(strokeCap)
	s.joiner = joinFactory(join)

	// The '4' below matches the fill scan converter's error term.
	s.invResScale = geom.ScalarInvert(resScale * 4)
	s.invResScaleSquared = s.invResScale * s.invResScale
	return s
}

// releasePathStroker returns a stroker to the pool, rewinding its working paths (retaining their storage for the next
// acquire) and dropping the cap/join closures so the pooled struct pins nothing.
func releasePathStroker(s *pathStroker) {
	s.inner.Rewind()
	s.outer.Rewind()
	s.cusper.Rewind()
	s.capper = nil
	s.joiner = nil
	pathStrokerPool.Put(s)
}

func (s *pathStroker) hasOnlyMoveTo() bool  { return s.segmentCount == 0 }
func (s *pathStroker) moveToPt() geom.Point { return s.firstPt }

func (s *pathStroker) close(isLine bool) { s.finishContour(true, isLine) }

func (s *pathStroker) done(dst *path.Path, isLine bool) {
	s.finishContour(false, isLine)
	// Copy the accumulated outer contour into dst, reusing dst's storage, rather than moving outer's slices away, so
	// outer's storage stays with the pooled stroker for reuse (see releasePathStroker). Any inverse/fill fixups the
	// caller applies after this append to dst, so only dst — never outer — is handed onward.
	dst.Set(s.outer)
}

func (s *pathStroker) isCurrentContourEmpty() bool {
	return isZeroLengthSincePoint(s.inner, 0) &&
		isZeroLengthSincePoint(s.outer, s.firstOuterPtIndexInContour)
}

///////////////////////////////////////////////////////////////////////////////

// preJoinTo computes the normal at currPt and, if there is a previous segment, appends its join.
func (s *pathStroker) preJoinTo(currPt geom.Point, currIsLine bool) (normal, unitNormal geom.Point, ok bool) {
	normal, unitNormal, ok = setNormalUnitNormal(s.prevPt, currPt, s.resScale, s.radius)
	if !ok {
		if s.capType == CapButt {
			return normal, unitNormal, false
		}
		/* Square caps and round caps draw even if the segment length is zero.
		   Since the zero length segment has no direction, set the orientation
		   to upright as the default orientation */
		normal = geom.Point{X: s.radius, Y: 0}
		unitNormal = geom.Point{X: 1, Y: 0}
	}

	if s.segmentCount == 0 {
		s.firstNormal = normal
		s.firstUnitNormal = unitNormal
		s.firstOuterPt = s.prevPt.Add(normal)

		s.outer.MoveToPt(s.firstOuterPt)
		s.inner.MoveToPt(s.prevPt.Sub(normal))
	} else { // we have a previous segment
		s.joiner(s.outer, s.inner, s.prevUnitNormal, s.prevPt, unitNormal, s.radius,
			s.invMiterLimit, s.prevIsLine, currIsLine)
	}
	s.prevIsLine = currIsLine
	return normal, unitNormal, true
}

// postJoinTo records currPt/normal/unitNormal as the new "previous" state after a segment is stroked.
func (s *pathStroker) postJoinTo(currPt, normal, unitNormal geom.Point) {
	s.joinCompleted = true
	s.prevPt = currPt
	s.prevUnitNormal = unitNormal
	s.prevNormal = normal
	s.segmentCount++
}

// finishContour closes or caps the current contour and appends it to outer.
func (s *pathStroker) finishContour(closeContour, currIsLine bool) {
	if s.segmentCount > 0 {
		if closeContour {
			s.joiner(s.outer, s.inner, s.prevUnitNormal, s.prevPt, s.firstUnitNormal, s.radius,
				s.invMiterLimit, s.prevIsLine, currIsLine)
			s.outer.Close()

			if s.canIgnoreCenter {
				// If we can ignore the center just make sure the larger of the two paths is preserved and don't add the
				// smaller one.
				if s.inner.Bounds().ContainsRect(s.outer.Bounds()) {
					s.outer, s.inner = s.inner, s.outer
				}
			} else {
				// now add fInner as its own contour
				if pt, ok := s.inner.LastPt(); ok {
					s.outer.MoveToPt(pt)
					s.outer.ReversePathTo(s.inner)
					s.outer.Close()
				}
			}
		} else { // add caps to start and end
			// cap the end
			if pt, ok := s.inner.LastPt(); ok {
				s.capper(s.outer, s.prevPt, s.prevNormal, pt, currIsLine)
				s.outer.ReversePathTo(s.inner)
				// cap the start
				s.capper(s.outer, s.firstPt, s.firstNormal.Negated(), s.firstOuterPt, s.prevIsLine)
				s.outer.Close()
			}
		}
		if !s.cusper.IsEmpty() {
			s.outer.AddPath(s.cusper, path.AddPathAppend)
			s.cusper.Rewind() // Rewind (not Reset) to keep the scratch storage for pooled reuse
		}
	}
	s.inner.Rewind() // Rewind (not Reset) to keep the scratch storage for pooled reuse
	s.segmentCount = -1
	s.firstOuterPtIndexInContour = s.outer.CountPoints()
}

///////////////////////////////////////////////////////////////////////////////

// moveTo starts a new contour at pt, finishing the previous one first if needed.
func (s *pathStroker) moveTo(pt geom.Point) {
	if s.segmentCount > 0 {
		s.finishContour(false, false)
	}
	s.segmentCount = 0
	s.firstPt = pt
	s.prevPt = pt
	s.joinCompleted = false
}

// strokeLineTo appends the outer/inner line segments offset by normal.
func (s *pathStroker) strokeLineTo(currPt, normal geom.Point) {
	s.outer.LineToPt(currPt.Add(normal))
	s.inner.LineToPt(currPt.Sub(normal))
}

// hasValidTangent reports whether the rest of the contour (from a copy of the iterator) contains a segment with a
// direction.
func hasValidTangent(iter *path.Iter) bool {
	cp := *iter
	for {
		verb, pts, ok := cp.Next()
		if !ok {
			return false
		}
		switch verb {
		case path.VerbMove:
			return false
		case path.VerbLine:
			if pts[0] == pts[1] {
				continue
			}
			return true
		case path.VerbQuad, path.VerbConic:
			if pts[0] == pts[1] && pts[1] == pts[2] {
				continue
			}
			return true
		case path.VerbCubic:
			if pts[0] == pts[1] && pts[1] == pts[2] && pts[2] == pts[3] {
				continue
			}
			return true
		case path.VerbClose:
			return false
		}
	}
}

// lineTo strokes a line segment to currPt. iter is non-nil only when called from the path walker.
func (s *pathStroker) lineTo(currPt geom.Point, iter *path.Iter) {
	teenyLine := s.prevPt.EqualsWithinToleranceTol(currPt, geom.ScalarNearlyZeroTol*s.invResScale)
	if s.capType == CapButt && teenyLine {
		return
	}
	if teenyLine && (s.joinCompleted || (iter != nil && hasValidTangent(iter))) {
		return
	}

	normal, unitNormal, ok := s.preJoinTo(currPt, true)
	if !ok {
		return
	}
	s.strokeLineTo(currPt, normal)
	s.postJoinTo(currPt, normal, unitNormal)
}

// setQuadEndNormal computes the normal at the quad's end point, falling back to normalAB/unitNormalAB if it
// degenerates.
func (s *pathStroker) setQuadEndNormal(quad []geom.Point, normalAB, unitNormalAB geom.Point) (normalBC, unitNormalBC geom.Point) {
	var ok bool
	normalBC, unitNormalBC, ok = setNormalUnitNormal(quad[1], quad[2], s.resScale, s.radius)
	if !ok {
		return normalAB, unitNormalAB
	}
	return normalBC, unitNormalBC
}

// setConicEndNormal computes the normal at the conic's end point.
func (s *pathStroker) setConicEndNormal(conic *geom.Conic, normalAB, unitNormalAB geom.Point) (normalBC, unitNormalBC geom.Point) {
	return s.setQuadEndNormal(conic.Pts[:], normalAB, unitNormalAB)
}

// setCubicEndNormal computes the normal at the cubic's end point, falling back through progressively earlier control
// points when the tangent degenerates.
func (s *pathStroker) setCubicEndNormal(cubic []geom.Point, normalAB, unitNormalAB geom.Point) (normalCD, unitNormalCD geom.Point) {
	ab := cubic[1].Sub(cubic[0])
	cd := cubic[3].Sub(cubic[2])

	degenerateAB := degenerateVector(ab)
	degenerateCD := degenerateVector(cd)

	if !degenerateAB || !degenerateCD {
		if degenerateAB {
			ab = cubic[2].Sub(cubic[0])
			degenerateAB = degenerateVector(ab)
		}
		if degenerateCD {
			cd = cubic[3].Sub(cubic[1])
			degenerateCD = degenerateVector(cd)
		}
		if !degenerateAB && !degenerateCD {
			normalCD, unitNormalCD, _ = setNormalUnitNormalVec(cd, s.radius)
			return normalCD, unitNormalCD
		}
	}
	return normalAB, unitNormalAB
}

// initQuad resets stroke-side state and the quad-construction bounds for a new curve segment.
func (s *pathStroker) initQuad(st strokeType, quadPts *quadConstruct, tStart, tEnd float32) {
	s.strokeType = st
	s.foundTangents = false
	quadPts.init(tStart, tEnd)
}

///////////////////////////////////////////////////////////////////////////////

// ptToLine returns the distance squared from the point to the segment.
func ptToLine(pt, lineStart, lineEnd geom.Point) float32 {
	dxy := lineEnd.Sub(lineStart)
	ab0 := pt.Sub(lineStart)
	numer := dxy.Dot(ab0)
	denom := dxy.Dot(dxy)
	t := numer / denom
	if t >= 0 && t <= 1 {
		hit := geom.Point{
			X: float32(lineStart.X*(1-t)) + float32(lineEnd.X*t),
			Y: float32(lineStart.Y*(1-t)) + float32(lineEnd.Y*t),
		}
		return hit.DistanceToSqd(pt)
	}
	return pt.DistanceToSqd(lineStart)
}

// ptToTangentLine returns the distance squared from the point to the tangent ray from lineStart.
func ptToTangentLine(pt, lineStart, tangent geom.Point) float32 {
	dxy := tangent
	ab0 := pt.Sub(lineStart)
	numer := dxy.Dot(ab0)
	denom := dxy.Dot(dxy)
	t := numer / denom
	if t >= 0 && t <= 1 {
		hit := lineStart.Add(tangent.Scaled(t))
		return hit.DistanceToSqd(pt)
	}
	return pt.DistanceToSqd(lineStart)
}

// cubicInLine reports whether all four points are in a line.
func cubicInLine(cubic []geom.Point) bool {
	ptMax := float32(-1)
	outer1 := 0
	outer2 := 0
	for index := 0; index < 3; index++ {
		for inner := index + 1; inner < 4; inner++ {
			testDiff := cubic[inner].Sub(cubic[index])
			testMax := max(geom.ScalarAbs(testDiff.X), geom.ScalarAbs(testDiff.Y))
			if ptMax < testMax {
				outer1 = index
				outer2 = inner
				ptMax = testMax
			}
		}
	}
	mid1 := (1 + (2 >> outer2)) >> outer1
	mid2 := outer1 ^ outer2 ^ mid1
	lineSlop := ptMax * ptMax * 0.00001 // this multiplier is pulled out of the air
	return ptToLine(cubic[mid1], cubic[outer1], cubic[outer2]) <= lineSlop &&
		ptToLine(cubic[mid2], cubic[outer1], cubic[outer2]) <= lineSlop
}

// quadInLine reports whether all three points are in a line.
func quadInLine(quad []geom.Point) bool {
	ptMax := float32(-1)
	outer1 := 0
	outer2 := 0
	for index := 0; index < 2; index++ {
		for inner := index + 1; inner < 3; inner++ {
			testDiff := quad[inner].Sub(quad[index])
			testMax := max(geom.ScalarAbs(testDiff.X), geom.ScalarAbs(testDiff.Y))
			if ptMax < testMax {
				outer1 = index
				outer2 = inner
				ptMax = testMax
			}
		}
	}
	mid := outer1 ^ outer2 ^ 3
	const kCurvatureSlop = float32(0.000005) // this multiplier is pulled out of the air
	lineSlop := ptMax * ptMax * kCurvatureSlop
	return ptToLine(quad[mid], quad[outer1], quad[outer2]) <= lineSlop
}

func conicInLine(conic *geom.Conic) bool {
	return quadInLine(conic.Pts[:])
}

// checkCubicLinear classifies how degenerate the cubic is for stroking, filling reduction with any points of maximum
// curvature found along the way.
func checkCubicLinear(cubic, reduction []geom.Point) (reductionType, *geom.Point) {
	degenerateAB := degenerateVector(cubic[1].Sub(cubic[0]))
	degenerateBC := degenerateVector(cubic[2].Sub(cubic[1]))
	degenerateCD := degenerateVector(cubic[3].Sub(cubic[2]))
	if degenerateAB && degenerateBC && degenerateCD {
		return reductionPoint, nil
	}
	degenerateCount := 0
	for _, d := range []bool{degenerateAB, degenerateBC, degenerateCD} {
		if d {
			degenerateCount++
		}
	}
	if degenerateCount == 2 {
		return reductionLine, nil
	}
	if !cubicInLine(cubic) {
		tanPt := &cubic[1]
		if degenerateAB {
			tanPt = &cubic[2]
		}
		return reductionQuad, tanPt
	}
	var tValues [3]float32
	count := geom.FindCubicMaxCurvature(cubic, &tValues)
	rCount := 0
	// Now loop over the t-values, and reject any that evaluate to either end-point
	for index := 0; index < count; index++ {
		t := tValues[index]
		if t <= 0 || t >= 1 {
			continue
		}
		reduction[rCount] = geom.EvalCubicPosAt(cubic, t)
		if reduction[rCount] != cubic[0] && reduction[rCount] != cubic[3] {
			rCount++
		}
	}
	if rCount == 0 {
		return reductionLine, nil
	}
	return reductionQuad + reductionType(rCount), nil
}

// checkConicLinear classifies how degenerate the conic is for stroking, filling reduction with the point of maximum
// curvature if found.
func checkConicLinear(conic *geom.Conic, reduction *geom.Point) reductionType {
	degenerateAB := degenerateVector(conic.Pts[1].Sub(conic.Pts[0]))
	degenerateBC := degenerateVector(conic.Pts[2].Sub(conic.Pts[1]))
	if degenerateAB && degenerateBC {
		return reductionPoint
	}
	if degenerateAB || degenerateBC {
		return reductionLine
	}
	if !conicInLine(conic) {
		return reductionQuad
	}
	// An exact conic-curvature solver would be a better fit here, once one is available. Quad curvature is a reasonable
	// substitute.
	t := geom.FindQuadMaxCurvature(conic.Pts[:])
	if t == 0 || geom.ScalarIsNaN(t) {
		return reductionLine
	}
	*reduction = conic.EvalAt(t)
	return reductionDegenerate
}

// checkQuadLinear classifies how degenerate the quad is for stroking, filling reduction with the point of maximum
// curvature if found.
func checkQuadLinear(quad []geom.Point, reduction *geom.Point) reductionType {
	degenerateAB := degenerateVector(quad[1].Sub(quad[0]))
	degenerateBC := degenerateVector(quad[2].Sub(quad[1]))
	if degenerateAB && degenerateBC {
		return reductionPoint
	}
	if degenerateAB || degenerateBC {
		return reductionLine
	}
	if !quadInLine(quad) {
		return reductionQuad
	}
	t := geom.FindQuadMaxCurvature(quad)
	if t == 0 || t == 1 {
		return reductionLine
	}
	*reduction = geom.EvalQuadAt(quad, t)
	return reductionDegenerate
}

///////////////////////////////////////////////////////////////////////////////

// conicTo strokes a conic segment to pt2, handling degenerate reductions before falling back to the quad-approximation
// stroke.
func (s *pathStroker) conicTo(pt1, pt2 geom.Point, weight float32) {
	conic := geom.MakeConic(s.prevPt, pt1, pt2, weight)
	var reduction geom.Point
	rtype := checkConicLinear(&conic, &reduction)
	if rtype == reductionPoint {
		/* If the stroke consists of a moveTo followed by a degenerate curve, treat it
		   as if it were followed by a zero-length line. Lines without length
		   can have square and round end caps. */
		s.lineTo(pt2, nil)
		return
	}
	if rtype == reductionLine {
		s.lineTo(pt2, nil)
		return
	}
	if rtype == reductionDegenerate {
		s.lineTo(reduction, nil)
		saveJoiner := s.joiner
		s.joiner = joinFactory(JoinRound)
		s.lineTo(pt2, nil)
		s.joiner = saveJoiner
		return
	}
	normalAB, unitAB, ok := s.preJoinTo(pt1, false)
	if !ok {
		s.lineTo(pt2, nil)
		return
	}
	var quadPts quadConstruct
	s.initQuad(strokeTypeOuter, &quadPts, 0, 1)
	s.conicStroke(&conic, &quadPts)
	s.initQuad(strokeTypeInner, &quadPts, 0, 1)
	s.conicStroke(&conic, &quadPts)
	normalBC, unitBC := s.setConicEndNormal(&conic, normalAB, unitAB)
	s.postJoinTo(pt2, normalBC, unitBC)
}

// quadTo strokes a quad segment to pt2, handling degenerate reductions before falling back to the quad-approximation
// stroke.
func (s *pathStroker) quadTo(pt1, pt2 geom.Point) {
	quad := [3]geom.Point{s.prevPt, pt1, pt2}
	var reduction geom.Point
	rtype := checkQuadLinear(quad[:], &reduction)
	if rtype == reductionPoint {
		/* If the stroke consists of a moveTo followed by a degenerate curve, treat it
		   as if it were followed by a zero-length line. Lines without length
		   can have square and round end caps. */
		s.lineTo(pt2, nil)
		return
	}
	if rtype == reductionLine {
		s.lineTo(pt2, nil)
		return
	}
	if rtype == reductionDegenerate {
		s.lineTo(reduction, nil)
		saveJoiner := s.joiner
		s.joiner = joinFactory(JoinRound)
		s.lineTo(pt2, nil)
		s.joiner = saveJoiner
		return
	}
	normalAB, unitAB, ok := s.preJoinTo(pt1, false)
	if !ok {
		s.lineTo(pt2, nil)
		return
	}
	var quadPts quadConstruct
	s.initQuad(strokeTypeOuter, &quadPts, 0, 1)
	s.quadStroke(quad[:], &quadPts)
	s.initQuad(strokeTypeInner, &quadPts, 0, 1)
	s.quadStroke(quad[:], &quadPts)
	normalBC, unitBC := s.setQuadEndNormal(quad[:], normalAB, unitAB)
	s.postJoinTo(pt2, normalBC, unitBC)
}

///////////////////////////////////////////////////////////////////////////////

// setRayPts, given a point on the curve and its derivative, scales the derivative by the radius, and computes the
// perpendicular point and its tangent.
func (s *pathStroker) setRayPts(tPt geom.Point, dxy, onPt, tangent *geom.Point) {
	if !dxy.SetLength(s.radius) {
		*dxy = geom.Point{X: s.radius, Y: 0}
	}
	axisFlip := float32(s.strokeType) // go opposite ways for outer, inner
	onPt.X = tPt.X + axisFlip*dxy.Y
	onPt.Y = tPt.Y - axisFlip*dxy.X
	if tangent != nil {
		*tangent = *dxy
	}
}

// conicPerpRay, given a conic and t, returns the point on the curve, its perpendicular, and the perpendicular tangent.
func (s *pathStroker) conicPerpRay(conic *geom.Conic, t float32, tPt, onPt, tangent *geom.Point) {
	*tPt = conic.EvalAt(t)
	dxy := conic.EvalTangentAt(t)
	if dxy.IsZero() {
		dxy = conic.Pts[2].Sub(conic.Pts[0])
	}
	s.setRayPts(*tPt, &dxy, onPt, tangent)
}

// conicQuadEnds fills in quadPts' start/end points and tangents from the conic, if not already set.
func (s *pathStroker) conicQuadEnds(conic *geom.Conic, quadPts *quadConstruct) {
	if !quadPts.startSet {
		var conicStartPt geom.Point
		s.conicPerpRay(conic, quadPts.startT, &conicStartPt, &quadPts.quad[0], &quadPts.tangentStart)
		quadPts.startSet = true
	}
	if !quadPts.endSet {
		var conicEndPt geom.Point
		s.conicPerpRay(conic, quadPts.endT, &conicEndPt, &quadPts.quad[2], &quadPts.tangentEnd)
		quadPts.endSet = true
	}
}

// cubicPerpRay, given a cubic and t, returns the point on the curve, its perpendicular, and the perpendicular tangent,
// subdividing to resolve the tangent at a cusp if needed.
func (s *pathStroker) cubicPerpRay(cubic []geom.Point, t float32, tPt, onPt, tangent *geom.Point) {
	*tPt = geom.EvalCubicPosAt(cubic, t)
	dxy := geom.EvalCubicTangentAt(cubic, t)
	var chopped [7]geom.Point
	if dxy.IsZero() {
		cPts := cubic
		switch {
		case geom.ScalarNearlyZero(t):
			dxy = cubic[2].Sub(cubic[0])
		case geom.ScalarNearlyZero(1 - t):
			dxy = cubic[3].Sub(cubic[1])
		default:
			// If the cubic inflection falls on the cusp, subdivide the cubic to find the tangent at that point.
			geom.ChopCubicAt(cubic, chopped[:], t)
			dxy = chopped[3].Sub(chopped[2])
			if dxy.IsZero() {
				dxy = chopped[3].Sub(chopped[1])
				cPts = chopped[:]
			}
		}
		if dxy.IsZero() {
			dxy = cPts[3].Sub(cPts[0])
		}
	}
	s.setRayPts(*tPt, &dxy, onPt, tangent)
}

// cubicQuadEnds fills in quadPts' start/end points and tangents from the cubic, if not already set.
func (s *pathStroker) cubicQuadEnds(cubic []geom.Point, quadPts *quadConstruct) {
	if !quadPts.startSet {
		var cubicStartPt geom.Point
		s.cubicPerpRay(cubic, quadPts.startT, &cubicStartPt, &quadPts.quad[0], &quadPts.tangentStart)
		quadPts.startSet = true
	}
	if !quadPts.endSet {
		var cubicEndPt geom.Point
		s.cubicPerpRay(cubic, quadPts.endT, &cubicEndPt, &quadPts.quad[2], &quadPts.tangentEnd)
		quadPts.endSet = true
	}
}

// cubicQuadMid returns the perpendicular point on the cubic at quadPts' midpoint t.
func (s *pathStroker) cubicQuadMid(cubic []geom.Point, quadPts *quadConstruct, mid *geom.Point) {
	var cubicMidPt geom.Point
	s.cubicPerpRay(cubic, quadPts.midT, &cubicMidPt, mid, nil)
}

// quadPerpRay, given a quad and t, returns the point on the curve, its perpendicular, and the perpendicular tangent.
func (s *pathStroker) quadPerpRay(quad []geom.Point, t float32, tPt, onPt, tangent *geom.Point) {
	*tPt = geom.EvalQuadAt(quad, t)
	dxy := geom.EvalQuadTangentAt(quad, t)
	if dxy.IsZero() {
		dxy = quad[2].Sub(quad[0])
	}
	s.setRayPts(*tPt, &dxy, onPt, tangent)
}

// intersectRay finds the intersection of the stroke tangents to construct a stroke quad, returning whether the stroke
// is a degenerate (a line), a quad, or must be split. Optionally computes the quad's control point.
func (s *pathStroker) intersectRay(quadPts *quadConstruct, rayType intersectRayType) resultType {
	start := quadPts.quad[0]
	end := quadPts.quad[2]
	aLen := quadPts.tangentStart
	bLen := quadPts.tangentEnd
	/* Slopes match when denom goes to zero:
	                  axLen / ayLen ==                   bxLen / byLen
	(ayLen * byLen) * axLen / ayLen == (ayLen * byLen) * bxLen / byLen
	         byLen  * axLen         ==  ayLen          * bxLen
	         byLen  * axLen         -   ayLen          * bxLen         ( == denom )
	*/
	denom := aLen.Cross(bLen)
	if denom == 0 || !geom.IsFinite(denom) {
		quadPts.oppositeTangents = aLen.Dot(bLen) < 0
		return resultDegenerate
	}
	quadPts.oppositeTangents = false
	ab0 := start.Sub(end)
	numerA := bLen.Cross(ab0)
	numerB := aLen.Cross(ab0)
	if (numerA >= 0) == (numerB >= 0) { // if the control point is outside the quad ends
		// if the perpendicular distances from the quad points to the opposite tangent line are small, a straight line
		// is good enough
		dist1 := ptToTangentLine(start, end, quadPts.tangentEnd)
		dist2 := ptToTangentLine(end, start, quadPts.tangentStart)
		if max(dist1, dist2) <= s.invResScaleSquared {
			return resultDegenerate
		}
		return resultSplit
	}
	// check to see if the denominator is teeny relative to the numerator; if the offset by one will be lost, the ratio
	// is too large
	numerA /= denom
	// A deliberate floating-point precision probe, not a tautology: when numerA is so large that subtracting one is
	// lost to rounding (or numerA is NaN), numerA > numerA-1 is false and the divide is rejected.
	validDivide := numerA > numerA-1 //nolint:gocritic // see above
	if validDivide {
		if rayType == ctrlPtRayType {
			// the intersection of the tangents need not be on the tangent segment, so 0 <= numerA <= 1 is not
			// necessarily true
			quadPts.quad[1] = start.Add(quadPts.tangentStart.Scaled(numerA))
		}
		return resultQuad
	}
	quadPts.oppositeTangents = aLen.Dot(bLen) < 0
	// if the lines are parallel, straight line is good enough
	return resultDegenerate
}

// tangentsMeet computes the cubic's quad-approximation endpoints and checks whether their tangents intersect in a
// usable quad.
func (s *pathStroker) tangentsMeet(cubic []geom.Point, quadPts *quadConstruct) resultType {
	s.cubicQuadEnds(cubic, quadPts)
	return s.intersectRay(quadPts, resultTypeRayType)
}

///////////////////////////////////////////////////////////////////////////////

// intersectQuadRay intersects the line with the quad and returns the t values on the quad where the line crosses.
func intersectQuadRay(line, quad []geom.Point, roots *[2]float32) int {
	vec := line[1].Sub(line[0])
	var r [3]float32
	for n := 0; n < 3; n++ {
		r[n] = vec.Cross(quad[n].Sub(line[0]))
	}
	a := r[2]
	b := r[1]
	c := r[0]
	a += c - 2*b // A = a - 2*b + c
	b -= c       // B = -(b - c)
	return geom.FindUnitQuadRoots(a, 2*b, c, roots)
}

// ptInQuadBounds returns true if the point is close to the bounds of the quad. This is used as a quick reject.
func (s *pathStroker) ptInQuadBounds(quad []geom.Point, pt geom.Point) bool {
	xMin := min(quad[0].X, quad[1].X, quad[2].X)
	if pt.X+s.invResScale < xMin {
		return false
	}
	xMax := max(quad[0].X, quad[1].X, quad[2].X)
	if pt.X-s.invResScale > xMax {
		return false
	}
	yMin := min(quad[0].Y, quad[1].Y, quad[2].Y)
	if pt.Y+s.invResScale < yMin {
		return false
	}
	yMax := max(quad[0].Y, quad[1].Y, quad[2].Y)
	return pt.Y-s.invResScale <= yMax
}

// pointsWithinDist reports whether nearPt and farPt are within limit of each other.
func pointsWithinDist(nearPt, farPt geom.Point, limit float32) bool {
	return nearPt.DistanceToSqd(farPt) <= limit*limit
}

// sharpAngle reports whether the quad's control point forms a sharp angle relative to its endpoints.
func sharpAngle(quad []geom.Point) bool {
	smaller := quad[1].Sub(quad[0])
	larger := quad[1].Sub(quad[2])
	smallerLen := smaller.LengthSqd()
	largerLen := larger.LengthSqd()
	if smallerLen > largerLen {
		smaller, larger = larger, smaller
		largerLen = smallerLen
	}
	if !smaller.SetLength(largerLen) {
		return false
	}
	dot := smaller.Dot(larger)
	return dot > 0
}

// strokeCloseEnough reports whether the quad-approximation stroke is an acceptable approximation of the true curve
// offset, or whether it must be split or accepted as-is.
func (s *pathStroker) strokeCloseEnough(stroke, ray []geom.Point, quadPts *quadConstruct) resultType {
	strokeMid := geom.EvalQuadAt(stroke, 0.5)
	// measure the distance from the curve to the quad-stroke midpoint, compare to radius
	if pointsWithinDist(ray[0], strokeMid, s.invResScale) { // if the difference is small
		if sharpAngle(quadPts.quad[:]) {
			return resultSplit
		}
		return resultQuad
	}
	// measure the distance to quad's bounds (quick reject); an alternative: look for point in triangle
	if !s.ptInQuadBounds(stroke, ray[0]) { // if far, subdivide
		return resultSplit
	}
	// measure the curve ray distance to the quad-stroke
	var roots [2]float32
	rootCount := intersectQuadRay(ray, stroke, &roots)
	if rootCount != 1 {
		return resultSplit
	}
	quadPt := geom.EvalQuadAt(stroke, roots[0])
	errorTol := s.invResScale * (1 - geom.ScalarAbs(roots[0]-0.5)*2)
	if pointsWithinDist(ray[0], quadPt, errorTol) { // if the difference is small, we're done
		if sharpAngle(quadPts.quad[:]) {
			return resultSplit
		}
		return resultQuad
	}
	// otherwise, subdivide
	return resultSplit
}

// compareQuadCubic checks whether the quad approximation of cubic's offset is close enough to accept.
func (s *pathStroker) compareQuadCubic(cubic []geom.Point, quadPts *quadConstruct) resultType {
	// get the quadratic approximation of the stroke
	s.cubicQuadEnds(cubic, quadPts)
	rtype := s.intersectRay(quadPts, ctrlPtRayType)
	if rtype != resultQuad {
		return rtype
	}
	// project a ray from the curve to the stroke
	var ray [2]geom.Point // points near midpoint on quad, midpoint on cubic
	s.cubicPerpRay(cubic, quadPts.midT, &ray[1], &ray[0], nil)
	return s.strokeCloseEnough(quadPts.quad[:], ray[:], quadPts)
}

// compareQuadConic checks whether the quad approximation of conic's offset is close enough to accept.
func (s *pathStroker) compareQuadConic(conic *geom.Conic, quadPts *quadConstruct) resultType {
	// get the quadratic approximation of the stroke
	s.conicQuadEnds(conic, quadPts)
	rtype := s.intersectRay(quadPts, ctrlPtRayType)
	if rtype != resultQuad {
		return rtype
	}
	// project a ray from the curve to the stroke
	var ray [2]geom.Point // points near midpoint on quad, midpoint on conic
	s.conicPerpRay(conic, quadPts.midT, &ray[1], &ray[0], nil)
	return s.strokeCloseEnough(quadPts.quad[:], ray[:], quadPts)
}

// compareQuadQuad checks whether the quad approximation of quad's offset is close enough to accept.
func (s *pathStroker) compareQuadQuad(quad []geom.Point, quadPts *quadConstruct) resultType {
	// get the quadratic approximation of the stroke
	if !quadPts.startSet {
		var quadStartPt geom.Point
		s.quadPerpRay(quad, quadPts.startT, &quadStartPt, &quadPts.quad[0], &quadPts.tangentStart)
		quadPts.startSet = true
	}
	if !quadPts.endSet {
		var quadEndPt geom.Point
		s.quadPerpRay(quad, quadPts.endT, &quadEndPt, &quadPts.quad[2], &quadPts.tangentEnd)
		quadPts.endSet = true
	}
	rtype := s.intersectRay(quadPts, ctrlPtRayType)
	if rtype != resultQuad {
		return rtype
	}
	// project a ray from the curve to the stroke
	var ray [2]geom.Point
	s.quadPerpRay(quad, quadPts.midT, &ray[1], &ray[0], nil)
	return s.strokeCloseEnough(quadPts.quad[:], ray[:], quadPts)
}

// addDegenerateLine appends a straight line to the current stroke side's sink.
func (s *pathStroker) addDegenerateLine(quadPts *quadConstruct) {
	sink := s.inner
	if s.strokeType == strokeTypeOuter {
		sink = s.outer
	}
	sink.LineToPt(quadPts.quad[2])
}

// cubicMidOnLine reports whether the cubic's midpoint ray lands on the quad's chord.
func (s *pathStroker) cubicMidOnLine(cubic []geom.Point, quadPts *quadConstruct) bool {
	var strokeMid geom.Point
	s.cubicQuadMid(cubic, quadPts, &strokeMid)
	dist := ptToLine(strokeMid, quadPts.quad[0], quadPts.quad[2])
	return dist < s.invResScaleSquared
}

// cubicStroke recursively approximates one side of a cubic's offset with quad segments, subdividing until the
// approximation is close enough or the recursion limit is hit.
func (s *pathStroker) cubicStroke(cubic []geom.Point, quadPts *quadConstruct) bool {
	if !s.foundTangents {
		rtype := s.tangentsMeet(cubic, quadPts)
		if rtype != resultQuad {
			if (rtype == resultDegenerate ||
				pointsWithinDist(quadPts.quad[0], quadPts.quad[2], s.invResScale)) &&
				s.cubicMidOnLine(cubic, quadPts) {
				s.addDegenerateLine(quadPts)
				return true
			}
		} else {
			s.foundTangents = true
		}
	}
	if s.foundTangents {
		rtype := s.compareQuadCubic(cubic, quadPts)
		if rtype == resultQuad {
			sink := s.inner
			if s.strokeType == strokeTypeOuter {
				sink = s.outer
			}
			sink.QuadToPt(quadPts.quad[1], quadPts.quad[2])
			return true
		}
		if rtype == resultDegenerate {
			if !quadPts.oppositeTangents {
				s.addDegenerateLine(quadPts)
				return true
			}
		}
	}
	if !quadPts.quad[2].IsFinite() {
		return false // just abort if projected quad isn't representable
	}
	limit := kRecursiveLimits[tangentRecursiveLimit]
	if s.foundTangents {
		limit = kRecursiveLimits[cubicRecursiveLimit]
	}
	s.recursionDepth++
	if s.recursionDepth > limit {
		// If we stop making progress, just emit a line and move on
		s.addDegenerateLine(quadPts)
		return true
	}
	var half quadConstruct
	if !half.initWithStart(quadPts) {
		s.addDegenerateLine(quadPts)
		s.recursionDepth--
		return true
	}
	if !s.cubicStroke(cubic, &half) {
		return false
	}
	if !half.initWithEnd(quadPts) {
		s.addDegenerateLine(quadPts)
		s.recursionDepth--
		return true
	}
	if !s.cubicStroke(cubic, &half) {
		return false
	}
	s.recursionDepth--
	return true
}

// conicStroke recursively approximates one side of a conic's offset with quad segments, subdividing until the
// approximation is close enough or the recursion limit is hit.
func (s *pathStroker) conicStroke(conic *geom.Conic, quadPts *quadConstruct) bool {
	rtype := s.compareQuadConic(conic, quadPts)
	if rtype == resultQuad {
		sink := s.inner
		if s.strokeType == strokeTypeOuter {
			sink = s.outer
		}
		sink.QuadToPt(quadPts.quad[1], quadPts.quad[2])
		return true
	}
	if rtype == resultDegenerate {
		s.addDegenerateLine(quadPts)
		return true
	}
	s.recursionDepth++
	if s.recursionDepth > kRecursiveLimits[conicRecursiveLimit] {
		// If we stop making progress, just emit a line and move on
		s.addDegenerateLine(quadPts)
		return true
	}
	var half quadConstruct
	half.initWithStart(quadPts)
	if !s.conicStroke(conic, &half) {
		return false
	}
	half.initWithEnd(quadPts)
	if !s.conicStroke(conic, &half) {
		return false
	}
	s.recursionDepth--
	return true
}

// quadStroke recursively approximates one side of a quad's offset with quad segments, subdividing until the
// approximation is close enough or the recursion limit is hit.
func (s *pathStroker) quadStroke(quad []geom.Point, quadPts *quadConstruct) bool {
	rtype := s.compareQuadQuad(quad, quadPts)
	if rtype == resultQuad {
		sink := s.inner
		if s.strokeType == strokeTypeOuter {
			sink = s.outer
		}
		sink.QuadToPt(quadPts.quad[1], quadPts.quad[2])
		return true
	}
	if rtype == resultDegenerate {
		s.addDegenerateLine(quadPts)
		return true
	}
	s.recursionDepth++
	if s.recursionDepth > kRecursiveLimits[quadRecursiveLimit] {
		// If we stop making progress, just emit a line and move on
		s.addDegenerateLine(quadPts)
		return true
	}
	var half quadConstruct
	half.initWithStart(quadPts)
	if !s.quadStroke(quad, &half) {
		return false
	}
	half.initWithEnd(quadPts)
	if !s.quadStroke(quad, &half) {
		return false
	}
	s.recursionDepth--
	return true
}

// cubicTo strokes a cubic segment to pt3, splitting at inflection points and handling degenerate reductions and cusps
// before falling back to the quad-approximation stroke.
func (s *pathStroker) cubicTo(pt1, pt2, pt3 geom.Point) {
	cubic := [4]geom.Point{s.prevPt, pt1, pt2, pt3}
	var reduction [3]geom.Point
	rtype, tangentPt := checkCubicLinear(cubic[:], reduction[:])
	if rtype == reductionPoint {
		/* If the stroke consists of a moveTo followed by a degenerate curve, treat it
		   as if it were followed by a zero-length line. Lines without length
		   can have square and round end caps. */
		s.lineTo(pt3, nil)
		return
	}
	if rtype == reductionLine {
		s.lineTo(pt3, nil)
		return
	}
	if reductionDegenerate <= rtype && rtype <= reductionDegenerate3 {
		s.lineTo(reduction[0], nil)
		saveJoiner := s.joiner
		s.joiner = joinFactory(JoinRound)
		if reductionDegenerate2 <= rtype {
			s.lineTo(reduction[1], nil)
		}
		if reductionDegenerate3 == rtype {
			s.lineTo(reduction[2], nil)
		}
		s.lineTo(pt3, nil)
		s.joiner = saveJoiner
		return
	}
	normalAB, unitAB, ok := s.preJoinTo(*tangentPt, false)
	if !ok {
		s.lineTo(pt3, nil)
		return
	}
	var tValues [2]float32
	count := geom.FindCubicInflections(cubic[:], &tValues)
	lastT := float32(0)
	for index := 0; index <= count; index++ {
		nextT := float32(1)
		if index < count {
			nextT = tValues[index]
		}
		var quadPts quadConstruct
		s.initQuad(strokeTypeOuter, &quadPts, lastT, nextT)
		s.cubicStroke(cubic[:], &quadPts)
		s.initQuad(strokeTypeInner, &quadPts, lastT, nextT)
		s.cubicStroke(cubic[:], &quadPts)
		lastT = nextT
	}
	cusp := geom.FindCubicCusp(cubic[:])
	if cusp > 0 {
		cuspLoc := geom.EvalCubicPosAt(cubic[:], cusp)
		s.cusper.AddCircle(cuspLoc.X, cuspLoc.Y, s.radius, geom.DirectionCW)
	}
	// emit the join even if one stroke succeeded but the last one failed; this avoids reversing an inner stroke with a
	// partial path followed by another moveto
	normalCD, unitCD := s.setCubicEndNormal(cubic[:], normalAB, unitAB)
	s.postJoinTo(pt3, normalCD, unitCD)
}

///////////////////////////////////////////////////////////////////////////////

// Stroke is the public stroking object: it holds the paint's stroke geometry (width, cap, join, miter limit) and
// converts a path into its stroke outline.
type Stroke struct {
	width      float32
	miterLimit float32
	resScale   float32
	cap        Cap
	join       Join
	doFill     bool
}

// NewStroke returns a Stroke with the paint defaults: width 1, miter limit 4, butt cap, miter join, resScale 1, no
// fill.
func NewStroke() *Stroke {
	return &Stroke{width: 1, miterLimit: 4, resScale: 1}
}

// SetWidth sets the stroke width.
func (st *Stroke) SetWidth(width float32) { st.width = width }

// SetMiterLimit sets the miter limit.
func (st *Stroke) SetMiterLimit(miterLimit float32) { st.miterLimit = miterLimit }

// SetCap sets the cap style.
func (st *Stroke) SetCap(c Cap) { st.cap = c }

// SetJoin sets the join style.
func (st *Stroke) SetJoin(j Join) { st.join = j }

// SetResScale sets the resolution scale used to tessellate curves.
func (st *Stroke) SetResScale(rs float32) { st.resScale = rs }

// SetDoFill sets whether the stroke outline also includes the fill of the original path.
func (st *Stroke) SetDoFill(doFill bool) { st.doFill = doFill }

// StrokePath strokes src into dst. dst's previous contents are replaced; src and dst may be the same path.
func (st *Stroke) StrokePath(src, dst *path.Path) {
	if src == dst {
		tmp := path.Borrow()
		st.StrokePath(src, tmp)
		dst.Set(tmp)
		path.Recycle(tmp)
		return
	}

	radius := st.width / 2
	if radius <= 0 {
		return
	}

	// If src is really a rect, call our specialty strokeRect() method
	if rect, isClosed, dir, isRect := src.IsRectWithDirection(); isRect && isClosed {
		st.StrokeRect(rect, dst, dir)
		// our answer should preserve the inverseness of the src
		if src.IsInverseFillType() {
			dst.ToggleInverseFillType()
		}
		return
	}

	// We can always ignore centers for stroke and fill convex line-only paths TODO: remove the line-only restriction
	ignoreCenter := st.doFill && src.SegmentMasks() == path.SegmentLine &&
		src.IsLastContourClosed() && src.IsConvex()

	stroker := acquirePathStroker(radius, st.miterLimit, st.cap, st.join, st.resScale, ignoreCenter)
	defer releasePathStroker(stroker)

	iter := path.NewIter(src, false)
	lastSegment := path.VerbMove
	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			stroker.moveTo(pts[0])
		case path.VerbLine:
			stroker.lineTo(pts[1], iter)
			lastSegment = path.VerbLine
		case path.VerbQuad:
			stroker.quadTo(pts[1], pts[2])
			lastSegment = path.VerbQuad
		case path.VerbConic:
			stroker.conicTo(pts[1], pts[2], iter.ConicWeight())
			lastSegment = path.VerbConic
		case path.VerbCubic:
			stroker.cubicTo(pts[1], pts[2], pts[3])
			lastSegment = path.VerbCubic
		case path.VerbClose:
			if st.cap != CapButt {
				/* If the stroke consists of a moveTo followed by a close, treat it
				   as if it were followed by a zero-length line. Lines without length
				   can have square and round end caps. */
				if stroker.hasOnlyMoveTo() {
					stroker.lineTo(stroker.moveToPt(), nil)
					lastSegment = path.VerbLine
					continue
				}
				/* If the stroke consists of a moveTo followed by one or more zero-length
				   verbs, then followed by a close, treat is as if it were followed by a
				   zero-length line. Lines without length can have square & round end caps. */
				if stroker.isCurrentContourEmpty() {
					lastSegment = path.VerbLine
					continue
				}
			}
			stroker.close(lastSegment == path.VerbLine)
		}
	}
	stroker.done(dst, lastSegment == path.VerbLine)

	if st.doFill && !ignoreCenter {
		if src.ComputeFirstDirectionRaw() == path.FirstDirectionCCW {
			dst.ReverseAddPath(src)
		} else {
			dst.AddPath(src, path.AddPathAppend)
		}
	}

	// our answer should preserve the inverseness of the src
	if src.IsInverseFillType() {
		dst.ToggleInverseFillType()
	}
}

// reverseDirection returns the opposite path direction.
func reverseDirection(dir geom.PathDirection) geom.PathDirection {
	if dir == geom.DirectionCW {
		return geom.DirectionCCW
	}
	return geom.DirectionCW
}

// addBevel appends the bevel-join stroke outline of rect r with outer bounds outer.
func addBevel(p *path.Path, r, outer geom.Rect, dir geom.PathDirection) {
	var pts [8]geom.Point
	if dir == geom.DirectionCW {
		pts[0] = geom.Point{X: r.Left, Y: outer.Top}
		pts[1] = geom.Point{X: r.Right, Y: outer.Top}
		pts[2] = geom.Point{X: outer.Right, Y: r.Top}
		pts[3] = geom.Point{X: outer.Right, Y: r.Bottom}
		pts[4] = geom.Point{X: r.Right, Y: outer.Bottom}
		pts[5] = geom.Point{X: r.Left, Y: outer.Bottom}
		pts[6] = geom.Point{X: outer.Left, Y: r.Bottom}
		pts[7] = geom.Point{X: outer.Left, Y: r.Top}
	} else {
		pts[7] = geom.Point{X: r.Left, Y: outer.Top}
		pts[6] = geom.Point{X: r.Right, Y: outer.Top}
		pts[5] = geom.Point{X: outer.Right, Y: r.Top}
		pts[4] = geom.Point{X: outer.Right, Y: r.Bottom}
		pts[3] = geom.Point{X: r.Right, Y: outer.Bottom}
		pts[2] = geom.Point{X: r.Left, Y: outer.Bottom}
		pts[1] = geom.Point{X: outer.Left, Y: r.Bottom}
		pts[0] = geom.Point{X: outer.Left, Y: r.Top}
	}
	p.AddPoly(pts[:], true)
}

// StrokeRect strokes origRect into dst, resetting dst first.
func (st *Stroke) StrokeRect(origRect geom.Rect, dst *path.Path, dir geom.PathDirection) {
	dst.Reset()

	radius := st.width / 2
	if radius <= 0 {
		return
	}

	rw := origRect.Width()
	rh := origRect.Height()
	if (rw < 0) != (rh < 0) {
		dir = reverseDirection(dir)
	}
	rect := origRect.Sorted()
	// reassign these, now that we know they'll be >= 0
	rw = rect.Width()
	rh = rect.Height()

	r := rect.Outset(radius, radius)

	join := st.join
	if join == JoinMiter && st.miterLimit < float32(math.Sqrt2) {
		join = JoinBevel
	}

	switch join {
	case JoinMiter:
		dst.AddRect(r, dir)
	case JoinBevel:
		addBevel(dst, rect, r, dir)
	case JoinRound:
		dst.AddRRect(geom.MakeRRect(r, radius, radius), dir)
	}

	if st.width < min(rw, rh) && !st.doFill {
		r = rect.Inset(radius, radius)
		dst.AddRect(r, reverseDirection(dir))
	}
}
