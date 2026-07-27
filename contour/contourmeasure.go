// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Per-contour arc-length measurement over a path, with position/tangent queries and sub-segment extraction. The path
// effects (dash/trim/discrete/1D) and their PathMeasure convenience wrapper sit on top of this.

package contour

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// segType identifies the curve kind a measured segment belongs to.
type segType uint8

const (
	segLine segType = iota
	segQuad
	segCubic
	segConic
)

// maxTValue is the largest stored t value: segment t values are stored as 30-bit integers.
const maxTValue = 0x3FFFFFFF

// tValue2Scalar converts a stored t to a float in [0, 1]. 1/maxTValue can't be represented as a float, but it's close
// and the limits work fine (0 -> 0.0, maxTValue -> 1.0 exactly).
func tValue2Scalar(t int32) float32 {
	const maxTReciprocal = float32(1.0) / float32(maxTValue)
	return float32(t) * maxTReciprocal
}

// segment records the accumulated distance up to a point on one curve of the contour.
type segment struct {
	distance float32 // total distance up to this point
	ptIndex  int32   // index into the pts array
	tValue   int32   // 30 bits
	segType  segType
}

func (s *segment) scalarT() float32 {
	return tValue2Scalar(s.tValue)
}

// Measure is the arc-length measurement of a single contour.
type Measure struct {
	segments []segment
	pts      []geom.Point // points used to define the segments
	length   float32
	isClosed bool
}

// Length returns the total length of the contour.
func (m *Measure) Length() float32 { return m.length }

// IsClosed reports whether the contour is closed.
func (m *Measure) IsClosed() bool { return m.isClosed }

///////////////////////////////////////////////////////////////////////////////
// segment emission

// segTo appends the [startT, stopT] slice of the segment starting at pts to dst. startT and stopT are in [0, 1] with
// startT <= stopT.
func segTo(pts []geom.Point, seg segType, startT, stopT float32, dst *path.Path) {
	if startT == stopT {
		if !dst.IsEmpty() {
			// if the dash as a zero-length on segment, add a corresponding zero-length line. The stroke code will add
			// end caps to zero length lines as appropriate
			if lastPt, ok := dst.LastPt(); ok {
				dst.LineToPt(lastPt)
			}
		}
		return
	}

	var tmp0, tmp1 [7]geom.Point

	switch seg {
	case segLine:
		if stopT == 1 {
			dst.LineToPt(pts[1])
		} else {
			dst.LineTo(geom.ScalarInterp(pts[0].X, pts[1].X, stopT), geom.ScalarInterp(pts[0].Y, pts[1].Y, stopT))
		}
	case segQuad:
		if startT == 0 {
			if stopT == 1 {
				dst.QuadToPt(pts[1], pts[2])
			} else {
				geom.ChopQuadAt(pts, tmp0[:], stopT)
				dst.QuadToPt(tmp0[1], tmp0[2])
			}
		} else {
			geom.ChopQuadAt(pts, tmp0[:], startT)
			if stopT == 1 {
				dst.QuadToPt(tmp0[3], tmp0[4])
			} else {
				geom.ChopQuadAt(tmp0[2:], tmp1[:], (stopT-startT)/(1-startT))
				dst.QuadToPt(tmp1[1], tmp1[2])
			}
		}
	case segConic:
		conic := geom.MakeConic(pts[0], pts[2], pts[3], pts[1].X)
		if startT == 0 {
			if stopT == 1 {
				dst.ConicToPt(conic.Pts[1], conic.Pts[2], conic.W)
			} else {
				var tmp [2]geom.Conic
				if conic.ChopAt(stopT, &tmp) {
					dst.ConicToPt(tmp[0].Pts[1], tmp[0].Pts[2], tmp[0].W)
				}
			}
		} else {
			if stopT == 1 {
				var tmp [2]geom.Conic
				if conic.ChopAt(startT, &tmp) {
					dst.ConicToPt(tmp[1].Pts[1], tmp[1].Pts[2], tmp[1].W)
				}
			} else {
				var tmp geom.Conic
				conic.ChopAtRange(&tmp, startT, stopT)
				dst.ConicToPt(tmp.Pts[1], tmp.Pts[2], tmp.W)
			}
		}
	case segCubic:
		if startT == 0 {
			if stopT == 1 {
				dst.CubicToPt(pts[1], pts[2], pts[3])
			} else {
				geom.ChopCubicAt(pts, tmp0[:], stopT)
				dst.CubicToPt(tmp0[1], tmp0[2], tmp0[3])
			}
		} else {
			geom.ChopCubicAt(pts, tmp0[:], startT)
			if stopT == 1 {
				dst.CubicToPt(tmp0[4], tmp0[5], tmp0[6])
			} else {
				geom.ChopCubicAt(tmp0[3:], tmp1[:], (stopT-startT)/(1-startT))
				dst.CubicToPt(tmp1[1], tmp1[2], tmp1[3])
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// curviness tests

// tspanBigEnough reports whether the stored-t span is still wide enough to be worth subdividing.
func tspanBigEnough(tspan int32) bool {
	return tspan>>10 != 0
}

// cheapDistLimit is the base flatness tolerance: can't use tangents, since we need [0..1..................2] to be seen
// as definitely not a line (it is when drawn, but not parametrically), so we compare midpoints.
const cheapDistLimit = float32(0.5) // just made this value up

// quadTooCurvy reports whether the quad's midpoint strays from its endpoints' midpoint by more than the tolerance.
func quadTooCurvy(pts []geom.Point, tolerance float32) bool {
	// diff = (a/4 + b/2 + c/4) - (a/2 + c/2) diff = -a/4 + b/2 - c/4
	dx := 0.5*pts[1].X - 0.5*(0.5*(pts[0].X+pts[2].X))
	dy := 0.5*pts[1].Y - 0.5*(0.5*(pts[0].Y+pts[2].Y))
	dist := max(geom.ScalarAbs(dx), geom.ScalarAbs(dy))
	return dist > tolerance
}

// conicTooCurvy reports whether the conic's mid-t point strays from its endpoints' midpoint by more than the tolerance.
func conicTooCurvy(firstPt, midTPt, lastPt geom.Point, tolerance float32) bool {
	midEnds := firstPt.Add(lastPt).Scaled(0.5)
	dxy := midTPt.Sub(midEnds)
	dist := max(geom.ScalarAbs(dxy.X), geom.ScalarAbs(dxy.Y))
	return dist > tolerance
}

// cheapDistExceedsLimit reports whether pt strays from (x, y) by more than the tolerance (Chebyshev distance).
func cheapDistExceedsLimit(pt geom.Point, x, y, tolerance float32) bool {
	dist := max(geom.ScalarAbs(x-pt.X), geom.ScalarAbs(y-pt.Y))
	// just made up the 1/2
	return dist > tolerance
}

// cubicTooCurvy reports whether either cubic control point strays too far from the corresponding third of the endpoint
// chord.
func cubicTooCurvy(pts []geom.Point, tolerance float32) bool {
	return cheapDistExceedsLimit(pts[1],
		geom.ScalarInterp(pts[0].X, pts[3].X, 1.0/3),
		geom.ScalarInterp(pts[0].Y, pts[3].Y, 1.0/3), tolerance) ||
		cheapDistExceedsLimit(pts[2],
			geom.ScalarInterp(pts[0].X, pts[3].X, 2.0/3),
			geom.ScalarInterp(pts[0].Y, pts[3].Y, 2.0/3), tolerance)
}

// maxRecursionDepth puts a cap on the total size of our output, since the client can pass in arbitrarily large values
// for resScale.
const maxRecursionDepth = 8

///////////////////////////////////////////////////////////////////////////////
// Iter

// Iter iterates the contours of a path, producing a Measure per non-empty contour. The path is not copied; it must not
// be mutated while the Iter is in use.
type Iter struct {
	iter *path.RawIter
	// temporaries reused across buildSegments calls
	segments    []segment
	pts         []geom.Point
	tolerance   float32
	forceClosed bool
}

// NewIter returns an iterator over p's contours. forceClosed measures every contour as if closed; a larger resScale
// asks for finer curve approximation.
func NewIter(p *path.Path, forceClosed bool, resScale float32) *Iter {
	it := &Iter{}
	it.Reset(p, forceClosed, resScale)
	return it
}

// Reset assigns a new path, restarting iteration.
func (it *Iter) Reset(p *path.Path, forceClosed bool, resScale float32) {
	if p != nil && p.IsFinite() {
		it.iter = path.NewRawIter(p)
		it.tolerance = cheapDistLimit * geom.IEEEFloatDivide(1, resScale)
		it.forceClosed = forceClosed
	} else {
		it.iter = nil
	}
}

// Next returns the measurement of the next non-empty contour, or nil when the path is exhausted.
func (it *Iter) Next() *Measure {
	if it.iter == nil {
		return nil
	}
	for it.hasNextSegments() {
		if cm := it.buildSegments(); cm != nil {
			return cm
		}
	}
	return nil
}

func (it *Iter) hasNextSegments() bool {
	_, ok := it.iter.PeekVerb()
	return ok
}

// computeLineSeg accumulates one line's length, appending its segment record when it advanced the distance.
func (it *Iter) computeLineSeg(p0, p1 geom.Point, distance float32, ptIndex int32) float32 {
	d := p0.DistanceTo(p1)
	prevD := distance
	distance += d
	if distance > prevD {
		it.segments = append(it.segments, segment{
			distance: distance,
			ptIndex:  ptIndex,
			segType:  segLine,
			tValue:   maxTValue,
		})
	}
	return distance
}

// computeQuadSegs recursively subdivides a quad until flat enough, accumulating chord lengths.
func (it *Iter) computeQuadSegs(pts []geom.Point, distance float32, mint, maxt, ptIndex int32, recursionDepth int) float32 {
	if recursionDepth < maxRecursionDepth && tspanBigEnough(maxt-mint) && quadTooCurvy(pts, it.tolerance) {
		var tmp [5]geom.Point
		halft := (mint + maxt) >> 1
		geom.ChopQuadAtHalf(pts, tmp[:])
		recursionDepth++
		distance = it.computeQuadSegs(tmp[:3], distance, mint, halft, ptIndex, recursionDepth)
		distance = it.computeQuadSegs(tmp[2:], distance, halft, maxt, ptIndex, recursionDepth)
	} else {
		d := pts[0].DistanceTo(pts[2])
		prevD := distance
		distance += d
		if distance > prevD {
			it.segments = append(it.segments, segment{
				distance: distance,
				ptIndex:  ptIndex,
				segType:  segQuad,
				tValue:   maxt,
			})
		}
	}
	return distance
}

// computeConicSegs recursively subdivides a conic until flat enough, accumulating chord lengths.
func (it *Iter) computeConicSegs(conic *geom.Conic, distance float32, mint int32, minPt geom.Point, maxt int32, maxPt geom.Point, ptIndex int32, recursionDepth int) float32 {
	halft := (mint + maxt) >> 1
	halfPt := conic.EvalAt(tValue2Scalar(halft))
	if !halfPt.IsFinite() {
		return distance
	}
	if recursionDepth < maxRecursionDepth && tspanBigEnough(maxt-mint) &&
		conicTooCurvy(minPt, halfPt, maxPt, it.tolerance) {
		recursionDepth++
		distance = it.computeConicSegs(conic, distance, mint, minPt, halft, halfPt, ptIndex, recursionDepth)
		distance = it.computeConicSegs(conic, distance, halft, halfPt, maxt, maxPt, ptIndex, recursionDepth)
	} else {
		d := minPt.DistanceTo(maxPt)
		prevD := distance
		distance += d
		if distance > prevD {
			it.segments = append(it.segments, segment{
				distance: distance,
				ptIndex:  ptIndex,
				segType:  segConic,
				tValue:   maxt,
			})
		}
	}
	return distance
}

// computeCubicSegs recursively subdivides a cubic until flat enough, accumulating chord lengths.
func (it *Iter) computeCubicSegs(pts []geom.Point, distance float32, mint, maxt, ptIndex int32, recursionDepth int) float32 {
	if recursionDepth < maxRecursionDepth && tspanBigEnough(maxt-mint) && cubicTooCurvy(pts, it.tolerance) {
		var tmp [7]geom.Point
		halft := (mint + maxt) >> 1
		geom.ChopCubicAtHalf(pts, tmp[:])
		recursionDepth++
		distance = it.computeCubicSegs(tmp[:4], distance, mint, halft, ptIndex, recursionDepth)
		distance = it.computeCubicSegs(tmp[3:], distance, halft, maxt, ptIndex, recursionDepth)
	} else {
		d := pts[0].DistanceTo(pts[3])
		prevD := distance
		distance += d
		if distance > prevD {
			it.segments = append(it.segments, segment{
				distance: distance,
				ptIndex:  ptIndex,
				segType:  segCubic,
				tValue:   maxt,
			})
		}
	}
	return distance
}

// buildSegments measures the next contour, returning nil when it produced no segments (empty or degenerate contour).
func (it *Iter) buildSegments() *Measure {
	ptIndex := int32(-1)
	distance := float32(0)
	haveSeenClose := it.forceClosed
	haveSeenMoveTo := false

	// Note: as we accumulate distance, we have to check that the result of += actually made it larger, since a very
	// small delta might be > 0, but still have no effect on distance (if distance >>> delta).
	//
	// We do this check below, and in compute_quad_segs and compute_cubic_segs

	it.segments = it.segments[:0]
	it.pts = it.pts[:0]

	for {
		verb, ok := it.iter.PeekVerb()
		if !ok {
			break
		}
		if haveSeenMoveTo && verb == path.VerbMove {
			break
		}
		_, pts, w, _ := it.iter.Next()
		switch verb {
		case path.VerbMove:
			ptIndex++
			it.pts = append(it.pts, pts[0])
			haveSeenMoveTo = true

		case path.VerbLine:
			prevD := distance
			distance = it.computeLineSeg(pts[0], pts[1], distance, ptIndex)
			if distance > prevD {
				it.pts = append(it.pts, pts[1])
				ptIndex++
			}

		case path.VerbQuad:
			prevD := distance
			distance = it.computeQuadSegs(pts[:3], distance, 0, maxTValue, ptIndex, 0)
			if distance > prevD {
				it.pts = append(it.pts, pts[1], pts[2])
				ptIndex += 2
			}

		case path.VerbConic:
			conic := geom.MakeConic(pts[0], pts[1], pts[2], w)
			prevD := distance
			distance = it.computeConicSegs(&conic, distance, 0, conic.Pts[0], maxTValue, conic.Pts[2], ptIndex, 0)
			if distance > prevD {
				// we store the conic weight in our next point, followed by the last 2 pts; to reconstitute a conic, use
				// MakeConic(pts[0], pts[2], pts[3], weight = pts[1].X)
				it.pts = append(it.pts, geom.Point{X: conic.W}, pts[1], pts[2])
				ptIndex += 3
			}

		case path.VerbCubic:
			prevD := distance
			distance = it.computeCubicSegs(pts[:4], distance, 0, maxTValue, ptIndex, 0)
			if distance > prevD {
				it.pts = append(it.pts, pts[1], pts[2], pts[3])
				ptIndex += 3
			}

		case path.VerbClose:
			haveSeenClose = true
		}
	}

	if !scalarIsFinite(distance) {
		return nil
	}
	if len(it.segments) == 0 {
		return nil
	}

	if haveSeenClose {
		prevD := distance
		firstPt := it.pts[0]
		distance = it.computeLineSeg(it.pts[ptIndex], firstPt, distance, ptIndex)
		if distance > prevD {
			it.pts = append(it.pts, firstPt)
		}
	}

	m := &Measure{
		segments: append([]segment(nil), it.segments...),
		pts:      append([]geom.Point(nil), it.pts...),
		length:   distance,
		isClosed: haveSeenClose,
	}
	return m
}

func scalarIsFinite(x float32) bool {
	return math.Float32bits(x)&0x7F800000 != 0x7F800000
}

func scalarIsNaN(x float32) bool {
	return x != x
}

///////////////////////////////////////////////////////////////////////////////
// queries

// computePosTan returns the position and normalized tangent of the segment starting at pts at parameter t.
func computePosTan(pts []geom.Point, seg segType, t float32, wantPos, wantTan bool) (pos, tangent geom.Point) {
	switch seg {
	case segLine:
		if wantPos {
			pos = geom.Point{
				X: geom.ScalarInterp(pts[0].X, pts[1].X, t),
				Y: geom.ScalarInterp(pts[0].Y, pts[1].Y, t),
			}
		}
		if wantTan {
			tangent.SetNormalize(pts[1].X-pts[0].X, pts[1].Y-pts[0].Y)
		}
	case segQuad:
		if wantPos {
			pos = geom.EvalQuadAt(pts, t)
		}
		if wantTan {
			tangent = geom.EvalQuadTangentAt(pts, t)
			tangent.Normalize()
		}
	case segConic:
		conic := geom.MakeConic(pts[0], pts[2], pts[3], pts[1].X)
		if wantPos {
			pos = conic.EvalAt(t)
		}
		if wantTan {
			tangent = conic.EvalTangentAt(t)
			tangent.Normalize()
		}
	case segCubic:
		if wantPos {
			pos = geom.EvalCubicPosAt(pts, t)
		}
		if wantTan {
			tangent = geom.EvalCubicTangentAt(pts, t)
			tangent.Normalize()
		}
	}
	return pos, tangent
}

// distanceToSegment locates the segment containing the given distance, returning the segment index and the interpolated
// t within it. A Measure with no segments (e.g. the zero value) has nothing to locate, so it reports an index of -1 and
// a NaN t; callers must reject that before indexing.
func (m *Measure) distanceToSegment(distance float32) (segIndex int, t float32) {
	segs := m.segments
	count := len(segs)
	if count == 0 {
		return -1, float32(math.NaN())
	}

	// binary search over segment distances
	lo, hi := 0, count-1
	for lo < hi {
		mid := (hi + lo) >> 1
		if segs[mid].distance < distance {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	index := hi
	if segs[hi].distance < distance {
		index = hi + 1
		index = ^index
	} else if distance < segs[hi].distance {
		index = ^index
	}
	// don't care if we hit an exact match or not, so we xor index if it is negative
	if index < 0 {
		index = ^index
	}
	seg := &segs[index]

	// now interpolate t-values with the prev segment (if possible)
	var startT, startD float32
	// check if the prev segment is legal, and references the same set of points
	if index > 0 {
		startD = segs[index-1].distance
		if segs[index-1].ptIndex == seg.ptIndex {
			startT = segs[index-1].scalarT()
		}
	}

	t = startT + (seg.scalarT()-startT)*(distance-startD)/(seg.distance-startD)
	return index, t
}

// GetPosTan pins distance to [0, length] and returns the corresponding position and tangent. wantPos/wantTan select the
// outputs (both may be true). Reports false for a NaN distance or a Measure with no segments (e.g. the zero value).
func (m *Measure) GetPosTan(distance float32, wantPos, wantTan bool) (pos, tangent geom.Point, ok bool) {
	if scalarIsNaN(distance) {
		return pos, tangent, false
	}

	length := m.length

	// pin the distance to a legal range
	if distance < 0 {
		distance = 0
	} else if distance > length {
		distance = length
	}

	segIndex, t := m.distanceToSegment(distance)
	if segIndex < 0 || scalarIsNaN(t) {
		return pos, tangent, false
	}

	seg := &m.segments[segIndex]
	pos, tangent = computePosTan(m.pts[seg.ptIndex:], seg.segType, t, wantPos, wantTan)
	return pos, tangent, true
}

// MatrixFlags selects which components GetMatrix builds.
type MatrixFlags uint8

// MatrixFlags values.
const (
	GetPositionMatrixFlag  MatrixFlags = 0x01
	GetTangentMatrixFlag   MatrixFlags = 0x02
	GetPosAndTanMatrixFlag             = GetPositionMatrixFlag | GetTangentMatrixFlag
)

// GetMatrix builds the matrix positioning (and optionally orienting, per flags) geometry at the given distance along
// the contour. Reports false, leaving matrix untouched, whenever GetPosTan does.
func (m *Measure) GetMatrix(distance float32, matrix *geom.Matrix, flags MatrixFlags) bool {
	position, tangent, ok := m.GetPosTan(distance, true, true)
	if !ok {
		return false
	}
	if matrix != nil {
		if flags&GetTangentMatrixFlag != 0 {
			matrix.SetSinCos(tangent.Y, tangent.X)
		} else {
			matrix.SetIdentity()
		}
		if flags&GetPositionMatrixFlag != 0 {
			matrix.PostTranslate(position.X, position.Y)
		}
	}
	return true
}

// nextSegment returns the index of the first segment referencing a different point set.
func (m *Measure) nextSegment(segIndex int) int {
	ptIndex := m.segments[segIndex].ptIndex
	for {
		segIndex++
		if m.segments[segIndex].ptIndex != ptIndex {
			return segIndex
		}
	}
}

// GetSegment appends the contour slice from startD to stopD to dst. If startWithMoveTo is true, the slice begins with a
// moveTo to its starting position.
func (m *Measure) GetSegment(startD, stopD float32, dst *path.Path, startWithMoveTo bool) bool {
	length := m.length

	if startD < 0 {
		startD = 0
	}
	if stopD > length {
		stopD = length
	}
	if !(startD <= stopD) { // catch NaN values as well
		return false
	}
	if len(m.segments) == 0 {
		return false
	}

	segIndex, startT := m.distanceToSegment(startD)
	if !scalarIsFinite(startT) {
		return false
	}
	stopSegIndex, stopT := m.distanceToSegment(stopD)
	if !scalarIsFinite(stopT) {
		return false
	}

	seg := &m.segments[segIndex]
	stopSeg := &m.segments[stopSegIndex]
	if startWithMoveTo {
		p, _ := computePosTan(m.pts[seg.ptIndex:], seg.segType, startT, true, false)
		dst.MoveToPt(p)
	}

	if seg.ptIndex == stopSeg.ptIndex {
		segTo(m.pts[seg.ptIndex:], seg.segType, startT, stopT, dst)
	} else {
		for {
			segTo(m.pts[seg.ptIndex:], seg.segType, startT, 1, dst)
			segIndex = m.nextSegment(segIndex)
			seg = &m.segments[segIndex]
			startT = 0
			if seg.ptIndex >= stopSeg.ptIndex {
				break
			}
		}
		segTo(m.pts[seg.ptIndex:], seg.segType, 0, stopT, dst)
	}

	return true
}
