// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Turns a path.Path (and, for a binary operation, a second operand path) into the contour -> segment -> span data model
// built in the opcontour/opsegment/opspan slice. preFetch flattens the path's verb/point/weight streams (running the
// reduceOrder family to drop degenerate curves and forceSmallToZero to snap tiny coordinates), then walk feeds the
// contour builder, splitting quads/conics at max curvature and complex cubics via complexBreak (dcubic.go) so
// downstream intersection succeeds.

package pathops

import (
	"math"
	"slices"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// pathOpsMask identifies which fill rule (winding vs. even-odd) governs a path's contours.
type pathOpsMask int

const (
	windingMask pathOpsMask = -1
	noMask      pathOpsMask = 0
	evenOddMask pathOpsMask = 1
)

// opVerbDone is the internal terminator stored at the end of the fetched verb stream, one past the last real verb value
// (path.VerbClose). The path package does not expose a done verb of its own.
const opVerbDone path.Verb = 6

// forceSmallToZero snaps a point's coordinates to zero when they are already too tiny to be numerically meaningful,
// since very small nonzero coordinates cause instability further down the pipeline.
func forceSmallToZero(pt geom.Point) geom.Point {
	if math.Abs(float64(pt.X)) < fltEpsilonOrderableErr {
		pt.X = 0
	}
	if math.Abs(float64(pt.Y)) < fltEpsilonOrderableErr {
		pt.Y = 0
	}
	return pt
}

// pointsAreFinite reports whether every point in pts is finite.
func pointsAreFinite(pts []geom.Point) bool {
	for i := range pts {
		if !pts[i].IsFinite() {
			return false
		}
	}
	return true
}

// canAddCurve snaps the curve's points to zero and reports false for a move verb or a degenerate (zero-length) line,
// both of which the contour builder should skip. The points are snapped in place, so the caller adds the snapped curve.
func canAddCurve(verb path.Verb, curve []geom.Point) bool {
	if verb == path.VerbMove {
		return false
	}
	for i := 0; i <= opVerbToPoints(verb); i++ {
		curve[i] = forceSmallToZero(curve[i])
	}
	return verb != path.VerbLine || !dPointsApproximatelyEqual(curve[0], curve[1])
}

// opEdgeBuilder converts one or two path.Path operands into the contour/segment/span model.
type opEdgeBuilder struct {
	globalState       *opGlobalState
	path              *path.Path
	contoursHead      *opContourHead
	pathPts           []geom.Point
	weights           []float32
	pathVerbs         []path.Verb
	contourBuilder    opContourBuilder
	xorMask           [2]pathOpsMask
	secondHalf        int
	operand           bool
	allowOpenContours bool
	unparseable       bool
}

// newOpEdgeBuilder creates a builder over p, with the contour builder starting on the head contour.
func newOpEdgeBuilder(p *path.Path, contours *opContourHead, globalState *opGlobalState) *opEdgeBuilder {
	b := &opEdgeBuilder{
		globalState:  globalState,
		path:         p,
		contoursHead: contours,
	}
	b.contourBuilder.contour = &contours.opContour
	b.init()
	return b
}

// init records the first operand's fill mask and pre-fetches its verb/point/weight streams.
func (b *opEdgeBuilder) init() {
	b.operand = false
	mask := windingMask
	if int(b.path.FillType())&1 != 0 {
		mask = evenOddMask
	}
	b.xorMask[0] = mask
	b.xorMask[1] = mask
	b.unparseable = false
	b.secondHalf = b.preFetch()
}

// addOperand appends a second path's verbs after popping the first's done terminator, recording the second operand's
// fill mask.
func (b *opEdgeBuilder) addOperand(p *path.Path) {
	b.pathVerbs = b.pathVerbs[:len(b.pathVerbs)-1] // drop the trailing done
	b.path = p
	mask := windingMask
	if int(p.FillType())&1 != 0 {
		mask = evenOddMask
	}
	b.xorMask[1] = mask
	b.preFetch()
}

// complete flushes the contour builder and, if the current contour holds any segments, finalizes it and clears the
// builder's current contour.
func (b *opEdgeBuilder) complete() {
	b.contourBuilder.flush()
	contour := b.contourBuilder.contour
	if contour != nil && contour.count > 0 {
		contour.complete()
		b.contourBuilder.setContour(nil)
	}
}

// close completes the current contour. It always succeeds; the bool result lets callers use it interchangeably with the
// fallible steps in walk.
func (b *opEdgeBuilder) close() bool {
	b.complete()
	return true
}

// finish walks the fetched path into the data model, then trims a trailing empty contour. Returns false if the path was
// unparseable or a contour failed to build.
func (b *opEdgeBuilder) finish() bool {
	b.operand = false
	if b.unparseable || !b.walk() {
		return false
	}
	b.complete()
	contour := b.contourBuilder.contour
	if contour != nil && contour.count == 0 {
		b.contoursHead.remove(contour)
	}
	return true
}

// closeContour materializes the closing line (or folds it into the last line) before appending a close verb.
func (b *opEdgeBuilder) closeContour(curveEnd, curveStart geom.Point) {
	if !dPointsApproximatelyEqual(curveEnd, curveStart) {
		b.pathVerbs = append(b.pathVerbs, path.VerbLine)
		b.pathPts = append(b.pathPts, curveStart)
	} else {
		verbCount := len(b.pathVerbs)
		ptsCount := len(b.pathPts)
		if b.pathVerbs[verbCount-1] == path.VerbLine && b.pathPts[ptsCount-2] == curveStart {
			b.pathVerbs = b.pathVerbs[:verbCount-1]
			b.pathPts = b.pathPts[:ptsCount-1]
		} else {
			b.pathPts[ptsCount-1] = curveStart
		}
	}
	b.pathVerbs = append(b.pathVerbs, path.VerbClose)
}

// preFetch flattens the path's verbs/points/weights, reducing degenerate curves and closing open contours. Returns the
// index of the done terminator (the split between operands).
func (b *opEdgeBuilder) preFetch() int {
	if !b.path.IsFinite() {
		b.unparseable = true
		return 0
	}
	var curveStart geom.Point
	var curve [4]geom.Point
	lastCurve := false
	it := path.NewRawIter(b.path)
	for {
		rawVerb, pts, w, ok := it.Next()
		if !ok {
			break
		}
		verb := rawVerb
		switch verb {
		case path.VerbMove:
			if !b.allowOpenContours && lastCurve {
				b.closeContour(curve[0], curveStart)
			}
			b.pathVerbs = append(b.pathVerbs, verb)
			curve[0] = forceSmallToZero(pts[0])
			b.pathPts = append(b.pathPts, curve[0])
			curveStart = curve[0]
			lastCurve = false
			continue
		case path.VerbLine:
			curve[1] = forceSmallToZero(pts[1])
			if dPointsApproximatelyEqual(curve[0], curve[1]) {
				lastVerb := b.pathVerbs[len(b.pathVerbs)-1]
				if lastVerb != path.VerbLine && lastVerb != path.VerbMove {
					curve[0] = curve[1]
					b.pathPts[len(b.pathPts)-1] = curve[1]
				}
				continue // skip degenerate points
			}
		case path.VerbQuad:
			curve[1] = forceSmallToZero(pts[1])
			curve[2] = forceSmallToZero(pts[2])
			verb = reduceOrderQuad([3]geom.Point{curve[0], curve[1], curve[2]}, curve[0:])
			if verb == path.VerbMove {
				continue // skip degenerate points
			}
		case path.VerbConic:
			curve[1] = forceSmallToZero(pts[1])
			curve[2] = forceSmallToZero(pts[2])
			verb = reduceOrderQuad([3]geom.Point{curve[0], curve[1], curve[2]}, curve[0:])
			if verb == path.VerbQuad && w != 1 {
				verb = path.VerbConic
			} else if verb == path.VerbMove {
				continue // skip degenerate points
			}
		case path.VerbCubic:
			curve[1] = forceSmallToZero(pts[1])
			curve[2] = forceSmallToZero(pts[2])
			curve[3] = forceSmallToZero(pts[3])
			verb = reduceOrderCubic([4]geom.Point{curve[0], curve[1], curve[2], curve[3]}, curve[0:])
			if verb == path.VerbMove {
				continue // skip degenerate points
			}
		case path.VerbClose:
			b.closeContour(curve[0], curveStart)
			lastCurve = false
			continue
		}
		b.pathVerbs = append(b.pathVerbs, verb)
		ptCount := opVerbToPoints(verb)
		b.pathPts = append(b.pathPts, curve[1:1+ptCount]...)
		if verb == path.VerbConic {
			b.weights = append(b.weights, w)
		}
		curve[0] = curve[ptCount]
		lastCurve = true
	}
	if !b.allowOpenContours && lastCurve {
		b.closeContour(curve[0], curveStart)
	}
	b.pathVerbs = append(b.pathVerbs, opVerbDone)
	return len(b.pathVerbs) - 1
}

// walk drives the contour builder over the fetched verbs, splitting quads, conics, and complex cubics as required for
// intersection.
func (b *opEdgeBuilder) walk() bool {
	verbIdx := 0
	endOfFirstHalf := b.secondHalf
	pointsIdx := 0
	weightIdx := 0
	contour := b.contourBuilder.contour
	moveToPtrBump := 0
	for verbIdx < len(b.pathVerbs) && b.pathVerbs[verbIdx] != opVerbDone {
		verb := b.pathVerbs[verbIdx]
		if verbIdx == endOfFirstHalf {
			b.operand = true
		}
		verbIdx++
		switch verb {
		case path.VerbMove:
			if contour != nil && contour.count > 0 {
				if b.allowOpenContours {
					b.complete()
				} else if !b.close() {
					return false
				}
			}
			if contour == nil {
				contour = b.contoursHead.appendContour()
				b.contourBuilder.setContour(contour)
			}
			contour.init(b.globalState, b.operand, b.xorMask[b2i(b.operand)] == evenOddMask)
			pointsIdx += moveToPtrBump
			moveToPtrBump = 1
			continue
		case path.VerbLine:
			b.contourBuilder.addLine(b.pathPts[pointsIdx:])
		case path.VerbQuad:
			if !b.walkQuad(b.pathPts[pointsIdx:]) {
				return false
			}
		case path.VerbConic:
			b.walkConic(b.pathPts[pointsIdx:], b.weights[weightIdx])
			weightIdx++
		case path.VerbCubic:
			if !b.walkCubic(b.pathPts[pointsIdx:]) {
				return false
			}
		case path.VerbClose:
			if !b.close() {
				return false
			}
			contour = nil
			continue
		default:
			return false
		}
		pointsIdx += opVerbToPoints(verb)
	}
	b.contourBuilder.flush()
	if contour != nil && contour.count > 0 && !b.allowOpenContours && !b.close() {
		return false
	}
	return true
}

// walkQuad is the quad lane of walk: splits a quad whose control tangent reverses at its max curvature into two curves,
// falling back to adding the whole quad. Returns false only when the chopped points are non-finite.
func (b *opEdgeBuilder) walkQuad(p []geom.Point) bool {
	vec1 := p[1].Sub(p[0])
	vec2 := p[2].Sub(p[1])
	if vec1.Dot(vec2) < 0 {
		var pair [5]geom.Point
		if geom.ChopQuadAtMaxCurvature(p[:3], pair[:]) != 1 {
			if !pointsAreFinite(pair[:]) {
				return false
			}
			for i := range pair {
				pair[i] = forceSmallToZero(pair[i])
			}
			var cStorage [2][2]geom.Point
			v1 := reduceOrderQuad([3]geom.Point{pair[0], pair[1], pair[2]}, cStorage[0][:])
			v2 := reduceOrderQuad([3]geom.Point{pair[2], pair[3], pair[4]}, cStorage[1][:])
			curve1 := pair[0:3]
			if v1 == path.VerbLine {
				curve1 = cStorage[0][:]
			}
			curve2 := pair[2:5]
			if v2 == path.VerbLine {
				curve2 = cStorage[1][:]
			}
			if canAddCurve(v1, curve1) && canAddCurve(v2, curve2) {
				b.contourBuilder.addCurve(v1, curve1, 1)
				b.contourBuilder.addCurve(v2, curve2, 1)
				return true
			}
		}
	}
	b.contourBuilder.addQuad(p[:3])
	return true
}

// walkConic is the conic lane of walk: like walkQuad, it splits a conic whose control tangent reverses at its max
// curvature into two conics, falling back to adding the whole conic.
func (b *opEdgeBuilder) walkConic(p []geom.Point, weight float32) {
	vec1 := p[1].Sub(p[0])
	vec2 := p[2].Sub(p[1])
	if vec1.Dot(vec2) < 0 {
		// There is no conic max-curvature routine in this tree, so the conic is measured as if its control hull were a
		// quad's. The substitute is sound rather than exact: the weight shifts where the true maximum sits, so the
		// split lands off it and the two halves are less balanced than they could be. Both halves are still valid
		// conics and the walk still terminates, which makes this curve quality, not correctness -- and writing the
		// real thing would move every conic chop point in the frozen goldens for no known failure.
		maxCurvature := geom.FindQuadMaxCurvature(p[:3])
		if 0 < maxCurvature && maxCurvature < 1 {
			conic := geom.Conic{Pts: [3]geom.Point{p[0], p[1], p[2]}, W: weight}
			var pair [2]geom.Conic
			if !conic.ChopAt(maxCurvature, &pair) {
				// if the result can't be computed, use the original
				b.contourBuilder.addConic(p[:3], weight)
				return
			}
			var cStorage [2][3]geom.Point
			v1 := reduceOrderConic(pair[0], cStorage[0][:])
			v2 := reduceOrderConic(pair[1], cStorage[1][:])
			curve1 := pair[0].Pts[:]
			if v1 == path.VerbLine {
				curve1 = cStorage[0][:]
			}
			curve2 := pair[1].Pts[:]
			if v2 == path.VerbLine {
				curve2 = cStorage[1][:]
			}
			if canAddCurve(v1, curve1) && canAddCurve(v2, curve2) {
				b.contourBuilder.addCurve(v1, curve1, pair[0].W)
				b.contourBuilder.addCurve(v2, curve2, pair[1].W)
				return
			}
		}
	}
	b.contourBuilder.addConic(p[:3], weight)
}

// splitsville holds one sub-range of a cubic being split in walkCubic: its T range, its raw and order-reduced points,
// the reduced verb, and whether it can be added on its own.
type splitsville struct {
	t       [2]float64
	pts     [4]geom.Point
	reduced [4]geom.Point
	verb    path.Verb
	canAdd  bool
}

// walkCubic is the cubic lane of walk: splits complex cubics (self-intersecting or high-curvature) before adding,
// coalescing runs of un-addable sub-curves. Returns false on a non-finite subdivision.
func (b *opEdgeBuilder) walkCubic(p []geom.Point) bool {
	var cubicPts [4]geom.Point
	copy(cubicPts[:], p[:4])
	var splitT [3]float32
	breaks := complexBreak(cubicPts, splitT[:])
	if breaks == 0 {
		b.contourBuilder.addCubic(p[:4])
		return true
	}
	var splits [4]splitsville
	slices.Sort(splitT[:breaks])
	for index := 0; index <= breaks; index++ {
		split := &splits[index]
		if index != 0 {
			split.t[0] = float64(splitT[index-1])
		}
		if index < breaks {
			split.t[1] = float64(splitT[index])
		} else {
			split.t[1] = 1
		}
		part := subDivideCubicPts(cubicPts, split.t[0], split.t[1])
		if !part.toFloatPoints(&split.pts) {
			return false
		}
		split.verb = reduceOrderCubic(split.pts, split.reduced[:])
		curve := split.reduced[:]
		if split.verb == path.VerbCubic {
			curve = split.pts[:]
		}
		split.canAdd = canAddCurve(split.verb, curve)
	}
	for index := 0; index <= breaks; index++ {
		split := &splits[index]
		if !split.canAdd {
			continue
		}
		prior := index
		for prior > 0 && !splits[prior-1].canAdd {
			prior--
		}
		if prior < index {
			split.t[0] = splits[prior].t[0]
			split.pts[0] = splits[prior].pts[0]
		}
		next := index
		breakLimit := min(breaks, len(splits)-1)
		for next < breakLimit && !splits[next+1].canAdd {
			next++
		}
		if next > index {
			split.t[1] = splits[next].t[1]
			split.pts[3] = splits[next].pts[3]
		}
		if prior < index || next > index {
			split.verb = reduceOrderCubic(split.pts, split.reduced[:])
		}
		curve := split.reduced[:]
		if split.verb == path.VerbCubic {
			curve = split.pts[:]
		}
		if !canAddCurve(split.verb, curve) {
			return false
		}
		b.contourBuilder.addCurve(split.verb, curve, 1)
	}
	return true
}
