// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The line, quad, and cubic lanes of degenerate-curve reduction: collapse a curve to the lowest-order form it actually
// represents (point/line/quad/cubic). The reduced geometry is stored in the receiver's line/quad/cubic fields (only one
// is populated per call, playing the role a union would in C). Also includes the
// reduceOrderQuad/reduceOrderConic/reduceOrderCubic verb classifiers (a conic reduces via its quad control points). The
// conic reduce lane arrives with a later curve-pair solver slice.

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// quadratics reports whether the cubic reducer is allowed to collapse a cubic to a quadratic.
type quadratics int

const (
	noQuadratics quadratics = iota
	allowQuadratics
)

// reduceOrder holds the reduced geometry produced by the reduceLine/reduceQuad/reduceCubic methods (only the field
// matching the returned order is populated).
type reduceOrder struct {
	line  dLine
	quad  dQuad
	cubic dCubic
}

// reduceLine reduces a line, returning 1 for a degenerate point or 2 for a proper line.
func (r *reduceOrder) reduceLine(line dLine) int {
	r.line.pts[0] = line.pts[0]
	different := 0
	if !line.pts[0].equals(line.pts[1]) {
		different = 1
	}
	r.line.pts[1] = line.pts[different]
	return 1 + different
}

// reductionLineCountQuad returns 2 if reduction's two line endpoints differ, else 1 (both endpoints collapsed to the
// same point).
func reductionLineCountQuad(reduction dQuad) int {
	return 1 + b2i(!reduction.pts[0].approximatelyEqual(reduction.pts[1]))
}

// reduceQuad reduces a quad to its lowest-order form, returning 1 (point), 2 (line), or 3 (proper quad).
func (r *reduceOrder) reduceQuad(quad dQuad) int {
	var minX, maxX, minY, maxY int
	minXSet, minYSet := 0, 0
	for index := 1; index < 3; index++ {
		if quad.pts[minX].x > quad.pts[index].x {
			minX = index
		}
		if quad.pts[minY].y > quad.pts[index].y {
			minY = index
		}
		if quad.pts[maxX].x < quad.pts[index].x {
			maxX = index
		}
		if quad.pts[maxY].y < quad.pts[index].y {
			maxY = index
		}
	}
	for index := 0; index < 3; index++ {
		if almostEqualUlps(quad.pts[index].x, quad.pts[minX].x) {
			minXSet |= 1 << uint(index)
		}
		if almostEqualUlps(quad.pts[index].y, quad.pts[minY].y) {
			minYSet |= 1 << uint(index)
		}
	}
	if (minXSet&0x05) == 0x5 && (minYSet&0x05) == 0x5 { // test for degenerate
		// this quad starts and ends at the same place, so never contributes to the fill
		r.quad.pts[0] = quad.pts[0]
		r.quad.pts[1] = quad.pts[0]
		return 1
	}
	if minXSet == 0x7 { // test for vertical line
		r.quad.pts[0] = quad.pts[0]
		r.quad.pts[1] = quad.pts[2]
		return reductionLineCountQuad(r.quad)
	}
	if minYSet == 0x7 { // test for horizontal line
		r.quad.pts[0] = quad.pts[0]
		r.quad.pts[1] = quad.pts[2]
		return reductionLineCountQuad(r.quad)
	}
	if quad.isLinear(0, 2) { // four are colinear: return line formed by outside
		r.quad.pts[0] = quad.pts[0]
		r.quad.pts[1] = quad.pts[2]
		return reductionLineCountQuad(r.quad)
	}
	r.quad = quad
	return 3
}

// pointsToVerb maps a reduced point count to its verb: 0->move, 1->line, 2->quad, 3->cubic, via (1<<points)>>1 (which
// deliberately skips the conic verb — reduceOrderConic promotes to it separately).
func pointsToVerb(points int) path.Verb {
	return path.Verb((1 << uint(points)) >> 1)
}

// reduceOrderQuad reduces a float32 quad and returns the resulting verb (move/line/quad); when the quad becomes a line,
// the two reduced line endpoints are written into reducePts.
func reduceOrderQuad(a [3]geom.Point, reducePts []geom.Point) path.Verb {
	var quad dQuad
	quad.set(a)
	var reducer reduceOrder
	order := reducer.reduceQuad(quad)
	if order == 2 { // quad became line
		reducePts[0] = reducer.quad.pts[0].asPoint()
		reducePts[1] = reducer.quad.pts[1].asPoint()
	}
	return pointsToVerb(order - 1)
}

// reduceOrderConic reduces a conic by its float32 quad control points. A quad-order result with weight 1 is a plain
// quad; a quad-order result otherwise is a conic; anything lower passes through unchanged. When it becomes a line, the
// reduced line endpoints are written into reducePts.
func reduceOrderConic(c geom.Conic, reducePts []geom.Point) path.Verb {
	verb := reduceOrderQuad(c.Pts, reducePts)
	if verb > path.VerbLine && c.W == 1 {
		return path.VerbQuad
	}
	if verb == path.VerbQuad {
		return path.VerbConic
	}
	return verb
}

// reduceConic reduces a conic and returns only the verb, discarding any reduced points.
func reduceConic(pts [3]geom.Point, w float32) path.Verb {
	var scratch [2]geom.Point
	return reduceOrderConic(geom.Conic{Pts: pts, W: w}, scratch[:])
}

// coincidentLineCubic handles the case where all four of a cubic's points collapse to one, recording that single point
// as a degenerate zero-length line.
func coincidentLineCubic(cubic dCubic, reduction *dCubic) int {
	reduction.pts[0] = cubic.pts[0]
	reduction.pts[1] = cubic.pts[0]
	return 1
}

// reductionLineCountCubic returns 2 if reduction's two line endpoints differ, else 1 (both endpoints collapsed to the
// same point).
func reductionLineCountCubic(reduction dCubic) int {
	return 1 + b2i(!reduction.pts[0].approximatelyEqual(reduction.pts[1]))
}

// verticalLineCubic reduces a cubic known to be vertical to its outer two points as a line.
func verticalLineCubic(cubic dCubic, reduction *dCubic) int {
	reduction.pts[0] = cubic.pts[0]
	reduction.pts[1] = cubic.pts[3]
	return reductionLineCountCubic(*reduction)
}

// horizontalLineCubic reduces a cubic known to be horizontal to its outer two points as a line.
func horizontalLineCubic(cubic dCubic, reduction *dCubic) int {
	reduction.pts[0] = cubic.pts[0]
	reduction.pts[1] = cubic.pts[3]
	return reductionLineCountCubic(*reduction)
}

// checkQuadraticCubic reports whether cubic degree-reduces to a single quadratic (its control points lie along the
// lines a true quad's degree-elevation to a cubic would produce); if so it stores that quad in reduction and returns 3,
// else 0.
func checkQuadraticCubic(cubic dCubic, reduction *dCubic) int {
	dx10 := cubic.pts[1].x - cubic.pts[0].x
	dx23 := cubic.pts[2].x - cubic.pts[3].x
	midX := cubic.pts[0].x + dx10*3/2
	sideAx := midX - cubic.pts[3].x
	sideBx := dx23 * 3 / 2
	var badX bool
	if approximatelyZero(sideAx) {
		badX = !approximatelyEqual(sideAx, sideBx)
	} else {
		badX = !almostEqualUlpsPin(sideAx, sideBx)
	}
	if badX {
		return 0
	}
	dy10 := cubic.pts[1].y - cubic.pts[0].y
	dy23 := cubic.pts[2].y - cubic.pts[3].y
	midY := cubic.pts[0].y + dy10*3/2
	sideAy := midY - cubic.pts[3].y
	sideBy := dy23 * 3 / 2
	var badY bool
	if approximatelyZero(sideAy) {
		badY = !approximatelyEqual(sideAy, sideBy)
	} else {
		badY = !almostEqualUlpsPin(sideAy, sideBy)
	}
	if badY {
		return 0
	}
	reduction.pts[0] = cubic.pts[0]
	reduction.pts[1].x = midX
	reduction.pts[1].y = midY
	reduction.pts[2] = cubic.pts[3]
	return 3
}

// checkLinearCubic reports whether all four of cubic's points are colinear; if so it stores the outer line in reduction
// and returns its line count, else 0.
func checkLinearCubic(cubic dCubic, reduction *dCubic) int {
	if !cubic.isLinear(0, 3) {
		return 0
	}
	// four are colinear: return line formed by outside
	reduction.pts[0] = cubic.pts[0]
	reduction.pts[1] = cubic.pts[3]
	return reductionLineCountCubic(*reduction)
}

// reduceCubic reduces a cubic to its lowest-order form, returning 1 (point), 2 (line), 3 (quad), or 4 (proper cubic).
// Unlike the quad reducer, the coincidence test here uses a magnitude-relative half-epsilon comparison of each
// coordinate.
func (r *reduceOrder) reduceCubic(cubic dCubic, allow quadratics) int {
	var minX, maxX, minY, maxY int
	minXSet, minYSet := 0, 0
	for index := 1; index < 4; index++ {
		if cubic.pts[minX].x > cubic.pts[index].x {
			minX = index
		}
		if cubic.pts[minY].y > cubic.pts[index].y {
			minY = index
		}
		if cubic.pts[maxX].x < cubic.pts[index].x {
			maxX = index
		}
		if cubic.pts[maxY].y < cubic.pts[index].y {
			maxY = index
		}
	}
	for index := 0; index < 4; index++ {
		cx := cubic.pts[index].x
		cy := cubic.pts[index].y
		denom := math.Max(math.Abs(cx), math.Max(math.Abs(cy),
			math.Max(math.Abs(cubic.pts[minX].x), math.Abs(cubic.pts[minY].y))))
		if denom == 0 {
			minXSet |= 1 << uint(index)
			minYSet |= 1 << uint(index)
			continue
		}
		inv := 1 / denom
		if approximatelyEqualHalf(cx*inv, cubic.pts[minX].x*inv) {
			minXSet |= 1 << uint(index)
		}
		if approximatelyEqualHalf(cy*inv, cubic.pts[minY].y*inv) {
			minYSet |= 1 << uint(index)
		}
	}
	if minXSet == 0xF { // test for vertical line
		if minYSet == 0xF { // return 1 if all four are coincident
			return coincidentLineCubic(cubic, &r.cubic)
		}
		return verticalLineCubic(cubic, &r.cubic)
	}
	if minYSet == 0xF { // test for horizontal line
		return horizontalLineCubic(cubic, &r.cubic)
	}
	if result := checkLinearCubic(cubic, &r.cubic); result != 0 {
		return result
	}
	if allow == allowQuadratics {
		if result := checkQuadraticCubic(cubic, &r.cubic); result != 0 {
			return result
		}
	}
	r.cubic = cubic
	return 4
}

// reduceOrderCubic reduces a float32 cubic and returns the resulting verb (move/line/quad/cubic), allowing quadratic
// collapse; when the cubic becomes a line or a quad the reduced points are written into reducePts, as is the single
// point for the all-coincident move case.
func reduceOrderCubic(a [4]geom.Point, reducePts []geom.Point) path.Verb {
	if dPointsApproximatelyEqual(a[0], a[1]) && dPointsApproximatelyEqual(a[0], a[2]) &&
		dPointsApproximatelyEqual(a[0], a[3]) {
		reducePts[0] = a[0]
		return path.VerbMove
	}
	var cubic dCubic
	cubic.set(a)
	var reducer reduceOrder
	order := reducer.reduceCubic(cubic, allowQuadratics)
	if order == 2 || order == 3 { // cubic became line or quad
		for i := 0; i < order; i++ {
			reducePts[i] = reducer.cubic.pts[i].asPoint()
		}
	}
	return pointsToVerb(order - 1)
}

// reduceCubicToVerb reduces a float cubic and returns only the verb, discarding any reduced points.
func reduceCubicToVerb(a [4]geom.Point) path.Verb {
	var scratch [3]geom.Point
	return reduceOrderCubic(a, scratch[:])
}
