// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The operation-time curve holder that opAngle carries: dCurve stores a shared four-point array (plus the conic weight)
// that can be viewed as a line, quad, conic, or cubic on demand — the typed views (line/quad/conic/cubic) are rebuilt
// from the same underlying points rather than each holding its own copy. dCurveSweep pairs a curve with the two sweep
// vectors describing its convex hull direction. Pathops computes in double precision, so the points are float64; the
// conic weight is float32.

package pathops

import (
	"math"

	"github.com/richardwilkes/canvas/path"
)

// dCurve holds the four curve types over shared four-point storage, indexed by curve.pts[n]; the typed views
// (line/quad/conic/cubic) reinterpret that storage as the corresponding curve type.
type dCurve struct {
	pts    [4]dPoint // shared line/quad/conic/cubic point storage
	weight float32   // the conic weight, unused by the other curve types
}

// line views the curve's storage as a two-point line segment.
func (c dCurve) line() dLine { return dLine{pts: [2]dPoint{c.pts[0], c.pts[1]}} }

// quad views the curve's storage as a three-point quadratic Bezier.
func (c dCurve) quad() dQuad { return dQuad{pts: [3]dPoint{c.pts[0], c.pts[1], c.pts[2]}} }

// conic views the curve's storage as a three-point conic with its weight.
func (c dCurve) conic() dConic {
	return dConic{pts: dQuad{pts: [3]dPoint{c.pts[0], c.pts[1], c.pts[2]}}, weight: c.weight}
}

// cubic views the curve's storage as a four-point cubic Bezier.
func (c dCurve) cubic() dCubic {
	return dCubic{pts: [4]dPoint{c.pts[0], c.pts[1], c.pts[2], c.pts[3]}}
}

// dCurveSweep is a curve plus the two sweep vectors describing its convex hull direction, with the is-curve/is-ordered
// flags opAngle's sort machinery consumes.
type dCurveSweep struct {
	curve   dCurve
	sweep   [2]dVector
	curveIs bool // whether the curve bends (is not effectively a line)
	ordered bool // cleared when a cubic's control point isn't between the sweep vectors
}

// isCurve reports whether the curve's hull actually bends (isn't effectively a straight line).
func (s *dCurveSweep) isCurve() bool { return s.curveIs }

// isOrdered reports whether the sweep vectors are in a consistent rotational order.
func (s *dCurveSweep) isOrdered() bool { return s.ordered }

// setCurveHullSweep computes the sweep vectors from the curve's control hull, folding degenerate leading control points
// and detecting whether the hull actually bends.
func (s *dCurveSweep) setCurveHullSweep(verb path.Verb) {
	s.ordered = true
	s.sweep[0] = s.curve.pts[1].sub(s.curve.pts[0])
	if verb == path.VerbLine {
		s.sweep[1] = s.sweep[0]
		s.curveIs = false
		return
	}
	s.sweep[1] = s.curve.pts[2].sub(s.curve.pts[0])
	// OPTIMIZE: the val-is-small-compared-to-curve check appears a lot
	maxVal := 0.0
	for index := 0; index <= opVerbToPoints(verb); index++ {
		maxVal = max(maxVal, max(math.Abs(s.curve.pts[index].x), math.Abs(s.curve.pts[index].y)))
	}
	if verb != path.VerbCubic {
		if roughlyZeroWhenComparedTo(s.sweep[0].x, maxVal) &&
			roughlyZeroWhenComparedTo(s.sweep[0].y, maxVal) {
			s.sweep[0] = s.sweep[1]
		}
		s.curveIs = s.sweep[0].crossCheck(s.sweep[1]) != 0
		return
	}
	thirdSweep := s.curve.pts[3].sub(s.curve.pts[0])
	if s.sweep[0].x == 0 && s.sweep[0].y == 0 {
		s.sweep[0] = s.sweep[1]
		s.sweep[1] = thirdSweep
		if roughlyZeroWhenComparedTo(s.sweep[0].x, maxVal) &&
			roughlyZeroWhenComparedTo(s.sweep[0].y, maxVal) {
			s.sweep[0] = s.sweep[1]
			s.curve.pts[1] = s.curve.pts[3]
		}
		s.curveIs = s.sweep[0].crossCheck(s.sweep[1]) != 0
		return
	}
	s1x3 := s.sweep[0].crossCheck(thirdSweep)
	s3x2 := thirdSweep.crossCheck(s.sweep[1])
	if s1x3*s3x2 >= 0 { // if third vector is on or between first two vectors
		s.curveIs = s.sweep[0].crossCheck(s.sweep[1]) != 0
		return
	}
	s2x1 := s.sweep[1].crossCheck(s.sweep[0])
	// Known limitation: this reordering assumes the hull's sweep is at most 180 degrees. Past that the cross products
	// stop distinguishing "third vector is between the first two" from its reflection, so the branch below can pick the
	// wrong sweep pair. Nothing here enforces the bound; it is a documented gap in the test, not a pending edit.
	if s3x2*s2x1 < 0 {
		s.sweep[0] = s.sweep[1]
		s.ordered = false
	}
	s.sweep[1] = thirdSweep
	s.curveIs = s.sweep[0].crossCheck(s.sweep[1]) != 0
}
