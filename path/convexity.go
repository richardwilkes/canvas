// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import (
	"github.com/richardwilkes/canvas/geom"
)

// dirChange classifies the turn between two consecutive edge vectors during convexity analysis.
type dirChange uint8

const (
	dirChangeUnknown dirChange = iota
	dirChangeLeft
	dirChangeRight
	dirChangeStraight
	dirChangeBackwards // if double back, allow simple lines to be convex
	dirChangeInvalid
)

// firstDirection is the winding direction (CW/CCW) established by the first turn found while analyzing a contour.
type firstDirection uint8

const (
	firstDirectionCW firstDirection = iota
	firstDirectionCCW
	firstDirectionUnknown
)

// firstDirectionToConvexity converts a resolved winding direction into the corresponding convex Convexity value.
func firstDirectionToConvexity(dir firstDirection) Convexity {
	switch dir {
	case firstDirectionCW:
		return ConvexityConvexCW
	case firstDirectionCCW:
		return ConvexityConvexCCW
	default:
		return ConvexityConvexDegenerate
	}
}

// convexicator performs incremental convexity analysis of a single contour, one point at a time.
type convexicator struct {
	reversals int
	firstPt   geom.Point // The first point of the contour, e.g. moveTo(x,y)
	firstVec  geom.Point // The direction leaving firstPt to the next vertex
	lastPt    geom.Point // The last point passed to addPt()
	lastVec   geom.Point // The direction that brought the path to lastPt
	expected  dirChange
	firstDir  firstDirection
	finite    bool
}

func newConvexicator() convexicator {
	return convexicator{expected: dirChangeInvalid, firstDir: firstDirectionUnknown, finite: true}
}

func (c *convexicator) setMovePt(pt geom.Point) {
	c.firstPt = pt
	c.lastPt = pt
	c.expected = dirChangeInvalid
}

func (c *convexicator) addPt(pt geom.Point) bool {
	if c.lastPt == pt {
		return true
	}
	// should only be true for first non-zero vector after setMovePt was called. It is possible we doubled backed at the
	// start so need to check if lastVec is zero or not.
	if c.firstPt == c.lastPt && c.expected == dirChangeInvalid && c.lastVec == (geom.Point{}) {
		c.lastVec = pt.Sub(c.lastPt)
		c.firstVec = c.lastVec
	} else if !c.addVec(pt.Sub(c.lastPt)) {
		return false
	}
	c.lastPt = pt
	return true
}

func (c *convexicator) close() bool {
	// If this was an explicit close, there was already a lineTo to firstPt, so this addPt() is a no-op. Otherwise, the
	// addPt implicitly closes the contour. In either case, we have to check the direction change along the first vector
	// in case it is concave.
	return c.addPt(c.firstPt) && c.addVec(c.firstVec)
}

func (c *convexicator) directionChange(curVec geom.Point) dirChange {
	cross := c.lastVec.Cross(curVec)
	if !geom.IsFinite(cross) {
		return dirChangeUnknown
	}
	if cross == 0 {
		if c.lastVec.Dot(curVec) < 0 {
			return dirChangeBackwards
		}
		return dirChangeStraight
	}
	if signAsInt(cross) == 1 {
		return dirChangeRight
	}
	return dirChangeLeft
}

func (c *convexicator) addVec(curVec geom.Point) bool {
	dir := c.directionChange(curVec)
	switch dir {
	case dirChangeLeft, dirChangeRight:
		if c.expected == dirChangeInvalid {
			c.expected = dir
			if dir == dirChangeRight {
				c.firstDir = firstDirectionCW
			} else {
				c.firstDir = firstDirectionCCW
			}
		} else if dir != c.expected {
			c.firstDir = firstDirectionUnknown
			return false
		}
		c.lastVec = curVec
	case dirChangeStraight:
	case dirChangeBackwards:
		// allow path to reverse direction twice
		//   Given path.moveTo(0, 0); path.lineTo(1, 1);
		//   - 1st reversal: direction change formed by line (0,0 1,1), line (1,1 0,0)
		//   - 2nd reversal: direction change formed by line (1,1 0,0), line (0,0 1,1)
		c.lastVec = curVec
		c.reversals++
		return c.reversals < 3
	case dirChangeUnknown:
		c.finite = false
		return false
	default:
	}
	return true
}

// isConcaveBySign is a quick concavity test that counts sign changes of the vector components; more than three changes
// in either axis means concave. Returning false means "may be convex, don't know yet".
func isConcaveBySign(points []geom.Point) bool {
	count := len(points)
	if count <= 3 {
		// point, line, or triangle are always convex
		return false
	}

	const valueNeverReturnedBySign = 2
	sign := func(x float32) int {
		if x < 0 {
			return 1
		}
		return 0
	}

	idx := 0
	currPt := points[idx]
	idx++
	firstPt := currPt
	var wrap [1]geom.Point
	dxes := 0
	dyes := 0
	lastSx := valueNeverReturnedBySign
	lastSy := valueNeverReturnedBySign
	for outerLoop := 0; outerLoop < 2; outerLoop++ {
		for idx < count {
			vec := points[idx].Sub(currPt)
			if !vec.IsZero() {
				// give up if vector construction failed
				if !vec.IsFinite() {
					return true // treat as concave
				}
				sx := sign(vec.X)
				sy := sign(vec.Y)
				dxes += b2i(sx != lastSx)
				dyes += b2i(sy != lastSy)
				if dxes > 3 || dyes > 3 {
					return true
				}
				lastSx = sx
				lastSy = sy
			}
			currPt = points[idx]
			idx++
			if outerLoop > 0 {
				break
			}
		}
		// wrap around to the first point for one more vector. Backed by a stack array rather than a slice literal: this
		// runs on every convexity computation (every non-AA path fill), and the literal was a per-call heap allocation.
		wrap[0] = firstPt
		points = wrap[:]
		count = 1
		idx = 0
	}
	return false // that is, it may be convex, don't know yet
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// computeConvexitySpans computes the convexity of a path given as raw point/verb/weight spans. Callers must supply
// finite points.
func computeConvexitySpans(points []geom.Point, verbs []Verb, conicWeights []float32) Convexity {
	// Trim trailing moveTo verbs, which contribute nothing to the shape.
	vbCount := len(verbs)
	for vbCount > 0 && verbs[vbCount-1] == VerbMove {
		vbCount--
	}
	if delta := len(verbs) - vbCount; delta > 0 {
		points = points[:len(points)-delta]
		verbs = verbs[:vbCount]
	}

	if len(verbs) == 0 {
		return ConvexityConvexDegenerate
	}

	// Check to see if path changes direction more than three times as quick concave test
	if isConcaveBySign(points) {
		return ConvexityConcave
	}

	contourCount := 0
	needsClose := false
	state := newConvexicator()

	it := rawIter{p: &Path{points: points, verbs: verbs, conicWeights: conicWeights}}
	for {
		verb, pts, _, ok := it.next()
		if !ok {
			break
		}

		// Looking for the last moveTo before non-move verbs start
		if contourCount == 0 {
			if verb == VerbMove {
				state.setMovePt(pts[0])
			} else {
				// Starting the actual contour, fall through to c=1 to add the points
				contourCount++
				needsClose = true
			}
		}
		// Accumulating points into the convexicator until we hit a close or another move
		switch {
		case contourCount == 1:
			if verb == VerbClose || verb == VerbMove {
				if !state.close() {
					return ConvexityConcave
				}
				needsClose = false
				contourCount++
			} else {
				// lines add 1 point, cubics add 3, conics and quads add 2
				count := verb.pointsForVerb()
				for i := 1; i <= count; i++ {
					if !state.addPt(pts[i]) {
						return ConvexityConcave
					}
				}
			}
		case contourCount > 1:
			// The first contour has closed and anything other than spurious trailing moves means there's multiple
			// contours and the path can't be convex
			if verb != VerbMove {
				return ConvexityConcave
			}
		}
	}

	// If the path isn't explicitly closed do so implicitly
	if needsClose && !state.close() {
		return ConvexityConcave
	}

	if state.firstDir == firstDirectionUnknown && state.reversals >= 3 {
		return ConvexityConcave
	}
	return firstDirectionToConvexity(state.firstDir)
}

// GetConvexity returns the path's cached convexity, computing (and caching) it when unknown.
func (p *Path) GetConvexity() Convexity {
	if p.convexity != ConvexityUnknown {
		return p.convexity
	}
	convexity := ConvexityConcave
	if p.IsFinite() {
		convexity = computeConvexitySpans(p.points, p.verbs, p.conicWeights)
	}
	p.convexity = convexity
	return convexity
}

// IsConvex reports whether the path is convex.
func (p *Path) IsConvex() bool {
	return p.GetConvexity().IsConvex()
}

// SetConvexity overrides the cached convexity. Callers must only set a value consistent with the geometry (used where
// construction proves convexity, e.g. CreateDrawArcPath); the raster convex fast paths trust it.
func (p *Path) SetConvexity(c Convexity) {
	p.convexity = c
}
