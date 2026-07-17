// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The corner path effect: rounds sharp corners between line segments with quads of the given radius.

package patheffect

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// computeStep returns the offset from a toward b by radius (or half the distance if the segment is too short to fit two
// radii), and whether a straight segment remains to be drawn between the rounded ends.
func computeStep(a, b geom.Point, radius float32) (geom.Point, bool) {
	dist := a.DistanceTo(b)
	step := b.Sub(a)
	if dist <= radius*2 {
		return step.Scaled(0.5), false
	}
	return step.Scaled(radius / dist), true
}

// cornerEffect is the corner path effect implementation.
type cornerEffect struct {
	noAsPoints
	radius float32
}

// MakeCorner creates a corner-rounding path effect; returns nil unless radius is finite and > 0.
func MakeCorner(radius float32) stroke.PathEffect {
	if scalarIsFinite(radius) && radius > 0 {
		return &cornerEffect{radius: radius}
	}
	return nil
}

func (c *cornerEffect) FilterPath(dst, src *path.Path, _ *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	if c.radius <= 0 {
		return false
	}

	// just need a value that won't match when we compare it initially
	prevVerb := path.Verb(0xFF)

	var closed bool
	var moveTo, lastCorner geom.Point
	var firstStep, step geom.Point
	prevIsValid := true

	iter := path.NewIter(src, false)
	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			// close out the previous (open) contour
			if prevVerb == path.VerbLine {
				dst.LineToPt(lastCorner)
			}
			closed = iter.IsClosedContour()
			if closed {
				moveTo = pts[0]
				prevIsValid = false
			} else {
				dst.MoveToPt(pts[0])
				prevIsValid = true
			}
		case path.VerbLine:
			var drawSegment bool
			step, drawSegment = computeStep(pts[0], pts[1], c.radius)
			// prev corner
			if !prevIsValid {
				dst.MoveToPt(moveTo.Add(step))
			} else {
				dst.QuadTo(pts[0].X, pts[0].Y, pts[0].X+step.X, pts[0].Y+step.Y)
			}
			if drawSegment {
				dst.LineTo(pts[1].X-step.X, pts[1].Y-step.Y)
			}
			lastCorner = pts[1]
			prevIsValid = true
		case path.VerbQuad:
			// TBD - just replicate the curve for now
			if !prevIsValid {
				dst.MoveToPt(pts[0])
				prevIsValid = true
			}
			dst.QuadToPt(pts[1], pts[2])
			lastCorner = pts[2]
			firstStep = geom.Point{}
		case path.VerbConic:
			// TBD - just replicate the curve for now
			if !prevIsValid {
				dst.MoveToPt(pts[0])
				prevIsValid = true
			}
			dst.ConicToPt(pts[1], pts[2], iter.ConicWeight())
			lastCorner = pts[2]
			firstStep = geom.Point{}
		case path.VerbCubic:
			if !prevIsValid {
				dst.MoveToPt(pts[0])
				prevIsValid = true
			}
			// TBD - just replicate the curve for now
			dst.CubicToPt(pts[1], pts[2], pts[3])
			lastCorner = pts[3]
			firstStep = geom.Point{}
		case path.VerbClose:
			if firstStep.X != 0 || firstStep.Y != 0 {
				dst.QuadTo(lastCorner.X, lastCorner.Y, lastCorner.X+firstStep.X, lastCorner.Y+firstStep.Y)
			}
			dst.Close()
			prevIsValid = false
		}
		if prevVerb == path.VerbMove {
			firstStep = step
		}
		prevVerb = verb
	}
	if prevIsValid {
		dst.LineToPt(lastCorner)
	}
	return true
}

func (c *cornerEffect) ComputeFastBounds(*geom.Rect) bool {
	// Rounding sharp corners within a path produces a new path that is still contained within the original's bounds, so
	// leave 'bounds' unmodified.
	return true
}
