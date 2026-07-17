// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A cursor that walks over a contour's segments, classifying each as a general curve or an axis-aligned line and
// exposing the bounds/point accessors the intersection dispatch needs.

package pathops

import "github.com/richardwilkes/canvas/path"

// segmentType classifies a segment for the intersection dispatch. The line/quad/conic/cubic values equal the matching
// path.Verb so a segment's verb can be used directly wherever a segmentType is expected; horizontal/vertical are two
// synthetic subclasses of line used to route axis-aligned segments to their specialized (and cheaper) intersection
// lanes.
type segmentType int

const (
	horizontalLineSegment segmentType = -1
	verticalLineSegment   segmentType = 0
	lineSegment           segmentType = 1
	quadSegment           segmentType = 2
	conicSegment          segmentType = 3
	cubicSegment          segmentType = 4
)

// intersectionHelper is a cursor over one contour's segments, used by the intersection dispatch to walk and compare
// segments from two (possibly different) contours.
type intersectionHelper struct {
	segment *opSegment
}

// init positions the cursor at the contour's first segment.
func (h *intersectionHelper) init(contour *opContour) { h.segment = contour.first() }

// advance steps to the next segment; returns false at the end of the contour.
func (h *intersectionHelper) advance() bool {
	h.segment = h.segment.next
	return h.segment != nil
}

// startAfter positions the cursor just past another cursor's segment, used to avoid intersecting a contour's segment
// with itself and everything before it.
func (h *intersectionHelper) startAfter(after *intersectionHelper) bool {
	h.segment = after.segment.next
	return h.segment != nil
}

// bounds returns the current segment's bounding box.
func (h *intersectionHelper) bounds() pathOpsBounds { return h.segment.bounds }

func (h *intersectionHelper) left() float64   { return float64(h.segment.bounds.left) }
func (h *intersectionHelper) top() float64    { return float64(h.segment.bounds.top) }
func (h *intersectionHelper) right() float64  { return float64(h.segment.bounds.right) }
func (h *intersectionHelper) bottom() float64 { return float64(h.segment.bounds.bottom) }
func (h *intersectionHelper) x() float64      { return float64(h.segment.bounds.left) }
func (h *intersectionHelper) y() float64      { return float64(h.segment.bounds.top) }

// weight returns the current segment's conic weight (meaningless for non-conic segments).
func (h *intersectionHelper) weight() float32 { return h.segment.weight }

// xFlipped reports whether the bounds' left edge is the curve's second, not first, x coordinate.
func (h *intersectionHelper) xFlipped() bool { return h.segment.bounds.left != h.segment.pts[0].X }

// yFlipped reports whether the bounds' top edge is the curve's second, not first, y coordinate.
func (h *intersectionHelper) yFlipped() bool { return h.segment.bounds.top != h.segment.pts[0].Y }

// segmentType classifies the current segment (see the segmentType constants above).
func (h *intersectionHelper) segmentType() segmentType {
	switch h.segment.verb {
	case path.VerbQuad:
		return quadSegment
	case path.VerbConic:
		return conicSegment
	case path.VerbCubic:
		return cubicSegment
	default: // line
		if h.segment.isHorizontal() {
			return horizontalLineSegment
		}
		if h.segment.isVertical() {
			return verticalLineSegment
		}
		return lineSegment
	}
}
