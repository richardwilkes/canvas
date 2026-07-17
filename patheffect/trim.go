// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The trim path effect: keeps (or removes, when inverted) the [startT, stopT] fraction of the path's total arc length.

package patheffect

import (
	"github.com/richardwilkes/canvas/contour"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// TrimMode selects whether the trimmed range or its complement is kept.
type TrimMode uint8

// TrimMode values.
const (
	TrimNormal   TrimMode = iota // return the subset path [start, stop]
	TrimInverted                 // return the complement/subset paths [0, start] + [stop, 1]
)

// addSegments appends the arc-length range [start, stop] of src to dst, returning the number of contours iterated to
// satisfy the request.
func addSegments(src *path.Path, start, stop float32, dst *path.Path, requiresMoveto bool) int {
	measure := contour.NewPathMeasure(src, false, 1)

	currentSegmentOffset := float32(0)
	contourCount := 1

	for {
		nextOffset := currentSegmentOffset + measure.Length()

		if start < nextOffset {
			measure.GetSegment(start-currentSegmentOffset, stop-currentSegmentOffset, dst, requiresMoveto)
			if stop <= nextOffset {
				break
			}
		}

		contourCount++
		currentSegmentOffset = nextOffset
		if !measure.NextContour() {
			break
		}
	}

	return contourCount
}

// trimEffect is the trim path effect implementation.
type trimEffect struct {
	noAsPoints
	startT float32
	stopT  float32
	mode   TrimMode
}

// MakeTrim creates a trim path effect. Returns nil for non-finite inputs, for a full-coverage normal trim (nothing to
// do), and for an empty inverted trim.
func MakeTrim(startT, stopT float32, mode TrimMode) stroke.PathEffect {
	if !scalarIsFinite(startT) || !scalarIsFinite(stopT) {
		return nil
	}

	if startT <= 0 && stopT >= 1 && mode == TrimNormal {
		return nil
	}

	startT = min(max(startT, 0), 1)
	stopT = min(max(stopT, 0), 1)

	if startT >= stopT && mode == TrimInverted {
		return nil
	}

	return &trimEffect{startT: startT, stopT: stopT, mode: mode}
}

func (t *trimEffect) FilterPath(dst, src *path.Path, _ *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	if t.startT >= t.stopT {
		return true
	}

	// First pass: compute the total len.
	length := float32(0)
	meas := contour.NewPathMeasure(src, false, 1)
	for {
		length += meas.Length()
		if !meas.NextContour() {
			break
		}
	}

	arcStart := length * t.startT
	arcStop := length * t.stopT

	// Second pass: actually add segments.
	if t.mode == TrimNormal {
		// Normal mode -> one span.
		if arcStart < arcStop {
			addSegments(src, arcStart, arcStop, dst, true)
		}
	} else {
		// Inverted mode -> one logical span which wraps around at the end -> two actual spans. In order to preserve
		// closed path continuity:
		//
		//   1) add the second/tail span first
		//
		//   2) skip the head span move-to for single-closed-contour paths
		requiresMoveto := true
		if arcStop < length {
			// since we're adding the "tail" first, this is the total number of contours
			contourCount := addSegments(src, arcStop, length, dst, true)

			// if the path consists of a single closed contour, we don't want to disconnect the two parts with a moveto.
			if contourCount == 1 && src.IsLastContourClosed() {
				requiresMoveto = false
			}
		}
		if arcStart > 0 {
			addSegments(src, 0, arcStart, dst, requiresMoveto)
		}
	}

	return true
}

func (t *trimEffect) ComputeFastBounds(*geom.Rect) bool {
	// Trimming a path returns a subset of the input path so just return true and leave bounds unmodified
	return true
}
