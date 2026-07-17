// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// calculatePathGap computes the horizontal gap a glyph's outline carves out of a horizontal band — the geometry behind
// underline carve-outs when drawing text-blob intercepts.

package font

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// calculatePathGap returns the left/right extent of p's crossings with the two band edges plus any on/off-curve points
// strictly inside the band. Returns (ScalarMax, ScalarMin) when nothing intersects.
func calculatePathGap(topOffset, bottomOffset float32, p *path.Path) (left, right float32) {
	left = scalarMax
	right = scalarMin

	expandGap := func(v float32) {
		left = min(left, v)
		right = max(right, v)
	}

	// Handle the line verb: intersect the segment with the horizontal line at offset.
	addLine := func(pts *[4]geom.Point, offset float32) {
		denom := pts[1].Y - pts[0].Y
		t := ieeeFloatDivide(offset-pts[0].Y, denom)
		if 0 <= t && t < 1 { // this handles divide by zero above
			expandGap(pts[0].X + t*(pts[1].X-pts[0].X))
		}
	}

	var scratch [3]float32
	addQuad := func(pts *[4]geom.Point, offset float32) {
		for _, x := range geom.BezierQuadIntersectHorizontal(pts[0], pts[1], pts[2], offset, scratch[:0]) {
			expandGap(x)
		}
	}
	addCubic := func(pts *[4]geom.Point, offset float32) {
		for _, x := range geom.BezierCubicIntersectHorizontal(pts[0], pts[1], pts[2], pts[3], offset, scratch[:0]) {
			expandGap(x)
		}
	}

	// Handle when a verb's points are in the gap between top and bottom.
	addPts := func(pts []geom.Point) {
		for _, pt := range pts {
			if topOffset < pt.Y && pt.Y < bottomOffset {
				expandGap(pt.X)
			}
		}
	}

	iter := path.NewIter(p, false)
	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			// skip
		case path.VerbLine:
			lineTop := min(pts[0].Y, pts[1].Y)
			lineBottom := max(pts[0].Y, pts[1].Y)
			// The y-coordinates of the points intersect the top and bottom offsets.
			if topOffset <= lineBottom && lineTop <= bottomOffset {
				addLine(&pts, topOffset)
				addLine(&pts, bottomOffset)
				addPts(pts[:2])
			}
		case path.VerbQuad:
			quadTop := min(pts[0].Y, pts[1].Y, pts[2].Y)
			quadBottom := max(pts[0].Y, pts[1].Y, pts[2].Y)
			if topOffset <= quadBottom && quadTop <= bottomOffset {
				addQuad(&pts, topOffset)
				addQuad(&pts, bottomOffset)
				addPts(pts[:3])
			}
		case path.VerbConic:
			// There should be no conic primitives in glyph outlines: outlines come from sfnt quad/cubic segments only.
		case path.VerbCubic:
			cubicTop := min(pts[0].Y, pts[1].Y, pts[2].Y, pts[3].Y)
			cubicBottom := max(pts[0].Y, pts[1].Y, pts[2].Y, pts[3].Y)
			if topOffset <= cubicBottom && cubicTop <= bottomOffset {
				addCubic(&pts, topOffset)
				addCubic(&pts, bottomOffset)
				addPts(pts[:4])
			}
		case path.VerbClose:
			// skip
		}
	}
	return left, right
}

// ieeeFloatDivide is an ordinary float32 divide with IEEE inf/nan semantics.
func ieeeFloatDivide(a, b float32) float32 { return a / b }
