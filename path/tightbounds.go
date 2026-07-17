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

// computeQuadExtremas fills extremas with the quad's axis extrema points plus its end point, returning the count.
func computeQuadExtremas(pts []geom.Point, extremas *[5]geom.Point) int {
	var ts [2]float32
	n := 0
	if t, ok := geom.FindQuadExtrema(pts[0].X, pts[1].X, pts[2].X); ok {
		ts[n] = t
		n++
	}
	if t, ok := geom.FindQuadExtrema(pts[0].Y, pts[1].Y, pts[2].Y); ok {
		ts[n] = t
		n++
	}
	for i := 0; i < n; i++ {
		extremas[i] = geom.EvalQuadAt(pts, ts[i])
	}
	extremas[n] = pts[2]
	return n + 1
}

// computeConicExtremas fills extremas with the conic's axis extrema points plus its end point, returning the count.
func computeConicExtremas(pts []geom.Point, w float32, extremas *[5]geom.Point) int {
	conic := geom.MakeConic(pts[0], pts[1], pts[2], w)
	var ts [2]float32
	n := 0
	if t, ok := conic.FindXExtrema(); ok {
		ts[n] = t
		n++
	}
	if t, ok := conic.FindYExtrema(); ok {
		ts[n] = t
		n++
	}
	for i := 0; i < n; i++ {
		extremas[i] = conic.EvalAt(ts[i])
	}
	extremas[n] = pts[2]
	return n + 1
}

// computeCubicExtremas fills extremas with the cubic's axis extrema points plus its end point, returning the count.
func computeCubicExtremas(pts []geom.Point, extremas *[5]geom.Point) int {
	var ts [4]float32
	var tx, ty [2]float32
	n := geom.FindCubicExtrema(pts[0].X, pts[1].X, pts[2].X, pts[3].X, &tx)
	copy(ts[:], tx[:n])
	ny := geom.FindCubicExtrema(pts[0].Y, pts[1].Y, pts[2].Y, pts[3].Y, &ty)
	copy(ts[n:], ty[:ny])
	n += ny
	for i := 0; i < n; i++ {
		extremas[i] = geom.EvalCubicPosAt(pts, ts[i])
	}
	extremas[n] = pts[3]
	return n + 1
}

// ComputeTightBounds returns the bounds of the lines and curves themselves (using curve extrema rather than control
// points). Unlike Bounds, the result is not cached.
func (p *Path) ComputeTightBounds() geom.Rect {
	if len(p.verbs) == 0 {
		return geom.Rect{}
	}

	if p.SegmentMasks() == SegmentLine {
		return p.Bounds()
	}

	var extremas [5]geom.Point // big enough to hold worst-case curve type (cubic) extremas + 1

	// initialize with the first MoveTo, so we don't have to check inside the switch
	first := p.Point(0)
	minX, minY := first.X, first.Y
	maxX, maxY := first.X, first.Y
	iter := newRawIter(p)
	for {
		verb, pts, w, ok := iter.next()
		if !ok {
			break
		}
		count := 0
		switch verb {
		case VerbMove:
			extremas[0] = pts[0]
			count = 1
		case VerbLine:
			extremas[0] = pts[1]
			count = 1
		case VerbQuad:
			count = computeQuadExtremas(pts[:3], &extremas)
		case VerbConic:
			count = computeConicExtremas(pts[:3], w, &extremas)
		case VerbCubic:
			count = computeCubicExtremas(pts[:4], &extremas)
		case VerbClose:
		}
		for i := 0; i < count; i++ {
			if extremas[i].X < minX {
				minX = extremas[i].X
			}
			if extremas[i].Y < minY {
				minY = extremas[i].Y
			}
			if extremas[i].X > maxX {
				maxX = extremas[i].X
			}
			if extremas[i].Y > maxY {
				maxY = extremas[i].Y
			}
		}
	}
	return geom.RectLTRB(minX, minY, maxX, maxY)
}
