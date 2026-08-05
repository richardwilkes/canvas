// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/path"
)

// convergingCubicWedge is the smallest shape distilled from a real page (the 'C' of an embedded NewAster-Black glyph
// at 285 px) that made the analytic walker lose a third of the shape's coverage. Every ingredient is required. The
// cubic starts with a horizontal tangent, so its first forward-differenced segment spans much more x than y — and once
// that segment's endpoints snap (its start to the fine endpoint grid, its end to the coarse interior curve grid), the
// stored slope is amplified to several times the true one. The long left edges converge toward it, arming the walker's
// intersection probe on the same strips. dy shifts the whole shape off the coarse grid, putting the segment's first
// boundary between quarter-scanline probe stops.
func convergingCubicWedge(dy float32) *path.Path {
	p := path.New()
	p.MoveTo(197.3080, 54.2695+dy)
	p.LineTo(120.7940, 28.2890+dy)
	p.LineTo(125.9330, 43.7060+dy)
	p.CubicTo(144.2050, 43.7060+dy, 159.0510, 53.4130+dy, 163.0480, 95.9525+dy)
	p.Close()
	return p
}

// TestAAFillIntersectionProbeKeepsCurveBoundary pins the interaction that let the intersection probe skip a curve
// segment boundary. The probe (checkIntersection/checkIntersectionFwd) answers "how far may this strip run before
// edge order must be re-checked" with nextY + a quarter scanline; a curve segment whose boundary sits on the finer
// endpoint grid can end within that window, and overwriting the already-recorded boundary instead of keeping the
// earlier of the two made the walker advance the expired segment's line past its end — at the amplified slope, tens
// of pixels of x error, which keepContinuous then carried through every remaining segment of the curve. Filled
// coverage is the observable: with the boundary skipped this wedge lost ~37% of its area; with it kept the converter
// is within its normal sub-percent coverage accuracy. The sub-pixel sweep keeps the shape crossing grid alignments so
// the test cannot go quiet if the snap grids change.
func TestAAFillIntersectionProbeKeepsCurveBoundary(t *testing.T) {
	const w, h int32 = 220, 120
	const offsets = 8
	for i := range offsets {
		dy := float32(i) / offsets
		shape := convergingCubicWedge(dy)
		exact := pathEnclosedArea(shape, 1<<14) * 255
		got := float64(sumAlpha(aaFill(t, shape, w, h), w, h))
		if pct := 100 * (got - exact) / exact; pct < -2 || pct > 2 {
			t.Errorf("dy=%.3f: coverage is %+.2f%% of the enclosed area, want within +/-2%%", dy, pct)
		}
	}
}
