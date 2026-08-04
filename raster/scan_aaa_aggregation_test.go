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

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// aaFillInto fills p with AA into an existing pixmap, so a caller can lay several fills onto one target.
func aaFillInto(dev *Pixmap, p *path.Path, w, h int32) {
	AntiFillPath(p, geom.IRectLTRB(0, 0, w, h), NewSolidBlitter(dev, colorcore.Color(0xFF000000)))
}

// TestAAFillDisjointContourAggregation pins that appending disjoint contours into one path and filling it once totals
// the same coverage as filling each of them on its own. Disjoint geometry cannot interact in a scan converter that
// resolves exact per-pixel area: whichever contour a pixel's coverage comes from, the area is the same, and the
// nonzero winding rule sums windings that never overlap. Both arms are measured against the analytic area the contours
// enclose (pathEnclosedArea, computed in float64 without touching the rasterizer), so the test says which arm is wrong
// rather than only that the two disagree.
//
// The contours are laid out at 4 px pitch with staggered sub-pixel offsets in both axes — glyphs on a line of text,
// which is where this shows up in practice.
func TestAAFillDisjointContourAggregation(t *testing.T) {
	const h int32 = 16
	for _, n := range []int32{1, 2, 4, 8, 16, 32} {
		w := n*4 + 8
		merged := NewPixmap(w, h)
		separate := NewPixmap(w, h)
		all := path.New()
		var exact float64
		for i := range n {
			c := tinyGlyphContour(2+float32(i)*4+float32(i%3)/4, 4+float32(i%5)/5)
			exact += pathEnclosedArea(c, 1<<14) * 255
			all.AddPath(c, path.AddPathAppend)
			aaFillInto(separate, c, w, h)
		}
		aaFillInto(merged, all, w, h)
		mergedPct := 100 * (float64(sumAlpha(merged, w, h)) - exact) / exact
		separatePct := 100 * (float64(sumAlpha(separate, w, h)) - exact) / exact
		// v0.2.1 held both arms inside 2.5% at every count; 5% leaves room for the converter's own edge rounding
		// without admitting a real coverage loss.
		if mergedPct < -5 || mergedPct > 5 {
			t.Errorf("%d contours merged into one path total %+.2f%% of the enclosed area (filled separately: %+.2f%%)",
				n, mergedPct, separatePct)
		}
		if separatePct < -5 || separatePct > 5 {
			t.Errorf("%d contours filled separately total %+.2f%% of the enclosed area", n, separatePct)
		}
	}
}

// TestAAFillMergedMatchesSeparateInk is the same invariant measured against the other arm rather than against the
// enclosed area, and swept over sub-pixel offsets in both axes so no single lucky alignment can carry it. It is the
// form a text renderer actually feels: a run drawn per-glyph and the same run drawn as one appended path have to put
// down the same ink, and the divergence being *signed* here is the point — the defect this guards was a one-directional
// loss that grew with the number of contours, not the symmetric jitter of two different but equally valid roundings.
func TestAAFillMergedMatchesSeparateInk(t *testing.T) {
	const h int32 = 16
	for _, n := range []int32{2, 4, 8, 16, 32} {
		w := n*4 + 8
		var divergence, ink int64
		for _, ox := range []float32{0, 0.125, 0.3, 0.5, 0.7, 0.9} {
			for _, oy := range []float32{0, 0.4, 0.8} {
				merged := NewPixmap(w, h)
				separate := NewPixmap(w, h)
				all := path.New()
				for i := range n {
					c := tinyGlyphContour(2+float32(i)*4+float32(i%3)/4+ox, 4+float32(i%5)/5+oy)
					all.AddPath(c, path.AddPathAppend)
					aaFillInto(separate, c, w, h)
				}
				aaFillInto(merged, all, w, h)
				divergence += sumAlpha(merged, w, h) - sumAlpha(separate, w, h)
				ink += sumAlpha(separate, w, h)
			}
		}
		// The two arms round differently at every edge pixel, so they will never agree exactly; 4% is well clear of
		// the ~2% that leaves and well inside the ~6-to-10% the loss reached at these counts.
		if pct := 100 * float64(divergence) / float64(ink); pct < -4 || pct > 4 {
			t.Errorf("%d contours merged into one path lay down %+.2f%% of the ink the same contours filled separately do",
				n, pct)
		}
	}
}

// TestPartialAlphaAccumulatesWithoutDrift pins the property the analytic walker leans on whenever it cuts a pixel row
// into sub-scanlines. The walker stops at every distinct edge y, accumulating the row's coverage one sub-row at a time
// through getPartialAlphaMul, so whatever bias that one call carries is multiplied by however many sub-rows the row was
// cut into — and that count is set by the surrounding geometry, not by the shape being measured. Appending more
// contours into one path is exactly what drives it up: their edge y values interleave and cut every row finer without
// changing what any one contour covers.
//
// Each iteration models a row cut into subRows equal pieces, down to the 1/(1<<analyticSnapAccuracy) the edge grid
// allows, and totals what a pixel of constant coverage accumulates over them. Truncating the scale — Skia's
// `(alpha * fullAlpha) >> 8`, which it can afford because its edges snap to a quarter scanline and cap a row at four
// pieces — drifts -(subRows-1)/2 levels: -1.5 at four sub-rows, -31.5 at sixty-four, out of the 255 a full pixel holds.
func TestPartialAlphaAccumulatesWithoutDrift(t *testing.T) {
	for shift := 0; shift <= analyticSnapAccuracy; shift++ {
		subRows := 1 << shift
		fullAlpha := fixedToAlpha(FixedOne >> shift)
		var drift int
		for alpha := range 256 {
			drift += subRows*int(getPartialAlphaMul(Alpha(alpha), fullAlpha)) - alpha
		}
		// Rounding to nearest leaves half a level either way whatever subRows is. Anything beyond a whole level is a
		// bias that scales with the cut count, and it reaches the page as ink a dense path loses (or invents).
		if mean := float64(drift) / 256; mean < -1 || mean > 1 {
			t.Errorf("a pixel row accumulated over %d sub-scanlines (fullAlpha %d) drifts %+.2f levels per pixel, want within +/-1",
				subRows, fullAlpha, mean)
		}
	}
}
