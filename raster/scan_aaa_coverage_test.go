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

// tinyGlyphContour is one contour lifted from a 7 pt embedded TrueType glyph rendered at 72 dpi — 1.09 x 1.69 px of
// eight quadratic segments and one line, nothing degenerate, no self-intersection — translated so the contour's own
// origin lands at (dx, dy). Body text is built almost entirely of contours this size, which is why a coverage error
// that only shows up at this scale reaches a whole page of text.
func tinyGlyphContour(dx, dy float32) *path.Path {
	const ox, oy = 386.4355, 219.59393
	p := path.New()
	x := func(v float32) float32 { return v - ox + dx }
	y := func(v float32) float32 { return v - oy + dy }
	p.MoveTo(x(387.27924), y(221.15399))
	p.QuadTo(x(387.3847), y(221.07489), x(387.52972), y(220.9079))
	p.LineTo(x(387.52972), y(219.59393))
	p.QuadTo(x(387.16937), y(219.68182), x(386.8969), y(219.87079))
	p.QuadTo(x(386.4355), y(220.19159), x(386.4355), y(220.68817))
	p.QuadTo(x(386.4355), y(220.9826), x(386.56952), y(221.13422))
	p.QuadTo(x(386.70355), y(221.28583), x(386.87054), y(221.28583))
	p.QuadTo(x(387.09027), y(221.28583), x(387.27924), y(221.15399))
	p.Close()
	return p
}

// pathEnclosedArea returns the area a path encloses, computed in float64 by flattening every segment into steps line
// pieces and applying the shoelace formula. It never touches the rasterizer, so it is the same number on every
// revision — which is the point: comparing one scan converter against a supersampled run of the SAME scan converter
// cannot tell a coverage bug from a coverage bug at 16x.
func pathEnclosedArea(p *path.Path, steps int) float64 {
	var sum, cx, cy, sx, sy float64
	accum := func(nx, ny float64) {
		sum += cx*ny - nx*cy
		cx, cy = nx, ny
	}
	it := path.NewRawIter(p)
	for {
		verb, pts, _, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			cx, cy = float64(pts[0].X), float64(pts[0].Y)
			sx, sy = cx, cy
		case path.VerbLine:
			accum(float64(pts[1].X), float64(pts[1].Y))
		case path.VerbQuad:
			x0, y0 := cx, cy
			x1, y1 := float64(pts[1].X), float64(pts[1].Y)
			x2, y2 := float64(pts[2].X), float64(pts[2].Y)
			for i := 1; i <= steps; i++ {
				t := float64(i) / float64(steps)
				u := 1 - t
				accum(u*u*x0+2*t*u*x1+t*t*x2, u*u*y0+2*t*u*y1+t*t*y2)
			}
		case path.VerbCubic:
			x0, y0 := cx, cy
			x1, y1 := float64(pts[1].X), float64(pts[1].Y)
			x2, y2 := float64(pts[2].X), float64(pts[2].Y)
			x3, y3 := float64(pts[3].X), float64(pts[3].Y)
			for i := 1; i <= steps; i++ {
				t := float64(i) / float64(steps)
				u := 1 - t
				accum(u*u*u*x0+3*t*u*u*x1+3*t*t*u*x2+t*t*t*x3, u*u*u*y0+3*t*u*u*y1+3*t*t*u*y2+t*t*t*y3)
			}
		case path.VerbClose:
			accum(sx, sy)
		}
	}
	if sum < 0 {
		sum = -sum
	}
	return sum / 2
}

// sumAlpha totals the coverage in a pixmap's alpha channel.
func sumAlpha(dev *Pixmap, w, h int32) int64 {
	var total int64
	for y := range h {
		for x := range w {
			total += int64(alphaAt(dev, x, y))
		}
	}
	return total
}

// TestAAFillGlyphScaleCoverage pins the defining property of a coverage-based scan converter: the alpha a filled shape
// totals is its area times 255, whatever sub-pixel offset it sits at. Antialiasing redistributes coverage between
// neighboring pixels; it does not create or destroy it, so translating a shape may move alpha around but must not
// change the sum.
//
// The tolerances are set from what v0.2.1 achieved on this contour (mean +0.76%, worst -5.02%) with roughly 60% of
// headroom, so this is a floor the converter has already cleared rather than an aspiration.
func TestAAFillGlyphScaleCoverage(t *testing.T) {
	const w, h int32 = 16, 16
	const offsets = 16
	exact := pathEnclosedArea(tinyGlyphContour(4, 4), 1<<14) * 255
	var sum, worst float64
	for i := range offsets {
		for j := range offsets {
			p := tinyGlyphContour(4+float32(i)/offsets, 4+float32(j)/offsets)
			pct := 100 * (float64(sumAlpha(aaFill(t, p, w, h), w, h)) - exact) / exact
			sum += pct
			worst = min(worst, pct)
		}
	}
	if mean := sum / (offsets * offsets); mean < -2 || mean > 2 {
		t.Errorf("mean coverage over %d sub-pixel offsets is %+.2f%% of the enclosed area, want within +/-2%%",
			offsets*offsets, mean)
	}
	if worst < -8 {
		t.Errorf("worst coverage over %d sub-pixel offsets is %+.2f%% of the enclosed area, want no worse than -8%%",
			offsets*offsets, worst)
	}
}
