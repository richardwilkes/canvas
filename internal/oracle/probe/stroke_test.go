// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package probe

import (
	"fmt"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// Stroker probes.
//
// TestStrokeFillPathProbe is the structural differential: Skia's fill-path result vs stroke.FillPathWithPaintResScale
// over a stroke-focused path corpus × paint matrix, comparing the returned doFill flag, fill type, point counts (exact
// — a count mismatch means a subdivision decision diverged) and the point streams (bit-exact preferred, agree()
// tolerance for the normalize/sqrt float lanes, whose contraction behavior is compiler-specific on the C side).

// strokePaint is one stroke configuration applied to both implementations.
type strokePaint struct {
	width    float32
	miter    float32
	cap      int
	join     int
	fill     bool // stroke-and-fill
	resScale float32
}

func (sp *strokePaint) String() string {
	return fmt.Sprintf("w=%v miter=%v cap=%d join=%d fill=%v res=%v",
		sp.width, sp.miter, sp.cap, sp.join, sp.fill, sp.resScale)
}

func (sp *strokePaint) spec() stroke.PaintSpec {
	style := stroke.PaintStyleStroke
	if sp.fill {
		style = stroke.PaintStyleStrokeAndFill
	}
	miter := sp.miter
	if miter == 0 {
		miter = 4 // Skia's PaintDefaults_MiterLimit
	}
	return stroke.PaintSpec{
		Style:      style,
		Width:      sp.width,
		MiterLimit: miter,
		Cap:        stroke.Cap(sp.cap),
		Join:       stroke.Join(sp.join),
	}
}

// strokeCorpus is the stroke-focused path corpus.
func strokeCorpus() []*scenario.PathSpec {
	var out []*scenario.PathSpec
	add := func(s *scenario.PathSpec) { out = append(out, s) }
	sp := scenario.NewPathSpec

	// Lines: open segment, axis-aligned, polyline with shallow and sharp turns, closed polygon.
	add(sp().MoveTo(10, 10).LineTo(90, 70))
	add(sp().MoveTo(10, 40).LineTo(110, 40))
	add(sp().MoveTo(10, 80).LineTo(40, 20).LineTo(70, 80).LineTo(100, 20))
	add(sp().MoveTo(10, 10).LineTo(100, 12).LineTo(12, 30)) // very sharp angle (miter limit territory)
	add(sp().MoveTo(20, 20).LineTo(90, 25).LineTo(60, 80).Close())
	add(sp().AddPoly([]geom.Point{{X: 15, Y: 15}, {X: 95, Y: 20}, {X: 80, Y: 85}, {X: 25, Y: 70}}, true))

	// The closed-rect specialization (strokeRect), both directions, plus an unsorted rect.
	add(sp().AddRect(geom.RectLTRB(20, 20, 100, 70), geom.DirectionCW))
	add(sp().AddRect(geom.RectLTRB(20, 20, 100, 70), geom.DirectionCCW))
	add(sp().AddRect(geom.RectLTRB(100, 70, 20, 20), geom.DirectionCW))
	// A rect drawn with explicit verbs but left open takes the general stroker.
	add(sp().MoveTo(20, 20).LineTo(100, 20).LineTo(100, 70).LineTo(20, 70))

	// Quads: generic, near-linear (reduction lanes), control point beyond the ends (degenerate lane).
	add(sp().MoveTo(10, 70).QuadTo(60, 10, 110, 70))
	add(sp().MoveTo(10, 40).QuadTo(60, 40.001, 110, 40))
	add(sp().MoveTo(10, 40).QuadTo(150, 40, 110, 40))
	add(sp().MoveTo(10, 40).QuadTo(60, 40, 10, 40)) // ends coincide

	// Conics: circle-ish weight, hyperbolic weight, near-line.
	add(sp().MoveTo(10, 70).ConicTo(60, 10, 110, 70, 0.70710678))
	add(sp().MoveTo(10, 70).ConicTo(60, 10, 110, 70, 2.5))
	add(sp().MoveTo(10, 40).ConicTo(60, 40.0005, 110, 40, 0.5))

	// Cubics: S-curve (inflection), loop with cusp-adjacent geometry, coincident control points (degenerate
	// reductions), collinear controls outside the ends.
	add(sp().MoveTo(10, 40).CubicTo(40, 10, 70, 70, 100, 40))
	add(sp().MoveTo(20, 60).CubicTo(110, 10, 10, 10, 100, 60))
	add(sp().MoveTo(20, 20).CubicTo(20, 20, 80, 30, 90, 80))
	add(sp().MoveTo(20, 20).CubicTo(80, 30, 80, 30, 90, 80))
	add(sp().MoveTo(10, 40).CubicTo(130, 40, -20, 40, 100, 40))

	// Shapes through the general stroker: circle, oval, round rect.
	add(sp().AddCircle(60, 45, 30, geom.DirectionCW))
	add(sp().AddOval(geom.RectLTRB(15, 25, 105, 65), geom.DirectionCW))
	add(sp().AddRoundRect(geom.RectLTRB(20, 20, 100, 70), 12, 8, geom.DirectionCW))

	// Contour bookkeeping: zero-length contours (cap-dependent), multiple contours, close-only.
	add(sp().MoveTo(50, 40).Close())
	add(sp().MoveTo(50, 40).LineTo(50, 40).Close())
	add(sp().MoveTo(50, 40).LineTo(50, 40))
	add(sp().MoveTo(10, 10).LineTo(40, 30).MoveTo(60, 10).LineTo(90, 30).LineTo(75, 50).Close())

	// Inverse fill preservation.
	add(sp().MoveTo(20, 20).LineTo(90, 25).LineTo(60, 80).Close().SetFill(path.FillInverseWinding))

	return out
}

// strokePaintMatrix: the paint configurations each corpus entry is stroked with.
func strokePaintMatrix() []strokePaint {
	var out []strokePaint
	// caps × joins at a representative width.
	for capStyle := 0; capStyle <= 2; capStyle++ {
		for join := 0; join <= 2; join++ {
			out = append(out, strokePaint{width: 5, cap: capStyle, join: join, resScale: 1})
		}
	}
	// width sweep (sub-pixel through fat).
	for _, w := range []float32{0.4, 1, 11, 40} {
		out = append(out, strokePaint{width: w, cap: 1, join: 1, resScale: 1})
	}
	// miter-limit sweep on the default join.
	for _, m := range []float32{0.9, 1.5, 2, 10, 1000} {
		out = append(out, strokePaint{width: 6, miter: m, resScale: 1})
	}
	// resScale sweep (subdivision tolerance).
	for _, rs := range []float32{0.3, 4.7} {
		out = append(out, strokePaint{width: 5, cap: 1, join: 1, resScale: rs})
	}
	// stroke-and-fill.
	out = append(out, strokePaint{width: 5, resScale: 1, fill: true},
		strokePaint{width: 5, cap: 1, join: 1, resScale: 1, fill: true})
	return out
}

func TestStrokeFillPathProbe(t *testing.T) {
	corpus := strokeCorpus()
	paints := strokePaintMatrix()

	exact := 0
	toleranced := 0
	for si := range corpus {
		gp := corpus[si].BuildGo()
		for pi := range paints {
			pp := &paints[pi]

			want := refStrokeFillPath(corpus[si], pp)

			gSpec := pp.spec()
			gDst := &path.Path{}
			gIsFill := stroke.FillPathWithPaintResScale(gp, &gSpec, gDst, nil, pp.resScale)

			what := fmt.Sprintf("corpus[%d] paint[%d] (%s)", si, pi, pp.String())

			if gIsFill != want.isFill {
				t.Errorf("%s: isFill go %v c %v", what, gIsFill, want.isFill)
			}
			if got := int(gDst.FillType()); got != want.fillType {
				t.Errorf("%s: fill type go %d c %d", what, got, want.fillType)
			}
			gN, cN := gDst.CountPoints(), len(want.points)
			if gN != cN {
				t.Errorf("%s: point count go %d c %d", what, gN, cN)
				continue
			}
			gPts := make([]geom.Point, gN)
			gDst.Points(gPts)
			cPts := want.points
			magX, magY := coordMag(cPts)
			// The stroker's normalize/setLength run in double (exact), but join/subdivision decisions and perpendicular
			// offsets mix float mul-adds that the C compiler may contract; allow the standard agree() band scaled by
			// the radius-ish magnitude.
			allExact := true
			for i := range gPts {
				if f32Eq(gPts[i].X, cPts[i].X) && f32Eq(gPts[i].Y, cPts[i].Y) {
					continue
				}
				allExact = false
				if !agree(gPts[i].X, cPts[i].X, magX) || !agree(gPts[i].Y, cPts[i].Y, magY) {
					t.Errorf("%s: point %d go %v c %v", what, i, gPts[i], cPts[i])
					break
				}
			}
			if allExact {
				exact++
			} else {
				toleranced++
			}
		}
	}
	fmt.Printf("stroke fill-path probe done: %d comparisons bit-exact, %d under tolerance\n", exact, toleranced)
}
