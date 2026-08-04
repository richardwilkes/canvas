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

	"github.com/richardwilkes/canvas/geom"
)

// TestQuickInverseSign pins quickInverse's documented result: +4194304/x with C-style truncating division, so the
// result carries x's sign and an already-absolute slope yields the non-negative reciprocal the DY callers want. The
// lookup table it replaces stores negative entries and cancels the sign in its indexing, which is easy to misread as a
// negation of the result.
func TestQuickInverseSign(t *testing.T) {
	if got := quickInverse(0); got != 0 {
		t.Fatalf("quickInverse(0) = %d, want 0", got)
	}
	for _, x := range []FDot6{1, 2, 3, 7, 64, 511, 512, 1023, kInverseTableSize} {
		want := Fixed(FDot6One << 16 / x)
		if want <= 0 {
			t.Fatalf("bad test setup: 4194304/%d is not positive", x)
		}
		if got := quickInverse(x); got != want {
			t.Fatalf("quickInverse(%d) = %d, want %d", x, got, want)
		}
		if got := quickInverse(-x); got != -want {
			t.Fatalf("quickInverse(%d) = %d, want %d", -x, got, -want)
		}
	}
}

// TestAnalyticUpdateLineDYUsesFDot6Slope pins the reference convention: updateLine converts the slope to FDot6 before
// the inverse lookup, which for a slope inside the table's reach yields the true reciprocal slope AnalyticEdge.DY
// documents. 0.5 of x over 8 of y has reciprocal slope 16.
func TestAnalyticUpdateLineDYUsesFDot6Slope(t *testing.T) {
	const x1 = FixedOne / 2
	const y1 = 8 * FixedOne
	slope := quickDiv(FixedToFDot6(x1), FixedToFDot6(y1))
	var e AnalyticEdge
	e.Winding = WindingCW
	if !e.updateLine(0, 0, x1, y1, slope) {
		t.Fatal("updateLine rejected a non-degenerate segment")
	}
	if want := quickInverse(FixedToFDot6(FixedAbs(slope))); e.DY != want {
		t.Fatalf("updateLine DY = %d, want %d (the FDot6-converted slope)", e.DY, want)
	}
	if want := 16 * FixedOne; e.DY != want {
		t.Fatalf("updateLine DY = %d, want %d (dy/dx for 0.5 over 8)", e.DY, want)
	}
}

// setLineQuantize reproduces SetLine's endpoint quantization (times 4, to FDot6, to Fixed, back down, y snapped) so a
// test can drive updateLine over the exact same fixed-point endpoints SetLine derived.
func setLineQuantize(p geom.Point) (x, y Fixed) {
	const accuracy = analyticAccuracy
	const multiplier = 1 << analyticAccuracy
	x = FDot6ToFixed(FloatToFDot6(p.X*multiplier)) >> accuracy
	y = snapY(FDot6ToFixed(FloatToFDot6(p.Y*multiplier)) >> accuracy)
	return x, y
}

// TestAnalyticSetLineDYMatchesUpdateLine: SetLine used to hand quickInverse the raw 16.16 slope while updateLine
// converted to FDot6 first, so a line edge's DY was 1024x smaller than a curve segment's for the same geometry — and
// 1024x off the "abs(1/DX)" contract on AnalyticEdge.DY, which feeds partialTriangleToAlpha. Both setters must now
// derive DY the same way, across the inverse-table branch, the quickDiv branch, and the vertical MaxInt32 case.
func TestAnalyticSetLineDYMatchesUpdateLine(t *testing.T) {
	for _, tc := range []struct {
		name   string
		p0, p1 geom.Point
	}{
		// 6/256 of x (exactly representable at SetLine's 1/256 px quantization) over 190 of y: near-vertical, the
		// case where the two conventions used to diverge.
		{name: "near-vertical", p0: geom.Pt(10, 10), p1: geom.Pt(10.0234375, 200)},
		{name: "shallow", p0: geom.Pt(10, 10), p1: geom.Pt(10.5, 18)},
		{name: "diagonal", p0: geom.Pt(4, 4), p1: geom.Pt(84, 84)},
		{name: "steep", p0: geom.Pt(4, 4), p1: geom.Pt(84, 12)},
		{name: "vertical", p0: geom.Pt(10, 10), p1: geom.Pt(10, 90)},
		{name: "reversed", p0: geom.Pt(60, 90), p1: geom.Pt(58.25, 10)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var line AnalyticEdge
			if !line.SetLine(tc.p0, tc.p1) {
				t.Fatal("SetLine rejected a non-degenerate edge")
			}
			// Drive updateLine over the exact same fixed-point endpoints and slope.
			x0, y0 := setLineQuantize(tc.p0)
			x1, y1 := setLineQuantize(tc.p1)
			if y0 > y1 {
				x0, x1 = x1, x0
				y0, y1 = y1, y0
			}
			slope := quickDiv(FixedToFDot6(x1-x0), FixedToFDot6(y1-y0))
			if slope != line.DX {
				t.Fatalf("bad test setup: recomputed slope %d != SetLine's DX %d", slope, line.DX)
			}
			var curve AnalyticEdge
			curve.Winding = line.Winding
			if !curve.updateLine(x0, y0, x1, y1, slope) {
				t.Fatal("updateLine rejected the same segment")
			}
			if line.DY != curve.DY {
				t.Fatalf("SetLine DY = %d but updateLine DY = %d for slope %d", line.DY, curve.DY, slope)
			}
		})
	}
}

// TestSnapYRoundsToTheGrid: snapY leaves grid lines alone and rounds everything else to the nearest one, in both
// directions and on both sides of zero (it does the arithmetic unsigned so negative y wraps instead of shifting in sign
// bits).
func TestSnapYRoundsToTheGrid(t *testing.T) {
	const step = FixedOne >> analyticSnapAccuracy
	for _, base := range []Fixed{-3 * FixedOne, -step, 0, step, 161 * FixedOne} {
		for _, off := range []Fixed{0, 1, step/2 - 1, step / 2, step/2 + 1, step - 1} {
			want := base
			if off >= step/2 {
				want = base + step
			}
			if got := snapY(base + off); got != want {
				t.Fatalf("snapY(%d + %d) = %d, want %d", base, off, got, want)
			}
		}
	}
}

// TestSnapYGridSurvivesTheEdgeHeightFloor pins the reasoning behind analyticSnapAccuracy's value. SetLine and
// updateLine drop an edge whose height rounds to zero in FDot6 (1/64 of a pixel), so the snap grid must be no finer
// than that: a finer grid could snap two path vertices onto distinct grid lines less than 1/64 apart, the resulting
// edge would then be discarded, and the contour would carry a winding gap where that edge no longer meets its
// neighbors. A shorter grid step must therefore survive SetLine intact.
func TestSnapYGridSurvivesTheEdgeHeightFloor(t *testing.T) {
	const step = FixedOne >> analyticSnapAccuracy
	if FixedToFDot6(step) == 0 {
		t.Fatalf("snap grid step %d rounds to zero in FDot6; edges one grid step tall would be dropped", step)
	}
	var e AnalyticEdge
	p0 := geom.Pt(10, 20)
	p1 := geom.Pt(11, 20+1/float32(int32(1)<<analyticSnapAccuracy))
	if !e.SetLine(p0, p1) {
		t.Fatalf("SetLine dropped an edge one grid step (%v to %v) tall", p0.Y, p1.Y)
	}
	if got := e.LowerY - e.UpperY; got != step {
		t.Fatalf("edge height %d, want one grid step (%d)", got, step)
	}
}

// snappedEndpointY reproduces the y a curve piece's endpoint ends up at once the setup paths have quantized and snapped
// it, so a test can say what "the piece has no height left" means without restating the arithmetic.
func snappedEndpointY(y float32) Fixed {
	const multiplier = 1 << analyticAccuracy
	return snapY(FDot6ToFixed(FloatToFDot6(y*multiplier)) >> analyticAccuracy)
}

// quarterPixelRound is the quarter-pixel rounding the curve setup paths used to decide a piece had no height, and the
// one Skia still uses: the endpoint taken to the analyticAccuracy-scaled FDot6 space and rounded to a whole unit there.
func quarterPixelRound(y float32) int32 {
	return FDot6Round(FDot6(y * (1 << (analyticAccuracy + 6))))
}

// TestCurvePieceDropTestFollowsTheEndpointSnapGrid pins the safety condition on dropping a zero-height curve piece: a
// piece may only be discarded when its two endpoints snap onto the same grid line, because that is exactly when the
// neighboring pieces — whose shared vertices snap the same way — close the contour back up over it. The guard used to
// round to a quarter of a scanline while analyticSnapAccuracy snapped endpoints to 1/64, so it discarded pieces the
// neighbors left a gap across, and a glyph-scale contour bled coverage through the hole (issue #3).
//
// Stated as agreement with SetLine: a straight edge between the same two endpoints is dropped on precisely the same
// condition, so "the curve piece survives" and "a line across it survives" can never disagree.
func TestCurvePieceDropTestFollowsTheEndpointSnapGrid(t *testing.T) {
	// Sub-quarter-scanline heights, straddling the old quarter-pixel rounding, from starts at assorted sub-pixel
	// offsets. 0 and one whole grid step are the boundaries of the condition itself.
	const grid = 1 / float32(int32(1)<<analyticSnapAccuracy)
	for _, y0 := range []float32{20, 20.1, 20.376, 20.5, 20.62, 20.9} {
		for _, h := range []float32{0, grid / 4, grid / 2, grid, 0.02, 0.05, 0.13, 0.24, 0.246, 0.3} {
			y1 := y0 + h
			var line AnalyticEdge
			wantKept := line.SetLine(geom.Pt(4, y0), geom.Pt(5, y1))
			if got := snappedEndpointY(y0) != snappedEndpointY(y1); got != wantKept {
				t.Fatalf("bad test setup: SetLine kept=%v for y %v..%v but snapped endpoints differ=%v",
					wantKept, y0, y1, got)
			}

			var quad AnalyticQuadraticEdge
			if got := quad.SetQuadratic([]geom.Point{
				geom.Pt(4, y0), geom.Pt(4.5, (y0+y1)/2), geom.Pt(5, y1),
			}); got != wantKept {
				t.Errorf("SetQuadratic kept=%v for y %v..%v, want %v (as SetLine does)", got, y0, y1, wantKept)
			}

			var cubic AnalyticCubicEdge
			if got := cubic.SetCubic([]geom.Point{
				geom.Pt(4, y0), geom.Pt(4.3, y0+h/3), geom.Pt(4.7, y1-h/3), geom.Pt(5, y1),
			}); got != wantKept {
				t.Errorf("SetCubic kept=%v for y %v..%v, want %v (as SetLine does)", got, y0, y1, wantKept)
			}
		}
	}
}

// checkCurvePieceSpan asserts that the segment stream a curve piece walks out covers exactly the span between its two
// snapped endpoints: it starts on the snapped start, ends on the snapped end, and hands off from one segment to the
// next without a gap. Anything less is a hole in the contour, the same defect as dropping the piece outright — and
// short pieces are where it hides, because the boundaries within a curve snap to a coarser grid
// (analyticCurveSnapAccuracy) than the piece's own endpoints do.
func checkCurvePieceSpan(t *testing.T, kind string, segs []AnalyticEdgeSegment, y0, yLast float32) {
	t.Helper()
	const upperY, lowerY = 3, 4
	if len(segs) == 0 {
		t.Errorf("%s spanning y %v..%v was dropped, but its endpoints snap to distinct grid lines", kind, y0, yLast)
		return
	}
	if got, want := Fixed(segs[0][upperY]), snappedEndpointY(y0); got != want {
		t.Errorf("%s spanning y %v..%v starts at %d, want the snapped start %d", kind, y0, yLast, got, want)
	}
	if got, want := Fixed(segs[len(segs)-1][lowerY]), snappedEndpointY(yLast); got != want {
		t.Errorf("%s spanning y %v..%v ends at %d, want the snapped end %d", kind, y0, yLast, got, want)
	}
	for i := 1; i < len(segs); i++ {
		if segs[i][upperY] != segs[i-1][lowerY] {
			t.Errorf("%s spanning y %v..%v: gap between segment %d (ends %d) and %d (starts %d)",
				kind, y0, yLast, i-1, segs[i-1][lowerY], i, segs[i][upperY])
		}
	}
}

// TestShortCurvePieceSpansItsSnappedEndpoints sweeps the heights that used to fall through the cracks — everything from
// half a snap-grid step up to past a quarter of a scanline, from starts at assorted sub-pixel offsets — and holds every
// piece to covering its own span. It is the structural half of issue #3: the quad case is a piece the drop guard threw
// away for rounding onto one quarter-pixel, and the cubic case is a piece that survived the guard only to have its last
// segment rounded back onto the interior grid, leaving it zero-height and discarded anyway.
func TestShortCurvePieceSpansItsSnappedEndpoints(t *testing.T) {
	const grid = 1 / float32(int32(1)<<analyticSnapAccuracy)
	var sawQuarterPixelCollision bool
	for _, y0 := range []float32{5.38, 20, 20.1, 20.5, 20.62, 20.9} {
		for _, h := range []float32{grid / 2, grid, 0.02, 0.05, 0.13, 0.24, 0.246, 0.3, 0.7} {
			yLast := y0 + h
			if snappedEndpointY(y0) == snappedEndpointY(yLast) {
				continue // legitimately zero-height once snapped; the neighbors close over it
			}
			if quarterPixelRound(y0) == quarterPixelRound(yLast) {
				sawQuarterPixelCollision = true
			}
			checkCurvePieceSpan(t, "quad", AnalyticQuadSegments([]geom.Point{
				geom.Pt(4.84, y0), geom.Pt(4.95, (y0+yLast)/2), geom.Pt(5.09, yLast),
			}, analyticSnapAccuracy), y0, yLast)
			checkCurvePieceSpan(t, "cubic", AnalyticCubicSegments([]geom.Point{
				geom.Pt(4.84, y0), geom.Pt(4.9, y0+h/3), geom.Pt(5.0, yLast-h/3), geom.Pt(5.09, yLast),
			}, analyticSnapAccuracy), y0, yLast)
		}
	}
	if !sawQuarterPixelCollision {
		t.Error("bad test setup: no case left where the endpoints collide on the quarter-pixel grid the old guard " +
			"rounded to, so the sweep no longer covers the regression")
	}
}

// TestCurvePieceDropTestUnchangedOnSkiasGrid: the drop test is derived from the snap grid it is handed, so on Skia's own
// grid it has to reduce to the quarter-pixel rounding Skia does. The differential probe replays Skia's frozen segment
// streams through setQuadratic/setCubic at SkiaSnapAccuracy, and a piece the port kept where Skia dropped it (or the
// reverse) would silently invalidate every one of those references rather than fail a comparison.
func TestCurvePieceDropTestUnchangedOnSkiasGrid(t *testing.T) {
	for _, y0 := range []float32{20, 20.1, 20.376, 20.5, 20.62, 20.9} {
		for _, h := range []float32{0, 0.02, 0.05, 0.13, 0.24, 0.246, 0.3, 0.5} {
			y1 := y0 + h
			skiaDrops := quarterPixelRound(y0) == quarterPixelRound(y1)
			var quad AnalyticQuadraticEdge
			if got := quad.setQuadratic([]geom.Point{
				geom.Pt(4, y0), geom.Pt(4.5, (y0+y1)/2), geom.Pt(5, y1),
			}, SkiaSnapAccuracy); got == skiaDrops {
				t.Errorf("setQuadratic kept=%v for y %v..%v at SkiaSnapAccuracy, want %v", got, y0, y1,
					!skiaDrops)
			}
			var cubic AnalyticCubicEdge
			if got := cubic.setCubic([]geom.Point{
				geom.Pt(4, y0), geom.Pt(4.3, y0+h/3), geom.Pt(4.7, y1-h/3), geom.Pt(5, y1),
			}, SkiaSnapAccuracy); got == skiaDrops {
				t.Errorf("setCubic kept=%v for y %v..%v at SkiaSnapAccuracy, want %v", got, y0, y1, !skiaDrops)
			}
		}
	}
}

// TestAnalyticSetLineDYIsReciprocalSlope checks the contract itself, not just agreement: for a slope the inverse table
// covers, DY is abs(1/DX) in Fixed. 0.5 of x over 8 of y has reciprocal slope 16.
func TestAnalyticSetLineDYIsReciprocalSlope(t *testing.T) {
	var e AnalyticEdge
	if !e.SetLine(geom.Pt(10, 10), geom.Pt(10.5, 18)) {
		t.Fatal("SetLine rejected a non-degenerate edge")
	}
	if want := 16 * FixedOne; e.DY != want {
		t.Fatalf("SetLine DY = %d, want %d (dy/dx for 0.5 over 8)", e.DY, want)
	}
}
