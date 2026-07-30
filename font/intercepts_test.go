// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The gap geometry behind textblob's underline/strikeout carve-outs. Strike.FindIntercepts caches and charges what
// calculatePathGap computes (strike_test.go covers that side, through an 'H' whose outline is nothing but lines); this
// file pins the geometry itself, one verb lane at a time, against analytically known answers.

package font

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/path"
)

// gapIsEmpty reports whether calculatePathGap found nothing: the sentinel it seeds left/right with, which
// bandInterval turns into the empty interval.
func gapIsEmpty(left, right float32) bool { return left == scalarMax && right == scalarMin }

func TestCalculatePathGapLines(t *testing.T) {
	// A "V" of two line segments: down from (0,0) to (10,20), back up to (20,0). Each leg is x = y/2 and x = 20-y/2,
	// so a band at y in [8, 12] crosses at x = 4/16 on the way down and x = 16/4 on the way up. The band's own edges
	// are what the intersections are taken against, so the answer is the outermost pair: 4 and 16.
	v := path.New().MoveTo(0, 0).LineTo(10, 20).LineTo(20, 0)
	left, right := calculatePathGap(8, 12, v)
	if !near(left, 4, 1e-4) || !near(right, 16, 1e-4) {
		t.Errorf("V gap = (%v, %v), want (4, 16)", left, right)
	}
	// A band wide enough to swallow the whole V picks up the on-curve points strictly inside it instead: the apex at
	// x = 10 and both endpoints at x = 0 and x = 20.
	if left, right = calculatePathGap(-1, 21, v); !near(left, 0, 1e-4) || !near(right, 20, 1e-4) {
		t.Errorf("swallowing band gap = (%v, %v), want (0, 20)", left, right)
	}
	// A band the path never reaches reports the empty sentinel.
	if left, right = calculatePathGap(100, 110, v); !gapIsEmpty(left, right) {
		t.Errorf("gap over an unreached band = (%v, %v), want the empty sentinel", left, right)
	}
	// The close verb is tolerated and adds nothing: the closing edge from (20,0) back to (0,0) runs along y = 0, clear
	// of the band, so a closed contour answers exactly as the open one does.
	closed := path.New().MoveTo(0, 0).LineTo(10, 20).LineTo(20, 0).Close()
	if left, right = calculatePathGap(8, 12, closed); !near(left, 4, 1e-4) || !near(right, 16, 1e-4) {
		t.Errorf("closed V gap = (%v, %v), want (4, 16)", left, right)
	}
}

func TestCalculatePathGapHorizontalSegments(t *testing.T) {
	// A horizontal segment has a zero denominator, so the parameter comes back as an IEEE infinity (or a NaN when the
	// segment sits exactly on the band edge, where the numerator is zero too). Both fail 0 <= t && t < 1, which is the
	// whole reason the divide is spelled out rather than guarded — and the segment's endpoints are on the edge rather
	// than strictly inside it, so they contribute nothing either.
	onEdge := path.New().MoveTo(0, 10).LineTo(20, 10)
	if left, right := calculatePathGap(10, 20, onEdge); !gapIsEmpty(left, right) {
		t.Errorf("segment on the band's top edge = (%v, %v), want the empty sentinel", left, right)
	}
	if left, right := calculatePathGap(0, 10, onEdge); !gapIsEmpty(left, right) {
		t.Errorf("segment on the band's bottom edge = (%v, %v), want the empty sentinel", left, right)
	}
	// Strictly inside the band it does contribute — through its points, not through either edge crossing.
	if left, right := calculatePathGap(5, 15, onEdge); !near(left, 0, 1e-4) || !near(right, 20, 1e-4) {
		t.Errorf("segment inside the band = (%v, %v), want (0, 20)", left, right)
	}
}

func TestCalculatePathGapQuads(t *testing.T) {
	// A symmetric quad: P0 (0,0), P1 (10,20), P2 (20,0) parameterizes to x = 20t, y = 40t(1-t), so y = 7.5 at
	// t = 0.25 and t = 0.75 — x = 5 and x = 15. It peaks at y = 10, so the band's far edge at 18 never meets it and
	// no control point lies strictly inside the band.
	q := path.New().MoveTo(0, 0).QuadTo(10, 20, 20, 0)
	if left, right := calculatePathGap(7.5, 18, q); !near(left, 5, 1e-3) || !near(right, 15, 1e-3) {
		t.Errorf("quad gap = (%v, %v), want (5, 15)", left, right)
	}
	// A band that swallows the curve falls back to the verb's points, off-curve control included: x = 0, 10, 20.
	if left, right := calculatePathGap(-1, 21, q); !near(left, 0, 1e-4) || !near(right, 20, 1e-4) {
		t.Errorf("swallowing band quad gap = (%v, %v), want (0, 20)", left, right)
	}
	// Above the curve's whole y-extent the lane is skipped outright.
	if left, right := calculatePathGap(30, 40, q); !gapIsEmpty(left, right) {
		t.Errorf("quad gap over an unreached band = (%v, %v), want the empty sentinel", left, right)
	}
}

func TestCalculatePathGapCubics(t *testing.T) {
	// A symmetric cubic: P0 (0,0), P1 (0,30), P2 (30,30), P3 (30,0) parameterizes to y = 90t(1-t) and
	// x = 90t^2 - 60t^3, so y = 20 at t = 1/3 and t = 2/3 — x = 70/9 and x = 200/9. It peaks at y = 22.5, below the
	// band's far edge at 25, and no control point lies strictly inside the band.
	c := path.New().MoveTo(0, 0).CubicTo(0, 30, 30, 30, 30, 0)
	if left, right := calculatePathGap(20, 25, c); !near(left, 70.0/9, 1e-3) || !near(right, 200.0/9, 1e-3) {
		t.Errorf("cubic gap = (%v, %v), want (%v, %v)", left, right, 70.0/9, 200.0/9)
	}
	// Swallowed, the verb's four points answer instead: x = 0, 0, 30, 30.
	if left, right := calculatePathGap(-1, 31, c); !near(left, 0, 1e-4) || !near(right, 30, 1e-4) {
		t.Errorf("swallowing band cubic gap = (%v, %v), want (0, 30)", left, right)
	}
	if left, right := calculatePathGap(40, 50, c); !gapIsEmpty(left, right) {
		t.Errorf("cubic gap over an unreached band = (%v, %v), want the empty sentinel", left, right)
	}
}

func TestCalculatePathGapIgnoresConics(t *testing.T) {
	// Glyph outlines come from sfnt quad/cubic segments, so a conic never reaches here and the lane deliberately does
	// nothing with one. A conic sweeping straight through the band still reports the empty sentinel, which is the
	// documented "nothing intersects" answer rather than a wrong gap.
	c := path.New().MoveTo(0, 0).ConicTo(10, 20, 20, 0, 2)
	verbs := 0
	iter := path.NewIter(c, false)
	for {
		verb, _, ok := iter.Next()
		if !ok {
			break
		}
		if verb == path.VerbConic {
			verbs++
		}
	}
	if verbs != 1 {
		t.Fatalf("the test path holds %d conic verbs, want 1; the lane is not being reached", verbs)
	}
	if left, right := calculatePathGap(5, 8, c); !gapIsEmpty(left, right) {
		t.Errorf("conic gap = (%v, %v), want the empty sentinel", left, right)
	}
}

func TestIEEEFloatDivide(t *testing.T) {
	// calculatePathGap relies on the divide keeping IEEE semantics rather than trapping or clamping: a zero
	// denominator has to produce an infinity (or a NaN when the numerator is zero too) so that the 0 <= t && t < 1
	// range check is what rejects the degenerate segment.
	if got := ieeeFloatDivide(3, 2); got != 1.5 {
		t.Errorf("3/2 = %v, want 1.5", got)
	}
	if got := ieeeFloatDivide(1, 0); !math.IsInf(float64(got), 1) {
		t.Errorf("1/0 = %v, want +Inf", got)
	}
	if got := ieeeFloatDivide(-1, 0); !math.IsInf(float64(got), -1) {
		t.Errorf("-1/0 = %v, want -Inf", got)
	}
	if got := ieeeFloatDivide(0, 0); !math.IsNaN(float64(got)) {
		t.Errorf("0/0 = %v, want NaN", got)
	}
	for _, v := range []float32{float32(math.Inf(1)), float32(math.Inf(-1)), 0, 1} {
		if got := ieeeFloatDivide(v, v); v != 1 && !math.IsNaN(float64(got)) {
			t.Errorf("%v/%v = %v, want NaN", v, v, got)
		}
	}
	// The range check is what rejects each of those, not the divide.
	for _, tv := range []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), 1, -0.001} {
		if 0 <= tv && tv < 1 {
			t.Errorf("t = %v passes the range check; a degenerate segment would contribute a gap", tv)
		}
	}
}

// TestStrikeInterceptsCurvedGlyph runs the curved lanes through the caching entry point the text blob actually calls,
// with a glyph whose outline is quads rather than the straight-sided 'H' strike_test.go uses.
func TestStrikeInterceptsCurvedGlyph(t *testing.T) {
	tf := loadTypeface(t, "Roboto-Regular.ttf", 0)
	spec := MakeWithNoDeviceSpec(NewFont(tf, 64, 1, 0), nil)
	s := NewStrikeCache().FindOrCreateStrike(&spec)
	gid := tf.UnicharToGlyph('o')
	if gid == 0 {
		t.Fatal("Roboto should map 'o'")
	}
	g := s.PreparePaths([]uint16{gid}, nil)[0]
	if g.Path() == nil {
		t.Fatal("'o' should retain a device path")
	}
	quads := 0
	iter := path.NewIter(g.Path(), false)
	for {
		verb, _, ok := iter.Next()
		if !ok {
			break
		}
		if verb == path.VerbQuad {
			quads++
		}
	}
	if quads == 0 {
		t.Fatal("'o' has no quad segments; this case adds nothing over the 'H' one")
	}
	// A thin band across the middle of the bowl (device space is y-down, so the glyph sits above the baseline) carves
	// out the counter between the two sides of the ring, which has to be narrower than the glyph itself.
	var count int
	mid := (g.Rect().Top + g.Rect().Bottom) / 2
	intervals := s.FindIntercepts([2]float32{mid - 1, mid + 1}, 1, 0, g, nil, &count)
	if count != 2 || len(intervals) != 0 {
		t.Fatalf("count-only call returned %v and counted %d, want nil and 2", intervals, count)
	}
	count = 0
	intervals = s.FindIntercepts([2]float32{mid - 1, mid + 1}, 1, 0, g, make([]float32, 0, 2), &count)
	if len(intervals) != 2 || intervals[0] >= intervals[1] {
		t.Fatalf("intercepts = %v, want one nonempty interval across the bowl", intervals)
	}
	if bounds := g.Rect(); intervals[0] < bounds.Left || intervals[1] > bounds.Right {
		t.Errorf("intercepts %v escape the glyph bounds %v", intervals, bounds)
	}
	if intervals[1]-intervals[0] >= g.Rect().Width() {
		t.Errorf("intercepts %v span the whole %v-wide glyph; the counter should be narrower", intervals,
			g.Rect().Width())
	}
	// Scale and x-offset are applied to the cached interval on the way out.
	count = 0
	scaled := s.FindIntercepts([2]float32{mid - 1, mid + 1}, 2, 100, g, make([]float32, 0, 2), &count)
	for i := range scaled {
		if want := intervals[i]*2 + 100; !near(scaled[i], want, 1e-4) {
			t.Errorf("scaled intercept %d = %v, want %v", i, scaled[i], want)
		}
	}
	// A band nowhere near the glyph carves nothing out and contributes no interval at all.
	count = 0
	if got := s.FindIntercepts([2]float32{1000, 1001}, 1, 0, g, make([]float32, 0, 2), &count); len(got) != 0 ||
		count != 0 {
		t.Errorf("a band clear of the glyph produced %v (count %d), want none", got, count)
	}
}

// TestStrikeInterceptsPathlessGlyph covers the lane a glyph with no outline at all takes — a bitmap-only face, which
// is what an underline drawn across an emoji run meets. It carves nothing out of any band, and the empty answer is
// still cached so a caller sweeping band after band does not re-ask forever.
func TestStrikeInterceptsPathlessGlyph(t *testing.T) {
	tf := loadTypeface(t, "sbix.ttf", 0)
	spec := MakeWithNoDeviceSpec(NewFont(tf, 64, 1, 0), nil)
	s := NewStrikeCache().FindOrCreateStrike(&spec)
	gid := tf.UnicharToGlyph(smiley)
	if gid == 0 {
		t.Fatal("sbix.ttf should map U+1F600")
	}
	g := s.PreparePaths([]uint16{gid}, nil)[0]
	if g.Path() != nil {
		t.Fatal("an sbix glyph should retain no outline; this case is testing the wrong lane")
	}
	var count int
	if got := s.FindIntercepts([2]float32{-40, -30}, 1, 0, g, make([]float32, 0, 2), &count); len(got) != 0 ||
		count != 0 {
		t.Errorf("a pathless glyph produced %v (count %d), want none", got, count)
	}
	if len(g.intercepts) != 1 {
		t.Errorf("the empty answer cached %d entries, want 1", len(g.intercepts))
	}
	s.FindIntercepts([2]float32{-40, -30}, 1, 0, g, nil, &count)
	if len(g.intercepts) != 1 {
		t.Errorf("re-asking the same band cached %d entries, want 1", len(g.intercepts))
	}
}
