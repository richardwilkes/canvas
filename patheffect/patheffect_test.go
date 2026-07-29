// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package patheffect

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

func strokeRec(width float32) stroke.Rec {
	spec := stroke.PaintSpec{Style: stroke.PaintStyleStroke, Width: width, MiterLimit: 4}
	return stroke.NewStrokeRecFromPaint(&spec, 1)
}

func line(x0, y0, x1, y1 float32) *path.Path {
	p := &path.Path{}
	p.MoveTo(x0, y0)
	p.LineTo(x1, y1)
	return p
}

func countVerb(p *path.Path, verb path.Verb) int {
	n := 0
	iter := path.NewRawIter(p)
	for {
		v, _, _, ok := iter.Next()
		if !ok {
			return n
		}
		if v == verb {
			n++
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// dash

func TestMakeDashValidation(t *testing.T) {
	if MakeDash([]float32{10}, 0) != nil {
		t.Error("odd interval count should fail")
	}
	if MakeDash([]float32{10, -2}, 0) != nil {
		t.Error("negative interval should fail")
	}
	if MakeDash([]float32{0, 0}, 0) != nil {
		t.Error("zero total length should fail")
	}
	if MakeDash([]float32{10, 10}, float32(math.NaN())) != nil {
		t.Error("NaN phase should fail")
	}
	if MakeDash([]float32{10, 10}, 5) == nil {
		t.Error("valid dash should succeed")
	}
}

func TestDashSpecialLine(t *testing.T) {
	// A wide butt-capped line takes the SpecialLineRec lane: the dashes are emitted directly as filled quads and the
	// stroke rec is converted to fill.
	pe := MakeDash([]float32{10, 10}, 0)
	rec := strokeRec(4)
	dst := &path.Path{}
	if !pe.FilterPath(dst, line(0, 0, 100, 0), &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if rec.Style() != stroke.StyleFill {
		t.Errorf("special line should set the rec to fill, got %v", rec.Style())
	}
	// 100/20 = 5 dashes, each a 4-point polygon (move + 3 lines).
	if got := countVerb(dst, path.VerbMove); got != 5 {
		t.Errorf("dash count: got %d moves, want 5", got)
	}
	bounds := dst.Bounds()
	want := geom.RectLTRB(0, -2, 90, 2)
	if bounds != want {
		t.Errorf("dashed quads bounds: got %v, want %v", bounds, want)
	}
}

func TestDashHairlineAndFillBehavior(t *testing.T) {
	pe := MakeDash([]float32{10, 10}, 0)

	// hairline: dashes via the measure, rec stays hairline
	rec := strokeRec(0)
	dst := &path.Path{}
	if !pe.FilterPath(dst, line(0, 0, 100, 0), &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if rec.Style() != stroke.StyleHairline {
		t.Errorf("rec should stay hairline, got %v", rec.Style())
	}
	if got := countVerb(dst, path.VerbMove); got != 5 {
		t.Errorf("hairline dash count: got %d, want 5", got)
	}

	// fill style: dashing does nothing
	spec := stroke.PaintSpec{Style: stroke.PaintStyleFill, MiterLimit: 4}
	fillRec := stroke.NewStrokeRecFromPaint(&spec, 1)
	if pe.FilterPath(&path.Path{}, line(0, 0, 100, 0), &fillRec, nil, identity()) {
		t.Error("dash must not apply to fill-style paints")
	}
}

func TestDashPhaseEquivalence(t *testing.T) {
	// phase -20 on a 100-length pattern equals phase 80.
	src := line(0, 0, 200, 0)
	intervals := []float32{60, 40}
	a := &path.Path{}
	b := &path.Path{}
	recA := strokeRec(0)
	recB := strokeRec(0)
	MakeDash(intervals, -20).FilterPath(a, src, &recA, nil, identity())
	MakeDash(intervals, 80).FilterPath(b, src, &recB, nil, identity())
	if !path.Equal(a, b) {
		t.Error("phase -20 and phase 80 should produce identical dashes")
	}
}

func TestDashClosedContourJoin(t *testing.T) {
	// On a closed contour that starts and ends inside an "on" interval, the first segment is skipped and re-emitted
	// joined to the final segment (no seam at the start point).
	p := &path.Path{}
	p.AddRect(geom.RectLTRB(0, 0, 40, 40), geom.DirectionCW)
	pe := MakeDash([]float32{30, 10}, 0)
	rec := strokeRec(0)
	dst := &path.Path{}
	if !pe.FilterPath(dst, p, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if dst.IsEmpty() {
		t.Fatal("expected dashes")
	}
	// Perimeter 160 = 4 cycles of 40: four on-segments; the wrap join merges the last with the first.
	if got := countVerb(dst, path.VerbMove); got != 4 {
		t.Errorf("closed-rect dash contours: got %d, want 4", got)
	}
}

func TestDashCullRect(t *testing.T) {
	// A long horizontal line culled to a small rect keeps its dash phase.
	src := line(-10000, 5, 10000, 5)
	cull := geom.RectLTRB(0, 0, 100, 10)
	full := &path.Path{}
	culled := &path.Path{}
	recA := strokeRec(2)
	recB := strokeRec(2)
	pe := MakeDash([]float32{7, 3}, 0)
	pe.FilterPath(full, src, &recA, nil, identity())
	pe.FilterPath(culled, src, &recB, &cull, identity())
	if culled.IsEmpty() {
		t.Fatal("culled dash should not be empty")
	}
	if culled.CountPoints() >= full.CountPoints() {
		t.Errorf("culling should shrink the output: %d >= %d", culled.CountPoints(), full.CountPoints())
	}
	// Both outputs cover the cull rect with the same phase: each culled dash quad must appear in the full output
	// (compare X spans of the moves).
	fullXs := map[float32]bool{}
	it := path.NewRawIter(full)
	for {
		v, pts, _, ok := it.Next()
		if !ok {
			break
		}
		if v == path.VerbMove {
			fullXs[pts[0].X] = true
		}
	}
	it = path.NewRawIter(culled)
	for {
		v, pts, _, ok := it.Next()
		if !ok {
			break
		}
		if v == path.VerbMove && pts[0].X >= 0 && pts[0].X < 93 && !fullXs[pts[0].X] {
			t.Errorf("culled dash at X=%v not present in full output", pts[0].X)
		}
	}
}

func TestDashAsPoints(t *testing.T) {
	pe := MakeDash([]float32{2, 2}, 0).(*dashEffect)
	rec := strokeRec(2)
	ctm := identity()
	cull := geom.RectLTRB(-100, -100, 100, 100)

	var pd stroke.PointData
	if !pe.AsPoints(&pd, line(0, 5, 20, 5), &rec, ctm, &cull) {
		t.Fatal("AsPoints should handle an integer on==off horizontal line")
	}
	if pd.Size != (geom.Point{X: 1, Y: 1}) {
		t.Errorf("point size: got %v, want (1, 1)", pd.Size)
	}
	if len(pd.Points) == 0 {
		t.Fatal("expected points")
	}
	// centers advance by the interval length (4), starting at the center of the first dash (1)
	for i, pt := range pd.Points {
		nearly := geom.Point{X: 1 + float32(i)*4, Y: 5}
		if pt != nearly {
			t.Errorf("point %d: got %v, want %v", i, pt, nearly)
		}
	}

	// hairlines, non-integer intervals, angled lines, and round-capped angles are rejected
	rec0 := strokeRec(0)
	if pe.AsPoints(&pd, line(0, 5, 20, 5), &rec0, ctm, &cull) {
		t.Error("hairline should be rejected")
	}
	pe2 := MakeDash([]float32{2.5, 2.5}, 0).(*dashEffect)
	if pe2.AsPoints(&pd, line(0, 5, 20, 5), &rec, ctm, &cull) {
		t.Error("non-integer intervals should be rejected")
	}
	if pe.AsPoints(&pd, line(0, 0, 20, 20), &rec, ctm, &cull) {
		t.Error("angled butt-capped line should be rejected")
	}
}

///////////////////////////////////////////////////////////////////////////////
// corner

func TestCornerEffect(t *testing.T) {
	if MakeCorner(0) != nil || MakeCorner(-1) != nil || MakeCorner(float32(math.NaN())) != nil {
		t.Error("invalid radii should fail")
	}

	// open right angle: quad inserted at the corner
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(100, 0)
	p.LineTo(100, 100)
	pe := MakeCorner(10)
	rec := strokeRec(0)
	dst := &path.Path{}
	if !pe.FilterPath(dst, p, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	// two quads: the degenerate one emitted at the open contour's start, and the real corner
	if got := countVerb(dst, path.VerbQuad); got != 2 {
		t.Errorf("corner quads: got %d, want 2", got)
	}
	// the quad control point is the original corner
	found := false
	it := path.NewRawIter(dst)
	for {
		v, pts, _, ok := it.Next()
		if !ok {
			break
		}
		if v == path.VerbQuad && pts[1] == (geom.Point{X: 100, Y: 0}) {
			found = true
		}
	}
	if !found {
		t.Error("corner quad should pivot on the original corner point")
	}

	// closed rect: all four corners rounded
	rect := &path.Path{}
	rect.AddRect(geom.RectLTRB(0, 0, 100, 100), geom.DirectionCW)
	dst.Reset()
	rec = strokeRec(0)
	if !pe.FilterPath(dst, rect, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if got := countVerb(dst, path.VerbQuad); got != 4 {
		t.Errorf("rect corner quads: got %d, want 4", got)
	}
	// bounds stay within the source
	if !rect.Bounds().ContainsRect(dst.Bounds()) {
		t.Error("rounded rect must stay within the source bounds")
	}
	var b geom.Rect
	if !pe.ComputeFastBounds(&b) {
		t.Error("corner fast bounds should succeed")
	}
}

///////////////////////////////////////////////////////////////////////////////
// discrete

func TestDiscreteEffect(t *testing.T) {
	if MakeDiscrete(0, 1, 0) != nil || MakeDiscrete(float32(math.NaN()), 1, 0) != nil {
		t.Error("invalid segLength should fail")
	}

	src := line(0, 0, 100, 0)
	rec := strokeRec(2)
	a := &path.Path{}
	b := &path.Path{}
	pe := MakeDiscrete(10, 4, 0)
	recA := rec
	recB := rec
	if !pe.FilterPath(a, src, &recA, nil, identity()) ||
		!pe.FilterPath(b, src, &recB, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if !path.Equal(a, b) {
		t.Error("discrete must be deterministic for the same input")
	}
	// 100/10 = 10 segments -> 11 points
	if a.CountPoints() != 11 {
		t.Errorf("discrete points: got %d, want 11", a.CountPoints())
	}
	// deviations bounded by the deviation parameter (all displacement is in Y for a horizontal line)
	for i := 0; i < a.CountPoints(); i++ {
		if geom.ScalarAbs(a.Point(i).Y) > 4.0001 {
			t.Errorf("point %d deviates too far: %v", i, a.Point(i))
		}
	}

	// different seed assist changes the output
	c := &path.Path{}
	recC := rec
	MakeDiscrete(10, 4, 7).FilterPath(c, src, &recC, nil, identity())
	if path.Equal(a, c) {
		t.Error("seed assist should perturb differently")
	}

	// short path is passed through unmangled
	shortSrc := line(0, 0, 15, 0)
	d := &path.Path{}
	recD := rec
	MakeDiscrete(10, 4, 0).FilterPath(d, shortSrc, &recD, nil, identity())
	if d.CountPoints() != 2 || d.Point(1) != (geom.Point{X: 15}) {
		t.Errorf("short path should pass through: %d pts", d.CountPoints())
	}

	var bounds geom.Rect
	if !pe.ComputeFastBounds(&bounds) {
		t.Fatal("discrete fast bounds should succeed")
	}
	if bounds != geom.RectLTRB(-4, -4, 4, 4) {
		t.Errorf("fast bounds should outset by deviation: %v", bounds)
	}
}

///////////////////////////////////////////////////////////////////////////////
// trim

func TestTrimEffect(t *testing.T) {
	if MakeTrim(0, 1, TrimNormal) != nil {
		t.Error("full normal trim is a no-op and should return nil")
	}
	if MakeTrim(0.5, 0.25, TrimInverted) != nil {
		t.Error("empty inverted trim should return nil")
	}
	if MakeTrim(float32(math.NaN()), 1, TrimNormal) != nil {
		t.Error("NaN should fail")
	}

	src := line(0, 0, 100, 0)

	// normal: keep the middle half
	pe := MakeTrim(0.25, 0.75, TrimNormal)
	rec := strokeRec(0)
	dst := &path.Path{}
	if !pe.FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	want := line(25, 0, 75, 0)
	if !path.Equal(dst, want) {
		t.Errorf("normal trim mismatch: %v", dst.Bounds())
	}

	// inverted: the two outer quarters, tail first
	pe = MakeTrim(0.25, 0.75, TrimInverted)
	rec = strokeRec(0)
	dst.Reset()
	if !pe.FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if got := countVerb(dst, path.VerbMove); got != 2 {
		t.Errorf("inverted trim contours: got %d, want 2", got)
	}
	if dst.Point(0) != (geom.Point{X: 75}) {
		t.Errorf("inverted trim should add the tail first: %v", dst.Point(0))
	}

	// reversed normal trim (start >= stop after pinning) emits nothing but succeeds
	pe = MakeTrim(0.75, 0.25, TrimNormal)
	rec = strokeRec(0)
	dst.Reset()
	if !pe.FilterPath(dst, src, &rec, nil, identity()) || !dst.IsEmpty() {
		t.Error("reversed normal trim should succeed with empty output")
	}

	// single closed contour: inverted trim joins head to tail without a moveTo
	circle := &path.Path{}
	circle.AddCircle(50, 50, 25, geom.DirectionCW)
	pe = MakeTrim(0.25, 0.75, TrimInverted)
	rec = strokeRec(0)
	dst.Reset()
	if !pe.FilterPath(dst, circle, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if got := countVerb(dst, path.VerbMove); got != 1 {
		t.Errorf("closed-contour inverted trim should stay connected: %d moves", got)
	}
}

///////////////////////////////////////////////////////////////////////////////
// 1D

func TestPath1DEffect(t *testing.T) {
	stamp := &path.Path{}
	stamp.AddRect(geom.RectLTRB(-1, -1, 1, 1), geom.DirectionCW)

	if MakePath1D(stamp, 0, 0, Path1DTranslate) != nil {
		t.Error("advance <= 0 should fail")
	}
	if MakePath1D(&path.Path{}, 10, 0, Path1DTranslate) != nil {
		t.Error("empty stamp should fail")
	}

	// translate: stamps every 10 units along a 100 line starting at phase 0 -> 10 stamps
	pe := MakePath1D(stamp, 10, 0, Path1DTranslate)
	rec := strokeRec(4)
	dst := &path.Path{}
	if !pe.FilterPath(dst, line(0, 0, 100, 0), &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if rec.Style() != stroke.StyleFill {
		t.Error("1D effect should set the rec to fill")
	}
	if got := countVerb(dst, path.VerbMove); got != 10 {
		t.Errorf("stamp count: got %d, want 10", got)
	}
	// first stamp centered at distance 0
	if dst.Bounds().Left != -1 {
		t.Errorf("first stamp position: bounds %v", dst.Bounds())
	}

	// phase shifts the first stamp
	pe = MakePath1D(stamp, 10, 5, Path1DTranslate)
	rec = strokeRec(4)
	dst.Reset()
	pe.FilterPath(dst, line(0, 0, 100, 0), &rec, nil, identity())
	if dst.Bounds().Left != 4 {
		t.Errorf("phased stamp position: bounds %v", dst.Bounds())
	}

	// rotate: on a vertical line the stamp is rotated 90 degrees; with an asymmetric stamp the bounds swap axes
	tall := &path.Path{}
	tall.AddRect(geom.RectLTRB(-1, -3, 1, 3), geom.DirectionCW)
	pe = MakePath1D(tall, 50, 0, Path1DRotate)
	rec = strokeRec(4)
	dst.Reset()
	pe.FilterPath(dst, line(0, 0, 0, 100), &rec, nil, identity())
	b := dst.Bounds()
	if b.Width() != 6 || b.Height() != 52 {
		t.Errorf("rotated stamp bounds: %v", b)
	}

	// morph smoke: lines become quads
	pe = MakePath1D(stamp, 10, 0, Path1DMorph)
	rec = strokeRec(4)
	dst.Reset()
	pe.FilterPath(dst, line(0, 0, 100, 0), &rec, nil, identity())
	if countVerb(dst, path.VerbQuad) == 0 {
		t.Error("morph should emit quads for the stamp's lines")
	}
}

///////////////////////////////////////////////////////////////////////////////
// 2D

func TestPath2DEffects(t *testing.T) {
	if MakeLine2D(-1, identity()) != nil {
		t.Error("negative width should fail")
	}

	// 10x10 lattice over a 40x20 rect: lines at each covered row
	var lattice geom.Matrix
	lattice.SetScale(10, 10)
	pe := MakeLine2D(2, &lattice)
	src := &path.Path{}
	src.AddRect(geom.RectLTRB(0, 0, 40, 20), geom.DirectionCW)
	rec := strokeRec(0)
	dst := &path.Path{}
	if !pe.FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if rec.Style() != stroke.StyleStroke || rec.Width() != 2 {
		t.Errorf("line 2D should set stroke style width 2: %v %v", rec.Style(), rec.Width())
	}
	if got := countVerb(dst, path.VerbMove); got != 2 {
		t.Errorf("lattice line count: got %d, want 2", got)
	}

	// singular matrix fails
	var singular geom.Matrix
	singular.SetScale(0, 0)
	rec = strokeRec(0)
	if MakeLine2D(2, &singular).FilterPath(&path.Path{}, src, &rec, nil, identity()) {
		t.Error("singular lattice should fail")
	}

	// path 2D: stamps at each covered cell (4x2 cells)
	stamp := &path.Path{}
	stamp.AddCircle(0, 0, 2, geom.DirectionCW)
	pe = MakePath2D(&lattice, stamp)
	rec = strokeRec(0)
	dst.Reset()
	if !pe.FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if got := countVerb(dst, path.VerbMove); got != 8 {
		t.Errorf("stamp count: got %d, want 8", got)
	}
}

///////////////////////////////////////////////////////////////////////////////
// sum / compose

func TestSumAndCompose(t *testing.T) {
	dash := MakeDash([]float32{10, 10}, 0)
	trim := MakeTrim(0, 0.5, TrimNormal)

	if MakeSum(nil, dash) != dash || MakeSum(dash, nil) != dash {
		t.Error("MakeSum with nil should return the other effect")
	}
	if MakeCompose(nil, dash) != dash || MakeCompose(dash, nil) != dash {
		t.Error("MakeCompose with nil should return the other effect")
	}

	src := line(0, 0, 100, 0)

	// sum: both effects applied to the source independently, outputs concatenated
	sum := MakeSum(trim, dash)
	rec := strokeRec(0)
	dst := &path.Path{}
	if !sum.FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("sum FilterPath failed")
	}
	trimOnly := &path.Path{}
	recA := strokeRec(0)
	trim.FilterPath(trimOnly, src, &recA, nil, identity())
	dashOnly := &path.Path{}
	recB := strokeRec(0)
	dash.FilterPath(dashOnly, src, &recB, nil, identity())
	want := trimOnly.Clone()
	want.AddPath(dashOnly, path.AddPathAppend)
	if !path.Equal(dst, want) {
		t.Error("sum should concatenate both effects' outputs")
	}

	// compose: outer(inner(path)) — dash the trimmed half
	comp := MakeCompose(dash, trim)
	rec = strokeRec(0)
	dst.Reset()
	if !comp.FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("compose FilterPath failed")
	}
	wantComp := &path.Path{}
	recC := strokeRec(0)
	dash.FilterPath(wantComp, trimOnly, &recC, nil, identity())
	if !path.Equal(dst, wantComp) {
		t.Error("compose should apply inner then outer")
	}
	// the trimmed half is 50 long -> 3 dashes (0-10, 20-30, 40-50)
	if got := countVerb(dst, path.VerbMove); got != 3 {
		t.Errorf("composed dash count: got %d, want 3", got)
	}
}

func TestDashKeepsLoneMoveToInDst(t *testing.T) {
	// A dst holding exactly one verb is non-empty, but Path.AddPath treats it as replaceable and copies over it rather
	// than appending, so merging the dash's scratch path back with AddPath would silently drop that contour. MakeSum
	// reaches this: its first child can leave nothing but a MoveTo behind before the dash runs. The special-line lane
	// is the one that shows it — its AddPoly appends a fresh contour, where the measure lane's leading MoveTo would
	// collapse the pre-existing one either way.
	pe := MakeDash([]float32{10, 10}, 0)
	src := line(0, 0, 100, 0)

	dashOnly := &path.Path{}
	recDash := strokeRec(4)
	if !pe.FilterPath(dashOnly, src, &recDash, nil, identity()) {
		t.Fatal("FilterPath failed")
	}

	dst := &path.Path{}
	dst.MoveTo(-50, -50)
	rec := strokeRec(4)
	if !pe.FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("FilterPath failed")
	}
	if got, want := dst.CountVerbs(), dashOnly.CountVerbs()+1; got != want {
		t.Fatalf("dst verb count = %d, want %d (the pre-existing MoveTo plus the dashes)", got, want)
	}
	if got, want := dst.CountPoints(), dashOnly.CountPoints()+1; got != want {
		t.Fatalf("dst point count = %d, want %d", got, want)
	}
	if got := dst.Point(0); got != geom.Pt(-50, -50) {
		t.Errorf("dst's first point = %v, want the pre-existing MoveTo at (-50,-50)", got)
	}
	for i := range dashOnly.CountPoints() {
		if got, want := dst.Point(i+1), dashOnly.Point(i); got != want {
			t.Fatalf("dst point %d = %v, want %v", i+1, got, want)
		}
	}
	// The dropped MoveTo was observable through the bounds, which is what the discard would have changed.
	if got, want := dst.Bounds(), geom.RectLTRB(-50, -50, 90, 2); got != want {
		t.Errorf("dst bounds = %v, want %v", got, want)
	}
}

func TestDashOverflowKeepsPriorOutput(t *testing.T) {
	// Two segments rather than a straight line, so the dash takes the measure-based lane and leaves the stroke rec
	// alone (the special-line lane would switch it to fill, and the second dash would then decline the source).
	src := &path.Path{}
	src.MoveTo(0, 0)
	src.LineTo(1000000, 0)
	src.LineTo(1000000, 1000000)

	coarse := MakeDash([]float32{1000, 1000}, 0) // ~1000 dashes: well under maxDashCount
	tooFine := MakeDash([]float32{0.5, 0.5}, 0)  // 2,000,000 dashes: over maxDashCount, so it gives up

	// The coarse dash on its own is the yardstick.
	want := &path.Path{}
	recWant := strokeRec(0)
	if !coarse.FilterPath(want, src, &recWant, nil, identity()) {
		t.Fatal("coarse dash FilterPath failed")
	}
	if want.IsEmpty() {
		t.Fatal("expected the coarse dash to emit something")
	}

	// Giving up reports failure and contributes nothing, even when an earlier contour dashed successfully: the budget
	// accumulates across contours, so the short contour prefixed here is fully dashed before the long one blows it.
	twoContours := &path.Path{}
	twoContours.MoveTo(0, -10)
	twoContours.LineTo(100, -10)
	twoContours.AddPath(src, path.AddPathAppend)
	solo := &path.Path{}
	recSolo := strokeRec(0)
	if tooFine.FilterPath(solo, twoContours, &recSolo, nil, identity()) {
		t.Error("a dash past maxDashCount should report failure")
	}
	if !solo.IsEmpty() {
		t.Errorf("a dash that gave up left %d points behind", solo.CountPoints())
	}

	// Summed, the two share a dst: the bail-out must discard only its own (empty) contribution.
	dst := &path.Path{}
	rec := strokeRec(0)
	if !MakeSum(coarse, tooFine).FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("the sum should report the coarse dash's success")
	}
	if !path.Equal(dst, want) {
		t.Errorf("the dash bail-out erased the other effect's output: %d points, want %d", dst.CountPoints(),
			want.CountPoints())
	}
}

func TestComposeDiscardsFailedInnerOutput(t *testing.T) {
	// The 1D stamp effect writes stamps as it walks and only then gives up, when its iteration governor trips.
	stamp := line(0, 0, 1, 0)
	inner := MakePath1D(stamp, 1, 0, Path1DTranslate)
	src := line(0, 0, 200000, 0) // 200,000 advances, past maxReasonable1DIterations

	recInner := strokeRec(0)
	abandoned := &path.Path{}
	if inner.FilterPath(abandoned, src, &recInner, nil, identity()) {
		t.Fatal("precondition: the 1D governor should have tripped")
	}
	if abandoned.IsEmpty() {
		t.Fatal("precondition: the failed 1D effect should have stamped before giving up")
	}

	// Trim ignores the stroke rec, so it still runs after the 1D effect has switched the rec to fill.
	outer := MakeTrim(0, 0.5, TrimNormal)
	want := &path.Path{}
	recWant := strokeRec(0)
	if !outer.FilterPath(want, src, &recWant, nil, identity()) {
		t.Fatal("outer FilterPath failed")
	}

	dst := &path.Path{}
	rec := strokeRec(0)
	if !MakeCompose(outer, inner).FilterPath(dst, src, &rec, nil, identity()) {
		t.Fatal("compose should report the outer effect's success")
	}
	if !path.Equal(dst, want) {
		t.Errorf("compose kept geometry from the failed inner effect: %d points, want %d", dst.CountPoints(),
			want.CountPoints())
	}
}

///////////////////////////////////////////////////////////////////////////////
// FillPathWithPaint integration

func TestFillPathWithPaintAppliesEffect(t *testing.T) {
	src := line(0, 0, 100, 0)

	// dashed wide stroke: the effect runs, then the stroker; doFill true
	spec := stroke.PaintSpec{
		PathEffect: MakeDash([]float32{10, 10}, 0),
		Style:      stroke.PaintStyleStroke,
		Width:      4,
		MiterLimit: 4,
	}
	dst := &path.Path{}
	ident := identity()
	if !stroke.FillPathWithPaint(src, &spec, dst, nil, ident) {
		t.Fatal("dashed wide stroke should fill")
	}
	// The special line lane emits the quads directly; 5 dashes with bounds height = width
	if got := countVerb(dst, path.VerbMove); got != 5 {
		t.Errorf("dashes: got %d, want 5", got)
	}
	if dst.Bounds() != geom.RectLTRB(0, -2, 90, 2) {
		t.Errorf("bounds: %v", dst.Bounds())
	}

	// dashed hairline: effect runs, no stroking, doFill false
	spec.Width = 0
	dst.Reset()
	if stroke.FillPathWithPaint(src, &spec, dst, nil, ident) {
		t.Error("dashed hairline should report hairline (doFill false)")
	}
	if got := countVerb(dst, path.VerbMove); got != 5 {
		t.Errorf("hairline dashes: got %d, want 5", got)
	}
}

// identity returns a pointer to a fresh identity matrix.
func identity() *geom.Matrix {
	m := geom.IdentityMatrix()
	return &m
}
