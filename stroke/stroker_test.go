// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package stroke

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

func strokeWith(width float32, c Cap, j Join, miter float32, src *path.Path) *path.Path {
	st := NewStroke()
	st.SetWidth(width)
	st.SetCap(c)
	st.SetJoin(j)
	st.SetMiterLimit(miter)
	dst := &path.Path{}
	st.StrokePath(src, dst)
	return dst
}

// A horizontal line stroked with butt caps must fill exactly the [x0,x1] x [y-r,y+r] rectangle.
func TestStrokeLineButt(t *testing.T) {
	src := (&path.Path{}).MoveTo(10, 40).LineTo(90, 40)
	dst := strokeWith(6, CapButt, JoinMiter, 4, src)
	b := dst.Bounds()
	want := geom.RectLTRB(10, 37, 90, 43)
	if b != want {
		t.Fatalf("butt stroke bounds %v want %v", b, want)
	}
	if !dst.Contains(50, 39) || !dst.Contains(50, 41) {
		t.Fatal("interior of stroke band not contained")
	}
	if dst.Contains(9, 40) || dst.Contains(91, 40) {
		t.Fatal("butt caps must not extend past the endpoints")
	}
}

// Square caps extend half the width past each endpoint; round caps too (as an arc).
func TestStrokeLineCaps(t *testing.T) {
	src := (&path.Path{}).MoveTo(10, 40).LineTo(90, 40)

	sq := strokeWith(6, CapSquare, JoinMiter, 4, src)
	if b, want := sq.Bounds(), geom.RectLTRB(7, 37, 93, 43); b != want {
		t.Fatalf("square cap bounds %v want %v", b, want)
	}

	rd := strokeWith(6, CapRound, JoinMiter, 4, src)
	b := rd.Bounds()
	if b.Left != 7 || b.Right != 93 {
		t.Fatalf("round cap bounds %v want left 7 right 93", b)
	}
	if !rd.Contains(8, 40) { // inside the round cap
		t.Fatal("round cap area not contained")
	}
	if rd.Contains(7.2, 37.2) { // corner outside the arc but inside the square cap
		t.Fatal("round cap contains a square-cap corner")
	}
}

// A closed contour produces an outer contour plus a reversed inner contour; the band between them is filled, the hole
// is not.
func TestStrokeClosedTriangle(t *testing.T) {
	src := (&path.Path{}).MoveTo(20, 20).LineTo(80, 20).LineTo(50, 70).Close()
	dst := strokeWith(4, CapButt, JoinMiter, 4, src)
	if !dst.Contains(50, 21) { // on the top edge
		t.Fatal("stroke band on the top edge not contained")
	}
	if dst.Contains(50, 40) { // centroid: inside the hole
		t.Fatal("hole of closed stroke should not be filled")
	}
}

// The closed-rect specialization: miter join emits outer+inner rects, bevel an octagon, round a round-rect; a stroke
// wider than the rect fills it solid (no inner contour).
func TestStrokeRectLanes(t *testing.T) {
	rect := geom.RectLTRB(20, 20, 80, 60)
	src := (&path.Path{}).AddRect(rect, geom.DirectionCW)

	mi := strokeWith(10, CapButt, JoinMiter, 4, src)
	if b, want := mi.Bounds(), geom.RectLTRB(15, 15, 85, 65); b != want {
		t.Fatalf("miter rect stroke bounds %v want %v", b, want)
	}
	if !mi.Contains(16, 16) { // miter corner is square
		t.Fatal("miter corner missing")
	}
	if mi.Contains(50, 40) {
		t.Fatal("rect stroke hole filled")
	}

	bv := strokeWith(10, CapButt, JoinBevel, 4, src)
	if bv.Contains(15.5, 15.5) { // bevel cuts the corner
		t.Fatal("bevel corner should be cut")
	}
	if !bv.Contains(20, 15.5) { // but the edge band is there
		t.Fatal("bevel edge band missing")
	}

	solid := strokeWith(45, CapButt, JoinMiter, 4, src) // width >= min(w,h): no inner contour
	if !solid.Contains(50, 40) {
		t.Fatal("over-wide rect stroke should fill the center")
	}
}

// Zero-length contours draw round/square caps but nothing with butt caps.
func TestStrokeZeroLengthContour(t *testing.T) {
	src := (&path.Path{}).MoveTo(50, 50).Close()

	if dst := strokeWith(8, CapButt, JoinMiter, 4, src); !dst.IsEmpty() {
		t.Fatal("butt-capped zero-length contour should draw nothing")
	}
	rd := strokeWith(8, CapRound, JoinMiter, 4, src)
	if !rd.Contains(50, 52) || !rd.Contains(52, 50) {
		t.Fatal("round-capped zero-length contour should draw a dot")
	}
	sq := strokeWith(8, CapSquare, JoinMiter, 4, src)
	if !sq.Contains(53.5, 53.5) {
		t.Fatal("square-capped zero-length contour should draw a square")
	}
}

// A tight miter limit converts sharp corners to bevels: the spike must not extend to the miter point.
func TestStrokeMiterLimit(t *testing.T) {
	src := (&path.Path{}).MoveTo(10, 20).LineTo(90, 22).LineTo(12, 40) // very sharp turn at (90,22)
	wide := strokeWith(6, CapButt, JoinMiter, 100, src)
	tight := strokeWith(6, CapButt, JoinMiter, 1.5, src)
	if wide.Bounds().Right <= tight.Bounds().Right+1 {
		t.Fatalf("miter spike should extend well past the beveled join: wide %v tight %v",
			wide.Bounds().Right, tight.Bounds().Right)
	}
}

// Cubic cusps get a round patch (addCircle) at the cusp location.
func TestStrokeCubicCuspAddsCircle(t *testing.T) {
	// symmetric cusp: tangent collapses at t=0.5
	src := (&path.Path{}).MoveTo(20, 20).CubicTo(80, 80, 20, 80, 80, 20)
	cusp := geom.FindCubicCusp([]geom.Point{{X: 20, Y: 20}, {X: 80, Y: 80}, {X: 20, Y: 80}, {X: 80, Y: 20}})
	if cusp <= 0 {
		t.Skip("geometry no longer has a cusp")
	}
	dst := strokeWith(6, CapButt, JoinMiter, 4, src)
	if dst.SegmentMasks()&path.SegmentConic == 0 {
		t.Fatal("cusp circle (conics) missing from stroke output")
	}
}

// Stroke-and-fill of a convex line-only closed path keeps only the outer contour (ignore-center lane).
func TestStrokeAndFillConvex(t *testing.T) {
	src := (&path.Path{}).AddPoly([]geom.Point{{X: 30, Y: 30}, {X: 70, Y: 30}, {X: 70, Y: 70}, {X: 30, Y: 70}}, true)
	st := NewStroke()
	st.SetWidth(10)
	st.SetDoFill(true)
	dst := &path.Path{}
	st.StrokePath(src, dst)
	if !dst.Contains(50, 50) {
		t.Fatal("stroke-and-fill must cover the interior")
	}
	if !dst.Contains(24.9+1, 30) { // outer edge band (26, 30)
		t.Fatal("stroke-and-fill must cover the outer band")
	}
}

// Inverseness of the source must be preserved.
func TestStrokePreservesInverse(t *testing.T) {
	src := (&path.Path{}).MoveTo(20, 20).LineTo(80, 25).LineTo(50, 70)
	src.SetFillType(path.FillInverseWinding)
	dst := strokeWith(4, CapButt, JoinMiter, 4, src)
	if !dst.FillType().IsInverse() {
		t.Fatal("inverse fill type not preserved through stroking")
	}
	// and through the rect specialization
	rsrc := (&path.Path{}).AddRect(geom.RectLTRB(20, 20, 80, 60), geom.DirectionCW)
	rsrc.SetFillType(path.FillInverseWinding)
	rdst := strokeWith(4, CapButt, JoinMiter, 4, rsrc)
	if !rdst.FillType().IsInverse() {
		t.Fatal("inverse fill type not preserved through strokeRect")
	}
}

// StrokePath with src == dst must behave as with distinct paths.
func TestStrokePathAliased(t *testing.T) {
	mk := func() *path.Path { return (&path.Path{}).MoveTo(10, 40).LineTo(90, 40) }
	want := strokeWith(6, CapButt, JoinMiter, 4, mk())
	aliased := mk()
	st := NewStroke()
	st.SetWidth(6)
	st.StrokePath(aliased, aliased)
	if !path.Equal(want, aliased) {
		t.Fatal("aliased StrokePath differs from non-aliased result")
	}
}

// A degenerate (zero/negative) stroke width must replace dst's previous contents with an empty path rather than
// leaving stale geometry from a prior stroke behind, per StrokePath's documented contract.
func TestStrokeZeroWidthClearsDst(t *testing.T) {
	src := (&path.Path{}).MoveTo(10, 40).LineTo(90, 40)
	for _, width := range []float32{0, -3} {
		// Prime dst with real geometry from an ordinary stroke, then confirm the degenerate width wipes it.
		dst := strokeWith(6, CapButt, JoinMiter, 4, src)
		if dst.IsEmpty() {
			t.Fatal("expected non-empty dst after a normal stroke")
		}
		st := NewStroke()
		st.SetWidth(width)
		st.StrokePath(src, dst)
		if !dst.IsEmpty() {
			t.Fatalf("width %v: dst not cleared, still holds %d verbs", width, dst.CountVerbs())
		}
	}
}

func TestStrokeRecStyles(t *testing.T) {
	rec := NewStrokeRec(InitStyleFill)
	if rec.Style() != StyleFill || rec.IsHairlineStyle() || !rec.IsFillStyle() {
		t.Fatal("fill init style wrong")
	}
	rec = NewStrokeRec(InitStyleHairline)
	if rec.Style() != StyleHairline || !rec.IsHairlineStyle() {
		t.Fatal("hairline init style wrong")
	}
	rec.SetStrokeStyle(5, false)
	if rec.Style() != StyleStroke || !rec.NeedToApply() {
		t.Fatal("stroke style wrong")
	}
	rec.SetStrokeStyle(5, true)
	if rec.Style() != StyleStrokeAndFill {
		t.Fatal("stroke-and-fill style wrong")
	}
	rec.SetStrokeStyle(0, true) // hairline+fill == fill
	if rec.Style() != StyleFill {
		t.Fatal("width-0 stroke-and-fill should collapse to fill")
	}
	rec.SetStrokeStyle(0, false)
	if rec.Style() != StyleHairline {
		t.Fatal("width-0 stroke should be hairline")
	}

	paint := &PaintSpec{Style: PaintStyleStrokeAndFill, Width: 0, MiterLimit: 4}
	rec = NewStrokeRecFromPaint(paint, 1)
	if rec.Style() != StyleFill {
		t.Fatal("paint width-0 stroke-and-fill should collapse to fill")
	}
}

func TestStrokeRecApplyToPath(t *testing.T) {
	src := (&path.Path{}).MoveTo(10, 40).LineTo(90, 40)
	rec := NewStrokeRec(InitStyleHairline)
	dst := &path.Path{}
	if rec.ApplyToPath(dst, src) {
		t.Fatal("hairline must not apply")
	}
	rec.SetStrokeStyle(6, false)
	if !rec.ApplyToPath(dst, src) {
		t.Fatal("stroke must apply")
	}
	if b, want := dst.Bounds(), geom.RectLTRB(10, 37, 90, 43); b != want {
		t.Fatalf("applied stroke bounds %v want %v", b, want)
	}
}

func TestGetInflationRadius(t *testing.T) {
	if got := GetInflationRadius(JoinMiter, 4, CapButt, -1); got != 0 {
		t.Fatalf("fill inflation %v want 0", got)
	}
	if got := GetInflationRadius(JoinMiter, 4, CapButt, 0); got != 1 {
		t.Fatalf("hairline inflation %v want 1", got)
	}
	if got := GetInflationRadius(JoinMiter, 4, CapButt, 10); got != 20 {
		t.Fatalf("miter inflation %v want 20", got)
	}
	if got := GetInflationRadius(JoinBevel, 4, CapButt, 10); got != 5 {
		t.Fatalf("bevel inflation %v want 5", got)
	}
	want := float32(10) / 2 * float32(math.Sqrt2)
	if got := GetInflationRadius(JoinBevel, 4, CapSquare, 10); got != want {
		t.Fatalf("square-cap inflation %v want %v", got, want)
	}
}

func TestFillPathWithPaint(t *testing.T) {
	src := (&path.Path{}).MoveTo(10, 40).LineTo(90, 40)
	var ident geom.Matrix
	ident.SetScale(1, 1)

	// hairline: dst is a copy of src, returns false
	hair := &PaintSpec{Style: PaintStyleStroke, Width: 0, MiterLimit: 4}
	dst := &path.Path{}
	if FillPathWithPaint(src, hair, dst, nil, &ident) {
		t.Fatal("hairline should return false")
	}
	if !path.Equal(dst, src) {
		t.Fatal("hairline dst should equal src")
	}

	// fill style: dst is a copy, returns true
	fill := &PaintSpec{Style: PaintStyleFill, MiterLimit: 4}
	if !FillPathWithPaint(src, fill, dst, nil, &ident) {
		t.Fatal("fill should return true")
	}
	if !path.Equal(dst, src) {
		t.Fatal("fill dst should equal src")
	}

	// stroke: dst is the stroked outline
	strokeSpec := &PaintSpec{Style: PaintStyleStroke, Width: 6, MiterLimit: 4}
	if !FillPathWithPaint(src, strokeSpec, dst, nil, &ident) {
		t.Fatal("stroke should return true")
	}
	if b, want := dst.Bounds(), geom.RectLTRB(10, 37, 90, 43); b != want {
		t.Fatalf("stroked bounds %v want %v", b, want)
	}

	// resScale scales the subdivision tolerance through the ctm form
	if !FillPathWithPaintResScale(src, strokeSpec, dst, nil, 4) {
		t.Fatal("resScale form should return true")
	}

	// non-finite src: empty dst, false
	bad := (&path.Path{}).MoveTo(float32(math.Inf(1)), 0).LineTo(1, 1)
	if FillPathWithPaint(bad, strokeSpec, dst, nil, &ident) {
		t.Fatal("non-finite src should return false")
	}
	if !dst.IsEmpty() {
		t.Fatal("non-finite src should produce an empty dst")
	}

	// src == dst aliasing
	aliased := (&path.Path{}).MoveTo(10, 40).LineTo(90, 40)
	if !FillPathWithPaint(aliased, strokeSpec, aliased, nil, &ident) {
		t.Fatal("aliased call should return true")
	}
	if b, want := aliased.Bounds(), geom.RectLTRB(10, 37, 90, 43); b != want {
		t.Fatalf("aliased stroked bounds %v want %v", b, want)
	}
}
