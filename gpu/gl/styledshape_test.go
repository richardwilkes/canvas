// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for StyledShape: unstyled keys (state flags, geometry words, the small-path key-from-data lane and the
// gen-ID lane), the inherited style-key machinery (closed/no-joins flag trimming), the stroke simplifications,
// MakeFilled, and asNestedRects, covering the in-scope lanes.

package gl

import (
	"slices"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

func unstyledKey(t *testing.T, s *StyledShape) []uint32 {
	t.Helper()
	n := s.UnstyledKeySize()
	if n < 0 {
		return nil
	}
	key := make([]uint32, n)
	s.WriteUnstyledKey(key)
	return key
}

func strokeRecWith(width float32, capType stroke.Cap, join stroke.Join, miter float32, strokeAndFill bool) stroke.Rec {
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(width, strokeAndFill)
	rec.SetStrokeParams(capType, join, miter)
	return rec
}

func starPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(50, 10)
	p.LineTo(70, 90)
	p.LineTo(10, 40)
	p.LineTo(90, 40)
	p.LineTo(30, 90)
	p.Close()
	return p
}

func zigzagPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(0, 0)
	for i := 1; i <= 12; i++ {
		y := float32(0)
		if i%2 == 1 {
			y = 10
		}
		p.LineTo(float32(i*7), y+float32(i))
	}
	return p
}

func TestStyledShapeGeometryKeys(t *testing.T) {
	fill := SimpleFillStyle()

	rectA := MakeStyledShapeRect(geom.RectLTRB(1, 2, 30, 40), fill, DoSimplifyYes)
	rectA2 := MakeStyledShapeRect(geom.RectLTRB(1, 2, 30, 40), fill, DoSimplifyYes)
	rectB := MakeStyledShapeRect(geom.RectLTRB(1, 2, 30, 41), fill, DoSimplifyYes)

	keyA := unstyledKey(t, &rectA)
	if len(keyA) != 1+4 {
		t.Fatalf("rect key length = %d, want 5", len(keyA))
	}
	if !slices.Equal(keyA, unstyledKey(t, &rectA2)) {
		t.Fatal("identical rects produced different keys")
	}
	if slices.Equal(keyA, unstyledKey(t, &rectB)) {
		t.Fatal("different rects produced identical keys")
	}

	rr := geom.MakeRRect(geom.RectLTRB(0, 0, 20, 20), 4, 6)
	rrectA := MakeStyledShapeRRect(rr, fill, DoSimplifyYes)
	if got := len(unstyledKey(t, &rrectA)); got != 1+12 {
		t.Fatalf("rrect key length = %d, want 13", got)
	}
}

func TestStyledShapePathKeyFromData(t *testing.T) {
	fill := SimpleFillStyle()

	// Small concave paths (<= 10 verbs) are keyed from their data: two independently built, identical paths share a key
	// even though their generation IDs differ.
	a := MakeStyledShapePath(starPath(), fill, DoSimplifyYes)
	b := MakeStyledShapePath(starPath(), fill, DoSimplifyYes)
	keyA := unstyledKey(t, &a)
	if keyA == nil {
		t.Fatal("small path shape has no key")
	}
	if !slices.Equal(keyA, unstyledKey(t, &b)) {
		t.Fatal("identical small paths produced different keys")
	}

	other := starPath()
	other.LineTo(50, 50)
	c := MakeStyledShapePath(other, fill, DoSimplifyYes)
	if slices.Equal(keyA, unstyledKey(t, &c)) {
		t.Fatal("different small paths produced identical keys")
	}

	// The fill type is part of the state key.
	inv := starPath()
	inv.SetFillType(path.FillInverseWinding)
	d := MakeStyledShapePath(inv, fill, DoSimplifyYes)
	if slices.Equal(keyA, unstyledKey(t, &d)) {
		t.Fatal("inverse fill produced the same key as the normal fill")
	}
}

func TestStyledShapePathKeyFromGenID(t *testing.T) {
	fill := SimpleFillStyle()

	// Large paths key on the generation ID: the same source path keys equal, an independently rebuilt identical path
	// keys differently, and a volatile path has no key.
	src := zigzagPath()
	a := MakeStyledShapePath(src, fill, DoSimplifyYes)
	b := MakeStyledShapePath(src, fill, DoSimplifyYes)
	keyA := unstyledKey(t, &a)
	if got := len(keyA); got != 2 {
		t.Fatalf("gen-ID path key length = %d, want 2 (state + genID)", got)
	}
	if !slices.Equal(keyA, unstyledKey(t, &b)) {
		t.Fatal("the same source path produced different keys")
	}
	rebuilt := MakeStyledShapePath(zigzagPath(), fill, DoSimplifyYes)
	if slices.Equal(keyA, unstyledKey(t, &rebuilt)) {
		t.Fatal("independently rebuilt path shares the gen-ID key")
	}

	vol := zigzagPath()
	vol.SetVolatile(true)
	v := MakeStyledShapePath(vol, fill, DoSimplifyYes)
	if v.UnstyledKeySize() >= 0 {
		t.Fatal("volatile path must not be keyable")
	}
	if v.HasUnstyledKey() {
		t.Fatal("volatile path reports a key")
	}
}

func TestStyledShapeInheritedStyleKey(t *testing.T) {
	// Applying a style produces an inherited (geo,stroke) key: equal styles on equal geometry agree; the stroke
	// parameters that matter are included.
	mk := func(width float32, capType stroke.Cap, join stroke.Join) StyledShape {
		style := MakeStyle(strokeRecWith(width, capType, join, 4, false))
		s := MakeStyledShapePath(starPath(), style, DoSimplifyYes)
		return s.ApplyStyle(StyleApplyPathEffectAndStrokeRec, 1)
	}
	a := mk(4, stroke.CapButt, stroke.JoinMiter)
	b := mk(4, stroke.CapButt, stroke.JoinMiter)
	keyA := unstyledKey(t, &a)
	if keyA == nil {
		t.Fatal("styled output has no inherited key")
	}
	if !slices.Equal(keyA, unstyledKey(t, &b)) {
		t.Fatal("identical styled shapes produced different keys")
	}
	wide := mk(8, stroke.CapButt, stroke.JoinMiter)
	if slices.Equal(keyA, unstyledKey(t, &wide)) {
		t.Fatal("different stroke widths share a key")
	}
	joined := mk(4, stroke.CapButt, stroke.JoinBevel)
	if slices.Equal(keyA, unstyledKey(t, &joined)) {
		t.Fatal("different joins share a key on an open shape")
	}

	// The starPath is closed, so cap style cannot affect the output and is dropped from the key (kClosed_KeyFlag).
	capped := mk(4, stroke.CapRound, stroke.JoinMiter)
	if !slices.Equal(keyA, unstyledKey(t, &capped)) {
		t.Fatal("cap variation altered the key of a closed shape")
	}

	// A diagonal line has no joins, so join style is dropped (kNoJoins_KeyFlag) but caps count.
	mkLine := func(capType stroke.Cap, join stroke.Join) StyledShape {
		p := &path.Path{}
		p.MoveTo(3, 4).LineTo(23, 37)
		style := MakeStyle(strokeRecWith(4, capType, join, 4, false))
		s := MakeStyledShapePath(p, style, DoSimplifyYes)
		return s.ApplyStyle(StyleApplyPathEffectAndStrokeRec, 1)
	}
	l1 := mkLine(stroke.CapButt, stroke.JoinMiter)
	l2 := mkLine(stroke.CapButt, stroke.JoinRound)
	l3 := mkLine(stroke.CapSquare, stroke.JoinMiter)
	keyL := unstyledKey(t, &l1)
	if !slices.Equal(keyL, unstyledKey(t, &l2)) {
		t.Fatal("join variation altered the key of a line")
	}
	if slices.Equal(keyL, unstyledKey(t, &l3)) {
		t.Fatal("cap variation did not alter the key of a line")
	}
}

func TestStyledShapeSimplifyStrokeAndFillRect(t *testing.T) {
	rect := geom.RectLTRB(10, 10, 30, 30)

	// Mitered stroke+fill becomes a bigger filled rect.
	mitered := MakeStyledShapeRect(rect,
		MakeStyle(strokeRecWith(4, stroke.CapButt, stroke.JoinMiter, 4, true)), DoSimplifyYes)
	if !mitered.Style().IsSimpleFill() || !mitered.Shape().IsRect() {
		t.Fatalf("mitered stroke+fill rect did not reduce to a filled rect")
	}
	if got, want := mitered.Shape().Rect(), rect.Outset(2, 2); got != want {
		t.Fatalf("reduced rect = %v, want %v", got, want)
	}
	if !mitered.Simplified() {
		t.Fatal("reduction not flagged as simplified")
	}

	// Round-join stroke+fill becomes a filled round rect.
	round := MakeStyledShapeRect(rect,
		MakeStyle(strokeRecWith(4, stroke.CapButt, stroke.JoinRound, 4, true)), DoSimplifyYes)
	if !round.Style().IsSimpleFill() || !round.Shape().IsRRect() {
		t.Fatal("round stroke+fill rect did not reduce to a filled rrect")
	}
	rr := round.Shape().RRect()
	if rr.Rect != rect.Outset(2, 2) || rr.RadiusX != 2 || rr.RadiusY != 2 {
		t.Fatalf("reduced rrect = %+v", rr)
	}

	// Bevel joins need path rendering: no reduction.
	bevel := MakeStyledShapeRect(rect,
		MakeStyle(strokeRecWith(4, stroke.CapButt, stroke.JoinBevel, 4, true)), DoSimplifyYes)
	if bevel.Style().IsSimpleFill() {
		t.Fatal("bevel stroke+fill rect must not reduce")
	}
}

func TestStyledShapeSimplifyStrokedLine(t *testing.T) {
	mkLine := func(capType stroke.Cap) StyledShape {
		p := &path.Path{}
		p.MoveTo(2, 5).LineTo(12, 5)
		return MakeStyledShapePath(p,
			MakeStyle(strokeRecWith(4, capType, stroke.JoinMiter, 4, false)), DoSimplifyYes)
	}

	butt := mkLine(stroke.CapButt)
	if !butt.Style().IsSimpleFill() || !butt.Shape().IsRect() {
		t.Fatal("butt-cap horizontal line did not reduce to a rect")
	}
	if got, want := butt.Shape().Rect(), geom.RectLTRB(2, 3, 12, 7); got != want {
		t.Fatalf("butt-cap line rect = %v, want %v", got, want)
	}

	square := mkLine(stroke.CapSquare)
	if got, want := square.Shape().Rect(), geom.RectLTRB(0, 3, 14, 7); got != want {
		t.Fatalf("square-cap line rect = %v, want %v", got, want)
	}

	round := mkLine(stroke.CapRound)
	if !round.Style().IsSimpleFill() || !round.Shape().IsRRect() {
		t.Fatal("round-cap horizontal line did not reduce to a rrect")
	}
	rr := round.Shape().RRect()
	if rr.Rect != geom.RectLTRB(0, 3, 14, 7) || rr.RadiusX != 2 || rr.RadiusY != 2 {
		t.Fatalf("round-cap line rrect = %+v", rr)
	}

	// Points: butt caps draw nothing; round caps draw a disc.
	mkPoint := func(capType stroke.Cap) StyledShape {
		p := &path.Path{}
		p.MoveTo(5, 5).LineTo(5, 5)
		return MakeStyledShapePath(p,
			MakeStyle(strokeRecWith(4, capType, stroke.JoinMiter, 4, false)), DoSimplifyYes)
	}
	if empty := mkPoint(stroke.CapButt); !empty.IsEmpty() {
		t.Fatal("butt-cap point must reduce to empty")
	}
	dot := mkPoint(stroke.CapRound)
	if !dot.Shape().IsRRect() {
		t.Fatal("round-cap point must reduce to an oval")
	}
	if got := dot.Shape().RRect().Rect; got != geom.RectLTRB(3, 3, 7, 7) {
		t.Fatalf("round-cap point bounds = %v", got)
	}
}

func TestStyledShapeMakeFilled(t *testing.T) {
	rect := geom.RectLTRB(1, 1, 9, 9)
	stroked := MakeStyledShapeRect(rect,
		MakeStyle(strokeRecWith(4, stroke.CapButt, stroke.JoinMiter, 4, false)), DoSimplifyYes)
	if stroked.Style().IsSimpleFill() {
		t.Fatal("precondition: stroked rect keeps its style")
	}
	filled := MakeFilledStyledShape(&stroked, FillInversionPreserve)
	if !filled.Style().IsSimpleFill() {
		t.Fatal("MakeFilled must produce a simple fill")
	}
	if !filled.Shape().IsRect() || filled.Shape().Rect() != rect {
		t.Fatalf("MakeFilled changed the geometry: %v", filled.Shape().Rect())
	}
	if filled.InverseFilled() {
		t.Fatal("inversion not preserved")
	}
	flipped := MakeFilledStyledShape(&stroked, FillInversionForceInverted)
	if !flipped.InverseFilled() {
		t.Fatal("forced inversion not applied")
	}
}

func TestStyledShapeAsNestedRects(t *testing.T) {
	outer := geom.RectLTRB(0, 0, 20, 20)
	inner := geom.RectLTRB(5, 5, 15, 15)

	build := func(innerDir geom.PathDirection, fillType path.FillType) StyledShape {
		p := &path.Path{}
		p.AddRect(outer, geom.DirectionCW)
		p.AddRect(inner, innerDir)
		p.SetFillType(fillType)
		return MakeStyledShapePath(p, SimpleFillStyle(), DoSimplifyYes)
	}

	opp := build(geom.DirectionCCW, path.FillWinding)
	rects, ok := opp.AsNestedRects()
	if !ok {
		t.Fatal("opposite-winding nested rects not detected")
	}
	if rects[0] != outer || rects[1] != inner {
		t.Fatalf("nested rects = %v", rects)
	}

	sameWinding := build(geom.DirectionCW, path.FillWinding)
	if _, ok = sameWinding.AsNestedRects(); ok {
		t.Fatal("same-winding rects must fail under the winding fill rule")
	}
	evenOdd := build(geom.DirectionCW, path.FillEvenOdd)
	if _, ok = evenOdd.AsNestedRects(); !ok {
		t.Fatal("same-winding rects must pass under the even-odd fill rule")
	}
}
