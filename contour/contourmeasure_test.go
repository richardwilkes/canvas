// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package contour

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

func nearly(t *testing.T, got, want, tol float32, what string) {
	t.Helper()
	if math.Abs(float64(got-want)) > float64(tol) {
		t.Errorf("%s: got %v, want %v (tol %v)", what, got, want, tol)
	}
}

func TestMeasureLine(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(30, 40) // 3-4-5 triangle, length 50

	m := NewIter(p, false, 1).Next()
	if m == nil {
		t.Fatal("expected a contour")
	}
	if m.Length() != 50 {
		t.Errorf("length: got %v, want 50", m.Length())
	}
	if m.IsClosed() {
		t.Error("open line should not be closed")
	}

	pos, tan, ok := m.GetPosTan(25, true, true)
	if !ok {
		t.Fatal("GetPosTan failed")
	}
	nearly(t, pos.X, 15, 1e-4, "pos.X")
	nearly(t, pos.Y, 20, 1e-4, "pos.Y")
	nearly(t, tan.X, 0.6, 1e-6, "tan.X")
	nearly(t, tan.Y, 0.8, 1e-6, "tan.Y")

	// distance pinning
	pos, _, _ = m.GetPosTan(-5, true, false)
	if pos != (geom.Point{}) {
		t.Errorf("distance < 0 should pin to start, got %v", pos)
	}
	pos, _, _ = m.GetPosTan(1000, true, false)
	nearly(t, pos.X, 30, 1e-4, "pinned end X")
	nearly(t, pos.Y, 40, 1e-4, "pinned end Y")

	// NaN distance
	if _, _, ok = m.GetPosTan(float32(math.NaN()), true, true); ok {
		t.Error("NaN distance should fail")
	}
}

func TestMeasureForceClosed(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(10, 0)
	p.LineTo(10, 10)

	open := NewIter(p, false, 1).Next()
	closed := NewIter(p, true, 1).Next()
	if open.Length() != 20 {
		t.Errorf("open length: got %v, want 20", open.Length())
	}
	if closed.Length() <= open.Length() {
		t.Errorf("forceClosed should add the closing segment: %v <= %v", closed.Length(), open.Length())
	}
	nearly(t, closed.Length(), 20+float32(math.Sqrt(200)), 1e-3, "closed length")
	if open.IsClosed() || !closed.IsClosed() {
		t.Error("isClosed mismatch")
	}
}

func TestMeasureCircleLength(t *testing.T) {
	p := &path.Path{}
	p.AddCircle(0, 0, 10, geom.DirectionCW)
	m := NewIter(p, false, 1).Next()
	if m == nil {
		t.Fatal("expected a contour")
	}
	if !m.IsClosed() {
		t.Error("circle should be closed")
	}
	// The flattened length approaches 2*pi*r from below; the 0.5 base tolerance at resScale 1 yields ~0.65% error at
	// r=10 (exact parity with Skia is the oracle probe's job).
	want := float32(2 * math.Pi * 10)
	if m.Length() > want || m.Length() < want*0.99 {
		t.Errorf("circle length: got %v, want ~%v", m.Length(), want)
	}
	// A point 1/4 of the way around a CW circle from (10, 0) should be near (0, 10) with tangent (-1, 0).
	pos, tan, ok := m.GetPosTan(m.Length()/4, true, true)
	if !ok {
		t.Fatal("GetPosTan failed")
	}
	nearly(t, pos.X, 0, 0.05, "quarter pos.X")
	nearly(t, pos.Y, 10, 0.05, "quarter pos.Y")
	nearly(t, tan.X, -1, 0.01, "quarter tan.X")
	nearly(t, geom.ScalarAbs(tan.Y), 0, 0.01, "quarter tan.Y")
}

func TestMeasureMultipleContoursAndEmpty(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(10, 0)
	p.MoveTo(100, 100) // degenerate lone move: no segments, skipped
	p.MoveTo(20, 0)
	p.LineTo(20, 30)

	it := NewIter(p, false, 1)
	first := it.Next()
	if first == nil || first.Length() != 10 {
		t.Fatalf("first contour: %+v", first)
	}
	second := it.Next()
	if second == nil || second.Length() != 30 {
		t.Fatalf("second contour should skip the empty one: %+v", second)
	}
	if it.Next() != nil {
		t.Error("expected exhaustion")
	}
}

func TestMeasureGetSegment(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(100, 0)
	m := NewIter(p, false, 1).Next()

	dst := &path.Path{}
	if !m.GetSegment(25, 75, dst, true) {
		t.Fatal("GetSegment failed")
	}
	want := &path.Path{}
	want.MoveTo(25, 0)
	want.LineTo(75, 0)
	if !path.Equal(dst, want) {
		t.Errorf("segment mismatch: got %d pts", dst.CountPoints())
	}

	// reversed / NaN inputs fail
	if m.GetSegment(75, 25, &path.Path{}, true) {
		t.Error("start > stop should fail")
	}
	// clamped inputs succeed
	dst.Reset()
	if !m.GetSegment(-10, 1000, dst, true) {
		t.Fatal("clamped GetSegment failed")
	}
	last, _ := dst.LastPt()
	if last != (geom.Point{X: 100}) {
		t.Errorf("clamped segment end: %v", last)
	}
}

func TestMeasureGetSegmentAcrossCurves(t *testing.T) {
	// A contour mixing line, quad, and cubic; extract a span crossing all three and verify the endpoints via GetPosTan.
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(50, 0)
	p.QuadTo(75, 0, 75, 25)
	p.CubicTo(75, 60, 40, 80, 0, 80)
	m := NewIter(p, false, 1).Next()

	total := m.Length()
	dst := &path.Path{}
	if !m.GetSegment(10, total-10, dst, true) {
		t.Fatal("GetSegment failed")
	}
	if dst.IsEmpty() {
		t.Fatal("empty extraction")
	}
	first := dst.Point(0)
	wantFirst, _, _ := m.GetPosTan(10, true, false)
	nearly(t, first.X, wantFirst.X, 1e-3, "segment start X")
	nearly(t, first.Y, wantFirst.Y, 1e-3, "segment start Y")
	last, _ := dst.LastPt()
	wantLast, _, _ := m.GetPosTan(total-10, true, false)
	nearly(t, last.X, wantLast.X, 1e-3, "segment end X")
	nearly(t, last.Y, wantLast.Y, 1e-3, "segment end Y")
}

func TestMeasureZeroLengthSegTo(t *testing.T) {
	// A zero-length extraction into a non-empty dst must add a zero-length line (the dash zero-length on-interval
	// rule).
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(100, 0)
	m := NewIter(p, false, 1).Next()

	dst := &path.Path{}
	dst.MoveTo(-5, -5)
	if !m.GetSegment(50, 50, dst, false) {
		t.Fatal("zero-length GetSegment failed")
	}
	if dst.CountVerbs() != 2 || dst.CountPoints() != 2 {
		t.Errorf("expected an appended zero-length line, verbs=%d pts=%d", dst.CountVerbs(), dst.CountPoints())
	}
}

func TestMeasureGetMatrix(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(0, 100) // straight up: tangent (0, 1)
	m := NewIter(p, false, 1).Next()

	var mat geom.Matrix
	if !m.GetMatrix(50, &mat, GetPosAndTanMatrixFlag) {
		t.Fatal("GetMatrix failed")
	}
	// setSinCos(1, 0) rotated 90 degrees, then translated to (0, 50): (1, 0) maps to (0, 51).
	got := mat.MapPoint(geom.Point{X: 1, Y: 0})
	nearly(t, got.X, 0, 1e-5, "mapped X")
	nearly(t, got.Y, 51, 1e-5, "mapped Y")

	// position-only
	if !m.GetMatrix(50, &mat, GetPositionMatrixFlag) {
		t.Fatal("GetMatrix failed")
	}
	got = mat.MapPoint(geom.Point{X: 1, Y: 0})
	nearly(t, got.X, 1, 1e-5, "translate-only X")
	nearly(t, got.Y, 50, 1e-5, "translate-only Y")
}

func TestMeasureEmptyQueries(t *testing.T) {
	// A zero-valued Measure has no segments; every query must report failure rather than panic on segs[-1].
	var m Measure
	if m.Length() != 0 {
		t.Errorf("zero measure length: got %v, want 0", m.Length())
	}
	for _, distance := range []float32{-1, 0, 1, float32(math.Inf(1)), float32(math.NaN())} {
		if _, _, ok := m.GetPosTan(distance, true, true); ok {
			t.Errorf("GetPosTan(%v) on an empty measure reported success", distance)
		}
		sentinel := geom.TranslateMatrix(7, 7) // an unsuccessful GetMatrix must not write through
		mat := sentinel
		if m.GetMatrix(distance, &mat, GetPosAndTanMatrixFlag) {
			t.Errorf("GetMatrix(%v) on an empty measure reported success", distance)
		}
		if mat != sentinel {
			t.Errorf("GetMatrix(%v) on an empty measure overwrote the matrix", distance)
		}
		dst := &path.Path{}
		if m.GetSegment(distance, distance+10, dst, true) {
			t.Errorf("GetSegment(%v) on an empty measure reported success", distance)
		}
		if !dst.IsEmpty() {
			t.Errorf("GetSegment(%v) on an empty measure appended geometry", distance)
		}
	}
}

func TestIterResetToUnusablePath(t *testing.T) {
	// Resetting onto a nil or non-finite path must leave the iterator exhausted rather than measuring the old path.
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(30, 40)
	it := NewIter(p, false, 1)

	it.Reset(nil, false, 1)
	if m := it.Next(); m != nil {
		t.Errorf("nil path produced a contour of length %v", m.Length())
	}

	bad := &path.Path{}
	bad.MoveTo(0, 0)
	bad.LineTo(float32(math.Inf(1)), 0)
	it.Reset(bad, false, 1)
	if m := it.Next(); m != nil {
		t.Errorf("non-finite path produced a contour of length %v", m.Length())
	}

	// ...and resetting back onto a usable path measures it again.
	it.Reset(p, false, 1)
	m := it.Next()
	if m == nil {
		t.Fatal("expected a contour after resetting onto a usable path")
	}
	if m.Length() != 50 {
		t.Errorf("length after reset: got %v, want 50", m.Length())
	}
}

func TestPathMeasureWrapper(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(10, 0)
	p.MoveTo(0, 10)
	p.LineTo(0, 40)
	p.Close()

	m := NewPathMeasure(p, false, 1)
	if m.Length() != 10 {
		t.Errorf("first contour length: %v", m.Length())
	}
	if m.IsClosed() {
		t.Error("first contour should be open")
	}
	if !m.NextContour() {
		t.Fatal("expected a second contour")
	}
	if m.Length() != 60 || !m.IsClosed() {
		t.Errorf("second contour: length %v closed %v", m.Length(), m.IsClosed())
	}
	if m.NextContour() {
		t.Error("expected exhaustion")
	}
	if m.Length() != 0 {
		t.Error("exhausted measure should have length 0")
	}

	// non-finite path -> no contours
	bad := &path.Path{}
	bad.MoveTo(0, 0)
	bad.LineTo(float32(math.Inf(1)), 0)
	if NewPathMeasure(bad, false, 1).Length() != 0 {
		t.Error("non-finite path should measure empty")
	}
}
