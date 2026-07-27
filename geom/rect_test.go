// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import (
	"math"
	"testing"
)

func TestRectEmpty(t *testing.T) {
	nan := float32(math.NaN())
	cases := []struct {
		r    Rect
		want bool
	}{
		{r: RectLTRB(0, 0, 10, 10), want: false},
		{r: RectLTRB(10, 10, 10, 10), want: true},
		{r: RectLTRB(10, 0, 0, 10), want: true},   // unsorted
		{r: RectLTRB(nan, 0, 10, 10), want: true}, // NaN edges are empty
		{r: RectLTRB(0, 0, 10, nan), want: true},
	}
	for i, c := range cases {
		if got := c.r.IsEmpty(); got != c.want {
			t.Errorf("case %d: IsEmpty = %v, want %v", i, got, c.want)
		}
	}
}

func TestRectIntersect(t *testing.T) {
	r := RectLTRB(0, 0, 10, 10)
	if !r.Intersect(RectLTRB(5, 5, 20, 20)) {
		t.Fatal("expected intersection")
	}
	if r != RectLTRB(5, 5, 10, 10) {
		t.Errorf("intersection = %+v", r)
	}
	// Failure leaves the rect unchanged.
	r2 := RectLTRB(0, 0, 10, 10)
	if r2.Intersect(RectLTRB(20, 20, 30, 30)) {
		t.Fatal("expected no intersection")
	}
	if r2 != RectLTRB(0, 0, 10, 10) {
		t.Errorf("failed intersect modified rect: %+v", r2)
	}
	// Empty operand always fails.
	if r2.Intersect(Rect{}) {
		t.Error("intersect with empty should fail")
	}
	// NaN semantics follow the max/min operand ordering used by Intersect: the *other* rect's edges are passed first,
	// so its NaN survives max/min (a false comparison returns the first operand) and fails the L<R check, while NaN in
	// the receiver's edges is silently discarded and the intersection degenerates to the other rect's edge. Verified by
	// the oracle probe TestRectIntersectProbe (an earlier version of this test had the direction backwards).
	nan := float32(math.NaN())
	nr := RectLTRB(nan, 0, 10, 10)
	if !nr.Intersect(RectLTRB(0, 0, 10, 10)) || nr != RectLTRB(0, 0, 10, 10) {
		t.Errorf("intersect with NaN in receiver should degrade to other rect's edge, got %+v", nr)
	}
	r3 := RectLTRB(0, 0, 10, 10)
	if r3.Intersect(RectLTRB(nan, 0, 10, 10)) {
		t.Error("intersect with NaN in other should fail")
	}
	// Touching edges do not intersect (strict <).
	if r2.Intersect(RectLTRB(10, 0, 20, 10)) {
		t.Error("edge-touching rects should not intersect")
	}
}

func TestRectJoin(t *testing.T) {
	r := RectLTRB(0, 0, 10, 10)
	r.Join(RectLTRB(20, 20, 30, 30))
	if r != RectLTRB(0, 0, 30, 30) {
		t.Errorf("join = %+v", r)
	}
	// Joining an empty rect is a no-op.
	r.Join(Rect{})
	if r != RectLTRB(0, 0, 30, 30) {
		t.Errorf("join with empty modified rect: %+v", r)
	}
	// Joining onto an empty rect adopts the operand.
	var e Rect
	e.Join(RectLTRB(1, 2, 3, 4))
	if e != RectLTRB(1, 2, 3, 4) {
		t.Errorf("join onto empty = %+v", e)
	}
}

func TestRectRounding(t *testing.T) {
	r := RectLTRB(0.5, 0.4, 10.5, 10.6)
	if got := r.Round(); got != IRectLTRB(1, 0, 11, 11) {
		t.Errorf("Round = %+v", got)
	}
	if got := r.RoundOut(); got != IRectLTRB(0, 0, 11, 11) {
		t.Errorf("RoundOut = %+v", got)
	}
	if got := r.RoundIn(); got != IRectLTRB(1, 1, 10, 10) {
		t.Errorf("RoundIn = %+v", got)
	}
	if got := r.RoundOutRect(); got != RectLTRB(0, 0, 11, 11) {
		t.Errorf("RoundOutRect = %+v", got)
	}
}

func TestRectSetBounds(t *testing.T) {
	var r Rect
	ok := r.SetBounds([]Point{{X: 3, Y: 4}, {X: -1, Y: 2}, {X: 5, Y: -7}})
	if !ok || r != RectLTRB(-1, -7, 5, 4) {
		t.Errorf("SetBounds = %+v ok=%v", r, ok)
	}
	// Non-finite input collapses to empty and returns false.
	ok = r.SetBounds([]Point{{X: 1, Y: 1}, {X: float32(math.Inf(1)), Y: 0}})
	if ok || r != (Rect{}) {
		t.Errorf("SetBounds with inf = %+v ok=%v", r, ok)
	}
	// NaN is a failure too.
	ok = r.SetBounds([]Point{{X: 1, Y: 1}, {X: 2, Y: float32(math.NaN())}})
	if ok || r != (Rect{}) {
		t.Errorf("SetBounds with NaN = %+v ok=%v", r, ok)
	}
	// An empty point list is not a failure: it succeeds with the empty rect, replacing whatever r held.
	r = RectLTRB(1, 2, 3, 4)
	ok = r.SetBounds(nil)
	if !ok || r != (Rect{}) {
		t.Errorf("SetBounds(nil) = %+v ok=%v", r, ok)
	}
	r = RectLTRB(1, 2, 3, 4)
	ok = r.SetBounds([]Point{})
	if !ok || r != (Rect{}) {
		t.Errorf("SetBounds(empty) = %+v ok=%v", r, ok)
	}
}

func TestRectContains(t *testing.T) {
	r := RectLTRB(0, 0, 10, 10)
	if !r.ContainsPoint(0, 0) || r.ContainsPoint(10, 5) || r.ContainsPoint(5, 10) {
		t.Error("half-open contains semantics violated")
	}
	if !r.ContainsRect(RectLTRB(1, 1, 9, 9)) || r.ContainsRect(RectLTRB(1, 1, 11, 9)) {
		t.Error("rect contains wrong")
	}
	if r.ContainsRect(Rect{}) {
		t.Error("containing an empty rect should be false")
	}
}

func TestIRectEmptyOverflow(t *testing.T) {
	r := IRectLTRB(math.MinInt32, 0, math.MaxInt32, 1)
	// Width overflows int32, so the rect must report empty even though right > left.
	if !r.IsEmpty() {
		t.Error("overflow-width rect should be empty")
	}
	if r.isEmpty64() {
		t.Error("isEmpty64 should be false (right > left)")
	}
}

func TestIRectIntersectJoin(t *testing.T) {
	r := IRectLTRB(0, 0, 10, 10)
	if !r.Intersect(IRectLTRB(5, 5, 20, 20)) || r != IRectLTRB(5, 5, 10, 10) {
		t.Errorf("intersect = %+v", r)
	}
	r2 := IRectLTRB(0, 0, 10, 10)
	if r2.Intersect(IRectLTRB(10, 0, 20, 10)) {
		t.Error("edge-touching should not intersect")
	}
	r2.Join(IRectLTRB(-5, -5, 3, 3))
	if r2 != IRectLTRB(-5, -5, 10, 10) {
		t.Errorf("join = %+v", r2)
	}
}

func TestRRect(t *testing.T) {
	rr := MakeRRect(RectLTRB(0, 0, 100, 50), 10, 10)
	if rr.Type != RRectSimple || rr.RadiusX != 10 || rr.RadiusY != 10 {
		t.Errorf("simple rrect = %+v", rr)
	}
	// Oversized radii scale down proportionally: rx=100 ry=50 on a 100x50 rect scales by 0.5.
	rr = MakeRRect(RectLTRB(0, 0, 100, 50), 100, 50)
	if rr.Type != RRectOval || rr.RadiusX != 50 || rr.RadiusY != 25 {
		t.Errorf("oval rrect = %+v", rr)
	}
	// Zero radii yield a rect type.
	rr = MakeRRect(RectLTRB(0, 0, 100, 50), 0, 10)
	if rr.Type != RRectRect || rr.RadiusX != 0 || rr.RadiusY != 0 {
		t.Errorf("rect rrect = %+v", rr)
	}
	// Empty rect yields empty.
	rr = MakeRRect(RectLTRB(5, 5, 5, 5), 10, 10)
	if rr.Type != RRectEmpty {
		t.Errorf("empty rrect = %+v", rr)
	}
	// Unsorted input is sorted first.
	rr = MakeRRect(RectLTRB(100, 50, 0, 0), 10, 10)
	if rr.Type != RRectSimple || rr.Rect != RectLTRB(0, 0, 100, 50) {
		t.Errorf("sorted rrect = %+v", rr)
	}
}
