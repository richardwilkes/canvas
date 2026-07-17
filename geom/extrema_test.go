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

func TestChopQuadAtYExtrema(t *testing.T) {
	// Non-monotonic in Y: chops into two Y-monotonic quads sharing a flattened extreme.
	src := []Point{{X: 0, Y: 0}, {X: 50, Y: 100}, {X: 100, Y: 0}}
	var dst [5]Point
	n := ChopQuadAtYExtrema(src, dst[:])
	if n != 1 {
		t.Fatalf("n = %d, want 1", n)
	}
	if dst[0] != src[0] || dst[4] != src[2] {
		t.Error("endpoints must be preserved")
	}
	// flattened: neighbors share the chop point's Y
	if dst[1].Y != dst[2].Y || dst[3].Y != dst[2].Y {
		t.Errorf("extreme not flattened: %v %v %v", dst[1].Y, dst[2].Y, dst[3].Y)
	}
	if dst[2].Y != 50 { // apex of this symmetric quad
		t.Errorf("apex Y = %v, want 50", dst[2].Y)
	}

	// Already monotonic: copied through, n == 0.
	src = []Point{{X: 0, Y: 0}, {X: 50, Y: 50}, {X: 100, Y: 100}}
	n = ChopQuadAtYExtrema(src, dst[:])
	if n != 0 {
		t.Fatalf("monotonic n = %d, want 0", n)
	}
	for i := 0; i < 3; i++ {
		if dst[i] != src[i] {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], src[i])
		}
	}
}

func TestChopCubicAtYExtrema(t *testing.T) {
	// A cubic with two interior Y extrema.
	src := []Point{{X: 0, Y: 0}, {X: 30, Y: 100}, {X: 70, Y: -100}, {X: 100, Y: 0}}
	var dst [10]Point
	n := ChopCubicAtYExtrema(src, dst[:])
	if n != 2 {
		t.Fatalf("n = %d, want 2", n)
	}
	if dst[0] != src[0] || dst[9] != src[3] {
		t.Error("endpoints must be preserved")
	}
	// Each chop point's neighbors share its Y (flattened), making each piece monotonic.
	if dst[2].Y != dst[3].Y || dst[4].Y != dst[3].Y {
		t.Error("first extreme not flattened")
	}
	if dst[5].Y != dst[6].Y || dst[7].Y != dst[6].Y {
		t.Error("second extreme not flattened")
	}
}

func TestConicTransformW(t *testing.T) {
	pts := []Point{{X: 0, Y: 0}, {X: 50, Y: 50}, {X: 100, Y: 0}}

	// Non-perspective matrices leave the weight alone.
	var m Matrix
	m.SetScaleTranslate(2, 3, 4, 5)
	if got := TransformW(pts, 0.75, &m); got != 0.75 {
		t.Errorf("non-persp TransformW = %v", got)
	}

	// Perspective: w' = sqrt(w1*w1/(w0*w2)) with wi the mapped homogeneous Z values.
	const p0, p1, p2 = 0.001, 0.002, 1
	m.SetAll(1, 0, 0, 0, 1, 0, p0, p1, p2)
	w := float32(0.75)
	w0 := float64(p0*pts[0].X + p1*pts[0].Y + p2)
	w1 := float64(p0*(pts[1].X*w) + p1*(pts[1].Y*w) + p2*w)
	w2 := float64(p0*pts[2].X + p1*pts[2].Y + p2)
	want := DoubleToScalar(math.Sqrt(w1 * w1 / (w0 * w2)))
	if got := TransformW(pts, w, &m); got != want {
		t.Errorf("perspective TransformW = %v, want %v", got, want)
	}
}
