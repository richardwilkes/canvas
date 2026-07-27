// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit coverage of the lattice div validator and the degenerate-leading-patch case that depends on it. The accepted div
// range is [start, end], not (start, end]: a div sitting exactly on the bounds edge is a legal input that both
// latticeValid and newLatticeIter give special meaning to, so it is pinned here rather than left to the doc comment.

package canvas

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func TestValidDivsAcceptedRange(t *testing.T) {
	for _, one := range []struct {
		name  string
		divs  []int32
		start int32
		end   int32
		want  bool
	}{
		{name: "empty", divs: nil, start: 0, end: 8, want: true},
		{name: "div at start", divs: []int32{2}, start: 2, end: 8, want: true},
		{name: "div below start", divs: []int32{1}, start: 2, end: 8, want: false},
		{name: "div at end", divs: []int32{8}, start: 2, end: 8, want: true},
		{name: "div above end", divs: []int32{9}, start: 2, end: 8, want: false},
		{name: "spanning the whole range", divs: []int32{2, 8}, start: 2, end: 8, want: true},
		{name: "increasing", divs: []int32{3, 4, 7}, start: 2, end: 8, want: true},
		{name: "repeated", divs: []int32{4, 4}, start: 2, end: 8, want: false},
		{name: "repeated at start", divs: []int32{2, 2}, start: 2, end: 8, want: false},
		{name: "decreasing", divs: []int32{5, 4}, start: 2, end: 8, want: false},
	} {
		if got := validDivs(one.divs, one.start, one.end); got != one.want {
			t.Errorf("%s: validDivs(%v, %d, %d) = %v, want %v", one.name, one.divs, one.start, one.end, got, one.want)
		}
	}
}

// TestLatticeValidAcceptsDivAtBoundsEdge covers the two consumers that give a div on the bounds edge its meaning: it
// must survive validDivs, and newLatticeIter must then drop it and mark the leading patch scalable.
func TestLatticeValidAcceptsDivAtBoundsEdge(t *testing.T) {
	// A single x div sitting on the left edge makes the x axis degenerate, but the y axis still yields real patches, so
	// the lattice as a whole is usable and must not be rejected by validDivs.
	lattice := latticeSpec{xDivs: []int32{0}, yDivs: []int32{2, 6}, bounds: geom.IRectWH(8, 8)}
	if !latticeValid(8, 8, &lattice) {
		t.Errorf("a div equal to bounds.Left must be a legal input")
	}

	// Both axes degenerate: no real patch remains, so this one is rejected by the zeroX/zeroY test rather than by
	// validDivs.
	degenerate := latticeSpec{xDivs: []int32{0}, yDivs: []int32{0}, bounds: geom.IRectWH(8, 8)}
	if latticeValid(8, 8, &degenerate) {
		t.Errorf("a lattice with no real patch must be rejected")
	}

	// The leading div is consumed as the "first patch is scalable" flag, leaving 2 x patches and 3 y patches. With dst
	// twice the size of src, the 4 scalable x pixels absorb the whole 8 pixel difference.
	iter := newLatticeIter(&latticeSpec{
		xDivs:  []int32{0, 4},
		yDivs:  []int32{2, 6},
		bounds: geom.IRectWH(8, 8),
	}, geom.RectLTRB(0, 0, 16, 16))
	if iter.numRects != 6 {
		t.Errorf("numRects = %d, want 6", iter.numRects)
	}
	checkLatticePoints(t, "srcX", iter.srcX, []int32{0, 4, 8})
	checkLatticeDstPoints(t, "dstX", iter.dstX, []float32{0, 12, 16})
	checkLatticePoints(t, "srcY", iter.srcY, []int32{0, 2, 6, 8})
	checkLatticeDstPoints(t, "dstY", iter.dstY, []float32{0, 2, 14, 16})

	var src geom.IRect
	var dst geom.Rect
	var count int
	for iter.next(&src, &dst) {
		count++
	}
	if count != 6 {
		t.Errorf("iterated %d patches, want 6", count)
	}
}

func checkLatticePoints(t *testing.T, name string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s = %v, want %v", name, got, want)
		return
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%s = %v, want %v", name, got, want)
			return
		}
	}
}

func checkLatticeDstPoints(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s = %v, want %v", name, got, want)
		return
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%s = %v, want %v", name, got, want)
			return
		}
	}
}
