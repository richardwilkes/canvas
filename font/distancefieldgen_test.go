// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for the distance-field generator: the byte packing formula, and structural properties of the
// generated field (threshold at the coverage edge, monotonicity away from the shape, symmetry) that the 8SSEDT
// transform must satisfy.

package font

import "testing"

func TestPackDistanceFieldVal(t *testing.T) {
	// byte = round((clamp(-dist, -4, 4*127/128) + 4) / 8 * 256).
	cases := []struct {
		dist float32
		want uint8
	}{
		{dist: 0, want: 128},        // zero distance is the 128 threshold
		{dist: 4, want: 0},          // far outside clamps to 0
		{dist: 100, want: 0},        // beyond the magnitude clamps too
		{dist: -4, want: 255},       // deep inside clamps to 4*127/128 → 255
		{dist: -100, want: 255},     //
		{dist: 1, want: 96},         // one texel outside: (-1+4)/8*256 = 96
		{dist: -1, want: 160},       // one texel inside: (1+4)/8*256 = 160
		{dist: 0.5, want: 112},      // half a texel outside
		{dist: -3.96875, want: 255}, // exactly the positive clamp value
	}
	for _, c := range cases {
		if got := packDistanceFieldVal(c.dist); got != c.want {
			t.Errorf("packDistanceFieldVal(%v) = %d, want %d", c.dist, got, c.want)
		}
	}
}

// solidSquareField generates the distance field of a size x size solid-255 A8 block.
func solidSquareField(size int) (field []uint8, fieldWidth int) {
	img := make([]uint8, size*size)
	for i := range img {
		img[i] = 255
	}
	dfW := size + 2*DistanceFieldPad
	df := make([]uint8, ComputeDistanceFieldSize(size, size))
	GenerateDistanceFieldFromA8Image(df, img, size, size, size)
	return df, dfW
}

func TestGenerateDistanceFieldSolidSquare(t *testing.T) {
	const size = 8
	df, dfW := solidSquareField(size)

	// The field dimensions include the DistanceFieldPad ring on each side.
	if len(df) != dfW*dfW {
		t.Fatalf("field size = %d, want %d", len(df), dfW*dfW)
	}

	// Interior texels are inside (> 128); the pad ring corners are far outside (< 128, and the outermost corner is at
	// least sqrt(2)*pad away — clamped to 0).
	center := df[(dfW/2)*dfW+dfW/2]
	if center <= 128 {
		t.Errorf("center = %d, want > 128 (inside)", center)
	}
	if corner := df[0]; corner != 0 {
		t.Errorf("pad corner = %d, want 0 (clamped far outside)", corner)
	}

	// Monotonic non-increasing along the center row moving outward from the center.
	row := (dfW / 2) * dfW
	for x := dfW / 2; x > 0; x-- {
		if df[row+x-1] > df[row+x] {
			t.Fatalf("row not monotonic at x=%d: %d > %d", x-1, df[row+x-1], df[row+x])
		}
	}

	// The field has the dihedral symmetry of the square within the sampled (DistanceFieldInset) region. The outermost
	// two columns are excluded: the backward pass's scan window covers x in [-1, W-3] of each row, so the last two
	// columns are under-propagated — outside the quad the SDFT vertex filler ever samples (it insets the glyph rect by
	// 2).
	at := func(x, y int) uint8 { return df[y*dfW+x] }
	for y := DistanceFieldInset; y < dfW-DistanceFieldInset; y++ {
		for x := DistanceFieldInset; x < dfW-DistanceFieldInset; x++ {
			if at(x, y) != at(dfW-1-x, y) {
				t.Fatalf("x-mirror asymmetry at (%d,%d): %d vs %d", x, y, at(x, y), at(dfW-1-x, y))
			}
			if at(x, y) != at(y, x) {
				t.Fatalf("transpose asymmetry at (%d,%d): %d vs %d", x, y, at(x, y), at(y, x))
			}
		}
	}

	// The coverage edge sits between the last solid texel and the first pad texel: crossing the image boundary crosses
	// the 128 threshold within one texel.
	edgeInside := at(DistanceFieldPad, dfW/2)
	edgeOutside := at(DistanceFieldPad-1, dfW/2)
	if edgeInside < 128 || edgeOutside > 128 {
		t.Errorf("threshold not at the boundary: inside=%d outside=%d", edgeInside, edgeOutside)
	}
}

func TestGenerateDistanceFieldAntiAliasedEdgeShifts(t *testing.T) {
	// A soft (partial coverage) edge must move the zero crossing relative to a hard edge: a column of 64-coverage reads
	// as an edge closer to the solid side than a column of 192.
	const size = 6
	makeField := func(edge uint8) []uint8 {
		img := make([]uint8, size*size)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				switch {
				case x < size/2:
					img[y*size+x] = 255
				case x == size/2:
					img[y*size+x] = edge
				default:
					img[y*size+x] = 0
				}
			}
		}
		df := make([]uint8, ComputeDistanceFieldSize(size, size))
		GenerateDistanceFieldFromA8Image(df, img, size, size, size)
		return df
	}
	dfW := size + 2*DistanceFieldPad
	low := makeField(64)
	high := makeField(192)
	y := dfW / 2
	x := DistanceFieldPad + size/2 // the partial-coverage column
	if low[y*dfW+x] >= high[y*dfW+x] {
		t.Errorf("soft-edge ordering wrong: cov64=%d !< cov192=%d",
			low[y*dfW+x], high[y*dfW+x])
	}
}
