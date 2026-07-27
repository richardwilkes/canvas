// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package vp8enc

import "testing"

// TestLoopFilterIsAlwaysNormal pins the invariant that lets applyLoopFilter port §15.3 alone: setupFilterStrength
// selects the normal filter at every quality, and putFilterHeader tells the decoder the same thing. If a simple-filter
// lane is ever added, this fails and the §15.2 port has to come back with it.
func TestLoopFilterIsAlwaysNormal(t *testing.T) {
	for quality := 0; quality <= 100; quality += 5 {
		e := newEncoder(32, 32)
		e.setSegmentParams(float32(quality))
		if e.filterSimple {
			t.Fatalf("quality %d: filterSimple is set, but only the normal filter is ported", quality)
		}
		var bw bitWriter
		bw.init()
		e.putFilterHeader(&bw)
		dec := newBoolDecoder(bw.finish())
		if dec.readBool(128) {
			t.Fatalf("quality %d: frame header advertises the simple filter", quality)
		}
		if got := int(dec.readUint(6)); got != e.filterLevel {
			t.Fatalf("quality %d: header filter level %d, want %d", quality, got, e.filterLevel)
		}
		if got := int(dec.readUint(3)); got != e.filterSharpness {
			t.Fatalf("quality %d: header sharpness %d, want %d", quality, got, e.filterSharpness)
		}
	}
}

// TestApplyLoopFilterDeblocksMacroblockEdge drives applyLoopFilter over a two-macroblock luma plane holding a small
// step at the shared vertical edge and checks the six-pixel normal-filter result (RFC 6386 §15.3) against values
// computed by hand: level 63 / sharpness 0 gives level2 189, ilevel 63, hlevel 2, so the edge passes the threshold and
// interior checks, is not high-variance, and gets a = 20, a1 = 4, a2 = 3, a3 = 1.
func TestApplyLoopFilterDeblocksMacroblockEdge(t *testing.T) {
	const (
		left  = 100
		right = 110
	)
	e := newEncoder(32, 16) // 2x1 macroblocks
	for y := range 16 {
		for x := range 32 {
			v := uint8(left)
			if x >= 16 {
				v = right
			}
			e.reconY[y*e.reconYStride+x] = v
		}
	}
	e.filterLevel = 63
	e.filterSharpness = 0
	e.applyLoopFilter()

	want := make([]uint8, 32)
	for x := range want {
		if x < 16 {
			want[x] = left
		} else {
			want[x] = right
		}
	}
	want[13] = left + 1  // p2 += a3
	want[14] = left + 3  // p1 += a2
	want[15] = left + 4  // p0 += a1
	want[16] = right - 4 // q0 -= a1
	want[17] = right - 3 // q1 -= a2
	want[18] = right - 1 // q2 -= a3
	for y := range 16 {
		row := e.reconY[y*e.reconYStride:][:32]
		for x := range row {
			if row[x] != want[x] {
				t.Fatalf("row %d: got %v, want %v", y, row, want)
			}
		}
	}

	// A level of 0 disables the filter entirely, leaving the step intact.
	e2 := newEncoder(32, 16)
	copy(e2.reconY, e.reconY)
	before := append([]uint8(nil), e2.reconY...)
	e2.filterLevel = 0
	e2.applyLoopFilter()
	for i := range before {
		if e2.reconY[i] != before[i] {
			t.Fatalf("filter level 0 modified the plane at %d", i)
		}
	}
}
