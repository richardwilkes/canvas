// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// TestMaskAdditiveBlitterClearsReusedStorage verifies that init zeroes reused storage. The mask blitter accumulates
// coverage additively, so a smaller fill reusing a larger pooled fill's backing array must start from all-zero, else it
// would carry stale coverage. (make() zeroes for a fresh array; the reuse path must clear() explicitly.)
func TestMaskAdditiveBlitterClearsReusedStorage(t *testing.T) {
	var m maskAdditiveBlitter
	clip := geom.IRectLTRB(0, 0, 1000, 1000)

	// First fill establishes a larger backing array; dirty every byte to simulate accumulated coverage.
	m.init(nil, geom.IRectLTRB(2, 2, 22, 22), clip)
	firstCap := cap(m.storage)
	for i := range m.storage[:firstCap] {
		m.storage[i] = 0xAB
	}

	// A smaller fill reuses that array. init must reslice down and clear the bytes this fill will touch.
	m.init(nil, geom.IRectLTRB(5, 5, 10, 10), clip)
	if cap(m.storage) != firstCap {
		t.Fatalf("storage reallocated (cap %d -> %d); the smaller fill should reuse the larger backing array",
			firstCap, cap(m.storage))
	}
	for i, b := range m.storage {
		if b != 0 {
			t.Fatalf("reused storage byte %d = %#x, want 0 (stale coverage not cleared)", i, b)
		}
	}
}

// TestRunBasedAdditiveBlitterReusesRuns verifies that init reslices the runs arrays down for a narrower fill (reusing
// the wider backing storage rather than reallocating), re-establishes the RLE for the new width via advanceRuns, and
// grows the arrays for a wider-than-before fill.
func TestRunBasedAdditiveBlitterReusesRuns(t *testing.T) {
	var r runBasedAdditiveBlitter
	clip := geom.IRectLTRB(0, 0, 1000, 1000)

	// Wide fill establishes the runs backing arrays (width+1 runs, width+2 alphas).
	r.init(nil, geom.IRectLTRB(0, 0, 100, 10), clip, false)
	wideRunsCap, wideAlphaCap := cap(r.runs.Runs), cap(r.runs.Alpha)
	if len(r.runs.Runs) != 101 || len(r.runs.Alpha) != 102 {
		t.Fatalf("wide init: runs len %d alpha len %d, want 101/102", len(r.runs.Runs), len(r.runs.Alpha))
	}

	// Narrower fill reuses the same backing arrays (no realloc) and reslices down.
	r.init(nil, geom.IRectLTRB(0, 0, 20, 10), clip, false)
	if cap(r.runs.Runs) != wideRunsCap || cap(r.runs.Alpha) != wideAlphaCap {
		t.Fatal("narrow init reallocated the runs arrays instead of reusing the wider backing storage")
	}
	if len(r.runs.Runs) != 21 || len(r.runs.Alpha) != 22 {
		t.Fatalf("narrow init: runs len %d alpha len %d, want 21/22", len(r.runs.Runs), len(r.runs.Alpha))
	}
	// advanceRuns (called by init) must re-establish the RLE for the new width.
	if r.runs.Runs[0] != 20 || r.runs.Runs[20] != 0 || r.runs.Alpha[0] != 0 {
		t.Fatalf("narrow init: RLE not reset for width 20 (Runs[0]=%d Runs[20]=%d Alpha[0]=%d)",
			r.runs.Runs[0], r.runs.Runs[20], r.runs.Alpha[0])
	}

	// A wider-than-before fill must grow the arrays.
	r.init(nil, geom.IRectLTRB(0, 0, 500, 10), clip, false)
	if len(r.runs.Runs) != 501 || cap(r.runs.Runs) < 501 {
		t.Fatalf("grow init: runs len %d cap %d, want len 501 cap >=501", len(r.runs.Runs), cap(r.runs.Runs))
	}
}
