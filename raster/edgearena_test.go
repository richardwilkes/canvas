// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import "testing"

// TestEdgeArenaPointerStability verifies that pointers handed out by alloc stay valid as the arena grows past a block
// boundary — the invariant the edge builders rely on (the edge list holds *Edge pointers into the arena while more
// edges are still being allocated).
func TestEdgeArenaPointerStability(t *testing.T) {
	var a edgeArena[AnalyticEdge]
	const n = edgeArenaBlock*3 + 7 // spans several blocks, with a partial last block
	ptrs := make([]*AnalyticEdge, n)
	for i := 0; i < n; i++ {
		p := a.alloc()
		p.X = Fixed(i + 1) // marker written before later allocations grow the arena
		ptrs[i] = p
	}
	seen := make(map[*AnalyticEdge]bool, n)
	for i := 0; i < n; i++ {
		if got := int(ptrs[i].X); got != i+1 {
			t.Fatalf("edge %d: X = %d, want %d (pointer moved when the arena grew)", i, got, i+1)
		}
		if seen[ptrs[i]] {
			t.Fatalf("edge %d aliases an earlier allocation", i)
		}
		seen[ptrs[i]] = true
	}
}

// TestEdgeArenaResetReusesAndZeroes verifies that reset rewinds the arena so a pooled builder reuses the same backing
// storage (no reallocation) and that alloc hands back a zeroed element despite the stale contents left by the previous
// build — matching the &T{} semantics the builders replaced.
func TestEdgeArenaResetReusesAndZeroes(t *testing.T) {
	var a edgeArena[AnalyticEdge]
	const n = edgeArenaBlock + 5 // crosses one block boundary
	first := make([]*AnalyticEdge, n)
	for i := 0; i < n; i++ {
		p := a.alloc()
		p.X = 12345
		p.Winding = 1
		first[i] = p
	}
	a.reset()
	for i := 0; i < n; i++ {
		p := a.alloc()
		if p != first[i] {
			t.Fatalf("alloc %d after reset: got %p, want reused %p (blocks not retained/reused)", i, p, first[i])
		}
		if p.X != 0 || p.Winding != 0 {
			t.Fatalf("alloc %d after reset: not zeroed (X=%d Winding=%d)", i, p.X, p.Winding)
		}
	}
}
