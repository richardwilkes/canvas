// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pointer-stable bump allocation for the edge builders (pooled temporaries). A naive edge builder does one heap
// allocation per segment, which dominates per-fill heap traffic on curve/stroke fills (a stroked path expands into
// hundreds of edges). edgeArena uses a bump-allocation model instead: edges come from fixed-size blocks that are never
// resized once created, so a *T handed out by alloc stays valid as the arena grows, and reset rewinds it for reuse by a
// pooled builder (blocks are retained, so steady-state fills reallocate nothing).

package raster

// edgeArenaBlock is the number of edges allocated per arena block. Blocks are never individually resized, so pointers
// taken into them via alloc remain valid as the arena grows. 64 keeps a typical filled path within a single block while
// bounding the wasted tail of the last block.
const edgeArenaBlock = 64

// edgeArena is a pointer-stable bump allocator for a single edge struct type T. It is not safe for concurrent use: each
// fill (and each band under FillPathParallel) uses its own builder, so its arenas are single-goroutine.
type edgeArena[T any] struct {
	blocks [][]T
	blk    int // index of the current (partially filled) block
	off    int // next free slot within the current block
}

// alloc returns a pointer to a zeroed T, matching the semantics of &T{}. The element is zeroed on every allocation
// because a reset arena hands back blocks whose slots still hold stale edges.
func (a *edgeArena[T]) alloc() *T {
	if a.blk == len(a.blocks) {
		a.blocks = append(a.blocks, make([]T, edgeArenaBlock))
	}
	block := a.blocks[a.blk]
	p := &block[a.off]
	var zero T
	*p = zero
	a.off++
	if a.off == len(block) {
		a.blk++
		a.off = 0
	}
	return p
}

// reset rewinds the arena so its blocks are reused by the next build, without freeing them.
func (a *edgeArena[T]) reset() {
	a.blk = 0
	a.off = 0
}
