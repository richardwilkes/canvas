// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Chunks of vertex (or instance) data written out when the final vertex count is not known up front, plus the builder
// that allocates them from the flush-time vertex pool. Each chunk's data is a []byte window into the pool buffer; the
// builder must have close() called explicitly when writing is complete.

package gl

// vertexChunk is a chunk of vertex data with its buffer and base vertex (or base instance, depending on the use case).
type vertexChunk struct {
	buffer AnyBuffer
	count  int
	base   int
}

// vertexChunkBuilder allocates chunks of vertex data from the flush-time vertex pool as needed. The provided target
// must not be used externally throughout the entire lifetime of this object, and close() must be called when writing is
// complete.
type vertexChunkBuilder struct {
	target                  *OpFlushState
	chunks                  *[]vertexChunk
	currChunkVertexData     []byte
	stride                  uint64
	minVerticesPerChunk     int
	currChunkVertexCount    int
	currChunkVertexCapacity int
	// currChunkOpen reports whether the last entry in *chunks is the chunk being written, i.e. whether close() still
	// has a count to finalize. A failed allocChunk clears it: the previous chunk's count is already final at that
	// point, and overwriting it with the reset counter would discard geometry that is already in the GPU buffer.
	currChunkOpen bool
}

func newVertexChunkBuilder(target *OpFlushState, chunks *[]vertexChunk, stride uint64, minVerticesPerChunk int) *vertexChunkBuilder {
	if minVerticesPerChunk <= 0 {
		panic("minVerticesPerChunk must be positive")
	}
	return &vertexChunkBuilder{
		target:              target,
		chunks:              chunks,
		stride:              stride,
		minVerticesPerChunk: minVerticesPerChunk,
	}
}

// close returns the unused reserve to the vertex pool and finalizes the current chunk's count. It must be called once
// writing is complete.
func (b *vertexChunkBuilder) close() {
	if b.currChunkOpen {
		b.target.PutBackVertices(b.currChunkVertexCapacity-b.currChunkVertexCount, b.stride)
		(*b.chunks)[len(*b.chunks)-1].count = b.currChunkVertexCount
	}
}

// appendVertices appends count contiguous vertices, which are not guaranteed to be contiguous with previous or future
// calls. Returns nil on allocation failure.
func (b *vertexChunkBuilder) appendVertices(count int) []byte {
	if count <= 0 {
		panic("count must be positive")
	}
	if b.currChunkVertexCount+count > b.currChunkVertexCapacity && !b.allocChunk(count) {
		return nil
	}
	off := uint64(b.currChunkVertexCount) * b.stride
	b.currChunkVertexCount += count
	return b.currChunkVertexData[off : off+uint64(count)*b.stride]
}

// allocChunk finalizes the current chunk (if any) and allocates a new one from the vertex pool with room for at least
// minCount vertices.
func (b *vertexChunkBuilder) allocChunk(minCount int) bool {
	if b.currChunkOpen {
		// No need to put back vertices; the buffer is full.
		(*b.chunks)[len(*b.chunks)-1].count = b.currChunkVertexCount
	}
	b.currChunkVertexCount = 0
	minAllocCount := max(minCount, b.minVerticesPerChunk)
	data, buffer, base, capacity := b.target.MakeVertexSpaceAtLeast(b.stride, minAllocCount,
		minAllocCount)
	if data == nil || buffer == nil || capacity < minCount {
		b.currChunkVertexData = nil
		b.currChunkVertexCapacity = 0
		// The chunk just finalized above (if any) keeps its count; there is no new chunk to finalize at close().
		b.currChunkOpen = false
		return false
	}
	*b.chunks = append(*b.chunks, vertexChunk{buffer: buffer, base: base})
	b.currChunkVertexData = data
	b.currChunkVertexCapacity = capacity
	b.currChunkOpen = true
	b.minVerticesPerChunk *= 2
	return true
}
