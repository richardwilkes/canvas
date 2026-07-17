// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// CpuBuffer and BufferAllocPool with its vertex/index specializations: the transient geometry rings that op recording
// writes vertices/indices into. A pool hands out sub-ranges of large dynamic buffers, staging through CPU memory when
// mapping is unavailable or not worthwhile, and recycles everything at reset. Only the vertex and index pools exist;
// there is no draw-indirect pool.
//
// Ref/lifetime notes: buffers returned by MakeSpace are owned by the pool's blocks (an additional ref is unneeded,
// since pool buffers are only consumed by ops between prepare and execute, strictly inside the pool's validity window).
// The returned byte windows remain writable until the next MakeSpace, Unmap, or Reset.

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/gpu"
)

// DefaultBufferSize is the default size for a fresh geometry buffer block.
const DefaultBufferSize = 1 << 15

// AnyBuffer is the common interface of GPU buffer objects (Buffer) and CPU-side arrays (CpuBuffer) that pooled geometry
// can land in.
type AnyBuffer interface {
	Size() uint64
	IsCpuBuffer() bool
}

// CpuBuffer is a ref-counted client-side array. The ref count is only consulted by CpuBufferCache to decide whether a
// cached buffer is free for reuse.
type CpuBuffer struct {
	data   []byte
	refCnt int32
}

// MakeCpuBuffer creates a ref-counted CPU-side buffer of the given size.
func MakeCpuBuffer(size uint64) *CpuBuffer {
	if size == 0 {
		panic("CpuBuffer size must be > 0")
	}
	return &CpuBuffer{data: make([]byte, size), refCnt: 1}
}

// Ref increments the reference count.
func (b *CpuBuffer) Ref() { b.refCnt++ }

// Unref decrements the reference count (the Go object is GC-reclaimed; the count only tracks use).
func (b *CpuBuffer) Unref() {
	b.refCnt--
	if b.refCnt < 0 {
		panic("CpuBuffer over-unref")
	}
}

// Unique reports whether this buffer has exactly one outstanding reference.
func (b *CpuBuffer) Unique() bool { return b.refCnt == 1 }

// Size implements AnyBuffer.
func (b *CpuBuffer) Size() uint64 { return uint64(len(b.data)) }

// IsCpuBuffer implements AnyBuffer.
func (b *CpuBuffer) IsCpuBuffer() bool { return true }

// Data returns the buffer's backing byte slice.
func (b *CpuBuffer) Data() []byte { return b.data }

// CpuBufferCache is a cache of default-size CPU buffers shared by multiple pools to avoid reallocation.
type CpuBufferCache struct {
	buffers []struct {
		buffer  *CpuBuffer
		cleared bool
	}
}

// MakeCpuBufferCache creates a cache that holds up to maxBuffersToCache default-size buffers.
func MakeCpuBufferCache(maxBuffersToCache int) *CpuBufferCache {
	c := &CpuBufferCache{}
	if maxBuffersToCache > 0 {
		c.buffers = make([]struct {
			buffer  *CpuBuffer
			cleared bool
		}, maxBuffersToCache)
	}
	return c
}

// MakeBuffer hands out a cached default-size buffer that is no longer in use, or makes a fresh one.
func (c *CpuBufferCache) MakeBuffer(size uint64, mustBeInitialized bool) *CpuBuffer {
	if size == 0 {
		panic("CpuBufferCache buffer size must be > 0")
	}
	var result *struct {
		buffer  *CpuBuffer
		cleared bool
	}
	if size == DefaultBufferSize {
		i := 0
		for ; i < len(c.buffers) && c.buffers[i].buffer != nil; i++ {
			if c.buffers[i].buffer.Size() != DefaultBufferSize {
				panic("cached CPU buffer has wrong size")
			}
			if c.buffers[i].buffer.Unique() {
				result = &c.buffers[i]
			}
		}
		if result == nil && i < len(c.buffers) {
			c.buffers[i].buffer = MakeCpuBuffer(size)
			result = &c.buffers[i]
		}
	}
	var temp struct {
		buffer  *CpuBuffer
		cleared bool
	}
	if result == nil {
		temp.buffer = MakeCpuBuffer(size)
		result = &temp
	}
	if mustBeInitialized && !result.cleared {
		result.cleared = true
		clear(result.buffer.data)
	}
	result.buffer.Ref()
	return result.buffer
}

// ReleaseAll releases every cached buffer.
func (c *CpuBufferCache) ReleaseAll() {
	for i := range c.buffers {
		if c.buffers[i].buffer == nil {
			break
		}
		c.buffers[i].buffer.Unref()
		c.buffers[i].buffer = nil
		c.buffers[i].cleared = false
	}
}

// bufferBlock is one buffer in a BufferAllocPool's chain, tracking how much of it is still free.
type bufferBlock struct {
	buffer    AnyBuffer
	bytesFree uint64
}

// BufferAllocPool is a pool of geometry buffers tied to a Gpu. Clients make space for data, may put back over-allocated
// excess (LIFO), unmap before drawing, and reset after drawing to recycle space.
type BufferAllocPool struct {
	gpu              *Gpu
	provider         *ResourceProvider
	cpuBufferCache   *CpuBufferCache
	cpuStagingBuffer *CpuBuffer
	blocks           []bufferBlock
	// bufferPtr is the writable window over the back block: the mapped GPU range, the CPU buffer's data, or the staging
	// buffer's data. nil when nothing is open for writing.
	bufferPtr  []byte
	bytesInUse uint64
	bufferType gpu.BufferType
}

// NewBufferAllocPool creates a pool of the given buffer type, backed by g and provider.
func NewBufferAllocPool(g *Gpu, bufferType gpu.BufferType, cpuBufferCache *CpuBufferCache, provider *ResourceProvider) *BufferAllocPool {
	return &BufferAllocPool{
		gpu:            g,
		provider:       provider,
		bufferType:     bufferType,
		cpuBufferCache: cpuBufferCache,
	}
}

// NewVertexBufferAllocPool creates a pool specialized for vertex buffers.
func NewVertexBufferAllocPool(g *Gpu, cpuBufferCache *CpuBufferCache, provider *ResourceProvider) *BufferAllocPool {
	return NewBufferAllocPool(g, gpu.BufferTypeVertex, cpuBufferCache, provider)
}

// NewIndexBufferAllocPool creates a pool specialized for index buffers.
func NewIndexBufferAllocPool(g *Gpu, cpuBufferCache *CpuBufferCache, provider *ResourceProvider) *BufferAllocPool {
	return NewBufferAllocPool(g, gpu.BufferTypeIndex, cpuBufferCache, provider)
}

func (p *BufferAllocPool) unmapBlockBuffer(block *bufferBlock) {
	block.buffer.(*Buffer).Unmap()
}

// Unmap ensures all buffers are unmapped and have all data written to them. Call before drawing using buffers from the
// pool.
func (p *BufferAllocPool) Unmap() {
	if p.bufferPtr != nil {
		block := &p.blocks[len(p.blocks)-1]
		if buffer, ok := block.buffer.(*Buffer); ok {
			if buffer.IsMapped() {
				p.unmapBlockBuffer(block)
			} else {
				flushSize := block.buffer.Size() - block.bytesFree
				p.flushCpuData(block, flushSize)
			}
		}
		p.bufferPtr = nil
	}
}

// Reset invalidates all data in the pool and unrefs its buffers.
func (p *BufferAllocPool) Reset() {
	p.bytesInUse = 0
	p.deleteBlocks()
	p.resetCpuData(0)
}

func (p *BufferAllocPool) deleteBlocks() {
	if len(p.blocks) > 0 {
		block := &p.blocks[len(p.blocks)-1]
		if buffer, ok := block.buffer.(*Buffer); ok && buffer.IsMapped() {
			p.unmapBlockBuffer(block)
		}
	}
	for len(p.blocks) > 0 {
		p.destroyBlock()
	}
	if p.bufferPtr != nil {
		panic("bufferPtr should be nil after deleting blocks")
	}
}

func alignUpPad(x, alignment uint64) uint64 { return (alignment - x%alignment) % alignment }

func alignDown(x, alignment uint64) uint64 { return (x / alignment) * alignment }

// MakeSpace reserves a block of memory to hold size bytes at the given alignment from the start of the buffer. Returns
// the writable window, the buffer designated to hold the data, and the offset into that buffer. The window stays valid
// until MakeSpace is called again, Unmap, or Reset.
func (p *BufferAllocPool) MakeSpace(size, alignment uint64) (ptr []byte, buffer AnyBuffer, offset uint64) {
	if p.bufferPtr != nil {
		back := &p.blocks[len(p.blocks)-1]
		usedBytes := back.buffer.Size() - back.bytesFree
		pad := alignUpPad(usedBytes, alignment)
		alignedSize := pad + size
		if alignedSize >= pad && alignedSize <= back.bytesFree {
			clear(p.bufferPtr[usedBytes : usedBytes+pad])
			usedBytes += pad
			back.bytesFree -= alignedSize
			p.bytesInUse += alignedSize
			return p.bufferPtr[usedBytes : usedBytes+size], back.buffer, usedBytes
		}
	}
	// We could honor the space request with a partial update of the current buffer (if there is room), but draw calls
	// don't tell the driver that previously issued draws won't read from the updated region, so a fresh block is used
	// instead.
	if !p.createBlock(size) {
		return nil, nil, 0
	}
	back := &p.blocks[len(p.blocks)-1]
	back.bytesFree -= size
	p.bytesInUse += size
	return p.bufferPtr[:size], back.buffer, 0
}

// MakeSpaceAtLeast is like MakeSpace, but the caller gets all remaining space in the block (aligned), at least minSize;
// a new block is fallbackSize.
func (p *BufferAllocPool) MakeSpaceAtLeast(minSize, fallbackSize, alignment uint64) (ptr []byte, buffer AnyBuffer, offset, actualSize uint64) {
	var usedBytes uint64
	if len(p.blocks) > 0 {
		back := &p.blocks[len(p.blocks)-1]
		usedBytes = back.buffer.Size() - back.bytesFree
	}
	pad := alignUpPad(usedBytes, alignment)
	if p.bufferPtr == nil || len(p.blocks) == 0 || minSize+pad > p.blocks[len(p.blocks)-1].bytesFree {
		// We either don't have a block yet or the current block doesn't have enough free space.
		if !p.createBlock(fallbackSize) {
			return nil, nil, 0, 0
		}
		usedBytes = 0
		pad = 0
	}
	back := &p.blocks[len(p.blocks)-1]

	// Consume padding first, to make subsequent alignment math easier.
	clear(p.bufferPtr[usedBytes : usedBytes+pad])
	usedBytes += pad
	back.bytesFree -= pad
	p.bytesInUse += pad

	// Give caller all remaining space in this block (but aligned correctly).
	size := alignDown(back.bytesFree, alignment)
	offset = usedBytes
	back.bytesFree -= size
	p.bytesInUse += size
	return p.bufferPtr[usedBytes : usedBytes+size], back.buffer, offset, size
}

// PutBack frees bytes from MakeSpace calls in LIFO order.
func (p *BufferAllocPool) PutBack(bytes uint64) {
	if bytes == 0 {
		return
	}
	if len(p.blocks) == 0 {
		panic("putBack with no blocks")
	}
	block := &p.blocks[len(p.blocks)-1]
	// All uses are sequential with a single MakeSpaceAtLeast call, so the returned bytes must fit within the current
	// block (possibly returning all of them).
	if bytes > block.buffer.Size()-block.bytesFree {
		panic("putBack of more bytes than allocated from the block")
	}
	block.bytesFree += bytes
	p.bytesInUse -= bytes

	// We don't allow blocks without any used bytes: destroy the block if it just became empty.
	if block.bytesFree == block.buffer.Size() {
		if buffer, ok := block.buffer.(*Buffer); ok && buffer.IsMapped() {
			p.unmapBlockBuffer(block)
		}
		p.destroyBlock()
	}
}

// createBlock allocates and appends a new buffer block of at least requestSize bytes.
func (p *BufferAllocPool) createBlock(requestSize uint64) bool {
	size := max(requestSize, DefaultBufferSize)

	buffer := p.getBuffer(size)
	if buffer == nil {
		return false
	}
	p.blocks = append(p.blocks, bufferBlock{buffer: buffer, bytesFree: buffer.Size()})

	if p.bufferPtr != nil {
		if len(p.blocks) < 2 {
			panic("bufferPtr without a previous block")
		}
		prev := &p.blocks[len(p.blocks)-2]
		if prevBuffer, ok := prev.buffer.(*Buffer); ok {
			if prevBuffer.IsMapped() {
				p.unmapBlockBuffer(prev)
			} else {
				p.flushCpuData(prev, prev.buffer.Size()-prev.bytesFree)
			}
		}
		p.bufferPtr = nil
	}

	// If the buffer is CPU-backed we "map" it because it is free to do so and saves a copy. Otherwise, when buffer
	// mapping is supported, we map if the buffer size is over the caps threshold.
	block := &p.blocks[len(p.blocks)-1]
	if cpuBuffer, ok := block.buffer.(*CpuBuffer); ok {
		p.bufferPtr = cpuBuffer.Data()
	} else if p.gpu.glCaps().MapBufferType() != MapBufferNone &&
		size > uint64(p.gpu.Caps().BufferMapThreshold) {
		gpuBuffer := block.buffer.(*Buffer)
		if mapPtr := gpuBuffer.Map(); mapPtr != nil {
			p.bufferPtr = unsafe.Slice((*byte)(mapPtr), size)
		}
	}
	if p.bufferPtr == nil {
		p.resetCpuData(block.bytesFree)
		p.bufferPtr = p.cpuStagingBuffer.Data()[:block.bytesFree]
	}
	return true
}

// destroyBlock releases and removes the pool's last (back) block.
func (p *BufferAllocPool) destroyBlock() {
	if len(p.blocks) == 0 {
		panic("destroyBlock with no blocks")
	}
	block := &p.blocks[len(p.blocks)-1]
	switch buffer := block.buffer.(type) {
	case *Buffer:
		if buffer.IsMapped() {
			panic("destroying a block with a mapped buffer")
		}
		buffer.Unref()
	case *CpuBuffer:
		buffer.Unref()
	}
	// Zero the slot before shrinking so the retained backing array — now reused across flushes because the pool is
	// persistent (DrawingManager.flushState) — pins no dropped buffer. Reslicing alone would leave the AnyBuffer
	// reference live beyond len.
	last := len(p.blocks) - 1
	p.blocks[last] = bufferBlock{}
	p.blocks = p.blocks[:last]
	p.bufferPtr = nil
}

// resetCpuData ensures the pool's CPU staging buffer is at least newSize bytes, replacing or releasing it as needed.
func (p *BufferAllocPool) resetCpuData(newSize uint64) {
	if newSize != 0 && newSize < DefaultBufferSize {
		panic("staging buffer smaller than the default buffer size")
	}
	if newSize == 0 {
		if p.cpuStagingBuffer != nil {
			p.cpuStagingBuffer.Unref()
			p.cpuStagingBuffer = nil
		}
		return
	}
	if p.cpuStagingBuffer != nil && newSize <= p.cpuStagingBuffer.Size() {
		return
	}
	if p.cpuStagingBuffer != nil {
		p.cpuStagingBuffer.Unref()
	}
	mustInitialize := p.gpu.Caps().MustClearUploadedBufferData
	if p.cpuBufferCache != nil {
		p.cpuStagingBuffer = p.cpuBufferCache.MakeBuffer(newSize, mustInitialize)
	} else {
		p.cpuStagingBuffer = MakeCpuBuffer(newSize)
	}
}

// flushCpuData pushes staged bytes into the GPU buffer via map+copy or UpdateData.
func (p *BufferAllocPool) flushCpuData(block *bufferBlock, flushSize uint64) {
	buffer := block.buffer.(*Buffer)
	if buffer.IsMapped() {
		panic("flushing to a mapped buffer")
	}
	if p.cpuStagingBuffer == nil || flushSize > buffer.Size() {
		panic("flushCpuData preconditions violated")
	}
	if p.gpu.glCaps().MapBufferType() != MapBufferNone &&
		flushSize > uint64(p.gpu.Caps().BufferMapThreshold) {
		if mapPtr := buffer.Map(); mapPtr != nil {
			copy(unsafe.Slice((*byte)(mapPtr), flushSize), p.bufferPtr[:flushSize])
			p.unmapBlockBuffer(block)
			return
		}
	}
	buffer.UpdateData(p.bufferPtr[:flushSize], 0, false)
}

// getBuffer allocates a fresh buffer of size bytes, CPU- or GPU-backed depending on caps.
func (p *BufferAllocPool) getBuffer(size uint64) AnyBuffer {
	caps := p.gpu.Caps()
	if caps.PreferClientSideDynamicBuffers ||
		(p.bufferType == gpu.BufferTypeDrawIndirect && caps.UseClientSideIndirectBuffers) {
		mustInitialize := caps.MustClearUploadedBufferData
		if p.cpuBufferCache != nil {
			return p.cpuBufferCache.MakeBuffer(size, mustInitialize)
		}
		return MakeCpuBuffer(size)
	}
	buffer := p.provider.CreateBuffer(size, p.bufferType, gpu.AccessPatternDynamic, ZeroInitNo)
	if buffer == nil {
		return nil
	}
	return buffer
}

// MakeVertexSpace reserves space for vertexCount vertices of vertexSize bytes each. startVertex is in units of
// vertexSize.
func (p *BufferAllocPool) MakeVertexSpace(vertexSize uint64, vertexCount int) (ptr []byte, buffer AnyBuffer, startVertex int) {
	if vertexCount < 0 {
		panic("negative vertex count")
	}
	ptr, buffer, offset := p.MakeSpace(vertexSize*uint64(vertexCount), vertexSize)
	if ptr == nil {
		return nil, nil, 0
	}
	return ptr, buffer, int(offset / vertexSize)
}

// MakeVertexSpaceAtLeast reserves at least minVertexCount vertices, up to a full block.
func (p *BufferAllocPool) MakeVertexSpaceAtLeast(vertexSize uint64, minVertexCount, fallbackVertexCount int) (ptr []byte, buffer AnyBuffer, startVertex, actualVertexCount int) {
	if minVertexCount < 0 || fallbackVertexCount < minVertexCount {
		panic("invalid vertex counts")
	}
	ptr, buffer, offset, actualSize := p.MakeSpaceAtLeast(vertexSize*uint64(minVertexCount),
		vertexSize*uint64(fallbackVertexCount), vertexSize)
	if ptr == nil {
		return nil, nil, 0, 0
	}
	return ptr, buffer, int(offset / vertexSize), int(actualSize / vertexSize)
}

// MakeIndexSpace reserves space for indexCount 16-bit indices.
func (p *BufferAllocPool) MakeIndexSpace(indexCount int) (ptr []byte, buffer AnyBuffer, startIndex int) {
	if indexCount < 0 {
		panic("negative index count")
	}
	ptr, buffer, offset := p.MakeSpace(uint64(indexCount)*2, 2)
	if ptr == nil {
		return nil, nil, 0
	}
	return ptr, buffer, int(offset / 2)
}

// MakeIndexSpaceAtLeast reserves at least minIndexCount 16-bit indices, up to a full block.
func (p *BufferAllocPool) MakeIndexSpaceAtLeast(minIndexCount, fallbackIndexCount int) (ptr []byte, buffer AnyBuffer, startIndex, actualIndexCount int) {
	if minIndexCount < 0 || fallbackIndexCount < minIndexCount {
		panic("invalid index counts")
	}
	ptr, buffer, offset, actualSize := p.MakeSpaceAtLeast(uint64(minIndexCount)*2,
		uint64(fallbackIndexCount)*2, 2)
	if ptr == nil {
		return nil, nil, 0, 0
	}
	return ptr, buffer, int(offset / 2), int(actualSize / 2)
}
