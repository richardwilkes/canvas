// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A GPU buffer object (vertex, index, or transfer), flattened into one type since GL is the only backend. Desktop
// trims: the Chromium map-buffer and NV-PBO transfer lanes are unreachable and dropped, as is the
// has-been-attached-to-a-texture flag, which only serves the GL_TEXTURE_BUFFER lane this trim does not port.

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/gpu"
)

// Buffer is a GL buffer object managed by the resource cache.
type Buffer struct {
	g      *Gpu
	mapPtr unsafe.Pointer
	gpu.ResourceBase
	sizeInBytes   uint64
	intendedType  gpu.BufferType
	accessPattern gpu.AccessPattern
	bufferID      uint32
	usage         uint32
}

// bufferMapType is how a buffer is mapped for CPU access, determined entirely by the buffer type.
type bufferMapType bool

const (
	mapTypeRead         bufferMapType = false
	mapTypeWriteDiscard bufferMapType = true
)

func (b *Buffer) mapType() bufferMapType {
	if b.intendedType == gpu.BufferTypeXferGpuToCpu {
		return mapTypeRead
	}
	return mapTypeWriteDiscard
}

// toGLAccessPattern maps an access pattern (and buffer type) to the GL usage hint (with the NV-PBO special case
// dropped).
func toGLAccessPattern(bufferType gpu.BufferType, accessPattern gpu.AccessPattern) uint32 {
	if bufferType == gpu.BufferTypeXferGpuToCpu {
		switch accessPattern {
		case gpu.AccessPatternDynamic:
			return DYNAMIC_READ
		case gpu.AccessPatternStatic:
			return STATIC_READ
		case gpu.AccessPatternStream:
			return STREAM_READ
		}
	}
	switch accessPattern {
	case gpu.AccessPatternDynamic:
		// STREAM_DRAW is used for dynamic access to trigger a Chromium GPU-process optimization; it is kept on
		// non-Chromium targets too.
		return STREAM_DRAW
	case gpu.AccessPatternStatic:
		return STATIC_DRAW
	case gpu.AccessPatternStream:
		return STREAM_DRAW
	}
	panic("unreachable access pattern")
}

// MakeBuffer creates a GPU buffer of the given size, type, and access pattern. Returns nil if the buffer object could
// not be created.
func MakeBuffer(g *Gpu, size uint64, intendedType gpu.BufferType, accessPattern gpu.AccessPattern) *Buffer {
	if g.glCaps().TransferBufferType() == TransferBufferNone &&
		(intendedType == gpu.BufferTypeXferCpuToGpu ||
			intendedType == gpu.BufferTypeXferGpuToCpu) {
		return nil
	}
	b := &Buffer{
		g:             g,
		sizeInBytes:   size,
		intendedType:  intendedType,
		accessPattern: accessPattern,
		usage:         toGLAccessPattern(intendedType, accessPattern),
	}
	// The creation bind below is recorded in the buffer state shadow under our unique ID, so the base state must exist
	// first.
	b.InitResource("MakeGlBuffer")
	g.fns().GenBuffers(1, &b.bufferID)
	if b.bufferID != 0 {
		target := g.bindBuffer(intendedType, b)
		if err := g.allocCall(func(f *Functions) {
			f.BufferData(target, int(size), 0, b.usage)
		}); err != NO_ERROR {
			g.fns().DeleteBuffers(1, &b.bufferID)
			b.bufferID = 0
		}
	}
	b.RegisterWithCache(g.ResourceCache(), b, gpu.BudgetedYes, "MakeGlBuffer")
	if b.bufferID == 0 {
		// Drop the scratch key first so the dead resource isn't offered up for reuse, then release the ref
		// InitResource took, which lets the cache free it. Without the unref it would stay registered forever, with
		// its full requested size still counted against the cache's budget.
		b.RemoveScratchKey()
		b.Unref()
		return nil
	}
	return b
}

// BufferID returns the underlying GL buffer object name.
func (b *Buffer) BufferID() uint32 { return b.bufferID }

// Size returns the buffer's size in bytes.
func (b *Buffer) Size() uint64 { return b.sizeInBytes }

// IntendedType returns the buffer's intended use (vertex, index, or transfer).
func (b *Buffer) IntendedType() gpu.BufferType { return b.intendedType }

// AccessPattern returns the buffer's access pattern (static/dynamic/stream).
func (b *Buffer) AccessPattern() gpu.AccessPattern { return b.accessPattern }

// IsMapped reports whether the buffer is currently mapped for CPU access.
func (b *Buffer) IsMapped() bool { return b.mapPtr != nil }

// IsCpuBuffer reports whether this is a CPU-only buffer (always false; CPU buffer pools are a future feature).
func (b *Buffer) IsCpuBuffer() bool { return false }

// ResourceType implements gpu.Resource.
func (b *Buffer) ResourceType() string { return "Buffer Object" }

// OnGpuMemorySize implements gpu.Resource.
func (b *Buffer) OnGpuMemorySize() uint64 { return b.sizeInBytes }

var dynamicBufferScratchResourceType = gpu.GenerateScratchKeyResourceType()

// ComputeScratchKeyForDynamicBuffer computes the scratch key under which dynamic buffers of a given size and type are
// shared.
func ComputeScratchKeyForDynamicBuffer(size uint64, intendedType gpu.BufferType, key *gpu.ScratchKey) {
	kb := gpu.ScratchKeyBuilder(key, dynamicBufferScratchResourceType, 3)
	s := kb.Slice()
	s[0] = uint32(intendedType)
	s[1] = uint32(size)
	s[2] = uint32(size >> 32)
	kb.Finish()
}

// ComputeScratchKey implements gpu.Resource: only "dynamic" buffers are cached and reused.
func (b *Buffer) ComputeScratchKey(key *gpu.ScratchKey) {
	if b.accessPattern == gpu.AccessPatternDynamic {
		ComputeScratchKeyForDynamicBuffer(b.sizeInBytes, b.intendedType, key)
	}
}

// OnRelease implements gpu.Resource.
func (b *Buffer) OnRelease() {
	if !b.WasDestroyed() {
		if b.bufferID != 0 {
			b.g.fns().DeleteBuffers(1, &b.bufferID)
			b.bufferID = 0
		}
		b.mapPtr = nil
	}
}

// OnAbandon implements gpu.Resource.
func (b *Buffer) OnAbandon() {
	b.bufferID = 0
	b.mapPtr = nil
}

// invalidateBuffer discards the buffer's prior contents via whichever mechanism the driver supports (a resubmit of null
// data, or an explicit invalidate call).
func (b *Buffer) invalidateBuffer(target uint32) uint32 {
	g := b.g
	switch g.glCaps().InvalidateBufferType() {
	case InvalidateBufferNone:
		return NO_ERROR
	case InvalidateBufferNullData:
		return g.allocCall(func(f *Functions) {
			f.BufferData(target, int(b.sizeInBytes), 0, b.usage)
		})
	case InvalidateBufferInvalidate:
		g.fns().InvalidateBufferData(b.bufferID)
		return NO_ERROR
	}
	panic("unreachable invalidate buffer type")
}

// Map maps the buffer for CPU access, returning nil on failure. Xfer GPU-to-CPU buffers map for reading; everything
// else maps write-discard.
func (b *Buffer) Map() unsafe.Pointer {
	if b.WasDestroyed() {
		return nil
	}
	if b.mapPtr == nil {
		b.onMap(b.mapType())
	}
	return b.mapPtr
}

// Unmap unmaps a previously mapped buffer.
func (b *Buffer) Unmap() {
	if b.WasDestroyed() {
		return
	}
	if b.mapPtr == nil {
		panic("buffer not mapped")
	}
	b.onUnmap()
	b.mapPtr = nil
}

// onMap maps the buffer using whichever mapping mechanism the driver supports (desktop lanes only).
func (b *Buffer) onMap(mt bufferMapType) {
	g := b.g
	switch g.glCaps().MapBufferType() {
	case MapBufferNone:
		return
	case MapBufferMapBuffer:
		target := g.bindBuffer(b.intendedType, b)
		if mt == mapTypeWriteDiscard {
			if b.invalidateBuffer(target) != NO_ERROR {
				return
			}
		}
		access := uint32(WRITE_ONLY)
		if mt == mapTypeRead {
			access = READ_ONLY
		}
		b.mapPtr = g.fns().MapBuffer(target, access)
	case MapBufferMapBufferRange:
		target := g.bindBuffer(b.intendedType, b)
		var access uint32
		if mt == mapTypeRead {
			access = MAP_READ_BIT
		} else {
			access = MAP_WRITE_BIT | MAP_INVALIDATE_BUFFER_BIT
		}
		b.mapPtr = g.fns().MapBufferRange(target, 0, int(b.sizeInBytes), access)
	}
}

// onUnmap unmaps the buffer via whichever mapping mechanism was used to map it.
func (b *Buffer) onUnmap() {
	g := b.g
	switch g.glCaps().MapBufferType() {
	case MapBufferNone:
		panic("unmap without map support")
	case MapBufferMapBuffer, MapBufferMapBufferRange:
		target := g.bindBuffer(b.intendedType, b)
		g.fns().UnmapBuffer(target)
	}
}

// ClearToZero zeroes the buffer's contents, returning false if it could not be cleared.
func (b *Buffer) ClearToZero() bool {
	if b.IsMapped() {
		panic("buffer is mapped")
	}
	if b.WasDestroyed() || b.intendedType == gpu.BufferTypeXferGpuToCpu {
		return false
	}
	// We could improve this on GL 4.3+ with glClearBufferData.
	b.onMap(mapTypeWriteDiscard)
	if b.mapPtr != nil {
		zero(unsafe.Slice((*byte)(b.mapPtr), b.sizeInBytes))
		b.onUnmap()
		b.mapPtr = nil
		return true
	}
	return b.UpdateData(make([]byte, b.sizeInBytes), 0, false)
}

// UpdateData writes src at offset. If preserve is false the remaining contents become undefined; preserving updates
// must meet the caps' alignment requirement.
func (b *Buffer) UpdateData(src []byte, offset uint64, preserve bool) bool {
	if b.IsMapped() {
		panic("buffer is mapped")
	}
	size := uint64(len(src))
	if size == 0 || offset+size > b.sizeInBytes {
		panic("invalid update range")
	}
	if b.WasDestroyed() {
		return false
	}
	if preserve {
		a := uint64(b.g.Caps().BufferUpdateDataPreserveAlignment)
		if offset%a != 0 || size%a != 0 {
			return false
		}
	}
	if b.intendedType == gpu.BufferTypeXferGpuToCpu {
		return false
	}
	target := b.g.bindBuffer(b.intendedType, b)
	if !preserve {
		if b.invalidateBuffer(target) != NO_ERROR {
			return false
		}
	}
	b.g.fns().BufferSubData(target, int(offset), int(size), uintptr(unsafe.Pointer(&src[0])))
	return true
}

// OnSetLabel implements gpu.Resource.
func (b *Buffer) OnSetLabel() {
	if b.bufferID != 0 && b.Label() != "" && b.g.glCaps().DebugSupport() {
		b.g.fns().ObjectLabel(BUFFER, b.bufferID, -1, cString("_Skia_"+b.Label()))
	}
}

func zero(s []byte) {
	for i := range s {
		s[i] = 0
	}
}
