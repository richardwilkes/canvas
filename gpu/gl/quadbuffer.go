// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-op storage for accumulated (device quad, optional local quad, metadata) entries that batching concatenates.
// Entries are kept in insertion order over a plain slice, with a per-entry has-locals flag and buffer-level tracking of
// the most general quad type seen; iterators hand out damageable scratch copies. A denser packed layout is a
// memory-layout detail the pooling work revisits.

package gl

// quadBufferEntry is one (device, optional local, metadata) tuple.
type quadBufferEntry[T any] struct {
	metadata  T
	device    Quad
	local     Quad
	hasLocals bool
}

// QuadBuffer accumulates (device quad, optional local quad, metadata) entries for a batched op.
type QuadBuffer[T any] struct {
	entries    []quadBufferEntry[T]
	deviceType QuadType
	localType  QuadType
}

// NewQuadBuffer creates a QuadBuffer with capacity reserved for count entries. If needsLocals is true, space is
// reserved assuming each entry also carries a local quad.
func NewQuadBuffer[T any](count int, needsLocals bool) *QuadBuffer[T] {
	_ = needsLocals // Reservation shape only; the Go slice stores both forms uniformly.
	return &QuadBuffer[T]{entries: make([]quadBufferEntry[T], 0, count)}
}

// borrowQuadBuffer is NewQuadBuffer from a free list: a pooled op that owns a QuadBuffer borrows the shell (and, when
// batching grew it past a single entry on some earlier frame, its preserved entries backing) instead of allocating both
// every frame. The entry payload is pointer-free POD (device/local Quad, the metadata color/AA/subset structs), so a
// preserved backing pins nothing.
func borrowQuadBuffer[T any](p *opPool[QuadBuffer[T]], count int, needsLocals bool) *QuadBuffer[T] {
	_ = needsLocals
	b := p.borrow()
	if cap(b.entries) < count {
		b.entries = make([]quadBufferEntry[T], 0, count)
	}
	return b
}

// recycleQuadBuffer returns a dead op's QuadBuffer to its free list, preserving the entries backing (cleared, length 0)
// the way the op pools preserve instance backings. Unlike recycleKeepingBacking it also keeps a capacity-1 backing:
// every QuadBuffer starts at capacity 1 (the constructor's reserve), so dropping it would just re-allocate the same
// array next frame, and the pointer-free POD entries make retention free. Must only be called from the owning op's
// recycle(), where the op — and therefore its uniquely owned buffer — is provably dead (see oppool.go).
func recycleQuadBuffer[T any](p *opPool[QuadBuffer[T]], b *QuadBuffer[T]) {
	backing := b.entries[:cap(b.entries)]
	clear(backing)
	var zero QuadBuffer[T]
	*b = zero
	b.entries = backing[:0]
	p.pool.Put(b)
}

// Count returns the number of entries in the buffer.
func (b *QuadBuffer[T]) Count() int { return len(b.entries) }

// DeviceQuadType returns the most general type among the device quads added so far.
func (b *QuadBuffer[T]) DeviceQuadType() QuadType { return b.deviceType }

// LocalQuadType returns the most general type among the local quads added so far; if no local quads were added this
// reports axis-aligned.
func (b *QuadBuffer[T]) LocalQuadType() QuadType { return b.localType }

// Append adds deviceQuad with its metadata, and localQuad's coordinates when it is non-nil.
func (b *QuadBuffer[T]) Append(deviceQuad *Quad, metadata T, localQuad *Quad) {
	entry := quadBufferEntry[T]{device: *deviceQuad, metadata: metadata}
	if localQuad != nil {
		entry.local = *localQuad
		entry.hasLocals = true
	}
	b.entries = append(b.entries, entry)
	if deviceQuad.QuadType() > b.deviceType {
		b.deviceType = deviceQuad.QuadType()
	}
	if localQuad != nil && localQuad.QuadType() > b.localType {
		b.localType = localQuad.QuadType()
	}
}

// Concat copies all entries from that buffer to this one.
func (b *QuadBuffer[T]) Concat(that *QuadBuffer[T]) {
	b.entries = append(b.entries, that.entries...)
	if that.deviceType > b.deviceType {
		b.deviceType = that.deviceType
	}
	if that.localType > b.localType {
		b.localType = that.localType
	}
}

// QuadBufferIter is a read iterator whose DeviceQuad/LocalQuad results are scratch copies that callers may damage (the
// tessellator does); subsequent Next calls overwrite them.
type QuadBufferIter[T any] struct {
	buffer     *QuadBuffer[T]
	deviceQuad Quad
	localQuad  Quad
	index      int
}

// Iterator returns a read iterator over the buffer's entries.
func (b *QuadBuffer[T]) Iterator() QuadBufferIter[T] {
	return QuadBufferIter[T]{buffer: b, index: -1}
}

// Next advances to the next entry, returning false once the buffer is exhausted.
func (i *QuadBufferIter[T]) Next() bool {
	i.index++
	if i.index >= len(i.buffer.entries) {
		return false
	}
	entry := &i.buffer.entries[i.index]
	i.deviceQuad = entry.device
	if entry.hasLocals {
		i.localQuad = entry.local
	}
	return true
}

// Metadata returns a pointer to the current entry's metadata.
func (i *QuadBufferIter[T]) Metadata() *T { return &i.buffer.entries[i.index].metadata }

// DeviceQuad returns the current entry's device quad; the returned pointer is mutable scratch.
func (i *QuadBufferIter[T]) DeviceQuad() *Quad { return &i.deviceQuad }

// LocalQuad returns the current entry's local quad, or nil when the entry has no local coordinates.
func (i *QuadBufferIter[T]) LocalQuad() *Quad {
	if !i.IsLocalValid() {
		return nil
	}
	return &i.localQuad
}

// IsLocalValid reports whether the current entry carries local coordinates.
func (i *QuadBufferIter[T]) IsLocalValid() bool { return i.buffer.entries[i.index].hasLocals }

// QuadBufferMetadataIter is a mutable iterator over just the metadata, for op finalization that rewrites state such as
// color.
type QuadBufferMetadataIter[T any] struct {
	buffer *QuadBuffer[T]
	index  int
}

// Metadata returns a mutable metadata iterator over the buffer's entries.
func (b *QuadBuffer[T]) Metadata() QuadBufferMetadataIter[T] {
	return QuadBufferMetadataIter[T]{buffer: b, index: -1}
}

// Next advances to the next entry, returning false once the buffer is exhausted.
func (i *QuadBufferMetadataIter[T]) Next() bool {
	i.index++
	return i.index < len(i.buffer.entries)
}

// Get returns the current entry's metadata for read/write.
func (i *QuadBufferMetadataIter[T]) Get() *T { return &i.buffer.entries[i.index].metadata }
