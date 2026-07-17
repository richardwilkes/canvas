// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The eager vertex allocator interface the triangulators use to obtain vertex space for a worst-case vertex count,
// unlocking the space they did not use. eagerDynamicVertexAllocator locks space from the flush-time vertex pool;
// cpuVertexAllocator (used by the prePrepare lane and by hermetic tests) backs the lock with CPU memory that can be
// detached as thread-safe-cache vertex data.

package gl

// eagerVertexAllocator is the interface for locking vertex space ahead of a worst-case vertex count. lockWriter returns
// space for eagerCount vertices of the given stride (nil on failure); unlock returns the space that was not used, and
// must be called exactly once after a successful lock.
type eagerVertexAllocator interface {
	lockWriter(stride uint64, eagerCount int) []byte
	unlock(actualCount int)
}

// eagerDynamicVertexAllocator allocates via the draw target's flush-time vertex pool. The final buffer and start vertex
// are published to its own fields at unlock time.
type eagerDynamicVertexAllocator struct {
	buffer      AnyBuffer
	target      *OpFlushState
	vertexData  []byte
	firstVertex int
	lockStride  uint64
	lockCount   int
}

func newEagerDynamicVertexAllocator(target *OpFlushState) *eagerDynamicVertexAllocator {
	return &eagerDynamicVertexAllocator{target: target}
}

func (a *eagerDynamicVertexAllocator) lockWriter(stride uint64, eagerCount int) []byte {
	if a.lockStride != 0 {
		panic("lock while locked")
	}
	if stride == 0 || eagerCount == 0 {
		panic("invalid lock request")
	}
	a.vertexData, a.buffer, a.firstVertex = a.target.MakeVertexSpace(stride, eagerCount)
	if a.vertexData == nil {
		return nil
	}
	a.lockStride = stride
	a.lockCount = eagerCount
	return a.vertexData
}

func (a *eagerDynamicVertexAllocator) unlock(actualCount int) {
	if a.lockStride == 0 {
		panic("unlock without lock")
	}
	a.target.PutBackVertices(a.lockCount-actualCount, a.lockStride)
	if actualCount == 0 {
		a.buffer = nil
		a.firstVertex = 0
	}
	a.vertexData = nil
	a.lockStride = 0
	a.lockCount = 0
}

// cpuVertexAllocator provides CPU-side vertex space whose final contents can be detached (as thread-safe-cache
// VertexData in the prePrepare lane).
type cpuVertexAllocator struct {
	vertexData []byte
	lockStride uint64
}

func (a *cpuVertexAllocator) lockWriter(stride uint64, eagerCount int) []byte {
	if a.lockStride != 0 {
		panic("lock while locked")
	}
	if stride == 0 || eagerCount == 0 {
		panic("invalid lock request")
	}
	a.vertexData = make([]byte, uint64(eagerCount)*stride)
	a.lockStride = stride
	return a.vertexData
}

func (a *cpuVertexAllocator) unlock(actualCount int) {
	if a.lockStride == 0 {
		panic("unlock without lock")
	}
	a.vertexData = a.vertexData[:uint64(actualCount)*a.lockStride]
	a.lockStride = 0
}

// detachVertexData returns the locked vertex data and clears the allocator's reference to it, for use by hermetic
// tests.
func (a *cpuVertexAllocator) detachVertexData() []byte {
	if a.lockStride != 0 {
		panic("detach while locked")
	}
	data := a.vertexData
	a.vertexData = nil
	return data
}
