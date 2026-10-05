// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The eager vertex allocator interface the triangulators use to obtain vertex space for a worst-case vertex count,
// unlocking the space they did not use. eagerDynamicVertexAllocator locks space from the flush-time vertex pool.

package gl

// eagerVertexAllocator is the interface for locking vertex space ahead of a worst-case vertex count. lockWriter returns
// space for eagerCount vertices of the given stride (nil on failure); unlock returns the space that was not used, and
// must be called exactly once after a successful lock.
type eagerVertexAllocator interface {
	lockWriter(stride uint64, eagerCount int) []byte
	unlock(actualCount int)
}

// eagerDynamicVertexAllocator allocates via the draw target's flush-time vertex pool. After unlock, its buffer and
// firstVertex fields locate the vertices (both are cleared when none were kept).
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
