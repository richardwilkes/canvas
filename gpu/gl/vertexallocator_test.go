// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

// cpuVertexAllocator is an eagerVertexAllocator that backs the lock with CPU memory whose final contents can be
// detached, for the hermetic tests.
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

func (a *cpuVertexAllocator) detachVertexData() []byte {
	if a.lockStride != 0 {
		panic("detach while locked")
	}
	data := a.vertexData
	a.vertexData = nil
	return data
}
