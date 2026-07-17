// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The vertex array object wrapper and the per-VAO attrib pointer/enable state shadow. AttribArrayState.Set chooses
// glVertexAttribPointer vs glVertexAttribIPointer based on the caller- supplied gpuTypeIsFloat flag rather than a
// shader type, since the caller already knows which GL call the destination shader attribute needs.

package gl

import "github.com/richardwilkes/canvas/gpu"

// attribLayout describes how a vertex attribute type maps onto a glVertexAttribPointer call.
type attribLayout struct {
	normalized bool
	count      int32
	glType     uint32
}

// attribLayoutForType returns the GL attrib layout for a vertex attribute type.
func attribLayoutForType(t gpu.VertexAttribType) attribLayout {
	switch t {
	case gpu.VertexAttribTypeFloat:
		return attribLayout{normalized: false, count: 1, glType: FLOAT}
	case gpu.VertexAttribTypeFloat2:
		return attribLayout{normalized: false, count: 2, glType: FLOAT}
	case gpu.VertexAttribTypeFloat3:
		return attribLayout{normalized: false, count: 3, glType: FLOAT}
	case gpu.VertexAttribTypeFloat4:
		return attribLayout{normalized: false, count: 4, glType: FLOAT}
	case gpu.VertexAttribTypeHalf:
		return attribLayout{normalized: false, count: 1, glType: HALF_FLOAT}
	case gpu.VertexAttribTypeHalf2:
		return attribLayout{normalized: false, count: 2, glType: HALF_FLOAT}
	case gpu.VertexAttribTypeHalf4:
		return attribLayout{normalized: false, count: 4, glType: HALF_FLOAT}
	case gpu.VertexAttribTypeInt2:
		return attribLayout{normalized: false, count: 2, glType: INT}
	case gpu.VertexAttribTypeInt3:
		return attribLayout{normalized: false, count: 3, glType: INT}
	case gpu.VertexAttribTypeInt4:
		return attribLayout{normalized: false, count: 4, glType: INT}
	case gpu.VertexAttribTypeByte:
		return attribLayout{normalized: false, count: 1, glType: BYTE}
	case gpu.VertexAttribTypeByte2:
		return attribLayout{normalized: false, count: 2, glType: BYTE}
	case gpu.VertexAttribTypeByte4:
		return attribLayout{normalized: false, count: 4, glType: BYTE}
	case gpu.VertexAttribTypeUByte:
		return attribLayout{normalized: false, count: 1, glType: UNSIGNED_BYTE}
	case gpu.VertexAttribTypeUByte2:
		return attribLayout{normalized: false, count: 2, glType: UNSIGNED_BYTE}
	case gpu.VertexAttribTypeUByte4:
		return attribLayout{normalized: false, count: 4, glType: UNSIGNED_BYTE}
	case gpu.VertexAttribTypeUByteNorm:
		return attribLayout{normalized: true, count: 1, glType: UNSIGNED_BYTE}
	case gpu.VertexAttribTypeUByte4Norm:
		return attribLayout{normalized: true, count: 4, glType: UNSIGNED_BYTE}
	case gpu.VertexAttribTypeShort2:
		return attribLayout{normalized: false, count: 2, glType: SHORT}
	case gpu.VertexAttribTypeShort4:
		return attribLayout{normalized: false, count: 4, glType: SHORT}
	case gpu.VertexAttribTypeUShort2:
		return attribLayout{normalized: false, count: 2, glType: UNSIGNED_SHORT}
	case gpu.VertexAttribTypeUShort2Norm:
		return attribLayout{normalized: true, count: 2, glType: UNSIGNED_SHORT}
	case gpu.VertexAttribTypeInt:
		return attribLayout{normalized: false, count: 1, glType: INT}
	case gpu.VertexAttribTypeUInt:
		return attribLayout{normalized: false, count: 1, glType: UNSIGNED_INT}
	case gpu.VertexAttribTypeUShortNorm:
		return attribLayout{normalized: true, count: 1, glType: UNSIGNED_SHORT}
	case gpu.VertexAttribTypeUShort4Norm:
		return attribLayout{normalized: true, count: 4, glType: UNSIGNED_SHORT}
	}
	panic("unknown vertex attrib type")
}

const invalidDivisor = -1

// attribArrayState tracks the state of one glVertexAttribArray index.
type attribArrayState struct {
	vertexBufferUniqueID gpu.ResourceUniqueID
	cpuType              gpu.VertexAttribType
	gpuTypeIsFloat       bool
	stride               int32
	offset               uintptr
	divisor              int32
}

func (a *attribArrayState) invalidate() {
	a.vertexBufferUniqueID = 0
	a.divisor = invalidDivisor
}

// AttribArrayState sets and tracks vertex attribute array state for the currently bound vertex array object.
type AttribArrayState struct {
	states                  []attribArrayState
	numEnabledArrays        int
	primitiveRestartEnabled bool
	enableStateIsValid      bool
}

// NewAttribArrayState returns a new AttribArrayState tracking arrayCount attribute slots.
func NewAttribArrayState(arrayCount int) *AttribArrayState {
	s := &AttribArrayState{}
	s.Resize(arrayCount)
	return s
}

// Resize changes the number of tracked attribute slots, invalidating all cached state.
func (s *AttribArrayState) Resize(newCount int) {
	states := make([]attribArrayState, newCount)
	copy(states, s.states)
	s.states = states
	s.Invalidate()
}

// Count returns the number of tracked attribute slots.
func (s *AttribArrayState) Count() int { return len(s.states) }

// Invalidate marks all cached attribute and enable state as stale, forcing the next Set/ EnableVertexArrays call to
// re-issue the corresponding GL calls.
func (s *AttribArrayState) Invalidate() {
	for i := range s.states {
		s.states[i].invalidate()
	}
	s.enableStateIsValid = false
}

// Set enables and sets vertex attrib state for the given attrib index, assuming this object tracks the currently bound
// vertex array object. gpuTypeIsFloat selects glVertexAttribPointer (true) vs glVertexAttribIPointer (false).
func (s *AttribArrayState) Set(g *Gpu, index int, vertexBuffer *Buffer, cpuType gpu.VertexAttribType, gpuTypeIsFloat bool, stride int32, offsetInBytes uintptr, divisor int32) {
	if index < 0 || index >= len(s.states) {
		panic("attrib index out of range")
	}
	if divisor != 0 && !g.Caps().DrawInstancedSupport {
		panic("instancing without support")
	}
	array := &s.states[index]
	bufferChanged := false
	if array.vertexBufferUniqueID != vertexBuffer.UniqueID() {
		bufferChanged = true
		array.vertexBufferUniqueID = vertexBuffer.UniqueID()
	}
	if bufferChanged || array.cpuType != cpuType || array.gpuTypeIsFloat != gpuTypeIsFloat ||
		array.stride != stride || array.offset != offsetInBytes {
		// We always have to call this if we're going to change the array pointer. 'array' tracks the last buffer used
		// to set up attrib pointers, not the last buffer bound. bindBuffer avoids redundant binds.
		g.bindBuffer(gpu.BufferTypeVertex, vertexBuffer)
		layout := attribLayoutForType(cpuType)
		if gpuTypeIsFloat {
			g.fns().VertexAttribPointer(uint32(index), layout.count, layout.glType,
				layout.normalized, stride, offsetInBytes)
		} else {
			if layout.normalized {
				panic("integer attribs cannot be normalized")
			}
			g.fns().VertexAttribIPointer(uint32(index), layout.count, layout.glType, stride,
				offsetInBytes)
		}
		array.cpuType = cpuType
		array.gpuTypeIsFloat = gpuTypeIsFloat
		array.stride = stride
		array.offset = offsetInBytes
	}
	if g.Caps().DrawInstancedSupport && array.divisor != divisor {
		if divisor != 0 && divisor != 1 {
			panic("unexpected divisor")
		}
		g.fns().VertexAttribDivisor(uint32(index), uint32(divisor))
		array.divisor = divisor
	}
}

// EnableVertexArrays enables the first enabledCount vertex arrays and disables the rest, and flushes primitive restart
// state.
func (s *AttribArrayState) EnableVertexArrays(g *Gpu, enabledCount int, primitiveRestart bool) {
	if enabledCount > len(s.states) {
		panic("enabling more arrays than tracked")
	}
	if !s.enableStateIsValid || enabledCount != s.numEnabledArrays {
		firstIdxToEnable := 0
		endIdxToDisable := len(s.states)
		if s.enableStateIsValid {
			firstIdxToEnable = s.numEnabledArrays
			endIdxToDisable = s.numEnabledArrays
		}
		for i := firstIdxToEnable; i < enabledCount; i++ {
			g.fns().EnableVertexAttribArray(uint32(i))
		}
		for i := enabledCount; i < endIdxToDisable; i++ {
			g.fns().DisableVertexAttribArray(uint32(i))
		}
		s.numEnabledArrays = enabledCount
	}

	if primitiveRestart && !g.Caps().UsePrimitiveRestart {
		panic("primitive restart without support")
	}
	if g.Caps().UsePrimitiveRestart &&
		(!s.enableStateIsValid || primitiveRestart != s.primitiveRestartEnabled) {
		if primitiveRestart {
			g.fns().Enable(PRIMITIVE_RESTART_FIXED_INDEX)
		} else {
			g.fns().Disable(PRIMITIVE_RESTART_FIXED_INDEX)
		}
		s.primitiveRestartEnabled = primitiveRestart
	}
	s.enableStateIsValid = true
}

// VertexArray is a GL vertex array object plus the tracking of its attrib and index buffer bindings.
type VertexArray struct {
	attribArrays        *AttribArrayState
	id                  uint32
	indexBufferUniqueID gpu.ResourceUniqueID
}

// NewVertexArray returns a new VertexArray wrapping the given GL vertex array object id.
func NewVertexArray(id uint32, attribCount int) *VertexArray {
	return &VertexArray{id: id, attribArrays: NewAttribArrayState(attribCount)}
}

// ArrayID returns the underlying GL vertex array object id.
func (v *VertexArray) ArrayID() uint32 { return v.id }

// Bind binds the VAO and returns its attrib state tracker, or nil if the id has been deleted or abandoned.
func (v *VertexArray) Bind(g *Gpu) *AttribArrayState {
	if v.id == 0 {
		return nil
	}
	g.BindVertexArray(v.id)
	return v.attribArrays
}

// BindWithIndexBuffer binds the VAO and its index buffer, returning the attrib state tracker.
func (v *VertexArray) BindWithIndexBuffer(g *Gpu, indexBuffer *Buffer) *AttribArrayState {
	state := v.Bind(g)
	if state == nil {
		return nil
	}
	if v.indexBufferUniqueID != indexBuffer.UniqueID() {
		g.fns().BindBuffer(ELEMENT_ARRAY_BUFFER, indexBuffer.BufferID())
		v.indexBufferUniqueID = indexBuffer.UniqueID()
	}
	return state
}

// InvalidateCachedState marks this VAO's attrib and index-buffer tracking as stale.
func (v *VertexArray) InvalidateCachedState() {
	v.attribArrays.Invalidate()
	v.indexBufferUniqueID = 0
}
