// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Shared GPU type enums used by the resource layer. Some orderings are fixed because tables index by value. Protected
// content is out of scope for this codebase, so no Protected type is defined.

package gpu

// Budgeted describes whether a resource counts against the resource cache budget.
type Budgeted bool

// Budgeted values.
const (
	BudgetedNo  Budgeted = false
	BudgetedYes Budgeted = true
)

// Mipmapped describes whether a texture has mip levels.
type Mipmapped bool

// Mipmapped values.
const (
	MipmappedNo  Mipmapped = false
	MipmappedYes Mipmapped = true
)

// BudgetedType refines Budgeted with the reasons a resource may be unbudgeted.
type BudgetedType uint8

// BudgetedType values.
const (
	// BudgetedTypeBudgeted: the resource is budgeted and subject to purging under budget pressure.
	BudgetedTypeBudgeted BudgetedType = iota
	// BudgetedTypeUnbudgetedUncacheable: unbudgeted; purged as soon as it has no refs regardless of whether it has a
	// unique or scratch key.
	BudgetedTypeUnbudgetedUncacheable
	// BudgetedTypeUnbudgetedCacheable: unbudgeted; allowed to remain in the cache with no refs if it has a unique key
	// (scratch keys are ignored). Only wrapped resources are in this state.
	BudgetedTypeUnbudgetedCacheable
)

// WrapCacheable describes whether a wrapped backend object should stay cached with a unique key after its last external
// ref drops.
type WrapCacheable bool

// WrapCacheable values.
const (
	WrapCacheableNo  WrapCacheable = false
	WrapCacheableYes WrapCacheable = true
)

// Renderable describes whether a surface can be used as a render target.
type Renderable bool

// Renderable values.
const (
	RenderableNo  Renderable = false
	RenderableYes Renderable = true
)

// BufferType identifies the role a GPU buffer plays (vertex data, index data, and so on).
type BufferType int32

// BufferType values.
const (
	BufferTypeVertex BufferType = iota
	BufferTypeIndex
	BufferTypeDrawIndirect
	BufferTypeXferCpuToGpu
	BufferTypeXferGpuToCpu
	BufferTypeUniform

	// BufferTypeCount is the number of valid BufferType values.
	BufferTypeCount = int(BufferTypeUniform) + 1
)

// AccessPattern is a performance hint regarding the frequency at which a data store will be accessed.
type AccessPattern int32

// AccessPattern values.
const (
	// AccessPatternDynamic: the data store will be respecified repeatedly and used many times.
	AccessPatternDynamic AccessPattern = iota
	// AccessPatternStatic: specified once and used many times (disqualified from scratch caching).
	AccessPatternStatic
	// AccessPatternStream: specified once and used at most a few times (also can't be cached).
	AccessPatternStream
)

// IOType describes whether an access is a read, a write, or both.
type IOType int32

// IOType values.
const (
	IOTypeRead IOType = iota
	IOTypeWrite
	IOTypeRW
)

// MipmapStatus tracks how far along a texture's mip chain is toward being fully valid.
type MipmapStatus int32

// MipmapStatus values.
const (
	// MipmapStatusNotAllocated: mips have not been allocated.
	MipmapStatusNotAllocated MipmapStatus = iota
	// MipmapStatusDirty: mips are allocated but the full mip tree does not have valid data.
	MipmapStatusDirty
	// MipmapStatusValid: all levels fully allocated with valid data.
	MipmapStatusValid
)

// SurfaceFlags are internal per-surface flags tracked by the GPU resource layer.
type SurfaceFlags uint32

// SurfaceFlags values.
const (
	// SurfaceFlagReadOnly means the pixels in the texture are read-only.
	SurfaceFlagReadOnly SurfaceFlags = 1 << 0
	// SurfaceFlagGLRTFBOIDIs0 tells us that the internal render target wraps FBO 0.
	SurfaceFlagGLRTFBOIDIs0 SurfaceFlags = 1 << 1
	// SurfaceFlagRequiresManualMSAAResolve means the render target is multisampled and internally holds a non-msaa
	// texture for resolving into.
	SurfaceFlagRequiresManualMSAAResolve SurfaceFlags = 1 << 2
	// SurfaceFlagFramebufferOnly means the pixels in the render target are write-only.
	SurfaceFlagFramebufferOnly SurfaceFlags = 1 << 3
)

// Ownership describes whether the holder of a backend object is responsible for destroying it.
type Ownership bool

// Ownership values.
const (
	// OwnershipBorrowed: the holder does not destroy the backend object.
	OwnershipBorrowed Ownership = false
	// OwnershipOwned: the holder destroys the backend object.
	OwnershipOwned Ownership = true
)

// MipLevel is one level of texel data for an upload. A nil Pixels skips the level.
type MipLevel struct {
	Pixels   []byte
	RowBytes int
}

// ComputeMipLevelCount returns the number of mip levels *below* the base level, so a 1x1 texture has 0. Lives here
// because the resource layer sizes mip chains with it.
func ComputeMipLevelCount(width, height int32) int {
	if width < 1 || height < 1 {
		return 0
	}
	largest := uint32(width)
	if uint32(height) > largest {
		largest = uint32(height)
	}
	count := 0
	for largest > 1 {
		largest >>= 1
		count++
	}
	return count
}

// LoadOp describes how a render pass sources the prior contents of its attachments.
type LoadOp int32

// LoadOp values.
const (
	LoadOpLoad LoadOp = iota
	LoadOpClear
	LoadOpDiscard
)

// StoreOp describes what happens to an attachment's contents after a render pass finishes.
type StoreOp int32

// StoreOp values.
const (
	StoreOpStore StoreOp = iota
	StoreOpDiscard
)

// XferBarrierFlags are barriers a render pass must support between draws.
type XferBarrierFlags uint32

// XferBarrierFlags values.
const (
	XferBarrierFlagTexture XferBarrierFlags = 1 << 0
	XferBarrierFlagBlend   XferBarrierFlags = 1 << 1
)

// DstSampleFlags describe how the destination is sampled when a draw reads it.
type DstSampleFlags uint32

// DstSampleFlags values.
const (
	// DstSampleFlagRequiresTextureBarrier: reading the dst as a texture needs a barrier between draws.
	DstSampleFlagRequiresTextureBarrier DstSampleFlags = 1 << 0
)
