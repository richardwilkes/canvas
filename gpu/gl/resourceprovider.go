// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The factory that layers scratch and unique-key reuse over the raw Gpu creation entry points. Functions that return
// resources return them with a ref owned by the caller. Trims: compressed textures, semaphores, the backend-texture
// wrap paths, and the callback-initialized overload of FindOrMakeStaticBuffer are not implemented (semaphores are not
// publicly reachable). Cross-color-type conversion of initial texel data is deferred until the imagecore integration;
// the color types the provider is asked to upload today map straight through, and a mismatch fails creation just like
// an unsupported write.

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// ZeroInit selects whether a newly created buffer must be cleared to zero.
type ZeroInit bool

// ZeroInit values.
const (
	ZeroInitNo  ZeroInit = false
	ZeroInitYes ZeroInit = true
)

// ResourceProvider is the factory that layers scratch and unique-key reuse over the raw Gpu creation entry points.
type ResourceProvider struct {
	cache *gpu.ResourceCache
	gpu   *Gpu

	nonAAQuadIndexBuffer *Buffer
	aaQuadIndexBuffer    *Buffer

	// dynBufKeys memoizes the transient scratch keys CreateBuffer builds to look up a recyclable dynamic buffer, keyed
	// on the (binned size, type) that fully determines the key. CreateBuffer runs single-threaded per context and never
	// lets the cache retain the query key, so caching one built key per distinct request lets the size/type-stable
	// per-frame geometry-pool requests (a vertex pool and an index pool, both at the default block size) rebuild
	// nothing, avoiding a fresh heap-allocated key on every call. A first-seen request builds and caches its key
	// exactly as the un-memoized path did, so the memo is never worse. Lazily created; nil until the first
	// dynamic-buffer request.
	dynBufKeys map[dynBufKeyID]*gpu.ScratchKey
}

// dynBufKeyID is the (binned size, type) pair that uniquely determines a dynamic-buffer scratch key, used as the
// ResourceProvider.dynBufKeys memo key.
type dynBufKeyID struct {
	size uint64
	typ  gpu.BufferType
}

// NewResourceProvider builds a resource provider backed by g and cache.
func NewResourceProvider(g *Gpu, cache *gpu.ResourceCache) *ResourceProvider {
	return &ResourceProvider{cache: cache, gpu: g}
}

// Abandon drops the provider's references to its cache and GPU backend.
func (rp *ResourceProvider) Abandon() {
	rp.cache = nil
	rp.gpu = nil
}

// IsAbandoned reports whether the provider has been abandoned.
func (rp *ResourceProvider) IsAbandoned() bool { return rp.cache == nil }

// Caps returns the GL caps.
func (rp *ResourceProvider) Caps() *Caps { return rp.gpu.GLCaps() }

// OverBudget reports whether the resource cache is currently over its budget.
func (rp *ResourceProvider) OverBudget() bool { return rp.cache.OverBudget() }

// Gpu exposes the backend.
func (rp *ResourceProvider) Gpu() *Gpu { return rp.gpu }

// Cache exposes the resource cache.
func (rp *ResourceProvider) Cache() *gpu.ResourceCache { return rp.cache }

// FindByUniqueKey looks up a cached resource by unique key. The returned resource carries a ref owned by the caller.
func (rp *ResourceProvider) FindByUniqueKey(key *gpu.UniqueKey) gpu.Resource {
	if rp.IsAbandoned() {
		return nil
	}
	return rp.cache.FindAndRefUniqueResource(key)
}

// AssignUniqueKeyToResource assigns key to resource.
func (rp *ResourceProvider) AssignUniqueKeyToResource(key *gpu.UniqueKey, resource gpu.Resource) {
	if rp.IsAbandoned() || resource == nil {
		return
	}
	resource.(interface{ SetUniqueKey(*gpu.UniqueKey) }).SetUniqueKey(key)
}

// CreateTexture builds an exact-fit texture with no initial data, preferring an exactly matching scratch texture.
func (rp *ResourceProvider) CreateTexture(dims geom.ISize, format Format, textureType gpu.TextureType, renderable gpu.Renderable, renderTargetSampleCnt int, mipmapped gpu.Mipmapped, budgeted gpu.Budgeted, label string) *Surface {
	if rp.IsAbandoned() {
		return nil
	}
	if !rp.Caps().ValidateSurfaceParams(dims, format, bool(renderable), renderTargetSampleCnt,
		bool(mipmapped), textureType) {
		return nil
	}
	if tex := rp.getExactScratch(dims, format, textureType, renderable, renderTargetSampleCnt,
		budgeted, mipmapped, label); tex != nil {
		return tex
	}
	mipLevelCount := 1
	if mipmapped == gpu.MipmappedYes {
		mipLevelCount = gpu.ComputeMipLevelCount(dims.Width, dims.Height) + 1
	}
	return rp.gpu.CreateTexture(dims, format, textureType, renderable, renderTargetSampleCnt,
		budgeted, mipLevelCount, gpu.ColorTypeUnknown, gpu.ColorTypeUnknown, nil, label)
}

// CreateTextureWithData builds an exact-fit texture whose base level (or full mip chain) is uploaded from texels.
func (rp *ResourceProvider) CreateTextureWithData(dims geom.ISize, format Format, textureType gpu.TextureType, colorType gpu.ColorType, renderable gpu.Renderable, renderTargetSampleCnt int, budgeted gpu.Budgeted, mipmapped gpu.Mipmapped, texels []gpu.MipLevel, label string) *Surface {
	if rp.IsAbandoned() {
		return nil
	}
	numMipLevels := 1
	if mipmapped == gpu.MipmappedYes {
		numMipLevels = gpu.ComputeMipLevelCount(dims.Width, dims.Height) + 1
	}
	if !rp.Caps().ValidateSurfaceParams(dims, format, bool(renderable), renderTargetSampleCnt,
		bool(mipmapped), textureType) {
		return nil
	}
	// Current rule is that you can provide no level data, just the base, or all the levels.
	hasPixels := len(texels) > 0 && texels[0].Pixels != nil
	if scratch := rp.getExactScratch(dims, format, textureType, renderable, renderTargetSampleCnt,
		budgeted, mipmapped, label); scratch != nil {
		if !hasPixels {
			return scratch
		}
		return rp.writePixels(scratch, colorType, dims, texels, numMipLevels)
	}
	tempColorType := gpu.ColorTypeUnknown
	var tmpTexels []gpu.MipLevel
	if hasPixels {
		var ok bool
		tempColorType, tmpTexels, ok = rp.prepareLevels(format, colorType, dims, texels,
			numMipLevels)
		if !ok {
			return nil
		}
	}
	return rp.gpu.CreateTexture(dims, format, textureType, renderable, renderTargetSampleCnt,
		budgeted, numMipLevels, colorType, tempColorType, tmpTexels, label)
}

// getExactScratch finds a recyclable scratch texture matching the given description, making it unbudgeted if the
// request is unbudgeted.
func (rp *ResourceProvider) getExactScratch(dims geom.ISize, format Format, textureType gpu.TextureType, renderable gpu.Renderable, renderTargetSampleCnt int, budgeted gpu.Budgeted, mipmapped gpu.Mipmapped, label string) *Surface {
	tex := rp.FindAndRefScratchTexture(dims, format, textureType, renderable,
		renderTargetSampleCnt, mipmapped, label)
	if tex != nil && budgeted == gpu.BudgetedNo {
		tex.MakeUnbudgeted()
	}
	return tex
}

// CreateApproxTexture builds (or recycles) a texture at least as big as dims, rounding up to an approx size so it can
// be recycled by other requests of similar size.
func (rp *ResourceProvider) CreateApproxTexture(dims geom.ISize, format Format, textureType gpu.TextureType, renderable gpu.Renderable, renderTargetSampleCnt int, label string) *Surface {
	if rp.IsAbandoned() {
		return nil
	}
	if !rp.Caps().ValidateSurfaceParams(dims, format, bool(renderable), renderTargetSampleCnt,
		false, textureType) {
		return nil
	}
	copyDims := gpu.GetApproxSize(dims)
	if tex := rp.FindAndRefScratchTexture(copyDims, format, textureType, renderable,
		renderTargetSampleCnt, gpu.MipmappedNo, label); tex != nil {
		return tex
	}
	return rp.gpu.CreateTexture(copyDims, format, textureType, renderable, renderTargetSampleCnt,
		gpu.BudgetedYes, 1, gpu.ColorTypeUnknown, gpu.ColorTypeUnknown, nil, label)
}

// FindAndRefScratchTextureByKey looks up a recyclable scratch texture by its precomputed scratch key.
func (rp *ResourceProvider) FindAndRefScratchTextureByKey(key *gpu.ScratchKey, label string) *Surface {
	if rp.IsAbandoned() {
		panic("findAndRefScratchTexture on abandoned provider")
	}
	if !key.IsValid() {
		panic("findAndRefScratchTexture requires a valid key")
	}
	if resource := rp.cache.FindAndRefScratchResource(key); resource != nil {
		surface := resource.(*Surface)
		surface.SetLabel(label)
		return surface
	}
	return nil
}

// FindAndRefScratchTexture looks up a recyclable scratch texture matching the given description, computing its scratch
// key from the description.
func (rp *ResourceProvider) FindAndRefScratchTexture(dims geom.ISize, format Format, textureType gpu.TextureType, renderable gpu.Renderable, renderTargetSampleCnt int, mipmapped gpu.Mipmapped, label string) *Surface {
	if rp.IsAbandoned() {
		panic("findAndRefScratchTexture on abandoned provider")
	}
	if !rp.Caps().ValidateSurfaceParams(dims, format, bool(renderable), renderTargetSampleCnt,
		false, textureType) {
		panic("invalid scratch texture params")
	}
	// We could make initial clears work with scratch textures but it is a rare case so we just opt to fall back to
	// making a new texture.
	if rp.Caps().ReuseScratchTextures || renderable == gpu.RenderableYes {
		var key gpu.ScratchKey
		ComputeTextureScratchKey(rp.Caps(), format, dims, renderable, renderTargetSampleCnt,
			mipmapped, &key)
		return rp.FindAndRefScratchTextureByKey(&key, label)
	}
	return nil
}

// WrapBackendRenderTarget wraps a GL render target description as an instantiated Surface.
func (rp *ResourceProvider) WrapBackendRenderTarget(dims geom.ISize, format Format, sampleCnt, stencilBits int, fboID uint32) *Surface {
	if rp.IsAbandoned() {
		return nil
	}
	return rp.gpu.WrapBackendRenderTarget(dims, format, sampleCnt, stencilBits, fboID)
}

// FindOrMakeStaticBuffer returns the uniquely-keyed static buffer for key, creating and uploading it from staticData if
// not already cached.
func (rp *ResourceProvider) FindOrMakeStaticBuffer(intendedType gpu.BufferType, size uint64, staticData []byte, key *gpu.UniqueKey) *Buffer {
	if buffer, ok := rp.FindByUniqueKey(key).(*Buffer); ok && buffer != nil {
		return buffer
	}
	buffer := rp.CreateBufferWithData(staticData, size, intendedType, gpu.AccessPatternStatic)
	if buffer == nil {
		return nil
	}
	// We shouldn't bin and/or cache static buffers.
	if buffer.Size() != size || buffer.ScratchKey().IsValid() {
		panic("static buffer was binned or cached")
	}
	buffer.SetUniqueKey(key)
	return buffer
}

// createPatternedIndexBuffer builds a static index buffer by repeating pattern reps times, each repetition's indices
// offset by vertCount vertices.
func (rp *ResourceProvider) createPatternedIndexBuffer(pattern []uint16, patternSize, reps, vertCount int, key *gpu.UniqueKey) *Buffer {
	bufferSize := uint64(patternSize * reps * 2)
	buffer := rp.CreateBuffer(bufferSize, gpu.BufferTypeIndex, gpu.AccessPatternStatic,
		ZeroInitNo)
	if buffer == nil {
		return nil
	}
	data := make([]uint16, reps*patternSize)
	for i := 0; i < reps; i++ {
		baseIdx := i * patternSize
		baseVert := uint16(i * vertCount)
		for j := 0; j < patternSize; j++ {
			data[baseIdx+j] = baseVert + pattern[j]
		}
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*2)
	if !buffer.UpdateData(bytes, 0, false) {
		buffer.Unref()
		return nil
	}
	if key != nil {
		if !key.IsValid() {
			panic("invalid patterned index buffer key")
		}
		rp.AssignUniqueKeyToResource(key, buffer)
	}
	return buffer
}

// Quad index buffer geometry constants.
const (
	// MaxNumNonAAQuads is the maximum number of non-AA quads one index buffer covers (max possible: (1 << 14) - 1).
	MaxNumNonAAQuads = 1 << 12
	// NumVertsPerNonAAQuad is the vertex count per non-AA quad.
	NumVertsPerNonAAQuad = 4
	// NumIndicesPerNonAAQuad is the index count per non-AA quad.
	NumIndicesPerNonAAQuad = 6
	// MaxNumAAQuads is the maximum number of AA quads one index buffer covers (max possible: (1 << 13) - 1).
	MaxNumAAQuads = 1 << 9
	// NumVertsPerAAQuad is the vertex count per AA quad.
	NumVertsPerAAQuad = 8
	// NumIndicesPerAAQuad is the index count per AA quad.
	NumIndicesPerAAQuad = 30
)

// RefNonAAQuadIndexBuffer returns the shared index buffer for non-antialiased quads (6 indices per quad: 0, 1, 2, 2, 1,
// 3; draw with triangles). The returned buffer is owned by the provider; callers ref it if they retain it.
func (rp *ResourceProvider) RefNonAAQuadIndexBuffer() *Buffer {
	if rp.nonAAQuadIndexBuffer == nil {
		pattern := []uint16{0, 1, 2, 2, 1, 3}
		rp.nonAAQuadIndexBuffer = rp.createPatternedIndexBuffer(pattern, NumIndicesPerNonAAQuad,
			MaxNumNonAAQuads, NumVertsPerNonAAQuad, nil)
	}
	return rp.nonAAQuadIndexBuffer
}

// RefAAQuadIndexBuffer returns the shared index buffer for antialiased quads (30 indices and 8 vertices per quad; draw
// with triangles).
func (rp *ResourceProvider) RefAAQuadIndexBuffer() *Buffer {
	if rp.aaQuadIndexBuffer == nil {
		pattern := []uint16{
			0, 1, 2, 1, 3, 2,
			0, 4, 1, 4, 5, 1,
			0, 6, 4, 0, 2, 6,
			2, 3, 6, 3, 7, 6,
			1, 5, 3, 3, 5, 7,
		}
		rp.aaQuadIndexBuffer = rp.createPatternedIndexBuffer(pattern, NumIndicesPerAAQuad,
			MaxNumAAQuads, NumVertsPerAAQuad, nil)
	}
	return rp.aaQuadIndexBuffer
}

// CreateBuffer builds a buffer of the given size and type. Dynamic buffers are binned by pow2+midpoint sizes and
// recycled through the scratch map.
func (rp *ResourceProvider) CreateBuffer(size uint64, intendedType gpu.BufferType, accessPattern gpu.AccessPattern, zeroInit ZeroInit) *Buffer {
	if rp.IsAbandoned() {
		return nil
	}
	if accessPattern != gpu.AccessPatternDynamic {
		if rp.Caps().BuffersAreInitiallyZero {
			zeroInit = ZeroInitNo
		}
		buffer := rp.gpu.CreateBuffer(size, intendedType, accessPattern)
		if buffer != nil && zeroInit == ZeroInitYes && !buffer.ClearToZero() {
			buffer.Unref()
			return nil
		}
		return buffer
	}
	// Bin by pow2+midpoint with a reasonable min.
	const minSize = 1 << 12
	const minUniformSize = 1 << 7
	allocSize := size
	if intendedType == gpu.BufferTypeUniform {
		allocSize = max(allocSize, minUniformSize)
	} else {
		allocSize = max(allocSize, minSize)
	}
	// Round up to the next power of two.
	ceilPow2 := uint64(1)
	for ceilPow2 < allocSize {
		ceilPow2 <<= 1
	}
	floorPow2 := ceilPow2 >> 1
	mid := floorPow2 + (floorPow2 >> 1)
	if allocSize <= mid {
		allocSize = mid
	} else {
		allocSize = ceilPow2
	}

	// Reuse the transient lookup key for this (size, type) across calls (see the dynBufKeys field comment); a
	// first-seen request builds and caches it, which is what the un-memoized path did on every call.
	id := dynBufKeyID{size: allocSize, typ: intendedType}
	key := rp.dynBufKeys[id]
	if key == nil {
		key = &gpu.ScratchKey{}
		ComputeScratchKeyForDynamicBuffer(allocSize, intendedType, key)
		if rp.dynBufKeys == nil {
			rp.dynBufKeys = make(map[dynBufKeyID]*gpu.ScratchKey)
		}
		rp.dynBufKeys[id] = key
	}
	var buffer *Buffer
	if resource := rp.cache.FindAndRefScratchResource(key); resource != nil {
		buffer = resource.(*Buffer)
	} else {
		if rp.Caps().BuffersAreInitiallyZero {
			zeroInit = ZeroInitNo
		}
		buffer = rp.gpu.CreateBuffer(allocSize, intendedType, gpu.AccessPatternDynamic)
	}
	if buffer != nil && zeroInit == ZeroInitYes && !buffer.ClearToZero() {
		buffer.Unref()
		return nil
	}
	return buffer
}

// CreateBufferWithData builds a buffer of the given size and type, uploaded from data.
func (rp *ResourceProvider) CreateBufferWithData(data []byte, size uint64, intendedType gpu.BufferType, pattern gpu.AccessPattern) *Buffer {
	if data == nil {
		panic("createBuffer requires data")
	}
	buffer := rp.CreateBuffer(size, intendedType, pattern, ZeroInitNo)
	if buffer == nil {
		return nil
	}
	if !buffer.UpdateData(data[:size], 0, false) {
		buffer.Unref()
		return nil
	}
	return buffer
}

// AttachStencilAttachment returns true if the render target already has a stencil attachment on the given surface, else
// attempts to attach one (shared between same-shaped render targets via the shared attachment unique key). Under DMSAA
// (numSamples == 1 with useMSAASurface) the stencil buffer is allocated at the internal multisample count.
func (rp *ResourceProvider) AttachStencilAttachment(rt *RenderTarget, useMSAASurface bool) bool {
	if rp.Caps().AvoidStencilBuffers {
		panic("attachStencilAttachment with avoidStencilBuffers")
	}
	if stencil := rt.StencilAttachment(useMSAASurface); stencil != nil {
		return true
	}
	if !rt.Surface().WasDestroyed() && rt.CanAttemptStencilAttachment(useMSAASurface) {
		stencilFormat, ok := rp.gpu.PreferredStencilFormat(rt.GLFormat())
		if !ok {
			return false
		}
		// Under dynamic MSAA the stencil buffer matches the internal multisample count (the caller ensured DMSAA
		// support before using it).
		numStencilSamples := rt.NumSamples()
		if numStencilSamples == 1 && useMSAASurface {
			numStencilSamples = rp.Caps().InternalMultisampleCount(rt.GLFormat())
			if numStencilSamples <= 1 {
				panic("DMSAA stencil attachment requested without DMSAA support")
			}
		}
		var sbKey gpu.UniqueKey
		ComputeSharedAttachmentUniqueKey(rp.Caps(), stencilFormat, rt.Surface().Dimensions(),
			AttachmentUsageStencil, numStencilSamples, &sbKey)
		keyedStencil, _ := rp.FindByUniqueKey(&sbKey).(*Attachment)
		if keyedStencil == nil {
			// Need to try and create a new stencil.
			keyedStencil = rp.gpu.MakeStencilAttachment(rt.GLFormat(), rt.Surface().Dimensions(),
				numStencilSamples)
			if keyedStencil == nil {
				return false
			}
			rp.AssignUniqueKeyToResource(&sbKey, keyedStencil)
		}
		// The ref from find/create moves into the render target.
		rt.AttachStencilAttachment(keyedStencil, useMSAASurface)
	}
	return rt.StencilAttachment(useMSAASurface) != nil
}

// GetDiscardableMSAAAttachment returns the MSAA color renderbuffer DMSAA render targets share, keyed by the shared
// attachment unique key so same-shaped render targets reuse one attachment (memoryless is a Vulkan/Metal-only concept
// and is trimmed). The caller owns the returned ref.
func (rp *ResourceProvider) GetDiscardableMSAAAttachment(dims geom.ISize, format Format, sampleCnt int) *Attachment {
	if sampleCnt <= 1 {
		panic("discardable MSAA attachment requires >1 sample")
	}
	if rp.IsAbandoned() {
		return nil
	}
	if !rp.Caps().ValidateSurfaceParams(dims, format, true, sampleCnt, false,
		gpu.TextureTypeNone) {
		return nil
	}

	var key gpu.UniqueKey
	ComputeSharedAttachmentUniqueKey(rp.Caps(), format, dims, AttachmentUsageColor, sampleCnt,
		&key)
	if attachment, ok := rp.FindByUniqueKey(&key).(*Attachment); ok && attachment != nil {
		return attachment
	}
	attachment := rp.MakeMSAAAttachment(dims, format, sampleCnt)
	if attachment != nil {
		rp.AssignUniqueKeyToResource(&key, attachment)
	}
	return attachment
}

// MakeMSAAAttachment builds (or recycles) an MSAA color renderbuffer attachment (memoryless is a Vulkan/Metal-only
// concept and is trimmed).
func (rp *ResourceProvider) MakeMSAAAttachment(dims geom.ISize, format Format, sampleCnt int) *Attachment {
	if sampleCnt <= 1 {
		panic("MSAA attachment requires >1 sample")
	}
	if rp.IsAbandoned() {
		return nil
	}
	if !rp.Caps().ValidateSurfaceParams(dims, format, true, sampleCnt, false,
		gpu.TextureTypeNone) {
		return nil
	}
	if scratch := rp.refScratchMSAAAttachment(dims, format, sampleCnt,
		"MakeMSAAAttachment"); scratch != nil {
		return scratch
	}
	return rp.gpu.MakeMSAAAttachment(dims, format, sampleCnt)
}

// refScratchMSAAAttachment looks up a recyclable scratch MSAA attachment matching the given description.
func (rp *ResourceProvider) refScratchMSAAAttachment(dims geom.ISize, format Format, sampleCnt int, label string) *Attachment {
	var key gpu.ScratchKey
	ComputeAttachmentScratchKey(rp.Caps(), format, dims, AttachmentUsageColor, sampleCnt, &key)
	if resource := rp.cache.FindAndRefScratchResource(&key); resource != nil {
		attachment := resource.(*Attachment)
		attachment.SetLabel(label)
		return attachment
	}
	return nil
}

// prepareLevels prepares mip level data for upload: row bytes are populated (never 0) and tightened by copying when the
// backend lacks row-byte support; a color type the format cannot accept directly fails (cross-color-type conversion
// arrives with the image integration). Returns the write color type, the prepared levels, and success.
func (rp *ResourceProvider) prepareLevels(format Format, colorType gpu.ColorType, baseSize geom.ISize, texels []gpu.MipLevel, mipLevelCount int) (gpu.ColorType, []gpu.MipLevel, bool) {
	if mipLevelCount == 0 || len(texels) == 0 || texels[0].Pixels == nil {
		panic("prepareLevels requires base level data")
	}
	allowedColorType := rp.Caps().SupportedWritePixelsColorType(colorType, format,
		colorType).ColorType
	if allowedColorType != colorType {
		// Cross-color-type conversion is not yet implemented, so treat it as an unsupported write.
		return gpu.ColorTypeUnknown, nil, false
	}
	rowBytesSupport := rp.Caps().WritePixelsRowBytesSupport
	out := make([]gpu.MipLevel, mipLevelCount)
	size := baseSize
	for i := 0; i < mipLevelCount; i++ {
		if i < len(texels) && texels[i].Pixels != nil {
			minRB := int(size.Width) * colorType.BytesPerPixel()
			actualRB := texels[i].RowBytes
			if actualRB == 0 {
				actualRB = minRB
			}
			if actualRB < minRB {
				return gpu.ColorTypeUnknown, nil, false
			}
			if actualRB == minRB || rowBytesSupport {
				out[i] = gpu.MipLevel{Pixels: texels[i].Pixels, RowBytes: actualRB}
			} else {
				// Copy to a tight temporary. The source must cover every row it declares: the last row starts at
				// actualRB*(height-1) and spans minRB bytes, so a Pixels buffer shorter than that is an unsupported
				// write rather than an out-of-range slice panic.
				if len(texels[i].Pixels) < actualRB*(int(size.Height)-1)+minRB {
					return gpu.ColorTypeUnknown, nil, false
				}
				tight := make([]byte, minRB*int(size.Height))
				for y := 0; y < int(size.Height); y++ {
					copy(tight[y*minRB:(y+1)*minRB], texels[i].Pixels[y*actualRB:])
				}
				out[i] = gpu.MipLevel{Pixels: tight, RowBytes: minRB}
			}
		}
		size = geom.ISize{Width: max(size.Width/2, 1), Height: max(size.Height/2, 1)}
	}
	return allowedColorType, out, true
}

// writePixels fills a (possibly scratch-recycled) texture with the provided level data.
func (rp *ResourceProvider) writePixels(texture *Surface, colorType gpu.ColorType, baseSize geom.ISize, texels []gpu.MipLevel, mipLevelCount int) *Surface {
	if rp.IsAbandoned() || texture == nil || colorType == gpu.ColorTypeUnknown {
		panic("writePixels preconditions violated")
	}
	tempColorType, tmpTexels, ok := rp.prepareLevels(texture.Format(), colorType, baseSize, texels,
		mipLevelCount)
	if !ok {
		texture.Unref()
		return nil
	}
	if !rp.gpu.WritePixels(texture, geom.IRectSize(baseSize), colorType, tempColorType, tmpTexels,
		false) {
		// A backend upload failure is recoverable: fail the texture creation the same way the prepareLevels failure
		// above does, rather than aborting the process.
		texture.Unref()
		return nil
	}
	return texture
}
