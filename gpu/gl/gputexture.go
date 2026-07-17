// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Texture/render-target creation and upload, with validation folded in: CreateTexture, createRenderTargetObjects,
// uploadColorTypeTexData/uploadTexData/uploadColorToTexZeros, WrapBackendRenderTarget, getCompatibleStencilIndex,
// MakeStencilAttachment/MakeMSAAAttachment, RegenerateMipmapLevels, bindSurfaceFBOForPixelOps. Trims: compressed
// textures, protected content, and backend-texture creation are out of scope; the manual-mipmap draw path
// (DoManualMipmapping — old Mac Intel) runs through the programs in mipmapprogram.go; uploadColorToTexZeros supports
// only the transparent-black clear color the texture clear mask needs (arbitrary clear colors are not implemented).

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// allocCall runs a GL allocation call and returns the error state, respecting skipErrorChecks.
func (g *Gpu) allocCall(call func(f *Functions)) uint32 {
	if g.glCaps().SkipErrorChecks() {
		call(g.fns())
		return NO_ERROR
	}
	g.clearErrorsAndCheckForOOM()
	call(g.fns())
	return g.getErrorAndCheckForOOM()
}

// setInitialTextureParams sets baseline filter/wrap state on target: some drivers like to know filter/wrap before
// seeing glTexImage2D; some have a bug where an FBO won't be complete if it includes a texture that is not mipmap
// complete (considering the filter in use).
func (g *Gpu) setInitialTextureParams(target uint32) SamplerOverriddenState {
	state := MakeSamplerOverriddenState()
	state.MinFilter = NEAREST
	state.MagFilter = NEAREST
	state.WrapS = CLAMP_TO_EDGE
	state.WrapT = CLAMP_TO_EDGE
	f := g.fns()
	f.TexParameteri(target, TEXTURE_MAG_FILTER, int32(state.MagFilter))
	f.TexParameteri(target, TEXTURE_MIN_FILTER, int32(state.MinFilter))
	f.TexParameteri(target, TEXTURE_WRAP_S, int32(state.WrapS))
	f.TexParameteri(target, TEXTURE_WRAP_T, int32(state.WrapT))
	return state
}

// createTexture creates and allocates a GL texture object, leaving it bound to the scratch texture unit. Returns 0 on
// failure.
func (g *Gpu) createTexture(dims geom.ISize, format Format, target uint32, initialState *SamplerOverriddenState, mipLevelCount int, label string) uint32 {
	if format == FormatUnknown || format.IsCompressed() {
		panic("invalid texture format")
	}
	var id uint32
	g.fns().GenTextures(1, &id)
	if id == 0 {
		return 0
	}

	g.bindTextureToScratchUnit(target, id)

	if g.glCaps().DebugSupport() {
		if label == "" {
			label = "canvas"
		}
		g.fns().ObjectLabel(TEXTURE, id, -1, cString(label))
	}

	if initialState != nil {
		*initialState = g.setInitialTextureParams(target)
	} else {
		g.setInitialTextureParams(target)
	}

	internalFormat := g.glCaps().TexImageOrStorageInternalFormat(format)
	success := false
	if internalFormat != 0 {
		if g.glCaps().FormatSupportsTexStorage(format) {
			levelCount := int32(max(mipLevelCount, 1))
			err := g.allocCall(func(f *Functions) {
				f.TexStorage2D(target, levelCount, internalFormat, dims.Width, dims.Height)
			})
			success = err == NO_ERROR
		} else {
			externalFormat, externalType := g.glCaps().TexImageExternalFormatAndType(format)
			if externalFormat != 0 && externalType != 0 {
				// If we don't unbind here then nullptr is treated as a zero offset into the bound transfer buffer
				// rather than an indication that there is no data to copy.
				g.unbindXferBuffer(gpu.BufferTypeXferCpuToGpu)
				err := uint32(NO_ERROR)
				for level := 0; level < mipLevelCount && err == NO_ERROR; level++ {
					w := max(dims.Width>>level, 1)
					h := max(dims.Height>>level, 1)
					err = g.allocCall(func(f *Functions) {
						f.TexImage2D(target, int32(level), int32(internalFormat), w, h, 0,
							externalFormat, externalType, 0)
					})
				}
				success = err == NO_ERROR
			}
		}
	}
	if success {
		return id
	}
	g.fns().DeleteTextures(1, &id)
	return 0
}

// createRenderTargetObjects builds the FBO(s) (and MSAA color renderbuffer if needed) for a renderable texture.
func (g *Gpu) createRenderTargetObjects(desc *TextureDesc, sampleCount int, ids *RenderTargetIDs) bool {
	f := g.fns()
	*ids = RenderTargetIDs{RTFBOOwnership: gpu.OwnershipOwned}

	cleanupOnFail := func() {
		if ids.MSColorRenderbufferID != 0 {
			f.DeleteRenderbuffers(1, &ids.MSColorRenderbufferID)
		}
		if ids.MultisampleFBOID != ids.SingleSampleFBOID && ids.MultisampleFBOID != 0 {
			g.DeleteFramebuffer(ids.MultisampleFBOID)
		}
		if ids.SingleSampleFBOID != 0 {
			g.DeleteFramebuffer(ids.SingleSampleFBOID)
		}
	}

	if desc.Format == FormatUnknown {
		return false
	}
	if sampleCount > 1 && g.glCaps().MSFBOType() == MSFBONone {
		return false
	}

	f.GenFramebuffers(1, &ids.SingleSampleFBOID)
	if ids.SingleSampleFBOID == 0 {
		return false
	}

	// If we are using multisampling we will create two FBOs: we render to one and then resolve to the texture bound to
	// the other. (The implicit-resolve extensions are ES-only and dropped.)
	var colorRenderbufferFormat uint32
	if sampleCount <= 1 {
		ids.MultisampleFBOID = UnresolvableFBOID
	} else {
		f.GenFramebuffers(1, &ids.MultisampleFBOID)
		if ids.MultisampleFBOID == 0 {
			cleanupOnFail()
			return false
		}
		f.GenRenderbuffers(1, &ids.MSColorRenderbufferID)
		if ids.MSColorRenderbufferID == 0 {
			cleanupOnFail()
			return false
		}
		colorRenderbufferFormat = g.glCaps().RenderbufferInternalFormat(desc.Format)
	}

	// Below here we may bind the FBO.
	g.hwBoundRenderTargetUniqueID = 0
	if ids.MSColorRenderbufferID != 0 {
		f.BindRenderbuffer(RENDERBUFFER, ids.MSColorRenderbufferID)
		if !g.renderbufferStorageMSAA(sampleCount, colorRenderbufferFormat,
			desc.Size.Width, desc.Size.Height) {
			cleanupOnFail()
			return false
		}
		g.BindFramebuffer(FRAMEBUFFER, ids.MultisampleFBOID)
		f.FramebufferRenderbuffer(FRAMEBUFFER, COLOR_ATTACHMENT0, RENDERBUFFER,
			ids.MSColorRenderbufferID)
		if !g.glCaps().SkipErrorChecks() {
			if f.CheckFramebufferStatus(FRAMEBUFFER) != FRAMEBUFFER_COMPLETE {
				cleanupOnFail()
				return false
			}
		}
		ids.TotalMemorySamplesPerPixel += sampleCount
	}
	g.BindFramebuffer(FRAMEBUFFER, ids.SingleSampleFBOID)
	f.FramebufferTexture2D(FRAMEBUFFER, COLOR_ATTACHMENT0, desc.Target, desc.ID, 0)
	if !g.glCaps().SkipErrorChecks() {
		if f.CheckFramebufferStatus(FRAMEBUFFER) != FRAMEBUFFER_COMPLETE {
			cleanupOnFail()
			return false
		}
	}
	ids.TotalMemorySamplesPerPixel++
	return true
}

// validateTexelLevels checks that the given mip levels are internally consistent: row bytes match what the driver
// requires, dimensions halve correctly down to 1x1, and pixel data is present either for just the base level, all
// levels, or no levels.
func (g *Gpu) validateTexelLevels(dims geom.ISize, texelColorType gpu.ColorType, texels []gpu.MipLevel) bool {
	if len(texels) == 0 {
		panic("no texel levels")
	}
	hasBasePixels := texels[0].Pixels != nil
	levelsWithPixelsCnt := 0
	bpp := texelColorType.BytesPerPixel()
	w := dims.Width
	h := dims.Height
	for currentMipLevel := range texels {
		if texels[currentMipLevel].Pixels != nil {
			minRowBytes := int(w) * bpp
			if g.Caps().WritePixelsRowBytesSupport {
				if texels[currentMipLevel].RowBytes < minRowBytes {
					return false
				}
				if texels[currentMipLevel].RowBytes%bpp != 0 {
					return false
				}
			} else if texels[currentMipLevel].RowBytes != minRowBytes {
				return false
			}
			levelsWithPixelsCnt++
		}
		if w == 1 && h == 1 {
			if currentMipLevel != len(texels)-1 {
				return false
			}
		} else {
			w = max(w/2, 1)
			h = max(h/2, 1)
		}
	}
	// Either just a base layer or a full stack is required.
	if len(texels) != 1 && (w != 1 || h != 1) {
		return false
	}
	// Can specify just the base, all levels, or no levels.
	if !hasBasePixels {
		return levelsWithPixelsCnt == 0
	}
	return levelsWithPixelsCnt == 1 || levelsWithPixelsCnt == len(texels)
}

// CreateTexture validates the request, creates the GL texture object, creates render target objects when renderable,
// and computes the level-clear mask. texels may be nil (uninitialized) or hold initial contents for level 0 or the full
// mip chain.
func (g *Gpu) CreateTexture(dims geom.ISize, format Format, textureType gpu.TextureType, renderable gpu.Renderable, renderTargetSampleCnt int, budgeted gpu.Budgeted, mipLevelCount int, textureColorType, srcColorType gpu.ColorType, texels []gpu.MipLevel, label string) *Surface {
	if len(texels) > 0 {
		if !g.validateTexelLevels(dims, srcColorType, texels) {
			return nil
		}
		if mipLevelCount != len(texels) {
			panic("mip level count mismatch")
		}
	}
	if mipLevelCount < 1 {
		panic("mipLevelCount must be >= 1")
	}

	// Determine which mip levels need clearing: levels with no supplied pixel data, when the caps require newly created
	// textures to start out cleared.
	var levelClearMask uint32
	if g.Caps().ShouldInitializeTextures {
		if len(texels) > 0 {
			for i := range texels {
				if texels[i].Pixels == nil {
					levelClearMask |= 1 << i
				}
			}
		} else {
			levelClearMask = (1 << mipLevelCount) - 1
		}
	}

	// createTextureCommon: caps validation.
	mipmapped := gpu.MipmappedNo
	if mipLevelCount > 1 {
		mipmapped = gpu.MipmappedYes
	}
	if !g.glCaps().ValidateSurfaceParams(dims, format, renderable == gpu.RenderableYes,
		renderTargetSampleCnt, mipmapped == gpu.MipmappedYes, textureType) {
		return nil
	}
	if renderable == gpu.RenderableYes {
		renderTargetSampleCnt = g.glCaps().GetRenderTargetSampleCount(renderTargetSampleCnt, format)
	}
	g.handleDirtyContext()

	// Set up the texture descriptor and allocate the GL texture object.
	mipmapStatus := gpu.MipmapStatusNotAllocated
	if mipLevelCount > 1 {
		mipmapStatus = gpu.MipmapStatusDirty
	}
	texDesc := TextureDesc{
		Size:      dims,
		Format:    format,
		Ownership: gpu.OwnershipOwned,
	}
	switch textureType {
	case gpu.TextureType2D:
		texDesc.Target = TEXTURE_2D
	case gpu.TextureTypeRectangle:
		if mipLevelCount > 1 || !g.glCaps().RectangleTextureSupport() {
			return nil
		}
		texDesc.Target = TEXTURE_RECTANGLE
	default:
		return nil
	}
	var initialState SamplerOverriddenState
	texDesc.ID = g.createTexture(dims, format, texDesc.Target, &initialState, mipLevelCount, label)
	if texDesc.ID == 0 {
		return nil
	}

	var tex *Surface
	if renderable == gpu.RenderableYes {
		// Unbind the texture from the texture unit before binding it to the frame buffer.
		g.fns().BindTexture(texDesc.Target, 0)
		var rtIDs RenderTargetIDs
		if !g.createRenderTargetObjects(&texDesc, renderTargetSampleCnt, &rtIDs) {
			g.fns().DeleteTextures(1, &texDesc.ID)
			return nil
		}
		tex = NewTextureRenderTarget(g, budgeted, renderTargetSampleCnt, &texDesc, &rtIDs,
			mipmapStatus, label)
		tex.AsTexture().BaseLevelWasBoundToFBO()
	} else {
		tex = NewTexture(g, budgeted, &texDesc, mipmapStatus, label)
	}
	// The non-sampler params are still at their default values.
	tex.AsTexture().Parameters().Set(&initialState, MakeNonsamplerState(),
		g.resetTimestampForTextureParameters)

	if levelClearMask != 0 {
		g.clearTextureLevels(tex, &texDesc, mipLevelCount, levelClearMask)
	}

	// Finish up: scratch-key policy, MSAA-resolve flag, initial upload.
	if !g.Caps().ReuseScratchTextures && renderable == gpu.RenderableNo {
		tex.RemoveScratchKey()
	}
	if renderTargetSampleCnt > 1 && !g.Caps().MSAAResolvesAutomatically {
		tex.SetRequiresManualMSAAResolve()
	}

	markMipLevelsClean := false
	if len(texels) > 0 && texels[0].Pixels != nil {
		if !g.WritePixels(tex, geom.IRectSize(dims), textureColorType, srcColorType, texels, false) {
			tex.Unref()
			return nil
		}
		// If level[1] of the mip map has pixel data then so must all other levels.
		markMipLevelsClean = len(texels) > 1 && levelClearMask == 0 && texels[1].Pixels != nil
	} else if levelClearMask != 0 && mipLevelCount > 1 {
		markMipLevelsClean = true
	}
	if markMipLevelsClean {
		tex.AsTexture().MarkMipmapsClean()
	}
	return tex
}

// clearTextureLevels clears the mip levels marked in levelClearMask, using the fastest method the caps support:
// ClearTexImage, an FBO clear, or an upload of zeroed pixel data.
func (g *Gpu) clearTextureLevels(tex *Surface, texDesc *TextureDesc, mipLevelCount int, levelClearMask uint32) {
	switch {
	case g.glCaps().ClearTextureSupport():
		externalFormat, externalType, colorType := g.glCaps().
			TexSubImageDefaultFormatTypeAndColorType(texDesc.Format)
		_ = colorType
		for i := 0; i < mipLevelCount; i++ {
			if levelClearMask&(1<<i) != 0 {
				g.fns().ClearTexImage(tex.AsTexture().TextureID(), int32(i), externalFormat,
					externalType, 0)
			}
		}
	case g.glCaps().CanFormatBeFBOColorAttachment(texDesc.Format) &&
		!g.Caps().PerformColorClearsAsDraws:
		g.flushScissorTest(false)
		g.disableWindowRectangles()
		g.flushColorWrite(true)
		g.flushClearColor([4]float32{0, 0, 0, 0})
		for i := 0; i < mipLevelCount; i++ {
			if levelClearMask&(1<<i) != 0 {
				g.bindSurfaceFBOForPixelOps(tex, i, FRAMEBUFFER, tempFBODst)
				g.fns().Clear(COLOR_BUFFER_BIT)
				g.unbindSurfaceFBOForPixelOps(tex, i, FRAMEBUFFER)
			}
		}
		g.hwBoundRenderTargetUniqueID = 0
	default:
		g.bindTextureToScratchUnit(texDesc.Target, tex.AsTexture().TextureID())
		g.uploadColorToTexZeros(texDesc.Format, texDesc.Size, texDesc.Target, levelClearMask)
	}
}

// WritePixels validates the request and uploads texel data to a texture-backed surface.
func (g *Gpu) WritePixels(surface *Surface, rect geom.IRect, surfaceColorType, srcColorType gpu.ColorType, texels []gpu.MipLevel, prepForTexSampling bool) bool {
	_ = prepForTexSampling // consumed by Vulkan; kept for signature parity
	if surface.FramebufferOnly() {
		panic("write to framebuffer-only surface")
	}
	if surface.ReadOnly() {
		return false
	}
	mipLevelCount := len(texels)
	switch {
	case mipLevelCount == 0:
		return false
	case mipLevelCount == 1:
		// We require that if we are not mipped, then the write region is contained in the surface.
		if !geom.IRectSize(surface.Dimensions()).ContainsRect(rect) {
			return false
		}
	case rect != geom.IRectSize(surface.Dimensions()):
		// We require that if the texels are mipped, then the write region is the entire surface.
		return false
	}
	if !g.validateTexelLevels(rect.Size(), srcColorType, texels) {
		return false
	}
	g.handleDirtyContext()

	// Bind the target texture and upload.
	glTex := surface.AsTexture()
	if glTex == nil {
		return false
	}
	g.bindTextureToScratchUnit(glTex.Target(), glTex.TextureID())

	// If we have mips make sure the base/max levels cover the full range so that the uploads go to the right levels
	// (some Radeons require this).
	if mipLevelCount > 0 && g.glCaps().MipmapLevelControlSupport() {
		params := glTex.Parameters()
		nonsamplerState := *params.NonsamplerState()
		maxLevel := int32(glTex.MaxMipmapLevel())
		changed := false
		if nonsamplerState.BaseMipmapLevel != 0 {
			g.fns().TexParameteri(glTex.Target(), TEXTURE_BASE_LEVEL, 0)
			nonsamplerState.BaseMipmapLevel = 0
			changed = true
		}
		if nonsamplerState.MaxMipmapLevel != maxLevel {
			g.fns().TexParameteri(glTex.Target(), TEXTURE_MAX_LEVEL, maxLevel)
			// Record the max level actually set here, not the base level — an easy mix-up since both fields are updated
			// in this block.
			nonsamplerState.MaxMipmapLevel = maxLevel
			changed = true
		}
		if changed {
			params.Set(nil, nonsamplerState, g.resetTimestampForTextureParameters)
		}
	}

	if glTex.GLFormat().IsCompressed() {
		panic("compressed formats not in scope")
	}
	if !g.uploadColorTypeTexData(glTex.GLFormat(), surfaceColorType, glTex.Surface().Dimensions(),
		glTex.Target(), rect, srcColorType, texels) {
		return false
	}
	g.didWriteToSurface(surface)
	return true
}

// uploadColorTypeTexData resolves the external GL format/type for the given color-type pairing and uploads the texel
// data.
func (g *Gpu) uploadColorTypeTexData(textureFormat Format, textureColorType gpu.ColorType, texDims geom.ISize, target uint32, dstRect geom.IRect, srcColorType gpu.ColorType, texels []gpu.MipLevel) bool {
	if textureFormat.IsCompressed() {
		panic("compressed formats not in scope")
	}
	bpp := srcColorType.BytesPerPixel()
	// External format and type come from the upload data.
	externalFormat, externalType := g.glCaps().TexSubImageExternalFormatAndType(
		textureFormat, textureColorType, srcColorType,
	)
	if externalFormat == 0 || externalType == 0 {
		return false
	}
	g.uploadTexData(texDims, target, dstRect, externalFormat, externalType, bpp, texels)
	return true
}

// uploadTexData pushes data to the texture bound on the active unit. dstRect must be the texture bounds if more than
// one level is uploaded.
func (g *Gpu) uploadTexData(texDims geom.ISize, target uint32, dstRect geom.IRect, externalFormat, externalType uint32, bpp int, texels []gpu.MipLevel) {
	if texDims.IsEmpty() || dstRect.IsEmpty() || !geom.IRectSize(texDims).ContainsRect(dstRect) {
		panic("invalid upload rect")
	}
	mipLevelCount := len(texels)
	if mipLevelCount == 0 || mipLevelCount > gpu.ComputeMipLevelCount(texDims.Width, texDims.Height)+1 {
		panic("invalid mip level count")
	}
	if mipLevelCount > 1 && dstRect != geom.IRectSize(texDims) {
		panic("multi-level upload must cover the texture")
	}

	f := g.fns()
	restoreGLRowLength := false

	g.unbindXferBuffer(gpu.BufferTypeXferCpuToGpu)
	f.PixelStorei(UNPACK_ALIGNMENT, 1)

	w := dstRect.Width()
	h := dstRect.Height()
	for level := 0; level < mipLevelCount; level++ {
		if texels[level].Pixels != nil {
			trimRowBytes := int(w) * bpp
			rowBytes := texels[level].RowBytes
			if g.Caps().WritePixelsRowBytesSupport && (rowBytes != trimRowBytes || restoreGLRowLength) {
				f.PixelStorei(UNPACK_ROW_LENGTH, int32(rowBytes/bpp))
				restoreGLRowLength = true
			} else if rowBytes != trimRowBytes {
				panic("row bytes mismatch without support")
			}
			f.TexSubImage2D(target, int32(level), dstRect.Left, dstRect.Top, w, h,
				externalFormat, externalType, uintptr(unsafe.Pointer(&texels[level].Pixels[0])))
		}
		w = max(w>>1, 1)
		h = max(h>>1, 1)
	}
	if restoreGLRowLength {
		f.PixelStorei(UNPACK_ROW_LENGTH, 0)
	}
}

// uploadColorToTexZeros clears texture levels to transparent black when neither ClearTexImage nor an FBO clear is
// available, by uploading zeroed pixel data. Zero bytes are the representation of transparent black in every supported
// format.
func (g *Gpu) uploadColorToTexZeros(textureFormat Format, texDims geom.ISize, target, levelMask uint32) bool {
	externalFormat, externalType, colorType := g.glCaps().
		TexSubImageDefaultFormatTypeAndColorType(textureFormat)
	if colorType == gpu.ColorTypeUnknown {
		return false
	}
	numLevels := gpu.ComputeMipLevelCount(texDims.Width, texDims.Height) + 1
	bpp := colorType.BytesPerPixel()
	var pixelStorage []byte
	levels := make([]gpu.MipLevel, numLevels)
	levelDims := texDims
	for i := 0; i < numLevels; i++ {
		if levelMask&(1<<i) != 0 {
			if pixelStorage == nil {
				// Make one tight image at the first size and reuse it for smaller levels.
				pixelStorage = make([]byte, int(levelDims.Width)*int(levelDims.Height)*bpp)
			}
			levels[i] = gpu.MipLevel{Pixels: pixelStorage, RowBytes: int(levelDims.Width) * bpp}
		}
		levelDims = geom.ISize{Width: max(levelDims.Width>>1, 1), Height: max(levelDims.Height>>1, 1)}
	}
	g.uploadTexData(texDims, target, geom.IRectSize(texDims), externalFormat, externalType, bpp,
		levels)
	return true
}

// CreateBuffer creates a GPU buffer of the given size and type, or nil if the request pairs a transfer-buffer type with
// a static access pattern (never valid).
func (g *Gpu) CreateBuffer(size uint64, intendedType gpu.BufferType, accessPattern gpu.AccessPattern) *Buffer {
	g.handleDirtyContext()
	if (intendedType == gpu.BufferTypeXferCpuToGpu ||
		intendedType == gpu.BufferTypeXferGpuToCpu) &&
		accessPattern == gpu.AccessPatternStatic {
		return nil
	}
	buffer := MakeBuffer(g, size, intendedType, accessPattern)
	if buffer != nil && !g.Caps().ReuseScratchBuffers {
		buffer.RemoveScratchKey()
	}
	return buffer
}

// WrapBackendRenderTarget wraps a caller-owned FBO as a Surface (unison's FBO 0 with its stencil bits is the typical
// case).
func (g *Gpu) WrapBackendRenderTarget(dims geom.ISize, format Format, sampleCnt, stencilBits int, fboID uint32) *Surface {
	g.handleDirtyContext()
	if !g.glCaps().IsFormatRenderable(format, sampleCnt) {
		return nil
	}
	sampleCount := g.glCaps().GetRenderTargetSampleCount(sampleCnt, format)

	var ids RenderTargetIDs
	if sampleCount <= 1 {
		ids.SingleSampleFBOID = fboID
		ids.MultisampleFBOID = UnresolvableFBOID
	} else {
		ids.SingleSampleFBOID = UnresolvableFBOID
		ids.MultisampleFBOID = fboID
	}
	ids.RTFBOOwnership = gpu.OwnershipBorrowed
	ids.TotalMemorySamplesPerPixel = sampleCount

	return MakeWrappedRenderTarget(g, dims, format, sampleCount, &ids, stencilBits,
		"GLGpu_WrapBackendRenderTarget")
}

// getCompatibleStencilIndex probes (once per format) which stencil format produces a complete FBO alongside a color
// attachment of the given format.
func (g *Gpu) getCompatibleStencilIndex(format Format) int {
	if g.Caps().AvoidStencilBuffers {
		return -1
	}
	const probeSize = 16
	if !g.glCaps().CanFormatBeFBOColorAttachment(format) {
		panic("format cannot be a color attachment")
	}

	if !g.glCaps().HasStencilFormatBeenDeterminedForFormat(format) {
		f := g.fns()
		// Default to unsupported; set this if we find a stencil format that works.
		firstWorkingStencilFormatIndex := -1

		dims := geom.ISize{Width: probeSize, Height: probeSize}
		colorID := g.createTexture(dims, format, TEXTURE_2D, nil, 1, "canvas")
		if colorID == 0 {
			return -1
		}
		// Unbind the texture from the texture unit before binding it to the frame buffer.
		f.BindTexture(TEXTURE_2D, 0)

		var fb uint32
		f.GenFramebuffers(1, &fb)
		g.BindFramebuffer(FRAMEBUFFER, fb)
		g.hwBoundRenderTargetUniqueID = 0
		f.FramebufferTexture2D(FRAMEBUFFER, COLOR_ATTACHMENT0, TEXTURE_2D, colorID, 0)
		var sbRBID uint32
		f.GenRenderbuffers(1, &sbRBID)

		// Look over the formats until we find a compatible one.
		if sbRBID != 0 {
			f.BindRenderbuffer(RENDERBUFFER, sbRBID)
			stencilFormats := g.glCaps().StencilFormats()
			for i := range stencilFormats {
				sFmt := stencilFormats[i]
				err := g.allocCall(func(fn *Functions) {
					fn.RenderbufferStorage(RENDERBUFFER, sFmt.ToEnum(), probeSize, probeSize)
				})
				if err != NO_ERROR {
					continue
				}
				f.FramebufferRenderbuffer(FRAMEBUFFER, STENCIL_ATTACHMENT, RENDERBUFFER, sbRBID)
				if sFmt.IsPackedDepthStencil() {
					f.FramebufferRenderbuffer(FRAMEBUFFER, DEPTH_ATTACHMENT, RENDERBUFFER, sbRBID)
				} else {
					f.FramebufferRenderbuffer(FRAMEBUFFER, DEPTH_ATTACHMENT, RENDERBUFFER, 0)
				}
				if f.CheckFramebufferStatus(FRAMEBUFFER) == FRAMEBUFFER_COMPLETE {
					firstWorkingStencilFormatIndex = i
					break
				}
				f.FramebufferRenderbuffer(FRAMEBUFFER, STENCIL_ATTACHMENT, RENDERBUFFER, 0)
				if sFmt.IsPackedDepthStencil() {
					f.FramebufferRenderbuffer(FRAMEBUFFER, DEPTH_ATTACHMENT, RENDERBUFFER, 0)
				}
			}
			f.DeleteRenderbuffers(1, &sbRBID)
		}
		f.DeleteTextures(1, &colorID)
		g.BindFramebuffer(FRAMEBUFFER, 0)
		g.DeleteFramebuffer(fb)
		g.glCaps().SetStencilFormatIndexForFormat(format, firstWorkingStencilFormatIndex)
	}
	return g.glCaps().StencilFormatIndexForFormat(format)
}

// MakeStencilAttachment creates a stencil buffer compatible with the given color format.
func (g *Gpu) MakeStencilAttachment(colorFormat Format, dims geom.ISize, numStencilSamples int) *Attachment {
	sIdx := g.getCompatibleStencilIndex(colorFormat)
	if sIdx < 0 {
		return nil
	}
	sFmt := g.glCaps().StencilFormats()[sIdx]
	return NewStencilAttachment(g, dims, numStencilSamples, sFmt)
}

// MakeMSAAAttachment creates a multisampled color renderbuffer attachment.
func (g *Gpu) MakeMSAAAttachment(dims geom.ISize, format Format, numSamples int) *Attachment {
	return NewMSAAAttachment(g, dims, numSamples, format)
}

// RegenerateMipmapLevels regenerates dirty mip levels: the glGenerateMipmap fast path, or the manual draw-based lane
// (DoManualMipmapping — see mipmapprogram.go) on drivers where glGenerateMipmap is buggy.
func (g *Gpu) RegenerateMipmapLevels(texture *Texture) bool {
	if !g.Caps().MipmapSupport || texture.Mipmapped() != gpu.MipmappedYes {
		panic("regenerating mipmaps without mipmaps")
	}
	if !texture.MipmapsAreDirty() {
		// This can happen when the proxy expects mipmaps to be dirty but they are not dirty on the actual target (e.g.
		// ops that ended up drawing nothing).
		return true
	}
	if texture.Surface().ReadOnly() {
		return false
	}
	g.handleDirtyContext()
	if texture.Target() != TEXTURE_2D {
		return false
	}
	if g.glCaps().DoManualMipmapping() && g.glCaps().IsFormatRenderable(texture.GLFormat(), 1) {
		if !g.regenerateMipmapLevelsByDraws(texture) {
			return false
		}
		texture.MarkMipmapsClean()
		return true
	}
	g.bindTextureToScratchUnit(texture.Target(), texture.TextureID())
	g.fns().GenerateMipmap(texture.Target())
	texture.MarkMipmapsClean()
	return true
}

// tempFBOTarget selects which of the two scratch FBO slots bindSurfaceFBOForPixelOps uses.
type tempFBOTarget bool

const (
	tempFBOSrc tempFBOTarget = false
	tempFBODst tempFBOTarget = true
)

// bindSurfaceFBOForPixelOps binds a surface as an FBO for copying, reading, or clearing, using a temporary FBO when the
// surface doesn't own one. Must be paired with unbindSurfaceFBOForPixelOps.
func (g *Gpu) bindSurfaceFBOForPixelOps(surface *Surface, mipLevel int, fboTarget uint32, tempTarget tempFBOTarget) {
	rt := surface.AsRenderTarget()
	if rt == nil || mipLevel > 0 {
		texture := surface.AsTexture()
		if texture == nil {
			panic("surface is neither texture nor render target")
		}
		tempFBOID := &g.tempSrcFBOID
		if tempTarget == tempFBODst {
			tempFBOID = &g.tempDstFBOID
		}
		if *tempFBOID == 0 {
			g.fns().GenFramebuffers(1, tempFBOID)
		}
		g.BindFramebuffer(fboTarget, *tempFBOID)
		g.fns().FramebufferTexture2D(fboTarget, COLOR_ATTACHMENT0, texture.Target(),
			texture.TextureID(), int32(mipLevel))
		if mipLevel == 0 {
			texture.BaseLevelWasBoundToFBO()
		}
	} else {
		rt.BindForPixelOps(fboTarget)
	}
}

// unbindSurfaceFBOForPixelOps detaches the texture bound by bindSurfaceFBOForPixelOps, undoing it.
func (g *Gpu) unbindSurfaceFBOForPixelOps(surface *Surface, mipLevel int, fboTarget uint32) {
	if mipLevel > 0 || surface.AsRenderTarget() == nil {
		texture := surface.AsTexture()
		if texture == nil {
			panic("surface is neither texture nor render target")
		}
		g.fns().FramebufferTexture2D(fboTarget, COLOR_ATTACHMENT0, texture.Target(), 0, 0)
	}
}

// PreferredStencilFormat returns the stencil format the probe found compatible with the given color format, or ok=false
// when stencil is unavailable.
func (g *Gpu) PreferredStencilFormat(colorFormat Format) (Format, bool) {
	sIdx := g.getCompatibleStencilIndex(colorFormat)
	if sIdx < 0 {
		return FormatUnknown, false
	}
	return g.glCaps().StencilFormats()[sIdx], true
}
