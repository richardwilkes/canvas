// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Caps' accessors and format-level query methods, plus the base-caps query logic those build on — with a single backend
// the two layers are flattened here. A backend format is represented by (Format, gpu.TextureType) parameters rather
// than a backend-surface type. The surface/proxy-level entry points (surfaceSupportsReadPixels, canCopySurface,
// dst-copy restrictions, program descriptors) live here too.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// MSFBOType reports the type of MSAA FBO support.
func (c *Caps) MSFBOType() MSFBOType { return c.msFBOType }

// UsesMSAARenderBuffers reports whether the preferred MSAA FBO approach uses MSAA renderbuffers (always true on desktop
// when MSAA is supported at all).
func (c *Caps) UsesMSAARenderBuffers() bool { return c.msFBOType != MSFBONone }

// FramebufferResolvesMustBeFullSize reports whether MSAA resolves must cover the full framebuffer.
func (c *Caps) FramebufferResolvesMustBeFullSize() bool {
	return c.blitFramebufferFlags&ResolveMustBeFullBlitFramebufferFlag != 0
}

// CanResolveSingleToMSAA reports whether a single-sample source can resolve into an MSAA destination.
func (c *Caps) CanResolveSingleToMSAA() bool {
	return c.blitFramebufferFlags&NoMSAADstBlitFramebufferFlag == 0
}

// InvalidateFBType reports how framebuffer invalidation is done.
func (c *Caps) InvalidateFBType() InvalidateFBType { return c.invalidateFBType }

// InvalidateBufferType reports how buffer invalidation is done.
func (c *Caps) InvalidateBufferType() InvalidateBufferType { return c.invalidateBufferType }

// MapBufferType reports what type of buffer mapping is supported.
func (c *Caps) MapBufferType() MapBufferType { return c.mapBufferType }

// TransferBufferType reports what type of transfer buffer is supported.
func (c *Caps) TransferBufferType() TransferBufferType { return c.transferBufferType }

// FenceSyncSupport reports support for GL sync objects.
func (c *Caps) FenceSyncSupport() bool { return c.fenceSyncSupport }

// FenceType reports how GPU fences are implemented.
func (c *Caps) FenceType() FenceType { return c.fenceType }

// TimerQueryType reports what type of timer queries are supported.
func (c *Caps) TimerQueryType() TimerQueryType { return c.timerQueryType }

// MultiDrawType reports how multi draws are implemented (if at all).
func (c *Caps) MultiDrawType() MultiDrawType { return c.multiDrawType }

// RegenerateMipmapType reports how restricting sampled miplevels during mipmap regeneration is done.
func (c *Caps) RegenerateMipmapType() RegenerateMipmapType { return c.regenerateMipmapType }

// MaxFragmentUniformVectors is the maximum number of fragment uniform vectors.
func (c *Caps) MaxFragmentUniformVectors() int { return c.maxFragmentUniformVectors }

// MaxTextureMaxAnisotropy is the largest anisotropic filtering ratio.
func (c *Caps) MaxTextureMaxAnisotropy() float32 { return c.maxTextureMaxAnisotropy }

// ImagingSupport reports whether GL_ARB_imaging is supported.
func (c *Caps) ImagingSupport() bool { return c.imagingSupport }

// VertexArrayObjectSupport reports support for vertex array objects.
func (c *Caps) VertexArrayObjectSupport() bool { return c.vertexArrayObjectSupport }

// DebugSupport reports support for KHR_debug-style debug output (GL 4.3+).
func (c *Caps) DebugSupport() bool { return c.debugSupport }

// ES2CompatibilitySupport reports support for GL_ARB_ES2_compatibility.
func (c *Caps) ES2CompatibilitySupport() bool { return c.es2CompatibilitySupport }

// DrawRangeElementsSupport reports support for glDrawRangeElements.
func (c *Caps) DrawRangeElementsSupport() bool { return c.drawRangeElementsSupport }

// BaseVertexBaseInstanceSupport reports whether the glDraw*Base(VertexBase)Instance methods and baseInstance fields in
// indirect draw commands are supported.
func (c *Caps) BaseVertexBaseInstanceSupport() bool { return c.baseVertexBaseInstanceSupport }

// IsCoreProfile reports whether the context is a core profile.
func (c *Caps) IsCoreProfile() bool { return c.isCoreProfile }

// BindFragDataLocationSupport reports support for glBindFragDataLocation.
func (c *Caps) BindFragDataLocationSupport() bool { return c.bindFragDataLocationSupport }

// RectangleTextureSupport reports support for GL_TEXTURE_RECTANGLE textures.
func (c *Caps) RectangleTextureSupport() bool { return c.rectangleTextureSupport }

// MipmapLevelControlSupport reports whether the BASE and MAX mipmap levels can be set.
func (c *Caps) MipmapLevelControlSupport() bool { return c.mipmapLevelControlSupport }

// MipmapLodControlSupport reports whether the MIN/MAX LOD values can be set.
func (c *Caps) MipmapLodControlSupport() bool { return c.mipmapLodControlSupport }

// ClearTextureSupport reports glClearTex(Sub)Image support.
func (c *Caps) ClearTextureSupport() bool { return c.clearTextureSupport }

// DoManualMipmapping reports whether mipmaps are generated via iterative downsampling draws.
func (c *Caps) DoManualMipmapping() bool { return c.doManualMipmapping }

// ClearToBoundaryValuesIsBroken reports whether clears of colors made up of only 1s and 0s fail (certain Intel GPUs
// on Mac).
func (c *Caps) ClearToBoundaryValuesIsBroken() bool { return c.clearToBoundaryValuesIsBroken }

// BindDefaultFramebufferOnPresent reports whether FBO 0 must be bound when presenting a surface.
func (c *Caps) BindDefaultFramebufferOnPresent() bool { return c.bindDefaultFramebufferOnPresent }

// MaxInstancesPerDrawWithoutCrashing caps the instance count for a single instanced draw. The only setters were mobile
// workarounds (trimmed), so the pending count is always safe.
func (c *Caps) MaxInstancesPerDrawWithoutCrashing(pendingInstanceCount int) int {
	return pendingInstanceCount
}

// ProgramBinarySupport reports whether program binaries can be retrieved and loaded.
func (c *Caps) ProgramBinarySupport() bool { return c.programBinarySupport }

// ProgramParameterSupport reports glProgramParameteri support.
func (c *Caps) ProgramParameterSupport() bool { return c.programParameterSupport }

// ProgramBinaryFormatIsValid reports whether binaryFormat is one the driver reported as usable.
func (c *Caps) ProgramBinaryFormatIsValid(binaryFormat uint32) bool {
	for _, f := range c.programBinaryFormats {
		if f == binaryFormat {
			return true
		}
	}
	return false
}

// SamplerObjectSupport reports whether sampler objects are available.
func (c *Caps) SamplerObjectSupport() bool { return c.samplerObjectSupport }

// UseSamplerObjects reports whether sampler objects are used in favor of texture parameters. (Only true when
// SamplerObjectSupport.)
func (c *Caps) UseSamplerObjects() bool { return c.useSamplerObjects }

// TextureSwizzleSupport reports texture swizzle support.
func (c *Caps) TextureSwizzleSupport() bool { return c.textureSwizzleSupport }

// SRGBWriteControl reports support for toggling sRGB writes for sRGB-capable color buffers.
func (c *Caps) SRGBWriteControl() bool { return c.srgbWriteControl }

// SkipErrorChecks reports whether glGetError and framebuffer completeness checks are skipped (shader compilation and
// link status are still checked).
func (c *Caps) SkipErrorChecks() bool { return c.skipErrorChecks }

// ClientCanDisableMultisample reports whether GL_MULTISAMPLE can be disabled (always true on desktop).
func (c *Caps) ClientCanDisableMultisample() bool { return c.clientCanDisableMultisample }

// BlitFramebufferFlags reports the glBlitFramebuffer limitations.
func (c *Caps) BlitFramebufferFlags() uint32 { return c.blitFramebufferFlags }

// StencilFormats returns the legal stencil formats — not guaranteed supported by the driver — from most preferred to
// least.
func (c *Caps) StencilFormats() []Format { return c.stencilFormats }

// HasStencilFormatBeenDeterminedForFormat reports whether a stencil format index has been found for the format (or
// we've found that no format works).
func (c *Caps) HasStencilFormatBeenDeterminedForFormat(format Format) bool {
	return c.formatTable[format].stencilFormatIndex != unknownStencilIndex
}

// StencilFormatIndexForFormat returns the index into StencilFormats() of the best stencil format for the format,
// negative when no stencil format is supported. HasStencilFormatBeenDeterminedForFormat must already be true.
func (c *Caps) StencilFormatIndexForFormat(format Format) int {
	return c.formatTable[format].stencilFormatIndex
}

// SetStencilFormatIndexForFormat records an index into StencilFormats() as the best stencil format for the format; a
// negative index records that the format has no supported stencil format.
func (c *Caps) SetStencilFormatIndexForFormat(format Format, index int) {
	if index < 0 {
		index = unsupportedStencilFormatIndex
	}
	c.formatTable[format].stencilFormatIndex = index
}

// IsFormatSRGB reports whether format is an sRGB format.
func (c *Caps) IsFormatSRGB(format Format) bool { return format == FormatSRGB8ALPHA8 }

// IsFormatTexturable reports whether format is texturable for the given texture type.
func (c *Caps) IsFormatTexturable(format Format, textureType gpu.TextureType) bool {
	if textureType == gpu.TextureTypeRectangle && !c.rectangleTextureSupport {
		return false
	}
	return c.IsFormatTexturableAny(format)
}

// IsFormatTexturableAny reports whether format is texturable for any texture type.
func (c *Caps) IsFormatTexturableAny(format Format) bool {
	return c.formatTable[format].flags&texturableFormatFlag != 0
}

// IsFormatAsColorTypeRenderable reports whether ct/format is renderable at sampleCount.
func (c *Caps) IsFormatAsColorTypeRenderable(ct gpu.ColorType, format Format, sampleCount int) bool {
	if c.formatTable[format].colorTypeFlags(ct)&renderableColorTypeFlag == 0 {
		return false
	}
	return c.IsFormatRenderable(format, sampleCount)
}

// IsFormatRenderable reports whether format is renderable at sampleCount.
func (c *Caps) IsFormatRenderable(format Format, sampleCount int) bool {
	return sampleCount <= c.MaxRenderTargetSampleCount(format)
}

// GetRenderTargetSampleCount returns the smallest supported sample count >= requestedCount, or 0 when unsupported.
// requestedCount == 0 is treated as 1; a request of 1 returns 1 only when non-MSAA rendering is supported.
func (c *Caps) GetRenderTargetSampleCount(requestedCount int, format Format) int {
	table := c.formatTable[format].colorSampleCounts
	if len(table) == 0 {
		return 0
	}
	if requestedCount < 1 {
		requestedCount = 1
	}
	if requestedCount == 1 {
		if table[0] == 1 {
			return 1
		}
		return 0
	}
	for _, sampleCount := range table {
		if sampleCount >= requestedCount {
			if c.DriverBugWorkarounds.MaxMSAASampleCount4 && sampleCount > 4 {
				sampleCount = 4
			}
			return sampleCount
		}
	}
	return 0
}

// MaxRenderTargetSampleCount returns the largest supported sample count for format.
func (c *Caps) MaxRenderTargetSampleCount(format Format) int {
	table := c.formatTable[format].colorSampleCounts
	if len(table) == 0 {
		return 0
	}
	count := table[len(table)-1]
	if c.DriverBugWorkarounds.MaxMSAASampleCount4 && count > 4 {
		count = 4
	}
	return count
}

// InternalMultisampleCount returns the number of samples to use for internal MSAA to the given format, 0 to not attempt
// internal multisampling.
func (c *Caps) InternalMultisampleCount(format Format) int {
	maxCount := c.MaxRenderTargetSampleCount(format)
	if c.InternalMultisampleCountCap < maxCount {
		return c.InternalMultisampleCountCap
	}
	return maxCount
}

// SupportsDynamicMSAA reports whether a single-sample render target whose format supports internal multisampling can be
// promoted to a DMSAA attachment. The disallow-dynamic-MSAA flag some backends set only applies to ANGLE-D3D11 and
// WebGL driver paths — both outside the desktop trim — so it is constant true here.
func (c *Caps) SupportsDynamicMSAA(rtProxy *RenderTargetProxy) bool {
	return rtProxy.NumSamples() == 1 &&
		c.InternalMultisampleCount(rtProxy.Proxy().Format()) > 1
}

// DmsaaResolveCanBeUsedAsTextureInSameRenderPass reports whether a DMSAA resolve texture can be sampled within the same
// render pass that produced it (true for GL).
func (c *Caps) DmsaaResolveCanBeUsedAsTextureInSameRenderPass() bool { return true }

// CanFormatBeFBOColorAttachment reports whether format can be attached to an FBO as a color attachment.
func (c *Caps) CanFormatBeFBOColorAttachment(format Format) bool {
	return c.formatTable[format].flags&fboColorAttachmentFormatFlag != 0
}

// IsFormatCopyable reports whether format can be a copy source/destination: in GL, copying (CopyTexImage, blit, or
// draw) requires the format to be bindable to a FBO.
func (c *Caps) IsFormatCopyable(format Format) bool {
	return c.CanFormatBeFBOColorAttachment(format)
}

// FormatSupportsTexStorage reports whether format can be allocated via glTexStorage.
func (c *Caps) FormatSupportsTexStorage(format Format) bool {
	return c.formatTable[format].flags&useTexStorageFormatFlag != 0
}

// GetFormatFromColorType returns the backend format used for ct.
func (c *Caps) GetFormatFromColorType(ct gpu.ColorType) Format {
	return c.colorTypeToFormatTable[ct]
}

// TexImageOrStorageInternalFormat returns the internal format for glTexImage/glTexStorage (which one depends on
// FormatSupportsTexStorage).
func (c *Caps) TexImageOrStorageInternalFormat(format Format) uint32 {
	return c.formatTable[format].internalFormatForTexImageOrStorage
}

// RenderbufferInternalFormat returns the internal format for glRenderbufferStorage(Multisample).
func (c *Caps) RenderbufferInternalFormat(format Format) uint32 {
	return c.formatTable[format].internalFormatForRenderbuffer
}

// FormatDefaultExternalType returns format's default external GL type.
func (c *Caps) FormatDefaultExternalType(format Format) uint32 {
	return c.formatTable[format].defaultExternalType
}

// TexImageExternalFormatAndType returns the external format/type to pass to glTexImage2D with nil data to create an
// uninitialized texture.
func (c *Caps) TexImageExternalFormatAndType(surfaceFormat Format) (externalFormat, externalType uint32) {
	info := &c.formatTable[surfaceFormat]
	return info.defaultExternalFormat, info.defaultExternalType
}

// TexSubImageDefaultFormatTypeAndColorType returns the external format, type, and color type to use when uploading
// solid color data via glTexSubImage to clear the texture at creation.
func (c *Caps) TexSubImageDefaultFormatTypeAndColorType(format Format) (externalFormat, externalType uint32, colorType gpu.ColorType) {
	info := &c.formatTable[format]
	return info.defaultExternalFormat, info.defaultExternalType, info.defaultColorType
}

// TexSubImageExternalFormatAndType returns the external GL format/type for glTexSubImage2D given a surface format+color
// type and a source data color type. Zero results mean the combination is unsupported.
func (c *Caps) TexSubImageExternalFormatAndType(surfaceFormat Format, surfaceColorType, memoryColorType gpu.ColorType) (externalFormat, externalType uint32) {
	return c.externalFormatAndType(surfaceFormat, surfaceColorType, memoryColorType,
		texImageExternalFormatUsage)
}

// ReadPixelsFormat returns the external format/type for glReadPixels given a surface format+color type and a
// destination memory color type.
func (c *Caps) ReadPixelsFormat(surfaceFormat Format, surfaceColorType, memoryColorType gpu.ColorType) (externalFormat, externalType uint32) {
	return c.externalFormatAndType(surfaceFormat, surfaceColorType, memoryColorType,
		readPixelsExternalFormatUsage)
}

func (c *Caps) externalFormatAndType(surfaceFormat Format, surfaceColorType, memoryColorType gpu.ColorType, usage externalFormatUsage) (externalFormat, externalType uint32) {
	info := &c.formatTable[surfaceFormat]
	return info.externalFormat(surfaceColorType, memoryColorType, usage),
		info.externalType(surfaceColorType, memoryColorType)
}

// ShouldQueryImplementationReadSupport reports whether format's read-format support still needs runtime confirmation
// via glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT/_TYPE). Only ES paths mark combinations as needing the query,
// so on desktop this self-answers false after first call.
func (c *Caps) ShouldQueryImplementationReadSupport(format Format) bool {
	info := &c.formatTable[format]
	if !info.haveQueriedImplementationReadSupport {
		needQuery := false
		for i := range info.colorTypeInfos {
			for j := range info.colorTypeInfos[i].externalIOFormats {
				if info.colorTypeInfos[i].externalIOFormats[j].requiresImplementationReadQuery {
					needQuery = true
				}
			}
		}
		if !needQuery {
			// Pretend we already checked it.
			info.haveQueriedImplementationReadSupport = true
		}
	}
	return !info.haveQueriedImplementationReadSupport
}

// DidQueryImplementationReadSupport records the result of the runtime implementation-read-format query for format.
func (c *Caps) DidQueryImplementationReadSupport(format Format, readFormat, readType uint32) {
	info := &c.formatTable[format]
	for i := range info.colorTypeInfos {
		for j := range info.colorTypeInfos[i].externalIOFormats {
			iof := &info.colorTypeInfos[i].externalIOFormats[j]
			if iof.requiresImplementationReadQuery {
				if iof.externalReadFormat != readFormat || iof.externalType != readType {
					// Don't zero the external type: it's also used for writing data to the texture.
					iof.externalReadFormat = 0
				}
			}
		}
	}
	info.haveQueriedImplementationReadSupport = true
}

// AreColorTypeAndFormatCompatible reports whether ct is a valid color type for format (compression is excluded by the
// trim).
func (c *Caps) AreColorTypeAndFormatCompatible(ct gpu.ColorType, format Format) bool {
	if ct == gpu.ColorTypeUnknown {
		return false
	}
	info := &c.formatTable[format]
	for i := range info.colorTypeInfos {
		if info.colorTypeInfos[i].colorType == ct {
			return true
		}
	}
	return false
}

// GetDefaultBackendFormat returns the default backend format for ct, or FormatUnknown if there is none.
func (c *Caps) GetDefaultBackendFormat(ct gpu.ColorType, renderable bool) Format {
	if ct == gpu.ColorTypeUnknown {
		return FormatUnknown
	}
	format := c.colorTypeToFormatTable[ct]
	if format == FormatUnknown {
		return FormatUnknown
	}
	if !c.IsFormatTexturable(format, gpu.TextureType2D) {
		return FormatUnknown
	}
	if !c.AreColorTypeAndFormatCompatible(ct, format) {
		return FormatUnknown
	}
	// We require that it be possible to write pixels into the "default" format.
	if c.SupportedWritePixelsColorType(ct, format, ct).ColorType == gpu.ColorTypeUnknown {
		return FormatUnknown
	}
	if renderable && !c.IsFormatAsColorTypeRenderable(ct, format, 1) {
		return FormatUnknown
	}
	return format
}

// colorTypeFallback returns the next color type to try when ct's backend format cannot be made renderable.
func colorTypeFallback(ct gpu.ColorType) gpu.ColorType {
	switch ct {
	// RGBA8888 is the default fallback for many color types that may not have renderable backend formats.
	case gpu.ColorTypeAlpha8, gpu.ColorTypeBGR565, gpu.ColorTypeRGB565, gpu.ColorTypeABGR4444,
		gpu.ColorTypeBGRA8888, gpu.ColorTypeRGBA1010102, gpu.ColorTypeBGRA1010102,
		gpu.ColorTypeRGBAF16, gpu.ColorTypeRGBAF16Clamped:
		return gpu.ColorTypeRGBA8888
	case gpu.ColorTypeAlphaF16:
		return gpu.ColorTypeRGBAF16
	case gpu.ColorTypeGray8, gpu.ColorTypeRGBF16F16F16x, gpu.ColorTypeRGB101010x:
		return gpu.ColorTypeRGB888x
	default:
		return gpu.ColorTypeUnknown
	}
}

// GetFallbackColorTypeAndFormat walks the fallback chain until a renderable color type/format pair supporting
// sampleCount is found.
func (c *Caps) GetFallbackColorTypeAndFormat(ct gpu.ColorType, sampleCount int) (gpu.ColorType, Format) {
	for ct != gpu.ColorTypeUnknown {
		format := c.GetDefaultBackendFormat(ct, true)
		if format != FormatUnknown && c.IsFormatRenderable(format, sampleCount) {
			return ct, format
		}
		ct = colorTypeFallback(ct)
	}
	return gpu.ColorTypeUnknown, FormatUnknown
}

// ReadSwizzle returns the swizzle to apply when reading ct data from format. Returns SwizzleRGBA for unknown
// combinations.
func (c *Caps) ReadSwizzle(format Format, ct gpu.ColorType) gpu.Swizzle {
	info := &c.formatTable[format]
	for i := range info.colorTypeInfos {
		if info.colorTypeInfos[i].colorType == ct {
			return info.colorTypeInfos[i].readSwizzle
		}
	}
	return gpu.SwizzleRGBA
}

// WriteSwizzle returns the swizzle to apply when writing ct data to format. Returns SwizzleRGBA for unknown
// combinations.
func (c *Caps) WriteSwizzle(format Format, ct gpu.ColorType) gpu.Swizzle {
	info := &c.formatTable[format]
	for i := range info.colorTypeInfos {
		if info.colorTypeInfos[i].colorType == ct {
			return info.colorTypeInfos[i].writeSwizzle
		}
	}
	return gpu.SwizzleRGBA
}

// ComputeFormatKey returns the program-key value for format.
func (c *Caps) ComputeFormatKey(format Format) uint64 { return uint64(format) }

// offsetAlignmentForTransferBuffer returns the required transfer-buffer offset alignment for externalType, derived from
// the OpenGL spec's "Pixel data type parameter values and the corresponding GL data types" table.
func offsetAlignmentForTransferBuffer(externalType uint32) int {
	switch externalType {
	case UNSIGNED_BYTE, BYTE:
		return 1
	case UNSIGNED_SHORT, SHORT, HALF_FLOAT, UNSIGNED_SHORT_5_6_5, UNSIGNED_SHORT_4_4_4_4,
		UNSIGNED_SHORT_5_5_5_1:
		return 2
	case UNSIGNED_INT, INT, FLOAT, UNSIGNED_INT_2_10_10_10_REV:
		return 4
	default:
		return 0
	}
}

// SupportedWritePixelsColorType returns, given a surface format+color type and a src data color type, the color type
// the caller must coax the data into for GPU writePixels.
func (c *Caps) SupportedWritePixelsColorType(surfaceColorType gpu.ColorType, surfaceFormat Format, srcColorType gpu.ColorType) gpu.SupportedWrite {
	// We first try to find a supported write pixels color type that matches the data's srcColorType. If that doesn't
	// exist we use any supported color type.
	fallbackCT := gpu.ColorTypeUnknown
	info := &c.formatTable[surfaceFormat]
	transferOffsetAlignment := 0
	if info.flags&transfersFormatFlag != 0 {
		transferOffsetAlignment = 1
	}
	for i := range info.colorTypeInfos {
		if info.colorTypeInfos[i].colorType != surfaceColorType {
			continue
		}
		cti := &info.colorTypeInfos[i]
		for j := range cti.externalIOFormats {
			iof := &cti.externalIOFormats[j]
			if iof.externalTexImageFormat == 0 {
				continue
			}
			if iof.colorType == srcColorType {
				return gpu.SupportedWrite{
					ColorType:                        srcColorType,
					OffsetAlignmentForTransferBuffer: transferOffsetAlignment,
				}
			}
			// Currently we just pick the first supported format that we find as our fallback.
			if fallbackCT == gpu.ColorTypeUnknown {
				fallbackCT = iof.colorType
			}
		}
		break
	}
	return gpu.SupportedWrite{
		ColorType:                        fallbackCT,
		OffsetAlignmentForTransferBuffer: transferOffsetAlignment,
	}
}

// SupportedReadPixelsColorType returns a legal color type for GPU readPixels given a src surface color type/format and
// the color type the caller wants. The returned color type may differ from dstColorType, in which case the caller must
// convert, applying ReadSwizzle. Check for ColorTypeUnknown.
func (c *Caps) SupportedReadPixelsColorType(srcColorType gpu.ColorType, srcFormat Format, dstColorType gpu.ColorType) gpu.SupportedRead {
	read := c.onSupportedReadPixelsColorType(srcColorType, srcFormat, dstColorType)

	// There are known problems with 24 vs 32 bit BPP with this color type. Just fail for now if using a transfer
	// buffer.
	if read.ColorType == gpu.ColorTypeRGB888x {
		read.OffsetAlignmentForTransferBuffer = 0
	}
	// It's very convenient to access 1 byte-per-channel 32 bit color types as uint32_t on the CPU. Make those aligned
	// reads out of the buffer even if the underlying API doesn't require it.
	channelFlags := read.ColorType.ChannelFlags()
	if (channelFlags == gpu.RGBAChannelFlags || channelFlags == gpu.RGBChannelFlags ||
		channelFlags == gpu.AlphaChannelFlag || channelFlags == gpu.GrayChannelFlag) &&
		read.ColorType.BytesPerPixel() == 4 {
		switch read.OffsetAlignmentForTransferBuffer % 4 {
		case 0:
			// Already a multiple of 4.
		case 2:
			// A multiple of 2 but not 4.
			read.OffsetAlignmentForTransferBuffer *= 2
		default:
			// Not a multiple of 2.
			read.OffsetAlignmentForTransferBuffer *= 4
		}
	}
	return read
}

func (c *Caps) onSupportedReadPixelsColorType(srcColorType gpu.ColorType, srcFormat Format, dstColorType gpu.ColorType) gpu.SupportedRead {
	// We first try to find a supported read pixels color type that matches the requested dstColorType. If that doesn't
	// exist we use any valid read pixels color type.
	fallbackRead := gpu.SupportedRead{ColorType: gpu.ColorTypeUnknown}
	info := &c.formatTable[srcFormat]
	for i := range info.colorTypeInfos {
		if info.colorTypeInfos[i].colorType != srcColorType {
			continue
		}
		cti := &info.colorTypeInfos[i]
		for j := range cti.externalIOFormats {
			iof := &cti.externalIOFormats[j]
			if iof.externalReadFormat == 0 {
				continue
			}
			if !info.haveQueriedImplementationReadSupport && iof.requiresImplementationReadQuery {
				continue
			}
			transferOffsetAlignment := 0
			if info.flags&transfersFormatFlag != 0 {
				transferOffsetAlignment = offsetAlignmentForTransferBuffer(iof.externalType)
			}
			if iof.colorType == dstColorType {
				return gpu.SupportedRead{
					ColorType:                        dstColorType,
					OffsetAlignmentForTransferBuffer: transferOffsetAlignment,
				}
			}
			// Currently we just pick the first supported format that we find as our fallback.
			if fallbackRead.ColorType == gpu.ColorTypeUnknown {
				fallbackRead = gpu.SupportedRead{
					ColorType:                        iof.colorType,
					OffsetAlignmentForTransferBuffer: transferOffsetAlignment,
				}
			}
		}
		break
	}
	return fallbackRead
}

// ValidateSurfaceParams reports whether a surface with the given parameters is legal to create (sans compression, per
// the trim).
func (c *Caps) ValidateSurfaceParams(dimensions geom.ISize, format Format, renderable bool, renderTargetSampleCnt int, mipmapped bool, textureType gpu.TextureType) bool {
	if textureType != gpu.TextureTypeNone && !c.IsFormatTexturable(format, textureType) {
		return false
	}
	if mipmapped && !c.MipmapSupport {
		return false
	}
	if dimensions.Width < 1 || dimensions.Height < 1 {
		return false
	}
	if renderable {
		if !c.IsFormatRenderable(format, renderTargetSampleCnt) {
			return false
		}
		maxRTSize := int32(c.MaxRenderTargetSize)
		return dimensions.Width <= maxRTSize && dimensions.Height <= maxRTSize
	}
	// We currently do not support multisampled textures.
	if renderTargetSampleCnt != 1 {
		return false
	}
	maxSize := int32(c.MaxTextureSize)
	return dimensions.Width <= maxSize && dimensions.Height <= maxSize
}

// CanCopyTexSubImage reports whether glCopyTexSubImage2D can be used for this src/dst pair (the ES-only channel
// restrictions are trimmed; external textures do not exist under the trim). dstTypeIfTexture/srcTypeIfTexture are nil
// when the surface is not a texture.
func (c *Caps) CanCopyTexSubImage(dstFormat Format, dstHasMSAARenderBuffer bool, dstTypeIfTexture *gpu.TextureType, srcFormat Format, srcHasMSAARenderBuffer bool, srcTypeIfTexture *gpu.TextureType) bool {
	_ = srcTypeIfTexture
	// The GL spec's rules about which format types can be copied to which other types are complex, so we require the
	// types to match exactly.
	if c.FormatDefaultExternalType(dstFormat) != c.FormatDefaultExternalType(srcFormat) {
		return false
	}
	// Either both the src and dst formats need to be SRGB or both need to not be SRGB.
	if dstFormat.IsSRGB() != srcFormat.IsSRGB() {
		return false
	}
	// CopyTexSubImage is invalid or doesn't copy what we want when we have MSAA render buffers.
	if dstHasMSAARenderBuffer || srcHasMSAARenderBuffer {
		return false
	}
	// CopyTex(Sub)Image writes to a texture and we have no way of dynamically wrapping a RT in a texture.
	if dstTypeIfTexture == nil {
		return false
	}
	// Check that we could wrap the source in an FBO.
	return c.CanFormatBeFBOColorAttachment(srcFormat)
}

// CanCopyAsBlit reports whether glBlitFramebuffer can be used for this src/dst pair.
func (c *Caps) CanCopyAsBlit(dstFormat Format, dstSampleCnt int, dstTypeIfTexture *gpu.TextureType, srcFormat Format, srcSampleCnt int, srcTypeIfTexture *gpu.TextureType, srcBounds geom.Rect, srcBoundsExact bool, srcRect, dstRect geom.IRect) bool {
	_ = dstTypeIfTexture
	_ = srcTypeIfTexture
	if !c.CanFormatBeFBOColorAttachment(dstFormat) || !c.CanFormatBeFBOColorAttachment(srcFormat) {
		return false
	}
	if c.blitFramebufferFlags&NoSupportBlitFramebufferFlag != 0 {
		return false
	}
	if dstSampleCnt > 1 && dstSampleCnt != srcSampleCnt {
		// Regardless of support level, all blits require src and dst sample counts to match if the dst is MSAA.
		return false
	}
	if srcRect.Width() != dstRect.Width() || srcRect.Height() != dstRect.Height() {
		// If the blit would scale contents, it's only valid for non-MSAA framebuffers that we can write directly to.
		if c.blitFramebufferFlags&NoScalingOrMirroringBlitFramebufferFlag != 0 || srcSampleCnt > 1 {
			return false
		}
	}
	if c.blitFramebufferFlags&ResolveMustBeFullBlitFramebufferFlag != 0 && srcSampleCnt > 1 {
		if dstSampleCnt == 1 {
			return false
		}
		if srcRect.ToRect() != srcBounds || !srcBoundsExact {
			return false
		}
	}
	if c.blitFramebufferFlags&NoMSAADstBlitFramebufferFlag != 0 && dstSampleCnt > 1 {
		return false
	}
	if c.blitFramebufferFlags&NoFormatConversionBlitFramebufferFlag != 0 && srcFormat != dstFormat {
		return false
	}
	if c.blitFramebufferFlags&NoFormatConversionForMSAASrcBlitFramebufferFlag != 0 &&
		srcSampleCnt > 1 && srcFormat != dstFormat {
		return false
	}
	if c.blitFramebufferFlags&RectsMustMatchForMSAASrcBlitFramebufferFlag != 0 && srcSampleCnt > 1 &&
		dstRect != srcRect {
		return false
	}
	return true
}

// CanCopyAsDraw reports whether a textured draw can be used to copy into dstFormat. (The Mali scaling-copy workaround
// is trimmed.)
func (c *Caps) CanCopyAsDraw(dstFormat Format, srcIsTexturable bool) bool {
	return c.IsFormatRenderable(dstFormat, 1) && srcIsTexturable
}
