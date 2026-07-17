// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GL capability detection, desktop-trimmed:
//
//   - Only the desktop-GL branches are implemented. ES/WebGL branches, the ANGLE/command-buffer/Android/iOS
//     workarounds, and the mobile-renderer (Adreno/Mali/PowerVR/Tegra) workaround zoo are dropped — those renderers
//     only appear behind ES contexts, which interface assembly already rejects. Vendor-keyed quirks and the desktop
//     renderer families (Intel generations, AMD Radeon, Apple, llvmpipe) are kept.
//   - The format table is populated only for the supported color-type matrix: RGBA8 (kRGBA_8888, kBGRA_8888,
//     kRGB_888x), R8 (kR_8, kAlpha_8, kGray_8), ALPHA8/LUMINANCE8 (non-core profiles), BGRA8 (structurally present;
//     desktop GL leaves it capability-less), RGB565, RGBA16F (kRGBA_F16, kRGBA_F16_Clamped, kRGB_F16F16F16x), R16F
//     (kAlpha_F16, for the GPU blur intermediates), and RGB8 (kRGB_888x). All other formats keep zeroed entries and
//     report unsupported.
//   - Protected content, compressed textures, and external/rectangle-texture-specific paths are excluded by the desktop
//     trim. The surface/proxy-level queries (surfaceSupportsReadPixels, onCanCopySurface, getDstCopyRestrictions,
//     makeDesc) live in capsqueries.go; the format-level primitives they build on are ported here.

package gl

import (
	"math"
	"runtime"

	"github.com/richardwilkes/canvas/gpu"
)

// MSFBOType is how multisampled FBOs are supported, desktop members only (the ES render-to-texture variants are
// unreachable).
type MSFBOType int32

// MSFBOType values.
const (
	// MSFBONone: no support for MSAA FBOs.
	MSFBONone MSFBOType = iota
	// MSFBOStandard: OpenGL 3.0+, GL_ARB_framebuffer_object, or GL_EXT_framebuffer_multisample +
	// GL_EXT_framebuffer_blit.
	MSFBOStandard
)

// BlitFramebufferFlags describe glBlitFramebuffer's restrictions on this driver.
const (
	NoSupportBlitFramebufferFlag                    uint32 = 1 << 0
	NoScalingOrMirroringBlitFramebufferFlag         uint32 = 1 << 1
	ResolveMustBeFullBlitFramebufferFlag            uint32 = 1 << 2
	NoMSAADstBlitFramebufferFlag                    uint32 = 1 << 3
	NoFormatConversionBlitFramebufferFlag           uint32 = 1 << 4
	NoFormatConversionForMSAASrcBlitFramebufferFlag uint32 = 1 << 5
	RectsMustMatchForMSAASrcBlitFramebufferFlag     uint32 = 1 << 6
)

// InvalidateFBType is how a framebuffer's contents can be discarded.
type InvalidateFBType int32

// InvalidateFBType values.
const (
	InvalidateFBNone       InvalidateFBType = iota
	InvalidateFBDiscard                     // glDiscardFramebuffer()
	InvalidateFBInvalidate                  // glInvalidateFramebuffer()
)

// InvalidateBufferType is how a buffer's prior contents can be discarded.
type InvalidateBufferType int32

// InvalidateBufferType values.
const (
	InvalidateBufferNone InvalidateBufferType = iota
	// InvalidateBufferNullData: call glBufferData with a null data pointer.
	InvalidateBufferNullData
	// InvalidateBufferInvalidate: glInvalidateBufferData.
	InvalidateBufferInvalidate
)

// MapBufferType is how a buffer can be mapped for CPU access (the Chromium variant is dropped).
type MapBufferType int32

// MapBufferType values.
const (
	MapBufferNone           MapBufferType = iota
	MapBufferMapBuffer                    // glMapBuffer()
	MapBufferMapBufferRange               // glMapBufferRange()
)

// TransferBufferType is how transfer (pixel buffer) buffers are supported (NV/Chromium variants dropped; desktop GL
// always uses ARB-style PBOs).
type TransferBufferType int32

// TransferBufferType values.
const (
	TransferBufferNone   TransferBufferType = iota
	TransferBufferARBPBO                    // ARB_pixel_buffer_object
)

// FenceType is the GPU fence mechanism this driver supports.
type FenceType int32

// FenceType values.
const (
	FenceNone FenceType = iota
	FenceSyncObject
	FenceNVFence
)

// MultiDrawType is how multi-draw-indirect is supported (the ANGLE/WebGL variant is dropped).
type MultiDrawType int32

// MultiDrawType values.
const (
	MultiDrawNone MultiDrawType = iota
	// MultiDrawIndirect: ARB_multi_draw_indirect or GL 4.3 core.
	MultiDrawIndirect
)

// RegenerateMipmapType is how mipmaps are regenerated after a base-level update (the ANGLE base-plus-sync variant is
// dropped).
type RegenerateMipmapType int32

// RegenerateMipmapType values.
const (
	RegenerateMipmapBaseLevel RegenerateMipmapType = iota
	RegenerateMipmapBasePlusMaxLevel
)

// TimerQueryType is the GPU timer query mechanism this driver supports (the ES disjoint variant is dropped).
type TimerQueryType int32

// TimerQueryType values.
const (
	TimerQueryNone TimerQueryType = iota
	TimerQueryRegular
)

// colorTypeInfo flags.
const (
	uploadDataColorTypeFlag uint32 = 0x1
	// renderableColorTypeFlag: does this codebase itself support rendering to this colorType & format pair.
	// Renderability additionally depends on if the format can be an FBO color attachment.
	renderableColorTypeFlag uint32 = 0x2
)

// formatInfo flags.
const (
	texturableFormatFlag uint32 = 0x01
	// fboColorAttachmentFormatFlag means that even if the format cannot be a render target, we can still attach it to a
	// FBO for blitting or reading pixels.
	fboColorAttachmentFormatFlag         uint32 = 0x02
	fboColorAttachmentWithMSAAFormatFlag uint32 = 0x04
	useTexStorageFormatFlag              uint32 = 0x08
	// transfersFormatFlag: are pixel buffer objects supported in/out of this format?
	transfersFormatFlag uint32 = 0x10
)

// formatType is a format's underlying numeric representation.
type formatType int32

const (
	formatTypeUnknown formatType = iota
	formatTypeNormalizedFixedPoint
	formatTypeFloat
)

// externalIOFormat holds the external format and type to use when uploading/downloading data of colorType to a texture
// of the owning format interpreted as the owning ColorTypeInfo's color type. A zero
// externalTexImageFormat/externalReadFormat means TexImage or ReadPixels is unsupported for the combination.
type externalIOFormat struct {
	colorType              gpu.ColorType
	externalType           uint32
	externalTexImageFormat uint32
	externalReadFormat     uint32
	// requiresImplementationReadQuery: must check GL_IMPLEMENTATION_COLOR_READ_FORMAT/_TYPE before using
	// externalReadFormat with glReadPixels. Only ES paths set this; it stays false on desktop.
	requiresImplementationReadQuery bool
}

// colorTypeInfo describes how one color type maps onto a specific backend format.
type colorTypeInfo struct {
	externalIOFormats []externalIOFormat
	colorType         gpu.ColorType
	flags             uint32
	readSwizzle       gpu.Swizzle
	writeSwizzle      gpu.Swizzle
}

type externalFormatUsage int32

const (
	texImageExternalFormatUsage externalFormatUsage = iota
	readPixelsExternalFormatUsage
)

func (cti *colorTypeInfo) externalFormat(externalColorType gpu.ColorType, usage externalFormatUsage, haveQueriedImplementationReadFormat bool) uint32 {
	for i := range cti.externalIOFormats {
		iof := &cti.externalIOFormats[i]
		if iof.colorType == externalColorType {
			if usage == texImageExternalFormatUsage {
				return iof.externalTexImageFormat
			}
			if !haveQueriedImplementationReadFormat && iof.requiresImplementationReadQuery {
				return 0
			}
			return iof.externalReadFormat
		}
	}
	return 0
}

func (cti *colorTypeInfo) externalType(externalColorType gpu.ColorType) uint32 {
	for i := range cti.externalIOFormats {
		if cti.externalIOFormats[i].colorType == externalColorType {
			return cti.externalIOFormats[i].externalType
		}
	}
	return 0
}

// Stencil-format-index sentinels.
const (
	// unknownStencilIndex indicates a stencil format has not yet been determined for the format.
	unknownStencilIndex = -1
	// unsupportedStencilFormatIndex indicates no supported stencil format exists for the format.
	unsupportedStencilFormatIndex = -2
)

// formatInfo describes one backend format's capabilities: texturability, renderability, sample counts, and its
// color-type mappings.
type formatInfo struct {
	colorTypeInfos     []colorTypeInfo
	colorSampleCounts  []int
	stencilFormatIndex int
	flags              uint32
	formatType         formatType
	// internalFormatForTexImageOrStorage is the "internalformat" argument to glTexImage or glTexStorage. It is only
	// guaranteed compatible with glTexStorage when useTexStorageFormatFlag is set, and with glTexImage when it is not.
	internalFormatForTexImageOrStorage uint32
	// internalFormatForRenderbuffer is the "internalformat" argument to glRenderbufferStorageMultisample.
	internalFormatForRenderbuffer uint32
	// defaultExternalFormat/Type are used with glTexImage2D when no data is provided or when clearing by uploading a
	// block of solid color data (of defaultColorType).
	defaultExternalFormat                uint32
	defaultExternalType                  uint32
	defaultColorType                     gpu.ColorType
	haveQueriedImplementationReadSupport bool
}

func (fi *formatInfo) colorTypeFlags(colorType gpu.ColorType) uint32 {
	for i := range fi.colorTypeInfos {
		if fi.colorTypeInfos[i].colorType == colorType {
			return fi.colorTypeInfos[i].flags
		}
	}
	return 0
}

func (fi *formatInfo) externalFormat(surfaceColorType, externalColorType gpu.ColorType, usage externalFormatUsage) uint32 {
	for i := range fi.colorTypeInfos {
		if fi.colorTypeInfos[i].colorType == surfaceColorType {
			return fi.colorTypeInfos[i].externalFormat(externalColorType, usage,
				fi.haveQueriedImplementationReadSupport)
		}
	}
	return 0
}

func (fi *formatInfo) externalType(surfaceColorType, externalColorType gpu.ColorType) uint32 {
	for i := range fi.colorTypeInfos {
		if fi.colorTypeInfos[i].colorType == surfaceColorType {
			return fi.colorTypeInfos[i].externalType(externalColorType)
		}
	}
	return 0
}

// Caps holds the capabilities of a GL context, determined by the GL version, the extension set, and the driver
// identity. It embeds the backend-independent gpu.Caps storage.
type Caps struct {
	stencilFormats       []Format
	programBinaryFormats []uint32
	formatTable          [ColorFormatCount]formatInfo
	gpu.Caps
	maxFragmentUniformVectors     int
	colorTypeToFormatTable        [gpu.ColorTypeCount]Format
	invalidateFBType              InvalidateFBType
	msFBOType                     MSFBOType
	maxTextureMaxAnisotropy       float32
	invalidateBufferType          InvalidateBufferType
	mapBufferType                 MapBufferType
	transferBufferType            TransferBufferType
	fenceType                     FenceType
	timerQueryType                TimerQueryType
	multiDrawType                 MultiDrawType
	regenerateMipmapType          RegenerateMipmapType
	blitFramebufferFlags          uint32
	es2CompatibilitySupport       bool
	clearTextureSupport           bool
	debugSupport                  bool
	imagingSupport                bool
	drawRangeElementsSupport      bool
	baseVertexBaseInstanceSupport bool
	isCoreProfile                 bool
	bindFragDataLocationSupport   bool
	rectangleTextureSupport       bool
	mipmapLevelControlSupport     bool
	mipmapLodControlSupport       bool
	vertexArrayObjectSupport      bool
	programBinarySupport          bool
	programParameterSupport       bool
	samplerObjectSupport          bool
	useSamplerObjects             bool
	textureSwizzleSupport         bool
	fenceSyncSupport              bool
	srgbWriteControl              bool
	skipErrorChecks               bool
	clientCanDisableMultisample   bool
	// Driver workarounds (desktop set).
	doManualMipmapping              bool
	clearToBoundaryValuesIsBroken   bool
	bindDefaultFramebufferOnPresent bool
}

// newCaps constructs and initializes a Caps from the context's GL version, extensions, and driver identity.
func newCaps(options *gpu.ContextOptions, ctxInfo *ContextInfo, iface *Interface) *Caps {
	c := &Caps{Caps: gpu.MakeCaps(options)}
	for i := range c.formatTable {
		c.formatTable[i].stencilFormatIndex = unknownStencilIndex
	}
	c.init(options, ctxInfo, iface)
	return c
}

func getIntegerv(f *Functions, pname uint32) int32 {
	var v int32
	f.GetIntegerv(pname, &v)
	return v
}

func getFloatv(f *Functions, pname uint32) float32 {
	var v float32
	f.GetFloatv(pname, &v)
	return v
}

func (c *Caps) init(options *gpu.ContextOptions, ctxInfo *ContextInfo, iface *Interface) {
	f := &iface.Functions
	version := ctxInfo.Version()

	c.maxFragmentUniformVectors = int(getIntegerv(f, MAX_FRAGMENT_UNIFORM_COMPONENTS)) / 4
	if version >= Ver(3, 2) {
		profileMask := getIntegerv(f, CONTEXT_PROFILE_MASK)
		c.isCoreProfile = profileMask&CONTEXT_CORE_PROFILE_BIT != 0
	}
	if c.DriverBugWorkarounds.MaxFragmentUniformVectors32 && c.maxFragmentUniformVectors > 32 {
		c.maxFragmentUniformVectors = 32
	}
	c.MaxVertexAttributes = int(getIntegerv(f, MAX_VERTEX_ATTRIBS))

	c.WritePixelsRowBytesSupport = true
	c.ReadPixelsRowBytesSupport = true
	c.TransferPixelsToRowBytesSupport = c.WritePixelsRowBytesSupport
	if c.DriverBugWorkarounds.PackParametersWorkaroundWithPackBuffer {
		// In some cases drivers handle copying the last row incorrectly when using GL_PACK_ROW_LENGTH. A scratch buffer
		// fallback exists when pack row length is unsupported, so use that.
		c.ReadPixelsRowBytesSupport = false
	}

	c.TextureBarrierSupport = version >= Ver(4, 5) ||
		ctxInfo.HasExtension("GL_ARB_texture_barrier") ||
		ctxInfo.HasExtension("GL_NV_texture_barrier")

	c.SampleLocationsSupport = version >= Ver(3, 2) ||
		ctxInfo.HasExtension("GL_ARB_texture_multisample")

	c.imagingSupport = ctxInfo.HasExtension("GL_ARB_imaging")

	if version >= Ver(4, 3) || ctxInfo.HasExtension("GL_ARB_invalidate_subdata") {
		c.invalidateFBType = InvalidateFBInvalidate
	} else if ctxInfo.HasExtension("GL_EXT_discard_framebuffer") {
		c.invalidateFBType = InvalidateFBDiscard
	}

	if ctxInfo.Vendor() == VendorARM || ctxInfo.Vendor() == VendorImagination ||
		ctxInfo.Vendor() == VendorQualcomm {
		c.PreferFullscreenClears = true
	}

	c.vertexArrayObjectSupport = version >= Ver(3, 0) ||
		ctxInfo.HasExtension("GL_ARB_vertex_array_object") ||
		ctxInfo.HasExtension("GL_APPLE_vertex_array_object")

	c.debugSupport = version >= Ver(4, 3)

	c.es2CompatibilitySupport = ctxInfo.HasExtension("GL_ARB_ES2_compatibility")

	c.clientCanDisableMultisample = true

	// 3.1 has draw_instanced but not instanced_arrays; for the time being we only care about instanced arrays.
	c.DrawInstancedSupport = version >= Ver(3, 2) ||
		(ctxInfo.HasExtension("GL_ARB_draw_instanced") &&
			ctxInfo.HasExtension("GL_ARB_instanced_arrays"))

	c.bindFragDataLocationSupport = version >= Ver(3, 0)

	c.rectangleTextureSupport = version >= Ver(3, 1) ||
		ctxInfo.HasExtension("GL_ARB_texture_rectangle") ||
		ctxInfo.HasExtension("GL_ANGLE_texture_rectangle")

	// gpu.Caps defaults ClampToBorderSupport to true; clamp to border was added in GL 1.3.
	if version < Ver(1, 3) && !ctxInfo.HasExtension("GL_ARB_texture_border_clamp") {
		c.ClampToBorderSupport = false
	}

	c.textureSwizzleSupport = version >= Ver(3, 3) || ctxInfo.HasExtension("GL_ARB_texture_swizzle")

	c.mipmapLevelControlSupport = true
	c.mipmapLodControlSupport = true

	if ctxInfo.HasExtension("GL_ARB_invalidate_subdata") {
		c.invalidateBufferType = InvalidateBufferInvalidate
	} else {
		c.invalidateBufferType = InvalidateBufferNullData
	}

	c.clearTextureSupport = version >= Ver(4, 4) || ctxInfo.HasExtension("GL_ARB_clear_texture")

	c.srgbWriteControl = version >= Ver(3, 0) ||
		ctxInfo.HasExtension("GL_ARB_framebuffer_sRGB") ||
		ctxInfo.HasExtension("GL_EXT_framebuffer_sRGB")

	// When we are abandoning the context we cannot call into GL, so skip any sync work.
	c.MustSyncGpuDuringAbandon = false

	// ShaderCaps fields. This must run after isCoreProfile is set.
	c.initGLSL(ctxInfo, f)
	shaderCaps := c.ShaderCaps

	shaderCaps.DualSourceBlendingSupport = (version >= Ver(3, 3) ||
		ctxInfo.HasExtension("GL_ARB_blend_func_extended")) &&
		ctxInfo.GLSLGeneration >= gpu.GLSL130
	shaderCaps.ShaderDerivativeSupport = true
	shaderCaps.ExplicitTextureLodSupport = ctxInfo.GLSLGeneration >= gpu.GLSL130
	shaderCaps.IntegerSupport = version >= Ver(3, 0) && ctxInfo.GLSLGeneration >= gpu.GLSL130
	shaderCaps.NonsquareMatrixSupport = ctxInfo.GLSLGeneration >= gpu.GLSL130
	shaderCaps.InverseHyperbolicSupport = ctxInfo.GLSLGeneration >= gpu.GLSL130

	c.ConservativeRasterSupport = ctxInfo.HasExtension("GL_NV_conservative_raster")
	c.WireframeSupport = true

	shaderCaps.RewriteSwitchStatements = ctxInfo.GLSLGeneration < gpu.GLSL130 // introduced in GLSL 1.3

	// Protect ourselves against tracking huge amounts of texture state.
	const maxSaneSamplers = 32
	maxSamplers := int(getIntegerv(f, MAX_TEXTURE_IMAGE_UNITS))
	if maxSamplers > maxSaneSamplers {
		maxSamplers = maxSaneSamplers
	}
	shaderCaps.MaxFragmentSamplers = maxSamplers

	// Tiled-architecture GPUs benefit from non-VBO vertex data for dynamic content, but client side buffers are not
	// allowed in core profiles.
	if !c.isCoreProfile &&
		(ctxInfo.Vendor() == VendorARM || ctxInfo.Vendor() == VendorImagination ||
			ctxInfo.Vendor() == VendorQualcomm) {
		c.PreferClientSideDynamicBuffers = true
	}

	if !options.AvoidStencilBuffers {
		// To reduce surface area, if we avoid stencil buffers, we also disable MSAA.
		c.initFSAASupport(ctxInfo)
		c.initStencilSupport(ctxInfo)
	}

	// Setup blit framebuffer.
	c.blitFramebufferFlags = NoSupportBlitFramebufferFlag
	if version >= Ver(3, 0) || ctxInfo.HasExtension("GL_ARB_framebuffer_object") ||
		ctxInfo.HasExtension("GL_EXT_framebuffer_blit") {
		c.blitFramebufferFlags = 0
	}

	c.initBlendEquationSupport(ctxInfo)

	// We require VBO support and the desktop VBO extension includes glMapBuffer.
	c.MapBufferFlags = gpu.CanMapFlag
	if version >= Ver(3, 0) || ctxInfo.HasExtension("GL_ARB_map_buffer_range") {
		c.MapBufferFlags |= gpu.SubsetMapFlag
		c.mapBufferType = MapBufferMapBufferRange
	} else {
		c.mapBufferType = MapBufferMapBuffer
	}

	if version >= Ver(2, 1) || ctxInfo.HasExtension("GL_ARB_pixel_buffer_object") ||
		ctxInfo.HasExtension("GL_EXT_pixel_buffer_object") {
		c.TransferFromBufferToTextureSupport = true
		c.TransferFromSurfaceToBufferSupport = true
		c.transferBufferType = TransferBufferARBPBO
	}
	if version >= Ver(3, 1) || ctxInfo.HasExtension("GL_ARB_copy_buffer") {
		c.TransferFromBufferToBufferSupport = true
	}

	// On many GPUs map memory is very expensive, so effectively disable it unless the client hints that map memory is
	// cheap.
	if c.BufferMapThreshold < 0 {
		c.BufferMapThreshold = math.MaxInt32
	}

	c.NPOTTextureTileSupport = true
	c.MipmapSupport = true
	c.AnisoSupport = version >= Ver(4, 6) ||
		ctxInfo.HasExtension("GL_ARB_texture_filter_anisotropic") ||
		ctxInfo.HasExtension("GL_EXT_texture_filter_anisotropic")
	c.maxTextureMaxAnisotropy = 1
	if c.AnisoSupport {
		c.maxTextureMaxAnisotropy = getFloatv(f, MAX_TEXTURE_MAX_ANISOTROPY)
	}

	c.MaxTextureSize = int(getIntegerv(f, MAX_TEXTURE_SIZE))
	c.MaxRenderTargetSize = int(getIntegerv(f, MAX_RENDERBUFFER_SIZE))
	c.MaxPreferredRenderTargetSize = c.MaxRenderTargetSize
	if ctxInfo.Vendor() == VendorARM && c.MaxPreferredRenderTargetSize > 4096 {
		// On Mali G71, RTs above 4k have been observed to incur a performance cost.
		c.MaxPreferredRenderTargetSize = 4096
	}

	c.GpuTracingSupport = ctxInfo.HasExtension("GL_EXT_debug_marker")

	// Disable scratch texture reuse on Mali and Adreno devices.
	c.ReuseScratchTextures = ctxInfo.Vendor() != VendorARM

	if ctxInfo.HasExtension("GL_EXT_window_rectangles") {
		c.MaxWindowRectanglesCap = int(getIntegerv(f, MAX_WINDOW_RECTANGLES))
	}

	// ARB allows mixed size FBO attachments, EXT does not.
	c.OversizedStencilSupport = version >= Ver(3, 0) ||
		ctxInfo.HasExtension("GL_ARB_framebuffer_object")

	c.baseVertexBaseInstanceSupport = version >= Ver(4, 2) ||
		ctxInfo.HasExtension("GL_ARB_base_instance")
	if c.baseVertexBaseInstanceSupport {
		c.NativeDrawIndirectSupport = version >= Ver(4, 0) ||
			ctxInfo.HasExtension("GL_ARB_draw_indirect")
		if version >= Ver(4, 3) || ctxInfo.HasExtension("GL_ARB_multi_draw_indirect") {
			c.multiDrawType = MultiDrawIndirect
		}
	}
	c.drawRangeElementsSupport = version >= Ver(2, 0)

	// We do not support backend semaphores for GL because clients cannot make GL sync objects ahead of time without
	// talking to the GPU.
	c.BackendSemaphoreSupport = false
	// We prefer GL sync objects (which also implement semaphore semantics) but also support NV_fence.
	if version >= Ver(3, 2) || ctxInfo.HasExtension("GL_ARB_sync") {
		c.SemaphoreSupport = true
		c.fenceSyncSupport = true
		c.fenceType = FenceSyncObject
	} else if ctxInfo.HasExtension("GL_NV_fence") {
		c.fenceSyncSupport = true
		c.fenceType = FenceNVFence
	}
	c.FinishedProcAsyncCallbackSupport = c.fenceSyncSupport

	if version >= Ver(3, 3) || ctxInfo.HasExtension("GL_EXT_timer_query") ||
		ctxInfo.HasExtension("GL_ARB_timer_query") {
		c.timerQueryType = TimerQueryRegular
	}

	// Safely moving textures between contexts requires semaphores.
	c.CrossContextTextureSupport = c.SemaphoreSupport

	// Half float vertex attributes require GL3.
	c.HalfFloatVertexAttributeSupport = version >= Ver(3, 0)

	c.DynamicStateArrayGeometryProcessorTextureSupport = true

	c.programBinarySupport = version >= Ver(4, 1)
	c.programParameterSupport = version >= Ver(4, 1)
	if c.programBinarySupport {
		count := getIntegerv(f, NUM_PROGRAM_BINARY_FORMATS)
		if count > 0 {
			formats := make([]int32, count)
			f.GetIntegerv(PROGRAM_BINARY_FORMATS, &formats[0])
			c.programBinaryFormats = make([]uint32, count)
			for i, v := range formats {
				c.programBinaryFormats[i] = uint32(v)
			}
		} else {
			c.programBinarySupport = false
		}
	}

	c.samplerObjectSupport = version >= Ver(3, 3) || ctxInfo.HasExtension("GL_ARB_sampler_objects")
	// We currently use sampler objects whenever they are available.
	c.useSamplerObjects = c.samplerObjectSupport

	if ctxInfo.Vendor() == VendorARM {
		c.ShouldCollapseSrcOverToSrcWhenAble = true
	}

	if !options.DisableDriverCorrectnessWorkarounds {
		c.applyDriverCorrectnessWorkarounds(ctxInfo, options)
	}

	// Requires MSAA support and ES compatibility to have already been detected.
	c.initFormatTable(ctxInfo)
	c.setupSampleCounts(ctxInfo, f)

	c.finishInitialization(options)

	// Besides the user-specifiable override we also want to avoid stencil buffers in Protected mode; protected content
	// is excluded by the trim, leaving just the option.
	c.AvoidStencilBuffers = options.AvoidStencilBuffers

	// For now these two are equivalent, but we could have dst read in shader via some other method.
	shaderCaps.DstReadInShaderSupport = shaderCaps.FBFetchSupport
}

// glslVersionDeclString returns the "#version" declaration line for the given GLSL generation (desktop switch).
func glslVersionDeclString(generation gpu.GLSLGeneration, isCoreProfile bool) string {
	switch generation {
	case gpu.GLSL110:
		return "#version 110\n"
	case gpu.GLSL130:
		return "#version 130\n"
	case gpu.GLSL140:
		return "#version 140\n"
	case gpu.GLSL150:
		if isCoreProfile {
			return "#version 150\n"
		}
		return "#version 150 compatibility\n"
	case gpu.GLSL330:
		if isCoreProfile {
			return "#version 330\n"
		}
		return "#version 330 compatibility\n"
	case gpu.GLSL400:
		if isCoreProfile {
			return "#version 400\n"
		}
		return "#version 400 compatibility\n"
	case gpu.GLSL420:
		if isCoreProfile {
			return "#version 420\n"
		}
		return "#version 420 compatibility\n"
	default:
		return "<no version>"
	}
}

// isFloatFP32 reports whether the given shader precision qualifies as full 32-bit float.
func isFloatFP32(ctxInfo *ContextInfo, f *Functions, precision uint32) bool {
	if ctxInfo.Version() < Ver(4, 1) && !ctxInfo.HasExtension("GL_ARB_ES2_compatibility") {
		// We're on a desktop GL that doesn't have precision info. Assume they're all 32-bit float.
		return true
	}
	// glGetShaderPrecisionFormat doesn't accept GL_GEOMETRY_SHADER as a shader type. Hopefully geometry shaders don't
	// have lower precision than vertex and fragment.
	for _, shader := range []uint32{FRAGMENT_SHADER, VERTEX_SHADER} {
		var rng [2]int32
		var bits int32
		f.GetShaderPrecisionFormat(shader, precision, &rng[0], &bits)
		if rng[0] < 127 || rng[1] < 127 || bits < 23 {
			return false
		}
	}
	return true
}

// initGLSL populates the shader-capability fields that depend on the GLSL generation (desktop branches).
func (c *Caps) initGLSL(ctxInfo *ContextInfo, f *Functions) {
	version := ctxInfo.Version()
	shaderCaps := c.ShaderCaps
	shaderCaps.GLSLGeneration = ctxInfo.GLSLGeneration

	if ctxInfo.HasExtension("GL_EXT_shader_framebuffer_fetch") {
		shaderCaps.FBFetchNeedsCustomOutput = version >= Ver(3, 0)
		shaderCaps.FBFetchSupport = true
		shaderCaps.FBFetchColorName = "gl_LastFragData[0]"
		shaderCaps.FBFetchExtensionString = "GL_EXT_shader_framebuffer_fetch"
	}

	shaderCaps.FlatInterpolationSupport = ctxInfo.GLSLGeneration >= gpu.GLSL130
	// Flat interpolation is slow on Qualcomm GPUs; the ANGLE/WebGL exclusions are trimmed.
	shaderCaps.PreferFlatInterpolation = shaderCaps.FlatInterpolationSupport &&
		ctxInfo.Vendor() != VendorQualcomm

	shaderCaps.NoPerspectiveInterpolationSupport = ctxInfo.GLSLGeneration >= gpu.GLSL130

	shaderCaps.SampleMaskSupport = ctxInfo.GLSLGeneration >= gpu.GLSL400

	shaderCaps.VersionDeclString = glslVersionDeclString(shaderCaps.GLSLGeneration, c.isCoreProfile)

	shaderCaps.VertexIDSupport = true

	// isinf() exists in GLSL 1.3 and above, but hardware without proper IEEE support is allowed to always return false,
	// so it's potentially meaningless. In GLSL 3.3 isinf() is required to actually identify infinite values.
	shaderCaps.InfinitySupport = ctxInfo.GLSLGeneration >= gpu.GLSL330

	shaderCaps.NonconstantArrayIndexSupport = true

	shaderCaps.BitManipulationSupport = ctxInfo.GLSLGeneration >= gpu.GLSL400

	shaderCaps.FloatIs32Bits = isFloatFP32(ctxInfo, f, HIGH_FLOAT)
	shaderCaps.HalfIs32Bits = isFloatFP32(ctxInfo, f, MEDIUM_FLOAT)
	shaderCaps.HasLowFragmentPrecision = false

	shaderCaps.BuiltinFMASupport = ctxInfo.GLSLGeneration >= gpu.GLSL400
	shaderCaps.BuiltinDeterminantSupport = ctxInfo.GLSLGeneration >= gpu.GLSL150
}

// initFSAASupport detects multisampled-FBO support (desktop branch).
func (c *Caps) initFSAASupport(ctxInfo *ContextInfo) {
	if ctxInfo.Version() >= Ver(3, 0) || ctxInfo.HasExtension("GL_ARB_framebuffer_object") {
		c.msFBOType = MSFBOStandard
	} else if ctxInfo.HasExtension("GL_EXT_framebuffer_multisample") &&
		ctxInfo.HasExtension("GL_EXT_framebuffer_blit") {
		c.msFBOType = MSFBOStandard
	}
}

// initBlendEquationSupport detects advanced-blend-equation support.
func (c *Caps) initBlendEquationSupport(ctxInfo *ContextInfo) {
	shaderCaps := c.ShaderCaps
	layoutQualifierSupport := shaderCaps.GLSLGeneration >= gpu.GLSL140

	switch {
	case ctxInfo.HasExtension("GL_NV_blend_equation_advanced_coherent"):
		c.BlendEquationSupport = gpu.AdvancedCoherentBlendEquationSupport
		shaderCaps.AdvBlendEqInteraction = gpu.AutomaticAdvBlendEqInteraction
	case ctxInfo.HasExtension("GL_KHR_blend_equation_advanced_coherent") && layoutQualifierSupport:
		c.BlendEquationSupport = gpu.AdvancedCoherentBlendEquationSupport
		shaderCaps.AdvBlendEqInteraction = gpu.GeneralEnableAdvBlendEqInteraction
	case ctxInfo.HasExtension("GL_NV_blend_equation_advanced"):
		c.BlendEquationSupport = gpu.AdvancedBlendEquationSupport
		shaderCaps.AdvBlendEqInteraction = gpu.AutomaticAdvBlendEqInteraction
	case ctxInfo.HasExtension("GL_KHR_blend_equation_advanced") && layoutQualifierSupport:
		c.BlendEquationSupport = gpu.AdvancedBlendEquationSupport
		shaderCaps.AdvBlendEqInteraction = gpu.GeneralEnableAdvBlendEqInteraction
	}
}

// initStencilSupport populates the list of legal stencil formats — though perhaps not supported by the particular
// GPU/driver — from most preferred to least.
func (c *Caps) initStencilSupport(ctxInfo *ContextInfo) {
	supportsPackedDS := ctxInfo.Version() >= Ver(3, 0) ||
		ctxInfo.HasExtension("GL_EXT_packed_depth_stencil") ||
		ctxInfo.HasExtension("GL_ARB_framebuffer_object")

	// S1 through S16 formats are in GL 3.0+, EXT_FBO, and ARB_FBO; since we require FBO support these are legal formats
	// and we don't check.
	c.stencilFormats = append(c.stencilFormats, FormatSTENCILINDEX8, FormatSTENCILINDEX16)
	if supportsPackedDS {
		c.stencilFormats = append(c.stencilFormats, FormatDEPTH24STENCIL8)
	}
}

// applyDriverCorrectnessWorkarounds applies the desktop-reachable subset of known driver correctness workarounds; see
// the file comment for the trim rationale.
func (c *Caps) applyDriverCorrectnessWorkarounds(ctxInfo *ContextInfo, options *gpu.ContextOptions) {
	isMac := runtime.GOOS == "darwin"
	shaderCaps := c.ShaderCaps

	if c.DriverBugWorkarounds.DisableDiscardFramebuffer {
		c.invalidateFBType = InvalidateFBNone
	}

	// On linux Intel HD405 (CherryView) blending with a dst after a texture-barrier draw behaves as if the barrier draw
	// didn't happen; disable texture barrier support for the GPU.
	if ctxInfo.Renderer() == RendererIntelCherryView {
		c.TextureBarrierSupport = false
	}

	// glClearTexImage has a bug in NVIDIA drivers fixed between 340.96 and 367.57.
	if ctxInfo.Driver() == DriverNVIDIA && ctxInfo.DriverVersion() < DriverVer(367, 57, 0) {
		c.clearTextureSupport = false
	}

	// A driver bug on Intel Sandy Bridge with the Intel (non-Mesa) driver: mapping is broken and a shader compilation
	// error appears with nonsquare matrices (see skbug.com/40040603).
	if ctxInfo.Renderer() == RendererIntelSandyBridge && ctxInfo.Driver() == DriverIntel {
		c.mapBufferType = MapBufferNone
		c.MapBufferFlags = gpu.NoMapFlags
		shaderCaps.NonsquareMatrixSupport = false
	}

	// A lot of GPUs have trouble with full screen clears (skbug.com/40038435).
	if ctxInfo.Renderer() == RendererAMDRadeonHD7xxx ||
		ctxInfo.Renderer() == RendererAMDRadeonR9M4xx {
		c.PerformColorClearsAsDraws = true
	}
	if isMac {
		// crbug.com/768134 + crbug.com/773107 + crbug.com/1039912: on MacBook Pros a wide range of Intel GPUs don't
		// always perform full screen clears; fixed in driver 10.30.12 except for Broadwell.
		if ctxInfo.Vendor() == VendorIntel &&
			(ctxInfo.DriverVersion() < DriverVer(10, 30, 12) ||
				ctxInfo.Renderer() == RendererIntelBroadwell) {
			c.PerformColorClearsAsDraws = true
		}
		// crbug.com/969609: NVIDIA on Mac sometimes segfaults during glClear.
		if ctxInfo.Vendor() == VendorNVIDIA {
			c.PerformColorClearsAsDraws = true
		}
	}

	if c.DriverBugWorkarounds.GLClearBroken {
		c.PerformColorClearsAsDraws = true
		c.PerformStencilClearsAsDraws = true
	}

	if ctxInfo.Vendor() == VendorQualcomm {
		// All Adreno GPUs have less than optimal performance drawing with large index buffers.
		c.AvoidLargeIndexBufferDraws = true
	}

	// We support manual mipmap generation (via iterative downsampling draw calls). This fixes bugs on some
	// cards/drivers that produce incorrect mipmaps for sRGB textures when using glGenerateMipmap, and requires
	// mip-level sampling control.
	if c.mipmapLevelControlSupport &&
		(options.DoManualMipmapping ||
			ctxInfo.Vendor() == VendorIntel ||
			(ctxInfo.Driver() == DriverNVIDIA && isMac) ||
			ctxInfo.Vendor() == VendorATI) {
		c.doManualMipmapping = true
	}

	// See http://crbug.com/710443: certain Intel GPUs on Mac fail to clear if the glClearColor is made up of only 1s
	// and 0s.
	if isMac && ctxInfo.Renderer() == RendererIntelBroadwell {
		c.clearToBoundaryValuesIsBroken = true
	}

	// https://b.corp.google.com/issues/188410972: running over virgl.
	if ctxInfo.DriverInfo.IsRunningOverVirgl {
		c.DrawInstancedSupport = false
	}

	// http://anglebug.com/4538
	if c.baseVertexBaseInstanceSupport && !c.DrawInstancedSupport {
		c.baseVertexBaseInstanceSupport = false
		c.NativeDrawIndirectSupport = false
		c.multiDrawType = MultiDrawNone
	}

	// On Intel GPUs there is an issue where the driver reads the second argument to atan as an int.
	if ctxInfo.Vendor() == VendorIntel {
		shaderCaps.MustForceNegatedAtanParamToFloat = true
	}
	if isMac && ctxInfo.Vendor() == VendorATI {
		// The Radeon GLSL compiler on Mac gets confused by ldexp(..., -x). skbug.com/40043167
		shaderCaps.MustForceNegatedLdexpParamToMultiply = true
	}
	// On some Intel GPUs the driver outputs bogus values in the shader when floor and abs are called on the same line.
	if ctxInfo.Vendor() == VendorIntel {
		shaderCaps.MustDoOpBetweenFloorAndAbs = true
	}

	if c.DriverBugWorkarounds.AddAndTrueToLoopCondition {
		shaderCaps.AddAndTrueToLoopCondition = true
	}
	if c.DriverBugWorkarounds.UnfoldShortCircuitAsTernaryOperation {
		shaderCaps.UnfoldShortCircuitAsTernary = true
	}
	if c.DriverBugWorkarounds.EmulateAbsIntFunction {
		shaderCaps.EmulateAbsIntFunction = true
	}
	if c.DriverBugWorkarounds.RewriteDoWhileLoops {
		shaderCaps.RewriteDoWhileLoops = true
	}
	if c.DriverBugWorkarounds.RemovePowWithConstantExponent {
		shaderCaps.RemovePowWithConstantExponent = true
	}
	if c.DriverBugWorkarounds.DisableDualSourceBlendingSupport {
		shaderCaps.DualSourceBlendingSupport = false
	}

	// Disabling advanced blend on platforms with major known issues.
	if ctxInfo.Driver() == DriverIntel {
		c.BlendEquationSupport = gpu.BasicBlendEquationSupport
		shaderCaps.AdvBlendEqInteraction = gpu.NotSupportedAdvBlendEqInteraction
	}
	// Non-coherent advanced blend has an issue on NVIDIA pre 337.00.
	if ctxInfo.Driver() == DriverNVIDIA && ctxInfo.DriverVersion() < DriverVer(337, 0, 0) &&
		c.BlendEquationSupport == gpu.AdvancedBlendEquationSupport {
		c.BlendEquationSupport = gpu.BasicBlendEquationSupport
		shaderCaps.AdvBlendEqInteraction = gpu.NotSupportedAdvBlendEqInteraction
	}
	if c.DriverBugWorkarounds.DisableBlendEquationAdvanced {
		c.BlendEquationSupport = gpu.BasicBlendEquationSupport
		shaderCaps.AdvBlendEqInteraction = gpu.NotSupportedAdvBlendEqInteraction
	}
	if c.AdvancedBlendEquationSupport() {
		if ctxInfo.Driver() == DriverNVIDIA && ctxInfo.DriverVersion() < DriverVer(355, 0, 0) {
			// Disable color-dodge and color-burn on pre-355.00 NVIDIA.
			c.AdvBlendEqDisableFlags |= (1 << uint32(gpu.BlendEquationColorDodge)) |
				(1 << uint32(gpu.BlendEquationColorBurn))
		}
		if ctxInfo.Vendor() == VendorARM {
			// Disable color-burn on ARM until the fix is released.
			c.AdvBlendEqDisableFlags |= 1 << uint32(gpu.BlendEquationColorBurn)
		}
	}

	// IntelIris640 drops draws completely; other Intel parts draw corrupted curves.
	if ctxInfo.Vendor() == VendorIntel {
		c.DisableTessellationPathRendererCap = true
	}

	// We disable MSAA for older Intel GPUs: before Gen9 performance was very bad, and even with Gen9 driver crashes
	// were seen in the wild (crbug.com/527565, crbug.com/983926).
	if ctxInfo.Vendor() == VendorIntel {
		if ctxInfo.Renderer() >= RendererIntelIceLake && options.AllowMSAAOnNewIntel {
			// Gen11 seems mostly ok, except we avoid drawing lines with MSAA (anglebug.com/7796).
			if c.msFBOType != MSFBONone {
				c.AvoidLineDraws = true
			}
		} else {
			c.msFBOType = MSFBONone
		}
	}

	// skbug.com/40045272: setting the max level makes it clear to Intel drivers that a rendering feedback loop is not
	// occurring during mipmap regeneration, avoiding a slow path.
	if ctxInfo.Vendor() == VendorIntel {
		c.regenerateMipmapType = RegenerateMipmapBasePlusMaxLevel
	}

	if isMac && ctxInfo.Vendor() == VendorApple {
		// skbug.com/398631003: bind the framebuffer to 0 when presenting a surface.
		c.bindDefaultFramebufferOnPresent = true
	}
}

// finishInitialization applies base-caps and GL-specific option overrides, inlined into one ordered sequence.
func (c *Caps) finishInitialization(options *gpu.ContextOptions) {
	if !c.NativeDrawIndirectSupport {
		// We will implement indirect draws with a polyfill, so the commands need to reside in CPU memory.
		c.UseClientSideIndirectBuffers = true
	}

	// Base caps option overrides:
	c.ShaderCaps.ApplyOptionsOverrides(options)

	// GL-specific option overrides:
	if options.ShaderCacheStrategy < gpu.ShaderCacheStrategyBackendBinary {
		c.programBinarySupport = false
	}
	switch options.SkipGLErrorChecks {
	case gpu.EnableNo:
		c.skipErrorChecks = false
	case gpu.EnableYes:
		c.skipErrorChecks = true
	case gpu.EnableDefault:
	}

	switch options.UseDrawInsteadOfClear {
	case gpu.EnableNo:
		c.PerformColorClearsAsDraws = false
		c.PerformStencilClearsAsDraws = false
	case gpu.EnableYes:
		c.PerformColorClearsAsDraws = true
		c.PerformStencilClearsAsDraws = true
	case gpu.EnableDefault:
	}

	if c.MaxTextureSize > options.MaxTextureSizeOverride {
		c.MaxTextureSize = options.MaxTextureSizeOverride
	}
	if options.SuppressMipmapSupport {
		c.MipmapSupport = false
	}
	if c.MaxWindowRectanglesCap > gpu.MaxWindowRectangles {
		c.MaxWindowRectanglesCap = gpu.MaxWindowRectangles
	}
	c.InternalMultisampleCountCap = options.InternalMultisampleCount
	c.AvoidStencilBuffers = options.AvoidStencilBuffers
	c.DriverBugWorkarounds.ApplyOverrides(&options.DriverBugWorkarounds)
	if options.DisableTessellationPathRenderer {
		c.DisableTessellationPathRendererCap = true
	}

	// Our render targets are always created with textures as the color attachment, hence this min:
	if c.MaxRenderTargetSize > c.MaxTextureSize {
		c.MaxRenderTargetSize = c.MaxTextureSize
	}
	if c.MaxPreferredRenderTargetSize > c.MaxRenderTargetSize {
		c.MaxPreferredRenderTargetSize = c.MaxRenderTargetSize
	}
}
