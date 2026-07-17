// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Builds the format capability table and sample-count table, desktop branches only and bounded to the supported
// color-type matrix. Formats outside the matrix (LUMINANCE8_ALPHA8, RGBX8, RG8, RGB10_A2, RGBA4, SRGB8_ALPHA8,
// LUMINANCE16F, R16, RG16, RGBA16, RG16F, and the compressed formats) keep zeroed table entries and report unsupported
// everywhere a populated row would otherwise be consulted.

package gl

import "github.com/richardwilkes/canvas/gpu"

func (c *Caps) setColorTypeFormat(colorType gpu.ColorType, format Format) {
	c.colorTypeToFormatTable[colorType] = format
}

// initFormatTable populates the per-format capability table for every format in the supported color-type matrix.
func (c *Caps) initFormatTable(ctxInfo *ContextInfo) {
	version := ctxInfo.Version()

	nonMSAARenderFlags := fboColorAttachmentFormatFlag
	msaaRenderFlags := nonMSAARenderFlags
	if c.msFBOType != MSFBONone {
		msaaRenderFlags |= fboColorAttachmentWithMSAAFormatFlag
	}

	texStorageSupported := version >= Ver(4, 2) ||
		ctxInfo.HasExtension("GL_ARB_texture_storage") ||
		ctxInfo.HasExtension("GL_EXT_texture_storage")
	if c.DriverBugWorkarounds.DisableTextureStorage {
		texStorageSupported = false
	}

	// Desktop GL always supports sized internal formats with glTexImage (the unsized fallback is an ES 2.0 concern) and
	// floating-point MSAA.
	fpRenderFlags := msaaRenderFlags

	for i := range c.colorTypeToFormatTable {
		c.colorTypeToFormatTable[i] = FormatUnknown
	}

	halfFloatType := uint32(HALF_FLOAT)

	// Format: RGBA8
	{
		info := &c.formatTable[FormatRGBA8]
		info.formatType = formatTypeNormalizedFixedPoint
		info.internalFormatForRenderbuffer = RGBA8
		info.defaultExternalFormat = RGBA
		info.defaultExternalType = UNSIGNED_BYTE
		info.defaultColorType = gpu.ColorTypeRGBA8888
		info.flags = texturableFormatFlag | transfersFormatFlag | msaaRenderFlags
		if texStorageSupported {
			info.flags |= useTexStorageFormatFlag
		}
		info.internalFormatForTexImageOrStorage = RGBA8

		supportsBGRAColorType := version >= Ver(1, 2) || ctxInfo.HasExtension("GL_EXT_bgra")

		// Format: RGBA8, Surface: kRGBA_8888
		{
			cti := colorTypeInfo{
				colorType:    gpu.ColorTypeRGBA8888,
				flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: RGBA8, Surface: kRGBA_8888, Data: kRGBA_8888
					{
						colorType:              gpu.ColorTypeRGBA8888,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: RGBA,
						externalReadFormat:     RGBA,
					},
					// Format: RGBA8, Surface: kRGBA_8888, Data: kBGRA_8888
					{
						colorType:              gpu.ColorTypeBGRA8888,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0, // TODO: enable this on non-ES GL.
						externalReadFormat:     BGRA,
					},
				},
			}
			info.colorTypeInfos = append(info.colorTypeInfos, cti)
			c.setColorTypeFormat(gpu.ColorTypeRGBA8888, FormatRGBA8)
		}

		// Format: RGBA8, Surface: kBGRA_8888
		if supportsBGRAColorType {
			cti := colorTypeInfo{
				colorType:    gpu.ColorTypeBGRA8888,
				flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: RGBA8, Surface: kBGRA_8888, Data: kBGRA_8888
					{
						colorType:              gpu.ColorTypeBGRA8888,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: BGRA,
						externalReadFormat:     BGRA,
					},
					// Format: RGBA8, Surface: kBGRA_8888, Data: kRGBA_8888
					{
						colorType:              gpu.ColorTypeRGBA8888,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			}
			info.colorTypeInfos = append(info.colorTypeInfos, cti)
			c.setColorTypeFormat(gpu.ColorTypeBGRA8888, FormatRGBA8)
		}

		// Format: RGBA8, Surface: kRGB_888x
		{
			cti := colorTypeInfo{
				colorType:    gpu.ColorTypeRGB888x,
				flags:        uploadDataColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGB1,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: RGBA8, Surface: kRGB_888x, Data: kRGB_888x
					{
						colorType:              gpu.ColorTypeRGB888x,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: RGBA,
						externalReadFormat:     RGBA,
					},
				},
			}
			info.colorTypeInfos = append(info.colorTypeInfos, cti)
		}
	}

	// Format: R8
	{
		info := &c.formatTable[FormatR8]
		info.formatType = formatTypeNormalizedFixedPoint
		info.internalFormatForRenderbuffer = R8
		info.defaultExternalFormat = RED
		info.defaultExternalType = UNSIGNED_BYTE
		info.defaultColorType = gpu.ColorTypeR8
		r8Support := version >= Ver(3, 0) || ctxInfo.HasExtension("GL_ARB_texture_rg")
		if r8Support {
			info.flags |= texturableFormatFlag | transfersFormatFlag | msaaRenderFlags
		}
		if texStorageSupported {
			info.flags |= useTexStorageFormatFlag
		}
		info.internalFormatForTexImageOrStorage = R8

		if r8Support {
			// Format: R8, Surface: kR_8
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeR8,
				flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: R8, Surface: kR_8, Data: kR_8
					{
						colorType:              gpu.ColorTypeR8,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: RED,
						externalReadFormat:     RED,
					},
					// Format: R8, Surface: kR_8, Data: kR_8xxx
					{
						colorType:              gpu.ColorTypeR8xxx,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeR8, FormatR8)

			// Format: R8, Surface: kAlpha_8
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeAlpha8,
				flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
				readSwizzle:  gpu.MakeSwizzle("000r"),
				writeSwizzle: gpu.MakeSwizzle("a000"),
				externalIOFormats: []externalIOFormat{
					// Format: R8, Surface: kAlpha_8, Data: kAlpha_8
					{
						colorType:              gpu.ColorTypeAlpha8,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: RED,
						externalReadFormat:     RED,
					},
					// Format: R8, Surface: kAlpha_8, Data: kAlpha_8xxx
					{
						colorType:              gpu.ColorTypeAlpha8xxx,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeAlpha8, FormatR8)

			// Format: R8, Surface: kGray_8
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeGray8,
				flags:        uploadDataColorTypeFlag,
				readSwizzle:  gpu.MakeSwizzle("rrr1"),
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: R8, Surface: kGray_8, Data: kGray_8
					{
						colorType:              gpu.ColorTypeGray8,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: RED,
						externalReadFormat:     RED,
					},
					// Format: R8, Surface: kGray_8, Data: kGray_8xxx
					{
						colorType:              gpu.ColorTypeGray8xxx,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeGray8, FormatR8)
		}
	}

	// Format: ALPHA8
	{
		// Core profile removes ALPHA8 support.
		alpha8IsValidForGL := !c.isCoreProfile || version <= Ver(3, 0)

		info := &c.formatTable[FormatALPHA8]
		info.formatType = formatTypeNormalizedFixedPoint
		alpha8SizedEnumSupported := alpha8IsValidForGL
		alpha8TexStorageSupported := alpha8SizedEnumSupported && texStorageSupported

		// OpenGL 3.0+ (and GL_ARB_framebuffer_object) supports ALPHA8 as renderable.
		alpha8IsRenderable := alpha8IsValidForGL &&
			(version >= Ver(3, 0) || ctxInfo.HasExtension("GL_ARB_framebuffer_object"))
		info.internalFormatForRenderbuffer = ALPHA8
		info.defaultExternalFormat = ALPHA
		info.defaultExternalType = UNSIGNED_BYTE
		info.defaultColorType = gpu.ColorTypeAlpha8
		if alpha8IsValidForGL {
			info.flags = texturableFormatFlag | transfersFormatFlag
		}
		if alpha8IsRenderable && alpha8IsValidForGL {
			// We will use ALPHA8 to create MSAA renderbuffers.
			info.flags |= msaaRenderFlags
		}
		switch {
		case alpha8TexStorageSupported:
			info.flags |= useTexStorageFormatFlag
			info.internalFormatForTexImageOrStorage = ALPHA8
		case alpha8SizedEnumSupported:
			info.internalFormatForTexImageOrStorage = ALPHA8
		default:
			info.internalFormatForTexImageOrStorage = ALPHA
		}

		if alpha8IsValidForGL {
			// Format: ALPHA8, Surface: kAlpha_8
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeAlpha8,
				flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: ALPHA8, Surface: kAlpha_8, Data: kAlpha_8
					{
						colorType:              gpu.ColorTypeAlpha8,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: ALPHA,
						externalReadFormat:     ALPHA,
					},
					// Format: ALPHA8, Surface: kAlpha_8, Data: kRGBA_8888
					{
						colorType:              gpu.ColorTypeRGBA8888,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			if c.colorTypeToFormatTable[gpu.ColorTypeAlpha8] == FormatUnknown {
				c.setColorTypeFormat(gpu.ColorTypeAlpha8, FormatALPHA8)
			}
		}
	}

	// Format: LUMINANCE8
	{
		lum8Supported := !c.isCoreProfile

		info := &c.formatTable[FormatLUMINANCE8]
		info.formatType = formatTypeNormalizedFixedPoint
		info.internalFormatForRenderbuffer = LUMINANCE8
		info.defaultExternalFormat = LUMINANCE
		info.defaultExternalType = UNSIGNED_BYTE
		info.defaultColorType = gpu.ColorTypeGray8
		if lum8Supported {
			info.flags = texturableFormatFlag | transfersFormatFlag
			if texStorageSupported {
				info.flags |= useTexStorageFormatFlag
			}
			info.internalFormatForTexImageOrStorage = LUMINANCE8
		} else {
			info.internalFormatForTexImageOrStorage = LUMINANCE
		}
		// FBO attachment for LUMINANCE8 is not enabled: the specs never clearly made it color-renderable.

		if lum8Supported {
			// Format: LUMINANCE8, Surface: kGray_8
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeGray8,
				flags:        uploadDataColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: LUMINANCE8, Surface: kGray_8, Data: kGray_8
					{
						colorType:              gpu.ColorTypeGray8,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: LUMINANCE,
						externalReadFormat:     0,
					},
					// Format: LUMINANCE8, Surface: kGray_8, Data: kRGBA_8888
					{
						colorType:              gpu.ColorTypeRGBA8888,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			if c.colorTypeToFormatTable[gpu.ColorTypeGray8] == FormatUnknown {
				c.setColorTypeFormat(gpu.ColorTypeGray8, FormatLUMINANCE8)
			}
		}
	}

	// Format: BGRA8. On desktop GL the BGRA color type rides the RGBA8 format; the BGRA8 format itself gains
	// capabilities only through ES extensions, so the entry keeps its identity fields but no flags or color types.
	{
		info := &c.formatTable[FormatBGRA8]
		info.formatType = formatTypeNormalizedFixedPoint
		info.defaultExternalFormat = BGRA
		info.defaultExternalType = UNSIGNED_BYTE
		info.defaultColorType = gpu.ColorTypeBGRA8888
		// If BGRA is supported as an internal format it must always be specified to glTex[Sub]Image as a base format;
		// which base format depends on which (ES) extension is present.
		if ctxInfo.HasExtension("GL_EXT_texture_format_BGRA8888") {
			info.internalFormatForTexImageOrStorage = BGRA
		} else {
			info.internalFormatForTexImageOrStorage = RGBA
		}
	}

	// Format: RGB565
	{
		info := &c.formatTable[FormatRGB565]
		info.formatType = formatTypeNormalizedFixedPoint
		info.internalFormatForRenderbuffer = RGB565
		info.defaultExternalFormat = RGB
		info.defaultExternalType = UNSIGNED_SHORT_5_6_5
		info.defaultColorType = gpu.ColorTypeBGR565
		if version >= Ver(4, 2) || ctxInfo.HasExtension("GL_ARB_ES2_compatibility") {
			info.flags = texturableFormatFlag | transfersFormatFlag | msaaRenderFlags
		}
		// 565 is not a sized internal format on desktop GL, so TexStorage is disallowed; TexImage uses the sized format
		// and lets the system pick the best conversion.
		info.internalFormatForTexImageOrStorage = RGB565

		if info.flags&texturableFormatFlag != 0 {
			// Format: RGB565, Surface: kBGR_565
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeBGR565,
				flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: RGB565, Surface: kBGR_565, Data: kBGR_565
					{
						colorType:              gpu.ColorTypeBGR565,
						externalType:           UNSIGNED_SHORT_5_6_5,
						externalTexImageFormat: RGB,
						externalReadFormat:     RGB,
					},
					// Format: RGB565, Surface: kBGR_565, Data: kRGBA_8888
					{
						colorType:              gpu.ColorTypeRGBA8888,
						externalType:           UNSIGNED_BYTE,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeBGR565, FormatRGB565)
		}
	}

	// Format: RGBA16F
	{
		info := &c.formatTable[FormatRGBA16F]
		info.formatType = formatTypeFloat
		info.internalFormatForRenderbuffer = RGBA16F
		info.defaultExternalFormat = RGBA
		info.defaultExternalType = halfFloatType
		info.defaultColorType = gpu.ColorTypeRGBAF16
		rgba16FTextureSupport := false
		rgba16FRenderTargetSupport := false
		if version >= Ver(3, 0) {
			rgba16FTextureSupport = true
			rgba16FRenderTargetSupport = true
		} else if ctxInfo.HasExtension("GL_ARB_texture_float") {
			rgba16FTextureSupport = true
		}
		if rgba16FTextureSupport {
			info.flags = texturableFormatFlag | transfersFormatFlag
			if rgba16FRenderTargetSupport {
				info.flags |= fpRenderFlags
			}
		}
		if texStorageSupported {
			info.flags |= useTexStorageFormatFlag
		}
		info.internalFormatForTexImageOrStorage = RGBA16F

		if rgba16FTextureSupport {
			flags := uploadDataColorTypeFlag | renderableColorTypeFlag

			// Format: RGBA16F, Surface: kRGBA_F16
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeRGBAF16,
				flags:        flags,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: RGBA16F, Surface: kRGBA_F16, Data: kRGBA_F16
					{
						colorType:              gpu.ColorTypeRGBAF16,
						externalType:           halfFloatType,
						externalTexImageFormat: RGBA,
						externalReadFormat:     RGBA,
					},
					// Format: RGBA16F, Surface: kRGBA_F16, Data: kRGBA_F32
					{
						colorType:              gpu.ColorTypeRGBAF32,
						externalType:           FLOAT,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeRGBAF16, FormatRGBA16F)

			// Format: RGBA16F, Surface: kRGBA_F16_Clamped
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeRGBAF16Clamped,
				flags:        flags,
				readSwizzle:  gpu.SwizzleRGBA,
				writeSwizzle: gpu.SwizzleRGBA,
				externalIOFormats: []externalIOFormat{
					// Format: RGBA16F, Surface: kRGBA_F16_Clamped, Data: kRGBA_F16_Clamped
					{
						colorType:              gpu.ColorTypeRGBAF16Clamped,
						externalType:           halfFloatType,
						externalTexImageFormat: RGBA,
						externalReadFormat:     RGBA,
					},
					// Format: RGBA16F, Surface: kRGBA_F16_Clamped, Data: kRGBA_F32
					{
						colorType:              gpu.ColorTypeRGBAF32,
						externalType:           FLOAT,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeRGBAF16Clamped, FormatRGBA16F)

			// Format: RGBA16F, Surface: kRGB_F16F16F16x
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeRGBF16F16F16x,
				flags:        uploadDataColorTypeFlag,
				readSwizzle:  gpu.SwizzleRGB1,
				writeSwizzle: gpu.SwizzleRGB1,
				externalIOFormats: []externalIOFormat{
					// Format: RGBA16F, Surface: kRGB_F16F16F16x, Data: kRGB_F16F16F16x
					{
						colorType:              gpu.ColorTypeRGBF16F16F16x,
						externalType:           halfFloatType,
						externalTexImageFormat: RGBA,
						externalReadFormat:     RGBA,
					},
					// Format: RGBA16F, Surface: kRGB_F16F16F16x, Data: kRGBA_F32
					{
						colorType:              gpu.ColorTypeRGBAF32,
						externalType:           FLOAT,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeRGBF16F16F16x, FormatRGBA16F)
		}
	}

	// Format: R16F
	{
		info := &c.formatTable[FormatR16F]
		info.formatType = formatTypeFloat
		info.internalFormatForRenderbuffer = R16F
		info.defaultExternalFormat = RED
		info.defaultExternalType = halfFloatType
		info.defaultColorType = gpu.ColorTypeRF16
		r16FTextureSupport := false
		r16FRenderTargetSupport := false
		if version >= Ver(3, 0) || ctxInfo.HasExtension("GL_ARB_texture_rg") {
			r16FTextureSupport = true
			r16FRenderTargetSupport = true
		}
		if r16FTextureSupport {
			info.flags = texturableFormatFlag | transfersFormatFlag
			if r16FRenderTargetSupport {
				info.flags |= fpRenderFlags
			}
		}
		if texStorageSupported {
			info.flags |= useTexStorageFormatFlag
		}
		info.internalFormatForTexImageOrStorage = R16F

		if r16FTextureSupport {
			// Format: R16F, Surface: kAlpha_F16
			info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
				colorType:    gpu.ColorTypeAlphaF16,
				flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
				readSwizzle:  gpu.MakeSwizzle("000r"),
				writeSwizzle: gpu.MakeSwizzle("a000"),
				externalIOFormats: []externalIOFormat{
					// Format: R16F, Surface: kAlpha_F16, Data: kAlpha_F16
					{
						colorType:              gpu.ColorTypeAlphaF16,
						externalType:           halfFloatType,
						externalTexImageFormat: RED,
						externalReadFormat:     RED,
					},
					// Format: R16F, Surface: kAlpha_F16, Data: kAlpha_F32xxx
					{
						colorType:              gpu.ColorTypeAlphaF32xxx,
						externalType:           FLOAT,
						externalTexImageFormat: 0,
						externalReadFormat:     RGBA,
					},
				},
			})
			c.setColorTypeFormat(gpu.ColorTypeAlphaF16, FormatR16F)
		}
	}

	// Format: RGB8
	{
		info := &c.formatTable[FormatRGB8]
		info.formatType = formatTypeNormalizedFixedPoint
		info.internalFormatForRenderbuffer = RGB8
		info.defaultExternalFormat = RGB
		info.defaultExternalType = UNSIGNED_BYTE
		info.defaultColorType = gpu.ColorTypeRGB888
		// Even in OpenGL 4.6 GL_RGB8 is required to be color-renderable but not required to be a supported renderbuffer
		// format, and renderbuffers are what we use for MSAA on non-ES GL, so no MSAA for RGB8.
		info.flags = texturableFormatFlag | transfersFormatFlag | nonMSAARenderFlags
		if texStorageSupported {
			info.flags |= useTexStorageFormatFlag
		}
		info.internalFormatForTexImageOrStorage = RGB8

		// Format: RGB8, Surface: kRGB_888x
		info.colorTypeInfos = append(info.colorTypeInfos, colorTypeInfo{
			colorType:    gpu.ColorTypeRGB888x,
			flags:        uploadDataColorTypeFlag | renderableColorTypeFlag,
			readSwizzle:  gpu.SwizzleRGBA,
			writeSwizzle: gpu.SwizzleRGBA,
			externalIOFormats: []externalIOFormat{
				// Format: RGB8, Surface: kRGB_888x, Data: kRGB_888
				{
					colorType:              gpu.ColorTypeRGB888,
					externalType:           UNSIGNED_BYTE,
					externalTexImageFormat: RGB,
					externalReadFormat:     0,
				},
				// Format: RGB8, Surface: kRGB_888x, Data: kRGBA_8888
				{
					colorType:              gpu.ColorTypeRGBA8888,
					externalType:           UNSIGNED_BYTE,
					externalTexImageFormat: 0,
					externalReadFormat:     RGBA,
				},
			},
		})
		if c.colorTypeToFormatTable[gpu.ColorTypeRGB888x] == FormatUnknown {
			c.setColorTypeFormat(gpu.ColorTypeRGB888x, FormatRGB8)
		}
	}
}

// setupSampleCounts populates each format's supported MSAA sample-count list (desktop branches).
func (c *Caps) setupSampleCounts(ctxInfo *ContextInfo, f *Functions) {
	version := ctxInfo.Version()

	maxSampleCnt := int32(1)
	if c.msFBOType != MSFBONone {
		maxSampleCnt = getIntegerv(f, MAX_SAMPLES)
	}
	// Chrome has a mock GL implementation that returns 0.
	if maxSampleCnt < 1 {
		maxSampleCnt = 1
	}

	for i := range c.formatTable {
		info := &c.formatTable[i]
		switch {
		case info.flags&fboColorAttachmentWithMSAAFormatFlag != 0:
			// We assume MSAA rendering is supported only if we support non-MSAA rendering.
			if version >= Ver(4, 2) || ctxInfo.HasExtension("GL_ARB_internalformat_query") {
				glFormat := info.internalFormatForRenderbuffer
				var count int32
				f.GetInternalformativ(RENDERBUFFER, glFormat, NUM_SAMPLE_COUNTS, 1, &count)
				if count > 0 {
					temp := make([]int32, count)
					f.GetInternalformativ(RENDERBUFFER, glFormat, SAMPLES, count, &temp[0])
					// GL has a concept of MSAA rasterization with a single sample, but we do not.
					if temp[count-1] == 1 {
						count--
					}
					// We initialize our supported values with 1 (no msaa) and reverse the order returned by GL so that
					// the array is ascending.
					info.colorSampleCounts = make([]int, 0, count+1)
					info.colorSampleCounts = append(info.colorSampleCounts, 1)
					for j := int32(0); j < count; j++ {
						info.colorSampleCounts = append(info.colorSampleCounts, int(temp[count-j-1]))
					}
				}
			} else {
				// Fake out the table using some semi-standard counts up to the max allowed sample count.
				defaultSamples := []int{1, 2, 4, 8}
				count := len(defaultSamples)
				for ; count > 0; count-- {
					if defaultSamples[count-1] <= int(maxSampleCnt) {
						break
					}
				}
				if count > 0 {
					info.colorSampleCounts = append(info.colorSampleCounts, defaultSamples[:count]...)
				}
			}
		case info.flags&fboColorAttachmentFormatFlag != 0:
			info.colorSampleCounts = []int{1}
		}
	}
}
