// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"unsafe"

	"github.com/ebitengine/purego"
)

// GL constants
const (
	CONTEXT_PROFILE_MASK                         = 0x9126
	CONTEXT_CORE_PROFILE_BIT                     = 0x00000001
	CONTEXT_COMPATIBILITY_PROFILE_BIT            = 0x00000002
	DEPTH_BUFFER_BIT                             = 0x00000100
	STENCIL_BUFFER_BIT                           = 0x00000400
	COLOR_BUFFER_BIT                             = 0x00004000
	FALSE                                        = 0
	TRUE                                         = 1
	POINTS                                       = 0x0000
	LINES                                        = 0x0001
	LINE_LOOP                                    = 0x0002
	LINE_STRIP                                   = 0x0003
	TRIANGLES                                    = 0x0004
	TRIANGLE_STRIP                               = 0x0005
	TRIANGLE_FAN                                 = 0x0006
	PATCHES                                      = 0x000E
	FUNC_ADD                                     = 0x8006
	FUNC_SUBTRACT                                = 0x800A
	FUNC_REVERSE_SUBTRACT                        = 0x800B
	SCREEN                                       = 0x9295
	OVERLAY                                      = 0x9296
	DARKEN                                       = 0x9297
	LIGHTEN                                      = 0x9298
	COLORDODGE                                   = 0x9299
	COLORBURN                                    = 0x929A
	HARDLIGHT                                    = 0x929B
	SOFTLIGHT                                    = 0x929C
	DIFFERENCE                                   = 0x929E
	EXCLUSION                                    = 0x92A0
	MULTIPLY                                     = 0x9294
	HSL_HUE                                      = 0x92AD
	HSL_SATURATION                               = 0x92AE
	HSL_COLOR                                    = 0x92AF
	HSL_LUMINOSITY                               = 0x92B0
	ZERO                                         = 0
	ONE                                          = 1
	SRC_COLOR                                    = 0x0300
	ONE_MINUS_SRC_COLOR                          = 0x0301
	SRC_ALPHA                                    = 0x0302
	ONE_MINUS_SRC_ALPHA                          = 0x0303
	DST_ALPHA                                    = 0x0304
	ONE_MINUS_DST_ALPHA                          = 0x0305
	DST_COLOR                                    = 0x0306
	ONE_MINUS_DST_COLOR                          = 0x0307
	SRC_ALPHA_SATURATE                           = 0x0308
	SRC1_COLOR                                   = 0x88F9
	ONE_MINUS_SRC1_COLOR                         = 0x88FA
	ONE_MINUS_SRC1_ALPHA                         = 0x88FB
	BLEND_DST_RGB                                = 0x80C8
	BLEND_SRC_RGB                                = 0x80C9
	BLEND_DST_ALPHA                              = 0x80CA
	BLEND_SRC_ALPHA                              = 0x80CB
	CONSTANT_COLOR                               = 0x8001
	ONE_MINUS_CONSTANT_COLOR                     = 0x8002
	CONSTANT_ALPHA                               = 0x8003
	ONE_MINUS_CONSTANT_ALPHA                     = 0x8004
	BLEND_COLOR                                  = 0x8005
	ARRAY_BUFFER                                 = 0x8892
	ELEMENT_ARRAY_BUFFER                         = 0x8893
	DRAW_INDIRECT_BUFFER                         = 0x8F3F
	TEXTURE_BUFFER                               = 0x8C2A
	ARRAY_BUFFER_BINDING                         = 0x8894
	ELEMENT_ARRAY_BUFFER_BINDING                 = 0x8895
	DRAW_INDIRECT_BUFFER_BINDING                 = 0x8F43
	PIXEL_PACK_BUFFER                            = 0x88EB
	PIXEL_UNPACK_BUFFER                          = 0x88EC
	PIXEL_UNPACK_TRANSFER_BUFFER_CHROMIUM        = 0x78EC
	PIXEL_PACK_TRANSFER_BUFFER_CHROMIUM          = 0x78ED
	STREAM_DRAW                                  = 0x88E0
	STREAM_READ                                  = 0x88E1
	STATIC_DRAW                                  = 0x88E4
	STATIC_READ                                  = 0x88E5
	DYNAMIC_DRAW                                 = 0x88E8
	DYNAMIC_READ                                 = 0x88E9
	BUFFER_SIZE                                  = 0x8764
	BUFFER_USAGE                                 = 0x8765
	CURRENT_VERTEX_ATTRIB                        = 0x8626
	FRONT                                        = 0x0404
	BACK                                         = 0x0405
	FRONT_AND_BACK                               = 0x0408
	TEXTURE_NONE                                 = 0x0000
	TEXTURE_2D                                   = 0x0DE1
	CULL_FACE                                    = 0x0B44
	BLEND                                        = 0x0BE2
	DITHER                                       = 0x0BD0
	STENCIL_TEST                                 = 0x0B90
	DEPTH_TEST                                   = 0x0B71
	SCISSOR_TEST                                 = 0x0C11
	POLYGON_OFFSET_FILL                          = 0x8037
	SAMPLE_ALPHA_TO_COVERAGE                     = 0x809E
	SAMPLE_COVERAGE                              = 0x80A0
	POLYGON_SMOOTH                               = 0x0B41
	POLYGON_STIPPLE                              = 0x0B42
	COLOR_LOGIC_OP                               = 0x0BF2
	COLOR_TABLE                                  = 0x80D0
	INDEX_LOGIC_OP                               = 0x0BF1
	VERTEX_PROGRAM_POINT_SIZE                    = 0x8642
	LINE_STIPPLE                                 = 0x0B24
	FRAMEBUFFER_SRGB                             = 0x8DB9
	SHADER_PIXEL_LOCAL_STORAGE                   = 0x8F64
	SAMPLE_SHADING                               = 0x8C36
	NO_ERROR                                     = 0
	INVALID_ENUM                                 = 0x0500
	INVALID_VALUE                                = 0x0501
	INVALID_OPERATION                            = 0x0502
	OUT_OF_MEMORY                                = 0x0505
	CONTEXT_LOST                                 = 0x300E
	CW                                           = 0x0900
	CCW                                          = 0x0901
	LINE_WIDTH                                   = 0x0B21
	ALIASED_POINT_SIZE_RANGE                     = 0x846D
	ALIASED_LINE_WIDTH_RANGE                     = 0x846E
	CULL_FACE_MODE                               = 0x0B45
	FRONT_FACE                                   = 0x0B46
	DEPTH_RANGE                                  = 0x0B70
	DEPTH_WRITEMASK                              = 0x0B72
	DEPTH_CLEAR_VALUE                            = 0x0B73
	DEPTH_FUNC                                   = 0x0B74
	STENCIL_CLEAR_VALUE                          = 0x0B91
	STENCIL_FUNC                                 = 0x0B92
	STENCIL_FAIL                                 = 0x0B94
	STENCIL_PASS_DEPTH_FAIL                      = 0x0B95
	STENCIL_PASS_DEPTH_PASS                      = 0x0B96
	STENCIL_REF                                  = 0x0B97
	STENCIL_VALUE_MASK                           = 0x0B93
	STENCIL_WRITEMASK                            = 0x0B98
	STENCIL_BACK_FUNC                            = 0x8800
	STENCIL_BACK_FAIL                            = 0x8801
	STENCIL_BACK_PASS_DEPTH_FAIL                 = 0x8802
	STENCIL_BACK_PASS_DEPTH_PASS                 = 0x8803
	STENCIL_BACK_REF                             = 0x8CA3
	STENCIL_BACK_VALUE_MASK                      = 0x8CA4
	STENCIL_BACK_WRITEMASK                       = 0x8CA5
	VIEWPORT                                     = 0x0BA2
	SCISSOR_BOX                                  = 0x0C10
	COLOR_CLEAR_VALUE                            = 0x0C22
	COLOR_WRITEMASK                              = 0x0C23
	UNPACK_ALIGNMENT                             = 0x0CF5
	PACK_ALIGNMENT                               = 0x0D05
	PACK_REVERSE_ROW_ORDER                       = 0x93A4
	MAX_TEXTURE_SIZE                             = 0x0D33
	TEXTURE_MIN_LOD                              = 0x813A
	TEXTURE_MAX_LOD                              = 0x813B
	TEXTURE_BASE_LEVEL                           = 0x813C
	TEXTURE_MAX_LEVEL                            = 0x813D
	MAX_VIEWPORT_DIMS                            = 0x0D3A
	SUBPIXEL_BITS                                = 0x0D50
	RED_BITS                                     = 0x0D52
	GREEN_BITS                                   = 0x0D53
	BLUE_BITS                                    = 0x0D54
	ALPHA_BITS                                   = 0x0D55
	DEPTH_BITS                                   = 0x0D56
	STENCIL_BITS                                 = 0x0D57
	POLYGON_OFFSET_UNITS                         = 0x2A00
	POLYGON_OFFSET_FACTOR                        = 0x8038
	TEXTURE_BINDING_2D                           = 0x8069
	SAMPLE_BUFFERS                               = 0x80A8
	SAMPLES                                      = 0x80A9
	SAMPLE_COVERAGE_VALUE                        = 0x80AA
	SAMPLE_COVERAGE_INVERT                       = 0x80AB
	RENDERBUFFER_COVERAGE_SAMPLES                = 0x8CAB
	RENDERBUFFER_COLOR_SAMPLES                   = 0x8E10
	MAX_MULTISAMPLE_COVERAGE_MODES               = 0x8E11
	MULTISAMPLE_COVERAGE_MODES                   = 0x8E12
	MAX_TEXTURE_BUFFER_SIZE                      = 0x8C2B
	CONTEXT_FLAGS                                = 0x821E
	CONTEXT_FLAG_PROTECTED_CONTENT_BIT_EXT       = 0x00000010
	TEXTURE_PROTECTED_EXT                        = 0x8BFA
	NUM_COMPRESSED_TEXTURE_FORMATS               = 0x86A2
	COMPRESSED_TEXTURE_FORMATS                   = 0x86A3
	COMPRESSED_RGB_S3TC_DXT1_EXT                 = 0x83F0
	COMPRESSED_RGBA_S3TC_DXT1_EXT                = 0x83F1
	COMPRESSED_RGBA_S3TC_DXT3_EXT                = 0x83F2
	COMPRESSED_RGBA_S3TC_DXT5_EXT                = 0x83F3
	COMPRESSED_SRGB_S3TC_DXT1_EXT                = 0x8C4C
	COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT          = 0x8C4D
	COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT          = 0x8C4E
	COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT          = 0x8C4F
	COMPRESSED_RGB_PVRTC_4BPPV1_IMG              = 0x8C00
	COMPRESSED_RGB_PVRTC_2BPPV1_IMG              = 0x8C01
	COMPRESSED_RGBA_PVRTC_4BPPV1_IMG             = 0x8C02
	COMPRESSED_RGBA_PVRTC_2BPPV1_IMG             = 0x8C03
	COMPRESSED_RGBA_PVRTC_2BPPV2_IMG             = 0x9137
	COMPRESSED_RGBA_PVRTC_4BPPV2_IMG             = 0x9138
	COMPRESSED_ETC1_RGB8                         = 0x8D64
	COMPRESSED_R11_EAC                           = 0x9270
	COMPRESSED_SIGNED_R11_EAC                    = 0x9271
	COMPRESSED_RG11_EAC                          = 0x9272
	COMPRESSED_SIGNED_RG11_EAC                   = 0x9273
	COMPRESSED_RGB8_ETC2                         = 0x9274
	COMPRESSED_SRGB8_ETC2                        = 0x9275
	COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1          = 0x9276
	COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1         = 0x9277
	COMPRESSED_RGBA8_ETC2_EAC                    = 0x9278
	COMPRESSED_SRGB8_ALPHA8_ETC2_EAC             = 0x9279
	COMPRESSED_LUMINANCE_LATC1                   = 0x8C70
	COMPRESSED_SIGNED_LUMINANCE_LATC1            = 0x8C71
	COMPRESSED_LUMINANCE_ALPHA_LATC2             = 0x8C72
	COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2      = 0x8C73
	COMPRESSED_RED_RGTC1                         = 0x8DBB
	COMPRESSED_SIGNED_RED_RGTC1                  = 0x8DBC
	COMPRESSED_RED_GREEN_RGTC2                   = 0x8DBD
	COMPRESSED_SIGNED_RED_GREEN_RGTC2            = 0x8DBE
	COMPRESSED_3DC_X                             = 0x87F9
	COMPRESSED_3DC_XY                            = 0x87FA
	COMPRESSED_RGBA_BPTC_UNORM                   = 0x8E8C
	COMPRESSED_SRGB_ALPHA_BPTC_UNORM             = 0x8E8D
	COMPRESSED_RGB_BPTC_SIGNED_FLOAT             = 0x8E8E
	COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT           = 0x8E8F
	COMPRESSED_RGBA_ASTC_4x4                     = 0x93B0
	COMPRESSED_RGBA_ASTC_5x4                     = 0x93B1
	COMPRESSED_RGBA_ASTC_5x5                     = 0x93B2
	COMPRESSED_RGBA_ASTC_6x5                     = 0x93B3
	COMPRESSED_RGBA_ASTC_6x6                     = 0x93B4
	COMPRESSED_RGBA_ASTC_8x5                     = 0x93B5
	COMPRESSED_RGBA_ASTC_8x6                     = 0x93B6
	COMPRESSED_RGBA_ASTC_8x8                     = 0x93B7
	COMPRESSED_RGBA_ASTC_10x5                    = 0x93B8
	COMPRESSED_RGBA_ASTC_10x6                    = 0x93B9
	COMPRESSED_RGBA_ASTC_10x8                    = 0x93BA
	COMPRESSED_RGBA_ASTC_10x10                   = 0x93BB
	COMPRESSED_RGBA_ASTC_12x10                   = 0x93BC
	COMPRESSED_RGBA_ASTC_12x12                   = 0x93BD
	COMPRESSED_SRGB8_ALPHA8_ASTC_4x4             = 0x93D0
	COMPRESSED_SRGB8_ALPHA8_ASTC_5x4             = 0x93D1
	COMPRESSED_SRGB8_ALPHA8_ASTC_5x5             = 0x93D2
	COMPRESSED_SRGB8_ALPHA8_ASTC_6x5             = 0x93D3
	COMPRESSED_SRGB8_ALPHA8_ASTC_6x6             = 0x93D4
	COMPRESSED_SRGB8_ALPHA8_ASTC_8x5             = 0x93D5
	COMPRESSED_SRGB8_ALPHA8_ASTC_8x6             = 0x93D6
	COMPRESSED_SRGB8_ALPHA8_ASTC_8x8             = 0x93D7
	COMPRESSED_SRGB8_ALPHA8_ASTC_10x5            = 0x93D8
	COMPRESSED_SRGB8_ALPHA8_ASTC_10x6            = 0x93D9
	COMPRESSED_SRGB8_ALPHA8_ASTC_10x8            = 0x93DA
	COMPRESSED_SRGB8_ALPHA8_ASTC_10x10           = 0x93DB
	COMPRESSED_SRGB8_ALPHA8_ASTC_12x10           = 0x93DC
	COMPRESSED_SRGB8_ALPHA8_ASTC_12x12           = 0x93DD
	DONT_CARE                                    = 0x1100
	FASTEST                                      = 0x1101
	NICEST                                       = 0x1102
	GENERATE_MIPMAP_HINT                         = 0x8192
	BYTE                                         = 0x1400
	UNSIGNED_BYTE                                = 0x1401
	SHORT                                        = 0x1402
	UNSIGNED_SHORT                               = 0x1403
	INT                                          = 0x1404
	UNSIGNED_INT                                 = 0x1405
	FLOAT                                        = 0x1406
	HALF_FLOAT                                   = 0x140B
	FIXED                                        = 0x140C
	HALF_FLOAT_OES                               = 0x8D61
	LIGHTING                                     = 0x0B50
	LIGHT0                                       = 0x4000
	LIGHT1                                       = 0x4001
	LIGHT2                                       = 0x4002
	LIGHT3                                       = 0x4003
	LIGHT4                                       = 0x4004
	LIGHT5                                       = 0x4005
	LIGHT6                                       = 0x4006
	LIGHT7                                       = 0x4007
	SPOT_EXPONENT                                = 0x1205
	SPOT_CUTOFF                                  = 0x1206
	CONSTANT_ATTENUATION                         = 0x1207
	LINEAR_ATTENUATION                           = 0x1208
	QUADRATIC_ATTENUATION                        = 0x1209
	AMBIENT                                      = 0x1200
	DIFFUSE                                      = 0x1201
	SPECULAR                                     = 0x1202
	SHININESS                                    = 0x1601
	EMISSION                                     = 0x1600
	POSITION                                     = 0x1203
	SPOT_DIRECTION                               = 0x1204
	AMBIENT_AND_DIFFUSE                          = 0x1602
	COLOR_INDEXES                                = 0x1603
	LIGHT_MODEL_TWO_SIDE                         = 0x0B52
	LIGHT_MODEL_LOCAL_VIEWER                     = 0x0B51
	LIGHT_MODEL_AMBIENT                          = 0x0B53
	SHADE_MODEL                                  = 0x0B54
	FLAT                                         = 0x1D00
	SMOOTH                                       = 0x1D01
	COLOR_MATERIAL                               = 0x0B57
	COLOR_MATERIAL_FACE                          = 0x0B55
	COLOR_MATERIAL_PARAMETER                     = 0x0B56
	NORMALIZE                                    = 0x0BA1
	MATRIX_MODE                                  = 0x0BA0
	MODELVIEW                                    = 0x1700
	PROJECTION                                   = 0x1701
	TEXTURE                                      = 0x1702
	MULTISAMPLE                                  = 0x809D
	SAMPLE_POSITION                              = 0x8E50
	POINT_SMOOTH                                 = 0x0B10
	POINT_SIZE                                   = 0x0B11
	POINT_SIZE_GRANULARITY                       = 0x0B13
	POINT_SIZE_RANGE                             = 0x0B12
	LINE_SMOOTH                                  = 0x0B20
	LINE_STIPPLE_PATTERN                         = 0x0B25
	LINE_STIPPLE_REPEAT                          = 0x0B26
	LINE_WIDTH_GRANULARITY                       = 0x0B23
	LINE_WIDTH_RANGE                             = 0x0B22
	POINT                                        = 0x1B00
	LINE                                         = 0x1B01
	FILL                                         = 0x1B02
	STENCIL_INDEX                                = 0x1901
	DEPTH_COMPONENT                              = 0x1902
	DEPTH_STENCIL                                = 0x84F9
	RED                                          = 0x1903
	RED_INTEGER                                  = 0x8D94
	GREEN                                        = 0x1904
	BLUE                                         = 0x1905
	ALPHA                                        = 0x1906
	LUMINANCE                                    = 0x1909
	LUMINANCE_ALPHA                              = 0x190A
	RG_INTEGER                                   = 0x8228
	RGB                                          = 0x1907
	RGB_INTEGER                                  = 0x8D98
	SRGB                                         = 0x8C40
	RGBA                                         = 0x1908
	RG                                           = 0x8227
	SRGB_ALPHA                                   = 0x8C42
	RGBA_INTEGER                                 = 0x8D99
	BGRA                                         = 0x80E1
	STENCIL_INDEX4                               = 0x8D47
	STENCIL_INDEX8                               = 0x8D48
	STENCIL_INDEX16                              = 0x8D49
	DEPTH_COMPONENT16                            = 0x81A5
	DEPTH24_STENCIL8                             = 0x88F0
	R8                                           = 0x8229
	R16                                          = 0x822A
	R16F                                         = 0x822D
	R32F                                         = 0x822E
	R8I                                          = 0x8231
	R8UI                                         = 0x8232
	R16I                                         = 0x8233
	R16UI                                        = 0x8234
	R32I                                         = 0x8235
	R32UI                                        = 0x8236
	LUMINANCE8                                   = 0x8040
	LUMINANCE8_ALPHA8                            = 0x8045
	LUMINANCE16F                                 = 0x881E
	ALPHA8                                       = 0x803C
	ALPHA16                                      = 0x803E
	ALPHA16F                                     = 0x881C
	ALPHA32F                                     = 0x8816
	ALPHA8I                                      = 0x8D90
	ALPHA8UI                                     = 0x8D7E
	ALPHA16I                                     = 0x8D8A
	ALPHA16UI                                    = 0x8D78
	ALPHA32I                                     = 0x8D84
	ALPHA32UI                                    = 0x8D72
	RG8                                          = 0x822B
	RG16                                         = 0x822C
	RG16F                                        = 0x822F
	RG8I                                         = 0x8237
	RG8UI                                        = 0x8238
	RG16I                                        = 0x8239
	RG16UI                                       = 0x823A
	RG32I                                        = 0x823B
	RG32UI                                       = 0x823C
	RGB5                                         = 0x8050
	RGB565                                       = 0x8D62
	RGB8                                         = 0x8051
	SRGB8                                        = 0x8C41
	RGBX8                                        = 0x96BA
	RGB8I                                        = 0x8D8F
	RGB8UI                                       = 0x8D7D
	RGB16I                                       = 0x8D89
	RGB16UI                                      = 0x8D77
	RGB32I                                       = 0x8D83
	RGB32UI                                      = 0x8D71
	RGBA4                                        = 0x8056
	RGB5_A1                                      = 0x8057
	RGBA8                                        = 0x8058
	RGB10_A2                                     = 0x8059
	SRGB8_ALPHA8                                 = 0x8C43
	RGBA16F                                      = 0x881A
	RGBA32F                                      = 0x8814
	RG32F                                        = 0x8230
	RGBA16                                       = 0x805B
	RGBA8I                                       = 0x8D8E
	RGBA8UI                                      = 0x8D7C
	RGBA16I                                      = 0x8D88
	RGBA16UI                                     = 0x8D76
	RGBA32I                                      = 0x8D82
	RGBA32UI                                     = 0x8D70
	BGRA8                                        = 0x93A1
	UNSIGNED_SHORT_4_4_4_4                       = 0x8033
	UNSIGNED_SHORT_5_5_5_1                       = 0x8034
	UNSIGNED_SHORT_5_6_5                         = 0x8363
	UNSIGNED_INT_2_10_10_10_REV                  = 0x8368
	FRAGMENT_SHADER                              = 0x8B30
	VERTEX_SHADER                                = 0x8B31
	TESS_CONTROL_SHADER                          = 0x8E88
	TESS_EVALUATION_SHADER                       = 0x8E87
	MAX_VERTEX_ATTRIBS                           = 0x8869
	MAX_VERTEX_UNIFORM_VECTORS                   = 0x8DFB
	MAX_VARYING_VECTORS                          = 0x8DFC
	MAX_COMBINED_TEXTURE_IMAGE_UNITS             = 0x8B4D
	MAX_VERTEX_TEXTURE_IMAGE_UNITS               = 0x8B4C
	MAX_GEOMETRY_TEXTURE_IMAGE_UNITS             = 0x8C29
	MAX_TEXTURE_IMAGE_UNITS                      = 0x8872
	MAX_FRAGMENT_UNIFORM_VECTORS                 = 0x8DFD
	SHADER_TYPE                                  = 0x8B4F
	DELETE_STATUS                                = 0x8B80
	LINK_STATUS                                  = 0x8B82
	VALIDATE_STATUS                              = 0x8B83
	ATTACHED_SHADERS                             = 0x8B85
	ACTIVE_UNIFORMS                              = 0x8B86
	ACTIVE_UNIFORM_MAX_LENGTH                    = 0x8B87
	ACTIVE_ATTRIBUTES                            = 0x8B89
	ACTIVE_ATTRIBUTE_MAX_LENGTH                  = 0x8B8A
	SHADING_LANGUAGE_VERSION                     = 0x8B8C
	CURRENT_PROGRAM                              = 0x8B8D
	MAX_FRAGMENT_UNIFORM_COMPONENTS              = 0x8B49
	MAX_VERTEX_UNIFORM_COMPONENTS                = 0x8B4A
	MAX_SHADER_PIXEL_LOCAL_STORAGE_FAST_SIZE     = 0x8F63
	SHADER_BINARY_FORMATS                        = 0x8DF8
	NEVER                                        = 0x0200
	LESS                                         = 0x0201
	EQUAL                                        = 0x0202
	LEQUAL                                       = 0x0203
	GREATER                                      = 0x0204
	NOTEQUAL                                     = 0x0205
	GEQUAL                                       = 0x0206
	ALWAYS                                       = 0x0207
	KEEP                                         = 0x1E00
	REPLACE                                      = 0x1E01
	INCR                                         = 0x1E02
	DECR                                         = 0x1E03
	INVERT                                       = 0x150A
	INCR_WRAP                                    = 0x8507
	DECR_WRAP                                    = 0x8508
	VENDOR                                       = 0x1F00
	RENDERER                                     = 0x1F01
	VERSION                                      = 0x1F02
	EXTENSIONS                                   = 0x1F03
	NUM_EXTENSIONS                               = 0x821D
	UNPACK_ROW_LENGTH                            = 0x0CF2
	PACK_ROW_LENGTH                              = 0x0D02
	NEAREST                                      = 0x2600
	LINEAR                                       = 0x2601
	NEAREST_MIPMAP_NEAREST                       = 0x2700
	LINEAR_MIPMAP_NEAREST                        = 0x2701
	NEAREST_MIPMAP_LINEAR                        = 0x2702
	LINEAR_MIPMAP_LINEAR                         = 0x2703
	FRAMEBUFFER_ATTACHMENT                       = 0x93A3
	TEXTURE_MAG_FILTER                           = 0x2800
	TEXTURE_MIN_FILTER                           = 0x2801
	TEXTURE_WRAP_S                               = 0x2802
	TEXTURE_WRAP_T                               = 0x2803
	TEXTURE_USAGE                                = 0x93A2
	TEXTURE_MAX_ANISOTROPY                       = 0x84FE
	MAX_TEXTURE_MAX_ANISOTROPY                   = 0x84FF
	TEXTURE_CUBE_MAP                             = 0x8513
	TEXTURE_BINDING_CUBE_MAP                     = 0x8514
	TEXTURE_CUBE_MAP_POSITIVE_X                  = 0x8515
	TEXTURE_CUBE_MAP_NEGATIVE_X                  = 0x8516
	TEXTURE_CUBE_MAP_POSITIVE_Y                  = 0x8517
	TEXTURE_CUBE_MAP_NEGATIVE_Y                  = 0x8518
	TEXTURE_CUBE_MAP_POSITIVE_Z                  = 0x8519
	TEXTURE_CUBE_MAP_NEGATIVE_Z                  = 0x851A
	MAX_CUBE_MAP_TEXTURE_SIZE                    = 0x851C
	TEXTURE0                                     = 0x84C0
	TEXTURE1                                     = 0x84C1
	TEXTURE2                                     = 0x84C2
	TEXTURE3                                     = 0x84C3
	TEXTURE4                                     = 0x84C4
	TEXTURE5                                     = 0x84C5
	TEXTURE6                                     = 0x84C6
	TEXTURE7                                     = 0x84C7
	TEXTURE8                                     = 0x84C8
	TEXTURE9                                     = 0x84C9
	TEXTURE10                                    = 0x84CA
	TEXTURE11                                    = 0x84CB
	TEXTURE12                                    = 0x84CC
	TEXTURE13                                    = 0x84CD
	TEXTURE14                                    = 0x84CE
	TEXTURE15                                    = 0x84CF
	TEXTURE16                                    = 0x84D0
	TEXTURE17                                    = 0x84D1
	TEXTURE18                                    = 0x84D2
	TEXTURE19                                    = 0x84D3
	TEXTURE20                                    = 0x84D4
	TEXTURE21                                    = 0x84D5
	TEXTURE22                                    = 0x84D6
	TEXTURE23                                    = 0x84D7
	TEXTURE24                                    = 0x84D8
	TEXTURE25                                    = 0x84D9
	TEXTURE26                                    = 0x84DA
	TEXTURE27                                    = 0x84DB
	TEXTURE28                                    = 0x84DC
	TEXTURE29                                    = 0x84DD
	TEXTURE30                                    = 0x84DE
	TEXTURE31                                    = 0x84DF
	ACTIVE_TEXTURE                               = 0x84E0
	MAX_TEXTURE_UNITS                            = 0x84E2
	MAX_TEXTURE_COORDS                           = 0x8871
	REPEAT                                       = 0x2901
	CLAMP_TO_EDGE                                = 0x812F
	MIRRORED_REPEAT                              = 0x8370
	CLAMP_TO_BORDER                              = 0x812D
	TEXTURE_SWIZZLE_R                            = 0x8E42
	TEXTURE_SWIZZLE_G                            = 0x8E43
	TEXTURE_SWIZZLE_B                            = 0x8E44
	TEXTURE_SWIZZLE_A                            = 0x8E45
	TEXTURE_SWIZZLE_RGBA                         = 0x8E46
	TEXTURE_ENV                                  = 0x2300
	TEXTURE_ENV_MODE                             = 0x2200
	TEXTURE_1D                                   = 0x0DE0
	TEXTURE_ENV_COLOR                            = 0x2201
	TEXTURE_GEN_S                                = 0x0C60
	TEXTURE_GEN_T                                = 0x0C61
	TEXTURE_GEN_R                                = 0x0C62
	TEXTURE_GEN_Q                                = 0x0C63
	TEXTURE_GEN_MODE                             = 0x2500
	TEXTURE_BORDER_COLOR                         = 0x1004
	TEXTURE_WIDTH                                = 0x1000
	TEXTURE_HEIGHT                               = 0x1001
	TEXTURE_BORDER                               = 0x1005
	TEXTURE_COMPONENTS                           = 0x1003
	TEXTURE_RED_SIZE                             = 0x805C
	TEXTURE_GREEN_SIZE                           = 0x805D
	TEXTURE_BLUE_SIZE                            = 0x805E
	TEXTURE_ALPHA_SIZE                           = 0x805F
	TEXTURE_LUMINANCE_SIZE                       = 0x8060
	TEXTURE_INTENSITY_SIZE                       = 0x8061
	TEXTURE_INTERNAL_FORMAT                      = 0x1003
	OBJECT_LINEAR                                = 0x2401
	OBJECT_PLANE                                 = 0x2501
	EYE_LINEAR                                   = 0x2400
	EYE_PLANE                                    = 0x2502
	SPHERE_MAP                                   = 0x2402
	DECAL                                        = 0x2101
	MODULATE                                     = 0x2100
	CLAMP                                        = 0x2900
	S                                            = 0x2000
	T                                            = 0x2001
	R                                            = 0x2002
	Q                                            = 0x2003
	COMBINE                                      = 0x8570
	COMBINE_RGB                                  = 0x8571
	COMBINE_ALPHA                                = 0x8572
	SOURCE0_RGB                                  = 0x8580
	SOURCE1_RGB                                  = 0x8581
	SOURCE2_RGB                                  = 0x8582
	SOURCE0_ALPHA                                = 0x8588
	SOURCE1_ALPHA                                = 0x8589
	SOURCE2_ALPHA                                = 0x858A
	OPERAND0_RGB                                 = 0x8590
	OPERAND1_RGB                                 = 0x8591
	OPERAND2_RGB                                 = 0x8592
	OPERAND0_ALPHA                               = 0x8598
	OPERAND1_ALPHA                               = 0x8599
	OPERAND2_ALPHA                               = 0x859A
	RGB_SCALE                                    = 0x8573
	ADD_SIGNED                                   = 0x8574
	INTERPOLATE                                  = 0x8575
	SUBTRACT                                     = 0x84E7
	CONSTANT                                     = 0x8576
	PRIMARY_COLOR                                = 0x8577
	PREVIOUS                                     = 0x8578
	SRC0_RGB                                     = 0x8580
	SRC1_RGB                                     = 0x8581
	SRC2_RGB                                     = 0x8582
	SRC0_ALPHA                                   = 0x8588
	SRC1_ALPHA                                   = 0x8589
	SRC2_ALPHA                                   = 0x858A
	FLOAT_VEC2                                   = 0x8B50
	FLOAT_VEC3                                   = 0x8B51
	FLOAT_VEC4                                   = 0x8B52
	INT_VEC2                                     = 0x8B53
	INT_VEC3                                     = 0x8B54
	INT_VEC4                                     = 0x8B55
	BOOL                                         = 0x8B56
	BOOL_VEC2                                    = 0x8B57
	BOOL_VEC3                                    = 0x8B58
	BOOL_VEC4                                    = 0x8B59
	FLOAT_MAT2                                   = 0x8B5A
	FLOAT_MAT3                                   = 0x8B5B
	FLOAT_MAT4                                   = 0x8B5C
	SAMPLER_2D                                   = 0x8B5E
	SAMPLER_CUBE                                 = 0x8B60
	VERTEX_ATTRIB_ARRAY_ENABLED                  = 0x8622
	VERTEX_ATTRIB_ARRAY_SIZE                     = 0x8623
	VERTEX_ATTRIB_ARRAY_STRIDE                   = 0x8624
	VERTEX_ATTRIB_ARRAY_TYPE                     = 0x8625
	VERTEX_ATTRIB_ARRAY_NORMALIZED               = 0x886A
	VERTEX_ATTRIB_ARRAY_POINTER                  = 0x8645
	VERTEX_ATTRIB_ARRAY_BUFFER_BINDING           = 0x889F
	VERTEX_ARRAY                                 = 0x8074
	NORMAL_ARRAY                                 = 0x8075
	COLOR_ARRAY                                  = 0x8076
	SECONDARY_COLOR_ARRAY                        = 0x845E
	INDEX_ARRAY                                  = 0x8077
	TEXTURE_COORD_ARRAY                          = 0x8078
	EDGE_FLAG_ARRAY                              = 0x8079
	VERTEX_ARRAY_SIZE                            = 0x807A
	VERTEX_ARRAY_TYPE                            = 0x807B
	VERTEX_ARRAY_STRIDE                          = 0x807C
	NORMAL_ARRAY_TYPE                            = 0x807E
	NORMAL_ARRAY_STRIDE                          = 0x807F
	COLOR_ARRAY_SIZE                             = 0x8081
	COLOR_ARRAY_TYPE                             = 0x8082
	COLOR_ARRAY_STRIDE                           = 0x8083
	INDEX_ARRAY_TYPE                             = 0x8085
	INDEX_ARRAY_STRIDE                           = 0x8086
	TEXTURE_COORD_ARRAY_SIZE                     = 0x8088
	TEXTURE_COORD_ARRAY_TYPE                     = 0x8089
	TEXTURE_COORD_ARRAY_STRIDE                   = 0x808A
	EDGE_FLAG_ARRAY_STRIDE                       = 0x808C
	VERTEX_ARRAY_POINTER                         = 0x808E
	NORMAL_ARRAY_POINTER                         = 0x808F
	COLOR_ARRAY_POINTER                          = 0x8090
	INDEX_ARRAY_POINTER                          = 0x8091
	TEXTURE_COORD_ARRAY_POINTER                  = 0x8092
	EDGE_FLAG_ARRAY_POINTER                      = 0x8093
	V2F                                          = 0x2A20
	V3F                                          = 0x2A21
	C4UB_V2F                                     = 0x2A22
	C4UB_V3F                                     = 0x2A23
	C3F_V3F                                      = 0x2A24
	N3F_V3F                                      = 0x2A25
	C4F_N3F_V3F                                  = 0x2A26
	T2F_V3F                                      = 0x2A27
	T4F_V4F                                      = 0x2A28
	T2F_C4UB_V3F                                 = 0x2A29
	T2F_C3F_V3F                                  = 0x2A2A
	T2F_N3F_V3F                                  = 0x2A2B
	T2F_C4F_N3F_V3F                              = 0x2A2C
	T4F_C4F_N3F_V4F                              = 0x2A2D
	PRIMITIVE_RESTART_FIXED_INDEX                = 0x8D69
	READ_ONLY                                    = 0x88B8
	WRITE_ONLY                                   = 0x88B9
	READ_WRITE                                   = 0x88BA
	BUFFER_MAPPED                                = 0x88BC
	MAP_READ_BIT                                 = 0x0001
	MAP_WRITE_BIT                                = 0x0002
	MAP_INVALIDATE_RANGE_BIT                     = 0x0004
	MAP_INVALIDATE_BUFFER_BIT                    = 0x0008
	MAP_FLUSH_EXPLICIT_BIT                       = 0x0010
	MAP_UNSYNCHRONIZED_BIT                       = 0x0020
	IMPLEMENTATION_COLOR_READ_TYPE               = 0x8B9A
	IMPLEMENTATION_COLOR_READ_FORMAT             = 0x8B9B
	COMPILE_STATUS                               = 0x8B81
	INFO_LOG_LENGTH                              = 0x8B84
	SHADER_SOURCE_LENGTH                         = 0x8B88
	SHADER_COMPILER                              = 0x8DFA
	NUM_SHADER_BINARY_FORMATS                    = 0x8DF9
	NUM_PROGRAM_BINARY_FORMATS                   = 0x87FE
	PROGRAM_BINARY_FORMATS                       = 0x87FF
	LOW_FLOAT                                    = 0x8DF0
	MEDIUM_FLOAT                                 = 0x8DF1
	HIGH_FLOAT                                   = 0x8DF2
	LOW_INT                                      = 0x8DF3
	MEDIUM_INT                                   = 0x8DF4
	HIGH_INT                                     = 0x8DF5
	QUERY_COUNTER_BITS                           = 0x8864
	CURRENT_QUERY                                = 0x8865
	QUERY_RESULT                                 = 0x8866
	QUERY_RESULT_AVAILABLE                       = 0x8867
	SAMPLES_PASSED                               = 0x8914
	ANY_SAMPLES_PASSED                           = 0x8C2F
	TIME_ELAPSED                                 = 0x88BF
	TIMESTAMP                                    = 0x8E28
	GPU_DISJOINT                                 = 0x8FBB
	PRIMITIVES_GENERATED                         = 0x8C87
	TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN        = 0x8C88
	FRAMEBUFFER                                  = 0x8D40
	READ_FRAMEBUFFER                             = 0x8CA8
	DRAW_FRAMEBUFFER                             = 0x8CA9
	RENDERBUFFER                                 = 0x8D41
	MAX_SAMPLES                                  = 0x8D57
	MAX_SAMPLES_IMG                              = 0x9135
	RENDERBUFFER_WIDTH                           = 0x8D42
	RENDERBUFFER_HEIGHT                          = 0x8D43
	RENDERBUFFER_INTERNAL_FORMAT                 = 0x8D44
	RENDERBUFFER_RED_SIZE                        = 0x8D50
	RENDERBUFFER_GREEN_SIZE                      = 0x8D51
	RENDERBUFFER_BLUE_SIZE                       = 0x8D52
	RENDERBUFFER_ALPHA_SIZE                      = 0x8D53
	RENDERBUFFER_DEPTH_SIZE                      = 0x8D54
	RENDERBUFFER_STENCIL_SIZE                    = 0x8D55
	FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE           = 0x8CD0
	FRAMEBUFFER_ATTACHMENT_OBJECT_NAME           = 0x8CD1
	FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL         = 0x8CD2
	FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE = 0x8CD3
	FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER         = 0x8CD4
	FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING        = 0x8210
	FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE        = 0x8211
	FRAMEBUFFER_ATTACHMENT_RED_SIZE              = 0x8212
	FRAMEBUFFER_ATTACHMENT_GREEN_SIZE            = 0x8213
	FRAMEBUFFER_ATTACHMENT_BLUE_SIZE             = 0x8214
	FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE            = 0x8215
	FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE            = 0x8216
	FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE          = 0x8217
	DEPTH_STENCIL_ATTACHMENT                     = 0x821A
	COLOR_ATTACHMENT0                            = 0x8CE0
	DEPTH_ATTACHMENT                             = 0x8D00
	STENCIL_ATTACHMENT                           = 0x8D20
	COLOR                                        = 0x1800
	DEPTH                                        = 0x1801
	STENCIL                                      = 0x1802
	NONE                                         = 0
	FRAMEBUFFER_DEFAULT                          = 0x8218
	FRAMEBUFFER_COMPLETE                         = 0x8CD5
	FRAMEBUFFER_INCOMPLETE_ATTACHMENT            = 0x8CD6
	FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT    = 0x8CD7
	FRAMEBUFFER_INCOMPLETE_DIMENSIONS            = 0x8CD9
	FRAMEBUFFER_UNSUPPORTED                      = 0x8CDD
	FRAMEBUFFER_BINDING                          = 0x8CA6
	RENDERBUFFER_BINDING                         = 0x8CA7
	MAX_RENDERBUFFER_SIZE                        = 0x84E8
	INVALID_FRAMEBUFFER_OPERATION                = 0x0506
	CLOSE_PATH                                   = 0x00
	MOVE_TO                                      = 0x02
	LINE_TO                                      = 0x04
	QUADRATIC_CURVE_TO                           = 0x0A
	CUBIC_CURVE_TO                               = 0x0C
	CONIC_CURVE_TO                               = 0x1A
	PATH_STROKE_WIDTH                            = 0x9075
	PATH_END_CAPS                                = 0x9076
	PATH_JOIN_STYLE                              = 0x9079
	PATH_MITER_LIMIT                             = 0x907A
	PATH_STROKE_BOUND                            = 0x9086
	COUNT_UP                                     = 0x9088
	BOUNDING_BOX                                 = 0x908D
	BOUNDING_BOX_OF_BOUNDING_BOXES               = 0x909C
	TRANSLATE_X                                  = 0x908E
	TRANSLATE_Y                                  = 0x908F
	TRANSLATE_2D                                 = 0x9090
	TRANSPOSE_AFFINE_2D                          = 0x9096
	SQUARE                                       = 0x90A3
	ROUND                                        = 0x90A4
	BEVEL                                        = 0x90A6
	MITER_REVERT                                 = 0x90A7
	STANDARD_FONT_FORMAT                         = 0x936C
	FONT_GLYPHS_AVAILABLE                        = 0x9368
	FRAGMENT_INPUT                               = 0x936D
	PATH_PROJECTION                              = 0x1701
	PATH_MODELVIEW                               = 0x1700
	FETCH_PER_SAMPLE                             = 0x8F65
	DEBUG_OUTPUT                                 = 0x92E0
	DEBUG_OUTPUT_SYNCHRONOUS                     = 0x8242
	CONTEXT_FLAG_DEBUG_BIT                       = 0x00000002
	MAX_DEBUG_MESSAGE_LENGTH                     = 0x9143
	MAX_DEBUG_LOGGED_MESSAGES                    = 0x9144
	DEBUG_LOGGED_MESSAGES                        = 0x9145
	DEBUG_NEXT_LOGGED_MESSAGE_LENGTH             = 0x8243
	MAX_DEBUG_GROUP_STACK_DEPTH                  = 0x826C
	DEBUG_GROUP_STACK_DEPTH                      = 0x826D
	MAX_LABEL_LENGTH                             = 0x82E8
	DEBUG_SOURCE_API                             = 0x8246
	DEBUG_SOURCE_WINDOW_SYSTEM                   = 0x8247
	DEBUG_SOURCE_SHADER_COMPILER                 = 0x8248
	DEBUG_SOURCE_THIRD_PARTY                     = 0x8249
	DEBUG_SOURCE_APPLICATION                     = 0x824A
	DEBUG_SOURCE_OTHER                           = 0x824B
	DEBUG_TYPE_ERROR                             = 0x824C
	DEBUG_TYPE_DEPRECATED_BEHAVIOR               = 0x824D
	DEBUG_TYPE_UNDEFINED_BEHAVIOR                = 0x824E
	DEBUG_TYPE_PORTABILITY                       = 0x824F
	DEBUG_TYPE_PERFORMANCE                       = 0x8250
	DEBUG_TYPE_OTHER                             = 0x8251
	DEBUG_TYPE_MARKER                            = 0x8268
	DEBUG_TYPE_PUSH_GROUP                        = 0x8269
	DEBUG_TYPE_POP_GROUP                         = 0x826A
	DEBUG_SEVERITY_HIGH                          = 0x9146
	DEBUG_SEVERITY_MEDIUM                        = 0x9147
	DEBUG_SEVERITY_LOW                           = 0x9148
	DEBUG_SEVERITY_NOTIFICATION                  = 0x826B
	STACK_UNDERFLOW                              = 0x0504
	STACK_OVERFLOW                               = 0x0503
	BUFFER                                       = 0x82E0
	SHADER                                       = 0x82E1
	PROGRAM                                      = 0x82E2
	QUERY                                        = 0x82E3
	PROGRAM_PIPELINE                             = 0x82E4
	SAMPLER                                      = 0x82E6
	TEXTURE_EXTERNAL                             = 0x8D65
	TEXTURE_BINDING_EXTERNAL                     = 0x8D67
	TEXTURE_RECTANGLE                            = 0x84F5
	TEXTURE_BINDING_RECTANGLE                    = 0x84F6
	MAX_WINDOW_RECTANGLES                        = 0x8f14
	INCLUSIVE                                    = 0x8f10
	EXCLUSIVE                                    = 0x8f11
	COLOR_BUFFER_BIT0                            = 0x00000001
	COLOR_BUFFER_BIT1                            = 0x00000002
	COLOR_BUFFER_BIT2                            = 0x00000004
	COLOR_BUFFER_BIT3                            = 0x00000008
	COLOR_BUFFER_BIT4                            = 0x00000010
	COLOR_BUFFER_BIT5                            = 0x00000020
	COLOR_BUFFER_BIT6                            = 0x00000040
	COLOR_BUFFER_BIT7                            = 0x00000080
	DEPTH_BUFFER_BIT0                            = 0x00000100
	DEPTH_BUFFER_BIT1                            = 0x00000200
	DEPTH_BUFFER_BIT2                            = 0x00000400
	DEPTH_BUFFER_BIT3                            = 0x00000800
	DEPTH_BUFFER_BIT4                            = 0x00001000
	DEPTH_BUFFER_BIT5                            = 0x00002000
	DEPTH_BUFFER_BIT6                            = 0x00004000
	DEPTH_BUFFER_BIT7                            = 0x00008000
	STENCIL_BUFFER_BIT0                          = 0x00010000
	STENCIL_BUFFER_BIT1                          = 0x00020000
	STENCIL_BUFFER_BIT2                          = 0x00040000
	STENCIL_BUFFER_BIT3                          = 0x00080000
	STENCIL_BUFFER_BIT4                          = 0x00100000
	STENCIL_BUFFER_BIT5                          = 0x00200000
	STENCIL_BUFFER_BIT6                          = 0x00400000
	STENCIL_BUFFER_BIT7                          = 0x00800000
	MULTISAMPLE_BUFFER_BIT0                      = 0x01000000
	MULTISAMPLE_BUFFER_BIT1                      = 0x02000000
	MULTISAMPLE_BUFFER_BIT2                      = 0x04000000
	MULTISAMPLE_BUFFER_BIT3                      = 0x08000000
	MULTISAMPLE_BUFFER_BIT4                      = 0x10000000
	MULTISAMPLE_BUFFER_BIT5                      = 0x20000000
	MULTISAMPLE_BUFFER_BIT6                      = 0x40000000
	MULTISAMPLE_BUFFER_BIT7                      = 0x80000000
	SYNC_GPU_COMMANDS_COMPLETE                   = 0x9117
	ALREADY_SIGNALED                             = 0x911A
	TIMEOUT_EXPIRED                              = 0x911B
	CONDITION_SATISFIED                          = 0x911C
	WAIT_FAILED                                  = 0x911D
	SYNC_FLUSH_COMMANDS_BIT                      = 0x00000001
	TIMEOUT_IGNORED                              = 0xFFFFFFFFFFFFFFFF
	LINES_ADJACENCY                              = 0x000A
	PATCH_VERTICES                               = 0x8E72
	NUM_SAMPLE_COUNTS                            = 0x9380
	EGL_EXTENSIONS                               = 0x3055
	EGL_GL_TEXTURE_2D                            = 0x30B1
	EGL_GL_TEXTURE_LEVEL                         = 0x30BC
	EGL_IMAGE_PRESERVED                          = 0x30D2
	EGL_FALSE                                    = 0x0
	EGL_TRUE                                     = 0x1
	EGL_NONE                                     = 0x3038
	PROGRAM_BINARY_RETRIEVABLE_HINT              = 0x8257
	CONSERVATIVE_RASTERIZATION                   = 0x9346
	VERTEX_ATTRIB_ARRAY_BARRIER_BIT              = 0x0001
	ELEMENT_ARRAY_BARRIER_BIT                    = 0x0002
	UNIFORM_BARRIER_BIT                          = 0x0004
	TEXTURE_FETCH_BARRIER_BIT                    = 0x0008
	SHADER_IMAGE_ACCESS_BARRIER_BIT              = 0x0020
	COMMAND_BARRIER_BIT                          = 0x0040
	PIXEL_BUFFER_BARRIER_BIT                     = 0x0080
	TEXTURE_UPDATE_BARRIER_BIT                   = 0x0100
	BUFFER_UPDATE_BARRIER_BIT                    = 0x0200
	FRAMEBUFFER_BARRIER_BIT                      = 0x0400
	TRANSFORM_FEEDBACK_BARRIER_BIT               = 0x0800
	ATOMIC_COUNTER_BARRIER_BIT                   = 0x1000
	ALL_BARRIER_BITS                             = 0xffffffff
	ALL_COMPLETED                                = 0x84F2
	MAX_TESS_GEN_LEVEL_OES                       = 0x8E7E
	CLIENT_ARRAYS_ANGLE                          = 0x93AA
)

// Functions that pass floats by value cannot go through purego.SyscallN when the signature mixes integer and float
// registers (SyscallN copies the first eight arguments into both register banks), so any function with a float-by-value
// parameter is bound through purego.RegisterFunc instead. The same goes for functions whose stack arguments (beyond the
// eight register slots) include a sub-8-byte value followed by further arguments: Apple's arm64 ABI packs stack
// arguments at natural size/alignment while SyscallN spills 8-byte slots, so glBlitFramebuffer's (mask, filter) tail —
// and anything shaped like it — would be mis-marshaled (the callee reads filter from the high half of mask's slot).
// Everything else uses the SyscallN fast path.

// glFuncIndex indexes the per-entry-point call counters (Functions.callCounts) and the glFuncNames table. See
// callcounts.go for the reporting API.
type glFuncIndex int

const (
	idxActiveTexture glFuncIndex = iota
	idxAttachShader
	idxBeginQuery
	idxBindAttribLocation
	idxBindBuffer
	idxBindFragDataLocation
	idxBindFragDataLocationIndexed
	idxBindFramebuffer
	idxBindRenderbuffer
	idxBindSampler
	idxBindTexture
	idxBindUniformLocation
	idxBindVertexArray
	idxBlendBarrier
	idxBlendColor
	idxBlendEquation
	idxBlendFunc
	idxBlitFramebuffer
	idxBufferData
	idxBufferSubData
	idxCheckFramebufferStatus
	idxClear
	idxClearColor
	idxClearStencil
	idxClearTexImage
	idxClearTexSubImage
	idxClientWaitSync
	idxColorMask
	idxCompileShader
	idxCompressedTexImage2D
	idxCompressedTexSubImage2D
	idxCopyBufferSubData
	idxCopyTexSubImage2D
	idxCoverageModulation
	idxCreateProgram
	idxCreateShader
	idxCullFace
	idxDebugMessageCallback
	idxDebugMessageControl
	idxDebugMessageInsert
	idxDeleteBuffers
	idxDeleteFences
	idxDeleteFramebuffers
	idxDeleteProgram
	idxDeleteQueries
	idxDeleteRenderbuffers
	idxDeleteSamplers
	idxDeleteShader
	idxDeleteSync
	idxDeleteTextures
	idxDeleteVertexArrays
	idxDepthMask
	idxDisable
	idxDisableVertexAttribArray
	idxDiscardFramebuffer
	idxDrawArrays
	idxDrawArraysIndirect
	idxDrawArraysInstanced
	idxDrawArraysInstancedBaseInstance
	idxDrawBuffer
	idxDrawBuffers
	idxDrawElements
	idxDrawElementsIndirect
	idxDrawElementsInstanced
	idxDrawElementsInstancedBaseVertexBaseInstance
	idxDrawRangeElements
	idxEnable
	idxEnableVertexAttribArray
	idxEndQuery
	idxEndTiling
	idxFenceSync
	idxFinish
	idxFinishFence
	idxFlush
	idxFlushMappedBufferRange
	idxFramebufferRenderbuffer
	idxFramebufferTexture2D
	idxFramebufferTexture2DMultisample
	idxFrontFace
	idxGenBuffers
	idxGenFences
	idxGenFramebuffers
	idxGenQueries
	idxGenRenderbuffers
	idxGenSamplers
	idxGenTextures
	idxGenVertexArrays
	idxGenerateMipmap
	idxGetBufferParameteriv
	idxGetDebugMessageLog
	idxGetError
	idxGetFloatv
	idxGetFramebufferAttachmentParameteriv
	idxGetIntegerv
	idxGetInternalformativ
	idxGetMultisamplefv
	idxGetProgramBinary
	idxGetProgramInfoLog
	idxGetProgramiv
	idxGetQueryObjecti64v
	idxGetQueryObjectui64v
	idxGetQueryObjectuiv
	idxGetQueryiv
	idxGetRenderbufferParameteriv
	idxGetShaderInfoLog
	idxGetShaderPrecisionFormat
	idxGetShaderiv
	idxGetString
	idxGetStringi
	idxGetTexLevelParameteriv
	idxGetUniformLocation
	idxInsertEventMarker
	idxInvalidateBufferData
	idxInvalidateBufferSubData
	idxInvalidateFramebuffer
	idxInvalidateSubFramebuffer
	idxInvalidateTexImage
	idxInvalidateTexSubImage
	idxIsSync
	idxIsTexture
	idxLineWidth
	idxLinkProgram
	idxMapBuffer
	idxMapBufferRange
	idxMapBufferSubData
	idxMapTexSubImage2D
	idxMemoryBarrier
	idxMultiDrawArraysIndirect
	idxMultiDrawArraysInstancedBaseInstance
	idxMultiDrawElementsIndirect
	idxMultiDrawElementsInstancedBaseVertexBaseInstance
	idxObjectLabel
	idxPatchParameteri
	idxPixelStorei
	idxPolygonMode
	idxPopDebugGroup
	idxPopGroupMarker
	idxProgramBinary
	idxProgramParameteri
	idxPushDebugGroup
	idxPushGroupMarker
	idxQueryCounter
	idxReadBuffer
	idxReadPixels
	idxRenderbufferStorage
	idxRenderbufferStorageMultisample
	idxResolveMultisampleFramebuffer
	idxSamplerParameterf
	idxSamplerParameteri
	idxSamplerParameteriv
	idxScissor
	idxSetFence
	idxShaderSource
	idxStartTiling
	idxStencilFunc
	idxStencilFuncSeparate
	idxStencilMask
	idxStencilMaskSeparate
	idxStencilOp
	idxStencilOpSeparate
	idxTestFence
	idxTexBuffer
	idxTexBufferRange
	idxTexImage2D
	idxTexParameterf
	idxTexParameterfv
	idxTexParameteri
	idxTexParameteriv
	idxTexStorage2D
	idxTexSubImage2D
	idxTextureBarrier
	idxUniform1f
	idxUniform1fv
	idxUniform1i
	idxUniform1iv
	idxUniform2f
	idxUniform2fv
	idxUniform2i
	idxUniform2iv
	idxUniform3f
	idxUniform3fv
	idxUniform3i
	idxUniform3iv
	idxUniform4f
	idxUniform4fv
	idxUniform4i
	idxUniform4iv
	idxUniformMatrix2fv
	idxUniformMatrix3fv
	idxUniformMatrix4fv
	idxUnmapBuffer
	idxUnmapBufferSubData
	idxUnmapTexSubImage2D
	idxUseProgram
	idxVertexAttrib1f
	idxVertexAttrib2fv
	idxVertexAttrib3fv
	idxVertexAttrib4fv
	idxVertexAttribDivisor
	idxVertexAttribIPointer
	idxVertexAttribPointer
	idxViewport
	idxWaitSync
	idxWindowRectangles
	glFuncCount
)

// glFuncNames maps glFuncIndex to the GL entry-point name.
var glFuncNames = [glFuncCount]string{
	idxActiveTexture:                   "glActiveTexture",
	idxAttachShader:                    "glAttachShader",
	idxBeginQuery:                      "glBeginQuery",
	idxBindAttribLocation:              "glBindAttribLocation",
	idxBindBuffer:                      "glBindBuffer",
	idxBindFragDataLocation:            "glBindFragDataLocation",
	idxBindFragDataLocationIndexed:     "glBindFragDataLocationIndexed",
	idxBindFramebuffer:                 "glBindFramebuffer",
	idxBindRenderbuffer:                "glBindRenderbuffer",
	idxBindSampler:                     "glBindSampler",
	idxBindTexture:                     "glBindTexture",
	idxBindUniformLocation:             "glBindUniformLocation",
	idxBindVertexArray:                 "glBindVertexArray",
	idxBlendBarrier:                    "glBlendBarrier",
	idxBlendColor:                      "glBlendColor",
	idxBlendEquation:                   "glBlendEquation",
	idxBlendFunc:                       "glBlendFunc",
	idxBlitFramebuffer:                 "glBlitFramebuffer",
	idxBufferData:                      "glBufferData",
	idxBufferSubData:                   "glBufferSubData",
	idxCheckFramebufferStatus:          "glCheckFramebufferStatus",
	idxClear:                           "glClear",
	idxClearColor:                      "glClearColor",
	idxClearStencil:                    "glClearStencil",
	idxClearTexImage:                   "glClearTexImage",
	idxClearTexSubImage:                "glClearTexSubImage",
	idxClientWaitSync:                  "glClientWaitSync",
	idxColorMask:                       "glColorMask",
	idxCompileShader:                   "glCompileShader",
	idxCompressedTexImage2D:            "glCompressedTexImage2D",
	idxCompressedTexSubImage2D:         "glCompressedTexSubImage2D",
	idxCopyBufferSubData:               "glCopyBufferSubData",
	idxCopyTexSubImage2D:               "glCopyTexSubImage2D",
	idxCoverageModulation:              "glCoverageModulation",
	idxCreateProgram:                   "glCreateProgram",
	idxCreateShader:                    "glCreateShader",
	idxCullFace:                        "glCullFace",
	idxDebugMessageCallback:            "glDebugMessageCallback",
	idxDebugMessageControl:             "glDebugMessageControl",
	idxDebugMessageInsert:              "glDebugMessageInsert",
	idxDeleteBuffers:                   "glDeleteBuffers",
	idxDeleteFences:                    "glDeleteFences",
	idxDeleteFramebuffers:              "glDeleteFramebuffers",
	idxDeleteProgram:                   "glDeleteProgram",
	idxDeleteQueries:                   "glDeleteQueries",
	idxDeleteRenderbuffers:             "glDeleteRenderbuffers",
	idxDeleteSamplers:                  "glDeleteSamplers",
	idxDeleteShader:                    "glDeleteShader",
	idxDeleteSync:                      "glDeleteSync",
	idxDeleteTextures:                  "glDeleteTextures",
	idxDeleteVertexArrays:              "glDeleteVertexArrays",
	idxDepthMask:                       "glDepthMask",
	idxDisable:                         "glDisable",
	idxDisableVertexAttribArray:        "glDisableVertexAttribArray",
	idxDiscardFramebuffer:              "glDiscardFramebuffer",
	idxDrawArrays:                      "glDrawArrays",
	idxDrawArraysIndirect:              "glDrawArraysIndirect",
	idxDrawArraysInstanced:             "glDrawArraysInstanced",
	idxDrawArraysInstancedBaseInstance: "glDrawArraysInstancedBaseInstance",
	idxDrawBuffer:                      "glDrawBuffer",
	idxDrawBuffers:                     "glDrawBuffers",
	idxDrawElements:                    "glDrawElements",
	idxDrawElementsIndirect:            "glDrawElementsIndirect",
	idxDrawElementsInstanced:           "glDrawElementsInstanced",
	idxDrawElementsInstancedBaseVertexBaseInstance: "glDrawElementsInstancedBaseVertexBaseInstance",
	idxDrawRangeElements:                           "glDrawRangeElements",
	idxEnable:                                      "glEnable",
	idxEnableVertexAttribArray:                     "glEnableVertexAttribArray",
	idxEndQuery:                                    "glEndQuery",
	idxEndTiling:                                   "glEndTiling",
	idxFenceSync:                                   "glFenceSync",
	idxFinish:                                      "glFinish",
	idxFinishFence:                                 "glFinishFence",
	idxFlush:                                       "glFlush",
	idxFlushMappedBufferRange:                      "glFlushMappedBufferRange",
	idxFramebufferRenderbuffer:                     "glFramebufferRenderbuffer",
	idxFramebufferTexture2D:                        "glFramebufferTexture2D",
	idxFramebufferTexture2DMultisample:             "glFramebufferTexture2DMultisample",
	idxFrontFace:                                   "glFrontFace",
	idxGenBuffers:                                  "glGenBuffers",
	idxGenFences:                                   "glGenFences",
	idxGenFramebuffers:                             "glGenFramebuffers",
	idxGenQueries:                                  "glGenQueries",
	idxGenRenderbuffers:                            "glGenRenderbuffers",
	idxGenSamplers:                                 "glGenSamplers",
	idxGenTextures:                                 "glGenTextures",
	idxGenVertexArrays:                             "glGenVertexArrays",
	idxGenerateMipmap:                              "glGenerateMipmap",
	idxGetBufferParameteriv:                        "glGetBufferParameteriv",
	idxGetDebugMessageLog:                          "glGetDebugMessageLog",
	idxGetError:                                    "glGetError",
	idxGetFloatv:                                   "glGetFloatv",
	idxGetFramebufferAttachmentParameteriv:         "glGetFramebufferAttachmentParameteriv",
	idxGetIntegerv:                                 "glGetIntegerv",
	idxGetInternalformativ:                         "glGetInternalformativ",
	idxGetMultisamplefv:                            "glGetMultisamplefv",
	idxGetProgramBinary:                            "glGetProgramBinary",
	idxGetProgramInfoLog:                           "glGetProgramInfoLog",
	idxGetProgramiv:                                "glGetProgramiv",
	idxGetQueryObjecti64v:                          "glGetQueryObjecti64v",
	idxGetQueryObjectui64v:                         "glGetQueryObjectui64v",
	idxGetQueryObjectuiv:                           "glGetQueryObjectuiv",
	idxGetQueryiv:                                  "glGetQueryiv",
	idxGetRenderbufferParameteriv:                  "glGetRenderbufferParameteriv",
	idxGetShaderInfoLog:                            "glGetShaderInfoLog",
	idxGetShaderPrecisionFormat:                    "glGetShaderPrecisionFormat",
	idxGetShaderiv:                                 "glGetShaderiv",
	idxGetString:                                   "glGetString",
	idxGetStringi:                                  "glGetStringi",
	idxGetTexLevelParameteriv:                      "glGetTexLevelParameteriv",
	idxGetUniformLocation:                          "glGetUniformLocation",
	idxInsertEventMarker:                           "glInsertEventMarker",
	idxInvalidateBufferData:                        "glInvalidateBufferData",
	idxInvalidateBufferSubData:                     "glInvalidateBufferSubData",
	idxInvalidateFramebuffer:                       "glInvalidateFramebuffer",
	idxInvalidateSubFramebuffer:                    "glInvalidateSubFramebuffer",
	idxInvalidateTexImage:                          "glInvalidateTexImage",
	idxInvalidateTexSubImage:                       "glInvalidateTexSubImage",
	idxIsSync:                                      "glIsSync",
	idxIsTexture:                                   "glIsTexture",
	idxLineWidth:                                   "glLineWidth",
	idxLinkProgram:                                 "glLinkProgram",
	idxMapBuffer:                                   "glMapBuffer",
	idxMapBufferRange:                              "glMapBufferRange",
	idxMapBufferSubData:                            "glMapBufferSubData",
	idxMapTexSubImage2D:                            "glMapTexSubImage2D",
	idxMemoryBarrier:                               "glMemoryBarrier",
	idxMultiDrawArraysIndirect:                     "glMultiDrawArraysIndirect",
	idxMultiDrawArraysInstancedBaseInstance:        "glMultiDrawArraysInstancedBaseInstance",
	idxMultiDrawElementsIndirect:                   "glMultiDrawElementsIndirect",
	idxMultiDrawElementsInstancedBaseVertexBaseInstance: "glMultiDrawElementsInstancedBaseVertexBaseInstance",
	idxObjectLabel:                    "glObjectLabel",
	idxPatchParameteri:                "glPatchParameteri",
	idxPixelStorei:                    "glPixelStorei",
	idxPolygonMode:                    "glPolygonMode",
	idxPopDebugGroup:                  "glPopDebugGroup",
	idxPopGroupMarker:                 "glPopGroupMarker",
	idxProgramBinary:                  "glProgramBinary",
	idxProgramParameteri:              "glProgramParameteri",
	idxPushDebugGroup:                 "glPushDebugGroup",
	idxPushGroupMarker:                "glPushGroupMarker",
	idxQueryCounter:                   "glQueryCounter",
	idxReadBuffer:                     "glReadBuffer",
	idxReadPixels:                     "glReadPixels",
	idxRenderbufferStorage:            "glRenderbufferStorage",
	idxRenderbufferStorageMultisample: "glRenderbufferStorageMultisample",
	idxResolveMultisampleFramebuffer:  "glResolveMultisampleFramebuffer",
	idxSamplerParameterf:              "glSamplerParameterf",
	idxSamplerParameteri:              "glSamplerParameteri",
	idxSamplerParameteriv:             "glSamplerParameteriv",
	idxScissor:                        "glScissor",
	idxSetFence:                       "glSetFence",
	idxShaderSource:                   "glShaderSource",
	idxStartTiling:                    "glStartTiling",
	idxStencilFunc:                    "glStencilFunc",
	idxStencilFuncSeparate:            "glStencilFuncSeparate",
	idxStencilMask:                    "glStencilMask",
	idxStencilMaskSeparate:            "glStencilMaskSeparate",
	idxStencilOp:                      "glStencilOp",
	idxStencilOpSeparate:              "glStencilOpSeparate",
	idxTestFence:                      "glTestFence",
	idxTexBuffer:                      "glTexBuffer",
	idxTexBufferRange:                 "glTexBufferRange",
	idxTexImage2D:                     "glTexImage2D",
	idxTexParameterf:                  "glTexParameterf",
	idxTexParameterfv:                 "glTexParameterfv",
	idxTexParameteri:                  "glTexParameteri",
	idxTexParameteriv:                 "glTexParameteriv",
	idxTexStorage2D:                   "glTexStorage2D",
	idxTexSubImage2D:                  "glTexSubImage2D",
	idxTextureBarrier:                 "glTextureBarrier",
	idxUniform1f:                      "glUniform1f",
	idxUniform1fv:                     "glUniform1fv",
	idxUniform1i:                      "glUniform1i",
	idxUniform1iv:                     "glUniform1iv",
	idxUniform2f:                      "glUniform2f",
	idxUniform2fv:                     "glUniform2fv",
	idxUniform2i:                      "glUniform2i",
	idxUniform2iv:                     "glUniform2iv",
	idxUniform3f:                      "glUniform3f",
	idxUniform3fv:                     "glUniform3fv",
	idxUniform3i:                      "glUniform3i",
	idxUniform3iv:                     "glUniform3iv",
	idxUniform4f:                      "glUniform4f",
	idxUniform4fv:                     "glUniform4fv",
	idxUniform4i:                      "glUniform4i",
	idxUniform4iv:                     "glUniform4iv",
	idxUniformMatrix2fv:               "glUniformMatrix2fv",
	idxUniformMatrix3fv:               "glUniformMatrix3fv",
	idxUniformMatrix4fv:               "glUniformMatrix4fv",
	idxUnmapBuffer:                    "glUnmapBuffer",
	idxUnmapBufferSubData:             "glUnmapBufferSubData",
	idxUnmapTexSubImage2D:             "glUnmapTexSubImage2D",
	idxUseProgram:                     "glUseProgram",
	idxVertexAttrib1f:                 "glVertexAttrib1f",
	idxVertexAttrib2fv:                "glVertexAttrib2fv",
	idxVertexAttrib3fv:                "glVertexAttrib3fv",
	idxVertexAttrib4fv:                "glVertexAttrib4fv",
	idxVertexAttribDivisor:            "glVertexAttribDivisor",
	idxVertexAttribIPointer:           "glVertexAttribIPointer",
	idxVertexAttribPointer:            "glVertexAttribPointer",
	idxViewport:                       "glViewport",
	idxWaitSync:                       "glWaitSync",
	idxWindowRectangles:               "glWindowRectangles",
}

// Functions holds one proc address per GL entry point this package can call. A zero proc means the function was not
// resolved for the context's version and extension set. All-integer entry points dispatch through the fixed-arity
// glCall lane (fastcall_*.go; zero-allocation trampolines on the supported platforms, purego.SyscallN elsewhere).
// Float-by-value entry points additionally carry a purego.RegisterFunc-bound typed func created by initRegistered,
// because the integer call lanes cannot pass mixed integer/float arguments. Every wrapper increments its callCounts
// slot before dispatching, so per-entry-point GL call counts are always available (callcounts.go); a plain array
// increment costs nothing measurable next to the FFI transition, and counts are only meaningful on the GL context
// thread, which is the only place calls originate.
type Functions struct {
	fnVertexAttrib1f                                 func(uint32, float32)
	fnBlitFramebuffer                                func(int32, int32, int32, int32, int32, int32, int32, int32, uint32, uint32)
	fnClearColor                                     func(float32, float32, float32, float32)
	fnClearTexSubImage                               func(uint32, int32, int32, int32, int32, int32, int32, int32, uint32, uint32, uintptr)
	fnLineWidth                                      func(float32)
	fnBlendColor                                     func(float32, float32, float32, float32)
	fnUniform4f                                      func(int32, float32, float32, float32, float32)
	fnUniform3f                                      func(int32, float32, float32, float32)
	fnUniform2f                                      func(int32, float32, float32)
	fnUniform1f                                      func(int32, float32)
	fnTexParameterf                                  func(uint32, uint32, float32)
	fnSamplerParameterf                              func(uint32, uint32, float32)
	callCounts                                       [glFuncCount]uint64
	getRenderbufferParameteriv                       uintptr
	drawArrays                                       uintptr
	bindTexture                                      uintptr
	bindUniformLocation                              uintptr
	bindVertexArray                                  uintptr
	blendBarrier                                     uintptr
	blendColor                                       uintptr
	blendEquation                                    uintptr
	blendFunc                                        uintptr
	blitFramebuffer                                  uintptr
	bufferData                                       uintptr
	getStringi                                       uintptr
	checkFramebufferStatus                           uintptr
	clear                                            uintptr
	clearColor                                       uintptr
	clearStencil                                     uintptr
	clearTexImage                                    uintptr
	clearTexSubImage                                 uintptr
	clientWaitSync                                   uintptr
	colorMask                                        uintptr
	compileShader                                    uintptr
	compressedTexImage2D                             uintptr
	compressedTexSubImage2D                          uintptr
	copyBufferSubData                                uintptr
	copyTexSubImage2D                                uintptr
	coverageModulation                               uintptr
	createProgram                                    uintptr
	createShader                                     uintptr
	cullFace                                         uintptr
	debugMessageCallback                             uintptr
	debugMessageControl                              uintptr
	debugMessageInsert                               uintptr
	deleteBuffers                                    uintptr
	deleteFences                                     uintptr
	deleteFramebuffers                               uintptr
	deleteProgram                                    uintptr
	deleteQueries                                    uintptr
	deleteRenderbuffers                              uintptr
	deleteSamplers                                   uintptr
	deleteShader                                     uintptr
	deleteSync                                       uintptr
	deleteTextures                                   uintptr
	deleteVertexArrays                               uintptr
	depthMask                                        uintptr
	disable                                          uintptr
	disableVertexAttribArray                         uintptr
	discardFramebuffer                               uintptr
	getTexLevelParameteriv                           uintptr
	drawArraysIndirect                               uintptr
	drawArraysInstanced                              uintptr
	drawArraysInstancedBaseInstance                  uintptr
	drawBuffer                                       uintptr
	drawBuffers                                      uintptr
	drawElements                                     uintptr
	drawElementsIndirect                             uintptr
	drawElementsInstanced                            uintptr
	drawElementsInstancedBaseVertexBaseInstance      uintptr
	drawRangeElements                                uintptr
	enable                                           uintptr
	enableVertexAttribArray                          uintptr
	endQuery                                         uintptr
	endTiling                                        uintptr
	fenceSync                                        uintptr
	finish                                           uintptr
	finishFence                                      uintptr
	flush                                            uintptr
	flushMappedBufferRange                           uintptr
	framebufferRenderbuffer                          uintptr
	framebufferTexture2D                             uintptr
	framebufferTexture2DMultisample                  uintptr
	frontFace                                        uintptr
	genBuffers                                       uintptr
	genFences                                        uintptr
	genFramebuffers                                  uintptr
	genQueries                                       uintptr
	genRenderbuffers                                 uintptr
	genSamplers                                      uintptr
	genTextures                                      uintptr
	genVertexArrays                                  uintptr
	generateMipmap                                   uintptr
	getBufferParameteriv                             uintptr
	getDebugMessageLog                               uintptr
	getError                                         uintptr
	getFloatv                                        uintptr
	getFramebufferAttachmentParameteriv              uintptr
	getIntegerv                                      uintptr
	getInternalformativ                              uintptr
	getMultisamplefv                                 uintptr
	getProgramBinary                                 uintptr
	getProgramInfoLog                                uintptr
	getProgramiv                                     uintptr
	getQueryObjecti64v                               uintptr
	getQueryObjectui64v                              uintptr
	getQueryObjectuiv                                uintptr
	getQueryiv                                       uintptr
	bindRenderbuffer                                 uintptr
	getShaderInfoLog                                 uintptr
	getShaderPrecisionFormat                         uintptr
	getShaderiv                                      uintptr
	getString                                        uintptr
	bufferSubData                                    uintptr
	bindSampler                                      uintptr
	vertexAttrib2fv                                  uintptr
	insertEventMarker                                uintptr
	invalidateBufferData                             uintptr
	invalidateBufferSubData                          uintptr
	invalidateFramebuffer                            uintptr
	invalidateSubFramebuffer                         uintptr
	invalidateTexImage                               uintptr
	invalidateTexSubImage                            uintptr
	isSync                                           uintptr
	isTexture                                        uintptr
	lineWidth                                        uintptr
	linkProgram                                      uintptr
	mapBuffer                                        uintptr
	mapBufferRange                                   uintptr
	mapBufferSubData                                 uintptr
	mapTexSubImage2D                                 uintptr
	memoryBarrier                                    uintptr
	multiDrawArraysIndirect                          uintptr
	multiDrawArraysInstancedBaseInstance             uintptr
	multiDrawElementsIndirect                        uintptr
	multiDrawElementsInstancedBaseVertexBaseInstance uintptr
	objectLabel                                      uintptr
	patchParameteri                                  uintptr
	pixelStorei                                      uintptr
	polygonMode                                      uintptr
	popDebugGroup                                    uintptr
	popGroupMarker                                   uintptr
	programBinary                                    uintptr
	programParameteri                                uintptr
	pushDebugGroup                                   uintptr
	pushGroupMarker                                  uintptr
	queryCounter                                     uintptr
	readBuffer                                       uintptr
	readPixels                                       uintptr
	renderbufferStorage                              uintptr
	renderbufferStorageMultisample                   uintptr
	resolveMultisampleFramebuffer                    uintptr
	samplerParameterf                                uintptr
	samplerParameteri                                uintptr
	samplerParameteriv                               uintptr
	scissor                                          uintptr
	setFence                                         uintptr
	shaderSource                                     uintptr
	startTiling                                      uintptr
	stencilFunc                                      uintptr
	stencilFuncSeparate                              uintptr
	stencilMask                                      uintptr
	stencilMaskSeparate                              uintptr
	stencilOp                                        uintptr
	stencilOpSeparate                                uintptr
	testFence                                        uintptr
	texBuffer                                        uintptr
	texBufferRange                                   uintptr
	texImage2D                                       uintptr
	texParameterf                                    uintptr
	texParameterfv                                   uintptr
	texParameteri                                    uintptr
	texParameteriv                                   uintptr
	texStorage2D                                     uintptr
	texSubImage2D                                    uintptr
	textureBarrier                                   uintptr
	uniform1f                                        uintptr
	uniform1fv                                       uintptr
	uniform1i                                        uintptr
	uniform1iv                                       uintptr
	uniform2f                                        uintptr
	uniform2fv                                       uintptr
	uniform2i                                        uintptr
	uniform2iv                                       uintptr
	uniform3f                                        uintptr
	uniform3fv                                       uintptr
	uniform3i                                        uintptr
	uniform3iv                                       uintptr
	uniform4f                                        uintptr
	uniform4fv                                       uintptr
	uniform4i                                        uintptr
	uniform4iv                                       uintptr
	uniformMatrix2fv                                 uintptr
	uniformMatrix3fv                                 uintptr
	uniformMatrix4fv                                 uintptr
	unmapBuffer                                      uintptr
	unmapBufferSubData                               uintptr
	unmapTexSubImage2D                               uintptr
	useProgram                                       uintptr
	vertexAttrib1f                                   uintptr
	getUniformLocation                               uintptr
	vertexAttrib3fv                                  uintptr
	vertexAttrib4fv                                  uintptr
	vertexAttribDivisor                              uintptr
	bindFramebuffer                                  uintptr
	bindFragDataLocationIndexed                      uintptr
	bindFragDataLocation                             uintptr
	bindBuffer                                       uintptr
	bindAttribLocation                               uintptr
	beginQuery                                       uintptr
	attachShader                                     uintptr
	activeTexture                                    uintptr
	vertexAttribIPointer                             uintptr
	vertexAttribPointer                              uintptr
	viewport                                         uintptr
	waitSync                                         uintptr
	windowRectangles                                 uintptr
}

// initRegistered creates the typed bindings for the float-by-value entry points that were resolved.
func (f *Functions) initRegistered() {
	if f.blendColor != 0 {
		purego.RegisterFunc(&f.fnBlendColor, f.blendColor)
	}
	if f.blitFramebuffer != 0 {
		purego.RegisterFunc(&f.fnBlitFramebuffer, f.blitFramebuffer)
	}
	if f.clearColor != 0 {
		purego.RegisterFunc(&f.fnClearColor, f.clearColor)
	}
	if f.clearTexSubImage != 0 {
		purego.RegisterFunc(&f.fnClearTexSubImage, f.clearTexSubImage)
	}
	if f.lineWidth != 0 {
		purego.RegisterFunc(&f.fnLineWidth, f.lineWidth)
	}
	if f.samplerParameterf != 0 {
		purego.RegisterFunc(&f.fnSamplerParameterf, f.samplerParameterf)
	}
	if f.texParameterf != 0 {
		purego.RegisterFunc(&f.fnTexParameterf, f.texParameterf)
	}
	if f.uniform1f != 0 {
		purego.RegisterFunc(&f.fnUniform1f, f.uniform1f)
	}
	if f.uniform2f != 0 {
		purego.RegisterFunc(&f.fnUniform2f, f.uniform2f)
	}
	if f.uniform3f != 0 {
		purego.RegisterFunc(&f.fnUniform3f, f.uniform3f)
	}
	if f.uniform4f != 0 {
		purego.RegisterFunc(&f.fnUniform4f, f.uniform4f)
	}
	if f.vertexAttrib1f != 0 {
		purego.RegisterFunc(&f.fnVertexAttrib1f, f.vertexAttrib1f)
	}
}

// ActiveTexture calls glActiveTexture.
func (f *Functions) ActiveTexture(texture uint32) {
	f.callCounts[idxActiveTexture]++
	glCall(f.activeTexture, uintptr(texture), 0, 0, 0, 0, 0, 0, 0, 0)
}

// AttachShader calls glAttachShader.
func (f *Functions) AttachShader(program, shader uint32) {
	f.callCounts[idxAttachShader]++
	glCall(f.attachShader, uintptr(program), uintptr(shader), 0, 0, 0, 0, 0, 0, 0)
}

// BeginQuery calls glBeginQuery.
func (f *Functions) BeginQuery(target, id uint32) {
	f.callCounts[idxBeginQuery]++
	glCall(f.beginQuery, uintptr(target), uintptr(id), 0, 0, 0, 0, 0, 0, 0)
}

// BindAttribLocation calls glBindAttribLocation.
func (f *Functions) BindAttribLocation(program, index uint32, name *byte) {
	f.callCounts[idxBindAttribLocation]++
	glCall(f.bindAttribLocation, uintptr(program), uintptr(index), uintptr(unsafe.Pointer(name)), 0, 0, 0, 0, 0, 0)
}

// BindBuffer calls glBindBuffer.
func (f *Functions) BindBuffer(target, buffer uint32) {
	f.callCounts[idxBindBuffer]++
	glCall(f.bindBuffer, uintptr(target), uintptr(buffer), 0, 0, 0, 0, 0, 0, 0)
}

// BindFragDataLocation calls glBindFragDataLocation.
func (f *Functions) BindFragDataLocation(program, colorNumber uint32, name *byte) {
	f.callCounts[idxBindFragDataLocation]++
	glCall(f.bindFragDataLocation, uintptr(program), uintptr(colorNumber), uintptr(unsafe.Pointer(name)),
		0, 0, 0, 0, 0, 0)
}

// BindFragDataLocationIndexed calls glBindFragDataLocationIndexed.
func (f *Functions) BindFragDataLocationIndexed(program, colorNumber, index uint32, name *byte) {
	f.callCounts[idxBindFragDataLocationIndexed]++
	glCall(f.bindFragDataLocationIndexed, uintptr(program), uintptr(colorNumber), uintptr(index),
		uintptr(unsafe.Pointer(name)), 0, 0, 0, 0, 0)
}

// BindFramebuffer calls glBindFramebuffer.
func (f *Functions) BindFramebuffer(target, framebuffer uint32) {
	f.callCounts[idxBindFramebuffer]++
	glCall(f.bindFramebuffer, uintptr(target), uintptr(framebuffer), 0, 0, 0, 0, 0, 0, 0)
}

// BindRenderbuffer calls glBindRenderbuffer.
func (f *Functions) BindRenderbuffer(target, renderbuffer uint32) {
	f.callCounts[idxBindRenderbuffer]++
	glCall(f.bindRenderbuffer, uintptr(target), uintptr(renderbuffer), 0, 0, 0, 0, 0, 0, 0)
}

// BindSampler calls glBindSampler.
func (f *Functions) BindSampler(unit, sampler uint32) {
	f.callCounts[idxBindSampler]++
	glCall(f.bindSampler, uintptr(unit), uintptr(sampler), 0, 0, 0, 0, 0, 0, 0)
}

// BindTexture calls glBindTexture.
func (f *Functions) BindTexture(target, texture uint32) {
	f.callCounts[idxBindTexture]++
	glCall(f.bindTexture, uintptr(target), uintptr(texture), 0, 0, 0, 0, 0, 0, 0)
}

// BindUniformLocation calls glBindUniformLocation.
func (f *Functions) BindUniformLocation(program uint32, location int32, name *byte) {
	f.callCounts[idxBindUniformLocation]++
	glCall(f.bindUniformLocation, uintptr(program), uintptr(location), uintptr(unsafe.Pointer(name)), 0, 0, 0, 0, 0, 0)
}

// BindVertexArray calls glBindVertexArray.
func (f *Functions) BindVertexArray(array uint32) {
	f.callCounts[idxBindVertexArray]++
	glCall(f.bindVertexArray, uintptr(array), 0, 0, 0, 0, 0, 0, 0, 0)
}

// BlendBarrier calls glBlendBarrier.
func (f *Functions) BlendBarrier() {
	f.callCounts[idxBlendBarrier]++
	glCall(f.blendBarrier, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

// BlendColor calls glBlendColor.
func (f *Functions) BlendColor(red, green, blue, alpha float32) {
	f.callCounts[idxBlendColor]++
	f.fnBlendColor(red, green, blue, alpha)
}

// BlendEquation calls glBlendEquation.
func (f *Functions) BlendEquation(mode uint32) {
	f.callCounts[idxBlendEquation]++
	glCall(f.blendEquation, uintptr(mode), 0, 0, 0, 0, 0, 0, 0, 0)
}

// BlendFunc calls glBlendFunc.
func (f *Functions) BlendFunc(sfactor, dfactor uint32) {
	f.callCounts[idxBlendFunc]++
	glCall(f.blendFunc, uintptr(sfactor), uintptr(dfactor), 0, 0, 0, 0, 0, 0, 0)
}

// BlitFramebuffer calls glBlitFramebuffer.
func (f *Functions) BlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1 int32, mask, filter uint32) {
	f.callCounts[idxBlitFramebuffer]++
	f.fnBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter)
}

// BufferData calls glBufferData.
//
//go:uintptrescapes
func (f *Functions) BufferData(target uint32, size int, data uintptr, usage uint32) {
	f.callCounts[idxBufferData]++
	glCall(f.bufferData, uintptr(target), uintptr(size), data, uintptr(usage), 0, 0, 0, 0, 0)
}

// BufferSubData calls glBufferSubData.
//
//go:uintptrescapes
func (f *Functions) BufferSubData(target uint32, offset, size int, data uintptr) {
	f.callCounts[idxBufferSubData]++
	glCall(f.bufferSubData, uintptr(target), uintptr(offset), uintptr(size), data, 0, 0, 0, 0, 0)
}

// CheckFramebufferStatus calls glCheckFramebufferStatus.
func (f *Functions) CheckFramebufferStatus(target uint32) uint32 {
	f.callCounts[idxCheckFramebufferStatus]++
	return uint32(glCall(f.checkFramebufferStatus, uintptr(target), 0, 0, 0, 0, 0, 0, 0, 0))
}

// Clear calls glClear.
func (f *Functions) Clear(mask uint32) {
	f.callCounts[idxClear]++
	glCall(f.clear, uintptr(mask), 0, 0, 0, 0, 0, 0, 0, 0)
}

// ClearColor calls glClearColor.
func (f *Functions) ClearColor(red, green, blue, alpha float32) {
	f.callCounts[idxClearColor]++
	f.fnClearColor(red, green, blue, alpha)
}

// ClearStencil calls glClearStencil.
func (f *Functions) ClearStencil(s int32) {
	f.callCounts[idxClearStencil]++
	glCall(f.clearStencil, uintptr(s), 0, 0, 0, 0, 0, 0, 0, 0)
}

// ClearTexImage calls glClearTexImage.
//
//go:uintptrescapes
func (f *Functions) ClearTexImage(texture uint32, level int32, format, typ uint32, data uintptr) {
	f.callCounts[idxClearTexImage]++
	glCall(f.clearTexImage, uintptr(texture), uintptr(level), uintptr(format), uintptr(typ), data,
		0, 0, 0, 0)
}

// ClearTexSubImage calls glClearTexSubImage.
//
//go:uintptrescapes
func (f *Functions) ClearTexSubImage(texture uint32, level, xoffset, yoffset, zoffset, width, height, depth int32, format, typ uint32, data uintptr) {
	f.callCounts[idxClearTexSubImage]++
	f.fnClearTexSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, typ, data)
}

// ClientWaitSync calls glClientWaitSync.
func (f *Functions) ClientWaitSync(sync Sync, flags uint32, timeout uint64) uint32 {
	f.callCounts[idxClientWaitSync]++
	return uint32(glCall(f.clientWaitSync, uintptr(sync), uintptr(flags), uintptr(timeout), 0, 0, 0, 0, 0, 0))
}

// ColorMask calls glColorMask.
func (f *Functions) ColorMask(red, green, blue, alpha bool) {
	f.callCounts[idxColorMask]++
	glCall(f.colorMask, glBool(red), glBool(green), glBool(blue), glBool(alpha), 0, 0, 0, 0, 0)
}

// CompileShader calls glCompileShader.
func (f *Functions) CompileShader(shader uint32) {
	f.callCounts[idxCompileShader]++
	glCall(f.compileShader, uintptr(shader), 0, 0, 0, 0, 0, 0, 0, 0)
}

// CompressedTexImage2D calls glCompressedTexImage2D.
//
//go:uintptrescapes
func (f *Functions) CompressedTexImage2D(target uint32, level int32, internalformat uint32, width, height, border, imageSize int32, data uintptr) {
	f.callCounts[idxCompressedTexImage2D]++
	glCall(f.compressedTexImage2D, uintptr(target), uintptr(level), uintptr(internalformat), uintptr(width),
		uintptr(height), uintptr(border), uintptr(imageSize), data, 0)
}

// CompressedTexSubImage2D calls glCompressedTexSubImage2D.
//
//go:uintptrescapes
func (f *Functions) CompressedTexSubImage2D(target uint32, level, xoffset, yoffset, width, height int32, format uint32, imageSize int32, data uintptr) {
	f.callCounts[idxCompressedTexSubImage2D]++
	glCall(f.compressedTexSubImage2D, uintptr(target), uintptr(level), uintptr(xoffset), uintptr(yoffset),
		uintptr(width), uintptr(height), uintptr(format), uintptr(imageSize), data)
}

// CopyBufferSubData calls glCopyBufferSubData.
func (f *Functions) CopyBufferSubData(readTargt, writeTarget uint32, readOffset, writeOffset, size int) {
	f.callCounts[idxCopyBufferSubData]++
	glCall(f.copyBufferSubData, uintptr(readTargt), uintptr(writeTarget), uintptr(readOffset), uintptr(writeOffset),
		uintptr(size), 0, 0, 0, 0)
}

// CopyTexSubImage2D calls glCopyTexSubImage2D.
func (f *Functions) CopyTexSubImage2D(target uint32, level, xoffset, yoffset, x, y, width, height int32) {
	f.callCounts[idxCopyTexSubImage2D]++
	glCall(f.copyTexSubImage2D, uintptr(target), uintptr(level), uintptr(xoffset), uintptr(yoffset), uintptr(x),
		uintptr(y), uintptr(width), uintptr(height), 0)
}

// CoverageModulation calls glCoverageModulation.
func (f *Functions) CoverageModulation(components uint32) {
	f.callCounts[idxCoverageModulation]++
	glCall(f.coverageModulation, uintptr(components), 0, 0, 0, 0, 0, 0, 0, 0)
}

// CreateProgram calls glCreateProgram.
func (f *Functions) CreateProgram() uint32 {
	f.callCounts[idxCreateProgram]++
	return uint32(glCall(f.createProgram, 0, 0, 0, 0, 0, 0, 0, 0, 0))
}

// CreateShader calls glCreateShader.
func (f *Functions) CreateShader(typ uint32) uint32 {
	f.callCounts[idxCreateShader]++
	return uint32(glCall(f.createShader, uintptr(typ), 0, 0, 0, 0, 0, 0, 0, 0))
}

// CullFace calls glCullFace.
func (f *Functions) CullFace(mode uint32) {
	f.callCounts[idxCullFace]++
	glCall(f.cullFace, uintptr(mode), 0, 0, 0, 0, 0, 0, 0, 0)
}

// DebugMessageCallback calls glDebugMessageCallback.
//
//go:uintptrescapes
func (f *Functions) DebugMessageCallback(callback, userParam uintptr) {
	f.callCounts[idxDebugMessageCallback]++
	glCall(f.debugMessageCallback, callback, userParam, 0, 0, 0, 0, 0, 0, 0)
}

// DebugMessageControl calls glDebugMessageControl.
func (f *Functions) DebugMessageControl(source, typ, severity uint32, count int32, ids *uint32, enabled bool) {
	f.callCounts[idxDebugMessageControl]++
	glCall(f.debugMessageControl, uintptr(source), uintptr(typ), uintptr(severity), uintptr(count),
		uintptr(unsafe.Pointer(ids)), glBool(enabled), 0, 0, 0)
}

// DebugMessageInsert calls glDebugMessageInsert.
func (f *Functions) DebugMessageInsert(source, typ, id, severity uint32, length int32, buf *byte) {
	f.callCounts[idxDebugMessageInsert]++
	glCall(f.debugMessageInsert, uintptr(source), uintptr(typ), uintptr(id), uintptr(severity), uintptr(length),
		uintptr(unsafe.Pointer(buf)), 0, 0, 0)
}

// DeleteBuffers calls glDeleteBuffers.
func (f *Functions) DeleteBuffers(n int32, buffers *uint32) {
	f.callCounts[idxDeleteBuffers]++
	glCall(f.deleteBuffers, uintptr(n), uintptr(unsafe.Pointer(buffers)), 0, 0, 0, 0, 0, 0, 0)
}

// DeleteFences calls glDeleteFences.
func (f *Functions) DeleteFences(n int32, fences *uint32) {
	f.callCounts[idxDeleteFences]++
	glCall(f.deleteFences, uintptr(n), uintptr(unsafe.Pointer(fences)), 0, 0, 0, 0, 0, 0, 0)
}

// DeleteFramebuffers calls glDeleteFramebuffers.
func (f *Functions) DeleteFramebuffers(n int32, framebuffers *uint32) {
	f.callCounts[idxDeleteFramebuffers]++
	glCall(f.deleteFramebuffers, uintptr(n), uintptr(unsafe.Pointer(framebuffers)), 0, 0, 0, 0, 0, 0, 0)
}

// DeleteProgram calls glDeleteProgram.
func (f *Functions) DeleteProgram(program uint32) {
	f.callCounts[idxDeleteProgram]++
	glCall(f.deleteProgram, uintptr(program), 0, 0, 0, 0, 0, 0, 0, 0)
}

// DeleteQueries calls glDeleteQueries.
func (f *Functions) DeleteQueries(n int32, ids *uint32) {
	f.callCounts[idxDeleteQueries]++
	glCall(f.deleteQueries, uintptr(n), uintptr(unsafe.Pointer(ids)), 0, 0, 0, 0, 0, 0, 0)
}

// DeleteRenderbuffers calls glDeleteRenderbuffers.
func (f *Functions) DeleteRenderbuffers(n int32, renderbuffers *uint32) {
	f.callCounts[idxDeleteRenderbuffers]++
	glCall(f.deleteRenderbuffers, uintptr(n), uintptr(unsafe.Pointer(renderbuffers)), 0, 0, 0, 0, 0, 0, 0)
}

// DeleteSamplers calls glDeleteSamplers.
func (f *Functions) DeleteSamplers(count int32, samplers *uint32) {
	f.callCounts[idxDeleteSamplers]++
	glCall(f.deleteSamplers, uintptr(count), uintptr(unsafe.Pointer(samplers)), 0, 0, 0, 0, 0, 0, 0)
}

// DeleteShader calls glDeleteShader.
func (f *Functions) DeleteShader(shader uint32) {
	f.callCounts[idxDeleteShader]++
	glCall(f.deleteShader, uintptr(shader), 0, 0, 0, 0, 0, 0, 0, 0)
}

// DeleteSync calls glDeleteSync.
func (f *Functions) DeleteSync(sync Sync) {
	f.callCounts[idxDeleteSync]++
	glCall(f.deleteSync, uintptr(sync), 0, 0, 0, 0, 0, 0, 0, 0)
}

// DeleteTextures calls glDeleteTextures.
func (f *Functions) DeleteTextures(n int32, textures *uint32) {
	f.callCounts[idxDeleteTextures]++
	glCall(f.deleteTextures, uintptr(n), uintptr(unsafe.Pointer(textures)), 0, 0, 0, 0, 0, 0, 0)
}

// DeleteVertexArrays calls glDeleteVertexArrays.
func (f *Functions) DeleteVertexArrays(n int32, arrays *uint32) {
	f.callCounts[idxDeleteVertexArrays]++
	glCall(f.deleteVertexArrays, uintptr(n), uintptr(unsafe.Pointer(arrays)), 0, 0, 0, 0, 0, 0, 0)
}

// DepthMask calls glDepthMask.
func (f *Functions) DepthMask(flag bool) {
	f.callCounts[idxDepthMask]++
	glCall(f.depthMask, glBool(flag), 0, 0, 0, 0, 0, 0, 0, 0)
}

// Disable calls glDisable.
func (f *Functions) Disable(capability uint32) {
	f.callCounts[idxDisable]++
	glCall(f.disable, uintptr(capability), 0, 0, 0, 0, 0, 0, 0, 0)
}

// DisableVertexAttribArray calls glDisableVertexAttribArray.
func (f *Functions) DisableVertexAttribArray(index uint32) {
	f.callCounts[idxDisableVertexAttribArray]++
	glCall(f.disableVertexAttribArray, uintptr(index), 0, 0, 0, 0, 0, 0, 0, 0)
}

// DiscardFramebuffer calls glDiscardFramebuffer.
func (f *Functions) DiscardFramebuffer(target uint32, numAttachments int32, attachments *uint32) {
	f.callCounts[idxDiscardFramebuffer]++
	glCall(f.discardFramebuffer, uintptr(target), uintptr(numAttachments), uintptr(unsafe.Pointer(attachments)),
		0, 0, 0, 0, 0, 0)
}

// DrawArrays calls glDrawArrays.
func (f *Functions) DrawArrays(mode uint32, first, count int32) {
	f.callCounts[idxDrawArrays]++
	glCall(f.drawArrays, uintptr(mode), uintptr(first), uintptr(count), 0, 0, 0, 0, 0, 0)
}

// DrawArraysIndirect calls glDrawArraysIndirect.
//
//go:uintptrescapes
func (f *Functions) DrawArraysIndirect(mode uint32, indirect uintptr) {
	f.callCounts[idxDrawArraysIndirect]++
	glCall(f.drawArraysIndirect, uintptr(mode), indirect, 0, 0, 0, 0, 0, 0, 0)
}

// DrawArraysInstanced calls glDrawArraysInstanced.
func (f *Functions) DrawArraysInstanced(mode uint32, first, count, primcount int32) {
	f.callCounts[idxDrawArraysInstanced]++
	glCall(f.drawArraysInstanced, uintptr(mode), uintptr(first), uintptr(count), uintptr(primcount), 0, 0, 0, 0, 0)
}

// DrawArraysInstancedBaseInstance calls glDrawArraysInstancedBaseInstance.
func (f *Functions) DrawArraysInstancedBaseInstance(mode uint32, first, count, instancecount int32, baseinstance uint32) {
	f.callCounts[idxDrawArraysInstancedBaseInstance]++
	glCall(f.drawArraysInstancedBaseInstance, uintptr(mode), uintptr(first), uintptr(count), uintptr(instancecount),
		uintptr(baseinstance), 0, 0, 0, 0)
}

// DrawBuffer calls glDrawBuffer.
func (f *Functions) DrawBuffer(mode uint32) {
	f.callCounts[idxDrawBuffer]++
	glCall(f.drawBuffer, uintptr(mode), 0, 0, 0, 0, 0, 0, 0, 0)
}

// DrawBuffers calls glDrawBuffers.
func (f *Functions) DrawBuffers(n int32, bufs *uint32) {
	f.callCounts[idxDrawBuffers]++
	glCall(f.drawBuffers, uintptr(n), uintptr(unsafe.Pointer(bufs)), 0, 0, 0, 0, 0, 0, 0)
}

// DrawElements calls glDrawElements.
//
//go:uintptrescapes
func (f *Functions) DrawElements(mode uint32, count int32, typ uint32, indices uintptr) {
	f.callCounts[idxDrawElements]++
	glCall(f.drawElements, uintptr(mode), uintptr(count), uintptr(typ), indices, 0, 0, 0, 0, 0)
}

// DrawElementsIndirect calls glDrawElementsIndirect.
//
//go:uintptrescapes
func (f *Functions) DrawElementsIndirect(mode, typ uint32, indirect uintptr) {
	f.callCounts[idxDrawElementsIndirect]++
	glCall(f.drawElementsIndirect, uintptr(mode), uintptr(typ), indirect, 0, 0, 0, 0, 0, 0)
}

// DrawElementsInstanced calls glDrawElementsInstanced.
//
//go:uintptrescapes
func (f *Functions) DrawElementsInstanced(mode uint32, count int32, typ uint32, indices uintptr, primcount int32) {
	f.callCounts[idxDrawElementsInstanced]++
	glCall(f.drawElementsInstanced, uintptr(mode), uintptr(count), uintptr(typ), indices, uintptr(primcount), 0, 0, 0, 0)
}

// DrawElementsInstancedBaseVertexBaseInstance calls glDrawElementsInstancedBaseVertexBaseInstance.
//
//go:uintptrescapes
func (f *Functions) DrawElementsInstancedBaseVertexBaseInstance(mode uint32, count int32, typ uint32, indices uintptr, instancecount, basevertex int32, baseinstance uint32) {
	f.callCounts[idxDrawElementsInstancedBaseVertexBaseInstance]++
	glCall(f.drawElementsInstancedBaseVertexBaseInstance, uintptr(mode), uintptr(count), uintptr(typ),
		indices, uintptr(instancecount), uintptr(basevertex), uintptr(baseinstance), 0, 0)
}

// DrawRangeElements calls glDrawRangeElements.
//
//go:uintptrescapes
func (f *Functions) DrawRangeElements(mode, start, end uint32, count int32, typ uint32, indices uintptr) {
	f.callCounts[idxDrawRangeElements]++
	glCall(f.drawRangeElements, uintptr(mode), uintptr(start), uintptr(end), uintptr(count), uintptr(typ),
		indices, 0, 0, 0)
}

// Enable calls glEnable.
func (f *Functions) Enable(capability uint32) {
	f.callCounts[idxEnable]++
	glCall(f.enable, uintptr(capability), 0, 0, 0, 0, 0, 0, 0, 0)
}

// EnableVertexAttribArray calls glEnableVertexAttribArray.
func (f *Functions) EnableVertexAttribArray(index uint32) {
	f.callCounts[idxEnableVertexAttribArray]++
	glCall(f.enableVertexAttribArray, uintptr(index), 0, 0, 0, 0, 0, 0, 0, 0)
}

// EndQuery calls glEndQuery.
func (f *Functions) EndQuery(target uint32) {
	f.callCounts[idxEndQuery]++
	glCall(f.endQuery, uintptr(target), 0, 0, 0, 0, 0, 0, 0, 0)
}

// EndTiling calls glEndTiling.
func (f *Functions) EndTiling(preserveMask uint32) {
	f.callCounts[idxEndTiling]++
	glCall(f.endTiling, uintptr(preserveMask), 0, 0, 0, 0, 0, 0, 0, 0)
}

// FenceSync calls glFenceSync.
func (f *Functions) FenceSync(condition, flags uint32) Sync {
	f.callCounts[idxFenceSync]++
	return Sync(glCall(f.fenceSync, uintptr(condition), uintptr(flags), 0, 0, 0, 0, 0, 0, 0))
}

// Finish calls glFinish.
func (f *Functions) Finish() {
	f.callCounts[idxFinish]++
	glCall(f.finish, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

// FinishFence calls glFinishFence.
func (f *Functions) FinishFence(fence uint32) {
	f.callCounts[idxFinishFence]++
	glCall(f.finishFence, uintptr(fence), 0, 0, 0, 0, 0, 0, 0, 0)
}

// Flush calls glFlush.
func (f *Functions) Flush() {
	f.callCounts[idxFlush]++
	glCall(f.flush, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

// FlushMappedBufferRange calls glFlushMappedBufferRange.
func (f *Functions) FlushMappedBufferRange(target uint32, offset, length int) {
	f.callCounts[idxFlushMappedBufferRange]++
	glCall(f.flushMappedBufferRange, uintptr(target), uintptr(offset), uintptr(length), 0, 0, 0, 0, 0, 0)
}

// FramebufferRenderbuffer calls glFramebufferRenderbuffer.
func (f *Functions) FramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer uint32) {
	f.callCounts[idxFramebufferRenderbuffer]++
	glCall(f.framebufferRenderbuffer, uintptr(target), uintptr(attachment), uintptr(renderbuffertarget),
		uintptr(renderbuffer), 0, 0, 0, 0, 0)
}

// FramebufferTexture2D calls glFramebufferTexture2D.
func (f *Functions) FramebufferTexture2D(target, attachment, textarget, texture uint32, level int32) {
	f.callCounts[idxFramebufferTexture2D]++
	glCall(f.framebufferTexture2D, uintptr(target), uintptr(attachment), uintptr(textarget), uintptr(texture),
		uintptr(level), 0, 0, 0, 0)
}

// FramebufferTexture2DMultisample calls glFramebufferTexture2DMultisample.
func (f *Functions) FramebufferTexture2DMultisample(target, attachment, textarget, texture uint32, level, samples int32) {
	f.callCounts[idxFramebufferTexture2DMultisample]++
	glCall(f.framebufferTexture2DMultisample, uintptr(target), uintptr(attachment), uintptr(textarget),
		uintptr(texture), uintptr(level), uintptr(samples), 0, 0, 0)
}

// FrontFace calls glFrontFace.
func (f *Functions) FrontFace(mode uint32) {
	f.callCounts[idxFrontFace]++
	glCall(f.frontFace, uintptr(mode), 0, 0, 0, 0, 0, 0, 0, 0)
}

// GenBuffers calls glGenBuffers.
func (f *Functions) GenBuffers(n int32, buffers *uint32) {
	f.callCounts[idxGenBuffers]++
	glCall(f.genBuffers, uintptr(n), uintptr(unsafe.Pointer(buffers)), 0, 0, 0, 0, 0, 0, 0)
}

// GenFences calls glGenFences.
func (f *Functions) GenFences(n int32, fences *uint32) {
	f.callCounts[idxGenFences]++
	glCall(f.genFences, uintptr(n), uintptr(unsafe.Pointer(fences)), 0, 0, 0, 0, 0, 0, 0)
}

// GenFramebuffers calls glGenFramebuffers.
func (f *Functions) GenFramebuffers(n int32, framebuffers *uint32) {
	f.callCounts[idxGenFramebuffers]++
	glCall(f.genFramebuffers, uintptr(n), uintptr(unsafe.Pointer(framebuffers)), 0, 0, 0, 0, 0, 0, 0)
}

// GenQueries calls glGenQueries.
func (f *Functions) GenQueries(n int32, ids *uint32) {
	f.callCounts[idxGenQueries]++
	glCall(f.genQueries, uintptr(n), uintptr(unsafe.Pointer(ids)), 0, 0, 0, 0, 0, 0, 0)
}

// GenRenderbuffers calls glGenRenderbuffers.
func (f *Functions) GenRenderbuffers(n int32, renderbuffers *uint32) {
	f.callCounts[idxGenRenderbuffers]++
	glCall(f.genRenderbuffers, uintptr(n), uintptr(unsafe.Pointer(renderbuffers)), 0, 0, 0, 0, 0, 0, 0)
}

// GenSamplers calls glGenSamplers.
func (f *Functions) GenSamplers(count int32, samplers *uint32) {
	f.callCounts[idxGenSamplers]++
	glCall(f.genSamplers, uintptr(count), uintptr(unsafe.Pointer(samplers)), 0, 0, 0, 0, 0, 0, 0)
}

// GenTextures calls glGenTextures.
func (f *Functions) GenTextures(n int32, textures *uint32) {
	f.callCounts[idxGenTextures]++
	glCall(f.genTextures, uintptr(n), uintptr(unsafe.Pointer(textures)), 0, 0, 0, 0, 0, 0, 0)
}

// GenVertexArrays calls glGenVertexArrays.
func (f *Functions) GenVertexArrays(n int32, arrays *uint32) {
	f.callCounts[idxGenVertexArrays]++
	glCall(f.genVertexArrays, uintptr(n), uintptr(unsafe.Pointer(arrays)), 0, 0, 0, 0, 0, 0, 0)
}

// GenerateMipmap calls glGenerateMipmap.
func (f *Functions) GenerateMipmap(target uint32) {
	f.callCounts[idxGenerateMipmap]++
	glCall(f.generateMipmap, uintptr(target), 0, 0, 0, 0, 0, 0, 0, 0)
}

// GetBufferParameteriv calls glGetBufferParameteriv.
func (f *Functions) GetBufferParameteriv(target, pname uint32, params *int32) {
	f.callCounts[idxGetBufferParameteriv]++
	glCall(f.getBufferParameteriv, uintptr(target), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// GetDebugMessageLog calls glGetDebugMessageLog.
func (f *Functions) GetDebugMessageLog(count uint32, bufSize int32, sources, types, ids, severities *uint32, lengths *int32, messageLog *byte) uint32 {
	f.callCounts[idxGetDebugMessageLog]++
	return uint32(glCall(f.getDebugMessageLog, uintptr(count), uintptr(bufSize), uintptr(unsafe.Pointer(sources)),
		uintptr(unsafe.Pointer(types)), uintptr(unsafe.Pointer(ids)), uintptr(unsafe.Pointer(severities)),
		uintptr(unsafe.Pointer(lengths)), uintptr(unsafe.Pointer(messageLog)), 0))
}

// GetError calls glGetError.
func (f *Functions) GetError() uint32 {
	f.callCounts[idxGetError]++
	return uint32(glCall(f.getError, 0, 0, 0, 0, 0, 0, 0, 0, 0))
}

// GetFloatv calls glGetFloatv.
func (f *Functions) GetFloatv(pname uint32, params *float32) {
	f.callCounts[idxGetFloatv]++
	glCall(f.getFloatv, uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0, 0)
}

// GetFramebufferAttachmentParameteriv calls glGetFramebufferAttachmentParameteriv.
func (f *Functions) GetFramebufferAttachmentParameteriv(target, attachment, pname uint32, params *int32) {
	f.callCounts[idxGetFramebufferAttachmentParameteriv]++
	glCall(f.getFramebufferAttachmentParameteriv, uintptr(target), uintptr(attachment), uintptr(pname),
		uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0)
}

// GetIntegerv calls glGetIntegerv.
func (f *Functions) GetIntegerv(pname uint32, params *int32) {
	f.callCounts[idxGetIntegerv]++
	glCall(f.getIntegerv, uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0, 0)
}

// GetInternalformativ calls glGetInternalformativ.
func (f *Functions) GetInternalformativ(target, internalformat, pname uint32, bufSize int32, params *int32) {
	f.callCounts[idxGetInternalformativ]++
	glCall(f.getInternalformativ, uintptr(target), uintptr(internalformat), uintptr(pname), uintptr(bufSize),
		uintptr(unsafe.Pointer(params)), 0, 0, 0, 0)
}

// GetMultisamplefv calls glGetMultisamplefv.
func (f *Functions) GetMultisamplefv(pname, index uint32, val *float32) {
	f.callCounts[idxGetMultisamplefv]++
	glCall(f.getMultisamplefv, uintptr(pname), uintptr(index), uintptr(unsafe.Pointer(val)), 0, 0, 0, 0, 0, 0)
}

// GetProgramBinary calls glGetProgramBinary.
//
//go:uintptrescapes
func (f *Functions) GetProgramBinary(program uint32, bufsize int32, length *int32, binaryFormat *uint32, binary uintptr) {
	f.callCounts[idxGetProgramBinary]++
	glCall(f.getProgramBinary, uintptr(program), uintptr(bufsize), uintptr(unsafe.Pointer(length)),
		uintptr(unsafe.Pointer(binaryFormat)), binary, 0, 0, 0, 0)
}

// GetProgramInfoLog calls glGetProgramInfoLog.
func (f *Functions) GetProgramInfoLog(program uint32, bufsize int32, length *int32, infolog *byte) {
	f.callCounts[idxGetProgramInfoLog]++
	glCall(f.getProgramInfoLog, uintptr(program), uintptr(bufsize), uintptr(unsafe.Pointer(length)),
		uintptr(unsafe.Pointer(infolog)), 0, 0, 0, 0, 0)
}

// GetProgramiv calls glGetProgramiv.
func (f *Functions) GetProgramiv(program, pname uint32, params *int32) {
	f.callCounts[idxGetProgramiv]++
	glCall(f.getProgramiv, uintptr(program), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// GetQueryObjecti64v calls glGetQueryObjecti64v.
func (f *Functions) GetQueryObjecti64v(id, pname uint32, params *int64) {
	f.callCounts[idxGetQueryObjecti64v]++
	glCall(f.getQueryObjecti64v, uintptr(id), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// GetQueryObjectui64v calls glGetQueryObjectui64v.
func (f *Functions) GetQueryObjectui64v(id, pname uint32, params *uint64) {
	f.callCounts[idxGetQueryObjectui64v]++
	glCall(f.getQueryObjectui64v, uintptr(id), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// GetQueryObjectuiv calls glGetQueryObjectuiv.
func (f *Functions) GetQueryObjectuiv(id, pname uint32, params *uint32) {
	f.callCounts[idxGetQueryObjectuiv]++
	glCall(f.getQueryObjectuiv, uintptr(id), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// GetQueryiv calls glGetQueryiv.
func (f *Functions) GetQueryiv(target, pname uint32, params *int32) {
	f.callCounts[idxGetQueryiv]++
	glCall(f.getQueryiv, uintptr(target), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// GetRenderbufferParameteriv calls glGetRenderbufferParameteriv.
func (f *Functions) GetRenderbufferParameteriv(target, pname uint32, params *int32) {
	f.callCounts[idxGetRenderbufferParameteriv]++
	glCall(f.getRenderbufferParameteriv, uintptr(target), uintptr(pname), uintptr(unsafe.Pointer(params)),
		0, 0, 0, 0, 0, 0)
}

// GetShaderInfoLog calls glGetShaderInfoLog.
func (f *Functions) GetShaderInfoLog(shader uint32, bufsize int32, length *int32, infolog *byte) {
	f.callCounts[idxGetShaderInfoLog]++
	glCall(f.getShaderInfoLog, uintptr(shader), uintptr(bufsize), uintptr(unsafe.Pointer(length)),
		uintptr(unsafe.Pointer(infolog)), 0, 0, 0, 0, 0)
}

// GetShaderPrecisionFormat calls glGetShaderPrecisionFormat.
func (f *Functions) GetShaderPrecisionFormat(shadertype, precisiontype uint32, rng, precision *int32) {
	f.callCounts[idxGetShaderPrecisionFormat]++
	glCall(f.getShaderPrecisionFormat, uintptr(shadertype), uintptr(precisiontype), uintptr(unsafe.Pointer(rng)),
		uintptr(unsafe.Pointer(precision)), 0, 0, 0, 0, 0)
}

// GetShaderiv calls glGetShaderiv.
func (f *Functions) GetShaderiv(shader, pname uint32, params *int32) {
	f.callCounts[idxGetShaderiv]++
	glCall(f.getShaderiv, uintptr(shader), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// GetString calls glGetString.
func (f *Functions) GetString(name uint32) uintptr {
	f.callCounts[idxGetString]++
	return glCall(f.getString, uintptr(name), 0, 0, 0, 0, 0, 0, 0, 0)
}

// GetStringi calls glGetStringi.
func (f *Functions) GetStringi(name, index uint32) uintptr {
	f.callCounts[idxGetStringi]++
	return glCall(f.getStringi, uintptr(name), uintptr(index), 0, 0, 0, 0, 0, 0, 0)
}

// GetTexLevelParameteriv calls glGetTexLevelParameteriv.
func (f *Functions) GetTexLevelParameteriv(target uint32, level int32, pname uint32, params *int32) {
	f.callCounts[idxGetTexLevelParameteriv]++
	glCall(f.getTexLevelParameteriv, uintptr(target), uintptr(level), uintptr(pname), uintptr(unsafe.Pointer(params)),
		0, 0, 0, 0, 0)
}

// GetUniformLocation calls glGetUniformLocation.
func (f *Functions) GetUniformLocation(program uint32, name *byte) int32 {
	f.callCounts[idxGetUniformLocation]++
	return int32(glCall(f.getUniformLocation, uintptr(program), uintptr(unsafe.Pointer(name)), 0, 0, 0, 0, 0, 0, 0))
}

// InsertEventMarker calls glInsertEventMarker.
func (f *Functions) InsertEventMarker(length int32, marker *byte) {
	f.callCounts[idxInsertEventMarker]++
	glCall(f.insertEventMarker, uintptr(length), uintptr(unsafe.Pointer(marker)), 0, 0, 0, 0, 0, 0, 0)
}

// InvalidateBufferData calls glInvalidateBufferData.
func (f *Functions) InvalidateBufferData(buffer uint32) {
	f.callCounts[idxInvalidateBufferData]++
	glCall(f.invalidateBufferData, uintptr(buffer), 0, 0, 0, 0, 0, 0, 0, 0)
}

// InvalidateBufferSubData calls glInvalidateBufferSubData.
func (f *Functions) InvalidateBufferSubData(buffer uint32, offset, length int) {
	f.callCounts[idxInvalidateBufferSubData]++
	glCall(f.invalidateBufferSubData, uintptr(buffer), uintptr(offset), uintptr(length), 0, 0, 0, 0, 0, 0)
}

// InvalidateFramebuffer calls glInvalidateFramebuffer.
func (f *Functions) InvalidateFramebuffer(target uint32, numAttachments int32, attachments *uint32) {
	f.callCounts[idxInvalidateFramebuffer]++
	glCall(f.invalidateFramebuffer, uintptr(target), uintptr(numAttachments), uintptr(unsafe.Pointer(attachments)),
		0, 0, 0, 0, 0, 0)
}

// InvalidateSubFramebuffer calls glInvalidateSubFramebuffer.
func (f *Functions) InvalidateSubFramebuffer(target uint32, numAttachments int32, attachments *uint32, x, y, width, height int32) {
	f.callCounts[idxInvalidateSubFramebuffer]++
	glCall(f.invalidateSubFramebuffer, uintptr(target), uintptr(numAttachments), uintptr(unsafe.Pointer(attachments)),
		uintptr(x), uintptr(y), uintptr(width), uintptr(height), 0, 0)
}

// InvalidateTexImage calls glInvalidateTexImage.
func (f *Functions) InvalidateTexImage(texture uint32, level int32) {
	f.callCounts[idxInvalidateTexImage]++
	glCall(f.invalidateTexImage, uintptr(texture), uintptr(level), 0, 0, 0, 0, 0, 0, 0)
}

// InvalidateTexSubImage calls glInvalidateTexSubImage.
func (f *Functions) InvalidateTexSubImage(texture uint32, level, xoffset, yoffset, zoffset, width, height, depth int32) {
	f.callCounts[idxInvalidateTexSubImage]++
	glCall(f.invalidateTexSubImage, uintptr(texture), uintptr(level), uintptr(xoffset), uintptr(yoffset),
		uintptr(zoffset), uintptr(width), uintptr(height), uintptr(depth), 0)
}

// IsSync calls glIsSync.
func (f *Functions) IsSync(sync Sync) bool {
	f.callCounts[idxIsSync]++
	return byte(glCall(f.isSync, uintptr(sync), 0, 0, 0, 0, 0, 0, 0, 0)) != 0
}

// IsTexture calls glIsTexture.
func (f *Functions) IsTexture(texture uint32) bool {
	f.callCounts[idxIsTexture]++
	return byte(glCall(f.isTexture, uintptr(texture), 0, 0, 0, 0, 0, 0, 0, 0)) != 0
}

// LineWidth calls glLineWidth.
func (f *Functions) LineWidth(width float32) {
	f.callCounts[idxLineWidth]++
	f.fnLineWidth(width)
}

// LinkProgram calls glLinkProgram.
func (f *Functions) LinkProgram(program uint32) {
	f.callCounts[idxLinkProgram]++
	glCall(f.linkProgram, uintptr(program), 0, 0, 0, 0, 0, 0, 0, 0)
}

// MapBuffer calls glMapBuffer.
func (f *Functions) MapBuffer(target, access uint32) unsafe.Pointer {
	f.callCounts[idxMapBuffer]++
	return ptrFromUintptr(glCall(f.mapBuffer, uintptr(target), uintptr(access), 0, 0, 0, 0, 0, 0, 0))
}

// MapBufferRange calls glMapBufferRange.
func (f *Functions) MapBufferRange(target uint32, offset, length int, access uint32) unsafe.Pointer {
	f.callCounts[idxMapBufferRange]++
	return ptrFromUintptr(glCall(f.mapBufferRange, uintptr(target), uintptr(offset), uintptr(length),
		uintptr(access), 0, 0, 0, 0, 0))
}

// MapBufferSubData calls glMapBufferSubData.
func (f *Functions) MapBufferSubData(target uint32, offset, size int, access uint32) unsafe.Pointer {
	f.callCounts[idxMapBufferSubData]++
	return ptrFromUintptr(glCall(f.mapBufferSubData, uintptr(target), uintptr(offset), uintptr(size),
		uintptr(access), 0, 0, 0, 0, 0))
}

// MapTexSubImage2D calls glMapTexSubImage2D.
func (f *Functions) MapTexSubImage2D(target uint32, level, xoffset, yoffset, width, height int32, format, typ, access uint32) unsafe.Pointer {
	f.callCounts[idxMapTexSubImage2D]++
	return ptrFromUintptr(glCall(f.mapTexSubImage2D, uintptr(target), uintptr(level), uintptr(xoffset),
		uintptr(yoffset), uintptr(width), uintptr(height), uintptr(format), uintptr(typ), uintptr(access)))
}

// MemoryBarrier calls glMemoryBarrier.
func (f *Functions) MemoryBarrier(barriers uint32) {
	f.callCounts[idxMemoryBarrier]++
	glCall(f.memoryBarrier, uintptr(barriers), 0, 0, 0, 0, 0, 0, 0, 0)
}

// MultiDrawArraysIndirect calls glMultiDrawArraysIndirect.
//
//go:uintptrescapes
func (f *Functions) MultiDrawArraysIndirect(mode uint32, indirect uintptr, drawcount, stride int32) {
	f.callCounts[idxMultiDrawArraysIndirect]++
	glCall(f.multiDrawArraysIndirect, uintptr(mode), indirect, uintptr(drawcount), uintptr(stride),
		0, 0, 0, 0, 0)
}

// MultiDrawArraysInstancedBaseInstance calls glMultiDrawArraysInstancedBaseInstance.
func (f *Functions) MultiDrawArraysInstancedBaseInstance(mode uint32, firsts, counts, instanceCounts *int32, baseInstances *uint32, drawcount int32) {
	f.callCounts[idxMultiDrawArraysInstancedBaseInstance]++
	glCall(f.multiDrawArraysInstancedBaseInstance, uintptr(mode), uintptr(unsafe.Pointer(firsts)),
		uintptr(unsafe.Pointer(counts)), uintptr(unsafe.Pointer(instanceCounts)),
		uintptr(unsafe.Pointer(baseInstances)), uintptr(drawcount), 0, 0, 0)
}

// MultiDrawElementsIndirect calls glMultiDrawElementsIndirect.
//
//go:uintptrescapes
func (f *Functions) MultiDrawElementsIndirect(mode, typ uint32, indirect uintptr, drawcount, stride int32) {
	f.callCounts[idxMultiDrawElementsIndirect]++
	glCall(f.multiDrawElementsIndirect, uintptr(mode), uintptr(typ), indirect, uintptr(drawcount),
		uintptr(stride), 0, 0, 0, 0)
}

// MultiDrawElementsInstancedBaseVertexBaseInstance calls glMultiDrawElementsInstancedBaseVertexBaseInstance.
//
//go:uintptrescapes
func (f *Functions) MultiDrawElementsInstancedBaseVertexBaseInstance(mode uint32, counts *int32, typ uint32, indices uintptr, instanceCounts, baseVertices *int32, baseInstances *uint32, drawcount int32) {
	f.callCounts[idxMultiDrawElementsInstancedBaseVertexBaseInstance]++
	glCall(f.multiDrawElementsInstancedBaseVertexBaseInstance, uintptr(mode), uintptr(unsafe.Pointer(counts)),
		uintptr(typ), indices, uintptr(unsafe.Pointer(instanceCounts)),
		uintptr(unsafe.Pointer(baseVertices)), uintptr(unsafe.Pointer(baseInstances)), uintptr(drawcount), 0)
}

// ObjectLabel calls glObjectLabel.
func (f *Functions) ObjectLabel(identifier, name uint32, length int32, label *byte) {
	f.callCounts[idxObjectLabel]++
	glCall(f.objectLabel, uintptr(identifier), uintptr(name), uintptr(length), uintptr(unsafe.Pointer(label)),
		0, 0, 0, 0, 0)
}

// PatchParameteri calls glPatchParameteri.
func (f *Functions) PatchParameteri(pname uint32, value int32) {
	f.callCounts[idxPatchParameteri]++
	glCall(f.patchParameteri, uintptr(pname), uintptr(value), 0, 0, 0, 0, 0, 0, 0)
}

// PixelStorei calls glPixelStorei.
func (f *Functions) PixelStorei(pname uint32, param int32) {
	f.callCounts[idxPixelStorei]++
	glCall(f.pixelStorei, uintptr(pname), uintptr(param), 0, 0, 0, 0, 0, 0, 0)
}

// PolygonMode calls glPolygonMode.
func (f *Functions) PolygonMode(face, mode uint32) {
	f.callCounts[idxPolygonMode]++
	glCall(f.polygonMode, uintptr(face), uintptr(mode), 0, 0, 0, 0, 0, 0, 0)
}

// PopDebugGroup calls glPopDebugGroup.
func (f *Functions) PopDebugGroup() {
	f.callCounts[idxPopDebugGroup]++
	glCall(f.popDebugGroup, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

// PopGroupMarker calls glPopGroupMarker.
func (f *Functions) PopGroupMarker() {
	f.callCounts[idxPopGroupMarker]++
	glCall(f.popGroupMarker, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

// ProgramBinary calls glProgramBinary.
//
//go:uintptrescapes
func (f *Functions) ProgramBinary(program, binaryFormat uint32, binary uintptr, length int32) {
	f.callCounts[idxProgramBinary]++
	glCall(f.programBinary, uintptr(program), uintptr(binaryFormat), binary, uintptr(length), 0, 0, 0, 0, 0)
}

// ProgramParameteri calls glProgramParameteri.
func (f *Functions) ProgramParameteri(program, pname uint32, value int32) {
	f.callCounts[idxProgramParameteri]++
	glCall(f.programParameteri, uintptr(program), uintptr(pname), uintptr(value), 0, 0, 0, 0, 0, 0)
}

// PushDebugGroup calls glPushDebugGroup.
func (f *Functions) PushDebugGroup(source, id uint32, length int32, message *byte) {
	f.callCounts[idxPushDebugGroup]++
	glCall(f.pushDebugGroup, uintptr(source), uintptr(id), uintptr(length), uintptr(unsafe.Pointer(message)),
		0, 0, 0, 0, 0)
}

// PushGroupMarker calls glPushGroupMarker.
func (f *Functions) PushGroupMarker(length int32, marker *byte) {
	f.callCounts[idxPushGroupMarker]++
	glCall(f.pushGroupMarker, uintptr(length), uintptr(unsafe.Pointer(marker)), 0, 0, 0, 0, 0, 0, 0)
}

// QueryCounter calls glQueryCounter.
func (f *Functions) QueryCounter(id, target uint32) {
	f.callCounts[idxQueryCounter]++
	glCall(f.queryCounter, uintptr(id), uintptr(target), 0, 0, 0, 0, 0, 0, 0)
}

// ReadBuffer calls glReadBuffer.
func (f *Functions) ReadBuffer(src uint32) {
	f.callCounts[idxReadBuffer]++
	glCall(f.readBuffer, uintptr(src), 0, 0, 0, 0, 0, 0, 0, 0)
}

// ReadPixels calls glReadPixels.
//
//go:uintptrescapes
func (f *Functions) ReadPixels(x, y, width, height int32, format, typ uint32, pixels uintptr) {
	f.callCounts[idxReadPixels]++
	glCall(f.readPixels, uintptr(x), uintptr(y), uintptr(width), uintptr(height), uintptr(format), uintptr(typ),
		pixels, 0, 0)
}

// RenderbufferStorage calls glRenderbufferStorage.
func (f *Functions) RenderbufferStorage(target, internalformat uint32, width, height int32) {
	f.callCounts[idxRenderbufferStorage]++
	glCall(f.renderbufferStorage, uintptr(target), uintptr(internalformat), uintptr(width), uintptr(height),
		0, 0, 0, 0, 0)
}

// RenderbufferStorageMultisample calls glRenderbufferStorageMultisample.
func (f *Functions) RenderbufferStorageMultisample(target uint32, samples int32, internalformat uint32, width, height int32) {
	f.callCounts[idxRenderbufferStorageMultisample]++
	glCall(f.renderbufferStorageMultisample, uintptr(target), uintptr(samples), uintptr(internalformat),
		uintptr(width), uintptr(height), 0, 0, 0, 0)
}

// ResolveMultisampleFramebuffer calls glResolveMultisampleFramebuffer.
func (f *Functions) ResolveMultisampleFramebuffer() {
	f.callCounts[idxResolveMultisampleFramebuffer]++
	glCall(f.resolveMultisampleFramebuffer, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

// SamplerParameterf calls glSamplerParameterf.
func (f *Functions) SamplerParameterf(sampler, pname uint32, param float32) {
	f.callCounts[idxSamplerParameterf]++
	f.fnSamplerParameterf(sampler, pname, param)
}

// SamplerParameteri calls glSamplerParameteri.
func (f *Functions) SamplerParameteri(sampler, pname uint32, param int32) {
	f.callCounts[idxSamplerParameteri]++
	glCall(f.samplerParameteri, uintptr(sampler), uintptr(pname), uintptr(param), 0, 0, 0, 0, 0, 0)
}

// SamplerParameteriv calls glSamplerParameteriv.
func (f *Functions) SamplerParameteriv(sampler, pname uint32, params *int32) {
	f.callCounts[idxSamplerParameteriv]++
	glCall(f.samplerParameteriv, uintptr(sampler), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// Scissor calls glScissor.
func (f *Functions) Scissor(x, y, width, height int32) {
	f.callCounts[idxScissor]++
	glCall(f.scissor, uintptr(x), uintptr(y), uintptr(width), uintptr(height), 0, 0, 0, 0, 0)
}

// SetFence calls glSetFence.
func (f *Functions) SetFence(fence, condition uint32) {
	f.callCounts[idxSetFence]++
	glCall(f.setFence, uintptr(fence), uintptr(condition), 0, 0, 0, 0, 0, 0, 0)
}

// ShaderSource calls glShaderSource.
//
//go:uintptrescapes
func (f *Functions) ShaderSource(shader uint32, count int32, str uintptr, length *int32) {
	f.callCounts[idxShaderSource]++
	glCall(f.shaderSource, uintptr(shader), uintptr(count), str, uintptr(unsafe.Pointer(length)),
		0, 0, 0, 0, 0)
}

// StartTiling calls glStartTiling.
func (f *Functions) StartTiling(x, y, width, height, preserveMask uint32) {
	f.callCounts[idxStartTiling]++
	glCall(f.startTiling, uintptr(x), uintptr(y), uintptr(width), uintptr(height), uintptr(preserveMask), 0, 0, 0, 0)
}

// StencilFunc calls glStencilFunc.
func (f *Functions) StencilFunc(fn uint32, ref int32, mask uint32) {
	f.callCounts[idxStencilFunc]++
	glCall(f.stencilFunc, uintptr(fn), uintptr(ref), uintptr(mask), 0, 0, 0, 0, 0, 0)
}

// StencilFuncSeparate calls glStencilFuncSeparate.
func (f *Functions) StencilFuncSeparate(face, fn uint32, ref int32, mask uint32) {
	f.callCounts[idxStencilFuncSeparate]++
	glCall(f.stencilFuncSeparate, uintptr(face), uintptr(fn), uintptr(ref), uintptr(mask), 0, 0, 0, 0, 0)
}

// StencilMask calls glStencilMask.
func (f *Functions) StencilMask(mask uint32) {
	f.callCounts[idxStencilMask]++
	glCall(f.stencilMask, uintptr(mask), 0, 0, 0, 0, 0, 0, 0, 0)
}

// StencilMaskSeparate calls glStencilMaskSeparate.
func (f *Functions) StencilMaskSeparate(face, mask uint32) {
	f.callCounts[idxStencilMaskSeparate]++
	glCall(f.stencilMaskSeparate, uintptr(face), uintptr(mask), 0, 0, 0, 0, 0, 0, 0)
}

// StencilOp calls glStencilOp.
func (f *Functions) StencilOp(fail, zfail, zpass uint32) {
	f.callCounts[idxStencilOp]++
	glCall(f.stencilOp, uintptr(fail), uintptr(zfail), uintptr(zpass), 0, 0, 0, 0, 0, 0)
}

// StencilOpSeparate calls glStencilOpSeparate.
func (f *Functions) StencilOpSeparate(face, fail, zfail, zpass uint32) {
	f.callCounts[idxStencilOpSeparate]++
	glCall(f.stencilOpSeparate, uintptr(face), uintptr(fail), uintptr(zfail), uintptr(zpass), 0, 0, 0, 0, 0)
}

// TestFence calls glTestFence.
func (f *Functions) TestFence(fence uint32) bool {
	f.callCounts[idxTestFence]++
	return byte(glCall(f.testFence, uintptr(fence), 0, 0, 0, 0, 0, 0, 0, 0)) != 0
}

// TexBuffer calls glTexBuffer.
func (f *Functions) TexBuffer(target, internalformat, buffer uint32) {
	f.callCounts[idxTexBuffer]++
	glCall(f.texBuffer, uintptr(target), uintptr(internalformat), uintptr(buffer), 0, 0, 0, 0, 0, 0)
}

// TexBufferRange calls glTexBufferRange.
func (f *Functions) TexBufferRange(target, internalformat, buffer uint32, offset, size int) {
	f.callCounts[idxTexBufferRange]++
	glCall(f.texBufferRange, uintptr(target), uintptr(internalformat), uintptr(buffer), uintptr(offset), uintptr(size),
		0, 0, 0, 0)
}

// TexImage2D calls glTexImage2D.
//
//go:uintptrescapes
func (f *Functions) TexImage2D(target uint32, level, internalformat, width, height, border int32, format, typ uint32, pixels uintptr) {
	f.callCounts[idxTexImage2D]++
	glCall(f.texImage2D, uintptr(target), uintptr(level), uintptr(internalformat), uintptr(width), uintptr(height),
		uintptr(border), uintptr(format), uintptr(typ), pixels)
}

// TexParameterf calls glTexParameterf.
func (f *Functions) TexParameterf(target, pname uint32, param float32) {
	f.callCounts[idxTexParameterf]++
	f.fnTexParameterf(target, pname, param)
}

// TexParameterfv calls glTexParameterfv.
func (f *Functions) TexParameterfv(target, pname uint32, params *float32) {
	f.callCounts[idxTexParameterfv]++
	glCall(f.texParameterfv, uintptr(target), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// TexParameteri calls glTexParameteri.
func (f *Functions) TexParameteri(target, pname uint32, param int32) {
	f.callCounts[idxTexParameteri]++
	glCall(f.texParameteri, uintptr(target), uintptr(pname), uintptr(param), 0, 0, 0, 0, 0, 0)
}

// TexParameteriv calls glTexParameteriv.
func (f *Functions) TexParameteriv(target, pname uint32, params *int32) {
	f.callCounts[idxTexParameteriv]++
	glCall(f.texParameteriv, uintptr(target), uintptr(pname), uintptr(unsafe.Pointer(params)), 0, 0, 0, 0, 0, 0)
}

// TexStorage2D calls glTexStorage2D.
func (f *Functions) TexStorage2D(target uint32, levels int32, internalformat uint32, width, height int32) {
	f.callCounts[idxTexStorage2D]++
	glCall(f.texStorage2D, uintptr(target), uintptr(levels), uintptr(internalformat), uintptr(width), uintptr(height),
		0, 0, 0, 0)
}

// TexSubImage2D calls glTexSubImage2D.
//
//go:uintptrescapes
func (f *Functions) TexSubImage2D(target uint32, level, xoffset, yoffset, width, height int32, format, typ uint32, pixels uintptr) {
	f.callCounts[idxTexSubImage2D]++
	glCall(f.texSubImage2D, uintptr(target), uintptr(level), uintptr(xoffset), uintptr(yoffset), uintptr(width),
		uintptr(height), uintptr(format), uintptr(typ), pixels)
}

// TextureBarrier calls glTextureBarrier.
func (f *Functions) TextureBarrier() {
	f.callCounts[idxTextureBarrier]++
	glCall(f.textureBarrier, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

// Uniform1f calls glUniform1f.
func (f *Functions) Uniform1f(location int32, v0 float32) {
	f.callCounts[idxUniform1f]++
	f.fnUniform1f(location, v0)
}

// Uniform1fv calls glUniform1fv.
func (f *Functions) Uniform1fv(location, count int32, v *float32) {
	f.callCounts[idxUniform1fv]++
	glCall(f.uniform1fv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// Uniform1i calls glUniform1i.
func (f *Functions) Uniform1i(location, v0 int32) {
	f.callCounts[idxUniform1i]++
	glCall(f.uniform1i, uintptr(location), uintptr(v0), 0, 0, 0, 0, 0, 0, 0)
}

// Uniform1iv calls glUniform1iv.
func (f *Functions) Uniform1iv(location, count int32, v *int32) {
	f.callCounts[idxUniform1iv]++
	glCall(f.uniform1iv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// Uniform2f calls glUniform2f.
func (f *Functions) Uniform2f(location int32, v0, v1 float32) {
	f.callCounts[idxUniform2f]++
	f.fnUniform2f(location, v0, v1)
}

// Uniform2fv calls glUniform2fv.
func (f *Functions) Uniform2fv(location, count int32, v *float32) {
	f.callCounts[idxUniform2fv]++
	glCall(f.uniform2fv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// Uniform2i calls glUniform2i.
func (f *Functions) Uniform2i(location, v0, v1 int32) {
	f.callCounts[idxUniform2i]++
	glCall(f.uniform2i, uintptr(location), uintptr(v0), uintptr(v1), 0, 0, 0, 0, 0, 0)
}

// Uniform2iv calls glUniform2iv.
func (f *Functions) Uniform2iv(location, count int32, v *int32) {
	f.callCounts[idxUniform2iv]++
	glCall(f.uniform2iv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// Uniform3f calls glUniform3f.
func (f *Functions) Uniform3f(location int32, v0, v1, v2 float32) {
	f.callCounts[idxUniform3f]++
	f.fnUniform3f(location, v0, v1, v2)
}

// Uniform3fv calls glUniform3fv.
func (f *Functions) Uniform3fv(location, count int32, v *float32) {
	f.callCounts[idxUniform3fv]++
	glCall(f.uniform3fv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// Uniform3i calls glUniform3i.
func (f *Functions) Uniform3i(location, v0, v1, v2 int32) {
	f.callCounts[idxUniform3i]++
	glCall(f.uniform3i, uintptr(location), uintptr(v0), uintptr(v1), uintptr(v2), 0, 0, 0, 0, 0)
}

// Uniform3iv calls glUniform3iv.
func (f *Functions) Uniform3iv(location, count int32, v *int32) {
	f.callCounts[idxUniform3iv]++
	glCall(f.uniform3iv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// Uniform4f calls glUniform4f.
func (f *Functions) Uniform4f(location int32, v0, v1, v2, v3 float32) {
	f.callCounts[idxUniform4f]++
	f.fnUniform4f(location, v0, v1, v2, v3)
}

// Uniform4fv calls glUniform4fv.
func (f *Functions) Uniform4fv(location, count int32, v *float32) {
	f.callCounts[idxUniform4fv]++
	glCall(f.uniform4fv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// Uniform4i calls glUniform4i.
func (f *Functions) Uniform4i(location, v0, v1, v2, v3 int32) {
	f.callCounts[idxUniform4i]++
	glCall(f.uniform4i, uintptr(location), uintptr(v0), uintptr(v1), uintptr(v2), uintptr(v3), 0, 0, 0, 0)
}

// Uniform4iv calls glUniform4iv.
func (f *Functions) Uniform4iv(location, count int32, v *int32) {
	f.callCounts[idxUniform4iv]++
	glCall(f.uniform4iv, uintptr(location), uintptr(count), uintptr(unsafe.Pointer(v)), 0, 0, 0, 0, 0, 0)
}

// UniformMatrix2fv calls glUniformMatrix2fv.
func (f *Functions) UniformMatrix2fv(location, count int32, transpose bool, value *float32) {
	f.callCounts[idxUniformMatrix2fv]++
	glCall(f.uniformMatrix2fv, uintptr(location), uintptr(count), glBool(transpose), uintptr(unsafe.Pointer(value)),
		0, 0, 0, 0, 0)
}

// UniformMatrix3fv calls glUniformMatrix3fv.
func (f *Functions) UniformMatrix3fv(location, count int32, transpose bool, value *float32) {
	f.callCounts[idxUniformMatrix3fv]++
	glCall(f.uniformMatrix3fv, uintptr(location), uintptr(count), glBool(transpose), uintptr(unsafe.Pointer(value)),
		0, 0, 0, 0, 0)
}

// UniformMatrix4fv calls glUniformMatrix4fv.
func (f *Functions) UniformMatrix4fv(location, count int32, transpose bool, value *float32) {
	f.callCounts[idxUniformMatrix4fv]++
	glCall(f.uniformMatrix4fv, uintptr(location), uintptr(count), glBool(transpose), uintptr(unsafe.Pointer(value)),
		0, 0, 0, 0, 0)
}

// UnmapBuffer calls glUnmapBuffer.
func (f *Functions) UnmapBuffer(target uint32) bool {
	f.callCounts[idxUnmapBuffer]++
	return byte(glCall(f.unmapBuffer, uintptr(target), 0, 0, 0, 0, 0, 0, 0, 0)) != 0
}

// UnmapBufferSubData calls glUnmapBufferSubData.
//
//go:uintptrescapes
func (f *Functions) UnmapBufferSubData(mem uintptr) {
	f.callCounts[idxUnmapBufferSubData]++
	glCall(f.unmapBufferSubData, mem, 0, 0, 0, 0, 0, 0, 0, 0)
}

// UnmapTexSubImage2D calls glUnmapTexSubImage2D.
//
//go:uintptrescapes
func (f *Functions) UnmapTexSubImage2D(mem uintptr) {
	f.callCounts[idxUnmapTexSubImage2D]++
	glCall(f.unmapTexSubImage2D, mem, 0, 0, 0, 0, 0, 0, 0, 0)
}

// UseProgram calls glUseProgram.
func (f *Functions) UseProgram(program uint32) {
	f.callCounts[idxUseProgram]++
	glCall(f.useProgram, uintptr(program), 0, 0, 0, 0, 0, 0, 0, 0)
}

// VertexAttrib1f calls glVertexAttrib1f.
func (f *Functions) VertexAttrib1f(indx uint32, value float32) {
	f.callCounts[idxVertexAttrib1f]++
	f.fnVertexAttrib1f(indx, value)
}

// VertexAttrib2fv calls glVertexAttrib2fv.
func (f *Functions) VertexAttrib2fv(indx uint32, values *float32) {
	f.callCounts[idxVertexAttrib2fv]++
	glCall(f.vertexAttrib2fv, uintptr(indx), uintptr(unsafe.Pointer(values)), 0, 0, 0, 0, 0, 0, 0)
}

// VertexAttrib3fv calls glVertexAttrib3fv.
func (f *Functions) VertexAttrib3fv(indx uint32, values *float32) {
	f.callCounts[idxVertexAttrib3fv]++
	glCall(f.vertexAttrib3fv, uintptr(indx), uintptr(unsafe.Pointer(values)), 0, 0, 0, 0, 0, 0, 0)
}

// VertexAttrib4fv calls glVertexAttrib4fv.
func (f *Functions) VertexAttrib4fv(indx uint32, values *float32) {
	f.callCounts[idxVertexAttrib4fv]++
	glCall(f.vertexAttrib4fv, uintptr(indx), uintptr(unsafe.Pointer(values)), 0, 0, 0, 0, 0, 0, 0)
}

// VertexAttribDivisor calls glVertexAttribDivisor.
func (f *Functions) VertexAttribDivisor(index, divisor uint32) {
	f.callCounts[idxVertexAttribDivisor]++
	glCall(f.vertexAttribDivisor, uintptr(index), uintptr(divisor), 0, 0, 0, 0, 0, 0, 0)
}

// VertexAttribIPointer calls glVertexAttribIPointer.
//
//go:uintptrescapes
func (f *Functions) VertexAttribIPointer(indx uint32, size int32, typ uint32, stride int32, ptr uintptr) {
	f.callCounts[idxVertexAttribIPointer]++
	glCall(f.vertexAttribIPointer, uintptr(indx), uintptr(size), uintptr(typ), uintptr(stride), ptr,
		0, 0, 0, 0)
}

// VertexAttribPointer calls glVertexAttribPointer.
//
//go:uintptrescapes
func (f *Functions) VertexAttribPointer(indx uint32, size int32, typ uint32, normalized bool, stride int32, ptr uintptr) {
	f.callCounts[idxVertexAttribPointer]++
	glCall(f.vertexAttribPointer, uintptr(indx), uintptr(size), uintptr(typ), glBool(normalized),
		uintptr(stride), ptr, 0, 0, 0)
}

// Viewport calls glViewport.
func (f *Functions) Viewport(x, y, width, height int32) {
	f.callCounts[idxViewport]++
	glCall(f.viewport, uintptr(x), uintptr(y), uintptr(width), uintptr(height), 0, 0, 0, 0, 0)
}

// WaitSync calls glWaitSync.
func (f *Functions) WaitSync(sync Sync, flags uint32, timeout uint64) {
	f.callCounts[idxWaitSync]++
	glCall(f.waitSync, uintptr(sync), uintptr(flags), uintptr(timeout), 0, 0, 0, 0, 0, 0)
}

// WindowRectangles calls glWindowRectangles.
func (f *Functions) WindowRectangles(mode uint32, count, box int32) {
	f.callCounts[idxWindowRectangles]++
	glCall(f.windowRectangles, uintptr(mode), uintptr(count), uintptr(box), 0, 0, 0, 0, 0, 0)
}
