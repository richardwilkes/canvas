// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Format identifies the sized GL texture/renderbuffer formats the GPU backend distinguishes, plus helpers for
// converting to/from GL enums and querying per-format properties. The full enum is kept (caps tables index by it) even
// though the desktop trim only ever populates capability entries for the supported color-type matrix (enumerated in
// caps.go).

package gl

import "github.com/richardwilkes/canvas/gpu"

// Format identifies a sized GL texture/renderbuffer format.
type Format int32

// Format values, in a fixed enum order used to index the caps tables.
const (
	FormatUnknown Format = iota

	FormatRGBA8
	FormatR8
	FormatALPHA8
	FormatLUMINANCE8
	FormatLUMINANCE8ALPHA8
	FormatBGRA8
	FormatRGB565
	FormatRGBA16F
	FormatR16F
	FormatRGB8
	FormatRGBX8
	FormatRG8
	FormatRGB10A2
	FormatRGBA4
	FormatSRGB8ALPHA8
	FormatCompressedETC1RGB8
	FormatCompressedRGB8ETC2
	FormatCompressedRGB8BC1
	FormatCompressedRGBA8BC1
	FormatR16
	FormatRG16
	FormatRGBA16
	FormatRG16F
	FormatLUMINANCE16F

	FormatLastColorFormat = FormatLUMINANCE16F

	// Depth/stencil formats.
	FormatSTENCILINDEX8 = iota - 1 // continue the sequence after the alias constant above
	FormatSTENCILINDEX16
	FormatDEPTH24STENCIL8
)

// ColorFormatCount is the number of Format values usable as color formats.
const ColorFormatCount = int(FormatLastColorFormat) + 1

// FormatFromEnum returns the Format corresponding to a sized GL internal-format enum, or FormatUnknown if the enum
// isn't recognized.
func FormatFromEnum(glFormat uint32) Format {
	switch glFormat {
	case RGBA8:
		return FormatRGBA8
	case R8:
		return FormatR8
	case ALPHA8:
		return FormatALPHA8
	case LUMINANCE8:
		return FormatLUMINANCE8
	case LUMINANCE8_ALPHA8:
		return FormatLUMINANCE8ALPHA8
	case BGRA8:
		return FormatBGRA8
	case RGB565:
		return FormatRGB565
	case RGBA16F:
		return FormatRGBA16F
	case LUMINANCE16F:
		return FormatLUMINANCE16F
	case R16F:
		return FormatR16F
	case RGB8:
		return FormatRGB8
	case RGBX8:
		return FormatRGBX8
	case RG8:
		return FormatRG8
	case RGB10_A2:
		return FormatRGB10A2
	case RGBA4:
		return FormatRGBA4
	case SRGB8_ALPHA8:
		return FormatSRGB8ALPHA8
	case COMPRESSED_ETC1_RGB8:
		return FormatCompressedETC1RGB8
	case COMPRESSED_RGB8_ETC2:
		return FormatCompressedRGB8ETC2
	case COMPRESSED_RGB_S3TC_DXT1_EXT:
		return FormatCompressedRGB8BC1
	case COMPRESSED_RGBA_S3TC_DXT1_EXT:
		return FormatCompressedRGBA8BC1
	case R16:
		return FormatR16
	case RG16:
		return FormatRG16
	case RGBA16:
		return FormatRGBA16
	case RG16F:
		return FormatRG16F
	case STENCIL_INDEX8:
		return FormatSTENCILINDEX8
	case STENCIL_INDEX16:
		return FormatSTENCILINDEX16
	case DEPTH24_STENCIL8:
		return FormatDEPTH24STENCIL8
	default:
		return FormatUnknown
	}
}

// ToEnum returns the sized internal format (or compressed internal format) GL enum for the Format.
func (f Format) ToEnum() uint32 {
	switch f {
	case FormatRGBA8:
		return RGBA8
	case FormatR8:
		return R8
	case FormatALPHA8:
		return ALPHA8
	case FormatLUMINANCE8:
		return LUMINANCE8
	case FormatLUMINANCE8ALPHA8:
		return LUMINANCE8_ALPHA8
	case FormatBGRA8:
		return BGRA8
	case FormatRGB565:
		return RGB565
	case FormatRGBA16F:
		return RGBA16F
	case FormatLUMINANCE16F:
		return LUMINANCE16F
	case FormatR16F:
		return R16F
	case FormatRGB8:
		return RGB8
	case FormatRGBX8:
		return RGBX8
	case FormatRG8:
		return RG8
	case FormatRGB10A2:
		return RGB10_A2
	case FormatRGBA4:
		return RGBA4
	case FormatSRGB8ALPHA8:
		return SRGB8_ALPHA8
	case FormatCompressedETC1RGB8:
		return COMPRESSED_ETC1_RGB8
	case FormatCompressedRGB8ETC2:
		return COMPRESSED_RGB8_ETC2
	case FormatCompressedRGB8BC1:
		return COMPRESSED_RGB_S3TC_DXT1_EXT
	case FormatCompressedRGBA8BC1:
		return COMPRESSED_RGBA_S3TC_DXT1_EXT
	case FormatR16:
		return R16
	case FormatRG16:
		return RG16
	case FormatRGBA16:
		return RGBA16
	case FormatRG16F:
		return RG16F
	case FormatSTENCILINDEX8:
		return STENCIL_INDEX8
	case FormatSTENCILINDEX16:
		return STENCIL_INDEX16
	case FormatDEPTH24STENCIL8:
		return DEPTH24_STENCIL8
	default:
		return 0
	}
}

// ChannelFlags returns which color channels this format carries.
func (f Format) ChannelFlags() gpu.ColorChannelFlags {
	switch f {
	case FormatRGBA8, FormatBGRA8, FormatRGBA16F, FormatRGB10A2, FormatRGBA4, FormatSRGB8ALPHA8,
		FormatCompressedRGBA8BC1, FormatRGBA16:
		return gpu.RGBAChannelFlags
	case FormatR8, FormatR16F, FormatR16:
		return gpu.RedChannelFlag
	case FormatALPHA8:
		return gpu.AlphaChannelFlag
	case FormatLUMINANCE8, FormatLUMINANCE16F:
		return gpu.GrayChannelFlag
	case FormatLUMINANCE8ALPHA8:
		return gpu.GrayAlphaChannelFlags
	case FormatRGB565, FormatRGB8, FormatRGBX8, FormatCompressedETC1RGB8, FormatCompressedRGB8ETC2,
		FormatCompressedRGB8BC1:
		return gpu.RGBChannelFlags
	case FormatRG8, FormatRG16, FormatRG16F:
		return gpu.RGChannelFlags
	default:
		return 0
	}
}

// BytesPerBlock returns the size in bytes of one block (one pixel for uncompressed formats).
func (f Format) BytesPerBlock() int {
	switch f {
	case FormatRGBA8, FormatBGRA8, FormatRGB8, FormatRGBX8, FormatRGB10A2, FormatSRGB8ALPHA8,
		FormatRG16, FormatRG16F, FormatDEPTH24STENCIL8:
		// GL_RGB8 is assumed to be stored 4-byte aligned on the GPU.
		return 4
	case FormatR8, FormatALPHA8, FormatLUMINANCE8, FormatSTENCILINDEX8:
		return 1
	case FormatLUMINANCE8ALPHA8, FormatRGB565, FormatLUMINANCE16F, FormatR16F, FormatRG8, FormatRGBA4,
		FormatR16, FormatSTENCILINDEX16:
		return 2
	case FormatRGBA16F, FormatRGBA16, FormatCompressedETC1RGB8, FormatCompressedRGB8ETC2,
		FormatCompressedRGB8BC1, FormatCompressedRGBA8BC1:
		return 8
	default:
		return 0
	}
}

// StencilBits returns the number of stencil bits this format provides, or 0 if none.
func (f Format) StencilBits() int {
	switch f {
	case FormatSTENCILINDEX8, FormatDEPTH24STENCIL8:
		return 8
	case FormatSTENCILINDEX16:
		return 16
	default:
		return 0
	}
}

// IsPackedDepthStencil reports whether this format packs depth and stencil into one block.
func (f Format) IsPackedDepthStencil() bool { return f == FormatDEPTH24STENCIL8 }

// IsSRGB reports whether this format is an sRGB-encoded color format.
func (f Format) IsSRGB() bool { return f == FormatSRGB8ALPHA8 }

// IsCompressed reports whether this is a block-compressed texture format.
func (f Format) IsCompressed() bool {
	switch f {
	case FormatCompressedETC1RGB8, FormatCompressedRGB8ETC2, FormatCompressedRGB8BC1,
		FormatCompressedRGBA8BC1:
		return true
	default:
		return false
	}
}

// String returns a short debug name for the format.
func (f Format) String() string {
	switch f {
	case FormatRGBA8:
		return "RGBA8"
	case FormatR8:
		return "R8"
	case FormatALPHA8:
		return "ALPHA8"
	case FormatLUMINANCE8:
		return "LUMINANCE8"
	case FormatLUMINANCE8ALPHA8:
		return "LUMINANCE8_ALPHA8"
	case FormatBGRA8:
		return "BGRA8"
	case FormatRGB565:
		return "RGB565"
	case FormatRGBA16F:
		return "RGBA16F"
	case FormatLUMINANCE16F:
		return "LUMINANCE16F"
	case FormatR16F:
		return "R16F"
	case FormatRGB8:
		return "RGB8"
	case FormatRGBX8:
		return "RGBX8"
	case FormatRG8:
		return "RG8"
	case FormatRGB10A2:
		return "RGB10_A2"
	case FormatRGBA4:
		return "RGBA4"
	case FormatSRGB8ALPHA8:
		return "SRGB8_ALPHA8"
	case FormatCompressedETC1RGB8:
		return "ETC1"
	case FormatCompressedRGB8ETC2:
		return "ETC2"
	case FormatCompressedRGB8BC1:
		return "RGB8_BC1"
	case FormatCompressedRGBA8BC1:
		return "RGBA8_BC1"
	case FormatR16:
		return "R16"
	case FormatRG16:
		return "RG16"
	case FormatRGBA16:
		return "RGBA16"
	case FormatRG16F:
		return "RG16F"
	case FormatSTENCILINDEX8:
		return "STENCIL_INDEX8"
	case FormatSTENCILINDEX16:
		return "STENCIL_INDEX16"
	case FormatDEPTH24STENCIL8:
		return "DEPTH24_STENCIL8"
	default:
		return "Unknown"
	}
}
