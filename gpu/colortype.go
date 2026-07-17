// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ColorType and its descriptor helpers. The enum's ordering is fixed because caps tables index arrays by color type.

package gpu

// ColorChannelFlags is a bit set of the color channels a color type carries.
type ColorChannelFlags uint32

// Channel flag bits.
const (
	RedChannelFlag   ColorChannelFlags = 1 << 0
	GreenChannelFlag ColorChannelFlags = 1 << 1
	BlueChannelFlag  ColorChannelFlags = 1 << 2
	AlphaChannelFlag ColorChannelFlags = 1 << 3
	GrayChannelFlag  ColorChannelFlags = 1 << 4

	GrayAlphaChannelFlags ColorChannelFlags = GrayChannelFlag | AlphaChannelFlag
	RGChannelFlags        ColorChannelFlags = RedChannelFlag | GreenChannelFlag
	RGBChannelFlags       ColorChannelFlags = RGChannelFlags | BlueChannelFlag
	RGBAChannelFlags      ColorChannelFlags = RGBChannelFlags | AlphaChannelFlag
)

// ColorType is the GPU layer's superset of the public color types, including data-only and texture-init-only types. The
// ordering below is fixed since caps tables index arrays by it.
type ColorType int32

// ColorType values.
const (
	ColorTypeUnknown ColorType = iota
	ColorTypeAlpha8
	ColorTypeBGR565 // Distinct from ColorTypeRGB565: same bit depths, reversed channel order.
	ColorTypeRGB565
	ColorTypeABGR4444 // Distinct from ColorTypeARGB4444 further below: reversed channel order.
	ColorTypeRGBA8888
	ColorTypeRGBA8888SRGB
	ColorTypeRGB888x
	ColorTypeRG88
	ColorTypeBGRA8888
	ColorTypeRGBA1010102
	ColorTypeBGRA1010102
	ColorTypeRGB101010x
	ColorTypeRGBA10x6
	ColorTypeGray8
	ColorTypeGrayAlpha88
	ColorTypeAlphaF16
	ColorTypeRGBAF16
	ColorTypeRGBF16F16F16x
	ColorTypeRGBAF16Clamped
	ColorTypeRGBAF32

	ColorTypeAlpha16
	ColorTypeRG1616
	ColorTypeRGF16
	ColorTypeRGBA16161616

	// Unusual types that come up after reading back in cases where we are reassigning the meaning of a texture format's
	// channels to use for a particular color format but have to read back the data to a full RGBA quadruple. (e.g.
	// using a R8 texture format as A8 color type but the API only supports reading to RGBA8.)
	ColorTypeAlpha8xxx
	ColorTypeAlphaF32xxx
	ColorTypeGray8xxx
	ColorTypeR8xxx

	// Types used to initialize backend textures.
	ColorTypeRGB888
	ColorTypeR8
	ColorTypeR16
	ColorTypeRF16
	ColorTypeGrayF16
	ColorTypeBGRA4444
	ColorTypeARGB4444

	// ColorTypeCount is the number of valid ColorType values.
	ColorTypeCount = int(ColorTypeARGB4444) + 1
)

// ChannelFlags returns the set of color channels this color type carries.
func (ct ColorType) ChannelFlags() ColorChannelFlags {
	switch ct {
	case ColorTypeAlpha8, ColorTypeAlphaF16, ColorTypeAlpha16, ColorTypeAlpha8xxx, ColorTypeAlphaF32xxx:
		return AlphaChannelFlag
	case ColorTypeBGR565, ColorTypeRGB565, ColorTypeRGB888x, ColorTypeRGB101010x, ColorTypeRGBF16F16F16x,
		ColorTypeRGB888:
		return RGBChannelFlags
	case ColorTypeABGR4444, ColorTypeRGBA8888, ColorTypeRGBA8888SRGB, ColorTypeBGRA8888,
		ColorTypeRGBA1010102, ColorTypeBGRA1010102, ColorTypeRGBA10x6, ColorTypeRGBAF16,
		ColorTypeRGBAF16Clamped, ColorTypeRGBAF32, ColorTypeRGBA16161616, ColorTypeBGRA4444,
		ColorTypeARGB4444:
		return RGBAChannelFlags
	case ColorTypeRG88, ColorTypeRG1616, ColorTypeRGF16:
		return RGChannelFlags
	case ColorTypeGray8, ColorTypeGray8xxx, ColorTypeGrayF16:
		return GrayChannelFlag
	case ColorTypeGrayAlpha88:
		return GrayAlphaChannelFlags
	case ColorTypeR8xxx, ColorTypeR8, ColorTypeR16, ColorTypeRF16:
		return RedChannelFlag
	default:
		return 0
	}
}

// BytesPerPixel returns the storage size in bytes of one pixel of this color type.
func (ct ColorType) BytesPerPixel() int {
	switch ct {
	case ColorTypeAlpha8, ColorTypeGray8, ColorTypeR8:
		return 1
	case ColorTypeBGR565, ColorTypeRGB565, ColorTypeABGR4444, ColorTypeRG88, ColorTypeGrayAlpha88,
		ColorTypeAlphaF16, ColorTypeAlpha16, ColorTypeR16, ColorTypeRF16, ColorTypeGrayF16,
		ColorTypeBGRA4444, ColorTypeARGB4444:
		return 2
	case ColorTypeRGB888:
		return 3
	case ColorTypeRGBA8888, ColorTypeRGBA8888SRGB, ColorTypeRGB888x, ColorTypeBGRA8888,
		ColorTypeRGBA1010102, ColorTypeBGRA1010102, ColorTypeRGB101010x, ColorTypeRG1616, ColorTypeRGF16,
		ColorTypeAlpha8xxx, ColorTypeGray8xxx, ColorTypeR8xxx:
		return 4
	case ColorTypeRGBA10x6, ColorTypeRGBAF16, ColorTypeRGBF16F16F16x, ColorTypeRGBAF16Clamped,
		ColorTypeRGBA16161616:
		return 8
	case ColorTypeRGBAF32, ColorTypeAlphaF32xxx:
		return 16
	default:
		return 0
	}
}
