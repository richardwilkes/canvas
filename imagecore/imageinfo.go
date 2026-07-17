// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imagecore

// ColorType identifies how pixel bytes encode color and alpha channels. Only the supported color-type matrix is
// implemented (see ColorType.Supported); the rest of the enum exists so values round-trip faithfully through the public
// surface.
type ColorType int32

// ColorType values.
const (
	ColorTypeUnknown ColorType = iota
	ColorTypeAlpha8
	ColorTypeRGB565
	ColorTypeARGB4444
	ColorTypeRGBA8888
	ColorTypeRGB888x
	ColorTypeBGRA8888
	ColorTypeRGBA1010102
	ColorTypeBGRA1010102
	ColorTypeRGB101010x
	ColorTypeBGR101010x
	ColorTypeBGR101010xXR
	ColorTypeBGRA10101010XR
	ColorTypeRGBA10x6
	ColorTypeGray8
	ColorTypeRGBAF16Norm
	ColorTypeRGBAF16
	ColorTypeRGBF16F16F16x
	ColorTypeRGBAF32
	ColorTypeR8G8Unorm
	ColorTypeA16Float
	ColorTypeR16G16Float
	ColorTypeA16Unorm
	ColorTypeR16G16Unorm
	ColorTypeR16G16B16A16Unorm
	ColorTypeSRGBA8888
	ColorTypeR8Unorm
)

// ColorTypeN32 is the platform-native 32-bit color type: RGBA_8888 on every target. Note this is RGBA on every
// platform, unlike Skia, whose N32 is BGRA_8888 where the platform prefers it.
const ColorTypeN32 = ColorTypeRGBA8888

// BytesPerPixel returns the storage size of one pixel of ct (0 for unknown/unsupported types).
func (ct ColorType) BytesPerPixel() int {
	switch ct {
	case ColorTypeAlpha8, ColorTypeGray8, ColorTypeR8Unorm:
		return 1
	case ColorTypeRGB565, ColorTypeARGB4444, ColorTypeA16Unorm, ColorTypeA16Float:
		return 2
	case ColorTypeRGBA8888, ColorTypeRGB888x, ColorTypeBGRA8888, ColorTypeSRGBA8888,
		ColorTypeRGBA1010102, ColorTypeBGRA1010102, ColorTypeRGB101010x, ColorTypeBGR101010x,
		ColorTypeBGR101010xXR, ColorTypeR8G8Unorm, ColorTypeR16G16Unorm, ColorTypeR16G16Float:
		return 4
	case ColorTypeRGBAF16Norm, ColorTypeRGBAF16, ColorTypeRGBF16F16F16x, ColorTypeRGBA10x6,
		ColorTypeR16G16B16A16Unorm, ColorTypeBGRA10101010XR:
		return 8
	case ColorTypeRGBAF32:
		return 16
	default:
		return 0
	}
}

// Supported reports whether the port implements pixel storage and conversion for ct — the supported color-type matrix:
// N32 (RGBA/BGRA 8888), A8, Gray8, RGB565, RGBA_F16 (convert-only), 888x. This is the canonical definition of that
// matrix; everything else that speaks of it is bounded by this set.
func (ct ColorType) Supported() bool {
	switch ct {
	case ColorTypeAlpha8, ColorTypeRGB565, ColorTypeRGBA8888, ColorTypeRGB888x, ColorTypeBGRA8888,
		ColorTypeGray8, ColorTypeRGBAF16:
		return true
	default:
		return false
	}
}

// isAlphaOnly reports whether ct carries only an alpha channel, for the supported set.
func (ct ColorType) isAlphaOnly() bool {
	return ct == ColorTypeAlpha8 || ct == ColorTypeA16Unorm || ct == ColorTypeA16Float
}

// isAlwaysOpaque reports whether ct has no alpha channel, for the supported set.
func (ct ColorType) isAlwaysOpaque() bool {
	switch ct {
	case ColorTypeRGB565, ColorTypeRGB888x, ColorTypeGray8, ColorTypeRGB101010x,
		ColorTypeBGR101010x, ColorTypeBGR101010xXR, ColorTypeRGBF16F16F16x, ColorTypeR8G8Unorm,
		ColorTypeR16G16Unorm, ColorTypeR16G16Float, ColorTypeR8Unorm:
		return true
	default:
		return false
	}
}

// AlphaType identifies how a pixel's alpha channel relates to its color channels (opaque, premultiplied, or
// unpremultiplied).
type AlphaType int32

// AlphaType values.
const (
	AlphaTypeUnknown AlphaType = iota
	AlphaTypeOpaque
	AlphaTypePremul
	AlphaTypeUnpremul
)

// ValidateAlphaType canonicalizes at for ct (alpha-only types treat unpremul as premul; opaque color types force
// AlphaTypeOpaque) and rejects AlphaTypeUnknown for alpha-carrying types. ok=false means the (ct, at) pair cannot form
// a valid image info.
func ValidateAlphaType(ct ColorType, at AlphaType) (canonical AlphaType, ok bool) {
	switch {
	case ct == ColorTypeUnknown:
		return AlphaTypeUnknown, true
	case ct.isAlphaOnly():
		if at == AlphaTypeUnpremul {
			at = AlphaTypePremul
		}
		if at == AlphaTypeUnknown {
			return 0, false
		}
		return at, true
	case ct.isAlwaysOpaque():
		return AlphaTypeOpaque, true
	default:
		if at == AlphaTypeUnknown {
			return 0, false
		}
		return at, true
	}
}

// ImageInfo describes a pixel buffer's dimensions and pixel encoding.
type ImageInfo struct {
	Width     int32
	Height    int32
	ColorType ColorType
	AlphaType AlphaType
}

// MakeInfo returns an ImageInfo, canonicalizing the alpha type per ValidateAlphaType (raw values from the public
// surface funnel through here). ok=false for the invalid (ct, at) combinations ValidateAlphaType rejects.
func MakeInfo(width, height int32, ct ColorType, at AlphaType) (ImageInfo, bool) {
	canonical, ok := ValidateAlphaType(ct, at)
	if !ok {
		return ImageInfo{}, false
	}
	return ImageInfo{Width: width, Height: height, ColorType: ct, AlphaType: canonical}, true
}

// MakeN32Premul returns an N32-premul ImageInfo of the given dimensions.
func MakeN32Premul(width, height int32) ImageInfo {
	return ImageInfo{Width: width, Height: height, ColorType: ColorTypeN32, AlphaType: AlphaTypePremul}
}

// MinRowBytes returns the minimum row stride, in bytes, needed to hold one row of pixels.
func (info *ImageInfo) MinRowBytes() int {
	return int(info.Width) * info.ColorType.BytesPerPixel()
}

// IsOpaque reports whether the info's alpha type is always fully opaque.
func (info *ImageInfo) IsOpaque() bool { return info.AlphaType == AlphaTypeOpaque }

// IsEmpty reports whether either dimension is non-positive.
func (info *ImageInfo) IsEmpty() bool { return info.Width <= 0 || info.Height <= 0 }

// maxDimension is the largest width or height a valid ImageInfo may report (2^31 >> 2, leaving headroom so byte offsets
// and row-stride math stay within a signed 32-bit range).
const maxDimension = int32(0x7FFFFFFF >> 2)

// IsValid reports whether the info has positive, in-range dimensions and known color/alpha types.
func (info *ImageInfo) IsValid() bool {
	if info.Width <= 0 || info.Height <= 0 {
		return false
	}
	if info.Width > maxDimension || info.Height > maxDimension {
		return false
	}
	return info.ColorType != ColorTypeUnknown && info.AlphaType != AlphaTypeUnknown
}

// ValidConversion reports whether a pixel conversion between dst and src is permitted: both infos merely need to be
// valid.
func ValidConversion(dst, src *ImageInfo) bool {
	return dst.IsValid() && src.IsValid()
}
