// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The dither range per color type and the precomputed 8x8 ordered-dither lookup table the GPU dither effect samples per
// fragment (the same bit-mix the CPU dither stage computes on the fly).

package gpu

// DitherLUTSize is the dither table's side length: 8x8 (if changed, also change the value in the dither effect's
// shader).
const DitherLUTSize = 8

// DitherRangeForColorType returns 1/(2^bitdepth - 1) per channel for the given color type, or 0 (no dithering) for the
// float and unknown formats. The 16-bit unorm lane follows upstream's 1/32767 rather than the 1/65535 the formula
// implies.
func DitherRangeForColorType(ct ColorType) float32 {
	switch ct {
	case ColorTypeABGR4444, ColorTypeARGB4444, ColorTypeBGRA4444:
		return 1 / 15.0
	case ColorTypeBGR565, ColorTypeRGB565:
		return 1 / 63.0
	case ColorTypeAlpha8, ColorTypeGray8, ColorTypeGrayAlpha88, ColorTypeR8, ColorTypeRG88,
		ColorTypeRGB888, ColorTypeRGB888x, ColorTypeRGBA8888, ColorTypeRGBA8888SRGB,
		ColorTypeBGRA8888:
		return 1 / 255.0
	case ColorTypeRGBA1010102, ColorTypeBGRA1010102, ColorTypeRGB101010x, ColorTypeRGBA10x6:
		return 1 / 1023.0
	case ColorTypeAlpha16, ColorTypeR16, ColorTypeRG1616, ColorTypeRGBA16161616:
		return 1 / 32767.0
	default:
		return 0 // no dithering
	}
}

// MakeDitherLUT builds the 8x8 A8 dither table, row-major.
func MakeDitherLUT() [DitherLUTSize * DitherLUTSize]byte {
	var data [DitherLUTSize * DitherLUTSize]byte
	for x := 0; x < DitherLUTSize; x++ {
		for y := 0; y < DitherLUTSize; y++ {
			// The computation of 'm' and 'value' is lifted from the CPU backend.
			m := uint32(y&1)<<5 | uint32(x&1)<<4 |
				uint32(y&2)<<2 | uint32(x&2)<<1 |
				uint32(y&4)>>1 | uint32(x&4)>>2
			value := float32(m)*(1.0/64.0) - 63.0/128.0
			// Bias by 0.5 to be in 0..1, mul by 255 and round to nearest int to make a byte.
			data[y*DitherLUTSize+x] = byte((value+0.5)*255.0 + 0.5)
		}
	}
	return data
}
