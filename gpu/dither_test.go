// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gpu

import "testing"

// TestDitherRangeForColorType pins the per-channel dither range of every low-precision color type. A type that falls
// through to the zero default builds a dither effect that does nothing, which is indistinguishable from no dithering
// at all in the output but still costs a fragment processor.
func TestDitherRangeForColorType(t *testing.T) {
	for _, c := range []struct {
		ct   ColorType
		want float32
	}{
		// 4 bit.
		{ct: ColorTypeABGR4444, want: 1 / 15.0},
		{ct: ColorTypeARGB4444, want: 1 / 15.0},
		{ct: ColorTypeBGRA4444, want: 1 / 15.0},
		// 6 bit.
		{ct: ColorTypeBGR565, want: 1 / 63.0},
		{ct: ColorTypeRGB565, want: 1 / 63.0},
		// 8 bit.
		{ct: ColorTypeAlpha8, want: 1 / 255.0},
		{ct: ColorTypeGray8, want: 1 / 255.0},
		{ct: ColorTypeGrayAlpha88, want: 1 / 255.0},
		{ct: ColorTypeR8, want: 1 / 255.0},
		{ct: ColorTypeRG88, want: 1 / 255.0},
		{ct: ColorTypeRGB888, want: 1 / 255.0},
		{ct: ColorTypeRGB888x, want: 1 / 255.0},
		{ct: ColorTypeRGBA8888, want: 1 / 255.0},
		{ct: ColorTypeRGBA8888SRGB, want: 1 / 255.0},
		{ct: ColorTypeBGRA8888, want: 1 / 255.0},
		// 10 bit.
		{ct: ColorTypeRGBA1010102, want: 1 / 1023.0},
		{ct: ColorTypeBGRA1010102, want: 1 / 1023.0},
		{ct: ColorTypeRGB101010x, want: 1 / 1023.0},
		{ct: ColorTypeRGBA10x6, want: 1 / 1023.0},
		// 16 bit unorm: upstream's value, not the 1/65535 the formula implies.
		{ct: ColorTypeAlpha16, want: 1 / 32767.0},
		{ct: ColorTypeR16, want: 1 / 32767.0},
		{ct: ColorTypeRG1616, want: 1 / 32767.0},
		{ct: ColorTypeRGBA16161616, want: 1 / 32767.0},
		// Float and unknown formats need no dithering.
		{ct: ColorTypeUnknown, want: 0},
		{ct: ColorTypeAlphaF16, want: 0},
		{ct: ColorTypeRGBAF16, want: 0},
		{ct: ColorTypeRGBAF16Clamped, want: 0},
		{ct: ColorTypeRGBAF32, want: 0},
		{ct: ColorTypeRGF16, want: 0},
		{ct: ColorTypeGrayF16, want: 0},
	} {
		if got := DitherRangeForColorType(c.ct); got != c.want {
			t.Errorf("DitherRangeForColorType(%v) = %v, want %v", c.ct, got, c.want)
		}
	}
}

// TestDitherRangeCoversRenderableUnormTypes pins that no unorm color type of 16 bits per channel or less silently
// falls through to the no-dithering default: such a type would pass the shouldDitherPaint gate and then get a dither
// effect with a zero range, i.e. a no-op fragment processor.
func TestDitherRangeCoversRenderableUnormTypes(t *testing.T) {
	float16Or32 := map[ColorType]bool{
		ColorTypeAlphaF16: true, ColorTypeRGBAF16: true, ColorTypeRGBF16F16F16x: true,
		ColorTypeRGBAF16Clamped: true, ColorTypeRGBAF32: true, ColorTypeRGF16: true,
		ColorTypeRF16: true, ColorTypeGrayF16: true,
	}
	// The xxx types are data-only reinterpretations that never back a draw target.
	dataOnly := map[ColorType]bool{
		ColorTypeAlpha8xxx: true, ColorTypeAlphaF32xxx: true, ColorTypeGray8xxx: true,
		ColorTypeR8xxx: true,
	}
	for ct := ColorTypeAlpha8; int(ct) < ColorTypeCount; ct++ {
		if float16Or32[ct] || dataOnly[ct] {
			continue
		}
		if DitherRangeForColorType(ct) == 0 {
			t.Errorf("color type %v has no dither range", ct)
		}
	}
}
