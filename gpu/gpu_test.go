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

func TestSwizzleKeys(t *testing.T) {
	// The Swizzle bit encoding is load-bearing (program keys depend on it): "rgba" is 0x3210, one nibble per output
	// channel, output r in the low nibble.
	if SwizzleRGBA != 0x3210 {
		t.Errorf("SwizzleRGBA = %#x, want 0x3210", uint16(SwizzleRGBA))
	}
	if got := MakeSwizzle("bgra"); got != 0x3012 {
		t.Errorf("bgra = %#x, want 0x3012", uint16(got))
	}
	if got := MakeSwizzle("000r"); got != 0x0444 {
		t.Errorf("000r = %#x, want 0x0444", uint16(got))
	}
	for _, s := range []string{"rgba", "bgra", "000r", "a000", "rrr1", "rgb1", "aaa0", "rrra"} {
		if got := MakeSwizzle(s).String(); got != s {
			t.Errorf("round trip %q = %q", s, got)
		}
	}
}

func TestColorTypeDescriptors(t *testing.T) {
	cases := []struct {
		ct    ColorType
		flags ColorChannelFlags
		bpp   int
	}{
		{ct: ColorTypeRGBA8888, flags: RGBAChannelFlags, bpp: 4},
		{ct: ColorTypeBGRA8888, flags: RGBAChannelFlags, bpp: 4},
		{ct: ColorTypeRGB888x, flags: RGBChannelFlags, bpp: 4},
		{ct: ColorTypeAlpha8, flags: AlphaChannelFlag, bpp: 1},
		{ct: ColorTypeGray8, flags: GrayChannelFlag, bpp: 1},
		{ct: ColorTypeBGR565, flags: RGBChannelFlags, bpp: 2},
		{ct: ColorTypeRGBAF16, flags: RGBAChannelFlags, bpp: 8},
		{ct: ColorTypeAlphaF16, flags: AlphaChannelFlag, bpp: 2},
		{ct: ColorTypeRGB888, flags: RGBChannelFlags, bpp: 3},
		{ct: ColorTypeRGBAF32, flags: RGBAChannelFlags, bpp: 16},
		{ct: ColorTypeAlpha8xxx, flags: AlphaChannelFlag, bpp: 4},
		{ct: ColorTypeUnknown, flags: 0, bpp: 0},
	}
	for _, c := range cases {
		if got := c.ct.ChannelFlags(); got != c.flags {
			t.Errorf("ChannelFlags(%d) = %#x, want %#x", c.ct, got, c.flags)
		}
		if got := c.ct.BytesPerPixel(); got != c.bpp {
			t.Errorf("BytesPerPixel(%d) = %d, want %d", c.ct, got, c.bpp)
		}
	}
}

func TestShaderCapsDefaults(t *testing.T) {
	sc := NewShaderCaps()
	if !sc.FloatIs32Bits || !sc.BuiltinFMASupport || !sc.BuiltinDeterminantSupport ||
		!sc.CanUseMinAndAbsTogether || !sc.CanUseFragCoord || !sc.VectorClampMinMaxSupport {
		t.Error("ShaderCaps defaults do not match the C++ member initializers")
	}
	if sc.GLSLGeneration != GLSL330 {
		t.Errorf("default GLSL generation = %d, want GLSL330", sc.GLSLGeneration)
	}
	if sc.MustDeclareFragmentShaderOutput() != true {
		t.Error("GLSL330 must declare fragment shader output")
	}
}

func TestCapsBaseDefaults(t *testing.T) {
	options := DefaultContextOptions()
	c := MakeCaps(options)
	if !c.ReuseScratchTextures || !c.ReuseScratchBuffers || !c.MustSyncGpuDuringAbandon ||
		!c.PreferVRAMUseOverFlushes || !c.ClampToBorderSupport {
		t.Error("Caps defaults do not match the Caps constructor")
	}
	if c.MaxRenderTargetSize != 1 || c.MaxTextureSize != 1 {
		t.Error("size defaults wrong")
	}
	if c.BufferMapThreshold != -1 {
		t.Errorf("buffer map threshold = %d, want -1", c.BufferMapThreshold)
	}
	if c.AdvancedBlendEquationSupport() {
		t.Error("blend equation support should default to basic")
	}
	c.BlendEquationSupport = AdvancedBlendEquationSupport
	c.AdvBlendEqDisableFlags = 1 << uint32(BlendEquationColorBurn)
	if !c.IsAdvancedBlendEquationDisabled(BlendEquationColorBurn) ||
		c.IsAdvancedBlendEquationDisabled(BlendEquationColorDodge) {
		t.Error("advanced blend disable flags wrong")
	}
}
