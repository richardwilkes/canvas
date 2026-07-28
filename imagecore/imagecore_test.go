// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imagecore

import (
	"math"
	"testing"
)

func TestHalfRoundTrip(t *testing.T) {
	// Every finite half value must round-trip half → float → half exactly.
	for h := 0; h < 0x10000; h++ {
		f := halfToFloat(uint16(h))
		if em := h & 0x7FFF; em >= 0x7C00 {
			continue // inf/NaN payloads need not round-trip bit-exactly
		}
		if got := floatToHalf(f); got != uint16(h) {
			t.Fatalf("half 0x%04x -> %v -> 0x%04x", h, f, got)
		}
	}
	// Spot values.
	if halfToFloat(0x3C00) != 1 || halfToFloat(0xBC00) != -1 || halfToFloat(0x0000) != 0 {
		t.Fatal("wrong unit values")
	}
	if halfToFloat(0x7C00) != float32(math.Inf(1)) {
		t.Fatal("wrong +inf")
	}
	if got := halfToFloat(0x0001); got != float32(math.Ldexp(1, -24)) {
		t.Fatalf("denormal = %v", got)
	}
	// float→half rounding: ties to even (half ulp near 1.0 is 1/2048).
	if got := floatToHalf(1 + 1.0/2048); got != 0x3C00 { // exactly half-way, rounds to even (down)
		t.Fatalf("tie rounding = 0x%04x", got)
	}
	if got := floatToHalf(1 + 3.0/2048); got != 0x3C02 { // half-way, rounds to even (up)
		t.Fatalf("tie rounding = 0x%04x", got)
	}
	if got := floatToHalf(100000); got != 0x7C00 { // overflow → inf
		t.Fatalf("overflow = 0x%04x", got)
	}
}

func TestValidateAlphaType(t *testing.T) {
	if at, ok := ValidateAlphaType(ColorTypeAlpha8, AlphaTypeUnpremul); !ok || at != AlphaTypePremul {
		t.Fatalf("A8 unpremul -> %v %v", at, ok)
	}
	if at, ok := ValidateAlphaType(ColorTypeRGB565, AlphaTypePremul); !ok || at != AlphaTypeOpaque {
		t.Fatalf("565 -> %v %v", at, ok)
	}
	if _, ok := ValidateAlphaType(ColorTypeRGBA8888, AlphaTypeUnknown); ok {
		t.Fatal("8888 unknown should fail")
	}
}

func TestBytesPerPixel(t *testing.T) {
	// Every entry mirrors Skia's SkColorTypeBytesPerPixel for the equivalent SkColorType.
	want := map[ColorType]int{
		ColorTypeUnknown:           0,
		ColorTypeAlpha8:            1,
		ColorTypeRGB565:            2,
		ColorTypeARGB4444:          2,
		ColorTypeRGBA8888:          4,
		ColorTypeRGB888x:           4,
		ColorTypeBGRA8888:          4,
		ColorTypeRGBA1010102:       4,
		ColorTypeBGRA1010102:       4,
		ColorTypeRGB101010x:        4,
		ColorTypeBGR101010x:        4,
		ColorTypeBGR101010xXR:      4,
		ColorTypeBGRA10101010XR:    8,
		ColorTypeRGBA10x6:          8,
		ColorTypeGray8:             1,
		ColorTypeRGBAF16Norm:       8,
		ColorTypeRGBAF16:           8,
		ColorTypeRGBF16F16F16x:     8,
		ColorTypeRGBAF32:           16,
		ColorTypeR8G8Unorm:         2,
		ColorTypeA16Float:          2,
		ColorTypeR16G16Float:       4,
		ColorTypeA16Unorm:          2,
		ColorTypeR16G16Unorm:       4,
		ColorTypeR16G16B16A16Unorm: 8,
		ColorTypeSRGBA8888:         4,
		ColorTypeR8Unorm:           1,
	}
	for ct := ColorTypeUnknown; ct <= ColorTypeR8Unorm; ct++ {
		expected, ok := want[ct]
		if !ok {
			t.Fatalf("color type %d is missing from the expected table", ct)
		}
		if got := ct.BytesPerPixel(); got != expected {
			t.Errorf("BytesPerPixel(%d) = %d, want %d", ct, got, expected)
		}
	}
}

func TestR8G8UnormIsTwoBytesPerPixel(t *testing.T) {
	info, ok := MakeInfo(3, 2, ColorTypeR8G8Unorm, AlphaTypeOpaque)
	if !ok {
		t.Fatal("MakeInfo rejected R8G8_unorm")
	}
	if got := info.MinRowBytes(); got != 6 {
		t.Fatalf("MinRowBytes = %d, want 6", got)
	}
	if p := NewPixels(info); p.U16s == nil || p.Words != nil || p.Bytes != nil {
		t.Fatalf("wrong storage lane: words=%v u16s=%v bytes=%v", p.Words != nil, p.U16s != nil, p.Bytes != nil)
	}
	// A minimally-strided caller buffer (2 bytes per pixel) must be accepted, not rejected as short.
	p, ok := NewPixelsCopyFromBytes(info, []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 6)
	if !ok {
		t.Fatal("NewPixelsCopyFromBytes rejected a minimally-strided buffer")
	}
	for i, expected := range []uint16{0x0201, 0x0403, 0x0605, 0x0807, 0x0A09, 0x0C0B} {
		if got := p.U16s[i]; got != expected {
			t.Errorf("U16s[%d] = 0x%04x, want 0x%04x", i, got, expected)
		}
	}
}

func TestConvertMemcpyAndSwizzle(t *testing.T) {
	info, _ := MakeInfo(2, 2, ColorTypeRGBA8888, AlphaTypePremul)
	src := NewPixels(info)
	src.Words[0] = 0xFF804020 // A=FF B=80 G=40 R=20
	src.Words[1] = 0x80402010
	src.Words[2] = 0x01020304
	src.Words[3] = 0xFFFFFFFF

	// memcpy lane
	dst := make([]byte, 16)
	if !ConvertPixels(info, dst, 8, src) {
		t.Fatal("convert failed")
	}
	if dst[0] != 0x20 || dst[1] != 0x40 || dst[2] != 0x80 || dst[3] != 0xFF {
		t.Fatalf("memcpy lane wrong: % x", dst[:4])
	}

	// swizzle lane (RGBA→BGRA, same alpha type)
	bgraInfo, _ := MakeInfo(2, 2, ColorTypeBGRA8888, AlphaTypePremul)
	if !ConvertPixels(bgraInfo, dst, 8, src) {
		t.Fatal("convert failed")
	}
	if dst[0] != 0x80 || dst[1] != 0x40 || dst[2] != 0x20 || dst[3] != 0xFF {
		t.Fatalf("swizzle lane wrong: % x", dst[:4])
	}
}

func TestConvertToAlpha8(t *testing.T) {
	info, _ := MakeInfo(2, 1, ColorTypeRGBA8888, AlphaTypePremul)
	src := NewPixels(info)
	src.Words[0] = 0x7F000000
	src.Words[1] = 0x01000000
	a8, _ := MakeInfo(2, 1, ColorTypeAlpha8, AlphaTypePremul)
	dst := make([]byte, 2)
	if !ConvertPixels(a8, dst, 2, src) {
		t.Fatal("convert failed")
	}
	if dst[0] != 0x7F || dst[1] != 0x01 {
		t.Fatalf("a8 = % x", dst)
	}
}

func TestConvertF16ToAlpha8FaithfulTruncation(t *testing.T) {
	// convert_to_alpha8's F16 lane is (uint8_t)(255 * half): truncate toward zero, no clamp. This pins that
	// oracle-exact behavior (a Skia fidelity requirement, not a rounding bug) and the deterministic mod-256 wrap
	// for out-of-range extended-range F16 alphas.
	info, _ := MakeInfo(4, 1, ColorTypeRGBAF16, AlphaTypePremul)
	src := NewPixels(info)
	// alphas: 0.5, 1.0, 2.0 (out of range high), -1.0 (out of range low).
	for i, a := range []float32{0.5, 1.0, 2.0, -1.0} {
		src.U16s[4*i+3] = floatToHalf(a)
	}
	a8, _ := MakeInfo(4, 1, ColorTypeAlpha8, AlphaTypePremul)
	dst := make([]byte, 4)
	if !ConvertPixels(a8, dst, 4, src) {
		t.Fatal("convert failed")
	}
	// 255*0.5 = 127.5 -> truncates to 127 (round-to-nearest would give 128, proving the faithful truncation);
	// 255*1.0 = 255; 255*2.0 = 510 -> low 8 bits = 254; 255*-1.0 = -255 -> low 8 bits = 1.
	want := []byte{127, 255, 254, 1}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("alpha %d: got %d, want %d (full % x)", i, dst[i], want[i], dst)
		}
	}
}

func TestConvertPremulToUnpremulRoundTrip(t *testing.T) {
	// premul → unpremul pays the sRGB linearize/encode round trip; opaque pixels must survive it exactly (the
	// inv(srgb(1)) == 1 invariant holds per channel for byte-exact 8-bit values).
	info, _ := MakeInfo(256, 1, ColorTypeRGBA8888, AlphaTypePremul)
	src := NewPixels(info)
	for i := range 256 {
		src.Words[i] = 0xFF000000 | uint32(i) | uint32(i)<<8 | uint32(i)<<16
	}
	un, _ := MakeInfo(256, 1, ColorTypeRGBA8888, AlphaTypeUnpremul)
	dst := make([]byte, 256*4)
	if !ConvertPixels(un, dst, 256*4, src) {
		t.Fatal("convert failed")
	}
	for i := range 256 {
		if dst[4*i] != byte(i) || dst[4*i+3] != 0xFF {
			t.Fatalf("pixel %d: got % x", i, dst[4*i:4*i+4])
		}
	}
	// half-alpha: premul 0x80 at alpha 0x80 unpremuls to 255.
	src.Words[0] = 0x80808080
	if !ConvertPixels(un, dst, 256*4, src) {
		t.Fatal("convert failed")
	}
	if dst[0] != 0xFF || dst[3] != 0x80 {
		t.Fatalf("half alpha: % x", dst[:4])
	}
}

func TestConvertGrayAnd565AndF16(t *testing.T) {
	// Gray8 → RGBA
	ginfo, _ := MakeInfo(1, 1, ColorTypeGray8, AlphaTypeOpaque)
	src := NewPixels(ginfo)
	src.Bytes[0] = 0x40
	rgba, _ := MakeInfo(1, 1, ColorTypeRGBA8888, AlphaTypePremul)
	dst := make([]byte, 4)
	if !ConvertPixels(rgba, dst, 4, src) || dst[0] != 0x40 || dst[1] != 0x40 || dst[2] != 0x40 || dst[3] != 0xFF {
		t.Fatalf("gray -> rgba: % x", dst)
	}

	// RGBA → 565 → RGBA
	cinfo, _ := MakeInfo(1, 1, ColorTypeRGBA8888, AlphaTypeOpaque)
	csrc := NewPixels(cinfo)
	csrc.Words[0] = 0xFF0080FF // R=FF G=80 B=00
	c565, _ := MakeInfo(1, 1, ColorTypeRGB565, AlphaTypeOpaque)
	d16 := make([]byte, 2)
	if !ConvertPixels(c565, d16, 2, csrc) {
		t.Fatal("565 convert failed")
	}
	v := uint16(d16[0]) | uint16(d16[1])<<8
	if v>>11 != 31 || (v>>5)&63 != 32 || v&31 != 0 {
		t.Fatalf("565 = %04x", v)
	}

	// RGBA → F16 → RGBA round trips 8-bit values exactly.
	f16, _ := MakeInfo(1, 1, ColorTypeRGBAF16, AlphaTypePremul)
	d64 := make([]byte, 8)
	pinfo, _ := MakeInfo(1, 1, ColorTypeRGBA8888, AlphaTypePremul)
	psrc := NewPixels(pinfo)
	psrc.Words[0] = 0x80402010
	if !ConvertPixels(f16, d64, 8, psrc) {
		t.Fatal("f16 convert failed")
	}
	fp, _ := NewPixelsCopyFromBytes(f16, d64, 8)
	back := make([]byte, 4)
	if !ConvertPixels(pinfo, back, 4, fp) {
		t.Fatal("f16 back convert failed")
	}
	if back[0] != 0x10 || back[1] != 0x20 || back[2] != 0x40 || back[3] != 0x80 {
		t.Fatalf("f16 round trip: % x", back)
	}
}

func TestReadPixelsTrim(t *testing.T) {
	info := MakeN32Premul(4, 4)
	p := NewPixels(info)
	for i := range p.Words {
		p.Words[i] = uint32(i) | 0xFF000000
	}
	im := FromPixels(p)

	dstInfo := MakeN32Premul(2, 2)
	dst := make([]byte, 16)
	// fully inside
	if !im.ReadPixels(dstInfo, dst, 8, 1, 1, CachingAllow) {
		t.Fatal("read failed")
	}
	if dst[0] != 5 || dst[4] != 6 || dst[8] != 9 {
		t.Fatalf("inside read: % x", dst)
	}
	// negative origin trims and offsets dst
	dst = make([]byte, 16)
	if !im.ReadPixels(dstInfo, dst, 8, -1, -1, CachingAllow) {
		t.Fatal("read failed")
	}
	// only the (1,1) cell of dst is written, with src pixel (0,0)
	if dst[12] != 0 || dst[15] != 0xFF {
		t.Fatalf("trimmed read: % x", dst)
	}
	if dst[0] != 0 || dst[3] != 0 {
		t.Fatalf("untouched region written: % x", dst)
	}
	// fully outside
	if im.ReadPixels(dstInfo, dst, 8, 10, 10, CachingAllow) {
		t.Fatal("outside read should fail")
	}
	// negative origin beyond the destination extent must return false, not panic slicing dst past its length
	dst = make([]byte, 16)
	if im.ReadPixels(dstInfo, dst, 8, -100, -100, CachingAllow) {
		t.Fatal("far-negative read should fail")
	}
	if im.ReadPixels(dstInfo, dst, 8, -100, 0, CachingAllow) {
		t.Fatal("far-negative x read should fail")
	}
	if im.ReadPixels(dstInfo, dst, 8, 0, -100, CachingAllow) {
		t.Fatal("far-negative y read should fail")
	}
	// A srcX/srcY near MaxInt32 must trim to an empty rect and return false; computing x+w in int32 wraps the sum
	// negative, skips the clamp, and leaves Subset with an out-of-range offset that panics.
	dst = make([]byte, 16)
	for _, tc := range []struct {
		name       string
		srcX, srcY int32
	}{
		{name: "x at max", srcX: math.MaxInt32, srcY: 0},
		{name: "y at max", srcX: 0, srcY: math.MaxInt32},
		{name: "both at max", srcX: math.MaxInt32, srcY: math.MaxInt32},
		{name: "x one short of wrapping", srcX: math.MaxInt32 - dstInfo.Width, srcY: 0},
		{name: "y one short of wrapping", srcX: 0, srcY: math.MaxInt32 - dstInfo.Height},
	} {
		if im.ReadPixels(dstInfo, dst, 8, tc.srcX, tc.srcY, CachingAllow) {
			t.Fatalf("%s: far-positive read should fail", tc.name)
		}
	}
	for i, b := range dst {
		if b != 0 {
			t.Fatalf("far-positive read wrote dst[%d] = %#x", i, b)
		}
	}
	// The in-bounds edge case that abuts the clamp must still succeed and trim to a 1x1 rect.
	if !im.ReadPixels(dstInfo, dst, 8, 3, 3, CachingAllow) {
		t.Fatal("bottom-right corner read failed")
	}
	if dst[0] != 15 || dst[3] != 0xFF {
		t.Fatalf("corner read: % x", dst)
	}
}

func TestNewPixelsCopyFromBytesOverflow(t *testing.T) {
	info := MakeN32Premul(4, 4)
	// A rowBytes large enough to wrap int in rowBytes*(h-1)+w*bpp must be rejected, not slice data out of range.
	huge := math.MaxInt/int(info.Height-1) + 2
	if p, ok := NewPixelsCopyFromBytes(info, make([]byte, 64), huge); ok || p != nil {
		t.Fatal("overflowing rowBytes accepted")
	}
	// A merely short buffer is still rejected with a normal rowBytes.
	if p, ok := NewPixelsCopyFromBytes(info, make([]byte, 16), 16); ok || p != nil {
		t.Fatal("short buffer accepted")
	}
	// An exactly-sized buffer is accepted (rowBytes*(h-1)+w*bpp = 16*3+16 = 64).
	if p, ok := NewPixelsCopyFromBytes(info, make([]byte, 64), 16); !ok || p == nil {
		t.Fatal("exact buffer rejected")
	}
}

func TestNewRasterDataAndSubset(t *testing.T) {
	info, _ := MakeInfo(2, 2, ColorTypeRGBA8888, AlphaTypePremul)
	data := []byte{
		1, 2, 3, 255, 4, 5, 6, 255, 0, 0, // rowBytes 10 (2 slack)
		7, 8, 9, 255, 10, 11, 12, 255, 0, 0,
	}
	im := NewRasterData(info, data, 10)
	if im == nil {
		t.Fatal("raster data image nil")
	}
	if im.Width() != 2 || im.Height() != 2 {
		t.Fatal("dims wrong")
	}
	p := im.PeekPixels(CachingAllow)
	if p.Words[0] != 0xFF030201 || p.Words[3] != 0xFF0C0B0A {
		t.Fatalf("words: %08x %08x", p.Words[0], p.Words[3])
	}
	sub := p.Subset(1, 1, 2, 2)
	if sub.Info.Width != 1 || sub.Info.Height != 1 || sub.Words[0] != 0xFF0C0B0A {
		t.Fatal("subset wrong")
	}
	// too-small rowBytes rejected
	if NewRasterData(info, data, 7) != nil {
		t.Fatal("short rowBytes accepted")
	}
}

func TestElemsPerPixel(t *testing.T) {
	// Only F16 spreads a pixel across several elements of the active slice; everything else is one element per pixel.
	for ct := ColorTypeUnknown; ct <= ColorTypeR8Unorm; ct++ {
		want := int32(1)
		if ct == ColorTypeRGBAF16 {
			want = 4
		}
		if got := ct.ElemsPerPixel(); got != want {
			t.Errorf("ElemsPerPixel(%d) = %d, want %d", ct, got, want)
		}
	}
	// RowElems counts elements, so the stride of an F16 container is four per pixel and the element index of pixel
	// (x, y) is y*RowElems + x*ElemsPerPixel.
	info, ok := MakeInfo(3, 2, ColorTypeRGBAF16, AlphaTypePremul)
	if !ok {
		t.Fatal("MakeInfo rejected RGBA_F16")
	}
	p := NewPixels(info)
	if p.RowElems != 12 {
		t.Fatalf("F16 RowElems = %d, want 12", p.RowElems)
	}
	if len(p.U16s) != 24 {
		t.Fatalf("F16 storage = %d elements, want 24", len(p.U16s))
	}
	if got := int(p.RowElems)*1 + 2*int(info.ColorType.ElemsPerPixel()); got != 20 {
		t.Fatalf("element index of (2,1) = %d, want 20", got)
	}
}

// TestToUnormNaNIsZero pins toUnorm's NaN handling. NaN reaches it on the highp path (loadHighp's F16 lane returns
// halfToFloat of arbitrary caller-supplied halves, and F16 is a Supported() source type), where a plain `f < 0 … else
// if f > scale` clamp lets it straight through to a float→uint32 cast whose result the Go spec leaves
// implementation-dependent. 0 is what the ARMv8 lane this mirrors produces (FCVTNU of NaN is 0), and it matches the
// F16→Alpha8 lane's deliberate int32 intermediate: the two must not disagree about a determinism property this project
// pins goldens on.
func TestToUnormNaNIsZero(t *testing.T) {
	nan := float32(math.NaN())
	for _, scale := range []float32{255, 63, 31} {
		if got := toUnorm(nan, scale); got != 0 {
			t.Errorf("toUnorm(NaN, %v) = %d, want 0", scale, got)
		}
	}
	// The ordinary clamp and round-to-nearest-even are unchanged.
	for _, tc := range []struct {
		v, scale float32
		want     uint32
	}{
		{v: 0, scale: 255, want: 0},
		{v: 1, scale: 255, want: 255},
		{v: -0.5, scale: 255, want: 0},
		{v: 1.5, scale: 255, want: 255},
		{v: 0.5, scale: 255, want: 128}, // 127.5 ties to even
		{v: 0.5, scale: 31, want: 16},   // 15.5 ties to even
		{v: float32(math.Inf(-1)), scale: 255, want: 0},
		{v: float32(math.Inf(1)), scale: 255, want: 255},
	} {
		if got := toUnorm(tc.v, tc.scale); got != tc.want {
			t.Errorf("toUnorm(%v, %v) = %d, want %d", tc.v, tc.scale, got, tc.want)
		}
	}
}

// TestConvertPixelsF16NaNIsDeterministic drives the same NaN through the real conversion path an external caller can
// reach: an F16 source (whose halves are caller-supplied) converted to 8888, which runs the highp pipeline because the
// F16 endpoint forces it. Every channel must land on a defined byte rather than an implementation-dependent cast.
func TestConvertPixelsF16NaNIsDeterministic(t *testing.T) {
	srcInfo, ok := MakeInfo(2, 1, ColorTypeRGBAF16, AlphaTypeUnpremul)
	if !ok {
		t.Fatal("MakeInfo rejected RGBA_F16/unpremul")
	}
	src := NewPixels(srcInfo)
	for i := range src.U16s {
		src.U16s[i] = 0x7E00 // a quiet NaN half
	}
	dstInfo, ok := MakeInfo(2, 1, ColorTypeRGBA8888, AlphaTypePremul)
	if !ok {
		t.Fatal("MakeInfo rejected RGBA_8888/premul")
	}
	dst := make([]byte, 8)
	if !ConvertPixels(dstInfo, dst, 8, src) {
		t.Fatal("ConvertPixels failed")
	}
	for i, b := range dst {
		if b != 0 {
			t.Fatalf("dst[%d] = %d, want 0 (NaN must convert to transparent black)", i, b)
		}
	}

	// The 565 store quantizers take the same lane; NaN must not smear into the packed bit fields either.
	d565Info, ok := MakeInfo(2, 1, ColorTypeRGB565, AlphaTypeOpaque)
	if !ok {
		t.Fatal("MakeInfo rejected RGB565")
	}
	d565 := make([]byte, 4)
	if !ConvertPixels(d565Info, d565, 4, src) {
		t.Fatal("ConvertPixels to 565 failed")
	}
	for i, b := range d565 {
		if b != 0 {
			t.Fatalf("565 dst[%d] = %d, want 0", i, b)
		}
	}
}

// TestConvertF16ToAlpha8NonFiniteIsDeterministic covers the halves that halfToFloat maps to ±Inf and NaN. F16 is a
// Supported() source type, so those bits are caller-supplied, and int32(±Inf)/int32(NaN) is exactly the
// implementation-defined conversion the lane's deliberate int32 intermediate was written to eliminate: before the
// saturation in alpha8FromHalf, an alpha half of 0x7C00 measured 255 on darwin/arm64 and 0 on amd64. The bytes below
// are the ARMv8 FCVTZS results the rest of the F16 lanes already mirror, and must now be the same on every target.
func TestConvertF16ToAlpha8NonFiniteIsDeterministic(t *testing.T) {
	halves := []uint16{0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7DFF}
	want := []byte{
		0xFF, // +Inf saturates to INT32_MAX, whose low byte is 0xFF
		0x00, // -Inf saturates to INT32_MIN, whose low byte is 0
		0x00, // NaN
		0x00, // negative NaN
		0x00, // NaN with a different payload
	}
	info, ok := MakeInfo(int32(len(halves)), 1, ColorTypeRGBAF16, AlphaTypePremul)
	if !ok {
		t.Fatal("MakeInfo rejected RGBA_F16/premul")
	}
	src := NewPixels(info)
	for i, h := range halves {
		src.U16s[4*i+3] = h
	}
	a8, ok := MakeInfo(int32(len(halves)), 1, ColorTypeAlpha8, AlphaTypePremul)
	if !ok {
		t.Fatal("MakeInfo rejected Alpha8/premul")
	}
	dst := make([]byte, len(halves))
	if !ConvertPixels(a8, dst, len(halves), src) {
		t.Fatal("convert failed")
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("half %#04x: got %d, want %d (full % x)", halves[i], dst[i], want[i], dst)
		}
	}

	// The finite halves keep the faithful truncate-then-low-8-bits behavior the saturation must not disturb; the
	// largest finite half is the one nearest the int32 guard.
	for _, tc := range []struct {
		half uint16
		want byte
	}{
		{half: floatToHalf(0.5), want: 127}, // 127.5 truncates rather than rounding to 128
		{half: floatToHalf(1), want: 255},
		{half: floatToHalf(2), want: 254}, // 510 wraps mod 256
		{half: floatToHalf(-1), want: 1},  // -255 wraps mod 256
		{half: 0x7BFF, want: 32},          // 255 * 65504 = 16703520, the largest finite product; low 8 bits
		{half: 0xFBFF, want: 224},         // -16703520, low 8 bits
	} {
		if got := alpha8FromHalf(tc.half); got != tc.want {
			t.Errorf("alpha8FromHalf(%#04x) = %d, want %d", tc.half, got, tc.want)
		}
	}
}

// TestNewFromEncodedValidatesReportedInfo pins the validation NewFromEncoded applies to whatever a codec reports. A
// codec parses the encoded header and reports it verbatim, and several formats let a tiny file declare enormous
// dimensions, so without this check maxDimension — which exists to keep the byte-offset and row-stride math in range —
// was simply bypassed for every deferred-decode image, and the resulting *Image panicked on first use. NewRasterData
// has always applied the same two tests.
func TestNewFromEncodedValidatesReportedInfo(t *testing.T) {
	magic := "imagecore-stub-codec"
	var reported ImageInfo
	var reportedOK bool
	RegisterCodec(Codec{
		Name:       "imagecore-test-stub",
		Matches:    func(data []byte) bool { return len(data) >= len(magic) && string(data[:len(magic)]) == magic },
		DecodeInfo: func([]byte) (ImageInfo, bool) { return reported, reportedOK },
		Decode:     func([]byte) (*Pixels, error) { return nil, nil },
	})
	for _, tc := range []struct {
		name  string
		info  ImageInfo
		ok    bool
		valid bool
	}{
		{name: "valid", info: MakeN32Premul(4, 4), ok: true, valid: true},
		{name: "at the dimension limit", info: MakeN32Premul(maxDimension, 1), ok: true, valid: true},
		{name: "width past the limit", info: MakeN32Premul(maxDimension+1, 1), ok: true},
		{name: "height past the limit", info: MakeN32Premul(1, maxDimension+1), ok: true},
		{name: "the reproducing PNG's dimensions", info: MakeN32Premul(1<<30, 1<<28), ok: true},
		{name: "zero height", info: MakeN32Premul(4, 0), ok: true},
		{name: "negative width", info: MakeN32Premul(-4, 4), ok: true},
		{name: "unsupported color type", info: ImageInfo{
			Width: 4, Height: 4, ColorType: ColorTypeRGBA1010102,
			AlphaType: AlphaTypePremul,
		}, ok: true},
		{name: "codec rejected the header", info: MakeN32Premul(4, 4)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reported, reportedOK = tc.info, tc.ok
			switch im := NewFromEncoded([]byte(magic)); {
			case tc.valid && im == nil:
				t.Fatalf("NewFromEncoded returned nil for %+v", tc.info)
			case !tc.valid && im != nil:
				t.Fatalf("NewFromEncoded accepted %+v", tc.info)
			}
		})
	}
}
