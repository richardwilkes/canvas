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
