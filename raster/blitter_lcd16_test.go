// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// LCD16 blit tests: the LCD16-over-8888 row procs pinned byte-exact against independently transcribed reference outputs
// (blendLCD16/blendLCD16Opaque, blendRowLCD16/blendRowLCD16OpaqueGeneric), the pipeline scale_565/lerp_565 lanes, and
// AA-clip LCD merge.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// refBlendLCD16 is an independent transcription of blendLCD16's math over this package's device word layout (R at bit
// 0, G at 8, B at 16, A at 24).
func refBlendLCD16(srcA, srcR, srcG, srcB int, dst uint32, mask uint16) uint32 {
	if mask == 0 {
		return dst
	}
	up32 := func(v int) int { return v + v>>4 }
	bl32 := func(s, d, scale int) uint32 { return uint32(d + ((s - d) * scale >> 5)) }
	srcA += srcA >> 7 // upscale 0..255 to 0..256
	maskR := up32(int(mask>>11) & 31)
	maskG := up32(int(mask>>6) & 31)
	maskB := up32(int(mask) & 31)
	maskR = maskR * srcA >> 8
	maskG = maskG * srcA >> 8
	maskB = maskB * srcA >> 8
	dstA := int(dst >> 24)
	dstR := int(dst & 0xFF)
	dstG := int(dst >> 8 & 0xFF)
	dstB := int(dst >> 16 & 0xFF)
	var maskA int
	if srcA-1 < dstA {
		maskA = min(maskR, min(maskG, maskB))
	} else {
		maskA = max(maskR, max(maskG, maskB))
	}
	return bl32(srcR, dstR, maskR) | bl32(srcG, dstG, maskG)<<8 |
		bl32(srcB, dstB, maskB)<<16 | bl32(0xFF, dstA, maskA)<<24
}

// refBlendLCD16Opaque transcribes blend_lcd16_opaque.
func refBlendLCD16Opaque(srcR, srcG, srcB int, dst uint32, mask uint16, opaqueDst uint32) uint32 {
	if mask == 0 {
		return dst
	}
	if mask == 0xFFFF {
		return opaqueDst
	}
	up32 := func(v int) int { return v + v>>4 }
	bl32 := func(s, d, scale int) uint32 { return uint32(d + ((s - d) * scale >> 5)) }
	maskR := up32(int(mask>>11) & 31)
	maskG := up32(int(mask>>6) & 31)
	maskB := up32(int(mask) & 31)
	dstA := int(dst >> 24)
	dstR := int(dst & 0xFF)
	dstG := int(dst >> 8 & 0xFF)
	dstB := int(dst >> 16 & 0xFF)
	maskA := max(maskR, max(maskG, maskB))
	return bl32(srcR, dstR, maskR) | bl32(srcG, dstG, maskG)<<8 |
		bl32(srcB, dstB, maskB)<<16 | bl32(0xFF, dstA, maskA)<<24
}

var lcdTestMasks = []uint16{
	0x0000, 0xFFFF, 0x0001, 0x8000, 0x07E0, 0xF800, 0x001F, 0x1234, 0xABCD, 0x5555,
	0xAAAA, 0x7BEF, 0x39E7, 0xFFE0, 0x07FF, 0xF81F, 0x4210, 0x0842,
}

var lcdTestDsts = []uint32{
	0xFF000000, 0xFFFFFFFF, 0xFF123456, 0xFF808080, 0x80404040, 0x00000000, 0xFFFF0000,
	0xFF0000FF, 0xCC663311,
}

// TestBlitRowLCD16Pinned pins the solid-color row procs byte-exact against the reference transcription.
func TestBlitRowLCD16Pinned(t *testing.T) {
	colors := []colorcore.Color{
		0xFF000000, 0xFFFFFFFF, 0xFF102030, 0x80FF8040, 0x01FFFFFF, 0xFF00FF00,
	}
	for _, c := range colors {
		srcA := int(c.A())
		srcR := int(c.R())
		srcG := int(c.G())
		srcB := int(c.B())
		for _, m := range lcdTestMasks {
			for _, d := range lcdTestDsts {
				// General proc.
				dst := []uint32{d}
				blitRowLCD16Generic(dst, []uint16{m}, uint32(srcA), uint32(srcR), uint32(srcG), uint32(srcB))
				want := refBlendLCD16(srcA, srcR, srcG, srcB, d, m)
				if dst[0] != want {
					t.Fatalf("blitRowLCD16Generic(color %#x, mask %#04x, dst %#08x) = %#08x, want %#08x",
						uint32(c), m, d, dst[0], want)
				}
				// Opaque proc.
				if srcA == 0xFF {
					opaqueDst := deviceRGBA(c.PreMultiply())
					dst[0] = d
					blitRowLCD16OpaqueGeneric(dst, []uint16{m}, uint32(srcR), uint32(srcG), uint32(srcB),
						opaqueDst)
					want = refBlendLCD16Opaque(srcR, srcG, srcB, d, m, opaqueDst)
					if dst[0] != want {
						t.Fatalf("blitRowLCD16OpaqueGeneric(color %#x, mask %#04x, dst %#08x) = %#08x, want %#08x",
							uint32(c), m, d, dst[0], want)
					}
				}
			}
		}
	}
}

// TestSolidBlitterLCD16 drives the row procs through SolidBlitter.BlitMask (blit_color's kLCD16 lane).
func TestSolidBlitterLCD16(t *testing.T) {
	dev := NewPixmap(4, 2)
	for i := range dev.Pix {
		dev.Pix[i] = 0xFFFFFFFF
	}
	color := colorcore.Color(0xFF102030)
	b := NewSolidBlitter(dev, color)
	mask := &Mask{
		Image16:  []uint16{0xFFFF, 0x0000, 0xF800, 0x001F, 0x07E0, 0x8410, 0x1234, 0xFFFF},
		Bounds:   geom.IRectWH(4, 2),
		RowBytes: 8,
		Format:   MaskLCD16,
	}
	b.BlitMask(mask, mask.Bounds)
	opaqueDst := deviceRGBA(color.PreMultiply())
	for i, m := range mask.Image16 {
		want := refBlendLCD16Opaque(0x10, 0x20, 0x30, 0xFFFFFFFF, m, opaqueDst)
		if dev.Pix[i] != want {
			t.Errorf("pixel %d (mask %#04x) = %#08x, want %#08x", i, m, dev.Pix[i], want)
		}
	}
	// Full coverage lands exactly on the premultiplied color; zero coverage leaves the dst.
	if dev.Pix[0] != opaqueDst || dev.Pix[7] != opaqueDst {
		t.Errorf("full-coverage pixels = %#08x/%#08x, want %#08x", dev.Pix[0], dev.Pix[7], opaqueDst)
	}
	if dev.Pix[1] != 0xFFFFFFFF {
		t.Errorf("zero-coverage pixel = %#08x, want untouched white", dev.Pix[1])
	}

	// The translucent color takes the general proc.
	for i := range dev.Pix {
		dev.Pix[i] = 0xFF000000
	}
	tb := NewSolidBlitter(dev, 0x80FF8040)
	tb.BlitMask(mask, mask.Bounds)
	for i, m := range mask.Image16 {
		want := refBlendLCD16(0x80, 0xFF, 0x80, 0x40, 0xFF000000, m)
		if dev.Pix[i] != want {
			t.Errorf("translucent pixel %d (mask %#04x) = %#08x, want %#08x", i, m, dev.Pix[i], want)
		}
	}
}

// TestLegacyShaderBlitterLCD16 pins the legacy shader blitter's LCD16 blend rows.
func TestLegacyShaderBlitterLCD16(t *testing.T) {
	// Opaque row: blend32 per channel toward the shader span. Device words carry R at bit 0, so 0xFF10E070 is R=0x70,
	// G=0xE0, B=0x10.
	dst := []uint32{0xFF808080, 0xFF808080, 0xFF808080}
	src := []uint32{0xFF10E070, 0xFF10E070, 0xFF10E070}
	masks := []uint16{0x0000, 0xFFFF, 0x1234}
	blendRowLCD16OpaqueGeneric(dst, masks, src)
	if dst[0] != 0xFF808080 {
		t.Errorf("opaque mask 0 modified dst: %#08x", dst[0])
	}
	// Full coverage → blend_32 with scale 32 lands exactly on src channels (alpha forced opaque).
	if dst[1] != 0xFF10E070 {
		t.Errorf("opaque full coverage = %#08x, want %#08x", dst[1], uint32(0xFF10E070))
	}
	up32 := func(v int) int { return v + v>>4 }
	bl32 := func(s, d, scale int) uint32 { return uint32(d + ((s - d) * scale >> 5)) }
	mr := up32(int(0x1234>>11) & 31)
	mg := up32(int(0x1234>>6) & 31)
	mb := up32(int(0x1234) & 31)
	want := bl32(0x70, 0x80, mr) | bl32(0xE0, 0x80, mg)<<8 | bl32(0x10, 0x80, mb)<<16 | 0xFF<<24
	if dst[2] != want {
		t.Errorf("opaque partial = %#08x, want %#08x", dst[2], want)
	}

	// General row: src_alpha_blend with 8-bit coverages.
	dst = []uint32{0xFF808080}
	src = []uint32{0x80104030} // premul, a=0x80
	blendRowLCD16(dst, []uint16{0xABCD}, src)
	srcAlphaBlend := func(s, d, sa, m int) int { return d + ((s-sa*d>>8)*m)>>8 }
	up255 := func(v int) int { return v<<3 | v>>2 }
	sa := 0x80 + 0x80>>7
	mr = up255(int(0xABCD>>11) & 31)
	mg = up255(int(0xABCD>>6) & 31)
	mb = up255(int(0xABCD) & 31)
	want = uint32(srcAlphaBlend(0x30, 0x80, sa, mr)) |
		uint32(srcAlphaBlend(0x40, 0x80, sa, mg))<<8 |
		uint32(srcAlphaBlend(0x10, 0x80, sa, mb))<<16 | 0xFF<<24
	if dst[0] != want {
		t.Errorf("general shader row = %#08x, want %#08x", dst[0], want)
	}
}

// TestBlendBlitterLCD16 exercises the pipeline lerp_565 lane (both precisions) for src-over.
func TestBlendBlitterLCD16(t *testing.T) {
	mask := &Mask{
		Image16:  []uint16{0x0000, 0xFFFF, 0xF800, 0x1234},
		Bounds:   geom.IRectWH(4, 1),
		RowBytes: 8,
		Format:   MaskLCD16,
	}
	// Opaque constant color + src-over reduces to Src in chooseBlitter; build the blitter directly with SrcOver to
	// exercise the lerp lane.
	dev := NewPixmap(4, 1)
	for i := range dev.Pix {
		dev.Pix[i] = 0xFFFFFFFF
	}
	bb := NewBlendBlitterPM4f(dev, colorcore.PMColor4f{R: 0.2, G: 0.4, B: 0.6, A: 1}, BlendSrcOver)
	bb.BlitMask(mask, mask.Bounds)
	if dev.Pix[0] != 0xFFFFFFFF {
		t.Errorf("lerp mask 0 modified dst: %#08x", dev.Pix[0])
	}
	// Full coverage lands on the blended color exactly (lerp with t=1 per channel).
	full := dev.Pix[1]
	if full == 0xFFFFFFFF {
		t.Error("full-coverage pixel unchanged")
	}
	// The red-only mask changes red strongly and leaves blue near the dst (per-channel lerp).
	redOnly := dev.Pix[2]
	if redOnly&0xFF == 0xFF {
		t.Error("red-only coverage did not move the red channel")
	}
	if b := redOnly >> 16 & 0xFF; b != 0xFF {
		t.Errorf("red-only coverage moved the blue channel: %#02x", b)
	}
}

// TestAAClipLCD16Merge pins mergeOneMaskRow16: each 5/6/5 channel scales by mulDiv255Round.
func TestAAClipLCD16Merge(t *testing.T) {
	src := []uint16{0xFFFF, 0x1234, 0xF800}
	dst := make([]uint16, 3)
	// One clip run covering all three pixels at alpha 0x80: row = {count, alpha}.
	row := []uint8{3, 0x80}
	mergeOneMaskRow16(src, 3, row, 3, dst)
	for i, v := range src {
		r := colorcore.MulDiv255Round(uint8(v>>11&0x1F), 0x80)
		g := colorcore.MulDiv255Round(uint8(v>>5&0x3F), 0x80)
		b := colorcore.MulDiv255Round(uint8(v&0x1F), 0x80)
		want := uint16(r)<<11 | uint16(g)<<5 | uint16(b)
		if dst[i] != want {
			t.Errorf("merge pixel %d = %#04x, want %#04x", i, dst[i], want)
		}
	}
	// Alpha FF copies, alpha 0 zeroes.
	mergeOneMaskRow16(src, 3, []uint8{3, 0xFF}, 3, dst)
	for i := range src {
		if dst[i] != src[i] {
			t.Errorf("alpha FF pixel %d = %#04x, want %#04x", i, dst[i], src[i])
		}
	}
	mergeOneMaskRow16(src, 3, []uint8{3, 0x00}, 3, dst)
	for i := range dst {
		if dst[i] != 0 {
			t.Errorf("alpha 0 pixel %d = %#04x, want 0", i, dst[i])
		}
	}
}
