// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Sprite blitters: blit a source pixmap positioned at integer device coordinates. This is how layer contents return to
// their parent device at restore() and how integer-aligned image draws land (the sprite fast path for image draws).
//
// Byte-exactness note: the src-over row kernels reproduce the NEON forms that the darwin/arm64 oracle leg executes for
// every pixel of a row (including odd-length tails). The x86 SSE2 bodies compute s + ((d*(256-sa))>>8) where NEON
// computes s + mulDiv255Round(d, 255-sa), and x86 rows shorter than the vector width take scalar tails with a third
// rounding — the same class of per-leg ±1 divergence already recorded for the lowp pipeline.

package raster

import "github.com/richardwilkes/canvas/geom"

// spriteBase carries the common sprite state: the source pixmap and its position in destination space. Only BlitRect
// (and BlitH via BlitRect) is reachable: sprites are drawn through a rect-fill path whose clip wrappers only emit rect
// spans. The remaining Blitter methods are no-op stubs.
type spriteBase struct {
	dst  *Pixmap
	src  *Pixmap
	left int32
	top  int32
}

// blitHVia implements BlitH by falling back to BlitRect.
func (s *spriteBase) blitHVia(b Blitter, x, y, width int32) {
	b.BlitRect(x, y, width, 1)
}

// BlitAntiH is unreachable for sprites: no fallback strategy.
func (s *spriteBase) BlitAntiH(_, _ int32, _ []Alpha, _ []int16) {}

// BlitV is unreachable for sprites (would lower to BlitAntiH, which is a no-op).
func (s *spriteBase) BlitV(_, _, _ int32, _ Alpha) {}

// BlitAntiRect is unreachable for sprites.
func (s *spriteBase) BlitAntiRect(_, _, _, _ int32, _, _ Alpha) {}

// BlitMask is unreachable for sprites (no-op for this reachable set).
func (s *spriteBase) BlitMask(_ *Mask, _ geom.IRect) {}

// BlitAntiH2 is unreachable for sprites.
func (s *spriteBase) BlitAntiH2(_, _ int32, _, _ Alpha) {}

// BlitAntiV2 is unreachable for sprites.
func (s *spriteBase) BlitAntiV2(_, _ int32, _, _ Alpha) {}

///////////////////////////////////////////////////////////////////////////////

// spriteMemcpy blits a source sprite by copying pixels verbatim (no blending).
type spriteMemcpy struct {
	spriteBase
}

// BlitH implements Blitter.
func (s *spriteMemcpy) BlitH(x, y, width int32) { s.blitHVia(s, x, y, width) }

// BlitRect implements Blitter.
func (s *spriteMemcpy) BlitRect(x, y, width, height int32) {
	for row := int32(0); row < height; row++ {
		d := s.dst.addr(x, y+row)
		sr := s.src.addr(x-s.left, y+row-s.top)
		copy(s.dst.Pix[d:d+int(width)], s.src.Pix[sr:sr+int(width)])
	}
}

///////////////////////////////////////////////////////////////////////////////

// swarMask8 selects the low byte of each 16-bit lane of a spread64 word.
const swarMask8 = uint64(0x00FF00FF00FF00FF)

// spread64 splits an 8888 pixel's four bytes into the four 16-bit lanes of a uint64 (lane order [b0, b2, b1, b3]), so
// per-channel byte arithmetic runs four channels per integer op — the portable stand-in for the widening that SIMD
// kernels do with NEON/SSE lanes. Each lane has 8 headroom bits, enough for a byte times a 0..256 scale with no
// cross-lane carry.
func spread64(c uint32) uint64 {
	return (uint64(c) | uint64(c)<<24) & swarMask8
}

// pack64 packs the low byte of each 16-bit lane of a spread64-layout word back into an 8888 pixel. The caller must have
// reduced every lane to <= 0xFF.
func pack64(v uint64) uint32 {
	return uint32(v) | uint32(v>>24)
}

// pmSrcOverRowGeneric is the portable form of pmSrcOverRow: per channel saturating-add(src, mulDiv255Round(dst,
// 255-srcA)), run four channels per op in spread64 lanes. On arm64 pmSrcOverRow runs the NEON kernel instead and uses
// this only for the sub-quad tail. Bit-exactness per lane: prod = d*nalpha + 128 <= 255*255+128 = 0xFE81 stays in its
// 16-bit lane; (prod>>8)&0xFF is exactly the per-channel prod>>8 (prod < 2^16), and their sum <= 0xFF7F still carries
// nowhere, so the lane holds mulDiv255Round32's value. The saturating add's lanes reach at most 510, so bit 8 of a lane
// is the overflow flag; OR-ing 0xFF into overflowed lanes reproduces satAdd8's clamp on the low byte.
func pmSrcOverRowGeneric(dst, src []uint32) {
	const half = uint64(0x0080008000800080)
	const laneLSB = uint64(0x0001000100010001)
	for i, s := range src {
		nalpha := uint64(255 - (s >> 24))
		prod := spread64(dst[i])*nalpha + half
		prod += (prod >> 8) & swarMask8
		t := spread64(s) + ((prod >> 8) & swarMask8)
		t |= (t >> 8 & laneLSB) * 0xFF
		dst[i] = pack64(t & swarMask8)
	}
}

// pmBlendRow computes, per channel, (src*alpha256 + dst*alphaMulInv256(srcA, alpha256)) >> 8.
func pmBlendRow(dst, src []uint32, alpha256 uint32) {
	for i, s := range src {
		d := dst[i]
		dstScale := alphaMulInv256(s>>24, alpha256)
		r := (s&0xFF*alpha256 + d&0xFF*dstScale) >> 8
		g := (s>>8&0xFF*alpha256 + d>>8&0xFF*dstScale) >> 8
		b := (s>>16&0xFF*alpha256 + d>>16&0xFF*dstScale) >> 8
		a := (s>>24*alpha256 + d>>24*dstScale) >> 8
		dst[i] = r | g<<8 | b<<16 | a<<24
	}
}

// mulDiv255Round32 is mulDiv255Round on uint32 lanes.
func mulDiv255Round32(a, b uint32) uint32 {
	prod := a*b + 128
	return (prod + (prod >> 8)) >> 8
}

// satAdd8 is a uint8 saturating add on uint32 lanes.
func satAdd8(a, b uint32) uint32 {
	v := a + b
	if v > 255 {
		return 255
	}
	return v
}

// spriteD32S32 is the legacy N32 src-over sprite, honoring the paint's global alpha. It dispatches to one of four row
// procs depending on whether the source is opaque and whether the paint alpha is 255.
type spriteD32S32 struct {
	spriteBase
	alpha     uint32 // paint alpha, 0..255
	srcOpaque bool
}

// BlitH implements Blitter.
func (s *spriteD32S32) BlitH(x, y, width int32) { s.blitHVia(s, x, y, width) }

// BlitRect implements Blitter.
func (s *spriteD32S32) BlitRect(x, y, width, height int32) {
	alpha256 := alpha255To256(s.alpha)
	for row := int32(0); row < height; row++ {
		d := s.dst.addr(x, y+row)
		sr := s.src.addr(x-s.left, y+row-s.top)
		dstRow := s.dst.Pix[d : d+int(width)]
		srcRow := s.src.Pix[sr : sr+int(width)]
		switch {
		case s.srcOpaque && s.alpha == 255:
			copy(dstRow, srcRow) // opaque source, full alpha: verbatim copy
		case s.srcOpaque:
			for i, sp := range srcRow { // opaque source, scaled by paint alpha
				dstRow[i] = fastFourByteInterp256(sp, dstRow[i], alpha256)
			}
		case s.alpha == 255:
			pmSrcOverRow(dstRow, srcRow)
		default:
			pmBlendRow(dstRow, srcRow, alpha256)
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

// ImageSpriteBlitter is the raster-pipeline-equivalent sprite blitter for an integer-translated N32 source: each
// destination pixel loads the corresponding source pixel, scales it by the paint alpha (lowp: div255(v*alpha); highp:
// float multiply), applies the blend mode, and handles coverage the same way the raster-pipeline blitter does
// (pre-scale vs lerp-after-blend). It supports the full Blitter interface, so it also stands in as the general
// drawRect-style fallback when an AA clip prevents the sprite fast path.
type ImageSpriteBlitter struct {
	dst      *Pixmap
	src      *Pixmap
	left     int32
	top      int32
	alpha    uint32 // paint alpha, 0..255
	mode     BlendMode
	lowp     bool
	prescale bool
}

// NewImageSpriteBlitter returns a raster-pipeline-equivalent sprite blitter.
func NewImageSpriteBlitter(dst, src *Pixmap, left, top int32, alpha uint8, mode BlendMode) *ImageSpriteBlitter {
	return &ImageSpriteBlitter{
		dst:      dst,
		src:      src,
		left:     left,
		top:      top,
		mode:     mode,
		alpha:    uint32(alpha),
		lowp:     !blendNeedsHighp(mode),
		prescale: blendShouldPreScaleCoverage(mode),
	}
}

// srcPM8 returns the source pixel for destination pixel (x, y), scaled by the paint alpha in the lowp form: alpha/255
// converts into the lowp fixed-point form as alpha exactly, so the scale is div255(v*alpha).
func (ib *ImageSpriteBlitter) srcPM8(x, y int32) pm8 {
	s := loadPM8(ib.src.Pix[ib.src.addr(x-ib.left, y-ib.top)])
	if ib.alpha != 255 {
		s = pm8{
			r: div255(s.r * ib.alpha),
			g: div255(s.g * ib.alpha),
			b: div255(s.b * ib.alpha),
			a: div255(s.a * ib.alpha),
		}
	}
	return s
}

func (ib *ImageSpriteBlitter) srcPM4f(x, y int32) pmColor4f {
	c := loadPM4f(ib.src.Pix[ib.src.addr(x-ib.left, y-ib.top)])
	if ib.alpha != 255 {
		f := float32(ib.alpha) * (1.0 / 255.0)
		c.r *= f
		c.g *= f
		c.b *= f
		c.a *= f
	}
	return c
}

// blendPixel blends the source pixel for (x, y) into the destination word at full coverage.
func (ib *ImageSpriteBlitter) blendPixel(x, y int32, px uint32) uint32 {
	if ib.mode == BlendSrc {
		if ib.alpha == 255 {
			return ib.src.Pix[ib.src.addr(x-ib.left, y-ib.top)]
		}
		return storePM8(ib.srcPM8(x, y))
	}
	if ib.lowp {
		return storePM8(blendLowp(ib.mode, ib.srcPM8(x, y), loadPM8(px)))
	}
	return storeWord(blendHighp(ib.mode, ib.srcPM4f(x, y), loadPM4f(px)))
}

// blendPixelCoverage blends with constant coverage aa (0 < aa < 255).
func (ib *ImageSpriteBlitter) blendPixelCoverage(x, y int32, px, aa uint32) uint32 {
	if ib.lowp {
		if ib.prescale {
			s := ib.srcPM8(x, y)
			s = pm8{r: div255(s.r * aa), g: div255(s.g * aa), b: div255(s.b * aa), a: div255(s.a * aa)}
			return storePM8(blendLowp(ib.mode, s, loadPM8(px)))
		}
		d := loadPM8(px)
		var r pm8
		if ib.mode == BlendSrc {
			r = ib.srcPM8(x, y)
		} else {
			r = blendLowp(ib.mode, ib.srcPM8(x, y), d)
		}
		return storePM8(pm8{
			r: lowpLerp(d.r, r.r, aa),
			g: lowpLerp(d.g, r.g, aa),
			b: lowpLerp(d.b, r.b, aa),
			a: lowpLerp(d.a, r.a, aa),
		})
	}
	c := float32(aa) * (1.0 / 255.0)
	d := loadPM4f(px)
	r := blendHighp(ib.mode, ib.srcPM4f(x, y), d)
	return storeWord(pmColor4f{
		r: lerpf(d.r, r.r, c),
		g: lerpf(d.g, r.g, c),
		b: lerpf(d.b, r.b, c),
		a: lerpf(d.a, r.a, c),
	})
}

// BlitH implements Blitter.
func (ib *ImageSpriteBlitter) BlitH(x, y, width int32) {
	start := ib.dst.addr(x, y)
	row := ib.dst.Pix[start : start+int(width)]
	for i := range row {
		row[i] = ib.blendPixel(x+int32(i), y, row[i])
	}
}

// BlitRect implements Blitter.
func (ib *ImageSpriteBlitter) BlitRect(x, y, width, height int32) {
	for row := int32(0); row < height; row++ {
		ib.BlitH(x, y+row, width)
	}
}

// BlitAntiH implements Blitter.
func (ib *ImageSpriteBlitter) BlitAntiH(x, y int32, antialias []Alpha, runs []int16) {
	i := 0
	for runs[i] != 0 {
		n := int32(runs[i])
		aa := uint32(antialias[i])
		switch aa {
		case 0:
		case 255:
			ib.BlitH(x, y, n)
		default:
			start := ib.dst.addr(x, y)
			row := ib.dst.Pix[start : start+int(n)]
			for j := range row {
				row[j] = ib.blendPixelCoverage(x+int32(j), y, row[j], aa)
			}
		}
		x += n
		i += int(n)
	}
}

// BlitV implements Blitter.
func (ib *ImageSpriteBlitter) BlitV(x, y, height int32, alpha Alpha) {
	aa := uint32(alpha)
	for row := int32(0); row < height; row++ {
		start := ib.dst.addr(x, y+row)
		if aa == 255 {
			ib.dst.Pix[start] = ib.blendPixel(x, y+row, ib.dst.Pix[start])
		} else {
			ib.dst.Pix[start] = ib.blendPixelCoverage(x, y+row, ib.dst.Pix[start], aa)
		}
	}
}

// BlitAntiRect implements Blitter.
func (ib *ImageSpriteBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	BlitAntiRectViaVH(ib, x, y, width, height, leftAlpha, rightAlpha)
}

// BlitMask implements Blitter.
func (ib *ImageSpriteBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	for y := clip.Top; y < clip.Bottom; y++ {
		mi := mask.addr8(clip.Left, y)
		aaRow := mask.Image[mi : mi+int(clip.Width())]
		for i, cov := range aaRow {
			x := clip.Left + int32(i)
			aa := uint32(cov)
			start := ib.dst.addr(x, y)
			if aa == 255 {
				ib.dst.Pix[start] = ib.blendPixel(x, y, ib.dst.Pix[start])
			} else {
				ib.dst.Pix[start] = ib.blendPixelCoverage(x, y, ib.dst.Pix[start], aa)
			}
		}
	}
}

// BlitAntiH2 implements Blitter.
func (ib *ImageSpriteBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(ib, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter.
func (ib *ImageSpriteBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(ib, x, y, a0, a1)
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

// ChooseSprite selects the fastest sprite blitter for the N32 destination/source pair: the memcpy sprite when the paint
// copies pixels verbatim (BlendSrc, or src-over with an opaque source), the legacy src-over sprite otherwise for
// src-over, and the raster-pipeline-equivalent sprite for everything else. left/top position the source in destination
// space; alpha and mode come from the paint; srcOpaque is the source image's alpha-type opacity.
func ChooseSprite(dst, src *Pixmap, left, top int32, alpha uint8, mode BlendMode, srcOpaque bool) Blitter {
	// Full alpha and BlendSrc, or src-over with an opaque source: pixels copy verbatim.
	if alpha == 0xFF && (mode == BlendSrc || (mode == BlendSrcOver && srcOpaque)) {
		return &spriteMemcpy{spriteBase: spriteBase{dst: dst, src: src, left: left, top: top}}
	}
	// src-over takes the legacy blitter (it handles alpha, but no other blend mode).
	if mode == BlendSrcOver {
		return &spriteD32S32{
			spriteBase: spriteBase{dst: dst, src: src, left: left, top: top},
			alpha:      uint32(alpha),
			srcOpaque:  srcOpaque,
		}
	}
	return NewImageSpriteBlitter(dst, src, left, top, alpha, mode)
}
