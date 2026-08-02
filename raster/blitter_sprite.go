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
// The destination is always premultiplied, so a straight-alpha source (SpriteAlphaUnpremul) is premultiplied on the way
// in — a row at a time for the legacy src-over sprite, per pixel for the raster-pipeline sprite — rather than being
// composited as if its bytes were already premultiplied. Premultiplying in the blitter keeps such sources on the fast
// path; the verbatim-copy sprite is the one lane they cannot take.
//
// Byte-exactness note: the src-over row kernels reproduce the NEON forms that the darwin/arm64 oracle leg executes for
// every pixel of a row (including odd-length tails). The x86 SSE2 bodies compute s + ((d*(256-sa))>>8) where NEON
// computes s + mulDiv255Round(d, 255-sa), and x86 rows shorter than the vector width take scalar tails with a third
// rounding — the same class of per-leg ±1 divergence already recorded for the lowp pipeline.

package raster

import "github.com/richardwilkes/canvas/geom"

// SpriteAlphaType classifies how a sprite source's alpha channel relates to its color channels, which decides both
// which blitter can consume the source words verbatim and whether they must be premultiplied first. raster cannot name
// imagecore's AlphaType (imagecore is the layer above), so callers translate into this.
type SpriteAlphaType uint8

// SpriteAlphaType values.
const (
	// SpriteAlphaPremul is a source whose color channels are already multiplied by its alpha — the destination's own
	// form.
	SpriteAlphaPremul SpriteAlphaType = iota
	// SpriteAlphaOpaque is a source with no transparency at all, so its premultiplied and unpremultiplied forms
	// coincide.
	SpriteAlphaOpaque
	// SpriteAlphaUnpremul is a straight-alpha source: its color channels must be premultiplied before compositing onto
	// the premultiplied destination.
	SpriteAlphaUnpremul
)

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

// premul8888 premultiplies a straight-alpha 8888 word: every color channel is scaled by the pixel's alpha with the
// round-half-up div255 (mulDiv255Round), the integer form of the shader lane's premul stage, so both lanes agree on the
// bytes an unpremultiplied source contributes. The channels run in spread64 lanes with the same headroom argument as
// pmSrcOverRowGeneric (lane max 255*255+128 = 0xFE81, then +254 = 0xFF7F, so nothing carries across lanes); the alpha
// lane's product is discarded, since premultiplication leaves alpha alone.
func premul8888(c uint32) uint32 {
	a := c >> 24
	if a == 0xFF {
		return c
	}
	const half = uint64(0x0080008000800080)
	prod := spread64(c)*uint64(a) + half
	prod += (prod >> 8) & swarMask8
	return pack64((prod>>8)&swarMask8)&0x00FFFFFF | a<<24
}

// premulRow premultiplies a row of straight-alpha words into dst, which must be at least as long as src.
func premulRow(dst, src []uint32) {
	for i, s := range src {
		dst[i] = premul8888(s)
	}
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
// procs depending on whether the source is opaque and whether the paint alpha is 255. A straight-alpha source is
// premultiplied one row at a time into scratch and then handed to those same row procs, so it yields exactly the bytes
// a caller that premultiplied its pixels up front would have gotten.
type spriteD32S32 struct {
	spriteBase
	scratch     []uint32 // the premultiplied source row; allocated on first use, only for a straight-alpha source
	alpha       uint32   // paint alpha, 0..255
	srcOpaque   bool
	srcUnpremul bool
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
		if s.srcUnpremul {
			// Premultiply the row into scratch, so the row procs below see the destination's own form.
			if len(s.scratch) < int(width) {
				s.scratch = make([]uint32, width)
			}
			premulRow(s.scratch[:width], srcRow)
			srcRow = s.scratch[:width]
		}
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
// destination pixel loads the corresponding source pixel, premultiplies it if the source is straight-alpha, scales it
// by the paint alpha (lowp: div255(v*alpha); highp: float multiply), applies the blend mode, and handles coverage the
// same way the raster-pipeline blitter does (pre-scale vs lerp-after-blend). It supports the full Blitter interface, so
// it also stands in as the general drawRect-style fallback when an AA clip prevents the sprite fast path.
type ImageSpriteBlitter struct {
	dst         *Pixmap
	src         *Pixmap
	left        int32
	top         int32
	alpha       uint32 // paint alpha, 0..255
	mode        BlendMode
	lowp        bool
	prescale    bool
	srcUnpremul bool
}

// NewImageSpriteBlitter returns a raster-pipeline-equivalent sprite blitter. srcAlpha classifies the source's alpha
// channel; a SpriteAlphaUnpremul source is premultiplied as each pixel is loaded.
func NewImageSpriteBlitter(dst, src *Pixmap, left, top int32, alpha uint8, mode BlendMode, srcAlpha SpriteAlphaType) *ImageSpriteBlitter {
	return &ImageSpriteBlitter{
		dst:         dst,
		src:         src,
		left:        left,
		top:         top,
		mode:        mode,
		alpha:       uint32(alpha),
		lowp:        !blendNeedsHighp(mode),
		prescale:    blendShouldPreScaleCoverage(mode),
		srcUnpremul: srcAlpha == SpriteAlphaUnpremul,
	}
}

// srcWord returns the premultiplied source word for destination pixel (x, y), premultiplying in the lowp form
// (div255(v*a), premul8888) when the source carries straight alpha.
func (ib *ImageSpriteBlitter) srcWord(x, y int32) uint32 {
	s := ib.src.Pix[ib.src.addr(x-ib.left, y-ib.top)]
	if ib.srcUnpremul {
		s = premul8888(s)
	}
	return s
}

// srcPM8 returns the source pixel for destination pixel (x, y), scaled by the paint alpha in the lowp form: alpha/255
// converts into the lowp fixed-point form as alpha exactly, so the scale is div255(v*alpha).
func (ib *ImageSpriteBlitter) srcPM8(x, y int32) pm8 {
	s := loadPM8(ib.srcWord(x, y))
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
	if ib.srcUnpremul {
		// The highp lane premultiplies in float, matching the raster pipeline's premul stage, rather than rounding
		// through 8 bits first.
		c.r *= c.a
		c.g *= c.a
		c.b *= c.a
	}
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
			return ib.srcWord(x, y)
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
// space; alpha and mode come from the paint; srcAlpha classifies the source image's alpha channel. A straight-alpha
// source is premultiplied inside whichever blitter runs, so the only lane it cannot take is the verbatim copy — its
// bytes are not yet in the destination's premultiplied form.
func ChooseSprite(dst, src *Pixmap, left, top int32, alpha uint8, mode BlendMode, srcAlpha SpriteAlphaType) Blitter {
	// Full alpha and BlendSrc, or src-over with an opaque source: premultiplied pixels copy verbatim.
	if alpha == 0xFF && srcAlpha != SpriteAlphaUnpremul &&
		(mode == BlendSrc || (mode == BlendSrcOver && srcAlpha == SpriteAlphaOpaque)) {
		return &spriteMemcpy{spriteBase: spriteBase{dst: dst, src: src, left: left, top: top}}
	}
	// src-over takes the legacy blitter (it handles alpha, but no other blend mode).
	if mode == BlendSrcOver {
		return &spriteD32S32{
			spriteBase:  spriteBase{dst: dst, src: src, left: left, top: top},
			alpha:       uint32(alpha),
			srcOpaque:   srcAlpha == SpriteAlphaOpaque,
			srcUnpremul: srcAlpha == SpriteAlphaUnpremul,
		}
	}
	return NewImageSpriteBlitter(dst, src, left, top, alpha, mode, srcAlpha)
}
