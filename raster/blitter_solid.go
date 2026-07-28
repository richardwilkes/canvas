// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The solid-color legacy N32 blitters: for src-over solid-color paints into N32 devices, integer-math blitters are used
// rather than the float raster pipeline, since exact integer math is required to stay byte-exact with the oracle
// reference.

package raster

import (
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// SolidBlitter blits a constant premultiplied color with src-over semantics into a Pixmap, using exact integer math.
type SolidBlitter struct {
	dev     *Pixmap
	pmColor uint32 // device word
	srcA    uint32
	color   colorcore.Color // the unpremultiplied paint color (the LCD16 row procs blend from it)
	isBlack bool
}

// A SolidBlitter is built fresh for every solid-color draw, so pooling it keeps steady-state fills from allocating one
// at the blitter layer. It holds only scalar state plus the destination Pixmap pointer, so init fully overwrites a
// reused instance. A sync.Pool keeps concurrent FillPathParallel bands independent; each is fully consumed by the
// synchronous fill before RecycleSolidBlitter returns it. Callers that do not recycle (e.g. one-off glyph fills) simply
// get a fresh instance next time — missing a recycle is safe.
var solidBlitterPool = sync.Pool{New: func() any { return new(SolidBlitter) }}

// NewSolidBlitter returns a SolidBlitter for the given unpremultiplied color, choosing the black / opaque / general
// specialization based on the color.
func NewSolidBlitter(dev *Pixmap, color colorcore.Color) *SolidBlitter {
	pm := color.PreMultiply()
	s := solidBlitterPool.Get().(*SolidBlitter)
	s.dev = dev
	s.pmColor = deviceRGBA(pm)
	s.srcA = uint32(color.A())
	s.color = color
	s.isBlack = color == 0xFF000000
	return s
}

// RecycleSolidBlitter returns a SolidBlitter from NewSolidBlitter to the pool once the synchronous fill that used it
// (including all FillPathParallel bands) has completed. It must not be called on a blitter still in use or retained
// elsewhere.
func RecycleSolidBlitter(s *SolidBlitter) {
	s.dev = nil // drop the device reference so the pool does not pin the Pixmap
	solidBlitterPool.Put(s)
}

// color32 computes dst = color + dst*(256 - srcA)>>8 per channel, with the 0 and 255 alpha cases specialized.
func color32(dst []uint32, color uint32) {
	switch deviceAlpha(color) {
	case 0: // nothing to do
		return
	case 255:
		for i := range dst {
			dst[i] = color
		}
		return
	}
	invA := alpha255To256(255 - deviceAlpha(color))
	for i, d := range dst {
		// color is premul, so the channels have already been scaled by alpha. We just need to scale dst by (255 - a)
		// using the +1/256 trick and add.
		const mask = 0x00FF00FF
		rb := ((d & mask) * invA) >> 8 & mask
		ag := ((d >> 8 & mask) * invA) >> 8 & mask
		dst[i] = (rb | ag<<8) + color
	}
}

// row returns the device span [x, x+width) on row y.
func (s *SolidBlitter) row(x, y, width int32) []uint32 {
	start := s.dev.addr(x, y)
	return s.dev.Pix[start : start+int(width)]
}

// BlitH implements Blitter.
func (s *SolidBlitter) BlitH(x, y, width int32) {
	color32(s.row(x, y, width), s.pmColor)
}

// BlitRect implements Blitter.
func (s *SolidBlitter) BlitRect(x, y, width, height int32) {
	if s.srcA == 0 {
		return
	}
	if deviceAlpha(s.pmColor) == 0xFF {
		for row := y; row < y+height; row++ {
			span := s.row(x, row, width)
			for i := range span {
				span[i] = s.pmColor
			}
		}
		return
	}
	for row := y; row < y+height; row++ {
		color32(s.row(x, row, width), s.pmColor)
	}
}

// BlitV implements Blitter.
func (s *SolidBlitter) BlitV(x, y, height int32, alpha Alpha) {
	if alpha == 0 || s.srcA == 0 {
		return
	}
	color := s.pmColor
	if alpha != 255 {
		color = alphaMulQ(color, alpha255To256(uint32(alpha)))
	}
	dstScale := alpha255To256(255 - deviceAlpha(color))
	for row := y; row < y+height; row++ {
		i := s.dev.addr(x, row)
		s.dev.Pix[i] = color + alphaMulQ(s.dev.Pix[i], dstScale)
	}
}

// BlitAntiH implements Blitter, dispatching to the black, opaque, or general specialization chosen at construction.
func (s *SolidBlitter) BlitAntiH(x, y int32, antialias []Alpha, runs []int16) {
	switch {
	case s.isBlack:
		s.blitAntiHBlack(x, y, antialias, runs)
	case s.srcA == 0xFF:
		s.blitAntiHOpaque(x, y, antialias, runs)
	default:
		s.blitAntiHGeneral(x, y, antialias, runs)
	}
}

func (s *SolidBlitter) blitAntiHBlack(x, y int32, antialias []Alpha, runs []int16) {
	const black = uint32(0xFF) << deviceAlphaShift
	off := 0
	dev := s.dev.Pix[s.dev.addr(x, y):]
	for {
		count := int(runs[off])
		if count <= 0 {
			return
		}
		aa := uint32(antialias[off])
		if aa != 0 {
			if aa == 255 {
				for i := off; i < off+count; i++ {
					dev[i] = black
				}
			} else {
				src := aa << deviceAlphaShift
				dstScale := alpha255To256(255 - aa)
				for i := off; i < off+count; i++ {
					dev[i] = src + alphaMulQ(dev[i], dstScale)
				}
			}
		}
		off += count
	}
}

func (s *SolidBlitter) blitAntiHOpaque(x, y int32, antialias []Alpha, runs []int16) {
	off := 0
	dev := s.dev.Pix[s.dev.addr(x, y):]
	for {
		count := int(runs[off])
		if count <= 0 {
			return
		}
		aa := antialias[off]
		if aa == 255 {
			for i := off; i < off+count; i++ {
				dev[i] = s.pmColor
			}
		} else if aa > 0 {
			sc := alphaMulQ(s.pmColor, alpha255To256(uint32(aa)))
			color32(dev[off:off+count], sc)
		}
		off += count
	}
}

func (s *SolidBlitter) blitAntiHGeneral(x, y int32, antialias []Alpha, runs []int16) {
	if s.srcA == 0 {
		return
	}
	off := 0
	dev := s.dev.Pix[s.dev.addr(x, y):]
	for {
		count := int(runs[off])
		if count <= 0 {
			return
		}
		if aa := antialias[off]; aa != 0 {
			sc := alphaMulQ(s.pmColor, alpha255To256(uint32(aa)))
			color32(dev[off:off+count], sc)
		}
		off += count
	}
}

// BlitAntiRect implements Blitter via the default lowering.
func (s *SolidBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	BlitAntiRectViaVH(s, x, y, width, height, leftAlpha, rightAlpha)
}

// alphaMul computes (value * alpha256) >> 8.
func alphaMul(value, alpha256 uint32) uint32 {
	return (value * alpha256) >> 8
}

// alphaMulInv256 returns the 256-based inverse-coverage multiplier blendARGB32 scales the destination by: it computes
// 0xFFFF - value*alpha256 (in [255, 0xFFFF] for value in [0, 255] and alpha256 in [0, 256]) and divides by 255 with the
// usual (x + (x>>8)) >> 8 approximation, giving 256 - value*alpha256/255 in [0, 256] — 256 when the source contributes
// nothing, 0 for a fully opaque source at full coverage. Note the numerator is 0xFFFF = 257*255, not 255*255; the
// approximation falls one short of an exact divide only at that top end, which is what keeps the result within 256.
func alphaMulInv256(value, alpha256 uint32) uint32 {
	prod := 0xFFFF - value*alpha256
	return (prod + (prod >> 8)) >> 8
}

// blendARGB32 blends src over dst with coverage aa, computed on the packed ARGB32 device word.
func blendARGB32(src, dst, aa uint32) uint32 {
	srcScale := alpha255To256(aa)
	dstScale := alphaMulInv256(deviceAlpha(src), srcScale)

	const mask = 0x00FF00FF
	srcRB := (src & mask) * srcScale
	srcAG := ((src >> 8) & mask) * srcScale
	dstRB := (dst & mask) * dstScale
	dstAG := ((dst >> 8) & mask) * dstScale
	return (((srcRB + dstRB) >> 8) & mask) | ((srcAG + dstAG) &^ mask)
}

// fastFourByteInterp256 splays src and dst into 16-bit lanes of one 64-bit word and computes four 8-bit channel blends
// at once.
func fastFourByteInterp256(src, dst, scale uint32) uint32 {
	splay := func(c uint32) uint64 {
		const mask = 0x00FF00FF
		return uint64((c>>8)&mask)<<32 | uint64(c&mask)
	}
	agrb := splay(src)*uint64(scale) + uint64(256-scale)*splay(dst)
	const outMask = 0xFF00FF00
	return uint32((agrb&outMask)>>8) | uint32(agrb>>32)&outMask
}

// fastFourByteInterp blends src and dst by srcWeight; scale = srcWeight + (srcWeight >> 7) is more accurate than
// srcWeight + 1.
func fastFourByteInterp(src, dst, srcWeight uint32) uint32 {
	return fastFourByteInterp256(src, dst, srcWeight+(srcWeight>>7))
}

// BlitAntiH2 implements Blitter, dispatching to the black, opaque, or general specialization.
func (s *SolidBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	i := s.dev.addr(x, y)
	dev := s.dev.Pix
	switch {
	case s.isBlack:
		dev[i] = uint32(a0)<<deviceAlphaShift + alphaMulQ(dev[i], 256-uint32(a0))
		dev[i+1] = uint32(a1)<<deviceAlphaShift + alphaMulQ(dev[i+1], 256-uint32(a1))
	case s.srcA == 0xFF:
		dev[i] = fastFourByteInterp(s.pmColor, dev[i], uint32(a0))
		dev[i+1] = fastFourByteInterp(s.pmColor, dev[i+1], uint32(a1))
	default:
		dev[i] = blendARGB32(s.pmColor, dev[i], uint32(a0))
		dev[i+1] = blendARGB32(s.pmColor, dev[i+1], uint32(a1))
	}
}

// BlitAntiV2 implements Blitter, dispatching to the black, opaque, or general specialization.
func (s *SolidBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	i := s.dev.addr(x, y)
	j := s.dev.addr(x, y+1)
	dev := s.dev.Pix
	switch {
	case s.isBlack:
		dev[i] = uint32(a0)<<deviceAlphaShift + alphaMulQ(dev[i], 256-uint32(a0))
		dev[j] = uint32(a1)<<deviceAlphaShift + alphaMulQ(dev[j], 256-uint32(a1))
	case s.srcA == 0xFF:
		dev[i] = fastFourByteInterp(s.pmColor, dev[i], uint32(a0))
		dev[j] = fastFourByteInterp(s.pmColor, dev[j], uint32(a1))
	default:
		dev[i] = blendARGB32(s.pmColor, dev[i], uint32(a0))
		dev[j] = blendARGB32(s.pmColor, dev[j], uint32(a1))
	}
}

// blitMaskOpaqueRowGeneric is the portable opaque-color A8 mask blend: dev = alphaMulQ(pm, m+1) + alphaMulQ(dev, 256-m)
// per pixel. On arm64 blitMaskOpaqueRow runs the NEON kernel instead and uses this only for the sub-quad tail.
func blitMaskOpaqueRowGeneric(dev []uint32, aa []uint8, pm uint32) {
	for i, m := range aa {
		dev[i] = alphaMulQ(pm, alpha255To256(uint32(m))) + alphaMulQ(dev[i], 256-uint32(m))
	}
}

// BlitMask implements Blitter. The per-pixel math below matches both the NEON and the SSE lanes of the equivalent SIMD
// kernels (their scalar tails and vector lanes compute the same integer expressions), so this is byte-exact against
// both oracle unix legs. The general (translucent) case returns early when srcA == 0.
func (s *SolidBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	if mask.Format == MaskLCD16 {
		s.blitMaskLCD16(mask, clip)
		return
	}
	if s.srcA == 0 && !s.isBlack {
		return
	}
	for y := clip.Top; y < clip.Bottom; y++ {
		di := s.dev.addr(clip.Left, y)
		mi := mask.addr8(clip.Left, y)
		w := int(clip.Width())
		dev := s.dev.Pix[di : di+w]
		aa := mask.Image[mi : mi+w]
		switch {
		case s.isBlack, s.srcA == 0xFF:
			// The black lane's uint32(m)<<24 equals alphaMulQ(0xFF000000, m+1) for every m ((255*(m+1))>>8 == m
			// exactly), so both opaque lanes run the same row kernel — NEON on arm64 (span_arm64.s), the portable loop
			// elsewhere.
			blitMaskOpaqueRow(dev, aa, s.pmColor)
		default:
			for i, m := range aa {
				m256 := alpha255To256(uint32(m))
				scale := 256 - alphaMul(s.srcA, m256)
				dev[i] = alphaMulQ(s.pmColor, m256) + alphaMulQ(dev[i], scale)
			}
		}
	}
}
