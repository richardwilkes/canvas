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

// color32 computes dst = color + dst*(256 - srcA)>>8 per channel, with the 0 and 255 alpha cases specialized. Both
// non-trivial lanes run through the dispatch variables, so a vector build blits the whole span in its kernel.
func color32(dst []uint32, color uint32) {
	switch deviceAlpha(color) {
	case 0: // nothing to do
		return
	case 255:
		fillWordsFn(dst, color)
		return
	}
	color32RowFn(dst, color, alpha255To256(255-deviceAlpha(color)))
}

// fillWordsGeneric fills a device span with one word. It is the default fillWordsFn; where a vector kernel is wired
// instead this remains the sub-chunk tail that kernel calls.
func fillWordsGeneric(dst []uint32, v uint32) {
	for i := range dst {
		dst[i] = v
	}
}

// color32RowGeneric is the portable general lane of color32: every channel of dst is scaled by invA (a 0..256 scale,
// here 256 minus the color's alpha) with the +1/256 trick, then the premultiplied color — whose channels are already
// scaled by that alpha — is added to the packed result. It is the default color32RowFn; where a vector kernel is wired
// instead this remains the sub-chunk tail that kernel calls.
//
// The scale-and-pack half is alphaMulQ(d, invA) written out: (d>>8 & mask) * invA >> 8 & mask, shifted back up by 8, is
// the same bit field alphaMulQ keeps with &^mask. The final add is a plain 32-bit add, as in the reference; for a
// premultiplied color no channel sum can reach 256, so nothing carries across channels.
func color32RowGeneric(dst []uint32, color, invA uint32) {
	for i, d := range dst {
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
			fillWordsFn(s.row(x, row, width), s.pmColor)
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
	off := 0
	dev := s.dev.Pix[s.dev.addr(x, y):]
	for {
		count := int(runs[off])
		if count <= 0 {
			return
		}
		// Black premultiplied by coverage is aa<<24, and color32's three lanes are exactly this run's three cases:
		// coverage 0 skips the run, coverage 255 fills it with opaque black, and the general lane's
		// (rb | ag<<8) + color is alphaMulQ(dev[i], 256-aa) + aa<<24 term for term (see color32RowGeneric).
		color32(dev[off:off+count], uint32(antialias[off])<<deviceAlphaShift)
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
			fillWordsFn(dev[off:off+count], s.pmColor)
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

// interp256RowGeneric is the portable row form of fastFourByteInterp256: per channel (src*scale + dst*(256-scale)) >> 8
// for a loop-invariant scale in [0, 256]. It is the default interp256RowFn; where a vector kernel is wired instead this
// remains the sub-chunk tail that kernel calls. Every channel result is a convex combination of two bytes, so it stays
// inside a byte and the packed OR carries nowhere.
func interp256RowGeneric(dst, src []uint32, scale uint32) {
	for i, s := range src {
		dst[i] = fastFourByteInterp256(s, dst[i], scale)
	}
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
// per pixel. It is the default blitMaskOpaqueRowFn; where a vector kernel is wired instead (NEON on arm64, archsimd
// under goexperiment.simd) this remains the sub-quad tail those kernels call.
func blitMaskOpaqueRowGeneric(dev []uint32, aa []uint8, pm uint32) {
	for i, m := range aa {
		dev[i] = alphaMulQ(pm, alpha255To256(uint32(m))) + alphaMulQ(dev[i], 256-uint32(m))
	}
}

// blitMaskTranslucentRowGeneric is the portable non-opaque solid-color A8 mask blend: per pixel dev = alphaMulQ(pm,
// m+1) + alphaMulQ(dev, 256 - ((srcA*(m+1))>>8)), the general lane of SolidBlitter.BlitMask. It is the default
// blitMaskTranslucentRowFn; where a vector kernel is wired instead this remains the sub-chunk tail that kernel calls.
// srcA must be pm's alpha in [0, 255], which is what makes both alphaMulQ scales land in [1, 256] and every channel sum
// stay inside a byte.
func blitMaskTranslucentRowGeneric(dev []uint32, aa []uint8, pm, srcA uint32) {
	for i, m := range aa {
		m256 := alpha255To256(uint32(m))
		scale := 256 - alphaMul(srcA, m256)
		dev[i] = alphaMulQ(pm, m256) + alphaMulQ(dev[i], scale)
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
			// exactly), so both opaque lanes run the same row kernel — whichever one the build wired into
			// blitMaskOpaqueRowFn (see span.go); they are all bit-identical.
			blitMaskOpaqueRowFn(dev, aa, s.pmColor)
		default:
			blitMaskTranslucentRowFn(dev, aa, s.pmColor, s.srcA)
		}
	}
}
