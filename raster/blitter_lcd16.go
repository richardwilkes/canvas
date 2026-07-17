// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The LCD16-over-8888 mask lanes: the D32 row procs blitRowLCD16 / blitRowLCD16Opaque that SolidBlitter.BlitMask routes
// solid src-over color through (portable scalar lanes; SIMD kernels reproduce the same results), the legacy
// shader-blitter's blendRowLCD16 / blendRowLCD16Opaque procs for shader spans, and the float per-channel
// scale_565/lerp_565 raster-pipeline stages the pipeline blitters lower LCD16 masks through. LCD blitting assumes an
// opaque destination throughout.

package raster

import "github.com/richardwilkes/canvas/geom"

// lcd16MaskChannels extracts a 565 mask's channels as 5-bit coverages (the green channel drops its low bit: "We're
// ignoring the least significant bit of the green coverage channel here").
func lcd16MaskChannels(m uint16) (r, g, b uint32) {
	return uint32(m>>11) & 0x1F, uint32(m>>6) & 0x1F, uint32(m) & 0x1F
}

// upscale31To32 maps 0..31 to 0..32 so blend32 can shift by 5.
func upscale31To32(v uint32) uint32 { return v + (v >> 4) }

// upscale31To255 converts 5-bit to 8-bit coverage by bit replication.
func upscale31To255(v uint32) uint32 { return v<<3 | v>>2 }

// blend32 computes dst + ((src-dst) * scale >> 5) with scale in [0, 32].
func blend32(src, dst, scale uint32) uint32 {
	return uint32(int32(dst) + (int32(src)-int32(dst))*int32(scale)>>5)
}

// blendLCD16 blends an unpremultiplied color (srcA upscaled to [0, 256]) into an opaque device word under a 565
// coverage mask.
func blendLCD16(srcA256, srcR, srcG, srcB, dst uint32, mask uint16) uint32 {
	if mask == 0 {
		return dst
	}
	maskR, maskG, maskB := lcd16MaskChannels(mask)
	// Upscale to 0..32, so we can use blend32.
	maskR = upscale31To32(maskR)
	maskG = upscale31To32(maskG)
	maskB = upscale31To32(maskB)
	maskR = maskR * srcA256 >> 8
	maskG = maskG * srcA256 >> 8
	maskB = maskB * srcA256 >> 8

	dstA := dst >> 24
	dstR := dst & 0xFF
	dstG := dst >> 8 & 0xFF
	dstB := dst >> 16 & 0xFF

	// Bring srcA back to [0, 255] to compare against dstA: alpha uses either the min or the max of the LCD coverages
	// (skbug.com/40037823).
	var maskA uint32
	if srcA256-1 < dstA {
		maskA = min(maskR, maskG, maskB)
	} else {
		maskA = max(maskR, maskG, maskB)
	}
	return blend32(srcR, dstR, maskR) |
		blend32(srcG, dstG, maskG)<<8 |
		blend32(srcB, dstB, maskB)<<16 |
		blend32(0xFF, dstA, maskA)<<24
}

// blendLCD16Opaque is the srcA == 255 specialization of blendLCD16 (opaqueDst is the premultiplied source device word).
func blendLCD16Opaque(srcR, srcG, srcB, dst uint32, mask uint16, opaqueDst uint32) uint32 {
	if mask == 0 {
		return dst
	}
	if mask == 0xFFFF {
		return opaqueDst
	}
	maskR, maskG, maskB := lcd16MaskChannels(mask)
	maskR = upscale31To32(maskR)
	maskG = upscale31To32(maskG)
	maskB = upscale31To32(maskB)

	dstA := dst >> 24
	dstR := dst & 0xFF
	dstG := dst >> 8 & 0xFF
	dstB := dst >> 16 & 0xFF

	// Opaque src alpha always uses the max of the LCD coverages.
	maskA := max(maskR, maskG, maskB)
	return blend32(srcR, dstR, maskR) |
		blend32(srcG, dstG, maskG)<<8 |
		blend32(srcB, dstB, maskB)<<16 |
		blend32(0xFF, dstA, maskA)<<24
}

// blitRowLCD16 is the general solid-color row proc. srcR/G/B are the unpremultiplied color channels, srcA its [0, 255]
// alpha.
func blitRowLCD16(dst []uint32, mask []uint16, srcA, srcR, srcG, srcB uint32) {
	srcA256 := alpha255To256(srcA)
	for i, m := range mask {
		dst[i] = blendLCD16(srcA256, srcR, srcG, srcB, dst[i], m)
	}
}

// blitRowLCD16Opaque is the opaque-source specialization of blitRowLCD16.
func blitRowLCD16Opaque(dst []uint32, mask []uint16, srcR, srcG, srcB, opaqueDst uint32) {
	for i, m := range mask {
		dst[i] = blendLCD16Opaque(srcR, srcG, srcB, dst[i], m, opaqueDst)
	}
}

// blitMaskLCD16 is the LCD16 mask lane over the row procs above: the SolidBlitter dispatches here.
func (s *SolidBlitter) blitMaskLCD16(mask *Mask, clip geom.IRect) {
	if s.srcA == 0 {
		return
	}
	srcR := uint32(s.color.R())
	srcG := uint32(s.color.G())
	srcB := uint32(s.color.B())
	opaque := s.srcA == 0xFF
	for y := clip.Top; y < clip.Bottom; y++ {
		di := s.dev.addr(clip.Left, y)
		mi := mask.addr16(clip.Left, y)
		w := int(clip.Width())
		dev := s.dev.Pix[di : di+w]
		mrow := mask.Image16[mi : mi+w]
		if opaque {
			blitRowLCD16Opaque(dev, mrow, srcR, srcG, srcB, s.pmColor)
		} else {
			blitRowLCD16(dev, mrow, s.srcA, srcR, srcG, srcB)
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// The legacy shader blitter's LCD16 blend rows (legacy shader spans are premultiplied device words).

// blendRowLCD16 is the non-opaque shader-span variant. srcAlphaBlend(s, d, sa, m) = d + alphaMul(s - alphaMul(sa, d),
// m) with 8-bit coverages; only works on an opaque destination.
func blendRowLCD16(dst []uint32, mask []uint16, src []uint32) {
	srcAlphaBlend := func(s, d, sa, m int32) int32 {
		return d + (s-sa*d>>8)*m>>8
	}
	for i, m := range mask {
		if m == 0 {
			continue
		}
		s := src[i]
		d := dst[i]
		srcA := int32(s >> 24)
		srcA += srcA >> 7 // upscale to [0, 256]
		maskR, maskG, maskB := lcd16MaskChannels(m)
		// Scale up to 8-bit coverage to work with the >>8 alpha multiplies in srcAlphaBlend.
		mr := int32(upscale31To255(maskR))
		mg := int32(upscale31To255(maskG))
		mb := int32(upscale31To255(maskB))
		dst[i] = uint32(srcAlphaBlend(int32(s&0xFF), int32(d&0xFF), srcA, mr)) |
			uint32(srcAlphaBlend(int32(s>>8&0xFF), int32(d>>8&0xFF), srcA, mg))<<8 |
			uint32(srcAlphaBlend(int32(s>>16&0xFF), int32(d>>16&0xFF), srcA, mb))<<16 |
			0xFF<<24
	}
}

// blendRowLCD16Opaque is the opaque shader-span variant (blend32 per channel, destination assumed opaque).
func blendRowLCD16Opaque(dst []uint32, mask []uint16, src []uint32) {
	for i, m := range mask {
		if m == 0 {
			continue
		}
		s := src[i]
		d := dst[i]
		maskR, maskG, maskB := lcd16MaskChannels(m)
		maskR = upscale31To32(maskR)
		maskG = upscale31To32(maskG)
		maskB = upscale31To32(maskB)
		dst[i] = blend32(s&0xFF, d&0xFF, maskR) |
			blend32(s>>8&0xFF, d>>8&0xFF, maskG)<<8 |
			blend32(s>>16&0xFF, d>>16&0xFF, maskB)<<16 |
			0xFF<<24
	}
}

///////////////////////////////////////////////////////////////////////////////
// Raster-pipeline 565 stages (highp): the pipeline blitters' LCD16 mask lanes.

// from565f converts a 565 mask value to per-channel float coverage.
func from565f(m uint16) (cr, cg, cb float32) {
	w := uint32(m)
	cr = float32(w&(31<<11)) * (1.0 / float32(31<<11))
	cg = float32(w&(63<<5)) * (1.0 / float32(63<<5))
	cb = float32(w&31) * (1.0 / float32(31))
	return cr, cg, cb
}

// alphaCoverageFromRGBCoverage derives alpha's coverage from the rgb coverages and the values of src and dst alpha.
func alphaCoverageFromRGBCoverage(a, da, cr, cg, cb float32) float32 {
	if a < da {
		return min(cr, cg, cb)
	}
	return max(cr, cg, cb)
}

// blendShouldPreScaleCoverageRGB reports whether mode supports per-channel (RGB) coverage scaling: per-channel coverage
// scaling destroys the source-alpha term, so only the modes without one qualify.
func blendShouldPreScaleCoverageRGB(mode BlendMode) bool {
	switch mode {
	case BlendDst, BlendDstOver, BlendPlus:
		return true
	default:
		return false
	}
}
