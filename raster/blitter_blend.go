// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Solid-color blitter for arbitrary blend modes: the pipeline stages assembled for a constant color into an 8888
// surface (src-over solid colors take the legacy SolidBlitter instead, since it is faster for that common case):
//
//	blitRect:  uniform_color [clamp_01] (load_dst + blend | nothing for kSrc) store    (memset for kSrc)
//	blitAntiH: uniform_color [clamp_01] scale(coverage)? load_dst blend lerp(coverage)? store
//	blitMask:  the same, with scale_u8/lerp_u8 coverage
//
// The lowp/highp split and rounding notes live in blend.go.

package raster

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// BlendBlitter blits a constant color with an arbitrary blend mode into a Pixmap.
type BlendBlitter struct {
	dev      *Pixmap
	sc       pm8       // lowp uniform color (0..255 in 16-bit lanes)
	cf       pmColor4f // highp uniform color
	srcWord  uint32    // store_8888 of cf, for the kSrc memset path
	mode     BlendMode
	lowp     bool
	prescale bool
}

// NewBlendBlitter returns a blitter for the given unpremultiplied color and blend mode.
func NewBlendBlitter(dev *Pixmap, color colorcore.Color, mode BlendMode) *BlendBlitter {
	// Converting the 8-bit color to float is an exact /255; premultiplication happens in float.
	const inv255f = float32(1.0 / 255.0)
	a := float32(color.A()) * inv255f
	cf := pmColor4f{
		r: float32(color.R()) * inv255f * a,
		g: float32(color.G()) * inv255f * a,
		b: float32(color.B()) * inv255f * a,
		a: a,
	}
	// The lowp lanes: (uint16)(v*255 + 0.5)
	sc := pm8{
		r: uint32(cf.r*255 + 0.5),
		g: uint32(cf.g*255 + 0.5),
		b: uint32(cf.b*255 + 0.5),
		a: uint32(cf.a*255 + 0.5),
	}
	return &BlendBlitter{
		dev:      dev,
		mode:     mode,
		sc:       sc,
		cf:       cf,
		srcWord:  storeWord(cf),
		lowp:     !blendNeedsHighp(mode),
		prescale: blendShouldPreScaleCoverage(mode),
	}
}

// toUnorm computes round(min(max(0, v*255), 255)), with round-to-nearest-even to match the SIMD lanes.
func toUnorm(v float32) uint32 {
	f := v * 255
	if !(f > 0) { // also catches NaN
		return 0
	}
	if f > 255 {
		f = 255
	}
	return uint32(math.RoundToEven(float64(f)))
}

// storeWord packs a float premultiplied color into an 8888 device word.
func storeWord(c pmColor4f) uint32 {
	return toUnorm(c.r) | toUnorm(c.g)<<8 | toUnorm(c.b)<<16 | toUnorm(c.a)<<24
}

// loadPM8 unpacks an 8888 device word into the lowp channel representation.
func loadPM8(px uint32) pm8 {
	return pm8{r: px & 0xFF, g: px >> 8 & 0xFF, b: px >> 16 & 0xFF, a: px >> 24}
}

func storePM8(c pm8) uint32 {
	return c.r | c.g<<8 | c.b<<16 | c.a<<24
}

// loadPM4f unpacks an 8888 device word into the highp float channel representation.
func loadPM4f(px uint32) pmColor4f {
	const inv255f = float32(1.0 / 255.0)
	return pmColor4f{
		r: float32(px&0xFF) * inv255f,
		g: float32(px>>8&0xFF) * inv255f,
		b: float32(px>>16&0xFF) * inv255f,
		a: float32(px>>24) * inv255f,
	}
}

// lerpf computes the highp lerp: mad(to-from, t, from).
func lerpf(from, to, t float32) float32 {
	return fmaf(to-from, t, from)
}

// blendSpan blends the uniform color into the span at full coverage.
func (bb *BlendBlitter) blendSpan(span []uint32) {
	if bb.mode == BlendSrc {
		for i := range span {
			span[i] = bb.srcWord
		}
		return
	}
	if bb.lowp {
		for i, px := range span {
			span[i] = storePM8(blendLowp(bb.mode, bb.sc, loadPM8(px)))
		}
		return
	}
	for i, px := range span {
		d := loadPM4f(px)
		r := blendHighpAll(bb.mode, bb.cf, d)
		span[i] = storeWord(r)
	}
}

// blendSpanCoverage blends the uniform color into the span at constant coverage aa (0 < aa < 255).
func (bb *BlendBlitter) blendSpanCoverage(span []uint32, aa uint32) {
	if bb.lowp {
		if bb.prescale {
			// scale_1_float: from_float(aa/255) reproduces aa exactly, so src scales by div255(v*aa).
			s := pm8{
				r: div255(bb.sc.r * aa),
				g: div255(bb.sc.g * aa),
				b: div255(bb.sc.b * aa),
				a: div255(bb.sc.a * aa),
			}
			for i, px := range span {
				span[i] = storePM8(blendLowp(bb.mode, s, loadPM8(px)))
			}
			return
		}
		for i, px := range span {
			d := loadPM8(px)
			r := blendLowp(bb.mode, bb.sc, d)
			span[i] = storePM8(pm8{
				r: lowpLerp(d.r, r.r, aa),
				g: lowpLerp(d.g, r.g, aa),
				b: lowpLerp(d.b, r.b, aa),
				a: lowpLerp(d.a, r.a, aa),
			})
		}
		return
	}
	// The highp-only modes never pre-scale coverage; the pre-scaling modes reach highp only via the unbounded-constant
	// collapse (NewBlendBlitterPM4f), where the C pipeline still folds coverage.
	c := float32(aa) * (1.0 / 255.0)
	if bb.prescale {
		s := pmColor4f{r: bb.cf.r * c, g: bb.cf.g * c, b: bb.cf.b * c, a: bb.cf.a * c}
		for i, px := range span {
			d := loadPM4f(px)
			span[i] = storeWord(blendHighpAll(bb.mode, s, d))
		}
		return
	}
	for i, px := range span {
		d := loadPM4f(px)
		r := blendHighpAll(bb.mode, bb.cf, d)
		span[i] = storeWord(pmColor4f{
			r: lerpf(d.r, r.r, c),
			g: lerpf(d.g, r.g, c),
			b: lerpf(d.b, r.b, c),
			a: lerpf(d.a, r.a, c),
		})
	}
}

func (bb *BlendBlitter) row(x, y, width int32) []uint32 {
	start := bb.dev.addr(x, y)
	return bb.dev.Pix[start : start+int(width)]
}

// BlitH implements Blitter.
func (bb *BlendBlitter) BlitH(x, y, width int32) {
	bb.blendSpan(bb.row(x, y, width))
}

// BlitRect implements Blitter.
func (bb *BlendBlitter) BlitRect(x, y, width, height int32) {
	for r := y; r < y+height; r++ {
		bb.blendSpan(bb.row(x, r, width))
	}
}

// BlitAntiH implements Blitter, walking the alpha runs and dispatching each to the full-coverage or partial-coverage
// span blender.
func (bb *BlendBlitter) BlitAntiH(x, y int32, antialias []Alpha, runs []int16) {
	off := 0
	for {
		run := int32(runs[off])
		if run <= 0 {
			return
		}
		switch aa := antialias[off]; aa {
		case 0x00:
		case 0xFF:
			bb.blendSpan(bb.row(x, y, run))
		default:
			bb.blendSpanCoverage(bb.row(x, y, run), uint32(aa))
		}
		x += run
		off += int(run)
	}
}

// BlitV implements Blitter via the default lowering.
func (bb *BlendBlitter) BlitV(x, y, height int32, alpha Alpha) {
	BlitVViaAntiH(bb, x, y, height, alpha)
}

// BlitAntiRect implements Blitter via the default lowering.
func (bb *BlendBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	BlitAntiRectViaVH(bb, x, y, width, height, leftAlpha, rightAlpha)
}

// BlitMask implements Blitter for A8 masks (the scale_u8/lerp_u8 pipeline stages) and LCD16 masks (scale_565/lerp_565
// with the rgb-coverage prescale rule). Note there is no BlendSrc shortcut here — even Src runs load_dst + lerp.
func (bb *BlendBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	if mask.Format == MaskLCD16 {
		bb.blitMaskLCD16(mask, clip)
		return
	}
	for y := clip.Top; y < clip.Bottom; y++ {
		span := bb.row(clip.Left, y, clip.Width())
		mi := mask.addr8(clip.Left, y)
		aa := mask.Image[mi : mi+int(clip.Width())]
		if bb.lowp {
			for i, cov := range aa {
				d := loadPM8(span[i])
				c := uint32(cov)
				if bb.prescale {
					s := pm8{
						r: div255(bb.sc.r * c),
						g: div255(bb.sc.g * c),
						b: div255(bb.sc.b * c),
						a: div255(bb.sc.a * c),
					}
					span[i] = storePM8(blendLowp(bb.mode, s, d))
					continue
				}
				r := blendLowp(bb.mode, bb.sc, d)
				span[i] = storePM8(pm8{
					r: lowpLerp(d.r, r.r, c),
					g: lowpLerp(d.g, r.g, c),
					b: lowpLerp(d.b, r.b, c),
					a: lowpLerp(d.a, r.a, c),
				})
			}
			continue
		}
		for i, cov := range aa {
			d := loadPM4f(span[i])
			c := float32(cov) * (1.0 / 255.0)
			if bb.prescale {
				s := pmColor4f{r: bb.cf.r * c, g: bb.cf.g * c, b: bb.cf.b * c, a: bb.cf.a * c}
				span[i] = storeWord(blendHighpAll(bb.mode, s, d))
				continue
			}
			r := blendHighpAll(bb.mode, bb.cf, d)
			span[i] = storeWord(pmColor4f{
				r: lerpf(d.r, r.r, c),
				g: lerpf(d.g, r.g, c),
				b: lerpf(d.b, r.b, c),
				a: lerpf(d.a, r.a, c),
			})
		}
	}
}

// blitMaskLCD16 is BlitMask's LCD16 lane: the lowp/highp scale_565 and lerp_565 stages with the per-channel coverage
// and the min/max alpha-coverage rule.
func (bb *BlendBlitter) blitMaskLCD16(mask *Mask, clip geom.IRect) {
	prescale := blendShouldPreScaleCoverageRGB(bb.mode)
	for y := clip.Top; y < clip.Bottom; y++ {
		span := bb.row(clip.Left, y, clip.Width())
		mi := mask.addr16(clip.Left, y)
		mrow := mask.Image16[mi : mi+int(clip.Width())]
		if bb.lowp {
			for i, m := range mrow {
				// load_565_: bit-replicate 5/6/5 to 8 bits (the lowp lane keeps green's full 6 bits).
				mr := uint32(m>>11) & 0x1F
				mg := uint32(m>>5) & 0x3F
				mb := uint32(m) & 0x1F
				cr := mr<<3 | mr>>2
				cg := mg<<2 | mg>>4
				cb := mb<<3 | mb>>2
				d := loadPM8(span[i])
				if prescale {
					ca := lowpAlphaCoverage(bb.sc.a, d.a, cr, cg, cb)
					s := pm8{
						r: div255(bb.sc.r * cr),
						g: div255(bb.sc.g * cg),
						b: div255(bb.sc.b * cb),
						a: div255(bb.sc.a * ca),
					}
					span[i] = storePM8(blendLowp(bb.mode, s, d))
					continue
				}
				r := blendLowp(bb.mode, bb.sc, d)
				ca := lowpAlphaCoverage(r.a, d.a, cr, cg, cb)
				span[i] = storePM8(pm8{
					r: lowpLerp(d.r, r.r, cr),
					g: lowpLerp(d.g, r.g, cg),
					b: lowpLerp(d.b, r.b, cb),
					a: lowpLerp(d.a, r.a, ca),
				})
			}
			continue
		}
		for i, m := range mrow {
			cr, cg, cb := from565f(m)
			d := loadPM4f(span[i])
			if prescale {
				ca := alphaCoverageFromRGBCoverage(bb.cf.a, d.a, cr, cg, cb)
				s := pmColor4f{r: bb.cf.r * cr, g: bb.cf.g * cg, b: bb.cf.b * cb, a: bb.cf.a * ca}
				span[i] = storeWord(blendHighpAll(bb.mode, s, d))
				continue
			}
			r := blendHighpAll(bb.mode, bb.cf, d)
			ca := alphaCoverageFromRGBCoverage(r.a, d.a, cr, cg, cb)
			span[i] = storeWord(pmColor4f{
				r: lerpf(d.r, r.r, cr),
				g: lerpf(d.g, r.g, cg),
				b: lerpf(d.b, r.b, cb),
				a: lerpf(d.a, r.a, ca),
			})
		}
	}
}

// lowpAlphaCoverage derives alpha's coverage from the per-channel rgb coverages, in the lowp (integer) domain.
func lowpAlphaCoverage(a, da, cr, cg, cb uint32) uint32 {
	if a < da {
		return min(cr, cg, cb)
	}
	return max(cr, cg, cb)
}

// BlitAntiH2 implements Blitter via the default lowering.
func (bb *BlendBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(bb, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter via the default lowering.
func (bb *BlendBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(bb, x, y, a0, a1)
}
