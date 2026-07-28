// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Shader-span blitter for a non-constant color pipeline into an 8888 surface. A shader in the paint forces the pipeline
// into highp (the gradient stages have no lowp variants), so everything here blends in float:
//
//	blitRect:  shade [clamp_01] (load_dst + blend | nothing for BlendSrc) store
//	blitAntiH: shade [clamp_01] (scale(cov) load_dst blend | load_dst blend lerp(cov)) store
//	blitMask:  the same per pixel with the A8 coverage (scale_u8 / lerp_u8)
//
// The prescale split follows blendShouldPreScaleCoverage exactly as blitAntiH assembles it. Unlike the solid blitters,
// a ShaderBlitter carries mutable per-span state, so it must not be shared across goroutines (FillPathParallel requires
// per-worker blitters).

package raster

import (
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// SpanShader produces premultiplied float colors for a horizontal pixel run — the compiled color pipeline
// (shaders.Pipeline) satisfies this.
type SpanShader interface {
	ShadeSpan(x, y int32, dst []colorcore.PMColor4f)
}

// ShaderBlitter blits a shaded span source with any blend mode into a Pixmap.
type ShaderBlitter struct {
	dev      *Pixmap
	shade    SpanShader
	buf      []colorcore.PMColor4f
	mode     BlendMode
	prescale bool
}

// shaderBlitterPool retains ShaderBlitter storage — chiefly its per-span shade buffer — across draws so a repeated
// shaded fill does not reallocate it. NewShaderBlitter borrows; the draw hands the blitter back with
// RecycleShaderBlitter once the synchronous fill that consumed it is done.
var shaderBlitterPool = sync.Pool{New: func() any { return new(ShaderBlitter) }}

// NewShaderBlitter returns a blitter for the given span shader and blend mode, drawn from a shared pool. The caller
// performs the srcover→src strength reduction (when the pipeline is opaque) before choosing the mode, and hands the
// blitter back with RecycleShaderBlitter when done.
func NewShaderBlitter(dev *Pixmap, shade SpanShader, mode BlendMode) *ShaderBlitter {
	sb := shaderBlitterPool.Get().(*ShaderBlitter)
	sb.dev = dev
	sb.shade = shade
	sb.mode = mode
	sb.prescale = blendShouldPreScaleCoverage(mode)
	return sb
}

// Shade returns the span shader the blitter draws from, so the draw layer can recycle a pooled shaders.Pipeline once
// the blitter is done (raster cannot import shaders).
func (sb *ShaderBlitter) Shade() SpanShader { return sb.shade }

// RecycleShaderBlitter returns a blitter obtained from NewShaderBlitter to the shared pool, dropping its external
// references (device, shader) but retaining the shade buffer for reuse. After RecycleShaderBlitter the caller must
// treat sb as invalid.
func RecycleShaderBlitter(sb *ShaderBlitter) {
	sb.dev = nil
	sb.shade = nil
	shaderBlitterPool.Put(sb)
}

// shadeRow evaluates the color pipeline for [x, x+width) on row y with the 8888 normalization clamp
// (appendClampIfNormalized).
func (sb *ShaderBlitter) shadeRow(x, y, width int32) []colorcore.PMColor4f {
	buf := sb.shadeRowRaw(x, y, width)
	clampSpan01(buf)
	return buf
}

// shadeRowRaw evaluates the color pipeline without the normalization clamp pass. Only the store-only (BlendSrc
// full-coverage) lane may use it directly: for every float v, toUnorm(clamp01f(v)) == toUnorm(v) — both clamp to [0,
// 255] after the *255 scale and both send NaN to 0 — so the store's per-channel bytes are identical while the span
// skips a full read-modify-write pass. Any lane that feeds the floats into blend or coverage math must go through
// shadeRow, which the clamp semantically belongs to (appendClampIfNormalized runs before the blend stages in the C
// pipeline).
func (sb *ShaderBlitter) shadeRowRaw(x, y, width int32) []colorcore.PMColor4f {
	if cap(sb.buf) < int(width) {
		sb.buf = make([]colorcore.PMColor4f, width)
	}
	buf := sb.buf[:width]
	sb.shade.ShadeSpan(x, y, buf)
	return buf
}

func clamp01f(v float32) float32 {
	return minf(maxf(v, 0), 1)
}

func (sb *ShaderBlitter) row(x, y, width int32) []uint32 {
	start := sb.dev.addr(x, y)
	return sb.dev.Pix[start : start+int(width)]
}

// blendSpan blends the shaded colors into the span at full coverage.
func (sb *ShaderBlitter) blendSpan(x, y int32, span []uint32) {
	if sb.mode == BlendSrc {
		// Store-only lane: toUnorm clamps identically to the shadeRow pass (see shadeRowRaw), so skip it.
		storeSpanSrc(sb.shadeRowRaw(x, y, int32(len(span))), span)
		return
	}
	buf := sb.shadeRow(x, y, int32(len(span)))
	for i, px := range span {
		s := pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A}
		d := loadPM4f(px)
		span[i] = storeWord(blendHighpAll(sb.mode, s, d))
	}
}

// blendSpanCoverage blends the shaded colors into the span at constant coverage (0 < aa < 255).
func (sb *ShaderBlitter) blendSpanCoverage(x, y int32, span []uint32, aa uint32) {
	buf := sb.shadeRow(x, y, int32(len(span)))
	c := float32(aa) * (1.0 / 255.0)
	if sb.prescale {
		// scale_1_float, then blend.
		for i, px := range span {
			s := pmColor4f{r: buf[i].R * c, g: buf[i].G * c, b: buf[i].B * c, a: buf[i].A * c}
			d := loadPM4f(px)
			span[i] = storeWord(blendHighpAll(sb.mode, s, d))
		}
		return
	}
	// Blend, then lerp_1_float.
	for i, px := range span {
		s := pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A}
		d := loadPM4f(px)
		r := blendHighpAll(sb.mode, s, d)
		span[i] = storeWord(pmColor4f{
			r: lerpf(d.r, r.r, c),
			g: lerpf(d.g, r.g, c),
			b: lerpf(d.b, r.b, c),
			a: lerpf(d.a, r.a, c),
		})
	}
}

// BlitH implements Blitter.
func (sb *ShaderBlitter) BlitH(x, y, width int32) {
	sb.blendSpan(x, y, sb.row(x, y, width))
}

// BlitRect implements Blitter.
func (sb *ShaderBlitter) BlitRect(x, y, width, height int32) {
	for r := y; r < y+height; r++ {
		sb.blendSpan(x, r, sb.row(x, r, width))
	}
}

// BlitAntiH implements Blitter, walking the alpha runs and dispatching each to the full-coverage or partial-coverage
// span blender.
func (sb *ShaderBlitter) BlitAntiH(x, y int32, antialias []Alpha, runs []int16) {
	off := 0
	for {
		run := int32(runs[off])
		if run <= 0 {
			return
		}
		switch aa := antialias[off]; aa {
		case 0x00:
		case 0xFF:
			sb.blendSpan(x, y, sb.row(x, y, run))
		default:
			sb.blendSpanCoverage(x, y, sb.row(x, y, run), uint32(aa))
		}
		x += run
		off += int(run)
	}
}

// BlitV implements Blitter via the default lowering (a constant-stride mask; the per-pixel math is identical to
// BlitAntiH's).
func (sb *ShaderBlitter) BlitV(x, y, height int32, alpha Alpha) {
	BlitVViaAntiH(sb, x, y, height, alpha)
}

// BlitAntiRect implements Blitter via the default lowering.
func (sb *ShaderBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	BlitAntiRectViaVH(sb, x, y, width, height, leftAlpha, rightAlpha)
}

// BlitMask implements Blitter for A8 masks (scale_u8 / lerp_u8 per the prescale split) and LCD16 masks (scale_565 /
// lerp_565 with the rgb-coverage prescale rule — src-over lands on the lerp lane because per-channel scaling destroys
// the source-alpha term).
func (sb *ShaderBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	if mask.Format == MaskLCD16 {
		prescale := blendShouldPreScaleCoverageRGB(sb.mode)
		for y := clip.Top; y < clip.Bottom; y++ {
			span := sb.row(clip.Left, y, clip.Width())
			mi := mask.addr16(clip.Left, y)
			mrow := mask.Image16[mi : mi+int(clip.Width())]
			buf := sb.shadeRow(clip.Left, y, clip.Width())
			for i, m := range mrow {
				cr, cg, cb := from565f(m)
				d := loadPM4f(span[i])
				s := pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A}
				if prescale {
					// Somewhat unusually, scale_565 needs dst loaded first (its ca term compares the source alpha
					// against the dst alpha).
					ca := alphaCoverageFromRGBCoverage(s.a, d.a, cr, cg, cb)
					s = pmColor4f{r: s.r * cr, g: s.g * cg, b: s.b * cb, a: s.a * ca}
					span[i] = storeWord(blendHighpAll(sb.mode, s, d))
					continue
				}
				// lerp_565 runs after the blend, so its ca term compares the blended alpha with dst.
				r := blendHighpAll(sb.mode, s, d)
				ca := alphaCoverageFromRGBCoverage(r.a, d.a, cr, cg, cb)
				span[i] = storeWord(pmColor4f{
					r: lerpf(d.r, r.r, cr),
					g: lerpf(d.g, r.g, cg),
					b: lerpf(d.b, r.b, cb),
					a: lerpf(d.a, r.a, ca),
				})
			}
		}
		return
	}
	for y := clip.Top; y < clip.Bottom; y++ {
		span := sb.row(clip.Left, y, clip.Width())
		mi := mask.addr8(clip.Left, y)
		aa := mask.Image[mi : mi+int(clip.Width())]
		buf := sb.shadeRow(clip.Left, y, clip.Width())
		for i, cov := range aa {
			c := float32(cov) * (1.0 / 255.0)
			d := loadPM4f(span[i])
			if sb.prescale {
				s := pmColor4f{r: buf[i].R * c, g: buf[i].G * c, b: buf[i].B * c, a: buf[i].A * c}
				span[i] = storeWord(blendHighpAll(sb.mode, s, d))
				continue
			}
			s := pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A}
			r := blendHighpAll(sb.mode, s, d)
			span[i] = storeWord(pmColor4f{
				r: lerpf(d.r, r.r, c),
				g: lerpf(d.g, r.g, c),
				b: lerpf(d.b, r.b, c),
				a: lerpf(d.a, r.a, c),
			})
		}
	}
}

// BlitAntiH2 implements Blitter via the default lowering.
func (sb *ShaderBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(sb, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter via the default lowering.
func (sb *ShaderBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(sb, x, y, a0, a1)
}

///////////////////////////////////////////////////////////////////////////////

// NewBlendBlitterPM4f handles the case where a color pipeline reduces to a constant premultiplied float color: it
// re-enters the uniform-color blitter instead. Only in-premul-range colors take the lowp lanes, so out-of-range
// constants blend in highp.
func NewBlendBlitterPM4f(dev *Pixmap, c colorcore.PMColor4f, mode BlendMode) *BlendBlitter {
	cf := pmColor4f{r: c.R, g: c.G, b: c.B, a: c.A}
	// The alpha bound is part of the test, not just the per-channel comparisons: with cf.a > 1 the 8-bit lanes below
	// exceed 255, storePM8 then ORs the overflow into the neighboring channel and blendLowp's inv255 underflows. NaN
	// fails every comparison, so it takes the highp lane too.
	inRange := cf.a <= 1 &&
		0 <= cf.r && cf.r <= cf.a &&
		0 <= cf.g && cf.g <= cf.a &&
		0 <= cf.b && cf.b <= cf.a
	// To make loads more direct, we store 8-bit values in 16-bit slots.
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
		lowp:     inRange && !blendNeedsHighp(mode),
		prescale: blendShouldPreScaleCoverage(mode),
	}
}
