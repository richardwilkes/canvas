// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// LegacyShaderBlitter: the legacy src-over blitter over a shader context's ShadeSpan32, selected for non-dithered
// src-over paints whose shader has a legacy (fixed-point) context. When the shader's output is opaque the span is
// shaded directly into the device; otherwise the s32/s32a row procs blend it, and BlitMask uses the blendRowA8 kernels.

package raster

import (
	"sync"

	"github.com/richardwilkes/canvas/geom"
)

// SpanShader32 produces premultiplied N32 pixels for a horizontal pixel run — the legacy fixed-point image sampler
// (shaders.BitmapProcContext) satisfies this. Held as an interface (not a bound-method func value) so building the
// blitter does not allocate a closure per draw.
type SpanShader32 interface {
	ShadeSpan32(x, y int32, dst []uint32)
}

// LegacyShaderBlitter is the legacy src-over blitter over a legacy shader context.
type LegacyShaderBlitter struct {
	dst    *Pixmap
	shade  SpanShader32
	buffer []uint32
	// opaque is true when the shader's output is always fully opaque, so full-coverage spans can be shaded directly
	// into the device.
	opaque bool
}

// legacyShaderBlitterPool retains LegacyShaderBlitter storage — chiefly its per-row shade buffer — across draws so a
// repeated image-shader fill does not reallocate it. NewLegacyShaderBlitter borrows; the draw hands the blitter back
// with RecycleLegacyShaderBlitter once the synchronous fill is done.
var legacyShaderBlitterPool = sync.Pool{New: func() any { return new(LegacyShaderBlitter) }}

// NewLegacyShaderBlitter returns the legacy shader blitter, drawn from a shared pool; shade is the shader context's
// span shader, opaque whether its output is always fully opaque. Hand the blitter back with RecycleLegacyShaderBlitter
// when done.
func NewLegacyShaderBlitter(dst *Pixmap, shade SpanShader32, opaque bool) *LegacyShaderBlitter {
	lb := legacyShaderBlitterPool.Get().(*LegacyShaderBlitter)
	lb.dst = dst
	lb.shade = shade
	lb.opaque = opaque
	if cap(lb.buffer) < int(dst.Width) {
		lb.buffer = make([]uint32, dst.Width)
	} else {
		lb.buffer = lb.buffer[:dst.Width]
	}
	return lb
}

// Shade returns the span shader the blitter draws from, so the draw layer can recycle a pooled
// shaders.BitmapProcContext once the blitter is done (raster cannot import shaders).
func (lb *LegacyShaderBlitter) Shade() SpanShader32 { return lb.shade }

// RecycleLegacyShaderBlitter returns a blitter obtained from NewLegacyShaderBlitter to the shared pool, dropping its
// external references but retaining the shade buffer for reuse. After RecycleLegacyShaderBlitter the caller must treat
// lb as invalid.
func RecycleLegacyShaderBlitter(lb *LegacyShaderBlitter) {
	lb.dst = nil
	lb.shade = nil
	legacyShaderBlitterPool.Put(lb)
}

// fourByteInterp blends src and dst with alpha (0..255), reusing fastFourByteInterp256's kernel.
func fourByteInterp(src, dst, alpha uint32) uint32 {
	return fastFourByteInterp256(src, dst, alpha255To256(alpha))
}

// proc32 does a full-coverage row blend of the shaded span.
func (lb *LegacyShaderBlitter) proc32(dst, src []uint32) {
	if lb.opaque {
		copy(dst, src)
		return
	}
	pmSrcOverRowFn(dst, src)
}

// proc32Blend does a partial-coverage row blend.
func (lb *LegacyShaderBlitter) proc32Blend(dst, src []uint32, alpha uint32) {
	if lb.opaque {
		interp256RowFn(dst, src, alpha255To256(alpha))
		return
	}
	pmBlendRowFn(dst, src, alpha255To256(alpha))
}

// BlitH implements Blitter.
func (lb *LegacyShaderBlitter) BlitH(x, y, width int32) {
	d := lb.dst.addr(x, y)
	device := lb.dst.Pix[d : d+int(width)]
	if lb.opaque {
		lb.shade.ShadeSpan32(x, y, device)
		return
	}
	span := lb.buffer[:width]
	lb.shade.ShadeSpan32(x, y, span)
	lb.proc32(device, span)
}

// BlitRect implements Blitter.
func (lb *LegacyShaderBlitter) BlitRect(x, y, width, height int32) {
	for row := int32(0); row < height; row++ {
		lb.BlitH(x, y+row, width)
	}
}

// BlitAntiH implements Blitter.
func (lb *LegacyShaderBlitter) BlitAntiH(x, y int32, antialias []Alpha, runs []int16) {
	i := 0
	for runs[i] != 0 {
		n := int32(runs[i])
		aa := uint32(antialias[i])
		if aa != 0 {
			d := lb.dst.addr(x, y)
			device := lb.dst.Pix[d : d+int(n)]
			if lb.opaque && aa == 255 {
				// cool, have the shader draw right into the device
				lb.shade.ShadeSpan32(x, y, device)
			} else {
				span := lb.buffer[:n]
				lb.shade.ShadeSpan32(x, y, span)
				if aa == 255 {
					lb.proc32(device, span)
				} else {
					lb.proc32Blend(device, span, aa)
				}
			}
		}
		x += n
		i += int(n)
	}
}

// BlitV implements Blitter.
func (lb *LegacyShaderBlitter) BlitV(x, y, height int32, alpha Alpha) {
	aa := uint32(alpha)
	for row := int32(0); row < height; row++ {
		d := lb.dst.addr(x, y+row)
		device := lb.dst.Pix[d : d+1]
		if lb.opaque {
			if aa == 255 {
				lb.shade.ShadeSpan32(x, y+row, device)
			} else {
				var c [1]uint32
				lb.shade.ShadeSpan32(x, y+row, c[:])
				device[0] = fourByteInterp(c[0], device[0], aa)
			}
			continue
		}
		span := lb.buffer[:1]
		lb.shade.ShadeSpan32(x, y+row, span)
		if aa == 255 {
			lb.proc32(device, span)
		} else {
			lb.proc32Blend(device, span, aa)
		}
	}
}

// BlitAntiRect implements Blitter.
func (lb *LegacyShaderBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	BlitAntiRectViaVH(lb, x, y, width, height, leftAlpha, rightAlpha)
}

// approxScale computes (x*y + x) / 256.
func approxScale(x, y uint32) uint32 {
	return (x*y + x) >> 8
}

// div255Round computes (x + 127) / 255, bit-exact.
func div255Round(x uint32) uint32 {
	return (x + 127) / 255
}

// blendRowA8 computes sAA = approxScale(s, cov); dst = sAA + approxScale(d, 255 - sAA.alpha).
func blendRowA8(dst []uint32, cov []uint8, src []uint32) {
	for i := range dst {
		c := uint32(cov[i])
		s := src[i]
		var sAA uint32
		for shift := 0; shift < 32; shift += 8 {
			sAA |= approxScale(s>>shift&0xFF, c) << shift
		}
		alpha := sAA >> 24
		d := dst[i]
		var out uint32
		for shift := 0; shift < 32; shift += 8 {
			out |= (sAA>>shift&0xFF + approxScale(d>>shift&0xFF, 255-alpha)) & 0xFF << shift
		}
		dst[i] = out
	}
}

// blendRowA8Opaque computes dst = div255(s*c + d*(255-c)).
func blendRowA8Opaque(dst []uint32, cov []uint8, src []uint32) {
	for i := range dst {
		c := uint32(cov[i])
		s := src[i]
		d := dst[i]
		var out uint32
		for shift := 0; shift < 32; shift += 8 {
			out |= div255Round(s>>shift&0xFF*c+d>>shift&0xFF*(255-c)) & 0xFF << shift
		}
		dst[i] = out
	}
}

// BlitMask implements Blitter, with A8 and LCD16 blend rows.
func (lb *LegacyShaderBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	x := clip.Left
	width := clip.Width()
	if mask.Format == MaskLCD16 {
		for y := clip.Top; y < clip.Bottom; y++ {
			d := lb.dst.addr(x, y)
			device := lb.dst.Pix[d : d+int(width)]
			m := mask.addr16(x, y)
			maskRow := mask.Image16[m : m+int(width)]
			span := lb.buffer[:width]
			lb.shade.ShadeSpan32(x, y, span)
			if lb.opaque {
				blendRowLCD16OpaqueFn(device, maskRow, span)
			} else {
				blendRowLCD16(device, maskRow, span)
			}
		}
		return
	}
	for y := clip.Top; y < clip.Bottom; y++ {
		d := lb.dst.addr(x, y)
		device := lb.dst.Pix[d : d+int(width)]
		m := mask.addr8(x, y)
		maskRow := mask.Image[m : m+int(width)]
		span := lb.buffer[:width]
		lb.shade.ShadeSpan32(x, y, span)
		if lb.opaque {
			blendRowA8Opaque(device, maskRow, span)
		} else {
			blendRowA8(device, maskRow, span)
		}
	}
}

// BlitAntiH2 implements Blitter.
func (lb *LegacyShaderBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(lb, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter.
func (lb *LegacyShaderBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(lb, x, y, a0, a1)
}
