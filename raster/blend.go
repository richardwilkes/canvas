// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The 29 blend modes. 22 of the modes run in a lowp variant (8-bit values in 16-bit lanes) whenever the whole pipeline
// is lowp-capable — which it is for the solid-color blits this package performs — and the remaining 7 (color-dodge,
// color-burn, soft-light and the four HSL non-separable modes) run in the highp float variant. Both are implemented
// here.
//
// Rounding notes for the lowp math: a fast div255 is the exact (v+128+((v+128)>>8))>>8 on NEON but the ≤1-off
// (v+255)>>8 elsewhere; the accurate form is exact everywhere. This implementation uses the exact form for both,
// matching the NEON (darwin) oracle leg byte-for-byte; the x86 leg can be off by one where the oracle itself takes the
// inaccurate shortcut (scale/lerp stages and a few blend formulas).

package raster

// BlendMode identifies one of the supported blend modes (see the const block below for the ordering).
type BlendMode uint8

// BlendMode values.
const (
	BlendClear BlendMode = iota
	BlendSrc
	BlendDst
	BlendSrcOver
	BlendDstOver
	BlendSrcIn
	BlendDstIn
	BlendSrcOut
	BlendDstOut
	BlendSrcATop
	BlendDstATop
	BlendXor
	BlendPlus
	BlendModulate
	BlendScreen // last coefficient-based mode
	BlendOverlay
	BlendDarken
	BlendLighten
	BlendColorDodge
	BlendColorBurn
	BlendHardLight
	BlendSoftLight
	BlendDifference
	BlendExclusion
	BlendMultiply // last separable mode
	BlendHue
	BlendSaturation
	BlendColor
	BlendLuminosity
)

// blendNeedsHighp reports whether the mode must be evaluated in the highp (float) pipeline because it has no lowp
// stage.
func blendNeedsHighp(mode BlendMode) bool {
	switch mode {
	case BlendColorDodge, BlendColorBurn, BlendSoftLight,
		BlendHue, BlendSaturation, BlendColor, BlendLuminosity:
		return true
	default:
		return false
	}
}

// blendShouldPreScaleCoverage reports whether mode commutes with coverage scaling of the source, so the blitter can
// scale src by coverage before blending instead of lerping afterwards.
func blendShouldPreScaleCoverage(mode BlendMode) bool {
	switch mode {
	case BlendDst, BlendDstOver, BlendPlus, BlendDstOut, BlendSrcATop, BlendSrcOver, BlendXor:
		return true
	default:
		return false
	}
}

// BlendSupportsCoverageAsAlpha reports whether mode can express partial coverage by scaling the source alpha (the
// thin-stroke-as-hairline promotion relies on it).
func BlendSupportsCoverageAsAlpha(mode BlendMode) bool {
	return blendShouldPreScaleCoverage(mode)
}

///////////////////////////////////////////////////////////////////////////////
// lowp (8-bit in 16-bit lanes) kernels

// div255 is the exact-rounding division by 255 (formula [2]; see the file comment).
func div255(v uint32) uint32 {
	v += 128
	return (v + (v >> 8)) >> 8
}

func inv255(v uint32) uint32 { return 255 - v }

// lowpLerp computes the lowp lerp: div255(from*inv(t) + to*t).
func lowpLerp(from, to, t uint32) uint32 {
	return div255(from*inv255(t) + to*t)
}

// pm8 is a premultiplied 8-bit RGBA color in 16-bit-friendly lanes.
type pm8 struct {
	r, g, b, a uint32
}

// blendLowp applies mode to source s over destination d, both valid premul; mode must not be a highp-only mode.
func blendLowp(mode BlendMode, s, d pm8) pm8 {
	// The coefficient-style modes apply one formula to all four channels.
	var ch func(s, d, sa, da uint32) uint32
	switch mode {
	case BlendClear:
		ch = func(_, _, _, _ uint32) uint32 { return 0 }
	case BlendSrc:
		return s // identity: replace dst with src
	case BlendDst:
		return d // move_dst_src
	case BlendSrcOver:
		ch = func(s, d, sa, _ uint32) uint32 { return s + div255(d*inv255(sa)) }
	case BlendDstOver:
		ch = func(s, d, _, da uint32) uint32 { return d + div255(s*inv255(da)) }
	case BlendSrcIn:
		ch = func(s, _, _, da uint32) uint32 { return div255(s * da) }
	case BlendDstIn:
		ch = func(_, d, sa, _ uint32) uint32 { return div255(d * sa) }
	case BlendSrcOut:
		ch = func(s, _, _, da uint32) uint32 { return div255(s * inv255(da)) }
	case BlendDstOut:
		ch = func(_, d, sa, _ uint32) uint32 { return div255(d * inv255(sa)) }
	case BlendSrcATop:
		ch = func(s, d, sa, da uint32) uint32 { return div255(s*da + d*inv255(sa)) }
	case BlendDstATop:
		ch = func(s, d, sa, da uint32) uint32 { return div255(d*sa + s*inv255(da)) }
	case BlendXor:
		ch = func(s, d, sa, da uint32) uint32 { return div255(s*inv255(da) + d*inv255(sa)) }
	case BlendPlus:
		ch = func(s, d, _, _ uint32) uint32 { return min(s+d, 255) }
	case BlendModulate:
		ch = func(s, d, _, _ uint32) uint32 { return div255(s * d) }
	case BlendScreen:
		ch = func(s, d, _, _ uint32) uint32 { return s + d - div255(s*d) }
	case BlendMultiply:
		ch = func(s, d, sa, da uint32) uint32 { return div255(s*inv255(da) + d*inv255(sa) + s*d) }
	}
	if ch != nil {
		return pm8{
			r: ch(s.r, d.r, s.a, d.a),
			g: ch(s.g, d.g, s.a, d.a),
			b: ch(s.b, d.b, s.a, d.a),
			a: ch(s.a, d.a, s.a, d.a),
		}
	}

	// The remaining lowp modes apply the formula to color channels and src-over to alpha.
	switch mode {
	case BlendDarken:
		ch = func(s, d, sa, da uint32) uint32 { return s + d - div255(max(s*da, d*sa)) }
	case BlendLighten:
		ch = func(s, d, sa, da uint32) uint32 { return s + d - div255(min(s*da, d*sa)) }
	case BlendDifference:
		ch = func(s, d, sa, da uint32) uint32 { return s + d - 2*div255(min(s*da, d*sa)) }
	case BlendExclusion:
		ch = func(s, d, _, _ uint32) uint32 { return s + d - 2*div255(s*d) }
	case BlendHardLight:
		ch = func(s, d, sa, da uint32) uint32 {
			var m uint32
			if 2*s <= sa {
				m = 2 * s * d
			} else {
				m = sa*da - 2*(sa-s)*(da-d)
			}
			return div255(s*inv255(da) + d*inv255(sa) + m)
		}
	case BlendOverlay:
		ch = func(s, d, sa, da uint32) uint32 {
			var m uint32
			if 2*d <= da {
				m = 2 * s * d
			} else {
				m = sa*da - 2*(sa-s)*(da-d)
			}
			return div255(s*inv255(da) + d*inv255(sa) + m)
		}
	default:
		panic("raster: blendLowp called with a highp-only mode")
	}
	return pm8{
		r: ch(s.r, d.r, s.a, d.a),
		g: ch(s.g, d.g, s.a, d.a),
		b: ch(s.b, d.b, s.a, d.a),
		a: s.a + div255(d.a*inv255(s.a)),
	}
}
