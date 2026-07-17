// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The highp (float) blend kernels for the modes that have no lowp stage. mad() is implemented as a fused multiply-add
// because both oracle legs run fused code (NEON fmla / AVX2 fma); rcp_fast is implemented as exact division (SIMD lanes
// on other platforms use hardware reciprocal approximations, a per-platform difference the same order as fp-contract
// noise).

package raster

import "math"

// pmColor4f is a premultiplied RGBA color in float, the highp pipeline's working format.
type pmColor4f struct {
	r, g, b, a float32
}

// fmaf is a single-precision fused multiply-add: f*m + a with one rounding. (The float64 FMA of float32 inputs computes
// f*m exactly, so double rounding can only occur on results that land exactly on a float64 tie — negligible next to the
// kernels' tolerance class.)
func fmaf(f, m, a float32) float32 {
	return float32(math.FMA(float64(f), float64(m), float64(a)))
}

// nmadf computes a - f*m, fused.
func nmadf(f, m, a float32) float32 {
	return float32(math.FMA(-float64(f), float64(m), float64(a)))
}

func invf(x float32) float32 { return 1 - x }

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func colordodgeChannel(s, d, sa, da float32) float32 {
	switch {
	case d == 0:
		return s * invf(da)
	case s == sa:
		return s + d*invf(sa)
	default:
		return sa*minf(da, (d*sa)/(sa-s)) + s*invf(da) + d*invf(sa)
	}
}

func colorburnChannel(s, d, sa, da float32) float32 {
	switch {
	case d == da:
		return d + s*invf(da)
	case s == 0:
		return d * invf(sa)
	default:
		return sa*(da-minf(da, (da-d)*sa/s)) + s*invf(da) + d*invf(sa)
	}
}

func softlightChannel(s, d, sa, da float32) float32 {
	var m float32
	if da > 0 {
		m = d / da
	}
	s2 := s + s
	m4 := (m + m) + (m + m)

	// The logic forks three ways: 1. dark src? 2. light src, dark dst? 3. light src, light dst?
	darkSrc := d * (sa + (s2-sa)*(1-m))
	darkDst := (m4*m4+m4)*(m-1) + 7*m
	liteDst := float32(math.Sqrt(float64(m))) - m
	var liteSrc float32
	if (d+d)+(d+d) <= da {
		liteSrc = d*sa + da*(s2-sa)*darkDst
	} else {
		liteSrc = d*sa + da*(s2-sa)*liteDst
	}
	if s2 <= sa {
		return s*invf(da) + d*invf(sa) + darkSrc
	}
	return s*invf(da) + d*invf(sa) + liteSrc
}

///////////////////////////////////////////////////////////////////////////////
// HSL non-separable modes (see https://www.w3.org/TR/compositing-1/#blendingnonseparable)

func satf(r, g, b float32) float32 { return maxf(r, maxf(g, b)) - minf(r, minf(g, b)) }

func lumf(r, g, b float32) float32 { return fmaf(r, 0.30, fmaf(g, 0.59, b*0.11)) }

func setSat(r, g, b *float32, s float32) {
	mn := minf(*r, minf(*g, *b))
	mx := maxf(*r, maxf(*g, *b))
	sat := mx - mn

	// Map min channel to 0, max channel to s, and scale the middle proportionally.
	if sat == 0 {
		s = 0
	} else {
		s /= sat
	}
	*r = (*r - mn) * s
	*g = (*g - mn) * s
	*b = (*b - mn) * s
}

func setLum(r, g, b *float32, l float32) {
	diff := l - lumf(*r, *g, *b)
	*r += diff
	*g += diff
	*b += diff
}

func clipColor(r, g, b *float32, a float32) {
	mn := minf(*r, minf(*g, *b))
	mx := maxf(*r, maxf(*g, *b))
	l := lumf(*r, *g, *b)
	mnScale := l / (l - mn)
	mxScale := (a - l) / (mx - l)
	clipLow := mn < 0 && l != mn
	clipHigh := mx > a && l != mx

	clip := func(c float32) float32 {
		if clipLow {
			c = fmaf(mnScale, c-l, l)
		}
		if clipHigh {
			c = fmaf(mxScale, c-l, l)
		}
		// Sometimes without this we may dip just a little negative.
		return maxf(c, 0)
	}
	*r = clip(*r)
	*g = clip(*g)
	*b = clip(*b)
}

// blendHighp applies the highp-only blend modes.
func blendHighp(mode BlendMode, s, d pmColor4f) pmColor4f {
	switch mode {
	case BlendColorDodge, BlendColorBurn, BlendSoftLight:
		var ch func(s, d, sa, da float32) float32
		switch mode {
		case BlendColorDodge:
			ch = colordodgeChannel
		case BlendColorBurn:
			ch = colorburnChannel
		default:
			ch = softlightChannel
		}
		return pmColor4f{
			r: ch(s.r, d.r, s.a, d.a),
			g: ch(s.g, d.g, s.a, d.a),
			b: ch(s.b, d.b, s.a, d.a),
			a: fmaf(d.a, invf(s.a), s.a),
		}
	case BlendHue:
		R, G, B := s.r*s.a, s.g*s.a, s.b*s.a
		setSat(&R, &G, &B, satf(d.r, d.g, d.b)*s.a)
		setLum(&R, &G, &B, lumf(d.r, d.g, d.b)*s.a)
		clipColor(&R, &G, &B, s.a*d.a)
		return hslFinish(s, d, R, G, B)
	case BlendSaturation:
		R, G, B := d.r*s.a, d.g*s.a, d.b*s.a
		setSat(&R, &G, &B, satf(s.r, s.g, s.b)*d.a)
		setLum(&R, &G, &B, lumf(d.r, d.g, d.b)*s.a) // (This is not redundant.)
		clipColor(&R, &G, &B, s.a*d.a)
		return hslFinish(s, d, R, G, B)
	case BlendColor:
		R, G, B := s.r*d.a, s.g*d.a, s.b*d.a
		setLum(&R, &G, &B, lumf(d.r, d.g, d.b)*s.a)
		clipColor(&R, &G, &B, s.a*d.a)
		return hslFinish(s, d, R, G, B)
	case BlendLuminosity:
		R, G, B := d.r*s.a, d.g*s.a, d.b*s.a
		setLum(&R, &G, &B, lumf(s.r, s.g, s.b)*d.a)
		clipColor(&R, &G, &B, s.a*d.a)
		return hslFinish(s, d, R, G, B)
	default:
		panic("raster: blendHighp called with a lowp mode")
	}
}

// hslFinish applies the shared tail of the four HSL stages: out = src*inv(da) + dst*inv(sa) + blended, with src-over
// alpha.
func hslFinish(s, d pmColor4f, blendR, blendG, blendB float32) pmColor4f {
	return pmColor4f{
		r: fmaf(s.r, invf(d.a), fmaf(d.r, invf(s.a), blendR)),
		g: fmaf(s.g, invf(d.a), fmaf(d.g, invf(s.a), blendG)),
		b: fmaf(s.b, invf(d.a), fmaf(d.b, invf(s.a), blendB)),
		a: s.a + nmadf(s.a, d.a, d.a),
	}
}
