// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Highp (float) forms of the blend modes that normally run lowp. A shader in the paint forces the whole pipeline into
// highp (the gradient/noise stages have no lowp variants), so the shader blitter blends in float even for modes the
// uniform-color BlendBlitter evaluates in lowp. blendHighp (blend_highp.go) covers the seven modes that are highp-only
// in both blitters; blendHighpAll dispatches every mode.

package raster

import "github.com/richardwilkes/canvas/colorcore"

// blendHighpAll evaluates any blend mode with the highp stage formulas, on premultiplied float colors.
func blendHighpAll(mode BlendMode, s, d pmColor4f) pmColor4f {
	switch mode {
	case BlendClear:
		return pmColor4f{}
	case BlendSrc:
		return s // identity: replace dst with src
	case BlendDst:
		return d // move_dst_src
	case BlendSrcOver:
		return porterDuff(s, d, func(sc, dc, sa, _ float32) float32 { return fmaf(dc, invf(sa), sc) })
	case BlendDstOver:
		return porterDuff(s, d, func(sc, dc, _, da float32) float32 { return fmaf(sc, invf(da), dc) })
	case BlendSrcIn:
		return porterDuff(s, d, func(sc, _, _, da float32) float32 { return sc * da })
	case BlendDstIn:
		return porterDuff(s, d, func(_, dc, sa, _ float32) float32 { return dc * sa })
	case BlendSrcOut:
		return porterDuff(s, d, func(sc, _, _, da float32) float32 { return sc * invf(da) })
	case BlendDstOut:
		return porterDuff(s, d, func(_, dc, sa, _ float32) float32 { return dc * invf(sa) })
	case BlendSrcATop:
		return porterDuff(s, d, func(sc, dc, sa, da float32) float32 { return fmaf(sc, da, dc*invf(sa)) })
	case BlendDstATop:
		return porterDuff(s, d, func(sc, dc, sa, da float32) float32 { return fmaf(dc, sa, sc*invf(da)) })
	case BlendXor:
		return porterDuff(s, d, func(sc, dc, sa, da float32) float32 { return fmaf(sc, invf(da), dc*invf(sa)) })
	case BlendPlus:
		return porterDuff(s, d, func(sc, dc, _, _ float32) float32 { return minf(sc+dc, 1) })
	case BlendModulate:
		return porterDuff(s, d, func(sc, dc, _, _ float32) float32 { return sc * dc })
	case BlendScreen:
		return porterDuff(s, d, func(sc, dc, _, _ float32) float32 { return nmadf(sc, dc, sc+dc) })
	case BlendMultiply:
		return srcOverAlpha(s, d, func(sc, dc, sa, da float32) float32 {
			return fmaf(sc, dc, fmaf(sc, invf(da), dc*invf(sa)))
		})
	case BlendOverlay:
		return srcOverAlpha(s, d, func(sc, dc, sa, da float32) float32 {
			m := sa*da - 2*((da-dc)*(sa-sc))
			if 2*dc <= da {
				m = 2 * (sc * dc)
			}
			return sc*invf(da) + dc*invf(sa) + m
		})
	case BlendHardLight:
		return srcOverAlpha(s, d, func(sc, dc, sa, da float32) float32 {
			m := sa*da - 2*((da-dc)*(sa-sc))
			if 2*sc <= sa {
				m = 2 * (sc * dc)
			}
			return sc*invf(da) + dc*invf(sa) + m
		})
	case BlendDarken:
		return srcOverAlpha(s, d, func(sc, dc, sa, da float32) float32 {
			return sc + dc - maxf(sc*da, dc*sa)
		})
	case BlendLighten:
		return srcOverAlpha(s, d, func(sc, dc, sa, da float32) float32 {
			return sc + dc - minf(sc*da, dc*sa)
		})
	case BlendDifference:
		return srcOverAlpha(s, d, func(sc, dc, sa, da float32) float32 {
			return sc + dc - 2*minf(sc*da, dc*sa)
		})
	case BlendExclusion:
		return srcOverAlpha(s, d, func(sc, dc, _, _ float32) float32 {
			return sc + dc - 2*(sc*dc)
		})
	default:
		return blendHighp(mode, s, d)
	}
}

// porterDuff applies ch to each channel including alpha (with s=sa, d=da) — the shared shape of the coefficient-based
// Porter-Duff modes.
func porterDuff(s, d pmColor4f, ch func(sc, dc, sa, da float32) float32) pmColor4f {
	return pmColor4f{
		r: ch(s.r, d.r, s.a, d.a),
		g: ch(s.g, d.g, s.a, d.a),
		b: ch(s.b, d.b, s.a, d.a),
		a: ch(s.a, d.a, s.a, d.a),
	}
}

// srcOverAlpha applies ch to the color channels and src-over to alpha — the shared shape of the separable
// non-Porter-Duff modes.
func srcOverAlpha(s, d pmColor4f, ch func(sc, dc, sa, da float32) float32) pmColor4f {
	return pmColor4f{
		r: ch(s.r, d.r, s.a, d.a),
		g: ch(s.g, d.g, s.a, d.a),
		b: ch(s.b, d.b, s.a, d.a),
		a: fmaf(d.a, invf(s.a), s.a),
	}
}

// BlendHighp4f exposes the highp blend kernels on colorcore's premul float color, for the shaders package's
// blend-shader implementation.
func BlendHighp4f(mode BlendMode, s, d colorcore.PMColor4f) colorcore.PMColor4f {
	r := blendHighpAll(mode, pmColor4f{r: s.R, g: s.G, b: s.B, a: s.A}, pmColor4f{r: d.R, g: d.G, b: d.B, a: d.A})
	return colorcore.PMColor4f{R: r.r, G: r.g, B: r.b, A: r.a}
}
