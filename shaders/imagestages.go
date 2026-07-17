// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The raster-pipeline stages the image shader appends: the exclusive tile stages (repeat/mirror with the mirror bias),
// the decal stages, the per-color-type gathers with clamp_ex addressing, the bilinear/bicubic sampler stages over
// samplerCtx, and the fused bilerp/bicubic clamp_8888 fast paths — highp forms throughout (this pipeline computes in
// full float precision; a lowp integer pipeline would differ by at most the established ±1/255).

package shaders

import (
	"math"

	"github.com/richardwilkes/canvas/imagecore"
)

// gatherCtx is the gather stage's context over an imagecore pixel container. tileX/tileY hold the pixel-space
// repeat/mirror limits (one tileCtx per axis) the tile stages read: they are constant across every gather stage in a
// compile (derived from width/height/ roundDownAtInteger), so homing them here lets the shared static tile stages read
// them instead of each capturing a snapshot. setRGB holds the paint color the alpha-only set_rgb stage tints with (one
// per image shader, so a 1:1 home on the gatherCtx). Held in reused pipeline storage so the stages need not capture.
type gatherCtx struct {
	px                 *imagecore.Pixels
	width              float32
	height             float32
	weights            [16]float32 // for bicubic_clamp_8888
	tileX, tileY       tileCtx
	setRGB             [3]float32
	roundDownAtInteger bool
}

// setTileLimits fills tileX/tileY from the already-set width/height/roundDownAtInteger: scale/invScale from the
// dimension and the mirror ULP bias direction from roundDownAtInteger.
func (g *gatherCtx) setTileLimits() {
	g.tileX = tileCtx{scale: g.width, invScale: 1 / g.width}
	g.tileY = tileCtx{scale: g.height, invScale: 1 / g.height}
	bias := int32(-1)
	if g.roundDownAtInteger {
		bias = 1
	}
	g.tileX.mirrorBiasDir, g.tileY.mirrorBiasDir = bias, bias
}

// nextGatherCtx hands an image level a distinct gatherCtx, retained on the pipeline so its storage is reused across
// pooled compiles (a BlendShader can inline two image children; a single shared ctx would clobber). It returns the ctx
// zeroed — appendStages sets only some fields (weights, roundDownAtInteger, setRGB are conditional), so a reused ctx
// must not carry a prior compile's stale ones.
func (p *Pipeline) nextGatherCtx() *gatherCtx {
	if p.gatherCtxN == len(p.gatherCtxs) {
		p.gatherCtxs = append(p.gatherCtxs, &gatherCtx{})
	}
	g := p.gatherCtxs[p.gatherCtxN]
	p.gatherCtxN++
	*g = gatherCtx{}
	return g
}

// nextSamplerCtx hands a bilinear/bicubic image a distinct samplerCtx, retained on the pipeline for reuse across pooled
// compiles. The samplerCtx is pure scratch — every field a stage reads is written by an earlier stage in the same
// compile (save_xy writes x/y/fx/fy and, for bicubic, wx/wy; the axis stages write scalex/scaley; appendStages sets
// weights before the cubic setup reads it) — so a reused ctx needs no zeroing, matching the laneMask discipline.
func (p *Pipeline) nextSamplerCtx() *samplerCtx {
	if p.samplerCtxN == len(p.samplerCtxs) {
		p.samplerCtxs = append(p.samplerCtxs, &samplerCtx{})
	}
	c := p.samplerCtxs[p.samplerCtxN]
	p.samplerCtxN++
	return c
}

// nextDecalCtx hands a decal-tiled image a distinct decalTileCtx, retained on the pipeline for reuse across pooled
// compiles. It returns the ctx zeroed — inclusiveEdgeX/Y are set only when roundDownAtInteger, so a reused ctx must not
// carry a prior compile's stale edges (the mask is pure scratch, rewritten each ShadeSpan chunk, so zeroing it is
// merely harmless).
func (p *Pipeline) nextDecalCtx() *decalTileCtx {
	if p.decalCtxN == len(p.decalCtxs) {
		p.decalCtxs = append(p.decalCtxs, &decalTileCtx{})
	}
	d := p.decalCtxs[p.decalCtxN]
	p.decalCtxN++
	*d = decalTileCtx{}
	return d
}

// fltMin is the smallest positive normal (non-subnormal) float32 value.
var fltMin = math.Float32frombits(0x00800000)

// clampEx clamps to [FLT_MIN, limit - 1ulp] (exclusive-limit clamp whose minimum stays above zero so the
// roundDownAtInteger ULP subtraction cannot produce NaN).
func clampEx(v, limit float32) float32 {
	inclusiveL := math.Float32frombits(math.Float32bits(limit) - 1)
	if !(v > fltMin) { // NaN-aware max(inclusiveZ, v)
		v = fltMin
	}
	if v > inclusiveL {
		v = inclusiveL
	}
	return v
}

// index clamps the sample coordinates into bounds and returns the element index.
func (g *gatherCtx) index(x, y float32) int {
	x = clampEx(x, g.width)
	y = clampEx(y, g.height)
	if g.roundDownAtInteger {
		x = math.Float32frombits(math.Float32bits(x) - 1)
		y = math.Float32frombits(math.Float32bits(y) - 1)
	}
	return int(int32(y))*int(g.px.RowElems) + int(int32(x))
}

// gatherPixel loads one pixel as unpremultiplied-channel floats, covering every color type the image shader can gather
// (8888 with optional RB swap or forced opaque, alpha8, gray8 via alpha-to-gray, 565, f16).
func (g *gatherCtx) gatherPixel(x, y float32) (r, gg, b, a float32) {
	const inv255 = 1.0 / 255.0
	i := g.index(x, y)
	switch g.px.Info.ColorType {
	case imagecore.ColorTypeRGBA8888, imagecore.ColorTypeBGRA8888, imagecore.ColorTypeRGB888x:
		v := g.px.Words[i]
		r = float32(v&0xFF) * inv255
		gg = float32((v>>8)&0xFF) * inv255
		b = float32((v>>16)&0xFF) * inv255
		a = float32(v>>24) * inv255
		if g.px.Info.ColorType == imagecore.ColorTypeBGRA8888 {
			r, b = b, r // swap_rb
		}
		if g.px.Info.ColorType == imagecore.ColorTypeRGB888x {
			a = 1 // force_opaque
		}
	case imagecore.ColorTypeAlpha8:
		a = float32(g.px.Bytes[i]) * inv255
	case imagecore.ColorTypeGray8:
		// gather_a8 + alpha_to_gray
		v := float32(g.px.Bytes[i]) * inv255
		r, gg, b = v, v, v
		a = 1
	case imagecore.ColorTypeRGB565:
		v := uint32(g.px.U16s[i])
		r = float32(v&(31<<11)) * (1.0 / (31 << 11))
		gg = float32(v&(63<<5)) * (1.0 / (63 << 5))
		b = float32(v&31) * (1.0 / 31)
		a = 1
	case imagecore.ColorTypeRGBAF16:
		r = halfBitsToFloat(g.px.U16s[4*i])
		gg = halfBitsToFloat(g.px.U16s[4*i+1])
		b = halfBitsToFloat(g.px.U16s[4*i+2])
		a = halfBitsToFloat(g.px.U16s[4*i+3])
	}
	return r, gg, b, a
}

// halfBitsToFloat is from_half with hardware (denormal-honoring) semantics, matching imagecore.
func halfBitsToFloat(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	em := uint32(h & 0x7FFF)
	switch {
	case em == 0:
		return math.Float32frombits(sign)
	case em >= 0x7C00:
		return math.Float32frombits(sign | 0x7F800000 | (em&0x03FF)<<13)
	case em < 0x0400:
		v := float32(em) * float32(math.Ldexp(1, -24))
		if sign != 0 {
			v = -v
		}
		return v
	default:
		return math.Float32frombits(sign | (em<<13 + ((127 - 15) << 23)))
	}
}

// appendGather appends the tiling + gather + decal-mask stage group for one level (append_tiling_and_gather). The
// per-axis limits live on g (setTileLimits, filled once at compile setup), constant across every gather this compile
// appends, so the shared static tile stages read them from g rather than each capturing a snapshot.
func (p *Pipeline) appendGather(g *gatherCtx, tmx, tmy TileMode, decal *decalTileCtx) {
	if tmx == TileDecal && tmy == TileDecal {
		p.appendDecalXAndY(decal)
	} else {
		switch tmx {
		case TileClamp: // the gather clamps for us
		case TileMirror:
			p.appendTile(&g.tileX, true, true)
		case TileRepeat:
			p.appendTile(&g.tileX, true, false)
		case TileDecal:
			p.appendDecalAxis(decal, true)
		}
		switch tmy {
		case TileClamp:
		case TileMirror:
			p.appendTile(&g.tileY, false, true)
		case TileRepeat:
			p.appendTile(&g.tileY, false, false)
		case TileDecal:
			p.appendDecalAxis(decal, false)
		}
	}

	p.appendCtx(gatherStage, g)

	if decal != nil {
		// check_decal_mask
		p.appendApplyVectorMask(&decal.mask)
	}
}

// gatherStage loads one pixel per lane as unpremultiplied-channel floats, covering every color type the gather stage
// group supports. The gatherCtx is the stage context.
func gatherStage(z *lanes) {
	g := z.ctx.(*gatherCtx)
	for i := range z.n {
		z.r[i], z.g[i], z.b[i], z.a[i] = g.gatherPixel(z.r[i], z.g[i])
	}
}

///////////////////////////////////////////////////////////////////////////////
// tile stages (the image forms with pixel-space limits)

// tileCtx holds the per-axis pixel-space repeat/mirror limits a tile stage reads.
type tileCtx struct {
	scale         float32
	invScale      float32
	mirrorBiasDir int32
}

// exclusiveRepeat computes v - floor(v*invScale)*scale, with the final multiply-subtract contracted to an FMA.
func (t *tileCtx) exclusiveRepeat(v float32) float32 {
	return madf(-floorf(v*t.invScale), t.scale, v)
}

// exclusiveMirror computes the exclusive mirrored-tile coordinate, including the reflection ULP bias; the
// multiply-subtracts contract to FMAs.
func (t *tileCtx) exclusiveMirror(v float32) float32 {
	limit := t.scale
	invLimit := t.invScale
	u := madf(-(floorf(v*invLimit*0.5) * 2), limit, v)
	s := floorf(u * invLimit)
	m := madf(-(2 * s), u-limit, u)
	biasInUlps := int32(s)
	return math.Float32frombits(uint32(int32(math.Float32bits(m)) + t.mirrorBiasDir*biasInUlps))
}

// appendTile appends repeat_x/repeat_y/mirror_x/mirror_y as a static stage reading t (which lives on the gatherCtx, so
// it stays valid without a captured snapshot).
func (p *Pipeline) appendTile(t *tileCtx, isX, mirror bool) {
	switch {
	case isX && mirror:
		p.appendCtx(mirrorXStage, t)
	case isX:
		p.appendCtx(repeatXStage, t)
	case mirror:
		p.appendCtx(mirrorYStage, t)
	default:
		p.appendCtx(repeatYStage, t)
	}
}

func repeatXStage(z *lanes) {
	t := z.ctx.(*tileCtx)
	for i := range z.n {
		z.r[i] = t.exclusiveRepeat(z.r[i])
	}
}

func repeatYStage(z *lanes) {
	t := z.ctx.(*tileCtx)
	for i := range z.n {
		z.g[i] = t.exclusiveRepeat(z.g[i])
	}
}

func mirrorXStage(z *lanes) {
	t := z.ctx.(*tileCtx)
	for i := range z.n {
		z.r[i] = t.exclusiveMirror(z.r[i])
	}
}

func mirrorYStage(z *lanes) {
	t := z.ctx.(*tileCtx)
	for i := range z.n {
		z.g[i] = t.exclusiveMirror(z.g[i])
	}
}

// decalTileCtx holds the decal-tile stage's per-lane mask and axis limits.
type decalTileCtx struct {
	mask           laneMask
	limitX         float32
	limitY         float32
	inclusiveEdgeX float32
	inclusiveEdgeY float32
}

// appendDecalAxis appends decal_x or decal_y as a static stage reading the decalTileCtx.
func (p *Pipeline) appendDecalAxis(d *decalTileCtx, isX bool) {
	if isX {
		p.appendCtx(decalXAxisStage, d)
	} else {
		p.appendCtx(decalYAxisStage, d)
	}
}

func decalXAxisStage(z *lanes) {
	d := z.ctx.(*decalTileCtx)
	for i := range z.n {
		v := z.r[i]
		if (0 < v && v < d.limitX) || v == d.inclusiveEdgeX {
			d.mask[i] = 0xFFFFFFFF
		} else {
			d.mask[i] = 0
		}
	}
}

func decalYAxisStage(z *lanes) {
	d := z.ctx.(*decalTileCtx)
	for i := range z.n {
		v := z.g[i]
		if (0 < v && v < d.limitY) || v == d.inclusiveEdgeY {
			d.mask[i] = 0xFFFFFFFF
		} else {
			d.mask[i] = 0
		}
	}
}

// appendDecalXAndY appends decal_x_and_y as a static stage reading the decalTileCtx.
func (p *Pipeline) appendDecalXAndY(d *decalTileCtx) {
	p.appendCtx(decalXAndYStage, d)
}

func decalXAndYStage(z *lanes) {
	d := z.ctx.(*decalTileCtx)
	for i := range z.n {
		x, y := z.r[i], z.g[i]
		if ((0 < x && x < d.limitX) || x == d.inclusiveEdgeX) &&
			((0 < y && y < d.limitY) || y == d.inclusiveEdgeY) {
			d.mask[i] = 0xFFFFFFFF
		} else {
			d.mask[i] = 0
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// sampler stages

// samplerCtx holds the bilinear/bicubic sampler stages' scratch: saved coordinates, fractional offsets, per-lane
// weights, and (for bicubic) the per-axis weight rows.
type samplerCtx struct {
	x, y, fx, fy   [stride]float32
	scalex, scaley [stride]float32
	weights        [16]float32
	wx, wy         [4][stride]float32
}

func fractf(v float32) float32 { return v - floorf(v) }

// appendSaveXYInit appends bilinear_setup or (with weights) bicubic_setup: save_xy plus the accumulator clear, and for
// bicubic the per-axis weight tables. The samplerCtx is the stage context.
func (p *Pipeline) appendSaveXYInit(c *samplerCtx, cubic bool) {
	if cubic {
		p.appendCtx(saveXYInitCubicStage, c)
	} else {
		p.appendCtx(saveXYInitStage, c)
	}
}

func saveXYInitStage(z *lanes) {
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		c.x[i] = z.r[i]
		c.y[i] = z.g[i]
		c.fx[i] = fractf(z.r[i] + 0.5)
		c.fy[i] = fractf(z.g[i] + 0.5)
	}
	for i := range z.n {
		z.dr[i], z.dg[i], z.db[i], z.da[i] = 0, 0, 0, 0
	}
}

func saveXYInitCubicStage(z *lanes) {
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		c.x[i] = z.r[i]
		c.y[i] = z.g[i]
		c.fx[i] = fractf(z.r[i] + 0.5)
		c.fy[i] = fractf(z.g[i] + 0.5)
	}
	w := &c.weights
	for j := range 4 {
		for i := range z.n {
			c.wx[j][i] = bicubicWts(c.fx[i], w[j], w[j+4], w[j+8], w[j+12])
			c.wy[j][i] = bicubicWts(c.fy[i], w[j], w[j+4], w[j+8], w[j+12])
		}
	}
	for i := range z.n {
		z.dr[i], z.dg[i], z.db[i], z.da[i] = 0, 0, 0, 0
	}
}

// bicubicWts evaluates the cubic weight polynomial a + t*(b + t*(c + t*d)).
func bicubicWts(t, a, b, c, d float32) float32 {
	return madf(t, madf(t, madf(t, d, c), b), a)
}

// appendBilinearAxis appends bilinear_nx/px/ny/py (kScale ±1) as one of four static stages reading the samplerCtx: the
// sign of kScale (the ±0.5 sample offset and fx vs 1-fx weight) and the axis are folded into the stage choice rather
// than captured.
func (p *Pipeline) appendBilinearAxis(c *samplerCtx, isX bool, kScale float32) {
	switch {
	case isX && kScale > 0:
		p.appendCtx(bilinearPXStage, c)
	case isX:
		p.appendCtx(bilinearNXStage, c)
	case kScale > 0:
		p.appendCtx(bilinearPYStage, c)
	default:
		p.appendCtx(bilinearNYStage, c)
	}
}

func bilinearNXStage(z *lanes) { // bilinear_nx: kScale = -1
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		z.r[i] = c.x[i] - 0.5
		c.scalex[i] = 1 - c.fx[i]
	}
}

func bilinearPXStage(z *lanes) { // bilinear_px: kScale = +1
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		z.r[i] = c.x[i] + 0.5
		c.scalex[i] = c.fx[i]
	}
}

func bilinearNYStage(z *lanes) { // bilinear_ny: kScale = -1
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		z.g[i] = c.y[i] - 0.5
		c.scaley[i] = 1 - c.fy[i]
	}
}

func bilinearPYStage(z *lanes) { // bilinear_py: kScale = +1
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		z.g[i] = c.y[i] + 0.5
		c.scaley[i] = c.fy[i]
	}
}

// appendBicubicAxis appends one of eight static bicubic tap stages reading the samplerCtx (kScale ∈ {-3,-1,+1,+3}; the
// weight row is (kScale+3)/2), folding the axis and weight row into the stage choice rather than capturing them.
func (p *Pipeline) appendBicubicAxis(c *samplerCtx, isX bool, kScale float32) {
	if isX {
		switch kScale {
		case -3:
			p.appendCtx(bicubicN3XStage, c)
		case -1:
			p.appendCtx(bicubicN1XStage, c)
		case +1:
			p.appendCtx(bicubicP1XStage, c)
		default: // +3
			p.appendCtx(bicubicP3XStage, c)
		}
	} else {
		switch kScale {
		case -3:
			p.appendCtx(bicubicN3YStage, c)
		case -1:
			p.appendCtx(bicubicN1YStage, c)
		case +1:
			p.appendCtx(bicubicP1YStage, c)
		default: // +3
			p.appendCtx(bicubicP3YStage, c)
		}
	}
}

// bicubicX computes z.r = c.x + kScale*0.5, weight from row (kScale+3)/2.
func bicubicX(z *lanes, row int, kScale float32) {
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		z.r[i] = c.x[i] + kScale*0.5
		c.scalex[i] = c.wx[row][i]
	}
}

// bicubicY is the Y-axis analog of bicubicX.
func bicubicY(z *lanes, row int, kScale float32) {
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		z.g[i] = c.y[i] + kScale*0.5
		c.scaley[i] = c.wy[row][i]
	}
}

func bicubicN3XStage(z *lanes) { bicubicX(z, 0, -3) }
func bicubicN1XStage(z *lanes) { bicubicX(z, 1, -1) }
func bicubicP1XStage(z *lanes) { bicubicX(z, 2, +1) }
func bicubicP3XStage(z *lanes) { bicubicX(z, 3, +3) }
func bicubicN3YStage(z *lanes) { bicubicY(z, 0, -3) }
func bicubicN1YStage(z *lanes) { bicubicY(z, 1, -1) }
func bicubicP1YStage(z *lanes) { bicubicY(z, 2, +1) }
func bicubicP3YStage(z *lanes) { bicubicY(z, 3, +3) }

// appendAccumulate appends the accumulate stage reading the samplerCtx.
func (p *Pipeline) appendAccumulate(c *samplerCtx) {
	p.appendCtx(accumulateStage, c)
}

func accumulateStage(z *lanes) {
	c := z.ctx.(*samplerCtx)
	for i := range z.n {
		scale := c.scalex[i] * c.scaley[i]
		z.dr[i] = madf(scale, z.r[i], z.dr[i])
		z.dg[i] = madf(scale, z.g[i], z.dg[i])
		z.db[i] = madf(scale, z.b[i], z.db[i])
		z.da[i] = madf(scale, z.a[i], z.da[i])
	}
}

// appendMoveDstSrc appends move_dst_src.
func (p *Pipeline) appendMoveDstSrc() {
	p.append(func(z *lanes) {
		for i := range z.n {
			z.r[i], z.g[i], z.b[i], z.a[i] = z.dr[i], z.dg[i], z.db[i], z.da[i]
		}
	})
}

// AppendClampGamut appends the clamp_gamut stage: alpha to [0,1], rgb to [0,a].
func (p *Pipeline) AppendClampGamut() {
	p.append(func(z *lanes) {
		for i := range z.n {
			a := minf(maxf(z.a[i], 0), 1)
			z.a[i] = a
			z.r[i] = minf(maxf(z.r[i], 0), a)
			z.g[i] = minf(maxf(z.g[i], 0), a)
			z.b[i] = minf(maxf(z.b[i], 0), a)
		}
	})
}

// appendSetRGB appends the set_rgb stage (alpha-only image tinting by the paint color). The three tint channels are
// stored on the alpha-only image's own gatherCtx (a 1:1 home — one set_rgb per image shader), which is the stage
// context, so the stage need not capture them.
func (p *Pipeline) appendSetRGB(g *gatherCtx, r, gg, b float32) {
	g.setRGB = [3]float32{r, gg, b}
	p.appendCtx(setRGBStage, g)
}

func setRGBStage(z *lanes) {
	g := z.ctx.(*gatherCtx)
	r, gg, b := g.setRGB[0], g.setRGB[1], g.setRGB[2]
	for i := range z.n {
		z.r[i] = r
		z.g[i] = gg
		z.b[i] = b
	}
}

///////////////////////////////////////////////////////////////////////////////
// fused fast paths

// appendBilerpClamp8888 appends the fused clamp-tiled bilerp fast path — the highp form; a lowp integer kernel for
// simple pipelines would differ by at most ±1/255. The RB swap for BGRA sources happens inside gatherPixel, equivalent
// to a post-stage swap since the accumulation is channel-independent.
func (p *Pipeline) appendBilerpClamp8888(g *gatherCtx) {
	p.appendCtx(bilerpClamp8888Stage, g)
}

func bilerpClamp8888Stage(z *lanes) {
	g := z.ctx.(*gatherCtx)
	for i := range z.n {
		cx, cy := z.r[i], z.g[i]
		fx := fractf(cx + 0.5)
		fy := fractf(cy + 0.5)
		var r, gg, b, a float32
		for _, py := range [2]float32{-0.5, 0.5} {
			for _, px := range [2]float32{-0.5, 0.5} {
				sr, sg, sb, sa := g.gatherPixel(cx+px, cy+py)
				sx := 1 - fx
				if px > 0 {
					sx = fx
				}
				sy := 1 - fy
				if py > 0 {
					sy = fy
				}
				area := sx * sy
				r = madf(sr, area, r)
				gg = madf(sg, area, gg)
				b = madf(sb, area, b)
				a = madf(sa, area, a)
			}
		}
		z.r[i], z.g[i], z.b[i], z.a[i] = r, gg, b, a
	}
}

// appendBicubicClamp8888 appends the fused clamp-tiled bicubic fast path (RB swap inside gatherPixel, as above).
func (p *Pipeline) appendBicubicClamp8888(g *gatherCtx) {
	p.appendCtx(bicubicClamp8888Stage, g)
}

func bicubicClamp8888Stage(z *lanes) {
	g := z.ctx.(*gatherCtx)
	w := &g.weights
	for i := range z.n {
		cx, cy := z.r[i], z.g[i]
		fx := fractf(cx + 0.5)
		fy := fractf(cy + 0.5)
		var scalex, scaley [4]float32
		for j := range 4 {
			scalex[j] = bicubicWts(fx, w[j], w[j+4], w[j+8], w[j+12])
			scaley[j] = bicubicWts(fy, w[j], w[j+4], w[j+8], w[j+12])
		}
		var r, gg, b, a float32
		sampleY := cy - 1.5
		for yy := range 4 {
			sampleX := cx - 1.5
			for xx := range 4 {
				scale := scalex[xx] * scaley[yy]
				sr, sg, sb, sa := g.gatherPixel(sampleX, sampleY)
				r = madf(scale, sr, r)
				gg = madf(scale, sg, gg)
				b = madf(scale, sb, b)
				a = madf(scale, sa, a)
				sampleX++
			}
			sampleY++
		}
		z.r[i], z.g[i], z.b[i], z.a[i] = r, gg, b, a
	}
}
