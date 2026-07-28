// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The raster-pipeline stages color filters append: unpremul, matrix_4x5, the clamps, move_src_dst, the blend-mode
// stage, the parametric transfer-function stage (with its approximate-power math — the working-format color filters
// linearize/encode through it), and a per-lane color-function stage standing in for the runtime-effect color filters
// (luma, high contrast) under the no-shader-language rule.

package shaders

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/raster"
)

// AppendUnpremul appends the unpremul stage: scale rgb by 1/a when finite, else by 0.
func (p *Pipeline) AppendUnpremul() {
	p.append(func(z *lanes) {
		inf := math.Float32frombits(0x7f800000)
		for i := range z.n {
			var scale float32
			if s := 1.0 / z.a[i]; s < inf {
				scale = s
			}
			z.r[i] *= scale
			z.g[i] *= scale
			z.b[i] *= scale
		}
	})
}

// AppendMatrix4x5 appends the matrix_4x5 stage (row-major 4x5, translate column last). m is stored by reference as the
// stage context — the pipeline's convention that the caller keeps it valid and unmutated for the pipeline's lifetime
// (the reachable caller passes an immutable color-filter field), so the stage stays a static function rather than a
// closure capturing the coefficients.
func (p *Pipeline) AppendMatrix4x5(m *[20]float32) {
	p.appendCtx(matrix4x5Stage, m)
}

func matrix4x5Stage(z *lanes) {
	mat := z.ctx.(*[20]float32)
	for i := range z.n {
		r, g, b, a := z.r[i], z.g[i], z.b[i], z.a[i]
		z.r[i] = madf(r, mat[0], madf(g, mat[1], madf(b, mat[2], madf(a, mat[3], mat[4]))))
		z.g[i] = madf(r, mat[5], madf(g, mat[6], madf(b, mat[7], madf(a, mat[8], mat[9]))))
		z.b[i] = madf(r, mat[10], madf(g, mat[11], madf(b, mat[12], madf(a, mat[13], mat[14]))))
		z.a[i] = madf(r, mat[15], madf(g, mat[16], madf(b, mat[17], madf(a, mat[18], mat[19]))))
	}
}

// AppendClamp01 appends the clamp_01 stage.
func (p *Pipeline) AppendClamp01() {
	p.append(func(z *lanes) {
		for i := range z.n {
			z.r[i] = clamp01(z.r[i])
			z.g[i] = clamp01(z.g[i])
			z.b[i] = clamp01(z.b[i])
			z.a[i] = clamp01(z.a[i])
		}
	})
}

// AppendMoveSrcDst appends the move_src_dst stage.
func (p *Pipeline) AppendMoveSrcDst() {
	p.append(func(z *lanes) {
		z.dr = z.r
		z.dg = z.g
		z.db = z.b
		z.da = z.a
	})
}

// blendCtx carries a blend stage's mode in reused pipeline storage so the stage function need not capture it.
type blendCtx struct{ mode raster.BlendMode }

// nextBlendCtx hands AppendBlend a distinct blendCtx for one blend stage, retained on the pipeline so its storage is
// reused across pooled compiles (a shader tree can append several blend stages — nested blend shaders, or a blend
// shader plus a blend-mode color filter — a single shared ctx would clobber), matching the matrixCtxs/gradCtxs
// discipline.
func (p *Pipeline) nextBlendCtx() *blendCtx {
	if p.blendCtxN == len(p.blendCtxs) {
		p.blendCtxs = append(p.blendCtxs, &blendCtx{})
	}
	c := p.blendCtxs[p.blendCtxN]
	p.blendCtxN++
	return c
}

// AppendBlend appends a stage that blends src against dst using mode, leaving the result in src. The BlendShader and
// the blend-mode color filter share it.
func (p *Pipeline) AppendBlend(mode raster.BlendMode) {
	c := p.nextBlendCtx()
	c.mode = mode
	p.appendCtx(blendStage, c)
}

func blendStage(z *lanes) {
	mode := z.ctx.(*blendCtx).mode
	for i := range z.n {
		r := raster.BlendHighp4f(mode,
			pm4f{R: z.r[i], G: z.g[i], B: z.b[i], A: z.a[i]},
			pm4f{R: z.dr[i], G: z.dg[i], B: z.db[i], A: z.da[i]})
		z.r[i] = r.R
		z.g[i] = r.G
		z.b[i] = r.B
		z.a[i] = r.A
	}
}

// colorFuncCtx carries an AppendColorFunc stage's transform in reused pipeline storage so the wrapping stage function
// need not capture it. The transform fn itself may still be a closure the runtime-effect color filter builds (e.g. high
// contrast's uniforms) — that is the filter's own allocation, unaffected here — but a non-capturing fn (luma) then
// compiles allocation-free.
type colorFuncCtx struct {
	fn func(r, g, b, a float32) (float32, float32, float32, float32)
}

// nextColorFuncCtx hands AppendColorFunc a distinct colorFuncCtx per stage, retained on the pipeline so its storage is
// reused across pooled compiles, matching the matrixCtxs/gradCtxs discipline.
func (p *Pipeline) nextColorFuncCtx() *colorFuncCtx {
	if p.colorFuncCtxN == len(p.colorFuncCtxs) {
		p.colorFuncCtxs = append(p.colorFuncCtxs, &colorFuncCtx{})
	}
	c := p.colorFuncCtxs[p.colorFuncCtxN]
	p.colorFuncCtxN++
	return c
}

// AppendColorFunc appends a per-lane color transform — the stand-in for a runtime-effect color filter's compiled shader
// program (there is no shader language support; the ctx holds the same math directly).
func (p *Pipeline) AppendColorFunc(fn func(r, g, b, a float32) (float32, float32, float32, float32)) {
	c := p.nextColorFuncCtx()
	c.fn = fn
	p.appendCtx(colorFuncStage, c)
}

func colorFuncStage(z *lanes) {
	fn := z.ctx.(*colorFuncCtx).fn
	for i := range z.n {
		z.r[i], z.g[i], z.b[i], z.a[i] = fn(z.r[i], z.g[i], z.b[i], z.a[i])
	}
}

///////////////////////////////////////////////////////////////////////////////
// parametric transfer function

// TransferFn aliases colorcore.TransferFn (a transfer function restricted to the sRGBish parametric form). The scalar
// evaluation lives in colorcore so imagecore's pixel conversion code can apply the same bit-exact math without
// importing this package.
type TransferFn = colorcore.TransferFn

// AppendTransferFunction appends the parametric transfer-function stage (the only one reachable here — sRGB and its
// inverse are both sRGBish). tf is stored by reference as the stage context — the pipeline's convention that the caller
// keeps it valid and unmutated for the pipeline's lifetime (the reachable callers pass package-global sRGB transfer
// functions), so the stage stays a static function rather than a closure capturing the coefficients.
func (p *Pipeline) AppendTransferFunction(tf *TransferFn) {
	p.appendCtx(transferFunctionStage, tf)
}

func transferFunctionStage(z *lanes) {
	t := z.ctx.(*TransferFn)
	for i := range z.n {
		z.r[i] = t.EvalParametric(z.r[i])
		z.g[i] = t.EvalParametric(z.g[i])
		z.b[i] = t.EvalParametric(z.b[i])
	}
}
