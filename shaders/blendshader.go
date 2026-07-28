// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// BlendShader blends the outputs of two shaders using a blend mode.

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/raster"
)

// BlendShader combines a dst and src shader through a blend mode.
type BlendShader struct {
	dst, src Shader
	mode     raster.BlendMode
}

// NewBlend builds a shader that blends dst and src with mode: nil children yield nil, BlendSrc collapses to src,
// BlendDst to dst, and BlendClear to a transparent-black color shader. The clear collapse is not just a stage saving:
// the constant color shader reports IsConstant, which lets the blitter take its constant-color path instead of
// compiling and evaluating both children's stage trees per pixel for a result that is transparent black everywhere.
func NewBlend(mode raster.BlendMode, dst, src Shader) Shader {
	if dst == nil || src == nil {
		return nil
	}
	switch mode {
	case raster.BlendClear:
		return NewColor(colorcore.Transparent)
	case raster.BlendSrc:
		return src
	case raster.BlendDst:
		return dst
	default:
		return &BlendShader{dst: dst, src: src, mode: mode}
	}
}

// SetBlend re-initializes s to blend dst and src with mode, the scratch-construction counterpart of NewBlend (the
// SetImage discipline): a caller reuses one BlendShader value across draws — the DrawAtlas per-sprite color-modulation
// lane — instead of allocating a fresh *BlendShader per sprite. The caller must have handled NewBlend's nil-child and
// Src/Dst collapse cases itself (they do not produce a BlendShader). Clear may be passed — it evaluates correctly as
// transparent black — but it forgoes NewBlend's collapse to a constant color shader. Only for transient scratch shaders
// consumed synchronously within a draw.
func (s *BlendShader) SetBlend(mode raster.BlendMode, dst, src Shader) {
	s.mode = mode
	s.dst = dst
	s.src = src
}

// Mode returns the blend mode; Dst and Src return the children — the descriptor the GPU side consumes.
func (s *BlendShader) Mode() raster.BlendMode { return s.mode }

// Dst returns the dst child shader.
func (s *BlendShader) Dst() Shader { return s.dst }

// Src returns the src child shader.
func (s *BlendShader) Src() Shader { return s.src }

// IsOpaque implements Shader (a blend of two shaders is not known to be opaque).
func (s *BlendShader) IsOpaque() bool { return false }

// IsConstant implements Shader (a blend of two shaders is not treated as constant).
func (s *BlendShader) IsConstant() bool { return false }

// blendShaderCtx holds appendStages' store/load scratch: coords saves the seeded coordinates across the dst child
// (store_src_rg / load_src_rg), res0 saves dst's output across the src child (store_src / load_dst). Pure
// write-before-read scratch — each ShadeSpan chunk stores before it loads — held in reused pipeline storage so the four
// stages need not capture it.
type blendShaderCtx struct {
	coords [2][stride]float32
	res0   [4][stride]float32
}

// nextBlendShaderCtx hands a BlendShader a distinct blendShaderCtx for one blend, retained on the pipeline so its
// store/load scratch is reused across pooled compiles (a nested blend shader inlines another; a single shared ctx would
// clobber), matching the matrixCtxs/gradCtxs discipline.
func (p *Pipeline) nextBlendShaderCtx() *blendShaderCtx {
	if p.blendShaderCtxN == len(p.blendShaderCtxs) {
		p.blendShaderCtxs = append(p.blendShaderCtxs, new(blendShaderCtx))
	}
	c := p.blendShaderCtxs[p.blendShaderCtxN]
	p.blendShaderCtxN++
	return c
}

// appendStages runs dst, saves its output (store_src), reruns the coordinates through src (each child seeds its own
// coordinates when the incoming record has not applied the CTM), then loads dst's output into the dst registers
// (load_dst) and appends the blend-mode stages.
func (s *BlendShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	c := p.nextBlendShaderCtx()
	if m.coordsSeeded() {
		p.appendCtx(blendStoreSrcRGStage, c) // store_src_rg
	}
	if !s.dst.appendStages(p, m) {
		return false
	}
	p.appendCtx(blendStoreSrcStage, c) // store_src
	if m.coordsSeeded() {
		p.appendCtx(blendLoadSrcRGStage, c) // load_src_rg
	}
	if !s.src.appendStages(p, m) {
		return false
	}
	p.appendCtx(blendLoadDstStage, c) // load_dst
	p.AppendBlend(s.mode)
	return true
}

// blendStoreSrcRGStage saves the seeded coordinates before the dst child's matrix stage overwrites them. The
// blendShaderCtx is the stage context.
func blendStoreSrcRGStage(z *lanes) {
	c := z.ctx.(*blendShaderCtx)
	c.coords[0] = z.r
	c.coords[1] = z.g
}

// blendLoadSrcRGStage restores the seeded coordinates for the src child.
func blendLoadSrcRGStage(z *lanes) {
	c := z.ctx.(*blendShaderCtx)
	z.r = c.coords[0]
	z.g = c.coords[1]
}

// blendStoreSrcStage saves the dst child's output across the src child.
func blendStoreSrcStage(z *lanes) {
	c := z.ctx.(*blendShaderCtx)
	c.res0[0] = z.r
	c.res0[1] = z.g
	c.res0[2] = z.b
	c.res0[3] = z.a
}

// blendLoadDstStage loads the saved dst-child output into the dst registers for the blend-mode stage.
func blendLoadDstStage(z *lanes) {
	c := z.ctx.(*blendShaderCtx)
	z.dr = c.res0[0]
	z.dg = c.res0[1]
	z.db = c.res0[2]
	z.da = c.res0[3]
}
