// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Canvas.DrawAtlas: the atlas draw entry point and the raster device's lowering. BitmapDevice lowers per sprite — see
// BitmapDevice.DrawAtlas for why that is pixel-equal to a vertex pipeline. The GPU device draws through DrawAtlasOp
// (gpu/gl/drawatlasop.go).

package canvas

import (
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// cleanPaintForDrawVertices normalizes the paint in place for sprite drawing, which fills quads and ignores mask
// filters and path effects.
func cleanPaintForDrawVertices(paint *Paint) {
	paint.Style = StyleFill
	paint.MaskFilter = nil
	paint.PathEffect = nil
}

// DrawAtlas draws a set of sprites from the atlas. Sprite i is the atlas sub-rect tex[i] mapped by xforms[i] (a
// rotation+scale placement). With colors non-empty, sprite i's texels are combined with colors[i] through mode
// (colors[i] is the blend dst, the sprite the src). cull is an optional conservative bounds hint used only for quick
// rejection; paint may be nil. The draw count is min(len(xforms), len(tex)) (and len(colors) when non-empty). This API
// has no C-API entry point.
func (c *Canvas) DrawAtlas(atlas imagecore.DrawableImage, xforms []geom.RSXform, tex []geom.Rect, colors []colorcore.Color, mode raster.BlendMode, sampling shaders.SamplingOptions, cull *geom.Rect, paint *Paint) {
	if atlas == nil {
		return
	}
	count := min(len(xforms), len(tex))
	if len(colors) != 0 {
		count = min(count, len(colors))
	}
	if count == 0 {
		return
	}
	xforms = xforms[:count]
	tex = tex[:count]
	if len(colors) != 0 {
		colors = colors[:count]
	}

	// An atlas draw combines the sprite-quad and image-draw paint rules.
	realPaint := cleanPaintForDrawImage(paint)
	cleanPaintForDrawVertices(&realPaint)
	// The atlas image shader: clamp/clamp, no local matrix.
	realPaint.Shader = shaders.NewImageDrawable(atlas, shaders.TileClamp, shaders.TileClamp,
		sampling, nil)

	if cull != nil && c.internalQuickReject(*cull, &realPaint) {
		return
	}

	if dp, restore, ok := c.aboutToDraw(&realPaint, nil); ok {
		c.topDevice().DrawAtlas(xforms, tex, colors, mode, dp)
		if restore != nil {
			restore()
		}
	}
}

// atlasDrawScratch holds BitmapDevice.DrawAtlas's per-call temporaries so a bulk call makes no per-sprite allocations
// (the S118 imageDrawScratch discipline): the working paint, the per-sprite matrices (whose addresses flow into
// retaining callees — the draw's ctm reaches the blitter), the per-sprite color/blend scratch shaders, and the draw
// value itself. Pooled (not a shared instance) so nested/concurrent draws stay independent.
type atlasDrawScratch struct {
	dr        draw
	blend     shaders.BlendShader
	paint     Paint
	xformMtx  geom.Matrix // sprite matrix: tex[i] (at origin) onto xforms[i]'s quad
	ctm       geom.Matrix // localToDevice ∘ xformMtx, the per-sprite draw CTM
	spriteClr shaders.ColorShader
	paintClr  shaders.ColorShader
}

var atlasDrawScratchPool = sync.Pool{New: func() any { return &atlasDrawScratch{} }}

func acquireAtlasDrawScratch() *atlasDrawScratch {
	return atlasDrawScratchPool.Get().(*atlasDrawScratch)
}

func releaseAtlasDrawScratch(s *atlasDrawScratch) {
	// Drop the shader/pixmap references so the idle pooled scratch does not pin them.
	s.paint = Paint{}
	s.blend.SetBlend(raster.BlendSrcOver, nil, nil)
	s.dr = draw{}
	atlasDrawScratchPool.Put(s)
}

// DrawAtlas implements Device: the raster lowering. Each sprite is lowered directly: build the affine matrix mapping
// tex[i] (translated to the origin) onto xforms[i]'s quad, concatenate it onto the CTM, and fill the rect tex[i] non-AA
// — the paint's shader is the atlas image shader (identity local matrix), so local coords are atlas coords and the fill
// samples exactly the sprite's texels. A convex quad and its two-triangle split cover identical pixel centers non-AA,
// and constant per-sprite colors keep the color interpolation constant, so this per-sprite lowering is pixel-equal to
// feeding two triangles per sprite through a vertex pipeline (guarded by the atlas-* corpus differentials and the
// GPU-vs-CPU self-consistency check).
//
// Deliberately untiled: this stays a plain draw even past the 8191px tiling threshold, exactly like DrawPaint and the
// sprite lanes.
func (d *BitmapDevice) DrawAtlas(xforms []geom.RSXform, tex []geom.Rect, colors []colorcore.Color, mode raster.BlendMode, paint *Paint) {
	scratch := acquireAtlasDrawScratch()
	defer releaseAtlasDrawScratch(scratch)
	working := &scratch.paint
	*working = *paint
	working.Style = StyleFill
	// Sprite quads rasterize non-AA (paint AA is ignored, per the triangle-fill semantics).
	working.AntiAlias = false
	// The per-sprite fills must take the raster-pipeline lane, never the legacy fixed-point image-sampler lane: the two
	// sample differently (4-bit legacy filter weights vs the pipeline's float bilerp), and the atlas lane is defined in
	// terms of the pipeline's sampling.
	working.forceRasterPipeline = true

	// Blend-mode simplifications: Src drops the colors entirely, Dst drops the shader and keeps only the colors.
	useColors := len(colors) != 0
	blenderIsDst := false
	if useColors {
		switch mode {
		case raster.BlendSrc:
			useColors = false
		case raster.BlendDst:
			blenderIsDst = true
		}
	}
	if useColors {
		if blenderIsDst {
			working.Shader = &scratch.spriteClr
		} else {
			// Blend(mode, dst = colors[i], src = atlas shader). With no shader the blend applies to the sprite color
			// and the opaque paint color.
			src := working.Shader
			if src == nil {
				scratch.paintClr.SetColor(working.Color.WithAlpha(0xFF))
				src = &scratch.paintClr
			}
			scratch.blend.SetBlend(mode, &scratch.spriteClr, src)
			working.Shader = &scratch.blend
		}
	}

	for i := range xforms {
		if useColors {
			scratch.spriteClr.SetColor(colors[i])
		}
		scratch.xformMtx = xforms[i].Matrix()
		scratch.xformMtx.PreTranslate(-tex[i].Left, -tex[i].Top)
		scratch.ctm.SetConcat(&d.localToDevice, &scratch.xformMtx)
		scratch.dr = draw{dst: d.pix, ctm: &scratch.ctm, rc: d.rcStack.RC()}
		scratch.dr.drawRect(tex[i], working)
	}
}
