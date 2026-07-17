// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The batched sprite-atlas draw op behind Canvas.DrawAtlas's GPU lane. Each op bakes its sprites into a vertex stream
// of non-AA position/[color]/local-coord quads at construction (RSXform's tri-strip positions, tex-rect texcoords,
// optional constant per-sprite color), draws through the default geometry processor + the simple mesh draw op helper
// with the shared non-AA quad index buffer, and merges with compatible atlas ops on the CombineIfPossible quad-count
// rule (matching view matrix, color-inclusion, and processor set), so an N-sprite call — or a run of compatible calls —
// lands in one draw.

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

var drawAtlasOpClassID = GenOpClassID()

// drawAtlasGeometry is one source op's baked vertex bytes. hasColors records the stride the bytes were baked with —
// Finalize can clear the op's hasColors field after analysis, so the prepare-time copy must know whether each source
// geometry carries interleaved color bytes to strip.
type drawAtlasGeometry struct {
	verts     []byte
	color     colorcore.PMColor4f
	hasColors bool
}

// drawAtlasOp draws a batch of sprites from an atlas image.
type drawAtlasOp struct {
	drawOpNoClipToShape
	// Prepared state.
	vertexBuffer AnyBuffer
	programInfo  *ProgramInfo
	indexBuffer  *Buffer
	helper       SimpleMeshDrawOpHelper
	geoData      []drawAtlasGeometry
	OpBase
	quadCount  int
	baseVertex int
	viewMatrix geom.Matrix
	color      colorcore.PMColor4f
	hasColors  bool
}

// atlasVertexStride returns the vertex stride for the color-inclusion flag: position (2 floats), optional premul color
// (4 unorm bytes), local coord (2 floats) — the "position [color] texCoord" order.
func atlasVertexStride(hasColors bool) int {
	stride := 4 * 4 // position + local coord
	if hasColors {
		stride += 4
	}
	return stride
}

// NewDrawAtlasOp builds the op from a converted paint (consumed), the view matrix, and the sprite arrays. colors may be
// empty.
func NewDrawAtlasOp(paint *Paint, viewMatrix *geom.Matrix, aaType gpu.AAType, xforms []geom.RSXform, rects []geom.Rect, colors []colorcore.Color) DrawOp {
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := &drawAtlasOp{}
	o.InitOp(drawAtlasOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, aaType, nil, PipelineInputFlagNone)
	o.viewMatrix = *viewMatrix
	o.color = color
	o.hasColors = len(colors) != 0

	spriteCount := len(xforms)
	vertexStride := atlasVertexStride(o.hasColors)

	// Bail out if we'd overflow from a really large draw.
	if spriteCount > math.MaxInt32/(4*vertexStride) {
		return o
	}

	o.quadCount = spriteCount
	verts := make([]byte, 4*vertexStride*spriteCount)
	o.geoData = append(o.geoData, drawAtlasGeometry{
		verts:     verts,
		color:     color,
		hasColors: o.hasColors,
	})

	texOffset := 8 // position is first; color (if any) sits between position and texCoord
	if o.hasColors {
		texOffset += 4
	}

	bounds := largestInvertedRect()
	// TODO4F: preserve float colors. The paint alpha modulates the sprite colors in 8-bit math (unpacking the paint
	// color's alpha byte then round-dividing by 255).
	paintAlpha := pmColorToByte(color.A)
	offset := 0
	var strip [4]geom.Point
	for i := 0; i < spriteCount; i++ {
		r := rects[i]
		xforms[i].ToTriStrip(r.Width(), r.Height(), &strip)

		if o.hasColors {
			// Convert to premul RGBA bytes, sprite alpha scaled by the paint alpha.
			spriteColor := colors[i]
			if paintAlpha != 255 {
				spriteColor = spriteColor.WithAlpha(
					colorcore.MulDiv255Round(spriteColor.A(), paintAlpha),
				)
			}
			pm := spriteColor.PreMultiply()
			for v := 0; v < 4; v++ {
				base := offset + v*vertexStride + 8
				verts[base] = pm.R()
				verts[base+1] = pm.G()
				verts[base+2] = pm.B()
				verts[base+3] = pm.A()
			}
		}

		// Position and uv per vertex, in tri-strip corner order: (l,t), (l,b), (r,t), (r,b).
		uvs := [4][2]float32{{r.Left, r.Top}, {r.Left, r.Bottom}, {r.Right, r.Top}, {r.Right, r.Bottom}}
		for v := 0; v < 4; v++ {
			base := offset + v*vertexStride
			putAtlasF32(verts, base, strip[v].X)
			putAtlasF32(verts, base+4, strip[v].Y)
			putAtlasF32(verts, base+texOffset, uvs[v][0])
			putAtlasF32(verts, base+texOffset+4, uvs[v][1])
			bounds = growToInclude(bounds, strip[v])
		}
		offset += 4 * vertexStride
	}

	// Transform the accumulated local bounds into device space; the op has no AA bloat and isn't hairline geometry.
	devBounds, _ := o.viewMatrix.MapRect(bounds)
	o.SetBounds(devBounds, HasAABloat(false), IsHairline(false))
	return o
}

// putAtlasF32 writes a little-endian float32 at off.
func putAtlasF32(dst []byte, off int, v float32) {
	binary.LittleEndian.PutUint32(dst[off:], math.Float32bits(v))
}

// pmColorToByte quantizes a float channel to a byte with round-half-up.
func pmColorToByte(v float32) uint8 {
	f := v*255 + 0.5
	if f < 0 {
		f = 0
	} else if f > 255 {
		f = 255
	}
	return uint8(f)
}

// growToInclude returns r expanded (if needed) to contain pt.
func growToInclude(r geom.Rect, pt geom.Point) geom.Rect {
	return geom.Rect{
		Left:   min(r.Left, pt.X),
		Top:    min(r.Top, pt.Y),
		Right:  max(r.Right, pt.X),
		Bottom: max(r.Bottom, pt.Y),
	}
}

// Name implements Op.
func (o *drawAtlasOp) Name() string { return "DrawAtlasOp" }

// NumQuads is test access to the batched sprite count.
func (o *drawAtlasOp) NumQuads() int { return o.quadCount }

// VisitProxies implements Op.
func (o *drawAtlasOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *drawAtlasOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *drawAtlasOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp: with per-sprite colors the GP output is unknown, otherwise the constant paint color; if
// analysis resolves the color to a constant anyway (an overriding processor chain), the color attribute is dropped.
func (o *drawAtlasOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	var gpColor ProcessorAnalysisColor // zero value = unknown
	if !o.hasColors {
		gpColor = AnalysisColorConstant(o.color)
	}
	result := o.helper.FinalizeProcessors(caps, clip, clampType, AnalysisCoverageNone, &gpColor)
	if c, isConstant := gpColor.IsConstant(); isConstant {
		o.color = c
		o.hasColors = false
	}
	return result
}

// createProgramInfo builds the program info for this op's draw: the default geometry processor with explicit local
// coords, solid coverage, and either the uniform paint color or the premul per-vertex color attribute.
func (o *drawAtlasOp) createProgramInfo(state *OpFlushState) {
	colorType := ColorTypePremulUniform
	if o.hasColors {
		colorType = ColorTypePremulAttribute
	}
	gp := MakeDefaultGeoProc(colorType, o.color, CoverageTypeSolid, 0xff,
		LocalCoordsTypeHasExplicit, nil, &o.viewMatrix)
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op (the QuadHelper lane: 4 vertices per sprite against the shared non-AA quad index buffer).
func (o *drawAtlasOp) OnPrepare(state *OpFlushState) {
	if o.quadCount == 0 {
		return
	}
	stride := atlasVertexStride(o.hasColors)
	vdata, vertexBuffer, baseVertex := state.MakeVertexSpace(uint64(stride), 4*o.quadCount)
	if vdata == nil {
		// Could not allocate vertices.
		return
	}
	o.vertexBuffer = vertexBuffer
	o.baseVertex = baseVertex

	offset := 0
	for i := range o.geoData {
		g := &o.geoData[i]
		if g.hasColors == o.hasColors {
			offset += copy(vdata[offset:], g.verts)
			continue
		}
		// Finalize dropped the color attribute after this geometry was baked with interleaved colors (g.hasColors &&
		// !o.hasColors — the only possible mismatch, since Finalize never adds colors): re-copy per vertex, skipping
		// the 4 color bytes. A naive copy of the baked bytes against the new stride here would corrupt the stream; the
		// mismatch is unreachable in practice (it needs an analysis that overrides an unknown input color), but the
		// copy is made safe regardless.
		bakedStride := atlasVertexStride(true)
		for v := 0; v*bakedStride < len(g.verts); v++ {
			src := g.verts[v*bakedStride:]
			offset += copy(vdata[offset:], src[:8])             // position
			offset += copy(vdata[offset:], src[12:bakedStride]) // local coord
		}
	}

	o.indexBuffer = GetQuadIndexBuffer(state, IndexBufferOptionIndexedRects)
}

// OnExecute implements Op over this codebase's flush machinery. The draw is the QuadHelper mesh: an indexed-pattern
// draw over the shared non-AA quad index buffer, chunked by its 4096-quad repetition limit.
func (o *drawAtlasOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.vertexBuffer == nil || o.indexBuffer == nil {
		return
	}
	if o.programInfo == nil {
		o.createProgramInfo(state)
	}

	renderPass := state.OpsRenderPass()
	if !renderPass.BindPipeline(o.programInfo, chainBounds) {
		return
	}
	if o.programInfo.Pipeline().IsScissorTestEnabled() {
		renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
	}
	renderPass.BindBuffers(o.indexBuffer, nil, o.vertexBuffer)
	renderPass.BindTextures(o.programInfo.GeomProc(), nil, o.programInfo.Pipeline())
	renderPass.DrawIndexPattern(NumIndicesPerNonAAQuad, o.quadCount, MaxNumNonAAQuads,
		NumVertsPerNonAAQuad, o.baseVertex)
}

// OnCombineIfPossible implements Op: merges on a compatible helper (processor set / AA / pipeline flags), matching view
// matrix, matching color-inclusion (and matching constant color when there is no color attribute), and no quad-count
// overflow.
func (o *drawAtlasOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*drawAtlasOp)
	if !ok {
		return CombineResultCannotCombine
	}

	if !o.helper.IsCompatible(&that.helper, false) {
		return CombineResultCannotCombine
	}

	// We currently use a uniform view matrix for this op.
	if !o.viewMatrix.CheapEqual(&that.viewMatrix) {
		return CombineResultCannotCombine
	}

	if o.hasColors != that.hasColors {
		return CombineResultCannotCombine
	}

	if !o.hasColors && o.color != that.color {
		return CombineResultCannotCombine
	}

	if int64(o.quadCount)+int64(that.quadCount) > math.MaxInt32 {
		return CombineResultCannotCombine
	}

	o.geoData = append(o.geoData, that.geoData...)
	o.quadCount += that.quadCount
	return CombineResultMerged
}
