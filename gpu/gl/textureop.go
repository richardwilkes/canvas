// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The batched textured-quad op behind drawImageRect's fast path — src-over textured quads that sample the texture in
// the geometry processor rather than through FP trees. Trims: the multi-proxy image-set form and the cross-proxy
// chaining lane are unreachable/deferred (no public entry point exposes image sets; different-texture draws simply
// don't batch), and the texture color-space xform is identity in an sRGB-only pipeline.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

var textureOpClassID = GenOpClassID()

// axisAlignedQuadSize returns the lengths of the vertical and horizontal edges of an axis-aligned quad.
func axisAlignedQuadSize(quad *Quad) (w, h float32) {
	if quad.QuadType() != QuadTypeAxisAligned {
		panic("axis-aligned quad required")
	}
	// Simplification of the regular edge length equation, since it's axis aligned and can avoid sqrt.
	w = geom.ScalarAbs(quad.X(2)-quad.X(0)) + geom.ScalarAbs(quad.Y(2)-quad.Y(0))
	h = geom.ScalarAbs(quad.X(1)-quad.X(0)) + geom.ScalarAbs(quad.Y(1)-quad.Y(0))
	return w, h
}

func scalarFraction(x float32) float32 { return float32(math.Mod(float64(x), 1)) }

// FilterAndMipmapHaveEffect reports whether filtering and mipmapping would each have a visible effect for the given
// src/dst quad pair. A false result means the corresponding mode can be safely downgraded to nearest / no mipmaps.
func FilterAndMipmapHaveEffect(srcQuad, dstQuad *Quad) (filter, mm bool) {
	// If not axis-aligned in src or dst, then always say it has an effect.
	if srcQuad.QuadType() != QuadTypeAxisAligned ||
		dstQuad.QuadType() != QuadTypeAxisAligned {
		return true, true
	}

	srcRect, srcOK := srcQuad.AsRect()
	dstRect, dstOK := dstQuad.AsRect()
	if srcOK && dstOK {
		// Disable filtering when there is no scaling (width and height are the same), and the top-left corners have the
		// same fraction (so src and dst snap to the pixel grid identically).
		filter = srcRect.Width() != dstRect.Width() || srcRect.Height() != dstRect.Height() ||
			scalarFraction(srcRect.Left) != scalarFraction(dstRect.Left) ||
			scalarFraction(srcRect.Top) != scalarFraction(dstRect.Top)
		mm = srcRect.Width() > dstRect.Width() || srcRect.Height() > dstRect.Height()
		return filter, mm
	}
	// Extract edge lengths. Although the quads are axis-aligned, the local coordinate system is transformed such that
	// fractionally-aligned sample centers will not align with the device coordinate system, so disable filtering when
	// edges are the same length and both quads' 0th vertices are integer aligned.
	srcW, srcH := axisAlignedQuadSize(srcQuad)
	dstW, dstH := axisAlignedQuadSize(dstQuad)
	filter = srcW != dstW || srcH != dstH || !scalarIsInt(srcQuad.X(0)) ||
		!scalarIsInt(srcQuad.Y(0)) || !scalarIsInt(dstQuad.X(0)) || !scalarIsInt(dstQuad.Y(0))
	mm = srcW > dstW || srcH > dstH
	return filter, mm
}

// normalizationParams: [x * iw, y * invH + yOffset] normalizes src coords for regular and rectangle textures, with or
// without origin correction.
type normalizationParams struct {
	iw      float32 // 1 / width of texture, or 1.0 for texture rectangles
	invH    float32 // 1 / height (1.0 for tex rects), negated for bottom-left origin
	yOffset float32 // 0 for top-left origin, height of [normalized] tex if bottom-left
}

// proxyNormalizationParams derives the normalization params for sampling proxy given its origin.
func proxyNormalizationParams(proxy *SurfaceProxy, origin gpu.SurfaceOrigin) normalizationParams {
	dimensions := proxy.BackingStoreDimensions()
	var iw, ih, h float32
	if proxy.TextureType() == gpu.TextureTypeRectangle {
		iw = 1
		ih = 1
		h = float32(dimensions.Height)
	} else {
		iw = 1 / float32(dimensions.Width)
		ih = 1 / float32(dimensions.Height)
		h = 1
	}
	if origin == gpu.OriginBottomLeft {
		return normalizationParams{iw: iw, invH: -ih, yOffset: h}
	}
	return normalizationParams{iw: iw, invH: ih, yOffset: 0}
}

// textureOpLargeSubset is a subset that doesn't restrict normalized coords.
var textureOpLargeSubset = geom.Rect{
	Left: -100000, Top: -100000, Right: 1000000,
	Bottom: 1000000,
}

// normalizeAndInsetSubset normalizes subsetRect and insets it to pixel centers. subsetRect nil means no constraint.
func normalizeAndInsetSubset(filter gpu.FilterMode, params normalizationParams, subsetRect *geom.Rect) geom.Rect {
	if subsetRect == nil {
		return textureOpLargeSubset
	}

	l := subsetRect.Left
	t := subsetRect.Top
	r := subsetRect.Right
	b := subsetRect.Bottom
	if filter == gpu.FilterModeNearest {
		// Make sure our insetting puts us at pixel centers: ltrb = floor(ltrb*flipHi)*flipHi with flipHi = (1, 1, -1,
		// -1).
		l = float32(math.Floor(float64(l)))
		t = float32(math.Floor(float64(t)))
		r = -float32(math.Floor(float64(-r)))
		b = -float32(math.Floor(float64(-b)))
	}
	// Inset with pin to the rect center.
	const linearInset = 0.5 + 1.0/1024 // half a pixel plus a small epsilon
	l += linearInset
	t += linearInset
	r -= linearInset
	b -= linearInset
	midX := (l + r) * 0.5
	midY := (t + b) * 0.5
	l = min(l, midX)
	t = min(t, midY)
	r = max(r, midX)
	b = max(b, midY)

	// Normalize and offset.
	l *= params.iw
	r *= params.iw
	t = t*params.invH + params.yOffset
	b = b*params.invH + params.yOffset
	if params.invH < 0 {
		// Flip top and bottom to keep the rect sorted.
		t, b = b, t
	}
	return geom.Rect{Left: l, Top: t, Right: r, Bottom: b}
}

// normalizeSrcQuad normalizes a quad's coordinates in place per params.
func normalizeSrcQuad(params normalizationParams, srcQuad *Quad) {
	if srcQuad.QuadType() == QuadTypePerspective {
		panic("src quad cannot have perspective")
	}
	xs := srcQuad.Xs()
	ys := srcQuad.Ys()
	for i := range 4 {
		xs[i] *= params.iw
		ys[i] = ys[i]*params.invH + params.yOffset
	}
}

// safeToIgnoreSubsetRect reports whether the subset rect constraint can be dropped without changing the result.
func safeToIgnoreSubsetRect(aaType gpu.AAType, filter gpu.FilterMode, quad *DrawQuad, subsetRect geom.Rect) bool {
	// If both the device and local quads are axis-aligned and filtering is off, the local quad can push all the way up
	// to the edges of the subset rect and the sampler shouldn't overshoot. Unfortunately, antialiasing adds enough
	// jitter that we can only rely on this in the non-antialiased case.
	localBounds := quad.Local.Bounds()
	if aaType == gpu.AATypeNone && filter == gpu.FilterModeNearest &&
		quad.Device.QuadType() == QuadTypeAxisAligned &&
		quad.Local.QuadType() == QuadTypeAxisAligned &&
		subsetRect.ContainsRect(localBounds) {
		return true
	}

	// If the local quad is inset by at least 0.5 pixels into the subset rect's bounds, the sampler shouldn't overshoot,
	// even when antialiasing and filtering are taken into account.
	const linearInset = 0.5 + 1.0/1024
	return subsetRect.Inset(linearInset, linearInset).ContainsRect(localBounds)
}

// textureColorSubsetAndAA is the per-quad metadata a textureOp instance carries: color, subset rect, and edge AA flags.
type textureColorSubsetAndAA struct {
	color colorcore.PMColor4f
	// If the op doesn't use subsets, this is ignored. If the op uses subsets and the specific entry does not, this rect
	// equals the large rect, so it automatically has no effect.
	subsetRect geom.Rect
	aaFlags    gpu.QuadAAFlags
}

// textureOp draws a batch of textured quads sampling a single proxy.
type textureOp struct {
	drawOpNoClipToShape
	vertexBuffer AnyBuffer
	quads        *QuadBuffer[textureColorSubsetAndAA]
	proxy        *SurfaceProxy
	programInfo  *ProgramInfo
	indexBuffer  *Buffer
	OpBase
	baseVertex int
	// Prepared state: filled in by OnPrepare once the mesh is built.
	vertexSpec VertexSpec
	aaType     gpu.AAType
	colorType  QuadColorType
	// Metadata: fixed draw parameters that don't depend on preparing the mesh.
	swizzle    gpu.Swizzle
	specValid  bool
	saturate   bool
	subset     bool
	mipmapMode gpu.MipmapMode
	filter     gpu.FilterMode
}

// newTextureOpImpl builds a textureOp from a single quad. The view's origin is applied immediately, so only the proxy
// and swizzle are retained.
func newTextureOpImpl(proxyView SurfaceProxyView, filter gpu.FilterMode, mm gpu.MipmapMode, color colorcore.PMColor4f, saturate bool, aaType gpu.AAType, quad *DrawQuad, subsetRect *geom.Rect) *textureOp {
	o := &textureOp{
		proxy:      proxyView.Proxy(),
		swizzle:    proxyView.Swizzle(),
		filter:     filter,
		mipmapMode: mm,
		saturate:   saturate,
		subset:     subsetRect != nil,
		colorType:  QuadColorTypeNone,
	}
	o.InitOp(textureOpClassID, o)
	o.quads = NewQuadBuffer[textureColorSubsetAndAA](1, true /* includes locals */)

	// Clean up disparities between the overall aa type and edge configuration and apply optimizations based on the rect
	// and matrix when appropriate.
	aaType, quad.EdgeFlags = ResolveAAType(aaType, quad.EdgeFlags, &quad.Device)
	o.aaType = aaType

	// We expect the caller to have already caught this optimization.
	if subsetRect != nil &&
		subsetRect.ContainsRect(o.proxy.BackingStoreBoundsIRect().ToRect()) {
		panic("subset containing the backing store should have been dropped by the caller")
	}

	// We may have had a strict constraint with nearest filter solely due to possible AA bloat. Try to identify cases
	// where the subsetting isn't actually necessary, and skip it.
	if subsetRect != nil && safeToIgnoreSubsetRect(aaType, filter, quad, *subsetRect) {
		subsetRect = nil
		o.subset = false
	}

	// Normalize src coordinates and the subset (if set).
	params := proxyNormalizationParams(o.proxy, proxyView.Origin())
	normalizeSrcQuad(params, &quad.Local)
	subset := normalizeAndInsetSubset(filter, params, subsetRect)

	// Set bounds before clipping so we don't have to worry about unioning the bounds of the two potential quads
	// (Quad.Bounds() is perspective-safe).
	hairline := WillUseHairline(&quad.Device, aaType, quad.EdgeFlags)
	o.SetBounds(quad.Device.Bounds(), HasAABloat(aaType == gpu.AATypeCoverage),
		IsHairline(hairline))
	o.appendQuad(quad, color, subset)
	return o
}

// appendQuad adds quad (clipped against w=0) to the batch with color and subset metadata.
func (o *textureOp) appendQuad(quad *DrawQuad, color colorcore.PMColor4f, subset geom.Rect) {
	var extra DrawQuad
	// Always clip to W0 to stay consistent with Quad.Bounds().
	quadCount := ClipToW0(quad, &extra)
	if quadCount == 0 {
		// We can't discard the op at this point, but disable AA flags so it won't go through inset/outset processing.
		quad.EdgeFlags = gpu.QuadAAFlagsNone
		quadCount = 1
	}
	o.quads.Append(&quad.Device, textureColorSubsetAndAA{
		color: color, subsetRect: subset,
		aaFlags: quad.EdgeFlags,
	}, &quad.Local)
	if quadCount > 1 {
		o.quads.Append(&extra.Device, textureColorSubsetAndAA{
			color: color, subsetRect: subset,
			aaFlags: extra.EdgeFlags,
		}, &extra.Local)
	}
}

// Name implements Op.
func (o *textureOp) Name() string { return "TextureOp" }

// NumQuads is test access.
func (o *textureOp) NumQuads() int { return o.quads.Count() }

// VisitProxies implements Op.
func (o *textureOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	mipped := o.mipmapMode != gpu.MipmapModeNone
	fn(o.proxy, gpu.Mipmapped(mipped))
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	}
}

// UsesMSAA implements DrawOp (fixedFunctionFlags).
func (o *textureOp) UsesMSAA() bool { return o.aaType == gpu.AATypeMSAA }

// UsesStencil implements DrawOp.
func (o *textureOp) UsesStencil() bool { return false }

// Finalize implements DrawOp: determines the narrowest per-vertex color encoding across all quads.
func (o *textureOp) Finalize(*gpu.Caps, *AppliedClip, gpu.ClampType) ProcessorAnalysis {
	if o.colorType != QuadColorTypeNone {
		panic("color type already finalized")
	}
	iter := o.quads.Metadata()
	for iter.Next() {
		o.colorType = max(o.colorType, MinQuadColorType(iter.Get().color))
	}
	return EmptyProcessorSetAnalysis()
}

// characterize builds the vertex spec for the batch, once all quads have been added.
func (o *textureOp) characterize() {
	indexBufferOption := CalcIndexBufferOption(o.aaType, o.quads.Count())
	subset := QuadSubsetNo
	if o.subset {
		subset = QuadSubsetYes
	}
	o.vertexSpec = NewVertexSpec(o.quads.DeviceQuadType(), o.colorType,
		o.quads.LocalQuadType(), true /* hasLocal */, subset, o.aaType,
		true /* alpha as coverage */, indexBufferOption)
	o.specValid = true
	if o.quads.Count() > QuadLimit(indexBufferOption) {
		panic("texture op exceeds the index buffer's quad limit")
	}
}

// createProgramInfo builds the op's program info.
func (o *textureOp) createProgramInfo(state *OpFlushState) {
	samplerState := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, o.filter,
		gpu.MipmapModeNone)
	gp := MakeTexturedQuadProcessor(&o.vertexSpec, o.proxy.TextureType(),
		samplerState, o.swizzle, o.saturate)
	if gp.gpBase().VertexStride() != o.vertexSpec.VertexSize() {
		panic("GP vertex stride out of sync with the vertex spec")
	}

	args := state.OpArgs()
	pipeline := NewPipeline(&PipelineInitArgs{
		InputFlags:   PipelineInputFlagNone,
		Caps:         state.Caps(),
		DstProxyView: *args.DstProxyView(),
		WriteSwizzle: args.WriteView().Swizzle(),
	}, MakeEmptyProcessorSet(), args.AppliedClip())
	o.programInfo = NewProgramInfo(state.Caps(), args.WriteView(), args.UsesMSAASurface(),
		pipeline, UnusedStencilSettings(), gp, o.vertexSpec.PrimitiveType(),
		args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op: builds the vertex (and index) buffers for the batch.
func (o *textureOp) OnPrepare(state *OpFlushState) {
	if !o.specValid {
		o.characterize()
	}

	vertexSize := o.vertexSpec.VertexSize()
	totalNumVertices := o.quads.Count() * o.vertexSpec.VerticesPerQuad()
	vdata, vertexBuffer, baseVertex := state.MakeVertexSpace(uint64(vertexSize),
		totalNumVertices)
	if vdata == nil {
		return
	}
	o.vertexBuffer = vertexBuffer
	o.baseVertex = baseVertex

	if o.vertexSpec.NeedsIndexBuffer() {
		o.indexBuffer = GetQuadIndexBuffer(state, o.vertexSpec.IndexBufferOption())
		if o.indexBuffer == nil {
			return
		}
	}

	// FillInVertices reduced to this op (no chain).
	tess := NewQuadTessellator(o.vertexSpec, vdata)
	iter := o.quads.Iterator()
	for iter.Next() {
		if !iter.IsLocalValid() {
			panic("texture op quads always carry local coords")
		}
		info := iter.Metadata()
		tess.Append(iter.DeviceQuad(), iter.LocalQuad(), info.color, info.subsetRect,
			info.aaFlags)
	}
}

// OnExecute implements Op.
func (o *textureOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if o.vertexBuffer == nil {
		return
	}
	if o.vertexSpec.NeedsIndexBuffer() && o.indexBuffer == nil {
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
	renderPass.BindTextures(o.programInfo.GeomProc(), o.proxy, o.programInfo.Pipeline())
	IssueQuadDraw(state.Caps(), renderPass, &o.vertexSpec, 0, o.quads.Count(), o.baseVertex)
}

// OnCombineIfPossible implements Op: merges compatible texture ops into one batched draw (minus the cross-proxy
// chaining lane).
func (o *textureOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*textureOp)
	if !ok {
		return CombineResultCannotCombine
	}

	if o.subset != that.subset {
		// It is technically possible to combine operations across subset modes, but performance testing suggests it's
		// better to make more draw calls where some take advantage of the more optimal shader path without coordinate
		// clamping.
		return CombineResultCannotCombine
	}

	upgradeToCoverageAAOnMerge := false
	if o.aaType != that.aaType {
		if !CanUpgradeAAOnMerge(o.aaType, that.aaType) {
			return CombineResultCannotCombine
		}
		upgradeToCoverageAAOnMerge = true
	}

	if CombinedQuadCountWillOverflow(o.aaType, upgradeToCoverageAAOnMerge,
		o.quads.Count()+that.quads.Count()) {
		return CombineResultCannotCombine
	}

	if o.saturate != that.saturate {
		return CombineResultCannotCombine
	}
	if o.filter != that.filter {
		return CombineResultCannotCombine
	}
	if o.mipmapMode != that.mipmapMode {
		return CombineResultCannotCombine
	}
	if o.swizzle != that.swizzle {
		return CombineResultCannotCombine
	}
	if o.proxy != that.proxy {
		// We can't merge across different proxies (the chaining lane is deferred).
		return CombineResultCannotCombine
	}

	o.subset = o.subset || that.subset
	o.colorType = max(o.colorType, that.colorType)

	// Concatenate the quad lists together.
	o.quads.Concat(that.quads)

	if upgradeToCoverageAAOnMerge {
		o.aaType = gpu.AATypeCoverage
	}
	return CombineResultMerged
}

//////////////////////////////////////////////////////////////////////////////
// Entry points.

// NewTextureOp builds a textured-quad draw op: src-over textured quads go through textureOp directly; other blends are
// emulated with FillRectOp and a texture FP.
func NewTextureOp(caps *Caps, proxyView SurfaceProxyView, srcAlphaType gpu.AlphaType, filter gpu.FilterMode, mm gpu.MipmapMode, color colorcore.PMColor4f, saturate bool, blendMode raster.BlendMode, aaType gpu.AAType, quad *DrawQuad, subset *geom.Rect) DrawOp {
	// Apply optimizations that are valid whether or not using TextureOp or FillRectOp.
	if subset != nil && subset.ContainsRect(proxyView.Proxy().BackingStoreBoundsIRect().ToRect()) {
		// No need for a shader-based subset if hardware clamping achieves the same effect.
		subset = nil
	}

	if filter != gpu.FilterModeNearest || mm != gpu.MipmapModeNone {
		mustFilter, mustMM := FilterAndMipmapHaveEffect(&quad.Local, &quad.Device)
		if !mustFilter {
			filter = gpu.FilterModeNearest
		}
		if !mustMM {
			mm = gpu.MipmapModeNone
		}
	}

	if blendMode == raster.BlendSrcOver {
		return newTextureOpImpl(proxyView, filter, mm, color, saturate, aaType, quad, subset)
	}

	// Emulate complex blending using FillRectOp.
	samplerState := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, filter, mm)
	paint := NewPaint()
	paint.SetColor4f(color)
	paint.SetXPFactory(XPFactoryFromBlendMode(blendMode))

	identity := geom.IdentityMatrix()
	var fp FragmentProcessor
	if subset != nil {
		fp = MakeTextureEffectSubset(proxyView, srcAlphaType, identity, samplerState, *subset,
			caps, [4]float32{}, false)
	} else {
		fp = MakeTextureEffect(proxyView, srcAlphaType, identity, samplerState, caps,
			[4]float32{})
	}
	fp = BlendFP(fp, nil, raster.BlendModulate)
	if saturate {
		// Clamping the output is unreachable: the clamp type is always auto for every color type in the matrix, so
		// callers always pass saturate == false.
		panic("saturate is unreachable under the color-type matrix")
	}
	paint.SetColorFragmentProcessor(fp)
	return NewFillRectOp(paint, aaType, quad, nil, PipelineInputFlagNone)
}

// DrawTexture draws a textured rect, including the DMSAA lane that routes the draw through FillRRectOp via
// FillRectToRect (the color-space xform collapses to the identity).
func (sdc *SurfaceDrawContext) DrawTexture(clip Clip, view SurfaceProxyView, srcAlphaType gpu.AlphaType, filter gpu.FilterMode, mm gpu.MipmapMode, blendMode raster.BlendMode, color colorcore.PMColor4f, srcRect, dstRect geom.Rect, edgeAA gpu.QuadAAFlags, constraint SrcRectConstraint, viewMatrix *geom.Matrix) {
	// If we are using dmsaa then go through FillRRectOp (via fillRectToRect).
	if (sdc.alwaysAntialias() || sdc.Caps().ReducedShaderMode()) && edgeAA != gpu.QuadAAFlagsNone {
		deviceQuad := MakeQuadFromRect(dstRect, viewMatrix)
		srcQuad := MakeQuadFromRectNoTransform(srcRect)
		// deviceQuad is passed first; only alignment matters here, not the direction of scaling.
		mustFilter, _ := FilterAndMipmapHaveEffect(&deviceQuad, &srcQuad)
		if !mustFilter {
			// Filtering that has no effect is dropped so a pixel-aligned blit stays exact.
			filter = gpu.FilterModeNearest
		}

		paint := NewPaint()
		paint.SetColor4f(color)
		identity := geom.IdentityMatrix()
		var fp FragmentProcessor
		if constraint == SrcRectConstraintStrict {
			fp = MakeTextureEffectSubset(view, srcAlphaType, identity,
				gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, filter, mm), srcRect,
				sdc.Caps(), [4]float32{}, false)
		} else {
			fp = MakeTextureEffectSimple(view, srcAlphaType, identity, filter, mm)
		}
		fp = BlendFP(fp, nil, raster.BlendModulate)
		paint.SetColorFragmentProcessor(fp)
		if blendMode != raster.BlendSrcOver {
			paint.SetXPFactory(XPFactoryFromBlendMode(blendMode))
		}
		sdc.FillRectToRect(clip, paint, gpu.AAYes, viewMatrix, dstRect, srcRect)
		return
	}

	var subset *geom.Rect
	if constraint == SrcRectConstraintStrict {
		subset = &srcRect
	}
	quad := DrawQuad{
		Device: MakeQuadFromRect(dstRect, viewMatrix),
		Local:  MakeQuadFromRectNoTransform(srcRect), EdgeFlags: edgeAA,
	}
	sdc.DrawTexturedQuad(clip, view, srcAlphaType, filter, mm, color, blendMode, &quad, subset)
}

// SrcRectConstraint controls how strictly a source rect clamps sampling to avoid bleeding neighboring texels.
type SrcRectConstraint uint8

// SrcRectConstraint values.
const (
	SrcRectConstraintStrict SrcRectConstraint = iota
	SrcRectConstraintFast
)

// DrawTexturedQuad draws a textured quad through the quad-optimization funnel.
func (sdc *SurfaceDrawContext) DrawTexturedQuad(clip Clip, proxyView SurfaceProxyView, srcAlphaType gpu.AlphaType, filter gpu.FilterMode, mm gpu.MipmapMode, color colorcore.PMColor4f, blendMode raster.BlendMode, quad *DrawQuad, subset *geom.Rect) {
	if sdc.ctx.Abandoned() {
		return
	}
	if proxyView.AsTextureProxy() == nil {
		panic("drawTexturedQuad requires a texture proxy")
	}

	// Functionally this is very similar to drawFilledQuad except that there's no constColor to enable the
	// quadOptSubmitted optimizations, no stencil settings support, and it's a TextureOp.
	opt := sdc.attemptQuadOptimization(clip, nil /* stencil */, quad, nil /* paint */)
	if opt == quadOptSubmitted {
		panic("texture quads cannot reduce to native clears")
	}
	if opt != quadOptDiscarded {
		// Add the texture op if not discarded.
		finalClip := clip
		if opt == quadOptClipApplied {
			finalClip = nil
		}
		aaType := sdc.chooseAAType(gpu.AA(quad.EdgeFlags != gpu.QuadAAFlagsNone))
		// The clamp type is always auto for the color types, so saturate is never needed.
		sdc.AddDrawOp(finalClip, NewTextureOp(sdc.Caps(), proxyView, srcAlphaType, filter, mm,
			color, false /* saturate */, blendMode, aaType, quad, subset))
	}
}
