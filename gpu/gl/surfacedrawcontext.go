// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The draw surface's paint+clip fill-rect/quad entry points, the quad-optimization funnel (discard / native clear /
// clip folding / rrect-clip reduction / cropping), and the full AddDrawOp pipeline (clip application → processor
// finalization → dst-copy setup → OpsTask recording), including the DMSAA lanes: canUseDynamicMSAA, the per-op MSAA
// escalation in AddDrawOp, and the FillRRectOp lane in FillRectToRect.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// chooseAAType always triggers DMSAA when it's available (the coverage ops that know how to handle both single and
// multisample targets without popping do so without calling chooseAAType).
func (sdc *SurfaceDrawContext) chooseAAType(aa gpu.AA) gpu.AAType {
	if sdc.NumSamples() > 1 || sdc.canUseDynamicMSAA {
		return gpu.AATypeMSAA
	}
	if aa == gpu.AAYes {
		return gpu.AATypeCoverage
	}
	return gpu.AATypeNone
}

// quadOptimization is the outcome of attemptQuadOptimization.
type quadOptimization int32

const (
	// quadOptDiscarded: the draw doesn't intersect the clip or render target; add no op.
	quadOptDiscarded quadOptimization = iota
	// quadOptSubmitted: the draw was converted to another op (clear) and appended; add no op.
	quadOptSubmitted
	// quadOptClipApplied: the clip was folded into the device quad; the caller adds the op without the clip.
	quadOptClipApplied
	// quadOptCropped: no change to the clip, but the quad was updated to better fit the clip/render target; the caller
	// adds the op with the clip.
	quadOptCropped
)

// attemptQuadOptimization tries to reduce a quad draw to a discard, a native clear, or a clip fold, falling back to a
// crop. paint may be nil (the textured-quad lane); ss may be nil (unused stencil settings).
func (sdc *SurfaceDrawContext) attemptQuadOptimization(clip Clip, ss *UserStencilSettings, quad *DrawQuad, paint *Paint) quadOptimization {
	// Optimization requirements:
	//  1. quadOptDiscarded applies when clip bounds and quad bounds do not intersect.
	//  2. quadOptSubmitted applies when constColor and either the final geometry is a pixel-aligned rect (rect clip
	//     and (rect quad or quad covers clip)), or the clip is an rrect the quad covers.
	//  3. quadOptClipApplied applies when rect clip and (rect quad or quad covers clip).
	//  4. quadOptCropped in all other scenarios (although a crop may be a no-op).
	var constColor *colorcore.PMColor4f
	if ss == nil && paint != nil && !paint.HasCoverageFragmentProcessor() {
		if paintColor, isConstant := paint.IsConstantBlendedColor(); isConstant {
			// Only consider clears/rrects when it's easy to guarantee 100% fill with a single color.
			constColor = &paintColor
		}
	}

	// Save the old AA flags since CropToRect will modify 'quad' and if quadOptCropped is returned it's better to
	// just keep the old flags instead of introducing mixed edge flags.
	oldFlags := quad.EdgeFlags

	// Use the logical size of the render target, which allows for "fullscreen" clears even if the render target has an
	// approximate backing fit.
	rtRect := sdc.AsSurfaceProxy().BoundsIRect().ToRect()

	// For historical reasons, we assume AA for exact bounds checking in IsOutsideClip.
	drawBounds := quad.Device.Bounds()
	if !quad.Device.IsFinite() || drawBounds.IsEmpty() ||
		IsOutsideClip(geom.IRectSize(sdc.Dimensions()), drawBounds, gpu.AAYes) {
		return quadOptDiscarded
	} else if WillUseHairline(&quad.Device, gpu.AATypeCoverage, quad.EdgeFlags) {
		// Don't try to apply the clip early if we know rendering will use hairline methods, as this has an effect on
		// the op bounds not otherwise taken into account here.
		return quadOptCropped
	}

	drawUsesAA := gpu.AA(quad.EdgeFlags != gpu.QuadAAFlagsNone)
	conservativeCrop := func() {
		const largeDrawLimit = 15000
		// Crop the quad to the render target. This doesn't change the visual results of drawing but is meant to help
		// numerical stability for excessively large draws.
		if drawBounds.Width() > largeDrawLimit || drawBounds.Height() > largeDrawLimit {
			CropToRect(rtRect, drawUsesAA, quad, constColor == nil /* compute local */)
		}
	}

	simpleColor := ss == nil && constColor != nil
	var result PreClipResult
	if clip != nil {
		result = clip.PreApply(drawBounds, drawUsesAA)
	} else {
		result = MakePreClipResult(ClipEffectUnclipped)
	}
	switch result.Effect {
	case ClipEffectClippedOut:
		return quadOptDiscarded
	case ClipEffectUnclipped:
		if !simpleColor {
			conservativeCrop()
			return quadOptClipApplied
		}
		// Update the result to store the render target bounds and fall through to attempt the draw→native-clear
		// optimization. Pick an AA value such that any geometric clipping doesn't need to change aa or edge flags
		// (since this is on pixel boundaries it draws the same regardless).
		result = MakePreClipResultRRect(geom.MakeRRect(rtRect, 0, 0), drawUsesAA)
	case ClipEffectClipped:
		if !result.IsRRect || (ss != nil && result.AA != drawUsesAA) ||
			(result.RRect.Type != geom.RRectRect && !simpleColor) {
			// The clip and draw state are too complicated to try and reduce.
			conservativeCrop()
			return quadOptCropped
		}
	}

	// If we reached here, we know we're an axis-aligned clip that is either a rect or a round rect, so we can
	// potentially combine it with the draw geometry so that no clipping is needed.
	clippedBounds := result.RRect.Rect
	clippedBounds.Intersect(rtRect)
	if !drawBounds.Intersect(clippedBounds) {
		// The fractional bounds aren't actually inside the clip (preApply can think in terms of rounded-out bounds);
		// discard the draw.
		return quadOptDiscarded
	}
	// Guard against the clipped draw turning into a hairline draw after intersection.
	if drawBounds.Width() < 1 || drawBounds.Height() < 1 {
		return quadOptCropped
	}

	if result.RRect.Type == geom.RRectRect {
		// No rounded corners, so we might become a native clear or we might modify geometry and edge flags to represent
		// the intersected shape of the clip and draw.
		if CropToRect(clippedBounds, result.AA, quad, constColor == nil /* compute local */) {
			if simpleColor && quad.Device.QuadType() == QuadTypeAxisAligned {
				// The clear optimization is possible.
				drawBounds = quad.Device.Bounds()
				if drawBounds.ContainsRect(rtRect) {
					// Fullscreen clear.
					sdc.Clear([4]float32{constColor.R, constColor.G, constColor.B, constColor.A})
					return quadOptSubmitted
				} else if IsPixelAligned(drawBounds) && drawBounds.Width() > 256 &&
					drawBounds.Height() > 256 {
					// Scissor + clear (rounding shouldn't do anything since we are pixel aligned).
					scissorRect := drawBounds.Round()
					sdc.ClearRect(scissorRect,
						[4]float32{constColor.R, constColor.G, constColor.B, constColor.A})
					return quadOptSubmitted
				}
			}
			return quadOptClipApplied
		}
	} else {
		// Rounded corners and constant filled color (limited to solid colors because there is no way to use custom
		// local coordinates with drawRRect).
		if !simpleColor {
			panic("rrect-clip reduction requires a simple color")
		}
		if CropToRect(clippedBounds, result.AA, quad, false /* compute local */) &&
			quad.Device.QuadType() == QuadTypeAxisAligned &&
			quad.Device.Bounds().ContainsRect(clippedBounds) {
			// Since the cropped quad became a rectangle which covered the bounds of the rrect, we can draw the rrect
			// directly and ignore the edge flags.
			identity := geom.IdentityMatrix()
			sdc.DrawRRect(nil, paint, result.AA, &identity, result.RRect, nil)
			return quadOptSubmitted
		}
	}

	// The quads have been updated to better fit the clip bounds, but we can't get rid of the clip entirely.
	quad.EdgeFlags = oldFlags
	return quadOptCropped
}

// DrawFilledQuad draws quad filled with paint (subject to the quad-optimization funnel). The paint is consumed; ss may
// be nil.
func (sdc *SurfaceDrawContext) DrawFilledQuad(clip Clip, paint *Paint, quad *DrawQuad, ss *UserStencilSettings) {
	if sdc.ctx.Abandoned() {
		return
	}
	opt := sdc.attemptQuadOptimization(clip, ss, quad, paint)
	if opt >= quadOptClipApplied {
		// These optimizations require the caller to add an op itself.
		finalClip := clip
		if opt == quadOptClipApplied {
			finalClip = nil
		}
		// The quad being drawn requires AA if any of its edges requires AA.
		aa := gpu.AA(quad.EdgeFlags != gpu.QuadAAFlagsNone)
		var aaType gpu.AAType
		switch {
		case ss != nil:
			if aa == gpu.AAYes {
				aaType = gpu.AATypeMSAA
			} else {
				aaType = gpu.AATypeNone
			}
		case sdc.canUseDynamicMSAA && aa == gpu.AANo:
			// The device layer ensures aa is always AAYes when using dmsaa. If the caller calls into here with AANo,
			// trust that they know what they're doing and that the rendering will be equal with or without msaa.
			aaType = gpu.AATypeNone
		default:
			aaType = sdc.chooseAAType(aa)
		}
		sdc.AddDrawOp(finalClip, NewFillRectOp(paint, aaType, quad, ss, PipelineInputFlagNone))
	}
	// All other optimization levels were completely handled inside attempt(), so no extra op needed.
}

// FillRectToRect fills rectToDraw with the paint using localRect for the local coordinates. Under DMSAA (or reduced
// shader mode) it attempts to draw the rect with FillRRectOp first.
func (sdc *SurfaceDrawContext) FillRectToRect(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, rectToDraw, localRect geom.Rect) {
	edgeFlags := gpu.QuadAAFlagsNone
	if aa == gpu.AAYes {
		edgeFlags = gpu.QuadAAFlagsAll
	}
	quad := DrawQuad{
		Device: MakeQuadFromRect(rectToDraw, viewMatrix),
		Local:  MakeQuadFromRectNoTransform(localRect), EdgeFlags: edgeFlags,
	}

	// If we are using dmsaa then attempt to draw the rect with FillRRectOp. (If aa is AANo when using dmsaa, the rect
	// is axis aligned. Don't use FillRRectOp because it might require dual source blending.)
	if (sdc.Caps().ReducedShaderMode() || sdc.alwaysAntialias()) &&
		sdc.Caps().DrawInstancedSupport && aa == gpu.AAYes {
		opt := sdc.attemptQuadOptimization(clip, nil /* stencil */, &quad, paint)
		if opt < quadOptClipApplied {
			// The optimization was completely handled inside attempt().
			return
		}

		var croppedRect, croppedLocal geom.Rect
		optimizedClip := clip
		devRect, devIsRect := quad.Device.AsRect()
		localIsRect := false
		if clip != nil && viewMatrix.IsScaleTranslate() && devIsRect && paint.UsesLocalCoords() {
			// The local-rect check only runs (and only writes croppedLocal) when the preceding conditions hold and the
			// paint uses local coordinates.
			croppedLocal, localIsRect = quad.Local.AsRect()
			if !localIsRect {
				croppedLocal = geom.Rect{}
			}
		}
		if clip != nil && viewMatrix.IsScaleTranslate() && devIsRect &&
			(!paint.UsesLocalCoords() || localIsRect) {
			// The cropped quad is still a rect, and our view matrix preserves rects. Map it back to pre-matrix space.
			inverse, ok := viewMatrix.Invert()
			if !ok {
				return
			}
			// The inverse of a scale-translate matrix always keeps rects as rects.
			croppedRect, _ = inverse.MapRect(devRect)
			if opt == quadOptClipApplied {
				optimizedClip = nil
			}
		} else {
			// Even if attemptQuadOptimization gave us an optimized quad, FillRRectOp needs a rect in pre-matrix space,
			// so use the original rect. Also preserve the original clip.
			croppedRect = rectToDraw
			croppedLocal = localRect
		}

		if op := NewFillRRectOp(sdc.Caps(), paint, viewMatrix, geom.MakeRRect(croppedRect, 0, 0),
			LocalCoordsFromRect(croppedLocal), gpu.AAYes); op != nil {
			sdc.AddDrawOp(optimizedClip, op)
			return
		}
	}

	sdc.DrawFilledQuad(clip, paint, &quad, nil)
}

// FillRectWithEdgeAA fills rect with per-edge control over anti-aliasing. localRect may be nil to reuse rect.
func (sdc *SurfaceDrawContext) FillRectWithEdgeAA(clip Clip, paint *Paint, edgeAA gpu.QuadAAFlags, viewMatrix *geom.Matrix, rect geom.Rect, localRect *geom.Rect) {
	local := rect
	if localRect != nil {
		local = *localRect
	}
	if edgeAA == gpu.QuadAAFlagsAll {
		sdc.FillRectToRect(clip, paint, gpu.AAYes, viewMatrix, rect, local)
		return
	}
	quad := DrawQuad{
		Device: MakeQuadFromRect(rect, viewMatrix),
		Local:  MakeQuadFromRectNoTransform(local), EdgeFlags: edgeAA,
	}
	sdc.DrawFilledQuad(clip, paint, &quad, nil)
}

// FillPixelsWithLocalMatrix fills a block of pixels with a paint and a local matrix, respecting the clip.
func (sdc *SurfaceDrawContext) FillPixelsWithLocalMatrix(clip Clip, paint *Paint, bounds geom.IRect, localMatrix *geom.Matrix) {
	rect := bounds.ToRect()
	identity := geom.IdentityMatrix()
	quad := DrawQuad{
		Device: MakeQuadFromRect(rect, &identity),
		Local:  MakeQuadFromRect(rect, localMatrix), EdgeFlags: gpu.QuadAAFlagsNone,
	}
	sdc.DrawFilledQuad(clip, paint, &quad, nil)
}

// DrawPaint fills the render target with the paint.
func (sdc *SurfaceDrawContext) DrawPaint(clip Clip, paint *Paint, viewMatrix *geom.Matrix) {
	// Start with the render target, since that is the maximum content we could possibly fill. DrawFilledQuad() will
	// automatically restrict it to clip bounds for us if possible.
	if paint.NumTotalFragmentProcessors() == 0 {
		// The paint is trivial so we won't need to use local coordinates, so skip calculating the inverse view matrix.
		r := sdc.AsSurfaceProxy().BoundsIRect().ToRect()
		identity := geom.IdentityMatrix()
		sdc.FillRectToRect(clip, paint, gpu.AANo, &identity, r, r)
	} else {
		// Use the inverse view matrix to arrive at appropriate local coordinates for the paint.
		if localMatrix, ok := viewMatrix.Invert(); ok {
			bounds := geom.IRectSize(sdc.AsSurfaceProxy().Dimensions())
			sdc.FillPixelsWithLocalMatrix(clip, paint, bounds, &localMatrix)
		}
	}
}

// DrawQuadSet draws a batch of quads sharing one paint and view matrix.
func (sdc *SurfaceDrawContext) DrawQuadSet(clip Clip, paint *Paint, viewMatrix *geom.Matrix, quads []QuadSetEntry) {
	aaType := sdc.chooseAAType(gpu.AAYes)
	AddFillRectOps(sdc, clip, paint, aaType, viewMatrix, quads, nil)
}

// FillRectWithPaint is a convenience for the shader-pipeline swatch tests: draws the local-space rect transformed by
// viewMatrix with the paint (no clip, no AA). The paint is consumed.
func (sdc *SurfaceDrawContext) FillRectWithPaint(paint *Paint, viewMatrix *geom.Matrix, rect geom.Rect) {
	sdc.FillRectToRect(nil, paint, gpu.AANo, viewMatrix, rect, rect)
}

// setNeedsStencil marks the render target as requiring a stencil buffer. Don't clear stencil until after needsStencil
// is set, so we don't loop forever if there are driver bugs and we need to clear as a draw.
func (sdc *SurfaceDrawContext) setNeedsStencil() {
	hasInitializedStencil := sdc.needsStencil
	sdc.needsStencil = true
	if !hasInitializedStencil {
		sdc.AsRenderTargetProxy().SetNeedsStencil()
		if sdc.Caps().PerformStencilClearsAsDraws {
			// There is a driver bug with clearing stencil: use an op to manually clear the stencil buffer before the op
			// that required setNeedsStencil.
			sdc.internalStencilClear(nil, false /* inside mask */)
		} else {
			sdc.GetOpsTask().SetInitialStencilContent(StencilContentUserBitsCleared)
		}
	}
}

// StencilRect draws a rect carrying user stencil settings. While this can take a general clip, ClipStack relies on this
// function, so callers must take care to only provide hard clips or we could get stuck in a loop. Since this provides
// stencil settings to DrawFilledQuad, it performs a different AA type resolution compared to regular rect draws, which
// is the main reason it remains separate. localMatrix may be nil.
func (sdc *SurfaceDrawContext) StencilRect(clip Clip, ss *UserStencilSettings, paint *Paint, doStencilMSAA gpu.AA, viewMatrix *geom.Matrix, rect geom.Rect, localMatrix *geom.Matrix) {
	edgeFlags := gpu.QuadAAFlagsNone
	if doStencilMSAA == gpu.AAYes {
		edgeFlags = gpu.QuadAAFlagsAll
	}
	var local Quad
	if localMatrix != nil {
		local = MakeQuadFromRect(rect, localMatrix)
	} else {
		local = MakeQuadFromRectNoTransform(rect)
	}
	quad := DrawQuad{
		Device: MakeQuadFromRect(rect, viewMatrix), Local: local,
		EdgeFlags: edgeFlags,
	}
	sdc.DrawFilledQuad(clip, paint, &quad, ss)
}

// ClearStencilClip clears the stencil-clip bit within scissor.
func (sdc *SurfaceDrawContext) ClearStencilClip(scissor geom.IRect, insideStencilMask bool) {
	sdc.internalStencilClear(&scissor, insideStencilMask)
}

// internalStencilClear clears the stencil buffer (or stencil-clip bit) within scissor, via a draw when the driver
// requires it.
func (sdc *SurfaceDrawContext) internalStencilClear(scissor *geom.IRect, insideStencilMask bool) {
	sdc.setNeedsStencil()

	scissorState := gpu.MakeScissorState(sdc.AsSurfaceProxy().BackingStoreDimensions())
	if scissor != nil && !scissorState.Set(*scissor) {
		// The requested clear region is off screen, so nothing to do.
		return
	}

	clearWithDraw := sdc.Caps().PerformStencilClearsAsDraws ||
		(scissorState.Enabled() && sdc.Caps().PerformPartialClearsAsDraws())
	if clearWithDraw {
		ss := SetClipBitSettings(insideStencilMask)
		// Configure the paint to have no impact on the color buffer.
		paint := NewPaint()
		paint.SetXPFactory(DisableColorXPFactory())
		identity := geom.IdentityMatrix()
		sdc.AddDrawOp(nil, MakeNonAARectFillOp(paint, &identity,
			scissorState.Rect().ToRect(), ss))
	} else {
		sdc.addOp(MakeStencilClipClearOp(&scissorState, insideStencilMask))
	}
}

// SetLastClip records the stencil-clip state rendered into the current ops task so unchanged clips can skip
// re-rendering.
func (sdc *SurfaceDrawContext) SetLastClip(clipStackGenID uint32, devClipBounds geom.IRect, numClipAnalyticElements int) {
	opsTask := sdc.GetOpsTask()
	opsTask.lastClipStackGenID = clipStackGenID
	opsTask.lastDevClipBounds = devClipBounds
	opsTask.lastClipNumAnalyticElements = numClipAnalyticElements
}

// MustRenderClip reports whether the clip must be re-rendered into the stencil buffer.
func (sdc *SurfaceDrawContext) MustRenderClip(clipStackGenID uint32, devClipBounds geom.IRect, numClipAnalyticElements int) bool {
	opsTask := sdc.GetOpsTask()
	return opsTask.lastClipStackGenID != clipStackGenID ||
		!opsTask.lastDevClipBounds.ContainsRect(devClipBounds) ||
		opsTask.lastClipNumAnalyticElements != numClipAnalyticElements
}

// opBounds returns the op's bounds, outset for hairlines since we don't know which way the GPU will snap lines or
// points at integer coordinates.
func opBounds(op Op) geom.Rect {
	bounds := op.opBase().Bounds()
	if op.opBase().HasZeroArea() {
		if op.opBase().HasAABloat() {
			bounds = bounds.Outset(0.5, 0.5)
		} else {
			before := bounds
			bounds = bounds.RoundOutRect()
			if bounds.Left == before.Left {
				bounds.Left--
			}
			if bounds.Top == before.Top {
				bounds.Top--
			}
			if bounds.Right == before.Right {
				bounds.Right++
			}
			if bounds.Bottom == before.Bottom {
				bounds.Bottom++
			}
		}
	}
	return bounds
}

// AddDrawOp runs the full draw pipeline for op: clip application, processor finalization, dst-copy setup, and OpsTask
// recording, including the DMSAA per-op MSAA escalation (a willAddFn-style pre-add callback has no callers until the
// clip-atlas work).
func (sdc *SurfaceDrawContext) AddDrawOp(clip Clip, op DrawOp) {
	if sdc.ctx.Abandoned() {
		return
	}

	// bounds and appliedClip are passed by address into the Clip.Apply and DrawOp.Finalize calls below, which would
	// force both to the heap on every draw; borrow them from the pool instead to keep the steady-state frame
	// allocation-free (see drawopscratch.go). The pool (not a context field) is required because clip application
	// re-enters AddDrawOp on this same context for stencil/SW-mask clips.
	scratch := borrowDrawOpScratch()
	defer recycleDrawOpScratch(scratch)
	bounds := &scratch.bounds
	appliedClip := &scratch.appliedClip

	// Set up the clip.
	*bounds = opBounds(op)
	*appliedClip = MakeAppliedClipWithLogicalDims(sdc.Dimensions(),
		sdc.AsSurfaceProxy().BackingStoreDimensions())
	opUsesMSAA := op.UsesMSAA()
	skipDraw := false
	if clip != nil {
		// Have a complex clip, so defer to its early clip culling.
		var aaType gpu.AAType
		switch {
		case opUsesMSAA:
			aaType = gpu.AATypeMSAA
		case op.opBase().HasAABloat():
			aaType = gpu.AATypeCoverage
		default:
			aaType = gpu.AATypeNone
		}
		skipDraw = clip.Apply(sdc, op, aaType, appliedClip, bounds) == ClipEffectClippedOut
	} else {
		// No clipping, so just clip the bounds against the logical render target dimensions.
		skipDraw = !bounds.Intersect(sdc.AsSurfaceProxy().BoundsIRect().ToRect())
	}

	if skipDraw {
		return
	}

	// All color types in the matrix are normalized, so the clamp type is always auto.
	analysis := op.Finalize(&sdc.Caps().Caps, appliedClip, gpu.ClampTypeAuto)

	opUsesStencil := op.UsesStencil()

	// Always trigger DMSAA when there is stencil. This ensures stencil contents get properly preserved between render
	// passes, if needed.
	drawNeedsMSAA := opUsesMSAA || (sdc.canUseDynamicMSAA && opUsesStencil)

	// Must be called before setupDstProxyView so that it sees the final bounds of the op.
	op.opBase().SetClippedBounds(*bounds)

	// Determine if the op will trigger the use of a separate DMSAA attachment that requires manual resolves. The
	// texture-proxy check here is really only relevant for GL, which uses normal texture sampling when using barriers.
	usesAttachmentIfDMSAA := sdc.canUseDynamicMSAA &&
		(!sdc.Caps().MSAAResolvesAutomatically || sdc.AsTextureProxy() == nil)
	opRequiresDMSAAAttachment := usesAttachmentIfDMSAA && drawNeedsMSAA
	opTriggersDMSAAAttachment := opRequiresDMSAAAttachment &&
		!sdc.GetOpsTask().UsesMSAASurface()
	if opTriggersDMSAAAttachment {
		// This will be the op that actually triggers use of a DMSAA attachment. Texture barriers can't be moved to a
		// DMSAA attachment, so if there already are any on the current opsTask then we need to split.
		if sdc.GetOpsTask().RenderPassXferBarriers()&gpu.XferBarrierFlagTexture != 0 {
			if sdc.GetOpsTask().isColorNoOp() {
				panic("a task with texture barriers cannot be a color no-op")
			}
			sdc.replaceOpsTask().SetCannotMergeBackward()
		}
	}

	var dstProxyView DstProxyView
	if analysis.RequiresDstTexture() {
		if !sdc.setupDstProxyView(op.opBase().Bounds(), drawNeedsMSAA, &dstProxyView) {
			return
		}
	}

	// Note if the op needs stencil. Stencil clipping already called setNeedsStencil for itself, if needed.
	if opUsesStencil {
		sdc.setNeedsStencil()
	}

	sdc.GetOpsTask().AddDrawOp(sdc.drawingManager(), op, drawNeedsMSAA, analysis, *appliedClip,
		&dstProxyView, sdc.Caps())
}
