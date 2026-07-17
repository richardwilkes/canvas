// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// SurfaceFillContext is the render-target-owning context that manages the target's OpsTask across splits and implements
// clears/discards. SurfaceDrawContext builds on it with a full paint/clip draw surface; the draw entry points live in
// surfacedrawcontext.go. This file also carries the SurfaceDrawContext creation paths needed to wrap a caller-owned
// backend render target.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/text"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/surface"
)

// SurfaceFillContext is the render-target-owning context that manages the target's OpsTask across splits and implements
// clears/discards.
type SurfaceFillContext struct {
	// opsTask can be closed by another surface context that picked it up, so it must only be accessed via GetOpsTask.
	// The context holds a task ref.
	opsTask *OpsTask
	// canDiscardOverride is the hook canDiscardPreviousOpsOnFullClear defers to: the SurfaceDrawContext constructor
	// installs its stencil-aware override here. When nil, the default (discard is always allowed) applies.
	canDiscardOverride func() CanDiscardPreviousOps
	// willReplaceOpsTask, when set, is called just before the ops task is replaced (a no-op by default): the
	// SurfaceDrawContext constructor installs the override that preserves stencil values across a task split.
	willReplaceOpsTask func(prevTask, nextTask *OpsTask)
	writeView          SurfaceProxyView
	SurfaceContext
}

// initSurfaceFillContext is the SurfaceFillContext constructor. Adopting the last ops task here allows an in-progress
// OpsTask to be picked up and added to.
func (sfc *SurfaceFillContext) initSurfaceFillContext(ctx *DirectContext, readView, writeView SurfaceProxyView, colorType gpu.ColorType) {
	if readView.Proxy() != writeView.Proxy() || readView.Origin() != writeView.Origin() {
		panic("fill context read/write views must share a proxy and origin")
	}
	sfc.initSurfaceContext(ctx, readView, colorType)
	sfc.writeView = writeView
	if last := ctx.DrawingManager().GetLastOpsTask(sfc.AsSurfaceProxy()); last != nil {
		last.Ref()
		sfc.opsTask = last
	}
}

// Release drops the fill context's task and proxy refs.
func (sfc *SurfaceFillContext) Release() {
	if sfc.opsTask != nil {
		sfc.opsTask.Unref()
		sfc.opsTask = nil
	}
	sfc.SurfaceContext.Release()
}

// WriteSurfaceView returns the view this context writes to.
func (sfc *SurfaceFillContext) WriteSurfaceView() SurfaceProxyView { return sfc.writeView }

// NumSamples returns the render target's MSAA sample count.
func (sfc *SurfaceFillContext) NumSamples() int { return sfc.AsRenderTargetProxy().NumSamples() }

// arenas returns the render target's op/memory arenas.
func (sfc *SurfaceFillContext) arenas() *Arenas {
	return sfc.writeView.Proxy().AsRenderTargetProxy().Arenas()
}

// GetOpsTask returns the current ops task for this context, replacing it with a fresh one first if the current one is
// nil or already closed.
func (sfc *SurfaceFillContext) GetOpsTask() *OpsTask {
	if sfc.opsTask == nil || sfc.opsTask.IsClosed() {
		sfc.replaceOpsTask()
	}
	if sfc.opsTask.IsClosed() {
		panic("ops task still closed after replacement")
	}
	return sfc.opsTask
}

// PeekLastOpsTask returns the current ops task without replacing it, for testing.
func (sfc *SurfaceFillContext) PeekLastOpsTask() *OpsTask { return sfc.opsTask }

// replaceOpsTask installs a fresh ops task for this context, notifying willReplaceOpsTask and releasing the prior
// task's ref first.
func (sfc *SurfaceFillContext) replaceOpsTask() *OpsTask {
	newOpsTask := sfc.ctx.DrawingManager().NewOpsTask(sfc.writeView, sfc.arenas())
	if sfc.willReplaceOpsTask != nil {
		sfc.willReplaceOpsTask(sfc.opsTask, newOpsTask)
	}
	if sfc.opsTask != nil {
		sfc.opsTask.Unref()
	}
	sfc.opsTask = newOpsTask
	return newOpsTask
}

// replaceOpsTaskIfModifiesColor splits the task unless the current one has no color-modifying work.
func (sfc *SurfaceFillContext) replaceOpsTaskIfModifiesColor() *OpsTask {
	if !sfc.GetOpsTask().isColorNoOp() {
		sfc.replaceOpsTask()
	}
	return sfc.GetOpsTask()
}

// RefRenderTask returns the task that will render this context's pending work (the caller owns the returned ref).
func (sfc *SurfaceFillContext) RefRenderTask() *OpsTask {
	task := sfc.GetOpsTask()
	task.Ref()
	return task
}

// Discard is a performance hint that the render target's contents may become undefined.
func (sfc *SurfaceFillContext) Discard() {
	if sfc.ctx.Abandoned() {
		return
	}
	sfc.GetOpsTask().Discard()
}

// ResolveMSAA resolves this context's MSAA render target into its resolve texture.
func (sfc *SurfaceFillContext) ResolveMSAA() {
	if sfc.ctx.Abandoned() {
		return
	}
	sfc.drawingManager().NewTextureResolveRenderTask(sfc.AsSurfaceProxy(), ResolveFlagMSAA,
		sfc.Caps())
}

// ClearRect clears the rect of the render target to the given color, which is in the surface's alpha type (premul for
// all reachable surfaces — the alpha-type adjustment collapses).
func (sfc *SurfaceFillContext) ClearRect(rect geom.IRect, color [4]float32) {
	sfc.internalClear(&rect, color, false)
}

// Clear clears the entire render target to the given color.
func (sfc *SurfaceFillContext) Clear(color [4]float32) {
	sfc.internalClear(nil, color, false)
}

// ClearAtLeast clears at minimum the pixels within scissor, but may clear the full render target if that is more
// performant.
func (sfc *SurfaceFillContext) ClearAtLeast(scissor geom.IRect, color [4]float32) {
	sfc.internalClear(&scissor, color, true)
}

// FillRectWithFP fills dstRect with fp using a src blend, so the FP output replaces the destination.
func (sfc *SurfaceFillContext) FillRectWithFP(dstRect geom.IRect, fp FragmentProcessor) {
	if sfc.ctx.Abandoned() {
		return
	}
	paint := NewPaint()
	paint.SetColorFragmentProcessor(fp)
	paint.SetPorterDuffXPFactory(raster.BlendSrc)
	identity := geom.IdentityMatrix()
	op := MakeNonAARectFillOp(paint, &identity, dstRect.ToRect(), nil)
	sfc.addDrawOp(op)
}

// FillRectWithFPLocalMatrix is like FillRectWithFP but applies a coordinate transformation via a matrix effect first.
func (sfc *SurfaceFillContext) FillRectWithFPLocalMatrix(dstRect geom.IRect, localMatrix geom.Matrix, fp FragmentProcessor) {
	sfc.FillRectWithFP(dstRect, MatrixEffectFP(localMatrix, fp))
}

// FillRectToRectWithFP fills dstRect with fp using a local matrix that maps dstRect onto srcRect (so the FP's sample
// coords land in srcRect's space).
func (sfc *SurfaceFillContext) FillRectToRectWithFP(srcRect geom.Rect, dstRect geom.IRect, fp FragmentProcessor) {
	lm := rectToRectFillMatrix(dstRect.ToRect(), srcRect)
	sfc.FillRectWithFPLocalMatrix(dstRect, lm, fp)
}

// addOp records op on this context's ops task.
func (sfc *SurfaceFillContext) addOp(op Op) {
	drawingMgr := sfc.drawingManager()
	sfc.GetOpsTask().AddOp(drawingMgr, op, sfc.Caps())
}

// canDiscardPreviousOpsOnFullClear reports whether a full clear can discard all prior ops in the task.
// SurfaceDrawContext overrides via canDiscardOverride: when the render target is marked as needing a stencil buffer, a
// prior op may write to the stencil buffer that following draw ops read, so prior ops can't be discarded.
func (sfc *SurfaceFillContext) canDiscardPreviousOpsOnFullClear() CanDiscardPreviousOps {
	if sfc.canDiscardOverride != nil {
		return sfc.canDiscardOverride()
	}
	return CanDiscardPreviousOpsYes
}

// internalClear implements the clearing logic shared by ClearRect, Clear, and ClearAtLeast. There are three ways clears
// are handled: load ops (fullscreen only), native clears (fullscreen or scissored when supported), and draws.
func (sfc *SurfaceFillContext) internalClear(scissor *geom.IRect, color [4]float32, upgradePartialToFull bool) {
	if sfc.ctx.Abandoned() {
		return
	}
	scissorState := gpu.MakeScissorState(sfc.AsSurfaceProxy().BackingStoreDimensions())
	if scissor != nil && !scissorState.Set(*scissor) {
		// The clear is offscreen, so skip it (normally handled by addDrawOp, but clear ops are not draw ops).
		return
	}

	// If we have a scissor but it's okay to clear beyond it for performance reasons, disable the test. Only done when
	// the clear would be handled by a load op or natively.
	if scissorState.Enabled() && !sfc.Caps().PerformColorClearsAsDraws {
		if upgradePartialToFull && (sfc.Caps().PreferFullscreenClears ||
			sfc.Caps().ShouldInitializeTextures) {
			scissorState.SetDisabled()
		} else {
			// Unlike stencil clears, clears may extend to the logical dimensions of the render target, overflowing into
			// any approx-fit padding of the backing store.
			scissorState.RelaxTest(sfc.Dimensions())
		}
	}

	if !scissorState.Enabled() {
		// This is a fullscreen clear, so it could be handled as a load op. Regardless, all prior ops in the current
		// task can be discarded since the color buffer will be overwritten.
		opsTask := sfc.GetOpsTask()
		if opsTask.ResetForFullscreenClear(sfc.canDiscardPreviousOpsOnFullClear()) &&
			!sfc.Caps().PerformColorClearsAsDraws {
			color = sfc.writeView.Swizzle().ApplyTo(color)
			// The op list was emptied and native clears are allowed, so just use the load op.
			opsTask.SetColorLoadOp(gpu.LoadOpClear, color)
			return
		}
		// An op will be used for the clear: reset the load op to discard since the op will blow away the color buffer
		// contents.
		opsTask.SetColorLoadOp(gpu.LoadOpDiscard, [4]float32{})
	}

	// At this point we are either a partial clear or a fullscreen clear that couldn't be applied as a load op.
	clearAsDraw := sfc.Caps().PerformColorClearsAsDraws ||
		(scissorState.Enabled() && sfc.Caps().PerformPartialClearsAsDraws())
	if clearAsDraw {
		paint := NewPaint()
		clearToGrPaint(color, paint)
		identity := geom.IdentityMatrix()
		op := MakeNonAARectFillOp(paint, &identity, scissorState.Rect().ToRect(), nil)
		sfc.addDrawOp(op)
	} else {
		color = sfc.writeView.Swizzle().ApplyTo(color)
		sfc.addOp(MakeColorClearOp(&scissorState, color))
	}
}

// clearToGrPaint configures paint to perform a clear to color.
func clearToGrPaint(color [4]float32, paint *Paint) {
	paint.SetColor4f(colorcore.PMColor4f{R: color[0], G: color[1], B: color[2], A: color[3]})
	if color[3] == 1 {
		// Can just rely on the src-over blend mode to do the right thing; this may improve batching.
		paint.SetPorterDuffXPFactory(raster.BlendSrcOver)
	} else {
		// A clear overwrites the prior color, so even if it's transparent it behaves as if it were src blended.
		paint.SetPorterDuffXPFactory(raster.BlendSrc)
	}
}

// addDrawOp records a simple fill draw op with no clip beyond the surface bounds.
func (sfc *SurfaceFillContext) addDrawOp(op DrawOp) {
	// All color types reachable here are normalized, so the clamp type is always auto.
	clip := MakeAppliedClip(geom.ISize{Width: 1 << 29, Height: 1 << 29}) // an effectively-disabled clip
	analysis := op.Finalize(&sfc.Caps().Caps, &clip, gpu.ClampTypeAuto)
	if op.UsesStencil() {
		panic("fill contexts must not have stencil draws")
	}
	if analysis.RequiresDstTexture() {
		panic("fill contexts must not require dst textures")
	}
	// We shouldn't have coverage AA or hairline draws in fill contexts.
	if op.opBase().HasAABloat() || op.opBase().HasZeroArea() {
		panic("fill contexts must not have coverage-AA or hairline draws")
	}
	bounds := op.opBase().Bounds()
	if !bounds.Intersect(sfc.AsSurfaceProxy().BoundsIRect().ToRect()) {
		return
	}
	op.opBase().SetClippedBounds(op.opBase().Bounds())

	var dstProxyView DstProxyView
	sfc.GetOpsTask().AddDrawOp(sfc.drawingManager(), op, op.UsesMSAA(), analysis, clip,
		&dstProxyView, sfc.Caps())
}

// SurfaceDrawContext is a SurfaceFillContext with a full paint/clip draw surface; the draw entry points live in
// surfacedrawcontext.go.
type SurfaceDrawContext struct {
	// subRunControl caches the device's SDFT (signed-distance-field text) decision, built lazily on the first text draw
	// from the context caps/options and the props' device-independent-fonts flag (sdctext.go).
	subRunControl *text.SubRunControl
	SurfaceFillContext
	surfaceProps surface.Props
	// canUseDynamicMSAA is true when the props request dynamic MSAA AND the caps support DMSAA on this render target.
	canUseDynamicMSAA bool
	needsStencil      bool
}

// SurfaceProps returns the surface properties this context was created with.
func (sdc *SurfaceDrawContext) SurfaceProps() surface.Props { return sdc.surfaceProps }

// CanUseDynamicMSAA reports whether this context can use dynamic MSAA.
func (sdc *SurfaceDrawContext) CanUseDynamicMSAA() bool { return sdc.canUseDynamicMSAA }

// alwaysAntialias reports whether the props request dynamic MSAA (regardless of whether the caps could honor it).
func (sdc *SurfaceDrawContext) alwaysAntialias() bool {
	return sdc.surfaceProps.Flags&surface.DynamicMSAAFlag != 0
}

// MakeSurfaceDrawContext creates a new render-target-backed draw context with default surface props — the form used by
// internal offscreen contexts. MakeSurfaceDrawContextWithProps is the full form. Returns nil if the context is
// abandoned or the format/proxy can't be created.
func MakeSurfaceDrawContext(ctx *DirectContext, colorType gpu.ColorType, dims geom.ISize, fit gpu.BackingFit, sampleCnt int, mipmapped gpu.Mipmapped, origin gpu.SurfaceOrigin, budgeted gpu.Budgeted, label string) *SurfaceDrawContext {
	return MakeSurfaceDrawContextWithProps(ctx, colorType, dims, fit, sampleCnt, mipmapped, origin,
		budgeted, label, nil)
}

// MakeSurfaceDrawContextWithProps creates a new render-target-backed draw context carrying the surface props (nil =
// default). Returns nil if the context is abandoned or the format/proxy can't be created.
func MakeSurfaceDrawContextWithProps(ctx *DirectContext, colorType gpu.ColorType, dims geom.ISize, fit gpu.BackingFit, sampleCnt int, mipmapped gpu.Mipmapped, origin gpu.SurfaceOrigin, budgeted gpu.Budgeted, label string, props *surface.Props) *SurfaceDrawContext {
	if ctx.Abandoned() {
		return nil
	}
	format := ctx.GLCaps().GetDefaultBackendFormat(colorType, true)
	if format == FormatUnknown {
		return nil
	}
	proxy := ctx.ProxyProvider().CreateProxy(format, dims, gpu.RenderableYes, sampleCnt,
		mipmapped, fit, budgeted, label, 0, UseAllocatorYes)
	if proxy == nil {
		return nil
	}
	sdc := makeSDCFromProxy(ctx, proxy, colorType, origin, props)
	// The context holds its own refs now.
	proxy.Unref()
	return sdc
}

// MakeSurfaceDrawContextFromBackendRenderTarget wraps a caller-owned GL render target description (the
// `sk_surface_new_backend_render_target` path: unison's FBO 0 with its stencil bits) into a proxy-backed draw context
// carrying the surface props (nil = default).
func MakeSurfaceDrawContextFromBackendRenderTarget(ctx *DirectContext, colorType gpu.ColorType, dims geom.ISize, format Format, sampleCnt, stencilBits int, fboID uint32, origin gpu.SurfaceOrigin, props *surface.Props) *SurfaceDrawContext {
	if ctx.Abandoned() {
		return nil
	}
	proxy := ctx.ProxyProvider().WrapBackendRenderTarget(dims, format, sampleCnt, stencilBits,
		fboID)
	if proxy == nil {
		return nil
	}
	sdc := makeSDCFromProxy(ctx, proxy, colorType, origin, props)
	proxy.Unref()
	return sdc
}

func makeSDCFromProxy(ctx *DirectContext, proxy *SurfaceProxy, colorType gpu.ColorType, origin gpu.SurfaceOrigin, props *surface.Props) *SurfaceDrawContext {
	readSwizzle := ctx.GLCaps().ReadSwizzle(proxy.Format(), colorType)
	writeSwizzle := ctx.GLCaps().WriteSwizzle(proxy.Format(), colorType)
	sdc := &SurfaceDrawContext{}
	if props != nil {
		sdc.surfaceProps = *props
	}
	sdc.initSurfaceFillContext(ctx, MakeSurfaceProxyView(proxy, origin, readSwizzle),
		MakeSurfaceProxyView(proxy, origin, writeSwizzle), colorType)
	// canUseDynamicMSAA: the props flag AND caps support for dynamic MSAA on this render target.
	sdc.canUseDynamicMSAA = sdc.surfaceProps.Flags&surface.DynamicMSAAFlag != 0 &&
		ctx.GLCaps().SupportsDynamicMSAA(sdc.AsRenderTargetProxy())
	sdc.canDiscardOverride = func() CanDiscardPreviousOps {
		return CanDiscardPreviousOps(!sdc.needsStencil)
	}
	// When this context needed stencil, store the stencil values upon completion of the previous task and reload them
	// at the start of the next.
	sdc.willReplaceOpsTask = func(prevTask, nextTask *OpsTask) {
		if prevTask != nil && sdc.needsStencil {
			prevTask.SetMustPreserveStencil()
			nextTask.SetInitialStencilContent(StencilContentPreserved)
		}
	}
	return sdc
}
