// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The render pass is the series of commands (draws, clears, discards) targeting one render target; on GL they execute
// immediately: the begin/end lifecycle with its load/store-op handling (including the DMSAA attachment load/store
// resolves) and the clear entry points; the draw surface (bindPipeline/bindBuffers/draw*) lives with the shader
// pipeline and the draw ops. This desktop-only build drops the ES tiled-rendering (QCOM_tiled_rendering) lane, and the
// load-resolve fallback for when glBlitFramebuffer cannot target an MSAA destination is unreachable — desktop GL always
// supports single-to-MSAA resolves, so CanResolveSingleToMSAA is constant true.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// LoadAndStoreInfo describes how a render pass's color attachment should be loaded at the start of the pass and stored
// at the end.
type LoadAndStoreInfo struct {
	LoadOp     gpu.LoadOp
	StoreOp    gpu.StoreOp
	ClearColor [4]float32
}

// StencilLoadAndStoreInfo describes how a render pass's stencil attachment should be loaded at the start of the pass
// and stored at the end. Load-time clears of the stencil buffer are always to 0, so there is no stencil clear value.
type StencilLoadAndStoreInfo struct {
	LoadOp  gpu.LoadOp
	StoreOp gpu.StoreOp
}

// drawPipelineStatus tracks whether the render pass has a valid pipeline bound for drawing.
type drawPipelineStatus int8

const (
	drawPipelineNotConfigured drawPipelineStatus = iota
	drawPipelineOK
	drawPipelineFailedToBind
)

// OpsRenderPass represents one render pass: a series of commands (draws, clears, discards) targeting a single render
// target.
type OpsRenderPass struct {
	activeInstanceBuffer AnyBuffer
	activeIndexBuffer    AnyBuffer
	activeVertexBuffer   AnyBuffer
	attribArrayState     *AttribArrayState
	renderTarget         *RenderTarget
	gpu                  *Gpu
	colorInfo            LoadAndStoreInfo
	contentBounds        geom.IRect
	stencilInfo          StencilLoadAndStoreInfo
	origin               gpu.SurfaceOrigin
	useMultisampleFBO    bool
	primitiveRestart     bool
	drawPipelineStatus   drawPipelineStatus
	// Draw state.
	primitiveType gpu.PrimitiveType
}

// set configures a reset render pass for the given target, bounds, origin, and load/store info.
func (p *OpsRenderPass) set(rt *RenderTarget, useMSAASurface bool, contentBounds geom.IRect, origin gpu.SurfaceOrigin, colorInfo LoadAndStoreInfo, stencilInfo StencilLoadAndStoreInfo) {
	if p.renderTarget != nil {
		panic("render pass already set")
	}
	p.renderTarget = rt
	p.origin = origin
	p.useMultisampleFBO = useMSAASurface
	p.contentBounds = contentBounds
	p.colorInfo = colorInfo
	p.stencilInfo = stencilInfo
}

// reset returns the cached render pass to the unset state for reuse.
func (p *OpsRenderPass) reset() {
	p.renderTarget = nil
	p.drawPipelineStatus = drawPipelineNotConfigured
	p.primitiveRestart = false
	p.attribArrayState = nil
	p.activeVertexBuffer = nil
	p.activeIndexBuffer = nil
	p.activeInstanceBuffer = nil
}

// RenderTarget returns the pass's target.
func (p *OpsRenderPass) RenderTarget() *RenderTarget { return p.renderTarget }

// dmsaaLoadStoreBounds returns the region the DMSAA load/store resolves blit, in the surface's native coordinates.
func (p *OpsRenderPass) dmsaaLoadStoreBounds() geom.IRect {
	surfDims := p.renderTarget.Surface().Dimensions()
	if p.gpu.glCaps().FramebufferResolvesMustBeFullSize() {
		// If framebuffer resolves have to be full size, then resolve the entire render target during load and store
		// both, even if we will be doing so with a draw. We have no other choice than to do a full size resolve at the
		// end of the render pass, so the full DMSAA attachment needs to have valid content.
		return makeNativeIRect(p.origin, surfDims.Height, geom.IRectSize(surfDims))
	}
	return makeNativeIRect(p.origin, surfDims.Height, p.contentBounds)
}

// Begin loads the single sample FBO into the DMSAA attachment when the render pass loads its color, then binds the
// target and performs the load-op clears.
func (p *OpsRenderPass) Begin() {
	if p.useMultisampleFBO && p.colorInfo.LoadOp == gpu.LoadOpLoad &&
		p.renderTarget.HasDynamicMSAAAttachment() {
		// Load the single sample fbo into the dmsaa attachment. (The resolve fallback for when this isn't directly
		// supported is unreachable on desktop GL — see the file comment.)
		if !p.gpu.glCaps().CanResolveSingleToMSAA() {
			panic("single-to-MSAA resolves are always supported on desktop GL")
		}
		p.gpu.ResolveRenderFBOs(p.renderTarget, p.dmsaaLoadStoreBounds(), ResolveSingleToMSAA,
			false)
	}
	p.gpu.beginCommandBuffer(p.renderTarget, p.useMultisampleFBO, p.colorInfo, p.stencilInfo)
}

// End finishes the render pass; after the pass, resolves the DMSAA attachment back into the single sample FBO when the
// color is stored.
func (p *OpsRenderPass) End() {
	p.gpu.endCommandBuffer(p.renderTarget, p.useMultisampleFBO, p.colorInfo, p.stencilInfo)

	if p.useMultisampleFBO && p.colorInfo.StoreOp == gpu.StoreOpStore &&
		p.renderTarget.HasDynamicMSAAAttachment() {
		// Blit the msaa attachment into the single sample fbo.
		p.gpu.ResolveRenderFBOs(p.renderTarget, p.dmsaaLoadStoreBounds(), ResolveMSAAToSingle,
			true /* invalidateReadBufferAfterBlit */)
	}
}

// Clear clears the full target when scissor is disabled, else the scissor rect. The caller must have checked
// performColorClearsAsDraws / performPartialClearsAsDraws.
func (p *OpsRenderPass) Clear(scissor *gpu.ScissorState, color [4]float32) {
	if p.renderTarget == nil {
		panic("clear on unset render pass")
	}
	p.gpu.Clear(scissor, color, p.renderTarget, p.useMultisampleFBO, p.origin)
}

// InlineUpload runs a deferred texture upload mid-pass; GL is immediate-mode, so this simply performs the upload
// immediately.
func (p *OpsRenderPass) InlineUpload(state *OpFlushState, upload DeferredTextureUploadFn) {
	state.DoUpload(upload, false)
}

// ClearStencilClip clears the render target's stencil buffer, setting it either entirely inside or entirely outside the
// clip mask.
func (p *OpsRenderPass) ClearStencilClip(scissor *gpu.ScissorState, insideStencilMask bool) {
	if p.renderTarget == nil {
		panic("clearStencilClip on unset render pass")
	}
	p.gpu.ClearStencilClip(scissor, insideStencilMask, p.renderTarget, p.useMultisampleFBO,
		p.origin)
}

// GetOpsRenderPass returns the (single, reused) render pass object configured for the target, promoting a single-sample
// target to its dynamic MSAA attachment when the pass requests MSAA (DMSAA). The resource provider creates that
// attachment. The stencil attachment and sampled proxies are consumed by later phases; the GL pass doesn't key on them
// yet.
func (g *Gpu) GetOpsRenderPass(rp *ResourceProvider, rt *RenderTarget, useMSAASurface bool, stencil *Attachment, origin gpu.SurfaceOrigin, contentBounds geom.IRect, colorInfo LoadAndStoreInfo, stencilInfo StencilLoadAndStoreInfo, sampledProxies []*SurfaceProxy, renderPassXferBarriers gpu.XferBarrierFlags) *OpsRenderPass {
	_ = stencil
	_ = sampledProxies
	_ = renderPassXferBarriers
	if useMSAASurface && rt.NumSamples() == 1 {
		// We will be using dynamic msaa. Ensure there is an attachment.
		if !rt.EnsureDynamicMSAAAttachment(rp) {
			// The attachment could not be created; drop the render pass.
			return nil
		}
	}
	if g.cachedOpsRenderPass == nil {
		g.cachedOpsRenderPass = &OpsRenderPass{gpu: g}
	}
	g.cachedOpsRenderPass.reset()
	g.cachedOpsRenderPass.set(rt, useMSAASurface, contentBounds, origin, colorInfo, stencilInfo)
	return g.cachedOpsRenderPass
}

// Submit retires pass; on GL the commands already executed immediately, so this just resets it for reuse.
func (g *Gpu) Submit(pass *OpsRenderPass) {
	pass.reset()
}

// beginCommandBuffer binds the target FBO and applies the load-op clears. (This desktop-only build has no ES
// tiled-rendering lane to skip.)
func (g *Gpu) beginCommandBuffer(rt *RenderTarget, useMultisampleFBO bool, colorLoadStore LoadAndStoreInfo, stencilLoadStore StencilLoadAndStoreInfo) {
	g.handleDirtyContext()
	g.flushRenderTarget(rt, useMultisampleFBO)

	var clearMask uint32
	if colorLoadStore.LoadOp == gpu.LoadOpClear {
		if g.Caps().PerformColorClearsAsDraws {
			panic("load-op clear with performColorClearsAsDraws")
		}
		g.flushClearColor(colorLoadStore.ClearColor)
		g.flushColorWrite(true)
		clearMask |= COLOR_BUFFER_BIT
	}
	if stencilLoadStore.LoadOp == gpu.LoadOpClear {
		if g.Caps().PerformStencilClearsAsDraws {
			panic("stencil load-op clear with performStencilClearsAsDraws")
		}
		g.fns().StencilMask(0xffffffff)
		g.fns().ClearStencil(0)
		clearMask |= STENCIL_BUFFER_BIT
	}
	if clearMask != 0 {
		g.flushScissorTest(false)
		g.disableWindowRectangles()
		g.fns().Clear(clearMask)
		if clearMask&COLOR_BUFFER_BIT != 0 {
			g.didWriteToSurface(rt.Surface())
		}
	}
}

// endCommandBuffer finishes a command buffer: store-op discards become framebuffer invalidations where supported.
func (g *Gpu) endCommandBuffer(rt *RenderTarget, useMultisampleFBO bool, colorLoadStore LoadAndStoreInfo, stencilLoadStore StencilLoadAndStoreInfo) {
	g.handleDirtyContext()

	if rt.Surface().UniqueID() != g.hwBoundRenderTargetUniqueID ||
		useMultisampleFBO != g.hwBoundFramebufferIsMSAA {
		// The framebuffer binding changed in the middle of a command buffer.
		return
	}

	if g.glCaps().InvalidateFBType() != InvalidateFBNone {
		var discardAttachments []uint32
		if colorLoadStore.StoreOp == gpu.StoreOpDiscard {
			if rt.IsFBO0(useMultisampleFBO) {
				discardAttachments = append(discardAttachments, COLOR)
			} else {
				discardAttachments = append(discardAttachments, COLOR_ATTACHMENT0)
			}
		}
		if stencilLoadStore.StoreOp == gpu.StoreOpDiscard {
			if rt.IsFBO0(useMultisampleFBO) {
				discardAttachments = append(discardAttachments, STENCIL)
			} else {
				discardAttachments = append(discardAttachments, STENCIL_ATTACHMENT)
			}
		}
		if len(discardAttachments) > 0 {
			if g.glCaps().InvalidateFBType() == InvalidateFBInvalidate {
				g.fns().InvalidateFramebuffer(FRAMEBUFFER, int32(len(discardAttachments)),
					&discardAttachments[0])
			} else {
				// The DiscardFramebuffer lane is EXT_discard_framebuffer (ES-only); the desktop trim never selects it.
				panic("discard-framebuffer lane is unreachable on desktop GL")
			}
		}
	}
}

// ClearStencilClip clears the render target's stencil buffer. Our contract on OpsTask says that changing the clip
// between stencil passes may or may not zero the client's clip bits, so the whole stencil is cleared rather than just
// the clip bit.
func (g *Gpu) ClearStencilClip(scissor *gpu.ScissorState, insideStencilMask bool, target *RenderTarget, useMultisampleFBO bool, origin gpu.SurfaceOrigin) {
	if g.Caps().PerformStencilClearsAsDraws {
		panic("stencil clear should have been converted to a draw")
	}
	if scissor.Enabled() && g.Caps().PerformPartialClearsAsDraws() {
		panic("partial stencil clear should have been converted to a draw")
	}
	g.handleDirtyContext()

	sb := target.StencilAttachment(useMultisampleFBO)
	if sb == nil {
		// We should only get here if we marked a proxy as requiring a stencil buffer, but its creation later failed.
		// Likely clipping is going to go awry now.
		return
	}

	stencilBitCount := sb.GLFormat().StencilBits()
	var value int32
	if insideStencilMask {
		value = 1 << (stencilBitCount - 1)
	}
	g.flushRenderTarget(target, useMultisampleFBO)
	g.flushScissor(scissor, target.Surface().Height(), origin)
	g.disableWindowRectangles()

	g.fns().StencilMask(0xffffffff)
	g.fns().ClearStencil(value)
	g.fns().Clear(STENCIL_BUFFER_BIT)
	g.hwStencilSettings.Invalidate()
	g.hwStencilTestEnabled = triUnknown
}

// ResolveRenderTarget resolves the MSAA color buffer into the single-sample buffer.
func (g *Gpu) ResolveRenderTarget(rt *RenderTarget, resolveRect geom.IRect) {
	if g.glCaps().FramebufferResolvesMustBeFullSize() {
		resolveRect = geom.IRectSize(rt.Surface().Dimensions())
	}
	g.ResolveRenderFBOs(rt, resolveRect, ResolveMSAAToSingle, false)
}
