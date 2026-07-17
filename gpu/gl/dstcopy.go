// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The dst-read plumbing: SurfaceDrawContext.setupDstProxyView, the caps queries it consults, and Gpu.xferBarrier. When
// an XP must read the dst and the driver has texture barriers (GL 4.5 / ARB_texture_barrier) the render target samples
// itself; otherwise — notably macOS GL 4.1 — the op's bounds are copied out of the dst first and the shader reads the
// copy, except that under DMSAA the single-sampled RT texture itself can be sampled between render passes (the DMSAA
// lanes). Trim: no Vulkan secondary command buffers.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// GetDstSampleFlagsForProxy reports how a draw may sample its destination: a texture barrier lets a textureable render
// target be sampled as its own dst.
func (c *Caps) GetDstSampleFlagsForProxy(rt *RenderTargetProxy, drawUsesMSAA bool) gpu.DstSampleFlags {
	if c.TextureBarrierSupport && (!drawUsesMSAA || c.MSAAResolvesAutomatically) {
		if rt.Proxy().AsTextureProxy() != nil {
			return gpu.DstSampleFlagRequiresTextureBarrier
		}
	}
	return 0
}

// DstCopyRestrictions describes constraints on copying a destination surface for a dst read.
type DstCopyRestrictions struct {
	RectsMustMatch   bool
	MustCopyWholeSrc bool
}

// GetDstCopyRestrictions reports the restrictions on copying src for a dst read. Returns ok=false when the copy cannot
// be done at all (the caller would need a draw, which never happens on the desktop trim's format table).
func (c *Caps) GetDstCopyRestrictions(src *RenderTargetProxy, colorType gpu.ColorType) (DstCopyRestrictions, bool) {
	srcProxy := src.Proxy()
	// If the src is a texture, the blit can be a draw assuming the config is renderable.
	if srcProxy.AsTextureProxy() != nil &&
		!c.IsFormatAsColorTypeRenderable(colorType, srcProxy.Format(), 1) {
		return DstCopyRestrictions{}, false
	}
	// External textures are outside the desktop trim.

	// Look for opportunities to use CopyTexSubImage or FBO blit, preferring CopyTexSubImage.
	var blitFramebufferRestrictions DstCopyRestrictions
	if src.NumSamples() > 1 && c.BlitFramebufferFlags()&ResolveMustBeFullBlitFramebufferFlag != 0 {
		blitFramebufferRestrictions.RectsMustMatch = true
		blitFramebufferRestrictions.MustCopyWholeSrc = true
	} else if src.NumSamples() > 1 &&
		c.BlitFramebufferFlags()&RectsMustMatchForMSAASrcBlitFramebufferFlag != 0 {
		blitFramebufferRestrictions.RectsMustMatch = true
	}

	if srcProxy.Format() == FormatBGRA8 {
		// glCopyTexSubImage2D doesn't work with BGRA8; set up for FBO blit or fail.
		if c.CanFormatBeFBOColorAttachment(srcProxy.Format()) {
			return blitFramebufferRestrictions, true
		}
		return DstCopyRestrictions{}, false
	}

	if src.NumSamples() > 1 && c.UsesMSAARenderBuffers() {
		// It's illegal to call CopyTexSubImage2D on an MSAA renderbuffer; FBO blit or fail.
		if c.CanFormatBeFBOColorAttachment(srcProxy.Format()) {
			return blitFramebufferRestrictions, true
		}
		return DstCopyRestrictions{}, false
	}

	// CopyTexSubImage, no restrictions.
	return DstCopyRestrictions{}, true
}

// setupDstProxyView resolves how this draw's XP will read the destination, including the DMSAA lanes.
func (sdc *SurfaceDrawContext) setupDstProxyView(opBounds geom.Rect, opRequiresMSAA bool, dstProxyView *DstProxyView) bool {
	caps := sdc.Caps()

	// First get the dstSampleFlags as if the draw joins the current OpsTask.
	dstSampleFlags := caps.GetDstSampleFlagsForProxy(sdc.AsRenderTargetProxy(),
		sdc.GetOpsTask().UsesMSAASurface() || opRequiresMSAA)

	// If we don't have barriers for this draw then we will definitely be breaking up the OpsTask. However, if using
	// dynamic MSAA, the new OpsTask will not have MSAA already enabled on it and that may allow us to use texture
	// barriers. So we check if we can use barriers on the new ops task and then break it up if so.
	if dstSampleFlags&gpu.DstSampleFlagRequiresTextureBarrier == 0 &&
		sdc.canUseDynamicMSAA && sdc.GetOpsTask().UsesMSAASurface() && !opRequiresMSAA {
		newFlags := caps.GetDstSampleFlagsForProxy(sdc.AsRenderTargetProxy(),
			false /* opRequiresMSAA */)
		if newFlags&gpu.DstSampleFlagRequiresTextureBarrier != 0 {
			// We can't have an empty ops task if we are in DMSAA and the ops task already returns true for
			// usesMSAASurface.
			if sdc.GetOpsTask().isColorNoOp() {
				panic("a DMSAA ops task using the MSAA surface cannot be a color no-op")
			}
			sdc.replaceOpsTask().SetCannotMergeBackward()
			dstSampleFlags = newFlags
		}
	}

	if dstSampleFlags&gpu.DstSampleFlagRequiresTextureBarrier != 0 {
		// A barrier means the RT itself is sampled as a texture (or input attachment); the OpsTask does not need to be
		// broken up.
		dstProxyView.SetProxyView(sdc.ReadSurfaceView())
		dstProxyView.SetOffset(geom.IPoint{})
		dstProxyView.SetDstSampleFlags(dstSampleFlags)
		dstProxyView.Proxy().Ref() // the DstProxyView's view is a long-lived owner (see below)
		return true
	}
	if dstSampleFlags != 0 {
		panic("unexpected dst sample flags")
	}

	// We are using a different surface from the main color attachment to sample the dst from. If we are in DMSAA we can
	// just use the single sampled RT texture itself (the resolve at the end of the current render pass publishes it).
	// Otherwise, we must do a copy. This also falls through to the copy when there are no texture barriers but MSAA
	// resolves automatically (render-to-msaa-texture) — never true on desktop GL.
	if sdc.canUseDynamicMSAA && opRequiresMSAA && sdc.AsTextureProxy() != nil &&
		!caps.MSAAResolvesAutomatically &&
		caps.DmsaaResolveCanBeUsedAsTextureInSameRenderPass() {
		sdc.replaceOpsTaskIfModifiesColor().SetCannotMergeBackward()
		dstProxyView.SetProxyView(sdc.ReadSurfaceView())
		dstProxyView.SetOffset(geom.IPoint{})
		dstProxyView.SetDstSampleFlags(dstSampleFlags)
		dstProxyView.Proxy().Ref() // the DstProxyView's view is a long-lived owner (see below)
		return true
	}

	// Fall back to a copy.
	restrictions, ok := caps.GetDstCopyRestrictions(sdc.AsRenderTargetProxy(), sdc.ColorType())
	if !ok {
		return false
	}
	if restrictions.RectsMustMatch {
		// Only reachable via MSAA-source blitFramebuffer driver flags, none of which the desktop trim's caps tables
		// set.
		panic("rects-must-match dst copies are unreachable on the desktop trim")
	}

	copyRect := geom.IRectSize(sdc.AsSurfaceProxy().BackingStoreDimensions())
	if !restrictions.MustCopyWholeSrc {
		// Restrict to the op's bounds plus a pixel of padding for AA bloat and unpredictable rounding near pixel
		// centers.
		conservativeDrawBounds := opBounds.RoundOut()
		conservativeDrawBounds.Left--
		conservativeDrawBounds.Top--
		conservativeDrawBounds.Right++
		conservativeDrawBounds.Bottom++
		if !copyRect.Intersect(conservativeDrawBounds) {
			return false
		}
	}

	dstOffset := geom.IPoint{X: copyRect.Left, Y: copyRect.Top}
	copyProxy, copyTask := CopySurfaceProxy(sdc.Context(), sdc.AsSurfaceProxy(), sdc.Origin(),
		gpu.MipmappedNo, copyRect, gpu.BackingFitApprox, gpu.BudgetedYes, "SetupDstProxyView")
	if copyProxy == nil {
		return false
	}
	_ = copyTask
	readView := sdc.ReadSurfaceView()
	// The copy proxy arrives with one ref owned by the caller; the DstProxyView keeps it (the op releases it after
	// execution via releaseDstProxy).
	dstProxyView.SetProxyView(MakeSurfaceProxyView(copyProxy, sdc.Origin(), readView.Swizzle()))
	dstProxyView.SetOffset(dstOffset)
	dstProxyView.SetDstSampleFlags(dstSampleFlags)
	return true
}

// releaseDstProxy drops the ref setupDstProxyView took on the dst proxy.
func releaseDstProxy(dstProxyView *DstProxyView) {
	if p := dstProxyView.Proxy(); p != nil {
		p.Unref()
	}
}

// xferBarrier issues the GPU barrier a transfer function needs before it can safely sample the destination.
func (g *Gpu) xferBarrier(rt *RenderTarget, barrierType XferBarrierType) {
	switch barrierType {
	case XferBarrierTexture:
		if rt.Surface().AsTexture() == nil {
			panic("texture barrier on a non-texture render target")
		}
		if rt.Surface().RequiresManualMSAAResolve() {
			g.ResolveRenderTarget(rt, geom.IRectSize(rt.Surface().Dimensions()))
		}
		g.fns().TextureBarrier()
	case XferBarrierBlend:
		g.fns().BlendBarrier()
	case XferBarrierNone:
	}
}
