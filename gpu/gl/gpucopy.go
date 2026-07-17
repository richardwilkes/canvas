// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GL surface-copy lanes: copy-as-tex-sub-image, copy-as-blit-framebuffer, and their can_* helpers deciding which
// lane a given copy can use. The copy-as-draw lane is not implemented: copies that only a draw could satisfy (or that
// need scaling with neither lane available) fail.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// rtHasMSAARenderBuffer reports whether rt has a separate MSAA renderbuffer: true when it's multisampled, the driver
// uses an extension with separate MSAA renderbuffers, and it's not FBO 0 (which always auto-resolves).
func rtHasMSAARenderBuffer(rt *RenderTarget, caps *Caps) bool {
	return rt.NumSamples() > 1 && caps.UsesMSAARenderBuffers() && !rt.IsFBO0(true)
}

func surfaceTexTypePtr(s *Surface) *gpu.TextureType {
	if tex := s.AsTexture(); tex != nil {
		t := tex.TextureType()
		return &t
	}
	return nil
}

// canCopyTexSubImage reports whether dst and src are compatible enough to copy with glCopyTexSubImage2D.
func canCopyTexSubImage(dst, src *Surface, caps *Caps) bool {
	dstHasMSAARenderBuffer := false
	if rt := dst.AsRenderTarget(); rt != nil {
		dstHasMSAARenderBuffer = rtHasMSAARenderBuffer(rt, caps)
	}
	srcHasMSAARenderBuffer := false
	if rt := src.AsRenderTarget(); rt != nil {
		srcHasMSAARenderBuffer = rtHasMSAARenderBuffer(rt, caps)
	}
	return caps.CanCopyTexSubImage(dst.Format(), dstHasMSAARenderBuffer, surfaceTexTypePtr(dst),
		src.Format(), srcHasMSAARenderBuffer, surfaceTexTypePtr(src))
}

// canBlitFramebufferForCopySurface reports whether dst and src are compatible enough to copy with glBlitFramebuffer
// over the given rects.
func canBlitFramebufferForCopySurface(dst, src *Surface, srcRect, dstRect geom.IRect, caps *Caps) bool {
	dstSampleCnt := 0
	if rt := dst.AsRenderTarget(); rt != nil {
		dstSampleCnt = rt.NumSamples()
	}
	srcSampleCnt := 0
	if rt := src.AsRenderTarget(); rt != nil {
		srcSampleCnt = rt.NumSamples()
	}
	srcBounds := geom.IRectSize(src.Dimensions()).ToRect()
	return caps.CanCopyAsBlit(dst.Format(), dstSampleCnt, surfaceTexTypePtr(dst),
		src.Format(), srcSampleCnt, surfaceTexTypePtr(src), srcBounds, true, srcRect, dstRect)
}

// CopySurface copies pixels from srcRect in src to dstRect in dst. The rects must be contained in their respective
// surfaces; scaling requires the blit lane. filter follows gpu.SamplerState filtering (nearest/linear) for scaling
// blits.
func (g *Gpu) CopySurface(dst *Surface, dstRect geom.IRect, src *Surface, srcRect geom.IRect, filter gpu.FilterMode) bool {
	if dst.ReadOnly() {
		return false
	}
	if !geom.IRectSize(dst.Dimensions()).ContainsRect(dstRect) ||
		!geom.IRectSize(src.Dimensions()).ContainsRect(srcRect) {
		return false
	}
	g.handleDirtyContext()

	scalingCopy := dstRect.Size() != srcRect.Size()

	// Prefer copying with glCopyTexSubImage when the dimensions are the same.
	if !scalingCopy && canCopyTexSubImage(dst, src, g.glCaps()) {
		g.copySurfaceAsCopyTexSubImage(dst, src, srcRect, geom.IPoint{
			X: dstRect.Left,
			Y: dstRect.Top,
		})
		return true
	}

	if canBlitFramebufferForCopySurface(dst, src, srcRect, dstRect, g.glCaps()) {
		return g.copySurfaceAsBlitFramebuffer(dst, src, srcRect, dstRect, filter)
	}

	return false
}

// copySurfaceAsCopyTexSubImage copies srcRect in src to dstPoint in dst via glCopyTexSubImage2D.
func (g *Gpu) copySurfaceAsCopyTexSubImage(dst, src *Surface, srcRect geom.IRect, dstPoint geom.IPoint) {
	g.bindSurfaceFBOForPixelOps(src, 0, FRAMEBUFFER, tempFBOSrc)
	dstTex := dst.AsTexture()
	if dstTex == nil {
		panic("copyTexSubImage requires a texture destination")
	}
	// We modified the bound FBO.
	g.InvalidateBoundRenderTarget()

	g.bindTextureToScratchUnit(dstTex.Target(), dstTex.TextureID())
	g.fns().CopyTexSubImage2D(dstTex.Target(), 0, dstPoint.X, dstPoint.Y, srcRect.Left,
		srcRect.Top, srcRect.Width(), srcRect.Height())
	g.unbindSurfaceFBOForPixelOps(src, 0, FRAMEBUFFER)
	// The rect is already in device space so no flip is done (top-left origin).
	g.didWriteToSurface(dst)
}

// copySurfaceAsBlitFramebuffer copies srcRect in src to dstRect in dst via glBlitFramebuffer, scaling if the rects
// differ in size.
func (g *Gpu) copySurfaceAsBlitFramebuffer(dst, src *Surface, srcRect, dstRect geom.IRect, filter gpu.FilterMode) bool {
	if dst == src && dstRect.Intersects(srcRect) {
		return false
	}

	g.bindSurfaceFBOForPixelOps(dst, 0, DRAW_FRAMEBUFFER, tempFBODst)
	g.bindSurfaceFBOForPixelOps(src, 0, READ_FRAMEBUFFER, tempFBOSrc)
	// We modified the bound FBO.
	g.InvalidateBoundRenderTarget()

	// BlitFramebuffer respects the scissor, so disable it.
	g.flushScissorTest(false)
	g.disableWindowRectangles()

	glFilter := uint32(NEAREST)
	if filter == gpu.FilterModeLinear {
		glFilter = LINEAR
	}
	g.fns().BlitFramebuffer(srcRect.Left, srcRect.Top, srcRect.Right, srcRect.Bottom,
		dstRect.Left, dstRect.Top, dstRect.Right, dstRect.Bottom, COLOR_BUFFER_BIT, glFilter)
	g.unbindSurfaceFBOForPixelOps(dst, 0, DRAW_FRAMEBUFFER)
	g.unbindSurfaceFBOForPixelOps(src, 0, READ_FRAMEBUFFER)

	// The rect is already in device space so no flip is done (top-left origin).
	g.didWriteToSurface(dst)
	return true
}
