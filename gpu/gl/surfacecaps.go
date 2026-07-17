// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Surface- and proxy-level caps queries: whether a surface supports reading or writing pixels, and whether one surface
// can be copied to another.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// SurfaceReadPixelsSupport describes how (or whether) a surface's pixels can be read back.
type SurfaceReadPixelsSupport int32

// SurfaceReadPixelsSupport values.
const (
	// SurfaceReadPixelsSupported: glReadPixels can read directly from the surface.
	SurfaceReadPixelsSupported SurfaceReadPixelsSupport = iota
	// SurfaceReadPixelsCopyToTexture2D: the surface must be copied to a 2D texture first.
	SurfaceReadPixelsCopyToTexture2D
	// SurfaceReadPixelsUnsupported: the surface cannot be read back at all.
	SurfaceReadPixelsUnsupported
)

// SurfaceSupportsReadPixels reports how surface's pixels can be read back. Compressed and external textures are not
// modeled here, so the only case handled is a multisampled render target with no resolve texture, which must be copied
// first.
func (c *Caps) SurfaceSupportsReadPixels(surface *Surface) SurfaceReadPixelsSupport {
	if surface.AsTexture() == nil {
		if rt := surface.AsRenderTarget(); rt != nil {
			// glReadPixels does not allow reading back from a MSAA framebuffer. If the underlying surface doesn't have
			// a second FBO to resolve to then we must make a copy.
			if rt.NumSamples() > 1 {
				return SurfaceReadPixelsCopyToTexture2D
			}
		}
	}
	return SurfaceReadPixelsSupported
}

// SurfaceSupportsWritePixels reports whether surface's pixels can be written directly. A read-only surface never
// supports writes, and a multisampled render target backed by separate MSAA renderbuffers can only be written to if it
// is also a texture.
func (c *Caps) SurfaceSupportsWritePixels(surface *Surface) bool {
	if surface.ReadOnly() {
		return false
	}
	if rt := surface.AsRenderTarget(); rt != nil {
		if rt.NumSamples() > 1 && c.UsesMSAARenderBuffers() {
			return false
		}
		return surface.AsTexture() != nil
	}
	return true
}

func proxyTexTypePtr(p *SurfaceProxy) *gpu.TextureType {
	if tex := p.AsTextureProxy(); tex != nil {
		t := tex.TextureType()
		return &t
	}
	return nil
}

// CanCopySurface reports whether Gpu.CopySurface can succeed for these proxies once instantiated, checking format and
// bounds compatibility and then whether a blit-style copy can handle the requested regions and sample counts. Copies
// that only a draw-based copy could satisfy report false, since that path is not implemented.
func (c *Caps) CanCopySurface(dst *SurfaceProxy, dstRect geom.IRect, src *SurfaceProxy, srcRect geom.IRect) bool {
	if dst.ReadOnly() {
		return false
	}
	if dst.Format() != src.Format() {
		return false
	}
	// All Gpu.CopySurface calls can assume srcRect and dstRect are contained in their surfaces.
	if !geom.IRectSize(dst.Dimensions()).ContainsRect(dstRect) ||
		!geom.IRectSize(src.Dimensions()).ContainsRect(srcRect) {
		return false
	}

	dstSampleCnt := 0
	srcSampleCnt := 0
	if rtProxy := dst.AsRenderTargetProxy(); rtProxy != nil {
		dstSampleCnt = rtProxy.NumSamples()
	}
	if rtProxy := src.AsRenderTargetProxy(); rtProxy != nil {
		srcSampleCnt = rtProxy.NumSamples()
	}

	// has_msaa_render_buffer at the proxy level: multisampled, separate MSAA renderbuffers in use, and not FBO 0.
	proxyHasMSAARenderBuffer := func(p *SurfaceProxy, sampleCnt int) bool {
		rtProxy := p.AsRenderTargetProxy()
		return rtProxy != nil && sampleCnt > 1 && c.UsesMSAARenderBuffers() &&
			!rtProxy.GLRTFBOIDIs0()
	}

	// Only the blit (and eventually draw) lanes can handle scaling between src and dst.
	scalingCopy := srcRect.Size() != dstRect.Size()
	if !scalingCopy && c.CanCopyTexSubImage(dst.Format(),
		proxyHasMSAARenderBuffer(dst, dstSampleCnt), proxyTexTypePtr(dst), src.Format(),
		proxyHasMSAARenderBuffer(src, srcSampleCnt), proxyTexTypePtr(src)) {
		return true
	}
	return c.CanCopyAsBlit(dst.Format(), dstSampleCnt, proxyTexTypePtr(dst), src.Format(),
		srcSampleCnt, proxyTexTypePtr(src), src.BoundsIRect().ToRect(), src.isExact(), srcRect,
		dstRect)
}
