// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The read view over a proxy plus the read/write pixel entry points and proxy copies. Trims: the color info is reduced
// to the color type (alpha is always premul and the color space always sRGB), the canvas2D PM/UPM fast paths and the
// draw-based copy fallbacks are unported (they need fragment processors), the async rescale/read entry points are not
// publicly reachable, and CPU color-type conversion of read-back/uploaded pixels is limited to same-color-type
// repacking and vertical flips — mismatched color types fail exactly where an unsupported read/write would.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// Pixels describes CPU pixel data for the read/write entry points, pending the imagecore integration.
type Pixels struct {
	Data      []byte
	RowBytes  int
	Dims      geom.ISize
	ColorType gpu.ColorType
}

func (p *Pixels) minRowBytes() int { return p.ColorType.BytesPerPixel() * int(p.Dims.Width) }

// clip intersects with the surface bounds, adjusting pt and the base address. Returns false when nothing remains.
func (p *Pixels) clip(dims geom.ISize, pt *geom.IPoint) bool {
	target := geom.IRectXYWH(pt.X, pt.Y, p.Dims.Width, p.Dims.Height)
	if !target.Intersect(geom.IRectSize(dims)) {
		return false
	}
	byteOffset := (int(target.Top)-int(pt.Y))*p.RowBytes +
		(int(target.Left)-int(pt.X))*p.ColorType.BytesPerPixel()
	p.Data = p.Data[byteOffset:]
	p.Dims = target.Size()
	*pt = geom.IPoint{X: target.Left, Y: target.Top}
	return true
}

// SurfaceContext is a read view over a proxy, plus the read/write pixel entry points and proxy copies.
type SurfaceContext struct {
	ctx      *DirectContext
	readView SurfaceProxyView
	// colorType is the reachable slice of the full color info: everything else is fixed.
	colorType gpu.ColorType
}

// initSurfaceContext is the SurfaceContext constructor; the context takes a ref on the view's proxy for its lifetime
// (released by Release).
func (sc *SurfaceContext) initSurfaceContext(ctx *DirectContext, readView SurfaceProxyView, colorType gpu.ColorType) {
	if ctx.Abandoned() {
		panic("surface context on abandoned context")
	}
	sc.ctx = ctx
	sc.readView = readView
	sc.colorType = colorType
	readView.Proxy().Ref()
}

// Release drops the context's proxy ref; surfaces call this at close.
func (sc *SurfaceContext) Release() {
	if sc.readView.Proxy() != nil {
		sc.readView.Proxy().Unref()
		sc.readView.Reset()
	}
}

// Context returns the owning direct context.
func (sc *SurfaceContext) Context() *DirectContext { return sc.ctx }

// Caps returns the context's GL caps.
func (sc *SurfaceContext) Caps() *Caps { return sc.ctx.GLCaps() }

// drawingManager returns the owning context's drawing manager.
func (sc *SurfaceContext) drawingManager() *DrawingManager { return sc.ctx.DrawingManager() }

// AsSurfaceProxy returns the read view's proxy.
func (sc *SurfaceContext) AsSurfaceProxy() *SurfaceProxy { return sc.readView.Proxy() }

// AsTextureProxy returns the read view's texture facet, or nil.
func (sc *SurfaceContext) AsTextureProxy() *TextureProxy { return sc.readView.AsTextureProxy() }

// AsRenderTargetProxy returns the read view's render-target facet, or nil.
func (sc *SurfaceContext) AsRenderTargetProxy() *RenderTargetProxy {
	return sc.readView.AsRenderTargetProxy()
}

// ReadSurfaceView returns the context's read view.
func (sc *SurfaceContext) ReadSurfaceView() SurfaceProxyView { return sc.readView }

// Origin returns the read view's surface origin.
func (sc *SurfaceContext) Origin() gpu.SurfaceOrigin { return sc.readView.Origin() }

// Dimensions returns the read view's proxy dimensions.
func (sc *SurfaceContext) Dimensions() geom.ISize { return sc.readView.Proxy().Dimensions() }

// ColorType returns the context's color type.
func (sc *SurfaceContext) ColorType() gpu.ColorType { return sc.colorType }

// ReadPixels reads a rect of pixels starting at pt into dst. Flushes the surface before reading so pending work is
// visible.
func (sc *SurfaceContext) ReadPixels(dst Pixels, pt geom.IPoint) bool {
	if sc.ctx.Abandoned() {
		return false
	}
	if dst.ColorType == gpu.ColorTypeUnknown {
		return false
	}
	if dst.RowBytes%dst.ColorType.BytesPerPixel() != 0 {
		return false
	}
	if !dst.clip(sc.Dimensions(), &pt) {
		return false
	}

	srcProxy := sc.AsSurfaceProxy()
	if srcProxy.FramebufferOnly() {
		return false
	}
	if !srcProxy.Instantiate(sc.ctx.ResourceProvider()) {
		return false
	}
	srcSurface := srcProxy.PeekSurface()
	caps := sc.Caps()

	if caps.SurfaceSupportsReadPixels(srcSurface) != SurfaceReadPixelsSupported {
		// The copy-to-texture lane: blit into a readable texture, then read from that. (The canvas2D fast path and the
		// texture-source draw lane are unported.)
		restrictionsMustCopyWholeSrc := false
		if rtProxy := sc.AsRenderTargetProxy(); rtProxy != nil {
			// The GL caps never require whole-src copies on desktop GL for non-external textures; keep the variable for
			// structure.
			restrictionsMustCopyWholeSrc = false
			_ = rtProxy
		}
		var copyProxy *SurfaceProxy
		if restrictionsMustCopyWholeSrc {
			copyProxy, _ = CopySurfaceProxy(sc.ctx, srcProxy, sc.Origin(), gpu.MipmappedNo,
				geom.IRectSize(srcProxy.Dimensions()), gpu.BackingFitExact, gpu.BudgetedYes,
				"SurfaceContext_ReadPixelsWithCopyWholeSrc")
		} else {
			srcRect := geom.IRectXYWH(pt.X, pt.Y, dst.Dims.Width, dst.Dims.Height)
			copyProxy, _ = CopySurfaceProxy(sc.ctx, srcProxy, sc.Origin(), gpu.MipmappedNo,
				srcRect, gpu.BackingFitExact, gpu.BudgetedYes, "SurfaceContext_ReadPixels")
			pt = geom.IPoint{}
		}
		if copyProxy == nil {
			return false
		}
		var tempCtx SurfaceContext
		tempCtx.initSurfaceContext(sc.ctx,
			MakeSurfaceProxyView(copyProxy, sc.Origin(), sc.readView.Swizzle()), sc.colorType)
		// The temp context now holds its own ref; drop the creation ref from Copy.
		copyProxy.Unref()
		ok := tempCtx.ReadPixels(dst, pt)
		tempCtx.Release()
		return ok
	}

	flip := sc.Origin() == gpu.OriginBottomLeft

	supportedRead := caps.SupportedReadPixelsColorType(sc.colorType, srcProxy.Format(),
		dst.ColorType)
	if supportedRead.ColorType != dst.ColorType {
		// Cross-color-type conversion of read-back data waits for the imagecore integration.
		return false
	}

	makeTight := !caps.ReadPixelsRowBytesSupport && dst.RowBytes != dst.minRowBytes()
	convert := flip || makeTight

	readDst := dst.Data
	readRB := dst.RowBytes
	var tmp Pixels
	if convert {
		tmp = Pixels{
			ColorType: supportedRead.ColorType,
			Dims:      dst.Dims,
			RowBytes:  supportedRead.ColorType.BytesPerPixel() * int(dst.Dims.Width),
		}
		tmp.Data = make([]byte, tmp.RowBytes*int(dst.Dims.Height))
		readDst = tmp.Data
		readRB = tmp.RowBytes
		if flip {
			pt.Y = srcSurface.Height() - pt.Y - dst.Dims.Height
		}
	}

	sc.ctx.FlushSurfaces([]*SurfaceProxy{srcProxy}, FlushInfo{})
	sc.ctx.Submit(false)
	if !sc.ctx.Gpu().ReadPixels(srcSurface,
		geom.IRectXYWH(pt.X, pt.Y, dst.Dims.Width, dst.Dims.Height), sc.colorType,
		supportedRead.ColorType, readDst, readRB) {
		return false
	}

	if convert {
		return repackPixels(&dst, &tmp, flip)
	}
	return true
}

// WritePixels writes a single level of pixel data (the mip-level form arrives with the image work).
func (sc *SurfaceContext) WritePixels(src Pixels, dstPt geom.IPoint) bool {
	if sc.ctx.Abandoned() {
		return false
	}
	if !src.clip(sc.Dimensions(), &dstPt) {
		return false
	}
	if src.ColorType.BytesPerPixel() == 0 || src.RowBytes%src.ColorType.BytesPerPixel() != 0 {
		return false
	}
	if src.ColorType == gpu.ColorTypeUnknown {
		return false
	}

	dstProxy := sc.AsSurfaceProxy()
	if dstProxy.ReadOnly() || dstProxy.FramebufferOnly() {
		return false
	}
	if !dstProxy.Instantiate(sc.ctx.ResourceProvider()) {
		return false
	}
	dstSurface := dstProxy.PeekSurface()
	caps := sc.Caps()

	if !caps.SurfaceSupportsWritePixels(dstSurface) {
		// The draw/copy staging lanes are unported: they need fill contexts or an intermediate texture + copy; on the
		// desktop matrix this only occurs for MSAA render targets, which no reachable caller writes to.
		return false
	}

	allowed := caps.SupportedWritePixelsColorType(sc.colorType, dstProxy.Format(),
		src.ColorType).ColorType
	if allowed != src.ColorType {
		// Cross-color-type conversion waits for the imagecore integration.
		return false
	}

	flip := sc.Origin() == gpu.OriginBottomLeft
	mustBeTight := !caps.WritePixelsRowBytesSupport
	ownStorage := false
	if flip || (mustBeTight && src.RowBytes != src.minRowBytes()) {
		tmp := Pixels{
			ColorType: src.ColorType,
			Dims:      src.Dims,
			RowBytes:  src.minRowBytes(),
		}
		tmp.Data = make([]byte, tmp.RowBytes*int(src.Dims.Height))
		if !repackPixels(&tmp, &src, flip) {
			return false
		}
		src = tmp
		ownStorage = true
	}
	if flip {
		dstPt.Y = dstSurface.Height() - dstPt.Y - src.Dims.Height
	}

	if !sc.drawingManager().NewWritePixelsTask(dstProxy,
		geom.IRectXYWH(dstPt.X, dstPt.Y, src.Dims.Width, src.Dims.Height), src.ColorType,
		sc.colorType, []gpu.MipLevel{{Pixels: src.Data, RowBytes: src.RowBytes}}) {
		return false
	}
	if !ownStorage {
		// The task references the caller's storage; flush so the pixels are pushed to the GPU before returning
		// (whenever the pixel data isn't owned by us).
		sc.ctx.FlushSurfaces([]*SurfaceProxy{dstProxy}, FlushInfo{})
	}
	return true
}

// Copy records a copy of srcRect in src into this context's proxy at dstPoint. Returns the render task (nil on failure)
// so callers may mark it skippable.
func (sc *SurfaceContext) Copy(src *SurfaceProxy, srcRect geom.IRect, dstPoint geom.IPoint) *CopyRenderTask {
	if sc.ctx.Abandoned() {
		return nil
	}
	if sc.AsSurfaceProxy().FramebufferOnly() {
		return nil
	}
	dstRect := geom.IRectXYWH(dstPoint.X, dstPoint.Y, srcRect.Width(), srcRect.Height())
	if !sc.Caps().CanCopySurface(sc.AsSurfaceProxy(), dstRect, src, srcRect) {
		return nil
	}
	return sc.drawingManager().NewCopyRenderTask(sc.AsSurfaceProxy(), dstRect, src, srcRect,
		gpu.FilterModeNearest, sc.Origin())
}

// CopySurfaceProxy copies exact pixel values from src into a new non-renderable, non-multisampled proxy of the same
// format (the copy-task lane; the draw-based blit fallback is unported). Returns the new proxy (with a ref owned by the
// caller) and the copy task.
func CopySurfaceProxy(ctx *DirectContext, src *SurfaceProxy, origin gpu.SurfaceOrigin, mipmapped gpu.Mipmapped, srcRect geom.IRect, fit gpu.BackingFit, budgeted gpu.Budgeted, label string) (*SurfaceProxy, *CopyRenderTask) {
	if src.IsFullyLazy() {
		panic("cannot copy a fully lazy proxy")
	}
	if !srcRect.Intersect(geom.IRectSize(src.Dimensions())) {
		return nil, nil
	}
	// Size the dst proxy from the intersected rect so an oversized srcRect doesn't leave an uncopied border.
	dstProxy := ctx.ProxyProvider().CreateProxy(src.Format(),
		geom.ISize{Width: srcRect.Width(), Height: srcRect.Height()}, gpu.RenderableNo, 1, mipmapped, fit, budgeted,
		label, 0, UseAllocatorYes)
	if dstProxy == nil {
		return nil, nil
	}
	var dstCtx SurfaceContext
	dstCtx.initSurfaceContext(ctx, MakeSurfaceProxyView(dstProxy, origin, gpu.SwizzleRGBA),
		gpu.ColorTypeUnknown)
	copyTask := dstCtx.Copy(src, srcRect, geom.IPoint{})
	dstCtx.Release()
	if copyTask == nil {
		dstProxy.Unref()
		return nil, nil
	}
	return dstProxy, copyTask
}

// repackPixels repacks pixel data between two same-color-type layouts, with row-byte repacking and optional vertical
// flip.
func repackPixels(dst, src *Pixels, flip bool) bool {
	if dst.ColorType != src.ColorType || dst.Dims != src.Dims {
		return false
	}
	rowLen := src.ColorType.BytesPerPixel() * int(src.Dims.Width)
	h := int(src.Dims.Height)
	for y := 0; y < h; y++ {
		srcY := y
		if flip {
			srcY = h - 1 - y
		}
		copy(dst.Data[y*dst.RowBytes:y*dst.RowBytes+rowLen],
			src.Data[srcY*src.RowBytes:srcY*src.RowBytes+rowLen])
	}
	return true
}
