// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU-backed surface: a render-target-backed surface wrapping a SurfaceDrawContext + Device + Canvas, with image
// snapshots, the compatible-surface factory, and pixel readback. NewRenderTargetSurfaceFromBackendRenderTarget wraps a
// caller-owned FBO into a surface; MakeSurface makes a compatible offscreen GPU surface; MakeImageSnapshot produces a
// texture-backed image of the current content.
//
// Divergence: MakeImageSnapshot always takes the non-shareable path — it eagerly copies the target proxy into a
// read-only texture image rather than sharing the live proxy and copying on the surface's next write. The observable
// result is identical (the snapshot is frozen; the surface keeps drawing); only the timing of the copy differs. The
// share+copy-on-write proxy re-target is a deferred optimization.

package gl

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/surface"
)

// RenderTargetSurface is the GL-backed render-target surface.
type RenderTargetSurface struct {
	ctx    *DirectContext
	device *Device
	canvas *canvas.Canvas
}

// NewRenderTargetSurface wraps an existing SurfaceDrawContext as a GPU surface, building the canvas device over it.
// Returns nil for a nil draw context.
func NewRenderTargetSurface(sdc *SurfaceDrawContext) *RenderTargetSurface {
	if sdc == nil {
		return nil
	}
	dev := NewDevice(sdc)
	s := &RenderTargetSurface{
		ctx:    sdc.Context(),
		device: dev,
		canvas: canvas.New(dev),
	}
	return s
}

// MakeRenderTargetSurface creates a new offscreen GPU surface carrying the surface props (nil = default). Returns nil
// for bad dimensions, an unsupported format, or an abandoned context.
func MakeRenderTargetSurface(ctx *DirectContext, colorType gpu.ColorType, dims geom.ISize, sampleCnt int, mipmapped gpu.Mipmapped, origin gpu.SurfaceOrigin, budgeted gpu.Budgeted, label string, props *surface.Props) *RenderTargetSurface {
	if ctx == nil || dims.Width <= 0 || dims.Height <= 0 {
		return nil
	}
	sdc := MakeSurfaceDrawContextWithProps(ctx, colorType, dims, gpu.BackingFitExact, sampleCnt,
		mipmapped, origin, budgeted, label, props)
	if sdc == nil {
		return nil
	}
	return NewRenderTargetSurface(sdc)
}

// NewRenderTargetSurfaceFromBackendRenderTarget wraps a caller-owned GL FBO (unison's window FBO with its stencil bits)
// into a GPU surface carrying the surface props (nil = default). Returns nil on failure.
func NewRenderTargetSurfaceFromBackendRenderTarget(ctx *DirectContext, colorType gpu.ColorType, dims geom.ISize, format Format, sampleCnt, stencilBits int, fboID uint32, origin gpu.SurfaceOrigin, props *surface.Props) *RenderTargetSurface {
	sdc := MakeSurfaceDrawContextFromBackendRenderTarget(ctx, colorType, dims, format, sampleCnt,
		stencilBits, fboID, origin, props)
	if sdc == nil {
		return nil
	}
	return NewRenderTargetSurface(sdc)
}

// Canvas returns the surface's canvas.
func (s *RenderTargetSurface) Canvas() *canvas.Canvas { return s.canvas }

// Device exposes the surface's GPU device (tests and embedding callers).
func (s *RenderTargetSurface) Device() *Device { return s.device }

// SDC exposes the surface's draw context.
func (s *RenderTargetSurface) SDC() *SurfaceDrawContext { return s.device.SDC() }

// Context returns the owning direct context.
func (s *RenderTargetSurface) Context() *DirectContext { return s.ctx }

// Release drops what the surface's device holds beyond its own lifetime — the clip stack's cached SW masks (see
// Device.Release). A surface dropped with a clip still set would otherwise leave those mask textures unique-keyed in the
// resource cache until budget pressure or context teardown reclaims them. The surface's own render target belongs to the
// draw context and is not touched, so this is not a substitute for tearing down the context. Safe to call more than once.
func (s *RenderTargetSurface) Release() { s.device.Release() }

// Width returns the surface's width in pixels.
func (s *RenderTargetSurface) Width() int32 { return s.device.SDC().Dimensions().Width }

// Height returns the surface's height in pixels.
func (s *RenderTargetSurface) Height() int32 { return s.device.SDC().Dimensions().Height }

// MakeSurface returns a compatible offscreen GPU surface with the same color type, sample count, origin, mipmap policy,
// and surface props. Returns nil on failure.
func (s *RenderTargetSurface) MakeSurface(width, height int32) *RenderTargetSurface {
	sdc := s.device.SDC()
	mipmapped := gpu.MipmappedNo
	rv := sdc.ReadSurfaceView()
	if rv.Mipmapped() == gpu.MipmappedYes {
		mipmapped = gpu.MipmappedYes
	}
	props := sdc.SurfaceProps()
	return MakeRenderTargetSurface(s.ctx, sdc.ColorType(), geom.ISize{Width: width, Height: height},
		sdc.NumSamples(), mipmapped, sdc.Origin(), gpu.BudgetedYes, "Surface_makeSurface", &props)
}

// MakeImageSnapshot takes the copy branch: the current target proxy is copied into a read-only texture image so the
// snapshot is independent of subsequent draws to the surface. Returns nil when the copy fails (an abandoned context, an
// uncopyable proxy).
func (s *RenderTargetSurface) MakeImageSnapshot() *TextureImage {
	if s.ctx.Abandoned() {
		return nil
	}
	sdc := s.device.SDC()
	srcView := sdc.ReadSurfaceView()
	srcProxy := srcView.Proxy()
	dims := srcProxy.Dimensions()
	colorType := sdc.ColorType()

	copyProxy, _ := CopySurfaceProxy(s.ctx, srcProxy, srcView.Origin(), gpu.MipmappedNo,
		geom.IRectSize(dims), gpu.BackingFitExact, gpu.BudgetedYes, "SurfaceSnapshot")
	if copyProxy == nil {
		return nil
	}
	// Bake the snapshot at snapshot time (the eager-copy model): flushing the copy proxy runs the copy render task and
	// its dependency on the surface's current content, so the snapshot is frozen regardless of the order of later draws
	// and reads on the surface.
	s.ctx.FlushSurfaces([]*SurfaceProxy{copyProxy}, FlushInfo{})
	swizzle := s.ctx.GLCaps().ReadSwizzle(copyProxy.Format(), colorType)
	view := MakeSurfaceProxyView(copyProxy, srcView.Origin(), swizzle)
	img := newTextureImage(s.ctx, view, colorType, imagecore.AlphaTypePremul, dims)
	// The image took its own ref; drop the creation ref from Copy.
	copyProxy.Unref()
	return img
}

// ReadPixels reads the surface's current content and converts into dstInfo at (srcX, srcY). Like the image readback,
// the GPU read produces a CPU image at the native color type and the color-type/alpha-type conversion and read-rect
// trim reuse imagecore.
func (s *RenderTargetSurface) ReadPixels(dstInfo imagecore.ImageInfo, dst []byte, dstRowBytes int, srcX, srcY int32) bool {
	sdc := s.device.SDC()
	raster := readGPUToRasterImage(s.ctx, sdc.ReadSurfaceView(), sdc.ColorType(),
		imagecore.AlphaTypePremul)
	if raster == nil {
		return false
	}
	return raster.ReadPixels(dstInfo, dst, dstRowBytes, srcX, srcY, imagecore.CachingAllow)
}
