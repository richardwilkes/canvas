// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The texture-backed image type: a GPU image is a view over a texture proxy plus its color info. TextureFromImage
// uploads a CPU imagecore image to a texture (reusing the device's shared, uniquely-keyed image-upload lane);
// ReadPixels and MakeNonTextureImage read the texture back into CPU memory through a SurfaceContext. Trims: mipmap
// generation is deferred (the shared upload lane produces a single non-mipmapped level — cubic/mip sampling degrades to
// linear as elsewhere), and the "already texture-backed on this context" identity short-circuit is unreachable here
// because the only CPU image type is imagecore.Image; the budgeted flag is accepted but the shared upload lane always
// budgets.

package gl

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/imagecore"
)

// nextGLImageID assigns each GPU image its own unique ID, independent of the source bitmap's id and of the shared
// image-proxy cache key (which still dedups by source content).
var nextGLImageID atomic.Uint32

// TextureImage is the GL-backed GPU image: an immutable image whose pixels live in a texture proxy on a specific direct
// context.
type TextureImage struct {
	ctx       *DirectContext
	view      SurfaceProxyView
	colorType gpu.ColorType
	alphaType imagecore.AlphaType
	dims      geom.ISize
	uniqueID  uint32
}

// newTextureImage wraps an already-built proxy view as a GPU image, taking a ref on the proxy for the image's lifetime
// (released by Release).
func newTextureImage(ctx *DirectContext, view SurfaceProxyView, colorType gpu.ColorType, alphaType imagecore.AlphaType, dims geom.ISize) *TextureImage {
	view.Proxy().Ref()
	return &TextureImage{
		ctx:       ctx,
		view:      view,
		colorType: colorType,
		alphaType: alphaType,
		dims:      dims,
		uniqueID:  nextGLImageID.Add(1),
	}
}

// TextureFromImage uploads a CPU image to a texture-backed image on ctx. Returns nil for nil inputs, an abandoned
// context, or an unsupported/failed upload. mipmapped and budgeted are accepted but the shared upload lane produces a
// single budgeted level.
func TextureFromImage(ctx *DirectContext, img *imagecore.Image, mipmapped gpu.Mipmapped, budgeted gpu.Budgeted) *TextureImage {
	_ = mipmapped
	_ = budgeted
	if ctx == nil || img == nil || ctx.Abandoned() {
		return nil
	}
	view, colorType := imageAsView(ctx, img)
	if !view.IsValid() {
		return nil
	}
	// The shared upload lane hands back a lazy proxy (instantiated at flush when drawn). A texture-backed image must be
	// usable immediately — including readback, which has no draw to force instantiation — so pin the texture now.
	if proxy := view.Proxy(); proxy.IsLazy() {
		if !proxy.doLazyInstantiation(ctx.ResourceProvider()) {
			return nil
		}
	}
	return newTextureImage(ctx, view, colorType, img.AlphaType(),
		geom.ISize{Width: img.Width(), Height: img.Height()})
}

// Context returns the direct context the image's texture lives on.
func (im *TextureImage) Context() *DirectContext { return im.ctx }

// Width returns the image's width in pixels.
func (im *TextureImage) Width() int32 { return im.dims.Width }

// Height returns the image's height in pixels.
func (im *TextureImage) Height() int32 { return im.dims.Height }

// Dimensions returns the image's pixel dimensions.
func (im *TextureImage) Dimensions() geom.ISize { return im.dims }

// UniqueID returns the image's process-unique identifier.
func (im *TextureImage) UniqueID() uint32 { return im.uniqueID }

// ColorType returns the image's GPU color type (RGBA8888 or Alpha8 under the supported color-type matrix).
func (im *TextureImage) ColorType() gpu.ColorType { return im.colorType }

// AlphaType returns the image's alpha type.
func (im *TextureImage) AlphaType() imagecore.AlphaType { return im.alphaType }

// IsAlphaOnly reports whether the image carries only an alpha channel.
func (im *TextureImage) IsAlphaOnly() bool { return im.colorType == gpu.ColorTypeAlpha8 }

// IsTextureBacked reports whether the image's pixels live in a GPU texture: always true here.
func (im *TextureImage) IsTextureBacked() bool { return true }

// View returns the image's texture proxy view for the shader/draw consumers (this is wired into image shaders and
// drawImageRect). The returned view shares the underlying proxy; the caller must not outlive the image without taking
// its own ref.
func (im *TextureImage) View() SurfaceProxyView { return im.view }

// Release drops the image's ref on its texture proxy (a real refcounted release — GPU resources are freed
// deterministically through the resource cache, not by the GC). After Release the image must not be used. The
// debug-mode GC-finalizer leak detector is deferred; callers are responsible for calling Release themselves.
func (im *TextureImage) Release() {
	if im.view.Proxy() != nil {
		im.view.Proxy().Unref()
		im.view.Reset()
	}
}

// readToRasterImage reads the whole texture back into a CPU imagecore image at the GPU color type (premul); this is the
// shared readback path behind both MakeNonTextureImage and ReadPixels. Returns nil when the context is abandoned or the
// read fails.
func (im *TextureImage) readToRasterImage() *imagecore.Image {
	return readGPUToRasterImage(im.ctx, im.view, im.colorType, im.alphaType)
}

// MakeNonTextureImage returns a CPU image holding a readback of the texture's pixels. Returns nil when the read fails
// (an abandoned context, an unreadable format).
func (im *TextureImage) MakeNonTextureImage() *imagecore.Image {
	return im.readToRasterImage()
}

// ReadPixels reads the texture back and converts into dstInfo at (srcX, srcY). The GPU read produces a CPU image at the
// native color type; the color-type/alpha-type conversion and the read-rect trim are then handled by imagecore.
func (im *TextureImage) ReadPixels(dstInfo imagecore.ImageInfo, dst []byte, dstRowBytes int, srcX, srcY int32, hint imagecore.CachingHint) bool {
	raster := im.readToRasterImage()
	if raster == nil {
		return false
	}
	return raster.ReadPixels(dstInfo, dst, dstRowBytes, srcX, srcY, hint)
}

// gpuColorTypeToImagecore maps the GPU color types a surface/image can carry under the supported color-type matrix to
// their imagecore equivalents. ok is false for a color type with no CPU readback representation here.
func gpuColorTypeToImagecore(ct gpu.ColorType) (imagecore.ColorType, bool) {
	switch ct {
	case gpu.ColorTypeRGBA8888:
		return imagecore.ColorTypeRGBA8888, true
	case gpu.ColorTypeBGRA8888:
		return imagecore.ColorTypeBGRA8888, true
	case gpu.ColorTypeAlpha8:
		return imagecore.ColorTypeAlpha8, true
	default:
		return 0, false
	}
}

// readGPUToRasterImage reads the entire proxy behind view into a fresh CPU imagecore image at the GPU color type. It
// drives a bare SurfaceContext (the same read path the surface/image readbacks use), so it works for any readable proxy
// — a render target or a plain texture (bound to a temporary FBO for the read).
func readGPUToRasterImage(ctx *DirectContext, view SurfaceProxyView, colorType gpu.ColorType, alphaType imagecore.AlphaType) *imagecore.Image {
	if ctx == nil || ctx.Abandoned() {
		return nil
	}
	icColorType, ok := gpuColorTypeToImagecore(colorType)
	if !ok {
		return nil
	}
	dims := view.Proxy().Dimensions()
	if dims.IsEmpty() {
		return nil
	}
	rowBytes := colorType.BytesPerPixel() * int(dims.Width)
	buf := make([]byte, rowBytes*int(dims.Height))

	var sc SurfaceContext
	sc.initSurfaceContext(ctx, view, colorType)
	defer sc.Release()
	dst := Pixels{ColorType: colorType, Dims: dims, RowBytes: rowBytes, Data: buf}
	if !sc.ReadPixels(dst, geom.IPoint{}) {
		return nil
	}

	info, ok := imagecore.MakeInfo(dims.Width, dims.Height, icColorType, alphaType)
	if !ok {
		return nil
	}
	return imagecore.NewRasterData(info, buf, rowBytes)
}
