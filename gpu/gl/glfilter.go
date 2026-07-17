// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU device's image-filter hooks: the GPU-native filtercore backend plus the staging that routes filter DAGs
// between it and the readback→CPU→upload fallback. GPU-evaluable DAGs (per filtercore.CanEvaluateOnGPU — every node's
// shaders convert to FPs) evaluate entirely on the GPU: snapSpecial wraps the device's texture in a drawable-backed
// SpecialImage (no readback), intermediate surfaces are offscreen GPU devices, blur runs through GaussianBlur, and
// drawSpecial samples result textures directly. Everything else takes the readback→CPU→upload fallback — the DAG
// evaluates on the CPU raster backend — with a process-global visibility counter; the fallback's source snap is also
// texture-wrapped now, resolving to CPU pixels lazily inside filtercore (counted by filtercore.ResolveToRasterCount).

package gl

import (
	"math"
	"sync/atomic"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// filterReadbackCount tracks how many image-filter evaluations the GPU device has routed through the
// readback→CPU→upload fallback. It is a process-global visibility counter so the cost of any CPU-only filter DAG stays
// observable; GPU-evaluable DAGs bypass it. Now that the last six filter kernels have landed (morphology, displacement,
// lighting, arithmetic, matrix-convolution, magnifier — gpu/gl/filterkernelfp.go), every publicly reachable image
// filter evaluates GPU-natively, and the fallback remains only as the safety net for future non-evaluable nodes.
var filterReadbackCount atomic.Int64

// FilterReadbackCount returns the running count of GPU-device image-filter evaluations that took the
// readback→CPU→upload fallback — the visibility counter.
func FilterReadbackCount() int64 { return filterReadbackCount.Load() }

// AsFilterDevice implements canvas.Device: the filtercore.Device adapter that lowers filter draws, source snaps, and
// final composites through this GPU device; with the GPU backend it is also the intermediate-surface device the DAG
// renders through.
func (d *Device) AsFilterDevice() filtercore.Device { return &filterDevice{dev: d} }

// CreateFilterBackend implements canvas.Device: with the staged filter lowering, a DAG whose nodes all evaluate
// GPU-natively gets the GPU-native filtercore backend; anything else evaluates on the CPU raster backend, with the
// visibility counter firing once per fallback evaluation.
func (d *Device) CreateFilterBackend(filter filtercore.Filter) filtercore.Backend {
	if filtercore.CanEvaluateOnGPU(filter) {
		return &filterBackend{dev: d, cache: filtercore.NewFilterCache()}
	}
	filterReadbackCount.Add(1)
	return canvas.NewRasterFilterBackend()
}

// RelativeTransform returns the matrix mapping this device's coordinate space to dst's (dst.globalToDevice *
// this.deviceToGlobal). Needed so the canvas can build the layer→device mapping without knowing the concrete device
// type.
func (d *Device) RelativeTransform(dst canvas.Device) geom.Matrix {
	g2d := dst.GlobalToDevice()
	var m geom.Matrix
	m.SetConcat(&g2d, &d.deviceToGlobal)
	return m
}

// borrowTextureImage wraps a proxy view as a TextureImage without taking a proxy ref: filtercore's special images live
// only within one filter evaluation, during which the originating surface / render tasks keep the proxy alive, and any
// draw that samples it refs the proxy through the op stream. Nothing ever calls Release on these.
func borrowTextureImage(ctx *DirectContext, view SurfaceProxyView, colorType gpu.ColorType, dims geom.ISize) *TextureImage {
	return &TextureImage{
		ctx:       ctx,
		view:      view,
		colorType: colorType,
		alphaType: imagecore.AlphaTypePremul,
		dims:      dims,
		uniqueID:  nextGLImageID.Add(1),
	}
}

// SnapSpecial wraps this device's texture in a drawable-backed SpecialImage without a readback. A non-texturable target
// (a wrapped-FBO renderbuffer) takes the copy lane instead. The CPU-fallback DAG consuming the result resolves it to
// raster pixels lazily inside filtercore (the readback, counted there); the GPU DAG samples it directly.
func (d *Device) SnapSpecial(subset geom.IRect) *filtercore.SpecialImage {
	dims := d.sdc.Dimensions()
	if !subset.Intersect(geom.IRectSize(dims)) {
		return nil
	}
	view := d.sdc.ReadSurfaceView()
	if view.Proxy() == nil {
		return nil
	}
	ctx := d.sdc.Context()
	if view.AsTextureProxy() == nil {
		// The copy lane: copy the requested subset into a texture; the special image then views the whole copy.
		proxy, task := CopySurfaceProxy(ctx, view.Proxy(), view.Origin(), gpu.MipmappedNo, subset,
			gpu.BackingFitApprox, gpu.BudgetedYes, "Device_SnapSpecial")
		if proxy == nil || task == nil {
			return nil
		}
		copied := MakeSurfaceProxyView(proxy, view.Origin(), view.Swizzle())
		size := geom.ISize{Width: subset.Width(), Height: subset.Height()}
		img := borrowTextureImage(ctx, copied, d.sdc.ColorType(), size)
		// The copy task holds the reference that keeps the proxy alive; drop the caller-owned one.
		proxy.Unref()
		return filtercore.NewSpecialImageDrawable(geom.IRectWH(size.Width, size.Height), img)
	}
	img := borrowTextureImage(ctx, view, d.sdc.ColorType(), dims)
	return filtercore.NewSpecialImageDrawable(subset, img)
}

// MakeSubsetDrawable implements filtercore's subset-backing hook: a standalone texture copy of the subset, so
// strict-sampling shaders can tile at the subset bounds without a CPU resolve.
func (im *TextureImage) MakeSubsetDrawable(subset geom.IRect) imagecore.DrawableImage {
	proxy, task := CopySurfaceProxy(im.ctx, im.view.Proxy(), im.view.Origin(), gpu.MipmappedNo,
		subset, gpu.BackingFitApprox, gpu.BudgetedYes, "SpecialImage_MakeSubset")
	if proxy == nil || task == nil {
		return nil
	}
	copied := MakeSurfaceProxyView(proxy, im.view.Origin(), im.view.Swizzle())
	img := borrowTextureImage(im.ctx, copied, im.colorType,
		geom.ISize{Width: subset.Width(), Height: subset.Height()})
	proxy.Unref()
	return img
}

///////////////////////////////////////////////////////////////////////////////
// The GPU filtercore backend

// filterBackend is the GPU-native filtercore backend: intermediate devices are offscreen GPU render targets, images
// wrap textures, and blur runs on the GPU.
type filterBackend struct {
	dev   *Device
	cache *filtercore.FilterCache
}

var _ filtercore.Backend = (*filterBackend)(nil)

// MakeDevice returns a budgeted, approx-fit, single-sampled offscreen device. filtercore expects
// transparent-initialized devices, so clear here like Device.CreateDevice does.
func (b *filterBackend) MakeDevice(size geom.ISize) filtercore.Device {
	if size.IsEmpty() {
		return nil
	}
	sdc := MakeSurfaceDrawContext(b.dev.sdc.Context(), b.dev.sdc.ColorType(), size,
		gpu.BackingFitApprox, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes,
		"FilterDevice")
	if sdc == nil {
		return nil
	}
	dev := NewDevice(sdc)
	dev.clearAll()
	return dev.AsFilterDevice()
}

// MakeImage wraps image for use as a filter source: a texture-backed image on this context wraps directly; a raster
// image keeps its raster backing (uploading lazily at draw through the cached-bitmap lanes); a texture from another
// context resolves to CPU first (it cannot be sampled here).
func (b *filterBackend) MakeImage(subset geom.IRect, image imagecore.DrawableImage) *filtercore.SpecialImage {
	if ti, ok := image.(*TextureImage); ok && ti.Context() != b.dev.sdc.Context() {
		ci := ti.MakeNonTextureImage()
		if ci == nil {
			return nil
		}
		return filtercore.NewSpecialImageDrawable(subset, ci)
	}
	return filtercore.NewSpecialImageDrawable(subset, image)
}

// BlurEngine returns the GPU blur engine for this backend.
func (b *filterBackend) BlurEngine() filtercore.BlurEngine {
	return &blurEngine{dev: b.dev}
}

// Cache implements filtercore.Backend.
func (b *filterBackend) Cache() *filtercore.FilterCache { return b.cache }

///////////////////////////////////////////////////////////////////////////////
// The GPU blur engine

// maxBlurSigma is the largest sigma this algorithm handles directly; above it, filtercore pre-rescales the input before
// invoking the algorithm, keeping kernel widths bounded.
const maxBlurSigma = 4

// blurEngine is the GPU blur engine: one algorithm covers every sigma and tile mode.
type blurEngine struct {
	dev *Device
}

func (e *blurEngine) FindAlgorithm() filtercore.BlurAlgorithm {
	return &blurAlgorithm{dev: e.dev}
}

type blurAlgorithm struct {
	dev *Device
}

// MaxSigma implements filtercore.BlurAlgorithm.
func (a *blurAlgorithm) MaxSigma() float32 { return maxBlurSigma }

// SupportsOnlyDecalTiling implements filtercore.BlurAlgorithm: this algorithm handles all tile modes.
func (a *blurAlgorithm) SupportsOnlyDecalTiling() bool { return false }

// Blur implements filtercore.BlurAlgorithm via GaussianBlur. srcBounds and dstBounds arrive relative to src's logical
// bounds; a texture backing on this context translates them into backing-texel space and samples in place, anything
// else uploads the subset first.
func (a *blurAlgorithm) Blur(sigma geom.Size, src *filtercore.SpecialImage, srcBounds geom.IRect, tileMode shaders.TileMode, dstBounds geom.IRect) *filtercore.SpecialImage {
	ctx := a.dev.sdc.Context()
	var view SurfaceProxyView
	var colorType gpu.ColorType
	if ti, ok := src.DrawableBacking().(*TextureImage); ok && ti.Context() == ctx {
		view = ti.View()
		colorType = ti.ColorType()
		off := src.Subset()
		srcBounds = srcBounds.Offset(off.Left, off.Top)
		dstBounds = dstBounds.Offset(off.Left, off.Top)
	} else {
		ci := src.AsImage() // raster backing is direct; a foreign-context texture resolves (counted)
		if ci == nil {
			return nil
		}
		uploaded := TextureFromImage(ctx, ci, gpu.MipmappedNo, gpu.BudgetedYes)
		if uploaded == nil {
			return nil
		}
		view = uploaded.View()
		colorType = uploaded.ColorType()
	}
	if math.IsInf(float64(sigma.Width), 0) || math.IsInf(float64(sigma.Height), 0) {
		return nil
	}
	sdc := GaussianBlur(ctx, view, colorType, gpu.AlphaTypePremul, dstBounds, srcBounds,
		sigma.Width, sigma.Height, tileMode, gpu.BackingFitApprox)
	if sdc == nil {
		return nil
	}
	img := borrowTextureImage(ctx, sdc.ReadSurfaceView(), sdc.ColorType(), sdc.Dimensions())
	return filtercore.NewSpecialImageDrawable(geom.IRectWH(dstBounds.Width(), dstBounds.Height()), img)
}

///////////////////////////////////////////////////////////////////////////////
// The filtercore.Device adapter

// filterDevice adapts the GPU Device to filtercore.Device: on the GPU backend it is both the intermediate-surface
// device the DAG renders through and the endpoint adapter; on the CPU fallback it bridges only the endpoints (source
// snap and final composite) plus the shader-tiling DrawPaint/DrawRect lanes used for layer-filling effects.
type filterDevice struct {
	dev *Device
}

var _ filtercore.Device = (*filterDevice)(nil)

// Size implements filtercore.Device.
func (f *filterDevice) Size() geom.ISize { return f.dev.sdc.Dimensions() }

// LocalToDevice implements filtercore.Device.
func (f *filterDevice) LocalToDevice() geom.Matrix { return f.dev.localToDevice }

// SetLocalToDevice implements filtercore.Device.
func (f *filterDevice) SetLocalToDevice(m geom.Matrix) { f.dev.localToDevice = m }

// DevClipBounds implements filtercore.Device.
func (f *filterDevice) DevClipBounds() geom.IRect { return f.dev.DevClipBounds() }

// PushClipStack implements filtercore.Device.
func (f *filterDevice) PushClipStack() { f.dev.PushClipStack() }

// PopClipStack implements filtercore.Device.
func (f *filterDevice) PopClipStack() { f.dev.PopClipStack() }

// ClipRect implements filtercore.Device (local coordinates, intersect).
func (f *filterDevice) ClipRect(r geom.Rect, aa bool) {
	f.dev.ClipRect(r, raster.ClipIntersect, aa)
}

// ClipIRect implements filtercore.Device: intersect with a device-space integer rect (identity local matrix, non-AA).
func (f *filterDevice) ClipIRect(r geom.IRect) {
	identity := geom.IdentityMatrix()
	f.dev.clipStack.ClipRect(&identity, r.ToRect(), gpu.AANo, raster.ClipIntersect)
}

// DrawPaint implements filtercore.Device.
func (f *filterDevice) DrawPaint(pp *filtercore.PaintParams) {
	f.dev.DrawPaint(canvas.PaintFromFilterParams(pp))
}

// DrawRect implements filtercore.Device.
func (f *filterDevice) DrawRect(r geom.Rect, pp *filtercore.PaintParams) {
	f.dev.DrawRect(r, canvas.PaintFromFilterParams(pp))
}

// DrawSpecial implements filtercore.Device: the special image is placed by deviceMatrix (which already folds in the
// device's local-to-device — the device's own transform is ignored here). A drawable backing (texture in practice) is
// sampled directly over its subset (the strict constraint keeps neighbors in the backing store out); a raster one draws
// through the texture-upload lanes.
func (f *filterDevice) DrawSpecial(img *filtercore.SpecialImage, deviceMatrix geom.Matrix, sampling shaders.SamplingOptions, pp *filtercore.PaintParams) {
	p := canvas.PaintFromFilterParams(pp)
	saved := f.dev.localToDevice
	f.dev.localToDevice = deviceMatrix
	defer func() { f.dev.localToDevice = saved }()
	if backing := img.DrawableBacking(); backing != nil {
		src := img.Subset().ToRect()
		dst := geom.RectWH(float32(img.Width()), float32(img.Height()))
		f.dev.DrawImageRect(backing, &src, dst, sampling, p, canvas.ConstraintStrict)
		return
	}
	image := img.AsImage()
	if image == nil {
		return
	}
	dst := geom.RectWH(float32(image.Width()), float32(image.Height()))
	f.dev.DrawImageRect(image, &dst, dst, sampling, p, canvas.ConstraintFast)
}

// DrawImageRect implements filtercore.Device (the strict-constraint image draw the image-source leaf and the tiling
// lanes use); the GPU device's image draw is polymorphic (S118), so a texture-backed image samples directly.
func (f *filterDevice) DrawImageRect(img imagecore.DrawableImage, src, dst geom.Rect, sampling shaders.SamplingOptions, pp *filtercore.PaintParams) {
	f.dev.DrawImageRect(img, &src, dst, sampling, canvas.PaintFromFilterParams(pp),
		canvas.ConstraintStrict)
}

// SnapSpecial implements filtercore.Device.
func (f *filterDevice) SnapSpecial(subset geom.IRect) *filtercore.SpecialImage {
	return f.dev.SnapSpecial(subset)
}
