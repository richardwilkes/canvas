// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The mask-filter draw lane: DrawShapeWithMaskFilter and its helpers. A blurred (or otherwise mask-filtered) shape is
// turned into an A8 coverage mask that is filtered and then drawn as a device-space coverage rect. Large blurs generate
// and blur the mask entirely on the GPU (createMaskGPU + filterMask over the session-45 GaussianBlur engine); small
// blurs (and every non-blur filter) render the filtered mask on the CPU and upload it (swCreateFilteredMask), matching
// the standard minimum-GPU-blur-size gate.
//
// directFilterMask's analytic blur profiles are all implemented (rect/circle and simple-circular rrect), so a
// normal-blurred simple-fill rect/circle/rrect renders through one textured coverage draw.
//
// computeKeyAndClipBounds and its thread-safe cache: an axis-aligned, unstyled-keyable, >50%-visible blurred shape has
// its filtered A8 mask memoized under a unique key so a repeated draw (unison redraws the same shadowed primitives
// frame after frame) reuses the upload, and the clip no longer bounds the mask (boundsForClip becomes the unclipped
// shape bounds so the same key serves every clip). The HW lane (large blurs) takes an explicit long-lived ref on the
// rendered mask proxy before adding it (this codebase is single-threaded, and the HW lane runs before the SW lane; a
// given key deterministically takes one lane); the SW lane's lazy-upload A8 view is cached exactly as the analytic
// profiles are.

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// kMaskOrigin is the coordinate origin used for generated mask surfaces.
const kMaskOrigin = gpu.OriginTopLeft

// maskFilteredMaskDomain namespaces the mask-filtered-mask thread-safe cache entries.
var maskFilteredMaskDomain = gpu.GenerateUniqueKeyDomain()

// createMaskData stores the filtered mask's draw rect as an offset from the unclipped, integerized, device-space shape
// bounds plus the mask's size, serialized as the UniqueKey's custom data. On a cache hit the offset is re-anchored to
// the current draw's shape bounds so an integer-translated redraw reuses the mask.
func createMaskData(drawRect, origDevBounds geom.IRect) []byte {
	data := make([]byte, 16)
	binary.LittleEndian.PutUint32(data[0:], uint32(drawRect.Left-origDevBounds.Left))
	binary.LittleEndian.PutUint32(data[4:], uint32(drawRect.Top-origDevBounds.Top))
	binary.LittleEndian.PutUint32(data[8:], uint32(drawRect.Width()))
	binary.LittleEndian.PutUint32(data[12:], uint32(drawRect.Height()))
	return data
}

// extractDrawRectFromData is the inverse of createMaskData.
func extractDrawRectFromData(data []byte, origDevBounds geom.IRect) geom.IRect {
	offX := int32(binary.LittleEndian.Uint32(data[0:]))
	offY := int32(binary.LittleEndian.Uint32(data[4:]))
	w := int32(binary.LittleEndian.Uint32(data[8:]))
	h := int32(binary.LittleEndian.Uint32(data[12:]))
	return geom.IRectXYWH(origDevBounds.Left+offX, origDevBounds.Top+offY, w, h)
}

// blurMaskFilterQuery is the surface the mask-filter draw lane needs from a blur mask filter (the concrete
// *maskfilter.blurMaskFilter satisfies it). A maskfilter.MaskFilter that is not a blur fails the type assertion and
// takes the generic CPU swCreateFilteredMask lane.
type blurMaskFilterQuery interface {
	Sigma() float32
	Style() maskfilter.BlurStyle
	RespectCTM() bool
	ComputeXformedSigma(ctm *geom.Matrix) float32
}

// clipBoundsQuickReject reports whether rect can be trivially rejected against clipBounds.
func clipBoundsQuickReject(clipBounds, rect geom.IRect) bool {
	return clipBounds.IsEmpty() || rect.IsEmpty() || !clipBounds.Intersects(rect)
}

// drawMask draws mask as textured coverage: the coverage/geometry is already burnt into the mask, so this reduces to
// drawing a device-space rect textured (broadcast alpha) by the mask. Returns true if the mask was drawn. The paint is
// consumed.
func drawMask(sdc *SurfaceDrawContext, clip Clip, viewMatrix *geom.Matrix, maskBounds geom.IRect, paint *Paint, mask SurfaceProxyView) bool {
	inverse, ok := viewMatrix.Invert()
	if !ok {
		return false
	}

	mask.ConcatSwizzle(gpu.MakeSwizzle("aaaa"))

	matrix := geom.TranslateMatrix(-float32(maskBounds.Left), -float32(maskBounds.Top))
	matrix.PreConcat(viewMatrix)
	paint.SetCoverageFragmentProcessor(MakeTextureEffectSimple(mask, gpu.AlphaTypeUnknown, matrix,
		gpu.FilterModeNearest, gpu.MipmapModeNone))

	sdc.FillPixelsWithLocalMatrix(clip, paint, maskBounds, &inverse)
	return true
}

// createMaskGPU renders 'shape' into a fresh A8 mask with the shape's integerized top-left at the origin.
func createMaskGPU(ctx *DirectContext, maskRect geom.IRect, origViewMatrix *geom.Matrix, shape *StyledShape, sampleCnt int) *SurfaceDrawContext {
	// Use GetApproxSize for our own approximate size matching but demand an exact backing fit: the filter reaches
	// outside the src bounds, so the extra (cleared, transparent) border pixels give the decal sampling effect the blur
	// needs (reads outside the shape return alpha=0).
	approxSize := gpu.GetApproxSize(maskRect.Size())
	sdc := MakeSurfaceDrawContext(ctx, gpu.ColorTypeAlpha8, approxSize, gpu.BackingFitExact,
		sampleCnt, gpu.MipmappedNo, kMaskOrigin, gpu.BudgetedYes, "SurfaceDrawContext_CreateMaskGPU")
	if sdc == nil {
		return nil
	}

	sdc.Clear([4]float32{}) // transparent

	maskPaint := NewPaint()
	maskPaint.SetCoverageSetOpXPFactory(raster.RegionReplace, false)

	clip := NewFixedClipRect(sdc.Dimensions(), geom.IRectWH(maskRect.Width(), maskRect.Height()))

	viewMatrix := *origViewMatrix
	viewMatrix.PostTranslate(-float32(maskRect.Left), -float32(maskRect.Top))
	shapeCopy := shape.clone()
	sdc.DrawShape(clip, maskPaint, gpu.AAYes, &viewMatrix, &shapeCopy)
	return sdc
}

// blurShapeAndClipBounds returns the clip bounds, the unclipped shape dev bounds, and whether the shape has a
// (non-empty) styled bound. It does not require the shape to intersect the clip (an inverse-filled shape can still
// draw). getUnclippedShapeDevBounds uses the shape's finite styled bounds; an infinite-bounds inverse-fill lane is not
// implemented — inverse-filled mask-filtered shapes therefore blur over their finite bounds, matching this codebase's
// SW path renderer.
func blurShapeAndClipBounds(sdc *SurfaceDrawContext, clip Clip, shape *StyledShape, matrix *geom.Matrix) (unclippedDevShapeBounds, devClipBounds geom.IRect, ok bool) {
	devClipBounds = sdc.clipConservativeBounds(clip)
	unclippedDevShapeBounds, ok = getUnclippedShapeDevBounds(shape, matrix)
	if !ok {
		unclippedDevShapeBounds = geom.IRect{}
		return unclippedDevShapeBounds, devClipBounds, false
	}
	return unclippedDevShapeBounds, devClipBounds, true
}

// canFilterMask reports whether filterMask would succeed on the GPU, returning the device-space mask rect the filter
// needs. Blur only; small blurs prefer the CPU.
func canFilterMask(mf maskfilter.MaskFilter, devSpaceShapeBounds, clipBounds geom.IRect, ctm *geom.Matrix) (maskRect geom.IRect, ok bool) {
	bmf, isBlur := mf.(blurMaskFilterQuery)
	if !isBlur {
		return geom.IRect{}, false
	}
	xformedSigma := bmf.ComputeXformedSigma(ctm)
	if gpu.BlurIsEffectivelyIdentity(xformedSigma) {
		maskRect = devSpaceShapeBounds
		if !maskRect.Intersect(clipBounds) {
			return geom.IRect{}, false
		}
		return maskRect, true
	}

	sigma3 := int32(3 * xformedSigma)
	// Outset srcRect and clipRect by 3*sigma to compute the affected blur area.
	clipRect := clipBounds.Inset(-sigma3, -sigma3)
	srcRect := devSpaceShapeBounds.Inset(-sigma3, -sigma3)
	if !srcRect.Intersect(clipRect) {
		srcRect = geom.IRect{}
	}
	maskRect = srcRect

	// We prefer to blur paths with small blur radii on the CPU.
	const kMinGPUBlurSize = int32(64)
	const kMinGPUBlurSigma = float32(32)
	if devSpaceShapeBounds.Width() <= kMinGPUBlurSize &&
		devSpaceShapeBounds.Height() <= kMinGPUBlurSize && xformedSigma <= kMinGPUBlurSigma {
		return maskRect, false
	}
	return maskRect, true
}

// filterMask Gaussian-blurs the src mask on the GPU, compositing the original mask back in for the non-normal blur
// styles.
func filterMask(ctx *DirectContext, mf maskfilter.MaskFilter, srcView SurfaceProxyView, srcColorType gpu.ColorType, srcAlphaType gpu.AlphaType, ctm *geom.Matrix, maskRect geom.IRect) SurfaceProxyView {
	bmf, ok := mf.(blurMaskFilterQuery)
	if !ok {
		return SurfaceProxyView{}
	}
	// 'maskRect' isn't snapped to the UL corner but the mask in 'src' is.
	clipRect := geom.IRectWH(maskRect.Width(), maskRect.Height())

	xformedSigma := bmf.ComputeXformedSigma(ctm)

	// For a normal blur we can clobber the src in the GaussianBlur; otherwise we save it for the compositing pass
	// below.
	isNormalBlur := bmf.Style() == maskfilter.BlurNormal
	srcBounds := geom.IRectSize(srcView.Proxy().Dimensions())
	sdc := GaussianBlur(ctx, srcView, srcColorType, srcAlphaType, clipRect, srcBounds, xformedSigma,
		xformedSigma, shaders.TileClamp, gpu.BackingFitApprox)
	if sdc == nil || sdc.AsTextureProxy() == nil {
		return SurfaceProxyView{}
	}

	if !isNormalBlur {
		paint := NewPaint()
		// Blend the original mask over the blurred mask via a coverage set-op.
		paint.SetCoverageFragmentProcessor(MakeTextureEffectSimple(srcView, srcAlphaType,
			geom.IdentityMatrix(), gpu.FilterModeNearest, gpu.MipmapModeNone))
		switch bmf.Style() {
		case maskfilter.BlurInner:
			// inner: dst = dst * src
			paint.SetCoverageSetOpXPFactory(raster.RegionIntersect, false)
		case maskfilter.BlurSolid:
			// solid: dst = src + (1 - src) * dst
			paint.SetCoverageSetOpXPFactory(raster.RegionUnion, false)
		case maskfilter.BlurOuter:
			// outer: dst = (1 - src) * dst
			paint.SetCoverageSetOpXPFactory(raster.RegionDifference, false)
		default:
			paint.SetCoverageSetOpXPFactory(raster.RegionReplace, false)
		}
		identity := geom.IdentityMatrix()
		sdc.FillPixelsWithLocalMatrix(nil, paint, clipRect, &identity)
	}

	return sdc.ReadSurfaceView()
}

// hwCreateFilteredMask generates the mask on the GPU and Gaussian-blurs it, memoizing the result in the thread-safe
// cache. On failure the caller drops to the SW lane. This codebase is single-threaded and the HW lane runs before the
// SW lane in DrawShapeWithMaskFilter, so a plain find-then-generate-then-add gives GPU-filtered masks priority over SW
// ones. A given key deterministically takes one lane (canFilterMask's GPU-vs-CPU decision is a function of the shape
// bounds and sigma, both in the key), so HW keys are only ever read/written here.
func hwCreateFilteredMask(ctx *DirectContext, sdc *SurfaceDrawContext, viewMatrix *geom.Matrix, shape *StyledShape, filter maskfilter.MaskFilter, unclippedDevShapeBounds, clipBounds geom.IRect, key *gpu.UniqueKey) (view SurfaceProxyView, maskRect geom.IRect, ok bool) {
	maskRect, ok = canFilterMask(filter, unclippedDevShapeBounds, clipBounds, viewMatrix)
	if !ok {
		return SurfaceProxyView{}, geom.IRect{}, false
	}
	if clipBoundsQuickReject(clipBounds, maskRect) {
		return SurfaceProxyView{}, geom.IRect{}, false
	}

	threadSafeCache := ctx.ThreadSafeCache()
	if key.IsValid() {
		if cachedView, data := threadSafeCache.FindWithData(key); cachedView.IsValid() {
			maskRect = extractDrawRectFromData(data, unclippedDevShapeBounds)
			return cachedView, maskRect, true
		}
	}

	maskSDC := createMaskGPU(ctx, maskRect, viewMatrix, shape, sdc.NumSamples())
	if maskSDC == nil {
		return SurfaceProxyView{}, geom.IRect{}, false
	}

	view = filterMask(ctx, filter, maskSDC.ReadSurfaceView(), maskSDC.ColorType(),
		gpu.AlphaTypePremul, viewMatrix, maskRect)
	if view.Proxy() == nil {
		return SurfaceProxyView{}, geom.IRect{}, false
	}

	if key.IsValid() {
		// The filtered-mask view's proxy is owned by its producing render task; take an explicit ref so the cache is a
		// proper long-lived owner (matching the DstProxyView pattern) that Unrefs on eviction. Because the producing
		// task keeps a ref until flush, the cached proxy is never uniquely held (hence never purged) mid-recording, and
		// after flush the cache holds the sole ref and can be dropped under budget pressure, regenerating on the next
		// miss.
		view.Proxy().Ref()
		key.SetCustomData(createMaskData(maskRect, unclippedDevShapeBounds))
		var data []byte
		view, data = threadSafeCache.AddWithData(key, view)
		maskRect = extractDrawRectFromData(data, unclippedDevShapeBounds)
	}
	return view, maskRect, true
}

// swCreateFilteredMask rasterizes the shape to an A8 mask on the CPU, runs the filter, and uploads the result,
// memoizing it in the thread-safe cache. Works for any mask filter (blur or otherwise). The uploaded view is a
// lazy-upload proxy that regenerates from the captured A8 data, so caching it matches the recording-context lane the
// analytic-blur profile/mask textures use.
func swCreateFilteredMask(ctx *DirectContext, viewMatrix *geom.Matrix, shape *StyledShape, filter maskfilter.MaskFilter, unclippedDevShapeBounds, clipBounds geom.IRect, key *gpu.UniqueKey) (view SurfaceProxyView, drawRect geom.IRect, ok bool) {
	threadSafeCache := ctx.ThreadSafeCache()
	if key.IsValid() {
		if cachedView, data := threadSafeCache.FindWithData(key); cachedView.IsValid() {
			drawRect = extractDrawRectFromData(data, unclippedDevShapeBounds)
			return cachedView, drawRect, true
		}
	}

	doFill := !shape.Style().IsSimpleHairline()

	// DrawToMask consumes devPath synchronously (it rasterizes into the mask), so the transient device-space path is
	// pooled and returned.
	devPath := path.Borrow()
	defer path.Recycle(devPath)
	shape.AsPath().TransformTo(viewMatrix, devPath)

	srcM, ok := maskfilter.DrawToMask(devPath, clipBounds, filter, viewMatrix, doFill)
	if !ok {
		return SurfaceProxyView{}, geom.IRect{}, false
	}
	dstM, _, ok := filter.FilterMask(srcM, viewMatrix)
	if !ok {
		return SurfaceProxyView{}, geom.IRect{}, false
	}
	if clipBoundsQuickReject(clipBounds, dstM.Bounds) {
		return SurfaceProxyView{}, geom.IRect{}, false
	}
	if dstM.Bounds.IsEmpty() || len(dstM.Image) == 0 {
		return SurfaceProxyView{}, geom.IRect{}, false
	}

	view = a8DataToTextureView(ctx, dstM.Image, int(dstM.RowBytes), dstM.Bounds.Size(),
		"SW Filtered Mask")
	if view.Proxy() == nil {
		return SurfaceProxyView{}, geom.IRect{}, false
	}
	drawRect = dstM.Bounds

	if key.IsValid() {
		// The view owns a8DataToTextureView's single construction ref, which addWithData adopts (as the analytic path
		// does). On a key collision addWithData drops the redundant ref and returns the incumbent, whose stored draw
		// rect may differ.
		key.SetCustomData(createMaskData(drawRect, unclippedDevShapeBounds))
		var data []byte
		view, data = threadSafeCache.AddWithData(key, view)
		drawRect = extractDrawRectFromData(data, unclippedDevShapeBounds)
	}
	return view, drawRect, true
}

// rrectIsCircle reports whether rr is an oval rrect whose radii are equal (so its bounds are square). geom.RRect stores
// an oval as RRectOval with radii = half the bounds' dimensions.
func rrectIsCircle(rr geom.RRect) bool {
	return rr.Type == geom.RRectOval && geom.ScalarNearlyEqual(rr.RadiusX, rr.RadiusY)
}

// directFilterMask tries to render the mask filter into the target directly via an analytic blur FP — the rect, circle,
// and simple-circular rrect lanes. Returns true if it drew (consuming the paint); on false the paint is left
// unmodified.
func directFilterMask(sdc *SurfaceDrawContext, mf maskfilter.MaskFilter, clip Clip, paint *Paint, viewMatrix *geom.Matrix, shape *StyledShape) bool {
	bmf, isBlur := mf.(blurMaskFilterQuery)
	if !isBlur {
		return false
	}
	if bmf.Style() != maskfilter.BlurNormal {
		return false
	}
	// TODO: we could handle blurred stroked circles.
	if !shape.Style().IsSimpleFill() {
		return false
	}

	xformedSigma := bmf.ComputeXformedSigma(viewMatrix)
	if gpu.BlurIsEffectivelyIdentity(xformedSigma) {
		sc := shape.clone()
		sdc.DrawShape(clip, paint, gpu.AAYes, viewMatrix, &sc)
		return true
	}

	srcRRect, inverted, isRRect := shape.AsRRect()
	if !isRRect || inverted {
		return false
	}

	devRRect, hasDevRRect := srcRRect.Transform(viewMatrix)
	devRRectIsRect := hasDevRRect && devRRect.Type == geom.RRectRect
	devRRectIsCircle := hasDevRRect && rrectIsCircle(devRRect)

	canBeRect := (srcRRect.Type == geom.RRectRect && viewMatrix.PreservesRightAngles()) ||
		devRRectIsRect
	canBeCircle := (rrectIsCircle(srcRRect) && viewMatrix.IsSimilarity()) || devRRectIsCircle

	ctx := sdc.Context()

	if canBeRect || canBeCircle {
		var fp FragmentProcessor
		if canBeRect {
			var devRectOpt *geom.Rect
			if devRRectIsRect {
				r := devRRect.Rect
				devRectOpt = &r
			}
			fp = MakeRectBlur(ctx, sdc.Caps().ShaderCaps, srcRRect.Rect, devRectOpt, viewMatrix,
				xformedSigma)
		} else {
			var devBounds geom.Rect
			if devRRectIsCircle {
				devBounds = devRRect.Rect
			} else {
				center := viewMatrix.MapPoint(srcRRect.Rect.Center())
				radius := viewMatrix.MapVector(geom.Point{Y: srcRRect.Rect.Width() / 2}).Length()
				devBounds = geom.Rect{
					Left: center.X - radius, Top: center.Y - radius,
					Right: center.X + radius, Bottom: center.Y + radius,
				}
			}
			fp = MakeCircleBlur(ctx, devBounds, xformedSigma)
		}
		if fp == nil {
			return false
		}

		srcProxyRect := srcRRect.Rect
		// Outset the src rect enough to reach pixels within three sigma.
		outsetX := 3 * xformedSigma
		outsetY := 3 * xformedSigma
		if viewMatrix.IsScaleTranslate() {
			outsetX /= abs32(viewMatrix.Get(geom.MScaleX))
			outsetY /= abs32(viewMatrix.Get(geom.MScaleY))
		} else {
			var scale geom.Size
			if !viewMatrix.DecomposeScale(&scale, nil) {
				return false
			}
			outsetX /= scale.Width
			outsetY /= scale.Height
		}
		srcProxyRect = srcProxyRect.Outset(outsetX, outsetY)

		paint.SetCoverageFragmentProcessor(fp)
		sdc.DrawRect(clip, paint, gpu.AANo, viewMatrix, srcProxyRect, nil)
		return true
	}

	// The rrect analytic lane: a simple-circular rrect blurred via a nine-patch mask.
	if !viewMatrix.RectStaysRect() {
		return false
	}
	// Require all corners to share (RadiusX, RadiusY): in the uniform-radii subset this reduces to near-equal radii.
	if !hasDevRRect || !geom.ScalarNearlyEqual(devRRect.RadiusX, devRRect.RadiusY) {
		return false
	}

	fp := MakeRRectBlur(ctx, bmf.Sigma(), xformedSigma, srcRRect, devRRect)
	if fp == nil {
		return false
	}

	if bmf.RespectCTM() { // !ignoreXform: the blur is applied in the CTM's coordinate space.
		srcProxyRect := srcRRect.Rect.Outset(3*bmf.Sigma(), 3*bmf.Sigma())
		paint.SetCoverageFragmentProcessor(fp)
		sdc.DrawRect(clip, paint, gpu.AANo, viewMatrix, srcProxyRect, nil)
	} else { // ignoreXform: blur in device space, filling device pixels via the inverse local matrix.
		inverse, ok := viewMatrix.Invert()
		if !ok {
			return false
		}
		extra := 3 * geom.CeilToScalar(xformedSigma-1.0/6.0)
		proxyBounds := devRRect.Rect.Outset(extra, extra).RoundOut()
		paint.SetCoverageFragmentProcessor(fp)
		sdc.FillPixelsWithLocalMatrix(clip, paint, proxyBounds, &inverse)
	}
	return true
}

// computeKeyAndClipBounds decides whether the mask-filtered mask is cacheable and, if so, builds its unique key and
// lets the cache (rather than the clip) bound the mask so the same key serves every clip. To prevent overloading the
// cache during animations, caching is limited to axis-aligned matrices over shapes with an unstyled key, blur filters
// only, and masks that are >50% visible. ok=false means the shape is entirely clipped out.
func computeKeyAndClipBounds(caps *Caps, viewMatrix *geom.Matrix, inverseFilled bool, mf maskfilter.MaskFilter, shape *StyledShape, unclippedDevShapeBounds, devClipBounds geom.IRect) (maskKey gpu.UniqueKey, boundsForClip geom.IRect, ok bool) {
	boundsForClip = devClipBounds

	bmf, isBlur := mf.(blurMaskFilterQuery)
	// To prevent overloading the cache with entries during animations we limit the cache of masks to cases where the
	// matrix preserves axis alignment (preservesAxisAlignment == rectStaysRect).
	useCache := !inverseFilled && viewMatrix.RectStaysRect() && shape.HasUnstyledKey() && isBlur

	if useCache {
		// canFilterMask's bool return (the small-blur "prefer CPU" hint) is ignored here — only the mask rects matter.
		// The clipped rect determines whether the shape is visible; the unclipped rect gates caching by visibility
		// fraction.
		clippedMaskRect, _ := canFilterMask(mf, unclippedDevShapeBounds, devClipBounds, viewMatrix)
		if clippedMaskRect.IsEmpty() {
			return maskKey, boundsForClip, false
		}
		unClippedMaskRect, _ := canFilterMask(mf, unclippedDevShapeBounds, unclippedDevShapeBounds,
			viewMatrix)

		// Use the cache only if >50% of the filtered mask is visible.
		unclippedWidth := int64(unClippedMaskRect.Width())
		unclippedHeight := int64(unClippedMaskRect.Height())
		unclippedArea := unclippedWidth * unclippedHeight
		clippedArea := int64(clippedMaskRect.Width()) * int64(clippedMaskRect.Height())
		maxTextureSize := int64(caps.MaxTextureSize)
		if unclippedArea > 2*clippedArea || unclippedWidth > maxTextureSize ||
			unclippedHeight > maxTextureSize {
			useCache = false
		} else {
			// Make the clip not affect the mask.
			boundsForClip = unclippedDevShapeBounds
		}
	}

	if useCache {
		builder := gpu.UniqueKeyBuilder(&maskKey, maskFilteredMaskDomain, 7+shape.UnstyledKeySize(),
			"Mask Filtered Masks")
		data := builder.Slice()
		// We require the upper left 2x2 of the matrix to match exactly for a cache hit.
		data[0] = math.Float32bits(viewMatrix.Get(geom.MScaleX))
		data[1] = math.Float32bits(viewMatrix.Get(geom.MScaleY))
		data[2] = math.Float32bits(viewMatrix.Get(geom.MSkewX))
		data[3] = math.Float32bits(viewMatrix.Get(geom.MSkewY))
		// Allow 8 bits each in x and y of subpixel positioning; integer translations reuse. The floor-based fraction
		// matches a scalar-fraction computation after masking (the 1.0 offset for negatives lands above the retained
		// high subpixel byte).
		tx := viewMatrix.Get(geom.MTransX)
		ty := viewMatrix.Get(geom.MTransY)
		fracX := int32((tx-geom.FloorToScalar(tx))*65536) & 0x0000FF00
		fracY := int32((ty-geom.FloorToScalar(ty))*65536) & 0x0000FF00
		// Distinguish between hairline and filled paths; for hairlines include the cap (SW grows hairlines by 0.5 pixel
		// with round and square caps). Stroke-and-fill of hairlines is turned into pure fill by the stroke rec, so this
		// covers all cases.
		var styleBits uint32
		if shape.Style().IsSimpleHairline() {
			styleBits = uint32(shape.Style().Rec().Cap())<<1 | 1
		}
		data[4] = uint32(fracX) | uint32(fracY)>>8 | styleBits<<16
		data[5] = uint32(bmf.Style())
		data[6] = math.Float32bits(bmf.Sigma())
		shape.WriteUnstyledKey(data[7:])
		builder.Finish()
	}

	return maskKey, boundsForClip, true
}

// maskFilterLane identifies which lane a mask-filtered draw took. The exported entry point discards it; the live tests
// (through export_test.go) assert the lane a given shape/filter combination actually exercises, since the analytic
// direct lane silently absorbs many shapes the mask-generation lanes would otherwise handle.
type maskFilterLane uint8

const (
	// maskFilterLaneNone means nothing was drawn: an empty or entirely clipped-out shape, or both mask lanes declined.
	maskFilterLaneNone maskFilterLane = iota
	// maskFilterLaneDirect means directFilterMask drew the shape through an analytic blur FP.
	maskFilterLaneDirect
	// maskFilterLaneHW means the mask was generated and blurred on the GPU (hwCreateFilteredMask).
	maskFilterLaneHW
	// maskFilterLaneSW means the filtered mask was rasterized on the CPU and uploaded (swCreateFilteredMask).
	maskFilterLaneSW
)

// DrawShapeWithMaskFilter styles the shape, then draws it filtered through the GPU mask lane (large blurs) or the CPU
// mask lane (small blurs and non-blur filters). The paint is consumed.
func DrawShapeWithMaskFilter(sdc *SurfaceDrawContext, clip Clip, paint *Paint, viewMatrix *geom.Matrix, mf maskfilter.MaskFilter, origShape *StyledShape) {
	drawShapeWithMaskFilter(sdc, clip, paint, viewMatrix, mf, origShape)
}

// drawShapeWithMaskFilter is DrawShapeWithMaskFilter's implementation, reporting the lane it took.
func drawShapeWithMaskFilter(sdc *SurfaceDrawContext, clip Clip, paint *Paint, viewMatrix *geom.Matrix, mf maskfilter.MaskFilter, origShape *StyledShape) maskFilterLane {
	shape := origShape
	if origShape.Style().Applies() {
		styleScale := MatrixToScaleFactor(viewMatrix)
		if styleScale == 0 {
			return maskFilterLaneNone
		}
		tmpShape := origShape.ApplyStyle(StyleApplyPathEffectAndStrokeRec, styleScale)
		if tmpShape.IsEmpty() {
			return maskFilterLaneNone
		}
		shape = &tmpShape
	}

	// Try to draw the mask filter directly through an analytic blur FP (rect/circle/simple-circular rrect profiles).
	// Shapes that do not qualify fall through to the general filtered-mask path.
	if directFilterMask(sdc, mf, clip, paint, viewMatrix, shape) {
		// The mask filter drew itself directly, so there's nothing left to do.
		return maskFilterLaneDirect
	}

	// If the path is hairline, ignore inverse fill.
	_, isHairline := isStrokeHairlineOrEquivalent(shape.Style(), viewMatrix)
	inverseFilled := shape.InverseFilled() && !isHairline

	unclippedDevShapeBounds, devClipBounds, ok := blurShapeAndClipBounds(sdc, clip, shape, viewMatrix)
	if !ok && !inverseFilled {
		return maskFilterLaneNone
	}

	maskKey, boundsForClip, ok := computeKeyAndClipBounds(sdc.Caps(), viewMatrix, inverseFilled, mf,
		shape, unclippedDevShapeBounds, devClipBounds)
	if !ok {
		return maskFilterLaneNone // 'shape' was entirely clipped out
	}

	// GPU-filtered mask (direct context only; there is no DDL recording thread in the port).
	hwView, maskRect, hwOK := hwCreateFilteredMask(sdc.Context(), sdc, viewMatrix, shape, mf,
		unclippedDevShapeBounds, boundsForClip, &maskKey)
	if hwOK {
		if drawMask(sdc, clip, viewMatrix, maskRect, ClonePaint(paint), hwView) {
			return maskFilterLaneHW
		}
	}

	// Either the HW mask lane declined (small blur / non-blur filter) or it failed.
	swView, drawRect, swOK := swCreateFilteredMask(sdc.Context(), viewMatrix, shape, mf,
		unclippedDevShapeBounds, boundsForClip, &maskKey)
	if swOK && drawMask(sdc, clip, viewMatrix, drawRect, paint, swView) {
		return maskFilterLaneSW
	}
	return maskFilterLaneNone
}
