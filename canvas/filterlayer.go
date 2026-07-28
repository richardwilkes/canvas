// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The image-filter layer machinery: the layer mapping/bounds computation that decomposes the CTM per the filter DAG's
// matrix capability (getLayerMappingAndBounds), the evaluation driver that snaps a device into a filtercore source and
// draws the filtered result (internalDrawDeviceWithFilter), the image-filter-to-color-filter collapse, and the
// automatic layer wrapping applied to draws whose paint carries an image filter (aboutToDraw). Plus the
// filtercore.Device/filtercore.Backend adapters over BitmapDevice. The public surface exposes no backdrop filters,
// layer flags, filter spans, or coverage layers, so those lanes reduce away.

package canvas

import (
	"math"

	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

///////////////////////////////////////////////////////////////////////////////
// filtercore adapters

// bitmapFilterDevice adapts BitmapDevice to filtercore.Device.
type bitmapFilterDevice struct {
	dev *BitmapDevice
}

func (f *bitmapFilterDevice) Size() geom.ISize {
	return geom.ISize{Width: f.dev.pix.Width, Height: f.dev.pix.Height}
}

func (f *bitmapFilterDevice) LocalToDevice() geom.Matrix { return f.dev.localToDevice }

func (f *bitmapFilterDevice) SetLocalToDevice(m geom.Matrix) { f.dev.localToDevice = m }

func (f *bitmapFilterDevice) DevClipBounds() geom.IRect { return f.dev.DevClipBounds() }

func (f *bitmapFilterDevice) PushClipStack() { f.dev.PushClipStack() }

func (f *bitmapFilterDevice) PopClipStack() { f.dev.PopClipStack() }

func (f *bitmapFilterDevice) ClipRect(r geom.Rect, aa bool) {
	f.dev.ClipRect(r, raster.ClipIntersect, aa)
}

func (f *bitmapFilterDevice) ClipIRect(r geom.IRect) {
	identity := geom.IdentityMatrix()
	f.dev.rcStack.ClipRect(&identity, r.ToRect(), raster.ClipIntersect, false)
}

// paintFromParams converts filtercore's paint slice to a canvas Paint. When a shader is present the default
// opaque-black paint color is kept so the shader's output is unmodulated.
func paintFromParams(pp *filtercore.PaintParams) Paint {
	p := *NewPaint()
	p.Style = StyleFill
	p.Shader = pp.Shader
	p.ColorFilter = pp.ColorFilter
	if pp.Shader == nil {
		p.Color = pp.Color.ToColor()
	}
	p.BlendMode = pp.Blend
	p.AntiAlias = pp.AntiAlias
	p.Dither = pp.Dither
	return p
}

// PaintFromFilterParams converts filtercore's paint slice to a canvas Paint, exported for the GPU filtercore.Device
// adapter, which lowers filter draws through the GPU device's draw entry points exactly as the raster adapter does
// through the bitmap device's.
func PaintFromFilterParams(pp *filtercore.PaintParams) *Paint {
	p := paintFromParams(pp)
	return &p
}

func (f *bitmapFilterDevice) DrawPaint(pp *filtercore.PaintParams) {
	p := paintFromParams(pp)
	f.dev.DrawPaint(&p)
}

func (f *bitmapFilterDevice) DrawRect(r geom.Rect, pp *filtercore.PaintParams) {
	p := paintFromParams(pp)
	f.dev.DrawRect(r, &p)
}

func (f *bitmapFilterDevice) DrawSpecial(img *filtercore.SpecialImage, deviceMatrix geom.Matrix, sampling shaders.SamplingOptions, pp *filtercore.PaintParams) {
	p := paintFromParams(pp)
	f.dev.DrawSpecial(img, deviceMatrix, sampling, &p)
}

func (f *bitmapFilterDevice) DrawImageRect(img imagecore.DrawableImage, src, dst geom.Rect, sampling shaders.SamplingOptions, pp *filtercore.PaintParams) {
	// The raster device draws CPU pixels; resolve a texture backing (identity for raster images).
	ci := img.MakeNonTextureImage()
	if ci == nil {
		return
	}
	p := paintFromParams(pp)
	f.dev.DrawImageRect(ci, &src, dst, sampling, &p, ConstraintStrict)
}

func (f *bitmapFilterDevice) SnapSpecial(subset geom.IRect) *filtercore.SpecialImage {
	return f.dev.SnapSpecial(subset)
}

// rasterFilterBackend is the CPU implementation of filtercore.Backend (color type pinned to N32, the library's only
// surface format).
type rasterFilterBackend struct {
	cache *filtercore.FilterCache
}

func newRasterFilterBackend() filtercore.Backend {
	return &rasterFilterBackend{cache: filtercore.NewFilterCache()}
}

// NewRasterFilterBackend returns the CPU raster image-filtering backend (filters evaluated on N32 raster
// intermediates). The GPU device uses it for the readback→CPU→upload image-filter fallback: the whole filter DAG runs
// on CPU raster devices, with the GPU device snapping its source layer via readback (SnapSpecial) and uploading the
// filtered result on the final draw (DrawSpecial). GPU-evaluable DAGs bypass this entirely.
func NewRasterFilterBackend() filtercore.Backend { return newRasterFilterBackend() }

// maxFilterDeviceArea guards intermediate surface allocation: Go cannot report a failed allocation, so absurd sizes are
// rejected up front.
const maxFilterDeviceArea = int64(1) << 28 // pixels

func (b *rasterFilterBackend) MakeDevice(size geom.ISize) filtercore.Device {
	if size.IsEmpty() || size.Area() > maxFilterDeviceArea {
		return nil
	}
	return NewBitmapDevice(raster.NewPixmap(size.Width, size.Height)).AsFilterDevice()
}

// MakeImage wraps a drawable image's subset as a special image; lazy images decode on first use here, and a
// texture-backed image resolves to CPU pixels (the fallback's single readback for a GPU source, identity for a raster
// one).
func (b *rasterFilterBackend) MakeImage(subset geom.IRect, img imagecore.DrawableImage) *filtercore.SpecialImage {
	ci := img.MakeNonTextureImage()
	if ci == nil {
		return nil
	}
	px := ci.PeekPixels(imagecore.CachingAllow)
	if px == nil {
		return nil
	}
	return filtercore.NewSpecialImage(subset, px)
}

func (b *rasterFilterBackend) BlurEngine() filtercore.BlurEngine {
	return filtercore.RasterBlurEngine()
}

func (b *rasterFilterBackend) Cache() *filtercore.FilterCache { return b.cache }

///////////////////////////////////////////////////////////////////////////////
// layer mapping

// computeDecompositionCenter picks the local-space point the CTM decomposition pivots around: the center of the content
// bounds when known, else the mapped center of the target output.
func computeDecompositionCenter(dstToLocal *geom.Matrix, contentBounds *geom.Rect, targetOutput geom.IRect) geom.Point {
	var r geom.Rect
	if contentBounds != nil {
		r = *contentBounds
	} else {
		r = targetOutput.ToRect()
	}
	center := r.Center()
	if contentBounds == nil {
		// A homogeneous coordinate behind W=0 is handled later in Mapping.DecomposeCTM.
		center = dstToLocal.MapPoint(center)
	}
	return center
}

// pin64To32 clamps an int64 to the int32 range.
func pin64To32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < math.MinInt32 {
		return math.MinInt32
	}
	return int32(v)
}

// getLayerMappingAndBounds computes the layer's coordinate mapping and bounds for the reachable single-filter form (no
// backdrop scale factor). ok=false means the layer should be skipped.
func getLayerMappingAndBounds(filter filtercore.Filter, localToDst *geom.Matrix, targetOutput geom.IRect, contentBounds *geom.Rect) (filtercore.Mapping, geom.IRect, bool) {
	var mapping filtercore.Mapping
	dstToLocal, invertible := localToDst.Invert()
	if !localToDst.IsFinite() || !invertible {
		return mapping, geom.IRect{}, false
	}

	center := computeDecompositionCenter(&dstToLocal, contentBounds, targetOutput)

	capability := filtercore.MatrixCapabilityComplex
	if filter != nil {
		capability = filtercore.CTMCapability(filter)
	}
	if !mapping.DecomposeCTM(localToDst, capability, center) {
		return mapping, geom.IRect{}, false
	}

	// A reasonable upper limit so perspective/skew transforms don't trigger excessive allocations.
	const minDimThreshold = 2048
	maxLayerDim := max(pin64To32(2*max(targetOutput.Width64(), targetOutput.Height64())),
		int32(minDimThreshold))

	baseLayerBounds := mapping.DeviceToLayerIRect(targetOutput)
	if contentBounds != nil {
		// User bounds act as a hard clip on the layer's extent (the CSS filter-region feature).
		knownBounds := filtercore.RoundOut(mapping.ParamToLayerRect(*contentBounds))
		if !baseLayerBounds.Intersect(knownBounds) {
			baseLayerBounds = geom.IRect{}
		}
	}

	var layerBounds geom.IRect
	if filter != nil {
		layerBounds = filtercore.GetInputBounds(filter, &mapping, targetOutput, contentBounds)
		// The layer may exceed the default max due to required filter inputs (e.g. a displacement map with a large
		// radius).
		if layerBounds.Width() > maxLayerDim || layerBounds.Height() > maxLayerDim {
			idealMapping := filtercore.MappingOf(mapping.LayerMatrix())
			idealLayerBounds := filtercore.GetInputBounds(filter, &idealMapping, targetOutput, contentBounds)
			maxLayerDim = max(maxLayerDim, idealLayerBounds.Width(), idealLayerBounds.Height())
		}
	} else {
		if baseLayerBounds.IsEmpty() {
			return mapping, geom.IRect{}, false
		}
		layerBounds = baseLayerBounds
	}

	if layerBounds.Width() > maxLayerDim || layerBounds.Height() > maxLayerDim {
		newLayerBounds := geom.IRectWH(min(layerBounds.Width(), maxLayerDim),
			min(layerBounds.Height(), maxLayerDim))
		adjust := rectToRectMatrix(layerBounds.ToRect(), newLayerBounds.ToRect())
		if !mapping.AdjustLayerSpace(&adjust) {
			return mapping, geom.IRect{}, false
		}
		layerBounds = newLayerBounds
	}
	return mapping, layerBounds, true
}

// rectToRectMatrix returns the scale+translate matrix mapping from onto to (callers guarantee non-empty rects).
func rectToRectMatrix(from, to geom.Rect) geom.Matrix {
	var m geom.Matrix
	sx := to.Width() / from.Width()
	sy := to.Height() / from.Height()
	m.SetScaleTranslate(sx, sy, to.Left-from.Left*sx, to.Top-from.Top*sy)
	return m
}

///////////////////////////////////////////////////////////////////////////////
// filtered device draws

// deviceCompat describes whether a source device's layer space is already compatible with the filter's expected
// coordinate space.
type deviceCompat int32

const (
	// compatUnknown: the src device's logical space may not axis-align with the dst's.
	compatUnknown deviceCompat = iota
	// compatYes: the src is compatible with the filter's expected layer coordinate space.
	compatYes
	// compatYesWithPadding: compatible, and prepared with a 1px transparent border.
	compatYesWithPadding
)

// applyAlphaAndColorFilter applies the only paint effects that apply to layers (beyond the image filter itself):
// transparency and color filters.
func applyAlphaAndColorFilter(ctx *filtercore.Context, image *filtercore.FilterResult, paint *Paint) filtercore.FilterResult {
	result := *image
	if paint.Color.A() < 255 {
		if cf := colorfilter.NewBlend(paint.Color, raster.BlendDstIn); cf != nil {
			result = result.ApplyColorFilter(ctx, cf)
		}
	}
	if paint.ColorFilter != nil {
		result = result.ApplyColorFilter(ctx, paint.ColorFilter)
	}
	return result
}

// filterSnapDevice is the subset of Device that internalDrawDeviceWithFilter needs from the filter's *source* layer:
// snapping a source subset into a SpecialImage. BitmapDevice and the GPU device implement it. The PDF device does not
// (it has no pixel snapshot), but PDF is never the source: its image-filter layers render into a raster BitmapDevice,
// which is the snappable source, and the filtered result is drawn back onto the PDF device through its AsFilterDevice
// adapter. The destination side needs only RelativeTransform, a plain Device method, so any device can be the target.
type filterSnapDevice interface {
	Device
	SnapSpecial(subset geom.IRect) *filtercore.SpecialImage
}

// internalDrawDeviceWithFilter evaluates the filter DAG and draws the result onto dst, reduced to the publicly
// reachable lanes: src is a filter-compatible layer being restored (compatYes / compatYesWithPadding), or nil when the
// filter DAG required no input (compatUnknown). Backdrop filters, scale factors, non-decal source tiling, coverage
// layers, and the incompatible-src re-sampling lanes have no public entry points.
func (c *Canvas) internalDrawDeviceWithFilter(src, dst Device, filter filtercore.Filter, paint *Paint, compat deviceCompat) {
	backend := dst.CreateFilterBackend(filter)
	if backend == nil {
		// The device cannot evaluate image filters; the filtered draw is dropped, matching AsFilterDevice's nil
		// contract.
		return
	}
	// The backend lives only for this evaluation, so its intermediate devices are released with it (see releasable).
	defer release(backend)
	outputBounds := dst.DevClipBounds()

	// The filter parameters respect the current matrix, which is not necessarily the local matrix set on src (it has
	// already been popped off the stack).
	var localToSrc geom.Matrix
	var srcDims geom.ISize
	if src != nil {
		g2d := src.GlobalToDevice()
		ctm := c.top().matrix
		localToSrc.SetConcat(&g2d, &ctm)
		srcDims = geom.ISize{Width: src.Width(), Height: src.Height()}
	}

	var mapping filtercore.Mapping
	var requiredInput geom.IRect
	if compat != compatUnknown {
		// Use the relative transform from src to dst and the src's whole image; internalSaveLayer already determined
		// what was necessary. Only the source must be snappable; the destination needs only RelativeTransform (a plain
		// Device method), so a non-pixel target like the PDF device qualifies and draws the filtered result back
		// through AsFilterDevice.
		if _, okSrc := src.(filterSnapDevice); !okSrc {
			return
		}
		mapping = filtercore.NewMapping(src.RelativeTransform(dst), dst.RelativeTransform(src), localToSrc)
		requiredInput = geom.IRectWH(srcDims.Width, srcDims.Height)
	} else {
		// Recompute the image filter mapping by decomposing dst's local-to-device matrix.
		var ok bool
		mapping, _, ok = getLayerMappingAndBounds(filter, dst.LocalToDevice(), outputBounds, nil)
		if !ok {
			return
		}
		if src != nil {
			// Only backdrop filters could reach here with a source; they have no public entry point.
			return
		}
		// Trust the caller that no input was required, but keep the calculated mapping.
		requiredInput = geom.IRect{}
	}

	var stats filtercore.Stats
	ctx := filtercore.NewContext(backend, &mapping, requiredInput, &filtercore.FilterResult{}, &stats)

	var source filtercore.FilterResult
	if src != nil && !requiredInput.IsEmpty() {
		// srcToLayer is the identity for the reachable compat lanes, so snap the available subset directly (the layer
		// is included in the offscreen count).
		ctx.MarkNewSurface()
		availSrc := filtercore.RelevantSubset(geom.IRectWH(srcDims.Width, srcDims.Height), requiredInput,
			shaders.TileDecal)
		srcFS, ok := src.(filterSnapDevice)
		if !ok {
			return
		}
		source = filtercore.NewFilterResult(srcFS.SnapSpecial(availSrc),
			geom.IPoint{X: availSrc.Left, Y: availSrc.Top})
		if compat == compatYesWithPadding {
			// Padding was added when the src device was created; inset so bounds tracking can skip shader-based tiling
			// when possible.
			source = source.InsetForSaveLayer()
		}
		// compat == compatYes leaves source as-is; FilterResult augments sampling as needed.
	}

	ctx = ctx.WithNewDesiredOutput(mapping.DeviceToLayerIRect(outputBounds))
	ctx = ctx.WithNewSource(&source)

	var result filtercore.FilterResult
	if filter != nil {
		result = filtercore.FilterImage(filter, &ctx)
	} else {
		result = source
	}
	result = applyAlphaAndColorFilter(&ctx, &result, paint)

	var blender *raster.BlendMode
	if paint.BlendMode != raster.BlendSrcOver {
		mode := paint.BlendMode
		blender = &mode
	}
	fdev := dst.AsFilterDevice()
	if fdev == nil {
		return
	}
	result.Draw(&ctx, fdev, blender)
}

///////////////////////////////////////////////////////////////////////////////
// AutoLayerForImageFilter

// imageToColorFilter folds an image filter that is equivalent to a color filter into the paint's color filter slot.
// Requires the paint to have an image filter; returns true if the paint was modified.
func imageToColorFilter(paint *Paint) bool {
	// A mask filter changes when the color filter sees coverage, so disable the optimization when both are present.
	if paint.MaskFilter != nil {
		return false
	}
	imgCF, ok := filtercore.AsAColorFilter(paint.ImageFilter)
	if !ok {
		return false
	}
	if paint.ColorFilter != nil {
		// Combine the paint's color filter with the image filter's color filter.
		imgCF = colorfilter.NewCompose(imgCF, paint.ColorFilter)
	}
	paint.ColorFilter = imgCF
	paint.ImageFilter = nil
	return true
}

// aboutToDraw prepares a draw whose paint may carry an image filter, wrapping the draw in an automatic layer when
// needed (mask filters draw directly on the raster device, so no coverage layer). It returns the working paint for the
// draw (possibly modified), a restore func to invoke after the device draw, and ok=false when the draw should be
// skipped.
func (c *Canvas) aboutToDraw(paint *Paint, rawBounds *geom.Rect) (*Paint, func(), bool) {
	if !c.predrawNotify() {
		return nil, nil, false
	}
	if paint.ImageFilter == nil {
		return paint, nil, true
	}

	working := *paint
	if imageToColorFilter(&working) {
		return &working, nil, true
	}

	// The image-filter layer: the restore paint takes the image filter and blending off the original paint; blending
	// applies post-filter (against the real dst, not the layer's transparent dst).
	restorePaint := NewPaint()
	restorePaint.ImageFilter = working.ImageFilter
	restorePaint.BlendMode = working.BlendMode
	working.ImageFilter = nil
	working.BlendMode = raster.BlendSrcOver

	c.saveCount++
	c.internalSaveLayer(rawBounds, restorePaint)
	return &working, func() {
		c.saveCount--
		c.internalRestore()
	}, true
}

// (Color-type selection for filter intermediates reduces to the identity here: everything ≤32bpp upgrades to N32, the
// library's only surface format.)
