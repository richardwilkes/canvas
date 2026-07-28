// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU device behind the canvas device interface — a SurfaceDrawContext plus a ClipStack, canvas.Paint→Paint
// conversion at every draw, and the drawEdgeAAImage texture lanes. Trims: paints with path effects devolve to styled
// paths at the device (the device's style carries no path effect); general paths and points go through the shape funnel
// and the path-renderer chain; mask filters run through DrawShapeWithMaskFilter: large blurs generate and blur the mask
// on the GPU, small blurs and non-blur filters take the CPU DrawToMask+filterMask lane; the analytic direct-filter-mask
// rect/rrect/circle profiles are deferred. Image filters evaluate through glfilter.go; tiled huge-image draws and cubic
// image sampling degrade (cubic → linear).

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stroke"
	"github.com/richardwilkes/canvas/surface"
	"github.com/richardwilkes/canvas/textblob"
)

// Device is the GPU-backed implementation of canvas.Device.
type Device struct {
	sdc            *SurfaceDrawContext
	clipStack      *ClipStack
	localToDevice  geom.Matrix
	deviceToGlobal geom.Matrix
	globalToDevice geom.Matrix
}

var _ canvas.Device = (*Device)(nil)

// NewDevice wraps a SurfaceDrawContext as a canvas device.
func NewDevice(sdc *SurfaceDrawContext) *Device {
	// force_aa_clip: clips are always anti-aliased on multisampled or DMSAA targets.
	forceAA := sdc.NumSamples() > 1 || sdc.alwaysAntialias()
	return &Device{
		sdc:            sdc,
		clipStack:      NewClipStack(geom.IRectSize(sdc.Dimensions()), forceAA),
		localToDevice:  geom.IdentityMatrix(),
		deviceToGlobal: geom.IdentityMatrix(),
		globalToDevice: geom.IdentityMatrix(),
	}
}

// SDC exposes the device's draw context (tests and the surface plumbing).
func (d *Device) SDC() *SurfaceDrawContext { return d.sdc }

// Release drops what the device owns that outlives it on its own: the clip stack's live SW masks, which are cached in the
// resource cache under unique keys (see ClipStack.Release). Go has no destructor to run this, so a device that is used
// and discarded — a saveLayer device from CreateDevice, an image-filter intermediate from filterBackend.MakeDevice — is
// released explicitly by whoever discarded it; canvas.Canvas does so at restore through the optional interface. The
// device's own render target is owned by the SurfaceDrawContext and is not touched here, so already-recorded draws that
// reference it (a layer being composited back at restore) remain valid. Safe to call more than once.
func (d *Device) Release() { d.clipStack.Release() }

// clip returns the device's clip stack as a Clip.
func (d *Device) clip() Clip { return d.clipStack }

// Width implements canvas.Device.
func (d *Device) Width() int32 { return d.sdc.Dimensions().Width }

// Height implements canvas.Device.
func (d *Device) Height() int32 { return d.sdc.Dimensions().Height }

//////////////////////////////////////////////////////////////////////////////
// Coordinate systems (shared with BitmapDevice's device bookkeeping).

// SetGlobalCTM implements canvas.Device.
func (d *Device) SetGlobalCTM(ctm *geom.Matrix) {
	d.localToDevice.SetConcat(&d.globalToDevice, ctm)
}

// LocalToDevice implements canvas.Device.
func (d *Device) LocalToDevice() *geom.Matrix { return &d.localToDevice }

// Origin implements canvas.Device.
func (d *Device) Origin() geom.IPoint {
	return geom.IPoint{
		X: geom.FloorToInt(d.deviceToGlobal.Get(geom.MTransX)),
		Y: geom.FloorToInt(d.deviceToGlobal.Get(geom.MTransY)),
	}
}

// SetDeviceCoordinateSystem implements canvas.Device.
func (d *Device) SetDeviceCoordinateSystem(deviceToGlobal, globalToDevice, localToDevice *geom.Matrix, bufferOrigin geom.IPoint) {
	d.deviceToGlobal = *deviceToGlobal
	d.globalToDevice = *globalToDevice
	d.localToDevice = *localToDevice
	if bufferOrigin.X != 0 || bufferOrigin.Y != 0 {
		ox := float32(bufferOrigin.X)
		oy := float32(bufferOrigin.Y)
		d.deviceToGlobal.PreTranslate(ox, oy)
		d.globalToDevice.PostTranslate(-ox, -oy)
		d.localToDevice.PostTranslate(-ox, -oy)
	}
}

// DeviceToGlobal implements canvas.Device.
func (d *Device) DeviceToGlobal() geom.Matrix { return d.deviceToGlobal }

// GlobalToDevice implements canvas.Device.
func (d *Device) GlobalToDevice() geom.Matrix { return d.globalToDevice }

// AsFilterDevice and CreateFilterBackend (the image-filter hooks) live in glfilter.go.

//////////////////////////////////////////////////////////////////////////////
// Clipping (over ClipStack).

// PushClipStack implements canvas.Device.
func (d *Device) PushClipStack() { d.clipStack.Save() }

// PopClipStack implements canvas.Device.
func (d *Device) PopClipStack() { d.clipStack.Restore() }

// ClipRect implements canvas.Device.
func (d *Device) ClipRect(rect geom.Rect, op raster.ClipOp, aa bool) {
	d.clipStack.ClipRect(&d.localToDevice, rect, gpu.AA(aa), op)
}

// ClipPath implements canvas.Device.
func (d *Device) ClipPath(p *path.Path, op raster.ClipOp, aa bool) {
	d.clipStack.ClipPath(&d.localToDevice, p, gpu.AA(aa), op)
}

// ClipRegion implements canvas.Device: the region is in global coordinates and clips non-AA.
func (d *Device) ClipRegion(rgn *raster.Region, op raster.ClipOp) {
	globalToDevice := d.globalToDevice
	switch {
	case rgn.IsEmpty():
		d.clipStack.ClipRect(&globalToDevice, geom.Rect{}, gpu.AANo, op)
	case rgn.IsRect():
		d.clipStack.ClipRect(&globalToDevice, rgn.Bounds().ToRect(), gpu.AANo, op)
	default:
		// Fills the region's (disjoint) spans as contours, which covers the same pixels. ClipPath clones the path
		// (MakeShapePath), so the transient working path is pooled and returned.
		p := path.Borrow()
		for it := raster.NewRegionIterator(rgn); !it.Done(); it.Next() {
			p.AddRect(it.Rect().ToRect(), geom.DirectionCW)
		}
		d.clipStack.ClipPath(&globalToDevice, p, gpu.AANo, op)
		path.Recycle(p)
	}
}

// IsClipEmpty implements canvas.Device.
func (d *Device) IsClipEmpty() bool { return d.clipStack.ClipState() == ClipStackEmpty }

// IsClipRect implements canvas.Device.
func (d *Device) IsClipRect() bool {
	s := d.clipStack.ClipState()
	return s == ClipStackDeviceRect || s == ClipStackWideOpen
}

// IsClipWideOpen implements canvas.Device.
func (d *Device) IsClipWideOpen() bool { return d.clipStack.ClipState() == ClipStackWideOpen }

// DevClipBounds implements canvas.Device.
func (d *Device) DevClipBounds() geom.IRect { return d.clipStack.GetConservativeBounds() }

//////////////////////////////////////////////////////////////////////////////
// Paint conversion.

// paintParams converts a canvas paint to the paint-conversion parameter set.
func paintParams(paint *canvas.Paint) PaintParams {
	return PaintParams{
		Color:          colorcore.Color4fFromColor(paint.Color),
		BlendMode:      paint.BlendMode,
		Shader:         paint.Shader,
		ColorFilter:    paint.ColorFilter,
		Dither:         paint.Dither,
		HasMaskFilter:  paint.MaskFilter != nil,
		HasImageFilter: paint.ImageFilter != nil,
	}
}

// chooseAA reports the effective AA for this draw: per the paint, or always under DMSAA.
func (d *Device) chooseAA(paint *canvas.Paint) gpu.AA {
	return gpu.AA(paint.AntiAlias || d.sdc.alwaysAntialias())
}

// strokeRecFromPaint builds the SDC style from the paint (the style's stroke rec; the path effect is handled separately
// at the device).
func strokeRecFromPaint(paint *canvas.Paint) stroke.Rec {
	spec := stroke.PaintSpec{
		Style:      stroke.PaintStyle(paint.Style),
		Width:      paint.StrokeWidth,
		MiterLimit: paint.MiterLimit,
		Cap:        stroke.Cap(paint.Cap),
		Join:       stroke.Join(paint.Join),
	}
	return stroke.NewStrokeRecFromPaint(&spec, 1)
}

//////////////////////////////////////////////////////////////////////////////
// Draws.

// DrawPaint implements canvas.Device.
func (d *Device) DrawPaint(paint *canvas.Paint) {
	pp := paintParams(paint)
	gpuPaint, ok := MakePaint(d.sdc, &pp, d.localToDevice)
	if !ok {
		return
	}
	d.sdc.DrawPaint(d.clip(), gpuPaint, &d.localToDevice)
	recyclePaint(gpuPaint)
}

// DrawRect implements canvas.Device.
func (d *Device) DrawRect(rect geom.Rect, paint *canvas.Paint) {
	if paint.MaskFilter != nil || paint.PathEffect != nil {
		p := path.Borrow()
		p.AddRect(rect, geom.DirectionCW)
		d.drawStyledShapeSlow(p, paint)
		path.Recycle(p)
		return
	}
	pp := paintParams(paint)
	gpuPaint, ok := MakePaint(d.sdc, &pp, d.localToDevice)
	if !ok {
		return
	}
	rec := strokeRecFromPaint(paint)
	d.sdc.DrawRect(d.clip(), gpuPaint, d.chooseAA(paint), &d.localToDevice, rect, &rec)
	recyclePaint(gpuPaint)
}

// DrawOval implements canvas.Device.
func (d *Device) DrawOval(oval geom.Rect, paint *canvas.Paint) {
	if paint.MaskFilter != nil || paint.PathEffect != nil {
		p := path.Borrow()
		p.AddOval(oval, geom.DirectionCW)
		d.drawStyledShapeSlow(p, paint)
		path.Recycle(p)
		return
	}
	pp := paintParams(paint)
	gpuPaint, ok := MakePaint(d.sdc, &pp, d.localToDevice)
	if !ok {
		return
	}
	rec := strokeRecFromPaint(paint)
	d.sdc.DrawOval(d.clip(), gpuPaint, d.chooseAA(paint), &d.localToDevice, oval, &rec)
	recyclePaint(gpuPaint)
}

// DrawRRect implements canvas.Device.
func (d *Device) DrawRRect(rrect geom.RRect, paint *canvas.Paint) {
	if paint.MaskFilter != nil || paint.PathEffect != nil {
		p := path.Borrow()
		p.AddRRect(rrect, geom.DirectionCW)
		d.drawStyledShapeSlow(p, paint)
		path.Recycle(p)
		return
	}
	pp := paintParams(paint)
	gpuPaint, ok := MakePaint(d.sdc, &pp, d.localToDevice)
	if !ok {
		return
	}
	rec := strokeRecFromPaint(paint)
	d.sdc.DrawRRect(d.clip(), gpuPaint, d.chooseAA(paint), &d.localToDevice, rrect, &rec)
	recyclePaint(gpuPaint)
}

// DrawArc implements canvas.Device.
func (d *Device) DrawArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, paint *canvas.Paint) {
	if paint.MaskFilter != nil || paint.PathEffect != nil || sweepAngle == 0 || oval.IsEmpty() {
		p := path.CreateDrawArcPath(oval, startAngle, sweepAngle, useCenter,
			paint.Style == canvas.StyleFill && paint.PathEffect == nil)
		d.drawStyledShapeSlow(p, paint)
		return
	}
	pp := paintParams(paint)
	gpuPaint, ok := MakePaint(d.sdc, &pp, d.localToDevice)
	if !ok {
		return
	}
	rec := strokeRecFromPaint(paint)
	d.sdc.DrawArc(d.clip(), gpuPaint, d.chooseAA(paint), &d.localToDevice, oval, startAngle,
		sweepAngle, useCenter, &rec)
	recyclePaint(gpuPaint)
}

// DrawPoints implements canvas.Device: points and lines lower the way a basic raster draw does — round-cap points
// become circles, other caps become squares, lines/polygons become stroked paths, which the hairline and tessellation
// renderers then refine.
func (d *Device) DrawPoints(mode canvas.PointMode, pts []geom.Point, paint *canvas.Paint) {
	if len(pts) == 0 {
		return
	}
	width := paint.StrokeWidth
	switch mode {
	case canvas.PointModePoints:
		if width > 0 && paint.PathEffect == nil && paint.MaskFilter == nil {
			// The width>0 point lanes: circles for round caps, width×width squares otherwise.
			for _, pt := range pts {
				r := geom.Rect{
					Left: pt.X - width/2, Top: pt.Y - width/2,
					Right: pt.X + width/2, Bottom: pt.Y + width/2,
				}
				fillPaint := *paint
				fillPaint.Style = canvas.StyleFill
				if paint.Cap == canvas.StrokeCap(stroke.CapRound) {
					d.DrawOval(r, &fillPaint)
				} else {
					d.DrawRect(r, &fillPaint)
				}
			}
			return
		}
		// Hairline points (and effect/filter carriers) ride the styled-path lane as degenerate segments, which the
		// stroker expands per its cap rules.
		p := path.Borrow()
		for _, pt := range pts {
			p.MoveTo(pt.X, pt.Y)
			p.LineTo(pt.X, pt.Y)
		}
		strokePaint := *paint
		strokePaint.Style = canvas.StyleStroke
		d.drawStyledShapeSlow(p, &strokePaint)
		path.Recycle(p)
	case canvas.PointModeLines, canvas.PointModePolygon:
		strokePaint := *paint
		strokePaint.Style = canvas.StyleStroke
		if mode == canvas.PointModeLines {
			for i := 0; i+1 < len(pts); i += 2 {
				d.drawLine(pts[i], pts[i+1], &strokePaint)
			}
		} else {
			p := path.Borrow()
			p.MoveTo(pts[0].X, pts[0].Y)
			for _, pt := range pts[1:] {
				p.LineTo(pt.X, pt.Y)
			}
			d.drawStyledShapeSlow(p, &strokePaint)
			path.Recycle(p)
		}
	}
}

// drawLine draws a single stroked line segment.
func (d *Device) drawLine(p0, p1 geom.Point, paint *canvas.Paint) {
	if paint.MaskFilter == nil && paint.PathEffect == nil && paint.StrokeWidth > 0 &&
		paint.Cap != canvas.StrokeCap(stroke.CapRound) && paint.AntiAlias {
		pp := paintParams(paint)
		gpuPaint, ok := MakePaint(d.sdc, &pp, d.localToDevice)
		if !ok {
			return
		}
		rec := strokeRecFromPaint(paint)
		d.sdc.DrawStrokedLine(d.clip(), gpuPaint, d.chooseAA(paint), &d.localToDevice,
			[2]geom.Point{p0, p1}, &rec)
		recyclePaint(gpuPaint)
		return
	}
	p := path.Borrow()
	p.MoveTo(p0.X, p0.Y)
	p.LineTo(p1.X, p1.Y)
	d.drawStyledShapeSlow(p, paint)
	path.Recycle(p)
}

// DrawPath implements canvas.Device: general paths ride the shape funnel, which reduces simple shapes back to the
// dedicated ops and otherwise hands off to the path-renderer chain (whose last resort is the SW mask).
func (d *Device) DrawPath(p *path.Path, paint *canvas.Paint) {
	d.drawStyledShapeSlow(p, paint)
}

// drawStyledShapeSlow routes a path + paint through the mask-filter lane or the SDC shape funnel, devolving path
// effects on the CPU first (the device's style carries no path effect).
func (d *Device) drawStyledShapeSlow(p *path.Path, paint *canvas.Paint) {
	pp := paintParams(paint)
	gpuPaint, ok := MakePaint(d.sdc, &pp, d.localToDevice)
	if !ok {
		return
	}
	defer recyclePaint(gpuPaint)

	rec := strokeRecFromPaint(paint)
	src := p
	if paint.PathEffect != nil {
		// Devolve the path effect (and its interaction with the stroke) on the CPU.
		spec := stroke.PaintSpec{
			PathEffect: paint.PathEffect,
			Style:      stroke.PaintStyle(paint.Style),
			Width:      paint.StrokeWidth,
			MiterLimit: paint.MiterLimit,
			Cap:        stroke.Cap(paint.Cap),
			Join:       stroke.Join(paint.Join),
		}
		dst := path.Borrow()
		defer path.Recycle(dst)
		if stroke.FillPathWithPaint(src, &spec, dst, nil, &d.localToDevice) {
			rec = stroke.NewStrokeRec(stroke.InitStyleFill)
		} else {
			rec = stroke.NewStrokeRec(stroke.InitStyleHairline)
		}
		src = dst
	}

	if paint.MaskFilter != nil {
		d.drawShapeWithMaskFilter(src, &rec, gpuPaint, paint.MaskFilter)
		return
	}

	shape := MakeShapePath(src)
	d.sdc.DrawShapeWithStyle(d.clip(), gpuPaint, d.chooseAA(paint), &d.localToDevice, &shape,
		&rec)
}

// drawShapeWithMaskFilter routes a shape + mask filter through DrawShapeWithMaskFilter: large blurs generate and blur
// the A8 mask on the GPU, while small blurs and non-blur filters render the filtered mask on the CPU and upload it. The
// style carried by 'rec' is applied inside DrawShapeWithMaskFilter via the styled shape.
func (d *Device) drawShapeWithMaskFilter(src *path.Path, rec *stroke.Rec, gpuPaint *Paint, mf maskfilter.MaskFilter) {
	shape := MakeStyledShapePath(src, MakeStyle(*rec), DoSimplifyYes)
	DrawShapeWithMaskFilter(d.sdc, d.clip(), gpuPaint, &d.localToDevice, mf, &shape)
}

//////////////////////////////////////////////////////////////////////////////
// Image draws.

var imageProxyKeyDomain = gpu.GenerateUniqueKeyDomain()

// imageAsView resolves a CPU-backed image to a texture proxy view: uploads (and caches by the image's unique ID) the
// image's pixels as a texture. Alpha-only images upload as A8; everything else converts to N32 RGBA premul.
func imageAsView(ctx *DirectContext, img *imagecore.Image) (SurfaceProxyView, gpu.ColorType) {
	dims := geom.ISize{Width: img.Width(), Height: img.Height()}

	var key gpu.UniqueKey
	builder := gpu.UniqueKeyBuilder(&key, imageProxyKeyDomain, 1, "ImageProxy")
	builder.Slice()[0] = img.UniqueID()
	builder.Finish()

	if img.IsAlphaOnly() {
		px := img.PeekPixels(imagecore.CachingAllow)
		if px == nil {
			return SurfaceProxyView{}, 0
		}
		if px.Bytes != nil {
			view := FindOrCreatePixelsProxyView(ctx, &key, dims, gpu.ColorTypeAlpha8, px.Bytes,
				int(px.RowElems), "ImageProxy")
			return view, gpu.ColorTypeAlpha8
		}
	}

	pm := img.Pixmap()
	if pm == nil {
		// Convert non-N32 images through ReadPixels into N32 premul.
		info := img.Info()
		info.ColorType = imagecore.ColorTypeRGBA8888
		info.AlphaType = imagecore.AlphaTypePremul
		buf := make([]byte, int(dims.Width)*int(dims.Height)*4)
		if !img.ReadPixels(info, buf, int(dims.Width)*4, 0, 0, imagecore.CachingAllow) {
			return SurfaceProxyView{}, 0
		}
		view := FindOrCreatePixelsProxyView(ctx, &key, dims, gpu.ColorTypeRGBA8888, buf,
			int(dims.Width)*4, "ImageProxy")
		return view, gpu.ColorTypeRGBA8888
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&pm.Pix[0])), len(pm.Pix)*4)
	view := FindOrCreatePixelsProxyView(ctx, &key, dims, gpu.ColorTypeRGBA8888, bytes,
		int(pm.RowPixels)*4, "ImageProxy")
	return view, gpu.ColorTypeRGBA8888
}

// drawableAsView resolves a polymorphic drawable image to a texture proxy view for drawing on ctx. A texture-backed
// image already living on ctx contributes its own view directly — no readback, no re-upload (the GPU-native lane). Any
// other image — a CPU raster image, or a texture image from a different direct context — is read back to CPU pixels
// (identity for a raster image) and uploaded through the shared image-proxy cache, exactly as imageAsView does. Returns
// an invalid view on a failed readback/upload.
func drawableAsView(ctx *DirectContext, img imagecore.DrawableImage) (SurfaceProxyView, gpu.ColorType) {
	if tex, ok := img.(*TextureImage); ok && tex.Context() == ctx {
		return tex.View(), tex.ColorType()
	}
	rasterImg := img.MakeNonTextureImage()
	if rasterImg == nil {
		return SurfaceProxyView{}, 0
	}
	return imageAsView(ctx, rasterImg)
}

// canUseDrawTexture reports whether the paint/sampling combination qualifies for the fast draw-texture lane.
func canUseDrawTexture(paint *canvas.Paint, sampling shaders.SamplingOptions) bool {
	return paint.ColorFilter == nil && paint.Shader == nil && paint.MaskFilter == nil &&
		paint.ImageFilter == nil && sampling.MaxAniso <= 1 && !sampling.UseCubic &&
		sampling.Mipmap == shaders.MipmapNone
}

// textureColor computes the constant color to modulate a texture draw by.
func textureColor(paintColor colorcore.Color4f, alphaOnly bool) colorcore.PMColor4f {
	if alphaOnly {
		return paintColor.Premul()
	}
	a := min(max(paintColor.A, 0), 1)
	return colorcore.PMColor4f{R: a, G: a, B: a, A: a}
}

// downgradeToFilter reports the filter mode to use: if the sampling was "fancier" than bilerp, only do bilerp.
func downgradeToFilter(sampling shaders.SamplingOptions) gpu.FilterMode {
	if sampling.MaxAniso > 1 || sampling.UseCubic ||
		sampling.Mipmap != shaders.MipmapNone {
		return gpu.FilterModeLinear
	}
	if sampling.Filter == shaders.FilterLinear {
		return gpu.FilterModeLinear
	}
	return gpu.FilterModeNearest
}

// DrawImageRect implements canvas.Device, following the drawImageQuadDirect → drawEdgeAAImage flow for the clamp tile
// mode (image filters were already handled by the canvas).
func (d *Device) DrawImageRect(img imagecore.DrawableImage, src *geom.Rect, dst geom.Rect, sampling shaders.SamplingOptions, paint *canvas.Paint, constraint canvas.SrcRectConstraint) {
	if img == nil || dst.IsEmpty() {
		return
	}
	imageBounds := geom.RectWH(float32(img.Width()), float32(img.Height()))
	srcRect := imageBounds
	if src != nil {
		srcRect = *src
	}
	if srcRect.IsEmpty() {
		return
	}

	// Clip the src rect to the image bounds and adjust dst to match.
	srcToDst := rectToRectFillMatrix(srcRect, dst)
	clippedSrc := srcRect
	if !clippedSrc.Intersect(imageBounds) {
		return
	}
	clippedDst, _ := srcToDst.MapRect(clippedSrc)

	// chooseAA, not paint.AntiAlias: a DMSAA surface antialiases every draw, and DrawFilledQuad's
	// canUseDynamicMSAA && aa == AANo special case relies on the device layer never handing it AANo there.
	aaFlags := gpu.QuadAAFlagsNone
	if d.chooseAA(paint) {
		aaFlags = gpu.QuadAAFlagsAll
	}
	glConstraint := SrcRectConstraintFast
	if constraint == canvas.ConstraintStrict {
		glConstraint = SrcRectConstraintStrict
	}

	view, _ := drawableAsView(d.sdc.Context(), img)
	if view.Proxy() == nil {
		return
	}
	alphaType := gpu.AlphaTypePremul
	if img.AlphaType() == imagecore.AlphaTypeOpaque {
		alphaType = gpu.AlphaTypeOpaque
	}

	if canUseDrawTexture(paint, sampling) {
		// The fast lane through TextureOp.
		if img.IsAlphaOnly() {
			view.ConcatSwizzle(gpu.MakeSwizzle("aaaa"))
		}
		filter := gpu.FilterModeNearest
		if sampling.Filter == shaders.FilterLinear {
			filter = gpu.FilterModeLinear
		}
		color := textureColor(colorcore.Color4fFromColor(paint.Color), img.IsAlphaOnly())
		d.sdc.DrawTexture(d.clip(), view, alphaType, filter, gpu.MipmapModeNone,
			paint.BlendMode, color, clippedSrc, clippedDst, aaFlags, glConstraint,
			&d.localToDevice)
		return
	}

	// The general lane (drawEdgeAAImage below the draw_texture check): a texture FP combined with the paint via the
	// replace-shader conversion. Cubic/aniso/mipmap sampling degrades to linear; the shader is only consulted for
	// alpha-only images, exactly as use_shader does.
	mfLane := paint.MaskFilter != nil
	canUseTextureCoordsAsLocalCoords := (!img.IsAlphaOnly() || paint.Shader == nil) && !mfLane

	restrictToSubset := glConstraint == SrcRectConstraintStrict
	coordsAllInsideSrcRect := aaFlags == gpu.QuadAAFlagsNone && !mfLane

	textureMatrix := geom.IdentityMatrix()
	if !canUseTextureCoordsAsLocalCoords {
		inv, ok := srcToDst.Invert()
		if !ok {
			return
		}
		textureMatrix = inv
	}

	filter := downgradeToFilter(sampling)
	samplerState := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, filter,
		gpu.MipmapModeNone)
	var fp FragmentProcessor
	switch {
	case restrictToSubset && coordsAllInsideSrcRect:
		fp = MakeTextureEffectSubsetWithDomain(view, alphaType, textureMatrix, samplerState,
			clippedSrc, clippedSrc, d.sdc.Caps(), [4]float32{})
	case restrictToSubset:
		fp = MakeTextureEffectSubset(view, alphaType, textureMatrix, samplerState, clippedSrc,
			d.sdc.Caps(), [4]float32{}, false)
	default:
		fp = MakeTextureEffect(view, alphaType, textureMatrix, samplerState, d.sdc.Caps(),
			[4]float32{})
	}
	if img.IsAlphaOnly() {
		if paint.Shader != nil {
			mRec := shaders.NewMatrixRec(d.localToDevice)
			shaderFP := MakeShaderFP(paint.Shader, MakeFPArgs(d.sdc), &mRec)
			if shaderFP == nil {
				return
			}
			fp = BlendFP(fp, shaderFP, raster.BlendDstIn)
		} else {
			// Multiply the input (paint) color by the texture (alpha).
			fp = MulInputByChildAlphaFP(fp)
		}
	}

	pp := paintParams(paint)
	gpuPaint, ok := MakePaintReplaceShader(d.sdc, &pp, d.localToDevice, fp)
	if !ok {
		return
	}
	defer recyclePaint(gpuPaint)

	if mfLane {
		// Draw the mask-filtered dst rect; this loses per-edge AA, which is not noticeable since the mask filter is
		// probably a blur.
		fillRec := stroke.NewStrokeRec(stroke.InitStyleFill)
		p := path.Borrow()
		p.AddRect(clippedDst, geom.DirectionCW)
		d.drawShapeWithMaskFilter(p, &fillRec, gpuPaint, paint.MaskFilter)
		path.Recycle(p)
		return
	}

	var localRect *geom.Rect
	if canUseTextureCoordsAsLocalCoords {
		localRect = &clippedSrc
	}
	d.sdc.FillRectWithEdgeAA(d.clip(), gpuPaint, aaFlags, &d.localToDevice, clippedDst,
		localRect)
}

// DrawAtlas implements canvas.Device: the paint — whose shader is already the atlas image shader — converts through
// MakePaintWithBlend when per-sprite colors are present (the geometry processor supplies them as the primitive color,
// blended with the shader by mode) and plain MakePaint otherwise, then draws through SurfaceDrawContext.DrawAtlas →
// DrawAtlasOp.
func (d *Device) DrawAtlas(xforms []geom.RSXform, tex []geom.Rect, colors []colorcore.Color, mode raster.BlendMode, paint *canvas.Paint) {
	pp := paintParams(paint)
	var gpuPaint *Paint
	var ok bool
	if len(colors) != 0 {
		gpuPaint, ok = MakePaintWithBlend(d.sdc, &pp, d.localToDevice, mode)
	} else {
		gpuPaint, ok = MakePaint(d.sdc, &pp, d.localToDevice)
	}
	if !ok {
		return
	}
	d.sdc.DrawAtlas(d.clip(), gpuPaint, &d.localToDevice, xforms, tex, colors)
	recyclePaint(gpuPaint)
}

//////////////////////////////////////////////////////////////////////////////
// Layers.

// clearAll clears the device to transparent black.
func (d *Device) clearAll() {
	dims := d.sdc.Dimensions()
	d.sdc.ClearAtLeast(geom.IRect{Right: dims.Width, Bottom: dims.Height}, [4]float32{})
}

// CreateDevice implements canvas.Device: creates a compatible offscreen render target for a layer. The layer paint is
// unused: the GPU device is itself a valid image-filter render target (its source is snapped via GPU→CPU readback when
// a DAG falls back to the CPU), so unlike the PDF device it never needs to inspect the filter to pick a raster fallback
// backing.
func (d *Device) CreateDevice(width, height int32, _ *canvas.Paint) canvas.Device {
	// Clone the parent's surface props with the layer's pixel geometry — always "unknown" here (surface props preserve
	// geometry only for opaque LCD-text layers, E.4) — so a DMSAA parent gets DMSAA layers.
	props := d.sdc.SurfaceProps()
	props.PixelGeometry = surface.PixelGeometryUnknown
	sdc := MakeSurfaceDrawContextWithProps(d.sdc.Context(), d.sdc.ColorType(),
		geom.ISize{Width: width, Height: height}, gpu.BackingFitApprox, d.sdc.NumSamples(),
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "DeviceLayer", &props)
	if sdc == nil {
		return nil
	}
	dev := NewDevice(sdc)
	// The convention is to only clear a device if it is non-opaque. Every reachable layer is premul RGBA, so the clear
	// is unconditional: without it the layer inherits whatever the driver left in the (possibly recycled) texture
	// memory, which leaks through the restore composite wherever the layer's own draws leave partial or no coverage.
	dev.clearAll()
	return dev
}

// DrawDevice implements canvas.Device: composites a layer onto this device at the relative integer translation with the
// paint's alpha, blend, and color filter.
func (d *Device) DrawDevice(src canvas.Device, paint *canvas.Paint) {
	srcDev, ok := src.(*Device)
	if !ok {
		return
	}
	view := srcDev.sdc.ReadSurfaceView()
	if view.Proxy() == nil {
		return
	}
	srcOrigin := srcDev.Origin()
	dstOrigin := d.Origin()
	dims := srcDev.sdc.Dimensions()
	srcRect := geom.RectWH(float32(dims.Width), float32(dims.Height))
	dstRect := srcRect
	dstRect = dstRect.Offset(float32(srcOrigin.X-dstOrigin.X), float32(srcOrigin.Y-dstOrigin.Y))

	identity := geom.IdentityMatrix()
	// Pick the edge AA from chooseAA(paint): all edges when the paint is AA or the surface always antialiases (DMSAA),
	// none otherwise. Under DMSAA this is what routes the composite through drawTexture's FillRRectOp lane instead of
	// escalating the pass.
	edgeAA := gpu.QuadAAFlagsNone
	if d.chooseAA(paint) == gpu.AAYes {
		edgeAA = gpu.QuadAAFlagsAll
	}

	if canUseDrawTexture(paint, shaders.SamplingOptions{}) {
		// The fast lane most restores hit, since there won't be a color filter.
		color := textureColor(colorcore.Color4fFromColor(paint.Color), false)
		d.sdc.DrawTexture(d.clip(), view, gpu.AlphaTypePremul, gpu.FilterModeNearest,
			gpu.MipmapModeNone, paint.BlendMode, color, srcRect, dstRect, edgeAA,
			SrcRectConstraintFast, &identity)
		return
	}

	// The general drawEdgeAAImage lane: a restore paint carrying a color filter (a saveLayer paint whose filter does
	// not affect transparent black takes the trivial-restore path with the filter still on the paint) composites
	// through a texture FP combined with the paint via the replace-shader conversion, exactly as DrawImageRect's
	// general lane. The layer sits at an integer translation of the destination, so texture coords are local coords and
	// the draw happens in device space (identity view matrix).
	samplerState := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp,
		gpu.FilterModeNearest, gpu.MipmapModeNone)
	fp := MakeTextureEffect(view, gpu.AlphaTypePremul, identity, samplerState, d.sdc.Caps(),
		[4]float32{})
	pp := paintParams(paint)
	gpuPaint, ok := MakePaintReplaceShader(d.sdc, &pp, identity, fp)
	if !ok {
		return
	}
	defer recyclePaint(gpuPaint)
	d.sdc.FillRectWithEdgeAA(d.clip(), gpuPaint, edgeAA, &identity, dstRect, &srcRect)
}

// rectToRectFillMatrix returns the scale+translate matrix mapping src onto dst.
func rectToRectFillMatrix(src, dst geom.Rect) geom.Matrix {
	var m geom.Matrix
	if src.IsEmpty() {
		m.SetIdentity()
		return m
	}
	sx := dst.Width() / src.Width()
	sy := dst.Height() / src.Height()
	m.SetScaleTranslate(sx, sy, dst.Left-src.Left*sx, dst.Top-src.Top*sy)
	return m
}

// DrawGlyphRunList implements canvas.Device, delegating to SurfaceDrawContext.DrawGlyphRunList. Some designs route
// glyph-run lists without a blob through a cached "slug" object; slugs are unreachable here, so those lists ride the
// same coordinator path uncached, producing the identical SubRunContainer staging either way.
func (d *Device) DrawGlyphRunList(c *canvas.Canvas, glyphRunList *textblob.GlyphRunList, paint *canvas.Paint) {
	d.sdc.DrawGlyphRunList(c, d.clip(), &d.localToDevice, glyphRunList, paint)
}
