// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Device is the page/layer drawing context that turns canvas draw ops into a PDF content stream. This file covers the
// direct geometry lanes — drawPaint/drawRect/drawOval/drawRRect/drawPath/drawArc/drawPoints, clips
// (rect/path/region/replace), transforms, solid colors, per-paint alpha and the PDF-expressible blend modes — via
// setUpContentEntry over the graphic-stack state and the path emitters, plus the path-effect (FillPathWithPaint) and
// inverse-fill (pathops) lowerings and the /Resources + content() output the document consumes at EndPage.
//
// Image draws (drawImageRect → the direct Image XObject lanes) land via internalDrawImageRect + the image serializer
// (bitmap.go), including the alpha-only luminosity-SMask, mask-filter, perspective, and color-filter sub-lanes.
//
// The saveLayer transparency-group compositing (drawDevice via makeFormXObjectFromDevice) and the advanced-blend-mode
// form-XObject dance in setUp/finishContentEntry (drawFormXObjectWithMask + the two content buffers) landed in the
// compositing slice.
//
// The mask-filter soft-mask lane (internalDrawPathWithFilter → a luminosity SMask over the path shape) and the
// color-filter fold (cleanPaint's removal of the color filter, into the shader or paint color) landed in the
// mask/color-filter slice.
//
// A saveLayer whose paint carries an image or color filter renders on a raster device (createDevice's bitmap-device
// branch) and is drawn back as an Image XObject at restore — the color-filter layer through DrawDevice →
// drawRasterLayerBack, the image-filter layer through the canvas's internalDrawDeviceWithFilter over AsFilterDevice's
// adapter (filterdevice.go). Every other layer stays a PDF device and composites as a transparency-group form XObject.

package pdf

import (
	"math"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/pathops"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stream"
	"github.com/richardwilkes/canvas/stroke"
	"github.com/richardwilkes/canvas/surface"
	"github.com/richardwilkes/canvas/textblob"
)

// Device is the drawing context for a page or layer of PDF content. It implements canvas.Device so a canvas can draw
// into it.
type Device struct {
	shaderResources       map[IndirectReference]struct{}
	clipStack             *ClipStack
	activeStackState      *graphicStackState
	doc                   *Document
	fontResources         map[IndirectReference]struct{}
	graphicStateResources map[IndirectReference]struct{}
	xObjectResources      map[IndirectReference]struct{}
	contentBuffer         stream.MemoryWStream
	content               stream.MemoryWStream
	initialTransform      geom.Matrix
	globalToDevice        geom.Matrix
	deviceToGlobal        geom.Matrix
	localToDevice         geom.Matrix
	pageSize              geom.ISize
	needsExtraSave        bool
}

// newDevice creates a page/layer device: page size in device pixels, a back-pointer to the document, and the page/layer
// initial transform.
func newDevice(pageSize geom.ISize, doc *Document, transform geom.Matrix) *Device {
	return &Device{
		doc:                   doc,
		clipStack:             NewClipStack(),
		pageSize:              pageSize,
		initialTransform:      transform,
		localToDevice:         geom.IdentityMatrix(),
		deviceToGlobal:        geom.IdentityMatrix(),
		globalToDevice:        geom.IdentityMatrix(),
		graphicStateResources: map[IndirectReference]struct{}{},
		xObjectResources:      map[IndirectReference]struct{}{},
		shaderResources:       map[IndirectReference]struct{}{},
		fontResources:         map[IndirectReference]struct{}{},
	}
}

// makeCongruentDevice returns a same-size PDF device with the identity initial transform.
func (d *Device) makeCongruentDevice() *Device {
	return newDevice(d.pageSize, d.doc, geom.IdentityMatrix())
}

func (d *Device) bounds() geom.IRect { return geom.IRectWH(d.pageSize.Width, d.pageSize.Height) }

// InitialTransform returns the page/layer initial transform.
func (d *Device) InitialTransform() geom.Matrix { return d.initialTransform }

// ---- canvas.Device: geometry / coordinate systems ---------------------------------------------------

// Width implements canvas.Device.
func (d *Device) Width() int32 { return d.pageSize.Width }

// Height implements canvas.Device.
func (d *Device) Height() int32 { return d.pageSize.Height }

// SetGlobalCTM implements canvas.Device.
func (d *Device) SetGlobalCTM(ctm *geom.Matrix) { d.localToDevice.SetConcat(&d.globalToDevice, ctm) }

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

// AsFilterDevice implements canvas.Device: the filtercore.Device adapter that draws a filtered layer's resolved raster
// result back onto this PDF device as an Image XObject. The image-filter layer itself renders on a raster bitmap device
// (CreateDevice), so the whole filter DAG evaluates on the CPU raster backend and only the final composite lands here.
func (d *Device) AsFilterDevice() filtercore.Device { return &filterDevice{dev: d} }

// CreateFilterBackend implements canvas.Device: image filters are evaluated on the CPU raster backend (PDF has no
// GPU/native filter path — every intermediate surface is raster and the result is drawn back as an image).
func (d *Device) CreateFilterBackend(_ filtercore.Filter) filtercore.Backend {
	return canvas.NewRasterFilterBackend()
}

// ---- canvas.Device: clips ---------------------------------------------------------------------------

// PushClipStack implements canvas.Device.
func (d *Device) PushClipStack() { d.clipStack.Save() }

// PopClipStack implements canvas.Device.
func (d *Device) PopClipStack() { d.clipStack.Restore() }

// ClipRect implements canvas.Device.
func (d *Device) ClipRect(rect geom.Rect, op raster.ClipOp, aa bool) {
	d.clipStack.ClipRect(rect, &d.localToDevice, op, aa)
}

// ClipPath implements canvas.Device.
func (d *Device) ClipPath(p *path.Path, op raster.ClipOp, aa bool) {
	d.clipStack.ClipPath(p, &d.localToDevice, op, aa)
}

// ClipRegion implements canvas.Device: the region's boundary is clipped as a device-space path, offset by the device
// origin.
func (d *Device) ClipRegion(rgn *raster.Region, op raster.ClipOp) {
	origin := d.Origin()
	boundary := regionBoundaryPath(rgn)
	t := geom.TranslateMatrix(float32(-origin.X), float32(-origin.Y))
	boundary.Transform(&t)
	identity := geom.IdentityMatrix()
	d.clipStack.ClipPath(boundary, &identity, op, false)
}

// regionBoundaryPath builds a device-space path covering the region. The region's constituent rects are disjoint, so a
// winding-filled union of them has the region's exact coverage, which is all a PDF clip needs.
func regionBoundaryPath(rgn *raster.Region) *path.Path {
	p := &path.Path{}
	for it := raster.NewRegionIterator(rgn); !it.Done(); it.Next() {
		p.AddRect(it.Rect().ToRect(), geom.DirectionCW)
	}
	return p
}

// IsClipEmpty implements canvas.Device.
func (d *Device) IsClipEmpty() bool { return d.clipStack.isEmpty(d.bounds()) }

// IsClipWideOpen implements canvas.Device.
func (d *Device) IsClipWideOpen() bool { return d.clipStack.quickContains(d.bounds().ToRect()) }

// IsClipRect implements canvas.Device.
func (d *Device) IsClipRect() bool {
	if d.IsClipWideOpen() {
		return true
	}
	if d.IsClipEmpty() {
		return false
	}
	_, boundType, isIntersectionOfRects := d.clipStack.getBounds()
	return isIntersectionOfRects && boundType == normalBounds
}

// DevClipBounds implements canvas.Device.
func (d *Device) DevClipBounds() geom.IRect { return d.clipStack.bounds(d.bounds()).RoundOut() }

func (d *Device) hasEmptyClip() bool { return d.clipStack.isEmpty(d.bounds()) }

// ---- canvas.Device: draws ---------------------------------------------------------------------------

// DrawPaint implements canvas.Device.
func (d *Device) DrawPaint(paint *canvas.Paint) {
	if d.hasEmptyClip() {
		return
	}
	// Clip is in device space, so the shader (if any) is transformed into device space too.
	bbox := d.clipStack.bounds(d.bounds()).RoundOut().ToRect()
	np := *paint
	np.Style = canvas.StyleFill
	if np.Shader != nil {
		transformShader(&np, &d.localToDevice)
	}
	identity := geom.IdentityMatrix()
	d.internalDrawPath(d.clipStack, &identity, rectPath(bbox), &np, true)
}

// DrawRect implements canvas.Device.
func (d *Device) DrawRect(rect geom.Rect, paint *canvas.Paint) {
	d.internalDrawPath(d.clipStack, &d.localToDevice, rectPath(rect.Sorted()), paint, true)
}

// DrawOval implements canvas.Device.
func (d *Device) DrawOval(oval geom.Rect, paint *canvas.Paint) {
	p := (&path.Path{}).AddOval(oval, geom.DirectionCW)
	d.internalDrawPath(d.clipStack, &d.localToDevice, p, paint, true)
}

// DrawRRect implements canvas.Device.
func (d *Device) DrawRRect(rrect geom.RRect, paint *canvas.Paint) {
	p := (&path.Path{}).AddRRect(rrect, geom.DirectionCW)
	d.internalDrawPath(d.clipStack, &d.localToDevice, p, paint, true)
}

// DrawArc implements canvas.Device via the default path lowering.
func (d *Device) DrawArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, paint *canvas.Paint) {
	isFillNoPathEffect := paint.Style == canvas.StyleFill && paint.PathEffect == nil
	p := path.CreateDrawArcPath(oval, startAngle, sweepAngle, useCenter, isFillNoPathEffect)
	d.internalDrawPath(d.clipStack, &d.localToDevice, p, paint, true)
}

// DrawPath implements canvas.Device.
func (d *Device) DrawPath(p *path.Path, paint *canvas.Paint) {
	d.internalDrawPath(d.clipStack, &d.localToDevice, p, paint, false)
}

// DrawPoints implements canvas.Device for the geometry lanes.
func (d *Device) DrawPoints(mode canvas.PointMode, pts []geom.Point, paint *canvas.Paint) {
	if d.hasEmptyClip() || len(pts) == 0 {
		return
	}
	p := d.cleanPaint(paint)
	if mode != canvas.PointModePoints {
		p.Style = canvas.StyleStroke
	}
	// The path-effect / perspective lane for point drawing is deferred.
	if p.PathEffect != nil || d.localToDevice.HasPerspective() {
		return
	}
	if mode == canvas.PointModePoints && p.Cap != canvas.CapRound {
		if p.StrokeWidth != 0 {
			// PDF can't draw a single square/butt point unambiguously; draw a rect instead.
			p.Style = canvas.StyleFill
			half := p.StrokeWidth * 0.5
			for _, pt := range pts {
				r := geom.Rect{Left: pt.X, Top: pt.Y, Right: pt.X, Bottom: pt.Y}.Outset(half, half)
				d.DrawRect(r, &p)
			}
			return
		}
		p.Cap = canvas.CapRound
	}

	sce := d.newScopedContentEntry(d.clipStack, &d.localToDevice, &p, 0)
	if sce.content == nil {
		return
	}
	defer sce.finish()
	content := sce.content
	switch mode {
	case canvas.PointModePolygon:
		moveTo(pts[0].X, pts[0].Y, content)
		for i := 1; i < len(pts); i++ {
			appendLine(pts[i].X, pts[i].Y, content)
		}
		strokePath(content)
	case canvas.PointModeLines:
		for i := 0; i < len(pts)/2; i++ {
			moveTo(pts[i*2].X, pts[i*2].Y, content)
			appendLine(pts[i*2+1].X, pts[i*2+1].Y, content)
			strokePath(content)
		}
	case canvas.PointModePoints:
		for i := range pts {
			moveTo(pts[i].X, pts[i].Y, content)
			closePath(content)
			strokePath(content)
		}
	}
}

// DrawImageRect implements canvas.Device. Sampling and the src-rect constraint do not affect the emitted Image XObject
// (PDF resamples per the viewer, and /Interpolate is left unset), so they are unused.
func (d *Device) DrawImageRect(img imagecore.DrawableImage, src *geom.Rect, dst geom.Rect, _ shaders.SamplingOptions, paint *canvas.Paint, _ canvas.SrcRectConstraint) {
	if img == nil {
		return
	}
	// The PDF backend serializes CPU pixels (Image XObjects), so a texture-backed drawable is read back through
	// MakeNonTextureImage (identity for a raster image).
	rasterImg := img.MakeNonTextureImage()
	if rasterImg == nil {
		return
	}
	d.internalDrawImageRect(newKeyedImage(rasterImg), src, dst, paint, d.localToDevice)
}

// DrawAtlas implements canvas.Device. PDF has no drawVertices/atlas primitive to lower to, so a PDF DrawAtlas is a
// no-op beyond the empty-clip check, rather than routed through the raster per-sprite lowering.
func (d *Device) DrawAtlas(_ []geom.RSXform, _ []geom.Rect, _ []colorcore.Color, _ raster.BlendMode, _ *canvas.Paint) {
}

// internalDrawImageRect implements the direct Image XObject lanes (src→dst transform, subsetting, the opaque+srcOver
// fast path, sub-pixel clipping, the scaled content entry + drawFormXObject), plus the alpha-only luminosity-SMask lane
// (blend-before-color-filter and the greyscale mask form XObject), the mask-filter lane (image-as-shader through
// internalDrawPath), the perspective rasterization lane, and the color-filter lane (color-filtering into an N32 image).
// The subset is always serialized as-is, with no size comparison against the original; DCT/JPEG encoding and ICC
// profiles remain deferred.
func (d *Device) internalDrawImageRect(imageSubset keyedImage, src *geom.Rect, dst geom.Rect, paint *canvas.Paint, ctm geom.Matrix) {
	if d.hasEmptyClip() || !imageSubset.valid() {
		return
	}

	// Figure out the src→dst transform and subset the image if needed.
	bounds := imageSubset.bounds()
	srcRect := bounds.ToRect()
	if src != nil {
		srcRect = *src
	}
	transform := rectToRect(srcRect, dst)
	if src != nil && *src != bounds.ToRect() {
		if !srcRect.Intersect(bounds.ToRect()) {
			return
		}
		bounds = srcRect.RoundOut()
		transform.PreTranslate(float32(bounds.Left), float32(bounds.Top))
		if bounds != imageSubset.bounds() {
			imageSubset = imageSubset.subset(bounds)
		}
		if !imageSubset.valid() {
			return
		}
	}

	// If the image is opaque and the paint's blend is a fast-path src-over, use src-over (crbug.com/473572).
	p := *paint
	if p.BlendMode != raster.BlendSrcOver && imageSubset.isOpaque() && checkFastPathIsSrcOver(&p) {
		p.BlendMode = raster.BlendSrcOver
	}

	// Alpha-only images need their color from the shader before the color filter applies, so blend the alpha image with
	// the shader/color into an N32 image (the shader's coordinate system is the image's).
	if imageSubset.isAlphaOnly() && p.ColorFilter != nil {
		b := imageSubset.bounds()
		surf := surface.NewRasterN32Premul(b.Width(), b.Height(), nil)
		if surf == nil {
			return
		}
		c := surf.Canvas()
		c.Clear(colorcore.Color(0))
		tmp := canvas.NewPaint()
		tmp.Shader = p.Shader
		tmp.Color = p.Color
		c.DrawImage(imageSubset.image(), 0, 0, shaders.SamplingOptions{}, tmp)
		p.Shader = nil
		snap := surf.MakeImageSnapshot()
		if snap == nil {
			return
		}
		imageSubset = newKeyedImage(snap)
	}

	if imageSubset.isAlphaOnly() {
		// The color filter (if any) applied to the paint above; the alpha layer becomes a luminosity mask.
		mask := alphaImageToGreyscaleImage(imageSubset.image())
		if mask == nil {
			return
		}
		// PDF can't mask vector graphics with an Image XObject, so draw the mask into a congruent device.
		maskDevice := d.makeCongruentDevice()
		mc := canvas.New(maskDevice)
		// This clip keeps the mask from covering the whole device unnecessarily.
		mc.ClipRect(d.clipStack.bounds(d.bounds()), raster.ClipIntersect, false)
		mc.Concat(&ctm)
		if p.MaskFilter != nil {
			tmp := canvas.NewPaint()
			tmp.Shader = shaders.NewImage(mask, shaders.TileClamp, shaders.TileClamp,
				shaders.SamplingOptions{}, &transform)
			tmp.MaskFilter = p.MaskFilter
			mc.DrawRect(dst, tmp)
		} else {
			if src != nil && !isIntegralRect(*src) {
				mc.ClipRect(dst, raster.ClipIntersect, false)
			}
			mc.Concat(&transform)
			mc.DrawImage(mask, 0, 0, shaders.SamplingOptions{}, canvas.NewPaint())
		}
		maskDeviceBounds := maskDevice.clipStack.bounds(maskDevice.bounds()).RoundOut()
		if !ctm.IsIdentity() && p.Shader != nil {
			transformShader(&p, &ctm)
		}
		identity := geom.IdentityMatrix()
		sce := d.newScopedContentEntry(d.clipStack, &identity, &p, 0)
		if sce.content == nil {
			return
		}
		defer sce.finish()
		d.setGraphicState(getSMaskGraphicState(
			maskDevice.makeFormXObjectFromDeviceBounds(maskDeviceBounds, true), false, sMaskLuminosity, d.doc,
		),
			sce.content)
		appendRectangle(d.bounds().ToRect(), sce.content)
		paintPath(styleFill, path.FillWinding, sce.content)
		d.clearMaskOnGraphicState(sce.content)
		return
	}

	if p.MaskFilter != nil {
		// A mask-filtered image draws its rect shape with the image as the shader (transform maps the image into local
		// space; internalDrawPath supplies the CTM). The rect path also handles non-integral clip.
		p.Shader = shaders.NewImage(imageSubset.image(), shaders.TileClamp, shaders.TileClamp,
			shaders.SamplingOptions{}, &transform)
		d.internalDrawPath(d.clipStack, &d.localToDevice, rectPath(dst), &p, true)
		return
	}

	transform.PostConcat(&ctm)
	matrix := transform

	// Rasterize the image with perspective into a new image (PDF has no perspective image transform).
	if matrix.HasPerspective() {
		newImage, newMatrix, ok := d.rasterizePerspectiveImage(imageSubset.image(), transform)
		if !ok {
			return
		}
		imageSubset = newKeyedImage(newImage)
		matrix = newMatrix
	}

	// Color filter: bake it into the image by rendering the image color-filtered into an N32 surface.
	if p.ColorFilter != nil {
		img := colorFilterImage(imageSubset.image(), p.ColorFilter)
		if img == nil {
			return
		}
		imageSubset = newKeyedImage(img)
	}

	// Sub-pixel clipping for non-integral src rects (skbug.com/40035524).
	if src != nil && !isIntegralRect(*src) {
		d.clipStack.Save()
		defer d.clipStack.Restore()
		d.clipStack.ClipRect(dst, &ctm, raster.ClipIntersect, true)
	}

	// Scale the 1x1 image XObject up to WxH, flipping for the PDF (bottom-left, y-up) origin.
	var scaled geom.Matrix
	scaled.SetScale(1, -1)
	scaled.PostTranslate(0, 1)
	subset := imageSubset.bounds()
	scaled.PostScale(float32(subset.Width()), float32(subset.Height()))
	scaled.PostConcat(&matrix)

	sce := d.newScopedContentEntry(d.clipStack, &scaled, &p, 0)
	if sce.content == nil {
		return
	}
	defer sce.finish()
	// The drawing's shape is the image's rect mapped into device space (images have a rectangular shape, which the
	// advanced-blend compositing needs; the scaled matrix carries the extra 1x1→WxH flip).
	shape := rectPath(subset.ToRect())
	shape.Transform(&matrix)
	if sce.needShape() {
		sce.setShape(shape)
	}
	if !sce.needSource() {
		return
	}
	d.drawFormXObject(d.doc.serializeImageCached(imageSubset), sce.content)
}

// rasterizePerspectiveImage handles the perspective case for internalDrawImageRect: render the image through the
// perspective transform into a fresh N32 surface sized to the physical (initial-transform scaled) bounds, and return
// that image plus the axis-aligned matrix that places it. transform is already in device space (post-concatenated with
// the CTM). ok is false when the image is fully clipped out.
func (d *Device) rasterizePerspectiveImage(img *imagecore.Image, transform geom.Matrix) (*imagecore.Image, geom.Matrix, bool) {
	if img == nil {
		return nil, geom.Matrix{}, false
	}
	imageBounds := geom.RectWH(float32(img.Width()), float32(img.Height()))
	outline := rectPath(imageBounds)
	outline.Transform(&transform)
	outlineBounds := outline.Bounds()
	if !outlineBounds.Intersect(d.DevClipBounds().ToRect()) {
		return nil, geom.Matrix{}, false
	}

	// Scale into the final (DPI) space so the rasterized bitmap has enough resolution.
	physicalBounds, _ := d.initialTransform.MapRect(outlineBounds)
	scaleX := physicalBounds.Width() / outlineBounds.Width()
	scaleY := physicalBounds.Height() / outlineBounds.Height()

	surf := surface.NewRasterN32Premul(geom.CeilToInt(physicalBounds.Width()),
		geom.CeilToInt(physicalBounds.Height()), nil)
	if surf == nil {
		return nil, geom.Matrix{}, false
	}
	c := surf.Canvas()
	c.Clear(colorcore.Color(0))

	deltaX := outlineBounds.Left
	deltaY := outlineBounds.Top
	offsetMatrix := transform
	offsetMatrix.PostTranslate(-deltaX, -deltaY)
	offsetMatrix.PostScale(scaleX, scaleY)

	// Fit the shape exactly into the bitmap.
	c.SetMatrix(&offsetMatrix)
	c.DrawImage(img, 0, 0, shaders.SamplingOptions{}, canvas.NewPaint())

	// In the new space, the placement matrix is the identity translated and scaled to reflect DPI.
	var matrix geom.Matrix
	matrix.SetScale(1/scaleX, 1/scaleY)
	matrix.PostTranslate(deltaX, deltaY)

	snap := surf.MakeImageSnapshot()
	if snap == nil {
		return nil, geom.Matrix{}, false
	}
	return snap, matrix, true
}

// DrawGlyphRunList implements canvas.Device: each run is lowered individually. Glyph runs here never carry RSXform or
// cluster text (unison does its own layout).
func (d *Device) DrawGlyphRunList(_ *canvas.Canvas, glyphRunList *textblob.GlyphRunList, paint *canvas.Paint) {
	for i := range glyphRunList.Runs {
		d.internalDrawGlyphRun(&glyphRunList.Runs[i], glyphRunList.Origin, paint)
	}
}

// internalDrawGlyphRun handles the CIDFontType2 lane: filled text in a glyf-outline (TrueType) face becomes a
// BT/Tm/Td/Tj block referencing the embedded font resource. Perspective, mask filters, non-fill/effected paints, and
// non-TrueType (CFF/variable/not-embeddable) faces fall back to drawing the glyph outlines as paths
// (drawGlyphRunAsPath) rather than a Type3 fallback. Since the runs carry no source text, per-cluster grouping and the
// ActualText marked-content machinery are unreachable.
func (d *Device) internalDrawGlyphRun(run *textblob.GlyphRun, offset geom.Point, runPaint *canvas.Paint) {
	glyphIDs := run.Glyphs
	runFont := run.Font
	if len(glyphIDs) == 0 || runFont.Size() <= 0 || d.hasEmptyClip() {
		return
	}
	tf := runFont.Typeface()

	// Mask-filtered and perspective-transformed text draw as paths instead of PDF text operators: viewers mishandle
	// mask-filtered PDF text, and PDF text can't be perspective-transformed.
	if d.localToDevice.HasPerspective() || runPaint.MaskFilter != nil {
		d.drawGlyphRunAsPath(run, offset, runPaint)
		return
	}

	metrics := d.doc.getFontMetrics(tf)
	if metrics == nil {
		return
	}
	fontType := pdfFontType(runPaint.MaskFilter != nil, metrics)
	// Only glyf (TrueType) faces embed as CIDFontType2; stroked/effected text modifies the glyph outline the embedded
	// program can't express. Everything else draws as paths.
	if fontType != font.FontTypeTrueType || runPaint.Style != canvas.StyleFill || runPaint.PathEffect != nil {
		d.drawGlyphRunAsPath(run, offset, runPaint)
		return
	}

	// The size, skewX, and scaleX are applied here.
	textSize := runFont.Size()
	upem := float32(tf.UnitsPerEm())
	advanceScale := textSize * runFont.ScaleX() / upem
	// textScaleX and textScaleY get a conservative glyph bounding box for the per-glyph clip reject.
	textScaleY := textSize / upem
	textScaleX := advanceScale + runFont.SkewX()*textScaleY

	clipStackBounds := d.clipStack.bounds(d.bounds())

	// Clear everything the strike would apply; the glyphs are always filled with the fill color.
	fillPaint := *runPaint
	fillPaint.Style = canvas.StyleFill
	fillPaint.StrokeWidth = 0
	fillPaint.PathEffect = nil
	fillPaint.MaskFilter = nil
	paint := d.cleanPaint(&fillPaint)

	sce := d.newScopedContentEntry(d.clipStack, &d.localToDevice, &paint, runFont.ScaleX())
	if sce.content == nil {
		return
	}
	defer sce.finish()
	content := sce.content

	writeText(content, "BT\n")
	positioner := newGlyphPositioner(content, runFont.SkewX(), offset)
	var f *pdfFont
	numGlyphs := tf.CountGlyphs()

	for i, gid := range glyphIDs {
		if int(gid) >= numGlyphs {
			continue
		}
		xy := run.Positions[i]
		// Do a glyph-by-glyph bounds reject (positions are absolute, so a rejected glyph doesn't disturb the others):
		// scale the font-unit rect, offset by the absolute position, then map through the CTM.
		glyphBounds := getGlyphBoundsDeviceSpace(tf, gid, textScaleX, textScaleY, xy.Add(offset),
			&d.localToDevice)
		if glyphBounds.IsEmpty() {
			if !containsPointInclusive(clipStackBounds, glyphBounds.Left, glyphBounds.Top) {
				continue
			}
		} else if !clipStackBounds.Intersects(glyphBounds) {
			continue // reject glyphs as out of bounds
		}
		if f == nil {
			f = d.doc.getFontResource(tf, fontType)
			positioner.setFont(f)
			writeResourceName(content, resFont, addResource(d.fontResources, f.ref))
			writeText(content, " ")
			writeScalar(content, textSize)
			writeText(content, " Tf\n")
		}
		f.noteGlyphUsage(gid)
		encodedGlyph := f.glyphToPDFFontEncoding(gid)
		advance := advanceScale * tf.DesignAdvance(gid)
		positioner.writeGlyph(encodedGlyph, advance, xy)
	}
	positioner.flush()
	writeText(content, "ET\n")
}

// containsPointInclusive reports whether (x, y) lies within r, inclusive on all four edges (unlike a typical half-open
// rect-contains test); used to keep zero-area (e.g. whitespace) glyphs whose origin lands on the clip boundary.
func containsPointInclusive(r geom.Rect, x, y float32) bool {
	return r.Left <= x && x <= r.Right && r.Top <= y && y <= r.Bottom
}

// getGlyphBoundsDeviceSpace maps a glyph's font-unit design bounds into device space at the given position and CTM.
func getGlyphBoundsDeviceSpace(tf *font.Typeface, gid uint16, xScale, yScale float32, xy geom.Point, ctm *geom.Matrix) geom.Rect {
	r := tf.GlyphDesignBounds(gid)
	// Scale, then offset by the absolute position. xScale carries the font's ScaleX + SkewX, which Font.SetScaleX and
	// Font.SetSkewX accept as arbitrary values, so a negative xScale (and hence an unsorted scaled rect) is reachable;
	// MapRect sorts its result on every branch, so the returned device bounds are sorted regardless.
	scaled := geom.RectLTRB(r.Left*xScale, r.Top*yScale, r.Right*xScale, r.Bottom*yScale).Offset(xy.X, xy.Y)
	devBounds, _ := ctm.MapRect(scaled)
	return devBounds
}

// drawGlyphRunAsPath accumulates every glyph outline (in source space, positioned) into one path and draws it through
// internalDrawPath, so stroke/fill/effects apply exactly as for any geometry. A transparent-glyph re-draw pass for text
// selection is not implemented: it would depend on per-cluster grouping and the ActualText machinery, which is
// unreachable here (no cluster text).
func (d *Device) drawGlyphRunAsPath(run *textblob.GlyphRun, offset geom.Point, runPaint *canvas.Paint) {
	runFont := run.Font
	scalerPaint := runPaint.ScalerPaint()
	spec, strikeToSourceScale := font.MakePathSpec(runFont, &scalerPaint)
	strike := spec.FindOrCreateStrike()
	builder := &path.Path{}
	for i, gid := range run.Glyphs {
		g, action := strike.DigestFor(font.ActionPath, font.PackGlyphID(gid))
		if action != font.GlyphActionAccept {
			continue
		}
		gp := g.Path()
		if gp == nil || gp.IsEmpty() {
			continue
		}
		pos := run.Positions[i]
		var m geom.Matrix
		m.SetScaleTranslate(strikeToSourceScale, strikeToSourceScale, pos.X+offset.X, pos.Y+offset.Y)
		builder.AddPathMatrix(gp, &m, path.AddPathAppend)
	}
	d.internalDrawPath(d.clipStack, &d.localToDevice, builder, runPaint, true)
}

// CreateDevice implements canvas.Device. PDF content cannot express an image filter, and applying a layer color filter
// to a transparency group is likewise not a PDF operator, so a layer whose paint carries either filter renders into a
// raster BitmapDevice; DrawDevice (color-filter layers) and the AsFilterDevice adapter (image-filter layers) draw that
// raster back as an Image XObject at restore. Every other layer stays a PDF device and composites as a
// transparency-group form XObject.
func (d *Device) CreateDevice(width, height int32, layerPaint *canvas.Paint) canvas.Device {
	if layerPaint != nil && (layerPaint.ImageFilter != nil || layerPaint.ColorFilter != nil) {
		return canvas.NewBitmapDevice(raster.NewPixmap(width, height))
	}
	return newDevice(geom.ISize{Width: width, Height: height}, d.doc, geom.IdentityMatrix())
}

// DrawDevice implements canvas.Device: composite a saveLayer's device back into this one. A PDF layer device becomes a
// transparency-group form XObject drawn through a scoped content entry, so the layer paint's alpha, the PDF-expressible
// blend modes, and the advanced-blend form-XObject compositing all apply exactly as for any other draw. A raster layer
// device — CreateDevice's bitmap-device lane for a color-filter layer — is instead drawn back as an Image XObject, so
// the layer paint's color filter, alpha, and blend apply; internalDrawImageRect bakes the color filter into the image.
func (d *Device) DrawDevice(device canvas.Device, paint *canvas.Paint) {
	if bd, ok := device.(*canvas.BitmapDevice); ok {
		d.drawRasterLayerBack(bd, paint)
		return
	}
	pdfDevice, ok := device.(*Device)
	if !ok {
		return
	}
	if pdfDevice.isContentEmpty() {
		return
	}
	matrix := pdfDevice.RelativeTransform(d)
	sce := d.newScopedContentEntry(d.clipStack, &matrix, paint, 0)
	if sce.content == nil {
		return
	}
	defer sce.finish()
	// A device has a rectangular shape: its bounds mapped into this device's space.
	shape := rectPath(geom.IRectWH(pdfDevice.Width(), pdfDevice.Height()).ToRect())
	shape.Transform(&matrix)
	if sce.needShape() {
		sce.setShape(shape)
	}
	if !sce.needSource() {
		return
	}
	d.drawFormXObject(pdfDevice.makeFormXObjectFromDevice(false), sce.content)
}

// RelativeTransform returns the matrix mapping this device's coordinate space to dst's (dst.globalToDevice *
// this.deviceToGlobal).
func (d *Device) RelativeTransform(dst canvas.Device) geom.Matrix {
	g2d := dst.GlobalToDevice()
	var m geom.Matrix
	m.SetConcat(&g2d, &d.deviceToGlobal)
	return m
}

// ---- internal draw plumbing -------------------------------------------------------------------------

// cleanPaint reduces a definitely-opaque Src blend to SrcOver, then folds any color filter into the shader or the paint
// color so the emitted paint never carries one. PDF content is all sRGB, so the fold uses the sRGB working space.
func (d *Device) cleanPaint(src *canvas.Paint) canvas.Paint {
	p := *src
	if p.BlendMode != raster.BlendSrcOver && checkFastPathIsSrcOver(&p) {
		p.BlendMode = raster.BlendSrcOver
	}
	if p.ColorFilter != nil {
		if p.Shader != nil {
			// The color-filter shader modulates the shader color by the paint alpha before applying the filter, so
			// reset the paint to opaque.
			p.Shader = shaders.NewColorFilterShader(p.Shader, float32(p.Color.A())*(1.0/255.0), p.ColorFilter)
			p.Color = p.Color.WithAlpha(0xFF)
		} else {
			// filterColor4f: premul (sRGB unpremul → premul), run the filter, pin alpha, unpremul. Folded to the
			// byte-precision paint color the graphic-state entry emits.
			out, ok := shaders.FilterColor4f(p.ColorFilter, colorcore.Color4fFromColor(p.Color).Premul())
			if !ok {
				out = colorcore.PMColor4f{}
			}
			if out.A < 0 {
				out.A = 0
			} else if out.A > 1 {
				out.A = 1
			}
			p.Color = out.Unpremul().ToColor()
		}
		p.ColorFilter = nil
	}
	return p
}

// transformShader re-expresses the shader in device space by folding the CTM into its local matrix, used when a draw's
// content entry runs at identity.
func transformShader(paint *canvas.Paint, ctm *geom.Matrix) {
	paint.Shader = shaders.NewWithLocalMatrix(paint.Shader, *ctm)
}

// checkFastPathIsSrcOver reports whether the paint's blend mode reduces to SrcOver for the reachable modes.
func checkFastPathIsSrcOver(p *canvas.Paint) bool {
	solid := p.Color.A() == 0xFF && p.ColorFilter == nil && p.Shader == nil
	switch p.BlendMode {
	case raster.BlendSrcOver:
		return true
	case raster.BlendSrc:
		return solid
	default:
		return false
	}
}

// strokeSpecOf builds the stroke.PaintSpec for a paint.
func strokeSpecOf(p *canvas.Paint) stroke.PaintSpec {
	return stroke.PaintSpec{
		PathEffect: p.PathEffect,
		Style:      stroke.PaintStyle(p.Style),
		Width:      p.StrokeWidth,
		MiterLimit: p.MiterLimit,
		Cap:        stroke.Cap(p.Cap),
		Join:       stroke.Join(p.Join),
	}
}

// internalDrawPath draws a path (already lowered from geometry-primitive draw calls) through the clip stack, applying
// the path effect, mask filter, and inverse-fill handling before emitting it into a content entry.
func (d *Device) internalDrawPath(clipStack *ClipStack, ctm *geom.Matrix, origPath *path.Path, srcPaint *canvas.Paint, pathIsMutable bool) {
	if clipStack.isEmpty(d.bounds()) {
		return
	}
	paint := d.cleanPaint(srcPaint)
	pathPtr := origPath

	if paint.MaskFilter != nil {
		d.internalDrawPathWithFilter(clipStack, ctm, origPath, &paint)
		return
	}

	matrix := *ctm

	if paint.PathEffect != nil {
		spec := strokeSpecOf(&paint)
		builder := &path.Path{}
		if stroke.FillPathWithPaint(pathPtr, &spec, builder, nil, &matrix) {
			paint.Style = canvas.StyleFill
		} else {
			paint.Style = canvas.StyleStroke
			paint.StrokeWidth = 0
		}
		pathPtr = builder
		pathIsMutable = true
		paint.PathEffect = nil
	}

	if d.handleInversePath(pathPtr, &paint) {
		return
	}

	if matrix.HasPerspective() {
		transformed := pathPtr
		if !pathIsMutable {
			transformed = pathPtr.Clone()
		}
		transformed.Transform(&matrix)
		pathPtr = transformed
		if paint.Shader != nil {
			transformShader(&paint, &matrix) // the content entry runs at identity below
		}
		matrix = geom.IdentityMatrix()
	}

	sce := d.newScopedContentEntry(clipStack, &matrix, &paint, 0)
	if sce.content == nil {
		return
	}
	defer sce.finish()
	content := sce.content
	const toleranceScale = float32(0.0625) // smaller = better conics (circles)
	matrixScale := matrix.MapRadius(1.0)
	tolerance := toleranceScale
	if matrixScale > 0 {
		tolerance = toleranceScale / matrixScale
	}
	consumeDegenerates := paint.Style == canvas.StyleFill ||
		(paint.Cap != canvas.CapRound && paint.Cap != canvas.CapSquare)
	emitPath(pathPtr, paintStyle(paint.Style), consumeDegenerates, content, tolerance)
	paintPath(paintStyle(paint.Style), pathPtr.FillType(), content)
}

// internalDrawPathWithFilter draws a path with a mask filter: it becomes a luminosity soft mask (the filtered coverage
// rendered into a congruent form XObject) applied over a fill of the path shape. paint carries the mask filter and is
// already cleaned.
func (d *Device) internalDrawPathWithFilter(clipStack *ClipStack, ctm *geom.Matrix, origPath *path.Path, origPaint *canvas.Paint) {
	paint := *origPaint

	// FillPathWithPaint applies the paint's stroke/effect in local space; the result is then moved to device space. A
	// hairline result (doFill == false) renders the mask as a hairline.
	spec := strokeSpecOf(&paint)
	builder := &path.Path{}
	doFill := stroke.FillPathWithPaintResScale(origPath, &spec, builder, nil, 1)
	builder.Transform(ctm)

	bounds := clipStack.bounds(d.bounds()).RoundOut()

	// The mask bounds/margin are computed with an identity filter matrix (the path is already in device space), but the
	// coverage is filtered with the real CTM (a blur's sigma respects the CTM).
	identity := geom.IdentityMatrix()
	sourceMask, ok := maskfilter.DrawToMask(builder, bounds, paint.MaskFilter, &identity, doFill)
	if !ok {
		return
	}
	dstMask, _, ok := paint.MaskFilter.FilterMask(sourceMask, ctm)
	if !ok {
		return
	}
	dstMaskBounds := dstMask.Bounds
	mask := maskToGreyscaleImage(dstMask)
	if mask == nil {
		return
	}

	// PDF can't mask vector graphics with an Image XObject, so draw the mask into a congruent device and use its form
	// XObject as a luminosity soft mask.
	maskDevice := d.makeCongruentDevice()
	mc := canvas.New(maskDevice)
	mc.DrawImage(mask, float32(dstMaskBounds.Left), float32(dstMaskBounds.Top), shaders.SamplingOptions{},
		canvas.NewPaint())

	if !ctm.IsIdentity() && paint.Shader != nil {
		transformShader(&paint, ctm) // the content entry runs at identity below
	}
	sce := d.newScopedContentEntry(clipStack, &identity, &paint, 0)
	if sce.content == nil {
		return
	}
	defer sce.finish()
	d.setGraphicState(getSMaskGraphicState(maskDevice.makeFormXObjectFromDeviceBounds(dstMaskBounds, true),
		false, sMaskLuminosity, d.doc), sce.content)
	appendRectangle(dstMaskBounds.ToRect(), sce.content)
	paintPath(styleFill, builder.FillType(), sce.content)
	d.clearMaskOnGraphicState(sce.content)
}

// handleInversePath draws an inverse-filled path by computing its positive equivalent within the current clip (assumes
// path effects already removed).
func (d *Device) handleInversePath(origPath *path.Path, srcPaint *canvas.Paint) bool {
	if !origPath.IsInverseFillType() {
		return false
	}
	if d.hasEmptyClip() {
		return false
	}
	paint := *srcPaint
	pathPtr := origPath

	if paint.Style == canvas.StyleStroke || paint.Style == canvas.StyleStrokeAndFill {
		spec := strokeSpecOf(&paint)
		builder := &path.Path{}
		doFillPath := stroke.FillPathWithPaintResScale(origPath, &spec, builder, nil, 1)
		pathPtr = builder
		if doFillPath {
			paint.Style = canvas.StyleFill
			paint.StrokeWidth = 0
		} else {
			// Hairline strokes render non-inverted.
			builder.ToggleInverseFillType()
			d.internalDrawPath(d.clipStack, &d.localToDevice, builder, &paint, true)
			return true
		}
	}

	// Clip is in device space, so both the path and the shader are transformed into device space.
	bounds := d.clipStack.bounds(d.bounds())
	modified := pathPtr.Clone()
	modified.Transform(&d.localToDevice)
	inv, ok := pathops.Op(rectPath(bounds), modified, pathops.Intersect)
	if !ok {
		return false
	}
	if paint.Shader != nil {
		transformShader(&paint, &d.localToDevice)
	}
	identity := geom.IdentityMatrix()
	d.internalDrawPath(d.clipStack, &identity, inv, &paint, true)
	return true
}

// treatAsRegularPDFBlendMode reports whether a blend mode is "regular": PDF can express it directly as a /BM blend-mode
// name (blendModeName != ""). Every other mode is composited through form XObjects in
// setUpContentEntry/finishContentEntry.
func treatAsRegularPDFBlendMode(blendMode raster.BlendMode) bool {
	return blendModeName(blendMode) != ""
}

// scopedContentEntry pairs a setUpContentEntry call with the matching finishContentEntry (run by finish, deferred at
// each call site) and carries the state the two share — the blend mode, the saved destination form XObject, and the
// drawing's shape. content is nil when nothing should be drawn.
type scopedContentEntry struct {
	device         *Device
	clipStack      *ClipStack
	content        *stream.MemoryWStream
	shape          *path.Path
	dstFormXObject IndirectReference
	blendMode      raster.BlendMode
}

// newScopedContentEntry sets up a content entry for a draw and returns the scopedContentEntry that tracks it.
func (d *Device) newScopedContentEntry(clipStack *ClipStack, matrix *geom.Matrix, paint *canvas.Paint, textScale float32) *scopedContentEntry {
	sce := &scopedContentEntry{device: d, clipStack: clipStack, blendMode: raster.BlendSrcOver}
	if matrix.HasPerspective() {
		// The perspective content-entry lane is not implemented; nothing is drawn.
		return sce
	}
	sce.blendMode = paint.BlendMode
	sce.content, sce.dstFormXObject = d.setUpContentEntry(clipStack, matrix, paint, textScale)
	return sce
}

// finish completes the content entry (with the drawing's shape, if any) when content was set up.
func (sce *scopedContentEntry) finish() {
	if sce.content == nil {
		return
	}
	shape := sce.shape
	if shape != nil && shape.IsEmpty() {
		shape = nil
	}
	sce.device.finishContentEntry(sce.clipStack, sce.blendMode, sce.dstFormXObject, shape)
}

// needShape reports whether the blend mode's form-XObject compositing masks by the source's shape and so needs it
// explicitly.
func (sce *scopedContentEntry) needShape() bool {
	switch sce.blendMode {
	case raster.BlendClear, raster.BlendSrc, raster.BlendSrcIn, raster.BlendSrcOut, raster.BlendDstIn,
		raster.BlendDstOut, raster.BlendSrcATop, raster.BlendDstATop, raster.BlendModulate:
		return true
	default:
		return false
	}
}

// needSource reports whether the blend mode needs the source at all: everything except Clear (which paints nothing from
// the source) does.
func (sce *scopedContentEntry) needSource() bool { return sce.blendMode != raster.BlendClear }

// setShape records the drawing's shape when it differs from the alpha component of the content (images and devices have
// a rectangular shape).
func (sce *scopedContentEntry) setShape(shape *path.Path) { sce.shape = shape }

// isContentEmpty reports whether both the main content and the compositing buffer are empty.
func (d *Device) isContentEmpty() bool {
	return d.content.BytesWritten() == 0 && d.contentBuffer.BytesWritten() == 0
}

// reset drops the per-form-XObject resources and content. The active graphics stack is already drained by
// content()/contentBytes() beforehand; makeFormXObjectFromDevice calls this after the device's content has been pulled
// into a form XObject.
func (d *Device) reset() {
	d.graphicStateResources = map[IndirectReference]struct{}{}
	d.xObjectResources = map[IndirectReference]struct{}{}
	d.shaderResources = map[IndirectReference]struct{}{}
	d.fontResources = map[IndirectReference]struct{}{}
	d.content.Reset()
	d.activeStackState = nil
}

// setUpContentEntry returns the content stream to draw into (nil when the draw is a no-op) and, for an advanced
// (non-regular, non-DstOver) blend mode over existing content, the destination form XObject that finishContentEntry
// composites against.
func (d *Device) setUpContentEntry(clipStack *ClipStack, matrix *geom.Matrix, paint *canvas.Paint, textScale float32) (*stream.MemoryWStream, IndirectReference) {
	var dst IndirectReference
	blendMode := paint.BlendMode

	// Dst xfer mode doesn't draw source at all.
	if blendMode == raster.BlendDst {
		return nil, dst
	}

	// For the following modes we handle source and destination separately, so capture what's already there as a form
	// XObject.
	if !treatAsRegularPDFBlendMode(blendMode) && blendMode != raster.BlendDstOver {
		if !d.isContentEmpty() {
			dst = d.makeFormXObjectFromDevice(false)
			// isContentEmpty() is now true.
		} else if blendMode != raster.BlendSrc && blendMode != raster.BlendSrcOut {
			// Except for Src and SrcOut, if there isn't anything already there, then we're done.
			return nil, dst
		}
	}
	// Xor and Plus fold to SrcOver/Normal via blendModeName, so treatAsRegularPDFBlendMode treats them as regular.

	var contentStream *stream.MemoryWStream
	if treatAsRegularPDFBlendMode(blendMode) {
		if d.activeStackState == nil {
			if d.content.BytesWritten() != 0 {
				writeText(&d.content, "Q\nq\n")
				d.needsExtraSave = true
			}
			d.activeStackState = newGraphicStackState(&d.content)
		}
		// Otherwise the active stack already targets d.content.
		contentStream = &d.content
	} else {
		if d.activeStackState != nil {
			d.activeStackState.drainStack()
		}
		d.activeStackState = newGraphicStackState(&d.contentBuffer)
		contentStream = &d.contentBuffer
	}
	var entry gsEntry
	d.populateGraphicStateEntry(matrix, clipStack, paint, textScale, &entry)
	d.activeStackState.updateClip(clipStack, d.bounds())
	d.activeStackState.updateMatrix(&entry.matrix)
	d.activeStackState.updateDrawingState(&entry)
	return contentStream, dst
}

// finishContentEntry completes a content entry: for the advanced blend modes it drains the compositing buffer into the
// main content and composites the source (the just-drawn content) and destination (the saved dst form XObject) form
// XObjects through the soft-mask dance the mode requires. Regular modes (and DstOver, apart from the buffer prepend)
// need no compositing.
func (d *Device) finishContentEntry(clipStack *ClipStack, blendMode raster.BlendMode, dst IndirectReference, shape *path.Path) {
	if treatAsRegularPDFBlendMode(blendMode) {
		// dst is unset for regular modes.
		return
	}

	d.activeStackState.drainStack()
	d.activeStackState = nil

	identity := geom.IdentityMatrix()

	if blendMode == raster.BlendDstOver {
		if d.contentBuffer.BytesWritten() != 0 {
			if d.content.BytesWritten() != 0 {
				writeText(&d.contentBuffer, "Q\nq\n")
				d.needsExtraSave = true
			}
			d.contentBuffer.PrependToAndReset(&d.content)
		}
		return
	}
	if d.contentBuffer.BytesWritten() != 0 {
		if d.content.BytesWritten() != 0 {
			writeText(&d.content, "Q\nq\n")
			d.needsExtraSave = true
		}
		d.contentBuffer.WriteToAndReset(&d.content)
	}

	if !dst.IsValid() {
		// blendMode is Src or SrcOut.
		return
	}

	// Changing the current content into a form XObject destroys the clip objects, which is fine since the XObject is
	// already clipped. But if the source has shape, it needs clipping too, so a copy of the clip (clipStack) was saved.
	stockPaint := canvas.NewPaint()

	var srcFormXObject IndirectReference
	if d.isContentEmpty() {
		// Nothing was drawn and there's no shape: the draw was a no-op, but dst needs restoring for that to be true.
		// With a shape, an empty source with Src/SrcIn/SrcOut/DstIn/DstAtop/Modulate reduces to Clear, and
		// DstOut/SrcAtop reduces to Dst.
		if shape == nil || blendMode == raster.BlendDstOut || blendMode == raster.BlendSrcATop {
			sce := d.newScopedContentEntry(nil, &identity, stockPaint, 0)
			if sce.content != nil {
				d.drawFormXObject(dst, sce.content)
			}
			sce.finish()
			return
		}
		blendMode = raster.BlendClear
	} else {
		srcFormXObject = d.makeFormXObjectFromDevice(false)
	}

	// srcFormXObject may contain alpha, but here we want it without alpha (a known limitation, not addressed).
	switch {
	case blendMode == raster.BlendSrcATop:
		d.drawFormXObjectWithMask(srcFormXObject, dst, raster.BlendSrcOver, true)
	case shape != nil:
		// Draw shape into a form XObject on a scratch device, then use it as the mask.
		filledPaint := canvas.NewPaint()
		filledPaint.Color = colorcore.Black
		filledPaint.Style = canvas.StyleFill
		cs := clipStack
		if cs == nil {
			cs = NewClipStack()
		}
		shapeDev := newDevice(d.pageSize, d.doc, d.initialTransform)
		shapeDev.internalDrawPath(cs, &identity, shape, filledPaint, true)
		d.drawFormXObjectWithMask(dst, shapeDev.makeFormXObjectFromDevice(false), raster.BlendSrcOver, true)
	default:
		d.drawFormXObjectWithMask(dst, srcFormXObject, raster.BlendSrcOver, true)
	}

	if blendMode == raster.BlendClear {
		return
	}
	switch blendMode {
	case raster.BlendSrc, raster.BlendDstATop:
		sce := d.newScopedContentEntry(nil, &identity, stockPaint, 0)
		if sce.content != nil {
			d.drawFormXObject(srcFormXObject, sce.content)
		}
		sce.finish()
		if blendMode == raster.BlendSrc {
			return
		}
	case raster.BlendSrcATop:
		sce := d.newScopedContentEntry(nil, &identity, stockPaint, 0)
		if sce.content != nil {
			d.drawFormXObject(dst, sce.content)
		}
		sce.finish()
	}

	// blendMode is now one of SrcIn/DstIn/SrcOut/DstOut/SrcATop/DstATop/Modulate.
	if blendMode == raster.BlendSrcIn || blendMode == raster.BlendSrcOut || blendMode == raster.BlendSrcATop {
		d.drawFormXObjectWithMask(srcFormXObject, dst, raster.BlendSrcOver, blendMode == raster.BlendSrcOut)
		return
	}
	mode := raster.BlendSrcOver
	if blendMode == raster.BlendModulate {
		d.drawFormXObjectWithMask(srcFormXObject, dst, raster.BlendSrcOver, false)
		mode = raster.BlendMultiply
	}
	d.drawFormXObjectWithMask(dst, srcFormXObject, mode, blendMode == raster.BlendDstOut)
}

// makeFormXObjectFromDevice captures the whole device (0,0,width,height) as a form XObject.
func (d *Device) makeFormXObjectFromDevice(alpha bool) IndirectReference {
	return d.makeFormXObjectFromDeviceBounds(geom.IRectWH(d.pageSize.Width, d.pageSize.Height), alpha)
}

// makeFormXObjectFromDeviceBounds pulls the device's content and resources into a form XObject placed by the inverse of
// the initial transform (so the XObject's own pre-transform content lands correctly when the parent re-applies its
// transform), then resets the device. alpha selects a DeviceGray (alpha/luminosity SMask) group.
func (d *Device) makeFormXObjectFromDeviceBounds(bounds geom.IRect, alpha bool) IndirectReference {
	inverseTransform := geom.IdentityMatrix()
	if inv, ok := d.initialTransform.Invert(); ok {
		inverseTransform = inv
	}
	colorSpace := ""
	if alpha {
		colorSpace = "DeviceGray"
	}
	// content() drains the graphics stack and resets the main content; makeResourceDict() reads the resource sets
	// before reset() clears them.
	contentData := d.contentBytes()
	resourceDict := d.makeResourceDict()
	xobject := makeFormXObject(d.doc, contentData,
		makeArrayInts(bounds.Left, bounds.Top, bounds.Right, bounds.Bottom),
		resourceDict, &inverseTransform, colorSpace)
	// We always draw the form XObjects we create back into the device, so we preserve font usage by simply resetting
	// instead of pulling it out and merging it back later.
	d.reset()
	return xobject
}

// drawFormXObjectWithMask draws xObject through an alpha soft mask (sMask), optionally inverting the mask's clip, then
// turns the soft mask back off for later draws.
func (d *Device) drawFormXObjectWithMask(xObject, sMask IndirectReference, mode raster.BlendMode, invertClip bool) {
	paint := canvas.NewPaint()
	paint.BlendMode = mode
	identity := geom.IdentityMatrix()
	sce := d.newScopedContentEntry(nil, &identity, paint, 0)
	if sce.content == nil {
		return
	}
	defer sce.finish()
	d.setGraphicState(getSMaskGraphicState(sMask, invertClip, sMaskAlpha, d.doc), sce.content)
	d.drawFormXObject(xObject, sce.content)
	d.clearMaskOnGraphicState(sce.content)
}

// setGraphicState selects the /ExtGState gs into content.
func (d *Device) setGraphicState(gs IndirectReference, content stream.WStream) {
	applyGraphicState(addResource(d.graphicStateResources, gs), content)
}

// clearMaskOnGraphicState selects the document's shared "/SMask /None" /ExtGState (lazily emitted) to turn the soft
// mask back off for later draws.
func (d *Device) clearMaskOnGraphicState(content stream.WStream) {
	if !d.doc.noSmaskGraphicState.IsValid() {
		tmp := NewTypedDict("ExtGState")
		tmp.InsertName("SMask", "None")
		d.doc.noSmaskGraphicState = d.doc.Emit(tmp)
	}
	d.setGraphicState(d.doc.noSmaskGraphicState, content)
}

// populateGraphicStateEntry fills entry from paint and the current matrix/clip: solid color, a color shader folded to a
// color, alpha + blend + stroke state, and — for every other shader — the /Pattern resource makeShader builds
// (gradient, image, or the rasterized generic fallback), recorded in entry.shaderIndex. A shader that cannot be
// represented as a pattern (empty surface bounds, or a non-invertible transform) leaves shaderIndex at -1, so the draw
// falls back to the paint color.
func (d *Device) populateGraphicStateEntry(matrix *geom.Matrix, clipStack *ClipStack, paint *canvas.Paint, textScale float32, entry *gsEntry) {
	entry.matrix = *matrix
	entry.clipStackGenID = wideOpenGenID
	if clipStack != nil {
		entry.clipStackGenID = clipStack.getTopmostGenID()
	}
	color := colorcore.Color4fFromColor(paint.Color)
	entry.color = colorcore.Color4f{R: color.R, G: color.G, B: color.B, A: 1}
	entry.shaderIndex = -1

	// PDF treats a shader as a color, so we only set one or the other.
	gsPaint := *paint
	if paint.Shader != nil {
		if cs, ok := paint.Shader.(*shaders.ColorShader); ok {
			sc := cs.Color()
			// entry color is the shader's color made opaque; the graphic-state alpha carries shaderA*paintA.
			entry.color = colorcore.Color4f{R: sc.R, G: sc.G, B: sc.B, A: 1}
			effAlpha := sc.A * color.A
			gsPaint.Color = colorcore.Color4f{R: sc.R, G: sc.G, B: sc.B, A: effAlpha}.ToColor()
			gsPaint.Shader = nil
		} else {
			// PDF positions patterns relative to the initial transform, so apply the current transform to the shader
			// parameters.
			transform := *matrix
			transform.PostConcat(&d.initialTransform)

			// PDF has no clamp tile mode, so it is simulated by a pattern the size of the current clip; map that clip
			// through the initial transform for a consistent coordinate system.
			clipStackBounds := d.bounds().ToRect()
			if clipStack != nil {
				clipStackBounds = clipStack.bounds(d.bounds())
			}
			clipStackBounds, _ = d.initialTransform.MapRect(clipStackBounds)
			bounds := clipStackBounds.RoundOut()

			// Use alpha 1 for the shader; the paint alpha is applied via the graphic state below.
			pdfShader := makeShader(d.doc, paint.Shader, transform, bounds,
				colorcore.Color4f{R: color.R, G: color.G, B: color.B, A: 1})
			if pdfShader.IsValid() {
				entry.shaderIndex = addResource(d.shaderResources, pdfShader)
			}
		}
	}

	newGraphicState := getGraphicStateForPaint(d.doc, &gsPaint)
	entry.graphicStateIndex = addResource(d.graphicStateResources, newGraphicState)
	entry.textScaleX = textScale
}

// addResource records the reference in set and returns its object number.
func addResource(set map[IndirectReference]struct{}, ref IndirectReference) int {
	set[ref] = struct{}{}
	return int(ref.value)
}

// drawFormXObject invokes the XObject (image or form) named by its /Resources entry. Tagged-PDF marked content (which
// would use a shape parameter) is not implemented.
func (d *Device) drawFormXObject(xObject IndirectReference, content stream.WStream) {
	writeResourceName(content, resXObject, addResource(d.xObjectResources, xObject))
	writeText(content, " Do\n")
}

// rectToRect returns the scale+translate matrix mapping from onto to (scale-to-fit, no aspect preservation), or
// identity when either rect is empty.
func rectToRect(from, to geom.Rect) geom.Matrix {
	if from.IsEmpty() || to.IsEmpty() {
		return geom.IdentityMatrix()
	}
	var m geom.Matrix
	sx := to.Width() / from.Width()
	sy := to.Height() / from.Height()
	m.SetScaleTranslate(sx, sy, to.Left-from.Left*sx, to.Top-from.Top*sy)
	return m
}

// isIntegralRect reports whether all four edges of r are integers.
func isIntegralRect(r geom.Rect) bool {
	return isInteger(r.Left) && isInteger(r.Top) && isInteger(r.Right) && isInteger(r.Bottom)
}

// isInteger reports whether x has no fractional part.
func isInteger(x float32) bool { return x == float32(math.Trunc(float64(x))) }

// ---- content / resource output ----------------------------------------------------------------------

// contentBytes drains the active graphics stack and returns the finished content stream bytes (prefixed by the initial
// transform), resetting the internal buffer.
func (d *Device) contentBytes() []byte {
	if d.activeStackState != nil {
		d.activeStackState.drainStack()
		d.activeStackState = nil
	}
	if d.content.BytesWritten() == 0 {
		return nil
	}
	buffer := stream.NewMemoryWStream()
	if d.initialTransform.Type() != geom.TypeIdentity {
		appendTransform(&d.initialTransform, buffer)
	}
	if d.needsExtraSave {
		writeText(buffer, "q\n")
	}
	buffer.Write(d.content.Bytes())
	d.content.Reset()
	if d.needsExtraSave {
		writeText(buffer, "Q\n")
	}
	d.needsExtraSave = false
	return buffer.DetachAsData()
}

// makeResourceDict builds the /Resources dict from the device's tracked graphic-state, shader, XObject, and font
// resources.
func (d *Device) makeResourceDict() *Dict {
	return makeResourceDict(
		sortedRefs(d.graphicStateResources),
		sortedRefs(d.shaderResources),
		sortedRefs(d.xObjectResources),
		sortedRefs(d.fontResources),
	)
}
