// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The device layer: the canvas targets the Device interface, so the GPU device and the PDF device plug in behind it.
// The raster device wraps a Pixmap and a raster clip stack. Layer devices (saveLayer) have a non-identity
// device-to-global transform; the public surface can only produce the integer translation form (layer buffer origin).

package canvas

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/textblob"
)

// Device is the drawing-target interface the canvas consumes.
type Device interface {
	// Width and Height return the device's pixel dimensions.
	Width() int32
	Height() int32

	// SetGlobalCTM installs the canvas's total matrix, mapped into this device's coordinate system (localToDevice =
	// globalToDevice * ctm).
	SetGlobalCTM(ctm *geom.Matrix)
	// LocalToDevice returns the current local-to-device matrix.
	LocalToDevice() *geom.Matrix

	// Origin returns the device's position in the root device's coordinate space, extracted from the device-to-global
	// matrix (valid only when that matrix is an integer translation, which the callers guarantee).
	Origin() geom.IPoint
	// SetDeviceCoordinateSystem installs the device-to-global/global-to-device pair and the local-to-device (layer)
	// matrix, then folds the integer buffer origin into all three.
	SetDeviceCoordinateSystem(deviceToGlobal, globalToDevice, localToDevice *geom.Matrix,
		bufferOrigin geom.IPoint)
	// DeviceToGlobal and GlobalToDevice return the device↔global transforms.
	DeviceToGlobal() geom.Matrix
	GlobalToDevice() geom.Matrix

	// AsFilterDevice returns the device's filtercore.Device adapter for image filtering, or nil when the device cannot
	// support it.
	AsFilterDevice() filtercore.Device
	// CreateFilterBackend returns the image-filtering backend for this device. The filter DAG about to be evaluated is
	// passed so the GPU device can stage the evaluation: DAGs whose nodes all evaluate GPU-natively get the GPU
	// backend, everything else the raster readback→CPU→upload fallback. filter may be nil (a restore-paint layer with
	// no DAG).
	CreateFilterBackend(filter filtercore.Filter) filtercore.Backend

	// PushClipStack/PopClipStack save and restore the device's clip stack.
	PushClipStack()
	PopClipStack()

	// ClipRect/ClipPath/ClipRegion mutate the device's clip.
	ClipRect(rect geom.Rect, op raster.ClipOp, aa bool)
	ClipPath(p *path.Path, op raster.ClipOp, aa bool)
	ClipRegion(rgn *raster.Region, op raster.ClipOp)

	// Clip queries.
	IsClipEmpty() bool
	IsClipRect() bool
	IsClipWideOpen() bool
	DevClipBounds() geom.IRect

	// Draw entry points.
	DrawPaint(paint *Paint)
	DrawRect(rect geom.Rect, paint *Paint)
	DrawOval(oval geom.Rect, paint *Paint)
	DrawRRect(rrect geom.RRect, paint *Paint)
	DrawArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, paint *Paint)
	DrawPoints(mode PointMode, pts []geom.Point, paint *Paint)
	DrawPath(p *path.Path, paint *Paint)
	// DrawImageRect takes the polymorphic imagecore.DrawableImage: a raster device rasterizes it, a GPU device draws a
	// texture-backed image natively.
	DrawImageRect(img imagecore.DrawableImage, src *geom.Rect, dst geom.Rect,
		sampling shaders.SamplingOptions, paint *Paint, constraint SrcRectConstraint)
	// DrawAtlas draws len(xforms) sprites, sprite i mapping the atlas sub-rect tex[i] onto xforms[i]'s quad. The
	// paint's shader is already the atlas image shader (Canvas.DrawAtlas installs it); with colors non-empty, sprite i
	// is modulated by colors[i] through mode. The raster device lowers per sprite, the GPU device draws through
	// DrawAtlasOp, and the PDF device draws nothing.
	DrawAtlas(xforms []geom.RSXform, tex []geom.Rect, colors []colorcore.Color,
		mode raster.BlendMode, paint *Paint)

	// DrawGlyphRunList draws positioned glyphs. The canvas is passed so glyphs that must be drawn as paths can route
	// back through it (mask filters, layers).
	DrawGlyphRunList(canvas *Canvas, glyphRunList *textblob.GlyphRunList, paint *Paint)

	// CreateDevice returns a compatible device for a layer. layerPaint is the saveLayer paint (nil for a plain layer);
	// a device that cannot serve as an image-filter render target (the PDF device) inspects its image/color filter to
	// fall back to a raster layer device that is drawn back as an image at restore.
	CreateDevice(width, height int32, layerPaint *Paint) Device
	// DrawDevice composites src (a layer) onto this device using the paint's alpha and blend mode.
	DrawDevice(src Device, paint *Paint)
	// RelativeTransform returns the matrix mapping this device's coordinate space to dst's. Every device implements it
	// so the canvas can build the layer→device mapping without knowing the concrete device type.
	RelativeTransform(dst Device) geom.Matrix
}

// releasable is the optional teardown hook a device (or an image-filter backend) implements when it owns state that
// outlives it on its own — the GPU device's cached clip masks, which are registered in the GPU resource cache under
// unique keys and can only be evicted by their owner. Go has no destructor to run this, so the canvas releases the
// devices and backends it creates itself: layer devices at restore and filter backends once their evaluation is done.
// Devices with no such state (BitmapDevice, the PDF device) do not implement it and are simply dropped.
type releasable interface {
	// Release drops the object's externally-cached state. It must be safe to call more than once, and must leave
	// already-recorded draws that reference the object's pixels valid (a layer is composited before it is released).
	Release()
}

// release runs v's teardown hook, if it has one.
func release(v any) {
	if r, ok := v.(releasable); ok {
		r.Release()
	}
}

// BitmapDevice is the raster device: it draws into a Pixmap through the scan converters.
type BitmapDevice struct {
	pix            *raster.Pixmap
	rcStack        *raster.ClipStack
	localToDevice  geom.Matrix
	deviceToGlobal geom.Matrix
	globalToDevice geom.Matrix
	// props carries the surface properties the text scaler consumes (pixel geometry for LCD16, E.4). The zero value is
	// unknown geometry — layers and filter devices keep it; only the surface package installs a real geometry.
	props font.DeviceProps
}

// NewBitmapDevice returns a device rendering into pix (unknown pixel geometry).
func NewBitmapDevice(pix *raster.Pixmap) *BitmapDevice {
	return &BitmapDevice{
		pix:            pix,
		rcStack:        raster.NewRasterClipStack(pix.Width, pix.Height),
		localToDevice:  geom.IdentityMatrix(),
		deviceToGlobal: geom.IdentityMatrix(),
		globalToDevice: geom.IdentityMatrix(),
	}
}

// SetDeviceProps installs the device's surface props (surface creation only — layers must keep the zero value's unknown
// geometry).
func (d *BitmapDevice) SetDeviceProps(props font.DeviceProps) { d.props = props }

// Width implements Device.
func (d *BitmapDevice) Width() int32 { return d.pix.Width }

// Height implements Device.
func (d *BitmapDevice) Height() int32 { return d.pix.Height }

// Pixmap exposes the device's pixels for snapshots and layer composition.
func (d *BitmapDevice) Pixmap() *raster.Pixmap { return d.pix }

// SetGlobalCTM implements Device: localToDevice = globalToDevice * ctm.
func (d *BitmapDevice) SetGlobalCTM(ctm *geom.Matrix) {
	d.localToDevice.SetConcat(&d.globalToDevice, ctm)
}

// LocalToDevice implements Device.
func (d *BitmapDevice) LocalToDevice() *geom.Matrix { return &d.localToDevice }

// Origin implements Device: the device-to-global translation, floored (only meaningful when that matrix is
// pixel-aligned, which callers of Origin guarantee).
func (d *BitmapDevice) Origin() geom.IPoint {
	return geom.IPoint{
		X: geom.FloorToInt(d.deviceToGlobal.Get(geom.MTransX)),
		Y: geom.FloorToInt(d.deviceToGlobal.Get(geom.MTransY)),
	}
}

// SetDeviceCoordinateSystem implements Device.
func (d *BitmapDevice) SetDeviceCoordinateSystem(deviceToGlobal, globalToDevice, localToDevice *geom.Matrix, bufferOrigin geom.IPoint) {
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

// DeviceToGlobal implements Device.
func (d *BitmapDevice) DeviceToGlobal() geom.Matrix { return d.deviceToGlobal }

// GlobalToDevice implements Device.
func (d *BitmapDevice) GlobalToDevice() geom.Matrix { return d.globalToDevice }

// PushClipStack implements Device.
func (d *BitmapDevice) PushClipStack() { d.rcStack.Save() }

// PopClipStack implements Device.
func (d *BitmapDevice) PopClipStack() { d.rcStack.Restore() }

// ClipRect implements Device.
func (d *BitmapDevice) ClipRect(rect geom.Rect, op raster.ClipOp, aa bool) {
	d.rcStack.ClipRect(&d.localToDevice, rect, op, aa)
}

// ClipPath implements Device.
func (d *BitmapDevice) ClipPath(p *path.Path, op raster.ClipOp, aa bool) {
	d.rcStack.ClipPath(&d.localToDevice, p, op, aa)
}

// ClipRegion implements Device. The region is in global coordinates; layers translate it into device space by applying
// the global-to-device translation.
func (d *BitmapDevice) ClipRegion(rgn *raster.Region, op raster.ClipOp) {
	origin := d.Origin()
	if origin.X != 0 || origin.Y != 0 {
		// Translate builds fresh run storage, so a shallow copy keeps the caller's region intact.
		translated := *rgn
		translated.Translate(-origin.X, -origin.Y)
		d.rcStack.ClipRegion(&translated, op)
		return
	}
	d.rcStack.ClipRegion(rgn, op)
}

// IsClipEmpty implements Device.
func (d *BitmapDevice) IsClipEmpty() bool { return d.rcStack.RC().IsEmpty() }

// IsClipRect implements Device.
func (d *BitmapDevice) IsClipRect() bool {
	rc := d.rcStack.RC()
	return !rc.IsEmpty() && rc.IsRect()
}

// IsClipWideOpen implements Device.
func (d *BitmapDevice) IsClipWideOpen() bool {
	rc := d.rcStack.RC()
	// If we're AA, we can't be wide-open (we would represent that as BW)
	return rc.IsBW() && rc.BWRgn().IsRect() &&
		rc.BWRgn().Bounds() == geom.IRectWH(d.pix.Width, d.pix.Height)
}

// DevClipBounds implements Device.
func (d *BitmapDevice) DevClipBounds() geom.IRect { return d.rcStack.RC().Bounds() }

// draw assembles the per-draw lowering state for this device.
func (d *BitmapDevice) draw() draw {
	return draw{dst: d.pix, ctm: &d.localToDevice, rc: d.rcStack.RC()}
}

// DrawPaint implements Device. It never routes through the tiler: the paint fill lowers to integer region fills with no
// scan-converter coordinate limits.
func (d *BitmapDevice) DrawPaint(paint *Paint) {
	dr := d.draw()
	dr.drawPaint(paint)
}

// DrawRect implements Device (tiling with the paint-grown rect bounds past 8191px).
func (d *BitmapDevice) DrawRect(rect geom.Rect, paint *Paint) {
	if d.clipMayNeedTiling() {
		var storage geom.Rect
		var tiler drawTiler
		tiler.init(d, paintBounds(rect, paint, &storage))
		for dr := tiler.next(); dr != nil; dr = tiler.next() {
			dr.drawRect(rect, paint)
		}
		return
	}
	dr := d.draw()
	dr.drawRect(rect, paint)
}

// DrawOval implements Device (tiling with the paint-grown oval bounds past 8191px).
func (d *BitmapDevice) DrawOval(oval geom.Rect, paint *Paint) {
	if d.clipMayNeedTiling() {
		var storage geom.Rect
		var tiler drawTiler
		tiler.init(d, paintBounds(oval, paint, &storage))
		for dr := tiler.next(); dr != nil; dr = tiler.next() {
			dr.drawOval(oval, paint)
		}
		return
	}
	dr := d.draw()
	dr.drawOval(oval, paint)
}

// DrawRRect implements Device (tiling with the paint-grown rrect bounds past 8191px).
func (d *BitmapDevice) DrawRRect(rrect geom.RRect, paint *Paint) {
	if d.clipMayNeedTiling() {
		var storage geom.Rect
		var tiler drawTiler
		tiler.init(d, paintBounds(rrect.Rect, paint, &storage))
		for dr := tiler.next(); dr != nil; dr = tiler.next() {
			dr.drawRRect(rrect, paint)
		}
		return
	}
	dr := d.draw()
	dr.drawRRect(rrect, paint)
}

// DrawArc implements Device by lowering the arc to a path; the lowering routes through drawPathImpl so it tiles like
// DrawPath.
func (d *BitmapDevice) DrawArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, paint *Paint) {
	isFillNoPathEffect := paint.Style == StyleFill && paint.PathEffect == nil
	p := path.CreateDrawArcPath(oval, startAngle, sweepAngle, useCenter, isFillNoPathEffect)
	d.drawPathImpl(p, paint, true)
}

// DrawPoints implements Device (tiling with no bounds past 8191px — can't easily get bounds of points).
func (d *BitmapDevice) DrawPoints(mode PointMode, pts []geom.Point, paint *Paint) {
	if d.clipMayNeedTiling() {
		var tiler drawTiler
		tiler.init(d, nil)
		for dr := tiler.next(); dr != nil; dr = tiler.next() {
			dr.drawPoints(mode, pts, paint)
		}
		return
	}
	dr := d.draw()
	dr.drawPoints(mode, pts, paint)
}

// DrawPath implements Device.
func (d *BitmapDevice) DrawPath(p *path.Path, paint *Paint) {
	d.drawPathImpl(p, paint, false)
}

// drawPathImpl draws the path with optional tiling: bounds are computed (and paint-grown) only when the device itself
// needs tiling and the path is not inverse-filled; a tiling draw forces pathIsMutable off so every tile transforms the
// same source path.
func (d *BitmapDevice) drawPathImpl(p *path.Path, paint *Paint, pathIsMutable bool) {
	if d.clipMayNeedTiling() {
		var storage geom.Rect
		var bounds *geom.Rect
		if tilerNeedsTiling(d) && !p.IsInverseFillType() {
			bounds = paintBounds(p.Bounds(), paint, &storage)
		}
		var tiler drawTiler
		tiler.init(d, bounds)
		if tiler.needsTiling {
			pathIsMutable = false
		}
		for dr := tiler.next(); dr != nil; dr = tiler.next() {
			dr.drawPath(p, paint, nil, pathIsMutable)
		}
		return
	}
	dr := d.draw()
	dr.drawPath(p, paint, nil, pathIsMutable)
}

// DrawGlyphRunList implements Device, routing to the glyph-run painter (tiling with no bounds past 8191px).
func (d *BitmapDevice) DrawGlyphRunList(canvas *Canvas, glyphRunList *textblob.GlyphRunList, paint *Paint) {
	if d.clipMayNeedTiling() {
		var tiler drawTiler
		tiler.init(d, nil)
		for dr := tiler.next(); dr != nil; dr = tiler.next() {
			drawGlyphRunListForBitmapDevice(canvas, dr, glyphRunList, paint, dr.ctm, &d.props)
		}
		return
	}
	dr := d.draw()
	if dr.rc.IsEmpty() {
		return
	}
	drawGlyphRunListForBitmapDevice(canvas, &dr, glyphRunList, paint, dr.ctm, &d.props)
}

// CreateDevice implements Device: a fresh premul raster device. The layer paint is unused — a raster device is always a
// valid image-filter render target, so unlike the PDF device it never needs to inspect the filter to pick a fallback
// backing.
func (d *BitmapDevice) CreateDevice(width, height int32, _ *Paint) Device {
	return NewBitmapDevice(raster.NewPixmap(width, height))
}

// DrawDevice implements Device: snapshot the layer and draw it at the relative integer translation, which lowers to the
// sprite blitters. It routes around the draw tiler: the sprite lanes are integer blits with no scan-converter
// coordinate limits.
func (d *BitmapDevice) DrawDevice(src Device, paint *Paint) {
	bd, ok := src.(*BitmapDevice)
	if !ok {
		return
	}
	srcOrigin := bd.Origin()
	dstOrigin := d.Origin()
	dr := d.draw()
	dr.drawSprite(bd.pix, srcOrigin.X-dstOrigin.X, srcOrigin.Y-dstOrigin.Y, paint)
}

// RelativeTransform implements Device: the matrix mapping this device's coordinate space to dst's (dst.globalToDevice *
// this.deviceToGlobal).
func (d *BitmapDevice) RelativeTransform(dst Device) geom.Matrix {
	g2d := dst.GlobalToDevice()
	var m geom.Matrix
	m.SetConcat(&g2d, &d.deviceToGlobal)
	return m
}

// SnapSpecial returns a special image viewing a subset of this device's pixels without copying (the callers only snap
// devices that are not subsequently drawn to).
func (d *BitmapDevice) SnapSpecial(subset geom.IRect) *filtercore.SpecialImage {
	if !subset.Intersect(geom.IRectWH(d.pix.Width, d.pix.Height)) {
		return nil
	}
	return filtercore.NewSpecialImage(subset, imagecore.PixelsFromPixmap(*d.pix))
}

// DrawSpecial draws the subset view of img using the given device-space matrix (the current local-to-device is
// ignored).
func (d *BitmapDevice) DrawSpecial(img *filtercore.SpecialImage, deviceMatrix geom.Matrix, sampling shaders.SamplingOptions, paint *Paint) {
	px := img.AsImage().PeekPixels(imagecore.CachingAllow)
	if px == nil {
		return
	}
	dr := draw{dst: d.pix, ctm: &deviceMatrix, rc: d.rcStack.RC()}
	identity := geom.IdentityMatrix()
	dr.drawBitmap(px, &identity, nil, sampling, paint)
}

// AsFilterDevice implements Device.
func (d *BitmapDevice) AsFilterDevice() filtercore.Device { return &bitmapFilterDevice{dev: d} }

// CreateFilterBackend implements Device: the raster image-filtering backend.
func (d *BitmapDevice) CreateFilterBackend(_ filtercore.Filter) filtercore.Backend {
	return newRasterFilterBackend()
}
