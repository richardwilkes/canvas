// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The canvas core: the matrix save stack with deferred saves, clip ops delegated to the device, quick-reject bounds
// maintenance, saveLayer/saveLayerAlpha with layer devices and restore-time composition (the machinery specific to
// image-filtered layers lives in filterlayer.go), and the draw entry points (paint, color, clear, rect, oval, rrect,
// round-rect, arc, circle, line, point(s), path, plus the image draws in drawimage.go and text in text.go). The
// canvas's total matrix is the 3x3 geom.Matrix, which is what the public surface exposes.

package canvas

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// SurfaceBase is the canvas→surface notification hook: the owning surface learns that pixels are about to change so it
// can copy-on-write an outstanding snapshot.
type SurfaceBase interface {
	AboutToDraw()
}

// layerRec is one saveLayer record: the device holding the layer's pixels, the paint that composites them at restore
// time, the image filter evaluated during that composition, and whether the layer was created with the 1px transparent
// padding filters use.
type layerRec struct {
	device          Device
	imageFilter     filtercore.Filter
	paint           Paint
	includesPadding bool
}

// mcRec is one entry of the matrix/clip save stack. Clip state lives in the devices' clip stacks; this records the
// total matrix, the top device at this level, the layer (when this record began one), and the deferred-save count.
type mcRec struct {
	device            Device
	layer             *layerRec
	matrix            geom.Matrix
	deferredSaveCount int
}

// Canvas is the drawing context: it manages the matrix/clip save stack over a device tree (the root device plus
// saveLayer layers) and dispatches draws to the top device.
type Canvas struct {
	rootDevice Device
	surface    SurfaceBase
	// mcStack is the matrix/clip save stack; the last element is the current record.
	mcStack []mcRec
	// quickRejectBounds is the device clip bounds mapped to the root device space and outset by 1 for AA slop, in float
	// coordinates.
	quickRejectBounds geom.Rect
	saveCount         int
}

// New returns a canvas drawing into the device.
func New(device Device) *Canvas {
	c := &Canvas{
		rootDevice: device,
		mcStack:    []mcRec{{matrix: geom.IdentityMatrix(), device: device}},
		saveCount:  1,
	}
	device.SetGlobalCTM(&c.mcStack[0].matrix)
	c.quickRejectBounds = c.computeDeviceClipBounds(true)
	return c
}

// NewForPixmap returns a canvas drawing into pix through a BitmapDevice.
func NewForPixmap(pix *raster.Pixmap) *Canvas {
	return New(NewBitmapDevice(pix))
}

func (c *Canvas) top() *mcRec { return &c.mcStack[len(c.mcStack)-1] }

// topDevice returns the device draws land on (the innermost layer, or the root).
func (c *Canvas) topDevice() Device { return c.top().device }

// BaseDevice returns the root device.
func (c *Canvas) BaseDevice() Device { return c.rootDevice }

// SetSurfaceBase installs the owning surface's copy-on-write hook.
func (c *Canvas) SetSurfaceBase(s SurfaceBase) { c.surface = s }

// SurfaceBaseValue returns the installed surface hook.
func (c *Canvas) SurfaceBaseValue() SurfaceBase { return c.surface }

// predrawNotify gives the owning surface a chance to copy-on-write before pixels change.
func (c *Canvas) predrawNotify() bool {
	if c.surface != nil {
		c.surface.AboutToDraw()
	}
	return true
}

///////////////////////////////////////////////////////////////////////////////
// save/restore

// SaveCount returns the number of saved states, including the initial one.
func (c *Canvas) SaveCount() int { return c.saveCount }

// Save pushes a save record; the record is deferred until a mutation actually needs it. Returns the save count before
// the save.
func (c *Canvas) Save() int {
	c.saveCount++
	c.top().deferredSaveCount++
	return c.saveCount - 1
}

// checkForDeferredSave materializes a pending deferred save before a mutation.
func (c *Canvas) checkForDeferredSave() {
	if c.top().deferredSaveCount > 0 {
		c.doSave()
	}
}

func (c *Canvas) doSave() {
	c.top().deferredSaveCount--
	c.internalSave()
}

func (c *Canvas) internalSave() {
	prev := c.top()
	c.mcStack = append(c.mcStack, mcRec{matrix: prev.matrix, device: prev.device})
	c.topDevice().PushClipStack()
}

// Restore undoes the most recent Save, restoring the matrix and clip; if the record began a layer, its contents are
// composited into the parent device.
func (c *Canvas) Restore() {
	if c.top().deferredSaveCount > 0 {
		c.saveCount--
		c.top().deferredSaveCount--
	} else if len(c.mcStack) > 1 { // check for underflow
		c.saveCount--
		c.internalRestore()
	}
}

func (c *Canvas) internalRestore() {
	layer := c.top().layer
	c.mcStack = c.mcStack[:len(c.mcStack)-1]
	c.topDevice().PopClipStack()
	c.topDevice().SetGlobalCTM(&c.top().matrix)

	// Draw the layer's device contents into the now-current older device. We can't call public draw functions since we
	// don't want to record them.
	if layer != nil && c.predrawNotify() {
		dstDev := c.topDevice()
		if layer.imageFilter != nil {
			compat := compatYes
			if layer.includesPadding {
				compat = compatYesWithPadding
			}
			c.internalDrawDeviceWithFilter(layer.device, dstDev, layer.imageFilter, &layer.paint,
				compat)
		} else {
			dstDev.DrawDevice(layer.device, &layer.paint)
		}
	}
	if layer != nil {
		// The layer device is done: it has been composited (or the layer was rejected) and nothing else can reach it, so
		// hand back whatever it caches outside itself. Its pixels stay valid for the draws recorded above — see
		// releasable.
		release(layer.device)
	}

	// Update the quick-reject bounds in case the restore changed the top device or the removed save record had included
	// modifications to the clip stack.
	c.quickRejectBounds = c.computeDeviceClipBounds(true)
}

// RestoreToCount restores until the save count matches count.
func (c *Canvas) RestoreToCount(count int) {
	if count < 1 {
		count = 1 // safety check
	}
	for n := c.saveCount - count; n > 0; n-- {
		c.Restore()
	}
}

///////////////////////////////////////////////////////////////////////////////
// saveLayer

// SaveLayer begins a new layer, drawn into an offscreen premul device and composited with paint's alpha/blend mode at
// the matching restore. bounds is an optional content hint that clips the layer's extent; either argument may be nil.
func (c *Canvas) SaveLayer(bounds *geom.Rect, paint *Paint) int {
	if paint != nil && paint.nothingToDraw() {
		// no need for the layer (or any of the draws until the matching restore()
		c.Save()
		c.ClipRect(geom.Rect{}, raster.ClipIntersect, false)
	} else {
		c.saveCount++
		c.internalSaveLayer(bounds, paint)
	}
	return c.saveCount - 1
}

// SaveLayerAlpha is SaveLayer with a paint carrying only the given alpha (255 skips the paint).
func (c *Canvas) SaveLayerAlpha(bounds *geom.Rect, alpha uint8) int {
	if alpha == 255 {
		return c.SaveLayer(bounds, nil)
	}
	tmp := NewPaint()
	tmp.Color = tmp.Color.WithAlpha(alpha)
	return c.SaveLayer(bounds, tmp)
}

// abortLayer marks the current top device empty so nothing draws until the canvas is restored past this saveLayer.
func (c *Canvas) abortLayer() {
	c.topDevice().ClipRect(geom.Rect{}, raster.ClipIntersect, false)
	c.quickRejectBounds = c.computeDeviceClipBounds(true)
}

// internalSaveLayer creates the layer's device for the reachable configuration: paint image filters, alpha, color
// filters, and blend modes; backdrops, save-layer flags, filter spans, and coverage layers have no public entry points.
// The layer's coordinate space comes from decomposing the CTM against the filter's matrix capability; filterless layers
// keep an identity remainder so they differ from the prior device by an integer translation.
func (c *Canvas) internalSaveLayer(bounds *geom.Rect, paint *Paint) {
	// Do this before we create the layer. We don't call the public save() since that would invoke a possibly overridden
	// virtual.
	c.internalSave()

	if c.IsClipEmpty() {
		// Early out if the layer wouldn't draw anything
		return
	}

	// Build up the paint for restoring the layer, taking only the pieces of rec.fPaint that are relevant. Filtering is
	// chosen in internalDrawDeviceWithFilter based on the device's coordinate space.
	var restorePaint Paint
	if paint != nil {
		restorePaint = *paint
	} else {
		restorePaint = *NewPaint()
	}
	restorePaint.Style = StyleFill // a layer is filled out "infinitely"
	restorePaint.PathEffect = nil  // path effects are ignored for saved layers
	restorePaint.MaskFilter = nil  // mask filters are ignored for saved layers
	restorePaint.ImageFilter = nil // the image filter is held separately
	// Smooth non-axis-aligned layer edges; this automatically downgrades to non-AA for aligned layer restores, matching
	// the historical behavior where the post-applied bilerp of a matrix-transform filter also smoothed cropped edges.
	restorePaint.AntiAlias = true

	var filter filtercore.Filter
	if paint != nil && paint.ImageFilter != nil {
		filter = paint.ImageFilter
	}

	// If the restorePaint has a transparency-affecting color filter or blend mode, the output is unbounded during
	// restore(); without an image filter the plain drawDevice would not apply effects beyond the layer's image, so such
	// restores cannot use the trivial path either.
	cf := restorePaint.ColorFilter
	drawDeviceMustFillClip := filter == nil &&
		((cf != nil && filtercore.ColorFilterAffectsTransparentBlack(cf)) ||
			raster.BlendModeAffectsTransparentBlack(restorePaint.BlendMode))
	trivialRestore := !drawDeviceMustFillClip

	// Size the new layer relative to the prior device, which may already be aligned for filters.
	priorDevice := c.topDevice()
	outputBounds := priorDevice.DevClipBounds()

	// Set the bounds hint if provided and there are no further effects on prior device content.
	var contentBounds *geom.Rect
	if bounds != nil && trivialRestore {
		contentBounds = bounds
	}

	newLayerMapping, layerBounds, ok := getLayerMappingAndBounds(filter,
		priorDevice.LocalToDevice(), outputBounds, contentBounds)
	if !ok {
		// The filtered content would not draw anything, or the new device space has an invalid coordinate system.
		c.abortLayer()
		return
	}

	paddedLayer := false
	if layerBounds.IsEmpty() {
		// The image filter graph does not require any input, so no layer is rendered for the source image, but the
		// filter output still needs rendering now — the paired restore() will be a no-op.
		if filter != nil {
			c.internalDrawDeviceWithFilter(nil, priorDevice, filter, &restorePaint, compatUnknown)
		}
		// Regardless, mark the layer as empty until the restore().
		c.abortLayer()
		return
	}
	if filter != nil {
		// A buffer of padding lets image filtering avoid accessing uninitialized data and switch from shader-decal'ing
		// to clamping.
		paddedLayerBounds := layerBounds.Inset(-1, -1)
		if paddedLayerBounds.Left < layerBounds.Left && paddedLayerBounds.Top < layerBounds.Top &&
			paddedLayerBounds.Right > layerBounds.Right &&
			paddedLayerBounds.Bottom > layerBounds.Bottom {
			// The outset was not saturated, so the transparent pixels can be preserved.
			layerBounds = paddedLayerBounds
			paddedLayer = true
		}
	}

	newDevice := priorDevice.CreateDevice(layerBounds.Width(), layerBounds.Height(), paint)
	if newDevice == nil {
		// The layer's pixels could not be allocated — the GPU device returns nil for an abandoned context, an unknown
		// format, or dimensions past Caps.MaxRenderTargetSize. Upstream substitutes a no-pixels device so the layer's
		// draws are squashed while the state still unwinds normally; marking the layer empty does the same here and
		// leaves the paired restore() a no-op.
		c.abortLayer()
		return
	}

	// Clip while the device coordinate space is the identity so it's easy to define the rect that excludes the added
	// padding pixels, ensuring they remain transparent black.
	if paddedLayer {
		newDevice.ClipRect(newDevice.DevClipBounds().Inset(1, 1).ToRect(), raster.ClipIntersect,
			false)
	}

	// Configure the device to match the determined mapping. The prior device's global transform is applied since the
	// mapping only defines the transforms between the two devices.
	priorD2G := priorDevice.DeviceToGlobal()
	layerToDev := newLayerMapping.LayerToDevice()
	var newD2G geom.Matrix
	newD2G.SetConcat(&priorD2G, &layerToDev)
	devToLayer := newLayerMapping.DeviceToLayer()
	priorG2D := priorDevice.GlobalToDevice()
	var newG2D geom.Matrix
	newG2D.SetConcat(&devToLayer, &priorG2D)
	layerMatrix := newLayerMapping.LayerMatrix()
	newDevice.SetDeviceCoordinateSystem(&newD2G, &newG2D, &layerMatrix,
		geom.IPoint{X: layerBounds.Left, Y: layerBounds.Top})

	c.top().layer = &layerRec{
		device:          newDevice,
		paint:           restorePaint,
		imageFilter:     filter,
		includesPadding: paddedLayer,
	}
	c.top().device = newDevice
	c.quickRejectBounds = c.computeDeviceClipBounds(true)
}

///////////////////////////////////////////////////////////////////////////////
// matrix

// TotalMatrix returns the current total matrix (CTM).
func (c *Canvas) TotalMatrix() geom.Matrix { return c.top().matrix }

// Translate pre-concatenates a translation onto the current matrix.
func (c *Canvas) Translate(dx, dy float32) {
	if dx != 0 || dy != 0 {
		c.checkForDeferredSave()
		c.top().matrix.PreTranslate(dx, dy)
		c.topDevice().SetGlobalCTM(&c.top().matrix)
	}
}

// Scale pre-concatenates a scale onto the current matrix.
func (c *Canvas) Scale(sx, sy float32) {
	if sx != 1 || sy != 1 {
		c.checkForDeferredSave()
		c.top().matrix.PreScale(sx, sy)
		c.topDevice().SetGlobalCTM(&c.top().matrix)
	}
}

// Rotate pre-concatenates a rotation (in degrees) onto the current matrix.
func (c *Canvas) Rotate(degrees float32) {
	var m geom.Matrix
	m.SetRotate(degrees)
	c.Concat(&m)
}

// RotatePivot pre-concatenates a rotation of degrees about the pivot (px, py).
func (c *Canvas) RotatePivot(degrees, px, py float32) {
	var m geom.Matrix
	m.SetRotatePivot(degrees, px, py)
	c.Concat(&m)
}

// Skew pre-concatenates a skew onto the current matrix.
func (c *Canvas) Skew(sx, sy float32) {
	var m geom.Matrix
	m.SetSkew(sx, sy)
	c.Concat(&m)
}

// Concat pre-concatenates matrix onto the current matrix.
func (c *Canvas) Concat(matrix *geom.Matrix) {
	if matrix.IsIdentity() {
		return
	}
	c.checkForDeferredSave()
	c.top().matrix.PreConcat(matrix)
	c.topDevice().SetGlobalCTM(&c.top().matrix)
}

// SetMatrix replaces the current total matrix.
func (c *Canvas) SetMatrix(matrix *geom.Matrix) {
	c.checkForDeferredSave()
	c.top().matrix = *matrix
	c.topDevice().SetGlobalCTM(&c.top().matrix)
}

// ResetMatrix resets the current total matrix to identity.
func (c *Canvas) ResetMatrix() {
	m := geom.IdentityMatrix()
	c.SetMatrix(&m)
}

///////////////////////////////////////////////////////////////////////////////
// clips

// ClipRect combines the rect (mapped by the current matrix) with the clip using op.
func (c *Canvas) ClipRect(rect geom.Rect, op raster.ClipOp, doAA bool) {
	if !rect.IsFinite() {
		return
	}
	c.checkForDeferredSave()
	c.topDevice().ClipRect(rect.Sorted(), op, doAA)
	c.quickRejectBounds = c.computeDeviceClipBounds(true)
}

// ClipPath combines the path (mapped by the current matrix) with the clip using op.
func (c *Canvas) ClipPath(p *path.Path, op raster.ClipOp, doAA bool) {
	c.checkForDeferredSave()
	c.topDevice().ClipPath(p, op, doAA)
	c.quickRejectBounds = c.computeDeviceClipBounds(true)
}

// ClipRegion combines the device-space region with the clip using op.
func (c *Canvas) ClipRegion(rgn *raster.Region, op raster.ClipOp) {
	c.checkForDeferredSave()
	c.topDevice().ClipRegion(rgn, op)
	c.quickRejectBounds = c.computeDeviceClipBounds(true)
}

// IsClipEmpty reports whether the clip excludes all drawing.
func (c *Canvas) IsClipEmpty() bool { return c.topDevice().IsClipEmpty() }

// IsClipRect reports whether the clip is a plain rectangle.
func (c *Canvas) IsClipRect() bool { return c.topDevice().IsClipRect() }

// computeDeviceClipBounds returns the top device's clip bounds mapped into the root device's space by the
// device-to-global transform (an integer translation for plain layers; filter layers can carry rotation/scale),
// optionally outset by 1 for AA slop.
func (c *Canvas) computeDeviceClipBounds(outsetForAA bool) geom.Rect {
	dev := c.topDevice()
	if dev.IsClipEmpty() {
		return geom.Rect{}
	}
	d2g := dev.DeviceToGlobal()
	bounds, _ := path.MapRect(&d2g, dev.DevClipBounds().ToRect())
	if outsetForAA {
		// Expand bounds out by 1 in case we are anti-aliasing. We store the bounds as floats to enable a faster quick
		// reject implementation.
		bounds = bounds.Outset(1, 1)
	}
	return bounds
}

// DeviceClipBounds returns the clip bounds in root device space, rounded out.
func (c *Canvas) DeviceClipBounds() geom.IRect {
	return c.computeDeviceClipBounds(false).RoundOut()
}

// LocalClipBounds returns the clip bounds mapped back into local (pre-matrix) coordinates, expanded for AA.
func (c *Canvas) LocalClipBounds() geom.Rect {
	ibounds := c.DeviceClipBounds()
	if ibounds.IsEmpty() {
		return geom.Rect{}
	}
	inverse, ok := c.top().matrix.Invert()
	if !ok {
		// if we can't invert the CTM, we can't return local clip bounds
		return geom.Rect{}
	}
	// adjust it outwards in case we are antialiasing
	const margin = 1
	mapped, _ := path.MapRect(&inverse, ibounds.Inset(-margin, -margin).ToRect())
	return mapped
}

// QuickReject reports whether the rect, mapped by the total matrix, definitely misses the clip.
func (c *Canvas) QuickReject(src geom.Rect) bool {
	m := c.top().matrix
	devRect, _ := path.MapRect(&m, src)
	return !devRect.IsFinite() || !devRect.Intersects(c.quickRejectBounds)
}

// QuickRejectPath is QuickReject on the path's bounds (empty paths always reject).
func (c *Canvas) QuickRejectPath(p *path.Path) bool {
	return p.IsEmpty() || c.QuickReject(p.Bounds())
}

// internalQuickReject rejects a draw when its bounds (adjusted by the paint's effects) miss the clip or the paint draws
// nothing.
func (c *Canvas) internalQuickReject(bounds geom.Rect, paint *Paint) bool {
	if !bounds.IsFinite() || paint.nothingToDraw() {
		return true
	}
	if paint.canComputeFastBounds() {
		return c.QuickReject(paint.computeFastBounds(bounds))
	}
	return false
}

///////////////////////////////////////////////////////////////////////////////
// draws

// DrawPaint fills the entire clip with the paint.
func (c *Canvas) DrawPaint(paint *Paint) {
	c.internalDrawPaint(paint)
}

func (c *Canvas) internalDrawPaint(paint *Paint) {
	// drawPaint does not call internalQuickReject() because computing its geometry is not free, and the two conditions
	// below are sufficient.
	if paint.nothingToDraw() || c.IsClipEmpty() {
		return
	}
	if dp, restore, ok := c.aboutToDraw(paint, nil); ok {
		c.topDevice().DrawPaint(dp)
		if restore != nil {
			restore()
		}
	}
}

// DrawColor fills the clip with the color using the blend mode.
func (c *Canvas) DrawColor(color colorcore.Color, mode raster.BlendMode) {
	paint := NewPaint()
	paint.Color = color
	paint.BlendMode = mode
	c.internalDrawPaint(paint)
}

// Clear fills the clip with the color in Src mode, replacing the existing pixels.
func (c *Canvas) Clear(color colorcore.Color) {
	c.DrawColor(color, raster.BlendSrc)
}

// DrawRect draws the rectangle with the paint. To avoid redundant logic in the culling code and backends, rects are
// always sorted before passing them along.
func (c *Canvas) DrawRect(r geom.Rect, paint *Paint) {
	c.onDrawRect(r.Sorted(), paint)
}

func (c *Canvas) onDrawRect(r geom.Rect, paint *Paint) {
	if c.internalQuickReject(r, paint) {
		return
	}
	if dp, restore, ok := c.aboutToDraw(paint, &r); ok {
		c.topDevice().DrawRect(r, dp)
		if restore != nil {
			restore()
		}
	}
}

// DrawOval draws an oval inscribed in r with the paint (the rect is sorted first, as DrawRect).
func (c *Canvas) DrawOval(r geom.Rect, paint *Paint) {
	c.onDrawOval(r.Sorted(), paint)
}

func (c *Canvas) onDrawOval(oval geom.Rect, paint *Paint) {
	if c.internalQuickReject(oval, paint) {
		return
	}
	if dp, restore, ok := c.aboutToDraw(paint, &oval); ok {
		c.topDevice().DrawOval(oval, dp)
		if restore != nil {
			restore()
		}
	}
}

// DrawRRect draws the rounded rect with the paint, delegating rect- and oval-shaped rrects to the simpler draw
// operations.
func (c *Canvas) DrawRRect(rrect geom.RRect, paint *Paint) {
	bounds := rrect.Rect

	switch rrect.Type {
	case geom.RRectRect:
		c.DrawRect(bounds, paint)
		return
	case geom.RRectOval:
		c.DrawOval(bounds, paint)
		return
	}

	if c.internalQuickReject(bounds, paint) {
		return
	}
	if dp, restore, ok := c.aboutToDraw(paint, &bounds); ok {
		c.topDevice().DrawRRect(rrect, dp)
		if restore != nil {
			restore()
		}
	}
}

// DrawRoundRect draws r as a rounded rect with corner radii rx/ry; non-positive radii draw a plain rect.
func (c *Canvas) DrawRoundRect(r geom.Rect, rx, ry float32, paint *Paint) {
	if rx > 0 && ry > 0 {
		c.DrawRRect(geom.MakeRRect(r, rx, ry), paint)
	} else {
		c.DrawRect(r, paint)
	}
}

// DrawCircle draws a circle centered at (cx, cy); a negative radius is treated as zero.
func (c *Canvas) DrawCircle(cx, cy, radius float32, paint *Paint) {
	if radius < 0 {
		radius = 0
	}
	c.DrawOval(geom.RectLTRB(cx-radius, cy-radius, cx+radius, cy+radius), paint)
}

// DrawArc draws an arc of the oval from startAngle spanning sweepAngle (both in degrees); useCenter connects the arc to
// the oval's center, forming a wedge.
func (c *Canvas) DrawArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, paint *Paint) {
	if oval.IsEmpty() || sweepAngle == 0 {
		return
	}
	if c.internalQuickReject(oval, paint) {
		return
	}
	if dp, restore, ok := c.aboutToDraw(paint, &oval); ok {
		c.topDevice().DrawArc(oval, startAngle, sweepAngle, useCenter, dp)
		if restore != nil {
			restore()
		}
	}
}

// DrawPoints draws the points per mode (points, lines, or polygon), always with stroke semantics.
func (c *Canvas) DrawPoints(mode PointMode, pts []geom.Point, paint *Paint) {
	if len(pts) == 0 || paint.nothingToDraw() {
		return
	}

	// Enforce paint style matches implicit behavior of drawPoints
	strokePaint := *paint
	strokePaint.Style = StyleStroke

	// Bounds (and the quick-reject) are only computed when the paint carries mask/image filters.
	var bounds *geom.Rect
	if strokePaint.MaskFilter != nil || strokePaint.ImageFilter != nil {
		var r geom.Rect
		if r.SetBounds(pts) {
			if c.internalQuickReject(r, &strokePaint) {
				return
			}
			bounds = &r
		}
	}

	if dp, restore, ok := c.aboutToDraw(&strokePaint, bounds); ok {
		c.topDevice().DrawPoints(mode, pts, dp)
		if restore != nil {
			restore()
		}
	}
}

// DrawPoint draws a single point (its size and shape follow the paint's stroke width and cap).
func (c *Canvas) DrawPoint(x, y float32, paint *Paint) {
	pt := [1]geom.Point{{X: x, Y: y}}
	c.DrawPoints(PointModePoints, pt[:], paint)
}

// DrawLine draws a line segment from (x0, y0) to (x1, y1).
func (c *Canvas) DrawLine(x0, y0, x1, y1 float32, paint *Paint) {
	pts := [2]geom.Point{{X: x0, Y: y0}, {X: x1, Y: y1}}
	c.DrawPoints(PointModeLines, pts[:], paint)
}

// DrawPath draws the path with the paint.
func (c *Canvas) DrawPath(p *path.Path, paint *Paint) {
	if !p.IsFinite() {
		return
	}

	pathBounds := p.Bounds()
	if !p.IsInverseFillType() && c.internalQuickReject(pathBounds, paint) {
		return
	}
	if p.IsInverseFillType() && pathBounds.Width() <= 0 && pathBounds.Height() <= 0 {
		c.internalDrawPaint(paint)
		return
	}

	if dp, restore, ok := c.aboutToDraw(paint, &pathBounds); ok {
		c.topDevice().DrawPath(p, dp)
		if restore != nil {
			restore()
		}
	}
}
