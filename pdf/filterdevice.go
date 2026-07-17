// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The PDF device's saveLayer raster-layer draw-back lanes. PDF content cannot express an image filter, nor a color
// filter applied to a transparency group, so a layer whose paint carries either renders into a raster BitmapDevice
// (canvas.Device.CreateDevice) and the result is drawn back onto this PDF device as an Image XObject at restore:
//
//   - a color-filter layer (no image filter) restores through DrawDevice, which routes here to drawRasterLayerBack —
//     the layer's pixels drawn back via internalDrawImageRect, which bakes the color filter into the image and applies
//     the paint alpha/blend;
//   - an image-filter layer restores through the canvas's internalDrawDeviceWithFilter, which snaps the raster layer,
//     evaluates the whole filter DAG on the CPU raster backend (CreateFilterBackend), and draws the resolved result
//     back through this filterDevice adapter's DrawSpecial (again internalDrawImageRect at the given device matrix).
//
// Filtered layers are always rasterized first and then embedded as pixels.

package pdf

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// drawRasterLayerBack draws a raster saveLayer device (CreateDevice's color-filter/image-filter lane) back onto this
// PDF device as an Image XObject: this is the fallback taken whenever the saveLayer's device turns out to hold raster
// pixels rather than PDF content. The layer paint's color filter is baked into the image by internalDrawImageRect; its
// alpha and blend mode apply through the scoped content entry. The relative transform is the (integer-translation)
// layer→page mapping, so the WxH layer image lands at the layer's origin.
func (d *Device) drawRasterLayerBack(bd *canvas.BitmapDevice, paint *canvas.Paint) {
	special := bd.SnapSpecial(geom.IRectWH(bd.Width(), bd.Height()))
	if special == nil {
		return
	}
	img := special.AsImage()
	if img == nil {
		return
	}
	matrix := bd.RelativeTransform(d)
	dst := geom.RectWH(float32(img.Width()), float32(img.Height()))
	d.internalDrawImageRect(newKeyedImage(img), nil, dst, paint, matrix)
}

// filterDevice adapts a PDF Device to filtercore.Device: the intermediate image-filter DAG evaluates on CPU raster
// devices (CreateFilterBackend returns the raster backend), so this adapter only bridges the final composite — drawing
// the resolved raster result back as a PDF Image XObject (DrawSpecial) — plus the shader-tiling DrawPaint/DrawRect
// lanes used for layer-filling effects. SnapSpecial returns nil: a PDF device has no pixels to snap, and it is never a
// filter source (its image-filter layers are the raster BitmapDevice, which is the source).
type filterDevice struct {
	dev *Device
}

var _ filtercore.Device = (*filterDevice)(nil)

// Size implements filtercore.Device.
func (f *filterDevice) Size() geom.ISize { return f.dev.pageSize }

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
	f.dev.clipStack.ClipRect(r.ToRect(), &identity, raster.ClipIntersect, false)
}

// DrawPaint implements filtercore.Device.
func (f *filterDevice) DrawPaint(pp *filtercore.PaintParams) {
	f.dev.DrawPaint(canvas.PaintFromFilterParams(pp))
}

// DrawRect implements filtercore.Device.
func (f *filterDevice) DrawRect(r geom.Rect, pp *filtercore.PaintParams) {
	f.dev.DrawRect(r, canvas.PaintFromFilterParams(pp))
}

// DrawSpecial implements filtercore.Device: the filtered raster image is embedded as an Image XObject placed by
// deviceMatrix (which already folds in the layer→device transform; the PDF device's own local-to-device is ignored,
// since internalDrawImageRect takes an explicit ctm argument). The image is the special image's subset view, so it maps
// directly.
func (f *filterDevice) DrawSpecial(img *filtercore.SpecialImage, deviceMatrix geom.Matrix, _ shaders.SamplingOptions, pp *filtercore.PaintParams) {
	image := img.AsImage()
	if image == nil {
		return
	}
	dst := geom.RectWH(float32(image.Width()), float32(image.Height()))
	f.dev.internalDrawImageRect(newKeyedImage(image), nil, dst, canvas.PaintFromFilterParams(pp),
		deviceMatrix)
}

// DrawImageRect implements filtercore.Device (the strict-constraint image draw the image-source leaf and the tiling
// lanes use). The PDF content stream embeds CPU pixels, so a texture backing resolves first (identity for raster
// images).
func (f *filterDevice) DrawImageRect(img imagecore.DrawableImage, src, dst geom.Rect, sampling shaders.SamplingOptions, pp *filtercore.PaintParams) {
	ci := img.MakeNonTextureImage()
	if ci == nil {
		return
	}
	f.dev.DrawImageRect(ci, &src, dst, sampling, canvas.PaintFromFilterParams(pp),
		canvas.ConstraintStrict)
}

// SnapSpecial implements filtercore.Device: a PDF device has no pixels to snap.
func (f *filterDevice) SnapSpecial(_ geom.IRect) *filtercore.SpecialImage { return nil }
