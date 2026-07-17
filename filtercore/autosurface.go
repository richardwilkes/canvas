// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// autoSurface manages a device for drawing to a layer-space bounding box and snapping it into a FilterResult, including
// the 1px boundary padding used to upgrade decal sampling to clamp sampling. Backends hand out zero-initialized
// (transparent) devices, so no explicit clear is needed here.

package filtercore

import (
	"github.com/richardwilkes/canvas/geom"
)

type autoSurface struct {
	ctx       *Context
	device    Device
	dstBounds geom.IRect // includes padding, if any
	boundary  pixelBoundary
	valid     bool
}

// newAutoSurface builds an autoSurface over dstBounds. The dstBounds are not intersected with the context's desired
// output — callers have already accounted for it. renderInParameterSpace concats the context's layer matrix so draws
// take parameter-space geometry.
func newAutoSurface(ctx *Context, dstBounds geom.IRect, boundary pixelBoundary, renderInParameterSpace bool) autoSurface {
	s := autoSurface{ctx: ctx, dstBounds: dstBounds, boundary: boundary}
	if dstBounds.IsEmpty() {
		return s
	}
	padding := s.padding()
	if padding != 0 {
		padded := s.dstBounds.Inset(-padding, -padding)
		// With pathological inputs near the int32 limits the outset saturates and the padding pixels don't exist; fail
		// rather than downgrade the boundary.
		if padded.Left >= dstBounds.Left || padded.Right <= dstBounds.Right ||
			padded.Top >= dstBounds.Top || padded.Bottom <= dstBounds.Bottom {
			return s
		}
		s.dstBounds = padded
	}
	device := ctx.Backend().MakeDevice(s.dstBounds.Size())
	if device == nil {
		return s
	}
	ctx.MarkNewSurface()

	// Configure the device's origin (canvas translate(-left, -top)) and clip.
	var m geom.Matrix
	m.SetTranslate(float32(-s.dstBounds.Left), float32(-s.dstBounds.Top))
	if renderInParameterSpace {
		layer := ctx.Mapping().LayerMatrix()
		m.PreConcat(&layer)
	}
	device.SetLocalToDevice(m)
	if s.boundary == boundaryTransparent {
		// Clip to the original un-padded dst bounds so the border pixels stay fully transparent.
		device.ClipIRect(dstBounds.Offset(-s.dstBounds.Left, -s.dstBounds.Top))
	}
	// Otherwise the device is exact-fit to the (possibly padded) dst bounds already, so an explicit clip to those
	// bounds would be a no-op (MakeDevice has no approx-fit lane).

	s.device = device
	s.valid = true
	return s
}

func (s *autoSurface) padding() int32 {
	if s.boundary == boundaryUnknown {
		return 0
	}
	return 1
}

// drawPaint fills the surface's clip with the paint (in layer coordinates via the device matrix).
func (s *autoSurface) drawPaint(pp *PaintParams) {
	s.device.DrawPaint(pp)
}

// drawRect draws r (layer coordinates) with the paint.
func (s *autoSurface) drawRect(r geom.Rect, pp *PaintParams) {
	s.device.DrawRect(r, pp)
}

// drawRectWithCTM draws r with the device's matrix pre-concatenated by extra, restoring the original matrix afterward
// (like a canvas save/concat/drawRect/restore sequence).
func (s *autoSurface) drawRectWithCTM(r geom.Rect, extra *geom.Matrix, pp *PaintParams) {
	prior := s.device.LocalToDevice()
	m := prior
	m.PreConcat(extra)
	s.device.SetLocalToDevice(m)
	s.device.DrawRect(r, pp)
	s.device.SetLocalToDevice(prior)
}

// snap finalizes the surface into a FilterResult, consuming the autoSurface (it may only be used once).
func (s *autoSurface) snap() FilterResult {
	if !s.valid {
		return FilterResult{}
	}
	// Snap the device with the padded dst bounds.
	image := s.device.SnapSpecial(geom.IRectWH(s.dstBounds.Width(), s.dstBounds.Height()))
	s.valid = false // only use the AutoSurface once
	if image != nil && s.boundary != boundaryUnknown {
		// Inset the subset relative to the image's reported size.
		padding := s.padding()
		subset := geom.IRectSize(image.Dimensions()).Inset(padding, padding)
		origin := geom.IPoint{X: s.dstBounds.Left + padding, Y: s.dstBounds.Top + padding}
		return newFilterResult(image.MakeSubset(subset), origin, s.boundary)
	}
	return newFilterResult(image, geom.IPoint{X: s.dstBounds.Left, Y: s.dstBounds.Top},
		boundaryUnknown)
}
