// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package surface implements the raster surface types: owned and borrowed-pixel raster surfaces, the compatible-surface
// factory, snapshots with copy-on-write semantics, surface props, and the canvas↔surface backrefs.
package surface

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
)

// PixelGeometry describes the subpixel layout of an LCD display, for LCD subpixel text rendering.
type PixelGeometry uint8

// PixelGeometry values.
const (
	PixelGeometryUnknown PixelGeometry = iota
	PixelGeometryRGBH
	PixelGeometryBGRH
	PixelGeometryRGBV
	PixelGeometryBGRV
)

// Surface Props flag bits.
const (
	// UseDeviceIndependentFontsFlag requests device-independent (as opposed to pixel-snapped) font rendering.
	UseDeviceIndependentFontsFlag uint32 = 1 << 0
	// DynamicMSAAFlag uses internal MSAA to render to non-MSAA GPU surfaces.
	DynamicMSAAFlag uint32 = 1 << 1
	// AlwaysDitherFlag requests dithering even where it would otherwise be skipped: on the GPU backend every draw with
	// a color fragment processor dithers, whatever the paint's own dither flag says. The raster backend ignores it and
	// dithers from the paint flag alone.
	AlwaysDitherFlag uint32 = 1 << 2
)

// Props holds flags plus pixel geometry. A known geometry enables LCD subpixel text for fonts whose edging is
// font.EdgingSubpixelAntiAlias; UseDeviceIndependentFontsFlag disables it.
type Props struct {
	Flags         uint32
	PixelGeometry PixelGeometry
}

// DeviceProps converts to the font package's device props type (the enums share ordinals).
func (p Props) DeviceProps() font.DeviceProps {
	return font.DeviceProps{
		PixelGeometry:             font.PixelGeometry(p.PixelGeometry),
		UseDeviceIndependentFonts: p.Flags&UseDeviceIndependentFontsFlag != 0,
	}
}

// Surface is a raster rendering surface.
type Surface struct {
	pix      *raster.Pixmap
	canvas   *canvas.Canvas
	snapshot *imagecore.Image // outstanding snapshot sharing pix's backing store
	props    Props
	owns     bool // false for WrapPixels (raster-direct) surfaces
}

func newSurface(pix *raster.Pixmap, props *Props, owns bool) *Surface {
	s := &Surface{pix: pix, owns: owns}
	if props != nil {
		s.props = *props
	}
	s.canvas = canvas.NewForPixmap(pix)
	// The bitmap device receives the surface's props at creation; layers created from this device keep unknown geometry
	// (canvas.BitmapDevice.CreateDevice), matching the canvas's layer-props cloning.
	if dev, ok := s.canvas.BaseDevice().(*canvas.BitmapDevice); ok {
		dev.SetDeviceProps(s.props.DeviceProps())
	}
	s.canvas.SetSurfaceBase(s)
	return s
}

// NewRasterN32Premul creates a zeroed N32-premul surface owning its pixels. Returns nil for non-positive dimensions.
func NewRasterN32Premul(width, height int32, props *Props) *Surface {
	if width <= 0 || height <= 0 {
		return nil
	}
	return newSurface(raster.NewPixmap(width, height), props, true)
}

// WrapPixels creates a surface rendering into caller-provided pixels. The pixmap's backing store is the caller's;
// snapshots always deep-copy.
func WrapPixels(pix *raster.Pixmap, props *Props) *Surface {
	// The backing slice must hold every pixel the strided layout can address. Rendering computes a pixel's index as
	// y*RowPixels+x, so the last valid pixel (Width-1, Height-1) lives at word (Height-1)*RowPixels+Width-1; the slice
	// must therefore span at least (Height-1)*RowPixels+Width words. Validating against Width*Height instead would let a
	// caller-built pixmap with RowPixels > Width pass yet write out of bounds. RowPixels < Width is likewise rejected as
	// a malformed stride.
	if pix == nil || pix.Width <= 0 || pix.Height <= 0 || pix.RowPixels < pix.Width ||
		len(pix.Pix) < int(pix.Height-1)*int(pix.RowPixels)+int(pix.Width) {
		return nil
	}
	return newSurface(pix, props, false)
}

// Canvas returns the canvas that draws into this surface.
func (s *Surface) Canvas() *canvas.Canvas { return s.canvas }

// Props returns the surface props.
func (s *Surface) Props() Props { return s.props }

// Width returns the surface width in pixels.
func (s *Surface) Width() int32 { return s.pix.Width }

// Height returns the surface height in pixels.
func (s *Surface) Height() int32 { return s.pix.Height }

// Pixmap exposes the surface's current backing store (peekPixels).
func (s *Surface) Pixmap() *raster.Pixmap { return s.pix }

// MakeSurface creates a compatible surface (same backing kind and props, fresh pixels).
func (s *Surface) MakeSurface(width, height int32) *Surface {
	return NewRasterN32Premul(width, height, &s.props)
}

// MakeImageSnapshot returns an immutable image of the surface's current contents. An owning surface shares its backing
// store with the image and copies-on-write at the next draw; a borrowed-pixel (raster-direct) surface deep-copies
// immediately, since it does not own its pixels.
func (s *Surface) MakeImageSnapshot() *imagecore.Image {
	if !s.owns {
		return imagecore.Copy(s.pix)
	}
	if s.snapshot == nil {
		s.snapshot = imagecore.FromPixmap(*s.pix)
	}
	return s.snapshot
}

// AboutToDraw implements canvas.SurfaceBase: when a snapshot shares our pixels, move the current backing store to the
// snapshot and continue on a copy. The device's *Pixmap identity is preserved so the canvas keeps drawing into the
// surface.
func (s *Surface) AboutToDraw() {
	if s.snapshot == nil {
		return
	}
	// The snapshot image already holds the old Pix slice (it captured the Pixmap by value); give the surface a fresh
	// copy to mutate, retaining its current contents.
	old := s.pix.Pix
	fresh := make([]uint32, len(old))
	copy(fresh, old)
	s.pix.Pix = fresh
	s.snapshot = nil
}
