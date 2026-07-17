// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Image helpers for the PDF device's soft-mask and color-filter draw lanes: converting an A8 coverage mask (or an
// alpha-only image) into an opaque Gray8 image the luminosity SMask form XObject draws, and applying a color filter to
// an image by rendering it through a raster surface. The JPEG-encode fast path for masks is not implemented, so mask
// conversion always takes the raster-from-pixmap lane.

package pdf

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/surface"
)

// maskToGreyscaleImage wraps an A8 coverage mask as an opaque Gray8 image (each coverage byte becomes the grey level).
// The mask's storage is copied into the image.
func maskToGreyscaleImage(mask *raster.Mask) *imagecore.Image {
	info := imagecore.ImageInfo{
		Width:     mask.Bounds.Width(),
		Height:    mask.Bounds.Height(),
		ColorType: imagecore.ColorTypeGray8,
		AlphaType: imagecore.AlphaTypeOpaque,
	}
	return imagecore.NewRasterData(info, mask.Image, int(mask.RowBytes))
}

// alphaImageToGreyscaleImage reads an alpha-only image's coverage as A8 into a fresh buffer and reinterprets that
// buffer as an opaque Gray8 image (grey = alpha).
func alphaImageToGreyscaleImage(mask *imagecore.Image) *imagecore.Image {
	if mask == nil {
		return nil
	}
	w := mask.Width()
	h := mask.Height()
	rowBytes := int(w)
	buf := make([]byte, rowBytes*int(h))
	a8 := imagecore.ImageInfo{
		Width:     w,
		Height:    h,
		ColorType: imagecore.ColorTypeAlpha8,
		AlphaType: imagecore.AlphaTypePremul,
	}
	if !mask.ReadPixels(a8, buf, rowBytes, 0, 0, imagecore.CachingAllow) {
		return nil
	}
	grey := imagecore.ImageInfo{
		Width:     w,
		Height:    h,
		ColorType: imagecore.ColorTypeGray8,
		AlphaType: imagecore.AlphaTypeOpaque,
	}
	return imagecore.NewRasterData(grey, buf, rowBytes)
}

// colorFilterImage applies a color filter to an image by drawing it (color-filtered) into a transparent N32-premul
// raster surface and snapshotting the result.
func colorFilterImage(img *imagecore.Image, cf shaders.ColorFilter) *imagecore.Image {
	if img == nil {
		return nil
	}
	surf := surface.NewRasterN32Premul(img.Width(), img.Height(), nil)
	if surf == nil {
		return nil
	}
	c := surf.Canvas()
	c.Clear(colorcore.Color(0)) // transparent
	p := canvas.NewPaint()
	p.ColorFilter = cf
	c.DrawImage(img, 0, 0, shaders.SamplingOptions{}, p)
	return surf.MakeImageSnapshot()
}
