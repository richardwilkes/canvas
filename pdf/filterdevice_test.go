// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pdf

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagefilter"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// TestDeviceColorFilterLayerDrawsBackAsImage exercises the color-filter saveLayer lane: a layer whose paint carries a
// color filter renders on a raster device and is drawn back as a color-filtered Image XObject. Before this lane the
// color filter was dropped and the layer composited as a plain transparency group.
func TestDeviceColorFilterLayerDrawsBackAsImage(t *testing.T) {
	lp := canvas.NewPaint()
	lp.ColorFilter = colorfilter.NewBlend(colorcore.ARGB(255, 0, 0, 255), raster.BlendSrc)
	data := renderPDFUncompressed(t, 40, 40, func(c *canvas.Canvas) {
		c.SaveLayer(nil, lp)
		red := canvas.NewPaint()
		red.Color = colorcore.ARGB(255, 255, 0, 0)
		c.DrawRect(geom.RectLTRB(10, 10, 30, 30), red)
		c.Restore()
	})
	validatePDF(t, data)
	// The raster layer is drawn back as an Image XObject, not a vector transparency group.
	if n := countSubstr(data, "/Subtype /Image"); n < 1 {
		t.Fatalf("color-filter layer not drawn back as an image (Image XObjects=%d)", n)
	}
	// The color filter is baked into the image, not dropped: a Src(blue) filter makes the layer blue.
	if !bytes.Contains(data, []byte{0x00, 0x00, 0xFF}) {
		t.Error("color filter was not baked into the layer image (no blue pixels)")
	}
	// The layer geometry was rasterized, not re-emitted as vector ops in a transparency group.
	if anyStreamContains(data, "10 10 20 20 re") {
		t.Error("layer rect leaked as vector geometry; expected a rasterized image")
	}
}

// TestDeviceImageFilterLayerDrawsFilteredImageBack exercises the image-filter saveLayer lane: a layer whose paint
// carries an image filter renders on a raster device; the filter DAG evaluates on the CPU raster backend
// (CreateFilterBackend) and the resolved result is drawn back as an Image XObject through the PDF device's
// AsFilterDevice adapter (DrawSpecial). Before this lane image-filter layers were dropped entirely.
func TestDeviceImageFilterLayerDrawsFilteredImageBack(t *testing.T) {
	lp := canvas.NewPaint()
	lp.ImageFilter = imagefilter.Blur(3, 3, shaders.TileDecal, nil, nil)
	data := renderPDF(t, 60, 60, func(c *canvas.Canvas) {
		c.SaveLayer(nil, lp)
		red := canvas.NewPaint()
		red.Color = colorcore.ARGB(255, 255, 0, 0)
		c.DrawRect(geom.RectLTRB(20, 20, 40, 40), red)
		c.Restore()
	})
	validatePDF(t, data)
	if n := countSubstr(data, "/Subtype /Image"); n < 1 {
		t.Fatalf("image-filter layer produced no Image XObject (dropped?); got %d", n)
	}
	if !anyStreamContains(data, " Do\n") {
		t.Error("the filtered image XObject is not invoked in any content stream")
	}
	// The layer rect was rendered into the raster layer and blurred, not re-emitted as vector ops.
	if anyStreamContains(data, "20 20 20 20 re") {
		t.Error("layer rect leaked as vector geometry; expected a filtered raster image")
	}
}

// TestDeviceColorFilterLayerAppliesAlpha confirms the color-filter layer draw-back honors the layer paint's alpha: the
// raster layer is drawn back through internalDrawImageRect's scoped content entry, which emits the /ca ExtGState (the
// image-filter path instead bakes alpha into the image via a DstIn color filter, so /ca is the color-filter draw-back's
// distinctive signature).
func TestDeviceColorFilterLayerAppliesAlpha(t *testing.T) {
	lp := canvas.NewPaint()
	lp.ColorFilter = colorfilter.NewBlend(colorcore.ARGB(255, 0, 0, 255), raster.BlendSrc)
	lp.Color = lp.Color.WithAlpha(128)
	data := renderPDF(t, 40, 40, func(c *canvas.Canvas) {
		c.SaveLayer(nil, lp)
		red := canvas.NewPaint()
		red.Color = colorcore.ARGB(255, 255, 0, 0)
		c.DrawRect(geom.RectLTRB(10, 10, 30, 30), red)
		c.Restore()
	})
	validatePDF(t, data)
	if n := countSubstr(data, "/Subtype /Image"); n < 1 {
		t.Fatalf("color-filter layer produced no Image XObject; got %d", n)
	}
	if !anyObjectContains(data, "/ca .502") {
		t.Errorf("layer alpha (~0.5) not applied to the color-filter layer draw-back; objects=%v",
			dictObjects(data))
	}
}
