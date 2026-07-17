// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package canvas

import (
	"testing"

	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// gradientImage builds a w x h opaque image with distinct per-pixel values.
func gradientImage(w, h int32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeOpaque)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			p.Words[y*p.RowElems+x] = 0xFF000000 | uint32(x*40) | uint32(y*40)<<8 | 0x30<<16
		}
	}
	return imagecore.FromPixels(p)
}

func TestDrawImageIntegerOffsetIsSprite(t *testing.T) {
	img := gradientImage(3, 3)
	pix := raster.NewPixmap(8, 8)
	c := NewForPixmap(pix)
	c.DrawImage(img, 2, 3, shaders.SamplingOptions{}, nil)

	src := img.PeekPixels(imagecore.CachingAllow)
	for y := int32(0); y < 3; y++ {
		for x := int32(0); x < 3; x++ {
			want := src.Words[y*src.RowElems+x]
			got := pix.Pix[(y+3)*8+(x+2)]
			if got != want {
				t.Fatalf("(%d,%d) = %08x want %08x", x, y, got, want)
			}
		}
	}
	if pix.Pix[0] != 0 || pix.Pix[8*7+7] != 0 {
		t.Fatal("pixels outside the image were written")
	}
}

func TestDrawImageRectScales(t *testing.T) {
	img := gradientImage(2, 2)
	pix := raster.NewPixmap(8, 8)
	c := NewForPixmap(pix)
	c.DrawImageRect(img, geom.RectWH(2, 2), geom.RectWH(8, 8),
		shaders.SamplingOptions{}, nil, ConstraintFast)

	src := img.PeekPixels(imagecore.CachingAllow)
	// Nearest upscale 4x: quadrant (1,1) of the destination shows source pixel (0,0), etc.
	if pix.Pix[1*8+1] != src.Words[0] {
		t.Fatalf("q00 = %08x want %08x", pix.Pix[9], src.Words[0])
	}
	if pix.Pix[1*8+6] != src.Words[1] {
		t.Fatalf("q10 = %08x want %08x", pix.Pix[14], src.Words[1])
	}
	if pix.Pix[6*8+6] != src.Words[3] {
		t.Fatalf("q11 = %08x want %08x", pix.Pix[54], src.Words[3])
	}
}

func TestDrawImageRectSrcSubset(t *testing.T) {
	img := gradientImage(4, 4)
	pix := raster.NewPixmap(4, 4)
	c := NewForPixmap(pix)
	// Draw only the bottom-right 2x2 of the source into the full destination.
	c.DrawImageRect(img, geom.RectLTRB(2, 2, 4, 4), geom.RectWH(4, 4),
		shaders.SamplingOptions{}, nil, ConstraintStrict)

	src := img.PeekPixels(imagecore.CachingAllow)
	if pix.Pix[0] != src.Words[2*4+2] {
		t.Fatalf("subset origin = %08x want %08x", pix.Pix[0], src.Words[10])
	}
	if pix.Pix[3*4+3] != src.Words[3*4+3] {
		t.Fatalf("subset corner = %08x want %08x", pix.Pix[15], src.Words[15])
	}
}

func TestDrawImageAlphaModulates(t *testing.T) {
	img := gradientImage(2, 2)
	pix := raster.NewPixmap(4, 4)
	c := NewForPixmap(pix)
	paint := NewPaint()
	paint.Color = paint.Color.WithAlpha(0x80)
	c.DrawImage(img, 0, 0, shaders.SamplingOptions{}, paint)

	src := img.PeekPixels(imagecore.CachingAllow)
	// The sprite lane blends with a premul lerp of scale 129 (src over dst=0): channel*129>>8.
	want := src.Words[0] & 0xFF * 129 >> 8
	if got := pix.Pix[0] & 0xFF; got != want {
		t.Fatalf("alpha-modulated r = %d want %d", got, want)
	}
}

func TestDrawImageNineGeometry(t *testing.T) {
	// 6x6 image: 2px border of red, 2x2 green center.
	info, _ := imagecore.MakeInfo(6, 6, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeOpaque)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < 6; y++ {
		for x := int32(0); x < 6; x++ {
			col := uint32(0xFF0000FF) // red
			if x >= 2 && x < 4 && y >= 2 && y < 4 {
				col = 0xFF00FF00 // green
			}
			p.Words[y*6+x] = col
		}
	}
	img := imagecore.FromPixels(p)

	pix := raster.NewPixmap(16, 16)
	c := NewForPixmap(pix)
	c.DrawImageNine(img, geom.IRect{Left: 2, Top: 2, Right: 4, Bottom: 4}, geom.RectWH(16, 16),
		shaders.FilterNearest, nil)

	// Corners keep the 2px red border; the stretched center is green.
	if pix.Pix[0] != 0xFF0000FF || pix.Pix[1*16+1] != 0xFF0000FF {
		t.Fatalf("corner = %08x %08x", pix.Pix[0], pix.Pix[17])
	}
	if pix.Pix[8*16+8] != 0xFF00FF00 {
		t.Fatalf("center = %08x", pix.Pix[8*16+8])
	}
	if pix.Pix[15*16+15] != 0xFF0000FF {
		t.Fatalf("far corner = %08x", pix.Pix[15*16+15])
	}
	// Edge midpoints: top edge (stretched horizontally) stays red.
	if pix.Pix[0*16+8] != 0xFF0000FF {
		t.Fatalf("top edge = %08x", pix.Pix[8])
	}
}

func TestDrawImageNineInvalidCenterFallsBack(t *testing.T) {
	img := gradientImage(4, 4)
	pix := raster.NewPixmap(8, 8)
	c := NewForPixmap(pix)
	// Empty center: invalid lattice → drawImageRect of the whole image.
	c.DrawImageNine(img, geom.IRect{Left: 2, Top: 2, Right: 2, Bottom: 2}, geom.RectWH(8, 8),
		shaders.FilterNearest, nil)
	src := img.PeekPixels(imagecore.CachingAllow)
	if pix.Pix[1*8+1] != src.Words[0] {
		t.Fatalf("fallback draw wrong: %08x want %08x", pix.Pix[9], src.Words[0])
	}
}

func TestSaveLayerRestoreWithColorFilter(t *testing.T) {
	// A saveLayer restore paint carrying a color filter must filter the layer contents (the drawSprite shader lane).
	pix := raster.NewPixmap(4, 4)
	c := NewForPixmap(pix)

	restorePaint := NewPaint()
	// SrcIn against green: output = green scaled by source alpha.
	restorePaint.ColorFilter = colorfilter.NewBlend(0xFF00FF00, raster.BlendSrcIn)

	c.SaveLayer(nil, restorePaint)
	red := NewPaint()
	red.Color = 0xFFFF0000
	c.DrawRect(geom.RectWH(4, 4), red)
	c.Restore()

	if got := pix.Pix[5]; got != 0xFF00FF00 {
		t.Fatalf("filtered restore = %08x want FF00FF00 (device word B<<16|G<<8|R)", got)
	}
}

func TestSaveLayerRestoreWithoutFilterUnchanged(t *testing.T) {
	// The plain restore continues through the sprite lane byte-exactly.
	pix := raster.NewPixmap(4, 4)
	c := NewForPixmap(pix)
	c.SaveLayerAlpha(nil, 255)
	red := NewPaint()
	red.Color = 0xFFFF0000
	c.DrawRect(geom.RectWH(4, 4), red)
	c.Restore()
	if got := pix.Pix[5]; got != 0xFF0000FF { // device word: R in byte 0
		t.Fatalf("restore = %08x", got)
	}
}

func TestDrawImageRotatedUsesShaderLane(t *testing.T) {
	img := gradientImage(4, 4)
	pix := raster.NewPixmap(12, 12)
	c := NewForPixmap(pix)
	c.Translate(6, 2)
	c.Rotate(45)
	paint := NewPaint()
	paint.AntiAlias = true
	c.DrawImage(img, 0, 0, shaders.SamplingOptions{Filter: shaders.FilterLinear}, paint)
	// Just verify pixels landed and the unrotated corner stayed clear.
	found := false
	for _, v := range pix.Pix {
		if v != 0 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("rotated draw produced no pixels")
	}
	if pix.Pix[11*12] != 0 {
		t.Fatal("bottom-left corner should be untouched")
	}
}

func TestDrawImageColorFilteredSpriteFallsToShader(t *testing.T) {
	// A color filter forces the sprite lane to fall through to the shader path, which folds the filter into the shader.
	img := gradientImage(2, 2)
	pix := raster.NewPixmap(4, 4)
	c := NewForPixmap(pix)
	paint := NewPaint()
	paint.ColorFilter = colorfilter.NewBlend(0xFFFFFFFF, raster.BlendSrcIn) // white by src alpha
	c.DrawImage(img, 0, 0, shaders.SamplingOptions{}, paint)
	if got := pix.Pix[0]; got != 0xFFFFFFFF {
		t.Fatalf("filtered image draw = %08x want FFFFFFFF", got)
	}
}
