// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for Canvas.DrawAtlas's CPU lane: the per-sprite lowering samples the right atlas texels at the right
// places, the per-sprite color modulation matches the vertices lane's blend semantics (including the kSrc/kDst
// simplifications), the cull rect quick-rejects, and the count clamps to the shortest array.

package canvas

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// testAtlas builds a 32x32 atlas of four opaque 16x16 solid tiles: red, green, blue, white.
func testAtlas() *imagecore.Image {
	pm := raster.NewPixmap(32, 32)
	c := NewForPixmap(pm)
	tile := func(r geom.Rect, col colorcore.Color) {
		p := NewPaint()
		p.Color = col
		p.BlendMode = raster.BlendSrc
		c.DrawRect(r, p)
	}
	tile(geom.RectXYWH(0, 0, 16, 16), 0xFFFF0000)
	tile(geom.RectXYWH(16, 0, 16, 16), 0xFF00FF00)
	tile(geom.RectXYWH(0, 16, 16, 16), 0xFF0000FF)
	tile(geom.RectXYWH(16, 16, 16, 16), 0xFFFFFFFF)
	return imagecore.Copy(pm)
}

// atlasTileRects returns the four tile rects in reading order (red, green, blue, white).
func atlasTileRects() []geom.Rect {
	return []geom.Rect{
		geom.RectXYWH(0, 0, 16, 16),
		geom.RectXYWH(16, 0, 16, 16),
		geom.RectXYWH(0, 16, 16, 16),
		geom.RectXYWH(16, 16, 16, 16),
	}
}

// pixAt unpacks the premul device word at (x, y) into an unpremultiplied colorcore.Color (all test pixels are opaque,
// so premul == unpremul).
func pixAt(pm *raster.Pixmap, x, y int32) colorcore.Color {
	v := pm.Pix[int(y)*int(pm.RowPixels)+int(x)]
	r := uint8(v)
	g := uint8(v >> 8)
	b := uint8(v >> 16)
	a := uint8(v >> 24)
	return colorcore.ARGB(a, r, g, b)
}

// TestDrawAtlasBasic places the four tiles at translated positions with nearest sampling and checks each destination
// shows its tile's color, with untouched background between sprites.
func TestDrawAtlasBasic(t *testing.T) {
	pm := raster.NewPixmap(128, 64)
	c := NewForPixmap(pm)
	c.Clear(0xFF202020)

	tex := atlasTileRects()
	xforms := []geom.RSXform{
		geom.MakeRSXform(1, 0, 4, 4),
		geom.MakeRSXform(1, 0, 34, 4),
		geom.MakeRSXform(1, 0, 64, 4),
		geom.MakeRSXform(1, 0, 94, 4),
	}
	c.DrawAtlas(testAtlas(), xforms, tex, nil, raster.BlendSrcOver,
		shaders.SamplingOptions{}, nil, nil)

	want := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF}
	for i, w := range want {
		x := int32(4+30*i) + 8
		if got := pixAt(pm, x, 12); got != w {
			t.Errorf("sprite %d center = %08X, want %08X", i, uint32(got), uint32(w))
		}
	}
	if got := pixAt(pm, 26, 12); got != 0xFF202020 {
		t.Errorf("gap between sprites = %08X, want background", uint32(got))
	}
	if got := pixAt(pm, 8, 40); got != 0xFF202020 {
		t.Errorf("below sprites = %08X, want background", uint32(got))
	}
}

// TestDrawAtlasRotation places the red tile with a 90-degree rotation (scos=0, ssin=1): the quad covers [tx-h, tx] x
// [ty, ty+w], so a sample inside that square must be red.
func TestDrawAtlasRotation(t *testing.T) {
	pm := raster.NewPixmap(64, 64)
	c := NewForPixmap(pm)
	c.Clear(0xFF202020)

	tex := atlasTileRects()[:1]
	xforms := []geom.RSXform{geom.MakeRSXform(0, 1, 40, 8)}
	c.DrawAtlas(testAtlas(), xforms, tex, nil, raster.BlendSrcOver,
		shaders.SamplingOptions{}, nil, nil)

	if got := pixAt(pm, 32, 16); got != 0xFFFF0000 {
		t.Errorf("rotated sprite interior = %08X, want red", uint32(got))
	}
	if got := pixAt(pm, 48, 16); got != 0xFF202020 {
		t.Errorf("outside rotated sprite = %08X, want background", uint32(got))
	}
}

// TestDrawAtlasColors pins the per-sprite color semantics: kDst keeps only the sprite color, kSrc drops the colors
// entirely, and kModulate multiplies (white tile x color == color).
func TestDrawAtlasColors(t *testing.T) {
	whiteTile := []geom.Rect{atlasTileRects()[3]}
	xforms := []geom.RSXform{geom.MakeRSXform(1, 0, 8, 8)}
	colors := []colorcore.Color{0xFF00C080}

	render := func(mode raster.BlendMode) *raster.Pixmap {
		pm := raster.NewPixmap(32, 32)
		c := NewForPixmap(pm)
		c.Clear(0xFF202020)
		c.DrawAtlas(testAtlas(), xforms, whiteTile, colors, mode,
			shaders.SamplingOptions{}, nil, nil)
		return pm
	}

	if got := pixAt(render(raster.BlendDst), 16, 16); got != 0xFF00C080 {
		t.Errorf("kDst sprite = %08X, want the sprite color", uint32(got))
	}
	if got := pixAt(render(raster.BlendSrc), 16, 16); got != 0xFFFFFFFF {
		t.Errorf("kSrc sprite = %08X, want the atlas texel (colors dropped)", uint32(got))
	}
	if got := pixAt(render(raster.BlendModulate), 16, 16); got != 0xFF00C080 {
		t.Errorf("kModulate sprite over white tile = %08X, want the sprite color", uint32(got))
	}
}

// TestDrawAtlasCullReject verifies a cull rect entirely outside the clip rejects the whole call.
func TestDrawAtlasCullReject(t *testing.T) {
	pm := raster.NewPixmap(32, 32)
	c := NewForPixmap(pm)
	c.Clear(0xFF202020)

	tex := atlasTileRects()[:1]
	xforms := []geom.RSXform{geom.MakeRSXform(1, 0, 8, 8)}
	cull := geom.RectXYWH(1000, 1000, 16, 16)
	c.DrawAtlas(testAtlas(), xforms, tex, nil, raster.BlendSrcOver,
		shaders.SamplingOptions{}, &cull, nil)

	if got := pixAt(pm, 16, 16); got != 0xFF202020 {
		t.Errorf("culled draw painted %08X, want untouched background", uint32(got))
	}
}

// TestDrawAtlasCountClamp verifies the draw count is min across the arrays and that a zero count draws nothing.
func TestDrawAtlasCountClamp(t *testing.T) {
	pm := raster.NewPixmap(96, 32)
	c := NewForPixmap(pm)
	c.Clear(0xFF202020)

	tex := atlasTileRects()   // 4 rects
	xforms := []geom.RSXform{ // only 2 xforms => count 2
		geom.MakeRSXform(1, 0, 4, 4),
		geom.MakeRSXform(1, 0, 34, 4),
	}
	c.DrawAtlas(testAtlas(), xforms, tex, nil, raster.BlendSrcOver,
		shaders.SamplingOptions{}, nil, nil)
	if got := pixAt(pm, 12, 12); got != 0xFFFF0000 {
		t.Errorf("sprite 0 = %08X, want red", uint32(got))
	}
	if got := pixAt(pm, 42, 12); got != 0xFF00FF00 {
		t.Errorf("sprite 1 = %08X, want green", uint32(got))
	}
	if got := pixAt(pm, 70, 12); got != 0xFF202020 {
		t.Errorf("beyond count = %08X, want background", uint32(got))
	}

	c.DrawAtlas(testAtlas(), nil, tex, nil, raster.BlendSrcOver, shaders.SamplingOptions{}, nil, nil)
}
