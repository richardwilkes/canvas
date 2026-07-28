// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package scenario

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// The drawAtlas corpus: the four atlas scenarios, drawing sprites from a shared procedural atlas through the
// scenario.Canvas DrawAtlas hook. They gate exactly as every other scenario does — the raster lane bit-exactly under
// `exact`, the gpu and gpudmsaa lanes under `exact1` — against this platform's self-captured goldens. The looser
// `gpu` profile bounds only the Go-GPU-vs-Go-CPU self-consistency check (gorender's
// TestAtlasGPUvsCPUSelfConsistency), which compares the port's two live backends against each other rather than
// against a golden, and guards the per-sprite CPU-lowering equivalence claim.

// atlasSize is the shared atlas's dimension (four 32x32 tiles in a 64x64 square).
const atlasSize = 64

var (
	atlasPixelsOnce sync.Once
	atlasPixelsData []byte
)

// AtlasPixels returns the shared procedural 64x64 RGBA8888-premul atlas (tightly packed): four 32x32 tiles — red,
// green, blue, and a translucent white — each with a dark border, an accent circle, and a diagonal shade ramp, so
// nearest-vs-linear sampling and premul handling are both observable. Both backends materialize their native image type
// from these exact bytes. The returned slice is shared; callers must not mutate it.
func AtlasPixels() (w, h int32, rgba []byte) {
	atlasPixelsOnce.Do(func() {
		atlasPixelsData = buildAtlasPixels()
	})
	return atlasSize, atlasSize, atlasPixelsData
}

// buildAtlasPixels fills the atlas procedurally (deterministic integer math only).
func buildAtlasPixels() []byte {
	bases := [4]colorcore.Color{red, green, blue, white}
	accents := [4]colorcore.Color{yellow, purple, teal, gray}
	out := make([]byte, atlasSize*atlasSize*4)
	for y := 0; y < atlasSize; y++ {
		for x := 0; x < atlasSize; x++ {
			tile := (x / 32) + 2*(y/32)
			lx, ly := x%32, y%32
			var c colorcore.Color
			switch dx, dy := lx-16, ly-16; {
			case lx < 2 || ly < 2 || lx >= 30 || ly >= 30:
				c = black
			case dx*dx+dy*dy < 100:
				c = accents[tile]
			default:
				// A diagonal shade ramp over the base color.
				shade := uint32(128 + (lx*3+ly*2)%128)
				base := bases[tile]
				c = colorcore.ARGB(base.A(),
					uint8(uint32(base.R())*shade/255),
					uint8(uint32(base.G())*shade/255),
					uint8(uint32(base.B())*shade/255))
				if tile == 3 {
					c = c.WithAlpha(0xA0) // translucent region: premul handling observable
				}
			}
			pm := c.PreMultiply()
			i := (y*atlasSize + x) * 4
			out[i] = pm.R()
			out[i+1] = pm.G()
			out[i+2] = pm.B()
			out[i+3] = pm.A()
		}
	}
	return out
}

// atlasTiles returns the four 32x32 tile rects (red, green, blue, translucent-white).
func atlasTiles() []geom.Rect {
	return []geom.Rect{
		geom.RectXYWH(0, 0, 32, 32),
		geom.RectXYWH(32, 0, 32, 32),
		geom.RectXYWH(0, 32, 32, 32),
		geom.RectXYWH(32, 32, 32, 32),
	}
}

// atlasSpriteRing builds n sprites on a ring: sprite i is tile i%4, rotated (i+0.35)*(2pi/n) radians and scaled between
// 0.5 and 1.5, anchored at the tile center. The 0.35 phase keeps every sprite away from exact axis alignment: at angles
// that are exact multiples of pi/2, nearest sampling lands exactly on texel boundaries across whole sprite columns,
// where the port's forward-matrix-then-invert coordinate derivation and the C++ vertices lane's PolyToPoly derivation
// round differently by 1 ulp and flip entire texel runs — a measure-zero degeneracy the differential should not sit on
// (the per-sprite equivalence is tolerance-level, not boundary-exact).
func atlasSpriteRing(n int, cx, cy, radius float32) (xforms []geom.RSXform, tex []geom.Rect) {
	tiles := atlasTiles()
	xforms = make([]geom.RSXform, n)
	tex = make([]geom.Rect, n)
	for i := range n {
		angle := (float32(i) + 0.35) * 2 * math.Pi / float32(n)
		scale := 0.5 + float32(i%5)*0.25
		x := cx + radius*geom.ScalarCos(angle)
		y := cy + radius*geom.ScalarSin(angle)
		xforms[i] = geom.RSXformFromRadians(scale, angle, x, y, 16, 16)
		tex[i] = tiles[i%4]
	}
	return xforms, tex
}

// atlasDraw issues one DrawAtlas over the shared atlas with the standard paint (AA flag is irrelevant — the lowering is
// non-AA — but set to mirror typical callers).
func atlasDraw(c Canvas, xforms []geom.RSXform, tex []geom.Rect, colors []colorcore.Color, mode BlendMode, linear bool, cull *geom.Rect, p Paint) {
	w, h, rgba := AtlasPixels()
	c.DrawAtlas(w, h, rgba, xforms, tex, colors, mode, linear, cull, p)
}

func init() {
	// atlas-basic: no per-sprite colors; rotation+scale xforms; the left ring samples nearest, the right ring linear,
	// so the sampling plumbing of both backends is compared in one image.
	reg("atlas-basic", func(c Canvas) {
		c.Clear(white)
		xf, tex := atlasSpriteRing(8, 68, 128, 52)
		atlasDraw(c, xf, tex, nil, BlendSrcOver, false, nil, NewPaint())
		xf2, tex2 := atlasSpriteRing(8, 188, 128, 52)
		atlasDraw(c, xf2, tex2, nil, BlendSrcOver, true, nil, NewPaint())
	})

	// atlas-colors-modulate: per-sprite colors through kModulate (the classic sprite-tint mode).
	reg("atlas-colors-modulate", func(c Canvas) {
		c.Clear(white)
		xf, tex := atlasSpriteRing(12, 128, 128, 76)
		colors := make([]colorcore.Color, len(xf))
		for i := range colors {
			colors[i] = colorcore.ARGB(0xFF, uint8(64+i*16), uint8(255-i*12), uint8(32+i*7))
		}
		atlasDraw(c, xf, tex, colors, BlendModulate, true, nil, NewPaint())
	})

	// atlas-blend: a non-default blend mode combining the sprite colors with the texels (kScreen), under a translucent
	// paint alpha, over a non-white ground so the paint-alpha compositing is observable; a few translucent sprite
	// colors exercise the alpha product.
	reg("atlas-blend", func(c Canvas) {
		c.Clear(teal)
		xf, tex := atlasSpriteRing(10, 128, 128, 68)
		colors := make([]colorcore.Color, len(xf))
		for i := range colors {
			a := uint8(0xFF)
			if i%3 == 0 {
				a = 0x90
			}
			colors[i] = colorcore.ARGB(a, uint8(40+i*20), uint8(20+i*11), uint8(200-i*13))
		}
		p := NewPaint()
		p.Color = colorcore.Color(0x80FFFFFF) // translucent paint alpha modulates the sprites
		atlasDraw(c, xf, tex, colors, BlendScreen, true, nil, p)
	})

	// atlas-cull: the cull-rect reject path. The first call's cull rect lies entirely outside the clip, so it must draw
	// nothing on either backend; the second call's cull covers its sprites and draws. The result shows only the second
	// ring.
	reg("atlas-cull", func(c Canvas) {
		c.Clear(white)
		c.ClipRect(geom.RectLTRB(0, 0, 256, 256), ClipIntersect, false)
		xf, tex := atlasSpriteRing(6, 128, 128, 40)
		rejected := geom.RectXYWH(1000, 1000, 64, 64)
		atlasDraw(c, xf, tex, nil, BlendSrcOver, false, &rejected, NewPaint())
		xf2, tex2 := atlasSpriteRing(6, 128, 128, 88)
		accepted := geom.RectXYWH(0, 0, 256, 256)
		atlasDraw(c, xf2, tex2, nil, BlendSrcOver, false, &accepted, NewPaint())
	})
}
