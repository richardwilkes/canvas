// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

func newTestTriangle() *path.Path {
	p := &path.Path{}
	p.MoveTo(5, 34)
	p.LineTo(20, 6)
	p.LineTo(34, 30)
	p.Close()
	return p
}

func lineAsPath(a, b geom.Point) *path.Path {
	p := &path.Path{}
	p.MoveToPt(a)
	p.LineToPt(b)
	return p
}

func newWhitePixmap(w, h int32) *Pixmap {
	pix := NewPixmap(w, h)
	for i := range pix.Pix {
		pix.Pix[i] = 0xFFFFFFFF
	}
	return pix
}

func fullClip(w, h int32) *Clip {
	return NewRasterClipRect(geom.IRectWH(w, h))
}

// TestAntiFillRectIntegralMatchesBW verifies that an integral rect fills identically through the AA (FDot8) and non-AA
// converters — every boundary is on a pixel edge, so coverage is all-or-nothing.
func TestAntiFillRectIntegralMatchesBW(t *testing.T) {
	r := geom.RectLTRB(3, 4, 17, 12)
	aa := newWhitePixmap(24, 16)
	bw := newWhitePixmap(24, 16)
	clip := fullClip(24, 16)
	color := colorcore.Color(0xFF336699)

	AntiFillRect(r, clip, NewSolidBlitter(aa, color))
	FillRectRasterClip(r, clip, NewSolidBlitter(bw, color))

	for i := range aa.Pix {
		if aa.Pix[i] != bw.Pix[i] {
			t.Fatalf("pixel %d: AA %08x BW %08x", i, aa.Pix[i], bw.Pix[i])
		}
	}
}

// TestAntiFillRectHalfPixelCoverage checks the FDot8 half-pixel coverage: a rect starting at x=3.5 blends its left
// column at coverage 128.
func TestAntiFillRectHalfPixelCoverage(t *testing.T) {
	dev := newWhitePixmap(24, 16)
	clip := fullClip(24, 16)
	AntiFillRect(geom.RectLTRB(3.5, 4, 17, 12), clip, NewSolidBlitter(dev, 0xFF000000))

	// Column 3 gets coverage 256-128 = 128 through blitV; compare against a direct BlitV at alpha 128 (the legacy
	// blitter's black fast-path kernel).
	ref := newWhitePixmap(24, 16)
	NewSolidBlitter(ref, 0xFF000000).BlitV(3, 6, 1, 128)
	got := dev.Pix[dev.addr(3, 6)]
	want := ref.Pix[ref.addr(3, 6)]
	if got != want {
		t.Fatalf("half-pixel column: got %08x want %08x", got, want)
	}
	if dev.Pix[dev.addr(4, 6)] != 0xFF000000 {
		t.Fatalf("interior not opaque: %08x", dev.Pix[dev.addr(4, 6)])
	}
	if dev.Pix[dev.addr(2, 6)] != 0xFFFFFFFF {
		t.Fatalf("outside touched: %08x", dev.Pix[dev.addr(2, 6)])
	}
}

// TestHairLineHorizontal verifies the non-AA hairline DDA draws exactly one row with FDot6-rounded endpoints.
func TestHairLineHorizontal(t *testing.T) {
	dev := newWhitePixmap(60, 20)
	clip := fullClip(60, 20)
	pts := []geom.Point{{X: 5, Y: 10.3}, {X: 50, Y: 10.3}}
	HairLine(pts, clip, NewSolidBlitter(dev, 0xFF000000))

	for x := int32(5); x < 50; x++ {
		if dev.Pix[dev.addr(x, 10)] != 0xFF000000 {
			t.Fatalf("x=%d not drawn", x)
		}
	}
	if dev.Pix[dev.addr(4, 10)] != 0xFFFFFFFF || dev.Pix[dev.addr(50, 10)] != 0xFFFFFFFF {
		t.Fatalf("endpoints wrong: %08x %08x", dev.Pix[dev.addr(4, 10)], dev.Pix[dev.addr(50, 10)])
	}
	for y := int32(0); y < 20; y++ {
		if y != 10 && dev.Pix[dev.addr(20, y)] != 0xFFFFFFFF {
			t.Fatalf("row %d touched", y)
		}
	}
}

// TestAntiHairLineCoverageSplit verifies the AA hairline's two-row coverage split for a horizontal line on a pixel
// boundary: alphas 128 (lower) and 127 (upper).
func TestAntiHairLineCoverageSplit(t *testing.T) {
	dev := newWhitePixmap(60, 20)
	clip := fullClip(60, 20)
	pts := []geom.Point{{X: 5, Y: 10}, {X: 50, Y: 10}}
	AntiHairLine(pts, clip, NewSolidBlitter(dev, 0xFF000000))

	ref := newWhitePixmap(60, 20)
	refBlit := NewSolidBlitter(ref, 0xFF000000)
	refBlit.BlitV(20, 15, 1, 128)
	refBlit.BlitV(20, 16, 1, 127)

	lower := dev.Pix[dev.addr(20, 10)]
	upper := dev.Pix[dev.addr(20, 9)]
	if want := ref.Pix[ref.addr(20, 15)]; lower != want {
		t.Fatalf("lower row: got %08x want %08x", lower, want)
	}
	if want := ref.Pix[ref.addr(20, 16)]; upper != want {
		t.Fatalf("upper row: got %08x want %08x", upper, want)
	}
}

// TestHairRectSmall verifies HairRect's interior-stroke semantics and the ≤2-wide collapse.
func TestHairRectSmall(t *testing.T) {
	dev := newWhitePixmap(20, 20)
	clip := fullClip(20, 20)
	HairRect(geom.RectLTRB(3, 3, 10, 9), clip, NewSolidBlitter(dev, 0xFF000000))

	// Enclosing bounds are [3,3..11,10); border pixels drawn, interior untouched.
	if dev.Pix[dev.addr(3, 3)] != 0xFF000000 || dev.Pix[dev.addr(10, 9)] != 0xFF000000 {
		t.Fatalf("corners not drawn")
	}
	if dev.Pix[dev.addr(5, 5)] != 0xFFFFFFFF {
		t.Fatalf("interior drawn")
	}
	if dev.Pix[dev.addr(2, 3)] != 0xFFFFFFFF || dev.Pix[dev.addr(11, 3)] != 0xFFFFFFFF {
		t.Fatalf("outside drawn")
	}
}

// TestFrameRectMatchesFourFills verifies FrameRect equals the union of its four filled edge rects.
func TestFrameRectMatchesFourFills(t *testing.T) {
	r := geom.RectLTRB(5, 5, 30, 20)
	stroke := geom.Point{X: 4, Y: 4}
	clip := fullClip(40, 30)

	frame := newWhitePixmap(40, 30)
	FrameRect(r, stroke, clip, NewSolidBlitter(frame, 0xFF000000))

	manual := newWhitePixmap(40, 30)
	blit := NewSolidBlitter(manual, 0xFF000000)
	outer := geom.RectLTRB(r.Left-2, r.Top-2, r.Right+2, r.Bottom+2)
	FillRectRasterClip(geom.RectLTRB(outer.Left, outer.Top, outer.Right, outer.Top+4), clip, blit)
	FillRectRasterClip(geom.RectLTRB(outer.Left, outer.Bottom-4, outer.Right, outer.Bottom), clip, blit)
	FillRectRasterClip(geom.RectLTRB(outer.Left, outer.Top+4, outer.Left+4, outer.Bottom-4), clip, blit)
	FillRectRasterClip(geom.RectLTRB(outer.Right-4, outer.Top+4, outer.Right, outer.Bottom-4), clip, blit)

	for i := range frame.Pix {
		if frame.Pix[i] != manual.Pix[i] {
			t.Fatalf("pixel %d: frame %08x manual %08x", i, frame.Pix[i], manual.Pix[i])
		}
	}
}

// TestAntiFrameRectIntegralPlusFillCoversOuter verifies that an integral AA frame plus an integral inner fill exactly
// reproduces the outer fill (the frame's inner edges carry the inverse coverage bias so the two partitions meet without
// seams).
func TestAntiFrameRectIntegralPlusFillCoversOuter(t *testing.T) {
	r := geom.RectLTRB(6, 6, 26, 18)
	stroke := geom.Point{X: 4, Y: 4}
	clip := fullClip(40, 30)

	composed := newWhitePixmap(40, 30)
	AntiFrameRectRegion(r, stroke, nil, NewSolidBlitter(composed, 0xFF000000))
	AntiFillRect(geom.RectLTRB(8, 8, 24, 16), clip, NewSolidBlitter(composed, 0xFF000000))

	direct := newWhitePixmap(40, 30)
	AntiFillRect(geom.RectLTRB(4, 4, 28, 20), clip, NewSolidBlitter(direct, 0xFF000000))

	for i := range composed.Pix {
		if composed.Pix[i] != direct.Pix[i] {
			t.Fatalf("pixel %d (%d,%d): composed %08x direct %08x",
				i, i%40, i/40, composed.Pix[i], direct.Pix[i])
		}
	}
}

// TestHairPathClosedTriangleTouchesAllEdges is a smoke test for the hair path walker.
func TestHairPathClosedTriangleTouchesAllEdges(t *testing.T) {
	p := newTestTriangle()
	dev := newWhitePixmap(40, 40)
	clip := fullClip(40, 40)
	HairPath(p, clip, NewSolidBlitter(dev, 0xFF000000))

	drawn := 0
	for _, v := range dev.Pix {
		if v != 0xFFFFFFFF {
			drawn++
		}
	}
	if drawn < 40 {
		t.Fatalf("triangle hairline drew only %d pixels", drawn)
	}
	// interior must stay empty
	if dev.Pix[dev.addr(18, 22)] != 0xFFFFFFFF {
		t.Fatalf("interior drawn")
	}
}

// TestAntiHairSquareCapsExtend verifies square caps extend an open segment by half a pixel.
func TestAntiHairSquareCapsExtend(t *testing.T) {
	pts := []geom.Point{{X: 10, Y: 10.5}, {X: 30, Y: 10.5}}
	butt := newWhitePixmap(40, 20)
	square := newWhitePixmap(40, 20)
	clip := fullClip(40, 20)

	pButt := lineAsPath(pts[0], pts[1])
	pSquare := lineAsPath(pts[0], pts[1])
	AntiHairPath(pButt, clip, NewSolidBlitter(butt, 0xFF000000))
	AntiHairSquarePath(pSquare, clip, NewSolidBlitter(square, 0xFF000000))

	if butt.Pix[butt.addr(9, 10)] != 0xFFFFFFFF {
		t.Fatalf("butt cap drew left of start")
	}
	if square.Pix[square.addr(9, 10)] == 0xFFFFFFFF {
		t.Fatalf("square cap did not extend left of start")
	}
}

///////////////////////////////////////////////////////////////////////////////
// sprites

// TestSpriteSrcOverOpaqueAlpha verifies the legacy src-over sprite kernel (NEON form).
func TestSpriteSrcOverOpaqueAlpha(t *testing.T) {
	src := NewPixmap(2, 1)
	src.Pix[0] = 0x80402010 // translucent premul
	src.Pix[1] = 0xFF0000FF // opaque red (RGBA word: R=0xFF... actually R in low byte)
	dst := newWhitePixmap(4, 2)

	blitter := ChooseSprite(dst, src, 1, 0, 255, BlendSrcOver, SpriteAlphaPremul)
	blitter.BlitRect(1, 0, 2, 1)

	for i, s := range []uint32{0x80402010, 0xFF0000FF} {
		d := uint32(0xFFFFFFFF)
		nalpha := 255 - (s >> 24)
		want := satAdd8(s&0xFF, mulDiv255Round32(d&0xFF, nalpha)) |
			satAdd8(s>>8&0xFF, mulDiv255Round32(d>>8&0xFF, nalpha))<<8 |
			satAdd8(s>>16&0xFF, mulDiv255Round32(d>>16&0xFF, nalpha))<<16 |
			satAdd8(s>>24, mulDiv255Round32(d>>24, nalpha))<<24
		if got := dst.Pix[dst.addr(int32(1+i), 0)]; got != want {
			t.Fatalf("pixel %d: got %08x want %08x", i, got, want)
		}
	}
	if dst.Pix[dst.addr(0, 0)] != 0xFFFFFFFF || dst.Pix[dst.addr(3, 0)] != 0xFFFFFFFF {
		t.Fatalf("sprite drew outside its bounds")
	}
}

// TestSpriteMemcpyForSrc verifies the kSrc memcpy sprite.
func TestSpriteMemcpyForSrc(t *testing.T) {
	src := NewPixmap(2, 2)
	for i := range src.Pix {
		src.Pix[i] = uint32(0x11223344 * (i + 1))
	}
	dst := newWhitePixmap(4, 4)
	blitter := ChooseSprite(dst, src, 1, 1, 255, BlendSrc, SpriteAlphaPremul)
	blitter.BlitRect(1, 1, 2, 2)
	for y := int32(0); y < 2; y++ {
		for x := int32(0); x < 2; x++ {
			if got, want := dst.Pix[dst.addr(x+1, y+1)], src.Pix[src.addr(x, y)]; got != want {
				t.Fatalf("(%d,%d): got %08x want %08x", x, y, got, want)
			}
		}
	}
}

// TestSpriteGlobalAlphaBlend verifies the alpha<255 src-over sprite kernel.
func TestSpriteGlobalAlphaBlend(t *testing.T) {
	src := NewPixmap(1, 1)
	src.Pix[0] = 0xFF204060
	dst := newWhitePixmap(1, 1)
	blitter := ChooseSprite(dst, src, 0, 0, 128, BlendSrcOver, SpriteAlphaPremul)
	blitter.BlitRect(0, 0, 1, 1)

	s := uint32(0xFF204060)
	d := uint32(0xFFFFFFFF)
	alpha256 := alpha255To256(128)
	dstScale := alphaMulInv256(s>>24, alpha256)
	want := (s&0xFF*alpha256+d&0xFF*dstScale)>>8 |
		(s>>8&0xFF*alpha256+d>>8&0xFF*dstScale)>>8<<8 |
		(s>>16&0xFF*alpha256+d>>16&0xFF*dstScale)>>8<<16 |
		(s>>24*alpha256+d>>24*dstScale)>>8<<24
	if dst.Pix[0] != want {
		t.Fatalf("got %08x want %08x", dst.Pix[0], want)
	}
}

// TestImageSpriteBlitterMatchesBlendBlitter verifies the raster-pipeline sprite reproduces the solid BlendBlitter when
// the source is a constant color, at full and partial coverage.
func TestImageSpriteBlitterMatchesBlendBlitter(t *testing.T) {
	const w, h = 8, 4
	color := colorcore.Color(0xC0336699)
	for _, mode := range []BlendMode{BlendMultiply, BlendXor, BlendSrc, BlendColorDodge} {
		src := NewPixmap(w, h)
		pm := color.PreMultiply()
		for i := range src.Pix {
			src.Pix[i] = deviceRGBA(pm)
		}

		viaSprite := newWhitePixmap(w, h)
		viaSolid := newWhitePixmap(w, h)

		sprite := NewImageSpriteBlitter(viaSprite, src, 0, 0, 255, mode, SpriteAlphaPremul)
		solid := NewBlendBlitter(viaSolid, color, mode)

		sprite.BlitRect(0, 0, w, h/2)
		solid.BlitRect(0, 0, w, h/2)

		aa := []Alpha{97, 200, 15, 255}
		runs := []int16{1, 1, 1, 1, 0}
		aa2 := append([]Alpha{}, aa...)
		runs2 := append([]int16{}, runs...)
		sprite.BlitAntiH(2, 3, aa, runs)
		solid.BlitAntiH(2, 3, aa2, runs2)

		for i := range viaSprite.Pix {
			if viaSprite.Pix[i] != viaSolid.Pix[i] {
				t.Fatalf("mode %d pixel %d: sprite %08x solid %08x", mode, i, viaSprite.Pix[i], viaSolid.Pix[i])
			}
		}
	}
}

// TestAlignThinStrokeSnapsFirstArgument pins alignThinStroke's real contract: it is always the *first* argument that
// snaps to the pixel boundary, with the second shifted by the same delta so the pair's separation survives. Its doc
// used to say it snapped "the outer edge", but AntiFrameRectRegion's right/bottom calls pass the inner edge first.
func TestAlignThinStrokeSnapsFirstArgument(t *testing.T) {
	// Both edges inside pixel 3, at 3.25 and 3.75.
	const lo = fDot8(3*256 + 64)
	const hi = fDot8(3*256 + 192)
	const gap = hi - lo

	a, b := lo, hi
	alignThinStroke(&a, &b)
	if a != 3*256 {
		t.Fatalf("first argument not snapped to the pixel boundary: got %d want %d", a, 3*256)
	}
	if b-a != gap {
		t.Fatalf("separation not preserved: got %d want %d", b-a, gap)
	}

	// Swapping the arguments snaps the other edge — the right/bottom call sites rely on this.
	a, b = hi, lo
	alignThinStroke(&a, &b)
	if a != 3*256 {
		t.Fatalf("swapped: first argument not snapped: got %d want %d", a, 3*256)
	}
	if a-b != gap {
		t.Fatalf("swapped: separation not preserved: got %d want %d", a-b, gap)
	}

	// Edges in different pixels are left alone.
	a, b = fDot8(3*256+128), fDot8(4*256+16)
	before := [2]fDot8{a, b}
	alignThinStroke(&a, &b)
	if a != before[0] || b != before[1] {
		t.Fatalf("edges in different pixels must be untouched: got %v want %v", [2]fDot8{a, b}, before)
	}
}
