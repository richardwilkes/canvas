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
	"math"
	"testing"
)

// unpremulTestPixmap builds a straight-alpha source sweeping alpha (including the fully transparent and fully opaque
// ends) against color channels that do not divide evenly by 255, so the premultiply rounding is exercised.
func unpremulTestPixmap(w, h int32) *Pixmap {
	alphas := []uint32{0, 1, 37, 64, 128, 200, 254, 255}
	pm := NewPixmap(w, h)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			i := int(y*w + x)
			r := uint32(i*37) & 0xFF
			g := uint32(i*91+13) & 0xFF
			b := uint32(i*211+7) & 0xFF
			pm.Pix[pm.addr(x, y)] = r | g<<8 | b<<16 | alphas[i%len(alphas)]<<24
		}
	}
	return pm
}

// premultipliedCopy returns a copy of src with every pixel premultiplied: the pixels a caller would have to hand over
// itself to stay correct on both draw lanes.
func premultipliedCopy(src *Pixmap) *Pixmap {
	pm := NewPixmap(src.Width, src.Height)
	for y := int32(0); y < src.Height; y++ {
		for x := int32(0); x < src.Width; x++ {
			pm.Pix[pm.addr(x, y)] = premul8888(src.Pix[src.addr(x, y)])
		}
	}
	return pm
}

// maxChannelDiff returns the largest per-channel difference between two 8888 words.
func maxChannelDiff(a, b uint32) int {
	worst := 0
	for shift := 0; shift < 32; shift += 8 {
		if d := int(a>>shift&0xFF) - int(b>>shift&0xFF); d < 0 {
			worst = max(worst, -d)
		} else {
			worst = max(worst, d)
		}
	}
	return worst
}

// TestPremul8888MatchesRoundedProduct pins premul8888 against what it stands for: each color channel scaled by the
// pixel's alpha, rounded half up, with alpha itself untouched. Exhaustive over alpha.
func TestPremul8888MatchesRoundedProduct(t *testing.T) {
	channels := []uint32{0, 1, 8, 16, 96, 127, 128, 129, 200, 254, 255}
	for a := uint32(0); a <= 255; a++ {
		for _, r := range channels {
			for _, g := range channels {
				b := 255 - g
				got := premul8888(r | g<<8 | b<<16 | a<<24)
				want := scaleByAlpha(r, a) | scaleByAlpha(g, a)<<8 | scaleByAlpha(b, a)<<16 | a<<24
				if got != want {
					t.Fatalf("premul8888(r=%d g=%d b=%d a=%d) = %08x want %08x", r, g, b, a, got, want)
				}
			}
		}
	}
}

// scaleByAlpha is the reference premultiply of one channel: round(c*a/255).
func scaleByAlpha(c, a uint32) uint32 {
	return uint32(math.Round(float64(c) * float64(a) / 255))
}

// TestChooseSpriteNeverCopiesUnpremulVerbatim pins the lane choice: a straight-alpha source can never take the
// verbatim-copy sprite, since its bytes are not in the destination's premultiplied form.
func TestChooseSpriteNeverCopiesUnpremulVerbatim(t *testing.T) {
	src := unpremulTestPixmap(4, 2)
	dst := newWhitePixmap(4, 2)
	for _, mode := range []BlendMode{BlendSrc, BlendSrcOver} {
		if _, isCopy := ChooseSprite(dst, src, 0, 0, 255, mode, SpriteAlphaUnpremul).(*spriteMemcpy); isCopy {
			t.Fatalf("mode %d: unpremultiplied source took the verbatim-copy sprite", mode)
		}
		// The premultiplied forms still do, so the fast lane is not lost for them.
		if _, isCopy := ChooseSprite(dst, src, 0, 0, 255, mode, SpriteAlphaOpaque).(*spriteMemcpy); !isCopy {
			t.Fatalf("mode %d: opaque source lost the verbatim-copy sprite", mode)
		}
	}
}

// TestSpriteUnpremulMatchesPremultipliedSource is the fix for the sprite lane compositing straight-alpha sources as if
// they were premultiplied: blitting an unpremultiplied source must land on the same bytes as blitting the
// premultiplied copy of it. Exact for every lowp lane; the highp lanes premultiply in float instead of rounding
// through 8 bits first, so they are allowed the one-step difference that costs.
func TestSpriteUnpremulMatchesPremultipliedSource(t *testing.T) {
	const w, h = 8, 4
	src := unpremulTestPixmap(w, h)
	pre := premultipliedCopy(src)
	for _, mode := range []BlendMode{BlendSrcOver, BlendSrc, BlendMultiply, BlendXor, BlendPlus, BlendColorDodge} {
		for _, alpha := range []uint8{255, 200, 128, 0} {
			viaUnpremul := newWhitePixmap(w, h)
			viaPremul := newWhitePixmap(w, h)
			ChooseSprite(viaUnpremul, src, 0, 0, alpha, mode, SpriteAlphaUnpremul).BlitRect(0, 0, w, h)
			ChooseSprite(viaPremul, pre, 0, 0, alpha, mode, SpriteAlphaPremul).BlitRect(0, 0, w, h)

			tolerance := 0
			if blendNeedsHighp(mode) {
				tolerance = 1
			}
			for i := range viaUnpremul.Pix {
				if maxChannelDiff(viaUnpremul.Pix[i], viaPremul.Pix[i]) > tolerance {
					t.Fatalf("mode %d alpha %d pixel %d: unpremul %08x premul %08x (src %08x)",
						mode, alpha, i, viaUnpremul.Pix[i], viaPremul.Pix[i], src.Pix[i])
				}
			}
		}
	}
}

// TestImageSpriteBlitterUnpremulCoverage covers the raster-pipeline sprite's partial-coverage and mask lanes, which
// load their source pixels one at a time rather than a row at a time.
func TestImageSpriteBlitterUnpremulCoverage(t *testing.T) {
	const w, h = 8, 4
	src := unpremulTestPixmap(w, h)
	pre := premultipliedCopy(src)
	for _, mode := range []BlendMode{BlendSrcOver, BlendSrc, BlendXor} {
		viaUnpremul := newWhitePixmap(w, h)
		viaPremul := newWhitePixmap(w, h)
		unpremulBlitter := NewImageSpriteBlitter(viaUnpremul, src, 0, 0, 255, mode, SpriteAlphaUnpremul)
		premulBlitter := NewImageSpriteBlitter(viaPremul, pre, 0, 0, 255, mode, SpriteAlphaPremul)

		aa := []Alpha{97, 200, 15, 255}
		runs := []int16{1, 1, 1, 1, 0}
		unpremulBlitter.BlitAntiH(2, 1, append([]Alpha{}, aa...), append([]int16{}, runs...))
		premulBlitter.BlitAntiH(2, 1, append([]Alpha{}, aa...), append([]int16{}, runs...))
		unpremulBlitter.BlitV(0, 0, h, 128)
		premulBlitter.BlitV(0, 0, h, 128)

		for i := range viaUnpremul.Pix {
			if viaUnpremul.Pix[i] != viaPremul.Pix[i] {
				t.Fatalf("mode %d pixel %d: unpremul %08x premul %08x", mode, i, viaUnpremul.Pix[i], viaPremul.Pix[i])
			}
		}
	}
}

// TestSpriteUnpremulTransparentPixelLeavesDestination is the reported symptom in its smallest form: a fully
// transparent straight-alpha pixel used to add its own color to the backdrop instead of leaving it alone.
func TestSpriteUnpremulTransparentPixelLeavesDestination(t *testing.T) {
	src := NewPixmap(2, 1)
	src.Pix[0] = 16 | 8<<8 | 96<<16 | 0<<24   // color (16,8,96), fully transparent
	src.Pix[1] = 16 | 8<<8 | 96<<16 | 128<<24 // the same color, half transparent
	dst := NewPixmap(2, 1)
	const bg = 229 | 229<<8 | 127<<16 | 255<<24
	dst.Pix[0] = bg
	dst.Pix[1] = bg

	ChooseSprite(dst, src, 0, 0, 255, BlendSrcOver, SpriteAlphaUnpremul).BlitRect(0, 0, 2, 1)

	if dst.Pix[0] != bg {
		t.Fatalf("transparent pixel changed the destination: %08x want %08x", dst.Pix[0], bg)
	}
	// Half-transparent: round(c*128/255) + round(bg*(255-128)/255) per channel.
	var want uint32
	for i, c := range []uint32{16, 8, 96} {
		bgc := []uint32{229, 229, 127}[i]
		want |= (scaleByAlpha(c, 128) + scaleByAlpha(bgc, 127)) << (8 * i)
	}
	want |= 255 << 24
	if dst.Pix[1] != want {
		t.Fatalf("half-transparent pixel = %08x want %08x", dst.Pix[1], want)
	}
}
