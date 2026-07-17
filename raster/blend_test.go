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
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// randPM8 returns a valid premultiplied color.
func randPM8(rng *rand.Rand) pm8 {
	a := uint32(rng.Intn(256))
	return pm8{
		r: uint32(rng.Intn(int(a) + 1)),
		g: uint32(rng.Intn(int(a) + 1)),
		b: uint32(rng.Intn(int(a) + 1)),
		a: a,
	}
}

var lowpModes = []BlendMode{
	BlendClear, BlendSrc, BlendDst, BlendSrcOver, BlendDstOver, BlendSrcIn, BlendDstIn,
	BlendSrcOut, BlendDstOut, BlendSrcATop, BlendDstATop, BlendXor, BlendPlus, BlendModulate,
	BlendScreen, BlendOverlay, BlendDarken, BlendLighten, BlendHardLight, BlendDifference,
	BlendExclusion, BlendMultiply,
}

var highpModes = []BlendMode{
	BlendColorDodge, BlendColorBurn, BlendSoftLight, BlendHue, BlendSaturation, BlendColor,
	BlendLuminosity,
}

// TestBlendLowpInvariants: results stay in range and remain valid premul (channels ≤ alpha, with 1 step of rounding
// slack), and the symmetric modes commute.
func TestBlendLowpInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, mode := range lowpModes {
		for i := 0; i < 2000; i++ {
			s, d := randPM8(rng), randPM8(rng)
			r := blendLowp(mode, s, d)
			for _, c := range []uint32{r.r, r.g, r.b, r.a} {
				if c > 255 {
					t.Fatalf("mode %d: channel %d out of range (s=%+v d=%+v)", mode, c, s, d)
				}
			}
			if r.r > r.a+1 || r.g > r.a+1 || r.b > r.a+1 {
				t.Fatalf("mode %d: unpremul result %+v (s=%+v d=%+v)", mode, r, s, d)
			}
			switch mode {
			case BlendPlus, BlendModulate, BlendScreen, BlendDarken, BlendLighten,
				BlendDifference, BlendExclusion, BlendMultiply:
				if rr := blendLowp(mode, d, s); rr != r {
					t.Fatalf("mode %d not symmetric: %+v vs %+v", mode, r, rr)
				}
			}
		}
	}
}

// TestBlendLowpKnownValues: hand-checked identities.
func TestBlendLowpKnownValues(t *testing.T) {
	s := pm8{r: 51, g: 102, b: 153, a: 255}
	d := pm8{r: 102, g: 102, b: 102, a: 255}
	if r := blendLowp(BlendClear, s, d); r != (pm8{}) {
		t.Fatalf("clear: %+v", r)
	}
	if r := blendLowp(BlendSrc, s, d); r != s {
		t.Fatalf("src: %+v", r)
	}
	if r := blendLowp(BlendDst, s, d); r != d {
		t.Fatalf("dst: %+v", r)
	}
	// Opaque srcover = src.
	if r := blendLowp(BlendSrcOver, s, d); r != s {
		t.Fatalf("srcover opaque: %+v", r)
	}
	// Opaque xor = transparent.
	if r := blendLowp(BlendXor, s, d); r != (pm8{}) {
		t.Fatalf("xor opaque: %+v", r)
	}
	// multiply of opaque colors: div255(s*d) per channel.
	want := pm8{r: div255(51 * 102), g: div255(102 * 102), b: div255(153 * 102), a: 255}
	if r := blendLowp(BlendMultiply, s, d); r != want {
		t.Fatalf("multiply: %+v, want %+v", r, want)
	}
	// plus clamps.
	if r := blendLowp(BlendPlus, pm8{r: 200, g: 200, b: 200, a: 200}, pm8{r: 100, g: 100, b: 100, a: 100}); r != (pm8{r: 255, g: 255, b: 255, a: 255}) {
		t.Fatalf("plus clamp: %+v", r)
	}
}

// TestBlendHighpInvariants: float modes produce premul results in range.
func TestBlendHighpInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, mode := range highpModes {
		for i := 0; i < 2000; i++ {
			s8, d8 := randPM8(rng), randPM8(rng)
			s := pmColor4f{r: float32(s8.r) / 255, g: float32(s8.g) / 255, b: float32(s8.b) / 255, a: float32(s8.a) / 255}
			d := pmColor4f{r: float32(d8.r) / 255, g: float32(d8.g) / 255, b: float32(d8.b) / 255, a: float32(d8.a) / 255}
			r := blendHighp(mode, s, d)
			for _, c := range []float32{r.r, r.g, r.b, r.a} {
				if !(c > -0.001 && c < 1.001) {
					t.Fatalf("mode %d: channel %v out of range (s=%+v d=%+v)", mode, c, s, d)
				}
			}
			if r.r > r.a+0.005 || r.g > r.a+0.005 || r.b > r.a+0.005 {
				t.Fatalf("mode %d: unpremul result %+v (s=%+v d=%+v)", mode, r, s, d)
			}
		}
	}
}

// TestBlendBlitterDstIdentity: BlendDst must leave the destination bytes untouched through every blit entry point (full
// span, coverage span, mask).
func TestBlendBlitterDstIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	dev := NewPixmap(32, 4)
	for i := range dev.Pix {
		c := randPM8(rng)
		dev.Pix[i] = storePM8(c)
	}
	before := make([]uint32, len(dev.Pix))
	copy(before, dev.Pix)

	bb := NewBlendBlitter(dev, colorcore.Color(0x80336699), BlendDst)
	bb.BlitH(0, 0, 32)
	bb.BlitRect(0, 1, 32, 1)
	aa := make([]Alpha, 33)
	runs := make([]int16, 33)
	for i, v := range []Alpha{7, 130, 255, 0} {
		runs[i*8] = 8
		aa[i*8] = v
	}
	bb.BlitAntiH(0, 2, aa, runs)
	mask := &Mask{Image: make([]uint8, 32), Bounds: geom.IRectLTRB(0, 3, 32, 4), RowBytes: 32}
	for i := range mask.Image {
		mask.Image[i] = uint8(i * 8)
	}
	bb.BlitMask(mask, mask.Bounds)

	for i := range dev.Pix {
		if dev.Pix[i] != before[i] {
			t.Fatalf("pixel %d changed: %08x -> %08x", i, before[i], dev.Pix[i])
		}
	}
}
