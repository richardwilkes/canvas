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

// TestBlendBlitterPM4fRangeGuard: NewBlendBlitterPM4f documents that only in-premul-range constants take the lowp
// lanes. The per-channel `0 <= c && c <= a` tests do not bound alpha itself, so an alpha above 1 used to slip through:
// the 8-bit lanes then exceed 255 (a == 1.5 gives 383), storePM8 ORs the overflow into the neighboring channel and
// blendLowp's inv255(sa) underflows in uint32. Each out-of-range constant here must take the highp lane and blit
// exactly what blendHighpAll produces.
func TestBlendBlitterPM4fRangeGuard(t *testing.T) {
	nan := float32(math.NaN())
	for _, tc := range []struct {
		name  string
		color colorcore.PMColor4f
		lowp  bool
	}{
		{name: "in range", color: colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.5, A: 1}, lowp: true},
		{name: "alpha above one", color: colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1.5}},
		{name: "alpha well above one", color: colorcore.PMColor4f{R: 0, G: 0, B: 0, A: 17}},
		{name: "channel above alpha", color: colorcore.PMColor4f{R: 1.5, G: 0, B: 0, A: 1}},
		{name: "negative channel", color: colorcore.PMColor4f{R: -0.5, G: 0, B: 0, A: 1}},
		{name: "nan alpha", color: colorcore.PMColor4f{R: 0, G: 0, B: 0, A: nan}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := NewPixmap(4, 1)
			dsts := []uint32{0xFF000000, 0xFFFFFFFF, 0x80404040, 0x00000000}
			copy(dev.Pix, dsts)
			bb := NewBlendBlitterPM4f(dev, tc.color, BlendSrcOver)
			if bb.lowp != tc.lowp {
				t.Fatalf("lowp = %v, want %v", bb.lowp, tc.lowp)
			}
			bb.BlitH(0, 0, 4)
			cf := pmColor4f{r: tc.color.R, g: tc.color.G, b: tc.color.B, a: tc.color.A}
			for i, d := range dsts {
				if want := storeWord(blendHighpAll(BlendSrcOver, cf, loadPM4f(d))); dev.Pix[i] != want {
					t.Fatalf("pixel %d over %08x: got %08x, want the highp result %08x", i, d, dev.Pix[i], want)
				}
			}
		})
	}
}

// TestBlendModeAffectsTransparentBlackMatchesKernels derives the answer from the blend kernels themselves: feed a
// transparent black source through blendHighpAll (which covers every mode, unlike blendLowp) and see whether the
// destination survives untouched. The predicate must agree for all 29 modes — it is what decides whether a saveLayer
// restore has to flood the whole clip instead of just the layer's bounds.
func TestBlendModeAffectsTransparentBlackMatchesKernels(t *testing.T) {
	dsts := []pmColor4f{
		{r: 0.25, g: 0.5, b: 0.75, a: 1},
		{r: 0.1, g: 0.2, b: 0.3, a: 0.4},
		{r: 1, g: 1, b: 1, a: 1},
		{r: 0, g: 0, b: 0, a: 0.5},
	}
	for mode := BlendClear; mode <= BlendLuminosity; mode++ {
		preserved := true
		for _, d := range dsts {
			if got := blendHighpAll(mode, pmColor4f{}, d); got != d {
				preserved = false
				break
			}
		}
		if got := BlendModeAffectsTransparentBlack(mode); got != !preserved {
			t.Errorf("mode %d: BlendModeAffectsTransparentBlack = %v, but the kernel %s the destination",
				mode, got, map[bool]string{true: "preserves", false: "modifies"}[preserved])
		}
	}
	// The seven modes the coefficient rule (dst coeff not One/ISA/ISC) singles out, spelled out so a table edit that
	// happens to agree with a broken kernel still fails.
	want := map[BlendMode]bool{
		BlendClear: true, BlendSrc: true, BlendSrcIn: true, BlendSrcOut: true,
		BlendDstIn: true, BlendDstATop: true, BlendModulate: true,
	}
	for mode := BlendClear; mode <= BlendLuminosity; mode++ {
		if got := BlendModeAffectsTransparentBlack(mode); got != want[mode] {
			t.Errorf("BlendModeAffectsTransparentBlack(%d) = %v, want %v", mode, got, want[mode])
		}
	}
}

// TestAlphaMulInv256Domain pins alphaMulInv256's domain and range. It is not the "255*255 - value*alpha256 in
// [0, 255*255], rounded up to [0, 256]" its old doc claimed: the numerator is 0xFFFF = 257*255, and the (x + (x>>8))>>8
// divide falls one short of an exact 0xFFFF/255 at the top end, which is precisely what keeps the result within 256.
func TestAlphaMulInv256Domain(t *testing.T) {
	if got := alphaMulInv256(0, 256); got != 256 {
		t.Fatalf("alphaMulInv256(0, 256) = %d, want 256", got)
	}
	if got := alphaMulInv256(255, 256); got != 0 {
		t.Fatalf("alphaMulInv256(255, 256) = %d, want 0", got)
	}
	if got := alphaMulInv256(0, 0); got != 256 {
		t.Fatalf("alphaMulInv256(0, 0) = %d, want 256", got)
	}
	for value := uint32(0); value <= 255; value++ {
		for alpha256 := uint32(0); alpha256 <= 256; alpha256++ {
			got := alphaMulInv256(value, alpha256)
			if got > 256 {
				t.Fatalf("alphaMulInv256(%d, %d) = %d, above the documented 256 ceiling", value, alpha256, got)
			}
			// Within one count of the exact 256 - value*alpha256/255 the doc describes.
			want := int(256 - (value*alpha256+127)/255)
			if d := int(got) - want; d < -1 || d > 1 {
				t.Fatalf("alphaMulInv256(%d, %d) = %d, more than one off 256 - value*alpha256/255 (%d)",
					value, alpha256, got, want)
			}
		}
	}
	// The 256 end is only reached by a source contributing nothing, so the destination survives untouched.
	for _, dst := range []uint32{0x00000000, 0x80402010, 0xFFFFFFFF} {
		if got := blendARGB32(0, dst, 0xFF); got != dst {
			t.Fatalf("blendARGB32(0, %08X, 0xFF) = %08X, want the destination unchanged", dst, got)
		}
	}
}
