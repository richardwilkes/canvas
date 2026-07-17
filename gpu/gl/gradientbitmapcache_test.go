// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for gradientBitmapCache (the textured-colorizer LUT bake): LRU behavior, keying, and the baked ramp bytes
// against the CPU gradient stages for representative stop sets.

package gl

import (
	"bytes"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
)

// rampStops builds n evenly spaced opaque stops with explicit positions (the form the FP-conversion site hands the
// cache: positions are always explicit there, synthesized when the descriptor is evenly spaced).
func rampStops(n int) (colors []colorcore.Color4f, positions []float32) {
	colors = make([]colorcore.Color4f, n)
	positions = make([]float32, n)
	denom := float32(n - 1)
	for i := range colors {
		colors[i] = colorcore.Color4f{
			R: float32(i%7) / 6,
			G: float32(i%11) / 10,
			B: float32(i%13) / 12,
			A: 1,
		}
		positions[i] = float32(i) / denom
	}
	return colors, positions
}

func TestGradientBitmapCacheLRUAndKeying(t *testing.T) {
	c := newGradientBitmapCache(2, 16)

	colorsA, posA := rampStops(3)
	colorsB, posB := rampStops(4)
	colorsC, posC := rampStops(5)

	a := c.getGradient(colorsA, posA, false, gpu.ColorTypeRGBA8888)
	if got := c.getGradient(colorsA, posA, false, gpu.ColorTypeRGBA8888); got != a {
		t.Fatal("identical gradient config must hit the cached entry")
	}
	b := c.getGradient(colorsB, posB, false, gpu.ColorTypeRGBA8888)
	if b == a {
		t.Fatal("different color count must miss")
	}
	if len(c.entries) != 2 {
		t.Fatalf("entry count = %d, want 2", len(c.entries))
	}

	// Touch A so B becomes the LRU tail, then insert C: B must be evicted, A retained.
	if got := c.getGradient(colorsA, posA, false, gpu.ColorTypeRGBA8888); got != a {
		t.Fatal("A must still hit")
	}
	c.getGradient(colorsC, posC, false, gpu.ColorTypeRGBA8888)
	if len(c.entries) != 2 {
		t.Fatalf("entry count after eviction = %d, want 2 (the LRU cap)", len(c.entries))
	}
	if got := c.getGradient(colorsA, posA, false, gpu.ColorTypeRGBA8888); got != a {
		t.Fatal("A (recently used) must survive the eviction")
	}
	reB := c.getGradient(colorsB, posB, false, gpu.ColorTypeRGBA8888)
	if reB == b {
		t.Fatal("B was evicted; re-getting it must build a fresh entry")
	}
	if reB.proxyKey.Equal(&b.proxyKey) {
		t.Fatal("a re-baked entry must carry a fresh proxy key (no texture aliasing)")
	}
	if !bytes.Equal(reB.pixels, b.pixels) {
		t.Fatal("re-baking the same gradient must produce identical ramp bytes")
	}

	// Keying: interior positions and color type participate; a same-shape gradient with a shifted interior stop or a
	// different LUT color type is a distinct entry.
	c2 := newGradientBitmapCache(maxNumCachedGradientBitmaps, 16)
	base := c2.getGradient(colorsA, posA, false, gpu.ColorTypeRGBA8888)
	shifted := []float32{0, 0.25, 1}
	if got := c2.getGradient(colorsA, shifted, false, gpu.ColorTypeRGBA8888); got == base {
		t.Fatal("interior position change must miss")
	}
	if got := c2.getGradient(colorsA, posA, false, gpu.ColorTypeRGBAF16); got == base {
		t.Fatal("color type change must miss")
	}
	altColors := append([]colorcore.Color4f{}, colorsA...)
	altColors[1].G += 0.125
	if got := c2.getGradient(altColors, posA, false, gpu.ColorTypeRGBA8888); got == base {
		t.Fatal("color change must miss")
	}
	if len(c2.entries) != 4 {
		t.Fatalf("distinct configs = %d entries, want 4", len(c2.entries))
	}
}

// referenceRampColor computes the gradient color at t by direct interpolation between the bracketing stops —
// independent math (no factor/bias precomputation, no FMA), so it cross-checks the CPU-stage bake rather than restating
// it. Hard stops resolve to the right interval, matching the searched stage's stop counting.
func referenceRampColor(colors []colorcore.Color4f, positions []float32, premul bool, t float32) colorcore.PMColor4f {
	n := len(colors)
	// Count the stops at or below t (the searched stage's index rule), then interpolate inside that interval.
	idx := 0
	for i := 1; i < n; i++ {
		if t >= positions[i] {
			idx = i
		}
	}
	var c colorcore.Color4f
	switch {
	case idx >= n-1:
		c = colors[n-1]
	default:
		tL, tR := positions[idx], positions[idx+1]
		if tR <= tL {
			c = colors[idx]
		} else {
			f := (t - tL) / (tR - tL)
			cl, cr := colors[idx], colors[idx+1]
			c = colorcore.Color4f{
				R: cl.R + f*(cr.R-cl.R),
				G: cl.G + f*(cr.G-cl.G),
				B: cl.B + f*(cr.B-cl.B),
				A: cl.A + f*(cr.A-cl.A),
			}
		}
	}
	if premul {
		return colorcore.PMColor4f{R: c.R * c.A, G: c.G * c.A, B: c.B * c.A, A: c.A}
	}
	return colorcore.PMColor4f(c)
}

func TestGradientRampBytesMatchCPUStages(t *testing.T) {
	many, manyPos := rampStops(200)
	cases := []struct {
		name      string
		colors    []colorcore.Color4f
		positions []float32
		premul    bool
	}{
		{
			name: "hard-stops-at-ends",
			colors: []colorcore.Color4f{
				{R: 0.1, G: 0.1, B: 0.1, A: 1}, {R: 1, A: 1}, {B: 1, A: 1}, {R: 0.2, G: 0.2, B: 0.2, A: 1},
			},
			positions: []float32{0, 0, 1, 1},
		},
		{
			name: "interior-hard-stops",
			colors: []colorcore.Color4f{
				{R: 1, A: 1}, {R: 1, A: 1}, {G: 1, A: 1}, {G: 1, A: 1}, {B: 1, A: 1}, {B: 1, A: 1},
			},
			positions: []float32{0, 0.33, 0.33, 0.66, 0.66, 1},
		},
		{name: "200-stops", colors: many, positions: manyPos},
		{
			name: "translucent",
			colors: []colorcore.Color4f{
				{R: 1, A: 0.75}, {G: 1, A: 0.25}, {R: 0.5, B: 1, A: 1},
			},
			positions: []float32{0, 0.4, 1},
			premul:    true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pixels := fillGradientRamp(tc.colors, tc.positions, tc.premul, gpu.ColorTypeRGBA8888,
				gradientTextureSize)
			for i := range gradientTextureSize {
				tex := (float32(i) + 0.5) / gradientTextureSize
				want := referenceRampColor(tc.colors, tc.positions, tc.premul, tex)
				got := pixels[i*4 : i*4+4]
				for ch, w := range []float32{want.R, want.G, want.B, want.A} {
					wb := rampToUnormByte(w)
					if d := int(got[ch]) - int(wb); d < -1 || d > 1 {
						t.Fatalf("texel %d channel %d = %d, reference %d (past ±1)", i, ch, got[ch], wb)
					}
				}
				if tc.premul {
					// Premul invariant: no channel exceeds alpha (+1 for rounding).
					for ch := range 3 {
						if int(got[ch]) > int(got[3])+1 {
							t.Fatalf("texel %d channel %d = %d exceeds alpha %d: ramp is not premultiplied",
								i, ch, got[ch], got[3])
						}
					}
				}
			}
			// End texels must carry the outer stop colors exactly (t = 0.5/256 sits inside the first interval, but a
			// hard stop at the ends makes the outermost interval constant).
			if tc.name == "hard-stops-at-ends" {
				if pixels[0] != rampToUnormByte(1) || pixels[1] != 0 || pixels[2] != 0 {
					t.Fatalf("first texel = %v, want the t=0 hard-stop color (red)", pixels[:4])
				}
				last := pixels[(gradientTextureSize-1)*4:]
				if last[2] != rampToUnormByte(1) || last[0] != 0 || last[1] != 0 {
					t.Fatalf("last texel = %v, want the t=1 hard-stop color (blue)", last)
				}
			}
		})
	}
}

func TestGradientRampF16StoresHalves(t *testing.T) {
	colors := []colorcore.Color4f{{R: 1, A: 0.5}, {G: 0.25, B: 1, A: 1}}
	positions := []float32{0, 1}
	const res = 64
	ramp := make([]colorcore.PMColor4f, res)
	shaders.EvalGradientRamp(colors, positions, true, ramp)
	pixels := fillGradientRamp(colors, positions, true, gpu.ColorTypeRGBAF16, res)
	if len(pixels) != res*8 {
		t.Fatalf("F16 ramp is %d bytes, want %d", len(pixels), res*8)
	}
	for i := range res {
		o := i * 8
		for ch, v := range []float32{ramp[i].R, ramp[i].G, ramp[i].B, ramp[i].A} {
			got := uint16(pixels[o+ch*2]) | uint16(pixels[o+ch*2+1])<<8
			if want := imagecore.FloatToHalf(v); got != want {
				t.Fatalf("texel %d channel %d = %#04x, want %#04x (float %g)", i, ch, got, want, v)
			}
			// And the halves decode back to within half precision of the float ramp.
			if d := math.Abs(float64(imagecore.HalfToFloat(got) - v)); d > 1.0/1024 {
				t.Fatalf("texel %d channel %d decodes to %g, float ramp %g", i, ch,
					imagecore.HalfToFloat(got), v)
			}
		}
	}
}
