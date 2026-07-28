// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Checks for the raster Gaussian blur pass over impulse and solid-clamp inputs. These pin the GaussianPass (true 1D
// kernel) lane directly, which only sub-2-sigma blurs reach.

package filtercore

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
)

// TestGaussianPassImpulse blurs a single opaque-white impulse with sigma 1.5 (< 2, the GaussianPass lane) and checks
// the separable response against the pass's own normalized 1D kernel, including the per-pass byte quantization: out(dx,
// dy) ≈ q(k[r+dy] * q(k[r+dx] * 255)) within ±1. Symmetry and decay follow from that, and pixels beyond the kernel
// radius stay untouched (decal tiling).
func TestGaussianPassImpulse(t *testing.T) {
	const sigma = 1.5
	const size = int32(25)
	c := size / 2
	radius := SigmaToRadius(sigma)
	if sigma >= 2 || radius <= 0 {
		t.Fatalf("test sigma %v must select the GaussianPass lane (radius %d)", sigma, radius)
	}
	window := 2*radius + 1
	kernel := make([]float32, window)
	compute1DBlurKernel(sigma, radius, kernel)

	info := imagecore.MakeN32Premul(size, size)
	px := imagecore.NewPixels(info)
	px.Words[c*px.RowElems+c] = 0xFFFFFFFF
	src := NewSpecialImage(geom.IRectWH(size, size), px)
	if src == nil {
		t.Fatal("NewSpecialImage returned nil")
	}

	bounds := geom.IRectWH(size, size)
	out := raster8888BlurAlgorithm{}.Blur(geom.Size{Width: sigma, Height: sigma}, src, bounds,
		shaders.TileDecal, bounds)
	if out == nil {
		t.Fatal("Blur returned nil")
	}
	op := out.subsetPixels()
	if op == nil {
		t.Fatal("blurred image has no pixels")
	}

	q := func(v float32) int32 {
		v += 0.5
		if v > 255 {
			v = 255
		} else if v < 0 {
			v = 0
		}
		return int32(v)
	}
	alphaAt := func(x, y int32) int32 { return int32(op.Words[y*op.RowElems+x] >> 24) }

	for dy := -radius - 1; dy <= radius+1; dy++ {
		for dx := -radius - 1; dx <= radius+1; dx++ {
			got := alphaAt(c+dx, c+dy)
			var want int32
			if dx >= -radius && dx <= radius && dy >= -radius && dy <= radius {
				pass1 := q(kernel[radius+dx] * 255)
				want = q(kernel[radius+dy] * float32(pass1))
			}
			if d := got - want; d > 1 || d < -1 {
				t.Errorf("impulse response at (%+d,%+d) = %d, want %d±1", dx, dy, got, want)
			}
		}
	}
	if center := alphaAt(c, c); center == 0 || center == 255 {
		t.Errorf("blurred impulse center alpha = %d; expected a proper spread", center)
	}
}

// TestGaussianPassSolidClamp blurs an all-opaque-white image through the GaussianPass lane: the interior response sums
// the full normalized kernel, so the +0.5 rounding pushes it just past 255 and the output clamp must pin every interior
// pixel at exactly opaque white (no wrap/overflow).
func TestGaussianPassSolidClamp(t *testing.T) {
	const sigma = 1.5
	const size = int32(15)
	info := imagecore.MakeN32Premul(size, size)
	px := imagecore.NewPixels(info)
	for i := range px.Words {
		px.Words[i] = 0xFFFFFFFF
	}
	src := NewSpecialImage(geom.IRectWH(size, size), px)
	bounds := geom.IRectWH(size, size)
	out := raster8888BlurAlgorithm{}.Blur(geom.Size{Width: sigma, Height: sigma}, src, bounds,
		shaders.TileDecal, bounds)
	if out == nil {
		t.Fatal("Blur returned nil")
	}
	op := out.subsetPixels()
	radius := SigmaToRadius(sigma)
	for y := radius; y < size-radius; y++ {
		for x := radius; x < size-radius; x++ {
			if w := op.Words[y*op.RowElems+x]; w != 0xFFFFFFFF {
				t.Fatalf("interior (%d,%d) = %08x, want opaque white (clamped)", x, y, w)
			}
		}
	}
}
