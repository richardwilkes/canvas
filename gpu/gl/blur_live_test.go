// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the GPU Gaussian-blur engine (gl.GaussianBlur). Each test uploads a premultiplied source,
// blurs it on the GPU, reads it back, and compares against a CPU reference that convolves the same discrete Gaussian
// kernel the GPU's linear-sampling passes reconstruct. The direct 2D and two-pass lanes are checked tightly; the
// large-sigma downscale→blur→re-expand ladder (which adds bilinear resampling error) is checked with a looser tolerance
// plus structural invariants. Skips when no GL context is available.

package gl_test

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// blurSquareSource builds a canvas-sized RGBA8888-premul image with an opaque white square in [sqLeft, sqLeft+sqSize) ×
// [sqTop, sqTop+sqSize) and transparent elsewhere. Because the square is premul white, every channel equals the
// coverage, so the blurred result is grayscale-premul and a single-channel reference convolution predicts all four
// channels.
func blurSquareSource(t *testing.T, canvas, sqLeft, sqTop, sqSize int32) (img *imagecore.Image, coverage []float64) {
	t.Helper()
	pm := raster.NewPixmap(canvas, canvas)
	field := make([]float64, canvas*canvas)
	for y := int32(0); y < canvas; y++ {
		for x := int32(0); x < canvas; x++ {
			if x >= sqLeft && x < sqLeft+sqSize && y >= sqTop && y < sqTop+sqSize {
				pm.Pix[y*canvas+x] = 0xFFFFFFFF // opaque white (R=G=B=A=255)
				field[y*canvas+x] = 1
			}
		}
	}
	img = imagecore.Copy(pm)
	if img == nil {
		t.Fatal("source image creation failed")
	}
	return img, field
}

// referenceBlur convolves 'field' (values in [0,1]) with the discrete Gaussian kernel for 'sigma' separably, applying
// 'mode' at the boundary. It reproduces what the GPU's linear-sampling passes do for the direct lanes (the linear
// kernel reconstructs this exact discrete kernel).
func referenceBlur(field []float64, canvas int32, sigma float32, mode shaders.TileMode) []float64 {
	radius := gpu.BlurSigmaRadius(sigma)
	width := gpu.BlurKernelWidth(radius)
	k := make([]float32, width)
	gpu.Compute1DBlurKernel(sigma, radius, k)

	sample := func(get func(i int32) float64, i, n int32) float64 {
		switch {
		case i >= 0 && i < n:
			return get(i)
		case mode == shaders.TileClamp:
			return get(min(max(i, 0), n-1))
		default: // decal
			return 0
		}
	}

	tmp := make([]float64, canvas*canvas)
	for y := int32(0); y < canvas; y++ {
		for x := int32(0); x < canvas; x++ {
			var sum float64
			for kx := int32(0); kx < width; kx++ {
				si := x + kx - radius
				sum += float64(k[kx]) * sample(func(i int32) float64 { return field[y*canvas+i] }, si, canvas)
			}
			tmp[y*canvas+x] = sum
		}
	}
	out := make([]float64, canvas*canvas)
	for y := int32(0); y < canvas; y++ {
		for x := int32(0); x < canvas; x++ {
			var sum float64
			for ky := int32(0); ky < width; ky++ {
				si := y + ky - radius
				sum += float64(k[ky]) * sample(func(i int32) float64 { return tmp[i*canvas+x] }, si, canvas)
			}
			out[y*canvas+x] = sum
		}
	}
	return out
}

// blurStats compares the GPU RGBA readback against the single-channel reference, returning the max per-channel diff,
// the fraction of channel samples exceeding 'tol', and the mean absolute diff.
func blurStats(got []byte, ref []float64, canvas int32, tol int) (maxDiff int, exceedFrac, mean float64) {
	total := 0
	exceed := 0
	sum := 0.0
	for i := int32(0); i < canvas*canvas; i++ {
		want := int(math.Round(ref[i] * 255))
		for c := 0; c < 4; c++ {
			g := int(got[i*4+int32(c)])
			d := g - want
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
			if d > tol {
				exceed++
			}
			sum += float64(d)
			total++
		}
	}
	return maxDiff, float64(exceed) / float64(total), sum / float64(total)
}

// runGPUBlur uploads the source, blurs it with dstBounds == srcBounds (blur in place), and reads the square result back
// as RGBA8888 bytes.
func runGPUBlur(t *testing.T, dc *gl.DirectContext, img *imagecore.Image, canvas int32, sigma float32, mode shaders.TileMode) []byte {
	t.Helper()
	tex := gl.TextureFromImage(dc, img, gpu.MipmappedNo, gpu.BudgetedYes)
	if tex == nil {
		t.Fatal("TextureFromImage returned nil")
	}
	defer tex.Release()

	bounds := geom.IRectXYWH(0, 0, canvas, canvas)
	sdc := gl.GaussianBlur(dc, tex.View(), gpu.ColorTypeRGBA8888, gpu.AlphaTypePremul, bounds,
		bounds, sigma, sigma, mode, gpu.BackingFitExact)
	if sdc == nil {
		t.Fatalf("GaussianBlur returned nil (sigma %v, mode %v)", sigma, mode)
	}
	defer sdc.Release() // the caller owns the blur result
	if sdc.Dimensions() != (geom.ISize{Width: canvas, Height: canvas}) {
		t.Fatalf("blur result dims = %v, want %dx%d", sdc.Dimensions(), canvas, canvas)
	}
	return readSDC(t, sdc)
}

func TestLiveGaussianBlur2D(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const canvas = 32
	img, field := blurSquareSource(t, canvas, 10, 10, 12)

	// sigma 0.6 -> radius 2 -> 5x5 kernel (25 <= 28): the single-pass non-separable 2D lane.
	const sigma = 0.6
	got := runGPUBlur(t, dc, img, canvas, sigma, shaders.TileDecal)
	ref := referenceBlur(field, canvas, sigma, shaders.TileDecal)
	maxDiff, exceed, mean := blurStats(got, ref, canvas, 2)
	t.Logf("2D blur: maxDiff=%d exceedFrac=%.4f mean=%.3f", maxDiff, exceed, mean)
	if exceed > 0.005 || mean > 0.5 {
		t.Errorf("2D blur diverges: maxDiff=%d exceedFrac=%.4f mean=%.3f", maxDiff, exceed, mean)
	}
}

func TestLiveGaussianBlurTwoPass(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const canvas = 48
	img, field := blurSquareSource(t, canvas, 16, 16, 16)

	// sigma 1.5 -> radius 5 -> 11x11 kernel (>28): separable X-then-Y linear passes.
	for _, mode := range []shaders.TileMode{shaders.TileDecal, shaders.TileClamp} {
		const sigma = 1.5
		got := runGPUBlur(t, dc, img, canvas, sigma, mode)
		ref := referenceBlur(field, canvas, sigma, mode)
		maxDiff, exceed, mean := blurStats(got, ref, canvas, 3)
		t.Logf("two-pass blur mode=%v: maxDiff=%d exceedFrac=%.4f mean=%.3f", mode, maxDiff,
			exceed, mean)
		if exceed > 0.01 || mean > 0.6 {
			t.Errorf("two-pass blur mode=%v diverges: maxDiff=%d exceedFrac=%.4f mean=%.3f", mode,
				maxDiff, exceed, mean)
		}
	}
}

// TestLiveGaussianBlurAlpha8 blurs an alpha-only source (the mask-filter consumer's format) and reads it back as
// Alpha8, exercising GaussianBlur's color-type-agnostic plumbing on A8.
func TestLiveGaussianBlurAlpha8(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const canvas = 40
	// A8 source: opaque square [12,28) x [12,28), transparent elsewhere.
	info, _ := imagecore.MakeInfo(canvas, canvas, imagecore.ColorTypeAlpha8, imagecore.AlphaTypePremul)
	data := make([]byte, canvas*canvas)
	field := make([]float64, canvas*canvas)
	for y := int32(0); y < canvas; y++ {
		for x := int32(0); x < canvas; x++ {
			if x >= 12 && x < 28 && y >= 12 && y < 28 {
				data[y*canvas+x] = 255
				field[y*canvas+x] = 1
			}
		}
	}
	img := imagecore.NewRasterData(info, data, canvas)
	if img == nil {
		t.Fatal("alpha source creation failed")
	}
	tex := gl.TextureFromImage(dc, img, gpu.MipmappedNo, gpu.BudgetedYes)
	if tex == nil {
		t.Fatal("TextureFromImage(alpha) returned nil")
	}
	defer tex.Release()

	const sigma = 1.5
	bounds := geom.IRectXYWH(0, 0, canvas, canvas)
	sdc := gl.GaussianBlur(dc, tex.View(), gpu.ColorTypeAlpha8, gpu.AlphaTypePremul, bounds, bounds,
		sigma, sigma, shaders.TileDecal, gpu.BackingFitExact)
	if sdc == nil {
		t.Fatal("GaussianBlur(alpha8) returned nil")
	}
	defer sdc.Release() // the caller owns the blur result
	got := gl.Pixels{
		ColorType: gpu.ColorTypeAlpha8, Dims: geom.ISize{Width: canvas, Height: canvas},
		RowBytes: canvas, Data: make([]byte, canvas*canvas),
	}
	if !sdc.ReadPixels(got, geom.IPoint{}) {
		t.Fatal("Alpha8 ReadPixels failed")
	}
	ref := referenceBlur(field, canvas, sigma, shaders.TileDecal)

	var maxDiff, exceed int
	var sum float64
	for i := int32(0); i < canvas*canvas; i++ {
		want := int(math.Round(ref[i] * 255))
		d := int(got.Data[i]) - want
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
		if d > 3 {
			exceed++
		}
		sum += float64(d)
	}
	mean := sum / float64(canvas*canvas)
	t.Logf("alpha8 blur: maxDiff=%d exceed=%d mean=%.3f", maxDiff, exceed, mean)
	if float64(exceed)/float64(canvas*canvas) > 0.01 || mean > 0.6 {
		t.Errorf("alpha8 blur diverges: maxDiff=%d exceed=%d mean=%.3f", maxDiff, exceed, mean)
	}
}

func TestLiveGaussianBlurLargeSigmaLadder(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const canvas = 48
	img, field := blurSquareSource(t, canvas, 16, 16, 16)

	// sigma 8 > the max linear-blur sigma (4): exercises the downscale -> blur -> re-expand ladder.
	const sigma = 8
	got := runGPUBlur(t, dc, img, canvas, sigma, shaders.TileDecal)
	ref := referenceBlur(field, canvas, sigma, shaders.TileDecal)
	// The ladder resamples through a half-size intermediate, so it is a close-but-not-exact match to the ideal discrete
	// kernel; verify magnitude + smoothness rather than bit parity.
	maxDiff, exceed, mean := blurStats(got, ref, canvas, 10)
	t.Logf("ladder blur: maxDiff=%d exceedFrac=%.4f mean=%.3f", maxDiff, exceed, mean)
	if exceed > 0.02 || mean > 3 {
		t.Errorf("ladder blur diverges: maxDiff=%d exceedFrac=%.4f mean=%.3f", maxDiff, exceed, mean)
	}

	// Structural: the result is a valid premul grayscale image with a bright center and dark corners (the blur is
	// centered and smooth).
	center := got[(canvas/2*canvas+canvas/2)*4+3]
	corner := got[3] // pixel (0,0) alpha
	if center < 40 {
		t.Errorf("ladder blur center too dark: alpha=%d", center)
	}
	if corner > center {
		t.Errorf("ladder blur corner (%d) brighter than center (%d)", corner, center)
	}
	for i := int32(0); i < canvas*canvas; i++ {
		a := got[i*4+3]
		if got[i*4] > a+1 || got[i*4+1] > a+1 || got[i*4+2] > a+1 {
			t.Fatalf("ladder blur pixel %d not premultiplied: rgba=%d,%d,%d,%d", i, got[i*4],
				got[i*4+1], got[i*4+2], a)
		}
	}
}
