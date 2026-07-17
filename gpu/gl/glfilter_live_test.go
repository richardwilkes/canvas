// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the GPU image-filter lane (the staging in glfilter.go). Each scene draws through an image
// filter, once on the CPU rasterizer and once on the live GPU device. Every publicly reachable filter DAG now evaluates
// natively on the GPU filtercore backend, so these comparisons are true differentials — the GPU's kernel FPs vs the
// CPU's raster-pipeline stages — under thresholds matched to each kernel's expected drift (AA rasterization of the
// source layer, GPU pow vs the CPU's approximate power function, bilinear precision; the same drift that made the
// Skia-era oracle compare GPU output under its looser `gpu` profile). The CPU path is gated bit-exactly by the
// oracle's raster goldens, so close agreement proves the GPU evaluation is correct. The two visibility counters
// discriminate which lane ran: FilterReadbackCount advances once per fallback evaluation,
// filtercore.ResolveToRasterCount once per drawable→CPU image resolution — both must stay flat across a GPU-native
// evaluation; the fallback lane itself is exercised through a test-local filter wrapper that hides GPUEvaluable. Skips
// when no GL context is available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagefilter"
	"github.com/richardwilkes/canvas/shaders"
)

// compareFiltered compares two RGBA buffers rendered through the CPU and GPU backends. tol/ maxBadFraction: per-channel
// tolerance and the fraction of pixels allowed to exceed it — tight (4, 1%) when both sides evaluated the identical CPU
// DAG, blur-tolerant (16, 5%) when the GPU evaluated natively with its own blur engine.
func compareFiltered(t *testing.T, gpuData, cpuData []byte, w, h, tol int, maxBadFraction float64, tag string) {
	t.Helper()
	total := w * h
	bad, worst := 0, 0
	for i := 0; i < total; i++ {
		off := i * 4
		exceeds := false
		for c := 0; c < 4; c++ {
			d := int(gpuData[off+c]) - int(cpuData[off+c])
			if d < 0 {
				d = -d
			}
			if d > worst {
				worst = d
			}
			if d > tol {
				exceeds = true
			}
		}
		if exceeds {
			bad++
		}
	}
	frac := float64(bad) / float64(total)
	t.Logf("%s: worst channel delta=%d, bad pixels=%d/%d (%.3f%%)", tag, worst, bad, total, frac*100)
	if frac > maxBadFraction {
		t.Errorf("%s: %d/%d pixels (%.3f%%) exceed tolerance %d (worst delta %d)", tag, bad, total,
			frac*100, tol, worst)
	}
}

// rgbaAt returns the R and A channels of the RGBA buffer at device pixel (x,y).
func rgbaAt(data []byte, w, x, y int) (r, a int) {
	off := (y*w + x) * 4
	return int(data[off]), int(data[off+3])
}

// TestLiveImageFilterBlur draws an opaque-red square through a Gaussian blur image filter on a paint (the aboutToDraw
// auto-layer path) and matches the GPU result against the CPU rasterizer. The blur DAG is GPU-evaluable, so neither
// visibility counter may advance.
func TestLiveImageFilterBlur(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 64, 64
	beforeFallback := gl.FilterReadbackCount()
	beforeResolve := filtercore.ResolveToRasterCount()

	scene := func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.Color = 0xFFFF0000 // opaque red
		p.ImageFilter = imagefilter.Blur(3, 3, shaders.TileDecal, nil, nil)
		c.DrawRect(geom.RectLTRB(20, 20, 40, 40), p)
	}
	gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
	cpuData := pixmapBytes(cpuPix)

	// The filter ran on the GPU device: red core, a partial-alpha blur fringe outside the square, a transparent far
	// corner.
	if r, a := rgbaAt(gpuData, w, 30, 30); a < 250 || r < 200 {
		t.Errorf("GPU blur center = (r=%d, a=%d), want near-opaque red", r, a)
	}
	if _, a := rgbaAt(gpuData, w, 44, 30); a == 0 || a == 255 {
		t.Errorf("GPU blur fringe alpha = %d, want partial", a)
	}
	if _, a := rgbaAt(gpuData, w, 2, 2); a != 0 {
		t.Errorf("GPU blur far corner alpha = %d, want 0", a)
	}
	if got := gl.FilterReadbackCount(); got != beforeFallback {
		t.Errorf("blur DAG took the CPU fallback: counter %d -> %d", beforeFallback, got)
	}
	if got := filtercore.ResolveToRasterCount(); got != beforeResolve {
		t.Errorf("blur DAG resolved a special image to CPU pixels: counter %d -> %d", beforeResolve,
			got)
	}

	compareFiltered(t, gpuData, cpuData, w, h, 16, 0.05, "blur-on-paint")
}

// TestLiveImageFilterDropShadow draws a square through a drop-shadow image filter (unison's primary GPU image-filter
// use) and matches the GPU result against the CPU rasterizer. The DAG (offset+blur+color-filter+merge) is GPU-evaluable
// end to end.
func TestLiveImageFilterDropShadow(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 96, 96
	beforeFallback := gl.FilterReadbackCount()
	beforeResolve := filtercore.ResolveToRasterCount()

	scene := func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.Color = 0xFF3060C0 // opaque blue
		p.ImageFilter = imagefilter.DropShadow(8, 8, 3, 3, 0xFF000000, nil, nil)
		c.DrawRect(geom.RectLTRB(24, 24, 56, 56), p)
	}
	gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
	cpuData := pixmapBytes(cpuPix)

	// The original blue shape is opaque at its center; the far corner is untouched.
	if r, a := rgbaAt(gpuData, w, 40, 40); a < 250 || r > 120 {
		t.Errorf("GPU drop-shadow shape center = (r=%d, a=%d), want opaque blue", r, a)
	}
	if _, a := rgbaAt(gpuData, w, 2, 2); a != 0 {
		t.Errorf("GPU drop-shadow far corner alpha = %d, want 0", a)
	}
	// The offset shadow tints a region past the shape's lower-right corner.
	if _, a := rgbaAt(gpuData, w, 61, 61); a == 0 {
		t.Errorf("GPU drop-shadow offset region alpha = 0, want shadowed")
	}
	if got := gl.FilterReadbackCount(); got != beforeFallback {
		t.Errorf("drop-shadow DAG took the CPU fallback: counter %d -> %d", beforeFallback, got)
	}
	if got := filtercore.ResolveToRasterCount(); got != beforeResolve {
		t.Errorf("drop-shadow DAG resolved a special image to CPU pixels: counter %d -> %d",
			beforeResolve, got)
	}

	compareFiltered(t, gpuData, cpuData, w, h, 16, 0.05, "drop-shadow")
}

// TestLiveImageFilterSaveLayer filters an explicit saveLayer holding two shapes (the direct
// SaveLayer(paint-with-filter) path rather than the per-draw auto layer).
func TestLiveImageFilterSaveLayer(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 96, 96

	scene := func(c *canvas.Canvas) {
		c.Clear(0)
		layer := canvas.NewPaint()
		layer.ImageFilter = imagefilter.Blur(4, 4, shaders.TileDecal, nil, nil)
		c.SaveLayer(nil, layer)
		p := canvas.NewPaint()
		p.Color = 0xFF00A000 // opaque green
		c.DrawRect(geom.RectLTRB(20, 20, 44, 44), p)
		c.DrawRect(geom.RectLTRB(52, 52, 76, 76), p)
		c.Restore()
	}
	gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
	cpuData := pixmapBytes(cpuPix)

	// Both blurred squares left ink; the gap between them and the far corner did not.
	if _, a := rgbaAt(gpuData, w, 32, 32); a < 150 {
		t.Errorf("GPU savelayer first square alpha = %d, want mostly covered", a)
	}
	if _, a := rgbaAt(gpuData, w, 64, 64); a < 150 {
		t.Errorf("GPU savelayer second square alpha = %d, want mostly covered", a)
	}
	if _, a := rgbaAt(gpuData, w, 2, 2); a != 0 {
		t.Errorf("GPU savelayer far corner alpha = %d, want 0", a)
	}

	compareFiltered(t, gpuData, cpuData, w, h, 16, 0.05, "savelayer-blur")
}

// cpuOnlyFilter wraps a filtercore.Filter while hiding its GPUEvaluable method (only filtercore.Filter's method set is
// promoted from the embedded interface), forcing filtercore.CanEvaluateOnGPU to stage the DAG onto the
// readback→CPU→upload fallback. Every publicly reachable filter is GPU-evaluable, so the fallback lane can only be
// exercised through a wrapper like this.
type cpuOnlyFilter struct {
	filtercore.Filter
}

// TestLiveImageFilterFallback draws through a dilate filter wrapped to hide its GPU lane: the DAG must take the
// readback→CPU→upload fallback (counter advances) and, because both sides then evaluate the identical CPU DAG with
// pixel-lossless endpoints, match the CPU render tightly.
func TestLiveImageFilterFallback(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 64, 64
	beforeFallback := gl.FilterReadbackCount()

	scene := func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.Color = 0xFFFF8000 // opaque orange
		p.ImageFilter = &cpuOnlyFilter{Filter: imagefilter.Dilate(3, 3, nil, nil)}
		c.DrawRect(geom.RectLTRB(24, 24, 40, 40), p)
	}
	gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
	cpuData := pixmapBytes(cpuPix)

	// Dilation grew the square: a pixel just outside the original bounds is covered.
	if _, a := rgbaAt(gpuData, w, 42, 32); a < 200 {
		t.Errorf("GPU dilate grown edge alpha = %d, want near-opaque", a)
	}
	if got := gl.FilterReadbackCount(); got <= beforeFallback {
		t.Errorf("fallback counter did not advance for the wrapped filter: before=%d after=%d",
			beforeFallback, got)
	}

	compareFiltered(t, gpuData, cpuData, w, h, 4, 0.01, "dilate-fallback")
}

// runFilterKernelScene renders the scene on both backends, asserts the DAG evaluated GPU-natively (both visibility
// counters flat), and compares under the given threshold.
func runFilterKernelScene(t *testing.T, w, h, tol int, maxBadFraction float64, tag string, scene func(c *canvas.Canvas)) (gpuData, cpuData []byte) {
	t.Helper()
	_, dc := newLiveDirectContext(t)
	beforeFallback := gl.FilterReadbackCount()
	beforeResolve := filtercore.ResolveToRasterCount()
	gpuData, cpuPix := renderAAScene(t, dc, w, h, scene)
	cpuData = pixmapBytes(cpuPix)
	if got := gl.FilterReadbackCount(); got != beforeFallback {
		t.Errorf("%s DAG took the CPU fallback: counter %d -> %d", tag, beforeFallback, got)
	}
	if got := filtercore.ResolveToRasterCount(); got != beforeResolve {
		t.Errorf("%s DAG resolved a special image to CPU pixels: counter %d -> %d", tag,
			beforeResolve, got)
	}
	compareFiltered(t, gpuData, cpuData, w, h, tol, maxBadFraction, tag)
	return gpuData, cpuData
}

// TestLiveImageFilterMorphology dilates and erodes a square GPU-natively. The morphology kernels are exact max/min math
// over an integer-aligned source, so the comparison is tight; the dilate radius (17) exceeds maxLinearMorphologyRadius
// to cover the kernel-doubling sparse passes too.
func TestLiveImageFilterMorphology(t *testing.T) {
	const w, h = 96, 96
	gpuData, _ := runFilterKernelScene(t, w, h, 8, 0.01, "dilate", func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.Color = 0xFFFF8000 // opaque orange
		p.ImageFilter = imagefilter.Dilate(17, 5, nil, nil)
		c.DrawRect(geom.RectLTRB(40, 40, 56, 56), p)
	})
	// Dilation grew the square by the radii: covered just inside, empty just outside.
	if _, a := rgbaAt(gpuData, w, 25, 48); a < 200 {
		t.Errorf("dilate grown-left alpha = %d, want near-opaque", a)
	}
	if _, a := rgbaAt(gpuData, w, 48, 36); a == 0 {
		t.Errorf("dilate grown-top alpha = 0, want covered")
	}
	if _, a := rgbaAt(gpuData, w, 48, 33); a != 0 {
		t.Errorf("dilate beyond-radius alpha = %d, want 0", a)
	}

	gpuData, _ = runFilterKernelScene(t, w, h, 8, 0.01, "erode", func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.Color = 0xFF008040
		p.ImageFilter = imagefilter.Erode(4, 4, nil, nil)
		c.DrawRect(geom.RectLTRB(24, 24, 72, 72), p)
	})
	// Erosion shrank the square: the original edge band is now transparent, the center remains.
	if _, a := rgbaAt(gpuData, w, 26, 48); a != 0 {
		t.Errorf("erode edge-band alpha = %d, want 0", a)
	}
	if _, a := rgbaAt(gpuData, w, 48, 48); a < 200 {
		t.Errorf("erode center alpha = %d, want near-opaque", a)
	}
}

// TestLiveImageFilterDisplacement displaces the source by its own channels (both filter inputs default to the source).
// Nearest sampling at displaced coordinates can flip whole source pixels on half-texel boundaries, so the bad-pixel
// budget is looser while the tolerance stays meaningful.
func TestLiveImageFilterDisplacement(t *testing.T) {
	const w, h = 96, 96
	gpuData, _ := runFilterKernelScene(t, w, h, 16, 0.03, "displacement", func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.ImageFilter = imagefilter.DisplacementMap(imagefilter.ChannelR, imagefilter.ChannelG,
			24, nil, nil, nil)
		p.Shader = shaders.NewLinearGradient(geom.Point{X: 24, Y: 24}, geom.Point{X: 72, Y: 72},
			[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF4040FF}, nil, shaders.TileClamp, nil)
		c.DrawRect(geom.RectLTRB(24, 24, 72, 72), p)
	})
	// The displaced gradient landed ink near the center and none in the far corner.
	if _, a := rgbaAt(gpuData, w, 48, 48); a < 100 {
		t.Errorf("displacement center alpha = %d, want covered", a)
	}
	if _, a := rgbaAt(gpuData, w, 2, 2); a != 0 {
		t.Errorf("displacement far corner alpha = %d, want 0", a)
	}
}

// TestLiveImageFilterLighting runs a diffuse point light and a specular distant light over a square. The GPU evaluates
// pow directly while the CPU pipeline uses ApproxPowf, and the Sobel normals amplify AA edge differences, so the
// threshold is the loosest of the kernel set: on the specular case a small fraction of shape-edge pixels differs
// outright because dot(normal, halfDir) goes negative there and pow of a negative base is undefined in GLSL while
// ApproxPowf returns a finite value — a GPU-vs-raster-pipeline characteristic inherent to comparing a GLSL pow against
// an approximation (the oracle's GPU golden lane never sees it: there GLSL output is compared against captured GLSL
// output).
func TestLiveImageFilterLighting(t *testing.T) {
	const w, h = 96, 96
	gpuData, _ := runFilterKernelScene(t, w, h, 24, 0.05, "point-lit-diffuse", func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.Color = 0xFF3060C0
		p.ImageFilter = imagefilter.PointLitDiffuse(geom.Point3{X: 48, Y: 30, Z: 40}, 0xFFFFFFFF,
			4, 1, nil, nil)
		c.DrawRect(geom.RectLTRB(24, 24, 72, 72), p)
	})
	// Diffuse lighting of a flat interior yields lit, opaque output over the square.
	if r, a := rgbaAt(gpuData, w, 48, 48); a < 250 || r < 100 {
		t.Errorf("diffuse center = (r=%d, a=%d), want lit opaque", r, a)
	}

	gpuData, _ = runFilterKernelScene(t, w, h, 24, 0.05, "distant-lit-specular",
		func(c *canvas.Canvas) {
			c.Clear(0)
			p := canvas.NewPaint()
			p.Color = 0xFFC03028
			p.ImageFilter = imagefilter.DistantLitSpecular(geom.Point3{X: 0, Y: -1, Z: 1},
				0xFFFFFFFF, 3, 1, 8, nil, nil)
			c.DrawRoundRect(geom.RectLTRB(24, 24, 72, 72), 12, 12, p)
		})
	// The specular highlight leaves ink somewhere along the lit top edge.
	if _, a := rgbaAt(gpuData, w, 48, 25); a == 0 {
		t.Errorf("specular top-edge alpha = 0, want highlighted")
	}
}

// TestLiveImageFilterArithmetic combines the source with a blurred copy of itself through the arithmetic blend (k1..k4
// chosen so no nearly-a-blend-mode collapse applies).
func TestLiveImageFilterArithmetic(t *testing.T) {
	const w, h = 96, 96
	gpuData, _ := runFilterKernelScene(t, w, h, 16, 0.02, "arithmetic", func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.Color = 0xFF3060C0
		p.ImageFilter = imagefilter.Arithmetic(0.5, 0.5, 0.5, 0, true, nil,
			imagefilter.Blur(3, 3, shaders.TileDecal, nil, nil), nil)
		c.DrawRect(geom.RectLTRB(30, 30, 66, 66), p)
	})
	if _, a := rgbaAt(gpuData, w, 48, 48); a < 200 {
		t.Errorf("arithmetic center alpha = %d, want near-opaque", a)
	}
	// The blurred foreground contributes partial coverage outside the square (k2 = 0.5).
	if _, a := rgbaAt(gpuData, w, 70, 48); a == 0 || a == 255 {
		t.Errorf("arithmetic fringe alpha = %d, want partial", a)
	}
}

// TestLiveImageFilterConvolution sharpens with a 3x3 kernel (convolveAlpha=false exercises the unpremul accumulation
// and origin-alpha restore) and box-blurs with a 6x5 kernel whose 30 taps cross the quantized-kernel threshold
// (imagefilter/convolution.go), covering both coefficient forms through one uniform lowering.
func TestLiveImageFilterConvolution(t *testing.T) {
	const w, h = 96, 96
	sharpen := []float32{
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0,
	}
	gpuData, _ := runFilterKernelScene(t, w, h, 16, 0.02, "convolution-sharpen",
		func(c *canvas.Canvas) {
			c.Clear(0)
			p := canvas.NewPaint()
			p.ImageFilter = imagefilter.MatrixConvolution(geom.ISize{Width: 3, Height: 3}, sharpen,
				1, 0, geom.IPoint{X: 1, Y: 1}, shaders.TileDecal, false, nil, nil)
			p.Shader = shaders.NewLinearGradient(geom.Point{X: 24, Y: 24}, geom.Point{X: 72, Y: 72},
				[]colorcore.Color{0xFF202020, 0xFFF0F0F0}, nil, shaders.TileClamp, nil)
			c.DrawRect(geom.RectLTRB(24, 24, 72, 72), p)
		})
	if _, a := rgbaAt(gpuData, w, 48, 48); a < 200 {
		t.Errorf("sharpen center alpha = %d, want near-opaque", a)
	}

	boxBlur := make([]float32, 30)
	for i := range boxBlur {
		boxBlur[i] = 1.0 / 30
	}
	gpuData, _ = runFilterKernelScene(t, w, h, 16, 0.02, "convolution-quantized",
		func(c *canvas.Canvas) {
			c.Clear(0)
			p := canvas.NewPaint()
			p.Color = 0xFF20A060
			p.ImageFilter = imagefilter.MatrixConvolution(geom.ISize{Width: 6, Height: 5}, boxBlur,
				1, 0, geom.IPoint{X: 2, Y: 2}, shaders.TileDecal, true, nil, nil)
			c.DrawRect(geom.RectLTRB(30, 30, 66, 66), p)
		})
	// The box blur spread partial coverage past the square's right edge (the kernel affects up to offset.X = 2 columns
	// beyond it).
	if _, a := rgbaAt(gpuData, w, 67, 48); a == 0 || a == 255 {
		t.Errorf("quantized box-blur fringe alpha = %d, want partial", a)
	}
}

// TestLiveImageFilterMagnifier magnifies the center of a checkered square with a nonzero inset (the zero-inset path is
// a plain transform + crop that skips the kernel).
func TestLiveImageFilterMagnifier(t *testing.T) {
	const w, h = 96, 96
	gpuData, _ := runFilterKernelScene(t, w, h, 16, 0.05, "magnifier", func(c *canvas.Canvas) {
		c.Clear(0)
		p := canvas.NewPaint()
		p.ImageFilter = imagefilter.Magnifier(geom.RectLTRB(24, 24, 72, 72), 2, 8,
			shaders.SamplingOptions{Filter: shaders.FilterLinear}, nil, nil)
		p.Shader = shaders.NewLinearGradient(geom.Point{X: 24, Y: 24}, geom.Point{X: 72, Y: 24},
			[]colorcore.Color{0xFF000000, 0xFFFFFFFF}, nil, shaders.TileClamp, nil)
		c.DrawRect(geom.RectLTRB(24, 24, 72, 72), p)
	})
	if _, a := rgbaAt(gpuData, w, 48, 48); a < 200 {
		t.Errorf("magnifier center alpha = %d, want near-opaque", a)
	}
	if _, a := rgbaAt(gpuData, w, 8, 8); a != 0 {
		t.Errorf("magnifier outside-lens alpha = %d, want 0", a)
	}
}

// TestLiveImageFilterTextureSource feeds a texture-backed image through an image-source filter composed with a blur,
// drawn on the GPU canvas: the whole DAG, including the leaf, must evaluate without any drawable→CPU resolution.
func TestLiveImageFilterTextureSource(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 64, 64

	tex := gl.TextureFromImage(dc, newCheckerImage(t), gpu.MipmappedNo, gpu.BudgetedYes)
	if tex == nil {
		t.Fatal("TextureFromImage returned nil")
	}
	beforeResolve := filtercore.ResolveToRasterCount()
	beforeFallback := gl.FilterReadbackCount()

	surf := gl.MakeRenderTargetSurface(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: w, Height: h},
		1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "filter-texture-source", nil)
	if surf == nil {
		t.Fatal("MakeRenderTargetSurface returned nil")
	}
	c := surf.Canvas()
	c.Clear(0)
	p := canvas.NewPaint()
	src := geom.RectWH(4, 4)
	dst := geom.RectLTRB(16, 16, 48, 48)
	p.ImageFilter = imagefilter.Blur(2, 2, shaders.TileDecal,
		imagefilter.Image(tex, src, dst, shaders.SamplingOptions{}), nil)
	c.DrawRect(dst, p)
	gpuData := readSurfaceRGBA(t, surf)

	// The blurred checker landed: the destination center has ink, the far corner does not.
	if _, a := rgbaAt(gpuData, w, 32, 32); a < 100 {
		t.Errorf("texture-source filter center alpha = %d, want covered", a)
	}
	if _, a := rgbaAt(gpuData, w, 2, 2); a != 0 {
		t.Errorf("texture-source filter far corner alpha = %d, want 0", a)
	}
	if got := filtercore.ResolveToRasterCount(); got != beforeResolve {
		t.Errorf("texture-source DAG resolved to CPU pixels: counter %d -> %d", beforeResolve, got)
	}
	if got := gl.FilterReadbackCount(); got != beforeFallback {
		t.Errorf("texture-source DAG took the CPU fallback: counter %d -> %d", beforeFallback,
			got)
	}
}
