// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the shader pipeline: per-FP and per-blend-mode swatches rendered through the minimal fill-rect
// op and compared against CPU-computed expectations (the raster blend kernels are the CPU backend's math, so this is
// the Go-GPU vs Go-CPU self-consistency check), plus program cache hit-rate metrics on the live driver. Skips when no
// GL context is available.

package gl_test

import (
	"fmt"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/raster"
)

func newLiveSDC(t *testing.T, dc *gl.DirectContext, size int32) *gl.SurfaceDrawContext {
	t.Helper()
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: size, Height: size}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "shader-live-target")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	return sdc
}

func pmToBytes(c colorcore.PMColor4f) [4]byte {
	clamp := func(v float32) byte {
		if v <= 0 {
			return 0
		}
		if v >= 1 {
			return 255
		}
		return byte(v*255 + 0.5)
	}
	return [4]byte{clamp(c.R), clamp(c.G), clamp(c.B), clamp(c.A)}
}

// TestLiveBlendModeSwatches renders one rect per Porter-Duff blend mode over a known destination and compares the
// inside/outside pixels against the CPU blend kernels.
func TestLiveBlendModeSwatches(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env

	dstColor := colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.125, A: 0.5} // premul translucent dst
	srcColor := colorcore.PMColor4f{R: 0.4, G: 0.1, B: 0.6, A: 0.75}   // premul translucent src
	identity := geom.Matrix{}
	identity.SetIdentity()

	// NOTE: no t.Run subtests here — the GL context is current on this test's locked OS thread and subtests run on
	// other goroutines.
	for mode := raster.BlendClear; mode <= raster.BlendScreen; mode++ {
		sdc := newLiveSDC(t, dc, 16)
		sdc.Clear([4]float32{dstColor.R, dstColor.G, dstColor.B, dstColor.A})

		paint := gl.NewPaint()
		paint.SetColor4f(srcColor)
		paint.SetPorterDuffXPFactory(mode)
		sdc.FillRectWithPaint(paint, &identity, geom.Rect{
			Left: 4, Top: 4, Right: 12,
			Bottom: 12,
		})

		data := readSDC(t, sdc)
		want := pmToBytes(raster.BlendHighp4f(mode, srcColor, dstColor))
		checkPixel(t, data, 16*4, 8, 8, want, fmt.Sprintf("mode %d inside", mode))
		checkPixel(t, data, 16*4, 1, 1, pmToBytes(dstColor),
			fmt.Sprintf("mode %d outside", mode))
		sweepGLErrors(t, dc.Gpu(), "blend swatch")
		sdc.Release()
	}
}

// TestLiveFPTreeSwatches renders FP chains (blend, override-input, swizzle, compose) and checks the composed color math
// end-to-end.
func TestLiveFPTreeSwatches(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env

	identity := geom.Matrix{}
	identity.SetIdentity()
	paintColor := colorcore.PMColor4f{R: 0.8, G: 0.6, B: 0.4, A: 1}
	teal := colorcore.PMColor4f{R: 0, G: 0.5, B: 0.5, A: 1}

	cases := []struct {
		fp   func() gl.FragmentProcessor
		name string
	}{
		{name: "blend-multiply", fp: func() gl.FragmentProcessor {
			// input (paint color) multiplied with a constant.
			return gl.BlendFP(gl.MakeColorFP(teal), nil, raster.BlendMultiply)
		}},
		{name: "swizzle-bgra", fp: func() gl.FragmentProcessor {
			return gl.SwizzleOutputFP(gl.BlendFP(gl.MakeColorFP(teal), nil,
				raster.BlendMultiply), gpu.SwizzleBGRA)
		}},
		{name: "override-input", fp: func() gl.FragmentProcessor {
			// The child sees teal instead of the paint color; srcin keeps input*child.a.
			return gl.OverrideInputFP(gl.BlendFP(gl.MakeColorFP(colorcore.PMColor4f{
				R: 0.5,
				G: 0.5, B: 0.5, A: 0.5,
			}), nil, raster.BlendSrcOver), teal)
		}},
		{name: "compose", fp: func() gl.FragmentProcessor {
			// f(g(x)): g multiplies by teal, f screens with a gray.
			return gl.ComposeFP(
				gl.BlendFP(gl.MakeColorFP(colorcore.PMColor4f{
					R: 0.25, G: 0.25, B: 0.25,
					A: 1,
				}), nil, raster.BlendScreen),
				gl.BlendFP(gl.MakeColorFP(teal), nil, raster.BlendMultiply),
			)
		}},
	}
	for _, tc := range cases {
		// CPU expectation via the FP's own constant folding (verified against the raster kernels in the fake-driver
		// tests).
		want := pmToBytes(gl.FPConstantOutput(tc.fp(), paintColor))

		sdc := newLiveSDC(t, dc, 8)
		sdc.Clear([4]float32{0, 0, 0, 0})

		paint := gl.NewPaint()
		paint.SetColor4f(paintColor)
		paint.SetPorterDuffXPFactory(raster.BlendSrc)
		paint.SetColorFragmentProcessor(tc.fp())
		sdc.FillRectWithPaint(paint, &identity, geom.Rect{Right: 8, Bottom: 8})

		data := readSDC(t, sdc)
		checkPixel(t, data, 8*4, 4, 4, want, tc.name)
		sweepGLErrors(t, dc.Gpu(), tc.name)
		sdc.Release()
	}
}

// TestLiveLocalCoordsAndMatrixEffect draws a coords-derived ramp twice — once with the local coords fed straight from
// the GP's varying and once through a MatrixEffect (the uniform-matrix lift) — and checks the interpolated values at
// several pixels.
func TestLiveLocalCoordsAndMatrixEffect(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env

	const size = 32
	// The rect is the unit square and the view matrix scales it to the full target, so the GP-provided local coords
	// land in [0,1] without any local matrix.
	view := geom.Matrix{}
	view.SetScale(size, size)
	halve := geom.Matrix{}
	halve.SetScale(0.5, 0.5)

	for _, wrapped := range []bool{false, true} {
		name := "direct-coords"
		scale := float32(1)
		fp := gl.NewTestCoordsFP() // returns vec4(coords, 0, 1)
		if wrapped {
			// The uniform-matrix lift: the child's coords go through a varying computed in the vertex shader from the
			// matrix uniform.
			name = "matrix-effect"
			scale = 0.5
			fp = gl.MatrixEffectFP(halve, gl.NewTestCoordsFP())
		}
		sdc := newLiveSDC(t, dc, size)
		sdc.Clear([4]float32{0, 0, 0, 1})

		paint := gl.NewPaint()
		paint.SetPorterDuffXPFactory(raster.BlendSrc)
		paint.SetColorFragmentProcessor(fp)
		sdc.FillRectWithPaint(paint, &view, geom.Rect{Right: 1, Bottom: 1})

		data := readSDC(t, sdc)
		for _, px := range [][2]int32{{4, 24}, {16, 16}, {28, 8}} {
			fx := scale * (float32(px[0]) + 0.5) / size
			fy := scale * (float32(px[1]) + 0.5) / size
			want := pmToBytes(colorcore.PMColor4f{R: fx, G: fy, B: 0, A: 1})
			checkPixel(t, data, size*4, px[0], px[1], want,
				fmt.Sprintf("%s (%d,%d)", name, px[0], px[1]))
		}
		sweepGLErrors(t, dc.Gpu(), name)
		sdc.Release()
	}
}

// TestLiveProgramCacheMetrics checks the cache hit rate across repeated and distinct pipelines on the live driver.
func TestLiveProgramCacheMetrics(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env

	identity := geom.Matrix{}
	identity.SetIdentity()
	sdc := newLiveSDC(t, dc, 16)
	defer sdc.Release()
	sdc.Clear([4]float32{0, 0, 0, 1})

	baseline := dc.Gpu().ProgramCacheStats()

	draw := func(mode raster.BlendMode, color colorcore.PMColor4f) {
		paint := gl.NewPaint()
		paint.SetColor4f(color)
		paint.SetPorterDuffXPFactory(mode)
		sdc.FillRectWithPaint(paint, &identity, geom.Rect{Right: 8, Bottom: 8})
		dc.FlushAndSubmit(false)
	}

	red := colorcore.PMColor4f{R: 1, A: 1}
	blue := colorcore.PMColor4f{B: 0.5, A: 0.5}
	draw(raster.BlendSrcOver, blue) // miss (new program)
	draw(raster.BlendSrcOver, red)  // hit: only the color uniform changed... but red is opaque
	// NOTE: opaque + no-coverage may collapse src-over to src on some caps, producing a second program; use translucent
	// colors for the hit check instead.
	draw(raster.BlendSrcOver, blue) // hit (identical key)
	draw(raster.BlendClear, blue)   // miss (different XP outputs)

	stats := dc.Gpu().ProgramCacheStats()
	misses := stats.NumProgramCacheMisses - baseline.NumProgramCacheMisses
	hits := stats.NumProgramCacheHits - baseline.NumProgramCacheHits
	if misses+hits != 4 {
		t.Fatalf("draws seen by the cache = %d, want 4", misses+hits)
	}
	if hits < 1 {
		t.Fatalf("program cache hits = %d, want >= 1", hits)
	}
	if misses > 3 {
		t.Fatalf("program cache misses = %d, want <= 3", misses)
	}
	sweepGLErrors(t, dc.Gpu(), "cache metrics")
}
