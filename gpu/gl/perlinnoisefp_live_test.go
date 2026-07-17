// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the GPU perlin-noise fragment-processor lane (perlinnoisefp.go): fractal-noise / turbulence
// shaders converted to a GPU fragment processor and rendered through the fill-rect op, compared against the CPU
// perlin-noise raster stage (the Go-GPU vs Go-CPU self-consistency lane). Before this lane a paint carrying a
// perlin-noise shader failed fragment-processor conversion and the draw was a silent no-op. The GPU path recombines
// ~16-bit gradient components from an 8-bit RGBA texture, so it tracks the full-precision CPU path within a small
// tolerance rather than byte-exact. Skips when no GL context is available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// comparePerlin renders no pixels itself; it compares an already-rendered GPU surface (data, with rowBytes = size*4)
// against the CPU perlin pipeline over the interior pixels, asserting the two agree within perChannelTol at all but a
// small outlier fraction, and that the GPU output actually varies (proving the noise rendered rather than a blank
// no-op).
func comparePerlin(t *testing.T, name string, shader shaders.Shader, paint colorcore.Color, data []byte, size int32) {
	t.Helper()
	const perChannelTol = 3
	rowBytes := size * 4
	var maxDiff, outliers, total int
	var sumDiff int64
	var minR, maxR byte = 255, 0
	for y := int32(1); y < size-1; y++ {
		for x := int32(1); x < size-1; x++ {
			want := pmToBytes(cpuShadePixel(t, shader, geom.IdentityMatrix(), paint, x, y))
			got := data[y*rowBytes+x*4 : y*rowBytes+x*4+4]
			if got[0] < minR {
				minR = got[0]
			}
			if got[0] > maxR {
				maxR = got[0]
			}
			worst := 0
			for c := 0; c < 4; c++ {
				d := int(got[c]) - int(want[c])
				if d < 0 {
					d = -d
				}
				sumDiff += int64(d)
				if d > worst {
					worst = d
				}
			}
			if worst > maxDiff {
				maxDiff = worst
			}
			if worst > perChannelTol {
				outliers++
			}
			total++
		}
	}
	outlierFrac := float64(outliers) / float64(total)
	meanDiff := float64(sumDiff) / float64(total*4)
	t.Logf("%s: maxDiff=%d meanDiff=%.3f outlierFrac=%.4f (R range %d..%d)",
		name, maxDiff, meanDiff, outlierFrac, minR, maxR)
	// A flip/transpose or a wrong lattice lookup would blow past a few percent of pixels.
	if outlierFrac > 0.02 {
		t.Errorf("%s: %.2f%% of pixels differ from the CPU pipeline by >%d (max %d) — GPU perlin "+
			"noise does not track the CPU stage", name, outlierFrac*100, perChannelTol, maxDiff)
	}
	// Before this lane the draw was a silent no-op (a constant transparent surface); a real noise render varies
	// substantially across the surface.
	if maxR-minR < 16 {
		t.Errorf("%s: GPU red channel spans only %d..%d — the noise did not render", name, minR, maxR)
	}
}

// TestLivePerlinNoiseSwatches renders fractal-noise and turbulence shaders (including a single octave, a nonzero seed,
// and a stitch-tiled variant) as a converted fragment processor through the fill-rect op and compares them against the
// CPU perlin-noise stage.
func TestLivePerlinNoiseSwatches(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 32
	ctm := geom.IdentityMatrix()
	white := colorcore.Color4f{R: 1, G: 1, B: 1, A: 1}

	cases := []struct {
		shader shaders.Shader
		name   string
	}{
		{name: "fractal", shader: shaders.NewFractalNoise(0.1, 0.1, 3, 0, 0, 0)},
		{name: "turbulence", shader: shaders.NewTurbulence(0.1, 0.1, 3, 0, 0, 0)},
		{name: "fractal-1octave", shader: shaders.NewFractalNoise(0.15, 0.05, 1, 2, 0, 0)},
		{name: "fractal-anisotropic", shader: shaders.NewFractalNoise(0.2, 0.08, 4, 7, 0, 0)},
		{name: "turbulence-stitch", shader: shaders.NewTurbulence(0.125, 0.125, 2, 0, size, size)},
	}
	for _, tc := range cases {
		if tc.shader == nil {
			t.Fatalf("%s: perlin-noise construction failed", tc.name)
		}
		if _, ok := tc.shader.(*shaders.PerlinNoiseShader); !ok {
			t.Fatalf("%s: expected a *PerlinNoiseShader, got %T (octave collapse?)", tc.name,
				tc.shader)
		}
		sdc := newLiveSDC(t, dc, size)
		sdc.Clear([4]float32{0, 0, 0, 0})
		gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
			Color:     white,
			BlendMode: raster.BlendSrc,
			Shader:    tc.shader,
		}, ctm)
		if !ok {
			t.Fatalf("%s: MakePaint failed (perlin noise unrepresentable on GPU)", tc.name)
		}
		sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: size, Bottom: size})
		data := readSDC(t, sdc)
		comparePerlin(t, tc.name, tc.shader, 0xFFFFFFFF, data, size)
		sweepGLErrors(t, dc.Gpu(), tc.name)
		sdc.Release()
	}
}
