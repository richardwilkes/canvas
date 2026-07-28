// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the bound on uniquely-keyed LUT texture proxies: the perlin-noise tables and baked gradient ramps mint a
// fresh key per shader instance / per bake, so without eviction an animated shader grows the proxy provider's map (and the
// textures behind it) by a couple of entries per frame, forever.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/shaders"
)

// perlinTables registers both painting-data tables of s, the way makePerlinNoiseFP does.
func perlinTables(t *testing.T, dc *DirectContext, s *shaders.PerlinNoiseShader) {
	t.Helper()
	perm := perlinTextureView(dc, s, 0, geom.ISize{Width: perlinBlockSizePx, Height: 1},
		gpu.ColorTypeAlpha8, s.PermutationsBitmap(), perlinBlockSizePx, "test-perm")
	noise := perlinTextureView(dc, s, 1, geom.ISize{Width: perlinBlockSizePx, Height: 4},
		gpu.ColorTypeRGBA8888, s.NoiseBitmap(), perlinBlockSizePx*4, "test-noise")
	if perm.Proxy() == nil || noise.Proxy() == nil {
		t.Fatal("perlin table proxies were not created")
	}
}

// TestPerlinLUTProxiesAreBounded walks many turbulence shaders — one per frame, as an animation does — and checks the
// provider's uniquely-keyed proxy count stops growing instead of gaining two entries per shader.
func TestPerlinLUTProxiesAreBounded(t *testing.T) {
	dc := newFakeDirectContext(t)
	frames := maxLiveLUTProxies + 40
	for i := 0; i < frames; i++ {
		s, ok := shaders.NewTurbulence(0.05, 0.05, 2, float32(i), 0, 0).(*shaders.PerlinNoiseShader)
		if !ok {
			t.Fatalf("frame %d: NewTurbulence did not return a perlin shader", i)
		}
		perlinTables(t, dc, s)
		if got := dc.ProxyProvider().NumUniqueKeyProxies(); got > maxLiveLUTProxies {
			t.Fatalf("frame %d: %d uniquely-keyed proxies, want at most %d", i, got,
				maxLiveLUTProxies)
		}
	}
	if got := dc.lutProxyKeys.numTracked(); got != maxLiveLUTProxies {
		t.Errorf("tracked LUT keys = %d, want the cache full at %d", got, maxLiveLUTProxies)
	}
}

// TestLUTProxyCacheKeepsTheWorkingSet checks the other half of the bound: a LUT that keeps being used is refreshed on
// every request, so it is never evicted in favor of one-shot keys — an animated shader must not cost the dither table or a
// steadily drawn gradient its texture.
func TestLUTProxyCacheKeepsTheWorkingSet(t *testing.T) {
	dc := newFakeDirectContext(t)
	steady, ok := shaders.NewTurbulence(0.05, 0.05, 2, 1, 0, 0).(*shaders.PerlinNoiseShader)
	if !ok {
		t.Fatal("NewTurbulence did not return a perlin shader")
	}
	perlinTables(t, dc, steady)
	steadyKeyed := dc.ProxyProvider().NumUniqueKeyProxies()

	for i := 0; i < maxLiveLUTProxies+40; i++ {
		s, isPerlin := shaders.NewTurbulence(0.05, 0.05, 2, float32(100+i),
			0, 0).(*shaders.PerlinNoiseShader)
		if !isPerlin {
			t.Fatalf("frame %d: NewTurbulence did not return a perlin shader", i)
		}
		perlinTables(t, dc, s)
		// Re-request the steady shader's tables, as a redraw of it would.
		perlinTables(t, dc, steady)
	}
	if got := dc.ProxyProvider().NumUniqueKeyProxies(); got > maxLiveLUTProxies {
		t.Fatalf("%d uniquely-keyed proxies, want at most %d", got, maxLiveLUTProxies)
	}
	// Its keys are still registered, so re-requesting them finds the existing proxies rather than uploading again.
	before := dc.ProxyProvider().NumUniqueKeyProxies()
	perlinTables(t, dc, steady)
	if got := dc.ProxyProvider().NumUniqueKeyProxies(); got != before || before < steadyKeyed {
		t.Errorf("steady LUT keys were evicted: %d proxies (was %d, %d after the first use)", got,
			before, steadyKeyed)
	}
}

// TestGradientRampLUTProxiesAreBounded covers the gradient lane's own fresh-per-bake key through the textured colorizer.
func TestGradientRampLUTProxiesAreBounded(t *testing.T) {
	dc := newFakeDirectContext(t)
	args := &FPArgs{Ctx: dc, Caps: dc.GLCaps(), DstColorType: gpu.ColorTypeRGBA8888}
	positions := []float32{0, 0.5, 1}
	for i := 0; i < maxLiveLUTProxies+40; i++ {
		// A distinct gradient per iteration, so each one bakes a fresh ramp under a fresh key.
		colors := []colorcore.Color4f{
			{R: float32(i%255) / 255, A: 1},
			{G: float32((i*7)%255) / 255, A: 1},
			{B: 1, A: 1},
		}
		if fp := makeTexturedColorizer(colors, positions, true, args); fp == nil {
			t.Fatalf("iteration %d: makeTexturedColorizer failed", i)
		}
		if got := dc.ProxyProvider().NumUniqueKeyProxies(); got > maxLiveLUTProxies {
			t.Fatalf("iteration %d: %d uniquely-keyed proxies, want at most %d", i, got,
				maxLiveLUTProxies)
		}
	}
}
