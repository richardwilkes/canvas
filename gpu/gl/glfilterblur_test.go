// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// White-box live-context test for the GPU blur algorithm's upload lane (glfilter.go): a raster-backed source is
// uploaded through the shared image-proxy cache, and the TextureImage that upload produces owns a proxy ref the lane
// must drop — otherwise every blur of a non-texture-backed special image pins another texture for the context's
// lifetime. This lives in package gl (not gl_test) because the assertion reads the proxy provider's uniquely-keyed
// proxy map, which is where the pinned proxies would accumulate. Skips when no GL context is available.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl/gltest"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

func TestBlurUploadLaneDropsItsTextureRef(t *testing.T) {
	glCtx, err := gltest.New()
	if err != nil {
		t.Skipf("no GL context available: %v", err)
	}
	defer glCtx.Destroy()
	dc := MakeGLDirectContext(MakeNativeInterface(), nil)
	if dc == nil {
		t.Fatal("MakeGLDirectContext returned nil on a live context")
	}
	const size = 16
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: size, Height: size}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "BlurUploadLaneTest")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext returned nil")
	}
	algorithm := &blurAlgorithm{dev: NewDevice(sdc)}

	// A raster-backed special image: NewSpecialImageDrawable downgrades an imagecore.Image to the pixels lane, so the
	// blur has no drawable backing to sample and must take the upload lane.
	pm := raster.NewPixmap(size, size)
	for i := range pm.Pix {
		pm.Pix[i] = 0xFF0000FF // opaque red (RGBA word, R in the low byte)
	}
	img := imagecore.Copy(pm)
	if img == nil {
		t.Fatal("imagecore.Copy returned nil")
	}
	src := filtercore.NewSpecialImageDrawable(geom.IRectWH(size, size), img)
	if src == nil {
		t.Fatal("NewSpecialImageDrawable returned nil")
	}
	if src.DrawableBacking() != nil {
		t.Fatal("test source must be raster-backed so the blur takes the upload lane")
	}

	pp := dc.ProxyProvider()
	before := make(map[string]struct{}, len(pp.uniquelyKeyedProxies))
	for key := range pp.uniquelyKeyedProxies {
		before[key] = struct{}{}
	}

	bounds := geom.IRectWH(size, size)
	if out := algorithm.Blur(geom.Size{Width: 2, Height: 2}, src, bounds, shaders.TileDecal,
		bounds); out == nil {
		t.Fatal("Blur returned nil for a raster-backed source")
	}

	var uploaded []*SurfaceProxy
	for key, proxy := range pp.uniquelyKeyedProxies {
		if _, existed := before[key]; !existed {
			uploaded = append(uploaded, proxy)
		}
	}
	if len(uploaded) != 1 {
		t.Fatalf("the upload lane registered %d new uniquely-keyed proxies, want 1", len(uploaded))
	}
	// One ref: the creation ref the image-proxy cache map holds. A second one means the lane kept the TextureImage's
	// ref, so the upload can never be unkeyed or purged.
	if got := uploaded[0].RefCnt(); got != 1 {
		t.Errorf("uploaded blur source proxy ref count = %d, want 1 — the upload lane leaked its "+
			"TextureImage ref", got)
	}
}
