// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the surface-level caps queries.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// TestSurfaceSupportsReadPixels pins the one condition the query actually evaluates: a multisampled render target that
// is not a texture has no resolve texture (and so no second FBO to resolve through), and must be copied before it can
// be read back. Everything else reads back directly.
func TestSurfaceSupportsReadPixels(t *testing.T) {
	dc := newFakeDirectContext(t)
	defer dc.Destroy()
	caps := dc.GLCaps()
	dims := geom.ISize{Width: 16, Height: 16}

	// A pure render target (a wrapped FBO, never a texture), single-sampled: read directly.
	single := dc.Gpu().WrapBackendRenderTarget(dims, FormatRGBA8, 1, 8, 0)
	if single == nil {
		t.Fatal("WrapBackendRenderTarget failed")
	}
	defer single.Unref()
	if single.AsTexture() != nil || single.AsRenderTarget() == nil {
		t.Fatal("expected a pure render target")
	}
	if got := caps.SurfaceSupportsReadPixels(single); got != SurfaceReadPixelsSupported {
		t.Fatalf("single-sampled render target = %v, want SurfaceReadPixelsSupported", got)
	}

	// The same, multisampled: no resolve texture, so it must be copied first.
	msaa := dc.Gpu().WrapBackendRenderTarget(dims, FormatRGBA8, 4, 8, 0)
	if msaa == nil {
		t.Skip("profile cannot wrap a 4x multisampled render target")
	}
	defer msaa.Unref()
	if msaa.AsRenderTarget().NumSamples() <= 1 {
		t.Fatalf("wrapped sample count = %d, want > 1", msaa.AsRenderTarget().NumSamples())
	}
	if got := caps.SurfaceSupportsReadPixels(msaa); got != SurfaceReadPixelsCopyToTexture2D {
		t.Fatalf("multisampled render target = %v, want SurfaceReadPixelsCopyToTexture2D", got)
	}

	// A multisampled render target that *is* a texture has a resolve texture, so it reads back directly.
	texRT := dc.ResourceProvider().CreateTexture(dims, FormatRGBA8, gpu.TextureType2D,
		gpu.RenderableYes, 4, gpu.MipmappedNo, gpu.BudgetedYes, "msaa-texture-rt")
	if texRT == nil {
		t.Skip("profile cannot create a 4x multisampled renderable texture")
	}
	defer texRT.Unref()
	if texRT.AsTexture() == nil || texRT.AsRenderTarget() == nil {
		t.Fatal("expected a texture-backed render target")
	}
	if texRT.AsRenderTarget().NumSamples() <= 1 {
		t.Fatalf("texture render target sample count = %d, want > 1",
			texRT.AsRenderTarget().NumSamples())
	}
	if got := caps.SurfaceSupportsReadPixels(texRT); got != SurfaceReadPixelsSupported {
		t.Fatalf("multisampled texture render target = %v, want SurfaceReadPixelsSupported", got)
	}
}
