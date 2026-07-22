// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Regression test for CopySurfaceProxy sizing the destination proxy from the intersected srcRect (matching Skia) rather
// than the pre-intersect request, so an oversized srcRect does not leave an uncopied border in the dst.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

func TestCopySurfaceProxyClampsToIntersectedRect(t *testing.T) {
	dc := newShaderRecordingContext(t)
	caps := dc.GLCaps()
	format := caps.GetDefaultBackendFormat(gpu.ColorTypeRGBA8888, false)
	src := dc.ProxyProvider().CreateProxy(format, geom.ISize{Width: 8, Height: 8},
		gpu.RenderableNo, 1, gpu.MipmappedNo, gpu.BackingFitExact, gpu.BudgetedYes, "copy-src",
		0, UseAllocatorYes)
	if src == nil {
		t.Fatal("CreateProxy failed")
	}
	defer src.Unref()

	// srcRect extends past the 8x8 source; CopySurfaceProxy must size the dst from the intersected 8x8 rect, not the
	// oversized request (which would leave a 12px uncopied border).
	srcRect := geom.IRectXYWH(0, 0, 20, 20)
	dst, task := CopySurfaceProxy(dc, src, gpu.OriginTopLeft, gpu.MipmappedNo, srcRect,
		gpu.BackingFitExact, gpu.BudgetedYes, "copy-dst")
	if dst == nil || task == nil {
		t.Fatal("CopySurfaceProxy returned nil")
	}
	defer dst.Unref()
	if got := dst.Dimensions(); got.Width != 8 || got.Height != 8 {
		t.Errorf("dst proxy dims = %dx%d, want 8x8 (the intersected srcRect)", got.Width, got.Height)
	}
}
