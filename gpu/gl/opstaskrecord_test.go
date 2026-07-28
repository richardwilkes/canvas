// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for OpsTask's op-recording bookkeeping: the dst-proxy reference recordOp consumes must be accounted for on every
// exit, including the paths that drop the op.

package gl

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// TestOpsTaskRecordOpReleasesDstProxy covers the reference setupDstProxyView leaves on the dst proxy for recordOp to
// consume: the merge path releases it and makeOpChain transfers it into the chain, so the non-finite-bounds early return
// (reachable through the exported OpsTask.AddDrawOp) must release it too or the backing texture never returns to the
// resource cache.
func TestOpsTaskRecordOpReleasesDstProxy(t *testing.T) {
	inf := float32(math.Inf(1))
	for _, c := range []struct {
		name   string
		bounds geom.Rect
		kept   bool
	}{
		{name: "finite bounds are recorded", bounds: geom.RectLTRB(0, 0, 20, 20), kept: true},
		{name: "non-finite bounds are dropped", bounds: geom.RectLTRB(0, 0, inf, inf)},
	} {
		t.Run(c.name, func(t *testing.T) {
			dc := newFakeDirectContext(t)
			sdc := newDrawTestSDC(t, dc, 64, 64)
			defer sdc.Release()

			dstProxy := makeDeferredProxy(t, dc, geom.ISize{Width: 64, Height: 64},
				gpu.RenderableYes, gpu.BackingFitExact)
			defer dstProxy.Unref()
			var dst DstProxyView
			dst.SetProxyView(MakeSurfaceProxyView(dstProxy, gpu.OriginTopLeft, gpu.SwizzleRGBA))
			// The reference setupDstProxyView takes on the caller's behalf and hands to recordOp.
			dstProxy.Ref()
			before := dstProxy.RefCnt()

			op, _ := applyTestOp(geom.RectLTRB(0, 0, 20, 20))
			op.opBase().SetBounds(c.bounds, HasAABloat(false), IsHairline(false))
			analysis := ProcessorAnalysis{isInitialized: true, requiresDstTexture: true}
			sdc.GetOpsTask().AddDrawOp(dc.DrawingManager(), op, false, analysis,
				MakeAppliedClip(sdc.Dimensions()), &dst, dc.GLCaps())

			got := dstProxy.RefCnt()
			if c.kept {
				// The recorded chain owns the reference until the task's ops are deleted.
				if got != before {
					t.Fatalf("recorded op: dst proxy refCnt = %d, want %d", got, before)
				}
				sdc.GetOpsTask().deleteOps()
				if got = dstProxy.RefCnt(); got != before-1 {
					t.Fatalf("after deleteOps: dst proxy refCnt = %d, want %d", got, before-1)
				}
				return
			}
			if got != before-1 {
				t.Fatalf("dropped op: dst proxy refCnt = %d, want %d (the reference leaked)", got,
					before-1)
			}
		})
	}
}
