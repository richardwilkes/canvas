// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for Device.DrawImageRect's edge-AA selection, which must come from chooseAA like every other draw entry point so
// a DMSAA surface never sees an AANo quad (the invariant DrawFilledQuad documents).

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/shaders"
)

// TestDrawImageRectUsesChooseAA draws a default (non-antialiased) paint through a rotating matrix, the configuration
// whose quad edges are visibly aliased when the AA choice ignores DMSAA. On a DMSAA surface the draw must promote to
// AAYes, which routes it through the FillRRect lane instead of the AANo TextureOp lane.
func TestDrawImageRectUsesChooseAA(t *testing.T) {
	for _, c := range []struct {
		name        string
		dmsaa       bool
		wantPromote bool
	}{
		{name: "dmsaa surface promotes to AAYes", dmsaa: true, wantPromote: true},
		{name: "plain surface honors the paint", dmsaa: false},
	} {
		t.Run(c.name, func(t *testing.T) {
			dc := newFakeDirectContext(t)
			sdc := newDMSAATestSDC(t, dc, c.dmsaa)
			defer sdc.Release()
			if c.dmsaa && !sdc.CanUseDynamicMSAA() {
				t.Fatal("DMSAA surface props did not enable dynamic MSAA")
			}

			src := makeDeferredProxy(t, dc, geom.ISize{Width: 16, Height: 16}, gpu.RenderableNo,
				gpu.BackingFitExact)
			defer src.Unref()
			img := borrowTextureImage(dc, MakeSurfaceProxyViewDefault(src), gpu.ColorTypeRGBA8888,
				geom.ISize{Width: 16, Height: 16})

			dev := NewDevice(sdc)
			rotate := geom.Matrix{}
			rotate.SetRotate(20)
			dev.SetGlobalCTM(&rotate)
			paint := canvas.NewPaint() // AntiAlias is false
			dev.DrawImageRect(img, nil, geom.RectWH(16, 16), shaders.SamplingOptions{}, paint,
				canvas.ConstraintFast)

			head := sdc.GetOpsTask().GetChainHead(0)
			if head == nil {
				t.Fatal("no op recorded")
			}
			_, isTexture := head.(*textureOp)
			if isTexture == c.wantPromote {
				t.Fatalf("recorded %s (TextureOp = %v), want promote-to-AAYes = %v", head.Name(),
					isTexture, c.wantPromote)
			}
		})
	}
}
