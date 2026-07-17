// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for DrawAtlasOp: the batching requirement (an N-sprite call is one draw after combine), the merge
// rules (color-inclusion, constant color, view matrix), and the per-sprite color vertex lane.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// atlasSprites builds n unit sprites: sprite i selects an 8x8 tex rect and lands translated at x = 10*i.
func atlasSprites(n int) (xforms []geom.RSXform, tex []geom.Rect) {
	xforms = make([]geom.RSXform, n)
	tex = make([]geom.Rect, n)
	for i := range n {
		xforms[i] = geom.MakeRSXform(1, 0, float32(10*i), 4)
		tex[i] = geom.RectXYWH(float32(8*(i%4)), 0, 8, 8)
	}
	return xforms, tex
}

func chainHeadDrawAtlas(t *testing.T, sdc *SurfaceDrawContext, index int) *drawAtlasOp {
	t.Helper()
	op, ok := sdc.GetOpsTask().GetChainHead(index).(*drawAtlasOp)
	if !ok {
		t.Fatalf("chain %d head is not a drawAtlasOp", index)
	}
	return op
}

// TestDrawAtlasOpBatching is the batching assert: an N-sprite DrawAtlas call is a single op, a compatible second call
// merges into it, and the merged op flushes as exactly one GL draw.
func TestDrawAtlasOpBatching(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	xforms, tex := atlasSprites(5)
	sdc.DrawAtlas(nil, solidPaint(1, 0, 0, 0.5), &identity, xforms, tex, nil)
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("op chains = %d, want 1", got)
	}
	if got := chainHeadDrawAtlas(t, sdc, 0).NumQuads(); got != 5 {
		t.Fatalf("op quads = %d, want 5", got)
	}

	// A second compatible call (same trivial paint, same view matrix, no colors) merges.
	xforms2, tex2 := atlasSprites(3)
	sdc.DrawAtlas(nil, solidPaint(1, 0, 0, 0.5), &identity, xforms2, tex2, nil)
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("op chains after merge = %d, want 1", got)
	}
	if got := chainHeadDrawAtlas(t, sdc, 0).NumQuads(); got != 8 {
		t.Fatalf("merged op quads = %d, want 8", got)
	}

	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawElementsInstancedBaseVertexBaseInstance") + counts("glDrawArrays"); draws != 1 {
		t.Fatalf("draw calls = %d, want 1 for the merged atlas op", draws)
	}
}

// TestDrawAtlasOpColorsBatching covers the per-sprite-color lane: colored calls merge with each other but not with
// colorless ones (the color-inclusion rule), and constant-color ops with different colors do not merge.
func TestDrawAtlasOpColorsBatching(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	xforms, tex := atlasSprites(4)
	colors := []colorcore.Color{0xFFFF0000, 0x8000FF00, 0xFF0000FF, 0x40FFFFFF}

	sdc.DrawAtlas(nil, solidPaint(1, 1, 1, 0.5), &identity, xforms, tex, colors)
	sdc.DrawAtlas(nil, solidPaint(1, 1, 1, 0.5), &identity, xforms, tex, colors)
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("colored op chains = %d, want 1 (merged)", got)
	}
	head := chainHeadDrawAtlas(t, sdc, 0)
	if got := head.NumQuads(); got != 8 {
		t.Fatalf("merged colored op quads = %d, want 8", got)
	}
	if !head.hasColors {
		t.Fatal("merged colored op lost its color attribute")
	}

	// A colorless call must not merge with the colored op.
	sdc.DrawAtlas(nil, solidPaint(1, 1, 1, 0.5), &identity, xforms, tex, nil)
	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("op chains after colorless call = %d, want 2", got)
	}

	// Colorless calls with different paint colors must not merge either (constant-color rule).
	sdc.DrawAtlas(nil, solidPaint(0, 1, 0, 0.5), &identity, xforms, tex, nil)
	if got := sdc.GetOpsTask().NumOpChains(); got != 3 {
		t.Fatalf("op chains after different-color call = %d, want 3", got)
	}
	dc.FlushAndSubmit(false)
}

// TestDrawAtlasOpNoMergeAcrossViewMatrix pins the uniform-view-matrix merge rule.
func TestDrawAtlasOpNoMergeAcrossViewMatrix(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()
	rot := geom.RotateDegMatrix(30)
	rot.PostTranslate(32, 0)

	xforms, tex := atlasSprites(2)
	sdc.DrawAtlas(nil, solidPaint(1, 0, 0, 0.5), &identity, xforms, tex, nil)
	sdc.DrawAtlas(nil, solidPaint(1, 0, 0, 0.5), &rot, xforms, tex, nil)
	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("op chains = %d, want 2 (different view matrices must not merge)", got)
	}
	dc.FlushAndSubmit(false)
}
