// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for FillRectOp and the SurfaceDrawContext draw funnel: op merging (the batching requirement's
// ops-recorded vs draws-issued shape), none↔coverage AA upgrades on merge, XP-compatibility gating, the
// attemptQuadOptimization clear conversions, and the FixedClip scissor lanes through AddDrawOp.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

func newDrawTestSDC(t *testing.T, dc *DirectContext, w, h int32) *SurfaceDrawContext {
	t.Helper()
	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "fill-rect-test")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	return sdc
}

// solidPaint builds a trivial paint whose blended output is not constant (translucent src-over), so
// attemptQuadOptimization cannot convert draws into clears.
func solidPaint(r, g, b, a float32) *Paint {
	paint := NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: r, G: g, B: b, A: a})
	return paint
}

func chainHeadFillRect(t *testing.T, sdc *SurfaceDrawContext, index int) *fillRectOp {
	t.Helper()
	op, ok := sdc.GetOpsTask().GetChainHead(index).(*fillRectOp)
	if !ok {
		t.Fatalf("chain %d head is not a fillRectOp", index)
	}
	return op
}

func TestFillRectOpBatching(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	// Four non-overlapping rects with distinct colors but identical (trivial) paints: all merge into one op at record
	// time, with the colors riding the per-quad metadata.
	for i := range 4 {
		x := float32(i * 16)
		sdc.FillRectToRect(nil, solidPaint(float32(i)*0.25, 0, 0, 0.5), gpu.AANo, &identity,
			geom.Rect{Left: x, Top: 0, Right: x + 8, Bottom: 8},
			geom.Rect{Left: x, Top: 0, Right: x + 8, Bottom: 8})
	}
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("op chains = %d, want 1 (merged)", got)
	}
	head := chainHeadFillRect(t, sdc, 0)
	if head.NumQuads() != 4 {
		t.Fatalf("merged op quads = %d, want 4", head.NumQuads())
	}

	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawArrays"); draws != 1 {
		t.Fatalf("draw calls = %d, want 1 for the merged op", draws)
	}
}

func TestFillRectOpAAUpgradeOnMerge(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	// A non-AA rect followed by a fractional coverage-AA rect with the same paint: the merge lifts the stored AA type
	// to coverage.
	sdc.FillRectToRect(nil, solidPaint(1, 0, 0, 0.5), gpu.AANo, &identity,
		geom.Rect{Left: 2, Top: 2, Right: 10, Bottom: 10},
		geom.Rect{Left: 2, Top: 2, Right: 10, Bottom: 10})
	sdc.FillRectToRect(nil, solidPaint(0, 1, 0, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 20.5, Top: 2.5, Right: 30.25, Bottom: 10.75},
		geom.Rect{Left: 20.5, Top: 2.5, Right: 30.25, Bottom: 10.75})

	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("op chains = %d, want 1 (none+coverage merged)", got)
	}
	head := chainHeadFillRect(t, sdc, 0)
	if head.NumQuads() != 2 {
		t.Fatalf("merged op quads = %d, want 2", head.NumQuads())
	}
	if head.helper.AAType() != gpu.AATypeCoverage {
		t.Fatalf("merged AA type = %v, want coverage", head.helper.AAType())
	}
	dc.FlushAndSubmit(false)
}

func TestFillRectOpNoMergeAcrossXP(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	paintA := solidPaint(1, 0, 0, 0.5)
	paintB := solidPaint(1, 0, 0, 0.5)
	paintB.SetPorterDuffXPFactory(raster.BlendSrcIn)

	sdc.FillRectToRect(nil, paintA, gpu.AANo, &identity,
		geom.Rect{Left: 2, Top: 2, Right: 10, Bottom: 10},
		geom.Rect{Left: 2, Top: 2, Right: 10, Bottom: 10})
	sdc.FillRectToRect(nil, paintB, gpu.AANo, &identity,
		geom.Rect{Left: 20, Top: 2, Right: 28, Bottom: 10},
		geom.Rect{Left: 20, Top: 2, Right: 28, Bottom: 10})

	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("op chains = %d, want 2 (different XPs must not merge)", got)
	}
	dc.FlushAndSubmit(false)
}

func TestFillRectClearConversions(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 400, 400)
	defer sdc.Release()
	identity := geom.IdentityMatrix()
	full := geom.Rect{Right: 400, Bottom: 400}

	// A fullscreen opaque src-over constant-color rect becomes a load-op clear (no op chain).
	sdc.FillRectToRect(nil, solidPaint(0, 0, 1, 1), gpu.AANo, &identity, full, full)
	if got := sdc.GetOpsTask().NumOpChains(); got != 0 {
		t.Fatalf("fullscreen const rect: op chains = %d, want 0 (load-op clear)", got)
	}

	// A pixel-aligned >256px partial rect becomes a scissored ClearOp.
	part := geom.Rect{Right: 300, Bottom: 300}
	sdc.FillRectToRect(nil, solidPaint(1, 0, 0, 1), gpu.AANo, &identity, part, part)
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("partial const rect: op chains = %d, want 1", got)
	}
	if name := sdc.GetOpsTask().GetChainHead(0).Name(); name != "Clear" {
		t.Fatalf("partial const rect became %q, want Clear", name)
	}

	// A small rect still draws.
	small := geom.Rect{Left: 2, Top: 2, Right: 20, Bottom: 20}
	sdc.FillRectToRect(nil, solidPaint(0, 1, 0, 1), gpu.AANo, &identity, small, small)
	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("small const rect: op chains = %d, want 2", got)
	}
	if name := sdc.GetOpsTask().GetChainHead(1).Name(); name != "FillRectOp" {
		t.Fatalf("small const rect became %q, want FillRectOp", name)
	}

	// A draw entirely outside the render target is discarded.
	out := geom.Rect{Left: 500, Top: 500, Right: 600, Bottom: 600}
	sdc.FillRectToRect(nil, solidPaint(0, 1, 1, 0.5), gpu.AANo, &identity, out, out)
	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("offscreen rect: op chains = %d, want 2 (discarded)", got)
	}
	dc.FlushAndSubmit(false)
}

func TestFixedClipScissorLanes(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	dims := geom.ISize{Width: 64, Height: 64}
	identity := geom.IdentityMatrix()

	// An axis-aligned draw against a rect clip folds the clip into the quad (no scissor).
	clip := NewFixedClipRect(dims, geom.IRectXYWH(8, 8, 16, 16))
	sdc.FillRectToRect(clip, solidPaint(1, 0, 0, 0.5), gpu.AANo, &identity,
		geom.Rect{Right: 64, Bottom: 64}, geom.Rect{Right: 64, Bottom: 64})
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("folded clip: op chains = %d, want 1", got)
	}
	head := chainHeadFillRect(t, sdc, 0)
	if want := (geom.Rect{Left: 8, Top: 8, Right: 24, Bottom: 24}); head.Bounds() != want {
		t.Fatalf("folded clip bounds = %v, want %v", head.Bounds(), want)
	}

	// A rotated draw cannot fold a partially-overlapping rect clip, so the scissor state is applied at flush.
	rot45 := geom.RotateDegMatrix(45)
	rot45.PostTranslate(32, 0)
	sdc.FillRectToRect(clip, solidPaint(0, 1, 0, 0.5), gpu.AANo, &rot45,
		geom.Rect{Right: 40, Bottom: 40}, geom.Rect{Right: 40, Bottom: 40})
	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("scissored draw: op chains = %d, want 2", got)
	}

	// A draw fully outside the scissor is clipped out entirely.
	sdc.FillRectToRect(clip, solidPaint(0, 0, 1, 0.5), gpu.AANo, &identity,
		geom.Rect{Left: 40, Top: 40, Right: 60, Bottom: 60},
		geom.Rect{Left: 40, Top: 40, Right: 60, Bottom: 60})
	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("clipped-out draw: op chains = %d, want 2", got)
	}

	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if counts("glScissor") == 0 {
		t.Fatal("expected a glScissor from the scissored draw")
	}
}

func TestFixedClipChainCompatibility(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	dims := geom.ISize{Width: 64, Height: 64}

	// Rotated quads whose bounds fully cover the same scissor produce identical applied scissors (the intersection of
	// draw bounds and clip) and merge; a different scissor rect blocks the merge. (Draws that only partially cover the
	// scissor get per-draw intersected scissors and legitimately do not merge, matching FixedClip.Apply.)
	rotA := geom.RotateDegMatrix(10)
	rotA.PostTranslate(-20, -20)
	rotB := geom.RotateDegMatrix(-10)
	rotB.PostTranslate(-24, -16)

	clip1 := NewFixedClipRect(dims, geom.IRectXYWH(8, 8, 16, 16))
	clip2 := NewFixedClipRect(dims, geom.IRectXYWH(6, 6, 20, 20))

	draw := func(clip Clip, view *geom.Matrix) {
		sdc.FillRectToRect(clip, solidPaint(1, 1, 0, 0.5), gpu.AANo, view,
			geom.Rect{Right: 100, Bottom: 100}, geom.Rect{Right: 100, Bottom: 100})
	}
	draw(clip1, &rotA)
	draw(clip1, &rotB)
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("same-scissor draws: op chains = %d, want 1", got)
	}
	draw(clip2, &rotA)
	if got := sdc.GetOpsTask().NumOpChains(); got != 2 {
		t.Fatalf("different-scissor draw: op chains = %d, want 2", got)
	}
	dc.FlushAndSubmit(false)
}

func TestDrawQuadSetBatching(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 128, 128)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	entries := make([]QuadSetEntry, 5)
	for i := range entries {
		x := float32(i * 20)
		entries[i] = QuadSetEntry{
			Rect:        geom.Rect{Left: x + 0.5, Top: 0.5, Right: x + 10.5, Bottom: 10.5},
			Color:       colorcore.PMColor4f{R: float32(i) * 0.2, G: 0, B: 0, A: 0.5},
			LocalMatrix: identity,
			AAFlags:     gpu.QuadAAFlagsAll,
		}
	}
	sdc.DrawQuadSet(nil, solidPaint(1, 1, 1, 0.5), &identity, entries)
	if got := sdc.GetOpsTask().NumOpChains(); got != 1 {
		t.Fatalf("quad set: op chains = %d, want 1", got)
	}
	head := chainHeadFillRect(t, sdc, 0)
	if head.NumQuads() != len(entries) {
		t.Fatalf("quad set op quads = %d, want %d", head.NumQuads(), len(entries))
	}
	if head.helper.AAType() != gpu.AATypeCoverage {
		t.Fatalf("quad set AA type = %v, want coverage", head.helper.AAType())
	}

	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawArrays"); draws != 1 {
		t.Fatalf("quad set draw calls = %d, want 1", draws)
	}
}
