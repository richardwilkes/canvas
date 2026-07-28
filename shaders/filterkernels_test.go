// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// shadeRow compiles s and evaluates n pixels of row y starting at x, through the real pooled Compile path.
func shadeRow(t *testing.T, s Shader, ctm geom.Matrix, x, y int32, n int) []colorcore.PMColor4f {
	t.Helper()
	p := Compile(s, ctm, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	out := make([]colorcore.PMColor4f, n)
	p.ShadeSpan(x, y, out)
	RecyclePipeline(p)
	return out
}

// filterTestChildAt is filterTestChildStage evaluated on the host, for tests that need the child's exact output at a
// known coordinate rather than a same-pipeline reference.
func filterTestChildAt(x, y float32) colorcore.PMColor4f {
	a := clamp01(0.02*x + 0.01*y + 0.2)
	return colorcore.PMColor4f{R: a * clamp01(0.03*x+0.1), G: a * clamp01(0.02*y+0.2), B: a * 0.3, A: a}
}

// TestLinearMorphologyRadiusZeroIsSingleTap pins NewLinearMorphology's documented (2*radius+1) taps at radius 0: the
// one center tap, evaluated at the unshifted coordinate. The form used to be inferred from the radius (0 meaning the
// sparse form), so a zero radius silently became the sparse kernel's two ±offset taps — a dilate/erode by a full offset
// where the caller asked for none. The sparse output is shaded alongside to show the two forms really do differ here,
// i.e. that this test would have failed under the old sentinel.
func TestLinearMorphologyRadiusZeroIsSingleTap(t *testing.T) {
	const row = 3
	const n = 8
	id := geom.IdentityMatrix()
	child := filterTestChild{}
	offset := geom.Point{X: 4}
	for _, tc := range []struct {
		name   string
		dilate bool
	}{{name: "dilate", dilate: true}, {name: "erode", dilate: false}} {
		got := shadeRow(t, NewLinearMorphology(child, offset, tc.dilate, 0), id, 0, row, n)
		sparse := shadeRow(t, NewSparseMorphology(child, offset, tc.dilate), id, 0, row, n)
		differs := false
		for i := range got {
			colorNear(t, got[i], filterTestChildAt(float32(i)+0.5, row+0.5), 0, tc.name+" radius-0 center tap")
			if got[i] != sparse[i] {
				differs = true
			}
		}
		if !differs {
			t.Fatalf("%s: the sparse form matched the single tap everywhere, so the case is untested", tc.name)
		}
	}
}

// TestFilterDecalContextReuseIsClean locks the pooled-Pipeline invariant for the decal ramp kernel, which moved off a
// per-compile new([2][stride]float32) plus two capturing closures: compiled into a pipeline whose retained
// filterDecalCtx carries a prior draw's stale coordinates and ramp bounds, it must produce byte-identical output to a
// fresh compile (the coordinates are write-before-read scratch and the bounds are overwritten on handout).
func TestFilterDecalContextReuseIsClean(t *testing.T) {
	id := geom.IdentityMatrix()
	sh := NewFilterDecal(filterTestChild{}, geom.Rect{Left: 2, Top: 1, Right: 9, Bottom: 6})

	var want [16]colorcore.PMColor4f
	ref := &Pipeline{}
	ref.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(ref, newMatrixRec(id)) {
		t.Fatal("reference appendStages failed")
	}
	ref.ShadeSpan(0, 3, want[:])

	poison := &filterDecalCtx{l: 9, t: 9, r: 9, b: 9}
	for i := range poison.coords[0] {
		poison.coords[0][i] = 9
		poison.coords[1][i] = 9
	}
	p := &Pipeline{filterDecalCtxs: []*filterDecalCtx{poison}}
	p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(p, newMatrixRec(id)) {
		t.Fatal("poisoned appendStages failed")
	}
	var got [16]colorcore.PMColor4f
	p.ShadeSpan(0, 3, got[:])
	for i := range got {
		colorNear(t, got[i], want[i], 0, "filter-decal reused-context pixel")
	}
	if p.filterDecalCtxN != 1 || p.filterDecalCtxs[0] != poison {
		t.Fatalf("expected the poisoned ctx to be reused (n=%d)", p.filterDecalCtxN)
	}
	// The ramp must actually be doing something, or a stale-bounds bug would be invisible above.
	if want[0] == filterTestChildAt(0.5, 3.5) {
		t.Fatal("the decal ramp left the child's output unchanged, so the case is untested")
	}
}

// TestFilterDecalsUseDistinctContexts locks that two decal ramps in one pipeline (an arithmetic blend inlines both) get
// their own pooled context — a shared one would let the second ramp's bounds and saved coordinates clobber the first's
// at ShadeSpan time.
func TestFilterDecalsUseDistinctContexts(t *testing.T) {
	child := filterTestChild{}
	d1 := NewFilterDecal(child, geom.Rect{Left: 0, Top: 0, Right: 8, Bottom: 8})
	d2 := NewFilterDecal(child, geom.Rect{Left: 4, Top: 4, Right: 20, Bottom: 20})
	p := Compile(NewArithmeticBlend([4]float32{0.5, 0.25, 0.25, 0}, false, d1, d2),
		geom.IdentityMatrix(), opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.filterDecalCtxN != 2 {
		t.Fatalf("expected 2 filter-decal contexts, got %d", p.filterDecalCtxN)
	}
	if p.filterDecalCtxs[0] == p.filterDecalCtxs[1] {
		t.Fatal("the two decal ramps share a context (would clobber)")
	}
	RecyclePipeline(p)
}
