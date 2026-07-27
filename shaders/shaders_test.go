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
	"github.com/richardwilkes/canvas/raster"
)

const opaqueBlack = colorcore.Color(0xFF000000)

// shadeAt compiles the shader with the given CTM and evaluates the single pixel whose center is (x+0.5, y+0.5).
func shadeAt(t *testing.T, s Shader, ctm geom.Matrix, x, y int32) colorcore.PMColor4f {
	t.Helper()
	p := Compile(s, ctm, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	var out [1]colorcore.PMColor4f
	p.ShadeSpan(x, y, out[:])
	return out[0]
}

func near(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func colorNear(t *testing.T, got, want colorcore.PMColor4f, tol float32, what string) {
	t.Helper()
	if !near(got.R, want.R, tol) || !near(got.G, want.G, tol) ||
		!near(got.B, want.B, tol) || !near(got.A, want.A, tol) {
		t.Fatalf("%s: got %+v, want %+v (tol %g)", what, got, want, tol)
	}
}

func TestColorShaderConstantCollapse(t *testing.T) {
	s := NewColor(0x80FF0000)
	if !s.IsConstant() || s.IsOpaque() {
		t.Fatal("translucent color shader traits wrong")
	}
	p := Compile(s, geom.IdentityMatrix(), opaqueBlack)
	c := p.EvalConstant()
	a := float32(0x80) * (1.0 / 255.0)
	colorNear(t, c, colorcore.PMColor4f{R: a, A: a}, 1e-6, "premul translucent red")

	if !NewColor(0xFF123456).IsOpaque() {
		t.Fatal("opaque color shader must be opaque")
	}
}

func TestLinearGradientTwoStop(t *testing.T) {
	s := NewLinearGradient(geom.Point{X: 0, Y: 0}, geom.Point{X: 100, Y: 0},
		[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, TileClamp, nil)
	if s.IsConstant() || !s.IsOpaque() {
		t.Fatal("gradient traits wrong")
	}
	id := geom.IdentityMatrix()
	// Pixel x has center x+0.5, so t = (x+0.5)/100.
	for _, tc := range []struct {
		x int32
		t float32
	}{{x: 0, t: 0.005}, {x: 49, t: 0.495}, {x: 99, t: 0.995}} {
		got := shadeAt(t, s, id, tc.x, 7)
		want := colorcore.PMColor4f{R: 1 - tc.t, B: tc.t, A: 1}
		colorNear(t, got, want, 1e-5, "two-stop lerp")
	}
	// Clamp beyond both ends (device coordinates are never negative — like the C pipeline's size_t dx — so probe the
	// low side with a gradient that starts to the right of the sample).
	low := NewLinearGradient(geom.Point{X: 60, Y: 0}, geom.Point{X: 160, Y: 0},
		[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, TileClamp, nil)
	colorNear(t, shadeAt(t, low, id, 5, 0), colorcore.PMColor4f{R: 1, A: 1}, 1e-6, "clamp low")
	colorNear(t, shadeAt(t, s, id, 200, 0), colorcore.PMColor4f{B: 1, A: 1}, 1e-6, "clamp high")
}

func TestLinearGradientEvenAndSearchedStops(t *testing.T) {
	id := geom.IdentityMatrix()
	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}

	// Evenly spaced three stops: t=0.25 sits mid red→green.
	even := NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, nil, TileClamp, nil)
	got := shadeAt(t, even, id, 24, 0) // t = 24.5/100
	colorNear(t, got, colorcore.PMColor4f{R: 1 - 0.49, G: 0.49, A: 1}, 1e-5, "even 3-stop")

	// Arbitrary positions: green pinned at 0.25.
	pos := NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors,
		[]float32{0, 0.25, 1}, TileClamp, nil)
	got = shadeAt(t, pos, id, 24, 0) // t=0.245 → just below the 0.25 stop: red→green at 0.98
	colorNear(t, got, colorcore.PMColor4f{R: 1 - 0.98, G: 0.98, A: 1}, 1e-5, "searched 3-stop below")
	got = shadeAt(t, pos, id, 62, 0) // t=0.625 → (0.625-0.25)/0.75 = 0.5 green→blue
	colorNear(t, got, colorcore.PMColor4f{G: 0.5, B: 0.5, A: 1}, 1e-5, "searched 3-stop above")

	// Hard stop at 0.5.
	hard := NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF},
		[]float32{0, 0.5, 0.5, 1}, TileClamp, nil)
	got = shadeAt(t, hard, id, 49, 0) // t=0.495: red→green at 0.99
	colorNear(t, got, colorcore.PMColor4f{R: 0.01, G: 0.99, A: 1}, 1e-5, "hard stop left")
	got = shadeAt(t, hard, id, 50, 0) // t=0.505: blue→white at 0.01
	colorNear(t, got, colorcore.PMColor4f{R: 0.01, G: 0.01, B: 1, A: 1}, 1e-5, "hard stop right")
}

func TestGradientStopFixing(t *testing.T) {
	// Implicit first/last stops get inserted and bracketed by [0, 1].
	s := NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, []float32{0.25, 0.75}, TileClamp, nil)
	g, ok := s.(*LinearGradient)
	if !ok {
		t.Fatalf("unexpected shader type %T", s)
	}
	if len(g.Colors()) != 4 {
		t.Fatalf("implicit stops not added: %d colors", len(g.Colors()))
	}
	wantPos := []float32{0, 0.25, 0.75, 1}
	for i, p := range g.Positions() {
		if p != wantPos[i] {
			t.Fatalf("positions %v, want %v", g.Positions(), wantPos)
		}
	}

	// Uniform positions collapse to implicit stops.
	s = NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.5, 1}, TileClamp, nil)
	if s.(*LinearGradient).Positions() != nil {
		t.Fatal("uniform stops should collapse to nil positions")
	}

	// Non-monotonic positions get pinned monotonic.
	s = NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.75, 0.25}, TileClamp, nil)
	pos := s.(*LinearGradient).Positions()
	for i := 1; i < len(pos); i++ {
		if pos[i] < pos[i-1] {
			t.Fatalf("positions not monotonic: %v", pos)
		}
	}

	// Triplicate interior stops dedupe to two.
	s = NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF888888, 0xFF0000FF, 0xFFFFFFFF},
		[]float32{0, 0.5, 0.5, 0.5, 1}, TileClamp, nil)
	pos = s.(*LinearGradient).Positions()
	count := 0
	for _, p := range pos {
		if p == 0.5 {
			count++
		}
	}
	if count != 2 {
		t.Fatalf("triplicate stop should dedupe to 2, got %d (%v)", count, pos)
	}
}

func TestTileModes(t *testing.T) {
	id := geom.IdentityMatrix()
	colors := []colorcore.Color{0xFFFF0000, 0xFF0000FF}
	p0, p1 := geom.Point{}, geom.Point{X: 10}

	repeat := NewLinearGradient(p0, p1, colors, nil, TileRepeat, nil)
	// t = 12.5/10 → frac 0.25
	colorNear(t, shadeAt(t, repeat, id, 12, 0),
		colorcore.PMColor4f{R: 0.75, B: 0.25, A: 1}, 1e-5, "repeat")

	mirror := NewLinearGradient(p0, p1, colors, nil, TileMirror, nil)
	// t = 12.5/10 = 1.25 → mirrored to 0.75
	colorNear(t, shadeAt(t, mirror, id, 12, 0),
		colorcore.PMColor4f{R: 0.25, B: 0.75, A: 1}, 1e-5, "mirror")

	decal := NewLinearGradient(p0, p1, colors, nil, TileDecal, nil)
	if !near(shadeAt(t, decal, id, 4, 0).A, 1, 1e-6) {
		t.Fatal("decal inside should be opaque")
	}
	colorNear(t, shadeAt(t, decal, id, 12, 0), colorcore.PMColor4f{}, 0, "decal outside")
	if decal.IsOpaque() {
		t.Fatal("decal gradients are never opaque")
	}
}

func TestGradientPremulAndPaintAlpha(t *testing.T) {
	// 50%-alpha white → premultiplied output.
	s := NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0x80FFFFFF, 0x80FFFFFF}, nil, TileClamp, nil)
	got := shadeAt(t, s, geom.IdentityMatrix(), 50, 0)
	a := float32(0x80) * (1.0 / 255.0)
	colorNear(t, got, colorcore.PMColor4f{R: a, G: a, B: a, A: a}, 1e-6, "premul stage")

	// Paint alpha scales the shader output (scale_1_float).
	p := Compile(s, geom.IdentityMatrix(), 0x80000000)
	var out [1]colorcore.PMColor4f
	p.ShadeSpan(50, 0, out[:])
	colorNear(t, out[0], colorcore.PMColor4f{R: a * a, G: a * a, B: a * a, A: a * a}, 1e-6,
		"paint alpha scale")
}

func TestLocalMatrixAndCTM(t *testing.T) {
	colors := []colorcore.Color{0xFFFF0000, 0xFF0000FF}
	plain := NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, nil, TileClamp, nil)

	// A local matrix passed to the factory matches wrapping manually.
	lm := geom.TranslateMatrix(20, 0)
	viaFactory := NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, nil, TileClamp, &lm)
	viaWrap := NewWithLocalMatrix(plain, lm)
	id := geom.IdentityMatrix()
	a := shadeAt(t, viaFactory, id, 40, 0)
	b := shadeAt(t, viaWrap, id, 40, 0)
	colorNear(t, a, b, 0, "factory vs wrap")
	// Translating by 20 shifts sampling: pixel 40 reads t=(40.5-20)/100.
	colorNear(t, a, shadeAt(t, plain, id, 20, 0), 0, "local translate")

	// The CTM composes the same way.
	ctm := geom.TranslateMatrix(20, 0)
	colorNear(t, shadeAt(t, plain, ctm, 40, 0), shadeAt(t, plain, id, 20, 0), 0, "ctm translate")

	// Nested wrappers combine as outer * inner.
	inner := NewWithLocalMatrix(plain, geom.ScaleMatrix(2, 2))
	nested := NewWithLocalMatrix(inner, geom.TranslateMatrix(10, 0))
	var combined geom.Matrix
	outerM := geom.TranslateMatrix(10, 0)
	innerM := geom.ScaleMatrix(2, 2)
	combined.SetConcat(&outerM, &innerM)
	direct := NewWithLocalMatrix(plain, combined)
	colorNear(t, shadeAt(t, nested, id, 33, 5), shadeAt(t, direct, id, 33, 5), 0, "nested LM")

	// A non-invertible CTM cannot draw.
	if Compile(plain, geom.Matrix{}, opaqueBlack) != nil {
		t.Fatal("singular CTM must fail Compile")
	}
}

// TestApplyForFragmentProcessorIgnoresCTM pins the fragment-processor matrix contract: the coordinates handed to the
// root shader's FP are already local-space, so ApplyForFragmentProcessor must invert only the pending local matrix
// (with postInv post-applied) and never the CTM. Folding the CTM into the inversion double-transforms every GPU
// gradient/image/noise shader the moment the canvas is translated or scaled — a drift the identity-CTM GPU swatch tests
// cannot see.
func TestApplyForFragmentProcessorIgnoresCTM(t *testing.T) {
	matrixNear := func(got, want geom.Matrix, what string) {
		t.Helper()
		for i := 0; i < 9; i++ {
			if !near(got.Get(i), want.Get(i), 1e-6) {
				t.Fatalf("%s: matrix[%d] = %g, want %g", what, i, got.Get(i), want.Get(i))
			}
		}
	}
	var ctm geom.Matrix
	ctm.SetScaleTranslate(2, 0.5, 7, -3)

	// With no pending local matrix the result is exactly postInv, regardless of the CTM.
	post := geom.TranslateMatrix(3, 4)
	rec := NewMatrixRec(ctm)
	got, ok := rec.ApplyForFragmentProcessor(&post)
	if !ok {
		t.Fatal("ApplyForFragmentProcessor failed with identity pending local matrix")
	}
	matrixNear(got, post, "identity pending local")

	// A pending local matrix is inverted and post-applied: postInv * lm^-1, still CTM-free. The expected inverse is
	// written out literally so this does not just re-run Invert.
	var lm, lmInv, want geom.Matrix
	lm.SetScaleTranslate(2, 4, 10, -8)
	lmInv.SetScaleTranslate(0.5, 0.25, -5, 2)
	want.SetConcat(&post, &lmInv)
	rec = NewMatrixRec(ctm)
	concatenated := rec.Concat(lm)
	got, ok = concatenated.ApplyForFragmentProcessor(&post)
	if !ok {
		t.Fatal("ApplyForFragmentProcessor failed with invertible pending local matrix")
	}
	matrixNear(got, want, "pending local inverted")

	// A singular pending local matrix cannot draw, even though the CTM is invertible.
	rec = NewMatrixRec(ctm)
	singular := rec.Concat(geom.Matrix{})
	if _, ok = singular.ApplyForFragmentProcessor(&post); ok {
		t.Fatal("singular pending local matrix must fail")
	}
}

func TestSweepGradient(t *testing.T) {
	colors := []colorcore.Color{0xFFFF0000, 0xFF0000FF}
	center := geom.Point{X: 20.5, Y: 10.5}
	s := NewSweepGradient(center, colors, nil, TileClamp, 0, 360, nil)
	id := geom.IdentityMatrix()
	// +x axis → t=0; +y (down) → t=0.25; -x → t=0.5.
	colorNear(t, shadeAt(t, s, id, 30, 10), colorcore.PMColor4f{R: 1, A: 1}, 1e-4, "sweep +x")
	colorNear(t, shadeAt(t, s, id, 20, 20),
		colorcore.PMColor4f{R: 0.75, B: 0.25, A: 1}, 1e-4, "sweep +y")
	colorNear(t, shadeAt(t, s, id, 10, 10),
		colorcore.PMColor4f{R: 0.5, B: 0.5, A: 1}, 1e-4, "sweep -x")

	// Partial sweeps remap the angle range.
	part := NewSweepGradient(center, colors, nil, TileClamp, 0, 180, nil)
	colorNear(t, shadeAt(t, part, id, 20, 20),
		colorcore.PMColor4f{R: 0.5, B: 0.5, A: 1}, 1e-4, "sweep 0..180 at +y")

	// Invalid angles.
	if NewSweepGradient(geom.Point{}, colors, nil, TileClamp, 90, 10, nil) != nil {
		t.Fatal("start > end must fail")
	}
}

func TestRadialGradient(t *testing.T) {
	colors := []colorcore.Color{0xFFFF0000, 0xFF0000FF}
	s := NewRadialGradient(geom.Point{X: 0.5, Y: 0.5}, 20, colors, nil, TileClamp, nil)
	id := geom.IdentityMatrix()
	colorNear(t, shadeAt(t, s, id, 0, 0), colorcore.PMColor4f{R: 1, A: 1}, 1e-5, "radial center")
	// Pixel (10,0): distance 10 → t=0.5.
	colorNear(t, shadeAt(t, s, id, 10, 0),
		colorcore.PMColor4f{R: 0.5, B: 0.5, A: 1}, 1e-5, "radial mid")
	colorNear(t, shadeAt(t, s, id, 40, 0), colorcore.PMColor4f{B: 1, A: 1}, 1e-5, "radial clamp")
}

func TestTwoPointConicalGradient(t *testing.T) {
	colors := []colorcore.Color{0xFFFF0000, 0xFF0000FF}
	id := geom.IdentityMatrix()

	// Concentric (radial variant): t = d/(r1-r0) - r0/(r1-r0) = d/10 - 1.
	con := NewTwoPointConicalGradient(geom.Point{X: 0.5, Y: 0.5}, 10,
		geom.Point{X: 0.5, Y: 0.5}, 20, colors, nil, TileClamp, nil)
	colorNear(t, shadeAt(t, con, id, 15, 0),
		colorcore.PMColor4f{R: 0.5, B: 0.5, A: 1}, 1e-5, "concentric mid")
	colorNear(t, shadeAt(t, con, id, 5, 0), colorcore.PMColor4f{R: 1, A: 1}, 1e-5, "concentric inner")

	// Focal (r0 = 0): the gradient parameter is the maximum t whose circle (center lerped to c1, radius 10t) passes
	// through the point: |d - 30t| = 10t → t = max(d/40, d/20).
	foc := NewTwoPointConicalGradient(geom.Point{X: 0.5, Y: 0.5}, 0,
		geom.Point{X: 30.5, Y: 0.5}, 10, colors, nil, TileClamp, nil)
	colorNear(t, shadeAt(t, foc, id, 10, 0),
		colorcore.PMColor4f{R: 0.5, B: 0.5, A: 1}, 1e-4, "focal mid") // d=10 → t=0.5
	colorNear(t, shadeAt(t, foc, id, 20, 0), colorcore.PMColor4f{B: 1, A: 1}, 1e-4, "focal t=1")
	colorNear(t, shadeAt(t, foc, id, 40, 0), colorcore.PMColor4f{B: 1, A: 1}, 1e-4, "focal clamp")

	// Degenerate: same centers and radii.
	deg := NewTwoPointConicalGradient(geom.Point{X: 5, Y: 5}, 10, geom.Point{X: 5, Y: 5}, 10,
		colors, nil, TileRepeat, nil)
	if !deg.IsConstant() {
		t.Fatal("repeat degenerate should collapse to the average color")
	}
}

func TestDegenerateGradients(t *testing.T) {
	colors := []colorcore.Color{0xFFFF0000, 0xFF0000FF}
	p := geom.Point{X: 5, Y: 5}

	clamp := NewLinearGradient(p, p, colors, nil, TileClamp, nil)
	pc := Compile(clamp, geom.IdentityMatrix(), opaqueBlack)
	colorNear(t, pc.EvalConstant(), colorcore.PMColor4f{B: 1, A: 1}, 1e-6, "degenerate clamp = last")

	repeat := NewLinearGradient(p, p, colors, nil, TileRepeat, nil)
	pr := Compile(repeat, geom.IdentityMatrix(), opaqueBlack)
	colorNear(t, pr.EvalConstant(), colorcore.PMColor4f{R: 0.5, B: 0.5, A: 1}, 1e-6,
		"degenerate repeat = average")

	decal := NewLinearGradient(p, p, colors, nil, TileDecal, nil)
	if Compile(decal, geom.IdentityMatrix(), opaqueBlack) != nil {
		t.Fatal("degenerate decal is the empty shader")
	}

	// Single color collapses to a color shader.
	one := NewLinearGradient(geom.Point{}, geom.Point{X: 10}, colors[:1], nil, TileClamp, nil)
	if _, ok := one.(*ColorShader); !ok {
		t.Fatalf("single-stop gradient should be a color shader, got %T", one)
	}
}

// TestGradientMismatchedPositions checks that every public gradient constructor rejects (returns nil rather than
// panicking) a position slice whose length does not match the color count. A short slice would otherwise index past the
// end of positions during stop preprocessing (and in the degenerate-geometry average path).
func TestGradientMismatchedPositions(t *testing.T) {
	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}
	shortPos := []float32{0, 1} // one offset short of the three colors
	longPos := []float32{0, 0.5, 0.75, 1}
	p0, p1 := geom.Point{}, geom.Point{X: 100}
	center := geom.Point{X: 5, Y: 5}

	for _, pos := range [][]float32{shortPos, longPos} {
		if s := NewLinearGradient(p0, p1, colors, pos, TileClamp, nil); s != nil {
			t.Fatalf("linear gradient with %d positions for %d colors should be nil", len(pos), len(colors))
		}
		if s := NewRadialGradient(center, 20, colors, pos, TileClamp, nil); s != nil {
			t.Fatalf("radial gradient with %d positions for %d colors should be nil", len(pos), len(colors))
		}
		if s := NewSweepGradient(center, colors, pos, TileClamp, 0, 360, nil); s != nil {
			t.Fatalf("sweep gradient with %d positions for %d colors should be nil", len(pos), len(colors))
		}
		if s := NewTwoPointConicalGradient(center, 10, geom.Point{X: 50, Y: 5}, 20,
			colors, pos, TileClamp, nil); s != nil {
			t.Fatalf("conical gradient with %d positions for %d colors should be nil", len(pos), len(colors))
		}
		// Degenerate geometry (coincident endpoints/radii) reaches the average-color path, which also indexes
		// positions per color.
		if s := NewLinearGradient(center, center, colors, pos, TileRepeat, nil); s != nil {
			t.Fatalf("degenerate linear gradient with %d positions for %d colors should be nil", len(pos), len(colors))
		}
	}

	// A matched-length position slice still succeeds, confirming the guard is not over-broad.
	if s := NewLinearGradient(p0, p1, colors, []float32{0, 0.5, 1}, TileClamp, nil); s == nil {
		t.Fatal("linear gradient with matching positions should succeed")
	}
}

func TestBlendShader(t *testing.T) {
	red := NewColor(0xFFFF0000)
	green := NewColor(0xFF00FF00)
	id := geom.IdentityMatrix()

	// Plus: red + green = yellow.
	plus := NewBlend(raster.BlendPlus, red, green)
	colorNear(t, shadeAt(t, plus, id, 0, 0),
		colorcore.PMColor4f{R: 1, G: 1, A: 1}, 1e-6, "plus")

	// Collapse rules.
	if NewBlend(raster.BlendSrc, red, green) != green {
		t.Fatal("kSrc must collapse to src")
	}
	if NewBlend(raster.BlendDst, red, green) != red {
		t.Fatal("kDst must collapse to dst")
	}
	if NewBlend(raster.BlendPlus, nil, green) != nil {
		t.Fatal("nil child must yield nil")
	}

	// A gradient child re-seeds device coordinates after the first child ran.
	grad := NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, TileClamp, nil)
	over := NewBlend(raster.BlendSrcOver, grad, NewColor(0x800000FF))
	got := shadeAt(t, over, id, 50, 0)
	// srcover of half-blue over gradient(t=0.505): d=(0.495,0,0.505), s=(0,0,0.502,0.502).
	sa := float32(0x80) * (1.0 / 255.0)
	want := colorcore.PMColor4f{
		R: 0.495 * (1 - sa),
		B: sa + 0.505*(1-sa),
		A: 1,
	}
	colorNear(t, got, want, 1e-4, "srcover blend shader over gradient")
}

func TestDitherStage(t *testing.T) {
	// Dither adds rate-scaled ±[0.5) offsets in a deterministic 8x8 pattern, clamped to [0, a].
	s := NewLinearGradient(geom.Point{}, geom.Point{X: 256},
		[]colorcore.Color{0xFF000000, 0xFFFFFFFF}, nil, TileClamp, nil)
	base := Compile(s, geom.IdentityMatrix(), opaqueBlack)
	dithered := Compile(s, geom.IdentityMatrix(), opaqueBlack)
	dithered.AppendDither(1.0 / 255.0)

	var plain, dith [16]colorcore.PMColor4f
	base.ShadeSpan(3, 5, plain[:])
	dithered.ShadeSpan(3, 5, dith[:])
	for i := range plain {
		x := uint32(3 + i)
		y := uint32(5) ^ x
		m := (y&1)<<5 | (x&1)<<4 | (y&2)<<2 | (x&2)<<1 | (y&4)>>1 | (x&4)>>2
		off := (float32(m)*(2.0/128.0) - 63.0/128.0) * (1.0 / 255.0)
		want := plain[i].R + off
		if want < 0 {
			want = 0
		}
		if want > plain[i].A {
			want = plain[i].A
		}
		if !near(dith[i].R, want, 1e-6) {
			t.Fatalf("dither pixel %d: got %g, want %g", i, dith[i].R, want)
		}
	}
}

// matrixNear reports whether every element of a and b matches within tol.
func matrixNear(a, b *geom.Matrix, tol float32) bool {
	for i := range 9 {
		if !near(a.Get(i), b.Get(i), tol) {
			return false
		}
	}
	return true
}

// TestMatrixRecTotalInverseSurvivesApply pins the total matrix contract: totalInverse is the inverse of ctm * every
// local matrix seen so far, including the ones apply already folded into the pipeline. A wrapper that applies the CTM
// and then runs a child (FilterDecalShader, the displacement map, the lighting normal map) used to leave the child with
// an identity total, which silently misreported the device transform.
func TestMatrixRecTotalInverseSurvivesApply(t *testing.T) {
	ctm := geom.ScaleMatrix(2, 4)
	lm := geom.TranslateMatrix(3, 5)
	rec := newMatrixRec(ctm)
	rec = rec.concat(&lm)

	var total geom.Matrix
	total.SetConcat(&ctm, &lm)
	want, ok := total.Invert()
	if !ok {
		t.Fatal("fixture matrix is not invertible")
	}

	got, ok := rec.totalInverse()
	if !ok {
		t.Fatal("totalInverse before apply returned ok=false")
	}
	if !matrixNear(&got, &want, 1e-6) {
		t.Errorf("totalInverse before apply = %v, want %v", got.As9(), want.As9())
	}

	p := borrowPipeline()
	defer RecyclePipeline(p)
	identity := geom.IdentityMatrix()
	applied, ok := rec.apply(p, &identity)
	if !ok {
		t.Fatal("apply returned ok=false")
	}
	if !applied.pendingLocal.IsIdentity() {
		t.Error("apply left a pending local matrix behind")
	}
	got, ok = applied.totalInverse()
	if !ok {
		t.Fatal("totalInverse after apply returned ok=false")
	}
	if !matrixNear(&got, &want, 1e-6) {
		t.Errorf("totalInverse after apply = %v, want %v", got.As9(), want.As9())
	}

	// A local matrix concatenated after the apply accumulates on top of the already-applied ones.
	lm2 := geom.ScaleMatrix(0.5, 0.5)
	nested := applied.concat(&lm2)
	var total2 geom.Matrix
	total2.SetConcat(&total, &lm2)
	want2, ok := total2.Invert()
	if !ok {
		t.Fatal("nested fixture matrix is not invertible")
	}
	got, ok = nested.totalInverse()
	if !ok {
		t.Fatal("nested totalInverse returned ok=false")
	}
	if !matrixNear(&got, &want2, 1e-6) {
		t.Errorf("nested totalInverse = %v, want %v", got.As9(), want2.As9())
	}
}

// TestMatrixRecTotalInverseRejectsSingular keeps the not-invertible report on the total matrix rather than on whatever
// is still pending.
func TestMatrixRecTotalInverseRejectsSingular(t *testing.T) {
	rec := newMatrixRec(geom.ScaleMatrix(0, 1))
	if _, ok := rec.totalInverse(); ok {
		t.Error("totalInverse on a singular ctm returned ok=true")
	}
	lm := geom.ScaleMatrix(0, 1)
	rec = newMatrixRec(geom.IdentityMatrix())
	rec = rec.concat(&lm)
	if _, ok := rec.totalInverse(); ok {
		t.Error("totalInverse on a singular local matrix returned ok=true")
	}
}
