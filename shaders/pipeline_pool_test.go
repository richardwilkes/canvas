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
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
)

// TestGradientContextReuseIsClean locks the pooled-Pipeline invariant: a gradient fill stage compiled into a Pipeline
// whose per-gradient context storage was left oversized and filled with stale data (as it is after a prior, larger
// gradient recycled the pipeline) must produce byte-identical output to a fresh compile. The mechanism under test is
// nextGradCtx + ensureF32 (reslice-and-zero of the retained factor/bias/ts storage).
func TestGradientContextReuseIsClean(t *testing.T) {
	id := geom.IdentityMatrix()
	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}
	for _, tc := range []struct {
		name string
		pos  []float32 // nil → evenly-spaced fill lane; non-nil → arbitrary-stops fill lane (uses ts)
		tile TileMode
	}{
		{name: "evenly-spaced", pos: nil, tile: TileClamp},
		{name: "arbitrary-stops", pos: []float32{0, 0.3, 1}, tile: TileRepeat},
	} {
		grad := NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, tc.pos, tc.tile, nil)

		// Fresh reference output (a clean, never-reused pipeline).
		var want [16]colorcore.PMColor4f
		Compile(grad, id, opaqueBlack).ShadeSpan(0, 3, want[:])

		// Poison a pipeline's gradient-context storage with oversized, stale data and reset its counter exactly as
		// RecyclePipeline leaves it, then compile the same gradient into it.
		p := &Pipeline{}
		ctx := &gradientCtx{}
		poisoned := gradStop{fr: 9, fg: 9, fb: 9, fa: 9, br: 9, bg: 9, bb: 9, ba: 9}
		ctx.stops = []gradStop{poisoned, poisoned, poisoned, poisoned, poisoned, poisoned, poisoned, poisoned}
		ctx.ts = []float32{9, 9, 9, 9, 9, 9, 9, 9}
		p.gradCtxs = []*gradientCtx{ctx}
		p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
		if !grad.appendStages(p, newMatrixRec(id)) {
			t.Fatalf("%s: appendStages failed", tc.name)
		}
		var got [16]colorcore.PMColor4f
		p.ShadeSpan(0, 3, got[:])

		for i := range got {
			colorNear(t, got[i], want[i], 0, tc.name+" reused-context pixel")
		}
		// The reused ctx must have been handed out (not a freshly-appended one).
		if p.gradCtxN != 1 || p.gradCtxs[0] != ctx {
			t.Fatalf("%s: expected the poisoned ctx to be reused (n=%d)", tc.name, p.gradCtxN)
		}
	}
}

// TestBlendTwoGradientsUseDistinctContexts locks that a single pipeline containing two gradient fill stages (a
// BlendShader inlines both children) hands each stage its own gradientCtx — a shared one would let the second compile
// clobber the first's factors/biases at ShadeSpan time.
func TestBlendTwoGradientsUseDistinctContexts(t *testing.T) {
	id := geom.IdentityMatrix()
	// Both children are 3-stop gradients. Every fill lane now draws its context from nextGradCtx (the 2-stop fast lane
	// included), so two gradient fill stages in one pipeline must receive distinct ones.
	g0 := NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.3, 1}, TileClamp, nil)
	g1 := NewRadialGradient(geom.Point{X: 50, Y: 20}, 40,
		[]colorcore.Color{0xFF102030, 0xFF405060, 0xFFA0B0C0}, nil, TileMirror, nil)
	blend := NewBlend(raster.BlendSrcOver, g0, g1)

	p := Compile(blend, id, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.gradCtxN != 2 {
		t.Fatalf("expected 2 gradient fill contexts, got %d", p.gradCtxN)
	}
	if p.gradCtxs[0] == p.gradCtxs[1] {
		t.Fatal("the two gradient fill stages share a context (would clobber)")
	}
}

// TestRecyclePipelineResetsState locks that RecyclePipeline leaves a Pipeline in the reset state Compile relies on: no
// stages, no handed-out gradient contexts, cleared paint color.
func TestRecyclePipelineResetsState(t *testing.T) {
	id := geom.IdentityMatrix()
	grad := NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.3, 1}, TileClamp, nil)
	p := Compile(grad, id, 0x80112233)
	var junk [16]colorcore.PMColor4f
	p.ShadeSpan(0, 0, junk[:])
	if len(p.stages) == 0 || p.gradCtxN == 0 {
		t.Fatal("expected a compiled gradient to have stages and a handed-out context")
	}
	RecyclePipeline(p)
	if len(p.stages) != 0 {
		t.Fatalf("recycled pipeline retains %d stages", len(p.stages))
	}
	if len(p.postStages) != 0 {
		t.Fatalf("recycled pipeline retains %d post stages", len(p.postStages))
	}
	if p.gradCtxN != 0 {
		t.Fatalf("recycled pipeline retains gradCtxN=%d", p.gradCtxN)
	}
	if p.matrixCtxN != 0 {
		t.Fatalf("recycled pipeline retains matrixCtxN=%d", p.matrixCtxN)
	}
	if (p.paintColor != colorcore.Color4f{}) {
		t.Fatalf("recycled pipeline retains paint color %+v", p.paintColor)
	}
	// The retained context storage stays for reuse (not dropped).
	if len(p.gradCtxs) == 0 {
		t.Fatal("recycled pipeline dropped its gradient-context storage")
	}
	if len(p.matrixCtxs) == 0 {
		t.Fatal("recycled pipeline dropped its matrix-context storage")
	}
}

// TestMatrixContextReuseIsClean locks the pooled-Pipeline invariant for matrix stages: a matrix stage compiled into a
// Pipeline whose matrixCtxs storage was left filled with stale coefficients (as it is after a prior compile recycled
// the pipeline) must produce byte-identical output to a fresh compile, and must reuse the retained matrixCtx rather
// than appending a new one. The mechanism under test is nextMatrixCtx (reload of the retained coefficient array).
func TestMatrixContextReuseIsClean(t *testing.T) {
	// A rotated CTM forces the general affine matrix stage (the one that reads all six 2x3 slots), so a stale
	// coefficient in any slot would surface as a pixel difference.
	var ctm geom.Matrix
	ctm.SetRotatePivot(31, 128, 128)
	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}
	grad := NewLinearGradient(geom.Point{X: 8, Y: 8}, geom.Point{X: 248, Y: 248}, colors,
		[]float32{0, 0.3, 1}, TileRepeat, nil)

	var want [16]colorcore.PMColor4f
	Compile(grad, ctm, opaqueBlack).ShadeSpan(0, 5, want[:])

	// Poison a pipeline's matrix-context storage with stale coefficients and reset its counter exactly as
	// RecyclePipeline leaves it, then compile the same shader into it.
	p := &Pipeline{}
	poison := &matrixCtx{m: [9]float32{9, 9, 9, 9, 9, 9, 9, 9, 9}}
	p.matrixCtxs = []*matrixCtx{poison}
	p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !grad.appendStages(p, newMatrixRec(ctm)) {
		t.Fatal("appendStages failed")
	}
	var got [16]colorcore.PMColor4f
	p.ShadeSpan(0, 5, got[:])

	for i := range got {
		colorNear(t, got[i], want[i], 0, "reused-matrix-context pixel")
	}
	if p.matrixCtxN != 1 || p.matrixCtxs[0] != poison {
		t.Fatalf("expected the poisoned matrix ctx to be reused (n=%d)", p.matrixCtxN)
	}
}

// TestBlendTwoGradientsUseDistinctMatrixContexts locks that two gradient children in one pipeline each receive their
// own matrixCtx — a shared one would let the second child's ptsToUnit stage clobber the first's coefficients at
// ShadeSpan time.
func TestBlendTwoGradientsUseDistinctMatrixContexts(t *testing.T) {
	id := geom.IdentityMatrix()
	g0 := NewLinearGradient(geom.Point{}, geom.Point{X: 100},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.3, 1}, TileClamp, nil)
	g1 := NewRadialGradient(geom.Point{X: 50, Y: 20}, 40,
		[]colorcore.Color{0xFF102030, 0xFF405060, 0xFFA0B0C0}, nil, TileMirror, nil)
	p := Compile(NewBlend(raster.BlendSrcOver, g0, g1), id, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.matrixCtxN != 2 {
		t.Fatalf("expected 2 matrix contexts (one per gradient ptsToUnit), got %d", p.matrixCtxN)
	}
	if p.matrixCtxs[0] == p.matrixCtxs[1] {
		t.Fatal("the two matrix stages share a context (would clobber)")
	}
}

// TestLaneMaskReuseIsClean locks the pooled-Pipeline invariant for the decal/conical masking stages: a masked gradient
// compiled into a Pipeline whose laneMask storage was left filled with stale bits (as it is after a prior compile
// recycled the pipeline) must produce byte-identical output to a fresh compile, and must reuse the retained mask rather
// than allocating a new one. Each ShadeSpan chunk writes the mask in full before its consumer reads, so the stale bits
// must never surface.
func TestLaneMaskReuseIsClean(t *testing.T) {
	id := geom.IdentityMatrix()
	// A short decal gradient shaded across 16 lanes: the early lanes fall inside [0,1) (decal_x keeps them), the later
	// lanes fall outside (decal_x masks them to transparent), so a leaked stale mask bit would surface as a wrong
	// pixel.
	grad := NewLinearGradient(geom.Point{}, geom.Point{X: 8},
		[]colorcore.Color{0xFFFF0000, 0xFF0000FF}, nil, TileDecal, nil)

	var want [16]colorcore.PMColor4f
	Compile(grad, id, opaqueBlack).ShadeSpan(0, 0, want[:])

	p := &Pipeline{}
	poison := &laneMask{}
	for i := range poison {
		poison[i] = 0x5A5A5A5A
	}
	p.laneMasks = []*laneMask{poison}
	p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !grad.appendStages(p, newMatrixRec(id)) {
		t.Fatal("appendStages failed")
	}
	var got [16]colorcore.PMColor4f
	p.ShadeSpan(0, 0, got[:])

	for i := range got {
		colorNear(t, got[i], want[i], 0, "reused-lane-mask pixel")
	}
	if p.laneMaskN != 1 || p.laneMasks[0] != poison {
		t.Fatalf("expected the poisoned lane mask to be reused (n=%d)", p.laneMaskN)
	}
}

// TestBlendTwoBlendsUseDistinctBlendContexts locks that two blend stages in one pipeline (a nested blend shader) each
// receive their own blendCtx — a shared one would let the second AppendBlend clobber the first's mode at ShadeSpan
// time. The inner blend's mode is appended first.
func TestBlendTwoBlendsUseDistinctBlendContexts(t *testing.T) {
	id := geom.IdentityMatrix()
	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}
	g0 := NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, nil, TileClamp, nil)
	g1 := NewLinearGradient(geom.Point{Y: 100}, geom.Point{}, colors, nil, TileClamp, nil)
	g2 := NewRadialGradient(geom.Point{X: 50, Y: 50}, 40, colors, nil, TileMirror, nil)
	outer := NewBlend(raster.BlendScreen, NewBlend(raster.BlendMultiply, g0, g1), g2)

	p := Compile(outer, id, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.blendCtxN != 2 {
		t.Fatalf("expected 2 blend contexts, got %d", p.blendCtxN)
	}
	if p.blendCtxs[0] == p.blendCtxs[1] {
		t.Fatal("the two blend stages share a context (would clobber)")
	}
	if p.blendCtxs[0].mode != raster.BlendMultiply || p.blendCtxs[1].mode != raster.BlendScreen {
		t.Fatalf("blend contexts hold the wrong modes: [0]=%d [1]=%d", p.blendCtxs[0].mode, p.blendCtxs[1].mode)
	}
}

// TestBlendShaderContextReuseIsClean locks the pooled-Pipeline invariant for the BlendShader store/load stages: a blend
// compiled into a Pipeline whose blendShaderCtx storage was left filled with stale coords/res0 data (as it is after a
// prior compile recycled the pipeline) must produce byte-identical output to a fresh compile, and must reuse the
// retained ctx rather than allocating a new one. Each ShadeSpan chunk writes res0 (store_src) before it reads it
// (load_dst), so the stale bits must never surface.
func TestBlendShaderContextReuseIsClean(t *testing.T) {
	id := geom.IdentityMatrix()
	// A gradient×gradient blend whose output varies across the 16 lanes, so a leaked stale res0 value (the dst child's
	// saved output the blend loads into the dst registers) would surface as a wrong pixel.
	g0 := NewLinearGradient(geom.Point{}, geom.Point{X: 12},
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.3, 1}, TileClamp, nil)
	g1 := NewRadialGradient(geom.Point{X: 8, Y: 4}, 20,
		[]colorcore.Color{0xFF102030, 0xFF405060, 0xFFA0B0C0}, nil, TileMirror, nil)
	blend := NewBlend(raster.BlendSrcOver, g0, g1)

	var want [16]colorcore.PMColor4f
	Compile(blend, id, opaqueBlack).ShadeSpan(0, 3, want[:])

	// Poison a pipeline's blend-shader-context storage with stale coords/res0 and reset its counter exactly as
	// RecyclePipeline leaves it, then compile the same blend into it.
	p := &Pipeline{}
	poison := &blendShaderCtx{}
	for i := range stride {
		poison.coords[0][i], poison.coords[1][i] = 9, 9
		poison.res0[0][i], poison.res0[1][i], poison.res0[2][i], poison.res0[3][i] = 9, 9, 9, 9
	}
	p.blendShaderCtxs = []*blendShaderCtx{poison}
	p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !blend.(*BlendShader).appendStages(p, newMatrixRec(id)) {
		t.Fatal("appendStages failed")
	}
	var got [16]colorcore.PMColor4f
	p.ShadeSpan(0, 3, got[:])

	for i := range got {
		colorNear(t, got[i], want[i], 0, "reused-blend-shader-context pixel")
	}
	if p.blendShaderCtxN != 1 || p.blendShaderCtxs[0] != poison {
		t.Fatalf("expected the poisoned blend-shader ctx to be reused (n=%d)", p.blendShaderCtxN)
	}
}

// TestNestedBlendUseDistinctBlendShaderContexts locks that two blend shaders in one pipeline (a blend whose dst child
// is itself a blend) each receive their own blendShaderCtx — a shared one would let the inner blend's store_src clobber
// the outer's saved dst output before the outer's load_dst reads it. The outer blend acquires its ctx first (at the top
// of appendStages, before recursing), the inner second.
func TestNestedBlendUseDistinctBlendShaderContexts(t *testing.T) {
	id := geom.IdentityMatrix()
	colors := []colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}
	g0 := NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, nil, TileClamp, nil)
	g1 := NewLinearGradient(geom.Point{Y: 100}, geom.Point{}, colors, nil, TileClamp, nil)
	g2 := NewRadialGradient(geom.Point{X: 50, Y: 50}, 40, colors, nil, TileMirror, nil)
	outer := NewBlend(raster.BlendScreen, NewBlend(raster.BlendMultiply, g0, g1), g2)

	p := Compile(outer, id, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.blendShaderCtxN != 2 {
		t.Fatalf("expected 2 blend-shader contexts, got %d", p.blendShaderCtxN)
	}
	if p.blendShaderCtxs[0] == p.blendShaderCtxs[1] {
		t.Fatal("the two blend shaders share a store/load context (would clobber)")
	}
}

// TestColorFuncTwoStagesUseDistinctContexts locks that two AppendColorFunc stages in one pipeline (as a compose color
// filter appends) each receive their own colorFuncCtx and run in order — a shared context would run the second
// transform twice. The values are dyadic so the composed result is exact.
func TestColorFuncTwoStagesUseDistinctContexts(t *testing.T) {
	p := &Pipeline{}
	p.AppendConstantColor(colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.75, A: 1})
	p.AppendColorFunc(func(r, g, b, a float32) (float32, float32, float32, float32) {
		return r + 0.125, g, b, a // stage A: bump R
	})
	p.AppendColorFunc(func(r, g, b, a float32) (float32, float32, float32, float32) {
		return r, g + 0.25, b, a // stage B: bump G
	})
	if p.colorFuncCtxN != 2 {
		t.Fatalf("expected 2 color-func contexts, got %d", p.colorFuncCtxN)
	}
	if p.colorFuncCtxs[0] == p.colorFuncCtxs[1] {
		t.Fatal("the two color-func stages share a context")
	}
	var out [1]colorcore.PMColor4f
	p.ShadeSpan(0, 0, out[:])
	colorNear(t, out[0], colorcore.PMColor4f{R: 0.375, G: 0.75, B: 0.75, A: 1}, 0, "two-color-func output")
}

// TestRecyclePipelineResetsNewCounters locks that RecyclePipeline resets the laneMask/blend/colorFunc/ image-sampling
// free-list counters (so a reused pipeline hands out fresh contexts) while retaining their storage.
func TestRecyclePipelineResetsNewCounters(t *testing.T) {
	p := borrowPipeline()
	p.nextLaneMask()
	p.nextBlendCtx()
	p.nextColorFuncCtx()
	p.nextBlendShaderCtx()
	p.nextGatherCtx()
	p.nextSamplerCtx()
	p.nextDecalCtx()
	p.nextMorphologyCtx()
	p.nextDisplacementCtx()
	p.nextNormalCtx()
	p.nextLightingCtx()
	p.nextMatrixConvCtx()
	p.nextArithmeticCtx()
	if p.laneMaskN == 0 || p.blendCtxN == 0 || p.colorFuncCtxN == 0 || p.blendShaderCtxN == 0 ||
		p.gatherCtxN == 0 || p.samplerCtxN == 0 || p.decalCtxN == 0 || p.morphCtxN == 0 || p.dispCtxN == 0 ||
		p.normalCtxN == 0 || p.lightingCtxN == 0 || p.matConvCtxN == 0 || p.arithCtxN == 0 {
		t.Fatal("expected the new free-list counters to be bumped")
	}
	RecyclePipeline(p)
	if p.laneMaskN != 0 || p.blendCtxN != 0 || p.colorFuncCtxN != 0 || p.blendShaderCtxN != 0 ||
		p.gatherCtxN != 0 || p.samplerCtxN != 0 || p.decalCtxN != 0 || p.morphCtxN != 0 || p.dispCtxN != 0 ||
		p.normalCtxN != 0 || p.lightingCtxN != 0 || p.matConvCtxN != 0 || p.arithCtxN != 0 {
		t.Fatalf("recycle left counters laneMaskN=%d blendCtxN=%d colorFuncCtxN=%d blendShaderCtxN=%d "+
			"gatherCtxN=%d samplerCtxN=%d decalCtxN=%d morphCtxN=%d dispCtxN=%d normalCtxN=%d lightingCtxN=%d "+
			"matConvCtxN=%d arithCtxN=%d", p.laneMaskN, p.blendCtxN, p.colorFuncCtxN, p.blendShaderCtxN,
			p.gatherCtxN, p.samplerCtxN, p.decalCtxN, p.morphCtxN, p.dispCtxN, p.normalCtxN, p.lightingCtxN,
			p.matConvCtxN, p.arithCtxN)
	}
	if len(p.laneMasks) == 0 || len(p.blendCtxs) == 0 || len(p.colorFuncCtxs) == 0 || len(p.blendShaderCtxs) == 0 ||
		len(p.gatherCtxs) == 0 || len(p.samplerCtxs) == 0 || len(p.decalCtxs) == 0 || len(p.morphCtxs) == 0 ||
		len(p.dispCtxs) == 0 || len(p.normalCtxs) == 0 || len(p.lightingCtxs) == 0 || len(p.matConvCtxs) == 0 ||
		len(p.arithCtxs) == 0 {
		t.Fatal("recycle dropped retained free-list storage")
	}
}

// TestRecyclePipelineClearsRegisterFileCtx locks that RecyclePipeline drops the register file's stage-context pointer.
// ShadeSpan leaves z.ctx aimed at the last-executed stage's context, which for a stage parameterized with per-draw
// external data — here a *PerlinNoiseShader and its painting-data tables — would keep that data alive for as long as the
// pipeline sits idle in the pool, contrary to the documented no-pinning discipline.
func TestRecyclePipelineClearsRegisterFileCtx(t *testing.T) {
	sh := NewTurbulence(0.1, 0.1, 2, 3, 0, 0)
	p := Compile(sh, geom.IdentityMatrix(), opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	var junk [16]colorcore.PMColor4f
	p.ShadeSpan(0, 0, junk[:])
	if _, ok := p.z.ctx.(*PerlinNoiseShader); !ok {
		t.Fatalf("expected the noise shader to be left in the register file's ctx, got %T", p.z.ctx)
	}
	RecyclePipeline(p)
	if p.z.ctx != nil {
		t.Fatalf("recycled pipeline still pins the last stage's context (%T)", p.z.ctx)
	}
}

// TestRecyclePipelineClearsColorFuncs locks that RecyclePipeline drops the retained colorFuncCtx transforms. A
// runtime-effect color filter's fn is a closure over that filter's per-draw uniforms, so a pooled pipeline that kept it
// would pin them; the ctx storage itself must survive for reuse.
func TestRecyclePipelineClearsColorFuncs(t *testing.T) {
	p := borrowPipeline()
	p.AppendConstantColor(colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.75, A: 1})
	uniforms := &[4]float32{0.125, 0, 0, 0} // stand-in for a runtime effect's uniform block
	p.AppendColorFunc(func(r, g, b, a float32) (float32, float32, float32, float32) {
		return r + uniforms[0], g, b, a
	})
	if p.colorFuncCtxN != 1 || p.colorFuncCtxs[0].fn == nil {
		t.Fatal("expected one color-func context holding a transform")
	}
	ctx := p.colorFuncCtxs[0]
	RecyclePipeline(p)
	if len(p.colorFuncCtxs) == 0 || p.colorFuncCtxs[0] != ctx {
		t.Fatal("recycle dropped the retained color-func context storage")
	}
	for i, c := range p.colorFuncCtxs {
		if c.fn != nil {
			t.Fatalf("recycled pipeline still pins the transform on colorFuncCtxs[%d]", i)
		}
	}

	// A cleared (or stale) retained context must still be usable: the next AppendColorFunc reuses it and overwrites fn.
	stale := &colorFuncCtx{}
	q := &Pipeline{colorFuncCtxs: []*colorFuncCtx{stale}}
	q.AppendConstantColor(colorcore.PMColor4f{R: 0.25, G: 0.5, B: 0.75, A: 1})
	q.AppendColorFunc(func(r, g, b, a float32) (float32, float32, float32, float32) {
		return r, g + 0.25, b, a
	})
	if q.colorFuncCtxN != 1 || q.colorFuncCtxs[0] != stale || stale.fn == nil {
		t.Fatal("expected the retained color-func context to be reused with a fresh transform")
	}
	var out [1]colorcore.PMColor4f
	q.ShadeSpan(0, 0, out[:])
	colorNear(t, out[0], colorcore.PMColor4f{R: 0.25, G: 0.75, B: 0.75, A: 1}, 0, "reused-color-func output")
}

// filterTestChild is an allocation-free, coordinate-dependent child shader for the image-filter runtime kernels: its
// stage is a named (non-capturing) function, so it contributes no per-compile heap allocation (isolating a kernel's own
// allocations in AllocsPerRun), and it reads z.r/z.g so a kernel that samples it at shifted coordinates produces
// varying, premultiplied-valid output.
type filterTestChild struct{}

func (filterTestChild) IsOpaque() bool   { return false }
func (filterTestChild) IsConstant() bool { return false }

func (filterTestChild) appendStages(p *Pipeline, _ MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	p.append(filterTestChildStage)
	return true
}

func filterTestChildStage(z *lanes) {
	for i := range z.n {
		x := z.r[i]
		y := z.g[i]
		a := clamp01(0.02*x + 0.01*y + 0.2)
		z.a[i] = a
		z.r[i] = a * clamp01(0.03*x+0.1)
		z.g[i] = a * clamp01(0.02*y+0.2)
		z.b[i] = a * 0.3
	}
}

// filterKernelCases builds one instance of each image-filter runtime kernel over filterTestChild, with valid uniforms
// (the same shapes the imagefilter package packs).
func filterKernelCases(child Shader) []struct {
	sh   Shader
	name string
} {
	bounds := geom.Rect{Left: 0.5, Top: 0.5, Right: 63.5, Bottom: 63.5}
	kernel := []float32{0, -1, 0, -1, 5, -1, 0, -1, 0} // a 3x3 sharpen
	return []struct {
		sh   Shader
		name string
	}{
		{name: "linear-morphology", sh: NewLinearMorphology(child, geom.Point{X: 1, Y: 0.5}, true, 5)},
		{name: "sparse-morphology", sh: NewSparseMorphology(child, geom.Point{X: 1, Y: 1}, false)},
		{name: "displacement", sh: NewDisplacement(child, child, geom.Point{X: 8, Y: 8}, 0, 3)},
		{name: "normal", sh: NewNormal(child, bounds, -1)},
		{name: "lighting-spot", sh: NewLighting(child, 1, 8, true, LightSpot, [3]float32{10, 10, 10}, 1,
			[3]float32{0, 0, 1}, 0.5, [3]float32{1, 1, 1})},
		{name: "magnifier", sh: NewMagnifier(child, bounds, [4]float32{1, 1, 1.5, 1.5}, [2]float32{0.1, 0.1})},
		{name: "matrix-conv", sh: NewMatrixConv(child, kernel, geom.ISize{Width: 3, Height: 3},
			geom.IPoint{X: 1, Y: 1}, 1, 0, true)},
		{name: "arithmetic", sh: NewArithmeticBlend([4]float32{0.5, 0.25, 0.25, 0}, false, child, child)},
		{name: "filter-decal", sh: NewFilterDecal(child, bounds)},
	}
}

// TestFilterKernelsCompileAllocFree locks the goal for the S94-line's final file: every image-filter runtime kernel
// compiles with zero per-draw heap allocations once the pipeline pool is warm (the capturing closures + new(...)
// scratch of the old form are gone, replaced by static stages reading pooled contexts). Measured through the real
// Compile → RecyclePipeline pool path with an allocation-free child.
func TestFilterKernelsCompileAllocFree(t *testing.T) {
	if raceEnabled {
		t.Skip("testing.AllocsPerRun over-reports under the race detector's allocation instrumentation")
	}
	id := geom.IdentityMatrix()
	for _, tc := range filterKernelCases(filterTestChild{}) {
		allocs := testing.AllocsPerRun(50, func() {
			RecyclePipeline(Compile(tc.sh, id, opaqueBlack))
		})
		if allocs != 0 {
			t.Errorf("%s: compile allocated %.2f objects/run (want 0)", tc.name, allocs)
		}
	}
}

// TestMorphologyContextReuseIsClean locks the pooled-Pipeline invariant for the loop-heavy linear morphology kernel:
// compiled into a pipeline whose retained morphologyCtx carries a prior larger draw's oversized/stale aggregate, +delta
// tap, coordinates, flip sign, and step list (with the counter left nonzero, as RecyclePipeline leaves it), it must
// produce byte-identical output to a fresh compile — exercising the write-before-read scratch discipline and nextStep's
// stepN reset + step refill.
func TestMorphologyContextReuseIsClean(t *testing.T) {
	id := geom.IdentityMatrix()
	sh := NewLinearMorphology(filterTestChild{}, geom.Point{X: 1, Y: 0.5}, true, 3)

	var want [16]colorcore.PMColor4f
	ref := &Pipeline{}
	ref.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(ref, newMatrixRec(id)) {
		t.Fatal("reference appendStages failed")
	}
	ref.ShadeSpan(0, 3, want[:])

	poison := &morphologyCtx{flip: 9, stepN: 99}
	for c := range poison.agg {
		for i := range poison.agg[c] {
			poison.agg[c][i] = 9
			poison.plus[c][i] = 9
		}
	}
	for i := range poison.coords[0] {
		poison.coords[0][i] = 9
		poison.coords[1][i] = 9
	}
	for range 8 { // stale steps a larger prior draw would have left
		poison.steps = append(poison.steps, &morphStep{c: poison, dx: 9, dy: 9})
	}
	p := &Pipeline{morphCtxs: []*morphologyCtx{poison}}
	p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(p, newMatrixRec(id)) {
		t.Fatal("poisoned appendStages failed")
	}
	var got [16]colorcore.PMColor4f
	p.ShadeSpan(0, 3, got[:])
	for i := range got {
		colorNear(t, got[i], want[i], 0, "morphology reused-context pixel")
	}
	if p.morphCtxN != 1 || p.morphCtxs[0] != poison {
		t.Fatalf("expected the poisoned ctx to be reused (n=%d)", p.morphCtxN)
	}
	if poison.stepN != 3 {
		t.Fatalf("expected stepN reset then refilled to the radius (3), got %d", poison.stepN)
	}
}

// TestMatrixConvContextReuseIsClean locks the same invariant for the matrix-convolution kernel, whose accumulator is
// zeroed by an init stage per ShadeSpan chunk and whose per-tap data lives in a retained tap list: a poisoned
// matrixConvCtx (stale sum/origAlpha/coords/scalars, oversized tap list, nonzero counter) must yield byte-identical
// output to a fresh compile.
func TestMatrixConvContextReuseIsClean(t *testing.T) {
	id := geom.IdentityMatrix()
	kernel := []float32{0, -1, 0, -1, 5, -1, 0, -1, 0}
	sh := NewMatrixConv(filterTestChild{}, kernel, geom.ISize{Width: 3, Height: 3},
		geom.IPoint{X: 1, Y: 1}, 1, 0, true)

	var want [16]colorcore.PMColor4f
	ref := &Pipeline{}
	ref.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(ref, newMatrixRec(id)) {
		t.Fatal("reference appendStages failed")
	}
	ref.ShadeSpan(0, 3, want[:])

	poison := &matrixConvCtx{ox: 9, oy: 9, gain: 9, bias: 9, convolveAlpha: false, tapN: 99}
	for c := range poison.sum {
		for i := range poison.sum[c] {
			poison.sum[c][i] = 9
		}
	}
	for i := range poison.origAlpha {
		poison.origAlpha[i] = 9
		poison.coords[0][i] = 9
		poison.coords[1][i] = 9
	}
	for range 25 { // stale taps a larger prior kernel would have left
		poison.taps = append(poison.taps, &matrixConvTap{c: poison, fkx: 9, fky: 9, k: 9, isOrigin: true})
	}
	p := &Pipeline{matConvCtxs: []*matrixConvCtx{poison}}
	p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(p, newMatrixRec(id)) {
		t.Fatal("poisoned appendStages failed")
	}
	var got [16]colorcore.PMColor4f
	p.ShadeSpan(0, 3, got[:])
	for i := range got {
		colorNear(t, got[i], want[i], 0, "matrix-conv reused-context pixel")
	}
	if p.matConvCtxN != 1 || poison.tapN != 9 {
		t.Fatalf("expected the poisoned ctx reused with tapN refilled to 9 (n=%d tapN=%d)",
			p.matConvCtxN, poison.tapN)
	}
}

// TestArithmeticContextReuseIsClean locks the store/load-scratch invariant for the arithmetic blender (the dst child's
// output is saved in res0 across the src child): a poisoned arithmeticCtx must yield byte-identical output to a fresh
// compile (res0 is written before it is read within each chunk).
func TestArithmeticContextReuseIsClean(t *testing.T) {
	id := geom.IdentityMatrix()
	child := filterTestChild{}
	sh := NewArithmeticBlend([4]float32{0.5, 0.25, 0.25, 0}, false, child, child)

	var want [16]colorcore.PMColor4f
	ref := &Pipeline{}
	ref.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(ref, newMatrixRec(id)) {
		t.Fatal("reference appendStages failed")
	}
	ref.ShadeSpan(0, 3, want[:])

	poison := &arithmeticCtx{k: [4]float32{9, 9, 9, 9}, pmClamp: 9}
	for c := range poison.res0 {
		for i := range poison.res0[c] {
			poison.res0[c][i] = 9
		}
	}
	for i := range poison.coords[0] {
		poison.coords[0][i] = 9
		poison.coords[1][i] = 9
	}
	p := &Pipeline{arithCtxs: []*arithmeticCtx{poison}}
	p.paintColor = colorcore.Color4fFromColor(opaqueBlack)
	if !sh.appendStages(p, newMatrixRec(id)) {
		t.Fatal("poisoned appendStages failed")
	}
	var got [16]colorcore.PMColor4f
	p.ShadeSpan(0, 3, got[:])
	for i := range got {
		colorNear(t, got[i], want[i], 0, "arithmetic reused-context pixel")
	}
	if p.arithCtxN != 1 || p.arithCtxs[0] != poison {
		t.Fatalf("expected the poisoned ctx to be reused (n=%d)", p.arithCtxN)
	}
}

// TestFilterKernelsUseDistinctContexts locks that two kernels of the same type in one pipeline (an arithmetic blend
// inlines two morphology children) each receive their own context — a shared one would let the second child's aggregate
// clobber the first's at ShadeSpan time.
func TestFilterKernelsUseDistinctContexts(t *testing.T) {
	id := geom.IdentityMatrix()
	child := filterTestChild{}
	m1 := NewLinearMorphology(child, geom.Point{X: 1}, true, 2)
	m2 := NewLinearMorphology(child, geom.Point{Y: 1}, false, 2)
	p := Compile(NewArithmeticBlend([4]float32{0.5, 0.25, 0.25, 0}, false, m1, m2), id, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.arithCtxN != 1 {
		t.Fatalf("expected 1 arithmetic context, got %d", p.arithCtxN)
	}
	if p.morphCtxN != 2 {
		t.Fatalf("expected 2 morphology contexts, got %d", p.morphCtxN)
	}
	if p.morphCtxs[0] == p.morphCtxs[1] {
		t.Fatal("the two morphology stages share a context (would clobber)")
	}
	RecyclePipeline(p)
}

// TestNextGatherCtxResetsStaleFields locks that nextGatherCtx returns a fully-zeroed gatherCtx even when the retained
// storage carries a prior compile's stale conditional fields (roundDownAtInteger, weights, setRGB): appendStages sets
// those only for some sampling modes, so a reused ctx that kept them would mis-sample a later draw of a different mode.
// (The end-to-end cross-mode transition through the pool — e.g. a nearest draw's stale roundDownAtInteger leaking into
// a following linear/cubic draw — was demonstrated by the oracle's since-removed image-shader render probe with this
// reset deleted; this test pins the reset directly.)
func TestNextGatherCtxResetsStaleFields(t *testing.T) {
	p := &Pipeline{}
	poison := &gatherCtx{
		px:                 &imagecore.Pixels{},
		width:              9,
		height:             9,
		roundDownAtInteger: true,
		setRGB:             [3]float32{9, 9, 9},
		tileX:              tileCtx{scale: 9, invScale: 9, mirrorBiasDir: 1},
		tileY:              tileCtx{scale: 9, invScale: 9, mirrorBiasDir: 1},
	}
	for i := range poison.weights {
		poison.weights[i] = 9
	}
	p.gatherCtxs = []*gatherCtx{poison}
	g := p.nextGatherCtx()
	if g != poison {
		t.Fatal("expected the retained gather ctx to be handed out")
	}
	if (*g != gatherCtx{}) {
		t.Fatalf("nextGatherCtx did not zero the reused ctx: %+v", *g)
	}
	if p.gatherCtxN != 1 {
		t.Fatalf("expected gatherCtxN=1, got %d", p.gatherCtxN)
	}
}

// TestNextDecalCtxResetsStaleFields locks that nextDecalCtx returns a fully-zeroed decalTileCtx even when the retained
// storage carries a prior compile's stale inclusiveEdgeX/Y (set only for nearest sampling) or stale mask bits: a reused
// ctx that kept a nonzero edge would make the decal test spuriously keep texels at that coordinate on a later
// linear/cubic draw.
func TestNextDecalCtxResetsStaleFields(t *testing.T) {
	p := &Pipeline{}
	poison := &decalTileCtx{limitX: 9, limitY: 9, inclusiveEdgeX: 3, inclusiveEdgeY: 3}
	for i := range poison.mask {
		poison.mask[i] = 0xFFFFFFFF
	}
	p.decalCtxs = []*decalTileCtx{poison}
	d := p.nextDecalCtx()
	if d != poison {
		t.Fatal("expected the retained decal ctx to be handed out")
	}
	if (*d != decalTileCtx{}) {
		t.Fatalf("nextDecalCtx did not zero the reused ctx: %+v", *d)
	}
	if p.decalCtxN != 1 {
		t.Fatalf("expected decalCtxN=1, got %d", p.decalCtxN)
	}
}

// TestBlendTwoImagesUseDistinctSamplingContexts locks that two image children in one pipeline (a BlendShader inlines
// both) each receive their own gatherCtx and samplerCtx — a shared context would let the second child's gather/sampling
// clobber the first's at ShadeSpan time. Both children use non-clamp tiling so neither takes the fused fast path (which
// skips the samplerCtx).
func TestBlendTwoImagesUseDistinctSamplingContexts(t *testing.T) {
	img := testImage(t)
	id := geom.IdentityMatrix()
	g0 := NewImage(img, TileRepeat, TileRepeat, SamplingOptions{Filter: FilterLinear}, nil)
	g1 := NewImage(img, TileMirror, TileMirror, SamplingOptions{Filter: FilterLinear}, nil)
	p := Compile(NewBlend(raster.BlendMultiply, g0, g1), id, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.gatherCtxN != 2 {
		t.Fatalf("expected 2 gather contexts, got %d", p.gatherCtxN)
	}
	if p.samplerCtxN != 2 {
		t.Fatalf("expected 2 sampler contexts, got %d", p.samplerCtxN)
	}
	if p.gatherCtxs[0] == p.gatherCtxs[1] {
		t.Fatal("the two image gather stages share a context (would clobber)")
	}
	if p.samplerCtxs[0] == p.samplerCtxs[1] {
		t.Fatal("the two image sampler stages share a context (would clobber)")
	}
}

// TestRecyclePipelineClearsGatherPixels locks that RecyclePipeline drops the source-pixel reference the image gather
// context holds (so an idle pooled pipeline does not pin image pixels) while retaining the ctx storage for reuse.
func TestRecyclePipelineClearsGatherPixels(t *testing.T) {
	img := testImage(t)
	id := geom.IdentityMatrix()
	sh := NewImage(img, TileClamp, TileClamp, SamplingOptions{Filter: FilterLinear}, nil)
	p := Compile(sh, id, opaqueBlack)
	if p == nil {
		t.Fatal("Compile returned nil")
	}
	if p.gatherCtxN == 0 || p.gatherCtxs[0].px == nil {
		t.Fatal("expected a compiled image shader to hold a gather context with pinned pixels")
	}
	RecyclePipeline(p)
	if len(p.gatherCtxs) == 0 {
		t.Fatal("recycle dropped the gather-context storage")
	}
	for i, g := range p.gatherCtxs {
		if g.px != nil {
			t.Fatalf("recycled pipeline still pins pixels on gatherCtxs[%d]", i)
		}
	}
}
