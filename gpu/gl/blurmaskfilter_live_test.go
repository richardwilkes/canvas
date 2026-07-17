// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the GPU mask-filter draw lane (gl.DrawShapeWithMaskFilter). Each test clears an RGBA
// destination, draws a premul-white shape through a blur mask filter, and reads it back. Small shapes exercise the CPU
// mask-generation lane; large shapes (>64px) exercise the GPU mask-generation (coverage set-op Replace) + blur
// (GaussianBlur) lane, and the four blur styles exercise the non-normal composite coverage set-ops
// (Intersect/Union/Difference). Skips when no GL context is available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/maskfilter"
)

// maskFilterDest makes a transparent-cleared RGBA8888-premul destination SDC.
func maskFilterDest(t *testing.T, dc *gl.DirectContext, size int32) *gl.SurfaceDrawContext {
	t.Helper()
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: size, Height: size}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "mask-filter-dest")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	sdc.Clear([4]float32{})
	return sdc
}

// drawBlurredRect draws the rect [l,t,r,b) with a premul-white paint through a blur mask filter of the given
// style/sigma, then returns the readback (RGBA8888). The premul-white paint makes every channel equal the coverage.
func drawBlurredRect(t *testing.T, dc *gl.DirectContext, size int32, l, tp, r, b float32, style maskfilter.BlurStyle, sigma float32) (rgba []byte, size2 int32) {
	t.Helper()
	sdc := maskFilterDest(t, dc, size)

	paint := gl.NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})

	mf := maskfilter.NewBlur(style, sigma, true)
	if mf == nil {
		t.Fatal("NewBlur returned nil")
	}
	shape := gl.MakeStyledShapeRect(geom.RectLTRB(l, tp, r, b), gl.SimpleFillStyle(),
		gl.DoSimplifyYes)
	vm := geom.IdentityMatrix()
	gl.DrawShapeWithMaskFilter(sdc, nil, paint, &vm, mf, &shape)

	return readSDC(t, sdc), size
}

// chanAt returns the R channel (== coverage for premul white) at device pixel (x,y).
func chanAt(data []byte, rowW, x, y int32) int {
	return int(data[(y*rowW+x)*4])
}

// TestLiveMaskFilterBlurCPU draws a small blurred rect (the CPU mask-generation lane, since the 20px shape is below the
// GPU-blur size gate) and checks the blurred structure.
func TestLiveMaskFilterBlurCPU(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size = 64
	data, w := drawBlurredRect(t, dc, size, 22, 22, 42, 42, maskfilter.BlurNormal, 3)

	center := chanAt(data, w, 32, 32)
	corner := chanAt(data, w, 2, 2)
	edge := chanAt(data, w, 32, 20) // 2px above the top edge
	t.Logf("CPU blur: center=%d corner=%d edge=%d", center, corner, edge)
	if center < 200 {
		t.Errorf("center coverage too low: %d", center)
	}
	if corner > 6 {
		t.Errorf("corner should be transparent: %d", corner)
	}
	if edge <= 6 || edge >= 250 {
		t.Errorf("edge should be an intermediate blur value: %d", edge)
	}
}

// TestLiveMaskFilterBlurGPU draws a large blurred rect (>64px), forcing the GPU lane: mask generation renders the
// coverage mask via a Replace coverage set-op, then blurs it.
func TestLiveMaskFilterBlurGPU(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size = 160
	data, w := drawBlurredRect(t, dc, size, 30, 30, 130, 130, maskfilter.BlurNormal, 8)

	center := chanAt(data, w, 80, 80)
	far := chanAt(data, w, 5, 5)
	edgeOut := chanAt(data, w, 137, 80) // 7px right of the right edge
	t.Logf("GPU blur: center=%d far=%d edgeOut=%d", center, far, edgeOut)
	if center < 200 {
		t.Errorf("center coverage too low: %d", center)
	}
	if far > 8 {
		t.Errorf("far corner should be transparent: %d", far)
	}
	if edgeOut <= 12 || edgeOut >= 250 {
		t.Errorf("just-outside edge should be an intermediate blur value: %d", edgeOut)
	}
}

// TestLiveMaskFilterBlurStylesGPU exercises the four blur styles through the GPU lane. The non-normal styles composite
// the sharp original mask over the blurred mask with a coverage set-op: Solid=Union, Outer=Difference, Inner=Intersect.
func TestLiveMaskFilterBlurStylesGPU(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size = 160
	const l, tp, r, b = 30, 30, 130, 130
	const sigma = 8

	// Sample points: center (inside, blur~1), insideEdge (2px inside the right edge), edgeOut (7px outside), far (well
	// outside).
	sample := func(style maskfilter.BlurStyle) (center, insideEdge, edgeOut, far int) {
		data, w := drawBlurredRect(t, dc, size, l, tp, r, b, style, sigma)
		return chanAt(data, w, 80, 80), chanAt(data, w, 128, 80), chanAt(data, w, 137, 80),
			chanAt(data, w, 158, 80)
	}

	nCenter, nInside, nEdgeOut, nFar := sample(maskfilter.BlurNormal)
	t.Logf("Normal: center=%d insideEdge=%d edgeOut=%d far=%d", nCenter, nInside, nEdgeOut, nFar)
	if nCenter < 200 || nEdgeOut <= 12 || nFar > 8 {
		t.Errorf("Normal signature wrong: center=%d edgeOut=%d far=%d", nCenter, nEdgeOut, nFar)
	}

	sCenter, sInside, sEdgeOut, _ := sample(maskfilter.BlurSolid)
	t.Logf("Solid: center=%d insideEdge=%d edgeOut=%d", sCenter, sInside, sEdgeOut)
	if sCenter < 250 || sInside < 250 || sEdgeOut <= 12 {
		t.Errorf("Solid should be fully solid inside with a fuzzy outside: center=%d insideEdge=%d edgeOut=%d",
			sCenter, sInside, sEdgeOut)
	}
	// The union makes the inside strictly more covered than the plain blur (Normal) near the edge.
	if sInside <= nInside {
		t.Errorf("Solid inside-edge (%d) should exceed Normal's blurred inside-edge (%d)", sInside,
			nInside)
	}

	oCenter, _, oEdgeOut, oFar := sample(maskfilter.BlurOuter)
	t.Logf("Outer: center=%d edgeOut=%d far=%d", oCenter, oEdgeOut, oFar)
	if oCenter > 8 || oEdgeOut <= 12 || oFar > 8 {
		t.Errorf("Outer should be empty inside with a fuzzy outside: center=%d edgeOut=%d far=%d",
			oCenter, oEdgeOut, oFar)
	}

	iCenter, _, iEdgeOut, iFar := sample(maskfilter.BlurInner)
	t.Logf("Inner: center=%d edgeOut=%d far=%d", iCenter, iEdgeOut, iFar)
	if iCenter < 200 || iEdgeOut > 8 || iFar > 8 {
		t.Errorf("Inner should be covered inside with nothing outside: center=%d edgeOut=%d far=%d",
			iCenter, iEdgeOut, iFar)
	}
}
