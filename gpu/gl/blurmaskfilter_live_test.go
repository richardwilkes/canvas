// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the GPU mask-filter draw lane (gl.DrawShapeWithMaskFilter). Each test clears an RGBA
// destination, draws a premul-white shape through a blur mask filter, and reads it back. A rect/circle/simple-circular
// rrect blurred Normal with a simple fill is absorbed by directFilterMask's analytic blur FPs, so the mask-generation
// lanes are reached only by shapes those profiles reject: the diamond used here is a general path, so a small diamond
// exercises the CPU mask-generation lane and a large one (>64px) the GPU mask-generation (coverage set-op Replace) +
// blur (GaussianBlur) lane. The four blur styles then exercise the non-normal composite coverage set-ops
// (Intersect/Union/Difference). Every test asserts the lane its draw actually took. Skips when no GL context is
// available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
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

// drawBlurredShape draws 'shape' with a premul-white paint through a blur mask filter of the given style/sigma into a
// freshly cleared size×size destination, then returns the readback (RGBA8888) and the mask-filter lane the draw took.
// The premul-white paint makes every channel equal the coverage.
func drawBlurredShape(t *testing.T, dc *gl.DirectContext, size int32, shape *gl.StyledShape, style maskfilter.BlurStyle, sigma float32) (rgba []byte, lane string) {
	t.Helper()
	sdc := maskFilterDest(t, dc, size)

	paint := gl.NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})

	mf := maskfilter.NewBlur(style, sigma, true)
	if mf == nil {
		t.Fatal("NewBlur returned nil")
	}
	vm := geom.IdentityMatrix()
	lane = gl.DrawShapeWithMaskFilterLaneForTest(sdc, nil, paint, &vm, mf, shape)

	return readSDC(t, sdc), lane
}

// drawBlurredRect draws the rect [l,t,r,b) through drawBlurredShape and returns the readback plus its row width.
func drawBlurredRect(t *testing.T, dc *gl.DirectContext, size int32, l, tp, r, b float32, style maskfilter.BlurStyle, sigma float32) (rgba []byte, size2 int32) {
	t.Helper()
	shape := gl.MakeStyledShapeRect(geom.RectLTRB(l, tp, r, b), gl.SimpleFillStyle(),
		gl.DoSimplifyYes)
	rgba, _ = drawBlurredShape(t, dc, size, &shape, style, sigma)
	return rgba, size
}

// diamondShape builds the axis-aligned diamond inscribed in the rect [l,t,r,b) as a general path. Neither the rect, the
// circle, nor the rrect analytic blur profile accepts it, so a Normal-blurred simple fill of this shape falls through
// directFilterMask into the general filtered-mask path.
func diamondShape(l, tp, r, b float32) gl.StyledShape {
	cx := (l + r) / 2
	cy := (tp + b) / 2
	p := path.New()
	p.MoveTo(cx, tp)
	p.LineTo(r, cy)
	p.LineTo(cx, b)
	p.LineTo(l, cy)
	p.Close()
	return gl.MakeStyledShapePath(p, gl.SimpleFillStyle(), gl.DoSimplifyYes)
}

// chanAt returns the R channel (== coverage for premul white) at device pixel (x,y).
func chanAt(data []byte, rowW, x, y int32) int {
	return int(data[(y*rowW+x)*4])
}

// TestLiveMaskFilterBlurNormalRectIsDirect pins why the mask-generation tests below cannot use a rect: a Normal-blurred
// simple-fill rect always qualifies for MakeRectBlur's analytic profile (there is no minimum-size bailout), so the draw
// returns from directFilterMask at both the small and the large size, never reaching either mask-generation lane.
func TestLiveMaskFilterBlurNormalRectIsDirect(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	small := gl.MakeStyledShapeRect(geom.RectLTRB(22, 22, 42, 42), gl.SimpleFillStyle(), gl.DoSimplifyYes)
	if _, lane := drawBlurredShape(t, dc, 64, &small, maskfilter.BlurNormal, 3); lane != gl.MaskFilterLaneDirectName {
		t.Errorf("small Normal-blurred rect took the %q lane, want %q", lane, gl.MaskFilterLaneDirectName)
	}
	large := gl.MakeStyledShapeRect(geom.RectLTRB(30, 30, 130, 130), gl.SimpleFillStyle(), gl.DoSimplifyYes)
	if _, lane := drawBlurredShape(t, dc, 160, &large, maskfilter.BlurNormal, 8); lane != gl.MaskFilterLaneDirectName {
		t.Errorf("large Normal-blurred rect took the %q lane, want %q", lane, gl.MaskFilterLaneDirectName)
	}
}

// TestLiveMaskFilterBlurCPU draws a small blurred diamond — a general path, so no analytic profile applies — and checks
// both that the draw took the CPU mask-generation lane (the 20px shape is below the GPU-blur size gate) and that the
// blurred structure is right.
func TestLiveMaskFilterBlurCPU(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size int32 = 64
	shape := diamondShape(22, 22, 42, 42)
	data, lane := drawBlurredShape(t, dc, size, &shape, maskfilter.BlurNormal, 3)
	if lane != gl.MaskFilterLaneSWName {
		t.Fatalf("draw took the %q lane, want %q", lane, gl.MaskFilterLaneSWName)
	}
	w := size

	center := chanAt(data, w, 32, 32)
	corner := chanAt(data, w, 2, 2)
	edge := chanAt(data, w, 37, 27) // on the upper-right edge (x-y == 10), so ~half covered
	t.Logf("CPU blur: center=%d corner=%d edge=%d", center, corner, edge)
	if center < 200 {
		t.Errorf("center coverage too low: %d", center)
	}
	if corner > 6 {
		t.Errorf("corner should be transparent: %d", corner)
	}
	if edge <= 60 || edge >= 200 {
		t.Errorf("on-edge coverage should be about half: %d", edge)
	}
}

// TestLiveMaskFilterBlurGPU draws a large blurred diamond (>64px, and again a general path so no analytic profile
// applies), forcing the GPU lane: mask generation renders the coverage mask via a Replace coverage set-op, then blurs
// it.
func TestLiveMaskFilterBlurGPU(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size int32 = 160
	shape := diamondShape(30, 30, 130, 130)
	data, lane := drawBlurredShape(t, dc, size, &shape, maskfilter.BlurNormal, 8)
	if lane != gl.MaskFilterLaneHWName {
		t.Fatalf("draw took the %q lane, want %q", lane, gl.MaskFilterLaneHWName)
	}
	w := size

	center := chanAt(data, w, 80, 80)
	far := chanAt(data, w, 5, 5)
	edgeOut := chanAt(data, w, 124, 64) // 7px outside the upper-right edge, along its normal
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
// The diamond keeps the Normal baseline on the same lane as the other three (a rect would take the analytic lane).
func TestLiveMaskFilterBlurStylesGPU(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size int32 = 160
	const sigma = 8
	shape := diamondShape(30, 30, 130, 130)

	// Sample points, all relative to the upper-right edge (the diamond boundary is x-y == 50 there): center (inside,
	// blur~1), insideEdge (~3px inside), edgeOut (~7px outside), far (well outside).
	sample := func(style maskfilter.BlurStyle) (center, insideEdge, edgeOut, far int) {
		data, lane := drawBlurredShape(t, dc, size, &shape, style, sigma)
		if lane != gl.MaskFilterLaneHWName {
			t.Fatalf("%v draw took the %q lane, want %q", style, lane, gl.MaskFilterLaneHWName)
		}
		w := size
		return chanAt(data, w, 80, 80), chanAt(data, w, 114, 68), chanAt(data, w, 124, 64),
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
