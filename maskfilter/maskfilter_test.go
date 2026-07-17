// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package maskfilter

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

func TestNewBlurRejections(t *testing.T) {
	if NewBlur(BlurNormal, 0, true) != nil {
		t.Error("sigma 0 accepted")
	}
	if NewBlur(BlurNormal, -2, true) != nil {
		t.Error("negative sigma accepted")
	}
	nan := float32(math.NaN())
	if NewBlur(BlurNormal, nan, true) != nil {
		t.Error("NaN sigma accepted")
	}
	if NewBlur(BlurNormal, 3, true) == nil {
		t.Error("valid blur rejected")
	}
}

func TestComputeXformedSigma(t *testing.T) {
	f := NewBlur(BlurNormal, 4, true).(*blurMaskFilter)
	var m geom.Matrix
	m.SetScale(2, 2)
	if got := f.computeXformedSigma(&m); got != 8 {
		t.Errorf("respectCTM scale-2 sigma = %v, want 8", got)
	}
	fi := NewBlur(BlurNormal, 4, false).(*blurMaskFilter)
	if got := fi.computeXformedSigma(&m); got != 4 {
		t.Errorf("ignoreXform sigma = %v, want 4", got)
	}
	// The 128 cap.
	m.SetScale(100, 100)
	if got := f.computeXformedSigma(&m); got != 128 {
		t.Errorf("capped sigma = %v, want 128", got)
	}
}

func TestBlurFastBounds(t *testing.T) {
	f := NewBlur(BlurNormal, 5, true)
	src := geom.RectLTRB(10, 20, 30, 40)
	got := f.ComputeFastBounds(src)
	want := geom.RectLTRB(-5, 5, 45, 55)
	if got != want {
		t.Errorf("blur fast bounds = %+v, want %+v", got, want)
	}
}

func TestBoxBlurNoBlurLanes(t *testing.T) {
	src := &raster.Mask{
		Image:    []uint8{255},
		Bounds:   geom.IRectXYWH(10, 10, 1, 1),
		RowBytes: 1,
	}
	// Below the 1/3 window threshold: normal fails (caller draws unfiltered)...
	if _, _, ok := boxBlur(src, 0.2, BlurNormal); ok {
		t.Error("no-blur normal should report failure")
	}
	// ...but outer succeeds with an empty mask.
	dst, _, ok := boxBlur(src, 0.2, BlurOuter)
	if !ok || dst.Image != nil || !dst.Bounds.IsEmpty() {
		t.Errorf("no-blur outer: ok=%v dst=%+v", ok, dst)
	}
}

// referenceTripleBox computes the three-pass box blur PlanGauss implements directly from its definition: each pass
// averages a sliding window, chained three times in exact integer math via the same 32.32 rounding.
func referenceTripleBox(src []uint8, p *planGauss) []uint8 {
	window0 := p.pass0Size + 1
	window1 := p.pass1Size + 1
	window2 := p.pass2Size + 1
	n := len(src) + 2*p.border

	// Pass k: out[i] = sum(in[i-window+1..i]) — a running box sum matching the ring-buffer scan.
	pass := func(in []uint64, window int) []uint64 {
		out := make([]uint64, len(in)+window-1)
		for i := range out {
			var sum uint64
			for j := 0; j < window; j++ {
				k := i - j
				if k >= 0 && k < len(in) {
					sum += in[k]
				}
			}
			out[i] = sum
		}
		return out
	}
	in := make([]uint64, len(src))
	for i, v := range src {
		in[i] = uint64(v)
	}
	acc := pass(pass(pass(in, window0), window1), window2)
	out := make([]uint8, n)
	for i := range out {
		v := (p.weight*acc[i] + (1 << 31)) >> 32
		out[i] = uint8(v)
	}
	return out
}

func TestGaussScanMatchesReference(t *testing.T) {
	for _, sigma := range []float64{2.0, 3.7, 8.25} {
		p := newPlanGauss(sigma)
		src := []uint8{0, 12, 250, 255, 190, 3, 77, 255, 255, 255, 0, 0, 40}
		dst := make([]uint8, len(src)+2*p.border)
		scan := p.makeBlurScan(len(src), make([]uint32, p.bufferSize()))
		scan.blur(src, dst, 0, 1, len(dst))
		want := referenceTripleBox(src, &p)
		for i := range dst {
			if dst[i] != want[i] {
				t.Fatalf("sigma %v: dst[%d] = %d, want %d", sigma, i, dst[i], want[i])
			}
		}
	}
}

func TestGaussScanNarrowMask(t *testing.T) {
	// A mask narrower than the sliding window exercises the noChangeCount lane.
	p := newPlanGauss(6)
	src := []uint8{255, 128}
	dst := make([]uint8, len(src)+2*p.border)
	scan := p.makeBlurScan(len(src), make([]uint32, p.bufferSize()))
	scan.blur(src, dst, 0, 1, len(dst))
	want := referenceTripleBox(src, &p)
	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d] = %d, want %d", i, dst[i], want[i])
		}
	}
}

func TestSmallBlurConservation(t *testing.T) {
	// The discrete Gaussian is normalized, so blurring a wide uniform run keeps its center at the source value and
	// produces a symmetric ramp at the edges.
	src := &raster.Mask{
		Image:    make([]uint8, 32*8),
		Bounds:   geom.IRectXYWH(0, 0, 32, 8),
		RowBytes: 32,
	}
	for i := range src.Image {
		src.Image[i] = 200
	}
	dst, _, ok := boxBlur(src, 1.2, BlurNormal)
	if !ok {
		t.Fatal("small blur failed")
	}
	// Center pixel keeps the source value (within fixed-point rounding).
	cx := int(dst.Bounds.Width()) / 2
	cy := int(dst.Bounds.Height()) / 2
	center := dst.Image[cy*int(dst.RowBytes)+cx]
	if int(center) < 198 || int(center) > 200 {
		t.Errorf("center after normalized blur = %d, want ~200", center)
	}
	// Horizontal symmetry.
	row := dst.Image[cy*int(dst.RowBytes) : cy*int(dst.RowBytes)+int(dst.Bounds.Width())]
	for i, j := 0, len(row)-1; i < j; i, j = i+1, j-1 {
		if row[i] != row[j] {
			t.Fatalf("asymmetric blur: row[%d]=%d row[%d]=%d", i, row[i], j, row[j])
		}
	}
}

func TestBlurStylesOnBoxMask(t *testing.T) {
	// A solid box: solid style keeps the interior at 255, outer zeroes it, inner keeps the blur values inside and clips
	// the outside.
	w, h := 20, 16
	src := &raster.Mask{
		Image:    make([]uint8, w*h),
		Bounds:   geom.IRectXYWH(0, 0, int32(w), int32(h)),
		RowBytes: int32(w),
	}
	for i := range src.Image {
		src.Image[i] = 255
	}
	for _, style := range []BlurStyle{BlurNormal, BlurSolid, BlurOuter, BlurInner} {
		dst, margin, ok := boxBlur(src, 2.5, style)
		if !ok {
			t.Fatalf("style %d failed", style)
		}
		switch style {
		case BlurInner:
			if dst.Bounds != src.Bounds {
				t.Errorf("inner bounds = %+v, want %+v", dst.Bounds, src.Bounds)
			}
		default:
			wantW := src.Bounds.Width() + 2*margin.X
			if dst.Bounds.Width() != wantW {
				t.Errorf("style %d width = %d, want %d", style, dst.Bounds.Width(), wantW)
			}
		}
		cx := int(dst.Bounds.Width()) / 2
		cy := int(dst.Bounds.Height()) / 2
		center := dst.Image[cy*int(dst.RowBytes)+cx]
		switch style {
		case BlurSolid:
			if center != 255 {
				t.Errorf("solid center = %d, want 255", center)
			}
		case BlurOuter:
			if center != 0 {
				t.Errorf("outer center = %d, want 0", center)
			}
		default:
			if center < 250 {
				t.Errorf("style %d center = %d, want ~255", style, center)
			}
		}
	}
}

func TestTableFilters(t *testing.T) {
	var gamma [256]uint8
	MakeGammaTable(&gamma, 2.0)
	if gamma[0] != 0 || gamma[255] != 255 {
		t.Errorf("gamma endpoints: %d %d", gamma[0], gamma[255])
	}
	// x^2 at 128/255: (128/255)^2*255 ≈ 64.25 → 64.
	if gamma[128] != 64 {
		t.Errorf("gamma[128] = %d, want 64", gamma[128])
	}

	var clip [256]uint8
	MakeClipTable(&clip, 64, 192)
	if clip[64] != 0 || clip[192] != 255 || clip[0] != 0 || clip[255] != 255 {
		t.Errorf("clip endpoints: %d %d %d %d", clip[0], clip[64], clip[192], clip[255])
	}
	// Midpoint scales to ~half.
	if clip[128] < 126 || clip[128] > 129 {
		t.Errorf("clip[128] = %d", clip[128])
	}
	// Degenerate ranges are repaired (max 0 → 1; min >= max → max-1).
	MakeClipTable(&clip, 10, 0)
	if clip[0] != 0 || clip[1] != 255 {
		t.Errorf("degenerate clip: %d %d", clip[0], clip[1])
	}

	// FilterMask applies the LUT with align-4 row bytes and zeroed padding.
	var table [256]uint8
	for i := range table {
		table[i] = uint8(255 - i)
	}
	mf := NewTable(&table)
	src := &raster.Mask{
		Image:    []uint8{0, 128, 255, 10, 20, 30},
		Bounds:   geom.IRectXYWH(5, 5, 3, 2),
		RowBytes: 3,
	}
	ident := geom.IdentityMatrix()
	dst, margin, ok := mf.FilterMask(src, &ident)
	if !ok || margin != (geom.IPoint{}) {
		t.Fatalf("table FilterMask: ok=%v margin=%+v", ok, margin)
	}
	if dst.RowBytes != 4 {
		t.Errorf("row bytes = %d, want 4 (align4)", dst.RowBytes)
	}
	want := []uint8{255, 127, 0, 0, 245, 235, 225, 0}
	for i, w := range want {
		if dst.Image[i] != w {
			t.Errorf("dst[%d] = %d, want %d", i, dst.Image[i], w)
		}
	}
}

func TestBlurRectAnalytic(t *testing.T) {
	// An integral rect blurred analytically: bounds pad by profileSize/2, the mask is symmetric, and the center of a
	// large-enough rect saturates.
	scratch := &blurScratch{}
	var dst raster.Mask
	margin, ok := blurRect(2, geom.RectLTRB(10, 10, 40, 34), BlurNormal, computeBoundsAndRenderImage, scratch, &dst)
	if !ok {
		t.Fatal("blurRect failed")
	}
	pad := int32(int(geom.CeilToInt(6*2)) / 2)
	if margin.X != pad || margin.Y != pad {
		t.Errorf("margin = %+v, want %d", margin, pad)
	}
	if dst.Bounds != geom.IRectLTRB(10-pad, 10-pad, 40+pad, 34+pad) {
		t.Errorf("bounds = %+v", dst.Bounds)
	}
	w := int(dst.Bounds.Width())
	h := int(dst.Bounds.Height())
	center := dst.Image[h/2*w+w/2]
	if center != 255 {
		t.Errorf("center = %d, want 255", center)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w/2; x++ {
			if dst.Image[y*w+x] != dst.Image[y*w+w-1-x] {
				t.Fatalf("asymmetric at (%d,%d)", x, y)
			}
		}
	}
	// justComputeBounds produces no image (dst.Image stays nil).
	_, ok = blurRect(2, geom.RectLTRB(10, 10, 40, 34), BlurNormal, justComputeBounds, scratch, &dst)
	if !ok || dst.Image != nil {
		t.Errorf("bounds-only mode allocated an image")
	}
}

func TestShaderMaskFilterNil(t *testing.T) {
	if NewShader(nil) != nil {
		t.Error("nil shader accepted")
	}
}
