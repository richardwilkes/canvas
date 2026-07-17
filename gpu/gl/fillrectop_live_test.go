// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the fill-rect draw path: coverage-AA fill rects compared against the analytic axis-aligned
// coverage (the Go-GPU vs Go-CPU self-consistency lane — for axis-aligned rects both the CPU analytic-AA rasterizer and
// the GPU's inset/outset coverage ramp reduce to exact area coverage), rotated-quad AA sanity, batched multi-color
// draws, and the FixedClip scissor. Skips when no GL context is available.

package gl_test

import (
	"fmt"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
)

func newLiveDrawSDC(t testing.TB, dc *gl.DirectContext, w, h int32) *gl.SurfaceDrawContext {
	t.Helper()
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "live-fill-rect")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	return sdc
}

func livePaint(c colorcore.PMColor4f) *gl.Paint {
	paint := gl.NewPaint()
	paint.SetColor4f(c)
	return paint
}

// axisCoverage is the exact area coverage of pixel (x, y) under the axis-aligned rect.
func axisCoverage(r geom.Rect, x, y int32) float32 {
	ox := min(r.Right, float32(x+1)) - max(r.Left, float32(x))
	oy := min(r.Bottom, float32(y+1)) - max(r.Top, float32(y))
	if ox <= 0 || oy <= 0 {
		return 0
	}
	return min(ox, 1) * min(oy, 1)
}

func TestLiveFillRectAAPixels(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	bg := [4]float32{1, 1, 1, 1}
	sdc.Clear(bg)

	rect := geom.Rect{Left: 10.25, Top: 10.25, Right: 30.75, Bottom: 20.75}
	src := colorcore.PMColor4f{R: 0.1, G: 0.2, B: 0.4, A: 0.5}
	identity := geom.IdentityMatrix()
	sdc.FillRectToRect(nil, livePaint(src), gpu.AAYes, &identity, rect, rect)

	data := readSDC(t, sdc)
	rb := 64 * 4
	for y := int32(8); y < 24; y++ {
		for x := int32(8); x < 34; x++ {
			// Corner cells are exempt from the exact comparison: the GPU's AA ring interpolates coverage linearly
			// across the corner triangles (~0.75 here) while the exact area is the x/y product (0.5625) — an inherent
			// property of the per-edge-AA quad method that the GPU thresholds absorb. Edge (single fractional axis) and
			// interior pixels must match the analytic area exactly.
			ox := min(rect.Right, float32(x+1)) - max(rect.Left, float32(x))
			oy := min(rect.Bottom, float32(y+1)) - max(rect.Top, float32(y))
			corner := ox > 0 && ox < 1 && oy > 0 && oy < 1
			if corner {
				continue
			}
			cov := axisCoverage(rect, x, y)
			// Coverage-as-alpha src-over: out = src*cov + dst*(1 - a*cov).
			want := [4]float32{}
			srcv := [4]float32{src.R, src.G, src.B, src.A}
			for c := range 4 {
				want[c] = srcv[c]*cov + bg[c]*(1-src.A*cov)
			}
			off := int(y)*rb + int(x)*4
			for c := range 4 {
				wantByte := int(want[c]*255 + 0.5)
				got := int(data[off+c])
				if d := got - wantByte; d < -3 || d > 3 {
					t.Fatalf("pixel (%d,%d) channel %d = %d, want %d (cov %v)",
						x, y, c, got, wantByte, cov)
				}
			}
		}
	}
}

func TestLiveFillRectRotatedAA(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	// A 20x20 square rotated 45 degrees about its center (32, 32).
	view := geom.IdentityMatrix()
	view.SetRotatePivot(45, 32, 32)
	rect := geom.Rect{Left: 22, Top: 22, Right: 42, Bottom: 42}
	src := colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 1}
	sdc.FillRectToRect(nil, livePaint(src), gpu.AAYes, &view, rect, rect)

	data := readSDC(t, sdc)
	rb := 64 * 4
	greenAt := func(x, y int32) int { return int(data[int(y)*rb+int(x)*4+1]) }

	// Interior points (well inside the rotated square) are fully green; far exterior points stay black; every pixel is
	// within [0, 255] green with red/blue untouched.
	for _, p := range [][2]int32{{32, 32}, {32, 26}, {26, 32}, {32, 38}, {38, 32}} {
		if got := greenAt(p[0], p[1]); got != 255 {
			t.Fatalf("interior (%d,%d) green = %d, want 255", p[0], p[1], got)
		}
	}
	for _, p := range [][2]int32{{5, 5}, {58, 58}, {32, 10}, {10, 32}, {32, 54}} {
		if got := greenAt(p[0], p[1]); got != 0 {
			t.Fatalf("exterior (%d,%d) green = %d, want 0", p[0], p[1], got)
		}
	}
	// The diagonal edge midpoint lands half-covered (the rotated edge passes through pixel centers along x+y =
	// 32+14.14...): sample the edge band and require partial coverage somewhere within it.
	partial := false
	for d := int32(-2); d <= 2; d++ {
		g := greenAt(32+7+d, 32-7-d)
		if g > 30 && g < 225 {
			partial = true
			break
		}
	}
	if !partial {
		t.Fatal("no partial coverage found along the rotated AA edge")
	}
}

func TestLiveFillRectBatchedColors(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})
	identity := geom.IdentityMatrix()

	colors := []colorcore.PMColor4f{
		{R: 1, G: 0, B: 0, A: 1},
		{R: 0, G: 1, B: 0, A: 1},
		{R: 0, G: 0, B: 1, A: 1},
		{R: 1, G: 1, B: 0, A: 1},
	}
	// Distinct colors with identical trivial paints: these merge into one FillRectOp (asserted on the fake driver);
	// here we verify the merged draw writes each quad's own color.
	for i, c := range colors {
		x := float32(i * 16)
		r := geom.Rect{Left: x + 2, Top: 2, Right: x + 14, Bottom: 14}
		sdc.FillRectToRect(nil, livePaint(c), gpu.AANo, &identity, r, r)
	}

	data := readSDC(t, sdc)
	rb := 64 * 4
	for i, c := range colors {
		x := int32(i*16 + 8)
		off := int(8)*rb + int(x)*4
		want := [4]byte{byte(c.R*255 + 0.5), byte(c.G*255 + 0.5), byte(c.B*255 + 0.5), 255}
		got := [4]byte{data[off], data[off+1], data[off+2], data[off+3]}
		if got != want {
			t.Fatalf("batched rect %d pixel = %v, want %v", i, got, want)
		}
		gap := int(8)*rb + i*16*4 // x = 16*i lies in the 2px gap before each rect
		if data[gap] != 0 || data[gap+1] != 0 || data[gap+2] != 0 {
			t.Fatalf("gap before rect %d was painted: %v", i,
				[3]byte{data[gap], data[gap+1], data[gap+2]})
		}
	}
}

func TestLiveFillRectScissorClip(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	// A rotated quad covering the whole surface under a rect clip: the rotation prevents the clip from folding into the
	// geometry, so this exercises the real scissor state.
	clip := gl.NewFixedClipRect(geom.ISize{Width: 64, Height: 64}, geom.IRectXYWH(16, 16, 24, 24))
	view := geom.IdentityMatrix()
	view.SetRotatePivot(30, 32, 32)
	big := geom.Rect{Left: -64, Top: -64, Right: 128, Bottom: 128}
	sdc.FillRectToRect(clip, livePaint(colorcore.PMColor4f{R: 1, G: 0, B: 0, A: 1}), gpu.AANo,
		&view, big, big)

	data := readSDC(t, sdc)
	rb := 64 * 4
	check := func(x, y int32, wantRed byte, tag string) {
		t.Helper()
		off := int(y)*rb + int(x)*4
		if data[off] != wantRed {
			t.Fatalf("%s (%d,%d): red = %d, want %d", tag, x, y, data[off], wantRed)
		}
	}
	check(32, 32, 255, "inside scissor")
	check(17, 17, 255, "scissor corner inside")
	check(8, 32, 0, "left of scissor")
	check(32, 8, 0, "above scissor")
	check(56, 32, 0, "right of scissor")
	check(32, 56, 0, "below scissor")
}

// TestLiveFillRectLocalRectMapping exercises the local-quad plumbing with localRect != rectToDraw through a texture
// effect: sampling a 2x1 texture with distinct halves through a local rect covering only the left texel must paint the
// whole draw with that texel. (If the device rect leaked into the local coordinates the sample would clamp to the right
// texel instead.)
func TestLiveFillRectLocalRectMapping(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	sdc := newLiveDrawSDC(t, dc, 16, 16)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	// Left texel green, right texel blue.
	texData := []byte{0, 255, 0, 255, 0, 0, 255, 255}
	var key gpu.UniqueKey
	builder := gpu.UniqueKeyBuilder(&key, gpu.GenerateUniqueKeyDomain(), 1, "LocalRectTexture")
	builder.Slice()[0] = 7
	builder.Finish()
	view := gl.FindOrCreatePixelsProxyView(dc, &key, geom.ISize{Width: 2, Height: 1},
		gpu.ColorTypeRGBA8888, texData, 8, "local-rect-tex")
	if !view.IsValid() {
		t.Fatal("FindOrCreatePixelsProxyView failed")
	}
	sampler := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, gpu.FilterModeNearest,
		gpu.MipmapModeNone)
	identity := geom.IdentityMatrix()
	te := gl.MakeTextureEffect(view, gpu.AlphaTypePremul, identity, sampler, dc.GLCaps(),
		[4]float32{})

	paint := gl.NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	paint.SetColorFragmentProcessor(te)

	dst := geom.Rect{Right: 16, Bottom: 16}
	local := geom.Rect{Right: 1, Bottom: 1} // the left texel of the 2x1 texture
	sdc.FillRectToRect(nil, paint, gpu.AANo, &identity, dst, local)

	data := readSDC(t, sdc)
	rb := 16 * 4
	for _, p := range [][2]int32{{2, 2}, {8, 8}, {13, 13}} {
		off := int(p[1])*rb + int(p[0])*4
		got := fmt.Sprintf("%d,%d,%d", data[off], data[off+1], data[off+2])
		if got != "0,255,0" {
			t.Fatalf("local-rect pixel (%d,%d) = %s, want left texel green", p[0], p[1], got)
		}
	}
}
