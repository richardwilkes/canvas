// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the clip stack: the device-rect clip folding into edge AA (pixel-exact against analytic area
// coverage), the analytic rrect/convex-poly coverage FPs, the software clip mask (device-space texture sampling), the
// stencil clip (an element past the analytic FP budget really masks pixels), and the difference-op inverse fill. Skips
// when no GL context is available.

package gl_test

import (
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

func liveClipStack() *gl.ClipStack {
	return gl.NewClipStack(geom.IRect{Right: 64, Bottom: 64}, false)
}

// fillSurface draws rect with an opaque solid color through the clip.
func fillClipped(sdc *gl.SurfaceDrawContext, cs *gl.ClipStack, c colorcore.PMColor4f, rect geom.Rect, aa gpu.AA) {
	identity := geom.IdentityMatrix()
	sdc.FillRectToRect(cs, livePaint(c), aa, &identity, rect, rect)
}

func pxAt(data []byte, x, y int32) [4]int {
	off := int(y)*64*4 + int(x)*4
	return [4]int{int(data[off]), int(data[off+1]), int(data[off+2]), int(data[off+3])}
}

// TestLiveClipRectFold: an AA device-rect clip folds into the draw's edge AA through attemptQuadOptimization; the
// result is pixel-exact area coverage of the clip rect.
func TestLiveClipRectFold(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	bg := [4]float32{1, 1, 1, 1}
	sdc.Clear(bg)

	clipRect := geom.Rect{Left: 10.25, Top: 10.25, Right: 30.75, Bottom: 20.75}
	cs := liveClipStack()
	identity := geom.IdentityMatrix()
	cs.ClipRect(&identity, clipRect, gpu.AAYes, raster.ClipIntersect)

	src := colorcore.PMColor4f{R: 0.1, G: 0.2, B: 0.4, A: 0.5}
	fillClipped(sdc, cs, src, geom.Rect{Right: 64, Bottom: 64}, gpu.AANo)

	data := readSDC(t, sdc)
	for y := int32(8); y < 24; y++ {
		for x := int32(8); x < 34; x++ {
			// Corner cells are exempt (per-edge-AA corner interpolation).
			ox := min(clipRect.Right, float32(x+1)) - max(clipRect.Left, float32(x))
			oy := min(clipRect.Bottom, float32(y+1)) - max(clipRect.Top, float32(y))
			if ox > 0 && ox < 1 && oy > 0 && oy < 1 {
				continue
			}
			cov := axisCoverage(clipRect, x, y)
			srcv := [4]float32{src.R, src.G, src.B, src.A}
			got := pxAt(data, x, y)
			for c := range 4 {
				want := int(srcv[c]*cov*255 + bg[c]*(1-src.A*cov)*255 + 0.5)
				if d := got[c] - want; d < -3 || d > 3 {
					t.Fatalf("pixel (%d,%d) channel %d = %d, want %d (cov %v)",
						x, y, c, got[c], want, cov)
				}
			}
		}
	}
}

// TestLiveClipRRectFP: a circular-corner rrect clip rides the CircularRRectEffect.
func TestLiveClipRRectFP(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	rr := geom.MakeRRect(geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 56}, 12, 12)
	cs := liveClipStack()
	identity := geom.IdentityMatrix()
	cs.ClipRRect(&identity, rr, gpu.AAYes, raster.ClipIntersect)

	src := colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 1}
	fillClipped(sdc, cs, src, geom.Rect{Right: 64, Bottom: 64}, gpu.AANo)

	data := readSDC(t, sdc)
	// Interior points are fully green.
	for _, p := range [][2]int32{{32, 32}, {12, 32}, {32, 12}, {51, 51}} {
		if g := pxAt(data, p[0], p[1])[1]; g < 250 {
			t.Fatalf("interior (%d,%d) green = %d, want ~255", p[0], p[1], g)
		}
	}
	// The square corner notches and the far exterior stay black.
	for _, p := range [][2]int32{{9, 9}, {54, 9}, {9, 54}, {54, 54}, {2, 2}, {60, 32}} {
		if g := pxAt(data, p[0], p[1])[1]; g > 5 {
			t.Fatalf("exterior (%d,%d) green = %d, want ~0", p[0], p[1], g)
		}
	}
	// Coverage decreases monotonically walking diagonally out through a corner arc.
	g1 := pxAt(data, 14, 14)[1]
	g2 := pxAt(data, 12, 12)[1]
	g3 := pxAt(data, 10, 10)[1]
	if g1 < g2 || g2 < g3 {
		t.Fatalf("corner arc not monotonic: %d %d %d", g1, g2, g3)
	}
}

// TestLiveClipConvexPolyFP: a rotated rect clip rides the convex-poly FP.
func TestLiveClipConvexPolyFP(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	// A 28x28 square rotated 45 degrees about the center: a diamond with vertices ~19.8 from the center along the axes.
	rot := geom.IdentityMatrix()
	rot.SetRotatePivot(45, 32, 32)
	cs := liveClipStack()
	cs.ClipRect(&rot, geom.Rect{Left: 18, Top: 18, Right: 46, Bottom: 46}, gpu.AAYes,
		raster.ClipIntersect)

	src := colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 1}
	fillClipped(sdc, cs, src, geom.Rect{Right: 64, Bottom: 64}, gpu.AANo)

	data := readSDC(t, sdc)
	for _, p := range [][2]int32{{32, 32}, {32, 18}, {18, 32}, {32, 46}, {46, 32}} {
		if g := pxAt(data, p[0], p[1])[1]; g < 250 {
			t.Fatalf("diamond interior (%d,%d) green = %d, want ~255", p[0], p[1], g)
		}
	}
	// The square's corners (outside the diamond) stay black.
	for _, p := range [][2]int32{{18, 18}, {46, 18}, {18, 46}, {46, 46}, {2, 2}} {
		if g := pxAt(data, p[0], p[1])[1]; g > 5 {
			t.Fatalf("diamond exterior (%d,%d) green = %d, want ~0", p[0], p[1], g)
		}
	}
}

// TestLiveClipSWMask: a concave path clip rasterizes on the CPU and samples as a device-space mask texture.
func TestLiveClipSWMask(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	// An arrow with a deep notch: concave, so the analytic lanes cannot express it.
	concave := &path.Path{}
	concave.MoveTo(8, 8)
	concave.LineTo(56, 8)
	concave.LineTo(32, 30)
	concave.LineTo(56, 56)
	concave.LineTo(8, 56)
	concave.Close()

	cs := liveClipStack()
	identity := geom.IdentityMatrix()
	cs.ClipPath(&identity, concave, gpu.AAYes, raster.ClipIntersect)

	src := colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 1}
	fillClipped(sdc, cs, src, geom.Rect{Right: 64, Bottom: 64}, gpu.AANo)

	data := readSDC(t, sdc)
	// Interior probes (well inside the arrow) are green; the notch and exterior are black.
	for _, p := range [][2]int32{{12, 12}, {12, 52}, {12, 32}, {40, 12}, {40, 52}} {
		if g := pxAt(data, p[0], p[1])[1]; g < 250 {
			t.Fatalf("mask interior (%d,%d) green = %d, want ~255", p[0], p[1], g)
		}
	}
	for _, p := range [][2]int32{{50, 32}, {40, 30}, {60, 32}, {4, 4}, {32, 60}} {
		if g := pxAt(data, p[0], p[1])[1]; g > 5 {
			t.Fatalf("mask exterior (%d,%d) green = %d, want ~0", p[0], p[1], g)
		}
	}
	// The mask must be aligned to device space: the boundary at y=8 transitions within one pixel (row 6 black, row 10
	// green at x=20).
	if g := pxAt(data, 20, 6)[1]; g > 5 {
		t.Fatalf("above the top edge green = %d, want ~0", g)
	}
	if g := pxAt(data, 20, 10)[1]; g < 250 {
		t.Fatalf("below the top edge green = %d, want ~255", g)
	}
}

// skipIfNonHardwareGL skips tests whose assertions are known to fail on macOS's non-hardware GL stacks. The Apple
// software renderer (kCGLRendererGenericFloatID, reachable via CANVAS_GLTEST_RENDERER=software) and the Apple
// Paravirtual device (GitHub's macOS CI runners) mis-rasterize the first stencil-enabled pipeline of a context: with a
// byte-identical GL stream to real hardware (draws, vertex data, stencil/scissor/blend/uniform state all verified
// equal), the stencil-write draw covers only a fraction of its pixels until a program switch forces the driver to
// revalidate. This is a driver defect, not a bug in this codebase; real hardware (the release matrix) is unaffected.
func skipIfNonHardwareGL(t *testing.T, dc *gl.DirectContext) {
	t.Helper()
	renderer := dc.Gpu().ContextInfo().DriverInfo.RendererString
	if strings.Contains(renderer, "Paravirtual") || strings.Contains(renderer, "Software") {
		t.Skipf("skipping on non-hardware GL (%q): first stencil-enabled pipeline is "+
			"mis-rasterized by this driver", renderer)
	}
}

// TestLiveClipStencil: five non-AA elements — four ride analytic FPs, the oldest (a rotated half-plane cutter)
// overflows the analytic budget and must clip through the stencil buffer.
func TestLiveClipStencil(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	skipIfNonHardwareGL(t, dc)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	cs := liveClipStack()
	// Oldest element: a rect rotated 15 degrees about the canvas center whose right edge passes just right of the
	// center — this one lands in the stencil.
	cutter := geom.IdentityMatrix()
	cutter.SetRotatePivot(15, 32, 32)
	cs.ClipRect(&cutter, geom.Rect{Left: -20, Top: -20, Right: 34, Bottom: 84}, gpu.AANo,
		raster.ClipIntersect)
	// Four spokes crossing at the center: these ride convex-poly FPs.
	for i := 0; i < 4; i++ {
		m := geom.IdentityMatrix()
		m.SetRotatePivot(float32(45+36*i), 32, 32)
		cs.ClipRect(&m, geom.Rect{Left: 10, Top: 27, Right: 54, Bottom: 37}, gpu.AANo,
			raster.ClipIntersect)
	}

	src := colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 1}
	fillClipped(sdc, cs, src, geom.Rect{Left: 22, Top: 22, Right: 42, Bottom: 42}, gpu.AANo)

	data := readSDC(t, sdc)
	// (28, 32) is inside all five elements: painted.
	if g := pxAt(data, 28, 32)[1]; g < 250 {
		t.Fatalf("inside-all probe green = %d, want ~255", g)
	}
	// (38, 32) is inside the four spokes and the draw, but outside the cutter — only the stencil element removes it.
	if g := pxAt(data, 38, 32)[1]; g > 5 {
		t.Fatalf("stencil-clipped probe green = %d, want ~0 (stencil element ignored?)", g)
	}
	// (32, 50) is outside the spokes entirely (and outside the draw).
	if g := pxAt(data, 32, 50)[1]; g > 5 {
		t.Fatalf("outside probe green = %d, want ~0", g)
	}
}

// TestLiveClipDifference: a difference rect clip rides the inverse-fill analytic FP.
func TestLiveClipDifference(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()

	sdc.Clear([4]float32{0, 0, 0, 1})

	cs := liveClipStack()
	identity := geom.IdentityMatrix()
	cs.ClipRect(&identity, geom.Rect{Left: 24, Top: 24, Right: 40, Bottom: 40}, gpu.AAYes,
		raster.ClipDifference)

	src := colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 1}
	fillClipped(sdc, cs, src, geom.Rect{Left: 10, Top: 10, Right: 54, Bottom: 54}, gpu.AANo)

	data := readSDC(t, sdc)
	// The hole stays black; the surrounding draw paints; outside the draw stays black.
	for _, p := range [][2]int32{{32, 32}, {26, 26}, {38, 38}} {
		if g := pxAt(data, p[0], p[1])[1]; g > 5 {
			t.Fatalf("hole probe (%d,%d) green = %d, want ~0", p[0], p[1], g)
		}
	}
	for _, p := range [][2]int32{{15, 15}, {45, 45}, {32, 15}, {15, 32}} {
		if g := pxAt(data, p[0], p[1])[1]; g < 250 {
			t.Fatalf("ring probe (%d,%d) green = %d, want ~255", p[0], p[1], g)
		}
	}
	for _, p := range [][2]int32{{4, 4}, {60, 60}} {
		if g := pxAt(data, p[0], p[1])[1]; g > 5 {
			t.Fatalf("outside-draw probe (%d,%d) green = %d, want ~0", p[0], p[1], g)
		}
	}
}
