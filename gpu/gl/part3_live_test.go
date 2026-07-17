// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the rrect/oval/arc/stroke-rect/texture op family: they render end-to-end (interior/exterior
// probes), the rrect-clip draw reduction, MSAA render passes on an offscreen multisampled target, and the GPU canvas
// device rendering a full scene compared against the CPU canvas (the Go-GPU vs Go-CPU self-consistency lane). Skips
// when no GL context is available.

package gl_test

import (
	"testing"
	"unsafe"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stroke"
)

// probeAt asserts pixel equality within tol.
func probeAt(t *testing.T, data []byte, rowBytes int, x, y int32, want [4]byte, tol int, tag string) {
	t.Helper()
	off := int(y)*rowBytes + int(x)*4
	for c := range 4 {
		d := int(data[off+c]) - int(want[c])
		if d < -tol || d > tol {
			t.Fatalf("%s: pixel (%d,%d) = [%d %d %d %d], want %v (±%d)", tag, x, y,
				data[off], data[off+1], data[off+2], data[off+3], want, tol)
		}
	}
}

func TestLiveFillRRectOp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	// Elliptical corners select FillRRectOp (circular ones go to CircularRRectOp, probed below).
	rrect := geom.MakeRRect(geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 48}, 10, 14)
	identity := geom.IdentityMatrix()
	sdc.DrawRRect(nil, livePaint(colorcore.PMColor4f{R: 1, G: 0, B: 0, A: 1}), gpu.AAYes,
		&identity, rrect, nil)

	data := readSDC(t, sdc)
	rb := 64 * 4
	red := [4]byte{255, 0, 0, 255}
	white := [4]byte{255, 255, 255, 255}
	probeAt(t, data, rb, 32, 28, red, 1, "rrect center")
	probeAt(t, data, rb, 10, 28, red, 1, "rrect left edge interior")
	probeAt(t, data, rb, 9, 9, white, 1, "rrect corner exterior")
	probeAt(t, data, rb, 58, 50, white, 1, "rrect outside")
	// Inside the corner curve: (8+10*(1-cos45), 8+14*(1-sin45)) ≈ (10.9, 12.1); probe inward.
	probeAt(t, data, rb, 14, 15, red, 1, "rrect corner interior")
}

func TestLiveCircularRRectOp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	rrect := geom.MakeRRect(geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 48}, 10, 10)
	identity := geom.IdentityMatrix()
	sdc.DrawRRect(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0, B: 1, A: 1}), gpu.AAYes,
		&identity, rrect, nil)

	data := readSDC(t, sdc)
	rb := 64 * 4
	blue := [4]byte{0, 0, 255, 255}
	white := [4]byte{255, 255, 255, 255}
	probeAt(t, data, rb, 32, 28, blue, 1, "circular rrect center")
	probeAt(t, data, rb, 9, 9, white, 1, "circular rrect corner exterior")
	probeAt(t, data, rb, 14, 14, blue, 1, "circular rrect corner interior")
}

// TestLiveStrokedCircularRRectOp is a regression test for a stroked circular round-rect overrunning its index buffer.
// rrectTypeToIndices returned the full nine-patch (54 indices, including the center quad) for the stroke case, but
// rrectTypeToIndexCount sizes the stroke index buffer for 48 (center quad dropped), so OnPrepare wrote 6 indices past
// the end of the allocation and panicked on the first drawn frame. It also confirms strokes leave the interior unfilled
// (the whole point of dropping the center quad).
func TestLiveStrokedCircularRRectOp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	// Circular corners (RadiusX == RadiusY) with a similarity matrix select CircularRRectOp; a plain stroke narrower
	// than the corner radius takes the rrectGeomStroke path.
	rrect := geom.MakeRRect(geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 48}, 10, 10)
	identity := geom.IdentityMatrix()
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)
	sdc.DrawRRect(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0, B: 1, A: 1}), gpu.AAYes,
		&identity, rrect, &rec)

	data := readSDC(t, sdc)
	rb := 64 * 4
	blue := [4]byte{0, 0, 255, 255}
	white := [4]byte{255, 255, 255, 255}
	// The 4px-wide stroke straddles each straight edge (x=32 and y=28 are away from the corners).
	probeAt(t, data, rb, 32, 8, blue, 2, "stroked rrect top edge")
	probeAt(t, data, rb, 8, 28, blue, 2, "stroked rrect left edge")
	// Interior is unfilled (center quad dropped) and the far exterior is clear.
	probeAt(t, data, rb, 32, 28, white, 1, "stroked rrect interior (unfilled)")
	probeAt(t, data, rb, 2, 2, white, 1, "stroked rrect exterior")
}

// TestLiveOverstrokedCircularRRectOp exercises the rrectGeomOverstroke path (stroke wider than twice the corner
// radius), which was correct but had no live coverage.
func TestLiveOverstrokedCircularRRectOp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	// Centered so the stroke's AA-bloated bounds leave clear exterior at the surface corners.
	rrect := geom.MakeRRect(geom.Rect{Left: 20, Top: 20, Right: 44, Bottom: 44}, 6, 6)
	identity := geom.IdentityMatrix()
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(14, false) // width > 2*radius -> overstroke
	sdc.DrawRRect(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0, B: 1, A: 1}), gpu.AAYes,
		&identity, rrect, &rec)

	data := readSDC(t, sdc)
	rb := 64 * 4
	blue := [4]byte{0, 0, 255, 255}
	white := [4]byte{255, 255, 255, 255}
	probeAt(t, data, rb, 32, 20, blue, 2, "overstroked rrect top edge")
	probeAt(t, data, rb, 2, 2, white, 1, "overstroked rrect exterior")
}

func TestLiveCircleAndArcOps(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	identity := geom.IdentityMatrix()

	// Stroked circle (CircleOp): center (20,20), radius 12, stroke width 4.
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)
	sdc.DrawOval(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0.5, B: 0, A: 1}), gpu.AAYes,
		&identity, geom.Rect{Left: 8, Top: 8, Right: 32, Bottom: 32}, &rec)

	// Arc wedge (CircleOp with clip planes): quarter wedge at center (46,46) radius 14, sweeping from 0° to 90° (the
	// +x/+y quadrant).
	fill := stroke.NewStrokeRec(stroke.InitStyleFill)
	sdc.DrawArc(nil, livePaint(colorcore.PMColor4f{R: 0.5, G: 0, B: 0.5, A: 1}), gpu.AAYes,
		&identity, geom.Rect{Left: 32, Top: 32, Right: 60, Bottom: 60}, 0, 90, true, &fill)

	data := readSDC(t, sdc)
	rb := 64 * 4
	green := [4]byte{0, 128, 0, 255}
	purple := [4]byte{128, 0, 128, 255}
	white := [4]byte{255, 255, 255, 255}

	// Circle ring probes: on the ring at (20+12, 20) = (32, 20); interior center is empty.
	probeAt(t, data, rb, 31, 20, green, 2, "circle stroke ring")
	probeAt(t, data, rb, 20, 20, white, 1, "circle stroke hole")
	probeAt(t, data, rb, 20, 5, white, 1, "circle exterior")

	// Wedge: inside the 0..90° quadrant at (52,52); outside the wedge at (40,40)±... the wedge occupies x,y >= center;
	// (40, 52) is 180°-ish → outside.
	probeAt(t, data, rb, 52, 52, purple, 2, "arc wedge interior")
	probeAt(t, data, rb, 40, 52, white, 1, "arc wedge exterior (wrong quadrant)")
	probeAt(t, data, rb, 59, 59, white, 1, "arc wedge exterior (beyond radius)")
}

func TestLiveEllipseOp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	identity := geom.IdentityMatrix()
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)
	// Non-circular stroked oval under an axis-aligned matrix → EllipseOp.
	sdc.DrawOval(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0, B: 0, A: 1}), gpu.AAYes,
		&identity, geom.Rect{Left: 6, Top: 16, Right: 58, Bottom: 48}, &rec)

	data := readSDC(t, sdc)
	rb := 64 * 4
	black := [4]byte{0, 0, 0, 255}
	white := [4]byte{255, 255, 255, 255}
	// Outer/inner x-radii at the midline are 28/24 (band x ∈ [4, 8]); the y band is [14, 18].
	probeAt(t, data, rb, 6, 32, black, 2, "ellipse stroke left")
	probeAt(t, data, rb, 32, 16, black, 2, "ellipse stroke top")
	probeAt(t, data, rb, 32, 32, white, 1, "ellipse hole")
	probeAt(t, data, rb, 8, 18, white, 1, "ellipse exterior")
}

func TestLiveStrokeRectOps(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	identity := geom.IdentityMatrix()
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)

	// AA stroked rect centered on (8..28)² with width 4: frame spans 6..30 per side.
	sdc.DrawRect(nil, livePaint(colorcore.PMColor4f{R: 1, G: 0, B: 0, A: 1}), gpu.AAYes,
		&identity, geom.Rect{Left: 8, Top: 8, Right: 28, Bottom: 28}, &rec)
	// Non-AA stroked rect.
	sdc.DrawRect(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0, B: 1, A: 1}), gpu.AANo,
		&identity, geom.Rect{Left: 38, Top: 38, Right: 58, Bottom: 58}, &rec)

	data := readSDC(t, sdc)
	rb := 64 * 4
	red := [4]byte{255, 0, 0, 255}
	blue := [4]byte{0, 0, 255, 255}
	white := [4]byte{255, 255, 255, 255}

	probeAt(t, data, rb, 8, 18, red, 1, "AA stroke frame left")
	probeAt(t, data, rb, 18, 8, red, 1, "AA stroke frame top")
	probeAt(t, data, rb, 18, 18, white, 1, "AA stroke hole")
	probeAt(t, data, rb, 3, 3, white, 1, "AA stroke exterior")

	probeAt(t, data, rb, 38, 48, blue, 1, "non-AA stroke frame left")
	probeAt(t, data, rb, 48, 48, white, 1, "non-AA stroke hole")
	probeAt(t, data, rb, 33, 33, white, 1, "non-AA stroke exterior")
}

func TestLiveTextureOp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	// A 2x2 texture: red, green / blue, black.
	pixels := []byte{
		255, 0, 0, 255, 0, 255, 0, 255,
		0, 0, 255, 255, 0, 0, 0, 255,
	}
	var key gpu.UniqueKey
	builder := gpu.UniqueKeyBuilder(&key, gpu.GenerateUniqueKeyDomain(), 1, "LiveTexOp")
	builder.Slice()[0] = 1
	builder.Finish()
	view := gl.FindOrCreatePixelsProxyView(dc, &key, geom.ISize{Width: 2, Height: 2},
		gpu.ColorTypeRGBA8888, pixels, 8, "LiveTexOp")
	if view.Proxy() == nil {
		t.Fatal("FindOrCreatePixelsProxyView failed")
	}

	identity := geom.IdentityMatrix()
	sdc.DrawTexture(nil, view, gpu.AlphaTypePremul, gpu.FilterModeNearest, gpu.MipmapModeNone,
		raster.BlendSrcOver, colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1},
		geom.Rect{Left: 0, Top: 0, Right: 2, Bottom: 2},
		geom.Rect{Left: 8, Top: 8, Right: 40, Bottom: 40}, gpu.QuadAAFlagsNone,
		gl.SrcRectConstraintFast, &identity)

	data := readSDC(t, sdc)
	rb := 64 * 4
	probeAt(t, data, rb, 12, 12, [4]byte{255, 0, 0, 255}, 1, "texel (0,0)")
	probeAt(t, data, rb, 36, 12, [4]byte{0, 255, 0, 255}, 1, "texel (1,0)")
	probeAt(t, data, rb, 12, 36, [4]byte{0, 0, 255, 255}, 1, "texel (0,1)")
	probeAt(t, data, rb, 36, 36, [4]byte{0, 0, 0, 255}, 1, "texel (1,1)")
	probeAt(t, data, rb, 50, 50, [4]byte{255, 255, 255, 255}, 1, "outside dst")
}

func TestLiveRRectClipReduction(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := newLiveDrawSDC(t, dc, 64, 64)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	// An axis-aligned rounded clip plus a covering constant-color fill reduces to a direct rrect draw; the corners must
	// be clipped out.
	identity := geom.IdentityMatrix()
	cs := gl.NewClipStack(geom.IRectWH(64, 64), false)
	cs.ClipRRect(&identity,
		geom.MakeRRect(geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 56}, 12, 12), gpu.AAYes,
		raster.ClipIntersect)
	sdc.FillRectToRect(cs, livePaint(colorcore.PMColor4f{R: 0, G: 0, B: 1, A: 1}), gpu.AAYes,
		&identity, geom.Rect{Right: 64, Bottom: 64}, geom.Rect{Right: 64, Bottom: 64})

	data := readSDC(t, sdc)
	rb := 64 * 4
	blue := [4]byte{0, 0, 255, 255}
	white := [4]byte{255, 255, 255, 255}
	probeAt(t, data, rb, 32, 32, blue, 1, "clipped fill interior")
	probeAt(t, data, rb, 9, 9, white, 1, "clipped fill corner exterior")
	probeAt(t, data, rb, 16, 16, blue, 1, "clipped fill corner interior")
	probeAt(t, data, rb, 4, 32, white, 1, "outside clip left")
}

func TestLiveMSAARenderPass(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	sdc := gl.MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: 64, Height: 64}, gpu.BackingFitExact, 4, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "live-msaa")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext(msaa) failed")
	}
	defer sdc.Release()
	if sdc.NumSamples() <= 1 {
		t.Skip("driver does not support MSAA render targets")
	}

	sdc.Clear([4]float32{1, 1, 1, 1})
	identity := geom.IdentityMatrix()

	// A non-AA rect with a half-pixel left edge: MSAA sampling should land the boundary column at partial coverage
	// after resolve (exact fraction depends on the driver's sample pattern), while interior and exterior stay exact.
	sdc.FillRectToRect(nil, livePaint(colorcore.PMColor4f{R: 1, G: 0, B: 0, A: 1}), gpu.AANo,
		&identity, geom.Rect{Left: 10.5, Top: 8, Right: 30, Bottom: 30},
		geom.Rect{Left: 10.5, Top: 8, Right: 30, Bottom: 30})

	// A filled rrect exercises FillRRectOp's kMSAAEnabled wide-ramp lane on the MSAA surface.
	sdc.DrawRRect(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0, B: 1, A: 1}), gpu.AAYes,
		&identity, geom.MakeRRect(geom.Rect{Left: 34, Top: 34, Right: 60, Bottom: 58}, 6, 8),
		nil)

	data := readSDC(t, sdc)
	rb := 64 * 4
	probeAt(t, data, rb, 20, 20, [4]byte{255, 0, 0, 255}, 1, "msaa rect interior")
	probeAt(t, data, rb, 9, 20, [4]byte{255, 255, 255, 255}, 1, "msaa rect exterior")
	// The boundary column must be a genuine partial mix (not 0% or 100%).
	off := 20*rb + 10*4
	if g := data[off+1]; g < 32 || g > 224 {
		t.Fatalf("msaa half-covered pixel G = %d, want a partial mix", g)
	}
	probeAt(t, data, rb, 46, 46, [4]byte{0, 0, 255, 255}, 1, "msaa rrect interior")
	// The MSAA lane widens the coverage ramp to a full pixel, so stay a pixel clear of the corner curve when probing
	// the exterior.
	probeAt(t, data, rb, 33, 33, [4]byte{255, 255, 255, 255}, 1, "msaa rrect corner exterior")
}

// sceneCanvas draws the shared device-comparison scene.
func sceneCanvas(c *canvas.Canvas, img *imagecore.Image) {
	white := canvas.NewPaint()
	white.Color = 0xFFFFFFFF
	c.DrawPaint(white)

	// Fractional AA clip.
	c.Save()
	c.ClipRect(geom.Rect{Left: 2.5, Top: 2.25, Right: 125.5, Bottom: 125.75},
		raster.ClipIntersect, true)

	// Translucent blue fill rrect (elliptical corners → FillRRectOp on GPU).
	blue := canvas.NewPaint()
	blue.Color = 0x800000FF
	blue.AntiAlias = true
	c.DrawRoundRect(geom.Rect{Left: 6, Top: 6, Right: 60, Bottom: 44}, 8, 11, blue)

	// Red stroked rect, AA.
	red := canvas.NewPaint()
	red.Color = 0xFFCC2200
	red.AntiAlias = true
	red.Style = canvas.StyleStroke
	red.StrokeWidth = 4
	c.DrawRect(geom.Rect{Left: 70, Top: 8, Right: 118, Bottom: 40}, red)

	// Green filled circle.
	green := canvas.NewPaint()
	green.Color = 0xFF117722
	green.AntiAlias = true
	c.DrawCircle(30, 80, 20, green)

	// Concave star path (the SW-mask lane on GPU).
	star := &path.Path{}
	star.MoveTo(96, 56)
	star.LineTo(104, 76)
	star.LineTo(124, 76)
	star.LineTo(108, 88)
	star.LineTo(114, 108)
	star.LineTo(96, 96)
	star.LineTo(78, 108)
	star.LineTo(84, 88)
	star.LineTo(68, 76)
	star.LineTo(88, 76)
	star.Close()
	purple := canvas.NewPaint()
	purple.Color = 0xFF663399
	purple.AntiAlias = true
	c.DrawPath(star, purple)

	// Scaled image draw, nearest.
	c.DrawImageRect(img, geom.Rect{Right: 4, Bottom: 4},
		geom.Rect{Left: 8, Top: 104, Right: 40, Bottom: 120},
		shaders.SamplingOptions{Filter: shaders.FilterNearest}, nil, canvas.ConstraintStrict)

	// A stroked diagonal line, butt cap, AA.
	line := canvas.NewPaint()
	line.Color = 0xFF008080
	line.AntiAlias = true
	line.Style = canvas.StyleStroke
	line.StrokeWidth = 3
	c.DrawLine(48, 104, 62, 122, line)

	// saveLayer alpha with an orange oval inside.
	c.SaveLayerAlpha(nil, 128)
	orange := canvas.NewPaint()
	orange.Color = 0xFFFF8800
	orange.AntiAlias = true
	c.DrawOval(geom.Rect{Left: 44, Top: 52, Right: 76, Bottom: 100}, orange)
	c.Restore()

	c.Restore()
}

// newCheckerImage builds a 4x4 checkerboard N32 image.
func newCheckerImage(t testing.TB) *imagecore.Image {
	t.Helper()
	pm := raster.NewPixmap(4, 4)
	for y := int32(0); y < 4; y++ {
		for x := int32(0); x < 4; x++ {
			if (x+y)%2 == 0 {
				pm.Pix[y*pm.RowPixels+x] = 0xFF0000FF // opaque red (RGBA word, R in the low byte)
			} else {
				pm.Pix[y*pm.RowPixels+x] = 0xFFFFFFFF
			}
		}
	}
	img := imagecore.Copy(pm)
	if img == nil {
		t.Fatal("checker image creation failed")
	}
	return img
}

func TestLiveGLDeviceScene(t *testing.T) {
	_, dc := newLiveDirectContext(t)

	const w, h = 128, 128

	img := newCheckerImage(t)

	// CPU reference.
	cpuPix := raster.NewPixmap(w, h)
	cpuCanvas := canvas.NewForPixmap(cpuPix)
	sceneCanvas(cpuCanvas, img)

	// GPU device.
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	dev := gl.NewDevice(sdc)
	gpuCanvas := canvas.New(dev)
	sceneCanvas(gpuCanvas, img)

	gpuData := readSDC(t, sdc)
	cpuData := unsafe.Slice((*byte)(unsafe.Pointer(&cpuPix.Pix[0])), len(cpuPix.Pix)*4)

	// Go-GPU vs Go-CPU self-consistency: interiors must agree tightly; AA edge pixels may differ between the CPU's
	// analytic coverage and the GPU's shader/ramp coverage. Budget: ≤1.5% of pixels may exceed the per-channel
	// tolerance, and no pixel may differ by more than 96.
	const tol = 9
	const maxDelta = 96
	budget := w * h * 15 / 1000
	over := 0
	worst := 0
	rb := w * 4
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := y*rb + x*4
			pixOver := false
			for c := range 4 {
				d := int(gpuData[off+c]) - int(cpuData[off+c])
				if d < 0 {
					d = -d
				}
				if d > tol {
					pixOver = true
				}
				if d > worst {
					worst = d
				}
				if d > maxDelta {
					t.Fatalf("pixel (%d,%d) channel %d: gpu %d vs cpu %d exceeds the max delta",
						x, y, c, gpuData[off+c], cpuData[off+c])
				}
			}
			if pixOver {
				over++
			}
		}
	}
	if over > budget {
		t.Fatalf("%d pixels beyond ±%d (budget %d, worst delta %d)", over, tol, budget, worst)
	}
	t.Logf("device scene: %d/%d pixels beyond ±%d (budget %d), worst delta %d",
		over, w*h, tol, budget, worst)
}
