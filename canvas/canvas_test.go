// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package canvas

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

const cvW, cvH = 120, 100

func newWhiteCanvas() (*Canvas, *raster.Pixmap) {
	pix := raster.NewPixmap(cvW, cvH)
	c := NewForPixmap(pix)
	c.Clear(0xFFFFFFFF)
	return c, pix
}

func solidPaint(color colorcore.Color, aa bool) *Paint {
	p := NewPaint()
	p.Color = color
	p.AntiAlias = aa
	return p
}

func TestCanvasClearAndDrawColor(t *testing.T) {
	c, pix := newWhiteCanvas()
	for _, v := range pix.Pix {
		if v != 0xFFFFFFFF {
			t.Fatal("clear(white) should fill every pixel")
		}
	}

	// DrawColor with kDst must be a no-op (CheckFastPath kSkipDrawing).
	c.DrawColor(0xFF000000, raster.BlendDst)
	for _, v := range pix.Pix {
		if v != 0xFFFFFFFF {
			t.Fatal("drawColor(kDst) must not draw")
		}
	}

	// Alpha-0 src-over paint draws nothing (nothingToDraw).
	c.DrawColor(0x00FF0000, raster.BlendSrcOver)
	for _, v := range pix.Pix {
		if v != 0xFFFFFFFF {
			t.Fatal("transparent srcover drawColor must not draw")
		}
	}
}

func TestCanvasSaveRestoreMatrix(t *testing.T) {
	c, _ := newWhiteCanvas()

	if c.SaveCount() != 1 {
		t.Fatalf("fresh canvas save count = %d, want 1", c.SaveCount())
	}
	if prev := c.Save(); prev != 1 {
		t.Fatalf("save returned %d, want 1", prev)
	}
	c.Translate(10, 20)
	m := c.TotalMatrix()
	if m.Get(2) != 10 || m.Get(5) != 20 {
		t.Fatalf("translate not applied: %v", m)
	}
	c.Save()
	c.Scale(2, 3)
	c.RestoreToCount(1)
	if c.SaveCount() != 1 {
		t.Fatalf("restoreToCount(1) left save count %d", c.SaveCount())
	}
	m = c.TotalMatrix()
	if !m.IsIdentity() {
		t.Fatalf("restore should recover identity, got %v", m)
	}

	// Restore under-flow is a no-op.
	c.Restore()
	if c.SaveCount() != 1 {
		t.Fatal("restore underflow changed save count")
	}
}

func TestCanvasClipAcrossSaveRestore(t *testing.T) {
	c, _ := newWhiteCanvas()

	c.Save()
	c.ClipRect(geom.RectLTRB(10, 10, 60, 50), raster.ClipIntersect, false)
	if got := c.DeviceClipBounds(); got != geom.IRectLTRB(10, 10, 60, 50) {
		t.Fatalf("clip bounds got %v", got)
	}
	if !c.IsClipRect() || c.IsClipEmpty() {
		t.Fatal("expected non-empty rect clip")
	}
	c.Restore()
	if got := c.DeviceClipBounds(); got != geom.IRectLTRB(0, 0, cvW, cvH) {
		t.Fatalf("restore should recover full clip, got %v", got)
	}

	// The clip is applied through the CTM.
	c.Save()
	c.Translate(20, 0)
	c.ClipRect(geom.RectLTRB(0, 0, 30, 30), raster.ClipIntersect, false)
	if got := c.DeviceClipBounds(); got != geom.IRectLTRB(20, 0, 50, 30) {
		t.Fatalf("translated clip bounds got %v", got)
	}
	c.Restore()

	// Intersecting to nothing empties the clip.
	c.Save()
	c.ClipRect(geom.RectLTRB(0, 0, 10, 10), raster.ClipIntersect, false)
	c.ClipRect(geom.RectLTRB(50, 50, 60, 60), raster.ClipIntersect, false)
	if !c.IsClipEmpty() {
		t.Fatal("disjoint intersect should empty the clip")
	}
	c.Restore()
}

func TestCanvasQuickReject(t *testing.T) {
	c, _ := newWhiteCanvas()
	c.ClipRect(geom.RectLTRB(10, 10, 60, 50), raster.ClipIntersect, true)

	if c.QuickReject(geom.RectLTRB(20, 20, 30, 30)) {
		t.Fatal("rect inside the clip must not be rejected")
	}
	if !c.QuickReject(geom.RectLTRB(80, 80, 100, 100)) {
		t.Fatal("rect far outside the clip must be rejected")
	}
	// The QR bounds are outset by 1 for AA slop, so a rect just touching is not rejected.
	if c.QuickReject(geom.RectLTRB(60.5, 20, 70, 30)) {
		t.Fatal("rect within the 1px AA slop must not be rejected")
	}

	// quickReject accounts for the CTM.
	c.Save()
	c.Translate(-100, 0)
	if c.QuickReject(geom.RectLTRB(110, 20, 140, 30)) {
		t.Fatal("CTM-translated rect lands inside the clip")
	}
	c.Restore()
}

func TestCanvasDrawPathMatchesRasterFill(t *testing.T) {
	// An AA fill through the canvas (CTM + clip + paint) must byte-match driving the raster layer directly with the
	// same device-space path and raster clip.
	c, pix := newWhiteCanvas()

	c.Translate(10.5, 5.25)
	c.Scale(1.5, 1.25)
	c.ClipRect(geom.RectLTRB(5, 5, 100, 90), raster.ClipIntersect, true)
	p := path.New().AddCircle(30, 30, 22.5, geom.DirectionCW)
	c.DrawPath(p, solidPaint(0xFF117733, true))

	want := raster.NewPixmap(cvW, cvH)
	for i := range want.Pix {
		want.Pix[i] = 0xFFFFFFFF
	}
	m := geom.IdentityMatrix()
	m.PreTranslate(10.5, 5.25)
	m.PreScale(1.5, 1.25)
	devPath := &path.Path{}
	p.TransformTo(&m, devPath)
	// the canvas maps the clip rect through the CTM, so the reference must too
	rc := raster.NewRasterClipRect(geom.IRectWH(cvW, cvH))
	rc.OpRect(geom.RectLTRB(5, 5, 100, 90), &m, raster.ClipIntersect, true)
	raster.AntiFillPathRasterClip(devPath, rc, raster.NewSolidBlitter(want, 0xFF117733))

	for i := range pix.Pix {
		if pix.Pix[i] != want.Pix[i] {
			t.Fatalf("pixel %d: canvas %08x, direct %08x", i, pix.Pix[i], want.Pix[i])
		}
	}
}

func TestCanvasDrawPaintThroughClip(t *testing.T) {
	c, pix := newWhiteCanvas()
	c.ClipPath(path.New().AddCircle(50, 50, 30, geom.DirectionCW), raster.ClipIntersect, false)
	c.DrawPaint(solidPaint(0xFF0000FF, false))

	// center inside the clip is blue, corner outside stays white
	if got := pix.Pix[50*cvW+50]; got != 0xFFFF0000 { // word layout R|G<<8|B<<16|A<<24: blue = 0xFFFF0000
		t.Fatalf("center pixel %08x, want blue", got)
	}
	if got := pix.Pix[0]; got != 0xFFFFFFFF {
		t.Fatalf("corner pixel %08x, want white", got)
	}
}

func TestCanvasInverseFillDegenerateBoundsDrawsPaint(t *testing.T) {
	// An inverse-filled path with degenerate bounds fills the whole clip.
	c, pix := newWhiteCanvas()
	p := path.New()
	p.MoveTo(30, 30)
	p.SetFillType(path.FillInverseWinding)
	c.DrawPath(p, solidPaint(0xFF000000, false))
	for _, v := range pix.Pix {
		if v != 0xFF000000 {
			t.Fatal("inverse degenerate path should fill everything")
		}
	}
}

func TestCanvasStrokeStyleDrawsRing(t *testing.T) {
	// The stroker converts a stroked circle to a filled annulus: pixels on the circle's radius are covered; the center
	// and the far corner are not.
	c, pix := newWhiteCanvas()
	paint := solidPaint(0xFF000000, true)
	paint.Style = StyleStroke
	paint.StrokeWidth = 4
	c.DrawPath(path.New().AddCircle(50, 50, 30, geom.DirectionCW), paint)
	rowPixels := int(pix.RowPixels)
	if pix.Pix[50*rowPixels+80] == 0xFFFFFFFF { // (80, 50): on the radius
		t.Fatal("stroke ring pixel not drawn")
	}
	if pix.Pix[50*rowPixels+50] != 0xFFFFFFFF { // center: inside the hole
		t.Fatal("annulus hole was filled")
	}
	if pix.Pix[5*rowPixels+5] != 0xFFFFFFFF { // far corner
		t.Fatal("pixel far outside the ring was drawn")
	}
}

func TestCanvasLocalClipBounds(t *testing.T) {
	c, _ := newWhiteCanvas()
	c.Translate(10, 10)
	c.ClipRect(geom.RectLTRB(0, 0, 50, 40), raster.ClipIntersect, false)
	lb := c.LocalClipBounds()
	// device clip is (10,10,60,50); local = inverse-translated and outset by 1
	want := geom.RectLTRB(-1, -1, 51, 41)
	if lb != want {
		t.Fatalf("local clip bounds got %v want %v", lb, want)
	}
}
