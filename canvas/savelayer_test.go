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
	"reflect"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// noLayerDevice is a device whose CreateDevice always fails, which is what the GPU device does for an abandoned
// context, an unknown format, or dimensions past Caps.MaxRenderTargetSize.
type noLayerDevice struct {
	*BitmapDevice
	creates int
}

func (d *noLayerDevice) CreateDevice(_, _ int32, _ *Paint) Device {
	d.creates++
	return nil
}

// TestSaveLayerSurvivesFailedDeviceCreation: a layer whose backing device cannot be allocated must squash the layer's
// draws and unwind normally rather than calling ClipRect/SetDeviceCoordinateSystem on a nil Device.
func TestSaveLayerSurvivesFailedDeviceCreation(t *testing.T) {
	for _, tc := range []struct {
		bounds *geom.Rect
		paint  *Paint
		name   string
	}{
		{name: "plain"},
		{name: "bounded", bounds: &geom.Rect{Left: 2, Top: 2, Right: 10, Bottom: 10}},
		{name: "alpha", paint: func() *Paint { p := NewPaint(); p.Color = p.Color.WithAlpha(128); return p }()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pix := raster.NewPixmap(16, 16)
			dev := &noLayerDevice{BitmapDevice: NewBitmapDevice(pix)}
			c := New(dev)
			c.Clear(0xFFFFFFFF)
			before := append([]uint32(nil), pix.Pix...)

			count := c.SaveCount()
			c.SaveLayer(tc.bounds, tc.paint)
			if dev.creates != 1 {
				t.Fatalf("CreateDevice called %d times, want 1", dev.creates)
			}
			// Every draw inside the failed layer is squashed.
			c.DrawRect(geom.RectWH(16, 16), solidPaint(0xFFFF0000, false))
			c.DrawOval(geom.RectWH(16, 16), solidPaint(0xFF00FF00, true))
			c.Restore()

			if c.SaveCount() != count {
				t.Errorf("save count = %d after the restore, want %d", c.SaveCount(), count)
			}
			for i := range pix.Pix {
				if pix.Pix[i] != before[i] {
					t.Fatalf("pixel %d changed (%08x -> %08x); a failed layer must draw nothing",
						i, before[i], pix.Pix[i])
				}
			}
			// The clip must be back to wide open once the layer is unwound.
			if !dev.IsClipWideOpen() {
				t.Error("the aborted layer's empty clip outlived its restore")
			}
			c.DrawRect(geom.RectWH(16, 16), solidPaint(0xFF0000FF, false))
			if pix.Pix[0] == before[0] {
				t.Error("draws after the restore must land again")
			}
		})
	}
}

// TestSaveLayerBoundsHintSkippedForTransparentBlackBlend: SaveLayer's bounds argument is only a content hint when the
// restore leaves everything outside it alone. A restore blend whose dst coefficient is not One/ISA/ISC (here BlendSrc)
// writes transparent black across the whole clip, so the hint must be dropped and the layer sized to the clip instead.
func TestSaveLayerBoundsHintSkippedForTransparentBlackBlend(t *testing.T) {
	// Pixmap words are B<<16|G<<8|R, so an opaque red layer lands as FF0000FF.
	const red, white = uint32(0xFF0000FF), uint32(0xFFFFFFFF)
	for _, tc := range []struct {
		name        string
		mode        raster.BlendMode
		wantInside  uint32
		wantOutside uint32
	}{
		{name: "src-floods-the-clip", mode: raster.BlendSrc, wantInside: red, wantOutside: 0},
		{name: "src-in-floods-the-clip", mode: raster.BlendSrcIn, wantInside: red, wantOutside: 0},
		{name: "src-over-keeps-the-hint", mode: raster.BlendSrcOver, wantInside: red, wantOutside: white},
		// Xor over an opaque destination zeroes what the layer covers, but leaves everything the layer does not.
		{name: "xor-keeps-the-hint", mode: raster.BlendXor, wantInside: 0, wantOutside: white},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pix := raster.NewPixmap(16, 16)
			c := NewForPixmap(pix)
			c.Clear(0xFFFFFFFF)

			lp := NewPaint()
			lp.BlendMode = tc.mode
			bounds := geom.RectWH(8, 8)
			c.SaveLayer(&bounds, lp)
			c.DrawRect(geom.RectWH(8, 8), solidPaint(0xFFFF0000, false))
			c.Restore()

			// Inside the hint the layer's own content lands either way.
			if got := pix.Pix[4*16+4]; got != tc.wantInside {
				t.Errorf("inside the layer bounds = %08x, want %08x", got, tc.wantInside)
			}
			// Outside it, only a transparent-black-affecting blend reaches.
			if got := pix.Pix[12*16+12]; got != tc.wantOutside {
				t.Errorf("outside the layer bounds = %08x, want %08x", got, tc.wantOutside)
			}
		})
	}
}

// TestDeviceInterfaceDropsUncalledClipSurface guards the two clip entry points that no code ever reached through the
// Device interface. Re-adding either one puts a method on the interface (and on all three implementations) that nothing
// dispatches to, which is what made them read as live capability.
func TestDeviceInterfaceDropsUncalledClipSurface(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeOf((*Device)(nil)).Elem(),
		reflect.TypeOf((*BitmapDevice)(nil)),
	} {
		for _, name := range []string{"ReplaceClip", "IsClipAntiAliased"} {
			if _, ok := typ.MethodByName(name); ok {
				t.Errorf("%s declares %s, which has no caller — add one or drop the method", typ, name)
			}
		}
	}
}
