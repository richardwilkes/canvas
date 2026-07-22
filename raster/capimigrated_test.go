// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// TestWrapBytes verifies the sk_surface_make_raster_direct pixel-wrapping contract raster.WrapBytes keeps: a valid
// buffer wraps without copying, and each unsupported configuration (bad dimensions, narrow or non-whole-pixel rowBytes,
// short buffer) returns nil.
func TestWrapBytes(t *testing.T) {
	const w, h = 4, 3
	buf := make([]byte, w*h*4)
	pm := WrapBytes(buf, w, h, w*4)
	if pm == nil {
		t.Fatal("WrapBytes with a valid buffer returned nil")
	}
	if pm.Width != w || pm.Height != h {
		t.Fatalf("wrapped pixmap is %dx%d, want %dx%d", pm.Width, pm.Height, w, h)
	}
	// No copy: a write through the pixmap words lands in the caller's bytes.
	pm.Pix[0] = 0xAABBCCDD
	if buf[0] != 0xDD || buf[3] != 0xAA {
		t.Error("WrapBytes did not alias the caller's buffer")
	}

	if WrapBytes(buf, 0, h, w*4) != nil {
		t.Error("zero width: want nil")
	}
	if WrapBytes(buf, w, -1, w*4) != nil {
		t.Error("negative height: want nil")
	}
	if WrapBytes(buf, w, h, w*4-4) != nil {
		t.Error("rowBytes narrower than the row: want nil")
	}
	if WrapBytes(buf, w, h, w*4+2) != nil {
		t.Error("non-whole-pixel rowBytes: want nil")
	}
	if WrapBytes(buf[:w*h*4-4], w, h, w*4) != nil {
		t.Error("too-small buffer: want nil")
	}
	// A width whose width*4 overflows int32 to a negative value must still be rejected: with a small rowBytes the
	// row cannot actually hold the row, and the guard must not be fooled by the wrapped-around product.
	if WrapBytes(buf, 1<<29, 1, 8) != nil {
		t.Error("width*4 overflow with narrow rowBytes: want nil")
	}
}

// TestAAFillCircleMaskLane AA-fills a circle sized for the additive-mask lane (the path the façade's 40x40 draw suite
// exercised): the interior must reach full coverage — the mask blitter's BlitRect rows — and the integer-aligned
// extremes produce the zero-alpha edge columns its BlitV guard skips.
func TestAAFillCircleMaskLane(t *testing.T) {
	p := path.New()
	p.AddCircle(17, 22, 12, geom.DirectionCW)
	dev := aaFill(t, p, 40, 40)

	if a := alphaAt(dev, 17, 22); a != 255 {
		t.Errorf("circle center alpha %d, want 255", a)
	}
	// A horizontal interior run near the vertical center is fully covered.
	for x := int32(10); x <= 24; x++ {
		if a := alphaAt(dev, x, 22); a != 255 {
			t.Errorf("interior (%d,22) alpha %d, want 255", x, a)
		}
	}
	// Outside the circle stays untouched.
	if a := alphaAt(dev, 2, 2); a != 0 {
		t.Errorf("outside alpha %d, want 0", a)
	}
	// The AA rim carries partial coverage somewhere along the top arc.
	partial := false
	for y := int32(9); y <= 11 && !partial; y++ {
		for x := int32(12); x <= 22; x++ {
			if a := alphaAt(dev, x, y); a > 0 && a < 255 {
				partial = true
				break
			}
		}
	}
	if !partial {
		t.Error("no partial-coverage AA rim found on the circle's top arc")
	}
}

// TestAAFillRoundRectMaskLane AA-fills a round rect with integer-aligned straight sides on the additive-mask lane (the
// façade's 40x40 draw_round_rect): the tall vertical sides emit anti-rect spans whose edge columns carry zero partial
// alpha (the mask blitter's BlitV zero-skip) around a full-coverage interior run (its BlitRect rows).
func TestAAFillRoundRectMaskLane(t *testing.T) {
	p := path.New()
	p.AddRoundRect(geom.RectLTRB(6, 9, 33, 27), 6, 3, geom.DirectionCW)
	dev := aaFill(t, p, 40, 40)

	// Interior rows between the corner radii are fully covered edge to edge.
	for y := int32(13); y < 24; y++ {
		for x := int32(6); x < 33; x++ {
			if a := alphaAt(dev, x, y); a != 255 {
				t.Fatalf("interior (%d,%d) alpha %d, want 255", x, y, a)
			}
		}
	}
	// The integer-aligned sides leave the columns just outside untouched.
	for y := int32(13); y < 24; y++ {
		if a := alphaAt(dev, 5, y); a != 0 {
			t.Fatalf("left of the shape (5,%d) alpha %d, want 0", y, a)
		}
		if a := alphaAt(dev, 33, y); a != 0 {
			t.Fatalf("right of the shape (33,%d) alpha %d, want 0", y, a)
		}
	}
	// The rounded corners carry partial coverage.
	partial := false
	for y := int32(9); y <= 12 && !partial; y++ {
		for x := int32(6); x <= 12; x++ {
			if a := alphaAt(dev, x, y); a > 0 && a < 255 {
				partial = true
				break
			}
		}
	}
	if !partial {
		t.Error("no partial-coverage pixels found at the rounded corner")
	}
}
