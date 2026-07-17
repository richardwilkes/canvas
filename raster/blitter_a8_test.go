// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// TestFillPathToMaskMatchesBlackFill: the A8 coverage mask must equal the alpha channel of an opaque black AA fill of
// the same path (the color pipeline is the identity there).
func TestFillPathToMaskMatchesBlackFill(t *testing.T) {
	p := path.New()
	p.AddCircle(20, 18, 14.3, geom.DirectionCW)
	p.AddRect(geom.RectLTRB(4.5, 3.25, 12.5, 30.75), geom.DirectionCW)

	bounds := geom.IRectLTRB(0, 0, 40, 36)
	for _, aa := range []bool{true, false} {
		mask := FillPathToMask(p, bounds, aa)
		dev := NewPixmap(40, 36)
		if aa {
			AntiFillPath(p, bounds, NewSolidBlitter(dev, colorcore.Color(0xFF000000)))
		} else {
			FillPath(p, bounds, NewSolidBlitter(dev, colorcore.Color(0xFF000000)))
		}
		for y := int32(0); y < 36; y++ {
			for x := int32(0); x < 40; x++ {
				m := mask.Image[mask.addr8(x, y)]
				d := uint8(dev.Pix[dev.addr(x, y)] >> 24)
				if m != d {
					t.Fatalf("aa=%v (%d,%d): mask %d, fill alpha %d", aa, x, y, m, d)
				}
			}
		}
	}
}

// TestFillPathToMaskOffsetBounds: mask bounds needn't start at the origin.
func TestFillPathToMaskOffsetBounds(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(10, 10, 20, 20), geom.DirectionCW)
	mask := FillPathToMask(p, geom.IRectLTRB(8, 8, 24, 24), true)
	if got := mask.Image[mask.addr8(15, 15)]; got != 0xFF {
		t.Fatalf("interior %d", got)
	}
	if got := mask.Image[mask.addr8(9, 9)]; got != 0 {
		t.Fatalf("exterior %d", got)
	}
}
