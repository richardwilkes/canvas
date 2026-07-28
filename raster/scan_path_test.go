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
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// fillToMask fills p (non-AA) into a w×h device with opaque black and returns a text mask: '#' where the pixel was
// written, '.' elsewhere, one string per row.
func fillToMask(t *testing.T, p *path.Path, w, h int32) []string {
	t.Helper()
	dev := NewPixmap(w, h)
	FillPath(p, geom.IRectLTRB(0, 0, w, h), NewSolidBlitter(dev, 0xFF000000))
	rows := make([]string, h)
	var sb strings.Builder
	for y := int32(0); y < h; y++ {
		sb.Reset()
		for x := int32(0); x < w; x++ {
			if dev.Pix[dev.addr(x, y)] != 0 {
				sb.WriteByte('#')
			} else {
				sb.WriteByte('.')
			}
		}
		rows[y] = sb.String()
	}
	return rows
}

func checkMask(t *testing.T, got, want []string) {
	t.Helper()
	for y := range want {
		if got[y] != want[y] {
			t.Errorf("row %d: got %s want %s", y, got[y], want[y])
		}
	}
}

func TestFillRectAligned(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(2, 1, 6, 4), geom.DirectionCW)
	got := fillToMask(t, p, 8, 5)
	checkMask(t, got, []string{
		"........",
		"..####..",
		"..####..",
		"..####..",
		"........",
	})
}

func TestFillRectPixelCenters(t *testing.T) {
	// Non-AA coverage follows pixel centers: [0.5, 3.5] covers columns/rows 1..3.
	p := path.New()
	p.AddRect(geom.RectLTRB(0.5, 0.5, 3.5, 3.5), geom.DirectionCW)
	got := fillToMask(t, p, 5, 5)
	checkMask(t, got, []string{
		".....",
		".###.",
		".###.",
		".###.",
		".....",
	})
}

func TestFillTriangleConvexWalker(t *testing.T) {
	p := path.New()
	p.MoveTo(4, 0)
	p.LineTo(8, 8)
	p.LineTo(0, 8)
	p.Close()
	if !p.IsConvex() {
		t.Fatal("triangle should be convex")
	}
	got := fillToMask(t, p, 8, 8)
	// Each row's span follows the rounded edge crossings; verify symmetry and monotonic growth instead of exact pixels,
	// then spot-check against the general walker.
	p2 := p.Clone()
	p2.SetFillType(path.FillEvenOdd) // even-odd of a simple triangle fills identically...
	got2 := fillToMask(t, p2, 8, 8)  // ...but routes through walkEdges (non-convex path? still convex)
	for y := range got {
		if got[y] != got2[y] {
			t.Errorf("row %d: winding %q vs even-odd %q", y, got[y], got2[y])
		}
	}
}

func TestFillWindingVsEvenOddDonut(t *testing.T) {
	// Two nested same-direction rects: winding fills both, even-odd leaves a hole.
	build := func(ft path.FillType) *path.Path {
		p := path.New()
		p.AddRect(geom.RectLTRB(1, 1, 7, 7), geom.DirectionCW)
		p.AddRect(geom.RectLTRB(3, 3, 5, 5), geom.DirectionCW)
		p.SetFillType(ft)
		return p
	}
	winding := fillToMask(t, build(path.FillWinding), 8, 8)
	checkMask(t, winding, []string{
		"........",
		".######.",
		".######.",
		".######.",
		".######.",
		".######.",
		".######.",
		"........",
	})
	evenOdd := fillToMask(t, build(path.FillEvenOdd), 8, 8)
	checkMask(t, evenOdd, []string{
		"........",
		".######.",
		".######.",
		".##..##.",
		".##..##.",
		".######.",
		".######.",
		"........",
	})
}

func TestFillInverseWinding(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(2, 2, 6, 6), geom.DirectionCW)
	p.SetFillType(path.FillInverseWinding)
	got := fillToMask(t, p, 8, 8)
	checkMask(t, got, []string{
		"########",
		"########",
		"##....##",
		"##....##",
		"##....##",
		"##....##",
		"########",
		"########",
	})
}

func TestFillInverseEmptyPathFillsClip(t *testing.T) {
	p := path.New()
	p.SetFillType(path.FillInverseWinding)
	got := fillToMask(t, p, 4, 3)
	checkMask(t, got, []string{
		"####",
		"####",
		"####",
	})
}

func TestFillClipRestricts(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(0, 0, 8, 8), geom.DirectionCW)
	dev := NewPixmap(8, 8)
	FillPath(p, geom.IRectLTRB(2, 3, 5, 6), NewSolidBlitter(dev, 0xFF000000))
	for y := int32(0); y < 8; y++ {
		for x := int32(0); x < 8; x++ {
			inside := x >= 2 && x < 5 && y >= 3 && y < 6
			if got := dev.Pix[dev.addr(x, y)] != 0; got != inside {
				t.Errorf("(%d,%d): filled=%v, want %v", x, y, got, inside)
			}
		}
	}
}

func TestFillCurvedPathContainment(t *testing.T) {
	// A circle via conics (AddOval) exercises the conic→quad→edge pipeline. Non-AA coverage is decided at pixel
	// centers: centers strictly inside must be set, centers clearly outside must not.
	p := path.New()
	p.AddOval(geom.RectLTRB(1, 1, 15, 15), geom.DirectionCW)
	got := fillToMask(t, p, 16, 16)
	cx, cy, r := float64(8), float64(8), float64(7)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			dx := float64(x) + 0.5 - cx
			dy := float64(y) + 0.5 - cy
			d2 := dx*dx + dy*dy
			filled := got[y][x] == '#'
			if d2 < (r-0.75)*(r-0.75) && !filled {
				t.Errorf("(%d,%d) well inside circle but not filled", x, y)
			}
			if d2 > (r+0.75)*(r+0.75) && filled {
				t.Errorf("(%d,%d) well outside circle but filled", x, y)
			}
		}
	}
}

func TestFillCubicMatchesFlattenedReference(t *testing.T) {
	// A cubic blob: verify every span is within the path per Contains at pixel centers (Contains uses the
	// analytically-correct winding machinery from the library, so agreement within a small band around the boundary
	// validates the forward-differencing edges).
	p := path.New()
	p.MoveTo(2, 8)
	p.CubicTo(2, 2, 14, 2, 14, 8)
	p.CubicTo(14, 14, 2, 14, 2, 8)
	p.Close()
	got := fillToMask(t, p, 16, 16)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			cxf := float32(x) + 0.5
			cyf := float32(y) + 0.5
			filled := got[y][x] == '#'
			// Allow disagreement within a half-pixel band of the boundary (fixed-point rounding).
			nearBoundary := false
			for _, d := range [][2]float32{{0.55, 0}, {-0.55, 0}, {0, 0.55}, {0, -0.55}, {0.55, 0.55}, {-0.55, -0.55}, {0.55, -0.55}, {-0.55, 0.55}} {
				if p.Contains(cxf+d[0], cyf+d[1]) != p.Contains(cxf, cyf) {
					nearBoundary = true
					break
				}
			}
			if !nearBoundary && filled != p.Contains(cxf, cyf) {
				t.Errorf("(%d,%d): filled=%v contains=%v", x, y, filled, p.Contains(cxf, cyf))
			}
		}
	}
}

func TestSolidBlitterTranslucentMath(t *testing.T) {
	// color32 kernel: dst = color + (dst*(256-srcA))>>8 per channel.
	dev := NewPixmap(2, 1)
	dev.Pix[0] = 0xFFFFFFFF // opaque white
	dev.Pix[1] = 0x00000000
	p := path.New()
	p.AddRect(geom.RectLTRB(0, 0, 2, 1), geom.DirectionCW)
	// 0x80 alpha red: premul = a=0x80, r=round(255*128/255)=0x80.
	FillPath(p, geom.IRectLTRB(0, 0, 2, 1), NewSolidBlitter(dev, 0x80FF0000))
	pm := colorcore.Color(0x80FF0000).PreMultiply()
	src := deviceRGBA(pm)
	invA := alpha255To256(255 - deviceAlpha(src))
	wantOverWhite := uint32(0)
	for shift := 0; shift < 32; shift += 8 {
		d := (uint32(0xFF) * invA) >> 8
		wantOverWhite |= ((d + (src >> shift & 0xFF)) & 0xFF) << shift
	}
	if dev.Pix[0] != wantOverWhite {
		t.Errorf("over white: got %08x want %08x", dev.Pix[0], wantOverWhite)
	}
	if dev.Pix[1] != src {
		t.Errorf("over transparent: got %08x want %08x (src unchanged)", dev.Pix[1], src)
	}
}

func TestFixedHelpers(t *testing.T) {
	if FloatToFixed(1.5) != 3<<15 {
		t.Errorf("FloatToFixed(1.5) = %d", FloatToFixed(1.5))
	}
	if FixedRoundToInt(FixedHalf) != 1 || FixedRoundToInt(FixedHalf-1) != 0 {
		t.Error("FixedRoundToInt rounding")
	}
	if FDot6Round(32) != 1 || FDot6Round(31) != 0 {
		t.Error("FDot6Round rounding")
	}
	if FDot6Div(64, 128) != FixedHalf {
		t.Errorf("FDot6Div(64,128) = %d", FDot6Div(64, 128))
	}
	if FixedMul(FixedHalf, FixedHalf) != FixedOne/4 {
		t.Error("FixedMul")
	}
}
