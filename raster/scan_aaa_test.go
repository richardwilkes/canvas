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

// aaFill fills p with AA into a fresh w×h pixmap using an opaque black solid blitter and returns the pixmap (so tests
// can read coverage out of the alpha channel).
func aaFill(t *testing.T, p *path.Path, w, h int32) *Pixmap {
	t.Helper()
	dev := NewPixmap(w, h)
	AntiFillPath(p, geom.IRectLTRB(0, 0, w, h), NewSolidBlitter(dev, colorcore.Color(0xFF000000)))
	return dev
}

func alphaAt(dev *Pixmap, x, y int32) uint32 {
	return dev.Pix[dev.addr(x, y)] >> 24
}

// TestAAFillFatRect: an axis-aligned rect wide enough for the blitFatAntiRect fast path; interior pixels are fully
// covered, boundary pixels get the snapped partial coverages of the scalar-to-alpha coverage math.
func TestAAFillFatRect(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(10.5, 20.25, 20.5, 30.75), geom.DirectionCW)
	dev := aaFill(t, p, 40, 40)

	// interior
	for y := int32(21); y < 30; y++ {
		for x := int32(11); x < 20; x++ {
			if a := alphaAt(dev, x, y); a != 255 {
				t.Fatalf("interior (%d,%d) alpha %d", x, y, a)
			}
		}
	}
	// outside
	if a := alphaAt(dev, 9, 25); a != 0 {
		t.Fatalf("outside alpha %d", a)
	}
	// left/right half-covered columns (0.5 coverage → scalar_to_alpha(0.5) = 127)
	for y := int32(21); y < 30; y++ {
		if a := alphaAt(dev, 10, y); a != 127 {
			t.Fatalf("left column (10,%d) alpha %d, want 127", y, a)
		}
		if a := alphaAt(dev, 20, y); a != 127 {
			t.Fatalf("right column (20,%d) alpha %d, want 127", y, a)
		}
	}
	// top row: 0.75 vertical coverage → 191; bottom row 0.75 → 191
	if a := alphaAt(dev, 15, 20); a != 191 {
		t.Fatalf("top row alpha %d, want 191", a)
	}
	if a := alphaAt(dev, 15, 30); a != 191 {
		t.Fatalf("bottom row alpha %d, want 191", a)
	}
	// corners: 0.5 * 0.75 = 0.375 → 95
	if a := alphaAt(dev, 10, 20); a != 95 {
		t.Fatalf("corner alpha %d, want 95", a)
	}
}

// TestAAFillSmallVsLargeConsistency: the same geometry rendered through the small-path MaskAdditiveBlitter route and
// the RLE route must agree closely; they are not identical because the RLE route snaps alphas near 0/255 and the second
// route also switches to the general (non-convex) walker. Force the route change via an extra faraway contour that only
// enlarges ir without touching the triangle's pixels.
func TestAAFillSmallVsLargeConsistency(t *testing.T) {
	tri := func(extra bool) *Pixmap {
		p := path.New()
		p.MoveTo(4.3, 3.2)
		p.LineTo(24.7, 9.1)
		p.LineTo(8.9, 24.6)
		p.Close()
		if extra {
			// a fully-covered faraway square, disjoint from the triangle's pixels
			p.AddRect(geom.RectLTRB(90, 90, 110, 110), geom.DirectionCW)
		}
		return aaFill(t, p, 120, 120)
	}
	small := tri(false) // bounds ≈ 21×22 → mask blitter
	large := tri(true)  // bounds ≈ 106×107 → RLE
	for y := int32(0); y < 40; y++ {
		for x := int32(0); x < 40; x++ {
			a, b := alphaAt(small, x, y), alphaAt(large, x, y)
			if d := int(a) - int(b); d < -8 || d > 8 {
				t.Fatalf("(%d,%d): mask route alpha %d, RLE route alpha %d", x, y, a, b)
			}
		}
	}
}

// TestAAFillEvenOddDonut: even-odd fill of two nested rects — the hole must be empty, the ring full.
func TestAAFillEvenOddDonut(t *testing.T) {
	p := path.New()
	p.AddRect(geom.RectLTRB(10, 10, 50, 50), geom.DirectionCW)
	p.AddRect(geom.RectLTRB(20, 20, 40, 40), geom.DirectionCW)
	p.SetFillType(path.FillEvenOdd)
	dev := aaFill(t, p, 64, 64)
	if a := alphaAt(dev, 30, 30); a != 0 {
		t.Fatalf("hole alpha %d", a)
	}
	if a := alphaAt(dev, 15, 30); a != 255 {
		t.Fatalf("ring alpha %d", a)
	}
	if a := alphaAt(dev, 5, 5); a != 0 {
		t.Fatalf("outside alpha %d", a)
	}
}

// TestAAFillInverse: an inverse winding fill covers the outside fully and leaves the inside empty, with complementary
// coverage on the boundary.
func TestAAFillInverse(t *testing.T) {
	build := func(inverse bool) *path.Path {
		p := path.New()
		p.MoveTo(10.5, 10)
		p.LineTo(40.25, 12)
		p.LineTo(30, 38.75)
		p.Close()
		if inverse {
			p.SetFillType(path.FillInverseWinding)
		}
		return p
	}
	normal := aaFill(t, build(false), 64, 64)
	inverse := aaFill(t, build(true), 64, 64)
	if a := alphaAt(inverse, 2, 2); a != 255 {
		t.Fatalf("inverse outside alpha %d", a)
	}
	if a := alphaAt(inverse, 25, 20); a != 0 {
		t.Fatalf("inverse inside alpha %d", a)
	}
	// Coverage roughly complements everywhere (the two fills accumulate boundary coverage with independent rounding, so
	// allow a few steps of slack).
	for y := int32(0); y < 64; y++ {
		for x := int32(0); x < 64; x++ {
			sum := alphaAt(normal, x, y) + alphaAt(inverse, x, y)
			if sum < 238 || sum > 264 {
				t.Fatalf("(%d,%d): normal %d + inverse %d = %d", x, y,
					alphaAt(normal, x, y), alphaAt(inverse, x, y), sum)
			}
		}
	}
}

// TestAAFillClip: AA fills honor a rect clip exactly (nothing outside, byte-identical inside).
func TestAAFillClip(t *testing.T) {
	p := path.New()
	p.AddCircle(30, 30, 25, geom.DirectionCW)
	full := aaFill(t, p, 64, 64)

	clipped := NewPixmap(64, 64)
	clip := geom.IRectLTRB(20, 15, 45, 50)
	AntiFillPath(p, clip, NewSolidBlitter(clipped, colorcore.Color(0xFF000000)))
	for y := int32(0); y < 64; y++ {
		for x := int32(0); x < 64; x++ {
			in := x >= clip.Left && x < clip.Right && y >= clip.Top && y < clip.Bottom
			switch {
			case !in && alphaAt(clipped, x, y) != 0:
				t.Fatalf("(%d,%d): outside clip alpha %d", x, y, alphaAt(clipped, x, y))
			case in && alphaAt(clipped, x, y) != alphaAt(full, x, y):
				t.Fatalf("(%d,%d): clipped %d, full %d", x, y, alphaAt(clipped, x, y), alphaAt(full, x, y))
			}
		}
	}
}

// TestAAFillHugeCoordsFallsBack: coordinates that overflow the fixed-point range take the documented non-AA fallback
// rather than misrendering; just verify it doesn't panic and fills something sane.
func TestAAFillHugeCoordsFallsBack(t *testing.T) {
	p := path.New()
	p.MoveTo(-1e6, -1e6)
	p.LineTo(1e6, -1e6)
	p.LineTo(1e6, 1e6)
	p.LineTo(-1e6, 1e6)
	p.Close()
	dev := aaFill(t, p, 16, 16)
	for y := int32(0); y < 16; y++ {
		for x := int32(0); x < 16; x++ {
			if a := alphaAt(dev, x, y); a != 255 {
				t.Fatalf("(%d,%d) alpha %d", x, y, a)
			}
		}
	}
}

// TestAlphaRunsAdd: the RLE accumulation matches the documented example step by step.
func TestAlphaRunsAdd(t *testing.T) {
	var runs AlphaRuns
	runs.Runs = make([]int16, 9)
	runs.Alpha = make([]Alpha, 10)
	runs.Reset(8)
	if !runs.Empty() {
		t.Fatal("fresh runs not empty")
	}
	runs.Add(2, 10, 3, 20, 0xFF, 0)
	// second composed span overlaps the first's start pixel (callers reset offsetX to 0 when moving left); expand the
	// RLE and check the accumulated alphas.
	runs.Add(1, 0, 1, 5, 0, 0)
	var expanded [8]int
	x := 0
	for x < 8 {
		n := int(runs.Runs[x])
		if n <= 0 {
			break
		}
		for i := 0; i < n; i++ {
			expanded[x+i] = int(runs.Alpha[x])
		}
		x += n
	}
	want := [8]int{0, 0, 15, 255, 255, 255, 20, 0}
	if expanded != want {
		t.Fatalf("expanded %v, want %v", expanded, want)
	}
}
