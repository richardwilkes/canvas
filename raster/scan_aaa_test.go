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

// TestAAFillFractionalHorizontalEdge: a horizontal edge that lands a fraction of a scanline past a pixel boundary must
// cover that row in proportion to the fraction, in every lane. The analytic converter snaps edge y coordinates to a
// grid; while that grid was a quarter of a scanline, a sliver thinner than 1/8 px snapped onto the boundary and its row
// rendered empty — and everything else resolved in steps of 64 alpha — so a path fill disagreed with the FDot8 rect
// lane on identical geometry.
func TestAAFillFractionalHorizontalEdge(t *testing.T) {
	const (
		w = 144
		h = 162
		// A column the geometry covers edge to edge, so only the vertical fraction is under test.
		col = w / 2
	)
	clip := geom.IRectLTRB(0, 0, w, h)
	for _, tc := range []struct {
		name   string
		bottom float32
		want   uint32
	}{
		{name: "sliver", bottom: 161.11, want: 28}, // the reported case: 0.11 px of coverage, lost entirely
		{name: "under-half", bottom: 161.3, want: 76},
		{name: "half", bottom: 161.5, want: 128},
		{name: "over-half", bottom: 161.75, want: 191},
		{name: "near-whole", bottom: 161.9, want: 231},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := geom.RectLTRB(0, 0, w, tc.bottom)

			// The FDot8 rect lane has always resolved the fraction at 1/256 px; it is the reference the path lanes
			// have to agree with.
			ref := NewPixmap(w, h)
			AntiFillRectRegion(r, NewRegionRect(clip), NewSolidBlitter(ref, colorcore.Color(0xFF000000)))
			refAlpha := alphaAt(ref, col, h-1)

			// A non-rect polygon carrying the same bottom edge, to reach the general walker rather than the convex
			// one. Its left edge is slanted off the top of the surface so col stays fully covered in every row.
			poly := path.New()
			poly.MoveTo(-40, -10).LineTo(w, -10).LineTo(w, tc.bottom).LineTo(0, tc.bottom).Close()

			for _, lane := range []struct {
				p    *path.Path
				name string
			}{
				{p: path.New().AddRect(r, geom.DirectionCW), name: "rect path"},
				{p: poly, name: "polygon"},
			} {
				dev := aaFill(t, lane.p, w, h)
				if got := alphaAt(dev, col, h-1); got != tc.want {
					t.Errorf("%s: partial row alpha %d, want %d", lane.name, got, tc.want)
				} else if d := int(got) - int(refAlpha); d < -1 || d > 1 {
					// The 1/64 snap grid puts the path lanes within one step of the rect lane's 1/256.
					t.Errorf("%s: partial row alpha %d, rect lane %d", lane.name, got, refAlpha)
				}
				// The last whole row stays whole, so an over-eager sliver cannot pass as a correct one.
				if got := alphaAt(dev, col, h-2); got != 255 {
					t.Errorf("%s: last whole row alpha %d, want 255", lane.name, got)
				}
			}
		})
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
	// allow a few steps of slack). The measured spread for this path is [248, 265]; the bounds keep ~10 steps of
	// headroom on each side so the guard catches a structural break rather than tracking every ±1 rounding shift.
	for y := int32(0); y < 64; y++ {
		for x := int32(0); x < 64; x++ {
			sum := alphaAt(normal, x, y) + alphaAt(inverse, x, y)
			if sum < 238 || sum > 275 {
				t.Fatalf("(%d,%d): normal %d + inverse %d = %d", x, y,
					alphaAt(normal, x, y), alphaAt(inverse, x, y), sum)
			}
		}
	}
}

// TestAAFillClip: AA fills honor a rect clip — nothing at all outside it, and inside it the same coverage the unclipped
// fill produces. Inside agreement is close rather than exact: the edge builder clips edges against the clip rect, so the
// clipped fill walks a different edge set, and where the path runs tangent to the clip boundary the trapezoid math can
// land a step apart. Here that costs 3 alpha on the single pixel whose corner the circle passes exactly through; other
// circle/clip pairs have always differed by more, so exactness is not the invariant to assert.
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
			case in:
				if d := int(alphaAt(clipped, x, y)) - int(alphaAt(full, x, y)); d < -4 || d > 4 {
					t.Fatalf("(%d,%d): clipped %d, full %d", x, y, alphaAt(clipped, x, y),
						alphaAt(full, x, y))
				}
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

// expandRLE flattens an AlphaRuns scanline into one alpha per pixel. The blitters are pooled, so the alpha backing
// array carries stale values from earlier fills outside the runs the RLE actually describes; only the structure
// starting at index 0 is meaningful.
func expandRLE(runs *AlphaRuns, width int32) []Alpha {
	out := make([]Alpha, width)
	for i := int32(0); i < width; {
		n := int32(runs.Runs[i])
		if n <= 0 {
			break
		}
		for j := int32(0); j < n && i+j < width; j++ {
			out[i+j] = runs.Alpha[i]
		}
		i += n
	}
	return out
}

// TestAdditiveBlitterRunLeftOfBounds: the RLE additive blitters translate the run's x into blitter space and drop what
// falls left of it. Reslicing antialias[-x:] before clamping the length panicked with "slice bounds out of range"
// whenever a run lay entirely left of the blitter's left bound by more than its own length; Skia does the same step as
// pointer arithmetic, where the negative length is caught right after. Runs that partially overlap must still blit the
// surviving tail, and the blitter must stay usable afterwards.
func TestAdditiveBlitterRunLeftOfBounds(t *testing.T) {
	const left, width = 20, 16
	ir := geom.IRectLTRB(left, 0, left+width, 4)
	run := func(n int, alpha Alpha) []Alpha {
		aa := make([]Alpha, n)
		for i := range aa {
			aa[i] = alpha
		}
		return aa
	}

	for _, tc := range []struct {
		make func(Blitter) (*runBasedAdditiveBlitter, func(int32, []Alpha))
		name string
	}{
		{
			name: "runBased",
			make: func(dst Blitter) (*runBasedAdditiveBlitter, func(int32, []Alpha)) {
				r := getRunBasedAdditiveBlitter(dst, ir, ir, false)
				return r, func(x int32, aa []Alpha) { r.blitAntiHRun(x, 0, aa) }
			},
		},
		{
			name: "safeRLE",
			make: func(dst Blitter) (*runBasedAdditiveBlitter, func(int32, []Alpha)) {
				s := getSafeRLEAdditiveBlitter(dst, ir, ir, false)
				return &s.runBasedAdditiveBlitter, func(x int32, aa []Alpha) { s.blitAntiHRun(x, 0, aa) }
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := NewPixmap(left+width, 4)
			r, blit := tc.make(NewSolidBlitter(dev, colorcore.Color(0xFF000000)))

			// Entirely left of the bound by more (and by exactly) its own length: dropped, not a panic.
			blit(left-9, run(4, 0x40))
			blit(left-4, run(4, 0x40))
			for i, got := range expandRLE(&r.runs, width) {
				if got != 0 {
					t.Fatalf("a fully-clipped run wrote alpha %#x at %d", got, i)
				}
			}

			// Straddling the bound: only the part at or right of it survives, at the blitter's own x origin.
			blit(left-2, run(6, 0x40))
			for i, got := range expandRLE(&r.runs, width) {
				want := Alpha(0)
				if i < 4 {
					want = 0x40
				}
				if got != want {
					t.Fatalf("straddling run: alpha %#x at %d, want %#x", got, i, want)
				}
			}
		})
	}
}

// TestMaskAdditiveBlitterRowCacheAfterVerticalBlits: BlitV and BlitRect walk down rows by stepping the row pointer by
// RowBytes, so the row cache getRow already set up for row y stays correct. The trailing `m.rowY = y` fixups those two
// carried were no-ops, and the comment calling the cache stale invited "fixing" rowOff too — which would silently move
// every later blit to the wrong row. This pins that a blit after a vertical walk still lands where it should.
func TestMaskAdditiveBlitterRowCacheAfterVerticalBlits(t *testing.T) {
	ir := geom.IRectLTRB(0, 0, 8, 8)
	var m maskAdditiveBlitter
	m.init(nil, ir, ir)

	m.BlitV(1, 2, 4, 0x40)   // x=1, rows 2..5
	m.blitAntiH1(3, 2, 0x10) // must still land on row 2
	m.BlitRect(5, 3, 2, 2)   // x=5..6, rows 3..4
	m.blitAntiH1(0, 3, 0x20) // must still land on row 3

	want := map[[2]int32]uint8{
		{1, 2}: 0x40, {1, 3}: 0x40, {1, 4}: 0x40, {1, 5}: 0x40,
		{3, 2}: 0x10,
		{5, 3}: 0xFF, {6, 3}: 0xFF, {5, 4}: 0xFF, {6, 4}: 0xFF,
		{0, 3}: 0x20,
	}
	for y := int32(0); y < 8; y++ {
		for x := int32(0); x < 8; x++ {
			if got := m.mask.Image[m.mask.addr8(x, y)]; got != want[[2]int32{x, y}] {
				t.Fatalf("(%d,%d) got %#x want %#x", x, y, got, want[[2]int32{x, y}])
			}
		}
	}
}
