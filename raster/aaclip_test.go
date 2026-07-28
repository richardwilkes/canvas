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
	"bytes"
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

const aaGrid = 96

// aaClipToGrid expands an AAClip's coverage into a dense aaGrid x aaGrid alpha grid.
func aaClipToGrid(c *AAClip) []uint8 {
	out := make([]uint8, aaGrid*aaGrid)
	if c.IsEmpty() {
		return out
	}
	mask := c.CopyToMask()
	b := mask.Bounds
	for y := max(b.Top, 0); y < min(b.Bottom, aaGrid); y++ {
		for x := max(b.Left, 0); x < min(b.Right, aaGrid); x++ {
			out[y*aaGrid+x] = mask.Image[mask.addr8(x, y)]
		}
	}
	return out
}

// maskToGrid expands an A8 mask into a dense grid.
func maskToGrid(mask *Mask) []uint8 {
	out := make([]uint8, aaGrid*aaGrid)
	b := mask.Bounds
	for y := max(b.Top, 0); y < min(b.Bottom, aaGrid); y++ {
		for x := max(b.Left, 0); x < min(b.Right, aaGrid); x++ {
			out[y*aaGrid+x] = mask.Image[mask.addr8(x, y)]
		}
	}
	return out
}

func gridsEqual(t *testing.T, what string, got, want []uint8) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: pixel (%d,%d) got %d want %d", what, i%aaGrid, i/aaGrid, got[i], want[i])
		}
	}
}

func TestAAClipSetRect(t *testing.T) {
	var c AAClip
	if c.SetRect(geom.IRect{}) {
		t.Fatal("empty rect should produce empty clip")
	}
	if !c.IsEmpty() {
		t.Fatal("expected empty")
	}

	r := geom.IRectLTRB(10, 20, 40, 50)
	if !c.SetRect(r) {
		t.Fatal("SetRect returned false")
	}
	if c.IsEmpty() || !c.IsRect() || c.Bounds() != r {
		t.Fatalf("bad rect clip state: empty=%v isRect=%v bounds=%v", c.IsEmpty(), c.IsRect(), c.Bounds())
	}

	mask := c.CopyToMask()
	for _, a := range mask.Image {
		if a != 0xFF {
			t.Fatal("rect clip mask should be all 0xFF")
		}
	}

	// QuickContains requires the row band to extend one row past the query's exclusive bottom, so a rect clip does not
	// quickContain rects touching its bottom edge.
	if c.QuickContains(r) {
		t.Fatal("quickContains(bounds) is conservatively false")
	}
	if !c.QuickContains(geom.IRectLTRB(10, 20, 40, 49)) {
		t.Fatal("expected quickContains for rect ending one row above the bottom")
	}
	if c.QuickContains(geom.IRectLTRB(9, 20, 40, 45)) {
		t.Fatal("should not contain rect outside bounds")
	}
}

func TestAAClipSetPathMatchesFillMask(t *testing.T) {
	// AAClip.SetPath's coverage must equal an AA fill of the same path (the builder consumes the same scan converter),
	// modulo the clip's bounds trimming.
	specs := []*path.Path{
		path.New().AddCircle(48, 48, 30, geom.DirectionCW),
		path.New().AddOval(geom.RectLTRB(5.25, 10.75, 90.5, 60.25), geom.DirectionCW),
		path.New().MoveTo(10.5, 10.25).LineTo(80.75, 30.5).LineTo(20.25, 85.75).Close(),
	}
	clipTo := geom.IRectLTRB(0, 0, aaGrid, aaGrid)
	for i, p := range specs {
		var c AAClip
		if !c.SetPath(p, clipTo, true) {
			t.Fatalf("spec %d: SetPath returned false", i)
		}
		want := maskToGrid(FillPathToMask(p, clipTo, true))
		got := aaClipToGrid(&c)
		gridsEqual(t, "setPath vs fill mask", got, want)
	}
}

func TestAAClipPathWithHole(t *testing.T) {
	// Two rects separated by an empty scanline gap must produce an explicit empty row (the checkForYGap logic), and the
	// clip's expanded coverage must match the path's fill exactly.
	p := path.New()
	p.AddRect(geom.RectXYWH(0, 0, 20, 4), geom.DirectionCW)
	p.AddRect(geom.RectXYWH(0, 8, 20, 4), geom.DirectionCW)

	bounds := geom.IRectLTRB(0, 0, 20, 12)
	for _, doAA := range []bool{false, true} {
		var c AAClip
		if !c.SetPath(p, bounds, doAA) {
			t.Fatal("SetPath returned false")
		}
		mask := c.CopyToMask()
		if mask.Bounds != bounds {
			t.Fatalf("bounds got %v want %v", mask.Bounds, bounds)
		}
		for y := int32(0); y < 12; y++ {
			want := uint8(0xFF)
			if y >= 4 && y < 8 {
				want = 0
			}
			for x := int32(0); x < 20; x++ {
				if got := mask.Image[mask.addr8(x, y)]; got != want {
					t.Fatalf("doAA=%v pixel (%d,%d) got %d want %d", doAA, x, y, got, want)
				}
			}
		}
	}
}

func TestAAClipSetRegion(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for iter := 0; iter < 50; iter++ {
		rgn := &Region{}
		n := 1 + rng.Intn(5)
		for i := 0; i < n; i++ {
			rgn.OpRect(randRect(rng), RegionUnion)
		}
		var c AAClip
		c.SetRegion(rgn)
		if rgn.IsEmpty() {
			if !c.IsEmpty() {
				t.Fatal("empty region should produce empty clip")
			}
			continue
		}
		if c.Bounds() != rgn.Bounds() {
			t.Fatalf("bounds got %v want %v", c.Bounds(), rgn.Bounds())
		}
		got := aaClipToGrid(&c)
		for y := int32(0); y < aaGrid; y++ {
			for x := int32(0); x < aaGrid; x++ {
				want := uint8(0)
				if rgn.ContainsPoint(x, y) {
					want = 0xFF
				}
				if got[y*aaGrid+x] != want {
					t.Fatalf("iter %d pixel (%d,%d) got %d want %d", iter, x, y, got[y*aaGrid+x], want)
				}
			}
		}
	}
}

func TestAAClipOpsVsGrids(t *testing.T) {
	// Differential: AAClip.Op must equal the per-pixel alpha math (mulDiv255Round for intersect, a*(255-b)/255 for
	// difference) applied to the operands' expanded coverage.
	rng := rand.New(rand.NewSource(23))
	makeRandomClip := func() *AAClip {
		var c AAClip
		switch rng.Intn(3) {
		case 0:
			c.SetRect(randRect(rng))
		case 1:
			cx := float32(rng.Intn(aaGrid))
			cy := float32(rng.Intn(aaGrid))
			r := 4 + float32(rng.Intn(aaGrid/3))
			c.SetPath(path.New().AddCircle(cx, cy, r, geom.DirectionCW),
				geom.IRectLTRB(0, 0, aaGrid, aaGrid), true)
		default:
			x := float32(rng.Intn(aaGrid))
			y := float32(rng.Intn(aaGrid))
			w := 1 + float32(rng.Intn(aaGrid/2)) + float32(rng.Intn(4))/4
			h := 1 + float32(rng.Intn(aaGrid/2)) + float32(rng.Intn(4))/4
			c.SetPath(path.New().AddRect(geom.RectXYWH(x, y, w, h), geom.DirectionCW),
				geom.IRectLTRB(0, 0, aaGrid, aaGrid), true)
		}
		return &c
	}

	for iter := 0; iter < 200; iter++ {
		a := makeRandomClip()
		b := makeRandomClip()
		gridA := aaClipToGrid(a)
		gridB := aaClipToGrid(b)

		for _, op := range []ClipOp{ClipIntersect, ClipDifference} {
			target := &AAClip{}
			*target = *a // struct copy shares the head
			target.Op(b, op)
			got := aaClipToGrid(target)
			for i := range got {
				var want uint8
				if op == ClipIntersect {
					want = colorcore.MulDiv255Round(gridA[i], gridB[i])
				} else {
					want = colorcore.MulDiv255Round(gridA[i], 0xFF-gridB[i])
				}
				if got[i] != want {
					t.Fatalf("iter %d op %d pixel (%d,%d) got %d want %d",
						iter, op, i%aaGrid, i/aaGrid, got[i], want)
				}
			}
		}
	}
}

func TestAAClipOpIRectQuickPaths(t *testing.T) {
	base := geom.IRectLTRB(10, 10, 60, 60)

	// Intersect with a disjoint rect -> empty.
	var c AAClip
	c.SetRect(base)
	if c.OpIRect(geom.IRectLTRB(70, 70, 90, 90), ClipIntersect) || !c.IsEmpty() {
		t.Fatal("disjoint intersect should empty the clip")
	}

	// Difference with a disjoint rect -> unchanged.
	c.SetRect(base)
	if !c.OpIRect(geom.IRectLTRB(70, 70, 90, 90), ClipDifference) || c.Bounds() != base {
		t.Fatal("disjoint difference should leave the clip unchanged")
	}

	// Difference with a covering rect -> empty.
	c.SetRect(base)
	if c.OpIRect(geom.IRectLTRB(0, 0, 100, 100), ClipDifference) || !c.IsEmpty() {
		t.Fatal("covering difference should empty the clip")
	}

	// Intersect with a covering rect -> unchanged.
	c.SetRect(base)
	if !c.OpIRect(geom.IRectLTRB(0, 0, 100, 100), ClipIntersect) || c.Bounds() != base {
		t.Fatal("covering intersect should leave the clip unchanged")
	}

	// Partial intersect of a hard rect: brute-force comparison (the quickContains fast path cannot trigger due to the
	// conservative row check, so this exercises the general op machinery).
	c.SetRect(base)
	c.OpIRect(geom.IRectLTRB(30, 5, 90, 40), ClipIntersect)
	want := geom.IRectLTRB(30, 10, 60, 40)
	if c.Bounds() != want || !c.IsRect() {
		t.Fatalf("partial intersect got bounds %v (isRect=%v) want %v", c.Bounds(), c.IsRect(), want)
	}
}

func TestAAClipTranslate(t *testing.T) {
	var c AAClip
	c.SetPath(path.New().AddCircle(40, 40, 20, geom.DirectionCW), geom.IRectLTRB(0, 0, aaGrid, aaGrid), true)
	orig := aaClipToGrid(&c)

	var moved AAClip
	if !c.Translate(7, 3, &moved) {
		t.Fatal("translate returned false")
	}
	if moved.Bounds() != c.Bounds().Offset(7, 3) {
		t.Fatal("translated bounds wrong")
	}
	got := aaClipToGrid(&moved)
	for y := int32(0); y < aaGrid; y++ {
		for x := int32(0); x < aaGrid; x++ {
			var want uint8
			if x >= 7 && y >= 3 {
				want = orig[(y-3)*aaGrid+(x-7)]
			}
			if got[y*aaGrid+x] != want {
				t.Fatalf("pixel (%d,%d) got %d want %d", x, y, got[y*aaGrid+x], want)
			}
		}
	}
}

func TestAAClipBlitterReproducesClip(t *testing.T) {
	// Blitting a full-coverage rect over the clip's bounds through an AAClipBlitter into an A8 target must reproduce
	// the clip's own coverage exactly.
	var c AAClip
	c.SetPath(path.New().AddCircle(45.5, 47.25, 33.3, geom.DirectionCW),
		geom.IRectLTRB(0, 0, aaGrid, aaGrid), true)

	bounds := c.Bounds()
	target := &Mask{
		Image:    make([]uint8, int(bounds.Width())*int(bounds.Height())),
		Bounds:   bounds,
		RowBytes: bounds.Width(),
	}
	var b AAClipBlitter
	b.Init(NewA8CoverageBlitter(target), &c)
	b.BlitRect(bounds.Left, bounds.Top, bounds.Width(), bounds.Height())

	gridsEqual(t, "blitter vs clip mask", maskToGrid(target), aaClipToGrid(&c))
}

func TestAAClipBlitterBlitMask(t *testing.T) {
	// BlitMask through the AA clip must multiply the mask by the clip coverage.
	var c AAClip
	c.SetPath(path.New().AddCircle(48, 48, 30, geom.DirectionCW), geom.IRectLTRB(0, 0, aaGrid, aaGrid), true)
	clipGrid := aaClipToGrid(&c)

	p := path.New().AddOval(geom.RectLTRB(20.5, 15.25, 80.75, 70.5), geom.DirectionCW)
	srcMask := FillPathToMask(p, geom.IRectLTRB(0, 0, aaGrid, aaGrid), true)
	srcGrid := maskToGrid(srcMask)

	target := &Mask{
		Image:    make([]uint8, aaGrid*aaGrid),
		Bounds:   geom.IRectLTRB(0, 0, aaGrid, aaGrid),
		RowBytes: aaGrid,
	}
	var b AAClipBlitter
	b.Init(NewA8CoverageBlitter(target), &c)

	// restrict to the clip bounds before blitting
	clip := srcMask.Bounds
	if !clip.Intersect(c.Bounds()) {
		t.Fatal("expected overlap")
	}
	b.BlitMask(srcMask, clip)

	got := maskToGrid(target)
	for y := clip.Top; y < clip.Bottom; y++ {
		for x := clip.Left; x < clip.Right; x++ {
			i := y*aaGrid + x
			want := colorcore.MulDiv255Round(srcGrid[i], clipGrid[i])
			if got[i] != want {
				t.Fatalf("pixel (%d,%d) got %d want %d", x, y, got[i], want)
			}
		}
	}
}

func TestRasterClipBWToAAPromotion(t *testing.T) {
	rc := NewRasterClipRect(geom.IRectLTRB(0, 0, aaGrid, aaGrid))
	if !rc.IsBW() || !rc.IsRect() {
		t.Fatal("fresh rect clip should be BW rect")
	}

	// Clipping with an AA path promotes to AA.
	im := geom.IdentityMatrix()
	m := &im
	rc.OpPath(path.New().AddCircle(48, 48, 30, geom.DirectionCW), m, ClipIntersect, true)
	if rc.IsBW() {
		t.Fatal("AA path clip should promote to AA")
	}
	if rc.IsEmpty() || rc.IsRect() {
		t.Fatal("circle clip should be a non-rect, non-empty clip")
	}

	// Clipping an AA clip with an integral rect that leaves only full-coverage interior... still AA; but an AA rect op
	// whose result is a hard rect must collapse back to BW (detectAARect).
	rc2 := NewRasterClipRect(geom.IRectLTRB(0, 0, aaGrid, aaGrid))
	rc2.OpPath(path.New().AddRect(geom.RectLTRB(10, 10, 50, 50), geom.DirectionCW), m, ClipIntersect, true)
	if !rc2.IsBW() || !rc2.IsRect() || rc2.Bounds() != geom.IRectLTRB(10, 10, 50, 50) {
		t.Fatalf("integral AA rect path should collapse back to BW rect; isBW=%v isRect=%v bounds=%v",
			rc2.IsBW(), rc2.IsRect(), rc2.Bounds())
	}
}

func TestRasterClipOpRectNearlyIntegral(t *testing.T) {
	im := geom.IdentityMatrix()
	m := &im

	// Nearly-integral AA rect stays BW.
	rc := NewRasterClipRect(geom.IRectLTRB(0, 0, 100, 100))
	rc.OpRect(geom.RectLTRB(10.05, 10, 50, 49.96), m, ClipIntersect, true)
	if !rc.IsBW() {
		t.Fatal("nearly-integral AA rect should stay BW")
	}
	if rc.Bounds() != geom.IRectLTRB(10, 10, 50, 50) {
		t.Fatalf("bounds got %v", rc.Bounds())
	}

	// A genuinely fractional AA rect goes AA.
	rc2 := NewRasterClipRect(geom.IRectLTRB(0, 0, 100, 100))
	rc2.OpRect(geom.RectLTRB(10.5, 10, 50, 50), m, ClipIntersect, true)
	if rc2.IsBW() {
		t.Fatal("fractional AA rect should become AA")
	}
}

func TestRasterClipFillsMatchRegionLeg(t *testing.T) {
	// For a BW raster clip, the Clip entry points must produce exactly the region-clipped fill.
	p := path.New().AddCircle(48, 48, 35, geom.DirectionCW)
	clipR := geom.IRectLTRB(20, 10, 80, 70)

	for _, aa := range []bool{false, true} {
		want := &Mask{Image: make([]uint8, aaGrid*aaGrid), Bounds: geom.IRectLTRB(0, 0, aaGrid, aaGrid), RowBytes: aaGrid}
		got := &Mask{Image: make([]uint8, aaGrid*aaGrid), Bounds: geom.IRectLTRB(0, 0, aaGrid, aaGrid), RowBytes: aaGrid}

		if aa {
			AntiFillPathRegion(p, NewRegionRect(clipR), NewA8CoverageBlitter(want))
		} else {
			FillPathRegion(p, NewRegionRect(clipR), NewA8CoverageBlitter(want))
		}
		rc := NewRasterClipRect(clipR)
		if aa {
			AntiFillPathRasterClip(p, rc, NewA8CoverageBlitter(got))
		} else {
			FillPathRasterClip(p, rc, NewA8CoverageBlitter(got))
		}
		gridsEqual(t, "BW rasterclip fill", maskToGrid(got), maskToGrid(want))
	}
}

func TestRasterClipAAFillMultipliesCoverage(t *testing.T) {
	// AA fill through an AA raster clip == per-pixel MulDiv255Round of unclipped fill coverage and clip coverage (the
	// AAClipBlitter's merge is exactly that multiply).
	full := geom.IRectLTRB(0, 0, aaGrid, aaGrid)
	clipPath := path.New().AddCircle(40, 44, 28.5, geom.DirectionCW)
	fillPath := path.New().AddOval(geom.RectLTRB(15.25, 20.5, 85.5, 75.25), geom.DirectionCW)

	rc := NewRasterClipRect(full)
	im := geom.IdentityMatrix()
	m := &im
	rc.OpPath(clipPath, m, ClipIntersect, true)
	if rc.IsBW() {
		t.Fatal("expected AA clip")
	}
	clipGrid := aaClipToGrid(rc.AARgn())

	// The AA leg rasterizes the fill restricted to the AA clip's *bounds* (the tmp region in AntiFillPathRasterClip)
	// before the per-pixel multiply, and that restriction routes curve segments through the edge clipper — so build the
	// reference the same way rather than from an unrestricted fill.
	fillGrid := maskToGrid(FillPathToMask(fillPath, rc.Bounds(), true))

	got := &Mask{Image: make([]uint8, aaGrid*aaGrid), Bounds: full, RowBytes: aaGrid}
	AntiFillPathRasterClip(fillPath, rc, NewA8CoverageBlitter(got))
	gotGrid := maskToGrid(got)

	for i := range gotGrid {
		want := colorcore.MulDiv255Round(fillGrid[i], clipGrid[i])
		if gotGrid[i] != want {
			t.Fatalf("pixel (%d,%d) got %d want %d", i%aaGrid, i/aaGrid, gotGrid[i], want)
		}
	}
}

func TestRasterClipStack(t *testing.T) {
	s := NewRasterClipStack(100, 80)
	if s.RC().Bounds() != geom.IRectLTRB(0, 0, 100, 80) {
		t.Fatal("root bounds wrong")
	}

	im := geom.IdentityMatrix()
	m := &im

	// Deferred save: no new entry until a mutation.
	s.Save()
	s.ClipRect(m, geom.RectLTRB(10, 10, 50, 50), ClipIntersect, false)
	if s.RC().Bounds() != geom.IRectLTRB(10, 10, 50, 50) {
		t.Fatalf("clip after save got %v", s.RC().Bounds())
	}
	s.Restore()
	if s.RC().Bounds() != geom.IRectLTRB(0, 0, 100, 80) {
		t.Fatalf("restore should recover root clip, got %v", s.RC().Bounds())
	}

	// Nested saves with sequential clips.
	s.Save()
	s.ClipRect(m, geom.RectLTRB(0, 0, 60, 60), ClipIntersect, false)
	s.Save()
	s.ClipRect(m, geom.RectLTRB(20, 20, 100, 100), ClipIntersect, false)
	if s.RC().Bounds() != geom.IRectLTRB(20, 20, 60, 60) {
		t.Fatalf("nested clip got %v", s.RC().Bounds())
	}
	s.Restore()
	if s.RC().Bounds() != geom.IRectLTRB(0, 0, 60, 60) {
		t.Fatalf("after inner restore got %v", s.RC().Bounds())
	}
	s.Restore()
	if s.RC().Bounds() != geom.IRectLTRB(0, 0, 100, 80) {
		t.Fatalf("after outer restore got %v", s.RC().Bounds())
	}

	// Save with no clip then restore must be a no-op.
	s.Save()
	s.Restore()
	if s.RC().Bounds() != geom.IRectLTRB(0, 0, 100, 80) {
		t.Fatal("empty save/restore should not change the clip")
	}

	// ReplaceClip clamps to root bounds.
	s.ReplaceClip(geom.IRectLTRB(-10, -10, 30, 300))
	if s.RC().Bounds() != geom.IRectLTRB(0, 0, 30, 80) {
		t.Fatalf("replaceClip got %v", s.RC().Bounds())
	}
}

// TestAAClipBlitterReInitGrowsScratch: Init rebinds the blitter and clip but the scanline scratch is sized lazily, so
// re-Initing one instance with a wider clip must grow it. Sizing on nil-ness instead of length overran the buffers the
// narrow clip left behind (an index-out-of-range inside mergeAARuns/expandToRuns).
func TestAAClipBlitterReInitGrowsScratch(t *testing.T) {
	const wide = 64
	var narrowClip, wideClip AAClip
	if !narrowClip.SetRect(geom.IRectLTRB(0, 0, 4, 8)) || !wideClip.SetRect(geom.IRectLTRB(0, 0, wide, 8)) {
		t.Fatal("SetRect returned false")
	}
	color := colorcore.Color(0xFF3366CC)

	antiRun := func(width int32) (aa []Alpha, runs []int16) {
		aa = make([]Alpha, width+1)
		runs = make([]int16, width+1)
		aa[0] = 0x80
		runs[0] = int16(width)
		return aa, runs
	}

	// Reused instance: narrow clip first (which sizes the scratch), then the wide one.
	reused := NewPixmap(wide, 8)
	var b AAClipBlitter
	b.Init(NewSolidBlitter(reused, color), &narrowClip)
	narrowAA, narrowRuns := antiRun(4)
	b.BlitAntiH(0, 0, narrowAA, narrowRuns)
	b.Init(NewSolidBlitter(reused, color), &wideClip)
	wideAA, wideRuns := antiRun(wide)
	b.BlitAntiH(0, 1, wideAA, wideRuns)
	b.BlitH(0, 2, wide)

	// Fresh instance doing only the wide work.
	fresh := NewPixmap(wide, 8)
	var f AAClipBlitter
	f.Init(NewSolidBlitter(fresh, color), &narrowClip)
	narrowAA, narrowRuns = antiRun(4)
	f.BlitAntiH(0, 0, narrowAA, narrowRuns)
	f2 := &AAClipBlitter{}
	f2.Init(NewSolidBlitter(fresh, color), &wideClip)
	wideAA, wideRuns = antiRun(wide)
	f2.BlitAntiH(0, 1, wideAA, wideRuns)
	f2.BlitH(0, 2, wide)

	if !bytes.Equal(reused.RGBA8888Bytes(), fresh.RGBA8888Bytes()) {
		t.Fatal("a re-Inited AAClipBlitter did not match a freshly Inited one")
	}
	for x := int32(0); x < wide; x++ {
		if reused.Pix[reused.addr(x, 1)]>>24 == 0 {
			t.Fatalf("wide BlitAntiH left pixel (%d,1) untouched", x)
		}
	}
}
