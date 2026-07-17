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
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

const rgnGrid = 96

// rasterizeRegion renders a region into a boolean grid via its span visitor.
func rasterizeRegion(rgn *Region) []bool {
	out := make([]bool, rgnGrid*rgnGrid)
	visitRegionSpans(rgn, func(r geom.IRect) {
		for y := max(r.Top, 0); y < min(r.Bottom, rgnGrid); y++ {
			for x := max(r.Left, 0); x < min(r.Right, rgnGrid); x++ {
				out[y*rgnGrid+x] = true
			}
		}
	})
	return out
}

func randRect(rng *rand.Rand) geom.IRect {
	x := int32(rng.Intn(rgnGrid))
	y := int32(rng.Intn(rgnGrid))
	w := int32(rng.Intn(rgnGrid / 2))
	h := int32(rng.Intn(rgnGrid / 2))
	return geom.IRectLTRB(x, y, min(x+w, rgnGrid), min(y+h, rgnGrid))
}

func randRegion(rng *rand.Rand) (region *Region, gridBits []bool) {
	rgn := &Region{}
	bits := make([]bool, rgnGrid*rgnGrid)
	n := 1 + rng.Intn(5)
	for i := 0; i < n; i++ {
		r := randRect(rng)
		rgn.OpRect(r, RegionUnion)
		for y := r.Top; y < r.Bottom; y++ {
			for x := r.Left; x < r.Right; x++ {
				bits[y*rgnGrid+x] = true
			}
		}
	}
	return rgn, bits
}

// TestRegionOpsVsBitmap: the boolean op machine against per-pixel boolean reference sets, plus
// containment/intersection/iteration consistency.
func TestRegionOpsVsBitmap(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	ops := []struct {
		fn func(a, b bool) bool
		op RegionOp
	}{
		{op: RegionDifference, fn: func(a, b bool) bool { return a && !b }},
		{op: RegionIntersect, fn: func(a, b bool) bool { return a && b }},
		{op: RegionUnion, fn: func(a, b bool) bool { return a || b }},
		{op: RegionXor, fn: func(a, b bool) bool { return a != b }},
		{op: RegionReverseDifference, fn: func(a, b bool) bool { return b && !a }},
		{op: RegionReplace, fn: func(_, b bool) bool { return b }},
	}
	for iter := 0; iter < 200; iter++ {
		ra, ba := randRegion(rng)
		rb, bb := randRegion(rng)
		for _, tc := range ops {
			res := &Region{}
			res.OpRegion(ra, rb, tc.op)
			got := rasterizeRegion(res)
			for i := range got {
				if got[i] != tc.fn(ba[i], bb[i]) {
					t.Fatalf("iter %d op %d: pixel (%d,%d) got %v", iter, tc.op,
						i%rgnGrid, i/rgnGrid, got[i])
				}
			}
			// bounds must be tight
			bounds := res.Bounds()
			if !res.IsEmpty() {
				onEdge := false
				for x := bounds.Left; x < bounds.Right && !onEdge; x++ {
					onEdge = got[int(bounds.Top)*rgnGrid+int(x)] || got[int(bounds.Bottom-1)*rgnGrid+int(x)]
				}
				if !onEdge {
					t.Fatalf("iter %d op %d: loose vertical bounds %v", iter, tc.op, bounds)
				}
			}
		}
		// point containment + spanerator consistency on a row
		res := &Region{}
		res.OpRegion(ra, rb, RegionXor)
		bits := rasterizeRegion(res)
		for i := 0; i < 50; i++ {
			x := int32(rng.Intn(rgnGrid))
			y := int32(rng.Intn(rgnGrid))
			if res.ContainsPoint(x, y) != bits[y*rgnGrid+x] {
				t.Fatalf("iter %d: ContainsPoint(%d,%d) mismatch", iter, x, y)
			}
		}
		y := int32(rng.Intn(rgnGrid))
		row := make([]bool, rgnGrid)
		span := NewRegionSpanerator(res, y, 0, rgnGrid)
		for {
			l, r, ok := span.Next()
			if !ok {
				break
			}
			for x := l; x < r; x++ {
				row[x] = true
			}
		}
		for x := 0; x < rgnGrid; x++ {
			if row[x] != bits[int(y)*rgnGrid+x] {
				t.Fatalf("iter %d: spanerator row %d x %d mismatch", iter, y, x)
			}
		}
	}
}

// TestRegionSetPathMatchesFill: a region built from a path must contain exactly the pixels the non-AA fill of that path
// covers.
func TestRegionSetPathMatchesFill(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	clip := NewRegionRect(geom.IRectLTRB(0, 0, rgnGrid, rgnGrid))
	for iter := 0; iter < 60; iter++ {
		p := randomTestPath(rng, rgnGrid, rgnGrid, iter%2 == 1)
		p.SetFillType(path.FillType(rng.Intn(4)))

		rgn := &Region{}
		rgn.SetPath(p, clip)
		got := rasterizeRegion(rgn)

		dev := NewPixmap(rgnGrid, rgnGrid)
		FillPath(p, geom.IRectLTRB(0, 0, rgnGrid, rgnGrid), NewSolidBlitter(dev, colorcore.Color(0xFF000000)))
		for y := int32(0); y < rgnGrid; y++ {
			for x := int32(0); x < rgnGrid; x++ {
				want := dev.Pix[dev.addr(x, y)]>>24 != 0
				if got[y*rgnGrid+x] != want {
					t.Fatalf("iter %d: (%d,%d) region %v, fill %v", iter, x, y, got[y*rgnGrid+x], want)
				}
			}
		}
	}
}

// TestRegionClippedFills: filling through a complex region clip must equal the fill clipped to the region's *bounds
// rect*, masked by the region — the RgnClipBlitter intersects blits with the region exactly, while the edge clipping
// (shared with the rect-clipped fill) may shift boundary rounding relative to an unclipped fill. Skipped when the path
// fits inside the region bounds (there the rect-clip route skips edge clipping and the two references diverge; the
// oracle probes cover that case against the C library directly).
func TestRegionClippedFills(t *testing.T) {
	rng := rand.New(rand.NewSource(321))
	for iter := 0; iter < 30; iter++ {
		clipRgn, clipBits := randRegion(rng)
		if clipRgn.IsEmpty() {
			continue
		}
		p := randomTestPath(rng, rgnGrid, rgnGrid, iter%2 == 1)
		p.SetFillType(path.FillType(rng.Intn(2))) // winding / even-odd
		if clipRgn.Bounds().ContainsRect(p.Bounds().RoundOut().Inset(-2, -2)) {
			continue
		}
		for _, aa := range []bool{false, true} {
			ref := NewPixmap(rgnGrid, rgnGrid)
			clipped := NewPixmap(rgnGrid, rgnGrid)
			if aa {
				AntiFillPath(p, clipRgn.Bounds(), NewSolidBlitter(ref, colorcore.Color(0xFF000000)))
				AntiFillPathRegion(p, clipRgn, NewSolidBlitter(clipped, colorcore.Color(0xFF000000)))
			} else {
				FillPath(p, clipRgn.Bounds(), NewSolidBlitter(ref, colorcore.Color(0xFF000000)))
				FillPathRegion(p, clipRgn, NewSolidBlitter(clipped, colorcore.Color(0xFF000000)))
			}
			for y := int32(0); y < rgnGrid; y++ {
				for x := int32(0); x < rgnGrid; x++ {
					want := uint32(0)
					if clipBits[y*rgnGrid+x] {
						want = ref.Pix[ref.addr(x, y)]
					}
					if got := clipped.Pix[clipped.addr(x, y)]; got != want {
						t.Fatalf("iter %d aa=%v: (%d,%d) got %08x want %08x",
							iter, aa, x, y, got, want)
					}
				}
			}
		}
	}
}

// TestRegionTranslateAndEqual: translate shifts every span; Equal detects equality.
func TestRegionTranslateAndEqual(t *testing.T) {
	rng := rand.New(rand.NewSource(555))
	for iter := 0; iter < 40; iter++ {
		rgn, bits := randRegion(rng)
		cp := &Region{}
		cp.SetRegion(rgn)
		if !cp.Equal(rgn) {
			t.Fatal("copy not equal")
		}
		dx := int32(rng.Intn(21) - 10)
		dy := int32(rng.Intn(21) - 10)
		cp.Translate(dx, dy)
		got := make([]bool, rgnGrid*rgnGrid)
		visitRegionSpans(cp, func(r geom.IRect) {
			for y := max(r.Top, 0); y < min(r.Bottom, rgnGrid); y++ {
				for x := max(r.Left, 0); x < min(r.Right, rgnGrid); x++ {
					got[y*rgnGrid+x] = true
				}
			}
		})
		for y := int32(0); y < rgnGrid; y++ {
			for x := int32(0); x < rgnGrid; x++ {
				sx, sy := x-dx, y-dy
				want := sx >= 0 && sx < rgnGrid && sy >= 0 && sy < rgnGrid && bits[sy*rgnGrid+sx]
				if got[y*rgnGrid+x] != want {
					t.Fatalf("iter %d: translate (%d,%d) mismatch at (%d,%d)", iter, dx, dy, x, y)
				}
			}
		}
		if !rgn.IsEmpty() {
			if cp2 := (&Region{}); true {
				cp2.SetRegion(rgn)
				cp2.Translate(dx, dy)
				cp2.Translate(-dx, -dy)
				if !cp2.Equal(rgn) {
					t.Fatalf("iter %d: round-trip translate not equal", iter)
				}
			}
		}
	}
}

// TestRegionContainsIntersects: rect/region containment and intersection against the bitmaps.
func TestRegionContainsIntersects(t *testing.T) {
	rng := rand.New(rand.NewSource(777))
	for iter := 0; iter < 100; iter++ {
		rgn, bits := randRegion(rng)
		r := randRect(rng)
		allIn := !r.IsEmpty()
		anyIn := false
		for y := r.Top; y < r.Bottom; y++ {
			for x := r.Left; x < r.Right; x++ {
				if bits[y*rgnGrid+x] {
					anyIn = true
				} else {
					allIn = false
				}
			}
		}
		if r.IsEmpty() {
			allIn = false
		}
		if got := rgn.ContainsRect(r); got != allIn {
			t.Fatalf("iter %d: ContainsRect(%v) = %v, want %v", iter, r, got, allIn)
		}
		if got := rgn.Intersects(r); got != anyIn {
			t.Fatalf("iter %d: Intersects(%v) = %v, want %v", iter, r, got, anyIn)
		}
		other, otherBits := randRegion(rng)
		wantInt := false
		wantContains := !other.IsEmpty() && !rgn.IsEmpty()
		for i := range bits {
			if bits[i] && otherBits[i] {
				wantInt = true
			}
			if otherBits[i] && !bits[i] {
				wantContains = false
			}
		}
		if rgn.IsEmpty() || other.IsEmpty() {
			wantInt = false
			wantContains = false
		}
		if got := rgn.IntersectsRegion(other); got != wantInt {
			t.Fatalf("iter %d: IntersectsRegion = %v, want %v", iter, got, wantInt)
		}
		if got := rgn.ContainsRegion(other); got != wantContains {
			t.Fatalf("iter %d: ContainsRegion = %v, want %v", iter, got, wantContains)
		}
	}
}
