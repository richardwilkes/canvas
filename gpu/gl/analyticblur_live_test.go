// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the analytic rect/circle/rrect blur lanes (MakeRectBlur / MakeCircleBlur / MakeRRectBlur). A
// normal-blurred fill of an axis-aligned rect, a circle, or a simple-circular rrect now renders through a single
// textured coverage draw (an integral/profile/nine-patch-mask texture sampled by the corresponding blur FP) instead of
// the general render-mask + separable-convolution path. These tests confirm the analytic output is a true Gaussian
// blur: the rect lane is compared against the closed-form normal-distribution integral (erf) along a corner-free
// centerline, the circle lane is checked for radial symmetry and falloff, and the rrect lane is checked for symmetry
// plus the rounded-corner signature. Skips when no GL context is available.

package gl_test

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/maskfilter"
)

// TestLiveAnalyticRectBlurMatchesErf draws a wide/tall blurred rect (so the horizontal centerline near the right edge
// is far from the corners) through the analytic rect lane and compares the coverage against the closed-form blurred
// half-plane: coverage(x) = 0.5*(1 + erf((edge - center)/(sigma*sqrt2))).
func TestLiveAnalyticRectBlurMatchesErf(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size = 160
	const l, tp, r, b = 20.0, 20.0, 120.0, 140.0
	const sigma = 5.0
	data, w := drawBlurredRect(t, dc, size, l, tp, r, b, maskfilter.BlurNormal, sigma)

	const y = 80 // centerline, far from the top/bottom edges
	maxDiff := 0
	for x := int32(104); x <= 136; x++ {
		got := chanAt(data, w, x, y)
		// The pixel at column x samples at its center (x + 0.5); positive distance is inside.
		d := float64(r) - (float64(x) + 0.5)
		want := 0.5 * (1 + math.Erf(d/(float64(sigma)*math.Sqrt2)))
		wantB := int(want*255 + 0.5)
		diff := got - wantB
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("analytic rect blur vs erf: maxDiff=%d", maxDiff)
	// The integral texture is 8-bit with 2 texels/pixel and linear interpolation, so a handful of units of error is
	// expected; a large diff means the FP is not computing a Gaussian blur.
	if maxDiff > 6 {
		t.Errorf("analytic rect blur deviates from the Gaussian integral by %d (> 6)", maxDiff)
	}

	// Deep interior is fully covered and well outside is transparent.
	if c := chanAt(data, w, 60, 80); c < 250 {
		t.Errorf("interior coverage %d, want ~255", c)
	}
	if c := chanAt(data, w, 150, 80); c > 4 {
		t.Errorf("far-outside coverage %d, want ~0", c)
	}
}

// TestLiveAnalyticCircleBlur draws a blurred circle (an oval rrect) through the analytic circle lane and checks the
// radial structure: opaque center, four-way symmetric coverage at equal radial distances, monotone radial falloff, and
// a fuzzy edge that reaches zero far outside.
func TestLiveAnalyticCircleBlur(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size = 160
	const cx, cy, radius = 80.0, 80.0, 40.0
	const sigma = 6.0

	sdc := maskFilterDest(t, dc, size)
	paint := gl.NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	mf := maskfilter.NewBlur(maskfilter.BlurNormal, sigma, true)
	if mf == nil {
		t.Fatal("NewBlur returned nil")
	}
	circle := geom.MakeRRect(geom.RectLTRB(cx-radius, cy-radius, cx+radius, cy+radius), radius, radius)
	if circle.Type != geom.RRectOval {
		t.Fatalf("expected an oval rrect, got type %d", circle.Type)
	}
	shape := gl.MakeStyledShapeRRect(circle, gl.SimpleFillStyle(), gl.DoSimplifyYes)
	vm := geom.IdentityMatrix()
	gl.DrawShapeWithMaskFilter(sdc, nil, paint, &vm, mf, &shape)
	data := readSDC(t, sdc)

	center := chanAt(data, size, 80, 80)
	if center < 250 {
		t.Errorf("center coverage %d, want ~255", center)
	}

	// Four-way symmetry at equal radial distance, straddling the blurred rim. The circle center is at 80.0 (a pixel
	// boundary), so pixel centers (index+0.5) are symmetric about it when the +/- pixel indices sum to 159: pixel 122
	// (center 122.5, +42.5) pairs with pixel 37 (center 37.5, -42.5).
	e := chanAt(data, size, 122, 80)
	west := chanAt(data, size, 37, 80)
	n := chanAt(data, size, 80, 37)
	s := chanAt(data, size, 80, 122)
	t.Logf("circle rim samples E=%d W=%d N=%d S=%d", e, west, n, s)
	maxv, minv := e, e
	for _, v := range []int{west, n, s} {
		if v > maxv {
			maxv = v
		}
		if v < minv {
			minv = v
		}
	}
	if maxv-minv > 6 {
		t.Errorf("circle blur not radially symmetric: spread %d (E=%d W=%d N=%d S=%d)", maxv-minv,
			e, west, n, s)
	}
	if e <= 8 || e >= 250 {
		t.Errorf("rim coverage %d, want an intermediate blur value", e)
	}

	// Monotone radial falloff along +x: interior > rim > well-outside.
	inner := chanAt(data, size, 100, 80) // radius ~20, deep inside
	outer := chanAt(data, size, 132, 80) // radius ~52, well outside
	if inner <= e || e <= outer {
		t.Errorf("radial falloff not monotone: inner=%d rim=%d outer=%d", inner, e, outer)
	}
	if far := chanAt(data, size, 5, 5); far > 4 {
		t.Errorf("far corner coverage %d, want ~0", far)
	}
}

// TestLiveAnalyticRRectBlur draws a normal-blurred simple-circular rrect through the analytic rrect lane
// (MakeRRectBlur, the nine-patch blurred-rrect mask). It confirms the blur is correct and carries the rounded-corner
// signature: opaque center, four-way symmetry at the straight-edge rim, a receded (low-coverage) corner versus a
// well-covered straight edge at the same inset (a plain rect would cover both), and zero far outside.
func TestLiveAnalyticRRectBlur(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size = 160
	const l, tp, r, b = 30.0, 30.0, 130.0, 130.0
	const radius = 25.0
	const sigma = 5.0

	rr := geom.MakeRRect(geom.RectLTRB(l, tp, r, b), radius, radius)
	if rr.Type != geom.RRectSimple {
		t.Fatalf("expected a simple rrect, got type %d", rr.Type)
	}

	// The analytic rrect FP must build for this shape (identity CTM, so devRRect == srcRRect).
	if fp := gl.MakeRRectBlur(dc, sigma, sigma, rr, rr); fp == nil {
		t.Fatal("MakeRRectBlur returned nil for a nine-patchable simple-circular rrect")
	}

	sdc := maskFilterDest(t, dc, size)
	paint := gl.NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	mf := maskfilter.NewBlur(maskfilter.BlurNormal, sigma, true)
	if mf == nil {
		t.Fatal("NewBlur returned nil")
	}
	shape := gl.MakeStyledShapeRRect(rr, gl.SimpleFillStyle(), gl.DoSimplifyYes)
	vm := geom.IdentityMatrix()
	gl.DrawShapeWithMaskFilter(sdc, nil, paint, &vm, mf, &shape)
	data := readSDC(t, sdc)

	if c := chanAt(data, size, 80, 80); c < 250 {
		t.Errorf("center coverage %d, want ~255", c)
	}

	// Four-way symmetry at the straight-edge rim midpoints (~0.5px outside each edge; the mirror indices sum to 159,
	// i.e. reflect about the center at 80.0).
	e := chanAt(data, size, 130, 80)
	west := chanAt(data, size, 29, 80)
	n := chanAt(data, size, 80, 29)
	s := chanAt(data, size, 80, 130)
	t.Logf("rrect rim samples E=%d W=%d N=%d S=%d", e, west, n, s)
	maxv, minv := e, e
	for _, v := range []int{west, n, s} {
		if v > maxv {
			maxv = v
		}
		if v < minv {
			minv = v
		}
	}
	if maxv-minv > 6 {
		t.Errorf("rrect blur not symmetric: spread %d (E=%d W=%d N=%d S=%d)", maxv-minv, e, west, n, s)
	}
	if e <= 8 || e >= 250 {
		t.Errorf("rim coverage %d, want an intermediate blur value", e)
	}

	// Rounded-corner signature: a point just outside the rounded corner recedes to low coverage, while a point at the
	// same inset from a straight edge stays well covered.
	cornerOut := chanAt(data, size, 33, 33) // ~5px outside the radius-25 corner arc centered at (55,55)
	edgeIn := chanAt(data, size, 33, 80)    // ~3.5px inside the left straight edge
	t.Logf("rrect corner vs edge: cornerOut=%d edgeIn=%d", cornerOut, edgeIn)
	if edgeIn < 150 {
		t.Errorf("straight-edge interior coverage %d, want well covered (>150)", edgeIn)
	}
	if cornerOut > 100 {
		t.Errorf("rounded-corner coverage %d, want receded (<100)", cornerOut)
	}
	if cornerOut >= edgeIn {
		t.Errorf("rounded corner (%d) should be less covered than the straight edge (%d)", cornerOut,
			edgeIn)
	}

	if far := chanAt(data, size, 5, 5); far > 4 {
		t.Errorf("far corner coverage %d, want ~0", far)
	}
}

// TestLiveAnalyticBlurCaches confirms the profile/mask textures are memoized in the DirectContext's thread-safe cache:
// the first analytic blur of a given kind adds one entry (under its own domain), a repeat with identical parameters
// hits the cache without adding an entry, and a different sigma keys a fresh entry.
func TestLiveAnalyticBlurCaches(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	cache := dc.ThreadSafeCache()
	base := cache.NumEntries()

	circle := geom.RectLTRB(40, 40, 120, 120)
	rr := geom.MakeRRect(geom.RectLTRB(30, 30, 130, 130), 25, 25)

	if fp := gl.MakeCircleBlur(dc, circle, 6); fp == nil {
		t.Fatal("MakeCircleBlur returned nil")
	}
	if fp := gl.MakeRRectBlur(dc, 5, 5, rr, rr); fp == nil {
		t.Fatal("MakeRRectBlur returned nil")
	}
	// A circle-profile entry and an rrect-mask entry, under distinct domains.
	if got := cache.NumEntries() - base; got != 2 {
		t.Fatalf("entries added = %d, want 2 (circle profile + rrect mask)", got)
	}

	adds := cache.Stats().Adds
	hits := cache.Stats().Hits
	// Repeats with identical parameters reuse the cached uploads.
	if fp := gl.MakeCircleBlur(dc, circle, 6); fp == nil {
		t.Fatal("second MakeCircleBlur returned nil")
	}
	if fp := gl.MakeRRectBlur(dc, 5, 5, rr, rr); fp == nil {
		t.Fatal("second MakeRRectBlur returned nil")
	}
	if got := cache.NumEntries() - base; got != 2 {
		t.Fatalf("entries after repeats = %d, want 2 (cache reuse)", got)
	}
	if cache.Stats().Adds != adds {
		t.Fatalf("repeats added entries: adds %d -> %d", adds, cache.Stats().Adds)
	}
	if cache.Stats().Hits != hits+2 {
		t.Fatalf("repeats produced %d hits, want 2", cache.Stats().Hits-hits)
	}

	// A different sigma is a different profile key -> a fresh entry.
	if fp := gl.MakeCircleBlur(dc, circle, 12); fp == nil {
		t.Fatal("MakeCircleBlur with a new sigma returned nil")
	}
	if got := cache.NumEntries() - base; got != 3 {
		t.Fatalf("entries after new sigma = %d, want 3", got)
	}
}

// TestLiveAnalyticBlurCacheOutputIdentical draws the same blurred circle twice through the device mask lane; the first
// draw creates and caches the profile, the second reuses it. The rendered buffers must be byte-identical, proving the
// cached texture yields the same result as a freshly generated one.
func TestLiveAnalyticBlurCacheOutputIdentical(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const size = 160
	drawCircle := func() []byte {
		sdc := maskFilterDest(t, dc, size)
		paint := gl.NewPaint()
		paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
		mf := maskfilter.NewBlur(maskfilter.BlurNormal, 6, true)
		if mf == nil {
			t.Fatal("NewBlur returned nil")
		}
		circle := geom.MakeRRect(geom.RectLTRB(40, 40, 120, 120), 40, 40)
		shape := gl.MakeStyledShapeRRect(circle, gl.SimpleFillStyle(), gl.DoSimplifyYes)
		vm := geom.IdentityMatrix()
		gl.DrawShapeWithMaskFilter(sdc, nil, paint, &vm, mf, &shape)
		return readSDC(t, sdc)
	}

	first := drawCircle()  // cache miss: creates and caches the circle profile
	second := drawCircle() // cache hit: reuses the cached profile
	if len(first) != len(second) {
		t.Fatalf("buffer lengths differ: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("cached blur output differs from the fresh blur at byte %d: %d vs %d", i,
				first[i], second[i])
		}
	}
}
