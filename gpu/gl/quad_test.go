// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for quad classification, bounds (incl. projected), AA-type resolution, rect cropping, w=0 clipping, and
// the TessellationHelper inset/outset math against hand-computed values.

package gl

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

func rectQuadPoints(r geom.Rect) [4]geom.Point {
	// Rect-to-quad point order: TL, TR, BR, BL.
	return [4]geom.Point{
		{X: r.Left, Y: r.Top},
		{X: r.Right, Y: r.Top},
		{X: r.Right, Y: r.Bottom},
		{X: r.Left, Y: r.Bottom},
	}
}

func TestQuadClassification(t *testing.T) {
	r := geom.Rect{Left: 10, Top: 10, Right: 30, Bottom: 20}

	identity := geom.IdentityMatrix()
	if got := MakeQuadFromRect(r, &identity); got.QuadType() != QuadTypeAxisAligned {
		t.Fatalf("identity: got %v, want axis-aligned", got.QuadType())
	}
	st := geom.Matrix{}
	st.SetScaleTranslate(2, 3, 5, 7)
	if got := MakeQuadFromRect(r, &st); got.QuadType() != QuadTypeAxisAligned {
		t.Fatalf("scale-translate: got %v, want axis-aligned", got.QuadType())
	}
	rot90 := geom.RotateDegMatrix(90)
	if got := MakeQuadFromRect(r, &rot90); got.QuadType() != QuadTypeAxisAligned {
		t.Fatalf("rotate 90: got %v, want axis-aligned (rect stays rect)", got.QuadType())
	}
	rot45 := geom.RotateDegMatrix(45)
	if got := MakeQuadFromRect(r, &rot45); got.QuadType() != QuadTypeRectilinear {
		t.Fatalf("rotate 45: got %v, want rectilinear", got.QuadType())
	}
	skew := geom.Matrix{}
	skew.SetSkew(0.5, 0)
	if got := MakeQuadFromRect(r, &skew); got.QuadType() != QuadTypeGeneral {
		t.Fatalf("skew: got %v, want general", got.QuadType())
	}
	persp := geom.IdentityMatrix()
	persp.Set(geom.MPersp0, 0.001)
	if got := MakeQuadFromRect(r, &persp); got.QuadType() != QuadTypePerspective {
		t.Fatalf("perspective: got %v, want perspective", got.QuadType())
	}

	// MakeQuadFromPoints: rect-shaped points inherit the matrix's class; arbitrary points are general.
	pts := rectQuadPoints(r)
	if got := MakeQuadFromPoints(&pts, &rot45); got.QuadType() != QuadTypeRectilinear {
		t.Fatalf("sk quad rect points rotate 45: got %v, want rectilinear", got.QuadType())
	}
	arb := [4]geom.Point{{X: 0, Y: 0}, {X: 10, Y: 1}, {X: 11, Y: 12}, {X: -1, Y: 9}}
	if got := MakeQuadFromPoints(&arb, &identity); got.QuadType() != QuadTypeGeneral {
		t.Fatalf("arbitrary points: got %v, want general", got.QuadType())
	}

	// Tri-strip vertex order: TL, BL, TR, BR.
	q := MakeQuadFromRect(r, &identity)
	wantPts := [4]geom.Point{{X: 10, Y: 10}, {X: 10, Y: 20}, {X: 30, Y: 10}, {X: 30, Y: 20}}
	for i := range 4 {
		if q.Point(i) != wantPts[i] {
			t.Fatalf("vertex %d = %v, want %v", i, q.Point(i), wantPts[i])
		}
	}
}

func TestQuadBoundsAsRectAndFinite(t *testing.T) {
	r := geom.Rect{Left: 10, Top: 10, Right: 30, Bottom: 20}
	identity := geom.IdentityMatrix()
	q := MakeQuadFromRect(r, &identity)
	if got := q.Bounds(); got != r {
		t.Fatalf("bounds = %v, want %v", got, r)
	}
	if rect, ok := q.AsRect(); !ok || rect != r {
		t.Fatalf("asRect = %v/%v, want %v/true", rect, ok, r)
	}

	// A 90-degree rotation stays axis-aligned and fills its bounds, but v0 moves so asRect reports false.
	rot90 := geom.RotateDegMatrix(90)
	q90 := MakeQuadFromRect(r, &rot90)
	if _, ok := q90.AsRect(); ok {
		t.Fatal("asRect should be false after a 90-degree rotation")
	}

	// Projected bounds with uniform w: coordinates divide through.
	persp := geom.IdentityMatrix()
	persp.Set(geom.MPersp2, 2)
	qp := MakeQuadFromRect(r, &persp)
	want := geom.Rect{Left: 5, Top: 5, Right: 15, Bottom: 10}
	if got := qp.Bounds(); !nearRect(got, want, 1e-4) {
		t.Fatalf("projected bounds = %v, want %v", got, want)
	}

	if !q.IsFinite() {
		t.Fatal("finite quad reported non-finite")
	}
	bad := q
	bad.Xs()[2] = float32(math.Inf(1))
	if bad.IsFinite() {
		t.Fatal("non-finite quad reported finite")
	}
}

func nearRect(a, b geom.Rect, tol float32) bool {
	near := func(x, y float32) bool { d := x - y; return d < tol && d > -tol }
	return near(a.Left, b.Left) && near(a.Top, b.Top) && near(a.Right, b.Right) &&
		near(a.Bottom, b.Bottom)
}

func TestResolveAAType(t *testing.T) {
	identity := geom.IdentityMatrix()
	intRect := MakeQuadFromRect(geom.Rect{Left: 5, Top: 5, Right: 10, Bottom: 10}, &identity)
	fracRect := MakeQuadFromRect(geom.Rect{Left: 5.5, Top: 5, Right: 10, Bottom: 10}, &identity)

	// Coverage + no-edge flags disables AA entirely.
	aa, flags := ResolveAAType(gpu.AATypeCoverage, gpu.QuadAAFlagsNone, &fracRect)
	if aa != gpu.AATypeNone || flags != gpu.QuadAAFlagsNone {
		t.Fatalf("coverage/none: got %v/%v", aa, flags)
	}
	// Coverage on an integer-aligned rect disables AA and the flags.
	aa, flags = ResolveAAType(gpu.AATypeCoverage, gpu.QuadAAFlagsAll, &intRect)
	if aa != gpu.AATypeNone || flags != gpu.QuadAAFlagsNone {
		t.Fatalf("coverage/int rect: got %v/%v", aa, flags)
	}
	// Coverage on a fractional rect is kept.
	aa, flags = ResolveAAType(gpu.AATypeCoverage, gpu.QuadAAFlagsAll, &fracRect)
	if aa != gpu.AATypeCoverage || flags != gpu.QuadAAFlagsAll {
		t.Fatalf("coverage/frac rect: got %v/%v", aa, flags)
	}
	// Only the left edge is fractional: partial flags survive.
	aa, flags = ResolveAAType(gpu.AATypeCoverage, gpu.QuadAAFlagLeft|gpu.QuadAAFlagTop,
		&fracRect)
	if aa != gpu.AATypeCoverage || flags != gpu.QuadAAFlagLeft|gpu.QuadAAFlagTop {
		t.Fatalf("coverage/partial: got %v/%v", aa, flags)
	}
	// None clears the edge flags; MSAA forces all.
	if aa, flags = ResolveAAType(gpu.AATypeNone, gpu.QuadAAFlagsAll, &fracRect); aa !=
		gpu.AATypeNone || flags != gpu.QuadAAFlagsNone {
		t.Fatalf("none: got %v/%v", aa, flags)
	}
	if aa, flags = ResolveAAType(gpu.AATypeMSAA, gpu.QuadAAFlagLeft, &fracRect); aa !=
		gpu.AATypeMSAA || flags != gpu.QuadAAFlagsAll {
		t.Fatalf("msaa: got %v/%v", aa, flags)
	}
}

func TestCropToRectSimple(t *testing.T) {
	identity := geom.IdentityMatrix()
	quad := DrawQuad{
		Device:    MakeQuadFromRect(geom.Rect{Left: 0, Top: 0, Right: 20, Bottom: 10}, &identity),
		Local:     MakeQuadFromRectNoTransform(geom.Rect{Left: 0, Top: 0, Right: 1, Bottom: 1}),
		EdgeFlags: gpu.QuadAAFlagsNone,
	}
	crop := geom.Rect{Left: 5, Top: 0, Right: 30, Bottom: 8}
	if !CropToRect(crop, gpu.AAYes, &quad, true) {
		t.Fatal("axis-aligned crop must succeed")
	}
	wantDev := geom.Rect{Left: 5, Top: 0, Right: 20, Bottom: 8}
	if got, ok := quad.Device.AsRect(); !ok || got != wantDev {
		t.Fatalf("cropped device = %v/%v, want %v", got, ok, wantDev)
	}
	// Local coordinates shift proportionally: x by 5/20, y bottom to 8/10.
	wantLocal := geom.Rect{Left: 0.25, Top: 0, Right: 1, Bottom: 0.8}
	if got, ok := quad.Local.AsRect(); !ok || !nearRect(got, wantLocal, 1e-6) {
		t.Fatalf("cropped local = %v/%v, want %v", got, ok, wantLocal)
	}
	// Only the clipped edges (left, bottom) gained AA.
	if quad.EdgeFlags != gpu.QuadAAFlagLeft|gpu.QuadAAFlagBottom {
		t.Fatalf("edge flags = %v, want left|bottom", quad.EdgeFlags)
	}
}

func TestCropToRectGeneralContains(t *testing.T) {
	// A large rotated quad fully contains the crop rect, so the draw is reduced to exactly the crop rect with all edges
	// clipped.
	rot45 := geom.RotateDegMatrix(45)
	quad := DrawQuad{
		Device: MakeQuadFromRect(geom.Rect{Left: -100, Top: -100, Right: 100, Bottom: 100},
			&rot45),
		EdgeFlags: gpu.QuadAAFlagsNone,
	}
	crop := geom.Rect{Left: -20, Top: -20, Right: 20, Bottom: 20}
	if !CropToRect(crop, gpu.AAYes, &quad, false) {
		t.Fatal("containing crop must succeed")
	}
	if quad.Device.QuadType() != QuadTypeAxisAligned {
		t.Fatalf("cropped quad type = %v, want axis-aligned", quad.Device.QuadType())
	}
	if got, ok := quad.Device.AsRect(); !ok || got != crop {
		t.Fatalf("cropped device = %v/%v, want %v", got, ok, crop)
	}
	if quad.EdgeFlags != gpu.QuadAAFlagsAll {
		t.Fatalf("edge flags = %v, want all", quad.EdgeFlags)
	}

	// A quad that only partially covers the crop cannot be reduced.
	partial := DrawQuad{
		Device: MakeQuadFromRect(geom.Rect{Left: -5, Top: -5, Right: 5, Bottom: 5}, &rot45),
	}
	if CropToRect(crop, gpu.AAYes, &partial, false) {
		t.Fatal("partial general crop should fail")
	}
}

func perspQuad(t *testing.T, r geom.Rect, p0, p1 float32) Quad {
	t.Helper()
	m := geom.IdentityMatrix()
	m.Set(geom.MPersp0, p0)
	m.Set(geom.MPersp1, p1)
	q := MakeQuadFromRect(r, &m)
	if q.QuadType() != QuadTypePerspective {
		t.Fatal("expected a perspective quad")
	}
	return q
}

func TestClipToW0(t *testing.T) {
	local := MakeQuadFromRectNoTransform(geom.Rect{Right: 1, Bottom: 1})

	// Non-perspective: nothing to do.
	identity := geom.IdentityMatrix()
	flat := DrawQuad{
		Device: MakeQuadFromRect(geom.Rect{Right: 10, Bottom: 10}, &identity),
		Local:  local, EdgeFlags: gpu.QuadAAFlagsAll,
	}
	var extra DrawQuad
	if got := ClipToW0(&flat, &extra); got != 1 {
		t.Fatalf("non-perspective: got %d, want 1", got)
	}

	// All w valid: unchanged.
	valid := DrawQuad{
		Device: perspQuad(t, geom.Rect{Right: 4, Bottom: 4}, 0.01, 0),
		Local:  local, EdgeFlags: gpu.QuadAAFlagsAll,
	}
	if got := ClipToW0(&valid, &extra); got != 1 {
		t.Fatalf("all valid: got %d, want 1", got)
	}

	// All behind: w = 1 - x over x in [100, 200] makes every w negative.
	behind := DrawQuad{
		Device: perspQuad(t,
			geom.Rect{Left: 100, Top: 0, Right: 200, Bottom: 10}, -1, 0), Local: local,
		EdgeFlags: gpu.QuadAAFlagsAll,
	}
	if got := ClipToW0(&behind, &extra); got != 0 {
		t.Fatalf("all behind: got %d, want 0", got)
	}

	// Two vertices behind (w = 1 - 0.1x over x in [0, 30]): still one quad.
	two := DrawQuad{
		Device: perspQuad(t, geom.Rect{Right: 30, Bottom: 10}, -0.1, 0),
		Local:  local, EdgeFlags: gpu.QuadAAFlagsAll,
	}
	if got := ClipToW0(&two, &extra); got != 1 {
		t.Fatalf("two behind: got %d, want 1", got)
	}
	for i := range 4 {
		if two.Device.W(i) < quadW0PlaneDistance-1e-7 {
			t.Fatalf("clipped w[%d] = %v still behind the plane", i, two.Device.W(i))
		}
	}

	// One vertex behind (w = 1 - 0.05x - 0.05y over [0,15]^2, only (15,15) negative): a pentagon split into two quads,
	// each dropping the AA flag on the new shared edge.
	one := DrawQuad{
		Device: perspQuad(t, geom.Rect{Right: 15, Bottom: 15}, -0.05, -0.05),
		Local:  local, EdgeFlags: gpu.QuadAAFlagsAll,
	}
	if got := ClipToW0(&one, &extra); got != 2 {
		t.Fatalf("one behind: got %d, want 2", got)
	}
	if one.EdgeFlags == gpu.QuadAAFlagsAll || extra.EdgeFlags == gpu.QuadAAFlagsAll {
		t.Fatal("the split quads must drop the AA flag on the new interior edge")
	}
	for i := range 4 {
		if one.Device.W(i) < quadW0PlaneDistance-1e-7 ||
			extra.Device.W(i) < quadW0PlaneDistance-1e-7 {
			t.Fatal("split vertices must be in front of the w=0 plane")
		}
	}
}

func TestWillUseHairline(t *testing.T) {
	identity := geom.IdentityMatrix()
	thin := MakeQuadFromRect(geom.Rect{Left: 10, Top: 10, Right: 10.4, Bottom: 20}, &identity)
	wide := MakeQuadFromRect(geom.Rect{Left: 10, Top: 10, Right: 30, Bottom: 20}, &identity)

	if !WillUseHairline(&thin, gpu.AATypeCoverage, gpu.QuadAAFlagsAll) {
		t.Fatal("subpixel-wide rect should hairline")
	}
	if WillUseHairline(&wide, gpu.AATypeCoverage, gpu.QuadAAFlagsAll) {
		t.Fatal("wide rect should not hairline")
	}
	// Mixed edge flags and non-coverage AA never hairline.
	if WillUseHairline(&thin, gpu.AATypeCoverage, gpu.QuadAAFlagLeft) {
		t.Fatal("mixed edge flags should not hairline")
	}
	if WillUseHairline(&thin, gpu.AATypeMSAA, gpu.QuadAAFlagsAll) {
		t.Fatal("msaa should not hairline")
	}

	// The rectilinear lane goes through the tessellation helper.
	rot45 := geom.RotateDegMatrix(45)
	thinRot := MakeQuadFromRect(geom.Rect{Left: 10, Top: 10, Right: 10.4, Bottom: 20}, &rot45)
	if !WillUseHairline(&thinRot, gpu.AATypeCoverage, gpu.QuadAAFlagsAll) {
		t.Fatal("rotated subpixel rect should hairline")
	}
}

func TestTessellationHelperInsetOutset(t *testing.T) {
	identity := geom.IdentityMatrix()
	device := MakeQuadFromRect(geom.Rect{Left: 10, Top: 10, Right: 30, Bottom: 20}, &identity)
	local := MakeQuadFromRectNoTransform(geom.Rect{Right: 1, Bottom: 1})

	var h TessellationHelper
	h.Reset(&device, &local)

	devInset := device
	localInset := local
	coverage := h.Inset(f4Splat(0.5), &devInset, &localInset)
	for i := range 4 {
		if coverage[i] != 1 {
			t.Fatalf("full-size inset coverage[%d] = %v, want 1", i, coverage[i])
		}
	}
	wantDev := geom.Rect{Left: 10.5, Top: 10.5, Right: 29.5, Bottom: 19.5}
	if got, ok := devInset.AsRect(); !ok || !nearRect(got, wantDev, 1e-5) {
		t.Fatalf("inset device = %v/%v, want %v", got, ok, wantDev)
	}
	// Local coordinates move proportionally: 0.5/20 horizontally, 0.5/10 vertically.
	wantLocal := geom.Rect{Left: 0.025, Top: 0.05, Right: 0.975, Bottom: 0.95}
	if got, ok := localInset.AsRect(); !ok || !nearRect(got, wantLocal, 1e-6) {
		t.Fatalf("inset local = %v/%v, want %v", got, ok, wantLocal)
	}

	devOutset := device
	localOutset := local
	h.Outset(f4Splat(0.5), &devOutset, &localOutset)
	wantDev = geom.Rect{Left: 9.5, Top: 9.5, Right: 30.5, Bottom: 20.5}
	if got, ok := devOutset.AsRect(); !ok || !nearRect(got, wantDev, 1e-5) {
		t.Fatalf("outset device = %v/%v, want %v", got, ok, wantDev)
	}
	wantLocal = geom.Rect{Left: -0.025, Top: -0.05, Right: 1.025, Bottom: 1.05}
	if got, ok := localOutset.AsRect(); !ok || !nearRect(got, wantLocal, 1e-6) {
		t.Fatalf("outset local = %v/%v, want %v", got, ok, wantLocal)
	}

	// Edge lengths are ordered LBTR.
	lengths := h.GetEdgeLengths()
	want := [4]float32{10, 20, 20, 10}
	for i := range 4 {
		if diff := lengths[i] - want[i]; diff > 1e-4 || diff < -1e-4 {
			t.Fatalf("edge length[%d] = %v, want %v", i, lengths[i], want[i])
		}
	}
}

func TestTessellationHelperDegenerateInset(t *testing.T) {
	// A 0.4px-wide rect collapses under a 0.5px inset: the returned coverage estimates the subpixel area (width 0.4 ->
	// ~0.4 coverage).
	identity := geom.IdentityMatrix()
	device := MakeQuadFromRect(geom.Rect{Left: 10, Top: 10, Right: 10.4, Bottom: 20}, &identity)

	var h TessellationHelper
	h.Reset(&device, nil)
	if !h.IsSubpixel() {
		t.Fatal("0.4px-wide quad should be subpixel")
	}

	devInset := device
	coverage := h.Inset(f4Splat(0.5), &devInset, nil)
	for i := range 4 {
		if coverage[i] <= 0.35 || coverage[i] >= 0.45 {
			t.Fatalf("degenerate inset coverage[%d] = %v, want ~0.4", i, coverage[i])
		}
	}
	// The inset geometry collapsed to the vertical center line x = 10.2.
	for i := range 4 {
		if diff := devInset.X(i) - 10.2; diff > 1e-3 || diff < -1e-3 {
			t.Fatalf("collapsed x[%d] = %v, want 10.2", i, devInset.X(i))
		}
	}
}
