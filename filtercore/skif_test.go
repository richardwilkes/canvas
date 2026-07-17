// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package filtercore

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

func TestRoundOutInTolerance(t *testing.T) {
	// kRoundEpsilon absorbs float error just past integers.
	r := geom.RectLTRB(9.9995, 10.0005, 20.0004, 29.9996)
	if got := RoundOut(r); got != geom.IRectLTRB(10, 10, 20, 30) {
		t.Fatalf("RoundOut = %v", got)
	}
	if got := RoundIn(r); got != geom.IRectLTRB(10, 10, 20, 30) {
		t.Fatalf("RoundIn = %v", got)
	}
	// Genuine fractions still round out/in.
	r = geom.RectLTRB(9.4, 10.6, 20.4, 29.6)
	if got := RoundOut(r); got != geom.IRectLTRB(9, 10, 21, 30) {
		t.Fatalf("RoundOut frac = %v", got)
	}
	if got := RoundIn(r); got != geom.IRectLTRB(10, 11, 20, 29) {
		t.Fatalf("RoundIn frac = %v", got)
	}
}

func TestMapIRectScaleTranslatePrecision(t *testing.T) {
	// The double-precision scale-translate lane preserves 1px precision where float math would wobble: 16777216 + 1 is
	// not representable in float32.
	var m geom.Matrix
	m.SetScaleTranslate(1, 1, 16777216, 0)
	r := geom.IRectLTRB(1, 0, 3, 2)
	got := mapIRect(&m, r)
	want := geom.IRectLTRB(16777217, 0, 16777219, 2)
	if got != want {
		t.Fatalf("mapIRect = %v, want %v", got, want)
	}
	// Negative scale sorts the edges.
	m.SetScaleTranslate(-2, 3, 10, -5)
	got = mapIRect(&m, geom.IRectLTRB(1, 2, 4, 5))
	want = geom.IRectLTRB(2, 1, 8, 10)
	if got != want {
		t.Fatalf("mapIRect neg = %v, want %v", got, want)
	}
	// Inverse of the same lane round-trips.
	inv, ok := inverseMapIRect(&m, got)
	if !ok || inv != geom.IRectLTRB(1, 2, 4, 5) {
		t.Fatalf("inverseMapIRect = %v ok=%v", inv, ok)
	}
}

func TestRelevantSubset(t *testing.T) {
	src := geom.IRectLTRB(0, 0, 10, 10)
	dst := geom.IRectLTRB(4, 4, 20, 20)
	if got := RelevantSubset(src, dst, shaders.TileDecal); got != geom.IRectLTRB(4, 4, 10, 10) {
		t.Fatalf("decal overlap = %v", got)
	}
	// Disjoint decal is empty; disjoint clamp takes the closest edge/corner.
	far := geom.IRectLTRB(30, 4, 40, 12)
	if got := RelevantSubset(src, far, shaders.TileDecal); !got.IsEmpty() {
		t.Fatalf("decal disjoint = %v", got)
	}
	if got := RelevantSubset(src, far, shaders.TileClamp); got != geom.IRectLTRB(9, 4, 10, 10) {
		t.Fatalf("clamp disjoint = %v", got)
	}
	corner := geom.IRectLTRB(30, 30, 40, 40)
	if got := RelevantSubset(src, corner, shaders.TileClamp); got != geom.IRectLTRB(9, 9, 10, 10) {
		t.Fatalf("clamp corner = %v", got)
	}
	// Periodic modes keep the whole source.
	if got := RelevantSubset(src, far, shaders.TileRepeat); got != src {
		t.Fatalf("repeat = %v", got)
	}
}

func TestPeriodicAxisTransform(t *testing.T) {
	crop := geom.IRectLTRB(0, 0, 10, 10)
	// Output fully inside one period to the right: repeat is a pure translation moving the image onto the visible
	// period.
	m, ok := periodicAxisTransform(shaders.TileRepeat, crop, geom.IRectLTRB(12, 2, 18, 8))
	if !ok {
		t.Fatalf("expected single-period repeat to simplify")
	}
	if p := m.MapPoint(geom.Point{X: 2, Y: 2}); p.X != 12 || p.Y != 2 {
		t.Fatalf("repeat maps (2,2) -> %v, want (12,2)", p)
	}
	// Mirror in an odd period reflects the image into the period.
	m, ok = periodicAxisTransform(shaders.TileMirror, crop, geom.IRectLTRB(12, 2, 18, 8))
	if !ok {
		t.Fatalf("expected single-period mirror to simplify")
	}
	if p := m.MapPoint(geom.Point{X: 2, Y: 2}); p.X != 18 || p.Y != 2 {
		t.Fatalf("mirror maps (2,2) -> %v, want (18,2)", p)
	}
	// An output spanning both crop edges cannot simplify.
	if _, ok = periodicAxisTransform(shaders.TileRepeat, crop, geom.IRectLTRB(-2, 0, 14, 8)); ok {
		t.Fatalf("multi-period tiling must not simplify")
	}
}

func TestIsNearlyIntegerTranslation(t *testing.T) {
	var m geom.Matrix
	m.SetTranslate(5.0004, -3.0007)
	var out geom.IPoint
	if !isNearlyIntegerTranslation(&m, &out) || out != (geom.IPoint{X: 5, Y: -3}) {
		t.Fatalf("nearly-integer translation not detected: %v", out)
	}
	m.SetTranslate(5.4, 0)
	if isNearlyIntegerTranslation(&m, nil) {
		t.Fatalf("fractional translation misdetected")
	}
	m.SetScaleTranslate(1.002, 1, 2, 2)
	if isNearlyIntegerTranslation(&m, nil) {
		t.Fatalf("scaled matrix misdetected")
	}
}

func TestDecomposeCTM(t *testing.T) {
	// Scale+translate CTMs stay whole in layer space for kScaleTranslate filters.
	var ctm geom.Matrix
	ctm.SetScaleTranslate(2, 3, 5, 7)
	var m Mapping
	if !m.DecomposeCTM(&ctm, MatrixCapabilityScaleTranslate, geom.Point{}) {
		t.Fatalf("decompose failed")
	}
	if lm := m.LayerMatrix(); lm != ctm {
		t.Fatalf("layer matrix = %v", lm)
	}
	if l2d := m.LayerToDevice(); !l2d.IsIdentity() {
		t.Fatalf("layer-to-device should be identity")
	}
	// A rotation with a kScaleTranslate filter factors into scale (layer) x rotation (device).
	ctm.SetRotate(30)
	ctm.PreScale(2, 2)
	if !m.DecomposeCTM(&ctm, MatrixCapabilityScaleTranslate, geom.Point{X: 10, Y: 10}) {
		t.Fatalf("decompose rotation failed")
	}
	lm := m.LayerMatrix()
	if lm.Get(geom.MSkewX) != 0 || lm.Get(geom.MSkewY) != 0 {
		t.Fatalf("layer matrix should be pure scale, got %v", lm)
	}
	if got := lm.Get(geom.MScaleX); got < 1.99 || got > 2.01 {
		t.Fatalf("layer scale = %g, want ~2", got)
	}
	// layerToDevice * layerMatrix == ctm.
	total := m.TotalMatrix()
	for i := range 9 {
		if d := absf32(total.Get(i) - ctm.Get(i)); d > 1e-4 {
			t.Fatalf("total[%d] = %g vs ctm %g", i, total.Get(i), ctm.Get(i))
		}
	}
	// kTranslate defers the whole CTM to the device side.
	if !m.DecomposeCTM(&ctm, MatrixCapabilityTranslate, geom.Point{}) {
		t.Fatalf("decompose translate failed")
	}
	if lm = m.LayerMatrix(); !lm.IsIdentity() {
		t.Fatalf("translate capability should give identity layer matrix")
	}
}

func TestQuadContainsRect(t *testing.T) {
	identity := geom.IdentityMatrix()
	a := geom.RectLTRB(0, 0, 10, 10)
	if !quadContainsRect(&identity, a, geom.RectLTRB(2, 2, 8, 8), 0) {
		t.Fatalf("identity containment failed")
	}
	if quadContainsRect(&identity, a, geom.RectLTRB(2, 2, 12, 8), 0) {
		t.Fatalf("overhang not detected")
	}
	// Rotated 45°, the quad's inscribed square shrinks.
	var rot geom.Matrix
	rot.SetRotatePivot(45, 5, 5)
	if !quadContainsRect(&rot, a, geom.RectLTRB(4, 4, 6, 6), 0) {
		t.Fatalf("rotated quad should contain the center square")
	}
	if quadContainsRect(&rot, a, geom.RectLTRB(1, 1, 9, 9), 0) {
		t.Fatalf("rotated quad cannot contain the large square")
	}
	if quadContainsRect(&identity, geom.Rect{}, geom.RectLTRB(0, 0, 0, 0), 0) {
		t.Fatalf("empty quad contains nothing")
	}
}

func TestMinMaxScales(t *testing.T) {
	var m geom.Matrix
	m.SetScaleTranslate(2, -3, 7, 8)
	mn, mx, ok := minMaxScales(&m)
	if !ok || mn != 2 || mx != 3 {
		t.Fatalf("scale minMax = %g,%g ok=%v", mn, mx, ok)
	}
	m.SetRotate(37)
	mn, mx, ok = minMaxScales(&m)
	if !ok || absf32(mn-1) > 1e-3 || absf32(mx-1) > 1e-3 {
		t.Fatalf("rotation minMax = %g,%g", mn, mx)
	}
}

func TestDownscaleStepCount(t *testing.T) {
	cases := []struct {
		scale float32
		want  int
	}{
		{scale: 1, want: 0},
		{scale: 0.9999, want: 0}, // near-identity collapse
		{scale: 0.6, want: 1},    // single sub-1/2 step... 1/0.6=1.67 ceil 2 -> 1 step; finalScale 0.6*1=0.6 < 0.9
		{scale: 0.5, want: 1},
		{scale: 0.26, want: 2},
		{scale: 0.25, want: 2},
		{scale: 0.24, want: 3}, // 1/0.24 = 4.17 ceil 5 -> nextLog2 3; finalScale 0.24*4 = 0.96 >= 0.9 -> 2
	}
	// Recompute expectations from the step-count formula:
	for _, c := range cases {
		got := downscaleStepCount(c.scale)
		switch c.scale {
		case 0.24:
			if got != 2 {
				t.Fatalf("steps(0.24) = %d, want 2 (final-step collapse)", got)
			}
		default:
			if got != c.want {
				t.Fatalf("steps(%g) = %d, want %d", c.scale, got, c.want)
			}
		}
	}
}

func TestBoxBlurWindowAndKernel(t *testing.T) {
	// window = floor(sigma * 3*sqrt(2π)/4 + 0.5)
	if w := boxBlurWindow(2); w != 4 {
		t.Fatalf("window(2) = %d, want 4", w)
	}
	if w := boxBlurWindow(10); w != 19 {
		t.Fatalf("window(10) = %d, want 19", w)
	}
	// Gaussian kernel is symmetric and normalized.
	kernel := make([]float32, 2*SigmaToRadius(1.5)+1)
	compute1DBlurKernel(1.5, SigmaToRadius(1.5), kernel)
	sum := float32(0)
	for i, k := range kernel {
		sum += k
		if k != kernel[len(kernel)-1-i] {
			t.Fatalf("kernel not symmetric at %d", i)
		}
	}
	if absf32(sum-1) > 1e-6 {
		t.Fatalf("kernel sum = %g", sum)
	}
}

func TestBlendModeAffectsTransparentBlack(t *testing.T) {
	// True unless the dst coefficient is One/ISA/ISC; advanced modes false.
	trueModes := []int32{0 /*clear*/, 1 /*src*/, 5 /*srcIn*/, 6 /*dstIn*/, 7 /*srcOut*/, 10 /*dstATop*/, 13 /*modulate*/}
	falseModes := []int32{2 /*dst*/, 3 /*srcOver*/, 4 /*dstOver*/, 8 /*dstOut*/, 9 /*srcATop*/, 11 /*xor*/, 12 /*plus*/, 14 /*screen*/, 15 /*overlay: advanced*/, 24 /*multiply: advanced*/}
	for _, m := range trueModes {
		if !blendModeAffectsTransparentBlack(raster.BlendMode(m)) {
			t.Fatalf("mode %d should affect transparent black", m)
		}
	}
	for _, m := range falseModes {
		if blendModeAffectsTransparentBlack(raster.BlendMode(m)) {
			t.Fatalf("mode %d should not affect transparent black", m)
		}
	}
}
