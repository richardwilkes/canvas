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

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
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

// TestPeriodicAxisTransformNegativePeriods covers outputs that sit left of / above the crop, where the period indices
// are negative. Parity has to be read off the integer period: a signed remainder reports -1 for an odd negative period,
// which silently dropped the mirror flip and rendered that tile as a plain translation.
func TestPeriodicAxisTransformNegativePeriods(t *testing.T) {
	crop := geom.IRectLTRB(0, 0, 10, 10)
	// Period -1 on X: the tile immediately left of the crop is mirrored, so image x=2 lands at layer x=-2 (matching
	// what shaders' exclusiveMirror computes for the same tiling).
	m, ok := periodicAxisTransform(shaders.TileMirror, crop, geom.IRectLTRB(-8, 2, -2, 8))
	if !ok {
		t.Fatalf("expected single-period mirror to simplify")
	}
	if p := m.MapPoint(geom.Point{X: 2, Y: 2}); p.X != -2 || p.Y != 2 {
		t.Fatalf("x period -1 maps (2,2) -> %v, want (-2,2)", p)
	}
	// Period -1 on Y behaves the same way.
	if m, ok = periodicAxisTransform(shaders.TileMirror, crop, geom.IRectLTRB(2, -8, 8, -2)); !ok {
		t.Fatalf("expected single-period mirror to simplify")
	}
	if p := m.MapPoint(geom.Point{X: 2, Y: 2}); p.X != 2 || p.Y != -2 {
		t.Fatalf("y period -1 maps (2,2) -> %v, want (2,-2)", p)
	}
	// Period -2 is even, so it stays an unmirrored translation of two periods.
	if m, ok = periodicAxisTransform(shaders.TileMirror, crop, geom.IRectLTRB(-18, 2, -12, 8)); !ok {
		t.Fatalf("expected single-period mirror to simplify")
	}
	if p := m.MapPoint(geom.Point{X: 2, Y: 2}); p.X != -18 || p.Y != 2 {
		t.Fatalf("x period -2 maps (2,2) -> %v, want (-18,2)", p)
	}
	// Repeat never flips, negative period or not.
	if m, ok = periodicAxisTransform(shaders.TileRepeat, crop, geom.IRectLTRB(-8, 2, -2, 8)); !ok {
		t.Fatalf("expected single-period repeat to simplify")
	}
	if p := m.MapPoint(geom.Point{X: 2, Y: 2}); p.X != -8 || p.Y != 2 {
		t.Fatalf("repeat period -1 maps (2,2) -> %v, want (-8,2)", p)
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

// TestDecomposeTransformOptionalOutputs pins decomposeTransform's documented contract that either output may be nil: a
// caller wanting only the post-scaling remainder used to hit a nil dereference on the scaling output.
func TestDecomposeTransformOptionalOutputs(t *testing.T) {
	for _, tc := range []struct {
		transform func(m *geom.Matrix)
		name      string
	}{
		{name: "affine", transform: func(m *geom.Matrix) { m.SetRotate(30); m.PreScale(2, 3) }},
		{name: "perspective", transform: func(m *geom.Matrix) { m.SetAll(2, 0, 5, 0, 2, 7, 0.001, 0.002, 1) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var transform geom.Matrix
			tc.transform(&transform)
			pt := geom.Point{X: 10, Y: 10}
			var both, scaling geom.Matrix
			decomposeTransform(&transform, pt, &both, &scaling)
			// Dropping the scaling output must neither panic nor change the remainder.
			var postOnly geom.Matrix
			decomposeTransform(&transform, pt, &postOnly, nil)
			if postOnly != both {
				t.Fatalf("post-scaling differs without the scaling output: %v vs %v", postOnly, both)
			}
			// Dropping the post-scaling output must not change the scaling either.
			var scalingOnly geom.Matrix
			decomposeTransform(&transform, pt, nil, &scalingOnly)
			if scalingOnly != scaling {
				t.Fatalf("scaling differs without the post-scaling output: %v vs %v", scalingOnly, scaling)
			}
			// Both nil is a well-defined no-op.
			decomposeTransform(&transform, pt, nil, nil)
		})
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

// The raster blur reads pixel storage as 32-bit words, so a source it cannot read that way must come back as the
// contract's nil failure signal (Builder.Blur drops the whole result then) rather than panicking inside the passes.
func TestRasterBlurRejectsUnusableSources(t *testing.T) {
	algorithm := RasterBlurEngine().FindAlgorithm()
	bounds := geom.IRectWH(4, 4)
	sigma := geom.Size{Width: 1, Height: 1}
	blur := func(src *SpecialImage) *SpecialImage {
		return algorithm.Blur(sigma, src, bounds, shaders.TileDecal, bounds)
	}

	// Control: an N32 source blurs normally.
	n32 := NewSpecialImage(bounds, imagecore.NewPixels(imagecore.MakeN32Premul(4, 4)))
	if n32 == nil {
		t.Fatal("NewSpecialImage(N32) = nil")
	}
	if blur(n32) == nil {
		t.Fatal("Blur of an N32 source = nil, want a blurred image")
	}

	// A drawable backing whose CPU resolution fails (the GPU→CPU readback on the filter fallback lane).
	unresolvable := NewSpecialImageDrawable(bounds, &failingDrawable{})
	if unresolvable == nil {
		t.Fatal("NewSpecialImageDrawable = nil")
	}
	if got := blur(unresolvable); got != nil {
		t.Fatalf("Blur of a failed readback = %v, want nil", got)
	}

	// A non-N32 raster backing: rescale re-renders these before the blur, so reaching here is a failure, not garbage.
	for _, ct := range []imagecore.ColorType{
		imagecore.ColorTypeAlpha8, imagecore.ColorTypeRGB565, imagecore.ColorTypeRGBAF16,
		imagecore.ColorTypeBGRA8888,
	} {
		info, ok := imagecore.MakeInfo(4, 4, ct, imagecore.AlphaTypePremul)
		if !ok {
			t.Fatalf("MakeInfo(%v) failed", ct)
		}
		si := NewSpecialImage(bounds, imagecore.NewPixels(info))
		if si == nil {
			t.Fatalf("NewSpecialImage(%v) = nil", ct)
		}
		if got := blur(si); got != nil {
			t.Fatalf("Blur of a %v source = %v, want nil", ct, got)
		}
	}
}

// Only the destination half of each mode's Porter-Duff pair is transcribed (the source half multiplies a transparent
// black source and cannot change the answer), so pin the transcription and the advanced-mode miss.
func TestBlendModeDstCoeff(t *testing.T) {
	want := map[raster.BlendMode]blendCoeff{
		raster.BlendClear:    coeffZero,
		raster.BlendSrc:      coeffZero,
		raster.BlendDst:      coeffOne,
		raster.BlendSrcOver:  coeffISA,
		raster.BlendDstOver:  coeffOne,
		raster.BlendSrcIn:    coeffZero,
		raster.BlendDstIn:    coeffSA,
		raster.BlendSrcOut:   coeffZero,
		raster.BlendDstOut:   coeffISA,
		raster.BlendSrcATop:  coeffISA,
		raster.BlendDstATop:  coeffSA,
		raster.BlendXor:      coeffISA,
		raster.BlendPlus:     coeffOne,
		raster.BlendModulate: coeffSC,
		raster.BlendScreen:   coeffISC,
	}
	if len(want) != len(gDstCoeffs) {
		t.Fatalf("table covers %d modes, want %d", len(gDstCoeffs), len(want))
	}
	for mode, coeff := range want {
		got, ok := blendModeDstCoeff(mode)
		if !ok {
			t.Fatalf("mode %d not in the coefficient table", mode)
		}
		if got != coeff {
			t.Fatalf("mode %d dst coefficient = %d, want %d", mode, got, coeff)
		}
	}
	for _, mode := range []raster.BlendMode{raster.BlendOverlay, raster.BlendMultiply, raster.BlendLuminosity} {
		if _, ok := blendModeDstCoeff(mode); ok {
			t.Fatalf("advanced mode %d reported a coefficient", mode)
		}
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

// invertOrIdentity falls back to the identity matrix — not geom.Matrix's all-zeros zero value — so the layer-to-
// parameter matrix built by createInputShaders leaves sample coordinates alone when the layer matrix is singular.
func TestInvertOrIdentityFallsBackToIdentity(t *testing.T) {
	var scale geom.Matrix
	scale.SetScale(2, 4)
	scaleInv := invertOrIdentity(&scale)
	if got := scaleInv.MapPoint(geom.Point{X: 2, Y: 4}); got != (geom.Point{X: 1, Y: 1}) {
		t.Fatalf("inverse of scale(2,4) maps (2,4) to %v, want (1,1)", got)
	}

	var singular geom.Matrix
	singular.SetScale(0, 0)
	if _, ok := singular.Invert(); ok {
		t.Fatal("zero-scale matrix reported itself invertible")
	}
	fallback := invertOrIdentity(&singular)
	for _, p := range []geom.Point{{X: 3, Y: 7}, {X: -12, Y: 5}} {
		if got := fallback.MapPoint(p); got != p {
			t.Fatalf("non-invertible fallback maps %v to %v, want %v (identity)", p, got, p)
		}
	}

	// The two downstream consumers in createInputShaders: an identity layer-to-parameter matrix is an integer
	// translation (so no non-trivial-sampling flag is forced) and leaves the input shader unwrapped. The all-zeros
	// matrix fails both.
	if !isNearlyIntegerTranslation(&fallback, nil) {
		t.Fatal("identity fallback is not an integer translation")
	}
	shader := shaders.NewColor4f(colorcore.Color4f{R: 1, A: 1})
	if got := shaders.NewWithLocalMatrix(shader, fallback); got != shaders.Shader(shader) {
		t.Fatalf("identity fallback wrapped the shader as %T, want it unchanged", got)
	}
}

// SpecialImage.ColorType reports the raster backing's own color type — rescale's hasEffectsToApply compares it against
// N32 to decide whether the image must be re-rendered before the pipeline (and the raster blur engine, which reads
// pixel storage as 32-bit words) touches it. A drawable backing reports N32 instead of resolving to CPU.
func TestSpecialImageColorType(t *testing.T) {
	for _, ct := range []imagecore.ColorType{
		imagecore.ColorTypeRGBA8888, imagecore.ColorTypeBGRA8888, imagecore.ColorTypeRGB888x,
		imagecore.ColorTypeGray8, imagecore.ColorTypeAlpha8, imagecore.ColorTypeRGB565,
		imagecore.ColorTypeRGBAF16,
	} {
		info, ok := imagecore.MakeInfo(4, 4, ct, imagecore.AlphaTypePremul)
		if !ok {
			t.Fatalf("MakeInfo(%v) failed", ct)
		}
		si := NewSpecialImage(geom.IRectWH(4, 4), imagecore.NewPixels(info))
		if si == nil {
			t.Fatalf("NewSpecialImage(%v) = nil", ct)
		}
		if got := si.ColorType(); got != ct {
			t.Fatalf("ColorType() = %v, want %v", got, ct)
		}
	}

	si := NewSpecialImageDrawable(geom.IRectWH(4, 4), &fakeDrawable{})
	if si == nil {
		t.Fatal("NewSpecialImageDrawable = nil")
	}
	if got := si.ColorType(); got != imagecore.ColorTypeN32 {
		t.Fatalf("drawable-backed ColorType() = %v, want N32", got)
	}
}

// fakeDrawable is a non-*imagecore.Image DrawableImage, so NewSpecialImageDrawable keeps the drawable lane instead of
// downgrading to the raster one. It never resolves to CPU pixels; ColorType must not need it to.
type fakeDrawable struct{}

func (*fakeDrawable) Width() int32                   { return 4 }
func (*fakeDrawable) Height() int32                  { return 4 }
func (*fakeDrawable) AlphaType() imagecore.AlphaType { return imagecore.AlphaTypePremul }
func (*fakeDrawable) IsAlphaOnly() bool              { return false }
func (*fakeDrawable) UniqueID() uint32               { return 1 }
func (*fakeDrawable) MakeNonTextureImage() *imagecore.Image {
	panic("ColorType must not resolve a drawable backing to CPU pixels")
}

// failingDrawable is a drawable backing whose CPU resolution fails, the way a GPU→CPU readback does when the context is
// abandoned; subsetPixels then reports nil pixels for it.
type failingDrawable struct{ fakeDrawable }

func (*failingDrawable) MakeNonTextureImage() *imagecore.Image { return nil }
