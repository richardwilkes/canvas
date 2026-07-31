// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import (
	"math"
	"math/rand"
	"testing"
)

func TestMatrixTypeMask(t *testing.T) {
	m := IdentityMatrix()
	if m.Type() != TypeIdentity || !m.RectStaysRect() || m.HasPerspective() {
		t.Error("identity classification wrong")
	}
	m.SetTranslate(3, 4)
	if m.Type() != TypeTranslate || !m.RectStaysRect() {
		t.Error("translate classification wrong")
	}
	m.SetScale(2, 3)
	if m.Type() != TypeScale || !m.RectStaysRect() {
		t.Error("scale classification wrong")
	}
	m.SetScale(0, 3)
	if m.RectStaysRect() {
		t.Error("zero scale must not preserve rects")
	}
	m.SetRotate(45)
	if m.Type() != TypeAffine|TypeScale || m.RectStaysRect() {
		t.Error("rotate 45 classification wrong")
	}
	// A 90-degree rotation preserves rect-ness: primary diagonal zero, secondary non-zero.
	m.SetRotate(90)
	if m.Type() != TypeAffine|TypeScale || !m.RectStaysRect() {
		t.Errorf("rotate 90 classification wrong: type=%#x staysRect=%v", m.Type(), m.RectStaysRect())
	}
	m.SetAll(1, 0, 0, 0, 1, 0, 0.001, 0, 1)
	if m.Type() != TypeTranslate|TypeScale|TypeAffine|TypePerspective {
		t.Errorf("perspective type = %#x", m.Type())
	}
	// -0.0 elements must classify like 0.
	m.SetAll(1, float32(math.Copysign(0, -1)), 0, 0, 1, 0, 0, 0, 1)
	if m.Type() != TypeIdentity {
		t.Errorf("-0 skew type = %#x, want identity", m.Type())
	}
}

func TestMatrixRotate90Exact(t *testing.T) {
	m := RotateDegMatrix(90)
	got := m.MapXY(10, 0)
	// With sin/cos snapping, rotating (10,0) by 90 degrees must be exactly (0,10).
	if got.X != 0 || got.Y != 10 {
		t.Errorf("rotate90(10,0) = %+v, want (0,10)", got)
	}
	m = RotateDegMatrix(180)
	got = m.MapXY(10, 0)
	if got.X != -10 || got.Y != 0 {
		t.Errorf("rotate180(10,0) = %+v, want (-10,0)", got)
	}
}

func TestMatrixConcatTranslateScale(t *testing.T) {
	var a, b, m Matrix
	a.SetTranslate(10, 20)
	b.SetScale(2, 3)
	m.SetConcat(&a, &b)
	// a*b: scale then translate.
	got := m.MapXY(1, 1)
	if got.X != 12 || got.Y != 23 {
		t.Errorf("(T*S)(1,1) = %+v, want (12,23)", got)
	}
	if m.Type() != TypeScale|TypeTranslate {
		t.Errorf("concat type = %#x", m.Type())
	}
	// Identity short-circuits copy the other operand.
	id := IdentityMatrix()
	m.SetConcat(&id, &a)
	if m.As9() != a.As9() {
		t.Error("identity*a != a")
	}
}

// mapAffineRef is a float64 reference for affine point mapping.
func mapAffineRef(mat [9]float32, x, y float64) (mx, my float64) {
	sx, kx, tx := float64(mat[MScaleX]), float64(mat[MSkewX]), float64(mat[MTransX])
	ky, sy, ty := float64(mat[MSkewY]), float64(mat[MScaleY]), float64(mat[MTransY])
	return sx*x + kx*y + tx, ky*x + sy*y + ty
}

func TestMatrixMapPointsMatchesFloat64(t *testing.T) {
	rng := rand.New(rand.NewSource(12345))
	for iter := 0; iter < 200; iter++ {
		var m Matrix
		m.SetAll(
			rng.Float32()*4-2, rng.Float32()*2-1, rng.Float32()*100-50,
			rng.Float32()*2-1, rng.Float32()*4-2, rng.Float32()*100-50,
			0, 0, 1,
		)
		x := rng.Float32()*200 - 100
		y := rng.Float32()*200 - 100
		got := m.MapXY(x, y)
		wantX, wantY := mapAffineRef(m.As9(), float64(x), float64(y))
		// float32 evaluation of a 2x2 mul-add differs from float64 by bounded rounding error.
		if math.Abs(float64(got.X)-wantX) > 1e-3 || math.Abs(float64(got.Y)-wantY) > 1e-3 {
			t.Fatalf("iter %d: MapXY(%g,%g) = %+v, want (%g,%g)", iter, x, y, got, wantX, wantY)
		}
	}
}

func TestMatrixMapPointsSliceMatchesSingle(t *testing.T) {
	rng := rand.New(rand.NewSource(999))
	var persp Matrix
	persp.SetAll(1, 0.1, 2, -0.2, 0.9, 3, 0.0005, -0.0002, 1)
	mats := []Matrix{IdentityMatrix(), TranslateMatrix(5, -3), ScaleMatrix(2, 0.5), RotateDegMatrix(30), persp}
	for mi := range mats {
		m := &mats[mi]
		src := make([]Point, 17)
		for i := range src {
			src[i] = Point{X: rng.Float32()*100 - 50, Y: rng.Float32()*100 - 50}
		}
		dst := make([]Point, len(src))
		m.MapPoints(dst, src)
		for i := range src {
			if got := m.MapPoint(src[i]); got != dst[i] {
				t.Fatalf("matrix %d point %d: MapPoints %+v != MapPoint %+v", mi, i, dst[i], got)
			}
		}
	}
}

func TestMatrixInvertRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(777))
	for iter := 0; iter < 200; iter++ {
		var m Matrix
		// Random affine with a determinant bounded away from zero.
		for {
			m.SetAll(
				rng.Float32()*4-2, rng.Float32()*2-1, rng.Float32()*20-10,
				rng.Float32()*2-1, rng.Float32()*4-2, rng.Float32()*20-10,
				0, 0, 1,
			)
			det := float64(m.Get(MScaleX))*float64(m.Get(MScaleY)) -
				float64(m.Get(MSkewX))*float64(m.Get(MSkewY))
			if math.Abs(det) > 0.05 {
				break
			}
		}
		inv, ok := m.Invert()
		if !ok {
			t.Fatalf("iter %d: invert failed for %v", iter, m.As9())
		}
		var prod Matrix
		prod.SetConcat(&m, &inv)
		p := prod.MapXY(7, -3)
		if math.Abs(float64(p.X)-7) > 1e-2 || math.Abs(float64(p.Y)+3) > 1e-2 {
			t.Fatalf("iter %d: m*inv maps (7,-3) to %+v", iter, p)
		}
	}
}

func TestMatrixInvertSpecialCases(t *testing.T) {
	// Identity inverts to itself.
	id := IdentityMatrix()
	inv, ok := id.Invert()
	if !ok || !inv.IsIdentity() {
		t.Error("identity invert failed")
	}
	// Translate-only fast path.
	tr := TranslateMatrix(5, -7)
	inv, ok = tr.Invert()
	if !ok || inv.Get(MTransX) != -5 || inv.Get(MTransY) != 7 {
		t.Errorf("translate invert = %v", inv.As9())
	}
	// Scale+translate fast path.
	var st Matrix
	st.SetScaleTranslate(2, 4, 10, 20)
	inv, ok = st.Invert()
	if !ok {
		t.Fatal("scale-translate invert failed")
	}
	if inv.Get(MScaleX) != 0.5 || inv.Get(MScaleY) != 0.25 || inv.Get(MTransX) != -5 || inv.Get(MTransY) != -5 {
		t.Errorf("scale-translate invert = %v", inv.As9())
	}
	// Degenerate matrix fails.
	var z Matrix
	z.SetScale(0, 1)
	if _, ok = z.Invert(); ok {
		t.Error("zero-scale matrix must not invert")
	}
	// The inverse's type mask for the scale path is mask | rectStaysRect.
	if !st.RectStaysRect() {
		t.Error("scale-translate should preserve rects")
	}
}

func TestMatrixMapRect(t *testing.T) {
	src := RectLTRB(0, 0, 10, 20)
	m := TranslateMatrix(5, 5)
	dst, exact := m.MapRect(src)
	if !exact || dst != RectLTRB(5, 5, 15, 25) {
		t.Errorf("translate MapRect = %+v exact=%v", dst, exact)
	}
	// Negative scale must produce a sorted rect.
	m = ScaleMatrix(-1, 2)
	dst, exact = m.MapRect(src)
	if !exact || dst != RectLTRB(-10, 0, 0, 40) {
		t.Errorf("negative scale MapRect = %+v exact=%v", dst, exact)
	}
	// Rotation by 90 degrees is not the scale-translate path but still stays-rect.
	m = RotateDegMatrix(90)
	dst, exact = m.MapRect(src)
	if !exact || dst != RectLTRB(-20, 0, 0, 10) {
		t.Errorf("rotate90 MapRect = %+v exact=%v", dst, exact)
	}
	// Rotation by 45 degrees reports inexact bounds.
	m = RotateDegMatrix(45)
	if _, exact = m.MapRect(src); exact {
		t.Error("rotate45 MapRect should be inexact")
	}
	// Perspective reports inexact.
	var persp Matrix
	persp.SetAll(1, 0, 0, 0, 1, 0, 0.001, 0, 1)
	if _, exact = persp.MapRect(src); exact {
		t.Error("perspective MapRect should be inexact")
	}
}

func TestMatrixPrePostOps(t *testing.T) {
	// preTranslate on an affine matrix folds through the linear part.
	m := ScaleMatrix(2, 3)
	m.PreTranslate(10, 100)
	got := m.MapXY(0, 0)
	if got.X != 20 || got.Y != 300 {
		t.Errorf("preTranslate result maps origin to %+v, want (20,300)", got)
	}
	m = ScaleMatrix(2, 3)
	m.PostTranslate(10, 100)
	got = m.MapXY(0, 0)
	if got.X != 10 || got.Y != 100 {
		t.Errorf("postTranslate result maps origin to %+v, want (10,100)", got)
	}
	// preScale fast path keeps classification consistent.
	m = IdentityMatrix()
	m.PreScale(2, 2)
	if m.Type() != TypeScale {
		t.Errorf("preScale type = %#x", m.Type())
	}
	m.PreScale(0.5, 0.5)
	if m.Type() != TypeIdentity {
		t.Errorf("preScale back to identity type = %#x", m.Type())
	}
	// preScale by zero clears rect-stays-rect.
	m = IdentityMatrix()
	m.PreScale(0, 1)
	if m.RectStaysRect() {
		t.Error("preScale(0,1) should clear rect-stays-rect")
	}
	// PreConcat equivalence with SetConcat.
	a := RotateDegMatrix(30)
	b := TranslateMatrix(3, 4)
	m1 := a
	m1.PreConcat(&b)
	var m2 Matrix
	m2.SetConcat(&a, &b)
	if m1.As9() != m2.As9() {
		t.Error("PreConcat != SetConcat(a,b)")
	}
}

func TestMatrixZeroValueIsHonest(t *testing.T) {
	// The zero Matrix is the all-zero matrix (degenerate), and must classify itself honestly rather than pretending to
	// be identity.
	var m Matrix
	if m.IsIdentity() {
		t.Error("zero-value Matrix must not claim to be identity")
	}
	got := m.MapXY(5, 7)
	if got.X != 0 || got.Y != 0 {
		t.Errorf("zero matrix should map everything to origin, got %+v", got)
	}
}

func TestMatrixZeroValuePreScaleStaysHonest(t *testing.T) {
	// PreScale is the one mask-fixup path reachable without a prior Set*/Type() call, so it must not stamp a clean mask
	// onto a zero-valued Matrix. Every classification has to agree with an honest recompute of the same nine elements.
	for _, sc := range []struct {
		name   string
		sx, sy float32
	}{
		{name: "nonzero", sx: 2, sy: 3},
		{name: "zeroScaleX", sx: 0, sy: 3},
		{name: "shrinkToOne", sx: 1, sy: 2},
	} {
		var m Matrix
		m.PreScale(sc.sx, sc.sy)
		honest := MatrixFrom9(m.As9())
		if m.Type() != honest.Type() {
			t.Errorf("%s: type = %#x, want %#x", sc.name, m.Type(), honest.Type())
		}
		if m.HasPerspective() != honest.HasPerspective() {
			t.Errorf("%s: HasPerspective = %v, want %v", sc.name, m.HasPerspective(), honest.HasPerspective())
		}
		if m.RectStaysRect() != honest.RectStaysRect() {
			t.Errorf("%s: RectStaysRect = %v, want %v", sc.name, m.RectStaysRect(), honest.RectStaysRect())
		}
		if _, ok := m.AsAffine(); ok {
			t.Errorf("%s: AsAffine succeeded on the degenerate all-zero matrix", sc.name)
		}
		if m.IsIdentity() || m.IsScaleTranslate() || m.IsTranslate() {
			t.Errorf("%s: zero matrix classified as identity/scale-translate/translate", sc.name)
		}
	}
}

func TestMatrixPreservesRightAngles(t *testing.T) {
	cases := []struct {
		name string
		m    Matrix
		want bool
	}{
		{name: "identity", m: IdentityMatrix(), want: true},
		{name: "translate", m: TranslateMatrix(3, 4), want: true},
		{name: "scale", m: ScaleMatrix(2, 5), want: true},
		{name: "rotate 30", m: RotateDegMatrix(30), want: true},
		{name: "rotate 45 scaled", m: func() Matrix {
			m := RotateDegMatrix(45)
			m.PostScale(2, 2)
			return m
		}(), want: true},
		{name: "skew", m: func() Matrix {
			var m Matrix
			m.SetSkew(0.5, 0)
			return m
		}(), want: false},
		{name: "degenerate scale", m: ScaleMatrix(0, 5), want: false},
		{name: "perspective", m: func() Matrix {
			m := IdentityMatrix()
			m.Set(MPersp0, 0.01)
			return m
		}(), want: false},
	}
	for _, c := range cases {
		if got := c.m.PreservesRightAngles(); got != c.want {
			t.Errorf("%s: preservesRightAngles = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestMatrixMaxScale(t *testing.T) {
	const huge = float32(1e30) // squaring this overflows float32
	cases := []struct {
		name string
		m    Matrix
		want float32
	}{
		{name: "identity", m: IdentityMatrix(), want: 1},
		{name: "translate", m: TranslateMatrix(3, 4), want: 1},
		{name: "scale", m: ScaleMatrix(2, -5), want: 5},
		{name: "rotate 45 scaled", m: func() Matrix {
			m := RotateDegMatrix(45)
			m.PostScale(2, 2)
			return m
		}(), want: 2},
		// Perspective and anything that makes the squared singular value non-finite report the -1 sentinel, since the
		// callers that fall back on it (stroke tessellation, path renderers) cannot use an infinite or NaN scale.
		{name: "perspective", m: func() Matrix {
			m := IdentityMatrix()
			m.Set(MPersp0, 0.01)
			return m
		}(), want: -1},
		{name: "overflowing skew", m: MatrixFrom9([9]float32{0, huge, 0, huge, 0, 0, 0, 0, 1}), want: -1},
		{name: "overflowing affine", m: MatrixFrom9([9]float32{huge, huge, 0, huge, huge, 0, 0, 0, 1}), want: -1},
		{name: "NaN affine", m: MatrixFrom9([9]float32{float32(math.NaN()), 1, 0, 1, 1, 0, 0, 0, 1}), want: -1},
		// The scale-only lane skips the quadratic entirely, but it must still honor the -1 sentinel rather than handing
		// back an infinite or NaN scale for callers to use as a real one.
		{name: "infinite scale only", m: ScaleMatrix(float32(math.Inf(1)), 2), want: -1},
		{name: "negative infinite scale only", m: ScaleMatrix(2, float32(math.Inf(-1))), want: -1},
		{name: "NaN scale only", m: ScaleMatrix(float32(math.NaN()), 2), want: -1},
	}
	for _, c := range cases {
		// NaN has to be rejected explicitly, since every comparison against it is false.
		if got := c.m.MaxScale(); ScalarIsNaN(got) || ScalarAbs(got-c.want) > 1e-4 {
			t.Errorf("%s: MaxScale = %v, want %v", c.name, got, c.want)
		}
	}
}

// orthogonalRoundingProne is [3e10 -4; 4e10 3]: its two columns are exactly orthogonal (the dot product's two terms are
// both 1.2e11, which does not fit in a float32 mantissa), and its determinant is far from zero. Every cross/dot term
// derived from it is mathematically exact only when each product is rounded to float32 before the add. Contracting any
// of them into a fused multiply-add — which the Go compiler does on arm64 for unpinned expressions — leaves the
// rounding error behind, which is enormous at this magnitude and flips the results below.
var orthogonalRoundingProne = MatrixFrom9([9]float32{3e10, -4, 0, 4e10, 3, 0, 0, 0, 1})

// degeneracyProbeScale is large enough that 6*degeneracyProbeScale does not fit in a float32 mantissa.
var degeneracyProbeScale = float32(1e10)

func TestMatrixDegeneracyIsUnfused(t *testing.T) {
	// Both halves of the determinant are the same real number, 6e10, reached through different factor pairs (so the
	// compiler cannot fold them into one value): the determinant is exactly zero when each product is rounded to
	// float32 first. Fused, the surviving term keeps the exact product and the difference is the rounding error of
	// 6e10, which is thousands of times the tolerance, so the matrix reads as non-degenerate.
	// The operands come from a var so the compiler cannot constant-fold the products and hide the contraction.
	x := degeneracyProbeScale
	if !isDegenerate2x2(x, 2*x, 3, 6) {
		t.Error("isDegenerate2x2 must report an exactly-zero determinant as degenerate on every platform")
	}
	if isDegenerate2x2(3e10, -4, 4e10, 3) {
		t.Error("isDegenerate2x2 must not report a well-conditioned matrix as degenerate")
	}
}

func TestMatrixPreservesRightAnglesIsUnfused(t *testing.T) {
	m := orthogonalRoundingProne
	if !m.PreservesRightAngles() {
		t.Error("exactly-orthogonal columns must preserve right angles on every platform")
	}
}

func TestMatrixMaxScaleIsUnfused(t *testing.T) {
	// The columns are (3e10, 4e10) and (-4, 3), so A^T*A is diagonal with entries 25e20 and 25, and the largest
	// singular value is 5e10. Fusing the b term makes bSqd exceed the orthogonality tolerance, which sends the
	// computation down the quadratic branch where aminusc*aminusc overflows float32 and MaxScale returns the -1
	// sentinel instead.
	if got := orthogonalRoundingProne.MaxScale(); got != 5e10 {
		t.Errorf("MaxScale = %v, want 5e10", got)
	}
}

// referenceMaxScale recomputes MaxScale's affine branch with every intermediate rounded to float32 explicitly. Each
// step is evaluated in float64 and then rounded once, which reproduces correctly-rounded unfused float32 arithmetic and
// cannot itself be contracted into an FMA, so it is an independent oracle for the pinning in MaxScale.
func referenceMaxScale(m *Matrix) float32 {
	r32 := func(v float64) float32 { return float32(v) }
	sx, ky := float64(m.mat[MScaleX]), float64(m.mat[MSkewY])
	kx, sy := float64(m.mat[MSkewX]), float64(m.mat[MScaleY])
	a := r32(float64(r32(sx*sx)) + float64(r32(ky*ky)))
	b := r32(float64(r32(sx*kx)) + float64(r32(sy*ky)))
	c := r32(float64(r32(kx*kx)) + float64(r32(sy*sy)))
	bSqd := r32(float64(b) * float64(b))
	var result float32
	if bSqd <= ScalarNearlyZeroTol*ScalarNearlyZeroTol {
		result = max(a, c)
	} else {
		aminusc := r32(float64(a) - float64(c))
		apluscdiv2 := r32(0.5 * (float64(a) + float64(c)))
		disc := r32(float64(r32(float64(aminusc)*float64(aminusc))) + float64(r32(4*float64(bSqd))))
		x := r32(0.5 * float64(ScalarSqrt(disc)))
		result = r32(float64(apluscdiv2) + float64(x))
	}
	if !IsFinite(result) {
		return -1
	}
	if result < 0 {
		result = 0
	}
	return ScalarSqrt(result)
}

func TestMaxScaleMatchesUnfusedReference(t *testing.T) {
	// Pins MaxScale's affine branch against an independent float64 oracle. Note this does NOT distinguish fused from
	// unfused halving: p + 0.5*s contracts to an FMADDS on arm64, but halving is exact outside the subnormal range and
	// TestMaxScaleHalvingsStayNormal shows the branch guard keeps it out of that range. The oracle instead guards the
	// dot products and the discriminant, where fusion would be observable.
	check := func(m *Matrix) {
		t.Helper()
		if !m.HasPerspective() {
			if got, want := m.MaxScale(), referenceMaxScale(m); got != want {
				t.Errorf("MaxScale(%v) = %v, want %v (unfused); the discriminant was folded into an FMA",
					m.mat, got, want)
			}
		}
	}
	build := func(sx, kx, ky, sy float32) *Matrix {
		var m Matrix
		m.SetAll(sx, kx, 0, ky, sy, 0, 0, 0, 1)
		return &m
	}
	for _, v := range [][4]float32{
		{2, 0, 0, 3},
		{1, 1e-7, 1e-7, 1},
		{0.1, 0.2, 0.3, 0.4},
		{1.0000001, 0.9999999, 1.0000002, 0.9999998},
		{3, 4, 5, 6},
		{1e-8, 1, 1, 1e-8},
	} {
		check(build(v[0], v[1], v[2], v[3]))
	}
	rng := rand.New(rand.NewSource(7))
	for range 200000 {
		check(build(rng.Float32()*20-10, rng.Float32()*20-10, rng.Float32()*20-10, rng.Float32()*20-10))
	}
}

func TestMaxScaleHalvingsStayNormal(t *testing.T) {
	// MaxScale's affine branch halves (a + c) and sqrt(disc). Halving is exact in binary floating point unless the
	// result underflows to subnormal, which is why the FMA contraction on that final add is harmless. The bSqd guard is
	// what keeps both operands out of the subnormal range: bSqd > (1/4096)^2 forces sqrt(disc) >= 4.8e-4, and
	// |b| <= (a+c)/2 by Cauchy-Schwarz forces a + c > 4.9e-4. If a rework ever breaks that, the halvings stop being
	// exact and the contraction starts to matter.
	const smallestNormal = 1.1754944e-38
	check := func(sx, kx, ky, sy float32) {
		t.Helper()
		a := float32(sx*sx) + float32(ky*ky)
		b := float32(sx*kx) + float32(sy*ky)
		c := float32(kx*kx) + float32(sy*sy)
		bSqd := b * b
		if bSqd <= ScalarNearlyZeroTol*ScalarNearlyZeroTol {
			return // the orthogonal branch, which does no halving
		}
		if !IsFinite(a, c, bSqd) {
			return
		}
		aminusc := a - c
		disc := float32(aminusc*aminusc) + float32(4*bSqd)
		if sum := a + c; sum < smallestNormal*2 {
			t.Errorf("a+c = %v for [%v %v; %v %v]: halving (a+c) can underflow to subnormal", sum, sx, kx, ky, sy)
		}
		if root := ScalarSqrt(disc); root < smallestNormal*2 {
			t.Errorf("sqrt(disc) = %v for [%v %v; %v %v]: halving it can underflow to subnormal", root, sx, kx, ky, sy)
		}
	}
	rng := rand.New(rand.NewSource(11))
	for range 300000 {
		// Include magnitudes small enough to reach subnormal squares if the guard did not exclude them.
		scale := float32(math.Pow(10, rng.Float64()*40-38))
		check(rng.Float32()*scale, rng.Float32()*scale, rng.Float32()*scale, rng.Float32()*scale)
	}
}
