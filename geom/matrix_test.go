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
	}
	for _, c := range cases {
		// NaN has to be rejected explicitly, since every comparison against it is false.
		if got := c.m.MaxScale(); ScalarIsNaN(got) || ScalarAbs(got-c.want) > 1e-4 {
			t.Errorf("%s: MaxScale = %v, want %v", c.name, got, c.want)
		}
	}
}
