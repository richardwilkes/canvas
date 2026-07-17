// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package probe holds the oracle numeric probes: bit-exact differential tests that drive the Go port over shared input
// corpora and compare against the oracle's frozen answers. These lock in the math-core and path requirement that pure
// math match the oracle exactly.
//
// The probes originally drove the C library live (via skiac, including its C++ shim). The C side is gone; its answers
// were captured once, keyed by a hash of the inputs, into testdata/ref (see ref_test.go). This package is cgo-free.
package probe

import (
	"math"
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// Comparison policy. Where the C++ computes through double before storing (affine setConcat via muladdmul, affine
// invert via dcross, rect ops, scalar rounding), the Go port must match bit-exactly. Where the C++ does float
// arithmetic (perspective rowcol3/scross, mapPoints procs, Skia's Geometry interpolation), clang's default
// -ffp-contract=on fuses mul+add chains into FMAs, and which chains fuse is a per-compiler, per-platform choice the
// port cannot (economically) replicate, and the tolerance-based policy declines to chase it. For those paths the probes
// instead require both results to lie within probeK*eps32*mag of each other, where mag is the magnitude of the
// expression's terms evaluated in float64 — tight enough that any formula or fast-path porting bug fails, while
// contraction-level noise (including its amplification through cancellation and perspective division) passes by
// construction.

const eps32 = 1.0 / (1 << 23) // 2^-23, one part in the last place of a float32 mantissa

// probeK is the headroom multiplier over a single rounding error; measured maxima on darwin/arm64 are well under half
// this.
const probeK = 64

func nonFinite(v float32) bool {
	f := float64(v)
	return math.IsNaN(f) || math.IsInf(f, 0)
}

// ptsAgree compares two point slices under agree() per axis (see agree). It lives in this cross-platform helper file —
// rather than beside the geometry probes that first used it — because the probes that shared it were once split by
// whether they needed the C++ shim, which was build-excluded on Windows (mingw cannot resolve Skia's MSVC-mangled C++
// exports). Replay needs no linker, so every probe now runs everywhere and the split is gone.
func ptsAgree(t *testing.T, label string, got, want []geom.Point, magX, magY float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: length %d vs %d", label, len(got), len(want))
		return
	}
	for i := range got {
		if !agree(got[i].X, want[i].X, magX) || !agree(got[i].Y, want[i].Y, magY) {
			t.Errorf("%s pt %d: go %v, c %v", label, i, got[i], want[i])
		}
	}
}

// agree reports whether a and b are plausibly the same float32 expression evaluated under different fp-contract
// choices, given the float64 term magnitude of the expression.
func agree(a, b float32, mag float64) bool {
	if f32Eq(a, b) {
		return true
	}
	if nonFinite(a) && nonFinite(b) {
		return true
	}
	if math.IsNaN(mag) || mag >= math.MaxFloat32 {
		return true // overflow territory: rounding order decides between Inf/NaN/huge, all equally (in)valid
	}
	return math.Abs(float64(a)-float64(b)) <= probeK*eps32*mag+math.SmallestNonzeroFloat32
}

// refRowCol3 evaluates row·col (3 terms) in float64, returning the value and the term magnitude.
func refRowCol3(a9, b9 [9]float32, row, col int) (val, mag float64) {
	for i := range 3 {
		p := float64(a9[row*3+i]) * float64(b9[i*3+col])
		val += p
		mag += math.Abs(p)
	}
	return val, mag
}

// refMapPoint evaluates the full 3x3 mapping of p in float64, returning the mapped point and per-axis term magnitudes
// scaled for the perspective division (i.e. agree()-ready tolerances).
func refMapPoint(m *geom.Matrix, p geom.Point) (x, y, magX, magY float64) {
	m9 := m.As9()
	px, py := float64(p.X), float64(p.Y)
	dot := func(r int) (v, mag float64) {
		t0 := float64(m9[r*3]) * px
		t1 := float64(m9[r*3+1]) * py
		t2 := float64(m9[r*3+2])
		return t0 + t1 + t2, math.Abs(t0) + math.Abs(t1) + math.Abs(t2)
	}
	xv, xm := dot(0)
	yv, ym := dot(1)
	if !m.HasPerspective() {
		return xv, yv, xm, ym
	}
	zv, zm := dot(2)
	if zv == 0 {
		return xv, yv, math.Inf(1), math.Inf(1)
	}
	// error(X/Z) <= err(X)/|Z| + |X|*err(Z)/Z^2, with err(T) = K*eps*mag(T); fold the /|Z| factors into mag.
	az := math.Abs(zv)
	return xv / zv, yv / zv, xm/az + math.Abs(xv)*zm/(az*az), ym/az + math.Abs(yv)*zm/(az*az)
}

// f32Eq is bit-exact float comparison, except that all NaNs compare equal (payloads may legitimately differ).
func f32Eq(a, b float32) bool {
	if math.Float32bits(a) == math.Float32bits(b) {
		return true
	}
	return a != a && b != b
}

func ptEq(a, b geom.Point) bool { return f32Eq(a.X, b.X) && f32Eq(a.Y, b.Y) }

func ptsEq(a, b []geom.Point) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !ptEq(a[i], b[i]) {
			return false
		}
	}
	return true
}

func rectEq(a, b geom.Rect) bool {
	return f32Eq(a.Left, b.Left) && f32Eq(a.Top, b.Top) && f32Eq(a.Right, b.Right) && f32Eq(a.Bottom, b.Bottom)
}

func matEq(a, b geom.Matrix) bool {
	av, bv := a.As9(), b.As9()
	for i := range av {
		if !f32Eq(av[i], bv[i]) {
			return false
		}
	}
	return true
}

// matrixCorpus builds a deterministic corpus covering every Skia's Matrix type-mask class, built through the same
// setters Skia would use so lazy type masks are realistic, plus SetAll cases with awkward float values.
func matrixCorpus() []geom.Matrix {
	var out []geom.Matrix
	add := func(m geom.Matrix) { out = append(out, m) }

	add(geom.IdentityMatrix())
	add(geom.TranslateMatrix(10, -20))
	add(geom.TranslateMatrix(0.5, 1e-8))
	add(geom.TranslateMatrix(3e38, -3e38))
	add(geom.ScaleMatrix(2, 3))
	add(geom.ScaleMatrix(-1, 1))
	add(geom.ScaleMatrix(1e-20, 1e20))
	add(geom.ScaleMatrix(0, 0))
	add(geom.RotateDegMatrix(30))
	add(geom.RotateDegMatrix(45))
	add(geom.RotateDegMatrix(90))
	add(geom.RotateDegMatrix(180))
	add(geom.RotateDegMatrix(-137.25))

	var m geom.Matrix
	m.SetScaleTranslate(2.5, -0.25, 17.5, 42)
	add(m)
	m.SetSkew(0.5, -0.25)
	add(m)
	m.SetSkewPivot(1.5, 0.75, 10, 20)
	add(m)
	m.SetRotatePivot(60, 128, 128)
	add(m)
	m.SetSinCos(0.6, 0.8)
	add(m)
	m.SetAll(1, 0, 0, 0, 1, 0, 0.001, -0.002, 1) // pure perspective factors
	add(m)
	m.SetAll(2, 0.5, -30, -0.25, 1.5, 12, 0.0005, 0.001, 0.75) // full perspective
	add(m)
	m.SetAll(1, 2, 3, 4, 5, 6, 7, 8, 9)
	add(m)
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	m.SetAll(nan, 0, 0, 0, 1, 0, 0, 0, 1)
	add(m)
	m.SetAll(1, 0, inf, 0, 1, -inf, 0, 0, 1)
	add(m)
	m.SetAll(1e38, 1e38, 0, -1e38, 1e38, 0, 0, 0, 1)
	add(m)

	rng := rand.New(rand.NewSource(20260701))
	scales := []float32{1e-10, 0.001, 1, 1000, 1e10}
	for i := range 20 {
		s := scales[i%len(scales)]
		m.SetAll(
			(rng.Float32()*2-1)*s, (rng.Float32()*2-1)*s, (rng.Float32()*2-1)*s,
			(rng.Float32()*2-1)*s, (rng.Float32()*2-1)*s, (rng.Float32()*2-1)*s,
			(rng.Float32()*2-1)*0.01, (rng.Float32()*2-1)*0.01, 1+(rng.Float32()*2-1)*0.5,
		)
		add(m)
	}
	return out
}

// pointCorpus is a deterministic set of map targets: a coarse grid, awkward magnitudes, and specials.
func pointCorpus() []geom.Point {
	out := make([]geom.Point, 0, 8*5+4+40)
	for _, x := range []float32{-1000, -1.5, -1e-8, 0, 0.5, 3, 127.75, 1e6} {
		for _, y := range []float32{-256, -0.125, 0, 1, 999.5} {
			out = append(out, geom.Point{X: x, Y: y})
		}
	}
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	out = append(
		out,
		geom.Point{X: nan, Y: 1},
		geom.Point{X: 1, Y: -inf},
		geom.Point{X: 3e38, Y: 3e38},
		geom.Point{X: 1e-40, Y: -1e-40}, // subnormals
	)
	rng := rand.New(rand.NewSource(42))
	for range 40 {
		out = append(out, geom.Point{X: (rng.Float32()*2 - 1) * 512, Y: (rng.Float32()*2 - 1) * 512})
	}
	return out
}

// rectCorpus is a deterministic set of rects: sorted, unsorted, empty, point-like, huge, and non-finite.
func rectCorpus() []geom.Rect {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	out := make([]geom.Rect, 0, 42)
	out = append(out,
		geom.RectLTRB(0, 0, 0, 0),
		geom.RectLTRB(0, 0, 100, 50),
		geom.RectLTRB(-25.5, -10.25, 30.75, 42.125),
		geom.RectLTRB(100, 50, 0, 0),  // inverted
		geom.RectLTRB(10, 10, 10, 90), // zero width
		geom.RectLTRB(10, 90, 90, 10), // inverted vertically
		geom.RectLTRB(-3e38, -3e38, 3e38, 3e38),
		geom.RectLTRB(nan, 0, 10, 10),
		geom.RectLTRB(0, 0, nan, 10),
		geom.RectLTRB(0, -inf, 10, inf),
		geom.RectLTRB(1e-40, -1e-40, 2e-40, 3e-40),
		geom.RectLTRB(0.49999997, -0.49999997, 16777215, -16777215),
	)
	rng := rand.New(rand.NewSource(7))
	for range 30 {
		out = append(out, geom.RectLTRB(
			(rng.Float32()*2-1)*200, (rng.Float32()*2-1)*200,
			(rng.Float32()*2-1)*200, (rng.Float32()*2-1)*200,
		))
	}
	return out
}
