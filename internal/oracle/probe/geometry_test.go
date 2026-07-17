// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This probe checks the port's curve chopping against Skia's Geometry chop/conic routines, replayed from the frozen
// reference fixtures (see ref.go). It once drove Skia's C++ internals live through the skiac shim, which confined it to
// darwin/linux — mingw g++ cannot resolve Skia's MSVC-mangled C++ exports. Replay needs no linker, so it now runs
// everywhere.

package probe

import (
	"math"
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func randPts(rng *rand.Rand, n int) []geom.Point {
	out := make([]geom.Point, n)
	for i := range out {
		out[i] = geom.Point{X: (rng.Float32()*2 - 1) * 300, Y: (rng.Float32()*2 - 1) * 300}
	}
	return out
}

// chopMags returns per-axis agree() magnitudes for curve-chopping outputs: results are repeated interpolations within
// the control hull, so a small multiple of the largest input coordinate bounds the term magnitudes at every level.
func chopMags(src []geom.Point) (magX, magY float64) {
	for _, p := range src {
		magX = math.Max(magX, math.Abs(float64(p.X)))
		magY = math.Max(magY, math.Abs(float64(p.Y)))
	}
	return 2 * magX, 2 * magY
}

var chopTs = []float32{1e-7, 0.25, 0.5, 0.7071, 0.9999999}

// TestChopQuadAtProbe: geom.ChopQuadAt vs Skia's ChopQuadAt (float interpolation → contraction tolerance).
func TestChopQuadAtProbe(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	for range 50 {
		src := randPts(rng, 3)
		magX, magY := chopMags(src)
		for _, tv := range chopTs {
			want := refChopQuadAt([3]geom.Point(src), tv)
			dst := make([]geom.Point, 5)
			geom.ChopQuadAt(src, dst, tv)
			ptsAgree(t, "ChopQuadAt", dst, want[:], magX, magY)
		}
	}
}

// TestChopCubicAtProbe: geom.ChopCubicAt vs Skia's ChopCubicAt.
func TestChopCubicAtProbe(t *testing.T) {
	rng := rand.New(rand.NewSource(102))
	for range 50 {
		src := randPts(rng, 4)
		magX, magY := chopMags(src)
		for _, tv := range chopTs {
			want := refChopCubicAt([4]geom.Point(src), tv)
			dst := make([]geom.Point, 7)
			geom.ChopCubicAt(src, dst, tv)
			ptsAgree(t, "ChopCubicAt", dst, want[:], magX, magY)
		}
	}
}

// TestConicQuadsProbe: Skia's Conic::computeQuadPOW2 + chopIntoQuadsPOW2 vs the geom port. The count must match exactly
// (it feeds verb streams); the points get the contraction tolerance.
func TestConicQuadsProbe(t *testing.T) {
	rng := rand.New(rand.NewSource(103))
	weights := []float32{0.1, 0.5, 0.7071068, 0.99, 1, 1.5, 20, 1e-6, 8388608}
	tols := []float32{0.25, 1.0 / 1024, 4}
	for range 30 {
		pts := randPts(rng, 3)
		magX, magY := chopMags(pts)
		for _, w := range weights {
			conic := geom.MakeConic(pts[0], pts[1], pts[2], w)
			for _, tol := range tols {
				gotN := conic.ComputeQuadPOW2(tol)
				wantN := refConicComputeQuadPOW2([3]geom.Point(pts), w, tol)
				if gotN != wantN {
					t.Errorf("ComputeQuadPOW2(%v w=%g tol=%g): go %d, c %d", pts, w, tol, gotN, wantN)
					continue
				}
				if gotN < 0 {
					continue
				}
				goPts := make([]geom.Point, 1+2*(1<<uint(gotN)))
				gotQuads := conic.ChopIntoQuadsPOW2(goPts, gotN)
				wantQuads, wantPts := refConicChopIntoQuadsPOW2([3]geom.Point(pts), w, gotN)
				if gotQuads != wantQuads {
					// At pow2 == kMaxConicToQuadPOW2 (5), extreme weights trigger a line-collapse special case gated on
					// bitwise equality of chopped midpoints — a contraction cliff where the C library's own behavior is
					// compiler/platform-dependent. 2-vs-32 is that cliff.
					if gotN == 5 && (gotQuads == 2) != (wantQuads == 2) {
						continue
					}
					t.Errorf("ChopIntoQuadsPOW2(%v w=%g pow2=%d) count: go %d, c %d", pts, w, gotN, gotQuads, wantQuads)
					continue
				}
				ptsAgree(t, "ChopIntoQuadsPOW2", goPts[:1+2*gotQuads], wantPts, magX, magY)
			}
		}
	}
}

// TestBuildUnitArcProbe: Skia's Conic::BuildUnitArc vs geom.BuildUnitArc over an angle sweep, both directions, with and
// without a user matrix. Conic counts must match exactly; points/weights get the contraction tolerance (unit-circle
// scale, so mag is small).
func TestBuildUnitArcProbe(t *testing.T) {
	um := geom.RotateDegMatrix(33)
	um.PostTranslate(12.5, -7.25)
	matrices := []*geom.Matrix{nil, &um}
	for startDeg := 0; startDeg < 360; startDeg += 15 {
		for stopDeg := 0; stopDeg < 360; stopDeg += 15 {
			u := geom.Point{
				X: float32(math.Cos(float64(startDeg) * math.Pi / 180)),
				Y: float32(math.Sin(float64(startDeg) * math.Pi / 180)),
			}
			v := geom.Point{
				X: float32(math.Cos(float64(stopDeg) * math.Pi / 180)),
				Y: float32(math.Sin(float64(stopDeg) * math.Pi / 180)),
			}
			// Skip (near-)coincident and (near-)opposite vector pairs: the coincident-vector early return branches on
			// the sign of cross(u, v), whose tiny residue depends on the C compiler's fp-contract choice (Skia behaves
			// differently on mac-arm64 vs x86 here). The Go port pins the unfused evaluation — cross(u, u) is exactly
			// zero — so these cases diverge from this platform's C library by design.
			cross64 := float64(u.X)*float64(v.Y) - float64(u.Y)*float64(v.X)
			if math.Abs(cross64) < 1e-6 {
				continue
			}
			for dir := range 2 {
				for mi, m := range matrices {
					mag := 2.0
					if m != nil {
						mag = 40 // rotate + translate(12.5, -7.25) of unit-circle points
					}
					var dst [geom.MaxConicsForArc]geom.Conic
					gotN := geom.BuildUnitArc(u, v, geom.RotationDirection(dir), m, &dst)
					wantPts, wantW := refConicBuildUnitArc(u, v, dir, m)
					if gotN != len(wantW) {
						t.Errorf("BuildUnitArc(%d°→%d° dir=%d m=%d) count: go %d, c %d",
							startDeg, stopDeg, dir, mi, gotN, len(wantW))
						continue
					}
					for k := range gotN {
						if !agree(dst[k].W, wantW[k], 2) {
							t.Errorf("BuildUnitArc(%d°→%d° dir=%d m=%d) conic %d weight: go %v, c %v",
								startDeg, stopDeg, dir, mi, k, dst[k].W, wantW[k])
						}
						ptsAgree(t, "BuildUnitArc", dst[k].Pts[:], wantPts[k][:], mag, mag)
					}
				}
			}
		}
	}
}
