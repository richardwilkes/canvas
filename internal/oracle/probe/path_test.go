// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package probe

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/path"
)

// pathCorpus builds the probe corpus (~110 specs): every builder op, the shape adders in both directions, all three arc
// forms with their degenerate cases, addPath composition, contour bookkeeping edge cases, and seeded random builder
// mixes.
func pathCorpus() []*scenario.PathSpec {
	var out []*scenario.PathSpec
	add := func(s *scenario.PathSpec) { out = append(out, s) }
	sp := scenario.NewPathSpec

	// Basics and contour bookkeeping.
	add(sp()) // empty
	add(sp().MoveTo(10, 20))
	add(sp().MoveTo(10, 20).MoveTo(30, 40).MoveTo(50, 60)) // trailing-move replacement
	add(sp().MoveTo(0, 0).LineTo(100, 0).LineTo(100, 80).Close())
	add(sp().LineTo(50, 50))                                 // injected moveTo
	add(sp().Close())                                        // close with no contour
	add(sp().MoveTo(1, 1).Close().Close())                   // double close
	add(sp().MoveTo(1, 1).LineTo(2, 2).Close().LineTo(9, 9)) // draw after close: reopens at (1,1)
	add(sp().MoveTo(5, 5).LineTo(5, 5).Close())              // degenerate contour
	add(sp().RMoveTo(10, 10).RLineTo(5, 0).RLineTo(0, 5).Close().RMoveTo(20, 0).RLineTo(4, 4))

	// Curves, including conic degenerate-weight rules.
	add(sp().MoveTo(0, 0).QuadTo(50, 100, 100, 0))
	add(sp().MoveTo(0, 0).CubicTo(30, 90, 70, -90, 100, 0))
	add(sp().MoveTo(0, 0).ConicTo(50, 100, 100, 0, 0.5))
	add(sp().MoveTo(0, 0).ConicTo(50, 100, 100, 0, 1))            // w=1 → quad
	add(sp().MoveTo(0, 0).ConicTo(50, 100, 100, 0, 0))            // w≤0 → line
	add(sp().MoveTo(0, 0).ConicTo(50, 100, 100, 0, -2))           // negative → line
	add(sp().MoveTo(0, 0).ConicTo(50, 100, 100, 0, float32(inf))) // ∞ → two lines
	add(sp().MoveTo(0, 0).ConicTo(50, 100, 100, 0, float32(nan))) // NaN → line
	add(sp().MoveTo(0, 0).RConicTo(25, 50, 50, 0, 2.5).RCubicTo(10, 10, 20, -10, 30, 0))

	// Shapes, both directions.
	for _, dir := range []geom.PathDirection{geom.DirectionCW, geom.DirectionCCW} {
		add(sp().AddRect(geom.RectLTRB(10, 20, 110, 80), dir))
		add(sp().AddOval(geom.RectLTRB(10, 20, 110, 80), dir))
		add(sp().AddCircle(60, 60, 45.5, dir))
		add(sp().AddRoundRect(geom.RectLTRB(10, 20, 110, 80), 12, 18, dir))
		add(sp().AddRoundRect(geom.RectLTRB(10, 20, 110, 80), 80, 60, dir)) // radii clamp → oval
		add(sp().AddRoundRect(geom.RectLTRB(10, 20, 110, 80), 0, 5, dir))   // zero rx → rect
	}
	add(sp().AddRect(geom.RectLTRB(50, 50, 10, 10), geom.DirectionCW)) // unsorted rect
	add(sp().AddOval(geom.Rect{}, geom.DirectionCW))                   // empty oval
	add(sp().AddCircle(10, 10, 0, geom.DirectionCW))                   // zero radius
	add(sp().AddCircle(10, 10, -5, geom.DirectionCW))                  // negative radius
	add(sp().AddPoly([]geom.Point{{X: 1, Y: 2}, {X: 30, Y: 4}, {X: 15, Y: 40}}, true))
	add(sp().AddPoly([]geom.Point{{X: 1, Y: 2}, {X: 30, Y: 4}, {X: 15, Y: 40}, {X: 2, Y: 20}}, false))
	add(sp().AddPoly([]geom.Point{{X: 7, Y: 7}}, false))

	// Oval arcs (arcTo with oval, AddArc semantics via ArcToOval + forceMoveTo).
	oval := geom.RectLTRB(20, 30, 120, 100)
	add(sp().ArcToOval(oval, 0, 90, true))
	add(sp().ArcToOval(oval, 30, 270, false))
	add(sp().MoveTo(0, 0).ArcToOval(oval, -45, 120, false))
	add(sp().ArcToOval(oval, 90, 360, true))   // full sweep
	add(sp().ArcToOval(oval, 15, 359.9, true)) // the 359.x coincident-vector tweak loop
	add(sp().ArcToOval(oval, 0, -300, true))
	add(sp().ArcToOval(oval, 0, 0, true)) // zero sweep

	// Tangent arcs.
	add(sp().MoveTo(10, 100).ArcToTangent(60, 20, 110, 100, 25))
	add(sp().MoveTo(10, 100).ArcToTangent(60, 20, 110, 100, 0))   // zero radius → lineTo
	add(sp().MoveTo(10, 100).ArcToTangent(10, 100, 110, 100, 25)) // coincident start
	add(sp().MoveTo(10, 100).ArcToTangent(60, 100, 110, 100, 25)) // collinear
	add(sp().ArcToTangent(60, 20, 110, 100, 25))                  // no moveTo: injected

	// SVG endpoint arcs, absolute and relative, all flag combos and degenerate radii.
	for _, large := range []path.ArcSize{path.ArcSizeSmall, path.ArcSizeLarge} {
		for _, sweep := range []geom.PathDirection{geom.DirectionCW, geom.DirectionCCW} {
			add(sp().MoveTo(20, 60).ArcToRotated(40, 25, 30, large, sweep, 120, 90))
		}
	}
	add(sp().MoveTo(20, 60).ArcToRotated(0, 25, 0, path.ArcSizeSmall, geom.DirectionCW, 120, 90)) // rx=0 → line
	add(sp().MoveTo(20, 60).ArcToRotated(5, 5, 0, path.ArcSizeLarge, geom.DirectionCW, 120, 90))  // radii scale up
	add(sp().MoveTo(20, 60).ArcToRotated(40, 25, 0, path.ArcSizeSmall, geom.DirectionCW, 20, 60)) // zero-length arc
	add(sp().MoveTo(20, 60).RArcToRotated(40, 25, -15, path.ArcSizeLarge, geom.DirectionCCW, 80, 20))

	// addPath composition.
	tri := sp().MoveTo(0, 0).LineTo(40, 0).LineTo(20, 30).Close()
	openArc := sp().MoveTo(60, 0).QuadTo(80, 40, 100, 0)
	add(sp().AddRect(geom.RectLTRB(0, 0, 20, 20), geom.DirectionCW).AddPath(tri, path.AddPathAppend))
	add(sp().MoveTo(-10, -10).LineTo(-20, 5).AddPath(openArc, path.AddPathExtend))
	add(sp().AddPath(openArc, path.AddPathExtend)) // extend onto empty
	add(sp().AddPathOffset(tri, 100, 50, path.AddPathAppend))
	m := geom.RotateDegMatrix(30)
	add(sp().AddPathMatrix(tri, &m, path.AddPathAppend))
	pm := geom.IdentityMatrix()
	pm.SetAll(1, 0, 5, 0, 1, 5, 0.001, 0.002, 1)
	add(sp().AddPathMatrix(tri, &pm, path.AddPathAppend)) // perspective addPath
	add(sp().AddPathReverse(tri))
	add(sp().AddPathReverse(openArc))

	// Fill types on a self-intersecting star.
	star := sp().MoveTo(60, 0).LineTo(95, 110).LineTo(0, 40).LineTo(120, 40).LineTo(25, 110).Close()
	for _, ft := range []path.FillType{path.FillWinding, path.FillEvenOdd, path.FillInverseWinding, path.FillInverseEvenOdd} {
		s := *star
		add(s.SetFill(ft))
	}

	// Non-finite coordinates.
	add(sp().MoveTo(0, 0).LineTo(float32(inf), 10))
	add(sp().MoveTo(float32(nan), 0).LineTo(10, 10).Close())

	// Seeded random builder mixes.
	rng := rand.New(rand.NewSource(20260702))
	for range 24 {
		s := sp()
		coord := func() float32 { return (rng.Float32()*2 - 1) * 200 }
		for op := 0; op < 8; op++ {
			switch rng.Intn(8) {
			case 0:
				s.MoveTo(coord(), coord())
			case 1:
				s.LineTo(coord(), coord())
			case 2:
				s.QuadTo(coord(), coord(), coord(), coord())
			case 3:
				s.ConicTo(coord(), coord(), coord(), coord(), rng.Float32()*3)
			case 4:
				s.CubicTo(coord(), coord(), coord(), coord(), coord(), coord())
			case 5:
				s.RLineTo(coord()/10, coord()/10)
			case 6:
				// float32() forces the sweep's multiply to round before the subtract; see coord() in aaedge_test.go for
				// why an unrounded mul-then-sub makes this corpus (and so the fixture keys) platform-specific.
				// coord()'s own (x*2-1)*s is safe: x*2 is exact.
				s.ArcToOval(geom.RectLTRB(coord(), coord(), coord(), coord()), rng.Float32()*360,
					float32(rng.Float32()*720)-360, rng.Intn(2) == 0)
			case 7:
				s.Close()
			}
		}
		add(s)
	}
	return out
}

var (
	inf = math.Inf(1)
	nan = math.NaN()
)

// Noise classes for a spec's point streams, from the C-side computation that produces them:
//   - noiseExact: pure bookkeeping — bit-exact required.
//   - noiseFloat: float arithmetic with compiler-contraction and/or libm (trig) last-ULP noise — the standard agree()
//     tolerance. Covers oval/tangent arcs (sin/cos/atan2 differ between Apple libm and Go's math package in the last
//     ULP) and addPath-with-matrix (mapPoints is float SIMD in C).
//   - noiseSqrtAmp: the SVG endpoint arc's center computation takes sqrt of a discriminant that is mathematically zero
//     when the radii scale up, so float residue amplifies to ~sqrt(eps) relative noise; tolerances widen by
//     sqrtAmpFactor. Plain shape adders (oval/circle/rrect) use fixed conic layouts with the constant weight sqrt(2)/2
//     and stay exact.
const (
	noiseExact = iota
	noiseFloat
	noiseSqrtAmp

	sqrtAmpFactor = 4096 // ≈ sqrt(eps32) / eps32, with headroom folded into probeK
)

func specNoise(spec *scenario.PathSpec) int {
	class := noiseExact
	up := func(c int) {
		if c > class {
			class = c
		}
	}
	for _, op := range spec.Ops {
		switch op.Kind {
		case scenario.OpArcToRotated, scenario.OpRArcToRotated:
			up(noiseSqrtAmp)
		case scenario.OpArcToOval, scenario.OpArcToTangent:
			up(noiseFloat)
		case scenario.OpAddPathMatrix:
			up(noiseFloat)
			up(specNoise(op.Sub))
		case scenario.OpAddPath, scenario.OpAddPathOffset, scenario.OpAddPathReverse:
			up(specNoise(op.Sub))
		}
	}
	return class
}

// noiseMagScale converts a noise class into the multiplier applied to agree() magnitudes.
func noiseMagScale(class int) float64 {
	if class == noiseSqrtAmp {
		return sqrtAmpFactor
	}
	return 1
}

// coordMag returns a per-axis agree() magnitude for a point stream.
func coordMag(pts []geom.Point) (magX, magY float64) {
	for _, p := range pts {
		magX = math.Max(magX, math.Abs(float64(p.X)))
		magY = math.Max(magY, math.Abs(float64(p.Y)))
	}
	return 2 * magX, 2 * magY
}

// TestPathStructureProbe: point streams, counts, last point, emptiness, and fill type must match bit-exactly for every
// corpus entry — path construction is pure bookkeeping plus the (double-precision) arc machinery.
func TestPathStructureProbe(t *testing.T) {
	for i, spec := range pathCorpus() {
		gp := spec.BuildGo()
		want := refPathStructure(spec)
		if gp.IsEmpty() != want.isEmpty {
			t.Errorf("corpus[%d] IsEmpty: go %v, c %v", i, gp.IsEmpty(), want.isEmpty)
		}
		noise := specNoise(spec)
		if got, wantN := gp.CountPoints(), len(want.points); got != wantN {
			t.Errorf("corpus[%d] CountPoints: go %d, c %d", i, got, wantN)
		} else {
			goPts := make([]geom.Point, got)
			gp.Points(goPts)
			if noise != noiseExact {
				magX, magY := coordMag(want.points)
				scale := noiseMagScale(noise)
				ptsAgree(t, fmt.Sprintf("corpus[%d] points (noisy)", i), goPts, want.points, magX*scale, magY*scale)
			} else if !ptsEq(goPts, want.points) {
				t.Errorf("corpus[%d] points:\n  go %v\n  c  %v", i, goPts, want.points)
			}
		}
		gLast, gOK := gp.LastPt()
		if gOK != want.lastOK {
			t.Errorf("corpus[%d] LastPt ok: go %v, c %v", i, gOK, want.lastOK)
		} else if gOK {
			cLast := want.lastPt
			if noise != noiseExact {
				scale := noiseMagScale(noise)
				if !agree(gLast.X, cLast.X, 2*scale*math.Abs(float64(cLast.X))) ||
					!agree(gLast.Y, cLast.Y, 2*scale*math.Abs(float64(cLast.Y))) {
					t.Errorf("corpus[%d] LastPt: go %v, c %v", i, gLast, cLast)
				}
			} else if !ptEq(gLast, cLast) {
				t.Errorf("corpus[%d] LastPt: go %v, c %v", i, gLast, cLast)
			}
		}
		if got := int(gp.FillType()); got != want.fillType {
			t.Errorf("corpus[%d] FillType: go %d, c %d", i, got, want.fillType)
		}
	}
}

// TestPathBoundsProbe: control-point bounds and tight bounds — bit-exact except for trig-arc specs, whose control
// points already carry libm/contraction noise.
func TestPathBoundsProbe(t *testing.T) {
	for i, spec := range pathCorpus() {
		gp := spec.BuildGo()
		cBounds, cTight := refPathBounds(spec)
		noise := specNoise(spec)
		check := func(what string, got, want geom.Rect, tolerant bool) {
			if tolerant {
				mag := 2 * noiseMagScale(noise) * math.Max(math.Abs(float64(want.Left)), math.Max(math.Abs(float64(want.Top)),
					math.Max(math.Abs(float64(want.Right)), math.Abs(float64(want.Bottom)))))
				if !agree(got.Left, want.Left, mag) || !agree(got.Top, want.Top, mag) ||
					!agree(got.Right, want.Right, mag) || !agree(got.Bottom, want.Bottom, mag) {
					t.Errorf("corpus[%d] %s (trig): go %v, c %v", i, what, got, want)
				}
			} else if !rectEq(got, want) {
				t.Errorf("corpus[%d] %s: go %v, c %v", i, what, got, want)
			}
		}
		check("Bounds", gp.Bounds(), cBounds, noise != noiseExact)
		// Tight bounds evaluate curve extrema in float even when construction is exact bookkeeping, so any spec with
		// curve segments gets the contraction tolerance.
		hasCurves := gp.SegmentMasks()&(path.SegmentQuad|path.SegmentConic|path.SegmentCubic) != 0
		check("TightBounds", gp.ComputeTightBounds(), cTight, noise != noiseExact || hasCurves)
	}
}

// TestPathContainsProbe: winding/even-odd point containment over a grid spanning each path's bounds (plus its exact
// corner/edge points). Noisy specs are skipped: containment is discontinuous in the control points, so grid points that
// land near a boundary can legitimately flip when the two implementations' control points differ in the last ULP. The
// exact-class corpus covers every winding code path (lines, quads, conics, cubics) with identical control points on
// both sides.
func TestPathContainsProbe(t *testing.T) {
	for i, spec := range pathCorpus() {
		if specNoise(spec) != noiseExact {
			continue
		}
		gp := spec.BuildGo()
		b := gp.Bounds()
		if b.IsEmpty() || !b.IsFinite() {
			continue
		}
		// The whole grid is queried as one batch: 147 points per path, and a fixture entry apiece would be pure noise.
		// The points derive from the library's bounds, so replay regenerates them identically and the answers stay
		// index-aligned with what was captured.
		const n = 12
		pts := make([]geom.Point, 0, n*n+3)
		for yi := range n {
			for xi := range n {
				pts = append(pts, geom.Point{
					X: b.Left + (b.Right-b.Left)*(float32(xi)+0.5)/n,
					Y: b.Top + (b.Bottom-b.Top)*(float32(yi)+0.5)/n,
				})
			}
		}
		pts = append(pts, geom.Point{X: b.Left, Y: b.Top}, geom.Point{X: b.Right, Y: b.Bottom},
			geom.Point{X: b.CenterX(), Y: b.CenterY()})
		want := refPathContains(spec, pts)
		if len(want) != len(pts) {
			t.Fatalf("corpus[%d] Contains: fixture has %d answers for %d points", i, len(want), len(pts))
		}
		for k, p := range pts {
			if got := gp.Contains(p.X, p.Y); got != want[k] {
				t.Errorf("corpus[%d] Contains(%g, %g): go %v, c %v", i, p.X, p.Y, got, want[k])
			}
		}
	}
}

// TestPathTransformProbe: TransformTo through translate/scale/rotate/skew and an all-positive-w perspective matrix;
// point streams compared bit-exactly except rotate/skew/perspective, which map points through float SIMD in C and get
// the contraction tolerance.
func TestPathTransformProbe(t *testing.T) {
	translate := geom.TranslateMatrix(12.5, -7.25)
	scale := geom.ScaleMatrix(2.5, 0.5)
	rotate := geom.RotateDegMatrix(30)
	var skew geom.Matrix
	skew.SetSkew(0.5, -0.25)
	var persp geom.Matrix
	// Tiny perspective factors keep every corpus coordinate (|x|,|y| ≤ ~360) far from the w=0 plane.
	persp.SetAll(1, 0, 5, 0, 1, 5, 1e-6, 2e-6, 1)
	matrices := []struct {
		m     *geom.Matrix
		name  string
		exact bool
	}{
		{name: "translate", m: &translate, exact: true},
		{name: "scale", m: &scale, exact: true},
		{name: "rotate", m: &rotate, exact: false},
		{name: "skew", m: &skew, exact: false},
		{name: "persp", m: &persp, exact: false},
	}
	for i, spec := range pathCorpus() {
		gp := spec.BuildGo()
		for _, tc := range matrices {
			gDst := path.New()
			gp.TransformTo(tc.m, gDst)
			cPts := refPathTransform(spec, tc.m)
			gN, cN := gDst.CountPoints(), len(cPts)
			if gN != cN {
				t.Errorf("corpus[%d] transform %s point count: go %d, c %d", i, tc.name, gN, cN)
				continue
			}
			goPts := make([]geom.Point, gN)
			gDst.Points(goPts)
			srcPts := make([]geom.Point, gp.CountPoints())
			gp.Points(srcPts)
			// The perspective transform promotes quads to conics and subdivides cubics, so dst points do not correspond
			// 1:1 to src points; use a whole-path magnitude for it (the matrix is near identity, so mapped magnitudes
			// track source magnitudes).
			var pathMagX, pathMagY float64
			for _, p := range srcPts {
				_, _, mx, my := refMapPoint(tc.m, p)
				pathMagX = math.Max(pathMagX, mx)
				pathMagY = math.Max(pathMagY, my)
			}
			noise := specNoise(spec)
			for k := range goPts {
				if tc.exact && noise == noiseExact {
					if !ptEq(goPts[k], cPts[k]) {
						t.Errorf("corpus[%d] transform %s pt %d: go %v, c %v", i, tc.name, k, goPts[k], cPts[k])
					}
					continue
				}
				magX, magY := pathMagX, pathMagY
				if tc.name != "persp" && noise == noiseExact && k < len(srcPts) {
					_, _, magX, magY = refMapPoint(tc.m, srcPts[k])
				}
				scale := noiseMagScale(noise)
				if !agree(goPts[k].X, cPts[k].X, magX*scale) || !agree(goPts[k].Y, cPts[k].Y, magY*scale) {
					t.Errorf("corpus[%d] transform %s pt %d: go %v, c %v", i, tc.name, k, goPts[k], cPts[k])
				}
			}
		}
	}
}
