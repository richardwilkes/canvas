// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for the distance-field generator: the byte packing formula, and structural properties of the
// generated field (threshold at the coverage edge, monotonicity away from the shape, symmetry) that the 8SSEDT
// transform must satisfy.

package font

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func TestPackDistanceFieldVal(t *testing.T) {
	// byte = round((clamp(-dist, -4, 4*127/128) + 4) / 8 * 256).
	cases := []struct {
		dist float32
		want uint8
	}{
		{dist: 0, want: 128},        // zero distance is the 128 threshold
		{dist: 4, want: 0},          // far outside clamps to 0
		{dist: 100, want: 0},        // beyond the magnitude clamps too
		{dist: -4, want: 255},       // deep inside clamps to 4*127/128 → 255
		{dist: -100, want: 255},     //
		{dist: 1, want: 96},         // one texel outside: (-1+4)/8*256 = 96
		{dist: -1, want: 160},       // one texel inside: (1+4)/8*256 = 160
		{dist: 0.5, want: 112},      // half a texel outside
		{dist: -3.96875, want: 255}, // exactly the positive clamp value
	}
	for _, c := range cases {
		if got := packDistanceFieldVal(c.dist); got != c.want {
			t.Errorf("packDistanceFieldVal(%v) = %d, want %d", c.dist, got, c.want)
		}
	}
}

// solidSquareField generates the distance field of a size x size solid-255 A8 block.
func solidSquareField(size int) (field []uint8, fieldWidth int) {
	img := make([]uint8, size*size)
	for i := range img {
		img[i] = 255
	}
	dfW := size + 2*DistanceFieldPad
	df := make([]uint8, ComputeDistanceFieldSize(size, size))
	GenerateDistanceFieldFromA8Image(df, img, size, size, size)
	return df, dfW
}

func TestGenerateDistanceFieldSolidSquare(t *testing.T) {
	const size = 8
	df, dfW := solidSquareField(size)

	// The field dimensions include the DistanceFieldPad ring on each side.
	if len(df) != dfW*dfW {
		t.Fatalf("field size = %d, want %d", len(df), dfW*dfW)
	}

	// Interior texels are inside (> 128); the pad ring corners are far outside (< 128, and the outermost corner is at
	// least sqrt(2)*pad away — clamped to 0).
	center := df[(dfW/2)*dfW+dfW/2]
	if center <= 128 {
		t.Errorf("center = %d, want > 128 (inside)", center)
	}
	if corner := df[0]; corner != 0 {
		t.Errorf("pad corner = %d, want 0 (clamped far outside)", corner)
	}

	// Monotonic non-increasing along the center row moving outward from the center.
	row := (dfW / 2) * dfW
	for x := dfW / 2; x > 0; x-- {
		if df[row+x-1] > df[row+x] {
			t.Fatalf("row not monotonic at x=%d: %d > %d", x-1, df[row+x-1], df[row+x])
		}
	}

	// The field has the dihedral symmetry of the square over its whole extent, padding included: both 8SSEDT passes
	// sweep the same interior window, so no column is left under-propagated (an off-by-two backward start shifts every
	// row's window two columns left, which shows up here as an x-mirror mismatch in the outermost columns).
	at := func(x, y int) uint8 { return df[y*dfW+x] }
	for y := 0; y < dfW; y++ {
		for x := 0; x < dfW; x++ {
			if at(x, y) != at(dfW-1-x, y) {
				t.Fatalf("x-mirror asymmetry at (%d,%d): %d vs %d", x, y, at(x, y), at(dfW-1-x, y))
			}
			if at(x, y) != at(y, x) {
				t.Fatalf("transpose asymmetry at (%d,%d): %d vs %d", x, y, at(x, y), at(y, x))
			}
		}
	}

	// The coverage edge sits between the last solid texel and the first pad texel: crossing the image boundary crosses
	// the 128 threshold within one texel.
	edgeInside := at(DistanceFieldPad, dfW/2)
	edgeOutside := at(DistanceFieldPad-1, dfW/2)
	if edgeInside < 128 || edgeOutside > 128 {
		t.Errorf("threshold not at the boundary: inside=%d outside=%d", edgeInside, edgeOutside)
	}
}

// TestSetLengthFastIsExactlyNormalized pins the precision setLengthFast returns at. The name is upstream Skia's, where
// the scale is a fast inverse-square-root approximation, and the invitation to substitute one back is standing — but a
// single Newton step carries a relative error around 1.8e-3, and the seeded edge distances that error moves are enough
// to shift ~1.5% of a generated field's texels and every SDFT golden with them. Requiring the result within a few
// float32 ulps of the requested length is four orders of magnitude tighter than any such approximation can reach, so
// the substitution cannot be made without this failing.
func TestSetLengthFastIsExactlyNormalized(t *testing.T) {
	const eps = 1.1920929e-7 // float32 epsilon
	// Gradient-shaped vectors: initDistances only ever asks for length 1, over sums of neighboring alpha differences.
	for _, p := range []geom.Point{
		geom.Pt(1, 0), geom.Pt(0, -1), geom.Pt(1, 1), geom.Pt(-3, 4), geom.Pt(0.001, -0.002),
		geom.Pt(1e-6, 1e-6), geom.Pt(1e6, -1e6), geom.Pt(sqrt2, sqrt2), geom.Pt(0.5, -1.75),
	} {
		for _, length := range []float32{1, 0.25, 7} {
			got := setLengthFast(p, length)
			// math.Hypot in float64 is exact enough to measure a float32 result against.
			gotLen := float32(math.Hypot(float64(got.X), float64(got.Y)))
			if d := absF32(gotLen - length); d > 4*eps*length {
				t.Errorf("setLengthFast(%v, %v) = %v, |%v| = %v: off by %v (%.1f ulps), want within 4 ulps",
					p, length, got, got, gotLen, d, float64(d)/float64(eps*length))
			}
		}
	}
	// The degenerate lanes return the zero vector rather than a NaN or an infinity.
	for _, p := range []geom.Point{
		geom.Pt(0, 0),
		geom.Pt(float32(math.Inf(1)), 0),
		geom.Pt(float32(math.NaN()), 1),
		geom.Pt(1e30, 1e30), // the squared magnitude overflows float32
	} {
		if got := setLengthFast(p, 1); got != (geom.Point{}) {
			t.Errorf("setLengthFast(%v, 1) = %v, want the zero point", p, got)
		}
	}
}

// TestDistanceFieldOuterBandIsSaturated pins the margin the backward-in-y start's comment rests on. Upstream Skia's
// shifted start denies the rightmost interior column of every row its backward propagation, and the texels the two
// starts disagree about land in the field's outer band — inside the region DistanceFieldInset trims as well as outside
// it. What keeps that invisible in a rendered glyph is not the trim but this margin: those texels already sit whole
// texels away from the zero crossing, far outside the smoothstep band the shader reads around DistanceFieldThreshold.
func TestDistanceFieldOuterBandIsSaturated(t *testing.T) {
	const size = 8
	df, dfW := solidSquareField(size)
	// Invert the packing: zero distance is at 128 and the byte range spans 2*DistanceFieldMagnitude texels.
	distanceAt := func(x, y int) float32 {
		return DistanceFieldMagnitude - float32(df[y*dfW+x])*(2*DistanceFieldMagnitude)/256
	}
	const wantMargin = 2 // texels; the shader's smoothstep band spans well under one
	nonClamped := 0
	for y := range dfW {
		for x := range dfW {
			if x >= 2 && x < dfW-2 && y >= 2 && y < dfW-2 {
				continue // not the outer band
			}
			if d := distanceAt(x, y); d < wantMargin {
				t.Errorf("outer-band texel (%d,%d) is %v texels outside, want at least %v", x, y, d, wantMargin)
			}
			if df[y*dfW+x] != 0 {
				nonClamped++
			}
		}
	}
	// Non-vacuity: the band must carry real propagated distances, not a uniform clamp. A band that were entirely
	// clamped would satisfy the margin above — and would make the mirror-invariance test blind exactly where the
	// backward-pass window matters.
	if nonClamped == 0 {
		t.Error("the outer band is entirely clamped; it exercises no propagation at all")
	}
}

func TestGenerateDistanceFieldMirrorInvariance(t *testing.T) {
	// The transform is built from four passes that between them cover every direction, so it must commute with a mirror
	// of its input: the field of a flipped shape is the flip of the shape's field, texel for texel, padding included.
	// An asymmetric shape (an L) makes the property bite in both axes, and a scan window that is not the same interval
	// in every pass breaks it (the x flip catches a shifted backward-in-y start).
	const size = 7
	shape := make([]uint8, size*size)
	for y := range size {
		for x := range size {
			if x < 2 || y >= size-2 {
				shape[y*size+x] = 255
			}
		}
	}
	dfW := size + 2*DistanceFieldPad
	field := func(img []uint8) []uint8 {
		df := make([]uint8, ComputeDistanceFieldSize(size, size))
		GenerateDistanceFieldFromA8Image(df, img, size, size, size)
		return df
	}
	flip := func(src []uint8, w int, horizontal bool) []uint8 {
		h := len(src) / w
		out := make([]uint8, len(src))
		for y := range h {
			for x := range w {
				if horizontal {
					out[y*w+x] = src[y*w+(w-1-x)]
				} else {
					out[y*w+x] = src[(h-1-y)*w+x]
				}
			}
		}
		return out
	}
	base := field(shape)
	for _, horizontal := range []bool{true, false} {
		axis := "y"
		if horizontal {
			axis = "x"
		}
		got := field(flip(shape, size, horizontal))
		want := flip(base, dfW, horizontal)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s-flip: field differs at (%d,%d): %d, want %d",
					axis, i%dfW, i/dfW, got[i], want[i])
			}
		}
	}
}

func TestGenerateDistanceFieldAntiAliasedEdgeShifts(t *testing.T) {
	// A soft (partial coverage) edge must move the zero crossing relative to a hard edge: a column of 64-coverage reads
	// as an edge closer to the solid side than a column of 192.
	const size = 6
	makeField := func(edge uint8) []uint8 {
		img := make([]uint8, size*size)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				switch {
				case x < size/2:
					img[y*size+x] = 255
				case x == size/2:
					img[y*size+x] = edge
				default:
					img[y*size+x] = 0
				}
			}
		}
		df := make([]uint8, ComputeDistanceFieldSize(size, size))
		GenerateDistanceFieldFromA8Image(df, img, size, size, size)
		return df
	}
	dfW := size + 2*DistanceFieldPad
	low := makeField(64)
	high := makeField(192)
	y := dfW / 2
	x := DistanceFieldPad + size/2 // the partial-coverage column
	if low[y*dfW+x] >= high[y*dfW+x] {
		t.Errorf("soft-edge ordering wrong: cov64=%d !< cov192=%d",
			low[y*dfW+x], high[y*dfW+x])
	}
}

// TestInitDistancesOutsideBand covers the one-texel band around the working buffer. Seeding an edge texel reads all
// eight of its neighbors, so a marked edge in that band would read off the ends of the buffer (top and bottom rows) or
// wrap onto the neighboring row (leftmost and rightmost columns). The generator's own lane can never mark one there —
// initGlyphData insets the image by DistanceFieldPad — but initDistances must not rely on that to stay in bounds: it
// seeds the band as "far away" and leaves the edge lane to the interior.
func TestInitDistancesOutsideBand(t *testing.T) {
	const width, height = 6, 5
	data := make([]dfData, width*height)
	edges := make([]uint8, width*height)
	for i := range data {
		data[i].alpha = 0.5 // right at the coverage threshold, so an edge seed lands at distance 0
		edges[i] = 255      // every texel an edge, the whole outside band included
	}
	initDistances(data, edges, width, height)
	for j := range height {
		for i := range width {
			at := j*width + i
			if i > 0 && i < width-1 && j > 0 && j < height-1 {
				if data[at].distSq != 0 {
					t.Errorf("interior texel (%d,%d) distSq = %v, want the edge seed 0", i, j, data[at].distSq)
				}
				continue
			}
			if data[at].distSq != 2000000 || data[at].distVector != geom.Pt(1000, 1000) {
				t.Errorf("outside-band texel (%d,%d) = (distSq %v, vector %v), want the far-away seed (2000000, 1000,1000)",
					i, j, data[at].distSq, data[at].distVector)
			}
		}
	}
}
