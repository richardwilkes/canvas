// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

func randomTestPath(rng *rand.Rand, w, h float32, curves bool) *path.Path {
	p := path.New()
	coord := func(limit float32) float32 { return rng.Float32()*(limit+40) - 20 }
	for c := 0; c < 3; c++ {
		p.MoveTo(coord(w), coord(h))
		for s := 0; s < 8; s++ {
			switch {
			case !curves || rng.Intn(2) == 0:
				p.LineTo(coord(w), coord(h))
			case rng.Intn(2) == 0:
				p.QuadTo(coord(w), coord(h), coord(w), coord(h))
			default:
				p.CubicTo(coord(w), coord(h), coord(w), coord(h), coord(w), coord(h))
			}
		}
		p.Close()
	}
	return p
}

// TestFillPathParallelDeterministic: the concurrent banded fill must produce exactly the bytes of the same bands filled
// sequentially (concurrency introduces no nondeterminism), and each band is just a clipped fill (see parallel.go for
// why band seams may differ from the unbanded fill).
func TestFillPathParallelDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const w, h = 160, 220
	clip := geom.IRectLTRB(0, 0, w, h)
	for iter := 0; iter < 40; iter++ {
		p := randomTestPath(rng, w, h, iter%2 == 1)
		p.SetFillType(path.FillType(rng.Intn(4)))
		for _, aa := range []bool{false, true} {
			for _, workers := range []int{2, 5} {
				// sequential reference over the same bands
				want := NewPixmap(w, h)
				bandRows := (clip.Height() + int32(workers) - 1) / int32(workers)
				for top := clip.Top; top < clip.Bottom; top += bandRows {
					band := geom.IRectLTRB(clip.Left, top, clip.Right, min(top+bandRows, clip.Bottom))
					fillPathSerial(p, band, NewSolidBlitter(want, colorcore.Color(0xFF224466)), aa)
				}
				par := NewPixmap(w, h)
				FillPathParallel(p, clip, NewSolidBlitter(par, colorcore.Color(0xFF224466)), aa, workers)
				if !bytes.Equal(want.RGBA8888Bytes(), par.RGBA8888Bytes()) {
					t.Fatalf("iter %d aa=%v workers=%d: concurrent banding nondeterministic", iter, aa, workers)
				}
			}
		}
	}
}

// TestFillPathParallelPrimesPathCaches: every band reads the same *path.Path, and a Path memoizes its bounds (plus the
// finiteness flag) and its convexity on the first reader that needs them, so FillPathParallel must prime both before
// spawning. Without that, `go test -race` reports concurrent writes to Path.bounds and Path.convexity from two band
// goroutines, and a torn geom.Rect read means wrong path bounds and wrong output, not just a detector warning. The path
// must be untouched here: any prior fill, Bounds or GetConvexity call primes the caches and hides the bug (which is why
// TestFillPathParallelDeterministic, whose serial reference fill runs first, does not catch it).
func TestFillPathParallelPrimesPathCaches(t *testing.T) {
	const w, h = 96, 8 * minBandRows
	clip := geom.IRectLTRB(0, 0, w, h)
	color := colorcore.Color(0xFF224466)
	for _, aa := range []bool{false, true} {
		for iter := 0; iter < 8; iter++ {
			fresh := randomTestPath(rand.New(rand.NewSource(int64(iter))), w, h, iter%2 == 1)
			got := NewPixmap(w, h)
			FillPathParallel(fresh, clip, NewSolidBlitter(got, color), aa, 8)

			// The same geometry, primed before the fill, must produce the same pixels.
			primed := randomTestPath(rand.New(rand.NewSource(int64(iter))), w, h, iter%2 == 1)
			primed.Bounds()
			primed.GetConvexity()
			want := NewPixmap(w, h)
			FillPathParallel(primed, clip, NewSolidBlitter(want, color), aa, 8)
			if !bytes.Equal(want.RGBA8888Bytes(), got.RGBA8888Bytes()) {
				t.Fatalf("iter %d aa=%v: filling an unprimed path differed from filling a primed one", iter, aa)
			}
		}
	}
}

// TestRectFillBandBounds checks the band-boundary policy: short rects and single-worker requests do not band; otherwise
// the ends are preserved exactly, interior cuts fall on integer scanlines, the bands are monotonic and never
// single-scanline (each spans well over one pixel row, so no band isolates a fractional edge row), and the band count
// never exceeds the worker request.
func TestRectFillBandBounds(t *testing.T) {
	if b := RectFillBandBounds(10, 10+2*minBandRows-1, 8); b != nil {
		t.Fatalf("a rect shorter than two bands must not band: %v", b)
	}
	if b := RectFillBandBounds(10, 400, 1); b != nil {
		t.Fatalf("a single worker must not band: %v", b)
	}
	for _, tc := range []struct{ top, bottom float32 }{
		{top: 8, bottom: 248}, {top: 8.4, bottom: 247.6}, {top: 0.3, bottom: 255.9}, {top: 8.9, bottom: 200.1}, {top: -3.5, bottom: 300.5},
	} {
		for _, workers := range []int{2, 3, 4, 5, 8, 16} {
			b := RectFillBandBounds(tc.top, tc.bottom, workers)
			if b == nil {
				t.Fatalf("top=%v bottom=%v workers=%d: expected bands", tc.top, tc.bottom, workers)
			}
			if b[0] != tc.top || b[len(b)-1] != tc.bottom {
				t.Fatalf("top=%v bottom=%v workers=%d: ends not preserved: %v", tc.top, tc.bottom, workers, b)
			}
			if n := len(b) - 1; n > workers || n < 2 {
				t.Fatalf("top=%v bottom=%v workers=%d: band count %d out of range", tc.top, tc.bottom, workers, n)
			}
			for i := 1; i < len(b)-1; i++ {
				if b[i] != float32(int32(b[i])) {
					t.Fatalf("workers=%d: interior cut %v is not an integer scanline", workers, b[i])
				}
			}
			for i := 0; i+1 < len(b); i++ {
				// >= 2.0 guarantees the band spans at least two pixel rows, so antiFillDot8 never takes its
				// single-scanline path for a fractional edge band (the byte-exactness precondition).
				if b[i+1]-b[i] < 2 {
					t.Fatalf("workers=%d: band [%v,%v) is too short to be byte-exact", workers, b[i], b[i+1])
				}
			}
		}
	}
}

// TestIRectFillBandBounds checks the integer band-boundary policy (the drawPaint lane): short rects and single-worker
// requests do not band; otherwise the ends are preserved exactly, the boundaries are strictly increasing (no empty
// band), and the band count is in [2, workers].
func TestIRectFillBandBounds(t *testing.T) {
	if b := IRectFillBandBounds(10, 10+2*minBandRows-1, 8); b != nil {
		t.Fatalf("a rect shorter than two bands must not band: %v", b)
	}
	if b := IRectFillBandBounds(0, 400, 1); b != nil {
		t.Fatalf("a single worker must not band: %v", b)
	}
	for _, tc := range []struct{ top, bottom int32 }{
		{top: 0, bottom: 256}, {top: 0, bottom: 240}, {top: 8, bottom: 248}, {top: -16, bottom: 300}, {top: 5, bottom: 5 + 2*minBandRows},
	} {
		for _, workers := range []int{2, 3, 4, 5, 8, 16} {
			b := IRectFillBandBounds(tc.top, tc.bottom, workers)
			if b == nil {
				t.Fatalf("top=%d bottom=%d workers=%d: expected bands", tc.top, tc.bottom, workers)
			}
			if b[0] != tc.top || b[len(b)-1] != tc.bottom {
				t.Fatalf("top=%d bottom=%d workers=%d: ends not preserved: %v", tc.top, tc.bottom, workers, b)
			}
			if n := len(b) - 1; n > workers || n < 2 {
				t.Fatalf("top=%d bottom=%d workers=%d: band count %d out of range", tc.top, tc.bottom, workers, n)
			}
			for i := 0; i+1 < len(b); i++ {
				if b[i+1] <= b[i] {
					t.Fatalf("workers=%d: band [%d,%d) is empty or non-monotonic", workers, b[i], b[i+1])
				}
			}
		}
	}
}

func benchFill(b *testing.B, aa bool, workers int) {
	rng := rand.New(rand.NewSource(7))
	const w, h = 1024, 1024
	p := randomTestPath(rng, w, h, true)
	clip := geom.IRectLTRB(0, 0, w, h)
	dev := NewPixmap(w, h)
	blitter := NewSolidBlitter(dev, colorcore.Color(0x80336699))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if workers == 1 {
			fillPathSerial(p, clip, blitter, aa)
		} else {
			FillPathParallel(p, clip, blitter, aa, workers)
		}
	}
}

func BenchmarkFillPathSerial(b *testing.B)       { benchFill(b, false, 1) }
func BenchmarkFillPathParallel(b *testing.B)     { benchFill(b, false, 0) }
func BenchmarkAntiFillPathSerial(b *testing.B)   { benchFill(b, true, 1) }
func BenchmarkAntiFillPathParallel(b *testing.B) { benchFill(b, true, 0) }

// TestFillPathParallelBandPolicy: FillPathParallel must split with the shared bandCount policy its doc names. A ceiling
// divide put the band floor in the wrong place — 33 rows split into 17/16, both under minBandRows — and never consulted
// bandCount at all, so a clip spanning fewer than two full bands still fanned out. Now a short clip is byte-for-byte
// the serial fill, and a tall one matches bandCount's split exactly.
func TestFillPathParallelBandPolicy(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	const w, h = 160, 320
	color := colorcore.Color(0xFF224466)
	p := randomTestPath(rng, w, h, true)

	for _, aa := range []bool{false, true} {
		for _, rows := range []int32{1, minBandRows, minBandRows + 1, 2*minBandRows - 1} {
			clip := geom.IRectLTRB(0, 0, w, rows)
			want := NewPixmap(w, rows)
			fillPathSerial(p, clip, NewSolidBlitter(want, color), aa)
			got := NewPixmap(w, rows)
			FillPathParallel(p, clip, NewSolidBlitter(got, color), aa, 8)
			if !bytes.Equal(want.RGBA8888Bytes(), got.RGBA8888Bytes()) {
				t.Fatalf("aa=%v rows=%d: a clip shorter than two bands must fall back to the serial fill", aa, rows)
			}
		}

		for _, workers := range []int{0, 2, 3, 8} {
			clip := geom.IRectLTRB(0, 0, w, h)
			bands := bandCount(h, workers)
			if bands < 2 {
				t.Fatalf("bad test setup: %d rows should band with workers=%d", h, workers)
			}
			bandRows := (h + bands - 1) / bands
			if bandRows < minBandRows {
				t.Fatalf("workers=%d: nominal band of %d rows is below the %d floor", workers, bandRows, minBandRows)
			}
			want := NewPixmap(w, h)
			for top := clip.Top; top < clip.Bottom; top += bandRows {
				band := geom.IRectLTRB(clip.Left, top, clip.Right, min(top+bandRows, clip.Bottom))
				fillPathSerial(p, band, NewSolidBlitter(want, color), aa)
			}
			got := NewPixmap(w, h)
			FillPathParallel(p, clip, NewSolidBlitter(got, color), aa, workers)
			if !bytes.Equal(want.RGBA8888Bytes(), got.RGBA8888Bytes()) {
				t.Fatalf("aa=%v workers=%d: split does not match bandCount's %d bands", aa, workers, bands)
			}
		}
	}
}
