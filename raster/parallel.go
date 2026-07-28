// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Row-band parallelism for path fills: the CPU rasterizer parallelizes internally by row bands with goroutines. Each
// band is an independent fill clipped to the band, which is byte-identical to filling that same clipped region directly
// — but *not* always byte-identical to the unbanded fill: the scan converters accumulate edge X positions incrementally
// from the top of the (clipped) edge list, so restarting at a band boundary can shift a boundary pixel's rounding, and
// AA coverage near the seam can differ by a step. Output is fully deterministic for a given clip/worker split. Callers
// that need the exact serial bytes (the oracle goldens do) use the serial entry points; this is the opt-in throughput
// path.

package raster

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// minBandRows is the smallest band worth dispatching to its own goroutine; cheaper fills don't amortize the startup
// cost. For RectFillBandBounds it is also a correctness floor: a band this tall can never isolate a rect's fractional
// top/bottom edge row as its own single scanline, which antiFillDot8 covers with a different (off-by-one) coverage
// formula than the multi-row top/bottom path.
const minBandRows = 32

// FillPathParallel fills p into blitter restricted to clip, splitting the clip into horizontal bands (each at least
// minBandRows tall) filled concurrently, and falling back to a serial fill when the clip is too short to band. blitter
// must tolerate concurrent blits to disjoint rows — the package's device blitters (SolidBlitter, BlendBlitter,
// A8CoverageBlitter) all do, since their only state is the destination pixels. p is shared read-only across the bands
// and must not be mutated until this returns; its lazy caches are primed here, so callers need not prime them. The band
// count comes from the shared bandCount policy, so workers <= 0 uses the fixed machine-independent default split. aa
// selects the analytic-AA converter.
func FillPathParallel(p *path.Path, clip geom.IRect, blitter Blitter, aa bool, workers int) {
	rows := clip.Height()
	bands := bandCount(rows, workers)
	if bands <= 1 {
		fillPathSerial(p, clip, blitter, aa)
		return
	}

	// Resolve the lazily-cached state the concurrent bands would otherwise race to compute: the path's bounds (and with
	// it its finiteness flag) and its convexity, both memoized on the shared Path and queried per band by the scan
	// converters. Priming here makes the bands' accesses read-only; without it, `go test -race` reports concurrent
	// writes to Path.bounds and Path.convexity, and a torn geom.Rect read yields wrong bounds and wrong output. The
	// generation ID is the Path's only other lazy cache and nothing in this package reads it.
	p.Bounds()
	p.GetConvexity()

	bandRows := (rows + bands - 1) / bands
	var wg sync.WaitGroup
	for top := clip.Top; top < clip.Bottom; top += bandRows {
		band := geom.IRectLTRB(clip.Left, top, clip.Right, min(top+bandRows, clip.Bottom))
		wg.Add(1)
		go func() {
			defer wg.Done()
			fillPathSerial(p, band, blitter, aa)
		}()
	}
	wg.Wait()
}

func fillPathSerial(p *path.Path, clip geom.IRect, blitter Blitter, aa bool) {
	if aa {
		AntiFillPath(p, clip, blitter)
	} else {
		FillPath(p, clip, blitter)
	}
}

// maxDefaultBands caps how many bands a default (workers <= 0) split produces. It bounds goroutine/blitter-rebuild
// overhead on very tall fills while staying above any realistic core count, so large fills still saturate the machine.
const maxDefaultBands = 64

// bandCount is the shared band-count policy for the rect-fill band splitters: how many bands to split a range of rows
// integer-scanlines tall, each band at least minBandRows tall. It returns a value < 2 (do not parallelize) when the
// range spans fewer than two full bands or a single worker is requested.
//
// workers <= 0 selects the default policy: as many bands as minBandRows allows, capped at maxDefaultBands — a pure
// function of rows, deliberately NOT of GOMAXPROCS. Band geometry must not depend on the machine: the path-fill scan
// converters accumulate edge X positions from the top of each band, so AA seam pixels depend on where the cuts fall,
// and a core-count-derived split would render the same scene differently on different machines (caught by the
// bit-exact golden gates as a cross-machine diff in shaded path fills). A fixed split renders identically everywhere;
// the scheduler spreads however many bands there are across the cores actually present. Explicit positive workers
// values remain for tests that pin a specific split.
func bandCount(rows int32, workers int) int32 {
	if workers <= 0 {
		workers = maxDefaultBands
	}
	if rows < 2*minBandRows {
		return 1
	}
	bands := int32(workers)
	if maxBands := rows / minBandRows; bands > maxBands {
		bands = maxBands
	}
	return bands
}

// RectFillBandBounds splits a device-space rect spanning rows [top, bottom) into up to workers bands (each at least
// minBandRows integer scanlines tall) for concurrent filling, returning the n+1 band boundary y-coordinates [top, c_1,
// …, c_{n-1}, bottom]. The interior cuts c_i are integer scanlines while the fractional top and bottom are preserved.
// It returns nil when the rect is too short to be worth parallelizing (fewer than two bands). workers <= 0 uses the
// fixed machine-independent default split (see bandCount).
//
// Because a rect fill's per-row coverage is independent of the other rows and the interior cuts fall on integer
// scanlines, filling each [bounds[i], bounds[i+1]) sub-rect with the same AA/BW rect converter (and an independent
// per-worker blitter — shader blitters carry per-span scratch and cannot cross goroutines) is byte-identical to filling
// the whole rect serially. The minBandRows floor guarantees no band isolates the fractional top or bottom edge row on
// its own (see the antiFillDot8 note above).
func RectFillBandBounds(top, bottom float32, workers int) []float32 {
	iTop := int32(math.Ceil(float64(top)))
	iBot := int32(math.Floor(float64(bottom)))
	rows := iBot - iTop
	bands := bandCount(rows, workers)
	if bands < 2 {
		return nil
	}
	bounds := make([]float32, 0, bands+1)
	bounds = append(bounds, top)
	for i := int32(1); i < bands; i++ {
		bounds = append(bounds, float32(iTop+int32(int64(i)*int64(rows)/int64(bands))))
	}
	bounds = append(bounds, bottom)
	return bounds
}

// IRectFillBandBounds splits an integer device rect spanning rows [top, bottom) into up to workers bands (each at least
// minBandRows scanlines tall), returning the n+1 integer band boundaries [top, c_1, …, c_{n-1}, bottom], or nil when
// the rect is too short to be worth parallelizing. workers <= 0 uses the fixed machine-independent default split (see
// bandCount). This is the drawPaint (full-device
// shaded fill) counterpart of RectFillBandBounds: every boundary is an integer scanline, so — because an integer rect
// fill through FillIRectRasterClip is per-row independent for any clip (each row intersects the clip spans on its own,
// with no AA-rect edge coverage) — filling each [bounds[i], bounds[i+1]) band with an independent per-worker blitter is
// byte-identical to the serial fill regardless of the clip shape.
func IRectFillBandBounds(top, bottom int32, workers int) []int32 {
	rows := bottom - top
	bands := bandCount(rows, workers)
	if bands < 2 {
		return nil
	}
	bounds := make([]int32, 0, bands+1)
	bounds = append(bounds, top)
	for i := int32(1); i < bands; i++ {
		bounds = append(bounds, top+int32(int64(i)*int64(rows)/int64(bands)))
	}
	bounds = append(bounds, bottom)
	return bounds
}
