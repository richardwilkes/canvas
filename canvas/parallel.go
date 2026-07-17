// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Row-band parallelism for large shaded rect fills. A gradient/image span evaluates per pixel in scalar Go, so a big
// shaded rect fill is the CPU throughput bottleneck: GradientRectFill and ImageScale both exceeded the ≤3× gate.
// Splitting such a fill into horizontal row bands filled concurrently closes the gap. A rect fill's per-row coverage is
// independent of the other rows, so banding at integer scanlines is byte-identical to the serial fill; each band builds
// its own blitter because shader blitters carry per-span scratch and cannot be shared across goroutines. This is scoped
// to the rect-fill lane (the only fill whose banding is byte-exact — path fills restart edge accumulation at each band
// top).

package canvas

import (
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// isBandableShaderBlitter reports whether a blitter is one of the per-span-scratch shader blitters whose per-pixel
// throughput dominates a large fill: a gradient/noise ShaderBlitter or the image LegacyShaderBlitter. The
// constant-collapse (BlendBlitter) and solid lanes are cheap and not worth the goroutine/blitter-rebuild overhead of
// banding.
func isBandableShaderBlitter(b raster.Blitter) bool {
	switch b.(type) {
	case *raster.ShaderBlitter, *raster.LegacyShaderBlitter:
		return true
	default:
		return false
	}
}

// fillRectShadedParallel fills devRect through per-band shader blitters concurrently. bounds are the band boundaries
// from raster.RectFillBandBounds (n+1 values; interior cuts integer-aligned, fractional top/bottom preserved). Each
// band builds its own blitter and fills its sub-rect; the union is byte-identical to the serial fill. The first band
// runs on the calling goroutine (one fewer worker, and it keeps the caller busy); the rest run in goroutines joined
// before returning, so every band's blitter is fully consumed by the time the draw completes.
//
// It takes the destination pixmap and raster clip directly rather than a *draw receiver: the goroutines necessarily
// capture (and so heap-escape) their inputs, and escape analysis is flow-insensitive, so a *draw receiver here would
// mark the whole draw struct as escaping in every caller — even the common non-shaded rect fill that never reaches this
// branch. Passing the already-heap pixmap/clip pointers keeps the draw value itself on the caller's stack.
func fillRectShadedParallel(dst *raster.Pixmap, rc *raster.Clip, devRect geom.Rect, paint *Paint, matrix *geom.Matrix, aa bool, bounds []float32) {
	var wg sync.WaitGroup
	for i := 1; i+1 < len(bounds); i++ {
		band := geom.RectLTRB(devRect.Left, bounds[i], devRect.Right, bounds[i+1])
		wg.Add(1)
		// A top-level worker (not a capturing closure): go f(args…) copies its value/pointer arguments onto the new
		// goroutine's stack without heap-allocating a closure per band.
		go fillRectShadedBandWorker(&wg, dst, rc, band, paint, matrix, aa)
	}
	fillRectShadedBand(dst, rc, geom.RectLTRB(devRect.Left, bounds[0], devRect.Right, bounds[1]), paint, matrix, aa)
	wg.Wait()
}

// fillRectShadedBandWorker is the goroutine entry point for a single band: fill it, then signal done.
func fillRectShadedBandWorker(wg *sync.WaitGroup, dst *raster.Pixmap, rc *raster.Clip, band geom.Rect, paint *Paint, matrix *geom.Matrix, aa bool) {
	defer wg.Done()
	fillRectShadedBand(dst, rc, band, paint, matrix, aa)
}

// fillRectShadedBand builds an independent blitter for a single band and fills it, then recycles the blitter. It is
// safe to run concurrently with sibling bands: chooseBlitter/recycleBlitter use goroutine-safe pools, the blitter
// samples read-only paint/matrix/source-image state, and it writes only its band's (disjoint from the other bands')
// device rows. This is exactly the work the serial rect-fill lane does, restricted to the band's rows.
func fillRectShadedBand(dst *raster.Pixmap, rc *raster.Clip, band geom.Rect, paint *Paint, matrix *geom.Matrix, aa bool) {
	blitter := chooseBlitter(dst, paint, matrix)
	if aa {
		raster.AntiFillRect(band, rc, blitter)
	} else {
		raster.FillRectRasterClip(band, rc, blitter)
	}
	recycleBlitter(blitter)
}

// fillIRectShadedParallel fills the integer device rect devRect through per-band shader blitters concurrently — the
// drawPaint (full-device shaded fill) counterpart of fillRectShadedParallel. bounds are the integer band boundaries
// from raster.IRectFillBandBounds. Unlike the rect-fill lane this fills an integer rect via FillIRectRasterClip (no
// AA-rect edge coverage), which is per-row independent for any clip, so the union of the bands is byte-identical to the
// serial fill regardless of the clip shape. It takes dst/rc explicitly rather than a *draw receiver for the same
// escape-analysis reason fillRectShadedParallel does (keep the draw value on the caller's stack); the goroutine workers
// are top-level functions (go f(args…), no capturing closure).
func fillIRectShadedParallel(dst *raster.Pixmap, rc *raster.Clip, devRect geom.IRect, paint *Paint, matrix *geom.Matrix, bounds []int32) {
	var wg sync.WaitGroup
	for i := 1; i+1 < len(bounds); i++ {
		band := geom.IRectLTRB(devRect.Left, bounds[i], devRect.Right, bounds[i+1])
		wg.Add(1)
		go fillIRectShadedBandWorker(&wg, dst, rc, band, paint, matrix)
	}
	fillIRectShadedBand(dst, rc, geom.IRectLTRB(devRect.Left, bounds[0], devRect.Right, bounds[1]), paint, matrix)
	wg.Wait()
}

// fillIRectShadedBandWorker is the goroutine entry point for a single integer-rect band.
func fillIRectShadedBandWorker(wg *sync.WaitGroup, dst *raster.Pixmap, rc *raster.Clip, band geom.IRect, paint *Paint, matrix *geom.Matrix) {
	defer wg.Done()
	fillIRectShadedBand(dst, rc, band, paint, matrix)
}

// fillIRectShadedBand builds an independent blitter for a single integer-rect band and fills it, then recycles the
// blitter. Concurrency-safe with sibling bands for the same reasons as fillRectShadedBand: goroutine-safe blitter
// pools, read-only paint/matrix/source sampling, and disjoint device rows.
func fillIRectShadedBand(dst *raster.Pixmap, rc *raster.Clip, band geom.IRect, paint *Paint, matrix *geom.Matrix) {
	blitter := chooseBlitter(dst, paint, matrix)
	raster.FillIRectRasterClip(band, rc, blitter)
	recycleBlitter(blitter)
}

// fillPathShadedParallel fills devPath through per-band shader blitters concurrently — the drawDevPath counterpart of
// fillRectShadedParallel for large shaded path fills. bounds are the integer band boundaries from
// raster.IRectFillBandBounds over the clip∩path-bounds rows; each band is the full clip width restricted to its rows.
// Unlike the rect lanes this banding is *not* byte-exact against the serial fill: the scan converters accumulate edge X
// positions incrementally from the top of the (band-clipped) edge list, so a boundary pixel's rounding or an AA seam
// step can differ (see raster.FillPathParallel's contract). Output is fully deterministic for a given clip/band split.
// The callers gate this lane to a simple-rect raster clip and a non-inverse fill, so each band is exactly
// AntiFillPath/FillPath over the band rect — the region form the serial rect-clip lane lowers to. devPath is shared
// read-only across the bands (the scan converters copy it into per-band edge lists) and outlives them (the wait below);
// dst/rc-free arguments keep the caller's draw value on its stack for the same escape-analysis reason the rect lanes
// do.
func fillPathShadedParallel(dst *raster.Pixmap, devPath *path.Path, paint *Paint, matrix *geom.Matrix, aa bool, clip geom.IRect, bounds []int32) {
	// Resolve the lazily-cached state the concurrent bands would otherwise race to compute: the path's bounds and
	// convexity (both memoized on the shared Path; the scan converters query them per band) and the CTM's type mask
	// (memoized on the shared Matrix; chooseBlitter's shader compile reads it). After this, the bands touch that state
	// read-only. The paint's shader tree carries the same kind of lazily-memoized matrix masks internally (ptsToUnit,
	// local matrices); those are warmed by the caller's own serial chooseBlitter — drawDevPath always compiles one
	// blitter for this paint/matrix before taking this lane, exactly as the rect band lanes' callers do — so this
	// function requires that a chooseBlitter for the same paint/matrix happened-before on the calling goroutine.
	devPath.Bounds()
	devPath.GetConvexity()
	matrix.Type()
	var wg sync.WaitGroup
	for i := 1; i+1 < len(bounds); i++ {
		band := geom.IRectLTRB(clip.Left, bounds[i], clip.Right, bounds[i+1])
		wg.Add(1)
		go fillPathShadedBandWorker(&wg, dst, devPath, paint, matrix, aa, band)
	}
	fillPathShadedBand(dst, devPath, paint, matrix, aa, geom.IRectLTRB(clip.Left, bounds[0], clip.Right, bounds[1]))
	wg.Wait()
}

// fillPathShadedBandWorker is the goroutine entry point for a single path band.
func fillPathShadedBandWorker(wg *sync.WaitGroup, dst *raster.Pixmap, devPath *path.Path, paint *Paint, matrix *geom.Matrix, aa bool, band geom.IRect) {
	defer wg.Done()
	fillPathShadedBand(dst, devPath, paint, matrix, aa, band)
}

// fillPathShadedBand builds an independent blitter for a single band and fills the path restricted to it, then recycles
// the blitter. Concurrency-safe with sibling bands: goroutine-safe blitter/edge pools, read-only
// path/paint/matrix/source sampling, and disjoint device rows.
func fillPathShadedBand(dst *raster.Pixmap, devPath *path.Path, paint *Paint, matrix *geom.Matrix, aa bool, band geom.IRect) {
	blitter := chooseBlitter(dst, paint, matrix)
	if aa {
		raster.AntiFillPath(devPath, band, blitter)
	} else {
		raster.FillPath(devPath, band, blitter)
	}
	recycleBlitter(blitter)
}
