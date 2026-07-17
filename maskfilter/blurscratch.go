// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package maskfilter

import (
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// blurScratch holds the reusable per-draw math buffers for the blur nine-patch fast path. These buffers are pooled
// because the analytic rect blur (blurRect) and the nine-patch stretch (drawNineClipped) otherwise allocate the output
// mask image plus the profile/scanline/run scratch fresh on every draw — the dominant heap traffic for a blurred fill
// (progress §14.1). One scratch is borrowed at the FilterRects/FilterRRect entry, threaded through the filter →
// drawNine call, and recycled after the patch is fully blitted: it is owned by exactly one goroutine for that span
// (nothing here re-enters the filters), so a process-wide sync.Pool is safe under concurrent draws on separate
// canvases.
//
// The image buffer has a cross-function lifetime: blurRect writes it, aliases it into the returned nine-patch mask, and
// drawNine reads it — all before the recycle. image is read (as the mask) while runs/alpha are written (the stretch
// RLE), but they are distinct fields, so there is no aliasing.
type blurScratch struct {
	image      []uint8        // blurRect's analytic output mask image (dp); aliased into the nine-patch mask
	profile    []uint8        // blurRect's pre-inverted half-plane blur profile
	hScan      []uint8        // blurRect's horizontal blurred scanline
	vScan      []uint8        // blurRect's vertical blurred scanline
	runs       []int16        // drawNineClipped's stretched-edge RLE run lengths
	alpha      []raster.Alpha // drawNineClipped's stretched-edge RLE coverages
	boundsMask raster.Mask    // blurRect's justComputeBounds output (dstM)
	imageMask  raster.Mask    // blurRect's computeBoundsAndRenderImage output (filterM)
	cornerMask raster.Mask    // drawNineClipped's four-corner subset
	leftMask   raster.Mask    // drawNineClipped's stretched left edge
	rightMask  raster.Mask    // drawNineClipped's stretched right edge
	// Reused per-draw scalars whose addresses cross a non-inlinable / interface-method boundary and so escape to the
	// heap. They live on the pooled (already heap-resident) scratch purely as a home the existing FilterPath →
	// filterRects → drawNine chain already threads, so taking their address costs no per-draw allocation. These are
	// headers/probes (mask value structs; the nested-rect probe array) — distinct from the byte buffers above — and,
	// like the buffers, every field that is read is written first each draw (blurRect's `*dst =`, drawNineClipped's `*m
	// =`, countNestedRects filling rects), so a recycled scratch's stale values are never observed.
	rects [2]geom.Rect // FilterPath's nested-rect probe output
}

var blurScratchPool = sync.Pool{New: func() any { return &blurScratch{} }}

// borrowBlurScratch returns a scratch with (possibly stale, possibly short) buffers; callers grow and overwrite what
// they read via growScratch before reading it.
func borrowBlurScratch() *blurScratch { return blurScratchPool.Get().(*blurScratch) }

// recycleBlurScratch returns the scratch (retaining its grown buffers) to the pool. Callers must have finished every
// read of image/runs/alpha first — i.e. after drawNine, not before.
func recycleBlurScratch(s *blurScratch) { blurScratchPool.Put(s) }

// growScratch returns buf resliced to length n, reusing the backing array whenever it already has the capacity
// (mirroring the make([]T, n) it replaces). It does not zero: every caller writes each element it later reads. blurRect
// fills its image/profile/scanline buffers completely; drawNineClipped writes only runs[0]/runs[width]/alpha[0] and the
// RLE blitter reads only those (blit_antih walks runs[0] → runs[width]=0, so the untouched interior is never observed).
func growScratch[T any](buf []T, n int) []T {
	if cap(buf) < n {
		return make([]T, n)
	}
	return buf[:n]
}
