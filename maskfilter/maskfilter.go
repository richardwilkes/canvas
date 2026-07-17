// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package maskfilter implements the mask filters that are publicly reachable: the blur mask filter (over the
// box-blur/mask-blur-filter engines), the table/gamma/clip filters, and the shader mask filter, plus the shared draw
// machinery: FilterPath/FilterRects/FilterRRect with the blur nine-patch fast paths and the drawNine blitting that
// stretches a small blurred patch over the destination.

package maskfilter

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// MaskFilter is the publicly reachable mask filter surface.
type MaskFilter interface {
	// FilterMask produces the filtered mask for src. A src with a nil Image computes bounds only (dst.Image stays nil).
	// ok is false when the filter cannot apply (the caller draws the geometry unfiltered).
	FilterMask(src *raster.Mask, ctm *geom.Matrix) (dst *raster.Mask, margin geom.IPoint, ok bool)

	// ComputeFastBounds returns a fast (possibly loose) upper bound on the filtered output's bounds, for the paint's
	// fast-bounds computation.
	ComputeFastBounds(src geom.Rect) geom.Rect
}

// FilterReturn is the three-way result of a nine-patch fast-path attempt: succeeded, failed definitively, or not
// implemented by this filter (fall back to the slow path).
type FilterReturn uint8

// FilterReturn values.
const (
	FilterFalse FilterReturn = iota
	FilterTrue
	FilterUnimplemented
)

// ninePatch is a small filtered mask whose center row/column stretch to cover outerRect.
type ninePatch struct {
	mask      raster.Mask // bounds offset to (0, 0)
	outerRect geom.IRect
	center    geom.IPoint
}

// rectsToNiner is implemented by filters providing filterRectsToNine (only blur does).
type rectsToNiner interface {
	filterRectsToNine(rects []geom.Rect, ctm *geom.Matrix, scratch *blurScratch) (ninePatch, FilterReturn)
}

// rrectToNiner is implemented by filters providing filterRRectToNine (only blur does).
type rrectToNiner interface {
	filterRRectToNine(rr geom.RRect, ctm *geom.Matrix) (ninePatch, bool)
}

// computeImageSize returns rowBytes * height, zero on overflow (the too-big-to-allocate signal).
func computeImageSize(bounds geom.IRect, rowBytes int32) int {
	h := int64(bounds.Height())
	size := int64(rowBytes) * h
	if h < 0 || size <= 0 || size > 1<<31-1 {
		return 0
	}
	return int(size)
}

// extractMaskSubset returns a sub-mask sharing storage, repositioned to (newX, newY).
func extractMaskSubset(src *raster.Mask, bounds geom.IRect, newX, newY int32) raster.Mask {
	dx := bounds.Left - src.Bounds.Left
	dy := bounds.Top - src.Bounds.Top
	moved := geom.IRectXYWH(newX, newY, bounds.Width(), bounds.Height())
	return raster.Mask{
		Image:    src.Image[int(dy)*int(src.RowBytes)+int(dx):],
		Bounds:   moved,
		RowBytes: src.RowBytes,
	}
}

// blitClippedMask blits mask through blitter, clipped to the intersection of bounds and clipR.
func blitClippedMask(blitter raster.Blitter, mask *raster.Mask, bounds, clipR geom.IRect) {
	r := bounds
	if r.Intersect(clipR) {
		blitter.BlitMask(mask, r)
	}
}

// blitClippedRect blits a solid rect through blitter, clipped to the intersection of rect and clipR.
func blitClippedRect(blitter raster.Blitter, rect, clipR geom.IRect) {
	r := rect
	if r.Intersect(clipR) {
		blitter.BlitRect(r.Left, r.Top, r.Width(), r.Height())
	}
}

// drawNineClipped blits the four corners, the center rect, and the stretched edges of the nine-patch, clipped to clipR.
func drawNineClipped(mask *raster.Mask, outerR geom.IRect, center geom.IPoint, fillCenter bool, clipR geom.IRect, blitter raster.Blitter, scratch *blurScratch) {
	cx := center.X
	cy := center.Y

	// m is reused across the four corner subsets: its address escapes into BlitMask (an interface call), so a plain
	// local would heap-allocate. It aliases the reused scratch.cornerMask header (heap-resident via the pool), and each
	// corner fully overwrites it before the synchronous blit — no per-draw alloc, observationally identical.
	m := &scratch.cornerMask

	// top-left
	bounds := mask.Bounds
	bounds.Right = cx
	bounds.Bottom = cy
	if bounds.Width() > 0 && bounds.Height() > 0 {
		*m = extractMaskSubset(mask, bounds, outerR.Left, outerR.Top)
		blitClippedMask(blitter, m, m.Bounds, clipR)
	}

	// top-right
	bounds = mask.Bounds
	bounds.Left = cx + 1
	bounds.Bottom = cy
	if bounds.Width() > 0 && bounds.Height() > 0 {
		*m = extractMaskSubset(mask, bounds, outerR.Right-bounds.Width(), outerR.Top)
		blitClippedMask(blitter, m, m.Bounds, clipR)
	}

	// bottom-left
	bounds = mask.Bounds
	bounds.Right = cx
	bounds.Top = cy + 1
	if bounds.Width() > 0 && bounds.Height() > 0 {
		*m = extractMaskSubset(mask, bounds, outerR.Left, outerR.Bottom-bounds.Height())
		blitClippedMask(blitter, m, m.Bounds, clipR)
	}

	// bottom-right
	bounds = mask.Bounds
	bounds.Left = cx + 1
	bounds.Top = cy + 1
	if bounds.Width() > 0 && bounds.Height() > 0 {
		*m = extractMaskSubset(mask, bounds, outerR.Right-bounds.Width(), outerR.Bottom-bounds.Height())
		blitClippedMask(blitter, m, m.Bounds, clipR)
	}

	var innerR geom.IRect
	innerR.Left = outerR.Left + cx - mask.Bounds.Left
	innerR.Top = outerR.Top + cy - mask.Bounds.Top
	innerR.Right = outerR.Right + (cx + 1 - mask.Bounds.Right)
	innerR.Bottom = outerR.Bottom + (cy + 1 - mask.Bounds.Bottom)
	if fillCenter {
		blitClippedRect(blitter, innerR, clipR)
	}

	innerW := innerR.Width()
	// The stretched-edge RLE scratch comes from the pooled scratch: only runs[0]/runs[width] and alpha[0] are ever
	// written and read (BlitAntiH walks runs[0] up to the terminating runs[width]=0), so a reused buffer's untouched
	// interior is never observed.
	scratch.runs = growScratch(scratch.runs, int(innerW)+1)
	runs := scratch.runs
	scratch.alpha = growScratch(scratch.alpha, int(innerW)+1)
	alpha := scratch.alpha

	// top
	r := geom.IRect{Left: innerR.Left, Top: outerR.Top, Right: innerR.Right, Bottom: innerR.Top}
	if r.Intersect(clipR) {
		startY := max32(0, r.Top-outerR.Top)
		stopY := startY + r.Height()
		width := r.Width()
		runs[0] = int16(width)
		runs[width] = 0
		for y := startY; y < stopY; y++ {
			alpha[0] = mask.Image[maskAddr8(mask, cx, mask.Bounds.Top+y)]
			blitter.BlitAntiH(r.Left, outerR.Top+y, alpha, runs)
		}
	}
	// bottom
	r = geom.IRect{Left: innerR.Left, Top: innerR.Bottom, Right: innerR.Right, Bottom: outerR.Bottom}
	if r.Intersect(clipR) {
		startY := outerR.Bottom - r.Bottom
		stopY := startY + r.Height()
		width := r.Width()
		runs[0] = int16(width)
		runs[width] = 0
		for y := startY; y < stopY; y++ {
			alpha[0] = mask.Image[maskAddr8(mask, cx, mask.Bounds.Bottom-y-1)]
			blitter.BlitAntiH(r.Left, outerR.Bottom-y-1, alpha, runs)
		}
	}
	// left. leftMask/rightMask alias the reused scratch.leftMask/scratch.rightMask headers (heap-resident via the
	// pool); their address escapes into BlitMask, so a plain local would heap-allocate.
	r = geom.IRect{Left: outerR.Left, Top: innerR.Top, Right: innerR.Left, Bottom: innerR.Bottom}
	if r.Intersect(clipR) {
		leftMask := &scratch.leftMask
		*leftMask = raster.Mask{
			Image:    mask.Image[maskAddr8(mask, mask.Bounds.Left+r.Left-outerR.Left, mask.Bounds.Top+cy):],
			Bounds:   r,
			RowBytes: 0, // so we repeat the scanline for our height
		}
		blitter.BlitMask(leftMask, r)
	}
	// right
	r = geom.IRect{Left: innerR.Right, Top: innerR.Top, Right: outerR.Right, Bottom: innerR.Bottom}
	if r.Intersect(clipR) {
		rightMask := &scratch.rightMask
		*rightMask = raster.Mask{
			Image:    mask.Image[maskAddr8(mask, mask.Bounds.Right-outerR.Right+r.Left, mask.Bounds.Top+cy):],
			Bounds:   r,
			RowBytes: 0,
		}
		blitter.BlitMask(rightMask, r)
	}
}

// maskAddr8 returns the byte offset of pixel (x, y) within m's image.
func maskAddr8(m *raster.Mask, x, y int32) int {
	return int(y-m.Bounds.Top)*int(m.RowBytes) + int(x-m.Bounds.Left)
}

// drawNine resolves the clip and blitter, then draws the patch per clip rect. scratch carries the pooled per-nine-patch
// RLE buffers used by drawNineClipped.
func drawNine(patch *ninePatch, fillCenter bool, clip *raster.Clip, blitter raster.Blitter, scratch *blurScratch) {
	// AAClipBlitterWrapper is a no-op for a BW clip (Rgn is the clip's own region and Blitter is the blitter
	// unchanged), so wrap only for an AA clip. The wrapper returns a self-referential pointer (&w.aaBlitter) through
	// the blitter interface, forcing it to the heap when used — skipping it on the common BW path avoids that per-draw
	// allocation.
	rgn := clip.BWRgn()
	if !clip.IsBW() {
		var wrapper raster.AAClipBlitterWrapper
		wrapper.Init(clip, blitter)
		blitter = wrapper.Blitter()
		rgn = wrapper.Rgn()
	}

	var clipper raster.RegionCliperator
	clipper.Init(rgn, patch.outerRect)
	for !clipper.Done() {
		drawNineClipped(&patch.mask, patch.outerRect, patch.center, fillCenter, clipper.Rect(), blitter, scratch)
		clipper.Next()
	}
}

// countNestedRects reports whether devPath is a single rect (1) or a nested pair of rects (2, filled into rects), else
// 0.
func countNestedRects(devPath *path.Path, rects *[2]geom.Rect) int {
	if devPath.IsNestedFillRects(rects) {
		return 2
	}
	if r, ok := devPath.IsRect(); ok {
		rects[0] = r
		return 1
	}
	return 0
}

// FilterRRect attempts the nine-patch fast path for a blurred round rect; false lets the caller draw another way.
func FilterRRect(mf MaskFilter, devRRect geom.RRect, ctm *geom.Matrix, clip *raster.Clip, blitter raster.Blitter) bool {
	niner, ok := mf.(rrectToNiner)
	if !ok {
		return false
	}
	patch, ok := niner.filterRRectToNine(devRRect, ctm)
	if !ok {
		return false
	}
	scratch := borrowBlurScratch()
	defer recycleBlurScratch(scratch)
	drawNine(&patch, true, clip, blitter, scratch)
	return true
}

// FilterRects attempts the nine-patch fast path for one or more blurred rects.
func FilterRects(mf MaskFilter, devRects []geom.Rect, ctm *geom.Matrix, clip *raster.Clip, blitter raster.Blitter) FilterReturn {
	niner, ok := mf.(rectsToNiner)
	if !ok {
		return FilterUnimplemented
	}
	scratch := borrowBlurScratch()
	defer recycleBlurScratch(scratch)
	return filterRectsScratch(niner, devRects, ctm, clip, blitter, scratch)
}

// filterRectsScratch is the shared body of FilterRects and FilterPath's nested-rect lane; the caller owns the scratch
// borrow. The scratch spans filterRectsToNine (which fills scratch.image with the analytic blur for the single-rect
// fast path) through drawNine (which reads that image and writes the RLE scratch), so it must stay borrowed until the
// patch is fully blitted.
func filterRectsScratch(niner rectsToNiner, devRects []geom.Rect, ctm *geom.Matrix, clip *raster.Clip, blitter raster.Blitter, scratch *blurScratch) FilterReturn {
	patch, ret := niner.filterRectsToNine(devRects, ctm, scratch)
	if ret == FilterTrue {
		drawNine(&patch, len(devRects) == 1, clip, blitter, scratch)
	}
	return ret
}

// FilterPath tries the nested-rect nine-patch lane, else renders the path to an A8 mask, filters it, and blits the
// result. Returns false when the caller should draw the path unfiltered. doFill distinguishes a filled path from a
// hairline stroke.
func FilterPath(mf MaskFilter, devPath *path.Path, ctm *geom.Matrix, clip *raster.Clip, blitter raster.Blitter, doFill bool) bool {
	// Borrow the scratch up front so the nested-rect probe array (scratch.rects) is heap-resident: it flows through
	// filterRectsScratch into filterRectsToNine (an interface method) and would otherwise escape and allocate on every
	// filtered path draw. The scratch is unused by the A8-mask fallback below, but holding it there costs only a pool
	// get/put.
	scratch := borrowBlurScratch()
	defer recycleBlurScratch(scratch)

	rectCount := 0
	if doFill {
		rectCount = countNestedRects(devPath, &scratch.rects)
	}
	if rectCount > 0 {
		if niner, ok := mf.(rectsToNiner); ok {
			switch filterRectsScratch(niner, scratch.rects[:rectCount], ctm, clip, blitter, scratch) {
			case FilterFalse:
				return false
			case FilterTrue:
				return true
			case FilterUnimplemented:
			}
		}
	}

	srcM, ok := DrawToMask(devPath, clip.Bounds(), mf, ctm, doFill)
	if !ok {
		return false
	}

	dstM, _, ok := mf.FilterMask(srcM, ctm)
	if !ok {
		return false
	}

	// if we get here, we need to (possibly) resolve the clip and blitter
	var wrapper raster.AAClipBlitterWrapper
	wrapper.Init(clip, blitter)
	blitter = wrapper.Blitter()

	clipper := raster.NewRegionCliperator(wrapper.Rgn(), dstM.Bounds)
	for !clipper.Done() {
		blitter.BlitMask(dstM, clipper.Rect())
		clipper.Next()
	}
	return true
}

func max32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}
