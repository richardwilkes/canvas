// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The blitter model: blitters write pixels into memory; besides efficiency, they handle clipping and antialiasing.
// Coordinates passed to the Blit* calls are in destination pixel space.

package raster

import "github.com/richardwilkes/canvas/geom"

// Alpha is an 8-bit coverage/alpha value.
type Alpha = uint8

// Blitter is the pixel-writing interface used throughout the rasterizer. Implementations without a specialized version
// of an optional method should call the corresponding Blit*Via helper, which provides a default implementation built
// from the required methods.
type Blitter interface {
	// BlitH blits a horizontal run of one or more pixels.
	BlitH(x, y, width int32)

	// BlitAntiH blits a horizontal run of antialiased pixels. runs is a sparse zero-terminated run-length encoding:
	// runs[i] holds the number of pixels sharing the alpha at antialias[i], and the next valid entry is that many slots
	// away.
	BlitAntiH(x, y int32, antialias []Alpha, runs []int16)

	// BlitV blits a vertical run of pixels with a constant alpha value.
	BlitV(x, y, height int32, alpha Alpha)

	// BlitRect blits a solid rectangle one or more pixels wide.
	BlitRect(x, y, width, height int32)

	// BlitAntiRect blits a rectangle with one alpha-blended column on the left, width (zero or more) opaque pixels, and
	// one alpha-blended column on the right; the result is always at least two pixels wide.
	BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha)

	// BlitMask blits a pattern of pixels defined by a rectangle-clipped A8 mask.
	BlitMask(mask *Mask, clip geom.IRect)

	// BlitAntiH2 blits the two pixels (x, y) and (x+1, y) with the given alphas.
	BlitAntiH2(x, y int32, a0, a1 Alpha)

	// BlitAntiV2 blits the two pixels (x, y) and (x, y+1) with the given alphas.
	BlitAntiV2(x, y int32, a0, a1 Alpha)
}

// BlitAntiH2Via is the default implementation of BlitAntiH2, built from BlitAntiH.
func BlitAntiH2Via(b Blitter, x, y int32, a0, a1 Alpha) {
	runs := [3]int16{1, 1, 0}
	aa := [2]Alpha{a0, a1}
	b.BlitAntiH(x, y, aa[:], runs[:])
}

// BlitAntiV2Via is the default implementation of BlitAntiV2, built from BlitAntiH.
func BlitAntiV2Via(b Blitter, x, y int32, a0, a1 Alpha) {
	runs := [2]int16{1, 0}
	aa := [1]Alpha{a0}
	b.BlitAntiH(x, y, aa[:], runs[:])
	// reset in case the clipping blitter modified runs
	runs[0] = 1
	runs[1] = 0
	aa[0] = a1
	b.BlitAntiH(x, y+1, aa[:], runs[:])
}

// BlitVViaAntiH is the default implementation of BlitV, built from BlitRect and BlitAntiH.
func BlitVViaAntiH(b Blitter, x, y, height int32, alpha Alpha) {
	if alpha == 255 {
		b.BlitRect(x, y, 1, height)
		return
	}
	var runs [2]int16
	var aa [1]Alpha
	for height > 0 {
		height--
		runs[0] = 1
		runs[1] = 0
		aa[0] = alpha
		b.BlitAntiH(x, y, aa[:], runs[:])
		y++
	}
}

// BlitRectViaH is the default implementation of BlitRect, built from BlitH.
func BlitRectViaH(b Blitter, x, y, width, height int32) {
	for height > 0 {
		height--
		b.BlitH(x, y, width)
		y++
	}
}

// BlitAntiRectViaVH is the default implementation of BlitAntiRect, built from BlitV and BlitRect.
func BlitAntiRectViaVH(b Blitter, x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	if leftAlpha > 0 { // we may send in x = -1 with leftAlpha = 0
		b.BlitV(x, y, height, leftAlpha)
	}
	x++
	if width > 0 {
		b.BlitRect(x, y, width, height)
		x += width
	}
	if rightAlpha > 0 {
		b.BlitV(x, y, height, rightAlpha)
	}
}

///////////////////////////////////////////////////////////////////////////////

// NullBlitter silently never blits.
type NullBlitter struct{}

// BlitH implements Blitter.
func (NullBlitter) BlitH(_, _, _ int32) {}

// BlitAntiH implements Blitter.
func (NullBlitter) BlitAntiH(_, _ int32, _ []Alpha, _ []int16) {}

// BlitV implements Blitter.
func (NullBlitter) BlitV(_, _, _ int32, _ Alpha) {}

// BlitRect implements Blitter.
func (NullBlitter) BlitRect(_, _, _, _ int32) {}

// BlitAntiRect implements Blitter.
func (NullBlitter) BlitAntiRect(_, _, _, _ int32, _, _ Alpha) {}

// BlitMask implements Blitter.
func (NullBlitter) BlitMask(_ *Mask, _ geom.IRect) {}

// BlitAntiH2 implements Blitter.
func (NullBlitter) BlitAntiH2(_, _ int32, _, _ Alpha) {}

// BlitAntiV2 implements Blitter.
func (NullBlitter) BlitAntiV2(_, _ int32, _, _ Alpha) {}

///////////////////////////////////////////////////////////////////////////////

// blitterClipper picks the most-efficient wrapper blitter to apply a region clip. The returned blitter points into the
// clipper's own storage, so the clipper must outlive it.
type blitterClipper struct {
	null NullBlitter
	rgn  RgnClipBlitter
	rect RectClipBlitter
}

// apply wraps blitter to restrict it to clip; ir may be nil (no bounds hint).
func (c *blitterClipper) apply(blitter Blitter, clip *Region, ir *geom.IRect) Blitter {
	if clip != nil {
		clipR := clip.Bounds()
		switch {
		case clip.IsEmpty() || (ir != nil && !clipR.Intersects(*ir)):
			blitter = &c.null
		case clip.IsRect():
			if ir == nil || !clipR.ContainsRect(*ir) {
				c.rect.Init(blitter, clipR)
				blitter = &c.rect
			}
		default:
			c.rgn.Init(blitter, clip)
			blitter = &c.rgn
		}
	}
	return blitter
}

///////////////////////////////////////////////////////////////////////////////

// scalarToAlpha converts a [0, 1] coverage value to an Alpha, snapping alphas near the ends to 0/255.
func scalarToAlpha(a float32) Alpha {
	alpha := Alpha(a * 255)
	switch {
	case alpha > 247:
		return 0xFF
	case alpha < 8:
		return 0
	default:
		return alpha
	}
}

// blitFatAntiRect blits an AA rect of width at least 3 (small rects have too many edge cases).
func blitFatAntiRect(b Blitter, rect geom.Rect) {
	bounds := rect.RoundOut()

	// skbug.com/40039068: rects with zero height can arrive via horizontal tiling; do nothing.
	if bounds.Height() == 0 {
		return
	}

	width := bounds.Width()
	runs := make([]int16, width+1)
	alphas := make([]Alpha, width+1)

	runs[0] = 1
	runs[1] = int16(width - 2)
	runs[width-1] = 1
	runs[width] = 0

	partialL := float32(bounds.Left+1) - rect.Left
	partialR := rect.Right - float32(bounds.Right-1)
	partialT := float32(bounds.Top+1) - rect.Top
	partialB := rect.Bottom - float32(bounds.Bottom-1)

	if bounds.Height() == 1 {
		partialT = rect.Bottom - rect.Top
	}

	alphas[0] = scalarToAlpha(partialL * partialT)
	alphas[1] = scalarToAlpha(partialT)
	alphas[width-1] = scalarToAlpha(partialR * partialT)
	b.BlitAntiH(bounds.Left, bounds.Top, alphas, runs)

	if bounds.Height() > 2 {
		b.BlitAntiRect(bounds.Left, bounds.Top+1, width-2, bounds.Height()-2,
			scalarToAlpha(partialL), scalarToAlpha(partialR))
	}

	if bounds.Height() > 1 {
		// re-init the runs, in case the first BlitAntiH went through a clipping blitter that broke them
		runs[0] = 1
		runs[1] = int16(width - 2)
		runs[width-1] = 1
		runs[width] = 0
		alphas[0] = scalarToAlpha(partialL * partialB)
		alphas[1] = scalarToAlpha(partialB)
		alphas[width-1] = scalarToAlpha(partialR * partialB)
		b.BlitAntiH(bounds.Left, bounds.Bottom-1, alphas, runs)
	}
}

///////////////////////////////////////////////////////////////////////////////
// Alpha-run break helpers

// AlphaRunsBreak breaks the runs in the buffer at offsets x and x+count, properly updating the runs to the right and
// left, so a caller can sum another run into the new sub-runs.
func AlphaRunsBreak(runs []int16, alpha []Alpha, x, count int) {
	nextOff := x

	off := 0
	for x > 0 {
		n := int(runs[off])
		if x < n {
			alpha[off+x] = alpha[off]
			runs[off] = int16(x)
			runs[off+x] = int16(n - x)
			break
		}
		off += n
		x -= n
	}

	off = nextOff
	x = count
	for {
		n := int(runs[off])
		if x < n {
			alpha[off+x] = alpha[off]
			runs[off] = int16(x)
			runs[off+x] = int16(n - x)
			break
		}
		x -= n
		if x <= 0 {
			break
		}
		off += n
	}
}

// AlphaRunsBreakAt cuts (at offset x in the buffer) a run into two shorter runs with matching alpha values. Used by
// RectClipBlitter to trim an RLE encoding to the clip.
func AlphaRunsBreakAt(runs []int16, alpha []Alpha, x int) {
	off := 0
	for x > 0 {
		n := int(runs[off])
		if x < n {
			alpha[off+x] = alpha[off]
			runs[off] = int16(x)
			runs[off+x] = int16(n - x)
			break
		}
		off += n
		x -= n
	}
}

///////////////////////////////////////////////////////////////////////////////

// RectClipBlitter wraps another blitter, restricting all blits to a rect before forwarding.
type RectClipBlitter struct {
	blitter  Blitter
	clipRect geom.IRect
}

// Init sets up the blitter to restrict blitter's blits to clipRect.
func (r *RectClipBlitter) Init(blitter Blitter, clipRect geom.IRect) {
	r.blitter = blitter
	r.clipRect = clipRect
}

func yInRect(y int32, rect geom.IRect) bool {
	return y >= rect.Top && y < rect.Bottom
}

func xInRect(x int32, rect geom.IRect) bool {
	return x >= rect.Left && x < rect.Right
}

// computeAntiWidth returns the total pixel width of a zero-terminated run array.
func computeAntiWidth(runs []int16) int32 {
	width := int32(0)
	for runs[width] != 0 {
		width += int32(runs[width])
	}
	return width
}

// BlitH implements Blitter.
func (r *RectClipBlitter) BlitH(left, y, width int32) {
	if !yInRect(y, r.clipRect) {
		return
	}
	right := left + width
	if left < r.clipRect.Left {
		left = r.clipRect.Left
	}
	if right > r.clipRect.Right {
		right = r.clipRect.Right
	}
	width = right - left
	if width > 0 {
		r.blitter.BlitH(left, y, width)
	}
}

// BlitAntiH implements Blitter.
func (r *RectClipBlitter) BlitAntiH(left, y int32, aa []Alpha, runs []int16) {
	if !yInRect(y, r.clipRect) || left >= r.clipRect.Right {
		return
	}

	x0 := left
	x1 := left + computeAntiWidth(runs)

	if x1 <= r.clipRect.Left {
		return
	}

	off := 0
	if x0 < r.clipRect.Left {
		dx := r.clipRect.Left - x0
		AlphaRunsBreakAt(runs, aa, int(dx))
		off = int(dx)
		x0 = r.clipRect.Left
	}

	if x1 > r.clipRect.Right {
		x1 = r.clipRect.Right
		AlphaRunsBreakAt(runs[off:], aa[off:], int(x1-x0))
		runs[off+int(x1-x0)] = 0
	}

	r.blitter.BlitAntiH(x0, y, aa[off:], runs[off:])
}

// BlitV implements Blitter.
func (r *RectClipBlitter) BlitV(x, y, height int32, alpha Alpha) {
	if !xInRect(x, r.clipRect) {
		return
	}
	y0 := y
	y1 := y + height
	if y0 < r.clipRect.Top {
		y0 = r.clipRect.Top
	}
	if y1 > r.clipRect.Bottom {
		y1 = r.clipRect.Bottom
	}
	if y0 < y1 {
		r.blitter.BlitV(x, y0, y1-y0, alpha)
	}
}

// BlitRect implements Blitter.
func (r *RectClipBlitter) BlitRect(left, y, width, height int32) {
	rect := geom.IRectLTRB(left, y, left+width, y+height)
	if rect.Intersect(r.clipRect) {
		r.blitter.BlitRect(rect.Left, rect.Top, rect.Width(), rect.Height())
	}
}

// BlitMask implements Blitter.
func (r *RectClipBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	rect := clip
	if rect.Intersect(r.clipRect) {
		r.blitter.BlitMask(mask, rect)
	}
}

// BlitAntiH2 implements Blitter via the default lowering.
func (r *RectClipBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(r, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter via the default lowering.
func (r *RectClipBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(r, x, y, a0, a1)
}

// BlitAntiRect implements Blitter.
func (r *RectClipBlitter) BlitAntiRect(left, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	// The *true* width of the rectangle blitted is width+2.
	rect := geom.IRectLTRB(left, y, left+width+2, y+height)
	if !rect.Intersect(r.clipRect) {
		return
	}
	if rect.Left != left {
		leftAlpha = 255
	}
	if rect.Right != left+width+2 {
		rightAlpha = 255
	}
	switch {
	case leftAlpha == 255 && rightAlpha == 255:
		r.blitter.BlitRect(rect.Left, rect.Top, rect.Width(), rect.Height())
	case rect.Width() == 1:
		if rect.Left == left {
			r.blitter.BlitV(rect.Left, rect.Top, rect.Height(), leftAlpha)
		} else {
			r.blitter.BlitV(rect.Left, rect.Top, rect.Height(), rightAlpha)
		}
	default:
		r.blitter.BlitAntiRect(rect.Left, rect.Top, rect.Width()-2, rect.Height(), leftAlpha, rightAlpha)
	}
}
