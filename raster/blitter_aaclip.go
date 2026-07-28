// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// AAClipBlitter wraps another blitter, multiplying every blit's coverage by the AA clip's per-pixel alpha. It
// implements A8 and LCD16 merge lanes for BlitMask (mergeOneMaskRow/mergeOneMaskRow16); a BW-upscale lane is not needed
// since this package has no BW masks.

package raster

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// AAClipBlitter multiplies coverage passed to another blitter by an AAClip's per-pixel alpha.
type AAClipBlitter struct {
	blitter Blitter
	clip    *AAClip
	// scanline scratch (lazily allocated to the clip width)
	runs []int16
	aa   []Alpha
	// row scratch for BlitMask's per-row merge
	rowScratch   []uint8
	rowScratch16 []uint16
	clipBounds   geom.IRect
}

// Init sets up the blitter to wrap blitter's blits with clip's coverage. clip must be non-empty. One instance may be
// re-Inited with a different blitter and clip; the scratch buffers are sized on demand and grow with the clip.
func (b *AAClipBlitter) Init(blitter Blitter, clip *AAClip) {
	b.blitter = blitter
	b.clip = clip
	b.clipBounds = clip.Bounds()
}

// ensureRunsAndAA lazily allocates the scanline scratch buffers to the clip width. It sizes on length rather than
// nil-ness so that re-Initing one instance with a wider clip grows the buffers instead of overrunning the ones the
// previous clip sized (the row scratch in BlitMask grows the same way).
func (b *AAClipBlitter) ensureRunsAndAA() {
	// add 1 so we can store the terminating run count of 0
	count := b.clipBounds.Width() + 1
	if int32(len(b.runs)) < count {
		b.runs = make([]int16, count)
		b.aa = make([]Alpha, count)
	}
}

// expandToRuns expands the clip row's RLE into a runs/alpha pair covering width pixels. We don't read our initial n
// from data, since the caller may have had to clip it, hence the initialCount parameter.
func expandToRuns(data []uint8, initialCount, width int32, runs []int16, aa []Alpha) {
	n := initialCount
	off := int32(0)
	for {
		if n > width {
			n = width
		}
		runs[off] = int16(n)
		aa[off] = data[1]

		data = data[2:]
		width -= n
		if width == 0 {
			break
		}
		off += n
		// load the next count
		n = int32(data[0])
	}
	runs[off+n] = 0 // sentinel
}

// BlitH implements Blitter.
func (b *AAClipBlitter) BlitH(x, y, width int32) {
	row, _ := b.clip.findRow(y)
	row, initialCount := b.clip.findX(row, x)

	if initialCount >= width {
		alpha := row[1]
		if alpha == 0 {
			return
		}
		if alpha == 0xFF {
			b.blitter.BlitH(x, y, width)
			return
		}
	}

	b.ensureRunsAndAA()
	expandToRuns(row, initialCount, width, b.runs, b.aa)

	b.blitter.BlitAntiH(x, y, b.aa, b.runs)
}

// mergeAARuns combines a clip RLE row with an incoming runs/alpha pair into the destination runs/alpha pair,
// multiplying alphas.
func mergeAARuns(row []uint8, rowN int32, srcAA []Alpha, srcRuns []int16, dstAA []Alpha, dstRuns []int16) {
	srcOff := int32(0)
	dstOff := int32(0)
	srcN := int32(srcRuns[0])
	// do we need this check?
	if srcN == 0 {
		return
	}

	for {
		newAlpha := colorcore.MulDiv255Round(srcAA[srcOff], row[1])
		minN := min(srcN, rowN)
		dstRuns[dstOff] = int16(minN)
		dstAA[dstOff] = newAlpha
		dstOff += minN

		srcN -= minN
		if srcN == 0 {
			n := int32(srcRuns[srcOff]) // refresh
			srcOff += n
			srcN = int32(srcRuns[srcOff]) // reload
			if srcN == 0 {
				break
			}
		}
		rowN -= minN
		if rowN == 0 {
			row = row[2:]
			rowN = int32(row[0]) // reload
		}
	}
	dstRuns[dstOff] = 0
}

// BlitAntiH implements Blitter.
func (b *AAClipBlitter) BlitAntiH(x, y int32, aa []Alpha, runs []int16) {
	row, _ := b.clip.findRow(y)
	row, initialCount := b.clip.findX(row, x)

	b.ensureRunsAndAA()

	mergeAARuns(row, initialCount, aa, runs, b.aa, b.runs)
	b.blitter.BlitAntiH(x, y, b.aa, b.runs)
}

// BlitV implements Blitter.
func (b *AAClipBlitter) BlitV(x, y, height int32, alpha Alpha) {
	if b.clip.quickContains(x, y, x+1, y+height) {
		b.blitter.BlitV(x, y, height, alpha)
		return
	}

	for {
		row, lastY := b.clip.findRow(y)
		dy := lastY - y + 1
		if dy > height {
			dy = height
		}
		height -= dy

		row, _ = b.clip.findX(row, x)
		newAlpha := colorcore.MulDiv255Round(alpha, row[1])
		if newAlpha != 0 {
			b.blitter.BlitV(x, y, dy, newAlpha)
		}
		if height <= 0 {
			break
		}
		y = lastY + 1
	}
}

// BlitRect implements Blitter.
func (b *AAClipBlitter) BlitRect(x, y, width, height int32) {
	if b.clip.quickContains(x, y, x+width, y+height) {
		b.blitter.BlitRect(x, y, width, height)
		return
	}

	for height > 0 {
		height--
		b.BlitH(x, y, width)
		y++
	}
}

// BlitAntiRect implements Blitter via the default lowering.
func (b *AAClipBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	BlitAntiRectViaVH(b, x, y, width, height, leftAlpha, rightAlpha)
}

// BlitMask implements Blitter: merge the mask with the clip row by row, blitting one-row masks through the wrapped
// blitter.
func (b *AAClipBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	if b.clip.QuickContains(clip) {
		b.blitter.BlitMask(mask, clip)
		return
	}

	// No ensureRunsAndAA() here: this method works exclusively out of the row scratch it sizes below and never touches
	// b.runs/b.aa. Upstream needs the call because there the row mask points into the single shared scanline scratch;
	// this port keeps the two separate.
	width := clip.Width()
	isLCD := mask.Format == MaskLCD16
	var src int
	rowMask := &Mask{
		RowBytes: mask.RowBytes, // doesn't matter, since our height==1
		Format:   mask.Format,
	}
	if isLCD {
		if int32(len(b.rowScratch16)) < width {
			b.rowScratch16 = make([]uint16, width)
		}
		src = mask.addr16(clip.Left, clip.Top)
		rowMask.Image16 = b.rowScratch16
	} else {
		if int32(len(b.rowScratch)) < width {
			b.rowScratch = make([]uint8, width)
		}
		src = mask.addr8(clip.Left, clip.Top)
		rowMask.Image = b.rowScratch
	}
	rowMask.Bounds.Left = clip.Left
	rowMask.Bounds.Right = clip.Right

	y := clip.Top
	stopY := y + clip.Height()

	for {
		row, localStopY := b.clip.findRow(y)
		// findRow returns last Y, not stop, so we add 1
		localStopY = min(localStopY+1, stopY)

		row, initialCount := b.clip.findX(row, clip.Left)
		for {
			if isLCD {
				mergeOneMaskRow16(mask.Image16[src:], width, row, initialCount, b.rowScratch16)
			} else {
				mergeOneMaskRow(mask.Image[src:], width, row, initialCount, b.rowScratch)
			}
			rowMask.Bounds.Top = y
			rowMask.Bounds.Bottom = y + 1
			b.blitter.BlitMask(rowMask, rowMask.Bounds)
			if isLCD {
				src += int(mask.RowBytes / 2)
			} else {
				src += int(mask.RowBytes)
			}
			y++
			if y >= localStopY {
				break
			}
		}
		if y >= stopY {
			break
		}
	}
}

// mergeOneMaskRow16 multiplies one LCD16 mask row by the clip row's alphas, each 5/6/5 channel scaled by
// mulDiv255Round.
func mergeOneMaskRow16(src []uint16, srcN int32, row []uint8, rowN int32, dst []uint16) {
	off := int32(0)
	for {
		n := min(rowN, srcN)
		rowA := row[1]
		switch rowA {
		case 0xFF:
			copy(dst[off:off+n], src[off:off+n])
		case 0:
			for i := off; i < off+n; i++ {
				dst[i] = 0
			}
		default:
			for i := off; i < off+n; i++ {
				v := src[i]
				r := colorcore.MulDiv255Round(uint8(v>>11&0x1F), rowA)
				g := colorcore.MulDiv255Round(uint8(v>>5&0x3F), rowA)
				bl := colorcore.MulDiv255Round(uint8(v&0x1F), rowA)
				dst[i] = uint16(r)<<11 | uint16(g)<<5 | uint16(bl)
			}
		}

		srcN -= n
		if srcN == 0 {
			break
		}
		off += n

		row = row[2:]
		rowN = int32(row[0])
	}
}

// mergeOneMaskRow multiplies one mask row by the clip row's alphas.
func mergeOneMaskRow(src []uint8, srcN int32, row []uint8, rowN int32, dst []uint8) {
	off := int32(0)
	for {
		n := min(rowN, srcN)
		rowA := row[1]
		switch rowA {
		case 0xFF:
			copy(dst[off:off+n], src[off:off+n])
		case 0:
			for i := off; i < off+n; i++ {
				dst[i] = 0
			}
		default:
			for i := off; i < off+n; i++ {
				dst[i] = colorcore.MulDiv255Round(src[i], rowA)
			}
		}

		srcN -= n
		if srcN == 0 {
			break
		}
		off += n

		row = row[2:]
		rowN = int32(row[0])
	}
}

// BlitAntiH2 implements Blitter via the default lowering.
func (b *AAClipBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(b, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter via the default lowering.
func (b *AAClipBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(b, x, y, a0, a1)
}
