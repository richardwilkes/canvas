// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A8CoverageBlitter writes coverage directly into an A8 mask with overwrite (not blend) semantics. This is how paths
// get rasterized into masks: the input to mask filters (blur), glyph mask generation, and software path-renderer masks.
// (Drawing color into A8 *surfaces* belongs to the color-type matrix and arrives with its consumers.)

package raster

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// A8CoverageBlitter writes path coverage into an A8 Mask. Blit coordinates are device coordinates; the mask's Bounds
// locate the storage in device space.
type A8CoverageBlitter struct {
	mask *Mask
}

// NewA8CoverageBlitter returns a coverage blitter writing into mask.
func NewA8CoverageBlitter(mask *Mask) *A8CoverageBlitter {
	return &A8CoverageBlitter{mask: mask}
}

// BlitH implements Blitter.
func (a *A8CoverageBlitter) BlitH(x, y, width int32) {
	i := a.mask.addr8(x, y)
	row := a.mask.Image[i : i+int(width)]
	for j := range row {
		row[j] = 0xFF
	}
}

// BlitAntiH implements Blitter.
func (a *A8CoverageBlitter) BlitAntiH(x, y int32, antialias []Alpha, runs []int16) {
	base := a.mask.addr8(x, y)
	off := 0
	for {
		count := int(runs[off])
		if count == 0 {
			return
		}
		if aa := antialias[off]; aa != 0 {
			span := a.mask.Image[base+off : base+off+count]
			for j := range span {
				span[j] = aa
			}
		}
		off += count
	}
}

// BlitV implements Blitter.
func (a *A8CoverageBlitter) BlitV(x, y, height int32, alpha Alpha) {
	if alpha == 0 {
		return
	}
	i := a.mask.addr8(x, y)
	for h := int32(0); h < height; h++ {
		a.mask.Image[i] = alpha
		i += int(a.mask.RowBytes)
	}
}

// BlitRect implements Blitter.
func (a *A8CoverageBlitter) BlitRect(x, y, width, height int32) {
	i := a.mask.addr8(x, y)
	for h := int32(0); h < height; h++ {
		row := a.mask.Image[i : i+int(width)]
		for j := range row {
			row[j] = 0xFF
		}
		i += int(a.mask.RowBytes)
	}
}

// BlitAntiRect implements Blitter via the default lowering.
func (a *A8CoverageBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	BlitAntiRectViaVH(a, x, y, width, height, leftAlpha, rightAlpha)
}

// BlitMask implements Blitter: a straight copy for A8.
func (a *A8CoverageBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	src := mask.addr8(clip.Left, clip.Top)
	dst := a.mask.addr8(clip.Left, clip.Top)
	for h := clip.Height(); h > 0; h-- {
		copy(a.mask.Image[dst:dst+int(clip.Width())], mask.Image[src:src+int(clip.Width())])
		src += int(mask.RowBytes)
		dst += int(a.mask.RowBytes)
	}
}

// BlitAntiH2 implements Blitter via the default lowering.
func (a *A8CoverageBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(a, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter via the default lowering.
func (a *A8CoverageBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(a, x, y, a0, a1)
}

// FillPathToMask rasterizes p's coverage into a fresh A8 mask covering bounds (device space), with or without
// antialiasing (the wrapper with matrix handling and mask-filter integration arrives with its consumers).
func FillPathToMask(p *path.Path, bounds geom.IRect, aa bool) *Mask {
	mask := &Mask{
		Image:    make([]uint8, int(bounds.Width())*int(bounds.Height())),
		Bounds:   bounds,
		RowBytes: bounds.Width(),
	}
	blitter := NewA8CoverageBlitter(mask)
	if aa {
		AntiFillPath(p, bounds, blitter)
	} else {
		FillPath(p, bounds, blitter)
	}
	return mask
}
