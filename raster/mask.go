// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Mask: a coverage mask over a device-space rectangle, supporting the A8 and LCD16 formats. A8 masks are the format the
// AA scan converter's mask blitter accumulates into, and the format glyph masks and the software path-renderer produce;
// LCD16 masks (565 words holding three-channel LCD coverage) arrive from the subpixel text scaler lane. BW/ARGB32
// formats have no blitter-visible consumers here (color glyphs ride the sprite lane before any blitter sees the mask).

package raster

import "github.com/richardwilkes/canvas/geom"

// MaskFormat identifies a Mask's pixel format. The zero value is A8, keeping every existing Mask construction site an
// A8 mask.
type MaskFormat uint8

// MaskFormat values.
const (
	MaskA8    MaskFormat = iota // 8-bit linear coverage in Image
	MaskLCD16                   // 565 words holding 3-channel LCD coverage in Image16
)

// Mask is a coverage mask, supporting the A8 and LCD16 formats.
type Mask struct {
	// Image holds RowBytes bytes per row, covering Bounds. Image[0] is the alpha at (Bounds.Left, Bounds.Top). Used by
	// A8 masks; nil for LCD16.
	Image []uint8
	// Image16 holds RowBytes/2 565 words per row for LCD16 masks; nil for A8.
	Image16  []uint16
	Bounds   geom.IRect
	RowBytes int32
	Format   MaskFormat
}

// addr8 returns the index of (x, y) in Image.
func (m *Mask) addr8(x, y int32) int {
	return int(y-m.Bounds.Top)*int(m.RowBytes) + int(x-m.Bounds.Left)
}

// addr16 returns the index of (x, y) in Image16. RowBytes is a byte stride, so the word row stride is RowBytes/2.
func (m *Mask) addr16(x, y int32) int {
	return int(y-m.Bounds.Top)*int(m.RowBytes/2) + int(x-m.Bounds.Left)
}

// BlitMaskViaAntiH is a default implementation of BlitMask's A8 branch: lower the mask onto BlitAntiH one row at a time
// with all-1 runs.
func BlitMaskViaAntiH(b Blitter, mask *Mask, clip geom.IRect) {
	width := clip.Width()
	runs := make([]int16, width+1)
	for i := int32(0); i < width; i++ {
		runs[i] = 1
	}
	runs[width] = 0

	aa := mask.addr8(clip.Left, clip.Top)
	for y := clip.Top; y < clip.Bottom; y++ {
		b.BlitAntiH(clip.Left, y, mask.Image[aa:], runs)
		// reset in case a clipping blitter modified the runs
		for i := int32(0); i < width; i++ {
			runs[i] = 1
		}
		runs[width] = 0
		aa += int(mask.RowBytes)
	}
}
