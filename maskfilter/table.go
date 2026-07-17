// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The table mask filter: a 256-entry LUT applied to the mask's coverage, with the gamma and clip table builders.

package maskfilter

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// tableMaskFilter is the table mask filter implementation.
type tableMaskFilter struct {
	table [256]uint8
}

// NewTable creates a mask filter that maps each coverage byte through table.
func NewTable(table *[256]uint8) MaskFilter {
	return &tableMaskFilter{table: *table}
}

// NewGamma creates a mask filter that applies a gamma curve to coverage.
func NewGamma(gamma float32) MaskFilter {
	var table [256]uint8
	MakeGammaTable(&table, gamma)
	return &tableMaskFilter{table: table}
}

// NewClip creates a mask filter that remaps coverage to the [minV, maxV] range.
func NewClip(minV, maxV uint8) MaskFilter {
	var table [256]uint8
	MakeClipTable(&table, minV, maxV)
	return &tableMaskFilter{table: table}
}

// Table returns the LUT — the descriptor the GPU side consumes.
func (f *tableMaskFilter) Table() [256]uint8 { return f.table }

// MakeGammaTable fills table with a gamma curve: table[i] = round((i/255)^gamma * 255).
func MakeGammaTable(table *[256]uint8, gamma float32) {
	const dx = float32(1) / 255.0
	x := float32(0)
	for i := 0; i < 256; i++ {
		v := geom.RoundToInt(float32(math.Pow(float64(x), float64(gamma))) * 255)
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		table[i] = uint8(v)
		x += dx
	}
}

// MakeClipTable fills table so values below minV map to 0, above maxV map to 255, and values between ramp linearly.
func MakeClipTable(table *[256]uint8, minV, maxV uint8) {
	if maxV == 0 {
		maxV = 1
	}
	if minV >= maxV {
		minV = maxV - 1
	}
	scale := (1 << 16) * 255 / (int32(maxV) - int32(minV))
	for i := 0; i <= int(minV); i++ {
		table[i] = 0
	}
	for i := int(minV) + 1; i < int(maxV); i++ {
		value := (scale*(int32(i)-int32(minV)) + (1 << 15)) >> 16 // round a 16.16 fixed-point value to int
		table[i] = uint8(value)
	}
	for i := int(maxV); i < 256; i++ {
		table[i] = 255
	}
}

// FilterMask implements MaskFilter: maps each source coverage byte through the LUT.
func (f *tableMaskFilter) FilterMask(src *raster.Mask, _ *geom.Matrix) (*raster.Mask, geom.IPoint, bool) {
	dst := &raster.Mask{Bounds: src.Bounds}
	dst.RowBytes = (dst.Bounds.Width() + 3) &^ 3 // round up to a multiple of 4

	if src.Image != nil {
		size := computeImageSize(dst.Bounds, dst.RowBytes)
		if size == 0 {
			return nil, geom.IPoint{}, false
		}
		dst.Image = make([]uint8, size)

		dstWidth := int(dst.Bounds.Width())
		extraZeros := int(dst.RowBytes) - dstWidth
		si, di := 0, 0
		for y := int(dst.Bounds.Height()) - 1; y >= 0; y-- {
			for x := dstWidth - 1; x >= 0; x-- {
				dst.Image[di+x] = f.table[src.Image[si+x]]
			}
			si += int(src.RowBytes)
			// zero the padding between width and row bytes so blitters can read it safely
			di += dstWidth
			for i := extraZeros - 1; i >= 0; i-- {
				dst.Image[di] = 0
				di++
			}
		}
	}
	return dst, geom.IPoint{}, true
}

// ComputeFastBounds implements MaskFilter: the LUT does not move bounds, so this is the identity.
func (f *tableMaskFilter) ComputeFastBounds(src geom.Rect) geom.Rect {
	return src
}
