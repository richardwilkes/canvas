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
	"unsafe"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// Pixmap is an RGBA8888 premultiplied pixel buffer — this package's N32 device format. Pixels are stored as device
// words: on a little-endian target the byte order in memory is R, G, B, A, so the word layout is R | G<<8 | B<<16 |
// A<<24. This package is N32-only; the full color-type matrix and the conversions into it live in imagecore.
type Pixmap struct {
	// Pix holds the pixels, RowPixels words per row (rows are contiguous; a row-bytes stride in units other than whole
	// pixels only matters for borrowed-memory surfaces).
	Pix       []uint32
	Width     int32
	Height    int32
	RowPixels int32
}

// NewPixmap returns a zeroed (transparent) Pixmap of the given size.
func NewPixmap(width, height int32) *Pixmap {
	return &Pixmap{
		Pix:       make([]uint32, int(width)*int(height)),
		Width:     width,
		Height:    height,
		RowPixels: width,
	}
}

// WrapBytes wraps caller-provided N32 pixel memory (RGBA8888-premul, this package's device format) as a Pixmap that
// renders into those exact bytes without copying — the storage backing a raster surface created directly over
// caller-owned pixels. pix is the caller's buffer, laid out little-endian R, G, B, A per pixel with rowBytes bytes per
// row; the returned Pixmap aliases it as device words (writes to the Pixmap land in pix). It returns nil for an
// unsupported configuration: when the dimensions are non-positive, rowBytes is not a whole number of pixels or is
// narrower than width, the buffer cannot hold rowBytes*height bytes, or the buffer is not 4-byte aligned (a []uint32
// view requires uint32 alignment; Go-allocated pixel buffers always satisfy this, so the guard only rejects
// deliberately misaligned sub-slices).
func WrapBytes(pix []byte, width, height, rowBytes int32) *Pixmap {
	if width <= 0 || height <= 0 || rowBytes < width*4 || rowBytes%4 != 0 {
		return nil
	}
	rowPixels := rowBytes / 4
	if len(pix) < int(rowBytes)*int(height) {
		return nil
	}
	if uintptr(unsafe.Pointer(&pix[0]))%4 != 0 {
		return nil
	}
	// The view spans rowPixels*height words = rowBytes*height bytes, which the length check above guarantees fits
	// within pix's backing array, so the reinterpreted slice never exceeds the caller's memory.
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&pix[0])), int(rowPixels)*int(height))
	return &Pixmap{
		Pix:       words,
		Width:     width,
		Height:    height,
		RowPixels: rowPixels,
	}
}

// addr returns the index of (x, y) in Pix.
func (p *Pixmap) addr(x, y int32) int {
	return int(y)*int(p.RowPixels) + int(x)
}

// ExtractSubset points result at the intersection of area and this pixmap's bounds, sharing the pixel memory (writes
// through result land in p). Returns false — leaving result untouched — when the intersection is empty. result keeps
// p's RowPixels stride, so rows of the subset are not contiguous unless the subset spans p's full width.
func (p *Pixmap) ExtractSubset(result *Pixmap, area geom.IRect) bool {
	if !area.Intersect(geom.IRectWH(p.Width, p.Height)) {
		return false
	}
	result.Pix = p.Pix[p.addr(area.Left, area.Top):]
	result.Width = area.Width()
	result.Height = area.Height()
	result.RowPixels = p.RowPixels
	return true
}

// RGBA8888Bytes serializes the pixels in memory byte order (R, G, B, A per pixel), the same raw premul layout the
// oracle harness compares.
func (p *Pixmap) RGBA8888Bytes() []byte {
	out := make([]byte, 4*int(p.Width)*int(p.Height))
	i := 0
	for y := int32(0); y < p.Height; y++ {
		row := p.Pix[p.addr(0, y) : p.addr(0, y)+int(p.Width)]
		for _, v := range row {
			out[i] = byte(v)
			out[i+1] = byte(v >> 8)
			out[i+2] = byte(v >> 16)
			out[i+3] = byte(v >> 24)
			i += 4
		}
	}
	return out
}

// deviceRGBA converts a conceptual-ARGB premultiplied color to this pixmap's device word.
func deviceRGBA(c colorcore.PMColor) uint32 {
	return uint32(c.R()) | uint32(c.G())<<8 | uint32(c.B())<<16 | uint32(c.A())<<24
}

// deviceAlphaShift is where alpha lives in a device word (bit 24 for the RGBA configuration).
const deviceAlphaShift = 24

// deviceAlpha extracts the alpha channel of a device word.
func deviceAlpha(v uint32) uint32 {
	return v >> deviceAlphaShift
}

// alpha255To256 computes alpha + 1, which keeps opaque destinations opaque when used as a scale (less accurate than
// a+(a>>7) for non-opaque destinations, but matches the reference implementation this package stays byte-exact with).
func alpha255To256(alpha uint32) uint32 {
	return alpha + 1
}

// alphaMulQ scales all four channels of a packed pixel by scale (0..256).
func alphaMulQ(c, scale uint32) uint32 {
	const mask = 0x00FF00FF
	rb := ((c & mask) * scale) >> 8
	ag := ((c >> 8) & mask) * scale
	return (rb & mask) | (ag &^ mask)
}
