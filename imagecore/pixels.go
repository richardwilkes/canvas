// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imagecore

import (
	"encoding/binary"

	"github.com/richardwilkes/canvas/raster"
)

// Pixels is an ImageInfo plus pixel storage. Storage is typed by the color type's element width so the hot consumers
// avoid per-pixel byte assembly: 32-bit device words for the 8888/888x family (byte 0..3 of each word is the color
// type's declared channel order — for RGBA_8888 that is exactly raster.Pixmap's layout), 16-bit elements for 565 and
// F16 (four elements per pixel), plain bytes for A8/Gray8. Exactly one of Words/U16s/Bytes is non-nil.
type Pixels struct {
	Words []uint32
	U16s  []uint16
	Bytes []byte
	Info  ImageInfo
	// RowElems is the row stride in elements of the active slice.
	RowElems int32
}

// elemsPerPixel returns how many elements of the active slice make up one pixel.
func elemsPerPixel(ct ColorType) int32 {
	if ct == ColorTypeRGBAF16 {
		return 4
	}
	return 1
}

// NewPixels returns zeroed pixel storage for info (which must be a supported color type).
func NewPixels(info ImageInfo) *Pixels {
	p := &Pixels{Info: info, RowElems: info.Width * elemsPerPixel(info.ColorType)}
	n := int(p.RowElems) * int(info.Height)
	switch info.ColorType.BytesPerPixel() {
	case 1:
		p.Bytes = make([]byte, n)
	case 2, 8:
		p.U16s = make([]uint16, n)
	default:
		p.Words = make([]uint32, n)
	}
	return p
}

// PixelsFromPixmap wraps pm's storage (shared, not copied) as N32-premul pixels. Surface snapshots use this to share
// the surface's backing store under the copy-on-write protocol.
func PixelsFromPixmap(pm raster.Pixmap) *Pixels {
	return &Pixels{
		Info:     MakeN32Premul(pm.Width, pm.Height),
		Words:    pm.Pix,
		RowElems: pm.RowPixels,
	}
}

// PixmapView returns the pixels as a raster.Pixmap when they are stored as 32-bit RGBA words (the N32 family view the
// sprite blitters and legacy sampler consume). ok=false otherwise.
func (p *Pixels) PixmapView() (raster.Pixmap, bool) {
	if p.Words == nil || p.Info.ColorType != ColorTypeRGBA8888 {
		return raster.Pixmap{}, false
	}
	return raster.Pixmap{
		Pix:       p.Words,
		Width:     p.Info.Width,
		Height:    p.Info.Height,
		RowPixels: p.RowElems,
	}, true
}

// NewPixelsCopyFromBytes decodes the little-endian byte stream data (rowBytes apart per row) into fresh element-typed
// storage. ok=false when rowBytes is too small or data is short.
func NewPixelsCopyFromBytes(info ImageInfo, data []byte, rowBytes int) (*Pixels, bool) {
	bpp := info.ColorType.BytesPerPixel()
	if bpp == 0 || rowBytes < info.MinRowBytes() {
		return nil, false
	}
	h := int(info.Height)
	w := int(info.Width)
	if h > 0 {
		// Required length is rowBytes*(h-1)+w*bpp, computed so an attacker-large rowBytes cannot wrap the sum and
		// bypass the check. A rowBytes big enough to overflow can never be backed by a real buffer, so any overflow
		// is treated as a short buffer. w*bpp is bounded by MinRowBytes (already validated non-negative above).
		lastRow := (h - 1) * rowBytes
		if h > 1 && lastRow/(h-1) != rowBytes {
			return nil, false
		}
		need := lastRow + w*bpp
		if need < lastRow || len(data) < need {
			return nil, false
		}
	}
	p := NewPixels(info)
	for y := 0; y < h; y++ {
		row := data[y*rowBytes:]
		switch {
		case p.Bytes != nil:
			copy(p.Bytes[y*int(p.RowElems):(y*int(p.RowElems))+w], row)
		case p.U16s != nil:
			dst := p.U16s[y*int(p.RowElems):]
			for i := 0; i < w*int(elemsPerPixel(info.ColorType)); i++ {
				dst[i] = binary.LittleEndian.Uint16(row[2*i:])
			}
		default:
			dst := p.Words[y*int(p.RowElems):]
			for i := range w {
				dst[i] = binary.LittleEndian.Uint32(row[4*i:])
			}
		}
	}
	return p, true
}

// Subset returns a view (shared storage) of the pixel rect [l, t, r, b), which must be contained in the pixel bounds.
// Callers are expected to always pass an in-range rect.
func (p *Pixels) Subset(l, t, r, b int32) *Pixels {
	epp := elemsPerPixel(p.Info.ColorType)
	off := int(t)*int(p.RowElems) + int(l)*int(epp)
	sub := &Pixels{
		Info:     p.Info,
		RowElems: p.RowElems,
	}
	sub.Info.Width = r - l
	sub.Info.Height = b - t
	// Keep whole-storage tails so the final row's stride slack stays addressable.
	switch {
	case p.Bytes != nil:
		sub.Bytes = p.Bytes[off:]
	case p.U16s != nil:
		sub.U16s = p.U16s[off:]
	default:
		sub.Words = p.Words[off:]
	}
	return sub
}

// writeToBytes serializes the pixel rows into dst as little-endian bytes with the given row stride (the inverse of
// NewPixelsCopyFromBytes); used by the ReadPixels store path.
func (p *Pixels) writeToBytes(dst []byte, dstRowBytes int) {
	w := int(p.Info.Width) * int(elemsPerPixel(p.Info.ColorType))
	for y := 0; y < int(p.Info.Height); y++ {
		row := dst[y*dstRowBytes:]
		src := y * int(p.RowElems)
		switch {
		case p.Bytes != nil:
			copy(row[:w], p.Bytes[src:src+w])
		case p.U16s != nil:
			for i := range w {
				binary.LittleEndian.PutUint16(row[2*i:], p.U16s[src+i])
			}
		default:
			for i := range w {
				binary.LittleEndian.PutUint32(row[4*i:], p.Words[src+i])
			}
		}
	}
}
