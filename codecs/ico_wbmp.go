// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Small native decoders for the two formats Go has no library for: ICO (the entry whose reported info has the largest
// min byte size wins, with the embedded PNG or DIB decoded AND-mask included) and WBMP (type 0, multi-byte-field
// dimensions, 1-bit rows expanded to Gray8).

package codecs

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image/color"
	"image/png"

	"github.com/richardwilkes/canvas/imagecore"
)

///////////////////////////////////////////////////////////////////////////////
// ICO

// maxICODimension bounds a DIB entry's declared width/height, keeping decodeDIB's stride and offset arithmetic within
// int range (the WBMP decoder applies the same 0xFFFF cap to its multi-byte dimensions).
const maxICODimension = 0xFFFF

type icoEntry struct {
	offset int
	size   int
	isPNG  bool
	info   imagecore.ImageInfo // reported info of the embedded codec
}

// icoEntries parses the directory and each referenced entry's header, dropping invalid entries.
func icoEntries(data []byte) []icoEntry {
	if len(data) < 6 || binary.LittleEndian.Uint16(data) != 0 || binary.LittleEndian.Uint16(data[2:]) != 1 {
		return nil
	}
	count := int(binary.LittleEndian.Uint16(data[4:]))
	var out []icoEntry
	for i := range count {
		off := 6 + 16*i
		if off+16 > len(data) {
			break
		}
		size := int(binary.LittleEndian.Uint32(data[off+8:]))
		imgOff := int(binary.LittleEndian.Uint32(data[off+12:]))
		if imgOff < 0 || size <= 0 || imgOff+size > len(data) {
			continue
		}
		sub := data[imgOff : imgOff+size]
		e := icoEntry{offset: imgOff, size: size}
		switch {
		case hasPrefix(sub, pngMagic):
			cfg, err := png.DecodeConfig(bytes.NewReader(sub))
			if err != nil {
				continue
			}
			pal, _ := cfg.ColorModel.(color.Palette)
			e.isPNG = true
			e.info = pngModelInfo(cfg.Width, cfg.Height, cfg.ColorModel, pal, pngHasTRNS(sub))
		default:
			w, h, _, ok := dibHeader(sub)
			if !ok {
				continue
			}
			// BMP-in-ICO reports N32 with unpremul alpha (AND mask / alpha channel), upgraded to premul by the
			// generator.
			e.info = makeDecodedInfo(w, h, false, true)
		}
		out = append(out, e)
	}
	return out
}

// icoBest picks the entry with the largest min byte size (strictly greater, so the first of equals wins).
func icoBest(entries []icoEntry) *icoEntry {
	var best *icoEntry
	maxSize := 0
	for i := range entries {
		sz := entries[i].info.MinRowBytes() * int(entries[i].info.Height)
		if sz > maxSize {
			maxSize = sz
			best = &entries[i]
		}
	}
	return best
}

func icoCodec() imagecore.Codec {
	return imagecore.Codec{
		Name: "ico",
		Matches: func(data []byte) bool {
			return len(data) >= 6 && binary.LittleEndian.Uint16(data) == 0 &&
				binary.LittleEndian.Uint16(data[2:]) == 1 && binary.LittleEndian.Uint16(data[4:]) > 0
		},
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			best := icoBest(icoEntries(data))
			if best == nil {
				return imagecore.ImageInfo{}, false
			}
			return best.info, true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			best := icoBest(icoEntries(data))
			if best == nil {
				return nil, errors.New("ico: no valid directory entry")
			}
			sub := data[best.offset : best.offset+best.size]
			if best.isPNG {
				m, err := decodePNGImage(sub)
				if err != nil {
					return nil, err
				}
				return pixelsFromImage(m, best.info), nil
			}
			return decodeDIB(sub, best.info)
		},
	}
}

// dibHeader parses a BITMAPINFOHEADER-led ICO entry: the stored height is doubled (XOR + AND masks).
func dibHeader(data []byte) (w, h, bpp int, ok bool) {
	if len(data) < 40 {
		return 0, 0, 0, false
	}
	if binary.LittleEndian.Uint32(data) != 40 { // BITMAPINFOHEADER only
		return 0, 0, 0, false
	}
	w = int(int32(binary.LittleEndian.Uint32(data[4:])))
	h2 := int(int32(binary.LittleEndian.Uint32(data[8:])))
	bpp = int(binary.LittleEndian.Uint16(data[14:]))
	compression := binary.LittleEndian.Uint32(data[16:])
	// Cap the dimensions (matching the WBMP sibling) so the downstream stride/offset arithmetic in decodeDIB stays
	// well within int range: an unbounded int32 width/height would overflow xorStride*h, bypassing the short-pixel-data
	// guard and reaching an outsized make.
	if w <= 0 || h2 <= 0 || h2%2 != 0 || compression != 0 || w > maxICODimension || h2 > maxICODimension {
		return 0, 0, 0, false
	}
	switch bpp {
	case 1, 2, 4, 8, 24, 32:
		return w, h2 / 2, bpp, true
	default:
		return 0, 0, 0, false
	}
}

// decodeDIB decodes the ICO DIB payload: bottom-up XOR rows (palette, 24bpp BGR, or 32bpp BGRA) then the bottom-up
// 1-bit AND mask; mask bits force pixels transparent. A 32bpp entry whose alpha channel is entirely zero is treated as
// opaque color data (mask-only transparency), the conventional pre-Vista interpretation.
func decodeDIB(data []byte, info imagecore.ImageInfo) (*imagecore.Pixels, error) {
	w, h, bpp, ok := dibHeader(data)
	if !ok {
		return nil, errors.New("ico: bad DIB header")
	}
	clrUsed := int(binary.LittleEndian.Uint32(data[32:]))
	palOff := 40
	var pal [][4]byte
	if bpp <= 8 {
		n := clrUsed
		if n == 0 {
			n = 1 << bpp
		}
		if palOff+4*n > len(data) {
			return nil, errors.New("ico: short palette")
		}
		pal = make([][4]byte, n)
		for i := range n {
			copy(pal[i][:], data[palOff+4*i:]) // B, G, R, reserved
		}
		palOff += 4 * n
	}

	xorStride := ((w*bpp + 31) / 32) * 4
	andStride := ((w + 31) / 32) * 4
	xorOff := palOff
	andOff := xorOff + xorStride*h
	if andOff+andStride*h > len(data) {
		return nil, errors.New("ico: short pixel data")
	}

	// unpremultiplied BGRA per pixel first
	type px struct{ b, g, r, a byte }
	pixels := make([]px, w*h)
	allAlphaZero := bpp == 32
	for y := range h {
		row := data[xorOff+(h-1-y)*xorStride:] // bottom-up
		for x := range w {
			var v px
			switch bpp {
			case 32:
				v = px{b: row[4*x], g: row[4*x+1], r: row[4*x+2], a: row[4*x+3]}
				if v.a != 0 {
					allAlphaZero = false
				}
			case 24:
				v = px{b: row[3*x], g: row[3*x+1], r: row[3*x+2], a: 0xFF}
			default: // palette: 1/2/4/8 bpp, MSB first
				bits := x * bpp
				idx := int(row[bits/8]) >> (8 - bpp - bits%8) & (1<<bpp - 1)
				if idx >= len(pal) {
					idx = 0
				}
				v = px{b: pal[idx][0], g: pal[idx][1], r: pal[idx][2], a: 0xFF}
			}
			pixels[y*w+x] = v
		}
	}
	if allAlphaZero {
		for i := range pixels {
			pixels[i].a = 0xFF
		}
	}
	// AND mask: set bit → transparent.
	for y := range h {
		row := data[andOff+(h-1-y)*andStride:]
		for x := range w {
			if row[x/8]>>(7-x%8)&1 != 0 {
				pixels[y*w+x] = px{}
			}
		}
	}

	out := imagecore.NewPixels(info)
	for y := range h {
		dst := out.Words[y*int(out.RowElems):]
		for x := range w {
			v := pixels[y*w+x]
			dst[x] = premulWord(uint32(v.r), uint32(v.g), uint32(v.b), uint32(v.a))
		}
	}
	return out, nil
}

///////////////////////////////////////////////////////////////////////////////
// WBMP

// wbmpHeader parses the WBMP header: type 0, fixed header with (data & 0x9F) == 0, multi-byte-field dimensions in (0,
// 0xFFFF].
func wbmpHeader(data []byte) (w, h, hdrLen int, ok bool) {
	if len(data) < 4 || data[0] != 0 || data[1]&0x9F != 0 {
		return 0, 0, 0, false
	}
	pos := 2
	readMBF := func() (uint64, bool) {
		var v uint64
		for {
			if pos >= len(data) || v > 0xFFFF {
				return 0, false
			}
			b := data[pos]
			pos++
			v = v<<7 | uint64(b&0x7F)
			if b&0x80 == 0 {
				return v, true
			}
		}
	}
	wv, ok1 := readMBF()
	hv, ok2 := readMBF()
	if !ok1 || !ok2 || wv == 0 || hv == 0 || wv > 0xFFFF || hv > 0xFFFF {
		return 0, 0, 0, false
	}
	return int(wv), int(hv), pos, true
}

func wbmpCodec() imagecore.Codec {
	return imagecore.Codec{
		Name: "wbmp",
		Matches: func(data []byte) bool {
			_, _, _, ok := wbmpHeader(data)
			return ok
		},
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			w, h, _, ok := wbmpHeader(data)
			if !ok {
				return imagecore.ImageInfo{}, false
			}
			return makeDecodedInfo(w, h, true, false), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			w, h, hdrLen, ok := wbmpHeader(data)
			if !ok {
				return nil, errors.New("wbmp: bad header")
			}
			stride := (w + 7) / 8
			if hdrLen+stride*h > len(data) {
				return nil, errors.New("wbmp: short pixel data")
			}
			info := makeDecodedInfo(w, h, true, false)
			p := imagecore.NewPixels(info)
			for y := range h {
				row := data[hdrLen+y*stride:]
				dst := p.Bytes[y*int(p.RowElems):]
				for x := range w {
					if row[x/8]>>(7-x%8)&1 != 0 {
						dst[x] = 0xFF
					}
				}
			}
			return p, nil
		},
	}
}
