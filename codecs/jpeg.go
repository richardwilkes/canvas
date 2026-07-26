// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package codecs

import (
	"bytes"
	"encoding/binary"
	"image/color"
	"image/jpeg"

	"github.com/richardwilkes/canvas/imagecore"
)

func jpegCodec() imagecore.Codec {
	return imagecore.Codec{
		Name:    "jpeg",
		Matches: func(data []byte) bool { return hasPrefix(data, []byte{0xFF, 0xD8, 0xFF}) },
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			cfg, err := jpeg.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return imagecore.ImageInfo{}, false
			}
			w, h := cfg.Width, cfg.Height
			if jpegOrigin(data) >= 5 { // origins 5..8 swap width and height
				w, h = h, w
			}
			return makeDecodedInfo(w, h, cfg.ColorModel == color.GrayModel, false), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			m, err := jpeg.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			b := m.Bounds()
			info := makeDecodedInfo(b.Dx(), b.Dy(), m.ColorModel() == color.GrayModel, false)
			p := pixelsFromImage(m, info)
			return orientPixels(p, jpegOrigin(data)), nil
		},
	}
}

// jpegOrigin extracts the EXIF orientation (1..8; 1 when absent/invalid) from the APP1 Exif segment.
func jpegOrigin(data []byte) int {
	// Walk JPEG segments looking for APP1 "Exif\0\0".
	i := 2 // skip SOI
	for i+1 < len(data) {
		if data[i] != 0xFF {
			return 1
		}
		// ITU T.81 §B.1.1.2 permits any number of 0xFF fill bytes before a marker, so the marker is the first
		// non-0xFF byte after the marker prefix.
		m := i + 1
		for m < len(data) && data[m] == 0xFF {
			m++
		}
		if m >= len(data) {
			return 1
		}
		marker := data[m]
		if marker >= 0xD0 && marker <= 0xD9 { // standalone markers (RSTn, SOI, EOI)
			i = m + 1
			continue
		}
		if m+3 > len(data) {
			return 1
		}
		segLen := int(binary.BigEndian.Uint16(data[m+1:]))
		if segLen < 2 || m+1+segLen > len(data) {
			return 1
		}
		if marker == 0xE1 { // APP1
			seg := data[m+3 : m+1+segLen]
			if o := exifOrientation(seg); o != 0 {
				return o
			}
		}
		if marker == 0xDA { // start of scan: no more headers
			return 1
		}
		i = m + 1 + segLen
	}
	return 1
}

// exifOrientation parses the TIFF structure inside an APP1 Exif payload for tag 0x0112.
func exifOrientation(seg []byte) int {
	if !hasPrefix(seg, []byte("Exif\x00\x00")) {
		return 0
	}
	tiff := seg[6:]
	if len(tiff) < 8 {
		return 0
	}
	var bo binary.ByteOrder
	switch {
	case tiff[0] == 'I' && tiff[1] == 'I':
		bo = binary.LittleEndian
	case tiff[0] == 'M' && tiff[1] == 'M':
		bo = binary.BigEndian
	default:
		return 0
	}
	if bo.Uint16(tiff[2:]) != 42 {
		return 0
	}
	ifd := int(bo.Uint32(tiff[4:]))
	if ifd < 8 || ifd+2 > len(tiff) {
		return 0
	}
	n := int(bo.Uint16(tiff[ifd:]))
	for e := range n {
		off := ifd + 2 + 12*e
		if off+12 > len(tiff) {
			return 0
		}
		if bo.Uint16(tiff[off:]) == 0x0112 && bo.Uint16(tiff[off+2:]) == 3 { // SHORT
			v := int(bo.Uint16(tiff[off+8:]))
			if v >= 1 && v <= 8 {
				return v
			}
			return 0
		}
	}
	return 0
}

// orientPixels applies an EXIF origin to decoded pixels. The returned pixels' info carries the (possibly swapped)
// upright dimensions. Both the RGBA_8888 (Words) and Gray8 (Bytes) storage the JPEG decoder produces are handled; any
// other storage is returned unchanged.
func orientPixels(p *imagecore.Pixels, origin int) *imagecore.Pixels {
	if origin <= 1 || origin > 8 || (p.Words == nil && p.Bytes == nil) {
		return p
	}
	w, h := int(p.Info.Width), int(p.Info.Height)
	outInfo := p.Info
	if origin >= 5 { // origins 5..8 transpose, swapping width and height
		outInfo.Width, outInfo.Height = p.Info.Height, p.Info.Width
	}
	out := imagecore.NewPixels(outInfo)
	srcStride, dstStride := int(p.RowElems), int(out.RowElems)
	for y := range h {
		for x := range w {
			var dx, dy int
			switch origin {
			case 2: // top-right: mirror X
				dx, dy = w-1-x, y
			case 3: // bottom-right: rotate 180
				dx, dy = w-1-x, h-1-y
			case 4: // bottom-left: mirror Y
				dx, dy = x, h-1-y
			case 5: // left-top: transpose
				dx, dy = y, x
			case 6: // right-top: rotate 90 CW
				dx, dy = h-1-y, x
			case 7: // right-bottom: anti-transpose
				dx, dy = h-1-y, w-1-x
			case 8: // left-bottom: rotate 270 CW
				dx, dy = y, w-1-x
			}
			si, di := y*srcStride+x, dy*dstStride+dx
			if p.Words != nil {
				out.Words[di] = p.Words[si]
			} else {
				out.Bytes[di] = p.Bytes[si]
			}
		}
	}
	return out
}
