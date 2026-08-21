// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// RGB → YCbCr 4:2:0 conversion using the standard WebP BT.601 fixed-point coefficients with plain (non-gamma-corrected)
// 2x2 chroma averaging.

package vp8enc

import (
	"image"
	"image/color"
)

const (
	yuvFix  = 16 // fixed-point precision for RGB->YUV
	yuvHalf = 1 << (yuvFix - 1)
)

func rgbToY(r, g, b, rounding int) uint8 {
	luma := 16839*r + 33059*g + 6420*b
	return uint8((luma + rounding + (16 << yuvFix)) >> yuvFix) // no need to clip
}

func clipUV(uv, rounding int) uint8 {
	uv = (uv + rounding + (128 << (yuvFix + 2))) >> (yuvFix + 2)
	if uv&^0xff == 0 {
		return uint8(uv)
	}
	if uv < 0 {
		return 0
	}
	return 255
}

func rgbToU(r, g, b, rounding int) uint8 {
	return clipUV(-9719*r-19081*g+28800*b, rounding)
}

func rgbToV(r, g, b, rounding int) uint8 {
	return clipUV(28800*r-24116*g-4684*b, rounding)
}

// importPicture converts img into the encoder's Y/U/V source planes. Alpha is ignored (lossy WebP alpha is a separate
// ALPH chunk, out of scope for the encoder core).
//
// Nothing in this file has a simd kernel, deliberately. Color conversion runs once per image while the DSP kernels
// run thousands of times per macroblock: measured on an M4 Max, this whole function is ~1.0-1.5 ms of a 41 ms 512x384
// photo encode (~3%), and a good part of that is allocating and filling the interleaved rgb staging buffer rather than
// the arithmetic a kernel would replace. The arithmetic that is left would need a three-byte deinterleave (a
// shuffle-heavy, per-arch affair) before any lane could be multiplied, and the RGBA lane would need the 16-bit
// unpremultiply divide reproduced exactly. Reworking the staging buffer away is the larger and safer win here, and it
// is a portable change, not a vector one.
func (e *encoder) importPicture(img image.Image) {
	w, h := e.width, e.height
	uvW := (w + 1) >> 1
	uvH := (h + 1) >> 1
	e.yPlane = make([]uint8, w*h)
	e.uPlane = make([]uint8, uvW*uvH)
	e.vPlane = make([]uint8, uvW*uvH)

	// Gather the RGB rows once (8-bit, alpha dropped), then convert.
	rgb := make([]uint8, 3*w*h)
	fillRGB(img, rgb, w, h)

	for y := 0; y < h; y++ {
		row := rgb[3*w*y:]
		for x := 0; x < w; x++ {
			e.yPlane[y*e.yStride+x] = rgbToY(int(row[3*x]), int(row[3*x+1]), int(row[3*x+2]), yuvHalf)
		}
	}
	// 2x2 chroma sites; boundary pixels are duplicated on odd dimensions, so each site always accumulates four samples
	// (plain averaging, no gamma correction).
	for cy := 0; cy < uvH; cy++ {
		y0 := 2 * cy
		y1 := minInt(y0+1, h-1)
		for cx := 0; cx < uvW; cx++ {
			x0 := 2 * cx
			x1 := minInt(x0+1, w-1)
			var sr, sg, sb int
			for _, pos := range [4][2]int{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}} {
				p := rgb[3*(pos[1]*w+pos[0]):]
				sr += int(p[0])
				sg += int(p[1])
				sb += int(p[2])
			}
			e.uPlane[cy*e.uvStride+cx] = rgbToU(sr, sg, sb, yuvHalf<<2)
			e.vPlane[cy*e.uvStride+cx] = rgbToV(sr, sg, sb, yuvHalf<<2)
		}
	}
}

// fillRGB extracts 8-bit RGB rows from img, with fast paths for the common raster formats.
func fillRGB(img image.Image, rgb []uint8, w, h int) {
	b := img.Bounds()
	switch src := img.(type) {
	case *image.NRGBA:
		for y := 0; y < h; y++ {
			s := src.Pix[src.PixOffset(b.Min.X, b.Min.Y+y):]
			d := rgb[3*w*y:]
			for x := 0; x < w; x++ {
				d[3*x] = s[4*x]
				d[3*x+1] = s[4*x+1]
				d[3*x+2] = s[4*x+2]
			}
		}
	case *image.RGBA:
		// image.RGBA is alpha-premultiplied, so un-premultiply to recover the true RGB, matching the generic
		// color.NRGBAModel path below. Doing a straight byte copy would emit darkened (premultiplied) color.
		for y := range h {
			s := src.Pix[src.PixOffset(b.Min.X, b.Min.Y+y):]
			d := rgb[3*w*y:]
			for x := range w {
				a := s[4*x+3]
				switch a {
				case 0xff:
					d[3*x] = s[4*x]
					d[3*x+1] = s[4*x+1]
					d[3*x+2] = s[4*x+2]
				case 0:
					d[3*x] = 0
					d[3*x+1] = 0
					d[3*x+2] = 0
				default:
					// Replicate color.NRGBAModel's 16-bit division: expand each channel to 16 bits, scale by
					// 0xffff/a, then truncate back to 8 bits.
					a16 := uint32(a) | uint32(a)<<8
					d[3*x] = uint8(((uint32(s[4*x]) | uint32(s[4*x])<<8) * 0xffff / a16) >> 8)
					d[3*x+1] = uint8(((uint32(s[4*x+1]) | uint32(s[4*x+1])<<8) * 0xffff / a16) >> 8)
					d[3*x+2] = uint8(((uint32(s[4*x+2]) | uint32(s[4*x+2])<<8) * 0xffff / a16) >> 8)
				}
			}
		}
	case *image.Gray:
		for y := 0; y < h; y++ {
			s := src.Pix[src.PixOffset(b.Min.X, b.Min.Y+y):]
			d := rgb[3*w*y:]
			for x := 0; x < w; x++ {
				d[3*x] = s[x]
				d[3*x+1] = s[x]
				d[3*x+2] = s[x]
			}
		}
	default:
		for y := 0; y < h; y++ {
			d := rgb[3*w*y:]
			for x := 0; x < w; x++ {
				c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				d[3*x] = c.R
				d[3*x+1] = c.G
				d[3*x+2] = c.B
			}
		}
	}
}
