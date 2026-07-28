// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package codecs registers the image decoders: PNG, JPEG (with EXIF orientation applied), GIF (first frame), BMP, WebP,
// plus small native ICO and WBMP decoders, and hosts the image encoders. Stdlib and golang.org/x/image do the
// entropy decoding, rather than porting what Go already provides; the adapters produce the reported image info (with
// the unpremul→premul upgrade) and the codec premultiply rounding.
package codecs

import (
	"bytes"
	"image"
	"image/color"

	"github.com/richardwilkes/canvas/imagecore"
)

// Register installs the seven format decoders into imagecore. Safe to call more than once.
func Register() {
	imagecore.RegisterCodec(pngCodec())
	imagecore.RegisterCodec(jpegCodec())
	imagecore.RegisterCodec(webpCodec())
	imagecore.RegisterCodec(gifCodec())
	imagecore.RegisterCodec(icoCodec())
	imagecore.RegisterCodec(bmpCodec())
	imagecore.RegisterCodec(wbmpCodec())
}

// mulDiv255Round returns a*b/255 with round-to-nearest ((a*b + 128) * 257 >> 16) — the premultiply rounding the codec
// swizzlers use.
func mulDiv255Round(a, b uint32) uint32 {
	prod := a*b + 128
	return (prod + (prod >> 8)) >> 8
}

// premulWord packs unpremultiplied RGBA bytes into a premultiplied RGBA_8888 device word with round-to-nearest
// premultiplication.
func premulWord(r, g, b, a uint32) uint32 {
	if a == 255 {
		return r | g<<8 | b<<16 | 0xFF000000
	}
	if a == 0 {
		return 0
	}
	return mulDiv255Round(r, a) | mulDiv255Round(g, a)<<8 | mulDiv255Round(b, a)<<16 | a<<24
}

// hasAlpha reports whether m contains any non-opaque pixel.
func hasAlpha(m image.Image) bool {
	b := m.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if _, _, _, a := m.At(x, y).RGBA(); a != 0xFFFF {
				return true
			}
		}
	}
	return false
}

// pixelsFromImage converts a decoded Go image into pixel storage matching info (Gray8 or premul/opaque RGBA_8888), with
// the codec premultiply rounding on the type-specific fast paths.
func pixelsFromImage(m image.Image, info imagecore.ImageInfo) *imagecore.Pixels {
	p := imagecore.NewPixels(info)
	b := m.Bounds()
	w, h := int(info.Width), int(info.Height)

	if info.ColorType == imagecore.ColorTypeGray8 {
		switch src := m.(type) {
		case *image.Gray:
			for y := range h {
				copy(p.Bytes[y*int(p.RowElems):y*int(p.RowElems)+w], src.Pix[y*src.Stride:])
			}
		case *image.Gray16:
			for y := range h {
				row := src.Pix[y*src.Stride:]
				dst := p.Bytes[y*int(p.RowElems):]
				for x := range w {
					dst[x] = row[2*x] // 16→8 by high byte
				}
			}
		default:
			for y := range h {
				dst := p.Bytes[y*int(p.RowElems):]
				for x := range w {
					dst[x] = color.GrayModel.Convert(m.At(b.Min.X+x, b.Min.Y+y)).(color.Gray).Y
				}
			}
		}
		return p
	}

	switch src := m.(type) {
	case *image.NRGBA:
		for y := range h {
			row := src.Pix[y*src.Stride:]
			dst := p.Words[y*int(p.RowElems):]
			for x := range w {
				dst[x] = premulWord(uint32(row[4*x]), uint32(row[4*x+1]), uint32(row[4*x+2]), uint32(row[4*x+3]))
			}
		}
	case *image.NRGBA64:
		for y := range h {
			row := src.Pix[y*src.Stride:]
			dst := p.Words[y*int(p.RowElems):]
			for x := range w {
				// 16→8 by high byte, then premultiply.
				dst[x] = premulWord(uint32(row[8*x]), uint32(row[8*x+2]), uint32(row[8*x+4]), uint32(row[8*x+6]))
			}
		}
	case *image.RGBA:
		for y := range h {
			row := src.Pix[y*src.Stride:]
			dst := p.Words[y*int(p.RowElems):]
			for x := range w {
				// already premultiplied
				dst[x] = uint32(row[4*x]) | uint32(row[4*x+1])<<8 | uint32(row[4*x+2])<<16 | uint32(row[4*x+3])<<24
			}
		}
	case *image.RGBA64:
		for y := range h {
			row := src.Pix[y*src.Stride:]
			dst := p.Words[y*int(p.RowElems):]
			for x := range w {
				dst[x] = uint32(row[8*x]) | uint32(row[8*x+2])<<8 | uint32(row[8*x+4])<<16 | uint32(row[8*x+6])<<24
			}
		}
	case *image.Paletted:
		// Premultiply the palette once, then index (the swizzler's palette color xform).
		lut := make([]uint32, len(src.Palette))
		for i, c := range src.Palette {
			nc := color.NRGBAModel.Convert(c).(color.NRGBA)
			lut[i] = premulWord(uint32(nc.R), uint32(nc.G), uint32(nc.B), uint32(nc.A))
		}
		for y := range h {
			row := src.Pix[y*src.Stride:]
			dst := p.Words[y*int(p.RowElems):]
			for x := range w {
				dst[x] = lut[row[x]]
			}
		}
	case *image.Gray:
		for y := range h {
			row := src.Pix[y*src.Stride:]
			dst := p.Words[y*int(p.RowElems):]
			for x := range w {
				v := uint32(row[x])
				dst[x] = v | v<<8 | v<<16 | 0xFF000000
			}
		}
	default:
		// The generic lane (YCbCr, NYCbCrA, CMYK, ...): At() colors are unpremultiplied except via RGBA(), which
		// premultiplies at 16-bit precision; convert through NRGBA to keep the 8-bit premul rounding.
		for y := range h {
			dst := p.Words[y*int(p.RowElems):]
			for x := range w {
				nc := color.NRGBAModel.Convert(m.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				dst[x] = premulWord(uint32(nc.R), uint32(nc.G), uint32(nc.B), uint32(nc.A))
			}
		}
	}
	return p
}

// makeDecodedInfo builds the reported ImageInfo for a color-carrying format: Gray8 for grayscale, RGBA_8888 otherwise;
// premul when alpha is present (the unpremul→premul upgrade), else opaque.
func makeDecodedInfo(w, h int, gray, alpha bool) imagecore.ImageInfo {
	ct := imagecore.ColorTypeRGBA8888
	at := imagecore.AlphaTypeOpaque
	if gray {
		ct = imagecore.ColorTypeGray8
	} else if alpha {
		at = imagecore.AlphaTypePremul
	}
	info, _ := imagecore.MakeInfo(int32(w), int32(h), ct, at)
	return info
}

// hasPrefix sniffs a magic number.
func hasPrefix(data, prefix []byte) bool {
	return len(data) >= len(prefix) && bytes.Equal(data[:len(prefix)], prefix)
}
