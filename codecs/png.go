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
	"image"
	"image/color"
	"image/png"

	"github.com/richardwilkes/canvas/imagecore"
)

var (
	pngMagic    = []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}
	pngTypeTRNS = []byte("tRNS")
	pngTypeIDAT = []byte("IDAT")
	pngTypeIEND = []byte("IEND")
)

// pngHasTRNS reports whether the stream carries a tRNS chunk. png.DecodeConfig stops parsing at IHDR for every
// non-paletted PNG, so the color model it reports never accounts for tRNS even though png.Decode honors it — a
// grayscale or truecolor image with tRNS decodes to an *image.NRGBA/*image.NRGBA64, not to the *image.Gray/*image.RGBA
// the config implies. Sniffing the chunk stream restores the alpha that lane drops. Paletted PNGs need no sniff:
// DecodeConfig runs on to IDAT for those, so the tRNS alpha is already folded into the palette it reports.
func pngHasTRNS(data []byte) bool {
	for off := uint64(len(pngMagic)); off+8 <= uint64(len(data)); {
		length := uint64(binary.BigEndian.Uint32(data[off:]))
		switch typ := data[off+4 : off+8]; {
		case bytes.Equal(typ, pngTypeTRNS):
			return true
		case bytes.Equal(typ, pngTypeIDAT), bytes.Equal(typ, pngTypeIEND):
			return false // tRNS must precede the image data
		}
		off += 12 + length // 4 length + 4 type + data + 4 CRC
	}
	return false
}

// pngModelInfo maps a decoded/DecodeConfig color model to the reported info: grayscale PNGs are Gray8; everything else
// is N32, opaque unless the PNG carries alpha (an alpha channel, gray+alpha, a tRNS chunk, or a translucent palette
// entry). trns must come from pngHasTRNS: for the gray and truecolor models DecodeConfig reports, the model alone
// cannot say whether tRNS is present, yet its presence changes both the decoded image type and the alpha type.
func pngModelInfo(w, h int, model color.Model, palette color.Palette, trns bool) imagecore.ImageInfo {
	gray := model == color.GrayModel || model == color.Gray16Model
	alpha := model == color.NRGBAModel || model == color.NRGBA64Model
	if trns && (gray || model == color.RGBAModel || model == color.RGBA64Model) {
		gray = false
		alpha = true
	}
	for _, c := range palette {
		if _, _, _, a := c.RGBA(); a != 0xFFFF {
			alpha = true
			break
		}
	}
	return makeDecodedInfo(w, h, gray, alpha)
}

func pngCodec() imagecore.Codec {
	return imagecore.Codec{
		Name:    "png",
		Matches: func(data []byte) bool { return hasPrefix(data, pngMagic) },
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			cfg, err := png.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return imagecore.ImageInfo{}, false
			}
			pal, _ := cfg.ColorModel.(color.Palette)
			return pngModelInfo(cfg.Width, cfg.Height, cfg.ColorModel, pal, pngHasTRNS(data)), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			m, err := png.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			cfg, err := png.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			pal, _ := cfg.ColorModel.(color.Palette)
			info := pngModelInfo(cfg.Width, cfg.Height, cfg.ColorModel, pal, pngHasTRNS(data))
			return pixelsFromImage(m, info), nil
		},
	}
}

// decodePNGImage decodes for the ICO container (PNG-compressed directory entries).
func decodePNGImage(data []byte) (image.Image, error) {
	return png.Decode(bytes.NewReader(data))
}
