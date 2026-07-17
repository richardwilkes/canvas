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
	"image"
	"image/color"
	"image/png"

	"github.com/richardwilkes/canvas/imagecore"
)

var pngMagic = []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}

// pngModelInfo maps a decoded/DecodeConfig color model to the reported info: grayscale PNGs are Gray8; everything else
// is N32, opaque unless the PNG carries alpha (an alpha channel, gray+alpha, or tRNS — Go's decoder surfaces all of
// those as (N)RGBA models).
func pngModelInfo(w, h int, model color.Model, palette color.Palette) imagecore.ImageInfo {
	gray := model == color.GrayModel || model == color.Gray16Model
	alpha := model == color.NRGBAModel || model == color.NRGBA64Model
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
			return pngModelInfo(cfg.Width, cfg.Height, cfg.ColorModel, pal), true
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
			info := pngModelInfo(cfg.Width, cfg.Height, cfg.ColorModel, pal)
			return pixelsFromImage(m, info), nil
		},
	}
}

// decodePNGImage decodes for the ICO container (PNG-compressed directory entries).
func decodePNGImage(data []byte) (image.Image, error) {
	return png.Decode(bytes.NewReader(data))
}
