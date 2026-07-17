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
	"image/draw"
	"image/gif"

	"golang.org/x/image/bmp"
	"golang.org/x/image/webp"

	"github.com/richardwilkes/canvas/imagecore"
)

///////////////////////////////////////////////////////////////////////////////
// GIF — first frame only, per the public surface's contract.

// gifFirstFrame composites frame 0 onto the full logical canvas (frames may be smaller than the canvas; uncovered area
// is transparent).
func gifFirstFrame(data []byte) (image.Image, error) {
	g, err := gif.DecodeAll(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	frame := g.Image[0]
	canvas := image.Rect(0, 0, g.Config.Width, g.Config.Height)
	if frame.Bounds() == canvas {
		return frame, nil
	}
	full := image.NewNRGBA(canvas)
	draw.Draw(full, frame.Bounds(), frame, frame.Bounds().Min, draw.Src)
	return full, nil
}

func gifCodec() imagecore.Codec {
	return imagecore.Codec{
		Name:    "gif",
		Matches: func(data []byte) bool { return hasPrefix(data, []byte("GIF8")) },
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			// Frame-0 transparency lives in the graphic control extension, which DecodeConfig does not surface; the
			// reported info includes it, so decode the first frame to answer faithfully.
			m, err := gifFirstFrame(data)
			if err != nil {
				return imagecore.ImageInfo{}, false
			}
			b := m.Bounds()
			return makeDecodedInfo(b.Dx(), b.Dy(), false, hasAlpha(m)), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			m, err := gifFirstFrame(data)
			if err != nil {
				return nil, err
			}
			b := m.Bounds()
			return pixelsFromImage(m, makeDecodedInfo(b.Dx(), b.Dy(), false, hasAlpha(m))), nil
		},
	}
}

///////////////////////////////////////////////////////////////////////////////
// BMP

func bmpCodec() imagecore.Codec {
	return imagecore.Codec{
		Name:    "bmp",
		Matches: func(data []byte) bool { return hasPrefix(data, []byte("BM")) },
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			cfg, err := bmp.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return imagecore.ImageInfo{}, false
			}
			return makeDecodedInfo(cfg.Width, cfg.Height, false, cfg.ColorModel == color.NRGBAModel), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			m, err := bmp.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			b := m.Bounds()
			_, isNRGBA := m.(*image.NRGBA)
			return pixelsFromImage(m, makeDecodedInfo(b.Dx(), b.Dy(), false, isNRGBA)), nil
		},
	}
}

///////////////////////////////////////////////////////////////////////////////
// WebP

// webpHasAlpha sniffs the container for the alpha bit: the VP8X feature flag, the VP8L header's alpha_is_used bit, or
// (plain lossy VP8) never.
func webpHasAlpha(data []byte) bool {
	if len(data) < 21 || !hasPrefix(data, []byte("RIFF")) || !bytes.Equal(data[8:12], []byte("WEBP")) {
		return false
	}
	switch {
	case bytes.Equal(data[12:16], []byte("VP8X")):
		return data[20]&0x10 != 0
	case bytes.Equal(data[12:16], []byte("VP8L")):
		if len(data) < 25 || data[20] != 0x2F {
			return false
		}
		hdr := binary.LittleEndian.Uint32(data[21:])
		return hdr>>28&1 != 0 // 14 bits w-1, 14 bits h-1, then alpha_is_used
	default:
		return false
	}
}

func webpCodec() imagecore.Codec {
	return imagecore.Codec{
		Name: "webp",
		Matches: func(data []byte) bool {
			return len(data) >= 12 && hasPrefix(data, []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP"))
		},
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			cfg, err := webp.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return imagecore.ImageInfo{}, false
			}
			return makeDecodedInfo(cfg.Width, cfg.Height, false, webpHasAlpha(data)), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			m, err := webp.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			b := m.Bounds()
			return pixelsFromImage(m, makeDecodedInfo(b.Dx(), b.Dy(), false, webpHasAlpha(data))), nil
		},
	}
}
