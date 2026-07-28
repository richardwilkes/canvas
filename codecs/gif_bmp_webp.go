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
	"image/draw"
	"image/gif"

	"golang.org/x/image/bmp"
	"golang.org/x/image/webp"

	"github.com/richardwilkes/canvas/imagecore"
)

///////////////////////////////////////////////////////////////////////////////
// GIF — first frame only, per the public surface's contract.

// gifFirstFrame composites frame 0 onto the full logical canvas (frames may be smaller than the canvas; uncovered area
// is transparent). gif.Decode stops after frame 0 rather than LZW-decoding and retaining every frame of an animation
// the way gif.DecodeAll does, so an animated GIF from untrusted input costs one frame of memory rather than the whole
// animation; gif.DecodeConfig supplies the logical screen size that frame is composited onto.
func gifFirstFrame(data []byte) (image.Image, error) {
	cfg, err := gif.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	frame, err := gif.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	canvas := image.Rect(0, 0, cfg.Width, cfg.Height)
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

// bmpAllowsAlpha sniffs the DIB header for the one layout whose alpha channel survives decoding: 32bpp with a header
// larger than BITMAPINFOHEADER (a V4/V5 header). x/image/bmp forces alpha to 0xFF for every other layout, and neither
// end of its API surfaces the distinction — DecodeConfig reports color.RGBAModel for both 24bpp and 32bpp (so an
// NRGBAModel test is dead), while every 32bpp BMP decodes to an *image.NRGBA (so an image-type test over-reports). Both
// DecodeInfo and Decode go through here so they agree, per the imagecore.Codec contract.
func bmpAllowsAlpha(data []byte) bool {
	const fileHeaderLen, infoHeaderLen = 14, 40
	if len(data) < fileHeaderLen+infoHeaderLen {
		return false
	}
	return binary.LittleEndian.Uint16(data[28:]) == 32 && binary.LittleEndian.Uint32(data[14:]) > infoHeaderLen
}

func bmpCodec() imagecore.Codec {
	return imagecore.Codec{
		Name:    "bmp",
		Matches: func(data []byte) bool { return hasPrefix(data, []byte("BM")) },
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			cfg, err := bmp.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return imagecore.ImageInfo{}, false
			}
			return makeDecodedInfo(cfg.Width, cfg.Height, false, bmpAllowsAlpha(data)), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			m, err := bmp.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			b := m.Bounds()
			return pixelsFromImage(m, makeDecodedInfo(b.Dx(), b.Dy(), false, bmpAllowsAlpha(data))), nil
		},
	}
}

///////////////////////////////////////////////////////////////////////////////
// WebP

// webpHeaderHasAlpha sniffs the container for the VP8X feature flag. A set flag is authoritative: x/image/webp gates
// its ALPH parsing on the same bit, so a lossy payload cannot produce alpha without it. A clear flag is not
// authoritative, because a lossless payload's alpha does not come from an ALPH chunk at all — see webpAlpha.
func webpHeaderHasAlpha(data []byte) bool {
	return len(data) >= 21 && hasPrefix(data, []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")) &&
		bytes.Equal(data[12:16], []byte("VP8X")) && data[20]&0x10 != 0
}

// webpIsLossless reports whether the container's image payload is a VP8L chunk, either as the bare stream or inside a
// VP8X container. The walk reads chunk headers only: a 4-byte FourCC, a 4-byte little-endian payload size, then the
// payload padded to an even length.
func webpIsLossless(data []byte) bool {
	if len(data) < 12 || !hasPrefix(data, []byte("RIFF")) || !bytes.Equal(data[8:12], []byte("WEBP")) {
		return false
	}
	for off := 12; off+8 <= len(data); {
		switch fourCC := data[off : off+4]; {
		case bytes.Equal(fourCC, []byte("VP8L")):
			return true
		case bytes.Equal(fourCC, []byte("VP8 ")):
			return false // the lossy payload; no later chunk produces the pixels
		}
		size := int(binary.LittleEndian.Uint32(data[off+4:]))
		off += 8 + size + (size & 1)
	}
	return false
}

// webpAlpha reports whether the decoded WebP carries alpha, given the image its decode produced. The VP8X feature bit
// answers "yes" without looking at pixels, but nothing in the container answers "no" for a lossless payload: the VP8L
// header's alpha_is_used bit is defined by the spec as a hint that "should not impact decoding", and x/image/webp
// ignores it and always decodes the full ARGB plane. A non-conforming file with that bit clear therefore decodes to
// real alpha, and trusting the hint stamped AlphaTypeOpaque on those pixels; the blend fast paths that branch on
// IsOpaque then rendered them with their transparency dropped. The lossless lane is answered by scanning the decoded
// pixels instead, the way the GIF lane is.
func webpAlpha(data []byte, m image.Image) bool {
	if webpHeaderHasAlpha(data) {
		return true
	}
	// Opaque() is exact for every image type the decoder produces (and free for the lossy YCbCr forms, which have no
	// alpha to scan); anything that does not implement it falls back to the per-pixel scan.
	if o, ok := m.(interface{ Opaque() bool }); ok {
		return !o.Opaque()
	}
	return hasAlpha(m)
}

func webpCodec() imagecore.Codec {
	return imagecore.Codec{
		Name: "webp",
		Matches: func(data []byte) bool {
			return len(data) >= 12 && hasPrefix(data, []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP"))
		},
		DecodeInfo: func(data []byte) (imagecore.ImageInfo, bool) {
			if !webpHeaderHasAlpha(data) && webpIsLossless(data) {
				// A lossless payload's alpha cannot be answered from the header, so decode for it here the way
				// the GIF lane does. Decode repeats the work, which is what keeps the two answers identical as
				// the imagecore.Codec contract requires.
				m, err := webp.Decode(bytes.NewReader(data))
				if err != nil {
					return imagecore.ImageInfo{}, false
				}
				b := m.Bounds()
				return makeDecodedInfo(b.Dx(), b.Dy(), false, webpAlpha(data, m)), true
			}
			cfg, err := webp.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				return imagecore.ImageInfo{}, false
			}
			return makeDecodedInfo(cfg.Width, cfg.Height, false, webpHeaderHasAlpha(data)), true
		},
		Decode: func(data []byte) (*imagecore.Pixels, error) {
			m, err := webp.Decode(bytes.NewReader(data))
			if err != nil {
				return nil, err
			}
			b := m.Bounds()
			return pixelsFromImage(m, makeDecodedInfo(b.Dx(), b.Dy(), false, webpAlpha(data, m))), nil
		},
	}
}
