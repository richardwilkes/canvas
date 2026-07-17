// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Lossy WebP assembly: the VP8 keyframe comes from the pure-Go VP8 encoder (codecs/internal/vp8enc); a non-opaque image
// adds the ALPH chunk in a VP8X extended container using the common alpha configuration (alpha_compression 1: the plane
// coded losslessly as a headerless VP8L stream; alpha_quality 100, so no level pre-processing), falling back to an
// uncompressed (method 0) plane when compression does not shrink it. The VP8L stream is produced by the nativewebp
// encoder the lossless lane already uses, on a synthetic w×h gray image carrying the alpha values in the green channel
// (the channel the ALPH decoder reads); since nativewebp only writes complete WebP files, the headerless stream is
// extracted from the file's VP8L chunk by stripping the fixed 5-byte VP8L header (signature byte + packed 14-bit
// dimensions/alpha-hint/version) — the exact bytes the ALPH spec omits and decoders re-synthesize from the frame
// dimensions. The ALPH prediction filter is always "none", which only affects compressed size, never the decoded plane.

package codecs

import (
	"bytes"
	"encoding/binary"
	"image"

	"github.com/HugoSmits86/nativewebp"

	"github.com/richardwilkes/canvas/codecs/internal/vp8enc"
)

// encodeWebPLossy encodes unpremultiplied RGBA into a complete lossy WebP file. Opaque images get the simple RIFF/VP8
// container (no ALPH chunk or VP8X wrapper); anything else gets VP8X + ALPH + VP8. Returns nil on failure.
func encodeWebPLossy(src *image.NRGBA, quality float32) []byte {
	alpha, opaque := alphaPlane(src)
	if opaque {
		out, err := vp8enc.Encode(src, quality)
		if err != nil {
			return nil
		}
		return out
	}
	frame, err := vp8enc.EncodeFrame(src, quality)
	if err != nil {
		return nil
	}
	b := src.Bounds()
	return assembleExtendedWebP(frame, alphaChunkPayload(alpha, b.Dx(), b.Dy()), b.Dx(), b.Dy())
}

// alphaPlane extracts the w*h alpha plane from src and reports whether every pixel is opaque.
func alphaPlane(src *image.NRGBA) (alpha []byte, opaque bool) {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	alpha = make([]byte, w*h)
	opaque = true
	for y := 0; y < h; y++ {
		row := src.Pix[src.PixOffset(b.Min.X, b.Min.Y+y):]
		for x := 0; x < w; x++ {
			a := row[4*x+3]
			alpha[y*w+x] = a
			opaque = opaque && a == 0xFF
		}
	}
	return alpha, opaque
}

// alphaChunkPayload builds the ALPH chunk payload: the 1-byte header (2-bit compression method, 2-bit prediction
// filter, 2-bit pre-processing, per the container spec) followed by the coded plane.
func alphaChunkPayload(alpha []byte, w, h int) []byte {
	if data := encodeAlphaVP8L(alpha, w, h); data != nil && len(data) < len(alpha) {
		return append([]byte{1}, data...) // method 1 (VP8L), filter none, no pre-processing
	}
	return append([]byte{0}, alpha...) // method 0 (uncompressed), filter none
}

// encodeAlphaVP8L codes the alpha plane as a headerless VP8L stream (ALPH compression method 1), or returns nil if the
// encode fails.
func encodeAlphaVP8L(alpha []byte, w, h int) []byte {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i, a := range alpha {
		m.Pix[4*i] = a   // R = G = B keeps the palette / subtract-green transforms cheap
		m.Pix[4*i+1] = a // the ALPH decoder reads the alpha values back from the green channel
		m.Pix[4*i+2] = a
		m.Pix[4*i+3] = 0xFF // fully opaque, so the encoder cannot treat the payload channels as discardable
	}
	var buf bytes.Buffer
	if err := nativewebp.Encode(&buf, m, nil); err != nil {
		return nil
	}
	payload := riffChunk(buf.Bytes(), "VP8L")
	if len(payload) <= 5 || payload[0] != 0x2F {
		return nil
	}
	return payload[5:]
}

// riffChunk returns the payload of the first chunk with the given FourCC in a RIFF/WEBP file, or nil.
func riffChunk(data []byte, fourCC string) []byte {
	for off := 12; off+8 <= len(data); {
		size := int(binary.LittleEndian.Uint32(data[off+4:]))
		if size < 0 || off+8+size > len(data) {
			return nil
		}
		if string(data[off:off+4]) == fourCC {
			return data[off+8 : off+8+size]
		}
		off += 8 + size + (size & 1)
	}
	return nil
}

// assembleExtendedWebP wraps the coded planes in the extended container: RIFF/WEBP + VP8X (alpha flag) + ALPH + VP8,
// each chunk padded to even length per RIFF.
func assembleExtendedWebP(frame, alphaPayload []byte, w, h int) []byte {
	framePad := len(frame) & 1
	alphaPad := len(alphaPayload) & 1
	riffSize := 4 + (8 + 10) + (8 + len(alphaPayload) + alphaPad) + (8 + len(frame) + framePad)

	out := make([]byte, 0, 8+riffSize)
	out = append(out, 'R', 'I', 'F', 'F')
	out = binary.LittleEndian.AppendUint32(out, uint32(riffSize))
	out = append(out, 'W', 'E', 'B', 'P', 'V', 'P', '8', 'X')
	out = binary.LittleEndian.AppendUint32(out, 10)
	out = append(out, 0x10, 0, 0, 0) // flags byte (bit 4: alpha) + 3 reserved bytes
	out = appendUint24(out, uint32(w-1))
	out = appendUint24(out, uint32(h-1))

	out = append(out, 'A', 'L', 'P', 'H')
	out = binary.LittleEndian.AppendUint32(out, uint32(len(alphaPayload)))
	out = append(out, alphaPayload...)
	if alphaPad != 0 {
		out = append(out, 0)
	}

	out = append(out, 'V', 'P', '8', ' ')
	out = binary.LittleEndian.AppendUint32(out, uint32(len(frame)))
	out = append(out, frame...)
	if framePad != 0 {
		out = append(out, 0)
	}
	return out
}

// appendUint24 appends v as a little-endian 24-bit value.
func appendUint24(out []byte, v uint32) []byte {
	return append(out, byte(v), byte(v>>8), byte(v>>16))
}
