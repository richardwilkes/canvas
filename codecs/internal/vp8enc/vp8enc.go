// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package vp8enc is a pure-Go lossy VP8 (WebP) intra-frame encoder. Scope: intra-only single-keyframe encode per RFC
// 6386, a single segment, a single coefficient partition, the normal loop filter with strength derived from the
// quantizer (filter_strength 60, sharpness 0), full rate-distortion mode decision over all 16x16, 4x4 and chroma 8x8
// intra modes with precomputed level-cost tables (single-pass, no trellis), and the one-pass adaptive token-probability
// update. No alpha (the lossy ALPH chunk is out of scope for the encoder core).
package vp8enc

import (
	"errors"
	"image"
)

// ErrInvalidDimensions is returned when the image is empty or exceeds VP8's 16383x16383 limit.
var ErrInvalidDimensions = errors.New("vp8enc: image dimensions must be in [1, 16383]")

// Encode compresses img into a complete lossy WebP file (RIFF/WEBP container with a VP8 chunk). quality follows the
// standard WebP 0..100 scale (values outside the range are clamped). Alpha is ignored.
func Encode(img image.Image, quality float32) ([]byte, error) {
	e, err := encode(img, quality)
	if err != nil {
		return nil, err
	}
	return e.assemble()
}

// EncodeFrame compresses img into a raw VP8 keyframe bitstream — the payload of a WebP "VP8 " chunk, without the RIFF
// container — for callers that assemble an extended (VP8X) container around it, e.g. to add an ALPH alpha chunk.
// quality follows the standard WebP 0..100 scale (values outside the range are clamped). Alpha is ignored (a lossy
// alpha plane is coded separately by the caller).
func EncodeFrame(img image.Image, quality float32) ([]byte, error) {
	e, err := encode(img, quality)
	if err != nil {
		return nil, err
	}
	return e.frameBytes()
}

// encode runs the encoder and returns it with its state (reconstruction planes, coded partitions) intact, for the
// benefit of tests that verify the reconstruction invariant.
func encode(img image.Image, quality float32) (*encoder, error) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 1 || h < 1 || w > maxDimension || h > maxDimension {
		return nil, ErrInvalidDimensions
	}
	e := newEncoder(w, h)
	e.importPicture(img)
	e.encodeFrame(quality)
	return e, nil
}
