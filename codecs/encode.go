// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The image encoders: PNG via image/png with the zlib level mapped onto Go's coarser scale (encode parity is
// decode-back equality, not byte equality), JPEG via image/jpeg (quality passthrough; alpha is ignored on premul input,
// which feeding Go's encoder premul bytes as RGBA matches), and WebP: lossless through the pure-Go VP8L encoder, lossy
// through the pure-Go VP8 encoder (codecs/internal/vp8enc + webplossy.go).

package codecs

import (
	"bytes"
	"image"
	"image/jpeg"
	"image/png"

	"github.com/HugoSmits86/nativewebp"

	"github.com/richardwilkes/canvas/imagecore"
)

// readNRGBA reads the image back as unpremultiplied RGBA bytes (the encoders' source form; the premul→unpremul
// conversion runs through the standard pixel-conversion steps).
func readNRGBA(img *imagecore.Image) *image.NRGBA {
	w, h := int(img.Width()), int(img.Height())
	info, _ := imagecore.MakeInfo(int32(w), int32(h), imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeUnpremul)
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	if !img.ReadPixels(info, out.Pix, out.Stride, 0, 0, imagecore.CachingAllow) {
		return nil
	}
	return out
}

// readRGBA reads the image back as premultiplied RGBA bytes.
func readRGBA(img *imagecore.Image) *image.RGBA {
	w, h := int(img.Width()), int(img.Height())
	info, _ := imagecore.MakeInfo(int32(w), int32(h), imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	if !img.ReadPixels(info, out.Pix, out.Stride, 0, 0, imagecore.CachingAllow) {
		return nil
	}
	return out
}

// EncodePNG encodes img as PNG: compressionLevel is a zlib level (0..9), mapped onto Go's levels. Returns nil on
// failure.
func EncodePNG(img *imagecore.Image, compressionLevel int) []byte {
	if img == nil {
		return nil
	}
	src := readNRGBA(img)
	if src == nil {
		return nil
	}
	var level png.CompressionLevel
	switch {
	case compressionLevel == 0:
		level = png.NoCompression
	case compressionLevel <= 3:
		level = png.BestSpeed
	case compressionLevel >= 8:
		level = png.BestCompression
	default:
		level = png.DefaultCompression
	}
	var buf bytes.Buffer
	enc := png.Encoder{CompressionLevel: level}
	if err := enc.Encode(&buf, src); err != nil {
		return nil
	}
	return buf.Bytes()
}

// EncodeJPEG encodes img as JPEG. Alpha is ignored on premul input, which feeding the premultiplied bytes as RGBA
// reproduces. Returns nil on failure.
func EncodeJPEG(img *imagecore.Image, quality int) []byte {
	if img == nil {
		return nil
	}
	src := readRGBA(img)
	if src == nil {
		return nil
	}
	if quality < 1 {
		quality = 1
	} else if quality > 100 {
		quality = 100
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: quality}); err != nil {
		return nil
	}
	return buf.Bytes()
}

// EncodeWebP encodes img as WebP. A lossless request encodes through the pure-Go VP8L encoder; a lossy request
// encodes through the pure-Go VP8 encoder with quality on the standard WebP 0..100 scale, adding a losslessly-coded
// ALPH chunk in a VP8X container when the image is not opaque (webplossy.go). Both lanes consume unpremultiplied RGBA.
// Out-of-range quality fails for both compression modes; the in-range value has no lossless meaning (VP8L here is
// always method 0). Returns nil on failure.
func EncodeWebP(img *imagecore.Image, quality float32, lossy bool) []byte {
	if img == nil || quality < 0 || quality > 100 {
		return nil
	}
	src := readNRGBA(img)
	if src == nil {
		return nil
	}
	if lossy {
		return encodeWebPLossy(src, quality)
	}
	var buf bytes.Buffer
	if err := nativewebp.Encode(&buf, src, nil); err != nil {
		return nil
	}
	return buf.Bytes()
}
