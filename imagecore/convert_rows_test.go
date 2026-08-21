// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imagecore

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

// convertReference is the pixel-at-a-time form of ConvertPixels: the lanes that now dispatch through row kernels are
// written here exactly as they read before that refactor, so TestConvertPixelsMatchesPerPixelReference can prove the
// hoisted color/alpha decisions and the row kernels produce byte-identical output for every source/destination pair in
// the supported matrix. It shares the untouched helpers (writeToBytes, loadLowp/storeLowp, loadHighp/storeHighp) with
// the real function rather than duplicating them, since those are not what the refactor moved.
func convertReference(dstInfo ImageInfo, dst []byte, dstRowBytes int, src *Pixels) bool {
	if dstInfo.Width != src.Info.Width || dstInfo.Height != src.Info.Height {
		return false
	}
	if !dstInfo.ColorType.Supported() || !src.Info.ColorType.Supported() {
		return false
	}
	w := int(src.Info.Width)
	h := int(src.Info.Height)
	steps := computeXformSteps(src.Info.AlphaType, dstInfo.AlphaType)

	if dstInfo.ColorType == src.Info.ColorType &&
		(dstInfo.ColorType == ColorTypeAlpha8 || steps.none()) {
		src.writeToBytes(dst, dstRowBytes)
		return true
	}

	is8888 := func(ct ColorType) bool { return ct == ColorTypeRGBA8888 || ct == ColorTypeBGRA8888 }
	if is8888(dstInfo.ColorType) && is8888(src.Info.ColorType) {
		swap := dstInfo.ColorType != src.Info.ColorType
		for y := range h {
			row := dst[y*dstRowBytes:]
			s := src.Words[y*int(src.RowElems):]
			for x := range w {
				v := s[x]
				a := v >> 24
				r := v & 0xFF
				g := (v >> 8) & 0xFF
				b := (v >> 16) & 0xFF
				switch {
				case steps.premul:
					r = div255Round(r * a)
					g = div255Round(g * a)
					b = div255Round(b * a)
				case steps.unpremul:
					r = unpremulChannelRP(r, a)
					g = unpremulChannelRP(g, a)
					b = unpremulChannelRP(b, a)
				}
				if swap {
					r, b = b, r
				}
				binary.LittleEndian.PutUint32(row[4*x:], r|g<<8|b<<16|a<<24)
			}
		}
		return true
	}

	if dstInfo.ColorType == ColorTypeAlpha8 {
		for y := range h {
			row := dst[y*dstRowBytes:]
			switch src.Info.ColorType {
			case ColorTypeGray8, ColorTypeRGB565, ColorTypeRGB888x:
				for x := range w {
					row[x] = 0xFF
				}
			case ColorTypeRGBA8888, ColorTypeBGRA8888:
				s := src.Words[y*int(src.RowElems):]
				for x := range w {
					row[x] = byte(s[x] >> 24)
				}
			case ColorTypeRGBAF16:
				s := src.U16s[y*int(src.RowElems):]
				for x := range w {
					row[x] = alpha8FromHalf(s[4*x+3])
				}
			}
		}
		return true
	}

	useLowp := src.Info.ColorType != ColorTypeRGBAF16 && dstInfo.ColorType != ColorTypeRGBAF16 &&
		!steps.unpremul
	if useLowp {
		for y := range h {
			row := dst[y*dstRowBytes:]
			for x := range w {
				px := loadLowp(src, x, y)
				if steps.premul {
					px.r = div255Round(px.r * px.a)
					px.g = div255Round(px.g * px.a)
					px.b = div255Round(px.b * px.a)
				}
				storeLowp(dstInfo.ColorType, row, x, px)
			}
		}
		return true
	}
	inf := math.Float32frombits(0x7F800000)
	for y := range h {
		row := dst[y*dstRowBytes:]
		for x := range w {
			r, g, b, a := loadHighp(src, x, y)
			if steps.unpremul {
				var scale float32
				if v := 1.0 / a; v < inf {
					scale = v
				}
				r *= scale
				g *= scale
				b *= scale
			}
			if steps.premul {
				r *= a
				g *= a
				b *= a
			}
			storeHighp(dstInfo.ColorType, row, x, r, g, b, a)
		}
	}
	return true
}

// convertTestPixels builds a (w+3)x(h+2) image of ct filled with deterministic pseudo-random bytes and returns the
// (1,1)-anchored w x h subset of it, so every test conversion runs over a source whose row stride exceeds its width —
// the shape ReadPixels hands ConvertPixels for a sub-rect read.
func convertTestPixels(rng *rand.Rand, ct ColorType, at AlphaType, w, h int32) *Pixels {
	p := NewPixels(ImageInfo{Width: w + 3, Height: h + 2, ColorType: ct, AlphaType: at})
	for i := range p.Bytes {
		p.Bytes[i] = byte(rng.Uint32())
	}
	for i := range p.U16s {
		p.U16s[i] = uint16(rng.Uint32())
	}
	for i := range p.Words {
		p.Words[i] = rng.Uint32()
	}
	return p.Subset(1, 1, 1+w, 1+h)
}

// TestConvertPixelsGray8Exhaustive sweeps every gray level through the Gray8 → 8888 fast path, for all three
// 8888-family destinations and all three destination alpha types, against the pixel-at-a-time reference. The fast path
// asserts that the lowp pipeline collapses for Gray8 sources — the premultiply stage is the identity because alpha is
// 255, and both the R/B swap and force_opaque are no-ops — and this is what proves that for the whole domain rather
// than for the sample TestConvertPixelsMatchesPerPixelReference happens to draw.
func TestConvertPixelsGray8Exhaustive(t *testing.T) {
	src := NewPixels(ImageInfo{Width: 256, Height: 1, ColorType: ColorTypeGray8, AlphaType: AlphaTypeOpaque})
	for i := range src.Bytes {
		src.Bytes[i] = byte(i)
	}
	got := make([]byte, 4*256)
	want := make([]byte, 4*256)
	for _, srcAT := range []AlphaType{AlphaTypeOpaque, AlphaTypePremul, AlphaTypeUnpremul} {
		src.Info.AlphaType = srcAT
		for _, dstCT := range []ColorType{ColorTypeRGBA8888, ColorTypeBGRA8888, ColorTypeRGB888x} {
			for _, dstAT := range []AlphaType{AlphaTypeOpaque, AlphaTypePremul, AlphaTypeUnpremul} {
				dstInfo := ImageInfo{Width: 256, Height: 1, ColorType: dstCT, AlphaType: dstAT}
				if !ConvertPixels(dstInfo, got, 4*256, src) || !convertReference(dstInfo, want, 4*256, src) {
					t.Fatalf("Gray8/%v -> %v/%v conversion failed", srcAT, dstCT, dstAT)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("Gray8/%v -> %v/%v: got %x, want %x", srcAT, dstCT, dstAT, got, want)
				}
			}
		}
	}
}

// TestConvertPixelsMatchesPerPixelReference runs every (source color/alpha, destination color/alpha) pair in the
// supported matrix through both ConvertPixels and the pixel-at-a-time reference above and requires byte-identical
// output, including the destination row's stride slack. Alpha types are enumerated raw rather than through MakeInfo so
// the combinations MakeInfo canonicalizes away (an unpremultiplied 565 destination, for instance) are covered too:
// ConvertPixels accepts a caller-built ImageInfo and must behave the same for them.
func TestConvertPixelsMatchesPerPixelReference(t *testing.T) {
	const w, h = 37, 5
	cts := []ColorType{
		ColorTypeAlpha8, ColorTypeRGB565, ColorTypeRGBA8888, ColorTypeRGB888x, ColorTypeBGRA8888,
		ColorTypeGray8, ColorTypeRGBAF16,
	}
	ats := []AlphaType{AlphaTypeOpaque, AlphaTypePremul, AlphaTypeUnpremul}
	rng := rand.New(rand.NewPCG(51, 52))
	for _, srcCT := range cts {
		for _, srcAT := range ats {
			src := convertTestPixels(rng, srcCT, srcAT, w, h)
			for _, dstCT := range cts {
				for _, dstAT := range ats {
					dstInfo := ImageInfo{Width: w, Height: h, ColorType: dstCT, AlphaType: dstAT}
					rowBytes := w*dstCT.BytesPerPixel() + 7 // deliberate stride slack
					got := make([]byte, rowBytes*h)
					want := make([]byte, rowBytes*h)
					for i := range got {
						got[i] = 0x3C
						want[i] = 0x3C
					}
					if !ConvertPixels(dstInfo, got, rowBytes, src) {
						t.Fatalf("ConvertPixels(%v/%v -> %v/%v) failed", srcCT, srcAT, dstCT, dstAT)
					}
					if !convertReference(dstInfo, want, rowBytes, src) {
						t.Fatalf("reference(%v/%v -> %v/%v) failed", srcCT, srcAT, dstCT, dstAT)
					}
					if !bytes.Equal(got, want) {
						t.Fatalf("%v/%v -> %v/%v: got %x, want %x", srcCT, srcAT, dstCT, dstAT, got, want)
					}
				}
			}
		}
	}
}
