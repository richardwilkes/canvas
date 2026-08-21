// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pixel conversion for the supported color-type matrix (see ColorType.Supported). Since all content is sRGB,
// color-space transform cancellation removes the linearize/encode pair (equal transfer-function hashes), so the steps
// reduce to pure alpha-type conversions: premul→unpremul is a lone unpremul stage, unpremul→premul a lone premul stage,
// and opaque sources convert with no steps at all. Lane selection then follows:
//
//   - a same-color-type memcpy and a dedicated to-alpha8 lane for the common trivial cases;
//   - 8888↔8888 alpha changes ride NEON swizzle kernels: premultiply is (x*a+127)/255, unpremultiply is the normalized
//     reciprocal with round-to-nearest-even (the oracle's arm64 leg; the non-NEON library routes unpremul through the
//     float pipeline instead, so that comparison diverges off arm64);
//   - everything else runs a generic pixel pipeline, which executes in lowp (16-bit integer kernels: 565
//     bit-replication loads, the brute-searched 565 store quantizers, the /256 luma) whenever every stage has a lowp
//     form, and in highp float only when an F16 endpoint or an unpremul stage forces it.
//
// The rows the hot lanes walk dispatch through package-level function variables, the shape raster's span/blit kernels
// use: the portable form of each row lives here under a Generic name with no build tag, so every build compiles it,
// and a goexperiment.simd build's init (convert_simd.go) repoints the variable at an archsimd kernel on qualifying
// hardware. Every substitute is bit-identical to the portable form (locked by TestConvertSIMDMatchesScalar, whose
// integer subtests enumerate the complete (channel, alpha) domains), so the choice of lane changes throughput only,
// never converted bytes. The Generic forms are also the sub-chunk tail of every vector kernel.

package imagecore

import (
	"encoding/binary"
	"math"
)

// xformSteps holds the alpha-type conversion flags remaining after cancellation: only the alpha ops survive for
// sRGB-only content.
type xformSteps struct {
	unpremul bool
	premul   bool
}

// computeXformSteps derives the alpha-conversion steps needed to go from srcAT to dstAT, for sRGB-only content.
func computeXformSteps(srcAT, dstAT AlphaType) xformSteps {
	if dstAT == AlphaTypeOpaque {
		dstAT = srcAT
	}
	if srcAT == dstAT {
		return xformSteps{}
	}
	s := xformSteps{
		unpremul: srcAT == AlphaTypePremul,
		premul:   srcAT != AlphaTypeOpaque && dstAT == AlphaTypePremul,
	}
	// (linearize/encode cancel: same transfer function. unpremul+premul with nothing between also cancels, but that
	// pair only forms when srcAT == dstAT == premul, which returned above.)
	return s
}

func (s xformSteps) none() bool { return s == xformSteps{} }

///////////////////////////////////////////////////////////////////////////////
// shared integer helpers

// div255Round is the exact rounding divide-by-255: (x + 127) / 255; the NEON premultiply kernels compute exactly this.
func div255Round(v uint32) uint32 {
	v += 128
	return (v + v/256) / 256
}

// unpremulChannelRP computes round-to-nearest-even of min(255, (c/255) * (1/(a/255)) * 255), with a == 0 mapping to 0 —
// the NEON unpremultiply kernel's arithmetic.
func unpremulChannelRP(c, a uint32) uint32 {
	if a == 0 {
		return 0
	}
	normA := float32(a) * (1.0 / 255.0)
	invA := 1.0 / normA
	normC := float32(c) * (1.0 / 255.0)
	denorm := normC * invA * 255.0
	v := uint32(math.RoundToEven(float64(denorm)))
	if v > 255 {
		v = 255
	}
	return v
}

// toUnorm scales v by scale and rounds to the nearest integer, ties to even (ARMv8 semantics), clamping to [0, scale].
// NaN maps to 0: it is reachable (loadHighp's F16 lane returns halfToFloat of arbitrary caller-supplied halves, and F16
// is a Supported() source type), a plain `f < 0` clamp would let it through to a cast whose result the Go spec leaves
// implementation-dependent, and 0 is what the ARMv8 lane this mirrors produces (FCVTNU of NaN is 0). The negated
// comparison, rather than a separate IsNaN test, keeps the in-range path a single compare.
func toUnorm(v, scale float32) uint32 {
	f := v * scale
	switch {
	case !(f >= 0): // false for both negatives and NaN
		f = 0
	case f > scale:
		f = scale
	}
	return uint32(math.RoundToEven(float64(f)))
}

// alpha8FromHalf reproduces convert_to_alpha8's F16 lane, (uint8_t)(255 * half): faithful to Skia, it truncates toward
// zero (no round-to-nearest) and does not clamp, so out-of-range extended F16 alphas wrap mod 256. The int32
// intermediate reproduces that truncate-then-low-8-bits semantics deterministically instead of relying on Go's
// implementation-defined out-of-range float→uint8 conversion. Every finite half satisfies |255*half| <= 255*65504, so
// only the em >= 0x7C00 encodings can leave int32 range — halfToFloat maps those to ±Inf and NaN, and they are
// caller-supplied because RGBA_F16 is a Supported() source type, so int32(+Inf) would be exactly the
// implementation-defined conversion the truncation was written to avoid (255 on darwin/arm64, 0 on amd64). They are
// therefore saturated first, the way ARMv8's FCVTZS does: NaN and -Inf take the low byte of INT32_MIN (0) and +Inf the
// low byte of INT32_MAX (0xFF), on every target. The sibling toUnorm guards NaN for the same reason; the two F16 lanes
// must not disagree about determinism.
func alpha8FromHalf(bits uint16) byte {
	f := 255.0 * halfToFloat(bits)
	switch {
	case !(f >= math.MinInt32): // false for NaN as well as -Inf
		return 0
	case f > math.MaxInt32:
		return 0xFF
	}
	return byte(int32(f))
}

///////////////////////////////////////////////////////////////////////////////
// lowp integer pixel forms

// lowpPixel is one pixel in the lowp register file: 8-bit values in u16 lanes.
type lowpPixel struct{ r, g, b, a uint32 }

// loadLowp loads one pixel of p at (x, y) into the lowp register form.
func loadLowp(p *Pixels, x, y int) lowpPixel {
	switch p.Info.ColorType {
	case ColorTypeRGBA8888, ColorTypeBGRA8888, ColorTypeRGB888x:
		v := p.Words[y*int(p.RowElems)+x]
		px := lowpPixel{r: v & 0xFF, g: (v >> 8) & 0xFF, b: (v >> 16) & 0xFF, a: v >> 24}
		if p.Info.ColorType == ColorTypeBGRA8888 {
			px.r, px.b = px.b, px.r
		}
		if p.Info.ColorType == ColorTypeRGB888x {
			px.a = 255 // force_opaque
		}
		return px
	case ColorTypeAlpha8:
		return lowpPixel{a: uint32(p.Bytes[y*int(p.RowElems)+x])}
	case ColorTypeGray8:
		// load_a8 + alpha_to_gray
		v := uint32(p.Bytes[y*int(p.RowElems)+x])
		return lowpPixel{r: v, g: v, b: v, a: 255}
	case ColorTypeRGB565:
		// from_565: bit replication to 8 bits
		v := uint32(p.U16s[y*int(p.RowElems)+x])
		r5 := (v >> 11) & 31
		g6 := (v >> 5) & 63
		b5 := v & 31
		return lowpPixel{r: r5<<3 | r5>>2, g: g6<<2 | g6>>4, b: b5<<3 | b5>>2, a: 255}
	default:
		return lowpPixel{}
	}
}

// storeLowp stores one lowp-register pixel px at x into row, encoded per ct.
func storeLowp(ct ColorType, row []byte, x int, px lowpPixel) {
	switch ct {
	case ColorTypeRGBA8888, ColorTypeBGRA8888, ColorTypeRGB888x:
		if ct == ColorTypeBGRA8888 {
			px.r, px.b = px.b, px.r // swap_rb
		}
		if ct == ColorTypeRGB888x {
			px.a = 255 // force_opaque
		}
		binary.LittleEndian.PutUint32(row[4*x:], px.r|px.g<<8|px.b<<16|px.a<<24)
	case ColorTypeAlpha8:
		row[x] = byte(px.a)
	case ColorTypeGray8:
		// bt709_luminance_or_luma_to_alpha (/256 integer luma) + store_a8
		row[x] = byte((px.r*54 + px.g*183 + px.b*19) / 256)
	case ColorTypeRGB565:
		// store_565_: the brute-searched round-to-nearest quantizers
		r := min(px.r, 255)
		g := min(px.g, 255)
		b := min(px.b, 255)
		v := (r*9+36)/74<<11 | (g*21+42)/85<<5 | (b*9+36)/74
		binary.LittleEndian.PutUint16(row[2*x:], uint16(v))
	}
}

///////////////////////////////////////////////////////////////////////////////
// highp float pixel forms (F16 endpoints and unpremul steps)

// loadHighp loads one pixel of p at (x, y) into normalized float channels.
func loadHighp(p *Pixels, x, y int) (r, g, b, a float32) {
	const inv255 = 1.0 / 255.0
	switch p.Info.ColorType {
	case ColorTypeRGBA8888, ColorTypeBGRA8888, ColorTypeRGB888x:
		v := p.Words[y*int(p.RowElems)+x]
		r = float32(v&0xFF) * inv255
		g = float32((v>>8)&0xFF) * inv255
		b = float32((v>>16)&0xFF) * inv255
		a = float32(v>>24) * inv255
		if p.Info.ColorType == ColorTypeBGRA8888 {
			r, b = b, r
		}
		if p.Info.ColorType == ColorTypeRGB888x {
			a = 1
		}
	case ColorTypeAlpha8:
		a = float32(p.Bytes[y*int(p.RowElems)+x]) * inv255
	case ColorTypeGray8:
		v := float32(p.Bytes[y*int(p.RowElems)+x]) * inv255
		r, g, b = v, v, v
		a = 1
	case ColorTypeRGB565:
		v := uint32(p.U16s[y*int(p.RowElems)+x])
		r = float32(v&(31<<11)) * (1.0 / (31 << 11))
		g = float32(v&(63<<5)) * (1.0 / (63 << 5))
		b = float32(v&31) * (1.0 / 31)
		a = 1
	case ColorTypeRGBAF16:
		i := y*int(p.RowElems) + 4*x
		r = halfToFloat(p.U16s[i])
		g = halfToFloat(p.U16s[i+1])
		b = halfToFloat(p.U16s[i+2])
		a = halfToFloat(p.U16s[i+3])
	}
	return r, g, b, a
}

// storeHighp stores normalized float channels (r, g, b, a) at x into row, encoded per ct.
func storeHighp(ct ColorType, row []byte, x int, r, g, b, a float32) {
	switch ct {
	case ColorTypeRGBA8888, ColorTypeBGRA8888, ColorTypeRGB888x:
		if ct == ColorTypeBGRA8888 {
			r, b = b, r
		}
		if ct == ColorTypeRGB888x {
			a = 1
		}
		v := toUnorm(r, 255) | toUnorm(g, 255)<<8 | toUnorm(b, 255)<<16 | toUnorm(a, 255)<<24
		binary.LittleEndian.PutUint32(row[4*x:], v)
	case ColorTypeAlpha8:
		row[x] = byte(toUnorm(a, 255))
	case ColorTypeGray8:
		// bt709_luminance_or_luma_to_alpha in float (fma chain, as the compiled C contracts it)
		luma := float32(math.FMA(float64(b), 0.0722, math.FMA(float64(g), 0.7152, float64(r)*0.2126)))
		row[x] = byte(toUnorm(luma, 255))
	case ColorTypeRGB565:
		v := toUnorm(r, 31)<<11 | toUnorm(g, 63)<<5 | toUnorm(b, 31)
		binary.LittleEndian.PutUint16(row[2*x:], uint16(v))
	case ColorTypeRGBAF16:
		binary.LittleEndian.PutUint16(row[8*x:], floatToHalf(r))
		binary.LittleEndian.PutUint16(row[8*x+2:], floatToHalf(g))
		binary.LittleEndian.PutUint16(row[8*x+4:], floatToHalf(b))
		binary.LittleEndian.PutUint16(row[8*x+6:], floatToHalf(a))
	}
}

///////////////////////////////////////////////////////////////////////////////
// conversion row kernels
//
// The rows the three hot lanes walk, factored out of their loops so the loop-invariant color/alpha decisions are made
// once per conversion rather than once per pixel, and so a vector build has something row-shaped to substitute. Only
// the lanes that carry real traffic have kernels; see convert_simd.go for which ones and why.

// The dispatch signatures.
type (
	// convertWordRowFn converts a row of 8888 source words into 8888 destination bytes, swapping R and B when
	// swapRB is set. len(src) is the pixel count; dst holds at least 4*len(src) bytes.
	convertWordRowFn func(dst []byte, src []uint32, swapRB bool)
	// alphaFromWordRowFn writes the alpha byte of each 8888 source word. len(dst) == len(src).
	alphaFromWordRowFn func(dst []byte, src []uint32)
	// fillByteRowFn fills a byte row with one value.
	fillByteRowFn func(dst []byte, v byte)
	// grayToWordRowFn expands a Gray8 row into opaque 8888 destination bytes. dst holds at least 4*len(src) bytes.
	grayToWordRowFn func(dst, src []byte)
)

// The dispatch variables. Each defaults to the portable form below; convert_simd.go's init is the only assignment.
var (
	swizzleWordRowFn    convertWordRowFn   = swizzleWordRowGeneric
	premulWordRowFn     convertWordRowFn   = premulWordRowGeneric
	unpremulWordRowFn   convertWordRowFn   = unpremulWordRowGeneric
	alphaFromWordsRowFn alphaFromWordRowFn = alphaFromWordsRowGeneric
	fillBytesRowFn      fillByteRowFn      = fillBytesRowGeneric
	grayToWordsRowFn    grayToWordRowFn    = grayToWordsRowGeneric
)

// swizzleWordRowGeneric is the 8888↔8888 lane with no alpha step: a straight word copy, or the R/B exchange that turns
// RGBA_8888 into BGRA_8888 (and back — the swap is its own inverse). G and A ride through untouched, so they are kept
// as their already-positioned bit fields rather than re-extracted.
func swizzleWordRowGeneric(dst []byte, src []uint32, swapRB bool) {
	if !swapRB {
		for x, v := range src {
			binary.LittleEndian.PutUint32(dst[4*x:], v)
		}
		return
	}
	for x, v := range src {
		binary.LittleEndian.PutUint32(dst[4*x:], (v>>16)&0xFF|v&0x0000FF00|(v&0xFF)<<16|v&0xFF000000)
	}
}

// premulWordRowGeneric is the 8888↔8888 premultiply lane (RGBA_to_rgbA / RGBA_to_bgrA): each color channel scaled by
// the pixel's alpha with div255Round's exact rounding divide, alpha unchanged.
func premulWordRowGeneric(dst []byte, src []uint32, swapRB bool) {
	for x, v := range src {
		a := v >> 24
		r := div255Round((v & 0xFF) * a)
		g := div255Round((v >> 8 & 0xFF) * a)
		b := div255Round((v >> 16 & 0xFF) * a)
		if swapRB {
			r, b = b, r
		}
		binary.LittleEndian.PutUint32(dst[4*x:], r|g<<8|b<<16|a<<24)
	}
}

// unpremulWordRowGeneric is the 8888↔8888 unpremultiply lane (rgbA_to_RGBA / rgbA_to_BGRA): each color channel divided
// by the pixel's alpha through unpremulChannelRP's normalized reciprocal, alpha unchanged.
func unpremulWordRowGeneric(dst []byte, src []uint32, swapRB bool) {
	for x, v := range src {
		a := v >> 24
		r := unpremulChannelRP(v&0xFF, a)
		g := unpremulChannelRP(v>>8&0xFF, a)
		b := unpremulChannelRP(v>>16&0xFF, a)
		if swapRB {
			r, b = b, r
		}
		binary.LittleEndian.PutUint32(dst[4*x:], r|g<<8|b<<16|a<<24)
	}
}

// alphaFromWordsRowGeneric is convert_to_alpha8's 8888 lane: the top byte of each source word.
func alphaFromWordsRowGeneric(dst []byte, src []uint32) {
	for x, v := range src {
		dst[x] = byte(v >> 24)
	}
}

// fillBytesRowGeneric fills a byte row with one value — convert_to_alpha8's always-opaque lane, where the source color
// type carries no alpha.
func fillBytesRowGeneric(dst []byte, v byte) {
	for i := range dst {
		dst[i] = v
	}
}

// grayToWordsRowGeneric is the lowp pipeline's Gray8 → 8888 lane. loadLowp's Gray8 case is load_a8 + alpha_to_gray,
// which yields {v, v, v, 255}; a premul step on it is the identity (div255Round(v*255) == v for every byte v), and so
// is storeLowp's R/B swap (r == b) and its RGB_888x force_opaque (a is already 255) — so this one kernel covers
// Gray8 to any of the three 8888-family destinations, with or without the premultiply stage.
func grayToWordsRowGeneric(dst, src []byte) {
	for x, v := range src {
		w := uint32(v)
		binary.LittleEndian.PutUint32(dst[4*x:], w|w<<8|w<<16|0xFF000000)
	}
}

///////////////////////////////////////////////////////////////////////////////

// ConvertPixels converts src into dst (rows dstRowBytes apart, little-endian bytes) with dstInfo's color/alpha types.
// Dimensions must match. Returns false for unsupported color types (outside the supported color-type matrix, which is
// this package's support contract).
func ConvertPixels(dstInfo ImageInfo, dst []byte, dstRowBytes int, src *Pixels) bool {
	if dstInfo.Width != src.Info.Width || dstInfo.Height != src.Info.Height {
		return false
	}
	if !dstInfo.ColorType.Supported() || !src.Info.ColorType.Supported() {
		return false
	}
	w := int(src.Info.Width)
	h := int(src.Info.Height)
	steps := computeXformSteps(src.Info.AlphaType, dstInfo.AlphaType)

	// rect_memcpy: same color type and either no steps or an A8 destination (alpha carries through).
	if dstInfo.ColorType == src.Info.ColorType &&
		(dstInfo.ColorType == ColorTypeAlpha8 || steps.none()) {
		src.writeToBytes(dst, dstRowBytes)
		return true
	}

	// 8888↔8888 with at most an R/B swap and one alpha op, on the NEON kernels. The alpha step and the swap are
	// loop-invariant, so the lane is chosen once here and the rows run a specialized kernel rather than re-testing the
	// flags per pixel.
	is8888 := func(ct ColorType) bool { return ct == ColorTypeRGBA8888 || ct == ColorTypeBGRA8888 }
	if is8888(dstInfo.ColorType) && is8888(src.Info.ColorType) {
		swap := dstInfo.ColorType != src.Info.ColorType
		row := swizzleWordRowFn
		switch {
		case steps.premul:
			row = premulWordRowFn
		case steps.unpremul:
			row = unpremulWordRowFn
		}
		for y := range h {
			row(dst[y*dstRowBytes:], src.Words[y*int(src.RowElems):][:w], swap)
		}
		return true
	}

	// convert_to_alpha8 (alpha-type steps are irrelevant for the alpha channel).
	if dstInfo.ColorType == ColorTypeAlpha8 {
		for y := range h {
			row := dst[y*dstRowBytes:]
			switch src.Info.ColorType {
			case ColorTypeGray8, ColorTypeRGB565, ColorTypeRGB888x:
				fillBytesRowFn(row[:w], 0xFF)
			case ColorTypeRGBA8888, ColorTypeBGRA8888:
				alphaFromWordsRowFn(row[:w], src.Words[y*int(src.RowElems):][:w])
			case ColorTypeRGBAF16:
				s := src.U16s[y*int(src.RowElems):]
				for x := range w {
					// See alpha8FromHalf for the truncate-then-low-8-bits semantics and the saturation
					// that keeps the non-finite halves off Go's implementation-defined cast.
					row[x] = alpha8FromHalf(s[4*x+3])
				}
			}
		}
		return true
	}

	// convert_with_pipeline. The oracle runs lowp when every stage has a lowp form: an F16 endpoint or the unpremul
	// stage forces highp.
	useLowp := src.Info.ColorType != ColorTypeRGBAF16 && dstInfo.ColorType != ColorTypeRGBAF16 &&
		!steps.unpremul
	if useLowp {
		// Gray8 → 8888 is the one lowp pair with real traffic (a decoded grayscale image uploading as N32), and its
		// whole pipeline collapses to one broadcast; see grayToWordsRowGeneric. Every other pair keeps the per-pixel
		// pipeline below.
		if src.Info.ColorType == ColorTypeGray8 && (dstInfo.ColorType == ColorTypeRGBA8888 ||
			dstInfo.ColorType == ColorTypeBGRA8888 || dstInfo.ColorType == ColorTypeRGB888x) {
			for y := range h {
				grayToWordsRowFn(dst[y*dstRowBytes:], src.Bytes[y*int(src.RowElems):][:w])
			}
			return true
		}
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
