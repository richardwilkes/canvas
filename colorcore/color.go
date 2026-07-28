// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package colorcore provides the core color types: unpremultiplied ARGB packed in a uint32, premultiplied byte and
// float colors with well-defined rounding, and the sRGB transfer helpers. Everything here is color-space-agnostic
// content treated as sRGB; the library is sRGB-only.
package colorcore

import "math"

// Color is an unpremultiplied color packed as A<<24 | R<<16 | G<<8 | B.
type Color uint32

// Common colors.
const (
	Transparent Color = 0x00000000
	Black       Color = 0xFF000000
	DkGray      Color = 0xFF444444
	Gray        Color = 0xFF888888
	LtGray      Color = 0xFFCCCCCC
	White       Color = 0xFFFFFFFF
	Red         Color = 0xFFFF0000
	Green       Color = 0xFF00FF00
	Blue        Color = 0xFF0000FF
	Yellow      Color = 0xFFFFFF00
	Cyan        Color = 0xFF00FFFF
	Magenta     Color = 0xFFFF00FF
)

// ARGB packs the four components into a Color.
func ARGB(a, r, g, b uint8) Color {
	return Color(uint32(a)<<24 | uint32(r)<<16 | uint32(g)<<8 | uint32(b))
}

// RGB packs the three components into a fully opaque Color.
func RGB(r, g, b uint8) Color {
	return ARGB(0xFF, r, g, b)
}

// A returns the alpha channel.
func (c Color) A() uint8 { return uint8(c >> 24) }

// R returns the red channel.
func (c Color) R() uint8 { return uint8(c >> 16) }

// G returns the green channel.
func (c Color) G() uint8 { return uint8(c >> 8) }

// B returns the blue channel.
func (c Color) B() uint8 { return uint8(c) }

// WithAlpha returns the color with the alpha channel replaced.
func (c Color) WithAlpha(a uint8) Color {
	return Color(uint32(c)&0x00FFFFFF | uint32(a)<<24)
}

// PMColor is a premultiplied color in conceptual ARGB order; memory-order swizzling for specific color types is the
// raster layer's concern.
type PMColor uint32

// PMColorARGB packs a premultiplied color.
func PMColorARGB(a, r, g, b uint8) PMColor {
	return PMColor(uint32(a)<<24 | uint32(r)<<16 | uint32(g)<<8 | uint32(b))
}

// A returns the alpha channel.
func (c PMColor) A() uint8 { return uint8(c >> 24) }

// R returns the premultiplied red channel.
func (c PMColor) R() uint8 { return uint8(c >> 16) }

// G returns the premultiplied green channel.
func (c PMColor) G() uint8 { return uint8(c >> 8) }

// B returns the premultiplied blue channel.
func (c PMColor) B() uint8 { return uint8(c) }

// MulDiv255Round returns a*b/255 with round-to-nearest, computed entirely in integer arithmetic.
func MulDiv255Round(a, b uint8) uint8 {
	prod := uint32(a)*uint32(b) + 128
	return uint8((prod + (prod >> 8)) >> 8)
}

// PreMultiplyARGB scales the color channels by alpha and packs the result into a PMColor.
func PreMultiplyARGB(a, r, g, b uint8) PMColor {
	if a != 255 {
		r = MulDiv255Round(r, a)
		g = MulDiv255Round(g, a)
		b = MulDiv255Round(b, a)
	}
	return PMColorARGB(a, r, g, b)
}

// PreMultiply converts the color to its premultiplied form.
func (c Color) PreMultiply() PMColor {
	return PreMultiplyARGB(c.A(), c.R(), c.G(), c.B())
}

// unpremulScale returns the 8.24 fixed-point reciprocal (255 << 24) / alpha with round-to-nearest division ((255<<24) +
// a/2) / a, which lets un-premultiplication replace a per-channel divide with a multiply.
func unpremulScale(a uint8) uint32 {
	if a == 0 {
		return 0
	}
	return uint32((uint64(255<<24) + uint64(a)/2) / uint64(a))
}

// applyUnpremulScale un-premultiplies one channel: (component*scale + 2^23) >> 24, saturated to 255. The shifted
// product only stays within a byte for a canonical premultiplied color (component <= alpha); PMColorARGB is exported
// and validates nothing, so a caller can hand UnPreMultiply a color whose channel exceeds its alpha. Saturating there
// keeps the result a valid color rather than letting the uint8 conversion wrap it to an unrelated value.
func applyUnpremulScale(scale uint32, component uint8) uint8 {
	if v := (uint64(component)*uint64(scale) + 1<<23) >> 24; v < 255 {
		return uint8(v)
	}
	return 255
}

// UnPreMultiply converts the premultiplied color back to an unpremultiplied Color, dividing each color channel by alpha
// with round-to-nearest arithmetic. A zero alpha yields transparent black. A channel that exceeds the alpha — possible
// only for a non-canonical premultiplied color, which PMColorARGB does not reject — saturates to 255.
func (c PMColor) UnPreMultiply() Color {
	a := c.A()
	if a == 0 {
		return 0
	}
	if a == 255 {
		return Color(c)
	}
	scale := unpremulScale(a)
	return ARGB(a,
		applyUnpremulScale(scale, c.R()),
		applyUnpremulScale(scale, c.G()),
		applyUnpremulScale(scale, c.B()))
}

// Color4f is an unpremultiplied color with float RGBA components.
type Color4f struct {
	R float32
	G float32
	B float32
	A float32
}

// Color4fFromColor converts a packed Color to float components, scaling each byte by the float constant 1/255.
func Color4fFromColor(c Color) Color4f {
	const inv255 = float32(1.0 / 255)
	return Color4f{
		R: float32(c.R()) * inv255,
		G: float32(c.G()) * inv255,
		B: float32(c.B()) * inv255,
		A: float32(c.A()) * inv255,
	}
}

// toUnorm8 converts one float channel to a byte: v = x*255 + 0.5, pinned to [0, 255], then a truncating cast — the +0.5
// makes the truncation round half away from zero for the expected positive inputs (a 76.5 tie goes to 77 where
// round-to-even would give 76). The explicit float32 conversion around the multiply keeps the two roundings the doc
// describes from being contracted into one: without it arm64 emits FMADDS while amd64 emits a separate multiply and
// add, so the executed arithmetic differs by platform. Both forms happen to agree on every one of the 2^32 float32
// inputs (TestToUnorm8MatchesTwoStepRounding pins the contract), but the library pins goldens on bit-exact output, so
// the arithmetic is written so that agreement does not depend on which instructions a back end chooses to emit.
func toUnorm8(x float32) uint8 {
	v := float32(x*255) + 0.5
	if !(v > 0) { // handles NaN
		return 0
	}
	if v > 255 {
		v = 255
	}
	return uint8(v)
}

// ToColor converts the float color to a packed Color, rounding each channel to a byte.
func (c Color4f) ToColor() Color {
	return ARGB(toUnorm8(c.A), toUnorm8(c.R), toUnorm8(c.G), toUnorm8(c.B))
}

// Premul returns the premultiplied form of the color.
func (c Color4f) Premul() PMColor4f {
	return PMColor4f{R: c.R * c.A, G: c.G * c.A, B: c.B * c.A, A: c.A}
}

// IsOpaque reports whether the color is fully opaque.
func (c Color4f) IsOpaque() bool {
	return c.A >= 1
}

// PMColor4f is a premultiplied color with float RGBA components.
type PMColor4f struct {
	R float32
	G float32
	B float32
	A float32
}

// Unpremul returns the unpremultiplied form of the color; a zero alpha yields transparent black.
func (c PMColor4f) Unpremul() Color4f {
	if c.A == 0 {
		return Color4f{}
	}
	inv := 1 / c.A
	return Color4f{R: c.R * inv, G: c.G * inv, B: c.B * inv, A: c.A}
}

// Luminance coefficients (ITU-R BT.709).
const (
	LumCoeffR = float32(0.2126)
	LumCoeffG = float32(0.7152)
	LumCoeffB = float32(0.0722)
)

// SRGBToLinear applies the sRGB EOTF (the g=2.4 piecewise curve) to one channel value.
func SRGBToLinear(x float32) float32 {
	if x <= 0.04045 {
		return x / 12.92
	}
	return float32(math.Pow((float64(x)+0.055)/1.055, 2.4))
}

// LinearToSRGB applies the sRGB OETF to one channel value.
func LinearToSRGB(x float32) float32 {
	if x <= 0.0031308 {
		return x * 12.92
	}
	return float32(1.055*math.Pow(float64(x), 1/2.4) - 0.055)
}
