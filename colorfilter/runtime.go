// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The two shader-math color filters that are publicly reachable — luma and high contrast — implemented as direct
// per-pixel math, with no shader compiler. High contrast runs inside a linear-unpremul working format, which for
// sRGB-only content reduces to: unpremul, linearize sRGB, filter, encode sRGB, premul.

package colorfilter

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/shaders"
)

// srgbTF is the parametric sRGB transfer function.
var srgbTF = colorcore.SRGBTF

// srgbInvTF is the inverse of srgbTF, computed at init.
var srgbInvTF = colorcore.SRGBInvTF

///////////////////////////////////////////////////////////////////////////////
// luma

// lumaFilter operates on the premultiplied color, moving its luminance into alpha and zeroing RGB.
type lumaFilter struct{}

// NewLuma returns the luma color filter.
func NewLuma() shaders.ColorFilter {
	return lumaFilter{}
}

// IsAlphaUnchanged implements shaders.ColorFilter: the effect replaces alpha.
func (lumaFilter) IsAlphaUnchanged() bool { return false }

// AppendStages implements shaders.ColorFilter: saturate(dot((0.2126, 0.7152, 0.0722), rgb)) becomes the new alpha, RGB
// becomes zero.
func (lumaFilter) AppendStages(p *shaders.Pipeline, _ bool) bool {
	p.AppendColorFunc(func(r, g, b, _ float32) (float32, float32, float32, float32) {
		return 0, 0, 0, saturate(colorcore.LumCoeffR*r + colorcore.LumCoeffG*g + colorcore.LumCoeffB*b)
	})
	return true
}

///////////////////////////////////////////////////////////////////////////////
// high contrast

// HighContrastInvertStyle selects how the high-contrast filter inverts colors.
type HighContrastInvertStyle int32

// HighContrastInvertStyle values.
const (
	InvertNone HighContrastInvertStyle = iota
	InvertBrightness
	InvertLightness
)

// HighContrastConfig configures the high-contrast filter's grayscale, invert style, and contrast.
type HighContrastConfig struct {
	Grayscale   bool
	InvertStyle HighContrastInvertStyle
	Contrast    float32
}

// isValid reports whether the invert style and contrast are within their legal ranges.
func (c HighContrastConfig) isValid() bool {
	return c.InvertStyle >= InvertNone && c.InvertStyle <= InvertLightness &&
		c.Contrast >= -1 && c.Contrast <= 1
}

// highContrastFilter carries the effect's uniforms.
type highContrastFilter struct {
	grayscale   bool
	invertStyle HighContrastInvertStyle
	contrast    float32 // the (1+c)/(1-c) uniform
}

// NewHighContrast returns the high-contrast color filter, or nil for an invalid config.
func NewHighContrast(config HighContrastConfig) shaders.ColorFilter {
	if !config.isValid() {
		return nil
	}
	// A contrast setting of exactly +1 would divide by zero (1+c)/(1-c), so pull in to +1-ε.
	const fltEpsilon = 1.1920928955078125e-07
	c := config.Contrast
	if c < -1+fltEpsilon {
		c = -1 + fltEpsilon
	}
	if c > 1-fltEpsilon {
		c = 1 - fltEpsilon
	}
	return &highContrastFilter{
		grayscale:   config.Grayscale,
		invertStyle: config.InvertStyle,
		contrast:    (1 + c) / (1 - c),
	}
}

// IsAlphaUnchanged implements shaders.ColorFilter: the effect returns the input alpha (the working format only
// unpremuls and re-premuls around it).
func (f *highContrastFilter) IsAlphaUnchanged() bool { return true }

// AppendStages implements shaders.ColorFilter: the working-format sandwich (unpremul + linearize, re-encode + premul)
// around the high-contrast math.
func (f *highContrastFilter) AppendStages(p *shaders.Pipeline, _ bool) bool {
	// dst (premul sRGB) → working (unpremul linear-sRGB)
	p.AppendUnpremul()
	p.AppendTransferFunction(&srgbTF)
	gray, invertStyle, contrast := f.grayscale, f.invertStyle, f.contrast
	p.AppendColorFunc(func(r, g, b, a float32) (float32, float32, float32, float32) {
		if gray {
			y := colorcore.LumCoeffR*r + colorcore.LumCoeffG*g + colorcore.LumCoeffB*b
			r, g, b = y, y, y
		}
		switch invertStyle {
		case InvertBrightness:
			r, g, b = 1-r, 1-g, 1-b
		case InvertLightness:
			h, s, l := highContrastRGBToHSL(r, g, b)
			l = 1 - l
			r, g, b = hslToRGB(h, s, l)
		default:
		}
		// saturate(mix(half3(0.5), color, contrast))
		r = saturate(0.5 + (r-0.5)*contrast)
		g = saturate(0.5 + (g-0.5)*contrast)
		b = saturate(0.5 + (b-0.5)*contrast)
		return r, g, b, a
	})
	// working (unpremul linear-sRGB) → dst (premul sRGB)
	p.AppendTransferFunction(&srgbInvTF)
	p.AppendPremul()
	return true
}

// highContrastRGBToHSL converts linear RGB to HSL using the filter's exact branch structure.
func highContrastRGBToHSL(r, g, b float32) (h, s, l float32) {
	mx := maxf(maxf(r, g), b)
	mn := minf(minf(r, g), b)
	d := mx - mn
	invd := 1.0 / d
	gLtB := float32(0)
	if g < b {
		gLtB = 6
	}
	switch {
	case mx == mn:
		h = 0
	case r >= g && r >= b:
		h = invd*(g-b) + gLtB
	case g >= b:
		h = invd*(b-r) + 2
	default:
		h = invd*(r-g) + 4
	}
	h *= 1.0 / 6.0
	sum := mx + mn
	l = sum * 0.5
	switch {
	case mx == mn:
		s = 0
	case l > 0.5:
		s = d / (2 - sum)
	default:
		s = d / sum
	}
	return h, s, l
}

// hslToRGB converts HSL back to RGB.
func hslToRGB(h, s, l float32) (r, g, b float32) {
	c := (1 - absf(2*l-1)) * s
	f := func(offset float32) float32 {
		p := h + offset
		q := saturate(absf(fractf(p)*6-3) - 1)
		return (q-0.5)*c + l
	}
	return f(0), f(2.0 / 3.0), f(1.0 / 3.0)
}

///////////////////////////////////////////////////////////////////////////////
// float helpers (float32 forms of the shader ops)

func saturate(v float32) float32 {
	return minf(maxf(v, 0), 1)
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func absf(v float32) float32 {
	return math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
}

func fractf(v float32) float32 {
	return v - float32(math.Floor(float64(v)))
}
