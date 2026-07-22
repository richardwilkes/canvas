// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GPU-side accessors: gpu/gl's fragment-processor conversion for color filters dispatches on the concrete filter type.
// The concrete types stay unexported; these helpers expose exactly the descriptor data the FP constructors need, plus
// the sRGB transfer-function pair the working-format sandwich uses.

package colorfilter

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// AsMatrix reports whether cf is the 4x5 matrix filter (RGBA domain, clamping — the reachable form), returning its
// row-major 4x5 matrix.
func AsMatrix(cf shaders.ColorFilter) ([20]float32, bool) {
	if f, ok := cf.(*matrixFilter); ok {
		return f.mat, true
	}
	return [20]float32{}, false
}

// AsBlend reports whether cf is the blend-mode filter, returning its (unpremul sRGB) color and mode.
func AsBlend(cf shaders.ColorFilter) (colorcore.Color4f, raster.BlendMode, bool) {
	if f, ok := cf.(*blendFilter); ok {
		return f.color, f.mode, true
	}
	return colorcore.Color4f{}, 0, false
}

// AsCompose reports whether cf is the compose filter, returning outer and inner.
func AsCompose(cf shaders.ColorFilter) (outer, inner shaders.ColorFilter, ok bool) {
	if f, isCompose := cf.(*composeFilter); isCompose {
		return f.outer, f.inner, true
	}
	return nil, nil, false
}

// AsLuma reports whether cf is the luma filter.
func AsLuma(cf shaders.ColorFilter) bool {
	_, ok := cf.(lumaFilter)
	return ok
}

// AsHighContrast reports whether cf is the high-contrast filter, returning its shader uniforms: grayscale, invert
// style, and the precomputed (1+c)/(1-c) contrast value.
func AsHighContrast(cf shaders.ColorFilter) (grayscale bool, invertStyle HighContrastInvertStyle, contrast float32, ok bool) {
	if f, isHC := cf.(*highContrastFilter); isHC {
		return f.grayscale, f.invertStyle, f.contrast, true
	}
	return false, InvertNone, 0, false
}

// SRGBTF returns the parametric sRGB transfer function: the linearizing EOTF the working-format sandwich applies on
// entry.
func SRGBTF() shaders.TransferFn { return srgbTF }

// SRGBInvTF returns the inverse sRGB transfer function (derived from SRGBTF at init): the encoding OETF the sandwich
// applies on exit.
func SRGBInvTF() shaders.TransferFn { return srgbInvTF }

// EvalTransferFn evaluates a parametric (sRGB-style) transfer function at x with the scalar approximation math — the
// CPU reference the GPU FPs' constant folding uses.
func EvalTransferFn(tf *shaders.TransferFn, x float32) float32 { return colorcore.EvalSkcmsTF(tf, x) }

// EvalHighContrast runs the high-contrast math (grayscale/invert/contrast, without the working-format sandwich) over
// one unpremul linear color — the CPU reference for the GPU FP's constant folding.
func EvalHighContrast(grayscale bool, invertStyle HighContrastInvertStyle, contrast, r, g, b float32) (outR, outG, outB float32) {
	if grayscale {
		y := 0.2126*r + 0.7152*g + 0.0722*b
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
	r = saturate(0.5 + (r-0.5)*contrast)
	g = saturate(0.5 + (g-0.5)*contrast)
	b = saturate(0.5 + (b-0.5)*contrast)
	return r, g, b
}
