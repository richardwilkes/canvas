// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Mask-gamma tables for the LCD16 lane: the per-channel gamma correcting LUTs ("pre-blend") keyed by the paint's
// luminance color, three luminance bits per channel, built from a fixed text contrast and gamma. Neither knob is
// exposed at the API, so the tables are built once at fixed defaults: contrast 0.5 and gamma 0.0 (meaning the sRGB
// curve), quantized through the same 0.8+1 and 2.6 fixed-point representations a scaler rec would store them in. The A8
// lane never sees a pre-blend: it is ignored for every non-LCD rec.

package font

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
)

// Mask-gamma table geometry: 3 luminance bits per channel, 8 tables of 256 entries each.
const (
	maskGammaLumBits   = 3
	maskGammaNumTables = 1 << maskGammaLumBits // 8
	maskGammaLumShift  = 8 - maskGammaLumBits  // 5
)

// maskGammaContrast and maskGammaGamma are the fixed default contrast and gamma, quantized the same way a scaler rec's
// internal fixed-point fields would store them: contrast 0.5 quantizes to 128/255, and gamma 0.0 selects the sRGB
// luminance curve below.
const (
	maskGammaContrast = float32(128) / 255
	maskGammaGamma    = float32(0) // 0 selects the sRGB luminance curve
)

// scale255Lum3 scales a 3-bit value to [0, 255] by bit replication.
func scale255Lum3(base uint32) uint32 {
	base <<= maskGammaLumShift
	lum := base
	for i := uint32(maskGammaLumBits); i < 8; i += maskGammaLumBits {
		lum |= base >> i
	}
	return lum
}

// MaskGammaCanonicalColor returns the closest color representable with 3 luminance bits per channel (alpha forced
// opaque).
func MaskGammaCanonicalColor(c colorcore.Color) colorcore.Color {
	return colorcore.RGB(
		uint8(scale255Lum3(uint32(c.R())>>maskGammaLumShift)),
		uint8(scale255Lum3(uint32(c.G())>>maskGammaLumShift)),
		uint8(scale255Lum3(uint32(c.B())>>maskGammaLumShift)),
	)
}

// srgbToLuma converts an sRGB-encoded luminance value to linear luma.
func srgbToLuma(luminance float32) float32 {
	if luminance <= 0.04045 {
		return luminance / 12.92
	}
	return float32(math.Pow(float64((luminance+0.055)/1.055), 2.4))
}

// srgbFromLuma converts a linear luma value to sRGB encoding.
func srgbFromLuma(luma float32) float32 {
	if luma <= 0.0031308 {
		return luma * 12.92
	}
	return 1.055*float32(math.Pow(float64(luma), 1/2.4)) - 0.055
}

// applyContrast boosts srca's coverage by contrast, more so as srca approaches 1.
func applyContrast(srca, contrast float32) float32 {
	return srca + (1.0-srca)*contrast*srca
}

// roundToU8 rounds x to the nearest integer and truncates to a byte, for in-range inputs.
func roundToU8(x float32) uint8 {
	return uint8(int(math.Floor(float64(x) + 0.5)))
}

// buildCorrectingLUT builds a gamma-correcting LUT using the sRGB luminance curve on both sides (the paint color is
// always in the device color space, so the source and destination conversions are the same curve).
func buildCorrectingLUT(table *[256]uint8, srcI uint32, contrast float32) {
	src := float32(srcI) / 255.0
	linSrc := srgbToLuma(src)
	// Guess at the dst. The perceptual inverse provides smaller visual discontinuities when slight changes to
	// desaturated colors cause a channel to map to a different correcting lut with neighboring srcI.
	dst := 1.0 - src
	linDst := srgbToLuma(dst)

	// Contrast value tapers off to 0 as the src luminance becomes white.
	adjustedContrast := contrast * linDst

	// Remove discontinuity and instability when src is close to dst. The value 1/256 is arbitrary and appears to
	// contain the instability.
	if math.Abs(float64(src-dst)) < 1.0/256.0 {
		for i := range 256 {
			rawSrca := float32(i) / 255.0
			srca := applyContrast(rawSrca, adjustedContrast)
			table[i] = roundToU8(255.0 * srca)
		}
		return
	}
	for i := range 256 {
		rawSrca := float32(i) / 255.0
		srca := applyContrast(rawSrca, adjustedContrast)
		dsta := 1.0 - srca

		// Calculate the output we want.
		linOut := linSrc*srca + dsta*linDst
		out := srgbFromLuma(linOut)

		// Undo what the blit blend will do.
		result := (out - dst) / (src - dst)
		table[i] = roundToU8(255.0 * result)
	}
}

// maskGammaTables holds the table block for the single (default) configuration: 8 tables of 256 entries, table i built
// for the luminance scale255Lum3(i).
var (
	maskGammaOnce   sync.Once
	maskGammaTables [maskGammaNumTables][256]uint8
)

func maskGammaInit() {
	for i := uint32(0); i < maskGammaNumTables; i++ {
		buildCorrectingLUT(&maskGammaTables[i], scale255Lum3(i), maskGammaContrast)
	}
}

// maskPreBlend holds the r/g/b correcting tables for a rec's luminance color. The zero value (all nil) is the
// non-applicable pre-blend.
type maskPreBlend struct {
	r, g, b *[256]uint8
}

// isApplicable reports whether this pre-blend has been populated.
func (p *maskPreBlend) isApplicable() bool { return p.g != nil }

// getMaskPreBlend expands the rec's luminance color into a pre-blend, over the single cached table set (never a linear
// table: the reachable contrast/gamma are the non-linear defaults, and non-LCD recs never call this — their pre-blend
// is ignored).
func getMaskPreBlend(lumBits colorcore.Color) maskPreBlend {
	maskGammaOnce.Do(maskGammaInit)
	return maskPreBlend{
		r: &maskGammaTables[uint32(lumBits.R())>>maskGammaLumShift],
		g: &maskGammaTables[uint32(lumBits.G())>>maskGammaLumShift],
		b: &maskGammaTables[uint32(lumBits.B())>>maskGammaLumShift],
	}
}

// GammaLUTData returns the data block for the single reachable configuration (the fixed contrast + sRGB device gamma;
// see the file comment): 8 rows by 256 entries, row i built for luminance scale255Lum3(i). The E.1 distance-field
// adjust table (gpu/text) derives its per-luminance distance corrections from these rows.
func GammaLUTData() *[maskGammaNumTables][256]uint8 {
	maskGammaOnce.Do(maskGammaInit)
	return &maskGammaTables
}
