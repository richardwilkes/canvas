// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Mask-gamma tables for the LCD16 lane: the per-channel gamma-correcting LUTs ("pre-blend") keyed by the paint's
// luminance color at three luminance bits per channel. The tables are built once, on first use, for the device gamma,
// text contrast and correction weight set before then (SetMaskGamma, SetMaskGammaWeight). The A8 lane never sees a
// pre-blend: it is ignored for every non-LCD rec.

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

// maskGammaDefaultContrast is the default text contrast, 0.5, quantized as a scaler rec's fixed-point field would
// store it (128/255).
const maskGammaDefaultContrast = float32(128) / 255

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

// applyContrast boosts srca's coverage by contrast, most at mid coverage; 0 and 1 are unchanged.
func applyContrast(srca, contrast float32) float32 {
	return srca + (1.0-srca)*contrast*srca
}

// roundToU8 rounds x to the nearest integer and truncates to a byte, for in-range inputs.
func roundToU8(x float32) uint8 {
	return uint8(int(math.Floor(float64(x) + 0.5)))
}

// buildCorrectingLUT builds a gamma-correcting LUT over the luminance curve for gamma (sRGB when zero, a power law
// otherwise) on both sides, since the paint color is always in the device color space. weight scales the correction,
// from 0 (raw linear coverage) to 1 (the full correction).
//
// Upstream guards the final divide by (src - dst) with a "src is close to dst" branch, a plain contrast ramp whenever
// |src - dst| < 1/256, because it serves any luminance-bit count. Here the geometry is fixed at 3 bits: maskGammaInit
// only passes scale255Lum3(i) for i in 0..7, so srcI is one of {0, 36, 73, 109, 146, 182, 219, 255} and |2*src - 1|,
// which is |src - dst|, never drops below 0.1451, ~37x the threshold. The branch is therefore omitted;
// TestMaskGammaStabilityBranchUnneeded pins the margin, so a change to the table geometry fails there rather than
// silently reaching an unguarded divide.
func buildCorrectingLUT(table *[256]uint8, srcI uint32, contrast, gamma, weight float32) {
	toLuma, fromLuma := srgbToLuma, srgbFromLuma
	if gamma != 0 {
		toLuma = func(v float32) float32 { return float32(math.Pow(float64(v), float64(gamma))) }
		fromLuma = func(v float32) float32 { return float32(math.Pow(float64(v), 1/float64(gamma))) }
	}
	src := float32(srcI) / 255.0
	linSrc := toLuma(src)
	// Guess at the dst. The perceptual inverse provides smaller visual discontinuities when slight changes to
	// desaturated colors cause a channel to map to a different correcting lut with neighboring srcI.
	dst := 1.0 - src
	linDst := toLuma(dst)

	// Contrast value tapers off to 0 as the src luminance becomes white.
	adjustedContrast := contrast * linDst

	for i := range 256 {
		rawSrca := float32(i) / 255.0
		srca := applyContrast(rawSrca, adjustedContrast)
		dsta := 1.0 - srca

		// Calculate the output we want.
		linOut := linSrc*srca + dsta*linDst
		out := fromLuma(linOut)

		// Undo what the blit blend will do.
		result := (out - dst) / (src - dst)

		// Skipping the lerp at weight 1 keeps the default tables bit-identical to upstream's.
		if weight != 1 {
			result = rawSrca + (result-rawSrca)*weight
		}
		table[i] = roundToU8(255.0 * result)
	}
}

// maskGammaTables holds 8 tables of 256 entries, table i built for the luminance scale255Lum3(i). Once built they are
// immutable: scaler contexts on any goroutine read them through the row pointers getMaskPreBlend hands out, so the
// parameters guarded by maskGammaLock are fixed from then on.
var (
	maskGammaOnce   sync.Once
	maskGammaTables [maskGammaNumTables][256]uint8

	maskGammaLock     sync.Mutex
	maskGammaBuilt    bool
	maskGammaExponent float32 // zero selects the sRGB curve
	maskGammaContrast = maskGammaDefaultContrast
	maskGammaWeight   = float32(1)
)

// SetMaskGamma sets the device gamma and text contrast the LCD lane's correcting tables are built for, the two values
// Skia takes from SkSurfaceProps and a platform from the user's text settings. A gamma of zero selects the sRGB curve,
// which with a contrast of 0.5 is the default. It must be called before any LCD text is rendered and reports whether
// the values were taken.
func SetMaskGamma(gamma, contrast float32) bool {
	if !(gamma >= 0) {
		gamma = 0
	}
	if !(contrast >= 0) {
		contrast = 0
	}
	maskGammaLock.Lock()
	defer maskGammaLock.Unlock()
	if maskGammaBuilt {
		return false
	}
	maskGammaExponent = min(gamma, 4)
	maskGammaContrast = min(contrast, 1)
	return true
}

// MaskGamma returns the device gamma (zero for sRGB) and text contrast the correcting tables are or will be built with.
func MaskGamma() (gamma, contrast float32) {
	maskGammaLock.Lock()
	defer maskGammaLock.Unlock()
	return maskGammaExponent, maskGammaContrast
}

// SetMaskGammaWeight sets how much of the gamma correction the LCD lane applies, from 0 (raw linear coverage, which is
// what FreeType hands its clients) to 1 (the full correction, the default). The full correction is what blending in
// sRGB-encoded space calls for, but it draws text visibly heavier than the platform text stacks do. It must be called
// before any LCD text is rendered and reports whether the weight was taken.
func SetMaskGammaWeight(weight float32) bool {
	if !(weight >= 0) {
		weight = 0
	}
	maskGammaLock.Lock()
	defer maskGammaLock.Unlock()
	if maskGammaBuilt {
		return false
	}
	maskGammaWeight = min(weight, 1)
	return true
}

// MaskGammaWeight returns the weight the correcting tables are or will be built with.
func MaskGammaWeight() float32 {
	maskGammaLock.Lock()
	defer maskGammaLock.Unlock()
	return maskGammaWeight
}

func maskGammaInit() {
	maskGammaLock.Lock()
	maskGammaBuilt = true
	gamma, contrast, weight := maskGammaExponent, maskGammaContrast, maskGammaWeight
	maskGammaLock.Unlock()
	for i := uint32(0); i < maskGammaNumTables; i++ {
		buildCorrectingLUT(&maskGammaTables[i], scale255Lum3(i), contrast, gamma, weight)
	}
}

// maskPreBlend holds the r/g/b correcting tables for a rec's luminance color. The zero value (all nil) is the
// non-applicable pre-blend.
type maskPreBlend struct {
	r, g, b *[256]uint8
}

// isApplicable reports whether this pre-blend has been populated.
func (p *maskPreBlend) isApplicable() bool { return p.g != nil }

// getMaskPreBlend expands the rec's luminance color into a pre-blend over the cached table set. Non-LCD recs never
// call this.
func getMaskPreBlend(lumBits colorcore.Color) maskPreBlend {
	maskGammaOnce.Do(maskGammaInit)
	return maskPreBlend{
		r: &maskGammaTables[uint32(lumBits.R())>>maskGammaLumShift],
		g: &maskGammaTables[uint32(lumBits.G())>>maskGammaLumShift],
		b: &maskGammaTables[uint32(lumBits.B())>>maskGammaLumShift],
	}
}

// GammaLUTData returns a copy of the mask-gamma tables (see the file comment): 8 rows by 256 entries, row i built for
// luminance scale255Lum3(i). The distance-field adjust table (gpu/text) derives its per-luminance distance corrections
// from these rows.
//
// The block is returned by value because scaler contexts read the tables concurrently through the row pointers
// getMaskPreBlend hands out: a caller writing through a pointer would corrupt later glyphs and race with those reads.
func GammaLUTData() [maskGammaNumTables][256]uint8 {
	maskGammaOnce.Do(maskGammaInit)
	return maskGammaTables
}
