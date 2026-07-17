// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imagecore

import "math"

// HalfToFloat is the exported form of halfToFloat for the GPU texture lanes (gpu/gl's F16 gradient LUT bake).
func HalfToFloat(h uint16) float32 { return halfToFloat(h) }

// FloatToHalf is the exported form of floatToHalf for the GPU texture lanes: round-to-nearest-even, denormals honored,
// overflow saturates to infinity.
func FloatToHalf(f float32) uint16 { return floatToHalf(f) }

// halfToFloat converts an IEEE binary16 value to float32, honoring denormals, infinities and NaN — the semantics of the
// aarch64 hardware conversion the oracle's F16 lanes ride.
func halfToFloat(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	em := uint32(h & 0x7FFF)
	switch {
	case em == 0:
		return math.Float32frombits(sign)
	case em >= 0x7C00: // inf / NaN
		return math.Float32frombits(sign | 0x7F800000 | (em&0x03FF)<<13)
	case em < 0x0400: // denormal half: value = em * 2^-24
		v := float32(em) * float32(math.Ldexp(1, -24))
		if sign != 0 {
			v = -v
		}
		return v
	default:
		return math.Float32frombits(sign | (em<<13 + ((127 - 15) << 23)))
	}
}

// floatToHalf converts float32 to IEEE binary16 with round-to-nearest-even, honoring denormals and saturating overflow
// to infinity — again the hardware semantics.
func floatToHalf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	em := bits & 0x7FFFFFFF

	if em >= 0x7F800000 { // inf / NaN
		if em > 0x7F800000 { // NaN: keep a payload bit set
			return sign | 0x7C00 | uint16((em>>13)&0x03FF) | 0x0200
		}
		return sign | 0x7C00
	}

	// Round-to-nearest-even by adding the rounding bias in the integer domain after rebiasing the exponent. Values
	// below the half denormal range flush toward zero through the shift.
	const f16max = (127 + 16) << 23
	const denormMagic = ((127 - 15) + (23 - 10) + 1) << 23

	if em >= f16max { // result overflows half range -> inf
		return sign | 0x7C00
	}
	if em < (113 << 23) { // subnormal (or zero) half
		// Use the float-addition trick: adding denorm_magic shifts the mantissa into place with correct
		// round-to-nearest-even.
		v := math.Float32frombits(em) + math.Float32frombits(denormMagic)
		return sign | uint16(math.Float32bits(v)-denormMagic)
	}
	mantOdd := (em >> 13) & 1
	em -= (127 - 15) << 23 // rebias the exponent from binary32 to binary16
	em += 0xFFF + mantOdd  // round to nearest, ties to even
	return sign | uint16(em>>13)
}
