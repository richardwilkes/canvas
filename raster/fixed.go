// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fixed-point types and helpers. The scan converter's positional math is all 16.16 (Fixed) and 26.6 (FDot6);
// implementing it with exact integer arithmetic keeps the results reproducible bit-for-bit.

// Package raster is the CPU rasterizer: edge lists, scan conversion, blitters and the span pipeline.
package raster

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// Fixed is a 16.16 signed fixed-point value.
type Fixed int32

// Fixed constants.
const (
	FixedOne  Fixed = 1 << 16
	FixedHalf Fixed = 1 << 15
	FixedMax  Fixed = math.MaxInt32
	FixedMin  Fixed = -FixedMax
)

// FloatToFixed converts x to Fixed with truncation and saturation.
func FloatToFixed(x float32) Fixed {
	return Fixed(geom.SaturateToInt(x * 65536))
}

// FixedToFloat converts x to float32 exactly.
func FixedToFloat(x Fixed) float32 {
	return float32(x) * 1.52587890625e-5
}

// FixedRoundToInt rounds x to the nearest integer.
func FixedRoundToInt(x Fixed) int32 {
	return int32(x+FixedHalf) >> 16
}

// FixedCeilToInt rounds x up to the next integer.
func FixedCeilToInt(x Fixed) int32 {
	return int32(x+FixedOne-1) >> 16
}

// FixedFloorToInt rounds x down to the previous integer.
func FixedFloorToInt(x Fixed) int32 {
	return int32(x) >> 16
}

// FixedFloorToFixed rounds x down to the previous whole Fixed unit.
func FixedFloorToFixed(x Fixed) Fixed {
	return x &^ (FixedOne - 1)
}

// FixedCeilToFixed rounds x up to the next whole Fixed unit.
func FixedCeilToFixed(x Fixed) Fixed {
	return (x + FixedOne - 1) &^ (FixedOne - 1)
}

// FixedRoundToFixed rounds x to the nearest whole Fixed unit.
func FixedRoundToFixed(x Fixed) Fixed {
	return (x + FixedHalf) &^ (FixedOne - 1)
}

// FixedAbs returns the absolute value of x.
func FixedAbs(x Fixed) Fixed {
	if x < 0 {
		return -x
	}
	return x
}

// sat32Add computes a saturating int32 addition.
func sat32Add(a, b int32) int32 {
	v := int64(a) + int64(b)
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < math.MinInt32 {
		return math.MinInt32
	}
	return int32(v)
}

// sat32Sub computes a saturating int32 subtraction.
func sat32Sub(a, b int32) int32 {
	v := int64(a) - int64(b)
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < math.MinInt32 {
		return math.MinInt32
	}
	return int32(v)
}

// FixedMul computes (a*b)>>16 in 64-bit.
func FixedMul(a, b Fixed) Fixed {
	return Fixed(int64(a) * int64(b) >> 16)
}

// FixedDiv computes (numer<<16)/denom in 64-bit, clamped to the signed 32-bit range.
func FixedDiv(numer, denom Fixed) Fixed {
	v := (int64(numer) << 16) / int64(denom)
	if v > math.MaxInt32 {
		return FixedMax
	}
	if v < math.MinInt32 {
		return Fixed(math.MinInt32)
	}
	return Fixed(v)
}

///////////////////////////////////////////////////////////////////////////////
// FDot6 (26.6)

// FDot6 is a 26.6 signed fixed-point value.
type FDot6 int32

// FDot6 constants.
const (
	FDot6One  FDot6 = 64
	FDot6Half FDot6 = 32
)

// FloatToFDot6 converts x to FDot6 with truncation (no saturation; callers guarantee range).
func FloatToFDot6(x float32) FDot6 {
	return FDot6(x * 64)
}

// FDot6ToFloat converts x to float32 exactly.
func FDot6ToFloat(x FDot6) float32 {
	return float32(x) * 0.015625
}

// FDot6Floor rounds x down to the previous integer.
func FDot6Floor(x FDot6) int32 {
	return int32(x) >> 6
}

// FDot6Ceil rounds x up to the next integer.
func FDot6Ceil(x FDot6) int32 {
	return int32(x+63) >> 6
}

// FDot6Round rounds x to the nearest integer.
func FDot6Round(x FDot6) int32 {
	return int32(x+FDot6Half) >> 6
}

// FixedToFDot6 converts a 16.16 value to 26.6.
func FixedToFDot6(x Fixed) FDot6 {
	return FDot6(x >> 10)
}

// FDot6ToFixed converts a 26.6 value to 16.16.
func FDot6ToFixed(x FDot6) Fixed {
	return Fixed(x << 10)
}

// FDot6Div computes a/b as Fixed, using the fast 32-bit path when a fits in 16 bits.
func FDot6Div(a, b FDot6) Fixed {
	if a >= math.MinInt16 && a <= math.MaxInt16 {
		return Fixed(int32(a<<16) / int32(b))
	}
	return FixedDiv(Fixed(a), Fixed(b))
}
