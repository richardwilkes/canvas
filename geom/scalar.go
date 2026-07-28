// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package geom provides the scalar, point, rect, matrix and curve math underlying the rendering pipeline, with float32
// arithmetic and the rounding, saturation, and degenerate-case handling rules the rest of the pipeline depends on for
// float-level consistency.
package geom

import "math"

// The module targets 64-bit platforms only: int and uintptr are assumed to be 64 bits wide, so int arithmetic over
// values already bounded by the int32-typed dimensions, offsets and strides the API uses can never overflow. Code must
// still guard genuine int32 overflow (those types wrap identically on every platform), but never needs to widen int
// math or cap int values merely to survive a 32-bit int. This is the module's base package, so the assertion below
// fails the build of anything that depends on it — "invalid operation: division by zero" — where int is 32 bits: the
// shift yields 1 only when uint has a bit 63.
const _ = 1 / (^uint(0) >> 63)

const (
	// ScalarNearlyZeroTol (1/4096) is the tolerance used for "nearly zero" comparisons throughout geometry code.
	ScalarNearlyZeroTol = 1.0 / (1 << 12)
	// ScalarSinCosNearlyZeroTol is the tolerance below which sin/cos results snap to exactly zero so that rotations
	// by multiples of 90 degrees produce exact matrices.
	ScalarSinCosNearlyZeroTol = 1.0 / (1 << 16)
	// ScalarRoot2Over2 is the conic weight used for 90-degree circular arcs.
	ScalarRoot2Over2 = float32(0.707106781)
	// ScalarPI is pi as a float32 constant.
	ScalarPI = float32(3.14159265)
	// ScalarMax is the largest finite float32 value.
	ScalarMax = float32(3.402823466e+38)

	// maxS32FitsInFloat is the largest int32 exactly representable as a float32. Float-to-int conversions saturate to
	// this value (not MaxInt32) so the clamped float converts to int32 without overflow.
	maxS32FitsInFloat = 2147483520
	minS32FitsInFloat = -maxS32FitsInFloat
)

// IsFinite reports whether all arguments are finite (not NaN and not infinite).
func IsFinite(values ...float32) bool {
	// Accumulate by multiplying each value by zero; any NaN or infinity poisons the product.
	prod := float32(0)
	for _, v := range values {
		prod *= v
	}
	return prod == 0
}

// ScalarIsNaN reports whether x is NaN.
func ScalarIsNaN(x float32) bool {
	return x != x
}

// RoundToScalar computes floor(double(x) + 0.5), in double precision so that tricky values like 0.49999997 and 2^24
// round correctly.
func RoundToScalar(x float32) float32 {
	return float32(math.Floor(float64(x) + 0.5))
}

// SaturateToInt clamps values to [minS32FitsInFloat, maxS32FitsInFloat] before conversion, and maps NaN to
// maxS32FitsInFloat ("x < max ? x : max" is false for NaN). Clamping to the FitsInFloat bounds keeps int32(x) in range,
// so the conversion result is the same on every architecture (out-of-range float-to-int conversions are
// implementation-defined in Go).
func SaturateToInt(x float32) int32 {
	if !(x < maxS32FitsInFloat) {
		x = maxS32FitsInFloat
	}
	if !(x > minS32FitsInFloat) {
		x = minS32FitsInFloat
	}
	return int32(x)
}

// RoundToInt rounds via double floor(x+0.5), then saturates to int32 range.
func RoundToInt(x float32) int32 {
	return SaturateToInt(RoundToScalar(x))
}

// FloorToScalar rounds x down to the nearest integer, as a float32.
func FloorToScalar(x float32) float32 {
	return float32(math.Floor(float64(x)))
}

// CeilToScalar rounds x up to the nearest integer, as a float32.
func CeilToScalar(x float32) float32 {
	return float32(math.Ceil(float64(x)))
}

// FloorToInt rounds x down to the nearest integer, then saturates to int32 range.
func FloorToInt(x float32) int32 {
	return SaturateToInt(FloorToScalar(x))
}

// CeilToInt rounds x up to the nearest integer, then saturates to int32 range.
func CeilToInt(x float32) int32 {
	return SaturateToInt(CeilToScalar(x))
}

// TruncToInt truncates x toward zero, saturating to int32 range.
func TruncToInt(x float32) int32 {
	return SaturateToInt(x)
}

// DegreesToRadians computes degrees * (pi/180) in float32.
func DegreesToRadians(degrees float32) float32 {
	return degrees * (ScalarPI / 180)
}

// RadiansToDegrees computes radians * (180/pi) in float32.
func RadiansToDegrees(radians float32) float32 {
	return radians * (180 / ScalarPI)
}

// ScalarNearlyZero reports |x| <= ScalarNearlyZeroTol, the default tolerance.
func ScalarNearlyZero(x float32) bool {
	return ScalarNearlyZeroTolerance(x, ScalarNearlyZeroTol)
}

// ScalarNearlyZeroTolerance reports |x| <= tolerance.
func ScalarNearlyZeroTolerance(x, tolerance float32) bool {
	return ScalarAbs(x) <= tolerance
}

// ScalarNearlyEqual reports whether x and y are within the default tolerance of each other.
func ScalarNearlyEqual(x, y float32) bool {
	return ScalarAbs(x-y) <= ScalarNearlyZeroTol
}

// ScalarNearlyEqualTolerance reports whether x and y are within the given tolerance of each other.
func ScalarNearlyEqualTolerance(x, y, tolerance float32) bool {
	return ScalarAbs(x-y) <= tolerance
}

// ScalarAbs is |x| in float32, preserving NaN.
func ScalarAbs(x float32) float32 {
	return float32(math.Abs(float64(x)))
}

// ScalarSqrt computes sqrt(x) in float32 (Go's math.Sqrt on the widened value is correctly rounded, and rounding the
// exact double result to float32 gives a correctly-rounded float32 sqrt).
func ScalarSqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// ScalarSin computes sin(radians), in double precision truncated to float32.
func ScalarSin(radians float32) float32 {
	return float32(math.Sin(float64(radians)))
}

// ScalarCos computes cos(radians), in double precision truncated to float32.
func ScalarCos(radians float32) float32 {
	return float32(math.Cos(float64(radians)))
}

// ScalarSinSnapToZero computes sin(radians), with results near zero snapped to exactly zero.
func ScalarSinSnapToZero(radians float32) float32 {
	v := ScalarSin(radians)
	if ScalarNearlyZeroTolerance(v, ScalarSinCosNearlyZeroTol) {
		return 0
	}
	return v
}

// ScalarCosSnapToZero computes cos(radians), with results near zero snapped to exactly zero.
func ScalarCosSnapToZero(radians float32) float32 {
	v := ScalarCos(radians)
	if ScalarNearlyZeroTolerance(v, ScalarSinCosNearlyZeroTol) {
		return 0
	}
	return v
}

// ScalarInterp computes a + (b-a)*t.
func ScalarInterp(a, b, t float32) float32 {
	return a + (b-a)*t
}

// ScalarInvert computes 1/x.
func ScalarInvert(x float32) float32 {
	return 1 / x
}

// ScalarMod computes fmod(x, y) in float32. fmod's exact result needs no more significand bits than its inputs carry,
// so math.Mod over the float64-widened inputs rounds back to the correct float32 result.
func ScalarMod(x, y float32) float32 {
	return float32(math.Mod(float64(x), float64(y)))
}

// IEEEFloatDivide is an unchecked float32 division that may produce infinity or NaN. Go's float division is already
// IEEE-754 compliant; the named wrapper marks intent at call sites.
func IEEEFloatDivide(numer, denom float32) float32 {
	return numer / denom
}

// DoubleToScalar converts a float64 to float32.
func DoubleToScalar(x float64) float32 {
	return float32(x)
}

// scalarAs2sCompliment reinterprets the float bits as sign-magnitude and converts to two's complement so that exact
// comparisons against 0 and 1 work on the integer form. Notably, -0.0 maps to 0.
func scalarAs2sCompliment(x float32) int32 {
	bits := int32(math.Float32bits(x))
	if bits < 0 {
		bits &= 0x7FFFFFFF
		bits = -bits
	}
	return bits
}
