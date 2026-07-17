// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Float-to-decimal conversion for PDF scalars. PDF does not support numbers in exponential form (6.02e23), so scalars
// are serialized as fixed-point decimal strings of the form /[-]?([0-9]*.)?[0-9]+/ carrying enough precision that a
// float parser reads back the exact original value.

package pdf

import "math"

// maxFloatToDecimalLength is the longest possible encoded length: -FLT_MIN serializes as
// "-.0000000000000000000000000000000000000117549435" (48 characters), and 48 bytes suffice since the result is returned
// as a length-carrying slice with no terminating NUL to reserve space for.
const maxFloatToDecimalLength = 48

// powBySquaring returns value * pow(base, e), assuming e is positive.
func powBySquaring(value, base float64, e int) float64 {
	for {
		if e&1 != 0 {
			value *= base
		}
		e >>= 1
		if e == 0 {
			return value
		}
		base *= base
	}
}

// pow10 returns pow(10.0, e), with fast paths for the small exponents that occur most often.
func pow10(e int) float64 {
	switch e {
	case 0:
		return 1.0
	case 1:
		return 10.0
	case 2:
		return 100.0
	case 3:
		return 1e+03
	case 4:
		return 1e+04
	case 5:
		return 1e+05
	case 6:
		return 1e+06
	case 7:
		return 1e+07
	case 8:
		return 1e+08
	case 9:
		return 1e+09
	case 10:
		return 1e+10
	case 11:
		return 1e+11
	case 12:
		return 1e+12
	case 13:
		return 1e+13
	case 14:
		return 1e+14
	case 15:
		return 1e+15
	default:
		if e > 15 {
			return powBySquaring(1e+15, 10.0, e-15)
		}
		// e < 0.
		return powBySquaring(1.0, 0.1, -e)
	}
}

// appendFloatToDecimal appends value's PDF-scalar decimal form to dst and returns the extended slice. It accepts any
// input, including non-finite values, always emitting a syntactically valid number.
func appendFloatToDecimal(dst []byte, value float32) []byte {
	// This function is written to accept any possible input value, including non-finite values such as INF and NAN; in
	// those cases correctness is ignored and a syntactically-valid number is emitted.
	if value == float32(math.Inf(1)) {
		value = math.MaxFloat32 // nearest finite float.
	}
	if value == float32(math.Inf(-1)) {
		value = -math.MaxFloat32 // nearest finite float.
	}
	if v64 := float64(value); math.IsNaN(v64) || math.IsInf(v64, 0) || value == 0.0 {
		// NAN is unsupported in PDF; always output a valid number. Also catch zero here as a special case.
		return append(dst, '0')
	}
	// start marks the pre-emission length so the denormalized end-of-buffer break below can compare the total
	// characters written against the fixed maxFloatToDecimalLength cap.
	start := len(dst)
	if value < 0.0 {
		dst = append(dst, '-')
		value = -value
	}

	_, binaryExponent := math.Frexp(float64(value))
	const kLog2 = 0.3010299956639812 // log10(2.0)
	decimalExponent := int(math.Floor(kLog2 * float64(binaryExponent)))
	decimalShift := decimalExponent - 8
	power := pow10(-decimalShift)
	d := int(float64(value)*power + 0.5)
	if d > 167772159 { // floor(pow(10,1+log10(1<<24)))
		// Need one fewer decimal digit for 24-bit precision.
		decimalShift = decimalExponent - 7
		// Recalculate to get rounding right.
		d = int(float64(value)*(power*0.1) + 0.5)
	}
	for d%10 == 0 {
		d /= 10
		decimalShift++
	}
	var buffer [9]byte // decimal value buffer.
	bufferIndex := 0
	for {
		buffer[bufferIndex] = byte(d % 10)
		bufferIndex++
		d /= 10
		if d == 0 {
			break
		}
	}
	if decimalShift >= 0 {
		for {
			bufferIndex--
			dst = append(dst, '0'+buffer[bufferIndex])
			if bufferIndex == 0 {
				break
			}
		}
		for i := 0; i < decimalShift; i++ {
			dst = append(dst, '0')
		}
	} else {
		placesBeforeDecimal := bufferIndex + decimalShift
		if placesBeforeDecimal > 0 {
			for placesBeforeDecimal > 0 {
				placesBeforeDecimal--
				bufferIndex--
				dst = append(dst, '0'+buffer[bufferIndex])
			}
			dst = append(dst, '.')
		} else {
			dst = append(dst, '.')
			placesAfterDecimal := -placesBeforeDecimal
			for placesAfterDecimal > 0 {
				placesAfterDecimal--
				dst = append(dst, '0')
			}
		}
		for bufferIndex > 0 {
			bufferIndex--
			dst = append(dst, '0'+buffer[bufferIndex])
			if len(dst)-start == maxFloatToDecimalLength {
				break // denormalized: don't need extra precision.
			}
		}
	}
	return dst
}
