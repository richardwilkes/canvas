// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The object-serialization helpers: color/scalar formatting, UTF-16 hex, RectToArray, together with the stream integer
// text writers (writeDecAsText / writeBigDecAsText) those routines and the document serializer depend on. Path
// emission, matrix transforms, and the graphic state helpers arrive with the PDF device.

package pdf

import (
	"strconv"
	"unicode/utf16"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stream"
)

// hexUpper is the uppercase hex digit alphabet.
const hexUpper = "0123456789ABCDEF"

// writeText writes text's bytes to the stream.
func writeText(s stream.WStream, text string) { s.Write([]byte(text)) }

// writeDecAsText writes dec as a base-10 string.
func writeDecAsText(s stream.WStream, dec int) { s.Write([]byte(strconv.Itoa(dec))) }

// writeBigDecAsText writes dec as a base-10 string, left-padded with '0' to at least minDigits characters. dec is
// assumed non-negative in all PDF call sites.
func writeBigDecAsText(s stream.WStream, dec int64, minDigits int) {
	digits := strconv.FormatInt(dec, 10)
	for len(digits) < minDigits {
		digits = "0" + digits
	}
	s.Write([]byte(digits))
}

// appendScalar appends value's PDF decimal form to dst.
func appendScalar(dst []byte, value float32) []byte { return appendFloatToDecimal(dst, value) }

// permilFactor is 65536 * 1000 / 255 (truncated), the 16.16 fixed-point scale from a byte value to permil.
const permilFactor = (1 << 16) * 1000 / 255

// appendColorComponent appends (value/255) with three significant digits.
func appendColorComponent(dst []byte, value uint8) []byte {
	if value == 255 || value == 0 {
		if value != 0 {
			return append(dst, '1')
		}
		return append(dst, '0')
	}
	// int x = 0.5 + (1000.0 / 255.0) * value, computed in 16.16 fixed point then rounded to int.
	x := int(permilFactor*int32(value)+(1<<15)) >> 16
	return printPermilAsDecimal(dst, x, 3)
}

// floatColorDecimalCount is the number of significant digits appendColorComponentF emits.
const floatColorDecimalCount = 4

// appendColorComponentF appends value clamped to [0,1] with four significant digits.
func appendColorComponentF(dst []byte, value float32) []byte {
	const factor = 10 * 10 * 10 * 10 // 10^4, the four significant digits
	x := int(geom.RoundToInt(value * factor))
	if x >= factor || x <= 0 { // clamp to 0-1
		if x > 0 {
			return append(dst, '1')
		}
		return append(dst, '0')
	}
	return printPermilAsDecimal(dst, x, floatColorDecimalCount)
}

// printPermilAsDecimal appends ".DDD" for x in [1, 10^places), trimming trailing zeros but keeping at least one
// fractional digit.
func printPermilAsDecimal(dst []byte, x, places int) []byte {
	result := make([]byte, places+1)
	result[0] = '.'
	for i := places; i > 0; i-- {
		result[i] = byte('0' + x%10)
		x /= 10
	}
	j := places
	for ; j > 1; j-- {
		if result[j] != '0' {
			break
		}
	}
	return append(dst, result[:j+1]...)
}

// writeUInt16BE writes value as four uppercase hex digits, big-endian.
func writeUInt16BE(s stream.WStream, value uint16) {
	s.Write([]byte{
		hexUpper[value>>12],
		hexUpper[0xF&(value>>8)],
		hexUpper[0xF&(value>>4)],
		hexUpper[0xF&value],
	})
}

// writeUInt8 writes value as two uppercase hex digits.
func writeUInt8(s stream.WStream, value uint8) {
	s.Write([]byte{hexUpper[value>>4], hexUpper[0xF&value]})
}

// writeUTF16beHex writes utf32 as one or two big-endian UTF-16 code units in uppercase hex.
func writeUTF16beHex(s stream.WStream, utf32 rune) {
	units := utf16.Encode([]rune{utf32})
	for _, u := range units {
		writeUInt16BE(s, u)
	}
}

// rectToArray builds the four-element PDF array [left top right bottom].
func rectToArray(r geom.Rect) *Array {
	return makeArrayScalars(r.Left, r.Top, r.Right, r.Bottom)
}
