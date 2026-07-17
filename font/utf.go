// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The UTF-8/16/32 counting and iteration primitives Typeface/Font use for text→glyph conversion. Validation is strict
// (the count functions return -1 for any invalid input) and the next* iterators fail by returning -1 and jumping to the
// end. UTF-16 and UTF-32 text arrives as raw bytes in native byte order, which on every supported target is
// little-endian; the decoders here read little-endian pairs/quads accordingly.

package font

// utf8ByteType classifies a UTF-8 leading byte: -1 invalid, 0 continuation, else the sequence length (1..4).
func utf8ByteType(c byte) int {
	switch {
	case c < 0x80:
		return 1
	case c < 0xC0:
		return 0
	case c >= 0xF5 || (c&0xFE) == 0xC0: // "octet values c0, c1, f5 to ff never appear"
		return -1
	default:
		// Arithmetic right shift of the int32 bit pattern 0xE5000000 (-0x1B000000): a packed lookup table, 2 bits per
		// high nibble, giving the sequence length minus 1 for each leading-byte range.
		const e5 = int32(-0x1B000000)
		return int((e5>>((uint(c)>>4)<<1))&3 + 1)
	}
}

func utf8ByteIsContinuation(c byte) bool { return utf8ByteType(c) == 0 }

func utf16IsHighSurrogate(c uint16) bool { return c&0xFC00 == 0xD800 }
func utf16IsLowSurrogate(c uint16) bool  { return c&0xFC00 == 0xDC00 }

// countUTF8 returns the number of unichars in the byte sequence, or -1 if invalid.
func countUTF8(utf8 []byte) int {
	count := 0
	i := 0
	for i < len(utf8) {
		t := utf8ByteType(utf8[i])
		if t <= 0 || i+t > len(utf8) { // invalid leading byte, or sequence extends beyond end
			return -1
		}
		for n := t; n > 1; n-- {
			i++
			if !utf8ByteIsContinuation(utf8[i]) {
				return -1
			}
		}
		i++
		count++
	}
	return count
}

// countUTF16 counts unichars over raw bytes: -1 for odd byte lengths or invalid surrogate pairing.
func countUTF16(b []byte) int {
	if len(b)%2 != 0 {
		return -1
	}
	count := 0
	for i := 0; i < len(b); i += 2 {
		c := uint16(b[i]) | uint16(b[i+1])<<8
		if utf16IsLowSurrogate(c) {
			return -1
		}
		if utf16IsHighSurrogate(c) {
			i += 2
			if i >= len(b) {
				return -1
			}
			c = uint16(b[i]) | uint16(b[i+1])<<8
			if !utf16IsLowSurrogate(c) {
				return -1
			}
		}
		count++
	}
	return count
}

// nextUTF8 decodes the unichar at i, returning (unichar, next index). On invalid input it returns (-1, len(b)), jumping
// to the end.
func nextUTF8(b []byte, i int) (unichar int32, next int) {
	if i >= len(b) {
		return -1, len(b)
	}
	c := int32(b[i])
	hic := c << 24
	if utf8ByteType(b[i]) <= 0 {
		return -1, len(b)
	}
	if hic < 0 {
		mask := uint32(0xFFFFFFC0) // ~0x3F
		hic <<= 1
		for {
			i++
			if i >= len(b) {
				return -1, len(b)
			}
			nextByte := b[i]
			if !utf8ByteIsContinuation(nextByte) {
				return -1, len(b)
			}
			c = c<<6 | int32(nextByte&0x3F)
			mask <<= 5
			hic <<= 1
			if hic >= 0 {
				break
			}
		}
		c &= int32(^mask)
	}
	return c, i + 1
}

// nextUTF16 decodes the unichar at byte index i over raw little-endian bytes, returning (unichar, next index). On
// invalid input it returns (-1, len(b)).
func nextUTF16(b []byte, i int) (unichar int32, next int) {
	if i+2 > len(b) {
		return -1, len(b)
	}
	c := uint16(b[i]) | uint16(b[i+1])<<8
	i += 2
	result := int32(c)
	if utf16IsLowSurrogate(c) {
		return -1, len(b) // should never point at a low surrogate
	}
	if utf16IsHighSurrogate(c) {
		if i+2 > len(b) {
			return -1, len(b) // truncated string
		}
		low := uint16(b[i]) | uint16(b[i+1])<<8
		i += 2
		if !utf16IsLowSurrogate(low) {
			return -1, len(b)
		}
		result = result<<10 + int32(low) - ((0xD800 << 10) + 0xDC00 - 0x10000)
	}
	return result, i
}
