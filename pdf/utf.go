// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A strict UTF-8 decoding primitive that writeTextString uses to validate and iterate a PDF text string. Implemented
// locally so the pdf package does not depend on the font stack; the semantics match font.nextUTF8 exactly (strict
// validation, -1 on any invalid byte with the cursor jumping to the end).

package pdf

// utf8ByteType classifies the leading byte c of a UTF-8 sequence: -1 invalid, 0 continuation, else the sequence length
// (1..4).
func utf8ByteType(c byte) int {
	switch {
	case c < 0x80:
		return 1
	case c < 0xC0:
		return 0
	case c >= 0xF5 || (c&0xFE) == 0xC0: // "octet values c0, c1, f5 to ff never appear"
		return -1
	default:
		// Arithmetic right shift of the int32 bit pattern for (0xe5 << 24), a lookup packed into one word.
		const e5 = int32(-0x1B000000)
		return int((e5>>((uint(c)>>4)<<1))&3 + 1)
	}
}

func utf8ByteIsContinuation(c byte) bool { return utf8ByteType(c) == 0 }

// nextUTF8 decodes the unichar at i, returning (unichar, next index). On invalid input it returns (-1, len(b)): the
// cursor jumps to the end.
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
		mask := uint32(0xFFFFFFC0) // ^0x3F
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
