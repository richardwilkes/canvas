// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Swizzle is a compact encoding of a 4-channel swizzle. Each output channel is 4 bits selecting an input channel (r=0,
// g=1, b=2, a=3) or a constant (0=4, 1=5); output r is the low nibble.

package gpu

// Swizzle is the 16-bit swizzle key. Note that the Go zero value is "rrrr", not "rgba"; code that stores swizzles must
// initialize them explicitly (use SwizzleRGBA for the identity).
type Swizzle uint16

// The named swizzles used by the caps tables.
var (
	SwizzleRGBA = MakeSwizzle("rgba")
	SwizzleBGRA = MakeSwizzle("bgra")
	SwizzleRRRA = MakeSwizzle("rrra")
	SwizzleRGB1 = MakeSwizzle("rgb1")
)

func swizzleCToI(c byte) Swizzle {
	switch c {
	// r...a must map to 0...3 because apply methods use them as indices into the input color.
	case 'r':
		return 0
	case 'g':
		return 1
	case 'b':
		return 2
	case 'a':
		return 3
	case '0':
		return 4
	case '1':
		return 5
	default:
		panic("invalid swizzle char")
	}
}

func swizzleIToC(idx Swizzle) byte {
	switch idx {
	case 0:
		return 'r'
	case 1:
		return 'g'
	case 2:
		return 'b'
	case 3:
		return 'a'
	case 4:
		return '0'
	case 5:
		return '1'
	default:
		panic("invalid swizzle index")
	}
}

// MakeSwizzle builds a Swizzle from a 4-character specification using the characters r, g, b, a, 0 and 1.
func MakeSwizzle(s string) Swizzle {
	if len(s) != 4 {
		panic("swizzle spec must be 4 characters")
	}
	return swizzleCToI(s[0]) | swizzleCToI(s[1])<<4 | swizzleCToI(s[2])<<8 | swizzleCToI(s[3])<<12
}

// String returns the 4-character specification for the swizzle.
func (s Swizzle) String() string {
	return string([]byte{
		swizzleIToC(s & 0xF),
		swizzleIToC((s >> 4) & 0xF),
		swizzleIToC((s >> 8) & 0xF),
		swizzleIToC((s >> 12) & 0xF),
	})
}

// ApplyTo applies the swizzle to a float RGBA color: each output channel selects an input channel or the constants 0/1.
func (s Swizzle) ApplyTo(color [4]float32) [4]float32 {
	var out [4]float32
	k := s
	for i := 0; i < 4; i++ {
		switch idx := k & 0xF; idx {
		case 4:
			out[i] = 0
		case 5:
			out[i] = 1
		default:
			out[i] = color[idx]
		}
		k >>= 4
	}
	return out
}

// ConcatSwizzles returns the swizzle that applies a then b (b indexes into a's outputs; constants pass through).
func ConcatSwizzles(a, b Swizzle) Swizzle {
	var key Swizzle
	for i := uint(0); i < 4; i++ {
		idx := (b >> (4 * i)) & 0xF
		if idx != 4 && idx != 5 { // not the '0'/'1' constants
			// Get the index value stored in a at location idx.
			idx = (a >> (4 * idx)) & 0xF
		}
		key |= idx << (4 * i)
	}
	return key
}
