// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

import "math/bits"

// Point is a position in 26.6 fixed point pixels, y up.
type Point struct {
	X, Y int32
}

// The 26.6 helpers. All arithmetic wraps, as the reference does.

func floor26(x int32) int32 { return x &^ 63 }

func round26(x int32) int32 { return floor26(x + 32) }

func ceil26(x int32) int32 { return floor26(x + 63) }

func roundPad(x, n int32) int32 { return (x + n/2) &^ (n - 1) }

func abs32(x int32) int32 {
	if x < 0 {
		return -x
	}
	return x
}

// mulFixed multiplies two 16.16 values, rounding half away from zero.
func mulFixed(a, b int32) int32 {
	ab := int64(a) * int64(b)
	if ab < 0 {
		return int32((ab + 0x8000 - 1) >> 16)
	}
	return int32((ab + 0x8000) >> 16)
}

// divFixed divides two 16.16 values, rounding to nearest.
func divFixed(a, b int32) int32 {
	negative := (a < 0) != (b < 0)
	au := uint64(uint32(abs32(a)))
	bu := uint64(uint32(abs32(b)))
	q := uint32(0x7FFFFFFF)
	if bu != 0 {
		q = uint32(((au << 16) + (bu >> 1)) / bu)
	}
	if negative {
		return -int32(q)
	}
	return int32(q)
}

func absU64(v int32) (u uint64, negative bool) {
	u = uint64(int64(v))
	if v < 0 {
		return -u, true
	}
	return u, false
}

// mulDiv computes a*b/c with rounding.
func mulDiv(a, b, c int32) int32 {
	au, an := absU64(a)
	bu, bn := absU64(b)
	cu, cn := absU64(c)
	result := uint64(0x7FFFFFFF)
	if cu > 0 {
		result = (au*bu + (cu >> 1)) / cu
	}
	if (an != bn) != cn {
		return -int32(result)
	}
	return int32(result)
}

// mulDivNoRound computes a*b/c, truncating.
func mulDivNoRound(a, b, c int32) int32 {
	au, an := absU64(a)
	bu, bn := absU64(b)
	cu, cn := absU64(c)
	result := uint64(0x7FFFFFFF)
	if cu > 0 {
		result = au * bu / cu
	}
	if (an != bn) != cn {
		return -int32(result)
	}
	return int32(result)
}

// mul14 multiplies by a 2.14 value.
func mul14(a, b int32) int32 {
	v := int64(a) * int64(b)
	v += 0x2000 + (v >> 63)
	return int32(v >> 14)
}

func dot14(ax, ay, bx, by int32) int32 {
	v := int64(ax)*int64(bx) + int64(ay)*int64(by)
	v += 0x2000 + (v >> 63)
	return int32(v >> 14)
}

// normalize14 normalizes a vector to 2.14 unit length.
func normalize14(x, y int32) Point {
	sx, sy := int32(1), int32(1)
	ux, uy := uint32(x), uint32(y)
	var result Point
	if x < 0 {
		ux = -ux
		sx = -sx
	}
	if y < 0 {
		uy = -uy
		sy = -sy
	}
	if ux == 0 {
		result.X = x / 4
		if uy > 0 {
			result.Y = sy * 0x10000 / 4
		}
		return result
	}
	if uy == 0 {
		result.Y = y / 4
		if ux > 0 {
			result.X = sx * 0x10000 / 4
		}
		return result
	}
	estimate := func() uint32 {
		if ux > uy {
			return ux + (uy >> 1)
		}
		return uy + (ux >> 1)
	}
	length := estimate()
	shift := int32(bits.LeadingZeros32(length))
	adjust := int32(15)
	if length >= uint32(0xAAAAAAAA)>>uint(shift) {
		adjust++
	}
	shift -= adjust
	if shift > 0 {
		ux <<= uint(shift)
		uy <<= uint(shift)
		length = estimate()
	} else {
		ux >>= uint(-shift)
		uy >>= uint(-shift)
		length >>= uint(-shift)
	}
	b := int32(0x10000) - int32(length)
	xi, yi := int32(ux), int32(uy)
	var u, v uint32
	for {
		u = uint32(xi + ((xi * b) >> 16))
		v = uint32(yi + ((yi * b) >> 16))
		z := -int32(u*u+v*v) / 0x200
		z = z * ((0x10000 + b) >> 8) / 0x10000
		b += z
		if z <= 0 {
			break
		}
	}
	return Point{X: int32(u) * sx / 4, Y: int32(v) * sy / 4}
}
