// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hash32/Hash64 implement wyhash, from https://github.com/wangyi-fudan/wyhash, with the reference implementation's
// default secret. Resource keys hash with it here.

package gpu

import (
	"encoding/binary"
	"math/bits"
)

// The default wyhash secret parameters (named _wyp in the reference implementation).
const (
	wyp0 = 0xa0761d6478bd642f
	wyp1 = 0xe7037ed1a0b428db
	wyp2 = 0x8ebc6af09c88c6e3
	wyp3 = 0x589965cc75374cc3
)

// wymix is the multiply-and-xor mix function (aka MUM).
func wymix(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return lo ^ hi
}

func wyr8(p []byte) uint64 { return binary.LittleEndian.Uint64(p) }

func wyr4(p []byte) uint64 { return uint64(binary.LittleEndian.Uint32(p)) }

func wyr3(p []byte, k int) uint64 {
	return uint64(p[0])<<16 | uint64(p[k>>1])<<8 | uint64(p[k-1])
}

func wyhash(p []byte, seed uint64) uint64 {
	length := len(p)
	seed ^= wymix(seed^wyp0, wyp1)
	var a, b uint64
	switch {
	case length <= 16:
		switch {
		case length >= 4:
			a = wyr4(p)<<32 | wyr4(p[(length>>3)<<2:])
			b = wyr4(p[length-4:])<<32 | wyr4(p[length-4-((length>>3)<<2):])
		case length > 0:
			a = wyr3(p, length)
		default:
			a, b = 0, 0
		}
	default:
		i := length
		off := 0
		if i > 48 {
			see1, see2 := seed, seed
			for {
				seed = wymix(wyr8(p[off:])^wyp1, wyr8(p[off+8:])^seed)
				see1 = wymix(wyr8(p[off+16:])^wyp2, wyr8(p[off+24:])^see1)
				see2 = wymix(wyr8(p[off+32:])^wyp3, wyr8(p[off+40:])^see2)
				off += 48
				i -= 48
				if i <= 48 {
					break
				}
			}
			seed ^= see1 ^ see2
		}
		for i > 16 {
			seed = wymix(wyr8(p[off:])^wyp1, wyr8(p[off+8:])^seed)
			i -= 16
			off += 16
		}
		// The final two reads take the last 16 bytes of the whole input (off+i == len(p)).
		a = wyr8(p[off+i-16:])
		b = wyr8(p[off+i-8:])
	}
	a ^= wyp1
	b ^= seed
	hi, lo := bits.Mul64(a, b)
	return wymix(lo^wyp0^uint64(length), hi^wyp1)
}

// Hash32 returns a 64-bit wyhash truncated to 32 bits, seed 0.
func Hash32(data []byte) uint32 { return uint32(wyhash(data, 0)) }

// Hash64 returns a 64-bit wyhash, seed 0.
func Hash64(data []byte) uint64 { return wyhash(data, 0) }

// hash32U32s hashes a []uint32 as its little-endian byte representation, matching the resource key hash's assumption
// that word data can be hashed directly (this codebase targets little-endian platforms).
func hash32U32s(data []uint32) uint32 {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], v)
	}
	return Hash32(buf)
}
