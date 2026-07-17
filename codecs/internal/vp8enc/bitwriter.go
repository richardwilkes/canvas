// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The VP8 boolean (arithmetic) coder, encoder side, implementing the coder described in RFC 6386 §7.

package vp8enc

// bitWriter is the boolean entropy coder. The zero value is NOT ready for use; call init.
type bitWriter struct {
	buf    []byte
	rng    int32 // range-1, in [127, 254]
	value  int32 // pending bits, not yet flushed
	run    int   // number of pending 0xff bytes (carry propagation)
	nbBits int   // number of complete bits in value beyond the 8 whole-byte bits; -8..0 after flush
}

func (bw *bitWriter) init() {
	bw.buf = bw.buf[:0]
	bw.rng = 255 - 1
	bw.value = 0
	bw.run = 0
	bw.nbBits = -8
}

// flush emits the topmost complete byte of value, handling carry propagation over pending 0xff bytes.
func (bw *bitWriter) flush() {
	s := 8 + bw.nbBits
	bits := bw.value >> uint(s)
	bw.value -= bits << uint(s)
	bw.nbBits -= 8
	if bits&0xff != 0xff {
		if bits&0x100 != 0 { // overflow: propagate carry over pending 0xff bytes
			if n := len(bw.buf); n > 0 {
				bw.buf[n-1]++
			}
		}
		if bw.run > 0 {
			value := byte(0xff)
			if bits&0x100 != 0 {
				value = 0x00
			}
			for ; bw.run > 0; bw.run-- {
				bw.buf = append(bw.buf, value)
			}
		}
		bw.buf = append(bw.buf, byte(bits&0xff))
	} else {
		bw.run++ // delay writing 0xff bytes, pending an eventual carry
	}
}

// putBit codes one boolean with probability prob (of the bit being 0), returning the bit.
func (bw *bitWriter) putBit(bit bool, prob uint8) bool {
	split := (bw.rng * int32(prob)) >> 8
	if bit {
		bw.value += split + 1
		bw.rng -= split + 1
	} else {
		bw.rng = split
	}
	if bw.rng < 127 { // emit 'shift' bits out and renormalize
		shift := int(normTable[bw.rng])
		bw.rng = int32(rangeTable[bw.rng])
		bw.value <<= uint(shift)
		bw.nbBits += shift
		if bw.nbBits > 0 {
			bw.flush()
		}
	}
	return bit
}

// putBitUniform codes one boolean with the uniform probability 128.
func (bw *bitWriter) putBitUniform(bit bool) bool {
	split := bw.rng >> 1
	if bit {
		bw.value += split + 1
		bw.rng -= split + 1
	} else {
		bw.rng = split
	}
	if bw.rng < 127 {
		bw.rng = int32(rangeTable[bw.rng])
		bw.value <<= 1
		bw.nbBits++
		if bw.nbBits > 0 {
			bw.flush()
		}
	}
	return bit
}

// putBits codes nbBits raw bits of value (MSB first) with uniform probability.
func (bw *bitWriter) putBits(value uint32, nbBits int) {
	for mask := uint32(1) << uint(nbBits-1); mask != 0; mask >>= 1 {
		bw.putBitUniform(value&mask != 0)
	}
}

// putSignedBits codes a flag bit and, when non-zero, the magnitude and sign of value.
func (bw *bitWriter) putSignedBits(value, nbBits int) {
	if !bw.putBitUniform(value != 0) {
		return
	}
	if value < 0 {
		bw.putBits(uint32(-value)<<1|1, nbBits+1)
	} else {
		bw.putBits(uint32(value)<<1, nbBits+1)
	}
}

// finish pads the stream with zeros and flushes the remaining bytes, returning the buffer.
func (bw *bitWriter) finish() []byte {
	bw.putBits(0, 9-bw.nbBits)
	bw.nbBits = 0 // pad with zeroes
	bw.flush()
	return bw.buf
}
