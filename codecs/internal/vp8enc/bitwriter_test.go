// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package vp8enc

import (
	"math/rand"
	"testing"
)

// boolDecoder is an in-test boolean decoder following RFC 6386 §7.3's reference implementation, used to verify the
// encoder side round-trips.
type boolDecoder struct {
	input    []byte
	pos      int
	rng      uint32
	value    uint32
	bitCount int
}

func newBoolDecoder(b []byte) *boolDecoder {
	d := &boolDecoder{input: b, rng: 255}
	d.value = uint32(d.next())<<8 | uint32(d.next())
	return d
}

func (d *boolDecoder) next() uint8 {
	if d.pos < len(d.input) {
		v := d.input[d.pos]
		d.pos++
		return v
	}
	return 0
}

func (d *boolDecoder) readBool(prob uint8) bool {
	split := 1 + ((d.rng-1)*uint32(prob))>>8
	bigSplit := split << 8
	var ret bool
	if d.value >= bigSplit {
		ret = true
		d.rng -= split
		d.value -= bigSplit
	} else {
		d.rng = split
	}
	for d.rng < 128 {
		d.value <<= 1
		d.rng <<= 1
		d.bitCount++
		if d.bitCount == 8 {
			d.bitCount = 0
			d.value |= uint32(d.next())
		}
	}
	return ret
}

func (d *boolDecoder) readUint(nbBits int) uint32 {
	v := uint32(0)
	for i := 0; i < nbBits; i++ {
		v <<= 1
		if d.readBool(128) {
			v |= 1
		}
	}
	return v
}

func TestBitWriterRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for trial := 0; trial < 100; trial++ {
		n := 1 + rng.Intn(4000)
		bits := make([]bool, n)
		probs := make([]uint8, n)
		for i := range bits {
			// Mix extreme and mid probabilities, including the carry-propagation-heavy cases.
			switch rng.Intn(4) {
			case 0:
				probs[i] = 1
			case 1:
				probs[i] = 255
			default:
				probs[i] = uint8(1 + rng.Intn(255))
			}
			bits[i] = rng.Intn(100) < 60
		}
		var bw bitWriter
		bw.init()
		for i := range bits {
			bw.putBit(bits[i], probs[i])
		}
		buf := bw.finish()

		dec := newBoolDecoder(buf)
		for i := range bits {
			if got := dec.readBool(probs[i]); got != bits[i] {
				t.Fatalf("trial %d bit %d: got %v want %v", trial, i, got, bits[i])
			}
		}
	}
}

func TestBitWriterUniformAndValues(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	for trial := 0; trial < 50; trial++ {
		type item struct {
			kind  int // 0 = uniform bit, 1 = value, 2 = signed value
			bit   bool
			value int
			nbits int
		}
		n := 1 + rng.Intn(500)
		items := make([]item, n)
		for i := range items {
			it := &items[i]
			it.kind = rng.Intn(3)
			switch it.kind {
			case 0:
				it.bit = rng.Intn(2) == 1
			case 1:
				it.nbits = 1 + rng.Intn(16)
				it.value = rng.Intn(1 << uint(it.nbits))
			case 2:
				it.nbits = 1 + rng.Intn(8)
				it.value = rng.Intn(1<<uint(it.nbits)) - (1 << uint(it.nbits-1))
			}
		}
		var bw bitWriter
		bw.init()
		for i := range items {
			it := &items[i]
			switch it.kind {
			case 0:
				bw.putBitUniform(it.bit)
			case 1:
				bw.putBits(uint32(it.value), it.nbits)
			case 2:
				bw.putSignedBits(it.value, it.nbits)
			}
		}
		buf := bw.finish()

		dec := newBoolDecoder(buf)
		for i := range items {
			it := &items[i]
			switch it.kind {
			case 0:
				if got := dec.readBool(128); got != it.bit {
					t.Fatalf("trial %d item %d: uniform bit got %v want %v", trial, i, got, it.bit)
				}
			case 1:
				if got := dec.readUint(it.nbits); got != uint32(it.value) {
					t.Fatalf("trial %d item %d: value got %d want %d", trial, i, got, it.value)
				}
			case 2:
				want := it.value
				got := 0
				if dec.readBool(128) {
					got = int(dec.readUint(it.nbits))
					if dec.readBool(128) {
						got = -got
					}
				}
				if got != want {
					t.Fatalf("trial %d item %d: signed value got %d want %d", trial, i, got, want)
				}
			}
		}
	}
}
