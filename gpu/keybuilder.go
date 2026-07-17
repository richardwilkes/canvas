// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// KeyBuilder and StringKeyBuilder: the bit-packing builder used to assemble program keys. Program descriptors are
// hashed and compared as raw words, so the packing scheme must stay stable.

package gpu

// KeyBuilder packs fields of varying bit widths into a []uint32 key. Create one with NewKeyBuilder over the destination
// slice, call the add methods, and Flush before using the key.
type KeyBuilder struct {
	data     *[]uint32
	describe func(label string, v uint32)
	comment  func(comment string)
	curValue uint32
	bitsUsed uint32
}

// NewKeyBuilder returns a KeyBuilder appending to *data.
func NewKeyBuilder(data *[]uint32) *KeyBuilder {
	return &KeyBuilder{data: data}
}

// Reset reinitializes the builder to append to *data, clearing any in-progress bit state, so a single builder can be
// reused across keys. A hot-path caller keeps one builder (its methods pass it to interface calls, which the compiler
// treats as escaping, so a fresh per-key builder would heap- allocate) and resets it per key to avoid that per-call
// allocation. The describe/comment hooks are left untouched.
func (b *KeyBuilder) Reset(data *[]uint32) {
	b.data = data
	b.curValue = 0
	b.bitsUsed = 0
}

// AddBits packs the low numBits bits of val into the key, labeling the field for description purposes.
func (b *KeyBuilder) AddBits(numBits, val uint32, label string) {
	if numBits == 0 || numBits > 32 {
		panic("numBits out of range")
	}
	if numBits != 32 && val >= 1<<numBits {
		panic("value does not fit in numBits")
	}
	if b.describe != nil {
		b.describe(label, val)
	}

	b.curValue |= val << b.bitsUsed
	b.bitsUsed += numBits

	if b.bitsUsed >= 32 {
		// Overflow, start a new working value.
		*b.data = append(*b.data, b.curValue)
		excess := b.bitsUsed - 32
		if excess != 0 {
			b.curValue = val >> (numBits - excess)
		} else {
			b.curValue = 0
		}
		b.bitsUsed = excess
	}
}

// AddBytes appends each byte of data to the key as its own 8-bit field.
func (b *KeyBuilder) AddBytes(data []byte, label string) {
	for _, v := range data {
		b.AddBits(8, uint32(v), label)
	}
}

// AddBool appends a single bit to the key.
func (b *KeyBuilder) AddBool(v bool, label string) {
	var bit uint32
	if v {
		bit = 1
	}
	b.AddBits(1, bit, label)
}

// Add32 appends a full 32-bit word to the key.
func (b *KeyBuilder) Add32(v uint32, label string) {
	b.AddBits(32, v, label)
}

// AppendComment records a free-text comment in the description (a no-op unless describing).
func (b *KeyBuilder) AppendComment(comment string) {
	if b.comment != nil {
		b.comment(comment)
	}
}

// Flush introduces a word boundary in the key. Must be called before using the key with any cache.
func (b *KeyBuilder) Flush() {
	if b.bitsUsed != 0 {
		*b.data = append(*b.data, b.curValue)
		b.curValue = 0
		b.bitsUsed = 0
	}
}

// StringKeyBuilder is a KeyBuilder that additionally produces a human-readable description of every field added, for
// program-descriptor debugging and parity-harness dumps.
type StringKeyBuilder struct {
	KeyBuilder
	description []byte
}

// NewStringKeyBuilder returns a describing KeyBuilder appending to *data.
func NewStringKeyBuilder(data *[]uint32) *StringKeyBuilder {
	b := &StringKeyBuilder{KeyBuilder: KeyBuilder{data: data}}
	b.describe = func(label string, v uint32) {
		b.description = append(b.description, label...)
		b.description = append(b.description, ": "...)
		b.description = appendUint(b.description, v)
		b.description = append(b.description, '\n')
	}
	b.comment = func(comment string) {
		b.description = append(b.description, comment...)
		b.description = append(b.description, '\n')
	}
	return b
}

// Description returns the accumulated human-readable field description.
func (b *StringKeyBuilder) Description() string { return string(b.description) }

func appendUint(dst []byte, v uint32) []byte {
	if v >= 10 {
		dst = appendUint(dst, v/10)
	}
	return append(dst, byte('0'+v%10))
}
