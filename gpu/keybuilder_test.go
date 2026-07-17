// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gpu

import "testing"

func equalU32(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestKeyBuilderResetReuseByteIdentical locks the invariant the reused program-cache key builder depends on: a builder
// Reset over fresh storage produces byte-identical key words to a fresh NewKeyBuilder, and Reset clears any in-progress
// bit state so nothing leaks between keys.
func TestKeyBuilderResetReuseByteIdentical(t *testing.T) {
	build := func(b *KeyBuilder) {
		b.Add32(0xdeadbeef, "a")
		b.AddBool(true, "b")
		b.AddBits(5, 17, "c")
		b.Add32(0x01020304, "d")
		b.Flush()
	}

	var fresh []uint32
	build(NewKeyBuilder(&fresh))

	// A reused builder, reset over new storage, must produce the identical key words.
	var reused []uint32
	var rb KeyBuilder
	rb.Reset(&reused)
	build(&rb)
	if !equalU32(fresh, reused) {
		t.Fatalf("reset+reuse key = %v, want %v", reused, fresh)
	}

	// Reusing the builder for a shorter key after a longer one must match a fresh short key.
	var short, shortFresh []uint32
	rb.Reset(&short)
	rb.AddBits(3, 5, "x")
	rb.Flush()
	sfb := NewKeyBuilder(&shortFresh)
	sfb.AddBits(3, 5, "x")
	sfb.Flush()
	if !equalU32(short, shortFresh) {
		t.Fatalf("short reuse key = %v, want %v", short, shortFresh)
	}

	// Reset must clear a dirty in-progress bit state (bits added but never flushed): the leftover bits must not corrupt
	// the next key. This has teeth — without clearing curValue/bitsUsed the leftover 0x3ff would fold into the
	// following key.
	var dirty []uint32
	rb.Reset(&dirty)
	rb.AddBits(10, 0x3ff, "leftover")
	var clean, cleanFresh []uint32
	rb.Reset(&clean)
	rb.Add32(0xcafebabe, "e")
	rb.Flush()
	cfb := NewKeyBuilder(&cleanFresh)
	cfb.Add32(0xcafebabe, "e")
	cfb.Flush()
	if !equalU32(clean, cleanFresh) {
		t.Fatalf("dirty-then-reset key = %v, want %v", clean, cleanFresh)
	}
}
