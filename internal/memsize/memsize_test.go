// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package memsize

import (
	"testing"
	"unsafe"
)

// pair stands in for the key/element pair a map slot holds: a pointer and a narrower int, so the slot is padded out
// past the sum of its fields.
type pair struct {
	val *int
	key uint32
}

// wide stands in for a key the runtime is unwilling to store inline in a slot.
type wide struct {
	bytes [maxInlineSlotBytes + 1]byte
}

func TestOfMatchesSizeof(t *testing.T) {
	if got, want := Of[pair](), int(unsafe.Sizeof(pair{})); got != want {
		t.Errorf("Of[pair]() = %d, want %d", got, want)
	}
	// A pointer-and-narrow-int pair is padded out past its last field to the pointer's alignment. Of has to report the
	// padded size, since that is what an array or map slot of them actually spends per element.
	if offset := unsafe.Offsetof(pair{}.val); offset != 0 {
		t.Fatalf("pair.val is at offset %d, want 0", offset)
	}
	fieldsEnd := int(unsafe.Offsetof(pair{}.key) + unsafe.Sizeof(pair{}.key))
	if got := Of[pair](); got <= fieldsEnd {
		t.Errorf("Of[pair]() = %d, but the fields already end at %d (padding is not being counted)", got, fieldsEnd)
	}
}

// The slot a map spends on an entry is laid out the way a struct holding the key and element would be, so MapEntry has
// to arrive at the same padded size the compiler does — never merely the sum of the two field sizes.
func TestMapEntryCountsSlotPadding(t *testing.T) {
	slot := int(unsafe.Sizeof(pair{}))
	got := MapEntry[uint32, *int]()
	if got <= slot {
		t.Errorf("MapEntry[uint32, *int]() = %d, want more than the %d-byte slot (control byte and load factor are "+
			"not being counted)", got, slot)
	}
	// The slot plus its control byte, scaled for the headroom the load factor reserves.
	if want := (slot + 1) * slotsPerGroup / maxFullSlotsPerGroup; got != want {
		t.Errorf("MapEntry[uint32, *int]() = %d, want %d", got, want)
	}
}

// A key or element too wide to live in the slot is stored indirectly, so the entry costs a pointer's worth of slot
// plus the separate allocation. Charging only the slot would under-count it by the whole value.
func TestMapEntryCountsIndirectStorage(t *testing.T) {
	var w wide
	if len(w.bytes) <= maxInlineSlotBytes {
		t.Fatalf("wide is %d bytes, which the runtime would still store inline", len(w.bytes))
	}
	got := MapEntry[wide, *int]()
	if got <= Of[wide]() {
		t.Errorf("MapEntry[wide, *int]() = %d, want more than the %d-byte key it stores out of line", got, Of[wide]())
	}
	// The same map with an inline key of pointer width, plus the out-of-line key itself.
	if want := MapEntry[uintptr, *int]() + Of[wide](); got != want {
		t.Errorf("MapEntry[wide, *int]() = %d, want %d", got, want)
	}
}

// SliceElem answers for storage, not for the element, so it must exceed the element and must never round down to it —
// a one-byte element still carries slack, and reporting 1 would make an append-grown []byte look free of it.
func TestSliceElemChargesGrowthSlack(t *testing.T) {
	if got := SliceElem[byte](); got <= Of[byte]() {
		t.Errorf("SliceElem[byte]() = %d, want more than the %d-byte element", got, Of[byte]())
	}
	if got, want := SliceElem[*int](), Of[*int]()*sliceSlackNum/sliceSlackDen; got != want {
		t.Errorf("SliceElem[*int]() = %d, want %d", got, want)
	}
	// The slack is a fraction of the element, so it must stay under the 2x worst case that would halve a cache's
	// effective capacity.
	if got, limit := SliceElem[*int](), 2*Of[*int](); got >= limit {
		t.Errorf("SliceElem[*int]() = %d, want less than the %d worst case", got, limit)
	}
}

func TestMapIsNonZero(t *testing.T) {
	if got := Map(); got <= 0 {
		t.Errorf("Map() = %d, want a positive header cost", got)
	}
}

func TestRoundUp(t *testing.T) {
	for _, tc := range []struct{ v, align, want int }{
		{v: 0, align: 8, want: 0},
		{v: 1, align: 8, want: 8},
		{v: 8, align: 8, want: 8},
		{v: 9, align: 8, want: 16},
		{v: 4, align: 1, want: 4},
	} {
		if got := roundUp(tc.v, tc.align); got != tc.want {
			t.Errorf("roundUp(%d, %d) = %d, want %d", tc.v, tc.align, got, tc.want)
		}
	}
}
