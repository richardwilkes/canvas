// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package memsize derives the per-item byte costs the byte-budgeted caches charge, from the layout of the types those
// caches actually retain. A hand-picked constant is a guess that stops being true the moment a struct gains a field:
// it then either lets the cache overrun its budget silently or purges long before it has to. Everything here is
// computed once at init from unsafe.Sizeof/Alignof, so a budget tracks the types it is accounting for.
//
// The map and slice models encode the runtime's storage strategy rather than any one map or slice, so they remain
// approximations — the point is that they are approximations with a stated derivation instead of numbers somebody
// liked the look of.

package memsize

import "unsafe"

// ptrBytes is the width of a pointer on this platform.
var ptrBytes = int(unsafe.Sizeof(uintptr(0)))

const (
	// maxInlineSlotBytes is the widest key or element the runtime's map stores inline in a slot (its maxKeyOrElemSize).
	// Anything wider is stored indirectly: the slot holds a pointer to a separate allocation.
	maxInlineSlotBytes = 128
	// slotsPerGroup is how many slots the runtime's map packs into one group, each with a one-byte control word.
	slotsPerGroup = 8
	// maxFullSlotsPerGroup is how many of a group's slots may be occupied before the table grows. The reserved
	// remainder is real retained memory, so a live entry's cost is scaled by slotsPerGroup/maxFullSlotsPerGroup.
	maxFullSlotsPerGroup = 7
	// mapHeaderWords is the width, in pointer-sized words, of the header every make(map[K]V) allocates before a single
	// entry goes in (the runtime's maps.Map: a use count, a seed, a directory pointer and length, a word of packed
	// flags, and a clear sequence).
	mapHeaderWords = 6
	// sliceSlackNum/sliceSlackDen express how much storage a slice grown with append actually holds per element it is
	// carrying. append doubles the capacity of a small slice, which every slice these caches build is, so a slice of n
	// elements has room for somewhere between n and 2n — and the unused remainder is retained exactly as firmly as the
	// used part. A length lands anywhere in that doubled span with roughly equal likelihood, putting the middle of the
	// range at 3/2. These budgets sum over thousands of items, so the middle is the estimator that makes the total come
	// out right; charging the 2x worst case for every one of them would leave a cache holding half the data it was
	// configured to hold.
	sliceSlackNum = 3
	sliceSlackDen = 2
)

// Of returns the size of a value of type T, padding included.
func Of[T any]() int {
	var zero T
	return int(unsafe.Sizeof(zero))
}

// Map returns what an empty map costs: the header, which is allocated whether or not anything is ever put in it.
func Map() int { return mapHeaderWords * ptrBytes }

// MapEntry returns what one live entry of a map[K]V costs, its share of the table included. The runtime stores entries
// in groups of slots, each slot holding the key and the element laid out the way a struct holding the pair would be,
// and each carrying one control byte; the table grows once a group would fill past its load factor. So an entry's cost
// is its padded slot plus its control byte, scaled up for the headroom the load factor reserves.
func MapEntry[K comparable, V any]() int {
	keyBytes, keyAlign, keyIndirect := slotShape[K]()
	elemBytes, elemAlign, elemIndirect := slotShape[V]()
	slot := roundUp(roundUp(keyBytes, elemAlign)+elemBytes, max(keyAlign, elemAlign))
	return (slot+1)*slotsPerGroup/maxFullSlotsPerGroup + keyIndirect + elemIndirect
}

// SliceElem returns what one element of a []T costs once the slice has been grown with append, which is more than the
// element itself — see sliceSlackNum. The result is rounded up, so even a one-byte element is charged for some slack.
// Use Of instead for a slice allocated at its final length in one shot, which carries none.
func SliceElem[T any]() int {
	return (Of[T]()*sliceSlackNum + sliceSlackDen - 1) / sliceSlackDen
}

// slotShape returns how T occupies a map slot: the bytes and alignment the slot spends on it, plus the bytes held
// outside the slot when T is too wide to be stored inline.
func slotShape[T any]() (bytes, align, indirect int) {
	var zero T
	bytes, align = int(unsafe.Sizeof(zero)), int(unsafe.Alignof(zero))
	if bytes > maxInlineSlotBytes {
		return ptrBytes, ptrBytes, bytes
	}
	return bytes, align, 0
}

// roundUp rounds v up to the next multiple of align, which is always a power of two.
func roundUp(v, align int) int { return (v + align - 1) &^ (align - 1) }
