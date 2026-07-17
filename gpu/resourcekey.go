// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ResourceKey, ScratchKey, and UniqueKey: a key is a []uint32 whose first two words are metadata (word 0 the hash of
// everything after it, word 1 the domain in the low 16 bits and the total byte size in the high 16), followed by the
// caller's data words. Equality compares the whole array; hashing is wyhash-based. The invalid state is a nil slice
// (domain 0 == invalidKeyDomain).

package gpu

import "sync/atomic"

const (
	keyMetaDataCount = 2 // hash word + domain-and-size word
	invalidKeyDomain = 0
)

// resourceKey is the shared representation behind ScratchKey and UniqueKey.
type resourceKey struct {
	str string   // memoized mapKey() result; "" until first computed (and cleared on Reset/rebuild)
	key []uint32 // nil when invalid
}

// IsValid reports whether the key holds a real (non-nil) key.
func (k *resourceKey) IsValid() bool { return k.key != nil }

// Hash returns the key's precomputed hash word.
func (k *resourceKey) Hash() uint32 { return k.key[0] }

// Reset returns the key to the invalid state.
func (k *resourceKey) Reset() {
	k.key = nil
	k.str = ""
}

func (k *resourceKey) domain() uint32 {
	if k.key == nil {
		return invalidKeyDomain
	}
	return k.key[1] & 0xffff
}

// Size returns the total key size in bytes including metadata.
func (k *resourceKey) Size() int { return int(k.key[1] >> 16) }

// data returns the caller data words (excluding metadata).
func (k *resourceKey) data() []uint32 { return k.key[keyMetaDataCount:] }

// equalTo reports whether two keys hold identical words.
func (k *resourceKey) equalTo(that *resourceKey) bool {
	if len(k.key) != len(that.key) {
		return false
	}
	for i, v := range k.key {
		if that.key[i] != v {
			return false
		}
	}
	return true
}

// mapKey returns the key's full contents as a string for use as a Go map key, memoizing the result on the key so
// repeated uses share one allocation. Only valid keys may be used. The metadata words participate the same way they do
// in the resource-cache lookups (two keys compare equal iff hash, domain, size, and data all match).
//
// The memo exists so the resource-cache scratch keying avoids per-frame allocation churn: a resource's own scratch key
// is built once at registration and reused on every per-frame scratch insert/remove, and reused lookup keys cache after
// their first use. Because the cached string is byte-identical to a freshly built one, the scratch/unique map key set,
// eviction, and reuse order are unchanged — a map assignment or delete with an already-allocated string variable does
// not allocate a new string, which is what removes the per-frame churn.
func (k *resourceKey) mapKey() string {
	if k.str == "" && k.key != nil {
		k.str = mapKeyBytes(k.key)
	}
	return k.str
}

// mapKeyBytes packs the key words little-endian into a string for use as a hash-table byte comparison. A valid key
// always has at least the two metadata words, so the result is never empty (which lets mapKey use "" as the
// not-yet-computed sentinel).
func mapKeyBytes(key []uint32) string {
	buf := make([]byte, 0, len(key)*4)
	for _, v := range key {
		buf = append(buf, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
	}
	return string(buf)
}

// ResourceKeyBuilder initializes a key's storage. Callers assign the data words via Slice or indexed writes and then call
// Finish.
type ResourceKeyBuilder struct {
	target *resourceKey
}

func makeKeyBuilder(key *resourceKey, domain uint32, data32Count int) ResourceKeyBuilder {
	words := make([]uint32, keyMetaDataCount+data32Count)
	size := uint32(len(words) * 4)
	words[1] = domain | size<<16
	key.key = words
	key.str = "" // drop any memoized string from a reused key struct; recomputed lazily after Finish
	return ResourceKeyBuilder{target: key}
}

// Slice returns the caller-writable data words.
func (b ResourceKeyBuilder) Slice() []uint32 { return b.target.key[keyMetaDataCount:] }

// Finish computes and stores the key's hash word from its data words.
func (b ResourceKeyBuilder) Finish() {
	b.target.key[0] = hash32U32s(b.target.key[1:])
}

// ScratchKeyResourceType uniquely identifies the type of resource cached as scratch (the scratch key's domain).
type ScratchKeyResourceType uint32

var nextScratchResourceType atomic.Int32

// GenerateScratchKeyResourceType returns a fresh, process-unique ScratchKeyResourceType.
func GenerateScratchKeyResourceType() ScratchKeyResourceType {
	t := nextScratchResourceType.Add(1)
	if t > 0xffff {
		panic("too many scratch key resource types")
	}
	return ScratchKeyResourceType(t)
}

// ScratchKey is a key for interchangeable resources whose allocations are cached but not their contents. Multiple
// resources may share one scratch key.
type ScratchKey struct {
	resourceKey
}

// ResourceType returns the scratch key's resource type (its domain).
func (k *ScratchKey) ResourceType() ScratchKeyResourceType {
	return ScratchKeyResourceType(k.domain())
}

// Equal reports whether two scratch keys hold identical words.
func (k *ScratchKey) Equal(that *ScratchKey) bool { return k.equalTo(&that.resourceKey) }

// MapKey returns the key's full contents as a string for use as a Go map key (used by, e.g., the resource allocator's
// free pool). Only valid keys may be used.
func (k *ScratchKey) MapKey() string { return k.mapKey() }

// ScratchKeyBuilder returns a builder for initializing key with the given resource type and data word count.
func ScratchKeyBuilder(key *ScratchKey, resourceType ScratchKeyResourceType, data32Count int) ResourceKeyBuilder {
	return makeKeyBuilder(&key.resourceKey, uint32(resourceType), data32Count)
}

// UniqueKeyDomain namespaces unique keys per use case.
type UniqueKeyDomain uint32

var nextUniqueKeyDomain atomic.Int32

// GenerateUniqueKeyDomain returns a fresh, process-unique UniqueKeyDomain.
func GenerateUniqueKeyDomain() UniqueKeyDomain {
	d := nextUniqueKeyDomain.Add(1)
	if d > 0xffff {
		panic("too many unique key domains")
	}
	return UniqueKeyDomain(d)
}

// UniqueKey identifies a resource that at most one resource may hold at a time, and a resource holds at most one.
// Unique keys preempt scratch keys.
type UniqueKey struct {
	resourceKey
	tag        string
	customData []byte // arbitrary caller-attached data
}

// Equal reports whether two unique keys hold identical words (custom data and tag do not participate).
func (k *UniqueKey) Equal(that *UniqueKey) bool { return k.equalTo(&that.resourceKey) }

// SetCustomData attaches arbitrary caller data to the key.
func (k *UniqueKey) SetCustomData(data []byte) { k.customData = data }

// CustomData returns the key's attached custom data, if any.
func (k *UniqueKey) CustomData() []byte { return k.customData }

// Tag returns the key's debug tag.
func (k *UniqueKey) Tag() string { return k.tag }

// Reset returns the key to the invalid state, dropping custom data and tag.
func (k *UniqueKey) Reset() {
	k.resourceKey.Reset()
	k.customData = nil
	k.tag = ""
}

// Data returns the caller data words.
func (k *UniqueKey) Data() []uint32 { return k.data() }

// MapKey returns the key's full contents as a string for use as a Go map key (used by, e.g., a uniquely-keyed proxy
// map). Only valid keys may be used.
func (k *UniqueKey) MapKey() string { return k.mapKey() }

// UniqueKeyBuilder returns a builder for initializing key with the given domain, data word count, and debug tag.
func UniqueKeyBuilder(key *UniqueKey, domain UniqueKeyDomain, data32Count int, tag string) ResourceKeyBuilder {
	b := makeKeyBuilder(&key.resourceKey, uint32(domain), data32Count)
	key.tag = tag
	return b
}

// UniqueKeyBuilderWithInnerKey initializes key so it wraps innerKey, adding extraData32Count leading data words
// (indexable as usual; the inner key is appended after them).
func UniqueKeyBuilderWithInnerKey(key, innerKey *UniqueKey, domain UniqueKeyDomain, extraData32Count int, tag string) ResourceKeyBuilder {
	innerData := innerKey.data()
	b := makeKeyBuilder(&key.resourceKey, uint32(domain), extraData32Count+len(innerData)+1)
	s := b.Slice()
	s[extraData32Count] = innerKey.domain()
	copy(s[extraData32Count+1:], innerData)
	key.tag = tag
	return b
}
