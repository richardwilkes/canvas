// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Differential test for the wyhash implementation: the expected values were produced by a reference wyhash driver
// hashing buf[i] = i*131+7 at each listed length, verifying this implementation is bit-compatible across every
// size-class branch (<=3, 4..16, 17..48, >48, and the 48-byte unrolled loop with tails).

package gpu

import "testing"

func TestHashMatchesSkiaChecksum(t *testing.T) {
	buf := make([]byte, 257)
	for i := range buf {
		buf[i] = byte(i*131 + 7)
	}
	cases := []struct {
		size   int
		hash32 uint32
		hash64 uint64
	}{
		{size: 0, hash32: 3804095577, hash64: 290873116282709081},
		{size: 1, hash32: 2361553052, hash64: 18293321427576844444},
		{size: 2, hash32: 1752797707, hash64: 11418718351182631435},
		{size: 3, hash32: 1960534921, hash64: 10254622385556382601},
		{size: 4, hash32: 1857996338, hash64: 16509635570767414834},
		{size: 5, hash32: 593936001, hash64: 10277628655683681921},
		{size: 7, hash32: 3328506793, hash64: 15814254296564038569},
		{size: 8, hash32: 3864621293, hash64: 7697494267825582317},
		{size: 9, hash32: 4029049249, hash64: 5105065739134855585},
		{size: 15, hash32: 129278395, hash64: 4950948410140434875},
		{size: 16, hash32: 4279618134, hash64: 5130725914121194070},
		{size: 17, hash32: 4225536043, hash64: 9728009292243308587},
		{size: 24, hash32: 284539897, hash64: 14579073092769135609},
		{size: 31, hash32: 975065617, hash64: 7913523530717614609},
		{size: 32, hash32: 3690730447, hash64: 10846653637283550159},
		{size: 33, hash32: 3860931963, hash64: 12632815744269231483},
		{size: 47, hash32: 1111017980, hash64: 2227616760917969404},
		{size: 48, hash32: 1916380911, hash64: 13122402444534982383},
		{size: 49, hash32: 797070376, hash64: 6922478815038034984},
		{size: 63, hash32: 3183743118, hash64: 10313449694839443598},
		{size: 64, hash32: 939573025, hash64: 1242861346245099297},
		{size: 65, hash32: 706304969, hash64: 14542232431388875721},
		{size: 100, hash32: 3392140998, hash64: 11534784569776535238},
		{size: 128, hash32: 251659448, hash64: 2595988606023632056},
		{size: 200, hash32: 791782143, hash64: 10663844600607777535},
		{size: 256, hash32: 3224732602, hash64: 3621477909945222074},
		{size: 257, hash32: 3639594660, hash64: 14905867687273942692},
	}
	for _, c := range cases {
		if got := Hash32(buf[:c.size]); got != c.hash32 {
			t.Errorf("Hash32(size %d) = %d, want %d", c.size, got, c.hash32)
		}
		if got := Hash64(buf[:c.size]); got != c.hash64 {
			t.Errorf("Hash64(size %d) = %d, want %d", c.size, got, c.hash64)
		}
	}
}

func TestResourceKeyBasics(t *testing.T) {
	// Invalid until built.
	var k ScratchKey
	if k.IsValid() {
		t.Error("zero key should be invalid")
	}
	rt := GenerateScratchKeyResourceType()
	b := ScratchKeyBuilder(&k, rt, 2)
	s := b.Slice()
	s[0] = 0xdeadbeef
	s[1] = 42
	b.Finish()
	if !k.IsValid() {
		t.Error("built key should be valid")
	}
	if k.ResourceType() != rt {
		t.Errorf("resource type = %d, want %d", k.ResourceType(), rt)
	}
	if k.Size() != 16 {
		t.Errorf("size = %d, want 16", k.Size())
	}
	// The hash covers the domain/size word and the data words.
	want := hash32U32s(k.key[1:])
	if k.Hash() != want {
		t.Errorf("hash = %d, want %d", k.Hash(), want)
	}
	k.Reset()
	if k.IsValid() {
		t.Error("reset key should be invalid")
	}

	// Unique keys with an inner key wrap the inner data after the extra words.
	var inner, outer UniqueKey
	d1 := GenerateUniqueKeyDomain()
	d2 := GenerateUniqueKeyDomain()
	ib := UniqueKeyBuilder(&inner, d1, 2, "inner")
	ib.Slice()[0] = 7
	ib.Slice()[1] = 9
	ib.Finish()
	ob := UniqueKeyBuilderWithInnerKey(&outer, &inner, d2, 1, "outer")
	ob.Slice()[0] = 55
	ob.Finish()
	if !outer.IsValid() || outer.Tag() != "outer" {
		t.Fatal("outer key not built")
	}
	data := outer.Data()
	if len(data) != 4 || data[0] != 55 || data[1] != uint32(d1) || data[2] != 7 || data[3] != 9 {
		t.Errorf("outer data = %v", data)
	}
}

// TestMapKeyMemoization verifies the mapKey string cache: the returned string is byte-identical to a freshly built one,
// is memoized on the key, is dropped on Reset, and — critical for the reused-key-struct path
// (ResourceProvider.dynBufKeys and any recycled key) — is refreshed when the key is rebuilt through a builder rather
// than returning the stale string.
func TestMapKeyMemoization(t *testing.T) {
	rt := GenerateScratchKeyResourceType()

	var k ScratchKey
	b := ScratchKeyBuilder(&k, rt, 2)
	b.Slice()[0] = 0x11223344
	b.Slice()[1] = 0x55667788
	b.Finish()

	// The cached value must equal a raw rebuild over the full key words, and be memoized (str field populated after the
	// first call, and stable across calls).
	want := mapKeyBytes(k.key)
	if k.str != "" {
		t.Fatal("mapKey must be lazily computed, not populated before first use")
	}
	got := k.mapKey()
	if got != want {
		t.Fatalf("mapKey = %q, want %q", got, want)
	}
	if k.str != want {
		t.Fatalf("mapKey result not memoized on the key: str = %q", k.str)
	}
	if k.mapKey() != want {
		t.Fatal("second mapKey call returned a different value")
	}

	// Reset drops the memo.
	k.Reset()
	if k.str != "" || k.IsValid() {
		t.Fatal("Reset must clear both the key and the memoized string")
	}

	// Rebuilding a reused key struct with different data must refresh the memo (not return the stale string from the
	// previous build) — the invariant the provider's key memo relies on when a (size,type) slot is first built into a
	// zero key and when any recycled key is reused.
	var reused ScratchKey
	b1 := ScratchKeyBuilder(&reused, rt, 2)
	b1.Slice()[0] = 1
	b1.Slice()[1] = 2
	b1.Finish()
	first := reused.mapKey()
	b2 := ScratchKeyBuilder(&reused, rt, 2)
	b2.Slice()[0] = 9
	b2.Slice()[1] = 8
	b2.Finish()
	if reused.str != "" {
		t.Fatal("rebuilding via the builder must clear the stale memoized string")
	}
	second := reused.mapKey()
	if second == first {
		t.Fatal("rebuilt key returned the stale mapKey string")
	}
	if second != mapKeyBytes(reused.key) {
		t.Fatalf("rebuilt mapKey = %q, want %q", second, mapKeyBytes(reused.key))
	}

	// A value copy carries the memoized string (immutable and consistent with the copied bytes) — this is how
	// changeUniqueKey (b.uniqueKey = *newKey) and the thread-safe cache entry copy keep the cache map lookups
	// allocation-free.
	var u UniqueKey
	ub := UniqueKeyBuilder(&u, GenerateUniqueKeyDomain(), 1, "tag")
	ub.Slice()[0] = 0xabcdef01
	ub.Finish()
	uWant := u.mapKey()
	cp := u
	if cp.mapKey() != uWant {
		t.Fatal("value copy lost the memoized mapKey string")
	}
	cp.Reset()
	if cp.mapKey() != "" || u.mapKey() != uWant {
		t.Fatal("Reset on a copy must not affect the original's memoized string")
	}
}
