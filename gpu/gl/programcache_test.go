// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import "testing"

// setDescKey stamps raw key words onto a ProgramDesc, bypassing the GL pipeline that BuildProgramDesc needs — the cache
// lookup path only reads the key bytes, so hand-built descriptors exercise it without a live context. It reuses the
// existing storage (append over key[:0]) to mirror the cache's reuse discipline.
func setDescKey(d *ProgramDesc, words ...uint32) {
	d.key = append(d.key[:0], words...)
}

// insertProgram populates a cache entry exactly as a miss would (a persistent string key + LRU node), so subsequent
// lookups of the same descriptor are hits.
func insertProgram(c *ProgramCache, desc *ProgramDesc, program *Program) {
	key := desc.MapKey()
	c.entries[key] = c.lru.PushFront(&programCacheEntry{program: program, key: key})
}

// TestProgramCacheHitReuseByteIdentical proves that reusing the cache's keyScratch across interleaved lookups does not
// corrupt the stored map keys — each descriptor keeps resolving to its own program. (The stored keys are independent
// string copies of the scratch, so a later lookup's scratch contents cannot alias an earlier entry's key.)
func TestProgramCacheHitReuseByteIdentical(t *testing.T) {
	c := newProgramCache(nil, programCacheMaxEntries)
	progA := &Program{}
	progB := &Program{}
	progC := &Program{} // a longer key, to force keyScratch to grow then shrink between lookups
	var descA, descB, descC ProgramDesc
	setDescKey(&descA, 0x11111111, 0x22222222)
	setDescKey(&descB, 0x33333333)
	setDescKey(&descC, 0x44444444, 0x55555555, 0x66666666, 0x77777777)
	insertProgram(c, &descA, progA)
	insertProgram(c, &descB, progB)
	insertProgram(c, &descC, progC)

	lookup := func(desc *ProgramDesc, want *Program) {
		t.Helper()
		got, res := c.findOrCreateProgramImpl(desc, nil)
		if res != ProgramCacheHit {
			t.Fatalf("expected cache hit, got %v", res)
		}
		if got != want {
			t.Fatalf("cache returned the wrong program for the key")
		}
	}
	// Interleave short/long/short lookups through the shared keyScratch (which grows for descC then is resliced back
	// down for descA/descB); every one must still resolve correctly.
	for i := 0; i < 4; i++ {
		lookup(&descC, progC)
		lookup(&descA, progA)
		lookup(&descB, progB)
		lookup(&descA, progA)
	}
	if hits := c.Stats().NumProgramCacheHits; hits != 16 {
		t.Fatalf("NumProgramCacheHits = %d, want 16", hits)
	}
	if misses := c.Stats().NumProgramCacheMisses; misses != 0 {
		t.Fatalf("NumProgramCacheMisses = %d, want 0", misses)
	}
}

// TestProgramCacheHitIsAllocFree locks the S104 win: a program-cache hit — the steady-state per-draw case — allocates
// nothing (the reused descriptor storage plus the m[string(keyScratch)] no-allocation map lookup). This is what takes
// the GPU-frame benchmark's per-draw program lookup to zero allocs.
func TestProgramCacheHitIsAllocFree(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector instruments allocations, so AllocsPerRun over-reports")
	}
	c := newProgramCache(nil, programCacheMaxEntries)
	prog := &Program{}
	var desc ProgramDesc
	setDescKey(&desc, 0x0a0b0c0d, 0x01020304, 0x10203040)
	insertProgram(c, &desc, prog)

	if got := testing.AllocsPerRun(200, func() {
		if p, res := c.findOrCreateProgramImpl(&desc, nil); res != ProgramCacheHit || p != prog {
			t.Fatalf("expected cache hit for prog, got res=%v p=%p", res, p)
		}
	}); got != 0 {
		t.Fatalf("program-cache hit allocated %v allocs/op, want 0", got)
	}
}
