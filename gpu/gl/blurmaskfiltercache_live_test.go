// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the general filter-mask thread-safe cache. A mask-filtered shape's filtered A8 mask is
// memoized under a unique key so a repeated draw reuses the upload. A Solid-style blur is used so the draw skips the
// analytic direct-filter-mask rect lane and takes the general filter-mask path; a small shape exercises the SW
// (lazy-upload) lane and a large shape exercises the HW (rendered-proxy) lane. Skips when no GL context is available.

package gl_test

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/maskfilter"
)

// TestLiveMaskFilterCacheSW verifies the SW lane's filtered mask is memoized: a repeated draw adds one entry then hits
// it without growing the cache, and the two readbacks are byte-identical.
func TestLiveMaskFilterCacheSW(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	cache := dc.ThreadSafeCache()

	base := cache.NumEntries()
	adds0 := cache.Stats().Adds
	// A 20px Solid-blur rect stays under the kMIN_GPU_BLUR gate, so it takes the SW lane.
	first, _ := drawBlurredRect(t, dc, 64, 22, 22, 42, 42, maskfilter.BlurSolid, 3)
	if got := cache.NumEntries() - base; got != 1 {
		t.Fatalf("expected 1 new cache entry after the first SW draw, got %d", got)
	}
	if got := cache.Stats().Adds - adds0; got != 1 {
		t.Fatalf("expected 1 cache Add after the first SW draw, got %d", got)
	}

	hits0 := cache.Stats().Hits
	second, _ := drawBlurredRect(t, dc, 64, 22, 22, 42, 42, maskfilter.BlurSolid, 3)
	if got := cache.NumEntries() - base; got != 1 {
		t.Fatalf("cache should stay at 1 entry after the SW repeat, got %d", got)
	}
	if cache.Stats().Hits <= hits0 {
		t.Fatalf("expected a cache hit on the SW repeat draw (hits stayed at %d)", hits0)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("cached SW mask reuse produced a different image")
	}
}

// TestLiveMaskFilterCacheHW verifies the HW lane's rendered filtered mask is memoized the same way. A 100px shape
// clears the kMIN_GPU_BLUR gate, so the mask is generated (create_mask_GPU) and blurred (GaussianBlur) on the GPU, then
// cached with an explicit long-lived ref; the repeat reuses it.
func TestLiveMaskFilterCacheHW(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	cache := dc.ThreadSafeCache()

	base := cache.NumEntries()
	adds0 := cache.Stats().Adds
	first, _ := drawBlurredRect(t, dc, 160, 30, 30, 130, 130, maskfilter.BlurSolid, 8)
	if got := cache.NumEntries() - base; got != 1 {
		t.Fatalf("expected 1 new cache entry after the first HW draw, got %d", got)
	}
	if got := cache.Stats().Adds - adds0; got != 1 {
		t.Fatalf("expected 1 cache Add after the first HW draw, got %d", got)
	}

	hits0 := cache.Stats().Hits
	second, _ := drawBlurredRect(t, dc, 160, 30, 30, 130, 130, maskfilter.BlurSolid, 8)
	if got := cache.NumEntries() - base; got != 1 {
		t.Fatalf("cache should stay at 1 entry after the HW repeat, got %d", got)
	}
	if cache.Stats().Hits <= hits0 {
		t.Fatalf("expected a cache hit on the HW repeat draw (hits stayed at %d)", hits0)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("cached HW mask reuse produced a different image")
	}
}

// TestLiveMaskFilterCacheDistinctSigmas verifies distinct blur parameters key distinct entries: two draws that differ
// only in sigma each add their own entry rather than colliding.
func TestLiveMaskFilterCacheDistinctSigmas(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	cache := dc.ThreadSafeCache()

	base := cache.NumEntries()
	drawBlurredRect(t, dc, 64, 22, 22, 42, 42, maskfilter.BlurSolid, 3)
	drawBlurredRect(t, dc, 64, 22, 22, 42, 42, maskfilter.BlurSolid, 4)
	if got := cache.NumEntries() - base; got != 2 {
		t.Fatalf("expected 2 distinct cache entries for two sigmas, got %d", got)
	}
}
