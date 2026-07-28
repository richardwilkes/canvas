// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// lutProxyKeyCache bounds the small uniquely-keyed LUT textures the pixels-proxy lane mints (pixelsproxy.go): the
// perlin-noise painting-data tables, keyed per shader instance, and the baked gradient ramps, keyed per bake. Neither key
// is content-derived, so an animated shader — a fresh PerlinNoiseShader or a re-baked ramp per frame — produces a fresh
// key every frame, and a uniquely-keyed proxy is only ever unregistered when its ref count reaches zero or when someone
// issues ProcessInvalidUniqueKey. Nobody did, so the provider's map and the textures behind it grew without bound. This
// cache is that missing issuer: it holds the most recently used keys and invalidates the ones that fall off the end, the
// same eviction the clip stack performs for its cached masks. A key that is invalidated while an already-recorded draw
// still samples its proxy stays correct — the proxy holds its own reference, and the next request for that LUT simply
// re-uploads under a fresh proxy.

package gl

import "github.com/richardwilkes/canvas/gpu"

// maxLiveLUTProxies is the number of LUT keys kept registered. Each LUT is at most a 256x4 RGBA8 texture (4KB), and the
// working set is tiny: one dither table, one perlin pair per live noise shader, and one per cached gradient ramp
// (gradientBitmapCache caps its own entries at maxNumCachedGradientBitmaps).
const maxLiveLUTProxies = 2*maxNumCachedGradientBitmaps + 8

// lutProxyKeyCache is an MRU-ordered list of the LUT proxy keys registered with the proxy provider. It is per context (it
// hangs off DirectContext) because the keys it holds are the provider's.
type lutProxyKeyCache struct {
	keys       []gpu.UniqueKey // most recently used first
	maxEntries int
}

func newLUTProxyKeyCache(maxEntries int) *lutProxyKeyCache {
	if maxEntries < 1 {
		panic("the LUT proxy key cache needs room for at least one key")
	}
	return &lutProxyKeyCache{maxEntries: maxEntries}
}

// track records key as the most recently used LUT key, invalidating the least recently used keys once the cache is over
// capacity. Callers pass every key they hand to FindOrCreatePixelsProxyView, hit or miss, so a LUT that keeps being drawn
// keeps its texture.
func (c *lutProxyKeyCache) track(proxyProvider *ProxyProvider, key *gpu.UniqueKey) {
	if proxyProvider == nil || !key.IsValid() {
		return
	}
	for i := range c.keys {
		if c.keys[i].Equal(key) {
			copy(c.keys[1:i+1], c.keys[:i])
			c.keys[0] = *key
			return
		}
	}
	c.keys = append(c.keys, gpu.UniqueKey{})
	copy(c.keys[1:], c.keys)
	c.keys[0] = *key
	for len(c.keys) > c.maxEntries {
		last := len(c.keys) - 1
		proxyProvider.ProcessInvalidUniqueKey(&c.keys[last], nil, InvalidateGPUResourceYes)
		c.keys[last].Reset()
		c.keys = c.keys[:last]
	}
}

// numTracked reports how many keys are registered (tests).
func (c *lutProxyKeyCache) numTracked() int { return len(c.keys) }
