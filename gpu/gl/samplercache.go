// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// An LRU cache of GL sampler objects keyed on the packed sampler state, plus a shadow of which sampler is bound to each
// texture unit.

package gl

import "github.com/richardwilkes/canvas/gpu"

// maxCachedSamplers is the maximum number of distinct sampler objects the cache keeps alive.
const maxCachedSamplers = 32

type samplerUnitState struct {
	known            bool
	samplerIDIfKnown uint32
}

type samplerObjectCache struct {
	g                 *Gpu
	samplers          map[uint32]uint32 // state key -> sampler id
	lru               []uint32          // keys, least recently used first
	textureUnitStates []samplerUnitState
	numTextureUnits   int
}

func newSamplerObjectCache(g *Gpu) *samplerObjectCache {
	n := g.numTextureUnits()
	return &samplerObjectCache{
		g:                 g,
		samplers:          make(map[uint32]uint32),
		textureUnitStates: make([]samplerUnitState, n),
		numTextureUnits:   n,
	}
}

func (c *samplerObjectCache) touch(key uint32) {
	for i, k := range c.lru {
		if k == key {
			c.lru = append(append(c.lru[:i:i], c.lru[i+1:]...), key)
			return
		}
	}
	c.lru = append(c.lru, key)
}

// bindSampler ensures a GL sampler object matching state exists (creating and configuring one if needed, evicting the
// least-recently-used entry if the cache is full) and binds it to unitIdx if it isn't already bound there.
func (c *samplerObjectCache) bindSampler(unitIdx int, state gpu.SamplerState) {
	if unitIdx >= c.numTextureUnits {
		return
	}
	// In GL the max aniso value is specified in addition to min/mag filters and the driver is encouraged to consider
	// the other filter settings when doing aniso.
	key := state.AsKey(true /* anisoIsOrthogonal */)
	id, ok := c.samplers[key]
	if !ok {
		f := c.g.fns()
		var s uint32
		f.GenSamplers(1, &s)
		if s == 0 {
			return
		}
		if len(c.samplers) >= maxCachedSamplers && len(c.lru) > 0 {
			// Evict the least recently used sampler.
			evictKey := c.lru[0]
			c.lru = c.lru[1:]
			if evictID, exists := c.samplers[evictKey]; exists {
				f.DeleteSamplers(1, &evictID)
				delete(c.samplers, evictKey)
				// Any unit shadow pointing at the evicted sampler is now stale.
				for i := range c.textureUnitStates {
					if c.textureUnitStates[i].known &&
						c.textureUnitStates[i].samplerIDIfKnown == evictID {
						c.textureUnitStates[i].known = false
					}
				}
			}
		}
		c.samplers[key] = s
		id = s
		minFilter := filterToGLMinFilter(state.Filter, state.MipmapMode)
		magFilter := filterToGLMagFilter(state.Filter)
		wrapX := wrapModeToGLWrap(state.WrapModeX, c.g.Caps())
		wrapY := wrapModeToGLWrap(state.WrapModeY, c.g.Caps())
		f.SamplerParameteri(s, TEXTURE_MIN_FILTER, int32(minFilter))
		f.SamplerParameteri(s, TEXTURE_MAG_FILTER, int32(magFilter))
		f.SamplerParameteri(s, TEXTURE_WRAP_S, int32(wrapX))
		f.SamplerParameteri(s, TEXTURE_WRAP_T, int32(wrapY))
		if c.g.Caps().AnisoSupport {
			maxAniso := min(float32(state.MaxAniso), c.g.glCaps().MaxTextureMaxAnisotropy())
			f.SamplerParameterf(s, TEXTURE_MAX_ANISOTROPY, maxAniso)
		} else if state.IsAniso() {
			panic("aniso sampler without support")
		}
	}
	c.touch(key)
	if !c.textureUnitStates[unitIdx].known ||
		c.textureUnitStates[unitIdx].samplerIDIfKnown != id {
		c.g.fns().BindSampler(uint32(unitIdx), id)
		c.textureUnitStates[unitIdx] = samplerUnitState{known: true, samplerIDIfKnown: id}
	}
}

// unbindSampler binds sampler 0 to unitIdx if a non-zero sampler is currently bound there.
func (c *samplerObjectCache) unbindSampler(unitIdx int) {
	if !c.textureUnitStates[unitIdx].known ||
		c.textureUnitStates[unitIdx].samplerIDIfKnown != 0 {
		c.g.fns().BindSampler(uint32(unitIdx), 0)
		c.textureUnitStates[unitIdx] = samplerUnitState{known: true, samplerIDIfKnown: 0}
	}
}

// invalidateBindings forgets which sampler is bound to each texture unit, forcing the next bindSampler/unbindSampler
// call on each unit to re-issue the GL bind.
func (c *samplerObjectCache) invalidateBindings() {
	for i := range c.textureUnitStates {
		c.textureUnitStates[i] = samplerUnitState{}
	}
}

// abandon drops all cached state without issuing any GL calls (for use after the context is lost).
func (c *samplerObjectCache) abandon() {
	c.samplers = nil
	c.lru = nil
	c.textureUnitStates = nil
	c.numTextureUnits = 0
}

// release deletes all cached sampler objects. Deleting a bound sampler implicitly binds sampler 0, so all binding
// knowledge is invalidated.
func (c *samplerObjectCache) release() {
	if c.numTextureUnits == 0 {
		// We've already been abandoned.
		return
	}
	f := c.g.fns()
	for _, id := range c.samplers {
		sampler := id
		f.DeleteSamplers(1, &sampler)
	}
	c.samplers = make(map[uint32]uint32)
	c.lru = nil
	for i := range c.textureUnitStates {
		c.textureUnitStates[i] = samplerUnitState{}
	}
}
