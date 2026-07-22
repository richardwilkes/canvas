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

// TestSamplerStateAsKeyMaxAnisoNoSpill verifies that the maxAniso field is wide enough to hold MaxMaxAniso (1024)
// without spilling into the adjacent filter/mipmap bits, and that MaxAniso does not alias the filter setting.
func TestSamplerStateAsKeyMaxAnisoNoSpill(t *testing.T) {
	// At the extreme clamp value the maxAniso field must not overlap the filter bit. With aniso active (MaxAniso != 1)
	// and anisoIsOrthogonal false the filter/mipmap bits are not written, so the whole key is just the maxAniso field
	// shifted left by 4: 1024<<4 == 16384. A 10-bit field would have overflowed to 0 in its own slot and set bit 14.
	s := SamplerStateAniso(WrapModeClamp, WrapModeClamp, MaxMaxAniso, false)
	if s.MaxAniso != MaxMaxAniso {
		t.Fatalf("SamplerStateAniso clamped MaxAniso to %d, want %d", s.MaxAniso, MaxMaxAniso)
	}
	got := s.AsKey(false)
	want := uint32(MaxMaxAniso) << 4
	if got != want {
		t.Fatalf("AsKey(false) at MaxMaxAniso = %#x, want %#x", got, want)
	}

	// Distinct MaxAniso values (including the top of the range) must never collide with an unrelated filter-bearing
	// state, which was the failure mode of a too-narrow field spilling into the filter bit.
	linearNoAniso := SamplerState{Filter: FilterModeLinear, MaxAniso: 1}
	if linearNoAniso.AsKey(false) == got {
		t.Fatalf("MaxMaxAniso key %#x collides with a linear-filter no-aniso key", got)
	}
}

// TestSamplerStateAsKeyDistinct verifies the packed key is injective across the full MaxAniso range and the orthogonal
// filter/mipmap fields, so no two distinct sampler states share a key.
func TestSamplerStateAsKeyDistinct(t *testing.T) {
	seen := make(map[uint32]SamplerState)
	anisoValues := []int32{1, 2, 15, 16, 1023, 1024}
	for _, orthogonal := range []bool{false, true} {
		for _, aniso := range anisoValues {
			for f := FilterMode(0); int(f) < FilterModeCount; f++ {
				for m := MipmapMode(0); int(m) < MipmapModeCount; m++ {
					s := SamplerState{Filter: f, MipmapMode: m, MaxAniso: aniso}
					key := s.AsKey(orthogonal)
					// When aniso is active and not orthogonal, filter/mipmap are deliberately not part of the key, so
					// those states legitimately share a key; skip the collision check for them.
					if aniso != 1 && !orthogonal {
						continue
					}
					if prev, ok := seen[key]; ok && prev != s {
						t.Fatalf("key %#x collision between %+v and %+v", key, prev, s)
					}
					seen[key] = s
				}
			}
		}
	}
}
