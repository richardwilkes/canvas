// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Regression test for the unbindSampler bounds guard: like bindSampler, unbindSampler must ignore a unit index at or
// above the unit count (and a call after abandon() has niled textureUnitStates) instead of indexing out of range.

package gl

import "testing"

func TestUnbindSamplerBoundsGuard(_ *testing.T) {
	// A cache whose GPU back-pointer is nil: reaching any actual GL call would nil-panic, so a clean return proves the
	// bounds guard fired before textureUnitStates was indexed.
	c := &samplerObjectCache{numTextureUnits: 2, textureUnitStates: make([]samplerUnitState, 2)}

	// A unit index at or above the unit count must be ignored, not indexed.
	c.unbindSampler(2)
	c.unbindSampler(99)

	// After abandon(), textureUnitStates is nil and numTextureUnits is 0, so every unit index is out of range.
	c.abandon()
	c.unbindSampler(0)
}
