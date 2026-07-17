// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests.

package geom

import (
	"math"
	"testing"
)

// TestRadiansToDegrees pins the radians→degrees conversion at the quarter/half turn and zero.
func TestRadiansToDegrees(t *testing.T) {
	cases := []struct{ rad, deg float32 }{
		{rad: 0, deg: 0},
		{rad: float32(math.Pi / 2), deg: 90},
		{rad: float32(math.Pi), deg: 180},
		{rad: float32(-math.Pi / 4), deg: -45},
	}
	for _, c := range cases {
		got := RadiansToDegrees(c.rad)
		if d := got - c.deg; d > 1e-3 || d < -1e-3 {
			t.Errorf("RadiansToDegrees(%v) = %v, want %v", c.rad, got, c.deg)
		}
	}
}
