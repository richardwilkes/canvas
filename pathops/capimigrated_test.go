// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Argument-guard checks for the public boolean-op entry points.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// TestOpOutOfRangeOperator verifies Op's operator-range guard (the deprecated bool Op wrapper's contract): an
// out-of-range operator fails with (nil, false) instead of computing anything.
func TestOpOutOfRangeOperator(t *testing.T) {
	one := path.New()
	one.AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW)
	two := path.New()
	two.AddRect(geom.RectLTRB(5, 5, 15, 15), geom.DirectionCW)

	if result, ok := Op(one, two, PathOp(99)); ok || result != nil {
		t.Errorf("Op with an out-of-range operator = (%v, %v), want (nil, false)", result, ok)
	}
	if result, ok := Op(one, two, PathOp(-1)); ok || result != nil {
		t.Errorf("Op with a negative operator = (%v, %v), want (nil, false)", result, ok)
	}
	// The guard passes valid operators through.
	if result, ok := Op(one, two, Union); !ok || result == nil {
		t.Error("Op(Union) on valid rects should succeed")
	}
}
