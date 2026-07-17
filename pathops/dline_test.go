// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// TestPathOpsLineUtilities round-trips a line through the float32 host point type and checks the midpoint.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func TestPathOpsLineUtilities(t *testing.T) {
	tests := []dLine{
		{pts: [2]dPoint{{x: 2, y: 1}, {x: 2, y: 1}}},
		{pts: [2]dPoint{{x: 2, y: 1}, {x: 1, y: 1}}},
		{pts: [2]dPoint{{x: 2, y: 1}, {x: 2, y: 2}}},
		{pts: [2]dPoint{{x: 1, y: 1}, {x: 2, y: 2}}},
		{pts: [2]dPoint{{x: 3, y: 0}, {x: 2, y: 1}}},
		{pts: [2]dPoint{{x: 3, y: 2}, {x: 1, y: 1}}},
	}
	for i, line := range tests {
		var line2 dLine
		line2.set([2]geom.Point{line.pts[0].asPoint(), line.pts[1].asPoint()})
		if !line.pts[0].equals(line2.pts[0]) || !line.pts[1].equals(line2.pts[1]) {
			t.Fatalf("line %d round-trip mismatch", i)
		}
		mid := line.ptAtT(.5)
		if !approximatelyEqual((line.pts[0].x+line.pts[1].x)/2, mid.x) {
			t.Fatalf("line %d mid x mismatch", i)
		}
		if !approximatelyEqual((line.pts[0].y+line.pts[1].y)/2, mid.y) {
			t.Fatalf("line %d mid y mismatch", i)
		}
	}
}
