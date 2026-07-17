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

package path

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

// TestArcToRotatedEndpoints verifies the SVG-style elliptical arcs land on their endpoints: ArcToRotated (including the
// CCW sweep-flip lane) ends exactly at (x, y), and RArcToRotated ends at the relative offset from the current point.
func TestArcToRotatedEndpoints(t *testing.T) {
	p := New()
	p.MoveTo(50, 50)
	p.ArcToRotated(10, 20, 30, ArcSizeLarge, geom.DirectionCCW, 70, 80)
	last, ok := p.LastPt()
	if !ok {
		t.Fatal("no last point after ArcToRotated")
	}
	if !nearPt(last, geom.Point{X: 70, Y: 80}, 1e-3) {
		t.Errorf("ArcToRotated endpoint = %v, want (70, 80)", last)
	}
	if p.CountPoints() < 3 {
		t.Errorf("ArcToRotated emitted %d points; expected curve geometry", p.CountPoints())
	}

	p.RArcToRotated(5, 5, 0, ArcSizeSmall, geom.DirectionCW, 5, 5)
	rlast, ok := p.LastPt()
	if !ok {
		t.Fatal("no last point after RArcToRotated")
	}
	if !nearPt(rlast, geom.Point{X: last.X + 5, Y: last.Y + 5}, 1e-3) {
		t.Errorf("RArcToRotated endpoint = %v, want (%v, %v)", rlast, last.X+5, last.Y+5)
	}
}

func nearPt(a, b geom.Point, tol float32) bool {
	dx, dy := a.X-b.X, a.Y-b.Y
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	return dx <= tol && dy <= tol
}
