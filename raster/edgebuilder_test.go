// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// TestBuildPolyEdgesClippedIsAllocationFree: the clipped all-lines fast path must take its line-clipper output from the
// caller-owned edgeScratch. A local [LineClipperMaxPoints]geom.Point escapes through the edgeSink interface calls, so
// it becomes one 32-byte heap allocation per clipped line segment on the non-AA polygon path — exactly the per-segment
// traffic edgeScratch exists to eliminate.
func TestBuildPolyEdgesClippedIsAllocationFree(t *testing.T) {
	// A polygon much larger than the clip, so every edge is clipped and most split into several segments.
	p := path.New()
	p.MoveTo(-60, -40)
	p.LineTo(90, -30)
	p.LineTo(120, 70)
	p.LineTo(30, 110)
	p.LineTo(-70, 60)
	p.Close()
	if p.SegmentMasks() != path.SegmentLine {
		t.Fatal("the test path must be all lines to reach buildPolyEdges")
	}
	clip := geom.IRectLTRB(0, 0, 40, 40)

	var basic edgeBuilder
	var analytic analyticEdgeBuilder
	// Warm up: prime the path's cached convexity/segment masks and let the arenas and edge list reach steady state.
	for i := 0; i < 3; i++ {
		if n := basic.buildEdges(p, &clip); n == 0 {
			t.Fatal("edgeBuilder produced no edges")
		}
		basic.reset()
		if n := analytic.buildEdges(p, &clip); n == 0 {
			t.Fatal("analyticEdgeBuilder produced no edges")
		}
		analytic.reset()
	}
	if allocs := testing.AllocsPerRun(50, func() {
		basic.buildEdges(p, &clip)
		basic.reset()
	}); allocs != 0 {
		t.Fatalf("clipped edgeBuilder build allocated %v times per run, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(50, func() {
		analytic.buildEdges(p, &clip)
		analytic.reset()
	}); allocs != 0 {
		t.Fatalf("clipped analyticEdgeBuilder build allocated %v times per run, want 0", allocs)
	}
}

// TestEdgeBuilderResetReleasesPath: reset zeroes head/tail/inv so an idle pooled builder pins neither arena edges nor a
// caller's blitter; it must drop the scratch iterators' *path.Path for the same reason, or a pooled builder keeps the
// last-filled path (and all of its point/verb storage) alive. path.TestEdgeIterReleaseDropsPath covers what Release
// itself clears; this covers reset calling it — and that the builder still works afterwards.
func TestEdgeBuilderResetReleasesPath(t *testing.T) {
	var releasedIter path.EdgeIter
	releasedIter.Release()

	p := path.New().AddCircle(20, 20, 10, geom.DirectionCW)
	clip := geom.IRectLTRB(0, 0, 40, 40)

	var basic edgeBuilder
	var analytic analyticEdgeBuilder
	for _, shiftedClip := range []*geom.IRect{nil, &clip} {
		if n := basic.buildEdges(p, shiftedClip); n == 0 {
			t.Fatalf("clip=%v: edgeBuilder produced no edges", shiftedClip)
		}
		basic.reset()
		if basic.scratch.iter != releasedIter {
			t.Fatalf("clip=%v: edgeBuilder.reset left the scratch iterator holding the path", shiftedClip)
		}
		if n := analytic.buildEdges(p, shiftedClip); n == 0 {
			t.Fatalf("clip=%v: analyticEdgeBuilder produced no edges", shiftedClip)
		}
		analytic.reset()
		if analytic.scratch.iter != releasedIter {
			t.Fatalf("clip=%v: analyticEdgeBuilder.reset left the scratch iterator holding the path", shiftedClip)
		}
	}

	// A reset builder must still build the same edge list, both clipped and unclipped.
	for _, shiftedClip := range []*geom.IRect{nil, &clip} {
		want := basic.buildEdges(p, shiftedClip)
		basic.reset()
		if got := basic.buildEdges(p, shiftedClip); got != want {
			t.Fatalf("clip=%v: after reset, edgeBuilder produced %d edges, want %d", shiftedClip, got, want)
		}
		basic.reset()
	}
}
