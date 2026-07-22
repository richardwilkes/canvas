// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// makeFillShape wraps a path as a simple-fill StyledShape without simplification (so the raw verbs — and therefore the
// segment mask — are preserved for the test).
func makeFillShape(p *path.Path) StyledShape {
	return MakeStyledShapePath(p, MakeStyle(stroke.NewStrokeRec(stroke.InitStyleFill)), DoSimplifyNo)
}

// TestOnStencilPathChopGuards covers the guards OnStencilPath relies on via tessChopPathIfNecessary. Historically
// OnStencilPath called preChopPathCurves unconditionally once the device bounds were large, which panics
// ("preChopPathCurves viewport is too large") for a large straight-line-only non-convex stencil/clip element — a shape
// OnCanDrawPath accepts. The shared helper must instead skip line-only paths and bail out gracefully when the
// conservative clip viewport is too large for preChopPathCurves.
func TestOnStencilPathChopGuards(t *testing.T) {
	identity := geom.IdentityMatrix()

	// A gigantic straight-line-only concave polygon: device bounds are far past the chop threshold, and the
	// conservative clip viewport is large enough that preChopPathCurves would panic. Because the path is line-only,
	// tessChopPathIfNecessary must skip chopping entirely (return true, leave the path untouched, never panic).
	lineOnly := &path.Path{}
	lineOnly.MoveTo(0, 0)
	lineOnly.LineTo(2_000_000, 0)
	lineOnly.LineTo(1_000_000, 1_000)
	lineOnly.LineTo(2_000_000, 2_000_000)
	lineOnly.LineTo(0, 2_000_000)
	lineOnly.Close()
	if mask := lineOnly.SegmentMasks(); mask != path.SegmentLine {
		t.Fatalf("expected a line-only path, got segment mask %d", mask)
	}
	lineShape := makeFillShape(lineOnly)
	hugeViewport := geom.IRect{Left: 0, Top: 0, Right: 2_000_000, Bottom: 2_000_000}
	choppedLine := lineOnly
	if !tessChopPathIfNecessary(&identity, &lineShape, hugeViewport, lineShape.Style().Rec(), &choppedLine) {
		t.Fatal("line-only stencil path should report chopping as possible")
	}
	if choppedLine != lineOnly {
		t.Fatal("line-only stencil path must not be chopped")
	}

	// A curved path whose device bounds exceed the threshold and whose conservative clip viewport is itself too large
	// for preChopPathCurves must bail out (return false) rather than panic. OnStencilPath ignores the result and leaves
	// the path unchopped.
	curved := &path.Path{}
	curved.MoveTo(0, 0)
	curved.CubicTo(2_000_000, -1_000_000, -1_000_000, 2_000_000, 2_000_000, 2_000_000)
	curved.Close()
	curvedShape := makeFillShape(curved)
	if tessChopPathIfNecessary(&identity, &curvedShape, hugeViewport, curvedShape.Style().Rec(), nil) {
		t.Fatal("a curved path with an oversized viewport should report chopping as impossible")
	}
}

// TestOnStencilPathHugeLineOnly drives OnStencilPath end-to-end for the exact case the issue describes: a large
// straight-line-only non-convex simple-fill clip element. Before the fix OnStencilPath called preChopPathCurves
// unconditionally and panicked ("preChopPathCurves viewport is too large"); now it must record a draw op without
// panicking.
func TestOnStencilPathHugeLineOnly(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()

	lineOnly := &path.Path{}
	lineOnly.MoveTo(0, 0)
	lineOnly.LineTo(2_000_000, 0)
	lineOnly.LineTo(1_000_000, 1_000)
	lineOnly.LineTo(2_000_000, 2_000_000)
	lineOnly.LineTo(0, 2_000_000)
	lineOnly.Close()
	shape := makeFillShape(lineOnly)
	identity := geom.IdentityMatrix()

	before := sdc.GetOpsTask().NumOpChains()
	r := NewTessellationPathRenderer()
	r.OnStencilPath(&StencilPathArgs{
		Context:                dc,
		SurfaceDrawContext:     sdc,
		ViewMatrix:             &identity,
		Shape:                  &shape,
		ClipConservativeBounds: geom.IRect{Left: 0, Top: 0, Right: 2_000_000, Bottom: 2_000_000},
		DoStencilMSAA:          gpu.AANo,
	})
	if after := sdc.GetOpsTask().NumOpChains(); after <= before {
		t.Fatalf("OnStencilPath recorded no draw op (chains %d -> %d)", before, after)
	}
}

// TestPreChopPathCurvesPanicsOnHugeViewport pins the trap the OnStencilPath guards exist to avoid: preChopPathCurves
// itself panics when the viewport is too large to tessellate a fully-contained curve within the segment budget.
func TestPreChopPathCurvesPanicsOnHugeViewport(t *testing.T) {
	identity := geom.IdentityMatrix()
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(2_000_000, 2_000_000)
	p.Close()
	defer func() {
		if recover() == nil {
			t.Fatal("expected preChopPathCurves to panic on an oversized viewport")
		}
	}()
	preChopPathCurves(tessPrecision, p, &identity, geom.Rect{Right: 2_000_000, Bottom: 2_000_000})
}
