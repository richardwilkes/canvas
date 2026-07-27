// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the SurfaceDrawContext draw entry points' op selection: the reduced-shader-mode routing of the
// StrokeRectOp lane, and the unsimplified fallback shape that lets drawShapeUsingPathRenderer retry the dedicated ops.

package gl

import (
	"slices"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/stroke"
)

// strokeRectOpNames are the two dedicated stroke-rect ops DrawRect builds on its StrokeRectOp lane.
var strokeRectOpNames = []string{"AAStrokeRect", "NonAAStrokeRectOp"}

// pathRendererOpNames are the ops the path-renderer chain produces — the general-purpose fallback the dedicated
// analytic ops exist to avoid.
var pathRendererOpNames = []string{
	"AAConvexPathOp", "AAFlatteningConvexPathOp", "AAHairlineOp", "DefaultPathOp",
	"PathInnerTriangulateOp", "PathStencilCoverOp", "PathTessellateOp", "StrokeTessellateOp",
	"TriangulatingPathOp",
}

// chainHeadNames returns the Name() of every recorded op-chain head, in record order.
func chainHeadNames(sdc *SurfaceDrawContext) []string {
	task := sdc.GetOpsTask()
	names := make([]string, task.NumOpChains())
	for i := range names {
		names[i] = task.GetChainHead(i).Name()
	}
	return names
}

// drawStrokedTestRect records one wide stroked-rect draw (wide enough to stay off the hairline lane) and returns the
// resulting op-chain head names.
func drawStrokedTestRect(t *testing.T, sdc *SurfaceDrawContext) []string {
	t.Helper()
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)
	identity := geom.IdentityMatrix()
	sdc.DrawRect(nil, solidPaint(1, 0, 0, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 56}, &rec)
	return chainHeadNames(sdc)
}

// TestDrawRectReducedShaderModeSkipsStrokeRectOp: reduced shader mode exists to cut the number of compiled programs, so
// the stroke-rect ops and their GP variants must not be built at all — the draw routes to the path renderer instead.
// The sibling nested-rect lane in drawSimpleShape already honors the flag; the DrawRect lane must too. The two cases
// run as subtests because the recording driver keeps package-level state that a second live context would reset.
func TestDrawRectReducedShaderModeSkipsStrokeRectOp(t *testing.T) {
	// Baseline: with reduced shader mode off, the stroke rect takes a dedicated stroke-rect op. Without this the
	// reduced-mode assertion below could pass for the wrong reason.
	t.Run("default", func(t *testing.T) {
		dc := newShaderRecordingContext(t)
		sdc := newDrawTestSDC(t, dc, 64, 64)
		defer sdc.Release()
		if sdc.Caps().ReducedShaderMode() {
			t.Fatal("the default fake driver must not enable reduced shader mode")
		}
		names := drawStrokedTestRect(t, sdc)
		if len(names) != 1 || !slices.Contains(strokeRectOpNames, names[0]) {
			t.Fatalf("baseline ops = %v, want one stroke-rect op (the lane this test gates)", names)
		}
	})

	t.Run("reduced", func(t *testing.T) {
		options := gpu.DefaultContextOptions()
		options.ReducedShaderVariations = true
		dc := newShaderRecordingContextWithOptions(t, options)
		sdc := newDrawTestSDC(t, dc, 64, 64)
		defer sdc.Release()
		if !sdc.Caps().ReducedShaderMode() {
			t.Fatal("ReducedShaderVariations did not enable reduced shader mode")
		}
		for _, name := range drawStrokedTestRect(t, sdc) {
			if slices.Contains(strokeRectOpNames, name) {
				t.Fatalf("reduced shader mode still built %q", name)
			}
		}
	})
}

// TestDrawRectStrokeAndFillRetriesDedicatedOps: the fallback styled shape must be built unsimplified. Simplifying it in
// the constructor makes drawShapeUsingPathRenderer's own simplify() a no-op, which clears StyledShape.Simplified and so
// skips the drawSimpleShape retry — sending a shape that reduces to a dedicated-op form (here a stroke-and-fill rect
// with a round join, which reduces to a filled round rect) to the general path renderer instead of an analytic
// round-rect op.
func TestDrawRectStrokeAndFillRetriesDedicatedOps(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(6, true /* strokeAndFill */)
	rec.SetStrokeParams(stroke.CapButt, stroke.JoinRound, 4)
	if rec.Style() != stroke.StyleStrokeAndFill {
		t.Fatalf("style = %v, want stroke-and-fill (the lane this test gates)", rec.Style())
	}

	sdc.DrawRect(nil, solidPaint(0, 0, 1, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 12, Top: 12, Right: 52, Bottom: 52}, &rec)

	names := chainHeadNames(sdc)
	if len(names) != 1 {
		t.Fatalf("op chains = %v, want exactly one", names)
	}
	if slices.Contains(pathRendererOpNames, names[0]) {
		t.Errorf("op = %q: the simplified shape must be retried against the dedicated ops rather "+
			"than going to the path renderer", names[0])
	}
}
