// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the rrect/oval/stroke-rect/texture ops: FillRRectOp batching and its clipToShape geometric
// reduction, the attemptQuadOptimization rrect-clip → drawRRect conversion, circle/ellipse op merging, StrokeRectOp's
// compute_aa_rects math, and TextureOp's same-proxy merge gate.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stroke"
)

func TestFillRRectOpBatching(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	// Two compatible filled rrects merge into a single instanced op.
	sdc.DrawRRect(nil, solidPaint(1, 0, 0, 0.5), gpu.AAYes, &identity,
		geom.MakeRRect(geom.Rect{Left: 4, Top: 4, Right: 24, Bottom: 24}, 4, 6), nil)
	sdc.DrawRRect(nil, solidPaint(0, 1, 0, 0.5), gpu.AAYes, &identity,
		geom.MakeRRect(geom.Rect{Left: 30, Top: 30, Right: 60, Bottom: 50}, 6, 8), nil)

	task := sdc.GetOpsTask()
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("expected 1 chain after merge, got %d", n)
	}
	op, ok := task.GetChainHead(0).(*fillRRectOp)
	if !ok {
		t.Fatalf("chain head is %T, want *fillRRectOp", task.GetChainHead(0))
	}
	if op.NumInstances() != 2 {
		t.Fatalf("expected 2 instances, got %d", op.NumInstances())
	}
}

func TestFillRRectOpClipToShape(t *testing.T) {
	dc := newShaderRecordingContext(t)

	// Rect-vs-rect: the geometry clips in place through the isectRect lane.
	rectRR := geom.MakeRRect(geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 40}, 0, 0)
	identity := geom.IdentityMatrix()
	op := NewFillRRectOp(dc.GLCaps(), solidPaint(1, 0, 0, 0.5), &identity, rectRR,
		LocalCoordsFromRect(rectRR.Rect), gpu.AAYes)
	if op == nil {
		t.Fatal("NewFillRRectOp returned nil")
	}
	rro := op.(*fillRRectOp)

	clipRect := geom.Rect{Left: 20, Top: 0, Right: 64, Bottom: 64}
	shape := MakeShapeRect(clipRect)
	result := op.ClipToShape(raster.ClipIntersect, &identity, &shape, gpu.AAYes)
	if result != ClipResultClippedGeometrically {
		t.Fatalf("ClipToShape = %d, want geometrically clipped", result)
	}
	got := rro.instances[0].rrect.Rect
	want := geom.Rect{Left: 20, Top: 10, Right: 50, Bottom: 40}
	if got != want {
		t.Fatalf("clipped rrect rect = %+v, want %+v", got, want)
	}

	// A rounded op whose corners the clip would slice through mid-radius is unrepresentable with uniform radii, so the
	// conservative intersect refuses and the clip stays external.
	rrect := geom.MakeRRect(geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 40}, 5, 5)
	opRound := NewFillRRectOp(dc.GLCaps(), solidPaint(1, 0, 0, 0.5), &identity, rrect,
		LocalCoordsFromRect(rrect.Rect), gpu.AAYes)
	if r := opRound.ClipToShape(raster.ClipIntersect, &identity, &shape,
		gpu.AAYes); r != ClipResultFail {
		t.Fatalf("mid-corner ClipToShape = %d, want fail (conservative)", r)
	}

	// A clip rect containing the whole rrect keeps the geometry (still a geometric clip).
	containing := MakeShapeRect(geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64})
	opContained := NewFillRRectOp(dc.GLCaps(), solidPaint(1, 0, 0, 0.5), &identity, rrect,
		LocalCoordsFromRect(rrect.Rect), gpu.AAYes)
	if r := opContained.ClipToShape(raster.ClipIntersect, &identity, &containing,
		gpu.AAYes); r != ClipResultClippedGeometrically {
		t.Fatalf("containing ClipToShape = %d, want geometrically clipped", r)
	}
	if gotRR := opContained.(*fillRRectOp).instances[0].rrect; gotRR != rrect {
		t.Fatalf("contained rrect changed: %+v, want %+v", gotRR, rrect)
	}

	// A clipped-out intersection reports so.
	shape2 := MakeShapeRect(geom.Rect{Left: 60, Top: 60, Right: 64, Bottom: 64})
	op2 := NewFillRRectOp(dc.GLCaps(), solidPaint(1, 0, 0, 0.5), &identity, rectRR,
		LocalCoordsFromRect(rectRR.Rect), gpu.AAYes)
	if r := op2.ClipToShape(raster.ClipIntersect, &identity, &shape2,
		gpu.AAYes); r != ClipResultClippedOut {
		t.Fatalf("disjoint ClipToShape = %d, want clipped out", r)
	}

	// AA mismatch fails.
	op3 := NewFillRRectOp(dc.GLCaps(), solidPaint(1, 0, 0, 0.5), &identity, rrect,
		LocalCoordsFromRect(rrect.Rect), gpu.AAYes)
	if r := op3.ClipToShape(raster.ClipIntersect, &identity, &shape,
		gpu.AANo); r != ClipResultFail {
		t.Fatalf("AA-mismatched ClipToShape = %d, want fail", r)
	}
}

func TestRRectClipDrawReduction(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	// An axis-aligned rrect clip plus a constant-color fill covering the clip reduces to a drawRRect of the clip shape
	// (attemptQuadOptimization's kSubmitted rrect lane).
	cs := NewClipStack(geom.IRectWH(64, 64), false)
	cs.ClipRRect(&identity,
		geom.MakeRRect(geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 56}, 10, 12), gpu.AAYes,
		raster.ClipIntersect)

	paint := NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 0, G: 0, B: 1, A: 1})
	sdc.FillRectToRect(cs, paint, gpu.AAYes, &identity,
		geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64},
		geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64})

	task := sdc.GetOpsTask()
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("expected 1 chain, got %d", n)
	}
	if _, ok := task.GetChainHead(0).(*fillRRectOp); !ok {
		t.Fatalf("chain head is %T, want *fillRRectOp (the rrect-clip reduction)",
			task.GetChainHead(0))
	}
}

func TestCircleOpMerge(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)
	sdc.DrawOval(nil, solidPaint(1, 0, 0, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 8, Top: 8, Right: 28, Bottom: 28}, &rec)
	sdc.DrawOval(nil, solidPaint(0, 1, 0, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 34, Top: 34, Right: 58, Bottom: 58}, &rec)

	task := sdc.GetOpsTask()
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("expected 1 chain after circle merge, got %d", n)
	}
	op, ok := task.GetChainHead(0).(*circleOp)
	if !ok {
		t.Fatalf("chain head is %T, want *circleOp", task.GetChainHead(0))
	}
	if len(op.circles) != 2 {
		t.Fatalf("expected 2 circles, got %d", len(op.circles))
	}
	if !op.helper.UsesMSAA() == false && op.helper.AAType() != gpu.AATypeCoverage {
		t.Fatal("circle op must be coverage AA")
	}
}

func TestEllipseOpSelection(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	// A non-circular stroked oval under an axis-aligned matrix selects EllipseOp.
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(3, false)
	sdc.DrawOval(nil, solidPaint(1, 0, 0, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 8, Top: 8, Right: 56, Bottom: 40}, &rec)

	task := sdc.GetOpsTask()
	if _, ok := task.GetChainHead(task.NumOpChains() - 1).(*ellipseOp); !ok {
		t.Fatalf("chain head is %T, want *ellipseOp",
			task.GetChainHead(task.NumOpChains()-1))
	}
}

func TestComputeAARects(t *testing.T) {
	dc := newShaderRecordingContext(t)
	caps := dc.GLCaps()
	identity := geom.IdentityMatrix()

	rect := geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 40}
	devOutside, devOutsideAssist, devInside, degenerate, halfStroke, ok := computeAARects(caps, &identity, rect, 4 /* strokeWidth */, true /* miter */)
	if !ok {
		t.Fatal("computeAARects failed")
	}
	if degenerate {
		t.Fatal("unexpected degenerate stroke")
	}
	if want := (geom.Rect{Left: 8, Top: 8, Right: 52, Bottom: 42}); devOutside != want {
		t.Fatalf("devOutside = %+v, want %+v", devOutside, want)
	}
	if want := (geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 40}); devOutsideAssist != want {
		t.Fatalf("miter devOutsideAssist = %+v, want the unmodified dev rect %+v",
			devOutsideAssist, want)
	}
	if want := (geom.Rect{Left: 12, Top: 12, Right: 48, Bottom: 38}); devInside != want {
		t.Fatalf("devInside = %+v, want %+v", devInside, want)
	}
	if halfStroke != (geom.Point{X: 2, Y: 2}) {
		t.Fatalf("devHalfStrokeSize = %+v, want (2,2)", halfStroke)
	}

	// A stroke wider than the rect collapses the inside to the center point.
	_, _, devInside, degenerate, _, ok = computeAARects(caps, &identity,
		geom.Rect{Left: 10, Top: 10, Right: 14, Bottom: 40}, 8, true)
	if !ok || !degenerate {
		t.Fatalf("expected degenerate stroke, ok=%v degenerate=%v", ok, degenerate)
	}
	if devInside.Left != devInside.Right || devInside.Top != devInside.Bottom {
		t.Fatalf("degenerate devInside not a point: %+v", devInside)
	}
}

func TestStrokeRectOpSelection(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)

	// AA stroked rect → AAStrokeRectOp; two compatible ones merge.
	sdc.DrawRect(nil, solidPaint(1, 0, 0, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 8, Top: 8, Right: 28, Bottom: 28}, &rec)
	sdc.DrawRect(nil, solidPaint(0, 1, 0, 0.5), gpu.AAYes, &identity,
		geom.Rect{Left: 36, Top: 36, Right: 56, Bottom: 56}, &rec)
	task := sdc.GetOpsTask()
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("expected 1 chain after AA stroke-rect merge, got %d", n)
	}
	op, ok := task.GetChainHead(0).(*aaStrokeRectOp)
	if !ok {
		t.Fatalf("chain head is %T, want *aaStrokeRectOp", task.GetChainHead(0))
	}
	if len(op.rects) != 2 {
		t.Fatalf("expected 2 stroke rects, got %d", len(op.rects))
	}

	// Non-AA stroked rect → NonAAStrokeRectOp.
	sdc.DrawRect(nil, solidPaint(0, 0, 1, 0.5), gpu.AANo, &identity,
		geom.Rect{Left: 8, Top: 40, Right: 28, Bottom: 56}, &rec)
	if _, ok = task.GetChainHead(task.NumOpChains() - 1).(*nonAAStrokeRectOp); !ok {
		t.Fatalf("chain head is %T, want *nonAAStrokeRectOp",
			task.GetChainHead(task.NumOpChains()-1))
	}
}

func TestTextureOpMergeGate(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	makeView := func(label string) SurfaceProxyView {
		caps := dc.GLCaps()
		format := caps.GetDefaultBackendFormat(gpu.ColorTypeRGBA8888, false)
		proxy := dc.ProxyProvider().CreateProxy(format, geom.ISize{Width: 8, Height: 8},
			gpu.RenderableNo, 1, gpu.MipmappedNo, gpu.BackingFitExact, gpu.BudgetedYes, label,
			0, UseAllocatorYes)
		if proxy == nil {
			t.Fatal("CreateProxy failed")
		}
		swizzle := caps.ReadSwizzle(format, gpu.ColorTypeRGBA8888)
		return MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, swizzle)
	}

	viewA := makeView("texA")
	white := colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}
	src := geom.Rect{Left: 0, Top: 0, Right: 8, Bottom: 8}
	sdc.DrawTexture(nil, viewA, gpu.AlphaTypePremul, gpu.FilterModeNearest, gpu.MipmapModeNone,
		raster.BlendSrcOver, white, src, geom.Rect{Left: 0, Top: 0, Right: 16, Bottom: 16},
		gpu.QuadAAFlagsNone, SrcRectConstraintFast, &identity)
	sdc.DrawTexture(nil, viewA, gpu.AlphaTypePremul, gpu.FilterModeNearest, gpu.MipmapModeNone,
		raster.BlendSrcOver, white, src, geom.Rect{Left: 20, Top: 20, Right: 36, Bottom: 36},
		gpu.QuadAAFlagsNone, SrcRectConstraintFast, &identity)

	task := sdc.GetOpsTask()
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("expected same-proxy texture quads to merge into 1 chain, got %d", n)
	}
	op, ok := task.GetChainHead(0).(*textureOp)
	if !ok {
		t.Fatalf("chain head is %T, want *textureOp", task.GetChainHead(0))
	}
	if op.NumQuads() != 2 {
		t.Fatalf("expected 2 quads in the merged texture op, got %d", op.NumQuads())
	}

	// A different proxy cannot merge (the chaining lane is deferred).
	viewB := makeView("texB")
	sdc.DrawTexture(nil, viewB, gpu.AlphaTypePremul, gpu.FilterModeNearest, gpu.MipmapModeNone,
		raster.BlendSrcOver, white, src, geom.Rect{Left: 40, Top: 40, Right: 56, Bottom: 56},
		gpu.QuadAAFlagsNone, SrcRectConstraintFast, &identity)
	if n := task.NumOpChains(); n != 2 {
		t.Fatalf("expected cross-proxy texture quads in separate chains, got %d", n)
	}
}
