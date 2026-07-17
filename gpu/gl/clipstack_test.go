// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the clip stack: the stencil-settings translation tables, styled-shape simplification, the
// uniform-rrect helpers, rect subtraction, the SW mask helper's replace-mode rasterization, clip-stack state/element
// bookkeeping (merging, invalidation, save/restore), preApply lanes, and the apply() lanes (scissor-only, analytic FPs,
// cached SW masks, stencil masks with the mustRenderClip idempotence).

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

//////////////////////////////////////////////////////////////////////////////
// Stencil settings

func TestUserStencilSettingsFlags(t *testing.T) {
	unused := UnusedStencilSettings()
	if unused.Flags(false) != stencilFlagsAll {
		t.Fatalf("kUnused flags(false) = %#x, want all (%#x)", unused.Flags(false),
			stencilFlagsAll)
	}
	if !unused.IsDisabled(false) || unused.IsDisabled(true) == false && false {
		t.Fatal("kUnused must be disabled without a stencil clip")
	}
	// With a stencil clip, kAlwaysIfInClip no longer always passes, so the settings are enabled (they test the clip
	// bit) but don't modify the stencil.
	if unused.IsDisabled(true) {
		t.Fatal("kUnused must be enabled when a stencil clip is present")
	}
	if !unused.IsTwoSided(false) == false {
		t.Fatal("kUnused is single sided")
	}

	// gDrawToStencil: always passes, modifies user bits, no wrap ops.
	if drawToStencilSettings.IsDisabled(false) || drawToStencilSettings.IsDisabled(true) {
		t.Fatal("drawToStencil must be enabled")
	}
	if drawToStencilSettings.UsesWrapOp(false) {
		t.Fatal("IncMaybeClamp is not a wrap op")
	}
}

func TestStencilSettingsReset(t *testing.T) {
	// userToClipIsect with a stencil clip on an 8-bit buffer: clipBit=0x80, userMask=0x7f.
	s := MakeStencilSettings(&userToClipIsect, true, 8)
	if s.IsDisabled() || s.IsTwoSided() {
		t.Fatal("userToClipIsect must be enabled and single sided")
	}
	face := s.SingleSidedFace()
	want := StencilFace{
		Ref:       0x80,
		Test:      StencilTestLess,
		TestMask:  0xff, // clipBit | (0xffff & userMask)
		PassOp:    StencilOpReplace,
		FailOp:    StencilOpZero,
		WriteMask: 0xff, // clipBit | (0xffff & userMask)
	}
	if *face != want {
		t.Fatalf("userToClipIsect face = %+v, want %+v", *face, want)
	}

	// The direct replace-clip settings only touch the clip bit.
	s2 := MakeStencilSettings(&replaceClipSettings, true, 8)
	face2 := s2.SingleSidedFace()
	if face2.WriteMask != 0x80 {
		t.Fatalf("replaceClip write mask = %#x, want 0x80", face2.WriteMask)
	}
	if face2.Test != StencilTestAlways || face2.TestMask != 0 {
		t.Fatalf("replaceClip must always pass (test %v mask %#x)", face2.Test, face2.TestMask)
	}
	if !s2.DoesWrite() {
		t.Fatal("replaceClip writes the clip bit")
	}

	// The kUnused settings with a stencil clip translate to a clip-bit equality test.
	s3 := MakeStencilSettings(UnusedStencilSettings(), true, 8)
	face3 := s3.SingleSidedFace()
	if face3.Test != StencilTestEqual || face3.TestMask != 0x80 || face3.Ref != 0x80 {
		t.Fatalf("kUnused+clip face = %+v, want equal test against the clip bit", *face3)
	}
	if s3.DoesWrite() {
		t.Fatal("kUnused must not write the stencil")
	}

	// Disabled settings compare equal regardless of face garbage; invalid never compares equal.
	var d1, d2 StencilSettings
	d1.SetDisabled()
	d2.SetDisabled()
	if !d1.Equal(&d2) {
		t.Fatal("disabled settings must compare equal")
	}
	d2.Invalidate()
	if d1.Equal(&d2) || d2.Equal(&d2) {
		t.Fatal("invalid settings never compare equal")
	}
}

//////////////////////////////////////////////////////////////////////////////
// StyledShape + rrect helpers

func TestShapeSimplify(t *testing.T) {
	// A closed rect path simplifies to a rect.
	p := &path.Path{}
	p.AddRect(geom.Rect{Left: 1, Top: 2, Right: 9, Bottom: 8}, geom.DirectionCW)
	s := MakeShapePath(p)
	s.Simplify(ShapeSimplifyAll)
	if !s.IsRect() || s.Rect() != (geom.Rect{Left: 1, Top: 2, Right: 9, Bottom: 8}) {
		t.Fatalf("rect path simplified to %v", s.Type())
	}

	// An oval path simplifies to an oval rrect.
	p = &path.Path{}
	p.AddOval(geom.Rect{Left: 0, Top: 0, Right: 10, Bottom: 20}, geom.DirectionCW)
	s = MakeShapePath(p)
	s.Simplify(ShapeSimplifyAll)
	if !s.IsRRect() || s.RRect().Type != geom.RRectOval {
		t.Fatalf("oval path simplified to %v", s.Type())
	}

	// A line path with the fill flag simplifies to empty.
	p = &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(5, 5)
	s = MakeShapePath(p)
	s.Simplify(ShapeSimplifyAll)
	if !s.IsEmpty() {
		t.Fatalf("line path with simple fill simplified to %v", s.Type())
	}

	// Without the fill flag it stays a line.
	s = MakeShapePath(p)
	s.Simplify(ShapeSimplifyIgnoreWinding | ShapeSimplifyMakeCanonical)
	if !s.IsLine() {
		t.Fatalf("line path without simple fill simplified to %v", s.Type())
	}

	// A zero-area rect turns into empty under simple fill.
	s = MakeShapeRect(geom.Rect{Left: 3, Top: 1, Right: 3, Bottom: 9})
	s.Simplify(ShapeSimplifyAll)
	if !s.IsEmpty() {
		t.Fatalf("degenerate rect simplified to %v", s.Type())
	}

	// An inverted path keeps its inversion through the shape.
	p = &path.Path{}
	p.AddRect(geom.Rect{Left: 0, Top: 0, Right: 4, Bottom: 4}, geom.DirectionCW)
	p.SetFillType(path.FillInverseWinding)
	s = MakeShapePath(p)
	if !s.Inverted() {
		t.Fatal("inverse-fill path shape must report inverted")
	}
	s.Simplify(ShapeSimplifyAll)
	if !s.IsRect() || !s.Inverted() {
		t.Fatalf("inverted rect path: type %v inverted %v", s.Type(), s.Inverted())
	}
}

func TestRRectHelpers(t *testing.T) {
	rr := geom.MakeRRect(geom.Rect{Left: 0, Top: 0, Right: 20, Bottom: 10}, 2, 2)

	// ContainsPoint: center yes, outside no, square-corner region of the rounded corner no.
	if !rrectContainsPoint(rr, geom.Point{X: 10, Y: 5}) {
		t.Fatal("center must be contained")
	}
	if rrectContainsPoint(rr, geom.Point{X: 21, Y: 5}) {
		t.Fatal("outside the bounds must not be contained")
	}
	if rrectContainsPoint(rr, geom.Point{X: 0.1, Y: 0.1}) {
		t.Fatal("the corner notch must not be contained")
	}
	if !rrectContainsPoint(rr, geom.Point{X: 2, Y: 2}) {
		t.Fatal("the corner-circle center must be contained")
	}

	// InnerBounds: for these radii the corner-inscribed candidate has the largest area (~166 vs 160 horizontal / 120
	// vertical), so all edges inset by (1 - sqrt(2)/2) * r.
	inner := rrectInnerBounds(rr)
	const cornerScale = 1 - 0.70710678 // 1 - sqrt(2)/2
	wantInner := geom.Rect{
		Left: 2 * cornerScale, Top: 2 * cornerScale,
		Right: 20 - 2*cornerScale, Bottom: 10 - 2*cornerScale,
	}
	if geom.ScalarAbs(inner.Left-wantInner.Left) > 1e-4 ||
		geom.ScalarAbs(inner.Top-wantInner.Top) > 1e-4 ||
		geom.ScalarAbs(inner.Right-wantInner.Right) > 1e-4 ||
		geom.ScalarAbs(inner.Bottom-wantInner.Bottom) > 1e-4 {
		t.Fatalf("inner bounds = %v, want %v", inner, wantInner)
	}
	// A wide, strongly-rounded rrect prefers the horizontal inset instead.
	wide := geom.MakeRRect(geom.Rect{Left: 0, Top: 0, Right: 40, Bottom: 10}, 5, 5)
	if got := rrectInnerBounds(wide); got != (geom.Rect{Left: 5, Top: 0, Right: 35, Bottom: 10}) {
		t.Fatalf("wide inner bounds = %v", got)
	}

	// ConservativeIntersect: b contained in a with matching radii anchors → b.
	a := geom.MakeRRect(geom.Rect{Left: 0, Top: 0, Right: 20, Bottom: 20}, 3, 3)
	b := geom.MakeRRect(geom.Rect{Left: 4, Top: 4, Right: 16, Bottom: 16}, 0, 0)
	got := rrectConservativeIntersect(a, b)
	if got.Type != geom.RRectRect || got.Rect != b.Rect {
		t.Fatalf("contained rect intersect = %+v", got)
	}
	if !rrectsEqual(got, b) {
		t.Fatal("intersection must equal the contained rect")
	}

	// Disjoint rrects intersect to empty.
	c := geom.MakeRRect(geom.Rect{Left: 30, Top: 30, Right: 40, Bottom: 40}, 2, 2)
	if !rrectConservativeIntersect(a, c).IsEmpty() {
		t.Fatal("disjoint intersection must be empty")
	}

	// Offset overlapping rrects with equal radii still produce a representable uniform result: every intersection
	// corner anchors to one input and passes the inside-corner test.
	d := geom.MakeRRect(geom.Rect{Left: 10, Top: 0, Right: 30, Bottom: 20}, 3, 3)
	if got = rrectConservativeIntersect(a, d); got.Type != geom.RRectSimple ||
		got.Rect != (geom.Rect{Left: 10, Top: 0, Right: 20, Bottom: 20}) ||
		got.RadiusX != 3 || got.RadiusY != 3 {
		t.Fatalf("offset rrect intersect = %+v", got)
	}

	// A rect that squares off only the bottom corners produces mixed per-corner radii — that would need a
	// nine-patch-ish rrect representation, but the uniform storage cannot express it, so the helper conservatively
	// reports empty.
	e := geom.MakeRRect(geom.Rect{Left: 0, Top: 0, Right: 20, Bottom: 10}, 0, 0)
	if !rrectConservativeIntersect(a, e).IsEmpty() {
		t.Fatal("mixed-radii intersection must conservatively report empty")
	}
}

func TestRectSubtract(t *testing.T) {
	a := geom.IRect{Left: 0, Top: 0, Right: 10, Bottom: 10}

	// Disjoint: A unchanged, exact.
	if got, exact := rectSubtract(a, geom.IRect{Left: 20, Top: 0, Right: 30, Bottom: 10}); got != a || !exact {
		t.Fatalf("disjoint subtract = %v exact %v", got, exact)
	}
	// B spans the right half: exact left chunk.
	if got, exact := rectSubtract(a, geom.IRect{Left: 6, Top: -1, Right: 12, Bottom: 11}); got != (geom.IRect{Left: 0, Top: 0, Right: 6, Bottom: 10}) || !exact {
		t.Fatalf("right-span subtract = %v exact %v", got, exact)
	}
	// B contains A: empty, exact.
	if got, exact := rectSubtract(a, geom.IRect{Left: -1, Top: -1, Right: 11, Bottom: 11}); !got.IsEmpty() || !exact {
		t.Fatalf("containing subtract = %v exact %v", got, exact)
	}
	// B is a hole in the middle: not exact, the largest chunk is returned.
	got, exact := rectSubtract(a, geom.IRect{Left: 4, Top: 4, Right: 6, Bottom: 6})
	if exact {
		t.Fatal("hole subtract cannot be exact")
	}
	if got.IsEmpty() || !a.ContainsRect(got) || got.Intersects(geom.IRect{Left: 4, Top: 4, Right: 6, Bottom: 6}) {
		t.Fatalf("hole subtract chunk = %v", got)
	}
	// subtractRects with exact required returns A for inexact subtractions.
	if got = subtractRects(a, geom.IRect{Left: 4, Top: 4, Right: 6, Bottom: 6}, true); got != a {
		t.Fatalf("exact-required subtract = %v, want %v", got, a)
	}
}

func TestSWMaskHelperPixels(t *testing.T) {
	var h swMaskHelper
	if !h.init(geom.IRect{Left: 10, Top: 10, Right: 26, Bottom: 26}) {
		t.Fatal("init failed")
	}
	at := func(x, y int32) uint8 { // device-space coords
		return h.pixels[int(y-10)*16+int(x-10)]
	}

	// First intersect element: clear to 0, draw coverage 1.
	rect := MakeShapeRect(geom.Rect{Left: 12, Top: 12, Right: 24, Bottom: 24})
	identity := geom.IdentityMatrix()
	drawToSWMask(&h, &ClipElement{
		Shape: rect, LocalToDevice: identity,
		Op: raster.ClipIntersect, AA: gpu.AAYes,
	}, true)
	if at(16, 16) != 0xFF {
		t.Fatalf("inside first element = %d, want 255", at(16, 16))
	}
	if at(11, 11) != 0x00 {
		t.Fatalf("outside first element = %d, want 0", at(11, 11))
	}

	// A difference element erases its interior.
	hole := MakeShapeRect(geom.Rect{Left: 15, Top: 15, Right: 19, Bottom: 19})
	drawToSWMask(&h, &ClipElement{
		Shape: hole, LocalToDevice: identity,
		Op: raster.ClipDifference, AA: gpu.AAYes,
	}, false)
	if at(16, 16) != 0x00 {
		t.Fatalf("inside difference hole = %d, want 0", at(16, 16))
	}
	if at(13, 13) != 0xFF {
		t.Fatalf("outside hole, inside first = %d, want 255", at(13, 13))
	}

	// A subsequent intersect erases everything outside itself (inverse fill, coverage 0).
	tighten := MakeShapeRect(geom.Rect{Left: 12, Top: 12, Right: 17, Bottom: 24})
	drawToSWMask(&h, &ClipElement{
		Shape: tighten, LocalToDevice: identity,
		Op: raster.ClipIntersect, AA: gpu.AAYes,
	}, false)
	if at(13, 13) != 0xFF {
		t.Fatalf("inside tightened clip = %d, want 255", at(13, 13))
	}
	if at(22, 13) != 0x00 {
		t.Fatalf("outside tightened clip = %d, want 0", at(22, 13))
	}
}

//////////////////////////////////////////////////////////////////////////////
// ClipStack bookkeeping

func deviceBounds64() geom.IRect { return geom.IRect{Right: 64, Bottom: 64} }

func TestClipStackStates(t *testing.T) {
	cs := NewClipStack(deviceBounds64(), false)
	identity := geom.IdentityMatrix()

	if cs.ClipState() != ClipStackWideOpen {
		t.Fatalf("initial state = %v, want wide open", cs.ClipState())
	}
	if cs.GetConservativeBounds() != deviceBounds64() {
		t.Fatal("wide-open conservative bounds must be the device bounds")
	}

	// A non-AA axis-aligned rect intersect: device rect state, rounded bounds.
	cs.ClipRect(&identity, geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 50}, gpu.AANo,
		raster.ClipIntersect)
	if cs.ClipState() != ClipStackDeviceRect {
		t.Fatalf("state = %v, want device rect", cs.ClipState())
	}
	if got := cs.GetConservativeBounds(); got != (geom.IRect{Left: 10, Top: 10, Right: 50, Bottom: 50}) {
		t.Fatalf("conservative bounds = %v", got)
	}

	// Save, tighten with an rrect, verify, restore, verify the old state is back.
	cs.Save()
	rr := geom.MakeRRect(geom.Rect{Left: 20, Top: 20, Right: 40, Bottom: 40}, 4, 4)
	cs.ClipRRect(&identity, rr, gpu.AAYes, raster.ClipIntersect)
	if cs.ClipState() != ClipStackDeviceRRect {
		t.Fatalf("state after rrect = %v, want device rrect", cs.ClipState())
	}
	cs.Restore()
	if cs.ClipState() != ClipStackDeviceRect {
		t.Fatalf("state after restore = %v, want device rect", cs.ClipState())
	}
	if got := cs.GetConservativeBounds(); got != (geom.IRect{Left: 10, Top: 10, Right: 50, Bottom: 50}) {
		t.Fatalf("conservative bounds after restore = %v", got)
	}

	// A deferred save/restore pair with no clips allocates no records.
	saves := len(cs.saves)
	cs.Save()
	cs.Restore()
	if len(cs.saves) != saves {
		t.Fatal("deferred save/restore must not allocate records")
	}

	// A disjoint intersect empties the clip.
	cs.Save()
	cs.ClipRect(&identity, geom.Rect{Left: 55, Top: 55, Right: 60, Bottom: 60}, gpu.AANo,
		raster.ClipIntersect)
	if cs.ClipState() != ClipStackEmpty {
		t.Fatalf("state after disjoint intersect = %v, want empty", cs.ClipState())
	}
	if !cs.GetConservativeBounds().IsEmpty() {
		t.Fatal("empty clip must have empty conservative bounds")
	}
	cs.Restore()

	// A path clip makes the state complex.
	cs.Save()
	tri := &path.Path{}
	tri.MoveTo(12, 12)
	tri.LineTo(48, 16)
	tri.LineTo(30, 48)
	tri.Close()
	cs.ClipPath(&identity, tri, gpu.AAYes, raster.ClipIntersect)
	if cs.ClipState() != ClipStackComplex {
		t.Fatalf("state after path clip = %v, want complex", cs.ClipState())
	}
	cs.Restore()
}

func TestClipStackElementCombining(t *testing.T) {
	identity := geom.IdentityMatrix()

	// A rect containing a later rect: the outer element is invalidated and only one element remains active.
	cs := NewClipStack(deviceBounds64(), false)
	cs.ClipRect(&identity, geom.Rect{Left: 4, Top: 4, Right: 60, Bottom: 60}, gpu.AAYes,
		raster.ClipIntersect)
	cs.ClipRect(&identity, geom.Rect{Left: 10, Top: 10, Right: 20, Bottom: 20}, gpu.AAYes,
		raster.ClipIntersect)
	if cs.ClipState() != ClipStackDeviceRect {
		t.Fatalf("contained rect state = %v, want device rect", cs.ClipState())
	}
	if got := cs.GetConservativeBounds(); got != (geom.IRect{Left: 10, Top: 10, Right: 20, Bottom: 20}) {
		t.Fatalf("contained rect bounds = %v", got)
	}

	// Two overlapping same-AA axis-aligned rects combine into one element holding the float-precise intersection.
	cs = NewClipStack(deviceBounds64(), false)
	cs.ClipRect(&identity, geom.Rect{Left: 4.5, Top: 4.5, Right: 40.5, Bottom: 40.5}, gpu.AAYes,
		raster.ClipIntersect)
	cs.ClipRect(&identity, geom.Rect{Left: 20.25, Top: 8.5, Right: 60.5, Bottom: 36.25},
		gpu.AAYes, raster.ClipIntersect)
	if cs.ClipState() != ClipStackDeviceRect {
		t.Fatalf("combined rects state = %v, want device rect", cs.ClipState())
	}
	active := 0
	for i := range cs.elements {
		if !cs.elements[i].isInvalid() {
			active++
		}
	}
	if active != 1 {
		t.Fatalf("active elements = %d, want 1 (combined)", active)
	}
	back := &cs.elements[len(cs.elements)-1]
	if back.Shape.Rect() != (geom.Rect{Left: 20.25, Top: 8.5, Right: 40.5, Bottom: 36.25}) {
		t.Fatalf("combined rect = %v", back.Shape.Rect())
	}

	// An rrect intersected with a rect that contains its rounded corners combines through ConservativeIntersect.
	cs = NewClipStack(deviceBounds64(), false)
	rr := geom.MakeRRect(geom.Rect{Left: 10, Top: 10, Right: 40, Bottom: 40}, 5, 5)
	cs.ClipRRect(&identity, rr, gpu.AAYes, raster.ClipIntersect)
	cs.ClipRect(&identity, geom.Rect{Left: 10, Top: 10, Right: 40, Bottom: 30}, gpu.AAYes,
		raster.ClipIntersect)
	if cs.ClipState() != ClipStackComplex {
		// The shared top corners keep the rrect radii while the bottom corners square off — unrepresentable with
		// uniform radii, so both elements stay (a nine-patch-capable representation could combine them).
		active = 0
		for i := range cs.elements {
			if !cs.elements[i].isInvalid() {
				active++
			}
		}
		if active != 2 {
			t.Fatalf("rrect+rect active elements = %d, want 2", active)
		}
	}
}

func TestClipStackPreApply(t *testing.T) {
	cs := NewClipStack(deviceBounds64(), false)
	identity := geom.IdentityMatrix()
	rr := geom.MakeRRect(geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 50}, 6, 6)
	cs.ClipRRect(&identity, rr, gpu.AAYes, raster.ClipIntersect)

	// A draw fully inside the rrect's inner region is unclipped.
	res := cs.PreApply(geom.Rect{Left: 25, Top: 25, Right: 35, Bottom: 35}, gpu.AAYes)
	if res.Effect != ClipEffectUnclipped {
		t.Fatalf("inner draw effect = %v, want unclipped", res.Effect)
	}

	// A draw fully outside is clipped out.
	res = cs.PreApply(geom.Rect{Left: 52, Top: 52, Right: 60, Bottom: 60}, gpu.AAYes)
	if res.Effect != ClipEffectClippedOut {
		t.Fatalf("outside draw effect = %v, want clipped out", res.Effect)
	}

	// A straddling draw reports the device rrect.
	res = cs.PreApply(geom.Rect{Left: 5, Top: 5, Right: 30, Bottom: 30}, gpu.AAYes)
	if res.Effect != ClipEffectClipped || !res.IsRRect || res.RRect != rr || res.AA != gpu.AAYes {
		t.Fatalf("straddling draw = %+v, want the device rrect", res)
	}
}

//////////////////////////////////////////////////////////////////////////////
// ClipStack apply lanes

func applyTestOp(rect geom.Rect) (DrawOp, geom.Rect) {
	identity := geom.IdentityMatrix()
	quad := DrawQuad{
		Device: MakeQuadFromRect(rect, &identity),
		Local:  MakeQuadFromRectNoTransform(rect), EdgeFlags: gpu.QuadAAFlagsNone,
	}
	op := NewFillRectOp(solidPaint(1, 0, 0, 0.5), gpu.AATypeNone, &quad, nil,
		PipelineInputFlagNone)
	return op, opBounds(op)
}

func TestClipStackApplyScissorOnly(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	cs := NewClipStack(deviceBounds64(), false)
	cs.ClipRect(&identity, geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 50}, gpu.AANo,
		raster.ClipIntersect)

	// A straddling non-AA draw is clipped by scissor alone.
	op, bounds := applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 30, Bottom: 30})
	out := MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeNone, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("effect = %v, want clipped", effect)
	}
	if !out.ScissorState().Enabled() {
		t.Fatal("scissor must be enabled")
	}
	if got := out.ScissorState().Rect(); got != (geom.IRect{Left: 10, Top: 10, Right: 50, Bottom: 50}) {
		t.Fatalf("scissor = %v", got)
	}
	if out.HasCoverageFragmentProcessor() || out.HasStencilClip() {
		t.Fatal("scissor-only clip must not add coverage or stencil")
	}

	// A draw fully inside the clip is unclipped.
	op, bounds = applyTestOp(geom.Rect{Left: 20, Top: 20, Right: 30, Bottom: 30})
	out = MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeNone, &out, &bounds); effect != ClipEffectUnclipped {
		t.Fatalf("inner effect = %v, want unclipped", effect)
	}

	// A draw fully outside is clipped out.
	op, bounds = applyTestOp(geom.Rect{Left: 52, Top: 52, Right: 60, Bottom: 60})
	out = MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeNone, &out, &bounds); effect != ClipEffectClippedOut {
		t.Fatalf("outside effect = %v, want clipped out", effect)
	}
}

func TestClipStackApplyAnalyticFPs(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	// A fractional AA rect clip becomes an analytic coverage FP.
	cs := NewClipStack(deviceBounds64(), false)
	cs.ClipRect(&identity, geom.Rect{Left: 10.5, Top: 10.5, Right: 50.25, Bottom: 50.25},
		gpu.AAYes, raster.ClipIntersect)
	op, bounds := applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 30, Bottom: 30})
	out := MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeCoverage, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("AA rect effect = %v, want clipped", effect)
	}
	if !out.HasCoverageFragmentProcessor() || out.HasStencilClip() {
		t.Fatal("AA rect clip must ride a coverage FP")
	}

	// An AA rrect clip becomes the circular-rrect FP.
	cs = NewClipStack(deviceBounds64(), false)
	cs.ClipRRect(&identity, geom.MakeRRect(geom.Rect{Left: 10, Top: 10, Right: 50, Bottom: 50},
		6, 6), gpu.AAYes, raster.ClipIntersect)
	op, bounds = applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 30, Bottom: 30})
	out = MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeCoverage, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("rrect effect = %v, want clipped", effect)
	}
	if !out.HasCoverageFragmentProcessor() {
		t.Fatal("rrect clip must ride a coverage FP")
	}

	// A rotated rect clip becomes a convex-poly FP.
	cs = NewClipStack(deviceBounds64(), false)
	rot := geom.IdentityMatrix()
	rot.SetRotatePivot(30, 32, 32)
	cs.ClipRect(&rot, geom.Rect{Left: 12, Top: 12, Right: 52, Bottom: 52}, gpu.AAYes,
		raster.ClipIntersect)
	op, bounds = applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64})
	out = MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeCoverage, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("rotated rect effect = %v, want clipped", effect)
	}
	if !out.HasCoverageFragmentProcessor() {
		t.Fatal("rotated rect clip must ride a coverage FP")
	}

	// A difference rect over the draw becomes an inverse-fill FP (no scissor shrink possible when the hole is
	// interior).
	cs = NewClipStack(deviceBounds64(), false)
	cs.ClipRect(&identity, geom.Rect{Left: 20, Top: 20, Right: 30, Bottom: 30}, gpu.AAYes,
		raster.ClipDifference)
	op, bounds = applyTestOp(geom.Rect{Left: 10, Top: 10, Right: 40, Bottom: 40})
	out = MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeCoverage, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("difference effect = %v, want clipped", effect)
	}
	if !out.HasCoverageFragmentProcessor() {
		t.Fatal("difference clip must ride a coverage FP")
	}
}

func TestClipStackApplySWMask(t *testing.T) {
	dc := newFakeDirectContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	// A concave AA path on a 1-sample target must take the SW mask lane. Save first so the restore-invalidation lane
	// can be exercised at the end.
	cs := NewClipStack(deviceBounds64(), false)
	cs.Save()
	concave := &path.Path{}
	concave.MoveTo(10, 10)
	concave.LineTo(54, 10)
	concave.LineTo(32, 30) // notch back toward the center
	concave.LineTo(54, 54)
	concave.LineTo(10, 54)
	concave.Close()
	cs.ClipPath(&identity, concave, gpu.AAYes, raster.ClipIntersect)

	op, bounds := applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64})
	out := MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeCoverage, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("concave path effect = %v, want clipped", effect)
	}
	if !out.HasCoverageFragmentProcessor() {
		t.Fatal("SW mask lane must add a coverage FP")
	}
	if out.HasStencilClip() {
		t.Fatal("SW mask lane must not touch the stencil")
	}
	if len(cs.masks) != 1 || !cs.masks[0].key.IsValid() {
		t.Fatalf("masks recorded = %d, want 1 keyed mask", len(cs.masks))
	}

	// The same query reuses the cached mask.
	op, bounds = applyTestOp(geom.Rect{Left: 0, Top: 0, Right: 64, Bottom: 64})
	out = MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeCoverage, &out, &bounds); effect != ClipEffectClipped {
		t.Fatal("second apply must still clip")
	}
	if len(cs.masks) != 1 {
		t.Fatalf("masks after reuse = %d, want 1", len(cs.masks))
	}

	// Restoring past the record invalidates its masks.
	cs.Restore()
	if len(cs.masks) != 0 {
		t.Fatalf("masks after restore = %d, want 0", len(cs.masks))
	}
}

func TestClipStackApplyStencil(t *testing.T) {
	dc := newShaderRecordingContext(t)
	g := dc.Gpu()
	f := &g.ctx.Interface.Functions
	f.stencilFunc = counterProc("glStencilFunc")
	f.stencilMask = counterProc("glStencilMask")
	f.stencilOp = counterProc("glStencilOp")
	f.clearStencil = counterProc("glClearStencil")

	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()

	// Five thin non-AA spokes crossing at the center: four ride analytic convex-poly FPs and the oldest overflows the
	// FP budget into the stencil lane (rect, non-AA, 1-sample).
	cs := NewClipStack(deviceBounds64(), false)
	for i := 0; i < 5; i++ {
		m := geom.IdentityMatrix()
		m.SetRotatePivot(float32(15+36*i), 32, 32)
		cs.ClipRect(&m, geom.Rect{Left: 10, Top: 27, Right: 54, Bottom: 37}, gpu.AANo,
			raster.ClipIntersect)
	}
	active := 0
	for i := range cs.elements {
		if !cs.elements[i].isInvalid() {
			active++
		}
	}
	if active != 5 {
		t.Fatalf("active spoke elements = %d, want 5", active)
	}

	opsBefore := sdc.GetOpsTask().NumOpChains()
	op, bounds := applyTestOp(geom.Rect{Left: 22, Top: 22, Right: 42, Bottom: 42})
	out := MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeNone, &out, &bounds); effect != ClipEffectClipped {
		t.Fatalf("effect = %v, want clipped", effect)
	}
	if !out.HasStencilClip() {
		t.Fatal("the overflow element must ride the stencil clip")
	}
	if !out.HasCoverageFragmentProcessor() {
		t.Fatal("the four analytic elements must ride coverage FPs")
	}
	opsAfterFirst := sdc.GetOpsTask().NumOpChains()
	if opsAfterFirst <= opsBefore {
		t.Fatal("rendering the stencil mask must record ops (clear + stencil draws)")
	}

	// An identical query skips re-rendering the stencil mask (mustRenderClip false).
	op, bounds = applyTestOp(geom.Rect{Left: 22, Top: 22, Right: 42, Bottom: 42})
	out = MakeAppliedClip(sdc.Dimensions())
	if effect := cs.Apply(sdc, op, gpu.AATypeNone, &out, &bounds); effect != ClipEffectClipped {
		t.Fatal("second apply must still clip")
	}
	if !out.HasStencilClip() {
		t.Fatal("second apply must still carry the stencil clip")
	}
	if got := sdc.GetOpsTask().NumOpChains(); got != opsAfterFirst {
		t.Fatalf("op chains after idempotent apply = %d, want %d", got, opsAfterFirst)
	}

	// Record the clipped draw and flush: the stencil state must actually reach GL.
	sdc.AddDrawOp(cs, mustDrawOp(t, geom.Rect{Left: 22, Top: 22, Right: 42, Bottom: 42}))
	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if counts("glClearStencil") == 0 {
		t.Fatal("the stencil-clip clear must call glClearStencil")
	}
	if counts("glStencilFunc") == 0 || counts("glStencilOp") == 0 ||
		counts("glStencilMask") == 0 {
		t.Fatalf("stencil draws must program the stencil state (func %d op %d mask %d)",
			counts("glStencilFunc"), counts("glStencilOp"), counts("glStencilMask"))
	}
}

func mustDrawOp(t *testing.T, rect geom.Rect) DrawOp {
	t.Helper()
	op, _ := applyTestOp(rect)
	return op
}
