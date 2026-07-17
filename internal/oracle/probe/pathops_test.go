// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package probe

import (
	"fmt"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/pathops"
	"github.com/richardwilkes/canvas/raster"
)

// Faithful pathops probes. The port's boolean engine is now the faithful Skia's OpSegment machinery
// (runOp/runSimplify), so it keeps curves exactly as Skia does; results are compared by area sampling: both sides'
// result paths are rasterized through the port's (oracle-verified) rasterizer over the operands' joint bounds and the
// coverage masks must agree except in a thin sub-pixel band around the boundaries, where AA edge placement and the
// last-ULP conic-control-point noise from the SVG round-trip of the port's result legitimately differ. Success flags
// and result fill types (the gOpInverse/ gOutInverse algebra) must agree exactly. The fast paths that bypass the engine
// (rect intersect, empty operands with convex survivors) are compared bit-exactly through the verified SVG emitters.

const (
	pathOpsMaskRes = 256 // pixels across the sampling window
	// A boundary pixel legitimately disagrees by full coverage only in a sub-pixel band around curved edges, where the
	// two sides' AA edge placement and the SVG-round-tripped conic controls differ: allow a small fraction of large
	// diffs and a small mean. Measured worst case on the darwin leg with the faithful engine: mean 0.48, max delta 103,
	// fracOver96 3e-5 — the caps carry ~2x / ~160x margin.
	pathOpsMaxFracOver96 = 0.005
	pathOpsMaxMean       = 1.0
)

var pathOpNames = map[int]string{0: "diff", 1: "sect", 2: "union", 3: "xor", 4: "revdiff"}

// pathOpsShape is one corpus entry, built identically on both sides via scenario.PathSpec.
type pathOpsShape struct {
	spec *scenario.PathSpec
	name string
}

func pathOpsStarSpec(ft path.FillType) *scenario.PathSpec {
	// Five-pointed self-crossing star around (100, 100).
	s := scenario.NewPathSpec().SetFill(ft)
	pts := [5][2]float32{{100, 20}, {147, 165}, {24, 75}, {176, 75}, {53, 165}}
	s.MoveTo(pts[0][0], pts[0][1])
	for _, p := range pts[1:] {
		s.LineTo(p[0], p[1])
	}
	s.Close()
	return s
}

func pathOpsSingles() []pathOpsShape {
	return []pathOpsShape{
		{name: "rect", spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(20, 20, 160, 140), geom.DirectionCW)},
		{name: "circle", spec: scenario.NewPathSpec().AddCircle(100, 90, 70, geom.DirectionCW)},
		{name: "oval", spec: scenario.NewPathSpec().AddOval(geom.RectLTRB(10, 40, 190, 150), geom.DirectionCCW)},
		{name: "rrect", spec: scenario.NewPathSpec().AddRoundRect(geom.RectLTRB(30, 30, 170, 130), 25, 18, geom.DirectionCW)},
		{name: "star-winding", spec: pathOpsStarSpec(path.FillWinding)},
		{name: "star-evenodd", spec: pathOpsStarSpec(path.FillEvenOdd)},
		{name: "star-inv-winding", spec: pathOpsStarSpec(path.FillInverseWinding)},
		{name: "two-rects-winding", spec: scenario.NewPathSpec().
			AddRect(geom.RectLTRB(20, 20, 110, 110), geom.DirectionCW).
			AddRect(geom.RectLTRB(70, 60, 180, 150), geom.DirectionCW)},
		{name: "two-rects-evenodd", spec: scenario.NewPathSpec().SetFill(path.FillEvenOdd).
			AddRect(geom.RectLTRB(20, 20, 110, 110), geom.DirectionCW).
			AddRect(geom.RectLTRB(70, 60, 180, 150), geom.DirectionCW)},
		{name: "nested-same-dir", spec: scenario.NewPathSpec().
			AddRect(geom.RectLTRB(20, 20, 180, 150), geom.DirectionCW).
			AddRect(geom.RectLTRB(60, 55, 140, 115), geom.DirectionCW)},
		{name: "nested-opposite", spec: scenario.NewPathSpec().
			AddRect(geom.RectLTRB(20, 20, 180, 150), geom.DirectionCW).
			AddRect(geom.RectLTRB(60, 55, 140, 115), geom.DirectionCCW)},
		{name: "cubic-blob", spec: scenario.NewPathSpec().
			MoveTo(30, 100).
			CubicTo(60, -40, 140, 240, 170, 100).
			CubicTo(140, 40, 60, 160, 30, 100).
			Close()},
		{name: "conic-wedge", spec: scenario.NewPathSpec().
			MoveTo(100, 90).
			ConicTo(190, 90, 190, 170, 0.707107).
			LineTo(100, 170).
			Close()},
		{name: "figure8-lines", spec: scenario.NewPathSpec().
			MoveTo(20, 20).LineTo(180, 150).LineTo(20, 150).LineTo(180, 20).Close()},
		{name: "line-degenerate", spec: scenario.NewPathSpec().MoveTo(20, 20).LineTo(180, 140)},
		{name: "empty", spec: scenario.NewPathSpec()},
	}
}

type pathOpsPair struct {
	a    *scenario.PathSpec
	b    *scenario.PathSpec
	name string
}

func pathOpsPairs() []pathOpsPair {
	singles := pathOpsSingles()
	byName := map[string]*scenario.PathSpec{}
	for _, s := range singles {
		byName[s.name] = s.spec
	}
	rectB := scenario.NewPathSpec().AddRect(geom.RectLTRB(90, 70, 230, 200), geom.DirectionCW)
	return []pathOpsPair{
		{name: "rect-vs-rectB", a: byName["rect"], b: rectB},
		{
			name: "rect-vs-disjoint", a: byName["rect"],
			b: scenario.NewPathSpec().AddRect(geom.RectLTRB(200, 20, 300, 140), geom.DirectionCW),
		},
		{
			name: "rect-vs-shared-edge", a: byName["rect"],
			b: scenario.NewPathSpec().AddRect(geom.RectLTRB(160, 20, 300, 140), geom.DirectionCW),
		},
		{
			name: "rect-vs-corner-touch", a: byName["rect"],
			b: scenario.NewPathSpec().AddRect(geom.RectLTRB(160, 140, 300, 260), geom.DirectionCW),
		},
		{
			name: "rect-vs-nested", a: byName["rect"],
			b: scenario.NewPathSpec().AddRect(geom.RectLTRB(50, 50, 130, 110), geom.DirectionCW),
		},
		{name: "rect-vs-identical", a: byName["rect"], b: byName["rect"]},
		{name: "rect-vs-circle", a: byName["rect"], b: byName["circle"]},
		{
			name: "circle-vs-circleB", a: byName["circle"],
			b: scenario.NewPathSpec().AddCircle(160, 90, 70, geom.DirectionCW),
		},
		{
			name: "circle-vs-concentric", a: byName["circle"],
			b: scenario.NewPathSpec().AddCircle(100, 90, 35, geom.DirectionCCW),
		},
		{name: "circle-vs-oval", a: byName["circle"], b: byName["oval"]},
		{name: "star-vs-rect", a: byName["star-winding"], b: byName["rect"]},
		{name: "star-vs-circle", a: byName["star-evenodd"], b: byName["circle"]},
		{name: "cubic-vs-conic", a: byName["cubic-blob"], b: byName["conic-wedge"]},
		{name: "cubic-vs-rrect", a: byName["cubic-blob"], b: byName["rrect"]},
		{name: "empty-vs-star", a: byName["empty"], b: byName["star-winding"]},
		{name: "line-vs-rect", a: byName["line-degenerate"], b: byName["rect"]},
		{name: "inv-star-vs-rect", a: byName["star-inv-winding"], b: byName["rect"]},
		{name: "inv-star-vs-inv-star", a: byName["star-inv-winding"], b: byName["star-inv-winding"]},
		{name: "evenodd-rects-vs-circle", a: byName["two-rects-evenodd"], b: byName["circle"]},
		{name: "nested-vs-figure8", a: byName["nested-opposite"], b: byName["figure8-lines"]},
	}
}

// pathOpsWindow returns the sampling window: the operands' joint bounds outset by 8%.
func pathOpsWindow(paths ...*path.Path) (geom.Rect, bool) {
	var bounds geom.Rect
	first := true
	for _, p := range paths {
		if p == nil || p.IsEmpty() {
			continue
		}
		b := p.Bounds()
		if first {
			bounds, first = b, false
			continue
		}
		bounds.Left = min(bounds.Left, b.Left)
		bounds.Top = min(bounds.Top, b.Top)
		bounds.Right = max(bounds.Right, b.Right)
		bounds.Bottom = max(bounds.Bottom, b.Bottom)
	}
	if first {
		return geom.Rect{}, false
	}
	pad := 0.08 * max(bounds.Right-bounds.Left, bounds.Bottom-bounds.Top)
	if pad <= 0 {
		pad = 1
	}
	return geom.RectLTRB(bounds.Left-pad, bounds.Top-pad, bounds.Right+pad, bounds.Bottom+pad), true
}

// renderPathOpsMask rasterizes a result path over the window into a res x res A8 coverage mask through the port's
// rasterizer (fill rule and inverse-ness honored).
func renderPathOpsMask(p *path.Path, window geom.Rect, res int32) []uint8 {
	scale := float32(res) / max(window.Right-window.Left, window.Bottom-window.Top)
	var m geom.Matrix
	m.SetScaleTranslate(scale, scale, -window.Left*scale, -window.Top*scale)
	dev := &path.Path{}
	p.TransformTo(&m, dev)
	mask := &raster.Mask{
		Image:    make([]uint8, int(res)*int(res)),
		Bounds:   geom.IRectWH(res, res),
		RowBytes: res,
	}
	maskfilter.RenderPathIntoMask(mask, dev, true)
	return mask.Image
}

// comparePathOpsResults area-samples the two sides' results and enforces the fill-type algebra. The oracle's result
// arrives as a port path already — replayed from the frozen fixtures (see ref_test.go), which store its geometry rather
// than a rasterization, so both sides here go through the *current* rasterizer.
func comparePathOpsResults(t *testing.T, label string, portResult, oracleResult *path.Path, inputs ...*path.Path) {
	t.Helper()
	if got, want := int(portResult.FillType()), int(oracleResult.FillType()); got != want {
		t.Errorf("%s: fill type port=%d oracle=%d", label, got, want)
	}
	window, ok := pathOpsWindow(inputs...)
	if !ok {
		// Both operands empty: both results must be empty too.
		if !portResult.IsEmpty() || !oracleResult.IsEmpty() {
			t.Errorf("%s: empty inputs produced non-empty result (port=%v oracle=%v)",
				label, !portResult.IsEmpty(), !oracleResult.IsEmpty())
		}
		return
	}
	pm := renderPathOpsMask(portResult, window, pathOpsMaskRes)
	om := renderPathOpsMask(oracleResult, window, pathOpsMaskRes)
	var sumAbs, over96 int
	maxAbs := 0
	for i := range pm {
		d := int(pm[i]) - int(om[i])
		if d < 0 {
			d = -d
		}
		sumAbs += d
		if d > 96 {
			over96++
		}
		if d > maxAbs {
			maxAbs = d
		}
	}
	n := len(pm)
	mean := float64(sumAbs) / float64(n)
	frac := float64(over96) / float64(n)
	if mean > pathOpsMaxMean || frac > pathOpsMaxFracOver96 {
		t.Errorf("%s: masks diverge: mean=%.3f (cap %v) fracOver96=%.5f (cap %v) max=%d",
			label, mean, pathOpsMaxMean, frac, pathOpsMaxFracOver96, maxAbs)
	} else if testing.Verbose() && (mean > 0.2 || frac > 0) {
		t.Logf("%s: mean=%.3f fracOver96=%.5f max=%d", label, mean, frac, maxAbs)
	}
}

func TestPathOpsOpProbes(t *testing.T) {
	for _, pair := range pathOpsPairs() {
		for op := 0; op <= 4; op++ {
			label := fmt.Sprintf("%s %s", pair.name, pathOpNames[op])
			ga, gb := pair.a.BuildGo(), pair.b.BuildGo()
			ores, cOK := refPathOpsOp(pair.a, pair.b, op)
			gres, gOK := pathops.Op(ga, gb, pathops.PathOp(op))
			if gOK != cOK {
				t.Errorf("%s: success port=%v oracle=%v", label, gOK, cOK)
			} else if gOK {
				comparePathOpsResults(t, label, gres, ores, ga, gb)
			}
		}
	}
}

func TestPathOpsSimplifyProbes(t *testing.T) {
	for _, s := range pathOpsSingles() {
		label := "simplify " + s.name
		gp := s.spec.BuildGo()
		ores, cOK := refPathOpsSimplify(s.spec)
		gres, gOK := pathops.Simplify(gp)
		if gOK != cOK {
			t.Errorf("%s: success port=%v oracle=%v", label, gOK, cOK)
		} else if gOK {
			comparePathOpsResults(t, label, gres, ores, gp)
		}
	}
}

func TestPathOpsBuilderProbes(t *testing.T) {
	scripts := []struct {
		name string
		adds []struct {
			spec *scenario.PathSpec
			op   int
		}
	}{
		{name: "union-chain", adds: []struct {
			spec *scenario.PathSpec
			op   int
		}{
			{spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(0, 0, 60, 60), geom.DirectionCW), op: 2},
			{spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(40, 20, 120, 80), geom.DirectionCW), op: 2},
			{spec: scenario.NewPathSpec().AddCircle(90, 90, 40, geom.DirectionCW), op: 2},
		}},
		{name: "intersect-first", adds: []struct {
			spec *scenario.PathSpec
			op   int
		}{
			{spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(0, 0, 100, 100), geom.DirectionCW), op: 1},
		}},
		{name: "mixed-ops", adds: []struct {
			spec *scenario.PathSpec
			op   int
		}{
			{spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(0, 0, 100, 100), geom.DirectionCW), op: 2},
			{spec: scenario.NewPathSpec().AddCircle(100, 50, 40, geom.DirectionCW), op: 2},
			{spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(30, 30, 70, 70), geom.DirectionCW), op: 0},
			{spec: pathOpsStarSpec(path.FillWinding), op: 3},
		}},
		{name: "single-union-star", adds: []struct {
			spec *scenario.PathSpec
			op   int
		}{
			{spec: pathOpsStarSpec(path.FillWinding), op: 2},
		}},
	}
	for _, script := range scripts {
		var gb pathops.Builder
		var gInputs []*path.Path
		var adds []builderAdd
		for _, add := range script.adds {
			gp := add.spec.BuildGo()
			gb.Add(gp, pathops.PathOp(add.op))
			gInputs = append(gInputs, gp)
			adds = append(adds, builderAdd{spec: add.spec, op: add.op})
		}
		ores, cOK := refPathOpsBuilder(adds)
		gres, gOK := gb.Resolve()
		if gOK != cOK {
			t.Errorf("builder %s: success port=%v oracle=%v", script.name, gOK, cOK)
		} else if gOK {
			comparePathOpsResults(t, "builder "+script.name, gres, ores, gInputs...)
		}
	}
}

// TestPathOpsBuilderConvexUnionExact pins the all-union optimization's convex lanes bit-exactly against real Skia. A
// single convex union operand runs Skia's OpBuilder::resolve's all-union branch through Simplify's convex fast path (a
// clone), fixWinding's one-contour fast path (reverse iff the direction is CW), and Simplify's convex fast path again —
// all exact geometry, no engine, so the port must match Skia's output point stream bit-for-bit. This is the
// structural-parity evidence for the faithful fold (an earlier one produced the operand's own orientation, which
// diverged from Skia's fix-winding reversal). Both the reversing (CW) and non-reversing (CCW) fixWinding branches are
// exercised.
func TestPathOpsBuilderConvexUnionExact(t *testing.T) {
	cases := []struct {
		spec *scenario.PathSpec
		name string
	}{
		{name: "rect-cw", spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW)},
		{name: "rect-ccw", spec: scenario.NewPathSpec().AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCCW)},
		{name: "oval-cw", spec: scenario.NewPathSpec().AddOval(geom.RectLTRB(10, 20, 110, 100), geom.DirectionCW)},
		{name: "oval-ccw", spec: scenario.NewPathSpec().AddOval(geom.RectLTRB(10, 20, 110, 100), geom.DirectionCCW)},
	}
	for _, tc := range cases {
		gp := tc.spec.BuildGo()
		var gb pathops.Builder
		gb.Add(gp, pathops.Union)
		cPts, cFill, cOK := refPathOpsBuilderUnionExact(tc.spec)
		gres, gOK := gb.Resolve()
		if gOK != cOK {
			t.Errorf("builder-exact %s: success port=%v oracle=%v", tc.name, gOK, cOK)
		} else if gOK {
			pathOpsExact(t, "builder-exact "+tc.name, gres, cPts, cFill)
		}
	}
}

// pathOpsExact compares a result path against the oracle's bit-exactly by raw point stream and fill type (the fast-path
// lanes copy or construct geometry identically on both sides). The oracle's points are frozen straight off the C path —
// see the note in ref_ops_test.go on why they do not travel through a rebuilt path.
func pathOpsExact(t *testing.T, label string, gres *path.Path, cPts []geom.Point, cFill int) {
	t.Helper()
	gPts := make([]geom.Point, gres.CountPoints())
	gres.Points(gPts)
	if len(gPts) != len(cPts) || !ptsEq(gPts, cPts) {
		t.Errorf("%s: point streams differ (port %d pts, oracle %d pts)", label, len(gPts), len(cPts))
	}
	if got, want := int(gres.FillType()), cFill; got != want {
		t.Errorf("%s: fill port=%d oracle=%d", label, got, want)
	}
}

// TestPathOpsExactFastPaths pins the lanes that bypass both boolean engines — the rect-intersect fast path and the
// empty-operand handling with convex survivors — bit-exactly by point stream.
func TestPathOpsExactFastPaths(t *testing.T) {
	rectA := scenario.NewPathSpec().AddRect(geom.RectLTRB(10, 20, 110, 100), geom.DirectionCW)
	rectB := scenario.NewPathSpec().AddRect(geom.RectLTRB(50, 40, 200, 160), geom.DirectionCW)
	rectDisjoint := scenario.NewPathSpec().AddRect(geom.RectLTRB(300, 300, 400, 400), geom.DirectionCW)
	oval := scenario.NewPathSpec().AddOval(geom.RectLTRB(10, 20, 110, 100), geom.DirectionCW)
	empty := scenario.NewPathSpec()

	check := func(label string, aSpec, bSpec *scenario.PathSpec, op int, aFill, bFill path.FillType) {
		ga, gb := aSpec.BuildGo(), bSpec.BuildGo()
		ga.SetFillType(aFill)
		gb.SetFillType(bFill)
		cPts, cFill, cOK := refPathOpsOpExact(aSpec, bSpec, op, aFill, bFill)
		gres, gOK := pathops.Op(ga, gb, pathops.PathOp(op))
		if gOK != cOK {
			t.Errorf("%s: success port=%v oracle=%v", label, gOK, cOK)
		} else if gOK {
			pathOpsExact(t, label, gres, cPts, cFill)
		}
	}

	// Rect-intersect fast path, including the inverse remaps that reach it.
	check("rect-sect", rectA, rectB, 1, path.FillWinding, path.FillWinding)
	check("rect-sect-disjoint", rectA, rectDisjoint, 1, path.FillWinding, path.FillWinding)
	check("rect-diff-inv", rectA, rectB, 0, path.FillWinding, path.FillInverseWinding)
	check("rect-union-inv-inv", rectA, rectB, 2, path.FillInverseWinding, path.FillInverseWinding)
	check("rect-revdiff-inv", rectA, rectB, 4, path.FillInverseWinding, path.FillWinding)

	// Empty operands with convex survivors: the copy (with fill toggling) then Simplify's convex lane.
	for op := 0; op <= 4; op++ {
		check(fmt.Sprintf("empty-first-%s", pathOpNames[op]), empty, oval, op, path.FillWinding, path.FillWinding)
		check(fmt.Sprintf("empty-second-%s", pathOpNames[op]), oval, empty, op, path.FillWinding, path.FillWinding)
		check(fmt.Sprintf("empty-first-inv-%s", pathOpNames[op]), empty, oval, op,
			path.FillWinding, path.FillInverseEvenOdd)
	}

	// Simplify's convex fast path and trivial collapse.
	for _, tc := range []struct {
		spec *scenario.PathSpec
		name string
		fill path.FillType
	}{
		{name: "oval", spec: oval, fill: path.FillWinding},
		{name: "oval-inv", spec: oval, fill: path.FillInverseWinding},
		{name: "trivial-line", spec: scenario.NewPathSpec().MoveTo(10, 10).LineTo(90, 90), fill: path.FillWinding},
	} {
		gp := tc.spec.BuildGo()
		gp.SetFillType(tc.fill)
		cPts, cFill, cOK := refPathOpsSimplifyExact(tc.spec, tc.fill)
		gres, gOK := pathops.Simplify(gp)
		if gOK != cOK {
			t.Errorf("simplify %s: success port=%v oracle=%v", tc.name, gOK, cOK)
		} else if gOK {
			pathOpsExact(t, "simplify "+tc.name, gres, cPts, cFill)
		}
	}
}
