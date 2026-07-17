// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the interior check that gates coincident-run inference (coincidentRunIsReal in opcoincidence_resolve.go).
// addEndMovedSpans and addMissing infer a run from matched endpoints alone; where a curve meets the opposite segment at
// two points without coinciding between them -- a chord and the arc it subtends -- that inference invents a run and Op
// drops the region the false run walls off. The bowtie family below is the smallest shape that produces the pairing.

package pathops

import (
	"testing"

	"github.com/richardwilkes/canvas/path"
)

// bowtieFromCubic builds a self-crossing contour: the cubic plus the chord closing it back to the start.
func bowtieFromCubic(c dCubic) *path.Path {
	p := path.New()
	p.MoveTo(float32(c.pts[0].x), float32(c.pts[0].y))
	p.CubicTo(float32(c.pts[1].x), float32(c.pts[1].y), float32(c.pts[2].x), float32(c.pts[2].y),
		float32(c.pts[3].x), float32(c.pts[3].y))
	p.Close()
	p.SetFillType(path.FillWinding)
	return p
}

// evalPathOp computes the expected inside-ness of a point from the operands' own inside-ness.
func evalPathOp(op PathOp, inA, inB bool) bool {
	switch op {
	case Intersect:
		return inA && inB
	case Union:
		return inA || inB
	case Difference:
		return inA && !inB
	case XOR:
		return inA != inB
	default:
		return false
	}
}

// stableContains reports whether p contains (x,y), and whether every neighbor within delta agrees. Samples that
// straddle a boundary are ambiguous -- the result path's boundary is a different curve representation than the
// operands' -- so callers skip them rather than compare them.
func stableContains(p *path.Path, x, y, delta float32) (contains, stable bool) {
	contains = p.Contains(x, y)
	for _, offset := range [8][2]float32{
		{delta, 0},
		{-delta, 0},
		{0, delta},
		{0, -delta},
		{delta, delta},
		{-delta, -delta},
		{delta, -delta},
		{-delta, delta},
	} {
		if p.Contains(x+offset[0], y+offset[1]) != contains {
			return contains, false
		}
	}
	return contains, true
}

// TestOpCoincidentChordSubtendingArc checks the reduced case directly: b is base subdivided over [0.3,0.7], so b lies
// entirely inside a and Intersect must return b. Both are bowties crossing at (50,50), and b's cubic is exactly
// coincident with a subsegment of a's. (40,40) is inside both operands, so it must be inside the intersection.
func TestOpCoincidentChordSubtendingArc(t *testing.T) {
	base := dCubic{pts: [4]dPoint{{x: 0, y: 50}, {x: 40, y: -20}, {x: 60, y: 120}, {x: 100, y: 50}}}
	a := bowtieFromCubic(base)
	b := bowtieFromCubic(base.subDivide(0.3, 0.7))
	if !a.Contains(40, 40) || !b.Contains(40, 40) {
		t.Fatal("(40,40) must be inside both operands for this case to mean anything")
	}
	result, ok := Op(a, b, Intersect)
	if !ok {
		t.Fatal("Op(a, b, Intersect) failed")
	}
	if !result.Contains(40, 40) {
		t.Error("Intersect dropped the lower lobe: (40,40) is inside both operands but not the result")
	}
}

// TestOpCoincidentSelfCrossingSweep sweeps the bowtie family -- base curves crossed with subranges of themselves, in
// both operand orders, over every op -- and checks each result by grid-sampling its fill against the boolean of the
// operands' own Contains. Every case here is coincident and self-crossing, the combination that made the endpoint-only
// run inference fire.
func TestOpCoincidentSelfCrossingSweep(t *testing.T) {
	bases := []dCubic{
		{pts: [4]dPoint{{x: 0, y: 50}, {x: 40, y: -20}, {x: 60, y: 120}, {x: 100, y: 50}}},
		{pts: [4]dPoint{{x: 0, y: 50}, {x: 30, y: -40}, {x: 70, y: 140}, {x: 100, y: 50}}},
		{pts: [4]dPoint{{x: 10, y: 40}, {x: 50, y: -30}, {x: 50, y: 130}, {x: 90, y: 60}}},
		{pts: [4]dPoint{{x: 0, y: 60}, {x: 60, y: -10}, {x: 40, y: 110}, {x: 100, y: 40}}},
	}
	subranges := [][2]float64{
		{0.1, 0.9},
		{0.2, 0.8},
		{0.3, 0.7},
		{0.4, 0.6},
		{0.25, 0.75},
		{0.15, 0.85},
		{0.3, 0.9},
		{0.1, 0.7},
		{0.35, 0.65},
		{0.2, 0.6},
		{0.4, 0.8},
		{0.45, 0.55},
	}
	ops := map[PathOp]string{Intersect: "Intersect", Union: "Union", Difference: "Difference", XOR: "XOR"}
	const delta = 0.35
	for baseIndex, base := range bases {
		for _, sub := range subranges {
			for _, reversed := range []bool{false, true} {
				a := bowtieFromCubic(base)
				b := bowtieFromCubic(base.subDivide(sub[0], sub[1]))
				if reversed {
					a, b = b, a
				}
				for op, opName := range ops {
					result, ok := Op(a, b, op)
					if !ok {
						t.Errorf("base %d sub %v reversed=%v %s: Op failed", baseIndex, sub, reversed, opName)
						continue
					}
					wrong := 0
					for gridX := 0; gridX <= 60; gridX++ {
						for gridY := 0; gridY <= 60; gridY++ {
							x := float32(gridX)*1.8 - 5
							y := float32(gridY)*1.8 - 5
							inA, stableA := stableContains(a, x, y, delta)
							inB, stableB := stableContains(b, x, y, delta)
							got, stableResult := stableContains(result, x, y, delta)
							if !stableA || !stableB || !stableResult {
								continue
							}
							if got != evalPathOp(op, inA, inB) {
								wrong++
							}
						}
					}
					if wrong != 0 {
						t.Errorf("base %d sub %v reversed=%v %s: %d sampled points disagree with the operands' fill",
							baseIndex, sub, reversed, opName, wrong)
					}
				}
			}
		}
	}
}
