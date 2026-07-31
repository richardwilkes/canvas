// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import (
	"math"
	"math/rand"
	"testing"
)

// structuralMaxPieces is the most monotonic pieces ClipCubic can hand to clipMonoCubic: ChopCubicAtYExtrema and
// ChopCubicAtXExtrema each return a count in [0, 2], so both loops run at most 3 times regardless of what the float
// math reports. edgeClipperMaxVerbs/Points are sized from this.
const structuralMaxPieces = 3 * 3

// collectClipped drains the clipper into (verb, points) records.
type clippedSeg struct {
	pts  []Point
	verb ClipVerb
}

func drainClipper(e *EdgeClipper) []clippedSeg {
	var out []clippedSeg
	var pts [4]Point
	for {
		verb, ok := e.Next(pts[:])
		if !ok {
			return out
		}
		n := 0
		switch verb {
		case ClipVerbLine:
			n = 2
		case ClipVerbQuad:
			n = 3
		case ClipVerbCubic:
			n = 4
		}
		out = append(out, clippedSeg{verb: verb, pts: append([]Point(nil), pts[:n]...)})
	}
}

func TestClipLineInsideUnchanged(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	if !e.ClipLine(Point{X: 10, Y: 10}, Point{X: 90, Y: 90}, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) != 1 || segs[0].verb != ClipVerbLine {
		t.Fatalf("segs = %+v", segs)
	}
	if segs[0].pts[0] != (Point{X: 10, Y: 10}) || segs[0].pts[1] != (Point{X: 90, Y: 90}) {
		t.Errorf("pts = %v", segs[0].pts)
	}
}

func TestClipLineAboveRejected(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	if e.ClipLine(Point{X: 10, Y: -50}, Point{X: 90, Y: -1}, clip) {
		t.Fatal("expected no output for a line above the clip")
	}
}

func TestClipLineLeftBecomesVertical(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	if !e.ClipLine(Point{X: -50, Y: 10}, Point{X: -10, Y: 90}, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) != 1 {
		t.Fatalf("segs = %+v", segs)
	}
	p := segs[0].pts
	if p[0].X != 0 || p[1].X != 0 || p[0].Y != 10 || p[1].Y != 90 {
		t.Errorf("expected vertical line on left edge, got %v", p)
	}
}

func TestClipLineRightCulled(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(true)
	if e.ClipLine(Point{X: 110, Y: 10}, Point{X: 150, Y: 90}, clip) {
		t.Fatal("expected culled output with canCullToTheRight")
	}
	e = NewEdgeClipper(false)
	if !e.ClipLine(Point{X: 110, Y: 10}, Point{X: 150, Y: 90}, clip) {
		t.Fatal("expected vertical line without culling")
	}
	segs := drainClipper(e)
	if len(segs) != 1 || segs[0].pts[0].X != 100 || segs[0].pts[1].X != 100 {
		t.Errorf("segs = %+v", segs)
	}
}

func TestClipLineCrossingProducesSegments(t *testing.T) {
	clip := RectLTRB(0, 0, 10, 10)
	e := NewEdgeClipper(false)
	// Enters on the left, exits on the right.
	if !e.ClipLine(Point{X: -10, Y: 2}, Point{X: 20, Y: 8}, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) != 3 {
		t.Fatalf("expected 3 segments (vline, interior, vline), got %+v", segs)
	}
	// Segments must chain continuously.
	for i := 1; i < len(segs); i++ {
		last := segs[i-1].pts[len(segs[i-1].pts)-1]
		if segs[i].pts[0] != last {
			t.Errorf("discontinuity between segment %d and %d: %v vs %v", i-1, i, last, segs[i].pts[0])
		}
	}
}

func TestClipQuadInsideUnchanged(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	src := []Point{{X: 10, Y: 10}, {X: 50, Y: 90}, {X: 90, Y: 10}}
	if !e.ClipQuad(src, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	// The quad is chopped at its Y extrema (apex), so expect quads whose join point chains and whose endpoints match
	// the original.
	if segs[0].pts[0] != src[0] {
		t.Errorf("start = %v", segs[0].pts[0])
	}
	lastSeg := segs[len(segs)-1]
	if lastSeg.pts[len(lastSeg.pts)-1] != src[2] {
		t.Errorf("end = %v", lastSeg.pts[len(lastSeg.pts)-1])
	}
	for _, s := range segs {
		if s.verb != ClipVerbQuad {
			t.Errorf("expected only quads, got %+v", s)
		}
	}
}

func TestClipQuadChoppedInY(t *testing.T) {
	clip := RectLTRB(0, 20, 100, 100)
	e := NewEdgeClipper(false)
	src := []Point{{X: 10, Y: 0}, {X: 50, Y: 60}, {X: 90, Y: 0}}
	if !e.ClipQuad(src, clip) {
		t.Fatal("expected output")
	}
	for _, s := range drainClipper(e) {
		for _, p := range s.pts {
			if p.Y < clip.Top-0.001 || p.Y > clip.Bottom+0.001 {
				t.Errorf("point %v outside clip Y range", p)
			}
		}
	}
}

func TestClipCubicChoppedInXAndY(t *testing.T) {
	clip := RectLTRB(10, 10, 60, 60)
	e := NewEdgeClipper(false)
	src := []Point{{X: 0, Y: 0}, {X: 100, Y: 20}, {X: -40, Y: 60}, {X: 70, Y: 80}}
	if !e.ClipCubic(src, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) == 0 {
		t.Fatal("no segments")
	}
	for _, s := range segs {
		for _, p := range s.pts {
			if p.Y < clip.Top-0.01 || p.Y > clip.Bottom+0.01 {
				t.Errorf("point %v outside clip Y range", p)
			}
			// X may touch the clip edges exactly (vertical closure lines) but not exceed them.
			if p.X < clip.Left-0.01 || p.X > clip.Right+0.01 {
				t.Errorf("point %v outside clip X range", p)
			}
		}
	}
}

func TestClipCubicHugeFallsBackToLine(t *testing.T) {
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	src := []Point{{X: -1e7, Y: -1e7}, {X: 5e6, Y: 2e6}, {X: -5e6, Y: 8e6}, {X: 1e7, Y: 1e7}}
	if !e.ClipCubic(src, clip) {
		t.Fatal("expected output")
	}
	for _, s := range drainClipper(e) {
		if s.verb != ClipVerbLine {
			t.Errorf("expected line-only fallback for huge cubic, got verb %d", s.verb)
		}
	}
}

// clipperEmission drains the clipper, returning how many verbs and points the last Clip* call emitted.
func clipperEmission(e *EdgeClipper) (verbs, points int) {
	var pts [4]Point
	for {
		verb, ok := e.Next(pts[:])
		if !ok {
			return verbs, points
		}
		verbs++
		switch verb {
		case ClipVerbLine:
			points += 2
		case ClipVerbQuad:
			points += 3
		case ClipVerbCubic:
			points += 4
		}
	}
}

func TestClipCubicSinglePieceWorstCase(t *testing.T) {
	// A cubic with no extrema in either axis that crosses the clip on both sides is the per-piece worst case of the
	// buffer-limit derivation in edgeclipper.go: a left vline, the clipped cubic, then a right vline (3 verbs, 8
	// points).
	clip := RectLTRB(0, 0, 100, 100)
	e := NewEdgeClipper(false)
	src := []Point{{X: -50, Y: 10}, {X: 0, Y: 30}, {X: 100, Y: 70}, {X: 150, Y: 90}}
	if !e.ClipCubic(src, clip) {
		t.Fatal("expected output")
	}
	segs := drainClipper(e)
	if len(segs) != 3 {
		t.Fatalf("segs = %+v, want 3", segs)
	}
	if segs[0].verb != ClipVerbLine || segs[1].verb != ClipVerbCubic || segs[2].verb != ClipVerbLine {
		t.Errorf("verbs = %d/%d/%d, want line/cubic/line", segs[0].verb, segs[1].verb, segs[2].verb)
	}
	points := 0
	for _, s := range segs {
		points += len(s.pts)
	}
	if points != 8 {
		t.Errorf("points = %d, want 8", points)
	}
}

func TestEdgeClipperStaysWithinBufferLimits(t *testing.T) {
	// For well-conditioned inputs the exact-math argument holds — chopping in Y manufactures no new X extrema, so a
	// cubic yields at most 5 monotonic pieces of 3 verbs / 8 points and a quad at most 3 pieces of 3 verbs / 7 points.
	// These are the tight bounds for the inputs exercised here, NOT the buffer sizes: float32 rounding can push a cubic
	// past them, which is why edgeClipperMaxVerbs/Points are derived from the loops' structural bound instead. See
	// TestEdgeClipperFloat32ExceedsExactMathPieceBound.
	const (
		cubicMaxVerbs  = 15
		cubicMaxPoints = 40
		quadMaxVerbs   = 9
		quadMaxPoints  = 21
	)
	clip := RectLTRB(0, 0, 100, 100)
	coords := []float32{-60, 20, 80, 170}
	check := func(what string, verbs, points, wantVerbs, wantPoints int, src []Point) {
		t.Helper()
		if verbs > wantVerbs || points > wantPoints {
			t.Fatalf("%s %v: emitted %d verbs / %d points, want at most %d / %d", what, src, verbs, points, wantVerbs,
				wantPoints)
		}
		if verbs > edgeClipperMaxVerbs || points > edgeClipperMaxPoints {
			t.Fatalf("%s %v: emitted %d verbs / %d points, past the %d / %d buffers", what, src, verbs, points,
				edgeClipperMaxVerbs, edgeClipperMaxPoints)
		}
	}
	for _, cull := range []bool{false, true} {
		e := NewEdgeClipper(cull)
		var src [4]Point
		// Every combination of the grid coordinates covers the zigzags that produce 2 extrema in each axis.
		for i := range 1 << (4 * 4) { // 4 points x (x,y) x 4 choices, encoded 2 bits at a time
			for j := range 4 {
				src[j].X = coords[(i>>(4*j))&3]
				src[j].Y = coords[(i>>(4*j+2))&3]
			}
			e.ClipCubic(src[:], clip)
			verbs, points := clipperEmission(e)
			check("cubic", verbs, points, cubicMaxVerbs, cubicMaxPoints, src[:])
			e.ClipQuad(src[:3], clip)
			verbs, points = clipperEmission(e)
			check("quad", verbs, points, quadMaxVerbs, quadMaxPoints, src[:3])
		}
		rng := rand.New(rand.NewSource(1))
		for range 20000 {
			for j := range 4 {
				src[j].X = rng.Float32()*300 - 100
				src[j].Y = rng.Float32()*300 - 100
			}
			e.ClipCubic(src[:], clip)
			verbs, points := clipperEmission(e)
			check("cubic", verbs, points, cubicMaxVerbs, cubicMaxPoints, src[:])
			e.ClipQuad(src[:3], clip)
			verbs, points = clipperEmission(e)
			check("quad", verbs, points, quadMaxVerbs, quadMaxPoints, src[:3])
		}
	}
}

func TestIntersectLine(t *testing.T) {
	clip := RectLTRB(0, 0, 10, 10)
	src := [2]Point{{X: -5, Y: 5}, {X: 15, Y: 5}}
	var dst [2]Point
	if !IntersectLine(&src, clip, &dst) {
		t.Fatal("expected intersection")
	}
	if dst[0] != (Point{X: 0, Y: 5}) || dst[1] != (Point{X: 10, Y: 5}) {
		t.Errorf("dst = %v", dst)
	}
	miss := [2]Point{{X: -5, Y: 15}, {X: 15, Y: 20}}
	if IntersectLine(&miss, clip, &dst) {
		t.Error("expected miss")
	}
}

func TestEdgeClipperFloat32ExceedsExactMathPieceBound(t *testing.T) {
	// Regression for the buffer-limit derivation. ChopCubicAtXExtrema re-runs FindCubicExtrema on each already-rounded
	// Y-piece, so in float32 a piece can report an X extremum the unchopped curve did not have, defeating the
	// exact-math argument that 2 extrema per axis cap the split at 5 pieces.
	//
	// How far past 5 it goes is architecture-dependent: this seed splits into 6 pieces on arm64 (where the chop math
	// contracts into FMAs) and perturbations of it reach 7, while on amd64 the same family stays at 5. So the piece
	// count is reported, not asserted. What is asserted holds everywhere: the split never exceeds the 3x3 structural
	// bound the buffers are sized from, and clipping never overruns those buffers.
	seed := []Point{
		{X: -36124.6, Y: 100},
		{X: -1343.5917, Y: -4e6},
		{X: -236631.27, Y: 99.5},
		{X: -107986.55, Y: -4e6},
	}
	pieces := func(src []Point) int {
		if tooBigForReliableFloatMath(BoundsOrEmpty(src)) {
			return -1 // the line fallback, which does no chopping
		}
		var monoY [10]Point
		countY := ChopCubicAtYExtrema(src, monoY[:])
		n := 0
		for y := 0; y <= countY; y++ {
			var monoX [10]Point
			n += ChopCubicAtXExtrema(monoY[y*3:y*3+4], monoX[:]) + 1
		}
		return n
	}
	if pieces(seed) < 0 {
		t.Fatal("seed must reach the chopping path, not the too-big line fallback")
	}
	// Clip rects chosen to straddle the seed family in X and in Y, so pieces are kept rather than trivially rejected.
	clips := []Rect{
		RectLTRB(-200000, -100, 0, 100),
		RectLTRB(-150000, -5e6, -50000, 5e6),
		RectLTRB(-236000, -4e6, -1400, 4e6),
	}
	clippers := []*EdgeClipper{NewEdgeClipper(false), NewEdgeClipper(true)}
	maxPieces, maxVerbs, maxPoints, beyondExactMath := 0, 0, 0, 0
	check := func(src []Point) {
		t.Helper()
		n := pieces(src)
		if n < 0 {
			return
		}
		maxPieces = max(maxPieces, n)
		if n > 5 {
			beyondExactMath++
		}
		if n > structuralMaxPieces {
			t.Fatalf("%v split into %d pieces, past the %d the buffers are sized for", src, n, structuralMaxPieces)
		}
		for _, e := range clippers {
			for _, clip := range clips {
				e.ClipCubic(src, clip)
				verbs, points := clipperEmission(e)
				maxVerbs, maxPoints = max(maxVerbs, verbs), max(maxPoints, points)
				if verbs > edgeClipperMaxVerbs || points > edgeClipperMaxPoints {
					t.Fatalf("%v under %v: emitted %d verbs / %d points, past the %d / %d buffers", src, clip, verbs,
						points, edgeClipperMaxVerbs, edgeClipperMaxPoints)
				}
			}
		}
	}
	check(seed)
	// Perturbing the seed across a range of relative magnitudes is what surfaces the worst splits on arm64.
	rng := rand.New(rand.NewSource(4))
	for range 200000 {
		mag := math.Pow(10, rng.Float64()*6-7)
		src := make([]Point, 4)
		for i, p := range seed {
			jitter := func(v float32) float32 { return float32(float64(v) * (1 + (rng.Float64()*2-1)*mag)) }
			src[i] = Point{X: jitter(p.X), Y: jitter(p.Y)}
		}
		check(src)
	}
	t.Logf("max pieces %d (%d cases past the exact-math bound of 5), max emission %d verbs / %d points, buffers %d / %d",
		maxPieces, beyondExactMath, maxVerbs, maxPoints, edgeClipperMaxVerbs, edgeClipperMaxPoints)
}

func TestEdgeClipperBuffersCoverStructuralWorstCase(t *testing.T) {
	// Each of the structuralMaxPieces pieces emits at most 3 verbs / 8 points via clipMonoCubic. The buffers must cover
	// that regardless of any exact-math reasoning about how extrema distribute.
	const (
		verbsPerPiece    = 3
		pointsPerPiece   = 8
		structuralVerbs  = structuralMaxPieces * verbsPerPiece
		structuralPoints = structuralMaxPieces * pointsPerPiece
	)
	if edgeClipperMaxVerbs < structuralVerbs {
		t.Errorf("edgeClipperMaxVerbs = %d, want at least %d", edgeClipperMaxVerbs, structuralVerbs)
	}
	if edgeClipperMaxPoints < structuralPoints {
		t.Errorf("edgeClipperMaxPoints = %d, want at least %d", edgeClipperMaxPoints, structuralPoints)
	}
}

func TestClipLineCullToTheRightOnlyWhollyRight(t *testing.T) {
	// Documented contract: canCullToTheRight drops only a segment lying wholly at/right of clip.Right. A segment that
	// merely extends past the right edge while still crossing the clip keeps its right-edge vertical piece either way.
	clip := RectLTRB(0, 0, 100, 100)
	var lines [LineClipperMaxPoints]Point
	wholly := [2]Point{{X: 120, Y: 10}, {X: 150, Y: 90}}
	if n := ClipLine(&wholly, clip, &lines, true); n != 0 {
		t.Errorf("wholly-right segment with culling produced %d segments, want 0", n)
	}
	if n := ClipLine(&wholly, clip, &lines, false); n != 1 {
		t.Errorf("wholly-right segment without culling produced %d segments, want 1", n)
	}
	// Crosses the clip and exits to the right: the right-edge vertical piece survives regardless of the flag.
	crossing := [2]Point{{X: 50, Y: 10}, {X: 150, Y: 90}}
	culled := ClipLine(&crossing, clip, &lines, true)
	culledPts := make([]Point, 0, culled+1)
	culledPts = append(culledPts, lines[:culled+1]...)
	uncut := ClipLine(&crossing, clip, &lines, false)
	if culled != uncut {
		t.Errorf("culling changed segment count for a partially-right line: %d vs %d", culled, uncut)
	}
	for i := 0; i <= uncut; i++ {
		if culledPts[i] != lines[i] {
			t.Errorf("culling changed point %d for a partially-right line: %v vs %v", i, culledPts[i], lines[i])
		}
	}
	rightEdge := false
	for i := 0; i <= uncut; i++ {
		if lines[i].X == clip.Right {
			rightEdge = true
		}
	}
	if !rightEdge {
		t.Error("expected a point on the right clip edge, which canCullToTheRight does not suppress")
	}
}
