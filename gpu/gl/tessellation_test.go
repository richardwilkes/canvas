// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for the fixed-count tessellation support layer: Wang's formula (bit-level nextlog2 plus the
// mathematical flatness guarantee it exists to provide), the convex-180 cubic chopper, PreChopPathCurves' segment bound
// and offscreen flattening, the middle-out polygon triangulator's exact topology and area equivalence, the midpoint
// contour parser, the fixed-count static buffer contents, the PatchWriter's patch encodings and chopping, the
// non-convex fill-op selection heuristic, and PathTessellateOp's record-time batching on the fake-driver context.

package gl

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

func TestWangsNextLog2(t *testing.T) {
	cases := []struct {
		x    float32
		want int
	}{
		{x: float32(math.Inf(-1)), want: 0},
		{x: -5, want: 0},
		{x: 0, want: 0},
		{x: 0.5, want: 0},
		{x: 1, want: 0},
		{x: math.Nextafter32(1, 2), want: 1},
		{x: 2, want: 1},
		{x: math.Nextafter32(2, 3), want: 2},
		{x: 4, want: 2},
		{x: math.Nextafter32(4, 5), want: 3},
		{x: 8, want: 3},
		{x: 1024, want: 10},
		{x: math.Nextafter32(1024, 2048), want: 11},
		{x: float32(math.Inf(1)), want: 128},
		{x: float32(math.NaN()), want: 0},
	}
	for _, tc := range cases {
		if got := wangsNextLog2(tc.x); got != tc.want {
			t.Errorf("wangsNextLog2(%v) = %d, want %d", tc.x, got, tc.want)
		}
	}
	// Cross-check against ceil(log2(x)) for a magnitude sweep.
	for e := -10; e <= 20; e++ {
		for _, m := range []float64{1.0001, 1.5, 1.9999} {
			x := m * math.Ldexp(1, e)
			want := int(math.Ceil(math.Log2(x)))
			if want < 0 {
				want = 0
			}
			if got := wangsNextLog2(float32(x)); got != want {
				t.Fatalf("wangsNextLog2(%v) = %d, want %d", x, got, want)
			}
		}
	}
}

// evalQuad/evalCubic/evalConic are float64 reference evaluators.
func evalQuad(p [3]geom.Point, t float64) (x, y float64) {
	mt := 1 - t
	x = mt*mt*float64(p[0].X) + 2*mt*t*float64(p[1].X) + t*t*float64(p[2].X)
	y = mt*mt*float64(p[0].Y) + 2*mt*t*float64(p[1].Y) + t*t*float64(p[2].Y)
	return x, y
}

func evalCubic(p [4]geom.Point, t float64) (x, y float64) {
	mt := 1 - t
	a := mt * mt * mt
	b := 3 * mt * mt * t
	c := 3 * mt * t * t
	d := t * t * t
	x = a*float64(p[0].X) + b*float64(p[1].X) + c*float64(p[2].X) + d*float64(p[3].X)
	y = a*float64(p[0].Y) + b*float64(p[1].Y) + c*float64(p[2].Y) + d*float64(p[3].Y)
	return x, y
}

func evalConic(p [3]geom.Point, w, t float64) (x, y float64) {
	mt := 1 - t
	num := mt * mt
	numX := num * float64(p[0].X)
	numY := num * float64(p[0].Y)
	den := num
	num = 2 * mt * t * w
	numX += num * float64(p[1].X)
	numY += num * float64(p[1].Y)
	den += num
	num = t * t
	numX += num * float64(p[2].X)
	numY += num * float64(p[2].Y)
	den += num
	return numX / den, numY / den
}

// distToSegment returns the distance from (x, y) to the segment (x0,y0)-(x1,y1).
func distToSegment(x, y, x0, y0, x1, y1 float64) float64 {
	dx, dy := x1-x0, y1-y0
	lenSqd := dx*dx + dy*dy
	if lenSqd == 0 {
		return math.Hypot(x-x0, y-y0)
	}
	tt := ((x-x0)*dx + (y-y0)*dy) / lenSqd
	tt = math.Max(0, math.Min(1, tt))
	return math.Hypot(x-(x0+tt*dx), y-(y0+tt*dy))
}

// TestWangsFormulaFlatnessGuarantee verifies the property the formula exists to provide: chopping a curve into
// ceil(wangs) parametrically uniform segments keeps every point of the true curve within 1/precision of its chord.
func TestWangsFormulaFlatnessGuarantee(t *testing.T) {
	const precision = tessPrecision
	const tol = 1/float64(precision) + 1e-4
	xf := identityVectorXform()

	quads := [][3]geom.Point{
		{{X: 0, Y: 0}, {X: 50, Y: 100}, {X: 100, Y: 0}},
		{{X: -30, Y: 20}, {X: 400, Y: -300}, {X: 200, Y: 250}},
		{{X: 0, Y: 0}, {X: 1, Y: 2}, {X: 3, Y: 0}},
	}
	for i, q := range quads {
		n := int(math.Ceil(float64(wangsQuadratic(precision, q[0], q[1], q[2], xf))))
		n = max(n, 1)
		for s := 0; s < n; s++ {
			t0, t1 := float64(s)/float64(n), float64(s+1)/float64(n)
			x0, y0 := evalQuad(q, t0)
			x1, y1 := evalQuad(q, t1)
			for _, f := range []float64{0.25, 0.5, 0.75} {
				x, y := evalQuad(q, t0+(t1-t0)*f)
				if d := distToSegment(x, y, x0, y0, x1, y1); d > tol {
					t.Fatalf("quad %d segment %d/%d deviates %v > %v", i, s, n, d, tol)
				}
			}
		}
	}

	cubics := [][4]geom.Point{
		{{X: 0, Y: 0}, {X: 100, Y: 150}, {X: -50, Y: 150}, {X: 100, Y: 0}},
		{{X: 10, Y: 10}, {X: 20, Y: 12}, {X: 30, Y: 8}, {X: 40, Y: 10}},
		{{X: 0, Y: 0}, {X: 300, Y: -200}, {X: -300, Y: -200}, {X: 0, Y: 0}},
	}
	for i, c := range cubics {
		n := int(math.Ceil(float64(wangsCubic(precision, c[0], c[1], c[2], c[3], xf))))
		n = max(n, 1)
		for s := 0; s < n; s++ {
			t0, t1 := float64(s)/float64(n), float64(s+1)/float64(n)
			x0, y0 := evalCubic(c, t0)
			x1, y1 := evalCubic(c, t1)
			for _, f := range []float64{0.25, 0.5, 0.75} {
				x, y := evalCubic(c, t0+(t1-t0)*f)
				if d := distToSegment(x, y, x0, y0, x1, y1); d > tol {
					t.Fatalf("cubic %d segment %d/%d deviates %v > %v", i, s, n, d, tol)
				}
			}
		}
	}

	conics := []struct {
		p [3]geom.Point
		w float32
	}{
		{p: [3]geom.Point{{X: 0, Y: 0}, {X: 100, Y: 100}, {X: 200, Y: 0}}, w: 0.5},
		{p: [3]geom.Point{{X: 0, Y: 0}, {X: 100, Y: 100}, {X: 200, Y: 0}}, w: float32(math.Sqrt2 / 2)},
		{p: [3]geom.Point{{X: -50, Y: 30}, {X: 60, Y: -90}, {X: 80, Y: 100}}, w: 3},
	}
	for i, tc := range conics {
		n := int(math.Ceil(float64(wangsConic(precision, tc.p[0], tc.p[1], tc.p[2], tc.w, xf))))
		n = max(n, 1)
		for s := 0; s < n; s++ {
			t0, t1 := float64(s)/float64(n), float64(s+1)/float64(n)
			x0, y0 := evalConic(tc.p, float64(tc.w), t0)
			x1, y1 := evalConic(tc.p, float64(tc.w), t1)
			for _, f := range []float64{0.25, 0.5, 0.75} {
				x, y := evalConic(tc.p, float64(tc.w), t0+(t1-t0)*f)
				if d := distToSegment(x, y, x0, y0, x1, y1); d > tol {
					t.Fatalf("conic %d segment %d/%d deviates %v > %v", i, s, n, d, tol)
				}
			}
		}
	}

	// The log2 forms must round the plain forms up to the next power of two.
	for _, c := range cubics {
		n := wangsCubic(precision, c[0], c[1], c[2], c[3], xf)
		lg := wangsCubicLog2(precision, c[0], c[1], c[2], c[3], xf)
		if float32(math.Ldexp(1, lg)) < n {
			t.Fatalf("cubic log2 %d < segments %v", lg, n)
		}
		if lg > 0 && float32(math.Ldexp(1, lg-1)) >= n {
			t.Fatalf("cubic log2 %d not tight for %v", lg, n)
		}
	}

	// worst_case_cubic bounds any cubic whose control points live in the given box.
	for i, c := range cubics {
		minX, minY, maxX, maxY := c[0].X, c[0].Y, c[0].X, c[0].Y
		for _, p := range c[1:] {
			minX, maxX = min32(minX, p.X), max32(maxX, p.X)
			minY, maxY = min32(minY, p.Y), max32(maxY, p.Y)
		}
		worst := wangsWorstCaseCubic(precision, maxX-minX, maxY-minY)
		actual := wangsCubic(precision, c[0], c[1], c[2], c[3], xf)
		if worst < actual {
			t.Fatalf("cubic %d worst case %v < actual %v", i, worst, actual)
		}
	}
}

// chopCubicAt64 chops a cubic at t (float64 De Casteljau) and returns both halves.
func chopCubicAt64(c [4]geom.Point, t float64) (lo, hi [4]geom.Point) {
	mix := func(a, b geom.Point) geom.Point {
		return geom.Point{
			X: float32(float64(a.X) + (float64(b.X)-float64(a.X))*t),
			Y: float32(float64(a.Y) + (float64(b.Y)-float64(a.Y))*t),
		}
	}
	ab, bc, cd := mix(c[0], c[1]), mix(c[1], c[2]), mix(c[2], c[3])
	abc, bcd := mix(ab, bc), mix(bc, cd)
	abcd := mix(abc, bcd)
	return [4]geom.Point{c[0], ab, abc, abcd}, [4]geom.Point{abcd, bcd, cd, c[3]}
}

// maxRotation returns the maximum accumulated tangent rotation (radians) along the cubic, sampled densely.
func maxRotation(c [4]geom.Point) float64 {
	tangent := func(t float64) (float64, float64) {
		mt := 1 - t
		dx := 3 * (mt*mt*(float64(c[1].X)-float64(c[0].X)) +
			2*mt*t*(float64(c[2].X)-float64(c[1].X)) + t*t*(float64(c[3].X)-float64(c[2].X)))
		dy := 3 * (mt*mt*(float64(c[1].Y)-float64(c[0].Y)) +
			2*mt*t*(float64(c[2].Y)-float64(c[1].Y)) + t*t*(float64(c[3].Y)-float64(c[2].Y)))
		return dx, dy
	}
	total := 0.0
	prevX, prevY := tangent(1e-4)
	for i := 1; i <= 512; i++ {
		t := float64(i) / 512
		x, y := tangent(t)
		if x == 0 && y == 0 {
			continue
		}
		cross := prevX*y - prevY*x
		dot := prevX*x + prevY*y
		total += math.Abs(math.Atan2(cross, dot))
		prevX, prevY = x, y
	}
	return total
}

func TestFindCubicConvex180Chops(t *testing.T) {
	var tv [2]float32
	var areCusps bool

	// A serpentine cubic with both inflections inside (0, 1) chops twice (roots computed with a float64 reference:
	// ~0.167 and ~0.685).
	serpentine := [4]geom.Point{{X: 0, Y: 0}, {X: 84, Y: 103}, {X: 37, Y: 79}, {X: 100, Y: -27}}
	n := findCubicConvex180Chops(&serpentine, &tv, &areCusps)
	if n != 2 || areCusps {
		t.Fatalf("serpentine chops = %d (cusps=%v), want 2", n, areCusps)
	}
	if !(0 < tv[0] && tv[0] < tv[1] && tv[1] < 1) {
		t.Fatalf("serpentine chop Ts not sorted in (0,1): %v", tv)
	}
	if math.Abs(float64(tv[0])-0.16700094) > 1e-3 || math.Abs(float64(tv[1])-0.68452869) > 1e-3 {
		t.Fatalf("serpentine chop Ts = %v, want ~(0.167, 0.685)", tv)
	}

	// The symmetric double-root case: its inflection discriminant is exactly zero, which the chopper classifies as a
	// single cusp at the averaged root T=0.5.
	symmetric := [4]geom.Point{{X: 0, Y: 0}, {X: 100, Y: 100}, {X: 0, Y: 100}, {X: 100, Y: 0}}
	n = findCubicConvex180Chops(&symmetric, &tv, &areCusps)
	if n != 1 || !areCusps || tv[0] != 0.5 {
		t.Fatalf("symmetric cusp = %d chops (cusps=%v) at %v, want 1 cusp at 0.5", n, areCusps,
			tv[0])
	}

	// A loop that rotates more than 180 degrees chops once, and each side must rotate <= ~180.
	loop := [4]geom.Point{{X: 0, Y: 0}, {X: 200, Y: -100}, {X: 200, Y: 100}, {X: 0, Y: 10}}
	n = findCubicConvex180Chops(&loop, &tv, &areCusps)
	if n != 1 || areCusps {
		t.Fatalf("loop chops = %d (cusps=%v), want 1", n, areCusps)
	}
	lo, hi := chopCubicAt64(loop, float64(tv[0]))
	const maxRot = math.Pi + 0.05
	if r := maxRotation(lo); r > maxRot {
		t.Fatalf("loop lo side rotates %v > pi", r)
	}
	if r := maxRotation(hi); r > maxRot {
		t.Fatalf("loop hi side rotates %v > pi", r)
	}

	// A flat line with a 180-degree turnaround has a cusp.
	flat := [4]geom.Point{{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 50, Y: 0}, {X: 75, Y: 0}}
	n = findCubicConvex180Chops(&flat, &tv, &areCusps)
	if n == 0 || !areCusps {
		t.Fatalf("flat turnaround chops = %d (cusps=%v), want >=1 cusp", n, areCusps)
	}

	// Fully colocated control points: no chops (root is NaN).
	dot := [4]geom.Point{{X: 5, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}}
	if n = findCubicConvex180Chops(&dot, &tv, &areCusps); n != 0 {
		t.Fatalf("degenerate point chops = %d, want 0", n)
	}

	// A convex cubic that rotates less than 180 degrees: no chops.
	convex := [4]geom.Point{{X: 0, Y: 0}, {X: 30, Y: 40}, {X: 70, Y: 40}, {X: 100, Y: 0}}
	if n = findCubicConvex180Chops(&convex, &tv, &areCusps); n != 0 {
		t.Fatalf("convex chops = %d, want 0", n)
	}
}

func TestConicHasCusp(t *testing.T) {
	turnaround := [3]geom.Point{{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 50, Y: 0}}
	if !conicHasCusp(&turnaround) {
		t.Fatal("180-degree flat turnaround must have a cusp")
	}
	normal := [3]geom.Point{{X: 0, Y: 0}, {X: 50, Y: 50}, {X: 100, Y: 0}}
	if conicHasCusp(&normal) {
		t.Fatal("a normal conic must not have a cusp")
	}
	sameDir := [3]geom.Point{{X: 0, Y: 0}, {X: 50, Y: 0}, {X: 100, Y: 0}}
	if conicHasCusp(&sameDir) {
		t.Fatal("a monotone flat conic must not have a cusp")
	}
}

func TestPreChopPathCurves(t *testing.T) {
	identity := geom.IdentityMatrix()
	viewport := geom.Rect{Left: 0, Top: 0, Right: 1024, Bottom: 1024}

	// A gigantic cubic that crosses the viewport: every emitted curve must fit the segment budget.
	p := &path.Path{}
	p.MoveTo(-500000, 512)
	p.CubicTo(200000, -800000, 800000, 900000, 1500000, 512)
	p.QuadTo(500, 2000000, -100, 512)
	p.Close()
	p.SetFillType(path.FillEvenOdd)

	chopped := preChopPathCurves(tessPrecision, p, &identity, viewport)
	if chopped.FillType() != path.FillEvenOdd {
		t.Fatalf("fill type not preserved: %v", chopped.FillType())
	}
	xf := identityVectorXform()
	it := path.NewRawIter(chopped)
	sawCurve := false
	for {
		verb, pts, w, ok := it.Next()
		if !ok {
			break
		}
		var n float32
		switch verb {
		case path.VerbQuad:
			n = wangsQuadratic(tessPrecision, pts[0], pts[1], pts[2], xf)
			sawCurve = true
		case path.VerbConic:
			n = wangsConic(tessPrecision, pts[0], pts[1], pts[2], w, xf)
			sawCurve = true
		case path.VerbCubic:
			n = wangsCubic(tessPrecision, pts[0], pts[1], pts[2], pts[3], xf)
			sawCurve = true
		default:
			continue
		}
		if n > tessMaxSegmentsPerCurve {
			t.Fatalf("chopped %v still needs %v segments", verb, n)
		}
	}
	if !sawCurve {
		t.Fatal("expected the on-screen portions to stay curves")
	}

	// A curve entirely outside the viewport must flatten to lines.
	off := &path.Path{}
	off.MoveTo(2000, 2000)
	off.CubicTo(900000, 3000, 800000, 500000, 2000, 900000)
	choppedOff := preChopPathCurves(tessPrecision, off, &identity, viewport)
	it = path.NewRawIter(choppedOff)
	for {
		verb, _, _, ok := it.Next()
		if !ok {
			break
		}
		if verb == path.VerbQuad || verb == path.VerbConic || verb == path.VerbCubic {
			t.Fatalf("offscreen curve survived as %v", verb)
		}
	}

	// A small on-screen path is passed through with identical verbs.
	small := &path.Path{}
	small.MoveTo(10, 10)
	small.QuadTo(50, 80, 90, 10)
	small.Close()
	choppedSmall := preChopPathCurves(tessPrecision, small, &identity, viewport)
	if choppedSmall.CountVerbs() != small.CountVerbs() {
		t.Fatalf("small path verb count changed: %d -> %d", small.CountVerbs(),
			choppedSmall.CountVerbs())
	}
}

func TestMiddleOutPolygonTriangulator(t *testing.T) {
	// The documented topology for a run of 9 points (deltas 1, 2, 4).
	m := newMiddleOutPolygonTriangulator(9, geom.Point{X: 0, Y: 0})
	pt := func(i int) geom.Point { return geom.Point{X: float32(i), Y: float32(i * i)} }
	var tris [][3]geom.Point
	emit := func(a, b, c geom.Point) { tris = append(tris, [3]geom.Point{a, b, c}) }
	for i := 1; i <= 8; i++ {
		m.pushVertex(pt(i), emit)
	}
	m.close(emit)
	want := [][3]int{
		{0, 1, 2}, {2, 3, 4}, {0, 2, 4}, {4, 5, 6}, {6, 7, 8}, {4, 6, 8}, {0, 4, 8},
	}
	if len(tris) != len(want) {
		t.Fatalf("triangles = %d, want %d", len(tris), len(want))
	}
	idx := func(p geom.Point) int { return int(p.X) }
	for i, w := range want {
		got := [3]int{idx(tris[i][0]), idx(tris[i][1]), idx(tris[i][2])}
		// The stack pops yield each triangle as (left, middle, last-pushed); the vertex sets must match regardless of
		// rotation.
		gotSet := map[int]bool{got[0]: true, got[1]: true, got[2]: true}
		if !gotSet[w[0]] || !gotSet[w[1]] || !gotSet[w[2]] {
			t.Fatalf("triangle %d = %v, want vertices %v", i, got, w)
		}
	}

	// Signed-area equivalence over a multi-contour path: the fan triangulation's summed signed area must equal the
	// contours' summed signed areas.
	p := &path.Path{}
	p.MoveTo(0, 0).LineTo(40, 4).LineTo(37, 33).LineTo(21, 42).LineTo(-3, 25).Close()
	p.MoveTo(60, 60)
	p.QuadTo(80, 20, 100, 60)
	p.CubicTo(90, 90, 70, 95, 60, 60)
	signed := 0.0
	pathMiddleOutFanTriangles(p, func(a, b, c geom.Point) {
		signed += (float64(b.X)-float64(a.X))*(float64(c.Y)-float64(a.Y)) -
			(float64(b.Y)-float64(a.Y))*(float64(c.X)-float64(a.X))
	})
	signed /= 2

	// Reference: shoelace over the on-curve endpoints of each contour.
	ref := 0.0
	var first, last geom.Point
	started := false
	acc := func(p0, p1 geom.Point) {
		ref += (float64(p0.X)*float64(p1.Y) - float64(p1.X)*float64(p0.Y)) / 2
	}
	it := path.NewRawIter(p)
	for {
		verb, pts, _, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			if started && last != first {
				acc(last, first)
			}
			first, last = pts[0], pts[0]
			started = true
		case path.VerbLine:
			acc(pts[0], pts[1])
			last = pts[1]
		case path.VerbQuad, path.VerbConic:
			acc(pts[0], pts[2])
			last = pts[2]
		case path.VerbCubic:
			acc(pts[0], pts[3])
			last = pts[3]
		case path.VerbClose:
			if last != first {
				acc(last, first)
				last = first
			}
		}
	}
	if started && last != first {
		acc(last, first)
	}
	if math.Abs(signed-ref) > 1e-3*math.Abs(ref)+1e-6 {
		t.Fatalf("fan signed area %v != contour signed area %v", signed, ref)
	}
}

func TestMidpointContourParser(t *testing.T) {
	p := &path.Path{}
	// A leading degenerate moveTo must be skipped (replaced by the later one).
	p.MoveTo(1000, 1000)
	p.MoveTo(0, 0)
	p.LineTo(10, 0)
	p.LineTo(10, 10)
	// Implicit close: the start point joins the mean.
	p.MoveTo(100, 100)
	p.QuadTo(110, 90, 120, 100)
	p.LineTo(120, 120)
	p.Close()

	parser := newMidpointContourParser(p)
	if !parser.parseNextContour() {
		t.Fatal("expected a first contour")
	}
	// Endpoints (10,0), (10,10) plus the implicit-close start (0,0): mean = (20/3, 10/3).
	mid := parser.currentMidpoint()
	if math.Abs(float64(mid.X)-20.0/3) > 1e-5 || math.Abs(float64(mid.Y)-10.0/3) > 1e-5 {
		t.Fatalf("first contour midpoint = %v", mid)
	}

	if !parser.parseNextContour() {
		t.Fatal("expected a second contour")
	}
	// Endpoints (120,100), (120,120) plus the implicit-close start (100,100).
	mid = parser.currentMidpoint()
	if math.Abs(float64(mid.X)-340.0/3) > 1e-4 || math.Abs(float64(mid.Y)-320.0/3) > 1e-4 {
		t.Fatalf("second contour midpoint = %v", mid)
	}
	recs := parser.currentContour()
	verbs := make([]path.Verb, 0, len(recs))
	for _, r := range recs {
		verbs = append(verbs, r.verb)
	}
	wantVerbs := []path.Verb{path.VerbMove, path.VerbQuad, path.VerbLine, path.VerbClose}
	if len(verbs) != len(wantVerbs) {
		t.Fatalf("second contour verbs = %v", verbs)
	}
	for i := range verbs {
		if verbs[i] != wantVerbs[i] {
			t.Fatalf("second contour verbs = %v, want %v", verbs, wantVerbs)
		}
	}

	if parser.parseNextContour() {
		t.Fatal("expected exactly two contours")
	}
}

// decodeFixedCountVertex reads the (resolveLevel, idx) pair at vertex i.
func decodeFixedCountVertex(buf []byte, i int) (level, idx float32) {
	level = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*8:]))
	idx = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*8+4:]))
	return level, idx
}

func TestFixedCountBuffers(t *testing.T) {
	fixedCountStaticBuffers()

	// Curve vertex buffer: 33 middle-out (resolveLevel, idx) pairs.
	if len(fixedCountCurveVertexBytes) != fixedCountCurvesVertexSize {
		t.Fatalf("curve vertex buffer size = %d", len(fixedCountCurveVertexBytes))
	}
	tOf := func(level, idx float32) float64 {
		return float64(idx) / math.Ldexp(1, int(level))
	}
	// The first two vertices are T=0 and T=1; every later vertex is the odd midpoint of its resolve level, and all 33 T
	// values are distinct multiples of 1/32.
	seen := map[float64]bool{}
	for i := 0; i < fixedCountCurvesVertexCount; i++ {
		level, idx := decodeFixedCountVertex(fixedCountCurveVertexBytes, i)
		tv := tOf(level, idx)
		if tv < 0 || tv > 1 || seen[tv] {
			t.Fatalf("vertex %d has bad or duplicate T %v (level %v idx %v)", i, tv, level, idx)
		}
		seen[tv] = true
		if i >= 2 && int(idx)%2 != 1 {
			t.Fatalf("vertex %d idx %v is not odd", i, idx)
		}
	}
	if !seen[0] || !seen[1] || !seen[0.5] || !seen[1.0/32] {
		t.Fatal("expected T=0, 1, 1/2 and 1/32 vertices")
	}

	// Curve index buffer: 31 triangles, each spanning (a, (a+b)/2, b).
	if len(fixedCountCurveIndexBytes) != fixedCountCurvesIndexSize {
		t.Fatalf("curve index buffer size = %d", len(fixedCountCurveIndexBytes))
	}
	triangleCount := len(fixedCountCurveIndexBytes) / 6
	if triangleCount != numCurveTrianglesAtResolveLevel(tessMaxResolveLevel) {
		t.Fatalf("triangle count = %d", triangleCount)
	}
	for i := 0; i < triangleCount; i++ {
		var tv [3]float64
		for j := 0; j < 3; j++ {
			idx := binary.LittleEndian.Uint16(fixedCountCurveIndexBytes[i*6+j*2:])
			if int(idx) >= fixedCountCurvesVertexCount {
				t.Fatalf("triangle %d index %d out of range", i, idx)
			}
			level, vidx := decodeFixedCountVertex(fixedCountCurveVertexBytes, int(idx))
			tv[j] = tOf(level, vidx)
		}
		if !(tv[0] < tv[1] && tv[1] < tv[2]) || math.Abs(tv[1]-(tv[0]+tv[2])/2) > 1e-12 {
			t.Fatalf("triangle %d spans %v; want ascending with midpoint", i, tv)
		}
	}

	// Wedges: the fan vertex first (negative resolve level), the fan triangle first, and the curve topology shifted by
	// one.
	level, idx := decodeFixedCountVertex(fixedCountWedgeVertexBytes, 0)
	if level >= 0 || idx >= 0 {
		t.Fatalf("wedge vertex 0 = (%v, %v), want negative fan marker", level, idx)
	}
	i0 := binary.LittleEndian.Uint16(fixedCountWedgeIndexBytes[0:])
	i1 := binary.LittleEndian.Uint16(fixedCountWedgeIndexBytes[2:])
	i2 := binary.LittleEndian.Uint16(fixedCountWedgeIndexBytes[4:])
	if i0 != 0 || i1 != 1 || i2 != 2 {
		t.Fatalf("wedge fan triangle = (%d,%d,%d)", i0, i1, i2)
	}
	if len(fixedCountWedgeIndexBytes)/6 != numCurveTrianglesAtResolveLevel(tessMaxResolveLevel)+1 {
		t.Fatalf("wedge triangle count = %d", len(fixedCountWedgeIndexBytes)/6)
	}
}

// testPatchAllocator collects appended patches.
type testPatchAllocator struct {
	patches [][]byte
	stride  int
}

func (a *testPatchAllocator) append(*linearTolerances) []byte {
	buf := make([]byte, a.stride)
	a.patches = append(a.patches, buf)
	return buf
}

func patchPointAt(patch []byte, i int) geom.Point {
	return geom.Point{
		X: math.Float32frombits(binary.LittleEndian.Uint32(patch[i*8:])),
		Y: math.Float32frombits(binary.LittleEndian.Uint32(patch[i*8+4:])),
	}
}

func TestPatchWriterEncodings(t *testing.T) {
	newWriter := func(attribs PatchAttribs, addTris, discardFlat bool) (*patchWriter,
		*testPatchAllocator,
	) {
		alloc := &testPatchAllocator{stride: patchStride(attribs)}
		return newPatchWriter(attribs, alloc, addTris, discardFlat), alloc
	}

	// A small quad becomes one equivalent cubic patch.
	w, alloc := newWriter(PatchAttribsNone, false, false)
	p0 := geom.Point{X: 0, Y: 0}
	p1 := geom.Point{X: 5, Y: 8}
	p2 := geom.Point{X: 10, Y: 0}
	w.writeQuadratic(p0, p1, p2)
	if len(alloc.patches) != 1 {
		t.Fatalf("quad patches = %d, want 1", len(alloc.patches))
	}
	patch := alloc.patches[0]
	wantC1 := geom.Point{X: (p1.X - p0.X) * 2 / 3, Y: (p1.Y - p0.Y) * 2 / 3}
	if got := patchPointAt(patch, 1); math.Abs(float64(got.X-wantC1.X)) > 1e-5 ||
		math.Abs(float64(got.Y-wantC1.Y)) > 1e-5 {
		t.Fatalf("quad c1 = %v, want %v", got, wantC1)
	}
	if got := patchPointAt(patch, 3); got != p2 {
		t.Fatalf("quad endpoint = %v, want %v", got, p2)
	}

	// Conic: {w, inf} in the last control point.
	w, alloc = newWriter(PatchAttribsNone, false, false)
	w.writeConic(p0, p1, p2, 0.75)
	last := patchPointAt(alloc.patches[0], 3)
	if last.X != 0.75 || !math.IsInf(float64(last.Y), 1) {
		t.Fatalf("conic last point = %v, want {w, inf}", last)
	}

	// Triangle: {inf, inf}, and the explicit curve type when the attrib is on.
	w, alloc = newWriter(PatchAttribExplicitCurveType, false, false)
	w.writeTriangle(p0, p1, p2)
	patch = alloc.patches[0]
	last = patchPointAt(patch, 3)
	if !math.IsInf(float64(last.X), 1) || !math.IsInf(float64(last.Y), 1) {
		t.Fatalf("triangle last point = %v, want {inf, inf}", last)
	}
	curveType := math.Float32frombits(binary.LittleEndian.Uint32(patch[32:]))
	if curveType != tessTriangularConicCurveType {
		t.Fatalf("triangle curve type = %v", curveType)
	}

	// Lines: written as the standard linear cubic, or discarded with DiscardFlatCurves.
	w, alloc = newWriter(PatchAttribsNone, false, false)
	w.writeLine(p0, p2)
	if len(alloc.patches) != 1 {
		t.Fatalf("line patches = %d, want 1", len(alloc.patches))
	}
	c1 := patchPointAt(alloc.patches[0], 1)
	if math.Abs(float64(c1.X)-10.0/3) > 1e-5 || c1.Y != 0 {
		t.Fatalf("line c1 = %v", c1)
	}
	w, alloc = newWriter(PatchAttribsNone, false, true)
	w.writeLine(p0, p2)
	w.writeQuadratic(p0, geom.Point{X: 5, Y: 0.01}, p2) // needs <= 1 segment: discarded too
	if len(alloc.patches) != 0 {
		t.Fatalf("discard-flat patches = %d, want 0", len(alloc.patches))
	}

	// Degenerate patches are skipped.
	w, alloc = newWriter(PatchAttribsNone, false, false)
	w.writeCubic(p0, p0, p0, p0)
	if len(alloc.patches) != 0 {
		t.Fatalf("degenerate patches = %d, want 0", len(alloc.patches))
	}

	// Fan point and byte color follow the control points in attrib order.
	w, alloc = newWriter(PatchAttribFanPoint|PatchAttribColor, false, false)
	w.updateFanPointAttrib(geom.Point{X: 3, Y: 4})
	w.updateColorAttrib(colorcore.PMColor4f{R: 1, G: 0.5, B: 0, A: 1})
	w.writeQuadratic(p0, p1, p2)
	patch = alloc.patches[0]
	if got := patchPointAt(patch, 4); got != (geom.Point{X: 3, Y: 4}) {
		t.Fatalf("fan point = %v", got)
	}
	if patch[40] != 255 || patch[41] != 128 || patch[42] != 0 || patch[43] != 255 {
		t.Fatalf("byte color = %v", patch[40:44])
	}

	// Wide color writes four floats.
	w, alloc = newWriter(PatchAttribColor|PatchAttribWideColorIfEnabled, false, false)
	w.updateColorAttrib(colorcore.PMColor4f{R: 2, G: 0.25, B: -1, A: 1})
	w.writeQuadratic(p0, p1, p2)
	patch = alloc.patches[0]
	if got := math.Float32frombits(binary.LittleEndian.Uint32(patch[32:])); got != 2 {
		t.Fatalf("wide color r = %v", got)
	}
	if got := math.Float32frombits(binary.LittleEndian.Uint32(patch[40:])); got != -1 {
		t.Fatalf("wide color b = %v", got)
	}
}

func TestPatchWriterChopping(t *testing.T) {
	// A quad requiring more than tessMaxParametricSegments but at most 2x chops into 2 quads (plus the connecting
	// triangles when AddTrianglesWhenChopping is on).
	p0 := geom.Point{X: 0, Y: 0}
	p1 := geom.Point{X: 3000, Y: 6000}
	p2 := geom.Point{X: 6000, Y: 0}
	n4 := wangsQuadraticP4(tessPrecision, p0, p1, p2, identityVectorXform())
	if n4 <= tessMaxParametricSegmentsP4 {
		t.Fatal("test quad must require chopping")
	}
	numPatches := int(math.Ceil(float64(wangsRoot4(min32(n4, tessMaxSegmentsPerCurveP4) /
		tessMaxParametricSegmentsP4))))

	alloc := &testPatchAllocator{stride: patchStride(PatchAttribsNone)}
	w := newPatchWriter(PatchAttribsNone, alloc, false, false)
	w.writeQuadratic(p0, p1, p2)
	if len(alloc.patches) != numPatches {
		t.Fatalf("chopped quad patches = %d, want %d", len(alloc.patches), numPatches)
	}
	// The chops join end-to-start and preserve the original endpoints.
	if got := patchPointAt(alloc.patches[0], 0); got != p0 {
		t.Fatalf("first chop start = %v", got)
	}
	for i := 1; i < len(alloc.patches); i++ {
		prevEnd := patchPointAt(alloc.patches[i-1], 3)
		curStart := patchPointAt(alloc.patches[i], 0)
		if prevEnd != curStart {
			t.Fatalf("chop %d does not join: %v vs %v", i, prevEnd, curStart)
		}
	}
	if got := patchPointAt(alloc.patches[len(alloc.patches)-1], 3); got != p2 {
		t.Fatalf("last chop end = %v", got)
	}

	// With AddTrianglesWhenChopping, the space between the new closing edges is filled: for numPatches == 2 that is
	// exactly one triangle patch {p0, abc, p2}.
	if numPatches == 2 {
		alloc = &testPatchAllocator{stride: patchStride(PatchAttribsNone)}
		w = newPatchWriter(PatchAttribsNone, alloc, true, true)
		w.writeQuadratic(p0, p1, p2)
		if len(alloc.patches) != 3 {
			t.Fatalf("chopped quad with triangles = %d patches, want 3", len(alloc.patches))
		}
		tri := alloc.patches[1]
		lastPt := patchPointAt(tri, 3)
		if !math.IsInf(float64(lastPt.X), 1) || !math.IsInf(float64(lastPt.Y), 1) {
			t.Fatalf("middle patch is not a triangle: %v", lastPt)
		}
		if patchPointAt(tri, 0) != p0 || patchPointAt(tri, 2) != p2 {
			t.Fatalf("connecting triangle = %v..%v", patchPointAt(tri, 0), patchPointAt(tri, 2))
		}
	}

	// A cubic that requires more than the max segments per curve still clamps its per-patch tolerance to the
	// fixed-count max.
	big0 := geom.Point{X: 0, Y: 0}
	big1 := geom.Point{X: 100000, Y: 200000}
	big2 := geom.Point{X: -100000, Y: 200000}
	big3 := geom.Point{X: 0, Y: 0}
	alloc = &testPatchAllocator{stride: patchStride(PatchAttribsNone)}
	w = newPatchWriter(PatchAttribsNone, alloc, false, false)
	w.writeCubic(big0, big1, big2, big3)
	if len(alloc.patches) == 0 {
		t.Fatal("big cubic wrote no patches")
	}
	if got := w.tolerances.requiredResolveLevel(); got != tessMaxResolveLevel {
		t.Fatalf("clamped resolve level = %d, want %d", got, tessMaxResolveLevel)
	}
	if got := fixedCountCurvesVertexCountForTolerances(&w.tolerances); got !=
		numCurveTrianglesAtResolveLevel(tessMaxResolveLevel)*3 {
		t.Fatalf("fixed-count vertex count = %d", got)
	}
}

func TestLinearTolerances(t *testing.T) {
	tol := newLinearTolerances()
	if got := tol.requiredResolveLevel(); got != 0 {
		t.Fatalf("default resolve level = %d", got)
	}
	tol.setParametricSegments(5 * 5 * 5 * 5) // n=5 -> next power of two is 8 -> level 3
	if got := tol.requiredResolveLevel(); got != 3 {
		t.Fatalf("resolve level for n=5: %d, want 3", got)
	}
	other := newLinearTolerances()
	other.setParametricSegments(32 * 32 * 32 * 32)
	tol.accumulate(&other)
	if got := tol.requiredResolveLevel(); got != 5 {
		t.Fatalf("accumulated resolve level = %d, want 5", got)
	}
	if got := fixedCountWedgesVertexCountForTolerances(&tol); got !=
		(numCurveTrianglesAtResolveLevel(5)+1)*3 {
		t.Fatalf("wedge vertex count = %d", got)
	}
}

func TestNonConvexFillOpSelection(t *testing.T) {
	identity := geom.IdentityMatrix()
	star := &path.Path{}
	star.MoveTo(50, 5).LineTo(70, 70).LineTo(10, 30).LineTo(90, 30).LineTo(30, 70).Close()

	// Small draw bounds: the stencil-cover op.
	op := makeNonConvexFillOp(FillPathFlagsNone, gpu.AATypeNone,
		geom.Rect{Left: 0, Top: 0, Right: 100, Bottom: 100},
		geom.IRect{Left: 0, Top: 0, Right: 128, Bottom: 128}, &identity, star,
		solidPaint(1, 0, 0, 0.5))
	if _, ok := op.(*pathStencilCoverOp); !ok {
		t.Fatalf("small bounds selected %T, want *pathStencilCoverOp", op)
	}

	// Large clipped bounds with few verbs: CPU fan triangulation wins.
	big := geom.Rect{Left: 0, Top: 0, Right: 1024, Bottom: 1024}
	op = makeNonConvexFillOp(FillPathFlagsNone, gpu.AATypeNone, big,
		geom.IRect{Left: 0, Top: 0, Right: 1024, Bottom: 1024}, &identity, star,
		solidPaint(1, 0, 0, 0.5))
	if _, ok := op.(*pathInnerTriangulateOp); !ok {
		t.Fatalf("large bounds selected %T, want *pathInnerTriangulateOp", op)
	}

	// Inverse fills always take the stencil-cover op.
	inv := star.Clone()
	inv.SetFillType(path.FillInverseWinding)
	op = makeNonConvexFillOp(FillPathFlagsNone, gpu.AATypeNone, big,
		geom.IRect{Left: 0, Top: 0, Right: 1024, Bottom: 1024}, &identity, inv,
		solidPaint(1, 0, 0, 0.5))
	if _, ok := op.(*pathStencilCoverOp); !ok {
		t.Fatalf("inverse fill selected %T, want *pathStencilCoverOp", op)
	}
}

// tessConvexBlob returns a volatile convex path with curves that is not a rect/oval/rrect, so the SDC's dedicated-op
// lanes pass on it and the renderer chain decides.
func tessConvexBlob(dx, dy float32) *path.Path {
	p := &path.Path{}
	p.MoveTo(10+dx, 20+dy)
	p.QuadTo(30+dx, 2+dy, 50+dx, 20+dy)
	p.LineTo(45+dx, 50+dy)
	p.LineTo(15+dx, 50+dy)
	p.Close()
	p.SetVolatile(true)
	return p
}

func TestPathTessellateOpBatching(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 128, 128)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	blob := tessConvexBlob(0, 0)
	if !blob.IsConvex() {
		t.Fatal("test blob must be convex")
	}

	// Two convex volatile draws with different translucent colors and identical trivial paints:
	// TessellationPathRenderer's PathTessellateOp merges them, moving color into patch attribs.
	shape1 := MakeStyledShapePath(blob, SimpleFillStyle(), DoSimplifyNo)
	sdc.DrawShape(nil, solidPaint(1, 0, 0, 0.5), gpu.AANo, &identity, &shape1)
	shape2 := MakeStyledShapePath(tessConvexBlob(60, 40), SimpleFillStyle(), DoSimplifyNo)
	sdc.DrawShape(nil, solidPaint(0, 0, 1, 0.5), gpu.AANo, &identity, &shape2)

	task := sdc.GetOpsTask()
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("op chains = %d, want 1 (merged)", n)
	}
	op, ok := task.GetChainHead(0).(*pathTessellateOp)
	if !ok {
		t.Fatalf("chain head is %T, want *pathTessellateOp", task.GetChainHead(0))
	}
	if op.patchAttribs&PatchAttribColor == 0 {
		t.Fatal("merged op with differing colors must move color into patch attribs")
	}
	if op.pathDrawList == nil || op.pathDrawList.next == nil {
		t.Fatal("merged op must carry both draws")
	}
	if op.totalCombinedPathVerbCnt != blob.CountVerbs()*2 {
		t.Fatalf("combined verb count = %d", op.totalCombinedPathVerbCnt)
	}

	// A draw with a different stencil/aa/processor state must not merge: coverage AA goes to a different renderer
	// entirely, so use a non-mergeable blend instead.
	shape3 := MakeStyledShapePath(tessConvexBlob(20, 60), SimpleFillStyle(), DoSimplifyNo)
	xorPaint := NewPaint()
	xorPaint.SetColor4f(colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 0.5})
	xorPaint.SetXPFactory(XPFactoryFromBlendMode(raster.BlendXor))
	sdc.DrawShape(nil, xorPaint, gpu.AANo, &identity, &shape3)
	if n := task.NumOpChains(); n != 2 {
		t.Fatalf("op chains after xor draw = %d, want 2", n)
	}
}
