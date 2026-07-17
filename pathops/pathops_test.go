// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pathops

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// rectPath builds a closed rect contour with the given fill type.
func rectPath(l, t, r, b float32, ft path.FillType) *path.Path {
	p := path.New().AddRect(geom.RectLTRB(l, t, r, b), geom.DirectionCW)
	p.SetFillType(ft)
	return p
}

// emptyPath builds an empty path with the given fill type.
func emptyPath(ft path.FillType) *path.Path {
	p := path.New()
	p.SetFillType(ft)
	return p
}

// polyPath builds one closed polygon contour.
func polyPath(ft path.FillType, pts ...geom.Point) *path.Path {
	p := path.New()
	p.MoveToPt(pts[0])
	for _, pt := range pts[1:] {
		p.LineToPt(pt)
	}
	p.Close()
	p.SetFillType(ft)
	return p
}

// assertLineOnly verifies line-only inputs yield line-only output (the faithful engine copies each segment's verb
// through addCurveTo, so a polygon op/simplify introduces no spurious curves).
func assertLineOnly(t *testing.T, p *path.Path) {
	t.Helper()
	it := path.NewRawIter(p)
	for {
		verb, _, _, ok := it.Next()
		if !ok {
			return
		}
		if verb != path.VerbMove && verb != path.VerbLine && verb != path.VerbClose {
			t.Fatalf("result contains verb %d; line-only input must stay line-only", verb)
		}
	}
}

// assertHasCurve verifies the faithful engine preserved at least one curve verb (conic/quad/cubic).
func assertHasCurve(t *testing.T, p *path.Path) {
	t.Helper()
	it := path.NewRawIter(p)
	for {
		verb, _, _, ok := it.Next()
		if !ok {
			break
		}
		if verb == path.VerbConic || verb == path.VerbQuad || verb == path.VerbCubic {
			return
		}
	}
	t.Fatal("result has no curve verbs; the faithful engine should preserve curves")
}

// contourCount counts move verbs.
func contourCount(p *path.Path) int {
	n := 0
	it := path.NewRawIter(p)
	for {
		verb, _, _, ok := it.Next()
		if !ok {
			return n
		}
		if verb == path.VerbMove {
			n++
		}
	}
}

// regionArea returns the area of the region a result path fills, independent of contour orientation and correct for the
// path's own fill rule. A plain signed shoelace isn't enough: pathops output makes no guarantee that holes and bowtie
// lobes are oriented oppositely to their outer boundary (an even-odd hole may wind the same way as its container, and a
// bowtie's two lobes wind oppositely though both are filled). So each contour's absolute area is signed by whether a
// point just inside that contour is filled — an outer boundary adds area, a hole subtracts it. Curves are flattened
// finely, so line-only output is measured exactly and curved output within flattening error.
func regionArea(t *testing.T, p *path.Path) float64 {
	t.Helper()
	var total float64
	var contour []geom.Point
	flush := func() {
		defer func() { contour = contour[:0] }()
		// A pathops contour is closed with an explicit line back to its start vertex before the close verb, so the
		// flattened point list ends with a copy of its first point. Drop it: otherwise the start vertex appears twice
		// and the topmost-vertex interior probe below degenerates (prev == vertex) when a contour starts at its own
		// topmost-leftmost point, nudging the probe onto the boundary.
		for len(contour) >= 2 && contour[len(contour)-1] == contour[0] {
			contour = contour[:len(contour)-1]
		}
		if len(contour) < 3 {
			return
		}
		var s float64
		for i := range contour {
			a, b := contour[i], contour[(i+1)%len(contour)]
			s += float64(a.X)*float64(b.Y) - float64(b.X)*float64(a.Y)
		}
		s *= 0.5
		// Pick a point just inside this contour via the inward bisector at its topmost vertex (whose interior angle is
		// < 180°, so the bisector points into the interior). Test the region's fill there.
		mi := 0
		for i, pt := range contour {
			if pt.Y < contour[mi].Y || (pt.Y == contour[mi].Y && pt.X < contour[mi].X) {
				mi = i
			}
		}
		v := contour[mi]
		prev := contour[(mi-1+len(contour))%len(contour)]
		next := contour[(mi+1)%len(contour)]
		dx, dy := unitVec(prev, v)
		ex, ey := unitVec(next, v)
		bx, by := dx+ex, dy+ey
		bl := math.Hypot(bx, by)
		var ix, iy float32
		if bl < 1e-9 {
			ix, iy = v.X, v.Y+0.01 // collinear spike: nudge downward into the interior (Y-down)
		} else {
			ix = v.X + float32(bx/bl)*0.01
			iy = v.Y + float32(by/bl)*0.01
		}
		if p.Contains(ix, iy) {
			total += math.Abs(s)
		} else {
			total -= math.Abs(s)
		}
	}
	it := path.NewRawIter(p)
	for {
		verb, pts, w, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			flush()
			contour = append(contour, pts[0])
		case path.VerbLine:
			contour = append(contour, pts[1])
		case path.VerbQuad:
			for k := 1; k <= 32; k++ {
				contour = append(contour, geom.EvalQuadAt(pts[:3], float32(k)/32))
			}
		case path.VerbConic:
			c := geom.MakeConic(pts[0], pts[1], pts[2], w)
			for k := 1; k <= 32; k++ {
				contour = append(contour, c.EvalAt(float32(k)/32))
			}
		case path.VerbCubic:
			for k := 1; k <= 32; k++ {
				contour = append(contour, geom.EvalCubicPosAt(pts[:4], float32(k)/32))
			}
		case path.VerbClose:
			flush()
		}
	}
	flush()
	return total
}

// unitVec returns the unit vector from b toward a (zero if coincident).
func unitVec(a, b geom.Point) (x, y float64) {
	dx, dy := float64(a.X-b.X), float64(a.Y-b.Y)
	l := math.Hypot(dx, dy)
	if l < 1e-12 {
		return 0, 0
	}
	return dx / l, dy / l
}

func checkArea(t *testing.T, label string, p *path.Path, want, tolerance float64) {
	t.Helper()
	if got := regionArea(t, p); math.Abs(got-want) > tolerance {
		t.Errorf("%s: area = %v, want %v (±%v)", label, got, want, tolerance)
	}
}

func TestOpRectIntersectFastPath(t *testing.T) {
	a := rectPath(0, 0, 10, 10, path.FillWinding)
	b := rectPath(5, 5, 15, 15, path.FillWinding)
	result, ok := Op(a, b, Intersect)
	if !ok {
		t.Fatal("intersect failed")
	}
	want := path.New().AddRect(geom.RectLTRB(5, 5, 10, 10), geom.DirectionCW)
	want.SetFillType(path.FillEvenOdd)
	if !path.Equal(result, want) {
		t.Errorf("rect fast path result mismatch: got %v", result)
	}
	if result.FillType() != path.FillEvenOdd {
		t.Errorf("fill type = %d, want even-odd", result.FillType())
	}

	// Disjoint and edge-touching rects produce empty results (rectangle intersection is false for both).
	for _, tc := range []struct {
		b    *path.Path
		name string
	}{
		{name: "disjoint", b: rectPath(20, 20, 30, 30, path.FillWinding)},
		{name: "touching", b: rectPath(10, 0, 20, 10, path.FillWinding)},
	} {
		result, ok = Op(a, tc.b, Intersect)
		if !ok {
			t.Fatalf("%s: intersect failed", tc.name)
		}
		if !result.IsEmpty() || result.FillType() != path.FillEvenOdd {
			t.Errorf("%s: got non-empty or wrong fill (%d)", tc.name, result.FillType())
		}
	}

	// Difference against an inverse rect remaps to intersect and must hit the same fast path: A \ ¬B = A ∩ B,
	// non-inverse result.
	binv := rectPath(5, 5, 15, 15, path.FillInverseWinding)
	result, ok = Op(a, binv, Difference)
	if !ok {
		t.Fatal("difference vs inverse failed")
	}
	if !path.Equal(result, want) {
		t.Error("remapped rect fast path mismatch")
	}

	// Union of two inverse rects remaps to intersect with an inverse result: ¬A ∪ ¬B = ¬(A ∩ B).
	ainv := rectPath(0, 0, 10, 10, path.FillInverseWinding)
	result, ok = Op(ainv, binv, Union)
	if !ok {
		t.Fatal("union of inverses failed")
	}
	wantInv := want.Clone()
	wantInv.SetFillType(path.FillInverseEvenOdd)
	if !path.Equal(result, wantInv) {
		t.Errorf("inverse union fast path mismatch: fill=%d", result.FillType())
	}
}

func TestOpEmptyOperands(t *testing.T) {
	empty := path.New()
	rect := rectPath(0, 0, 10, 10, path.FillWinding)
	wantRect := rect.Clone()
	wantRect.SetFillType(path.FillEvenOdd)

	cases := []struct {
		one  *path.Path
		two  *path.Path
		want *path.Path
		name string
		op   PathOp
	}{
		{name: "union empty+rect", one: empty, two: rect, op: Union, want: wantRect},
		{name: "union rect+empty", one: rect, two: empty, op: Union, want: wantRect},
		{name: "xor empty+rect", one: empty, two: rect, op: XOR, want: wantRect},
		{name: "intersect rect+empty", one: rect, two: empty, op: Intersect, want: emptyPath(path.FillEvenOdd)},
		{name: "diff empty-rect", one: empty, two: rect, op: Difference, want: emptyPath(path.FillEvenOdd)},
		{name: "diff rect-empty", one: rect, two: empty, op: Difference, want: wantRect},
		{name: "revdiff empty,rect", one: empty, two: rect, op: ReverseDifference, want: wantRect},
		{name: "revdiff rect,empty", one: rect, two: empty, op: ReverseDifference, want: emptyPath(path.FillEvenOdd)},
	}
	for _, tc := range cases {
		result, ok := Op(tc.one, tc.two, tc.op)
		if !ok {
			t.Errorf("%s: failed", tc.name)
			continue
		}
		if !path.Equal(result, tc.want) || result.FillType() != tc.want.FillType() {
			t.Errorf("%s: mismatch (fill=%d want %d, empty=%v want %v)",
				tc.name, result.FillType(), tc.want.FillType(), result.IsEmpty(), tc.want.IsEmpty())
		}
	}

	// Inverse bookkeeping through the empty lane: XOR(empty, ¬rect) keeps the region and the inverse flag.
	invRect := rectPath(0, 0, 10, 10, path.FillInverseWinding)
	result, ok := Op(empty, invRect, XOR)
	if !ok {
		t.Fatal("xor vs inverse failed")
	}
	if result.FillType() != path.FillInverseEvenOdd {
		t.Errorf("fill = %d, want inverse even-odd", result.FillType())
	}
	// Intersect(empty, ¬rect) remaps to difference(empty, rect): work stays empty, non-inverse result.
	result, ok = Op(empty, invRect, Intersect)
	if !ok {
		t.Fatal("intersect vs inverse failed")
	}
	if !result.IsEmpty() || result.FillType() != path.FillEvenOdd {
		t.Errorf("intersect(empty, inverse rect) should be empty even-odd (fill=%d)", result.FillType())
	}
}

func TestSimplifyConvexFastPath(t *testing.T) {
	oval := path.New().AddOval(geom.RectLTRB(0, 0, 20, 10), geom.DirectionCW)
	result, ok := Simplify(oval)
	if !ok {
		t.Fatal("simplify failed")
	}
	wantOval := oval.Clone()
	wantOval.SetFillType(path.FillEvenOdd)
	if !path.Equal(result, wantOval) {
		t.Error("convex path should be copied verbatim with even-odd fill")
	}

	// Trivially convex (degenerate) paths simplify to empty.
	for _, tc := range []struct {
		p    *path.Path
		name string
	}{
		{name: "point", p: path.New().MoveTo(3, 4).Close()},
		{name: "line", p: path.New().MoveTo(0, 0).LineTo(10, 10).Close()},
		{name: "collinear", p: path.New().MoveTo(0, 0).LineTo(5, 5).LineTo(10, 10)},
	} {
		result, ok = Simplify(tc.p)
		if !ok {
			t.Fatalf("%s: simplify failed", tc.name)
		}
		if !result.IsEmpty() || result.FillType() != path.FillEvenOdd {
			t.Errorf("%s: want empty even-odd, got empty=%v fill=%d", tc.name, result.IsEmpty(), result.FillType())
		}
	}

	// Inverse convex path: copied, inverse even-odd.
	invOval := oval.Clone()
	invOval.SetFillType(path.FillInverseWinding)
	result, ok = Simplify(invOval)
	if !ok {
		t.Fatal("inverse simplify failed")
	}
	if result.FillType() != path.FillInverseEvenOdd {
		t.Errorf("fill = %d, want inverse even-odd", result.FillType())
	}
}

func TestOpNonFinite(t *testing.T) {
	bad := path.New().MoveTo(0, 0).LineTo(float32(math.NaN()), 5).LineTo(10, 10).LineTo(0, 10).Close()
	good := polyPath(path.FillWinding,
		geom.Point{X: 0, Y: 0}, geom.Point{X: 8, Y: 3}, geom.Point{X: 10, Y: 10}, geom.Point{X: 2, Y: 8})
	if _, ok := Op(bad, good, Union); ok {
		t.Error("union with non-finite operand should fail")
	}
	if _, ok := Op(good, bad, Intersect); ok {
		t.Error("intersect with non-finite operand should fail")
	}
	if _, ok := Simplify(bad); ok {
		t.Error("simplify of a non-finite non-convex path should fail")
	}
}

// overlappingRectsAsPolys returns two overlapping rect regions built as pentagons (an extra midpoint vertex) so IsRect
// answers false and the engine — not the rect fast path — runs.
func overlappingRectsAsPolys() (p1, p2 *path.Path) {
	a := polyPath(path.FillWinding,
		geom.Point{X: 0, Y: 0}, geom.Point{X: 5, Y: 0}, geom.Point{X: 10, Y: 0},
		geom.Point{X: 10, Y: 10}, geom.Point{X: 0, Y: 10})
	b := polyPath(path.FillWinding,
		geom.Point{X: 5, Y: 0}, geom.Point{X: 10, Y: 0}, geom.Point{X: 15, Y: 0},
		geom.Point{X: 15, Y: 10}, geom.Point{X: 5, Y: 10})
	return a, b
}

func TestOpBasicAreas(t *testing.T) {
	a, b := overlappingRectsAsPolys() // A = [0,10]x[0,10], B = [5,15]x[0,10], overlap 5x10
	cases := []struct {
		op   PathOp
		want float64
	}{
		{op: Union, want: 150},
		{op: Intersect, want: 50},
		{op: Difference, want: 50},
		{op: ReverseDifference, want: 50},
		{op: XOR, want: 100},
	}
	for _, tc := range cases {
		result, ok := Op(a, b, tc.op)
		if !ok {
			t.Fatalf("op %d failed", tc.op)
		}
		assertLineOnly(t, result)
		if result.FillType() != path.FillEvenOdd {
			t.Errorf("op %d: fill = %d, want even-odd", tc.op, result.FillType())
		}
		checkArea(t, fmt.Sprintf("op %d", tc.op), result, tc.want, 0.02)
	}

	// Sample-point sanity for each op.
	inA := geom.Point{X: 2, Y: 5}
	inBoth := geom.Point{X: 7, Y: 5}
	inB := geom.Point{X: 13, Y: 5}
	outside := geom.Point{X: 20, Y: 5}
	for _, tc := range []struct {
		op   PathOp
		want [4]bool // inA, inBoth, inB, outside
	}{
		{op: Union, want: [4]bool{true, true, true, false}},
		{op: Intersect, want: [4]bool{false, true, false, false}},
		{op: Difference, want: [4]bool{true, false, false, false}},
		{op: ReverseDifference, want: [4]bool{false, false, true, false}},
		{op: XOR, want: [4]bool{true, false, true, false}},
	} {
		result, _ := Op(a, b, tc.op)
		for i, pt := range []geom.Point{inA, inBoth, inB, outside} {
			if got := result.Contains(pt.X, pt.Y); got != tc.want[i] {
				t.Errorf("op %d: contains(%v) = %v, want %v", tc.op, pt, got, tc.want[i])
			}
		}
	}
}

func TestOpSharedEdge(t *testing.T) {
	// Two rects sharing the x=10 edge: the union has no interior edge and one 4-corner contour.
	a := polyPath(path.FillWinding,
		geom.Point{X: 0, Y: 0}, geom.Point{X: 10, Y: 0}, geom.Point{X: 10, Y: 10},
		geom.Point{X: 0, Y: 10}, geom.Point{X: 0, Y: 5})
	b := polyPath(path.FillWinding,
		geom.Point{X: 10, Y: 0}, geom.Point{X: 20, Y: 0}, geom.Point{X: 20, Y: 10},
		geom.Point{X: 10, Y: 10}, geom.Point{X: 10, Y: 5})
	result, ok := Op(a, b, Union)
	if !ok {
		t.Fatal("union failed")
	}
	checkArea(t, "shared-edge union", result, 200, 0.03)
	if n := contourCount(result); n != 1 {
		t.Errorf("contours = %d, want 1 (shared edge must cancel)", n)
	}
	if !result.Contains(10, 5) {
		t.Error("point on the canceled shared edge should be inside the union")
	}
	// Their intersection is a zero-area line: empty result.
	result, ok = Op(a, b, Intersect)
	if !ok {
		t.Fatal("intersect failed")
	}
	if !result.IsEmpty() {
		t.Errorf("shared-edge intersect should be empty, area=%v", regionArea(t, result))
	}
}

func TestOpCornerTouch(t *testing.T) {
	// Two squares touching at one corner. The union active-edge rule threads the trace through the shared vertex, so
	// the result is a single contour pinched at the corner. The region is identical either way, verified by
	// containment.
	a := polyPath(path.FillWinding,
		geom.Point{X: 0, Y: 0}, geom.Point{X: 10, Y: 0}, geom.Point{X: 10, Y: 10},
		geom.Point{X: 0, Y: 10}, geom.Point{X: 0, Y: 5})
	b := polyPath(path.FillWinding,
		geom.Point{X: 10, Y: 10}, geom.Point{X: 20, Y: 10}, geom.Point{X: 20, Y: 20},
		geom.Point{X: 10, Y: 20}, geom.Point{X: 10, Y: 15})
	result, ok := Op(a, b, Union)
	if !ok {
		t.Fatal("union failed")
	}
	checkArea(t, "corner-touch union", result, 200, 0.03)
	if n := contourCount(result); n != 1 {
		t.Errorf("contours = %d, want 1 (pinched at the shared corner)", n)
	}
	// Diagonal neighbors of the shared corner stay outside.
	if result.Contains(10.5, 9.5) || result.Contains(9.5, 10.5) {
		t.Error("points in the diagonal quadrants of the shared corner must be outside")
	}
	if !result.Contains(9.5, 9.5) || !result.Contains(10.5, 10.5) {
		t.Error("points inside each square must be inside")
	}
}

func TestOpCheckerboardXOR(t *testing.T) {
	// XOR of two rects overlapping in a middle square: two L-shaped contours that touch at two checkerboard corners and
	// must not merge.
	a, _ := overlappingRectsAsPolys()
	b := polyPath(path.FillWinding,
		geom.Point{X: 5, Y: 5}, geom.Point{X: 15, Y: 5}, geom.Point{X: 15, Y: 15},
		geom.Point{X: 5, Y: 15}, geom.Point{X: 5, Y: 10})
	result, ok := Op(a, b, XOR)
	if !ok {
		t.Fatal("xor failed")
	}
	assertLineOnly(t, result)
	checkArea(t, "checkerboard xor", result, 150, 0.03)
	if n := contourCount(result); n != 2 {
		t.Errorf("contours = %d, want 2", n)
	}
	// The overlap square is outside; both L bodies are inside.
	if result.Contains(7.5, 7.5) {
		t.Error("overlap square must be outside the xor")
	}
	if !result.Contains(2, 2) || !result.Contains(12, 12) {
		t.Error("both L bodies must be inside the xor")
	}
}

func TestSimplifySelfIntersections(t *testing.T) {
	// Five-pointed star drawn with crossing lines: winding fills the core, even-odd leaves it hollow.
	star := func(ft path.FillType) *path.Path {
		pts := make([]geom.Point, 5)
		for i := range pts {
			ang := float64(i*4) * math.Pi / 5 // every other vertex
			pts[i] = geom.Point{
				X: 50 + 40*float32(math.Sin(ang)),
				Y: 50 - 40*float32(math.Cos(ang)),
			}
		}
		return polyPath(ft, pts...)
	}
	result, ok := Simplify(star(path.FillWinding))
	if !ok {
		t.Fatal("winding star simplify failed")
	}
	assertLineOnly(t, result)
	if !result.Contains(50, 50) {
		t.Error("winding star core must be filled")
	}
	if !result.Contains(50, 15) {
		t.Error("star point must be filled")
	}
	if result.Contains(2, 2) {
		t.Error("outside must stay empty")
	}
	windingArea := regionArea(t, result)

	result, ok = Simplify(star(path.FillEvenOdd))
	if !ok {
		t.Fatal("even-odd star simplify failed")
	}
	if result.Contains(50, 50) {
		t.Error("even-odd star core must be hollow")
	}
	if !result.Contains(50, 15) {
		t.Error("star point must be filled")
	}
	if evenOddArea := regionArea(t, result); evenOddArea >= windingArea {
		t.Errorf("even-odd area %v should be less than winding area %v", evenOddArea, windingArea)
	}

	// Figure-8: both lobes fill under winding.
	fig8 := polyPath(path.FillWinding,
		geom.Point{X: 0, Y: 0}, geom.Point{X: 10, Y: 10}, geom.Point{X: 0, Y: 10},
		geom.Point{X: 10, Y: 0})
	result, ok = Simplify(fig8)
	if !ok {
		t.Fatal("figure-8 simplify failed")
	}
	if !result.Contains(5, 8) || !result.Contains(5, 2) {
		t.Error("both figure-8 lobes must be filled")
	}
	if result.Contains(1, 5) || result.Contains(9, 5) {
		t.Error("the waist quadrants of the figure-8 must be empty")
	}
	checkArea(t, "figure-8", result, 50, 0.02)

	// Nested same-direction rects: winding keeps the outer region solid; even-odd carves a hole.
	nested := path.New().
		AddRect(geom.RectLTRB(0, 0, 20, 20), geom.DirectionCW).
		AddRect(geom.RectLTRB(5, 5, 15, 15), geom.DirectionCW)
	result, ok = Simplify(nested)
	if !ok {
		t.Fatal("nested winding simplify failed")
	}
	checkArea(t, "nested winding", result, 400, 0.03)
	if n := contourCount(result); n != 1 {
		t.Errorf("nested winding contours = %d, want 1", n)
	}
	if !result.Contains(10, 10) {
		t.Error("inner region must stay filled under winding")
	}
	nested.SetFillType(path.FillEvenOdd)
	result, ok = Simplify(nested)
	if !ok {
		t.Fatal("nested even-odd simplify failed")
	}
	checkArea(t, "nested even-odd", result, 300, 0.03)
	if n := contourCount(result); n != 2 {
		t.Errorf("nested even-odd contours = %d, want 2", n)
	}
	if result.Contains(10, 10) {
		t.Error("inner region must be a hole under even-odd")
	}
}

func TestSimplifyOverlappingContours(t *testing.T) {
	// Two overlapping rect contours in one path: winding unions them; even-odd xors them.
	p := path.New().
		AddRect(geom.RectLTRB(0, 0, 10, 10), geom.DirectionCW).
		AddRect(geom.RectLTRB(5, 0, 15, 10), geom.DirectionCW)
	result, ok := Simplify(p)
	if !ok {
		t.Fatal("winding simplify failed")
	}
	checkArea(t, "overlap winding", result, 150, 0.02)
	p.SetFillType(path.FillEvenOdd)
	result, ok = Simplify(p)
	if !ok {
		t.Fatal("even-odd simplify failed")
	}
	checkArea(t, "overlap even-odd", result, 100, 0.02)
	if result.Contains(7.5, 5) {
		t.Error("overlap must vanish under even-odd")
	}
}

func TestOpCurves(t *testing.T) {
	// Circle vs rect through the real engine: area of the intersection is a half disc.
	circle := path.New().AddCircle(50, 50, 40, geom.DirectionCW)
	rect := polyPath(path.FillWinding,
		geom.Point{X: 50, Y: 0}, geom.Point{X: 100, Y: 0}, geom.Point{X: 100, Y: 100},
		geom.Point{X: 50, Y: 100}, geom.Point{X: 50, Y: 50})
	result, ok := Op(circle, rect, Intersect)
	if !ok {
		t.Fatal("circle∩rect failed")
	}
	// The faithful engine preserves the circle's conic arcs (the interim engine emitted line-only output).
	assertHasCurve(t, result)
	halfDisc := math.Pi * 40 * 40 / 2
	// Tolerance covers regionArea's 32-segment flattening of the arcs.
	checkArea(t, "half disc", result, halfDisc, 4)
	if !result.Contains(70, 50) || result.Contains(30, 50) || result.Contains(95, 5) {
		t.Error("half-disc membership wrong")
	}

	// Union of two discs (lens overlap) has the inclusion-exclusion area.
	c1 := path.New().AddCircle(40, 50, 30, geom.DirectionCW)
	c2 := path.New().AddCircle(80, 50, 30, geom.DirectionCW)
	result, ok = Op(c1, c2, Union)
	if !ok {
		t.Fatal("disc union failed")
	}
	d := 40.0
	r := 30.0
	lens := 2*r*r*math.Acos(d/(2*r)) - d/2*math.Sqrt(4*r*r-d*d)
	checkArea(t, "disc union", result, 2*math.Pi*r*r-lens, 5)
	if n := contourCount(result); n != 1 {
		t.Errorf("disc union contours = %d, want 1", n)
	}
}

func TestOpDeterminism(t *testing.T) {
	a, b := overlappingRectsAsPolys()
	r1, ok1 := Op(a, b, XOR)
	r2, ok2 := Op(a, b, XOR)
	if !ok1 || !ok2 || !path.Equal(r1, r2) {
		t.Error("op must be deterministic")
	}
}

func TestBuilder(t *testing.T) {
	// Empty builder resolves to an empty even-odd path.
	var b Builder
	result, ok := b.Resolve()
	if !ok || !result.IsEmpty() || result.FillType() != path.FillEvenOdd {
		t.Errorf("empty builder: ok=%v empty=%v fill=%d", ok, result.IsEmpty(), result.FillType())
	}

	// First add with a non-union op pads with an empty path: empty ∩ rect = empty.
	b.Add(rectPath(0, 0, 10, 10, path.FillWinding), Intersect)
	result, ok = b.Resolve()
	if !ok || !result.IsEmpty() {
		t.Errorf("intersect-first builder should resolve empty (ok=%v)", ok)
	}

	// A single all-union convex path runs Resolve's all-union branch (Simplify → fixWinding → Simplify), yielding the
	// same region with even-odd fill. fixWinding reverses the CW operand to a winding-consistent orientation (verified
	// bit-exactly by the oracle's TestPathOpsBuilderConvexUnionExact); here we assert the orientation-independent
	// region.
	rect := rectPath(0, 0, 10, 10, path.FillWinding)
	b.Add(rect, Union)
	result, ok = b.Resolve()
	if !ok {
		t.Fatal("single union resolve failed")
	}
	if result.FillType() != path.FillEvenOdd {
		t.Errorf("single all-union fill = %d, want even-odd", result.FillType())
	}
	if n := contourCount(result); n != 1 {
		t.Errorf("single all-union contours = %d, want 1", n)
	}
	checkArea(t, "single all-union rect", result, 100, 0.02)
	if !result.Contains(5, 5) || result.Contains(20, 20) {
		t.Error("single all-union rect membership wrong")
	}

	// A single inverse path is not all-union: returned unmodified, original fill preserved.
	inv := rectPath(0, 0, 10, 10, path.FillInverseWinding)
	b.Add(inv, Union)
	result, ok = b.Resolve()
	if !ok {
		t.Fatal("single inverse resolve failed")
	}
	if result.FillType() != path.FillInverseWinding || !path.Equal(result, inv) {
		t.Errorf("single non-all-union path must pass through unmodified (fill=%d)", result.FillType())
	}

	// Union chain of three disjoint squares.
	b.Add(rectPath(0, 0, 10, 10, path.FillWinding), Union)
	b.Add(rectPath(20, 0, 30, 10, path.FillWinding), Union)
	b.Add(rectPath(40, 0, 50, 10, path.FillWinding), Union)
	result, ok = b.Resolve()
	if !ok {
		t.Fatal("union chain failed")
	}
	checkArea(t, "union chain", result, 300, 0.05)
	if n := contourCount(result); n != 3 {
		t.Errorf("contours = %d, want 3", n)
	}

	// Mixed sequence: (A ∪ B) \ C.
	b.Add(rectPath(0, 0, 10, 10, path.FillWinding), Union)
	b.Add(rectPath(10, 0, 20, 10, path.FillWinding), Union)
	b.Add(rectPath(5, 0, 15, 10, path.FillWinding), Difference)
	result, ok = b.Resolve()
	if !ok {
		t.Fatal("mixed resolve failed")
	}
	checkArea(t, "mixed builder", result, 100, 0.05)
	if !result.Contains(2, 5) || !result.Contains(18, 5) || result.Contains(10, 5) {
		t.Error("mixed builder membership wrong")
	}

	// Resolve resets the builder.
	result, ok = b.Resolve()
	if !ok || !result.IsEmpty() {
		t.Error("builder must reset after resolve")
	}
}

// TestOpContainsProperty cross-checks the engine against direct point classification of the inputs: for random polygon
// pairs and every op, result.Contains must equal the op formula over the operands' own Contains for sample points away
// from all input edges (the interim contract perturbs boundaries by less than the snap grid; samples keep a wide
// margin).
func TestOpContainsProperty(t *testing.T) {
	rng := rand.New(rand.NewSource(28))
	fillTypes := []path.FillType{path.FillWinding, path.FillEvenOdd, path.FillInverseWinding, path.FillInverseEvenOdd}
	formula := func(op PathOp, inA, inB bool) bool {
		switch op {
		case Difference:
			return inA && !inB
		case Intersect:
			return inA && inB
		case Union:
			return inA || inB
		case XOR:
			return inA != inB
		default:
			return !inA && inB
		}
	}
	randPoly := func(ft path.FillType) (*path.Path, []geom.Point) {
		n := 5 + rng.Intn(8)
		pts := make([]geom.Point, n)
		for i := range pts {
			// Half-unit grid keeps the inputs representable and the margin test honest.
			pts[i] = geom.Point{
				X: float32(rng.Intn(200)) * 0.5,
				Y: float32(rng.Intn(200)) * 0.5,
			}
		}
		return polyPath(ft, pts...), pts
	}
	segDist := func(p, a, b geom.Point) float64 {
		px, py := float64(p.X-a.X), float64(p.Y-a.Y)
		dx, dy := float64(b.X-a.X), float64(b.Y-a.Y)
		len2 := dx*dx + dy*dy
		var t float64
		if len2 > 0 {
			t = math.Min(1, math.Max(0, (px*dx+py*dy)/len2))
		}
		ex, ey := px-t*dx, py-t*dy
		return math.Hypot(ex, ey)
	}
	const margin = 0.25
	for trial := 0; trial < 40; trial++ {
		ftA := fillTypes[rng.Intn(len(fillTypes))]
		ftB := fillTypes[rng.Intn(len(fillTypes))]
		a, aPts := randPoly(ftA)
		bp, bPts := randPoly(ftB)
		for op := Difference; op <= ReverseDifference; op++ {
			result, ok := Op(a, bp, op)
			if !ok {
				t.Fatalf("trial %d op %d failed", trial, op)
			}
			assertLineOnly(t, result)
			for sy := float32(-5); sy <= 105; sy += 5.5 {
				for sx := float32(-5); sx <= 105; sx += 5.5 {
					pt := geom.Point{X: sx, Y: sy}
					nearEdge := false
					for i := range aPts {
						if segDist(pt, aPts[i], aPts[(i+1)%len(aPts)]) < margin {
							nearEdge = true
							break
						}
					}
					if !nearEdge {
						for i := range bPts {
							if segDist(pt, bPts[i], bPts[(i+1)%len(bPts)]) < margin {
								nearEdge = true
								break
							}
						}
					}
					if nearEdge {
						continue
					}
					want := formula(op, a.Contains(sx, sy), bp.Contains(sx, sy))
					if got := result.Contains(sx, sy); got != want {
						t.Fatalf("trial %d op %d fills(%d,%d): contains(%v,%v) = %v, want %v",
							trial, op, ftA, ftB, sx, sy, got, want)
					}
				}
			}
		}
	}
}

// TestSimplifyContainsProperty is the single-path analog over self-intersecting random polygons.
func TestSimplifyContainsProperty(t *testing.T) {
	rng := rand.New(rand.NewSource(56))
	for trial := 0; trial < 40; trial++ {
		n := 6 + rng.Intn(8)
		pts := make([]geom.Point, n)
		for i := range pts {
			pts[i] = geom.Point{X: float32(rng.Intn(200)) * 0.5, Y: float32(rng.Intn(200)) * 0.5}
		}
		ft := path.FillWinding
		if trial%2 == 1 {
			ft = path.FillEvenOdd
		}
		p := polyPath(ft, pts...)
		if p.GetConvexity().IsConvex() {
			continue // fast path, exercised elsewhere
		}
		result, ok := Simplify(p)
		if !ok {
			t.Fatalf("trial %d simplify failed", trial)
		}
		segDist := func(pt, a, b geom.Point) float64 {
			px, py := float64(pt.X-a.X), float64(pt.Y-a.Y)
			dx, dy := float64(b.X-a.X), float64(b.Y-a.Y)
			len2 := dx*dx + dy*dy
			var t2 float64
			if len2 > 0 {
				t2 = math.Min(1, math.Max(0, (px*dx+py*dy)/len2))
			}
			return math.Hypot(px-t2*dx, py-t2*dy)
		}
		for sy := float32(-5); sy <= 105; sy += 5.5 {
			for sx := float32(-5); sx <= 105; sx += 5.5 {
				pt := geom.Point{X: sx, Y: sy}
				nearEdge := false
				for i := range pts {
					if segDist(pt, pts[i], pts[(i+1)%len(pts)]) < 0.25 {
						nearEdge = true
						break
					}
				}
				if nearEdge {
					continue
				}
				if got, want := result.Contains(sx, sy), p.Contains(sx, sy); got != want {
					t.Fatalf("trial %d fill %d: contains(%v,%v) = %v, want %v", trial, ft, sx, sy, got, want)
				}
			}
		}
	}
}
