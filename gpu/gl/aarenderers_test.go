// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for the AA path renderers: AAConvexPathRenderer's segment extraction / degenerate detection / vertex
// generation and the quadUVMatrix + conic KLM + constrained cubic-to-quad conversion machinery it consumes,
// aaConvexTessellator's ring structure for fills/strokes, AAHairLinePathRenderer's gather/bloat/add-line geometry, the
// renderer selection gates, the chain order, and record-time op merging on the fake driver.

package gl

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// convexPentagonPath is a convex pentagon that no dedicated draw op takes (not a rect, oval, or rrect), so it exercises
// the path-renderer chain.
func convexPentagonPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(50, 8)
	p.LineTo(90, 38)
	p.LineTo(75, 88)
	p.LineTo(25, 88)
	p.LineTo(10, 38)
	p.Close()
	return p
}

func squarePath(l, t, r, b float32) *path.Path {
	p := &path.Path{}
	p.MoveTo(l, t)
	p.LineTo(r, t)
	p.LineTo(r, b)
	p.LineTo(l, b)
	p.Close()
	return p
}

//////////////////////////////////////////////////////////////////////////////
// Path utility helpers

// QuadUVMatrix must map the quad's control points to the canonical (0,0), (1/2,0), (1,1).
func TestQuadUVMatrixCanonical(t *testing.T) {
	qpts := [3]geom.Point{{X: 10, Y: 20}, {X: 30, Y: 5}, {X: 50, Y: 40}}
	var m quadUVMatrix
	m.set(&qpts)
	var uv [3]geom.Point
	m.apply(qpts[:], uv[:])
	want := [3]geom.Point{{X: 0, Y: 0}, {X: 0.5, Y: 0}, {X: 1, Y: 1}}
	for i := range uv {
		if math.Abs(float64(uv[i].X-want[i].X)) > 1e-3 ||
			math.Abs(float64(uv[i].Y-want[i].Y)) > 1e-3 {
			t.Fatalf("control point %d maps to %v, want %v", i, uv[i], want[i])
		}
	}
}

// A degenerate (colinear) quad gets the u=0, v=distance-to-line form; a point quad maps everything far away.
func TestQuadUVMatrixDegenerate(t *testing.T) {
	line := [3]geom.Point{{X: 0, Y: 0}, {X: 5, Y: 0}, {X: 10, Y: 0}}
	var m quadUVMatrix
	m.set(&line)
	if m.m[0] != 0 || m.m[1] != 0 || m.m[2] != 0 {
		t.Fatalf("degenerate quad first row %v want zeros", m.m[:3])
	}
	// v is proportional to the signed distance to the line (the line vector is left unnormalized, so it is scaled by
	// the longest edge's length — here 10).
	pos := []geom.Point{{X: 3, Y: 4}}
	uv := make([]geom.Point, 1)
	m.apply(pos, uv)
	if math.Abs(math.Abs(float64(uv[0].Y))-40) > 1e-3 || uv[0].X != 0 {
		t.Fatalf("degenerate quad uv %v want (0, ±40)", uv[0])
	}

	pt := [3]geom.Point{{X: 7, Y: 7}, {X: 7, Y: 7}, {X: 7, Y: 7}}
	m.set(&pt)
	m.apply(pt[:1], uv)
	if uv[0].X != 100 || uv[0].Y != 100 {
		t.Fatalf("point quad uv %v want (100, 100)", uv[0])
	}
}

// getConicKLM: coefficients scale to max |coeff| = 10 and satisfy K^2 - L*M = 0 on the curve.
func TestConicKLM(t *testing.T) {
	pts := [3]geom.Point{{X: 0, Y: 0}, {X: 20, Y: 0}, {X: 20, Y: 20}}
	w := float32(0.8)
	klm := getConicKLM(&pts, w)

	maxAbs := float32(0)
	for _, v := range klm {
		maxAbs = max(maxAbs, geom.ScalarAbs(v))
	}
	if math.Abs(float64(maxAbs)-10) > 1e-4 {
		t.Fatalf("max |klm| = %v want 10", maxAbs)
	}

	conic := geom.MakeConic(pts[0], pts[1], pts[2], w)
	for _, tv := range []float32{0.1, 0.3, 0.5, 0.7, 0.9} {
		p := conic.EvalAt(tv)
		k := klm[0]*p.X + klm[1]*p.Y + klm[2]
		l := klm[3]*p.X + klm[4]*p.Y + klm[5]
		m := klm[6]*p.X + klm[7]*p.Y + klm[8]
		f := float64(k*k - l*m)
		scale := math.Max(math.Abs(float64(k*k))+math.Abs(float64(l*m)), 1)
		if math.Abs(f)/scale > 1e-3 {
			t.Fatalf("t=%v: K^2-LM = %v not ~0 (scale %v)", tv, f, scale)
		}
	}
}

// convertCubicToQuadsConstrainToTangents: emits a connected chain of quads from p0 to p3.
func TestConvertCubicToQuadsConstrainToTangents(t *testing.T) {
	cubic := [4]geom.Point{{X: 0, Y: 0}, {X: 10, Y: -20}, {X: 40, Y: -20}, {X: 50, Y: 0}}
	for _, dir := range []path.FirstDirection{path.FirstDirectionCW, path.FirstDirectionCCW} {
		var quads []geom.Point
		convertCubicToQuadsConstrainToTangents(&cubic, 1, dir, &quads)
		if len(quads) == 0 || len(quads)%3 != 0 {
			t.Fatalf("dir %d: quad point count %d not a positive multiple of 3", dir,
				len(quads))
		}
		if quads[0] != cubic[0] || quads[len(quads)-1] != cubic[3] {
			t.Fatalf("dir %d: chain endpoints %v %v want %v %v", dir, quads[0],
				quads[len(quads)-1], cubic[0], cubic[3])
		}
		for i := 3; i < len(quads); i += 3 {
			if quads[i] != quads[i-1] {
				t.Fatalf("dir %d: quad %d does not start where quad %d ended", dir, i/3,
					i/3-1)
			}
		}
		for _, p := range quads {
			if !p.IsFinite() {
				t.Fatalf("dir %d: non-finite quad point", dir)
			}
		}
	}
}

// convertCubicToQuads (unconstrained): same chain properties.
func TestConvertCubicToQuads(t *testing.T) {
	cubic := [4]geom.Point{{X: 0, Y: 0}, {X: 10, Y: 20}, {X: 40, Y: -20}, {X: 50, Y: 0}}
	var quads []geom.Point
	convertCubicToQuads(&cubic, 1, &quads)
	if len(quads) == 0 || len(quads)%3 != 0 {
		t.Fatalf("quad point count %d not a positive multiple of 3", len(quads))
	}
	if quads[0] != cubic[0] || quads[len(quads)-1] != cubic[3] {
		t.Fatalf("chain endpoints %v %v want %v %v", quads[0], quads[len(quads)-1], cubic[0],
			cubic[3])
	}
}

//////////////////////////////////////////////////////////////////////////////
// AAConvexPathRenderer geometry

// A CW square decomposes into 4 line segments with outward unit normals, the centroid fan point, and the per-segment
// vertex/index counts.
func TestAAConvexGetSegmentsSquare(t *testing.T) {
	p := squarePath(10, 10, 50, 50)
	identity := geom.IdentityMatrix()
	var segments []aaConvexSegment
	var fanPt geom.Point
	var vCount, iCount int
	if !aaConvexGetSegments(p, &identity, &segments, &fanPt, &vCount, &iCount) {
		t.Fatal("getSegments failed on a square")
	}
	if len(segments) != 4 {
		t.Fatalf("segment count %d want 4", len(segments))
	}
	// 4 line segments: (5 verts + 9 idxs) each, plus a corner wedge (4 verts + 6 idxs) each.
	if vCount != 36 || iCount != 60 {
		t.Fatalf("counts %d/%d want 36/60", vCount, iCount)
	}
	if math.Abs(float64(fanPt.X-30)) > 1e-3 || math.Abs(float64(fanPt.Y-30)) > 1e-3 {
		t.Fatalf("fan point %v want (30, 30)", fanPt)
	}
	for i := range segments {
		if segments[i].typ != 0 {
			t.Fatalf("segment %d is not a line", i)
		}
		n := segments[i].norms[0]
		if math.Abs(float64(n.Length())-1) > 1e-4 {
			t.Fatalf("segment %d normal %v not unit", i, n)
		}
		// Outward: the normal must point away from the centroid at the segment end point.
		if n.Dot(segments[i].endPt().Sub(fanPt)) <= 0 {
			t.Fatalf("segment %d normal %v points inward", i, n)
		}
		if math.Abs(float64(segments[i].mid.Length())-1) > 1e-4 {
			t.Fatalf("segment %d mid %v not unit", i, segments[i].mid)
		}
	}
}

// Nearly-line (zero area) paths are detected as degenerate and rejected.
func TestAAConvexGetSegmentsDegenerate(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(0, 0)
	p.LineTo(100, 0.01)
	p.LineTo(0, 0.02)
	p.Close()
	identity := geom.IdentityMatrix()
	var segments []aaConvexSegment
	var fanPt geom.Point
	var vCount, iCount int
	if aaConvexGetSegments(p, &identity, &segments, &fanPt, &vCount, &iCount) {
		t.Fatal("getSegments accepted a degenerate (nearly-line) path")
	}
}

// Direction detection: a CCW path flips the normal side; a negative-determinant matrix flips the direction.
func TestAAConvexGetDirection(t *testing.T) {
	cw := squarePath(0, 0, 10, 10)
	identity := geom.IdentityMatrix()
	dir, ok := aaConvexGetDirection(cw, &identity)
	if !ok || dir != path.FirstDirectionCW {
		t.Fatalf("square direction %v/%v want CW", dir, ok)
	}
	var flip geom.Matrix
	flip.SetScale(-1, 1)
	dir, ok = aaConvexGetDirection(cw, &flip)
	if !ok || dir != path.FirstDirectionCCW {
		t.Fatalf("flipped square direction %v/%v want CCW", dir, ok)
	}
}

// Quads: an obviously-curved quad stays a quad segment; a nearly-straight one collapses to a line segment.
func TestAAConvexAddQuadSegment(t *testing.T) {
	var segments []aaConvexSegment
	curved := [3]geom.Point{{X: 0, Y: 0}, {X: 10, Y: 10}, {X: 20, Y: 0}}
	aaConvexAddQuadSegment(&curved, &segments)
	if len(segments) != 1 || segments[0].typ != 1 {
		t.Fatalf("curved quad produced %+v, want one quad segment", segments)
	}
	segments = segments[:0]
	flat := [3]geom.Point{{X: 0, Y: 0}, {X: 10, Y: 0.001}, {X: 20, Y: 0}}
	aaConvexAddQuadSegment(&flat, &segments)
	if len(segments) != 1 || segments[0].typ != 0 || segments[0].pts[0] != flat[2] {
		t.Fatalf("flat quad produced %+v, want one line segment to %v", segments, flat[2])
	}
}

// create_vertices fills exactly the predicted vertex/index counts with draw-relative indices in range.
func TestAAConvexCreateVertices(t *testing.T) {
	p := convexPentagonPath()
	identity := geom.IdentityMatrix()
	var segments []aaConvexSegment
	var fanPt geom.Point
	var vCount, iCount int
	if !aaConvexGetSegments(p, &identity, &segments, &fanPt, &vCount, &iCount) {
		t.Fatal("getSegments failed on the pentagon")
	}
	const stride = 28 // pos(8) + color(4) + uv(8) + d0(4) + d1(4)
	buf := make([]byte, vCount*stride)
	idxs := make([]uint16, iCount)
	var draws []aaConvexDraw
	writer := &quadVertexWriter{buf: buf}
	aaConvexCreateVertices(segments, fanPt, colorcore.PMColor4f{R: 1, A: 1}, false, &draws,
		writer, idxs)
	if len(draws) != 1 {
		t.Fatalf("draw count %d want 1", len(draws))
	}
	if draws[0].vertexCnt != vCount || draws[0].indexCnt != iCount {
		t.Fatalf("draw counts %d/%d want %d/%d", draws[0].vertexCnt, draws[0].indexCnt,
			vCount, iCount)
	}
	if writer.offset != vCount*stride {
		t.Fatalf("writer wrote %d bytes, want %d", writer.offset, vCount*stride)
	}
	for i, idx := range idxs {
		if int(idx) >= vCount {
			t.Fatalf("index %d value %d out of range (%d verts)", i, idx, vCount)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// aaConvexTessellator

// A filled square: 4 original points at coverage 0.5, an 8-point outer ring at coverage 0, and a 4-point inner ring at
// coverage 1, triangulated as 22 triangles.
func TestAAConvexTessellatorFillSquare(t *testing.T) {
	tess := newAAConvexTessellator(stroke.StyleFill, -1, stroke.JoinBevel, 0)
	identity := geom.IdentityMatrix()
	if !tess.tessellate(&identity, squarePath(10, 10, 50, 50)) {
		t.Fatal("tessellate failed on a square")
	}
	if tess.numPts() != 16 {
		t.Fatalf("point count %d want 16", tess.numPts())
	}
	counts := map[float32]int{}
	for i := 0; i < tess.numPts(); i++ {
		counts[tess.coverage(i)]++
	}
	if counts[0.5] != 4 || counts[0] != 8 || counts[1] != 4 {
		t.Fatalf("coverage classes %v want {0.5:4, 0:8, 1:4}", counts)
	}
	if tess.numIndices() != 66 {
		t.Fatalf("index count %d want 66 (22 triangles)", tess.numIndices())
	}
	for i := 0; i < tess.numIndices(); i++ {
		if tess.index(i) < 0 || tess.index(i) >= tess.numPts() {
			t.Fatalf("index %d value %d out of range", i, tess.index(i))
		}
	}
}

// A stroked square (width 4): the stroke band keeps coverage 1 and both AA fringes drop to 0.
func TestAAConvexTessellatorStrokeSquare(t *testing.T) {
	tess := newAAConvexTessellator(stroke.StyleStroke, 4, stroke.JoinMiter, 4)
	identity := geom.IdentityMatrix()
	if !tess.tessellate(&identity, squarePath(20, 20, 60, 60)) {
		t.Fatal("tessellate failed on a stroked square")
	}
	sawZero, sawOne := false, false
	for i := 0; i < tess.numPts(); i++ {
		c := tess.coverage(i)
		if c == 0 {
			sawZero = true
		}
		if c == 1 {
			sawOne = true
		}
		if c < 0 || c > 1 {
			t.Fatalf("coverage %v out of range", c)
		}
	}
	if !sawZero || !sawOne {
		t.Fatalf("stroke coverages missing extremes (zero %v, one %v)", sawZero, sawOne)
	}
	if tess.numIndices()%3 != 0 || tess.numIndices() == 0 {
		t.Fatalf("index count %d not a positive multiple of 3", tess.numIndices())
	}
}

// Degenerate two-point paths: fills reject them; strokes process them via the fixed-up two-normal lane.
func TestAAConvexTessellatorDegenerateLine(t *testing.T) {
	line := &path.Path{}
	line.MoveTo(10, 10)
	line.LineTo(50, 10)
	identity := geom.IdentityMatrix()

	fill := newAAConvexTessellator(stroke.StyleFill, -1, stroke.JoinBevel, 0)
	if fill.tessellate(&identity, line) {
		t.Fatal("fill tessellate accepted a two-point degenerate path")
	}

	strokeTess := newAAConvexTessellator(stroke.StyleStroke, 4, stroke.JoinBevel, 4)
	if !strokeTess.tessellate(&identity, line) {
		t.Fatal("stroke tessellate rejected a two-point degenerate path")
	}
	if strokeTess.numPts() == 0 || strokeTess.numIndices() == 0 {
		t.Fatal("stroke tessellate produced no geometry for a degenerate line")
	}
}

// Colinear interior points are removed (with the accumulated-error brake).
func TestAAConvexTessellatorColinearRemoval(t *testing.T) {
	p := &path.Path{}
	p.MoveTo(10, 10)
	p.LineTo(30, 10) // colinear midpoint on the top edge
	p.LineTo(50, 10)
	p.LineTo(50, 50)
	p.LineTo(10, 50)
	p.Close()
	tess := newAAConvexTessellator(stroke.StyleFill, -1, stroke.JoinBevel, 0)
	identity := geom.IdentityMatrix()
	if !tess.tessellate(&identity, p) {
		t.Fatal("tessellate failed")
	}
	// The colinear midpoint is removed, so the structure matches the plain square: 16 points.
	if tess.numPts() != 16 {
		t.Fatalf("point count %d want 16 (colinear point kept?)", tess.numPts())
	}
}

//////////////////////////////////////////////////////////////////////////////
// AALinearizing local coordinates

// localCoordsInverse inverts an invertible view matrix and falls back to the identity matrix (not geom.Matrix's
// all-zeros zero value) when the view matrix is singular.
func TestAAFlatteningLocalCoordsInverse(t *testing.T) {
	var scale geom.Matrix
	scale.SetScale(2, 4)
	scaleInv := localCoordsInverse(&scale)
	if got := scaleInv.MapPoint(geom.Point{X: 2, Y: 4}); got != (geom.Point{X: 1, Y: 1}) {
		t.Fatalf("inverse of scale(2,4) maps (2,4) to %v, want (1,1)", got)
	}

	var singular geom.Matrix
	singular.SetScale(0, 0)
	if _, ok := singular.Invert(); ok {
		t.Fatal("zero-scale matrix reported itself invertible")
	}
	fallback := localCoordsInverse(&singular)
	for _, p := range []geom.Point{{X: 3, Y: 7}, {X: -12, Y: 5}, {X: 0.25, Y: -0.5}} {
		if got := fallback.MapPoint(p); got != p {
			t.Fatalf("non-invertible fallback maps %v to %v, want %v (identity)", p, got, p)
		}
	}
}

// With a non-invertible view matrix, the extracted local coordinates track the vertex positions rather than collapsing
// to (0,0).
func TestAAFlatteningExtractVertsSingularViewMatrix(t *testing.T) {
	var singular geom.Matrix
	singular.SetScale(0, 0)
	tess := newAAConvexTessellator(stroke.StyleFill, -1, stroke.JoinBevel, 0)
	identity := geom.IdentityMatrix()
	if !tess.tessellate(&identity, squarePath(10, 10, 50, 50)) {
		t.Fatal("tessellate failed on a square")
	}

	const stride = 24 // pos(8) + color(4) + local coords(8) + coverage(4)
	buf := make([]byte, tess.numPts()*stride)
	idxs := make([]uint16, tess.numIndices())
	ivm := localCoordsInverse(&singular)
	aaFlatteningExtractVerts(tess, &ivm, &quadVertexWriter{buf: buf},
		colorcore.PMColor4f{R: 1, A: 1}, false, 0, idxs)

	readF32 := func(off int) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(buf[off:]))
	}
	nonZero := 0
	for i := 0; i < tess.numPts(); i++ {
		base := i * stride
		pos := geom.Point{X: readF32(base), Y: readF32(base + 4)}
		lc := geom.Point{X: readF32(base + 12), Y: readF32(base + 16)}
		if lc != pos {
			t.Fatalf("vertex %d local coords %v, want the position %v", i, lc, pos)
		}
		if lc != (geom.Point{}) {
			nonZero++
		}
	}
	if nonZero != tess.numPts() {
		t.Fatalf("%d of %d local coords collapsed to (0,0)", tess.numPts()-nonZero,
			tess.numPts())
	}
}

//////////////////////////////////////////////////////////////////////////////
// AAHairline geometry

func TestSplitAndChopConic(t *testing.T) {
	pts := []geom.Point{{X: 0, Y: 0}, {X: 20, Y: 0}, {X: 20, Y: 20}}
	var dst2 [2]geom.Conic
	n := splitConic(pts, &dst2, 0.7)
	if n != 2 {
		t.Fatalf("splitConic count %d want 2", n)
	}
	if dst2[0].Pts[0] != pts[0] || dst2[1].Pts[2] != pts[2] {
		t.Fatalf("split endpoints %v %v want %v %v", dst2[0].Pts[0], dst2[1].Pts[2], pts[0],
			pts[2])
	}
	var dst4 [4]geom.Conic
	cnt := chopConic(pts, &dst4, 0.7)
	if cnt < 1 || cnt > 4 {
		t.Fatalf("chopConic count %d out of range", cnt)
	}
	if dst4[0].Pts[0] != pts[0] || dst4[cnt-1].Pts[2] != pts[2] {
		t.Fatalf("chop chain endpoints wrong")
	}
	for i := 1; i < cnt; i++ {
		if dst4[i].Pts[0] != dst4[i-1].Pts[2] {
			t.Fatalf("chop chain discontinuous at %d", i)
		}
	}
}

func TestNumQuadSubdivs(t *testing.T) {
	flat := []geom.Point{{X: 0, Y: 0}, {X: 10, Y: 5}, {X: 20, Y: 0}}
	if got := numQuadSubdivs(flat); got != 0 {
		t.Fatalf("small quad subdivs %d want 0", got)
	}
	degen := []geom.Point{{X: 0, Y: 0}, {X: 10, Y: 0.01}, {X: 20, Y: 0}}
	if got := numQuadSubdivs(degen); got != -1 {
		t.Fatalf("degenerate quad subdivs %d want -1", got)
	}
	tall := []geom.Point{{X: 0, Y: 0}, {X: 500, Y: 1000}, {X: 1000, Y: 0}}
	if got := numQuadSubdivs(tall); got != 4 {
		t.Fatalf("tall quad subdivs %d want 4 (clamped)", got)
	}
}

func TestAddLineGeometry(t *testing.T) {
	verts := make([]aaHairlineLineVertex, aaHairlineLineSegNumVerts)
	idx := 0
	addLine([]geom.Point{{X: 10, Y: 10}, {X: 20, Y: 10}}, nil, 0xff, verts, &idx)
	if idx != aaHairlineLineSegNumVerts {
		t.Fatalf("vertex cursor %d want %d", idx, aaHairlineLineSegNumVerts)
	}
	want := []struct {
		pos geom.Point
		cov float32
	}{
		{pos: geom.Point{X: 10.5, Y: 10}, cov: 1},
		{pos: geom.Point{X: 19.5, Y: 10}, cov: 1},
		{pos: geom.Point{X: 9.5, Y: 9}, cov: 0},
		{pos: geom.Point{X: 20.5, Y: 9}, cov: 0},
		{pos: geom.Point{X: 9.5, Y: 11}, cov: 0},
		{pos: geom.Point{X: 20.5, Y: 11}, cov: 0},
	}
	for i, w := range want {
		if verts[i].pos != w.pos || verts[i].coverage != w.cov {
			t.Fatalf("vertex %d = %v/%v want %v/%v", i, verts[i].pos, verts[i].coverage,
				w.pos, w.cov)
		}
	}

	// Sub-pixel line: coverage is modulated by length.
	idx = 0
	addLine([]geom.Point{{X: 10, Y: 10}, {X: 10.5, Y: 10}}, nil, 0xff, verts, &idx)
	if math.Abs(float64(verts[0].coverage-0.5)) > 1e-4 {
		t.Fatalf("short line coverage %v want 0.5", verts[0].coverage)
	}

	// Degenerate line: all vertices pushed to ScalarMax.
	idx = 0
	addLine([]geom.Point{{X: 10, Y: 10}, {X: 10, Y: 10}}, nil, 0xff, verts, &idx)
	for i := range verts {
		if verts[i].pos.X != geom.ScalarMax || verts[i].pos.Y != geom.ScalarMax {
			t.Fatalf("degenerate line vertex %d = %v want ScalarMax", i, verts[i].pos)
		}
	}
}

func TestBloatQuad(t *testing.T) {
	qpts := [3]geom.Point{{X: 10, Y: 10}, {X: 20, Y: 0}, {X: 30, Y: 10}}
	verts := make([]aaHairlineBezierVertex, aaHairlineQuadNumVertices)
	if !bloatQuad(&qpts, nil, nil, verts) {
		t.Fatal("bloatQuad failed on a healthy quad")
	}
	// a0/a1 are ±1px orthogonal offsets from a; likewise c0/c1 from c.
	if math.Abs(float64(verts[0].pos.DistanceTo(qpts[0]))-1) > 1e-4 ||
		math.Abs(float64(verts[1].pos.DistanceTo(qpts[0]))-1) > 1e-4 {
		t.Fatalf("a offsets %v %v not 1px from %v", verts[0].pos, verts[1].pos, qpts[0])
	}
	if math.Abs(float64(verts[3].pos.DistanceTo(qpts[2]))-1) > 1e-4 ||
		math.Abs(float64(verts[4].pos.DistanceTo(qpts[2]))-1) > 1e-4 {
		t.Fatalf("c offsets %v %v not 1px from %v", verts[3].pos, verts[4].pos, qpts[2])
	}
	// b0 lies on the line through a0 with normal abN and the line through c0 with normal cbN.
	ab := qpts[1].Sub(qpts[0])
	ab.Normalize()
	abN := ab.MakeOrthog(geom.PointSideLeft)
	if abN.Dot(qpts[2].Sub(qpts[0])) > 0 {
		abN = abN.Negated()
	}
	cb := qpts[1].Sub(qpts[2])
	cb.Normalize()
	cbN := cb.MakeOrthog(geom.PointSideLeft)
	if cbN.Dot(qpts[2].Sub(qpts[0])) < 0 {
		cbN = cbN.Negated()
	}
	if d := abN.Dot(verts[2].pos) - abN.Dot(verts[0].pos); math.Abs(float64(d)) > 1e-3 {
		t.Fatalf("b0 off the a-edge line by %v", d)
	}
	if d := cbN.Dot(verts[2].pos) - cbN.Dot(verts[3].pos); math.Abs(float64(d)) > 1e-3 {
		t.Fatalf("b0 off the c-edge line by %v", d)
	}

	// Fully degenerate quad: rejected.
	pt := [3]geom.Point{{X: 5, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}}
	if bloatQuad(&pt, nil, nil, verts) {
		t.Fatal("bloatQuad accepted a point quad")
	}
}

func TestIntersectLinesParallel(t *testing.T) {
	// Parallel normals: fall back to the midpoint plus the first normal.
	ptA := geom.Point{X: 0, Y: 0}
	ptB := geom.Point{X: 10, Y: 0}
	norm := geom.Point{X: 0, Y: 1}
	got := intersectLines(ptA, norm, ptB, norm)
	want := geom.Point{X: 5, Y: 1}
	if got != want {
		t.Fatalf("parallel intersect %v want %v", got, want)
	}
}

func TestGatherLinesAndQuads(t *testing.T) {
	identity := geom.IdentityMatrix()
	clip := geom.IRect{Left: 0, Top: 0, Right: 100, Bottom: 100}

	gather := func(p *path.Path, m *geom.Matrix, capLength float32) (lines, quads,
		conics []geom.Point, subdivs []int, weights []float32, quadCount int,
	) {
		quadCount = gatherLinesAndQuads(p, m, clip, capLength, false, &lines, &quads, &conics,
			&subdivs, &weights)
		return lines, quads, conics, subdivs, weights, quadCount
	}

	// A plain line lands in device space.
	line := &path.Path{}
	line.MoveTo(10, 10)
	line.LineTo(90, 90)
	var m geom.Matrix
	m.SetTranslate(5, 0)
	lines, quads, conics, _, _, quadCount := gather(line, &m, 0)
	if len(lines) != 2 || len(quads) != 0 || len(conics) != 0 || quadCount != 0 {
		t.Fatalf("line gather: %d lines %d quads %d conics", len(lines), len(quads),
			len(conics))
	}
	if lines[0] != (geom.Point{X: 15, Y: 10}) || lines[1] != (geom.Point{X: 95, Y: 90}) {
		t.Fatalf("line device points %v", lines)
	}

	// A closed contour materializes the closing line.
	closed := &path.Path{}
	closed.MoveTo(10, 10)
	closed.LineTo(90, 10)
	closed.Close()
	lines, _, _, _, _, _ = gather(closed, &identity, 0)
	if len(lines) != 4 {
		t.Fatalf("closed contour line count %d want 4", len(lines))
	}
	if lines[2] != (geom.Point{X: 90, Y: 10}) || lines[3] != (geom.Point{X: 10, Y: 10}) {
		t.Fatalf("closing line %v", lines[2:4])
	}

	// A gently curved quad stays quads with subdiv 0 (the symmetric quad chops at its max-curvature midpoint first,
	// yielding two).
	quad := &path.Path{}
	quad.MoveTo(10, 50)
	quad.QuadTo(50, 30, 90, 50)
	lines, quads, _, subdivs, _, quadCount := gather(quad, &identity, 0)
	if len(quads) != 6 || quadCount != 2 || len(subdivs) != 2 || subdivs[0] != 0 ||
		subdivs[1] != 0 {
		t.Fatalf("quad gather: %d quad pts, count %d, subdivs %v", len(quads), quadCount,
			subdivs)
	}
	if len(lines) != 0 {
		t.Fatalf("quad gather produced %d line points", len(lines))
	}

	// A degenerate quad decomposes into lines (after the max-curvature chop).
	degen := &path.Path{}
	degen.MoveTo(10, 50)
	degen.QuadTo(50, 50.01, 90, 50)
	lines, quads, _, _, _, _ = gather(degen, &identity, 0)
	if len(quads) != 0 || len(lines) == 0 || len(lines)%4 != 0 {
		t.Fatalf("degenerate quad gather: %d quad pts, %d line pts", len(quads), len(lines))
	}

	// Conics survive as conics when the context has 32-bit floats.
	conicPath := &path.Path{}
	conicPath.MoveTo(10, 50)
	conicPath.ConicTo(50, 10, 90, 50, 0.7)
	_, _, conics, _, weights, _ := gather(conicPath, &identity, 0)
	if len(conics) == 0 || len(conics)%3 != 0 || len(weights) != len(conics)/3 {
		t.Fatalf("conic gather: %d conic pts, %d weights", len(conics), len(weights))
	}

	// Everything outside the clip bounds is culled.
	far := &path.Path{}
	far.MoveTo(1000, 1000)
	far.LineTo(1100, 1100)
	lines, quads, conics, _, _, _ = gather(far, &identity, 0)
	if len(lines) != 0 || len(quads) != 0 || len(conics) != 0 {
		t.Fatal("clip-rejected geometry leaked through")
	}

	// A dangling zero-length verb with caps gets an x-aligned cap line.
	dot := &path.Path{}
	dot.MoveTo(50, 50)
	dot.LineTo(50, 50)
	lines, _, _, _, _, _ = gather(dot, &identity, 2)
	if len(lines) != 4 {
		t.Fatalf("dot-with-caps line count %d want 4", len(lines))
	}
	capA := lines[2]
	capB := lines[3]
	if capA != (geom.Point{X: 48, Y: 50}) || capB != (geom.Point{X: 52, Y: 50}) {
		t.Fatalf("cap line %v-%v want (48,50)-(52,50)", capA, capB)
	}

	// A (moveTo, close) contour with caps also produces a cap line.
	mc := &path.Path{}
	mc.MoveTo(30, 30)
	mc.Close()
	lines, _, _, _, _, _ = gather(mc, &identity, 2)
	if len(lines) != 2 {
		t.Fatalf("(moveTo, close) cap line count %d want 2", len(lines))
	}
	if lines[0] != (geom.Point{X: 28, Y: 30}) || lines[1] != (geom.Point{X: 32, Y: 30}) {
		t.Fatalf("cap line %v", lines[:2])
	}
}

//////////////////////////////////////////////////////////////////////////////
// Renderer selection gates and chain order

func aaTestCanDrawArgs(sdc *SurfaceDrawContext, shape *StyledShape, viewMatrix *geom.Matrix, aaType gpu.AAType) *CanDrawPathArgs {
	return &CanDrawPathArgs{
		Caps:                   sdc.Caps(),
		Proxy:                  sdc.AsRenderTargetProxy(),
		ClipConservativeBounds: geom.IRectSize(sdc.Dimensions()),
		ViewMatrix:             viewMatrix,
		Shape:                  shape,
		AAType:                 aaType,
	}
}

func TestAAConvexCanDrawPath(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()
	r := NewAAConvexPathRenderer()

	convex := MakeStyledShapePath(convexPentagonPath(), SimpleFillStyle(), DoSimplifyYes)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &convex, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathYes {
		t.Fatalf("convex coverage fill: %d want yes", got)
	}
	// Non-coverage AA: no.
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &convex, &identity,
		gpu.AATypeNone)); got != CanDrawPathNo {
		t.Fatalf("convex non-AA: %d want no", got)
	}
	// Concave: no.
	concave := &path.Path{}
	concave.MoveTo(0, 0)
	concave.LineTo(50, 20)
	concave.LineTo(100, 0)
	concave.LineTo(50, 100)
	concave.Close()
	concaveShape := MakeStyledShapePath(concave, SimpleFillStyle(), DoSimplifyYes)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &concaveShape, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("concave coverage fill: %d want no", got)
	}
	// Inverse fill: no.
	inv := convexPentagonPath()
	inv.SetFillType(path.FillInverseWinding)
	invShape := MakeStyledShapePath(inv, SimpleFillStyle(), DoSimplifyNo)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &invShape, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("inverse coverage fill: %d want no", got)
	}
	// Hairline style: no (not a simple fill).
	hair := MakeStyledShapePath(convexPentagonPath(), SimpleHairlineStyle(), DoSimplifyYes)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &hair, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("hairline style: %d want no", got)
	}
}

func TestAAHairlineCanDrawPath(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()
	r := NewAAHairLinePathRenderer()

	hair := MakeStyledShapePath(convexPentagonPath(), SimpleHairlineStyle(), DoSimplifyYes)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &hair, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathYes {
		t.Fatalf("hairline coverage: %d want yes", got)
	}
	// Non-coverage: no.
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &hair, &identity,
		gpu.AATypeNone)); got != CanDrawPathNo {
		t.Fatalf("hairline non-AA: %d want no", got)
	}
	// A fill style is not hairline-or-equivalent: no.
	fill := MakeStyledShapePath(convexPentagonPath(), SimpleFillStyle(), DoSimplifyYes)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &fill, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("fill style: %d want no", got)
	}
	// A stroke whose device width is sub-pixel is hairline-equivalent: yes.
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(0.4, false)
	thin := MakeStyledShapePath(convexPentagonPath(), MakeStyle(rec), DoSimplifyNo)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &thin, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathYes {
		t.Fatalf("thin stroke: %d want yes", got)
	}
}

func TestAALinearizingCanDrawPath(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()
	r := NewAALinearizingConvexPathRenderer()

	// Convex fill: yes.
	fill := MakeStyledShapePath(convexPentagonPath(), SimpleFillStyle(), DoSimplifyYes)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &fill, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathYes {
		t.Fatalf("convex fill: %d want yes", got)
	}

	strokeShape := func(width float32, join stroke.Join) StyledShape {
		rec := stroke.NewStrokeRec(stroke.InitStyleFill)
		rec.SetStrokeStyle(width, false)
		rec.SetStrokeParams(stroke.CapButt, join, 4)
		return MakeStyledShapePath(convexPentagonPath(), MakeStyle(rec), DoSimplifyNo)
	}

	// Miter-joined stroke in range: yes.
	miter := strokeShape(4, stroke.JoinMiter)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &miter, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathYes {
		t.Fatalf("miter stroke: %d want yes", got)
	}
	// Round join: no.
	round := strokeShape(4, stroke.JoinRound)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &round, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("round join: %d want no", got)
	}
	// Too-wide stroke on a non-rect: no.
	wide := strokeShape(25, stroke.JoinBevel)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &wide, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("wide stroke: %d want no", got)
	}
	// Sub-pixel stroke width: no (AAHairline territory).
	thin := strokeShape(0.4, stroke.JoinBevel)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &thin, &identity,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("thin stroke: %d want no", got)
	}
	// Non-similarity matrix for strokes: no.
	var skew geom.Matrix
	skew.SetSkew(0.5, 0)
	bevel := strokeShape(4, stroke.JoinBevel)
	if got := r.OnCanDrawPath(aaTestCanDrawArgs(sdc, &bevel, &skew,
		gpu.AATypeCoverage)); got != CanDrawPathNo {
		t.Fatalf("skewed stroke: %d want no", got)
	}
}

// The chain places AAConvex ahead of AAHairline ahead of AALinearizing ahead of Triangulating, and the option bits
// remove members individually.
func TestPathRendererChainOrderWithAARenderers(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	sel := func(opts gpu.PathRenderers, shape *StyledShape) string {
		chain := NewPathRendererChain(dc, PathRendererChainOptions{PathRenderers: opts})
		pr := chain.GetPathRenderer(aaTestCanDrawArgs(sdc, shape, &identity,
			gpu.AATypeCoverage), ChainDrawTypeColor, nil)
		if pr == nil {
			return "<none>"
		}
		return pr.Name()
	}

	fill := MakeStyledShapePath(convexPentagonPath(), SimpleFillStyle(), DoSimplifyYes)
	if got := sel(gpu.PathRenderersDefault, &fill); got != "AAConvex" {
		t.Fatalf("convex coverage fill selects %q want AAConvex", got)
	}
	noConvex := gpu.PathRenderersDefault &^ gpu.PathRenderersAAConvex
	fill2 := MakeStyledShapePath(convexPentagonPath(), SimpleFillStyle(), DoSimplifyYes)
	if got := sel(noConvex, &fill2); got != "AALinear" {
		t.Fatalf("convex coverage fill without AAConvex selects %q want AALinear", got)
	}
	// With both AA renderers disabled nothing in the chain takes a convex coverage-AA fill: Triangulating leaves convex
	// paths "to simpler algorithms" and Tessellation/Default reject coverage AA, so it falls to the SW mask fallback —
	// exactly the pre-part-5 behavior.
	neither := noConvex &^ gpu.PathRenderersAALinearizing
	fill3 := MakeStyledShapePath(convexPentagonPath(), SimpleFillStyle(), DoSimplifyYes)
	if got := sel(neither, &fill3); got != "<none>" {
		t.Fatalf("convex coverage fill without AA renderers selects %q want <none>", got)
	}

	hair := MakeStyledShapePath(convexPentagonPath(), SimpleHairlineStyle(), DoSimplifyYes)
	if got := sel(gpu.PathRenderersDefault, &hair); got != "AAHairline" {
		t.Fatalf("hairline coverage selects %q want AAHairline", got)
	}

	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)
	rec.SetStrokeParams(stroke.CapButt, stroke.JoinMiter, 4)
	stroked := MakeStyledShapePath(convexPentagonPath(), MakeStyle(rec), DoSimplifyNo)
	if got := sel(gpu.PathRenderersDefault, &stroked); got != "AALinear" {
		t.Fatalf("convex coverage stroke selects %q want AALinear", got)
	}
}

//////////////////////////////////////////////////////////////////////////////
// Record-time op merging on the fake driver

func TestAAConvexPathOpMerge(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	for i := range 2 {
		shape := MakeStyledShapePath(convexPentagonPath(), SimpleFillStyle(), DoSimplifyYes)
		sdc.DrawShape(nil, solidPaint(float32(i)*0.5, 0, 0, 0.5), gpu.AAYes, &identity,
			&shape)
	}
	if n := sdc.GetOpsTask().NumOpChains(); n != 1 {
		t.Fatalf("chain count %d want 1 (merge failed)", n)
	}
	op, ok := sdc.GetOpsTask().GetChainHead(0).(*aaConvexPathOp)
	if !ok {
		t.Fatalf("chain head is %T, want *aaConvexPathOp", sdc.GetOpsTask().GetChainHead(0))
	}
	if len(op.paths) != 2 {
		t.Fatalf("merged path count %d want 2", len(op.paths))
	}
}

func TestAAHairlineOpMergeRules(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	drawHair := func(r, g, b, a float32) {
		shape := MakeStyledShapePath(convexPentagonPath(), SimpleHairlineStyle(),
			DoSimplifyYes)
		sdc.DrawShape(nil, solidPaint(r, g, b, a), gpu.AAYes, &identity, &shape)
	}

	// Same color: merged.
	drawHair(0.5, 0, 0, 0.5)
	drawHair(0.5, 0, 0, 0.5)
	if n := sdc.GetOpsTask().NumOpChains(); n != 1 {
		t.Fatalf("chain count %d want 1 (same-color hairlines should merge)", n)
	}
	op, ok := sdc.GetOpsTask().GetChainHead(0).(*aaHairlineOp)
	if !ok {
		t.Fatalf("chain head is %T, want *aaHairlineOp", sdc.GetOpsTask().GetChainHead(0))
	}
	if len(op.paths) != 2 {
		t.Fatalf("merged path count %d want 2", len(op.paths))
	}

	// Different color: hairline ops carry a uniform color, so they cannot merge.
	drawHair(0, 0.5, 0, 0.5)
	if n := sdc.GetOpsTask().NumOpChains(); n != 2 {
		t.Fatalf("chain count %d want 2 (different-color hairlines must not merge)", n)
	}
}

func TestAAFlatteningConvexPathOpMerge(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()
	identity := geom.IdentityMatrix()

	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	rec.SetStrokeStyle(4, false)
	rec.SetStrokeParams(stroke.CapButt, stroke.JoinBevel, 4)
	for i := range 2 {
		shape := MakeStyledShapePath(convexPentagonPath(), MakeStyle(rec), DoSimplifyNo)
		sdc.DrawShape(nil, solidPaint(float32(i)*0.5, 0, 0, 0.5), gpu.AAYes, &identity,
			&shape)
	}
	if n := sdc.GetOpsTask().NumOpChains(); n != 1 {
		t.Fatalf("chain count %d want 1 (merge failed)", n)
	}
	op, ok := sdc.GetOpsTask().GetChainHead(0).(*aaFlatteningConvexPathOp)
	if !ok {
		t.Fatalf("chain head is %T, want *aaFlatteningConvexPathOp",
			sdc.GetOpsTask().GetChainHead(0))
	}
	if len(op.paths) != 2 {
		t.Fatalf("merged path count %d want 2", len(op.paths))
	}
}
